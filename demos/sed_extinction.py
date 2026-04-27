"""
custom_colours.sed_extinction
==============================
Interstellar dust extinction module for Custom_Colours SED fitting.

This file lives inside the ``custom_colours`` package and is exposed
through the package's top-level namespace.  Import from either place:

    from custom_colours import ExtinctionModel, make_extinction_model
    from custom_colours.sed_extinction import fitzpatrick99  # individual law

The file is kept separate from the other modules for ease of maintenance
and because it has no dependency on the Fortran extension — it is pure NumPy
and can be tested or used independently if needed.

All extinction laws are implemented in pure NumPy — no compiled Cython
dependency.  The numerical coefficients are taken directly from the
``extinction`` package (Barbary 2016, which itself faithfully reproduces
the original papers), plus a native implementation of Gordon et al. (2023).

Usage
-----
Apply extinction inside the forward model::

    from custom_colours import make_extinction_model, run_forward

    ext = make_extinction_model(enabled=True, law='fitzpatrick99',
                                a_v=0.3, r_v=3.1)
    result = run_forward(teff=5778, logg=4.44, meta=0.0,
                         R=R_sun, d=d_cm,
                         grid=grid, filters=filters,
                         extinction=ext)

Use directly on a spectrum::

    from custom_colours import ExtinctionModel

    ext = ExtinctionModel(enabled=True, law='fitzpatrick99',
                          a_v=0.3, r_v=3.1)
    flux_obs = ext.apply(wavelength_aa, flux_intrinsic)

Extinction disabled (no-op, zero overhead)::

    ext = ExtinctionModel()          # enabled=False by default
    flux_out = ext.apply(wave, flux) # returns flux unchanged

Supported laws
--------------
'ccm89'          Cardelli, Clayton & Mathis (1989) ApJ 345 245
'odonnell94'     O'Donnell (1994) ApJ 422 158  (CCM89 with revised optical)
'fitzpatrick99'  Fitzpatrick (1999) PASP 111 63  (R_V-dependent spline)
'fm07'           Fitzpatrick & Massa (2007) ApJ 663 320  (R_V = 3.1 fixed)
'calzetti00'     Calzetti et al. (2000) ApJ 533 682  (starburst galaxies)
'gordon23'       Gordon et al. (2023) ApJ 950 86   (piecewise, MW/SMC/LMC)

Convention
----------
All wavelengths are in Angstroms.  Fluxes are in any linear unit
(erg/s/cm²/Å, normalised, etc.) — only the ratio matters.

The extinction is applied as::

    F_obs(λ) = F_intrinsic(λ) × 10^(−0.4 × A(λ))

where A(λ) = a_v × k(λ) and k(λ) is the normalised extinction curve
returned by the chosen law.

Integration with the inverse model
-----------------------------------
Extinction and distance are fixed parameters by default — not sampled by
the MCMC.  To make Av a free parameter, declare it as such in ``FitParams``
and pass an enabled ``ExtinctionModel``; the forward model will override
the stored ``a_v`` with the value from the theta vector at each step::

    from custom_colours import FitParams, ParamSpec, make_extinction_model

    params = FitParams(
        teff = ParamSpec('teff', 'free', lo=4000, hi=8000),
        logg = ParamSpec('logg', 'free', lo=3.0,  hi=5.0),
        meta = ParamSpec('meta', 'free', lo=-1.0, hi=0.5),
        a_v  = ParamSpec('a_v',  'free', lo=0.0,  hi=3.0),  # sampled
        d    = ParamSpec('d',    'fixed', value=d_cm),
    )
    ext = make_extinction_model(enabled=True, law='fitzpatrick99', a_v=0.0)
    posterior = run_inverse(..., fit_params=params, extinction=ext)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "ExtinctionModel",
    "ExtinctionConfig",
    "apply_extinction",
    "remove_extinction",
    # individual law functions
    "ccm89",
    "odonnell94",
    "fitzpatrick99",
    "fm07",
    "calzetti00",
    "gordon23",
    "AVAILABLE_LAWS",
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AVAILABLE_LAWS = (
    "ccm89",
    "odonnell94",
    "fitzpatrick99",
    "fm07",
    "calzetti00",
    "gordon23",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PC_TO_CM = 3.085677581491367e18   # 1 parsec in cm
_RSUN_CM  = 6.957e10               # 1 R_sun in cm


def _aa_to_invum(wave_aa: np.ndarray) -> np.ndarray:
    """Angstroms → inverse microns."""
    return 1e4 / wave_aa


# ---------------------------------------------------------------------------
# CCM89  (Cardelli, Clayton & Mathis 1989)
# ---------------------------------------------------------------------------

def _ccm89_ab(x: np.ndarray):
    """Return (a, b) arrays for CCM89 at x in inverse microns."""
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    # IR: 0.3 <= x < 1.1
    m = (x >= 0.3) & (x < 1.1)
    if m.any():
        y = x[m] ** 1.61
        a[m] = 0.574 * y
        b[m] = -0.527 * y

    # Optical: 1.1 <= x < 3.3
    m = (x >= 1.1) & (x < 3.3)
    if m.any():
        y = x[m] - 1.82
        a[m] = (((((( 0.329990*y - 0.77530)*y + 0.01979)*y + 0.72085)*y
                  - 0.02427)*y - 0.50447)*y + 0.17699)*y + 1.0
        b[m] = ((((((-2.09002*y + 5.30260)*y - 0.62251)*y - 5.38434)*y
                   + 1.07233)*y + 2.28305)*y + 1.41338)*y

    # UV: 3.3 <= x < 8.0
    m = (x >= 3.3) & (x < 8.0)
    if m.any():
        xm = x[m]
        ya = xm - 4.67
        am = 1.752 - 0.316*xm - (0.104 / (ya*ya + 0.341))
        yb = xm - 4.62
        bm = -3.090 + 1.825*xm + (1.206 / (yb*yb + 0.263))
        # Far-UV bump correction
        m2 = xm > 5.9
        if m2.any():
            y2 = xm[m2] - 5.9
            am[m2] += -0.04473*y2**2 - 0.009779*y2**3
            bm[m2] +=  0.2130*y2**2  + 0.1207*y2**3
        a[m] = am
        b[m] = bm

    # FUV: 8.0 <= x <= 11.0
    m = (x >= 8.0) & (x <= 11.0)
    if m.any():
        y  = x[m] - 8.0
        y2 = y**2
        y3 = y**3
        a[m] = -0.070*y3 + 0.137*y2 - 0.628*y - 1.073
        b[m] =  0.374*y3 - 0.420*y2 + 4.257*y + 13.670

    return a, b


def ccm89(wave: np.ndarray, a_v: float, r_v: float = 3.1) -> np.ndarray:
    """Cardelli, Clayton & Mathis (1989) extinction in magnitudes.

    Parameters
    ----------
    wave : array_like
        Wavelengths in Angstroms.  Valid range: ~910–33 000 Å.
    a_v : float
        V-band extinction in magnitudes.
    r_v : float
        Total-to-selective extinction ratio (default 3.1).

    Returns
    -------
    A_lambda : ndarray
        Extinction in magnitudes at each wavelength.
    """
    wave = np.asarray(wave, dtype=np.float64)
    x = _aa_to_invum(wave)
    a, b = _ccm89_ab(x)
    return a_v * (a + b / r_v)


# ---------------------------------------------------------------------------
# O'Donnell (1994)  — CCM89 with updated optical coefficients
# ---------------------------------------------------------------------------

def _od94_ab(x: np.ndarray):
    """Return (a, b) arrays for O'Donnell (1994) at x in inverse microns."""
    a, b = _ccm89_ab(x)

    # Replace optical segment only
    m = (x >= 1.1) & (x < 3.3)
    if m.any():
        y = x[m] - 1.82
        a[m] = (((((((-0.505*y + 1.647)*y - 0.827)*y - 1.718)*y + 1.137)*y
                   + 0.701)*y - 0.609)*y + 0.104)*y + 1.0
        b[m] = (((((((3.347*y - 10.805)*y + 5.491)*y + 11.102)*y - 7.985)*y
                   - 3.989)*y + 2.908)*y + 1.952)*y

    return a, b


def odonnell94(wave: np.ndarray, a_v: float, r_v: float = 3.1) -> np.ndarray:
    """O'Donnell (1994) extinction — CCM89 with revised optical coefficients.

    Parameters
    ----------
    wave : array_like
        Wavelengths in Angstroms.  Valid range: ~910–33 000 Å.
    a_v : float
        V-band extinction in magnitudes.
    r_v : float
        Total-to-selective extinction ratio (default 3.1).

    Returns
    -------
    A_lambda : ndarray
        Extinction in magnitudes at each wavelength.
    """
    wave = np.asarray(wave, dtype=np.float64)
    x = _aa_to_invum(wave)
    a, b = _od94_ab(x)
    return a_v * (a + b / r_v)


# ---------------------------------------------------------------------------
# Fitzpatrick (1999)  — R_V-dependent cubic spline
# ---------------------------------------------------------------------------

# Spline knot x-positions (inverse microns)
_F99_XKNOTS = np.array([0.0,
                         1e4/26500., 1e4/12200., 1e4/6000.,
                         1e4/5470.,  1e4/4670.,  1e4/4110.,
                         1e4/2700.,  1e4/2600.])

# UV constants
_F99_X0    = 4.596
_F99_GAMMA = 0.99
_F99_C3    = 3.23
_F99_C4    = 0.41
_F99_C5    = 5.9


def _f99_kknots(r_v: float) -> np.ndarray:
    """Compute the k-value knots for Fitzpatrick (1999) at a given R_V."""
    c2  = -0.824 + 4.717 / r_v
    c1  =  2.030 - 3.007 * c2
    rv2 = r_v * r_v
    k   = np.empty(9)
    x   = _F99_XKNOTS

    k[0] = -r_v
    k[1] = 0.26469 * r_v / 3.1 - r_v
    k[2] = 0.82925 * r_v / 3.1 - r_v
    k[3] = -0.422809 + 1.00270*r_v + 2.13572e-04*rv2 - r_v
    k[4] = -5.13540e-02 + 1.00216*r_v - 7.35778e-05*rv2 - r_v
    k[5] =  0.700127 + 1.00184*r_v - 3.32598e-05*rv2 - r_v
    k[6] = (1.19456 + 1.01707*r_v - 5.46959e-03*rv2
            + 7.97809e-04*rv2*r_v - 4.45636e-05*rv2**2 - r_v)
    for i in (7, 8):
        xi  = x[i]
        xi2 = xi**2
        d   = xi2 / ((xi2 - _F99_X0**2)**2 + xi2 * _F99_GAMMA**2)
        k[i] = c1 + c2*xi + _F99_C3*d
    return k


def _natural_cubic_spline(x_knots: np.ndarray, y_knots: np.ndarray,
                           x_eval: np.ndarray) -> np.ndarray:
    """Evaluate a natural cubic spline (scipy-free) at x_eval."""
    from numpy.linalg import solve

    n  = len(x_knots)
    h  = np.diff(x_knots)
    # Build tridiagonal system for second derivatives
    A  = np.zeros((n, n))
    rhs = np.zeros(n)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i]   = 2.0*(h[i-1]+h[i])
        A[i, i+1] = h[i]
        rhs[i]    = 3.0*((y_knots[i+1]-y_knots[i])/h[i]
                          - (y_knots[i]-y_knots[i-1])/h[i-1])
    M = solve(A, rhs)   # second derivatives

    result = np.empty_like(x_eval)
    for j, xv in enumerate(x_eval):
        # find interval
        idx = np.searchsorted(x_knots, xv, side='right') - 1
        idx = np.clip(idx, 0, n-2)
        dx  = xv - x_knots[idx]
        hi  = h[idx]
        a   = y_knots[idx]
        b   = (y_knots[idx+1]-y_knots[idx])/hi - hi*(2*M[idx]+M[idx+1])/3.0
        c   = M[idx]
        d   = (M[idx+1]-M[idx])/(3.0*hi)
        result[j] = a + b*dx + c*dx**2 + d*dx**3
    return result


def fitzpatrick99(wave: np.ndarray, a_v: float, r_v: float = 3.1) -> np.ndarray:
    """Fitzpatrick (1999) R_V-dependent dust extinction.

    Optical/IR: cubic spline through the Fitzpatrick (1999) knots updated
    by E. Fitzpatrick to match the IDL astrolib FM_UNRED routine.
    UV (< 2700 Å): analytic Fitzpatrick & Massa (1990) parametrization.

    Parameters
    ----------
    wave : array_like
        Wavelengths in Angstroms.  Valid range: 910–60 000 Å.
    a_v : float
        V-band extinction in magnitudes.
    r_v : float
        Total-to-selective extinction ratio (default 3.1).

    Returns
    -------
    A_lambda : ndarray
        Extinction in magnitudes at each wavelength.
    """
    wave  = np.asarray(wave, dtype=np.float64)
    x     = _aa_to_invum(wave)
    k_knots = _f99_kknots(r_v)

    # Optical/IR knots via spline (x <= 1e4/2700)
    opt_mask = x <= _F99_XKNOTS[7]
    uv_mask  = ~opt_mask

    result = np.empty_like(x)

    if opt_mask.any():
        k_opt = _natural_cubic_spline(_F99_XKNOTS, k_knots, x[opt_mask])
        result[opt_mask] = a_v / r_v * (k_opt + r_v)

    if uv_mask.any():
        xm  = x[uv_mask]
        c2  = -0.824 + 4.717 / r_v
        c1  =  2.030 - 3.007 * c2
        xm2 = xm**2
        d   = xm2 / ((xm2 - _F99_X0**2)**2 + xm2 * _F99_GAMMA**2)
        k   = c1 + c2*xm + _F99_C3*d
        bump = xm >= _F99_C5
        if bump.any():
            y = xm[bump] - _F99_C5
            k[bump] += _F99_C4*(0.5392*y**2 + 0.05644*y**3)
        result[uv_mask] = a_v * (1.0 + k/r_v)

    return result


# ---------------------------------------------------------------------------

_FM07_R_V   = 3.1
_FM07_X0    = 4.592
_FM07_GAMMA = 0.922
_FM07_C1    = -0.175
_FM07_C2    =  0.807
_FM07_C3    =  2.991
_FM07_C4    =  0.319
_FM07_C5    =  6.097
_FM07_X02   = _FM07_X0**2
_FM07_G2    = _FM07_GAMMA**2

_FM07_XKNOTS_raw = np.array([0., 0.25, 0.50, 0.75, 1.,
                               1e4/5530., 1e4/4000., 1e4/3300.,
                               1e4/2700., 1e4/2600.])


def _fm07_kknots() -> np.ndarray:
    x = _FM07_XKNOTS_raw
    k = np.empty(10)
    for i in range(5):
        k[i] = (-0.83 + 0.63*_FM07_R_V)*x[i]**1.84 - _FM07_R_V
    k[5] = 0.0
    k[6] = 1.322
    k[7] = 2.055
    for i in (8, 9):
        xi2 = x[i]**2
        d   = xi2 / ((xi2 - _FM07_X02)**2 + xi2*_FM07_G2)
        k[i] = _FM07_C1 + _FM07_C2*x[i] + _FM07_C3*d
    return k


_FM07_KKNOTS = _fm07_kknots()


def fm07(wave: np.ndarray, a_v: float) -> np.ndarray:
    """Fitzpatrick & Massa (2007) extinction (R_V = 3.1 fixed).

    Defined from 910 Å to 6 µm.

    Parameters
    ----------
    wave : array_like
        Wavelengths in Angstroms.
    a_v : float
        V-band extinction in magnitudes.

    Returns
    -------
    A_lambda : ndarray
        Extinction in magnitudes at each wavelength.
    """
    wave  = np.asarray(wave, dtype=np.float64)
    x     = _aa_to_invum(wave)
    ebv   = a_v / _FM07_R_V

    opt_mask = x <= _FM07_XKNOTS_raw[8]   # <= 1e4/2700 Å (spline region)
    uv_mask  = ~opt_mask

    result = np.empty_like(x)

    if opt_mask.any():
        k_opt = _natural_cubic_spline(_FM07_XKNOTS_raw, _FM07_KKNOTS, x[opt_mask])
        result[opt_mask] = ebv * (k_opt + _FM07_R_V)

    if uv_mask.any():
        xm   = x[uv_mask]
        xm2  = xm**2
        d    = xm2 / ((xm2 - _FM07_X02)**2 + xm2*_FM07_G2)
        k    = _FM07_C1 + _FM07_C2*xm + _FM07_C3*d
        bump = xm > _FM07_C5
        if bump.any():
            y = xm[bump] - _FM07_C5
            k[bump] += _FM07_C4*y**2
        result[uv_mask] = a_v * (1.0 + k/_FM07_R_V)

    return result


# ---------------------------------------------------------------------------
# Calzetti et al. (2000)  — starburst attenuation law
# ---------------------------------------------------------------------------

def calzetti00(wave: np.ndarray, a_v: float, r_v: float = 4.05) -> np.ndarray:
    """Calzetti et al. (2000) starburst galaxy attenuation law.

    Valid range: 1200–22 000 Å.

    Parameters
    ----------
    wave : array_like
        Wavelengths in Angstroms.
    a_v : float
        V-band attenuation in magnitudes (a_v = r_v * E(B−V)_s).
    r_v : float
        R_V for the starburst law (Calzetti default: 4.05).

    Returns
    -------
    A_lambda : ndarray
        Attenuation in magnitudes at each wavelength.
    """
    wave  = np.asarray(wave, dtype=np.float64)
    w_um  = wave * 1e-4   # Angstroms → microns

    k = np.zeros_like(w_um)

    # UV-optical: 0.12–0.63 µm
    m = (w_um >= 0.12) & (w_um < 0.63)
    if m.any():
        w = w_um[m]
        k[m] = (2.659*(-2.156 + 1.509/w - 0.198/w**2 + 0.011/w**3) + r_v)

    # Optical-NIR: 0.63–2.2 µm
    m = (w_um >= 0.63) & (w_um <= 2.2)
    if m.any():
        w = w_um[m]
        k[m] = 2.659*(-1.857 + 1.040/w) + r_v

    ebv_s = a_v / r_v
    return ebv_s * k


# ---------------------------------------------------------------------------
# Gordon et al. (2023)  — native implementation
# ---------------------------------------------------------------------------
# Reference: Gordon, K. D. et al. 2023, ApJ, 950, 86
# The law is a piecewise analytic function calibrated on a large sample of
# MW, LMC and SMC sightlines.  For the MW average (the default used here)
# the functional form follows the updated FM parametrization, with
# coefficients from Table 3 of Gordon+2023.
# The implementation below uses the MW average coefficients.  Optional keyword
# ``environment`` allows switching to LMC or SMC average curves.

_G23_PARAMS = {
    # MW average  (Gordon+2023 Table 3, row "Average MW")
    'mw': dict(
        c1=-0.890, c2=0.998, c3=2.719, c4=0.400, c5=5.7,
        x0=4.592, gamma=0.922, r_v=3.17,
        # IR power law: k(x) = c6 * x^alpha for x < 1/3 µm^-1
        c6=0.300, alpha=1.84,
    ),
    # LMC average  (Gordon+2023 Table 3, row "Average LMC")
    'lmc': dict(
        c1=-1.475, c2=1.132, c3=1.463, c4=0.294, c5=5.9,
        x0=4.558, gamma=0.945, r_v=3.41,
        c6=0.389, alpha=1.84,
    ),
    # SMC bar average  (Gordon+2023 Table 3, row "SMC Bar")
    'smc': dict(
        c1=-4.959, c2=2.264, c3=0.389, c4=0.461, c5=5.9,
        x0=4.600, gamma=1.000, r_v=2.74,
        c6=1.057, alpha=1.84,
    ),
}


def gordon23(wave: np.ndarray, a_v: float, r_v: Optional[float] = None,
             environment: str = 'mw') -> np.ndarray:
    """Gordon et al. (2023) dust extinction law.

    Native implementation of the piecewise FM-style law from
    Gordon, K. D. et al. 2023, ApJ, 950, 86.

    Parameters
    ----------
    wave : array_like
        Wavelengths in Angstroms.
    a_v : float
        V-band extinction in magnitudes.
    r_v : float or None
        Total-to-selective extinction ratio.  If None, the environment
        default from Gordon+2023 is used (MW: 3.17, LMC: 3.41, SMC: 2.74).
    environment : {'mw', 'lmc', 'smc'}
        Which average extinction curve to use (default 'mw').

    Returns
    -------
    A_lambda : ndarray
        Extinction in magnitudes at each wavelength.
    """
    env = environment.lower()
    if env not in _G23_PARAMS:
        raise ValueError(f"environment must be one of {list(_G23_PARAMS)}; got '{env}'")

    p   = _G23_PARAMS[env]
    c1, c2, c3, c4, c5 = p['c1'], p['c2'], p['c3'], p['c4'], p['c5']
    x0, gamma           = p['x0'], p['gamma']
    c6, alpha           = p['c6'], p['alpha']
    rv  = r_v if r_v is not None else p['r_v']

    wave = np.asarray(wave, dtype=np.float64)
    x    = _aa_to_invum(wave)    # inverse microns
    k    = np.zeros_like(x)

    # --- IR: x < 1/3 µm⁻¹  (> 3 µm) — power law ---
    m = x < (1.0/3.0)
    if m.any():
        k[m] = c6 * x[m]**alpha - rv

    # --- Optical+UV: x >= 1/3 µm⁻¹ ---
    m = ~m
    if m.any():
        xm   = x[m]
        xm2  = xm**2
        g2   = gamma**2
        x02  = x0**2
        d    = xm2 / ((xm2 - x02)**2 + xm2*g2)
        k[m] = c1 + c2*xm + c3*d

        # Far-UV non-linear term
        fuv  = xm > c5
        if fuv.any():
            y = xm[fuv] - c5
            k[m][fuv] += c4*(0.5392*y**2 + 0.05644*y**3)

    # Convert k → A(λ) = a_v/r_v * (k + r_v)
    return a_v / rv * (k + rv)


# ---------------------------------------------------------------------------
# apply / remove helpers  (mirrors extinction package API)
# ---------------------------------------------------------------------------

def apply_extinction(a_lambda: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Apply extinction magnitudes to a flux array.

    F_obs = F_intrinsic × 10^(−0.4 × A_lambda)

    Parameters
    ----------
    a_lambda : ndarray
        Extinction in magnitudes at each wavelength (same shape as flux).
    flux : ndarray
        Intrinsic flux array.

    Returns
    -------
    flux_obs : ndarray
        Observed (extincted) flux.
    """
    return flux * 10.0**(-0.4 * a_lambda)


def remove_extinction(a_lambda: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Remove extinction from an observed flux (de-redden).

    F_intrinsic = F_obs × 10^(+0.4 × A_lambda)

    Parameters
    ----------
    a_lambda : ndarray
        Extinction in magnitudes at each wavelength.
    flux : ndarray
        Observed (extincted) flux.

    Returns
    -------
    flux_intrinsic : ndarray
        De-reddened flux.
    """
    return flux * 10.0**(0.4 * a_lambda)


# ---------------------------------------------------------------------------
# High-level config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExtinctionConfig:
    """Configuration container for extinction + distance settings.

    All parameters are fixed (not fitted) unless you rebuild the object
    for each likelihood evaluation in your fitter.

    Parameters
    ----------
    enabled : bool
        If False, ``ExtinctionModel.apply()`` is a no-op.  Default False.
    law : str
        Name of the extinction law.  One of ``AVAILABLE_LAWS``.
    r_v : float
        R_V value (ignored by 'fm07' which fixes R_V = 3.1,
        and by 'calzetti00' which uses its own R_V = 4.05 unless overridden).
    a_v : float
        V-band extinction in magnitudes.  Default 0.0.
    gordon23_env : str
        Only used when ``law='gordon23'``.  One of 'mw', 'lmc', 'smc'.
    distance_pc : float
        Distance in parsecs.  Default 1.0 (no dilution applied unless
        ``scale_distance`` is True and a stellar radius is provided).
    scale_distance : bool
        If True, multiply the flux by (R_star_cm / distance_cm)^2
        where R_star_cm must be supplied to ``ExtinctionModel.apply()``.
        Default False.
    """
    enabled: bool       = False
    law: str            = 'fitzpatrick99'
    r_v: float          = 3.1
    a_v: float          = 0.0
    gordon23_env: str   = 'mw'
    distance_pc: float  = 1.0
    scale_distance: bool = False

    def __post_init__(self) -> None:
        if self.law not in AVAILABLE_LAWS:
            raise ValueError(
                f"law '{self.law}' not recognised.  "
                f"Choose from {AVAILABLE_LAWS}."
            )
        if self.a_v < 0.0:
            raise ValueError("a_v must be >= 0")
        if self.distance_pc <= 0.0:
            raise ValueError("distance_pc must be > 0")


# ---------------------------------------------------------------------------
# ExtinctionModel — the main user-facing class
# ---------------------------------------------------------------------------

class ExtinctionModel:
    """Apply extinction and (optionally) distance scaling to a spectrum.

    Parameters
    ----------
    config : ExtinctionConfig, optional
        Full configuration object.  If None, a default (disabled) config is used.
    **kwargs
        Convenience keyword arguments forwarded to ``ExtinctionConfig``.
        These override fields in *config* if both are supplied.

    Examples
    --------
    Disabled (default) — passes flux through unchanged::

        ext = ExtinctionModel()
        flux_out = ext.apply(wave, flux)

    Enabled, Fitzpatrick99, Av=0.5 at 200 pc::

        ext = ExtinctionModel(enabled=True, law='fitzpatrick99',
                              a_v=0.5, r_v=3.1, distance_pc=200.)
        flux_obs = ext.apply(wave, flux)

    Gordon+2023 SMC curve::

        ext = ExtinctionModel(enabled=True, law='gordon23',
                              a_v=0.8, gordon23_env='smc')
        flux_obs = ext.apply(wave, flux)
    """

    def __init__(self,
                 config: Optional[ExtinctionConfig] = None,
                 **kwargs) -> None:
        if config is None:
            config = ExtinctionConfig(**kwargs)
        elif kwargs:
            # merge kwargs on top of config
            import dataclasses
            d = dataclasses.asdict(config)
            d.update(kwargs)
            config = ExtinctionConfig(**d)
        self.config = config

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def extinction_curve(self, wave: np.ndarray) -> np.ndarray:
        """Return A(λ) in magnitudes for the configured law and A_V.

        Parameters
        ----------
        wave : array_like
            Wavelengths in Angstroms.

        Returns
        -------
        A_lambda : ndarray
            Extinction in magnitudes.  Returns zeros if ``enabled`` is False.
        """
        wave = np.asarray(wave, dtype=np.float64)
        if not self.config.enabled or self.config.a_v == 0.0:
            return np.zeros(len(wave))

        law = self.config.law
        av  = self.config.a_v
        rv  = self.config.r_v

        if law == 'ccm89':
            return ccm89(wave, av, rv)
        elif law == 'odonnell94':
            return odonnell94(wave, av, rv)
        elif law == 'fitzpatrick99':
            return fitzpatrick99(wave, av, rv)
        elif law == 'fm07':
            return fm07(wave, av)
        elif law == 'calzetti00':
            return calzetti00(wave, av, rv)
        elif law == 'gordon23':
            return gordon23(wave, av, rv if rv != 3.1 else None,
                            self.config.gordon23_env)
        else:
            raise ValueError(f"Unknown law: {law}")

    def apply(self,
              wave: np.ndarray,
              flux: np.ndarray,
              r_star_cm: Optional[float] = None) -> np.ndarray:
        """Apply extinction (and optionally distance scaling) to a flux.

        Parameters
        ----------
        wave : array_like
            Wavelengths in Angstroms.
        flux : array_like
            Intrinsic flux in any linear unit.
        r_star_cm : float, optional
            Stellar radius in cm.  Required only when
            ``config.scale_distance`` is True.

        Returns
        -------
        flux_out : ndarray
            Processed flux.  If ``enabled`` is False this is identical to
            the input flux (no copy is made if no operation is needed).
        """
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)

        if not self.config.enabled:
            return flux

        # 1. Extinction
        a_lam = self.extinction_curve(wave)
        flux_out = apply_extinction(a_lam, flux)

        # 2. Distance dilution: F_obs = F_surface × (R / d)^2
        if self.config.scale_distance:
            if r_star_cm is None:
                warnings.warn(
                    "scale_distance=True but r_star_cm was not supplied; "
                    "distance scaling skipped.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                d_cm  = self.config.distance_pc * _PC_TO_CM
                flux_out = flux_out * (r_star_cm / d_cm)**2

        return flux_out

    def remove(self,
               wave: np.ndarray,
               flux: np.ndarray) -> np.ndarray:
        """Remove (de-redden) extinction from an observed flux.

        Distance scaling is NOT reversed here — this only undoes extinction.

        Parameters
        ----------
        wave : array_like
            Wavelengths in Angstroms.
        flux : array_like
            Observed flux.

        Returns
        -------
        flux_intrinsic : ndarray
            De-reddened flux.
        """
        wave = np.asarray(wave, dtype=np.float64)
        flux = np.asarray(flux, dtype=np.float64)
        if not self.config.enabled:
            return flux
        a_lam = self.extinction_curve(wave)
        return remove_extinction(a_lam, flux)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        c = self.config
        if not c.enabled:
            return "ExtinctionModel(enabled=False)"
        return (f"ExtinctionModel(law='{c.law}', a_v={c.a_v}, r_v={c.r_v}, "
                f"distance_pc={c.distance_pc}, scale_distance={c.scale_distance})")

    @classmethod
    def disabled(cls) -> "ExtinctionModel":
        """Return an extinction model that does nothing (convenience factory)."""
        return cls(ExtinctionConfig(enabled=False))

    @classmethod
    def from_dict(cls, d: dict) -> "ExtinctionModel":
        """Construct from a plain dict (e.g. loaded from a config file)."""
        return cls(ExtinctionConfig(**d))


# ---------------------------------------------------------------------------
# Integration helper for forward models
# ---------------------------------------------------------------------------

def make_extinction_model(
    enabled: bool = False,
    law: str = 'fitzpatrick99',
    r_v: float = 3.1,
    a_v: float = 0.0,
    distance_pc: float = 1.0,
    scale_distance: bool = False,
    gordon23_env: str = 'mw',
) -> ExtinctionModel:
    """Convenience factory — mirrors the ExtinctionConfig keyword signature.

    This is the recommended way to build an extinction model for use with
    ``run_forward`` or ``run_inverse``, e.g.::

        from custom_colours import make_extinction_model, run_forward

        ext = make_extinction_model(enabled=True, law='fitzpatrick99',
                                    a_v=0.3, r_v=3.1)
        result = run_forward(..., extinction=ext)

    Parameters
    ----------
    enabled : bool
        Master switch.  Default False — extinction is skipped entirely.
    law : str
        Extinction law name.  See ``AVAILABLE_LAWS``.
    r_v : float
        R_V (shape parameter).  Default 3.1.
    a_v : float
        V-band extinction.  Default 0.0.
    distance_pc : float
        Source distance in parsecs.  Default 1 pc (absolute flux, no dilution).
    scale_distance : bool
        Apply (R/d)^2 dilution.  Default False.
    gordon23_env : str
        Gordon+2023 environment preset: 'mw', 'lmc', or 'smc'.

    Returns
    -------
    ExtinctionModel
    """
    return ExtinctionModel(ExtinctionConfig(
        enabled=enabled,
        law=law,
        r_v=r_v,
        a_v=a_v,
        distance_pc=distance_pc,
        scale_distance=scale_distance,
        gordon23_env=gordon23_env,
    ))
