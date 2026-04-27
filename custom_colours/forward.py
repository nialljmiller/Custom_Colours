"""
custom_colours.forward
======================
Forward model: stellar parameters → SED + synthetic photometry.

Pipeline
--------
  (Teff, logg, [M/H])  →  SED interpolation  (Fortran, Hermite or linear)
                       →  distance dilution   (Fortran, (R/d)²)
                       →  extinction          (Python, optional, sed_extinction)
                       →  bolometric          (Fortran)
                       →  filter convolution  (Fortran, per filter)
                       →  ForwardResult

All five physical parameters — Teff, logg, [M/H], Av, distance — can be
fixed or free, described by a FitParams object (custom_colours.params).
This shared vocabulary is what makes the module bidirectional: the inverse
model unpacks a theta vector with FitParams.unpack and passes the result
directly to run_forward.

Backward-compatible call
------------------------
The original positional signature still works::

    run_forward(teff, logg, meta, R, d, grid, filters)

FitParams call (used by the inverse model)::

    run_forward(fit_params=params, theta=theta_vec,
                R=R_sun, grid=grid, filters=filters)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from .grid import AtmosphereGrid
from .filters import Filter
from .params import FitParams

if TYPE_CHECKING:
    from sed_extinction import ExtinctionModel


# ---------------------------------------------------------------------------
# Fortran extension
# ---------------------------------------------------------------------------

def _get_cc_api():
    try:
        from . import cc_api as _ext
        # f2py wraps each Fortran module as a submodule of the extension.
        # Functions defined in `module cc_api` live at _ext.cc_api.<function>.
        return _ext.cc_api
    except ImportError as exc:
        raise ImportError(
            "The Fortran extension 'cc_api' is not built. "
            "Run 'make' in the Custom_Colours root directory."
        ) from exc


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ForwardResult:
    """Output of one forward-model evaluation.

    The five physical parameters that produced this result are all stored
    so the result is self-describing.  The inverse model uses
    ``ForwardResult.magnitudes`` to compute the likelihood; the forward
    model reads parameters from FitParams — both sides work with the same
    data structure.
    """
    wavelengths:        np.ndarray
    surface_flux:       np.ndarray
    observed_flux:      np.ndarray
    magnitudes:         dict
    band_fluxes:        dict
    bol_flux:           float
    bol_mag:            float
    interp_radius:      float
    clamped:            bool
    teff:               float
    logg:               float
    meta:               float
    R:                  float
    d:                  float
    a_v:                float = 0.0
    mag_system:         str   = "AB"
    extinction_applied: bool  = False

    def __repr__(self) -> str:
        mags = ", ".join(f"{k}={v:.3f}" for k, v in self.magnitudes.items())
        ext  = f", Av={self.a_v:.3f}" if self.extinction_applied else ""
        return (
            f"ForwardResult("
            f"Teff={self.teff:.0f} K, logg={self.logg:.2f}, [M/H]={self.meta:.2f}"
            f"{ext}, bol_mag={self.bol_mag:.3f}, "
            f"mags=[{mags}], clamped={self.clamped})"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_forward(
    # classic positional args
    teff:    Optional[float] = None,
    logg:    Optional[float] = None,
    meta:    Optional[float] = None,
    R:       Optional[float] = None,
    d:       Optional[float] = None,
    grid:    Optional[AtmosphereGrid] = None,
    filters: Optional[list] = None,
    # options
    mag_system:    str = "AB",
    interp_method: str = "hermite",
    extinction:    Optional["ExtinctionModel"] = None,
    # FitParams interface
    fit_params: Optional[FitParams] = None,
    theta:      Optional[np.ndarray] = None,
) -> ForwardResult:
    """Evaluate the forward model for one set of stellar parameters.

    Two calling conventions are supported.

    **Classic** (backward-compatible)::

        run_forward(teff, logg, meta, R, d, grid, filters)

    **FitParams** (used by the inverse model)::

        run_forward(fit_params=params, theta=theta_vec,
                    R=R_sun, grid=grid, filters=filters)

    In the FitParams convention ``theta`` contains only the *free* parameters
    in canonical order (teff, logg, meta, a_v, d — skipping fixed ones).
    ``fit_params.unpack(theta)`` fills in the fixed values.  If Av or
    distance are free parameters they are taken from theta, not from any
    keyword argument.

    Parameters
    ----------
    teff, logg, meta : float
        Atmospheric parameters.  Required in classic mode.
    R : float
        Stellar radius in cm.  Always required.
    d : float
        Distance in cm.  Required in classic mode; ignored if distance is
        a free parameter in fit_params.
    grid : AtmosphereGrid
    filters : list of Filter
    mag_system : {'AB', 'Vega', 'ST'}
    interp_method : {'hermite', 'linear'}
    extinction : ExtinctionModel or None
        Applied after dilution and before filter convolution.
        When Av is free in fit_params, the model's stored a_v is
        overridden by the value from theta at each call.
    fit_params : FitParams or None
    theta : array-like or None
        Required when fit_params is provided.
    """
    # ------------------------------------------------------------------
    # Resolve parameters from whichever calling convention is used
    # ------------------------------------------------------------------
    if fit_params is not None:
        if theta is None:
            raise ValueError("theta must be supplied when fit_params is used")
        p     = fit_params.unpack(np.asarray(theta, dtype=np.float64))
        teff  = float(p['teff'])
        logg  = float(p['logg'])
        meta  = float(p['meta'])
        d_use = float(p['d'])
        av    = float(p['a_v'])
    else:
        if any(x is None for x in (teff, logg, meta, R, d, grid, filters)):
            raise ValueError(
                "teff, logg, meta, R, d, grid, and filters must all be "
                "supplied when fit_params is not used."
            )
        d_use = float(d)
        av    = 0.0

    cc = _get_cc_api()

    # ------------------------------------------------------------------
    # Compute interp_radius and clamped flag via grid helpers
    # (same logic as the original forward.py before the extinction rewrite)
    # ------------------------------------------------------------------
    clamped    = not grid.in_bounds(teff, logg, meta)
    interp_rad = grid.interp_radius(teff, logg, meta)
    teff_q, logg_q, meta_q = grid.clamp(teff, logg, meta)

    # ------------------------------------------------------------------
    # 1. SED interpolation (Fortran)
    #
    # cc.interp_sed_hermite / interp_sed_linear return (result_flux, ierr).
    # The flux array is C-contiguous float64; Fortran expects the cube as
    # (nt, nl, nm, nw) which is grid.flux's native shape.
    # ------------------------------------------------------------------
    flux_cube  = np.asfortranarray(grid.flux, dtype=np.float64)
    teff_grid  = np.ascontiguousarray(grid.teff_grid,   dtype=np.float64)
    logg_grid  = np.ascontiguousarray(grid.logg_grid,   dtype=np.float64)
    meta_grid  = np.ascontiguousarray(grid.meta_grid,   dtype=np.float64)
    wavelengths = np.ascontiguousarray(grid.wavelengths, dtype=np.float64)

    if interp_method == "hermite":
        surface_flux, ierr = cc.interp_sed_hermite(
            teff_q, logg_q, meta_q,
            teff_grid, logg_grid, meta_grid,
            flux_cube,
        )
    elif interp_method == "linear":
        surface_flux, ierr = cc.interp_sed_linear(
            teff_q, logg_q, meta_q,
            teff_grid, logg_grid, meta_grid,
            flux_cube,
        )
    else:
        raise ValueError(
            f"Unknown interp_method '{interp_method}'. "
            "Choose 'hermite' or 'linear'."
        )

    if ierr != 0:
        clamped = True

    # ------------------------------------------------------------------
    # 2. Distance dilution: F_obs = F_surface × (R/d)²  (Fortran)
    # ------------------------------------------------------------------
    observed_flux = cc.dilute_flux(surface_flux, float(R), d_use)

    # ------------------------------------------------------------------
    # 3. Extinction (Python, optional)
    #
    # Applied AFTER dilution, BEFORE filter convolution.
    # Physics: star → distance → dust column → telescope → filter.
    #
    # When Av is a free parameter, rebuild the ExtinctionModel with the
    # Av value from theta so every likelihood call uses the correct value.
    # ------------------------------------------------------------------
    extinction_applied = False
    if extinction is not None and getattr(extinction.config, 'enabled', False):
        if fit_params is not None and fit_params.a_v.is_free:
            from dataclasses import replace as _replace
            new_cfg   = _replace(extinction.config, a_v=av)
            from sed_extinction import ExtinctionModel as _EM
            extinction = _EM(new_cfg)

        observed_flux      = extinction.apply(wavelengths, observed_flux)
        extinction_applied = True
        av                 = extinction.config.a_v

    # ------------------------------------------------------------------
    # 4. Bolometric quantities (Fortran)
    # ------------------------------------------------------------------
    bol_flux, bol_mag, _ = cc.bolometric(wavelengths, observed_flux)

    # ------------------------------------------------------------------
    # 5. Synthetic photometry per filter (Fortran)
    # ------------------------------------------------------------------
    magnitudes:  dict = {}
    band_fluxes: dict = {}

    for filt in filters:
        zp         = filt.zero_point(mag_system)
        filt_wave  = np.ascontiguousarray(filt.wavelengths,  dtype=np.float64)
        filt_trans = np.ascontiguousarray(filt.transmission, dtype=np.float64)
        mag, band_flux, _ = cc.synthetic_magnitude(
            wavelengths, observed_flux,
            filt_wave, filt_trans,
            zp,
        )
        magnitudes[filt.name]  = float(mag)
        band_fluxes[filt.name] = float(band_flux)

    return ForwardResult(
        wavelengths=wavelengths,
        surface_flux=surface_flux,
        observed_flux=observed_flux,
        magnitudes=magnitudes,
        band_fluxes=band_fluxes,
        bol_flux=float(bol_flux),
        bol_mag=float(bol_mag),
        interp_radius=float(interp_rad),
        clamped=clamped,
        teff=float(teff),
        logg=float(logg),
        meta=float(meta),
        R=float(R),
        d=d_use,
        a_v=av,
        mag_system=mag_system,
        extinction_applied=extinction_applied,
    )


# ---------------------------------------------------------------------------
# Batch wrapper (classic convention only)
# ---------------------------------------------------------------------------

def run_forward_batch(
    params: np.ndarray,
    R: float,
    d: float,
    grid: AtmosphereGrid,
    filters: list,
    mag_system: str = "AB",
    interp_method: str = "hermite",
    extinction: Optional["ExtinctionModel"] = None,
) -> list:
    """Run run_forward over an array of (teff, logg, meta) rows.

    Parameters
    ----------
    params : ndarray, shape (N, 3)
    Returns
    -------
    list of ForwardResult, length N.
    """
    params = np.atleast_2d(params)
    if params.shape[1] != 3:
        raise ValueError("params must have shape (N, 3): teff, logg, meta")
    return [
        run_forward(
            teff=float(r[0]), logg=float(r[1]), meta=float(r[2]),
            R=R, d=d, grid=grid, filters=filters,
            mag_system=mag_system, interp_method=interp_method,
            extinction=extinction,
        )
        for r in params
    ]
