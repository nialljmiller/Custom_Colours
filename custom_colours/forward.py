"""
custom_colours.forward
======================
Forward model: (Teff, logg, [M/H], R, d) → SED + synthetic photometry.

Calls the Fortran kernels via cc_api for all numerically sensitive
steps.  All I/O and orchestration is in Python.

Returns a ForwardResult dataclass containing:
  - wavelengths   : ndarray (Å)
  - surface_flux  : ndarray (erg/s/cm²/Å) at stellar surface
  - observed_flux : ndarray (erg/s/cm²/Å) at observer (diluted, extincted)
  - magnitudes    : dict[filter_name -> float]
  - band_fluxes   : dict[filter_name -> float]
  - bol_flux      : float  (erg/s/cm²)
  - bol_mag       : float
  - interp_radius : float  (diagnostic — distance in normalised param space)
  - clamped       : bool   (True if params were outside grid and clamped)

Extinction
----------
Pass an ``ExtinctionModel`` from ``sed_extinction`` to apply interstellar
dust reddening after distance dilution and before filter convolution.
Extinction is disabled by default — passing ``None`` or an
``ExtinctionModel(enabled=False)`` is a strict no-op with zero overhead.

    from sed_extinction import ExtinctionModel
    ext = ExtinctionModel(enabled=True, law='fitzpatrick99', a_v=0.3, r_v=3.1)
    result = run_forward(..., extinction=ext)

Distance note
-------------
Distance dilution is still handled by the Fortran ``dilute_flux`` kernel
(R/d)^2 exactly as before.  The ``ExtinctionModel`` does NOT apply its own
distance scaling in this context — set ``scale_distance=False`` (the default)
on the model you pass in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from .grid import AtmosphereGrid
from .filters import Filter

if TYPE_CHECKING:
    # Avoid a hard import at module level so that sed_extinction remains
    # optional — if it is not installed forward.py still works as long as
    # no ExtinctionModel is passed.
    from sed_extinction import ExtinctionModel


# ---------------------------------------------------------------------------
# Lazy import of the Fortran extension
# ---------------------------------------------------------------------------

def _get_cc_api():
    try:
        from . import cc_api
        return cc_api
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
    """Output of a single forward-model evaluation.

    Attributes
    ----------
    wavelengths : ndarray, shape (n_wave,)
        Wavelength grid in Angstroms.
    surface_flux : ndarray, shape (n_wave,)
        Interpolated stellar surface flux (erg/s/cm²/Å).
    observed_flux : ndarray, shape (n_wave,)
        Observer-frame flux after (R/d)^2 dilution and, if an
        ExtinctionModel was supplied, interstellar reddening
        (erg/s/cm²/Å).
    magnitudes : dict[str, float]
        Synthetic magnitude per filter (key = filter name).
    band_fluxes : dict[str, float]
        In-band photon-counting flux per filter (erg/s/cm²/Å).
    bol_flux : float
        Bolometric flux integrated over the SED (erg/s/cm²).
    bol_mag : float
        Bolometric magnitude.
    interp_radius : float
        Euclidean distance in normalised (Teff, logg, [M/H]) space
        from the query point to the nearest grid node.  0 = exact node.
    clamped : bool
        True if the query point was outside the grid and was clamped
        to the nearest boundary before interpolation.
    teff, logg, meta, R, d, mag_system : scalars
        Input parameters echoed back for convenience.
    extinction_applied : bool
        True if an enabled ExtinctionModel was applied.
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
    mag_system:         str = "AB"
    extinction_applied: bool = False

    def __repr__(self) -> str:
        mags = ", ".join(f"{k}={v:.3f}" for k, v in self.magnitudes.items())
        ext_tag = " [ext]" if self.extinction_applied else ""
        return (
            f"ForwardResult(Teff={self.teff:.0f} K, logg={self.logg:.2f}, "
            f"[M/H]={self.meta:.2f}, "
            f"bol_mag={self.bol_mag:.3f}, "
            f"mags=[{mags}], "
            f"clamped={self.clamped}{ext_tag})"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_forward(
    teff: float,
    logg: float,
    meta: float,
    R: float,
    d: float,
    grid: AtmosphereGrid,
    filters: list,
    mag_system: str = "AB",
    interp_method: str = "hermite",
    extinction: Optional["ExtinctionModel"] = None,
) -> ForwardResult:
    """Run the forward model for a single set of stellar parameters.

    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin.
    logg : float
        Log surface gravity (log₁₀ g / cm s⁻²).
    meta : float
        Metallicity [M/H] in dex.
    R : float
        Stellar radius in cm.
    d : float
        Distance in cm.
    grid : AtmosphereGrid
        Loaded atmosphere grid (from ``load_grid``).
    filters : list of Filter
        Filter curves to compute photometry for.
    mag_system : {'AB', 'Vega', 'ST'}
        Photometric zero-point system.
    interp_method : {'hermite', 'linear'}
        Interpolation method in the flux cube.
    extinction : ExtinctionModel or None
        Optional dust extinction model from ``sed_extinction``.
        Applied to the diluted observed flux before filter convolution.
        Default None — no extinction.

    Returns
    -------
    ForwardResult
    """
    cc = _get_cc_api()

    # ------------------------------------------------------------------
    # 1. Interpolate SED from flux cube (Fortran)
    # ------------------------------------------------------------------
    if interp_method == "hermite":
        surface_flux, interp_rad, clamped = cc.interp_sed_hermite(
            teff, logg, meta,
            grid.teff_grid, grid.logg_grid, grid.meta_grid,
            grid.flux_cube,
        )
    elif interp_method == "linear":
        surface_flux, interp_rad, clamped = cc.interp_sed_linear(
            teff, logg, meta,
            grid.teff_grid, grid.logg_grid, grid.meta_grid,
            grid.flux_cube,
        )
    else:
        raise ValueError(f"Unknown interp_method '{interp_method}'. "
                         f"Choose 'hermite' or 'linear'.")

    wavelengths = grid.wavelengths.copy()
    clamped     = bool(clamped)

    # ------------------------------------------------------------------
    # 2. Distance dilution: F_obs = F_surface × (R/d)² (Fortran)
    # ------------------------------------------------------------------
    observed_flux = cc.dilute_flux(surface_flux, float(R), float(d))

    # ------------------------------------------------------------------
    # 3. Interstellar extinction (optional, Python)
    #
    # Applied AFTER dilution and BEFORE filter convolution.
    # F_extincted(λ) = F_diluted(λ) × 10^(−0.4 × A(λ))
    #
    # This is the physically correct position in the pipeline:
    # the photons travel from the star surface → distance dilution →
    # dust column → telescope → filter.
    # ------------------------------------------------------------------
    extinction_applied = False
    if extinction is not None and getattr(extinction.config, 'enabled', False):
        observed_flux      = extinction.apply(wavelengths, observed_flux)
        extinction_applied = True

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

        mag, band_flux, ierr_f = cc.synthetic_magnitude(
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
        teff=teff,
        logg=logg,
        meta=meta,
        R=float(R),
        d=float(d),
        mag_system=mag_system,
        extinction_applied=extinction_applied,
    )


# ---------------------------------------------------------------------------
# Vectorised convenience wrapper
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
    """Run the forward model over an array of (teff, logg, meta) rows.

    Parameters
    ----------
    params : ndarray, shape (N, 3)
        Each row is (teff, logg, meta).
    extinction : ExtinctionModel or None
        Passed through to each ``run_forward`` call unchanged.

    Returns
    -------
    list of ForwardResult, length N.
    """
    params = np.atleast_2d(params)
    if params.shape[1] != 3:
        raise ValueError("params must have shape (N, 3): columns are teff, logg, meta")

    return [
        run_forward(
            teff=float(row[0]),
            logg=float(row[1]),
            meta=float(row[2]),
            R=R,
            d=d,
            grid=grid,
            filters=filters,
            mag_system=mag_system,
            interp_method=interp_method,
            extinction=extinction,
        )
        for row in params
    ]
