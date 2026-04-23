"""
custom_colours.forward
======================
Forward model: (Teff, logg, [M/H], R, d) → SED + synthetic photometry.

Calls the Fortran kernels via cc_api for all numerically sensitive
steps.  All I/O and orchestration is in Python.

Returns a ForwardResult dataclass containing:
  - wavelengths   : ndarray (Å)
  - surface_flux  : ndarray (erg/s/cm²/Å) at stellar surface
  - observed_flux : ndarray (erg/s/cm²/Å) at observer (diluted)
  - magnitudes    : dict[filter_name -> float]
  - band_fluxes   : dict[filter_name -> float]
  - bol_flux      : float  (erg/s/cm²)
  - bol_mag       : float
  - interp_radius : float  (diagnostic — distance in normalised param space)
  - clamped       : bool   (True if params were outside grid and clamped)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import AtmosphereGrid
from .filters import Filter


# ---------------------------------------------------------------------------
# Lazy import of the Fortran extension so that import errors are informative
# ---------------------------------------------------------------------------

def _get_cc_api():
    try:
        from . import cc_api as _cc_api_mod
        # f2py wraps each Fortran module as a submodule of the extension.
        # Functions defined in module cc_api live at _cc_api_mod.cc_api.
        return _cc_api_mod.cc_api
    except ImportError as exc:
        raise ImportError(
            "The Fortran extension 'cc_api' is not built. "
            "Run 'pip install -e .' in the Custom_Colours root directory."
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
        Diluted observer-frame flux (erg/s/cm²/Å).
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
    teff : float
    logg : float
    meta : float
    R : float
        Stellar radius in cm.
    d : float
        Distance in cm.
    mag_system : str
        Photometric system used ('Vega', 'AB', or 'ST').
    """
    wavelengths:   np.ndarray
    surface_flux:  np.ndarray
    observed_flux: np.ndarray
    magnitudes:    dict[str, float]
    band_fluxes:   dict[str, float]
    bol_flux:      float
    bol_mag:       float
    interp_radius: float
    clamped:       bool
    teff:          float
    logg:          float
    meta:          float
    R:             float
    d:             float
    mag_system:    str = "AB"

    def __repr__(self) -> str:
        mags = ", ".join(f"{k}={v:.3f}" for k, v in self.magnitudes.items())
        return (
            f"ForwardResult(Teff={self.teff:.0f} K, logg={self.logg:.2f}, "
            f"[M/H]={self.meta:.2f}, "
            f"bol_mag={self.bol_mag:.3f}, "
            f"mags=[{mags}], "
            f"clamped={self.clamped})"
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
    filters: list[Filter],
    mag_system: str = "AB",
    interp_method: str = "hermite",
) -> ForwardResult:
    """Run the forward model for a single set of stellar parameters.

    Parameters
    ----------
    teff : float
        Effective temperature in Kelvin.
    logg : float
        log10(surface gravity / cm s^-2).
    meta : float
        Metallicity [M/H].
    R : float
        Stellar radius in cm.
    d : float
        Distance in cm.  Use 3.0857e19 for 10 pc (absolute magnitudes).
    grid : AtmosphereGrid
        Loaded atmosphere grid from ``custom_colours.grid.load_grid``.
    filters : list of Filter
        Loaded filters from ``custom_colours.filters.load_filters``.
        Each must have zero-points precomputed for *mag_system*.
    mag_system : str
        Photometric system: 'Vega', 'AB', or 'ST'.
    interp_method : str
        Interpolation method: 'hermite' (default) or 'linear'.

    Returns
    -------
    ForwardResult
    """
    cc = _get_cc_api()

    # ------------------------------------------------------------------
    # 1. Clamp to grid if necessary, record interp_radius
    # ------------------------------------------------------------------
    clamped = not grid.in_bounds(teff, logg, meta)
    teff_q, logg_q, meta_q = grid.clamp(teff, logg, meta)
    interp_rad = grid.interp_radius(teff, logg, meta)

    # ------------------------------------------------------------------
    # 2. SED interpolation (Fortran)
    # ------------------------------------------------------------------
    nt, nl, nm, nw = grid.flux.shape

    # Fortran expects C-contiguous float64 arrays
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
        raise ValueError(f"Unknown interp_method '{interp_method}'. Use 'hermite' or 'linear'.")

    if ierr != 0:
        clamped = True

    # ------------------------------------------------------------------
    # 3. Flux dilution (Fortran)
    # ------------------------------------------------------------------
    observed_flux = cc.dilute_flux(surface_flux, float(R), float(d))

    # ------------------------------------------------------------------
    # 4. Bolometric quantities (Fortran)
    # ------------------------------------------------------------------
    bol_flux, bol_mag, _ = cc.bolometric(wavelengths, observed_flux)

    # ------------------------------------------------------------------
    # 5. Synthetic photometry per filter (Fortran)
    # ------------------------------------------------------------------
    magnitudes:  dict[str, float] = {}
    band_fluxes: dict[str, float] = {}

    for filt in filters:
        zp = filt.zero_point(mag_system)
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
        interp_radius=interp_rad,
        clamped=clamped,
        teff=teff,
        logg=logg,
        meta=meta,
        R=float(R),
        d=float(d),
        mag_system=mag_system,
    )


# ---------------------------------------------------------------------------
# Vectorised convenience wrapper
# ---------------------------------------------------------------------------

def run_forward_batch(
    params: np.ndarray,
    R: float,
    d: float,
    grid: AtmosphereGrid,
    filters: list[Filter],
    mag_system: str = "AB",
    interp_method: str = "hermite",
) -> list[ForwardResult]:
    """Run the forward model over an array of (teff, logg, meta) rows.

    Parameters
    ----------
    params : ndarray, shape (N, 3)
        Each row is (teff, logg, meta).

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
        )
        for row in params
    ]
