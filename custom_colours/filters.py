"""
custom_colours.filters
======================
Loads photometric filter transmission curves from SED_Tools-prepared
``.dat`` files and precomputes Vega, AB, and ST zero-points using the
same photon-counting integrals as the MESA colors module
(colors/private/synthetic.f90).

Zero-point definitions
-----------------------
  Vega  :  F_zp = ∫ F_vega(λ) T(λ) λ dλ / ∫ T(λ) λ dλ
  AB    :  F_zp = ∫ F_AB(λ)   T(λ) λ dλ / ∫ T(λ) λ dλ
               where F_AB(λ) = 3.631e-20 × c / λ²  [erg/s/cm²/Å]
  ST    :  F_zp = 3.63e-9  (flat F_lambda, constant)

All integrals use the trapezoid rule for wavelength grids that may be
non-uniform.  The Fortran runtime uses adaptive Simpson; the difference
is negligible for the dense wavelength grids produced by SED_Tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants  (cgs, wavelength in Angstroms)
# ---------------------------------------------------------------------------

_CLIGHT_CM_S  = 2.99792458e10   # speed of light in cm/s
_AB_FNU_ZP    = 3.631e-20       # 3631 Jy in erg/s/cm^2/Hz
_ST_FLAM_ZP   = 3.63e-9         # flat f_lambda zero-point erg/s/cm^2/Å


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class Filter:
    """A single photometric filter with precomputed zero-points.

    Attributes
    ----------
    name : str
        Short filter identifier (filename stem, e.g. ``'B'``, ``'Gbp'``).
    path : Path
        Absolute path to the source ``.dat`` file.
    wavelengths : ndarray, shape (n,)
        Filter wavelength grid in Angstroms.
    transmission : ndarray, shape (n,)
        Dimensionless transmission in [0, 1].
    vega_zero_point : float
        Photon-counting flux zero-point for the Vega magnitude system.
        ``-1.0`` if no Vega SED was supplied at load time.
    ab_zero_point : float
        Photon-counting flux zero-point for the AB magnitude system.
    st_zero_point : float
        Photon-counting flux zero-point for the ST magnitude system.
    """
    name:            str
    path:            Path
    wavelengths:     np.ndarray
    transmission:    np.ndarray
    vega_zero_point: float = field(default=-1.0)
    ab_zero_point:   float = field(default=-1.0)
    st_zero_point:   float = field(default=-1.0)

    def zero_point(self, system: str) -> float:
        """Return the zero-point for *system* ('Vega', 'AB', or 'ST')."""
        s = system.upper()
        if s == "VEGA":
            if self.vega_zero_point < 0:
                raise ValueError(
                    f"Filter '{self.name}': Vega zero-point not available. "
                    "Supply a Vega SED path to load_filters()."
                )
            return self.vega_zero_point
        if s == "AB":
            return self.ab_zero_point
        if s == "ST":
            return self.st_zero_point
        raise ValueError(f"Unknown magnitude system '{system}'. Choose Vega, AB, or ST.")

    def __repr__(self) -> str:
        return (
            f"Filter(name='{self.name}', "
            f"λ={self.wavelengths[0]:.0f}–{self.wavelengths[-1]:.0f} Å, "
            f"n={len(self.wavelengths)})"
        )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_filters(
    filter_paths: list[str | Path],
    vega_sed_path: str | Path | None = None,
) -> list[Filter]:
    """Load a list of filter ``.dat`` files and return a list of
    :class:`Filter` objects with precomputed zero-points.

    Parameters
    ----------
    filter_paths:
        Paths to individual ``*.dat`` filter transmission files in the
        SED_Tools two-column format (wavelength Å, transmission 0–1).
        Comments starting with ``#`` are skipped.
    vega_sed_path:
        Optional path to a Vega reference SED CSV file
        (``wavelength,flux`` header, wavelength in Å,
        flux in erg/s/cm²/Å).  Required for Vega zero-points.

    Returns
    -------
    list of Filter, in the same order as *filter_paths*.
    """
    vega_wave: np.ndarray | None = None
    vega_flux: np.ndarray | None = None
    if vega_sed_path is not None:
        vega_wave, vega_flux = _load_vega_sed(Path(vega_sed_path))

    filters: list[Filter] = []
    for p in filter_paths:
        p = Path(p).resolve()
        wave, trans = _load_filter_dat(p)
        filt = Filter(
            name=p.stem,
            path=p,
            wavelengths=wave,
            transmission=trans,
        )
        filt.ab_zero_point = _compute_ab_zero_point(wave, trans)
        filt.st_zero_point = _compute_st_zero_point(wave, trans)
        if vega_wave is not None and vega_flux is not None:
            filt.vega_zero_point = _compute_vega_zero_point(
                vega_wave, vega_flux, wave, trans
            )
        filters.append(filt)

    return filters


def load_filters_from_instrument_dir(
    instrument_dir: str | Path,
    vega_sed_path: str | Path | None = None,
) -> list[Filter]:
    """Load all filters listed in a SED_Tools instrument directory.

    The directory must contain an index file whose name matches the
    last component of *instrument_dir* (e.g. ``Johnson/Johnson``) that
    lists one ``.dat`` filename per line.  This mirrors the structure
    expected by the MESA colors module.

    Parameters
    ----------
    instrument_dir:
        Path to a ``data/filters/<Facility>/<Instrument>/`` directory.
    vega_sed_path:
        Optional Vega reference SED for Vega zero-point computation.

    Returns
    -------
    list of Filter
    """
    instrument_dir = Path(instrument_dir).resolve()
    index_name = instrument_dir.name
    index_file = instrument_dir / index_name

    if not index_file.exists():
        # Fallback: load every .dat in the directory
        dat_files = sorted(instrument_dir.glob("*.dat"))
        if not dat_files:
            raise FileNotFoundError(
                f"No index file '{index_name}' and no .dat files in {instrument_dir}"
            )
    else:
        dat_files = []
        for line in index_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            dat_files.append(instrument_dir / line)

    return load_filters(dat_files, vega_sed_path=vega_sed_path)


# ---------------------------------------------------------------------------
# Internal I/O helpers
# ---------------------------------------------------------------------------

def _load_filter_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column filter .dat file (wavelength Å, transmission)."""
    if not path.exists():
        raise FileNotFoundError(f"Filter file not found: {path}")

    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"{path}: expected two-column file (wavelength, transmission)"
        )

    wave  = data[:, 0].astype(np.float64)
    trans = data[:, 1].astype(np.float64)

    # Enforce ascending wavelength
    if not np.all(np.diff(wave) > 0):
        order = np.argsort(wave)
        wave, trans = wave[order], trans[order]

    return wave, trans


def _load_vega_sed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Vega reference SED CSV (wavelength Å, flux erg/s/cm²/Å)."""
    if not path.exists():
        raise FileNotFoundError(f"Vega SED not found: {path}")

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    wave = data[:, 0].astype(np.float64)
    flux = data[:, 1].astype(np.float64)

    if not np.all(np.diff(wave) > 0):
        order = np.argsort(wave)
        wave, flux = wave[order], flux[order]

    return wave, flux


# ---------------------------------------------------------------------------
# Zero-point computation  (mirrors synthetic.f90)
# ---------------------------------------------------------------------------

def _trapz(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal integration (scalar result)."""
    return float(np.trapz(y, x))


def _compute_vega_zero_point(
    vega_wave: np.ndarray,
    vega_flux: np.ndarray,
    filt_wave: np.ndarray,
    filt_trans: np.ndarray,
) -> float:
    """Photon-counting Vega zero-point.

    Interpolates the filter transmission onto the Vega wavelength grid
    then integrates:
      F_zp = ∫ F_vega(λ) T(λ) λ dλ / ∫ T(λ) λ dλ
    """
    trans_on_vega = np.interp(vega_wave, filt_wave, filt_trans, left=0.0, right=0.0)
    num = _trapz(vega_wave, vega_flux * trans_on_vega * vega_wave)
    den = _trapz(vega_wave, trans_on_vega * vega_wave)
    return num / den if den > 0.0 else -1.0


def _compute_ab_zero_point(
    filt_wave: np.ndarray,
    filt_trans: np.ndarray,
) -> float:
    """Photon-counting AB zero-point.

    Constructs F_AB(λ) = 3.631e-20 × c_cm / λ² on the filter grid then:
      F_zp = ∫ F_AB(λ) T(λ) λ dλ / ∫ T(λ) λ dλ
    """
    # Convert f_nu (flat 3631 Jy) to f_lambda; wavelength in Å → multiply c
    # by 1e8 to convert cm/s → Å/s so units cancel correctly
    f_ab = _AB_FNU_ZP * (_CLIGHT_CM_S * 1e8) / (filt_wave ** 2)
    num = _trapz(filt_wave, f_ab * filt_trans * filt_wave)
    den = _trapz(filt_wave, filt_trans * filt_wave)
    return num / den if den > 0.0 else -1.0


def _compute_st_zero_point(
    filt_wave: np.ndarray,
    filt_trans: np.ndarray,
) -> float:
    """Photon-counting ST zero-point.

    F_ST(λ) = 3.63e-9 erg/s/cm²/Å (flat spectrum):
      F_zp = ∫ F_ST T(λ) λ dλ / ∫ T(λ) λ dλ
           = 3.63e-9  (constant cancels)
    Computed explicitly for consistency with the Fortran implementation.
    """
    f_st = np.full_like(filt_wave, _ST_FLAM_ZP)
    num = _trapz(filt_wave, f_st * filt_trans * filt_wave)
    den = _trapz(filt_wave, filt_trans * filt_wave)
    return num / den if den > 0.0 else -1.0
