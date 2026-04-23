"""
custom_colours.grid
===================
Loads a SED_Tools-prepared stellar atmosphere grid (flux_cube.bin +
lookup_table.csv) into memory and exposes the axes and flux array
for downstream interpolation.

Binary format (flux_cube.bin)
------------------------------
  Header  : 4 × int32  -> (n_teff, n_logg, n_meta, n_lambda)
  Axes    : n_teff + n_logg + n_meta + n_lambda float64 values (written
             consecutively, no padding)
  Payload : n_teff × n_logg × n_meta × n_lambda float64 values stored
             in (W, M, L, T) order on disk, loaded into (T, L, M, W).

This matches the layout written by SED_Tools and read by the MESA
colors module (colors_utils.f90 :: load_flux_cube).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtmosphereGrid:
    """Immutable container for a loaded stellar atmosphere grid.

    Attributes
    ----------
    teff_grid : ndarray, shape (n_T,)
        Effective temperature grid points in Kelvin.
    logg_grid : ndarray, shape (n_L,)
        log10(g / cm s^-2) grid points.
    meta_grid : ndarray, shape (n_M,)
        Metallicity [M/H] grid points.
    wavelengths : ndarray, shape (n_W,)
        Wavelength grid in Angstroms.
    flux : ndarray, shape (n_T, n_L, n_M, n_W)
        Surface flux in erg/s/cm^2/Å at each (Teff, logg, [M/H]) node.
    model_dir : Path
        Directory from which the grid was loaded.
    """
    teff_grid:   np.ndarray
    logg_grid:   np.ndarray
    meta_grid:   np.ndarray
    wavelengths: np.ndarray
    flux:        np.ndarray
    model_dir:   Path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def teff_bounds(self) -> tuple[float, float]:
        return float(self.teff_grid[0]), float(self.teff_grid[-1])

    @property
    def logg_bounds(self) -> tuple[float, float]:
        return float(self.logg_grid[0]), float(self.logg_grid[-1])

    @property
    def meta_bounds(self) -> tuple[float, float]:
        return float(self.meta_grid[0]), float(self.meta_grid[-1])

    def in_bounds(self, teff: float, logg: float, meta: float) -> bool:
        """Return True if (teff, logg, meta) is within the grid axes."""
        return (
            self.teff_bounds[0] <= teff <= self.teff_bounds[1]
            and self.logg_bounds[0] <= logg <= self.logg_bounds[1]
            and self.meta_bounds[0] <= meta <= self.meta_bounds[1]
        )

    def clamp(
        self, teff: float, logg: float, meta: float
    ) -> tuple[float, float, float]:
        """Clamp (teff, logg, meta) to the grid boundary."""
        teff = float(np.clip(teff, *self.teff_bounds))
        logg = float(np.clip(logg, *self.logg_bounds))
        meta = float(np.clip(meta, *self.meta_bounds))
        return teff, logg, meta

    def interp_radius(
        self, teff: float, logg: float, meta: float
    ) -> float:
        """Euclidean distance in normalised parameter space from the
        nearest grid point.  Mirrors the ``Interp_rad`` diagnostic
        produced by the MESA colors module."""
        t_norm = (self.teff_grid[-1] - self.teff_grid[0]) or 1.0
        l_norm = (self.logg_grid[-1] - self.logg_grid[0]) or 1.0
        m_norm = (self.meta_grid[-1] - self.meta_grid[0]) or 1.0

        i_t = int(np.argmin(np.abs(self.teff_grid - teff)))
        i_l = int(np.argmin(np.abs(self.logg_grid - logg)))
        i_m = int(np.argmin(np.abs(self.meta_grid - meta)))

        dt = (teff - self.teff_grid[i_t]) / t_norm
        dl = (logg - self.logg_grid[i_l]) / l_norm
        dm = (meta - self.meta_grid[i_m]) / m_norm

        return float(np.sqrt(dt**2 + dl**2 + dm**2))

    def __repr__(self) -> str:
        return (
            f"AtmosphereGrid("
            f"Teff={self.teff_bounds[0]:.0f}–{self.teff_bounds[1]:.0f} K, "
            f"logg={self.logg_bounds[0]:.2f}–{self.logg_bounds[1]:.2f}, "
            f"[M/H]={self.meta_bounds[0]:.2f}–{self.meta_bounds[1]:.2f}, "
            f"n_wave={len(self.wavelengths)}, "
            f"flux={self.flux.shape})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_grid(model_dir: str | Path) -> AtmosphereGrid:
    """Load a SED_Tools atmosphere grid from *model_dir*.

    The directory must contain:
      - ``flux_cube.bin``   — binary flux cube
      - ``lookup_table.csv`` — parameter index (used for validation only)

    Parameters
    ----------
    model_dir:
        Path to the directory produced by ``sed-tools rebuild``.

    Returns
    -------
    AtmosphereGrid

    Raises
    ------
    FileNotFoundError
        If ``flux_cube.bin`` is missing.
    ValueError
        If the binary header is invalid or the file is truncated.
    """
    model_dir = Path(model_dir).resolve()
    cube_path = model_dir / "flux_cube.bin"

    if not cube_path.exists():
        raise FileNotFoundError(
            f"flux_cube.bin not found in {model_dir}. "
            "Run 'sed-tools rebuild' to generate it."
        )

    teff_grid, logg_grid, meta_grid, wavelengths, flux = _read_flux_cube(cube_path)

    return AtmosphereGrid(
        teff_grid=teff_grid,
        logg_grid=logg_grid,
        meta_grid=meta_grid,
        wavelengths=wavelengths,
        flux=flux,
        model_dir=model_dir,
    )


# ---------------------------------------------------------------------------
# Binary reader
# ---------------------------------------------------------------------------

def _read_flux_cube(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a ``flux_cube.bin`` file.

    Returns (teff_grid, logg_grid, meta_grid, wavelengths, flux)
    where flux has shape (n_T, n_L, n_M, n_W).
    """
    with open(path, "rb") as fh:
        # --- header ---
        header_bytes = fh.read(16)          # 4 × int32 = 16 bytes
        if len(header_bytes) < 16:
            raise ValueError(f"{path}: file too short to contain a valid header")

        n_teff, n_logg, n_meta, n_lambda = struct.unpack("4i", header_bytes)

        if any(n <= 0 for n in (n_teff, n_logg, n_meta, n_lambda)):
            raise ValueError(
                f"{path}: invalid header dimensions "
                f"({n_teff}, {n_logg}, {n_meta}, {n_lambda})"
            )

        # --- axes ---
        n_axis_total = n_teff + n_logg + n_meta + n_lambda
        axes_raw = np.frombuffer(fh.read(n_axis_total * 8), dtype=np.float64)
        if axes_raw.size != n_axis_total:
            raise ValueError(f"{path}: truncated axis data")

        offset = 0
        teff_grid   = axes_raw[offset : offset + n_teff].copy();  offset += n_teff
        logg_grid   = axes_raw[offset : offset + n_logg].copy();  offset += n_logg
        meta_grid   = axes_raw[offset : offset + n_meta].copy();  offset += n_meta
        wavelengths = axes_raw[offset : offset + n_lambda].copy()

        # --- payload (W, M, L, T) on disk → (T, L, M, W) in memory ---
        n_total = n_teff * n_logg * n_meta * n_lambda
        payload = np.frombuffer(fh.read(n_total * 8), dtype=np.float64)
        if payload.size != n_total:
            raise ValueError(
                f"{path}: flux payload truncated "
                f"(got {payload.size}, expected {n_total})"
            )

        # disk layout: (n_lambda, n_meta, n_logg, n_teff)  [Fortran STREAM,
        # outermost written first = slowest-varying index in Fortran column-
        # major → wavelength is written outermost in the SED_Tools builder]
        flux = (
            payload
            .reshape(n_lambda, n_meta, n_logg, n_teff)
            .transpose(3, 2, 1, 0)          # → (T, L, M, W)
            .copy()
        )

    return teff_grid, logg_grid, meta_grid, wavelengths, flux


# ---------------------------------------------------------------------------
# Optional validation helper
# ---------------------------------------------------------------------------

def validate_lookup_table(grid: AtmosphereGrid) -> pd.DataFrame:
    """Read and return the ``lookup_table.csv`` from *grid.model_dir*.

    Useful for sanity-checking that the grid axes in the binary cube
    are consistent with the CSV metadata.  Not called at load time to
    avoid the pandas overhead on every initialisation.
    """
    lookup_path = grid.model_dir / "lookup_table.csv"
    if not lookup_path.exists():
        raise FileNotFoundError(f"lookup_table.csv not found in {grid.model_dir}")

    df = pd.read_csv(lookup_path, dtype=str)
    df.columns = [c.strip().lstrip("#").strip() for c in df.columns]
    return df
