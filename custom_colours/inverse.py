"""
custom_colours.inverse
======================
Inverse model: observed magnitudes → posterior on (Teff, logg, [M/H]).

Uses emcee (MCMC) with a Gaussian likelihood on the residuals between
observed and model magnitudes.  The forward model is called at every
walker step, so the Fortran kernels must be built.

Public API
----------
run_inverse(obs_magnitudes, obs_uncertainties, filter_names,
            R, d, grid, filters, ...)
    → InverseResult

The filter ordering in obs_magnitudes / obs_uncertainties must match
the order of filter_names, which in turn must match names in the
`filters` list passed in.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .grid import AtmosphereGrid
from .filters import Filter
from .forward import run_forward
from .io import InverseResult


# ---------------------------------------------------------------------------
# Likelihood and prior
# ---------------------------------------------------------------------------

def _log_prior(theta: np.ndarray, grid: AtmosphereGrid) -> float:
    """Flat prior within grid bounds, -inf outside."""
    teff, logg, meta = theta
    if (grid.teff_bounds[0] <= teff <= grid.teff_bounds[1]
            and grid.logg_bounds[0] <= logg <= grid.logg_bounds[1]
            and grid.meta_bounds[0] <= meta <= grid.meta_bounds[1]):
        return 0.0
    return -np.inf


def _log_likelihood(
    theta: np.ndarray,
    obs_mag: np.ndarray,
    obs_err: np.ndarray,
    filter_order: list[str],
    R: float,
    d: float,
    grid: AtmosphereGrid,
    filters: list[Filter],
    mag_system: str,
    interp_method: str,
) -> float:
    """Gaussian log-likelihood summed over all filters.

    L = -0.5 × Σ [ (m_obs - m_model)² / σ² + ln(2π σ²) ]
    """
    teff, logg, meta = theta
    try:
        result = run_forward(
            teff=teff, logg=logg, meta=meta,
            R=R, d=d,
            grid=grid,
            filters=filters,
            mag_system=mag_system,
            interp_method=interp_method,
        )
    except Exception:
        return -np.inf

    ll = 0.0
    for i, fname in enumerate(filter_order):
        m_model = result.magnitudes.get(fname, None)
        if m_model is None or not np.isfinite(m_model):
            return -np.inf
        sigma2 = obs_err[i] ** 2
        ll += -0.5 * ((obs_mag[i] - m_model) ** 2 / sigma2 + np.log(2.0 * np.pi * sigma2))

    return ll


def _log_posterior(
    theta: np.ndarray,
    obs_mag: np.ndarray,
    obs_err: np.ndarray,
    filter_order: list[str],
    R: float,
    d: float,
    grid: AtmosphereGrid,
    filters: list[Filter],
    mag_system: str,
    interp_method: str,
) -> float:
    lp = _log_prior(theta, grid)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(
        theta, obs_mag, obs_err, filter_order,
        R, d, grid, filters, mag_system, interp_method,
    )


# ---------------------------------------------------------------------------
# Initial position sampler
# ---------------------------------------------------------------------------

def _initial_positions(
    n_walkers: int,
    grid: AtmosphereGrid,
    p0_teff: Optional[float],
    p0_logg: Optional[float],
    p0_meta: Optional[float],
    scatter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate initial walker positions in a Gaussian ball.

    If p0_* values are not supplied, the grid centre is used.
    The ball is clamped to remain within grid bounds.
    """
    centre = np.array([
        p0_teff if p0_teff is not None else np.mean(grid.teff_grid),
        p0_logg if p0_logg is not None else np.mean(grid.logg_grid),
        p0_meta if p0_meta is not None else np.mean(grid.meta_grid),
    ])

    # Scatter relative to grid range
    scales = np.array([
        scatter * (grid.teff_bounds[1] - grid.teff_bounds[0]),
        scatter * (grid.logg_bounds[1] - grid.logg_bounds[0]),
        scatter * (grid.meta_bounds[1] - grid.meta_bounds[0]),
    ])

    pos = centre + scales * rng.standard_normal((n_walkers, 3))

    # Clamp each walker to bounds
    pos[:, 0] = np.clip(pos[:, 0], *grid.teff_bounds)
    pos[:, 1] = np.clip(pos[:, 1], *grid.logg_bounds)
    pos[:, 2] = np.clip(pos[:, 2], *grid.meta_bounds)

    return pos


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_inverse(
    obs_magnitudes: np.ndarray | list[float],
    obs_uncertainties: np.ndarray | list[float],
    filter_names: list[str],
    R: float,
    d: float,
    grid: AtmosphereGrid,
    filters: list[Filter],
    mag_system: str = "AB",
    interp_method: str = "hermite",
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    n_thin: int = 1,
    p0_teff: Optional[float] = None,
    p0_logg: Optional[float] = None,
    p0_meta: Optional[float] = None,
    p0_scatter: float = 0.05,
    progress: bool = True,
    seed: Optional[int] = None,
) -> InverseResult:
    """Run MCMC inference: observed magnitudes → posterior on stellar parameters.

    Parameters
    ----------
    obs_magnitudes : array-like, shape (n_filters,)
        Observed magnitudes, one per filter.
    obs_uncertainties : array-like, shape (n_filters,)
        1-sigma magnitude uncertainties.  Filters with no measured
        uncertainty should be given a generous value (e.g. 0.1 mag)
        rather than zero.
    filter_names : list of str
        Names identifying which filter each magnitude belongs to.
        Must match the ``Filter.name`` attribute of filters in *filters*.
    R : float
        Stellar radius in cm.
    d : float
        Distance in cm.
    grid : AtmosphereGrid
        Loaded atmosphere grid.
    filters : list of Filter
        Loaded filters — must include all names in *filter_names*.
    mag_system : str
        Photometric system: 'Vega', 'AB', or 'ST'.
    interp_method : str
        SED interpolation: 'hermite' (default) or 'linear'.
    n_walkers : int
        Number of emcee ensemble walkers.  Must be even and ≥ 6.
    n_steps : int
        Total steps per walker including burn-in.
    n_burn : int
        Steps to discard as burn-in.
    n_thin : int
        Thinning factor applied to the post-burn chain.
    p0_teff, p0_logg, p0_meta : float or None
        Starting point for the walker ball.  Defaults to grid centre.
    p0_scatter : float
        Fractional scatter of the initial ball relative to the grid range.
        Default 0.05 (5 %).
    progress : bool
        Show emcee progress bar.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    InverseResult
    """
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "emcee is required for the inverse model. "
            "Install it with: pip install emcee"
        ) from exc

    obs_mag = np.asarray(obs_magnitudes, dtype=np.float64)
    obs_err = np.asarray(obs_uncertainties, dtype=np.float64)

    if obs_mag.shape != obs_err.shape:
        raise ValueError("obs_magnitudes and obs_uncertainties must have the same shape.")
    if len(filter_names) != len(obs_mag):
        raise ValueError(
            f"filter_names has {len(filter_names)} entries but "
            f"obs_magnitudes has {len(obs_mag)}."
        )
    if np.any(obs_err <= 0):
        raise ValueError("All obs_uncertainties must be positive.")

    # Validate that every requested filter name exists in the filters list
    filter_map = {f.name: f for f in filters}
    missing = [n for n in filter_names if n not in filter_map]
    if missing:
        raise ValueError(
            f"Filter names {missing} are in filter_names but not in the "
            f"supplied filters list."
        )

    # Only pass through the filters needed for inference
    active_filters = [filter_map[n] for n in filter_names]

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Initial positions
    # ------------------------------------------------------------------
    pos = _initial_positions(
        n_walkers=n_walkers,
        grid=grid,
        p0_teff=p0_teff,
        p0_logg=p0_logg,
        p0_meta=p0_meta,
        scatter=p0_scatter,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # Sampler
    # ------------------------------------------------------------------
    n_dim = 3
    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        _log_posterior,
        args=(
            obs_mag, obs_err, filter_names,
            float(R), float(d),
            grid, active_filters, mag_system, interp_method,
        ),
    )

    sampler.run_mcmc(pos, n_steps, progress=progress)

    # ------------------------------------------------------------------
    # Flatten chain (discard burn-in, apply thinning)
    # ------------------------------------------------------------------
    samples  = sampler.get_chain(discard=n_burn, thin=n_thin, flat=True)
    log_prob = sampler.get_log_prob(discard=n_burn, thin=n_thin, flat=True)
    accept   = sampler.acceptance_fraction  # shape (n_walkers,)

    # ------------------------------------------------------------------
    # Autocorrelation time (best-effort — warns if chain is too short)
    # ------------------------------------------------------------------
    try:
        autocorr = sampler.get_autocorr_time(discard=n_burn, quiet=True)
    except emcee.autocorr.AutocorrError:
        autocorr = None
        warnings.warn(
            "Autocorrelation time estimation failed — the chain may be too short. "
            "Consider increasing n_steps.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Acceptance fraction sanity check
    # ------------------------------------------------------------------
    mean_af = float(np.mean(accept))
    if mean_af < 0.1:
        warnings.warn(
            f"Mean acceptance fraction is very low ({mean_af:.3f}). "
            "The sampler may not have converged. "
            "Try increasing n_walkers or adjusting p0_scatter.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif mean_af > 0.9:
        warnings.warn(
            f"Mean acceptance fraction is very high ({mean_af:.3f}). "
            "The likelihood may be very broad or the prior is dominating.",
            RuntimeWarning,
            stacklevel=2,
        )

    return InverseResult(
        samples=samples,
        log_prob=log_prob,
        filter_names=list(filter_names),
        obs_magnitudes=obs_mag,
        obs_uncertainties=obs_err,
        R=float(R),
        d=float(d),
        mag_system=mag_system,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        n_thin=n_thin,
        acceptance_fraction=accept,
        autocorr_time=autocorr,
    )
