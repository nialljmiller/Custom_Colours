"""
custom_colours.inverse
======================
Inverse model: observed magnitudes → posterior on (Teff, logg, [M/H]).

Uses emcee (MCMC) with a Gaussian likelihood on residuals between
observed and model magnitudes.  The forward model is called at every
walker step, so the Fortran kernels must be built.

Extinction
----------
Pass an ``ExtinctionModel`` from ``sed_extinction`` to the ``extinction``
keyword.  It is a **fixed parameter** — not sampled by the MCMC.  The
same model instance is forwarded to every ``run_forward`` call inside
the likelihood, so the reddening curve and Av are held constant while
Teff, logg, and [M/H] are inferred.

Extinction is disabled by default.  Enable it like this:

    from sed_extinction import ExtinctionModel
    ext = ExtinctionModel(enabled=True, law='fitzpatrick99',
                          a_v=0.3, r_v=3.1)
    posterior = run_inverse(..., extinction=ext)

Distance
--------
Distance is also a fixed parameter passed as ``d`` in cm — not sampled.
Set it to the known or estimated distance before calling run_inverse.
Default (if you omit it) remains 10 pc = 3.086e19 cm as before.

Public API
----------
run_inverse(obs_magnitudes, obs_uncertainties, filter_names,
            R, d, grid, filters,
            extinction=None,           # fixed, not fitted
            ...)
    → InverseResult
"""

from __future__ import annotations

import warnings
from typing import Optional, TYPE_CHECKING

import numpy as np

from .grid import AtmosphereGrid
from .filters import Filter
from .forward import run_forward
from .io import InverseResult

if TYPE_CHECKING:
    from sed_extinction import ExtinctionModel


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

def _log_prior(theta: np.ndarray, grid: AtmosphereGrid) -> float:
    """Flat prior within grid bounds, -inf outside."""
    teff, logg, meta = theta
    if (grid.teff_bounds[0] <= teff <= grid.teff_bounds[1]
            and grid.logg_bounds[0] <= logg <= grid.logg_bounds[1]
            and grid.meta_bounds[0] <= meta <= grid.meta_bounds[1]):
        return 0.0
    return -np.inf


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def _log_likelihood(
    theta: np.ndarray,
    obs_mag: np.ndarray,
    obs_err: np.ndarray,
    filter_order: list,
    R: float,
    d: float,
    grid: AtmosphereGrid,
    filters: list,
    mag_system: str,
    interp_method: str,
    extinction: Optional["ExtinctionModel"],
) -> float:
    """Gaussian log-likelihood summed over all filters.

    L = -0.5 × Σ [ (m_obs - m_model)² / σ² + ln(2π σ²) ]

    Extinction is forwarded to the forward model unchanged — it is
    applied inside run_forward after dilution and before convolution.
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
            extinction=extinction,          # <-- passed straight through
        )
    except Exception:
        return -np.inf

    ll = 0.0
    for i, fname in enumerate(filter_order):
        m_model = result.magnitudes.get(fname, None)
        if m_model is None or not np.isfinite(m_model):
            return -np.inf
        sigma2 = obs_err[i] ** 2
        ll += -0.5 * ((obs_mag[i] - m_model) ** 2 / sigma2
                      + np.log(2.0 * np.pi * sigma2))
    return ll


# ---------------------------------------------------------------------------
# Posterior
# ---------------------------------------------------------------------------

def _log_posterior(
    theta: np.ndarray,
    obs_mag: np.ndarray,
    obs_err: np.ndarray,
    filter_order: list,
    R: float,
    d: float,
    grid: AtmosphereGrid,
    filters: list,
    mag_system: str,
    interp_method: str,
    extinction: Optional["ExtinctionModel"],
) -> float:
    lp = _log_prior(theta, grid)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(
        theta, obs_mag, obs_err, filter_order,
        R, d, grid, filters, mag_system, interp_method, extinction,
    )


# ---------------------------------------------------------------------------
# Initial positions
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
        p0_teff if p0_teff is not None else 0.5 * (grid.teff_bounds[0] + grid.teff_bounds[1]),
        p0_logg if p0_logg is not None else 0.5 * (grid.logg_bounds[0] + grid.logg_bounds[1]),
        p0_meta if p0_meta is not None else 0.5 * (grid.meta_bounds[0] + grid.meta_bounds[1]),
    ])

    # Characteristic scale for the scatter ball
    scale = np.array([
        scatter * (grid.teff_bounds[1] - grid.teff_bounds[0]),
        scatter * (grid.logg_bounds[1] - grid.logg_bounds[0]),
        scatter * (grid.meta_bounds[1] - grid.meta_bounds[0]),
    ])

    pos = centre + scale * rng.standard_normal((n_walkers, 3))

    # Clamp to grid bounds
    lo = np.array([grid.teff_bounds[0], grid.logg_bounds[0], grid.meta_bounds[0]])
    hi = np.array([grid.teff_bounds[1], grid.logg_bounds[1], grid.meta_bounds[1]])
    pos = np.clip(pos, lo, hi)

    return pos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inverse(
    obs_magnitudes: list,
    obs_uncertainties: list,
    filter_names: list,
    R: float,
    d: float,
    grid: AtmosphereGrid,
    filters: list,
    mag_system: str = "AB",
    interp_method: str = "hermite",
    extinction: Optional["ExtinctionModel"] = None,
    n_walkers: int = 32,
    n_steps: int = 2000,
    n_burn: int = 500,
    n_thin: int = 1,
    p0_teff: Optional[float] = None,
    p0_logg: Optional[float] = None,
    p0_meta: Optional[float] = None,
    p0_scatter: float = 0.02,
    seed: Optional[int] = None,
    progress: bool = True,
) -> InverseResult:
    """Infer (Teff, logg, [M/H]) from observed broadband photometry.

    Parameters
    ----------
    obs_magnitudes : array-like, shape (n_filters,)
        Observed magnitudes, one per filter, in the same order as
        ``filter_names``.
    obs_uncertainties : array-like, shape (n_filters,)
        1-sigma magnitude uncertainties.
    filter_names : list of str
        Filter names in the same order as ``obs_magnitudes``.  Each
        name must match ``filt.name`` for some element of ``filters``.
    R : float
        Stellar radius in cm (fixed — not sampled).
    d : float
        Distance in cm (fixed — not sampled).
    grid : AtmosphereGrid
        Loaded atmosphere grid.
    filters : list of Filter
        Filter curves.  Must include every name in ``filter_names``.
    mag_system : {'AB', 'Vega', 'ST'}
        Photometric system.
    interp_method : {'hermite', 'linear'}
        Grid interpolation method.
    extinction : ExtinctionModel or None
        Fixed extinction to apply at every likelihood evaluation.
        Not sampled.  Default None (no extinction).
    n_walkers : int
        Number of emcee walkers (must be even, >= 2 × n_params = 6).
    n_steps : int
        Total steps per walker including burn-in.
    n_burn : int
        Steps to discard as burn-in (must be < n_steps).
    n_thin : int
        Thinning factor.
    p0_teff, p0_logg, p0_meta : float or None
        Starting point for the walker ball.  Defaults to grid centre.
    p0_scatter : float
        Width of the initial ball as a fraction of each parameter range.
    seed : int or None
        Random seed for reproducibility.
    progress : bool
        Show emcee progress bar.

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

    obs_mag = np.asarray(obs_magnitudes,  dtype=np.float64)
    obs_err = np.asarray(obs_uncertainties, dtype=np.float64)

    if obs_mag.shape != obs_err.shape:
        raise ValueError("obs_magnitudes and obs_uncertainties must have the same length.")
    if len(filter_names) != len(obs_mag):
        raise ValueError("filter_names must have the same length as obs_magnitudes.")
    if n_walkers < 6:
        raise ValueError("n_walkers must be >= 6 (2 × n_dim).")
    if n_burn >= n_steps:
        raise ValueError("n_burn must be < n_steps.")

    # Validate that all requested filter names exist in the filter list
    available = {f.name for f in filters}
    missing   = set(filter_names) - available
    if missing:
        raise ValueError(
            f"not in the supplied filters: {missing}. "
            f"Available: {available}"
        )

    rng    = np.random.default_rng(seed)
    n_dim  = 3
    p0     = _initial_positions(n_walkers, grid,
                                 p0_teff, p0_logg, p0_meta,
                                 p0_scatter, rng)

    # Log a note about extinction so it's clear in the user's output
    if extinction is not None and getattr(extinction.config, 'enabled', False):
        cfg = extinction.config
        print(
            f"[inverse] Extinction enabled: law={cfg.law}, "
            f"Av={cfg.a_v:.3f}, Rv={cfg.r_v:.2f} "
            f"(fixed — not sampled)"
        )
    else:
        print("[inverse] Extinction disabled (Av=0 fixed)")

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        _log_posterior,
        args=(
            obs_mag, obs_err, list(filter_names),
            R, d, grid, filters,
            mag_system, interp_method,
            extinction,                     # fixed, forwarded to every step
        ),
    )

    sampler.run_mcmc(p0, n_steps, progress=progress)

    # Acceptance fraction diagnostic
    acc = sampler.acceptance_fraction
    if np.any(acc < 0.1) or np.any(acc > 0.9):
        warnings.warn(
            f"Some walkers have extreme acceptance fractions "
            f"(min={acc.min():.2f}, max={acc.max():.2f}). "
            f"Consider adjusting n_walkers or p0_scatter.",
            UserWarning,
            stacklevel=2,
        )

    # Autocorrelation time (best-effort — short chains will raise)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = None

    # Flatten chain, discarding burn-in and applying thinning
    flat_samples  = sampler.get_chain(discard=n_burn, thin=n_thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=n_burn, thin=n_thin, flat=True)

    return InverseResult(
        samples=flat_samples,
        log_prob=flat_log_prob,
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
        acceptance_fraction=acc,
        autocorr_time=tau,
    )