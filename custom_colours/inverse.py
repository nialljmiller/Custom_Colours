"""
custom_colours.inverse
======================
Inverse model: observed magnitudes → posterior on stellar parameters.

The inverse model is the mirror of the forward model.  Both share the same
``FitParams`` object (from ``custom_colours.params``) which declares each
physical parameter as fixed or free.  The MCMC samples only the free
parameters; fixed ones are threaded to the forward model unchanged.

This bidirectionality is concrete, not cosmetic:
  - The forward model maps ``FitParams.unpack(theta)`` → magnitudes.
  - The inverse model maps magnitudes → ``FitParams`` + posterior on theta.
  - The same ``FitParams`` instance flows in both directions.

Parameters
----------
Every physical quantity — Teff, logg, [M/H], Av, distance — is described
by a ``ParamSpec`` inside a ``FitParams``.  Three behaviours:

``fixed(value)``
    Not sampled.  Passed unchanged to the forward model at every step.

``free(lo, hi)``
    Sampled by MCMC with a flat prior over [lo, hi].

The simplest use::

    from custom_colours import run_inverse
    from custom_colours.params import fit_params_from_grid, PC_TO_CM

    params = fit_params_from_grid(grid)                # Teff/logg/meta free
    result = run_inverse(obs_mags, obs_errs, filter_names,
                         R=R_sun, fit_params=params,
                         grid=grid, filters=filters)

To fit with extinction free::

    params = fit_params_from_grid(grid, a_v=(0.0, 3.0))   # Av free 0–3 mag
    ext    = ExtinctionModel(enabled=True, law='fitzpatrick99', a_v=0.0)
    result = run_inverse(..., fit_params=params, extinction=ext)

To fit with distance free (expect Av–d degeneracy in the posterior)::

    params = fit_params_from_grid(
        grid,
        a_v=(0.0, 2.0),
        d_cm=(100*PC_TO_CM, 2000*PC_TO_CM),
    )
"""

from __future__ import annotations

import warnings
from typing import Optional, TYPE_CHECKING

import numpy as np

from .grid import AtmosphereGrid
from .filters import Filter
from .forward import run_forward
from .params import FitParams, fit_params_from_grid, PC_TO_CM
from .io import InverseResult

if TYPE_CHECKING:
    from sed_extinction import ExtinctionModel


# ---------------------------------------------------------------------------
# Prior
# ---------------------------------------------------------------------------

def _log_prior(theta: np.ndarray, fit_params: FitParams) -> float:
    """Flat prior within bounds for all free parameters; -inf outside."""
    return 0.0 if fit_params.in_prior(theta) else -np.inf


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def _log_likelihood(
    theta: np.ndarray,
    obs_mag: np.ndarray,
    obs_err: np.ndarray,
    filter_order: list,
    R: float,
    fit_params: FitParams,
    grid: AtmosphereGrid,
    filters: list,
    mag_system: str,
    interp_method: str,
    extinction: Optional["ExtinctionModel"],
) -> float:
    """Gaussian log-likelihood summed over filters.

    Calls ``run_forward`` in FitParams mode so both directions share
    the same parameter unpacking logic.
    """
    try:
        result = run_forward(
            fit_params=fit_params,
            theta=theta,
            R=R,
            grid=grid,
            filters=filters,
            mag_system=mag_system,
            interp_method=interp_method,
            extinction=extinction,
        )
    except Exception:
        return -np.inf

    ll = 0.0
    for i, fname in enumerate(filter_order):
        m_model = result.magnitudes.get(fname)
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
    obs_mag, obs_err, filter_order,
    R, fit_params, grid, filters,
    mag_system, interp_method, extinction,
) -> float:
    lp = _log_prior(theta, fit_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(
        theta, obs_mag, obs_err, filter_order,
        R, fit_params, grid, filters,
        mag_system, interp_method, extinction,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inverse(
    obs_magnitudes:    list,
    obs_uncertainties: list,
    filter_names:      list,
    R:                 float,
    grid:              AtmosphereGrid,
    filters:           list,
    # FitParams — the shared parameter spec
    fit_params:        Optional[FitParams] = None,
    # convenience: if fit_params not supplied, these are used to build one
    d:                 Optional[float] = None,
    extinction_law:    Optional[str]   = None,
    # extinction model (required if a_v is free or if you want reddening)
    extinction:        Optional["ExtinctionModel"] = None,
    # sampler settings
    mag_system:        str   = "AB",
    interp_method:     str   = "hermite",
    n_walkers:         int   = 32,
    n_steps:           int   = 2000,
    n_burn:            int   = 500,
    n_thin:            int   = 1,
    p0_centre:         Optional[dict] = None,
    p0_teff:           Optional[float] = None,
    p0_logg:           Optional[float] = None,
    p0_meta:           Optional[float] = None,
    p0_scatter:        float = 0.02,
    seed:              Optional[int]  = None,
    progress:          bool  = True,
) -> InverseResult:
    """Infer stellar parameters from observed broadband photometry.

    Parameters
    ----------
    obs_magnitudes : array-like, shape (n_filters,)
        Observed magnitudes in the same order as ``filter_names``.
    obs_uncertainties : array-like, shape (n_filters,)
        1-sigma uncertainties.  Must all be > 0.
    filter_names : list of str
        Filter names matching elements of ``filters``.
    R : float
        Stellar radius in cm (always required; not currently sampled).
    grid : AtmosphereGrid
    filters : list of Filter
    fit_params : FitParams or None
        Full parameter specification.  If None, one is built from ``grid``
        using the ``d`` convenience argument (default: 1 pc, fixed).
    d : float or None
        Distance in cm.  Used only when ``fit_params`` is None to build a
        default ``FitParams`` with fixed distance.  If ``fit_params`` is
        supplied this argument is ignored — set distance via fit_params.
    extinction : ExtinctionModel or None
        Dust extinction model.  Required if Av is free in fit_params.
        Ignored if Av is fixed at 0.
    mag_system : {'AB', 'Vega', 'ST'}
    interp_method : {'hermite', 'linear'}
    n_walkers, n_steps, n_burn, n_thin : int
        emcee settings.
    p0_centre : dict or None
        Optional starting point keyed by parameter name, e.g.
        ``{'teff': 5800, 'logg': 4.4, 'meta': 0.0}``.
    p0_scatter : float
        Width of initial ball as fraction of each parameter's range.
    seed : int or None
    progress : bool

    Returns
    -------
    InverseResult
    """
    try:
        import emcee
    except ImportError as exc:
        raise ImportError(
            "emcee is required for the inverse model. "
            "Install with: pip install emcee"
        ) from exc

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    obs_mag = np.asarray(obs_magnitudes,   dtype=np.float64)
    obs_err = np.asarray(obs_uncertainties, dtype=np.float64)

    if obs_mag.shape != obs_err.shape:
        raise ValueError(
            "obs_magnitudes and obs_uncertainties must have the same length."
        )
    if len(filter_names) != len(obs_mag):
        raise ValueError(
            "filter_names must have the same length as obs_magnitudes."
        )
    if not np.all(obs_err > 0):
        raise ValueError(
            "All obs_uncertainties must be positive (> 0). "
            "Zero or negative uncertainties make the likelihood undefined."
        )
    if n_walkers < 6:
        raise ValueError("n_walkers must be >= 6 (2 × n_dim).")
    if n_burn >= n_steps:
        raise ValueError("n_burn must be < n_steps.")

    available = {f.name for f in filters}
    missing   = set(filter_names) - available
    if missing:
        raise ValueError(
            f"Filter names not in the supplied filters: {missing}. "
            f"Available: {available}"
        )

    # ------------------------------------------------------------------
    # Build FitParams if not supplied
    # ------------------------------------------------------------------
    if fit_params is None:
        d_cm = d if d is not None else PC_TO_CM
        fit_params = fit_params_from_grid(grid, d_cm=d_cm)

    n_dim = fit_params.n_free
    if n_dim == 0:
        raise ValueError(
            "All parameters are fixed — nothing to sample. "
            "Make at least one parameter free in fit_params."
        )
    if n_walkers < 2 * n_dim:
        raise ValueError(
            f"n_walkers ({n_walkers}) must be >= 2 × n_free ({2*n_dim}). "
            f"Free parameters: {fit_params.free_names}"
        )

    # ------------------------------------------------------------------
    # Log what is being sampled vs fixed
    # ------------------------------------------------------------------
    print(fit_params.summary())
    if extinction is not None and getattr(extinction.config, 'enabled', False):
        cfg = extinction.config
        av_mode = "free (sampled)" if fit_params.a_v.is_free else f"fixed={cfg.a_v:.3f}"
        print(
            f"[inverse] Extinction: law={cfg.law}, Rv={cfg.r_v:.2f}, "
            f"Av {av_mode}"
        )
    else:
        print("[inverse] Extinction: disabled")

    # ------------------------------------------------------------------
    # Initial walker positions
    # Merge individual p0_teff/logg/meta kwargs into p0_centre dict.
    # p0_centre takes precedence; individual kwargs are a convenience alias.
    # ------------------------------------------------------------------
    centre: dict = {}
    if p0_teff is not None: centre['teff'] = p0_teff
    if p0_logg is not None: centre['logg'] = p0_logg
    if p0_meta is not None: centre['meta'] = p0_meta
    if p0_centre:
        centre.update(p0_centre)   # p0_centre wins on conflict

    rng = np.random.default_rng(seed)
    p0  = fit_params.initial_ball(
        n_walkers, centre=centre or None, scatter=p0_scatter, rng=rng
    )

    # ------------------------------------------------------------------
    # Run MCMC
    # ------------------------------------------------------------------
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, _log_posterior,
        args=(
            obs_mag, obs_err, list(filter_names),
            R, fit_params, grid, filters,
            mag_system, interp_method, extinction,
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
            UserWarning, stacklevel=2,
        )

    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = None

    flat_samples  = sampler.get_chain(discard=n_burn, thin=n_thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=n_burn, thin=n_thin, flat=True)

    return InverseResult(
        samples=flat_samples,
        log_prob=flat_log_prob,
        filter_names=list(filter_names),
        obs_magnitudes=obs_mag,
        obs_uncertainties=obs_err,
        R=float(R),
        d=float(fit_params.d.value if fit_params.d.is_fixed else np.nan),
        mag_system=mag_system,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        n_thin=n_thin,
        acceptance_fraction=acc,
        autocorr_time=tau,
    )
