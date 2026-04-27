"""
demo_param_modes.py
===================
Demonstrates the three parameter modes available in the Custom_Colours
inverse model: fixed, bounded (free with limits), and fully open (free
with grid-wide bounds).

The four scenarios below all fit the same synthetic observations of a
solar-type star at 500 pc.  What changes is which parameters are sampled
and with what constraints.

Scenario A — Everything open
    Teff, logg, [M/H] free across the full grid.
    Av fixed at 0 (no extinction).
    Distance fixed at the true value.

Scenario B — Teff tightly bounded, logg fixed, [M/H] open
    Teff free but only within ±500 K of a prior guess.
    logg fixed at the catalogue value.
    [M/H] free across the full grid.

Scenario C — All three atmospheric params bounded, Av free
    Teff, logg, [M/H] each free within a physically-motivated window.
    Av free from 0 to 2 mag — Av and Teff will be correlated.
    Distance fixed.

Scenario D — Everything free including distance
    Teff, logg, [M/H], Av, distance all sampled simultaneously.
    This is the most degenerate case; the posterior will show the
    Av–distance correlation explicitly.

Usage
-----
    python demo_param_modes.py

Requirements: custom_colours built (make), emcee, sed_extinction.py on path.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from custom_colours import (
    load_grid, load_filters_from_instrument_dir,
    run_forward, run_inverse,
)
from custom_colours.params import (
    FitParams, ParamSpec, fit_params_from_grid,
    fixed, free, PC_TO_CM,
)

# =============================================================================
# DATA PATHS  —  edit these
# =============================================================================

GRID_DIR   = os.path.expanduser("~/SED_Tools/data/stellar_models/Kurucz2003all")
FILTER_DIR = os.path.expanduser("~/SED_Tools/data/filters/Generic/Johnson")
VEGA_SED   = os.path.expanduser("~/SED_Tools/data/stellar_models/vega_flam.csv")

# =============================================================================
# TRUE PARAMETERS (used to generate synthetic observations)
# =============================================================================

TRUE_TEFF  = 5800.0
TRUE_LOGG  = 4.40
TRUE_META  = 0.0
TRUE_AV    = 0.3          # injected extinction (Fitzpatrick99)
TRUE_D_PC  = 500.0        # pc
TRUE_R     = 6.957e10     # cm (1 R_sun)
OBS_SIGMA  = 0.02         # mag noise per band
MAG_SYSTEM = "AB"

TRUE_D_CM = TRUE_D_PC * PC_TO_CM

# =============================================================================
# MCMC SETTINGS  —  short for demo; increase for production
# =============================================================================

N_WALKERS = 32
N_STEPS   = 800
N_BURN    = 200
N_THIN    = 1

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading atmosphere grid ...")
grid = load_grid(GRID_DIR)
print(f"  {grid}")

print("\nLoading filters ...")
vega_path = VEGA_SED if os.path.isfile(VEGA_SED) else None
filters = load_filters_from_instrument_dir(FILTER_DIR, vega_sed_path=vega_path)
filter_names = [f.name for f in filters]
print(f"  {len(filters)} filters: {filter_names}")

# =============================================================================
# GENERATE SYNTHETIC OBSERVATIONS
# =============================================================================

try:
    from custom_colours import make_extinction_model
    true_ext = make_extinction_model(
        enabled=True, law="fitzpatrick99", a_v=TRUE_AV, r_v=3.1
    )
except ImportError:
    true_ext = None
    print("  Extinction not available — generating without extinction (Av=0)")

print(f"\nSynthesising observations ...")
print(f"  Teff={TRUE_TEFF:.0f} K  logg={TRUE_LOGG:.2f}  [M/H]={TRUE_META:.2f}")
print(f"  Av={TRUE_AV:.2f}  d={TRUE_D_PC:.0f} pc  noise={OBS_SIGMA:.3f} mag/band")

fwd_truth = run_forward(
    teff=TRUE_TEFF, logg=TRUE_LOGG, meta=TRUE_META,
    R=TRUE_R, d=TRUE_D_CM,
    grid=grid, filters=filters,
    mag_system=MAG_SYSTEM,
    extinction=true_ext,
)

rng = np.random.default_rng(0)
obs_mags = (np.array([fwd_truth.magnitudes[n] for n in filter_names])
            + rng.normal(0.0, OBS_SIGMA, len(filter_names)))
obs_errs = np.full(len(filter_names), OBS_SIGMA)

print(f"  Band    truth     observed")
for n, tm, om in zip(filter_names, [fwd_truth.magnitudes[n] for n in filter_names], obs_mags):
    print(f"  {n:>4s}   {tm:7.4f}   {om:7.4f}")

# =============================================================================
# DEFINE THE FOUR SCENARIOS
# =============================================================================
# Each entry is (label, description, FitParams, extinction_model_or_None)

def _ext(av):
    """Return an ExtinctionModel with Fitzpatrick99 at the given Av, or None."""
    from custom_colours import make_extinction_model
    return make_extinction_model(enabled=True, law="fitzpatrick99",
                                 a_v=av, r_v=3.1)


SCENARIOS = [

    # ------------------------------------------------------------------
    # A: Everything open — widest possible posteriors
    # ------------------------------------------------------------------
    (
        "A: All open",
        "Teff/logg/[M/H] free across full grid\n"
        "Av fixed=0, d fixed at true value",
        fit_params_from_grid(
            grid,
            a_v=0.0,              # fixed, no extinction
            d_cm=TRUE_D_CM,       # fixed at true distance
        ),
        None,                     # no extinction model
    ),

    # ------------------------------------------------------------------
    # B: Teff bounded, logg fixed from catalogue, [M/H] open
    # ------------------------------------------------------------------
    (
        "B: Teff bounded, logg fixed",
        "Teff free within ±500 K of prior guess\n"
        "logg fixed=4.4, [M/H] free, Av=0 fixed",
        FitParams(
            teff = ParamSpec('teff', 'free',
                             lo=TRUE_TEFF - 500, hi=TRUE_TEFF + 500),
            logg = ParamSpec('logg', 'fixed', value=4.4),
            meta = ParamSpec('meta', 'free',
                             lo=grid.meta_bounds[0],
                             hi=grid.meta_bounds[1]),
            a_v  = ParamSpec('a_v',  'fixed', value=0.0),
            d    = ParamSpec('d',    'fixed', value=TRUE_D_CM),
        ),
        None,
    ),

    # ------------------------------------------------------------------
    # C: All three atm params bounded + Av free (coupled degeneracy)
    # ------------------------------------------------------------------
    (
        "C: All bounded + Av free",
        "Teff free [4500,7000 K], logg free [3.5,5.0]\n"
        "[M/H] free [-0.5,0.5], Av free [0,2], d fixed",
        FitParams(
            teff = ParamSpec('teff', 'free', lo=4500.0, hi=7000.0),
            logg = ParamSpec('logg', 'free', lo=3.5,    hi=5.0),
            meta = ParamSpec('meta', 'free', lo=-0.5,   hi=0.5),
            a_v  = ParamSpec('a_v',  'free', lo=0.0,    hi=2.0),
            d    = ParamSpec('d',    'fixed', value=TRUE_D_CM),
        ),
        _ext(0.0),   # enabled but a_v overridden at each step from theta
    ),

    # ------------------------------------------------------------------
    # D: Everything free — full degeneracy on display
    # ------------------------------------------------------------------
    (
        "D: Everything free",
        "Teff/logg/[M/H] free across full grid\n"
        "Av free [0,2], d free [200,1000 pc]\n"
        "Expect Av–d posterior correlation",
        FitParams(
            teff = ParamSpec('teff', 'free',
                             lo=grid.teff_bounds[0],
                             hi=grid.teff_bounds[1]),
            logg = ParamSpec('logg', 'free',
                             lo=grid.logg_bounds[0],
                             hi=grid.logg_bounds[1]),
            meta = ParamSpec('meta', 'free',
                             lo=grid.meta_bounds[0],
                             hi=grid.meta_bounds[1]),
            a_v  = ParamSpec('a_v',  'free', lo=0.0, hi=2.0),
            d    = ParamSpec('d',    'free',
                             lo=200.0 * PC_TO_CM,
                             hi=1000.0 * PC_TO_CM),
        ),
        _ext(0.0),
    ),
]

# =============================================================================
# RUN ALL SCENARIOS
# =============================================================================

results = {}

for label, description, fit_params, ext_model in SCENARIOS:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  {description}")
    print(f"{'='*65}")

    # Starting point — always near the truth for all free params
    p0_centre = {}
    if fit_params.teff.is_free: p0_centre['teff'] = TRUE_TEFF
    if fit_params.logg.is_free: p0_centre['logg'] = TRUE_LOGG
    if fit_params.meta.is_free: p0_centre['meta'] = TRUE_META
    if fit_params.a_v.is_free:  p0_centre['a_v']  = TRUE_AV
    if fit_params.d.is_free:    p0_centre['d']     = TRUE_D_CM

    n_free = fit_params.n_free
    n_walkers = max(N_WALKERS, 2 * n_free + 2)
    # ensure even
    if n_walkers % 2 != 0:
        n_walkers += 1

    posterior = run_inverse(
        obs_magnitudes=obs_mags,
        obs_uncertainties=obs_errs,
        filter_names=filter_names,
        R=TRUE_R,
        grid=grid,
        filters=filters,
        fit_params=fit_params,
        extinction=ext_model,
        mag_system=MAG_SYSTEM,
        n_walkers=n_walkers,
        n_steps=N_STEPS,
        n_burn=N_BURN,
        n_thin=N_THIN,
        p0_centre=p0_centre,
        p0_scatter=0.02,
        progress=True,
        seed=42,
    )

    samples = posterior.samples
    free_names = fit_params.free_names

    print(f"\n  {'Parameter':<12}  {'Median':>10}  {'−1σ':>8}  {'+1σ':>8}  {'True':>10}")
    print(f"  {'-'*55}")

    truths = dict(teff=TRUE_TEFF, logg=TRUE_LOGG, meta=TRUE_META,
                  a_v=TRUE_AV,   d=TRUE_D_CM)

    for i, pname in enumerate(free_names):
        col = samples[:, i]
        med = float(np.median(col))
        lo  = float(np.percentile(col, 15.865))
        hi  = float(np.percentile(col, 84.135))
        tv  = truths[pname]
        # distance: convert to pc for display
        if pname == 'd':
            scale, unit = 1.0 / PC_TO_CM, "pc"
        else:
            scale, unit = 1.0, ""
        print(f"  {pname:<12}  {med*scale:10.3f}  "
              f"{(med-lo)*scale:8.3f}  {(hi-med)*scale:8.3f}  "
              f"{tv*scale:10.3f}  {unit}")

    results[label] = dict(
        posterior=posterior,
        fit_params=fit_params,
        free_names=free_names,
        truths=truths,
    )

# =============================================================================
# PLOT — one row per scenario, showing 1D marginals for free parameters
# =============================================================================

n_scenarios = len(SCENARIOS)
# Maximum number of free parameters across all scenarios (for column count)
max_free = max(r['fit_params'].n_free for r in results.values())

fig = plt.figure(figsize=(max(10, max_free * 2.5), n_scenarios * 2.8))
gs  = gridspec.GridSpec(n_scenarios, max_free,
                        hspace=0.6, wspace=0.4)

PARAM_LABELS = {
    'teff': r'$T_{\rm eff}$ (K)',
    'logg': r'$\log g$',
    'meta': r'$[\rm M/H]$',
    'a_v':  r'$A_V$ (mag)',
    'd':    r'$d$ (pc)',
}
PARAM_SCALE = {
    'teff': 1.0,
    'logg': 1.0,
    'meta': 1.0,
    'a_v':  1.0,
    'd':    1.0 / PC_TO_CM,
}
TRUE_VALUES = dict(teff=TRUE_TEFF, logg=TRUE_LOGG, meta=TRUE_META,
                   a_v=TRUE_AV, d=TRUE_D_PC)   # d already in pc here

for row_idx, (label, description, fit_params, _) in enumerate(SCENARIOS):
    r = results[label]
    samples   = r['posterior'].samples
    free_names = r['free_names']

    for col_idx, pname in enumerate(free_names):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        col = samples[:, col_idx] * PARAM_SCALE[pname]
        tv  = TRUE_VALUES[pname]

        ax.hist(col, bins=35, color="steelblue", alpha=0.75,
                density=True, edgecolor="none")
        med = float(np.median(col))
        lo  = float(np.percentile(col, 15.865))
        hi  = float(np.percentile(col, 84.135))
        ax.axvline(med, color="navy",   lw=1.5)
        ax.axvline(lo,  color="navy",   lw=0.8, ls="--")
        ax.axvline(hi,  color="navy",   lw=0.8, ls="--")
        ax.axvline(tv,  color="crimson", lw=1.5, ls="-", label="true")

        ax.set_xlabel(PARAM_LABELS[pname], fontsize=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

        # Show the prior window as a shaded region for bounded params
        spec = getattr(fit_params, pname)
        if spec.is_free:
            lo_prior = spec.lo * PARAM_SCALE[pname]
            hi_prior = spec.hi * PARAM_SCALE[pname]
            ax.axvspan(lo_prior, hi_prior, alpha=0.07, color="goldenrod",
                       label="prior window")

        if col_idx == 0:
            ax.set_ylabel(label.split(":")[0], fontsize=9, fontweight="bold")

    # blank out unused columns
    for col_idx in range(len(free_names), max_free):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.set_visible(False)

fig.suptitle(
    "Parameter mode comparison\n"
    rf"True: $T_{{\rm eff}}$={TRUE_TEFF:.0f} K, $\log g$={TRUE_LOGG:.2f}, "
    rf"$[\rm M/H]$={TRUE_META:.2f}, $A_V$={TRUE_AV:.2f}, $d$={TRUE_D_PC:.0f} pc",
    fontsize=11, y=1.01,
)

out = "param_modes_demo.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {out}")
plt.show()
