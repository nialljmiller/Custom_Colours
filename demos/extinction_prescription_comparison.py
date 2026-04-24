"""
extinction_prescription_comparison.py
======================================
Demonstrates how different extinction laws affect the inverse fit.

Scenario
--------
We have a star at 500 pc with genuine Av=0.8 reddening.  We generate
synthetic observed magnitudes from the forward model using Fitzpatrick99
(the "truth"), then re-fit it six times — once per prescription — and
compare how well each one recovers the true Teff, logg, and [M/H].

This is also useful as a sensitivity test: if all six prescriptions
agree on the posterior, the photometry data is not constraining Av much
and the choice of law doesn't matter.  If they disagree, you need to
think about which is appropriate for your sight line.

Usage
-----
    python extinction_prescription_comparison.py

Requirements: custom_colours built (make), emcee, sed_extinction.py on path.
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# sed_extinction lives one level up (or wherever you keep it)
# ---------------------------------------------------------------------------
sys.path.insert(0, "..")   # adjust if needed
from sed_extinction import ExtinctionModel, make_extinction_model, AVAILABLE_LAWS

from custom_colours import (
    load_grid, load_filters,
    run_forward, run_inverse,
    InverseResult,
)

# ---------------------------------------------------------------------------
# Configuration — edit these for your setup
# ---------------------------------------------------------------------------

GRID_PATH    = "/path/to/Kurucz2003all/"          # your flux cube directory
FILTER_PATHS = [
    "/path/to/filters/GAIA/GAIA/G.dat",
    "/path/to/filters/GAIA/GAIA/Gbp.dat",
    "/path/to/filters/GAIA/GAIA/Grp.dat",
    "/path/to/filters/2MASS/J.dat",
    "/path/to/filters/2MASS/H.dat",
    "/path/to/filters/2MASS/Ks.dat",
]

# "True" stellar parameters used to generate synthetic photometry
TRUE_TEFF  = 5800.0    # K
TRUE_LOGG  = 4.40
TRUE_META  = 0.0

# Physical setup
R_SUN_CM   = 6.957e10              # cm
PC_TO_CM   = 3.085677581491367e18  # cm per parsec
TRUE_R     = 1.0 * R_SUN_CM       # 1 R_sun
TRUE_D     = 500.0 * PC_TO_CM     # 500 pc

# True extinction used to generate the synthetic observations
TRUE_AV    = 0.8
TRUE_RV    = 3.1
TRUE_LAW   = "fitzpatrick99"

# Noise added to synthetic magnitudes
OBS_SIGMA  = 0.02   # mag (1-sigma per band)

# MCMC settings — keep short for a quick demo; increase for publication
N_WALKERS  = 32
N_STEPS    = 1500
N_BURN     = 400
N_THIN     = 1

MAG_SYSTEM = "Vega"

# ---------------------------------------------------------------------------
# Prescriptions to compare
# ---------------------------------------------------------------------------
# Each entry: (label, ExtinctionModel kwargs)
# We assume Av is known from some prior (e.g. dust maps) and is fixed.
# The only thing that changes between runs is the law (and R_V where relevant).

PRESCRIPTIONS = [
    # --- disabled ---
    ("No extinction",
     dict(enabled=False)),

    # --- standard laws, Av fixed at true value ---
    ("CCM89  Rv=3.1",
     dict(enabled=True, law="ccm89",         a_v=TRUE_AV, r_v=3.1)),

    ("O'Donnell94  Rv=3.1",
     dict(enabled=True, law="odonnell94",    a_v=TRUE_AV, r_v=3.1)),

    ("Fitzpatrick99  Rv=3.1",
     dict(enabled=True, law="fitzpatrick99", a_v=TRUE_AV, r_v=3.1)),

    ("FM07  Rv=3.1 (fixed)",
     dict(enabled=True, law="fm07",          a_v=TRUE_AV)),

    ("Calzetti00  Rv=4.05",
     dict(enabled=True, law="calzetti00",    a_v=TRUE_AV, r_v=4.05)),

    # --- Gordon+2023 environments ---
    ("Gordon+2023  MW",
     dict(enabled=True, law="gordon23",      a_v=TRUE_AV, gordon23_env="mw")),

    ("Gordon+2023  LMC",
     dict(enabled=True, law="gordon23",      a_v=TRUE_AV, gordon23_env="lmc")),

    ("Gordon+2023  SMC",
     dict(enabled=True, law="gordon23",      a_v=TRUE_AV, gordon23_env="smc")),

    # --- non-standard R_V ---
    ("Fitzpatrick99  Rv=2.5",
     dict(enabled=True, law="fitzpatrick99", a_v=TRUE_AV, r_v=2.5)),

    ("Fitzpatrick99  Rv=5.0",
     dict(enabled=True, law="fitzpatrick99", a_v=TRUE_AV, r_v=5.0)),

    ("CCM89  Rv=5.0",
     dict(enabled=True, law="ccm89",         a_v=TRUE_AV, r_v=5.0)),
]

# ---------------------------------------------------------------------------
# Helper: summary stats from an InverseResult
# ---------------------------------------------------------------------------

def posterior_stats(result: InverseResult):
    """Return (median, lo_1sigma, hi_1sigma) for each of Teff, logg, meta."""
    s = result.samples
    med  = np.percentile(s, 50,   axis=0)
    lo   = np.percentile(s, 15.87, axis=0)
    hi   = np.percentile(s, 84.13, axis=0)
    return med, lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -----------------------------------------------------------------------
    # Load grid and filters once
    # -----------------------------------------------------------------------
    print("Loading atmosphere grid...")
    grid = load_grid(GRID_PATH)
    print(f"  Grid: Teff {grid.teff_bounds}, logg {grid.logg_bounds}, "
          f"[M/H] {grid.meta_bounds}")

    print("Loading filters...")
    filters = load_filters(FILTER_PATHS)
    filter_names = [f.name for f in filters]
    print(f"  Filters: {filter_names}")

    # -----------------------------------------------------------------------
    # Generate synthetic observations using the true law (Fitzpatrick99)
    # -----------------------------------------------------------------------
    print(f"\nGenerating synthetic observations:")
    print(f"  Teff={TRUE_TEFF} K  logg={TRUE_LOGG}  [M/H]={TRUE_META}")
    print(f"  R={TRUE_R:.3e} cm  d={TRUE_D:.3e} cm  ({TRUE_D/PC_TO_CM:.0f} pc)")
    print(f"  True extinction: law={TRUE_LAW}  Av={TRUE_AV}  Rv={TRUE_RV}")

    true_ext = make_extinction_model(
        enabled=True, law=TRUE_LAW, a_v=TRUE_AV, r_v=TRUE_RV
    )
    truth = run_forward(
        teff=TRUE_TEFF, logg=TRUE_LOGG, meta=TRUE_META,
        R=TRUE_R, d=TRUE_D,
        grid=grid, filters=filters,
        mag_system=MAG_SYSTEM,
        extinction=true_ext,
    )

    rng = np.random.default_rng(42)
    obs_mags  = np.array([truth.magnitudes[n] for n in filter_names])
    obs_mags += rng.normal(0.0, OBS_SIGMA, size=len(filter_names))
    obs_errs  = np.full(len(filter_names), OBS_SIGMA)

    print("\n  Band         truth_mag   obs_mag")
    for name, tm, om in zip(filter_names, [truth.magnitudes[n] for n in filter_names], obs_mags):
        print(f"  {name:12s}  {tm:8.4f}    {om:8.4f}")

    # -----------------------------------------------------------------------
    # Run one inverse fit per prescription
    # -----------------------------------------------------------------------
    results = {}

    for label, ext_kwargs in PRESCRIPTIONS:
        print(f"\n{'='*60}")
        print(f"Fitting with: {label}")
        print(f"{'='*60}")

        ext_model = make_extinction_model(**ext_kwargs)

        posterior = run_inverse(
            obs_magnitudes=obs_mags,
            obs_uncertainties=obs_errs,
            filter_names=filter_names,
            R=TRUE_R,
            d=TRUE_D,
            grid=grid,
            filters=filters,
            mag_system=MAG_SYSTEM,
            extinction=ext_model,
            n_walkers=N_WALKERS,
            n_steps=N_STEPS,
            n_burn=N_BURN,
            n_thin=N_THIN,
            p0_teff=TRUE_TEFF,
            p0_logg=TRUE_LOGG,
            p0_meta=TRUE_META,
            p0_scatter=0.02,
            progress=True,
        )

        med, lo, hi = posterior_stats(posterior)
        results[label] = dict(
            posterior=posterior,
            med=med, lo=lo, hi=hi,
        )

        print(f"\n  Results:")
        print(f"  Teff  = {med[0]:.1f}  [{lo[0]:.1f}, {hi[0]:.1f}]  (true: {TRUE_TEFF})")
        print(f"  logg  = {med[1]:.3f}  [{lo[1]:.3f}, {hi[1]:.3f}]  (true: {TRUE_LOGG})")
        print(f"  [M/H] = {med[2]:.3f}  [{lo[2]:.3f}, {hi[2]:.3f}]  (true: {TRUE_META})")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print("SUMMARY: Teff (true = {:.0f} K)".format(TRUE_TEFF))
    print(f"{'='*80}")
    print(f"{'Prescription':<28}  {'Teff med':>9}  {'−1σ':>7}  {'+1σ':>7}  {'bias':>7}")
    print(f"{'-'*65}")
    for label, r in results.items():
        med, lo, hi = r["med"], r["lo"], r["hi"]
        bias = med[0] - TRUE_TEFF
        print(f"{label:<28}  {med[0]:9.1f}  {med[0]-lo[0]:7.1f}  {hi[0]-med[0]:7.1f}  {bias:+7.1f}")

    print(f"\n{'='*80}")
    print("SUMMARY: logg (true = {:.2f})".format(TRUE_LOGG))
    print(f"{'='*80}")
    print(f"{'Prescription':<28}  {'logg med':>9}  {'−1σ':>7}  {'+1σ':>7}  {'bias':>7}")
    print(f"{'-'*65}")
    for label, r in results.items():
        med, lo, hi = r["med"], r["lo"], r["hi"]
        bias = med[1] - TRUE_LOGG
        print(f"{label:<28}  {med[1]:9.3f}  {med[1]-lo[1]:7.3f}  {hi[1]-med[1]:7.3f}  {bias:+7.3f}")

    print(f"\n{'='*80}")
    print("SUMMARY: [M/H] (true = {:.2f})".format(TRUE_META))
    print(f"{'='*80}")
    print(f"{'Prescription':<28}  {'[M/H] med':>9}  {'−1σ':>7}  {'+1σ':>7}  {'bias':>7}")
    print(f"{'-'*65}")
    for label, r in results.items():
        med, lo, hi = r["med"], r["lo"], r["hi"]
        bias = med[2] - TRUE_META
        print(f"{label:<28}  {med[2]:9.3f}  {med[2]-lo[2]:7.3f}  {hi[2]-med[2]:7.3f}  {bias:+7.3f}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    _plot_comparison(results)


# ---------------------------------------------------------------------------
# Plot: one row per parameter, one column per prescription
# ---------------------------------------------------------------------------

def _plot_comparison(results: dict):
    labels      = list(results.keys())
    n           = len(labels)
    param_names = ["Teff (K)", "logg", "[M/H]"]
    true_vals   = [TRUE_TEFF, TRUE_LOGG, TRUE_META]

    fig = plt.figure(figsize=(max(14, n * 1.2), 11))
    gs  = gridspec.GridSpec(3, 1, hspace=0.55)

    colors = plt.cm.tab20.colors

    for p_idx, (pname, ptrue) in enumerate(zip(param_names, true_vals)):
        ax = fig.add_subplot(gs[p_idx])

        meds = np.array([results[l]["med"][p_idx]    for l in labels])
        los  = np.array([results[l]["med"][p_idx]
                          - results[l]["lo"][p_idx]  for l in labels])
        his  = np.array([results[l]["hi"][p_idx]
                          - results[l]["med"][p_idx] for l in labels])

        x = np.arange(n)
        for i, (label, med, lo, hi) in enumerate(zip(labels, meds, los, his)):
            ax.errorbar(i, med, yerr=[[lo], [hi]],
                        fmt="o", color=colors[i % len(colors)],
                        markersize=7, capsize=5, linewidth=1.8,
                        label=label)

        ax.axhline(ptrue, color="black", linestyle="--", linewidth=1.5,
                   label=f"True {pname}={ptrue}")
        ax.set_ylabel(pname, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-0.5, n - 0.5)

        # shade the 1-sigma truth band
        ax.axhspan(
            ptrue - OBS_SIGMA * (ptrue if p_idx == 0 else 1),
            ptrue + OBS_SIGMA * (ptrue if p_idx == 0 else 1),
            alpha=0.08, color="black"
        )

    # title with true parameters + extinction
    fig.suptitle(
        f"Extinction prescription comparison\n"
        f"True: Teff={TRUE_TEFF:.0f} K, logg={TRUE_LOGG:.2f}, [M/H]={TRUE_META:.2f}, "
        f"d={TRUE_D/PC_TO_CM:.0f} pc\n"
        f"Injected extinction: {TRUE_LAW}  Av={TRUE_AV}  Rv={TRUE_RV}  "
        f"(noise={OBS_SIGMA} mag)",
        fontsize=11,
    )

    out = "extinction_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
