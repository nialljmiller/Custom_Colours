"""
demo_inverse.py
===============
Demonstration of the Custom_Colours inverse model.

Given observed magnitudes (with uncertainties) in a set of filters,
recovers the posterior distribution on (Teff, logg, [M/H]) via MCMC.

Usage
-----
    python demo_inverse.py

Edit the OBSERVATIONS and DATA PATHS sections below to suit your setup.
The demo generates synthetic observations from known parameters so it
works out of the box.  Replace the OBS_* arrays with your real data.
"""

import numpy as np
import matplotlib.pyplot as plt

from custom_colours import (
    load_grid, load_filters_from_instrument_dir,
    run_forward, run_inverse, InverseResult,
)

# =============================================================================
# DATA PATHS  —  edit these
# =============================================================================

GRID_DIR    = "~/SED_Tools/data/stellar_models/Kurucz2003all"
FILTER_DIR  = "~/SED_Tools/data/filters/Generic/Johnson"
VEGA_SED    = "~/SED_Tools/data/stellar_models/vega_flam.csv"

# =============================================================================
# FIXED PARAMETERS
# =============================================================================

R      = 6.957e10    # Stellar radius (cm)  —  must be supplied by the user
D      = 3.0857e19   # Distance (cm)        —  10 pc

MAG_SYSTEM = "AB"

# =============================================================================
# OBSERVATIONS  —  replace with your real data
#
# If you have real magnitudes, set:
#   OBS_FILTER_NAMES  = ["B", "V", "R"]  (must match filter file stems)
#   OBS_MAGNITUDES    = [12.3, 11.8, 11.4]
#   OBS_UNCERTAINTIES = [0.02, 0.02, 0.03]
#
# Below we synthesise observations from known parameters for demonstration.
# =============================================================================

SYNTHESISE = True   # set False and fill arrays below to use real data

OBS_FILTER_NAMES  = None   # e.g. ["B", "V", "R", "I"]
OBS_MAGNITUDES    = None   # e.g. [12.3, 11.8, 11.4, 11.0]
OBS_UNCERTAINTIES = None   # e.g. [0.02, 0.02, 0.03, 0.03]

# =============================================================================
# MCMC SETTINGS
# =============================================================================

N_WALKERS = 32
N_STEPS   = 1000
N_BURN    = 300
N_THIN    = 2

# =============================================================================
# LOAD DATA
# =============================================================================

import os
GRID_DIR   = os.path.expanduser(GRID_DIR)
FILTER_DIR = os.path.expanduser(FILTER_DIR)
VEGA_SED   = os.path.expanduser(VEGA_SED)

print("Loading atmosphere grid ...")
grid = load_grid(GRID_DIR)
print(f"  {grid}")

print("\nLoading filters ...")
vega_path = VEGA_SED if os.path.isfile(VEGA_SED) else None
filters = load_filters_from_instrument_dir(FILTER_DIR, vega_sed_path=vega_path)
print(f"  {len(filters)} filters: {[f.name for f in filters]}")

# =============================================================================
# SYNTHESISE OBSERVATIONS (demo only)
# =============================================================================

TRUE_TEFF = 5778.0
TRUE_LOGG = 4.44
TRUE_META = 0.0

if SYNTHESISE:
    print(f"\nSynthesising observations from "
          f"Teff={TRUE_TEFF:.0f} K, logg={TRUE_LOGG:.2f}, [M/H]={TRUE_META:.2f} ...")
    fwd = run_forward(
        teff=TRUE_TEFF, logg=TRUE_LOGG, meta=TRUE_META,
        R=R, d=D, grid=grid, filters=filters, mag_system=MAG_SYSTEM,
    )
    rng = np.random.default_rng(42)
    OBS_FILTER_NAMES  = [f.name for f in filters]
    OBS_UNCERTAINTIES = np.full(len(filters), 0.02)
    OBS_MAGNITUDES    = (np.array([fwd.magnitudes[n] for n in OBS_FILTER_NAMES])
                         + rng.normal(0, OBS_UNCERTAINTIES))
    print("  Synthesised magnitudes (with 0.02 mag noise):")
    for name, mag, err in zip(OBS_FILTER_NAMES, OBS_MAGNITUDES, OBS_UNCERTAINTIES):
        print(f"    {name:>6s} : {mag:.4f} ± {err:.4f}")

# =============================================================================
# RUN INVERSE MODEL
# =============================================================================

print(f"\nRunning MCMC inference ...")
print(f"  Walkers={N_WALKERS}   Steps={N_STEPS}   Burn={N_BURN}   Thin={N_THIN}")
print(f"  This will take ~{N_WALKERS * N_STEPS / 1000:.0f}k forward evaluations ...")

posterior = run_inverse(
    obs_magnitudes=OBS_MAGNITUDES,
    obs_uncertainties=OBS_UNCERTAINTIES,
    filter_names=OBS_FILTER_NAMES,
    R=R, d=D,
    grid=grid,
    filters=filters,
    mag_system=MAG_SYSTEM,
    n_walkers=N_WALKERS,
    n_steps=N_STEPS,
    n_burn=N_BURN,
    n_thin=N_THIN,
    progress=True,
    seed=42,
)

# =============================================================================
# PRINT SUMMARY
# =============================================================================

posterior.print_summary()

# =============================================================================
# SAVE POSTERIOR
# =============================================================================

posterior.save("inverse_posterior.npz")
print("Saved: inverse_posterior.npz")
print("  Reload with: posterior = InverseResult.load('inverse_posterior.npz')")

# =============================================================================
# PLOT — corner plot + chain convergence
# =============================================================================

samples = posterior.samples   # shape (N, 3)
labels  = [r"$T_{\rm eff}$ (K)", r"$\log g$", r"$[\rm M/H]$"]
truths  = [TRUE_TEFF, TRUE_LOGG, TRUE_META] if SYNTHESISE else None

fig, axes = plt.subplots(3, 3, figsize=(9, 8))
fig.suptitle("Posterior: $(T_{\\rm eff},\\ \\log g,\\ [{\\rm M/H}])$", fontsize=13)

for i in range(3):
    for j in range(3):
        ax = axes[i, j]
        if i == j:
            # 1D marginal
            ax.hist(samples[:, i], bins=40, color="steelblue",
                    alpha=0.7, density=True, edgecolor="none")
            med = float(np.median(samples[:, i]))
            lo  = float(np.percentile(samples[:, i], 15.865))
            hi  = float(np.percentile(samples[:, i], 84.135))
            ax.axvline(med, color="navy", lw=1.5, label=f"median={med:.2f}")
            ax.axvline(lo,  color="navy", lw=0.8, ls="--")
            ax.axvline(hi,  color="navy", lw=0.8, ls="--")
            if truths is not None:
                ax.axvline(truths[i], color="crimson", lw=1.5, ls="-",
                           label=f"true={truths[i]:.2f}")
            ax.set_xlabel(labels[i], fontsize=9)
            ax.set_yticks([])
            ax.legend(fontsize=7, loc="upper right")
        elif i > j:
            # 2D scatter
            ax.scatter(samples[:, j], samples[:, i],
                       s=1, alpha=0.15, color="steelblue")
            if truths is not None:
                ax.plot(truths[j], truths[i], "r+", ms=10, mew=1.5)
            ax.set_xlabel(labels[j], fontsize=9)
            ax.set_ylabel(labels[i], fontsize=9)
        else:
            ax.set_visible(False)

plt.tight_layout()
plt.savefig("inverse_corner.png", dpi=150)
print("Saved: inverse_corner.png")

# --- Chain convergence ---
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
param_names = [r"$T_{\rm eff}$", r"$\log g$", r"$[\rm M/H]$"]
for i, (ax, name) in enumerate(zip(axes2, param_names)):
    ax.plot(samples[:, i], lw=0.4, alpha=0.6, color="steelblue")
    ax.set_ylabel(name, fontsize=10)
    ax.grid(True, alpha=0.3)
axes2[-1].set_xlabel("Sample index (post burn-in)", fontsize=10)
fig2.suptitle("Chain traces (post burn-in)", fontsize=11)
plt.tight_layout()
plt.savefig("inverse_chains.png", dpi=150)
print("Saved: inverse_chains.png")

plt.show()
