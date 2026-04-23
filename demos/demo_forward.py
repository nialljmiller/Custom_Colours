"""
demo_forward.py
===============
Demonstration of the Custom_Colours forward model.

Given stellar parameters (Teff, logg, [M/H], R, d), produces:
  - The full interpolated SED
  - Synthetic magnitudes in all loaded filters
  - Bolometric flux and magnitude

Usage
-----
    python demo_forward.py

Edit the STELLAR PARAMETERS and DATA PATHS sections below to suit your setup.
"""

import numpy as np
import matplotlib.pyplot as plt

from custom_colours import load_grid, load_filters_from_instrument_dir, run_forward
from custom_colours.io import save_sed, save_magnitudes

# =============================================================================
# DATA PATHS  —  edit these
# =============================================================================

GRID_DIR    = "~/SED_Tools/data/stellar_models/Kurucz2003all"
FILTER_DIR  = "~/SED_Tools/data/filters/Generic/Johnson"
VEGA_SED    = "~/SED_Tools/data/stellar_models/vega_flam.csv"

# =============================================================================
# STELLAR PARAMETERS  —  edit these
# =============================================================================

TEFF   = 5778.0          # Effective temperature (K)
LOGG   = 4.44            # log10(surface gravity / cm s^-2)
META   = 0.0             # Metallicity [M/H]
R      = 6.957e10        # Stellar radius (cm)  —  1 R_sun
D      = 3.0857e19       # Distance (cm)        —  10 pc  →  absolute magnitudes

MAG_SYSTEM = "AB"        # "AB", "Vega", or "ST"

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
# RUN FORWARD MODEL
# =============================================================================

print(f"\nRunning forward model ...")
print(f"  Teff={TEFF:.0f} K   logg={LOGG:.2f}   [M/H]={META:.2f}")
print(f"  R={R:.3e} cm   d={D:.3e} cm")

result = run_forward(
    teff=TEFF, logg=LOGG, meta=META,
    R=R, d=D,
    grid=grid,
    filters=filters,
    mag_system=MAG_SYSTEM,
)

if result.clamped:
    print(f"  WARNING: parameters clamped to grid boundary "
          f"(interp_radius={result.interp_radius:.4f})")
else:
    print(f"  Interpolation radius: {result.interp_radius:.6f}")

# =============================================================================
# PRINT RESULTS
# =============================================================================

print(f"\n{'─'*45}")
print(f"  Bolometric flux : {result.bol_flux:.4e} erg/s/cm²")
print(f"  Mag_bol         : {result.bol_mag:.4f}")
print(f"{'─'*45}")
print(f"  Synthetic photometry  [{MAG_SYSTEM}]")
print(f"{'─'*45}")
for name, mag in sorted(result.magnitudes.items()):
    print(f"  {name:>6s} : {mag:8.4f} mag    "
          f"F_band = {result.band_fluxes[name]:.4e} erg/s/cm²/Å")
print(f"{'─'*45}")

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

save_sed(result, "forward_sed.csv")
save_magnitudes(result, "forward_magnitudes.csv")
print("\nSaved: forward_sed.csv, forward_magnitudes.csv")

# =============================================================================
# PLOT
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(10, 7),
                         gridspec_kw={"height_ratios": [3, 1]})

# --- SED panel ---
ax = axes[0]
wl_um = result.wavelengths / 1e4   # Angstroms → microns
ax.semilogy(wl_um, result.observed_flux, color="steelblue", lw=1.2,
            label="Observed flux")
ax.set_xlim(0.3, 2.5)
ax.set_ylim(bottom=result.observed_flux[
    (wl_um > 0.3) & (wl_um < 2.5)].max() * 1e-4)
ax.set_ylabel(r"F$_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
ax.set_title(
    rf"Forward model: $T_{{\rm eff}}$={TEFF:.0f} K, "
    rf"log g={LOGG:.2f}, [M/H]={META:.2f}",
    fontsize=11,
)
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Mark filter pivot wavelengths
colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(filters)))
for filt, col in zip(filters, colors_cycle):
    pivot = np.trapezoid(filt.wavelengths * filt.transmission,
                         filt.wavelengths) / \
            np.trapezoid(filt.transmission, filt.wavelengths)
    pivot_um = pivot / 1e4
    if 0.3 < pivot_um < 2.5:
        ax.axvline(pivot_um, color=col, alpha=0.5, lw=0.8, ls="--")
        ax.text(pivot_um, ax.get_ylim()[1], filt.name,
                rotation=90, va="top", ha="center",
                fontsize=7, color=col, alpha=0.8)

# --- Magnitude bar chart ---
ax2 = axes[1]
names = sorted(result.magnitudes.keys())
mags  = [result.magnitudes[n] for n in names]
bars  = ax2.bar(names, mags, color="steelblue", alpha=0.7, edgecolor="navy")
ax2.set_ylabel(f"{MAG_SYSTEM} magnitude")
ax2.set_xlabel("Filter")
ax2.invert_yaxis()
ax2.grid(True, axis="y", alpha=0.3)
for bar, mag in zip(bars, mags):
    ax2.text(bar.get_x() + bar.get_width() / 2, mag + 0.05,
             f"{mag:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("forward_demo.png", dpi=150)
print("Saved: forward_demo.png")
plt.show()
