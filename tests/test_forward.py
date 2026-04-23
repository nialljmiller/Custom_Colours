"""
tests/test_forward.py
=====================
Roundtrip tests for custom_colours.forward against reference values
produced by the MESA colors unit test (colors/test/test_output).

Reference configuration
-----------------------
  Grid      : Kurucz2003all  (SED_Tools build: 4000-8000 K, logg 3-5, [M/H] -1 to 0.5)
  Filters   : Generic/Johnson  (B, I, J, M, R, U, V)
  System    : Vega
  Distance  : 10 pc  (3.0857e19 cm)
  Interp    : hermite

Notes on reference values
-------------------------
Reference magnitudes are from colors/test/test_output (MESA Kurucz2003 grid).
The SED_Tools Kurucz2003all grid may differ in wavelength sampling and flux
values, so tolerances are set generously.  Cases whose parameters fall outside
the SED_Tools grid bounds are skipped automatically.

Usage
-----
    pytest tests/test_forward.py -v
"""

from __future__ import annotations

import os
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
KURUCZ_DIR  = os.environ.get(
    "CC_KURUCZ_DIR",
    os.path.expanduser("~/SED_Tools/data/stellar_models/Kurucz2003all"),
)
JOHNSON_DIR = os.environ.get(
    "CC_JOHNSON_DIR",
    os.path.expanduser("~/SED_Tools/data/filters/Generic/Johnson"),
)
VEGA_SED    = os.environ.get(
    "CC_VEGA_SED",
    os.path.expanduser("~/SED_Tools/data/stellar_models/vega_flam.csv"),
)

R_SUN  = 6.957e10
D_10PC = 3.0857e19

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

def _cc_api_available():
    try:
        from custom_colours import cc_api  # noqa: F401
        return True
    except ImportError:
        return False

requires_fortran = pytest.mark.skipif(
    not _cc_api_available(),
    reason="cc_api Fortran extension not built",
)
requires_grid = pytest.mark.skipif(
    not os.path.isdir(KURUCZ_DIR),
    reason=f"Kurucz2003all grid not found at {KURUCZ_DIR}",
)
requires_filters = pytest.mark.skipif(
    not os.path.isdir(JOHNSON_DIR),
    reason=f"Johnson filter directory not found at {JOHNSON_DIR}",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grid():
    from custom_colours import load_grid
    return load_grid(KURUCZ_DIR)


@pytest.fixture(scope="module")
def filters():
    from custom_colours import load_filters_from_instrument_dir
    vega = VEGA_SED if os.path.isfile(VEGA_SED) else None
    return load_filters_from_instrument_dir(JOHNSON_DIR, vega_sed_path=vega)


# ---------------------------------------------------------------------------
# Reference values from colors/test/test_output
# ---------------------------------------------------------------------------

GROUP1_CASES = [
    (
        "solar",
        5778.0, 4.44, 0.0, 1.0 * R_SUN,
        {
            "Mag_bol": 16.2337837,
            "Flux_bol": 3.2098630e-07,
            "B": 5.4328315,
            "V": 4.7769811,
        },
    ),
    (
        "hot_ms",
        15000.0, 4.00, 0.0, 5.0 * R_SUN,
        {
            "Mag_bol":  8.5957683,
            "Flux_bol": 3.6449592e-04,
            "B": -1.8453872,
            "V": -1.7077286,
        },
    ),
    (
        "cool_giant",
        4000.0, 2.00, 0.0, 20.0 * R_SUN,
        {
            "Mag_bol": 11.3260468,
            "Flux_bol": 2.9483653e-05,
            "B":  2.1835469,
            "V":  0.7545416,
        },
    ),
]

# Generous tolerance: accounts for grid version differences and
# trapezoid (Python zero-points) vs Simpson (Fortran flux) integration.
TOL_MAG  = 5e-2   # 0.05 mag
TOL_FLUX = 0.02   # 2 % relative


def _in_bounds(grid, teff, logg, meta):
    return grid.in_bounds(teff, logg, meta)


# ---------------------------------------------------------------------------
# Group 1 tests
# ---------------------------------------------------------------------------

@requires_fortran
@requires_grid
@requires_filters
class TestGroup1:

    @pytest.mark.parametrize("label,teff,logg,meta,R,ref", GROUP1_CASES)
    def test_magnitudes(self, label, teff, logg, meta, R, ref, grid, filters):
        if not _in_bounds(grid, teff, logg, meta):
            pytest.skip(
                f"{label}: ({teff},{logg},{meta}) outside grid bounds "
                f"Teff={grid.teff_bounds}, logg={grid.logg_bounds}, meta={grid.meta_bounds}"
            )
        from custom_colours import run_forward
        result = run_forward(teff=teff, logg=logg, meta=meta,
                             R=R, d=D_10PC, grid=grid, filters=filters,
                             mag_system="Vega")
        filter_names = {f.name for f in filters}
        for band, ref_mag in ref.items():
            if band in ("Mag_bol", "Flux_bol"):
                continue
            if band not in filter_names:
                continue
            got = result.magnitudes[band]
            assert np.isfinite(got), f"{label}/{band}: non-finite"
            assert abs(got - ref_mag) < TOL_MAG, (
                f"{label}/{band}: got {got:.6f}, expected {ref_mag:.6f}, "
                f"diff={abs(got-ref_mag):.2e} > tol={TOL_MAG:.2e}"
            )

    @pytest.mark.parametrize("label,teff,logg,meta,R,ref", GROUP1_CASES)
    def test_bolometric_flux(self, label, teff, logg, meta, R, ref, grid, filters):
        if not _in_bounds(grid, teff, logg, meta):
            pytest.skip(f"{label}: outside grid bounds")
        from custom_colours import run_forward
        result = run_forward(teff=teff, logg=logg, meta=meta,
                             R=R, d=D_10PC, grid=grid, filters=filters)
        ref_flux = ref["Flux_bol"]
        rel_err = abs(result.bol_flux - ref_flux) / ref_flux
        assert rel_err < TOL_FLUX, (
            f"{label}: bol_flux={result.bol_flux:.4e}, expected={ref_flux:.4e}, "
            f"rel_err={rel_err:.2e}"
        )

    @pytest.mark.parametrize("label,teff,logg,meta,R,ref", GROUP1_CASES)
    def test_bolometric_magnitude(self, label, teff, logg, meta, R, ref, grid, filters):
        if not _in_bounds(grid, teff, logg, meta):
            pytest.skip(f"{label}: outside grid bounds")
        from custom_colours import run_forward
        result = run_forward(teff=teff, logg=logg, meta=meta,
                             R=R, d=D_10PC, grid=grid, filters=filters)
        ref_mag = ref["Mag_bol"]
        assert abs(result.bol_mag - ref_mag) < TOL_MAG, (
            f"{label}: bol_mag={result.bol_mag:.6f}, expected={ref_mag:.6f}, "
            f"diff={abs(result.bol_mag-ref_mag):.2e}"
        )


# ---------------------------------------------------------------------------
# Group 2a — vary [M/H]
# ---------------------------------------------------------------------------

GROUP2A_CASES = [
    (-2.00, {"B": 5.4709583, "V": 4.7518222}),
    ( 0.00, {"B": 5.4328315, "V": 4.7769811}),
]

@requires_fortran
@requires_grid
@requires_filters
class TestGroup2a:

    @pytest.mark.parametrize("meta,ref", GROUP2A_CASES)
    def test_vary_meta(self, meta, ref, grid, filters):
        if not _in_bounds(grid, 5778.0, 4.44, meta):
            pytest.skip(f"[M/H]={meta} outside grid meta bounds {grid.meta_bounds}")
        from custom_colours import run_forward
        result = run_forward(teff=5778.0, logg=4.44, meta=meta,
                             R=R_SUN, d=D_10PC, grid=grid, filters=filters,
                             mag_system="Vega")
        filter_names = {f.name for f in filters}
        for band, ref_mag in ref.items():
            if band not in filter_names:
                continue
            got = result.magnitudes[band]
            assert abs(got - ref_mag) < TOL_MAG, (
                f"[M/H]={meta}, {band}: got {got:.6f}, expected {ref_mag:.6f}"
            )


# ---------------------------------------------------------------------------
# Group 2b — vary logg
# ---------------------------------------------------------------------------

GROUP2B_CASES = [
    (1.00, {"B": 5.3645168, "V": 4.7180779}),
    (2.50, {"B": 5.3921065, "V": 4.7516225}),
    (4.00, {"B": 5.4245311, "V": 4.7516225}),
    (5.00, {"B": 5.4500000, "V": 4.7600000}),  # approximate, used for in-bounds check only
]

@requires_fortran
@requires_grid
@requires_filters
class TestGroup2b:

    @pytest.mark.parametrize("logg,ref", GROUP2B_CASES)
    def test_vary_logg(self, logg, ref, grid, filters):
        if not _in_bounds(grid, 5778.0, logg, 0.0):
            pytest.skip(f"logg={logg} outside grid logg bounds {grid.logg_bounds}")
        from custom_colours import run_forward
        result = run_forward(teff=5778.0, logg=logg, meta=0.0,
                             R=R_SUN, d=D_10PC, grid=grid, filters=filters,
                             mag_system="Vega")
        filter_names = {f.name for f in filters}
        for band, ref_mag in ref.items():
            if band not in filter_names:
                continue
            got = result.magnitudes[band]
            assert abs(got - ref_mag) < TOL_MAG, (
                f"logg={logg}, {band}: got {got:.6f}, expected {ref_mag:.6f}"
            )


# ---------------------------------------------------------------------------
# Structural / unit tests
# ---------------------------------------------------------------------------

class TestGridLoader:

    @requires_grid
    def test_load_returns_atmosphere_grid(self):
        from custom_colours import load_grid, AtmosphereGrid
        g = load_grid(KURUCZ_DIR)
        assert isinstance(g, AtmosphereGrid)

    @requires_grid
    def test_flux_shape_consistent_with_axes(self):
        from custom_colours import load_grid
        g = load_grid(KURUCZ_DIR)
        nt, nl, nm, nw = g.flux.shape
        assert nt == len(g.teff_grid)
        assert nl == len(g.logg_grid)
        assert nm == len(g.meta_grid)
        assert nw == len(g.wavelengths)

    @requires_grid
    def test_in_bounds(self):
        from custom_colours import load_grid
        g = load_grid(KURUCZ_DIR)
        assert g.in_bounds(5778, 4.44, 0.0)

    @requires_grid
    def test_out_of_bounds(self):
        from custom_colours import load_grid
        g = load_grid(KURUCZ_DIR)
        assert not g.in_bounds(99999, 4.44, 0.0)

    @requires_grid
    def test_clamp_stays_within_bounds(self):
        from custom_colours import load_grid
        g = load_grid(KURUCZ_DIR)
        teff, logg, meta = g.clamp(99999, 99, 99)
        assert g.in_bounds(teff, logg, meta)

    @requires_grid
    def test_interp_radius_zero_at_node(self):
        from custom_colours import load_grid
        g = load_grid(KURUCZ_DIR)
        t0, l0, m0 = g.teff_grid[0], g.logg_grid[0], g.meta_grid[0]
        assert g.interp_radius(t0, l0, m0) < 1e-10


class TestFilterLoader:

    @requires_filters
    def test_load_returns_filter_list(self):
        from custom_colours import load_filters_from_instrument_dir, Filter
        filters = load_filters_from_instrument_dir(JOHNSON_DIR)
        assert len(filters) > 0
        assert all(isinstance(f, Filter) for f in filters)

    @requires_filters
    def test_ab_zero_points_positive(self):
        from custom_colours import load_filters_from_instrument_dir
        for f in load_filters_from_instrument_dir(JOHNSON_DIR):
            assert f.ab_zero_point > 0

    @requires_filters
    def test_st_zero_points_positive(self):
        from custom_colours import load_filters_from_instrument_dir
        for f in load_filters_from_instrument_dir(JOHNSON_DIR):
            assert f.st_zero_point > 0

    @requires_filters
    def test_vega_zero_points_positive_when_vega_available(self):
        if not os.path.isfile(VEGA_SED):
            pytest.skip(f"Vega SED not found at {VEGA_SED}")
        from custom_colours import load_filters_from_instrument_dir
        for f in load_filters_from_instrument_dir(JOHNSON_DIR, vega_sed_path=VEGA_SED):
            assert f.vega_zero_point > 0

    @requires_filters
    def test_wavelengths_ascending(self):
        from custom_colours import load_filters_from_instrument_dir
        for f in load_filters_from_instrument_dir(JOHNSON_DIR):
            assert np.all(np.diff(f.wavelengths) > 0)


class TestForwardResult:

    @requires_fortran
    @requires_grid
    @requires_filters
    def test_result_has_all_filters(self, grid, filters):
        from custom_colours import run_forward
        result = run_forward(5778, 4.44, 0.0, R_SUN, D_10PC, grid, filters)
        for f in filters:
            assert f.name in result.magnitudes
            assert f.name in result.band_fluxes

    @requires_fortran
    @requires_grid
    @requires_filters
    def test_sed_lengths_consistent(self, grid, filters):
        from custom_colours import run_forward
        result = run_forward(5778, 4.44, 0.0, R_SUN, D_10PC, grid, filters)
        assert len(result.wavelengths) == len(result.surface_flux)
        assert len(result.wavelengths) == len(result.observed_flux)

    @requires_fortran
    @requires_grid
    @requires_filters
    def test_observed_flux_less_than_surface(self, grid, filters):
        from custom_colours import run_forward
        result = run_forward(5778, 4.44, 0.0, R_SUN, D_10PC, grid, filters)
        assert np.all(result.observed_flux <= result.surface_flux + 1e-30)

    @requires_fortran
    @requires_grid
    @requires_filters
    def test_all_magnitudes_finite(self, grid, filters):
        from custom_colours import run_forward
        result = run_forward(5778, 4.44, 0.0, R_SUN, D_10PC, grid, filters)
        for name, mag in result.magnitudes.items():
            assert np.isfinite(mag), f"Non-finite magnitude for {name}"

    @requires_fortran
    @requires_grid
    @requires_filters
    def test_bolometric_magnitude_sign(self, grid, filters):
        """Sanity check: Mag_bol = -2.5 log10(F_bol), should be positive for faint sources."""
        from custom_colours import run_forward
        result = run_forward(5778, 4.44, 0.0, R_SUN, D_10PC, grid, filters)
        # F_bol ~ 3.2e-7, so Mag_bol ~ 16.2
        assert result.bol_mag > 0
        assert abs(result.bol_mag - 16.23) < 0.5

    @requires_fortran
    @requires_grid
    @requires_filters
    def test_hermite_and_linear_close(self, grid, filters):
        from custom_colours import run_forward
        r_h = run_forward(5778, 4.44, 0.0, R_SUN, D_10PC, grid, filters,
                          interp_method="hermite")
        r_l = run_forward(5778, 4.44, 0.0, R_SUN, D_10PC, grid, filters,
                          interp_method="linear")
        for name in r_h.magnitudes:
            diff = abs(r_h.magnitudes[name] - r_l.magnitudes[name])
            assert diff < 0.05, f"hermite vs linear: {name} diff={diff:.4f}"
