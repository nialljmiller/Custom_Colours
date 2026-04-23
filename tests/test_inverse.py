"""
tests/test_inverse.py
=====================
Tests for custom_colours.inverse (photometry -> stellar parameters).

Strategy
--------
Rather than testing against external reference values, we test the
inverse model self-consistently:

  1. Run the forward model at known parameters to produce synthetic
     magnitudes.
  2. Perturb those magnitudes by a small known noise level.
  3. Run the inverse model and verify the posterior median recovers
     the true parameters within a physically reasonable tolerance.

MCMC chains are kept deliberately short (n_steps=300, n_burn=100,
n_walkers=16) so the test suite completes in under ~30 seconds.
The tolerances are set wide enough to accommodate shot noise from the
short chain while still being meaningful.

All tests that require the Fortran extension or data files skip
gracefully when those are not present.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths
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
# Skip guards
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


@pytest.fixture(scope="module")
def solar_obs(grid, filters):
    """Synthetic solar observations with 0.01 mag Gaussian noise."""
    from custom_colours import run_forward
    TRUE_TEFF = 5778.0
    TRUE_LOGG = 4.44
    TRUE_META = 0.0

    result = run_forward(
        teff=TRUE_TEFF, logg=TRUE_LOGG, meta=TRUE_META,
        R=R_SUN, d=D_10PC,
        grid=grid, filters=filters,
        mag_system="AB",
    )

    rng = np.random.default_rng(42)
    filter_names = [f.name for f in filters]
    obs_mag = np.array([result.magnitudes[n] for n in filter_names])
    obs_err = np.full(len(filter_names), 0.01)
    obs_mag += rng.normal(0, obs_err)

    return {
        "true_teff": TRUE_TEFF,
        "true_logg": TRUE_LOGG,
        "true_meta": TRUE_META,
        "obs_magnitudes":    obs_mag,
        "obs_uncertainties": obs_err,
        "filter_names":      filter_names,
    }


# ---------------------------------------------------------------------------
# Smoke test — does run_inverse return an InverseResult?
# ---------------------------------------------------------------------------

@requires_fortran
@requires_grid
@requires_filters
class TestInverseSmoke:

    def test_returns_inverse_result(self, grid, filters, solar_obs):
        from custom_colours import run_inverse, InverseResult
        result = run_inverse(
            obs_magnitudes=solar_obs["obs_magnitudes"],
            obs_uncertainties=solar_obs["obs_uncertainties"],
            filter_names=solar_obs["filter_names"],
            R=R_SUN, d=D_10PC,
            grid=grid, filters=filters,
            mag_system="AB",
            n_walkers=16, n_steps=200, n_burn=50,
            progress=False, seed=0,
        )
        assert isinstance(result, InverseResult)

    def test_samples_shape(self, grid, filters, solar_obs):
        from custom_colours import run_inverse
        result = run_inverse(
            obs_magnitudes=solar_obs["obs_magnitudes"],
            obs_uncertainties=solar_obs["obs_uncertainties"],
            filter_names=solar_obs["filter_names"],
            R=R_SUN, d=D_10PC,
            grid=grid, filters=filters,
            mag_system="AB",
            n_walkers=16, n_steps=200, n_burn=50,
            n_thin=1, progress=False, seed=0,
        )
        # (n_steps - n_burn) * n_walkers samples, 3 parameters
        expected_samples = (200 - 50) * 16
        assert result.samples.shape == (expected_samples, 3)
        assert result.log_prob.shape == (expected_samples,)

    def test_acceptance_fraction_reasonable(self, grid, filters, solar_obs):
        from custom_colours import run_inverse
        result = run_inverse(
            obs_magnitudes=solar_obs["obs_magnitudes"],
            obs_uncertainties=solar_obs["obs_uncertainties"],
            filter_names=solar_obs["filter_names"],
            R=R_SUN, d=D_10PC,
            grid=grid, filters=filters,
            mag_system="AB",
            n_walkers=16, n_steps=300, n_burn=100,
            progress=False, seed=0,
        )
        mean_af = float(np.mean(result.acceptance_fraction))
        assert 0.05 < mean_af < 0.99, (
            f"Suspicious acceptance fraction: {mean_af:.3f}"
        )


# ---------------------------------------------------------------------------
# Posterior recovery — do we get back roughly the right parameters?
# ---------------------------------------------------------------------------

@requires_fortran
@requires_grid
@requires_filters
class TestPosteriorRecovery:
    """
    Uses solar parameters which are well inside the grid.
    With 0.01 mag per-filter noise and 7 Johnson filters,
    the posterior should be narrow and centred near the truth.
    Tolerances are deliberately generous to accommodate short chains.
    """

    # These are physically motivated tolerances, not tight numerical ones.
    TEFF_TOL = 500.0    # K — grid spacing is ~250 K so 2 grid steps
    LOGG_TOL = 0.5      # dex — grid spacing is 0.5 dex
    META_TOL = 0.5      # dex — grid spacing is ~0.375 dex

    @pytest.fixture(scope="class")
    def posterior(self, grid, filters, solar_obs):
        from custom_colours import run_inverse
        return run_inverse(
            obs_magnitudes=solar_obs["obs_magnitudes"],
            obs_uncertainties=solar_obs["obs_uncertainties"],
            filter_names=solar_obs["filter_names"],
            R=R_SUN, d=D_10PC,
            grid=grid, filters=filters,
            mag_system="AB",
            n_walkers=16, n_steps=500, n_burn=150,
            p0_teff=solar_obs["true_teff"],
            p0_logg=solar_obs["true_logg"],
            p0_meta=solar_obs["true_meta"],
            p0_scatter=0.03,
            progress=False, seed=42,
        )

    def test_teff_recovered(self, posterior, solar_obs):
        median_teff = float(np.median(posterior.samples[:, 0]))
        err = abs(median_teff - solar_obs["true_teff"])
        assert err < self.TEFF_TOL, (
            f"Teff: median={median_teff:.1f}, true={solar_obs['true_teff']:.1f}, "
            f"err={err:.1f} K > tol={self.TEFF_TOL} K"
        )

    def test_logg_recovered(self, posterior, solar_obs):
        median_logg = float(np.median(posterior.samples[:, 1]))
        err = abs(median_logg - solar_obs["true_logg"])
        assert err < self.LOGG_TOL, (
            f"logg: median={median_logg:.3f}, true={solar_obs['true_logg']:.3f}, "
            f"err={err:.3f} > tol={self.LOGG_TOL}"
        )

    def test_meta_recovered(self, posterior, solar_obs):
        median_meta = float(np.median(posterior.samples[:, 2]))
        err = abs(median_meta - solar_obs["true_meta"])
        assert err < self.META_TOL, (
            f"[M/H]: median={median_meta:.3f}, true={solar_obs['true_meta']:.3f}, "
            f"err={err:.3f} > tol={self.META_TOL}"
        )

    def test_samples_within_grid(self, posterior, grid):
        """All posterior samples must lie within grid bounds."""
        t = posterior.samples[:, 0]
        l = posterior.samples[:, 1]
        m = posterior.samples[:, 2]
        assert np.all(t >= grid.teff_bounds[0]) and np.all(t <= grid.teff_bounds[1])
        assert np.all(l >= grid.logg_bounds[0]) and np.all(l <= grid.logg_bounds[1])
        assert np.all(m >= grid.meta_bounds[0]) and np.all(m <= grid.meta_bounds[1])

    def test_log_prob_finite(self, posterior):
        assert np.all(np.isfinite(posterior.log_prob))

    def test_posterior_not_degenerate(self, posterior):
        """Posterior should have non-zero spread in all three parameters."""
        for i, name in enumerate(["Teff", "logg", "[M/H]"]):
            std = float(np.std(posterior.samples[:, i]))
            assert std > 0, f"Posterior for {name} is a delta function (std=0)"


# ---------------------------------------------------------------------------
# InverseResult API
# ---------------------------------------------------------------------------

@requires_fortran
@requires_grid
@requires_filters
class TestInverseResultAPI:

    @pytest.fixture(scope="class")
    def result(self, grid, filters, solar_obs):
        from custom_colours import run_inverse
        return run_inverse(
            obs_magnitudes=solar_obs["obs_magnitudes"],
            obs_uncertainties=solar_obs["obs_uncertainties"],
            filter_names=solar_obs["filter_names"],
            R=R_SUN, d=D_10PC,
            grid=grid, filters=filters,
            mag_system="AB",
            n_walkers=16, n_steps=200, n_burn=50,
            progress=False, seed=0,
        )

    def test_summary_keys(self, result):
        s = result.summary()
        for key in ("teff", "logg", "meta"):
            assert key in s
            for sub in ("median", "lo", "hi", "lower_1sigma", "upper_1sigma"):
                assert sub in s[key]

    def test_summary_ordering(self, result):
        s = result.summary()
        for key in ("teff", "logg", "meta"):
            assert s[key]["lo"] <= s[key]["median"] <= s[key]["hi"]

    def test_map_estimate_returns_triple(self, result):
        teff, logg, meta = result.map_estimate()
        assert np.isfinite(teff)
        assert np.isfinite(logg)
        assert np.isfinite(meta)

    def test_print_summary_runs(self, result):
        result.print_summary()  # should not raise

    def test_save_and_load_roundtrip(self, result):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "posterior.npz")
            result.save(path)
            assert os.path.isfile(path)

            from custom_colours import InverseResult
            loaded = InverseResult.load(path)

        np.testing.assert_array_equal(result.samples, loaded.samples)
        np.testing.assert_array_equal(result.log_prob, loaded.log_prob)
        np.testing.assert_array_equal(result.obs_magnitudes, loaded.obs_magnitudes)
        np.testing.assert_array_equal(result.obs_uncertainties, loaded.obs_uncertainties)
        assert result.filter_names == loaded.filter_names
        assert result.R == loaded.R
        assert result.d == loaded.d
        assert result.mag_system == loaded.mag_system
        assert result.n_walkers == loaded.n_walkers
        assert result.n_steps == loaded.n_steps
        assert result.n_burn == loaded.n_burn
        assert result.n_thin == loaded.n_thin

    def test_save_adds_npz_extension(self, result):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "posterior")  # no extension
            result.save(path)
            assert os.path.isfile(path + ".npz")

    def test_thinning_reduces_samples(self, grid, filters, solar_obs):
        from custom_colours import run_inverse
        r1 = run_inverse(
            obs_magnitudes=solar_obs["obs_magnitudes"],
            obs_uncertainties=solar_obs["obs_uncertainties"],
            filter_names=solar_obs["filter_names"],
            R=R_SUN, d=D_10PC, grid=grid, filters=filters,
            n_walkers=16, n_steps=200, n_burn=50, n_thin=1,
            progress=False, seed=0,
        )
        r5 = run_inverse(
            obs_magnitudes=solar_obs["obs_magnitudes"],
            obs_uncertainties=solar_obs["obs_uncertainties"],
            filter_names=solar_obs["filter_names"],
            R=R_SUN, d=D_10PC, grid=grid, filters=filters,
            n_walkers=16, n_steps=200, n_burn=50, n_thin=5,
            progress=False, seed=0,
        )
        assert r5.samples.shape[0] < r1.samples.shape[0]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@requires_fortran
@requires_grid
@requires_filters
class TestInverseValidation:

    def test_mismatched_lengths_raises(self, grid, filters):
        from custom_colours import run_inverse
        filter_names = [filters[0].name]
        with pytest.raises(ValueError, match="filter_names"):
            run_inverse(
                obs_magnitudes=[5.0, 6.0],   # 2 mags
                obs_uncertainties=[0.01, 0.01],
                filter_names=filter_names,   # 1 name
                R=R_SUN, d=D_10PC,
                grid=grid, filters=filters,
                n_walkers=8, n_steps=10, n_burn=2,
                progress=False,
            )

    def test_zero_uncertainty_raises(self, grid, filters):
        from custom_colours import run_inverse
        with pytest.raises(ValueError, match="positive"):
            run_inverse(
                obs_magnitudes=[5.0],
                obs_uncertainties=[0.0],
                filter_names=[filters[0].name],
                R=R_SUN, d=D_10PC,
                grid=grid, filters=filters,
                n_walkers=8, n_steps=10, n_burn=2,
                progress=False,
            )

    def test_unknown_filter_name_raises(self, grid, filters):
        from custom_colours import run_inverse
        with pytest.raises(ValueError, match="not in the supplied filters"):
            run_inverse(
                obs_magnitudes=[5.0],
                obs_uncertainties=[0.01],
                filter_names=["NonExistentFilter_XYZ"],
                R=R_SUN, d=D_10PC,
                grid=grid, filters=filters,
                n_walkers=8, n_steps=10, n_burn=2,
                progress=False,
            )
