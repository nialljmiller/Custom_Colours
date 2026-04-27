"""
custom_colours.params
=====================
Shared parameter definitions used by both the forward and inverse models.

This is the single place that knows what a "stellar parameter" is, whether
it is being sampled or held fixed, and what its physical bounds are.  Having
it here — rather than scattered across forward.py and inverse.py — is what
makes the module genuinely bidirectional: both directions speak the same
language about the same quantities.

Parameter modes
---------------
Every physical parameter (Teff, logg, [M/H], Av, distance) is described by
a ``ParamSpec``.  Three modes are supported:

``fixed(value)``
    The parameter is not sampled.  It is passed directly to the forward
    model at every likelihood evaluation.  This is the default for Av and
    distance.

``free(lo, hi)``
    The parameter is sampled by the MCMC over the interval [lo, hi].  A flat
    prior is used.  This is the default for Teff, logg, and [M/H] (bounds
    taken from the atmosphere grid at construction time).

``bounded(lo, hi)``
    Alias for ``free`` — included so user code can be explicit that it wants
    the parameter sampled with a hard upper and lower limit.

The degeneracy between Av and distance is real and expected: if both are
free simultaneously the posterior will be correlated.  That is honest — the
data genuinely cannot break the degeneracy without additional information.
The MCMC will show it.

Usage
-----
::

    from custom_colours.params import ParamSpec, FitParams, fixed, free

    params = FitParams(
        teff = free(4000, 8000),
        logg = free(3.0, 5.0),
        meta = free(-1.0, 0.5),
        a_v  = fixed(0.3),           # Av fixed, not sampled
        d    = fixed(500 * PC_TO_CM), # distance fixed
    )

    # In the forward model:
    result = run_forward(..., fit_params=params, theta=[5800, 4.4, 0.0])
    # theta contains only the FREE parameters in the order teff, logg, meta, [av, d]

    # Inspect what is being sampled:
    print(params.free_names)   # ['teff', 'logg', 'meta']
    print(params.n_free)       # 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

PC_TO_CM  = 3.085677581491367e18   # 1 parsec in cm
RSUN_TO_CM = 6.957e10              # 1 R_sun in cm


# ---------------------------------------------------------------------------
# ParamSpec — describes one parameter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamSpec:
    """Specification for a single physical parameter.

    Attributes
    ----------
    name : str
        Human-readable parameter name (e.g. 'Teff', 'Av').
    mode : {'fixed', 'free'}
        Whether the parameter is held constant or sampled.
    value : float or None
        Fixed value.  Only meaningful when ``mode == 'fixed'``.
    lo, hi : float or None
        Sampling bounds.  Only meaningful when ``mode == 'free'``.
    """
    name:  str
    mode:  str           # 'fixed' | 'free'
    value: Optional[float] = None
    lo:    Optional[float] = None
    hi:    Optional[float] = None

    def __post_init__(self):
        if self.mode not in ('fixed', 'free'):
            raise ValueError(f"mode must be 'fixed' or 'free', got '{self.mode}'")
        if self.mode == 'fixed' and self.value is None:
            raise ValueError(f"ParamSpec '{self.name}': fixed mode requires a value")
        if self.mode == 'free':
            if self.lo is None or self.hi is None:
                raise ValueError(f"ParamSpec '{self.name}': free mode requires lo and hi")
            if self.lo >= self.hi:
                raise ValueError(
                    f"ParamSpec '{self.name}': lo ({self.lo}) must be < hi ({self.hi})"
                )

    @property
    def is_fixed(self) -> bool:
        return self.mode == 'fixed'

    @property
    def is_free(self) -> bool:
        return self.mode == 'free'

    def contains(self, v: float) -> bool:
        """True if *v* is within the sampling bounds (free) or equals the fixed value."""
        if self.is_fixed:
            return True  # fixed values are always "valid" from a prior standpoint
        return self.lo <= v <= self.hi

    def __repr__(self) -> str:
        if self.is_fixed:
            return f"ParamSpec({self.name}=fixed({self.value!r}))"
        return f"ParamSpec({self.name}=free({self.lo}, {self.hi}))"


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def fixed(value: float, name: str = "") -> ParamSpec:
    """Return a fixed ParamSpec."""
    return ParamSpec(name=name, mode='fixed', value=float(value))


def free(lo: float, hi: float, name: str = "") -> ParamSpec:
    """Return a free (sampled) ParamSpec with flat prior over [lo, hi]."""
    return ParamSpec(name=name, mode='free', lo=float(lo), hi=float(hi))


# bounded is an alias for free — use it when you want to be explicit
bounded = free


# ---------------------------------------------------------------------------
# FitParams — the full parameter set for one run
# ---------------------------------------------------------------------------

# Canonical ordering of parameters in the theta vector
_PARAM_ORDER = ('teff', 'logg', 'meta', 'a_v', 'd')


@dataclass
class FitParams:
    """Complete specification of all physical parameters for a fit.

    Parameters
    ----------
    teff : ParamSpec
        Effective temperature in Kelvin.
    logg : ParamSpec
        Log surface gravity (log10 g / cm s-2).
    meta : ParamSpec
        Metallicity [M/H] in dex.
    a_v : ParamSpec
        V-band extinction in magnitudes.  Default: fixed(0.0).
    d : ParamSpec
        Distance in cm.  Default: fixed(1 pc).

    Notes
    -----
    The canonical theta vector used by the MCMC sampler contains *only* the
    free parameters, in the order: teff, logg, meta, a_v, d (skipping any
    that are fixed).  Use ``pack`` / ``unpack`` to convert between the full
    parameter space and the reduced theta vector.
    """

    teff: ParamSpec
    logg: ParamSpec
    meta: ParamSpec
    a_v:  ParamSpec = field(default_factory=lambda: fixed(0.0, name='a_v'))
    d:    ParamSpec = field(default_factory=lambda: fixed(PC_TO_CM, name='d'))

    def __post_init__(self):
        # Stamp names if the user left them blank
        for attr in _PARAM_ORDER:
            spec = getattr(self, attr)
            if not spec.name:
                object.__setattr__(spec, 'name', attr)  # ParamSpec is frozen

        # Validate d > 0
        if self.d.is_fixed and self.d.value <= 0:
            raise ValueError("Distance must be > 0 cm")
        if self.d.is_free and self.d.lo <= 0:
            raise ValueError("Distance lower bound must be > 0 cm")

        # Validate Av >= 0
        if self.a_v.is_fixed and self.a_v.value < 0:
            raise ValueError("Av must be >= 0")
        if self.a_v.is_free and self.a_v.lo < 0:
            raise ValueError("Av lower bound must be >= 0")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def free_names(self) -> list:
        """Names of the free parameters in canonical order."""
        return [p for p in _PARAM_ORDER if getattr(self, p).is_free]

    @property
    def n_free(self) -> int:
        """Number of free parameters."""
        return len(self.free_names)

    @property
    def fixed_names(self) -> list:
        """Names of the fixed parameters."""
        return [p for p in _PARAM_ORDER if getattr(self, p).is_fixed]

    def _spec(self, name: str) -> ParamSpec:
        return getattr(self, name)

    # ------------------------------------------------------------------
    # theta packing / unpacking
    # ------------------------------------------------------------------

    def pack(self, teff: float, logg: float, meta: float,
             a_v: Optional[float] = None, d: Optional[float] = None) -> np.ndarray:
        """Pack full physical values into the reduced free-parameter theta vector.

        Only the free parameters are included, in canonical order.
        Fixed parameters are ignored (the values from the spec are used at
        unpack time).
        """
        full = dict(teff=teff, logg=logg, meta=meta,
                    a_v=a_v if a_v is not None else self.a_v.value,
                    d=d if d is not None else self.d.value)
        return np.array([full[p] for p in self.free_names], dtype=np.float64)

    def unpack(self, theta: np.ndarray) -> dict:
        """Expand the reduced theta vector into a full {param: value} dict.

        Fixed parameters are filled in from their specs.  Free parameters
        are taken from theta in canonical order.
        """
        if len(theta) != self.n_free:
            raise ValueError(
                f"theta has {len(theta)} elements but {self.n_free} are free "
                f"({self.free_names})"
            )
        result = {}
        free_iter = iter(theta)
        for p in _PARAM_ORDER:
            spec = self._spec(p)
            result[p] = next(free_iter) if spec.is_free else spec.value
        return result

    def in_prior(self, theta: np.ndarray) -> bool:
        """Return True if all free parameters in theta are within their bounds."""
        free_iter = iter(theta)
        for p in _PARAM_ORDER:
            spec = self._spec(p)
            if spec.is_free:
                v = next(free_iter)
                if not (spec.lo <= v <= spec.hi):
                    return False
        return True

    def initial_ball(self, n_walkers: int,
                     centre: Optional[dict] = None,
                     scatter: float = 0.02,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Generate initial walker positions in a Gaussian ball.

        Parameters
        ----------
        n_walkers : int
            Number of walkers.
        centre : dict or None
            Central values keyed by parameter name.  If None, the midpoint
            of the free bounds is used for free parameters.
        scatter : float
            Width of the ball as a fraction of the parameter range.
        rng : Generator or None

        Returns
        -------
        pos : ndarray, shape (n_walkers, n_free)
        """
        if rng is None:
            rng = np.random.default_rng()

        c = []
        s = []
        lo_arr = []
        hi_arr = []
        for p in self.free_names:
            spec = self._spec(p)
            mid = 0.5 * (spec.lo + spec.hi)
            c.append(centre[p] if (centre and p in centre) else mid)
            rng_width = spec.hi - spec.lo
            s.append(scatter * rng_width)
            lo_arr.append(spec.lo)
            hi_arr.append(spec.hi)

        c  = np.array(c)
        s  = np.array(s)
        lo = np.array(lo_arr)
        hi = np.array(hi_arr)

        pos = c + s * rng.standard_normal((n_walkers, self.n_free))
        pos = np.clip(pos, lo, hi)
        return pos

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = ["FitParams:"]
        labels = dict(
            teff="Teff (K)",
            logg="logg",
            meta="[M/H]",
            a_v="Av (mag)",
            d="distance (cm)",
        )
        for p in _PARAM_ORDER:
            spec = self._spec(p)
            label = labels.get(p, p)
            if spec.is_fixed:
                lines.append(f"  {label:<18} fixed = {spec.value!r}")
            else:
                lines.append(f"  {label:<18} free  in [{spec.lo}, {spec.hi}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Grid-aware constructor
# ---------------------------------------------------------------------------

def fit_params_from_grid(
    grid,
    a_v:  Union[float, Tuple[float, float], None] = None,
    d_cm: Union[float, Tuple[float, float], None] = None,
) -> FitParams:
    """Build a FitParams with Teff/logg/meta bounds taken from the grid.

    Parameters
    ----------
    grid : AtmosphereGrid
        The loaded atmosphere grid.  Teff, logg, and [M/H] are made free
        with bounds matching the grid.
    a_v : float, (lo, hi), or None
        - float  → fixed Av
        - (lo, hi) tuple → free Av over that range
        - None   → fixed at 0.0 (no extinction, default)
    d_cm : float, (lo, hi), or None
        - float  → fixed distance in cm
        - (lo, hi) tuple → free distance in cm over that range
        - None   → fixed at 1 pc (default)

    Returns
    -------
    FitParams
    """
    teff_spec = ParamSpec('teff', 'free',
                          lo=float(grid.teff_bounds[0]),
                          hi=float(grid.teff_bounds[1]))
    logg_spec = ParamSpec('logg', 'free',
                          lo=float(grid.logg_bounds[0]),
                          hi=float(grid.logg_bounds[1]))
    meta_spec = ParamSpec('meta', 'free',
                          lo=float(grid.meta_bounds[0]),
                          hi=float(grid.meta_bounds[1]))

    if a_v is None:
        av_spec = ParamSpec('a_v', 'fixed', value=0.0)
    elif isinstance(a_v, (int, float)):
        av_spec = ParamSpec('a_v', 'fixed', value=float(a_v))
    else:
        lo_av, hi_av = a_v
        av_spec = ParamSpec('a_v', 'free', lo=float(lo_av), hi=float(hi_av))

    if d_cm is None:
        d_spec = ParamSpec('d', 'fixed', value=PC_TO_CM)
    elif isinstance(d_cm, (int, float)):
        d_spec = ParamSpec('d', 'fixed', value=float(d_cm))
    else:
        lo_d, hi_d = d_cm
        d_spec = ParamSpec('d', 'free', lo=float(lo_d), hi=float(hi_d))

    return FitParams(
        teff=teff_spec,
        logg=logg_spec,
        meta=meta_spec,
        a_v=av_spec,
        d=d_spec,
    )
