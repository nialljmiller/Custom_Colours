"""
Custom_Colours
==============
MESA-free synthetic photometry and stellar parameter inference.

Forward model  (Teff, logg, [M/H], R, d) → SED + magnitudes
Inverse model  observed magnitudes + uncertainties → posterior on (Teff, logg, [M/H])

Quick start
-----------
>>> from custom_colours import load_grid, load_filters, run_forward, run_inverse
>>>
>>> grid    = load_grid("/path/to/Kurucz2003all/")
>>> filters = load_filters(["/path/to/filters/GAIA/G.dat",
...                         "/path/to/filters/2MASS/J.dat"])
>>>
>>> # Forward
>>> result = run_forward(teff=5777, logg=4.44, meta=0.0,
...                      R=6.957e10, d=3.086e19,
...                      grid=grid, filters=filters)
>>> print(result)
>>>
>>> # Inverse
>>> posterior = run_inverse(
...     obs_magnitudes=[5.03, 4.17],
...     obs_uncertainties=[0.01, 0.02],
...     filter_names=["G", "J"],
...     R=6.957e10, d=3.086e19,
...     grid=grid, filters=filters,
... )
>>> posterior.print_summary()
>>>
>>> # Extinction (optional, applied in the forward model)
>>> from custom_colours import ExtinctionModel, make_extinction_model
>>> ext = make_extinction_model(enabled=True, law='fitzpatrick99', a_v=0.3)
>>> result = run_forward(..., extinction=ext)
>>>
>>> # Parameter modes for the inverse model
>>> from custom_colours import FitParams, fit_params_from_grid, fixed, free
>>> params = fit_params_from_grid(grid, a_v=(0.0, 2.0))  # Av free
>>> posterior = run_inverse(..., fit_params=params)
"""

from .grid    import load_grid, AtmosphereGrid
from .filters import load_filters, load_filters_from_instrument_dir, Filter
from .forward import run_forward, run_forward_batch, ForwardResult
from .inverse import run_inverse
from .io      import InverseResult, save_sed, save_magnitudes

# Parameter specification — fixed, bounded, and free modes
from .params import (
    FitParams,
    ParamSpec,
    fit_params_from_grid,
    fixed,
    free,
    bounded,
    PC_TO_CM,
    RSUN_TO_CM,
)

# Extinction — separate file, fully part of the package
from .sed_extinction import (
    ExtinctionModel,
    ExtinctionConfig,
    make_extinction_model,
    apply_extinction,
    remove_extinction,
    ccm89,
    odonnell94,
    fitzpatrick99,
    fm07,
    calzetti00,
    gordon23,
    AVAILABLE_LAWS,
)

__all__ = [
    # Grid
    "load_grid",
    "AtmosphereGrid",
    # Filters
    "load_filters",
    "load_filters_from_instrument_dir",
    "Filter",
    # Forward model
    "run_forward",
    "run_forward_batch",
    "ForwardResult",
    # Inverse model
    "run_inverse",
    # IO
    "InverseResult",
    "save_sed",
    "save_magnitudes",
    # Parameter modes
    "FitParams",
    "ParamSpec",
    "fit_params_from_grid",
    "fixed",
    "free",
    "bounded",
    "PC_TO_CM",
    "RSUN_TO_CM",
    # Extinction
    "ExtinctionModel",
    "ExtinctionConfig",
    "make_extinction_model",
    "apply_extinction",
    "remove_extinction",
    "ccm89",
    "odonnell94",
    "fitzpatrick99",
    "fm07",
    "calzetti00",
    "gordon23",
    "AVAILABLE_LAWS",
]

__version__ = "0.1.0"
