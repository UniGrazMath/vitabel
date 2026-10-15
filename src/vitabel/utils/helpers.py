"""Backward compatible access to auxiliary functions.

The functions previously defined here have moved to dedicated modules:

- :mod:`vitabel.utils.serialization`, :mod:`vitabel.utils.matching`,
  :mod:`vitabel.utils.signal` (re-exported directly),
- :mod:`vitabel.io` and :mod:`vitabel.analysis` (re-exported lazily, i.e.,
  imported on attribute access, to avoid import cycles when importing
  :mod:`vitabel.utils`).
"""

from __future__ import annotations

import importlib

from vitabel.utils.matching import match_object  # noqa: F401
from vitabel.utils.serialization import (  # noqa: F401
    NumpyEncoder,
    compress_array,
    decompress_array,
)
from vitabel.utils.signal import (  # noqa: F401
    av_mean,
    deriv,
    determine_gaps_in_recording,
    gaussian_kernel_regression_point,
    integrate,
    linear_interpolate_gaps_in_recording,
    resample_to_common_index,
)

_LAZY_EXPORTS: dict[str, str] = {
    "rename_channels": "vitabel.io._common",
    "construct_snippets": "vitabel.analysis.circulation.features",
    "max_nontriv_autocorr": "vitabel.analysis.circulation.features",
    "qrs_detector": "vitabel.analysis.circulation.features",
    "acc_feature": "vitabel.analysis.circulation.features",
    "ek_feature": "vitabel.analysis.circulation.features",
    "ekg_acc_corr1": "vitabel.analysis.circulation.features",
    "data_filter": "vitabel.analysis.circulation.features",
    "predict_circulation": "vitabel.analysis.circulation.rosc",
    "find_ROSC_2": "vitabel.analysis.circulation.rosc",
    "area_under_threshold": "vitabel.analysis.threshold",
    "convert_two_alternating_list": "vitabel.analysis.cpr.compressions",
    "CCF_minute": "vitabel.analysis.cpr.compressions",
}
"""Names that have moved to other subpackages, mapped to their new module."""


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        return getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [  # noqa: F822 (partially lazy re-exports)
    "deriv",
    "integrate",
    "construct_snippets",
    "av_mean",
    "predict_circulation",
    "area_under_threshold",
    "rename_channels",
    "NumpyEncoder",
    "determine_gaps_in_recording",
    "linear_interpolate_gaps_in_recording",
    "resample_to_common_index",
    "gaussian_kernel_regression_point",
    "CCF_minute",
    "find_ROSC_2",
    "convert_two_alternating_list",
]
