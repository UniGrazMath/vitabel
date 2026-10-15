"""Utilities: constants, plot styles, and auxiliary functions.

Names that have moved to other subpackages (e.g., the loading routines, now
in :mod:`vitabel.io`) remain accessible from here for backward
compatibility. They are resolved lazily, so that importing
:mod:`vitabel.utils` does not import these subpackages (which avoids
import cycles).
"""

import importlib

from vitabel.typing import EOLifeRecord  # noqa: F401
from vitabel.utils import constants, helpers, stylesheet
from vitabel.utils.constants import *  # noqa: F403
from vitabel.utils.stylesheet import *  # noqa: F403

_LOADING_NAMES = [
    "find_wavetypevar",
    "lifepak2zoll_name",
    "new_names",
    "read_corpuls",
    "read_edfplus",
    "read_eolife_export",
    "read_lifepak",
    "read_lucas",
    "read_zollcsv",
    "read_zolljson",
    "read_zolljson_compr",
    "read_zolljson_contin",
    "read_zolljson_trend",
    "read_zollxml",
    "read_zollxml_compr",
    "read_zollxml_contin",
    "read_zollxml_trend",
]

# names from utils.helpers that are defined there (or in other utils modules)
globals().update(
    {
        name: getattr(helpers, name)
        for name in helpers.__all__
        if name not in helpers._LAZY_EXPORTS
    }
)

__all__ = sorted(
    {
        "EOLifeRecord",
        *constants.__all__,
        *stylesheet.__all__,
        *helpers.__all__,
        *_LOADING_NAMES,
    }
)


def __getattr__(name: str):
    if name == "loading":
        return importlib.import_module("vitabel.utils.loading")
    if name in _LOADING_NAMES:
        return getattr(importlib.import_module("vitabel.utils.loading"), name)
    if name in helpers._LAZY_EXPORTS:
        return getattr(helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"loading"})
