"""Backward compatible access to the data loading routines.

The loading routines have moved to the :mod:`vitabel.io` subpackage; this
module only re-exports them.
"""

from vitabel.io._common import rename_channels  # noqa: F401
from vitabel.io.corpuls import read_corpuls
from vitabel.io.edf import read_edfplus
from vitabel.io.eolife import read_eolife_export
from vitabel.io.lifepak import lifepak2zoll_name, read_lifepak
from vitabel.io.lucas import read_lucas
from vitabel.io.vitaldb import _track_to_timeseries  # noqa: F401
from vitabel.io.zoll import (
    find_wavetypevar,
    new_names,
    read_zollcsv,
    read_zolljson,
    read_zolljson_compr,
    read_zolljson_contin,
    read_zolljson_trend,
    read_zollxml,
    read_zollxml_compr,
    read_zollxml_contin,
    read_zollxml_trend,
)
from vitabel.typing import EOLifeRecord

__all__ = [
    "EOLifeRecord",
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
