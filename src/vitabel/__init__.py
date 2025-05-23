import logging

from importlib.metadata import version, PackageNotFoundError

from vitabel.vitals import Vitals
from vitabel.timeseries import Channel, Label, IntervalLabel, TimeDataCollection


__all__ = [
    "Vitals",
    "Channel",
    "Label",
    "IntervalLabel",
    "TimeDataCollection",
]

logger = logging.getLogger("vitabel")
logger.setLevel(logging.INFO)

try:
    __version__ = version("vitabel")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
    logger.warning("vitabel is not installed. Version information is not available.")
