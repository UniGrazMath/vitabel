"""Common type aliases used in the package."""

from __future__ import annotations

import pandas as pd
import numpy as np

from dataclasses import dataclass
from typing import Any, Union, TypeAlias, Literal, Iterator, TYPE_CHECKING

from dataclasses import dataclass

if TYPE_CHECKING:
    from vitabel import Channel, Label
    from pandas._libs.tslibs.timedeltas import UnitChoices as TimeUnitChoices

Timedelta: TypeAlias = pd.Timedelta | np.timedelta64
"""Type alias of a time difference / duration."""

Timestamp: TypeAlias = pd.Timestamp | np.datetime64
"""Type alias of a time stamp."""

ChannelSpecification: TypeAlias = Union[str, dict[str, Any], "Channel"]
"""Type alias for different ways to specify a Channel."""

LabelSpecification: TypeAlias = Union[str, dict[str, Any], "Label"]
"""Type alias for different ways to specify a Label."""

LabelPlotType: TypeAlias = Literal["scatter", "vline", "combined"]
LabelPlotVLineTextSource: TypeAlias = Literal["data", "text_data", "combined", "disabled"]

IntervalLabelPlotType: TypeAlias = Literal["box", "hline", "combined"]
# IntervalLabelPlotVLineTextSource: TypeAlias = Literal["data", "text_data", "combined", "disabled"] #TODO: yet not implemented

LabelAnnotationPresetType: TypeAlias = Literal["timestamp", "numerical", "textual", "combined"]


@dataclass
class EOLifeRecord:
    data: pd.DataFrame
    recording_start: pd.Timestamp
    metadata: dict[str, Any]
    column_metadata: dict[str, dict[str, str]]


@dataclass
class Metric:
    """Auxiliary dataclass used to store (numeric) values and their unit.
    
    Parameters
    ----------
    value
        A numeric value.
    unit
        String representation of the unit of the stored value.
    """
    value: float
    unit: str


@dataclass
class ThresholdMetrics:
    """Auxiliary dataclass used to represent threshold regions.

    Parameters
    ----------
    area_under_threshold
        The area under the curve below the threshold.
        Unit stored in :attr:`.Metric.unit` (e.g., ``"minutes × unit of singal"``).
    duration_under_threshold
        The total duration the signal remained below the threshold.
    time_weighted_average_under_threshold
        Area under the threshold divided by the ``observational_interval_duration``,
        Unit stored in :attr:`Metric.unit` (unit of signal).
    observational_interval_duration
        Time interval length from first last recording.
    """
    area_under_threshold: Metric
    duration_under_threshold: pd.Timedelta
    time_weighted_average_under_threshold: Metric
    observational_interval_duration: pd.Timedelta


@dataclass
class DataSlice:
    """Auxiliary dataclass holding a slice of data from a label or channel.
    
    Primarily used in the various ``get_data`` methods.
    """

    time_index: pd.DatetimeIndex | pd.TimedeltaIndex | np.typing.NDArray
    """The time index of the selected data range."""

    data: np.typing.NDArray | None = None
    """The data of the selected data range, or ``None`` if no data
    is available.
    """

    text_data: np.typing.NDArray | None = None
    """The text data of the selected data range, or ``None`` if no text data
    is available.
    """
    
    def __len__(self) -> int:
        return len(self.time_index)
    
    def __iter__(self) -> Iterator:
        return iter((self.time_index, self.data, self.text_data))
