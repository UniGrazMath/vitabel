"""Common type aliases used in the package."""

from __future__ import annotations

import pandas as pd
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from typing import Any, Union, TypeAlias, Literal, Iterator, TYPE_CHECKING


if TYPE_CHECKING:
    from vitabel import Channel, Label

Timedelta: TypeAlias = pd.Timedelta | np.timedelta64
"""Type alias of a time difference / duration."""

Timestamp: TypeAlias = pd.Timestamp | np.datetime64
"""Type alias of a time stamp."""

ChannelSpecification: TypeAlias = Union[str, dict[str, Any], "Channel"]
"""Type alias for different ways to specify a Channel."""

LabelSpecification: TypeAlias = Union[str, dict[str, Any], "Label"]
"""Type alias for different ways to specify a Label."""

LabelPlotType: TypeAlias = Literal["scatter", "vline", "combined"]
LabelPlotVLineTextSource: TypeAlias = Literal[
    "data", "text_data", "combined", "disabled"
]

IntervalLabelPlotType: TypeAlias = Literal["box", "hline", "combined"]
# IntervalLabelPlotVLineTextSource: TypeAlias = Literal["data", "text_data", "combined", "disabled"] #TODO: yet not implemented

LabelAnnotationPresetType: TypeAlias = Literal[
    "timestamp", "numerical", "textual", "combined"
]


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

    time_index: pd.DatetimeIndex | pd.TimedeltaIndex | npt.NDArray
    """The time index of the selected data range."""

    data: npt.NDArray | None = None
    """The data of the selected data range, or ``None`` if no data
    is available.
    """

    text_data: npt.NDArray | None = None
    """The text data of the selected data range, or ``None`` if no text data
    is available.
    """

    def __len__(self) -> int:
        return len(self.time_index)

    def __iter__(self) -> Iterator:
        return iter((self.time_index, self.data, self.text_data))


@dataclass
class PhaseData:
    """Data for a single respiratory phase (inspiration or expiration).

    Parameters
    ----------
    onsets_above_threshold
        Array of timestamps marking onsets above threshold.
        Solely fulfilling the condition of being above the threshold. No further filtering.

    filtered_onsets_above_threshold
        Array of timestamps marking onsets above threshold.
        Filtered by alternating expiration phases.

    candidates
        Array of candidate timestamps for the start of inspiration phases.
        Yet, not filtered as alternating phases.

    begins
        Array of timestamps marking the beginning of inspiration phases.
        Filtered by alternating expiration phases.

    intervals
        Array of intervals marking the intervals of inspiration phases.

    threshold
        The threshold value used to detect inspiration phases.

    """

    onsets_above_threshold: npt.NDArray
    filtered_onsets_above_threshold: npt.NDArray
    candidates: npt.NDArray
    begins: npt.NDArray
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]]
    threshold: float


@dataclass
class RespPhases:
    """Auxiliary dataclass used to represent respiratory phases information.

    Parameters
    ----------
    inspiration
        All inspiration-related phase data.
    expiration
        All expiration-related phase data.
    """

    inspiration: PhaseData
    expiration: PhaseData
