"""Common type aliases used in the package."""

from __future__ import annotations

import pandas as pd
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field
from typing import Any, Union, TypeAlias, Literal, Iterator, TYPE_CHECKING


if TYPE_CHECKING:
    from vitabel import Channel, Label, IntervalLabel

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
    data_timed: dict[str, pd.DataFrame] = field(default_factory=dict)
    """Timing-adjusted views of the data.

    Keys:
    - ``"cycle_start"``  — Ti, Te, Tp, Cycle number at breath-onset timestamp.
    - ``"insp_end"``     — Vi at breath-onset + Ti (end of inspiration).
    - ``"exp_end"``      — Vt, Freq, Leakage, Leakage ratio at breath-onset + Ti + Te
                           (end of expiration).
    - ``"vi_wave"``      — Reconstructed inspiratory volume waveform
                           (``Vi (displayed)``); reproduces the EOlife screen shape.
                           Per-breath anchor points: ``(onset, 0)``,
                           ``(onset+Ti, Vi)``, ``(next_onset−1ns, Vi)``. Vi is
                           held on the display through Te and the Tp pause, and
                           drops to 0 only at the next inspiration.
    - ``"v_exp"``       — ``V_exp``: graphical curve of expiratory tidal volume
                           (plot label: "Expired Volume") — ramps 0 → Vt across Te,
                           resets to 0 for the Tp pause. Useful for visual comparison
                           against ``Vi (displayed)``; *not* a screen reproduction.
    - ``"vt_displayed"`` — ``Vt (displayed)``: step function reproducing the
                           numeric Vt readout on the EOlife screen. Characterized
                           breaths step at ``onset + Ti + Te`` when Vt is not NaN.
                           Uncharacterised (Ti/Te=NaN) breaths with a saved Vt
                           step at breath onset. Uncharacterised breaths with
                           Vt=NaN drop to 0 after 3 s from the first such breath
                           in the group (EOlife internal timeout); subsequent
                           NaN-Vt breaths hold. Validated against GT recordings.
    - ``"f_displayed"``  — ``f (displayed)``: step function reproducing the
                           numeric frequency readout on the EOlife screen.
                           Updated at the onset of breath K+1 (= end of Tp of
                           breath K, i.e. the start of the next inspiration).
                           Value = ``round(60000 / mean(T[K], T[K-1], T[K-2]))``
                           where T[i] = Ti[i]+Te[i]+Tp[i] in ms and the 3-breath
                           rolling window covers only characterised breaths.
                           Reverse-engineered from validated OCR ground-truth
                           data captured on an **EOlife X** (firmware v2.1.3).
                           May not generalise to other models or firmware versions.

    For rows where Ti and Te are NaN (uncharacterised breaths) the original
    breath-onset timestamp is used as a fallback so no data are lost.
    Ti and Te are always either both present or both absent in EOlife exports;
    breaths with only one of the two are not supported.
    """

    eolife_uncharacterised_breaths: IntervalLabel | None = None
    """IntervalLabel covering contiguous runs of uncharacterised breaths, or
    ``None`` if all breaths are characterised.

    An uncharacterised breath is one for which the EOlife device could not
    determine the breath timing (Ti and Te are both exported as ``"Na"``).
    Ti and Te are always absent together; a breath with only one of the two
    missing is not a valid EOlife export.

    Each interval spans from the first uncharacterised breath onset to the next
    characterised breath onset (or to the estimated end of the last breath
    when the run reaches the end of the recording).

    Added automatically by :func:`.utils.loading.read_eolife_export`.
    Passed to the vitabel :class:`~vitabel.Vitals` as a global label by
    :meth:`~vitabel.Vitals.add_ventilatory_feedback` when it is not ``None``.
    """


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

    time_index: (
        pd.DatetimeIndex
        | pd.TimedeltaIndex
        | npt.NDArray[np.datetime64]
        | npt.NDArray[np.timedelta64]
    )
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
