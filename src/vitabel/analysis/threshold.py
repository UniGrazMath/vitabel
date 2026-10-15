"""Metrics for signals falling below a threshold (e.g., hypotension)."""

from __future__ import annotations

import logging
import warnings

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from vitabel.typing import Metric, ThresholdMetrics, Timedelta, Timestamp

if TYPE_CHECKING:
    from vitabel.typing import TimeUnitChoices

logger = logging.getLogger("vitabel")


def area_under_threshold(
    timeseries: pd.Series,
    start_time: Timestamp | Timedelta | None = None,
    stop_time: Timestamp | Timedelta | None = None,
    threshold: int = 0,
    time_unit: TimeUnitChoices = "minutes",
) -> ThresholdMetrics:
    """Calculates the area and duration where the signal falls
    below a threshold.

    This function operates on a given timeseries, subtracts a threshold,
    detects zero-crossings (sign changes), interpolates crossing points,
    and integrates the area under the threshold using the trapezoidal rule.

    Parameters
    ----------
    timeseries
        A :class:`pandas.Series` holding numerical data indexed by a timeseries.
    start_time
        Start time for truncating the timeseries (passed to meth:`.Vitals.truncate`).
    stop_time
        End time for truncating the timeseries (passed to meth:`.Vitals.truncate`).
    threshold
        The threshold of the signal under which the area is calcuated.
    time_unit
        The time unit according to which the result is scaled. Defaults to ``"minutes"``,
        accepts the same arguments as ``pandas.to_timedelta``.

    Returns
    -------
    :class:`.ThresholdMetrics`
    """
    if start_time is None or stop_time is None or start_time >= stop_time:
        warnings.warn(
            f"Please pass valid 'start_time' ({start_time}) and 'stop_time' ({stop_time}) values. "
            "The function returned 'np.nan'.",
            category=UserWarning,
        )
        return ThresholdMetrics(
            area_under_threshold=Metric(
                value=np.nan, unit=f"{time_unit} × value units"
            ),
            duration_under_threshold=pd.NaT,
            time_weighted_average_under_threshold=Metric(
                value=np.nan, unit="value units"
            ),
            observational_interval_duration=pd.NaT,
        )

    # Define the time points to interpolate at
    target_times = []
    if timeseries.index.min() <= start_time:
        target_times.append(start_time)
    if timeseries.index.max() >= stop_time:
        target_times.append(stop_time)
    target_times = sorted(set(target_times))

    if target_times:
        # Remove duplicate indices before reindexing to avoid ValueError
        timeseries = timeseries[~timeseries.index.duplicated(keep="first")]
        # Interpolation: union the index with new times, sort, interpolate, and extract
        timeseries = (
            timeseries.reindex(timeseries.index.union(target_times))
            .sort_index()
            .interpolate(method="time")
        )

    ts = timeseries.truncate(before=start_time, after=stop_time)
    ts -= threshold

    mask = (
        ts.values[1:] * ts.values[:-1] < 0
    )  # check whether a sign change has occurred
    if np.any(mask):
        # interpolate intersection points with axis
        interpolated_axis_intersections = ts.index[:-1][mask] - ts.values[:-1][mask] * (
            (ts.index[1:][mask] - ts.index[:-1][mask])
            / (ts.values[1:][mask] - ts.values[:-1][mask])
        )
        intersection_series = pd.Series(
            data=np.zeros(len(interpolated_axis_intersections)),
            index=interpolated_axis_intersections,
        )
        ts = pd.concat([ts, intersection_series]).sort_index()

    ts[ts > 0] = 0
    ts *= -1

    delta_t = pd.to_timedelta(ts.index[1:] - ts.index[:-1])
    trapez_lengths = ts.array[1:] + ts.array[:-1]
    mask = trapez_lengths != 0

    time_scale = pd.to_timedelta(1, unit=time_unit)

    area_value = 0.5 * np.sum(delta_t * trapez_lengths)  # timedelta * value units
    duration_under_threshold_value = np.sum(delta_t[trapez_lengths > 0])  # timedelta
    observational_interval_duration_value = ts.index.max() - ts.index.min()  # timedelta
    if observational_interval_duration_value != pd.Timedelta(0):
        twa_value = area_value / observational_interval_duration_value  # in value units
    else:
        twa_value = np.nan

    return ThresholdMetrics(
        area_under_threshold=Metric(
            value=area_value / time_scale, unit=f"{time_unit} × value units"
        ),
        duration_under_threshold=duration_under_threshold_value,
        time_weighted_average_under_threshold=Metric(
            value=twa_value, unit="value units"
        ),
        observational_interval_duration=observational_interval_duration_value,
    )
