"""Frozen reference implementations of time-series core routines.

These are verbatim copies (modulo ``self`` -> explicit arguments) of the
implementations in ``vitabel.timeseries`` as of main @ 8964251, *before*
the performance refactoring of the time-series core. They serve as an oracle
for differential tests: optimized implementations must produce identical
results.

DO NOT EDIT (except for fixing the oracle itself, with a justification).
"""

from __future__ import annotations

import logging
import numbers

import numpy as np
import pandas as pd

from vitabel.typing import Timedelta, Timestamp

logger = logging.getLogger("vitabel")


def parse_time_index(
    time_index,
    time_start=None,
    time_unit: str | None = None,
    offset=None,
) -> tuple[pd.TimedeltaIndex, pd.Timestamp | None, pd.Timedelta, str]:
    """Reference for the time parsing in ``TimeSeriesBase.__init__``.

    Returns ``(time_index, time_start, offset, time_unit)`` as they would be
    stored on the series.
    """
    time_unit = time_unit or "s"
    if offset is None:
        offset = pd.Timedelta(0)
    elif isinstance(offset, numbers.Number):
        offset = pd.to_timedelta(offset, unit=time_unit)

    time_index_arr = np.asarray(time_index)
    if time_index_arr.ndim != 1:
        raise ValueError(
            f"time_index must be a 1D array, got shape {time_index_arr.shape}"
        )

    cleaned_time_index_iter = (value for value in time_index if not pd.isna(value))

    try:
        value = next(cleaned_time_index_iter)
        time_type = type(value)
    except StopIteration:
        if hasattr(time_index, "dtype"):
            dtype = time_index.dtype
            if np.issubdtype(dtype, np.datetime64):
                time_type = pd.Timestamp
            elif np.issubdtype(dtype, np.timedelta64):
                time_type = pd.Timedelta
            else:
                time_type = pd.Timedelta
        else:
            time_type = pd.Timedelta

    all_same_time_type = True
    all_numeric = True
    for value in cleaned_time_index_iter:
        if all_same_time_type and not isinstance(value, time_type):
            all_same_time_type = False
        if all_numeric and not isinstance(value, numbers.Number):
            all_numeric = False
        if not all_same_time_type and not all_numeric:
            raise ValueError(
                "All time data must be of the same type or numeric, "
                f"but found {value} ({type(value)}) instead of {time_type}"
            )

    parsed_from_strings = time_type in (str, np.str_)
    parse_error_types = (
        ValueError,
        TypeError,
        OverflowError,
        pd.errors.OutOfBoundsDatetime,
        pd.errors.OutOfBoundsTimedelta,
    )

    if parsed_from_strings:
        for convert_func in [pd.to_datetime, pd.to_timedelta]:
            try:
                time_index = convert_func(time_index)
                time_type = type(time_index[0])
                break
            except parse_error_types:
                pass
        else:
            raise ValueError(
                "The time data could not be parsed to timestamps or timedeltas"
            )

    try:
        if time_type in (pd.Timestamp, np.datetime64):
            if time_start is not None:
                raise ValueError("time_start cannot be passed if time data is absolute")
            if len(time_index) > 0:
                time_start = pd.Timestamp(time_index[0])
            time_index = pd.to_timedelta([time - time_start for time in time_index])

        elif time_type in (pd.Timedelta, np.timedelta64):
            time_index = pd.to_timedelta(time_index)

        elif issubclass(time_type, numbers.Number):
            time_index = pd.to_timedelta(time_index, unit=time_unit)

        else:
            raise ValueError(f"The time data type {time_type} is not supported")

        time_index = time_index + offset
    except (
        OverflowError,
        pd.errors.OutOfBoundsDatetime,
        pd.errors.OutOfBoundsTimedelta,
    ) as exc:
        if parsed_from_strings:
            raise ValueError(
                "The time data could not be parsed to timestamps or timedeltas"
            ) from exc
        raise

    if time_start is not None:
        time_start = pd.Timestamp(time_start)
        time_start = time_start.tz_localize(None)

    return time_index, time_start, offset, time_unit


def numeric_time(time_index: pd.TimedeltaIndex, time_unit: str) -> np.ndarray:
    """Reference for ``TimeSeriesBase.numeric_time``."""
    return np.array([time / pd.to_timedelta(1, unit=time_unit) for time in time_index])


def convert_time_input(time_input, time_start, time_unit):
    """Reference for ``TimeSeriesBase.convert_time_input``."""
    if time_start is not None:
        if isinstance(time_input, numbers.Number):
            time_input = pd.to_timedelta(time_input, unit=time_unit)
            return time_start + time_input
        elif isinstance(time_input, Timedelta):
            return time_start + time_input
        elif isinstance(time_input, Timestamp):
            return time_input
        elif isinstance(time_input, str):
            return pd.Timestamp(time_input)

    if isinstance(time_input, Timestamp):
        raise ValueError(
            f"The channel time is relative, but {time_input} is a timestamp"
        )
    elif isinstance(time_input, Timedelta):
        return time_input
    elif isinstance(time_input, numbers.Number):
        return pd.to_timedelta(time_input, unit=time_unit)
    elif isinstance(time_input, str):
        return pd.to_timedelta(time_input)

    raise ValueError(
        f"Could not convert {time_input} to a valid time format for this channel"
    )


def get_time_mask(
    time_index: pd.TimedeltaIndex,
    time_start: pd.Timestamp | None,
    time_unit: str,
    start=None,
    stop=None,
    resolution=None,
    include_adjacent: bool = False,
) -> np.ndarray:
    """Reference for ``TimeSeriesBase.get_time_mask``."""
    if len(time_index) == 0:
        return np.array([], dtype=bool)

    if time_start is None and (
        isinstance(start, Timestamp) or isinstance(stop, Timestamp)
    ):
        raise ValueError(
            "Start or stop time is given as a timestamp for a relative "
            "time channel: add time_start to the channel, or pass timedeltas."
        )

    original_time_index = time_index
    time_index = time_index.copy()
    if time_start is not None:
        time_index += time_start

    bound_cond = np.ones_like(original_time_index, dtype=bool)
    if start is not None:
        start = convert_time_input(start, time_start, time_unit)
        bound_cond &= time_index >= start
    if stop is not None:
        stop = convert_time_input(stop, time_start, time_unit)
        bound_cond &= time_index <= stop

    if include_adjacent:
        region = np.where(bound_cond)[0]
        if len(region) > 0:
            first_index = region[0]
            if first_index > 0:
                bound_cond[first_index - 1] = True
            last_index = region[-1]
            if last_index < len(bound_cond) - 1:
                bound_cond[last_index + 1] = True

    if start is not None and stop is not None and start > stop:
        logger.warning(
            f"Start time {start} is after stop time {stop}, "
            "the queried interval is empty."
        )

    if resolution is None or resolution == 0 or not bound_cond.any():
        return bound_cond

    if isinstance(resolution, str):
        resolution = pd.to_timedelta(resolution)
    if isinstance(resolution, numbers.Number):
        resolution = pd.to_timedelta(resolution, unit=time_unit)

    bounded_time = time_index[bound_cond]
    if len(bounded_time) == 1:
        return bound_cond

    mean_dt_bounded_time = (bounded_time[1:] - bounded_time[:-1]).mean()
    if mean_dt_bounded_time == pd.Timedelta(0):
        logger.warning(
            "The time index has no variation, so the resolution "
            "cannot be applied. Returning the full time index."
        )
        return bound_cond
    n_downsample = resolution / mean_dt_bounded_time
    if n_downsample <= 2:
        return bound_cond

    (included_indices,) = np.where(bound_cond)
    start_index = included_indices[0]
    end_index = included_indices[-1]
    downsampled_stepsize = int(np.floor(n_downsample))
    bound_cond &= False
    bound_cond[start_index : end_index + 1 : downsampled_stepsize] = True
    return bound_cond
