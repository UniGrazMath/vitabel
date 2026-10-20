"""Module holding data structures for (annotated) time series data."""

from __future__ import annotations

from copy import copy
from typing import Any

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import hashlib
import json
import logging
import numbers
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from vitabel.typing import (
    ChannelSpecification,
    DataSlice,
    IntervalLabelPlotType,
    LabelAnnotationPresetType,
    LabelPlotType,
    LabelPlotVLineTextSource,
    LabelSpecification,
    Timedelta,
    Timestamp,
)
from vitabel.utils.matching import match_object
from vitabel.utils.serialization import NumpyEncoder, decompress_array

logger: logging.Logger = logging.getLogger("vitabel")
"""Global package logger."""


DATA_HASH_VERSION = 2
"""Version of the format of :meth:`Channel.data_hash`, stored in saved files.

Version 2 hashes the raw content of the arrays. (Version 1, not stored in
files, hashed compressed arrays, which made the hash platform dependent.)
"""


def _hash_json_default(obj: Any) -> Any:
    """Shared JSON conversion for hashing and saved metadata, without
    platform-dependent compressed arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        # Strings preserve nanosecond precision and timestamp UTC offsets.
        return obj.astype(str).tolist()
    return NumpyEncoder().default(obj)


def _update_hash_with_array(hasher, array: npt.ArrayLike | None) -> None:
    """Add the content of an array (dtype, shape, and values) to a hash."""
    if array is None:
        hasher.update(b"none;")
        return
    array = np.asarray(array)
    if array.dtype.kind in "biufcmM":
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
        hasher.update(f"{array.dtype.str};{array.shape};".encode())
        hasher.update(array.tobytes())
    else:
        hasher.update(f"object;{array.shape};".encode())
        hasher.update(
            json.dumps(array.tolist(), default=_hash_json_default).encode("utf-8")
        )
    hasher.update(b";")


def _timeseries_list_info(series_list: list[TimeSeriesBase]) -> pd.DataFrame:
    """Summarizes basic information about a list of time series data.

    If the time series objects have ``metadata`` attributes, the
    metadata will also be included in the summary.

    Parameters
    ----------
    series_list
        A list of time series data.
    """
    info_dict = {}
    for idx, series in enumerate(series_list):
        info_dict[idx] = {}
        if hasattr(series, "metadata") and isinstance(series.metadata, dict):
            info_dict[idx].update(series.metadata)
        if hasattr(series, "name"):
            info_dict[idx]["Name"] = series.name

        info_dict[idx].update(
            {
                "First Entry": series.first_entry,
                "Last Entry": series.last_entry,
                "Length": len(series),
                "Offset": series.offset,
            }
        )

        if hasattr(series, "anchored_channel"):
            if series.anchored_channel is not None:
                info_dict[idx]["Attached Channel"] = series.anchored_channel.name
            else:
                info_dict[idx]["Attached Channel"] = ""
        if hasattr(series, "labels"):
            info_dict[idx]["Attached Labels"] = ", ".join(
                [label.name for label in series.labels]
            )

    df = pd.DataFrame(info_dict).transpose()
    df.columns = [col.title() for col in df.columns]  # capitalize column names
    desired_columns = [
        "Name",
        "Units",
        "Length",
        "First Entry",
        "Last Entry",
        "Offset",
        "Source",
    ]
    df_columns = [col for col in desired_columns if col in df.columns]
    for col_name in ["Attached Channel", "Attached Labels"]:
        if col_name in df.columns:
            df_columns.append(col_name)
    metadata_columns = [col for col in df.columns.values if col not in df_columns]
    return df.reindex(columns=df_columns + metadata_columns)


_TYPED_ARRAY_TYPES = (np.ndarray, pd.Index, pd.Series)

_NS_PER_UNIT = {
    "W": 7 * 24 * 3600 * 10**9,
    "D": 24 * 3600 * 10**9,
    "h": 3600 * 10**9,
    "m": 60 * 10**9,
    "s": 10**9,
    "ms": 10**6,
    "us": 10**3,
    "ns": 1,
}
"""Nanoseconds per numpy datetime / timedelta unit (fixed-length units only)."""

_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _integer_ns(value: np.datetime64 | np.timedelta64) -> int:
    """Exact number of nanoseconds (since the epoch, for timestamps) of a numpy
    timestamp or timedelta, as a Python integer (no overflow). Raises
    ``KeyError`` for calendar units (years, months) and ``ValueError`` for
    NaT."""
    if np.isnat(value):
        raise ValueError("NaT has no integer value")
    unit, count = np.datetime_data(value.dtype)
    return int(value.astype(np.int64)) * _NS_PER_UNIT[unit] * count


def _is_nonempty_selection(selection: slice | npt.NDArray) -> bool:
    """Whether a selection (slice, boolean mask, or integer positions)
    selects anything."""
    if isinstance(selection, slice):
        return selection.start < selection.stop
    if selection.dtype == bool:
        return bool(selection.any())
    return len(selection) > 0


def _select(array: npt.NDArray, selection: slice | npt.NDArray) -> npt.NDArray:
    """Select from a numpy array, always returning a copy (slicing alone
    would return a view)."""
    if isinstance(selection, slice):
        return array[selection].copy()
    return array[selection]


def _typed_array_kind(time_index: Any) -> str | None:
    """Return the numpy dtype kind of a typed array (numpy array, pandas
    Index or Series), with ``"M"`` also for time zone aware timestamps, or
    ``None`` for other inputs (lists, extension dtypes, ...)."""
    if not isinstance(time_index, _TYPED_ARRAY_TYPES):
        return None
    dtype = time_index.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return "M"
    if isinstance(dtype, np.dtype):
        return dtype.kind
    return None


def _time_type_from_dtype(time_index: Any) -> type | None:
    """Fast path for the time type detection in :func:`_parse_time_index`.

    For typed arrays of numbers, timestamps or timedeltas, all elements have
    the same type, so the element-wise detection (which skips missing values)
    can be replaced by looking at the dtype. Returns a type that selects the
    same conversion branch as the element-wise detection, or ``None`` if the
    element-wise detection is needed (other inputs, empty or all-missing
    arrays).
    """
    kind = _typed_array_kind(time_index)
    if kind is None or kind not in "iufMm" or len(time_index) == 0:
        return None
    if kind in "fMm" and not pd.notna(time_index).any():
        return None
    return {"M": pd.Timestamp, "m": pd.Timedelta}.get(kind, numbers.Number)


def _as_time_array(time_data: Any) -> np.ndarray:
    """Times as a numpy array; timestamps and timedeltas (also in lists, or
    as scalars, which numpy keeps as objects) as ``datetime64`` /
    ``timedelta64``."""
    array = np.asarray(time_data)
    if array.dtype == object and array.size > 0:
        first = array.flat[0]
        if isinstance(first, (pd.Timestamp, np.datetime64)):
            array = array.astype("datetime64[ns]")
        elif isinstance(first, (pd.Timedelta, np.timedelta64)):
            array = array.astype("timedelta64[ns]")
    return array


def _parse_time_index(
    time_index: npt.ArrayLike,
    time_start: Timestamp | None,
    time_unit: str,
    offset: Timedelta,
) -> tuple[pd.TimedeltaIndex, Timestamp | None]:
    """Convert time data as passed to :class:`TimeSeriesBase` into a relative
    time index (with ``offset`` applied) and the reference time ``time_start``
    (``None`` for relative time data).
    """
    # we need to process the passed time_index. in the end,
    # the stored time_index should always be a pandas.TimedeltaIndex,
    # with time_start set to None for relative time data,
    # and to a pandas.Timestamp for absolute time data.
    # it is possible that the specified time_index is empty
    # or contains undefined values like pandas.NaT, numpy.nan, or None.

    # avoid np.asarray for typed arrays: for time zone aware timestamps it
    # creates an object array
    time_index_arr = (
        time_index
        if isinstance(time_index, _TYPED_ARRAY_TYPES)
        else np.asarray(time_index)
    )
    if time_index_arr.ndim != 1:
        raise ValueError(
            f"time_index must be a 1D array, got shape {time_index_arr.shape}"
        )

    time_type = _time_type_from_dtype(time_index)
    cleaned_time_index_iter = (
        iter(())  # typed array: all elements have the same type
        if time_type is not None
        else (value for value in time_index if not pd.isna(value))
    )

    try:
        if time_type is None:
            value = next(cleaned_time_index_iter)
            time_type = type(value)
    except StopIteration:
        # no non-na values in time_index
        if hasattr(time_index, "dtype"):
            # try to infer time type for empty time_index
            # based on the dtype
            dtype = time_index.dtype
            if np.issubdtype(dtype, np.datetime64):
                time_type = pd.Timestamp
            elif np.issubdtype(dtype, np.timedelta64):
                time_type = pd.Timedelta
            else:
                time_type = pd.Timedelta
        else:  # general fallback: assume relative time
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
        if time_type in (pd.Timestamp, np.datetime64):  # absolute time
            # check that time_start does not conflict
            if time_start is not None:
                raise ValueError("time_start cannot be passed if time data is absolute")
            if len(time_index) > 0:
                time_start = pd.Timestamp(time_index[0])
            if _typed_array_kind(time_index) == "M" and not pd.isna(time_start):
                # vectorized; same values and unit as the element-wise
                # subtraction below (also for NaT and time zone aware data).
                # a leading NaT (time_start = NaT) gives an all-NaT index whose
                # unit differs, so that case is left to the element-wise path
                time_index = pd.DatetimeIndex(time_index) - time_start
            else:
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
    return time_index, time_start


class TimeSeriesBase:
    """Base class for time series data.

    Time data in this class can represent both absolute
    and relative time.

    Parameters
    ----------

    time_index
        The time data. Can be a list of timestamps,
        timedeltas, numeric values (for which a unit needs
        to be specified), or datetime strings (which will
        be parsed by ``pandas``).
    time_start
        If the specified ``time_index`` holds relative time data
        (timedeltas or numeric values), this parameter can be used
        to set an absolute reference time (a timestamp).
    time_unit
        The unit of the time data as a string. Used for the
        conversion from numeric values. Defaults to
        seconds (``"s"``).
    offset
        An additional time offset specified as a timedelta
        or a numeric value (which is taken with respect to
        the given time unit). The offset is applied to
        :attr:`time_index`, but when the data is exported,
        the offset is removed (and exported separately).
    """

    time_index: pd.TimedeltaIndex
    """The time data (with the specified offset applied),
    always stored in relative time."""

    time_start: Timestamp | None
    """If the data of the time series is absolute, this
    holds the reference time. This is ``None`` for relative time series."""

    time_unit: str
    """The unit of the time data."""

    _offset: Timedelta
    """The offset applied to the time data."""

    def __init__(
        self,
        time_index: npt.ArrayLike[Timestamp | Timedelta | float | str],
        time_start: Timestamp | None = None,
        time_unit: str | None = None,
        offset: Timedelta | float | None = None,
    ):
        self.time_unit = time_unit or "s"
        if offset is None:
            offset = pd.Timedelta(0)
        elif isinstance(offset, numbers.Number):
            offset = pd.to_timedelta(offset, unit=self.time_unit)
        self._offset = offset

        self.time_index, self.time_start = _parse_time_index(
            time_index, time_start, self.time_unit, offset
        )

    def __len__(self) -> int:
        """Return the number of time points."""
        return len(self.time_index)

    @property
    def offset(self) -> Timedelta:
        """The offset applied to the time data."""
        return self._offset

    @property
    def first_entry(self) -> Timestamp | Timedelta | None:
        """The first (and earliest) entry in the time index of this series."""
        min_time = None
        if len(self) > 0:
            min_time = self.time_index[0]
            if self.is_time_absolute():
                min_time += self.time_start
        return min_time

    @property
    def last_entry(self) -> Timestamp | Timedelta | None:
        """The last (and latest) entry in the time index of this series."""
        max_time = None
        if len(self) > 0:
            max_time = self.time_index[-1]
            if self.is_time_absolute():
                max_time += self.time_start
        return max_time

    @offset.setter
    def offset(self, value: Timedelta | float):
        """Set the offset applied to the time data.

        Parameters
        ----------
        value
            The new offset to apply. Can be a timedelta or a numeric value
            (which is taken with respect to the time unit of the base class).
        """
        if isinstance(value, numbers.Number):
            value = pd.to_timedelta(value, unit=self.time_unit)
        delta_t = value - self._offset
        self.shift_time_index(delta_t, time_unit=self.time_unit)

    def is_empty(self) -> bool:
        """Return whether the time data is empty."""
        return len(self.time_index) == 0

    def is_time_relative(self) -> bool:
        """Return whether the time data is relative."""
        return self.time_start is None

    def is_time_absolute(self) -> bool:
        """Return whether the time data is absolute."""
        return self.time_start is not None

    def numeric_time(self, time_unit: str | None = None) -> npt.NDArray:
        """Return the relative time data as numeric values.

        Parameters
        ----------
        time_unit
            The unit of the time data as a string. If not
            specified, the unit of the time data at
            initialization is used.
        """
        time_unit = time_unit or self.time_unit
        # vectorized; bit-identical to dividing element by element
        # (NaT becomes NaN)
        return np.asarray(
            self.time_index / pd.to_timedelta(1, unit=time_unit), dtype=float
        )

    def shift_time_index(
        self,
        delta_t: pd.Timedelta | float,
        time_unit: str | None = None,
    ):
        """Shift the time index by a given time delta.

        Parameters
        ----------
        delta_t
            The time delta to shift the time index by.
        """
        time_unit = time_unit or self.time_unit
        if isinstance(delta_t, numbers.Number):
            delta_t = pd.to_timedelta(delta_t, unit=time_unit)
        self.time_index += delta_t
        self._offset += delta_t

    def convert_time_input(self, time_input: Timestamp | Timedelta | float | str):
        """Convert a given time input to either a timedelta or a timestamp,
        whatever is compatible with the time format of this channel.

        Parameters
        ----------
        time_input
            The time input to convert. If the channel time is absolute,
            the input is converted to a timestamp. If it is relative,
            the input is converted to a timedelta, if possible.
        """
        if self.is_time_absolute():
            if isinstance(time_input, numbers.Number):
                time_input = pd.to_timedelta(time_input, unit=self.time_unit)
                return self.time_start + time_input
            elif isinstance(time_input, Timedelta):
                return self.time_start + time_input
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
            return pd.to_timedelta(time_input, unit=self.time_unit)
        elif isinstance(time_input, str):
            return pd.to_timedelta(time_input)

        raise ValueError(
            f"Could not convert {time_input} to a valid time format for this channel"
        )

    def get_time_mask(
        self,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        resolution: Timedelta | float | str | None = None,
        include_adjacent: bool = False,
    ):
        """Return a boolean mask for the time index.

        Parameters
        ----------
        start
            The start time for the mask. If not specified, the
            mask starts from the beginning.
        stop
            The stop time for the mask. If not specified, the
            mask ends at the last time point.
        resolution
            The resolution for the mask. If specified, the mask
            is downsampled, by keeping every n-th data point, where
            n is resolution/ (mean time difference in time_index)
            Assumes that the time index is sorted.
        include_adjacent
            If ``True``, the mask includes the time points that are
            adjacent to (but outside of) the start and stop times; useful
            for plotting. Defaults to ``False``.
        """
        selection = self._time_selection(
            start=start,
            stop=stop,
            resolution=resolution,
            include_adjacent=include_adjacent,
        )
        if isinstance(selection, np.ndarray) and selection.dtype == bool:
            return selection
        mask = np.zeros(len(self), dtype=bool)
        mask[selection] = True
        return mask

    def _time_selection(
        self,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        resolution: Timedelta | float | str | None = None,
        include_adjacent: bool = False,
    ) -> slice | npt.NDArray:
        """Select the time points of a window, as in :meth:`get_time_mask`.

        If the time index is sorted, the window is found by binary search
        (cost independent of the length of the series) and returned as a
        slice, or as integer positions if it is downsampled. Otherwise, a
        boolean mask is returned.

        Indexing the time index with the result gives the same result as
        indexing with the mask from :meth:`_time_mask_by_comparison`,
        including the ``freq`` attribute (pandas converts contiguous boolean
        masks to slices, which keep ``freq``). Numpy arrays must be indexed
        with :func:`_select`, which copies slices.
        """
        if not self.is_empty() and not (
            self.is_time_relative()
            and (isinstance(start, Timestamp) or isinstance(stop, Timestamp))
        ):
            positions = self._time_positions(
                start=start,
                stop=stop,
                resolution=resolution,
                include_adjacent=include_adjacent,
            )
            if positions is not None:
                n_selected = len(range(positions.start, positions.stop, positions.step))
                if positions.step == 1 or n_selected <= 1:
                    # contiguous
                    return slice(positions.start, positions.start + n_selected)
                return np.arange(positions.start, positions.stop, positions.step)
        return self._time_mask_by_comparison(
            start=start,
            stop=stop,
            resolution=resolution,
            include_adjacent=include_adjacent,
        )

    def _relative_ns(self, time_value: Timestamp | Timedelta) -> int | None:
        """Exact time (as returned by :meth:`convert_time_input`) relative to
        :attr:`time_start` in integer nanoseconds, or ``None`` if it can't be
        compared with the time index this way."""
        if self.is_time_absolute():
            if isinstance(time_value, pd.Timestamp):
                if time_value.tz is not None:
                    return None  # comparison raises, see the fallback
                time_value = time_value.to_datetime64()
            if not isinstance(time_value, np.datetime64):
                return None
            return _integer_ns(time_value) - _integer_ns(
                self.time_start.to_datetime64()
            )
        if isinstance(time_value, pd.Timedelta):
            time_value = time_value.to_timedelta64()
        if not isinstance(time_value, np.timedelta64):
            return None
        return _integer_ns(time_value)

    def _time_positions(
        self,
        start: Timestamp | Timedelta | float | None,
        stop: Timestamp | Timedelta | float | None,
        resolution: Timedelta | float | str | None,
        include_adjacent: bool,
    ) -> slice | None:
        """Positions of the time points in a window via binary search, or
        ``None`` if the time index is not sorted (or contains NaT), or the
        bounds can't be handled exactly; then :meth:`_time_mask_by_comparison`
        is used.

        Gives the same result as the comparison: the bounds are converted
        exactly to integers in the unit of the time index (rounding the start
        up and the stop down), so no precision is lost when the bounds are
        finer than the unit of the time index.
        """
        time_index = self.time_index
        if (
            not isinstance(time_index, pd.TimedeltaIndex)
            or time_index.unit not in _NS_PER_UNIT
            or (self.is_time_absolute() and pd.isna(self.time_start))
            or not time_index.is_monotonic_increasing
        ):
            return None

        try:
            start_value = None if start is None else self.convert_time_input(start)
            stop_value = None if stop is None else self.convert_time_input(stop)
            start_ns = None if start is None else self._relative_ns(start_value)
            stop_ns = None if stop is None else self._relative_ns(stop_value)
        except (TypeError, ValueError, KeyError, OverflowError):
            # let the comparison raise (or handle) it in the same way as before
            return None
        if (start is not None and start_ns is None) or (
            stop is not None and stop_ns is None
        ):
            return None

        values = time_index.asi8
        unit_ns = _NS_PER_UNIT[time_index.unit]
        n_points = len(values)
        first, last = 0, n_points
        if start_ns is not None:
            lower = -(-start_ns // unit_ns)  # first value >= start
            if lower > _INT64_MAX:
                first = n_points
            elif lower >= _INT64_MIN:
                first = int(np.searchsorted(values, lower, side="left"))
        if stop_ns is not None:
            upper = stop_ns // unit_ns  # last value <= stop
            if upper < _INT64_MIN:
                last = 0
            elif upper <= _INT64_MAX:
                last = int(np.searchsorted(values, upper, side="right"))
        last = max(first, last)

        if include_adjacent and first < last:
            first = max(first - 1, 0)
            last = min(last + 1, n_points)

        if start is not None and stop is not None and start_value > stop_value:
            logger.warning(
                f"Start time {start_value} is after stop time {stop_value}, "
                "the queried interval is empty."
            )

        step = self._downsampling_step(first, last, resolution)
        return slice(first, last, step)

    def _downsampling_step(
        self,
        first: int,
        last: int,
        resolution: Timedelta | float | str | None,
    ) -> int:
        """Step size for downsampling the time points at positions
        ``first:last`` to the given resolution; same rule as in
        :meth:`_time_mask_by_comparison`."""
        if resolution is None or resolution == 0 or first >= last:
            return 1

        if isinstance(resolution, str):
            resolution = pd.to_timedelta(resolution)
        if isinstance(resolution, numbers.Number):
            resolution = pd.to_timedelta(resolution, unit=self.time_unit)

        bounded_time = self.time_index[first:last]
        if self.is_time_absolute():
            # same unit as in the comparison (relevant for the mean below)
            bounded_time = bounded_time + self.time_start
        if len(bounded_time) == 1:
            return 1

        mean_dt_bounded_time = (bounded_time[1:] - bounded_time[:-1]).mean()
        if mean_dt_bounded_time == pd.Timedelta(0):
            logger.warning(
                "The time index has no variation, so the resolution "
                "cannot be applied. Returning the full time index."
            )
            return 1
        n_downsample = resolution / mean_dt_bounded_time
        if n_downsample <= 2:
            return 1
        return int(np.floor(n_downsample))

    def _time_mask_by_comparison(
        self,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        resolution: Timedelta | float | str | None = None,
        include_adjacent: bool = False,
    ) -> npt.NDArray[np.bool_]:
        """Implementation of :meth:`get_time_mask` that compares every time
        point with the bounds. Works for unsorted time indices and NaT."""
        if self.is_empty():
            return np.array([], dtype=bool)

        if self.is_time_relative() and (
            isinstance(start, Timestamp) or isinstance(stop, Timestamp)
        ):
            raise ValueError(
                "Start or stop time is given as a timestamp for a relative "
                "time channel: add time_start to the channel, or pass timedeltas."
            )

        time_index = self.time_index.copy()
        if self.is_time_absolute():
            time_index += self.time_start

        bound_cond = np.ones_like(self.time_index, dtype=bool)
        if start is not None:
            start = self.convert_time_input(start)
            bound_cond &= time_index >= start
        if stop is not None:
            stop = self.convert_time_input(stop)
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
            resolution = pd.to_timedelta(resolution, unit=self.time_unit)

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

    def scale_time_index(
        self,
        scale_factor: float,
        reference_time: Timestamp | Timedelta | None = None,
    ) -> Self:
        """Scale the time index by a given factor while keeping
        a given reference time fixed.

        Parameters
        ----------
        scale_factor
            The factor to scale the time index by.
        reference_time
            If specified, the time index is scaled with respect to this
            reference time. If not specified, the time index is scaled
            relative to the start of the time index.
        """
        if scale_factor <= 0:
            raise ValueError(
                f"Time scale factor must be positive, but got {scale_factor}"
            )

        series = copy(self)
        # offset already applied to time_index, remove from copy
        series._offset = pd.Timedelta(0)

        if reference_time is None:
            if series.is_time_absolute():
                reference_time = pd.Timedelta(0)
            else:
                reference_time = series.time_index[0]

        if isinstance(reference_time, Timestamp):
            reference_time = pd.Timestamp(reference_time) - series.time_start

        scaled_index = (
            series.time_index - reference_time
        ) * scale_factor + reference_time
        series.time_index = scaled_index

        return series


class Channel(TimeSeriesBase):
    """A time data channel, holding a time series and optional data points.

    Parameters
    ----------

    name
        The name of the channel.
    time_index
        The time data of the channel. Can be a list of timestamps,
        timedeltas, numeric values (interpreted with respect to the specified time unit),
        or datetime strings (which will be parsed by ``pandas``).
    data
        The data points of the channel. If not specified, the channel
        only holds time data. If specified, the length of the data must
        match the length of the time index.
    time_start
        If the specified ``time_index`` holds relative time data
        (timedeltas or numeric values), this parameter can be used
        to set an absolute reference time (a timestamp).
    time_unit
        The unit of the time data as a string.
    offset
        An additional time offset specified as a timedelta
        or a numeric value (which is taken with respect to
        the given time unit). The offset is applied to
        :attr:`time_index`, but when the data is exported,
        the offset is removed (and exported separately).
    plotstyle
        A dictionary of key/value pairs that are passed to
        ``matplotlib.pyplot.plot`` when plotting channel data.
    metadata
        A dictionary that can be used to store additional
        information about the channel.
    """

    def __init__(
        self,
        name: str,
        time_index: npt.ArrayLike[
            pd.Timestamp | np.datetime64 | pd.Timedelta | np.timedelta64 | float | str
        ],
        data: npt.ArrayLike[float | np.number] | None = None,
        time_start: pd.Timestamp | None = None,
        time_unit: str | None = None,
        offset: pd.Timedelta | float | None = None,
        plotstyle: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = str(name)
        """The name of the channel."""

        self.labels = []
        """The :class:`Labels <.Label>` attached to the channel."""

        if data is not None:
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)

            if data.ndim != 1:
                raise ValueError(f"data must be a 1D array, got shape {data.shape}")

            if len(data) != len(time_index):
                raise ValueError(
                    f"For channels with data the time index length ({len(time_index)}) "
                    f"and data length ({len(data)}) must be the same"
                )

        self.data: npt.ArrayLike[float | np.number] | None = data
        """The data points of the channel, if any."""

        self.plotstyle = copy(plotstyle) or {}
        """Keyword arguments passed to the plotting routine."""

        self.metadata = copy(metadata) or {}
        """Additional channel metadata."""

        # TODO: figure out how "corrections" to values
        # are handled. Some sort of "override_value" argument
        # perhaps. Would need to be careful with the behavior
        # of potential offset.
        super().__init__(
            time_index=time_index,
            time_start=time_start,
            time_unit=time_unit,
            offset=offset,
        )

    def __repr__(self) -> str:
        """A string representation of the channel."""
        return f"{self.__class__.__name__}({self.name})"

    def data_hash(self) -> str:
        """Return a hash representing the data and the metadata of this channel.

        The hash only depends on the content (name, time and data values,
        metadata, time start, time unit, offset), and is independent of the
        platform. See :data:`DATA_HASH_VERSION`.
        """
        hasher = hashlib.sha256()
        header = {
            "name": self.name,
            "metadata": self.metadata,
            "time_start": str(self.time_start) if self.time_start is not None else None,
            "time_unit": self.time_unit,
            "offset": self.offset / pd.to_timedelta(1, unit=self.time_unit),
        }
        hasher.update(
            json.dumps(header, default=_hash_json_default, sort_keys=True).encode(
                "utf-8"
            )
        )
        for array in (self.numeric_time(), self.data):
            _update_hash_with_array(hasher, array)
        return hasher.hexdigest()

    def attach_label(self, label: Label):
        """Attach a label to this channel.

        Modifications to the channel offset also modify the offset of the
        attached labels.

        Parameters
        ----------
        label
            The label to attach.
        """
        # check whether time type is the same
        if not label.is_empty() and self.is_time_absolute() != label.is_time_absolute():
            raise ValueError(
                f"The time index of this channel and the label {label.name} "
                "must be both absolute or both relative"
            )
        if label in self.labels:
            raise ValueError(
                f"The label {label.name} is already attached to this channel"
            )
        self.labels.append(label)
        label.anchored_channel = self

    def detach_label(self, label: Label):
        """Detach a label from this channel.

        .. note::

            This method only removes the reference from the channel
            to the label, as well as the backreference from the label
            to the channel. While this does not delete the label
            object itself, the responsibility to manage the label
            object is with the user.

        Parameters
        ----------
        label
            The label to detach.
        """
        if label not in self.labels:
            raise ValueError(f"The label {label.name} is not attached to this channel")
        label.anchored_channel = None
        self.labels.remove(label)

    def shift_time_index(
        self, delta_t: pd.Timedelta | float, time_unit: str | None = None
    ):
        for label in self.labels:
            label.shift_time_index(delta_t=delta_t, time_unit=time_unit)
        return super().shift_time_index(delta_t, time_unit)

    def truncate(
        self,
        start_time: Timestamp | Timedelta | None = None,
        stop_time: Timestamp | Timedelta | None = None,
    ) -> Channel:
        """Return a new channel that is a truncated version of this channel.

        Parameters
        ----------
        start_time
            The start time for the truncated channel.
        stop_time
            The stop time for the truncated channel.
        """

        channel_data = self.get_data(start=start_time, stop=stop_time)
        truncated_time_index = channel_data.time_index
        truncated_data = channel_data.data
        # if channel is absolute time, the truncated time index is absolute,
        # no need to set start_time. also, offset is applied to the truncated
        # time index.
        truncated_channel = Channel(
            name=self.name,
            time_index=truncated_time_index,
            data=truncated_data,
            time_unit=self.time_unit,
            plotstyle=copy(self.plotstyle),
            metadata=copy(self.metadata),
        )
        for label in self.labels:
            truncated_label = label.truncate(start_time=start_time, stop_time=stop_time)
            truncated_channel.attach_label(truncated_label)
        return truncated_channel

    def to_dict(self) -> dict[str, Any]:
        """Construct a serializable dictionary that represents this channel.

        Metadata uses the same plain JSON representation as :meth:`data_hash`;
        the original metadata is not modified.
        """
        numeric_offset = self.offset / pd.to_timedelta(1, unit=self.time_unit)
        numeric_time = self.numeric_time() - numeric_offset

        return {
            "name": self.name,
            "time_index": numeric_time,
            "data": self.data,
            "time_start": str(self.time_start) if self.time_start is not None else None,
            "time_unit": self.time_unit,
            "offset": numeric_offset,
            "labels": [label.to_dict() for label in self.labels],
            "plotstyle": self.plotstyle,
            "metadata": json.loads(
                json.dumps(self.metadata, default=_hash_json_default)
            ),
        }

    def to_csv(
        self,
        filename: str | Path | None = None,
        start: Timestamp | Timedelta | None = None,
        stop: Timestamp | Timedelta | None = None,
    ) -> None:
        """Export the channel data to a CSV file.

        Parameters
        ----------
        filename
            The name of the file to export the data to. If not
            specified, the data is exported to a file with the
            name of the channel.
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        """

        channel_data = self.get_data(start=start, stop=stop)
        if filename is None:
            filename = f"{self.name}.csv"

        if isinstance(filename, str):
            filename = Path(filename)

        if channel_data.data is None:
            df = pd.DataFrame({"time": channel_data.time_index})
        else:
            df = pd.DataFrame(
                {"time": channel_data.time_index, "data": channel_data.data}
            )
        df.to_csv(filename)

    @classmethod
    def from_dict(cls, datadict: dict[str, Any]) -> Channel:
        """Create a channel from a dictionary representation."""
        time_index = datadict.get("time_index")
        try:
            time_index = decompress_array(time_index)
        except (TypeError, ValueError, EOFError):
            pass

        data = datadict.get("data")
        try:
            data = decompress_array(data)
        except (TypeError, ValueError, EOFError):
            pass

        channel = cls(
            name=datadict.get("name"),
            time_index=time_index,
            data=data,
            time_start=datadict.get("time_start"),
            time_unit=datadict.get("time_unit"),
            offset=datadict.get("offset"),
            plotstyle=datadict.get("plotstyle"),
            metadata=datadict.get("metadata"),
        )
        for label_dict in datadict.get("labels", []):
            if label_dict.get("is_interval", False):
                label = IntervalLabel.from_dict(label_dict)
            else:
                label = Label.from_dict(label_dict)
            label.attach_to(channel)
        return channel

    def is_time_only(self) -> bool:
        """Return whether the channel contains only time data."""
        return self.data is None

    def get_data(
        self,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        resolution: Timedelta | float | str | None = None,
        include_adjacent: bool = False,
    ) -> DataSlice:
        """Return a tuple of time and data values with optional
        filtering and downsampling.

        Parameters
        ----------
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        resolution
            The resolution for the data. If specified, the data
            is downsampled such that the difference between time
            points of the downsampled data is bounded below by
            the given resolution.
            Assumes that the time index is sorted.
        include_adjacent
            If ``True``, the returned data also includes the time
            points that are adjacent to (but outside of) the start and
            stop times; useful for plotting. Defaults to ``False``.
        """

        selection = self._time_selection(
            start=start,
            stop=stop,
            resolution=resolution,
            include_adjacent=include_adjacent,
        )

        time_index = self.time_index[selection]
        if self.is_time_absolute():
            time_index += self.time_start

        data = _select(self.data, selection) if self.data is not None else None
        return DataSlice(time_index=time_index, data=data)

    def plot(
        self,
        plot_axes: plt.Axes | None = None,
        plotstyle: dict[str, Any] | None = None,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        resolution: Timedelta | float | None = None,
        time_unit: str | None = None,
        reference_time: Timestamp | Timedelta | float | None = None,
    ):
        """Plot the channel data on a given axis.

        Parameters
        ----------
        plot_axes
            The (matplotlib) axes to plot the data on. If not specified,
            a new figure will be created.
        plotstyle
            Overrides for the :attr:`plotstyle` of the channel.
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        resolution
            The resolution for the data (a duration; numbers are taken in
            the time unit of the channel). If specified, the data is
            downsampled to the minimum and the maximum value in every
            interval of this length, so peaks remain visible.
        time_unit
            The time unit values used along the x-axis. If ``None``
            (the default), the time unit of the channel is used.
        """
        from vitabel.plotting.static import plot_channel

        return plot_channel(
            self,
            plot_axes=plot_axes,
            plotstyle=plotstyle,
            start=start,
            stop=stop,
            resolution=resolution,
            time_unit=time_unit,
            reference_time=reference_time,
        )

    def rename(self, new_name: str):
        """Change the name of the channel.

        Parameters
        ----------
        new_name
            The new name of the channel.
        """
        self.name = str(new_name)


# TODO: handling of name, data, plotstyle, metadata is
# the same as for channel and could be factored out
class Label(TimeSeriesBase):
    """A time data label, holding a time series and optional data points.

    Where :class:`.Channel` is intended for data extracted from some external,
    immutable source (e.g., a sensor), :class:`.Label` is intended for data
    that is annotated or otherwise modified by the user.

    In particular, :class:`.Label` supports adding and removing data points
    via :meth:`.add_data` and :meth:`.remove_data`, respectively.

    Parameters
    ----------

    name
        The name of the label.
    time_index
        The time data of the label. Can be a list of timestamps,
        timedeltas, numeric values (interpreted with respect to the specified time unit),
        or datetime strings (which will be parsed by ``pandas``).
    data
        The data points of the label. If not specified, the label
        only holds time data. If specified, the length of the data must
        match the length of the time index. Must be numeric, strings are not allowed
        (use :attr:`text_data` instead for string data).
    text_data
        If specified, the length of the data must match the length of the time index.
        Must contain strings or ``None``.
    time_start
        If the specified ``time_index`` holds relative time data
        (timedeltas or numeric values), this parameter can be used
        to set an absolute reference time (a timestamp).
    time_unit
        The unit of the time data as a string.
    offset
        An additional time offset specified as a timedelta
        or a numeric value (which is taken with respect to
        the given time unit). The offset is applied to
        :attr:`time_index`, but when the data is exported,
        the offset is removed (and exported separately).
    anchored_channel
        The channel the label is attached to. If the offset of
        the anchored channel is changed, the offset of the label
        is changed accordingly.
    plotstyle
        A dictionary of key/value pairs that are passed to
        ``matplotlib.pyplot.plot`` when plotting label data.
    metadata
        A dictionary that can be used to store additional
        information about the label.
    plot_type
        Determines how entries of the label are plotted. Available options are:

        - ``'scatter'``: Entries of the label are plotted as points. Requires a
          label with numeric data.
        - ``'vline'``: Entries of the label are plotted as vertical lines whose
          plot labels are determined by ``vline_text_source``.
        - ``'combined'``: Entries of the label are plotted as points when there
          is associated numeric data, and as vertical lines otherwise.

        Defaults to ``'combined'``.

    vline_text_source
        When plotting label data using vertical lines, this argument
        controls how text labels for the lines are determined. Available options are:

        - ``'text_data'``: Uses strings from :attr:`text_data` as line labels.
        - ``'data'``: Uses string representations of the numeric :attr:`.data`
          values as line labels.
        - ``'combined'``: Uses texts from :attr:`text_data` where available and
          fills the rest with entries from :attr:`.data`.
        - ``'disabled'``: No text is shown next to the vertical lines.

        Defaults to ``'text_data'``. The text labels can be customized by
        modifying the dictionary stored in the :attr:`plot_vline_bbox_settings`
        attribute of the label.

    annotation_preset_type
        Specifies the initial preselection of the annotation menu checkboxes
        when a label is selected.

        This preset is only applied **if the selected label has no existing data**.
        If the label already contains data, the menu's preselection will instead
        reflect the current contents of that label.

        Available options are:

        - ``'timestamp'``: Only the *Timestamp* checkbox is selected.
        - ``'numerical'``: *Timestamp* and *Numerical value* checkboxes are selected.
        - ``'textual'``: *Timestamp* and *Textual value* checkboxes are selected.
        - ``'combined'``: All checkboxes are selected.
        - ``None`` (the default): automatically selects the checkboxes based
          on other arguements like ``plot_type``.

        .. note::

            - This setting only affects the **initial** state of the annotation
              menu—users can still modify selections interactively.
            - There is **no internal consistency check** between this setting and
              the `plot_type` or `vline_text_source` parameters. For example, you
              can disable textual input using this argument even if you intend to
              display vertical lines with text.
    """

    def __init__(
        self,
        name: str,
        time_index: npt.ArrayLike[Timestamp | Timedelta | float | str] | None = None,
        data: npt.ArrayLike[float | np.number] | None = None,
        text_data: npt.ArrayLike[str | None] | None = None,
        time_start: Timestamp | None = None,
        time_unit: str | None = None,
        offset: Timedelta | float | None = None,
        anchored_channel: Channel | None = None,
        plotstyle: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        plot_type: LabelPlotType = "combined",
        vline_text_source: LabelPlotVLineTextSource = "text_data",
        annotation_preset_type: LabelAnnotationPresetType | None = None,
    ):
        self.name = str(name)
        """The name of the label."""

        self.anchored_channel: Channel | None = None
        """The channel the label is attached to, or ``None``."""

        if time_index is None:
            time_index = np.array([])

        if data is not None:
            if len(data) > 0 and any(
                isinstance(value, (str, np.str_)) for value in data
            ):
                # legacy support for string data: pass data as text_data
                # and adjust arguments accordingly
                if text_data is not None:
                    raise ValueError("Label data is not allowed to contain strings")

                text_data = np.array(data, dtype=object)
                data = None
                logger.warning(
                    "Label data is not allowed to contain strings. "
                    "Automatically populating text_data instead of data."
                )

                if plot_type == "vline" and vline_text_source == "data":
                    vline_text_source = "text_data"
                    logger.info(
                        "Automatically changed vline_text_source to 'text_data' as well."
                    )
            elif len(data) == 0:
                data = None
            else:
                # replace None with np.nan and convert to float
                data = np.asarray(data, dtype=float)
                if data.ndim != 1:
                    raise ValueError(f"data must be a 1D array, got shape {data.shape}")

        if text_data is not None and len(text_data) > 0:
            text_data = np.asarray(text_data, dtype=object)
            if text_data.ndim != 1:
                raise ValueError(
                    f"text_data must be a 1D array, got shape {text_data.shape}"
                )
            text_data[text_data == ""] = None
        else:
            text_data = None

        self._check_data_shape(time_index, data=data, text_data=text_data)

        self.data = data
        """The data points of the label, if any."""

        self.text_data = text_data
        """Text associated to the individual time data points, if any."""

        self.plotstyle = copy(plotstyle) or {
            "marker": "o",
            "ms": 5,
            "linestyle": "none",
        }
        """Keyword arguments passed to the plotting routine."""

        self.plot_vline_bbox_settings = {
            "boxstyle": "round",
            "alpha": 0.6,
            "facecolor": "white",
            "edgecolor": "black",
        }
        """Settings for the bounding box around the vertical line text."""

        self.metadata = copy(metadata) or {}
        """Additional label metadata."""

        if plot_type is None:
            plot_type = "combined"
        self._plot_type: LabelPlotType
        self.plot_type = plot_type
        """The type of plot to use to visualize the label."""

        if vline_text_source is None:
            vline_text_source = "text_data"

        self._vline_text_source: LabelPlotVLineTextSource
        self.vline_text_source = vline_text_source
        """The source of the text to be shown next to vertical lines."""

        self._annotation_preset_type: LabelAnnotationPresetType | None
        self.annotation_preset_type = annotation_preset_type
        """The preselection of the labels annotation menu."""

        super().__init__(
            time_index=time_index,
            time_start=time_start,
            time_unit=time_unit,
            offset=offset,
        )

        if anchored_channel is not None:
            self.attach_to(anchored_channel)

    def __eq__(self, other: Label) -> bool:
        """Check whether two labels are equal.

        Returns ``True`` if the name, time index, and data of the labels
        are equal, and ``False`` otherwise.
        """
        return (
            self.name == other.name
            and np.array_equal(self.time_index, other.time_index)
            and np.array_equal(self.data, other.data)
        )

    def __repr__(self) -> str:
        """A string representation of the label."""
        return f"{self.__class__.__name__}({self.name})"

    @classmethod
    def from_channel(cls, channel: Channel) -> Label:
        """Create a label from a channel.

        Parameters
        ----------
        channel
            The channel to create the label from. The label will
            have the same name, time index, data, and text data as the channel.
        """
        return Label(
            name=channel.name,
            time_index=channel.time_index,
            data=channel.data,
            text_data=None,  # channels do not have text data
            time_start=channel.time_start,
            time_unit=channel.time_unit,
            offset=channel.offset,
            anchored_channel=None,
            plotstyle=copy(channel.plotstyle),
            metadata=copy(channel.metadata),
        )

    @property
    def first_entry(self) -> Timestamp | Timedelta | None:
        """The earliest entry in the time index of this label."""
        min_time = None
        if len(self) > 0:
            min_time = self.time_index.min()
            if self.is_time_absolute():
                min_time += self.time_start
        return min_time

    @property
    def last_entry(self) -> Timestamp | Timedelta | None:
        """The latest entry in the time index of this label."""
        max_time = None
        if len(self) > 0:
            max_time = self.time_index.max()
            if self.is_time_absolute():
                max_time += self.time_start
        return max_time

    @property
    def plot_type(self) -> LabelPlotType | None:
        """The type of plot to use to visualize the label."""
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value: LabelPlotType):
        if value not in typing.get_args(LabelPlotType):
            raise ValueError(f"Value '{value}' is not a valid choice for plot_type")
        self._plot_type = value

    @property
    def vline_text_source(self) -> LabelPlotVLineTextSource | None:
        """The source attribute of the text to be shown next to vertical lines."""
        return self._vline_text_source

    @vline_text_source.setter
    def vline_text_source(self, value: LabelPlotVLineTextSource):
        if value not in typing.get_args(LabelPlotVLineTextSource):
            raise ValueError(
                f"Value '{value}' is not a valid choice for vline_text_source."
            )
        self._vline_text_source = value

    @property
    def annotation_preset_type(self) -> LabelAnnotationPresetType | None:
        """The preselection of the labels annotation menu."""
        return self._annotation_preset_type

    @annotation_preset_type.setter
    def annotation_preset_type(self, value: LabelAnnotationPresetType | None):
        if value is not None and value not in typing.get_args(
            LabelAnnotationPresetType
        ):
            raise ValueError(
                f"Value '{value}' is not a valid choice for annotation_preset_type."
            )
        self._annotation_preset_type = value

    def _check_data_shape(
        self,
        time_index: npt.ArrayLike,
        data: npt.ArrayLike | None,
        text_data: npt.ArrayLike | None,
    ):
        """Check that the data has the same length as the time index."""
        if data is not None and len(data) != len(time_index):
            raise ValueError(
                "The length of the data must be equal to the length of the time index"
            )

        if text_data is not None and len(text_data) != len(time_index):
            raise ValueError(
                "The length of the text data must be equal to the length of the time index"
            )

    def _check_plot_parameters(
        self,
        plot_type: LabelPlotType | None = None,
        vline_text_source: LabelPlotVLineTextSource | None = None,
    ) -> bool:
        """Check whether the data provided are sufficient to draw the plot
        as requested by the specified arguments.

        Parameters
        ----------
        plot_type
            The type of plot to use to visualize the label.

        vline_text_source
            The source of the text to be shown next to vertical lines.

        Returns
        -------
        bool
            ``True`` if the current state is consistent with the specified plot configuration,
            ``False`` otherwise.
        """

        if plot_type is None:
            plot_type = self.plot_type

        if vline_text_source is None:
            vline_text_source = self.vline_text_source

        # checks for scatter
        if plot_type == "scatter":
            if self.data is None:
                logger.warning(
                    f"Cannot plot label '{self.name}': `plot_type='scatter'` requires `data` to be set."
                )
                return False
            return True

        # checks for vline
        if plot_type == "vline":
            if vline_text_source == "disabled":
                return True  # valid: vline without text

            if vline_text_source == "combined":
                return True  # valid: automatic combined source selection

            if vline_text_source == "data":
                if self.data is None:
                    logger.warning(
                        f"Conflicting plot settings for label {self.name}: "
                        f"if vline_text_source='data' there needs to be data"
                    )
                    return False
                return True

            if vline_text_source == "text_data":
                if self.text_data is None:
                    logger.warning(
                        f"Conflicting plot settings for label {self.name}: "
                        f"if vline_text_source='text_data' there needs to be text data"
                    )
                    return False
                return True

        if plot_type == "combined":
            return True

        raise ValueError(
            f"Invalid arguments for plot_type ({plot_type}) "
            f"or vline_text_source ({vline_text_source}"
        )

    def add_data(
        self,
        time_data: Timestamp | Timedelta,
        value: float | np.number | str | None = None,
        text: str | None = None,
    ):
        """Add a data point to the label.

        Parameters
        ----------
        time
            The time of the data point.
        value
            The value of the data point. If not specified,
            the value is set to ``numpy.nan``.
        text
            The string to be inserted into :attr:`.Label.text_data`.
        """
        # check corner case first: is label empty and passed time absolute?
        if self.is_empty() and isinstance(time_data, Timestamp):
            self.time_start = time_data

        if self.is_time_absolute():
            if time_data < self.time_start:
                offset = self.time_start - time_data
                self.time_start = time_data
                self.time_index += offset

            time_data = time_data - self.time_start

        if isinstance(time_data, Timedelta) and len(self.time_index) > 0:
            self.time_index = self.time_index.as_unit("ns")

        [insert_index] = self.time_index.searchsorted([time_data])
        self.time_index = self.time_index.insert(insert_index, time_data)

        if isinstance(value, str):
            # legacy support: add string data
            if text is not None:
                raise ValueError("Only numeric values can be added to data")
            logger.warning(
                "Only numeric values can be added to data. Automatically passed "
                "value to the text argument instead."
            )
            text, value = value, None

        if self.data is None:
            if value is not None:  # data not initalized but a value has to be inserted
                self.data = np.full(len(self.time_index) - 1, np.nan, dtype=float)

        if self.data is not None:
            if len(self.data) == 0:  # TODO: still needed?
                # make sure that the data attribute is a suitable numpy array
                self.data = np.array([], dtype=float)
            if value is None:
                value = np.nan

            self.data = np.insert(self.data, insert_index, value)

        if self.text_data is None:
            if (
                text is not None
            ):  # text_data not initialized but a text has to be inserted
                self.text_data = np.full(len(self.time_index) - 1, None, dtype=object)

        if self.text_data is not None:
            if len(self.text_data) == 0:  # TODO: still needed?
                # make sure that the text_data attribute is a suitable numpy array
                self.text_data = np.array([], dtype=object)

            if text == "":
                text = None

            self.text_data = np.insert(self.text_data, insert_index, text)

    def remove_data(
        self,
        time_data: Timestamp | Timedelta,
    ):
        """Remove a data point from the label given its time.

        If the data point with the earliest time is removed,
        the :attr:`time_start` attribute of the label is updated.

        Parameters
        ----------
        time_data
            The time of the data point to remove.
        """
        if self.is_time_absolute():
            time_data = time_data - self.time_start

        matches = np.argwhere(self.time_index == time_data).flatten().tolist()
        if len(matches) == 0:
            raise ValueError(f"No data point found at time {time_data}")
        remove_index = min(matches)
        self.time_index = self.time_index.delete(remove_index)
        if self.data is not None:
            self.data = np.delete(self.data, remove_index)
        if self.text_data is not None:
            self.text_data = np.delete(self.text_data, remove_index)

        if self.data is not None and np.all(np.isnan(self.data)):
            self.data = None

        if self.text_data is not None and np.all(pd.isna(self.text_data)):
            self.text_data = None

        if remove_index == 0 and self.is_time_absolute():
            if self.is_empty():
                self.time_start = None
            else:
                offset = self.time_index[0]
                self.time_start += offset
                self.time_index -= offset

    def truncate(
        self,
        start_time: Timestamp | Timedelta | None = None,
        stop_time: Timestamp | Timedelta | None = None,
    ) -> Label:
        """Return a new label that is a truncated version of this label.

        Parameters
        ----------
        start_time
            The start time for the truncated label.
        stop_time
            The stop time for the truncated label.
        """
        label_data = self.get_data(start=start_time, stop=stop_time)
        truncated_time_index = label_data.time_index
        truncated_data = label_data.data
        truncated_text_data = label_data.text_data

        truncated_label = Label(
            name=self.name,
            time_index=truncated_time_index,
            data=truncated_data,
            text_data=truncated_text_data,
            time_unit=self.time_unit,
            plotstyle=copy(self.plotstyle),
            metadata=copy(self.metadata),
            plot_type=self.plot_type,
            vline_text_source=self.vline_text_source,
        )
        return truncated_label

    @classmethod
    def from_dict(cls, datadict: dict[str, Any]) -> Label:
        """Create a label from a serialized dictionary representation.

        Parameters
        ----------
        datadict
            The dictionary representation of the label.
        """
        time_index = datadict.get("time_index")
        try:
            time_index = decompress_array(time_index)
        except (TypeError, ValueError, EOFError):
            pass

        data = datadict.get("data")
        try:
            data = decompress_array(data)
        except (TypeError, ValueError, EOFError):
            pass

        text_data = datadict.get("text_data")
        try:
            text_data = decompress_array(text_data)
        except (TypeError, ValueError, EOFError):
            pass

        return cls(
            name=datadict.get("name"),
            time_index=time_index,
            data=data,
            text_data=text_data,
            time_start=datadict.get("time_start"),
            time_unit=datadict.get("time_unit"),
            offset=datadict.get("offset"),
            plotstyle=datadict.get("plotstyle"),
            metadata=datadict.get("metadata"),
            plot_type=datadict.get("plot_type"),
            vline_text_source=datadict.get("vline_text_source"),
        )

    def to_dict(self) -> dict[str, Any]:
        """A serialization of the label as a dictionary."""
        numeric_offset = self.offset / pd.to_timedelta(1, unit=self.time_unit)
        numeric_time = self.numeric_time() - numeric_offset

        return {
            "name": self.name,
            "time_index": numeric_time,
            "data": self.data,
            "text_data": self.text_data,
            "time_start": str(self.time_start) if self.time_start is not None else None,
            "time_unit": self.time_unit,
            "offset": numeric_offset,
            "is_interval": False,
            "plotstyle": self.plotstyle,
            "metadata": self.metadata,
            "plot_type": self.plot_type,
            "vline_text_source": self.vline_text_source,
        }

    def to_csv(
        self,
        filename: str | Path | None = None,
        start: Timestamp | Timedelta | None = None,
        stop: Timestamp | Timedelta | None = None,
    ) -> None:
        """Export the label data to a CSV file.

        Parameters
        ----------
        filename
            The name of the file to export the data to. If not
            specified, the data is exported to a file with the
            name of the channel.
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        """
        time_index, data, text_data = self.get_data(start=start, stop=stop)
        if filename is None:
            filename = f"{self.name}.csv"

        if isinstance(filename, str):
            filename = Path(filename)

        df_dict = {"time": time_index}
        if data is not None:
            df_dict["data"] = data
        if text_data is not None:
            df_dict["text_data"] = text_data

        df = pd.DataFrame(df_dict)
        df.to_csv(filename)

    def attach_to(self, channel: Channel):
        """Attach the label to a channel.

        Parameters
        ----------
        channel
            The channel to attach the label to.
        """
        channel.attach_label(self)
        self.anchored_channel = channel

    def detach(self):
        """Detach the label from the channel.

        .. note::

            This method only removes the reference from the channel
            to the label, as well as the backreference from the label
            to the channel. While this does not delete the label
            object itself, the responsibility to manage the label
            object is with the user.

        """
        if self.anchored_channel is None:
            raise ValueError(f"The label {self.name} is not attached to any channel")
        self.anchored_channel.detach_label(self)

    def get_data(
        self,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        include_adjacent: bool = False,
    ) -> DataSlice:
        """Return a tuple of time, data, and text data values with optional
        filtering.

        Parameters
        ----------
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        include_adjacent
            If ``True``, the returned data also includes the time
            points that are adjacent to (but outside of) the start and
            stop times; useful for plotting. Defaults to ``False``.
        """

        selection = self._time_selection(
            start=start, stop=stop, resolution=None, include_adjacent=include_adjacent
        )

        time_index = self.time_index[selection]
        if self.is_time_absolute():
            time_index += self.time_start

        data = None
        if self.data is not None and _is_nonempty_selection(selection):
            data = _select(self.data, selection)

        text_data = None
        if self.text_data is not None and _is_nonempty_selection(selection):
            text_data = _select(self.text_data, selection)

        return DataSlice(time_index=time_index, data=data, text_data=text_data)

    def plot(
        self,
        plot_axes: plt.Axes | None = None,
        plotstyle: dict[str, Any] | None = None,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        time_unit: str | None = None,
        reference_time: Timestamp | Timedelta | float | None = None,
        plot_type: LabelPlotType | None = None,
        vline_text_source: LabelPlotVLineTextSource | None = None,
    ):
        """Plot the label data.

        Parameters
        ----------
        plot_axes
            The (matplotlib) axes used to plot the data on. If not specified,
            a new figure will be created.
        plotstyle
            Overrides for the :attr:`plotstyle` of the label.
        start
            The start time for the data. If not specified, the data
            starts from the first time point.
        stop
            The start time for the data. If not specified, the data
            stops at the last time point.
        time_unit
            The time unit values used along the x-axis. If ``None``
            (the default), the time unit of the channel is used.
        plot_type
            Override for :attr:`.plot_type`, see :class:`.Label` for details.
            If ``None`` (the default) is passed, the value from the attribute
            is used.
        vline_text_source
            Override for :attr:`.vline_text_source`, see :class:`.Label` for details.
            If ``None`` (the default) is passed, the value from the attribute
            is used.

        Notes
        -----
        If ``plot_type`` and/or ``vline_text_source`` are inconsistent with the available
        (text) data, the routine will emit warnings.
        """
        from vitabel.plotting.static import plot_label

        return plot_label(
            self,
            plot_axes=plot_axes,
            plotstyle=plotstyle,
            start=start,
            stop=stop,
            time_unit=time_unit,
            reference_time=reference_time,
            plot_type=plot_type,
            vline_text_source=vline_text_source,
        )

    def rename(self, new_name: str):
        """Change the name of the label.

        Parameters
        ----------
        new_name
            The new name of the label.
        """
        self.name = str(new_name)


class IntervalLabel(Label):
    """A special type of label that holds time interval data.

    Parameters
    ----------

    name
        The name of the label.
    time_index
        The time data of the label, interpreted as a alternating
        sequence of interval start and end points. Must have even length.
        Can be a list of timestamps,
        timedeltas, numeric values (interpreted with respect to the specified time unit),
        or datetime strings (which will be parsed by ``pandas``).
    data
        The data points of the label. If not specified, the label
        only holds time data. If specified, the length of the data must
        match the number of time intervals. Must be numeric, strings are
        not allowed (use :attr:`text_data` instead for string data).
    text_data
        If specified, the length of the data must match the number of time intervals.
        Must contain strings or ``None``.
    time_start
        If the specified ``time_index`` holds relative time data
        (timedeltas or numeric values), this parameter can be used
        to set an absolute reference time (a timestamp).
    time_unit
        The unit of the time data as a string.
    offset
        An additional time offset specified as a timedelta
        or a numeric value (which is taken with respect to
        the given time unit). The offset is applied to
        :attr:`time_index`, but when the data is exported,
        the offset is removed (and exported separately).
    anchored_channel
        The channel the label is attached to. If the offset of
        the anchored channel is changed, the offset of the label
        is changed accordingly.
    plotstyle
        A dictionary of key/value pairs that are passed to
        ``matplotlib.pyplot.errorbar`` when plotting label data.
    metadata
        A dictionary that can be used to store additional
        information about the label.
    plot_type
        Determines how entries of the label are plotted. Available options are:

        - ``'box'``: Entries of the label are plotted as shaded areas.
          Requires no additional data.
        - ``'hline'``: Entries of the label are plotted as horizontal lines.
          Requires a label with numeric data.
        - ``'combined'``: Entries of the label are plotted as horizontal lines
          when there is associated numeric data, and as shaded areas otherwise.

        Defaults to ``'combined'``.

    annotation_preset_type
        Specifies the initial preselection of the annotation menu checkboxes
        when a label is selected.

        This preset is only applied **if the selected label has no existing data**.
        If the label already contains data, the menu's preselection will instead
        reflect the current contents of that label.

        Available options are:

        - ``'timestamp'``: Only the *Timestamp* checkbox is selected.
        - ``'numerical'``: *Timestamp* and *Numerical value* checkboxes are selected.
        - ``'textual'``: *Timestamp* and *Textual value* checkboxes are selected.
        - ``'combined'``: All checkboxes are selected.
        - ``None`` (the default): automatically selects the checkboxes based
          on other arguements like ``plot_type``.

        .. note::

            - This setting only affects the **initial** state of the annotation
              menu—users can still modify selections interactively.
            - There is **no internal consistency check** between this setting and
              the `plot_type` or `vline_text_source` parameters. For example, you
              can disable textual input using this argument even if you intend to
              display vertical lines with text.
    """

    def __init__(
        self,
        name: str,
        time_index: npt.ArrayLike[Timestamp | Timedelta | float | str] | None = None,
        data: npt.ArrayLike[float | np.number] | None = None,
        text_data: npt.ArrayLike[str | None] | None = None,
        time_start: Timestamp | None = None,
        time_unit: str | None = None,
        offset: Timedelta | float | None = None,
        anchored_channel: Channel | None = None,
        plotstyle: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        plot_type: IntervalLabelPlotType = "combined",
        annotation_preset_type: LabelAnnotationPresetType | None = None,
    ):
        if time_index is not None:
            time_index = np.array(time_index)
            if time_index.ndim == 2 and time_index.shape[1] == 2:
                # time data passed as pairs of interval end points
                time_index = time_index.reshape(-1)
        super().__init__(
            name=name,
            time_index=time_index,
            data=data,
            text_data=text_data,
            time_start=time_start,
            time_unit=time_unit,
            offset=offset,
            anchored_channel=anchored_channel,
            plotstyle=plotstyle,
            metadata=metadata,
            annotation_preset_type=annotation_preset_type,
        )

        if plot_type is None:
            plot_type = "combined"
        self._plot_type: IntervalLabelPlotType
        self.plot_type = plot_type

    @property
    def plot_type(self) -> IntervalLabelPlotType:  # type: ignore[override]
        """The type of plot to use to visualize the label."""
        return self._plot_type

    @plot_type.setter
    def plot_type(self, value: IntervalLabelPlotType):  # type: ignore[override]
        if value not in typing.get_args(IntervalLabelPlotType):
            raise ValueError(f"Value '{value}' is not a valid choice for plot_type")
        self._plot_type = value  # type: ignore[override]

    def _check_data_shape(
        self,
        time_index: npt.ArrayLike,
        data: npt.ArrayLike | None,
        text_data: npt.ArrayLike | None,
    ):
        """Check that the time data is well-formed, and that there is either no data,
        or as many data points as intervals.
        """
        if len(time_index) % 2 != 0:
            raise ValueError("The time index must contain an even number of elements")

        if data is not None and len(time_index) != 2 * len(data):
            raise ValueError(
                "The length of the data must be half the length of the time index"
            )

        if data is not None and len(time_index) != 2 * len(data):
            raise ValueError(
                "The length of the data must be half the length of the time index"
            )

        if text_data is not None and len(time_index) != 2 * len(text_data):
            raise ValueError(
                "The length of the text data must be half the length of the time index"
            )

    def __len__(self) -> int:
        """The number of intervals in the label."""
        return super().__len__() // 2

    @property
    def intervals(self) -> npt.NDArray:
        """A 2D array of interval start and end points representing
        the time intervals of the label. If the label is absolute in time,
        the intervals are timestamps. Otherwise, they are timedeltas.
        """
        time_intervals = np.array(self.time_index).reshape(-1, 2)
        if self.is_time_absolute():
            time_intervals = time_intervals + self.time_start
        return time_intervals

    def truncate(
        self,
        start_time: Timestamp | Timedelta | None = None,
        stop_time: Timestamp | Timedelta | None = None,
    ) -> IntervalLabel:
        """Return a new label that is a truncated version of this label.

        Parameters
        ----------
        start_time
            The start time for the truncated label.
        stop_time
            The stop time for the truncated label.
        """

        truncated_time_index, truncated_data, truncated_text_data = self.get_data(
            start=start_time, stop=stop_time
        )
        truncated_label = IntervalLabel(
            name=self.name,
            time_index=truncated_time_index.flatten(),
            data=truncated_data,
            text_data=truncated_text_data,
            time_unit=self.time_unit,
            plotstyle=copy(self.plotstyle),
            metadata=copy(self.metadata),
            plot_type=self.plot_type,
        )
        return truncated_label

    @classmethod
    def from_dict(cls, datadict: dict[str, Any]) -> Label:
        """Create a IntervalLabel from a serialized dictionary representation.

        Parameters
        ----------
        datadict
            The dictionary representation of the label.

        """
        if not datadict.get("is_interval", False):
            raise ValueError("Cannot create IntervalLabel from non-interval label data")

        time_index = datadict.get("time_index")
        try:
            time_index = decompress_array(time_index)
        except (TypeError, ValueError, EOFError):
            pass

        data = datadict.get("data")
        try:
            data = decompress_array(data)
        except (TypeError, ValueError, EOFError):
            pass

        text_data = datadict.get("text_data")
        try:
            text_data = decompress_array(text_data)
        except (TypeError, ValueError, EOFError):
            pass

        return cls(
            name=datadict.get("name"),
            time_index=time_index,
            data=data,
            text_data=text_data,
            time_start=datadict.get("time_start"),
            time_unit=datadict.get("time_unit"),
            offset=datadict.get("offset"),
            plotstyle=datadict.get("plotstyle"),
            metadata=datadict.get("metadata"),
            plot_type=datadict.get("plot_type"),
        )

    def to_dict(self) -> dict[str, Any]:
        """A serialization of the label as a dictionary."""
        label_dict = super().to_dict()
        label_dict["is_interval"] = True
        return label_dict

    def to_csv(
        self,
        filename: str | Path | None = None,
        start: Timestamp | Timedelta | None = None,
        stop: Timestamp | Timedelta | None = None,
    ) -> None:
        """Export the interval label data to a CSV file.

        Contains all intervals with a non-empty intersection
        of the specified data range.

        Parameters
        ----------
        filename
            The name of the file to export the data to. If not
            specified, the data is exported to a file with the
            name of the channel.
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.

        """
        time_index, data, text_data = self.get_data(start=start, stop=stop)
        time_start, time_stop = time_index.transpose()

        if filename is None:
            filename = f"{self.name}.csv"

        if isinstance(filename, str):
            filename = Path(filename)

        csv_data_dict = {
            "time_start": time_start,
            "time_stop": time_stop,
        }
        if data is not None:
            csv_data_dict["data"] = data
        if text_data is not None:
            csv_data_dict["text_data"] = text_data

        df = pd.DataFrame(csv_data_dict)
        df.to_csv(filename)

    def get_data(
        self,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
    ) -> DataSlice:
        """Return a tuple of interval endpoints and data values with optional
        filtering. This returns all intervals that intersect with the
        specified time range, shortening the intervals if necessary.

        Parameters
        ----------
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        """
        if self.is_empty() or (start is None and stop is None):
            return DataSlice(
                time_index=self.intervals, data=self.data, text_data=self.text_data
            )

        if start is None:
            start = self.time_index.min()

        if stop is None:
            stop = self.time_index.max()

        start = self.convert_time_input(start)
        stop = self.convert_time_input(stop)

        start_points, end_points = self.intervals.transpose()
        time_mask = ~(
            ((start_points < start) & (end_points < start))
            | ((start_points > stop) & (end_points > stop))
        )

        intervals = self.intervals[time_mask]
        # TODO: the time intervals should be clipped to the start and stop
        # times if they are not fully contained in the range
        data = self.data[time_mask] if self.data is not None else None
        text_data = self.text_data[time_mask] if self.text_data is not None else None

        # destructing zero length arrays
        data = None if data is not None and len(data) == 0 else data
        text_data = None if text_data is not None and len(text_data) == 0 else text_data

        return DataSlice(time_index=intervals, data=data, text_data=text_data)

    def add_data(
        self,
        time_data: tuple[Timestamp, Timestamp] | tuple[Timedelta, Timedelta],
        value: float | np.number | str | None = None,
        text: str | None = None,
    ):
        """Add data to the label.

        Parameters
        ----------
        time_data
            A time data tuple specifying an interval.
        value
            The numeric value of the data point. If not specified,
            the value is set to ``numpy.nan``.
        text
            The string to be inserted into :attr:`text_data`.
        """
        interval_start, interval_end = time_data
        if self.is_empty() and isinstance(interval_start, Timestamp):
            self.time_start = pd.to_datetime(interval_start)

        if self.is_time_absolute():
            interval_start -= self.time_start
            interval_end -= self.time_start

        # TODO: time_start currently does not reset when corresponding
        # first data point is removed, time index is not necessarily monotonous

        self.time_index = self.time_index.append(
            pd.TimedeltaIndex([interval_start, interval_end])
        )

        if isinstance(value, str):
            # legacy support: add string data
            if text is not None:
                raise ValueError("Only numeric values can be added to data")
            logger.warning(
                "Only numeric values can be added to data. Automatically passed "
                "value to the text argument instead."
            )
            text, value = value, None

        if self.data is None:
            if value is not None:  # data has to be initialized
                self.data = np.full(len(self) - 1, np.nan, dtype=float)

        if self.data is not None:
            if len(self.data) == 0:
                # make sure that the data attribute is a suitable numpy array
                self.data = np.array([], dtype=float)

            if value is None:
                value = np.nan

            self.data = np.append(self.data, value)

        if self.text_data is None:
            if (
                text is not None
            ):  # text_data not initialized but a text has to be inserted
                self.text_data = np.full(len(self) - 1, None, dtype=object)

        if self.text_data is not None:
            if len(self.text_data) == 0:
                # make sure that the text_data attribute is a suitable numpy array
                self.text_data = np.array([], dtype=object)

            if text == "":
                text = None

            self.text_data = np.append(self.text_data, text)

    def remove_data(
        self,
        time_data: tuple[Timestamp, Timestamp] | tuple[Timedelta, Timedelta],
    ):
        """Remove data from the label.

        Parameters
        ----------
        time_data
            A time data tuple specifying an interval.
        """
        interval_start, interval_end = time_data
        matching_intervals = (
            np.argwhere(
                (self.intervals[:, 0] == interval_start)
                & (self.intervals[:, 1] == interval_end)
            )
            .flatten()
            .tolist()
        )
        if len(matching_intervals) == 0:
            raise ValueError(f"No interval with endpoints {time_data} present in label")
        remove_index = min(matching_intervals)

        # TODO: adjust time_start if necessary (remove_index == 0)
        self.time_index = self.time_index.delete(
            [2 * remove_index, 2 * remove_index + 1]
        )
        if self.data is not None:
            self.data = np.delete(self.data, remove_index)
        if self.text_data is not None:
            self.text_data = np.delete(self.text_data, remove_index)

        if self.data is not None and np.all(np.isnan(self.data)):
            self.data = None

        if self.text_data is not None and np.all(pd.isna(self.text_data)):
            self.text_data = None

        if remove_index == 0 and self.is_time_absolute():
            if self.is_empty():
                self.time_start = None
            else:
                offset = self.time_index[0]
                self.time_start += offset
                self.time_index -= offset

        return

    def contains_time(
        self,
        time_data: npt.ArrayLike | Timestamp | Timedelta,
        include_start: bool = True,
        include_end: bool = True,
        interval_require_full_containment: bool = False,
    ) -> npt.NDArray[np.bool_]:
        """Check which of the specified timestamps or timedeltas are
        contained in any of the intervals of the label.

        If the specified time data is provided in a array of shape ``(n, 2)``,
        each row is interpreted as a start and end point of a time interval,
        and the containment check is performed for the entire interval.

        Parameters
        ----------
        time_data
            A list of timestamps or timedeltas, or an array with time interval
            endpoints to be checked.
        include_start
            If ``True``, the start point of each interval is considered
            to be contained in the interval. Defaults to ``True``.
        include_end
            If ``True``, the end point of each interval is considered
            to be contained in the interval. Defaults to ``True``.
        interval_require_full_containment
            If ``True``, the specified time intervals are only considered
            to be contained in the label if they are fully contained
            in any of the label's intervals. Defaults to ``False``.
            Only has an effect if interval data is provided to ``time_data``.

        Returns
        -------
        A boolean array indicating for each of the specified time points
        whether it is contained in any of the intervals of the label.
        """

        time_data = _as_time_array(time_data)
        start_cmp = np.greater_equal if include_start else np.greater
        end_cmp = np.less_equal if include_end else np.less

        if time_data.ndim == 2 and time_data.shape[1] == 2:
            # time interval data provided
            time_data_start = time_data[:, 0]
            time_data_end = time_data[:, 1]
            contained_mask = np.full(time_data_start.shape, False, dtype=bool)

            if interval_require_full_containment:
                for start, end in self.intervals:
                    contained_mask |= start_cmp(time_data_start, start) & end_cmp(
                        time_data_end, end
                    )
                return contained_mask

            return self.contains_time(
                time_data_start, include_start=include_start, include_end=include_end
            ) | self.contains_time(
                time_data_end, include_start=include_start, include_end=include_end
            )

        contained_mask = np.full(time_data.shape, False, dtype=bool)
        for start, end in self.intervals:
            contained_mask |= start_cmp(time_data, start) & end_cmp(time_data, end)
        return contained_mask

    def plot(
        self,
        plot_axes: plt.Axes | None = None,
        plotstyle: dict[str, Any] | None = None,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        time_unit: str | None = None,
        reference_time: Timestamp | Timedelta | float | None = None,
        plot_type: IntervalLabelPlotType | None = None,
    ):
        """Plot the label data using an error bar or a
        filled background region.

        If the label has (numeric) data, the intervals are plotted
        as error bars on the height of the data points. If there
        is no data, the intervals are plotted as a filled background region.

        Uses ``matplotlib.pyplot.errorbar`` or ``matplotlib.pyplot.axvspan``.

        Parameters
        ----------
        plot_axes
            The (matplotlib) axes used to plot the data on. If not specified,
            a new figure will be created.
        plotstyle
            Overrides for the :attr:`plotstyle` of the label.
        start
            The start time for the data. If not specified, the data
            starts from the first time point.
        stop
            The start time for the data. If not specified, the data
            stops at the last time point.
        time_unit
            The time unit values used along the x-axis. If ``None``
            (the default), the time unit of the channel is used.
        reference_time
            For labels with absolute time, a reference time to subtract from
            the time data. If not specified, the :attr:`time_start` attribute is used.
        """
        from vitabel.plotting.static import plot_interval_label

        return plot_interval_label(
            self,
            plot_axes=plot_axes,
            plotstyle=plotstyle,
            start=start,
            stop=stop,
            time_unit=time_unit,
            reference_time=reference_time,
            plot_type=plot_type,
        )


class TimeDataCollection:
    """A collection of channels and labels.

    Parameters
    ----------
    channels
        A list of data channels.
    labels
        A list of labels. Labels that are anchored
        to a passed channel do not need to be passed
        separately, they are part of the collection
        automatically.
    """

    def __init__(
        self,
        channels: list[Channel] | None = None,
        labels: list[Label] | None = None,
    ):
        channels = channels or []
        labels = labels or []

        data = channels + labels
        if data:
            if any(series.time_start is not None for series in data) and not all(
                series.is_empty() or series.time_start is not None for series in data
            ):
                raise ValueError(
                    "All time data in the collection must be either absolute or relative"
                )

        self.channels: list[Channel] = channels
        self.global_labels: list[Label] = []
        for label in labels:
            if label.anchored_channel is None:
                self.global_labels.append(label)
            else:
                if label.anchored_channel not in self.channels:
                    raise ValueError(
                        f"Label {label.name} is anchored to a channel that is not in the collection"
                    )

    def __eq__(self, other: TimeDataCollection) -> bool:
        """Check whether two collections are equal.

        Returns ``True`` if the channel data hashes and labels of the
        collections are equal, and ``False`` otherwise.
        """
        return (
            self.channel_data_hash() == other.channel_data_hash()
            and self.labels == other.labels
        )

    def __repr__(self) -> str:
        """A string representation of the collection."""
        num_channels = len(self.channels)
        channel_str = "channel" if num_channels == 1 else "channels"
        num_labels = len(self.labels)
        label_str = "label" if num_labels == 1 else "labels"
        num_labels_local = len(self.local_labels)
        num_labels_global = len(self.global_labels)
        return (
            f"{self.__class__.__name__}"
            f"({num_channels} {channel_str}, {num_labels} {label_str} "
            f"[{num_labels_local} local, {num_labels_global} global])"
        )

    @property
    def local_labels(self) -> list[Label]:
        """All labels anchored to some channel in the collection."""
        return [label for channel in self.channels for label in channel.labels]

    @property
    def labels(self) -> list[Label]:
        """All labels in the collection, both global and local ones."""
        return self.global_labels + self.local_labels

    def is_empty(self) -> bool:
        """Return whether the collection is empty."""
        return not self.channels and not self.global_labels

    def is_time_absolute(self) -> bool:
        """Return whether the collection contains only absolute time data."""
        return all(
            series.is_time_absolute() or series.is_empty()
            for series in self.channels + self.global_labels
        )

    def is_time_relative(self) -> bool:
        """Return whether the collection contains only relative time data."""
        return all(
            series.is_time_relative() or series.is_empty()
            for series in self.channels + self.global_labels
        )

    def print_summary(self) -> None:
        """Print a summary of channels and labels in the collection."""
        if self.is_empty():
            print("The collection is empty.")

        if self.channels:
            print("Channels:")
            for idx, channel in enumerate(self.channels):
                channel_string = f"{idx: 4} | {channel.name}"
                if channel.labels:
                    channel_string += f" ({len(channel.labels)} attached label{'s' if len(channel.labels) != 1 else ''})"
                print(channel_string)

            if self.labels:
                print()

        if self.labels:
            print("Labels:")
            for idx, label in enumerate(self.labels):
                label_string = f"{idx: 4} | {label.name}"
                if label.anchored_channel is not None:
                    label_string += f" (@ {label.anchored_channel.name})"
                label_string += f", {label.__class__.__name__}"
                if label.data is None:
                    label_string += ", time only"
                print(label_string)

    def add_channel(self, channel: Channel):
        """Add a channel to the collection.

        Parameters
        ----------
        channel
            The channel to add.
        """
        if channel in self.channels:
            raise ValueError(
                f"Identical channel {channel.name} has already "
                "been added to the collection"
            )
        if (
            not self.is_empty()
            and not channel.is_empty()  # empty channel will always be relative
            and self.is_time_absolute() != channel.is_time_absolute()
        ):
            raise ValueError(
                f"The time type (absolute or relative) of the channel {channel.name} "
                "does not match the time type of the collection"
            )
        self.channels.append(channel)

    def add_global_label(self, label: Label):
        """Add a label to the collection.

        Parameters
        ----------
        label
            The label to add.
        """
        if label.anchored_channel is not None:
            raise ValueError(
                f"Label {label.name} is attached to channel {label.anchored_channel.name} "
                "and cannot be added as a global label"
            )
        if label in self.global_labels:
            raise ValueError(
                f"Identical label {label.name} has already been added to the collection"
            )
        if (
            not self.is_empty()
            and not label.is_empty()
            and self.is_time_absolute() != label.is_time_absolute()
        ):
            raise ValueError(
                f"The time type (absolute or relative) of the label {label.name} "
                "does not match the time type of the collection"
            )
        self.global_labels.append(label)

    def detach_label_from_channel(
        self,
        *,
        label: Label | str,
        channel: Channel | str | None = None,
        reattach_as_global: bool = True,
    ) -> Label:
        """Detach a label from a channel in the collection.

        Parameters
        ----------
        label
            The label to detach. Can be specified either as a
            :class:`.Label` object or by its name.
        channel
            The channel to detach the label from or ``None`` (the default)
            if the channel should be determined from the label.
            Can be specified either as a :class:`.Channel` object or
            by its name.
        reattach_as_global
            If ``True``, the label is reattached as a global label
            after detaching it from the channel. If ``False``, the
            label is removed from the collection.
        """
        if isinstance(label, str):
            label = self.get_label(name=label)

        if channel is None:
            channel = label.anchored_channel
        elif isinstance(channel, str):
            channel = self.get_channel(name=channel)

        channel.detach_label(label)
        if reattach_as_global:
            self.add_global_label(label)

        return label

    def get_channels(self, name: str | None = None, **kwargs) -> list[Channel]:
        """Return a list of channels.

        Parameters
        ----------
        name
            The name of the channel to retrieve. Allowed to be passed
            either as a positional or a keyword argument.
        kwargs
            Keyword arguments to filter the channels by. The
            specified arguments are compared to the attributes
            of the channels.
        """
        if name is not None:
            kwargs["name"] = name
        return [channel for channel in self.channels if match_object(channel, **kwargs)]

    def get_channel(self, name: str | int | None = None, **kwargs) -> Channel:
        """Return a channel by name.

        Raises an error if no unique channel is found.

        Parameters
        ----------
        name
            The name of the channel to retrieve. Allowed to be passed
            either as a positional or a keyword argument. If an integer
            is passed, it is interpreted as the index of the channel
            in the collection.
        kwargs
            Keyword arguments to filter the channels by.
        """
        if name is not None:
            if isinstance(name, int):
                return self.channels[name]
            kwargs["name"] = name
        channels = self.get_channels(**kwargs)
        if len(channels) != 1:
            raise ValueError(
                "Channel specification was ambiguous, no unique channel "
                f"was identified. Query for {kwargs} returned: {channels}"
            )
        return channels[0]

    def channel_data_hash(self) -> str:
        """Return a hash representing the data and metadata of all channels
        in this collection.
        """
        data = [channel.data_hash() for channel in self.channels]
        data_buffer = json.dumps(data).encode("utf-8")
        return hashlib.sha256(data_buffer).hexdigest()

    def get_labels(self, name: str | None = None, **kwargs) -> list[Label]:
        """Return a list of labels.

        Parameters
        ----------
        name
            The name of the label to retrieve. Allowed to be passed
            either as a positional or a keyword argument.
        kwargs
            Keyword arguments to filter the labels by. The
            specified arguments are compared to the attributes
            of the labels.
        """
        if name is not None:
            kwargs["name"] = name
        return [label for label in self.labels if match_object(label, **kwargs)]

    def get_label(self, name: str | int | None = None, **kwargs) -> Label:
        """Return a label by name.

        Raises an error if no unique label is found.

        Parameters
        ----------
        name
            The name of the label to retrieve. Allowed to be passed
            either as a positional or a keyword argument. If an integer
            is passed, it is interpreted as the index of the label
            in the collection.
        kwargs
            Keyword arguments to filter the labels by. The
            specified arguments are compared to the attributes
            of the labels.
        """
        if name is not None:
            if isinstance(name, int):
                return self.labels[name]
            kwargs["name"] = name
        labels = self.get_labels(**kwargs)
        if len(labels) != 1:
            raise ValueError(
                "Label specification was ambiguous, no unique label "
                f"was identified. Query for {kwargs} returned: {labels}"
            )
        return labels[0]

    def remove_label(self, *, label: Label | None = None, **kwargs) -> Label:
        """Remove a local or global label from the collection.

        Local labels are removed by detaching them from their
        corresponding channel.

        Parameters
        ----------
        label
            The label object to delete, optional. Alternatively,
            the label can also be specified by keyword arguments
            as in :meth:`.get_label`.
        """
        if label is not None:
            if label not in self.labels:
                raise ValueError("The specified label is not in the collection")
        else:
            label = self.get_label(**kwargs)

        if label.anchored_channel is not None:
            label.detach()
        else:
            self.global_labels.remove(label)
        return label

    def remove_channel(self, *, channel: Channel | None = None, **kwargs) -> Channel:
        """Remove a channel by name.

        Parameters
        ----------
        channel
            The channel object to delete, optional. Alternatively,
            the channel can also be specified by keyword arguments
            as in :meth:`.get_channel`.
        kwargs
            The name of the channel to delete.
        """
        if channel is not None:
            if channel not in self.channels:
                raise ValueError("The specified channel is not in the collection")
        else:
            channel = self.get_channel(**kwargs)

        self.channels.remove(channel)
        return channel

    def set_channel_plotstyle(
        self, channel_specification: ChannelSpecification | None = None, **kwargs
    ):
        """Set the plot style for a channel.

        Parameters
        ----------
        channel_specification
            A specification of all channels to set the plot style for.
            See :meth:`.get_channels` for valid specifications.
        **kwargs
            The plot style properties to set. Passing ``None``
            unsets the key from the plotstyle dictionary.
        """
        if channel_specification is None:
            channel_specification = {}
        elif isinstance(channel_specification, str):
            channel_specification = {"name": channel_specification}

        if isinstance(channel_specification, dict):
            channels = self.get_channels(**channel_specification)
        else:
            channels = [channel_specification]

        for channel in channels:
            channel.plotstyle.update(kwargs)
            channel.plotstyle = {
                k: v for k, v in channel.plotstyle.items() if v is not None
            }

    def set_label_plotstyle(
        self, label_specification: LabelSpecification | None = None, **kwargs
    ):
        """Set the plot style for specified labels.

        Parameters
        ----------
        label_specification
            A specification of all labels to set the plot style for.
            See :meth:`.get_labels` for valid specifications.
        **kwargs
            The plot style properties to set. Passing ``None``
            unsets the key from the plotstyle dictionary.
        """
        if label_specification is None:
            label_specification = {}
        elif isinstance(label_specification, str):
            label_specification = {"name": label_specification}

        if isinstance(label_specification, dict):
            labels = self.get_labels(**label_specification)
        else:
            labels = [label_specification]

        for label in labels:
            label.plotstyle.update(kwargs)
            label.plotstyle = {
                k: v for k, v in label.plotstyle.items() if v is not None
            }

    def to_dict(self) -> dict[str, Any]:
        """Construct a serializable dictionary that represents
        this collection."""
        return {
            "channels": [channel.to_dict() for channel in self.channels],
            "labels": [label.to_dict() for label in self.global_labels],
        }

    @classmethod
    def from_dict(cls, datadict: dict[str, Any]) -> TimeDataCollection:
        """Create a collection from a dictionary representation.

        Parameters
        ----------
        datadict
            A dictionary representation of the collection.
        """
        channels = [
            Channel.from_dict(channel_dict) for channel_dict in datadict["channels"]
        ]
        labels = []
        for label_dict in datadict["labels"]:
            if label_dict.get("is_interval"):
                label = IntervalLabel.from_dict(label_dict)
            else:
                label = Label.from_dict(label_dict)
            labels.append(label)
        return cls(channels=channels, labels=labels)

    def _parse_time(
        self,
        time_spec: Timestamp | Timedelta | float | str | None,
    ) -> Timedelta | Timestamp | float | None:
        """Parse a time specification into a timestamp or timedelta.

        Parameters
        ----------
        time_spec
            The time specification to parse.
        """
        if isinstance(time_spec, str):
            if self.is_time_absolute():
                return pd.to_datetime(time_spec)
            else:
                return pd.to_timedelta(time_spec)
        elif isinstance(time_spec, (Timedelta, Timestamp, float)) or time_spec is None:
            return time_spec
        raise ValueError(f"Time specification {time_spec} could not be parsed")

    def plot(
        self,
        channels: list[list[ChannelSpecification | int]] | None = None,
        labels: list[list[LabelSpecification | int]] | None = None,
        start: Timestamp | Timedelta | float | str | None = None,
        stop: Timestamp | Timedelta | float | str | None = None,
        resolution: Timedelta | float | None = None,
        time_unit: str | None = None,
        include_attached_labels: bool = False,
        subplots_kwargs: dict[str, Any] | None = None,
    ):
        """Plot the data in the collection.

        Parameters
        ----------
        channels
            The channels to plot. If not specified, all channels are plotted.
            Specified as a list of lists with individual lists containing
            channels to be collected in one subplot.
        labels
            The labels to plot. If not specified, all labels are plotted.
            Specified as a list of lists, same as for the channels.
        start
            The start time for the plot. If not specified, the plot starts
            from the first time point.
        stop
            The stop time for the plot. If not specified, the plot stops
            at the last time point.
        resolution
            The resolution of the plot in the time unit of the channels.
            If not specified, the channel and label data is not downsampled.
        time_unit
            The time unit in which channel and label data are represented
            in. If not specified, the time unit of the channels is used.
        include_attached_labels
            Whether to automatically include labels attached to the
            specified channels.
        subplots_kwargs
            Keyword arguments passed to ``matplotlib.pyplot.subplots``.
        """
        from vitabel.plotting.static import plot_collection

        return plot_collection(
            self,
            channels=channels,
            labels=labels,
            start=start,
            stop=stop,
            resolution=resolution,
            time_unit=time_unit,
            include_attached_labels=include_attached_labels,
            subplots_kwargs=subplots_kwargs,
        )

    def plot_interactive(
        self,
        channels: list[list[ChannelSpecification | int]] | None = None,
        labels: list[list[LabelSpecification | int]] | None = None,
        start: Timestamp | Timedelta | float | str | None = None,
        stop: Timestamp | Timedelta | float | str | None = None,
        time_unit: str | None = None,
        include_attached_labels: bool = False,
        channel_overviews: list[list[ChannelSpecification | int]] | bool = False,
        limited_overview: bool = False,
        subplots_kwargs: dict[str, Any] | None = None,
    ):
        """Plot the data in the collection using ipywidgets.

        This allows to annotate the data with labels, and to modify
        channel offsets interactively.

        Parameters
        ----------
        channels
            The channels to plot. If not specified, all channels are plotted.
            Specified as a list of lists with individual lists containing
            channels to be collected in one subplot.
        labels
            The labels to plot. If not specified, all labels are plotted.
            Specified as a list of lists, same as for the channels.
        start
            The start time for the plot. If not specified, the plot starts
            from the first time point.
        stop
            The stop time for the plot. If not specified, the plot stops
            at the last time point.
        time_unit
            The time unit in which channel and label data are represented
            in. If not specified, the time unit of the channels is used.
        include_attached_labels
            Whether to automatically include labels attached to the
            specified channels.
        channel_overviews
            Similar to ``channel``, but plots the specified channels
            in a separate subplot in a condensed way including a
            location map of the main plot. If set to ``True``, all
            chosen channels are plotted in a single overview.
        limited_overview
            Whether the time interval of the overview subplot should be limited
            to the recording interval of the channels being plotted.
        subplots_kwargs
            Keyword arguments passed to ``matplotlib.pyplot.subplots``.

        Returns
        -------
        InteractivePlot
            An ipywidgets layout (control panel above the figure) that can be
            displayed and combined with other widgets. It also provides:

            - ``on(event, callback)``: react to changes, e.g.
              ``"labels_changed"`` after annotating;
            - ``add_overlay(overlay, subplot=0)``: persistent overlays, e.g.
              :class:`~vitabel.plotting.overlays.FillBetween` to shade the
              area of a signal below a threshold;
            - ``set_window(start, stop)``;
            - ``figure``: the matplotlib figure (customisations persist when
              the plot is updated).
        """
        from vitabel.plotting.interactive import plot_interactive

        return plot_interactive(
            self,
            channels=channels,
            labels=labels,
            start=start,
            stop=stop,
            time_unit=time_unit,
            include_attached_labels=include_attached_labels,
            channel_overviews=channel_overviews,
            limited_overview=limited_overview,
            subplots_kwargs=subplots_kwargs,
        )
