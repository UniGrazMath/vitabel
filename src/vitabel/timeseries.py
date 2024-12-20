from __future__ import annotations

from copy import copy
from typing import Any, TypeAlias, Union

import itertools as it
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numbers
import numpy as np
import numpy.typing as npt
import hashlib

from vitabel.utils.helpers import match_object, NumpyEncoder, decompress_array


Timedelta: TypeAlias = pd.Timedelta | np.timedelta64
Timestamp: TypeAlias = pd.Timestamp | np.datetime64
ChannelSpecification: TypeAlias = Union[str, dict[str, Any], "Channel"]
LabelSpecification: TypeAlias = Union[str, dict[str, Any], "Label"]

logger = logging.getLogger("vitabel")


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
        For relative time data (timedeltas or numeric values),
        this parameter specifies the reference time.
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
    time_start: Timestamp | None
    time_unit: str
    offset: Timedelta

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
        self.offset = offset

        if not isinstance(
            time_index, (pd.TimedeltaIndex, pd.DatetimeIndex, np.ndarray)
        ):
            time_index = np.array(time_index)

        if len(time_index) > 0:
            time_type = type(time_index[0])
        else:
            time_type = pd.Timedelta

        if not all(isinstance(t, time_type) for t in time_index) and not all(
            isinstance(t, numbers.Number) for t in time_index
        ):
            raise ValueError("All time data must be of the same type")

        if time_type in (str, np.str_):
            for convert_func in [pd.to_datetime, pd.to_timedelta]:
                try:
                    time_index = convert_func(time_index)
                    time_type = type(time_index[0])
                    break
                except ValueError:
                    pass
            else:
                raise ValueError(
                    "The time data could not be parsed to timestamps or timedeltas"
                )

        if time_type in (pd.Timestamp, np.datetime64):  # absolute time
            # check that time_start does not conflict
            if time_start is not None:
                raise ValueError("time_start cannot be passed if time data is absolute")
            time_start = pd.Timestamp(time_index[0])
            time_index = pd.to_timedelta([time - time_start for time in time_index])

        elif time_type in (pd.Timedelta, np.timedelta64):
            time_index = pd.to_timedelta(time_index)

        elif issubclass(time_type, numbers.Number):
            time_index = pd.to_timedelta(time_index, unit=self.time_unit)

        else:
            raise ValueError(f"The time data type {time_type} is not supported")

        self.time_index = time_index + offset

        if time_start is not None:
            time_start = pd.Timestamp(time_start)
            time_start = time_start.tz_localize(None)
        self.time_start = time_start

    def __len__(self):
        return len(self.time_index)

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
        return np.array(
            [time / pd.to_timedelta(1, unit=time_unit) for time in self.time_index]
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
        self.offset += delta_t

    def convert_time_input(self, time_input: Timestamp | Timedelta | float):
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

        if isinstance(time_input, Timestamp):
            raise ValueError(
                f"The channel time is relative, but {time_input} is a timestamp"
            )
        elif isinstance(time_input, Timedelta):
            return time_input
        elif isinstance(time_input, numbers.Number):
            return pd.to_timedelta(time_input, unit=self.time_unit)

        raise ValueError(
            f"Could not convert {time_input} to a valid time format for this channel"
        )

    def get_time_mask(
        self,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        resolution: Timedelta | float | str | None = None,
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
        """
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


class Channel(TimeSeriesBase):
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
        self.name = name
        self.labels = []

        if data is not None:
            if not isinstance(data, np.ndarray):
                data = np.asarray(data)

            if len(data) != len(time_index):
                raise ValueError(
                    f"For channels with data the time index length ({len(time_index)}) "
                    f"and data length ({len(data)}) must be the same"
                )
        self.data = data

        self.plotstyle = copy(plotstyle) or {}
        self.metadata = copy(metadata) or {}

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

    def data_hash(self) -> str:
        """Return a hash representing the data and the metadata of this channel."""
        data = {
            "name": self.name,
            "time_index": self.time_index,
            "data": self.data,
            "metadata": self.metadata,
            "time_start": self.time_start,
            "time_unit": self.time_unit,
            "offset": self.offset,
        }
        data_buffer = json.dumps(data, cls=NumpyEncoder).encode("utf-8")
        return hashlib.sha256(data_buffer).hexdigest()

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

        Parameters
        ----------
        label
            The label to detach.
        """
        if label not in self.labels:
            raise ValueError(f"The label {label.name} is not attached to this channel")
        self.labels.remove(label)

    def shift_time_index(
        self, delta_t: pd.Timedelta | float, time_unit: str | None = None
    ):
        for label in self.labels:
            label.shift_time_index(delta_t=delta_t, time_unit=time_unit)
        return super().shift_time_index(delta_t, time_unit)

    def to_dict(self) -> dict[str, Any]:
        """Construct a serializable dictionary that represents
        this channel."""
        return {
            "name": self.name,
            "time_index": self.numeric_time(),
            "data": self.data,
            "time_start": str(self.time_start) if self.time_start is not None else None,
            "time_unit": self.time_unit,
            "offset": self.offset / pd.to_timedelta(1, unit=self.time_unit),
            "labels": [label.to_dict() for label in self.labels],
            "plotstyle": self.plotstyle,
            "metadata": self.metadata,
        }

    def to_csv(
        self,
        start: Timestamp | Timedelta | None = None,
        stop: Timestamp | Timedelta | None = None,
        filename: str | None = None,
    ) -> None:
        """Export the channel data to a CSV file.

        Parameters
        ----------
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        filename
            The name of the file to export the data to. If not
            specified, the data is exported to a file with the
            name of the channel.
        """
        time_index, data = self.get_data(start=start, stop=stop)
        if filename is None:
            filename = f"{self.name}.csv"

        if data is None:
            df = pd.DataFrame({"time": time_index})
        else:
            df = pd.DataFrame({"time": time_index, "data": data})
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
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
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
        """

        time_mask = self.get_time_mask(start=start, stop=stop, resolution=resolution)

        time_index = self.time_index[time_mask]
        if self.is_time_absolute():
            time_index += self.time_start

        data = self.data[time_mask] if self.data is not None else None
        return time_index, data

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
            The resolution for the data. If specified, the data
            is downsampled such that the median time difference is
            bounded below by the given resolution. See :meth:`.get_data`.
        time_unit
            The time unit values used along the x-axis. If ``None``
            (the default), the time unit of the channel is used.
        """

        time_index, data = self.get_data(start=start, stop=stop, resolution=resolution)
        if data is None:
            data = np.zeros_like(time_index, dtype=float)

        # when plotting a single channel, the times should always
        # be numeric (matplotlib's datetime formatter is not ideal,
        # the labels have lots of overlap).
        if self.is_time_absolute():
            reference_time = reference_time or self.time_start
            time_index = time_index - reference_time

        if time_unit is None:
            time_unit = self.time_unit
        time_index /= pd.to_timedelta(1, unit=time_unit)

        if plot_axes is None:
            figure, plot_axes = plt.subplots()
        else:
            figure = plot_axes.get_figure()

        base_plotstyle = self.plotstyle.copy()
        base_plotstyle.update(plotstyle if plotstyle is not None else {})

        (line,) = plot_axes.plot(time_index, data, **base_plotstyle)
        if "label" not in base_plotstyle:
            line.set_label(self.name)

        return figure

    def rename(self, new_name: str):
        """
        Rename channel

        Parameters
        ----------
        new_name : str
            The new name of the channel.

        Returns
        -------
        None.

        """
        self.name = new_name


# TODO: handling of name, data, plotstyle, metadata is
# the same as for channel and could be factored out
class Label(TimeSeriesBase):
    def __init__(
        self,
        name: str,
        time_index: npt.ArrayLike[Timestamp | Timedelta | float | str] | None = None,
        data: npt.ArrayLike[float | np.number | str] | None = None,
        time_start: Timestamp | None = None,
        time_unit: str | None = None,
        offset: Timedelta | float | None = None,
        anchored_channel: Channel | None = None,
        plotstyle: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.anchored_channel = None

        if time_index is None:
            time_index = np.array([])

        if data is not None:
            if len(data) > 0 and isinstance(data[0], str):
                data = np.array(data, dtype=object)
            else:
                data = np.asarray(data)
        self._check_data_shape(time_index, data)
        self.data = data

        self.plotstyle = copy(plotstyle) or {
            "marker": "o",
            "ms": 5,
            "linestyle": "none",
        }
        self.metadata = copy(metadata) or {}

        super().__init__(
            time_index=time_index,
            time_start=time_start,
            time_unit=time_unit,
            offset=offset,
        )

        if anchored_channel is not None:
            self.attach_to(anchored_channel)

    def __eq__(self, other: Label) -> bool:
        return (
            self.name == other.name
            and np.array_equal(self.time_index, other.time_index)
            and np.array_equal(self.data, other.data)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def _check_data_shape(self, time_index: npt.ArrayLike, data: npt.ArrayLike):
        if data is not None and len(data) != len(time_index):
            raise ValueError(
                "The length of the data must be equal to the length of the time index"
            )

    def _data_can_hold_number(self) -> bool:
        return self.data is not None and (
            self.is_empty() or np.issubdtype(self.data.dtype, np.number)
        )

    def _data_can_hold_text(self) -> bool:
        return self.data is not None and (
            self.is_empty() or self.data.dtype == "object"
        )

    def add_data(
        self, time_data: Timestamp | Timedelta, value: float | str | None = None
    ):
        """Add a data point to the label.

        Parameters
        ----------
        time
            The time of the data point.
        value
            The value of the data point. If not specified,
            the data point is set to zero.
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

        [insert_index] = self.time_index.searchsorted([time_data])
        self.time_index = self.time_index.insert(insert_index, time_data)

        if self.data is not None:
            if len(self.data) == 0 and isinstance(value, str):
                # dtype object for arbitrary length strings
                self.data = np.array([], dtype=object)
            elif len(self.data) == 0:
                self.data = np.array([])

            self.data = np.insert(self.data, insert_index, value)

    def remove_data(
        self,
        time_data: Timestamp | Timedelta,
    ):
        """Remove a data point from the label given its time."""
        if self.is_time_absolute():
            time_data = time_data - self.time_start

        matches = np.argwhere(self.time_index == time_data).flatten().tolist()
        if len(matches) == 0:
            raise ValueError(f"No data point found at time {time_data}")
        remove_index = min(matches)
        self.time_index = self.time_index.delete(remove_index)
        if self.data is not None:
            self.data = np.delete(self.data, remove_index)

        if remove_index == 0 and self.is_time_absolute():
            if self.is_empty():
                self.time_start = None
            else:
                offset = self.time_index[0]
                self.time_start += offset
                self.time_index -= offset

    @classmethod
    def from_dict(cls, datadict: dict[str, Any]) -> Label:
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

        return cls(
            name=datadict.get("name"),
            time_index=time_index,
            data=data,
            time_start=datadict.get("time_start"),
            time_unit=datadict.get("time_unit"),
            offset=datadict.get("offset"),
            plotstyle=datadict.get("plotstyle"),
            metadata=datadict.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "time_index": self.numeric_time(),
            "data": self.data,
            "time_start": str(self.time_start) if self.time_start is not None else None,
            "time_unit": self.time_unit,
            "offset": self.offset / pd.to_timedelta(1, unit=self.time_unit),
            "is_interval": False,
            "plotstyle": self.plotstyle,
            "metadata": self.metadata,
        }

    def to_csv(
        self,
        start: Timestamp | Timedelta | None = None,
        stop: Timestamp | Timedelta | None = None,
        filename: str | None = None,
    ) -> None:
        """Export the label data to a CSV file.

        Parameters
        ----------
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        filename
            The name of the file to export the data to. If not
            specified, the data is exported to a file with the
            name of the channel.
        """
        time_index, data = self.get_data(start=start, stop=stop)
        if filename is None:
            filename = f"{self.name}.csv"

        if data is None:
            df = pd.DataFrame({"time": time_index})
        else:
            df = pd.DataFrame({"time": time_index, "data": data})
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
        """Detach the label from the channel."""
        if self.anchored_channel is None:
            raise ValueError(f"The label {self.name} is not attached to any channel")
        self.anchored_channel.detach_label(self)
        self.anchored_channel = None

    def get_data(
        self,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """Return a tuple of time and data values with optional
        filtering.

        Parameters
        ----------
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        """

        time_mask = self.get_time_mask(start=start, stop=stop, resolution=None)

        time_index = self.time_index[time_mask]
        if self.is_time_absolute():
            time_index += self.time_start

        data = self.data[time_mask] if self.data is not None else None
        return time_index, data

    def plot(
        self,
        plot_axes: plt.Axes | None = None,
        plotstyle: dict[str, Any] | None = None,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        time_unit: str | None = None,
        reference_time: Timestamp | Timedelta | float | None = None,
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
        """
        time_index, data = self.get_data(start=start, stop=stop)

        if self.is_time_absolute():
            reference_time = reference_time or self.time_start
            time_index = time_index - reference_time

        if time_unit is None:
            time_unit = self.time_unit
        time_index /= pd.to_timedelta(1, unit=time_unit)
        ymin, ymax = plot_axes.get_ylim()

        if data is None:
            data = np.ones_like(time_index, dtype=float) * (ymin + ymax) / 2

        if plot_axes is None:
            figure, plot_axes = plt.subplots()
        else:
            figure = plot_axes.get_figure()

        base_plotstyle = self.plotstyle.copy()
        if plotstyle is not None:
            base_plotstyle.update(plotstyle)

        # TODO: data entries might be strings too (or should this be changed?)
        if len(data) > 0 and isinstance(data[0], str):
            if plotstyle is None:
                base_plotstyle.update({"linestyle": "solid", "marker": None})
            for t, text in zip(time_index, data):
                artist = plot_axes.axvline(t, **base_plotstyle)
                line_color = artist.get_color()
                box_props = {
                    "boxstyle": "round",
                    "alpha": 0.6,
                    "facecolor": "white",
                    "edgecolor": "black",
                }
                plot_axes.text(
                    t,
                    0.9 * ymin + 0.1 * ymax,
                    text,
                    rotation=90,
                    clip_on=True,
                    color=line_color,
                    bbox=box_props,
                )
                artists = [artist]
        else:
            artists = plot_axes.plot(time_index, data, **base_plotstyle)

        # practically should just be one artist
        for artist in artists:
            if "label" not in base_plotstyle:
                artist.set_label(self.name)

        return figure


class IntervalLabel(Label):
    def _check_data_shape(self, time_index: npt.ArrayLike, data: npt.ArrayLike):
        if len(time_index) % 2 != 0:
            raise ValueError("The time index must contain an even number of elements")
        if data is not None and len(time_index) != 2 * len(data):
            raise ValueError(
                "The length of the data must be half the length of the time index"
            )

    def __len__(self):
        return super().__len__() // 2

    @property
    def intervals(self) -> npt.NDArray:
        time_intervals = np.array(self.time_index).reshape(-1, 2)
        if self.is_time_absolute():
            time_intervals += self.time_start
        return time_intervals

    def to_dict(self) -> dict[str, Any]:
        label_dict = super().to_dict()
        label_dict["is_interval"] = True
        return label_dict

    def to_csv(
        self,
        start: Timestamp | Timedelta | None = None,
        stop: Timestamp | Timedelta | None = None,
        filename: str | None = None,
    ) -> None:
        """Export the interval label data to a CSV file.

        Contains all intervals with a non-empty intersection
        of the specified data range.

        Parameters
        ----------
        start
            The start time for the data. If not specified, the
            data starts from the beginning.
        stop
            The stop time for the data. If not specified, the
            data ends at the last time point.
        filename
            The name of the file to export the data to. If not
            specified, the data is exported to a file with the
            name of the channel.
        """
        time_index, data = self.get_data(start=start, stop=stop)
        time_start, time_stop = time_index.transpose()

        if filename is None:
            filename = f"{self.name}.csv"

        if data is None:
            df = pd.DataFrame({"time_start": time_start, "time_stop": time_stop})
        else:
            df = pd.DataFrame(
                {"time_start": time_start, "time_stop": time_stop, "data": data}
            )
        df.to_csv(filename)

    def get_data(
        self,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
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
            return self.intervals, self.data

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
        return intervals, data

    def add_data(
        self,
        time_data: tuple[Timestamp, Timestamp] | tuple[Timedelta, Timedelta],
        value: float | None = None,
    ):
        interval_start, interval_end = time_data
        if self.is_empty() and isinstance(interval_start, Timestamp):
            self.time_start = interval_start

        if self.is_time_absolute():
            interval_start -= self.time_start
            interval_end -= self.time_start

        # TODO: time_start currently does not reset when corresponding
        # first data point is removed, time index is not necessarily monotonous

        self.time_index = self.time_index.append(
            pd.TimedeltaIndex([interval_start, interval_end])
        )
        if self.data is not None:
            if len(self.data) == 0 and isinstance(value, str):
                self.data = np.array([], dtype=object)
            elif len(self.data) == 0:
                self.data = np.array([])
            self.data = np.append(self.data, value)

    def remove_data(
        self,
        time_data: tuple[Timestamp, Timestamp] | tuple[Timedelta, Timedelta],
    ):
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

        if len(self.time_index) == 0:
            self.time_start = None

    def plot(
        self,
        plot_axes: plt.Axes | None = None,
        plotstyle: dict[str, Any] | None = None,
        start: Timestamp | Timedelta | float | None = None,
        stop: Timestamp | Timedelta | float | None = None,
        time_unit: str | None = None,
        reference_time: Timestamp | Timedelta | float | None = None,
    ):
        time_index, data = self.get_data(start=start, stop=stop)

        if self.is_time_absolute():
            reference_time = reference_time or self.time_start
            time_index = time_index - np.datetime64(reference_time)

        if time_unit is None:
            time_unit = self.time_unit
        time_index /= pd.to_timedelta(1, unit=time_unit)

        time_midpoints = np.mean(time_index, axis=1)
        time_radius = np.diff(time_index, axis=1).reshape(-1) / 2.0
        if data is None:
            data = np.zeros_like(time_midpoints, dtype=float)

        if plot_axes is None:
            figure, plot_axes = plt.subplots()
        else:
            figure = plot_axes.get_figure()

        base_plotstyle = {"fmt": "none", "capsize": 3}
        base_plotstyle.update(self.plotstyle.copy())
        if plotstyle is not None:
            base_plotstyle.update(plotstyle)

        # TODO: deal with data string entries (or disallow them)
        artist = plot_axes.errorbar(
            time_midpoints, data, xerr=time_radius, **base_plotstyle
        )
        artist.set_label(self.name)

        return figure


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
        return (
            self.channel_data_hash() == other.channel_data_hash()
            and self.labels == other.labels
        )

    def __repr__(self) -> str:
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
        """Return all labels anchored to a channel in the collection."""
        return [label for channel in self.channels for label in channel.labels]

    @property
    def labels(self) -> list[Label]:
        """Return all labels in the collection."""
        return self.global_labels + self.local_labels

    @property
    def channel_names(self) -> list[str]:
        """Return the list of channel names."""
        return [channel.name for channel in self.channels]

    @property
    def label_names(self) -> list[str]:
        """Return the list of label names."""
        return [label.name for label in self.labels]

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

    def print_summary(self):
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
                f"Identical label {label.name} has already "
                "been added to the collection"
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
        channel_list = [
            channel for channel in self.channels if match_object(channel, **kwargs)
        ]

        if len(channel_list) == 0:
            logger.warning(f"Channel specification {kwargs} returned no channels")

        return channel_list

    def get_channel(self, name: str | None = None, **kwargs) -> Channel:
        """Return a channel by name.

        Raises an error if no unique channel is found.

        Parameters
        ----------
        name
            The name of the channel to retrieve. Allowed to be passed
            either as a positional or a keyword argument.
        kwargs
            Keyword arguments to filter the channels by.
        """
        if name is not None:
            kwargs["name"] = name
        channels = self.get_channels(**kwargs)
        if len(channels) != 1:
            raise ValueError(
                "Channel specification was ambiguous, no unique channel "
                f"was identified. Query returned: {channels}"
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

    def get_label(self, name: str | None = None, **kwargs) -> Label:
        """Return a label by name.

        Raises an error if no unique label is found.

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
        labels = self.get_labels(**kwargs)
        if len(labels) != 1:
            raise ValueError(
                "Channel specification was ambiguous, no unique label "
                f"was identified. Query returned: {labels}"
            )
        return labels[0]

    def remove_label(self, *, label: Label | None = None, **kwargs):
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

    def remove_channel(self, *, channel: Channel | None = None, **kwargs):
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
        """Create a collection from a dictionary representation."""
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

    def _parse_time(self, time_spec: Timestamp | Timedelta | float | str | None):
        if isinstance(time_spec, str):
            if self.is_time_absolute():
                return pd.to_datetime(time_spec)
            else:
                return pd.to_timedelta(time_spec)
        elif isinstance(time_spec, (Timedelta, Timestamp, float)) or time_spec is None:
            return time_spec
        raise ValueError(f"Time specification {time_spec} could not be parsed")

    def _parse_channel_specification(
        self, channels: list[list[ChannelSpecification | int]] | None
    ) -> list[list[Channel]]:
        channel_lists = []
        if channels is None:
            channel_lists.append(self.channels)
        else:
            for spec_list in channels:
                channel_list: list[Channel] = []
                for spec in spec_list:
                    if isinstance(spec, str):
                        channel_list.extend(self.get_channels(name=spec))
                    elif isinstance(spec, dict):
                        channel_list.extend(self.get_channels(**spec))
                    elif isinstance(spec, int):
                        channel_list.append(self.channels[spec])
                    elif isinstance(spec, Channel):
                        channel_list.append(spec)
                    else:
                        raise ValueError(f"Invalid channel specification: {spec}")

                channel_lists.append(channel_list)
        return channel_lists

    def _parse_label_specification(
        self,
        labels: list[list[LabelSpecification | int]] | None,
        channel_lists: list[list[Channel]],
        include_attached_labels: bool = False,
    ) -> list[list[Label]]:
        num_subplots = len(channel_lists)
        label_lists: list[list[Label]] = []

        if labels is None:
            for _ in range(num_subplots):
                label_lists.append([label for label in self.global_labels])
        else:
            for spec_list in labels:
                label_list: list[Label] = []
                for spec in spec_list:
                    if isinstance(spec, str):
                        label_list.extend(self.get_labels(name=spec))
                    elif isinstance(spec, dict):
                        label_list.extend(self.get_labels(**spec))
                    elif isinstance(spec, int):
                        label_list.append(self.labels[spec])
                    elif isinstance(spec, Label):
                        label_list.append(spec)
                    else:
                        raise ValueError(f"Invalid label specification: {spec}")
                label_lists.append(label_list)
        if include_attached_labels:
            for idx in range(num_subplots):
                for channel in channel_lists[idx]:
                    for label in channel.labels:
                        if label not in label_lists[idx]:
                            label_lists[idx].append(label)
        return label_lists

    def _get_time_extremum(
        self,
        time: Timestamp | Timedelta | float | str | None,
        channel_lists: list[list[Channel]],
        minimum: bool = True,
    ):
        op = min
        if not minimum:
            op = max

        time = self._parse_time(time)
        if time is None:
            time_list = []
            for channel in it.chain.from_iterable(channel_lists):
                ex_time = op(channel.time_index)
                if self.is_time_absolute():
                    ex_time += channel.time_start
                time_list.append(ex_time)
            time = op(time_list)
        return time

    def _get_timeunit_from_channels(self, channel_lists: list[list[Channel]]) -> str:
        channel_iter = it.chain.from_iterable(channel_lists)
        time_unit = next(channel_iter).time_unit
        if not all(channel.time_unit == time_unit for channel in channel_iter):
            raise ValueError(
                "The channel time units are not uniform. Specify the plot time "
                "unit explicitly by specifying the time_unit argument"
            )
        return time_unit

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
        """Plot the data in the collection."""
        # 1) turn channels into proper list of (list of) Channels
        # 2) same for labels, respect include_attached_labels
        # 3) determine global start and end time (of selected channels)
        # 4) get_data, and construct the corresponding plot.

        channel_lists = self._parse_channel_specification(channels)
        num_subplots = len(channel_lists)

        if subplots_kwargs is None:
            subplots_kwargs = {}

        label_lists = self._parse_label_specification(
            labels, channel_lists, include_attached_labels=include_attached_labels
        )

        start = self._get_time_extremum(start, channel_lists, minimum=True)
        stop = self._get_time_extremum(stop, channel_lists, minimum=False)

        if time_unit is None:
            time_unit = self._get_timeunit_from_channels(channel_lists)

        fig, axes = plt.subplots(num_subplots, squeeze=False, **subplots_kwargs)
        axes = axes[:, 0]

        if resolution is None:
            screen_pixel_width, screen_pixel_height = fig.canvas.get_width_height()
            data_width = (stop - start).total_seconds()
            resolution = data_width / screen_pixel_width

        for channel_list, label_list, subax in zip(channel_lists, label_lists, axes):
            for channel in channel_list:
                channel.plot(
                    plot_axes=subax,
                    start=start,
                    stop=stop,
                    resolution=resolution,
                    time_unit=time_unit,
                )

            for label in label_list:
                label.plot(plot_axes=subax, start=start, stop=stop, time_unit=time_unit)

            plot_duration = (stop - start) / pd.to_timedelta(1, unit=time_unit)
            subax.set_xlim((0, plot_duration))
            subax.grid(True)
            subax.legend(loc="upper right")

        return fig, axes

    # for interactive plotting:
    # add repr-strings for labels (and channels too)
    # add methods for adding and removing data points to labels
    # interactive plot should also have a (linear) history with undo...

    def plot_interactive(
        self,
        channels: list[list[ChannelSpecification | int]] | None = None,
        labels: list[list[LabelSpecification | int]] | None = None,
        start: Timestamp | Timedelta | float | str | None = None,
        stop: Timestamp | Timedelta | float | str | None = None,
        time_unit: str | None = None,
        include_attached_labels: bool = False,
        channel_overviews: list[list[ChannelSpecification | int]] | bool = False,
        subplots_kwargs: dict[str, Any] | None = None,
    ):
        """Plot the data in the collection using ipywidgets.

        This allows to annotate the data with labels, and to modify
        channel offsets interactively.
        """
        import ipywidgets as widgets
        from enum import Enum
        from IPython import get_ipython
        from matplotlib.backend_bases import MouseButton, MouseEvent, KeyEvent

        CANVAS_SELECTION_TOLERANCE_PX = 5

        channel_lists = self._parse_channel_specification(channels)
        if channel_overviews is False:
            channel_overviews = []
        elif channel_overviews is True:
            channel_overviews = [list(set(it.chain.from_iterable(channel_lists)))]
        else:
            channel_overviews = self._parse_channel_specification(channel_overviews)

        if subplots_kwargs is None:
            subplots_kwargs = {}

        label_lists = self._parse_label_specification(
            labels, channel_lists, include_attached_labels=include_attached_labels
        )

        # if neither channel nor label is present, remove corresponding subplots
        empty_channel_indices = [
            idx
            for idx, (channel_list, label_list) in enumerate(
                zip(channel_lists, label_lists)
            )
            if len(channel_list) == 0 and len(label_list) == 0
        ]
        channel_lists = [
            channel_list
            for idx, channel_list in enumerate(channel_lists)
            if idx not in empty_channel_indices
        ]
        label_lists = [
            label_list
            for idx, label_list in enumerate(label_lists)
            if idx not in empty_channel_indices
        ]

        num_subplots = len(channel_lists) + len(channel_overviews)
        start = self._get_time_extremum(start, channel_lists, minimum=True)
        reference_time = start
        stop = self._get_time_extremum(stop, channel_lists, minimum=False)
        shift_span = (stop - start) * 0.25

        if time_unit is None:
            time_unit = self._get_timeunit_from_channels(channel_lists)

        ipy_shell = get_ipython()
        if ipy_shell is None:
            raise RuntimeError("This method can only be used in an IPython environment")

        class InteractionMode(Enum):
            ANNOTATE = 0
            ADJUST = 1
            SETTINGS = 2

        class LabelValueType(Enum):
            NUMERIC = "Numeric"
            ONLY_TIMESTAMP = "Only Timestamp"
            TEXTUAL = "Textual"

        value_type_dropdown = widgets.Dropdown(
            options=[(label_type.value, label_type) for label_type in LabelValueType],
            description="Value type:",
        )
        value_text_input = widgets.Text(placeholder="Label text ...")
        value_type_stack = widgets.Stack(
            [
                widgets.HTML(),
                widgets.HTML(),
                value_text_input,
            ]
        )
        widgets.jslink(
            (value_type_dropdown, "index"), (value_type_stack, "selected_index")
        )

        distinct_labels = []
        for label_list in label_lists:
            for label in label_list:
                if label not in distinct_labels:
                    distinct_labels.append(label)

        label_dropdown = widgets.Dropdown(
            options=[
                label for label in distinct_labels if label._data_can_hold_number()
            ],
            description="Active label",
            disabled=False,
        )
        DELETE_ANNOTATIONS = False
        delete_toggle_button = widgets.ToggleButton(
            value=False,
            description="Mode: Add Data",
            disabled=False,
            button_style="success",
        )

        def delete_toggle_handler(change):
            nonlocal DELETE_ANNOTATIONS

            if change["new"]:  # value of "new" attribute is new button value
                delete_toggle_button.description = "Mode: Delete Data"
                delete_toggle_button.button_style = "danger"
                DELETE_ANNOTATIONS = True
            else:
                delete_toggle_button.description = "Mode: Add Data"
                delete_toggle_button.button_style = "success"
                DELETE_ANNOTATIONS = False

        delete_toggle_button.observe(delete_toggle_handler, names="value")

        def value_type_dropdown_handler(change):
            new_type = change.get("new", None)
            if new_type == LabelValueType.NUMERIC:
                label_dropdown.options = [
                    label for label in distinct_labels if label._data_can_hold_number()
                ]
            elif new_type == LabelValueType.ONLY_TIMESTAMP:
                label_dropdown.options = [
                    label for label in distinct_labels if label.data is None
                ]
            elif new_type == LabelValueType.TEXTUAL:
                label_dropdown.options = [
                    label for label in distinct_labels if label._data_can_hold_text()
                ]

        value_type_dropdown.observe(handler=value_type_dropdown_handler, names="value")

        # ---------- WIDGETS FOR SHIFTING ----------------------

        shifting_channel_selection = widgets.SelectMultiple(
            value=[self.channels[0]],
            options=[(chan.name, chan) for chan in self.channels],
            description="Channels / Labels",
            disabled=False,
            continuous_update=True,
        )

        # ----------- WIDGETS FOR SETTINGS ----------------------------

        limit_widgets = []
        for idx, channel_list in enumerate(channel_lists, start=1):
            min_slider = widgets.FloatText(value=0, description="min")
            max_slider = widgets.FloatText(value=1, description="max")
            limit_widgets.append(
                widgets.HBox(
                    [
                        widgets.Label(f"Plot {idx} limits:"),
                        min_slider,
                        max_slider,
                    ]
                )
            )
        settings_apply_button = widgets.Button(description="Apply")

        # ------------------ ENTIRE WIDGET DESIGN ---------------------

        tab = widgets.Tab()
        tab.children = [
            widgets.VBox(
                [
                    widgets.HTML(
                        """
                    <p>
                    Right-click to add data points to the active label. Use the number
                    keys (<kbd style="color: black;">0</kbd> to <kbd style="color: black;">9</kbd>)
                    to quickly switch between active labels. Use the button below or the
                    <kbd style="color: black;">D</kbd> key to toggle between adding and deleting labels.
                    </p>
                    <p>
                    Use the left and right arrow keys to shift the plotted time window.
                    By clicking in an overview plot (if one is present), the plot window location
                    is moved.
                    The <kbd style="color: black;">+</kbd> and <kbd style="color: black;">-</kbd>
                    keys zoom in and out, respectively.
                    Mouse scrolling in a subplot zooms the vertical axis.
                    </p>
                    """
                    ),
                    widgets.HBox(
                        [
                            value_type_dropdown,
                            value_type_stack,
                        ]
                    ),
                    widgets.HBox(
                        [
                            label_dropdown,
                            delete_toggle_button,
                        ]
                    ),
                ]
            ),
            widgets.VBox(
                [
                    widgets.HTML(
                        """
                    <p>Select channels and labels to be shifted in the menu below.
                    Multiple values can be selected by holding down <kbd style="color: black;">Ctrl</kbd>
                    (on MacOS: <kbd style="color: black;">Cmd</kbd>) while clicking.
                    </p>
                    <p>Right-click into one of the plots to first set a reference time
                    (represented by a vertical dotted red line), then right-click again
                    to the position where the reference time should be moved to for the
                    selected channels. Press <kbd style="color: black;">Esc</kbd> to
                    clear a set reference time.</p>
                    """
                    ),
                    shifting_channel_selection,
                ]
            ),
            widgets.VBox(
                [
                    widgets.Label("Vertical plot limits:"),
                    *limit_widgets,
                    settings_apply_button,
                ]
            ),
        ]
        tab.titles = [
            "Annotate",
            "Align Timelines",
            "Settings",
        ]

        ipy_shell.enable_matplotlib(gui="widget")

        with plt.ioff():
            fig, axes = plt.subplots(num_subplots, squeeze=False, **subplots_kwargs)
            screen_pixel_width, screen_pixel_height = fig.canvas.get_width_height()

            axes = axes[:, 0]
            channel_axes = axes[: len(channel_lists)]
            overview_axes = axes[len(channel_lists) :]
            fig.canvas.toolbar.toolitems = [
                tool
                for tool in fig.canvas.toolbar.toolitems
                if tool[0] in ["Home", "Zoom", "Download"]
            ]
            fig.canvas.header_visible = True
            if self.is_time_absolute():
                fig.suptitle(f"Reference time: {reference_time}")

            # for channel_list, ax in zip(channel_lists,channel_axes):
            #     axes_title = ''
            #     for channel in channel_list:
            #         axes_title += channel.name + ', '
            #     ax.set_title(axes_title,loc='center')
            #     ax.set_ylabel(axes_title,loc='center')

        x_indicators = [
            ax.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
            for ax in channel_axes
        ]
        overview_indicators = []

        def update_ylim_settings():
            for ax, limit_widget in zip(channel_axes, limit_widgets):
                _, min_input, max_input = limit_widget.children
                ymin, ymax = ax.get_ylim()
                min_input.value = ymin
                max_input.value = ymax

        def format_coords(x, y):
            format_string = f"(x, y) = ({x:.2f}, {y:.2f})"
            if self.is_time_absolute():
                format_string += (
                    f"   x = {x * pd.to_timedelta(1, unit=time_unit) + reference_time}"
                )

            return format_string

        def repaint_plot(start, stop):
            nonlocal \
                fig, \
                channel_axes, \
                overview_axes, \
                overview_indicators, \
                screen_pixel_width
            data_width = (stop - start).total_seconds()
            resolution = data_width / screen_pixel_width

            with plt.ioff():
                for channel_list, label_list, indicator, subax in zip(
                    channel_lists, label_lists, x_indicators, channel_axes
                ):
                    old_ylims = subax.get_ylim()
                    old_ylabel = subax.yaxis.get_label()
                    subax.clear()
                    subax.add_artist(indicator)
                    subax.format_coord = format_coords
                    subax.set_xlim(
                        (
                            (start - reference_time)
                            / pd.to_timedelta(1, unit=time_unit),
                            (stop - reference_time)
                            / pd.to_timedelta(1, unit=time_unit),
                        )
                    )
                    subax.grid(True)
                    subax.set_xlabel(f"time [{time_unit}]", labelpad=-12, fontsize=7)
                    subax.yaxis.set_label(old_ylabel)
                    if old_ylims != (0, 1):
                        subax.set_ylim(old_ylims)
                    for channel in channel_list:
                        channel.plot(
                            plot_axes=subax,
                            start=start,
                            stop=stop,
                            resolution=resolution,
                            time_unit=time_unit,
                            reference_time=reference_time,
                        )

                    for label in label_list:
                        label.plot(
                            plot_axes=subax,
                            start=start,
                            stop=stop,
                            time_unit=time_unit,
                            reference_time=reference_time,
                        )
                    subax.legend(loc="lower right")
                for indicator in overview_indicators:
                    indicator.remove()
                overview_indicators = [
                    ax.axvspan(
                        xmin=(start - reference_time)
                        / pd.to_timedelta(1, unit=time_unit),
                        xmax=(stop - reference_time)
                        / pd.to_timedelta(1, unit=time_unit),
                        color="red",
                        alpha=0.25,
                    )
                    for ax in overview_axes
                ]
            return

        def repaint_overview_plot():
            for channel_list, subax in zip(channel_overviews, overview_axes):
                if not channel_list:
                    continue
                ov_start = self._get_time_extremum(
                    None, channel_lists=[channel_list], minimum=True
                )
                ov_stop = self._get_time_extremum(
                    None, channel_lists=[channel_list], minimum=False
                )
                data_width = (ov_stop - ov_start).total_seconds()
                resolution = data_width / screen_pixel_width
                subax.clear()
                for channel in channel_list:
                    channel.plot(
                        plot_axes=subax,
                        start=ov_start,
                        stop=ov_stop,
                        resolution=resolution,
                        time_unit=time_unit,
                        reference_time=reference_time,
                    )
                subax.set_xlim(
                    (
                        (ov_start - reference_time)
                        / pd.to_timedelta(1, unit=time_unit),
                        (ov_stop - reference_time) / pd.to_timedelta(1, unit=time_unit),
                    )
                )
                subax.grid(False)

        repaint_overview_plot()
        repaint_plot(start, stop)

        interactive_plot = widgets.AppLayout(
            header=tab, center=fig.canvas, pane_heights=[1, 4, 0]
        )

        def tab_listener(event):
            mode = InteractionMode(event["new"])
            if mode == InteractionMode.ANNOTATE:
                pass

            elif mode == InteractionMode.ADJUST:
                pass

            elif mode == InteractionMode.SETTINGS:
                pass

        partial_interval_data = None
        shifting_reference_time = None

        def key_press_listener(event: KeyEvent):
            nonlocal \
                start, \
                stop, \
                fig, \
                partial_interval_data, \
                shifting_reference_time, \
                shift_span

            if event.key in "123456789":
                new_index = int(event.key) - 1
                if new_index < len(label_dropdown.options):
                    label_dropdown.index = new_index
            elif event.key == "0":
                new_index = 9
                if new_index < len(label_dropdown.options):
                    label_dropdown.index = new_index
            elif event.key == "right":
                start += shift_span
                stop += shift_span
                repaint_plot(start, stop)
            elif event.key == "left":
                start -= shift_span
                stop -= shift_span
                repaint_plot(start, stop)
            elif event.key == "+":  # Zoom in
                span = stop - start
                start += span * 0.25
                stop -= span * 0.25
                shift_span = (stop - start) * 0.25

                repaint_plot(start, stop)
            elif event.key == "-":  # Zoom out
                span = stop - start
                start -= span * 0.25
                stop += span * 0.25
                shift_span = (stop - start) * 0.25

                repaint_plot(start, stop)
            elif event.key == "d":
                delete_toggle_handler({"new": not DELETE_ANNOTATIONS})
            elif event.key == "escape":
                partial_interval_data = None
                shifting_reference_time = None
                fig.canvas._figure_label = ""
                repaint_plot(start, stop)

        def mouse_click_listener(event: MouseEvent):
            nonlocal fig, partial_interval_data, shifting_reference_time, start, stop

            current_mode = InteractionMode(tab.selected_index)
            current_axes = event.inaxes
            if (
                current_axes in channel_axes
            ):  # If click is within current detail plot, annotate something
                if event.button is MouseButton.RIGHT:
                    if current_mode == InteractionMode.ANNOTATE:
                        active_label: Label = label_dropdown.value
                        if isinstance(active_label, IntervalLabel):
                            if DELETE_ANNOTATIONS:
                                time_data = (
                                    event.xdata * pd.to_timedelta(1, unit=time_unit)
                                    + reference_time
                                )
                                selected_intervals = (
                                    np.argwhere(
                                        (active_label.intervals[:, 0] <= time_data)
                                        & (active_label.intervals[:, 1] >= time_data)
                                    )
                                    .flatten()
                                    .tolist()
                                )
                                if len(selected_intervals) > 0:
                                    interval_index = min(selected_intervals)
                                    active_label.remove_data(
                                        active_label.intervals[interval_index, :]
                                    )
                                    repaint_plot(start, stop)
                                return

                            if partial_interval_data is None:
                                partial_interval_data = (event.xdata, event.ydata)
                                fig.canvas._figure_label = (
                                    "Creating interval label, select end point "
                                    "or press <ESC> to abort ..."
                                )

                            else:
                                t1, y1 = partial_interval_data
                                t2, y2 = event.xdata, event.ydata
                                t1 = reference_time + t1 * pd.to_timedelta(
                                    1, unit=time_unit
                                )
                                t2 = reference_time + t2 * pd.to_timedelta(
                                    1, unit=time_unit
                                )
                                if t2 < t1:
                                    t1, t2 = t2, t1
                                y = (y1 + y2) / 2
                                if value_type_dropdown.value == LabelValueType.TEXTUAL:
                                    y = value_text_input.value
                                active_label.add_data((t1, t2), value=y)
                                repaint_plot(start, stop)
                                partial_interval_data = None
                                fig.canvas._figure_label = ""
                        else:
                            time_data = (
                                event.xdata * pd.to_timedelta(1, unit=time_unit)
                                + reference_time
                            )
                            if DELETE_ANNOTATIONS:
                                tolerance = (
                                    (stop - start)
                                    / screen_pixel_width
                                    * CANVAS_SELECTION_TOLERANCE_PX
                                )
                                selected_times, _ = active_label.get_data(
                                    start=time_data - tolerance,
                                    stop=time_data + tolerance,
                                )
                                if len(selected_times) > 0:
                                    selected_time = min(selected_times)
                                    active_label.remove_data(time_data=selected_time)
                            else:
                                ydata = event.ydata
                                if value_type_dropdown.value == LabelValueType.TEXTUAL:
                                    ydata = value_text_input.value
                                active_label.add_data(time_data, value=ydata)
                            repaint_plot(start, stop)

                    elif current_mode == InteractionMode.ADJUST:
                        if shifting_reference_time is None:
                            current_axes.axvline(
                                x=event.xdata,
                                color="red",
                                linestyle="--",
                                linewidth=1.5,
                            )
                            shifting_reference_time = (
                                event.xdata * pd.to_timedelta(1, unit=time_unit)
                                + reference_time
                            )
                        else:
                            offset = (
                                event.xdata * pd.to_timedelta(1, unit=time_unit)
                                + reference_time
                                - shifting_reference_time
                            )
                            for channel in shifting_channel_selection.value:
                                channel: Channel
                                channel.shift_time_index(delta_t=offset)
                            shifting_reference_time = None
                            repaint_overview_plot()
                            repaint_plot(start, stop)

            elif (
                current_axes in overview_axes
            ):  # if click is within overview plot: move there
                time_data = (
                    event.xdata * pd.to_timedelta(1, unit=time_unit) + reference_time
                )
                plot_span = stop - start
                stop = time_data + 0.5 * plot_span
                start = time_data - 0.5 * plot_span
                repaint_plot(start, stop)

        def mouse_move_event(event: MouseEvent):
            nonlocal x_indicators

            if event.xdata is None:
                return

            for indicator in x_indicators:
                indicator.set_xdata(np.ones_like(indicator.get_xdata()) * event.xdata)

            fig.canvas.draw_idle()

        def scroll_event(event: MouseEvent):
            mouse_y = event.ydata
            axplot = event.inaxes
            if axplot is None:
                return
            bottom, top = axplot.axes.get_ylim()
            scale_factor = 0.1 * event.step
            new_bottom = mouse_y - ((mouse_y - bottom) * (1 + scale_factor))
            new_top = mouse_y + ((top - mouse_y) * (1 + scale_factor))

            axplot.axes.set_ylim(new_bottom, new_top)
            update_ylim_settings()
            fig.canvas.draw_idle()

        def settings_apply_handler(event):
            for ax, limit_widget in zip(channel_axes, limit_widgets):
                _, min_input, max_input = limit_widget.children
                ax.set_ylim((min_input.value, max_input.value))

            fig.canvas.draw_idle()

        tab.observe(tab_listener, names="selected_index")
        settings_apply_button.on_click(settings_apply_handler)
        fig.canvas.mpl_connect("key_press_event", key_press_listener)
        fig.canvas.mpl_connect("button_press_event", mouse_click_listener)
        fig.canvas.mpl_connect("motion_notify_event", mouse_move_event)
        fig.canvas.mpl_connect("scroll_event", scroll_event)
        fig.canvas.capture_scroll = True
        fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.97)

        interactive_plot.center._figure_label = ""  # is overwritten for some reason
        return interactive_plot
