"""What is plotted: resolution of channel / label specifications, time range
and time unit of plots."""

from __future__ import annotations

import itertools as it
import logging

from dataclasses import dataclass, field

import pandas as pd

from vitabel.timeseries import Channel, IntervalLabel, Label, TimeDataCollection
from vitabel.typing import (
    ChannelSpecification,
    LabelSpecification,
    Timedelta,
    Timestamp,
)

logger = logging.getLogger("vitabel")


def parse_channel_specification(
    collection, channels: list[list[ChannelSpecification | int]] | None
) -> list[list[Channel]]:
    """Parse (nested) channel specifications into nested lists of channels.

    Parameters
    ----------
    channels
        The nested list of channel specifications to parse.
    """
    channel_lists = []
    if channels is None:
        channel_lists.append(collection.channels)
    else:
        for spec_list in channels:
            channel_list: list[Channel] = []
            for spec in spec_list:
                if isinstance(spec, str):
                    channel_list.extend(collection.get_channels(name=spec))
                elif isinstance(spec, dict):
                    channel_list.extend(collection.get_channels(**spec))
                elif isinstance(spec, int):
                    channel_list.append(collection.channels[spec])
                elif isinstance(spec, Channel):
                    channel_list.append(spec)
                else:
                    raise ValueError(f"Invalid channel specification: {spec}")
            channel_lists.append(channel_list)
    return channel_lists


def parse_label_specification(
    collection,
    labels: list[list[LabelSpecification | int]] | None,
    channel_lists: list[list[Channel]],
    include_attached_labels: bool = False,
) -> list[list[Label]]:
    """Parse (nested) label specifications into nested lists of labels.

    Parameters
    ----------
    labels
        The nested label specifications to parse.
    channel_lists
        The nested channel lists that the labels are attached to.
    include_attached_labels
        Whether to include attached labels in the output.
    """
    num_subplots = len(channel_lists)
    label_lists: list[list[Label]] = []

    if labels is None:
        for _ in range(num_subplots):
            label_lists.append([label for label in collection.global_labels])
    else:
        for spec_list in labels:
            label_list: list[Label] = []
            for spec in spec_list:
                if isinstance(spec, str):
                    label_list.extend(collection.get_labels(name=spec))
                elif isinstance(spec, dict):
                    label_list.extend(collection.get_labels(**spec))
                elif isinstance(spec, int):
                    label_list.append(collection.labels[spec])
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


def get_time_extremum(
    collection,
    time: Timestamp | Timedelta | float | str | None,
    channel_lists: list[list[Channel]],
    minimum: bool = True,
) -> Timestamp | Timedelta | float | None:
    """Get the minimum or maximum time value from the specified channels,
    or return the specified time value if it is not ``None``.

    Parameters
    ----------
    time
        The time value to compare to the channel time values. The extremum
        is only calculated if this value is ``None``.
    channel_lists
        The channel lists to get the extremum time value from.
    minimum
        If ``True`` (the default), the minimum time value is returned,
        otherwise the maximum time value is returned.
    """
    op = min
    if not minimum:
        op = max

    time = collection._parse_time(time)
    if time is None:
        time_list = []
        for channel in it.chain.from_iterable(channel_lists):
            if channel.is_empty():
                continue
            # vectorized (the builtins min / max iterate over every entry)
            ex_time = channel.time_index.min() if minimum else channel.time_index.max()
            if collection.is_time_absolute():
                ex_time += channel.time_start
            time_list.append(ex_time)
        if len(time_list) > 0:
            time = op(time_list)
    return time


def get_timeunit_from_channels(collection, channel_lists: list[list[Channel]]) -> str:
    """Get the time unit from the specified channels.

    Ensures that all channels have the same time unit, and raises an error
    if they do not. In that case, the time unit must be passed explicitly
    to the calling method.

    Parameters
    ----------
    channel_lists
        The channel lists to get the time unit from.
    """
    channel_iter = it.chain.from_iterable(channel_lists)
    time_unit = next(channel_iter).time_unit
    if not all(channel.time_unit == time_unit for channel in channel_iter):
        raise ValueError(
            "The channel time units are not uniform. Specify the plot time "
            "unit explicitly by specifying the time_unit argument"
        )
    return time_unit


@dataclass
class PlotSpec:
    """Resolved description of a (static or interactive) plot: which series
    are plotted in which subplot, and the time range and unit.

    Build it with :meth:`from_arguments`.
    """

    collection: TimeDataCollection
    channels: list[list[Channel]]
    """Channels per subplot."""
    labels: list[list[Label]]
    """Labels per subplot."""
    overviews: list[list[Channel]] = field(default_factory=list)
    """Channels per overview subplot."""
    start: Timestamp | Timedelta | None = None
    stop: Timestamp | Timedelta | None = None
    time_unit: str = "s"
    limited_overview: bool = False

    @property
    def reference_time(self) -> Timestamp | Timedelta | None:
        """Time at the origin of the x axis (the plot start)."""
        return self.start

    @property
    def annotation_labels(self) -> list[Label]:
        """The plotted labels in the order offered for annotation: point
        labels by name, then interval labels by name."""
        unique = []  # labels define __eq__ (not hashable): compare like before
        for label in it.chain.from_iterable(self.labels):
            if label not in unique:
                unique.append(label)
        points = [label for label in unique if not isinstance(label, IntervalLabel)]
        intervals = [label for label in unique if isinstance(label, IntervalLabel)]
        return sorted(points, key=lambda label: label.name) + sorted(
            intervals, key=lambda label: label.name
        )

    def overview_range(self, index: int) -> tuple[Timestamp | Timedelta, ...]:
        """Time range of an overview subplot: all plotted channels, or (with
        ``limited_overview``) the channels of this overview."""
        channels = [self.overviews[index]] if self.limited_overview else self.channels
        return (
            get_time_extremum(self.collection, None, channels, minimum=True),
            get_time_extremum(self.collection, None, channels, minimum=False),
        )

    @classmethod
    def from_arguments(
        cls,
        collection: TimeDataCollection,
        channels: list[list[ChannelSpecification | int]] | None = None,
        labels: list[list[LabelSpecification | int]] | None = None,
        start: Timestamp | Timedelta | float | str | None = None,
        stop: Timestamp | Timedelta | float | str | None = None,
        time_unit: str | None = None,
        include_attached_labels: bool = False,
        channel_overviews: list[list[ChannelSpecification | int]] | bool = False,
        limited_overview: bool = False,
        range_from_labels: bool = True,
        empty_span: Timedelta = pd.Timedelta(minutes=1),
    ) -> PlotSpec:
        """Resolve the arguments of :meth:`.TimeDataCollection.plot` /
        :meth:`.TimeDataCollection.plot_interactive`.

        Parameters
        ----------
        range_from_labels
            Whether the default time range includes the labels (interactive
            plot) or only the channels (static plot).
        empty_span
            Length of the time range if the series contain no data (starting
            at the current time).
        """
        channel_lists = parse_channel_specification(collection, channels)
        label_lists = parse_label_specification(
            collection,
            labels,
            channel_lists,
            include_attached_labels=include_attached_labels,
        )
        if channel_overviews is False:
            overviews = []
        elif channel_overviews is True:
            # plotted channels, in order of appearance
            overviews = [list(dict.fromkeys(it.chain.from_iterable(channel_lists)))]
        else:
            overviews = parse_channel_specification(collection, channel_overviews)

        range_sources = (
            channel_lists + label_lists if range_from_labels else channel_lists
        )
        if not any(channel_lists):
            range_sources = label_lists  # no channels: the range of the labels
        start = get_time_extremum(collection, start, range_sources, minimum=True)
        stop = get_time_extremum(collection, stop, range_sources, minimum=False)
        if start is None and stop is None:
            logger.warning(
                "Specified channels and labels contain no data, setting start "
                "to the current time."
            )
            start = pd.Timestamp.now()
            stop = start + empty_span
        if start == stop:
            logger.warning(
                "Start and stop time are equal, setting stop time to "
                f"{empty_span} after start to allow plotting."
            )
            stop = start + empty_span

        if time_unit is None:
            if any(channel_lists):
                time_unit = get_timeunit_from_channels(collection, channel_lists)
            else:  # no channels: the unit of the labels, if they agree
                units = {
                    label.time_unit for label in it.chain.from_iterable(label_lists)
                }
                time_unit = units.pop() if len(units) == 1 else "s"

        return cls(
            collection=collection,
            channels=channel_lists,
            labels=label_lists,
            overviews=overviews,
            start=start,
            stop=stop,
            time_unit=time_unit,
            limited_overview=limited_overview,
        )
