"""What is plotted: resolution of channel / label specifications, time range
and time unit of plots."""

from __future__ import annotations

import itertools as it

from vitabel.timeseries import Channel, Label
from vitabel.typing import (
    ChannelSpecification,
    LabelSpecification,
    Timedelta,
    Timestamp,
)


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
    Empty labels are excluded.

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
            ex_time = op(channel.time_index)
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
