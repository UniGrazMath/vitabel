"""Operations on time series based on interval labels."""

from __future__ import annotations

from copy import deepcopy


from vitabel.timeseries import Channel, IntervalLabel, Label, TimeSeriesBase


def filter_by_intervallabel(
    channel_or_label_to_filter: Channel | Label,
    filter_by: IntervalLabel,
    invert: bool = False,
    start_inclusive: bool = True,
    end_inclusive: bool = True,
    full_cover: bool = False,
) -> Channel | Label:
    """Filters a channel or label by a given interval label.

    This creates a new channel or label containing only the data points
    that fall within the intervals specified in the given interval label.

    Parameters
    ----------
    channel_or_label_to_filter
        The channel or label to filter.
    filter_by
        The interval label to filter by.
    invert
        If ``True``, inverts the filter, i.e., keeps the parts
        outside the intervals defined by `filter_by`.
    start_inclusive
        If ``True``, the start of the intervals in `filter_by`
        is considered inclusive for filtering.
        If `invert` is ``True``, this parameter becomes inverted too.
    end_inclusive
        If ``True``, the end of the intervals in `filter_by`
        is considered inclusive for filtering.
        If `invert` is ``True``, this parameter becomes inverted too.
    full_cover
        Only relevant for interval Labels:
        - If True, keep label intervals only if they are **fully contained** in at least one filter interval.
        - If False, keep label intervals if they **overlap at all** with any filter interval.

    Returns
    -------
    Channel | Label
        The filtered channel or label.
    """

    if not isinstance(channel_or_label_to_filter, (Channel, Label)):
        raise ValueError(
            f"The specified channel_or_label_to_filter {channel_or_label_to_filter} is neither a "
            "Channel nor a Label"
        )
    if not isinstance(filter_by, IntervalLabel):
        raise ValueError(f"The specified filter_by {filter_by} is not an IntervalLabel")

    # Get timestamps and the data of the channel or label
    channel_or_label_to_filter = deepcopy(channel_or_label_to_filter)

    # early return for empty
    if len(channel_or_label_to_filter) == 0:
        return channel_or_label_to_filter

    object_data = channel_or_label_to_filter.get_data()
    if isinstance(channel_or_label_to_filter, IntervalLabel):
        filter_mask = filter_by.contains_time(
            channel_or_label_to_filter.intervals,
            include_start=start_inclusive,
            include_end=end_inclusive,
            interval_require_full_containment=full_cover,
        )
    else:
        filter_mask = filter_by.contains_time(
            object_data.time_index.to_numpy(),
            include_start=start_inclusive,
            include_end=end_inclusive,
        )

    if invert:
        filter_mask = ~filter_mask

    filtered_time_index = object_data.time_index[filter_mask]
    filtered_time_index = filtered_time_index.ravel()

    # Filter the given Channel or Label
    TS = TimeSeriesBase(filtered_time_index - channel_or_label_to_filter.offset)
    new_index = TS.time_index
    new_start = TS.time_start
    channel_or_label_to_filter.time_index = new_index
    channel_or_label_to_filter.time_start = new_start

    if object_data.data is not None and len(object_data.data) > 0:
        channel_or_label_to_filter.data = object_data.data[filter_mask]

    if object_data.text_data is not None and len(object_data.text_data) > 0:
        assert isinstance(channel_or_label_to_filter, Label)
        channel_or_label_to_filter.text_data = object_data.text_data[filter_mask]

    return channel_or_label_to_filter
