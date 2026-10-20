"""Static matplotlib plots of channels, labels, and collections."""

from __future__ import annotations

import logging
import numbers
import typing
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from vitabel.plotting.data import minmax_decimate
from vitabel.plotting.layout import PlotSpec
from vitabel.typing import (
    ChannelSpecification,
    IntervalLabelPlotType,
    LabelPlotType,
    LabelPlotVLineTextSource,
    LabelSpecification,
    Timedelta,
    Timestamp,
)

logger = logging.getLogger("vitabel")


def _relative_times(times, series, reference_time):
    """Times relative to the reference time: the given one, or by default the
    start of an absolute series (relative series are plotted as they are).
    ``times`` is a time index, or an array of interval end points."""
    if len(times) == 0:
        # empty labels may have lost their time_start: no reference needed,
        # just the right (timedelta) type
        return times - times
    if reference_time is None:
        if not series.is_time_absolute():
            return times
        reference_time = series.time_start
    if isinstance(times, np.ndarray):  # interval end points
        if isinstance(reference_time, pd.Timestamp):
            return times - np.datetime64(reference_time)
        return times - np.timedelta64(pd.Timedelta(reference_time))
    return times - reference_time


def plot_channel(
    channel,
    plot_axes: plt.Axes | None = None,
    plotstyle: dict[str, Any] | None = None,
    start: Timestamp | Timedelta | float | None = None,
    stop: Timestamp | Timedelta | float | None = None,
    resolution: Timedelta | float | None = None,
    time_unit: str | None = None,
    reference_time: Timestamp | Timedelta | float | None = None,
):
    """Implementation of :meth:`.Channel.plot`."""

    channel_data = channel.get_data(start=start, stop=stop, include_adjacent=True)
    time_index = channel_data.time_index
    data = channel_data.data
    if data is None:
        data = np.zeros_like(time_index, dtype=float)

    # when plotting a single channel, the times should always
    # be numeric (matplotlib's datetime formatter is not ideal,
    # the labels have lots of overlap).
    time_index = _relative_times(time_index, channel, reference_time)

    if time_unit is None:
        time_unit = channel.time_unit
    unit = pd.to_timedelta(1, unit=time_unit)
    time_index /= unit

    if resolution is not None and len(time_index) > 0:
        # keep the minimum and maximum per interval of the resolution's length
        # (peaks are kept), in buckets aligned to the plotted window
        if isinstance(resolution, str):
            resolution = pd.to_timedelta(resolution)
        elif isinstance(resolution, numbers.Number):
            resolution = pd.to_timedelta(resolution, unit=channel.time_unit)
        x = np.asarray(time_index, dtype=float)
        x0, x1 = x[0], x[-1]
        bounds = (
            _relative_times(
                pd.Index(
                    [
                        channel.convert_time_input(t)
                        for t in (start, stop)
                        if t is not None
                    ]
                ),
                channel,
                reference_time,
            )
            / unit
        )
        if start is not None:
            x0 = bounds[0]
        if stop is not None:
            x1 = bounds[-1]
        n_buckets = int(np.ceil((x1 - x0) / (resolution / unit)))
        x, data = minmax_decimate(x, np.asarray(data, dtype=float), x0, x1, n_buckets)
        time_index = x

    if plot_axes is None:
        figure, plot_axes = plt.subplots()
    else:
        figure = plot_axes.get_figure()

    base_plotstyle = channel.plotstyle.copy() if channel.plotstyle is not None else {}
    base_plotstyle.update(plotstyle if plotstyle is not None else {})

    (line,) = plot_axes.plot(time_index, data, **base_plotstyle)
    if "label" not in base_plotstyle:
        line.set_label(channel.name)

    return figure


def plot_label(
    label,
    plot_axes: plt.Axes | None = None,
    plotstyle: dict[str, Any] | None = None,
    start: Timestamp | Timedelta | float | None = None,
    stop: Timestamp | Timedelta | float | None = None,
    time_unit: str | None = None,
    reference_time: Timestamp | Timedelta | float | None = None,
    plot_type: LabelPlotType | None = None,
    vline_text_source: LabelPlotVLineTextSource | None = None,
):
    """Implementation of :meth:`.Label.plot`."""

    if plot_type is None:  # use plot_type from label
        plot_type = label.plot_type

    if plot_type not in typing.get_args(LabelPlotType):
        raise ValueError(f"Value '{plot_type}' is not a valid choice for plot_type")

    if vline_text_source is None:
        vline_text_source = label.vline_text_source

    if not label._check_plot_parameters(
        plot_type=plot_type, vline_text_source=vline_text_source
    ):
        logger.warning(
            f"Plotting specifications of the label '{label.name}' are inconsistent, "
            "data or labels might be missing from the plot."
        )

    time_index, data, text_data = label.get_data(
        start=start, stop=stop, include_adjacent=True
    )

    time_index = _relative_times(time_index, label, reference_time)

    if time_unit is None:
        time_unit = label.time_unit
    time_index /= pd.to_timedelta(1, unit=time_unit)

    if data is None:
        data = np.full_like(time_index, np.nan, dtype=float)

    if text_data is None:
        text_data = np.full_like(time_index, None, dtype=object)

    if plot_axes is None:
        figure, plot_axes = plt.subplots()
    else:
        figure = plot_axes.get_figure()

    base_plotstyle = label.plotstyle.copy()
    if plotstyle is not None:
        base_plotstyle.update(plotstyle)
    base_plotstyle.setdefault("label", label.name)

    if plot_type in {"scatter", "combined"}:
        nan_mask = np.isnan(data)
        if nan_mask.any() and plot_type == "scatter":
            logger.warning(
                f"Data in label {label.name} contains NaN values, "
                "skipping them in the scatter plot"
            )

        if (~nan_mask).any():
            if "marker" not in base_plotstyle and len(data[~nan_mask]) == 1:
                # no marker style set, override to make single value visible
                base_plotstyle.update({"marker": "X"})

            # can be assigned to variable, scatterplot_artist, if needed
            _ = plot_axes.plot(
                time_index[~nan_mask],
                data[~nan_mask],
                **base_plotstyle,
            )

    if plot_type in {"vline", "combined"}:
        if plot_type == "combined":
            # only need to plot lines where there is no scatter
            nan_mask = np.isnan(data)
            time_index = time_index[nan_mask]
            data = data[nan_mask]
            text_data = text_data[nan_mask]

        match vline_text_source:
            case "text_data":
                vline_text = text_data
            case "data":
                vline_text = data.astype(str)
                vline_text[np.isnan(data)] = ""  # replace NaN with empty string
            case "combined":
                vline_text = text_data.astype(str)
                vline_text[np.isnan(text_data)] = data.astype(str)[np.isnan(text_data)]
            case "disabled":
                vline_text = np.full_like(time_index, "", dtype=str)
            case _:
                raise ValueError(
                    f"Invalid choice for vline_text_source: '{vline_text_source}'"
                )

        if plotstyle is None:  # TODO: think about direction
            base_plotstyle.update({"marker": None})
        base_plotstyle.setdefault("label", label.name)

        ymin, ymax = plot_axes.get_ylim()
        for i, (t, text) in enumerate(zip(time_index, vline_text)):
            vline_artist = plot_axes.axvline(t, **base_plotstyle)
            if i == 0:
                base_plotstyle.pop("label", None)
            if text:
                line_color = vline_artist.get_color()
                vline_text_artist = plot_axes.text(
                    t,
                    0.9 * ymin + 0.1 * ymax,
                    text,
                    rotation=90,
                    clip_on=True,
                    color=line_color,
                    bbox=label.plot_vline_bbox_settings,
                )
                vline_text_artist._from_vitals_label = True

    return figure


def plot_interval_label(
    label,
    plot_axes: plt.Axes | None = None,
    plotstyle: dict[str, Any] | None = None,
    start: Timestamp | Timedelta | float | None = None,
    stop: Timestamp | Timedelta | float | None = None,
    time_unit: str | None = None,
    reference_time: Timestamp | Timedelta | float | None = None,
    plot_type: IntervalLabelPlotType | None = None,
):
    """Implementation of :meth:`.IntervalLabel.plot`."""
    time_index, data, text_data = label.get_data(start=start, stop=stop)
    artist = None

    if plot_type is None:  # use plot_type from label
        plot_type = label.plot_type

    if plot_type not in typing.get_args(IntervalLabelPlotType):
        raise ValueError(f"Value '{plot_type}' is not a valid choice for plot_type")

    time_index = _relative_times(time_index, label, reference_time)

    if time_unit is None:
        time_unit = label.time_unit
    time_index /= pd.to_timedelta(1, unit=time_unit)

    if plot_axes is None:
        figure, plot_axes = plt.subplots()
    else:
        figure = plot_axes.get_figure()

    base_plotstyle = label.plotstyle.copy()
    if plotstyle is not None:
        base_plotstyle.update(plotstyle)

    if plot_type in {"box", "combined"}:
        if "alpha" not in base_plotstyle:
            base_plotstyle.update({"alpha": 0.2})
        if "color" not in base_plotstyle:
            base_plotstyle.update({"color": "blue"})
        if not any(k in base_plotstyle for k in ("facecolor", "fc")):
            base_plotstyle.update({"facecolor": base_plotstyle["color"]})

        if plot_type == "combined" and data is not None:
            box_time_index = time_index[np.isnan(data)]
        else:
            box_time_index = time_index

        # Filter for props in kwargs which can be handled by axvspan / pathces.Rectangle
        rectangle_props = Rectangle((0, 0), 1, 1).properties().keys()
        filtered_base_plotstyle = {
            k: v for k, v in base_plotstyle.items() if k in rectangle_props
        }
        filtered_base_plotstyle.setdefault("label", label.name)

        for i, (xmin, xmax) in enumerate(box_time_index):
            artist = plot_axes.axvspan(xmin, xmax, **filtered_base_plotstyle)
            if i == 0:
                filtered_base_plotstyle.pop("label", None)

    if plot_type == "hline" and len(label) > 0 and data is None:
        logger.warning(f"Label {label.name} has no data, skipping horizontal line plot")
        return figure

    if plot_type in {"hline", "combined"} and data is not None:
        if plot_type == "combined":
            hline_time_index = time_index[~np.isnan(data)]
            hline_data = data[~np.isnan(data)]
        else:
            hline_time_index = time_index
            hline_data = data

        time_midpoints = np.mean(hline_time_index, axis=1)
        time_radius = np.diff(hline_time_index, axis=1).reshape(-1) / 2.0

        # TODO: deal with text_data
        # a fresh merge: the box styling above adds defaults for boxes only
        base_plotstyle = {**label.plotstyle, **(plotstyle or {})}
        filtered_base_plotstyle = {
            k: v for k, v in base_plotstyle.items() if k in Line2D([], []).properties()
        }
        filtered_base_plotstyle.update({"linestyle": ""})
        artist = plot_axes.errorbar(
            time_midpoints, hline_data, xerr=time_radius, **filtered_base_plotstyle
        )
        artist.set_label(label.name)

    return figure


def plot_collection(
    collection,
    channels: list[list[ChannelSpecification | int]] | None = None,
    labels: list[list[LabelSpecification | int]] | None = None,
    start: Timestamp | Timedelta | float | str | None = None,
    stop: Timestamp | Timedelta | float | str | None = None,
    resolution: Timedelta | float | None = None,
    time_unit: str | None = None,
    include_attached_labels: bool = False,
    subplots_kwargs: dict[str, Any] | None = None,
):
    """Implementation of :meth:`.TimeDataCollection.plot`."""
    spec = PlotSpec.from_arguments(
        collection,
        channels=channels,
        labels=labels,
        start=start,
        stop=stop,
        time_unit=time_unit,
        include_attached_labels=include_attached_labels,
        range_from_labels=False,
        empty_span=pd.Timedelta(hours=1),
    )
    channel_lists, label_lists = spec.channels, spec.labels
    start, stop, time_unit = spec.start, spec.stop, spec.time_unit
    num_subplots = len(channel_lists)
    if subplots_kwargs is None:
        subplots_kwargs = {}

    fig, axes = plt.subplots(num_subplots, squeeze=False, **subplots_kwargs)
    if collection.is_time_absolute():
        fig.suptitle(f"Reference time: {start}")
    axes = axes[:, 0]

    if resolution is None:
        screen_pixel_width, screen_pixel_height = fig.canvas.get_width_height()
        data_width = (stop - start).total_seconds()
        resolution = pd.Timedelta(seconds=data_width / screen_pixel_width)

    for channel_list, label_list, subax in zip(channel_lists, label_lists, axes):
        for channel in channel_list:
            channel.plot(
                plot_axes=subax,
                start=start,
                stop=stop,
                resolution=resolution,
                time_unit=time_unit,
                reference_time=start,
            )

        for label in label_list:
            label.plot(
                plot_axes=subax,
                start=start,
                stop=stop,
                time_unit=time_unit,
                reference_time=start,
            )

        plot_duration = (stop - start) / pd.to_timedelta(1, unit=time_unit)
        subax.set_xlim((0, plot_duration))
        subax.set_xlabel(f"time [{time_unit}]", labelpad=-12, fontsize=7)
        subax.grid(True)
        subax.legend(loc="upper right")

    return fig, axes
