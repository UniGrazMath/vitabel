"""Interactive plots (matplotlib, ipympl and ipywidgets)."""

from __future__ import annotations

import itertools as it
import logging

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.text import Text

from vitabel.timeseries import Channel, IntervalLabel, Label
from vitabel.typing import (
    ChannelSpecification,
    LabelSpecification,
    Timedelta,
    Timestamp,
)

logger = logging.getLogger("vitabel")


def plot_interactive(
    collection,
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
    """Implementation of :meth:`.TimeDataCollection.plot_interactive`."""
    import ipywidgets as widgets
    from enum import Enum
    from IPython import get_ipython
    from matplotlib.backend_bases import MouseButton, MouseEvent, KeyEvent

    CANVAS_SELECTION_TOLERANCE_PX = 5

    channel_lists = collection._parse_channel_specification(channels)
    if channel_overviews is False:
        channel_overviews = []
    elif channel_overviews is True:
        channel_overviews = [list(set(it.chain.from_iterable(channel_lists)))]
    else:
        channel_overviews = collection._parse_channel_specification(channel_overviews)

    if subplots_kwargs is None:
        subplots_kwargs = {}

    label_lists = collection._parse_label_specification(
        labels, channel_lists, include_attached_labels=include_attached_labels
    )

    all_data_sources = channel_lists + label_lists

    num_subplots = len(channel_lists) + len(channel_overviews)
    start = collection._get_time_extremum(start, all_data_sources, minimum=True)
    stop = collection._get_time_extremum(stop, all_data_sources, minimum=False)
    if start is None and stop is None:
        logger.warning(
            "Specified channels and labels contain no data, setting start "
            "to the current time."
        )
        start = pd.Timestamp.now()
        stop = start + pd.Timedelta(minutes=1)

    if start == stop:
        logger.warning(
            "Start and stop time are equal, setting stop time to "
            "1 minute after start to allow plotting."
        )
        stop = start + pd.Timedelta(minutes=1)

    reference_time = start
    shift_span = (stop - start) * 0.25

    if time_unit is None:
        time_unit = collection._get_timeunit_from_channels(channel_lists)

    ipy_shell = get_ipython()
    if ipy_shell is None:
        raise RuntimeError("This method can only be used in an IPython environment")

    # ---------- WIDGETS FOR ANNOTATION ----------------------
    # --------------------------------------------------------
    class InteractionMode(Enum):
        ANNOTATE = 0
        ADJUST = 1
        SETTINGS = 2

    value_text_input = widgets.Text(placeholder="Label text ...")
    value_text_stack = widgets.Stack(
        [
            widgets.HTML(),
            value_text_input,
        ]
    )
    value_text_stack.selected_index = 1

    distinct_labels = {"single": [], "interval": []}
    for label_list in label_lists:
        for label in label_list:
            if isinstance(label, IntervalLabel):
                if label not in distinct_labels["interval"]:
                    distinct_labels["interval"].append(label)
            elif label not in distinct_labels["single"]:
                distinct_labels["single"].append(label)

    label_master = sorted(
        distinct_labels["single"], key=lambda label: label.name
    ) + sorted(distinct_labels["interval"], key=lambda label: label.name)
    label_dict = dict(enumerate(label_master, start=1))
    dropdown_options = [
        (f"{label.name}   [Interval]", i)
        if isinstance(label, IntervalLabel)
        else (label.name, i)
        for i, label in label_dict.items()
    ]

    label_dropdown = widgets.Dropdown(
        options=dropdown_options,
        disabled=False,
        layout=widgets.Layout(margin="3px 0px 0px 0px"),
    )
    DELETE_ANNOTATIONS = False
    delete_toggle_button = widgets.ToggleButton(
        value=False,
        description="Mode: Add Data",
        disabled=False,
        button_style="success",
    )

    add_timestamp_check = widgets.Checkbox(
        value=True,
        description="",
        disabled=True,
        indent=False,
        layout=widgets.Layout(width="max-content"),
    )

    add_numeric_check = widgets.Checkbox(
        value=True,
        description="",
        disabled=False,
        indent=False,
        layout=widgets.Layout(width="max-content"),
    )

    add_text_check = widgets.Checkbox(
        value=True,
        description="",
        disabled=False,  # TODO make this dynamic in conjuncton with value_text_input
        indent=False,
        layout=widgets.Layout(width="max-content"),
    )

    def toggle_add_text(change):
        if change["name"] == "value":
            value_text_stack.selected_index = 1 if change["new"] else 0

    add_text_check.observe(toggle_add_text, names="value")

    add_stack = widgets.Stack(
        [
            widgets.HTML(),  # alternative to empty widget
            widgets.HBox(
                [  # Hbox to add alligned Label
                    widgets.Label(  # alligned label
                        "Data to add:",
                        layout=widgets.Layout(
                            min_width="85px",
                            margin="0px 5px 0px 0px",
                            text_align="right",
                            justify_content="flex-end",
                        ),
                    ),
                    widgets.HBox(
                        [  # Hbox for content
                            widgets.VBox(
                                [widgets.Label("Timestamp"), add_timestamp_check],
                                layout=widgets.Layout(margin="0 20px 0 0"),
                            ),
                            widgets.VBox(
                                [widgets.Label("Numeric value"), add_numeric_check],
                                layout=widgets.Layout(margin="0 20px 0 0"),
                            ),
                            widgets.VBox(
                                [
                                    widgets.Label("Textual value"),
                                    widgets.HBox([add_text_check, value_text_stack]),
                                ]
                            ),
                        ],
                        layout=widgets.Layout(
                            border="1px dotted gray", padding="3px 8px"
                        ),
                    ),
                ]
            ),
        ],
        layout=widgets.Layout(margin="3px 0 0 0 "),  # top margin only
    )
    add_stack.selected_index = 1

    def delete_toggle_handler(change):
        nonlocal DELETE_ANNOTATIONS, partial_interval_data, shifting_reference_time
        partial_interval_data = None
        shifting_reference_time = None
        fig.canvas._figure_label = " "
        active_label = label_dict[label_dropdown.value]
        if change["new"]:  # value of "new" attribute is new button value
            delete_toggle_button.description = "Mode: Delete Data"
            delete_toggle_button.button_style = "danger"
            add_stack.selected_index = 0
            DELETE_ANNOTATIONS = True
            if isinstance(active_label, IntervalLabel):
                fig.canvas._figure_label = (
                    "To delete an Interval click close to its "
                    "start with the right mouse button."
                )
        else:
            delete_toggle_button.description = "Mode: Add Data"
            delete_toggle_button.button_style = "success"
            add_stack.selected_index = 1
            DELETE_ANNOTATIONS = False
            if isinstance(active_label, IntervalLabel):
                fig.canvas._figure_label = (
                    "Right-click to set start time, then "
                    "right-click again to set end time."
                )

    def label_dropdown_change(change):
        nonlocal partial_interval_data, shifting_reference_time
        partial_interval_data = None
        shifting_reference_time = None
        fig.canvas._figure_label = " "
        label = label_dict[label_dropdown.value]
        if isinstance(label, IntervalLabel):
            if DELETE_ANNOTATIONS:
                fig.canvas._figure_label = (
                    "To delete an Interval click close to its "
                    "start with the right mouse button."
                )
            else:
                fig.canvas._figure_label = (
                    "Right-click to set start time, then "
                    "right-click again to set end time."
                )
        _populate_label_add_menu(label)

    def _populate_label_add_menu(label: Label | IntervalLabel):
        value_text_input.value = ""

        # empty label -> presets on plottype and vline_text_source
        if len(label) == 0:
            # defaulting
            add_numeric_check.value = False
            add_text_check.value = False

            # preset by annotation_preset_type specification
            if label.annotation_preset_type is not None:
                if label.annotation_preset_type in ("numerical", "combined"):
                    add_numeric_check.value = True
                if label.annotation_preset_type in ("textual", "combined"):
                    add_text_check.value = True
                return

            # preset by plot_type and vline_text_source
            if label.vline_text_source in ("text_data", "combined"):
                add_text_check.value = True
                value_text_input.value = label.name
            if label.vline_text_source in ("data", "combined"):
                add_numeric_check.value = True

            if label.plot_type in ("scatter", "hline", "combined"):
                add_numeric_check.value = True
            if label.plot_type in ("vline", "combined"):
                add_text_check.value = True

        # non empty label with no data
        elif label.data is None and label.text_data is None:
            add_numeric_check.value = False
            add_text_check.value = False
        # label with data
        else:
            add_numeric_check.value = bool(label.data is not None)

            add_text_check.value = bool(label.text_data is not None)
            if add_text_check.value:
                strings = label.text_data[~pd.isna(label.text_data)]
                if strings.size > 0:
                    values, counts = np.unique(strings, return_counts=True)
                    value_text_input.value = values[np.argmax(counts)]

    delete_toggle_button.observe(delete_toggle_handler, names="value")

    label_dropdown.observe(label_dropdown_change, names="value")

    if label_dropdown.value in label_dict:
        _populate_label_add_menu(label_dict[label_dropdown.value])

    # ---------- WIDGETS FOR SHIFTING ------------------------
    # --------------------------------------------------------
    shifting_channel_selection = widgets.SelectMultiple(
        value=[collection.channels[0]],
        options=[
            (f"[{idx}] {chan.name}", chan)
            for idx, chan in enumerate(collection.channels)
        ],
        description="Channels / Labels",
        disabled=False,
        continuous_update=True,
    )

    # ----------- WIDGETS FOR SETTINGS -----------------------
    # --------------------------------------------------------
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

    # ------------------ ENTIRE WIDGET DESIGN ----------------
    # --------------------------------------------------------
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
                        widgets.Label(
                            "Label:",
                            layout=widgets.Layout(
                                min_width="85px",
                                margin="0px 5px 0px 0px",
                                text_align="right",
                                justify_content="flex-end",
                            ),
                        ),
                        label_dropdown,
                        delete_toggle_button,
                    ]
                ),
                add_stack,
            ],
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

    tab.layout = widgets.Layout(
        min_height="270px",  # NOTE avoids ynamics witht concomittant plot. yet not flexibel
    )
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
        if collection.is_time_absolute():
            fig.suptitle(f"Reference time: {reference_time}")

    x_indicators = [
        ax.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
        for ax in channel_axes
    ]
    overview_indicators = []

    partial_interval_data = None
    partial_interval_artist = None
    shifting_reference_time = None
    shifting_reference_axis = None
    is_repainting = False

    def update_ylim_settings():
        for ax, limit_widget in zip(channel_axes, limit_widgets):
            _, min_input, max_input = limit_widget.children
            ymin, ymax = ax.get_ylim()
            min_input.value = ymin
            max_input.value = ymax
            text_labels = ax.findobj(
                lambda artist: (
                    isinstance(artist, Text) and hasattr(artist, "_from_vitals_label")
                )
            )
            for artist in text_labels:
                artist.set_y(ymin + 0.1 * (ymax - ymin))

    def format_coords(x, y):
        format_string = f"(x, y) = ({x:.2f}, {y:.2f})"
        if collection.is_time_absolute():
            format_string += (
                f"   x = {x * pd.to_timedelta(1, unit=time_unit) + reference_time}"
            )

        return format_string

    def _axis_xlim_to_window(ax):
        xmin, xmax = ax.get_xlim()
        window_start = reference_time + pd.to_timedelta(xmin, unit=time_unit)
        window_stop = reference_time + pd.to_timedelta(xmax, unit=time_unit)
        if window_stop < window_start:
            window_start, window_stop = window_stop, window_start
        return window_start, window_stop

    def sync_view_from_axis(ax):
        nonlocal start, stop, shift_span, is_repainting

        if is_repainting:
            return

        start, stop = _axis_xlim_to_window(ax)
        shift_span = (stop - start) * 0.25
        repaint_plot(start, stop)

    def repaint_plot(start, stop):
        nonlocal \
            fig, \
            channel_axes, \
            overview_axes, \
            overview_indicators, \
            screen_pixel_width, \
            is_repainting
        data_width = (stop - start).total_seconds()
        screen_pixel_width, _ = fig.canvas.get_width_height()
        resolution = pd.Timedelta(seconds=data_width / screen_pixel_width)

        partial_interval_artist_ax = None
        if partial_interval_artist is not None:
            partial_interval_artist_ax = partial_interval_artist.axes

        is_repainting = True
        try:
            with plt.ioff():
                for channel_list, label_list, indicator, subax in zip(
                    channel_lists, label_lists, x_indicators, channel_axes
                ):
                    old_ylims = subax.get_ylim()
                    old_ylabel = subax.yaxis.get_label_text()
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
                    subax.yaxis.set_label_text(old_ylabel)
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
                if shifting_reference_time is not None:
                    shifting_reference_axis.axvline(
                        x=(shifting_reference_time - reference_time)
                        / pd.to_timedelta(1, unit=time_unit),
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                    )
                if partial_interval_artist is not None:
                    partial_interval_artist_ax.add_artist(partial_interval_artist)
        finally:
            is_repainting = False
        return

    def repaint_overview_plot():
        nonlocal overview_indicators
        channels_for_xlims = channel_lists
        for channel_list, subax in zip(channel_overviews, overview_axes):
            if not channel_list:
                continue

            if limited_overview:  # xlim of overview based on selected channels
                channels_for_xlims = [channel_list]

            ov_start = collection._get_time_extremum(
                time=None, channel_lists=channels_for_xlims, minimum=True
            )
            ov_stop = collection._get_time_extremum(
                time=None, channel_lists=channels_for_xlims, minimum=False
            )

            data_width = (ov_stop - ov_start).total_seconds()
            resolution = pd.Timedelta(seconds=data_width / screen_pixel_width)
            subax.clear()
            overview_indicators = []
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
                    (ov_start - reference_time) / pd.to_timedelta(1, unit=time_unit),
                    (ov_stop - reference_time) / pd.to_timedelta(1, unit=time_unit),
                )
            )
            subax.grid(False)

    repaint_overview_plot()
    repaint_plot(start, stop)

    for ax in channel_axes:
        ax.callbacks.connect("xlim_changed", sync_view_from_axis)

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
            fig.canvas._figure_label = " "
            repaint_plot(start, stop)

    def mouse_click_listener(event: MouseEvent):
        nonlocal \
            fig, \
            partial_interval_data, \
            partial_interval_artist, \
            shifting_reference_time, \
            shifting_reference_axis, \
            start, \
            stop

        current_mode = InteractionMode(tab.selected_index)
        current_axes = event.inaxes
        if (
            current_axes in channel_axes
        ):  # If click is within current detail plot, annotate something
            if event.button is MouseButton.RIGHT:
                if current_mode == InteractionMode.ANNOTATE:
                    active_label: Label = label_dict[label_dropdown.value]
                    if isinstance(active_label, IntervalLabel):
                        if DELETE_ANNOTATIONS:
                            time_data = (
                                pd.to_timedelta(event.xdata, unit=time_unit)
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
                            label_color = active_label.plotstyle.get(
                                "color", "limegreen"
                            )
                            partial_interval_artist = current_axes.axvline(
                                x=event.xdata,
                                color=label_color,
                                linestyle="--",
                                linewidth=1.5,
                            )
                            partial_interval_data = (event.xdata, event.ydata)
                            fig.canvas._figure_label = (
                                "Creating interval label, select end point "
                                "or press <ESC> to abort ..."
                            )

                        else:
                            t1, y1 = partial_interval_data
                            t2, y2 = event.xdata, event.ydata
                            t1 = reference_time + pd.to_timedelta(t1, unit=time_unit)
                            t2 = reference_time + pd.to_timedelta(t2, unit=time_unit)
                            if t2 < t1:
                                t1, t2 = t2, t1
                            ydata = (y1 + y2) / 2 if add_numeric_check.value else None
                            text_input = (
                                value_text_input.value
                                if add_text_check.value
                                and value_text_input.value
                                and value_text_input.value != ""
                                else None
                            )
                            active_label.add_data(
                                (t1, t2), value=ydata, text=text_input
                            )  # TODO
                            partial_interval_data = None
                            partial_interval_artist = None
                            repaint_plot(start, stop)
                            fig.canvas._figure_label = " "
                    else:
                        time_data = (
                            pd.to_timedelta(event.xdata, unit=time_unit)
                            + reference_time
                        )
                        if DELETE_ANNOTATIONS:
                            tolerance = (
                                (stop - start)
                                / screen_pixel_width
                                * CANVAS_SELECTION_TOLERANCE_PX
                            )
                            selected_times = active_label.get_data(
                                start=time_data - tolerance,
                                stop=time_data + tolerance,
                            ).time_index
                            if len(selected_times) > 0:
                                selected_time = min(selected_times)
                                active_label.remove_data(time_data=selected_time)
                        else:
                            ydata = event.ydata if add_numeric_check.value else None
                            text_input = (
                                value_text_input.value
                                if add_text_check.value
                                and value_text_input.value
                                and value_text_input.value != ""
                                else None
                            )
                            active_label.add_data(
                                time_data=time_data, value=ydata, text=text_input
                            )
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
                            pd.to_timedelta(event.xdata, unit=time_unit)
                            + reference_time
                        )
                        shifting_reference_axis = current_axes
                    else:
                        offset = (
                            pd.to_timedelta(event.xdata, unit=time_unit)
                            + reference_time
                            - shifting_reference_time
                        )
                        for channel in shifting_channel_selection.value:
                            channel: Channel
                            channel.shift_time_index(delta_t=offset)
                        shifting_reference_time = None
                        shifting_reference_axis = None
                        repaint_overview_plot()
                        repaint_plot(start, stop)

        elif (
            current_axes in overview_axes
        ):  # if click is within overview plot: move there
            time_data = pd.to_timedelta(event.xdata, unit=time_unit) + reference_time
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
