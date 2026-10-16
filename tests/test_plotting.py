"""Behaviour of the plotting routines that the plotting refactoring (Phase 3,
M0) must preserve.

The interactive plot is driven headless: an IPython shell without GUI
integration, the ipympl canvas, and synthetic matplotlib events.
Known bugs are deliberately not pinned here.
"""

from __future__ import annotations

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from IPython.core.interactiveshell import InteractiveShell
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from vitabel import Channel, IntervalLabel, Label, Vitals
from vitabel.plotting.data import minmax_decimate

START = pd.Timestamp("2024-01-01 12:00:00")


class HeadlessShell(InteractiveShell):
    """IPython shell without GUI event loop integration."""

    def enable_gui(self, gui=None):
        pass


@pytest.fixture(scope="module")
def _shell():
    # one shell for all tests: IPython keeps matplotlib hooks registered on it
    backend = matplotlib.get_backend()
    yield HeadlessShell.instance()
    plt.close("all")
    matplotlib.use(backend)


@pytest.fixture
def ipython(_shell):
    yield _shell
    plt.close("all")


def _recording() -> Vitals:
    recording = Vitals()
    t = np.arange(0, 600, 0.5)
    recording.add_channel(
        Channel("signal", START + pd.to_timedelta(t, "s"), np.sin(t / 10))
    )
    recording.add_channel(
        Channel("other", START + pd.to_timedelta(t, "s"), np.cos(t / 10))
    )
    recording.add_global_label(
        Label("events", time_index=[], data=[], time_start=START)
    )
    recording.add_global_label(
        Label("notes", time_index=[], data=[], time_start=START, plot_type="vline")
    )
    recording.add_global_label(IntervalLabel("phases", time_index=[], time_start=START))
    return recording


class Interactive:
    """An interactive plot with helpers to send key / mouse events."""

    def __init__(self, recording: Vitals, **kwargs):
        kwargs = {
            "channels": [["signal"], ["other"]],
            "labels": [["events", "notes", "phases"], []],
            "channel_overviews": [["signal"]],
            "time_unit": "s",
            **kwargs,
        }
        self.recording = recording
        self.widget = recording.plot_interactive(**kwargs)
        self.figure = self.widget.center.figure
        self.axes = self.figure.axes

    def key(self, key: str) -> None:
        KeyEvent("key_press_event", self.figure.canvas, key=key)._process()

    def click(self, ax, x, y, button=MouseButton.RIGHT) -> None:
        px, py = ax.transData.transform((x, y))
        MouseEvent(
            "button_press_event", self.figure.canvas, px, py, button=button
        )._process()

    def scroll(self, ax, x, y, step: int) -> None:
        px, py = ax.transData.transform((x, y))
        MouseEvent("scroll_event", self.figure.canvas, px, py, step=step)._process()

    def window(self) -> tuple[float, float]:
        return tuple(float(v) for v in self.axes[0].get_xlim())

    def find(self, cls, predicate=lambda w: True) -> list:
        found, stack = [], [self.widget.header]
        while stack:
            widget = stack.pop(0)
            if isinstance(widget, cls) and predicate(widget):
                found.append(widget)
            stack.extend(getattr(widget, "children", ()))
        return found

    @property
    def dropdown(self) -> widgets.Dropdown:
        return self.find(widgets.Dropdown)[0]

    @property
    def checks(self) -> tuple[widgets.Checkbox, widgets.Checkbox, widgets.Checkbox]:
        """Timestamp, numeric, and text checkboxes of the annotation menu."""
        timestamp, numeric, text = self.find(widgets.Checkbox)
        return timestamp, numeric, text

    @property
    def text_input(self) -> widgets.Text:
        return self.find(widgets.Text, lambda w: w.placeholder == "Label text ...")[0]


# -- interactive plot: structure and navigation ------------------------------------


def test_interactive_plot_layout(ipython):
    plot = Interactive(_recording())
    assert isinstance(plot.widget, widgets.AppLayout)
    assert plot.widget.header.titles == ("Annotate", "Align Timelines", "Settings")
    assert len(plot.axes) == 3  # two subplots, one overview
    assert plot.figure._suptitle.get_text() == f"Reference time: {START}"
    # x axis: time relative to the plot start, in the given time unit
    assert plot.window() == (0.0, 599.5)


def test_interactive_plot_requires_ipython(monkeypatch):
    monkeypatch.setattr("IPython.get_ipython", lambda: None)
    with pytest.raises(RuntimeError, match="IPython"):
        _recording().plot_interactive(channels=[["signal"]])


def test_keyboard_navigation(ipython):
    plot = Interactive(_recording())
    plot.key("+")  # keep the middle half
    assert plot.window() == pytest.approx((149.875, 449.625))
    plot.key("right")  # shift by a quarter of the window
    assert plot.window() == pytest.approx((224.8125, 524.5625))
    plot.key("left")
    assert plot.window() == pytest.approx((149.875, 449.625))
    plot.key("-")  # extend by a quarter of the window on both sides
    assert plot.window() == pytest.approx((74.9375, 524.5625))


def test_overview_click_centers_window(ipython):
    plot = Interactive(_recording())
    plot.key("+")
    overview = plot.axes[2]
    plot.click(overview, 100, 0, button=MouseButton.LEFT)
    assert plot.window() == pytest.approx((-49.875, 249.875))


def test_scroll_zooms_y_axis(ipython):
    plot = Interactive(_recording())
    ax = plot.axes[0]
    bottom, top = ax.get_ylim()
    plot.scroll(ax, 300, 0.0, step=1)
    new_bottom, new_top = ax.get_ylim()
    assert new_top - new_bottom == pytest.approx(1.1 * (top - bottom))


def test_ylim_is_kept_when_navigating(ipython):
    plot = Interactive(_recording())
    ax = plot.axes[0]
    ax.set_ylim(-5, 5)
    plot.key("right")
    assert ax.get_ylim() == (-5, 5)


def test_settings_apply_sets_ylim(ipython):
    plot = Interactive(_recording())
    limits = plot.find(widgets.FloatText)
    limits[0].value, limits[1].value = -2.0, 3.0  # min / max of the first subplot
    plot.find(widgets.Button, lambda w: w.description == "Apply")[0].click()
    assert plot.axes[0].get_ylim() == (-2.0, 3.0)


# -- interactive plot: annotation ------------------------------------------------------


def test_label_dropdown_order_and_number_keys(ipython):
    plot = Interactive(_recording())
    # point labels by name, then interval labels by name
    assert [name for name, _ in plot.dropdown.options] == [
        "events",
        "notes",
        "phases   [Interval]",
    ]
    plot.key("3")
    assert plot.dropdown.index == 2
    plot.key("1")
    assert plot.dropdown.index == 0


def test_point_annotation(ipython):
    recording = _recording()
    plot = Interactive(recording)
    events = recording.get_label("events")
    _, numeric, text = plot.checks
    numeric.value, text.value = True, True
    plot.text_input.value = "check"

    plot.click(plot.axes[0], 300, 0.5)  # right click: add
    plot.click(plot.axes[0], 320, 0.5, button=MouseButton.LEFT)  # left click: nothing
    data = events.get_data()
    assert list(data.time_index) == [START + pd.Timedelta(seconds=300)]
    assert data.data.tolist() == pytest.approx([0.5])
    assert data.text_data.tolist() == ["check"]


def test_point_annotation_time_only(ipython):
    recording = _recording()
    plot = Interactive(recording)
    _, numeric, text = plot.checks
    numeric.value, text.value = False, False
    plot.click(plot.axes[0], 300, 0.5)
    events = recording.get_label("events")
    assert len(events) == 1
    assert events.data is None and events.text_data is None


def test_point_deletion_within_tolerance(ipython):
    recording = _recording()
    events = recording.get_label("events")
    for seconds in (100, 300):
        events.add_data(START + pd.Timedelta(seconds=seconds), value=0.0)
    plot = Interactive(recording)
    plot.key("d")  # delete mode
    plot.click(plot.axes[0], 200, 0)  # far from both points
    assert len(events) == 2
    plot.click(plot.axes[0], 300.5, 0)  # within 5 px
    assert list(events.get_data().time_index) == [START + pd.Timedelta(seconds=100)]


def test_interval_annotation(ipython):
    recording = _recording()
    phases = recording.get_label("phases")
    plot = Interactive(recording)
    plot.key("3")  # phases
    _, numeric, _ = plot.checks
    numeric.value = True

    plot.click(plot.axes[0], 360, 0.6)  # end first: intervals are ordered
    plot.click(plot.axes[0], 310, 0.2)
    assert pd.DatetimeIndex(phases.intervals[0]).tolist() == [
        START + pd.Timedelta(seconds=310),
        START + pd.Timedelta(seconds=360),
    ]
    assert phases.data.tolist() == pytest.approx([0.4])  # mean of the y values

    plot.click(plot.axes[0], 400, 0)  # start a second interval ...
    plot.key("escape")  # ... and cancel it
    plot.click(plot.axes[0], 420, 0)
    assert len(phases.intervals) == 1  # a new pending interval, not a second one


def test_interval_deletion_by_click_inside(ipython):
    recording = _recording()
    phases = recording.get_label("phases")
    phases.add_data(
        (START + pd.Timedelta(seconds=310), START + pd.Timedelta(seconds=360))
    )
    plot = Interactive(recording)
    plot.key("3")
    plot.key("d")
    plot.click(plot.axes[0], 300, 0)  # outside
    assert len(phases.intervals) == 1
    plot.click(plot.axes[0], 330, 0)  # inside
    assert len(phases.intervals) == 0


@pytest.mark.parametrize(
    "label_kwargs, expected_numeric, expected_text, expected_input",
    [
        # empty labels: presets from annotation_preset_type ...
        (dict(annotation_preset_type="numerical"), True, False, ""),
        (dict(annotation_preset_type="textual"), False, True, ""),
        # ... or from plot_type / vline_text_source
        (dict(plot_type="scatter", vline_text_source="disabled"), True, False, ""),
        (dict(plot_type="vline", vline_text_source="text_data"), False, True, "probe"),
        # labels with data: from the kind of data, most frequent text
        (dict(time_index=[0, 1, 2], data=[1.0, 2.0, 3.0]), True, False, ""),
        (dict(time_index=[0, 1, 2], text_data=["a", "b", "b"]), False, True, "b"),
    ],
)
def test_annotation_menu_presets(
    ipython, label_kwargs, expected_numeric, expected_text, expected_input
):
    recording = _recording()
    label_kwargs = {"time_index": [], "time_start": START, **label_kwargs}
    recording.add_global_label(Label("probe", **label_kwargs))
    plot = Interactive(recording, labels=[["probe"], []])
    _, numeric, text = plot.checks
    assert (numeric.value, text.value) == (expected_numeric, expected_text)
    assert plot.text_input.value == expected_input


def test_adjust_mode_shifts_selected_channels(ipython):
    recording = _recording()
    signal = recording.get_channel("signal")
    marker = Label("marker", time_index=[START + pd.Timedelta(seconds=50)])
    signal.attach_label(marker)
    plot = Interactive(recording)
    plot.widget.header.selected_index = (
        1  # "Align Timelines"; first channel preselected
    )
    plot.click(plot.axes[0], 300, 0)  # reference time
    plot.click(plot.axes[0], 310, 0)  # target time
    assert signal.first_entry == START + pd.Timedelta(seconds=10)
    assert marker.first_entry == START + pd.Timedelta(
        seconds=60
    )  # attached labels move along
    assert recording.get_channel("other").first_entry == START


# -- static plots ---------------------------------------------------------------------


@pytest.fixture
def axes():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def _texts(ax) -> list[str]:
    return [
        t.get_text()
        for t in ax.findobj(Text)
        if getattr(t, "_from_vitals_label", False)
    ]


def test_channel_plot(axes):
    channel = Channel(
        "signal",
        START + pd.to_timedelta([0, 1, 2], "s"),
        [1.0, 2.0, 3.0],
        plotstyle={"color": "red", "ls": "--"},
    )
    channel.plot(plot_axes=axes, time_unit="s")
    (line,) = axes.get_lines()
    assert line.get_label() == "signal"
    assert line.get_color() == "red" and line.get_linestyle() == "--"
    assert line.get_xdata().tolist() == [0, 1, 2]  # relative to the channel start


def test_label_plot_types(axes):
    times = [START + pd.Timedelta(seconds=s) for s in (1, 2, 3)]
    label = Label(
        "events", time_index=times, data=[1.0, np.nan, 3.0], text_data=["a", "b", "c"]
    )

    label.plot(plot_axes=axes, plot_type="combined", reference_time=START)
    lines = axes.get_lines()
    # entries with values as scatter, the others as vertical lines with text
    assert lines[0].get_xdata().tolist() == [1, 3]
    assert [line.get_xdata()[0] for line in lines[1:]] == [2]
    assert _texts(axes) == ["b"]

    axes.clear()
    label.plot(plot_axes=axes, plot_type="vline", reference_time=START)
    assert len(axes.get_lines()) == 3
    assert _texts(axes) == ["a", "b", "c"]

    axes.clear()
    label.plot(
        plot_axes=axes,
        plot_type="vline",
        vline_text_source="disabled",
        reference_time=START,
    )
    assert _texts(axes) == []


def test_interval_label_plot_types(axes):
    intervals = [
        (START + pd.Timedelta(seconds=1), START + pd.Timedelta(seconds=2)),
        (START + pd.Timedelta(seconds=4), START + pd.Timedelta(seconds=6)),
    ]
    label = IntervalLabel("phases", time_index=intervals, data=[np.nan, 5.0])

    label.plot(plot_axes=axes, plot_type="combined", reference_time=START)
    # without value: box; with value: horizontal bar
    boxes = [p for p in axes.patches if isinstance(p, Rectangle)] + [
        c for c in axes.collections if isinstance(c, PolyCollection)
    ]
    assert len(boxes) == 1
    (bar,) = [c for c in axes.containers if isinstance(c, ErrorbarContainer)]
    assert bar.lines[0].get_xdata().tolist() == [5.0]  # midpoint
    assert bar.lines[0].get_ydata().tolist() == [5.0]


# -- data layer -------------------------------------------------------------------------


def test_minmax_decimate_keeps_extremes_per_bucket():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 100, 10_000))
    y = rng.normal(size=len(x))
    y[1234] = 50.0  # a single peak
    y[5000:5100] = np.nan
    xd, yd = minmax_decimate(x, y, 0, 100, 200)
    assert len(xd) <= 400
    assert np.all(np.diff(xd) >= 0)  # order kept
    assert 50.0 in yd
    buckets = np.minimum((x // 0.5).astype(int), 199)
    buckets_d = np.minimum((xd // 0.5).astype(int), 199)
    for bucket in range(200):
        values = y[buckets == bucket]
        if np.isfinite(values).any():
            assert np.nanmin(values) == np.nanmin(yd[buckets_d == bucket])
            assert np.nanmax(values) == np.nanmax(yd[buckets_d == bucket])


def test_minmax_decimate_gaps_and_outside_samples():
    x = np.arange(-5, 105, 0.01)
    y = np.sin(x)
    y[(x > 40) & (x < 50)] = np.nan
    xd, yd = minmax_decimate(x, y, 0, 100, 50)
    assert np.isnan(yd[(xd > 41) & (xd < 49)]).all()  # gap stays a gap
    assert xd[0] == x[0] and xd[-1] == x[-1]  # samples outside are kept


def test_minmax_decimate_short_input_unchanged():
    x, y = np.arange(10.0), np.arange(10.0)
    xd, yd = minmax_decimate(x, y, 0, 10, 100)
    assert xd is x or np.array_equal(xd, x)
    assert np.array_equal(yd, y)


# -- regression tests --------------------------------------------------------------------


def test_static_plot_aligns_series_with_different_starts():
    """Static plots use a common reference time (#241)."""
    recording = Vitals()
    t = np.arange(0, 120, 1.0)
    recording.add_channel(Channel("first", START + pd.to_timedelta(t, "s"), t))
    recording.add_channel(Channel("later", START + pd.to_timedelta(t + 30, "s"), t))
    # a label that starts 15 minutes before the channels
    label = Label(
        "events",
        time_index=[START - pd.Timedelta(minutes=15), START + pd.Timedelta(seconds=60)],
        data=[1.0, 2.0],
    )
    recording.add_global_label(label)
    fig, axes = recording.plot(channels=[["first", "later"]], labels=[["events"]])
    first, later, events = axes[0].get_lines()[:3]
    assert first.get_xdata()[0] == 0
    assert later.get_xdata()[0] == 30
    assert 60 in events.get_xdata()
    plt.close(fig)


def test_relative_recording_not_starting_at_zero(ipython):
    recording = Vitals()
    t = np.arange(100, 200, 0.5)
    recording.add_channel(Channel("relative", t, np.sin(t)))
    fig, axes = recording.plot()
    assert axes[0].get_lines()[0].get_xdata()[0] == 0  # within the axis limits
    plot = recording.plot_interactive(channels=[["relative"]])
    ax = plot.center.figure.axes[0]
    (line,) = [line for line in ax.get_lines() if line.get_label() == "relative"]
    xmin, xmax = ax.get_xlim()
    assert xmin <= line.get_xdata()[0] < xmax
