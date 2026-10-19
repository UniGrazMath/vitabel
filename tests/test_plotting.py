"""Behaviour of the plotting routines that the plotting refactoring (Phase 3,
M0) must preserve.

The interactive plot is driven headless: an IPython shell without GUI
integration, the ipympl canvas, and synthetic matplotlib events.
Known bugs are deliberately not pinned here.
"""

from __future__ import annotations

import logging

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
from vitabel.plotting.interaction import AnnotationOptions, InteractionCore, LabelChange
from vitabel.plotting.layout import PlotSpec
from vitabel.plotting.overlays import FillBetween, HLine, VSpan

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


@pytest.fixture(autouse=True)
def _no_logged_errors(caplog):
    """Exceptions in event listeners (e.g. the renderer) are logged, not
    raised: fail on them, unless a test expects (and clears) them."""
    yield
    errors = [
        record
        for record in caplog.get_records("call")
        if record.name == "vitabel" and record.levelno >= logging.ERROR
    ]
    assert not errors, [record.getMessage() for record in errors]


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


def test_static_plot_keeps_peaks():
    """Downsampling keeps minimum and maximum per pixel column."""
    recording = Vitals()
    t = np.arange(0, 3600, 0.01)  # 360k samples
    values = np.sin(t)
    values[123_457] = 25.0  # a single-sample peak
    recording.add_channel(Channel("signal", START + pd.to_timedelta(t, "s"), values))
    fig, axes = recording.plot()
    (line,) = axes[0].get_lines()
    assert len(line.get_xdata()) < 10_000  # downsampled
    assert line.get_ydata().max() == 25.0
    plt.close(fig)


def test_static_plot_resolution_for_millisecond_channels():
    """The resolution (computed from the figure width) is a duration, not a
    number of seconds read in the channel's time unit."""
    recording = Vitals()
    t = np.arange(0, 600_000, 2.0)  # 10 minutes in ms, 300k samples
    recording.add_channel(Channel("signal", t, np.sin(t / 1000), time_unit="ms"))
    fig, axes = recording.plot()
    (line,) = axes[0].get_lines()
    assert len(line.get_xdata()) < 10_000
    plt.close(fig)


def test_interval_bars_use_the_plotstyle_argument(axes):
    label = IntervalLabel(
        "phases",
        time_index=[(START, START + pd.Timedelta(seconds=2))],
        data=[1.0],
        plotstyle={"color": "blue"},
    )
    label.plot(plot_axes=axes, plot_type="hline", plotstyle={"color": "red"})
    (bar,) = axes.containers
    assert matplotlib.colors.to_hex(bar.lines[0].get_color()) == "#ff0000"


# -- plot specification --------------------------------------------------------------------


def test_plot_spec_resolves_arguments():
    recording = _recording()
    spec = PlotSpec.from_arguments(
        recording.data,
        channels=[["signal"], [1]],
        labels=[["phases", "events"], []],
        channel_overviews=True,
    )
    assert [[c.name for c in channels] for channels in spec.channels] == [
        ["signal"],
        ["other"],
    ]
    # overview of all plotted channels, in order of appearance
    assert [c.name for c in spec.overviews[0]] == ["signal", "other"]
    assert spec.start == START
    assert spec.stop == START + pd.Timedelta(seconds=599.5)
    assert spec.reference_time == spec.start
    assert spec.time_unit == "s"
    assert [label.name for label in spec.annotation_labels] == ["events", "phases"]


def test_plots_of_collections_without_channels(ipython):
    recording = Vitals()
    recording.add_global_label(
        Label(
            "events",
            time_index=[START, START + pd.Timedelta(seconds=30)],
            data=[1.0, 2.0],
        )
    )
    fig, axes = recording.plot()
    assert axes[0].get_lines()
    plot = recording.plot_interactive()
    assert isinstance(plot, widgets.AppLayout)


# -- interaction core ------------------------------------------------------------------------


def _core(recording=None, **kwargs) -> InteractionCore:
    recording = recording or _recording()
    kwargs = {
        "channels": [["signal"], ["other"]],
        "labels": [["events", "notes", "phases"], []],
        **kwargs,
    }
    return InteractionCore(PlotSpec.from_arguments(recording.data, **kwargs))


def _seconds(value: float) -> pd.Timestamp:
    return START + pd.Timedelta(seconds=value)


class Recorder:
    def __init__(self, core):
        self.events = []
        for event in (
            "view_changed",
            "labels_changed",
            "channels_shifted",
            "state_changed",
        ):
            core.on(
                event, lambda *args, event=event: self.events.append((event, *args))
            )

    def of(self, event):
        return [args for name, *args in self.events if name == event]


def test_core_navigation():
    core = _core()
    events = Recorder(core)
    core.zoom_in()
    assert (core.start, core.stop) == (_seconds(149.875), _seconds(449.625))
    core.pan(+1)
    assert core.start == _seconds(224.8125)
    core.pan(-1)
    core.zoom_out()
    assert (core.start, core.stop) == (_seconds(74.9375), _seconds(524.5625))
    core.center_on(_seconds(100))
    assert (core.start, core.stop) == (_seconds(-124.8125), _seconds(324.8125))
    core.set_window(_seconds(20), _seconds(10))  # ordered
    assert (core.start, core.stop) == (_seconds(10), _seconds(20))
    assert core.shift_span == pd.Timedelta(seconds=2.5)
    assert events.of("view_changed")[-1] == [_seconds(10), _seconds(20)]
    assert len(events.of("view_changed")) == 6


def test_core_y_limits():
    core = _core()
    core.set_ylim(0, -1.0, 1.0)
    core.zoom_y(0, center=0.0, step=1)
    assert core.ylims[0] == pytest.approx((-1.1, 1.1))


def test_core_label_selection_and_options():
    core = _core()
    assert [label.name for label in core.labels] == ["events", "notes", "phases"]
    core.select_label(1)  # "notes": vline label, text preset with its name
    assert core.active_label.name == "notes"
    assert core.options == AnnotationOptions(
        use_value=False, use_text=True, text="notes"
    )
    core.set_options(text="other")
    assert core.options.text == "other"


def test_core_point_annotation_and_deletion():
    recording = _recording()
    events = recording.get_label("events")
    core = _core(recording)
    core.set_options(use_value=True, use_text=True, text="check")
    recorder = Recorder(core)

    core.click(_seconds(300), 0.5, subplot=0)
    assert events.get_data().data.tolist() == [0.5]
    label, change = recorder.of("labels_changed")[-1]
    assert label is events
    assert change == LabelChange("added", _seconds(300), 0.5, "check")

    core.toggle_delete()
    core.click(_seconds(310), 0, subplot=0, tolerance=pd.Timedelta(seconds=5))
    assert len(events) == 1  # not within the tolerance
    core.click(_seconds(303), 0, subplot=0, tolerance=pd.Timedelta(seconds=5))
    assert len(events) == 0
    assert recorder.of("labels_changed")[-1][1] == LabelChange(
        "removed", _seconds(300), 0.5, "check"
    )


def test_core_interval_annotation():
    recording = _recording()
    phases = recording.get_label("phases")
    core = _core(recording)
    core.select_label(phases)
    core.set_options(use_value=True, use_text=False)

    core.click(_seconds(360), 0.6, subplot=1)
    assert core.pending_interval.subplot == 1
    assert len(phases.intervals) == 0
    core.click(_seconds(310), 0.2, subplot=0)
    assert core.pending_interval is None
    assert pd.DatetimeIndex(phases.intervals[0]).tolist() == [
        _seconds(310),
        _seconds(360),
    ]
    assert phases.data.tolist() == pytest.approx([0.4])

    core.click(_seconds(400), 0, subplot=0)
    core.cancel()
    assert core.pending_interval is None

    core.toggle_delete()
    core.click(_seconds(330), 0, subplot=0)
    assert len(phases.intervals) == 0


def test_core_shift():
    recording = _recording()
    signal = recording.get_channel("signal")
    core = _core(recording)
    recorder = Recorder(core)
    core.set_mode("shift")
    core.click(_seconds(300), 0, subplot=0)
    assert core.shift_reference.time == _seconds(300)
    core.click(_seconds(310), 0, subplot=0)
    assert signal.first_entry == _seconds(10)
    assert recorder.of("channels_shifted") == [[[signal], pd.Timedelta(seconds=10)]]


def test_core_undo_redo():
    recording = _recording()
    events, signal = recording.get_label("events"), recording.get_channel("signal")
    core = _core(recording)
    core.set_options(use_value=True, use_text=False)
    core.click(_seconds(300), 0.5, subplot=0)
    core.toggle_delete()
    core.click(_seconds(300), 0, subplot=0)
    core.set_mode("shift")
    core.click(_seconds(0), 0, subplot=0)
    core.click(_seconds(10), 0, subplot=0)

    core.undo()  # shift
    assert signal.first_entry == START
    core.undo()  # deletion
    assert events.get_data().data.tolist() == [0.5]
    core.undo()  # addition
    assert len(events) == 0
    assert not core.can_undo()
    core.redo()
    core.redo()
    assert len(events) == 0 and core.can_redo()
    core.click(_seconds(5), 0, subplot=0)  # a new change (shift reference) ...
    core.click(_seconds(6), 0, subplot=0)
    assert not core.can_redo()  # ... clears the redo history


def test_core_failing_listener_does_not_break_the_history(caplog):
    recording = _recording()
    events = recording.get_label("events")
    core = _core(recording)
    notified = []

    def failing(label, change):
        raise RuntimeError("listener bug")

    core.on("labels_changed", failing)
    core.on("labels_changed", lambda label, change: notified.append(change.kind))
    core.click(_seconds(300), 0.5, subplot=0)
    assert len(events) == 1
    assert notified == ["added"]  # later listeners are still called
    assert core.can_undo()
    core.undo()
    assert len(events) == 0 and notified == ["added", "removed"]
    assert "listener bug" in caplog.text
    caplog.clear()


def test_core_undo_after_a_change_outside_of_the_plot(caplog):
    """An entry removed in the notebook can't be removed again by undo: the
    step is dropped from the history (instead of blocking it)."""
    recording = _recording()
    events = recording.get_label("events")
    core = _core(recording)
    core.click(_seconds(100), 0.5, subplot=0)
    core.click(_seconds(300), 0.5, subplot=0)
    events.remove_data(_seconds(300))
    with caplog.at_level(logging.WARNING, logger="vitabel"):
        core.undo()
    assert "Could not undo" in caplog.text
    assert len(events) == 1 and not core.can_redo()
    core.undo()  # the earlier step still works
    assert len(events) == 0


def test_core_undo_cancels_a_first_click():
    recording = _recording()
    events, phases = recording.get_label("events"), recording.get_label("phases")
    core = _core(recording)
    core.click(_seconds(100), 0.5, subplot=0)
    core.select_label(phases)
    core.click(_seconds(200), 0.5, subplot=0)  # interval start
    core.undo()
    assert core.pending_interval is None
    assert len(events) == 1  # not undone yet
    core.undo()
    assert len(events) == 0


def test_core_mode_none_ignores_clicks():
    recording = _recording()
    core = _core(recording)
    core.set_mode("none")
    core.click(_seconds(300), 0.5, subplot=0)
    assert len(recording.get_label("events")) == 0


# -- persistent drawing (M0.4) ---------------------------------------------------------------


def test_customisations_survive_navigation_and_annotation(ipython):
    plot = Interactive(_recording())
    ax = plot.axes[0]
    ax.get_legend().remove()
    ax.grid(False)
    ax.set_ylabel("custom")
    marker = ax.axhline(0.25, color="orange")

    plot.key("right")
    plot.key("+")
    plot.click(ax, 300, 0.5)  # annotation: redraws the label
    assert ax.get_legend() is None
    assert not any(line.get_visible() for line in ax.get_xgridlines())
    assert ax.get_ylabel() == "custom"
    assert marker in ax.lines


def test_drawn_artists_do_not_accumulate(ipython):
    plot = Interactive(_recording())
    ax = plot.axes[0]
    counts = len(ax.get_children())
    for key in ["right", "left", "+", "-"] * 3:
        plot.key(key)
    assert len(ax.get_children()) == counts


def test_pending_interval_marker_in_all_subplots_of_the_label(ipython):
    """#255"""
    plot = Interactive(_recording(), labels=[["phases"], ["phases"]])
    plot.key("1")  # phases (the only label)
    before = [len(ax.lines) for ax in plot.axes[:2]]
    plot.click(plot.axes[0], 300, 0)
    assert [len(ax.lines) for ax in plot.axes[:2]] == [n + 1 for n in before]
    plot.key("escape")
    assert [len(ax.lines) for ax in plot.axes[:2]] == before


def test_settings_show_the_current_limits(ipython):
    """Apply without editing keeps the limits (the fields used to start at
    (0, 1))."""
    plot = Interactive(_recording())
    limits = [ax.get_ylim() for ax in plot.axes[:2]]
    plot.find(widgets.Button, lambda w: w.description == "Apply")[0].click()
    assert [ax.get_ylim() for ax in plot.axes[:2]] == limits
    plot.axes[0].set_ylim(-3, 3)  # changed outside of the settings ...
    plot.widget.header.selected_index = 2  # ... shown when opening them
    fields = plot.find(widgets.FloatText)
    assert (fields[0].value, fields[1].value) == (-3, 3)


# -- public interface of the interactive plot ------------------------------------------------


def test_interactive_plot_events_and_window(ipython):
    plot = Interactive(_recording())
    changes, windows = [], []
    plot.widget.on(
        "labels_changed",
        lambda label, change: changes.append((label.name, change.kind)),
    )
    plot.widget.on("view_changed", lambda start, stop: windows.append((start, stop)))
    plot.click(plot.axes[0], 300, 0.5)
    assert changes == [("events", "added")]
    plot.widget.set_window(100, 200)  # in units of the x axis
    assert plot.window() == (100.0, 200.0)
    plot.widget.set_window("2024-01-01 12:01:00", "2024-01-01 12:02:00")
    assert windows[-1] == (_seconds(60), _seconds(120))


def test_fill_between_overlay(ipython):
    """Like notebook 03: the area of a signal below a threshold, recomputed
    for every window."""
    plot = Interactive(_recording())
    ax = plot.axes[0]
    n_collections = len(ax.collections)
    overlay = plot.widget.add_overlay(
        FillBetween("signal", 0.5, where="below", color="orange"), subplot=0
    )
    assert len(ax.collections) == n_collections + 1
    fill = ax.collections[-1]
    assert matplotlib.colors.to_hex(fill.get_facecolor()[0]) == "#ffa500"
    plot.key("+")
    assert len(ax.collections) == n_collections + 1  # replaced, not added
    assert ax.collections[-1] is not fill
    plot.widget.remove_overlay(overlay)
    assert len(ax.collections) == n_collections


def test_threshold_and_span_overlays(ipython):
    plot = Interactive(_recording())
    ax = plot.axes[1]
    n_lines, n_patches = len(ax.lines), len(ax.patches)
    plot.widget.add_overlay(HLine(0.5, color="red"), subplot=1)
    plot.widget.add_overlay(VSpan(_seconds(10), _seconds(20), alpha=0.2), subplot=1)
    plot.key("right")
    assert len(ax.lines) == n_lines + 1
    assert len(ax.patches) + len(ax.collections) > n_patches


def test_undo_redo_keys(ipython):
    recording = _recording()
    events = recording.get_label("events")
    plot = Interactive(recording)
    plot.click(plot.axes[0], 300, 0.5)
    plot.key("ctrl+z")
    assert len(events) == 0
    plot.key("ctrl+y")
    assert len(events) == 1


def test_default_colors_are_kept_when_redrawing(ipython):
    """Series without explicit color keep their color from the color cycle
    (redrawing must not advance the cycle)."""
    plot = Interactive(_recording())
    ax = plot.axes[0]

    def signal_color():
        (line,) = [line for line in ax.get_lines() if line.get_label() == "signal"]
        return matplotlib.colors.to_hex(line.get_color())

    color = signal_color()
    for key in ["right", "+", "-"]:
        plot.key(key)
    plot.click(ax, 300, 0.5)  # redraws the label (and its subplot)
    assert signal_color() == color


def test_crosshair_redraw_without_full_render(ipython):
    """Mouse moves request a redraw from the browser (flow control: no
    pushed images, which the browser may skip, leaving fragments); the
    requested redraw only draws the crosshair on the saved background."""
    plot = Interactive(_recording())
    canvas, ax = plot.figure.canvas, plot.axes[0]
    full_renders = []
    canvas.mpl_connect("draw_event", lambda event: full_renders.append(event))
    canvas.handle_event({"type": "draw"})  # first request: full render
    assert len(full_renders) == 1

    px, py = ax.transData.transform((300, 0))

    def dark_pixels_in_column():
        """Maximum number of dark pixels in the columns around ``px``."""
        pixels = np.asarray(canvas.get_renderer().buffer_rgba())
        columns = pixels[:, round(px) - 1 : round(px) + 2, :3].mean(axis=2)
        return int((columns < 100).sum(axis=0).max())

    before = dark_pixels_in_column()
    MouseEvent("motion_notify_event", canvas, px, py)._process()
    assert not plot.figure.stale
    canvas.handle_event({"type": "draw"})
    assert len(full_renders) == 1  # crosshair only
    assert dark_pixels_in_column() > before + 50  # the dashed line is there

    plot.key("right")  # a real change: full render
    canvas.handle_event({"type": "draw"})
    assert len(full_renders) == 2
