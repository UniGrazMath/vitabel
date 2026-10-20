# Interactive plots

{py:meth}`Vitals.plot_interactive <vitabel.vitals.Vitals.plot_interactive>`
shows the channels and labels of a recording in a figure with a control panel
above it. You can use it to browse a recording, annotate it with labels, and
align the timelines of different devices. It also serves as a building block for
your own review tools in a notebook.

```py
from vitabel import Vitals

case = Vitals()
case.add_defibrillator_recording("path/to/ZOLL_data_file.json")
case.compute_etco2_and_ventilations()

plot = case.plot_interactive(
    channels=[["cpr_acceleration"], ["capnography"], ["ecg_pads"]],
    labels=[[], ["etco2_from_capnography"], []],
    channel_overviews=[["cpr_acceleration"]],
    time_unit="s",
    subplots_kwargs={"figsize": (16, 8)},
)
plot
```

The plot needs a Jupyter environment (JupyterLab, Notebook, or VS Code) and uses
the interactive matplotlib backend from
[ipympl](https://matplotlib.org/ipympl/), which is installed with `vitabel`.
Creating the plot switches matplotlib to this backend (the same as
`%matplotlib widget`), so static plots created afterwards in the same notebook
are interactive figures as well.

## What is plotted

- `channels` and `labels` are lists with one entry per subplot. Each entry lists
  the channels (or labels) for that subplot, given by name, index, or as an
  object; `[]` leaves a subplot without channels or labels.
- `include_attached_labels=True` adds the labels attached to the plotted
  channels to their subplots.
- `channel_overviews` adds overview subplots below the main ones. They show the
  full recording, mark the current time window, and move the window when you
  click on them. `True` puts all plotted channels into one overview.
  `limited_overview=True` limits the overviews to the time range of the plotted
  channels.
- `start` / `stop` set the initial time window; `time_unit` sets the unit of
  the x axis (e.g. `"s"`, `"min"`). The x axis shows the time since the start
  of the initial window; for recordings with absolute time, this reference time
  is shown above the figure.
- `subplots_kwargs` are passed to `matplotlib.pyplot.subplots`, e.g.
  `{"figsize": (12, 8), "gridspec_kw": {"height_ratios": [3, 1]}}`.

The style of a channel or label comes from its `plotstyle` (matplotlib keyword
arguments such as `color`, `marker` or `linestyle`). Labels have a few more
display options:

- `plot_type`: {py:data}`~vitabel.typing.LabelPlotType` for point labels,
  {py:data}`~vitabel.typing.IntervalLabelPlotType` for interval labels;
- `vline_text_source`: {py:data}`~vitabel.typing.LabelPlotVLineTextSource`.

Long signals are downsampled to the width of the figure. For every pixel, the
minimum and maximum are kept, so short peaks stay visible.

## Navigating

Click into the figure first, so that it receives the key presses.

| Input | Effect |
|---|---|
| {kbd}`→` / {kbd}`←` | Move the window by a quarter of its length |
| {kbd}`+` / {kbd}`-` | Zoom in (keep the middle half) / zoom out (widen by a quarter on each side) |
| Zoom tool of the toolbar | Zoom to a rectangle (left mouse button) |
| Click into an overview | Center the window at the clicked time |
| Mouse wheel over a subplot | Zoom the vertical axis around the mouse position |
| *Settings* tab | Set the vertical limits of the subplots |

The vertical limits are set automatically at first. After that they stay fixed
while you navigate, until you scroll or change them in the *Settings* tab.

## Annotating

While the *Annotate* tab is open, a right-click adds entries to the **active
label**, which you select in this tab. The menu lists the labels of the plot (point labels first,
then interval labels); the number keys {kbd}`1`–{kbd}`9` and {kbd}`0` select
the first ten of them.

**Point labels.** Right-click at the time where an entry should be added.
Besides the time, an entry can store:

- a *numeric value*: the vertical position of the click;
- a *textual value*: the text from the text field.

The checkboxes preselect what a label stores. For labels that already have
entries, the preselection follows the data they have (and the text field
contains their most frequent text). For empty labels, it is set by the label's
`annotation_preset_type` ({py:data}`~vitabel.typing.LabelAnnotationPresetType`),
or otherwise derived from how the label is plotted.

**Interval labels.** Right-click at the start and then at the end of the
interval. After the first click, a dashed line marks the start in all subplots
that show the label. {kbd}`Esc` discards it. With *Numeric value*, the interval
stores the mean height of the two clicks.

**Deleting.** {kbd}`d` (or the *Mode* button) switches between adding and
deleting. In delete mode, a right-click removes an entry within a few pixels of the
click, or the interval that contains the click.

**Undo.** {kbd}`Ctrl`+{kbd}`Z` undoes the last change (an added or deleted
entry, or a time shift); {kbd}`Ctrl`+{kbd}`Y` redoes it. If the first click of
an interval or a time shift is pending, undo discards that click first. If you
change a label in code while the plot is open, steps that no longer apply are
skipped with a warning.

The entries are added to the label objects themselves: after annotating, the
label contains them, e.g. for `case.save_data(...)` or further analysis.

## Aligning timelines

Recordings from different devices are often not synchronised. While the *Align
Timelines* tab is open, right-clicks shift channels in time. Select the
channels to move in this tab. All channels of the recording are
listed, also those not plotted, so you can move all channels of a device at
once. Labels attached to these channels move along. Then:

1. right-click at a reference point, e.g. a characteristic peak (marked with a
   red dashed line);
2. right-click where that point should be.

The selected channels are shifted by the difference. {kbd}`Esc` discards the
reference point, and {kbd}`Ctrl`+{kbd}`Z` undoes the shift.

## Building on the plot

`plot_interactive` returns an
{py:class}`~vitabel.plotting.mpl.plot.InteractivePlot`, which is an ipywidgets
`AppLayout`: the control panel is `plot.header`, the figure canvas is
`plot.center`, and `plot.figure` is the matplotlib figure. You can display the
plot as it is, or arrange its parts with other widgets:

```py
import ipywidgets as widgets

results = widgets.HTML()
widgets.VBox([plot.header, widgets.HBox([plot.center, results])])
```

### Events

`plot.on(event, callback)` calls `callback` when something changes:

| Event | Arguments | When |
|---|---|---|
| `"labels_changed"` | `label`, `change` | An entry was added or removed (also by undo / redo). `change` is a {py:class}`~vitabel.plotting.interaction.LabelChange` with `kind` (`"added"` / `"removed"`), `time`, `value`, `text`. |
| `"view_changed"` | `start`, `stop` | The time window changed. |
| `"channels_shifted"` | `channels`, `offset` | Channels were shifted in time. |
| `"state_changed"` | `field` | The state of the controls changed, e.g. `"mode"`, `"active_label"`, `"pending_interval"`. |

Callbacks run after the change is complete. An exception in a callback is
logged (logger `"vitabel"`); it doesn't affect the plot or other callbacks. For
example, to keep a metric up to date while annotating (as in the anaesthesia
example notebook):

```py
def update_metrics(label, change):
    metrics = case.area_under_threshold(source="MAP", threshold=65)
    results.value = f"Area under 65 mmHg: {metrics.area_under_threshold.value:.0f}"

plot.on("labels_changed", update_metrics)
```

`plot.set_window(start, stop)` shows a time window. The times can be timestamps
or timedeltas (also as strings), or numbers in the unit of the x axis.

### Overlays

Overlays are additional elements of a subplot that are recomputed whenever the
plot is redrawn:

- {py:class}`~vitabel.plotting.overlays.FillBetween`: the area between a
  channel or label and a constant (or another channel or label), optionally
  only where it is `"below"` or `"above"`;
- {py:class}`~vitabel.plotting.overlays.HLine`: a horizontal line, e.g. a
  threshold;
- {py:class}`~vitabel.plotting.overlays.VSpan`: a shaded time range.

```py
from vitabel.plotting import FillBetween, HLine

hypotension = plot.add_overlay(
    FillBetween("MAP", 65, where="below", color="#ff7f45", alpha=0.8), subplot=0
)
plot.add_overlay(HLine(65, color="gray", linestyle=":"), subplot=0)
plot.remove_overlay(hypotension)
```

Their style arguments are passed to matplotlib (`fill_between`, `axhline`,
`axvspan`). An overlay computed from a label follows it: if you annotate the
label, the overlay is updated in every subplot that shows it.

### Customising the figure

The plot only replaces what it draws itself: the lines, markers and areas of
channels, labels and overlays. Other changes to the figure are kept when you
navigate or annotate, e.g. axis labels, ticks, grids, titles, and artists you
add yourself:

```py
fig = plot.figure
ax = fig.axes[0]
ax.set_ylabel("Blood pressure (mmHg)")
ax.grid(axis="y")
ax.axhline(65, color="red", linewidth=0.5)
```

Change the appearance of channels and labels with their `plotstyle` instead.
Changes to the artists the plot draws are lost at the next redraw.

Legends are created when the plot is first drawn. A legend you change (or
replace with `ax.legend(...)`) stays as it is while the series visible in the
window stay the same. When they change, e.g. because a label gets its first
entry in the window, the legend is rebuilt; its location and title are kept.
A legend you remove stays removed.

## Limitations

- The figure is rendered in the Python kernel and sent to the browser as an
  image. Every step (pan, zoom, annotation) takes a short moment: typically
  around 50–100 ms, also for a few signals with millions of samples.
- The figure only reacts while the kernel is idle. A long-running cell blocks
  it.
