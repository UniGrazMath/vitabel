"""Interaction state and commands of interactive plots.

Independent of the plotting backend: no rendering, no widgets. Backends
translate user input (keys, mouse clicks in data coordinates) into
commands, and react to the events of the core.
"""

from __future__ import annotations

import logging

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from vitabel.plotting.layout import PlotSpec
from vitabel.timeseries import Channel, IntervalLabel, Label
from vitabel.typing import Timedelta, Timestamp

logger = logging.getLogger("vitabel")

Mode = Literal["annotate", "delete", "shift", "none"]
"""What a click does: add to / delete from the active label, shift channels
in time (reference, then target), or nothing."""

EVENTS = ("view_changed", "labels_changed", "channels_shifted", "state_changed")


@dataclass(frozen=True)
class AnnotationOptions:
    """What is stored when annotating (besides the time)."""

    use_value: bool = True
    """Store the y position as value."""
    use_text: bool = True
    """Store :attr:`text` as text (if not empty)."""
    text: str = ""


@dataclass(frozen=True)
class LabelChange:
    """An entry added to or removed from a label (payload of
    ``labels_changed``)."""

    kind: Literal["added", "removed"]
    time: Timestamp | Timedelta | tuple[Timestamp | Timedelta, Timestamp | Timedelta]
    """Time of a point entry, or (start, stop) of an interval."""
    value: float | None = None
    text: str | None = None


@dataclass(frozen=True)
class PendingInterval:
    """First click of an interval annotation."""

    time: Timestamp | Timedelta
    y: float | None
    subplot: int


@dataclass(frozen=True)
class ShiftReference:
    """First click of a time shift."""

    time: Timestamp | Timedelta
    subplot: int


def annotation_defaults(label: Label) -> AnnotationOptions:
    """Default annotation options for a label.

    - Empty labels: from ``annotation_preset_type``; otherwise from
      ``vline_text_source`` (text: the label name as text) and ``plot_type``.
    - Labels with entries: from the kind of data they have (the most
      frequent text is preset).
    """
    if len(label) == 0:
        if label.annotation_preset_type is not None:
            return AnnotationOptions(
                use_value=label.annotation_preset_type in ("numerical", "combined"),
                use_text=label.annotation_preset_type in ("textual", "combined"),
            )
        use_value = use_text = False
        text = ""
        if label.vline_text_source in ("text_data", "combined"):
            use_text, text = True, label.name
        if label.vline_text_source in ("data", "combined"):
            use_value = True
        if label.plot_type in ("scatter", "hline", "combined"):
            use_value = True
        if label.plot_type in ("vline", "combined"):
            use_text = True
        return AnnotationOptions(use_value=use_value, use_text=use_text, text=text)

    if label.data is None and label.text_data is None:
        return AnnotationOptions(use_value=False, use_text=False)
    text = ""
    if label.text_data is not None:
        strings = label.text_data[~pd.isna(label.text_data)]
        if strings.size > 0:
            values, counts = np.unique(strings, return_counts=True)
            text = values[np.argmax(counts)]
    return AnnotationOptions(
        use_value=label.data is not None,
        use_text=label.text_data is not None,
        text=text,
    )


class InteractionCore:
    """State and commands of an interactive plot.

    Parameters
    ----------
    spec
        What is plotted.
    """

    def __init__(self, spec: PlotSpec):
        self.spec = spec
        self.start = spec.start
        self.stop = spec.stop
        self.shift_span = (self.stop - self.start) * 0.25
        """Step of :meth:`pan`: a quarter of the window at the last zoom."""
        self.labels: list[Label] = spec.annotation_labels
        """Labels that can be annotated, in menu order."""
        self.active_label: Label | None = self.labels[0] if self.labels else None
        self.mode: Mode = "annotate"
        self.options = (
            annotation_defaults(self.active_label)
            if self.active_label is not None
            else AnnotationOptions()
        )
        self.pending_interval: PendingInterval | None = None
        self.shift_reference: ShiftReference | None = None
        self.shift_channels: list[Channel] = list(spec.collection.channels[:1])
        """Channels moved by a time shift."""
        self.ylims: dict[int, tuple[float, float]] = {}
        """y limits per subplot, once set (otherwise: automatic)."""
        self._listeners: dict[str, list[Callable]] = defaultdict(list)
        self._undo: list[tuple[str, tuple]] = []
        self._redo: list[tuple[str, tuple]] = []

    # -- events --------------------------------------------------------------------

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback.

        - ``view_changed(start, stop)``
        - ``labels_changed(label, change)`` with a :class:`LabelChange`
        - ``channels_shifted(channels, offset)``
        - ``state_changed(field)``: ``"mode"``, ``"active_label"``,
          ``"options"``, ``"pending_interval"``, ``"shift_reference"``,
          ``"shift_channels"``, ``"ylims"``
        """
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}, expected one of {EVENTS}")
        self._listeners[event].append(callback)

    def _emit(self, event: str, *args) -> None:
        for callback in list(self._listeners[event]):
            callback(*args)

    # -- navigation ----------------------------------------------------------------

    def _set_view(self, start, stop) -> None:
        self.start, self.stop = start, stop
        self._emit("view_changed", start, stop)

    def pan(self, direction: int = 1) -> None:
        """Move the window by a quarter of the window at the last zoom
        (``direction``: +1 later, -1 earlier)."""
        self._set_view(
            self.start + direction * self.shift_span,
            self.stop + direction * self.shift_span,
        )

    def zoom_in(self) -> None:
        """Keep the middle half of the window."""
        span = self.stop - self.start
        start, stop = self.start + span * 0.25, self.stop - span * 0.25
        self.shift_span = (stop - start) * 0.25
        self._set_view(start, stop)

    def zoom_out(self) -> None:
        """Widen the window by a quarter on each side."""
        span = self.stop - self.start
        start, stop = self.start - span * 0.25, self.stop + span * 0.25
        self.shift_span = (stop - start) * 0.25
        self._set_view(start, stop)

    def set_window(self, start, stop) -> None:
        """Show the given window (e.g. after zooming with a tool)."""
        if stop < start:
            start, stop = stop, start
        self.shift_span = (stop - start) * 0.25
        self._set_view(start, stop)

    def center_on(self, time) -> None:
        """Keep the window length, center it at ``time``."""
        span = self.stop - self.start
        self._set_view(time - 0.5 * span, time + 0.5 * span)

    def set_ylim(self, subplot: int, low: float, high: float) -> None:
        self.ylims[subplot] = (low, high)
        self._emit("state_changed", "ylims")

    def zoom_y(self, subplot: int, center: float, step: float) -> None:
        """Scale the y range around ``center`` by 10 % per ``step``."""
        low, high = self.ylims[subplot]
        scale = 1 + 0.1 * step
        self.set_ylim(
            subplot, center - (center - low) * scale, center + (high - center) * scale
        )

    # -- selection and options -----------------------------------------------------

    def cancel(self) -> None:
        """Discard a pending interval and a shift reference."""
        changed = []
        if self.pending_interval is not None:
            self.pending_interval = None
            changed.append("pending_interval")
        if self.shift_reference is not None:
            self.shift_reference = None
            changed.append("shift_reference")
        for field_name in changed:
            self._emit("state_changed", field_name)

    def select_label(self, label: Label | int) -> None:
        """Make a label (or the label at a menu index) active; its annotation
        defaults become the options."""
        if isinstance(label, int):
            label = self.labels[label]
        self.cancel()
        self.active_label = label
        self.options = annotation_defaults(label)
        self._emit("state_changed", "active_label")
        self._emit("state_changed", "options")

    def set_mode(self, mode: Mode) -> None:
        self.cancel()
        self.mode = mode
        self._emit("state_changed", "mode")

    def toggle_delete(self) -> None:
        self.set_mode("annotate" if self.mode == "delete" else "delete")

    def set_options(self, **changes) -> None:
        """Change annotation options (``use_value``, ``use_text``, ``text``)."""
        self.options = replace(self.options, **changes)
        self._emit("state_changed", "options")

    def set_shift_channels(self, channels: list[Channel]) -> None:
        self.shift_channels = list(channels)
        self._emit("state_changed", "shift_channels")

    # -- clicks --------------------------------------------------------------------

    def click(self, time, y: float | None, subplot: int, tolerance=None) -> None:
        """A click at ``time`` / ``y`` in a subplot, depending on the mode:
        annotate the active label, delete from it (points within
        ``tolerance``, the interval containing ``time``), or set the
        reference / target of a time shift."""
        if self.mode == "shift":
            self._shift_click(time, subplot)
        elif self.mode in ("annotate", "delete") and self.active_label is not None:
            label = self.active_label
            if isinstance(label, IntervalLabel):
                if self.mode == "delete":
                    self._delete_interval(label, time)
                else:
                    self._interval_click(label, time, y, subplot)
            elif self.mode == "delete":
                self._delete_point(label, time, tolerance)
            else:
                value = y if self.options.use_value else None
                self._apply(("add", (label, time, value, self._text())))

    def _text(self) -> str | None:
        return (
            self.options.text if self.options.use_text and self.options.text else None
        )

    def _interval_click(self, label, time, y, subplot) -> None:
        if self.pending_interval is None:
            self.pending_interval = PendingInterval(time, y, subplot)
            self._emit("state_changed", "pending_interval")
            return
        first = self.pending_interval
        self.pending_interval = None
        start, stop = sorted([first.time, time])
        value = None
        if self.options.use_value and first.y is not None and y is not None:
            value = (first.y + y) / 2
        self._apply(("add", (label, (start, stop), value, self._text())))
        self._emit("state_changed", "pending_interval")

    def _delete_point(self, label, time, tolerance) -> None:
        if tolerance is None:
            tolerance = pd.Timedelta(0)
        entries = label.get_data(start=time - tolerance, stop=time + tolerance)
        if len(entries.time_index) == 0:
            return
        index = int(np.argmin(entries.time_index))
        self._apply(
            (
                "remove",
                (
                    label,
                    entries.time_index[index],
                    _entry(entries.data, index),
                    _entry(entries.text_data, index),
                ),
            )
        )

    def _delete_interval(self, label, time) -> None:
        intervals = label.intervals
        if len(intervals) == 0:
            return
        selected = np.flatnonzero((intervals[:, 0] <= time) & (intervals[:, 1] >= time))
        if len(selected) == 0:
            return
        index = int(selected.min())
        interval = (
            pd.Timestamp(intervals[index, 0]),
            pd.Timestamp(intervals[index, 1]),
        )
        if not label.is_time_absolute():
            interval = (
                pd.Timedelta(intervals[index, 0]),
                pd.Timedelta(intervals[index, 1]),
            )
        self._apply(
            (
                "remove",
                (
                    label,
                    interval,
                    _entry(label.data, index),
                    _entry(label.text_data, index),
                ),
            )
        )

    def _shift_click(self, time, subplot) -> None:
        if self.shift_reference is None:
            self.shift_reference = ShiftReference(time, subplot)
            self._emit("state_changed", "shift_reference")
            return
        offset = time - self.shift_reference.time
        self.shift_reference = None
        self._apply(("shift", (tuple(self.shift_channels), offset)))
        self._emit("state_changed", "shift_reference")

    # -- data changes and history ---------------------------------------------------

    def _apply(self, action: tuple[str, tuple], record: bool = True) -> None:
        kind, args = action
        if kind == "add":
            label, time, value, text = args
            label.add_data(time, value=value, text=text)
            self._emit("labels_changed", label, LabelChange("added", time, value, text))
        elif kind == "remove":
            label, time, value, text = args
            label.remove_data(time)
            self._emit(
                "labels_changed", label, LabelChange("removed", time, value, text)
            )
        elif kind == "shift":
            channels, offset = args
            for channel in channels:
                channel.shift_time_index(delta_t=offset)
            self._emit("channels_shifted", list(channels), offset)
        if record:
            self._undo.append(action)
            self._redo.clear()

    @staticmethod
    def _inverse(action: tuple[str, tuple]) -> tuple[str, tuple]:
        kind, args = action
        if kind == "add":
            return ("remove", args)
        if kind == "remove":
            return ("add", args)
        channels, offset = args
        return ("shift", (channels, -offset))

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        """Revert the last data change (label entry or time shift)."""
        if not self._undo:
            return
        action = self._undo.pop()
        self._apply(self._inverse(action), record=False)
        self._redo.append(action)

    def redo(self) -> None:
        if not self._redo:
            return
        action = self._redo.pop()
        self._apply(action, record=False)
        self._undo.append(action)


def _entry(values, index: int):
    """Value / text of an entry (``None`` if missing)."""
    if values is None:
        return None
    value = values[index]
    return None if pd.isna(value) else value
