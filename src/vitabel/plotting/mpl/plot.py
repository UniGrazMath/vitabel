"""The interactive matplotlib plot: an ipywidgets layout with the control
panel above the figure, driven by the interaction core."""

from __future__ import annotations

import ipywidgets as widgets

from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

from vitabel.plotting.interaction import InteractionCore
from vitabel.plotting.layout import PlotSpec
from vitabel.plotting.mpl.panel import ControlPanel
from vitabel.plotting.mpl.renderer import Renderer
from vitabel.timeseries import IntervalLabel

SELECTION_TOLERANCE_PX = 5
"""Clicks within this distance select a point (when deleting)."""


class InteractivePlot(widgets.AppLayout):
    """Interactive plot of a :class:`.PlotSpec` (matplotlib, ipympl).

    The control panel (annotation, time shifts, settings) is the header, the
    figure canvas the center of the layout.
    """

    def __init__(self, spec: PlotSpec, subplots_kwargs: dict | None = None):
        core = InteractionCore(spec)
        panel = ControlPanel(core)
        renderer = Renderer(core, subplots_kwargs)
        super().__init__(
            header=panel.tab, center=renderer.figure.canvas, pane_heights=[1, 4, 0]
        )
        self.core = core
        self.panel = panel
        self.renderer = renderer
        self.figure = renderer.figure

        canvas = self.figure.canvas
        canvas.mpl_connect("key_press_event", self._on_key)
        canvas.mpl_connect("button_press_event", self._on_click)
        canvas.mpl_connect("motion_notify_event", self._on_move)
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.capture_scroll = True
        core.on("state_changed", self._on_state_changed)
        panel.on_settings_opened = self._show_limits
        self._show_limits()
        self.figure.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.97)
        canvas._figure_label = ""

    # -- input -----------------------------------------------------------------------

    def _on_key(self, event: KeyEvent) -> None:
        core, key = self.core, event.key
        if key is None:
            return
        if len(key) == 1 and key.isdigit():
            index = 9 if key == "0" else int(key) - 1
            if index < len(core.labels):
                core.select_label(index)
        elif key == "right":
            core.pan(+1)
        elif key == "left":
            core.pan(-1)
        elif key == "+":
            core.zoom_in()
        elif key == "-":
            core.zoom_out()
        elif key == "d":
            self.panel.toggle_delete()
        elif key == "escape":
            core.cancel()

    def _on_click(self, event: MouseEvent) -> None:
        renderer = self.renderer
        if event.inaxes in renderer.channel_axes:
            if event.button is MouseButton.RIGHT:
                self.core.click(
                    renderer.to_time(event.xdata),
                    event.ydata,
                    subplot=renderer.channel_axes.index(event.inaxes),
                    tolerance=renderer.selection_tolerance(SELECTION_TOLERANCE_PX),
                )
        elif event.inaxes in renderer.overview_axes:
            self.core.center_on(renderer.to_time(event.xdata))

    def _on_move(self, event: MouseEvent) -> None:
        if event.xdata is not None:
            self.renderer.move_indicators(event.xdata)

    def _on_scroll(self, event: MouseEvent) -> None:
        ax = event.inaxes
        if ax is None:
            return
        renderer = self.renderer
        if ax in renderer.channel_axes:
            subplot = renderer.channel_axes.index(ax)
            self.core.ylims.setdefault(subplot, ax.get_ylim())
            self.core.zoom_y(subplot, event.ydata, event.step)
        else:  # overview
            bottom, top = ax.get_ylim()
            scale = 1 + 0.1 * event.step
            ax.set_ylim(
                event.ydata - (event.ydata - bottom) * scale,
                event.ydata + (top - event.ydata) * scale,
            )
            self.figure.canvas.draw_idle()
        self._show_limits()

    def _show_limits(self) -> None:
        """Show the current y limits in the settings."""
        self.panel.show_limits([ax.get_ylim() for ax in self.renderer.channel_axes])

    # -- status ----------------------------------------------------------------------

    def _on_state_changed(self, field: str) -> None:
        if field == "ylims":
            self._show_limits()
        if field in ("mode", "active_label", "pending_interval"):
            self.figure.canvas._figure_label = self._status()

    def _status(self) -> str:
        core = self.core
        if core.pending_interval is not None:
            return (
                "Creating interval label, select end point or press <ESC> to abort ..."
            )
        if isinstance(core.active_label, IntervalLabel):
            if core.mode == "delete":
                return "To delete an interval, right-click inside of it."
            if core.mode == "annotate":
                return (
                    "Right-click to set start time, then right-click again to "
                    "set end time."
                )
        return " "
