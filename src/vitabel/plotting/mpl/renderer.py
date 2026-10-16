"""Drawing of the interactive matplotlib plot, driven by the events of the
interaction core."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.text import Text

from vitabel.plotting.interaction import InteractionCore
from vitabel.plotting.layout import PlotSpec


class Renderer:
    """Figure, subplots and their content.

    Parameters
    ----------
    core
        The interaction core (state, events).
    subplots_kwargs
        Keyword arguments for ``matplotlib.pyplot.subplots``.
    """

    def __init__(self, core: InteractionCore, subplots_kwargs: dict | None = None):
        self.core = core
        self.spec: PlotSpec = core.spec
        self.unit = pd.to_timedelta(1, unit=self.spec.time_unit)
        n_channel_axes = len(self.spec.channels)
        with plt.ioff():
            figure, axes = plt.subplots(
                n_channel_axes + len(self.spec.overviews),
                squeeze=False,
                **(subplots_kwargs or {}),
            )
            figure.canvas.toolbar.toolitems = [
                tool
                for tool in figure.canvas.toolbar.toolitems
                if tool[0] in ["Home", "Zoom", "Download"]
            ]
            figure.canvas.header_visible = True
            if self.spec.collection.is_time_absolute():
                figure.suptitle(f"Reference time: {self.spec.reference_time}")
        self.figure = figure
        axes = axes[:, 0]
        self.channel_axes = list(axes[:n_channel_axes])
        self.overview_axes = list(axes[n_channel_axes:])
        self.screen_pixel_width, _ = figure.canvas.get_width_height()

        self.x_indicators = [
            ax.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
            for ax in self.channel_axes
        ]
        self.overview_indicators = []
        self.pending_interval_artist = None
        self.is_repainting = False

        core.on("view_changed", lambda start, stop: self.repaint())
        core.on("labels_changed", lambda label, change: self.repaint())
        core.on("channels_shifted", self._on_channels_shifted)
        core.on("state_changed", self._on_state_changed)

        self.repaint_overview()
        self.repaint()
        for ax in self.channel_axes:
            ax.callbacks.connect("xlim_changed", self._on_xlim_changed)

    # -- coordinates -------------------------------------------------------------

    def to_x(self, time) -> float:
        return (time - self.spec.reference_time) / self.unit

    def to_time(self, x: float):
        return self.spec.reference_time + pd.to_timedelta(x, unit=self.spec.time_unit)

    def selection_tolerance(self, pixels: int = 5):
        """Time span of a few pixels at the current zoom."""
        return (self.core.stop - self.core.start) / self.screen_pixel_width * pixels

    # -- event handlers ------------------------------------------------------------

    def _on_xlim_changed(self, ax) -> None:
        if self.is_repainting:
            return
        xmin, xmax = ax.get_xlim()
        self.core.set_window(self.to_time(xmin), self.to_time(xmax))

    def _on_channels_shifted(self, channels, offset) -> None:
        self.repaint_overview()
        self.repaint()

    def _on_state_changed(self, field: str) -> None:
        if field == "pending_interval":
            pending = self.core.pending_interval
            if pending is not None:
                label = self.core.active_label
                self.pending_interval_artist = self.channel_axes[
                    pending.subplot
                ].axvline(
                    x=self.to_x(pending.time),
                    color=label.plotstyle.get("color", "limegreen"),
                    linestyle="--",
                    linewidth=1.5,
                )
            else:
                self.pending_interval_artist = None
                self.repaint()
        elif field == "shift_reference":
            reference = self.core.shift_reference
            if reference is not None:
                self.channel_axes[reference.subplot].axvline(
                    x=self.to_x(reference.time),
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                )
            else:
                self.repaint()
        elif field == "ylims":
            for subplot, (low, high) in self.core.ylims.items():
                self.channel_axes[subplot].set_ylim(low, high)
            self.update_vline_texts()
            self.figure.canvas.draw_idle()

    def move_indicators(self, x: float) -> None:
        for indicator in self.x_indicators:
            indicator.set_xdata(np.ones_like(indicator.get_xdata()) * x)
        self.figure.canvas.draw_idle()

    def update_vline_texts(self) -> None:
        """Keep vline texts at 10 % of the y range."""
        for ax in self.channel_axes:
            ymin, ymax = ax.get_ylim()
            texts = ax.findobj(
                lambda artist: (
                    isinstance(artist, Text) and hasattr(artist, "_from_vitals_label")
                )
            )
            for artist in texts:
                artist.set_y(ymin + 0.1 * (ymax - ymin))

    # -- drawing -----------------------------------------------------------------

    def repaint(self) -> None:
        """Redraw the subplots for the current window."""
        start, stop = self.core.start, self.core.stop
        time_unit = self.spec.time_unit
        reference_time = self.spec.reference_time
        data_width = (stop - start).total_seconds()
        self.screen_pixel_width, _ = self.figure.canvas.get_width_height()
        resolution = pd.Timedelta(seconds=data_width / self.screen_pixel_width)

        pending_ax = None
        if self.pending_interval_artist is not None:
            pending_ax = self.pending_interval_artist.axes

        self.is_repainting = True
        try:
            with plt.ioff():
                for channel_list, label_list, indicator, subax in zip(
                    self.spec.channels,
                    self.spec.labels,
                    self.x_indicators,
                    self.channel_axes,
                ):
                    old_ylims = subax.get_ylim()
                    old_ylabel = subax.yaxis.get_label_text()
                    subax.clear()
                    subax.add_artist(indicator)
                    subax.format_coord = self._format_coords
                    subax.set_xlim((self.to_x(start), self.to_x(stop)))
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
                for indicator in self.overview_indicators:
                    indicator.remove()
                self.overview_indicators = [
                    ax.axvspan(
                        xmin=self.to_x(start),
                        xmax=self.to_x(stop),
                        color="red",
                        alpha=0.25,
                    )
                    for ax in self.overview_axes
                ]
                reference = self.core.shift_reference
                if reference is not None:
                    self.channel_axes[reference.subplot].axvline(
                        x=self.to_x(reference.time),
                        color="red",
                        linestyle="--",
                        linewidth=1.5,
                    )
                if self.pending_interval_artist is not None:
                    pending_ax.add_artist(self.pending_interval_artist)
        finally:
            self.is_repainting = False

    def repaint_overview(self) -> None:
        """Redraw the overview subplots (full time range)."""
        for index, (channel_list, subax) in enumerate(
            zip(self.spec.overviews, self.overview_axes)
        ):
            if not channel_list:
                continue
            ov_start, ov_stop = self.spec.overview_range(index)
            data_width = (ov_stop - ov_start).total_seconds()
            resolution = pd.Timedelta(seconds=data_width / self.screen_pixel_width)
            subax.clear()
            self.overview_indicators = []
            for channel in channel_list:
                channel.plot(
                    plot_axes=subax,
                    start=ov_start,
                    stop=ov_stop,
                    resolution=resolution,
                    time_unit=self.spec.time_unit,
                    reference_time=self.spec.reference_time,
                )
            subax.set_xlim((self.to_x(ov_start), self.to_x(ov_stop)))
            subax.grid(False)

    def _format_coords(self, x, y) -> str:
        format_string = f"(x, y) = ({x:.2f}, {y:.2f})"
        if self.spec.collection.is_time_absolute():
            format_string += f"   x = {self.to_time(x)}"
        return format_string
