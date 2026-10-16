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
        self.pending_interval_artists = []
        self.shift_reference_artist = None
        self.is_repainting = False
        self._artists: dict[tuple[int, int], list] = {}
        """Artists drawn per (subplot, series id); only these are ever replaced,
        so user changes to the axes and their own artists persist."""
        for ax in self.channel_axes:
            ax.format_coord = self._format_coords
            ax.grid(True)
            ax.set_xlabel(f"time [{self.spec.time_unit}]", labelpad=-12, fontsize=7)

        core.on("view_changed", lambda start, stop: self.repaint())
        core.on("labels_changed", lambda label, change: self.repaint(series=label))
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
            self._draw_pending_interval()
            self.figure.canvas.draw_idle()
        elif field == "shift_reference":
            self._draw_shift_reference()
            self.figure.canvas.draw_idle()
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

    def _subplots_showing(self, series) -> list[int]:
        """Subplots in which a series is plotted (compared by identity)."""
        return [
            index
            for index, (channels, labels) in enumerate(
                zip(self.spec.channels, self.spec.labels)
            )
            if any(item is series for item in [*channels, *labels])
        ]

    def _draw_series(self, subplot: int, series, resolution) -> None:
        """(Re)draw one channel / label, replacing only its own artists."""
        ax = self.channel_axes[subplot]
        for artist in self._artists.pop((subplot, id(series)), []):
            artist.remove()
        children, containers = set(ax.get_children()), list(ax.containers)
        kwargs = dict(
            plot_axes=ax,
            start=self.core.start,
            stop=self.core.stop,
            time_unit=self.spec.time_unit,
            reference_time=self.spec.reference_time,
        )
        if any(series is channel for channel in self.spec.channels[subplot]):
            kwargs["resolution"] = resolution
        series.plot(**kwargs)
        new_containers = [c for c in ax.containers if c not in containers]
        in_containers = {id(a) for c in new_containers for a in c.get_children()}
        self._artists[(subplot, id(series))] = [
            artist
            for artist in ax.get_children()
            if artist not in children and id(artist) not in in_containers
        ] + new_containers

    def repaint(self, series=None) -> None:
        """Redraw the content for the current window: all series, or only the
        given one. Axes settings and other artists are left as they are."""
        start, stop = self.core.start, self.core.stop
        data_width = (stop - start).total_seconds()
        self.screen_pixel_width, _ = self.figure.canvas.get_width_height()
        resolution = pd.Timedelta(seconds=data_width / self.screen_pixel_width)

        self.is_repainting = True
        try:
            with plt.ioff():
                subplots = (
                    range(len(self.channel_axes))
                    if series is None
                    else self._subplots_showing(series)
                )
                for subplot in subplots:
                    ax = self.channel_axes[subplot]
                    items = [*self.spec.channels[subplot], *self.spec.labels[subplot]]
                    for item in items:
                        if series is None or item is series:
                            self._draw_series(subplot, item, resolution)
                    ax.set_xlim((self.to_x(start), self.to_x(stop)))
                    self._update_legend(ax)
                    # y limits: automatic until the first non-empty draw, then kept
                    if ax.get_autoscaley_on() and ax.get_ylim() != (0, 1):
                        ax.set_ylim(ax.get_ylim())
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
        finally:
            self.is_repainting = False
        self.figure.canvas.draw_idle()

    def _update_legend(self, ax) -> None:
        """Create the legend once; afterwards only update an existing one
        (a legend removed by the user stays removed)."""
        if not getattr(ax, "_vitabel_legend_created", False) or ax.get_legend():
            ax.legend(loc="lower right")
            ax._vitabel_legend_created = True

    def _draw_pending_interval(self) -> None:
        """Marker of the first click of an interval, in every subplot showing
        the label (#255)."""
        for artist in self.pending_interval_artists:
            artist.remove()
        self.pending_interval_artists = []
        pending, label = self.core.pending_interval, self.core.active_label
        if pending is None:
            return
        subplots = self._subplots_showing(label) or [pending.subplot]
        self.pending_interval_artists = [
            self.channel_axes[subplot].axvline(
                x=self.to_x(pending.time),
                color=label.plotstyle.get("color", "limegreen"),
                linestyle="--",
                linewidth=1.5,
            )
            for subplot in subplots
        ]

    def _draw_shift_reference(self) -> None:
        if self.shift_reference_artist is not None:
            self.shift_reference_artist.remove()
            self.shift_reference_artist = None
        reference = self.core.shift_reference
        if reference is not None:
            self.shift_reference_artist = self.channel_axes[reference.subplot].axvline(
                x=self.to_x(reference.time), color="red", linestyle="--", linewidth=1.5
            )

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
