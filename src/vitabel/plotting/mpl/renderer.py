"""Drawing of the interactive matplotlib plot, driven by the events of the
interaction core."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.text import Text

from vitabel.plotting.data import minmax_decimate
from vitabel.plotting.interaction import InteractionCore
from vitabel.plotting.layout import PlotSpec
from vitabel.plotting.overlays import (
    FillBetween,
    HLine,
    Overlay,
    VSpan,
    resolve_series,
    series_values,
)


@dataclass(eq=False)
class _PlacedOverlay:
    """An overlay in a subplot, with the channels / labels it is computed
    from (resolved; a constant for ``FillBetween.y2``)."""

    overlay: Overlay
    subplot: int
    inputs: tuple = ()

    def depends_on(self, series) -> bool:
        return any(item is series for item in self.inputs)


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

        # the crosshair is drawn on top of the last full draw (animated: not
        # part of the full draw, does not make the figure stale), so moving
        # the mouse does not re-render the figure; see _handle_draw
        self.x_indicators = [
            ax.axvline(x=0, color="black", linestyle="--", linewidth=0.5, animated=True)
            for ax in self.channel_axes
        ]
        self._background = None
        figure.canvas.mpl_connect("draw_event", self._on_draw)
        self._full_draw = figure.canvas.handle_draw
        figure.canvas.handle_draw = self._handle_draw
        self.overview_indicators = []
        self.pending_interval_artists = []
        self.shift_reference_artist = None
        self.is_repainting = False
        self._artists: dict[tuple, list] = {}
        """Artists drawn per (subplot, key of a series / overlay); only these
        are ever replaced, so user changes to the axes and their own artists
        persist."""
        self.overlays: list[_PlacedOverlay] = []
        """Overlays, drawn in this order."""
        self._legend_created: set[int] = set()
        """Subplots whose legend was created (a removed one stays removed)."""
        for ax in self.channel_axes:
            ax.format_coord = self._format_coords
            ax.grid(True)
            ax.set_xlabel(f"time [{self.spec.time_unit}]", labelpad=-12, fontsize=7)

        core.on("view_changed", lambda start, stop: self.repaint())
        core.on("labels_changed", lambda label, change: self.repaint(changed=label))
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

    def _on_draw(self, event) -> None:
        """After a full draw: keep the image as background for the crosshair."""
        canvas = self.figure.canvas
        self._background = canvas.copy_from_bbox(self.figure.bbox)
        self._blit_indicators()

    def _blit_indicators(self) -> None:
        for indicator in self.x_indicators:
            indicator.axes.draw_artist(indicator)
        self.figure.canvas.blit(self.figure.bbox)

    def _handle_draw(self, event) -> None:
        """Draw request of the browser (ipympl): if only the crosshair changed
        (the figure is not stale), draw it on the saved background instead of
        rendering the whole figure.

        Images are only sent on request of the browser, which waits for the
        previous image: pushing images (blitting on mouse moves) lets the
        browser skip some of them, and since they only contain the changed
        pixels, skipped ones leave fragments.
        """
        if self._background is not None and not self.figure.stale:
            self.figure.canvas.restore_region(self._background)
            self._blit_indicators()
        else:
            self._full_draw(event)

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

    def _subplots_affected_by(self, series) -> list[int]:
        """Subplots showing a series, or an overlay computed from it."""
        subplots = set(self._subplots_showing(series))
        subplots.update(
            placed.subplot for placed in self.overlays if placed.depends_on(series)
        )
        return sorted(subplots)

    def _replace_artists(self, subplot: int, key, draw) -> None:
        """Remove the artists previously drawn under ``key``, call ``draw``,
        and remember the artists (and containers) it added."""
        ax = self.channel_axes[subplot]
        for artist in self._artists.pop((subplot, key), []):
            artist.remove()
        children, containers = set(ax.get_children()), list(ax.containers)
        draw(ax)
        new_containers = [c for c in ax.containers if c not in containers]
        in_containers = {id(a) for c in new_containers for a in c.get_children()}
        self._artists[(subplot, key)] = [
            artist
            for artist in ax.get_children()
            if artist not in children and id(artist) not in in_containers
        ] + new_containers

    def _draw_series(self, subplot: int, series, resolution) -> None:
        """(Re)draw one channel / label, replacing only its own artists."""
        kwargs = dict(
            start=self.core.start,
            stop=self.core.stop,
            time_unit=self.spec.time_unit,
            reference_time=self.spec.reference_time,
        )
        if any(series is channel for channel in self.spec.channels[subplot]):
            kwargs["resolution"] = resolution
        self._replace_artists(
            subplot, id(series), lambda ax: series.plot(plot_axes=ax, **kwargs)
        )

    # -- overlays ----------------------------------------------------------------

    def add_overlay(self, overlay: Overlay, subplot: int) -> None:
        inputs = ()
        if isinstance(overlay, FillBetween):
            collection = self.spec.collection
            y2 = overlay.y2
            if not isinstance(y2, (int, float)):
                y2 = resolve_series(collection, y2)
            inputs = (resolve_series(collection, overlay.y1), y2)
        placed = _PlacedOverlay(overlay, subplot, inputs)
        self.overlays.append(placed)
        self._draw_overlay(placed)
        self.figure.canvas.draw_idle()

    def remove_overlay(self, overlay: Overlay) -> None:
        for placed in [p for p in self.overlays if p.overlay is overlay]:
            for artist in self._artists.pop(self._overlay_key(placed), []):
                artist.remove()
            self.overlays.remove(placed)
        self.figure.canvas.draw_idle()

    @staticmethod
    def _overlay_key(placed: _PlacedOverlay) -> tuple:
        return (placed.subplot, ("overlay", id(placed)))

    def _draw_overlay(self, placed: _PlacedOverlay) -> None:
        overlay = placed.overlay

        def draw(ax):
            if isinstance(overlay, HLine):
                ax.axhline(overlay.y, **overlay.style)
            elif isinstance(overlay, VSpan):
                ax.axvspan(
                    self.to_x(overlay.start), self.to_x(overlay.stop), **overlay.style
                )
            elif isinstance(overlay, FillBetween):
                self._fill_between(ax, overlay, *placed.inputs)

        self._replace_artists(*self._overlay_key(placed), draw)

    def _fill_between(self, ax, overlay: FillBetween, y1, y2) -> None:
        start, stop = self.core.start, self.core.stop
        window = (self.to_x(start), self.to_x(stop))

        def values(series):
            x, y = series_values(
                series, start, stop, self.spec.reference_time, self.unit
            )
            return minmax_decimate(x, y, *window, self.screen_pixel_width)

        x, a = values(y1)
        if len(x) == 0:
            return
        if isinstance(y2, (int, float)):
            b = np.full(len(x), float(y2))
        else:
            x2, b2 = values(y2)
            b = np.interp(x, x2, b2) if len(x2) else np.full(len(x), np.nan)
        where = None
        if overlay.where == "below":
            where = a <= b
        elif overlay.where == "above":
            where = a >= b
        ax.fill_between(x, a, b, where=where, interpolate=True, **overlay.style)

    def repaint(self, changed=None) -> None:
        """Redraw the content for the current window: all subplots, or those
        affected by a changed series (showing it, or an overlay computed from
        it). Only artists drawn by the renderer are replaced; axes settings
        and other artists are left as they are."""
        start, stop = self.core.start, self.core.stop
        data_width = (stop - start).total_seconds()
        self.screen_pixel_width, _ = self.figure.canvas.get_width_height()
        resolution = pd.Timedelta(seconds=data_width / self.screen_pixel_width)

        self.is_repainting = True
        try:
            with plt.ioff():
                subplots = (
                    range(len(self.channel_axes))
                    if changed is None
                    else self._subplots_affected_by(changed)
                )
                for subplot in subplots:
                    ax = self.channel_axes[subplot]
                    # all series of the subplot, in order, with a fresh color
                    # cycle: series without explicit color keep their colors
                    ax.set_prop_cycle(None)
                    for item in [
                        *self.spec.channels[subplot],
                        *self.spec.labels[subplot],
                    ]:
                        self._draw_series(subplot, item, resolution)
                    for placed in self.overlays:
                        if placed.subplot == subplot:
                            self._draw_overlay(placed)
                    ax.set_xlim((self.to_x(start), self.to_x(stop)))
                    self._update_legend(subplot)
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

    def _update_legend(self, subplot: int) -> None:
        """Legend of the series visible in the window.

        Created on the first draw. Afterwards it is left alone (with all
        changes made to it) unless its entries change; then it is rebuilt,
        keeping its location and title. A legend removed by the user stays
        removed.
        """
        ax = self.channel_axes[subplot]
        legend = ax.get_legend()
        if legend is None:
            if subplot not in self._legend_created:
                ax.legend(loc="lower right")
                self._legend_created.add(subplot)
            return
        _, entries = ax.get_legend_handles_labels()
        if entries == [text.get_text() for text in legend.get_texts()]:
            return
        title = legend.get_title().get_text()
        # matplotlib has no public getter for the location of a legend
        ax.legend(loc=legend._loc, title=title or None)

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
