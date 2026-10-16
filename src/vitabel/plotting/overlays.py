"""Overlays: additional, persistent elements of interactive plots, recomputed
for every time window (e.g. the area of a signal below a threshold).

Styles are matplotlib keyword arguments.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from vitabel.timeseries import Channel, Label


class Overlay:
    """Base class of overlays."""

    def __init__(self, **style: Any):
        self.style = style


class FillBetween(Overlay):
    """Fill the area between two curves, like
    ``matplotlib.pyplot.fill_between(..., interpolate=True)``.

    Parameters
    ----------
    y1
        A channel or label (or its name).
    y2
        A channel or label (or its name), or a constant.
    where
        Only where ``y1 <= y2`` (``"below"``) or ``y1 >= y2`` (``"above"``).
    style
        Keyword arguments for ``fill_between`` (e.g. ``color``, ``alpha``).

    Example: the area of the mean arterial pressure below 65 mmHg::

        plot.add_overlay(FillBetween("MAP", 65, where="below", color="orange"))
    """

    def __init__(
        self,
        y1: str | Channel | Label,
        y2: str | Channel | Label | float = 0.0,
        where: Literal["below", "above"] | None = None,
        **style: Any,
    ):
        super().__init__(**style)
        if where not in (None, "below", "above"):
            raise ValueError(f"where must be 'below', 'above' or None, not {where!r}")
        self.y1, self.y2, self.where = y1, y2, where

    def depends_on(self, series) -> bool:
        return any(item is series for item in (self.y1, self.y2))


class HLine(Overlay):
    """A horizontal line at ``y`` (e.g. a threshold); style as for
    ``matplotlib.pyplot.axhline``."""

    def __init__(self, y: float, **style: Any):
        super().__init__(**style)
        self.y = y


class VSpan(Overlay):
    """A shaded time range; style as for ``matplotlib.pyplot.axvspan``."""

    def __init__(self, start, stop, **style: Any):
        super().__init__(**style)
        self.start, self.stop = start, stop


def resolve_series(collection, spec):
    """A channel or label given by name (channels first) or as object."""
    if isinstance(spec, (Channel, Label)):
        return spec
    if any(channel.name == spec for channel in collection.channels):
        return collection.get_channel(spec)
    return collection.get_label(spec)


def series_values(
    series, start, stop, reference_time, unit
) -> tuple[np.ndarray, np.ndarray]:
    """Times (numeric, relative to the reference time) and values of a
    channel or label in a window, including adjacent samples."""
    data = series.get_data(start=start, stop=stop, include_adjacent=True)
    times = data.time_index
    if len(times) == 0:
        return np.array([]), np.array([])
    x = np.asarray((times - reference_time) / unit, dtype=float)
    if data.data is None:
        return x, np.full(len(x), np.nan)
    return x, np.asarray(data.data, dtype=float)
