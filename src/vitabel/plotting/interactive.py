"""Interactive plots (matplotlib, ipympl and ipywidgets)."""

from __future__ import annotations

import logging
from typing import Any


from vitabel.plotting.layout import PlotSpec
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
    from IPython import get_ipython

    from vitabel.plotting.mpl.plot import InteractivePlot

    spec = PlotSpec.from_arguments(
        collection,
        channels=channels,
        labels=labels,
        start=start,
        stop=stop,
        time_unit=time_unit,
        include_attached_labels=include_attached_labels,
        channel_overviews=channel_overviews,
        limited_overview=limited_overview,
    )
    ipy_shell = get_ipython()
    if ipy_shell is None:
        raise RuntimeError("This method can only be used in an IPython environment")
    ipy_shell.enable_matplotlib(gui="widget")
    return InteractivePlot(spec, subplots_kwargs)
