"""Data layer of the plots: what to draw, independent of the plotting library.

Pure functions on numpy arrays.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def minmax_decimate(
    x: npt.NDArray, y: npt.NDArray, x0: float, x1: float, n_buckets: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """Reduce samples to at most two per bucket: the minimum and the maximum,
    in their original order.

    The interval ``[x0, x1]`` is divided into ``n_buckets`` buckets of equal
    width (e.g. one per pixel). Drawn as a line at that resolution, the
    result looks like the full data (unlike taking every n-th sample, peaks
    are kept). Samples outside of ``[x0, x1]`` are kept as they are (e.g.
    the adjacent samples for lines to the edges).

    Parameters
    ----------
    x
        Sorted sample times (numeric).
    y
        Sample values. Buckets without finite values yield a NaN (a gap).
    x0, x1
        The interval to divide into buckets.
    n_buckets
        The number of buckets.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if n_buckets <= 0 or len(x) <= 4 * n_buckets:
        return x, y
    edges = np.searchsorted(x, np.linspace(x0, x1, n_buckets + 1), side="left")
    edges[-1] = np.searchsorted(x, x1, side="right")
    starts, ends = edges[:-1], edges[1:]
    nonempty = ends > starts
    if not nonempty.any():
        return x, y
    starts, ends = starts[nonempty], ends[nonempty]
    offset, stop = starts[0], ends[-1]
    segment = y[offset:stop]
    local_starts = starts - offset
    counts = ends - starts

    with np.errstate(invalid="ignore"):
        lows = np.fmin.reduceat(segment, local_starts)
        highs = np.fmax.reduceat(segment, local_starts)
    positions = np.arange(len(segment))
    missing = len(segment)

    def first_position_of(values: npt.NDArray) -> npt.NDArray:
        is_value = segment == np.repeat(values, counts)
        found = np.minimum.reduceat(
            np.where(is_value, positions, missing), local_starts
        )
        # buckets without finite values: their first sample (a NaN)
        return np.where(found == missing, local_starts, found)

    at_low, at_high = first_position_of(lows), first_position_of(highs)
    selected = np.empty(2 * len(starts), dtype=np.intp)
    selected[0::2] = np.minimum(at_low, at_high)
    selected[1::2] = np.maximum(at_low, at_high)
    selected += offset
    keep = np.concatenate([np.arange(offset), selected, np.arange(stop, len(x))])
    return x[keep], y[keep]
