"""Signal processing utilities (derivatives, integrals, smoothing, gaps,
resampling, signal features)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.signal as sgn

from typing import TYPE_CHECKING

from vitabel.typing import DataSlice

if TYPE_CHECKING:
    from vitabel.timeseries import Channel

__all__ = [
    "deriv",
    "integrate",
    "av_mean",
    "determine_gaps_in_recording",
    "linear_interpolate_gaps_in_recording",
    "resample_to_common_index",
    "gaussian_kernel_regression_point",
]


def deriv(time, data):
    """
    Calculates the central point first derivative of x=time,y=data,
    and take forward/backward step derivative at the boundaries

    Parameters
    ----------
    time : np.array()
        equidistantly spaced x values.
    data : np.array()
        corresponding y values.

    Returns
    -------
    ddat : np.array()
        array with the derivative values.
    """
    h = time[2:] - time[:-2]
    if np.std(h) == 0:
        h = h[0] / 2
        ddat = data[2:] - data[:-2]
        ddat = ddat / (2 * h)
        ddat = np.insert(ddat, 0, (data[1] - data[0]) / h)
        ddat = np.append(ddat, (data[-1] - data[-2]) / h)
        return ddat
    else:
        ddat = data[2:] - data[0:-2]
        ddat = ddat / (h)
        ddat = np.insert(ddat, 0, (data[1] - data[0]) / (time[1] - time[0]))
        ddat = np.append(ddat, (data[-1] - data[-2]) / (time[-1] - time[-2]))
        return ddat


def integrate(time, data):
    """
    Integrates data after time with trapezoidal rule

    Parameters
    ----------
    time : pd.Series
        Timestamps in seconds.
    data : pd.Series
        Datapoints.

    Returns
    -------
    vel : pd.Series
        Integrated data. vel[0]=0

    """
    if len(time) != len(data):
        raise Exception("Length of time and data must agree")
    time = list(time)
    data = list(data)
    n = len(data)
    vel = [0]
    deltat = time[1] - time[0]
    for i in range(n - 1):
        meana = (data[i] + data[i + 1]) / 2
        vel.append(vel[i] + meana * deltat)
        i += 1
    return pd.Series(vel)


def av_mean(k: int, data: npt.ArrayLike) -> npt.NDArray:
    """
    Calculate a centered average mean of data over k elements.

    The data is zero-padded at the boundaries.
    """
    data = np.array(data)
    n = len(data)
    if k % 2 == 1:  # Calculate centered part of data
        erg = np.zeros(n - k + 1)
        for i in range(k):
            erg += data[i : n - k + i + 1]
        erg = erg / k
    else:
        erg = np.zeros(n - k)
        for i in range(k):
            erg += data[i : n - k + i]
        erg = erg / k
    erginit = np.zeros(k // 2)  # initial part
    ergfinal = np.zeros(k // 2)  # final part
    for i in range(k // 2):
        erginit[i] = (
            np.sum(data[: k // 2 + i + 1]) / k
        )  # since denominator ist still k, behaves like zero-padded data
        ergfinal[i] = np.sum(data[n - k // 2 - i - 1 :]) / (k)
    erg = np.append(erg, ergfinal)  # combine parts to one singal
    erg = np.insert(erg, 0, erginit)
    return erg


def determine_gaps_in_recording(
    time: npt.ArrayLike, data: npt.ArrayLike
) -> (npt.NDArray, npt.NDArray):
    dtime = time[1:] - time[:-1]
    median_dtime = np.nanmedian(dtime)
    gap_start_indices = np.nonzero(dtime > 2 * median_dtime)[0]
    gap_start = []
    gap_stop = []
    for start_index in gap_start_indices:
        starttime = time[start_index]
        stoptime = time[start_index + 1]
        gap_start.append(starttime)
        gap_stop.append(stoptime)
    return np.asarray(gap_start), np.asarray(gap_stop), np.array(gap_start_indices)


def linear_interpolate_gaps_in_recording(
    time: npt.ArrayLike, data: npt.ArrayLike
) -> (npt.NDArray, npt.NDArray):
    """Fill the gaps found by :func:`determine_gaps_in_recording` with points
    at the median time step, linearly interpolated between the samples
    around the gap. The result is sorted like the input."""
    time, data = np.asarray(time), np.asarray(data)
    step = np.nanmedian(np.diff(time))
    *_, gap_start_indices = determine_gaps_in_recording(time, data)
    insert_at, gap_time = [], []
    for index in gap_start_indices:
        start, stop = time[index], time[index + 1]
        count = int(round((stop - start) / step)) - 1  # points inside of the gap
        insert_at.append(np.full(count, index + 1))
        gap_time.append(start + step * np.arange(1, count + 1))
    if not insert_at:
        return time, data
    insert_at, gap_time = np.concatenate(insert_at), np.concatenate(gap_time)
    return (
        np.insert(time, insert_at, gap_time),
        np.insert(data, insert_at, np.interp(gap_time, time, data)),
    )


def resample_to_common_index(
    *channels: Channel,
) -> tuple[DataSlice, ...]:
    """Aligns two or more Channel objects by their time indices, ensuring
    all are sampled at the same time points, including zero-crossings.
    Data are interpolated to a common time base within their overlapping range.

    Parameters
    ----------
    *channels
        Two or more :class:`.Channel` objects to be aligned and interpolated.

    Returns
    -------
    tuple[DataSlice, ...]
        A tuple of :class:`.DataSlice` objects representing the aligned
        channels.

    Raises
    ------
    ValueError
        If there is no overlapping time range between the two channels.

    Notes
    -----
    Zero-crossings are inserted to improve alignment, especially for
    strongly oscillating signals.

    """
    series_list = []
    for idx, channel in enumerate(channels, start=1):
        channel_data = channel.get_data()
        series = pd.Series(
            index=channel_data.time_index,
            data=channel_data.data.flatten(),
            name=f"ch{idx}",
        )
        series_list.append(series)

    start = max([series.index.min() for series in series_list])
    end = min([series.index.max() for series in series_list])
    if start >= end:
        raise ValueError("No overlapping time range between channels.")

    # find and add zero crossings
    extended_series_dict = {}
    for series in series_list:
        zero_crossings = _find_zero_crossings(series)
        extended = pd.concat([series, zero_crossings]).sort_index()
        extended = extended[~extended.index.duplicated(keep="first")]
        extended_series_dict[series.name] = extended

    df = pd.DataFrame(extended_series_dict)

    common_index = df.index.unique().sort_values()
    common_index = common_index[(start <= common_index) & (common_index <= end)]
    df_common = df.interpolate(method="time", limit_area="inside").loc[common_index]

    data_slices = tuple(
        DataSlice(
            time_index=df_common.index,
            data=df_common[col].to_numpy(),
            text_data=None,
        )
        for col in df_common
    )
    return data_slices


def _find_zero_crossings(s: pd.Series) -> pd.Series:
    """Finds zero-crossings in a Series via linear interpolation."""
    y = s.values
    time = s.index

    if len(y) < 2:
        return pd.Series(index=pd.DatetimeIndex([]), data=[], dtype=float)

    cross_indices = np.where(y[:-1] * y[1:] < 0)[0]
    if len(cross_indices) == 0:
        return pd.Series(index=pd.DatetimeIndex([]), data=[], dtype=float)

    y0, y1 = y[cross_indices], y[cross_indices + 1]
    alpha = -y0 / (y1 - y0)
    t0, t1 = time[cross_indices], time[cross_indices + 1]
    cross_times = t0 + (t1 - t0) * alpha

    return pd.Series(data=0.0, index=cross_times)


def gaussian_kernel_regression_point(x0, x, y, sigma=1, max_width_factor=2):
    sigma2 = np.square(sigma)
    dx = x - x0
    if np.min(np.abs(dx)) > max_width_factor * sigma:
        return np.nan
    else:
        w = np.exp(-np.square(dx) / sigma2)
        res = np.sum(w * y) / np.sum(w)
    return res


def _hjorth_params(x, axis=-1):
    """Calculate Hjorth mobility and complexity on given axis.

    Ported from `antropy <https://github.com/raphaelvallat/antropy/blob/98970eb012771951d52b42696fa5f69aa39e6f6b/src/antropy/entropy.py#L937>`__.

    References
    ----------
    - https://en.wikipedia.org/wiki/Hjorth_parameters
    - cite:`10.1016/0013-46947090143-4`

    """
    x = np.asarray(x)
    x_var = np.var(x, axis=axis)

    dx = np.diff(x, axis=axis)
    dx_var = np.var(dx, axis=axis)

    ddx = np.diff(dx, axis=axis)
    ddx_var = np.var(ddx, axis=axis)

    mobility = np.sqrt(dx_var / x_var)
    complexity = np.sqrt(ddx_var / dx_var) / mobility
    return mobility, complexity


# spec_ent_acc = antropy.spectral_entropy(acc, sf=250, normalize=True, method="welch")
def _xlogx(x, base=2):
    r"""Return :math:`x \log_b x` if :math:`x` is positive,
    0 for :math:`x = 0`, and ``numpy.nan`` otherwise.

    Ported from `antropy <https://github.com/raphaelvallat/antropy/blob/98970eb012771951d52b42696fa5f69aa39e6f6b/src/antropy/utils.py#L131>`__.
    """
    x = np.asarray(x)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    positive = x > 0
    xlogx[positive] = x[positive] * np.log(x[positive]) / np.log(base)
    return xlogx


def _spectral_entropy_welch(x, sf, normalize=False, nperseg=None, axis=-1):
    """Spectral Entropy via welch periodogram.

    Ported from `antropy <https://github.com/raphaelvallat/antropy/blob/98970eb012771951d52b42696fa5f69aa39e6f6b/src/antropy/entropy.py#L147>`__.


    References
    ----------
    - https://en.wikipedia.org/wiki/Spectral_density
    - https://en.wikipedia.org/wiki/Welch%27s_method

    """
    x = np.asarray(x)
    _, psd = sgn.welch(x, sf, nperseg=nperseg, axis=axis)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = (-1) * _xlogx(psd_norm).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se
