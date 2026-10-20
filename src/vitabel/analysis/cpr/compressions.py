"""Analysis of chest compressions: periods of continuous chest compressions
(from single compression events or from the accelerometer signal of CPR
feedback sensors), and the chest compression fraction."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.signal as sgn

from vitabel.timeseries import Channel, IntervalLabel
from vitabel.utils.signal import (
    av_mean,
    deriv,
    determine_gaps_in_recording,
    linear_interpolate_gaps_in_recording,
)
from vitabel.utils.stylesheet import DEFAULT_PLOT_STYLE

logger = logging.getLogger("vitabel")


def cc_periods_from_compressions(cc_events_channel: Channel) -> IntervalLabel:
    """Determine periods of continuous chest compressions based on single
    chest compression events.

    Parameters
    ----------
    cc_events_channel
        The channel containing single chest compression events, such that
        every time point in the time index represents a chest compression.

    Returns
    -------
    An (unattached) interval label ``cc_periods``.

    .. SEEALSO::
        The method is described in :cite:`10.1016/j.resuscitation.2021.12.028`,
        or in the PhD thesis :cite:`kern:phd:2024` of Wolfgang Kern in more detail.
    """
    comp = cc_events_channel.get_data().time_index  # get data

    t_ref = cc_events_channel.time_start

    if cc_events_channel.is_time_relative():
        comp = comp.total_seconds().to_numpy()
    else:
        comp = np.sort(comp)
        comp = np.array(
            [pd.Timedelta(t - t_ref).total_seconds() for t in comp]
        )  # TODO: check if times have ti be coerced

    compression_counter = 1  # number of compressions in cc period
    last_c = comp[0]  # initilaize last compression
    sta = np.array([comp[0]])  # start ... = first compression
    sto = np.array([comp[0]])  # stop preliminary = first compression
    fre = np.array([0.6])  # take a cc length of 0.6 s = 100 bpm as start value
    for c in comp[1:]:  # Iterate through all compressions
        if c - last_c < 3 * np.mean(fre[-5:]):
            # If difference from next marker to last marker is smaller than
            # three times average cc length, then compressions are connected
            # (one period)
            if len(fre) == 1:  # Remove initial cc length guess
                fre = np.array([])
            fre = np.append(fre, c - last_c)  # estimate cc length with the actual value
            sto[-1] = c  # stop is new marker

            last_c = c  # reload last compression
            compression_counter += 1
        else:  # If difference betweeen markers is larger 3 times average cc length
            if (
                compression_counter < 3
            ):  # If less then three compressions delete start and stop markers
                sta = np.delete(sta, -1)
                sto = np.delete(sto, -1)
            else:  # If compression period is valid
                sta[-1] = sta[-1] - np.mean(fre[:5])  # correct starting point with
            sta = np.append(sta, c)
            sto = np.append(sto, c)
            fre = np.array([0.6])
            last_c = c
            compression_counter = 1
    if compression_counter < 3:
        sta = np.delete(sta, -1)
        sto = np.delete(sto, -1)
    else:
        sta[-1] = sta[-1] - np.mean(fre[:5])
    metadata = {
        "creator": "automatic",
        "creation_date": pd.Timestamp.now(),
        "method": "cycle_duration_analysis",
    }

    periods = np.empty(sta.size + sto.size, dtype=sta.dtype)
    periods[0::2] = sta
    periods[1::2] = sto

    cc_periods = IntervalLabel(
        name="cc_periods",
        time_index=periods,
        time_start=t_ref,
        metadata=metadata,
        plot_type="box",
        plotstyle=DEFAULT_PLOT_STYLE.get("cc_periods", None),
    )

    return cc_periods


def cc_periods_from_acceleration(accelerometer_channel: Channel) -> IntervalLabel:
    """Detect periods of continuous chest compressions from the accelerometer
    signal of a CPR feedback sensor.

    Uses the root mean square of the accelerometer signal to detect the rise
    in "power" of the signal caused by continuous chest compressions, as
    described in :cite:`10.1016/j.resuscitation.2021.12.028` and
    :cite:`10.1016/j.dib.2022.107973`.

    Parameters
    ----------
    accelerometer_channel
        The channel containing the accelerometer signal.

    Returns
    -------
    An (unattached) interval label ``cc_periods``; every interval is a
    single period of chest compressions.

    .. SEEALSO::
        Details are given in :cite:`10.1016/j.resuscitation.2021.12.028` or in
        the PhD thesis :cite:`kern:phd:2024` by Wolfgang Kern in more detail.
    """
    ACC_channel = accelerometer_channel
    acc_data = ACC_channel.get_data()  # get data
    acctime, acc = acc_data.time_index, acc_data.data
    freq = np.timedelta64(1, "s") / np.nanmedian(acctime.diff())
    t_ref = ACC_channel.time_start

    if ACC_channel.is_time_relative():
        acctime = np.asarray(acctime / pd.Timedelta(seconds=1))
    else:
        acctime = np.asarray([(t - t_ref).total_seconds() for t in acctime])

    acctime = np.asarray(acctime)
    acc = np.asarray(acc - np.mean(acc))
    gap_start, gap_stop, gap_start_indices = determine_gaps_in_recording(acctime, acc)
    acctime, acc = linear_interpolate_gaps_in_recording(acctime, acc)
    but = sgn.butter(4, (0.2 * 2 / freq, 50 * 2 / freq), btype="bandpass", output="sos")
    acce = sgn.sosfilt(but, acc)
    window_size = int(freq)  # Good to ignore short pauses
    softthres = 10
    avacc = av_mean(window_size, np.abs(acce))  # Average mean of abs(acc)
    davacc = deriv(acctime, avacc)  # Derivative of average mean
    avdavacc = av_mean(window_size, davacc)  # av_mean of derivative
    # Soft Thresholding to get rid of small extrema in derivative due to oscillations during cpr
    avdavacc = np.maximum(np.abs(avdavacc) - softthres, 0) * np.sign(avdavacc)

    n = len(acctime)
    thresh = 0
    peakmark = (avdavacc[2:] - avdavacc[1:-1]) * (
        avdavacc[1:-1] - avdavacc[:-2]
    )  # Determine peaks in averagre derivative
    pointcand = np.arange(0, n - 2)[
        (peakmark <= 0) & (np.abs(avdavacc[1:-1]) > thresh)
    ]  # possible starting poins
    points = {"Start": np.array([], int), "Stop": np.array([], int)}
    flag = False  # Start with search for starting point

    cand = 0
    icand = 0
    for i in pointcand:
        if (
            not flag and avdavacc[i] < 0
        ):  # while searching for start a stoplike value appears. save start value
            points["Start"] = np.append(points["Start"], icand)
            flag = not flag
            cand = 0
        elif (
            flag and avdavacc[i] > 0
        ):  # while searching for stop a startike value appears. save stop value
            points["Stop"] = np.append(points["Stop"], icand)
            flag = not flag
            cand = 0
        if not flag:  # Searching for start: Get maximum of derivative
            if avdavacc[i] > cand:
                icand = i
                cand = avdavacc[icand]
        else:  # Searching for end: Get minimum of derivative
            if avdavacc[i] < cand:
                icand = i
                cand = avdavacc[icand]
    if not flag:  # add last point to start or endpoint (not in loop included)
        points["Start"] = np.append(points["Start"], icand)
    else:
        points["Stop"] = np.append(points["Stop"], icand)

    badpoints = np.array([], int)
    for i in range(
        np.minimum(len(points["Start"]), len(points["Stop"])) - 1
    ):  # Delete pauses, where average mean stays over 0.5 * mean(CPR Phase before and CPR phase after)
        pausethresh = (
            0.35
            * 0.5
            * (
                np.mean(avacc[points["Start"][i] : points["Stop"][i]])
                + np.mean(avacc[points["Start"][i + 1] : points["Stop"][i + 1]])
            )
        )
        if np.min(avacc[points["Stop"][i] : points["Start"][i + 1]]) > pausethresh:
            badpoints = np.append(badpoints, i)

    points["Start"] = np.delete(points["Start"], badpoints + 1)
    points["Stop"] = np.delete(points["Stop"], badpoints)

    pauselen = 1.6  # a CPR phase must last at least 1.6 seconds, delete shorter ones
    badpoints2 = np.array([], int)
    for i in range(
        np.minimum(len(points["Start"]), len(points["Stop"]))
    ):  # Delete CPR-Periods which are shorter then 2.5s
        if acctime[points["Stop"][i]] - acctime[points["Start"][i]] < pauselen:
            badpoints2 = np.append(badpoints2, i)

    points["Start"] = np.delete(points["Start"], badpoints2)
    points["Stop"] = np.delete(points["Stop"], badpoints2)

    cpr_thresh = 28  # 30 #Acc_mean while cpr is not allowed to be below this threshold (unit presumably equivalent 3,5 inch /s^2 = 0.09 m/s^2, but units remain unclear)

    badpoints3 = np.array([], int)
    for i in range(
        np.minimum(len(points["Start"]), len(points["Stop"]))
    ):  # Delete CPR-Periods which are shorter then 2.5s
        if np.mean(avacc[points["Start"][i] : points["Stop"][i]]) < cpr_thresh:
            badpoints3 = np.append(badpoints3, i)

    points["Start"] = np.delete(points["Start"], badpoints3)
    points["Stop"] = np.delete(points["Stop"], badpoints3)
    nlen = int(freq // 2)
    for i in range(len(points["Start"])):
        elem = points["Start"][i]
        if elem > nlen and elem < n - nlen:
            points["Start"][i] = int(
                np.sum(
                    avdavacc[elem - nlen : elem + nlen]
                    * np.arange(elem - nlen, elem + nlen, 1)
                )
                / np.sum(avdavacc[elem - nlen : elem + nlen])
            )
    for i in range(len(points["Stop"])):
        elem = points["Stop"][i]
        if elem > nlen and elem < n - nlen:
            points["Stop"][i] = int(
                np.sum(
                    avdavacc[elem - nlen : elem + nlen]
                    * np.arange(elem - nlen, elem + nlen, 1)
                )
                / np.sum(avdavacc[elem - nlen : elem + nlen])
            )

    if points["Start"][0] > points["Stop"][0]:
        points["Stop"] = np.delete(points["Stop"], 0)
    if points["Start"][-1] > points["Stop"][-1]:
        points["Start"] = np.delete(points["Start"], -1)

    starts = acctime[points["Start"]]
    stops = acctime[points["Stop"]]

    gap_starts_to_append = []
    gap_stops_to_append = []

    for gap_i, gap_f in zip(gap_start, gap_stop):
        if len(starts[starts < gap_i]) > 0:
            if len(stops[stops < gap_i]) == 0:
                gap_starts_to_append.append(gap_i)
            elif starts[starts < gap_i][-1] > stops[stops < gap_i][-1]:
                gap_starts_to_append.append(gap_i)
        if len(stops[stops > gap_i]) > 0:
            if len(starts[starts > gap_i]) == 0:
                gap_stops_to_append.append(gap_f)
            elif starts[starts > gap_i][0] > stops[stops > gap_i][0]:
                gap_stops_to_append.append(gap_f)

    starts = np.append(starts, gap_stops_to_append)
    stops = np.append(stops, gap_starts_to_append)

    starts = np.sort(starts)
    stops = np.sort(stops)
    # NOTE: unsure if we have to maintain starts and stops, we might even just go wit append and sort
    periods = np.empty(starts.size + stops.size, dtype=starts.dtype)
    periods[0::2] = starts
    periods[1::2] = stops

    metadata = {
        "creator": "automatic",
        "creation_date": pd.Timestamp.now(),
        "method": "RMS_period_dection",
    }

    cc_periods = IntervalLabel(
        name="cc_periods",
        time_index=periods,
        time_start=t_ref,
        metadata=metadata,
        plot_type="box",
        plotstyle=DEFAULT_PLOT_STYLE.get("cc_periods", None),
    )

    return cc_periods


def convert_two_alternating_list(df):
    lis = []
    for index, value in df.iterrows():
        lis.extend([index, float(value.iloc[0])])
    return lis


def CCF_minute(t_start, t_stop, CC_starts, CC_stops):
    CC_starts_min = CC_starts[(CC_starts >= t_start) & (CC_starts < t_stop)]
    CC_stops_min = CC_stops[(CC_stops >= t_start) & (CC_stops < t_stop)]
    if len(CC_starts_min) > 0 and len(CC_stops_min) > 0:
        if len(CC_starts_min) == len(CC_stops_min):
            if CC_stops_min[0] < CC_starts_min[0]:
                CC_starts_min = np.insert(CC_starts_min, 0, t_start)
                CC_stops_min = np.append(CC_stops_min, t_stop)
        elif len(CC_starts_min) > len(CC_stops_min):
            CC_stops_min = np.append(CC_stops_min, t_stop)
        else:
            CC_starts_min = np.insert(CC_starts_min, 0, t_start)

    elif len(CC_starts_min) == 0 and len(CC_stops_min) == 0:
        last_CC_start = CC_starts[CC_starts < t_start]
        last_CC_stop = CC_stops[CC_stops < t_start]
        if len(last_CC_start) == 0 and len(last_CC_stop) == 0:
            CC_starts_min = np.array([t_start])
            CC_stops_min = np.array([t_start])
        elif len(last_CC_start) != 0 and len(last_CC_stop) == 0:
            CC_starts_min = np.array([t_start])
            CC_stops_min = np.array([t_stop])
        elif last_CC_start[-1] < last_CC_stop[-1]:
            CC_starts_min = np.array([t_start])
            CC_stops_min = np.array([t_start])
        else:
            CC_starts_min = np.array([t_start])
            CC_stops_min = np.array([t_stop])

    elif len(CC_starts_min) > 0:
        CC_stops_min = np.array([t_stop])
    elif len(CC_stops_min) > 0:
        CC_starts_min = np.array([t_start])

    return np.sum(CC_stops_min - CC_starts_min) / 60000
