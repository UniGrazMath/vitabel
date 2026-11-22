"""Detection of end-tidal CO₂ (etCO₂) values and ventilations from the
capnography waveform."""

from __future__ import annotations

import logging

from typing import Literal

import numpy as np
import pandas as pd
import scipy.signal as sgn

from vitabel.timeseries import Channel, Label
from vitabel.utils.stylesheet import DEFAULT_PLOT_STYLE

logger = logging.getLogger("vitabel")


def detect_etco2_and_ventilations(
    source: Channel,
    mode: Literal["threshold", "filter"] = "filter",
    breath_threshold: float = 2,
    etco2_threshold: float = 3,
) -> tuple[Label, Label] | None:
    """Detect end-tidal CO₂ (etCO₂) values and ventilations in a capnography
    waveform.

    Two detection methods are supported:

    - ``'filter'``: An unpublished method by Wolfgang Kern (default).
    - ``'threshold'``: The method described by Aramendi et al. in
      :cite:`10.1016/j.resuscitation.2016.08.033`.

    Parameters
    ----------
    source
        The capnography channel (CO₂ in mmHg).
    mode
        Method to use for detecting ventilations from the CO₂ signal.
    breath_threshold
        Threshold below which a minimum is identified as a ventilation
        (default: 2 mmHg). Used by the ``'filter'`` method.
    etco2_threshold
        Threshold above which a maximum is identified as an etCO₂ value of an
        expiration (default: 3 mmHg). Used by the ``'filter'`` method.

    Returns
    -------
    The (unattached) labels ``etco2_from_capnography`` and
    ``ventilations_from_capnography``. They are empty if no breaths are
    detected (e.g., for the ``'filter'`` method if the CO₂ signal does not
    drop below ``breath_threshold``).

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported methods.
    """
    if mode not in ("filter", "threshold"):
        raise ValueError(f"Unknown mode {mode!r}, use either 'filter' or 'threshold'")
    co2_data = source.get_data()  # get data
    cotime, co = co2_data.time_index, co2_data.data

    freq = np.timedelta64(1, "s") / np.nanmedian(cotime.diff())
    cotime = np.asarray(cotime)
    co = np.asarray(co)
    if mode == "filter":  # Wolfgang Kern's unpublished method
        but = sgn.butter(4, 1 * 2 / freq, btype="lowpass", output="sos")
        co2 = sgn.sosfiltfilt(but, co)  # Filter forwarsd and backward
        et_index = sgn.find_peaks(co2, distance=1 * freq, height=etco2_threshold)[
            0
        ]  # find peaks of filtered signal as markers for etco2
        resp_index = sgn.find_peaks(
            -co2, distance=1 * freq, height=-breath_threshold
        )[  # find dips of filtered signal as markers for ventilations
            0
        ]

        etco2time = cotime[et_index]  # take elements on this markers
        etco2 = co[et_index]
        resptime = cotime[resp_index]
        resp_height = co[resp_index]

        if len(resptime) == 0:
            logger.warning(
                "No ventilations detected: the CO₂ signal does not drop below "
                f"breath_threshold ({breath_threshold} mmHg)."
            )
            return _labels(etco2time[:0], etco2[:0], resptime, mode)

        # initialize search for other markers
        k = 0
        del_resp = []
        more_resp_flag = False
        min_resp_height = np.nan
        min_resp_index = np.nan

        # look a signal before first ventilation

        co2_maxtime = etco2time[(etco2time < resptime[0])]
        co2_max = etco2[(etco2time < resptime[0])]
        netco2 = len(co2_maxtime)

        # when there is more than a single maximum before first respiration, take only largest one
        if netco2 > 1:
            k_max = np.argmax(co2_max)
            for j in range(k + k_max, k, -1):
                etco2 = np.delete(etco2, k)
                etco2time = np.delete(etco2time, k)
            for j in range(k + netco2, k + k_max + 1, -1):
                etco2 = np.delete(etco2, k + 1)
                etco2time = np.delete(etco2time, k + 1)
            k += 1

        # if there is no maximum
        elif netco2 == 0:
            pass
        # if there is a single maximum
        else:
            k += 1

        for i, resp in enumerate(resptime[:-1]):
            next_resp = resptime[i + 1]
            # check maxima until next respiration same as bevfore
            co2_maxtime = etco2time[(etco2time >= resp) & (etco2time < next_resp)]
            co2_max = etco2[(etco2time >= resp) & (etco2time < next_resp)]
            netco2 = len(co2_maxtime)
            if netco2 > 1:  # take largest one
                k_max = np.argmax(co2_max)
                for j in range(k + k_max, k, -1):
                    etco2 = np.delete(etco2, k)
                    etco2time = np.delete(etco2time, k)
                for j in range(k + netco2, k + k_max + 1, -1):
                    etco2 = np.delete(etco2, k + 1)
                    etco2time = np.delete(etco2time, k + 1)
                k += 1
                more_resp_flag = False
            elif netco2 == 0:
                if more_resp_flag:
                    if resp_height[i] > min_resp_height:
                        del_resp.append(i)
                    else:
                        del_resp.append(min_resp_index)
                        min_resp_height = resp_height[i]
                else:
                    if resp_height[i] > resp_height[i + 1]:
                        del_resp.append(i)
                        min_resp_height = resp_height[i + 1]
                        min_resp_index = i + 1

                        more_resp_flag = True
                    else:
                        del_resp.append(i + 1)
                        min_resp_height = resp_height[i]
                        min_resp_index = i
                        more_resp_flag = True

            else:
                more_resp_flag = False
                k += 1

        del_resp.sort()
        for i in del_resp[::-1]:
            resptime = np.delete(resptime, i)

    elif mode == "threshold":  # Aramendi et al., 2016
        but = sgn.butter(4, 10 * 2 / freq, btype="lowpass", output="sos")
        co2 = sgn.sosfiltfilt(but, co)  # Filter forwarsd and backward
        d = freq * (co2[1:] - co2[:-1])
        exp_index2 = sgn.find_peaks(d, height=0.35 * freq)[0]
        ins_index2 = sgn.find_peaks(-d, height=0.45 * freq)[0]

        if len(exp_index2) < 2 or len(ins_index2) < 2:
            # a breath needs an inspiration, an expiration, and the next
            # inspiration; the search below assumes several of each
            logger.warning("No ventilations detected in the CO₂ signal.")
            return _labels([], [], [], mode)

        final_flag = False
        ins_index3 = []
        exp_index3 = []
        j_ins = 0
        j_exp = 0
        while not final_flag:
            ins_index3.append(ins_index2[j_ins])
            while exp_index2[j_exp] < ins_index2[j_ins]:
                j_exp += 1
                if j_exp == len(exp_index2) - 1:
                    final_flag = True
                    break
            exp_index3.append(exp_index2[j_exp])
            while ins_index2[j_ins] < exp_index2[j_exp]:
                j_ins += 1
                if j_ins == len(ins_index2) - 1:
                    final_flag = True
                    break

        resptime = []
        etco2time = []
        etco2 = []
        Th1_list = [5 for i in range(5)]
        Th2_list = [0.5 for i in range(5)]
        Th3_list = [0 for i in range(5)]

        k = 0
        for i_ins, i_exp, i_next_ins in zip(
            ins_index3[:-1], exp_index3[:-1], ins_index3[1:]
        ):
            D = (i_exp - i_ins) / freq
            A_exp = 1 / (i_next_ins - i_exp) * np.sum(co2[i_exp:i_next_ins])
            A_ins = 1 / (freq * D) * np.sum(co2[i_ins:i_exp])
            A_r = (A_exp - A_ins) / A_exp
            S = 1 / freq * np.sum(co2[i_exp : i_exp + int(freq)])
            if len(resptime) > 0:
                t_ref = pd.Timedelta((cotime[i_exp] - resptime[-1])).total_seconds()
            else:
                t_ref = 2  # if t_ref >1.5 then it is ok, so 2 does the job
            if D > 0.3:
                if (
                    A_exp > 0.4 * np.mean(Th1_list)
                    and A_r > np.min([0.7 * np.mean(Th2_list), 0.5])
                    and S > 0.4 * np.mean(Th3_list)
                ):
                    if t_ref > 1.5:
                        resptime.append(cotime[i_exp])
                        Th1_list[k] = A_exp
                        Th2_list[k] = A_r
                        Th3_list[k] = S
                        etco2time.append(
                            cotime[i_exp + np.argmax(co2[i_exp:i_next_ins])]
                        )
                        etco2.append(np.max(co2[i_exp:i_next_ins]))
                        k += 1
                        k = k % 5
    return _labels(etco2time, etco2, resptime, mode)


def detect_co2_rise_onsets(
    source: Channel,
    ventilations: Label,
    etco2: Label,
    rise_fraction: float = 0.1,
) -> Label:
    """Detect the onsets of the CO₂ rise (the expiratory upstroke) in a
    capnography waveform.

    For every ventilation, the CO₂ rise ends at the following etCO₂ value and
    is searched for after the previous etCO₂ value. In between, the baseline
    is the lowest CO₂ value, and the onset is the last sample before the
    etCO₂ value at which the CO₂ signal is at or below
    ``baseline + rise_fraction * (etco2 - baseline)``.

    .. note::
        The onset of the CO₂ rise marks the end of phase I of the capnogram,
        not the start of expiration: the gas exhaled first comes from the
        anatomical dead space and contains no CO₂, and sidestream
        capnographs add a sampling delay. The onsets are therefore later
        than the start of expiration, e.g., as detected from airflow by
        :meth:`.Vitals.compute_respiratory_phases`.

    The ventilations select the breaths: one onset per ventilation that is
    followed by an etCO₂ value before the next ventilation. Their exact times
    do not matter (the ``'filter'`` method places them at the minimum of the
    CO₂ signal before the rise, the ``'threshold'`` method at the steepest
    rise), so the onsets of both methods agree. If the CO₂ signal rises
    several times before an etCO₂ value (e.g., because of chest compression
    artifacts), the onset belongs to the last rise, the one that leads to the
    etCO₂ value.

    This works with the ventilations and etCO₂ values of either method of
    :func:`detect_etco2_and_ventilations` and does not assume that the CO₂
    signal returns to (close to) zero between breaths.

    Parameters
    ----------
    source
        The capnography channel (CO₂ in mmHg).
    ventilations
        The detected ventilations (e.g., ``ventilations_from_capnography``).
    etco2
        The detected etCO₂ values (e.g., ``etco2_from_capnography``); only
        their times are used.
    rise_fraction
        The fraction of the CO₂ rise (from the baseline to the etCO₂ value)
        that defines the onset, between 0 and 1 (default: 0.1).

    Returns
    -------
    The (unattached) label ``co2_rise_onsets_from_capnography``. Breaths
    without an etCO₂ value before the next ventilation, or without a rise,
    have no onset.
    """
    if not 0 <= rise_fraction < 1:
        raise ValueError(f"rise_fraction must be in [0, 1), but got {rise_fraction}")
    co2_data = source.get_data()
    time = np.asarray(co2_data.time_index)
    co2 = np.asarray(co2_data.data, dtype=float)
    vent_times = np.asarray(ventilations.get_data().time_index)
    etco2_times = np.asarray(etco2.get_data().time_index)

    onsets = []
    # the next etCO₂ value after each ventilation ...
    next_etco2 = np.searchsorted(etco2_times, vent_times, side="right")
    for k in range(len(vent_times)):
        if next_etco2[k] == len(etco2_times):
            continue
        peak_time = etco2_times[next_etco2[k]]
        # ... must come before the next ventilation
        if k + 1 < len(vent_times) and peak_time >= vent_times[k + 1]:
            continue
        # the rise is searched for after the previous etCO₂ value
        start = (
            np.searchsorted(time, etco2_times[next_etco2[k] - 1], side="right")
            if next_etco2[k] > 0
            else 0
        )
        peak = np.searchsorted(time, peak_time)
        if peak >= len(time) or peak <= start:
            continue
        segment = co2[start : peak + 1]
        baseline, top = np.nanmin(segment), segment[-1]
        if not top > baseline:
            continue
        level = baseline + rise_fraction * (top - baseline)
        below = np.flatnonzero(segment[:-1] <= level)
        if len(below):
            onsets.append(time[start + below[-1]])

    n_missing = len(vent_times) - len(onsets)
    if n_missing:
        logger.info(f"No CO₂ rise onset found for {n_missing} ventilation(s).")

    return Label(
        "co2_rise_onsets_from_capnography",
        time_index=onsets if onsets else time[:0],
        data=None,
        metadata={
            "creator": "automatic",
            "creation_date": pd.Timestamp.now(),
            "rise_fraction": rise_fraction,
        },
        plotstyle=DEFAULT_PLOT_STYLE.get("co2_rise_onsets_from_capnography", None),
    )


def _labels(etco2time, etco2, resptime, mode) -> tuple[Label, Label]:
    """The labels ``etco2_from_capnography`` and
    ``ventilations_from_capnography``."""
    metadata = {
        "creator": "automatic",
        "creation_date": pd.Timestamp.now(),
        "creation_mode": mode,
    }
    etco2_lab = Label(
        "etco2_from_capnography",
        time_index=etco2time,
        data=etco2,
        metadata=metadata,
        plotstyle=DEFAULT_PLOT_STYLE.get("etco2_from_capnography", None),
    )
    vent_lab = Label(
        "ventilations_from_capnography",
        time_index=resptime,
        data=None,
        metadata=metadata,
        plotstyle=DEFAULT_PLOT_STYLE.get("ventilations_from_capnography", None),
    )
    return etco2_lab, vent_lab
