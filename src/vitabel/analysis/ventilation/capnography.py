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
    ``ventilations_from_capnography``, or ``None`` if the mode is unknown.
    """
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
    if mode == "threshold" or mode == "filter":
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
    else:
        logger.error(
            f"mode {mode} not known. Please use either 'filter' or 'threshold' as argument"
        )
        return None
