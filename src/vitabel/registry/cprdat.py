"""Analysis of defibrillator recordings for the German Resuscitation Registry
(GRR / Deutsches Reanimationsregister).

Ported from the last known-good version (``b2b678c``, before the function was
merged into ``main``) to the current API; see ``tests/test_cprdat.py`` and
``scripts/compare_cprdat.py`` for the comparison with the reference outputs.
"""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
import pandas as pd

from vitabel.analysis.cpr.compressions import CCF_minute, convert_two_alternating_list
from vitabel.analysis.circulation.rosc import find_ROSC_2
from vitabel.timeseries import IntervalLabel
from vitabel.utils.signal import gaussian_kernel_regression_point

logger = logging.getLogger("vitabel")


def _time_and_data(series):
    """Time index and data of a channel or label (formerly obtained by
    unpacking ``get_data()``)."""
    data_slice = series.get_data()
    return data_slice.time_index, data_slice.data


def _first_available(metadata: dict, *keys: str):
    """The value of the first of the given metadata keys that is present."""
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _recording_file_name(vitals) -> str | None:
    """Name of the first recording file added to the Vitals object."""
    files = vitals.metadata.get("Recording_files_added", [])
    return Path(files[0]).name if files else None


def _cc_events_channel(vitals):
    """The channel with single chest compression events (the same channel the
    compressions ``CC`` are taken from in :func:`make_cprdat_analysis`)."""
    for name in ["cc_depth", "cc"]:
        if name in vitals.get_channel_names():
            return vitals.get_channel(name)
    return None


def _cc_periods(vitals, channel, compute) -> tuple:
    """Start and stop times of the chest compression periods (label
    ``cc_periods``) attached to the given channel; computed first (via
    ``compute``) if not present yet."""
    labels = [label for label in channel.labels if label.name == "cc_periods"]
    if not labels:
        compute()
        labels = [label for label in channel.labels if label.name == "cc_periods"]
    intervals = labels[-1].intervals
    return intervals[:, 0], intervals[:, 1]


def make_cprdat_analysis(vitals, model="") -> dict:
    """Automatically analyses a defibrillator recording and and returns a dictionary with all derived information.
    This functions is used in the German Resuscitation Registry to generate a case timeline as interactive visualisation.
    """

    gaussian_kernel_regression = np.vectorize(
        gaussian_kernel_regression_point, excluded={1, 2, 3}
    )

    CC_min_length = 10000

    cprdat_analysis = {}
    # ---------------RECORDING INFO --------------------------------------------------------------------

    metadata = vitals.channels[0].metadata
    recording_info = {}
    recording_info["filename"] = [
        _first_available(metadata, "File ID", "File Name")
        or _recording_file_name(vitals)
    ]
    recording_info["serial_number"] = _first_available(
        metadata, "Serial Nr", "Serial No"
    )
    recording_info["startdatetime"] = str(vitals.rec_start().isoformat())
    recording_info["endtime"] = (
        vitals.rec_stop() - vitals.rec_start()
    ).total_seconds() * 1000
    model = _first_available(metadata, "Model", "source") or model

    if "ZOLL" in model or "Zoll" in model:
        recording_info["DACTYP"] = "06"
        if "X Series" in model:
            recording_info["DACTYP2"] = "08"
        elif "E Series" in model:
            recording_info["DACTYP2"] = "07"
        elif "R Series" in model:
            recording_info["DACTYP2"] = "09"
        elif "AED Pro" in model:
            recording_info["DACTYP2"] = "06"
    elif "LIFEPAK15" in model or "LP15" in model:
        recording_info["DACTYP"] = "04"
        recording_info["DACTYP2"] = "10"

    elif "Corpuls" in model:
        recording_info["DACTYP"] = "03"
        recording_info["DACTYP2"] = None
    else:
        recording_info["DACTYP"] = "00"
    cprdat_analysis["recording_info"] = recording_info

    # ---------------CHEST COMPRESSION ANALYSIS -----------------------------------------

    chest_compression_analysis = []
    if "cc_depth" in vitals.get_channel_names():
        CC, CC_depth = _time_and_data(vitals.get_channel("cc_depth"))
    elif "CC" in vitals.get_channel_names():
        CC, CC_depth = _time_and_data(vitals.get_channel("cc"))
    else:
        CC = np.array([])
        CC_depth = np.array([])

    if "mech_cpr_depth" in vitals.get_channel_names():
        mech_CC_time, mech_CC_depth = _time_and_data(
            vitals.get_channel("mech_cpr_depth")
        )
        CC = np.append(CC, mech_CC_time)
        CC_depth = np.append(CC_depth, mech_CC_depth)

    CC = np.array([(c - vitals.rec_start()).total_seconds() * 1000 for c in CC])

    CC_final = np.asarray([], dtype=CC.dtype)
    CC_depth_final = np.asarray([])
    if "cpr_acceleration" in vitals.get_channel_names():
        CC_starts, CC_stops = _cc_periods(
            vitals,
            channel=vitals.get_channel("cpr_acceleration"),
            compute=vitals.find_CC_periods_acc,
        )
        CC_starts = np.array(
            [(c - vitals.rec_start()).total_seconds() * 1000 for c in CC_starts]
        )
        CC_stops = np.array(
            [(c - vitals.rec_start()).total_seconds() * 1000 for c in CC_stops]
        )

        for sta, sto in zip(CC_starts, CC_stops):
            CC_period_analyis = {"CC_start": sta, "CC_stop": sto}
            cond = (CC >= sta) & (CC < sto)
            CC_period = CC[cond]
            CC_final = np.append(CC_final, CC_period)
            if len(CC_period) > 1:
                freq = 1 / np.median(CC_period[1:] - CC_period[:-1])
                CC_period_analyis["CC_rate"] = freq * 1000 * 60
            else:
                CC_period_analyis["CC_rate"] = 0
            if len(CC_depth) > 0:
                CC_depth_period = CC_depth[cond]
                if len(CC_depth_period) > 0:
                    CC_period_analyis["CC_depth"] = np.median(CC_depth_period)
                    CC_depth_final = np.append(CC_depth_final, CC_depth_period)
                else:
                    CC_period_analyis["CC_depth"] = 0

            else:
                CC_period_analyis["CC_depth"] = None

            chest_compression_analysis.append(CC_period_analyis)

    else:
        if len(CC) > 0:
            cc_events_channel = _cc_events_channel(vitals)
            CC_starts, CC_stops = _cc_periods(
                vitals,
                channel=cc_events_channel,
                compute=lambda: vitals.cycle_duration_analysis(cc_events_channel),
            )
            CC_starts = np.array(
                [(c - vitals.rec_start()).total_seconds() * 1000 for c in CC_starts]
            )
            CC_stops = np.array(
                [(c - vitals.rec_start()).total_seconds() * 1000 for c in CC_stops]
            )
            for sta, sto in zip(CC_starts, CC_stops):
                CC_period_analyis = {"CC_start": sta, "CC_stop": sto}
                cond = (CC >= sta) & (CC < sto)
                CC_period = CC[cond]
                CC_final = np.append(CC_final, CC_period)
                if len(CC_period) > 1:
                    freq = 1 / np.median(CC_period[1:] - CC_period[:-1])
                    CC_period_analyis["CC_rate"] = freq * 60 * 1000
                else:
                    CC_period_analyis["CC_rate"] = 0
                if len(CC_depth) > 0:
                    CC_depth_period = CC_depth[cond]
                    if len(CC_depth_period) > 0:
                        CC_period_analyis["CC_depth"] = np.median(CC_depth_period)
                        CC_depth_final = np.append(CC_depth_final, CC_depth_period)
                    else:
                        CC_period_analyis["CC_depth"] = 0

                else:
                    CC_period_analyis["CC_depth"] = None

                chest_compression_analysis.append(CC_period_analyis)
        else:
            CC_starts = np.array([])
            CC_stops = np.array([])

    cprdat_analysis["chest_compression_analysis"] = chest_compression_analysis

    if "capnography" in vitals.get_channel_names():
        vitals.compute_etco2_and_ventilations(mode="threshold")
    if "ventilations_from_capnography" in vitals.get_label_names():
        ventilations, x = _time_and_data(
            vitals.get_label("ventilations_from_capnography")
        )
        ventilations = np.array(
            [(c - vitals.rec_start()).total_seconds() * 1000 for c in ventilations]
        )

    # ---------------MINUTE ANALYSIS --------------------------------------------------------------------

    cpr_per_minute_analysis = []
    t_stop = 0
    while t_stop < (vitals.rec_stop() - vitals.rec_start()).total_seconds() * 1000:
        t_start = t_stop
        t_stop += 60000
        minute_analysis = {"starttime": t_start}
        CC_cond = (CC_final >= t_start) & (CC_final < t_stop)
        CC_min = CC_final[CC_cond]
        if len(CC_min) > 5:
            minute_analysis["CC_rate"] = (
                60 / np.median((CC_min[1:] - CC_min[:-1])) * 1000
            )
        else:
            minute_analysis["CC_rate"] = None
        if len(CC_depth_final) > 0:
            CC_depth_min = CC_depth_final[CC_cond]
            if len(CC_min) > 5:
                minute_analysis["CC_depth"] = np.median(CC_depth_min)
            else:
                minute_analysis["CC_depth"] = None
        else:
            minute_analysis["CC_depth"] = None

        minute_analysis["CCF"] = CCF_minute(t_start, t_stop, CC_starts, CC_stops)

        if "ventilations_from_capnography" in vitals.get_label_names():
            ventilations_cond = (ventilations >= t_start) & (ventilations < t_stop)
            ventilations_min = np.array(ventilations[ventilations_cond])
            if len(ventilations_min) > 3:
                minute_analysis["Ventilation_rate"] = np.median(
                    60000 / (ventilations_min[1:] - ventilations_min[:-1])
                )
            else:
                minute_analysis["Ventilation_rate"] = None
        else:
            minute_analysis["Ventilation_rate"] = None
        cpr_per_minute_analysis.append(minute_analysis)

    cprdat_analysis["CPR_per_minute_analysis"] = cpr_per_minute_analysis

    # ---------------DEFIBRILLATIONS --------------------------------------------------------------------

    defibrillations = []

    for i in range(len(vitals.shocks())):
        shock = vitals.shocks().iloc[i]
        shocktime = (
            vitals.shocks().index[i] - vitals.rec_start()
        ).total_seconds() * 1000
        defib = {"time": shocktime}
        for register_key, cprdat_key in zip(
            ["energy_delivered", "energy_planned", "impedance"],
            ["DeliveredEnergy", "DefaultEnergy", "Impedance"],
        ):
            if cprdat_key in shock:
                defib[register_key] = shock[cprdat_key]
            else:
                defib[register_key] = None

        CC_stops_before = CC_stops[CC_stops < shocktime + 1000]
        CC_starts_after = CC_starts[CC_starts > shocktime - 1000]

        if len(CC_stops_before) > 0:
            if (defib["time"] - CC_stops_before[-1]) < 60000:
                defib["pre_shock_pause"] = np.max(
                    [defib["time"] - CC_stops_before[-1], 0]
                )
            else:
                defib["pre_shock_pause"] = None
        else:
            defib["pre_shock_pause"] = None

        if len(CC_starts_after) > 0:
            if -defib["time"] + CC_starts_after[0] < 60000:
                defib["post_shock_pause"] = np.max(
                    [-defib["time"] + CC_starts_after[0], 0]
                )
            else:
                defib["post_shock_pause"] = None
        else:
            defib["post_shock_pause"] = None
        defibrillations.append(defib)

    cprdat_analysis["defibrillations"] = defibrillations

    # ---------------REGISTRY TIMEPOINTS --------------------------------------------------------------------

    registry_time_points = {}

    if len(CC_starts) > 0:
        registry_time_points["ZHDM"] = CC_starts[0]
    else:
        registry_time_points["ZHDM"] = None
    registry_time_points["ZDEFIAN"] = 0
    if defibrillations:
        registry_time_points["ZDEFI1"] = defibrillations[0]["time"]
    else:
        registry_time_points["ZDEFI1"] = None
    intub_time = None
    if "CO2" in vitals.get_channel_names():
        co_channel = vitals.get_channel("CO2")
        cotime, co = _time_and_data(co_channel)
        for time, value in zip(cotime, co):
            if value != 0:
                intub_time = time
                break
        registry_time_points["ZINTUB"] = (
            intub_time - vitals.rec_start()
        ).total_seconds() * 1000
    else:
        registry_time_points["ZINTUB"] = None

    if {"cpr_acceleration", "ecg_pads"} <= set(vitals.get_channel_names()):
        vitals.predict_circulation()

    if len(CC_starts) > 0:
        if "rosc_probability" in vitals.get_label_names():
            rosctime, roscdata = _time_and_data(vitals.get_label("rosc_probability"))
            pred_time = np.array(
                [(c - vitals.rec_start()).total_seconds() * 1000 for c in rosctime]
            )

            roscs, arrests = find_ROSC_2(pred_time, roscdata, CC_starts, CC_stops)
            if len(roscs) > 0:
                registry_time_points["ZROSC1"] = roscs[0]
                registry_time_points["ZTOD"] = None
            else:
                ii = -1
                while (CC_stops[ii] - CC_starts[ii]) < CC_min_length:
                    ii -= 1
                registry_time_points["ZTOD"] = CC_stops[ii]
                registry_time_points["ZROSC1"] = None
        else:
            pauses = np.asarray(
                [sta - sto for sta, sto in zip(CC_starts[1:], CC_stops[:-1])]
            )
            if len(pauses) > 0:
                if np.max(pauses) < 60000:
                    if (recording_info["endtime"] - CC_stops[-1]) < 1200000:
                        ii = -1
                        while (CC_stops[ii] - CC_starts[ii]) < CC_min_length:
                            ii -= 1
                        registry_time_points["ZTOD"] = CC_stops[ii]
                        registry_time_points["ZROSC1"] = None
                    else:
                        ii = -1
                        while (CC_stops[ii] - CC_starts[ii]) < CC_min_length:
                            ii -= 1
                        registry_time_points["ZROSC1"] = CC_stops[ii]
                        registry_time_points["ZTOD"] = None
                else:
                    all_i = np.argwhere(pauses > 60000)
                    min_i = np.min(all_i)
                    registry_time_points["ZROSC1"] = CC_stops[min_i]
                    registry_time_points["ZTOD"] = None
            else:
                if len(CC_stops) > 0:
                    if (recording_info["endtime"] - CC_stops[-1]) < 1200000:
                        registry_time_points["ZTOD"] = CC_stops[-1]
                        registry_time_points["ZROSC1"] = None
                    else:
                        registry_time_points["ZROSC1"] = CC_stops[-1]
                        registry_time_points["ZTOD"] = None
                else:
                    registry_time_points["ZTOD"] = recording_info["endtime"]
                    registry_time_points["ZROSC1"] = None
    else:
        registry_time_points["ZTOD"] = None
        registry_time_points["ZROSC1"] = None

    cprdat_analysis["registry_time_points"] = registry_time_points

    # ---------------REGISTRY DETAILS --------------------------------------------------------------------

    registry_info = {}
    if "cpr_acceleration" in vitals.get_channel_or_label_names():
        registry_info["FBSYSTEM"] = "05"
        registry_info["TYPFBSYS"] = "02"
    elif "cc_depth" in vitals.get_channel_or_label_names():
        if "Corpuls" in model:
            registry_info["FBSYSTEM"] = "05"
            registry_info["TYPFBSYS"] = "06"
        else:
            registry_info["FBSYSTEM"] = "98"
            registry_info["TYPFBSYS"] = "98"

    else:
        registry_info["FBSYSTEM"] = "06"
        registry_info["TYPFBSYS"] = None

    if "autopulse_load" in vitals.get_channel_or_label_names():
        registry_info["AUTOCPR"] = "05"
        registry_info["TYPAUCPR"] = "01"
    elif "mech_cpr_depth" in vitals.get_channel_or_label_names():
        registry_info["AUTOCPR"] = "05"
        registry_info["TYPAUCPR"] = "05"
    else:
        registry_info["AUTOCPR"] = "06"
        registry_info["TYPAUCPR"] = None

    n_defi = len(defibrillations)
    if n_defi == 0:
        registry_info["ANZDEFI"] = "00"
    elif n_defi == 1:
        registry_info["ANZDEFI"] = "01"
    elif n_defi <= 3:
        registry_info["ANZDEFI"] = "02"
    elif n_defi <= 6:
        registry_info["ANZDEFI"] = "03"
    elif n_defi <= 9:
        registry_info["ANZDEFI"] = "04"
    elif n_defi >= 10:
        registry_info["ANZDEFI"] = "05"

    if registry_time_points["ZROSC1"] is None:
        registry_info["ROSC"] = "01"
    else:
        registry_info["ROSC"] = "02"
        registry_info["TECHSCHO"] = None
        shocks = vitals.shocks()
        if len(shocks) > 0:
            try:
                shocktime = (
                    vitals.shocks().index - vitals.rec_start()
                ).total_seconds() * 1000
                i_last_shock_before_rosc = np.argmax(
                    shocktime[shocktime < registry_time_points["ZROSC1"]]
                )
                if "DeliveredEnergy" in shocks.columns:
                    registry_info["ENERGIESCHO"] = int(
                        shocks.iloc[i_last_shock_before_rosc]["DeliveredEnergy"]
                    )
                elif "DefaultEnergy" in shocks.columns:
                    registry_info["ENERGIESCHO"] = int(
                        shocks.iloc[i_last_shock_before_rosc]["DefaultEnergy"]
                    )
                else:
                    registry_info["ENERGIESCHO"] = None
            except ValueError:
                registry_info["ENERGIESCHO"] = None
        else:
            registry_info["ENERGIESCHO"] = None

    cprdat_analysis["registry_info"] = registry_info

    # ---------------CHANNELS --------------------------------------------------------------------

    channels = []
    time_only_keys = ["cc", "ventilations_from_capnography", "time_12_lead_ecg"]
    sample_keys = [
        "cc_depth",
        "etco2_from_capnography",
        "ibp_1_sys",
        "ibp_1_dia",
        "ibp_1_map",
        "nibp_sys",
        "nibp_dia",
        "nibp_map",
    ]

    for key in vitals.get_channel_or_label_names():
        keychannel = vitals.get_channel_or_label(key)
        keytime, keydata = _time_and_data(keychannel)
        if isinstance(keychannel, IntervalLabel):
            # time span from the start of the first to the end of the last interval
            keytime = [keytime[0][0], keytime[-1][1]] if len(keytime) > 0 else []
        if key in time_only_keys:
            chan = {
                "name": key,
                "type": "timeonly",
                "data": [
                    (c - vitals.rec_start()).total_seconds() * 1000 for c in keytime
                ],
            }
            channels.append(chan)
        elif key in sample_keys:
            keydata = list(map(int, keydata))
            if key == "cc_depth":
                df = pd.DataFrame(CC_depth_final)
                df.set_index(CC_final, inplace=True)
                chan = {
                    "name": key,
                    "type": "sample",
                    "unit": "cm",
                    "data": convert_two_alternating_list(df),
                }
                channels.append(chan)
            elif key == "etco2_from_capnography":
                keytime = [
                    (c - vitals.rec_start()).total_seconds() * 1000 for c in keytime
                ]

                df = pd.DataFrame(keydata, index=keytime)
                chan = {
                    "name": key,
                    "type": "sample",
                    "unit": "mmHg",  # keychannel.metadata['Unit'],
                    "data": convert_two_alternating_list(df),
                }
                channels.append(chan)
            else:  # Blood - Pressure keys
                keytime = [
                    (c - vitals.rec_start()).total_seconds() * 1000 for c in keytime
                ]
                df = pd.DataFrame(keydata, index=keytime)
                chan = {
                    "name": key,
                    "type": "sample",
                    "unit": "mmHg",  # keychannel.metadata['Unit'],
                    "data": convert_two_alternating_list(df),
                }
                channels.append(chan)

        elif key not in np.append(time_only_keys, sample_keys):
            chan = {
                "name": key,
                "type": "startendtime",
                "starttime": (keytime[0] - vitals.rec_start()).total_seconds() * 1000,
                "endtime": (keytime[-1] - vitals.rec_start()).total_seconds() * 1000,
            }
            channels.append(chan)

    # compute necessary rates for plotting
    all_channel_names = [chan["name"] for chan in channels]
    if "ventilations_from_capnography" in all_channel_names:
        ventchannel = vitals.get_channel_or_label("ventilations_from_capnography")
        venttime, ventdata = _time_and_data(ventchannel)
        venttime = [(c - vitals.rec_start()).total_seconds() * 1000 for c in venttime]
        vent_rate = [60000 / (b - a) for a, b in zip(venttime[:-1], venttime[1:])]
        df = pd.DataFrame(vent_rate, index=venttime[1:])
        chan = {
            "name": "ventilation_rate_for_plotting",
            "type": "sample",
            "unit": "1/min",  # keychannel.metadata['Unit'],
            "data": convert_two_alternating_list(df),
        }
        channels.append(chan)

    if "etco2_from_capnography" in all_channel_names:
        etco2_channel = vitals.get_channel_or_label("etco2_from_capnography")
        etco2_time, etco2_data = _time_and_data(etco2_channel)
        etco2_time = [
            (c - vitals.rec_start()).total_seconds() * 1000 for c in etco2_time
        ]
        filtered_etco2 = gaussian_kernel_regression(
            np.asarray(etco2_time),
            np.asarray(etco2_time),
            np.asarray(etco2_data),
            10000,
        )
        df = pd.DataFrame(filtered_etco2, index=etco2_time)
        chan = {
            "name": "gaussian_filtered_etco2",
            "type": "sample",
            "unit": "mmHg",  # keychannel.metadata['Unit'],
            "data": convert_two_alternating_list(df),
        }
        channels.append(chan)

    if "cc" in all_channel_names:
        CC_channel = vitals.get_channel_or_label("cc")
        CCtime, CCdata = _time_and_data(CC_channel)
    elif "cc_depth" in all_channel_names:
        CC_channel = vitals.get_channel_or_label("cc_depth")
        CCtime, CCdata = _time_and_data(CC_channel)
    else:
        CCtime = []
    if len(CCtime) > 0:
        CCtime = [(c - vitals.rec_start()).total_seconds() * 1000 for c in CCtime]
        CC_rate = [60000 / (b - a) for a, b in zip(CCtime[:-1], CCtime[1:])]
        df = pd.DataFrame(CC_rate, index=CCtime[1:])
        chan = {
            "name": "cc_rate_for_plotting",
            "type": "sample",
            "unit": "1/min",  # keychannel.metadata['Unit'],
            "data": convert_two_alternating_list(df),
        }
        channels.append(chan)

    cprdat_analysis["channels"] = channels

    # ---------------CPR QUALITY ANALYSIS --------------------------------------------------------------------
    zrosc = cprdat_analysis["registry_time_points"]["ZROSC1"]
    zhdm = cprdat_analysis["registry_time_points"]["ZHDM"]
    ztod = cprdat_analysis["registry_time_points"]["ZTOD"]

    # TODO: investigate unused variable
    # starttime = pd.Timestamp(cprdat_analysis["recording_info"]["startdatetime"])

    if zrosc:
        z_end = zrosc
    else:
        z_end = ztod

    CC_starts = []
    CC_stops = []
    for elem in cprdat_analysis["chest_compression_analysis"]:
        CC_starts.append(elem["CC_start"])
        CC_stops.append(elem["CC_stop"])
    CC_start_analyze = [
        C for C in CC_starts if (C <= z_end) and (C >= zhdm)
    ]  # CC-intervals in first resuscitation episode
    CC_stops_analyze = [C for C in CC_stops if (C <= z_end) and (C >= zhdm)]
    if len(CC_start_analyze) > 0:
        if CC_start_analyze[0] > CC_stops_analyze[0]:
            CC_start_analyze.insert(0, zhdm)
        if CC_stops_analyze[-1] > CC_stops_analyze[-1]:
            CC_stops_analyze.append(z_end)
        CCF_glob = (np.sum(CC_stops_analyze) - np.sum(CC_start_analyze)) / (
            z_end - zhdm
        )
        pauses = [
            sta - sto for sta, sto in zip(CC_start_analyze[1:], CC_stops_analyze[:-1])
        ]
    else:
        CCF_glob = None
        pauses = []

    pre_shock = []
    post_shock = []
    for shock in cprdat_analysis["defibrillations"]:
        if shock["pre_shock_pause"]:
            pre_shock.append(shock["pre_shock_pause"])
        if shock["post_shock_pause"]:
            post_shock.append(shock["post_shock_pause"])

    cpr_quality_analysis = {}
    cpr_quality_analysis["global_ccf"] = CCF_glob
    if pauses:
        cpr_quality_analysis["pauses_median"] = np.median(pauses)
        cpr_quality_analysis["pauses_maximum"] = np.max(pauses)
    else:
        cpr_quality_analysis["pauses_median"] = None
        cpr_quality_analysis["pauses_maximum"] = None
    if pre_shock:
        cpr_quality_analysis["preshock_median"] = np.median(pre_shock)
        cpr_quality_analysis["preshock_maximum"] = np.max(pre_shock)
    else:
        cpr_quality_analysis["preshock_median"] = None
        cpr_quality_analysis["preshock_maximum"] = None
    if post_shock:
        cpr_quality_analysis["postshock_median"] = np.median(post_shock)
        cpr_quality_analysis["postshock_maximum"] = np.max(post_shock)
    else:
        cpr_quality_analysis["postshock_median"] = None
        cpr_quality_analysis["postshock_maximum"] = None

    cprdat_analysis["cpr_quality_analysis"] = cpr_quality_analysis
    return cprdat_analysis

    # -------------- CIRCULATORY STATE --------------------------------
    arrest_episodes = {"arrests": [], "roscs": []}
    CC_starts = []
    CC_stops = []
    for elem in cprdat_analysis["chest_compression_analysis"]:
        CC_starts.append(elem["CC_start"])
        CC_stops.append(elem["CC_stop"])
    if (
        "rosc_probability" in vitals.get_label_names()
    ):  # USE Accelerometry-Algorithm for ROSC detection
        rosctime, roscdata = _time_and_data(vitals.get_label("rosc_probability"))
        pred_time = np.array(
            [(c - vitals.rec_start()).total_seconds() * 1000 for c in rosctime]
        )

        roscs, arrests = find_ROSC_2(pred_time, roscdata, CC_starts, CC_stops)

        for ro in roscs:
            arrest_episodes["roscs"].append(ro)
        for ar in arrests:
            arrest_episodes["arrests"].append(ar)
        if len(arrests) > len(roscs):
            i = len(CC_stops) - 1
            while CC_stops[i] - CC_starts[i] < CC_min_length:
                i -= 1
            arrest_episodes["termination"] = CC_stops[i]
        else:
            arrest_episodes["termination"] = None

    else:
        arrest_episodes["arrests"].append(
            CC_starts[0]
        )  # Detect ROSCS and arrests directly on length of CC interruptions
        pauses = [sta - sto for sta, sto in zip(CC_starts[1:], CC_stops[:-1])]
        CC_lengths = [sto - sta for sta, sto in zip(CC_starts, CC_stops)]
        i = 0
        while i < len(pauses):
            pause = pauses[i]
            if pause < 120000:
                i += 1
            else:
                arrest_episodes["roscs"].append(CC_stops[i])
                j = 1
                while CC_lengths[i + j] < CC_min_length and i + j < len(CC_lengths) - 1:
                    j += 1
                if i + j < len(CC_lengths) - 1:
                    arrest_episodes["arrests"].append(CC_starts[i + j])

                else:
                    i = i + j
        if (
            arrest_episodes["roscs"][-1] > arrest_episodes["arrests"][-1]
        ):  # Create Termination marker i nthe end, if necessary
            arrest_episodes["termination"] = arrest_episodes["roscs"][-1]
            arrest_episodes["roscs"].pop(-1)
        else:
            i = len(CC_stops) - 1
            while CC_stops[i] - CC_starts[i] < CC_min_length:
                i -= 1
            arrest_episodes["termination"] = CC_stops[i]
            # DECIDE whether ROSC or Arrest

    cprdat_analysis["arrest_episodes"] = arrest_episodes

    version = "1.5.2"
    schema_version = "0.6.1"
    # --------------- TOTAL --------------------------------------------------------------------

    result_dict = {
        "cprdat_analysis": cprdat_analysis,
        "analyzationtime": pd.Timestamp.now().isoformat(),
        "schemaversion": schema_version,
        "cprdatversion": version,
        "result_code": "00",
    }

    return result_dict
