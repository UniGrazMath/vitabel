"""Loading of Stryker LIFEPAK 15 recordings (CodeStat XML export)."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import pandas as pd

from vitabel.utils.constants import (
    LP15_all_trend_data,
    LP15_collected_trend_data,
    LP_Unit_dict,
)

logger = logging.getLogger("vitabel")


def _find(element, tag):
    """Find ``tag`` among the children of ``element``, else anywhere below it.

    ``{*}`` matches the tag in any XML namespace or in none, so files with
    and without the CODE-STAT namespace are handled the same way.
    """
    found = element.find("{*}" + tag)
    if found is None:
        found = element.find(".//{*}" + tag)
    return found


def _defib_values(event, pcstr="{*}") -> tuple[float, float]:
    """Delivered energy and impedance of a ``Defib`` event, NaN if missing.

    One value per shock, so that the values stay aligned with the shock
    times also if a value is missing for some shocks.
    """
    energy, impedance = np.nan, np.nan
    values = event.find(pcstr + "Values")
    for value in values.findall(pcstr + "Value") if values is not None else []:
        if value.attrib["Type"] == "DefibEnergy" and value.text:
            energy = float(value.text)
        elif value.attrib["Type"] == "DefibVoltageCompImpedance" and value.text:
            impedance = float(value.text)
    return energy, impedance


def read_lifepak(f_cont, f_cont_wv, f_cpre, further_files=[], f_trend=None):
    pure_filename = f_cont.stem[: f_cont.stem.rindex("_")]

    tree = ET.parse(f_cont)  # LOAD XML FILES
    root = tree.getroot()

    tree1 = ET.parse(f_cont_wv)
    root1 = tree1.getroot()
    Compr_Flag = True
    try:
        tree2 = ET.parse(f_cpre)
        root2 = tree2.getroot()
    except (FileNotFoundError, ET.ParseError):
        logger.warning(
            f"No CprEventLog.xml found for file {pure_filename}. "
            "Chest_Compression Data is not available."
        )
        Compr_Flag = False
    # tree3 = ET.parse(f_cpr)
    # root3 = tree3.getroot()
    pcstr = "{*}"  # any namespace or none (CODE-STAT 12 uses a namespace)
    for event in _find(root, "Events"):  # LOAD MAIN DATA
        if event.attrib["Type"] == "PowerOn":
            starttime = pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
    device_container = _find(root, "Device")
    serial = device_container.find(pcstr + "SerialNumber").text
    model = device_container.find(pcstr + "Model").text  # should be LP15

    # Make one reading function which loads for fielnames and pcstr, try different pctry via try except
    rec_start = ""
    rec_stop = ""

    channel_info = {}  # LOAD WAVEFORM DATA From Channels
    channel_data = {}
    datas = _find(root1, "Record").findall(pcstr + "RecordData")
    for record in datas:
        cha_name = lifepak2zoll_name(record.find(pcstr + "Channel").text)
        cha_unit = (
            record.find(pcstr + "Waveforms").find(pcstr + "YValues").attrib["unit"]
        )
        if cha_name not in channel_info:
            channel_info[cha_name] = {}
            channel_info[cha_name]["Unit"] = cha_unit
        if cha_name not in channel_data:
            channel_data[cha_name] = {
                "timestamp": np.array([], dtype="datetime64[ns]"),
                cha_name: np.array([], dtype=float),
            }
        snippets = record.findall(pcstr + "Waveforms")
        for snip in snippets:
            xoff = pd.Timedelta(
                snip.find(pcstr + "XValues").find(pcstr + "XOffset").text
            )
            startsnip = starttime + xoff
            sample_rate = snip.find(pcstr + "XValues").find(pcstr + "SampleRate")
            if sample_rate.attrib["unit"] == "Hz":
                sample_rate = float(sample_rate.text)
            else:
                logger.warning(
                    f"Unknown unit {sample_rate.attrib['unit']} in sample rate."
                )
                sample_rate = float(sample_rate.text)
            channel_info[cha_name]["Sample Rate"] = sample_rate
            n_samples = int(
                float(snip.find(pcstr + "XValues").find(pcstr + "NumberOfSamples").text)
            )
            times = np.array(
                pd.date_range(
                    start=startsnip,
                    freq=str(int(1000000000 / sample_rate)) + "ns",
                    periods=n_samples,
                )
            )
            if rec_start == "":
                rec_start = times[0]
                rec_stop = times[-1]
            else:
                if times[0] < rec_start:
                    rec_start = times[0]
                if times[-1] > rec_stop:
                    rec_stop = times[-1]
            ydat = snip.find(pcstr + "YValues").find(pcstr + "RealValue")
            startdif = (
                pd.to_datetime(
                    pd.Timestamp(
                        str(pd.Timestamp(times[0]).date())
                        + " "
                        + ydat.find(pcstr + "From").text
                    )
                )
                - pd.to_datetime(times[0])
            ).total_seconds()
            enddif = (
                pd.to_datetime(
                    pd.Timestamp(
                        str(pd.Timestamp(times[-1]).date())
                        + " "
                        + ydat.find(pcstr + "To").text
                    )
                )
                - pd.to_datetime(times[-1])
            ).total_seconds()
            if np.abs(startdif) > 0.1 or np.abs(enddif) > 0.1:
                logger.info(f"Startdif: {startdif}, Enddif: {enddif}")
            # values are stored as device units, the scale converts them
            # to the unit given in YValues (CODE-STAT 11 exports use 1)
            scale = float(ydat.find(pcstr + "Scale").text)
            if ydat.find(pcstr + "Data").text is not None:
                ydata = np.array(ydat.find(pcstr + "Data").text.split(","))
                del_ind = np.where(ydata == "")
                ydata = np.delete(ydata, del_ind)
                times = np.delete(times, del_ind)
                try:
                    ydata = ydata.astype(np.float64)
                except ValueError:
                    del_ind = np.array([], int)
                    for i in range(len(ydata)):
                        st = ydata[i]
                        try:
                            ydata[i] = float(st)
                        except ValueError:
                            del_ind = np.append(del_ind, i)
                    ydata = np.delete(ydata, del_ind)
                    times = np.delete(times, del_ind)
                ydata = ydata.astype(np.float64) * scale

                channel_data[cha_name]["timestamp"] = np.append(
                    channel_data[cha_name]["timestamp"], times
                )
                channel_data[cha_name][cha_name] = np.append(
                    channel_data[cha_name][cha_name], ydata
                )

    del_keys = []  # Delete empty channels
    for key in channel_data:
        if len(channel_data[key][key]) > 0:
            channel_data[key] = pd.DataFrame.from_dict(channel_data[key])
            channel_data[key].set_index("timestamp", inplace=True)
        else:
            del_keys.append(key)
    for key in del_keys:
        channel_data.pop(key)
        channel_info.pop(key)

    if Compr_Flag:  # Add Compressions, Ventilations and shocks
        compr = np.array([])
        vent = np.array([])
        startcpr = np.array([])
        stopcpr = np.array([])
        defibtime = np.array([])
        defib_en = np.array([])
        defib_imp = np.array([])
        twelve_lead_ecg_time = np.array([])

        trend_data = {}
        # local copy: the list of known trend types must not be modified globally
        known_trend_data = list(LP15_all_trend_data)

        events = _find(root2, "Events")

        for event in events:
            if event.attrib["Type"] == "ChestCompression":
                compr = np.append(
                    compr, pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
                )
            elif event.attrib["Type"] == "Ventilation":
                vent = np.append(
                    vent, pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
                )
            elif event.attrib["Type"] == "StartCPR":
                startcpr = np.append(
                    startcpr, pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
                )
            elif event.attrib["Type"] == "StopCPR":
                stopcpr = np.append(
                    stopcpr, pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
                )
            elif event.attrib["Type"] == "Defib":
                defibtime = np.append(
                    defibtime, pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
                )
                energy, impedance = _defib_values(event, pcstr)
                defib_en = np.append(defib_en, energy)
                defib_imp = np.append(defib_imp, impedance)
            elif event.attrib["Type"] == "12Lead":
                twelve_lead_ecg_time = np.append(
                    twelve_lead_ecg_time,
                    pd.Timestamp(event.find(pcstr + "AdjustedTime").text),
                )
            else:  # Other Events might contain trend info as Heart rate, blood pressures and so on
                time = pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
                if event.find(pcstr + "Values").findall(pcstr + "Value"):
                    for value in event.find(pcstr + "Values").findall(pcstr + "Value"):
                        channel = value.attrib["Type"]
                        if channel not in known_trend_data:
                            logger.warning(f"New trend data type found: {channel}")
                            known_trend_data.append(channel)
                        elif channel in LP15_collected_trend_data:
                            if channel not in trend_data:
                                trend_data[channel] = {"timestamp": [], channel: []}
                            if value.text:
                                trend_data[channel]["timestamp"].append(time)
                                trend_data[channel][channel].append(value.text)

        if f_trend is not None:
            # _TrendData.xml has the same event structure, but many more
            # trend values. Use it instead of the trend values from the
            # CprEventLog, so values are not counted twice.
            trend_data = {}
            seen = {}  # channel -> times already read
            for event in _find(ET.parse(f_trend).getroot(), "Events"):
                time = pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
                values = event.find(pcstr + "Values")
                for value in (
                    values.findall(pcstr + "Value") if values is not None else []
                ):
                    channel = value.attrib["Type"]
                    if channel in LP15_collected_trend_data and value.text:
                        if channel not in trend_data:
                            trend_data[channel] = {"timestamp": [], channel: []}
                            seen[channel] = set()
                        if time in seen[channel]:
                            continue  # same value is listed in several events
                        seen[channel].add(time)
                        trend_data[channel]["timestamp"].append(time)
                        trend_data[channel][channel].append(value.text)

        del_keys = []

        for channel in trend_data:  ### Further CHECK
            if channel + "Status" in trend_data:
                times = []
                values = []
                for t, v in zip(
                    trend_data[channel]["timestamp"], trend_data[channel][channel]
                ):
                    if t in trend_data[channel + "Status"]["timestamp"]:
                        i = trend_data[channel + "Status"]["timestamp"].index(t)
                        if trend_data[channel + "Status"][channel + "Status"][i] == "0":
                            times.append(t)
                            values.append(v)

                trend_data[channel]["timestamp"] = times
                trend_data[channel][channel] = values
                del_keys.append(channel + "Status")
        # a status without values of its own channel is not a measurement
        del_keys += [
            k for k in trend_data if k.endswith("Status") and k not in del_keys
        ]

    for key in del_keys:
        trend_data.pop(key)

    n_defib = int(_find(root, "TotalShocks").text)  # number of defibrillations

    if Compr_Flag:  # Summarize compressions etc in dict
        compr_info = {
            "Compressions": pd.Series(compr),
            "Ventilations": pd.Series(vent),
            "StartCPR": pd.Series(startcpr),
            "StopCPR": pd.Series(stopcpr),
            "time_12_lead_ecg": pd.Series(twelve_lead_ecg_time),
        }

        if len(defibtime) != n_defib:
            logger.error(
                f"Number of defibrillations does not match: {len(defibtime)} != {n_defib}"
            )
        else:
            defib = {
                "timestamp": defibtime,
                "DeliveredEnergy": defib_en,
                "Impedance": defib_imp,
            }
            defib = pd.DataFrame.from_dict(defib)
            defib.set_index("timestamp", inplace=True)
            defib_data = {"defibrillations": defib}

        del_keys = []
        for key in compr_info:
            if len(compr_info[key]) == 0:
                del_keys.append(key)
        for key in del_keys:
            compr_info.pop(key)

        del_keys = []  # Delete empty channels
        for key in trend_data:
            if len(trend_data[key][key]) > 0:
                trend_data[key] = pd.DataFrame.from_dict(trend_data[key])
                trend_data[key].set_index("timestamp", inplace=True)
                trend_data[key] = trend_data[key].astype(float)
            else:
                del_keys.append(key)
        for key in del_keys:
            trend_data.pop(key)

    else:
        compr_info = {}
        defib_data = {}
        trend_data = {}

    for key in compr_info:  # Provide Backgound information
        channel_info[key] = {}

    for key in trend_data:
        channel_info[key] = {}

    for key in channel_data:
        channel_info[key]["Type"] = "Continous wave data"
        channel_info[key]["Start"] = channel_data[key].first_valid_index()
        channel_info[key]["Stop"] = channel_data[key].last_valid_index()
        channel_info[key]["Length"] = channel_data[key].size
    for key in trend_data:
        channel_info[key]["Type"] = "Trend data"
        channel_info[key]["Start"] = trend_data[key].first_valid_index()
        channel_info[key]["Stop"] = trend_data[key].last_valid_index()
        channel_info[key]["Length"] = trend_data[key].size
        if key in LP_Unit_dict:
            channel_info[key]["Unit"] = LP_Unit_dict[key]
        else:
            channel_info[key]["Unit"] = "Unknown"

    for key in compr_info:
        channel_info[key]["Type"] = "Event data"
        channel_info[key]["Sample Rate"] = "None"
        channel_info[key]["Unit"] = "Timestamp"
        if not compr_info[key].empty:
            channel_info[key]["Start"] = compr_info[key].iloc[0]
            channel_info[key]["Stop"] = compr_info[key].iloc[-1]
            channel_info[key]["Length"] = compr_info[key].size
        else:
            channel_info[key]["Start"] = None
            channel_info[key]["Stop"] = None
            channel_info[key]["Length"] = None
    # Load annotations:

    case_info = {
        "File ID": pure_filename,
        "Serial No": serial,
        "Model": model,
        "Start time": starttime,
        "Recording start": pd.Timestamp(rec_start),
        "Recording end": pd.Timestamp(rec_stop),
        "Recording Length": pd.Timestamp(rec_stop) - pd.Timestamp(rec_start),
    }

    case_info = pd.DataFrame(case_info, index=[0]).T
    channel_info = pd.DataFrame.from_dict(channel_info, orient="index")
    channel_info.reset_index(inplace=True)
    channel_info.rename(columns={"index": "Key"}, inplace=True)
    pat_dat = {"Main data": case_info, "Keys": channel_info, "Load Log": ""}
    data = {
        **channel_data,
        **trend_data,
        **compr_info,
        **defib_data,
    }
    return pat_dat, data


def read_cpr_event_log(f_cpre: Path):
    """Load a standalone ``*_CprEventLog.xml`` (CodeStat export of a CPR
    feedback device such as TrueCPR, without waveform data).

    The ``DeviceCompression`` events give the compressions (``Compressions``,
    renamed to ``cc``) and their depth and rate (``cc_depth`` in cm,
    ``cc_rate`` in 1/min, as for ZOLL recordings). Compressions marked as
    deleted by the user are skipped.

    Returns
    -------
    A tuple ``(pat_dat, data)`` as for :func:`read_lifepak`, or ``({}, {})``
    if the file contains no compressions.
    """
    f_cpre = Path(f_cpre)
    pcstr = "{*}"
    root = ET.parse(f_cpre).getroot()

    compressions, depths, rates = [], [], []
    for event in _find(root, "Events"):
        if event.attrib.get("Type") != "DeviceCompression":
            continue
        values = event.find(pcstr + "Values")
        values = {
            value.attrib["Type"]: value.text
            for value in (values.findall(pcstr + "Value") if values is not None else [])
        }
        if values.get("DeletedByUser") == "Yes":
            continue
        compressions.append(pd.Timestamp(event.find(pcstr + "AdjustedTime").text))
        depths.append(float(values["Depth"]) if values.get("Depth") else np.nan)
        rates.append(float(values["Rate"]) if values.get("Rate") else np.nan)

    if not compressions:
        logger.warning(f"No compressions found in {f_cpre.name}.")
        return {}, {}

    compressions = pd.DatetimeIndex(compressions)
    device = _find(root, "RecordingDevice")
    power_on = _find(root, "AdjustedPowerOn")
    case_info = {
        "File ID": f_cpre.name.removesuffix("_CprEventLog.xml"),
        "Serial No": _text(device, "SerialNumber"),
        "Model": _text(device, "Model"),
        "Start time": pd.Timestamp(power_on.text) if power_on is not None else pd.NaT,
        "Recording start": compressions[0],
        "Recording end": compressions[-1],
        "Recording Length": compressions[-1] - compressions[0],
    }
    channel_info = pd.DataFrame(
        {
            "Key": ["Compressions", "cc_depth", "cc_rate"],
            "Type": ["Event data", "Compression data", "Compression data"],
            "Unit": ["Timestamp", "cm", "1/min"],
            "Start": compressions[0],
            "Stop": compressions[-1],
            "Length": len(compressions),
        }
    )
    pat_dat = {
        "Main data": pd.DataFrame(case_info, index=[0]).T,
        "Keys": channel_info,
        "Load Log": "",
    }
    data = {
        "Compressions": pd.Series(compressions),
        # depth is exported in mm
        "cc_depth": pd.DataFrame(
            {"cc_depth": np.array(depths) / 10}, index=compressions
        ),
        "cc_rate": pd.DataFrame({"cc_rate": rates}, index=compressions),
    }
    return pat_dat, data


def _text(element, tag):
    """Text of the child ``tag`` of ``element`` (any namespace), or ``None``."""
    if element is None:
        return None
    child = element.find("{*}" + tag)
    return child.text if child is not None else None


def lifepak2zoll_name(lp):
    # if lp == "II":
    #     return "12-Lead II"
    # elif lp == "CO2":
    #     return "CO2 mmHg, Waveform"
    # elif lp == "SpO2":
    #     return "SpO2 %, Waveform"
    # elif lp == "Paddles (Generic)":
    #     return "Pads"
    # else:
    return lp
