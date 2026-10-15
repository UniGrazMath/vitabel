"""Loading of Stryker LUCAS recordings (CodeStat XML export)."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("vitabel")


def read_lucas(f_luc: Path, f_cpre: Path):
    pcstr = ""
    pure_filename = f_luc.name.removesuffix("_Lucas.xml")

    tree = ET.parse(f_luc)
    root = tree.getroot()

    tree2 = ET.parse(f_cpre)
    root2 = tree2.getroot()

    compr = np.array([])
    defibtime = np.array([])
    defib_en = np.array([])
    defib_imp = np.array([])

    event_container = root.find("Events")
    events = event_container.findall("Event")
    rec_start = 0
    rec_stop = 0
    for event in events:
        if event.attrib["Type"] == "PowerOn":
            starttime = pd.Timestamp(event.find("AdjustedTime").text)
        elif event.attrib["Type"] == "FirstCompression":
            rec_start = pd.Timestamp(event.find("AdjustedTime").text)
        elif event.attrib["Type"] == "LastCompression":
            rec_stop = pd.Timestamp(event.find("AdjustedTime").text)
    device_container = root.find("Device")
    serial = device_container.find("SerialNumber").text
    model = device_container.find("DeviceDescription").text.split("-")[
        0
    ]  # should be Lucas3

    event_container = root2.find("Events")
    events = event_container.findall("Event")
    for event in events:
        if event.attrib["Type"] == "DeviceCompression":
            compr = np.append(
                compr, pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
            )
        elif event.attrib["Type"] == "Defib":
            defibtime = np.append(
                defibtime, pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
            )
            for defib in event.find(pcstr + "Values").findall(pcstr + "Value"):
                if defib.attrib["Type"] == "DefibEnergy":
                    defib_en = np.append(defib_en, int(defib.text))
                if defib.attrib["Type"] == "DefibVoltageCompImpedance":
                    defib_imp = np.append(defib_imp, int(defib.text))

    channel_info = {}
    proc_dat = {}
    # Compute Acceleration

    n_defib = int(root.find(pcstr + "TotalShocks").text)
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
    compr_info = {"Compressions": pd.Series(compr)}
    for key in compr_info:
        channel_info[key] = {}

    for key in proc_dat:
        channel_info[key] = {}

    for key in compr_info:
        channel_info[key]["Type"] = "Compression data"
        channel_info[key]["Sample Rate"] = "None"
        channel_info[key]["Unit"] = "Number"
        if not compr_info[key].empty:
            channel_info[key]["Start"] = compr_info[key].iloc[0]
            channel_info[key]["Stop"] = compr_info[key].iloc[-1]
            channel_info[key]["Length"] = compr_info[key].size
        else:
            channel_info[key]["Start"] = None
            channel_info[key]["Stop"] = None
            channel_info[key]["Length"] = None

    case_info = {
        "File ID": pure_filename,
        "Serial No": serial,
        "Model": model,
        "Start time": starttime,
        "Recording start": pd.Timestamp(rec_start),
        "Recording end": pd.Timestamp(rec_stop),
        "Recording Length": pd.Timedelta(rec_stop - rec_start),
    }
    case_info = pd.DataFrame(case_info, index=[0]).T
    channel_info = pd.DataFrame.from_dict(channel_info, orient="index")
    channel_info.reset_index(inplace=True)
    channel_info.rename(columns={"index": "Key"}, inplace=True)
    pat_dat = {"Main data": case_info, "Keys": channel_info, "Load Log": ""}
    data = {**compr_info, **defib_data}
    if len(compr) > 0:
        return pat_dat, data
    else:
        return {}, {}
