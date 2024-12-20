import pandas as pd
import json
import os
import numpy as np
from scipy.stats import mode
import xml.etree.ElementTree as ET
import pyedflib
import sqlite3
import logging

from pathlib import Path

from vitabel.utils import (
    LP_Unit_dict,
    zoll_wavetype_dict,
    LP15_all_trend_data,
    LP15_collected_trend_data,
)


logger = logging.getLogger("vitabel")


def read_zolljson(filepath: Path | str):
    # Read File
    if isinstance(filepath, str):
        filepath = Path(filepath)
    with open(filepath, "r") as fd:
        data = json.load(fd)

    filename = filepath.name

    # Load internal structure
    zoll_record = data["ZOLL"]["FullDisclosure"][0]["FullDisclosureRecord"]

    # List with all known keys, to search for new content of a file
    all_keys = [
        "NewCase",
        "Ecg12LeadRec",
        "AnnotationEvt",
        "TrendRpt",
        "DeviceConfiguration",
        "DataChannels",
        "PatientInfo",
        "AlarmLimits",
        "TraceConfigs",
        "DeleteCaseRec",
        "Aed",
        "DisplayInfo",
        "ContinWaveRec",
        "CprAccelWaveRec",
        "CprCompression",
        "AedContinAnalysis",
        "InstrumentPacket",
        "AlarmEvt",
        "SnapshotRpt",
        "SysLogEntry",
        "Ecg12LeadDisc",
        "ReadinessLogRecord",
        "DefibTrace",
        "PrtTrace",
        "CaseSummary",
        "AedSingleAnalysis",
        "DefibFireEvt",
        "AplsLoadSignalRecord",
        "WebConsoleConfigParameter",
        "SelfTestFailure",
        "PaceEvt",
    ]

    # define containers for different data types
    single_wave_record = ["CprAccelWaveRec", "AplsLoadSignalRecord"]
    multi_waves_record = ["ContinWaveRec", "SnapshotRpt"]
    wave_recording_elements = []
    trend_rpt = []
    cpr_compr = []
    new_case = []
    dev_conf = []
    defib_fire = []
    twelve_lead_ecg = []

    # sort different data types
    for item in zoll_record:
        for key in item:
            if key in single_wave_record:
                wave_recording_elements.append(item[key])
            elif key in multi_waves_record:
                header = item[key]["StdHdr"]
                # Der übergeordnete Standardheader wird für jedes WaveRec Element gesetzt
                for item2 in item[key]["Waveform"]:
                    item2["StdHdr"] = header
                    wave_recording_elements.append(item2)
            elif key == "TrendRpt":
                trend_rpt.append(item[key])
            elif key == "CprCompression":
                cpr_compr.append(item[key])
            elif key == "NewCase":
                new_case.append(item[key])
            elif key == "DeviceConfiguration":
                dev_conf.append(item[key])
            elif key == "DefibFireEvt":
                defib_fire.append(item[key])
            elif key == "Ecg12LeadRec":
                twelve_lead_ecg.append(item[key])
            # If a unknown key is found
            if key not in all_keys:
                logger.info(
                    f"New key found: {key}. Check structure of values: {item[key]}"
                )
                all_keys.append(key)

    # Find Starttime of this Case
    starttime = pd.to_datetime(new_case[0]["StdHdr"]["DevDateTime"]) - pd.Timedelta(
        milliseconds=new_case[0]["StdHdr"]["MsecTime"]
    )
    for case in new_case:
        if not starttime == pd.to_datetime(
            case["StdHdr"]["DevDateTime"]
        ) - pd.Timedelta(milliseconds=case["StdHdr"]["MsecTime"]):
            logger.warning(
                f"Another NewCase found, starting time: {case['StdHdr']['DevDateTime']}"
            )

    # Call routines for different data types
    # contin_data=self.read_zolljson_contin(wave_recording_elements,starttime)
    # return contin_data

    (
        contin_data,
        contin_info,
        rec_start,
        rec_stop,
        inv_ch,
    ) = read_zolljson_contin(wave_recording_elements, starttime)
    contin_info = contin_info.loc["UnitsVal"].to_dict()

    contin_info["SpO2 %, Waveform"] = "1"

    trend_data, trend_info = read_zolljson_trend(trend_rpt, starttime)
    compr_data, compr_info = read_zolljson_compr(cpr_compr, starttime)

    defib = {
        "defibrillations": {
            "timestamp": [],
            "Nr": [],
            "DefaultEnergy": [],
            "DeliveredEnergy": [],
            "Impedance": [],
            "Current": [],
        }
    }
    for elem in defib_fire:
        defib["defibrillations"]["timestamp"].append(
            starttime + pd.Timedelta(milliseconds=elem["StdHdr"]["MsecTime"])
        )
        defib["defibrillations"]["Nr"].append(elem["ShockCnt"])
        defib["defibrillations"]["DefaultEnergy"].append(elem["EnergySel"])
        defib["defibrillations"]["DeliveredEnergy"].append(elem["EnergyDel"] / 10)
        defib["defibrillations"]["Impedance"].append(elem["PatientTti"] / 10)
        defib["defibrillations"]["Current"].append(elem["InitialCurrent"] / 10)

    defib["defibrillations"] = pd.DataFrame(defib["defibrillations"])
    defib["defibrillations"].set_index("timestamp", inplace=True)

    twelve_lead_dict = {"time_12_lead_ecg": np.array([])}
    for elem in twelve_lead_ecg:
        twelve_lead_dict["time_12_lead_ecg"] = np.append(
            twelve_lead_dict["time_12_lead_ecg"],
            starttime + pd.Timedelta(milliseconds=elem["StdHdr"]["MsecTime"]),
        )
    twelve_lead_dict["time_12_lead_ecg"] = pd.Series(
        twelve_lead_dict["time_12_lead_ecg"]
    )

    # Construct Data Info -dict
    data_info = {
        "Key": [],
        "Unit": [],
        "Type": [],
        "Start": [],
        "Stop": [],
        "Length": [],
    }

    for key in contin_info:
        data_info["Key"].append(key)
        data_info["Unit"].append(contin_info[key])
        data_info["Type"].append("Continous wave data")
        data_info["Start"].append(contin_data[key].first_valid_index())
        data_info["Stop"].append(contin_data[key].last_valid_index())
        data_info["Length"].append(contin_data[key].size)
    for item in trend_info["Trend data"]:
        data_info["Key"].append(item)
        data_info["Type"].append("Trend Data")
        data_info["Start"].append(trend_data[item].first_valid_index())
        data_info["Stop"].append(trend_data[item].last_valid_index())
        data_info["Length"].append(trend_data[item].size)
    for value in trend_info["Unit"]:
        data_info["Unit"].append(value)
    for item in compr_info["Trend data"]:
        data_info["Type"].append("Trend Data")
        data_info["Key"].append(item)
        data_info["Start"].append(compr_data[item].first_valid_index())
        data_info["Stop"].append(compr_data[item].last_valid_index())
        data_info["Length"].append(compr_data[item].size)
    for value in compr_info["Unit"]:
        data_info["Unit"].append(value)

    # If no recording is on data
    if rec_start == rec_stop:
        logger.error("Start and end of recording are the same. Check the file!")
        return None
    rec_length = rec_stop - rec_start

    # Info about Defi Models
    zoll_models = {
        "AF": ("ZOLL - R Series", "Monitor"),
        "AR": ("ZOLL - X Series", "Monitor"),
        "00": ("ZOLL - E Series", "Monitor"),
        "AA": ("ZOLL - AED Pro", "AED"),
    }
    ## Get Main Information about this case
    pat_dat = {"Main data": {}}
    pat_dat["Main data"]["File ID"] = filename

    nr = str(dev_conf[0]["DeviceSerialNumber"])
    logger.info(f"Device Serial Number: {nr}")
    pat_dat["Main data"]["Serial Nr"] = nr
    pat_dat["Main data"]["Product Code"] = nr[:2]
    pat_dat["Main data"]["Model"] = zoll_models[nr[:2]][0]
    pat_dat["Main data"]["Defib Category"] = zoll_models[nr[:2]][1]

    pat_dat["Main data"]["Start time"] = starttime
    pat_dat["Main data"]["Recording start"] = rec_start
    pat_dat["Main data"]["Recording end"] = rec_stop
    pat_dat["Main data"]["Recording length"] = rec_length

    pat_dat["Main data"] = pd.DataFrame.from_dict(pat_dat["Main data"], orient="index")
    pat_dat["Keys"] = pd.DataFrame(data_info)
    pat_dat["Load Log"] = pd.DataFrame(inv_ch, index=[0])

    # Construct big dict with all data
    data = {
        **contin_data,
        **trend_data,
        **compr_data,
        **defib,
        **twelve_lead_dict,
    }
    # for key in data:
    #    if 'timestamp'== data[key].index.name:
    #        delta_t=(data[key].index-rec_start).total_seconds().astype(float)
    #        data[key]['n_time']=delta_t
    #        data[key].set_index('n_time',drop=True,inplace=True)
    #        #data[key].drop('timestamp')
    #    elif 'timestamp' in data[key].columns:
    #        delta_t=(data[key]['timestamp']-rec_start).total_seconds().astype(float)
    #        data[key]['n_time']=delta_t
    #        data[key].drop('timestamp')

    return pat_dat, data


def read_zolljson_contin(wave_recording_elements, starttime):
    ## Routine to read continuum data of single json file:
    # Routine to get Infos on all Recorded Channels
    type_dict = {}
    calculated_intervals = {}  # liste der channels für die bereits ein interval berechnet wurde
    framesizes = {}
    t_previous = {}

    # coinfo={'Recording Interval':[], 'FrameSize':[],'Length Samples':[],'SampleTime':[]}

    for item in wave_recording_elements:
        try:
            item_type = item["WaveRec"]["WaveTypeVar"]
        except KeyError:
            item_type = find_wavetypevar(item["WaveRec"]["WaveType"])
        item_framesize = item["WaveRec"]["FrameSize"]
        # Es wird geprüft, ob zu dem Recording Item noch kein Eintrag registriert wurde
        # So wird diese angelegt
        if item_type not in type_dict:
            type_dict[item_type] = item["WaveRec"]
            # type_dict[item_type]["StdHdr"]=item["StdHdr"]
            type_dict[item_type]["SampleTime"] = item["WaveRec"]["SampleTime"]
            type_dict[item_type]["WaveType"] = item["WaveRec"]["WaveType"]
            # Verwerfe die Daten (UnpackedSamples) für die Kanalübersicht
            type_dict[item_type].pop("UnpackedSamples", None)
            type_dict[item_type].pop("SampleStatus", None)
            t_previous[item_type] = item["StdHdr"]["MsecTime"]
            # t_previous[item_type]=[item["StdHdr"]["MsecTime"]]
            calculated_intervals[item_type] = []
            framesizes[item_type] = []
            if "WaveTypeVar" not in type_dict[item_type].keys():
                type_dict[item_type]["WaveTypeVar"] = find_wavetypevar(
                    item["WaveRec"]["WaveType"]
                )

            # try ist nötig da CprAccelWave kein WaveSetting hat
            try:
                type_dict[item_type]["UnitsVal"] = item["WaveSetting"]["UnitsVal"]
            # Ausnahme für CprAccelWave
            except KeyError:
                type_dict[item_type]["UnitsVal"] = "Unknown"
        # Wenn das Item bereits vorliegt werden Framesize ggf. korrigiert und die Intervalle berechnet
        else:
            framesizes[item_type].append(item_framesize)
            # Berechnet Intervalllänge für den Typ
            t1 = t_previous[
                item_type
            ]  # Millisekundenwert des letzten Recording für diesen Type
            t2 = item["StdHdr"][
                "MsecTime"
            ]  # aktueller Millisekundenwert des letzten Recording für diesen Type
            calculated_intervals[item_type].append(
                (t2 - t1) * 1000
            )  # Intervalllänge in Mikrosekunden
            t_previous[item_type] = t2

    # KORREKTUR da falsche SampleTime (Zeit in Microsekunden zwischen Recordings) für CprAccelWaveRec gespeichert wird
    if "CPR Acceleration" in type_dict.keys():
        type_dict["CPR Acceleration"]["SampleTime"] = 4000

    # return coinfo

    delkeys = []  # Container for all keys to delete
    # Für alle Channels wird der Median der errechneten Intervalllänge eingetragen
    for key in type_dict.keys():
        try:
            if [f for f in calculated_intervals[key] if f != 0]:
                type_dict[key]["RecordingIntervals"] = int(
                    mode([f for f in calculated_intervals[key] if f != 0]).mode
                )
            if [f for f in framesizes[key] if f != 0]:
                type_dict[key]["FrameSize"] = int(
                    mode([f for f in framesizes[key] if f != 0]).mode
                )
        except ValueError:
            type_dict[key]["RecordingIntervals"] = 0
            type_dict[key]["FrameSize"] = 0
            delkeys.append(key)

    # Generating Pandas Dataframe
    df_DataChannels = pd.DataFrame(type_dict)
    df_DataChannels = df_DataChannels.reindex(sorted(df_DataChannels.columns), axis=1)

    # df_DataChannels.loc["CHECK SampleTime"]=df_DataChannels.loc["RecordingIntervals"]/df_DataChannels.loc["FrameSize"]
    # df_DataChannels.loc["Samplerate"]=1e6 / df_DataChannels.loc["SampleTime"]

    # Calculating SampleTime from Interval and Framesize and compare it with SampleTime
    df = df_DataChannels[df_DataChannels.columns[df_DataChannels.loc["FrameSize"] != 0]]
    if not df.empty:
        if (
            df.loc["SampleTime"]
            .compare(df.loc["RecordingIntervals"] / df.loc["FrameSize"])
            .empty
        ):
            pass
        else:
            probtyp = list(
                df.columns[
                    df.loc["SampleTime"]
                    != (df.loc["RecordingIntervals"] / df.loc["FrameSize"])
                ]
            )
            dfn = df[
                df.columns[
                    (
                        df.loc["SampleTime"]
                        == (df.loc["RecordingIntervals"] / df.loc["FrameSize"])
                    )
                ]
            ]
            for a in probtyp:
                if (
                    np.abs(
                        df[a]["SampleTime"]
                        / (df[a]["RecordingIntervals"] / df[a]["FrameSize"])
                        - 1
                    )
                    < 1e-3
                ):
                    # logger.warning(f"{a} has inexact Recording Intervals")
                    dfn[a] = df[a]
                # else:
                # logger.warning(
                #    "CAVE: Check Recording Intervals - Inconsistencies found in "
                #    f"{probtyp}. These channels have been dropped out"
                # )
            df = dfn
    ### Waves werden zu timeseries Data aneinander gehängt
    # routine to concatenate unpacked samples and generate timedelta timestamps
    dict_timestamps = {
        "start": starttime
        + pd.Timedelta(milliseconds=wave_recording_elements[0]["StdHdr"]["MsecTime"]),
        "end": starttime
        + pd.Timedelta(milliseconds=wave_recording_elements[0]["StdHdr"]["MsecTime"]),
    }
    wave_records_sample = dict((channel, []) for channel in list(df.columns.values))
    wave_records_timestamps = dict((channel, []) for channel in list(df.columns.values))
    msec_time_dev = dict((channel, []) for channel in list(df.columns.values))

    # Each item in wave-records is an ContinWaveRec containing StdHdr and Waveform
    invalid_ch = {
        "No Samples": 0,
        "Empty Samples": 0,
        "Wrong Framesize": 0,
        "Small time deviation": 0,
        "Invalid Msecs": 0,
    }
    n = -1
    for item in wave_recording_elements:
        n += 1
        try:
            item_type = item["WaveRec"]["WaveTypeVar"]
        except KeyError:
            item_type = find_wavetypevar(item["WaveRec"]["WaveType"])
        item_start = starttime + pd.Timedelta(milliseconds=item["StdHdr"]["MsecTime"])
        item_framesize = item["WaveRec"]["FrameSize"]
        item_properties = df_DataChannels[item_type]
        item_stop = item_start + pd.Timedelta(
            microseconds=item_properties["RecordingIntervals"]
        )

        try:
            item_samples = item["WaveRec"]["UnpackedSamples"]
            # testet ob ein vollständiges Record interval vorliegt  (ein fehler der beobachtet wurde:  volles Framesize bei CO2 ungleich länge des samples)
            if not len(item_samples) == 0:
                if (
                    len(item_samples) == item_framesize
                    and len(item_samples) == item_properties["FrameSize"]
                ):
                    if len(wave_records_timestamps[item_type]) != 0:
                        # One observed Error. MsecTime starts from the beginning
                        t_dif = item_start - pd.Timestamp(item["StdHdr"]["DevDateTime"])
                        t_dif = t_dif.total_seconds()
                        if np.abs(t_dif) > 10:
                            invalid_ch["Invalid Msecs"] += 1
                            item_start = pd.Timestamp(item["StdHdr"]["DevDateTime"])
                            item_stop = item_start + pd.Timedelta(
                                microseconds=item_properties["RecordingIntervals"]
                            )
                        else:
                            msec_time_dev[item_type].append(t_dif)

                        # One observed Error: Start times change slightly between different samples
                        # if difference is smaller than 20ms, ignore start time shift
                        if (
                            abs(
                                (
                                    item_start - wave_records_timestamps[item_type][-1]
                                ).total_seconds()
                            )
                            < 0.02
                            and abs(
                                (
                                    item_start - wave_records_timestamps[item_type][-1]
                                ).total_seconds()
                            )
                            != 0
                        ):
                            item_start = wave_records_timestamps[item_type][-1]
                            item_stop = item_start + pd.Timedelta(
                                microseconds=item_properties["RecordingIntervals"]
                            )
                            invalid_ch["Small time deviation"] += 1

                    wave_records_sample[item_type].extend(
                        item["WaveRec"]["UnpackedSamples"]
                    )
                    wave_records_timestamps[item_type].extend(
                        pd.date_range(
                            item_start,
                            end=item_stop,
                            periods=item_framesize + 1,
                            inclusive="right",
                        )
                    )
                    if item_start < dict_timestamps["start"]:
                        dict_timestamps["start"] = item_start
                        dict_timestamps["starttime"] = item["StdHdr"]["DevDateTime"]
                    if item_stop > dict_timestamps["end"]:
                        dict_timestamps["end"] = item_stop
                else:
                    invalid_ch["Wrong Framesize"] += 1
            else:
                invalid_ch["Empty Samples"] += 1
        except KeyError:
            invalid_ch["No Samples"] += 1

    logger.info(f"The file has a total number of {n} recordings.")
    # if invalid_ch["No Samples"] > 0:
    #    logger.warning(f"{invalid_ch['No Samples']} recordings contained no samples!")
    # if invalid_ch["Empty Samples"] > 0:
    #    logger.warning(f"{invalid_ch['Empty Samples']} recordings contained empty samples!")
    # if invalid_ch["Wrong Framesize"] > 0:
    #    logger.warning(f"{invalid_ch['Wrong Framesize']} recordings contained a wrong framesize!")
    # if invalid_ch["Small time deviation"] > 0:
    #    logger.warning(
    #        f"{invalid_ch['Small time deviation']} recordings had a start time deviation of <20ms!"
    #    )

    num_successfully_loaded = (
        n
        - invalid_ch["No Samples"]
        - invalid_ch["Empty Samples"]
        - invalid_ch["Wrong Framesize"]
    )
    logger.info(f"{num_successfully_loaded} recordings were successfully loaded.")

    ## Schreibe noch Rückgabe für recording_start und recording_end in allg Info-Datenfile
    recording_start = dict_timestamps["start"]
    recording_end = dict_timestamps["end"]

    # skipped_channel = {}  #Variabel zum Zweck des Debuggings
    timeseries_data = {}
    for key in wave_records_sample:
        # skipped_channel[key]=[]
        channel_name = df_DataChannels.loc["WaveTypeVar", key]
        if len(wave_records_sample[key]) == len(wave_records_timestamps[key]):
            timeseries_data[key] = pd.DataFrame(
                {channel_name: wave_records_sample[key]}, columns=[channel_name]
            )
            timeseries_data[key].index = wave_records_timestamps[key]
            timeseries_data[key].index.name = "timestamp"
        # else:
        #    skipped_channel[key].append(len(wave_records_sample[key]))

        # display(timeseries_data[key])

    # Change measure number to fit physical units
    # CO2 mmHg is given in multiples of 0.1 mmHg
    if "CO2 mmHg, Waveform" in timeseries_data:
        timeseries_data["CO2 mmHg, Waveform"]["CO2 mmHg, Waveform"] /= 10

    # For ECG data 400 internal units = 1 mV
    ecg_keys = [
        "Pads",
        "Filtered ECG",
        "12-Lead I",
        "12-Lead II",
        "12-Lead III",
        "12-Lead aVR",
        "12-Lead L",
        "12-Lead G",
        "12-Lead V",
        "12-Lead V1",
        "12-Lead V2",
        "12-Lead V3",
        "12-Lead V4",
        "12-Lead V5",
        "12-Lead V6",
    ]
    for key in ecg_keys:
        if key in timeseries_data:
            timeseries_data[key][key] /= 400
            df.at["UnitsVal", key] = "mV"

    shift_keys = {}
    refkey = ""
    i = 0
    while refkey == "" and i < len(ecg_keys):
        ekey = ecg_keys[i]
        if ekey in msec_time_dev:
            refkey = ekey
        i += 1
    if refkey != "":
        for key in msec_time_dev:
            shift_keys[key] = np.median(msec_time_dev[key]) - np.median(
                msec_time_dev[refkey]
            )

        for key in timeseries_data:
            timeseries_data[key].index = timeseries_data[key].index - pd.Timedelta(
                shift_keys[key], unit="s"
            )

    return timeseries_data, df, recording_start, recording_end, invalid_ch


# routine to read trend reports
def read_zolljson_trend(trend_rpt, starttime):
    trend_keys = [
        "LtaStateVal",
        "LtaState",
        "Temp",
        "Hr",
        "Fico2",
        "Spo2",
        "PatMode",
        "Nibp",
        "Ibp",
        "Etco2",
        "Resp",
    ]
    trend_data = {}
    trend_info = {"Trend data": [], "Unit": []}
    for report in trend_rpt:
        timedel = starttime + pd.Timedelta(milliseconds=report["StdHdr"]["MsecTime"])
        t_dif = timedel - pd.Timestamp(report["StdHdr"]["DevDateTime"])
        t_dif = t_dif.total_seconds()
        if np.abs(t_dif) > 10:
            timedel = pd.Timestamp(report["StdHdr"]["DevDateTime"])

        for key in report["Trend"]:
            # Check wheter this trend key is already known
            if key not in trend_keys:
                logger.warning(
                    f"Unknown trend key <{key}> found! Check out "
                    "structure and amend implementation."
                )
            else:
                # Check different trend keys. due to their changing structure we have to unpack the date for each key.
                if key == "Nibp":
                    bp_types = ["Map", "Sys", "Dia"]
                    for bpt in bp_types:
                        # Check, if DataState is valid, This structure is similar for all trends
                        try:
                            datastate = report["Trend"]["Nibp"][bpt]["TrendData"][
                                "DataState"
                            ]
                        except KeyError:
                            datastate = report["Trend"]["Nibp"][bpt]["TrendData"][
                                "DataStatus"
                            ]
                            if datastate == 0:
                                datastate = "valid"
                            else:
                                datastate = "invalid"
                        if datastate == "valid":
                            # Nibp has its own timestamps, overwrite timedel
                            nibp_timedel = pd.to_datetime(
                                report["Trend"]["Nibp"]["Time"]
                            )
                            # Create new dict in trend_data and fill with first entry
                            if key + bpt not in trend_data:
                                trend_data[key + bpt] = {
                                    "timestamp": [],
                                    key + bpt: [],
                                }
                                trend_data[key + bpt]["timestamp"].append(nibp_timedel)
                                trend_data[key + bpt][key + bpt].append(
                                    report["Trend"]["Nibp"][bpt]["TrendData"]["Val"][
                                        "#text"
                                    ]
                                )
                                trend_info["Trend data"].append(key + bpt)
                                try:
                                    trend_info["Unit"].append(
                                        report["Trend"]["Nibp"][bpt]["TrendData"][
                                            "Val"
                                        ]["@UnitsVal"]
                                    )
                                except KeyError:
                                    trend_info["Unit"].append(
                                        report["Trend"]["Nibp"][bpt]["TrendData"][
                                            "Val"
                                        ]["@Units"]
                                    )
                            # If Timestamp is not the same es before fill with new entries
                            elif (
                                not nibp_timedel
                                == trend_data[key + bpt]["timestamp"][-1]
                            ):
                                trend_data[key + bpt]["timestamp"].append(nibp_timedel)
                                trend_data[key + bpt][key + bpt].append(
                                    report["Trend"]["Nibp"][bpt]["TrendData"]["Val"][
                                        "#text"
                                    ]
                                )
                            else:
                                pass
                elif key == "Spo2":
                    spotypes = list(report["Trend"]["Spo2"].keys())
                    spotypes.remove("TrendData")
                    spotypes.remove("ChanState")
                    # spotypes=['SpCo', 'SpMet', 'PVI' , 'PI' , 'SpOC', 'SpHb']
                    # First for Main Spo2 # Check, if DataState is valid
                    try:
                        datastate = report["Trend"]["Spo2"]["TrendData"]["DataState"]
                    except KeyError:
                        datastate = report["Trend"]["Spo2"]["TrendData"]["DataStatus"]
                        if datastate == 0:
                            datastate = "valid"
                        else:
                            datastate = "invalid"
                    if datastate == "valid":
                        if "Spo2" not in trend_data:
                            trend_data["Spo2"] = {"timestamp": [], "Spo2": []}
                            trend_info["Trend data"].append("Spo2")
                            try:
                                trend_info["Unit"].append(
                                    report["Trend"]["Spo2"]["TrendData"]["Val"][
                                        "@UnitsVal"
                                    ]
                                )
                            except KeyError:
                                trend_info["Unit"].append(
                                    report["Trend"]["Spo2"]["TrendData"]["Val"][
                                        "@Units"
                                    ]
                                )
                        trend_data["Spo2"]["timestamp"].append(timedel)
                        trend_data["Spo2"]["Spo2"].append(
                            report["Trend"]["Spo2"]["TrendData"]["Val"]["#text"]
                        )
                    # For Spo2 subtypes
                    for typ in spotypes:
                        try:
                            datastate = report["Trend"]["Spo2"][typ]["TrendData"][
                                "DataState"
                            ]
                        except KeyError:
                            datastate = report["Trend"]["Spo2"][typ]["TrendData"][
                                "DataStatus"
                            ]
                            if datastate == 0:
                                datastate = "valid"
                            else:
                                datastate = "invalid"

                        if datastate == "valid":
                            typname = "Spo2" + typ
                            if typname not in trend_data:
                                trend_info["Trend data"].append(typname)
                                try:
                                    trend_info["Unit"].append(
                                        report["Trend"]["Spo2"][typ]["TrendData"][
                                            "Val"
                                        ]["@UnitsVal"]
                                    )
                                except KeyError:
                                    trend_info["Unit"].append(
                                        report["Trend"]["Spo2"][typ]["TrendData"][
                                            "Val"
                                        ]["@Units"]
                                    )
                                trend_data["Spo2" + typ] = {
                                    "timestamp": [],
                                    typname: [],
                                }
                            trend_data[typname]["timestamp"].append(timedel)
                            trend_data[typname][typname].append(
                                report["Trend"]["Spo2"][typ]["TrendData"]["Val"][
                                    "#text"
                                ]
                            )

                elif key == "Ibp":
                    bp_types = ["Map", "Sys", "Dia"]
                    for j in range(len(report["Trend"]["Ibp"])):
                        for typ in bp_types:
                            try:
                                datastate = report["Trend"]["Ibp"][j][typ]["TrendData"][
                                    "DataState"
                                ]
                            except KeyError:
                                datastate = report["Trend"]["Ibp"][j][typ]["TrendData"][
                                    "DataStatus"
                                ]
                                if datastate == 0:
                                    datastate = "valid"
                                else:
                                    datastate = "invalid"
                            if datastate == "valid":
                                typname = "Cha" + str(j + 1) + typ
                                if typname not in trend_data:
                                    trend_data[typname] = {
                                        "timestamp": [],
                                        typname: [],
                                    }
                                    trend_info["Trend data"].append(typname)
                                    try:
                                        trend_info["Unit"].append(
                                            report["Trend"]["Ibp"][j][typ]["TrendData"][
                                                "Val"
                                            ]["@UnitsVal"]
                                        )
                                    except KeyError:
                                        trend_info["Unit"].append(
                                            report["Trend"]["Ibp"][j][typ]["TrendData"][
                                                "Val"
                                            ]["@Units"]
                                        )
                                trend_data[typname]["timestamp"].append(timedel)
                                trend_data[typname][typname].append(
                                    report["Trend"]["Ibp"][j][typ]["TrendData"]["Val"][
                                        "#text"
                                    ]
                                )

                elif key == "Temp":
                    for j in range(len(report["Trend"]["Temp"])):
                        try:
                            datastate = report["Trend"]["Temp"][j]["TrendData"][
                                "DataState"
                            ]
                        except KeyError:
                            datastate = report["Trend"]["Temp"][j]["TrendData"][
                                "DataStatus"
                            ]
                            if datastate == 0:
                                datastate = "valid"
                            else:
                                datastate = "invalid"
                        if datastate == "valid":
                            try:
                                typname = report["Trend"]["Temp"][j]["SrcLabelVal"]
                            except KeyError:
                                typname = report["Trend"]["Temp"][j]["SrcLabel"]
                            if typname not in trend_data:
                                trend_data[typname] = {"timestamp": [], typname: []}
                                trend_info["Trend data"].append(typname)
                                try:
                                    trend_info["Unit"].append(
                                        report["Trend"]["Temp"][j]["TrendData"]["Val"][
                                            "@UnitsVal"
                                        ]
                                    )
                                except KeyError:
                                    trend_info["Unit"].append(
                                        report["Trend"]["Temp"][j]["TrendData"]["Val"][
                                            "@Units"
                                        ]
                                    )
                            trend_data[typname]["timestamp"].append(timedel)
                            trend_data[typname][typname].append(
                                report["Trend"]["Temp"][j]["TrendData"]["Val"]["#text"]
                            )
                elif key in ["Hr", "Fico2", "Etco2", "Resp"]:
                    try:
                        datastate = report["Trend"][key]["TrendData"]["DataState"]
                    except KeyError:
                        datastate = report["Trend"][key]["TrendData"]["DataStatus"]
                        if datastate == 0:
                            datastate = "valid"
                        else:
                            datastate = "invalid"
                    if datastate == "valid":
                        typname = key
                        if typname not in trend_data:
                            trend_data[typname] = {"timestamp": [], typname: []}
                            trend_info["Trend data"].append(typname)
                            try:
                                trend_info["Unit"].append(
                                    report["Trend"][key]["TrendData"]["Val"][
                                        "@UnitsVal"
                                    ]
                                )
                            except KeyError:
                                trend_info["Unit"].append(
                                    report["Trend"][key]["TrendData"]["Val"]["@Units"]
                                )
                        trend_data[typname]["timestamp"].append(timedel)
                        trend_data[typname][typname].append(
                            report["Trend"][key]["TrendData"]["Val"]["#text"]
                        )
                elif key in ["LtaState", "LtaStateVal", "PatMode"]:
                    pass
                else:
                    logger.warning(f"Unknown Trend key <{key}> found.")

    for key in trend_data:
        trend_data[key] = pd.DataFrame(trend_data[key])
        trend_data[key].set_index("timestamp", inplace=True)
    for key in trend_data:
        if "Temp" in key:
            trend_data[key][key] = trend_data[key][key] / 10

    return trend_data, trend_info


# routine to read compression data
def read_zolljson_compr(cpr_compr, starttime):
    cpr_compr_keys = ["CompDisp", "RelVelocity", "CompRate"]
    compr_data = {
        "CompDisp": {"timestamp": [], "CompDisp": []},
        "RelVelocity": {"timestamp": [], "RelVelocity": []},
        "CompRate": {"timestamp": [], "CompRate": []},
    }
    compr_info = {
        "Trend data": ["CompDisp", "RelVelocity", "CompRate"],
        "Unit": ["cm", "Unknown", "Compressions /min"],
    }
    for report in cpr_compr:
        # There is a second timestamp in this key which differs from the used one about 0.05-0.5 s.
        # timestamp=pd.to_datetime(report['Timestamp'])
        timestamp = starttime + pd.Timedelta(milliseconds=report["StdHdr"]["MsecTime"])
        t_dif = timestamp - pd.Timestamp(report["StdHdr"]["DevDateTime"])
        t_dif = t_dif.total_seconds()
        if np.abs(t_dif) > 10:
            timestamp = pd.Timestamp(report["StdHdr"]["DevDateTime"])
        for typ in cpr_compr_keys:
            compr_data[typ]["timestamp"].append(timestamp)
            compr_data[typ][typ].append(report[typ])

    for key in compr_data:
        compr_data[key] = pd.DataFrame(compr_data[key])
        compr_data[key].set_index("timestamp", inplace=True)

    compr_data["CompDisp"] = compr_data["CompDisp"] * 2.54 / 1000

    return compr_data, compr_info


def read_zollxml(filepath: Path | str):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    filename = filepath.name
    tree = ET.parse(filepath)
    root = tree.getroot()
    all_keys = [
        "NewCase",
        "Ecg12LeadRec",
        "AnnotationEvt",
        "TrendRpt",
        "DeviceConfiguration",
        "DataChannels",
        "PatientInfo",
        "AlarmLimits",
        "TraceConfigs",
        "DeleteCaseRec",
        "Aed",
        "DisplayInfo",
        "ContinWaveRec",
        "CprAccelWaveRec",
        "CprCompression",
        "AedContinAnalysis",
        "InstrumentPacket",
        "AlarmEvt",
        "SnapshotRpt",
        "SysLogEntry",
        "Ecg12LeadDisc",
        "ReadinessLogRecord",
        "DefibTrace",
        "PrtTrace",
        "CaseSummary",
        "AedSingleAnalysis",
        "DefibFireEvt",
        "AplsLoadSignalRecord",
        "CableChangeEvt",
        "WebConsoleConfigParameter",
        "SelfTestFailure",
        "PaceEvt",
    ]
    # all_keys=['NewCase', 'AnnotationEvt', 'TrendRpt', 'DeviceConfiguration',  'ContinWaveRec', 'CprAccelWaveRec', 'CprCompression']

    single_wave_record = ["CprAccelWaveRec", "AplsLoadSignalRecord"]
    multi_waves_record = ["ContinWaveRec", "SnapshotRpt"]
    wave_recording_elements = []
    trend_rpt = []
    cpr_compr = []
    new_case = []
    dev_conf = []
    defib_fire = []
    twelve_lead_ecg = []

    for elem in root[0]:
        if elem.tag in single_wave_record:
            wave_recording_elements.append(elem)
        elif elem.tag in multi_waves_record:
            stdhdr = elem[0]
            for waveform in elem.findall("Waveform"):
                waveform.append(stdhdr)
                wave_recording_elements.append(waveform)
        elif elem.tag == "TrendRpt":
            trend_rpt.append(elem)
        elif elem.tag == "CprCompression":
            cpr_compr.append(elem)
        elif elem.tag == "NewCase":
            new_case.append(elem)
        elif elem.tag == "DeviceConfiguration":
            dev_conf.append(elem)
        elif elem.tag == "DefibFireEvt":
            defib_fire.append(elem)
        elif elem.tag == "Ecg12LeadRec":
            twelve_lead_ecg.append(elem)
        # If a unknown key is found
        elif elem.tag not in all_keys:
            logger.warning(f"New key <{elem.tag}> found. Check structure and content.")

    for dev_dat_time in new_case[0].iter("DevDateTime"):
        date = dev_dat_time.text
    for msectime in new_case[0].iter("MsecTime"):
        msecs = msectime
    starttime = pd.to_datetime(date) - pd.Timedelta(milliseconds=float(msecs.text))

    for new_case_entry in new_case:
        dates = []
        times = []
        for date in new_case_entry.iter("DevDateTime"):
            dates.append(pd.to_datetime(date.text))
        for time in new_case_entry.iter("MsecTime"):
            times.append(pd.Timedelta(milliseconds=float(time.text)))
        alt_starttimes = [date - time for date in dates for time in times]
        for elem in alt_starttimes:
            if elem != starttime:
                logger.warning(f"Another NewCase found, starting time: {elem}")

    (
        contin_data,
        contin_info,
        rec_start,
        rec_stop,
        inv_ch,
    ) = read_zollxml_contin(wave_recording_elements, starttime)
    contin_info = contin_info.loc["UnitsVal"].to_dict()
    contin_info["SpO2 %, Waveform"] = "1"

    trend_data, trend_info = read_zollxml_trend(trend_rpt, starttime)

    compr_data, compr_info = read_zollxml_compr(cpr_compr, starttime)

    defib = {
        "defibrillations": {
            "timestamp": [],
            "Nr": [],
            "DefaultEnergy": [],
            "DeliveredEnergy": [],
            "Impedance": [],
            "Current": [],
        }
    }
    for elem in defib_fire:
        defib["defibrillations"]["timestamp"].append(
            starttime
            + pd.Timedelta(
                milliseconds=float(elem.find("StdHdr").find("MsecTime").text)
            )
        )
        defib["defibrillations"]["Nr"].append(int(elem.find("ShockCnt").text))
        defib["defibrillations"]["DefaultEnergy"].append(
            float(elem.find("EnergySel").text)
        )
        defib["defibrillations"]["DeliveredEnergy"].append(
            float(elem.find("EnergyDel").text) / 10
        )
        defib["defibrillations"]["Impedance"].append(
            float(elem.find("PatientTti").text) / 10
        )
        defib["defibrillations"]["Current"].append(
            float(elem.find("InitialCurrent").text) / 10
        )

    defib["defibrillations"] = pd.DataFrame(defib["defibrillations"])
    defib["defibrillations"].set_index("timestamp", inplace=True)

    twelve_lead_dict = {"time_12_lead_ecg": np.array([])}
    for elem in twelve_lead_ecg:
        twelve_lead_dict["time_12_lead_ecg"] = np.append(
            twelve_lead_dict["time_12_lead_ecg"],
            starttime
            + pd.Timedelta(
                milliseconds=float(elem.find("StdHdr").find("MsecTime").text)
            ),
        )
    twelve_lead_dict["time_12_lead_ecg"] = pd.Series(
        twelve_lead_dict["time_12_lead_ecg"]
    )

    # Construct Data Info -dict
    data_info = {
        "Key": [],
        "Unit": [],
        "Type": [],
        "Start": [],
        "Stop": [],
        "Length": [],
    }
    for key in contin_info:
        data_info["Key"].append(key)
        data_info["Unit"].append(contin_info[key])
        data_info["Type"].append("Continous wave data")
        data_info["Start"].append(contin_data[key].first_valid_index())
        data_info["Stop"].append(contin_data[key].last_valid_index())
        data_info["Length"].append(contin_data[key].size)
    for item in trend_info["Trend data"]:
        data_info["Key"].append(item)
        data_info["Type"].append("Trend Data")
        data_info["Start"].append(trend_data[item].first_valid_index())
        data_info["Stop"].append(trend_data[item].last_valid_index())
        data_info["Length"].append(trend_data[item].size)
    for value in trend_info["Unit"]:
        data_info["Unit"].append(value)
    for item in compr_info["Trend data"]:
        data_info["Type"].append("Trend Data")
        data_info["Key"].append(item)
        data_info["Start"].append(compr_data[item].first_valid_index())
        data_info["Stop"].append(compr_data[item].last_valid_index())
        data_info["Length"].append(compr_data[item].size)
    for value in compr_info["Unit"]:
        data_info["Unit"].append(value)

    # If no recording is on data
    if rec_start == rec_stop:
        logger.error("Start and end of recording are the same. Check the file!")
        return None
    rec_length = rec_stop - rec_start

    # Info about Defi Models
    zoll_models = {
        "AF": ("ZOLL - R Series", "Monitor"),
        "AR": ("ZOLL - X Series", "Monitor"),
        "00": ("ZOLL - E Series", "Monitor"),
        "AA": ("ZOLL - AED Pro", "AED"),
    }
    ## Get Main Information about this case
    pat_dat = {"Main data": {}}
    pat_dat["Main data"]["File ID"] = filename

    nr = str(dev_conf[0].find("DeviceSerialNumber").text)
    pat_dat["Main data"]["Serial Nr"] = nr
    pat_dat["Main data"]["Product Code"] = nr[:2]
    pat_dat["Main data"]["Model"] = zoll_models[nr[:2]][0]
    pat_dat["Main data"]["Defib Category"] = zoll_models[nr[:2]][1]
    pat_dat["Main data"]["Start time"] = starttime
    pat_dat["Main data"]["Recording start"] = rec_start
    pat_dat["Main data"]["Recording end"] = rec_stop
    pat_dat["Main data"]["Recording length"] = rec_length

    pat_dat["Main data"] = pd.DataFrame.from_dict(pat_dat["Main data"], orient="index")
    pat_dat["Keys"] = pd.DataFrame(data_info)
    pat_dat["Load Log"] = pd.DataFrame(inv_ch, index=[0])

    # Construct big dict with all data
    data = {
        **contin_data,
        **trend_data,
        **compr_data,
        **defib,
        **twelve_lead_dict,
    }
    return pat_dat, data


def read_zollxml_contin(wave_recording_elements, starttime):
    type_dict = {}
    calculated_intervals = {}  # liste der channels für die bereits ein interval berechnet wurde
    framesizes = {}
    t_previous = {}

    for elem in wave_recording_elements:
        item_type = elem.find("WaveRec").find("WaveType").text
        item_framesize = float(elem.find("WaveRec").find("FrameSize").text)
        if item_type not in type_dict:
            type_dict[item_type] = {}
            for rec_info in elem.find("WaveRec"):
                type_dict[item_type][rec_info.tag] = rec_info.text
            type_dict[item_type]["SampleTime"] = int(type_dict[item_type]["SampleTime"])
            type_dict[item_type].pop("UnpackedSamples", None)
            type_dict[item_type].pop("SampleStatus", None)
            t_previous[item_type] = float(elem.find("StdHdr").find("MsecTime").text)
            calculated_intervals[item_type] = []
            framesizes[item_type] = []
            try:
                type_dict[item_type]["UnitsVal"] = (
                    elem.find("WaveSetting").find("UnitsVal").text
                )
                # Ausnahme für CprAccelWave
            except AttributeError:
                type_dict[item_type]["UnitsVal"] = "Unknown"
        else:
            framesizes[item_type].append(item_framesize)
            t1 = t_previous[item_type]
            t2 = float(elem.find("StdHdr").find("MsecTime").text)
            calculated_intervals[item_type].append((t2 - t1) * 1000)
            t_previous[item_type] = t2

    delkeys = []  # Container for all keys to delete
    # Für alle Channels wird der Median der errechneten Intervalllänge eingetragen
    for key in type_dict.keys():
        try:
            if [f for f in calculated_intervals[key] if f != 0]:
                type_dict[key]["RecordingIntervals"] = int(
                    mode([f for f in calculated_intervals[key] if f != 0]).mode
                )  #
            if [f for f in framesizes[key] if f != 0]:
                type_dict[key]["FrameSize"] = int(
                    mode([f for f in framesizes[key] if f != 0]).mode
                )

        except ValueError:
            type_dict[key]["RecordingIntervals"] = 0
            type_dict[key]["FrameSize"] = 0
            delkeys.append(key)

    if "30" in type_dict.keys():
        type_dict["30"]["SampleTime"] = 4000

    df_DataChannels = pd.DataFrame(type_dict)
    df_DataChannels = df_DataChannels.reindex(sorted(df_DataChannels.columns), axis=1)

    # Calculating SampleTime from Interval and Framesize and compare it with SampleTime
    # Calculating SampleTime from Interval and Framesize and compare it with SampleTime
    df = df_DataChannels[df_DataChannels.columns[df_DataChannels.loc["FrameSize"] != 0]]
    if not df.empty:
        if (
            df.loc["SampleTime"]
            .compare(df.loc["RecordingIntervals"] / df.loc["FrameSize"])
            .empty
        ):
            pass
        else:
            probtyp = list(
                df.columns[
                    df.loc["SampleTime"]
                    != (df.loc["RecordingIntervals"] / df.loc["FrameSize"])
                ]
            )
            dfn = df[
                df.columns[
                    (
                        df.loc["SampleTime"]
                        == (df.loc["RecordingIntervals"] / df.loc["FrameSize"])
                    )
                ]
            ]
            for a in probtyp:
                if (
                    np.abs(
                        df[a]["SampleTime"]
                        / (df[a]["RecordingIntervals"] / df[a]["FrameSize"])
                        - 1
                    )
                    < 1e-3
                ):
                    # logger.warning(f"{a} has inexact Recording Intervals")
                    dfn[a] = df[a]
                # else:
                #     logger.warning(
                #         "CAVE: Check Recording Intervals - Inconsistencies found in "
                #         f"{probtyp}. These channels have been dropped out"
                #     )
            df = dfn
    ii = 0
    while float(wave_recording_elements[ii].find("StdHdr").find("MsecTime").text) > 1e8:
        ii += 1

    dict_timestamps = {
        "start": starttime
        + pd.Timedelta(
            milliseconds=float(
                wave_recording_elements[ii].find("StdHdr").find("MsecTime").text
            )
        ),
        "end": starttime
        + pd.Timedelta(
            milliseconds=float(
                wave_recording_elements[ii].find("StdHdr").find("MsecTime").text
            )
        ),
    }
    wave_records_sample = dict(
        (channel, []) for channel in list(df_DataChannels.columns.values)
    )
    wave_records_timestamps = dict(
        (channel, []) for channel in list(df_DataChannels.columns.values)
    )
    msec_time_dev = dict(
        (find_wavetypevar(channel), [])
        for channel in list(df_DataChannels.columns.values)
    )

    # Each item in wave-records is an ContinWaveRec containing StdHdr and Waveform
    invalid_ch = {
        "No Samples": 0,
        "Empty Samples": 0,
        "Wrong Framesize": 0,
        "Small time deviation": 0,
        "Invalid Starttime": 0,
        "Invalid Msecs": 0,
    }
    n = ii - 1

    for elem in wave_recording_elements[ii:]:
        n += 1
        item_type = elem.find("WaveRec").find("WaveType").text
        item_framesize = int(elem.find("WaveRec").find("FrameSize").text)
        item_properties = df_DataChannels[item_type]

        if float(elem.find("StdHdr").find("MsecTime").text) > 1e8:
            invalid_ch["Invalid Starttime"] += 1
        else:
            item_start = starttime + pd.Timedelta(
                milliseconds=float(elem.find("StdHdr").find("MsecTime").text)
            )
            item_stop = item_start + pd.Timedelta(
                microseconds=item_properties["RecordingIntervals"]
            )

            try:
                item_samples = list(
                    (elem.find("WaveRec").find("UnpackedSamples").text).split()
                )
                item_samples = [int(a) for a in item_samples]
                # testet ob ein vollständiges Record interval vorliegt  (ein fehler der beobachtet wurde:  volles Framesize bei CO2 ungleich länge des samples)
                if not len(item_samples) == 0:
                    if (
                        len(item_samples) == item_framesize
                        and len(item_samples) == item_properties["FrameSize"]
                    ):
                        # One observed Error: MsecTimes change sometimes abruptly. If Difference to DevDateTime too big, use DevDatetime as timestamp

                        if len(wave_records_timestamps[item_type]) != 0:
                            t_dif = item_start - pd.Timestamp(
                                elem.find("StdHdr").find("DevDateTime").text
                            )
                            t_dif = t_dif.total_seconds()
                            if np.abs(t_dif) > 10:
                                invalid_ch["Invalid Msecs"] += 1
                                item_start = pd.Timestamp(
                                    elem.find("StdHdr").find("DevDateTime").text
                                )
                                item_stop = item_start + pd.Timedelta(
                                    microseconds=item_properties["RecordingIntervals"]
                                )
                            else:
                                msec_time_dev[find_wavetypevar(item_type)].append(t_dif)
                            # One observed Error: Start times change slightly between different samples
                            # if difference is smaller than 20ms, ignore start time shift
                            if (
                                abs(
                                    (
                                        item_start
                                        - wave_records_timestamps[item_type][-1]
                                    ).total_seconds()
                                )
                                < 0.02
                                and abs(
                                    (
                                        item_start
                                        - wave_records_timestamps[item_type][-1]
                                    ).total_seconds()
                                )
                                != 0
                            ):
                                item_start = wave_records_timestamps[item_type][-1]
                                item_stop = item_start + pd.Timedelta(
                                    microseconds=item_properties["RecordingIntervals"]
                                )
                                invalid_ch["Small time deviation"] += 1
                        if (
                            len(wave_records_timestamps[item_type]) > 0
                            and item_start + pd.Timedelta(seconds=0.2)
                            < wave_records_timestamps[item_type][-1]
                        ):
                            logger.error(
                                f"Nr {n} Type: {item_type} Problem with new times found"
                            )
                        wave_records_sample[item_type].extend(item_samples)
                        wave_records_timestamps[item_type].extend(
                            pd.date_range(
                                item_start,
                                end=item_stop,
                                periods=item_framesize + 1,
                                inclusive="right",
                            )
                        )
                        if item_start < dict_timestamps["start"]:
                            dict_timestamps["start"] = item_start
                            dict_timestamps["starttime"] = pd.Timestamp(
                                elem.find("StdHdr").find("DevDateTime").text
                            )
                        if item_stop > dict_timestamps["end"]:
                            dict_timestamps["end"] = item_stop
                    else:
                        invalid_ch["Wrong Framesize"] += 1
                else:
                    invalid_ch["Empty Samples"] += 1
            except AttributeError:
                invalid_ch["No Samples"] += 1

    logger.info(f"The file has a total number of {n} recordings.")
    # if invalid_ch["No Samples"] > 0:
    #     logger.warning(f"{invalid_ch['No Samples']} recordings contained no samples!")
    # if invalid_ch["Empty Samples"] > 0:
    #     logger.warning(f"{invalid_ch['Empty Samples']} recordings contained empty samples!")
    # if invalid_ch["Wrong Framesize"] > 0:
    #     logger.warning(f"{invalid_ch['Wrong Framesize']} recordings contained a wrong framesize!")
    # if invalid_ch["Small time deviation"] > 0:
    #     logger.warning(
    #         f"{invalid_ch['Small time deviation']} recordings had a start time deviation of <20ms!"
    #     )
    #     if invalid_ch["Invalid Msecs"] > 0:
    #         logger.warning(
    #             f"{invalid_ch['Invalid Msecs']} recordings had a wrong Msec value!"
    #         )

    num_successfully_loaded = (
        n
        - invalid_ch["No Samples"]
        - invalid_ch["Empty Samples"]
        - invalid_ch["Wrong Framesize"]
    )
    logger.info(f"{num_successfully_loaded} recordings were successfully loaded.")

    ## Schreibe noch Rückgabe für recording_start und recording_end in allg Info-Datenfile
    recording_start = dict_timestamps["start"]
    recording_end = dict_timestamps["end"]

    timeseries_data = {}
    for key in wave_records_sample:
        wavtyp = find_wavetypevar(key)
        # skipped_channel[key]=[]
        channel_name = find_wavetypevar(df_DataChannels.loc["WaveType", key])
        if len(wave_records_sample[key]) == len(wave_records_timestamps[key]):
            timeseries_data[wavtyp] = pd.DataFrame(
                {channel_name: wave_records_sample[key]}, columns=[channel_name]
            )
            timeseries_data[wavtyp].index = wave_records_timestamps[key]
            timeseries_data[wavtyp].index.name = "timestamp"
        else:
            wave_records_sample.pop(key)

    types = list(df.columns)
    new_df = {}
    for entry in types:
        new_df[entry] = find_wavetypevar(entry)
    new_df = pd.DataFrame(new_df, index=["WaveTypeVar"])
    try:
        df = df.drop("WaveTypeVar")
    except KeyError:
        pass
    df = pd.concat([df, new_df])
    df.columns = df.transpose()["WaveTypeVar"]

    # Change measure number to fit physical units
    # CO2 mmHg is given in multiples of 0.1 mmHg
    if "CO2 mmHg, Waveform" in timeseries_data:
        timeseries_data["CO2 mmHg, Waveform"]["CO2 mmHg, Waveform"] /= 10

    # For ECG data 400 internal units = 1 mV
    ecg_keys = [
        "Pads",
        "Filtered ECG",
        "12-Lead I",
        "12-Lead II",
        "12-Lead III",
        "12-Lead aVR",
        "12-Lead L",
        "12-Lead G",
        "12-Lead V",
        "12-Lead V1",
        "12-Lead V2",
        "12-Lead V3",
        "12-Lead V4",
        "12-Lead V5",
        "12-Lead V6",
    ]
    for key in ecg_keys:
        if key in timeseries_data:
            timeseries_data[key][key] /= 400
            df.at["UnitsVal", key] = "mV"

    shift_keys = {}
    refkey = ""
    i = 0
    while refkey == "" and i < len(ecg_keys):
        ekey = ecg_keys[i]
        if ekey in msec_time_dev:
            refkey = ekey
        i += 1
    if refkey != 0:
        for key in msec_time_dev:
            if len(msec_time_dev[key]) > 0:
                shift_keys[key] = np.median(msec_time_dev[key]) - np.median(
                    msec_time_dev[refkey]
                )
            else:
                shift_keys[key] = 0
        for key in timeseries_data:
            try:
                timeseries_data[key].index = timeseries_data[key].index - pd.Timedelta(
                    shift_keys[key], unit="s"
                )
            except KeyError:
                pass

    return timeseries_data, df, recording_start, recording_end, invalid_ch


def read_zollxml_trend(trend_rpt, starttime):
    trend_keys = [
        "LtaStateVal",
        "LtaState",
        "Temp",
        "Hr",
        "Fico2",
        "Spo2",
        "PatMode",
        "Nibp",
        "Ibp",
        "Etco2",
        "Resp",
    ]
    trend_data = {}
    trend_info = {"Trend data": [], "Unit": []}
    for report in trend_rpt:
        timedel = starttime + pd.Timedelta(
            milliseconds=float(report.find("StdHdr").find("MsecTime").text)
        )
        t_dif = timedel - pd.Timestamp(report.find("StdHdr").find("DevDateTime").text)
        t_dif = t_dif.total_seconds()
        if np.abs(t_dif) > 10:
            timedel = pd.Timestamp(report.find("StdHdr").find("DevDateTime").text)

        for key in report.find("Trend"):
            # Check wheter this trend key is already known
            if key.tag not in trend_keys:
                logger.warning(
                    f"Unknown trend key {key} found! Check out structure and amend implementation."
                )
            else:
                # Check different trend keys. due to their changing structure we have to unpack the date for each key.
                if key.tag == "Nibp":
                    bp_types = ["Map", "Sys", "Dia"]
                    for bpt in bp_types:
                        # Check, if DataState is valid, This structure is similar for all trends

                        if (
                            report.find("Trend")
                            .find("Nibp")
                            .find(bpt)
                            .find("TrendData")
                            .find("DataStatus")
                            == "0"
                        ):
                            # Nibp has its own timestamps, overwrite timedel
                            nibp_timedel = pd.to_datetime(
                                report.find("Trend".find("Nibp").find("Time").text)
                            )
                            # Create new dict in trend_data and fill with first entry
                            if key + bpt not in trend_data:
                                trend_data[key + bpt] = {
                                    "timestamp": [],
                                    key.tag + bpt: [],
                                }
                                trend_data[key + bpt]["timestamp"].append(nibp_timedel)
                                trend_data[key + bpt][key + bpt].append(
                                    float(
                                        report.find("Trend")
                                        .find("Nibp")
                                        .find(bpt)
                                        .find("TrendData")
                                        .find("Val")
                                        .text
                                    )
                                )
                                trend_info["Trend data"].append(key + bpt)
                                trend_info["Unit"].append(
                                    report.find("Trend")
                                    .find("Nibp")
                                    .find(bpt)
                                    .find("TrendData")
                                    .find("Val")
                                    .attrib("Units")
                                )
                            # If Timestamp is not the same es before fill with new entries
                            elif (
                                not nibp_timedel
                                == trend_data[key + bpt]["timestamp"][-1]
                            ):
                                trend_data[key + bpt]["timestamp"].append(nibp_timedel)
                                trend_data[key + bpt][key + bpt].append(
                                    float(
                                        report.find("Trend")
                                        .find("Nibp")
                                        .find(bpt)
                                        .find("TrendData")
                                        .find("Val")
                                        .text
                                    )
                                )
                            else:
                                pass
                elif key.tag == "Spo2":
                    spotypes = ["SpCo", "SpMet", "SpHb"]
                    # First for Main Spo2 # Check, if DataState is valid
                    if (
                        report.find("Trend")
                        .find("Spo2")
                        .find("TrendData")
                        .find("DataStatus")
                        .text
                        == "0"
                    ):
                        if "Spo2" not in trend_data:
                            trend_data["Spo2"] = {"timestamp": [], "Spo2": []}
                            trend_info["Trend data"].append("Spo2")
                            trend_info["Unit"].append(
                                report.find("Trend")
                                .find("Spo2")
                                .find("TrendData")
                                .find("Val")
                                .attrib["Units"]
                            )
                        trend_data["Spo2"]["timestamp"].append(timedel)
                        trend_data["Spo2"]["Spo2"].append(
                            float(
                                report.find("Trend")
                                .find("Spo2")
                                .find("TrendData")
                                .find("Val")
                                .text
                            )
                        )
                    # For Spo2 subtypes
                    for typ in spotypes:
                        if (
                            report.find("Trend")
                            .find("Spo2")
                            .find(typ)
                            .find("TrendData")
                            .find("DataStatus")
                            == "valid"
                        ):
                            typname = "Spo2" + typ
                            if typname not in trend_data:
                                trend_info["Trend data"].append(typname)
                                trend_info["Unit"].append(
                                    report.find("Trend")
                                    .find("Spo2")
                                    .find(typ)
                                    .find("TrendData")
                                    .find("Val")
                                    .attrib["Units"]
                                )
                                trend_data["Spo2" + typ] = {
                                    "timestamp": [],
                                    typname: [],
                                }
                            trend_data[typname]["timestamp"].append(timedel)
                            trend_data[typname][typname].append(
                                float(
                                    report.find("Trend")
                                    .find("Spo2")
                                    .find(typ)
                                    .find("TrendData")
                                    .find("Val")
                                    .text
                                )
                            )

                elif key.tag == "Ibp":
                    bp_types = ["Map", "Sys", "Dia"]
                    for j in range(len(report.find("Trend").findall("Ibp"))):
                        for typ in bp_types:
                            if (
                                report.find("Trend")
                                .findall("Ibp")[j]
                                .find(typ)
                                .find("TrendData")
                                .find("DataStatus")
                                .text
                                == "0"
                            ):
                                typname = "Ch." + str(j) + " " + typ
                                if typname not in trend_data:
                                    trend_data[typname] = {
                                        "timestamp": [],
                                        typname: [],
                                    }
                                    trend_info["Trend data"].append(typname)
                                    trend_info["Unit"].append(
                                        report.find("Trend")
                                        .findall("Ibp")[j]
                                        .find(typ)
                                        .find("TrendData")
                                        .find("Val")
                                        .attrib["Units"]
                                    )
                                trend_data[typname]["timestamp"].append(timedel)
                                trend_data[typname][typname].append(
                                    float(
                                        report.find("Trend")
                                        .findall("Ibp")[j]
                                        .find(typ)
                                        .find("TrendData")
                                        .find("Val")
                                        .text
                                    )
                                )

                elif key.tag == "Temp":
                    for j in range(len(report.find("Trend").findall("Temp"))):
                        if (
                            report.find("Trend")
                            .findall("Temp")[j]
                            .find("TrendData")
                            .find("DataStatus")
                            == "0"
                        ):
                            typname = (
                                report.find("Trend").findall("Temp")[j]["SrcLabel"].text
                            )
                            if typname not in trend_data:
                                trend_data[typname] = {"timestamp": [], typname: []}
                                trend_info["Trend data"].append(typname)
                                trend_info["Unit"].append(
                                    report.find("Trend")
                                    .findall("Temp")[j]
                                    .find("TrendData")
                                    .find("Val")
                                    .attrib["Units"]
                                )
                            trend_data[typname]["timestamp"].append(timedel)
                            trend_data[typname][typname].append(
                                float(
                                    report.find("Trend")
                                    .findall("Temp")[j]
                                    .find("TrendData")
                                    .find("Val")
                                    .text
                                )
                            )
                elif key.tag in ["Hr", "Fico2", "Etco2", "Resp"]:
                    if (
                        report.find("Trend")
                        .find(key.tag)
                        .find("TrendData")
                        .find("DataStatus")
                        == "0"
                    ):
                        typname = key.tag
                        if typname not in trend_data:
                            trend_data[typname] = {"timestamp": [], typname: []}
                            trend_info["Trend data"].append(typname)
                            trend_info["Unit"].append(
                                report.find(
                                    "Trend".find(typname)
                                    .find("TrendData")
                                    .find("Val")
                                    .attrib["Units"]
                                )
                            )
                        trend_data[typname]["timestamp"].append(timedel)
                        trend_data[typname][typname].append(
                            float(
                                report.find(
                                    "Trend".find(key.tag)
                                    .find("TrendData")
                                    .find("Val")
                                    .text
                                )
                            )
                        )
                elif key.tag in ["LtaState", "LtaStateVal", "PatMode"]:
                    pass
                else:
                    logger.warning(f"Unknown trend key {key.tag} found.")

    for key in trend_data:
        trend_data[key] = pd.DataFrame(trend_data[key])
        trend_data[key].set_index("timestamp", inplace=True)

    return trend_data, trend_info


def read_zollxml_compr(cpr_compr, starttime):
    cpr_compr_keys = ["CompDisp", "RelVelocity", "CompRate"]
    compr_data = {
        "CompDisp": {"timestamp": [], "CompDisp": []},
        "RelVelocity": {"timestamp": [], "RelVelocity": []},
        "CompRate": {"timestamp": [], "CompRate": []},
    }
    compr_info = {
        "Trend data": ["CompDisp", "RelVelocity", "CompRate"],
        "Unit": ["cm", "Unknown", "Compressions /min"],
    }
    for report in cpr_compr:
        # There is a second timestamp in this key which differs from the used one about 0.05-0.5 s.
        # timestamp=pd.to_datetime(report['Timestamp'])
        timestamp = starttime + pd.Timedelta(
            milliseconds=float(report.find("StdHdr").find("MsecTime").text)
        )
        t_dif = timestamp - pd.Timestamp(report.find("StdHdr").find("DevDateTime").text)
        t_dif = t_dif.total_seconds()
        if np.abs(t_dif) > 10:
            timestamp = pd.Timestamp(report.find("StdHdr").find("DevDateTime").text)

        for typ in cpr_compr_keys:
            compr_data[typ]["timestamp"].append(timestamp)
            compr_data[typ][typ].append(float(report.find(typ).text))

    for key in compr_data:
        compr_data[key] = pd.DataFrame(compr_data[key])
        compr_data[key].set_index("timestamp", inplace=True)

    compr_data["CompDisp"] = compr_data["CompDisp"] * 2.54 / 1000

    return compr_data, compr_info


def find_wavetypevar(entry):
    if not isinstance(entry, str):
        entry = str(entry)
    if entry in zoll_wavetype_dict:
        return zoll_wavetype_dict[entry]
    else:
        return entry + ": Unknown Type"


# Method to read CXV-exported data


def read_zollcsv(filepath: Path | str, filepathxml: Path | str, quick=False):
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if isinstance(filepathxml, str):
        filepathxml = Path(filepathxml)
    filename = filepathxml.name

    # Read CSV file
    with open(filepath, newline="", encoding="utf-16") as csvfile:
        a = pd.read_csv(csvfile, delimiter=",")
    all_keys = [
        "Start",
        " Serial",
        " DeviceId",
        " RunNumber",
        " Date",
        " Real",
        " Elapsed",
        " X",
        " EcgVal",
        " EcgStatus",
        " CapnoVal",
        " CapnoStatus",
        " P1Val",
        " P1Status",
        " P2Val",
        " P2Status",
        " P3Val",
        " P3Status",
        " Spo2Val",
        " Spo2Status",
        " CprDepth",
        " CprFrequency",
        " CprStatus",
        " CprWaveVal",
        " FiltEcgVal",
        " FiltEcgStatus",
        " Ecg2Val",
        " Ecg2Status",
        " Ecg3Val",
        " Ecg3Status",
        " Ecg4Val",
        " Ecg4Status",
    ]
    # Warning if new key is found
    for key in list(a.columns):
        if key not in all_keys:
            logger.warning(f"New key {key} found!")

    filename = filepath.name

    unstatus_keys = [" CprWaveVal"]
    # status_keys=[' EcgVal',' CapnoVal',' P1Val',' P2Val',' P3Val',' Spo2Val',' FiltEcgVal',' Ecg2Val',' Ecg3Val',' Ecg4Val']
    # Achtung, Filtered ECG aus Liste genommen, Skalierungsprobleme Entdeckt!!!! (8.4.2022)
    status_keys = [
        " EcgVal",
        " CapnoVal",
        " P1Val",
        " P2Val",
        " P3Val",
        " Spo2Val",
        " Ecg2Val",
        " Ecg3Val",
        " Ecg4Val",
    ]

    zero_keys = [" CprDepth", " CprFrequency"]
    timestamps = np.array(
        pd.to_datetime(a[" Date"] + a[" Real"], format=" %m-%d-%Y %H:%M:%S.%f")
    )
    data = {}
    # Read keys without status-Values given+
    for key in unstatus_keys:
        series = np.array(a[key])
        timestamps1 = timestamps
        try:
            nonz = np.nonzero(series)[0]
            nonz1 = nonz[0]
            nonz2 = nonz[-1]
        except IndexError:
            nonz1 = 0
            nonz2 = -1
        series = series[nonz1:nonz2]
        timestamps1 = timestamps1[nonz1:nonz2]
        newname = new_names(key)
        df = {"timestamp": timestamps1, newname: series}
        df = pd.DataFrame(df)
        df.set_index("timestamp", inplace=True)
        data[newname] = df
    for key in status_keys:
        statuskey = key[: key.rfind("V")] + "Status"
        series = np.array(a[key])
        status = np.array(a[statuskey])
        series = series[status == 1]
        timestamps1 = timestamps[status == 1]
        try:
            nonz = np.nonzero(series)[0]
            nonz1 = nonz[0]
            nonz2 = nonz[-1]
        except IndexError:
            nonz1 = 0
            nonz2 = -1
        series = series[nonz1:nonz2]
        timestamps1 = timestamps1[nonz1:nonz2]

        newname = new_names(key)
        df = {"timestamp": timestamps1, newname: series}
        df = pd.DataFrame(df)
        df.set_index("timestamp", inplace=True)
        if df.empty:
            pass
        else:
            data[newname] = df

    for key in zero_keys:
        series = np.array(a[key])
        timestamps1 = timestamps[series != 0]
        series = series[series != 0]
        no_double_entries = (
            pd.Series(timestamps1[1:] - timestamps1[:-1]).dt.total_seconds() > 0.01
        )

        timestamps11 = timestamps1[:-1][no_double_entries]
        series1 = series[:-1][no_double_entries]
        timestamps1 = np.append(timestamps11, timestamps1[-1])
        series = np.append(series1, series[-1])

        newname = new_names(key)
        df = {"timestamp": timestamps1, newname: series}
        df = pd.DataFrame(df)
        df.set_index("timestamp", inplace=True)
        data[newname] = df
    data["CompDisp"]["CompDisp"] *= 2.54

    pat_dat = {}

    # Construct info dictionaries
    keys = list(data.keys())
    pat_dat["Keys"] = {"Unit": {}, "Type": {}, "Start": {}, "Stop": {}, "Length": {}}
    pat_dat["Load Log"] = {}
    for key in keys:
        pat_dat["Keys"]["Unit"][key] = "Unknown"
        pat_dat["Keys"]["Type"][key] = "Continous wave data"
        pat_dat["Keys"]["Start"][key] = data[key].first_valid_index()
        pat_dat["Keys"]["Stop"][key] = data[key].last_valid_index()
        pat_dat["Keys"]["Length"][key] = len(data[key][key])
    pat_dat["Keys"]["Unit"]["Displacement"] = "inch"
    for key in [
        "Pads",
        " Ecg2Val",
        " Ecg3Val",
        " Ecg4Val",
    ]:  #### !!!!!! 08.04. ,'Filtered ECG' entfernt
        pat_dat["Keys"]["Unit"][key] = "mV"

    for key in ["CompDisp", "CompRate"]:
        pat_dat["Keys"]["Type"][key] = "Trend data"
        pat_dat["Keys"]["Start"][key] = data[key].first_valid_index()
        pat_dat["Keys"]["Stop"][key] = data[key].last_valid_index()
        pat_dat["Keys"]["Length"][key] = len(data[key][key])
    pat_dat["Keys"]["Unit"]["CompDisp"] = "cm"
    pat_dat["Keys"]["Unit"]["CompRate"] = "1/min"

    pat_dat["Keys"] = (
        pd.DataFrame.from_dict(pat_dat["Keys"], orient="columns")
        .rename_axis(("Key"))
        .reset_index()
    )

    tree = ET.parse(filepathxml)
    root = tree.getroot()

    data["defibrillations"] = {
        "timestamp": [],
        "Nr": [],
        "DefaultEnergy": [],
        "DeliveredEnergy": [],
        "Impedance": [],
        "Current": [],
    }
    times = []
    ii = 1
    try:
        for elem in root[0].find("Defibrillator").find("Events").findall("Event"):
            if elem.attrib["type"] in ["14", "31", "32", "33"]:
                for subelem in elem.findall("Data"):
                    time = (
                        float(elem.attrib["second"]) + float(elem.attrib["ms"]) / 1000
                    )
                    if float(elem.attrib["second"]) not in times:
                        times.append(float(elem.attrib["second"]))
                        data["defibrillations"]["timestamp"].append(
                            timestamps[0] + pd.Timedelta(seconds=time)
                        )
                        data["defibrillations"]["Nr"].append(ii)
                        ii += 1
                    if float(elem.attrib["second"]) == times[-1]:
                        if elem.attrib["type"] == "14" and float(subelem.text) > 30:
                            data["defibrillations"]["DefaultEnergy"].append(
                                float(subelem.text)
                            )
                        elif elem.attrib["type"] == "31":
                            data["defibrillations"]["DeliveredEnergy"].append(
                                float(subelem.text)
                            )
                        elif elem.attrib["type"] == "32":
                            data["defibrillations"]["Impedance"].append(
                                float(subelem.text)
                            )
                        elif elem.attrib["type"] == "33":
                            data["defibrillations"]["Current"].append(
                                float(subelem.text)
                            )

    except AttributeError:
        pass
    delkeys = []
    for key in data["defibrillations"]:
        if (len(data["defibrillations"][key]) == 0) and key != "timestamp":
            delkeys.append(key)
    for key in delkeys:
        data["defibrillations"].pop(key)
    data["defibrillations"] = pd.DataFrame(data["defibrillations"])
    data["defibrillations"].set_index("timestamp", inplace=True)

    zoll_models = {
        "AF": ("ZOLL - R Series", "Monitor"),
        "AR": ("ZOLL - X Series", "Monitor"),
        "00": ("ZOLL - E Series", "Monitor"),
        "AA": ("ZOLL - AED Pro", "AED"),
    }
    pat_dat["Main data"] = {}
    pat_dat["Main data"]["File ID"] = filename

    ## Get Main Information about this case

    nr = str(root[0].find("Defibrillator").find("Serial").text)
    pat_dat["Main data"]["Serial Nr"] = nr
    pat_dat["Main data"]["Product Code"] = nr[:2]
    pat_dat["Main data"]["Model"] = zoll_models[nr[:2]][0]
    pat_dat["Main data"]["Defib Category"] = zoll_models[nr[:2]][1]

    pat_dat["Main data"]["Start time"] = pd.Timestamp(timestamps[0])
    pat_dat["Main data"]["Recording start"] = pd.Timestamp(timestamps[0])
    pat_dat["Main data"]["Recording end"] = pd.Timestamp(timestamps[-1])
    pat_dat["Main data"]["Recording length"] = pd.Timedelta(
        timestamps[-1] - timestamps[0]
    )
    pat_dat["Main data"] = pd.DataFrame.from_dict(pat_dat["Main data"], orient="index")

    return pat_dat, data


# Method which for each key in OLD CSV-Data the new key of JSON Data
def new_names(oldkey):
    if oldkey == " CprWaveVal":
        return "Displacement"
    elif oldkey == " CprDepth":
        return "CompDisp"
    elif oldkey == " CprFrequency":
        return "CompRate"
    elif oldkey == " EcgVal":
        return "Pads"
    elif oldkey == " Spo2Val":
        return "SpO2 %, Waveform"
    elif oldkey == " CapnoVal":
        return "CO2 mmHg, Waveform"
    elif oldkey == " FiltEcgVal":
        return "Filtered ECG"
    # elif oldkey==' Ecg2Val':
    #    return '12-Lead I'
    # elif oldkey==' Ecg3Val':
    #    return '12-Lead II'
    # elif oldkey==' Ecg4Val':
    #    return '12-Lead III'
    else:
        return oldkey


def read_lifepak(f_cont, f_cont_wv, f_cpre, further_files=[]):
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
    elem = root[1][0]
    tag = elem.tag
    pcstr = tag[tag.find("{") : tag.find("}") + 1]
    try:  # LOAD MAIN DATA
        for event in root[1].find(pcstr + "Events"):
            if event.attrib["Type"] == "PowerOn":
                starttime = pd.Timestamp(event.find(pcstr + "AdjustedTime").text)
        serial = (
            root1[1]
            .find(pcstr + "Record")
            .find(pcstr + "RecordingDevice")
            .find(pcstr + "SerialNumber")
            .text
        )
    except TypeError:
        event_container = root.find("Events")
        events = event_container.findall("Event")
        for event in events:
            if event.attrib["Type"] == "PowerOn":
                starttime = pd.Timestamp(event.find("AdjustedTime").text)
        device_container = root.find("Device")
        serial = device_container.find("SerialNumber").text
    # Make one reading function which loads for fielnames and pcstr, try different pctry via try except
    rec_start = ""
    rec_stop = ""

    channel_info = {}  # LOAD WAVEFORM DATA From Channels
    channel_data = {}
    try:
        datas = root1[1].find(pcstr + "Record").findall(pcstr + "RecordData")
    except AttributeError:
        Record_container = root1.find("Record")
        datas = Record_container.findall("RecordData")
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
            if ydat.find(pcstr + "Scale").text != "1":
                logger.warning(
                    f"Unknown scale found: {ydat.find(pcstr + 'Scale').text}"
                )
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

        events = root2[1].find(pcstr + "Events")
        if events is None:
            event_container = root2.find("Events")
            events = event_container.findall("Event")

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
                for defib in event.find(pcstr + "Values").findall(pcstr + "Value"):
                    if defib.attrib["Type"] == "DefibEnergy":
                        defib_en = np.append(defib_en, int(defib.text))
                    if defib.attrib["Type"] == "DefibVoltageCompImpedance":
                        defib_imp = np.append(defib_imp, int(defib.text))
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
                        if channel not in LP15_all_trend_data:
                            logger.warning(f"New trend data type found: {channel}")
                            LP15_all_trend_data.append(channel)
                        elif channel in LP15_collected_trend_data:
                            if channel not in trend_data:
                                trend_data[channel] = {"timestamp": [], channel: []}
                            if value.text:
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

    for key in del_keys:
        trend_data.pop(key)

    try:  # Compute number of defibrillations
        n_defib = int(root[1].find(pcstr + "TotalShocks").text)
    except AttributeError:
        n_defib = int(root.find(pcstr + "TotalShocks").text)

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
        "Serial Nr": serial,
        "Start time": starttime,
        "Recording start": pd.Timestamp(rec_start),
        "Recording end": pd.Timestamp(rec_stop),
        "Recording Length": pd.Timestamp(rec_stop) - pd.Timestamp(rec_start),
    }
    if serial[:4] == "LP15":
        case_info["Model"] = "LIFEPAK15"
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
        "Serial Nr": serial,
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


def read_corpuls(f_corpuls):  # Read Corpuls Data
    file_path = f_corpuls
    file = f_corpuls.name  # BDF-File
    file_number = f_corpuls.stem

    f = pyedflib.EdfReader(str(file_path))  # READ BDF File
    waveform_data = {}  # Empty dicts for data
    waveform_info = {}

    trend_data = {}

    trend_info = {}

    compression_data = {}
    compression_info = {}

    all_channels = f.getSignalHeaders()  # get channel headers in bdf
    starttime = f.getStartdatetime()  # get start time of bdf recording

    for k, channel in enumerate(all_channels):  # iterate htrough all channels
        key = channel["label"]
        sample_rate = channel["sample_rate"]
        dt = 1 / sample_rate
        unit = channel["dimension"]
        data = f.readSignal(k)
        lendata = len(data)
        t = np.arange(
            0, lendata * dt, dt
        )  # create appropriate time array for this channel
        timestamps = pd.Timestamp(starttime) + pd.to_timedelta(t, unit="s")
        if np.sum(np.abs(data)) > 0:  # ignore empty recorings # ignore if all data is 0
            if key == "CPR":
                # if channel is CPR, the discrete depth values of the chest compression device, are sampled
                # with high frequency. Thus ignore all 0 entries
                cond = np.where(data != 0)[0]
                timestamps = timestamps[cond]
                data = data[cond] * 2.54 / 1000  # convert inch to cm
                compression_data[key] = {"timestamp": timestamps, key: data}
                compression_info[key] = {
                    "Unit": "cm",
                    "Sample Rate": "",
                    "Type": "Compression data",
                    "Start": timestamps[0],
                    "Stop": timestamps[-1],
                    "Length": len(data),
                }

                compression_data[key] = pd.DataFrame().from_dict(compression_data[key])
                compression_data[key].set_index("timestamp", inplace=True)

            elif dt < 1:  # id sample time is <1s it is continuous data
                nonzeros = np.nonzero(data)
                i_min = nonzeros[0][0]
                i_max = nonzeros[0][-1]
                timestamps = timestamps[i_min:i_max]
                data = data[i_min:i_max]
                waveform_data[key] = {"timestamp": timestamps, key: data}
                waveform_info[key] = {
                    "Unit": unit,
                    "Sample Rate": sample_rate,
                    "Type": "Continous wave data",
                    "Start": timestamps[0],
                    "Stop": timestamps[-1],
                    "Length": lendata,
                }
                waveform_data[key] = pd.DataFrame().from_dict(waveform_data[key])
                waveform_data[key].set_index("timestamp", inplace=True)
                if key == "CO2":  # correct units
                    waveform_data[key][key] /= 10
                    waveform_info[key]["Unit"] = "mmHg"
                if key == "LT-ECG":
                    waveform_data[key][key] /= 100
                    waveform_info[key]["Unit"] = "mV"
            else:  # this is trend data
                d_data = data[1:] - data[:-1]
                cond = np.where(d_data != 0)[0] + 1
                # trend_data[key]={'time':timestamps[cond],
                #                   key:data[cond]}
                trend_data[key] = {"timestamp": timestamps, key: data}
                trend_info[key] = {
                    "Unit": unit,
                    "Sample Rate": sample_rate,
                    "Type": "Trend data",
                    "Start": timestamps[0],
                    "Stop": timestamps[-1],
                    "Length": lendata,
                }
                trend_data[key] = pd.DataFrame().from_dict(trend_data[key])
                trend_data[key].set_index("timestamp", inplace=True)

    f.close()

    if "LT-ECG" in waveform_data:
        waveform_data["LT-ECG"]["LT-ECG"] /= 100

    # CHECK for 12-lead ECG
    twelve_lead_ecg_time = np.array([])
    for file_in_dir in f_corpuls.parent.iterdir():
        if file_number in file_in_dir.name and "decg" in file_in_dir.name:
            ff = pyedflib.EdfReader(str(file_in_dir))
            # decg_all_channels = ff.getSignalHeaders()  # get channel headers in bdf
            decg_starttime = ff.getStartdatetime()  # get start time of bdf recording
            ff.close()
            twelve_lead_ecg_time = np.append(
                twelve_lead_ecg_time, pd.Timestamp(decg_starttime)
            )

    twelve_lead_ecg_dict = {
        "time_12_lead_ecg": pd.Series(np.sort(twelve_lead_ecg_time))
    }

    # Check whehter there is a directory with the name
    # it contains normally the events-db.bin and trend-db.bin sql databases
    dirpath = Path(file_path).parent

    if os.path.isdir(dirpath.joinpath(file_number)):
        event_dirpath = dirpath.joinpath(file_number)
        fd = event_dirpath.joinpath("events-db.bin")
        if os.path.isfile(fd):
            con = sqlite3.connect(fd)
            df = pd.read_sql("SELECT * FROM events", con)  # read the events data base

            defi_onoff = df[(df["id"] == 8192) & (df["sys_module"] == 0)]
            rec_start_num = (
                df[(df["id"] == 5632)]["rtd_ts"].iloc[0] / 1000
            )  # Fixed point for entire recording is recording start: It is equal to starttime from BDF files

            power_on_time_num = (
                defi_onoff[defi_onoff["param1"] == 1]["rtd_ts"].iloc[0] / 1000
                - rec_start_num
            )
            power_off_time_num = (
                defi_onoff[defi_onoff["param1"] == 0]["rtd_ts"].iloc[0] / 1000
                - rec_start_num
            )
            power_on_time = starttime + pd.Timedelta(power_on_time_num, unit="s")
            power_off_time = starttime + pd.Timedelta(power_off_time_num, unit="s")

            shock_df = df[(df["id"] == 2564)]  # that's the corpuls id for shocks
            shock_times_num = np.array(shock_df["rtd_ts"] / 1000) - rec_start_num
            shock_times = np.array(
                [starttime + pd.Timedelta(shock, unit="s") for shock in shock_times_num]
            )

            defib = pd.DataFrame()
            defib["timestamp"] = shock_times
            defib.set_index("timestamp", inplace=True)
            defib["Defibrillations_Nr"] = np.arange(1, len(shock_times) + 1, 1)

            defib_data = {"Defibrillations_Nr": defib}

            # defib = pd.DataFrame()
            # defib["timestamp"] = shock_times
            # defib.set_index('timestamp', inplace = True)
            # defib_data = {"defibrillations": defib}  # Create backgorund information

            fd2 = event_dirpath.joinpath("trend-db.bin")

            if os.path.isfile(fd2):  # chechk whether a trend data base is present
                con = sqlite3.connect(fd2)
                df_trend = pd.read_sql("SELECT * FROM vital", con)
                trend_data_2 = {}

                trend_keys = [
                    "hr_dem",
                    "hr_pam",
                    "pp",
                    "spo2",
                    "spo2_ss",
                    "nibp_sys",
                    "nibp_dia",
                    "nibp_mad",
                    "p1_sys",
                    "p1_dia",
                    "p1_mad",
                    "p2_sys",
                    "p2_dia",
                    "p2_mad",
                    "p3_sys",
                    "p3_dia",
                    "p3_mad",
                    "p4_sys",
                    "p4_dia",
                    "p4_mad",
                    "co2_ens",
                    "co2_ins",
                    "rr",
                    "t1",
                    "t2",
                    "spco",
                    "sphb",
                    "spmet",
                    "cpr_rate",
                    "nibp_qi",
                ]

                corpuls_bdf_sql_dict = {
                    "HR": "hr_dem",
                    "CPR_RATE": "cpr_rate",
                    "CO2_ENS": "co2_ens",
                    "CO2_RR": "rr",
                    "PP": "pp",
                    "SPO2": "spo2",
                    "SPO2_PI": "spo2_ss",
                    "NIBP_SYS": "nibp_sys",
                    "NIBP_DIA": "nibp_dia",
                    "NIBP_MAD": "nibp_mad",
                }

                for key in trend_keys:
                    if key in df_trend.columns:
                        cond = (
                            df_trend[key + "_inval"] != 1
                        )  # check whther data is valid
                        if cond.any():
                            data = df_trend[key][cond]
                            time = df_trend["ts"][cond] / 1000 - rec_start_num
                            timestamp = starttime + time.apply(
                                pd.to_timedelta, unit="s"
                            )
                            trend_data_2[key] = {}
                            trend_data_2[key]["timestamp"] = timestamp
                            trend_data_2[key][key] = data
                            if key in ["co2_ens", "co2_ins", "spo2", "spo2_ss"]:
                                trend_data_2[key][key] = data / 10
                            # elif key in []:
                            #     trend_data_2[key][key] = data/100
                            trend_data_2[key] = pd.DataFrame().from_dict(
                                trend_data_2[key]
                            )
                            trend_data_2[key].set_index("timestamp", inplace=True)

                del_keys = []  # if a channel is in both bdf end trend-db.bin take  the trend-db.bin values
                for key in trend_data:
                    if key in corpuls_bdf_sql_dict:
                        if corpuls_bdf_sql_dict[key] in trend_data_2:
                            del_keys.append(key)
                            trend_info[corpuls_bdf_sql_dict[key]] = {
                                "Unit": trend_info[key]["Unit"],
                                "Sample Rate": "",
                                "Type": "Trend data",
                                "Start": trend_data_2[
                                    corpuls_bdf_sql_dict[key]
                                ].first_valid_index(),
                                "Stop": trend_data_2[
                                    corpuls_bdf_sql_dict[key]
                                ].last_valid_index(),
                                "Length": len(
                                    trend_data_2[corpuls_bdf_sql_dict[key]].index
                                ),
                            }

                for key in del_keys:
                    trend_data.pop(key)
                    trend_info.pop(key)
            else:
                logger.warning(
                    "No trend-db.bin file found. Cannot read further information "
                    "about trend data."
                )
                trend_data_2 = {}

        else:
            logger.warning(
                "No events-db.bin file found. Cannot read information about "
                "shocks and trend data."
            )
            defib = pd.DataFrame()
            defib_data = {}  # Create backgorund information

            power_on_time = np.nan
            power_off_time = timestamps[-1]
            trend_data_2 = {}
    else:
        logger.warning(
            f"Subdirectory {dirpath / file_number} not found. Cannot "
            "read infos about shocks and trend data."
        )
        defib = pd.DataFrame()
        defib_data = {}  # Create backgorund information

        power_on_time = np.nan
        power_off_time = timestamps[-1]
        trend_data_2 = {}

    # twelve_channel_ecgs=[]
    # for file in glob(self.basepath+'/*-decg-*.bdf',recursive=True):
    #    f = pyedflib.EdfReader(file_path)
    #    decg_starttime=f.getStartdatetime()
    #    twelve_channel_ecgs.append(decg_starttime)

    channel_info = {**waveform_info, **trend_info, **compression_info}

    case_info = {
        "File ID": file[: file.rindex(".")],
        "Serial Nr": "",
        "Start time": power_on_time,
        "Recording start": starttime,
        "Recording end": power_off_time,
        "Recording Length": power_off_time - starttime,
    }
    case_info = pd.DataFrame(case_info, index=[0]).T
    channel_info = pd.DataFrame.from_dict(channel_info, orient="index")
    channel_info.reset_index(inplace=True)
    channel_info.rename(columns={"index": "Key"}, inplace=True)
    pat_dat = {"Main data": case_info, "Keys": channel_info, "Load Log": ""}
    data = {
        **waveform_data,
        **trend_data,
        **trend_data_2,
        **compression_data,
        **defib_data,
        **twelve_lead_ecg_dict,
    }
    return pat_dat, data
