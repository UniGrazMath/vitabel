"""Loading of Corpuls recordings (BDF export)."""

from __future__ import annotations

import logging
import os
import sqlite3

from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib

logger = logging.getLogger("vitabel")


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
        sample_rate = next(
            channel[k] for k in ("sample_rate", "sample_frequency") if k in channel
        )
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
        "Serial No": "",
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
