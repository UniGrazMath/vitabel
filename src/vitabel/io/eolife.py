"""Loading of EOlife ventilatory feedback exports (CSV)."""

from __future__ import annotations

import logging

from pathlib import Path

import pandas as pd

from vitabel.typing import EOLifeRecord

logger = logging.getLogger("vitabel")


def read_eolife_export(eolife_filepath: Path) -> EOLifeRecord:
    # EOlife export data are written in csv format
    # the first three line contain header information on the recording
    # line 4 contains the heder for the recording data
    # the rest of the file contains the data

    header_info = pd.read_csv(
        eolife_filepath,
        sep=";",
        header=0,
        nrows=1,
        encoding="latin1",
        na_values=["NA", "Na"],
        usecols=[
            "Date",
            "Time",
            "Patient Type",
            "Patient Size",
            "Mode",
            "Training",
            "FrequencyMode",
            "Leakage alarm",
            "EOlife",
        ],
        dtype=str,
    )

    header_info["recording_start"] = pd.to_datetime(
        header_info["Date"] + " " + header_info["Time"], format="%d/%m/%y %H:%M:%S"
    )

    header_info.drop(columns=["Date", "Time"], inplace=True)
    header_info.rename(
        columns={
            "EOlife": "serial_number",
        },
        inplace=True,
    )

    recording_start = header_info["recording_start"].iloc[0]
    metadata = (
        header_info[
            [
                "Patient Type",
                "Patient Size",
                "Mode",
                "Training",
                "FrequencyMode",
                "Leakage alarm",
            ]
        ]
        .iloc[0]
        .to_dict()
    )

    df_data = pd.read_csv(
        eolife_filepath,
        sep=";",
        decimal=",",
        skiprows=3,
        header=0,
        encoding="latin1",
        na_values=["NA", "Na"],
        usecols=[
            "Cycle number",
            "Time (hh:mm:ss:SS)",
            "Ti (ms)",
            "Te (ms)",
            "Tp (ms)",
            "Freq (/min)",
            "Vi (mL)",
            "Vt (mL)",
            "Leakage (mL)",
            "Leakage ratio (%)",
        ],
        dtype={
            "Cycle number": int,
            "Time (hh:mm:ss:SS)": str,
            "Ti (ms)": "Int64",
            "Te (ms)": "Int64",
            "Tp (ms)": "Int64",
            "Freq (/min)": "Int64",
            "Vi (mL)": "Int64",
            "Vt (mL)": "Int64",
            "Leakage (mL)": "Int64",
        },
        converters={
            "Leakage ratio (%)": lambda x: (
                float(x) / 100 if x not in ["Na", "NA"] else None
            )
        },
    )

    df_data.rename(
        columns={"Leakage ratio (%)": "Leakage ratio"},
        inplace=True,
    )

    df_data["timedelta"] = pd.to_timedelta(
        df_data["Time (hh:mm:ss:SS)"].str.replace(
            r"(\d{2}):(\d{2}):(\d{2}):(\d{2})",
            r"\1:\2:\3.\4",
            regex=True,
        )
    )
    df_data.set_index("timedelta", inplace=True)
    df_data.drop(columns=["Time (hh:mm:ss:SS)"], inplace=True)

    # Step 1: Extract units and create a rename + units mapping
    rename_dict = {}
    units_dict = {}

    for col in df_data.columns:
        if "(" in col and ")" in col:
            name_part = col[: col.find("(")].strip()
            unit_part = col[col.find("(") + 1 : col.find(")")].strip()
            rename_dict[col] = name_part
            units_dict[name_part] = unit_part

    # Step 2: Rename columns
    df_data.rename(columns=rename_dict, inplace=True)

    column_metadata = {k: {"units": v} for k, v in units_dict.items()}
    metadata = metadata | {"category": "raw", "source": "EOlife"}

    return EOLifeRecord(
        data=df_data,
        recording_start=recording_start,
        metadata=metadata,
        column_metadata=column_metadata,
    )
