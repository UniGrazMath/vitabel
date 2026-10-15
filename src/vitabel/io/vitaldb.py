"""Conversion of VitalDB recordings (VitalRecorder files)."""

from __future__ import annotations

import logging

from datetime import datetime, timedelta, timezone

import pandas as pd

from vitaldb.utils import VitalFile, Device

from vitabel.timeseries import Channel, Label

logger = logging.getLogger("vitabel")


def _track_to_timeseries(
    vit: VitalFile,
    track_name: str,
    metadata: dict,
) -> Channel | Label | None:
    """Extracts a track from a vitaldb dataset and returns it as channel or label.

    Parameter
    ---------
    vit
        The vitaldb dataset to extract the track from.
    track_name
        The name of the track to extract.
    """
    if not isinstance(vit, VitalFile):
        raise ValueError("Not a vitals file.")
    if track_name not in vit.get_track_names():
        raise ValueError(f"'{track_name}' is not a track in the given vitals file.")

    (ti, dt), *_ = vit.get_samples(
        track_names=track_name,
        interval=None,
        return_datetime=False,
        return_timestamp=True,
    )
    unix_start = vit.dtstart
    # local time of the recording site (dgmt = UTC - local time in minutes);
    # independent of the time zone of the machine loading the file
    recording_timezone = timezone(timedelta(minutes=-vit.dgmt))
    rec_start = datetime.fromtimestamp(unix_start, tz=recording_timezone).replace(
        tzinfo=None
    )
    ti = ti - unix_start

    trk = vit.find_track(dtname=track_name)
    name = trk.name
    source_name = trk.dname
    metadata.update(
        {
            "source_device": trk.dname,
            "source_details": {
                "source_type": vit.devs.get(source_name, Device("")).type,
                "source_port": vit.devs.get(source_name, Device("")).port,
            },
            "units": trk.unit,
            "recording_details": {
                "sampel_rate": trk.srate,
                "offset": trk.offset,
                "gain": trk.gain,
            },
        }
    )
    plotstyle = _argb_int_to_plotstyle(trk.col)

    if trk.type in {1, 2}:  # 1: wav, 2: numerical (vitaldb specification)
        mask = ~pd.isna(dt)
        return Channel(
            name=name,
            time_index=ti[mask],
            data=dt[mask],
            time_start=rec_start,
            time_unit="s",
            plotstyle=plotstyle,
            metadata=metadata,
        )
    elif trk.type == 5:  # 5: str (vitaldb specification)
        mask = ~pd.isna(dt)
        return Label(
            name=name,
            time_index=ti[mask],
            text_data=dt[mask],
            time_start=rec_start,
            time_unit="s",
            plotstyle=plotstyle,
            metadata=metadata,
        )
    return None


def _argb_int_to_plotstyle(color_int: int):
    """Converts an ARGB color integer to a plotstyle dictionary with color as rgba.

    color_int
        The ARGB color integer to convert.
    """
    if not color_int == 4294967295 and isinstance(color_int, int):
        # not transparrent white
        a = (color_int >> 24) & 0xFF
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        rgba = (r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        return {"color": rgba}
    return None
