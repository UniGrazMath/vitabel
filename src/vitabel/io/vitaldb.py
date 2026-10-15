"""Loading of VitalDB recordings (VitalRecorder ``*.vit`` files)."""

from __future__ import annotations

import logging

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from vitaldb.utils import VitalFile, Device

from vitabel.timeseries import Channel, Label

logger = logging.getLogger("vitabel")


def track_to_timeseries(
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
    metadata
        Metadata of the resulting channel or label; details of the track
        are added (the passed dictionary is not modified).
    """
    metadata = dict(metadata)
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


_track_to_timeseries = track_to_timeseries
"""Deprecated alias of :func:`track_to_timeseries`."""


def read_vitaldb_tracks(
    filepath: Path | str,
    metadata: dict | None = None,
) -> tuple[list[Channel], list[Label]]:
    """Read all tracks of a VitalDB recording.

    Parameters
    ----------
    filepath
        The path to the recording. Must be a ``*.vit`` file.
    metadata
        Metadata of the resulting channels and labels. Defaults to
        ``{"source": "VitalDB-Recording"}``.

    Returns
    -------
    The channels (waveform and numeric tracks) and the labels (string tracks)
    of the recording, in the order of the tracks in the file.
    """
    if metadata is None:
        metadata = {"source": "VitalDB-Recording"}

    filepath = Path(filepath)
    if not filepath.exists():
        raise ValueError(f"The path '{filepath}' does not exist.")
    vit = VitalFile(str(filepath))

    channels: list[Channel] = []
    labels: list[Label] = []
    for track_name in vit.get_track_names():
        series = track_to_timeseries(vit, track_name=track_name, metadata=metadata)
        if isinstance(series, Channel):
            channels.append(series)
        elif isinstance(series, Label):
            labels.append(series)
    return channels, labels


def assign_vitaldb_labels(
    labels: list[Label],
    channel_names: list[str],
) -> tuple[list[tuple[Label, list[str]]], list[Label]]:
    """Determine which VitalDB labels (string tracks) belong to which channels.

    The rules are based on observations, not on an in-depth study of the
    VitalDB format. Example channel names: ``PUMP2_RATE``, ``PUMP2_CONC``,
    ``PUMP2_PRES``; corresponding label names: ``PUMP2_DRUG``,
    ``PUMP2_RATE_UNIT``, ``PUMP2_CONC_UNIT``.

    1. If the label name without its trailing ``_<qualifier>`` is a channel
       name (``PUMP2_RATE_UNIT``), the label belongs to that channel.
    2. Otherwise, if it matches the name of several channels without their
       trailing qualifiers (``PUMP2_DRUG``), the label belongs to all of them.
    3. All remaining labels are global labels.

    Labels belonging to channels are renamed to their qualifier (e.g.,
    ``UNIT``, ``DRUG``).

    Parameters
    ----------
    labels
        The labels to assign.
    channel_names
        The names of the available channels.

    Returns
    -------
    A list of pairs ``(label, channel names)`` of labels to be attached to
    the channels with the given names, and the list of global labels.
    """
    split_label_names = {
        ln: ln.rsplit("_", 1)[0] if "_" in ln else ln
        for ln in [label.name for label in labels]
    }

    split_channel_names = defaultdict(set)
    for cn in channel_names:
        prefix = cn.rsplit("_", 1)[0] if "_" in cn else cn
        split_channel_names[prefix].add(cn)

    attachments: list[tuple[Label, list[str]]] = []
    global_labels: list[Label] = []
    for label in labels:
        short_ln = split_label_names[label.name]
        if short_ln in channel_names:  # rule 1
            label.rename(label.name.rsplit("_", 1)[1])
            attachments.append((label, [short_ln]))
        elif short_ln in split_channel_names:  # rule 2
            label.rename(label.name.rsplit("_", 1)[1])
            attachments.append((label, sorted(split_channel_names[short_ln])))
        else:  # rule 3
            global_labels.append(label)
    return attachments, global_labels
