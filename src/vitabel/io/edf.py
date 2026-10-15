"""Loading of EDF+ files."""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib

from vitabel.timeseries import Channel, IntervalLabel, Label

logger = logging.getLogger("vitabel")


def read_edfplus(file_path: Path | str) -> tuple[list, list, dict]:
    """Read an EDF+ file and return channels, labels, and metadata.

    Parameters
    ----------
    file_path
        Path to the EDF+ file.

    Returns
    -------
    channels : list[Channel]
        List of Channel objects for each biosignal track.
    labels : list[Label]
        List of Label objects for annotations.
    metadata : dict
        Recording-level metadata extracted from the EDF+ header.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    f = pyedflib.EdfReader(str(file_path))

    try:
        # --- Extract recording-level metadata ---
        rec_startdatetime = f.getStartdatetime()
        rec_start = pd.Timestamp(rec_startdatetime)

        header = f.getHeader()
        recording_metadata = {
            "source": "EDF+",
            "category": "raw",
            "file": file_path.name,
            "technician": header.get("technician", ""),
            "recording_additional": header.get("recording_additional", ""),
            "patient_name": header.get("patientname", ""),
            "patient_code": header.get("patientcode", ""),
            "patient_additional": header.get("patient_additional", ""),
            "admincode": header.get("admincode", ""),
            "sex": header.get("sex", ""),
            "birthdate": str(header.get("birthdate", "")),
            "equipment": header.get("equipment", ""),
            "n_signals": f.signals_in_file,
            "recording_start": rec_start,
            "data_record_duration_s": f.datarecord_duration,
            "n_data_records": f.datarecords_in_file,
        }

        # --- Import biosignal channels ---
        channels = []
        n_signals = f.signals_in_file
        signal_headers = f.getSignalHeaders()

        for k in range(n_signals):
            sig_header = signal_headers[k]
            label_name = sig_header["label"].strip()

            # Skip EDF Annotations channel (EDF+ stores annotations in a
            # pseudo-signal channel named "EDF Annotations")
            if label_name in ("EDF Annotations", "BDF Annotations"):
                continue

            sample_rate = next(
                sig_header[key]
                for key in ("sample_rate", "sample_frequency")
                if key in sig_header
            )
            if sample_rate <= 0:
                continue

            data = f.readSignal(k)
            if data is None or len(data) == 0:
                continue

            n_samples = len(data)
            dt = 1.0 / sample_rate
            ti = np.arange(n_samples) * dt

            unit = sig_header.get("dimension", "").strip()
            transducer = sig_header.get("transducer", "").strip()
            prefilter = sig_header.get("prefilter", "").strip()
            physical_min = sig_header.get("physical_min", None)
            physical_max = sig_header.get("physical_max", None)
            digital_min = sig_header.get("digital_min", None)
            digital_max = sig_header.get("digital_max", None)
            reserved = sig_header.get("reserved", None)

            channel_metadata = {
                "source": "EDF+",
                "category": "raw",
                "units": unit,
                "source_device": header.get("equipment", ""),
                "transducer": transducer,
                "prefilter": prefilter,
                "sample_rate": sample_rate,
                "physical_min": physical_min,
                "physical_max": physical_max,
                "digital_min": digital_min,
                "digital_max": digital_max,
                "samples_in_datarecord": f.samples_in_datarecord(k),
                "reserved": reserved,
            }

            # Remove NaN values
            mask = ~np.isnan(data)
            ch = Channel(
                name=label_name,
                time_index=ti[mask],
                data=data[mask],
                time_start=rec_startdatetime,
                time_unit="s",
                metadata=channel_metadata,
            )
            channels.append(ch)

        # --- Import annotations as labels ---
        labels = []
        annotations = f.readAnnotations()

        if annotations is not None and len(annotations[0]) > 0:
            onset_times = np.array(annotations[0], dtype=float)  # in seconds
            durations = np.array(annotations[1], dtype=float)  # in seconds
            texts = annotations[2]

            # Separate interval annotations (duration > 0) from point annotations
            interval_onsets = []
            interval_offsets = []
            interval_texts = []
            point_onsets = []
            point_texts = []

            for onset, dur, text in zip(onset_times, durations, texts):
                text = str(text).strip()
                if not text:
                    continue
                if dur > 0:
                    interval_onsets.append(onset)
                    interval_offsets.append(onset + dur)
                    interval_texts.append(text)
                else:
                    point_onsets.append(onset)
                    point_texts.append(text)

            # Create point-annotation Label (text data at specific time points)
            if point_onsets:
                point_label = Label(
                    name="EDF+ Annotations",
                    time_index=np.array(point_onsets),
                    text_data=np.array(point_texts, dtype=str),
                    time_start=rec_startdatetime,
                    time_unit="s",
                    metadata={
                        "source": "EDF+",
                        "category": "annotation",
                        "annotation_type": "point",
                    },
                )
                labels.append(point_label)

            # Create interval-annotation Labels (one per unique annotation text)
            unique_interval_texts = sorted(set(interval_texts))
            for ann_text in unique_interval_texts:
                indices = [
                    i for i, t in enumerate(interval_texts) if t == ann_text
                ]
                onsets = np.array([interval_onsets[i] for i in indices])
                offsets = np.array([interval_offsets[i] for i in indices])

                interval_label = IntervalLabel(
                    name=f"EDF+ Interval: {ann_text}",
                    time_index=np.column_stack([onsets, offsets]),
                    text_data=np.array([ann_text] * len(onsets), dtype=str),
                    time_start=rec_startdatetime,
                    time_unit="s",
                    metadata={
                        "source": "EDF+",
                        "category": "annotation",
                        "annotation_type": "interval",
                        "durations": (offsets - onsets).tolist(),
                    },
                )
                labels.append(interval_label)

    finally:
        f.close()

    return channels, labels, recording_metadata
