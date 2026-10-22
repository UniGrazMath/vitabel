"""Helpers that build (synthetic) test data files."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyedflib


def write_test_edfplus(path: Path):
    writer = pyedflib.EdfWriter(str(path), 1, file_type=pyedflib.FILETYPE_EDFPLUS)
    writer.setStartdatetime(datetime(2024, 1, 1, 12, 0, 0))
    writer.setSignalHeader(
        0,
        {
            "label": "sig",
            "dimension": "uV",
            "sample_frequency": 1,
            "physical_min": -1,
            "physical_max": 1,
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "",
            "prefilter": "",
        },
    )
    writer.writeSamples([np.array([0.0, 1.0] * 6, dtype=float)])
    writer.writeAnnotation(1.0, 2.5, "artifact")
    writer.writeAnnotation(5.0, 1.0, "artifact")
    writer.writeAnnotation(8.0, 0.5, "noise")
    writer.close()


def write_synthetic_zoll_csv_pair(base_path: Path) -> tuple[Path, Path]:
    csv_path = base_path.with_name(base_path.name + "_ecg.txt")
    xml_path = base_path.with_suffix(".xml")

    rows = []
    for i in range(3):
        rows.append(
            {
                "Start": 0,
                " Serial": "AR12345",
                " DeviceId": 1,
                " RunNumber": 1,
                " Date": " 01-01-2024",
                " Real": f" 12:00:0{i}.000",
                " Elapsed": i,
                " X": 0,
                " EcgVal": i + 1,
                " EcgStatus": 1,
                " CapnoVal": 10 + i,
                " CapnoStatus": 1,
                " P1Val": 0,
                " P1Status": 0,
                " P2Val": 0,
                " P2Status": 0,
                " P3Val": 0,
                " P3Status": 0,
                " Spo2Val": 95 + i,
                " Spo2Status": 1,
                " CprDepth": 0,
                " CprFrequency": 0,
                " CprStatus": 0,
                " CprWaveVal": 0,
                " FiltEcgVal": 0,
                " FiltEcgStatus": 0,
                " Ecg2Val": 0,
                " Ecg2Status": 0,
                " Ecg3Val": 0,
                " Ecg3Status": 0,
                " Ecg4Val": 0,
                " Ecg4Status": 0,
            }
        )

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-16")
    xml_path.write_text(
        "<Root><Record><Defibrillator><Serial>AR12345</Serial></Defibrillator></Record></Root>",
        encoding="utf-8",
    )
    return csv_path, xml_path
