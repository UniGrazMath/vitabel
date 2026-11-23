"""Tests for functions of the io subpackage that are not covered by the
loader tests (see test_vitals.py and test_golden.py)."""

from vitabel import Label
from vitabel.io.vitaldb import assign_vitaldb_labels, read_vitaldb_tracks

from tests.builders import write_synthetic_vital_file


def _labels(*names):
    return [Label(name, time_index=[0.0], text_data=["x"]) for name in names]


def test_assign_vitaldb_labels():
    labels = _labels("PUMP2_RATE_UNIT", "PUMP2_DRUG", "EVENT", "SOME_NOTE")
    attachments, global_labels = assign_vitaldb_labels(
        labels, ["PUMP2_RATE", "PUMP2_PRES", "PUMP2_CONC", "HR"]
    )
    assert [(label.name, names) for label, names in attachments] == [
        ("UNIT", ["PUMP2_RATE"]),
        ("DRUG", ["PUMP2_CONC", "PUMP2_PRES", "PUMP2_RATE"]),
    ]
    assert [label.name for label in global_labels] == ["EVENT", "SOME_NOTE"]


def test_assign_vitaldb_labels_without_channels():
    labels = _labels("PUMP2_RATE_UNIT", "EVENT")
    attachments, global_labels = assign_vitaldb_labels(labels, [])
    assert attachments == []
    assert [label.name for label in global_labels] == ["PUMP2_RATE_UNIT", "EVENT"]


def test_read_vitaldb_tracks_does_not_modify_metadata(tmp_path):
    path = write_synthetic_vital_file(tmp_path / "synthetic.vit")
    metadata = {"source": "test"}
    channels, labels = read_vitaldb_tracks(path, metadata=metadata)
    assert metadata == {"source": "test"}
    assert [channel.name for channel in channels] == [
        "PUMP2_RATE",
        "PUMP2_CONC",
        "PUMP2_PRES",
        "PLETH",
    ]
    assert channels[0].metadata["source"] == "test"
    assert channels[0].metadata["units"] == "ml/h"
    assert channels[1].metadata["units"] == ""
    assert [label.name for label in labels] == [
        "PUMP2_RATE_UNIT",
        "PUMP2_DRUG",
        "EVENT",
        "SOME_NOTE",
    ]


def test_tabular_dict_import_with_anchored_and_interval_labels():
    import pandas as pd

    from vitabel import Channel, IntervalLabel, Vitals

    recording = Vitals()
    channel = Channel("channel", time_index=[0.0, 1.0, 2.0], data=[1.0, 2.0, 3.0])
    recording.add_channel(channel)
    recording.add_data_from_dict(
        {"anchored": {"timestamp": [0.5, 1.5], "data": [1, 2]}},
        datatype="label",
        anchored_channel=channel,
    )
    recording.add_data_from_dict(
        {
            "intervals": {
                "timestamp": [pd.Timedelta(0), pd.Timedelta(1, unit="s")],
                "data": [],
            },
            "empty": {"timestamp": [], "data": []},
        },
        datatype="interval_label",
        metadata={"source": "test"},
    )
    assert [label.name for label in channel.labels] == ["anchored"]
    assert [label.name for label in recording.data.global_labels] == ["intervals"]
    intervals = recording.get_label("intervals")
    assert isinstance(intervals, IntervalLabel)
    assert intervals.metadata == {"source": "test"}


def test_tabular_dict_import_is_incremental():
    import pytest

    from vitabel import Vitals

    recording = Vitals()
    with pytest.raises(ValueError, match="For key broken the value is not a dict"):
        recording.add_data_from_dict(
            {"ok": {"timestamp": [0.0, 1.0], "data": [1, 2]}, "broken": [1, 2]}
        )
    # time series before the invalid entry have been added
    assert recording.get_channel_names() == ["ok"]


def test_tabular_dataframe_interval_labels_not_implemented():
    import pandas as pd
    import pytest

    from vitabel.io.tabular import iter_series_from_dataframe

    df = pd.DataFrame({"a": [1.0, 2.0]}, index=[0.0, 1.0])
    with pytest.raises(NotImplementedError):
        list(iter_series_from_dataframe(df, datatype="interval_label"))


def test_read_csv_table_missing_file(tmp_path):
    import pytest

    from vitabel.io.tabular import read_csv_table

    with pytest.raises(FileNotFoundError):
        read_csv_table(tmp_path / "missing.csv")


def _zoll_xml_wave_record(wave_type, msec_time, n_samples=4, value=1):
    """A ``ContinWaveRec`` element of a ZOLL XML export."""
    import xml.etree.ElementTree as ET

    samples = " ".join([str(value)] * n_samples)
    return ET.fromstring(
        "<ContinWaveRec>"
        f"<StdHdr><MsecTime>{msec_time}</MsecTime></StdHdr>"
        "<WaveSetting><UnitsVal>mV</UnitsVal></WaveSetting>"
        f"<WaveRec><WaveType>{wave_type}</WaveType>"
        f"<FrameSize>{n_samples}</FrameSize><SampleTime>4000</SampleTime>"
        f"<UnpackedSamples>{samples}</UnpackedSamples>"
        "<SampleStatus>0</SampleStatus></WaveRec>"
        "</ContinWaveRec>"
    )


def test_read_zollxml_wave_type_with_single_record():
    """A wave type with a single record has no recording interval and is
    skipped, like in ZOLL JSON exports (it raised a TypeError, #64); the other
    wave types are read as without it."""
    import pandas as pd

    from vitabel.io.zoll import read_zollxml_contin

    start = pd.Timestamp("2024-01-01")
    pads = [_zoll_xml_wave_record("1", msec) for msec in (0, 16, 32, 48)]
    single = _zoll_xml_wave_record("2", 0, value=7)

    data = read_zollxml_contin(pads + [single], start)[0]
    expected = read_zollxml_contin(pads, start)[0]

    pd.testing.assert_frame_equal(data["Pads"], expected["Pads"])
    assert len(data["Pads"]) == 4
    assert data["12-Lead I"].empty
