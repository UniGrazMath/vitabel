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
