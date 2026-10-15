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
