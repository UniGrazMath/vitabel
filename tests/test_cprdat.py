"""Tests for ``make_cprdat_analysis`` (German Resuscitation Registry).

The outputs for the ZOLL, LIFEPAK, and LUCAS test recordings are pinned by
golden files (``cprdat_*.json``): the registry processes them, so they must
only change on purpose.
"""

import copy
import json
import logging

import pandas as pd
import pytest

from vitabel import Label, Vitals

from tests.conftest import TEST_DATA_DIR
from tests.golden import assert_matches_golden, jsonable


@pytest.fixture(scope="module")
def cprdat_results(request) -> dict:
    """make_cprdat_analysis results of all test cases, computed once."""
    results = {}
    for case in ["zoll", "lifepak"]:
        recording = copy.deepcopy(request.getfixturevalue(f"{case}_vitals"))
        results[case] = recording.make_cprdat_analysis()
    lucas = Vitals()
    lucas.add_defibrillator_recording(TEST_DATA_DIR / "Lucas_file_Lucas.xml")
    results["lucas"] = lucas.make_cprdat_analysis()
    return results


@pytest.mark.golden
@pytest.mark.parametrize("case", ["zoll", "lifepak", "lucas"])
def test_cprdat_analysis_golden(cprdat_results, case):
    assert_matches_golden(f"cprdat_{case}", cprdat_results[case], mode="values")


def test_cprdat_analysis_from_compression_events():
    """Recording with single compression events (cc_depth) but without
    accelerometer: CC periods via cycle duration analysis (not covered by
    the golden cases)."""
    import numpy as np
    import pandas as pd

    from vitabel import Channel

    compressions = np.concatenate(
        [np.arange(0, 30, 0.6), np.arange(40, 70, 0.6), np.arange(90, 100, 0.55)]
    )
    recording = Vitals()
    recording.add_channel(
        Channel(
            "cc_depth",
            compressions,
            np.full(len(compressions), 5.2),
            time_start=pd.Timestamp("2024-01-01 12:00:00"),
            metadata={"Serial No": "123", "source": "ZOLL - X Series"},
        )
    )
    result = recording.make_cprdat_analysis()

    assert result["recording_info"]["serial_number"] == "123"
    assert result["recording_info"]["DACTYP2"] == "08"
    periods = result["chest_compression_analysis"]
    assert len(periods) == 3
    assert all(period["CC_depth"] == 5.2 for period in periods)
    assert result["registry_time_points"]["ZHDM"] == periods[0]["CC_start"]


@pytest.fixture
def short_zoll(zoll_vitals) -> Vitals:
    """The first 10 minutes of the ZOLL example recording (modifiable)."""
    start = zoll_vitals.rec_start()
    recording = zoll_vitals.truncate(start, start + pd.Timedelta(minutes=10))
    # channels without data in this window (Vitals.rec_start fails for those)
    for channel in [channel for channel in recording.channels if channel.is_empty()]:
        recording.remove_channel(channel=channel)
    return recording


def _dumps(result: dict) -> str:
    return json.dumps(jsonable(result), sort_keys=True)


def test_cprdat_analysis_adds_labels_to_recording(short_zoll):
    assert short_zoll.labels == []
    short_zoll.make_cprdat_analysis()
    assert sorted(label.name for label in short_zoll.labels) == [
        "cc_periods",
        "etco2_from_capnography",
        "rosc_decision_function",
        "rosc_prediction",
        "rosc_probability",
        "ventilations_from_capnography",
    ]


def test_cprdat_analysis_can_be_repeated(short_zoll, caplog):
    first = short_zoll.make_cprdat_analysis()
    label_names = sorted(label.name for label in short_zoll.labels)
    with caplog.at_level(logging.WARNING, logger="vitabel"):
        second = short_zoll.make_cprdat_analysis()
    assert "differs from the one computed" not in caplog.text
    assert sorted(label.name for label in short_zoll.labels) == label_names
    assert _dumps(second) == _dumps(first)


def test_cprdat_analysis_keeps_existing_labels(short_zoll, caplog):
    expected = copy.deepcopy(short_zoll).make_cprdat_analysis()
    # default method ("filter"); the registry analysis uses "threshold"
    short_zoll.compute_etco2_and_ventilations()
    existing = short_zoll.get_label("ventilations_from_capnography")

    with caplog.at_level(logging.WARNING, logger="vitabel"):
        result = short_zoll.make_cprdat_analysis()
    # similar enough: no warning
    assert "differs from the one computed" not in caplog.text
    assert short_zoll.get_label("ventilations_from_capnography") is existing
    assert _dumps(result) == _dumps(expected)


def test_cprdat_analysis_warns_about_strongly_differing_labels(short_zoll, caplog):
    start = short_zoll.rec_start()
    short_zoll.get_channel("capnography").attach_label(
        Label(
            "ventilations_from_capnography",
            time_index=[start + pd.Timedelta(seconds=s) for s in (10, 20)],
        )
    )
    with caplog.at_level(logging.WARNING, logger="vitabel"):
        short_zoll.make_cprdat_analysis()
    assert "label 'ventilations_from_capnography' that differs" in caplog.text
    assert "2 instead of" in caplog.text
