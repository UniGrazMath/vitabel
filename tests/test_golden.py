"""Golden (snapshot) tests for data loading and analysis.

These tests freeze the current behavior of the loaders and analysis routines
so that refactorings can be checked against it. See ``tests/golden.py`` for
details, and run ``uv run python scripts/regenerate_golden.py`` to regenerate
the golden files (only if a change in behavior is intended!).

Further golden assertions are part of the analysis tests in
``tests/test_vitals.py`` (to avoid repeating expensive computations).
"""

import copy

import numpy as np
import pandas as pd
import pytest

from vitabel import Channel, IntervalLabel, Label, Vitals
from vitabel.utils import loading

from tests.builders import write_synthetic_zoll_csv_pair, write_test_edfplus
from tests.conftest import EXAMPLE_DATA_DIR, TEST_DATA_DIR
from tests.golden import (
    assert_matches_golden,
    summarize_series,
    summarize_vitals,
)

pytestmark = pytest.mark.golden


def _labels_summary(vitals: Vitals) -> list[dict]:
    return [summarize_series(label) for label in vitals.labels]


# ---------------------------------------------------------------------------
# Loaders (exact)
# ---------------------------------------------------------------------------


def test_golden_zoll_json(zoll_vitals):
    assert_matches_golden("loader_zoll_json", summarize_vitals(zoll_vitals))


def test_golden_zoll_csv_synthetic(tmp_path):
    csv_path, xml_path = write_synthetic_zoll_csv_pair(tmp_path / "zoll_case")
    recording = Vitals()
    recording.add_defibrillator_recording(csv_path)
    assert_matches_golden("loader_zoll_csv_synthetic", summarize_vitals(recording))


def test_golden_lifepak15(lifepak_vitals):
    assert_matches_golden("loader_lifepak15", summarize_vitals(lifepak_vitals))


@pytest.mark.parametrize("case", ["Lucas_file", "empty_lucas"])
def test_golden_lucas(case):
    recording = Vitals()
    recording.add_defibrillator_recording(TEST_DATA_DIR / f"{case}_Lucas.xml")
    assert_matches_golden(f"loader_lucas_{case}", summarize_vitals(recording))


def test_golden_vitaldb():
    recording = Vitals()
    recording.add_vital_db_recording(TEST_DATA_DIR / "vital_file.vit")
    assert_matches_golden("loader_vitaldb", summarize_vitals(recording))


def test_golden_edfplus(edf_sample_path):
    recording = Vitals()
    recording.add_edfplus(edf_sample_path)
    assert_matches_golden("loader_edfplus_sample", summarize_vitals(recording))


def test_golden_edfplus_interval_annotations(tmp_path):
    path = tmp_path / "interval_annotations.edf"
    write_test_edfplus(path)
    recording = Vitals()
    recording.add_edfplus(path)
    assert_matches_golden("loader_edfplus_intervals", summarize_vitals(recording))


def test_golden_eolife():
    recording = Vitals()
    recording.add_ventilatory_feedback(
        TEST_DATA_DIR / "sample_signals" / "eolife_test_file_with_empty_values.csv"
    )
    assert_matches_golden("loader_eolife", summarize_vitals(recording))


def test_golden_eolife_record():
    record = loading.read_eolife_export(
        TEST_DATA_DIR / "sample_signals" / "eolife_test_file_with_empty_values.csv"
    )
    assert_matches_golden("loader_eolife_record", record)


def _volucapno_recording() -> Vitals:
    """Flow, airway pressure, and capnography from the animal lab example."""
    recording = Vitals()
    common = dict(
        time_start=pd.Timestamp(1970, 1, 1, 0, 0, 0),
        time_unit="ms",
        metadata={"source": "volucapno"},
    )
    recording.add_data_from_csv(
        EXAMPLE_DATA_DIR / "flow.csv.bz2", index_col="timestamp", **common
    )
    recording.add_data_from_csv(
        EXAMPLE_DATA_DIR / "capno.csv.bz2", index_col="Timestamp", **common
    )
    recording.add_data_from_csv(
        EXAMPLE_DATA_DIR / "p1.csv.bz2",
        index_col=0,
        names=["airway_pressure", "temperature_1"],
        **common,
    )
    return recording


@pytest.fixture(scope="module")
def volucapno_recording() -> Vitals:
    return _volucapno_recording()


def test_golden_csv(volucapno_recording):
    assert_matches_golden("loader_csv_volucapno", summarize_vitals(volucapno_recording))


def test_golden_dataframe_and_dict_import():
    rng = np.random.default_rng(1234)
    recording = Vitals()

    absolute = pd.DataFrame(
        {
            "signal a": rng.normal(size=50),
            "signal b": np.where(rng.random(50) < 0.2, np.nan, rng.normal(size=50)),
        },
        index=pd.date_range("2024-03-01 08:00:00", periods=50, freq="250ms"),
    )
    recording.add_data_from_DataFrame(
        absolute,
        metadata={"source": "synthetic"},
        column_metadata={"signal a": {"unit": "mV"}},
    )
    recording.add_data_from_DataFrame(
        absolute[["signal b"]].rename(columns={"signal b": "label b"}),
        datatype="label",
    )

    relative = pd.DataFrame(
        {"relative signal": rng.normal(size=20)},
        index=np.arange(20) * 0.5,
    )
    relative_recording = Vitals()
    relative_recording.add_data_from_DataFrame(relative, time_unit="s")

    recording.add_data_from_dict(
        {
            "dict channel": {
                "timestamp": pd.date_range("2024-03-01 08:00:00", periods=5, freq="1s"),
                "data": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
        },
        metadata={"source": "dict"},
    )
    recording.add_data_from_dict(
        {
            "dict label": {
                "timestamp": pd.date_range("2024-03-01 08:00:01", periods=3, freq="2s"),
                "data": [10, 20, 30],
            },
        },
        datatype="label",
    )

    assert_matches_golden(
        "loader_dataframe_and_dict",
        {
            "absolute": summarize_vitals(recording),
            "relative": summarize_vitals(relative_recording),
        },
    )


# ---------------------------------------------------------------------------
# Analysis (values)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zoll_analysed(zoll_vitals) -> Vitals:
    """ZOLL example after the standard analysis pipeline."""
    recording = copy.deepcopy(zoll_vitals)
    recording.compute_etco2_and_ventilations()
    recording.cycle_duration_analysis()
    recording.predict_circulation()
    return recording


def test_golden_zoll_analysis_pipeline(zoll_analysed):
    assert_matches_golden(
        "analysis_zoll_pipeline_labels", _labels_summary(zoll_analysed), mode="values"
    )


def test_golden_zoll_shocks(zoll_analysed):
    assert_matches_golden("analysis_zoll_shocks", zoll_analysed.shocks(), mode="values")


def test_golden_zoll_filter_and_threshold(zoll_analysed):
    cc_periods = zoll_analysed.get_label("cc_periods")
    etco2 = zoll_analysed.get_label("etco2_from_capnography")
    filtered = {
        "etco2_during_cc": zoll_analysed.filter_by_intervallabel(etco2, cc_periods),
        "etco2_outside_cc": zoll_analysed.filter_by_intervallabel(
            etco2, cc_periods, invert=True
        ),
        "capnography_during_cc": zoll_analysed.filter_by_intervallabel(
            zoll_analysed.get_channel("capnography"), cc_periods
        ),
    }
    assert_matches_golden(
        "analysis_zoll_filter_by_intervallabel",
        {name: summarize_series(series) for name, series in filtered.items()},
        mode="values",
    )
    assert_matches_golden(
        "analysis_zoll_area_under_threshold",
        {
            "etco2_below_35": zoll_analysed.area_under_threshold(
                "etco2_from_capnography", threshold=35
            ),
            "spo2_below_90": zoll_analysed.area_under_threshold("spo2", threshold=90),
        },
        mode="values",
    )


def test_golden_zoll_cc_periods_from_acceleration(zoll_vitals_copy):
    zoll_vitals_copy.find_CC_periods_acc()
    assert_matches_golden(
        "analysis_zoll_cc_periods_acc",
        _labels_summary(zoll_vitals_copy),
        mode="values",
    )


def test_golden_respiratory_phases_and_volumes(volucapno_recording):
    flow = volucapno_recording.get_channel("airflow")
    pressure = volucapno_recording.get_channel("airway_pressure")
    start = flow.first_entry + pd.Timedelta(minutes=2)
    stop = start + pd.Timedelta(minutes=2)

    recording = Vitals()
    flow = flow.truncate(start, stop)
    pressure = pressure.truncate(start, stop)
    recording.add_channel(flow)
    recording.add_channel(pressure)

    landmarks = recording.compute_respiratory_phases(
        flow,
        pressure,
        add_intermediate_channels=True,
        return_landmarks=True,
    )
    assert_matches_golden(
        "analysis_respiratory_phases",
        {"landmarks": landmarks, "recording": summarize_vitals(recording)},
        mode="values",
    )

    recording.compute_ventilation_volumes()
    assert_matches_golden(
        "analysis_ventilation_volumes", summarize_vitals(recording), mode="values"
    )


def test_golden_label_and_channel_constructs():
    """A small hand-written collection exercising various series types; guards
    the summary helpers themselves against accidental changes."""
    channel = Channel(
        "channel",
        time_index=[0.0, 0.5, 1.0, 2.0],
        data=[1.0, np.nan, 3.0, 4.0],
        time_start=pd.Timestamp("2024-01-01 00:00:00"),
        offset=0.25,
    )
    label = Label(
        "label",
        time_index=[
            pd.Timestamp("2024-01-01 00:00:00.1"),
            pd.Timestamp("2024-01-01 00:00:01.5"),
        ],
        data=[np.nan, 2.0],
        text_data=["a", None],
        plot_type="combined",
    )
    channel.attach_label(label)
    interval_label = IntervalLabel(
        "interval label",
        time_index=[
            pd.Timestamp("2024-01-01 00:00:00.5"),
            pd.Timestamp("2024-01-01 00:00:01.5"),
        ],
        data=[3.0],
    )
    recording = Vitals()
    recording.add_channel(channel)
    recording.add_global_label(interval_label)
    assert_matches_golden("constructs_small_collection", summarize_vitals(recording))
