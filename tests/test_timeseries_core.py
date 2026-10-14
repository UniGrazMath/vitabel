"""Characterization tests for the time-series core.

These tests pin the current behavior of time parsing, time conversion,
hashing, windowing, and mutation of time series, in preparation for the
performance refactoring of ``vitabel.timeseries``. Most tests compare against
the frozen reference implementation in ``tests/reference_impl.py``
(differential testing on hand-picked and randomized inputs), the remaining
ones against golden files or explicitly pinned values.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from vitabel import Channel, IntervalLabel, Label
from vitabel.timeseries import TimeSeriesBase

from tests import reference_impl
from tests.golden import assert_matches_golden, summarize_series

ABSOLUTE_START = pd.Timestamp("2024-01-01 12:00:00")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _construction_cases() -> dict[str, dict]:
    """Inputs for TimeSeriesBase(...). Built freshly for each test."""
    datetimes = pd.date_range("2024-01-01 12:00:00", periods=4, freq="250ms")
    return {
        "float_list": dict(time_index=[0.0, 0.5, 1.25, 3.0]),
        "int_list": dict(time_index=[0, 1, 2, 3]),
        "float_array_with_nan": dict(time_index=np.array([0.0, np.nan, 2.0, 3.5])),
        "int_array": dict(time_index=np.arange(5)),
        "float_array_ms": dict(time_index=np.array([0.0, 1.5, 3.0]), time_unit="ms"),
        "float_array_min": dict(time_index=np.array([0.0, 1.5, 3.0]), time_unit="min"),
        "numeric_offset": dict(time_index=[0.0, 1.0], offset=2.5),
        "timedelta_offset": dict(time_index=[0.0, 1.0], offset=pd.Timedelta("-1s")),
        "relative_with_time_start": dict(
            time_index=[0.0, 1.0, 2.0], time_start=ABSOLUTE_START
        ),
        "relative_with_string_time_start": dict(
            time_index=[0.0, 1.0], time_start="2024-01-01 12:00:00"
        ),
        "relative_with_tzaware_time_start": dict(
            time_index=[0.0, 1.0],
            time_start=pd.Timestamp("2024-01-01 12:00:00", tz="Europe/Vienna"),
        ),
        "datetime64_ns_array": dict(time_index=datetimes.to_numpy()),
        "datetime64_ms_array": dict(time_index=datetimes.to_numpy().astype("M8[ms]")),
        "datetime64_s_array": dict(
            time_index=np.array(["2024-01-01T12:00:00", "2024-01-01T12:00:05"], "M8[s]")
        ),
        "datetime_index": dict(time_index=datetimes),
        "datetime_index_tzaware": dict(
            time_index=datetimes.tz_localize("Europe/Vienna")
        ),
        "timestamp_list": dict(time_index=list(datetimes)),
        "datetime64_with_offset": dict(time_index=datetimes.to_numpy(), offset=1.0),
        "datetime64_with_nat_first": dict(
            time_index=np.array(
                ["NaT", "2024-01-01T12:00:00", "2024-01-01T12:00:01"], "M8[ns]"
            )
        ),
        "timedelta64_ns_array": dict(
            time_index=pd.to_timedelta([0, 1, 2], unit="s").to_numpy()
        ),
        "timedelta64_ms_array": dict(
            time_index=np.array([0, 1500, 3000], dtype="m8[ms]")
        ),
        "timedelta_index": dict(
            time_index=pd.timedelta_range("1s", periods=3, freq="1s")
        ),
        "timedelta_list": dict(time_index=[pd.Timedelta(seconds=s) for s in (0, 1, 2)]),
        # NOTE: strings in mixed formats currently fail to parse (pinned)
        "timestamp_strings": dict(
            time_index=["2024-01-01 12:00:00", "2024-01-01 12:00:01.5"]
        ),
        "timedelta_strings": dict(time_index=["0s", "1.5s", "00:01:00"]),
        "timestamp_strings_uniform": dict(
            time_index=["2024-01-01 12:00:00", "2024-01-01 12:00:01"]
        ),
        "timedelta_strings_uniform": dict(time_index=["0s", "1s", "2.5s"]),
        "none_first": dict(time_index=[None, 1.0, 2.0]),
        "nat_first_timestamps": dict(
            time_index=[pd.NaT, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        ),
        "empty_list": dict(time_index=[]),
        "empty_float_array": dict(time_index=np.array([], dtype=float)),
        "empty_datetime64_array": dict(time_index=np.array([], dtype="M8[ns]")),
        "empty_timedelta64_array": dict(time_index=np.array([], dtype="m8[ns]")),
        "all_nan": dict(time_index=np.array([np.nan, np.nan])),
        "series_of_floats": dict(time_index=pd.Series([0.0, 1.0, 2.0])),
        # error cases
        "mixed_types": dict(time_index=[pd.Timestamp("2024-01-01"), 1.0]),
        "mixed_timestamp_timedelta": dict(
            time_index=[pd.Timestamp("2024-01-01"), pd.Timedelta("1s")]
        ),
        "absolute_with_time_start": dict(
            time_index=list(datetimes), time_start=ABSOLUTE_START
        ),
        "unparsable_strings": dict(time_index=["foo", "bar"]),
        "two_dimensional": dict(time_index=np.zeros((2, 2))),
        "unsupported_type": dict(time_index=[object(), object()]),
    }


CONSTRUCTION_CASES = list(_construction_cases())


def _outcome(func, *args, **kwargs):
    try:
        return "ok", func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - we compare the exception
        return "error", (type(exc), str(exc))


@pytest.mark.parametrize("case", CONSTRUCTION_CASES)
def test_construction_matches_reference(case):
    expected_status, expected = _outcome(
        reference_impl.parse_time_index, **_construction_cases()[case]
    )
    status, series = _outcome(TimeSeriesBase, **_construction_cases()[case])
    assert status == expected_status, (series, expected)
    if status == "error":
        assert series == expected
        return

    ref_index, ref_start, ref_offset, ref_unit = expected
    # NOTE: the type is not always a TimedeltaIndex; e.g., a pandas.Series
    # passed as time_index currently results in a Series (pinned here).
    assert type(series.time_index) is type(ref_index)
    assert series.time_index.dtype == ref_index.dtype
    assert series.time_index.equals(ref_index)
    _assert_same_time_start(series.time_start, ref_start)
    assert series.offset == ref_offset
    assert series.time_unit == ref_unit


def _assert_same_time_start(time_start, expected):
    assert (time_start is None) == (expected is None)
    if expected is None:
        return
    if pd.isna(expected):
        # NOTE: a leading NaT currently results in time_start = NaT (pinned here)
        assert pd.isna(time_start)
        return
    assert time_start == expected
    assert time_start.tz is None


def test_construction_golden():
    """Pins the resulting time index (including its unit), time start and
    offset for all non-failing construction cases."""
    results = {}
    for case, kwargs in _construction_cases().items():
        status, series = _outcome(TimeSeriesBase, **kwargs)
        if status == "error":
            results[case] = {"error": series[0].__name__, "message": series[1]}
        else:
            results[case] = {
                "time_index": [str(value) for value in series.time_index],
                "time_index_dtype": str(series.time_index.dtype),
                "time_start": None
                if series.time_start is None
                else str(series.time_start),
                "offset": str(series.offset),
                "time_unit": series.time_unit,
            }
    assert_matches_golden("timeseries_construction", results)


@pytest.mark.parametrize("n_samples", [10, 10_000])
def test_construction_matches_reference_randomized(n_samples):
    rng = np.random.default_rng(n_samples)
    seconds = np.sort(rng.uniform(0, 3600, n_samples))
    inputs = [
        dict(time_index=seconds),
        dict(time_index=seconds, time_unit="ms"),
        dict(
            time_index=(ABSOLUTE_START + pd.to_timedelta(seconds, unit="s")).to_numpy()
        ),
        dict(time_index=pd.to_timedelta(seconds, unit="s").to_numpy()),
        dict(time_index=np.round(seconds * 1000).astype(np.int64), time_unit="ms"),
    ]
    for kwargs in inputs:
        ref_index, ref_start, ref_offset, ref_unit = reference_impl.parse_time_index(
            **kwargs
        )
        series = TimeSeriesBase(**kwargs)
        assert series.time_index.dtype == ref_index.dtype
        assert series.time_index.equals(ref_index)
        _assert_same_time_start(series.time_start, ref_start)
        assert series.offset == ref_offset


# ---------------------------------------------------------------------------
# numeric_time and data_hash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CONSTRUCTION_CASES)
@pytest.mark.parametrize("time_unit", ["s", "ms", "min", "h"])
def test_numeric_time_matches_reference(case, time_unit):
    status, series = _outcome(TimeSeriesBase, **_construction_cases()[case])
    if status == "error":
        pytest.skip("construction fails for this case")
    expected = reference_impl.numeric_time(series.time_index, time_unit)
    result = series.numeric_time(time_unit)
    assert isinstance(result, np.ndarray)
    assert result.dtype == expected.dtype
    assert np.array_equal(result, expected, equal_nan=True)


@pytest.mark.parametrize("index_unit", ["ns", "us", "ms", "s"])
def test_numeric_time_matches_reference_randomized(index_unit):
    rng = np.random.default_rng(42)
    ns_values = np.sort(rng.integers(0, 10**13, 5_000))
    time_index = pd.to_timedelta(ns_values, unit="ns").as_unit(index_unit)
    series = TimeSeriesBase(time_index)
    for time_unit in ["s", "ms", "us", "min"]:
        expected = reference_impl.numeric_time(series.time_index, time_unit)
        assert np.array_equal(series.numeric_time(time_unit), expected)


def _hash_channels() -> dict[str, Channel]:
    return {
        "relative": Channel(
            "relative", time_index=[0.0, 0.5, 1.0, 2.0], data=[1.0, 2.0, np.nan, 4.0]
        ),
        "absolute_with_offset": Channel(
            "absolute",
            time_index=pd.date_range("2024-01-01 12:00:00", periods=4, freq="125ms"),
            data=[1, 2, 3, 4],
            offset=1.5,
            metadata={"source": "test", "unit": "mV"},
        ),
        "time_only_ms": Channel("time only", time_index=[0, 250, 1000], time_unit="ms"),
    }


# Hashes are stored in saved files (``Vitals.save_data``), so they must not
# change unintentionally.
PINNED_DATA_HASHES = {
    "relative": "c93e5f65611843790542fb1a0ad7c74ba5b753cea26a4161328a1f3d148cd9ad",
    "absolute_with_offset": "f5f12577b74bc0437216ca63837597040d8e88573fcc0685d4219ee189850bf0",
    "time_only_ms": "4f6535cf1e0038b994f5542a6fb7f3aef548570ea870a717b7f32505aa175f67",
}


@pytest.mark.parametrize("name", list(PINNED_DATA_HASHES))
def test_data_hash_is_stable(name):
    assert _hash_channels()[name].data_hash() == PINNED_DATA_HASHES[name]


# ---------------------------------------------------------------------------
# Windowing: get_time_mask / get_data
# ---------------------------------------------------------------------------


def _random_series(rng: np.random.Generator) -> TimeSeriesBase:
    n_samples = int(rng.choice([0, 1, 2, 5, 50, 500]))
    times = np.round(rng.uniform(0, 100, n_samples), int(rng.choice([1, 3, 6])))
    if rng.random() < 0.8:
        times = np.sort(times)
    absolute = rng.random() < 0.5
    offset = rng.choice([None, 0.5, -2.0])
    time_unit = str(rng.choice(["s", "ms"]))
    if time_unit == "ms":
        times = times * 1000
    return Channel(
        "random",
        time_index=times,
        data=rng.normal(size=n_samples),
        time_start=ABSOLUTE_START if absolute else None,
        time_unit=time_unit,
        offset=None if offset is None else float(offset),
    )


def _random_time_input(rng: np.random.Generator, series: TimeSeriesBase):
    choice = rng.random()
    if choice < 0.15:
        return None
    unit = pd.to_timedelta(1, unit=series.time_unit)
    if choice < 0.45 and len(series) > 0:
        # exactly at a sample
        relative = series.time_index[int(rng.integers(len(series)))]
    else:
        relative = pd.to_timedelta(rng.uniform(-10, 110), unit="s")
    kind = rng.choice(["number", "timedelta", "timestamp", "string"])
    if kind == "number":
        return float(relative / unit)
    if kind == "timedelta":
        return relative
    if kind == "timestamp":
        # also used for relative series (where it is expected to raise)
        return (series.time_start or ABSOLUTE_START) + relative
    if series.is_time_absolute():
        return str(series.time_start + relative)
    return str(relative)


def _random_resolution(rng: np.random.Generator):
    return rng.choice(
        np.array(
            [None, 0, float(rng.uniform(0.001, 20)), "500ms", pd.Timedelta("2s")],
            dtype=object,
        )
    )


def _window_kwargs(rng, series):
    return dict(
        start=_random_time_input(rng, series),
        stop=_random_time_input(rng, series),
        resolution=_random_resolution(rng),
        include_adjacent=bool(rng.random() < 0.5),
    )


def _reference_mask(series, **kwargs):
    return reference_impl.get_time_mask(
        series.time_index, series.time_start, series.time_unit, **kwargs
    )


@pytest.mark.parametrize("seed", range(20))
def test_get_time_mask_matches_reference_randomized(seed, caplog):
    rng = np.random.default_rng(seed)
    for _ in range(25):
        series = _random_series(rng)
        kwargs = _window_kwargs(rng, series)
        expected_status, expected = _outcome(_reference_mask, series, **kwargs)
        status, result = _outcome(series.get_time_mask, **kwargs)
        assert status == expected_status, (kwargs, result, expected)
        if status == "error":
            assert result == expected
        else:
            assert result.dtype == bool
            np.testing.assert_array_equal(result, expected, err_msg=str(kwargs))


@pytest.mark.parametrize("seed", range(10))
def test_channel_and_label_get_data_match_reference_randomized(seed, caplog):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(20):
        channel = _random_series(rng)
        label = Label(
            "label",
            time_index=channel.time_index,
            data=None if rng.random() < 0.3 else channel.data,
            text_data=None
            if rng.random() < 0.5
            else np.array([f"t{i}" for i in range(len(channel))], dtype=object),
            time_start=channel.time_start,
            time_unit=channel.time_unit,
        )
        kwargs = _window_kwargs(rng, channel)
        resolution = kwargs.pop("resolution")

        for series, extra in [(channel, {"resolution": resolution}), (label, {})]:
            expected_status, mask = _outcome(
                _reference_mask, series, **kwargs, resolution=extra.get("resolution")
            )
            status, result = _outcome(series.get_data, **kwargs, **extra)
            assert status == expected_status, (kwargs, result, mask)
            if status == "error":
                assert result == mask
                continue

            expected_time = series.time_index[mask]
            if series.is_time_absolute():
                expected_time = expected_time + series.time_start
            assert result.time_index.equals(expected_time)

            if isinstance(series, Label):
                for attribute in ["data", "text_data"]:
                    values = getattr(series, attribute)
                    expected_values = (
                        values[mask] if values is not None and mask.any() else None
                    )
                    actual_values = getattr(result, attribute)
                    if expected_values is None:
                        assert actual_values is None
                    else:
                        np.testing.assert_array_equal(actual_values, expected_values)
            else:
                np.testing.assert_array_equal(result.data, series.data[mask])


class TestGetTimeMaskEdgeCases:
    """Explicit expectations for documented edge cases."""

    @pytest.fixture
    def channel(self):
        return Channel("c", time_index=[0.0, 1.0, 2.0, 3.0, 4.0], data=np.arange(5.0))

    def test_inclusive_bounds(self, channel):
        mask = channel.get_time_mask(start=1.0, stop=3.0)
        assert mask.tolist() == [False, True, True, True, False]

    def test_between_samples(self, channel):
        mask = channel.get_time_mask(start=0.5, stop=2.5)
        assert mask.tolist() == [False, True, True, False, False]

    def test_include_adjacent(self, channel):
        mask = channel.get_time_mask(start=1.5, stop=2.5, include_adjacent=True)
        assert mask.tolist() == [False, True, True, True, False]

    def test_include_adjacent_at_edges(self, channel):
        mask = channel.get_time_mask(start=-1.0, stop=10.0, include_adjacent=True)
        assert mask.all()

    def test_include_adjacent_with_empty_window(self, channel):
        mask = channel.get_time_mask(start=1.2, stop=1.8, include_adjacent=True)
        assert not mask.any()

    def test_start_after_stop_warns(self, channel, caplog):
        with caplog.at_level(logging.WARNING, logger="vitabel"):
            mask = channel.get_time_mask(start=3.0, stop=1.0)
        assert not mask.any()
        assert "is after stop time" in caplog.text

    def test_timestamp_bounds_for_relative_series_raise(self, channel):
        with pytest.raises(ValueError, match="relative time channel"):
            channel.get_time_mask(start=ABSOLUTE_START)

    def test_empty_series(self):
        channel = Channel("c", time_index=[], data=[])
        assert channel.get_time_mask(start=0.0, stop=1.0).tolist() == []

    def test_resolution_downsampling(self):
        channel = Channel("c", time_index=np.arange(100) * 0.1, data=np.arange(100.0))
        mask = channel.get_time_mask(start=1.0, stop=5.0, resolution=0.35)
        # stride floor(0.35 / 0.1) = 3, starting at the first included sample
        assert np.flatnonzero(mask).tolist() == list(range(10, 51, 3))

    def test_resolution_below_threshold_keeps_everything(self):
        channel = Channel("c", time_index=np.arange(100) * 0.1, data=np.arange(100.0))
        mask = channel.get_time_mask(resolution=0.2)  # n_downsample <= 2
        assert mask.all()

    def test_unsorted_index(self):
        channel = Channel("c", time_index=[3.0, 1.0, 2.0, 0.0], data=np.arange(4.0))
        mask = channel.get_time_mask(start=1.0, stop=2.5)
        assert mask.tolist() == [False, True, True, False]
        # NOTE: "adjacent" refers to positions in the index, not to time
        mask = channel.get_time_mask(start=1.0, stop=2.5, include_adjacent=True)
        assert mask.tolist() == [True, True, True, True]

    def test_absolute_series_with_offset(self):
        channel = Channel(
            "c",
            time_index=[0.0, 1.0, 2.0],
            data=[1.0, 2.0, 3.0],
            time_start=ABSOLUTE_START,
            offset=10.0,
        )
        data = channel.get_data(
            start=ABSOLUTE_START + pd.Timedelta(seconds=11),
            stop=ABSOLUTE_START + pd.Timedelta(seconds=12),
        )
        assert list(data.data) == [2.0, 3.0]
        assert data.time_index[0] == ABSOLUTE_START + pd.Timedelta(seconds=11)


def test_interval_label_get_data_golden():
    relative = IntervalLabel(
        "relative",
        time_index=[(0.0, 2.0), (1.0, 5.0), (6.0, 7.0), (10.0, 20.0)],
        data=[1.0, 2.0, 3.0, 4.0],
        text_data=["a", "b", "c", "d"],
    )
    absolute = IntervalLabel(
        "absolute",
        time_index=[
            (ABSOLUTE_START, ABSOLUTE_START + pd.Timedelta(seconds=2)),
            (
                ABSOLUTE_START + pd.Timedelta(seconds=5),
                ABSOLUTE_START + pd.Timedelta(seconds=9),
            ),
        ],
    )
    windows = [
        (None, None),
        (1.5, 6.5),
        (2.5, 4.0),  # window fully contained in an interval
        (12.0, 15.0),
        (30.0, 40.0),
        (None, 1.0),
        (6.5, None),
    ]
    results = {}
    for label in [relative, absolute]:
        for start, stop in windows:
            data = label.get_data(start=start, stop=stop)
            results[f"{label.name} {start} {stop}"] = {
                "intervals": data.time_index,
                "data": data.data,
                "text_data": data.text_data,
            }
    assert_matches_golden("timeseries_interval_label_get_data", results)


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def _record(steps: list, description: str, series):
    steps.append({"step": description, "series": summarize_series(series)})


def test_channel_mutations_golden():
    steps = []
    for time_start in [None, ABSOLUTE_START]:
        channel = Channel(
            "channel",
            time_index=[0.0, 0.5, 1.0, 1.5, 2.0, 4.0],
            data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            time_start=time_start,
        )
        kind = "relative" if time_start is None else "absolute"
        _record(steps, f"{kind}: initial", channel)
        channel.shift_time_index(1.5)
        _record(steps, f"{kind}: shift by 1.5", channel)
        channel.shift_time_index(pd.Timedelta("-250ms"))
        _record(steps, f"{kind}: shift by -250ms", channel)
        channel.offset = 10
        _record(steps, f"{kind}: set offset to 10", channel)
        _record(steps, f"{kind}: scaled by 2", channel.scale_time_index(2.0))
        _record(
            steps,
            f"{kind}: scaled by 0.5 around 11s",
            channel.scale_time_index(
                0.5,
                reference_time=pd.Timedelta(seconds=11)
                if time_start is None
                else time_start + pd.Timedelta(seconds=11),
            ),
        )
        start, stop = (
            (11.0, 12.0)
            if time_start is None
            else (
                time_start + pd.Timedelta(seconds=11),
                time_start + pd.Timedelta(seconds=12),
            )
        )
        _record(steps, f"{kind}: truncated", channel.truncate(start, stop))
    assert_matches_golden("timeseries_channel_mutations", steps)


def test_label_mutations_golden():
    steps = []
    for time_start in [None, ABSOLUTE_START]:
        kind = "relative" if time_start is None else "absolute"

        def t(seconds, time_start=time_start):
            delta = pd.Timedelta(seconds=seconds)
            return delta if time_start is None else time_start + delta

        label = Label(
            "label",
            time_index=[t(1.0), t(2.0), t(3.0)],
            data=[1.0, 2.0, 3.0],
        )
        _record(steps, f"{kind}: initial", label)
        label.add_data(t(2.5), value=2.5)
        _record(steps, f"{kind}: add in the middle", label)
        label.add_data(t(0.25), text="early")
        _record(steps, f"{kind}: add before first (with text)", label)
        label.add_data(t(0.125), value=0.5, text="earliest")
        _record(steps, f"{kind}: add value and text", label)
        label.remove_data(t(0.125))
        _record(steps, f"{kind}: remove first", label)
        label.remove_data(t(2.5))
        _record(steps, f"{kind}: remove in the middle", label)
        label.shift_time_index(pd.Timedelta(seconds=1))
        _record(steps, f"{kind}: shift by 1s", label)
        _record(steps, f"{kind}: truncated", label.truncate(t(2.0), t(3.5)))
        for seconds in [1.25, 2.0, 3.0, 4.0]:
            label.remove_data(t(seconds))
        _record(steps, f"{kind}: all removed", label)

    empty = Label("empty")
    _record(steps, "empty: initial", empty)
    empty.add_data(ABSOLUTE_START + pd.Timedelta(seconds=5), value=1.0)
    _record(steps, "empty: add absolute", empty)
    empty.add_data(ABSOLUTE_START + pd.Timedelta(seconds=1), value=2.0)
    _record(steps, "empty: add earlier", empty)
    assert_matches_golden("timeseries_label_mutations", steps)


def test_interval_label_mutations_golden():
    steps = []
    for time_start in [None, ABSOLUTE_START]:
        kind = "relative" if time_start is None else "absolute"

        def t(seconds, time_start=time_start):
            delta = pd.Timedelta(seconds=seconds)
            return delta if time_start is None else time_start + delta

        label = IntervalLabel(
            "intervals",
            time_index=[(t(1.0), t(2.0)), (t(4.0), t(6.0))],
            data=[1.0, 2.0],
        )
        _record(steps, f"{kind}: initial", label)
        label.add_data((t(3.0), t(3.5)), value=3.0, text="added")
        _record(steps, f"{kind}: add interval", label)
        label.add_data((t(0.5), t(0.75)))
        _record(steps, f"{kind}: add earlier interval", label)
        label.remove_data((t(1.0), t(2.0)))
        _record(steps, f"{kind}: remove first interval", label)
        label.shift_time_index(2.0)
        _record(steps, f"{kind}: shift by 2", label)
        _record(steps, f"{kind}: truncated", label.truncate(t(4.0), t(7.0)))

        def times(values, t=t):
            # NOTE: lists of pd.Timedelta / pd.Timestamp objects are currently
            # not supported by contains_time (see test below), numpy arrays are.
            return np.array([t(value).to_numpy() for value in values])

        intervals = times([5.0, 5.2, 5.2, 7.0, 9.0, 10.0]).reshape(-1, 2)
        steps.append(
            {
                "step": f"{kind}: contains_time",
                "points": label.contains_time(
                    times([0.0, 2.5, 5.0, 5.5, 6.0, 8.0, 9.0])
                ),
                "intervals": label.contains_time(intervals),
                "intervals_full": label.contains_time(
                    intervals, interval_require_full_containment=True
                ),
                "points_exclusive": label.contains_time(
                    times([5.0, 5.5, 6.0]), include_start=False, include_end=False
                ),
            }
        )
    assert_matches_golden("timeseries_interval_label_mutations", steps)
