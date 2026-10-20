"""Tests for the time-series core: time parsing, time conversion, hashing,
windowing, and mutation of time series."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from vitabel import Channel, IntervalLabel, Label
from vitabel.timeseries import TimeSeriesBase

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


@pytest.mark.golden
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


@pytest.mark.parametrize(
    "kwargs, expected_seconds, expected_start",
    [
        (dict(time_index=np.array([0.0, 1.25, np.nan])), [0, 1.25, np.nan], None),
        (dict(time_index=np.array([0, 1250]), time_unit="ms"), [0, 1.25], None),
        (
            dict(time_index=pd.to_timedelta([0, 1.25], unit="s").to_numpy()),
            [0, 1.25],
            None,
        ),
        (
            dict(
                time_index=np.array(
                    ["2024-01-01T12:00:00", "NaT", "2024-01-01T12:00:02.5"], "M8[ms]"
                )
            ),
            [0, np.nan, 2.5],
            ABSOLUTE_START,
        ),
        (
            # time_start is the local (wall clock) time of the first entry
            dict(
                time_index=pd.date_range(
                    ABSOLUTE_START, periods=2, freq="1250ms", tz="Europe/Vienna"
                )
            ),
            [0, 1.25],
            ABSOLUTE_START,
        ),
        (
            dict(time_index=pd.Series(pd.date_range(ABSOLUTE_START, periods=2))),
            [0, 86400],
            ABSOLUTE_START,
        ),
    ],
    ids=[
        "float_array",
        "int_array_ms",
        "timedelta64_array",
        "datetime64_array_with_nat",
        "tzaware_datetime_index",
        "datetime_series",
    ],
)
def test_construction_from_typed_arrays(kwargs, expected_seconds, expected_start):
    series = TimeSeriesBase(**kwargs)
    assert isinstance(series.time_index, pd.TimedeltaIndex)
    np.testing.assert_array_equal(series.numeric_time("s"), expected_seconds)
    assert series.time_start == expected_start


# ---------------------------------------------------------------------------
# numeric_time and data_hash
# ---------------------------------------------------------------------------


def test_numeric_time():
    series = TimeSeriesBase([])
    series.time_index = pd.to_timedelta(["1.5s", "NaT", "-2min"]).as_unit("ms")
    np.testing.assert_array_equal(series.numeric_time("s"), [1.5, np.nan, -120.0])
    np.testing.assert_array_equal(series.numeric_time("min"), [0.025, np.nan, -2.0])
    # exact also beyond 2**53 ns
    series.time_index = pd.to_timedelta([2**60], unit="ns")
    assert series.numeric_time("s")[0] == 2**60 / 10**9


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
# change unintentionally (and must be identical on all platforms).
PINNED_DATA_HASHES = {
    "relative": "7b24d5d113fe72df6fcf99fe16fce4b8e834994ac2e1765f6d6939fa1d94a3f3",
    "absolute_with_offset": (
        "bced44a11c2af47ade5412db1833fb81bb311b836304c4ec4a79f422f073b145"
    ),
    "time_only_ms": "9a34d87a8f194fb299b332a85e9efe0d89f5e14066a81adae3b1c9b8d51fc555",
}


@pytest.mark.parametrize("name", list(PINNED_DATA_HASHES))
def test_data_hash_is_stable(name):
    assert _hash_channels()[name].data_hash() == PINNED_DATA_HASHES[name]


def test_data_hash_does_not_depend_on_compressed_encoding(monkeypatch):
    # compressed arrays (np.savez_compressed) contain platform-specific ZIP
    # metadata, so they must not be part of the hash
    def fail(*args, **kwargs):
        raise AssertionError("data_hash must not compress arrays")

    monkeypatch.setattr(np, "savez_compressed", fail)
    channel = _hash_channels()["absolute_with_offset"]
    channel.metadata["array"] = np.arange(3)
    channel.metadata["timestamps"] = pd.date_range(
        "2024-01-01 12:00:00.123456789", periods=3, freq="ns"
    )
    channel.metadata["timedeltas"] = pd.timedelta_range("1ns", periods=3, freq="ns")
    assert len(channel.data_hash()) == 64


# ---------------------------------------------------------------------------
# Windowing: get_time_mask / get_data
# ---------------------------------------------------------------------------


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


@pytest.mark.golden
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


class TestWindowingOfSortedSeries:
    """Windows of sorted series are found by binary search; these cases need
    care to select the same time points as comparing all of them."""

    def test_bounds_finer_than_the_unit_of_the_time_index(self):
        channel = Channel("channel", np.zeros(6), data=np.arange(6.0))
        channel.time_index = pd.to_timedelta(np.arange(6), unit="s").as_unit("s")
        np.testing.assert_array_equal(
            channel.get_data(start=1.5, stop=3.5).data, [2.0, 3.0]
        )
        one_ns = pd.Timedelta(1, "ns")
        data = channel.get_data(
            start=pd.Timedelta(1, "s") + one_ns, stop=pd.Timedelta(3, "s") - one_ns
        )
        np.testing.assert_array_equal(data.data, [2.0])

    def test_bounds_far_outside_of_the_time_range(self):
        channel = Channel(
            "channel", np.arange(5.0), data=np.arange(5.0), time_start=ABSOLUTE_START
        )
        early, late = pd.Timestamp("1700-01-01"), pd.Timestamp("2300-01-01")
        assert len(channel.get_data(start=early, stop=late).data) == 5
        assert len(channel.get_data(start=late).data) == 0
        assert len(channel.get_data(stop=early).data) == 0

    def test_numpy_bounds(self):
        channel = Channel(
            "channel", np.arange(5.0), data=np.arange(5.0), time_start=ABSOLUTE_START
        )
        data = channel.get_data(
            start=np.datetime64("2024-01-01T12:00:01.5", "us"),
            stop=np.datetime64("2024-01-02", "D"),
        )
        np.testing.assert_array_equal(data.data, [2.0, 3.0, 4.0])

    def test_time_zone_aware_bounds_raise(self):
        channel = Channel(
            "channel", np.arange(5.0), data=np.arange(5.0), time_start=ABSOLUTE_START
        )
        with pytest.raises(TypeError):
            channel.get_data(start=ABSOLUTE_START.tz_localize("UTC"))

    def test_missing_times_are_never_selected(self):
        channel = Channel("channel", [0.0, np.nan, 2.0], data=[0.0, 1.0, 2.0])
        np.testing.assert_array_equal(channel.get_data(start=0, stop=5).data, [0, 2])


@pytest.mark.parametrize("absolute", [False, True])
def test_get_data_returns_copies(absolute):
    """Modifying returned data must not modify the series."""
    channel = Channel(
        "channel",
        time_index=np.arange(10.0),
        data=np.arange(10.0),
        time_start=ABSOLUTE_START if absolute else None,
    )
    label = Label(
        "label",
        time_index=np.arange(10.0),
        data=np.arange(10.0),
        text_data=np.array([str(i) for i in range(10)], dtype=object),
        time_start=ABSOLUTE_START if absolute else None,
    )
    for series in [channel, label]:
        result = series.get_data(start=2.0, stop=5.0)
        result.data[:] = -1
        if isinstance(series, Label):
            result.text_data[:] = "x"
            assert list(series.text_data) == [str(i) for i in range(10)]
        np.testing.assert_array_equal(series.data, np.arange(10.0))


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def _record(steps: list, description: str, series):
    steps.append({"step": description, "series": summarize_series(series)})


@pytest.mark.golden
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


@pytest.mark.golden
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


@pytest.mark.golden
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


@pytest.mark.parametrize("time_start", [None, ABSOLUTE_START])
def test_interval_label_contains_pandas_times(time_start):
    """Lists and scalars of pandas timestamps / timedeltas (numpy keeps them
    as objects)."""

    def t(seconds):
        delta = pd.Timedelta(seconds=seconds)
        return delta if time_start is None else time_start + delta

    label = IntervalLabel("intervals", time_index=[(t(1), t(2))])
    assert label.contains_time([t(1.5), t(3)]).tolist() == [True, False]
    assert label.contains_time(t(1.5))
    assert label.contains_time([[t(1.5), t(3)]]).tolist() == [True]
