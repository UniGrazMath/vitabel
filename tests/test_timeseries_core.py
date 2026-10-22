"""Tests for the time-series core: time parsing, time conversion, hashing,
windowing, and mutation of time series."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from vitabel import Channel, IntervalLabel, Label
from vitabel.timeseries import TimeSeriesBase


ABSOLUTE_START = pd.Timestamp("2024-01-01 12:00:00")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


_DATETIMES = pd.date_range(ABSOLUTE_START, periods=4, freq="250ms")
NAN = np.nan

CONSTRUCTION_CASES = {
    # case: (arguments, numeric_time("s"), time_start, offset in seconds)
    "float_list": (dict(time_index=[0.0, 0.5, 1.25, 3.0]), [0, 0.5, 1.25, 3], None, 0),
    "int_list": (dict(time_index=[0, 1, 2, 3]), [0, 1, 2, 3], None, 0),
    "float_array_with_nan": (
        dict(time_index=np.array([0.0, NAN, 2.0, 3.5])),
        [0, NAN, 2, 3.5],
        None,
        0,
    ),
    "int_array": (dict(time_index=np.arange(5)), [0, 1, 2, 3, 4], None, 0),
    "float_array_ms": (
        dict(time_index=np.array([0.0, 1.5, 3.0]), time_unit="ms"),
        [0, 0.0015, 0.003],
        None,
        0,
    ),
    "int_array_ms": (
        dict(time_index=np.array([0, 1250]), time_unit="ms"),
        [0, 1.25],
        None,
        0,
    ),
    "float_array_min": (
        dict(time_index=np.array([0.0, 1.5, 3.0]), time_unit="min"),
        [0, 90, 180],
        None,
        0,
    ),
    "numeric_offset": (dict(time_index=[0.0, 1.0], offset=2.5), [2.5, 3.5], None, 2.5),
    "timedelta_offset": (
        dict(time_index=[0.0, 1.0], offset=pd.Timedelta("-1s")),
        [-1, 0],
        None,
        -1,
    ),
    "series_of_floats": (
        dict(time_index=pd.Series([0.0, 1.0, 2.0])),
        [0, 1, 2],
        None,
        0,
    ),
    "none_first": (dict(time_index=[None, 1.0, 2.0]), [NAN, 1, 2], None, 0),
    "all_nan": (dict(time_index=np.array([NAN, NAN])), [NAN, NAN], None, 0),
    # relative time with a time start: absolute
    "relative_with_time_start": (
        dict(time_index=[0.0, 1.0, 2.0], time_start=ABSOLUTE_START),
        [0, 1, 2],
        ABSOLUTE_START,
        0,
    ),
    "relative_with_string_time_start": (
        dict(time_index=[0.0, 1.0], time_start="2024-01-01 12:00:00"),
        [0, 1],
        ABSOLUTE_START,
        0,
    ),
    "relative_with_tzaware_time_start": (  # local (wall clock) time
        dict(
            time_index=[0.0, 1.0],
            time_start=ABSOLUTE_START.tz_localize("Europe/Vienna"),
        ),
        [0, 1],
        ABSOLUTE_START,
        0,
    ),
    # timestamps
    "datetime64_ns_array": (
        dict(time_index=_DATETIMES.to_numpy()),
        [0, 0.25, 0.5, 0.75],
        ABSOLUTE_START,
        0,
    ),
    "datetime64_ms_array": (
        dict(time_index=_DATETIMES.to_numpy().astype("M8[ms]")),
        [0, 0.25, 0.5, 0.75],
        ABSOLUTE_START,
        0,
    ),
    "datetime64_s_array": (
        dict(
            time_index=np.array(["2024-01-01T12:00:00", "2024-01-01T12:00:05"], "M8[s]")
        ),
        [0, 5],
        ABSOLUTE_START,
        0,
    ),
    "datetime_index": (
        dict(time_index=_DATETIMES),
        [0, 0.25, 0.5, 0.75],
        ABSOLUTE_START,
        0,
    ),
    "datetime_index_tzaware": (  # local (wall clock) time
        dict(time_index=_DATETIMES.tz_localize("Europe/Vienna")),
        [0, 0.25, 0.5, 0.75],
        ABSOLUTE_START,
        0,
    ),
    "datetime_series": (
        dict(time_index=pd.Series(pd.date_range(ABSOLUTE_START, periods=2))),
        [0, 86400],
        ABSOLUTE_START,
        0,
    ),
    "timestamp_list": (
        dict(time_index=list(_DATETIMES)),
        [0, 0.25, 0.5, 0.75],
        ABSOLUTE_START,
        0,
    ),
    "timestamp_strings": (
        dict(time_index=["2024-01-01 12:00:00", "2024-01-01 12:00:01"]),
        [0, 1],
        ABSOLUTE_START,
        0,
    ),
    "datetime64_with_offset": (
        dict(time_index=_DATETIMES.to_numpy(), offset=1.0),
        [1, 1.25, 1.5, 1.75],
        ABSOLUTE_START,
        1,
    ),
    "datetime64_with_nat_first": (  # time start: the first timestamp
        dict(
            time_index=np.array(
                ["NaT", "2024-01-01T12:00:00", "2024-01-01T12:00:01"], "M8[ns]"
            )
        ),
        [NAN, 0, 1],
        ABSOLUTE_START,
        0,
    ),
    "datetime64_ms_array_with_nat": (
        dict(
            time_index=np.array(
                ["2024-01-01T12:00:00", "NaT", "2024-01-01T12:00:02.5"], "M8[ms]"
            )
        ),
        [0, NAN, 2.5],
        ABSOLUTE_START,
        0,
    ),
    "timestamp_list_with_nat_first": (
        dict(time_index=[pd.NaT, ABSOLUTE_START, ABSOLUTE_START + pd.Timedelta("1D")]),
        [NAN, 0, 86400],
        ABSOLUTE_START,
        0,
    ),
    # timedeltas
    "timedelta64_ns_array": (
        dict(time_index=pd.to_timedelta([0, 1, 2], unit="s").to_numpy()),
        [0, 1, 2],
        None,
        0,
    ),
    "timedelta64_ms_array": (
        dict(time_index=np.array([0, 1500, 3000], dtype="m8[ms]")),
        [0, 1.5, 3],
        None,
        0,
    ),
    "timedelta_index": (
        dict(time_index=pd.timedelta_range("1s", periods=3, freq="1s")),
        [1, 2, 3],
        None,
        0,
    ),
    "timedelta_list": (
        dict(time_index=[pd.Timedelta(seconds=s) for s in (0, 1, 2)]),
        [0, 1, 2],
        None,
        0,
    ),
    # empty
    "empty_list": (dict(time_index=[]), [], None, 0),
    "empty_float_array": (dict(time_index=np.array([], dtype=float)), [], None, 0),
    "empty_datetime64_array": (
        dict(time_index=np.array([], dtype="M8[ns]")),
        [],
        None,
        0,
    ),
    "empty_timedelta64_array": (
        dict(time_index=np.array([], dtype="m8[ns]")),
        [],
        None,
        0,
    ),
}
"""Not covered on purpose: timedelta strings (e.g. ``"1.5s"``) and
timestamp strings in mixed formats, whose parsing still has to be decided."""


@pytest.mark.parametrize(
    "kwargs, seconds, time_start, offset",
    CONSTRUCTION_CASES.values(),
    ids=CONSTRUCTION_CASES.keys(),
)
def test_construction(kwargs, seconds, time_start, offset):
    series = TimeSeriesBase(**kwargs)
    assert isinstance(series.time_index, pd.TimedeltaIndex)
    np.testing.assert_array_equal(series.numeric_time("s"), seconds)
    assert series.time_start == time_start
    assert series.offset == pd.Timedelta(seconds=offset)
    assert series.time_unit == kwargs.get("time_unit", "s")


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        (dict(time_index=[ABSOLUTE_START, 1.0]), (TypeError, ValueError), None),
        (
            dict(time_index=[ABSOLUTE_START, pd.Timedelta("1s")]),
            ValueError,
            "must be of the same type or numeric",
        ),
        (
            dict(time_index=list(_DATETIMES), time_start=ABSOLUTE_START),
            ValueError,
            "time_start cannot be passed if time data is absolute",
        ),
        (dict(time_index=["foo", "bar"]), ValueError, "could not be parsed"),
        (dict(time_index=np.zeros((2, 2))), ValueError, "must be a 1D array"),
        (dict(time_index=[object(), object()]), ValueError, "is not supported"),
    ],
    ids=[
        "mixed_types",
        "mixed_timestamp_timedelta",
        "absolute_with_time_start",
        "unparsable_strings",
        "two_dimensional",
        "unsupported_type",
    ],
)
def test_construction_errors(kwargs, error, match):
    with pytest.raises(error, match=match):
        TimeSeriesBase(**kwargs)


def test_construction_from_a_series():
    series = TimeSeriesBase(pd.Series([0.0, 1.5]))
    assert isinstance(series.time_index, pd.TimedeltaIndex)


@pytest.mark.parametrize(
    "values",
    [
        [ABSOLUTE_START, ABSOLUTE_START + pd.Timedelta(1, "s")],
        [pd.NaT, ABSOLUTE_START, ABSOLUTE_START + pd.Timedelta(1, "s")],
        [str(ABSOLUTE_START), str(ABSOLUTE_START + pd.Timedelta(1, "s"))],
    ],
    ids=["timestamps", "leading_nat", "strings"],
)
@pytest.mark.parametrize("index", ["shifted", "reversed"])
def test_construction_from_a_series_ignores_its_index(values, index):
    """The values of a Series are used by position, whatever its index."""
    labels = range(10, 10 + len(values))
    if index == "reversed":
        labels = reversed(range(len(values)))
    series = TimeSeriesBase(pd.Series(values, index=list(labels)))
    expected = TimeSeriesBase(values)
    assert series.time_start == ABSOLUTE_START
    assert series.time_index.equals(expected.time_index)


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


def _interval_labels() -> dict[str, IntervalLabel]:
    return {
        "relative": IntervalLabel(
            "relative",
            time_index=[(0.0, 2.0), (1.0, 5.0), (6.0, 7.0), (10.0, 20.0)],
            data=[1.0, 2.0, 3.0, 4.0],
            text_data=["a", "b", "c", "d"],
        ),
        "absolute": IntervalLabel(
            "absolute",
            time_index=[
                (ABSOLUTE_START, ABSOLUTE_START + pd.Timedelta(seconds=2)),
                (
                    ABSOLUTE_START + pd.Timedelta(seconds=5),
                    ABSOLUTE_START + pd.Timedelta(seconds=9),
                ),
            ],
        ),
    }


@pytest.mark.parametrize(
    "label, start, stop, intervals, data, text_data",
    [
        (
            "relative",
            None,
            None,
            [(0, 2), (1, 5), (6, 7), (10, 20)],
            [1, 2, 3, 4],
            ["a", "b", "c", "d"],
        ),
        ("relative", 1.5, 6.5, [(0, 2), (1, 5), (6, 7)], [1, 2, 3], ["a", "b", "c"]),
        ("relative", 2.5, 4.0, [(1, 5)], [2], ["b"]),  # window inside an interval
        ("relative", 12.0, 15.0, [(10, 20)], [4], ["d"]),
        ("relative", 30.0, 40.0, [], None, None),
        ("relative", None, 1.0, [(0, 2), (1, 5)], [1, 2], ["a", "b"]),
        ("relative", 6.5, None, [(6, 7), (10, 20)], [3, 4], ["c", "d"]),
        ("absolute", None, None, [(0, 2), (5, 9)], None, None),
        ("absolute", 1.5, 6.5, [(0, 2), (5, 9)], None, None),
        ("absolute", 2.5, 4.0, [], None, None),
        ("absolute", 30.0, 40.0, [], None, None),
        ("absolute", None, 1.0, [(0, 2)], None, None),
        ("absolute", 6.5, None, [(5, 9)], None, None),
    ],
)
def test_interval_label_get_data(label, start, stop, intervals, data, text_data):
    """Intervals that intersect the window are returned completely (numbers
    as bounds are seconds relative to the time start)."""
    result = _interval_labels()[label].get_data(start=start, stop=stop)
    reference = (
        np.datetime64(ABSOLUTE_START)
        if label == "absolute"
        else np.timedelta64(0, "ns")
    )
    seconds = (np.asarray(result.time_index) - reference) / np.timedelta64(1, "s")
    assert seconds.shape == (len(intervals), 2)
    assert [tuple(row) for row in seconds.tolist()] == intervals
    if data is None:
        assert result.data is None
    else:
        np.testing.assert_array_equal(result.data, data)
    if text_data is None:
        assert result.text_data is None
    else:
        assert list(result.text_data) == text_data


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


def _seconds(series) -> list[float]:
    """Times of a series in seconds (relative to ``ABSOLUTE_START`` for
    absolute series), independent of how they are stored."""
    time_index = series.time_index
    if series.is_time_absolute():
        time_index = time_index + (series.time_start - ABSOLUTE_START)
    return (time_index / pd.Timedelta(seconds=1)).tolist()


def _times(time_start):
    """``t(seconds)``: a timedelta, or a timestamp after ``time_start``."""

    def t(seconds):
        delta = pd.Timedelta(seconds=seconds)
        return delta if time_start is None else time_start + delta

    return t


def _assert_values(actual, expected):
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("time_start", [None, ABSOLUTE_START])
def test_channel_mutations(time_start):
    t = _times(time_start)
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    channel = Channel(
        "channel", [0.0, 0.5, 1.0, 1.5, 2.0, 4.0], data, time_start=time_start
    )

    channel.shift_time_index(1.5)
    assert _seconds(channel) == [1.5, 2.0, 2.5, 3.0, 3.5, 5.5]
    assert channel.offset == pd.Timedelta(seconds=1.5)
    channel.shift_time_index(pd.Timedelta("-250ms"))
    assert _seconds(channel) == [1.25, 1.75, 2.25, 2.75, 3.25, 5.25]
    assert channel.offset == pd.Timedelta(seconds=1.25)
    channel.offset = 10
    assert _seconds(channel) == [10.0, 10.5, 11.0, 11.5, 12.0, 14.0]

    # scaling and truncating return new channels (without offset)
    scaled = channel.scale_time_index(2.0, reference_time=t(10))
    assert _seconds(scaled) == [10.0, 11.0, 12.0, 13.0, 14.0, 18.0]
    assert scaled.offset == pd.Timedelta(0)
    scaled = channel.scale_time_index(0.5, reference_time=t(11))
    assert _seconds(scaled) == [10.5, 10.75, 11.0, 11.25, 11.5, 12.5]
    truncated = channel.truncate(t(11), t(12))
    assert _seconds(truncated) == [11.0, 11.5, 12.0]
    np.testing.assert_array_equal(truncated.data, [3.0, 4.0, 5.0])

    assert _seconds(channel) == [10.0, 10.5, 11.0, 11.5, 12.0, 14.0]
    np.testing.assert_array_equal(channel.data, data)


@pytest.mark.parametrize("time_start", [None, ABSOLUTE_START])
def test_scale_time_index_around_the_start(time_start):
    """Without reference time, the time start (or the first time point of a
    relative series) stays fixed."""
    channel = Channel("channel", [0.0, 0.5, 2.0], time_start=time_start)
    assert _seconds(channel.scale_time_index(2.0)) == [0.0, 1.0, 4.0]


@pytest.mark.parametrize("time_start", [None, ABSOLUTE_START])
def test_label_mutations(time_start):
    t = _times(time_start)
    label = Label("label", [t(1.0), t(2.0), t(3.0)], data=[1.0, 2.0, 3.0])

    label.add_data(t(2.5), value=2.5)
    assert _seconds(label) == [1.0, 2.0, 2.5, 3.0]
    _assert_values(label.data, [1.0, 2.0, 2.5, 3.0])
    assert label.text_data is None

    # entries are kept sorted; missing values / texts are NaN / None
    label.add_data(t(0.25), text="early")
    label.add_data(t(0.125), value=0.5, text="earliest")
    assert _seconds(label) == [0.125, 0.25, 1.0, 2.0, 2.5, 3.0]
    _assert_values(label.data, [0.5, NAN, 1.0, 2.0, 2.5, 3.0])
    assert list(label.text_data) == ["earliest", "early", None, None, None, None]

    label.remove_data(t(0.125))
    label.remove_data(t(2.5))
    assert _seconds(label) == [0.25, 1.0, 2.0, 3.0]
    _assert_values(label.data, [NAN, 1.0, 2.0, 3.0])
    assert list(label.text_data) == ["early", None, None, None]

    label.shift_time_index(pd.Timedelta(seconds=1))
    assert _seconds(label) == [1.25, 2.0, 3.0, 4.0]
    assert label.offset == pd.Timedelta(seconds=1)
    truncated = label.truncate(t(2.0), t(3.5))
    assert _seconds(truncated) == [2.0, 3.0]
    _assert_values(truncated.data, [1.0, 2.0])
    assert list(truncated.text_data) == [None, None]

    for seconds in [1.25, 2.0, 3.0, 4.0]:
        label.remove_data(t(seconds))
    assert len(label) == 0
    assert label.data is None
    assert label.text_data is None


def test_empty_label_becomes_absolute():
    label = Label("empty")
    label.add_data(ABSOLUTE_START + pd.Timedelta(seconds=5), value=1.0)
    assert label.is_time_absolute()
    label.add_data(ABSOLUTE_START + pd.Timedelta(seconds=1), value=2.0)
    assert _seconds(label) == [1.0, 5.0]
    _assert_values(label.data, [2.0, 1.0])


def _interval_entries(label: IntervalLabel) -> list[tuple]:
    """(start, stop, value, text) of all intervals (seconds as in
    ``_seconds``), sorted by start."""
    times = _seconds(label)
    values = [None] * len(label.intervals) if label.data is None else label.data
    texts = (
        [None] * len(label.intervals) if label.text_data is None else label.text_data
    )
    return sorted(
        (start, stop, None if pd.isna(value) else float(value), text)
        for start, stop, value, text in zip(times[::2], times[1::2], values, texts)
    )


@pytest.mark.parametrize("time_start", [None, ABSOLUTE_START])
def test_interval_label_mutations(time_start):
    t = _times(time_start)
    label = IntervalLabel(
        "intervals", time_index=[(t(1.0), t(2.0)), (t(4.0), t(6.0))], data=[1.0, 2.0]
    )

    label.add_data((t(3.0), t(3.5)), value=3.0, text="added")
    label.add_data((t(0.5), t(0.75)))
    assert _interval_entries(label) == [
        (0.5, 0.75, None, None),
        (1.0, 2.0, 1.0, None),
        (3.0, 3.5, 3.0, "added"),
        (4.0, 6.0, 2.0, None),
    ]

    label.remove_data((t(1.0), t(2.0)))
    label.shift_time_index(2.0)
    assert label.offset == pd.Timedelta(seconds=2)
    assert _interval_entries(label) == [
        (2.5, 2.75, None, None),
        (5.0, 5.5, 3.0, "added"),
        (6.0, 8.0, 2.0, None),
    ]

    # intervals intersecting the range are kept completely
    assert _interval_entries(label.truncate(t(4.0), t(7.0))) == [
        (5.0, 5.5, 3.0, "added"),
        (6.0, 8.0, 2.0, None),
    ]

    def times(*seconds):
        return np.array([t(value).to_numpy() for value in seconds])

    points = times(0.0, 2.5, 5.0, 5.5, 6.0, 8.0, 9.0)
    assert label.contains_time(points).tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert not label.contains_time(
        times(5.0, 5.5, 6.0), include_start=False, include_end=False
    ).any()
    intervals = times(5.0, 5.2, 5.2, 7.0, 9.0, 10.0).reshape(-1, 2)
    assert label.contains_time(intervals).tolist() == [True, True, False]
    assert label.contains_time(
        intervals, interval_require_full_containment=True
    ).tolist() == [True, False, False]


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


@pytest.mark.parametrize(
    "time_data",
    [
        pd.Timestamp("2024-01-01 12:00:01.5", tz="UTC"),
        [pd.Timestamp("2024-01-01 12:00:01.5", tz="Europe/Vienna")],
        pd.date_range("2024-01-01 12:00:01", periods=2, freq="s", tz="UTC"),
        [ABSOLUTE_START, pd.Timestamp("2024-01-01 12:00:01.5", tz="UTC")],
    ],
    ids=["scalar", "list", "index", "mixed"],
)
def test_interval_label_contains_time_rejects_time_zones(time_data):
    """Labels store time zone naive times: comparing them with time zone
    aware ones would be ambiguous (like the bounds of get_data)."""
    label = IntervalLabel(
        "intervals",
        time_index=[
            (
                ABSOLUTE_START + pd.Timedelta(1, "s"),
                ABSOLUTE_START + pd.Timedelta(2, "s"),
            )
        ],
    )
    with pytest.raises(TypeError, match="Time zone aware"):
        label.contains_time(time_data)
