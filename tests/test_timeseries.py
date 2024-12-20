from __future__ import annotations

import json
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vitabel import TimeDataCollection, Channel, Label, IntervalLabel
from vitabel.utils.helpers import NumpyEncoder
from vitabel.timeseries import TimeSeriesBase


def get_random_collection(
    num_channels: int = 0,
    num_labels: int = 0,
    num_attached_labels: int = 0,
):
    channels = []
    for j in range(num_channels):
        time_index = pd.date_range(
            "2020-01-01 12:00:00",
            "2020-01-01 13:00:00",
            periods=1001,
        )
        time_index += np.random.randint(-60, 60) * pd.to_timedelta(1, unit="min")
        data = None
        if np.random.rand() >= 0.5:
            data = np.random.rand(1001)
        channels.append(Channel(name=f"ch{j}", time_index=time_index, data=data))

    labels = []
    for j in range(num_labels):
        time_index = pd.date_range(
            "2020-01-01 12:00:00",
            "2020-01-01 13:00:00",
            periods=1001,
        )
        time_index += np.random.randint(-60, 60) * pd.to_timedelta(1, unit="min")
        data = None
        if np.random.rand() >= 0.5:
            data = np.random.rand(1001)
        labels.append(Label(name=f"lab{j}", time_index=time_index, data=data))

    chosen_labels = np.random.choice(labels, size=num_attached_labels, replace=False)
    for label in chosen_labels:
        channel = np.random.choice(channels)
        label.attach_to(channel)

    return TimeDataCollection(channels=channels, labels=labels)


def test_empty_timeseriesbase():
    ts = TimeSeriesBase(time_index=[])

    assert len(ts.time_index) == 0
    assert ts.time_unit == "s"
    assert ts.time_start is None
    assert ts.offset.total_seconds() == 0


def test_timeseriesbase_different_datatypes():
    with pytest.raises(ValueError, match="All time data must be of the same type"):
        TimeSeriesBase(time_index=[42, "12s", pd.Timestamp("2020-02-02 12:00:00")])


def test_timeseriesbase_with_string_timestamps():
    ts = TimeSeriesBase(
        time_index=[
            "2020-02-02 12:00:00",
            "2020-02-02 12:00:05",
            "2020-02-02 12:00:10",
        ]
    )
    assert len(ts) == 3
    assert isinstance(ts.time_index, pd.TimedeltaIndex)


def test_timeseriesbase_with_string_timedeltas():
    with pytest.warns(UserWarning):
        ts = TimeSeriesBase(time_index=["1s", "2s", "3s"])
    assert len(ts) == 3
    assert isinstance(ts.time_index, pd.TimedeltaIndex)


def test_timeseriesbase_with_unparsable_strings():
    with (
        pytest.warns(UserWarning),
        pytest.raises(
            ValueError, match="could not be parsed to timestamps or timedeltas"
        ),
    ):
        TimeSeriesBase(time_index=["1s", "2s", "3s", "2020-02-02 12:00:00"])


def test_timeseriesbase_with_timestamps():
    ts1 = TimeSeriesBase(
        time_index=pd.date_range(
            start="2020-02-02 12:00:00",
            end="2020-02-02 13:00:00",
            periods=20,
        )
    )
    assert len(ts1) == 20
    assert isinstance(ts1.time_index, pd.TimedeltaIndex)

    ts2 = TimeSeriesBase(
        time_index=np.arange(
            "2020-02-02 12:00:00", "2020-02-02 12:10:00", dtype="datetime64[m]"
        )
    )
    assert len(ts2) == 10
    assert isinstance(ts2.time_index, pd.TimedeltaIndex)


def test_timeseriesbase_absolute_with_start():
    with pytest.raises(ValueError, match="cannot be passed if time data is absolute"):
        TimeSeriesBase(
            time_index=pd.date_range(
                start="2020-02-02 12:00:00",
                end="2020-02-02 13:00:00",
                periods=20,
            ),
            time_start=pd.Timestamp("2020-02-02 10:00:00"),
        )


def test_timeseriesbase_relative_with_start():
    ts = TimeSeriesBase(
        time_index=np.arange(0, 10, 0.5), time_start=pd.Timestamp("2020-02-02 12:00:00")
    )
    assert ts.is_time_absolute()
    assert ts.time_unit == "s"


def test_timeseriesbase_relative():
    ts = TimeSeriesBase(time_index=np.arange(0, 100, 5), time_unit="ms")
    assert ts.is_time_relative()
    assert ts.time_unit == "ms"


def test_timeseriesbase_unsupported_type():
    with pytest.raises(ValueError, match="<class 'NoneType'> is not supported"):
        TimeSeriesBase(time_index=[None, None, None])


def test_timeseriesbase_numeric_time():
    ts = TimeSeriesBase(
        time_index=[0.0, 3.5, 10.0, 42.0],
        time_unit="min",
    )
    assert ts.time_unit == "min"
    np.testing.assert_array_equal(ts.numeric_time(), [0, 3.5, 10, 42])
    np.testing.assert_array_equal(ts.numeric_time(time_unit="s"), [0, 210, 600, 2520])


@pytest.mark.parametrize("time_offset", [60, pd.Timedelta("1min")])
def test_timeseriesbase_offset(time_offset):
    ts = TimeSeriesBase(
        time_index=np.arange(0, 10, 0.5),
        time_unit="s",
        offset=time_offset,
    )
    np.testing.assert_array_equal(ts.numeric_time(), np.arange(0, 10, 0.5) + 60)


def test_timeseriesbase_shift_index():
    time = np.arange(0, 10, 0.5)
    ts = TimeSeriesBase(
        time_index=time,
        time_unit="s",
    )
    ts.shift_time_index(60)
    np.testing.assert_array_equal(ts.numeric_time(), time + 60)
    ts.shift_time_index(pd.Timedelta("-30s"))
    np.testing.assert_array_equal(ts.numeric_time(), time + 30)


def test_timeseriesbase_convert_time():
    absolute_time_series = TimeSeriesBase(
        time_index=pd.date_range("2020-02-02", periods=10, freq="h")
    )
    relative_time_series = TimeSeriesBase(
        time_index=np.arange(0, 10, 0.5),
        time_unit="s",
    )

    absolute_time = pd.Timestamp("2020-02-02 12:00:00")
    assert absolute_time_series.convert_time_input(absolute_time) == absolute_time
    with pytest.raises(
        ValueError, match="The channel time is relative, but .* is a timestamp"
    ):
        relative_time_series.convert_time_input(absolute_time)

    relative_time = pd.Timedelta("1min")
    assert relative_time_series.convert_time_input(relative_time) == relative_time
    assert absolute_time_series.convert_time_input(relative_time) == pd.Timestamp(
        "2020-02-02 00:01:00"
    )

    relative_time = 42.0
    assert relative_time_series.convert_time_input(relative_time) == pd.Timedelta("42s")
    assert absolute_time_series.convert_time_input(relative_time) == pd.Timestamp(
        "2020-02-02 00:00:42"
    )

    with pytest.raises(
        ValueError, match="Could not convert nonsense to a valid time format"
    ):
        absolute_time_series.convert_time_input("nonsense")


def test_channel_creation():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)

    assert channel.name == "test"
    np.testing.assert_allclose(channel.numeric_time(), time)

    with pytest.raises(ValueError, match="length .* must be the same"):
        Channel(name="invalid", time_index=[1, 2, 3], data=[42])

    series_time = pd.Series(time)
    series_data = pd.Series(data)
    channel = Channel(name="test", time_index=series_time, data=series_data)
    assert channel.time_index.shape == (10,)
    assert channel.data.shape == (10,)


def test_channel_creation_data_from_list():
    time = [0, 0.12, 3.17, 6.42]
    data = [1, 2, 3, 4.5]
    channel = Channel(name="test", time_index=time, data=data)

    assert isinstance(channel.time_index, pd.TimedeltaIndex)
    assert isinstance(channel.data, np.ndarray)


def test_channel_time_only():
    time = np.arange(0, 1, 0.1)
    channel = Channel(name="test", time_index=time, data=None)

    assert channel.is_time_only()


def test_channel_get_data():
    time = np.arange(0, 5, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    time_channel = Channel(name="test timeonly", time_index=time, data=None)

    x, y = channel.get_data()
    np.testing.assert_allclose(x.total_seconds(), time)
    np.testing.assert_allclose(y, data)

    x, y = time_channel.get_data()
    np.testing.assert_allclose(x.total_seconds(), time)
    assert y is None

    x, y = channel.get_data(start=1.5, stop=3.5)
    np.testing.assert_allclose(x.total_seconds(), time[15:36])
    np.testing.assert_allclose(y, data[15:36])

    x, y = channel.get_data(start=1.0, stop=4.0, resolution="0.5s")
    np.testing.assert_allclose(x.total_seconds(), np.arange(1.0, 4.01, 0.5))
    np.testing.assert_allclose(y, 0, atol=1e-8)

    x, y = channel.get_data(start=1.0, stop=4.0, resolution=1.0)
    np.testing.assert_allclose(x.total_seconds(), np.arange(1.0, 4.01, 1.0))
    np.testing.assert_allclose(y, 0, atol=1e-8)


def test_channel_get_data_absolute():
    time = pd.date_range("2020-02-02 12:00:00", periods=100, freq="6s")
    data = np.random.random(len(time))
    channel = Channel(name="test", time_index=time, data=data)

    x, y = channel.get_data(
        start=pd.Timestamp("2020-02-02 12:02:30"),
        stop=pd.Timestamp("2020-02-02 12:05:00"),
    )
    assert np.all(x == time[25:51])
    np.testing.assert_allclose(y, data[25:51])


def test_channel_get_data_invalid_bounds():
    time = np.arange(0, 5, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)

    with pytest.raises(ValueError, match="Start or stop time is given as a timestamp"):
        channel.get_data(start=None, stop=pd.Timestamp("2020-02-02 12:15:00"))


def test_channel_to_and_from_dict():
    time = np.arange(0, 2, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(
        name="test",
        time_index=time,
        data=data,
        time_unit="s",
        time_start=pd.Timestamp("2020-02-02 12:00:00"),
        metadata={
            "creator": "Tester",
            "project_id": 42,
        },
    )
    label = Label(
        name="events",
        time_index=[0.15, 1.8],
        time_start="2020-02-02 12:00:00",
        data=[None, "test"],
        anchored_channel=channel,
    )

    label_dict = label.to_dict()
    label_dict_expected = {
        "name": "events",
        "time_index": [0.15, 1.8],
        "time_unit": "s",
        "time_start": "2020-02-02 12:00:00",
        "offset": 0,
        "is_interval": False,
        "data": [None, "test"],
        "plotstyle": {
            "marker": "o",
            "ms": 5,
            "linestyle": "none",
        },
        "metadata": {},
    }
    np.testing.assert_equal(label_dict, label_dict_expected)

    channel_dict = channel.to_dict()
    dict_time_index = channel_dict.pop("time_index")
    np.testing.assert_allclose(dict_time_index, time, atol=1e-8)
    dict_data = channel_dict.pop("data")
    np.testing.assert_allclose(dict_data, data, atol=1e-8)

    np.testing.assert_equal(
        channel_dict,
        {
            "name": "test",
            "time_unit": "s",
            "time_start": "2020-02-02 12:00:00",
            "offset": 0,
            "labels": [label_dict_expected],
            "plotstyle": {},
            "metadata": {
                "creator": "Tester",
                "project_id": 42,
            },
        },
    )

    channel_dict["time_index"] = dict_time_index
    channel_dict["data"] = dict_data
    new_channel = Channel.from_dict(channel_dict)
    assert new_channel.name == "test"
    np.testing.assert_allclose(new_channel.numeric_time(), time)
    np.testing.assert_allclose(new_channel.data, data)
    assert new_channel.time_unit == "s"
    assert new_channel.time_start == pd.Timestamp("2020-02-02 12:00:00")
    assert new_channel.metadata == {
        "creator": "Tester",
        "project_id": 42,
    }


@pytest.mark.mpl_image_compare
def test_channel_plot_absolute_time():
    time = np.arange(0, 2, 0.05)
    data = np.sin(2 * np.pi * time)
    channel = Channel(
        name="test",
        time_index=time,
        time_start=pd.Timestamp("2020-02-02 12:00:00"),
        time_unit="min",
        data=data,
    )
    return channel.plot(
        plotstyle={"color": "red"},
        start=pd.Timestamp("2020-02-02 12:00:30"),
    )


@pytest.mark.mpl_image_compare
def test_channel_plot_no_data():
    time = [
        pd.Timestamp("2020-02-02 12:00:00"),
        pd.Timestamp("2020-02-02 12:00:05"),
        pd.Timestamp("2020-02-02 12:00:30"),
        pd.Timestamp("2020-02-02 12:00:42"),
    ]
    channel = Channel(
        name="test",
        time_index=time,
        data=None,
        time_unit="s",
        plotstyle={"color": "goldenrod", "marker": "*"},
    )
    fig, ax = plt.subplots()
    channel.plot(plot_axes=ax, plotstyle={"label": "Testplot", "marker": "o"})
    ax.grid(True)
    return fig


def test_label_creation():
    label = Label(name="test", time_index=[0, 5, 12])

    assert label.name == "test"
    assert len(label) == 3
    assert label.time_unit == "s"
    assert label.time_start is None
    assert label.offset.total_seconds() == 0


def test_label_creation_errors():
    with pytest.raises(
        ValueError,
        match="length of the data must be equal to the length of the time index",
    ):
        Label(name="test", time_index=[0, 5, 12], data=[None, None])


def test_label_attached_to_channel():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(name="events", time_index=[0.5, 0.7, 0.9], anchored_channel=channel)

    assert label.anchored_channel == channel
    assert channel.labels == [label]

    label.detach()
    assert label.anchored_channel is None
    assert channel.labels == []


def test_label_attachment_errors():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(name="events", time_index=[0.5, 0.7, 0.9], anchored_channel=channel)

    with pytest.raises(
        ValueError, match="The label events is already attached to this channel"
    ):
        label.attach_to(channel)

    label.detach()
    with pytest.raises(
        ValueError, match="The label events is not attached to any channel"
    ):
        label.detach()

    with pytest.raises(
        ValueError, match="The label events is not attached to this channel"
    ):
        channel.detach_label(label)


def test_channel_label_time_type_mismatch():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(
        name="events",
        time_index=[0.5, 0.7, 0.9],
        time_start="2020-02-02 12:00:00",
    )
    with pytest.raises(ValueError, match="must be both absolute or both relative"):
        label.attach_to(channel)


def test_label_get_data():
    time_start = pd.to_datetime("2020-02-02 12:00:00")
    label = Label(
        name="events",
        time_index=[1.3, 4.5, 7.9, 13.15, 15.13, 17],
        time_start=time_start,
        data=[42, None, -100, 1.1, 2.2, 3.3],
    )
    t1, val1 = label.get_data(start=8, stop=16)
    t1 -= time_start
    np.testing.assert_array_equal(t1.total_seconds(), [13.15, 15.13])
    np.testing.assert_array_equal(val1, [1.1, 2.2])


def test_label_relative_time_add_remove_data():
    label = Label(
        name="events",
        time_index=[1.3, 4.5, 7.9, 13.15, 15.13, 17],
        data=[42, None, -100, 1.1, 2.2, 3.3],
        time_unit="s",
    )
    label.add_data(pd.to_timedelta(10, unit="s"), 42)
    label.add_data(pd.to_timedelta(20, unit="s"), 100)
    assert len(label) == 8
    np.testing.assert_equal(label.data, [42, None, -100, 42, 1.1, 2.2, 3.3, 100])

    label.remove_data(pd.to_timedelta(17, unit="s"))
    assert 3.3 not in label.data
    assert len(label) == 7

    with pytest.raises(ValueError, match="No data point found at"):
        label.remove_data(pd.to_timedelta(999, unit="s"))


def test_label_absolute_time_add_remove_data():
    label = Label(
        name="events",
        time_index=pd.date_range("2020-02-02 12:00:00", periods=6, freq="5min"),
    )
    label.add_data(pd.Timestamp("2020-01-01 00:00:00"))
    assert len(label) == 7
    assert label.time_start == pd.Timestamp("2020-01-01 00:00:00")

    label.remove_data(pd.Timestamp("2020-02-02 12:00:00"))
    assert len(label) == 6

    label.remove_data(pd.Timestamp("2020-01-01 00:00:00"))
    assert len(label) == 5
    assert label.time_start == pd.Timestamp("2020-02-02 12:05:00")


def test_empty_label_add_absolute_time_data():
    label = Label(name="events")
    label.add_data(pd.Timestamp("2020-02-02 12:00:00"))
    assert len(label) == 1
    assert label.time_start == pd.Timestamp("2020-02-02 12:00:00")

    label.remove_data(pd.Timestamp("2020-02-02 12:00:00"))
    assert len(label) == 0
    assert label.time_start is None


def test_channel_offset_propagation():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(
        name="events",
        time_index=[0.5, 0.7, 0.9],
        offset=1,
        anchored_channel=channel,
    )

    assert channel.offset.total_seconds() == 0
    assert label.offset.total_seconds() == 1

    channel.shift_time_index(1.5)
    assert channel.offset.total_seconds() == 1.5
    assert label.offset.total_seconds() == 2.5

    label.shift_time_index(0.5)
    assert channel.offset.total_seconds() == 1.5
    assert label.offset.total_seconds() == 3.0


def test_interval_label_creation():
    label = IntervalLabel(
        name="test",
        time_index=[0, 5, 12, 15],
        data=["A", "B"],
        time_unit="s",
    )
    assert len(label) == 2
    np.testing.assert_equal(label.intervals / pd.Timedelta("1s"), [(0, 5), (12, 15)])


def test_interval_label_creation_errors():
    with pytest.raises(ValueError, match="even number of elements"):
        IntervalLabel(name="test", time_index=[1, 2, 3, 4, 5])

    with pytest.raises(ValueError, match="must be half the length of the time index"):
        IntervalLabel(name="test", time_index=[1, 2, 3, 4], data=["A"])


def test_interval_label_get_data():
    time_index = [
        pd.Timestamp("2020-02-02 12:00:00"),
        pd.Timestamp("2020-02-02 12:45:00"),
        pd.Timestamp("2020-02-03 14:42:00"),
        pd.Timestamp("2020-02-03 18:00:00"),
    ]
    label = IntervalLabel(
        name="rainfall",
        time_index=time_index,
    )
    t, _ = label.get_data()
    assert len(t) == 2
    assert tuple(t[0]) == tuple(
        pd.Timestamp(stmp) for stmp in ["2020-02-02 12:00:00", "2020-02-02 12:45:00"]
    )

    t, _ = label.get_data(start=pd.Timestamp("2020-02-03 14:00:00"))
    assert len(t) == 1
    assert tuple(t[0]) == tuple(
        pd.Timestamp(stmp) for stmp in ["2020-02-03 14:42:00", "2020-02-03 18:00:00"]
    )

    t, _ = label.get_data(start=pd.Timestamp("2020-02-03 15:00:00"))
    assert len(t) == 1
    assert tuple(t[0]) == tuple(
        pd.Timestamp(stmp) for stmp in ["2020-02-03 14:42:00", "2020-02-03 18:00:00"]
    )


def test_empty_collection():
    collection = TimeDataCollection()

    assert collection.channels == []
    assert collection.labels == []
    assert collection.local_labels == []
    assert collection.global_labels == []


def test_add_and_remove_channel(data_time_2ecg):
    time, signal1, signal2 = data_time_2ecg

    channel1 = Channel(
        name="ECG 1",
        time_index=time,
        data=signal1,
    )
    channel2 = Channel(
        name="ECG 2",
        time_index=time,
        data=signal2,
    )

    collection = TimeDataCollection(channels=[channel1])
    collection.add_channel(channel2)

    assert len(collection.channels) == 2
    assert collection.channel_names == ["ECG 1", "ECG 2"]

    collection.remove_channel(name="ECG 1")

    assert len(collection.channels) == 1
    assert collection.channel_names == ["ECG 2"]


def test_collection_incompatible_time_types():
    time = np.arange(0, 1, 0.1)
    channel1 = Channel(name="ECG 1", time_index=time, data=np.sin(2 * np.pi * time))
    channel2 = Channel(
        name="ECG 2", time_index=pd.date_range("2020-02-02", periods=10, freq="h")
    )

    with pytest.raises(
        ValueError,
        match="time data in the collection must be either absolute or relative",
    ):
        TimeDataCollection(channels=[channel1, channel2])

    collection = TimeDataCollection(channels=[channel1])
    with pytest.raises(
        ValueError, match="channel ECG 2 does not match the time type of the collection"
    ):
        collection.add_channel(channel2)


def test_collection_with_local_and_global_labels():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(name="local 1", time_index=[0.5, 0.7, 0.9], anchored_channel=channel)
    global_label_1 = Label(name="global 1", time_index=[0.2, 0.4, 0.6])

    collection = TimeDataCollection(channels=[channel], labels=[label, global_label_1])
    assert len(collection.local_labels) == 1
    assert len(collection.global_labels) == 1
    assert len(collection.labels) == 2
    assert collection.is_time_absolute() != collection.is_time_relative()

    global_label_2 = Label(name="global 2", time_index=[0.2, 0.4, 0.6])
    collection.add_global_label(global_label_2)
    assert len(collection.local_labels) == 1
    assert len(collection.global_labels) == 2

    new_label = Label(name="local 2", time_index=[0.1, 0.3, 0.5])
    channel.attach_label(new_label)
    assert len(collection.local_labels) == 2
    assert collection.label_names == ["global 1", "global 2", "local 1", "local 2"]

    collection.remove_label(name="global 1")
    assert len(collection.global_labels) == 1

    channel.detach_label(new_label)
    assert collection.local_labels == [label]


def test_collection_add_label_with_foreign_channel():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    foreign_channel = Channel(name="test", time_index=time, data=data)
    label = Label(
        name="events", time_index=[0.5, 0.7, 0.9], anchored_channel=foreign_channel
    )

    with pytest.raises(ValueError, match="channel that is not in the collection"):
        TimeDataCollection(labels=[label])


def test_collection_label_duplicate():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(name="events", time_index=[0.5, 0.7, 0.9])

    collection = TimeDataCollection(channels=[channel], labels=[label])

    with pytest.raises(
        ValueError, match="label events has already been added to the collection"
    ):
        collection.add_global_label(label)


def test_collection_add_local_label_as_global():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(name="events", time_index=[0.5, 0.7, 0.9], anchored_channel=channel)

    collection = TimeDataCollection(channels=[channel])
    with pytest.raises(
        ValueError,
        match="attached to channel test and cannot be added as a global label",
    ):
        collection.add_global_label(label)


def test_collection_label_time_type_mismatch():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(
        name="events",
        time_index=[0.5, 0.7, 0.9],
        time_start="2020-02-02 12:00:00",
    )

    collection = TimeDataCollection(channels=[channel])
    with pytest.raises(
        ValueError, match="does not match the time type of the collection"
    ):
        collection.add_global_label(label)


def test_get_label():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(name="events", time_index=[0.5, 0.7, 0.9], anchored_channel=channel)

    label2 = Label(name="events 2", time_index=[0.1, 0.5])
    channel.attach_label(label2)

    collection = TimeDataCollection(channels=[channel], labels=[label])
    assert collection.get_label(name="events") == label

    with pytest.raises(ValueError, match="ambiguous"):
        assert collection.get_label(name="nonexistent")

    with pytest.raises(ValueError, match="ambiguous"):
        collection.get_label()


def test_remove_label():
    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    label = Label(name="events", time_index=[0.5, 0.7, 0.9], anchored_channel=channel)
    label2 = Label(name="events 2", time_index=[0.1, 0.5])
    channel.attach_label(label2)

    collection = TimeDataCollection(channels=[channel], labels=[label, label2])
    assert len(collection.labels) == 2
    assert collection.label_names == ["events", "events 2"]

    collection.remove_label(name="events")
    assert len(collection.labels) == 1
    assert collection.label_names == ["events 2"]

    with pytest.raises(ValueError, match="ambiguous"):
        collection.remove_label(name="nonexistent")

    with pytest.raises(ValueError, match="not in the collection"):
        collection.remove_label(label=label)


def test_delete_nonexistent_channel():
    collection = TimeDataCollection()

    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    collection.add_channel(Channel(name="test", time_index=time, data=data))

    assert len(collection.channels) == 1
    assert collection.channel_names == ["test"]

    with pytest.raises(ValueError) as exc:
        collection.remove_channel(name="nonexistent")
    assert str(exc.value) == (
        "Channel specification was ambiguous, no unique channel was "
        "identified. Query returned: []"
    )

    other_channel = Channel(name="other", time_index=time, data=2 * data)
    with pytest.raises(ValueError, match="not in the collection"):
        collection.remove_channel(channel=other_channel)


def test_add_existing_channel():
    collection = TimeDataCollection()

    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    channel = Channel(name="test", time_index=time, data=data)
    collection.add_channel(channel)

    with pytest.raises(ValueError) as exc:
        collection.add_channel(channel)
    assert (
        str(exc.value)
        == "Identical channel test has already been added to the collection"
    )


def test_channels_with_timestamps():
    collection = TimeDataCollection()

    time_1 = pd.timedelta_range(start="0ms", periods=100, freq="5ms")
    time_2 = pd.timedelta_range(start="0ms", periods=10, freq="35ms")

    channel_1 = Channel(
        name="Test 1",
        time_index=time_1,
        time_start=pd.Timestamp("2020-02-02 12:00:00.000"),
        data=np.random.rand(100),
    )
    channel_2 = Channel(
        name="Test 2",
        time_index=time_2,
        data=np.random.rand(10),
        time_start=pd.Timestamp("2020-02-02 12:00:00.300"),
    )

    collection.add_channel(channel_1)
    collection.add_channel(channel_2)


def test_collection_get_channels():
    collection = TimeDataCollection()

    time = np.arange(0, 1, 0.1)
    channels = [
        Channel(
            name=f"Random {i}",
            time_index=time,
            data=np.random.rand(len(time)),
            metadata={
                "creator": "Tester",
                "project_id": i % 2,
            },
        )
        for i in range(5)
    ]
    for channel in channels:
        collection.add_channel(channel)

    assert len(collection.get_channels()) == 5
    assert len(collection.get_channels(name="Random 3")) == 1
    assert len(collection.get_channels(metadata={"creator": "Tester"})) == 5
    assert len(collection.get_channels(metadata={"project_id": 1})) == 2


def test_collection_add_empty_label():
    collection = TimeDataCollection()
    channel = Channel(
        name="test", time_index=np.arange(0, 1, 0.1), data=np.random.rand(10)
    )
    collection.add_channel(channel)

    label = Label(name="empty label")
    collection.add_global_label(label)

    local_label = Label(name="empty local label")
    channel.attach_label(local_label)


def test_set_channel_plotstyle():
    collection = TimeDataCollection()

    time = np.arange(0, 1, 0.1)
    data = np.sin(2 * np.pi * time)
    plotstyle = {"color": "red", "linestyle": "--"}
    channels = []
    for idx in range(6):
        channel = Channel(
            name=f"ch{idx}",
            time_index=time,
            data=data * idx,
            plotstyle=plotstyle,
            metadata={"parity": idx % 2},
        )
        channels.append(channel)
    collection = TimeDataCollection(channels=channels)

    assert collection.get_channel("ch0").plotstyle == plotstyle

    collection.set_channel_plotstyle("ch2", color="blue")
    assert collection.get_channel("ch2").plotstyle == {
        "color": "blue",
        "linestyle": "--",
    }

    collection.set_channel_plotstyle(
        {"metadata": {"parity": 1}}, linestyle=":", label="odd"
    )
    expected_plotstyle = {"label": "odd", "linestyle": ":", "color": "red"}
    for channel_name in ["ch1", "ch3", "ch5"]:
        assert collection.get_channel(channel_name).plotstyle == expected_plotstyle

    channel = collection.get_channel("ch5")
    collection.set_channel_plotstyle(channel, color="yellow")
    assert channel.plotstyle == {"label": "odd", "linestyle": ":", "color": "yellow"}

    collection.set_channel_plotstyle(linestyle=None, label=None, color=None)
    assert all(channel.plotstyle == {} for channel in collection.channels)


def test_set_label_plotstyle():
    time = np.arange(0, 1, 0.1)
    plotstyle = {"color": "red", "linestyle": "", "marker": "o"}
    labels = []
    for idx in range(6):
        label = Label(
            name=f"lab{idx}",
            time_index=time,
            plotstyle=plotstyle,
            metadata={"parity": idx % 2},
        )
        labels.append(label)
    collection = TimeDataCollection(labels=labels)

    assert collection.get_label("lab0").plotstyle == plotstyle
    # check that the plotstyle is a proper copy, and not the identical object
    assert collection.get_label("lab0") is not plotstyle

    collection.set_label_plotstyle("lab2", color="blue")
    assert collection.get_label("lab2").plotstyle == {
        "color": "blue",
        "linestyle": "",
        "marker": "o",
    }

    collection.set_label_plotstyle({"metadata": {"parity": 1}}, label="odd", marker="x")
    expected_plotstyle = {
        "label": "odd",
        "linestyle": "",
        "color": "red",
        "marker": "x",
    }
    for label_name in ["lab1", "lab3", "lab5"]:
        assert collection.get_label(label_name).plotstyle == expected_plotstyle

    label = collection.get_label("lab5")
    collection.set_label_plotstyle(label, color="yellow")
    assert label.plotstyle == {
        "label": "odd",
        "linestyle": "",
        "color": "yellow",
        "marker": "x",
    }

    collection.set_label_plotstyle(linestyle=None, label=None, color=None, marker=None)
    assert all(label.plotstyle == {} for label in collection.labels)


def test_serialization():
    collection = get_random_collection(
        num_channels=10, num_labels=50, num_attached_labels=20
    )
    serialized_collection = json.dumps(collection.to_dict(), cls=NumpyEncoder)

    cloned_collection = TimeDataCollection.from_dict(json.loads(serialized_collection))

    assert collection.channel_data_hash() == cloned_collection.channel_data_hash()
    assert collection.label_names == cloned_collection.label_names
