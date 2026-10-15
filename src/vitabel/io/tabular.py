"""Construction of channels and labels from tabular data (dictionaries,
``pandas.DataFrame`` objects, and CSV files)."""

from __future__ import annotations

import json
import logging

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from vitabel.timeseries import Channel, IntervalLabel, Label

logger = logging.getLogger("vitabel")

DataType = Literal["channel", "label", "interval_label"]


def series_from_dict(
    source: dict[str, Any],
    name: str,
    time_start=None,
    datatype: DataType = "channel",
    anchored_channel: Channel | None = None,
    metadata: dict | None = None,
) -> Channel | Label | None:
    """Construct a channel or label from a dictionary containing a single
    time series, of the form ``{'timestamp': [...], 'data': [...]}``.

    Parameters
    ----------
    source
        The data, in the form ``{'timestamp': [...], 'data': [...]}``.
    name
        The name of the channel or label.
    time_start
        ``time_start`` of the time series, in case of relative time data.
    datatype
        Either ``'channel'``, ``'label'``, or ``'interval_label'``, depending
        on which kind of container to construct.
    anchored_channel
        If a label is constructed, the channel to which it is attached.
        Irrelevant if ``datatype`` is ``'channel'``.
    metadata
        Metadata of the time series.

    Returns
    -------
    The constructed channel or label, or ``None`` if the data is empty.

    Raises
    ------
    ValueError
        In case the dictionary does not contain the keys 'timestamp' and 'data'.
    """
    if not ("timestamp" in source.keys() and "data" in source.keys()):
        raise ValueError(
            "The dictionary must contain a 'timestamp' and a 'data' key which contain timestamps and data for this channel. \n \
                             In case of time_only or 'time_interval' channel_types choose 'data' to be an empty list."
        )

    if time_start:
        time_start = pd.Timestamp(time_start)

    time = source["timestamp"]
    data = source["data"]
    if len(time) == 0:
        logger.warning("The data is empty. No channel or label will be added.")
        return None

    if datatype == "channel":
        return Channel(
            name,
            time,
            data,
            metadata=metadata,
            time_start=time_start,
        )
    elif datatype == "label":
        return Label(
            name,
            time,
            data,
            metadata=metadata,
            time_start=time_start,
            anchored_channel=anchored_channel,
        )
    elif datatype == "interval_label":
        return IntervalLabel(
            name,
            time,
            data,
            metadata=metadata,
            time_start=time_start,
            anchored_channel=anchored_channel,
        )
    return None


def iter_series_from_dict(
    source: dict[str, dict] | Path,
    time_start=None,
    datatype: DataType = "channel",
    anchored_channel: Channel | None = None,
    metadata: dict | None = None,
) -> Iterator[Channel | Label]:
    """Construct channels or labels from a dictionary of the form
    ``{'name1': {'timestamp': [...], 'data': [...]}, 'name2': {...}, ...}``,
    see :func:`series_from_dict`. Empty time series are skipped.

    If ``source`` is a ``Path``, it is loaded as JSON file. The time series
    are constructed lazily (one by one, while iterating).

    Raises
    ------
    ValueError
        In case the dictionary does not have the expected form.
    """
    if isinstance(source, Path):
        with open(source, "r") as file:
            source = json.load(file)

    for key in source:
        if not isinstance(source[key], dict):
            raise ValueError(
                "Source must be a dictionary of the form "
                "{'channel1': {'timestamp': [...], 'data': [...]},"
                " 'channel2': { ... }, ... }. "
                f"For key {key} the value is not a dict."
            )
        series = series_from_dict(
            source[key],
            key,
            metadata=metadata,
            time_start=time_start,
            datatype=datatype,
            anchored_channel=anchored_channel,
        )
        if series is not None:
            yield series


def iter_series_from_dataframe(
    source: pd.DataFrame,
    time_start: str | None = None,
    time_unit=None,
    datatype: DataType = "channel",
    anchored_channel: Channel | None = None,
    metadata: dict[str, Any] | None = None,
    column_metadata: dict[str, dict[str, Any]] | None = None,
) -> Iterator[Channel | Label]:
    """Construct channels or labels from the columns of a ``pandas.DataFrame``.

    The index of the DataFrame contains the time (``DatetimeIndex``,
    ``TimedeltaIndex``, or numeric). NaN values are ignored, empty columns
    are skipped. The time series are constructed lazily (one by one, while
    iterating). See :meth:`.Vitals.add_data_from_DataFrame` for a description
    of the parameters.

    Raises
    ------
    ValueError
        The DataFrame does not have a datetime, timedelta, or numeric index.
    NotImplementedError
        For ``datatype='interval_label'``.
    """
    if metadata is None:
        metadata = {}
    if column_metadata is None:
        column_metadata = {}

    if not (
        isinstance(source.index, (pd.DatetimeIndex, pd.TimedeltaIndex))
        or (pd.api.types.is_numeric_dtype(source.index))
    ):
        raise ValueError(
            "The DataFrame needs to have a datetime or a numeric index, "
            "which describes the time of the timeseries."
        )

    for col in source.columns:
        series = source[col]
        series = series[series.notna()]
        time = np.array(series.index)
        data = series.values
        if len(time) == 0:
            continue

        if datatype == "channel":
            yield Channel(
                name=col,
                time_index=time,
                data=data,
                time_start=time_start,
                time_unit=time_unit,
                metadata=metadata | column_metadata.get(col, {}),
            )
        elif datatype == "label":
            yield Label(
                name=col,
                time_index=time,
                data=data,
                time_start=time_start,
                time_unit=time_unit,
                metadata=metadata | column_metadata.get(col, {}),
                anchored_channel=anchored_channel,
            )
        elif datatype == "interval_label":
            raise NotImplementedError(
                "Interval labels can currently not be added from DataFrames."
            )


def read_csv_table(file_path: Path | str, **kwargs) -> pd.DataFrame:
    """Read a CSV file into a ``pandas.DataFrame`` (via ``pandas.read_csv``,
    which receives all keyword arguments)."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
    return pd.read_csv(file_path, **kwargs)
