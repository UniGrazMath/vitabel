"""Helpers for ventilation-volume analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vitabel.timeseries import Channel, IntervalLabel, Label
from vitabel.typing import DataSlice
from vitabel.utils import DEFAULT_PLOT_STYLE


def integrate_trapezoid(
    signal: DataSlice,
    correction_factor: float = 1.0,
) -> DataSlice:
    """Integrate data with the trapezoidal rule over a full data slice."""
    if len(signal) < 2:
        raise ValueError("Length of time and data must be at least 2")
    if signal.data is None:
        raise ValueError("DataSlice contains no data to integrate")
    if len(signal.data) != len(signal.time_index):
        raise ValueError("Length of time and data must agree")

    data = signal.data * correction_factor
    dt = signal.time_index[1:] - signal.time_index[:-1]
    sample_sum = data[1:] + data[:-1]
    changes = sample_sum * 0.5 * dt.total_seconds()
    cumulative = np.concatenate(([0.0], np.cumsum(changes)))

    return DataSlice(
        time_index=(
            signal.get_data().time_index
            if isinstance(signal, (Channel, Label))
            else signal.time_index
        ),
        data=cumulative,
    )


def concat_and_sort_dataslices(dataslices: list[DataSlice]) -> DataSlice:
    """Concatenate and sort data slices into a single data slice."""
    time_index = pd.Index(np.concatenate([ds.time_index.values for ds in dataslices]))
    data = np.concatenate([ds.data for ds in dataslices])

    order = np.argsort(time_index.values)
    return DataSlice(
        time_index=time_index[order],
        data=data[order],
        text_data=None,
    )


def argmax_dataslices(dataslices: list[DataSlice]) -> DataSlice:
    """Compute the argmax within each data slice of a list."""
    valid = [
        ds
        for ds in dataslices
        if ds.data is not None and len(ds.data) > 0 and not np.all(np.isnan(ds.data))
    ]

    return DataSlice(
        time_index=pd.Index(
            [
                ds.get_data().time_index[int(np.nanargmax(ds.data))]
                if isinstance(ds, (Channel, Label))
                else ds.time_index[int(np.nanargmax(ds.data))]
                for ds in valid
            ]
        ),
        data=np.array([float(np.nanmax(ds.data)) for ds in valid]),
        text_data=None,
    )


def compute_breath_duration_and_rate_labels(
    inspiration: IntervalLabel,
    expiration: IntervalLabel,
) -> tuple[Label, Label, Label]:
    """Compute inspiratory/expiratory durations and respiratory rate labels."""
    insp_t = inspiration.get_data().time_index
    exp_t = expiration.get_data().time_index

    inspiratory_time = Label(
        name="Inspiratory Time",
        time_index=insp_t[:, 1],
        data=(insp_t[:, 1] - insp_t[:, 0]) / np.timedelta64(1, "s"),
        metadata={"source": "computed"},
    )
    expiratory_time = Label(
        name="Expiratory Time",
        time_index=exp_t[:, 1],
        data=(exp_t[:, 1] - exp_t[:, 0]) / np.timedelta64(1, "s"),
        metadata={"source": "computed"},
    )
    respiratory_rate = Label(
        name="Respiratory Rate",
        time_index=insp_t[1:, 0],
        data=60 / ((insp_t[1:, 0] - insp_t[:-1, 0]) / np.timedelta64(1, "s")),
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Respiratory Rate"),
    )
    return inspiratory_time, expiratory_time, respiratory_rate


def compute_inspiratory_pressure_labels(
    pressure: Channel,
    insp_t: np.ndarray,
) -> tuple[Label, Label]:
    """Compute maximum and minimum inspiratory pressure labels."""
    pressure_during_inspiration = [
        pressure.truncate(start_time=start, stop_time=stop) for start, stop in insp_t
    ]
    p_insp_max_data = argmax_dataslices(pressure_during_inspiration)
    p_insp_max = Label(
        name="Maximal Inspiratory Airway Pressure",
        time_index=p_insp_max_data.time_index,
        data=p_insp_max_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Maximal Inspiratory Airway Pressure"),
    )
    p_insp_min = Label(
        name="Minimal Inspiratory Airway Pressure",
        time_index=insp_t[:, 0],
        data=np.array(
            [
                float(np.min(pressure.truncate(start_time=start, stop_time=stop).data))
                for start, stop in insp_t
            ]
        ),
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Minimal Inspiratory Airway Pressure"),
    )
    return p_insp_max, p_insp_min


def compute_reverse_airflow_labels(
    flow: Channel,
    inspiration: IntervalLabel,
    expiration: IntervalLabel,
    correction_factor: float,
) -> tuple[Label, IntervalLabel, Label, Label, IntervalLabel, Label]:
    """Compute reverse-airflow interval and summary labels."""

    def _area(y, time_index):
        dt = (time_index[1:] - time_index[:-1]).total_seconds()
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * dt))

    def _area_segment(start, end):
        mask = (flow.get_data().time_index >= start) & (
            flow.get_data().time_index <= end
        )
        return _area(
            flow.data[mask] * correction_factor,
            flow.get_data().time_index[mask],
        )

    crossing = np.array(flow.get_data().time_index[flow.data == 0])
    segment = np.array(list(zip(crossing, crossing[1:])))
    segment_volume = np.array([_area_segment(start, end) for start, end in segment])

    s = crossing[:-1]
    e = crossing[1:]
    insp_s = inspiration.get_data().time_index[:, 0]
    insp_e = inspiration.get_data().time_index[:, 1]
    exp_s = expiration.get_data().time_index[:, 0]
    exp_e = expiration.get_data().time_index[:, 1]

    in_insp = (insp_s[None, :] <= s[:, None]) & (e[:, None] <= insp_e[None, :])
    in_exp = (exp_s[None, :] <= s[:, None]) & (e[:, None] <= exp_e[None, :])

    neg_values = segment_volume < 0
    pos_values = segment_volume > 0
    mask_insp_reverse = in_insp.any(axis=1) & neg_values
    mask_exp_reverse = in_exp.any(axis=1) & pos_values

    insp_reverse_volume = Label(
        name="Inspiratory Reverse Airflow Segment Volume",
        time_index=crossing[1:][mask_insp_reverse],
        data=-1 * segment_volume[mask_insp_reverse],
        metadata={"source": "computed"},
    )
    insp_reverse_airflow = IntervalLabel(
        name="Inspiratory Reverse Airflow",
        time_index=segment[mask_insp_reverse].tolist(),
        data=None,
        metadata={"source": "computed"},
    )
    insp_reverse_sum = Label(
        name="Inspiratory Reverse Airflow Sum per Inspiration",
        time_index=insp_e,
        data=-(in_insp.T @ np.where(neg_values, segment_volume, 0.0)),
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Inspiratory Reverse Airflow"),
    )

    exp_reverse_volume = Label(
        name="Expiratory Reverse Airflow Segment Volume",
        time_index=crossing[1:][mask_exp_reverse],
        data=segment_volume[mask_exp_reverse],
        metadata={"source": "computed"},
    )
    exp_reverse_airflow = IntervalLabel(
        name="Expiratory Reverse Airflow",
        time_index=segment[mask_exp_reverse].tolist(),
        data=None,
        metadata={"source": "computed"},
    )
    exp_reverse_sum = Label(
        name="Expiratory Reverse Airflow Sum per Expiration",
        time_index=exp_e,
        data=(in_exp.T @ np.where(pos_values, segment_volume, 0.0)),
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Expiratory Reverse Airflow"),
    )

    return (
        insp_reverse_volume,
        insp_reverse_airflow,
        insp_reverse_sum,
        exp_reverse_volume,
        exp_reverse_airflow,
        exp_reverse_sum,
    )
