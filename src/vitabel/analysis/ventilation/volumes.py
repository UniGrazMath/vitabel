"""Utilities for deriving ventilation-volume metrics from phase labels."""

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
    """Integrate a sampled signal cumulatively with the trapezoidal rule.

    Parameters
    ----------
    signal
        Input samples to integrate. The time index is expected to be ordered and
        to have the same length as the data array.
    correction_factor
        Multiplicative factor applied to the data before integration. This is
        typically used for unit conversion or device-specific flow correction.

    Returns
    -------
    DataSlice
        Cumulative integral with the same time index as the input.
    """
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
    """Concatenate multiple data slices and sort the result by time.

    Parameters
    ----------
    dataslices
        Data slices to combine. All slices are expected to contain numeric data.

    Returns
    -------
    DataSlice
        A single chronologically ordered data slice containing all samples.
    """
    time_index = pd.Index(np.concatenate([ds.time_index.values for ds in dataslices]))
    data = np.concatenate([ds.data for ds in dataslices])

    order = np.argsort(time_index.values)
    return DataSlice(
        time_index=time_index[order],
        data=data[order],
        text_data=None,
    )


def argmax_dataslices(dataslices: list[DataSlice]) -> DataSlice:
    """Extract the maximum value and its timestamp from each data slice.

    Empty slices and slices containing only ``NaN`` values are ignored.

    Parameters
    ----------
    dataslices
        Data slices for which peak values should be determined.

    Returns
    -------
    DataSlice
        One timestamp/value pair per valid input slice.
    """
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
    """Create breath-duration and respiratory-rate labels from phase intervals.

    Parameters
    ----------
    inspiration
        Inspiration intervals.
    expiration
        Expiration intervals.

    Returns
    -------
    tuple[Label, Label, Label]
        Labels for inspiratory time, expiratory time, and respiratory rate.
    """
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
    """Create per-breath inspiratory pressure summary labels.

    Parameters
    ----------
    pressure
        Pressure channel sampled on the common respiratory-phase time base.
    insp_t
        Array of inspiration intervals as ``(start, stop)`` pairs.

    Returns
    -------
    tuple[Label, Label]
        Labels for maximal and minimal inspiratory airway pressure.
    """
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


def _compute_cycle_integrals(
    flow: Channel,
    insp_t: np.ndarray,
    correction_factor: float,
) -> list[DataSlice]:
    """Integrate flow between consecutive inspiration starts."""
    return [
        integrate_trapezoid(
            signal=flow.truncate(start_time=start, stop_time=stop),
            correction_factor=correction_factor,
        )
        for start, stop in zip(insp_t[:-1, 0], insp_t[1:, 0])
    ]


def compute_volume_channels_and_labels(
    flow: Channel,
    inspiration: IntervalLabel,
    expiration: IntervalLabel,
    correction_factor: float,
) -> tuple[Channel, Channel, Channel, Label, Label, Label]:
    """Compute primary ventilation-volume channels and tidal-volume labels.

    This function derives the net cycle integral (``Volume``), a phase-specific
    inspiratory volume curve, a phase-specific expiratory volume curve, and the
    associated tidal-volume summary labels.

    Parameters
    ----------
    flow
        Interpolated flow channel.
    inspiration
        Inspiration intervals.
    expiration
        Expiration intervals.
    correction_factor
        Multiplicative factor applied before integrating flow.

    Returns
    -------
    tuple[Channel, Channel, Channel, Label, Label, Label]
        ``(volume, inspiratory_volume, expiratory_volume, delta_vt, vt_insp,
        vt_exp)``.
    """
    insp_t = inspiration.get_data().time_index
    exp_t = expiration.get_data().time_index
    cycle_integrals = _compute_cycle_integrals(flow, insp_t, correction_factor)

    volume_segments = [
        DataSlice(
            time_index=ds.time_index[:-1],
            data=ds.data[:-1],
            text_data=None,
        )
        for ds in cycle_integrals
    ]
    volume_data = concat_and_sort_dataslices(volume_segments)
    volume = Channel(
        name="Volume",
        time_index=volume_data.time_index,
        data=volume_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Volume"),
    )

    delta_vt_data = concat_and_sort_dataslices(
        [
            DataSlice(
                time_index=pd.Index([ds.time_index[-1]]),
                data=[ds.data[-1]],
                text_data=None,
            )
            for ds in cycle_integrals
        ]
    )
    delta_vt = Label(
        name="Delta VT",
        time_index=delta_vt_data.time_index,
        data=delta_vt_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Delta VT"),
    )

    volume_start = volume.first_entry
    volume_end = volume.last_entry
    inspiration_volume_segments = [
        volume.truncate(start_time=start, stop_time=stop)
        for start, stop in insp_t
        if start >= volume_start and stop <= volume_end
    ]
    vt_insp_data = argmax_dataslices(inspiration_volume_segments)
    vt_insp = Label(
        name="VTinsp",
        time_index=vt_insp_data.time_index,
        data=vt_insp_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Inspiratory Tidal Volume"),
    )
    volume.attach_label(vt_insp)
    volume.attach_label(delta_vt)

    inspiratory_volume_parts = [
        DataSlice(time_index=ds.get_data().time_index, data=ds.data, text_data=None)
        for ds in inspiration_volume_segments
    ]
    inspiratory_plateaus = [
        DataSlice(
            time_index=ds.get_data().time_index[1:-1],
            data=(
                ds.data
                if ds.data.size == 0
                else np.full_like(ds.data[1:-1], ds.data[0])
            ),
            text_data=None,
        )
        for ds in [
            volume.truncate(start_time=start, stop_time=stop)
            for start, stop in exp_t
            if start >= volume_start and stop < volume_end
        ]
    ]
    inspiratory_volume_parts.extend(inspiratory_plateaus)
    inspiratory_volume_data = concat_and_sort_dataslices(inspiratory_volume_parts)
    inspiratory_volume = Channel(
        name="Inspiratory Volume",
        time_index=inspiratory_volume_data.time_index,
        data=inspiratory_volume_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Inspiratory Volume"),
    )

    expiratory_segments = [
        DataSlice(
            time_index=ds.get_data().time_index[:-1],
            data=-1.0 * (ds.data[:-1] - ds.data[0]),
            text_data=None,
        )
        for ds in [
            volume.truncate(start_time=start, stop_time=stop)
            for start, stop in exp_t
            if start >= volume_start and stop <= volume_end
        ]
    ]
    vt_exp_data = argmax_dataslices(expiratory_segments)
    vt_exp = Label(
        name="VTexp",
        time_index=vt_exp_data.time_index,
        data=vt_exp_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Expiratory Tidal Volume"),
    )
    insp_mask = inspiration.contains_time(
        flow.get_data().time_index.to_numpy(),
        include_start=True,
        include_end=False,
    )
    expiratory_segments.append(
        DataSlice(
            time_index=flow.get_data().time_index[insp_mask],
            data=np.zeros(np.sum(insp_mask), dtype=float),
            text_data=None,
        )
    )
    expiratory_volume_data = concat_and_sort_dataslices(expiratory_segments)
    expiratory_volume = Channel(
        name="Expiratory Volume",
        time_index=expiratory_volume_data.time_index,
        data=expiratory_volume_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Expiratory Volume"),
    )
    expiratory_volume.attach_label(vt_exp)

    return volume, inspiratory_volume, expiratory_volume, delta_vt, vt_insp, vt_exp


def compute_cumulative_volume_channels(
    flow: Channel,
    inspiration: IntervalLabel,
    correction_factor: float,
) -> tuple[Channel, Channel, Label, Label]:
    """Compute cumulative inspiratory and expiratory volume channels.

    Positive and negative flow are integrated separately on a cycle basis to
    obtain cumulative inspiratory and expiratory volume curves.

    Parameters
    ----------
    flow
        Interpolated flow channel.
    inspiration
        Inspiration intervals used to define complete breathing cycles.
    correction_factor
        Multiplicative factor applied before integrating flow.

    Returns
    -------
    tuple[Channel, Channel, Label, Label]
        ``(cumulative_inspiratory_volume, cumulative_expiratory_volume,
        vt_insp_cum, vt_exp_cum)``.
    """
    insp_t = inspiration.get_data().time_index
    cycle_ranges = zip(insp_t[:-1, 0], insp_t[1:, 0])

    inspiratory_cycles = [
        integrate_trapezoid(
            signal=DataSlice(
                time_index=fs.get_data().time_index[:-1],
                data=np.maximum(fs.data[:-1], 0.0),
                text_data=None,
            ),
            correction_factor=correction_factor,
        )
        for fs in (
            flow.truncate(start_time=start, stop_time=stop)
            for start, stop in cycle_ranges
        )
    ]
    vt_insp_cum_data = argmax_dataslices(inspiratory_cycles)
    vt_insp_cum = Label(
        name="VTinsp_cum",
        time_index=vt_insp_cum_data.time_index,
        data=vt_insp_cum_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Cumulative Inspiratory Tidal Volume"),
    )
    inspiratory_volume_data = concat_and_sort_dataslices(inspiratory_cycles)
    cumulative_inspiratory_volume = Channel(
        name="Cumulative Inspiratory Volume",
        time_index=inspiratory_volume_data.time_index,
        data=inspiratory_volume_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Cumulative Inspiratory Volume"),
    )
    cumulative_inspiratory_volume.attach_label(vt_insp_cum)

    cycle_ranges = zip(insp_t[:-1, 0], insp_t[1:, 0])
    expiratory_cycles = [
        integrate_trapezoid(
            signal=DataSlice(
                time_index=fs.get_data().time_index[:-1],
                data=-1.0 * np.minimum(fs.data[:-1], 0.0),
                text_data=None,
            ),
            correction_factor=correction_factor,
        )
        for fs in (
            flow.truncate(start_time=start, stop_time=stop)
            for start, stop in cycle_ranges
        )
    ]
    vt_exp_cum_data = argmax_dataslices(expiratory_cycles)
    vt_exp_cum = Label(
        name="VTexp_cum",
        time_index=vt_exp_cum_data.time_index,
        data=vt_exp_cum_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Cumulative Expiratory Tidal Volume"),
    )
    expiratory_volume_data = concat_and_sort_dataslices(expiratory_cycles)
    cumulative_expiratory_volume = Channel(
        name="Cumulative Expiratory Volume",
        time_index=expiratory_volume_data.time_index,
        data=expiratory_volume_data.data,
        metadata={"source": "computed"},
        plotstyle=DEFAULT_PLOT_STYLE.get("Cumulative Expiratory Volume"),
    )
    cumulative_expiratory_volume.attach_label(vt_exp_cum)

    return (
        cumulative_inspiratory_volume,
        cumulative_expiratory_volume,
        vt_insp_cum,
        vt_exp_cum,
    )


def compute_reverse_airflow_labels(
    flow: Channel,
    inspiration: IntervalLabel,
    expiration: IntervalLabel,
    correction_factor: float,
) -> tuple[Label, IntervalLabel, Label, Label, IntervalLabel, Label]:
    """Compute labels describing reverse airflow within respiratory phases.

    Reverse airflow is quantified on zero-crossing-delimited flow segments and
    then summarized per inspiration and per expiration.

    Parameters
    ----------
    flow
        Interpolated flow channel.
    inspiration
        Inspiration intervals.
    expiration
        Expiration intervals.
    correction_factor
        Multiplicative factor applied before integrating segment flow.

    Returns
    -------
    tuple[Label, IntervalLabel, Label, Label, IntervalLabel, Label]
        ``(insp_reverse_volume, insp_reverse_airflow, insp_reverse_sum,
        exp_reverse_volume, exp_reverse_airflow, exp_reverse_sum)``.
    """

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
