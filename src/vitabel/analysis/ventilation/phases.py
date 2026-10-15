"""Detection of respiratory phases (inspiration / expiration) from airflow
and airway pressure, see Orlob et al. :cite:`10.1016/j.resuscitation.2026.111050`."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from vitabel.typing import DataSlice

logger = logging.getLogger("vitabel")


def _compute_slope(dataslice: DataSlice) -> DataSlice:
    """Compute the time derivative (slope) of a DataSlice.

    Uses numpy's gradient function with central differences for interior points
    and forward/backward differences at edges.

    Parameters
    ----------
    dataslice : DataSlice
        A DataSlice containing the time index and data to differentiate.

    Returns
    -------
    DataSlice
        A new DataSlice with the same time index and the computed slope as data.
        The slope values have dtype float. If the input has fewer than 3 data points,
        returns a DataSlice with zeros (insufficient points for gradient computation).
    """
    index = dataslice.time_index
    data = dataslice.data

    if len(dataslice) < 3:
        return DataSlice(time_index=index, data=np.zeros(len(index), dtype=float))

    # Convert time to seconds (float64) from nanoseconds
    t_ns = index.view("int64")
    t_rel_sec = (t_ns - t_ns[0]).astype(np.float64) * 1e-9

    # Calculate the slope using central differences for interior points,
    # forward/backward for edges
    slope = np.gradient(data, t_rel_sec, edge_order=1)

    return DataSlice(time_index=index, data=slope)



def _find_segments_above_threshold(
    data: npt.NDArray,
    time_index: pd.DatetimeIndex,
    threshold: float,
    max_gap: np.timedelta64,
    min_duration: np.timedelta64,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Identify segments where data exceeds a threshold, with gap/duration filtering.

    This function detects contiguous segments of data above a threshold value.
    It merges segments separated by gaps shorter than `max_gap` and discards
    segments whose duration is less than `min_duration`.

    Parameters
    ----------
    data : np.ndarray
        The data array to analyze.
    time_index : pd.DatetimeIndex
        The corresponding time index for the data.
    threshold : float
        The threshold value; segments where data > threshold are detected.
    max_gap : np.timedelta64
        Maximum gap duration between above-threshold indices to consider them
        as part of the same segment. Gaps shorter than this are bridged.
    min_duration : np.timedelta64
        Minimum segment duration for a segment to be retained.

    Returns
    -------
    onsets_above_threshold : np.ndarray
        An array of integer indices where each detected segment starts (before
        duration filtering). Empty array (dtype int) if no values exceed the threshold.
    filtered_onsets : np.ndarray
        An array of integer segment start indices after filtering by minimum duration.
        Empty array (dtype int) if no segments survive filtering.

    Notes
    -----
    Both returned arrays always have dtype int and maintain consistent shapes
    even when empty, allowing safe unpacking and downstream processing.
    """
    # Find indices where data is above the threshold
    above_idxs = np.flatnonzero(data > threshold).astype(int)

    # Early exit: nothing above threshold
    if above_idxs.size == 0:
        onsets_above_threshold = np.empty(0, dtype=int)
        filtered_onsets = onsets_above_threshold
        return onsets_above_threshold, filtered_onsets

    # Identify breaks (gaps below threshold) greater than max_gap,
    # thereby filtering out short breaks that are likely due to oscillations.
    breaks = np.diff(time_index.take(above_idxs)) > max_gap

    # Determine segment boundaries
    (break_idxs,) = breaks.nonzero()
    segment_starts_pos = np.r_[0, break_idxs + 1]
    segment_stops_pos = np.r_[break_idxs, above_idxs.size - 1]
    onsets_above_threshold = above_idxs[segment_starts_pos]

    # Early exit: no segments found
    if onsets_above_threshold.size == 0:
        filtered_onsets = onsets_above_threshold
        return onsets_above_threshold, filtered_onsets

    # Filter out segments shorter than min_duration
    starts = time_index.take(above_idxs[segment_starts_pos])
    stops = time_index.take(above_idxs[segment_stops_pos])
    segment_width = stops - starts
    segments_width_filter = segment_width >= min_duration
    filtered_onsets = onsets_above_threshold[segments_width_filter]

    return onsets_above_threshold, filtered_onsets



def _filter_alternating_phases(
    candidate_phase_idxs: np.ndarray, anchor_phase_idxs: np.ndarray
) -> np.ndarray:
    """Filter phase candidates to ensure alternation with anchor phases.

    This function ensures that respiratory phases (inspiration/expiration) strictly
    alternate by keeping only the first candidate in each interval defined by the
    anchor phases.

    Algorithm
    ---------
    1. Divide the timeline into intervals using anchor phases as boundaries:
       ``(-inf, anchor_0], (anchor_0, anchor_1], (anchor_1, anchor_2], ...``
    2. Assign each candidate to the interval it falls into.
    3. Keep only the first (earliest) candidate per interval.

    This guarantees that between any two consecutive anchor phases, at most one
    candidate phase is selected, enforcing the alternating pattern required for
    valid respiratory cycles.

    Parameters
    ----------
    candidate_phase_idxs : np.ndarray
        Indices of potential phase onsets (e.g., inspiration starts) to be filtered.
        Must be sorted in ascending order.
    anchor_phase_idxs : np.ndarray
        Indices of the complementary phase onsets (e.g., expiration starts) that
        define the interval boundaries. Must be sorted in ascending order.

    Returns
    -------
    np.ndarray
        Filtered indices containing at most one candidate per interval, preserving
        alternation with anchor phases. If ``candidate_phase_idxs`` is empty or
        ``anchor_phase_idxs`` is empty, returns ``candidate_phase_idxs`` unchanged.
        The dtype of the returned array matches the input ``candidate_phase_idxs``.

    Examples
    --------
    Given anchor phases at indices [100, 200, 300] and candidate phases at
    indices [50, 120, 150, 250]:

    - Interval (-inf, 100]: candidate 50, keep 50
    - Interval (100, 200]: candidates 120, 150, keep 120 (first only)
    - Interval (200, 300]: candidate 250, keep 250

    Result: [50, 120, 250]
    """
    if candidate_phase_idxs.size == 0 or anchor_phase_idxs.size == 0:
        return candidate_phase_idxs

    # Create interval boundaries from anchor phases, with infinity sentinel
    # to capture any candidates after the last anchor.
    # Intervals: (-inf, anchor_0], (anchor_0, anchor_1], ..., (anchor_n, +inf)
    interval_boundaries = np.r_[anchor_phase_idxs, np.inf]

    # Assign each candidate to an interval index.
    # searchsorted with side="right" gives the index of the first boundary > candidate,
    # so subtracting 1 gives the interval the candidate belongs to.
    interval_idx = (
        np.searchsorted(interval_boundaries, candidate_phase_idxs, side="right") - 1
    )

    # Determine which candidates fall within valid intervals.
    # Valid intervals are those bounded by actual anchor phases (not the sentinel).
    is_valid_interval = (interval_idx >= 0) & (
        interval_idx < len(interval_boundaries) - 1
    )

    # Check that candidates are strictly within their interval bounds:
    # candidate > lower_bound AND candidate <= upper_bound
    is_within_bounds = np.zeros_like(candidate_phase_idxs, dtype=bool)
    is_within_bounds[is_valid_interval] = (
        candidate_phase_idxs[is_valid_interval]
        > interval_boundaries[interval_idx[is_valid_interval]]
    ) & (
        candidate_phase_idxs[is_valid_interval]
        <= interval_boundaries[interval_idx[is_valid_interval] + 1]
    )

    # Special case: include candidates before the first anchor phase.
    # These belong to the implicit interval (-inf, anchor_0].
    if candidate_phase_idxs[0] < interval_boundaries[0]:
        is_within_bounds[0] = True

    # Select candidates that passed the bounds check
    selected_intervals = interval_idx[is_within_bounds]
    selected_candidates = candidate_phase_idxs[is_within_bounds]

    # Keep only the first candidate per interval.
    # np.unique returns indices of first occurrences when array is sorted.
    _, first_occurrence_idx = np.unique(selected_intervals, return_index=True)

    return selected_candidates[first_occurrence_idx]



def _create_phase_intervals(
    insp_begins: pd.DatetimeIndex,
    exp_begins: pd.DatetimeIndex,
) -> tuple[list[tuple], list[tuple]]:
    """Create inspiration and expiration intervals from phase onset times.

    Given the timestamps of inspiration and expiration phase beginnings,
    this function creates intervals by pairing consecutive phase boundaries:
    - Inspiration intervals: from each inspiration start to the next expiration start
    - Expiration intervals: from each expiration start to the next inspiration start

    This assumes the phases alternate (insp -> exp -> insp -> ...) and only
    creates intervals where proper pairing is possible.

    Parameters
    ----------
    insp_begins : pd.DatetimeIndex
        Timestamps of inspiration phase onsets, sorted chronologically.
    exp_begins : pd.DatetimeIndex
        Timestamps of expiration phase onsets, sorted chronologically.

    Returns
    -------
    insp_intervals : list[tuple]
        List of (start, end) tuples for inspiration phases. Each interval
        spans from an inspiration onset to the following expiration onset.
    exp_intervals : list[tuple]
        List of (start, end) tuples for expiration phases. Each interval
        spans from an expiration onset to the following inspiration onset.

    Examples
    --------
    >>> import pandas as pd
    >>> insp = pd.DatetimeIndex(['2021-01-01 00:00:00', '2021-01-01 00:00:04'])
    >>> exp = pd.DatetimeIndex(['2021-01-01 00:00:02', '2021-01-01 00:00:06'])
    >>> insp_ivl, exp_ivl = _create_phase_intervals(insp, exp)
    >>> insp_ivl  # [(insp[0], exp[0]), (insp[1], exp[1])]
    [(Timestamp('2021-01-01 00:00:00'), Timestamp('2021-01-01 00:00:02')),
     (Timestamp('2021-01-01 00:00:04'), Timestamp('2021-01-01 00:00:06'))]
    >>> exp_ivl  # [(exp[0], insp[1])]
    [(Timestamp('2021-01-01 00:00:02'), Timestamp('2021-01-01 00:00:04'))]

    Notes
    -----
    - If either input is empty, both outputs will be empty lists.
    - Inspiration intervals pair each inspiration with the next expiration
      that occurs after it (and before the last expiration).
    - Expiration intervals pair each expiration with the next inspiration
      that occurs after it (and before the last inspiration).
    """
    # Early exit: can't form intervals without both phase types
    if len(insp_begins) == 0 or len(exp_begins) == 0:
        return [], []

    # --- Inspiration intervals: [insp_start, next exp_start) ---
    # Only include inspirations that have a following expiration
    i_first = insp_begins[0]
    e_last = exp_begins[-1]
    i_for_insp = insp_begins[insp_begins < e_last]
    e_for_insp = exp_begins[exp_begins > i_first]
    insp_intervals = list(zip(i_for_insp, e_for_insp))

    # --- Expiration intervals: [exp_start, next insp_start) ---
    # Only include expirations that have a following inspiration
    i_last = insp_begins[-1]
    e_first = exp_begins[0]
    e_for_exp = exp_begins[exp_begins < i_last]
    i_for_exp = insp_begins[insp_begins > e_first]
    exp_intervals = list(zip(e_for_exp, i_for_exp))

    return insp_intervals, exp_intervals


def get_inspiration_starts(
    product: DataSlice,
    interpolated_flow: DataSlice,
    interpolated_pressure: DataSlice,
    slope_pressure: DataSlice,
    threshold: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Derives the indices of the start of inspirations from the product of flow and pressure.

    This function identifies segments of the product that are above a certain threshold
    and finds the last zero crossing before each segment. Preceding segments
    of less than 20ms of a non-positive product are filtered out to ensure that chest
    compressions do not fragment the product in the early phase of the inspiration.
    Potential candidates are further filtered by the duration of the following
    duration of zeros, which must be greater than 400ms (more than half the duty cycle of CC with 80 BPM)
    to be considered a valid inspiration. This avoids false positives due to chest compressions fragmenting the product.

    Parameters
    ----------
    product:class:`.DataSlice`
        A DataSlice object containing the time index and data of the product of flow and pressure.

    interpolated_flow:class:`.DataSlice`
        A DataSlice object containing the time index and data of the interpolated flow signal.
        Must have the same time index as the product.

    interpolated_pressure:class:`.DataSlice`
        A DataSlice object containing the time index and data of the interpolated pressure signal.
        Must have the same time index as the product.

    slope_pressure:class:`.DataSlice`
        A DataSlice object containing the time index and data of the slope of the pressure signal.
        Must have the same time index as the product.

    threshold: float
        The threshold value above which the inspiration segments are detected.

    Returns
    -------
    inspiration_starts : np.ndarray
        An array of integer indices for the time index of the given product where
        the inspirations start (filtered zero crossings). Empty array (dtype int)
        if no valid inspiration starts are detected.
    onsets_above_threshold : np.ndarray
        An array of integer indices where the product first exceeds the threshold
        for each detected segment. Empty array (dtype int) if no segments are detected.
    filtered_onsets : np.ndarray
        An array of integer indices for the onsets above threshold after additional
        filtering (e.g., slope and pressure criteria). Empty array (dtype int) if
        no onsets survive filtering.

    Notes
    -----
    All three returned arrays always have dtype int and maintain consistent shapes
    even when empty, allowing safe tuple unpacking and downstream processing.
    """

    # Filter parameters
    landmark_duration_max_interruption = np.timedelta64(15, "ms")
    landmark_min_duration = np.timedelta64(20, "ms")
    landmark_cooldown_duration = np.timedelta64(375, "ms")

    reverse_airflow_duration_limit = np.timedelta64(70, "ms")

    if not product.time_index.equals(interpolated_flow.time_index):
        raise ValueError(
            "The time indices of product and interpolated_flow must match."
        )

    if not product.time_index.equals(interpolated_pressure.time_index):
        raise ValueError(
            "The time indices of product and interpolated_pressure must match."
        )

    if not product.time_index.equals(slope_pressure.time_index):
        raise ValueError(
            "The time indices of product and slope_pressure must match."
        )

    times = product.time_index.copy()
    data = product.data.copy()

    # Detect segments above threshold using shared helper
    onsets_above_threshold, filtered_onsets = _find_segments_above_threshold(
        data=data,
        time_index=times,
        threshold=threshold,
        max_gap=landmark_duration_max_interruption,
        min_duration=landmark_min_duration,
    )

    # If nothing survives filtering, early return
    if filtered_onsets.size == 0:
        return np.empty(0, dtype=int), onsets_above_threshold, filtered_onsets
    # Otherwise filter oscillations by suppressing immediate neighbours
    elif filtered_onsets.size == 1:
        # size 1 -> keep it
        pass
    else:
        # keep the first; drop neighbours closer than the given cooldown duration
        dt = np.diff(
            times.take(filtered_onsets)
        )  # timedeltas between consecutive kept onsets
        spacing_ok = np.r_[True, dt > landmark_cooldown_duration]
        filtered_onsets = filtered_onsets[spacing_ok]

    # === Identification of inspiratory reverse airflow ===
    # A segment of expiratory flow is considered to be caused by chest
    # compressions if it meets the following criteria:
    # 1. It is short in duration (less than or equal to 70 ms).
    # 2. The airway pressure is positive throughout the segment.
    # 3. The slope of the airway pressure is positive throughout the segment.

    # 1. find short segments of expiratory flow
    flow = interpolated_flow.data
    assert flow is not None
    (segments_exp_flow,) = np.where(flow <= 0)
    if segments_exp_flow.size == 0:
        exp_flow_starts = np.array([], dtype=int)
        exp_flow_ends = np.array([], dtype=int)
        duration_filter = np.array([], dtype=bool)
    else:
        (breaks,) = np.where(np.diff(segments_exp_flow) > 1)
        exp_flow_starts = np.r_[segments_exp_flow[0], segments_exp_flow[breaks + 1]]
        exp_flow_ends = np.r_[segments_exp_flow[breaks], segments_exp_flow[-1]]
        duration_filter = (
            times[exp_flow_ends] - times[exp_flow_starts]
        ) <= reverse_airflow_duration_limit

    short_exp_segments = np.flatnonzero(duration_filter)
    starts = exp_flow_starts[short_exp_segments]
    ends = exp_flow_ends[short_exp_segments]

    # 2. check that for the entire segment the pressure is positive
    pressure = interpolated_pressure.data
    assert pressure is not None
    positive_pressure = np.array(
        [np.all(pressure[s : e + 1] > 0) for s, e in zip(starts, ends)],
        dtype=bool,
    )
    segment_pressure_filter = np.zeros_like(duration_filter, dtype=bool)
    segment_pressure_filter[short_exp_segments] = positive_pressure

    # 3. assuring that for the entire segment the slope is positive
    slopes = slope_pressure.data
    assert slopes is not None
    positive_slope = np.array(
        [np.all(slopes[s : e + 1] > 0) for s, e in zip(starts, ends)],
        dtype=bool,
    )
    segment_slope_filter = np.zeros_like(duration_filter, dtype=bool)
    segment_slope_filter[short_exp_segments] = positive_slope

    # Identify segments of expiratory flow that are likely caused by chest
    # compressions rather than true expiration.
    # Transient reverse airflow can occur due to the mechanical force,
    # but the airway pressure remains positive and rising,
    # distinguishing these events from normal expiratory phases.
    # Filtering out these segments helps prevent misclassification of
    # chest compression artifacts as respiratory events.
    compression_exp_flow = (
        duration_filter & segment_pressure_filter & segment_slope_filter
    )

    # Find zero crossing of product before positive segments
    (non_pos_product,) = np.where(data <= 0)
    (breaks,) = np.where(np.diff(non_pos_product) > 1)
    zero_crossings = np.r_[non_pos_product[breaks], non_pos_product[-1]]
    if exp_flow_ends[compression_exp_flow].size > 0:
        # removes inspiratory reverse airflow
        mask = ~np.isin(zero_crossings, exp_flow_ends[compression_exp_flow])
    else:
        # keep all zero crossings if there is nothing to filter
        mask = np.ones_like(zero_crossings, dtype=bool)
    valid_zero_crossings = zero_crossings[mask]

    # Find the last zero before each start_above_idx in the zero_idxs without the short ones
    inspiration_starts = valid_zero_crossings[
        np.searchsorted(valid_zero_crossings, filtered_onsets, side="right") - 1
    ]

    return np.unique(inspiration_starts), onsets_above_threshold, filtered_onsets


def get_expiration_starts(
    product: DataSlice, threshold: float
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Derives the indices of the start of expirations from the product of flow and slope of pressure.

    This function identifies segments of the product that are above a certain threshold
    and finds the preceding zero crossing before each segment as potential expiration start.
    It returns the first candidate between two inspiration starts as the expiration start.

    Parameters
    ----------
    product:class:`.DataSlice`
        A DataSlice object containing the time index and data of the product of flow and slope of pressure.

    threshold: float
        The threshold value above which the expiration segments are detected.

    Returns
    -------
    expiration_starts : np.ndarray
        An array of integer indices for the time index of the given product where
        the expirations start (filtered zero crossings). Empty array (dtype int)
        if no valid expiration starts are detected.
    onsets_above_threshold : np.ndarray
        An array of integer indices where the product first exceeds the threshold
        for each detected segment. Empty array (dtype int) if no segments are detected.
    filtered_onsets : np.ndarray
        An array of integer indices for the onsets above threshold after duration
        filtering. Empty array (dtype int) if no onsets survive filtering.

    Notes
    -----
    All three returned arrays always have dtype int and maintain consistent shapes
    even when empty, allowing safe tuple unpacking and downstream processing.
    """
    # --- Filter parameters ---
    # Minimum gap between above-threshold segments to treat them as separate.
    # Gaps shorter than this are bridged (segments merged) to filter oscillations.
    min_gap_between_segments = np.timedelta64(10, "ms")

    # Minimum duration for a segment above threshold to be considered valid.
    min_segment_duration = np.timedelta64(12, "ms")

    # Maximum duration for a below-threshold (negative) segment during zero-crossing
    # detection. Shorter segments are filtered out as oscillation artifacts.
    max_oscillation_duration = np.timedelta64(10, "ms")

    # Threshold to suppress small oscillations around zero when detecting zero crossings.
    # Values at or below this threshold are treated as "non-positive" for zero-crossing
    # detection. Unit: product of flow and pressure slope.
    zero_crossing_tolerance = 0.05

    index = product.time_index.copy()
    data = product.data.copy()

    # Detect segments above threshold using shared helper
    onsets_above_threshold, filtered_onsets = _find_segments_above_threshold(
        data=data,
        time_index=index,
        threshold=threshold,
        max_gap=min_gap_between_segments,
        min_duration=min_segment_duration,
    )

    # If nothing survives filtering, early return
    if filtered_onsets.size == 0:
        return np.empty(0, dtype=int), onsets_above_threshold, filtered_onsets

    # Find zero crossings before the filtered onsets.
    # Values at or below zero_crossing_tolerance are treated as "non-positive"
    # to suppress small oscillations around zero.
    non_pos_product = np.flatnonzero(data <= zero_crossing_tolerance)
    if non_pos_product.size == 0:
        # No zero/non-positive values -> no valid zero crossing candidates
        return np.empty(0, dtype=int), onsets_above_threshold, filtered_onsets

    # Identify contiguous segments of non-positive values.
    breaks = np.flatnonzero(np.diff(non_pos_product) > 1)
    segment_starts = np.r_[non_pos_product[0], non_pos_product[breaks + 1]]
    segment_ends = np.r_[non_pos_product[breaks], non_pos_product[-1]]

    # Filter out short segments (likely oscillation artifacts) and keep the end
    # of each remaining segment as a zero-crossing candidate.
    segment_durations = index[segment_ends] - index[segment_starts]
    is_short_segment = segment_durations <= max_oscillation_duration
    if is_short_segment.any():
        keep_mask = ~np.isin(segment_ends, segment_ends[is_short_segment])
        zero_candidates = segment_ends[keep_mask]
    else:
        zero_candidates = segment_ends

    # If none remain, no zeros to reference
    if zero_candidates.size == 0:
        return np.empty(0, dtype=int), onsets_above_threshold, filtered_onsets

    # For each filtered onset, pick the last zero <= onset
    pos = np.searchsorted(zero_candidates, filtered_onsets, side="right") - 1
    valid = pos >= 0  # drop onsets with no preceding zero
    potential_exp_starts = zero_candidates[pos[valid]]

    return np.unique(potential_exp_starts), onsets_above_threshold, filtered_onsets
