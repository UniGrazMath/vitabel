"""Tests for the signal processing helpers in ``vitabel.utils.signal``."""

import numpy as np
import pandas as pd
import pytest

from vitabel import Channel
from vitabel.utils.signal import (
    _find_zero_crossings,
    _hjorth_params,
    _spectral_entropy_welch,
    _xlogx,
    av_mean,
    deriv,
    determine_gaps_in_recording,
    gaussian_kernel_regression_point,
    integrate,
    linear_interpolate_gaps_in_recording,
    resample_to_common_index,
)


class TestDeriv:
    def test_quadratic_equidistant(self):
        time = np.linspace(0, 2, 21)
        data = time**2
        result = deriv(time, data)
        # central differences are exact for quadratics in the interior
        np.testing.assert_allclose(result[1:-1], 2 * time[1:-1])
        # one-sided differences at the boundaries
        np.testing.assert_allclose(result[0], (data[1] - data[0]) / 0.1)
        np.testing.assert_allclose(result[-1], (data[-1] - data[-2]) / 0.1)

    def test_linear_non_equidistant(self):
        time = np.array([0.0, 0.5, 2.0, 2.2, 4.0])
        result = deriv(time, 3 * time + 1)
        np.testing.assert_allclose(result, 3.0)

    def test_alternating_steps(self):
        """Steps alternating such that ``time[2:] - time[:-2]`` is constant
        (used to be mistaken for equidistant: wrong boundary values)."""
        time = np.array([0.0, 0.5, 2.0, 2.5, 4.0])
        np.testing.assert_allclose(deriv(time, 3 * time + 1), 3.0)


class TestIntegrate:
    def test_linear_function(self):
        time = np.linspace(0, 1, 11)
        result = integrate(time, 2 * time)
        assert isinstance(result, pd.Series)
        # trapezoidal rule is exact for linear functions
        np.testing.assert_allclose(result.to_numpy(), time**2)

    def test_non_equidistant(self):
        np.testing.assert_allclose(integrate([0, 1, 3], [1, 1, 1]), [0, 1, 3])

    def test_length_mismatch(self):
        with pytest.raises(Exception, match="Length of time and data must agree"):
            integrate([0, 1, 2], [0, 1])


class TestAvMean:
    @pytest.mark.parametrize("k", [1, 3, 4, 5])
    def test_constant_signal_interior(self, k):
        result = av_mean(k, np.ones(20))
        assert result.shape == (20,)
        np.testing.assert_allclose(result[k // 2 : 20 - k // 2], 1.0)

    def test_zero_padding_at_boundaries(self):
        result = av_mean(3, np.ones(5))
        np.testing.assert_allclose(result, [2 / 3, 1, 1, 1, 2 / 3])

    def test_odd_window(self):
        result = av_mean(3, np.arange(6.0))
        np.testing.assert_allclose(result, [1 / 3, 1, 2, 3, 4, 9 / 3])


class TestGaps:
    def test_determine_gaps(self):
        time = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 10.0])
        data = np.arange(7.0)
        gap_start, gap_stop, gap_indices = determine_gaps_in_recording(time, data)
        np.testing.assert_array_equal(gap_start, [2.0, 7.0])
        np.testing.assert_array_equal(gap_stop, [5.0, 10.0])
        np.testing.assert_array_equal(gap_indices, [2, 5])

    def test_no_gaps(self):
        time = np.arange(10.0)
        gap_start, gap_stop, gap_indices = determine_gaps_in_recording(time, time)
        assert len(gap_start) == len(gap_stop) == len(gap_indices) == 0

    def test_linear_interpolation_of_gaps(self):
        time = np.array([0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 10.0])
        new_time, new_data = linear_interpolate_gaps_in_recording(time, 2 * time)
        np.testing.assert_array_equal(new_time, np.arange(11.0))
        np.testing.assert_array_equal(new_data, 2 * np.arange(11.0))

    def test_linear_interpolation_without_gaps(self):
        time = np.arange(5.0)
        new_time, new_data = linear_interpolate_gaps_in_recording(time, 2 * time)
        np.testing.assert_array_equal(new_time, time)
        np.testing.assert_array_equal(new_data, 2 * time)


class TestZeroCrossingsAndResampling:
    def test_find_zero_crossings(self):
        index = pd.to_datetime("2024-01-01") + pd.to_timedelta([0, 1, 2, 3], unit="s")
        series = pd.Series([-1.0, 1.0, 3.0, -1.0], index=index)
        crossings = _find_zero_crossings(series)
        assert list(crossings.index) == [
            index[0] + pd.Timedelta(seconds=0.5),
            index[2] + pd.Timedelta(seconds=0.75),
        ]
        assert (crossings == 0.0).all()

    def test_find_zero_crossings_none(self):
        index = pd.to_datetime("2024-01-01") + pd.to_timedelta([0, 1], unit="s")
        assert _find_zero_crossings(pd.Series([1.0, 2.0], index=index)).empty
        assert _find_zero_crossings(pd.Series([1.0], index=index[:1])).empty

    def test_resample_to_common_index(self):
        start = pd.Timestamp("2024-01-01 12:00:00")
        first = Channel(
            "a", time_index=[0.0, 1.0, 2.0, 3.0], data=[-1, 1, 1, 1.0], time_start=start
        )
        second = Channel(
            "b", time_index=[0.5, 1.5, 2.5, 3.5], data=[0, 2, 4, 6.0], time_start=start
        )
        a, b = resample_to_common_index(first, second)
        expected_index = start + pd.to_timedelta(
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], unit="s"
        )
        assert list(a.time_index) == list(expected_index)
        assert list(b.time_index) == list(expected_index)
        # the zero crossing of the first channel at 0.5s is inserted
        np.testing.assert_allclose(a.data, [0, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(b.data, [0, 1, 2, 3, 4, 5])

    def test_resample_without_overlap_raises(self):
        start = pd.Timestamp("2024-01-01 12:00:00")
        first = Channel("a", time_index=[0.0, 1.0], data=[1, 2.0], time_start=start)
        second = Channel("b", time_index=[2.0, 3.0], data=[1, 2.0], time_start=start)
        with pytest.raises(ValueError, match="No overlapping time range"):
            resample_to_common_index(first, second)


class TestKernelRegressionAndFeatures:
    def test_gaussian_kernel_regression_point(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        # symmetric weights around the center
        assert gaussian_kernel_regression_point(1.0, x, y) == pytest.approx(2.0)
        weights = np.exp(-np.square(x - 0.0))
        assert gaussian_kernel_regression_point(0.0, x, y) == pytest.approx(
            np.sum(weights * y) / np.sum(weights)
        )
        # too far away from all points
        assert np.isnan(gaussian_kernel_regression_point(10.0, x, y))

    def test_hjorth_params_of_sine(self):
        t = np.arange(10_000) / 1000
        mobility, complexity = _hjorth_params(np.sin(2 * np.pi * 5 * t))
        # for a pure sine, mobility ~ angular frequency per sample, complexity ~ 1
        assert mobility == pytest.approx(2 * np.pi * 5 / 1000, rel=1e-3)
        assert complexity == pytest.approx(1.0, rel=1e-3)

    def test_xlogx(self):
        result = _xlogx(np.array([-1.0, 0.0, 1.0, 2.0, 4.0]))
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:], [0.0, 0.0, 2.0, 8.0])

    def test_spectral_entropy(self):
        rng = np.random.default_rng(0)
        sine = np.sin(2 * np.pi * 10 * np.arange(5000) / 250)
        noise = rng.normal(size=5000)
        sine_entropy = _spectral_entropy_welch(sine, sf=250, normalize=True)
        noise_entropy = _spectral_entropy_welch(noise, sf=250, normalize=True)
        assert 0 <= sine_entropy < noise_entropy <= 1
        assert noise_entropy > 0.9
