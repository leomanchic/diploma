"""Independent fractional-delay implementation tests."""

import numpy as np
import pytest
from scipy.signal import firwin

from simulation.fractional_delay import (
    DEFAULT_FIR_LENGTH,
    fractional_delay_valid_region,
    frequency_domain_delay,
    windowed_sinc_delay,
)


METHODS = (frequency_domain_delay, windowed_sinc_delay)
SUBSAMPLE_DELAYS = (0.1, 0.25, 0.5, 0.75, 0.9)
WORKING_FREQUENCIES = (0.03, 0.10, 0.20, 0.30, 0.40)


def _bandlimited_pulse(length=4096):
    signal = np.zeros(length)
    pulse = firwin(513, 0.70, window=("kaiser", 8.6))
    start = (length - pulse.size) // 2
    signal[start : start + pulse.size] = pulse
    return signal


def _energy_centroid(signal):
    energy = np.asarray(signal) ** 2
    return float(np.dot(np.arange(energy.size), energy) / np.sum(energy))


def _sinusoid_gain(signal, normalized_frequency, region):
    indices = np.arange(signal.size)[region]
    basis = np.column_stack(
        (
            np.cos(2.0 * np.pi * normalized_frequency * indices),
            np.sin(2.0 * np.pi * normalized_frequency * indices),
        )
    )
    coefficients = np.linalg.lstsq(basis, signal[region], rcond=None)[0]
    return coefficients[0] - 1j * coefficients[1]


@pytest.mark.parametrize("method", METHODS)
def test_zero_delay_returns_the_original_signal(method):
    signal = np.linspace(-1.0, 1.0, 257) ** 3
    np.testing.assert_array_equal(method(signal, 0.0), signal)


@pytest.mark.parametrize("method", METHODS)
def test_integer_delay_matches_a_zero_filled_shift(method):
    signal = np.sin(0.13 * np.arange(300))
    expected = np.zeros_like(signal)
    expected[7:] = signal[:-7]
    np.testing.assert_array_equal(method(signal, 7.0), expected)


@pytest.mark.parametrize("method", METHODS)
def test_positive_delay_moves_signal_to_the_right(method):
    signal = _bandlimited_pulse()
    delayed = method(signal, 3.75)
    assert abs((_energy_centroid(delayed) - _energy_centroid(signal)) - 3.75) < 3e-5


@pytest.mark.parametrize("method", METHODS)
def test_zero_padding_prevents_end_to_start_circular_wrap(method):
    signal = np.zeros(2048)
    signal[-100:-35] = np.hanning(65)
    delayed = method(signal, 20.5, output_length=2069)
    early_energy_fraction = np.sum(delayed[:256] ** 2) / np.sum(delayed**2)
    assert early_energy_fraction < 1e-18


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("delay", SUBSAMPLE_DELAYS)
def test_subsample_phase_and_amplitude_across_working_band(method, delay):
    length = 8192
    indices = np.arange(length)
    region = slice(512, length - 512)
    for frequency in WORKING_FREQUENCIES:
        signal = np.cos(2.0 * np.pi * frequency * indices)
        delayed = method(signal, delay)
        gain = _sinusoid_gain(delayed, frequency, region)
        expected_phase = -2.0 * np.pi * frequency * delay
        phase_error = abs(np.angle(gain * np.exp(-1j * expected_phase)))
        amplitude_error = abs(abs(gain) - 1.0)
        assert phase_error < 1e-5
        assert amplitude_error < 1e-5


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("delay", SUBSAMPLE_DELAYS)
def test_broadband_energy_group_delay(method, delay):
    signal = _bandlimited_pulse()
    delayed = method(signal, delay)
    measured_delay = _energy_centroid(delayed) - _energy_centroid(signal)
    assert abs(measured_delay - delay) < 5e-5


@pytest.mark.parametrize("delay", SUBSAMPLE_DELAYS)
def test_independent_implementations_agree_in_valid_region(delay):
    signal = _bandlimited_pulse()
    frequency_result = frequency_domain_delay(signal, delay)
    sinc_result = windowed_sinc_delay(signal, delay)
    start, stop = fractional_delay_valid_region(signal.size, delay)
    assert stop > start
    assert np.max(np.abs(frequency_result[start:stop] - sinc_result[start:stop])) < 4e-6


def test_valid_region_excludes_fir_boundaries_and_reports_intrinsic_guard():
    assert DEFAULT_FIR_LENGTH == 129
    assert fractional_delay_valid_region(1000, [0.0, 3.75]) == (68, 936)
