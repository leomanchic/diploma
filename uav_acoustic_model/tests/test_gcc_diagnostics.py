"""Peak diagnostics, invalid-data handling, and direct GCC reference tests."""

import numpy as np
import pytest

from estimators.gcc_phat import direct_gcc_phat_correlation, gcc_phat
from simulation.fractional_delay import frequency_domain_delay
from simulation.signals import deterministic_bandlimited_signal

FS = 48_000.0


def _gcc(first, second, **kwargs):
    options = dict(
        maximum_delay_seconds=12.0 / FS,
        interpolation_factor=16,
        minimum_frequency_hz=300.0,
        maximum_frequency_hz=10_000.0,
    )
    options.update(kwargs)
    return gcc_phat(first, second, FS, **options)


@pytest.mark.parametrize(
    "signal",
    [np.zeros(1024), np.full(1024, 1e-14)],
)
def test_silence_and_nearly_zero_energy_return_invalid_not_arbitrary_delay(signal):
    result = _gcc(signal, signal)
    assert result.invalid
    assert result.invalid_reason == "signal_energy_below_threshold"
    assert np.isnan(result.delay_seconds)
    assert np.isnan(result.delay_samples)
    assert not result.boundary_hit
    assert result.used_spectral_energy == 0.0


@pytest.mark.parametrize("signal_kind", ["tone", "harmonics", "broadband"])
def test_peak_diagnostics_are_returned_for_supported_signal_classes(signal_kind):
    sample_count = 2048
    time = np.arange(sample_count) / FS
    if signal_kind == "tone":
        signal = np.sin(2.0 * np.pi * 2000.0 * time)
    elif signal_kind == "harmonics":
        signal = sum(
            np.sin(2.0 * np.pi * frequency * time + phase)
            for frequency, phase in [(800.0, 0.1), (1600.0, 0.7), (2400.0, 1.3)]
        )
    else:
        signal = deterministic_bandlimited_signal(
            FS,
            sample_count / FS,
            minimum_frequency_hz=300.0,
            maximum_frequency_hz=10_000.0,
            tone_count=31,
        )
    delayed = frequency_domain_delay(signal, 3.7, output_length=sample_count + 6)
    reference = frequency_domain_delay(signal, 0.2, output_length=sample_count + 6)
    result = _gcc(delayed, reference)
    assert not result.invalid
    assert result.invalid_reason is None
    tolerance = 3e-3 if signal_kind == "broadband" else 0.08
    assert result.delay_samples == pytest.approx(3.5, abs=tolerance)
    assert np.isfinite(result.peak_value)
    assert result.peak_to_second_peak_ratio > 0.0
    assert np.isfinite(result.peak_curvature)
    assert result.peak_curvature > 0.0
    assert result.used_spectral_energy > 0.0
    assert 0.0 < result.spectral_energy_fraction <= 1.0 + 1e-12
    assert result.used_bin_count > 0


def test_empty_frequency_band_returns_invalid_and_malformed_band_raises():
    signal = deterministic_bandlimited_signal(FS, 0.02)
    empty = _gcc(
        signal,
        signal,
        minimum_frequency_hz=101.0,
        maximum_frequency_hz=102.0,
    )
    assert empty.invalid
    assert empty.invalid_reason == "empty_frequency_band"
    assert np.isnan(empty.delay_seconds)
    with pytest.raises(ValueError, match="frequency band"):
        _gcc(
            signal,
            signal,
            minimum_frequency_hz=10_000.0,
            maximum_frequency_hz=300.0,
        )


def test_direct_reference_matches_fft_sign_position_and_correlation_shape():
    signal = deterministic_bandlimited_signal(
        FS,
        0.04,
        minimum_frequency_hz=300.0,
        maximum_frequency_hz=10_000.0,
        tone_count=31,
    )
    output_length = signal.size + 8
    first = frequency_domain_delay(signal, 5.6, output_length=output_length)
    second = frequency_domain_delay(signal, 0.2, output_length=output_length)
    fft_result = _gcc(first, second)
    tau_grid = fft_result.lags_samples / FS
    direct = direct_gcc_phat_correlation(
        first,
        second,
        FS,
        tau_grid,
        minimum_frequency_hz=300.0,
        maximum_frequency_hz=10_000.0,
    )
    assert not direct.invalid
    direct_peak_samples = tau_grid[int(np.argmax(direct.correlation))] * FS
    fft_grid_peak_samples = fft_result.lags_samples[int(np.argmax(fft_result.correlation))]
    assert direct_peak_samples == pytest.approx(5.375, abs=1.0 / 16.0)
    assert direct_peak_samples == pytest.approx(fft_grid_peak_samples, abs=1e-12)
    direct_normalized = direct.correlation / np.max(direct.correlation)
    fft_normalized = fft_result.correlation / np.max(fft_result.correlation)
    np.testing.assert_allclose(direct_normalized, fft_normalized, atol=2e-12, rtol=2e-12)


def test_direct_reference_preserves_invalid_silence_state():
    result = direct_gcc_phat_correlation(
        np.zeros(128),
        np.zeros(128),
        FS,
        np.linspace(-1e-4, 1e-4, 11),
        minimum_frequency_hz=300.0,
        maximum_frequency_hz=10_000.0,
    )
    assert result.invalid
    assert result.invalid_reason == "signal_energy_below_threshold"
    assert np.all(result.correlation == 0.0)
