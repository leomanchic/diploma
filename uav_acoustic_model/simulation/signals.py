"""Deterministic, non-UAV-specific validation signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import tukey


def deterministic_bandlimited_signal(
    sampling_rate_hz: float,
    duration_s: float = 0.25,
    *,
    minimum_frequency_hz: float = 200.0,
    maximum_frequency_hz: float = 12_000.0,
    tone_count: int = 47,
    taper_fraction: float = 0.12,
    phase_offset_rad: float = 0.0,
) -> NDArray[np.float64]:
    """Return a reproducible tapered broadband multisine.

    Frequencies lie strictly inside the configured working band and phases
    follow a deterministic irrational sequence. This is a numerical test
    waveform, not a UAV harmonic or stochastic-noise model.
    """

    sampling_rate = float(sampling_rate_hz)
    duration = float(duration_s)
    low = float(minimum_frequency_hz)
    high = float(maximum_frequency_hz)
    count = int(tone_count)
    taper = float(taper_fraction)
    phase_offset = float(phase_offset_rad)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not (0.0 < low < high < sampling_rate / 2.0):
        raise ValueError("frequencies must satisfy 0 < minimum < maximum < Nyquist")
    if count < 2:
        raise ValueError("tone_count must be at least 2")
    if not np.isfinite(taper) or not 0.0 <= taper <= 1.0:
        raise ValueError("taper_fraction must lie in [0, 1]")
    if not np.isfinite(phase_offset):
        raise ValueError("phase_offset_rad must be finite")

    sample_count = max(2, int(np.rint(duration * sampling_rate)))
    time = np.arange(sample_count, dtype=float) / sampling_rate
    frequencies = np.linspace(low, high, count + 2, dtype=float)[1:-1]
    golden_fraction = (np.sqrt(5.0) - 1.0) / 2.0
    phases = 2.0 * np.pi * np.mod(np.arange(count) * golden_fraction, 1.0) + phase_offset
    amplitudes = 1.0 / np.sqrt(frequencies / low)
    signal = np.sum(
        amplitudes[:, None]
        * np.cos(2.0 * np.pi * frequencies[:, None] * time + phases[:, None]),
        axis=0,
    )
    signal *= tukey(sample_count, alpha=taper)
    peak = float(np.max(np.abs(signal)))
    if peak == 0.0:
        raise RuntimeError("constructed validation signal is identically zero")
    return signal / peak


def random_bandlimited_signal(
    sampling_rate_hz: float,
    sample_count: int,
    rng: np.random.Generator,
    *,
    minimum_frequency_hz: float = 300.0,
    maximum_frequency_hz: float = 10_000.0,
    taper_fraction: float = 0.08,
) -> NDArray[np.float64]:
    """Return one random real band-limited broadband realization.

    Independent complex Gaussian coefficients are drawn for every active FFT
    bin. The caller owns ``rng`` so calibration and evaluation streams remain
    explicitly reproducible and independent. The returned waveform has unit
    RMS after a Tukey taper.
    """

    sampling_rate = float(sampling_rate_hz)
    count = int(sample_count)
    low = float(minimum_frequency_hz)
    high = float(maximum_frequency_hz)
    taper = float(taper_fraction)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if count < 16:
        raise ValueError("sample_count must be at least 16")
    if not (0.0 < low < high < sampling_rate / 2.0):
        raise ValueError("frequencies must satisfy 0 < minimum < maximum < Nyquist")
    if not 0.0 <= taper <= 1.0:
        raise ValueError("taper_fraction must lie in [0, 1]")
    frequencies = np.fft.rfftfreq(count, d=1.0 / sampling_rate)
    active = (frequencies >= low) & (frequencies <= high)
    spectrum = np.zeros(frequencies.size, dtype=complex)
    spectrum[active] = rng.normal(size=np.count_nonzero(active)) + 1j * rng.normal(
        size=np.count_nonzero(active)
    )
    spectrum[0] = 0.0
    if count % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    signal = np.fft.irfft(spectrum, n=count)
    signal *= tukey(count, alpha=taper)
    rms = float(np.sqrt(np.mean(signal**2)))
    if rms == 0.0:
        raise RuntimeError("constructed random broadband signal is identically zero")
    return signal / rms


def harmonic_stress_signal(
    sampling_rate_hz: float,
    sample_count: int,
    *,
    fundamental_frequency_hz: float = 240.0,
    maximum_frequency_hz: float = 10_000.0,
    harmonic_count: int = 32,
    rng: np.random.Generator | None = None,
    taper_fraction: float = 0.08,
) -> NDArray[np.float64]:
    """Return a harmonic ambiguity stress-test, not a realistic UAV model.

    Harmonic amplitudes decay as ``h**-0.7``. Optional random phases provide
    independent realizations while preserving the deliberately sparse line
    spectrum that can create multiple GCC peaks.
    """

    sampling_rate = float(sampling_rate_hz)
    count = int(sample_count)
    fundamental = float(fundamental_frequency_hz)
    high = float(maximum_frequency_hz)
    harmonics_requested = int(harmonic_count)
    taper = float(taper_fraction)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    if count < 16:
        raise ValueError("sample_count must be at least 16")
    if not 0.0 < fundamental < high < sampling_rate / 2.0:
        raise ValueError("harmonic frequencies must lie below Nyquist")
    if harmonics_requested < 1:
        raise ValueError("harmonic_count must be positive")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator or None")
    if not 0.0 <= taper <= 1.0:
        raise ValueError("taper_fraction must lie in [0, 1]")
    harmonic_indices = np.arange(1, harmonics_requested + 1, dtype=float)
    frequencies = fundamental * harmonic_indices
    retained = frequencies <= high
    harmonic_indices = harmonic_indices[retained]
    frequencies = frequencies[retained]
    if frequencies.size == 0:
        raise ValueError("no harmonic lies in the requested band")
    if rng is None:
        phases = 2.0 * np.pi * np.mod(
            harmonic_indices * (np.sqrt(5.0) - 1.0) / 2.0, 1.0
        )
    else:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=frequencies.size)
    amplitudes = harmonic_indices**-0.7
    time = np.arange(count, dtype=float) / sampling_rate
    signal = np.sum(
        amplitudes[:, None]
        * np.sin(2.0 * np.pi * frequencies[:, None] * time + phases[:, None]),
        axis=0,
    )
    signal *= tukey(count, alpha=taper)
    rms = float(np.sqrt(np.mean(signal**2)))
    if rms == 0.0:
        raise RuntimeError("constructed harmonic signal is identically zero")
    return signal / rms
