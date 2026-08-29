"""GCC-PHAT for oriented TDOA pairs with explicit peak diagnostics.

For pair ``(i, j)`` the cross spectrum is ``X_i * conj(X_j)`` and the
reported sign is ``tau_ij = T_i - T_j``. A positive estimate therefore means
that channel ``i`` arrives later than channel ``j``.

Configuration errors raise ``ValueError``. Data-quality failures (silence,
negligible energy, or no usable spectral bins) return a result with
``invalid=True`` and ``delay_seconds=delay_samples=NaN``; they never return an
arbitrary lag. The FFT implementation remains the production estimator and
``direct_gcc_phat_correlation`` is a deliberately slow independent reference.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import next_fast_len

from model.geometry import Pair, validate_pairs


@dataclass(frozen=True)
class GCCPHATResult:
    """One oriented GCC-PHAT delay estimate and confidence diagnostics."""

    delay_seconds: float
    delay_samples: float
    peak_value: float
    peak_to_second_peak_ratio: float
    peak_curvature: float
    boundary_hit: bool
    invalid: bool
    invalid_reason: str | None
    used_spectral_energy: float
    spectral_energy_fraction: float
    used_bin_count: int
    lags_samples: NDArray[np.float64]
    correlation: NDArray[np.float64]
    interpolation_factor: int
    maximum_delay_seconds: float

    @property
    def peak_correlation(self) -> float:
        """Backward-compatible alias for ``peak_value``."""

        return self.peak_value


@dataclass(frozen=True)
class DirectGCCPHATResult:
    """Slow direct evaluation of the PHAT correlation on arbitrary lags."""

    tau_seconds: NDArray[np.float64]
    correlation: NDArray[np.float64]
    invalid: bool
    invalid_reason: str | None
    used_spectral_energy: float
    spectral_energy_fraction: float
    used_bin_count: int


@dataclass(frozen=True)
class _SpectrumData:
    phat_spectrum: NDArray[np.complex128]
    frequencies_hz: NDArray[np.float64]
    transform_length: int
    used_spectral_energy: float
    spectral_energy_fraction: float
    used_bin_count: int
    invalid_reason: str | None


def _finite_signal(signal: ArrayLike, name: str) -> NDArray[np.float64]:
    samples = np.asarray(signal, dtype=float)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least 2 samples")
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} must contain only finite values")
    return samples


def _positive_sampling_rate(value: float) -> float:
    sampling_rate = float(value)
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    return sampling_rate


def _frequency_band(
    sampling_rate: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float | None,
) -> tuple[float, float]:
    low = float(minimum_frequency_hz)
    high = sampling_rate / 2.0 if maximum_frequency_hz is None else float(maximum_frequency_hz)
    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or not 0.0 <= low < high <= sampling_rate / 2.0
    ):
        raise ValueError("frequency band must satisfy 0 <= minimum < maximum <= Nyquist")
    return low, high


def _prepare_phat_spectrum(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    sampling_rate: float,
    *,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float | None,
    relative_spectral_floor: float,
    minimum_signal_rms: float,
    minimum_spectral_energy_fraction: float,
) -> _SpectrumData:
    low, high = _frequency_band(
        sampling_rate, minimum_frequency_hz, maximum_frequency_hz
    )
    floor = float(relative_spectral_floor)
    if not np.isfinite(floor) or not 0.0 <= floor < 1.0:
        raise ValueError("relative_spectral_floor must lie in [0, 1)")
    rms_threshold = float(minimum_signal_rms)
    if not np.isfinite(rms_threshold) or rms_threshold < 0.0:
        raise ValueError("minimum_signal_rms must be finite and non-negative")
    energy_threshold = float(minimum_spectral_energy_fraction)
    if not np.isfinite(energy_threshold) or not 0.0 <= energy_threshold < 1.0:
        raise ValueError("minimum_spectral_energy_fraction must lie in [0, 1)")

    transform_length = next_fast_len(first.size + second.size)
    frequencies = np.fft.rfftfreq(transform_length, d=1.0 / sampling_rate)
    empty = np.zeros(frequencies.size, dtype=complex)
    first_rms = float(np.sqrt(np.mean(first**2)))
    second_rms = float(np.sqrt(np.mean(second**2)))
    if first_rms <= rms_threshold or second_rms <= rms_threshold:
        return _SpectrumData(
            empty,
            frequencies,
            transform_length,
            0.0,
            0.0,
            0,
            "signal_energy_below_threshold",
        )

    first_spectrum = np.fft.rfft(first, n=transform_length)
    second_spectrum = np.fft.rfft(second, n=transform_length)
    cross_spectrum = first_spectrum * np.conj(second_spectrum)
    magnitude = np.abs(cross_spectrum)
    band = (frequencies >= low) & (frequencies <= high)
    if not np.any(band):
        return _SpectrumData(
            empty, frequencies, transform_length, 0.0, 0.0, 0, "empty_frequency_band"
        )
    peak_magnitude = float(np.max(magnitude[band]))
    active = band & (magnitude > floor * peak_magnitude)
    if peak_magnitude == 0.0 or not np.any(active):
        return _SpectrumData(
            empty,
            frequencies,
            transform_length,
            0.0,
            0.0,
            0,
            "spectral_energy_below_threshold",
        )

    used_energy = float(np.sum(magnitude[active]))
    first_energy = float(np.sum(np.abs(first_spectrum[band]) ** 2))
    second_energy = float(np.sum(np.abs(second_spectrum[band]) ** 2))
    normalizer = np.sqrt(first_energy * second_energy)
    energy_fraction = used_energy / normalizer if normalizer > 0.0 else 0.0
    if not np.isfinite(energy_fraction) or energy_fraction <= energy_threshold:
        return _SpectrumData(
            empty,
            frequencies,
            transform_length,
            used_energy,
            float(energy_fraction) if np.isfinite(energy_fraction) else 0.0,
            int(np.count_nonzero(active)),
            "spectral_energy_below_threshold",
        )
    phat_spectrum = np.zeros_like(cross_spectrum)
    phat_spectrum[active] = cross_spectrum[active] / magnitude[active]
    return _SpectrumData(
        phat_spectrum,
        frequencies,
        transform_length,
        used_energy,
        float(energy_fraction),
        int(np.count_nonzero(active)),
        None,
    )


def _search_lags(
    maximum_delay: float,
    sampling_rate: float,
    interpolation: int,
    transform_length: int,
) -> tuple[int, NDArray[np.float64]]:
    maximum_shift = int(np.ceil(maximum_delay * sampling_rate * interpolation))
    maximum_supported = transform_length * interpolation // 2 - 1
    if maximum_shift > maximum_supported:
        raise ValueError("maximum_delay_seconds exceeds the supported non-aliased lag interval")
    lags = np.arange(-maximum_shift, maximum_shift + 1, dtype=float) / interpolation
    return maximum_shift, lags


def _invalid_result(
    reason: str,
    *,
    lags_samples: NDArray[np.float64],
    interpolation: int,
    maximum_delay: float,
    used_spectral_energy: float,
    spectral_energy_fraction: float,
    used_bin_count: int,
) -> GCCPHATResult:
    return GCCPHATResult(
        delay_seconds=float("nan"),
        delay_samples=float("nan"),
        peak_value=float("nan"),
        peak_to_second_peak_ratio=float("nan"),
        peak_curvature=float("nan"),
        boundary_hit=False,
        invalid=True,
        invalid_reason=reason,
        used_spectral_energy=used_spectral_energy,
        spectral_energy_fraction=spectral_energy_fraction,
        used_bin_count=used_bin_count,
        lags_samples=lags_samples,
        correlation=np.zeros(lags_samples.size, dtype=float),
        interpolation_factor=interpolation,
        maximum_delay_seconds=maximum_delay,
    )


def _second_peak_ratio(
    correlation: NDArray[np.float64], peak_index: int, exclusion_radius: int
) -> float:
    mask = np.ones(correlation.size, dtype=bool)
    start = max(0, peak_index - exclusion_radius)
    stop = min(correlation.size, peak_index + exclusion_radius + 1)
    mask[start:stop] = False
    if not np.any(mask):
        return float("inf")
    second = float(np.max(correlation[mask]))
    peak = float(correlation[peak_index])
    if second <= 0.0:
        return float("inf")
    return peak / second


def gcc_phat(
    signal_i: ArrayLike,
    signal_j: ArrayLike,
    sampling_rate_hz: float,
    *,
    maximum_delay_seconds: float,
    interpolation_factor: int = 32,
    minimum_frequency_hz: float = 0.0,
    maximum_frequency_hz: float | None = None,
    relative_spectral_floor: float = 1e-10,
    minimum_signal_rms: float = 1e-10,
    minimum_spectral_energy_fraction: float = 1e-12,
) -> GCCPHATResult:
    """Estimate ``T_i-T_j`` and return peak/confidence diagnostics.

    ``boundary_hit`` is independent from ``invalid``. A boundary peak can be
    physically valid when the configured search bound is exact; callers may
    reject it using calibration-derived thresholds.
    """

    first = _finite_signal(signal_i, "signal_i")
    second = _finite_signal(signal_j, "signal_j")
    sampling_rate = _positive_sampling_rate(sampling_rate_hz)
    maximum_delay = float(maximum_delay_seconds)
    if not np.isfinite(maximum_delay) or maximum_delay <= 0.0:
        raise ValueError("maximum_delay_seconds must be finite and positive")
    interpolation = int(interpolation_factor)
    if interpolation < 1 or interpolation != interpolation_factor:
        raise ValueError("interpolation_factor must be a positive integer")
    spectrum = _prepare_phat_spectrum(
        first,
        second,
        sampling_rate,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        relative_spectral_floor=relative_spectral_floor,
        minimum_signal_rms=minimum_signal_rms,
        minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
    )
    maximum_shift, lags_samples = _search_lags(
        maximum_delay, sampling_rate, interpolation, spectrum.transform_length
    )
    if spectrum.invalid_reason is not None:
        return _invalid_result(
            spectrum.invalid_reason,
            lags_samples=lags_samples,
            interpolation=interpolation,
            maximum_delay=maximum_delay,
            used_spectral_energy=spectrum.used_spectral_energy,
            spectral_energy_fraction=spectrum.spectral_energy_fraction,
            used_bin_count=spectrum.used_bin_count,
        )

    oversampled_length = spectrum.transform_length * interpolation
    circular_correlation = np.fft.irfft(
        spectrum.phat_spectrum, n=oversampled_length
    )
    correlation = np.concatenate(
        (circular_correlation[-maximum_shift:], circular_correlation[: maximum_shift + 1])
    )
    if not np.all(np.isfinite(correlation)):
        return _invalid_result(
            "nonfinite_correlation",
            lags_samples=lags_samples,
            interpolation=interpolation,
            maximum_delay=maximum_delay,
            used_spectral_energy=spectrum.used_spectral_energy,
            spectral_energy_fraction=spectrum.spectral_energy_fraction,
            used_bin_count=spectrum.used_bin_count,
        )
    integer_peak = int(np.argmax(correlation))
    boundary_hit = integer_peak in {0, correlation.size - 1}
    fractional_peak = 0.0
    curvature = float("nan")
    if not boundary_hit:
        left, centre, right = correlation[integer_peak - 1 : integer_peak + 2]
        curvature = float(centre - 0.5 * (left + right))
        denominator = left - 2.0 * centre + right
        if abs(denominator) > np.finfo(float).eps * max(1.0, abs(centre)):
            fractional_peak = float(
                np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)
            )
    delay_samples = (integer_peak - maximum_shift + fractional_peak) / interpolation
    ratio = _second_peak_ratio(correlation, integer_peak, interpolation)
    return GCCPHATResult(
        delay_seconds=float(delay_samples / sampling_rate),
        delay_samples=float(delay_samples),
        peak_value=float(correlation[integer_peak]),
        peak_to_second_peak_ratio=float(ratio),
        peak_curvature=curvature,
        boundary_hit=boundary_hit,
        invalid=False,
        invalid_reason=None,
        used_spectral_energy=spectrum.used_spectral_energy,
        spectral_energy_fraction=spectrum.spectral_energy_fraction,
        used_bin_count=spectrum.used_bin_count,
        lags_samples=lags_samples,
        correlation=correlation.copy(),
        interpolation_factor=interpolation,
        maximum_delay_seconds=maximum_delay,
    )


def direct_gcc_phat_correlation(
    signal_i: ArrayLike,
    signal_j: ArrayLike,
    sampling_rate_hz: float,
    tau_seconds: ArrayLike,
    *,
    minimum_frequency_hz: float = 0.0,
    maximum_frequency_hz: float | None = None,
    relative_spectral_floor: float = 1e-10,
    minimum_signal_rms: float = 1e-10,
    minimum_spectral_energy_fraction: float = 1e-12,
) -> DirectGCCPHATResult:
    r"""Evaluate ``sum_k Psi[k] exp(j*2*pi*f_k*tau)`` directly.

    The one-sided real spectrum is expanded analytically with weight two for
    conjugate positive-frequency bins and one for DC/Nyquist. The result is
    divided by the base transform length; peak position and normalized shape
    are comparable with the FFT implementation.
    """

    first = _finite_signal(signal_i, "signal_i")
    second = _finite_signal(signal_j, "signal_j")
    sampling_rate = _positive_sampling_rate(sampling_rate_hz)
    grid = np.asarray(tau_seconds, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("tau_seconds must be a non-empty finite one-dimensional grid")
    spectrum = _prepare_phat_spectrum(
        first,
        second,
        sampling_rate,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        relative_spectral_floor=relative_spectral_floor,
        minimum_signal_rms=minimum_signal_rms,
        minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
    )
    if spectrum.invalid_reason is not None:
        return DirectGCCPHATResult(
            tau_seconds=grid.copy(),
            correlation=np.zeros(grid.size),
            invalid=True,
            invalid_reason=spectrum.invalid_reason,
            used_spectral_energy=spectrum.used_spectral_energy,
            spectral_energy_fraction=spectrum.spectral_energy_fraction,
            used_bin_count=spectrum.used_bin_count,
        )
    weights = np.full(spectrum.frequencies_hz.size, 2.0)
    weights[0] = 1.0
    if spectrum.transform_length % 2 == 0:
        weights[-1] = 1.0
    exponent = np.exp(
        2j * np.pi * spectrum.frequencies_hz[:, None] * grid[None, :]
    )
    correlation = np.real(
        np.sum(
            weights[:, None] * spectrum.phat_spectrum[:, None] * exponent,
            axis=0,
        )
    ) / spectrum.transform_length
    return DirectGCCPHATResult(
        tau_seconds=grid.copy(),
        correlation=np.asarray(correlation, dtype=float),
        invalid=False,
        invalid_reason=None,
        used_spectral_energy=spectrum.used_spectral_energy,
        spectral_energy_fraction=spectrum.spectral_energy_fraction,
        used_bin_count=spectrum.used_bin_count,
    )


def estimate_tdoas_gcc_phat(
    channels: ArrayLike,
    sampling_rate_hz: float,
    pairs: Iterable[Sequence[int]],
    *,
    maximum_delay_seconds: float | ArrayLike,
    interpolation_factor: int = 32,
    minimum_frequency_hz: float = 0.0,
    maximum_frequency_hz: float | None = None,
    relative_spectral_floor: float = 1e-10,
    minimum_signal_rms: float = 1e-10,
    minimum_spectral_energy_fraction: float = 1e-12,
) -> tuple[NDArray[np.float64], tuple[GCCPHATResult, ...]]:
    """Estimate all oriented pairs, preserving invalid results as ``NaN``."""

    matrix = np.asarray(channels, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("channels must have shape (channel_count, sample_count)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("channels must contain only finite values")
    checked_pairs: tuple[Pair, ...] = validate_pairs(pairs, matrix.shape[0])
    bounds = np.asarray(maximum_delay_seconds, dtype=float)
    if bounds.ndim == 0:
        bounds = np.full(len(checked_pairs), float(bounds))
    if bounds.shape != (len(checked_pairs),):
        raise ValueError("maximum_delay_seconds must be scalar or have one value per pair")
    results = tuple(
        gcc_phat(
            matrix[first],
            matrix[second],
            sampling_rate_hz,
            maximum_delay_seconds=float(bound),
            interpolation_factor=interpolation_factor,
            minimum_frequency_hz=minimum_frequency_hz,
            maximum_frequency_hz=maximum_frequency_hz,
            relative_spectral_floor=relative_spectral_floor,
            minimum_signal_rms=minimum_signal_rms,
            minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
        )
        for (first, second), bound in zip(checked_pairs, bounds, strict=True)
    )
    estimates = np.asarray([result.delay_seconds for result in results])
    return estimates, results


__all__ = [
    "DirectGCCPHATResult",
    "GCCPHATResult",
    "direct_gcc_phat_correlation",
    "estimate_tdoas_gcc_phat",
    "gcc_phat",
]
