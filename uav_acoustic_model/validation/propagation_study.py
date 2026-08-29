"""Detailed deterministic far-field and fractional-delay study orchestration."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.signal import firwin

from model.geometry import comparison_arrays
from simulation.fractional_delay import (
    DEFAULT_FIR_LENGTH,
    frequency_domain_delay,
    fractional_delay_valid_region,
    windowed_sinc_delay,
)
from validation.far_field import (
    continuous_refine_far_field_error,
    direction_grid,
    far_field_error,
    minimum_far_field_distance,
    phase_error,
    sample_error,
)

DIAGNOSTIC_SAMPLING_RATE_HZ = 48_000.0
DIAGNOSTIC_MAXIMUM_FREQUENCIES_HZ = (2_000.0, 4_000.0, 8_000.0, 12_000.0)
DIAGNOSTIC_SAMPLE_ERROR_LIMIT = 0.1
DIAGNOSTIC_PHASE_ERROR_LIMIT_RAD = 0.1
DISTANCES_M = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
FINAL_ANGULAR_STEP_DEG = 0.5
REFINED_ANGULAR_STEP_DEG = 0.25
FRACTIONAL_DELAYS_SAMPLES = (0.1, 0.25, 0.5, 0.75, 0.9)
FRACTIONAL_TEST_FREQUENCIES_HZ = (1_500.0, 2_000.0, 4_000.0, 8_000.0, 12_000.0, 16_000.0, 19_000.0)
FRACTIONAL_VALIDATION_GUARD_SAMPLES = 512


def _write_records(records: list[dict[str, object]], output_path: str | Path) -> Path:
    if not records:
        raise ValueError("at least one record is required")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def run_far_field_boundary_study(
    *,
    output_csv: str | Path = "results/far_field_boundary.csv",
    angular_step_deg: float = FINAL_ANGULAR_STEP_DEG,
    refined_angular_step_deg: float | None = REFINED_ANGULAR_STEP_DEG,
    sampling_rate_hz: float = DIAGNOSTIC_SAMPLING_RATE_HZ,
    maximum_frequencies_hz: tuple[float, ...] = DIAGNOSTIC_MAXIMUM_FREQUENCIES_HZ,
    sample_error_limit: float = DIAGNOSTIC_SAMPLE_ERROR_LIMIT,
    phase_error_limit_rad: float = DIAGNOSTIC_PHASE_ERROR_LIMIT_RAD,
) -> list[dict[str, object]]:
    """Run distance sweeps and numerical boundary searches for all arrays."""

    refined_step = (
        angular_step_deg / 2.0
        if refined_angular_step_deg is None
        else float(refined_angular_step_deg)
    )
    grid = direction_grid(angular_step_deg, angular_step_deg)
    refined_grid = direction_grid(refined_step, refined_step)
    records: list[dict[str, object]] = []
    for geometry_name, positions in comparison_arrays().items():
        for distance in DISTANCES_M:
            result = far_field_error(positions, distance, grid=grid)
            record: dict[str, object] = {
                "record_type": "distance_sweep",
                "geometry": geometry_name,
                "distance_m": distance,
                "max_plane_error_s": result.max_plane_error_s,
                "max_plane_error_us": result.max_plane_error_s * 1e6,
                "max_second_order_error_s": result.max_second_order_error_s,
                "max_second_order_error_us": result.max_second_order_error_s * 1e6,
                "sample_error": float(sample_error(result.max_plane_error_s, sampling_rate_hz)),
                "worst_azimuth_deg": result.worst_plane_azimuth_deg,
                "worst_elevation_deg": result.worst_plane_elevation_deg,
                "worst_pair": f"{result.worst_plane_pair[0]}-{result.worst_plane_pair[1]}",
                "second_order_worst_azimuth_deg": result.worst_second_order_azimuth_deg,
                "second_order_worst_elevation_deg": result.worst_second_order_elevation_deg,
                "second_order_worst_pair": (
                    f"{result.worst_second_order_pair[0]}-{result.worst_second_order_pair[1]}"
                ),
                "sampling_rate_hz": sampling_rate_hz,
                "angular_step_deg": angular_step_deg,
                "max_elevation_deg": 80.0,
            }
            for frequency in maximum_frequencies_hz:
                record[f"phase_error_{int(frequency)}hz_rad"] = float(
                    phase_error(result.max_plane_error_s, frequency)
                )
            records.append(record)

        criteria = [
            (
                "sample",
                None,
                sample_error_limit / sampling_rate_hz,
                sample_error_limit,
                "samples",
            )
        ]
        criteria.extend(
            (
                "phase",
                frequency,
                phase_error_limit_rad / (2.0 * np.pi * frequency),
                phase_error_limit_rad,
                "rad",
            )
            for frequency in maximum_frequencies_hz
        )
        for criterion, frequency, threshold_s, limit_value, limit_unit in criteria:
            coarse_boundary = minimum_far_field_distance(
                positions,
                threshold_s,
                grid=grid,
                relative_distance_tolerance=2e-6,
            )
            refined_boundary = minimum_far_field_distance(
                positions,
                threshold_s,
                grid=refined_grid,
                relative_distance_tolerance=2e-6,
            )
            boundary_distance = refined_boundary.distance_m
            coarse_evaluation = far_field_error(
                positions, boundary_distance, grid=grid
            )
            refined_evaluation = far_field_error(
                positions, boundary_distance, grid=refined_grid
            )
            relative_grid_difference = abs(
                refined_evaluation.max_plane_error_s
                - coarse_evaluation.max_plane_error_s
            ) / refined_evaluation.max_plane_error_s
            continuous_used = relative_grid_difference > 1e-4
            continuous_result = (
                continuous_refine_far_field_error(
                    positions,
                    boundary_distance,
                    grid=refined_grid,
                    candidate_count=12,
                )
                if continuous_used
                else None
            )
            records.append(
                {
                    "record_type": "boundary",
                    "geometry": geometry_name,
                    "criterion": criterion,
                    "maximum_frequency_hz": frequency,
                    "limit_value": limit_value,
                    "limit_unit": limit_unit,
                    "target_error_s": threshold_s,
                    "target_error_us": threshold_s * 1e6,
                    "minimum_distance_m": boundary_distance,
                    "coarse_boundary_distance_m": coarse_boundary.distance_m,
                    "achieved_error_s": refined_evaluation.max_plane_error_s,
                    "achieved_error_us": refined_evaluation.max_plane_error_s * 1e6,
                    "coarse_max_error": coarse_evaluation.max_plane_error_s,
                    "refined_max_error": refined_evaluation.max_plane_error_s,
                    "relative_grid_difference": relative_grid_difference,
                    "refined_worst_azimuth_deg": refined_evaluation.worst_plane_azimuth_deg,
                    "refined_worst_elevation_deg": refined_evaluation.worst_plane_elevation_deg,
                    "refined_worst_pair": (
                        f"{refined_evaluation.worst_plane_pair[0]}-"
                        f"{refined_evaluation.worst_plane_pair[1]}"
                    ),
                    "worst_azimuth_deg": refined_evaluation.worst_plane_azimuth_deg,
                    "worst_elevation_deg": refined_evaluation.worst_plane_elevation_deg,
                    "worst_pair": (
                        f"{refined_evaluation.worst_plane_pair[0]}-"
                        f"{refined_evaluation.worst_plane_pair[1]}"
                    ),
                    "continuous_refinement_used": continuous_used,
                    "continuous_max_error": (
                        continuous_result.max_error_s if continuous_result else None
                    ),
                    "continuous_worst_azimuth_deg": (
                        continuous_result.worst_azimuth_deg if continuous_result else None
                    ),
                    "continuous_worst_elevation_deg": (
                        continuous_result.worst_elevation_deg if continuous_result else None
                    ),
                    "continuous_worst_pair": (
                        f"{continuous_result.worst_pair[0]}-{continuous_result.worst_pair[1]}"
                        if continuous_result
                        else None
                    ),
                    "coarse_search_iterations": coarse_boundary.iterations,
                    "refined_search_iterations": refined_boundary.iterations,
                    "sampling_rate_hz": sampling_rate_hz,
                    "angular_step_deg": angular_step_deg,
                    "refined_angular_step_deg": refined_step,
                    "max_elevation_deg": 80.0,
                }
            )

    _write_records(records, output_csv)
    return records


def _sinusoid_gain(
    signal: np.ndarray,
    frequency_hz: float,
    sampling_rate_hz: float,
    region: slice,
) -> complex:
    indices = np.arange(signal.size, dtype=float)[region]
    angular_frequency = 2.0 * np.pi * frequency_hz / sampling_rate_hz
    basis = np.column_stack(
        (np.cos(angular_frequency * indices), np.sin(angular_frequency * indices))
    )
    coefficients = np.linalg.lstsq(basis, signal[region], rcond=None)[0]
    return complex(coefficients[0], -coefficients[1])


def _bandlimited_pulse(length: int = 4096) -> np.ndarray:
    signal = np.zeros(length)
    pulse = firwin(513, 0.70, window=("kaiser", 8.6))
    start = (length - pulse.size) // 2
    signal[start : start + pulse.size] = pulse
    return signal


def _energy_centroid(signal: np.ndarray) -> float:
    energy = signal**2
    return float(np.dot(np.arange(signal.size), energy) / np.sum(energy))


def run_fractional_delay_accuracy_study(
    *,
    output_csv: str | Path = "results/fractional_delay_accuracy.csv",
    sampling_rate_hz: float = DIAGNOSTIC_SAMPLING_RATE_HZ,
    fir_length: int = DEFAULT_FIR_LENGTH,
    validation_guard_samples: int = FRACTIONAL_VALIDATION_GUARD_SAMPLES,
) -> list[dict[str, object]]:
    """Measure analytic sinusoid and deterministic broadband-delay errors."""

    sample_count = 8192
    indices = np.arange(sample_count, dtype=float)
    region = slice(validation_guard_samples, sample_count - validation_guard_samples)
    methods = {
        "frequency_domain": frequency_domain_delay,
        "windowed_sinc": lambda signal, delay: windowed_sinc_delay(
            signal, delay, fir_length=fir_length
        ),
    }
    records: list[dict[str, object]] = []
    for method_name, method in methods.items():
        for delay in FRACTIONAL_DELAYS_SAMPLES:
            for frequency in FRACTIONAL_TEST_FREQUENCIES_HZ:
                angular_frequency = 2.0 * np.pi * frequency / sampling_rate_hz
                signal = np.cos(angular_frequency * indices)
                delayed = method(signal, delay)
                analytic = np.cos(angular_frequency * (indices - delay))
                gain = _sinusoid_gain(delayed, frequency, sampling_rate_hz, region)
                expected_phase = -angular_frequency * delay
                records.append(
                    {
                        "record_type": "tone",
                        "method": method_name,
                        "delay_samples": delay,
                        "frequency_hz": frequency,
                        "normalized_frequency_cycles_per_sample": frequency
                        / sampling_rate_hz,
                        "amplitude_error": abs(abs(gain) - 1.0),
                        "phase_error_rad": abs(
                            np.angle(gain * np.exp(-1j * expected_phase))
                        ),
                        "max_waveform_error_valid": float(
                            np.max(np.abs(delayed[region] - analytic[region]))
                        ),
                        "sampling_rate_hz": sampling_rate_hz,
                        "fir_length": fir_length if method_name == "windowed_sinc" else None,
                        "fir_fixed_latency_samples": (
                            (fir_length - 1) // 2
                            if method_name == "windowed_sinc"
                            else None
                        ),
                        "validation_guard_samples": validation_guard_samples,
                    }
                )

    pulse = _bandlimited_pulse()
    pulse_centroid = _energy_centroid(pulse)
    for delay in FRACTIONAL_DELAYS_SAMPLES:
        frequency_result = frequency_domain_delay(pulse, delay)
        sinc_result = windowed_sinc_delay(pulse, delay, fir_length=fir_length)
        valid_start, valid_stop = fractional_delay_valid_region(
            pulse.size,
            delay,
            boundary_guard_samples=(fir_length - 1) // 2,
        )
        cross_method_error = float(
            np.max(
                np.abs(
                    frequency_result[valid_start:valid_stop]
                    - sinc_result[valid_start:valid_stop]
                )
            )
        )
        for method_name, delayed in (
            ("frequency_domain", frequency_result),
            ("windowed_sinc", sinc_result),
        ):
            measured_delay = _energy_centroid(delayed) - pulse_centroid
            records.append(
                {
                    "record_type": "broadband",
                    "method": method_name,
                    "delay_samples": delay,
                    "group_delay_error_samples": measured_delay - delay,
                    "cross_method_max_error_valid": cross_method_error,
                    "valid_start": valid_start,
                    "valid_stop": valid_stop,
                    "sampling_rate_hz": sampling_rate_hz,
                    "fir_length": fir_length if method_name == "windowed_sinc" else None,
                    "fir_fixed_latency_samples": (
                        (fir_length - 1) // 2
                        if method_name == "windowed_sinc"
                        else None
                    ),
                    "validation_guard_samples": validation_guard_samples,
                }
            )

    _write_records(records, output_csv)
    return records
