"""Deterministic GCC-PHAT validation and fractional-delay timing study."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from time import perf_counter

import numpy as np

from estimators.gcc_phat import estimate_tdoas_gcc_phat
from model.geometry import DEFAULT_SOUND_SPEED, all_pairs, comparison_arrays
from simulation.fractional_delay import frequency_domain_delay, windowed_sinc_delay
from simulation.propagation import simulate_propagation
from simulation.signals import deterministic_bandlimited_signal

GCC_SAMPLING_RATE_HZ = 48_000.0
GCC_MINIMUM_FREQUENCY_HZ = 300.0
GCC_MAXIMUM_FREQUENCY_HZ = 10_000.0
GCC_INTERPOLATION_FACTOR = 32
GCC_DIRECTIONS_DEG = ((20.0, 10.0), (45.0, 30.0), (120.0, 50.0))
GCC_ARRAY_NAMES = ("square", "tetrahedral")
GCC_PROPAGATION_MODELS = ("plane", "spherical")
GCC_DELAY_METHODS = ("frequency", "windowed_sinc")


def _write_records(records: list[dict[str, object]], output_path: str | Path) -> Path:
    if not records:
        raise ValueError("at least one record is required")
    fieldnames: list[str] = []
    for record in records:
        for name in record:
            if name not in fieldnames:
                fieldnames.append(name)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    content = buffer.getvalue().encode("utf-8")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)
    return path


def benchmark_fractional_delay_methods(
    *,
    sampling_rate_hz: float = GCC_SAMPLING_RATE_HZ,
    durations_s: tuple[float, ...] = (0.05, 0.25),
    delays_samples: tuple[float, ...] = (0.1, 7.25, 13.5, 27.9),
    warmup_count: int = 1,
    repeat_count: int = 7,
    output_csv: str | Path = "results/fractional_delay_benchmark.csv",
) -> list[dict[str, object]]:
    """Measure like-for-like synthesis time without asserting a timing threshold."""

    if warmup_count < 0 or repeat_count < 1:
        raise ValueError("warmup_count must be non-negative and repeat_count positive")
    implementations = {
        "frequency": frequency_domain_delay,
        "windowed_sinc": windowed_sinc_delay,
    }
    records: list[dict[str, object]] = []
    for duration in durations_s:
        signal = deterministic_bandlimited_signal(
            sampling_rate_hz,
            duration,
            minimum_frequency_hz=GCC_MINIMUM_FREQUENCY_HZ,
            maximum_frequency_hz=GCC_MAXIMUM_FREQUENCY_HZ,
            tone_count=41,
        )
        output_length = signal.size + int(np.ceil(max(delays_samples)))
        method_timings: dict[str, np.ndarray] = {}
        for method_name, implementation in implementations.items():

            def synthesize_all() -> None:
                for delay in delays_samples:
                    implementation(signal, delay, output_length=output_length)

            for _ in range(warmup_count):
                synthesize_all()
            elapsed = []
            for _ in range(repeat_count):
                start = perf_counter()
                synthesize_all()
                elapsed.append(perf_counter() - start)
            method_timings[method_name] = np.asarray(elapsed)
        frequency_median = float(np.median(method_timings["frequency"]))
        fir_median = float(np.median(method_timings["windowed_sinc"]))
        speedup = frequency_median / fir_median
        for method_name in implementations:
            elapsed = method_timings[method_name]
            records.append(
                {
                    "sampling_rate_hz": sampling_rate_hz,
                    "duration_s": duration,
                    "sample_count": signal.size,
                    "channel_count": len(delays_samples),
                    "delays_samples": ";".join(str(value) for value in delays_samples),
                    "method": method_name,
                    "warmup_count": warmup_count,
                    "repeat_count": repeat_count,
                    "minimum_time_s": float(np.min(elapsed)),
                    "median_time_s": float(np.median(elapsed)),
                    "maximum_time_s": float(np.max(elapsed)),
                    "frequency_to_fir_speedup": speedup,
                }
            )
    _write_records(records, output_csv)
    return records


def _pair_delay_bounds(positions: np.ndarray) -> np.ndarray:
    margin_seconds = 2.0 / GCC_SAMPLING_RATE_HZ
    return np.asarray(
        [
            np.linalg.norm(positions[first] - positions[second]) / DEFAULT_SOUND_SPEED
            + margin_seconds
            for first, second in all_pairs(len(positions))
        ]
    )


def run_gcc_phat_validation_study(
    *,
    output_csv: str | Path = "results/gcc_phat_validation.csv",
    signal_duration_s: float = 0.08,
) -> list[dict[str, object]]:
    """Compare clean GCC-PHAT TDOAs from both independent delay generators."""

    signal = deterministic_bandlimited_signal(
        GCC_SAMPLING_RATE_HZ,
        signal_duration_s,
        minimum_frequency_hz=GCC_MINIMUM_FREQUENCY_HZ,
        maximum_frequency_hz=GCC_MAXIMUM_FREQUENCY_HZ,
        tone_count=41,
    )
    arrays = comparison_arrays()
    records: list[dict[str, object]] = []
    for array_name in GCC_ARRAY_NAMES:
        positions = arrays[array_name]
        pairs = all_pairs(len(positions))
        bounds = _pair_delay_bounds(positions)
        for propagation_model in GCC_PROPAGATION_MODELS:
            for azimuth_deg, elevation_deg in GCC_DIRECTIONS_DEG:
                estimates_by_method: dict[str, np.ndarray] = {}
                truth: np.ndarray | None = None
                method_metrics: dict[str, float] = {}
                for delay_method in GCC_DELAY_METHODS:
                    propagation = simulate_propagation(
                        signal,
                        GCC_SAMPLING_RATE_HZ,
                        positions,
                        phi=np.deg2rad(azimuth_deg),
                        elevation=np.deg2rad(elevation_deg),
                        distance_m=20.0,
                        propagation_model=propagation_model,
                        pairs=pairs,
                        delay_method=delay_method,
                    )
                    estimates, _ = estimate_tdoas_gcc_phat(
                        propagation.channels,
                        GCC_SAMPLING_RATE_HZ,
                        pairs,
                        maximum_delay_seconds=bounds,
                        interpolation_factor=GCC_INTERPOLATION_FACTOR,
                        minimum_frequency_hz=GCC_MINIMUM_FREQUENCY_HZ,
                        maximum_frequency_hz=GCC_MAXIMUM_FREQUENCY_HZ,
                    )
                    truth = propagation.tdoa_seconds
                    estimates_by_method[delay_method] = estimates
                    errors_samples = (estimates - truth) * GCC_SAMPLING_RATE_HZ
                    prefix = "frequency" if delay_method == "frequency" else "fir"
                    method_metrics[f"{prefix}_max_abs_error_samples"] = float(
                        np.max(np.abs(errors_samples))
                    )
                    method_metrics[f"{prefix}_rmse_samples"] = float(
                        np.sqrt(np.mean(errors_samples**2))
                    )
                    method_metrics[f"{prefix}_bias_samples"] = float(np.mean(errors_samples))
                assert truth is not None
                cross_difference = (
                    estimates_by_method["frequency"]
                    - estimates_by_method["windowed_sinc"]
                ) * GCC_SAMPLING_RATE_HZ
                records.append(
                    {
                        "geometry": array_name,
                        "propagation_model": propagation_model,
                        "azimuth_deg": azimuth_deg,
                        "elevation_deg": elevation_deg,
                        "distance_m": 20.0,
                        "sampling_rate_hz": GCC_SAMPLING_RATE_HZ,
                        "minimum_frequency_hz": GCC_MINIMUM_FREQUENCY_HZ,
                        "maximum_frequency_hz": GCC_MAXIMUM_FREQUENCY_HZ,
                        "interpolation_factor": GCC_INTERPOLATION_FACTOR,
                        "pair_count": len(pairs),
                        **method_metrics,
                        "generator_max_difference_samples": float(
                            np.max(np.abs(cross_difference))
                        ),
                        "frequency_reference_retained": True,
                    }
                )
    _write_records(records, output_csv)
    return records


__all__ = [
    "GCC_ARRAY_NAMES",
    "GCC_DELAY_METHODS",
    "GCC_DIRECTIONS_DEG",
    "GCC_INTERPOLATION_FACTOR",
    "GCC_MAXIMUM_FREQUENCY_HZ",
    "GCC_MINIMUM_FREQUENCY_HZ",
    "GCC_PROPAGATION_MODELS",
    "GCC_SAMPLING_RATE_HZ",
    "benchmark_fractional_delay_methods",
    "run_gcc_phat_validation_study",
]
