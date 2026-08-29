"""Reproducible signal-level Monte Carlo validation of GCC-PHAT and WLS DOA.

The diagnostic noise model is deliberately explicit and limited: independent
zero-mean Gaussian noise is added to every channel/sample after deterministic
propagation.  For each channel, ``SNR = 20 log10(signal_rms/noise_std)`` and
``signal_rms`` is measured on that channel in the common propagation valid
region.  This is not asserted to be a measured UAV-noise model.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from estimators.gcc_phat import estimate_tdoas_gcc_phat
from estimators.wls_doa import estimate_doa_wls
from model.geometry import (
    DEFAULT_SOUND_SPEED,
    comparison_arrays,
    direction_vector,
    reference_pairs,
)
from simulation.propagation import simulate_propagation
from simulation.signals import deterministic_bandlimited_signal

GCC_MONTE_CARLO_SEED = 20260827
GCC_MONTE_CARLO_ARRAYS = ("square", "tetrahedral")
GCC_MONTE_CARLO_DIRECTIONS_DEG = ((20.0, 10.0), (45.0, 30.0), (120.0, 50.0))
GCC_MONTE_CARLO_SNR_DB = (-20.0, -10.0, 0.0, 5.0, 10.0, 20.0)
GCC_MONTE_CARLO_TRIALS = 200


def _write_records(records: list[dict[str, object]], output_path: str | Path) -> Path:
    if not records:
        raise ValueError("at least one record is required")
    fieldnames = list(records[0])
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


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _configuration_seed(array_index: int, direction_index: int, snr_index: int) -> int:
    sequence = np.random.SeedSequence(
        [GCC_MONTE_CARLO_SEED, array_index, direction_index, snr_index]
    )
    return int(sequence.generate_state(1)[0])


def run_gcc_phat_monte_carlo(
    *,
    array_names: tuple[str, ...] = GCC_MONTE_CARLO_ARRAYS,
    directions_deg: tuple[tuple[float, float], ...] = GCC_MONTE_CARLO_DIRECTIONS_DEG,
    snr_levels_db: tuple[float, ...] = GCC_MONTE_CARLO_SNR_DB,
    trial_count: int = GCC_MONTE_CARLO_TRIALS,
    sampling_rate_hz: float = 48_000.0,
    signal_duration_s: float = 0.05,
    minimum_frequency_hz: float = 300.0,
    maximum_frequency_hz: float = 10_000.0,
    interpolation_factor: int = 8,
    delay_method: str = "windowed_sinc",
    output_csv: str | Path = "results/gcc_phat_monte_carlo.csv",
) -> list[dict[str, object]]:
    """Run independent-channel Gaussian-noise GCC/WLS experiments.

    The default detailed study has 36 configurations and 200 trials each
    (7200 trials).  Reference-microphone pairs are linearly independent but
    their GCC errors are not assumed statistically independent.  WLS uses
    equal pair weights; no signal-level CRLB is claimed.
    """

    trials = int(trial_count)
    if trials < 2:
        raise ValueError("trial_count must be at least 2")
    if delay_method not in {"frequency", "windowed_sinc"}:
        raise ValueError("delay_method must be 'frequency' or 'windowed_sinc'")
    arrays = comparison_arrays()
    unknown_arrays = set(array_names) - set(arrays)
    if unknown_arrays:
        raise ValueError(f"unknown arrays: {sorted(unknown_arrays)}")
    source_signal = deterministic_bandlimited_signal(
        sampling_rate_hz,
        signal_duration_s,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        tone_count=41,
    )
    records: list[dict[str, object]] = []
    for array_index, array_name in enumerate(array_names):
        positions = arrays[array_name]
        pairs = reference_pairs(len(positions), reference=0)
        delay_bounds = np.asarray(
            [
                np.linalg.norm(positions[i] - positions[j]) / DEFAULT_SOUND_SPEED
                + 2.0 / sampling_rate_hz
                for i, j in pairs
            ]
        )
        for direction_index, (azimuth_deg, elevation_deg) in enumerate(directions_deg):
            true_angles = np.deg2rad([azimuth_deg, elevation_deg])
            true_direction = direction_vector(*true_angles)
            propagation = simulate_propagation(
                source_signal,
                sampling_rate_hz,
                positions,
                phi=true_angles[0],
                elevation=true_angles[1],
                distance_m=20.0,
                propagation_model="plane",
                pairs=pairs,
                delay_method=delay_method,
            )
            start, stop = propagation.valid_region
            clean_channels = propagation.channels[:, start:stop]
            channel_rms = np.sqrt(np.mean(clean_channels**2, axis=1))
            true_tdoa = propagation.tdoa_seconds
            for snr_index, snr_db in enumerate(snr_levels_db):
                seed = _configuration_seed(array_index, direction_index, snr_index)
                generator = np.random.default_rng(seed)
                noise_std = channel_rms / (10.0 ** (float(snr_db) / 20.0))
                tdoa_estimates = np.full((trials, len(pairs)), np.nan)
                angle_estimates = np.full((trials, 2), np.nan)
                success = np.zeros(trials, dtype=bool)
                for trial in range(trials):
                    noisy = clean_channels + generator.normal(
                        0.0, noise_std[:, None], size=clean_channels.shape
                    )
                    estimated_tdoa, _ = estimate_tdoas_gcc_phat(
                        noisy,
                        sampling_rate_hz,
                        pairs,
                        maximum_delay_seconds=delay_bounds,
                        interpolation_factor=interpolation_factor,
                        minimum_frequency_hz=minimum_frequency_hz,
                        maximum_frequency_hz=maximum_frequency_hz,
                        relative_spectral_floor=1e-8,
                    )
                    tdoa_estimates[trial] = estimated_tdoa
                    doa = estimate_doa_wls(
                        estimated_tdoa,
                        positions,
                        pairs,
                        sigma_tdoa=1.0 / sampling_rate_hz,
                    )
                    angle_estimates[trial] = [doa.phi, doa.elevation]
                    success[trial] = doa.success

                tdoa_errors = tdoa_estimates - true_tdoa
                tdoa_error_samples = tdoa_errors * sampling_rate_hz
                azimuth_errors = _wrap_angle(angle_estimates[:, 0] - true_angles[0])
                elevation_errors = angle_estimates[:, 1] - true_angles[1]
                estimated_directions = np.asarray(
                    [direction_vector(phi, elevation) for phi, elevation in angle_estimates]
                )
                geodesic_errors = np.arccos(
                    np.clip(estimated_directions @ true_direction, -1.0, 1.0)
                )
                per_trial_max_tdoa = np.max(np.abs(tdoa_error_samples), axis=1)
                records.append(
                    {
                        "geometry": array_name,
                        "azimuth_deg": azimuth_deg,
                        "elevation_deg": elevation_deg,
                        "snr_db": snr_db,
                        "trial_count": trials,
                        "seed": seed,
                        "noise_model": "independent_channel_sample_gaussian",
                        "snr_definition": "20log10(valid_region_signal_rms/noise_std)_per_channel",
                        "signal_model": "deterministic_bandlimited_multisine",
                        "delay_method": delay_method,
                        "frequency_reference": "deterministic_cross_generator_validation",
                        "sampling_rate_hz": sampling_rate_hz,
                        "signal_duration_s": signal_duration_s,
                        "minimum_frequency_hz": minimum_frequency_hz,
                        "maximum_frequency_hz": maximum_frequency_hz,
                        "interpolation_factor": interpolation_factor,
                        "pair_scheme": "reference_0_linearly_independent",
                        "pair_count": len(pairs),
                        "tdoa_bias_us": float(np.mean(tdoa_errors) * 1e6),
                        "tdoa_rmse_us": float(np.sqrt(np.mean(tdoa_errors**2)) * 1e6),
                        "tdoa_p95_abs_error_us": float(
                            np.percentile(np.abs(tdoa_errors), 95.0) * 1e6
                        ),
                        "tdoa_trial_outlier_fraction_gt_0p5_sample": float(
                            np.mean(per_trial_max_tdoa > 0.5)
                        ),
                        "tdoa_trial_outlier_fraction_gt_1_sample": float(
                            np.mean(per_trial_max_tdoa > 1.0)
                        ),
                        "azimuth_bias_deg": float(np.rad2deg(np.mean(azimuth_errors))),
                        "elevation_bias_deg": float(np.rad2deg(np.mean(elevation_errors))),
                        "geodesic_rmse_deg": float(
                            np.rad2deg(np.sqrt(np.mean(geodesic_errors**2)))
                        ),
                        "median_geodesic_error_deg": float(
                            np.rad2deg(np.median(geodesic_errors))
                        ),
                        "p95_geodesic_error_deg": float(
                            np.rad2deg(np.percentile(geodesic_errors, 95.0))
                        ),
                        "fraction_geodesic_error_gt_10deg": float(
                            np.mean(geodesic_errors > np.deg2rad(10.0))
                        ),
                        "wls_success_fraction": float(np.mean(success)),
                        "mirror_ambiguous": array_name == "square",
                        "equal_pair_weight_assumption": True,
                    }
                )
    _write_records(records, output_csv)
    return records


__all__ = [
    "GCC_MONTE_CARLO_ARRAYS",
    "GCC_MONTE_CARLO_DIRECTIONS_DEG",
    "GCC_MONTE_CARLO_SEED",
    "GCC_MONTE_CARLO_SNR_DB",
    "GCC_MONTE_CARLO_TRIALS",
    "run_gcc_phat_monte_carlo",
]
