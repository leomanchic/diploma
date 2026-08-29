"""Calibration/evaluation statistical validation for all-pair GCC-PHAT.

This module keeps raw trial arrays in memory only. CSV outputs contain pair,
DOA, covariance, confidence, and tail aggregates. Calibration and evaluation
use independent ``SeedSequence`` streams and are never pooled for covariance
estimation or quality reporting.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from estimators.cycle_projection import project_tdoa_cycles
from estimators.gcc_phat import estimate_tdoas_gcc_phat
from estimators.wls_doa import estimate_doa_spherical_wls, estimate_doa_wls
from model.geometry import (
    DEFAULT_SOUND_SPEED,
    Pair,
    all_pairs,
    comparison_arrays,
    direction_vector,
    incidence_matrix,
)
from model.tdoa import far_field_tdoa, tdoa_covariance_from_independent_toa
from simulation.fractional_delay import DEFAULT_FIR_LENGTH
from simulation.propagation import simulate_propagation
from simulation.signals import (
    deterministic_bandlimited_signal,
    harmonic_stress_signal,
    random_bandlimited_signal,
)
from validation.far_field import direction_grid, far_field_error

STATISTICAL_BASE_SEED = 20260828
CALIBRATION_TRIAL_COUNT = 1000
EVALUATION_TRIAL_COUNT = 2000
FINE_SNR_LEVELS_DB = (-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 30.0)
FRAME_LENGTHS = (1024, 2048, 4096, 8192)
FRAME_SNR_LEVELS_DB = (-5.0, 0.0, 5.0, 10.0)
SIGNAL_MODELS = ("deterministic_multisine", "random_broadband", "harmonic_stress")
DOA_VARIANTS = (
    "reference_3_equal",
    "all_6_equal",
    "all_6_cycle_equal",
    "all_6_cycle_calibrated_no_rejection",
    "all_6_cycle_calibrated_hard_p05",
    "all_6_cycle_calibrated",
    "all_6_cycle_calibrated_hard_p25",
    "all_6_cycle_calibrated_hard_p50",
    "all_6_cycle_calibrated_soft",
)
RISK_COVERAGE_PERCENTILES = (5.0, 10.0, 25.0, 50.0)


@dataclass(frozen=True)
class GCCStatisticalConfig:
    study_type: str
    signal_model: str
    geometry: str
    azimuth_deg: float
    elevation_deg: float
    snr_db: float
    frame_length: int
    propagation_model: str = "plane"
    distance_m: float = 20.0

    @property
    def direction(self) -> str:
        return f"az{self.azimuth_deg:g}_el{self.elevation_deg:g}"


@dataclass(frozen=True)
class GCCTrialSet:
    estimated_tdoa: NDArray[np.float64]
    peak_value: NDArray[np.float64]
    peak_ratio: NDArray[np.float64]
    peak_curvature: NDArray[np.float64]
    spectral_energy: NDArray[np.float64]
    spectral_energy_fraction: NDArray[np.float64]
    boundary_hit: NDArray[np.bool_]
    invalid: NDArray[np.bool_]
    seed: int


def _write_records(records: list[dict[str, object]], output_path: str | Path) -> Path:
    if not records:
        raise ValueError("at least one record is required")
    fields: list[str] = []
    for record in records:
        for field in record:
            if field not in fields:
                fields.append(field)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    content = buffer.getvalue().encode("utf-8")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)
    return path


def _split_seed(configuration_index: int, split: int) -> int:
    sequence = np.random.SeedSequence(
        [STATISTICAL_BASE_SEED, int(configuration_index), int(split)]
    )
    return int(sequence.generate_state(1)[0])


def _source_signal(
    model: str,
    sampling_rate_hz: float,
    sample_count: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if model == "deterministic_multisine":
        return deterministic_bandlimited_signal(
            sampling_rate_hz,
            sample_count / sampling_rate_hz,
            minimum_frequency_hz=300.0,
            maximum_frequency_hz=10_000.0,
            tone_count=41,
        )
    if model == "random_broadband":
        return random_bandlimited_signal(
            sampling_rate_hz,
            sample_count,
            rng,
            minimum_frequency_hz=300.0,
            maximum_frequency_hz=10_000.0,
        )
    if model == "harmonic_stress":
        return harmonic_stress_signal(
            sampling_rate_hz,
            sample_count,
            rng=rng,
            fundamental_frequency_hz=240.0,
            maximum_frequency_hz=10_000.0,
        )
    raise ValueError(f"unknown signal_model: {model}")


def _clean_frame(
    config: GCCStatisticalConfig,
    positions: NDArray[np.float64],
    pairs: tuple[Pair, ...],
    sampling_rate_hz: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    source_count = config.frame_length + 2 * DEFAULT_FIR_LENGTH + 64
    source = _source_signal(config.signal_model, sampling_rate_hz, source_count, rng)
    propagation = simulate_propagation(
        source,
        sampling_rate_hz,
        positions,
        phi=np.deg2rad(config.azimuth_deg),
        elevation=np.deg2rad(config.elevation_deg),
        distance_m=config.distance_m,
        propagation_model=config.propagation_model,
        pairs=pairs,
        delay_method="windowed_sinc",
    )
    start, stop = propagation.valid_region
    if stop - start < config.frame_length:
        raise RuntimeError("propagation valid region is shorter than the requested frame")
    return (
        propagation.channels[:, start : start + config.frame_length],
        propagation.tdoa_seconds,
    )


def simulate_gcc_trial_set(
    config: GCCStatisticalConfig,
    trial_count: int,
    seed: int,
    *,
    sampling_rate_hz: float = 48_000.0,
    interpolation_factor: int = 2,
) -> tuple[GCCTrialSet, NDArray[np.float64], tuple[Pair, ...]]:
    """Generate one independent split and estimate all six oriented pairs."""

    count = int(trial_count)
    if count < 2:
        raise ValueError("trial_count must be at least 2")
    if config.signal_model not in SIGNAL_MODELS:
        raise ValueError(f"signal_model must be one of {SIGNAL_MODELS}")
    positions = comparison_arrays()[config.geometry]
    pairs = all_pairs(len(positions))
    bounds = np.asarray(
        [
            np.linalg.norm(positions[first] - positions[second]) / DEFAULT_SOUND_SPEED
            + 2.0 / sampling_rate_hz
            for first, second in pairs
        ]
    )
    generator = np.random.default_rng(seed)
    shape = (count, len(pairs))
    estimated = np.full(shape, np.nan)
    peak_value = np.full(shape, np.nan)
    peak_ratio = np.full(shape, np.nan)
    peak_curvature = np.full(shape, np.nan)
    spectral_energy = np.full(shape, np.nan)
    spectral_fraction = np.full(shape, np.nan)
    boundary = np.zeros(shape, dtype=bool)
    invalid = np.zeros(shape, dtype=bool)
    true_tdoa: NDArray[np.float64] | None = None

    deterministic_clean: NDArray[np.float64] | None = None
    if config.signal_model == "deterministic_multisine":
        deterministic_clean, true_tdoa = _clean_frame(
            config, positions, pairs, sampling_rate_hz, generator
        )
    amplitude_ratio = 10.0 ** (config.snr_db / 20.0)
    for trial in range(count):
        if deterministic_clean is None:
            clean, trial_truth = _clean_frame(
                config, positions, pairs, sampling_rate_hz, generator
            )
            if true_tdoa is None:
                true_tdoa = trial_truth
            else:
                np.testing.assert_allclose(trial_truth, true_tdoa, rtol=0.0, atol=2e-18)
        else:
            clean = deterministic_clean
        channel_rms = np.sqrt(np.mean(clean**2, axis=1))
        noisy = clean + generator.normal(
            0.0, channel_rms[:, None] / amplitude_ratio, size=clean.shape
        )
        delays, diagnostics = estimate_tdoas_gcc_phat(
            noisy,
            sampling_rate_hz,
            pairs,
            maximum_delay_seconds=bounds,
            interpolation_factor=interpolation_factor,
            minimum_frequency_hz=200.0,
            maximum_frequency_hz=10_000.0,
            relative_spectral_floor=1e-8,
        )
        estimated[trial] = delays
        for pair_index, result in enumerate(diagnostics):
            peak_value[trial, pair_index] = result.peak_value
            peak_ratio[trial, pair_index] = result.peak_to_second_peak_ratio
            peak_curvature[trial, pair_index] = result.peak_curvature
            spectral_energy[trial, pair_index] = result.used_spectral_energy
            spectral_fraction[trial, pair_index] = result.spectral_energy_fraction
            boundary[trial, pair_index] = result.boundary_hit
            invalid[trial, pair_index] = result.invalid
    assert true_tdoa is not None
    return (
        GCCTrialSet(
            estimated_tdoa=estimated,
            peak_value=peak_value,
            peak_ratio=peak_ratio,
            peak_curvature=peak_curvature,
            spectral_energy=spectral_energy,
            spectral_energy_fraction=spectral_fraction,
            boundary_hit=boundary,
            invalid=invalid,
            seed=int(seed),
        ),
        true_tdoa,
        pairs,
    )


def _context(
    config: GCCStatisticalConfig,
    split: str,
    pair: str,
    calibration_count: int,
    evaluation_count: int,
    seed: int,
    estimator_variant: str,
) -> dict[str, object]:
    return {
        "study_type": config.study_type,
        "signal_model": config.signal_model,
        "split": split,
        "geometry": config.geometry,
        "direction": config.direction,
        "azimuth_deg": config.azimuth_deg,
        "elevation_deg": config.elevation_deg,
        "SNR": config.snr_db,
        "snr_db": config.snr_db,
        "frame_length": config.frame_length,
        "pair": pair,
        "calibration_trial_count": calibration_count,
        "evaluation_trial_count": evaluation_count,
        "seed": seed,
        "estimator_variant": estimator_variant,
        "propagation_model": config.propagation_model,
        "distance_m": config.distance_m,
    }


def _finite_mean(values: NDArray[np.float64]) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _pair_records(
    config: GCCStatisticalConfig,
    split_name: str,
    trials: GCCTrialSet,
    true_tdoa: NDArray[np.float64],
    pairs: tuple[Pair, ...],
    calibration_count: int,
    evaluation_count: int,
    sampling_rate_hz: float,
) -> list[dict[str, object]]:
    records = []
    for pair_index, pair in enumerate(pairs):
        errors = trials.estimated_tdoa[:, pair_index] - true_tdoa[pair_index]
        invalid = trials.invalid[:, pair_index] | ~np.isfinite(errors)
        valid_errors = errors[~invalid]
        absolute = np.abs(valid_errors)
        catastrophic = invalid.copy()
        catastrophic[~invalid] = absolute * sampling_rate_hz > 1.0
        records.append(
            {
                **_context(
                    config,
                    split_name,
                    f"{pair[0]}-{pair[1]}",
                    calibration_count,
                    evaluation_count,
                    trials.seed,
                    "gcc_phat_pair",
                ),
                "true_tdoa_s": true_tdoa[pair_index],
                "true_tdoa_us": true_tdoa[pair_index] * 1e6,
                "bias_s": float(np.mean(valid_errors)) if valid_errors.size else float("nan"),
                "bias_us": float(np.mean(valid_errors) * 1e6) if valid_errors.size else float("nan"),
                "standard_deviation_us": float(np.std(valid_errors, ddof=1) * 1e6)
                if valid_errors.size > 1
                else float("nan"),
                "rmse_us": float(np.sqrt(np.mean(valid_errors**2)) * 1e6)
                if valid_errors.size
                else float("nan"),
                "median_absolute_error_us": float(np.median(absolute) * 1e6)
                if absolute.size
                else float("nan"),
                "p95_absolute_error_us": float(np.percentile(absolute, 95.0) * 1e6)
                if absolute.size
                else float("nan"),
                "catastrophic_outlier_fraction_gt_1_sample": float(np.mean(catastrophic)),
                "mean_peak_value": _finite_mean(trials.peak_value[:, pair_index]),
                "mean_peak_to_second_peak_ratio": _finite_mean(
                    trials.peak_ratio[:, pair_index]
                ),
                "median_peak_to_second_peak_ratio": float(
                    np.nanmedian(trials.peak_ratio[:, pair_index])
                ),
                "mean_peak_curvature": _finite_mean(trials.peak_curvature[:, pair_index]),
                "mean_used_spectral_energy": _finite_mean(
                    trials.spectral_energy[:, pair_index]
                ),
                "mean_spectral_energy_fraction": _finite_mean(
                    trials.spectral_energy_fraction[:, pair_index]
                ),
                "boundary_hit_fraction": float(
                    np.mean(trials.boundary_hit[:, pair_index])
                ),
                "invalid_fraction": float(np.mean(invalid)),
            }
        )
    return records


def _calibration_statistics(
    calibration: GCCTrialSet,
    true_tdoa: NDArray[np.float64],
    sampling_rate_hz: float,
) -> dict[str, object]:
    errors = calibration.estimated_tdoa - true_tdoa
    complete = np.all(np.isfinite(errors) & ~calibration.invalid, axis=1)
    if np.count_nonzero(complete) < 2:
        raise RuntimeError("too few complete calibration trials")
    complete_errors = errors[complete]
    bias = np.mean(complete_errors, axis=0)
    covariance = np.cov(complete_errors, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    pair_count = covariance.shape[0]
    regularizer = max(float(np.trace(covariance)) / pair_count, 1e-20) * 1e-6
    covariance_regularized = covariance + regularizer * np.eye(pair_count)
    precision = np.linalg.inv(covariance_regularized)
    thresholds = np.zeros(pair_count)
    thresholds_by_percentile = {
        percentile: np.zeros(pair_count) for percentile in RISK_COVERAGE_PERCENTILES
    }
    ratio_medians = np.ones(pair_count)
    confidence_correlations = np.full(pair_count, np.nan)
    for pair_index in range(pair_count):
        ratios = calibration.peak_ratio[:, pair_index]
        absolute_samples = np.abs(errors[:, pair_index]) * sampling_rate_hz
        usable = np.isfinite(ratios) & np.isfinite(absolute_samples) & ~calibration.invalid[:, pair_index]
        finite_ratios = ratios[usable]
        if finite_ratios.size:
            cap = np.percentile(finite_ratios, 99.0)
            finite_ratios = np.minimum(finite_ratios, cap)
            ratio_medians[pair_index] = max(float(np.median(finite_ratios)), 1e-12)
            inliers = usable & (absolute_samples <= 1.0) & ~calibration.boundary_hit[:, pair_index]
            threshold_source = ratios[inliers] if np.any(inliers) else ratios[usable]
            for percentile in RISK_COVERAGE_PERCENTILES:
                thresholds_by_percentile[percentile][pair_index] = float(
                    np.percentile(threshold_source, percentile)
                )
            thresholds[pair_index] = thresholds_by_percentile[10.0][pair_index]
            if np.count_nonzero(usable) >= 3:
                correlation = spearmanr(ratios[usable], absolute_samples[usable]).statistic
                confidence_correlations[pair_index] = float(correlation)
        else:
            thresholds[pair_index] = float("inf")
            for percentile in RISK_COVERAGE_PERCENTILES:
                thresholds_by_percentile[percentile][pair_index] = float("inf")
    return {
        "bias": bias,
        "covariance": covariance,
        "precision": precision,
        "confidence_thresholds": thresholds,
        "confidence_thresholds_by_percentile": thresholds_by_percentile,
        "ratio_medians": ratio_medians,
        "confidence_error_spearman": confidence_correlations,
        "complete_fraction": float(np.mean(complete)),
    }


def _matrix_json(matrix: NDArray[np.float64]) -> str:
    return json.dumps(np.asarray(matrix, dtype=float).tolist(), separators=(",", ":"))


def _covariance_record(
    config: GCCStatisticalConfig,
    calibration: GCCTrialSet,
    true_tdoa: NDArray[np.float64],
    pairs: tuple[Pair, ...],
    statistics: dict[str, object],
    calibration_count: int,
    evaluation_count: int,
) -> dict[str, object]:
    covariance = np.asarray(statistics["covariance"])
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    correlation = covariance / np.outer(standard_deviations, standard_deviations)
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
    positive = eigenvalues[eigenvalues > max(np.max(eigenvalues), 1e-30) * 1e-12]
    condition = float(np.max(positive) / np.min(positive)) if positive.size else float("inf")
    mean_variance = float(np.mean(np.diag(covariance)))
    independent_tdoa = np.eye(len(pairs)) * mean_variance
    sigma_toa = np.sqrt(mean_variance / 2.0)
    independent_toa = tdoa_covariance_from_independent_toa(4, pairs, sigma_toa)
    scale = max(float(np.linalg.norm(covariance)), 1e-30)
    return {
        **_context(
            config,
            "calibration",
            "all_6",
            calibration_count,
            evaluation_count,
            calibration.seed,
            "gaussian_covariance_benchmark",
        ),
        "pair_bias_vector_us": _matrix_json(np.asarray(statistics["bias"]) * 1e6),
        "covariance_matrix_us2": _matrix_json(covariance * 1e12),
        "correlation_matrix": _matrix_json(correlation),
        "eigenvalues_us2": _matrix_json(eigenvalues * 1e12),
        "condition_number": condition,
        "rank": int(np.linalg.matrix_rank(covariance)),
        "confidence_threshold_vector": _matrix_json(
            np.asarray(statistics["confidence_thresholds"])
        ),
        "confidence_thresholds_by_percentile": json.dumps(
            {
                f"p{int(percentile):02d}": np.asarray(values, dtype=float).tolist()
                for percentile, values in dict(
                    statistics["confidence_thresholds_by_percentile"]
                ).items()
            },
            separators=(",", ":"),
        ),
        "confidence_error_spearman_vector": _matrix_json(
            np.asarray(statistics["confidence_error_spearman"])
        ),
        "mean_confidence_error_spearman": float(
            np.nanmean(np.asarray(statistics["confidence_error_spearman"]))
        ),
        "complete_calibration_fraction": statistics["complete_fraction"],
        "relative_error_vs_independent_tdoa_covariance": float(
            np.linalg.norm(covariance - independent_tdoa) / scale
        ),
        "relative_error_vs_independent_toa_covariance": float(
            np.linalg.norm(covariance - independent_toa) / scale
        ),
        "independent_tdoa_covariance_us2": _matrix_json(independent_tdoa * 1e12),
        "independent_toa_covariance_us2": _matrix_json(independent_toa * 1e12),
        "benchmark_name": "Gaussian covariance benchmark",
        "exact_crlb_claimed": False,
    }


def _graph_connected(accepted: NDArray[np.bool_], pairs: tuple[Pair, ...]) -> bool:
    if np.count_nonzero(accepted) < 3:
        return False
    matrix = incidence_matrix(
        tuple(pair for pair, keep in zip(pairs, accepted, strict=True) if keep), 4
    )
    return int(np.linalg.matrix_rank(matrix)) == 3


def _confidence_weights(
    ratios: NDArray[np.float64],
    medians: NDArray[np.float64],
    precision: NDArray[np.float64],
) -> NDArray[np.float64]:
    confidence = np.clip(ratios / medians, 0.05, 20.0)
    root = np.sqrt(confidence)
    weighted = root[:, None] * precision * root[None, :]
    return (weighted + weighted.T) / 2.0


_RISK_VARIANT_SPECS: dict[str, tuple[float | None, bool]] = {
    "all_6_cycle_calibrated_no_rejection": (None, False),
    "all_6_cycle_calibrated_hard_p05": (5.0, False),
    "all_6_cycle_calibrated": (10.0, False),
    "all_6_cycle_calibrated_hard_p25": (25.0, False),
    "all_6_cycle_calibrated_hard_p50": (50.0, False),
    "all_6_cycle_calibrated_soft": (None, True),
}


def _direction_metrics(
    estimates: NDArray[np.float64],
    valid: NDArray[np.bool_],
    true_angles: NDArray[np.float64],
) -> dict[str, float]:
    usable = estimates[valid]
    successful_count = int(np.count_nonzero(valid))
    total_count = int(valid.size)
    count_metrics = {
        "successful_trial_count": successful_count,
        "unsuccessful_trial_count": total_count - successful_count,
        "successful_fraction": float(successful_count / total_count) if total_count else 0.0,
    }
    if usable.size == 0:
        return {
            **count_metrics,
            "conditional_azimuth_bias_deg": float("nan"),
            "conditional_elevation_bias_deg": float("nan"),
            "conditional_direction_bias_deg": float("nan"),
            "conditional_geodesic_rmse_deg": float("nan"),
            "conditional_median_geodesic_error_deg": float("nan"),
            "conditional_p95_geodesic_error_deg": float("nan"),
            "conditional_p99_geodesic_error_deg": float("nan"),
            "conditional_p999_geodesic_error_deg": float("nan"),
            "p99_geodesic_error_deg": float("nan"),
            "p999_geodesic_error_deg": float("nan"),
            "fraction_error_gt_5deg": float("nan"),
            "fraction_error_gt_10deg": float("nan"),
            "fraction_error_gt_30deg": float("nan"),
        }
    azimuth_error = (usable[:, 0] - true_angles[0] + np.pi) % (2 * np.pi) - np.pi
    elevation_error = usable[:, 1] - true_angles[1]
    directions = np.asarray([direction_vector(*angles) for angles in usable])
    truth = direction_vector(*true_angles)
    geodesic = np.arccos(np.clip(directions @ truth, -1.0, 1.0))
    mean_direction = np.mean(directions, axis=0)
    mean_direction /= np.linalg.norm(mean_direction)
    direction_bias = np.arccos(np.clip(mean_direction @ truth, -1.0, 1.0))
    p99 = float(np.rad2deg(np.percentile(geodesic, 99.0)))
    p999 = float(np.rad2deg(np.percentile(geodesic, 99.9)))
    return {
        **count_metrics,
        "conditional_azimuth_bias_deg": float(np.rad2deg(np.mean(azimuth_error))),
        "conditional_elevation_bias_deg": float(np.rad2deg(np.mean(elevation_error))),
        "conditional_direction_bias_deg": float(np.rad2deg(direction_bias)),
        "conditional_geodesic_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(geodesic**2)))
        ),
        "conditional_median_geodesic_error_deg": float(
            np.rad2deg(np.median(geodesic))
        ),
        "conditional_p95_geodesic_error_deg": float(
            np.rad2deg(np.percentile(geodesic, 95.0))
        ),
        "conditional_p99_geodesic_error_deg": p99,
        "conditional_p999_geodesic_error_deg": p999,
        "p99_geodesic_error_deg": p99,
        "p999_geodesic_error_deg": p999,
        "fraction_error_gt_5deg": float(np.mean(geodesic > np.deg2rad(5.0))),
        "fraction_error_gt_10deg": float(np.mean(geodesic > np.deg2rad(10.0))),
        "fraction_error_gt_30deg": float(np.mean(geodesic > np.deg2rad(30.0))),
    }


def _doa_records(
    config: GCCStatisticalConfig,
    evaluation: GCCTrialSet,
    true_tdoa: NDArray[np.float64],
    pairs: tuple[Pair, ...],
    statistics: dict[str, object],
    calibration_count: int,
    evaluation_count: int,
    sampling_rate_hz: float,
) -> list[dict[str, object]]:
    positions = comparison_arrays()[config.geometry]
    true_angles = np.deg2rad([config.azimuth_deg, config.elevation_deg])
    estimates_by_variant = {
        variant: np.full((evaluation_count, 2), np.nan) for variant in DOA_VARIANTS
    }
    valid_by_variant = {
        variant: np.zeros(evaluation_count, dtype=bool) for variant in DOA_VARIANTS
    }
    cycle_before = {variant: [] for variant in DOA_VARIANTS}
    cycle_after = {variant: [] for variant in DOA_VARIANTS}
    rejected_pair_fraction = {variant: [] for variant in DOA_VARIANTS}
    unavailable_pair_fraction = {variant: [] for variant in DOA_VARIANTS}
    disconnected = {variant: np.zeros(evaluation_count, dtype=bool) for variant in DOA_VARIANTS}
    covariance = np.asarray(statistics["covariance"])
    precision = np.asarray(statistics["precision"])
    bias = np.asarray(statistics["bias"])
    thresholds = np.asarray(statistics["confidence_thresholds"])
    thresholds_by_percentile = {
        float(percentile): np.asarray(values, dtype=float)
        for percentile, values in dict(
            statistics["confidence_thresholds_by_percentile"]
        ).items()
    }
    medians = np.asarray(statistics["ratio_medians"])
    reference_indices = np.asarray([0, 1, 2])
    reference_pairs = tuple(pairs[index] for index in reference_indices)

    for trial in range(evaluation_count):
        observed = evaluation.estimated_tdoa[trial]
        finite = np.isfinite(observed) & ~evaluation.invalid[trial]
        if np.all(finite[reference_indices]):
            doa = estimate_doa_wls(
                observed[reference_indices],
                positions,
                reference_pairs,
                sigma_tdoa=1.0 / sampling_rate_hz,
            )
            estimates_by_variant["reference_3_equal"][trial] = [doa.phi, doa.elevation]
            valid_by_variant["reference_3_equal"][trial] = doa.success
        if np.all(finite):
            doa = estimate_doa_wls(
                observed, positions, pairs, sigma_tdoa=1.0 / sampling_rate_hz
            )
            estimates_by_variant["all_6_equal"][trial] = [doa.phi, doa.elevation]
            valid_by_variant["all_6_equal"][trial] = doa.success
            projected = project_tdoa_cycles(observed, pairs, 4)
            doa = estimate_doa_wls(
                projected.consistent_tdoa,
                positions,
                pairs,
                sigma_tdoa=1.0 / sampling_rate_hz,
            )
            estimates_by_variant["all_6_cycle_equal"][trial] = [doa.phi, doa.elevation]
            valid_by_variant["all_6_cycle_equal"][trial] = doa.success
            cycle_before["all_6_cycle_equal"].append(projected.cycle_residual_before)
            cycle_after["all_6_cycle_equal"].append(projected.cycle_residual_after)

        corrected = observed - bias
        ratios = evaluation.peak_ratio[trial]
        usable_confidence = np.isfinite(ratios)
        for variant, (percentile, soft_weighting) in _RISK_VARIANT_SPECS.items():
            available = finite.copy()
            unavailable_pair_fraction[variant].append(1.0 - float(np.mean(available)))
            if percentile is None:
                accepted = available
                rejected_pair_fraction[variant].append(0.0)
            else:
                threshold = thresholds_by_percentile[percentile]
                accepted = (
                    available
                    & ~evaluation.boundary_hit[trial]
                    & usable_confidence
                    & (ratios >= threshold)
                )
                rejected_pair_fraction[variant].append(
                    float(np.mean(available & ~accepted))
                )
            if not _graph_connected(accepted, pairs):
                disconnected[variant][trial] = True
                continue
            selected_pairs = tuple(
                pair for pair, keep in zip(pairs, accepted, strict=True) if keep
            )
            if soft_weighting or percentile is not None:
                safe_ratios = np.where(usable_confidence, ratios, medians)
                full_weights = _confidence_weights(safe_ratios, medians, precision)
            else:
                full_weights = precision
            selected_weights = full_weights[np.ix_(accepted, accepted)]
            calibrated = project_tdoa_cycles(
                corrected[accepted], selected_pairs, 4, weights=selected_weights
            )
            covariance_weighted = np.linalg.pinv(selected_weights, rcond=1e-12)
            doa = estimate_doa_wls(
                calibrated.consistent_tdoa,
                positions,
                selected_pairs,
                tdoa_covariance=covariance_weighted,
            )
            estimates_by_variant[variant][trial] = [doa.phi, doa.elevation]
            valid_by_variant[variant][trial] = doa.success
            cycle_before[variant].append(calibrated.cycle_residual_before)
            cycle_after[variant].append(calibrated.cycle_residual_after)

    records = []
    any_pair_invalid = np.any(evaluation.invalid, axis=1)
    any_boundary = np.any(evaluation.boundary_hit, axis=1)
    for variant in DOA_VARIANTS:
        valid = valid_by_variant[variant]
        metrics = _direction_metrics(estimates_by_variant[variant], valid, true_angles)
        is_risk_variant = variant in _RISK_VARIANT_SPECS
        percentile, soft_weighting = _RISK_VARIANT_SPECS.get(
            variant, (None, False)
        )
        records.append(
            {
                **_context(
                    config,
                    "evaluation",
                    "reference_3" if variant == "reference_3_equal" else "all_6",
                    calibration_count,
                    evaluation_count,
                    evaluation.seed,
                    variant,
                ),
                **metrics,
                "coverage": metrics["successful_fraction"],
                "failure_fraction": float(np.mean(~valid)),
                "failure_invalid_fraction": float(np.mean(~valid)),
                "gcc_any_pair_invalid_fraction": float(np.mean(any_pair_invalid)),
                "gcc_any_pair_boundary_fraction": float(np.mean(any_boundary)),
                "mean_rejected_pair_fraction": float(
                    np.mean(rejected_pair_fraction[variant])
                )
                if rejected_pair_fraction[variant]
                else 0.0,
                "mean_unavailable_pair_fraction": float(
                    np.mean(unavailable_pair_fraction[variant])
                )
                if unavailable_pair_fraction[variant]
                else 0.0,
                "disconnected_graph_failure_fraction": float(
                    np.mean(disconnected[variant])
                ),
                "catastrophic_fraction": metrics["fraction_error_gt_30deg"],
                "catastrophic_definition": "conditional_geodesic_error_gt_30deg",
                "mean_cycle_residual_before_us": float(
                    np.mean(cycle_before[variant]) * 1e6
                )
                if cycle_before[variant]
                else 0.0,
                "mean_cycle_residual_after_us": float(
                    np.mean(cycle_after[variant]) * 1e6
                )
                if cycle_after[variant]
                else 0.0,
                "cycle_residual_definition": "euclidean_norm_of_cycle_coordinates",
                "cycle_residual_aggregation": "mean_over_successfully_projected_trials",
                "risk_coverage_study": is_risk_variant,
                "confidence_threshold_percentile": percentile,
                "hard_confidence_rejection_used": is_risk_variant
                and percentile is not None,
                "soft_confidence_weighting_used": is_risk_variant and soft_weighting,
                "calibration_covariance_used": is_risk_variant,
                "confidence_weights_used": is_risk_variant
                and (soft_weighting or percentile is not None),
                "metric_conditioning": "successful_trials_only",
                "exact_crlb_claimed": False,
            }
        )
    return records


def run_gcc_statistical_configuration(
    config: GCCStatisticalConfig,
    configuration_index: int,
    *,
    calibration_trial_count: int = CALIBRATION_TRIAL_COUNT,
    evaluation_trial_count: int = EVALUATION_TRIAL_COUNT,
    sampling_rate_hz: float = 48_000.0,
    interpolation_factor: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Run independent calibration/evaluation and return three record groups."""

    calibration_seed = _split_seed(configuration_index, 0)
    evaluation_seed = _split_seed(configuration_index, 1)
    calibration, true_tdoa, pairs = simulate_gcc_trial_set(
        config,
        calibration_trial_count,
        calibration_seed,
        sampling_rate_hz=sampling_rate_hz,
        interpolation_factor=interpolation_factor,
    )
    evaluation, evaluation_truth, evaluation_pairs = simulate_gcc_trial_set(
        config,
        evaluation_trial_count,
        evaluation_seed,
        sampling_rate_hz=sampling_rate_hz,
        interpolation_factor=interpolation_factor,
    )
    np.testing.assert_allclose(evaluation_truth, true_tdoa, rtol=0.0, atol=2e-18)
    if evaluation_pairs != pairs:
        raise RuntimeError("calibration/evaluation pair sets differ")
    statistics = _calibration_statistics(calibration, true_tdoa, sampling_rate_hz)
    pair_records = _pair_records(
        config,
        "calibration",
        calibration,
        true_tdoa,
        pairs,
        calibration_trial_count,
        evaluation_trial_count,
        sampling_rate_hz,
    ) + _pair_records(
        config,
        "evaluation",
        evaluation,
        true_tdoa,
        pairs,
        calibration_trial_count,
        evaluation_trial_count,
        sampling_rate_hz,
    )
    doa_records = _doa_records(
        config,
        evaluation,
        true_tdoa,
        pairs,
        statistics,
        calibration_trial_count,
        evaluation_trial_count,
        sampling_rate_hz,
    )
    covariance_record = _covariance_record(
        config,
        calibration,
        true_tdoa,
        pairs,
        statistics,
        calibration_trial_count,
        evaluation_trial_count,
    )
    return pair_records, doa_records, covariance_record


def default_spherical_configurations() -> tuple[GCCStatisticalConfig, ...]:
    """Return the requested known-range spherical/plane comparison cases."""

    return tuple(
        GCCStatisticalConfig(
            "spherical_model",
            "deterministic_multisine",
            geometry,
            45.0,
            30.0,
            20.0,
            2048,
            propagation_model="spherical",
            distance_m=distance,
        )
        for geometry in ("square", "tetrahedral")
        for distance in (5.0, 10.0, 20.0, 50.0)
    )


def run_spherical_model_configuration(
    config: GCCStatisticalConfig,
    configuration_index: int,
    *,
    calibration_trial_count: int = CALIBRATION_TRIAL_COUNT,
    evaluation_trial_count: int = EVALUATION_TRIAL_COUNT,
    sampling_rate_hz: float = 48_000.0,
    interpolation_factor: int = 2,
    far_field_angular_step_deg: float = 0.5,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Separate GCC measurement error from known-range spherical model bias."""

    if config.propagation_model != "spherical":
        raise ValueError("spherical model study requires propagation_model='spherical'")
    calibration_seed = _split_seed(configuration_index, 0)
    evaluation_seed = _split_seed(configuration_index, 1)
    calibration, true_spherical, pairs = simulate_gcc_trial_set(
        config,
        calibration_trial_count,
        calibration_seed,
        sampling_rate_hz=sampling_rate_hz,
        interpolation_factor=interpolation_factor,
    )
    evaluation, evaluation_truth, evaluation_pairs = simulate_gcc_trial_set(
        config,
        evaluation_trial_count,
        evaluation_seed,
        sampling_rate_hz=sampling_rate_hz,
        interpolation_factor=interpolation_factor,
    )
    np.testing.assert_allclose(evaluation_truth, true_spherical, rtol=0.0, atol=2e-18)
    if evaluation_pairs != pairs:
        raise RuntimeError("calibration/evaluation pair sets differ")
    statistics = _calibration_statistics(calibration, true_spherical, sampling_rate_hz)
    pair_records = _pair_records(
        config,
        "calibration",
        calibration,
        true_spherical,
        pairs,
        calibration_trial_count,
        evaluation_trial_count,
        sampling_rate_hz,
    ) + _pair_records(
        config,
        "evaluation",
        evaluation,
        true_spherical,
        pairs,
        calibration_trial_count,
        evaluation_trial_count,
        sampling_rate_hz,
    )
    covariance_record = _covariance_record(
        config,
        calibration,
        true_spherical,
        pairs,
        statistics,
        calibration_trial_count,
        evaluation_trial_count,
    )
    positions = comparison_arrays()[config.geometry]
    true_angles = np.deg2rad([config.azimuth_deg, config.elevation_deg])
    true_plane = far_field_tdoa(*true_angles, positions, pairs)
    model_mismatch = true_spherical - true_plane
    calibration_covariance = np.asarray(statistics["covariance"])
    estimates = {
        "spherical_signal_plane_model": np.full((evaluation_trial_count, 2), np.nan),
        "spherical_signal_exact_known_range": np.full(
            (evaluation_trial_count, 2), np.nan
        ),
    }
    valid = {name: np.zeros(evaluation_trial_count, dtype=bool) for name in estimates}
    for trial in range(evaluation_trial_count):
        observed = evaluation.estimated_tdoa[trial]
        if np.any(evaluation.invalid[trial]) or not np.all(np.isfinite(observed)):
            continue
        plane = estimate_doa_wls(
            observed,
            positions,
            pairs,
            tdoa_covariance=calibration_covariance,
        )
        exact = estimate_doa_spherical_wls(
            observed,
            positions,
            config.distance_m,
            pairs,
            tdoa_covariance=calibration_covariance,
        )
        estimates["spherical_signal_plane_model"][trial] = [plane.phi, plane.elevation]
        estimates["spherical_signal_exact_known_range"][trial] = [
            exact.phi,
            exact.elevation,
        ]
        valid["spherical_signal_plane_model"][trial] = plane.success
        valid["spherical_signal_exact_known_range"][trial] = exact.success
    noiseless_plane = estimate_doa_wls(
        true_spherical, positions, pairs, sigma_tdoa=1e-6
    )
    noiseless_exact = estimate_doa_spherical_wls(
        true_spherical, positions, config.distance_m, pairs, sigma_tdoa=1e-6
    )
    truth_direction = direction_vector(*true_angles)
    noiseless_plane_bias = np.rad2deg(
        np.arccos(np.clip(noiseless_plane.direction @ truth_direction, -1.0, 1.0))
    )
    noiseless_exact_bias = np.rad2deg(
        np.arccos(np.clip(noiseless_exact.direction @ truth_direction, -1.0, 1.0))
    )
    measurement_errors = evaluation.estimated_tdoa - true_spherical
    measurement_valid = np.isfinite(measurement_errors) & ~evaluation.invalid
    measurement_rmse = float(
        np.sqrt(np.mean(measurement_errors[measurement_valid] ** 2)) * 1e6
    )
    global_error = far_field_error(
        positions,
        config.distance_m,
        grid=direction_grid(
            far_field_angular_step_deg, far_field_angular_step_deg
        ),
    ).max_plane_error_s
    any_invalid = np.any(evaluation.invalid, axis=1)
    any_boundary = np.any(evaluation.boundary_hit, axis=1)
    doa_records = []
    for variant in estimates:
        metrics = _direction_metrics(estimates[variant], valid[variant], true_angles)
        doa_records.append(
            {
                **_context(
                    config,
                    "evaluation",
                    "all_6",
                    calibration_trial_count,
                    evaluation_trial_count,
                    evaluation.seed,
                    variant,
                ),
                **metrics,
                "coverage": metrics["successful_fraction"],
                "failure_fraction": float(np.mean(~valid[variant])),
                "failure_invalid_fraction": float(np.mean(~valid[variant])),
                "gcc_any_pair_invalid_fraction": float(np.mean(any_invalid)),
                "gcc_any_pair_boundary_fraction": float(np.mean(any_boundary)),
                "gcc_measurement_rmse_us": measurement_rmse,
                "direction_specific_plane_model_bias_rms_us": float(
                    np.sqrt(np.mean(model_mismatch**2)) * 1e6
                ),
                "direction_specific_plane_model_bias_max_us": float(
                    np.max(np.abs(model_mismatch)) * 1e6
                ),
                "global_E_tau_us": global_error * 1e6,
                "far_field_grid_step_deg": far_field_angular_step_deg,
                "noiseless_plane_doa_bias_deg": float(noiseless_plane_bias),
                "noiseless_exact_doa_bias_deg": float(noiseless_exact_bias),
                "known_range_used": variant == "spherical_signal_exact_known_range",
                "calibration_covariance_used": True,
                "metric_conditioning": "successful_trials_only",
                "exact_crlb_claimed": False,
            }
        )
    return pair_records, doa_records, covariance_record


def default_statistical_configurations() -> tuple[GCCStatisticalConfig, ...]:
    """Return non-overlapping fine-SNR, signal, direction, and frame studies."""

    configs: list[GCCStatisticalConfig] = []
    for geometry in ("square", "tetrahedral"):
        for snr in FINE_SNR_LEVELS_DB:
            configs.append(
                GCCStatisticalConfig(
                    "fine_snr",
                    "deterministic_multisine",
                    geometry,
                    45.0,
                    30.0,
                    snr,
                    2048,
                )
            )
        for signal_model in SIGNAL_MODELS:
            for snr in (-10.0, 0.0, 10.0):
                configs.append(
                    GCCStatisticalConfig(
                        "signal_comparison",
                        signal_model,
                        geometry,
                        45.0,
                        30.0,
                        snr,
                        2048,
                    )
                )
        for azimuth, elevation in ((20.0, 10.0), (45.0, 30.0), (120.0, 50.0)):
            configs.append(
                GCCStatisticalConfig(
                    "direction_check",
                    "deterministic_multisine",
                    geometry,
                    azimuth,
                    elevation,
                    0.0,
                    2048,
                )
            )
    for frame_length in FRAME_LENGTHS:
        for snr in FRAME_SNR_LEVELS_DB:
            configs.append(
                GCCStatisticalConfig(
                    "frame_length",
                    "deterministic_multisine",
                    "tetrahedral",
                    45.0,
                    30.0,
                    snr,
                    frame_length,
                )
            )
    return tuple(configs)


def run_gcc_statistical_validation(
    *,
    configurations: tuple[GCCStatisticalConfig, ...] | None = None,
    calibration_trial_count: int = CALIBRATION_TRIAL_COUNT,
    evaluation_trial_count: int = EVALUATION_TRIAL_COUNT,
    interpolation_factor: int = 2,
    pair_output_csv: str | Path = "results/gcc_pair_error_summary.csv",
    doa_output_csv: str | Path = "results/gcc_doa_summary.csv",
    covariance_output_csv: str | Path = "results/gcc_covariance_summary.csv",
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Run the configured suite and write aggregate-only CSV artifacts."""

    selected = default_statistical_configurations() if configurations is None else configurations
    pair_records: list[dict[str, object]] = []
    doa_records: list[dict[str, object]] = []
    covariance_records: list[dict[str, object]] = []
    for index, config in enumerate(selected):
        pair_group, doa_group, covariance = run_gcc_statistical_configuration(
            config,
            index,
            calibration_trial_count=calibration_trial_count,
            evaluation_trial_count=evaluation_trial_count,
            interpolation_factor=interpolation_factor,
        )
        pair_records.extend(pair_group)
        doa_records.extend(doa_group)
        covariance_records.append(covariance)
    _write_records(pair_records, pair_output_csv)
    _write_records(doa_records, doa_output_csv)
    _write_records(covariance_records, covariance_output_csv)
    return pair_records, doa_records, covariance_records


def run_complete_gcc_statistical_validation(
    *,
    configurations: tuple[GCCStatisticalConfig, ...] | None = None,
    spherical_configurations: tuple[GCCStatisticalConfig, ...] | None = None,
    calibration_trial_count: int = CALIBRATION_TRIAL_COUNT,
    evaluation_trial_count: int = EVALUATION_TRIAL_COUNT,
    interpolation_factor: int = 2,
    far_field_angular_step_deg: float = 0.5,
    pair_output_csv: str | Path = "results/gcc_pair_error_summary.csv",
    doa_output_csv: str | Path = "results/gcc_doa_summary.csv",
    covariance_output_csv: str | Path = "results/gcc_covariance_summary.csv",
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Run plane statistical suites plus the requested spherical model study."""

    selected = default_statistical_configurations() if configurations is None else configurations
    selected_spherical = (
        default_spherical_configurations()
        if spherical_configurations is None
        else spherical_configurations
    )
    pair_records: list[dict[str, object]] = []
    doa_records: list[dict[str, object]] = []
    covariance_records: list[dict[str, object]] = []
    for index, config in enumerate(selected):
        pair_group, doa_group, covariance = run_gcc_statistical_configuration(
            config,
            index,
            calibration_trial_count=calibration_trial_count,
            evaluation_trial_count=evaluation_trial_count,
            interpolation_factor=interpolation_factor,
        )
        pair_records.extend(pair_group)
        doa_records.extend(doa_group)
        covariance_records.append(covariance)
    offset = len(selected)
    for local_index, config in enumerate(selected_spherical):
        pair_group, doa_group, covariance = run_spherical_model_configuration(
            config,
            offset + local_index,
            calibration_trial_count=calibration_trial_count,
            evaluation_trial_count=evaluation_trial_count,
            interpolation_factor=interpolation_factor,
            far_field_angular_step_deg=far_field_angular_step_deg,
        )
        pair_records.extend(pair_group)
        doa_records.extend(doa_group)
        covariance_records.append(covariance)
    _write_records(pair_records, pair_output_csv)
    _write_records(doa_records, doa_output_csv)
    _write_records(covariance_records, covariance_output_csv)
    return pair_records, doa_records, covariance_records


__all__ = [
    "CALIBRATION_TRIAL_COUNT",
    "DOA_VARIANTS",
    "EVALUATION_TRIAL_COUNT",
    "FINE_SNR_LEVELS_DB",
    "FRAME_LENGTHS",
    "FRAME_SNR_LEVELS_DB",
    "GCCStatisticalConfig",
    "GCCTrialSet",
    "RISK_COVERAGE_PERCENTILES",
    "SIGNAL_MODELS",
    "default_statistical_configurations",
    "default_spherical_configurations",
    "run_complete_gcc_statistical_validation",
    "run_gcc_statistical_configuration",
    "run_gcc_statistical_validation",
    "run_spherical_model_configuration",
    "simulate_gcc_trial_set",
]
