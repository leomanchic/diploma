"""Paired frame-wise DOA study for retarded-time subsonic source motion.

This module produces independent per-frame bearings; it does not implement a
trajectory filter or tracking. Moving and matched-static cases share the same
source waveform and additive-noise realization (common random numbers). The
propagation model is an exact retarded-time kinematic model in a homogeneous
stationary medium, not a complete environmental acoustics model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import tukey

from estimators.gcc_phat import estimate_tdoas_gcc_phat
from estimators.wls_doa import estimate_doa_wls
from model.geometry import (
    DEFAULT_SOUND_SPEED,
    Pair,
    all_pairs,
    array_centroid,
    baselines,
    comparison_arrays,
    direction_angles,
    direction_vector,
)
from simulation.fractional_delay import DEFAULT_FIR_LENGTH
from simulation.moving_source import (
    centroid_emission_time,
    retarded_time_doppler_factor,
    simulate_moving_source,
)
from simulation.signals import random_bandlimited_signal
from simulation.trajectory import ConstantVelocityTrajectory, StationaryTrajectory
from validation.gcc_statistical import _write_records
from validation.srp_statistical import _srp_from_gcc_diagnostics


MOVING_STUDY_BASE_SEED = 20260830
MOVING_STUDY_TRIAL_COUNT = 20
MOVING_STUDY_SPEEDS_MPS = (0.0, 5.0, 10.0, 20.0, 30.0)
MOVING_STUDY_DISTANCES_M = (10.0, 25.0, 50.0)
MOVING_STUDY_FRAME_LENGTHS = (256, 512, 1024, 2048)
MOVING_STUDY_MOTIONS = ("approach", "recede", "transverse")
MOVING_STUDY_GEOMETRIES = ("square", "tetrahedral")
MOVING_STUDY_SIGNALS = ("random_broadband", "deterministic_multisine")
MOVING_STUDY_SNR_DB = (-6.0, 5.0, 20.0)
MOVING_STUDY_METHODS = (
    "reference_3_gcc_wls",
    "all_6_equal_gcc_wls",
    "equal_weight_srp_phat",
)
MOVING_STUDY_DIRECTION_DEG = (45.0, 30.0)


@dataclass(frozen=True)
class MovingStudyConfig:
    geometry: str
    motion: str
    speed_mps: float
    distance_m: float
    frame_length: int
    signal_model: str
    snr_db: float

    @property
    def name(self) -> str:
        return (
            f"{self.geometry}_{self.motion}_v{self.speed_mps:g}_R{self.distance_m:g}_"
            f"N{self.frame_length}_{self.signal_model}_snr{self.snr_db:g}"
        )


@dataclass(frozen=True)
class FrameTruth:
    reception_time_s: float
    emission_time_s: float
    source_position_m: NDArray[np.float64]
    direction: NDArray[np.float64]
    azimuth_rad: float
    elevation_rad: float


def default_moving_study_configurations() -> tuple[MovingStudyConfig, ...]:
    return tuple(
        MovingStudyConfig(geometry, motion, speed, distance, length, signal, snr)
        for geometry in MOVING_STUDY_GEOMETRIES
        for motion in MOVING_STUDY_MOTIONS
        for speed in MOVING_STUDY_SPEEDS_MPS
        for distance in MOVING_STUDY_DISTANCES_M
        for length in MOVING_STUDY_FRAME_LENGTHS
        for signal in MOVING_STUDY_SIGNALS
        for snr in MOVING_STUDY_SNR_DB
    )


def _configuration_seed(index: int, stream: int) -> int:
    sequence = np.random.SeedSequence([MOVING_STUDY_BASE_SEED, int(index), int(stream)])
    return int(sequence.generate_state(1)[0])


def _clean_signal_seed(config: MovingStudyConfig) -> int:
    """Seed the clean waveform by all factors except the requested SNR."""

    factors = (
        MOVING_STUDY_GEOMETRIES.index(config.geometry),
        MOVING_STUDY_MOTIONS.index(config.motion),
        MOVING_STUDY_SPEEDS_MPS.index(float(config.speed_mps)),
        MOVING_STUDY_DISTANCES_M.index(float(config.distance_m)),
        MOVING_STUDY_FRAME_LENGTHS.index(int(config.frame_length)),
        MOVING_STUDY_SIGNALS.index(config.signal_model),
    )
    sequence = np.random.SeedSequence([MOVING_STUDY_BASE_SEED, 0, *factors])
    return int(sequence.generate_state(1)[0])


def _motion_velocity(direction: NDArray[np.float64], motion: str, speed: float) -> NDArray[np.float64]:
    if motion == "approach":
        return -speed * direction
    if motion == "recede":
        return speed * direction
    if motion == "transverse":
        tangent = np.cross(np.array([0.0, 0.0, 1.0]), direction)
        norm = float(np.linalg.norm(tangent))
        if norm == 0.0:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent /= norm
        return speed * tangent
    raise ValueError("motion must be approach, recede, or transverse")


def trajectory_for_configuration(
    config: MovingStudyConfig,
    positions: NDArray[np.float64],
) -> ConstantVelocityTrajectory:
    phi, elevation = np.deg2rad(MOVING_STUDY_DIRECTION_DEG)
    direction = direction_vector(phi, elevation)
    centroid = array_centroid(positions)
    source_at_center = centroid + config.distance_m * direction
    velocity = _motion_velocity(direction, config.motion, float(config.speed_mps))
    return ConstantVelocityTrajectory(source_at_center, velocity)


def frame_truth_at_reception(
    reception_time_s: float,
    positions: NDArray[np.float64],
    trajectory: ConstantVelocityTrajectory | StationaryTrajectory,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> FrameTruth:
    """Truth at centroid retarded time, never at uncorrected reception time."""

    reception = float(reception_time_s)
    emission = float(
        centroid_emission_time(reception, positions, trajectory, sound_speed)
    )
    source = trajectory.q(emission)
    displacement = source - array_centroid(positions)
    direction = displacement / np.linalg.norm(displacement)
    phi, elevation = direction_angles(direction)
    return FrameTruth(reception, emission, source, direction, phi, elevation)


def _source_signal(
    model: str,
    sample_count: int,
    sampling_rate_hz: float,
    source_times_s: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    if model == "random_broadband":
        return random_bandlimited_signal(
            sampling_rate_hz,
            sample_count,
            rng,
            minimum_frequency_hz=300.0,
            maximum_frequency_hz=10_000.0,
            taper_fraction=0.12,
        )
    if model == "deterministic_multisine":
        frequencies = np.linspace(300.0, 10_000.0, 43)[1:-1]
        phases = 2.0 * np.pi * np.mod(
            np.arange(frequencies.size) * (np.sqrt(5.0) - 1.0) / 2.0, 1.0
        )
        amplitudes = 1.0 / np.sqrt(frequencies / frequencies[0])
        signal = np.sum(
            amplitudes[:, None]
            * np.cos(2.0 * np.pi * frequencies[:, None] * source_times_s + phases[:, None]),
            axis=0,
        )
        signal *= tukey(sample_count, alpha=0.12)
        return signal / np.sqrt(np.mean(signal**2))
    raise ValueError("unknown signal model")


def _clean_matched_frames(
    config: MovingStudyConfig,
    positions: NDArray[np.float64],
    sampling_rate_hz: float,
    rng: np.random.Generator,
):
    guard = max(2 * DEFAULT_FIR_LENGTH, 256)
    source_count = int(config.frame_length + 2 * guard)
    source_times = (
        np.arange(source_count, dtype=float) - 0.5 * (source_count - 1)
    ) / sampling_rate_hz
    source = _source_signal(
        config.signal_model, source_count, sampling_rate_hz, source_times, rng
    )
    moving_trajectory = trajectory_for_configuration(config, positions)
    source_center = moving_trajectory.q(0.0)
    static_trajectory = StationaryTrajectory(source_center)
    center_reception = config.distance_m / DEFAULT_SOUND_SPEED
    reception = center_reception + (
        np.arange(config.frame_length, dtype=float) - 0.5 * (config.frame_length - 1)
    ) / sampling_rate_hz
    common = dict(
        source_start_time_s=float(source_times[0]),
        reception_times_s=reception,
        pairs=all_pairs(4),
        fir_length=DEFAULT_FIR_LENGTH,
    )
    moving = simulate_moving_source(
        source, sampling_rate_hz, positions, moving_trajectory, **common
    )
    static = simulate_moving_source(
        source, sampling_rate_hz, positions, static_trajectory, **common
    )
    if moving.valid_region != (0, config.frame_length) or static.valid_region != (
        0,
        config.frame_length,
    ):
        raise RuntimeError("source guard is insufficient for a full valid estimator frame")
    return moving, static, moving_trajectory, static_trajectory


def _delay_bounds(positions: NDArray[np.float64], pairs: tuple[Pair, ...], fs: float):
    return np.linalg.norm(baselines(positions, pairs), axis=1) / DEFAULT_SOUND_SPEED + 2.0 / fs


def gcc_boundary_flags(diagnostics) -> dict[str, bool]:
    """Aggregate GCC boundary flags over the pairs actually used by each WLS.

    The shared frontend computes all six pairs.  Reference-3 subsequently uses
    pair indices 0, 1, 2 (microphone 0 against microphones 1, 2, 3), whereas
    all-6 uses every pair.
    """

    flags = np.asarray([bool(item.boundary_hit) for item in diagnostics], dtype=bool)
    if flags.shape != (6,):
        raise ValueError("moving-source GCC frontend must report all six pair flags")
    return {
        "reference_3_gcc_wls": bool(np.any(flags[:3])),
        "all_6_equal_gcc_wls": bool(np.any(flags)),
    }


def estimate_independent_frame(
    channels: NDArray[np.float64],
    positions: NDArray[np.float64],
    sampling_rate_hz: float,
) -> dict[str, dict[str, object]]:
    pairs = all_pairs(4)
    started = perf_counter()
    observed, diagnostics = estimate_tdoas_gcc_phat(
        channels,
        sampling_rate_hz,
        pairs,
        maximum_delay_seconds=_delay_bounds(positions, pairs, sampling_rate_hz),
        interpolation_factor=2,
        minimum_frequency_hz=200.0,
        maximum_frequency_hz=10_000.0,
        relative_spectral_floor=1e-8,
    )
    gcc_runtime = perf_counter() - started
    finite = np.isfinite(observed) & ~np.asarray([item.invalid for item in diagnostics])
    method_boundary = gcc_boundary_flags(diagnostics)
    results: dict[str, dict[str, object]] = {}
    for method, indices in (
        ("reference_3_gcc_wls", np.array([0, 1, 2])),
        ("all_6_equal_gcc_wls", np.arange(6)),
    ):
        started = perf_counter()
        valid = bool(np.all(finite[indices]))
        direction = np.full(3, np.nan)
        if valid:
            estimate = estimate_doa_wls(
                observed[indices],
                positions,
                tuple(pairs[index] for index in indices),
                sigma_tdoa=1.0 / sampling_rate_hz,
            )
            valid = bool(estimate.success and np.all(np.isfinite(estimate.direction)))
            if valid:
                direction = estimate.direction
        backend_runtime = perf_counter() - started
        results[method] = {
            "direction": direction,
            "valid": valid,
            "boundary": method_boundary[method],
            "shared_gcc_frontend_runtime_s": gcc_runtime,
            "estimator_backend_runtime_s": backend_runtime,
            "total_runtime_s": gcc_runtime + backend_runtime,
            "gcc_frontend_pair_count": 6,
            "estimator_backend_pair_count": int(indices.size),
        }
    srp = _srp_from_gcc_diagnostics(
        diagnostics, positions, pairs, sampling_rate_hz, sample_count=channels.shape[1]
    )
    results["equal_weight_srp_phat"] = {
        "direction": srp.direction,
        "valid": not srp.invalid and np.all(np.isfinite(srp.direction)),
        "boundary": srp.boundary_hit,
        "shared_gcc_frontend_runtime_s": gcc_runtime,
        "estimator_backend_runtime_s": srp.runtime_seconds,
        "total_runtime_s": gcc_runtime + srp.runtime_seconds,
        "gcc_frontend_pair_count": 6,
        "estimator_backend_pair_count": 6,
    }
    return results


def _geodesic_errors_deg(
    directions: NDArray[np.float64], valid: NDArray[np.bool_], truth: NDArray[np.float64]
) -> NDArray[np.float64]:
    selected = directions[valid]
    if selected.size == 0:
        return np.empty(0)
    return np.rad2deg(np.arccos(np.clip(selected @ truth, -1.0, 1.0)))


def _along_track_lag_deg(
    estimates: NDArray[np.float64],
    valid: NDArray[np.bool_],
    truth: NDArray[np.float64],
    velocity: NDArray[np.float64],
) -> NDArray[np.float64]:
    tangent_velocity = velocity - truth * float(truth @ velocity)
    tangent_norm = float(np.linalg.norm(tangent_velocity))
    if tangent_norm < 1e-10 * max(float(np.linalg.norm(velocity)), 1.0):
        return np.full(np.count_nonzero(valid), np.nan)
    tangent = tangent_velocity / tangent_norm
    selected = estimates[valid]
    cosine = np.clip(selected @ truth, -1.0, 1.0)
    angle = np.arccos(cosine)
    denominator = np.sin(angle)
    log_vectors = np.zeros_like(selected)
    nonzero = denominator > 1e-12
    log_vectors[nonzero] = (
        angle[nonzero, None]
        * (selected[nonzero] - cosine[nonzero, None] * truth)
        / denominator[nonzero, None]
    )
    return np.rad2deg(log_vectors @ tangent)


def _conditional_metrics(errors: NDArray[np.float64]) -> dict[str, float]:
    if errors.size == 0:
        return {
            "conditional_rmse_deg": float("nan"),
            "conditional_median_deg": float("nan"),
            "conditional_p95_deg": float("nan"),
            "conditional_p99_deg": float("nan"),
        }
    return {
        "conditional_rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "conditional_median_deg": float(np.median(errors)),
        "conditional_p95_deg": float(np.percentile(errors, 95.0)),
        "conditional_p99_deg": float(np.percentile(errors, 99.0)),
    }


def _frame_diagnostics(moving, positions, trajectory) -> dict[str, float]:
    first_truth = frame_truth_at_reception(
        moving.reception_times_s[0], positions, trajectory
    )
    last_truth = frame_truth_at_reception(
        moving.reception_times_s[-1], positions, trajectory
    )
    cosine = float(np.clip(first_truth.direction @ last_truth.direction, -1.0, 1.0))
    doa_change = 0.0 if 1.0 - cosine < 32 * np.finfo(float).eps else np.rad2deg(np.arccos(cosine))
    center = moving.reception_times_s[moving.reception_times_s.size // 2]
    center_emission = float(centroid_emission_time(center, positions, trajectory))
    return {
        "doa_change_within_frame_deg": float(doa_change),
        "max_tdoa_change_within_frame_us": float(
            np.max(np.ptp(moving.tdoa_seconds, axis=1)) * 1e6
        ),
        "doppler_factor": float(
            retarded_time_doppler_factor(
                center_emission, array_centroid(positions), trajectory
            )
        ),
    }


def run_moving_configuration(
    config: MovingStudyConfig,
    configuration_index: int,
    *,
    trial_count: int = MOVING_STUDY_TRIAL_COUNT,
    sampling_rate_hz: float = 48_000.0,
) -> list[dict[str, object]]:
    """Run one paired moving/static Monte Carlo configuration."""

    count = int(trial_count)
    if count < 1:
        raise ValueError("trial_count must be positive")
    positions = comparison_arrays()[config.geometry]
    signal_seed = _clean_signal_seed(config)
    noise_seed = _configuration_seed(configuration_index, 1)
    moving, static, moving_trajectory, _ = _clean_matched_frames(
        config, positions, sampling_rate_hz, np.random.default_rng(signal_seed)
    )
    center_index = config.frame_length // 2
    truth = frame_truth_at_reception(
        moving.reception_times_s[center_index], positions, moving_trajectory
    )
    # With an even frame the chosen sample is half a sample after the geometric
    # center; truth is intentionally evaluated at its own centroid emission time.
    diagnostics = _frame_diagnostics(moving, positions, moving_trajectory)
    noise_rng = np.random.default_rng(noise_seed)
    moving_clean_rms = float(np.sqrt(np.mean(moving.channels**2)))
    static_clean_rms = float(np.sqrt(np.mean(static.channels**2)))
    noise_scale = static_clean_rms / (
        10.0 ** (config.snr_db / 20.0)
    )
    moving_directions = {name: np.full((count, 3), np.nan) for name in MOVING_STUDY_METHODS}
    static_directions = {name: np.full((count, 3), np.nan) for name in MOVING_STUDY_METHODS}
    moving_valid = {name: np.zeros(count, dtype=bool) for name in MOVING_STUDY_METHODS}
    static_valid = {name: np.zeros(count, dtype=bool) for name in MOVING_STUDY_METHODS}
    moving_boundary = {name: np.zeros(count, dtype=bool) for name in MOVING_STUDY_METHODS}
    static_boundary = {name: np.zeros(count, dtype=bool) for name in MOVING_STUDY_METHODS}
    runtime_components = (
        "shared_gcc_frontend_runtime_s",
        "estimator_backend_runtime_s",
        "total_runtime_s",
    )
    moving_runtime = {
        component: {name: np.zeros(count) for name in MOVING_STUDY_METHODS}
        for component in runtime_components
    }
    static_runtime = {
        component: {name: np.zeros(count) for name in MOVING_STUDY_METHODS}
        for component in runtime_components
    }
    effective_moving_snr_db = np.zeros(count)
    effective_static_snr_db = np.zeros(count)
    for trial in range(count):
        noise = noise_rng.normal(0.0, noise_scale, size=moving.channels.shape)
        realized_noise_rms = float(np.sqrt(np.mean(noise**2)))
        effective_moving_snr_db[trial] = 20.0 * np.log10(
            moving_clean_rms / realized_noise_rms
        )
        effective_static_snr_db[trial] = 20.0 * np.log10(
            static_clean_rms / realized_noise_rms
        )
        paired = (
            ("moving", moving.channels + noise),
            ("static", static.channels + noise),
        )
        for label, channels in paired:
            estimates = estimate_independent_frame(channels, positions, sampling_rate_hz)
            directions = moving_directions if label == "moving" else static_directions
            valid = moving_valid if label == "moving" else static_valid
            boundary = moving_boundary if label == "moving" else static_boundary
            runtime = moving_runtime if label == "moving" else static_runtime
            for method, result in estimates.items():
                directions[method][trial] = result["direction"]
                valid[method][trial] = result["valid"]
                boundary[method][trial] = result["boundary"]
                for component in runtime_components:
                    runtime[component][method][trial] = result[component]

    rows: list[dict[str, object]] = []
    for method in MOVING_STUDY_METHODS:
        moving_errors = _geodesic_errors_deg(
            moving_directions[method], moving_valid[method], truth.direction
        )
        static_errors = _geodesic_errors_deg(
            static_directions[method], static_valid[method], truth.direction
        )
        moving_metrics = _conditional_metrics(moving_errors)
        static_metrics = _conditional_metrics(static_errors)
        lags = _along_track_lag_deg(
            moving_directions[method],
            moving_valid[method],
            truth.direction,
            moving_trajectory.v(truth.emission_time_s),
        )
        rows.append(
            {
                "study_type": "paired_framewise_moving_source",
                "configuration": config.name,
                "geometry": config.geometry,
                "motion": config.motion,
                "speed_mps": config.speed_mps,
                "distance_m": config.distance_m,
                "frame_length": config.frame_length,
                "frame_duration_ms": 1000.0 * config.frame_length / sampling_rate_hz,
                "signal_model": config.signal_model,
                "snr_db": config.snr_db,
                "nominal_snr_db": config.snr_db,
                "nominal_moving_snr_db": config.snr_db,
                "nominal_static_snr_db": config.snr_db,
                "expected_effective_moving_snr_db": float(
                    20.0 * np.log10(moving_clean_rms / noise_scale)
                ),
                "expected_effective_static_snr_db": float(
                    20.0 * np.log10(static_clean_rms / noise_scale)
                ),
                "mean_effective_moving_snr_db": float(np.mean(effective_moving_snr_db)),
                "mean_effective_static_snr_db": float(np.mean(effective_static_snr_db)),
                "effective_snr_definition": "20log10(full_frame_clean_rms/full_frame_realized_noise_rms)",
                "noise_standard_deviation": noise_scale,
                "estimator_variant": method,
                "trial_count": count,
                "signal_seed": signal_seed,
                "noise_seed": noise_seed,
                "clean_signal_seed_scope": "all_configuration_factors_except_snr",
                "common_random_numbers": True,
                "truth_definition": "array_centroid_emission_time_at_selected_reception_sample",
                "truth_reception_time_s": truth.reception_time_s,
                "truth_emission_time_s": truth.emission_time_s,
                "truth_azimuth_deg": np.rad2deg(truth.azimuth_rad) % 360.0,
                "truth_elevation_deg": np.rad2deg(truth.elevation_rad),
                "moving_conditional_rmse_deg": moving_metrics["conditional_rmse_deg"],
                "moving_conditional_median_deg": moving_metrics["conditional_median_deg"],
                "moving_conditional_p95_deg": moving_metrics["conditional_p95_deg"],
                "moving_conditional_p99_deg": moving_metrics["conditional_p99_deg"],
                "moving_successful_trial_count": int(np.count_nonzero(moving_valid[method])),
                "moving_unsuccessful_trial_count": int(np.count_nonzero(~moving_valid[method])),
                "moving_coverage": float(np.mean(moving_valid[method])),
                "moving_failure_fraction": float(np.mean(~moving_valid[method])),
                "moving_boundary_hit_fraction": float(np.mean(moving_boundary[method])),
                "static_conditional_rmse_deg": static_metrics["conditional_rmse_deg"],
                "static_conditional_median_deg": static_metrics["conditional_median_deg"],
                "static_conditional_p95_deg": static_metrics["conditional_p95_deg"],
                "static_conditional_p99_deg": static_metrics["conditional_p99_deg"],
                "static_successful_trial_count": int(np.count_nonzero(static_valid[method])),
                "static_unsuccessful_trial_count": int(np.count_nonzero(~static_valid[method])),
                "static_coverage": float(np.mean(static_valid[method])),
                "static_failure_fraction": float(np.mean(~static_valid[method])),
                "static_boundary_hit_fraction": float(np.mean(static_boundary[method])),
                "motion_induced_excess_rmse_deg": (
                    moving_metrics["conditional_rmse_deg"] - static_metrics["conditional_rmse_deg"]
                ),
                "conditional_mean_along_track_angular_lag_deg": (
                    float(np.nanmean(lags)) if np.any(np.isfinite(lags)) else None
                ),
                "angular_lag_definition": "signed_log_map_projection_on_instantaneous_doa_tangent",
                **diagnostics,
                "runtime_accounting": "shared_all_6_gcc_frontend_plus_estimator_backend",
                "gcc_frontend_pair_count": 6,
                "estimator_backend_pair_count": 3 if method == "reference_3_gcc_wls" else 6,
                "mean_moving_shared_gcc_frontend_runtime_s": float(
                    np.mean(moving_runtime["shared_gcc_frontend_runtime_s"][method])
                ),
                "mean_moving_estimator_backend_runtime_s": float(
                    np.mean(moving_runtime["estimator_backend_runtime_s"][method])
                ),
                "mean_moving_total_runtime_per_estimate_s": float(
                    np.mean(moving_runtime["total_runtime_s"][method])
                ),
                "mean_static_shared_gcc_frontend_runtime_s": float(
                    np.mean(static_runtime["shared_gcc_frontend_runtime_s"][method])
                ),
                "mean_static_estimator_backend_runtime_s": float(
                    np.mean(static_runtime["estimator_backend_runtime_s"][method])
                ),
                "mean_static_total_runtime_per_estimate_s": float(
                    np.mean(static_runtime["total_runtime_s"][method])
                ),
                "independent_framewise_doa_not_tracking": True,
            }
        )
    return rows


def run_deterministic_moving_gate() -> list[dict[str, object]]:
    """Noise-free high-distance gate required before any Monte Carlo."""

    config = MovingStudyConfig(
        "tetrahedral", "transverse", 20.0, 50.0, 1024, "deterministic_multisine", 120.0
    )
    rows = run_moving_configuration(config, 90_000, trial_count=1)
    if not all(row["moving_coverage"] == 1.0 for row in rows):
        raise RuntimeError("deterministic moving gate failed coverage")
    if max(float(row["moving_conditional_rmse_deg"]) for row in rows) > 1.0:
        raise RuntimeError("deterministic moving gate exceeded 1 degree")
    return rows


def run_moving_smoke_gate() -> list[dict[str, object]]:
    """Small seeded paired Monte Carlo gate before the full Cartesian study."""

    config = MovingStudyConfig(
        "tetrahedral", "transverse", 20.0, 25.0, 512, "random_broadband", 10.0
    )
    rows = run_moving_configuration(config, 90_001, trial_count=6)
    if not all(float(row["moving_coverage"]) >= 5.0 / 6.0 for row in rows):
        raise RuntimeError("moving smoke gate coverage below 5/6")
    if not all(np.isfinite(float(row["moving_conditional_rmse_deg"])) for row in rows):
        raise RuntimeError("moving smoke gate produced a non-finite RMSE")
    return rows


def run_moving_source_study(
    *,
    configurations: tuple[MovingStudyConfig, ...] | None = None,
    trial_count: int = MOVING_STUDY_TRIAL_COUNT,
    output_csv: str | Path = "results/moving_source_summary.csv",
    run_gates: bool = True,
) -> list[dict[str, object]]:
    """Run gates, then the requested paired Cartesian experiment."""

    if run_gates:
        run_deterministic_moving_gate()
        run_moving_smoke_gate()
    selected = (
        default_moving_study_configurations()
        if configurations is None
        else configurations
    )
    records: list[dict[str, object]] = []
    for index, config in enumerate(selected):
        records.extend(
            run_moving_configuration(config, index, trial_count=trial_count)
        )
    _write_records(records, output_csv)
    return records


__all__ = [
    "FrameTruth",
    "MOVING_STUDY_BASE_SEED",
    "MOVING_STUDY_DIRECTION_DEG",
    "MOVING_STUDY_DISTANCES_M",
    "MOVING_STUDY_FRAME_LENGTHS",
    "MOVING_STUDY_GEOMETRIES",
    "MOVING_STUDY_METHODS",
    "MOVING_STUDY_MOTIONS",
    "MOVING_STUDY_SIGNALS",
    "MOVING_STUDY_SNR_DB",
    "MOVING_STUDY_SPEEDS_MPS",
    "MOVING_STUDY_TRIAL_COUNT",
    "MovingStudyConfig",
    "default_moving_study_configurations",
    "estimate_independent_frame",
    "gcc_boundary_flags",
    "frame_truth_at_reception",
    "run_deterministic_moving_gate",
    "run_moving_configuration",
    "run_moving_smoke_gate",
    "run_moving_source_study",
    "trajectory_for_configuration",
]
