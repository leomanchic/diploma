"""Paired GCC/WLS versus equal-weight far-field SRP-PHAT validation.

All estimators in one evaluation trial receive exactly the same noisy channel
matrix (common random numbers).  Calibration and evaluation use independent
deterministic ``SeedSequence`` children.  The full study uses a fast SRP score
obtained by cubic interpolation of the same all-pair GCC-PHAT correlations;
several trials per configuration are independently checked against the exact
vectorized frequency-domain SRP implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from estimators.cycle_projection import project_tdoa_cycles
from estimators.gcc_phat import GCCPHATResult, estimate_tdoas_gcc_phat
from estimators.srp_phat import SRPPHATResult, _local_score_curvature, srp_phat
from estimators.wls_doa import estimate_doa_wls
from model.geometry import (
    DEFAULT_SOUND_SPEED,
    Pair,
    all_pairs,
    baselines,
    comparison_arrays,
    direction_angles,
    direction_vector,
)
from simulation.fractional_delay import DEFAULT_FIR_LENGTH
from validation.gcc_statistical import (
    FINE_SNR_LEVELS_DB,
    GCCStatisticalConfig,
    GCCTrialSet,
    SIGNAL_MODELS,
    _calibration_statistics,
    _clean_frame,
    _confidence_weights,
    _direction_metrics,
    _graph_connected,
    _write_records,
)


SRP_STATISTICAL_BASE_SEED = 20260829
SRP_CALIBRATION_TRIAL_COUNT = 500
SRP_EVALUATION_TRIAL_COUNT = 1000
SRP_FRAME_LENGTH = 1024
SRP_DIRECTIONS_DEG = ((20.0, 10.0), (45.0, 30.0), (120.0, 50.0))
# At 10 kHz and D=0.20 m the main lobe is too narrow for the former 15-degree
# global grid: local refinement can then converge to a sampled sidelobe.
SRP_SEARCH_STEPS_DEG = (5.0, 1.0, 0.25)
SRP_EXACT_REFERENCE_TRIALS = 3
SRP_METHODS = (
    "reference_3_gcc_wls",
    "all_6_equal_gcc_wls",
    "all_6_calibrated_gcc_wls",
    "equal_weight_srp_phat",
)


@dataclass(frozen=True)
class SRPPairedTrialSet:
    gcc: GCCTrialSet
    srp_angles: NDArray[np.float64]
    srp_scores: NDArray[np.float64]
    srp_boundary_hit: NDArray[np.bool_]
    srp_invalid: NDArray[np.bool_]
    gcc_runtime_seconds: NDArray[np.float64]
    srp_search_runtime_seconds: NDArray[np.float64]
    exact_srp_runtime_seconds: NDArray[np.float64]
    exact_fast_disagreement_deg: NDArray[np.float64]


def _split_seed(configuration_index: int, split: int) -> int:
    sequence = np.random.SeedSequence(
        [SRP_STATISTICAL_BASE_SEED, int(configuration_index), int(split)]
    )
    return int(sequence.generate_state(1)[0])


def _bounds(positions: NDArray[np.float64], sampling_rate_hz: float) -> NDArray[np.float64]:
    pairs = all_pairs(len(positions))
    return np.asarray(
        [
            np.linalg.norm(positions[first] - positions[second])
            / DEFAULT_SOUND_SPEED
            + 2.0 / sampling_rate_hz
            for first, second in pairs
        ]
    )


def _grid(lower: float, upper: float, step: float, periodic: bool) -> NDArray[np.float64]:
    if periodic:
        count = int(np.ceil((upper - lower) / step))
        return lower + np.arange(count) * (upper - lower) / count
    values = np.arange(lower, upper + 0.25 * step, step)
    if values[-1] < upper - 1e-12:
        values = np.append(values, upper)
    values[-1] = min(values[-1], upper)
    return np.unique(values)


def _directions(
    azimuths: NDArray[np.float64], elevations: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    phi, elevation = np.meshgrid(azimuths, elevations, indexing="xy")
    return phi.ravel(), elevation.ravel(), direction_vector(phi.ravel(), elevation.ravel())


def _srp_from_gcc_diagnostics(
    diagnostics: tuple[GCCPHATResult, ...],
    positions: NDArray[np.float64],
    pairs: tuple[Pair, ...],
    sampling_rate_hz: float,
    *,
    sample_count: int | None = None,
    steps_deg: tuple[float, ...] = SRP_SEARCH_STEPS_DEG,
) -> SRPPHATResult:
    """Fast equal-pair SRP from cubic interpolation of GCC correlations."""

    started = perf_counter()
    if len(diagnostics) != len(pairs):
        raise ValueError("one GCC diagnostic is required per SRP pair")
    invalid = next((item for item in diagnostics if item.invalid), None)
    if invalid is not None:
        return SRPPHATResult(
            phi=float("nan"),
            elevation=float("nan"),
            direction=np.full(3, np.nan),
            score=float("nan"),
            boundary_hit=False,
            invalid=True,
            invalid_reason=invalid.invalid_reason,
            runtime_seconds=perf_counter() - started,
            coarse_candidate_count=0,
            fine_candidate_count=0,
            local_refinement_evaluations=0,
            local_refinement_success=False,
            used_spectral_energy=float(sum(item.used_spectral_energy for item in diagnostics)),
            mean_spectral_energy_fraction=float(
                np.mean([item.spectral_energy_fraction for item in diagnostics])
            ),
            used_bin_count=int(sum(item.used_bin_count for item in diagnostics)),
            pair_count=len(pairs),
            pairs=pairs,
            valid_region=(0, 0),
            search_azimuth_bounds_rad=(0.0, 2.0 * np.pi),
            search_elevation_bounds_rad=(0.0, np.pi / 2.0),
        )
    splines = [
        CubicSpline(item.lags_samples, item.correlation, extrapolate=False)
        for item in diagnostics
    ]
    pair_baselines = baselines(positions, pairs)

    def score(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        delays_samples = vectors @ pair_baselines.T / DEFAULT_SOUND_SPEED * sampling_rate_hz
        values = np.zeros(vectors.shape[0])
        for pair_index, spline in enumerate(splines):
            pair_values = spline(delays_samples[:, pair_index])
            if np.any(~np.isfinite(pair_values)):
                return np.full(vectors.shape[0], -np.inf)
            values += pair_values
        return values / len(pairs)

    steps = np.deg2rad(np.asarray(steps_deg, dtype=float))
    azimuths = _grid(0.0, 2.0 * np.pi, steps[0], True)
    elevations = _grid(0.0, np.pi / 2.0, steps[0], False)
    phi_grid, elevation_grid, vectors = _directions(azimuths, elevations)
    values = score(vectors)
    coarse_values = values.copy()
    best = int(np.argmax(values))
    best_phi = float(phi_grid[best])
    best_elevation = float(elevation_grid[best])
    best_score = float(values[best])
    coarse_count = int(values.size)
    fine_count = 0
    previous = steps[0]
    for step in steps[1:]:
        offsets = np.arange(-previous, previous + 0.25 * step, step)
        local_phi = np.unique((best_phi + offsets) % (2.0 * np.pi))
        local_elevation = np.unique(
            np.clip(best_elevation + offsets, 0.0, np.pi / 2.0)
        )
        phi_grid, elevation_grid, vectors = _directions(local_phi, local_elevation)
        values = score(vectors)
        fine_count += int(values.size)
        best = int(np.argmax(values))
        best_phi = float(phi_grid[best])
        best_elevation = float(elevation_grid[best])
        best_score = float(values[best])
        previous = step

    def objective(angles: NDArray[np.float64]) -> float:
        vector = direction_vector(float(angles[0]) % (2.0 * np.pi), float(angles[1]))[
            None, :
        ]
        return -float(score(vector)[0])

    optimized = minimize(
        objective,
        np.asarray([best_phi, best_elevation]),
        method="L-BFGS-B",
        bounds=(
            (best_phi - previous, best_phi + previous),
            (max(0.0, best_elevation - previous), min(np.pi / 2.0, best_elevation + previous)),
        ),
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 60},
    )
    if np.all(np.isfinite(optimized.x)) and np.isfinite(optimized.fun):
        candidate_score = -float(optimized.fun)
        if candidate_score >= best_score - 1e-12:
            best_phi = float(optimized.x[0]) % (2.0 * np.pi)
            best_elevation = float(optimized.x[1])
            best_score = candidate_score
    vector = direction_vector(best_phi, best_elevation)
    best_phi, best_elevation = direction_angles(vector)
    best_phi %= 2.0 * np.pi
    tolerance = float(steps[-1]) * 0.51
    second_score = (
        float(np.partition(coarse_values, -2)[-2])
        if coarse_values.size >= 2
        else float("nan")
    )
    local_hessian, curvature_eigenvalues = _local_score_curvature(
        score, best_phi, best_elevation
    )
    return SRPPHATResult(
        phi=best_phi,
        elevation=best_elevation,
        direction=vector,
        score=best_score,
        boundary_hit=best_elevation <= tolerance
        or best_elevation >= np.pi / 2.0 - tolerance,
        invalid=False,
        invalid_reason=None,
        runtime_seconds=perf_counter() - started,
        coarse_candidate_count=coarse_count,
        fine_candidate_count=fine_count,
        local_refinement_evaluations=int(optimized.nfev),
        local_refinement_success=bool(optimized.success),
        used_spectral_energy=float(sum(item.used_spectral_energy for item in diagnostics)),
        mean_spectral_energy_fraction=float(
            np.mean([item.spectral_energy_fraction for item in diagnostics])
        ),
        used_bin_count=int(sum(item.used_bin_count for item in diagnostics)),
        pair_count=len(pairs),
        pairs=pairs,
        valid_region=(0, int(sample_count))
        if sample_count is not None
        else (0, diagnostics[0].correlation.size),
        search_azimuth_bounds_rad=(0.0, 2.0 * np.pi),
        search_elevation_bounds_rad=(0.0, np.pi / 2.0),
        score_margin=float(best_score - second_score),
        local_negative_score_hessian=local_hessian,
        local_curvature_eigenvalues=curvature_eigenvalues,
    )


def _simulate_split(
    config: GCCStatisticalConfig,
    trial_count: int,
    seed: int,
    *,
    sampling_rate_hz: float,
    interpolation_factor: int,
    estimate_srp: bool,
    srp_backend: str,
    exact_reference_trials: int,
) -> tuple[SRPPairedTrialSet, NDArray[np.float64], tuple[Pair, ...]]:
    positions = comparison_arrays()[config.geometry]
    pairs = all_pairs(4)
    delay_bounds = _bounds(positions, sampling_rate_hz)
    rng = np.random.default_rng(seed)
    shape = (trial_count, len(pairs))
    estimated = np.full(shape, np.nan)
    peak_value = np.full(shape, np.nan)
    peak_ratio = np.full(shape, np.nan)
    peak_curvature = np.full(shape, np.nan)
    spectral_energy = np.full(shape, np.nan)
    spectral_fraction = np.full(shape, np.nan)
    boundary = np.zeros(shape, dtype=bool)
    invalid = np.zeros(shape, dtype=bool)
    srp_angles = np.full((trial_count, 2), np.nan)
    srp_scores = np.full(trial_count, np.nan)
    srp_boundary = np.zeros(trial_count, dtype=bool)
    srp_invalid = np.ones(trial_count, dtype=bool)
    gcc_runtime = np.zeros(trial_count)
    srp_runtime = np.zeros(trial_count)
    exact_runtime = []
    exact_disagreement = []
    true_tdoa: NDArray[np.float64] | None = None
    deterministic_clean: NDArray[np.float64] | None = None
    if config.signal_model == "deterministic_multisine":
        deterministic_clean, true_tdoa = _clean_frame(
            config, positions, pairs, sampling_rate_hz, rng
        )
    amplitude_ratio = 10.0 ** (config.snr_db / 20.0)
    for trial in range(trial_count):
        if deterministic_clean is None:
            clean, trial_truth = _clean_frame(
                config, positions, pairs, sampling_rate_hz, rng
            )
            if true_tdoa is None:
                true_tdoa = trial_truth
            else:
                np.testing.assert_allclose(trial_truth, true_tdoa, rtol=0.0, atol=2e-18)
        else:
            clean = deterministic_clean
        channel_rms = np.sqrt(np.mean(clean**2, axis=1))
        noisy = clean + rng.normal(
            0.0, channel_rms[:, None] / amplitude_ratio, size=clean.shape
        )
        started = perf_counter()
        delays, diagnostics = estimate_tdoas_gcc_phat(
            noisy,
            sampling_rate_hz,
            pairs,
            maximum_delay_seconds=delay_bounds,
            interpolation_factor=interpolation_factor,
            minimum_frequency_hz=200.0,
            maximum_frequency_hz=10_000.0,
            relative_spectral_floor=1e-8,
        )
        gcc_runtime[trial] = perf_counter() - started
        estimated[trial] = delays
        for pair_index, item in enumerate(diagnostics):
            peak_value[trial, pair_index] = item.peak_value
            peak_ratio[trial, pair_index] = item.peak_to_second_peak_ratio
            peak_curvature[trial, pair_index] = item.peak_curvature
            spectral_energy[trial, pair_index] = item.used_spectral_energy
            spectral_fraction[trial, pair_index] = item.spectral_energy_fraction
            boundary[trial, pair_index] = item.boundary_hit
            invalid[trial, pair_index] = item.invalid
        if estimate_srp:
            if srp_backend == "exact_vectorized":
                srp_result = srp_phat(
                    noisy,
                    sampling_rate_hz,
                    positions,
                    pairs=pairs,
                    coarse_to_fine_steps_deg=SRP_SEARCH_STEPS_DEG,
                    minimum_frequency_hz=200.0,
                    maximum_frequency_hz=10_000.0,
                    relative_spectral_floor=1e-8,
                )
            elif srp_backend == "gcc_correlation_interpolated":
                srp_result = _srp_from_gcc_diagnostics(
                    diagnostics,
                    positions,
                    pairs,
                    sampling_rate_hz,
                    sample_count=noisy.shape[1],
                )
            else:
                raise ValueError("unknown srp_backend")
            srp_runtime[trial] = srp_result.runtime_seconds
            srp_invalid[trial] = srp_result.invalid
            srp_boundary[trial] = srp_result.boundary_hit
            srp_scores[trial] = srp_result.score
            if not srp_result.invalid:
                srp_angles[trial] = [srp_result.phi, srp_result.elevation]
            if srp_backend == "gcc_correlation_interpolated" and trial < exact_reference_trials:
                exact = srp_phat(
                    noisy,
                    sampling_rate_hz,
                    positions,
                    pairs=pairs,
                    coarse_to_fine_steps_deg=SRP_SEARCH_STEPS_DEG,
                    minimum_frequency_hz=200.0,
                    maximum_frequency_hz=10_000.0,
                    relative_spectral_floor=1e-8,
                )
                exact_runtime.append(exact.runtime_seconds)
                if not exact.invalid and not srp_result.invalid:
                    exact_disagreement.append(
                        np.rad2deg(
                            np.arccos(
                                np.clip(exact.direction @ srp_result.direction, -1.0, 1.0)
                            )
                        )
                    )
    assert true_tdoa is not None
    gcc = GCCTrialSet(
        estimated_tdoa=estimated,
        peak_value=peak_value,
        peak_ratio=peak_ratio,
        peak_curvature=peak_curvature,
        spectral_energy=spectral_energy,
        spectral_energy_fraction=spectral_fraction,
        boundary_hit=boundary,
        invalid=invalid,
        seed=int(seed),
    )
    return (
        SRPPairedTrialSet(
            gcc=gcc,
            srp_angles=srp_angles,
            srp_scores=srp_scores,
            srp_boundary_hit=srp_boundary,
            srp_invalid=srp_invalid,
            gcc_runtime_seconds=gcc_runtime,
            srp_search_runtime_seconds=srp_runtime,
            exact_srp_runtime_seconds=np.asarray(exact_runtime, dtype=float),
            exact_fast_disagreement_deg=np.asarray(exact_disagreement, dtype=float),
        ),
        true_tdoa,
        pairs,
    )


def _base_context(
    config: GCCStatisticalConfig,
    method: str,
    calibration_count: int,
    evaluation_count: int,
    calibration_seed: int,
    evaluation_seed: int,
    srp_backend: str,
) -> dict[str, object]:
    return {
        "study_type": "srp_paired_full",
        "signal_model": config.signal_model,
        "split": "evaluation",
        "geometry": config.geometry,
        "direction": config.direction,
        "azimuth_deg": config.azimuth_deg,
        "elevation_deg": config.elevation_deg,
        "SNR": config.snr_db,
        "snr_db": config.snr_db,
        "frame_length": config.frame_length,
        "pair": "reference_3" if method == "reference_3_gcc_wls" else "all_6",
        "calibration_trial_count": calibration_count,
        "evaluation_trial_count": evaluation_count,
        "calibration_seed": calibration_seed,
        "evaluation_seed": evaluation_seed,
        "seed": evaluation_seed,
        "estimator_variant": method,
        "srp_backend": srp_backend if method == "equal_weight_srp_phat" else "not_applicable",
        "common_random_numbers": True,
    }


def run_srp_paired_configuration(
    config: GCCStatisticalConfig,
    configuration_index: int,
    *,
    calibration_trial_count: int = SRP_CALIBRATION_TRIAL_COUNT,
    evaluation_trial_count: int = SRP_EVALUATION_TRIAL_COUNT,
    sampling_rate_hz: float = 48_000.0,
    interpolation_factor: int = 2,
    srp_backend: str = "gcc_correlation_interpolated",
    exact_reference_trials: int = SRP_EXACT_REFERENCE_TRIALS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run one paired configuration and return DOA/runtime aggregate rows."""

    calibration_seed = _split_seed(configuration_index, 0)
    evaluation_seed = _split_seed(configuration_index, 1)
    calibration, true_tdoa, pairs = _simulate_split(
        config,
        calibration_trial_count,
        calibration_seed,
        sampling_rate_hz=sampling_rate_hz,
        interpolation_factor=interpolation_factor,
        estimate_srp=False,
        srp_backend=srp_backend,
        exact_reference_trials=0,
    )
    evaluation, evaluation_truth, evaluation_pairs = _simulate_split(
        config,
        evaluation_trial_count,
        evaluation_seed,
        sampling_rate_hz=sampling_rate_hz,
        interpolation_factor=interpolation_factor,
        estimate_srp=True,
        srp_backend=srp_backend,
        exact_reference_trials=exact_reference_trials,
    )
    np.testing.assert_allclose(evaluation_truth, true_tdoa, rtol=0.0, atol=2e-18)
    if evaluation_pairs != pairs:
        raise RuntimeError("calibration/evaluation pair sets differ")
    statistics = _calibration_statistics(calibration.gcc, true_tdoa, sampling_rate_hz)
    positions = comparison_arrays()[config.geometry]
    true_angles = np.deg2rad([config.azimuth_deg, config.elevation_deg])
    estimates = {name: np.full((evaluation_trial_count, 2), np.nan) for name in SRP_METHODS}
    valid = {name: np.zeros(evaluation_trial_count, dtype=bool) for name in SRP_METHODS}
    post_runtime = {name: np.zeros(evaluation_trial_count) for name in SRP_METHODS[:-1]}
    disconnected = np.zeros(evaluation_trial_count, dtype=bool)
    reference_indices = np.asarray([0, 1, 2])
    reference_pairs = tuple(pairs[index] for index in reference_indices)
    bias = np.asarray(statistics["bias"])
    precision = np.asarray(statistics["precision"])
    thresholds = np.asarray(statistics["confidence_thresholds"])
    medians = np.asarray(statistics["ratio_medians"])
    for trial in range(evaluation_trial_count):
        observed = evaluation.gcc.estimated_tdoa[trial]
        finite = np.isfinite(observed) & ~evaluation.gcc.invalid[trial]
        started = perf_counter()
        if np.all(finite[reference_indices]):
            estimate = estimate_doa_wls(
                observed[reference_indices],
                positions,
                reference_pairs,
                sigma_tdoa=1.0 / sampling_rate_hz,
            )
            estimates["reference_3_gcc_wls"][trial] = [estimate.phi, estimate.elevation]
            valid["reference_3_gcc_wls"][trial] = estimate.success
        post_runtime["reference_3_gcc_wls"][trial] = perf_counter() - started
        started = perf_counter()
        if np.all(finite):
            estimate = estimate_doa_wls(
                observed, positions, pairs, sigma_tdoa=1.0 / sampling_rate_hz
            )
            estimates["all_6_equal_gcc_wls"][trial] = [estimate.phi, estimate.elevation]
            valid["all_6_equal_gcc_wls"][trial] = estimate.success
        post_runtime["all_6_equal_gcc_wls"][trial] = perf_counter() - started
        started = perf_counter()
        ratios = evaluation.gcc.peak_ratio[trial]
        accepted = (
            finite
            & ~evaluation.gcc.boundary_hit[trial]
            & np.isfinite(ratios)
            & (ratios >= thresholds)
        )
        if _graph_connected(accepted, pairs):
            selected_pairs = tuple(
                pair for pair, keep in zip(pairs, accepted, strict=True) if keep
            )
            safe_ratios = np.where(np.isfinite(ratios), ratios, medians)
            weights = _confidence_weights(safe_ratios, medians, precision)
            selected_weights = weights[np.ix_(accepted, accepted)]
            projected = project_tdoa_cycles(
                (observed - bias)[accepted],
                selected_pairs,
                4,
                weights=selected_weights,
            )
            estimate = estimate_doa_wls(
                projected.consistent_tdoa,
                positions,
                selected_pairs,
                tdoa_covariance=np.linalg.pinv(selected_weights, rcond=1e-12),
            )
            estimates["all_6_calibrated_gcc_wls"][trial] = [
                estimate.phi,
                estimate.elevation,
            ]
            valid["all_6_calibrated_gcc_wls"][trial] = estimate.success
        else:
            disconnected[trial] = True
        post_runtime["all_6_calibrated_gcc_wls"][trial] = perf_counter() - started
    estimates["equal_weight_srp_phat"] = evaluation.srp_angles
    valid["equal_weight_srp_phat"] = ~evaluation.srp_invalid & np.all(
        np.isfinite(evaluation.srp_angles), axis=1
    )
    doa_records = []
    any_gcc_invalid = np.any(evaluation.gcc.invalid, axis=1)
    any_gcc_boundary = np.any(evaluation.gcc.boundary_hit, axis=1)
    for method in SRP_METHODS:
        metrics = _direction_metrics(estimates[method], valid[method], true_angles)
        if method == "equal_weight_srp_phat":
            total_runtime = (
                evaluation.gcc_runtime_seconds + evaluation.srp_search_runtime_seconds
                if srp_backend == "gcc_correlation_interpolated"
                else evaluation.srp_search_runtime_seconds
            )
            invalid_fraction = float(np.mean(evaluation.srp_invalid))
            boundary_fraction = float(np.mean(evaluation.srp_boundary_hit))
            disconnected_fraction = 0.0
        else:
            total_runtime = evaluation.gcc_runtime_seconds + post_runtime[method]
            invalid_fraction = float(np.mean(any_gcc_invalid))
            boundary_fraction = float(np.mean(any_gcc_boundary))
            disconnected_fraction = (
                float(np.mean(disconnected))
                if method == "all_6_calibrated_gcc_wls"
                else 0.0
            )
        doa_records.append(
            {
                **_base_context(
                    config,
                    method,
                    calibration_trial_count,
                    evaluation_trial_count,
                    calibration_seed,
                    evaluation_seed,
                    srp_backend,
                ),
                **metrics,
                "coverage": metrics["successful_fraction"],
                "failure_fraction": 1.0 - metrics["successful_fraction"],
                "invalid_fraction": invalid_fraction,
                "boundary_hit_fraction": boundary_fraction,
                "disconnected_graph_failure_fraction": disconnected_fraction,
                "catastrophic_fraction": metrics["fraction_error_gt_30deg"],
                "mean_runtime_per_estimate_s": float(np.mean(total_runtime)),
                "median_runtime_per_estimate_s": float(np.median(total_runtime)),
                "p95_runtime_per_estimate_s": float(np.percentile(total_runtime, 95.0)),
                "metric_conditioning": "successful_trials_only",
                "equal_pair_weights": method == "equal_weight_srp_phat",
                "gcc_confidence_used_by_srp": False,
            }
        )
    exact_trial_count = int(evaluation.exact_srp_runtime_seconds.size)
    mean_disagreement = (
        float(np.mean(evaluation.exact_fast_disagreement_deg))
        if evaluation.exact_fast_disagreement_deg.size
        else float("nan")
    )
    max_disagreement = (
        float(np.max(evaluation.exact_fast_disagreement_deg))
        if evaluation.exact_fast_disagreement_deg.size
        else float("nan")
    )
    runtime_base = {
        **_base_context(
            config,
            "runtime_diagnostic",
            calibration_trial_count,
            evaluation_trial_count,
            calibration_seed,
            evaluation_seed,
            srp_backend,
        ),
        "exact_reference_trials_per_configuration": exact_trial_count,
        "exact_reference_sampling_scope": (
            f"first_{exact_trial_count}_evaluation_trials_per_configuration"
        ),
        "exact_fast_disagreement_covers_all_evaluation_trials": False,
    }
    runtime_records = []
    for component, values in (
        ("gcc_all_pair_phat", evaluation.gcc_runtime_seconds),
        ("srp_interpolated_search", evaluation.srp_search_runtime_seconds),
        ("srp_exact_vectorized_reference", evaluation.exact_srp_runtime_seconds),
    ):
        is_exact_component = component == "srp_exact_vectorized_reference"
        runtime_records.append(
            {
                **runtime_base,
                "runtime_component": component,
                "runtime_sample_count": int(values.size),
                "exact_reference_trial_count": (
                    exact_trial_count if is_exact_component else 0
                ),
                "exact_fast_disagreement_trial_count": (
                    int(evaluation.exact_fast_disagreement_deg.size)
                    if is_exact_component
                    else 0
                ),
                "exact_fast_disagreement_scope": (
                    "sampled_exact_reference_trials_only"
                    if is_exact_component
                    else "not_applicable"
                ),
                "mean_exact_fast_disagreement_deg": (
                    mean_disagreement if is_exact_component else float("nan")
                ),
                "max_exact_fast_disagreement_deg": (
                    max_disagreement if is_exact_component else float("nan")
                ),
                "mean_runtime_s": float(np.mean(values)) if values.size else float("nan"),
                "median_runtime_s": float(np.median(values)) if values.size else float("nan"),
                "p95_runtime_s": float(np.percentile(values, 95.0))
                if values.size
                else float("nan"),
            }
        )
    return doa_records, runtime_records


def normalize_srp_runtime_reporting(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Normalize legacy rows so exact-trial counts are globally summable.

    Legacy CSV files repeated ``exact_reference_trial_count`` and the exact/fast
    disagreement in all three runtime-component rows. The normalized schema
    keeps the requested per-configuration count under an explicit name, while
    the unique-count contribution and disagreement occur only in the exact
    component row.
    """

    normalized: list[dict[str, object]] = []
    for source in records:
        row = dict(source)
        component = str(row.get("runtime_component", ""))
        is_exact = component == "srp_exact_vectorized_reference"
        old_count = int(float(row.get("exact_reference_trial_count", 0) or 0))
        exact_count = (
            int(float(row.get("runtime_sample_count", 0) or 0))
            if is_exact
            else old_count
        )
        row["exact_reference_trials_per_configuration"] = exact_count
        row["exact_reference_sampling_scope"] = (
            f"first_{exact_count}_evaluation_trials_per_configuration"
        )
        row["exact_fast_disagreement_covers_all_evaluation_trials"] = False
        row["exact_reference_trial_count"] = exact_count if is_exact else 0
        row["exact_fast_disagreement_trial_count"] = exact_count if is_exact else 0
        row["exact_fast_disagreement_scope"] = (
            "sampled_exact_reference_trials_only" if is_exact else "not_applicable"
        )
        if not is_exact:
            row["mean_exact_fast_disagreement_deg"] = float("nan")
            row["max_exact_fast_disagreement_deg"] = float("nan")
        normalized.append(row)
    return normalized


def default_srp_statistical_configurations() -> tuple[GCCStatisticalConfig, ...]:
    """Cartesian study: two arrays, three directions, eleven SNRs, three signals."""

    return tuple(
        GCCStatisticalConfig(
            "srp_paired_full",
            signal_model,
            geometry,
            azimuth,
            elevation,
            snr,
            SRP_FRAME_LENGTH,
        )
        for geometry in ("square", "tetrahedral")
        for azimuth, elevation in SRP_DIRECTIONS_DEG
        for signal_model in SIGNAL_MODELS
        for snr in FINE_SNR_LEVELS_DB
    )


def run_srp_statistical_validation(
    *,
    configurations: tuple[GCCStatisticalConfig, ...] | None = None,
    calibration_trial_count: int = SRP_CALIBRATION_TRIAL_COUNT,
    evaluation_trial_count: int = SRP_EVALUATION_TRIAL_COUNT,
    interpolation_factor: int = 2,
    srp_backend: str = "gcc_correlation_interpolated",
    exact_reference_trials: int = SRP_EXACT_REFERENCE_TRIALS,
    doa_output_csv: str | Path = "results/srp_doa_summary.csv",
    runtime_output_csv: str | Path = "results/srp_runtime_summary.csv",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected = (
        default_srp_statistical_configurations()
        if configurations is None
        else configurations
    )
    doa_records: list[dict[str, object]] = []
    runtime_records: list[dict[str, object]] = []
    for index, config in enumerate(selected):
        doa_group, runtime_group = run_srp_paired_configuration(
            config,
            index,
            calibration_trial_count=calibration_trial_count,
            evaluation_trial_count=evaluation_trial_count,
            interpolation_factor=interpolation_factor,
            srp_backend=srp_backend,
            exact_reference_trials=exact_reference_trials,
        )
        doa_records.extend(doa_group)
        runtime_records.extend(runtime_group)
    _write_records(doa_records, doa_output_csv)
    _write_records(runtime_records, runtime_output_csv)
    return doa_records, runtime_records


__all__ = [
    "SRP_CALIBRATION_TRIAL_COUNT",
    "SRP_DIRECTIONS_DEG",
    "SRP_EVALUATION_TRIAL_COUNT",
    "SRP_EXACT_REFERENCE_TRIALS",
    "SRP_FRAME_LENGTH",
    "SRP_METHODS",
    "SRP_SEARCH_STEPS_DEG",
    "SRP_STATISTICAL_BASE_SEED",
    "SRPPairedTrialSet",
    "default_srp_statistical_configurations",
    "run_srp_paired_configuration",
    "run_srp_statistical_validation",
    "normalize_srp_runtime_reporting",
]
