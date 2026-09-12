"""Held-out paired S7C-D2 comparison of opt-in robust strict-CV EKF variants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from estimators.retarded_ekf import (
    CausalRetardedTimeEKF,
    RetardedEKFRobustnessConfig,
)
from model.bearing_events import bearing_event_id
from validation.retarded_ekf_stress_study import (
    EVALUATION_TIMES_S,
    StressBaseBlock,
    StressScenario,
    _wilson_interval,
    default_stress_profiles,
    generate_stress_base_block,
    generate_stress_scenario,
)
from validation.retarded_ekf_study import P95_METHOD, linear_percentile


DEFAULT_ROBUST_EVALUATION_SEED = 20260912
DEFAULT_ROBUST_SMOKE_SEED = 20260911
DEFAULT_SEQUENCE_COUNT = 100
PAIRED_BOOTSTRAP_RESAMPLES = 2_000
PRE_UPDATE_NIS_THRESHOLD = 9.210340371976184
CONSENSUS_NIS_THRESHOLD = 10.596634733096073
STATE_CHI2_95 = 12.591587243743977
GEOMETRIES = ("informative", "poorly_conditioned")
MECHANISMS = (
    "truth",
    "nominal_bearing_noise",
    "transport_loss_uniform",
    "transport_delay_uniform",
    "outlier_mask_uniform",
    "outlier_direction",
)


@dataclass(frozen=True, slots=True)
class RobustVariant:
    name: str
    consensus_initialization: bool
    nis_gate: bool

    @property
    def config(self) -> RetardedEKFRobustnessConfig:
        return RetardedEKFRobustnessConfig(
            consensus_initialization=self.consensus_initialization,
            maximum_pre_update_nis=(
                PRE_UPDATE_NIS_THRESHOLD if self.nis_gate else None
            ),
            consensus_nis_threshold=CONSENSUS_NIS_THRESHOLD,
        )


ROBUST_VARIANTS = (
    RobustVariant("c1_baseline", False, False),
    RobustVariant("robust_initialization_only", True, False),
    RobustVariant("nis_gate_only", False, True),
    RobustVariant("robust_combined", True, True),
)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0", ""}:
            return False
    raise ValueError(f"cannot interpret boolean value: {value!r}")


def _state_metrics(publication, truth, evaluation_time_s: float) -> dict[str, object]:
    if not publication.valid or publication.state is None:
        return {
            "position_error_m": float("nan"),
            "velocity_error_mps": float("nan"),
            "state_nees": float("nan"),
            "state_95_covered": False,
            "covariance_symmetry_error": float("nan"),
            "covariance_minimum_eigenvalue": float("nan"),
            "covariance_condition_number": float("inf"),
        }
    covariance = np.asarray(publication.covariance_state, dtype=float)
    error = np.concatenate(
        (
            publication.state.position_at(evaluation_time_s)
            - truth.position_at(evaluation_time_s),
            publication.state.velocity_world_mps - truth.velocity_world_mps,
        )
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    nees = float(error @ np.linalg.solve(covariance, error))
    return {
        "position_error_m": float(np.linalg.norm(error[:3])),
        "velocity_error_mps": float(np.linalg.norm(error[3:])),
        "state_nees": nees,
        "state_95_covered": bool(nees <= STATE_CHI2_95),
        "covariance_symmetry_error": float(
            np.max(np.abs(covariance - covariance.T), initial=0.0)
        ),
        "covariance_minimum_eigenvalue": float(np.min(eigenvalues)),
        "covariance_condition_number": float(
            np.max(eigenvalues) / np.min(eigenvalues)
        ),
    }


def _first_initialization_time(publications) -> float:
    values = [
        item.processing_time_s
        for publication in publications
        for item in publication.new_lifecycle_diagnostics
        if item.action in {"initialized", "reinitialized_after_conflict"}
    ]
    return min(values, default=float("nan"))


def _run_variant(
    scenario: StressScenario, variant: RobustVariant
) -> dict[str, object]:
    processor = CausalRetardedTimeEKF(
        scenario.base_block.stations,
        scenario.events,
        estimator_variant="direct_bearing",
        robustness_config=variant.config,
    )
    publications = []
    update_diagnostics = []
    total_runtime = 0.0
    initialization_runtime = 0.0
    update_runtime = 0.0
    for evaluation_time in EVALUATION_TIMES_S:
        publication = processor.advance_to(evaluation_time)
        publications.append(publication)
        update_diagnostics.extend(publication.update_diagnostics)
        total_runtime += publication.total_runtime_s
        initialization_runtime += publication.initialization_runtime_s
        update_runtime += publication.measurement_update_runtime_s
    final = publications[-1]
    metrics = _state_metrics(final, scenario.base_block.truth_state, 14.5)
    truth_by_id = {item.event_id: item for item in scenario.event_truth}
    delivered_ids = {
        item.event_id for item in scenario.event_truth if item.delivered
    }
    outlier_ids = {
        item.event_id
        for item in scenario.event_truth
        if item.delivered and item.is_outlier
    }
    clean_ids = delivered_ids - outlier_ids
    initialization_ids = set(final.initialization_event_ids)
    applied_ids = set(final.applied_event_ids)
    rejection_reason_by_id = {
        item.event_id: item.reason for item in final.event_rejections
    }
    robust_rejected_ids = {
        identity
        for identity, reason in rejection_reason_by_id.items()
        if reason
        in {"robust_initialization_consensus_outlier", "pre_update_nis_gate"}
    }
    used_ids = initialization_ids | applied_ids
    classified_ids = used_ids | robust_rejected_ids
    remaining_ids = delivered_ids - classified_ids
    finite_nis = np.asarray(
        [
            item.normalized_innovation_squared
            for item in update_diagnostics
            if np.isfinite(item.normalized_innovation_squared)
        ],
        dtype=float,
    )
    accepted_nis = np.asarray(
        [
            item.normalized_innovation_squared
            for item in update_diagnostics
            if item.update_applied and np.isfinite(item.normalized_innovation_squared)
        ],
        dtype=float,
    )
    rejected_nis = np.asarray(
        [
            item.normalized_innovation_squared
            for item in update_diagnostics
            if item.failure_reason == "pre_update_nis_gate"
            and np.isfinite(item.normalized_innovation_squared)
        ],
        dtype=float,
    )
    failures = Counter(
        str(item.failure_reason) for item in publications if not item.valid
    )
    return {
        "geometry": scenario.base_block.geometry,
        "profile": scenario.profile.name,
        "sequence_index": scenario.base_block.sequence_index,
        "base_seed": scenario.base_block.base_seed,
        "sequence_id": scenario.sequence_id,
        "variant": variant.name,
        "consensus_initialization_enabled": variant.consensus_initialization,
        "pre_update_nis_gate_enabled": variant.nis_gate,
        "consensus_nis_threshold": CONSENSUS_NIS_THRESHOLD,
        "pre_update_nis_threshold": (
            PRE_UPDATE_NIS_THRESHOLD if variant.nis_gate else float("nan")
        ),
        "independent_unit": "whole_base_sequence",
        "variants_within_base_sequence_are_paired": True,
        "epochs_and_events_within_sequence_are_dependent": True,
        "potential_event_count": len(scenario.event_truth),
        "delivered_event_count": len(delivered_ids),
        "lost_event_count": len(scenario.event_truth) - len(delivered_ids),
        "delivered_clean_event_count": len(clean_ids),
        "delivered_outlier_event_count": len(outlier_ids),
        "initialization_succeeded": any(item.initialized for item in publications),
        "initialization_time_s": _first_initialization_time(publications),
        "final_valid": final.valid,
        "final_failure_reason": final.failure_reason or "",
        "failure_reason_counts_over_dependent_epochs": json.dumps(
            failures, sort_keys=True
        ),
        **metrics,
        "initialization_event_count": len(initialization_ids),
        "applied_update_event_count": len(applied_ids),
        "robust_rejected_event_count": len(robust_rejected_ids),
        "other_rejected_event_count": len(rejection_reason_by_id) - len(robust_rejected_ids),
        "remaining_unclassified_delivered_event_count": len(remaining_ids),
        "false_rejected_clean_event_count": len(robust_rejected_ids & clean_ids),
        "detected_outlier_event_count": len(robust_rejected_ids & outlier_ids),
        "missed_outlier_event_count": len(used_ids & outlier_ids),
        "unclassified_outlier_event_count": len(remaining_ids & outlier_ids),
        "classified_clean_event_count": len(classified_ids & clean_ids),
        "classified_outlier_event_count": len(classified_ids & outlier_ids),
        "initialization_event_ids_json": json.dumps(sorted(initialization_ids)),
        "applied_update_event_ids_json": json.dumps(sorted(applied_ids)),
        "robust_rejected_event_ids_json": json.dumps(sorted(robust_rejected_ids)),
        "rejection_reasons_json": json.dumps(rejection_reason_by_id, sort_keys=True),
        "pre_update_nis_defined_count": len(finite_nis),
        "pre_update_nis_undefined_count": len(update_diagnostics) - len(finite_nis),
        "pre_update_nis_all_mean": (
            float(np.mean(finite_nis)) if finite_nis.size else float("nan")
        ),
        "pre_update_nis_all_p95": (
            linear_percentile(finite_nis, 95) if finite_nis.size else float("nan")
        ),
        "pre_update_nis_accepted_count": len(accepted_nis),
        "pre_update_nis_accepted_mean": (
            float(np.mean(accepted_nis)) if accepted_nis.size else float("nan")
        ),
        "pre_update_nis_rejected_count": len(rejected_nis),
        "pre_update_nis_rejected_mean": (
            float(np.mean(rejected_nis)) if rejected_nis.size else float("nan")
        ),
        "pre_update_nis_dof": 2,
        "pre_update_nis_all_attempts_include_gate_rejections": True,
        "initialization_runtime_s": initialization_runtime,
        "measurement_update_runtime_s": update_runtime,
        "total_processing_runtime_s": total_runtime,
        "outlier_truth_used_by_estimator": False,
        "truth_by_id_count_evaluator_only": len(truth_by_id),
    }


def run_robust_scenario(scenario: StressScenario) -> list[dict[str, object]]:
    """Run all four variants on the exact same immutable event stream."""

    return [_run_variant(scenario, variant) for variant in ROBUST_VARIANTS]


def _bootstrap_interval(
    paired: Sequence[tuple[dict[str, object], dict[str, object]]],
    statistic,
    *,
    seed_coordinates: tuple[int, ...],
) -> tuple[float, float]:
    if not paired:
        return float("nan"), float("nan")
    rng = np.random.default_rng(
        np.random.SeedSequence([DEFAULT_ROBUST_EVALUATION_SEED, *seed_coordinates])
    )
    values = []
    for _ in range(PAIRED_BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(paired), size=len(paired))
        sample = [paired[int(index)] for index in indices]
        values.append(statistic(sample))
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan"), float("nan")
    return tuple(np.percentile(finite, [2.5, 97.5], method="linear"))


def _fraction_difference(sample, field: str) -> float:
    return float(
        np.mean([bool(variant[field]) - bool(baseline[field]) for variant, baseline in sample])
    )


def _rmse_difference(sample, field: str) -> float:
    def rmse(rows, index):
        values = np.asarray([float(pair[index][field]) for pair in rows], dtype=float)
        values = values[np.isfinite(values)]
        return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")

    return rmse(sample, 0) - rmse(sample, 1)


def _mean_rate_difference(sample, numerator: str, denominator: str) -> float:
    def rate(row):
        total = int(row[denominator])
        return float(row[numerator]) / total if total else float("nan")

    differences = [rate(variant) - rate(baseline) for variant, baseline in sample]
    finite = np.asarray(differences, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def summarize_robust_profiles(
    sequence_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in sequence_rows:
        groups.setdefault(
            (str(row["geometry"]), str(row["profile"]), str(row["variant"])), []
        ).append(row)
    result = []
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for profile_index, profile in enumerate(default_stress_profiles()):
            baseline_rows = sorted(
                groups[(geometry, profile.name, "c1_baseline")],
                key=lambda row: int(row["sequence_index"]),
            )
            baseline_by_sequence = {
                int(row["sequence_index"]): row for row in baseline_rows
            }
            for variant_index, variant in enumerate(ROBUST_VARIANTS):
                rows = sorted(
                    groups[(geometry, profile.name, variant.name)],
                    key=lambda row: int(row["sequence_index"]),
                )
                count = len(rows)
                valid = [row for row in rows if bool(row["final_valid"])]
                initialized = sum(bool(row["initialization_succeeded"]) for row in rows)
                covered = sum(
                    bool(row["final_valid"]) and bool(row["state_95_covered"])
                    for row in rows
                )
                valid_count = len(valid)
                conditional_covered = sum(bool(row["state_95_covered"]) for row in valid)
                position_errors = np.asarray(
                    [float(row["position_error_m"]) for row in valid], dtype=float
                )
                velocity_errors = np.asarray(
                    [float(row["velocity_error_mps"]) for row in valid], dtype=float
                )
                init_times = np.asarray(
                    [float(row["initialization_time_s"]) for row in rows], dtype=float
                )
                init_times = init_times[np.isfinite(init_times)]
                delivered_clean = sum(int(row["delivered_clean_event_count"]) for row in rows)
                delivered_outlier = sum(int(row["delivered_outlier_event_count"]) for row in rows)
                false_clean = sum(int(row["false_rejected_clean_event_count"]) for row in rows)
                detected_outlier = sum(int(row["detected_outlier_event_count"]) for row in rows)
                missed_outlier = sum(int(row["missed_outlier_event_count"]) for row in rows)
                unclassified_outlier = sum(
                    int(row["unclassified_outlier_event_count"]) for row in rows
                )
                paired = [
                    (row, baseline_by_sequence[int(row["sequence_index"])])
                    for row in rows
                ]
                bootstrap_base = (geometry_index, profile_index, variant_index)
                ci_valid = _bootstrap_interval(
                    paired,
                    lambda sample: _fraction_difference(sample, "final_valid"),
                    seed_coordinates=(*bootstrap_base, 0),
                )
                ci_coverage = _bootstrap_interval(
                    paired,
                    lambda sample: _fraction_difference(sample, "state_95_covered"),
                    seed_coordinates=(*bootstrap_base, 1),
                )
                ci_position = _bootstrap_interval(
                    paired,
                    lambda sample: _rmse_difference(sample, "position_error_m"),
                    seed_coordinates=(*bootstrap_base, 2),
                )
                ci_velocity = _bootstrap_interval(
                    paired,
                    lambda sample: _rmse_difference(sample, "velocity_error_mps"),
                    seed_coordinates=(*bootstrap_base, 3),
                )
                ci_false_rejection = _bootstrap_interval(
                    paired,
                    lambda sample: _mean_rate_difference(
                        sample,
                        "false_rejected_clean_event_count",
                        "delivered_clean_event_count",
                    ),
                    seed_coordinates=(*bootstrap_base, 4),
                )
                failures = Counter(
                    str(row["final_failure_reason"])
                    for row in rows
                    if not bool(row["final_valid"])
                )
                all_nis_count = sum(int(row["pre_update_nis_defined_count"]) for row in rows)
                all_nis_weighted = sum(
                    int(row["pre_update_nis_defined_count"])
                    * float(row["pre_update_nis_all_mean"])
                    for row in rows
                    if int(row["pre_update_nis_defined_count"])
                )
                valid_ci = _wilson_interval(valid_count, count)
                initialization_ci = _wilson_interval(initialized, count)
                unconditional_ci = _wilson_interval(covered, count)
                conditional_ci = (
                    _wilson_interval(conditional_covered, valid_count)
                    if valid_count
                    else (float("nan"), float("nan"))
                )
                result.append(
                    {
                        "geometry": geometry,
                        "profile": profile.name,
                        "variant": variant.name,
                        "independent_sequence_count": count,
                        "dependent_event_count": sum(
                            int(row["delivered_event_count"]) for row in rows
                        ),
                        "initialization_success_count": initialized,
                        "initialization_success_fraction": initialized / count,
                        "initialization_success_ci95_low": initialization_ci[0],
                        "initialization_success_ci95_high": initialization_ci[1],
                        "mean_initialization_time_s": (
                            float(np.mean(init_times)) if init_times.size else float("nan")
                        ),
                        "final_valid_count": valid_count,
                        "final_invalid_count": count - valid_count,
                        "final_valid_fraction": valid_count / count,
                        "final_valid_fraction_ci95_low": valid_ci[0],
                        "final_valid_fraction_ci95_high": valid_ci[1],
                        "final_failure_reason_counts": json.dumps(failures, sort_keys=True),
                        "conditional_position_rmse_m": (
                            float(np.sqrt(np.mean(position_errors**2)))
                            if position_errors.size
                            else float("nan")
                        ),
                        "conditional_position_p95_m": (
                            linear_percentile(position_errors, 95)
                            if position_errors.size
                            else float("nan")
                        ),
                        "conditional_velocity_rmse_mps": (
                            float(np.sqrt(np.mean(velocity_errors**2)))
                            if velocity_errors.size
                            else float("nan")
                        ),
                        "conditional_velocity_p95_mps": (
                            linear_percentile(velocity_errors, 95)
                            if velocity_errors.size
                            else float("nan")
                        ),
                        "conditional_metric_valid_denominator": valid_count,
                        "total_sequence_denominator": count,
                        "conditional_state_95_coverage_fraction": (
                            conditional_covered / valid_count
                            if valid_count
                            else float("nan")
                        ),
                        "conditional_state_95_coverage_ci95_low": conditional_ci[0],
                        "conditional_state_95_coverage_ci95_high": conditional_ci[1],
                        "unconditional_valid_and_covered_fraction": covered / count,
                        "unconditional_valid_and_covered_ci95_low": unconditional_ci[0],
                        "unconditional_valid_and_covered_ci95_high": unconditional_ci[1],
                        "delivered_clean_event_count": delivered_clean,
                        "false_rejected_clean_event_count": false_clean,
                        "false_rejected_clean_event_fraction": (
                            false_clean / delivered_clean if delivered_clean else float("nan")
                        ),
                        "delivered_outlier_event_count": delivered_outlier,
                        "detected_outlier_event_count": detected_outlier,
                        "detected_outlier_event_fraction": (
                            detected_outlier / delivered_outlier
                            if delivered_outlier
                            else float("nan")
                        ),
                        "missed_outlier_event_count": missed_outlier,
                        "missed_outlier_event_fraction": (
                            missed_outlier / delivered_outlier
                            if delivered_outlier
                            else float("nan")
                        ),
                        "unclassified_outlier_event_count": unclassified_outlier,
                        "mean_pre_update_nis_all_attempts": (
                            all_nis_weighted / all_nis_count
                            if all_nis_count
                            else float("nan")
                        ),
                        "pre_update_nis_defined_count": all_nis_count,
                        "pre_update_nis_rejected_count": sum(
                            int(row["pre_update_nis_rejected_count"]) for row in rows
                        ),
                        "mean_initialization_runtime_s": float(
                            np.mean([float(row["initialization_runtime_s"]) for row in rows])
                        ),
                        "mean_measurement_update_runtime_s": float(
                            np.mean(
                                [float(row["measurement_update_runtime_s"]) for row in rows]
                            )
                        ),
                        "mean_total_processing_runtime_s": float(
                            np.mean([float(row["total_processing_runtime_s"]) for row in rows])
                        ),
                        "maximum_covariance_symmetry_error": max(
                            (
                                float(row["covariance_symmetry_error"])
                                for row in valid
                            ),
                            default=float("nan"),
                        ),
                        "minimum_covariance_eigenvalue": min(
                            (
                                float(row["covariance_minimum_eigenvalue"])
                                for row in valid
                            ),
                            default=float("nan"),
                        ),
                        "maximum_covariance_condition_number": max(
                            (
                                float(row["covariance_condition_number"])
                                for row in valid
                                if np.isfinite(float(row["covariance_condition_number"]))
                            ),
                            default=float("nan"),
                        ),
                        "paired_valid_fraction_difference_vs_c1": (
                            valid_count / count
                            - np.mean([bool(row["final_valid"]) for row in baseline_rows])
                        ),
                        "paired_valid_fraction_difference_ci95_low": ci_valid[0],
                        "paired_valid_fraction_difference_ci95_high": ci_valid[1],
                        "paired_unconditional_coverage_difference_vs_c1": (
                            covered / count
                            - np.mean(
                                [
                                    bool(row["final_valid"])
                                    and bool(row["state_95_covered"])
                                    for row in baseline_rows
                                ]
                            )
                        ),
                        "paired_unconditional_coverage_difference_ci95_low": ci_coverage[0],
                        "paired_unconditional_coverage_difference_ci95_high": ci_coverage[1],
                        "paired_conditional_position_rmse_difference_vs_c1_m": (
                            _rmse_difference(paired, "position_error_m")
                        ),
                        "paired_position_rmse_difference_ci95_low_m": ci_position[0],
                        "paired_position_rmse_difference_ci95_high_m": ci_position[1],
                        "paired_conditional_velocity_rmse_difference_vs_c1_mps": (
                            _rmse_difference(paired, "velocity_error_mps")
                        ),
                        "paired_velocity_rmse_difference_ci95_low_mps": ci_velocity[0],
                        "paired_velocity_rmse_difference_ci95_high_mps": ci_velocity[1],
                        "paired_mean_false_rejection_rate_difference_vs_c1": (
                            _mean_rate_difference(
                                paired,
                                "false_rejected_clean_event_count",
                                "delivered_clean_event_count",
                            )
                        ),
                        "paired_false_rejection_rate_difference_ci95_low": ci_false_rejection[0],
                        "paired_false_rejection_rate_difference_ci95_high": ci_false_rejection[1],
                        "paired_bootstrap_resample_count": PAIRED_BOOTSTRAP_RESAMPLES,
                        "paired_bootstrap_unit": "whole_base_sequence",
                        "p95_method": P95_METHOD,
                        "outlier_truth_used_by_estimator": False,
                        "covariance_after_selection_assumed_calibrated": False,
                    }
                )
    return result


def _seed_row(block: StressBaseBlock) -> dict[str, object]:
    row: dict[str, object] = {
        "geometry": block.geometry,
        "geometry_index": block.geometry_index,
        "sequence_index": block.sequence_index,
        "base_seed": block.base_seed,
        "independent_unit": "whole_base_sequence",
    }
    for name, identifier, generated in zip(
        MECHANISMS,
        block.provenance.identifiers,
        block.provenance.generated_seeds,
        strict=True,
    ):
        row[f"{name}_provenance"] = identifier
        row[f"{name}_seed"] = generated
    return row


def _run_block_task(arguments):
    geometry, sequence_index, base_seed = arguments
    block = generate_stress_base_block(
        geometry, sequence_index, base_seed=base_seed
    )
    rows = []
    for profile in default_stress_profiles():
        rows.extend(run_robust_scenario(generate_stress_scenario(block, profile)))
    return rows, _seed_row(block)


def audit_robust_results(
    sequence_rows: Sequence[dict[str, object]],
    summary_rows: Sequence[dict[str, object]],
    seed_rows: Sequence[dict[str, object]],
    *,
    sequence_count: int,
) -> dict[str, int]:
    expected_sequences = (
        len(GEOMETRIES)
        * len(default_stress_profiles())
        * len(ROBUST_VARIANTS)
        * sequence_count
    )
    expected_summaries = (
        len(GEOMETRIES) * len(default_stress_profiles()) * len(ROBUST_VARIANTS)
    )
    if len(sequence_rows) != expected_sequences:
        raise AssertionError("unexpected robust sequence row count")
    if len(summary_rows) != expected_summaries:
        raise AssertionError("unexpected robust summary row count")
    if len(seed_rows) != len(GEOMETRIES) * sequence_count:
        raise AssertionError("unexpected robust provenance row count")
    identifiers = [
        str(row[f"{name}_provenance"]) for row in seed_rows for name in MECHANISMS
    ]
    seeds = [int(row[f"{name}_seed"]) for row in seed_rows for name in MECHANISMS]
    if len(identifiers) != len(set(identifiers)) or len(seeds) != len(set(seeds)):
        raise AssertionError("robust evaluation seed collision")
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in sequence_rows:
        grouped.setdefault(
            (
                str(row["geometry"]),
                str(row["profile"]),
                int(row["sequence_index"]),
            ),
            [],
        ).append(row)
        delivered = int(row["delivered_event_count"])
        partition = sum(
            int(row[name])
            for name in (
                "initialization_event_count",
                "applied_update_event_count",
                "robust_rejected_event_count",
                "other_rejected_event_count",
                "remaining_unclassified_delivered_event_count",
            )
        )
        if delivered != partition:
            raise AssertionError("delivered robust events do not partition")
        if _as_bool(row["outlier_truth_used_by_estimator"]):
            raise AssertionError("truth leaked into robust estimator")
    if any(len(rows) != len(ROBUST_VARIANTS) for rows in grouped.values()):
        raise AssertionError("variant pairing is incomplete")
    return {
        "sequence_row_count": len(sequence_rows),
        "summary_row_count": len(summary_rows),
        "seed_row_count": len(seed_rows),
        "unique_mechanism_seed_count": len(set(seeds)),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty robust CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_retarded_ekf_robust_study(
    output_directory: str | Path = "results",
    *,
    sequence_count: int = DEFAULT_SEQUENCE_COUNT,
    base_seed: int = DEFAULT_ROBUST_EVALUATION_SEED,
    workers: int = 1,
    progress: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Run the frozen held-out robust comparison and save audited CSVs."""

    if sequence_count <= 0 or workers <= 0:
        raise ValueError("sequence_count and workers must be positive")
    tasks = [
        (geometry, sequence_index, base_seed)
        for geometry in GEOMETRIES
        for sequence_index in range(sequence_count)
    ]
    sequence_rows = []
    seed_rows = []
    iterator = map(_run_block_task, tasks)
    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_run_block_task, tasks, chunksize=1)
    try:
        for completed, (rows, seed_row) in enumerate(iterator, start=1):
            sequence_rows.extend(rows)
            seed_rows.append(seed_row)
            if progress:
                print(f"S7C-D2 completed base blocks: {completed}/{len(tasks)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    summary_rows = summarize_robust_profiles(sequence_rows)
    audit_robust_results(
        sequence_rows,
        summary_rows,
        seed_rows,
        sequence_count=sequence_count,
    )
    output = Path(output_directory)
    _write_csv(output / "retarded_ekf_robust_sequence_results.csv", sequence_rows)
    _write_csv(output / "retarded_ekf_robust_profile_summary.csv", summary_rows)
    _write_csv(output / "retarded_ekf_robust_seed_provenance.csv", seed_rows)
    return sequence_rows, summary_rows, seed_rows


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default="results")
    parser.add_argument("--sequence-count", type=int, default=DEFAULT_SEQUENCE_COUNT)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_ROBUST_EVALUATION_SEED)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--progress", action="store_true")
    arguments = parser.parse_args()
    started = time.perf_counter()
    rows, summaries, seeds = run_retarded_ekf_robust_study(
        arguments.output_directory,
        sequence_count=arguments.sequence_count,
        base_seed=arguments.base_seed,
        workers=arguments.workers,
        progress=arguments.progress,
    )
    print(
        json.dumps(
            {
                "sequence_rows": len(rows),
                "summary_rows": len(summaries),
                "seed_rows": len(seeds),
                "elapsed_s": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()


__all__ = [
    "CONSENSUS_NIS_THRESHOLD",
    "DEFAULT_ROBUST_EVALUATION_SEED",
    "DEFAULT_ROBUST_SMOKE_SEED",
    "PRE_UPDATE_NIS_THRESHOLD",
    "ROBUST_VARIANTS",
    "RobustVariant",
    "audit_robust_results",
    "run_retarded_ekf_robust_study",
    "run_robust_scenario",
    "summarize_robust_profiles",
]
