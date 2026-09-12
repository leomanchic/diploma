"""Paired development/evaluation study for confirmed initialization recovery."""

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
from estimators.retarded_ekf_recovery import CausalConfirmedRetardedTimeEKF
from model.bearing_events import bearing_event_id
from validation.retarded_ekf_robust_study import (
    CONSENSUS_NIS_THRESHOLD,
    PRE_UPDATE_NIS_THRESHOLD,
    STATE_CHI2_95,
)
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


DEVELOPMENT_SEED = 20260913
EVALUATION_SEED = 20260914
SMOKE_SEED = 20260915
DEFAULT_SEQUENCE_COUNT = 100
BOOTSTRAP_RESAMPLES = 2_000
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
class RecoveryStudyVariant:
    name: str
    kind: str


VARIANTS = (
    RecoveryStudyVariant("c1_baseline", "c1"),
    RecoveryStudyVariant("d2_combined_published", "d2"),
    RecoveryStudyVariant("confirmed_recovery", "recovery"),
)

KNOWN_FAILURE_CASES = (
    ("informative", 57),
    ("poorly_conditioned", 37),
)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0", ""}:
            return False
    raise ValueError(f"cannot parse boolean {value!r}")


def _state_metrics(publication, truth, evaluation_time_s: float) -> dict[str, object]:
    if not publication.valid or publication.state is None:
        return {
            "position_error_m": float("nan"),
            "velocity_error_mps": float("nan"),
            "state_nees": float("nan"),
            "state_95_covered": False,
            "covariance_symmetry_error": float("nan"),
            "covariance_minimum_eigenvalue": float("nan"),
        }
    covariance = np.asarray(publication.covariance_state, dtype=float)
    error = np.concatenate(
        (
            publication.state.position_at(evaluation_time_s)
            - truth.position_at(evaluation_time_s),
            publication.state.velocity_world_mps - truth.velocity_world_mps,
        )
    )
    nees = float(error @ np.linalg.solve(covariance, error))
    return {
        "position_error_m": float(np.linalg.norm(error[:3])),
        "velocity_error_mps": float(np.linalg.norm(error[3:])),
        "state_nees": nees,
        "state_95_covered": bool(nees <= STATE_CHI2_95),
        "covariance_symmetry_error": float(
            np.max(np.abs(covariance - covariance.T), initial=0.0)
        ),
        "covariance_minimum_eigenvalue": float(
            np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T)))
        ),
    }


def _classic_processor(scenario: StressScenario, kind: str):
    if kind == "c1":
        robustness = RetardedEKFRobustnessConfig()
    elif kind == "d2":
        robustness = RetardedEKFRobustnessConfig(
            consensus_initialization=True,
            maximum_pre_update_nis=PRE_UPDATE_NIS_THRESHOLD,
            consensus_nis_threshold=CONSENSUS_NIS_THRESHOLD,
        )
    else:
        raise ValueError(f"unknown classic variant kind {kind}")
    return CausalRetardedTimeEKF(
        scenario.base_block.stations,
        scenario.events,
        estimator_variant="direct_bearing",
        robustness_config=robustness,
    )


def _recovery_durations(lifecycle) -> tuple[list[float], int]:
    starts = []
    durations = []
    for item in lifecycle:
        if item.action == "consistency_lost":
            starts.append(item.processing_time_s)
        elif item.action == "reinitialized" and starts:
            durations.append(item.processing_time_s - starts.pop(0))
    return durations, len(starts)


def _false_reset_count(scenario: StressScenario, lifecycle) -> int:
    outliers = [
        item
        for item in scenario.event_truth
        if item.delivered and item.is_outlier
    ]
    previous_confirmation = float("-inf")
    false_count = 0
    for item in lifecycle:
        if item.action in {"initialized", "reinitialized"}:
            previous_confirmation = item.processing_time_s
        elif item.action == "consistency_lost":
            explanatory = any(
                previous_confirmation
                <= truth.available_timestamp_s
                <= item.processing_time_s
                for truth in outliers
            )
            false_count += int(not explanatory)
    return false_count


def _run_variant(scenario: StressScenario, variant: RecoveryStudyVariant) -> dict[str, object]:
    processor = (
        CausalConfirmedRetardedTimeEKF(
            scenario.base_block.stations,
            scenario.events,
            estimator_variant="direct_bearing",
        )
        if variant.kind == "recovery"
        else _classic_processor(scenario, variant.kind)
    )
    publications = []
    updates = []
    runtime = 0.0
    for evaluation_time in EVALUATION_TIMES_S:
        publication = processor.advance_to(float(evaluation_time))
        publications.append(publication)
        updates.extend(publication.update_diagnostics)
        runtime += publication.total_runtime_s
    final = publications[-1]
    epoch_errors = [
        _state_metrics(item, scenario.base_block.truth_state, item.processing_time_s)
        for item in publications
    ]
    final_metrics = epoch_errors[-1]
    finite_epoch_errors = np.asarray(
        [float(item["position_error_m"]) for item in epoch_errors], dtype=float
    )
    finite_epoch_errors = finite_epoch_errors[np.isfinite(finite_epoch_errors)]
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
    if variant.kind == "recovery":
        used_ids = {item.event_id for item in final.event_uses}
        initialization_ids = set(final.initialization_event_ids)
        applied_ids = set(final.applied_event_ids)
        rejected_ids = set(final.rejected_event_ids)
        rejection_reasons = dict(final.rejection_reasons)
        lifecycle = final.lifecycle_diagnostics
        confirmed_flags = [item.confirmed for item in publications]
        first_confirmation = final.first_confirmation_time_s
        reset_count = final.reset_count
        durations, censored_recoveries = _recovery_durations(lifecycle)
        false_resets = _false_reset_count(scenario, lifecycle)
        state_generation = final.generation
        hypothesis_count = sum(
            item.action == "tentative_created" for item in final.hypothesis_diagnostics
        )
    else:
        initialization_ids = set(final.initialization_event_ids)
        applied_ids = set(final.applied_event_ids)
        used_ids = initialization_ids | applied_ids
        rejected_ids = set(final.rejected_event_ids)
        rejection_reasons = {
            item.event_id: item.reason for item in final.event_rejections
        }
        lifecycle = final.lifecycle_diagnostics
        confirmed_flags = [item.valid for item in publications]
        initialization_times = [
            item.processing_time_s
            for item in lifecycle
            if item.action in {"initialized", "reinitialized_after_conflict"}
        ]
        first_confirmation = min(initialization_times, default=float("nan"))
        reset_count = sum(item.action == "state_invalidated" for item in lifecycle)
        durations = []
        censored_recoveries = reset_count
        false_resets = 0
        state_generation = int(any(confirmed_flags))
        hypothesis_count = 1 if initialization_ids else 0
    finite_nis = np.asarray(
        [
            item.normalized_innovation_squared
            for item in updates
            if np.isfinite(item.normalized_innovation_squared)
        ],
        dtype=float,
    )
    unclassified_ids = delivered_ids - used_ids - rejected_ids
    return {
        "geometry": scenario.base_block.geometry,
        "profile": scenario.profile.name,
        "sequence_index": scenario.base_block.sequence_index,
        "base_seed": scenario.base_block.base_seed,
        "sequence_id": scenario.sequence_id,
        "variant": variant.name,
        "independent_unit": "whole_base_sequence",
        "epochs_within_sequence_are_dependent": True,
        "evaluation_epoch_count": len(publications),
        "final_valid": final.valid,
        "final_status": getattr(final, "status", "confirmed" if final.valid else "uninitialized"),
        "final_failure_reason": final.failure_reason or "",
        **final_metrics,
        "maximum_confirmed_epoch_position_error_m": (
            float(np.max(finite_epoch_errors)) if finite_epoch_errors.size else float("nan")
        ),
        "confirmed_epoch_fraction": float(np.mean(confirmed_flags)),
        "first_confirmation_time_s": first_confirmation,
        "state_generation": state_generation,
        "hypothesis_count": hypothesis_count,
        "reset_count": reset_count,
        "false_reset_count_evaluator_only": false_resets,
        "completed_recovery_count": len(durations),
        "censored_recovery_count": censored_recoveries,
        "mean_recovery_duration_s": (
            float(np.mean(durations)) if durations else float("nan")
        ),
        "maximum_recovery_duration_s": (
            float(np.max(durations)) if durations else float("nan")
        ),
        "delivered_event_count": len(delivered_ids),
        "delivered_clean_event_count": len(clean_ids),
        "delivered_outlier_event_count": len(outlier_ids),
        "initialization_event_count": len(initialization_ids),
        "applied_update_event_count": len(applied_ids),
        "rejected_event_count": len(rejected_ids),
        "unclassified_delivered_event_count": len(unclassified_ids),
        "detected_outlier_event_count_evaluator_only": len(rejected_ids & outlier_ids),
        "missed_outlier_event_count_evaluator_only": len(used_ids & outlier_ids),
        "false_rejected_clean_event_count_evaluator_only": len(rejected_ids & clean_ids),
        "initialization_event_ids_json": json.dumps(sorted(initialization_ids)),
        "applied_event_ids_json": json.dumps(sorted(applied_ids)),
        "rejected_event_ids_json": json.dumps(sorted(rejected_ids)),
        "rejection_reasons_json": json.dumps(rejection_reasons, sort_keys=True),
        "pre_update_nis_defined_count": len(finite_nis),
        "pre_update_nis_all_mean": (
            float(np.mean(finite_nis)) if finite_nis.size else float("nan")
        ),
        "runtime_s": runtime,
        "truth_used_by_estimator": False,
        "truth_by_id_count_evaluator_only": len(truth_by_id),
    }


def run_recovery_scenario(scenario: StressScenario) -> list[dict[str, object]]:
    return [_run_variant(scenario, variant) for variant in VARIANTS]


def _journal_state_fields(publication, truth) -> dict[str, object]:
    state = publication.state
    if state is None:
        return {
            "state_vector_json": "",
            "evaluator_only_position_error_m": float("nan"),
        }
    processing_time = float(publication.processing_time_s)
    return {
        "state_vector_json": json.dumps(state.vector.tolist()),
        "evaluator_only_position_error_m": float(
            np.linalg.norm(
                state.position_at(processing_time)
                - truth.position_at(processing_time)
            )
        ),
    }


def _journal_base(
    scenario: StressScenario,
    variant: RecoveryStudyVariant,
    publication,
) -> dict[str, object]:
    return {
        "case_id": (
            f"{scenario.base_block.geometry}:"
            f"{scenario.profile.name}:{scenario.base_block.sequence_index}"
        ),
        "geometry": scenario.base_block.geometry,
        "profile": scenario.profile.name,
        "sequence_index": scenario.base_block.sequence_index,
        "base_seed": scenario.base_block.base_seed,
        "variant": variant.name,
        "processing_time_s": publication.processing_time_s,
        "status": getattr(
            publication,
            "status",
            "confirmed" if publication.valid else "uninitialized",
        ),
        "valid": publication.valid,
        "generation": getattr(publication, "generation", int(publication.valid)),
        "truth_used_by_estimator": False,
        **_journal_state_fields(publication, scenario.base_block.truth_state),
    }


def generate_known_failure_journal() -> list[dict[str, object]]:
    """Return compact evaluator-annotated event logs for the two D2 failures."""

    rows: list[dict[str, object]] = []
    profile = next(
        item for item in default_stress_profiles() if item.name == "outlier_mild"
    )
    for geometry, sequence_index in KNOWN_FAILURE_CASES:
        block = generate_stress_base_block(
            geometry,
            sequence_index,
            base_seed=20260912,
        )
        scenario = generate_stress_scenario(block, profile)
        truth_by_id = {item.event_id: item for item in scenario.event_truth}
        event_by_id = {bearing_event_id(item): item for item in scenario.events}
        processing_times = sorted(
            {float(item.available_timestamp_s) for item in scenario.events}
        )
        for variant in VARIANTS:
            processor = (
                CausalConfirmedRetardedTimeEKF(
                    block.stations,
                    scenario.events,
                    estimator_variant="direct_bearing",
                )
                if variant.kind == "recovery"
                else _classic_processor(scenario, variant.kind)
            )
            for processing_time in (*processing_times, 14.5):
                publication = processor.advance_to(processing_time)
                base = _journal_base(scenario, variant, publication)
                for diagnostic in publication.new_lifecycle_diagnostics:
                    rows.append(
                        {
                            **base,
                            "record_kind": "lifecycle",
                            "action": diagnostic.action,
                            "event_id": "",
                            "station_id": "",
                            "accepted": "",
                            "reason": diagnostic.reason,
                            "nis": float("nan"),
                            "evaluator_only_is_outlier": "",
                            "event_ids_json": json.dumps(diagnostic.event_ids),
                            "preliminary_nis_json": "",
                            "final_nis_json": "",
                        }
                    )
                if variant.kind == "recovery":
                    for diagnostic in publication.new_hypothesis_diagnostics:
                        rows.append(
                            {
                                **base,
                                "record_kind": "hypothesis",
                                "action": diagnostic.action,
                                "event_id": "",
                                "station_id": "",
                                "accepted": diagnostic.action
                                == "hypothesis_confirmed",
                                "reason": diagnostic.reason,
                                "nis": float("nan"),
                                "evaluator_only_is_outlier": "",
                                "event_ids_json": json.dumps(
                                    diagnostic.construction_event_ids
                                    + diagnostic.confirmation_event_ids
                                ),
                                "preliminary_nis_json": json.dumps(
                                    diagnostic.preliminary_nis_values
                                ),
                                "final_nis_json": json.dumps(
                                    diagnostic.final_nis_values
                                ),
                            }
                        )
                else:
                    for diagnostic in publication.initialization_diagnostics:
                        rows.append(
                            {
                                **base,
                                "record_kind": "initialization",
                                "action": (
                                    "initialization_accepted"
                                    if diagnostic.succeeded
                                    else "initialization_rejected"
                                ),
                                "event_id": "",
                                "station_id": "",
                                "accepted": diagnostic.succeeded,
                                "reason": diagnostic.failure_reason or "",
                                "nis": float("nan"),
                                "evaluator_only_is_outlier": "",
                                "event_ids_json": json.dumps(
                                    diagnostic.used_event_ids
                                ),
                                "preliminary_nis_json": json.dumps(
                                    diagnostic.inlier_nis_values
                                ),
                                "final_nis_json": "",
                            }
                        )
                for update in publication.update_diagnostics:
                    event = event_by_id[update.event_id]
                    truth = truth_by_id[update.event_id]
                    rows.append(
                        {
                            **base,
                            "record_kind": "measurement",
                            "action": (
                                "update_accepted"
                                if update.update_applied
                                else "update_rejected"
                            ),
                            "event_id": update.event_id,
                            "station_id": event.station_id,
                            "accepted": update.update_applied,
                            "reason": update.failure_reason or "",
                            "nis": update.normalized_innovation_squared,
                            "evaluator_only_is_outlier": truth.is_outlier,
                            "event_ids_json": json.dumps([update.event_id]),
                            "preliminary_nis_json": "",
                            "final_nis_json": "",
                        }
                    )
            final = processor.publications[-1]
            rows.append(
                {
                    **_journal_base(scenario, variant, final),
                    "record_kind": "final",
                    "action": "final_publication",
                    "event_id": "",
                    "station_id": "",
                    "accepted": final.valid,
                    "reason": final.failure_reason or "",
                    "nis": float("nan"),
                    "evaluator_only_is_outlier": "",
                    "event_ids_json": "",
                    "preliminary_nis_json": "",
                    "final_nis_json": "",
                }
            )
    return rows


def _bootstrap_ci(pairs, statistic, *, coordinates: tuple[int, ...]) -> tuple[float, float]:
    rng = np.random.default_rng(
        np.random.SeedSequence([EVALUATION_SEED, *coordinates])
    )
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        values.append(statistic([pairs[int(index)] for index in indices]))
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return (
        tuple(np.percentile(finite, [2.5, 97.5], method="linear"))
        if finite.size
        else (float("nan"), float("nan"))
    )


def _rmse(rows, field: str) -> float:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


def _finite_mean(values) -> float:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def summarize_recovery_profiles(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["geometry"]), str(row["profile"]), str(row["variant"])), []
        ).append(row)
    result = []
    for geometry_index, geometry in enumerate(GEOMETRIES):
        for profile_index, profile in enumerate(default_stress_profiles()):
            baseline = groups[(geometry, profile.name, "c1_baseline")]
            d2 = groups[(geometry, profile.name, "d2_combined_published")]
            references = {
                "c1_baseline": {int(row["sequence_index"]): row for row in baseline},
                "d2_combined_published": {int(row["sequence_index"]): row for row in d2},
            }
            for variant_index, variant in enumerate(VARIANTS):
                selected = sorted(
                    groups[(geometry, profile.name, variant.name)],
                    key=lambda row: int(row["sequence_index"]),
                )
                count = len(selected)
                valid = [row for row in selected if _as_bool(row["final_valid"])]
                covered = sum(_as_bool(row["state_95_covered"]) for row in valid)
                valid_ci = _wilson_interval(len(valid), count)
                coverage_ci = _wilson_interval(covered, count)
                position = np.asarray(
                    [float(row["position_error_m"]) for row in valid], dtype=float
                )
                velocity = np.asarray(
                    [float(row["velocity_error_mps"]) for row in valid], dtype=float
                )
                first_confirmation = np.asarray(
                    [float(row["first_confirmation_time_s"]) for row in selected],
                    dtype=float,
                )
                first_confirmation = first_confirmation[
                    np.isfinite(first_confirmation)
                ]
                reset_sequence_count = sum(
                    int(row["reset_count"]) > 0 for row in selected
                )
                false_reset_count = sum(
                    int(row["false_reset_count_evaluator_only"])
                    for row in selected
                )
                summary = {
                    "geometry": geometry,
                    "profile": profile.name,
                    "variant": variant.name,
                    "independent_sequence_count": count,
                    "dependent_epoch_count": count * len(EVALUATION_TIMES_S),
                    "final_valid_count": len(valid),
                    "final_invalid_count": count - len(valid),
                    "conditional_error_sequence_count": len(valid),
                    "final_valid_fraction": len(valid) / count,
                    "final_valid_ci95_low": valid_ci[0],
                    "final_valid_ci95_high": valid_ci[1],
                    "conditional_position_rmse_m": _rmse(valid, "position_error_m"),
                    "conditional_position_p95_m": (
                        linear_percentile(position, 95) if position.size else float("nan")
                    ),
                    "conditional_position_maximum_m": (
                        float(np.max(position)) if position.size else float("nan")
                    ),
                    "conditional_velocity_rmse_mps": _rmse(valid, "velocity_error_mps"),
                    "conditional_velocity_p95_mps": (
                        linear_percentile(velocity, 95) if velocity.size else float("nan")
                    ),
                    "conditional_velocity_maximum_mps": (
                        float(np.max(velocity)) if velocity.size else float("nan")
                    ),
                    "conditional_position_error_gt_10m_fraction": (
                        float(np.mean(position > 10.0)) if position.size else float("nan")
                    ),
                    "conditional_position_error_gt_50m_fraction": (
                        float(np.mean(position > 50.0)) if position.size else float("nan")
                    ),
                    "conditional_state_95_coverage_fraction": (
                        covered / len(valid) if valid else float("nan")
                    ),
                    "state_95_covered_count": covered,
                    "unconditional_valid_and_covered_fraction": covered / count,
                    "unconditional_valid_and_covered_ci95_low": coverage_ci[0],
                    "unconditional_valid_and_covered_ci95_high": coverage_ci[1],
                    "mean_confirmed_epoch_fraction": float(
                        np.mean([float(row["confirmed_epoch_fraction"]) for row in selected])
                    ),
                    "mean_first_confirmation_time_s": _finite_mean(
                        float(row["first_confirmation_time_s"])
                        for row in selected
                    ),
                    "p95_first_confirmation_time_s": (
                        linear_percentile(first_confirmation, 95)
                        if first_confirmation.size
                        else float("nan")
                    ),
                    "mean_hypothesis_count": float(
                        np.mean([int(row["hypothesis_count"]) for row in selected])
                    ),
                    "sequence_with_reset_count": reset_sequence_count,
                    "sequence_with_reset_fraction": reset_sequence_count / count,
                    "total_reset_count": sum(int(row["reset_count"]) for row in selected),
                    "false_reset_count_evaluator_only": false_reset_count,
                    "false_reset_rate_per_sequence_evaluator_only": (
                        false_reset_count / count
                    ),
                    "completed_recovery_count": sum(
                        int(row["completed_recovery_count"]) for row in selected
                    ),
                    "censored_recovery_count": sum(
                        int(row["censored_recovery_count"]) for row in selected
                    ),
                    "mean_completed_recovery_duration_s": _finite_mean(
                        float(row["mean_recovery_duration_s"])
                        for row in selected
                    ),
                    "final_failure_reason_counts_json": json.dumps(
                        Counter(str(row["final_failure_reason"]) for row in selected if not _as_bool(row["final_valid"])),
                        sort_keys=True,
                    ),
                    "mean_runtime_s": float(np.mean([float(row["runtime_s"]) for row in selected])),
                    "p95_method": P95_METHOD,
                    "truth_used_by_estimator": False,
                    "selection_covariance_assumed_calibrated": False,
                    "diagnostic_position_thresholds_m": "10,50",
                }
                for reference_index, reference_name in enumerate(
                    ("c1_baseline", "d2_combined_published")
                ):
                    reference = references[reference_name]
                    pairs = [(row, reference[int(row["sequence_index"])]) for row in selected]
                    valid_difference = float(
                        np.mean(
                            [
                                int(_as_bool(left["final_valid"]))
                                - int(_as_bool(right["final_valid"]))
                                for left, right in pairs
                            ]
                        )
                    )
                    valid_ci_paired = _bootstrap_ci(
                        pairs,
                        lambda sample: float(
                            np.mean(
                                [
                                    int(_as_bool(left["final_valid"]))
                                    - int(_as_bool(right["final_valid"]))
                                    for left, right in sample
                                ]
                            )
                        ),
                        coordinates=(geometry_index, profile_index, variant_index, reference_index, 1),
                    )
                    rmse_difference = _rmse(selected, "position_error_m") - _rmse(
                        list(reference.values()), "position_error_m"
                    )
                    rmse_ci = _bootstrap_ci(
                        pairs,
                        lambda sample: _rmse([left for left, _ in sample], "position_error_m")
                        - _rmse([right for _, right in sample], "position_error_m"),
                        coordinates=(geometry_index, profile_index, variant_index, reference_index, 2),
                    )
                    prefix = f"paired_vs_{reference_name}"
                    summary[f"{prefix}_valid_fraction_difference"] = valid_difference
                    summary[f"{prefix}_valid_difference_ci95_low"] = valid_ci_paired[0]
                    summary[f"{prefix}_valid_difference_ci95_high"] = valid_ci_paired[1]
                    summary[f"{prefix}_position_rmse_difference_m"] = rmse_difference
                    summary[f"{prefix}_position_rmse_difference_ci95_low_m"] = rmse_ci[0]
                    summary[f"{prefix}_position_rmse_difference_ci95_high_m"] = rmse_ci[1]
                result.append(summary)
    return result


def _seed_row(block: StressBaseBlock) -> dict[str, object]:
    row: dict[str, object] = {
        "geometry": block.geometry,
        "geometry_index": block.geometry_index,
        "sequence_index": block.sequence_index,
        "base_seed": block.base_seed,
        "independent_unit": "whole_base_sequence",
    }
    for name, identity, seed in zip(
        MECHANISMS,
        block.provenance.identifiers,
        block.provenance.generated_seeds,
        strict=True,
    ):
        row[f"{name}_provenance"] = identity
        row[f"{name}_seed"] = seed
    return row


def _run_block(arguments):
    geometry, sequence_index, base_seed = arguments
    block = generate_stress_base_block(geometry, sequence_index, base_seed=base_seed)
    rows = []
    for profile in default_stress_profiles():
        rows.extend(run_recovery_scenario(generate_stress_scenario(block, profile)))
    return rows, _seed_row(block)


def audit_recovery_results(
    rows: Sequence[dict[str, object]],
    summaries: Sequence[dict[str, object]],
    seeds: Sequence[dict[str, object]],
    *,
    sequence_count: int,
) -> dict[str, int]:
    expected_rows = len(GEOMETRIES) * len(default_stress_profiles()) * len(VARIANTS) * sequence_count
    if len(rows) != expected_rows:
        raise AssertionError("unexpected recovery sequence row count")
    if len(summaries) != len(GEOMETRIES) * len(default_stress_profiles()) * len(VARIANTS):
        raise AssertionError("unexpected recovery summary row count")
    if len(seeds) != len(GEOMETRIES) * sequence_count:
        raise AssertionError("unexpected recovery provenance row count")
    seed_values = [int(row[f"{name}_seed"]) for row in seeds for name in MECHANISMS]
    provenance = [str(row[f"{name}_provenance"]) for row in seeds for name in MECHANISMS]
    if len(seed_values) != len(set(seed_values)) or len(provenance) != len(set(provenance)):
        raise AssertionError("recovery seed collision")
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["geometry"]), str(row["profile"]), int(row["sequence_index"])),
            [],
        ).append(row)
        if _as_bool(row["truth_used_by_estimator"]):
            raise AssertionError("truth leaked into estimator")
        delivered = int(row["delivered_event_count"])
        partition = (
            int(row["initialization_event_count"])
            + int(row["applied_update_event_count"])
            + int(row["rejected_event_count"])
            + int(row["unclassified_delivered_event_count"])
        )
        if delivered != partition:
            raise AssertionError("delivered event partition mismatch")
    if any(len(group) != len(VARIANTS) for group in groups.values()):
        raise AssertionError("paired variant group incomplete")
    return {
        "sequence_row_count": len(rows),
        "summary_row_count": len(summaries),
        "seed_row_count": len(seeds),
        "unique_mechanism_seed_count": len(set(seed_values)),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_initialization_recovery_study(
    output_directory: str | Path = "results",
    *,
    sequence_count: int = DEFAULT_SEQUENCE_COUNT,
    base_seed: int = EVALUATION_SEED,
    workers: int = 1,
    progress: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if sequence_count <= 0 or workers <= 0:
        raise ValueError("sequence_count and workers must be positive")
    tasks = [
        (geometry, sequence_index, base_seed)
        for geometry in GEOMETRIES
        for sequence_index in range(sequence_count)
    ]
    rows = []
    seeds = []
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    iterator = executor.map(_run_block, tasks, chunksize=1) if executor else map(_run_block, tasks)
    try:
        for completed, (block_rows, seed_row) in enumerate(iterator, start=1):
            rows.extend(block_rows)
            seeds.append(seed_row)
            if progress:
                print(f"recovery study blocks {completed}/{len(tasks)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    summaries = summarize_recovery_profiles(rows)
    audit_recovery_results(rows, summaries, seeds, sequence_count=sequence_count)
    output = Path(output_directory)
    _write_csv(output / "initialization_recovery_sequence_results.csv", rows)
    _write_csv(output / "initialization_recovery_summary.csv", summaries)
    _write_csv(output / "initialization_recovery_seed_provenance.csv", seeds)
    _write_csv(
        output / "initialization_recovery_failure_journal.csv",
        generate_known_failure_journal(),
    )
    return rows, summaries, seeds


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default="results")
    parser.add_argument("--sequence-count", type=int, default=DEFAULT_SEQUENCE_COUNT)
    parser.add_argument("--base-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    rows, summaries, seeds = run_initialization_recovery_study(
        args.output_directory,
        sequence_count=args.sequence_count,
        base_seed=args.base_seed,
        workers=args.workers,
        progress=args.progress,
    )
    print(json.dumps({"rows": len(rows), "summaries": len(summaries), "seeds": len(seeds), "elapsed_s": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "DEVELOPMENT_SEED",
    "EVALUATION_SEED",
    "SMOKE_SEED",
    "VARIANTS",
    "RecoveryStudyVariant",
    "audit_recovery_results",
    "generate_known_failure_journal",
    "run_initialization_recovery_study",
    "run_recovery_scenario",
    "summarize_recovery_profiles",
]
