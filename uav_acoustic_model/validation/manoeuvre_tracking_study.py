"""Frozen paired S7C-C bearing-level manoeuvre benchmark.

Truth generation and scores stay in this evaluator; both causal filters see
only the same immutable bearing events. See S7C_MANOEUVRE_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from estimators.retarded_ekf_manoeuvre import (
    CausalManoeuvreRetardedTimeEKF,
    ManoeuvreHistoryConfig,
)
from estimators.retarded_ekf_recovery import CausalConfirmedRetardedTimeEKF
from model.bearing_events import bearing_event_id
from model.measurements import BearingMeasurement
from simulation.manoeuvre_trajectory import BenchmarkManoeuvreTrajectory
from simulation.moving_source import solve_emission_time
from validation.retarded_ekf_stress_study import (
    _event_grid,
    generate_stress_base_block,
    spherical_exp_map_from_tangent,
)


DEVELOPMENT_SEED = 20260915
EVALUATION_SEED = 20260916
BOOTSTRAP_SEED = 20260917
DEVELOPMENT_SEQUENCE_COUNT = 2
EVALUATION_SEQUENCE_COUNT = 8
BOOTSTRAP_DRAWS = 500
QC_CANDIDATES_M2_S3 = (0.05, 0.25, 1.0)
TRUTH_KINDS = (
    "constant_velocity", "constant_acceleration_segment", "smooth_turn"
)
GEOMETRIES = ("informative", "poorly_conditioned")
PUBLICATION_TIMES_S = tuple(np.arange(0, 12.5 + 0.25, 0.5)) + (14.5,)
COVERAGE_THRESHOLD = float(chi2.ppf(0.95, 6))
RESULTS = Path(__file__).resolve().parents[1] / "results"


@dataclass(frozen=True)
class ManoeuvreScenario:
    geometry: str
    trajectory_kind: str
    sequence_index: int
    seed: int
    stations: tuple
    trajectory: BenchmarkManoeuvreTrajectory
    events: tuple[BearingMeasurement, ...]
    event_ids: tuple[str, ...]


def generate_manoeuvre_scenario(
    geometry: str, truth_kind: str, sequence_index: int, seed: int,
) -> ManoeuvreScenario:
    """Use disjoint D1 mechanism seeds and independent retarded truth solver."""
    if truth_kind not in TRUTH_KINDS:
        raise ValueError("unknown trajectory kind")
    block = generate_stress_base_block(geometry, sequence_index, base_seed=seed)
    trajectory = BenchmarkManoeuvreTrajectory(
        block.truth_state.position_at_reference_world_m,
        block.truth_state.velocity_world_mps, truth_kind,
    )
    sequence_id = f"s7cc-man-{seed}-{geometry}-{truth_kind}-{sequence_index}"
    covariance = np.diag(np.deg2rad([0.3, 0.5]) ** 2)
    events: list[BearingMeasurement] = []
    for index, (station_index, frame_index, reception) in enumerate(_event_grid()):
        station = block.stations[station_index]
        emission = solve_emission_time(reception, station.position_world_m, trajectory)
        displacement = trajectory.q(emission) - station.position_world_m
        direction = station.world_to_local_direction(displacement / np.linalg.norm(displacement))
        observed = spherical_exp_map_from_tangent(
            direction, block.nominal_tangent_errors_rad[index]
        )
        delay = 0.01 + block.delay_uniforms[index] * 0.41
        events.append(BearingMeasurement(
            station.station_id, sequence_id, frame_index, reception,
            reception + delay, observed, covariance, np.zeros(2),
            "direct_bearing", tangent_frame="measurement",
        ))
    return ManoeuvreScenario(
        geometry, truth_kind, sequence_index, seed, block.stations,
        trajectory, tuple(events), tuple(bearing_event_id(item) for item in events),
    )


def _period(time_s: float) -> str:
    if time_s < 5.0:
        return "pre"
    if time_s <= 9.0:
        return "during"
    return "post"


def _evaluate_publication(scenario: ManoeuvreScenario, variant: str, publication) -> dict:
    t = publication.processing_time_s
    truth = np.concatenate((scenario.trajectory.q(t), scenario.trajectory.v(t)))
    valid = bool(publication.valid and publication.state is not None)
    q_error = v_error = nees = float("nan")
    covered = False
    if valid:
        residual = publication.state.vector - truth
        q_error = float(np.linalg.norm(residual[:3]))
        v_error = float(np.linalg.norm(residual[3:]))
        matrix = publication.covariance_state
        try:
            np.linalg.cholesky(0.5 * (matrix + matrix.T))
            nees = float(residual @ np.linalg.solve(matrix, residual))
            covered = bool(nees <= COVERAGE_THRESHOLD)
        except np.linalg.LinAlgError:
            valid = False
    return {
        "geometry": scenario.geometry, "trajectory_kind": scenario.trajectory_kind,
        "sequence_index": scenario.sequence_index, "seed": scenario.seed,
        "variant": variant, "period": _period(t), "processing_time_s": t,
        "valid": int(valid), "confirmed": int(publication.confirmed),
        "status": publication.status, "failure_reason": publication.failure_reason or "",
        "position_error_m": q_error, "velocity_error_mps": v_error,
        "state_nees": nees, "valid_and_covered": int(valid and covered),
        "reset_count": publication.reset_count,
        "first_confirmation_time_s": publication.first_confirmation_time_s,
        "last_recovery_duration_s": publication.last_recovery_duration_s,
        "history_node_count": getattr(publication, "history_node_count", 0),
        "history_memory_bytes": getattr(publication, "history_memory_bytes", 0),
        "total_runtime_s": publication.total_runtime_s,
    }


def first_accepted_update_after_onset(
    publications, reception_by_id: dict[str, float], available_by_id: dict[str, float],
    onset_s: float,
) -> tuple[float, float]:
    """Return actual processing time and first user-visible publication time.

    The processing time is an update diagnostic property where available. The
    unchanged strict-CV diagnostic predates that field, so its event's causal
    availability timestamp is the equivalent group-processing time. The
    publication time is separately reported and may depend on the caller's
    publication schedule.
    """

    accepted: list[tuple[float, float]] = []
    for publication in publications:
        for diagnostic in publication.update_diagnostics:
            if (
                diagnostic.update_applied
                and reception_by_id[diagnostic.event_id] >= onset_s
            ):
                processing = float(getattr(
                    diagnostic, "processing_time_s",
                    available_by_id[diagnostic.event_id],
                ))
                accepted.append((processing, publication.processing_time_s))
    return min(accepted) if accepted else (float("nan"), float("nan"))


def run_paired_sequence(scenario: ManoeuvreScenario, alpha: float) -> tuple[list[dict], list[dict]]:
    """Identical events into accepted recovery and opt-in stochastic history."""
    config = ManoeuvreHistoryConfig(
        np.eye(3) * alpha, history_step_s=0.25, history_window_s=2.0,
        maximum_range_m=250.0, maximum_transport_delay_s=0.6,
    )
    filters = {
        "confirmed_recovery": CausalConfirmedRetardedTimeEKF(
            scenario.stations, scenario.events, estimator_variant="direct_bearing"
        ),
        "manoeuvre_history": CausalManoeuvreRetardedTimeEKF(
            scenario.stations, scenario.events, estimator_variant="direct_bearing",
            history_config=config,
        ),
    }
    rows: list[dict] = []
    sequence_rows: list[dict] = []
    for variant, estimator in filters.items():
        started = time.perf_counter()
        publications = [estimator.advance_to(t) for t in PUBLICATION_TIMES_S]
        runtime = time.perf_counter() - started
        rows.extend(_evaluate_publication(scenario, variant, item) for item in publications)
        last = publications[-1]
        reasons = Counter(dict(last.rejection_reasons).values())
        during_ids = {
            bearing_event_id(item) for item in scenario.events
            if 5.0 <= item.reception_center_timestamp_s <= 9.0
        }
        clean_rejected_during = sum(
            1 for identity in during_ids
            if dict(last.rejection_reasons).get(identity) == "pre_update_nis_gate"
        )
        reception_by_id = {
            bearing_event_id(event): event.reception_center_timestamp_s
            for event in scenario.events
        }
        available_by_id = {
            bearing_event_id(event): event.available_timestamp_s
            for event in scenario.events
        }
        accepted_processing, accepted_publication = first_accepted_update_after_onset(
            publications, reception_by_id, available_by_id, 5.0
        )
        sequence_rows.append({
            "geometry": scenario.geometry, "trajectory_kind": scenario.trajectory_kind,
            "sequence_index": scenario.sequence_index, "seed": scenario.seed,
            "variant": variant, "event_count": len(scenario.events),
            "initialization_event_count": len(last.initialization_event_ids),
            "accepted_update_count": len(last.applied_event_ids),
            "rejected_event_count": len(last.rejected_event_ids),
            "clean_rejected_during_count": clean_rejected_during,
            "during_event_count": len(during_ids),
            "reset_count": last.reset_count,
            "first_confirmation_time_s": last.first_confirmation_time_s,
            "last_recovery_duration_s": last.last_recovery_duration_s,
            "first_post_onset_accepted_update_processing_time_s": accepted_processing,
            "first_post_onset_accepted_update_first_publication_time_s": accepted_publication,
            "runtime_s": runtime,
            "maximum_history_memory_bytes": getattr(estimator, "maximum_history_memory_bytes", 0),
            "maximum_history_nodes": getattr(estimator, "maximum_history_node_count", 0),
            "rejection_reasons_json": json.dumps(reasons, sort_keys=True),
        })
    return rows, sequence_rows


def _finite_metric(rows: list[dict], name: str) -> np.ndarray:
    values = np.asarray([item[name] for item in rows], dtype=float)
    return values[np.isfinite(values)]


def _finite_mean(values) -> float:
    numbers = np.asarray(list(values), dtype=float)
    finite = numbers[np.isfinite(numbers)]
    return float(np.mean(finite)) if finite.size else float("nan")


def summarize(rows: list[dict], sequence_rows: list[dict], split: str, alpha: float) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["geometry"], row["trajectory_kind"], row["variant"], row["period"])].append(row)
    sequence_index = {
        (item["geometry"], item["trajectory_kind"], item["variant"], item["sequence_index"]): item
        for item in sequence_rows
    }
    summaries = []
    for (geometry, kind, variant, period), subset in sorted(groups.items()):
        sequence_keys = sorted({item["sequence_index"] for item in subset})
        position = _finite_metric(subset, "position_error_m")
        velocity = _finite_metric(subset, "velocity_error_mps")
        sequence_subset = [sequence_index[(geometry, kind, variant, key)] for key in sequence_keys]
        n = len(subset)
        valid_count = sum(item["valid"] for item in subset)
        def rmse(values):
            return float(np.sqrt(np.mean(values ** 2))) if values.size else float("nan")
        def p95(values):
            return float(np.percentile(values, 95)) if values.size else float("nan")
        def maximum(values):
            return float(np.max(values)) if values.size else float("nan")
        summaries.append({
            "split": split, "evaluation_seed": subset[0]["seed"],
            "qc_alpha_m2_s3": alpha, "geometry": geometry,
            "trajectory_kind": kind, "variant": variant, "period": period,
            "independent_sequence_count": len(sequence_keys),
            "dependent_publication_count": n, "valid_publication_count": valid_count,
            "valid_fraction": valid_count / n, "confirmed_fraction": sum(item["confirmed"] for item in subset) / n,
            "position_rmse_m_conditional": rmse(position),
            "position_p95_m_conditional": p95(position),
            "position_max_m_conditional": maximum(position),
            "velocity_rmse_mps_conditional": rmse(velocity),
            "velocity_p95_mps_conditional": p95(velocity),
            "velocity_max_mps_conditional": maximum(velocity),
            "coverage_conditional": sum(item["valid_and_covered"] for item in subset) / valid_count if valid_count else float("nan"),
            "valid_and_covered_fraction": sum(item["valid_and_covered"] for item in subset) / n,
            "mean_clean_rejected_during_per_sequence": float(np.mean([item["clean_rejected_during_count"] for item in sequence_subset])),
            "clean_nis_rejection_fraction_during": (
                sum(item["clean_rejected_during_count"] for item in sequence_subset)
                / sum(item["during_event_count"] for item in sequence_subset)
            ),
            "mean_reset_count": float(np.mean([item["reset_count"] for item in sequence_subset])),
            "mean_first_confirmation_time_s": _finite_mean(item["first_confirmation_time_s"] for item in sequence_subset),
            "mean_last_recovery_duration_s": _finite_mean(item["last_recovery_duration_s"] for item in sequence_subset),
            "mean_first_post_onset_accepted_update_processing_lag_s": _finite_mean(
                item["first_post_onset_accepted_update_processing_time_s"] - 5.0
                for item in sequence_subset
            ),
            "mean_first_post_onset_accepted_update_first_publication_lag_s": _finite_mean(
                item["first_post_onset_accepted_update_first_publication_time_s"] - 5.0
                for item in sequence_subset
            ),
            "mean_runtime_s_per_sequence": float(np.mean([item["runtime_s"] for item in sequence_subset])),
            "maximum_history_memory_bytes": max(item["maximum_history_memory_bytes"] for item in sequence_subset),
            "maximum_history_nodes": max(item["maximum_history_nodes"] for item in sequence_subset),
        })
    return summaries


def paired_sequence_bootstrap(rows: list[dict]) -> list[dict]:
    """Bootstrap whole sequence pairs on common-valid publication times."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    result = []
    for geometry in GEOMETRIES:
        for kind in TRUTH_KINDS:
            for period in ("pre", "during", "post"):
                sequence_blocks = []
                for sequence_index in range(EVALUATION_SEQUENCE_COUNT):
                    subset = [item for item in rows if item["geometry"] == geometry
                              and item["trajectory_kind"] == kind
                              and item["period"] == period
                              and item["sequence_index"] == sequence_index]
                    by_variant = {
                        variant: {item["processing_time_s"]: item for item in subset
                                  if item["variant"] == variant}
                        for variant in ("confirmed_recovery", "manoeuvre_history")
                    }
                    common = [
                        (by_variant["confirmed_recovery"][t], by_variant["manoeuvre_history"][t])
                        for t in by_variant["confirmed_recovery"]
                        if by_variant["confirmed_recovery"][t]["valid"]
                        and by_variant["manoeuvre_history"][t]["valid"]
                    ]
                    sequence_blocks.append(np.asarray([
                        [first["position_error_m"] ** 2, second["position_error_m"] ** 2]
                        for first, second in common
                    ]).reshape(-1, 2))
                def difference(indices):
                    values = np.concatenate([sequence_blocks[index] for index in indices])
                    if not len(values):
                        return float("nan")
                    return float(np.sqrt(np.mean(values[:, 1])) - np.sqrt(np.mean(values[:, 0])))
                point = difference(range(EVALUATION_SEQUENCE_COUNT))
                boot = [difference(rng.integers(0, EVALUATION_SEQUENCE_COUNT,
                                                EVALUATION_SEQUENCE_COUNT))
                        for _ in range(BOOTSTRAP_DRAWS)]
                finite = np.asarray(boot)[np.isfinite(boot)]
                result.append({
                    "geometry": geometry, "trajectory_kind": kind, "period": period,
                    "independent_paired_sequence_count": EVALUATION_SEQUENCE_COUNT,
                    "matched_valid_publication_count": sum(len(block) for block in sequence_blocks),
                    "manoeuvre_minus_recovery_position_rmse_m": point,
                    "paired_bootstrap_ci_low_m": float(np.percentile(finite, 2.5)) if len(finite) else float("nan"),
                    "paired_bootstrap_ci_high_m": float(np.percentile(finite, 97.5)) if len(finite) else float("nan"),
                    "bootstrap_draw_count": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "estimand": "conditional RMSE difference on matched valid timestamps",
                })
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty results")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_matrix(seed: int, sequence_count: int, alpha: float) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    sequence_rows: list[dict] = []
    for geometry in GEOMETRIES:
        for kind in TRUTH_KINDS:
            for sequence_index in range(sequence_count):
                scenario = generate_manoeuvre_scenario(geometry, kind, sequence_index, seed)
                frame, sequence = run_paired_sequence(scenario, alpha)
                rows.extend(frame)
                sequence_rows.extend(sequence)
    return rows, sequence_rows


def choose_qc_on_development() -> tuple[float, list[dict]]:
    diagnostics = []
    for alpha in QC_CANDIDATES_M2_S3:
        rows, _ = run_matrix(DEVELOPMENT_SEED, DEVELOPMENT_SEQUENCE_COUNT, alpha)
        during = [item for item in rows if item["period"] == "during" and
                  item["trajectory_kind"] != "constant_velocity" and
                  item["variant"] == "manoeuvre_history"]
        valid = _finite_metric(during, "position_error_m")
        coverage = len(valid) / len(during)
        penalized = np.asarray([
            item["position_error_m"] if np.isfinite(item["position_error_m"])
            else 50.0 for item in during
        ])
        score = float(np.sqrt(np.mean(penalized ** 2))) if coverage >= 0.5 else float("inf")
        diagnostics.append({"alpha_m2_s3": alpha, "development_selection_score_m": score,
                            "during_conditional_rmse_m": float(np.sqrt(np.mean(valid ** 2))) if valid.size else float("nan"),
                            "during_valid_fraction": coverage,
                            "development_seed": DEVELOPMENT_SEED,
                            "independent_base_block_count": len(GEOMETRIES) * DEVELOPMENT_SEQUENCE_COUNT,
                            "dependent_trajectory_run_count": len(GEOMETRIES) * len(TRUTH_KINDS) * DEVELOPMENT_SEQUENCE_COUNT})
    winner = min(diagnostics, key=lambda item: (item["development_selection_score_m"], item["alpha_m2_s3"]))
    if not np.isfinite(winner["development_selection_score_m"]):
        raise RuntimeError("development gate: no Qc candidate had >=50% coverage")
    return float(winner["alpha_m2_s3"]), diagnostics


def run_study() -> tuple[float, list[dict]]:
    alpha, development = choose_qc_on_development()
    _write_csv(RESULTS / "manoeuvre_development_selection.csv", development)
    rows, sequence_rows = run_matrix(EVALUATION_SEED, EVALUATION_SEQUENCE_COUNT, alpha)
    summaries = summarize(rows, sequence_rows, "evaluation", alpha)
    _write_csv(RESULTS / "manoeuvre_tracking_frames.csv", rows)
    _write_csv(RESULTS / "manoeuvre_tracking_sequences.csv", sequence_rows)
    _write_csv(RESULTS / "manoeuvre_tracking_summary.csv", summaries)
    _write_csv(RESULTS / "manoeuvre_paired_ci.csv", paired_sequence_bootstrap(rows))
    return alpha, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        for kind in TRUTH_KINDS:
            scenario = generate_manoeuvre_scenario("informative", kind, 0, DEVELOPMENT_SEED)
            rows, sequences = run_paired_sequence(scenario, QC_CANDIDATES_M2_S3[1])
            print(kind, len(rows), [item["accepted_update_count"] for item in sequences])
    else:
        alpha, summary = run_study()
        print("selected Qc alpha", alpha, "summary rows", len(summary))


if __name__ == "__main__":
    main()
