"""Post-hoc S8 calibration-transfer audit from committed CSVs only.

No waveform is loaded or synthesized. Evaluation truth is used solely after
truth-free replay, never for event construction or estimator decisions. The
zero-bias arm is a sensitivity analysis on already-viewed evaluation data,
not a newly selected operational calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from estimators.retarded_ekf_manoeuvre import (
    CausalManoeuvreRetardedTimeEKF, ManoeuvreHistoryConfig,
)
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.bearing_statistics import tangent_residual
from model.dynamic_state import ConstantVelocityState
from model.measurements import BearingMeasurement
from model.retarded_bearing import retarded_bearing_residual
from validation.independent_recordings_pilot import (
    INITIALIZATION_BATCH_OPTIMIZATION_BUDGET,
    INDEPENDENT_TRACKER_FRAME_STRIDE,
)
from validation.three_station_audio_tracking_study import (
    HISTORY_WINDOW_S, MAXIMUM_TRACKER_RANGE_M, MAXIMUM_TRANSPORT_DELAY_S,
    POSITION_COVERAGE_THRESHOLD, QC_ALPHA_M2_S3, pilot_stations,
    trajectory_for_audio_pilot,
)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "s8_tracking_feasibility_"
EVALUATION_KEY = (
    "source_model_comparison", "paired_session_id", "snr_db",
    "estimator_variant",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _vector(row: dict[str, str], prefix: str, length: int) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_{index}"]) for index in range(length)])


def _calibration_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    result = {
        (row["source_model_comparison"], row["station_id"], row["estimator_variant"]): row
        for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("duplicate pooled calibration keys")
    return result


def _covariance(row: dict[str, str]) -> np.ndarray:
    return np.asarray([
        [float(row["covariance_00_rad2"]), float(row["covariance_01_rad2"])],
        [float(row["covariance_01_rad2"]), float(row["covariance_11_rad2"])],
    ])


def _bias(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row["bias_az_arc_rad"]), float(row["bias_el_arc_rad"])])


def assert_calibration_session_grain_available(rows: list[dict[str, str]]) -> None:
    """Reject attempts to infer leave-one-session-out fits from pooled moments.

    A pooled mean/covariance and a JSON list of two IDs do not determine either
    session's mean, within-session covariance, or tail/outlier contribution.
    """

    required = {"source_session_id", "residual_rad_0", "residual_rad_1"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "session-grain calibration residuals were not persisted; "
            "leave-one-session-out and calibration outlier decomposition are not identifiable"
        )


def contract_audit(bearings: list[dict[str, str]]) -> dict[str, object]:
    """Independently compare serialized residuals and tracker residual at truth."""

    valid = [row for row in bearings if row["valid"] == "True"]
    # Stratified deterministic sample across source/model/station/method/SNR.
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in valid:
        key = tuple(row[field] for field in (
            "source_model_comparison", "paired_session_id", "snr_db",
            "station_id", "estimator_variant",
        ))
        groups[key].append(row)
    stations = {station.station_id: station for station in pilot_stations()}
    true_states = {}
    serialization_error = []
    tracker_error = []
    truth_direction_error = []
    for subset in groups.values():
        for row in (subset[0], subset[len(subset) // 2], subset[-1]):
            # The historical exporter used sequence_index=0 for each
            # independent source session; physical offset is encoded by
            # configuration_index = 2*session_index + snr_index.
            physical_session_index = int(row["configuration_index"]) // 2
            if physical_session_index not in true_states:
                trajectory = trajectory_for_audio_pilot("constant_velocity", physical_session_index)
                true_states[physical_session_index] = ConstantVelocityState(
                    trajectory.q(0.0), trajectory.v(0.0), 0.0
                )
            true_state = true_states[physical_session_index]
            truth = _vector(row, "truth_local", 3)
            estimate = _vector(row, "estimate_local", 3)
            saved = _vector(row, "residual_rad", 2)
            recomputed = tangent_residual(truth, estimate)
            serialization_error.append(float(np.max(np.abs(recomputed - saved))))
            measurement = BearingMeasurement(
                station_id=row["station_id"], sequence_id=row["sequence_id"],
                frame_index=int(row["frame_index"]),
                reception_center_timestamp_s=float(row["frame_center_reception_time_s"]),
                available_timestamp_s=float(row["available_timestamp_s"]),
                direction_local=estimate, covariance_tangent_rad2=np.eye(2),
                calibration_bias_tangent_rad=np.zeros(2),
                estimator_variant=row["estimator_variant"], tangent_frame="prediction",
            )
            tracker_residual = retarded_bearing_residual(
                true_state, stations[row["station_id"]], measurement
            )
            tracker_error.append(float(np.max(np.abs(tracker_residual - saved))))
            from model.retarded_bearing import predict_retarded_bearing_measurement
            predicted = predict_retarded_bearing_measurement(
                true_state, stations[row["station_id"]], measurement
            )
            truth_direction_error.append(float(np.max(np.abs(predicted.direction_local - truth))))
    return {
        "sampled_valid_rows": len(serialization_error),
        "sampled_group_count": len(groups),
        "maximum_serialized_residual_disagreement_rad": max(serialization_error),
        "maximum_tracker_residual_at_truth_disagreement_rad": max(tracker_error),
        "maximum_truth_direction_disagreement": max(truth_direction_error),
        "residual_convention": "Log_prediction(measurement), azimuth/elevation orthonormal arc radians",
        "tracker_calibration_operation": "residual_minus_calibration_bias",
        "calibration_tangent_frame": "prediction",
    }


def evaluation_residual_summaries(
    bearings: list[dict[str, str]], calibration: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, object]]:
    """Describe held-out residual distributions; never refit operational R."""

    groups: dict[tuple[str, ...], list[np.ndarray]] = defaultdict(list)
    for row in bearings:
        if row["valid"] != "True":
            continue
        key = tuple(row[field] for field in (
            "source_model_comparison", "paired_session_id", "snr_db",
            "station_id", "estimator_variant",
        ))
        groups[key].append(_vector(row, "residual_rad", 2))
    summaries = []
    for key, values in sorted(groups.items()):
        source, session, snr, station, method = key
        residuals = np.asarray(values)
        lengths = np.linalg.norm(residuals, axis=1)
        cutoff = float(np.percentile(lengths, 95))
        central = residuals[lengths <= cutoff]
        pooled = calibration[(source, station, method)]
        pooled_bias = _bias(pooled)
        pooled_covariance = _covariance(pooled)
        mean = np.mean(residuals, axis=0)
        empirical_covariance = np.cov(residuals.T)
        summaries.append({
            "split": "evaluation_posthoc_diagnostic",
            "source_model_comparison": source, "paired_session_id": session,
            "snr_db": snr, "station_id": station, "estimator_variant": method,
            "dependent_valid_frame_count": len(residuals),
            "mean_az_arc_rad": mean[0], "mean_el_arc_rad": mean[1],
            "pooled_calibration_bias_az_arc_rad": pooled_bias[0],
            "pooled_calibration_bias_el_arc_rad": pooled_bias[1],
            "mean_minus_pooled_bias_norm_deg": np.rad2deg(np.linalg.norm(mean - pooled_bias)),
            "raw_mean_norm_deg": np.rad2deg(np.linalg.norm(mean)),
            "median_geodesic_error_deg": np.rad2deg(np.median(lengths)),
            "p95_geodesic_error_deg": np.rad2deg(cutoff),
            "fraction_error_gt_30deg": float(np.mean(np.rad2deg(lengths) > 30)),
            "trimmed_95pct_mean_az_arc_rad": np.mean(central, axis=0)[0],
            "trimmed_95pct_mean_el_arc_rad": np.mean(central, axis=0)[1],
            "trimmed_95pct_mean_shift_deg": np.rad2deg(np.linalg.norm(mean - np.mean(central, axis=0))),
            "empirical_covariance_trace_rad2": float(np.trace(empirical_covariance)),
            "trimmed_95pct_covariance_trace_rad2": float(np.trace(np.cov(central.T))),
            "pooled_calibration_covariance_trace_rad2": float(np.trace(pooled_covariance)),
            "frame_count_is_independent_trial_count": False,
        })
    return summaries


def calibration_pool_comparison(
    current_rows: list[dict[str, str]], historical_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Compare two *pooled* fits; this does not recover session contributions."""

    historical = _calibration_lookup(historical_rows)
    result = []
    for row in current_rows:
        key = (row["source_model_comparison"], row["station_id"], row["estimator_variant"])
        old = historical[key]
        current_bias, old_bias = _bias(row), _bias(old)
        current_covariance, old_covariance = _covariance(row), _covariance(old)
        result.append({
            "source_model_comparison": key[0], "station_id": key[1],
            "estimator_variant": key[2],
            "current_duration_s": 4.5, "historical_duration_s": 2.0,
            "current_pooled_bias_norm_deg": np.rad2deg(np.linalg.norm(current_bias)),
            "historical_pooled_bias_norm_deg": np.rad2deg(np.linalg.norm(old_bias)),
            "pooled_bias_duration_change_deg": np.rad2deg(np.linalg.norm(current_bias - old_bias)),
            "current_pooled_covariance_trace_rad2": float(np.trace(current_covariance)),
            "historical_pooled_covariance_trace_rad2": float(np.trace(old_covariance)),
            "current_over_historical_trace_ratio": float(
                np.trace(current_covariance) / np.trace(old_covariance)
            ),
            "calibration_session_decomposition_identifiable": False,
        })
    return result


def _events(
    rows: list[dict[str, str]], calibration: dict[tuple[str, str, str], dict[str, str]],
    *, zero_bias: bool,
) -> tuple[BearingMeasurement, ...]:
    """Use only observable saved fields in the truth-free tracker payload."""

    result = []
    for row in rows:
        if int(row["frame_index"]) % INDEPENDENT_TRACKER_FRAME_STRIDE:
            continue
        common = dict(
            station_id=row["station_id"], sequence_id=row["sequence_id"],
            frame_index=int(row["frame_index"]),
            reception_center_timestamp_s=float(row["frame_center_reception_time_s"]),
            available_timestamp_s=float(row["available_timestamp_s"]),
            estimator_variant=row["estimator_variant"],
            quality_metadata=json.loads(row["quality_metadata_json"]),
        )
        if row["valid"] == "True":
            fit = calibration[(row["source_model_comparison"], row["station_id"], row["estimator_variant"])]
            result.append(BearingMeasurement(
                **common, direction_local=_vector(row, "estimate_local", 3),
                covariance_tangent_rad2=_covariance(fit),
                calibration_bias_tangent_rad=(np.zeros(2) if zero_bias else _bias(fit)),
                tangent_frame="prediction",
            ))
        else:
            result.append(BearingMeasurement.invalid(
                **common, invalid_reason=row["invalid_reason"]
            ))
    return tuple(result)


def replay_saved_stream(
    rows: list[dict[str, str]], calibration: dict[tuple[str, str, str], dict[str, str]],
    truth_rows: list[dict[str, str]], *, zero_bias: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Replay one saved stream; attach saved evaluator truth only afterwards."""

    if not rows:
        raise ValueError("empty saved stream")
    key = tuple(rows[0][field] for field in EVALUATION_KEY)
    if any(tuple(row[field] for field in EVALUATION_KEY) != key for row in rows):
        raise ValueError("mixed evaluation streams")
    events = _events(rows, calibration, zero_bias=zero_bias)
    method = key[-1]
    estimator = CausalManoeuvreRetardedTimeEKF(
        pilot_stations(), events, estimator_variant=method,
        history_config=ManoeuvreHistoryConfig(
            np.eye(3) * QC_ALPHA_M2_S3, history_step_s=0.25,
            history_window_s=HISTORY_WINDOW_S,
            maximum_range_m=MAXIMUM_TRACKER_RANGE_M,
            maximum_transport_delay_s=MAXIMUM_TRANSPORT_DELAY_S,
        ),
        recovery_config=InitializationRecoveryConfig(
            confirmation_reception_span_s=0.05,
            maximum_confirmation_failures=20,
            maximum_confirmation_events=180,
            maximum_initialization_buffer_events=360,
            maximum_batch_optimizations_per_generation=INITIALIZATION_BATCH_OPTIMIZATION_BUDGET,
        ),
    )
    publications = [
        estimator.advance_to(epoch)
        for epoch in sorted({event.available_timestamp_s for event in events})
    ]
    truth = {round(float(row["processing_time_s"]), 12): row for row in truth_rows}
    if len(truth) != len(publications):
        raise ValueError("saved truth publication grid differs from replay grid")
    position_errors = []
    velocity_errors = []
    covered = []
    valid_count = 0
    updates = []
    for publication in publications:
        saved = truth[round(publication.processing_time_s, 12)]
        updates.extend(publication.update_diagnostics)
        if publication.valid and publication.state is not None:
            valid_count += 1
            vector = publication.state.vector
            actual = np.asarray([
                float(saved[f"truth_position_{axis}_m"]) for axis in "xyz"
            ] + [
                float(saved[f"truth_velocity_{axis}_mps"]) for axis in "xyz"
            ]) if "truth_velocity_x_mps" in saved else None
            if actual is None:
                # Historical tracking CSV persists only truth position;
                # evaluator-only CV velocity is reconstructed from the frozen
                # published trajectory, never supplied to the estimator.
                trajectory = trajectory_for_audio_pilot(
                    "constant_velocity", int(rows[0]["configuration_index"]) // 2
                )
                actual = np.r_[
                    [float(saved[f"truth_position_{axis}_m"]) for axis in "xyz"],
                    trajectory.v(publication.processing_time_s),
                ]
            error = vector - actual
            position_errors.append(float(np.linalg.norm(error[:3])))
            velocity_errors.append(float(np.linalg.norm(error[3:])))
            covariance = publication.covariance_state
            try:
                nees = float(error @ np.linalg.solve(covariance, error))
            except np.linalg.LinAlgError:
                nees = float("inf")
            covered.append(nees <= POSITION_COVERAGE_THRESHOLD)
    final = publications[-1]
    common = dict(zip(EVALUATION_KEY, key))
    summary = {
        **common,
        "comparison_scope": "posthoc_previously_viewed_evaluation_no_algorithm_selection",
        "calibration_variant": "bias_zero_same_R" if zero_bias else "published_pooled_bias_and_R",
        "saved_event_count": len(events),
        "publication_count": len(publications),
        "confirmed_publication_count": sum(item.confirmed for item in publications),
        "valid_publication_count": valid_count,
        "final_confirmed": bool(final.confirmed),
        "final_valid": bool(final.valid),
        "first_confirmation_time_s": final.first_confirmation_time_s,
        "accepted_update_count": sum(item.update_applied for item in updates),
        "rejected_update_count": sum(not item.update_applied for item in updates),
        "position_rmse_m_conditional": (
            float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else float("nan")
        ),
        "position_p95_m_conditional": (
            float(np.percentile(position_errors, 95)) if position_errors else float("nan")
        ),
        "velocity_rmse_mps_conditional": (
            float(np.sqrt(np.mean(np.square(velocity_errors)))) if velocity_errors else float("nan")
        ),
        "coverage_conditional": float(np.mean(covered)) if covered else float("nan"),
        "valid_and_covered_publication_fraction": sum(covered) / len(publications),
        "failure_reason": final.failure_reason or "",
        "batch_optimization_count": final.batch_optimization_count,
        "batch_optimization_budget": final.batch_optimization_budget,
        "candidate_reason_counts_json": json.dumps(Counter(
            item.reason for item in estimator.candidate_diagnostics
        ), sort_keys=True),
        "batch_fit_reason_counts_json": json.dumps(Counter(
            f"{item.phase}:{item.reason}" for item in estimator.batch_fit_diagnostics
        ), sort_keys=True),
        "hypothesis_reason_counts_json": json.dumps(Counter(
            f"{item.action}:{item.reason}" for item in final.hypothesis_diagnostics
        ), sort_keys=True),
    }
    diagnostics = []
    for item in estimator.candidate_diagnostics:
        diagnostics.append({
            **common, "calibration_variant": summary["calibration_variant"],
            "diagnostic_kind": "construction_candidate", "processing_time_s": item.processing_time_s,
            "reason": item.reason,
            "construction_event_ids_json": json.dumps(item.construction_event_ids),
            "geometric_rank": item.geometric_rank,
            "batch_failure_reason": item.batch_failure_reason or "",
            "batch_local_rank": item.batch_local_rank,
            "batch_scaled_condition_number": item.batch_scaled_condition_number,
            "batch_maximum_angular_residual_rad": item.batch_maximum_angular_residual_rad,
            "batch_scaled_projected_kkt_residual": item.batch_scaled_projected_kkt_residual,
            "confirmation_event_ids_json": "[]", "confirmation_nis_json": "[]",
            "fit_phase": "", "fit_count_after_attempt": "",
        })
    for item in estimator.batch_fit_diagnostics:
        diagnostics.append({
            **common, "calibration_variant": summary["calibration_variant"],
            "diagnostic_kind": "batch_fit", "processing_time_s": item.processing_time_s,
            "reason": item.reason,
            "construction_event_ids_json": json.dumps(item.event_ids),
            "geometric_rank": "", "batch_failure_reason": item.batch_failure_reason or "",
            "batch_local_rank": item.local_observability_rank,
            "batch_scaled_condition_number": item.scaled_condition_number,
            "batch_maximum_angular_residual_rad": item.maximum_angular_residual_rad,
            "batch_scaled_projected_kkt_residual": item.scaled_projected_kkt_residual,
            "confirmation_event_ids_json": "[]", "confirmation_nis_json": "[]",
            "fit_phase": item.phase,
            "fit_count_after_attempt": item.fit_count_after_attempt,
        })
    for item in final.hypothesis_diagnostics:
        diagnostics.append({
            **common, "calibration_variant": summary["calibration_variant"],
            "diagnostic_kind": item.action, "processing_time_s": item.processing_time_s,
            "reason": item.reason,
            "construction_event_ids_json": json.dumps(item.construction_event_ids),
            "geometric_rank": "", "batch_failure_reason": "",
            "batch_local_rank": item.local_observability_rank,
            "batch_scaled_condition_number": item.scaled_condition_number,
            "batch_maximum_angular_residual_rad": "",
            "batch_scaled_projected_kkt_residual": "",
            "confirmation_event_ids_json": json.dumps(item.confirmation_event_ids),
            "confirmation_nis_json": json.dumps(item.confirmation_nis_values),
            "fit_phase": "", "fit_count_after_attempt": "",
        })
    return summary, diagnostics


def run_analysis(results_dir: Path = ROOT / "results", *, progress: bool = False) -> dict[str, object]:
    calibration_rows = _read(results_dir / f"{PREFIX}calibration.csv")
    historical_calibration_rows = _read(results_dir / "independent_recordings_calibration.csv")
    bearings = _read(results_dir / f"{PREFIX}bearing_results.csv")
    historical_tracking = _read(results_dir / f"{PREFIX}tracking_results.csv")
    historical_sessions = _read(results_dir / f"{PREFIX}session_results.csv")
    calibration = _calibration_lookup(calibration_rows)
    if not all(row["split"] == "evaluation" for row in bearings):
        raise ValueError("saved bearing CSV must contain evaluation only")
    if len({tuple(row[field] for field in EVALUATION_KEY) for row in bearings}) != 16:
        raise ValueError("expected 16 previously published evaluation streams")
    contract = contract_audit(bearings)
    summaries = evaluation_residual_summaries(bearings, calibration)
    pool_comparison = calibration_pool_comparison(calibration_rows, historical_calibration_rows)
    grouped_bearings: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    grouped_tracking: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in bearings:
        grouped_bearings[tuple(row[field] for field in EVALUATION_KEY)].append(row)
    for row in historical_tracking:
        grouped_tracking[tuple(row[field] for field in EVALUATION_KEY)].append(row)
    replay_rows = []
    diagnostics = []
    for key in sorted(grouped_bearings):
        for zero_bias in (False, True):
            summary, detail = replay_saved_stream(
                grouped_bearings[key], calibration, grouped_tracking[key],
                zero_bias=zero_bias,
            )
            replay_rows.append(summary)
            diagnostics.extend(detail)
        if progress:
            print("replayed", key, flush=True)
    saved_session_lookup = {
        tuple(row[field] for field in EVALUATION_KEY): row
        for row in historical_sessions
    }
    maximum_position_replay_difference_m = 0.0
    for row in replay_rows:
        if row["calibration_variant"] != "published_pooled_bias_and_R":
            continue
        key = tuple(str(row[field]) for field in EVALUATION_KEY)
        saved = saved_session_lookup[key]
        if row["final_confirmed"] != (saved["final_confirmed"] == "True"):
            raise AssertionError(f"published confirmation not reproduced for {key}")
        if row["failure_reason"] != saved["failure_reason"]:
            raise AssertionError(f"published failure reason not reproduced for {key}")
        if int(row["accepted_update_count"]) != int(saved["accepted_update_count"]):
            raise AssertionError(f"published update count not reproduced for {key}")
        if row["final_confirmed"]:
            difference = abs(float(row["position_rmse_m_conditional"]) - float(saved["position_rmse_m_conditional"]))
            maximum_position_replay_difference_m = max(maximum_position_replay_difference_m, difference)
    if maximum_position_replay_difference_m > 1e-8:
        raise AssertionError("published conditional position RMSE not reproduced")
    _write(results_dir / "s8_calibration_transfer_evaluation_residuals.csv", summaries)
    _write(results_dir / "s8_calibration_transfer_pool_comparison.csv", pool_comparison)
    _write(results_dir / "s8_calibration_transfer_replay.csv", replay_rows)
    _write(results_dir / "s8_calibration_transfer_hypothesis_diagnostics.csv", diagnostics)
    (results_dir / "s8_calibration_transfer_contract_audit.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "contract": contract,
        "evaluation_residual_group_count": len(summaries),
        "pool_comparison_row_count": len(pool_comparison),
        "replay_arm_count": len(replay_rows),
        "hypothesis_diagnostic_count": len(diagnostics),
        "calibration_session_grain_available": False,
        "maximum_published_position_replay_difference_m": maximum_position_replay_difference_m,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.results_dir, progress=args.progress), indent=2, sort_keys=True))
