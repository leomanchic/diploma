"""S8 paired benchmark with independent recorded-source sessions.

The unit of source-data independence is the original recording session, not a
frame, fragment or transcode.  Recorded and random-broadband members of each
pair share trajectory, timestamps and standard-normal AWGN draws.  Separate
calibration pools are frozen before held-out evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from estimators.retarded_ekf_manoeuvre import CausalManoeuvreRetardedTimeEKF, ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.bearing_events import bearing_event_id
from model.dynamic_state import ConstantVelocityState
from model.measurements import BearingMeasurement
from model.retarded_bearing import predict_retarded_bearing
from simulation.continuous_stream import reception_time_grid
from simulation.fractional_delay import DEFAULT_FIR_LENGTH
from simulation.multistation_audio import (
    MultistationAudioStream, _common_source_support, synthesize_multistation_audio,
)
from simulation.recorded_source import (
    RecordedSourceClip,
    load_recorded_source_clip,
    load_recorded_source_manifest,
    recorded_source_split_audit,
)
from validation.three_station_audio_tracking_study import (
    AudioPilotConfig,
    ESTIMATOR_VARIANTS,
    FRAME_LENGTH,
    HOP_LENGTH,
    QC_ALPHA_M2_S3,
    HISTORY_WINDOW_S,
    MAXIMUM_TRACKER_RANGE_M,
    MAXIMUM_TRANSPORT_DELAY_S,
    MODELED_PROCESSING_DELAY_S,
    RECEPTION_START_TIME_S,
    RESULTS,
    SOURCE_MAXIMUM_FREQUENCY_HZ,
    STATION_DELIVERY_DELAY_S,
    _calibration_rows,
    _csv_bearing_row,
    _write_csv,
    bearing_measurements_from_records,
    calibrate_audio_bearings,
    extract_audio_bearing_records,
    pilot_stations,
    run_tracker,
    trajectory_for_audio_pilot,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "recorded_sources" / "manifest.json"
CALIBRATION_BASE_SEED = 20260923
EVALUATION_BASE_SEED = 20260924
SMOKE_BASE_SEED = 20260925
DURATION_S = 4.5
SNR_LEVELS_DB = (-6.0, 10.0)
SOURCE_MODELS = ("recorded_source_approximation", "random_broadband")
RESULT_SCOPE = "held_out_independent_recording_tracking_feasibility"
INDEPENDENT_TRACKER_FRAME_STRIDE = 64
# Frozen before the corrected evaluation. Count actual nonlinear batch fits,
# including consensus refits; this is not a wall-clock timeout.
INITIALIZATION_BATCH_OPTIMIZATION_BUDGET = 4
RESULT_PREFIX = "s8_tracking_feasibility_"


def ideal_bearing_schedule(duration_s: float) -> tuple[BearingMeasurement, ...]:
    """Exact retarded bearings on the audio frame/reception/availability grid.

    Used solely as a positive scheduling control, before audio processing.
    Truth is used here to *generate* measurements, never by the tracker.
    """

    stations = pilot_stations()
    trajectory = trajectory_for_audio_pilot("constant_velocity", 0)
    state = ConstantVelocityState(trajectory.q(0.0), trajectory.v(0.0), 0.0)
    sample_count = reception_time_grid(
        RECEPTION_START_TIME_S, duration_s, 48_000.0
    ).size
    frame_count = 1 + (sample_count - FRAME_LENGTH) // HOP_LENGTH
    if frame_count < 1:
        raise ValueError("duration is shorter than one frame")
    covariance = np.diag(np.deg2rad([0.3, 0.5]) ** 2)
    events = []
    for frame_index in range(0, frame_count, INDEPENDENT_TRACKER_FRAME_STRIDE):
        start_sample = frame_index * HOP_LENGTH
        center = RECEPTION_START_TIME_S + (
            start_sample + (FRAME_LENGTH - 1) / 2
        ) / 48_000.0
        end = RECEPTION_START_TIME_S + (
            start_sample + FRAME_LENGTH - 1
        ) / 48_000.0
        for station in stations:
            direction = predict_retarded_bearing(state, station, center).direction_local
            events.append(BearingMeasurement(
                station_id=station.station_id,
                sequence_id="s8-exact-schedule-control",
                frame_index=frame_index,
                reception_center_timestamp_s=center,
                available_timestamp_s=(
                    end + MODELED_PROCESSING_DELAY_S
                    + STATION_DELIVERY_DELAY_S[station.station_id]
                ),
                direction_local=direction,
                covariance_tangent_rad2=covariance,
                calibration_bias_tangent_rad=np.zeros(2),
                estimator_variant="direct_bearing",
            ))
    return tuple(events)


def run_ideal_schedule_control(
    duration_s: float, *, batch_optimization_budget: int = INITIALIZATION_BATCH_OPTIMIZATION_BUDGET,
) -> dict[str, object]:
    """Check causal confirmation and later corrections on perfect data."""

    events = ideal_bearing_schedule(duration_s)
    estimator = CausalManoeuvreRetardedTimeEKF(
        pilot_stations(), events, estimator_variant="direct_bearing",
        history_config=ManoeuvreHistoryConfig(
            np.eye(3) * QC_ALPHA_M2_S3,
            history_step_s=0.25,
            history_window_s=HISTORY_WINDOW_S,
            maximum_range_m=MAXIMUM_TRACKER_RANGE_M,
            maximum_transport_delay_s=MAXIMUM_TRANSPORT_DELAY_S,
        ),
        recovery_config=InitializationRecoveryConfig(
            confirmation_reception_span_s=0.05,
            maximum_confirmation_failures=20,
            maximum_confirmation_events=180,
            maximum_initialization_buffer_events=360,
            maximum_batch_optimizations_per_generation=(
                batch_optimization_budget
            ),
        ),
    )
    updates = []
    publications = []
    for timestamp in sorted({event.available_timestamp_s for event in events}):
        publication = estimator.advance_to(timestamp)
        publications.append(publication)
        updates.extend(publication.update_diagnostics)
    final = publications[-1]
    frame_by_event = {bearing_event_id(event): event.frame_index for event in events}
    accepted_groups = {
        frame_by_event[item.event_id] for item in updates if item.update_applied
    }
    return {
        "duration_s": duration_s,
        "frame_group_count": len({event.frame_index for event in events}),
        "event_count": len(events),
        "confirmed": bool(final.confirmed),
        "first_confirmation_time_s": final.first_confirmation_time_s,
        "accepted_update_count": sum(item.update_applied for item in updates),
        "accepted_post_init_frame_group_count": len(accepted_groups),
        "rejected_update_count": sum(not item.update_applied for item in updates),
        "failure_reason": final.failure_reason or "",
        "batch_optimization_count": final.batch_optimization_count,
    }


def recording_support_preflight(duration_s: float = DURATION_S) -> list[dict[str, object]]:
    """Require the real clip to cover emission times plus FIR guards; no repeat."""

    reception = reception_time_grid(RECEPTION_START_TIME_S, duration_s, 48_000.0)
    stations = pilot_stations()
    audit = []
    for split in ("calibration", "evaluation"):
        for session_index, recording_id in enumerate(recording_ids_for_split(split)):
            clip = _clip(recording_id, split)
            source_start, needed = _common_source_support(
                reception, stations,
                trajectory_for_audio_pilot("constant_velocity", session_index),
                clip.sampling_rate_hz, 343.0, DEFAULT_FIR_LENGTH,
            )
            if needed > clip.samples.size:
                raise ValueError(
                    f"{recording_id}: selected interval has {clip.samples.size} "
                    f"samples but emission/FIR support requires {needed}"
                )
            audit.append({
                "split": split, "recording_id": recording_id,
                "selected_interval_samples": clip.samples.size,
                "required_source_samples_including_fir_guard": needed,
                "remaining_samples": clip.samples.size - needed,
                "required_source_start_time_s": source_start,
                "required_source_stop_time_s": source_start + (needed - 1) / clip.sampling_rate_hz,
                "repeat_or_padding_used": False,
            })
    return audit


def _manifest_and_audit() -> tuple[dict[str, object], dict[str, object]]:
    manifest = load_recorded_source_manifest(MANIFEST_PATH)
    audit = recorded_source_split_audit(manifest)
    if not audit["source_data_independent_between_splits"]:
        raise RuntimeError("manifest does not satisfy the frozen independent-session split")
    return manifest, audit


def recording_ids_for_split(split: str) -> tuple[str, ...]:
    if split not in {"calibration", "evaluation"}:
        raise ValueError("split must be calibration or evaluation")
    manifest, audit = _manifest_and_audit()
    declared = tuple(
        str(value)
        for value in manifest["independent_split_protocol"][f"{split}_recording_ids"]
    )
    audited = tuple(audit[f"{split}_recording_ids"])
    if tuple(sorted(declared)) != audited:
        raise RuntimeError("protocol recording IDs do not match audited manifest membership")
    return declared


def paired_sequence_seed(split: str, session_index: int, snr_index: int) -> int:
    bases = {
        "calibration": CALIBRATION_BASE_SEED,
        "evaluation": EVALUATION_BASE_SEED,
        "smoke": SMOKE_BASE_SEED,
    }
    if split not in bases:
        raise ValueError("split must be calibration, evaluation or smoke")
    return int(
        np.random.SeedSequence(
            [bases[split], 0x53384952, int(session_index), int(snr_index)]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def _clip(recording_id: str, split: str) -> RecordedSourceClip:
    return load_recorded_source_clip(
        MANIFEST_PATH,
        recording_id,
        split,
        target_sampling_rate_hz=48_000.0,
        maximum_frequency_hz=SOURCE_MAXIMUM_FREQUENCY_HZ,
    )


def _standardized_noise(stream: MultistationAudioStream) -> tuple[np.ndarray, ...]:
    values = []
    for station in stream.stations:
        clean_rms = float(np.sqrt(np.mean(station.clean_channels**2)))
        sigma = clean_rms / 10.0 ** (float(station.nominal_snr_db) / 20.0)
        values.append(station.noise / sigma)
    return tuple(values)


def _decorate_records(
    rows: list[dict[str, object]],
    *,
    source_model: str,
    split: str,
    recording: RecordedSourceClip,
    session_index: int,
    snr_index: int,
) -> None:
    sequence_id = (
        f"s8-independent-{split}-{source_model}-{session_index}-{snr_index}"
    )
    for row in rows:
        row.update(
            sequence_id=sequence_id,
            source_model_comparison=source_model,
            paired_recording_id=recording.recording_id,
            paired_session_id=recording.session_id,
            paired_origin_asset_id=recording.origin_asset_id,
            source_recording_id=(
                recording.recording_id
                if source_model == "recorded_source_approximation" else ""
            ),
            source_session_id=(
                recording.session_id
                if source_model == "recorded_source_approximation" else ""
            ),
            source_interval_start_s=recording.interval_start_s,
            source_interval_stop_s=recording.interval_stop_s,
            source_data_independent_between_splits=True,
            result_scope=RESULT_SCOPE,
            source_is_emitted_waveform_approximation=(
                source_model == "recorded_source_approximation"
            ),
            absolute_spl_or_detection_range_validated=False,
            frames_are_independent_trials=False,
        )


def generate_paired_sequences(
    recording_id: str,
    split: str,
    session_index: int,
    snr_index: int,
    *,
    duration_s: float = DURATION_S,
) -> dict[str, tuple[object, object, MultistationAudioStream, list[dict[str, object]], dict[str, float]]]:
    """Generate a recorded/broadband pair with common physical/noise inputs."""

    manifest_split = "calibration" if split == "calibration" else "evaluation"
    recording = _clip(recording_id, manifest_split)
    config = AudioPilotConfig("constant_velocity", SNR_LEVELS_DB[int(snr_index)])
    configuration_index = int(session_index) * len(SNR_LEVELS_DB) + int(snr_index)
    stations = pilot_stations()
    trajectory = trajectory_for_audio_pilot(config.trajectory_kind, int(session_index))
    seed = paired_sequence_seed(split, session_index, snr_index)
    result = {}
    for source_model in SOURCE_MODELS:
        pipeline_started = time.perf_counter()
        synthesis_started = time.perf_counter()
        stream = synthesize_multistation_audio(
            stations,
            trajectory,
            duration_s=duration_s,
            reception_start_time_s=RECEPTION_START_TIME_S,
            sampling_rate_hz=recording.sampling_rate_hz,
            signal_model=source_model,
            snr_db=config.snr_db,
            seed=seed,
            maximum_emitted_frequency_hz=recording.maximum_frequency_hz,
            external_source_signal=(
                recording.samples
                if source_model == "recorded_source_approximation" else None
            ),
            source_recording_id=(
                recording.recording_id
                if source_model == "recorded_source_approximation" else None
            ),
            source_session_id=(
                recording.session_id
                if source_model == "recorded_source_approximation" else None
            ),
        )
        synthesis_runtime = time.perf_counter() - synthesis_started
        rows, frontend_runtime = extract_audio_bearing_records(
            stream,
            stations,
            trajectory,
            split=manifest_split,
            configuration_index=configuration_index,
            sequence_index=0,
        )
        _decorate_records(
            rows,
            source_model=source_model,
            split=manifest_split,
            recording=recording,
            session_index=session_index,
            snr_index=snr_index,
        )
        result[source_model] = (
            stations,
            trajectory,
            stream,
            rows,
            {
                "audio_synthesis_wall_runtime_s": synthesis_runtime,
                "bearing_frontend_wall_runtime_s": frontend_runtime,
                "audio_pipeline_wall_runtime_s": time.perf_counter() - pipeline_started,
            },
        )

    recorded = result["recorded_source_approximation"]
    broadband = result["random_broadband"]
    if not np.array_equal(recorded[2].reception_times_s, broadband[2].reception_times_s):
        raise RuntimeError("paired source models received different reception timestamps")
    if tuple(item.noise_seed for item in recorded[2].stations) != tuple(
        item.noise_seed for item in broadband[2].stations
    ):
        raise RuntimeError("paired source models received different noise seeds")
    for left, right in zip(
        _standardized_noise(recorded[2]), _standardized_noise(broadband[2]), strict=True
    ):
        if not np.allclose(left, right, rtol=2e-15, atol=2e-15):
            raise RuntimeError("paired source models received different standardized noise")
    keys = []
    for rows in (recorded[3], broadband[3]):
        keys.append({
            (row["station_id"], row["estimator_variant"], int(row["frame_index"]))
            for row in rows
        })
    if keys[0] != keys[1]:
        raise RuntimeError("paired source models did not produce identical frame keys")
    return result


def _calibration_table(source_model: str, calibrations, recording_ids) -> list[dict[str, object]]:
    rows = _calibration_rows(calibrations)
    session_ids = {
        _clip(recording_id, "calibration").session_id for recording_id in recording_ids
    }
    for row in rows:
        row.update(
            source_model_comparison=source_model,
            independent_source_session_count=len(session_ids),
            dependent_calibration_sequence_count=row.pop("independent_sequence_count"),
            calibration_recording_ids_json=json.dumps(recording_ids),
            calibration_session_ids_json=json.dumps(sorted(session_ids)),
            calibration_selection_keys="station_id|estimator_variant|source_model",
            truth_scenario_used_for_calibration_selection=False,
            evaluation_used_for_calibration=False,
            result_scope=RESULT_SCOPE,
        )
    return rows


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _session_row(
    bearings: list[dict[str, object]],
    tracking: list[dict[str, object]],
    updates: list[dict[str, object]],
    sequence: dict[str, object],
    common: dict[str, object],
) -> dict[str, object]:
    valid_bearings = [row for row in bearings if row["valid"]]
    valid_tracking = [row for row in tracking if row["valid"]]
    bearing_errors = _finite([row["geodesic_error_deg"] for row in valid_bearings])
    position_errors = _finite([row["position_error_m"] for row in valid_tracking])
    velocity_errors = _finite([row["velocity_error_mps"] for row in valid_tracking])
    failures = Counter(str(row["failure_reason"]) for row in updates if row["failure_reason"])
    return {
        **common,
        "independent_source_session_trial": True,
        "dependent_frame_count": len(bearings),
        "valid_bearing_count": len(valid_bearings),
        "bearing_valid_fraction": len(valid_bearings) / len(bearings),
        "bearing_rmse_deg_conditional": (
            float(np.sqrt(np.mean(bearing_errors**2))) if bearing_errors.size else float("nan")
        ),
        "bearing_p95_deg_conditional": (
            float(np.percentile(bearing_errors, 95)) if bearing_errors.size else float("nan")
        ),
        "dependent_publication_count": len(tracking),
        "confirmed_publication_fraction": float(np.mean([row["confirmed"] for row in tracking])),
        "valid_publication_fraction": len(valid_tracking) / len(tracking),
        "position_rmse_m_conditional": (
            float(np.sqrt(np.mean(position_errors**2))) if position_errors.size else float("nan")
        ),
        "position_p95_m_conditional": (
            float(np.percentile(position_errors, 95)) if position_errors.size else float("nan")
        ),
        "velocity_rmse_mps_conditional": (
            float(np.sqrt(np.mean(velocity_errors**2))) if velocity_errors.size else float("nan")
        ),
        "coverage_conditional": (
            float(np.mean([row["valid_and_covered"] for row in valid_tracking]))
            if valid_tracking else float("nan")
        ),
        "valid_and_covered_fraction": float(np.mean([row["valid_and_covered"] for row in tracking])),
        "final_valid": bool(sequence["final_valid"]),
        "final_confirmed": bool(sequence["final_confirmed"]),
        "first_confirmation_time_s": sequence["first_confirmation_time_s"],
        "accepted_update_count": int(sequence["accepted_update_count"]),
        "rejected_update_count": int(sequence["rejected_update_count"]),
        "reset_count": int(sequence["reset_count"]),
        "failure_reason": str(sequence["failure_reason"]),
        "failure_class": (
            "computational_budget" if sequence["failure_reason"] == "computational_budget_exceeded"
            else "none" if sequence["final_valid"] else "statistical_or_geometric"
        ),
        "update_failure_reasons_json": json.dumps(failures, sort_keys=True),
        "tracker_runtime_s": float(sequence["tracker_runtime_s"]),
        "batch_optimization_count": int(sequence["batch_optimization_count"]),
        "batch_optimization_runtime_s": float(sequence["batch_optimization_runtime_s"]),
        "batch_optimization_budget": sequence["batch_optimization_budget"],
        "audio_synthesis_wall_runtime_s": float(sequence["audio_synthesis_wall_runtime_s"]),
        "bearing_frontend_wall_runtime_s": float(sequence["bearing_frontend_wall_runtime_s"]),
        "maximum_history_memory_bytes": int(sequence["maximum_history_memory_bytes"]),
        "maximum_history_nodes": int(sequence["maximum_history_nodes"]),
        "conditional_metrics_use_successful_results_only": True,
        "frames_are_independent_trials": False,
    }


def _aggregate_session_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_model_comparison"], row["snr_db"], row["estimator_variant"])].append(row)
    result = []
    for (source_model, snr_db, method), subset in sorted(groups.items()):
        session_ids = {str(row["paired_session_id"]) for row in subset}
        position = _finite([row["position_rmse_m_conditional"] for row in subset])
        velocity = _finite([row["velocity_rmse_mps_conditional"] for row in subset])
        bearing = _finite([row["bearing_rmse_deg_conditional"] for row in subset])
        coverage = _finite([row["coverage_conditional"] for row in subset])
        result.append({
            "split": "evaluation",
            "source_model_comparison": source_model,
            "snr_db": float(snr_db),
            "estimator_variant": method,
            "independent_source_session_count": len(session_ids),
            "dependent_sequence_count": len(subset),
            "final_valid_session_fraction": float(np.mean([row["final_valid"] for row in subset])),
            "mean_session_bearing_rmse_deg_conditional": float(np.mean(bearing)) if bearing.size else float("nan"),
            "maximum_session_bearing_rmse_deg_conditional": float(np.max(bearing)) if bearing.size else float("nan"),
            "mean_session_position_rmse_m_conditional": float(np.mean(position)) if position.size else float("nan"),
            "maximum_session_position_rmse_m_conditional": float(np.max(position)) if position.size else float("nan"),
            "mean_session_velocity_rmse_mps_conditional": float(np.mean(velocity)) if velocity.size else float("nan"),
            "mean_session_coverage_conditional": float(np.mean(coverage)) if coverage.size else float("nan"),
            "mean_first_confirmation_time_s": (
                float(np.mean(_finite([row["first_confirmation_time_s"] for row in subset])))
                if _finite([row["first_confirmation_time_s"] for row in subset]).size
                else float("nan")
            ),
            "accepted_update_count": sum(int(row["accepted_update_count"]) for row in subset),
            "rejected_update_count": sum(int(row["rejected_update_count"]) for row in subset),
            "failure_count": sum(not bool(row["final_valid"]) for row in subset),
            "computational_budget_failure_count": sum(
                row["failure_class"] == "computational_budget" for row in subset
            ),
            "statistical_or_geometric_failure_count": sum(
                row["failure_class"] == "statistical_or_geometric" for row in subset
            ),
            "mean_confirmed_publication_fraction": float(np.mean([
                row["confirmed_publication_fraction"] for row in subset
            ])),
            "mean_valid_publication_fraction": float(np.mean([
                row["valid_publication_fraction"] for row in subset
            ])),
            "mean_valid_and_covered_fraction": float(np.mean([
                row["valid_and_covered_fraction"] for row in subset
            ])),
            "session_level_confidence_interval_reported": False,
            "small_session_count_limitation": True,
            "result_scope": RESULT_SCOPE,
        })
    return result


def _comparison_rows(session_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (row["paired_session_id"], row["snr_db"], row["estimator_variant"], row["source_model_comparison"]): row
        for row in session_rows
    }
    result = []
    for key, recorded in lookup.items():
        session_id, snr_db, method, source_model = key
        if source_model != "recorded_source_approximation":
            continue
        broadband = lookup[(session_id, snr_db, method, "random_broadband")]
        row = {
            "paired_session_id": session_id,
            "snr_db": snr_db,
            "estimator_variant": method,
            "recorded_calibration": "recorded_calibration_sessions_683298_263022",
            "broadband_calibration": "paired_random_broadband_calibration_sequences",
            "same_trajectory_duration_timestamps": True,
            "same_standardized_noise_realization": True,
        }
        for metric in (
            "bearing_rmse_deg_conditional",
            "position_rmse_m_conditional",
            "velocity_rmse_mps_conditional",
            "coverage_conditional",
            "first_confirmation_time_s",
            "accepted_update_count",
        ):
            left = float(recorded[metric]) if recorded[metric] is not None else float("nan")
            right = float(broadband[metric]) if broadband[metric] is not None else float("nan")
            row[f"recorded_{metric}"] = left
            row[f"broadband_{metric}"] = right
            row[f"recorded_minus_broadband_{metric}"] = left - right
        result.append(row)
    return result


def _write_update_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write update diagnostics, preserving a schema when no update occurred."""

    if rows:
        _write_csv(path, rows)
        return
    fieldnames = [
        "event_id", "station_id", "frame_index", "estimator_variant",
        "processing_time_s", "reception_center_timestamp_s",
        "true_emission_time_s_evaluator_only",
        "update_motion_phase_evaluator_only", "update_applied",
        "failure_reason", "pre_update_nis", "truth_used_by_tracker",
        "split", "source_model_comparison", "paired_recording_id",
        "paired_session_id", "paired_origin_asset_id", "snr_db",
        "trajectory_kind", "sequence_seed", "result_scope",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        csv.DictWriter(output, fieldnames=fieldnames).writeheader()


def run_independent_recordings_pilot(
    *, output_directory: Path = RESULTS, progress: bool = True
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run the frozen paired calibration/evaluation protocol."""

    _, audit = _manifest_and_audit()
    # Complete these truth-generated scheduling/source-support controls before
    # any calibration or evaluation waveform is processed.
    old_control = run_ideal_schedule_control(2.0)
    corrected_control = run_ideal_schedule_control(DURATION_S)
    if (old_control["confirmed"] or not corrected_control["confirmed"]
            or corrected_control["accepted_update_count"] < 3):
        raise RuntimeError("the frozen tracker schedule failed its positive control")
    support_rows = recording_support_preflight(DURATION_S)
    calibration_records = {model: [] for model in SOURCE_MODELS}
    seed_rows = []
    recording_ids = recording_ids_for_split("calibration")
    for session_index, recording_id in enumerate(recording_ids):
        for snr_index, snr_db in enumerate(SNR_LEVELS_DB):
            pair = generate_paired_sequences(
                recording_id, "calibration", session_index, snr_index
            )
            for source_model, (_, _, stream, rows, runtimes) in pair.items():
                calibration_records[source_model].extend(rows)
                seed_rows.append({
                    "split": "calibration",
                    "paired_recording_id": recording_id,
                    "paired_session_id": rows[0]["paired_session_id"],
                    "snr_db": snr_db,
                    "source_model_comparison": source_model,
                    "sequence_seed": stream.base_seed,
                    "source_seed": stream.source_seed,
                    "noise_seeds_json": json.dumps(
                        [item.noise_seed for item in stream.stations]
                    ),
                    "same_standardized_noise_with_pair": True,
                    "duration_s": DURATION_S,
                    "frame_length_samples": FRAME_LENGTH,
                    "hop_length_samples": HOP_LENGTH,
                    **runtimes,
                })
            if progress:
                print(
                    f"independent calibration session={session_index} snr={snr_db:g}",
                    flush=True,
                )

    calibrations = {
        model: calibrate_audio_bearings(records, 1)
        for model, records in calibration_records.items()
    }
    evaluation_bearings = []
    tracking_rows = []
    update_rows = []
    session_rows = []
    evaluation_ids = recording_ids_for_split("evaluation")
    for session_index, recording_id in enumerate(evaluation_ids):
        for snr_index, snr_db in enumerate(SNR_LEVELS_DB):
            pair = generate_paired_sequences(
                recording_id, "evaluation", session_index, snr_index
            )
            for source_model, (stations, trajectory, stream, rows, runtimes) in pair.items():
                evaluation_bearings.extend(rows)
                seed_rows.append({
                    "split": "evaluation",
                    "paired_recording_id": recording_id,
                    "paired_session_id": rows[0]["paired_session_id"],
                    "snr_db": snr_db,
                    "source_model_comparison": source_model,
                    "sequence_seed": stream.base_seed,
                    "source_seed": stream.source_seed,
                    "noise_seeds_json": json.dumps([item.noise_seed for item in stream.stations]),
                    "same_standardized_noise_with_pair": True,
                    "duration_s": DURATION_S,
                    "frame_length_samples": FRAME_LENGTH,
                    "hop_length_samples": HOP_LENGTH,
                    **runtimes,
                })
                for method in ESTIMATOR_VARIANTS:
                    measurements = bearing_measurements_from_records(
                        rows, calibrations[source_model], method,
                        frame_stride=INDEPENDENT_TRACKER_FRAME_STRIDE,
                    )
                    track, sequence, updates = run_tracker(
                        stations, trajectory, measurements, method,
                        maximum_batch_optimizations_per_generation=(
                            INITIALIZATION_BATCH_OPTIMIZATION_BUDGET
                        ),
                    )
                    common = {
                        "split": "evaluation",
                        "source_model_comparison": source_model,
                        "paired_recording_id": recording_id,
                        "paired_session_id": rows[0]["paired_session_id"],
                        "paired_origin_asset_id": rows[0]["paired_origin_asset_id"],
                        "snr_db": snr_db,
                        "trajectory_kind": "constant_velocity",
                        "estimator_variant": method,
                        "sequence_seed": stream.base_seed,
                        "duration_s": DURATION_S,
                        "frame_length_samples": FRAME_LENGTH,
                        "hop_length_samples": HOP_LENGTH,
                        "tracker_frame_stride": INDEPENDENT_TRACKER_FRAME_STRIDE,
                        "batch_optimization_budget": INITIALIZATION_BATCH_OPTIMIZATION_BUDGET,
                        "qc_alpha_m2_s3": QC_ALPHA_M2_S3,
                        "calibration_source_model": source_model,
                        "calibration_selection_keys": (
                            "station_id|estimator_variant|source_model"
                        ),
                        "result_scope": RESULT_SCOPE,
                    }
                    for item in track:
                        item.update(common)
                    for item in updates:
                        item.update(common)
                    sequence.update(common, **runtimes)
                    tracking_rows.extend(track)
                    update_rows.extend(updates)
                    method_bearings = [row for row in rows if row["estimator_variant"] == method]
                    session_rows.append(
                        _session_row(method_bearings, track, updates, sequence, common)
                    )
            if progress:
                print(f"independent evaluation session={session_index} snr={snr_db:g}", flush=True)

    calibration_noise = {
        int(seed)
        for row in seed_rows if row["split"] == "calibration"
        for seed in json.loads(row["noise_seeds_json"])
    }
    evaluation_noise = {
        int(seed)
        for row in seed_rows if row["split"] == "evaluation"
        for seed in json.loads(row["noise_seeds_json"])
    }
    if not calibration_noise.isdisjoint(evaluation_noise):
        raise RuntimeError("calibration/evaluation noise seeds overlap")

    calibration_rows = []
    calibration_ids = recording_ids_for_split("calibration")
    for source_model in SOURCE_MODELS:
        calibration_rows.extend(
            _calibration_table(source_model, calibrations[source_model], calibration_ids)
        )
    summary_rows = _aggregate_session_rows(session_rows)
    comparison_rows = _comparison_rows(session_rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / f"{RESULT_PREFIX}schedule_control.csv", [old_control, corrected_control])
    _write_csv(output_directory / f"{RESULT_PREFIX}source_support.csv", support_rows)
    _write_csv(output_directory / f"{RESULT_PREFIX}calibration.csv", calibration_rows)
    _write_csv(output_directory / f"{RESULT_PREFIX}bearing_results.csv", [_csv_bearing_row(row) for row in evaluation_bearings])
    _write_csv(output_directory / f"{RESULT_PREFIX}tracking_results.csv", tracking_rows)
    _write_update_csv(output_directory / f"{RESULT_PREFIX}update_results.csv", update_rows)
    _write_csv(output_directory / f"{RESULT_PREFIX}session_results.csv", session_rows)
    _write_csv(output_directory / f"{RESULT_PREFIX}summary.csv", summary_rows)
    _write_csv(output_directory / f"{RESULT_PREFIX}recorded_broadband_comparison.csv", comparison_rows)
    _write_csv(output_directory / f"{RESULT_PREFIX}seed_provenance.csv", seed_rows)
    _write_csv(output_directory / f"{RESULT_PREFIX}split_audit.csv", [{
        key: json.dumps(value) if isinstance(value, tuple) else value
        for key, value in audit.items()
    }])
    return summary_rows, comparison_rows


def smoke_test() -> dict[str, object]:
    recording_id = recording_ids_for_split("evaluation")[0]
    pair = generate_paired_sequences(recording_id, "smoke", 0, 1, duration_s=0.08)
    recorded = pair["recorded_source_approximation"]
    broadband = pair["random_broadband"]
    return {
        "recording_id": recording_id,
        "frame_key_count": len(recorded[3]),
        "paired_frame_keys_equal": {
            (row["station_id"], row["estimator_variant"], row["frame_index"])
            for row in recorded[3]
        } == {
            (row["station_id"], row["estimator_variant"], row["frame_index"])
            for row in broadband[3]
        },
        "reception_timestamps_equal": np.array_equal(
            recorded[2].reception_times_s, broadband[2].reception_times_s
        ),
        "standardized_noise_equal": all(
            np.allclose(left, right, rtol=2e-15, atol=2e-15)
            for left, right in zip(
                _standardized_noise(recorded[2]),
                _standardized_noise(broadband[2]),
                strict=True,
            )
        ),
        "recorded_valid_bearings": sum(row["valid"] for row in recorded[3]),
        "broadband_valid_bearings": sum(row["valid"] for row in broadband[3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        print(json.dumps(smoke_test(), indent=2, sort_keys=True))
    else:
        summary, comparison = run_independent_recordings_pilot(progress=True)
        print("summary rows", len(summary), "comparison rows", len(comparison))


if __name__ == "__main__":
    main()


__all__ = [
    "DURATION_S",
    "INITIALIZATION_BATCH_OPTIMIZATION_BUDGET",
    "INDEPENDENT_TRACKER_FRAME_STRIDE",
    "MANIFEST_PATH",
    "RESULT_SCOPE",
    "RESULT_PREFIX",
    "SNR_LEVELS_DB",
    "SOURCE_MODELS",
    "generate_paired_sequences",
    "ideal_bearing_schedule",
    "paired_sequence_seed",
    "recording_ids_for_split",
    "recording_support_preflight",
    "run_ideal_schedule_control",
    "run_independent_recordings_pilot",
    "smoke_test",
]
