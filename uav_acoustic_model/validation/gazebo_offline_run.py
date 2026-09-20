"""Offline Gazebo-state to existing three-station acoustic tracking pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from estimators.retarded_ekf_manoeuvre import ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from simulation.gazebo_offline import load_gazebo_recording, shared_stations
from simulation.moving_source import simulate_moving_source, solve_emission_time
from simulation.multistation_audio import synthesize_multistation_audio
from simulation.signals import random_bandlimited_signal
from validation.gazebo_experiment import (
    calibration_from_experiment, check_processing_compatibility,
    create_experiment, initialize_experiment, load_experiment, migrate_legacy,
    refresh_code_identity,
    stations_from_experiment, verify_results, write_results_manifest,
)
from validation.three_station_audio_tracking_study import (
    ESTIMATOR_VARIANTS,
    _csv_bearing_row, bearing_measurements_from_records,
    extract_audio_bearing_records, run_tracker, trajectory_for_audio_pilot,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CALIBRATION = ROOT / "results" / "three_station_audio_calibration.csv"


def load_frozen_calibration(path: Path = FROZEN_CALIBRATION) -> dict:
    """Read the previous calibration pilot; no evaluation result can tune it."""

    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    calibrations = {}
    for row in rows:
        if row["split"] != "calibration" or row["evaluation_used_for_calibration"] != "False":
            raise ValueError("calibration source is not an untouched calibration split")
        covariance = np.array([
            [float(row["covariance_00_rad2"]), float(row["covariance_01_rad2"])],
            [float(row["covariance_01_rad2"]), float(row["covariance_11_rad2"])],
        ])
        if np.min(np.linalg.eigvalsh(covariance)) <= 0:
            raise ValueError("frozen bearing covariance must be positive definite")
        key = row["station_id"], row["estimator_variant"]
        if key in calibrations:
            raise ValueError("duplicate frozen calibration key")
        calibrations[key] = SimpleNamespace(
            covariance_rad2=covariance,
            mean_residual_rad=np.array([float(row["bias_az_arc_rad"]),
                                        float(row["bias_el_arc_rad"])]),
        )
    expected = {(station.station_id, method) for station in shared_stations()
                for method in ESTIMATOR_VARIANTS}
    if set(calibrations) != expected:
        raise ValueError("frozen calibration station/method keys do not match scene")
    return calibrations


def _write_result_csv(path: Path, rows: list[dict], *, empty_columns: tuple[str, ...] = ()) -> None:
    if not rows and not empty_columns:
        raise ValueError("cannot write an empty result table without a schema")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else list(empty_columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def process_recording(directory: Path, calibration_path: Path | None = None) -> dict:
    """Replay the frozen experiment with the existing acoustic pipeline."""

    directory = Path(directory)
    if calibration_path is not None:
        raise ValueError("calibration is frozen in experiment.json; use `new` for different calibration")
    experiment = load_experiment(directory, check_code=True)
    if (directory / "results_manifest.json").exists():
        verify_results(directory, experiment)
    check_processing_compatibility(experiment)
    recording = load_gazebo_recording(directory)
    manifest = recording.manifest
    processing = experiment["processing"]
    if [{key: item[key] for key in ("id", "position_m", "rpy_rad")}
        for item in processing["stations"]] != manifest["station_config"]:
        raise ValueError("frozen station geometry disagrees with Gazebo manifest")
    stations = stations_from_experiment(experiment)
    trajectory = recording.trajectory
    audio = processing["audio"]
    frontend = processing["frontend"]
    tracker_config = processing["tracker"]
    evaluation_phases = processing.get("evaluation_phases")

    def flight_phase(time_s: float) -> str:
        if evaluation_phases is None:
            raise AssertionError("flight phase requested for a legacy recording")
        for item in evaluation_phases:
            if item["start_s"] <= time_s < item["end_s"]:
                return item["name"]
        return "before_report" if time_s < evaluation_phases[0]["start_s"] else "after_report"
    reception_start = float(audio["reception_start_s"])
    duration = float(audio["duration_s"])
    reception_end = reception_start + duration
    microphones = np.vstack([station.microphone_positions_world_m for station in stations])
    emission_endpoints = np.array([
        solve_emission_time(t, microphone, trajectory)
        for microphone in microphones for t in (reception_start, reception_end)
    ])
    if emission_endpoints.min() <= trajectory.knot_times_s[0] or reception_end >= trajectory.knot_times_s[-1]:
        raise ValueError("recording lacks retarded-time history or reception tail")
    stream = synthesize_multistation_audio(
        stations, trajectory, duration_s=duration,
        reception_start_time_s=reception_start,
        sampling_rate_hz=float(audio["sampling_rate_hz"]),
        signal_model=audio["signal_model"], snr_db=audio["snr_db"],
        seed=int(audio["base_seed"]), sound_speed=float(audio["sound_speed_mps"]),
        maximum_emitted_frequency_hz=float(audio["source_maximum_frequency_hz"]),
        fir_length=int(audio["fir_length"]), chunk_size_samples=int(audio["chunk_size_samples"]),
        geometric_attenuation=bool(audio["geometric_attenuation"]),
    )
    bearings, frontend_runtime = extract_audio_bearing_records(
        stream, stations, trajectory, split="evaluation", configuration_index=0,
        sequence_index=0, frame_length=int(frontend["frame_length"]),
        hop_length=int(frontend["hop_length"]),
        modeled_processing_delay_s=float(frontend["modeled_processing_delay_s"]),
        station_delivery_delay_s=frontend["station_delivery_delay_s"],
        sequence_id=experiment["run_id"],
    )
    calibration = calibration_from_experiment(experiment)
    for row in bearings:
        row["run_id"] = experiment["run_id"]
    history = ManoeuvreHistoryConfig(np.asarray(tracker_config["qc_m2_s3"], dtype=float),
        history_step_s=tracker_config["history_step_s"],
        history_window_s=tracker_config["history_window_s"],
        maximum_range_m=tracker_config["maximum_range_m"],
        maximum_transport_delay_s=tracker_config["maximum_transport_delay_s"])
    recovery = InitializationRecoveryConfig(**tracker_config["recovery"])
    files: dict[str, list[dict]] = {"bearing_results.csv": [_csv_bearing_row(row) for row in bearings]}
    sequences = []
    tracking_rows = []
    update_rows = []
    for method in frontend["methods"]:
        measurements = bearing_measurements_from_records(
            bearings, calibration, method, frame_stride=int(tracker_config["frame_stride"])
        )
        track, sequence, updates = run_tracker(
            stations, trajectory, measurements, method,
            history_config=history, recovery_config=recovery,
            coverage_threshold=float(tracker_config["position_coverage_threshold"]),
            phase_classifier=flight_phase if evaluation_phases is not None else None,
            manoeuvre_start_s=float(tracker_config["manoeuvre_start_s"]),
        )
        for row in track + updates:
            row["run_id"] = experiment["run_id"]
            row["sequence_id"] = experiment["run_id"]
        sequence["run_id"] = experiment["run_id"]
        sequence["sequence_id"] = experiment["run_id"]
        files[f"tracking_{method}.csv"] = track
        files[f"updates_{method}.csv"] = updates
        sequences.append(sequence)
        tracking_rows.extend(track)
        update_rows.extend(updates)
    summary = {
        "schema_version": 2, "run_id": experiment["run_id"],
        "sequence_id": experiment["run_id"],
        "config_sha256": experiment["config_sha256"],
        "code_sha256": experiment["code"]["sha256"],
        "recording_kind": manifest["kind"],
        "gazebo_sim_version": manifest["gazebo_sim_version"],
        "gazebo_state_sha256": recording.csv_sha256,
        "frozen_calibration_sha256": processing["calibration"]["origin"]["sha256"],
        "frozen_calibration_source": processing["calibration"]["origin"]["source_name"],
        "seed": int(audio["base_seed"]), "snr_db": audio["snr_db"],
        "audio_sampling_rate_hz": float(audio["sampling_rate_hz"]),
        "gazebo_export_rate_hz": 1.0 / float(manifest["export_period_s"]),
        "audio_duration_s": duration, "audio_source_sample_count": len(stream.source_signal),
        "audio_source_start_time_s": stream.source_start_time_s,
        "minimum_emission_time_s": float(emission_endpoints.min()),
        "maximum_emission_time_s": float(emission_endpoints.max()),
        "recording_support_s": [float(trajectory.knot_times_s[0]), float(trajectory.knot_times_s[-1])],
        "maximum_export_gap_s": recording.maximum_time_gap_s,
        "maximum_interpolated_speed_mps": trajectory.maximum_speed_mps,
        "frame_length": frontend["frame_length"], "hop_length": frontend["hop_length"],
        "tracker_frame_stride": tracker_config["frame_stride"],
        "frontend_wall_runtime_s": frontend_runtime,
        "methods": {},
    }
    for method, sequence in zip(frontend["methods"], sequences, strict=True):
        track = [row for row in tracking_rows if row["estimator_variant"] == method]
        valid = [row for row in track if row["valid"]]
        bearing_subset = [row for row in bearings if row["estimator_variant"] == method]
        reasons = Counter(str(row["failure_reason"] or row["status"])
                          for row in track if not row["valid"])
        update_reasons = Counter(str(row["failure_reason"])
                                 for row in update_rows if row["estimator_variant"] == method
                                 and not row["update_applied"])
        errors = np.array([row["position_error_m"] for row in valid], dtype=float)
        summary["methods"][method] = {
            "bearing_frames": len(bearing_subset),
            "valid_bearing_frames": sum(bool(row["valid"]) for row in bearing_subset),
            "publication_count": len(track),
            "valid_publication_count": len(valid),
            "availability_fraction": len(valid) / len(track) if track else 0,
            "coverage_fraction_conditional": (sum(bool(row["valid_and_covered"]) for row in valid) / len(valid)
                                              if valid else None),
            "position_rmse_m_conditional": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "position_p95_m_conditional": float(np.percentile(errors, 95)) if len(errors) else None,
            "maximum_observed_error_m": float(np.max(errors)) if len(errors) else None,
            "track_loss_count": sum(bool(previous["valid"]) and not current["valid"]
                                    for previous, current in zip(track, track[1:])),
            "first_confirmation_time_s": (
                float(sequence["first_confirmation_time_s"])
                if np.isfinite(sequence["first_confirmation_time_s"]) else None
            ),
            "final_valid": bool(sequence["final_valid"]),
            "accepted_update_count": sequence["accepted_update_count"],
            "rejected_update_count": sequence["rejected_update_count"],
            "failure_reasons": dict(reasons),
            "update_rejection_reasons": dict(update_reasons),
            "final_failure_reason": sequence["failure_reason"],
        }
        if evaluation_phases is not None:
            phase_table = {}
            for phase in evaluation_phases:
                name = phase["name"]
                publications = [row for row in track
                                if row["publication_motion_phase_evaluator_only"] == name]
                phase_updates = [row for row in update_rows
                                 if row["estimator_variant"] == method
                                 and row["update_motion_phase_evaluator_only"] == name]
                phase_valid = [row for row in publications if row["valid"]]
                phase_errors = np.asarray([row["position_error_m"] for row in phase_valid], dtype=float)
                phase_table[name] = {
                    "start_s": phase["start_s"], "end_s": phase["end_s"],
                    "publication_count": len(publications),
                    "valid_publication_count": len(phase_valid),
                    "availability_fraction": len(phase_valid) / len(publications) if publications else 0.0,
                    "position_rmse_m_conditional": float(np.sqrt(np.mean(phase_errors**2))) if len(phase_errors) else None,
                    "position_p95_m_conditional": float(np.percentile(phase_errors, 95)) if len(phase_errors) else None,
                    "maximum_observed_error_m": float(np.max(phase_errors)) if len(phase_errors) else None,
                    "accepted_update_count": sum(bool(row["update_applied"]) for row in phase_updates),
                    "rejected_update_count": sum(not row["update_applied"] for row in phase_updates),
                    "track_loss_count": sum(bool(previous["valid"]) and not current["valid"]
                                            for previous, current in zip(publications, publications[1:])),
                    "failure_reasons": dict(Counter(str(row["failure_reason"] or row["status"])
                                                    for row in publications if not row["valid"])),
                }
            summary["methods"][method]["phase_metrics"] = phase_table
    # Publish all result files and their manifest only after successful calculation.
    with tempfile.TemporaryDirectory(prefix=".gazebo-process-", dir=directory) as temporary:
        staged = Path(temporary)
        for name, rows in files.items():
            _write_result_csv(staged / name, rows,
                              empty_columns=("run_id", "sequence_id", "event_id", "station_id",
                                             "frame_index", "estimator_variant", "update_applied", "failure_reason")
                              if name.startswith("updates_") else ())
        (staged / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        for name in (*files, "summary.json"):
            os.replace(staged / name, directory / name)
    write_results_manifest(directory, experiment, [*files, "summary.json"])
    return summary


def validate_recordings(root: Path) -> dict:
    """Numerical tests against the analytic command and same-source audio."""

    root = Path(root)
    straight = load_gazebo_recording(root / "constant_velocity")
    turn = load_gazebo_recording(root / "smooth_turn")
    fine = load_gazebo_recording(root / "smooth_turn_100hz")
    analytical_straight = trajectory_for_audio_pilot("constant_velocity", 0)
    analytical_turn = trajectory_for_audio_pilot("smooth_turn", 0)
    straight_times = straight.trajectory.knot_times_s
    straight_error = np.linalg.norm(straight.trajectory.q(straight_times)
                                    - analytical_straight.q(straight_times), axis=-1)
    dense = np.arange(0.02, 5.981, 0.001)
    turn_errors = []
    turn_velocity_errors = []
    for sampled in (turn.trajectory, fine.trajectory):
        turn_errors.append(float(np.max(np.linalg.norm(sampled.q(dense) - analytical_turn.q(dense), axis=1))))
        turn_velocity_errors.append(float(np.max(np.linalg.norm(sampled.v(dense) - analytical_turn.v(dense), axis=1))))
    microphone_positions = shared_stations()[0].microphone_positions_world_m
    reception = 1.0 + np.arange(9600) / 48000.0
    delays = []
    for trajectory in (analytical_straight, straight.trajectory):
        emission = np.array([solve_emission_time(reception, microphone, trajectory)
                             for microphone in microphone_positions])
        delays.append(reception[None, :] - emission)
    signal = random_bandlimited_signal(48000.0, 72000,
                                       np.random.default_rng(20260920),
                                       minimum_frequency_hz=300.0,
                                       maximum_frequency_hz=10000.0,
                                       taper_fraction=0.0)
    audio = [simulate_moving_source(signal, 48000, microphone_positions, trajectory,
                                    source_start_time_s=0.1, reception_times_s=reception,
                                    maximum_emitted_frequency_hz=10000.0).channels
             for trajectory in (analytical_straight, straight.trajectory)]
    report = {
        "straight_max_position_error_m": float(straight_error.max()),
        "straight_max_delay_error_s": float(np.max(np.abs(delays[0] - delays[1]))),
        "straight_audio_rms_difference": float(np.sqrt(np.mean((audio[0] - audio[1])**2))),
        "straight_audio_rms_reference": float(np.sqrt(np.mean(audio[0]**2))),
        "turn_50hz_max_position_error_m": turn_errors[0],
        "turn_100hz_max_position_error_m": turn_errors[1],
        "turn_50hz_max_velocity_error_mps": turn_velocity_errors[0],
        "turn_100hz_max_velocity_error_mps": turn_velocity_errors[1],
        "turn_position_convergence_ratio": turn_errors[0] / turn_errors[1],
        "turn_velocity_convergence_ratio": turn_velocity_errors[0] / turn_velocity_errors[1],
        "export_rates_hz": [50, 100], "audio_rate_hz": 48000,
        "comparison_reception_interval_s": [float(reception[0]), float(reception[-1])],
    }
    if (report["straight_max_position_error_m"] > 1e-6 or
        report["straight_max_delay_error_s"] > 1e-8 or
        report["straight_audio_rms_difference"] > 1e-5 or
        report["turn_50hz_max_position_error_m"] > 1e-5 or
        report["turn_100hz_max_position_error_m"] >= report["turn_50hz_max_position_error_m"] or
        report["turn_100hz_max_velocity_error_mps"] >= report["turn_50hz_max_velocity_error_mps"]):
        raise AssertionError(f"Gazebo integration comparison failed: {report}")
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "migrate", "new", "process", "refresh-code", "validate"))
    parser.add_argument("path", type=Path)
    parser.add_argument("destination", nargs="?", type=Path)
    parser.add_argument("--snr-db", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--comparison-group-id")
    parser.add_argument("--processing-config", type=Path)
    args = parser.parse_args()
    if args.action != "new" and args.destination is not None:
        parser.error("destination is allowed only with `new`")
    if args.action != "init" and args.processing_config is not None:
        parser.error("--processing-config is allowed only with `init`")
    if args.action != "new" and any(value is not None for value in (args.snr_db, args.seed)):
        parser.error("SNR and seed changes require `new` or an explicit `init` processing config")
    if args.action not in ("new", "init") and args.comparison_group_id is not None:
        parser.error("comparison group is allowed only with `new` or `init`")
    if args.action not in ("new", "migrate") and args.calibration is not None:
        parser.error("calibration changes require `new` or an explicit `init` processing config")
    if args.action == "init":
        if args.processing_config is None:
            parser.error("init requires --processing-config")
        result = initialize_experiment(args.path, args.processing_config,
                                       comparison_group_id=args.comparison_group_id)
    elif args.action == "process":
        result = process_recording(args.path)
    elif args.action == "refresh-code":
        result = refresh_code_identity(args.path)
    elif args.action == "validate":
        result = validate_recordings(args.path)
    elif args.action == "migrate":
        result = migrate_legacy(args.path, args.calibration or FROZEN_CALIBRATION)
    else:
        if args.destination is None:
            parser.error("new requires a separate destination directory")
        result = create_experiment(args.path, args.destination, snr_db=args.snr_db,
            seed=args.seed, calibration_path=args.calibration,
            comparison_group_id=args.comparison_group_id)
    print(json.dumps(result, indent=2, allow_nan=False))
