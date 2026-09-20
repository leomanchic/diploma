"""Offline Gazebo-state to existing three-station acoustic tracking pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simulation.gazebo_offline import load_gazebo_recording, shared_scene_config, shared_stations
from simulation.moving_source import simulate_moving_source, solve_emission_time
from simulation.multistation_audio import synthesize_multistation_audio
from simulation.signals import random_bandlimited_signal
from validation.three_station_audio_tracking_study import (
    ESTIMATOR_VARIANTS, FRAME_LENGTH, HOP_LENGTH, TRACKER_FRAME_STRIDE,
    SOURCE_MAXIMUM_FREQUENCY_HZ, SIGNAL_MODEL,
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


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_result_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def process_recording(directory: Path, calibration_path: Path = FROZEN_CALIBRATION) -> dict:
    """Run the fixed 48 kHz generator, both frontends, and the existing tracker."""

    directory = Path(directory)
    recording = load_gazebo_recording(directory)
    config = shared_scene_config()
    manifest = recording.manifest
    if manifest["station_config"] != config["stations"] or manifest["source_command"] != config["source"]:
        raise ValueError("recording scene differs from shared Python scene configuration")
    stations = shared_stations()
    trajectory = recording.trajectory
    reception_start = float(config["audio_reception_start_s"])
    duration = float(config["audio_duration_s"])
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
        sampling_rate_hz=float(config["audio_sampling_rate_hz"]),
        signal_model=SIGNAL_MODEL, snr_db=float(config["snr_db"]),
        seed=int(config["seed"]),
        maximum_emitted_frequency_hz=SOURCE_MAXIMUM_FREQUENCY_HZ,
    )
    bearings, frontend_runtime = extract_audio_bearing_records(
        stream, stations, trajectory, split="evaluation", configuration_index=0,
        sequence_index=0,
    )
    calibration = load_frozen_calibration(calibration_path)
    _write_result_csv(directory / "bearing_results.csv", [_csv_bearing_row(row) for row in bearings])
    sequences = []
    tracking_rows = []
    update_rows = []
    for method in ESTIMATOR_VARIANTS:
        measurements = bearing_measurements_from_records(
            bearings, calibration, method, frame_stride=TRACKER_FRAME_STRIDE
        )
        track, sequence, updates = run_tracker(stations, trajectory, measurements, method)
        _write_result_csv(directory / f"tracking_{method}.csv", track)
        if updates:
            _write_result_csv(directory / f"updates_{method}.csv", updates)
        sequences.append(sequence)
        tracking_rows.extend(track)
        update_rows.extend(updates)
    summary = {
        "recording_kind": manifest["kind"],
        "gazebo_sim_version": manifest["gazebo_sim_version"],
        "gazebo_state_sha256": recording.csv_sha256,
        "frozen_calibration_sha256": _hash(calibration_path),
        "frozen_calibration_source": str(calibration_path.relative_to(ROOT)) if calibration_path.is_relative_to(ROOT) else str(calibration_path),
        "seed": int(config["seed"]), "snr_db": float(config["snr_db"]),
        "audio_sampling_rate_hz": float(config["audio_sampling_rate_hz"]),
        "gazebo_export_rate_hz": 1.0 / float(manifest["export_period_s"]),
        "audio_duration_s": duration, "audio_source_sample_count": len(stream.source_signal),
        "audio_source_start_time_s": stream.source_start_time_s,
        "minimum_emission_time_s": float(emission_endpoints.min()),
        "maximum_emission_time_s": float(emission_endpoints.max()),
        "recording_support_s": [float(trajectory.knot_times_s[0]), float(trajectory.knot_times_s[-1])],
        "maximum_export_gap_s": recording.maximum_time_gap_s,
        "maximum_interpolated_speed_mps": trajectory.maximum_speed_mps,
        "frame_length": FRAME_LENGTH, "hop_length": HOP_LENGTH,
        "tracker_frame_stride": TRACKER_FRAME_STRIDE,
        "frontend_wall_runtime_s": frontend_runtime,
        "methods": {},
    }
    for method, sequence in zip(ESTIMATOR_VARIANTS, sequences, strict=True):
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
            "first_confirmation_time_s": sequence["first_confirmation_time_s"],
            "final_valid": bool(sequence["final_valid"]),
            "accepted_update_count": sequence["accepted_update_count"],
            "failure_reasons": dict(reasons),
            "update_rejection_reasons": dict(update_reasons),
            "final_failure_reason": sequence["failure_reason"],
        }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
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
    parser.add_argument("action", choices=("process", "validate"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = process_recording(args.path) if args.action == "process" else validate_recordings(args.path)
    print(json.dumps(result, indent=2, allow_nan=False))
