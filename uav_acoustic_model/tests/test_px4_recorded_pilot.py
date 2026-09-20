"""Integrity and physical-flight gates for the committed X500 pilot."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from simulation.gazebo_offline import load_gazebo_recording
from validation.gazebo_experiment import verify_results


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/px4_flight/pilot_003"


def test_committed_px4_flight_recording_and_replay_are_consistent() -> None:
    recording = load_gazebo_recording(RUN)
    manifest = recording.manifest
    status = json.loads((RUN / "flight_program_status.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    experiment = json.loads((RUN / "experiment.json").read_text())
    assert status["completed"] is True
    assert status["offboard_start_attempts"] == [
        "direct MAVLink: PX4 OFFBOARD confirmed"]
    assert manifest["px4_git_sha"] == "d6f12ad1c4f70ad3230afd7d86e971421e02fef4"
    assert manifest["px4_build_patch_applied"] is True
    assert manifest["gazebo_sim_version"] == "8.15.0"
    assert manifest["gazebo_seed_requested"] == 20260920
    assert manifest["gazebo_seed_applied"] is None
    assert recording.csv_sha256 == summary["gazebo_state_sha256"]
    assert recording.csv_sha256 == experiment["recording"]["state_sha256"]
    assert verify_results(RUN)["run_id"] == experiment["run_id"]
    assert summary["run_id"] == experiment["run_id"]
    assert summary["audio_duration_s"] >= 15.0
    assert summary["audio_duration_s"] <= 20.0
    assert summary["gazebo_export_rate_hz"] == 50.0
    assert summary["audio_sampling_rate_hz"] == 48000.0
    assert summary["seed"] == 20260920
    assert summary["snr_db"] == 10.0
    comparison = json.loads((RUN / "replay_comparison.json").read_text())
    assert comparison["new_run_id"] == summary["run_id"]
    assert comparison["method_metrics_identical"] is True
    assert all(item["maximum_numeric_difference"] == 0.0
               and item["status_mismatches"] == 0
               for item in comparison["files"].values())
    portable = json.loads((RUN / "portable_replay_comparison.json").read_text())
    assert portable["same_run_id"] == summary["run_id"]
    assert portable["same_recording_sha256"] == recording.csv_sha256
    assert portable["method_metrics_identical"] is True
    assert all(item["maximum_numeric_difference"] == 0.0
               and item["status_mismatches"] == 0
               for item in portable["files"].values())

    times = recording.trajectory.knot_times_s
    xyz = recording.trajectory.knot_positions_m
    velocity = recording.world_velocities_mps
    assert velocity is not None
    assert len(times) > 3000
    assert np.max(np.diff(times)) < 0.021
    assert np.max(xyz[:, 2]) > 15.5
    assert xyz[-1, 2] < 0.7
    phases = {item["name"]: item for item in manifest["phase_intervals"]}

    def segment(name: str) -> tuple[np.ndarray, np.ndarray]:
        phase = phases[name]
        mask = (times >= phase["start_s"]) & (times < phase["end_s"])
        return xyz[mask], velocity[mask]

    straight, straight_velocity = segment("straight")
    turn, turn_velocity = segment("turn")
    assert straight[-1, 0] - straight[0, 0] > 10.0
    assert abs(straight[-1, 1] - straight[0, 1]) < 2.0
    assert straight_velocity[-1, 0] > 2.0
    assert turn[-1, 1] - turn[0, 1] > 5.0
    assert turn_velocity[-1, 1] > 2.0
    assert turn_velocity[-1, 0] < 1.0

    previous_ids: set[str] = set()
    for name in ("constant_velocity", "smooth_turn"):
        with (ROOT / "results/gazebo_offline" / name / "updates_all_6_equal_gcc_wls.csv").open(newline="") as file:
            previous_ids.update(row["event_id"] for row in csv.DictReader(file))
    for method in ("all_6_equal_gcc_wls", "equal_weight_srp_phat"):
        metrics = summary["methods"][method]
        assert metrics["first_confirmation_time_s"] is not None
        assert metrics["valid_publication_count"] > 0
        assert metrics["availability_fraction"] > 0.9
        assert metrics["track_loss_count"] == 0
        assert set(metrics["phase_metrics"]) == {
            "hover_before", "straight", "turn", "hover_after"}
        with (RUN / f"updates_{method}.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        assert rows
        assert {row["run_id"] for row in rows} == {experiment["run_id"]}
        assert not {row["event_id"] for row in rows} & previous_ids
