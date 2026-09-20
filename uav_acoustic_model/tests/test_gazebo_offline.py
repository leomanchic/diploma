"""Finite-support import and recorded Gazebo integration regressions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from simulation.gazebo_offline import SampledTrajectory, load_gazebo_recording, shared_stations
from simulation.moving_source import solve_emission_time
from validation.gazebo_offline_run import load_frozen_calibration, validate_recordings
from validation.three_station_audio_tracking_study import pilot_stations, trajectory_for_audio_pilot


RESULTS = Path(__file__).resolve().parents[1] / "results" / "gazebo_offline"


def test_sampled_trajectory_derivatives_and_finite_support():
    times = np.linspace(0, 2, 9)
    positions = np.column_stack((times, times**2, times**3))
    sampled = SampledTrajectory(times, positions)
    query = np.array([0.1, 0.83, 1.75])
    np.testing.assert_allclose(sampled.q(query), np.column_stack((query, query**2, query**3)), atol=1e-12)
    np.testing.assert_allclose(sampled.v(query), np.column_stack((np.ones(3), 2*query, 3*query**2)), atol=1e-12)
    np.testing.assert_allclose(sampled.a(query), np.column_stack((np.zeros(3), np.full(3, 2.0), 6*query)), atol=1e-11)
    for method in (sampled.q, sampled.v, sampled.a):
        with pytest.raises(ValueError, match="outside"):
            method(-1e-9)
        with pytest.raises(ValueError, match="outside"):
            method(2.000000001)


def test_sampled_trajectory_rejects_supersonic_interpolation():
    with pytest.raises(ValueError, match="subsonic"):
        SampledTrajectory([0, 1, 2, 3], [[0, 0, 0], [400, 0, 0],
                                            [800, 0, 0], [1200, 0, 0]])


@pytest.mark.parametrize("fault", ("duplicate", "reset", "gap", "missing", "quaternion"))
def test_import_rejects_bad_gazebo_timeline_or_pose(tmp_path, fault):
    source = RESULTS / "constant_velocity"
    manifest = json.loads((source / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with (source / "gazebo_state.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        names, rows = reader.fieldnames, list(reader)
    if fault == "duplicate":
        rows[10]["sim_time_s"] = rows[9]["sim_time_s"]
    elif fault == "reset":
        rows[10]["sim_time_s"] = "0.01"
    elif fault == "gap":
        del rows[10:13]
    elif fault == "missing":
        rows[10]["x_m"] = ""
    else:
        rows[10]["qw"] = "2"
    with (tmp_path / "gazebo_state.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        load_gazebo_recording(tmp_path)


def test_shared_station_config_and_recorded_history():
    loaded = shared_stations()
    pilot = pilot_stations()
    assert len(loaded) == len(pilot) == 3
    for a, b in zip(loaded, pilot, strict=True):
        np.testing.assert_allclose(a.position_world_m, b.position_world_m)
        np.testing.assert_allclose(a.rotation_local_to_world, b.rotation_local_to_world)
    recording = load_gazebo_recording(RESULTS / "smooth_turn")
    trajectory = recording.trajectory
    for station in loaded:
        for microphone in station.microphone_positions_world_m:
            emission = solve_emission_time(0.5, microphone, trajectory)
            assert trajectory.knot_times_s[0] < emission < 0.5
    assert trajectory.knot_times_s[-1] > 5.0
    assert trajectory.maximum_speed_mps < trajectory.sound_speed


def test_recorded_straight_and_turn_convergence():
    validation = validate_recordings(RESULTS)
    assert validation["straight_max_position_error_m"] < 1e-6
    assert validation["straight_max_delay_error_s"] < 1e-8
    assert validation["straight_audio_rms_difference"] < 1e-5
    assert validation["turn_position_convergence_ratio"] > 2
    assert validation["turn_velocity_convergence_ratio"] > 2


def test_frozen_calibration_and_recorded_processing_results():
    calibration = load_frozen_calibration()
    assert len(calibration) == 6
    for kind in ("constant_velocity", "smooth_turn"):
        result = json.loads((RESULTS / kind / "summary.json").read_text())
        assert result["audio_sampling_rate_hz"] == 48000
        assert result["gazebo_export_rate_hz"] == 50
        for method in result["methods"].values():
            assert method["bearing_frames"] == 1260
            assert method["valid_publication_count"] < method["publication_count"]
            assert method["failure_reasons"]
            assert method["first_confirmation_time_s"] is not None
