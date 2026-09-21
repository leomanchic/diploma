"""Protocol and identity gates for the localization range study."""

from __future__ import annotations

import copy

import numpy as np

from simulation.gazebo_offline import load_gazebo_recording
from simulation.moving_source import solve_emission_time
from validation.gazebo_experiment import stations_from_experiment
from validation.localization_range_study import (
    DISTANCES_M,
    HISTORY_STEP_S,
    HISTORY_WINDOW_S,
    MAXIMUM_BATCH_OPTIMIZATIONS_PER_GENERATION,
    MAXIMUM_RANGE_M,
    MAXIMUM_TRANSPORT_DELAY_S,
    RECORDINGS,
    _matrix,
    _run_id,
    _translated,
    initialize,
)


def test_frozen_matrix_has_exact_volume_and_paired_seeds() -> None:
    matrix = _matrix()
    assert len(matrix) == 120
    assert sum(row["experiment"] == "geometry_control" for row in matrix) == 12
    assert sum(row["experiment"] == "fixed_background" for row in matrix) == 108
    assert [row["index"] for row in matrix] == list(range(120))

    for trajectory in RECORDINGS:
        subset = [row for row in matrix if row["trajectory"] == trajectory]
        assert len({row["source_seed"] for row in subset}) == 1
        fixed = [row for row in subset if row["experiment"] == "fixed_background"]
        for replicate in range(3):
            paired = [row for row in fixed if row["replicate"] == replicate]
            assert len({row["noise_seed"] for row in paired}) == 1
    assert MAXIMUM_RANGE_M / 343.0 + MAXIMUM_TRANSPORT_DELAY_S + HISTORY_STEP_S < HISTORY_WINDOW_S


def test_initialization_freezes_geometry_source_support_and_unique_ids(tmp_path) -> None:
    output = tmp_path / "study"
    manifest = initialize(output)
    assert manifest["run_count"] == 120
    assert manifest["fixed_background"]["no_per_stream_normalization"] is True
    assert (
        manifest["processing"]["tracker"]["recovery"]
        ["maximum_batch_optimizations_per_generation"]
        == MAXIMUM_BATCH_OPTIMIZATIONS_PER_GENERATION
    )

    stations = stations_from_experiment({"processing": manifest["processing"]})
    centroid = np.mean([station.position_world_m for station in stations], axis=0)
    microphones = np.vstack(
        [station.microphone_positions_world_m for station in stations]
    )
    for trajectory_name, recording_path in RECORDINGS.items():
        recording = load_gazebo_recording(recording_path)
        info = manifest["recordings"][trajectory_name]
        bank = manifest["source_banks"][trajectory_name]
        bank_stop = bank["start_time_s"] + bank["sample_count"] / bank["sampling_rate_hz"]
        source = np.load(output / bank["path"], allow_pickle=False)
        assert source.shape == (bank["sample_count"],)
        np.testing.assert_allclose(
            np.sqrt(np.mean(source**2)), 1.0, rtol=0.0, atol=2e-15
        )
        for distance in DISTANCES_M:
            trajectory, _ = _translated(
                recording, stations, info["reception_start_s"], distance,
                trajectory_name,
            )
            np.testing.assert_allclose(
                np.linalg.norm(trajectory.q(info["reception_start_s"]) - centroid),
                distance, rtol=0.0, atol=1e-10,
            )
            endpoints = np.asarray([
                solve_emission_time(
                    np.asarray([
                        info["reception_start_s"],
                        info["reception_start_s"] + info["duration_s"],
                    ]),
                    microphone,
                    trajectory,
                )
                for microphone in microphones
            ])
            assert endpoints.min() > recording.trajectory.knot_times_s[0]
            assert endpoints.max() < recording.trajectory.knot_times_s[-1]
            assert bank["start_time_s"] < endpoints.min()
            assert bank_stop > endpoints.max()

    run_ids = [_run_id(manifest, row) for row in manifest["runs"]]
    assert len(set(run_ids)) == 120
    assert run_ids == [_run_id(copy.deepcopy(manifest), copy.deepcopy(row))
                       for row in manifest["runs"]]
    changed = copy.deepcopy(manifest["runs"][0])
    changed["snr_db"] += 1.0
    assert _run_id(manifest, changed) != run_ids[0]
