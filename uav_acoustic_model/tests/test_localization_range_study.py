"""Protocol and identity gates for the localization range study."""

from __future__ import annotations

import copy

import numpy as np

from simulation.gazebo_offline import load_gazebo_recording
from simulation.moving_source import solve_emission_time
from validation.gazebo_experiment import sha256, stations_from_experiment
from validation.localization_range_study import (
    DEFAULT_OUTPUT,
    DISTANCES_M,
    HISTORY_STEP_S,
    HISTORY_WINDOW_S,
    MAXIMUM_BATCH_OPTIMIZATIONS_PER_GENERATION,
    MAXIMUM_RANGE_M,
    MAXIMUM_TRANSPORT_DELAY_S,
    RECORDINGS,
    _load_study,
    _matrix,
    _run_id,
    _translated,
    aggregate,
    initialize,
    run_all,
)


PUBLISHED_TABLE_SHA256 = {
    "group_summary.csv": "11190f40e0b86501992d1402dfa88c536d11d2bc65f1aa3c4193f5a08e5026eb",
    "range_boundaries.csv": "1f15b0dc0b5f0fc89e451415f0729314e925335a4e4cc7ba69c98b7c1cdb1848",
    "run_summary.csv": "e683ee48c2519beafb266b518ea9e55867dd5b79292b271d22ec6c7ea2f28f74",
    "station_summary.csv": "10c4e5a48cfdebc166633c8dca49d3395e04bfaf205f4090733374cb235d68f4",
}
PUBLISHED_RUN_IDS_SHA256 = (
    "6c7704f02e499976c25028c9a328cc1fcc52d851384ca989275cacf3f76bbd4c"
)
PUBLISHED_MANIFEST_SHA256 = (
    "00f3e2e259885d34abd9e2247e1aa1ed0f74c201ad63d4981dfd6fe8b5d194e6"
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


def test_published_study_resumes_without_recomputation_and_reaggregates() -> None:
    """Keep the checked-in 120-run study loadable as a published artifact."""
    assert sha256(DEFAULT_OUTPUT / "study_manifest.json") == PUBLISHED_MANIFEST_SHA256
    manifest = _load_study(DEFAULT_OUTPUT)
    original_run_ids = [_run_id(manifest, spec) for spec in manifest["runs"]]

    resumed = run_all(DEFAULT_OUTPUT)
    assert resumed["completed_now"] == 0
    assert resumed["skipped_verified"] == 120
    assert resumed["total"] == 120

    aggregated = aggregate(DEFAULT_OUTPUT)
    assert aggregated["audio_run_count"] == 120
    assert aggregated["method_result_count"] == 240
    assert aggregated["station_method_result_count"] == 720
    assert aggregated["completed_run_ids_sha256"] == PUBLISHED_RUN_IDS_SHA256
    assert aggregated["tables"] == PUBLISHED_TABLE_SHA256

    reloaded = _load_study(DEFAULT_OUTPUT)
    assert [_run_id(reloaded, spec) for spec in reloaded["runs"]] == original_run_ids
