"""ENU station-pose and truth-free bearing-contract tests."""

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from model.geometry import tetrahedral_array
from model.measurements import BearingMeasurement
from model.station import StationPose


def _pose(rotation: np.ndarray | None = None) -> StationPose:
    return StationPose(
        station_id="station-east",
        position_world_m=np.asarray([12.0, -4.0, 1.5]),
        rotation_local_to_world=np.eye(3) if rotation is None else rotation,
        microphone_positions_local_m=tetrahedral_array(),
    )


def test_local_world_local_round_trip_and_world_microphone_formula():
    rotation = Rotation.from_euler("zyx", [0.4, -0.2, 0.1]).as_matrix()
    pose = _pose(rotation)
    local = np.asarray([[1.0, 2.0, -0.5], [-0.3, 0.7, 4.0]])
    world = pose.local_to_world_points(local)
    np.testing.assert_allclose(pose.world_to_local_points(world), local, atol=2e-15)
    expected = pose.position_world_m + pose.microphone_positions_local_m @ rotation.T
    np.testing.assert_allclose(pose.microphone_positions_world_m, expected, atol=0.0)


def test_pose_is_deeply_immutable_and_rejects_bad_rotations_or_centroid():
    pose = _pose()
    with pytest.raises(FrozenInstanceError):
        pose.station_id = "other"
    with pytest.raises(ValueError):
        pose.position_world_m[0] = 0.0
    with pytest.raises(ValueError, match="orthogonal"):
        _pose(np.diag([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="determinant"):
        _pose(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="centroid"):
        StationPose("bad", np.zeros(3), np.eye(3), tetrahedral_array() + 0.01)


def test_global_translation_rotation_and_microphone_permutation_preserve_physics():
    pose = _pose(Rotation.from_euler("z", 0.3).as_matrix())
    target = np.asarray([31.0, 18.0, 9.0])
    local_direction = pose.world_to_local_direction(target - pose.position_world_m)
    local_direction /= np.linalg.norm(local_direction)

    translation = np.asarray([100.0, -80.0, 9.0])
    translated = StationPose(
        pose.station_id,
        pose.position_world_m + translation,
        pose.rotation_local_to_world,
        pose.microphone_positions_local_m,
    )
    translated_direction = translated.world_to_local_direction(
        target + translation - translated.position_world_m
    )
    translated_direction /= np.linalg.norm(translated_direction)
    np.testing.assert_allclose(translated_direction, local_direction, atol=2e-15)

    global_rotation = Rotation.from_euler("xyz", [0.2, -0.4, 0.1]).as_matrix()
    rotated = StationPose(
        pose.station_id,
        global_rotation @ pose.position_world_m,
        global_rotation @ pose.rotation_local_to_world,
        pose.microphone_positions_local_m,
    )
    rotated_target = global_rotation @ target
    rotated_direction = rotated.world_to_local_direction(
        rotated_target - rotated.position_world_m
    )
    rotated_direction /= np.linalg.norm(rotated_direction)
    np.testing.assert_allclose(rotated_direction, local_direction, atol=2e-15)

    permutation = np.asarray([2, 0, 3, 1])
    permuted = StationPose(
        pose.station_id,
        pose.position_world_m,
        pose.rotation_local_to_world,
        pose.microphone_positions_local_m[permutation],
    )
    np.testing.assert_allclose(
        permuted.microphone_positions_world_m,
        pose.microphone_positions_world_m[permutation],
        atol=0.0,
    )
    np.testing.assert_allclose(
        permuted.world_to_local_direction(target - permuted.position_world_m),
        pose.world_to_local_direction(target - pose.position_world_m),
        atol=0.0,
    )


def test_bearing_measurement_has_no_truth_fields_and_validates_covariance():
    measurement = BearingMeasurement(
        station_id="s0",
        sequence_id="sequence-1",
        frame_index=3,
        reception_center_timestamp_s=1.0,
        available_timestamp_s=1.02,
        direction_local=np.asarray([1.0, 0.0, 0.0]),
        covariance_tangent_rad2=np.diag(np.deg2rad([0.2, 0.3]) ** 2),
        calibration_bias_tangent_rad=np.deg2rad([0.05, -0.02]),
        estimator_variant="equal_weight_srp_phat",
        quality_metadata={"peak_score": 0.8},
    )
    forbidden = {"true_direction", "true_position", "angular_error", "true_emission_time"}
    assert forbidden.isdisjoint({item.name for item in fields(measurement)})
    assert measurement.quality_metadata["peak_score"] == 0.8
    with pytest.raises(TypeError):
        measurement.quality_metadata["peak_score"] = 0.1
    with pytest.raises(ValueError, match="positive semidefinite"):
        BearingMeasurement(
            "s0", "q", 0, 0.0, 0.1, [1.0, 0.0, 0.0],
            [[1.0, 2.0], [2.0, 1.0]], [0.0, 0.0], "gcc"
        )


def test_invalid_measurement_has_no_fictitious_direction_or_covariance():
    measurement = BearingMeasurement.invalid(
        station_id="s0",
        sequence_id="sequence-1",
        frame_index=4,
        reception_center_timestamp_s=2.0,
        available_timestamp_s=2.02,
        estimator_variant="gcc",
        invalid_reason="insufficient_energy",
    )
    assert not measurement.valid
    assert np.all(np.isnan(measurement.direction_local))
    assert np.all(np.isnan(measurement.covariance_tangent_rad2))

