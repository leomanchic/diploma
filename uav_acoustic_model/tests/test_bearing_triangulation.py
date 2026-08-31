"""Deterministic static spherical bearing-triangulation tests."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from estimators.bearing_triangulation import (
    bearing_residual,
    bearing_residual_jacobian,
    closest_rays_triangulation,
    numerical_bearing_residual_jacobian,
    triangulate_bearings_spherical_wls,
)
from model.bearing_statistics import tangent_basis
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.station import StationPose


SIGMA_RAD = np.deg2rad(0.2)
COVARIANCE = np.diag([SIGMA_RAD**2, (1.5 * SIGMA_RAD) ** 2])


def _exp_map(direction: np.ndarray, tangent_coordinates: np.ndarray) -> np.ndarray:
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ np.asarray(tangent_coordinates)
    theta = float(np.linalg.norm(tangent))
    if theta == 0.0:
        return np.asarray(direction, dtype=float)
    return np.cos(theta) * direction + np.sin(theta) * tangent / theta


def _scene(
    positions: list[np.ndarray] | None = None,
    target: np.ndarray | None = None,
    rotations: list[np.ndarray] | None = None,
    biases: list[np.ndarray] | None = None,
    covariance: np.ndarray = COVARIANCE,
):
    if positions is None:
        positions = [
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([20.0, 0.0, 0.0]),
            np.asarray([7.0, 18.0, 2.0]),
        ]
    if target is None:
        target = np.asarray([8.0, 7.0, 13.0])
    if rotations is None:
        rotations = [np.eye(3) for _ in positions]
    if biases is None:
        biases = [np.zeros(2) for _ in positions]
    stations = []
    measurements = []
    for index, (position, rotation, bias) in enumerate(
        zip(positions, rotations, biases, strict=True)
    ):
        station = StationPose(
            f"s{index}", position, rotation, tetrahedral_array()
        )
        world_direction = (target - position) / np.linalg.norm(target - position)
        local_direction = station.world_to_local_direction(world_direction)
        measured = _exp_map(local_direction, bias)
        measurement = BearingMeasurement(
            station_id=station.station_id,
            sequence_id="static-state-0",
            frame_index=0,
            reception_center_timestamp_s=3.0,
            available_timestamp_s=3.01 + index * 1e-3,
            direction_local=measured,
            covariance_tangent_rad2=covariance,
            calibration_bias_tangent_rad=bias,
            estimator_variant="direct_bearing",
        )
        stations.append(station)
        measurements.append(measurement)
    return stations, measurements, np.asarray(target, dtype=float)


def test_three_noncollinear_stations_recover_ideal_position_to_roundoff():
    stations, measurements, target = _scene()
    initial = closest_rays_triangulation(stations, measurements)
    result = triangulate_bearings_spherical_wls(stations, measurements)
    assert initial.valid and result.valid
    np.testing.assert_allclose(initial.position_world_m, target, atol=2e-14)
    np.testing.assert_allclose(result.position_world_m, target, atol=2e-13)
    assert result.objective < 1e-20
    assert result.information_rank == 3
    assert np.all(np.linalg.eigvalsh(result.covariance_position_m2) > 0.0)


def test_two_stations_work_for_good_crossing_but_are_not_claimed_always_stable():
    stations, measurements, target = _scene()
    result = triangulate_bearings_spherical_wls(stations[:2], measurements[:2])
    assert result.valid
    np.testing.assert_allclose(result.position_world_m, target, atol=3e-13)
    assert result.information_rank == 3

    far_target = np.asarray([1.0e7, 2.0e6, 3.0e6])
    far_stations, far_measurements, _ = _scene(target=far_target)
    far = triangulate_bearings_spherical_wls(far_stations[:2], far_measurements[:2])
    assert (not far.valid) or far.information_condition_number > 1e8


def test_different_local_station_orientations_and_station_permutation():
    rotations = [
        Rotation.from_euler("zyx", [0.2, -0.1, 0.05]).as_matrix(),
        Rotation.from_euler("zyx", [-0.7, 0.15, -0.2]).as_matrix(),
        Rotation.from_euler("zyx", [1.2, -0.25, 0.1]).as_matrix(),
    ]
    stations, measurements, target = _scene(rotations=rotations)
    baseline = triangulate_bearings_spherical_wls(stations, measurements)
    permutation = [2, 0, 1]
    permuted = triangulate_bearings_spherical_wls(
        [stations[index] for index in permutation],
        [measurements[index] for index in permutation],
    )
    assert baseline.valid and permuted.valid
    np.testing.assert_allclose(baseline.position_world_m, target, atol=3e-13)
    np.testing.assert_allclose(permuted.position_world_m, baseline.position_world_m, atol=3e-13)
    np.testing.assert_allclose(
        permuted.covariance_position_m2, baseline.covariance_position_m2, rtol=2e-13
    )


def test_global_translation_and_rotation_transform_position_and_covariance():
    stations, measurements, target = _scene()
    baseline = triangulate_bearings_spherical_wls(stations, measurements)
    rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.6]).as_matrix()
    translation = np.asarray([100.0, -40.0, 25.0])
    transformed_stations = [
        StationPose(
            station.station_id,
            rotation @ station.position_world_m + translation,
            rotation @ station.rotation_local_to_world,
            station.microphone_positions_local_m,
        )
        for station in stations
    ]
    transformed = triangulate_bearings_spherical_wls(
        transformed_stations, measurements
    )
    assert baseline.valid and transformed.valid
    np.testing.assert_allclose(
        transformed.position_world_m,
        rotation @ target + translation,
        atol=5e-13,
    )
    np.testing.assert_allclose(
        transformed.covariance_position_m2,
        rotation @ baseline.covariance_position_m2 @ rotation.T,
        rtol=2e-12,
        atol=2e-15,
    )


def test_backward_bearing_returns_explicit_invalid_result():
    stations, measurements, _ = _scene()
    bad = measurements[2]
    measurements[2] = BearingMeasurement(
        bad.station_id,
        bad.sequence_id,
        bad.frame_index,
        bad.reception_center_timestamp_s,
        bad.available_timestamp_s,
        -bad.direction_local,
        bad.covariance_tangent_rad2,
        bad.calibration_bias_tangent_rad,
        bad.estimator_variant,
    )
    result = triangulate_bearings_spherical_wls(stations, measurements)
    assert not result.valid
    assert result.failure_reason in {
        "estimated_source_not_forward_of_all_rays",
        "degenerate_ray_geometry",
    }


def test_wrap_safe_residual_and_analytic_numeric_jacobian_agreement():
    station = StationPose("s0", np.zeros(3), np.eye(3), tetrahedral_array())
    phi_predicted = np.deg2rad(179.9)
    elevation = np.deg2rad(20.0)
    target = 30.0 * np.asarray(
        [
            np.cos(elevation) * np.cos(phi_predicted),
            np.cos(elevation) * np.sin(phi_predicted),
            np.sin(elevation),
        ]
    )
    measured = np.asarray(
        [
            np.cos(elevation) * np.cos(np.deg2rad(-179.9)),
            np.cos(elevation) * np.sin(np.deg2rad(-179.9)),
            np.sin(elevation),
        ]
    )
    measurement = BearingMeasurement(
        "s0", "wrap", 0, 1.0, 1.01, measured, COVARIANCE,
        np.zeros(2), "direct"
    )
    residual = bearing_residual(target, station, measurement)
    assert np.linalg.norm(residual) < np.deg2rad(0.21)
    analytic = bearing_residual_jacobian(target, station, measurement)
    numerical = numerical_bearing_residual_jacobian(target, station, measurement)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-10)


def test_calibration_bias_is_used_in_spherical_model_not_evaluation_mean():
    bias = np.deg2rad([0.4, -0.25])
    stations, corrected_measurements, target = _scene(
        biases=[bias, bias, bias]
    )
    corrected = triangulate_bearings_spherical_wls(stations, corrected_measurements)
    assert corrected.valid
    np.testing.assert_allclose(corrected.position_world_m, target, atol=2e-11)

    uncorrected_measurements = [
        BearingMeasurement(
            item.station_id,
            item.sequence_id,
            item.frame_index,
            item.reception_center_timestamp_s,
            item.available_timestamp_s,
            item.direction_local,
            item.covariance_tangent_rad2,
            np.zeros(2),
            item.estimator_variant,
        )
        for item in corrected_measurements
    ]
    uncorrected = triangulate_bearings_spherical_wls(
        stations, uncorrected_measurements
    )
    assert uncorrected.valid
    assert np.linalg.norm(uncorrected.position_world_m - target) > 0.05
    assert not hasattr(corrected_measurements[0], "evaluation_mean_residual")


def test_static_fusion_uses_state_association_not_equal_reception_timestamp():
    stations, measurements, _ = _scene()
    item = measurements[1]
    measurements[1] = BearingMeasurement(
        item.station_id,
        item.sequence_id,
        item.frame_index,
        item.reception_center_timestamp_s + 0.01,
        item.available_timestamp_s + 0.01,
        item.direction_local,
        item.covariance_tangent_rad2,
        item.calibration_bias_tangent_rad,
        item.estimator_variant,
    )
    # Same associated static state may arrive at separated stations at
    # different reception times.  The default must not equate reception and
    # emission timestamps.
    result = triangulate_bearings_spherical_wls(stations, measurements)
    assert result.valid
    with pytest.raises(ValueError, match="reception timestamps"):
        triangulate_bearings_spherical_wls(
            stations, measurements, timestamp_tolerance_s=1e-6
        )

    item = measurements[1]
    measurements[1] = BearingMeasurement(
        item.station_id,
        item.sequence_id,
        item.frame_index + 1,
        item.reception_center_timestamp_s,
        item.available_timestamp_s,
        item.direction_local,
        item.covariance_tangent_rad2,
        item.calibration_bias_tangent_rad,
        item.estimator_variant,
    )
    with pytest.raises(ValueError, match="frame_index"):
        triangulate_bearings_spherical_wls(stations, measurements)
