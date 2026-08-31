"""Position-information rank, conditioning, and physical scaling tests."""

import numpy as np

from estimators.bearing_triangulation import triangulate_bearings_spherical_wls
from model.geometry import tetrahedral_array
from model.measurements import BearingMeasurement
from model.station import StationPose


def _ideal_scene(positions, target, covariance):
    stations = []
    measurements = []
    for index, position in enumerate(positions):
        station = StationPose(
            f"s{index}", np.asarray(position, dtype=float), np.eye(3), tetrahedral_array()
        )
        direction = np.asarray(target, dtype=float) - station.position_world_m
        direction /= np.linalg.norm(direction)
        measurements.append(
            BearingMeasurement(
                station.station_id,
                "state",
                0,
                0.0,
                0.01,
                direction,
                covariance,
                np.zeros(2),
                "ideal",
            )
        )
        stations.append(station)
    return stations, measurements


def test_collinear_stations_and_nearly_parallel_rays_report_bad_observability():
    covariance = np.diag(np.deg2rad([0.2, 0.2]) ** 2)
    positions = [np.asarray([0.0, 0.0, 0.0]), np.asarray([20.0, 0.0, 0.0]), np.asarray([40.0, 0.0, 0.0])]
    target = np.asarray([2.0e6, 3.0e5, 4.0e5])
    stations, measurements = _ideal_scene(positions, target, covariance)
    result = triangulate_bearings_spherical_wls(stations, measurements)
    assert (not result.valid) or result.information_condition_number > 1e8
    if result.valid:
        assert result.gdop_like_sqrt_trace_m > 1e4
    else:
        assert result.failure_reason in {
            "degenerate_ray_geometry",
            "degenerate_position_information",
        }


def test_position_covariance_scales_with_scene_scale_squared():
    covariance = np.diag(np.deg2rad([0.15, 0.25]) ** 2)
    positions = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([15.0, 0.0, 0.0]),
        np.asarray([4.0, 13.0, 2.0]),
    ]
    target = np.asarray([7.0, 6.0, 11.0])
    stations, measurements = _ideal_scene(positions, target, covariance)
    baseline = triangulate_bearings_spherical_wls(stations, measurements)
    scale = 7.5
    scaled_stations, scaled_measurements = _ideal_scene(
        [scale * item for item in positions], scale * target, covariance
    )
    scaled = triangulate_bearings_spherical_wls(
        scaled_stations, scaled_measurements
    )
    assert baseline.valid and scaled.valid
    np.testing.assert_allclose(scaled.position_world_m, scale * target, atol=2e-12)
    np.testing.assert_allclose(
        scaled.covariance_position_m2,
        scale**2 * baseline.covariance_position_m2,
        rtol=3e-13,
        atol=2e-14,
    )


def test_zero_and_singular_angular_covariance_are_not_hidden_by_epsilon():
    positions = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([20.0, 0.0, 0.0]),
        np.asarray([8.0, 16.0, 0.0]),
    ]
    target = np.asarray([7.0, 5.0, 12.0])
    zero_stations, zero_measurements = _ideal_scene(
        positions, target, np.zeros((2, 2))
    )
    zero = triangulate_bearings_spherical_wls(zero_stations, zero_measurements)
    assert not zero.valid
    assert zero.information_rank == 0
    assert np.all(np.isnan(zero.covariance_position_m2))
    assert zero.failure_reason == "degenerate_position_information"

    singular_stations, singular_measurements = _ideal_scene(
        positions, target, np.diag([np.deg2rad(0.2) ** 2, 0.0])
    )
    singular = triangulate_bearings_spherical_wls(
        singular_stations, singular_measurements
    )
    assert singular.information_rank <= 3
    if singular.information_rank < 3:
        assert not singular.valid
        assert np.all(np.isnan(singular.covariance_position_m2))
        assert singular.unobservable_directions_world.shape[0] >= 1


def test_nearly_parallel_geometry_has_larger_covariance_than_wide_intersection():
    covariance = np.diag(np.deg2rad([0.2, 0.2]) ** 2)
    positions = [
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([20.0, 0.0, 0.0]),
        np.asarray([8.0, 17.0, 0.0]),
    ]
    near_target = np.asarray([8.0, 7.0, 12.0])
    far_target = 100.0 * near_target
    near_stations, near_measurements = _ideal_scene(positions, near_target, covariance)
    far_stations, far_measurements = _ideal_scene(positions, far_target, covariance)
    near = triangulate_bearings_spherical_wls(near_stations, near_measurements)
    far = triangulate_bearings_spherical_wls(far_stations, far_measurements)
    assert near.valid and far.valid
    assert far.information_condition_number > near.information_condition_number
    assert far.gdop_like_sqrt_trace_m > 100.0 * near.gdop_like_sqrt_trace_m

