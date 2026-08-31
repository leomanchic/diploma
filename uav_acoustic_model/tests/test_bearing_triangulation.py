"""Deterministic static spherical bearing-triangulation tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import estimators.bearing_triangulation as triangulation_module
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


def _measurements_with_tangent_offsets(stations, target, offsets, covariance):
    measurements = []
    for station, offset in zip(stations, offsets, strict=True):
        world_direction = target - station.position_world_m
        world_direction /= np.linalg.norm(world_direction)
        local_direction = station.world_to_local_direction(world_direction)
        measurements.append(
            BearingMeasurement(
                station.station_id,
                "singular-covariance",
                0,
                0.0,
                0.01,
                _exp_map(local_direction, np.asarray(offset, dtype=float)),
                covariance,
                np.zeros(2),
                "direct_bearing",
            )
        )
    return measurements


def _preliminary_false_invalid_scene():
    """Compatible rank-1 scene whose preliminary residual exceeds 1e-10 rad."""

    positions = [
        np.zeros(3),
        np.asarray(
            [2.6187739630770173, -3.7639142858335752, -0.6555642820150913]
        ),
    ]
    rotations = [
        np.asarray(
            [
                [-0.1815239609872818, 0.9522455918315212, 0.24551452996701423],
                [-0.599401613341004, 0.09078324025270512, -0.7952836658786735],
                [-0.7795939696481045, -0.29152484649966937, 0.5542982106787417],
            ]
        ),
        np.asarray(
            [
                [0.6755760966566243, -0.5943121581691221, 0.43633702143934494],
                [-0.05288270742599477, -0.6293468449548808, -0.7753231377952249],
                [0.7353912950813979, 0.5007150960353574, -0.45660052093794945],
            ]
        ),
    ]
    target = np.asarray([4.720723664433317, -43.01134796374569, 73.40353800707986])
    covariance = np.asarray(
        [
            [3.0968741635773465e-06, -6.237830354392658e-06],
            [-6.237830354392658e-06, 1.256445224278504e-05],
        ]
    )
    offsets = [
        np.asarray([0.01890537856186321, -0.03807986312180334]),
        np.asarray([-0.00125966859196989, 0.00253726776240342]),
    ]
    stations = [
        StationPose(f"S{index}", position, rotation, tetrahedral_array(0.2))
        for index, (position, rotation) in enumerate(
            zip(positions, rotations, strict=True)
        )
    ]
    measurements = _measurements_with_tangent_offsets(
        stations, target, offsets, covariance
    )
    return stations, measurements, target


def _randomized_rank_one_scenes(*, scene_count=1000, seed=20260831):
    """Yield pinned deterministic compatible two-station rank-1 scenes."""

    rng = np.random.default_rng(seed)
    microphone_array = tetrahedral_array(0.2)
    for trial in range(scene_count):
        baseline = rng.uniform(8.0, 40.0)
        baseline_azimuth = rng.uniform(-np.pi, np.pi)
        second_position = np.asarray(
            [
                baseline * np.cos(baseline_azimuth),
                baseline * np.sin(baseline_azimuth),
                rng.uniform(-2.0, 2.0),
            ]
        )
        positions = [np.zeros(3), second_position]
        midpoint = 0.5 * second_position
        source_azimuth = rng.uniform(-np.pi, np.pi)
        source_elevation = np.deg2rad(rng.uniform(15.0, 70.0))
        source_direction = np.asarray(
            [
                np.cos(source_elevation) * np.cos(source_azimuth),
                np.cos(source_elevation) * np.sin(source_azimuth),
                np.sin(source_elevation),
            ]
        )
        target = midpoint + rng.uniform(25.0, 120.0) * source_direction
        rotations = [
            Rotation.random(random_state=rng).as_matrix() for _ in range(2)
        ]
        stations = [
            StationPose(f"S{index}", position, rotation, microphone_array)
            for index, (position, rotation) in enumerate(
                zip(positions, rotations, strict=True)
            )
        ]
        covariance_angle = rng.uniform(-np.pi, np.pi)
        covariance_basis = np.asarray(
            [
                [np.cos(covariance_angle), -np.sin(covariance_angle)],
                [np.sin(covariance_angle), np.cos(covariance_angle)],
            ]
        )
        sigma = np.deg2rad(rng.uniform(0.2, 2.0))
        covariance = (
            covariance_basis @ np.diag([sigma**2, 0.0]) @ covariance_basis.T
        )
        offsets = [
            rng.normal(0.0, 0.5 * sigma) * covariance_basis[:, 0]
            for _ in stations
        ]
        measurements = _measurements_with_tangent_offsets(
            stations, target, offsets, covariance
        )
        yield trial, stations, measurements


def _run_randomized_rank_one_audit(*, scene_count=1000, seed=20260831):
    """Return deterministic robustness diagnostics for compatible rank-1 scenes."""

    failures = []
    max_preliminary = 0.0
    max_final = 0.0
    max_raw_gradient = 0.0
    max_scaled_kkt = 0.0
    preliminary_exceedance_count = 0
    for trial, stations, measurements in _randomized_rank_one_scenes(
        scene_count=scene_count, seed=seed
    ):
        result = triangulate_bearings_spherical_wls(stations, measurements)
        if not result.valid:
            failures.append(
                (
                    trial,
                    result.failure_reason,
                    result.preliminary_constraint_max_abs_rad,
                    result.constraint_max_abs_rad,
                    result.raw_projected_gradient_norm,
                    result.scaled_projected_kkt_residual,
                    result.optimizer_success,
                    result.optimizer_message,
                )
            )
        preliminary_exceedance_count += int(
            result.preliminary_constraint_max_abs_rad > 1e-10
        )
        max_preliminary = max(
            max_preliminary, result.preliminary_constraint_max_abs_rad
        )
        max_final = max(max_final, result.constraint_max_abs_rad)
        max_raw_gradient = max(
            max_raw_gradient, result.raw_projected_gradient_norm
        )
        max_scaled_kkt = max(
            max_scaled_kkt, result.scaled_projected_kkt_residual
        )
    return {
        "scene_count": scene_count,
        "seed": seed,
        "false_invalid_count": len(failures),
        "failures": failures,
        "preliminary_exceedance_count": preliminary_exceedance_count,
        "max_preliminary_constraint_residual_rad": max_preliminary,
        "max_final_constraint_residual_rad": max_final,
        "max_raw_projected_gradient_norm": max_raw_gradient,
        "max_scaled_projected_kkt_residual": max_scaled_kkt,
    }


def test_three_exact_nonparallel_bearings_with_zero_covariance_recover_position():
    stations, _, target = _scene()
    measurements = _measurements_with_tangent_offsets(
        stations, target, np.zeros((3, 2)), np.zeros((2, 2))
    )
    result = triangulate_bearings_spherical_wls(stations, measurements)
    assert result.valid
    assert result.positive_variance_residual_dimension == 0
    assert result.exact_constraint_dimension == 6
    assert result.constraint_rank == 3
    assert result.free_parameter_dimension == 0
    assert result.local_observability_rank == 3
    assert result.constraints_satisfied
    assert result.constraint_max_abs_rad < 1e-13
    np.testing.assert_allclose(result.position_world_m, target, atol=2e-13)
    np.testing.assert_array_equal(result.covariance_position_m2, np.zeros((3, 3)))


def test_rank_one_covariance_enforces_zero_variance_component_exactly():
    stations, _, target = _scene()
    offsets = np.column_stack((np.deg2rad([0.8, -0.5]), np.zeros(2)))
    covariance = np.diag([np.deg2rad(1.0) ** 2, 0.0])
    measurements = _measurements_with_tangent_offsets(
        stations[:2], target, offsets, covariance
    )
    result = triangulate_bearings_spherical_wls(stations[:2], measurements)
    assert result.valid
    assert result.positive_variance_residual_dimension == 2
    assert result.exact_constraint_dimension == 2
    assert result.constraint_rank == 2
    assert result.free_parameter_dimension == 1
    assert result.reduced_information_rank == 1
    assert result.local_observability_rank == 3
    assert result.constraints_satisfied
    assert result.constraint_max_abs_rad < 1e-11
    np.testing.assert_allclose(result.exact_constraint_residuals, 0.0, atol=1e-11)
    np.testing.assert_allclose(
        result.constraint_jacobian
        @ result.covariance_position_m2
        @ result.constraint_jacobian.T,
        0.0,
        atol=1e-13,
    )


def test_preliminary_feasibility_exceedance_does_not_cause_false_invalid():
    stations, measurements, _ = _preliminary_false_invalid_scene()
    result = triangulate_bearings_spherical_wls(stations, measurements)
    assert result.constrained_optimization_performed
    assert result.preliminary_constraint_max_abs_rad > 1e-10
    assert result.preliminary_constraint_max_abs_rad < 2e-10
    assert result.valid
    assert result.failure_reason is None
    assert result.constraints_satisfied
    assert result.constraint_max_abs_rad < 1e-14
    assert result.projected_kkt_satisfied
    assert result.scaled_projected_kkt_residual < 1e-10


@pytest.mark.parametrize("trial_index", [771, 793])
def test_pinned_linux_rank_one_regressions_are_valid(trial_index):
    scene = next(
        scene
        for scene in _randomized_rank_one_scenes(
            scene_count=trial_index + 1, seed=20260831
        )
        if scene[0] == trial_index
    )
    _, stations, measurements = scene
    result = triangulate_bearings_spherical_wls(stations, measurements)
    assert result.valid, (
        result.failure_reason,
        result.optimizer_success,
        result.optimizer_message,
        result.constraint_max_abs_rad,
        result.raw_projected_gradient_norm,
        result.scaled_projected_kkt_residual,
    )
    assert np.all(np.isfinite(result.position_world_m))
    assert result.constraints_satisfied
    assert result.constraint_max_abs_rad < 1e-12
    assert result.local_observability_rank == 3
    assert result.projected_kkt_satisfied
    assert result.scaled_projected_kkt_residual <= 1e-6


def test_optimizer_exit_failure_is_diagnostic_when_acceptance_checks_pass(
    monkeypatch,
):
    stations, measurements, _ = _preliminary_false_invalid_scene()
    optimum = triangulate_bearings_spherical_wls(stations, measurements)
    assert optimum.valid

    def fake_failed_minimize(*args, **kwargs):
        return SimpleNamespace(
            x=np.asarray(optimum.position_world_m),
            success=False,
            message="The maximum number of function evaluations is exceeded.",
            niter=1,
        )

    monkeypatch.setattr(triangulation_module, "minimize", fake_failed_minimize)
    result = triangulation_module.triangulate_bearings_spherical_wls(
        stations, measurements
    )
    assert not result.optimizer_success
    assert result.optimizer_message.startswith("The maximum number")
    assert np.all(np.isfinite(result.position_world_m))
    assert result.constraints_satisfied
    assert result.local_observability_rank == 3
    assert result.projected_kkt_satisfied
    assert result.valid
    assert result.failure_reason is None


def test_scaled_projected_kkt_default_tolerance_is_dimensionless_one_millionth():
    stations, measurements, _ = _preliminary_false_invalid_scene()
    result = triangulate_bearings_spherical_wls(stations, measurements)
    assert result.valid
    assert result.projected_kkt_tolerance == pytest.approx(1e-6)
    assert result.scaled_projected_kkt_residual <= 1e-6


def test_xtol_success_is_rejected_when_projected_kkt_is_not_satisfied(monkeypatch):
    stations, measurements, target = _preliminary_false_invalid_scene()

    def fake_xtol_minimize(*args, **kwargs):
        return SimpleNamespace(
            x=np.asarray(target),
            success=True,
            message="`xtol` termination condition is satisfied.",
            niter=1,
        )

    monkeypatch.setattr(triangulation_module, "minimize", fake_xtol_minimize)
    result = triangulation_module.triangulate_bearings_spherical_wls(
        stations, measurements
    )
    assert result.constraints_satisfied
    assert result.optimizer_message.startswith("`xtol`")
    assert not result.projected_kkt_satisfied
    assert result.scaled_projected_kkt_residual > result.projected_kkt_tolerance
    assert not result.valid
    assert result.failure_reason == "projected_kkt_not_satisfied"


def test_randomized_compatible_two_station_rank_one_scenes_have_no_false_invalid():
    audit = _run_randomized_rank_one_audit(scene_count=1000, seed=20260831)
    assert audit["scene_count"] == 1000
    assert audit["false_invalid_count"] == 0, audit["failures"][:5]
    assert audit["preliminary_exceedance_count"] > 0
    assert audit["max_final_constraint_residual_rad"] < 1e-12
    assert np.isfinite(audit["max_raw_projected_gradient_norm"])
    assert audit["max_scaled_projected_kkt_residual"] <= 1e-6


def test_incompatible_deterministic_bearings_return_explicit_invalid():
    stations, measurements, _ = _scene()
    incompatible = []
    for index, item in enumerate(measurements):
        direction = item.direction_local
        if index == 2:
            direction = _exp_map(direction, np.deg2rad([4.0, -3.0]))
        incompatible.append(
            BearingMeasurement(
                item.station_id,
                "incompatible",
                0,
                0.0,
                0.01,
                direction,
                np.zeros((2, 2)),
                np.zeros(2),
                "direct_bearing",
            )
        )
    result = triangulate_bearings_spherical_wls(stations, incompatible)
    assert not result.valid
    assert result.failure_reason == "incompatible_exact_constraints"
    assert not result.constraints_satisfied
    assert result.constraint_max_abs_rad > np.deg2rad(0.1)
    assert np.all(np.isnan(result.covariance_position_m2))


def test_positive_variance_limit_converges_to_constrained_solution():
    stations, _, target = _scene()
    offsets = np.column_stack((np.deg2rad([0.8, -0.5, 0.3]), np.zeros(3)))
    azimuth_variance = np.deg2rad(1.0) ** 2
    constrained = triangulate_bearings_spherical_wls(
        stations,
        _measurements_with_tangent_offsets(
            stations, target, offsets, np.diag([azimuth_variance, 0.0])
        ),
    )
    distances = []
    for variance in (1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        finite = triangulate_bearings_spherical_wls(
            stations,
            _measurements_with_tangent_offsets(
                stations,
                target,
                offsets,
                np.diag([azimuth_variance, variance]),
            ),
        )
        assert finite.valid
        assert finite.exact_constraint_dimension == 0
        distances.append(
            float(np.linalg.norm(finite.position_world_m - constrained.position_world_m))
        )
    assert all(current < previous for previous, current in zip(distances, distances[1:]))
    assert distances[-1] < 2e-9


def test_singular_constrained_solution_preserves_permutation_and_rigid_invariance():
    rotations = [
        Rotation.from_euler("zyx", [0.2, -0.1, 0.05]).as_matrix(),
        Rotation.from_euler("zyx", [-0.7, 0.15, -0.2]).as_matrix(),
    ]
    stations, _, target = _scene(rotations=rotations + [np.eye(3)])
    stations = stations[:2]
    offsets = np.column_stack((np.deg2rad([0.8, -0.5]), np.zeros(2)))
    covariance = np.diag([np.deg2rad(1.0) ** 2, 0.0])
    measurements = _measurements_with_tangent_offsets(
        stations, target, offsets, covariance
    )
    baseline = triangulate_bearings_spherical_wls(stations, measurements)
    permuted = triangulate_bearings_spherical_wls(
        stations[::-1], measurements[::-1]
    )
    assert baseline.valid and permuted.valid
    np.testing.assert_allclose(
        permuted.position_world_m, baseline.position_world_m, atol=2e-9
    )
    np.testing.assert_allclose(
        permuted.covariance_position_m2,
        baseline.covariance_position_m2,
        rtol=2e-8,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        permuted.scaled_projected_kkt_residual,
        baseline.scaled_projected_kkt_residual,
        rtol=5e-2,
        atol=1e-8,
    )

    global_rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.6]).as_matrix()
    translation = np.asarray([100.0, -40.0, 25.0])
    transformed_stations = [
        StationPose(
            station.station_id,
            global_rotation @ station.position_world_m + translation,
            global_rotation @ station.rotation_local_to_world,
            station.microphone_positions_local_m,
        )
        for station in stations
    ]
    transformed = triangulate_bearings_spherical_wls(
        transformed_stations, measurements
    )
    assert transformed.valid
    np.testing.assert_allclose(
        transformed.position_world_m,
        global_rotation @ baseline.position_world_m + translation,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        transformed.covariance_position_m2,
        global_rotation @ baseline.covariance_position_m2 @ global_rotation.T,
        rtol=2e-8,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        transformed.scaled_projected_kkt_residual,
        baseline.scaled_projected_kkt_residual,
        rtol=5e-2,
        atol=1e-8,
    )

    scale = 7.5
    scaled_stations = [
        StationPose(
            station.station_id,
            scale * station.position_world_m,
            station.rotation_local_to_world,
            scale * station.microphone_positions_local_m,
        )
        for station in stations
    ]
    scaled = triangulate_bearings_spherical_wls(scaled_stations, measurements)
    assert scaled.valid
    np.testing.assert_allclose(
        scaled.position_world_m, scale * baseline.position_world_m, atol=2e-8
    )
    np.testing.assert_allclose(
        scaled.covariance_position_m2,
        scale**2 * baseline.covariance_position_m2,
        rtol=2e-8,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        scaled.scaled_projected_kkt_residual,
        baseline.scaled_projected_kkt_residual,
        rtol=5e-2,
        atol=1e-8,
    )
