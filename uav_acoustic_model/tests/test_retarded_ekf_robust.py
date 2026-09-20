"""Deterministic gates for the opt-in robust strict-CV bearing EKF."""

from dataclasses import replace

import numpy as np

from estimators.retarded_ekf import (
    CausalRetardedTimeEKF,
    RetardedEKFRobustnessConfig,
    update_retarded_ekf,
)
from model.bearing_events import bearing_event_id
from model.bearing_statistics import tangent_basis
from model.dynamic_state import ConstantVelocityState
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import predict_retarded_bearing
from model.station import StationPose
from validation.retarded_ekf_stress_study import (
    default_stress_profiles,
    generate_stress_base_block,
    generate_stress_scenario,
)


NIS_99 = 9.210340371976184


def _stations(*, degenerate: bool = False) -> tuple[StationPose, ...]:
    positions = (
        ([0.0, 0.0, 0.0],) * 3
        if degenerate
        else ([0.0, 0.0, 0.0], [100.0, 0.0, 5.0], [10.0, 90.0, -2.0])
    )
    return tuple(
        StationPose(f"S{index}", position, np.eye(3), tetrahedral_array())
        for index, position in enumerate(positions)
    )


def _exp_map(direction: np.ndarray, tangent_offset: np.ndarray) -> np.ndarray:
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ np.asarray(tangent_offset)
    angle = float(np.linalg.norm(tangent))
    if angle == 0.0:
        return np.asarray(direction)
    return np.cos(angle) * direction + np.sin(angle) * tangent / angle


def _measurements(
    state: ConstantVelocityState,
    stations: tuple[StationPose, ...],
    *,
    count_per_station: int = 5,
) -> list[BearingMeasurement]:
    covariance = np.diag(np.deg2rad([0.3, 0.5]) ** 2)
    starts = (0.60, 0.75, 0.90)
    result: list[BearingMeasurement] = []
    for station_index, station in enumerate(stations):
        for frame_index in range(count_per_station):
            reception = starts[station_index] + 0.8 * frame_index
            prediction = predict_retarded_bearing(state, station, reception)
            result.append(
                BearingMeasurement(
                    station.station_id,
                    "robust-test",
                    frame_index,
                    reception,
                    reception + 0.02 + 0.003 * station_index,
                    prediction.direction_local,
                    covariance,
                    np.zeros(2),
                    "direct",
                    tangent_frame="prediction",
                )
            )
    return result


def _corrupt(
    measurements: list[BearingMeasurement], event_id: str, angle_deg: float
) -> list[BearingMeasurement]:
    result = []
    for measurement in measurements:
        if bearing_event_id(measurement) == event_id:
            result.append(
                replace(
                    measurement,
                    direction_local=_exp_map(
                        measurement.direction_local,
                        np.deg2rad([angle_deg, 0.0]),
                    ),
                )
            )
        else:
            result.append(measurement)
    return result


def _combined() -> RetardedEKFRobustnessConfig:
    return RetardedEKFRobustnessConfig(
        consensus_initialization=True,
        maximum_pre_update_nis=NIS_99,
    )


def test_default_and_explicit_empty_robustness_config_reproduce_c1():
    base = generate_stress_base_block("informative", 3, base_seed=20260911)
    profile = next(item for item in default_stress_profiles() if item.name == "nominal")
    scenario = generate_stress_scenario(base, profile)
    default = CausalRetardedTimeEKF(
        base.stations, scenario.events, estimator_variant="direct_bearing"
    ).advance_to(14.5)
    explicit = CausalRetardedTimeEKF(
        base.stations,
        scenario.events,
        estimator_variant="direct_bearing",
        robustness_config=RetardedEKFRobustnessConfig(),
    ).advance_to(14.5)
    assert default.valid and explicit.valid
    np.testing.assert_allclose(default.state.vector, explicit.state.vector, rtol=0, atol=0)
    np.testing.assert_allclose(
        default.covariance_state, explicit.covariance_state, rtol=0, atol=0
    )
    assert default.initialization_event_ids == explicit.initialization_event_ids
    assert default.applied_event_ids == explicit.applied_event_ids
    assert default.rejected_event_ids == explicit.rejected_event_ids
    np.testing.assert_allclose(
        [item.normalized_innovation_squared for item in default.update_diagnostics],
        [item.normalized_innovation_squared for item in explicit.update_diagnostics],
        rtol=0,
        atol=0,
    )


def test_pre_update_nis_gate_preserves_prior_exactly_and_retains_nis():
    stations = _stations()
    state = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0], 2.0)
    measurement = _measurements(state, stations)[0]
    corrupted = replace(
        measurement,
        direction_local=_exp_map(measurement.direction_local, np.deg2rad([20.0, 0.0])),
    )
    covariance = np.diag([0.2, 0.2, 0.2, 0.02, 0.02, 0.02])
    result = update_retarded_ekf(
        state,
        covariance,
        stations[0],
        corrupted,
        maximum_pre_update_nis=NIS_99,
    )
    assert not result.update_applied
    assert result.failure_reason == "pre_update_nis_gate"
    assert np.isfinite(result.normalized_innovation_squared)
    assert result.normalized_innovation_squared > NIS_99
    np.testing.assert_allclose(result.posterior_state.vector, state.vector, rtol=0, atol=0)
    np.testing.assert_allclose(result.posterior_covariance, covariance, rtol=0, atol=0)


def test_clean_update_passes_nis_gate():
    stations = _stations()
    state = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0], 2.0)
    measurement = _measurements(state, stations)[0]
    result = update_retarded_ekf(
        state,
        np.diag([0.2, 0.2, 0.2, 0.02, 0.02, 0.02]),
        stations[0],
        measurement,
        maximum_pre_update_nis=NIS_99,
    )
    assert result.valid and result.update_applied
    assert result.failure_reason is None
    assert result.normalized_innovation_squared < NIS_99


def test_consensus_excludes_early_outlier_and_never_reuses_prefix_events():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    ordered = sorted(measurements, key=lambda item: item.available_timestamp_s)
    outlier_id = bearing_event_id(ordered[1])
    corrupted = _corrupt(measurements, outlier_id, 20.0)
    result = CausalRetardedTimeEKF(
        stations,
        corrupted,
        estimator_variant="direct",
        robustness_config=RetardedEKFRobustnessConfig(consensus_initialization=True),
    ).advance_to(10.0)
    assert result.valid
    reasons = {item.event_id: item.reason for item in result.event_rejections}
    assert reasons[outlier_id] == "robust_initialization_consensus_outlier"
    assert outlier_id not in result.initialization_event_ids
    assert outlier_id not in result.applied_event_ids
    assert set(result.initialization_event_ids).isdisjoint(result.applied_event_ids)
    assert set(result.rejected_event_ids).isdisjoint(result.applied_event_ids)
    successful = [item for item in result.initialization_diagnostics if item.succeeded]
    assert len(successful) == 1
    assert outlier_id in successful[0].excluded_event_ids


def test_insufficient_consensus_and_degenerate_geometry_are_explicit():
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    stations = _stations()
    measurements = sorted(
        _measurements(truth, stations), key=lambda item: item.available_timestamp_s
    )[:6]
    corrupted = _corrupt(measurements, bearing_event_id(measurements[0]), 20.0)
    result = CausalRetardedTimeEKF(
        stations,
        corrupted,
        estimator_variant="direct",
        robustness_config=RetardedEKFRobustnessConfig(consensus_initialization=True),
    ).advance_to(10.0)
    assert not result.valid
    assert result.failure_reason == "initialization_failed:robust_consensus_not_found"
    assert result.initialization_diagnostics[-1].failure_reason == "robust_consensus_not_found"

    degenerate = _stations(degenerate=True)
    stationary_truth = ConstantVelocityState([50.0, 40.0, 30.0], [0.0, 0.0, 0.0])
    degenerate_result = CausalRetardedTimeEKF(
        degenerate,
        _measurements(stationary_truth, degenerate),
        estimator_variant="direct",
        robustness_config=RetardedEKFRobustnessConfig(consensus_initialization=True),
    ).advance_to(10.0)
    assert not degenerate_result.valid
    assert degenerate_result.failure_reason is not None


def test_late_outlier_gate_matches_stream_with_event_absent():
    stations = _stations()
    truth = ConstantVelocityState([50.0, 40.0, 30.0], [5.0, -2.0, 1.0])
    measurements = _measurements(truth, stations)
    ordered = sorted(measurements, key=lambda item: item.available_timestamp_s)
    outlier_id = bearing_event_id(ordered[9])
    corrupted = _corrupt(measurements, outlier_id, 20.0)
    gated = CausalRetardedTimeEKF(
        stations,
        corrupted,
        estimator_variant="direct",
        robustness_config=_combined(),
    ).advance_to(10.0)
    absent = CausalRetardedTimeEKF(
        stations,
        [item for item in corrupted if bearing_event_id(item) != outlier_id],
        estimator_variant="direct",
        robustness_config=_combined(),
    ).advance_to(10.0)
    assert gated.valid and absent.valid
    assert {item.event_id: item.reason for item in gated.event_rejections}[outlier_id] == (
        "pre_update_nis_gate"
    )
    np.testing.assert_allclose(gated.state.vector, absent.state.vector, rtol=0, atol=2e-12)
    np.testing.assert_allclose(
        gated.covariance_state, absent.covariance_state, rtol=0, atol=2e-12
    )


def test_robust_result_is_independent_of_external_advance_schedule():
    base = generate_stress_base_block("informative", 0, base_seed=20260911)
    profile = next(
        item for item in default_stress_profiles() if item.name == "outlier_strong"
    )
    scenario = generate_stress_scenario(base, profile)
    frequent = CausalRetardedTimeEKF(
        base.stations,
        scenario.events,
        estimator_variant="direct_bearing",
        robustness_config=_combined(),
    )
    for timestamp in sorted({item.available_timestamp_s for item in scenario.events}):
        frequent_result = frequent.advance_to(timestamp)
    one_shot = CausalRetardedTimeEKF(
        base.stations,
        scenario.events,
        estimator_variant="direct_bearing",
        robustness_config=_combined(),
    ).advance_to(max(item.available_timestamp_s for item in scenario.events))
    assert frequent_result.valid and one_shot.valid
    np.testing.assert_allclose(
        frequent_result.state.vector, one_shot.state.vector, rtol=0, atol=3e-10
    )
    np.testing.assert_allclose(
        frequent_result.covariance_state,
        one_shot.covariance_state,
        rtol=0,
        atol=3e-9,
    )
    assert frequent_result.initialization_event_ids == one_shot.initialization_event_ids
    assert frequent_result.applied_event_ids == one_shot.applied_event_ids
    assert frequent_result.rejected_event_ids == one_shot.rejected_event_ids


def test_dropout_and_long_delay_smoke_remain_causal():
    base = generate_stress_base_block("poorly_conditioned", 1, base_seed=20260911)
    profiles = {item.name: item for item in default_stress_profiles()}
    for profile_name in ("dropout_50", "long_delay"):
        scenario = generate_stress_scenario(base, profiles[profile_name])
        processor = CausalRetardedTimeEKF(
            base.stations,
            scenario.events,
            estimator_variant="direct_bearing",
            robustness_config=_combined(),
        )
        early = processor.advance_to(7.0)
        late = processor.advance_to(14.5)
        assert all(
            event.available_timestamp_s <= early.processing_time_s
            for event in early.prefix.measurements
        )
        assert late.valid
        assert set(late.initialization_event_ids).isdisjoint(late.applied_event_ids)
        assert set(late.rejected_event_ids).isdisjoint(late.applied_event_ids)
