"""Regression gates for the frozen opt-in confirmation comparison."""

from dataclasses import replace

import numpy as np
import pytest

from analysis.robust_track_confirmation import CHI3, NOISE_SEEDS, nominal_state_uncertainty
from estimators.retarded_ekf_manoeuvre import AugmentedMotionHistory, ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import (
    CausalConfirmedRetardedTimeEKF, InitializationRecoveryConfig,
)
from model.bearing_events import bearing_event_id
from model.bearing_statistics import tangent_basis
from model.dynamic_state import ConstantVelocityState
from model.geometry import direction_angles, tetrahedral_array
from model.measurements import BearingMeasurement
from model.retarded_bearing import predict_retarded_bearing
from model.station import StationPose
from validation.retarded_ekf_stress_study import (
    default_stress_profiles, generate_stress_base_block, generate_stress_scenario,
)


def test_frozen_noise_seeds_are_distinct_and_new():
    seeds = [seed for pair in NOISE_SEEDS for seed in pair]
    assert len(set(seeds)) == 4
    for trajectory_index in range(2):
        for replicate in range(2):
            expected = np.random.SeedSequence(
                [20260927, trajectory_index, replicate]
            ).generate_state(1, dtype=np.uint64)[0]
            assert NOISE_SEEDS[trajectory_index][replicate] == int(expected)


def test_covariance_report_contains_full_blocks_and_nominal_axes():
    p = np.diag([1.0, 4.0, 9.0, 0.25, 1.0, 4.0])
    p[0, 1] = p[1, 0] = 0.3
    result = nominal_state_uncertainty(p)
    np.testing.assert_allclose(result["position_covariance"], p[:3, :3])
    np.testing.assert_allclose(result["velocity_covariance"], p[3:, 3:])
    np.testing.assert_allclose(
        sorted(result["position_nominal_95_axes"]),
        np.sqrt(CHI3 * np.linalg.eigvalsh(p[:3, :3])),
    )
    np.testing.assert_allclose(result["velocity_sigma_rss"], np.sqrt(5.25))
    with pytest.raises(ValueError, match="positive definite"):
        nominal_state_uncertainty(np.diag([1.0, 0.0, 1.0, 1.0, 1.0, 1.0]))


def test_logged_pre_update_residual_reconstructs_nis_and_missing_is_explicit():
    station = StationPose("S0", [0, 0, 0], np.eye(3), tetrahedral_array())
    state = ConstantVelocityState([70, 55, 40], [7, -3, 1.5], 0.0)
    history = AugmentedMotionHistory(state, np.eye(6) * 0.5,
                                     ManoeuvreHistoryConfig(np.eye(3) * 0.25))
    history.propagate_to(1.5)
    direction = predict_retarded_bearing(state, station, 1.0).direction_local
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ np.deg2rad([0.2, 0.1])
    angle = np.linalg.norm(tangent)
    measured = np.cos(angle) * direction + np.sin(angle) * tangent / angle
    measurement = BearingMeasurement(
        "S0", "residual-check", 0, 1.0, 1.2, measured,
        np.diag(np.deg2rad([0.3, 0.5])**2), np.zeros(2),
        "direct_bearing", tangent_frame="measurement")
    result = history.update_bearing(station, measurement, maximum_pre_update_nis=1e9)
    assert result.update_applied
    residual = np.asarray(result.residual_tangent_rad)
    innovation = np.asarray(result.innovation_covariance_tangent_rad2)
    np.testing.assert_allclose(residual @ np.linalg.solve(innovation, residual),
                               result.pre_update_nis, rtol=0, atol=1e-11)
    assert result.residual_unavailable_reason is None
    history.propagate_to(6.0)
    missing = history.update_bearing(station, measurement, maximum_pre_update_nis=1e9)
    assert missing.residual_tangent_rad is None
    assert missing.residual_unavailable_reason == "emission_outside_history"


def test_opt_in_confirmation_keeps_prefix_causality_and_event_ledger():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    profile = next(item for item in default_stress_profiles() if item.name == "nominal")
    events = generate_stress_scenario(block, profile).events
    ordered = sorted(events, key=lambda item: item.available_timestamp_s)
    cutoff = ordered[len(ordered) // 2].available_timestamp_s
    config = replace(InitializationRecoveryConfig(), confirmation_station_count=3)
    full = CausalConfirmedRetardedTimeEKF(
        block.stations, events, estimator_variant="direct_bearing", recovery_config=config)
    prefix = CausalConfirmedRetardedTimeEKF(
        block.stations, tuple(item for item in events if item.available_timestamp_s <= cutoff),
        estimator_variant="direct_bearing", recovery_config=config)
    left = full.advance_to(cutoff)
    right = prefix.advance_to(cutoff)
    assert left.status == right.status
    assert left.confirmed == right.confirmed
    if left.state is not None:
        np.testing.assert_allclose(left.state.vector, right.state.vector, rtol=0, atol=1e-12)
    roles = {}
    for use in left.event_uses:
        roles.setdefault(use.event_id, set()).add(use.role)
    assert all(not {"initialization", "update"} <= role_set for role_set in roles.values())
    assert all(bearing_event_id(item) in {bearing_event_id(event) for event in events}
               for item in events)


def test_historical_correction_is_versioned_and_consistent():
    import csv
    import json
    from pathlib import Path

    from validation.gazebo_experiment import sha256

    root = Path(__file__).resolve().parents[1]
    old = root / "results" / "localization_error_attribution"
    correction = root / "results" / "robust_track_confirmation" / "historical_corrections_v2"
    manifest = json.loads((correction / "correction_manifest.json").read_text())
    assert sha256(old / "variant_summary.csv") == manifest["source_artifact_sha256"]["variant_summary.csv"]
    with (correction / "variant_summary_v2.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == manifest["source_variant_row_count"] == 48
    denied = [row for row in rows if row["first_budget_exhaustion_time_s"]]
    assert len(denied) == manifest["budget_exhaustion_row_count"] == 4
    assert all(row["budget_time_source"] == "batch_fit_and_lifecycle_agree" for row in denied)
    with (old / "initialization_summary.csv").open(newline="") as file:
        initialization = list(csv.DictReader(file))
    by_key = {(row["index"], row["estimator_variant"], row["diagnostic_variant"]): row
              for row in initialization}
    for row in rows:
        key = (row["index"], row["estimator_variant"], row["diagnostic_variant"])
        assert row["first_budget_exhaustion_time_s"] == by_key[key]["budget_exhaustion_time_s"]
    with (correction / "residual_availability_v2.csv").open(newline="") as file:
        updates = list(csv.DictReader(file))
    assert len(updates) == manifest["historical_update_row_count"]
    assert all(not row["residual_tangent_norm_rad"] and
               row["residual_unavailable_reason"] == "not_recorded_in_v1" for row in updates)


def test_explicit_baseline_flag_preserves_default_publications():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    profile = next(item for item in default_stress_profiles() if item.name == "nominal")
    events = generate_stress_scenario(block, profile).events
    default = CausalConfirmedRetardedTimeEKF(
        block.stations, events, estimator_variant="direct_bearing")
    explicit = CausalConfirmedRetardedTimeEKF(
        block.stations, events, estimator_variant="direct_bearing",
        recovery_config=replace(InitializationRecoveryConfig(), confirmation_station_count=2))
    for epoch in sorted({item.available_timestamp_s for item in events}):
        left, right = default.advance_to(epoch), explicit.advance_to(epoch)
        assert (left.status, left.confirmed, left.valid, left.failure_reason) == (
            right.status, right.confirmed, right.valid, right.failure_reason)
        assert left.batch_optimization_count == right.batch_optimization_count
        if left.state is not None:
            np.testing.assert_allclose(left.state.vector, right.state.vector, rtol=0, atol=1e-12)
            np.testing.assert_allclose(left.covariance_state, right.covariance_state,
                                       rtol=0, atol=1e-12)


def test_contradictory_bearing_has_measured_residual_and_rejection_reason():
    station = StationPose("S0", [0, 0, 0], np.eye(3), tetrahedral_array())
    state = ConstantVelocityState([70, 55, 40], [7, -3, 1.5], 0.0)
    history = AugmentedMotionHistory(state, np.eye(6) * 0.5,
                                     ManoeuvreHistoryConfig(np.eye(3) * 0.25))
    history.propagate_to(1.5)
    direction = predict_retarded_bearing(state, station, 1.0).direction_local
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ np.deg2rad([20.0, 0.0])
    angle = np.linalg.norm(tangent)
    measured = np.cos(angle) * direction + np.sin(angle) * tangent / angle
    measurement = BearingMeasurement(
        "S0", "contradiction-check", 0, 1.0, 1.2, measured,
        np.diag(np.deg2rad([0.3, 0.5])**2), np.zeros(2),
        "direct_bearing", tangent_frame="measurement")
    result = history.update_bearing(station, measurement, maximum_pre_update_nis=0.1)
    assert not result.update_applied
    assert result.failure_reason == "pre_update_nis_gate"
    assert result.residual_tangent_rad is not None
    assert result.innovation_covariance_tangent_rad2 is not None
    assert np.isfinite(result.pre_update_nis) and result.pre_update_nis > 0.1


def test_truth_changes_only_evaluation_and_position_nees_is_independent():
    import json
    from analysis.robust_track_confirmation import _run_tracker

    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    profile = next(item for item in default_stress_profiles() if item.name == "nominal")
    events = generate_stress_scenario(block, profile).events

    class Truth:
        def __init__(self, east_shift_m):
            self.east_shift_m = east_shift_m

        def q(self, time_s):
            return block.truth_state.position_at(time_s) + np.array([self.east_shift_m, 0, 0])

        def v(self, time_s):
            return block.truth_state.vector[3:]

    arguments = (block.stations, events, "direct_bearing", "baseline",
                 ManoeuvreHistoryConfig(np.eye(3) * 0.25), InitializationRecoveryConfig(), 0.0)
    left, left_updates, _, left_summary = _run_tracker(
        arguments[0], Truth(0), *arguments[1:])
    right, right_updates, _, right_summary = _run_tracker(
        arguments[0], Truth(100), *arguments[1:])
    assert left_summary["ever_confirmed"] and right_summary["ever_confirmed"]
    assert left_summary["first_confirmation_position_error_m"] != right_summary["first_confirmation_position_error_m"]
    for first, second in zip(left_updates, right_updates, strict=True):
        for name in ("event_id", "update_applied", "failure_reason", "residual_tangent_rad_json",
                     "residual_unavailable_reason", "innovation_covariance_tangent_rad2_json"):
            assert first[name] == second[name]
        if np.isfinite(first["pre_update_nis"]):
            np.testing.assert_allclose(first["pre_update_nis"], second["pre_update_nis"], rtol=0, atol=1e-12)
        else:
            assert np.isnan(second["pre_update_nis"])
    for first, second in zip(left, right, strict=True):
        for name in ("status", "confirmed", "valid", "position_enu_m_json",
                     "velocity_enu_mps_json", "position_covariance_m2_json",
                     "velocity_covariance_m2ps2_json"):
            assert first[name] == second[name]
    sample = next(row for row in left if row["confirmed"] and row["valid"])
    epoch = sample["processing_time_s"]
    position = np.asarray(json.loads(sample["position_enu_m_json"]))
    covariance = np.asarray(json.loads(sample["position_covariance_m2_json"]))
    delta = position - Truth(0).q(epoch)
    expected_nees = delta @ np.linalg.solve(covariance, delta)
    np.testing.assert_allclose(sample["position_nees"], expected_nees, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        json.loads(sample["position_nominal_95_axes_m_json"]),
        np.sqrt(CHI3 * np.linalg.eigvalsh(covariance)), rtol=0, atol=1e-10)
    assert sample["position_nominal_95_covered"] == (expected_nees <= CHI3)
