"""Deterministic gates for tentative initialization and causal recovery."""

from dataclasses import replace

import numpy as np
import pytest

from estimators.retarded_ekf_recovery import CausalConfirmedRetardedTimeEKF
from model.bearing_events import bearing_event_id
from model.bearing_statistics import tangent_basis
from model.geometry import direction_angles
from model.retarded_bearing import retarded_bearing_residual
from validation.retarded_ekf_stress_study import (
    default_stress_profiles,
    generate_stress_base_block,
    generate_stress_scenario,
)


def _profile(name: str):
    return next(item for item in default_stress_profiles() if item.name == name)


def _processor(geometry: str, sequence_index: int, profile_name: str, *, seed: int = 20260912):
    block = generate_stress_base_block(geometry, sequence_index, base_seed=seed)
    scenario = generate_stress_scenario(block, _profile(profile_name))
    processor = CausalConfirmedRetardedTimeEKF(
        block.stations,
        scenario.events,
        estimator_variant="direct_bearing",
    )
    return block, scenario, processor


def _run_eventwise(processor, events):
    publications = [
        processor.advance_to(timestamp)
        for timestamp in sorted({item.available_timestamp_s for item in events})
    ]
    publications.append(processor.advance_to(14.5))
    return publications


def _offset_direction(direction: np.ndarray, angle_deg: float) -> np.ndarray:
    phi, elevation = direction_angles(direction)
    tangent = tangent_basis(phi, elevation).T @ np.deg2rad([angle_deg, 0.0])
    angle = float(np.linalg.norm(tangent))
    return np.cos(angle) * direction + np.sin(angle) * tangent / angle


@pytest.mark.parametrize(
    ("geometry", "sequence_index", "maximum_error_m"),
    (("informative", 57, 1.0), ("poorly_conditioned", 37, 2.0)),
)
def test_known_d2_mild_outlier_lock_is_prevented(
    geometry: str, sequence_index: int, maximum_error_m: float
):
    block, scenario, processor = _processor(
        geometry, sequence_index, "outlier_mild"
    )
    final = _run_eventwise(processor, scenario.events)[-1]
    assert final.valid and final.confirmed and final.status == "confirmed"
    error = np.linalg.norm(
        final.state.position_at(14.5) - block.truth_state.position_at(14.5)
    )
    assert error < maximum_error_m
    assert any(item.action == "tentative_rejected" for item in final.hypothesis_diagnostics)
    assert any(item.action == "hypothesis_confirmed" for item in final.hypothesis_diagnostics)


def test_tentative_state_is_not_reported_as_confirmed_measurement():
    _, scenario, processor = _processor("informative", 0, "nominal", seed=20260913)
    timestamps = sorted({item.available_timestamp_s for item in scenario.events})
    publication = processor.advance_to(timestamps[5])
    assert publication.status == "tentative"
    assert publication.tentative_state is not None
    assert publication.state is None
    assert not publication.valid
    assert not publication.confirmed


def test_clean_sequence_confirms_without_reset_and_final_scores_match_refit():
    block, scenario, processor = _processor("informative", 2, "nominal", seed=20260913)
    final = _run_eventwise(processor, scenario.events)[-1]
    assert final.valid and final.reset_count == 0
    assert np.linalg.norm(
        final.state.position_at(14.5) - block.truth_state.position_at(14.5)
    ) < 2.0
    confirmed = next(
        item for item in final.hypothesis_diagnostics if item.action == "hypothesis_confirmed"
    )
    station_map = {item.station_id: item for item in block.stations}
    measurement_by_id = {bearing_event_id(item): item for item in scenario.events}
    final_batch_state = next(
        publication.state
        for publication in processor.publications
        if any(item.action == "initialized" for item in publication.new_lifecycle_diagnostics)
    )
    for identity, stored in confirmed.final_nis_values:
        measurement = measurement_by_id[identity]
        residual = retarded_bearing_residual(
            final_batch_state,
            station_map[measurement.station_id],
            measurement,
        )
        expected = residual @ np.linalg.solve(
            measurement.covariance_tangent_rad2, residual
        )
        np.testing.assert_allclose(stored, expected, rtol=0.0, atol=2e-10)
    assert confirmed.preliminary_nis_values != confirmed.final_nis_values


def test_late_outlier_does_not_change_state_when_rejected():
    block = generate_stress_base_block("informative", 5, base_seed=20260913)
    scenario = generate_stress_scenario(block, _profile("nominal"))
    ordered = sorted(scenario.events, key=lambda item: item.available_timestamp_s)
    target = ordered[20]
    corrupted = tuple(
        replace(item, direction_local=_offset_direction(item.direction_local, 20.0))
        if bearing_event_id(item) == bearing_event_id(target)
        else item
        for item in scenario.events
    )
    with_outlier = CausalConfirmedRetardedTimeEKF(
        block.stations, corrupted, estimator_variant="direct_bearing"
    )
    without_outlier = CausalConfirmedRetardedTimeEKF(
        block.stations,
        tuple(item for item in corrupted if bearing_event_id(item) != bearing_event_id(target)),
        estimator_variant="direct_bearing",
    )
    first = with_outlier.advance_to(target.available_timestamp_s)
    second = without_outlier.advance_to(target.available_timestamp_s)
    rejected = next(
        item for item in first.update_diagnostics if item.event_id == bearing_event_id(target)
    )
    assert rejected.failure_reason == "pre_update_nis_gate"
    np.testing.assert_allclose(first.state.vector, second.state.vector, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(
        first.covariance_state, second.covariance_state, rtol=0.0, atol=2e-12
    )
    assert first.reset_count == 0


def test_sustained_contradictions_trigger_reset_and_fresh_recovery():
    block = generate_stress_base_block("informative", 6, base_seed=20260913)
    scenario = generate_stress_scenario(block, _profile("nominal"))
    ordered = sorted(scenario.events, key=lambda item: item.available_timestamp_s)
    corrupt_ids = {bearing_event_id(item) for item in ordered[18:24]}
    corrupted = tuple(
        replace(item, direction_local=_offset_direction(item.direction_local, 20.0))
        if bearing_event_id(item) in corrupt_ids
        else item
        for item in scenario.events
    )
    processor = CausalConfirmedRetardedTimeEKF(
        block.stations, corrupted, estimator_variant="direct_bearing"
    )
    final = _run_eventwise(processor, corrupted)[-1]
    assert final.valid and final.confirmed
    assert final.reset_count >= 1 and final.generation >= 2
    assert any(item.action == "consistency_lost" for item in final.lifecycle_diagnostics)
    assert any(item.action == "reinitialized" for item in final.lifecycle_diagnostics)
    uses = [item.event_id for item in final.event_uses]
    assert len(uses) == len(set(uses))


def test_packet_absence_does_not_trigger_consistency_loss():
    _, scenario, processor = _processor("informative", 0, "all_station_gap", seed=20260913)
    final = _run_eventwise(processor, scenario.events)[-1]
    assert final.valid
    assert final.reset_count == 0
    assert not any(item.action == "consistency_lost" for item in final.lifecycle_diagnostics)


def test_contradictory_prefix_without_confirming_set_remains_invalid():
    block = generate_stress_base_block("informative", 7, base_seed=20260913)
    scenario = generate_stress_scenario(block, _profile("nominal"))
    ordered = sorted(scenario.events, key=lambda item: item.available_timestamp_s)[:12]
    contradictory = tuple(
        replace(
            item,
            direction_local=_offset_direction(
                item.direction_local, 20.0 if index % 2 else -20.0
            ),
        )
        for index, item in enumerate(ordered)
    )
    processor = CausalConfirmedRetardedTimeEKF(
        block.stations, contradictory, estimator_variant="direct_bearing"
    )
    final = processor.advance_to(14.5)
    assert not final.valid and not final.confirmed
    assert final.status in {"uninitialized", "tentative"}


def test_schedule_invariance_and_no_repeated_statistical_use():
    block, scenario, frequent = _processor(
        "poorly_conditioned", 8, "outlier_strong", seed=20260913
    )
    frequent_final = _run_eventwise(frequent, scenario.events)[-1]
    one_shot = CausalConfirmedRetardedTimeEKF(
        block.stations, scenario.events, estimator_variant="direct_bearing"
    ).advance_to(14.5)
    assert frequent_final.valid == one_shot.valid
    assert frequent_final.status == one_shot.status
    assert frequent_final.generation == one_shot.generation
    if frequent_final.valid:
        np.testing.assert_allclose(
            frequent_final.state.vector, one_shot.state.vector, rtol=0.0, atol=3e-10
        )
        np.testing.assert_allclose(
            frequent_final.covariance_state,
            one_shot.covariance_state,
            rtol=0.0,
            atol=3e-9,
        )
    assert frequent_final.event_uses == one_shot.event_uses
    used = [item.event_id for item in frequent_final.event_uses]
    assert len(used) == len(set(used))


def _confirmed_nominal_sequence():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    scenario = generate_stress_scenario(block, _profile("nominal"))
    processor = CausalConfirmedRetardedTimeEKF(
        block.stations, scenario.events, estimator_variant="direct_bearing"
    )
    final = processor.advance_to(13.0)
    assert final.valid and final.confirmed
    return block, scenario, final


def _changed_late_payload(measurement, *, available_timestamp_s: float = 14.0):
    return replace(
        measurement,
        available_timestamp_s=available_timestamp_s,
        direction_local=_offset_direction(measurement.direction_local, 5.0),
    )


def test_conflicting_tentative_construction_event_rejects_hypothesis():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    scenario = generate_stress_scenario(block, _profile("nominal"))
    timestamps = sorted({item.available_timestamp_s for item in scenario.events})
    probe = CausalConfirmedRetardedTimeEKF(
        block.stations, scenario.events, estimator_variant="direct_bearing"
    )
    tentative = probe.advance_to(timestamps[5])
    target_id = tentative.tentative_construction_event_ids[0]
    target = next(item for item in scenario.events if bearing_event_id(item) == target_id)
    conflict_time = timestamps[6]
    processor = CausalConfirmedRetardedTimeEKF(
        block.stations,
        (*scenario.events, _changed_late_payload(target, available_timestamp_s=conflict_time)),
        estimator_variant="direct_bearing",
    )
    before = processor.advance_to(timestamps[5])
    after = processor.advance_to(conflict_time)
    assert before.status == "tentative"
    rejected = [
        item
        for item in after.new_hypothesis_diagnostics
        if item.action == "tentative_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].reason == "conflicted_tentative_construction_event"
    assert rejected[0].excluded_event_ids == (target_id,)
    assert target_id in after.conflicted_event_ids
    assert target_id not in after.tentative_construction_event_ids


@pytest.mark.parametrize("event_role", ("initialization", "update"))
def test_conflicting_active_state_event_invalidates_generation(event_role: str):
    block, scenario, confirmed = _confirmed_nominal_sequence()
    target_ids = (
        confirmed.initialization_event_ids
        if event_role == "initialization"
        else confirmed.applied_event_ids
    )
    target_id = target_ids[0]
    target = next(item for item in scenario.events if bearing_event_id(item) == target_id)
    processor = CausalConfirmedRetardedTimeEKF(
        block.stations,
        (*scenario.events, _changed_late_payload(target)),
        estimator_variant="direct_bearing",
    )
    before = processor.advance_to(13.0)
    after = processor.advance_to(14.5)
    assert before.valid and target_id in before.active_state_event_ids
    assert not after.valid and not after.confirmed
    assert after.status == "questionable"
    assert after.failure_reason == "conflicted_active_state_event_requires_recovery"
    assert target_id in after.conflicted_event_ids
    assert target_id in after.historical_state_event_ids
    assert target_id not in after.active_state_event_ids
    invalidation = next(
        item
        for item in after.new_lifecycle_diagnostics
        if item.action == "state_invalidated"
    )
    assert invalidation.reason == "conflicted_active_state_event_requires_recovery"
    assert invalidation.event_ids == (target_id,)


def test_exact_duplicate_of_used_event_is_safe():
    block, scenario, baseline = _confirmed_nominal_sequence()
    duplicate = next(
        item
        for item in scenario.events
        if bearing_event_id(item) == baseline.initialization_event_ids[0]
    )
    processor = CausalConfirmedRetardedTimeEKF(
        block.stations,
        (*scenario.events, duplicate),
        estimator_variant="direct_bearing",
    )
    actual = processor.advance_to(13.0)
    assert actual.valid and actual.confirmed and actual.reset_count == 0
    assert not actual.conflicted_event_ids
    assert any(item.action == "duplicate_exact" for item in actual.prefix.journal)
    np.testing.assert_allclose(actual.state.vector, baseline.state.vector, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        actual.covariance_state, baseline.covariance_state, rtol=0.0, atol=0.0
    )


def test_large_availability_group_processes_all_confirmed_ekf_updates():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    scenario = generate_stress_scenario(block, _profile("nominal"))
    ordered = sorted(
        scenario.events,
        key=lambda item: (item.available_timestamp_s, bearing_event_id(item)),
    )
    late_ids = {bearing_event_id(item) for item in ordered[15:]}
    batched = tuple(
        item if index < 15 else replace(item, available_timestamp_s=14.0)
        for index, item in enumerate(ordered)
    )
    final = CausalConfirmedRetardedTimeEKF(
        block.stations, batched, estimator_variant="direct_bearing"
    ).advance_to(14.5)
    attempted_late_ids = {
        item.event_id for item in final.update_diagnostics if item.event_id in late_ids
    }
    accounted_ids = (
        set(final.initialization_event_ids)
        | set(final.applied_event_ids)
        | set(final.rejected_event_ids)
    )
    assert final.valid and final.confirmed and final.reset_count == 0
    assert attempted_late_ids == late_ids
    assert late_ids <= accounted_ids
    assert accounted_ids == set(final.prefix.accepted_event_ids)


def test_reset_inside_large_group_classifies_every_remaining_event():
    block = generate_stress_base_block("informative", 2, base_seed=20260913)
    scenario = generate_stress_scenario(block, _profile("nominal"))
    ordered = sorted(
        scenario.events,
        key=lambda item: (item.available_timestamp_s, bearing_event_id(item)),
    )
    batched = tuple(
        item
        if index < 15
        else replace(
            item,
            available_timestamp_s=14.0,
            direction_local=_offset_direction(item.direction_local, 20.0),
        )
        for index, item in enumerate(ordered)
    )
    final = CausalConfirmedRetardedTimeEKF(
        block.stations, batched, estimator_variant="direct_bearing"
    ).advance_to(14.5)
    accounted_ids = (
        set(final.initialization_event_ids)
        | set(final.applied_event_ids)
        | set(final.rejected_event_ids)
    )
    reasons = dict(final.rejection_reasons)
    assert not final.valid and final.reset_count == 1
    assert accounted_ids == set(final.prefix.accepted_event_ids)
    assert "recovery_group_excluded_after_reset" in reasons.values()


def test_conflict_in_historical_generation_does_not_reset_active_generation():
    block = generate_stress_base_block("informative", 6, base_seed=20260913)
    scenario = generate_stress_scenario(block, _profile("nominal"))
    ordered = sorted(scenario.events, key=lambda item: item.available_timestamp_s)
    corrupt_ids = {bearing_event_id(item) for item in ordered[18:24]}
    corrupted = tuple(
        replace(item, direction_local=_offset_direction(item.direction_local, 20.0))
        if bearing_event_id(item) in corrupt_ids
        else item
        for item in scenario.events
    )
    probe = CausalConfirmedRetardedTimeEKF(
        block.stations, corrupted, estimator_variant="direct_bearing"
    ).advance_to(14.5)
    assert probe.valid and probe.generation >= 2 and probe.historical_state_event_ids
    target_id = probe.historical_state_event_ids[0]
    target = next(item for item in corrupted if bearing_event_id(item) == target_id)
    processor = CausalConfirmedRetardedTimeEKF(
        block.stations,
        (*corrupted, _changed_late_payload(target, available_timestamp_s=15.0)),
        estimator_variant="direct_bearing",
    )
    before = processor.advance_to(14.5)
    after = processor.advance_to(15.5)
    assert before.valid and before.generation >= 2
    assert after.valid and after.confirmed
    assert after.generation == before.generation
    assert after.reset_count == before.reset_count
    assert target_id in after.historical_state_event_ids
    assert any(
        item.action == "historical_conflict_quarantined"
        and item.event_ids == (target_id,)
        for item in after.new_lifecycle_diagnostics
    )
