"""Regressions for the S7C-C1 event-processing review contract."""

from dataclasses import replace

import numpy as np

from estimators.retarded_ekf import CausalRetardedTimeEKF
from model.bearing_events import bearing_event_id
from model.geometry import direction_angles
from model.bearing_statistics import tangent_basis
from validation.retarded_ekf_study import (
    default_retarded_ekf_configurations,
    generate_retarded_ekf_scenario,
)


def _scenario():
    return generate_retarded_ekf_scenario(
        default_retarded_ekf_configurations()[11], 0
    )


def _times(events):
    return sorted({item.available_timestamp_s for item in events})


def _processor(scenario, events=None):
    return CausalRetardedTimeEKF(
        scenario.stations,
        scenario.events if events is None else events,
        estimator_variant="direct_bearing",
    )


def _run(processor, times):
    result = None
    for processing_time in times:
        result = processor.advance_to(processing_time)
    assert result is not None
    return result


def _assert_same_posterior(
    first, second, *, atol=2e-12, compare_rejections=True
):
    assert first.valid and second.valid
    np.testing.assert_allclose(first.state.vector, second.state.vector, rtol=0.0, atol=atol)
    np.testing.assert_allclose(
        first.covariance_state, second.covariance_state, rtol=0.0, atol=atol
    )
    assert first.initialization_event_ids == second.initialization_event_ids
    assert first.applied_event_ids == second.applied_event_ids
    if compare_rejections:
        assert first.rejected_event_ids == second.rejected_event_ids


def _perturb_direction(direction, tangent_offset):
    phi, elevation = direction_angles(direction)
    world_tangent = tangent_basis(phi, elevation).T @ np.asarray(tangent_offset)
    angle = float(np.linalg.norm(world_tangent))
    return (
        np.cos(angle) * np.asarray(direction)
        + np.sin(angle) * world_tangent / angle
    )


def test_unknown_station_before_initialization_is_audited_and_does_not_poison_prefix():
    scenario = _scenario()
    first_time = min(_times(scenario.events))
    unknown = replace(
        scenario.events[0],
        station_id="unknown_station",
        frame_index=999,
        available_timestamp_s=first_time,
    )
    final_time = max(_times(scenario.events))
    actual = _processor(scenario, [unknown, *scenario.events]).advance_to(final_time)
    control = _processor(scenario).advance_to(final_time)
    _assert_same_posterior(actual, control, compare_rejections=False)
    assert bearing_event_id(unknown) in actual.rejected_event_ids
    assert any(
        item.event_id == bearing_event_id(unknown)
        and item.reason == "unknown_station_id"
        for item in actual.event_rejections
    )


def test_unknown_station_after_initialization_has_no_partial_update_and_next_valid_event_runs():
    scenario = _scenario()
    final_time = max(_times(scenario.events))
    template = scenario.events[-1]
    unknown = replace(
        template,
        station_id="unknown_station",
        frame_index=999,
        available_timestamp_s=final_time + 1.0,
    )
    followup = replace(
        template,
        frame_index=998,
        available_timestamp_s=final_time + 2.0,
    )
    actual_processor = _processor(
        scenario, [*scenario.events, unknown, followup]
    )
    control_processor = _processor(scenario, [*scenario.events, followup])
    actual_before = actual_processor.advance_to(final_time + 1.0)
    control_before = control_processor.advance_to(final_time + 1.0)
    _assert_same_posterior(
        actual_before, control_before, compare_rejections=False
    )
    assert actual_before.valid
    assert actual_before.failure_reason is None
    assert [item.reason for item in actual_before.new_event_rejections] == [
        "unknown_station_id"
    ]
    actual_after = actual_processor.advance_to(final_time + 2.0)
    control_after = control_processor.advance_to(final_time + 2.0)
    _assert_same_posterior(
        actual_after, control_after, compare_rejections=False
    )
    assert bearing_event_id(followup) in actual_after.applied_event_ids


def test_multiple_individual_rejection_reasons_are_visible_in_one_call():
    scenario = _scenario()
    final_time = max(_times(scenario.events))
    unknown = replace(
        scenario.events[0], station_id="unknown_station", frame_index=999
    )
    singular = replace(
        scenario.events[1],
        frame_index=998,
        covariance_tangent_rad2=np.diag([1e-6, 0.0]),
    )
    result = _processor(
        scenario, [unknown, singular, *scenario.events]
    ).advance_to(final_time)
    assert result.valid
    assert {item.reason for item in result.new_event_rejections} == {
        "unknown_station_id",
        "unsupported_singular_covariance",
    }


def test_singular_first_measurement_is_rejected_but_remaining_eligible_stream_initializes():
    scenario = _scenario()
    singular = replace(
        scenario.events[0], covariance_tangent_rad2=np.diag([1e-6, 0.0])
    )
    actual_events = [singular, *scenario.events[1:]]
    control_events = list(scenario.events[1:])
    final_time = max(_times(scenario.events))
    actual = _processor(scenario, actual_events).advance_to(final_time)
    control = _processor(scenario, control_events).advance_to(final_time)
    _assert_same_posterior(actual, control, compare_rejections=False)
    assert actual.event_rejections[0].reason == "unsupported_singular_covariance"
    assert bearing_event_id(singular) not in actual.initialization_event_ids
    assert bearing_event_id(singular) not in actual.applied_event_ids


def test_multiple_singular_measurements_do_not_block_later_eligible_initialization():
    scenario = _scenario()
    singular = [
        replace(item, covariance_tangent_rad2=np.diag([1e-6, 0.0]))
        for item in scenario.events[:2]
    ]
    result = _processor(
        scenario, [*singular, *scenario.events[2:]]
    ).advance_to(max(_times(scenario.events)))
    assert result.valid and result.initialized
    assert len(result.rejected_event_ids) == 2
    assert all(
        identity not in result.initialization_event_ids
        for identity in map(bearing_event_id, singular)
    )


def test_singular_post_initialization_duplicate_is_rejected_once_without_update():
    scenario = _scenario()
    final_time = max(_times(scenario.events))
    singular = replace(
        scenario.events[-1],
        frame_index=999,
        available_timestamp_s=final_time + 1.0,
        covariance_tangent_rad2=np.diag([1e-6, 0.0]),
    )
    actual = _processor(
        scenario, [*scenario.events, singular, singular]
    ).advance_to(final_time + 1.0)
    control = _processor(scenario).advance_to(final_time + 1.0)
    _assert_same_posterior(actual, control, compare_rejections=False)
    assert sum(item.event_id == bearing_event_id(singular) for item in actual.event_rejections) == 1
    assert all(item.event_id != bearing_event_id(singular) for item in actual.update_diagnostics)
    assert sum(item.action == "duplicate_exact" for item in actual.prefix.journal) == 1


def test_external_publication_frequency_does_not_change_state_covariance_or_ids():
    scenario = _scenario()
    times = _times(scenario.events)
    frequent = _run(_processor(scenario), times)
    one_shot = _processor(scenario).advance_to(times[-1])
    sparse = _run(_processor(scenario), [times[3], times[8], times[-1]])
    midpoints = [
        times[0] - 0.001,
        *[(left + right) / 2.0 for left, right in zip(times, times[1:])],
        times[-1],
    ]
    no_event_publications = _run(_processor(scenario), midpoints)
    for candidate in (one_shot, sparse, no_event_publications):
        _assert_same_posterior(candidate, frequent, atol=3e-12)
        assert candidate.lifecycle_diagnostics == frequent.lifecycle_diagnostics


def test_equal_availability_group_is_invariant_to_input_permutation():
    scenario = _scenario()
    common_time = max(_times(scenario.events)) + 1.0
    grouped = [
        replace(item, available_timestamp_s=common_time)
        for item in scenario.events
    ]
    forward = _processor(scenario, grouped).advance_to(common_time)
    reverse = _processor(scenario, list(reversed(grouped))).advance_to(common_time)
    _assert_same_posterior(forward, reverse)
    assert forward.lifecycle_diagnostics == reverse.lifecycle_diagnostics


def test_conflict_and_later_group_recover_inside_one_outer_call_with_history():
    scenario = _scenario()
    times = _times(scenario.events)
    used = min(scenario.events, key=lambda item: item.available_timestamp_s)
    conflict_time = (times[-2] + times[-1]) / 2.0
    conflict = replace(
        used,
        available_timestamp_s=conflict_time,
        direction_local=_perturb_direction(used.direction_local, [0.01, 0.0]),
    )
    events = [*scenario.events, conflict]
    one_shot = _processor(scenario, events).advance_to(times[-1])
    frequent = _run(_processor(scenario, events), _times(events))
    _assert_same_posterior(one_shot, frequent, atol=3e-12)
    assert [item.action for item in one_shot.lifecycle_diagnostics][-2:] == [
        "state_invalidated",
        "reinitialized_after_conflict",
    ]
    assert bearing_event_id(used) not in one_shot.initialization_event_ids
