"""Deterministic causality and audit tests for asynchronous bearing events."""

from dataclasses import replace

import numpy as np

from model.bearing_events import (
    CausalBearingEventStream,
    ScheduledBearingEvent,
    bearing_event_id,
)
from model.measurements import BearingMeasurement


def _measurement(
    station: str,
    frame: int,
    reception: float,
    available: float,
    *,
    variant: str = "direct",
    direction=(1.0, 0.0, 0.0),
) -> BearingMeasurement:
    return BearingMeasurement(
        station,
        "event-sequence",
        frame,
        reception,
        available,
        direction,
        np.diag(np.deg2rad([0.2, 0.3]) ** 2),
        np.zeros(2),
        variant,
    )


def test_only_available_events_enter_prefix_and_time_cannot_move_backwards():
    early = _measurement("A", 0, 1.0, 1.1)
    late = _measurement("B", 0, 1.0, 2.0)
    stream = CausalBearingEventStream([late, early], estimator_variant="direct")
    first = stream.advance_to(1.5)
    assert first.accepted_event_ids == (bearing_event_id(early),)
    second = stream.advance_to(2.0)
    assert set(second.accepted_event_ids) == {
        bearing_event_id(early),
        bearing_event_id(late),
    }
    with np.testing.assert_raises_regex(ValueError, "backwards"):
        stream.advance_to(1.9)


def test_equal_availability_delays_duplicates_invalid_drops_and_variant_are_audited():
    accepted = _measurement("A", 0, 1.0, 3.0)
    duplicate = accepted
    wrong_variant = _measurement("B", 0, 1.2, 3.0, variant="srp")
    invalid = BearingMeasurement.invalid(
        station_id="C",
        sequence_id="event-sequence",
        frame_index=0,
        reception_center_timestamp_s=1.3,
        available_timestamp_s=3.0,
        estimator_variant="direct",
        invalid_reason="silence",
    )
    dropped = ScheduledBearingEvent(
        _measurement("B", 1, 2.0, 3.0),
        dropped=True,
        drop_reason="temporary_station_outage",
    )
    stream = CausalBearingEventStream(
        [wrong_variant, duplicate, dropped, invalid, accepted],
        estimator_variant="direct",
    )
    prefix = stream.advance_to(3.0)
    assert prefix.accepted_event_ids == (bearing_event_id(accepted),)
    actions = [entry.action for entry in prefix.journal]
    assert actions.count("accepted") == 1
    assert actions.count("duplicate_exact") == 1
    assert "excluded_invalid" in actions
    assert "excluded_dropped" in actions
    assert "excluded_wrong_estimator_variant" in actions


def test_conflicting_same_identity_quarantines_both_payloads():
    first = _measurement("A", 2, 2.0, 2.1)
    conflict = replace(
        first,
        direction_local=np.asarray([0.9998, 0.02, 0.0])
        / np.linalg.norm([0.9998, 0.02, 0.0]),
    )
    stream = CausalBearingEventStream([first, conflict], estimator_variant="direct")
    prefix = stream.advance_to(2.1)
    assert prefix.measurements == ()
    assert prefix.conflicted_event_ids == (bearing_event_id(first),)
    assert [entry.action for entry in prefix.journal] == [
        "accepted",
        "excluded_conflict",
    ]


def test_late_conflict_changes_only_future_prefix_not_published_snapshot():
    first = _measurement("A", 3, 1.0, 1.1)
    conflict = replace(first, available_timestamp_s=4.0)
    stream = CausalBearingEventStream([first, conflict], estimator_variant="direct")
    published = stream.advance_to(2.0)
    assert published.accepted_event_ids == (bearing_event_id(first),)
    later = stream.advance_to(4.0)
    assert later.accepted_event_ids == ()
    assert published.accepted_event_ids == (bearing_event_id(first),)


def test_replay_is_reproducible_under_input_permutation_for_equal_availability():
    events = [
        _measurement("C", 1, 1.4, 2.0),
        _measurement("A", 0, 1.0, 2.0),
        _measurement("B", 3, 1.2, 2.0),
    ]
    forward = CausalBearingEventStream(events, estimator_variant="direct").advance_to(2.0)
    reverse = CausalBearingEventStream(
        list(reversed(events)), estimator_variant="direct"
    ).advance_to(2.0)
    assert forward.accepted_event_ids == reverse.accepted_event_ids
    assert forward.journal == reverse.journal
