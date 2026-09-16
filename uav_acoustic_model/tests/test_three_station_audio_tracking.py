"""Integration gates for continuous three-station audio bearing tracking."""

from dataclasses import fields

import numpy as np
import pytest

from model.bearing_events import CausalBearingEventStream
from model.geometry import tetrahedral_array
from model.measurements import BearingMeasurement
from model.station import StationPose
from simulation.continuous_stream import extract_overlapping_frames
from simulation.multistation_audio import multistation_audio_seeds, synthesize_multistation_audio
from validation.three_station_audio_tracking_study import (
    AudioPilotConfig,
    ESTIMATOR_VARIANTS,
    FRAME_LENGTH,
    HOP_LENGTH,
    MODELED_PROCESSING_DELAY_S,
    STATION_DELIVERY_DELAY_S,
    audio_sequence_seed,
    bearing_measurements_from_records,
    calibrate_audio_bearings,
    extract_audio_bearing_records,
    pilot_stations,
    trajectory_for_audio_pilot,
)


def _short_stream(seed=1234, duration_s=0.05):
    stations = pilot_stations()
    trajectory = trajectory_for_audio_pilot("constant_velocity", 0)
    stream = synthesize_multistation_audio(
        stations,
        trajectory,
        duration_s=duration_s,
        reception_start_time_s=0.5,
        snr_db=10.0,
        seed=seed,
    )
    return stations, trajectory, stream


def test_multistation_audio_uses_one_source_and_is_reproducible():
    stations, trajectory, first = _short_stream()
    _, _, second = _short_stream()
    np.testing.assert_array_equal(first.source_signal, second.source_signal)
    np.testing.assert_array_equal(first.reception_times_s, second.reception_times_s)
    assert first.source_seed == second.source_seed
    assert len({item.noise_seed for item in first.stations}) == len(stations)
    assert first.source_seed not in {item.noise_seed for item in first.stations}
    for left, right in zip(first.stations, second.stations, strict=True):
        np.testing.assert_array_equal(left.clean_channels, right.clean_channels)
        np.testing.assert_array_equal(left.noise, right.noise)
        np.testing.assert_array_equal(left.channels, right.channels)
        assert left.propagation.source_time_support_s[0] == first.source_start_time_s
        assert left.propagation.source_time_support_s == right.propagation.source_time_support_s
    assert first.noise_generated_once_per_station_stream
    assert not first.frames_resynthesized_independently


def test_overlapping_station_frames_share_exact_samples():
    _, _, stream = _short_stream(duration_s=0.06)
    frames = extract_overlapping_frames(
        stream.stations[0].channels,
        stream.reception_times_s,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    overlap = FRAME_LENGTH - HOP_LENGTH
    np.testing.assert_array_equal(
        frames.frames[0, :, HOP_LENGTH:], frames.frames[1, :, :overlap]
    )


def test_audio_bearing_coordinates_and_availability_are_explicit():
    stations, trajectory, stream = _short_stream(duration_s=0.04)
    records, _ = extract_audio_bearing_records(
        stream,
        stations,
        trajectory,
        split="calibration",
        configuration_index=0,
        sequence_index=0,
    )
    assert records
    station_map = {station.station_id: station for station in stations}
    for row in records:
        station = station_map[row["station_id"]]
        np.testing.assert_allclose(
            station.local_to_world_direction(row["truth_local"]),
            row["truth_world"],
            rtol=0.0,
            atol=2e-15,
        )
        if row["valid"]:
            np.testing.assert_allclose(
                station.local_to_world_direction(row["estimate_local"]),
                row["estimate_world"],
                rtol=0.0,
                atol=2e-15,
            )
        assert row["available_timestamp_s"] == pytest.approx(
            row["frame_end_reception_time_s"]
            + MODELED_PROCESSING_DELAY_S
            + STATION_DELIVERY_DELAY_S[row["station_id"]],
            rel=0.0,
            abs=1e-15,
        )
        assert row["available_timestamp_s"] > row["frame_center_reception_time_s"]
        assert row["measured_algorithm_runtime_s"] >= 0.0
        assert not row["truth_used_by_audio_estimator"]


def test_audio_bearing_world_transform_handles_rotated_station_pose():
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    stations = list(pilot_stations())
    stations[0] = StationPose("S0", [0.0, 0.0, 0.0], rotation, tetrahedral_array())
    trajectory = trajectory_for_audio_pilot("constant_velocity", 0)
    stream = synthesize_multistation_audio(
        tuple(stations), trajectory, duration_s=0.04,
        reception_start_time_s=0.5, snr_db=10.0, seed=9321,
    )
    records, _ = extract_audio_bearing_records(
        stream, tuple(stations), trajectory, split="calibration",
        configuration_index=0, sequence_index=0,
    )
    rotated = [row for row in records if row["station_id"] == "S0"]
    assert rotated
    assert any(not np.allclose(row["truth_local"], row["truth_world"]) for row in rotated)
    for row in rotated:
        np.testing.assert_allclose(
            stations[0].local_to_world_direction(row["truth_local"]),
            row["truth_world"], rtol=0.0, atol=2e-15,
        )


def test_calibration_is_split_isolated_and_measurement_contract_is_truth_free():
    stations, trajectory, stream = _short_stream(duration_s=0.05)
    calibration_rows, _ = extract_audio_bearing_records(
        stream,
        stations,
        trajectory,
        split="calibration",
        configuration_index=0,
        sequence_index=0,
    )
    calibrations = calibrate_audio_bearings(calibration_rows, 1)
    evaluation_rows = [dict(row, split="evaluation") for row in calibration_rows]
    with pytest.raises(ValueError, match="calibration must not consume evaluation"):
        calibrate_audio_bearings(evaluation_rows, 1)
    measurements = bearing_measurements_from_records(
        evaluation_rows, calibrations, ESTIMATOR_VARIANTS[0]
    )
    assert measurements
    forbidden = {"truth_direction", "true_position", "true_emission_time", "angular_error"}
    assert forbidden.isdisjoint({field.name for field in fields(BearingMeasurement)})
    for measurement in measurements:
        key = (
            measurement.station_id,
            measurement.estimator_variant,
            "constant_velocity",
            10.0,
        )
        if measurement.valid:
            np.testing.assert_array_equal(
                measurement.covariance_tangent_rad2, calibrations[key].covariance_rad2
            )


def test_split_seeds_are_disjoint_and_deterministic():
    calibration = {
        audio_sequence_seed("calibration", config, sequence)
        for config in range(6) for sequence in range(4)
    }
    evaluation = {
        audio_sequence_seed("evaluation", config, sequence)
        for config in range(6) for sequence in range(4)
    }
    assert len(calibration) == 24
    assert len(evaluation) == 24
    assert calibration.isdisjoint(evaluation)
    assert audio_sequence_seed("calibration", 3, 2) == audio_sequence_seed(
        "calibration", 3, 2
    )
    calibration_roles = set(multistation_audio_seeds(audio_sequence_seed("calibration", 0, 0), 3))
    evaluation_roles = set(multistation_audio_seeds(audio_sequence_seed("evaluation", 0, 0), 3))
    assert calibration_roles.isdisjoint(evaluation_roles)


def test_future_audio_events_are_not_exposed_before_availability():
    stations, trajectory, stream = _short_stream(duration_s=0.05)
    calibration_rows, _ = extract_audio_bearing_records(
        stream,
        stations,
        trajectory,
        split="calibration",
        configuration_index=0,
        sequence_index=0,
    )
    calibrations = calibrate_audio_bearings(calibration_rows, 1)
    evaluation_rows = [dict(row, split="evaluation") for row in calibration_rows]
    measurements = bearing_measurements_from_records(
        evaluation_rows, calibrations, ESTIMATOR_VARIANTS[0]
    )
    stream_events = CausalBearingEventStream(
        measurements, estimator_variant=ESTIMATOR_VARIANTS[0]
    )
    first_availability = min(item.available_timestamp_s for item in measurements)
    assert not stream_events.advance_to(first_availability - 1e-12).measurements
    prefix = stream_events.advance_to(first_availability)
    assert prefix.measurements
    assert all(item.available_timestamp_s <= first_availability for item in prefix.measurements)


def test_tracker_frame_stride_is_deterministic_and_truth_free():
    stations, trajectory, stream = _short_stream(duration_s=0.12)
    rows, _ = extract_audio_bearing_records(
        stream, stations, trajectory, split="calibration",
        configuration_index=0, sequence_index=0,
    )
    calibrations = calibrate_audio_bearings(rows, 1)
    evaluation = [dict(row, split="evaluation") for row in rows]
    measurements = bearing_measurements_from_records(
        evaluation, calibrations, ESTIMATOR_VARIANTS[0], frame_stride=4
    )
    assert measurements
    assert {measurement.frame_index % 4 for measurement in measurements} == {0}
    assert len(measurements) == sum(
        row["estimator_variant"] == ESTIMATOR_VARIANTS[0]
        and int(row["frame_index"]) % 4 == 0
        for row in evaluation
    )


def test_vectorized_smooth_turn_matches_independent_quad_reference():
    trajectory = trajectory_for_audio_pilot("smooth_turn", 1)
    times = np.linspace(-0.2, 2.2, 121)
    reference_positions = np.vstack([
        trajectory._q_scalar(float(epoch)) for epoch in times
    ])
    reference_velocities = np.vstack([
        trajectory._v_scalar(float(epoch)) for epoch in times
    ])
    np.testing.assert_allclose(trajectory.q(times), reference_positions, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(trajectory.v(times), reference_velocities, rtol=0.0, atol=2e-14)
