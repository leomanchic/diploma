"""Integration gates for the frozen S8 recorded-source demonstration."""

import json

import pytest

from simulation.multistation_audio import multistation_audio_seeds
from validation.recorded_source_pilot import (
    CONFIGURATIONS,
    RECORDING_ID,
    RESULT_SCOPE,
    generate_recorded_source_sequence,
    recorded_sequence_seed,
    smoke_test,
)
from validation.three_station_audio_tracking_study import calibrate_audio_bearings
from validation.three_station_audio_tracking_study import (
    ESTIMATOR_VARIANTS,
    summarize_pilot,
)


def test_recorded_pilot_protocol_is_small_and_predeclared():
    assert [(item.trajectory_kind, item.snr_db) for item in CONFIGURATIONS] == [
        ("constant_velocity", -6.0),
        ("constant_velocity", 10.0),
    ]
    assert RESULT_SCOPE == "single_session_integration_demonstration"


def test_recorded_pilot_seed_scopes_are_disjoint_and_deterministic():
    calibration = {recorded_sequence_seed("calibration", index) for index in range(2)}
    evaluation = {recorded_sequence_seed("evaluation", index) for index in range(2)}
    assert len(calibration) == len(evaluation) == 2
    assert calibration.isdisjoint(evaluation)
    assert recorded_sequence_seed("evaluation", 1) == recorded_sequence_seed(
        "evaluation", 1
    )
    calibration_noise = {
        seed for base in calibration for seed in multistation_audio_seeds(base, 3)[1]
    }
    evaluation_noise = {
        seed for base in evaluation for seed in multistation_audio_seeds(base, 3)[1]
    }
    assert calibration_noise.isdisjoint(evaluation_noise)


def test_recorded_pilot_gcc_and_srp_receive_identical_frame_keys():
    _, _, _, rows, _ = generate_recorded_source_sequence(
        CONFIGURATIONS[1], 1, "evaluation", duration_s=0.04
    )
    methods = sorted({row["estimator_variant"] for row in rows})
    key_sets = {
        method: {
            (row["sequence_id"], row["station_id"], row["frame_index"])
            for row in rows if row["estimator_variant"] == method
        }
        for method in methods
    }
    assert len(methods) == 2
    assert key_sets[methods[0]] == key_sets[methods[1]]
    assert all(row["source_recording_id"] == RECORDING_ID for row in rows)
    assert all(not row["source_data_independent_between_splits"] for row in rows)
    assert all(not row["absolute_spl_or_detection_range_validated"] for row in rows)


def test_recorded_evaluation_rows_cannot_enter_calibration():
    _, _, _, rows, _ = generate_recorded_source_sequence(
        CONFIGURATIONS[1], 1, "evaluation", duration_s=0.04
    )
    with pytest.raises(ValueError, match="calibration must not consume evaluation"):
        calibrate_audio_bearings(rows, 1)


def test_recorded_source_smoke_is_truthfully_single_session_and_paired():
    result = smoke_test()
    assert result["recording_id"] == RECORDING_ID
    assert result["same_source_session"]
    assert not result["source_data_independent_between_splits"]
    assert result["calibration_valid_bearings"] == result["evaluation_valid_bearings"]
    assert result["calibration_valid_bearings"] > 0
    assert result["calibration_count"] == 6
    assert result["paired_frame_keys"]
    json.dumps(result)


def test_summary_accepts_the_explicit_recorded_configuration_subset():
    config = CONFIGURATIONS[0]
    bearings = []
    tracks = []
    sequences = []
    for method in ESTIMATOR_VARIANTS:
        common = {
            "trajectory_kind": config.trajectory_kind,
            "snr_db": config.snr_db,
            "estimator_variant": method,
        }
        bearings.append(dict(common, valid=True, geodesic_error_deg=1.0))
        tracks.append(dict(
            common, valid=True, confirmed=True, position_error_m=2.0,
            velocity_error_mps=0.5, valid_and_covered=True,
        ))
        sequences.append(dict(
            common, final_valid=True, first_confirmation_time_s=1.0,
            confirmed_before_manoeuvre=True, accepted_update_count=1,
            reset_count=0, failure_reason="", tracker_runtime_s=0.1,
            audio_synthesis_wall_runtime_s=0.2,
            bearing_frontend_wall_runtime_s=0.3,
            audio_pipeline_wall_runtime_s=0.5,
            maximum_history_memory_bytes=100,
            maximum_history_nodes=3,
        ))
    summary = summarize_pilot(bearings, tracks, sequences, (config,))
    assert len(summary) == 2
    assert all(row["dependent_bearing_count"] == 1 for row in summary)
