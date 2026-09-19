"""Acceptance gates for the S8 independent-recording paired pilot."""

import copy
from dataclasses import replace

import numpy as np
import pytest

import validation.independent_recordings_pilot as independent_pilot

from simulation.recorded_source import (
    load_recorded_source_manifest,
    recorded_source_split_audit,
)
from validation.independent_recordings_pilot import (
    DURATION_S,
    INDEPENDENT_TRACKER_FRAME_STRIDE,
    INITIALIZATION_BATCH_OPTIMIZATION_BUDGET,
    MANIFEST_PATH,
    RESULT_SCOPE,
    SNR_LEVELS_DB,
    SOURCE_MODELS,
    _write_update_csv,
    generate_paired_sequences,
    paired_sequence_seed,
    recording_ids_for_split,
    recording_support_preflight,
    run_ideal_schedule_control,
    smoke_test,
)
from validation.three_station_audio_tracking_study import (
    bearing_measurements_from_records,
    calibrate_audio_bearings,
)
from validation.s8_time_coverage import time_weighted_coverage_rows


def test_frozen_split_has_two_disjoint_sessions_and_origins_per_side():
    manifest = load_recorded_source_manifest(MANIFEST_PATH)
    audit = recorded_source_split_audit(manifest)
    assert recording_ids_for_split("calibration") == (
        "freesound-683298-sadiquecat-mavic-mini-2",
        "freesound-263022-alcappuccino-phantom-2",
    )
    assert recording_ids_for_split("evaluation") == (
        "freesound-383904-simeonradivoev-mini-quadcopter",
        "freesound-321687-n-audioman-drone-takeoff",
    )
    assert audit["calibration_session_count"] == 2
    assert audit["evaluation_session_count"] == 2
    assert not audit["overlapping_session_ids"]
    assert not audit["overlapping_origin_asset_ids"]
    assert audit["source_data_independent_between_splits"]
    assert RESULT_SCOPE == "held_out_independent_recording_tracking_feasibility"
    assert INDEPENDENT_TRACKER_FRAME_STRIDE == 64
    assert DURATION_S == 4.5
    assert INITIALIZATION_BATCH_OPTIMIZATION_BUDGET == 4


def test_exact_retarded_bearings_expose_old_schedule_defect_and_new_confirmation():
    old = run_ideal_schedule_control(2.0)
    corrected = run_ideal_schedule_control(4.5)
    assert old["frame_group_count"] == 3
    assert old["event_count"] == 9
    assert not old["confirmed"]
    assert old["failure_reason"] == "tentative_initialization_unconfirmed"
    assert old["accepted_update_count"] == 0
    assert old["accepted_post_init_frame_group_count"] == 0
    assert corrected["frame_group_count"] == 7
    assert corrected["event_count"] == 21
    assert corrected["confirmed"]
    assert corrected["first_confirmation_time_s"] > 2.0
    assert corrected["accepted_update_count"] >= 3
    assert corrected["accepted_post_init_frame_group_count"] >= 2
    assert corrected["batch_optimization_count"] <= INITIALIZATION_BATCH_OPTIMIZATION_BUDGET


def test_batch_fit_budget_has_distinct_invalid_reason_and_no_extra_fits():
    limited = run_ideal_schedule_control(4.5, batch_optimization_budget=1)
    assert not limited["confirmed"]
    assert limited["failure_reason"] == "computational_budget_exceeded"
    assert limited["batch_optimization_count"] == 1


def test_all_recorded_intervals_cover_emission_and_fir_support_without_repeat():
    rows = recording_support_preflight()
    assert len(rows) == 4
    assert {row["split"] for row in rows} == {"calibration", "evaluation"}
    assert all(row["remaining_samples"] > 0 for row in rows)
    assert all(not row["repeat_or_padding_used"] for row in rows)


def test_insufficient_clip_is_rejected_instead_of_repeated(monkeypatch):
    original = independent_pilot._clip
    def shortened(recording_id, split):
        clip = original(recording_id, split)
        return replace(clip, samples=clip.samples[:1000])
    monkeypatch.setattr(independent_pilot, "_clip", shortened)
    with pytest.raises(ValueError, match="emission/FIR support requires"):
        recording_support_preflight()


def test_same_session_transcode_cannot_pass_split_audit_even_with_true_flag():
    manifest = load_recorded_source_manifest(MANIFEST_PATH)
    duplicated = copy.deepcopy(manifest)
    first = duplicated["recordings"][0]
    second = copy.deepcopy(first)
    second["recording_id"] = "a-different-file-name"
    second["split_membership"] = ["evaluation"]
    second["selected_intervals_s"] = {"evaluation": [12.0, 18.0]}
    second["independent_source_split"] = True
    first["split_membership"] = ["calibration"]
    duplicated["recordings"] = [first, second]
    audit = recorded_source_split_audit(duplicated, minimum_sessions_per_split=1)
    assert audit["calibration_recording_ids"] != audit["evaluation_recording_ids"]
    assert audit["overlapping_session_ids"] == (first["session_id"],)
    assert audit["overlapping_origin_asset_ids"] == (first["origin_asset_id"],)
    assert not audit["source_data_independent_between_splits"]


def test_pair_uses_identical_trajectory_timestamps_frames_and_standard_noise():
    pair = generate_paired_sequences(
        recording_ids_for_split("evaluation")[0],
        "evaluation",
        0,
        1,
        duration_s=0.08,
    )
    recorded = pair["recorded_source_approximation"]
    broadband = pair["random_broadband"]
    assert set(pair) == set(SOURCE_MODELS)
    np.testing.assert_array_equal(
        recorded[2].reception_times_s, broadband[2].reception_times_s
    )
    sample_times = np.asarray([0.5, 0.54, 0.58])
    np.testing.assert_allclose(
        recorded[1].q(sample_times), broadband[1].q(sample_times), rtol=0.0, atol=0.0
    )
    for left, right in zip(recorded[2].stations, broadband[2].stations, strict=True):
        assert left.noise_seed == right.noise_seed
        left_sigma = np.sqrt(np.mean(left.clean_channels**2)) / 10.0 ** (
            float(left.nominal_snr_db) / 20.0
        )
        right_sigma = np.sqrt(np.mean(right.clean_channels**2)) / 10.0 ** (
            float(right.nominal_snr_db) / 20.0
        )
        np.testing.assert_allclose(
            left.noise / left_sigma,
            right.noise / right_sigma,
            rtol=2e-15,
            atol=2e-15,
        )
    recorded_keys = {
        (row["station_id"], row["estimator_variant"], row["frame_index"])
        for row in recorded[3]
    }
    broadband_keys = {
        (row["station_id"], row["estimator_variant"], row["frame_index"])
        for row in broadband[3]
    }
    assert recorded_keys == broadband_keys


def test_calibration_and_evaluation_seed_scopes_are_disjoint_but_pairs_match():
    calibration = {
        paired_sequence_seed("calibration", session, snr)
        for session in range(2) for snr in range(2)
    }
    evaluation = {
        paired_sequence_seed("evaluation", session, snr)
        for session in range(2) for snr in range(2)
    }
    assert len(calibration) == len(evaluation) == 4
    assert calibration.isdisjoint(evaluation)
    assert paired_sequence_seed("evaluation", 1, 1) == paired_sequence_seed(
        "evaluation", 1, 1
    )


def test_truth_scenario_labels_do_not_select_measurement_calibration():
    calibration_pair = generate_paired_sequences(
        recording_ids_for_split("calibration")[0],
        "calibration",
        0,
        1,
        duration_s=0.20,
    )
    evaluation_pair = generate_paired_sequences(
        recording_ids_for_split("evaluation")[0],
        "evaluation",
        0,
        1,
        duration_s=0.08,
    )
    calibration_rows = calibration_pair["recorded_source_approximation"][3]
    evaluation_rows = evaluation_pair["recorded_source_approximation"][3]
    calibration = calibrate_audio_bearings(calibration_rows, 1)
    original = bearing_measurements_from_records(
        evaluation_rows, calibration, "all_6_equal_gcc_wls", frame_stride=1
    )
    relabelled = copy.deepcopy(evaluation_rows)
    for row in relabelled:
        row["trajectory_kind"] = "truth-label-removed"
        row["snr_db"] = 12345.0
        row["paired_session_id"] = "truth-label-changed"
    changed = bearing_measurements_from_records(
        relabelled, calibration, "all_6_equal_gcc_wls", frame_stride=1
    )
    assert len(original) == len(changed)
    for left, right in zip(original, changed, strict=True):
        np.testing.assert_array_equal(left.direction_local, right.direction_local)
        np.testing.assert_array_equal(
            left.covariance_tangent_rad2, right.covariance_tangent_rad2
        )
        np.testing.assert_array_equal(
            left.calibration_bias_tangent_rad, right.calibration_bias_tangent_rad
        )


def test_independent_recordings_smoke_chain_is_paired_and_valid():
    result = smoke_test()
    assert result["paired_frame_keys_equal"]
    assert result["reception_timestamps_equal"]
    assert result["standardized_noise_equal"]
    assert result["recorded_valid_bearings"] > 0
    assert result["broadband_valid_bearings"] > 0
    assert SNR_LEVELS_DB == (-6.0, 10.0)


def test_zero_updates_are_reported_as_empty_table_with_schema(tmp_path):
    path = tmp_path / "updates.csv"
    _write_update_csv(path, [])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "event_id" in lines[0]
    assert "update_applied" in lines[0]


def test_time_weighted_availability_and_conditional_coverage_are_distinct():
    common = {
        "paired_session_id": "one-original-session",
        "source_model_comparison": "random_broadband",
        "snr_db": "10.0", "estimator_variant": "all_6_equal_gcc_wls",
    }
    rows = [
        {**common, "processing_time_s": 0.0, "valid": True,
         "confirmed": True, "valid_and_covered": True},
        {**common, "processing_time_s": 1.0, "valid": False,
         "confirmed": False, "valid_and_covered": False},
        {**common, "processing_time_s": 3.0, "valid": True,
         "confirmed": True, "valid_and_covered": False},
    ]
    result = time_weighted_coverage_rows(rows)[0]
    assert result["available_estimate_time_fraction"] == 1 / 3
    assert result["coverage_given_available_time"] == 1.0
    assert result["available_and_covered_time_fraction"] == 1 / 3
    assert result["dependent_publication_count"] == 3
    rows[0]["valid"] = rows[0]["confirmed"] = rows[0]["valid_and_covered"] = False
    missing = time_weighted_coverage_rows(rows)[0]
    assert missing["available_estimate_time_fraction"] == 0.0
    assert np.isnan(missing["coverage_given_available_time"])
