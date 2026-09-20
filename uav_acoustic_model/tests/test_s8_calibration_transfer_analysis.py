"""Saved-data regressions for the S8 post-hoc calibration-transfer audit."""

from __future__ import annotations

import numpy as np
import pytest

from model.dynamic_state import ConstantVelocityState
from model.retarded_bearing import retarded_bearing_residual
from validation.s8_calibration_transfer_analysis import (
    ROOT, PREFIX, _calibration_lookup, _events, _read,
    assert_calibration_session_grain_available, contract_audit,
    replay_saved_stream,
)
from validation.three_station_audio_tracking_study import (
    pilot_stations, trajectory_for_audio_pilot,
)


def _saved():
    root = ROOT / "results"
    return (
        _read(root / f"{PREFIX}calibration.csv"),
        _read(root / f"{PREFIX}bearing_results.csv"),
        _read(root / f"{PREFIX}tracking_results.csv"),
    )


def test_serialized_spherical_residual_matches_tracker_sign_units_and_frame():
    _, bearings, _ = _saved()
    audit = contract_audit(bearings)
    assert audit["sampled_group_count"] == 48
    assert audit["maximum_serialized_residual_disagreement_rad"] <= 1e-14
    assert audit["maximum_truth_direction_disagreement"] <= 1e-10
    assert audit["maximum_tracker_residual_at_truth_disagreement_rad"] <= 1e-10


def test_pooled_calibration_cannot_be_mistaken_for_session_grain():
    calibration, _, _ = _saved()
    with pytest.raises(ValueError, match="not identifiable"):
        assert_calibration_session_grain_available(calibration)


def test_zero_bias_replay_changes_only_bias_not_R_or_observed_bearing():
    calibration, bearings, _ = _saved()
    lookup = _calibration_lookup(calibration)
    rows = [
        row for row in bearings
        if row["source_model_comparison"] == "recorded_source_approximation"
        and row["paired_session_id"] == "n-audioman-quadcopter-takeoff-hover-distance"
        and row["snr_db"] == "10.0"
        and row["estimator_variant"] == "all_6_equal_gcc_wls"
    ]
    pooled = _events(rows, lookup, zero_bias=False)
    zero = _events(rows, lookup, zero_bias=True)
    assert len(pooled) == len(zero) == 21
    for original, alternative in zip(pooled, zero, strict=True):
        np.testing.assert_allclose(
            original.direction_local, alternative.direction_local, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            original.covariance_tangent_rad2,
            alternative.covariance_tangent_rad2, rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            alternative.calibration_bias_tangent_rad, np.zeros(2), rtol=0, atol=0
        )
        assert original.reception_center_timestamp_s == alternative.reception_center_timestamp_s
        assert original.available_timestamp_s == alternative.available_timestamp_s


def test_calibration_bias_is_subtracted_in_prediction_tangent_arc_radians():
    calibration, bearings, _ = _saved()
    lookup = _calibration_lookup(calibration)
    row = next(row for row in bearings if row["valid"] == "True")
    event = _events([row], lookup, zero_bias=False)[0]
    station = next(s for s in pilot_stations() if s.station_id == row["station_id"])
    physical_session_index = int(row["configuration_index"]) // 2
    trajectory = trajectory_for_audio_pilot("constant_velocity", physical_session_index)
    state = ConstantVelocityState(trajectory.q(0.0), trajectory.v(0.0), 0.0)
    actual = retarded_bearing_residual(state, station, event)
    saved = np.asarray([float(row[f"residual_rad_{axis}"]) for axis in range(2)])
    np.testing.assert_allclose(
        actual, saved - event.calibration_bias_tangent_rad,
        rtol=0, atol=1e-13,
    )


def test_frozen_broadband_failure_retains_pre_budget_fit_reasons():
    calibration, bearings, tracking = _saved()
    lookup = _calibration_lookup(calibration)
    key = (
        "random_broadband", "n-audioman-quadcopter-takeoff-hover-distance",
        "-6.0", "all_6_equal_gcc_wls",
    )
    rows = [row for row in bearings if tuple(row[field] for field in (
        "source_model_comparison", "paired_session_id", "snr_db", "estimator_variant",
    )) == key]
    truths = [row for row in tracking if tuple(row[field] for field in (
        "source_model_comparison", "paired_session_id", "snr_db", "estimator_variant",
    )) == key]
    summary, diagnostics = replay_saved_stream(rows, lookup, truths, zero_bias=False)
    assert summary["final_confirmed"] is False
    assert summary["failure_reason"] == "computational_budget_exceeded"
    assert summary["batch_optimization_count"] == 4
    fits = [row for row in diagnostics if row["diagnostic_kind"] == "batch_fit"]
    assert len(fits) == 5  # four completed fits and one budget-denied attempt
    assert fits[0]["fit_phase"] == "construction"
    assert all(row["fit_phase"] == "confirmation_refit" for row in fits[1:])
    assert fits[-1]["reason"] == "computational_budget_exceeded_before_fit"
    assert all(row["construction_event_ids_json"] for row in fits)
