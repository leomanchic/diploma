"""Independent offline uncertainty and nominal-coordinate-quality checks."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.tracking_uncertainty_quality import (
    CHI3, _audit_epoch, _covariance, _quality_summary, analyze,
    count_batch_fits, nominal_coordinate_precision, summarize_pair, verify,
)


def _raw_epoch(position, covariance):
    p = np.asarray(covariance, float)
    q = np.asarray(position, float)
    pv = np.diag([0.25, 1.0, 4.0])
    eig, directions = np.linalg.eigh(p)
    veig, vdirs = np.linalg.eigh(pv)
    nees = q @ np.linalg.solve(p, q)
    return {
        "processing_time_s": "0.0", "valid": "True", "confirmed": "True",
        "status": "confirmed", "generation": "1", "reset_count": "0",
        "last_accepted_update_time_s": "", "time_since_last_accepted_update_s": "",
        "position_enu_m_json": json.dumps(q.tolist()),
        "velocity_enu_mps_json": json.dumps([0.5, 1.0, 2.0]),
        "position_covariance_m2_json": json.dumps(p.tolist()),
        "velocity_covariance_m2ps2_json": json.dumps(pv.tolist()),
        "position_nominal_95_axes_m_json": json.dumps(np.sqrt(CHI3 * eig).tolist()),
        "position_nominal_95_axis_directions_json": json.dumps(directions.tolist()),
        "velocity_nominal_95_axes_mps_json": json.dumps(np.sqrt(CHI3 * veig).tolist()),
        "velocity_nominal_95_axis_directions_json": json.dumps(vdirs.tolist()),
        "position_error_m": str(np.linalg.norm(q)), "velocity_error_mps": str(np.linalg.norm([0.5, 1.0, 2.0])),
        "position_nees": str(nees), "position_nominal_95_covered": str(nees <= CHI3),
        "position_sigma_rss_m": str(np.sqrt(np.trace(p))),
        "velocity_sigma_rss_mps": str(np.sqrt(np.trace(pv))),
    }


class _ZeroTruth:
    def q(self, time_s):
        return np.zeros(3)

    def v(self, time_s):
        return np.zeros(3)


def test_analytic_nees_axes_directions_and_coverage():
    p = np.diag([1.0, 4.0, 9.0])
    row = _audit_epoch(_raw_epoch([1.0, 2.0, 3.0], p), _ZeroTruth(), {1: 0.0},
                       None, 0, 0, None)
    np.testing.assert_allclose(row["position_nees"], 3.0, atol=1e-12)
    np.testing.assert_allclose(row["position_maximum_axis_m"], 3.0 * np.sqrt(CHI3))
    assert row["position_covered"]
    np.testing.assert_allclose(row["velocity_nees"], 3.0, atol=1e-12)
    assert row["velocity_covered"]
    assert row["phase"] == "first_confirmation"


def test_enu_rotation_invariance_of_error_nees_axes_and_quality():
    p = np.array([[4.0, 0.3, 0.1], [0.3, 2.0, 0.2], [0.1, 0.2, 1.0]])
    e = np.array([3.0, -2.0, 1.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    first = _audit_epoch(_raw_epoch(e, p), _ZeroTruth(), {1: 0.0}, None, 0, 0, None)
    second = _audit_epoch(_raw_epoch(rotation @ e, rotation @ p @ rotation.T),
                          _ZeroTruth(), {1: 0.0}, None, 0, 0, None)
    for name in ("position_error_m", "position_nees", "position_maximum_axis_m"):
        np.testing.assert_allclose(first[name], second[name], rtol=0, atol=1e-12)
    assert first["precision_5m"] == second["precision_5m"]


def test_maximum_axis_is_not_ellipsoid_membership():
    p = np.diag([0.01, 0.01, 25.0])
    error = np.array([1.0, 0.0, 0.0])
    assert np.linalg.norm(error) < np.sqrt(CHI3 * 25.0)
    assert error @ np.linalg.solve(p, error) > CHI3


@pytest.mark.parametrize("bad", [
    np.diag([1.0, 0.0, 2.0]),
    np.diag([1.0, -1.0, 2.0]),
    np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    np.diag([1.0, np.nan, 2.0]),
])
def test_invalid_covariance_is_rejected(bad):
    with pytest.raises(ValueError):
        _covariance(bad, "Pqq")
    with pytest.raises(ValueError):
        nominal_coordinate_precision([0, 0, 0], "confirmed", bad, 5.0)


def test_precision_boundary_and_truth_free_signature():
    p = np.diag([1.0, 2.0, 3.0])
    expected = np.sqrt(CHI3 * 3.0)
    at = nominal_coordinate_precision([0, 0, 0], "confirmed", p, expected)
    below = nominal_coordinate_precision([0, 0, 0], "confirmed", p,
                                         np.nextafter(expected, -np.inf))
    assert at.status == "nominal_precision_within_target"
    assert below.status == "nominal_precision_insufficient"
    assert at.maximum_nominal_95_axis_m == expected
    assert nominal_coordinate_precision(None, "recovering", None, 5.0).status == "unavailable"
    assert "truth" not in str(inspect.signature(nominal_coordinate_precision))
    with pytest.raises(ValueError, match="tolerance_m"):
        nominal_coordinate_precision([0, 0, 0], "confirmed", p, 0.0)


def test_fit_counters_cross_reset_and_exclude_denied_attempt():
    def fit(generation, count, reason="batch_passed"):
        return {"generation": str(generation), "fit_count_after_attempt": str(count), "reason": reason}
    rows = [fit(1, 1), fit(1, 2), fit(1, 3), fit(2, 1), fit(2, 2), fit(2, 2, "computational_budget_exceeded_before_fit")]
    result = count_batch_fits(rows, 2)
    assert result["total_executed_optimizations"] == 5
    assert result["final_generation_optimizations"] == 2
    assert result["maximum_single_generation_optimizations"] == 3
    assert result["attempts_rejected_before_optimization"] == 1
    assert json.loads(result["per_generation_optimizations_json"]) == {"1": 3, "2": 2}
    with pytest.raises(ValueError, match="final generation"):
        count_batch_fits(rows, 3)


def test_zero_common_epochs_and_no_nominal_success_keep_denominators():
    def row(t, valid):
        return {"processing_time_s": t, "valid": valid, "position_covered": True,
                "position_error_m": 1.0, "velocity_error_mps": 2.0,
                "position_nees": 1.0, "position_maximum_axis_m": 3.0,
                "velocity_sigma_rss_mps": 1.0, "time_since_last_accepted_update_s": None,
                "accepted_update_count": 0, "rejected_update_count": 0,
                "precision_5m": "nominal_precision_insufficient" if valid else "unavailable"}
    baseline = [row(0.0, True), row(1.0, False)]
    improved = [row(0.0, False), row(1.0, True)]
    pair = summarize_pair(baseline, improved)
    assert pair["publication_count"] == 2
    assert pair["baseline_valid_count"] == pair["new_valid_count"] == 1
    assert pair["shared_valid_count"] == 0
    assert pair["baseline_shared_coverage"] is None
    quality = _quality_summary(baseline, 5.0)
    assert quality["all_publications"] == 2 and quality["nominal_precision_within_target_count"] == 0
    assert quality["conditional_position_error_median_m"] is None
    assert quality["false_precision_fraction_conditional"] is None


def test_all_published_tracks_audit_without_audio_or_tracker_replay(tmp_path, monkeypatch):
    import analysis.robust_track_confirmation as frozen

    def forbidden(*args, **kwargs):
        raise AssertionError("audio synthesis or tracker replay is forbidden")

    monkeypatch.setattr(frozen, "synthesize_multistation_audio", forbidden)
    monkeypatch.setattr(frozen, "CausalManoeuvreRetardedTimeEKF", forbidden)
    destination = tmp_path / "independent-audit"
    result = analyze(destination)
    assert result["source_tracker_result_count"] == 48
    assert result["all_publication_count"] == 7488
    assert result["valid_publication_count"] == 5106
    assert result["tracker_replay_count"] == result["audio_synthesis_count"] == 0
    assert result["per_method_executed_fit_totals"] == {
        "all_6_equal_gcc_wls:baseline": 633,
        "all_6_equal_gcc_wls:three_station_confirmation": 690,
        "equal_weight_srp_phat:baseline": 379,
        "equal_weight_srp_phat:three_station_confirmation": 467,
    }
    assert verify(destination)["status"] == "verified"
    with (destination / "cost_by_method.csv").open("a") as file:
        file.write("\n")
    with pytest.raises(ValueError, match="derived SHA"):
        verify(destination)


def test_published_summary_table_tamper_is_detected(monkeypatch):
    import analysis.tracking_uncertainty_quality as audit

    original = audit.sha256
    def wrong_summary_digest(path):
        return "0" * 64 if Path(path).name == "run_summary.csv" else original(path)
    monkeypatch.setattr(audit, "sha256", wrong_summary_digest)
    with pytest.raises(ValueError, match="published summary table SHA changed"):
        audit._source_snapshot()
