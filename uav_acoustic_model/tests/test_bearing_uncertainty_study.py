import json

import numpy as np

from validation.bearing_uncertainty_study import (
    BearingUncertaintyConfig,
    _covariance_fields,
    default_bearing_uncertainty_configurations,
    run_bearing_uncertainty_configuration,
    split_sequence_seeds,
)


def test_default_grid_contains_primary_and_secondary_geometries_and_all_physics():
    configs = default_bearing_uncertainty_configurations()
    assert len(configs) == 36
    assert {item.geometry for item in configs} == {"tetrahedral", "square"}
    assert {item.snr_db for item in configs} == {-6.0, 5.0, 20.0}
    assert {item.signal_model for item in configs} == {
        "random_broadband",
        "deterministic_multisine",
    }
    assert {item.trajectory_kind for item in configs} == {
        "stationary",
        "transverse",
        "piecewise",
    }


def test_calibration_and_evaluation_sequence_seeds_are_disjoint():
    calibration = set(split_sequence_seeds("calibration", 4, 20))
    evaluation = set(split_sequence_seeds("evaluation", 4, 20))
    assert len(calibration) == len(evaluation) == 20
    assert not calibration & evaluation


def test_small_sequence_split_uses_calibration_only_for_psd_covariance_and_nis():
    config = BearingUncertaintyConfig(
        "tetrahedral", 20.0, "random_broadband", "stationary"
    )
    covariance, quality, split_rows = run_bearing_uncertainty_configuration(
        config,
        77,
        calibration_sequence_count=2,
        evaluation_sequence_count=2,
        duration_s=0.05,
    )
    assert len(covariance) == 6
    for row in covariance:
        assert row["covariance_source_split"] == "calibration"
        assert not row["evaluation_residual_used_to_fit_covariance"]
        assert row["covariance_symmetric"]
        assert row["covariance_positive_semidefinite"]
        assert row["nis_sample_count"] > 0
        assert not row["dependent_frames_are_independent_trials"]
        assert not row["truth_used_by_online_estimator"]
        assert not row["signal_level_crlb_claimed"]
    evaluation = [row for row in covariance if row["split"] == "evaluation"]
    assert all(not row["covariance_fit_used_this_split"] for row in evaluation)
    calibration_seeds = set(json.loads(evaluation[0]["calibration_sequence_seeds_json"]))
    evaluation_seeds = set(json.loads(evaluation[0]["evaluation_sequence_seeds_json"]))
    assert not calibration_seeds & evaluation_seeds
    assert quality
    assert all(not row["quality_score_probability_claimed"] for row in quality)
    for rows in split_rows.values():
        hashes_by_frame = {}
        for row in rows:
            hashes_by_frame.setdefault((row["sequence_name"], row["frame_index"]), set()).add(
                row["frame_content_sha256"]
            )
            assert not row["truth_used_by_estimator"]
            assert not row["future_doa_estimates_used"]
        assert all(len(hashes) == 1 for hashes in hashes_by_frame.values())


def test_deterministic_multisine_sequences_have_distinct_source_realizations():
    config = BearingUncertaintyConfig(
        "tetrahedral", 20.0, "deterministic_multisine", "stationary"
    )
    covariance, _, _ = run_bearing_uncertainty_configuration(
        config,
        78,
        calibration_sequence_count=2,
        evaluation_sequence_count=2,
        duration_s=0.04,
    )
    row = covariance[0]
    calibration_sources = json.loads(row["source_seeds_json"])
    evaluation_sources = json.loads(
        next(item for item in covariance if item["split"] == "evaluation")["source_seeds_json"]
    )
    assert len(set(calibration_sources + evaluation_sources)) == 4


def test_observable_quality_reporting_has_no_truth_derived_online_probability():
    config = BearingUncertaintyConfig(
        "tetrahedral", 5.0, "random_broadband", "transverse"
    )
    _, quality, _ = run_bearing_uncertainty_configuration(
        config,
        79,
        calibration_sequence_count=2,
        evaluation_sequence_count=2,
        duration_s=0.04,
    )
    names = {row["quality_metric"] for row in quality}
    assert "gcc_mean_peak_ratio" in names
    assert "gcc_mean_peak_curvature" in names
    assert "gcc_total_spectral_energy" in names
    assert "gcc_boundary_count" in names
    assert "gcc_valid_pair_count" in names
    assert "srp_peak_score" in names
    assert "srp_score_margin" in names
    assert "srp_local_curvature_eigenvalue_min" in names
    assert all(not row["quality_metric_uses_truth_online"] for row in quality)
    assert all(row["offline_error_correlation_uses_truth"] for row in quality)


def test_no_valid_calibration_measurements_produces_no_fictitious_covariance():
    fields = _covariance_fields(None)
    assert not fields["covariance_available"]
    assert fields["covariance_rad2_json"] is None
    assert fields["covariance_eigenvalue_min_rad2"] is None
