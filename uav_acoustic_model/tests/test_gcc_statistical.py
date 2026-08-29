"""Fast calibration/evaluation and aggregate-schema tests."""

import csv
import json

from validation.gcc_statistical import (
    DOA_VARIANTS,
    FINE_SNR_LEVELS_DB,
    FRAME_LENGTHS,
    GCCStatisticalConfig,
    default_statistical_configurations,
    default_spherical_configurations,
    run_gcc_statistical_configuration,
    run_gcc_statistical_validation,
    run_spherical_model_configuration,
)


def _config():
    return GCCStatisticalConfig(
        "pytest",
        "deterministic_multisine",
        "tetrahedral",
        45.0,
        30.0,
        10.0,
        1024,
    )


def test_configuration_keeps_all_pair_components_and_independent_splits():
    pair_records, doa_records, covariance = run_gcc_statistical_configuration(
        _config(),
        91,
        calibration_trial_count=24,
        evaluation_trial_count=32,
        interpolation_factor=2,
    )
    assert len(pair_records) == 12
    for split in ("calibration", "evaluation"):
        rows = [record for record in pair_records if record["split"] == split]
        assert len(rows) == 6
        assert len({record["pair"] for record in rows}) == 6
        assert all(record["estimator_variant"] == "gcc_phat_pair" for record in rows)
    calibration_seed = next(
        record["seed"] for record in pair_records if record["split"] == "calibration"
    )
    evaluation_seed = next(
        record["seed"] for record in pair_records if record["split"] == "evaluation"
    )
    assert calibration_seed != evaluation_seed
    assert {record["estimator_variant"] for record in doa_records} == set(DOA_VARIANTS)
    assert len(doa_records) == 9
    for record in doa_records:
        assert record["successful_trial_count"] + record["unsuccessful_trial_count"] == 32
        assert record["successful_fraction"] == record["coverage"]
        assert record["metric_conditioning"] == "successful_trials_only"
        assert record["conditional_p95_geodesic_error_deg"] <= record[
            "conditional_p99_geodesic_error_deg"
        ]
        assert record["conditional_p99_geodesic_error_deg"] <= record[
            "conditional_p999_geodesic_error_deg"
        ]
        assert record["p99_geodesic_error_deg"] == record[
            "conditional_p99_geodesic_error_deg"
        ]
        assert record["p999_geodesic_error_deg"] == record[
            "conditional_p999_geodesic_error_deg"
        ]
    projected = next(
        record for record in doa_records if record["estimator_variant"] == "all_6_cycle_equal"
    )
    assert projected["mean_cycle_residual_before_us"] > 0.0
    assert projected["mean_cycle_residual_after_us"] < 1e-9
    assert len(json.loads(covariance["pair_bias_vector_us"])) == 6
    assert len(json.loads(covariance["covariance_matrix_us2"])) == 6
    assert covariance["benchmark_name"] == "Gaussian covariance benchmark"
    assert covariance["exact_crlb_claimed"] is False
    thresholds = json.loads(covariance["confidence_thresholds_by_percentile"])
    assert set(thresholds) == {"p05", "p10", "p25", "p50"}


def test_risk_coverage_modes_separate_hard_soft_and_no_rejection():
    _, records, _ = run_gcc_statistical_configuration(
        _config(),
        92,
        calibration_trial_count=40,
        evaluation_trial_count=60,
        interpolation_factor=2,
    )
    by_name = {row["estimator_variant"]: row for row in records}
    no_rejection = by_name["all_6_cycle_calibrated_no_rejection"]
    soft = by_name["all_6_cycle_calibrated_soft"]
    p05 = by_name["all_6_cycle_calibrated_hard_p05"]
    p10 = by_name["all_6_cycle_calibrated"]
    p25 = by_name["all_6_cycle_calibrated_hard_p25"]
    p50 = by_name["all_6_cycle_calibrated_hard_p50"]
    assert no_rejection["risk_coverage_study"] is True
    assert no_rejection["hard_confidence_rejection_used"] is False
    assert no_rejection["soft_confidence_weighting_used"] is False
    assert no_rejection["mean_rejected_pair_fraction"] == 0.0
    assert soft["soft_confidence_weighting_used"] is True
    assert soft["hard_confidence_rejection_used"] is False
    assert soft["mean_rejected_pair_fraction"] == 0.0
    assert [row["confidence_threshold_percentile"] for row in (p05, p10, p25, p50)] == [
        5.0,
        10.0,
        25.0,
        50.0,
    ]
    assert p05["mean_rejected_pair_fraction"] <= p10["mean_rejected_pair_fraction"]
    assert p10["mean_rejected_pair_fraction"] <= p25["mean_rejected_pair_fraction"]
    assert p25["mean_rejected_pair_fraction"] <= p50["mean_rejected_pair_fraction"]
    assert p05["coverage"] >= p50["coverage"]
    assert p50["disconnected_graph_failure_fraction"] >= p05[
        "disconnected_graph_failure_fraction"
    ]


def test_statistical_csv_has_required_context_fields_and_is_reproducible(tmp_path):
    paths = [tmp_path / name for name in ("pair.csv", "doa.csv", "cov.csv")]
    kwargs = dict(
        configurations=(_config(),),
        calibration_trial_count=16,
        evaluation_trial_count=24,
        interpolation_factor=2,
        pair_output_csv=paths[0],
        doa_output_csv=paths[1],
        covariance_output_csv=paths[2],
    )
    first = run_gcc_statistical_validation(**kwargs)
    first_bytes = [path.read_bytes() for path in paths]
    second = run_gcc_statistical_validation(**kwargs)
    assert first == second
    assert [path.read_bytes() for path in paths] == first_bytes
    required = {
        "study_type",
        "signal_model",
        "split",
        "geometry",
        "direction",
        "SNR",
        "frame_length",
        "pair",
        "calibration_trial_count",
        "evaluation_trial_count",
        "seed",
        "estimator_variant",
    }
    for path in paths:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        assert rows
        assert required <= set(rows[0])


def test_frame_length_study_is_separate_from_fine_snr_study():
    configurations = default_statistical_configurations()
    fine = [config for config in configurations if config.study_type == "fine_snr"]
    frame = [config for config in configurations if config.study_type == "frame_length"]
    assert {config.snr_db for config in fine} == set(FINE_SNR_LEVELS_DB)
    assert {config.frame_length for config in frame} == set(FRAME_LENGTHS)
    assert all(config.frame_length == 2048 for config in fine)
    assert all(config.study_type != "fine_snr" for config in frame)


def test_spherical_study_separates_measurement_and_plane_model_bias():
    config = default_spherical_configurations()[0]
    pair_records, doa_records, _ = run_spherical_model_configuration(
        config,
        301,
        calibration_trial_count=16,
        evaluation_trial_count=24,
        interpolation_factor=2,
        far_field_angular_step_deg=20.0,
    )
    assert len(pair_records) == 12
    assert {row["estimator_variant"] for row in doa_records} == {
        "spherical_signal_plane_model",
        "spherical_signal_exact_known_range",
    }
    for row in doa_records:
        assert row["gcc_measurement_rmse_us"] > 0.0
        assert row["direction_specific_plane_model_bias_max_us"] > 0.0
        assert row["global_E_tau_us"] >= row["direction_specific_plane_model_bias_max_us"]
        assert row["noiseless_exact_doa_bias_deg"] < 1e-6
        assert row["noiseless_plane_doa_bias_deg"] > row["noiseless_exact_doa_bias_deg"]
