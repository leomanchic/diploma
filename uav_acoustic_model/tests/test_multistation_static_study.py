"""Fast smoke checks for the bearing-level static multi-station study."""

import json

import numpy as np

from validation.multistation_static_study import (
    default_static_scenarios,
    default_static_study_configurations,
    run_multistation_static_study,
    run_static_configuration,
)


def test_default_study_sweeps_geometry_scale_range_altitude_and_mismatch():
    configurations = default_static_study_configurations()
    scenarios = default_static_scenarios()
    assert len(configurations) == 9
    assert {item.station_geometry for item in configurations} == {
        "equilateral",
        "elongated",
        "near_collinear",
    }
    assert len({item.baseline_m for item in configurations}) >= 4
    assert max(item.target_horizontal_range_over_baseline for item in configurations) >= 10.0
    assert {item.orientation_mode for item in configurations} == {"aligned", "varied"}
    assert {len(item.station_indices) for item in scenarios} == {2, 3}
    assert {item.pose_mismatch_kind for item in scenarios} == {
        "none",
        "position",
        "orientation",
    }
    assert any(item.bearing_covariance_mismatch for item in scenarios)
    assert any(item.outlier_station_index is not None for item in scenarios)


def test_smoke_study_uses_disjoint_splits_calibration_only_and_bearing_level_truth_boundary():
    records, geometry = run_static_configuration(
        default_static_study_configurations()[0],
        0,
        calibration_realizations=12,
        evaluation_realizations=16,
    )
    assert len(records) == 2 * len(default_static_scenarios())
    assert geometry["closest_rays_rank"] == 3
    assert not geometry["truth_used_online"]
    assert {row["split"] for row in records} == {"calibration", "evaluation"}
    assert all(row["bearing_covariance_source_split"] == "calibration" for row in records)
    assert all(not row["evaluation_residual_used_to_fit_bearing_covariance"] for row in records)
    assert all(row["calibration_evaluation_seed_overlap_count"] == 0 for row in records)
    assert all(not row["residual_components_are_independent_trials"] for row in records)
    assert all(not row["truth_used_online"] for row in records)
    assert all(not row["dynamic_tracking_implemented"] for row in records)
    assert all(not row["retarded_time_fusion_implemented"] for row in records)
    assert all(row["coverage"] == 1.0 for row in records)
    for row in records:
        assert row["dependent_residual_component_count"] == (
            row["independent_realization_count"] * row["station_count"] * 2
        )
        calibration = json.loads(row["calibrated_bearing_statistics_json"])
        assert len(calibration) == row["station_count"]
        assert all(item["R_rank"] == 2 for item in calibration)

    evaluation = {row["scenario"]: row for row in records if row["split"] == "evaluation"}
    assert evaluation["ideal_three"]["position_rmse_m"] < evaluation["ideal_two"]["position_rmse_m"]
    assert evaluation["one_erroneous_bearing"]["position_rmse_m"] > 3.0 * evaluation["ideal_three"]["position_rmse_m"]
    assert np.isfinite(evaluation["ideal_three"]["normalized_position_error_p95"])


def test_csv_writer_produces_separate_summary_and_geometry_tables(tmp_path):
    summary_path = tmp_path / "summary.csv"
    geometry_path = tmp_path / "geometry.csv"
    records, geometries = run_multistation_static_study(
        calibration_realizations=4,
        evaluation_realizations=4,
        summary_csv=summary_path,
        geometry_csv=geometry_path,
    )
    assert len(records) == 9 * 7 * 2
    assert len(geometries) == 9
    assert summary_path.read_text(encoding="utf-8").count("\n") == len(records) + 1
    assert geometry_path.read_text(encoding="utf-8").count("\n") == len(geometries) + 1

