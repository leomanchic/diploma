"""Fast smoke and numerical criteria for persisted propagation studies."""

import csv

import numpy as np

from validation.propagation_study import (
    DISTANCES_M,
    run_far_field_boundary_study,
    run_fractional_delay_accuracy_study,
)


def test_small_grid_far_field_study_writes_complete_consistent_csv(tmp_path):
    output = tmp_path / "far_field.csv"
    records = run_far_field_boundary_study(
        output_csv=output,
        angular_step_deg=30.0,
        refined_angular_step_deg=15.0,
    )
    sweep = [row for row in records if row["record_type"] == "distance_sweep"]
    boundaries = [row for row in records if row["record_type"] == "boundary"]
    assert len(sweep) == 5 * len(DISTANCES_M)
    assert len(boundaries) == 25
    assert all(row["achieved_error_s"] <= row["target_error_s"] for row in boundaries)
    required = {
        "coarse_max_error",
        "refined_max_error",
        "relative_grid_difference",
        "refined_worst_azimuth_deg",
        "refined_worst_elevation_deg",
        "refined_worst_pair",
    }
    assert all(required <= set(row) for row in boundaries)
    assert all(
        bool(row["continuous_refinement_used"])
        == (row["relative_grid_difference"] > 1e-4)
        for row in boundaries
    )
    for geometry in {row["geometry"] for row in sweep}:
        errors = [
            row["max_plane_error_s"] for row in sweep if row["geometry"] == geometry
        ]
        assert np.all(np.diff(errors) < 0.0)
    with output.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(records)


def test_fractional_delay_study_writes_accuracy_metrics(tmp_path):
    output = tmp_path / "fractional.csv"
    records = run_fractional_delay_accuracy_study(output_csv=output)
    tones = [row for row in records if row["record_type"] == "tone"]
    broadband = [row for row in records if row["record_type"] == "broadband"]
    assert len(tones) == 70
    assert len(broadband) == 10
    assert max(row["amplitude_error"] for row in tones) < 2e-5
    assert max(row["phase_error_rad"] for row in tones) < 1e-5
    assert max(abs(row["group_delay_error_samples"]) for row in broadband) < 2e-6
    assert max(row["cross_method_max_error_valid"] for row in broadband) < 3e-6
    with output.open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == len(records)
