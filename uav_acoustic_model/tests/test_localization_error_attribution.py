"""Frozen selection and truth-boundary tests for localization diagnostics."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.localization_error_attribution import (
    SELECTED_INDEXES, SOURCE_MANIFEST_SHA256, SOURCE_RUN_IDS_SHA256,
    SOURCE_STUDY, VARIANTS,
    _diagnostic_id, _geometry_rows, _load_diagnostic, _make_measurements, initialize,
)
from model.bearing_events import bearing_event_id
from model.measurements import BearingMeasurement
from simulation.gazebo_offline import shared_stations
from validation.gazebo_experiment import canonical_sha256, sha256
from validation.localization_range_study import _load_study, _run_id

EXPECTED_IDS = (
    "range-272dde7dd08ac45c22d609f9", "range-78be84b1f71af5d1f2e0b5f9",
    "range-858892c886a576ff30a1e939", "range-a8865603f5b0100fb597490c",
    "range-d50f18cf1a39e2e16bab8348", "range-62f93e0cb51c04249c66dded",
    "range-6226b1780a8e2d97ad7ffbc1", "range-61aa70fe53f59e5908252f7b",
)


def test_initialization_freezes_exact_published_cases(tmp_path) -> None:
    source = _load_study(SOURCE_STUDY)
    assert sha256(SOURCE_STUDY / "study_manifest.json") == SOURCE_MANIFEST_SHA256
    assert tuple(_run_id(source, source["runs"][i]) for i in SELECTED_INDEXES) == EXPECTED_IDS
    manifest = initialize(tmp_path / "diagnostic")
    assert manifest["configuration_frozen_before_processing"] is True
    assert manifest["case_count"] == manifest["audio_restoration_limit"] == 8
    assert tuple(item["source_run_id"] for item in manifest["cases"]) == EXPECTED_IDS
    assert len({item["diagnostic_id"] for item in manifest["cases"]}) == 8
    assert manifest["processing_snapshot_sha256"] == canonical_sha256(source["processing"])
    assert all(item["diagnostic_id"] == _diagnostic_id(item["source_run_id"])
               for item in manifest["cases"])


def _row(frame_index: int, *, valid: bool) -> dict:
    truth = np.asarray([1.0, 0.0, 0.0])
    estimate = np.asarray([0.99995, 0.01, 0.0])
    estimate /= np.linalg.norm(estimate)
    probe = BearingMeasurement(
        station_id="S0", sequence_id="source-run", frame_index=frame_index,
        reception_center_timestamp_s=1.0 + frame_index,
        available_timestamp_s=1.1 + frame_index, direction_local=truth,
        covariance_tangent_rad2=np.eye(2), calibration_bias_tangent_rad=np.zeros(2),
        estimator_variant="all_6_equal_gcc_wls",
    )
    return {
        "station_id": "S0", "sequence_id": "source-run", "frame_index": frame_index,
        "frame_center_reception_time_s": 1.0 + frame_index,
        "available_timestamp_s": 1.1 + frame_index,
        "estimator_variant": "all_6_equal_gcc_wls", "quality_metadata": {"score": 2.0},
        "truth_local": truth, "estimate_local": estimate, "valid": valid,
        "invalid_reason": "" if valid else "audio_bearing_invalid",
        "event_id": bearing_event_id(probe),
    }


def test_variants_share_schedule_and_keep_truth_out_of_measurement() -> None:
    rows = [_row(0, valid=True), _row(1, valid=False)]
    calibration = SimpleNamespace(
        covariance_rad2=np.diag([1e-4, 2e-4]),
        mean_residual_rad=np.asarray([0.02, -0.01]),
    )
    calibrations = {("S0", "all_6_equal_gcc_wls"): calibration}
    built = {variant: _make_measurements(
        rows, calibrations, "all_6_equal_gcc_wls", variant, 1
    ) for variant in VARIANTS}
    schedules = [[(m.station_id, m.frame_index, m.reception_center_timestamp_s,
                   m.available_timestamp_s) for m in measurements]
                 for measurements in built.values()]
    assert schedules[0] == schedules[1] == schedules[2]
    assert built["original"][0].calibration_bias_tangent_rad.tolist() == [0.02, -0.01]
    assert built["original"][1].valid is False
    assert built["zero_bias"][1].valid is False
    assert all(m.valid for m in built["ideal_bearing"])
    assert all(np.array_equal(m.calibration_bias_tangent_rad, np.zeros(2))
               for variant in ("ideal_bearing", "zero_bias") for m in built[variant] if m.valid)
    assert not hasattr(built["ideal_bearing"][0], "truth_position")
    assert not hasattr(built["ideal_bearing"][0], "true_emission_time_s")


def test_static_geometry_benchmark_is_full_rank_and_labeled() -> None:
    stations = shared_stations()
    trajectory = SimpleNamespace(q=lambda time: np.asarray([300.0 + time, 280.0, 80.0]))
    rows = _geometry_rows(stations, trajectory, [0.0, 1.0])
    assert [row["rank"] for row in rows] == [3, 3]
    assert all(np.isfinite(row["condition_number"]) and row["condition_number"] > 1.0 for row in rows)
    assert all(0.0 <= row["weak_direction_radial_alignment_abs"] <= 1.0 for row in rows)
    assert all(row["benchmark_kind"] == "local_static_Gaussian_linearization" for row in rows)


EXPECTED_RESULT_TABLES = {
    "bearing_station_summary.csv": "0367eb4383b576d44f89d95e8405502e021fed4f1b2775468eef9847ae120862",
    "geometry_summary.csv": "a476813d0d1c2a1b67d8b60fb9907b5b67767309620afdcc582a5d0b86cdb4dc",
    "reproduction_summary.csv": "1ba1e737c914791b6a82a8020b506dbb2cd7356ae5f2a5a7b3d46eb5d55625c6",
    "variant_summary.csv": "bb311e4e301191206cf0c7fac0c9a1f526e8d8d02f7fbf305c19357cffb7b420",
}

EXPECTED_DERIVED_TABLES = {
    "diagnosis_table.csv": "fabd96776d16c141be6e2c490c0d0e4a533fdb225dd24cc7db50ea2155f41987",
    "first_confirmation_summary.csv": "d76a91758c8c33bc0cfc8e39bb0c3c01ab6a9b0533b5887873e7955fb86f9107",
    "geometry_case_summary.csv": "e3aa7ae963e355792e6c447dd3d9e0c2c419095c375489b9078c34617e941578",
    "initialization_summary.csv": "e97ac4ada5dc15ad17a9d515656feecb545ef059531d333a7b25a854eba083b5",
    "phase_summary.csv": "1c6f5cae690372312c9f0f9812ca85b5ac2efa6a1a9c67f8bd020b956d7a9b89",
    "uncertainty_error_summary.csv": "28e39f62fd4d29d6be07818a4428acc857f3001abad23e85c34644eeb3b40e1b",
    "update_summary.csv": "dde1d1cdb30e7a8cec79c85ef4622f8c70236b01379352410f6171d78e31cd20",
}


def test_checked_in_diagnostic_is_complete_and_byte_verified() -> None:
    output = Path("results/localization_error_attribution")
    manifest = _load_diagnostic(output)
    summary = json.loads((output / "study_summary.json").read_text())
    assert summary["case_count"] == 8
    assert summary["method_variant_count"] == 48
    assert summary["audio_restoration_count"] == 8
    assert summary["all_original_reproductions_passed"] is True
    assert summary["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256
    assert summary["source_completed_run_ids_sha256"] == SOURCE_RUN_IDS_SHA256
    assert summary["tables"] == EXPECTED_RESULT_TABLES
    assert all(sha256(output / name) == digest
               for name, digest in EXPECTED_RESULT_TABLES.items())

    for case in manifest["cases"]:
        prefix = f"{int(case['index']):03d}_{case['source_run_id']}_{case['diagnostic_id']}"
        directory = output / "cases" / prefix
        experiment = json.loads((directory / "experiment.json").read_text())
        assert experiment["status"] == "complete"
        assert experiment["audio_restoration_count"] == 1
        assert all(sha256(directory / name) == digest
                   for name, digest in experiment["result_sha256"].items())

    with (output / "reproduction_summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert all(row["passed"] == "True" and row["exact_mismatch_count"] == "0"
               for row in rows)
    numeric_fields = (
        "maximum_bearing_error_difference_deg",
        "maximum_tracking_numeric_difference",
        "maximum_tracking_time_difference_s",
    )
    assert all(float(row[field]) == 0.0 for row in rows for field in numeric_fields)


    analysis = json.loads((output / "analysis_summary.json").read_text())
    assert analysis["derived_tables"] == EXPECTED_DERIVED_TABLES
    assert all(sha256(output / name) == digest
               for name, digest in EXPECTED_DERIVED_TABLES.items())

    with (output / "first_confirmation_summary.csv").open(newline="") as stream:
        confirmations = list(csv.DictReader(stream))
    assert len(confirmations) == 48
    confirmed = [row for row in confirmations if row["confirmed"] == "True"]
    failed = [row for row in confirmations if row["confirmed"] == "False"]
    assert len(confirmed) == 44 and len(failed) == 4
    assert all(row["first_confirmation_velocity_error_mps"] for row in confirmed)
    assert all(not row["first_confirmation_velocity_error_mps"] for row in failed)

    with (output / "uncertainty_error_summary.csv").open(newline="") as stream:
        uncertainty = list(csv.DictReader(stream))
    assert len(uncertainty) == 48
    assert sum(int(row["valid_publication_count"]) == 0 for row in uncertainty) == 4
