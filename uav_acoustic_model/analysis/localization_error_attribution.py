"""Reproducible diagnostics for eight frozen localization range-study runs.

The published 120-run study is immutable. This module restores exactly eight
continuous audio streams, saves complete bearing records, then reuses those
records for original, ideal-bearing, and zero-bias tracker inputs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.stats import chi2

from estimators.retarded_ekf_manoeuvre import CausalManoeuvreRetardedTimeEKF, ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.bearing_events import bearing_event_id
from model.bearing_statistics import tangent_basis
from model.geometry import direction_angles
from model.measurements import BearingMeasurement
from simulation.gazebo_offline import load_gazebo_recording
from simulation.multistation_audio import synthesize_multistation_audio
from simulation.moving_source import solve_emission_time
from validation.gazebo_experiment import (
    calibration_from_experiment, canonical_sha256, sha256, stations_from_experiment,
)
from validation.localization_range_study import (
    DEFAULT_OUTPUT as SOURCE_STUDY, REFERENCE_DISTANCE_M, ROOT, ROOT_SEED,
    _load_study, _phase_classifier, _run_directory, _run_id,
    _standard_noise_sha256, _translated,
)
from validation.three_station_audio_tracking_study import extract_audio_bearing_records

SCHEMA_VERSION = 1
DEFAULT_OUTPUT = ROOT / "results" / "localization_error_attribution"
PROTOCOL = ROOT / "LOCALIZATION_ERROR_ATTRIBUTION_PROTOCOL.md"
RUNNER = Path(__file__).resolve()
SELECTED_INDEXES = (2, 3, 4, 8, 9, 10, 39, 61)
METHODS = ("all_6_equal_gcc_wls", "equal_weight_srp_phat")
VARIANTS = ("original", "ideal_bearing", "zero_bias")
SOURCE_MANIFEST_SHA256 = "00f3e2e259885d34abd9e2247e1aa1ed0f74c201ad63d4981dfd6fe8b5d194e6"
SOURCE_PROCESSING_SHA256 = "7a1f64919d5eeaaf41bbe5ffd39ed3043f37b880016d6cf6b35bb55e23b468d6"
SOURCE_RUN_IDS_SHA256 = "6c7704f02e499976c25028c9a328cc1fcc52d851384ca989275cacf3f76bbd4c"
POSITION_TOLERANCE_M = 1e-7
ANGULAR_TOLERANCE_DEG = 1e-10
TIME_TOLERANCE_S = 1e-12
GEOMETRY_SIGMA_RAD = float(np.deg2rad(1.0))
POSITION_COVERAGE_THRESHOLD = float(chi2.ppf(0.95, 3))
STATE_COVERAGE_THRESHOLD = float(chi2.ppf(0.95, 6))

BEARING_COLUMNS = (
    "run_id", "sequence_id", "event_id", "station_id", "estimator_variant", "frame_index",
    "frame_start_reception_time_s", "frame_center_reception_time_s", "frame_end_reception_time_s",
    "true_emission_time_s_evaluator_only", "available_timestamp_s", "modeled_processing_delay_s",
    "modeled_delivery_delay_s", "measured_algorithm_runtime_s", "effective_station_snr_db",
    "valid", "invalid_reason", "boundary_hit", "truth_local_x", "truth_local_y", "truth_local_z",
    "truth_world_x", "truth_world_y", "truth_world_z", "estimate_local_x", "estimate_local_y",
    "estimate_local_z", "estimate_world_x", "estimate_world_y", "estimate_world_z",
    "residual_az_arc_rad", "residual_el_arc_rad", "geodesic_error_deg", "quality_metadata_json",
    "source_seed", "noise_seed", "truth_used_by_audio_estimator", "frame_from_continuous_stream",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _plain(value: Any) -> Any:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: _plain(row.get(name, "")) for name in names} for row in rows)


def _write_csv_gz(path: Path, rows: list[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    names = list(columns or (list(rows[0]) if rows else ()))
    if not names:
        raise ValueError(f"empty compressed CSV requires columns: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=names, lineterminator="\n")
                writer.writeheader()
                writer.writerows({name: _plain(row.get(name, "")) for name in names} for row in rows)
    os.replace(temporary, path)


def _read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _diagnostic_id(run_id: str) -> str:
    return "attr-" + canonical_sha256({
        "schema_version": SCHEMA_VERSION, "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "protocol_sha256": sha256(PROTOCOL), "source_run_id": run_id,
    })[:24]


def _case_directory(output: Path, case: dict[str, Any]) -> Path:
    return Path(output) / "cases" / f"{int(case['index']):03d}_{case['source_run_id']}_{case['diagnostic_id']}"


def initialize(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    source = _load_study(SOURCE_STUDY)
    if sha256(SOURCE_STUDY / "study_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise ValueError("published range-study manifest SHA-256 mismatch")
    if source["processing_sha256"] != SOURCE_PROCESSING_SHA256:
        raise ValueError("published processing snapshot SHA-256 mismatch")
    if canonical_sha256(sorted(_run_id(source, spec) for spec in source["runs"])) != SOURCE_RUN_IDS_SHA256:
        raise ValueError("published range-study run IDs changed")
    cases = []
    for index in SELECTED_INDEXES:
        spec = source["runs"][index]
        run_id = _run_id(source, spec)
        source_directory = _run_directory(SOURCE_STUDY, spec, run_id)
        experiment = json.loads((source_directory / "experiment.json").read_text())
        if int(spec["index"]) != index or experiment["run_id"] != run_id or experiment["status"] != "complete":
            raise ValueError(f"published source run is not complete: {run_id}")
        cases.append({**spec, "source_run_id": run_id, "diagnostic_id": _diagnostic_id(run_id),
                      "source_experiment_sha256": sha256(source_directory / "experiment.json"),
                      "source_summary_sha256": sha256(source_directory / "summary.json")})
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "configuration_frozen_before_processing": True,
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(), "protocol_sha256": sha256(PROTOCOL),
        "diagnostic_runner_path": RUNNER.relative_to(ROOT).as_posix(),
        "diagnostic_runner_sha256": sha256(RUNNER),
        "source_study_path": SOURCE_STUDY.relative_to(ROOT).as_posix(),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_protocol_sha256": source["protocol_sha256"], "source_code_sha256": source["code_sha256"],
        "source_processing_sha256": source["processing_sha256"],
        "source_completed_run_ids_sha256": SOURCE_RUN_IDS_SHA256,
        "git_commit_sha_at_initialization": _git_sha(), "selected_indexes": list(SELECTED_INDEXES),
        "case_count": len(cases), "audio_restoration_limit": 8, "methods": list(METHODS),
        "variants": list(VARIANTS), "processing": source["processing"], "recordings": source["recordings"],
        "source_banks": source["source_banks"], "geometry": source["geometry"], "cases": cases,
        "reproduction_tolerances": {"position_absolute_m": POSITION_TOLERANCE_M,
            "angular_absolute_deg": ANGULAR_TOLERANCE_DEG, "timestamp_absolute_s": TIME_TOLERANCE_S,
            "statuses_counts_reasons": "exact", "runtime": "excluded"},
        "geometry_benchmark": {"kind": "local_static_Gaussian_linearization",
            "angular_standard_deviation_rad": GEOMETRY_SIGMA_RAD, "angular_standard_deviation_deg": 1.0,
            "shared_static_source_position_per_publication_epoch": True,
            "dynamic_asynchronous_ray_intersection": False},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
    }
    manifest["processing_snapshot_sha256"] = canonical_sha256(manifest["processing"])
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "diagnostic_manifest.json", manifest)
    return manifest


def _load_diagnostic(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    manifest = json.loads((Path(output) / "diagnostic_manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION or sha256(PROTOCOL) != manifest["protocol_sha256"]:
        raise ValueError("diagnostic schema/protocol integrity check failed")
    if sha256(RUNNER) != manifest["diagnostic_runner_sha256"]:
        raise ValueError("diagnostic runner changed after initialization")
    if sha256(SOURCE_STUDY / "study_manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("published source manifest changed")
    source = _load_study(SOURCE_STUDY)
    if canonical_sha256(manifest["processing"]) != manifest["processing_snapshot_sha256"] or manifest["processing"] != source["processing"]:
        raise ValueError("diagnostic processing snapshot changed")
    if manifest["selected_indexes"] != list(SELECTED_INDEXES):
        raise ValueError("diagnostic case selection changed")
    for case in manifest["cases"]:
        if _run_id(source, source["runs"][int(case["index"])]) != case["source_run_id"]:
            raise ValueError("diagnostic source run identity changed")
    return manifest


def _serialize_bearing(row: dict[str, Any], run_id: str) -> dict[str, Any]:
    truth_local, truth_world = np.asarray(row["truth_local"], float), np.asarray(row["truth_world"], float)
    estimate_local, estimate_world = np.asarray(row["estimate_local"], float), np.asarray(row["estimate_world"], float)
    residual = np.asarray(row["residual_rad"], float)
    saved = {
        "run_id": run_id, "sequence_id": str(row["sequence_id"]), "station_id": str(row["station_id"]),
        "estimator_variant": str(row["estimator_variant"]), "frame_index": int(row["frame_index"]),
        "frame_start_reception_time_s": row["frame_start_reception_time_s"],
        "frame_center_reception_time_s": row["frame_center_reception_time_s"],
        "frame_end_reception_time_s": row["frame_end_reception_time_s"],
        "true_emission_time_s_evaluator_only": row["true_emission_time_s_evaluator_only"],
        "available_timestamp_s": row["available_timestamp_s"],
        "modeled_processing_delay_s": row["modeled_processing_delay_s"],
        "modeled_delivery_delay_s": row["modeled_delivery_delay_s"],
        "measured_algorithm_runtime_s": row["measured_algorithm_runtime_s"],
        "effective_station_snr_db": row["effective_station_snr_db"], "valid": bool(row["valid"]),
        "invalid_reason": str(row["invalid_reason"]), "boundary_hit": bool(row["boundary_hit"]),
        "truth_local_x": truth_local[0], "truth_local_y": truth_local[1], "truth_local_z": truth_local[2],
        "truth_world_x": truth_world[0], "truth_world_y": truth_world[1], "truth_world_z": truth_world[2],
        "estimate_local_x": estimate_local[0], "estimate_local_y": estimate_local[1],
        "estimate_local_z": estimate_local[2], "estimate_world_x": estimate_world[0],
        "estimate_world_y": estimate_world[1], "estimate_world_z": estimate_world[2],
        "residual_az_arc_rad": residual[0], "residual_el_arc_rad": residual[1],
        "geodesic_error_deg": row["geodesic_error_deg"],
        "quality_metadata_json": json.dumps(row["quality_metadata"], sort_keys=True, separators=(",", ":")),
        "source_seed": row["source_seed"], "noise_seed": row["noise_seed"],
        "truth_used_by_audio_estimator": False,
        "frame_from_continuous_stream": bool(row["frame_from_continuous_stream"]),
    }
    identity_probe = BearingMeasurement(
        station_id=saved["station_id"], sequence_id=saved["sequence_id"], frame_index=saved["frame_index"],
        reception_center_timestamp_s=saved["frame_center_reception_time_s"],
        available_timestamp_s=saved["available_timestamp_s"], direction_local=truth_local,
        covariance_tangent_rad2=np.eye(2), calibration_bias_tangent_rad=np.zeros(2),
        estimator_variant=saved["estimator_variant"],
    )
    saved["event_id"] = bearing_event_id(identity_probe)
    return saved


def _bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else value == "True"


def _vector(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray([float(row[f"{prefix}_{axis}"]) for axis in "xyz"])


def _measurement_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _read_csv_gz(path):
        row: dict[str, Any] = dict(source)
        for name in ("frame_index", "source_seed", "noise_seed"):
            row[name] = int(source[name])
        for name in (
            "frame_start_reception_time_s", "frame_center_reception_time_s", "frame_end_reception_time_s",
            "true_emission_time_s_evaluator_only", "available_timestamp_s", "modeled_processing_delay_s",
            "modeled_delivery_delay_s", "measured_algorithm_runtime_s", "effective_station_snr_db",
            "residual_az_arc_rad", "residual_el_arc_rad", "geodesic_error_deg",
        ):
            row[name] = float(source[name])
        for name in ("valid", "boundary_hit", "truth_used_by_audio_estimator", "frame_from_continuous_stream"):
            row[name] = _bool(source[name])
        row["quality_metadata"] = json.loads(source["quality_metadata_json"])
        for prefix in ("truth_local", "truth_world", "estimate_local", "estimate_world"):
            row[prefix] = _vector(source, prefix)
        rows.append(row)
    return rows


def _restore_bearings(output: Path, diagnostic: dict[str, Any], case: dict[str, Any], source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    directory = _case_directory(output, case)
    bearing_path, manifest_path = directory / "bearing_records.csv.gz", directory / "bearing_manifest.json"
    if bearing_path.exists() and manifest_path.exists():
        cache = json.loads(manifest_path.read_text())
        if cache["source_run_id"] != case["source_run_id"] or sha256(bearing_path) != cache["bearing_records_sha256"]:
            raise ValueError("bearing cache identity or SHA-256 mismatch")
        return _measurement_rows(bearing_path), cache, False
    processing = diagnostic["processing"]
    stations = stations_from_experiment({"processing": processing})
    info = source["recordings"][case["trajectory"]]
    recording = load_gazebo_recording(ROOT / info["path"])
    reception_start, duration = float(info["reception_start_s"]), float(info["duration_s"])
    trajectory, offset = _translated(recording, stations, reception_start, case["distance_m"], case["trajectory"])
    bank = source["source_banks"][case["trajectory"]]
    source_signal = np.load(SOURCE_STUDY / bank["path"], allow_pickle=False)
    if hashlib.sha256(source_signal.tobytes()).hexdigest() != bank["signal_sha256"]:
        raise ValueError("source signal content SHA-256 mismatch")
    audio = processing["audio"]
    started = time.perf_counter()
    stream = synthesize_multistation_audio(
        stations, trajectory, duration_s=duration, reception_start_time_s=reception_start,
        sampling_rate_hz=float(audio["sampling_rate_hz"]), sound_speed=float(audio["sound_speed_mps"]),
        signal_model=str(audio["signal_model"]), snr_db=float(case["snr_db"]), seed=ROOT_SEED,
        chunk_size_samples=int(audio["chunk_size_samples"]), fir_length=int(audio["fir_length"]),
        geometric_attenuation=bool(case["geometric_attenuation"]),
        maximum_emitted_frequency_hz=float(audio["source_maximum_frequency_hz"]),
        external_source_signal=source_signal, external_source_start_time_s=float(bank["start_time_s"]),
        noise_model=str(case["noise_model"]), reference_distance_m=REFERENCE_DISTANCE_M,
        source_seed=int(case["source_seed"]), noise_seed=int(case["noise_seed"]), reference_source_rms=1.0,
    )
    synthesis_runtime = time.perf_counter() - started
    frontend = processing["frontend"]
    extracted, frontend_runtime = extract_audio_bearing_records(
        stream, stations, trajectory, split="diagnostic_replay", configuration_index=int(case["index"]),
        sequence_index=int(case["replicate"]), frame_length=int(frontend["frame_length"]),
        hop_length=int(frontend["hop_length"]), modeled_processing_delay_s=float(frontend["modeled_processing_delay_s"]),
        station_delivery_delay_s=frontend["station_delivery_delay_s"], sequence_id=case["source_run_id"],
    )
    serialized = [_serialize_bearing(row, case["source_run_id"]) for row in extracted]
    _write_csv_gz(bearing_path, serialized, BEARING_COLUMNS)
    cache = {
        "schema_version": SCHEMA_VERSION, "source_run_id": case["source_run_id"],
        "diagnostic_id": case["diagnostic_id"], "audio_restoration_count": 1,
        "bearing_record_count": len(serialized), "bearing_records_sha256": sha256(bearing_path),
        "recording_state_sha256": info["state_sha256"], "recording_manifest_sha256": info["manifest_sha256"],
        "source_signal_sha256": bank["signal_sha256"], "standardized_noise_sha256": _standard_noise_sha256(stream),
        "translation_enu_m": offset.tolist(), "reception_start_s": reception_start, "duration_s": duration,
        "actual_station_snr_db": {item.station_id: item.effective_snr_db for item in stream.stations},
        "audio_synthesis_wall_runtime_s": synthesis_runtime, "bearing_frontend_wall_runtime_s": frontend_runtime,
    }
    _write_json(manifest_path, cache)
    return _measurement_rows(bearing_path), cache, True


def _make_measurements(rows: list[dict[str, Any]], calibrations: dict[tuple[str, str], Any], method: str, variant: str, frame_stride: int) -> tuple[BearingMeasurement, ...]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown diagnostic variant: {variant}")
    measurements = []
    for row in rows:
        if row["estimator_variant"] != method or row["frame_index"] % frame_stride:
            continue
        calibration = calibrations[(row["station_id"], method)]
        common = dict(station_id=row["station_id"], sequence_id=row["sequence_id"], frame_index=row["frame_index"],
                      reception_center_timestamp_s=row["frame_center_reception_time_s"],
                      available_timestamp_s=row["available_timestamp_s"], estimator_variant=method,
                      quality_metadata=row["quality_metadata"])
        if variant == "ideal_bearing":
            measurement = BearingMeasurement(**common, direction_local=row["truth_local"],
                covariance_tangent_rad2=calibration.covariance_rad2,
                calibration_bias_tangent_rad=np.zeros(2), tangent_frame="prediction")
        elif row["valid"]:
            measurement = BearingMeasurement(**common, direction_local=row["estimate_local"],
                covariance_tangent_rad2=calibration.covariance_rad2,
                calibration_bias_tangent_rad=(calibration.mean_residual_rad if variant == "original" else np.zeros(2)),
                tangent_frame="prediction")
        else:
            measurement = BearingMeasurement.invalid(**common, invalid_reason=row["invalid_reason"])
        if bearing_event_id(measurement) != row["event_id"]:
            raise ValueError("saved bearing event identity changed")
        measurements.append(measurement)
    return tuple(measurements)


def _safe_float(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _diagnostic_rows(items: Iterable[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        row = asdict(item)
        for key, value in tuple(row.items()):
            if isinstance(value, (tuple, list, dict)):
                row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        rows.append(row)
    return rows


def _run_tracker_diagnostic(stations: tuple[Any, ...], trajectory: Any,
        measurements: tuple[BearingMeasurement, ...], method: str, variant: str,
        history: ManoeuvreHistoryConfig, recovery: InitializationRecoveryConfig,
        phase_classifier: Any) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    estimator = CausalManoeuvreRetardedTimeEKF(
        stations, measurements, estimator_variant=method, history_config=history, recovery_config=recovery)
    publication_times = sorted({item.available_timestamp_s for item in measurements})
    started = time.perf_counter()
    publications = [estimator.advance_to(epoch) for epoch in publication_times]
    runtime = time.perf_counter() - started
    measurement_map = {bearing_event_id(item): item for item in measurements}
    station_map = {station.station_id: station for station in stations}
    centroid = np.mean([station.position_world_m for station in stations], axis=0)
    track_rows, update_rows = [], []
    last_accepted = float("nan")
    for publication in publications:
        epoch = float(publication.processing_time_s)
        for diagnostic in publication.update_diagnostics:
            measurement = measurement_map[diagnostic.event_id]
            station = station_map[measurement.station_id]
            true_emission = float(solve_emission_time(
                measurement.reception_center_timestamp_s, station.position_world_m, trajectory))
            processing_time = float(getattr(diagnostic, "processing_time_s", epoch))
            applied = bool(diagnostic.update_applied)
            if applied:
                last_accepted = processing_time
            residual = np.asarray(getattr(diagnostic, "residual_tangent_rad", np.full(2, np.nan)), float)
            update_rows.append({
                "event_id": diagnostic.event_id, "station_id": measurement.station_id,
                "frame_index": measurement.frame_index, "estimator_variant": method,
                "diagnostic_variant": variant, "processing_time_s": processing_time,
                "reception_center_timestamp_s": measurement.reception_center_timestamp_s,
                "true_emission_time_s_evaluator_only": true_emission,
                "update_motion_phase_evaluator_only": phase_classifier(true_emission),
                "update_applied": applied, "failure_reason": diagnostic.failure_reason or "",
                "pre_update_nis": float(getattr(diagnostic, "pre_update_nis",
                    getattr(diagnostic, "normalized_innovation_squared", np.nan))),
                "residual_tangent_norm_rad": float(np.linalg.norm(residual)),
                "history_node_count": int(getattr(diagnostic, "history_node_count", 0)),
                "history_memory_bytes": int(getattr(diagnostic, "history_memory_bytes", 0)),
                "truth_used_by_tracker": False,
            })
        truth = np.concatenate((trajectory.q(epoch), trajectory.v(epoch)))
        valid = bool(publication.valid and publication.state is not None)
        estimate, covariance = np.full(6, np.nan), np.full((6, 6), np.nan)
        position_error = velocity_error = state_nees = position_nees = float("nan")
        radial_error = transverse_error = float("nan")
        radial_sigma = transverse_sigma = maximum_sigma = p95_scale = float("nan")
        state_covered = position_covered = False
        if valid:
            estimate = np.asarray(publication.state.vector, float)
            covariance = 0.5 * (np.asarray(publication.covariance_state, float) + np.asarray(publication.covariance_state, float).T)
            residual_state = estimate - truth
            position_error, velocity_error = float(np.linalg.norm(residual_state[:3])), float(np.linalg.norm(residual_state[3:]))
            radial = truth[:3] - centroid
            radial /= np.linalg.norm(radial)
            radial_error = float(residual_state[:3] @ radial)
            transverse_error = float(np.linalg.norm(residual_state[:3] - radial_error * radial))
            try:
                np.linalg.cholesky(covariance)
                state_nees = float(residual_state @ np.linalg.solve(covariance, residual_state))
                state_covered = state_nees <= STATE_COVERAGE_THRESHOLD
                position_covariance = covariance[:3, :3]
                position_nees = float(residual_state[:3] @ np.linalg.solve(position_covariance, residual_state[:3]))
                position_covered = position_nees <= POSITION_COVERAGE_THRESHOLD
                radial_variance = float(radial @ position_covariance @ radial)
                radial_sigma = float(np.sqrt(max(radial_variance, 0.0)))
                transverse_sigma = float(np.sqrt(max(np.trace(position_covariance) - radial_variance, 0.0)))
                maximum_sigma = float(np.sqrt(np.max(np.linalg.eigvalsh(position_covariance))))
                p95_scale = float(np.sqrt(POSITION_COVERAGE_THRESHOLD) * maximum_sigma)
            except np.linalg.LinAlgError:
                valid = False
        track_rows.append({
            "processing_time_s": epoch, "estimator_variant": method, "diagnostic_variant": variant,
            "valid": valid, "confirmed": bool(publication.confirmed), "status": publication.status,
            "failure_reason": publication.failure_reason or "", "truth_position_x_m": truth[0],
            "truth_position_y_m": truth[1], "truth_position_z_m": truth[2],
            "truth_velocity_x_mps": truth[3], "truth_velocity_y_mps": truth[4], "truth_velocity_z_mps": truth[5],
            "estimate_position_x_m": estimate[0], "estimate_position_y_m": estimate[1],
            "estimate_position_z_m": estimate[2], "estimate_velocity_x_mps": estimate[3],
            "estimate_velocity_y_mps": estimate[4], "estimate_velocity_z_mps": estimate[5],
            "position_error_m": position_error, "velocity_error_mps": velocity_error,
            "radial_position_error_m": radial_error, "transverse_position_error_m": transverse_error,
            "state_nees": state_nees, "position_nees": position_nees,
            "valid_and_state_covered": state_covered, "valid_and_position_covered": position_covered,
            "position_radial_sigma_m": radial_sigma, "position_transverse_sigma_rss_m": transverse_sigma,
            "position_maximum_axis_sigma_m": maximum_sigma, "position_maximum_axis_p95_scale_m": p95_scale,
            "reset_count": int(publication.reset_count), "generation": int(publication.generation),
            "publication_motion_phase_evaluator_only": phase_classifier(epoch),
            "last_accepted_update_processing_time_s": last_accepted,
            "time_since_last_accepted_update_s": epoch - last_accepted if np.isfinite(last_accepted) else float("nan"),
            "batch_optimization_count": int(publication.batch_optimization_count),
            "batch_optimization_budget": publication.batch_optimization_budget,
            "history_node_count": int(publication.history_node_count),
            "history_memory_bytes": int(publication.history_memory_bytes), "truth_used_by_tracker": False,
        })
    final = publications[-1]
    accepted = sum(row["update_applied"] for row in update_rows)
    errors = np.asarray([row["position_error_m"] for row in track_rows if row["valid"]], float)
    first_confirmation = float(final.first_confirmation_time_s)
    first_row = next((row for row in track_rows if row["valid"] and row["confirmed"]), None)
    sequence = {
        "estimator_variant": method, "diagnostic_variant": variant, "event_count": len(measurements),
        "valid_measurement_count": sum(item.valid for item in measurements), "publication_count": len(track_rows),
        "valid_publication_count": int(errors.size), "availability_fraction": errors.size / len(track_rows),
        "position_rmse_m_conditional": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
        "position_p95_m_conditional": float(np.percentile(errors, 95)) if errors.size else None,
        "first_confirmation_time_s_absolute": _safe_float(first_confirmation),
        "first_confirmation_position_error_m": first_row["position_error_m"] if first_row else None,
        "first_confirmation_velocity_error_mps": first_row["velocity_error_mps"] if first_row else None,
        "first_confirmation_radial_error_m": first_row["radial_position_error_m"] if first_row else None,
        "first_confirmation_transverse_error_m": first_row["transverse_position_error_m"] if first_row else None,
        "accepted_update_count": int(accepted), "rejected_update_count": int(len(update_rows) - accepted),
        "update_rejection_reasons": dict(Counter(row["failure_reason"] or "unspecified"
            for row in update_rows if not row["update_applied"])),
        "final_valid": bool(final.valid), "final_confirmed": bool(final.confirmed),
        "final_failure_reason": final.failure_reason or "", "reset_count": int(final.reset_count),
        "batch_optimization_count": int(final.batch_optimization_count),
        "batch_optimization_runtime_s": float(final.batch_optimization_runtime_s),
        "batch_optimization_budget": final.batch_optimization_budget,
        "first_budget_exhaustion_time_s": next((float(item.processing_time_s)
            for item in estimator.batch_fit_diagnostics if item.reason == "computational_budget_exceeded"), None),
        "tracker_runtime_s": runtime,
    }
    tables = {"batch_fits": _diagnostic_rows(estimator.batch_fit_diagnostics),
              "candidates": _diagnostic_rows(estimator.candidate_diagnostics),
              "hypotheses": _diagnostic_rows(final.hypothesis_diagnostics),
              "lifecycle": _diagnostic_rows(final.lifecycle_diagnostics)}
    return track_rows, sequence, update_rows, tables


def _geometry_rows(stations: tuple[Any, ...], trajectory: Any, epochs: list[float]) -> list[dict[str, Any]]:
    centroid = np.mean([station.position_world_m for station in stations], axis=0)
    rows = []
    for epoch in epochs:
        position = np.asarray(trajectory.q(epoch), float)
        blocks = []
        for station in stations:
            displacement = position - station.position_world_m
            distance = float(np.linalg.norm(displacement))
            direction = displacement / distance
            blocks.append(tangent_basis(*direction_angles(direction)) @ (np.eye(3) - np.outer(direction, direction)) / distance)
        jacobian = np.vstack(blocks)
        information = jacobian.T @ jacobian / GEOMETRY_SIGMA_RAD**2
        eigenvalues, eigenvectors = np.linalg.eigh(information)
        tolerance = max(eigenvalues[-1] * 1e-12, np.finfo(float).tiny)
        rank = int(np.count_nonzero(eigenvalues > tolerance))
        covariance = np.linalg.inv(information) if rank == 3 else np.full((3, 3), np.nan)
        radial = position - centroid
        radial /= np.linalg.norm(radial)
        weak = eigenvectors[:, 0]
        radial_variance = float(radial @ covariance @ radial) if rank == 3 else np.nan
        total_variance = float(np.trace(covariance)) if rank == 3 else np.nan
        rows.append({
            "processing_time_s": epoch, "rank": rank,
            "condition_number": float(eigenvalues[-1] / eigenvalues[0]) if rank == 3 else np.inf,
            "information_eigenvalue_min": eigenvalues[0], "information_eigenvalue_mid": eigenvalues[1],
            "information_eigenvalue_max": eigenvalues[2], "weak_direction_x": weak[0],
            "weak_direction_y": weak[1], "weak_direction_z": weak[2],
            "weak_direction_radial_alignment_abs": abs(float(weak @ radial)),
            "radial_sigma_m": np.sqrt(max(radial_variance, 0.0)),
            "transverse_sigma_rss_m": np.sqrt(max(total_variance - radial_variance, 0.0)),
            "maximum_axis_sigma_m": np.sqrt(1.0 / eigenvalues[0]) if rank == 3 else np.inf,
            "maximum_axis_p95_scale_m": np.sqrt(POSITION_COVERAGE_THRESHOLD / eigenvalues[0]) if rank == 3 else np.inf,
            "benchmark_sigma_theta_deg": 1.0, "benchmark_kind": "local_static_Gaussian_linearization",
        })
    return rows


def _numeric_difference(first: str, second: Any) -> float:
    if first == "" and second is None:
        return 0.0
    left, right = float(first), float(second)
    return 0.0 if np.isnan(left) and np.isnan(right) else abs(left - right)


def _verify_original_reproduction(source_directory: Path, method: str,
        bearing_rows: list[dict[str, Any]], tracking: list[dict[str, Any]],
        updates: list[dict[str, Any]], sequence: dict[str, Any]) -> dict[str, Any]:
    expected = json.loads((source_directory / "summary.json").read_text())["methods"][method]
    historical_track = _read_csv_gz(source_directory / f"tracking_{method}.csv.gz")
    historical_updates = _read_csv_gz(source_directory / f"updates_{method}.csv.gz")
    if len(historical_track) != len(tracking) or len(historical_updates) != len(updates):
        raise AssertionError("published tracking/update row count changed")
    exact_mismatches, maximum_position_difference, maximum_time_difference = 0, 0.0, 0.0
    for old, new in zip(historical_track, tracking, strict=True):
        for field in ("valid", "confirmed"):
            exact_mismatches += int(_bool(old[field]) != bool(new[field]))
        for field in ("status", "failure_reason"):
            exact_mismatches += int(old[field] != str(new[field]))
        maximum_time_difference = max(maximum_time_difference,
            abs(float(old["processing_time_s"]) - float(new["processing_time_s"])))
        for field in ("estimate_position_x_m", "estimate_position_y_m", "estimate_position_z_m",
                      "position_error_m", "velocity_error_mps"):
            maximum_position_difference = max(maximum_position_difference, _numeric_difference(old[field], new[field]))
    for old, new in zip(historical_updates, updates, strict=True):
        for field in ("event_id", "station_id", "estimator_variant", "failure_reason"):
            exact_mismatches += int(old[field] != str(new[field]))
        exact_mismatches += int(_bool(old["update_applied"]) != bool(new["update_applied"]))
    angular = _read_csv_gz(source_directory / "angular_errors.csv.gz")
    old_lookup = {(row["station_id"], row["estimator_variant"], int(row["frame_index"])): row
                  for row in angular if row["estimator_variant"] == method}
    current = [row for row in bearing_rows if row["estimator_variant"] == method]
    if len(old_lookup) != len(current):
        raise AssertionError("published bearing row count changed")
    maximum_angular_difference = 0.0
    for row in current:
        old = old_lookup[(row["station_id"], method, row["frame_index"])]
        exact_mismatches += int(_bool(old["valid"]) != bool(row["valid"]))
        exact_mismatches += int(old["invalid_reason"] != row["invalid_reason"])
        if row["valid"]:
            maximum_angular_difference = max(maximum_angular_difference,
                abs(float(old["geodesic_error_deg"]) - row["geodesic_error_deg"]))
    for field in ("accepted_update_count", "rejected_update_count", "final_valid", "final_confirmed",
                  "final_failure_reason", "batch_optimization_count", "batch_optimization_budget"):
        exact_mismatches += int(expected[field] != sequence[field])
    summary_differences = {}
    for field, tolerance in {"first_confirmation_time_s_absolute": TIME_TOLERANCE_S,
            "position_rmse_m_conditional": POSITION_TOLERANCE_M,
            "position_p95_m_conditional": POSITION_TOLERANCE_M, "availability_fraction": 0.0}.items():
        difference = _numeric_difference("" if expected[field] is None else str(expected[field]), sequence[field])
        summary_differences[field] = difference
        exact_mismatches += int(difference > tolerance)
    passed = (exact_mismatches == 0 and maximum_time_difference <= TIME_TOLERANCE_S
              and maximum_position_difference <= POSITION_TOLERANCE_M
              and maximum_angular_difference <= ANGULAR_TOLERANCE_DEG)
    result = {"passed": bool(passed), "exact_mismatch_count": int(exact_mismatches),
              "maximum_tracking_time_difference_s": maximum_time_difference,
              "maximum_tracking_numeric_difference": maximum_position_difference,
              "maximum_bearing_error_difference_deg": maximum_angular_difference,
              "summary_differences": summary_differences}
    if not passed:
        raise AssertionError(f"published result reproduction failed: {result}")
    return result


def _table_name(kind: str, method: str, variant: str) -> str:
    return f"{kind}_{method}_{variant}.csv.gz"


def restore_one(output: Path, index: int) -> dict[str, Any]:
    output = Path(output)
    diagnostic, source = _load_diagnostic(output), _load_study(SOURCE_STUDY)
    case = next((item for item in diagnostic["cases"] if item["index"] == index), None)
    if case is None:
        raise IndexError("index is not one of the eight frozen diagnostic cases")
    directory = _case_directory(output, case)
    directory.mkdir(parents=True, exist_ok=True)
    experiment_path = directory / "experiment.json"
    if experiment_path.exists():
        existing = json.loads(experiment_path.read_text())
        if existing.get("status") == "complete":
            for name, expected in existing["result_sha256"].items():
                if sha256(directory / name) != expected:
                    raise ValueError(f"diagnostic result SHA mismatch: {name}")
            return {"status": "skipped_verified", "index": index, "audio_restored": False}
    experiment = {"schema_version": SCHEMA_VERSION, "status": "running",
                  "source_run_id": case["source_run_id"], "diagnostic_id": case["diagnostic_id"],
                  "case": case, "configuration_frozen_before_processing": True}
    _write_json(experiment_path, experiment)
    started = time.perf_counter()
    try:
        bearing_rows, bearing_manifest, audio_restored = _restore_bearings(output, diagnostic, case, source)
        processing = diagnostic["processing"]
        stations = stations_from_experiment({"processing": processing})
        calibrations = calibration_from_experiment({"processing": processing})
        info = source["recordings"][case["trajectory"]]
        recording = load_gazebo_recording(ROOT / info["path"])
        trajectory, _ = _translated(recording, stations, info["reception_start_s"], case["distance_m"], case["trajectory"])
        tracker = processing["tracker"]
        history = ManoeuvreHistoryConfig(np.asarray(tracker["qc_m2_s3"], float),
            history_step_s=float(tracker["history_step_s"]), history_window_s=float(tracker["history_window_s"]),
            maximum_range_m=float(tracker["maximum_range_m"]),
            maximum_transport_delay_s=float(tracker["maximum_transport_delay_s"]))
        recovery = InitializationRecoveryConfig(**tracker["recovery"])
        classifier = _phase_classifier(info["evaluation_phases"])
        summaries, reproduction = [], {}
        output_names = ["bearing_records.csv.gz", "bearing_manifest.json"]
        geometry_epochs: list[float] | None = None
        for method in METHODS:
            for variant in VARIANTS:
                measurements = _make_measurements(bearing_rows, calibrations, method, variant, int(tracker["frame_stride"]))
                tracks, sequence, updates, diagnostics = _run_tracker_diagnostic(
                    stations, trajectory, measurements, method, variant, history, recovery, classifier)
                absolute = sequence["first_confirmation_time_s_absolute"]
                sequence["first_confirmation_time_s_relative"] = (
                    absolute - float(info["reception_start_s"]) if absolute is not None else None)
                sequence.update({"source_run_id": case["source_run_id"], "diagnostic_id": case["diagnostic_id"],
                    "index": index, "trajectory": case["trajectory"], "experiment": case["experiment"],
                    "distance_m": case["distance_m"], "snr_db": case["snr_db"], "replicate": case["replicate"]})
                summaries.append(sequence)
                for kind, rows in {"tracking": tracks, "updates": updates, **diagnostics}.items():
                    name = _table_name(kind, method, variant)
                    if rows:
                        _write_csv_gz(directory / name, rows)
                    else:
                        columns = {"updates": ("event_id", "update_applied", "failure_reason"),
                                   "batch_fits": ("processing_time_s", "reason"),
                                   "candidates": ("processing_time_s", "reason"),
                                   "hypotheses": ("processing_time_s", "action", "reason"),
                                   "lifecycle": ("processing_time_s", "action", "reason")}[kind]
                        _write_csv_gz(directory / name, [], columns)
                    output_names.append(name)
                if variant == "original":
                    source_directory = _run_directory(SOURCE_STUDY, source["runs"][index], case["source_run_id"])
                    reproduction[method] = _verify_original_reproduction(
                        source_directory, method, bearing_rows, tracks, updates, sequence)
                if geometry_epochs is None:
                    geometry_epochs = [row["processing_time_s"] for row in tracks]
        geometry = _geometry_rows(stations, trajectory, geometry_epochs or [])
        for row in geometry:
            row.update({"source_run_id": case["source_run_id"], "index": index,
                        "trajectory": case["trajectory"], "distance_m": case["distance_m"]})
        _write_csv_gz(directory / "geometry_sensitivity.csv.gz", geometry)
        output_names.append("geometry_sensitivity.csv.gz")
        case_summary = {"schema_version": SCHEMA_VERSION, "source_run_id": case["source_run_id"],
            "diagnostic_id": case["diagnostic_id"], "case": case, "bearing_manifest": bearing_manifest,
            "variants": summaries, "original_reproduction": reproduction,
            "wall_runtime_s": time.perf_counter() - started, "audio_restored_this_invocation": audio_restored}
        _write_json(directory / "summary.json", case_summary)
        output_names.append("summary.json")
        experiment["status"] = "complete"
        experiment["audio_restoration_count"] = bearing_manifest["audio_restoration_count"]
        experiment["result_sha256"] = {name: sha256(directory / name) for name in sorted(output_names)}
        _write_json(experiment_path, experiment)
        return {"status": "completed", "index": index, "audio_restored": audio_restored,
                "source_run_id": case["source_run_id"]}
    except Exception as error:
        experiment["status"] = "failed"
        experiment["error"] = f"{type(error).__name__}: {error}"
        _write_json(experiment_path, experiment)
        raise


def restore_all(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    diagnostic = _load_diagnostic(output)
    completed = skipped = audio_restorations = 0
    for case in diagnostic["cases"]:
        result = subprocess.run([sys.executable, "-m", "analysis.localization_error_attribution", "restore-one",
            "--index", str(case["index"]), "--output", str(Path(output).resolve())],
            cwd=ROOT, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        completed += payload["status"] == "completed"
        skipped += payload["status"] != "completed"
        audio_restorations += int(payload["audio_restored"])
        print(f"[{completed + skipped}/{len(diagnostic['cases'])}] {payload['status']} index={case['index']}", flush=True)
    return {"completed_now": completed, "skipped_verified": skipped,
            "audio_restorations_now": audio_restorations, "total_cases": len(diagnostic["cases"])}


def _bearing_station_summary(case_dir: Path, case: dict[str, Any]) -> list[dict[str, Any]]:
    records, result = _measurement_rows(case_dir / "bearing_records.csv.gz"), []
    for method in METHODS:
        for station in sorted({row["station_id"] for row in records}):
            subset = [row for row in records if row["estimator_variant"] == method and row["station_id"] == station]
            valid = np.asarray([row["geodesic_error_deg"] for row in subset if row["valid"]])
            for variant in VARIANTS:
                values = np.zeros(len(subset)) if variant == "ideal_bearing" else valid
                result.append({"source_run_id": case["source_run_id"], "index": case["index"],
                    "trajectory": case["trajectory"], "experiment": case["experiment"],
                    "distance_m": case["distance_m"], "snr_db": case["snr_db"], "replicate": case["replicate"],
                    "estimator_variant": method, "diagnostic_variant": variant, "station_id": station,
                    "frame_count": len(subset), "valid_frame_count": len(subset) if variant == "ideal_bearing" else len(valid),
                    "angular_rmse_deg": float(np.sqrt(np.mean(values**2))),
                    "angular_p95_deg": float(np.percentile(values, 95))})
    return result


def aggregate(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output, diagnostic = Path(output), _load_diagnostic(output)
    variants, stations, geometry, reproductions = [], [], [], []
    audio_restoration_count = 0
    for case in diagnostic["cases"]:
        directory = _case_directory(output, case)
        experiment = json.loads((directory / "experiment.json").read_text())
        if experiment.get("status") != "complete":
            raise ValueError(f"diagnostic case incomplete: {case['source_run_id']}")
        for name, expected in experiment["result_sha256"].items():
            if sha256(directory / name) != expected:
                raise ValueError(f"diagnostic artifact SHA mismatch: {name}")
        summary = json.loads((directory / "summary.json").read_text())
        variants.extend(summary["variants"])
        stations.extend(_bearing_station_summary(directory, case))
        geometry.extend(_read_csv_gz(directory / "geometry_sensitivity.csv.gz"))
        audio_restoration_count += int(experiment["audio_restoration_count"])
        for method, comparison in summary["original_reproduction"].items():
            reproductions.append({"source_run_id": case["source_run_id"], "index": case["index"],
                                  "estimator_variant": method, **comparison})
    _write_csv(output / "variant_summary.csv", variants, list(variants[0]))
    _write_csv(output / "bearing_station_summary.csv", stations, list(stations[0]))
    _write_csv(output / "geometry_summary.csv", geometry, list(geometry[0]))
    _write_csv(output / "reproduction_summary.csv", reproductions, list(reproductions[0]))
    result = {"schema_version": SCHEMA_VERSION, "case_count": len(diagnostic["cases"]),
        "method_variant_count": len(variants), "audio_restoration_count": audio_restoration_count,
        "all_original_reproductions_passed": all(row["passed"] for row in reproductions),
        "source_manifest_sha256": diagnostic["source_manifest_sha256"],
        "source_completed_run_ids_sha256": SOURCE_RUN_IDS_SHA256,
        "tables": {name: sha256(output / name) for name in (
            "variant_summary.csv", "bearing_station_summary.csv", "geometry_summary.csv", "reproduction_summary.csv")}}
    _write_json(output / "study_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "restore-one", "restore-all", "aggregate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "init":
        result = initialize(args.output)
        payload = {"case_count": result["case_count"], "output": str(args.output)}
    elif args.action == "restore-one":
        if args.index is None:
            parser.error("restore-one requires --index")
        payload = restore_one(args.output, args.index)
    elif args.action == "restore-all":
        payload = restore_all(args.output)
    else:
        payload = aggregate(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
