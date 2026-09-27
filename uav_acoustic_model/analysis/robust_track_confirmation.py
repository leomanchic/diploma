"""Frozen 12-stream paired check of three-station track confirmation.

The archived 120-run study is read only. This runner synthesizes each new
continuous stream once and reuses its saved bearings for both tracker variants.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import chi2

from analysis.localization_error_attribution import (
    BEARING_COLUMNS, _measurement_rows, _make_measurements, _read_csv_gz,
    _serialize_bearing, _standard_noise_sha256, _write_csv, _write_csv_gz,
    _write_json,
)
from estimators.retarded_ekf_manoeuvre import CausalManoeuvreRetardedTimeEKF, ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.bearing_events import bearing_event_id
from simulation.gazebo_offline import load_gazebo_recording
from simulation.multistation_audio import synthesize_multistation_audio
from validation.gazebo_experiment import (
    calibration_from_experiment, canonical_sha256, code_sha256, sha256,
    stations_from_experiment,
)
from validation.localization_range_study import (
    DEFAULT_OUTPUT as SOURCE, REFERENCE_DISTANCE_M, ROOT, ROOT_SEED,
    _load_study, _translated,
)
from validation.three_station_audio_tracking_study import extract_audio_bearing_records

SCHEMA_VERSION = 1
PROTOCOL = ROOT / "ROBUST_TRACK_CONFIRMATION_PROTOCOL.md"
RUNNER = Path(__file__).resolve()
DEFAULT_OUTPUT = ROOT / "results" / "robust_track_confirmation"
METHODS = ("all_6_equal_gcc_wls", "equal_weight_srp_phat")
VARIANTS = ("baseline", "three_station_confirmation")
TRAJECTORIES = ("single_turn", "opposite_turns")
DISTANCES_M = (200, 700, 1000)
NOISE_SEEDS = (
    (2279586615842805297, 6929539839069591369),
    (2191817683501416574, 6674498936964276922),
)
SOURCE_MANIFEST_SHA256 = "00f3e2e259885d34abd9e2247e1aa1ed0f74c201ad63d4981dfd6fe8b5d194e6"
SOURCE_CODE_SHA256 = "344ac04bf9a4c74aa7e447c2533f746c99bc623da5227adc346c3833efddcd0a"
CHI3 = float(chi2.ppf(0.95, 3))
CHI6 = float(chi2.ppf(0.95, 6))


def _specs(source: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for ti, trajectory in enumerate(TRAJECTORIES):
        for distance in DISTANCES_M:
            for replicate in (0, 1):
                spec = {"index": len(specs), "trajectory": trajectory,
                        "distance_m": distance, "replicate": replicate,
                        "snr_ref_db": 10.0, "noise_seed": NOISE_SEEDS[ti][replicate],
                        "source_seed": int(source["source_banks"][trajectory]["seed"]),
                        "noise_model": "fixed_reference_snr", "geometric_attenuation": True}
                identity = {
                    "schema_version": SCHEMA_VERSION, "protocol_sha256": sha256(PROTOCOL),
                    "processing_code_sha256": code_sha256(), "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                    "recording_sha256": source["recordings"][trajectory]["state_sha256"],
                    "source_signal_sha256": source["source_banks"][trajectory]["signal_sha256"],
                    "translation_enu_m": source["geometry"]["translations"][f"{trajectory}:{distance}"]["translation_enu_m"],
                    "spec": spec,
                }
                spec["run_id"] = "rconf-" + canonical_sha256(identity)[:24]
                specs.append(spec)
    return specs


def initialize(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"robust confirmation output already exists: {output}")
    source = _load_study(SOURCE)
    if sha256(SOURCE / "study_manifest.json") != SOURCE_MANIFEST_SHA256 or source["code_sha256"] != SOURCE_CODE_SHA256:
        raise ValueError("archived range-study provenance changed")
    if int(source["processing"]["tracker"]["recovery"]["confirmation_station_count"]) != 2:
        raise ValueError("baseline confirmation rule changed")
    specs = _specs(source)
    if len(specs) != 12 or len({spec["run_id"] for spec in specs}) != 12:
        raise AssertionError("evaluation matrix is not 12 unique streams")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION, "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(RUNNER), "processing_code_sha256": code_sha256(),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_code_sha256": SOURCE_CODE_SHA256,
        "source_processing_sha256": source["processing_sha256"],
        "processing": source["processing"], "recordings": source["recordings"],
        "source_banks": source["source_banks"], "geometry": source["geometry"],
        "noise_seed_root": 20260927, "methods": list(METHODS),
        "variants": {"baseline": {"confirmation_station_count": 2},
                     "three_station_confirmation": {"confirmation_station_count": 3}},
        "specs": specs, "audio_stream_limit": 12, "tracker_replay_limit": 48,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
    }
    manifest["processing_sha256"] = canonical_sha256(manifest["processing"])
    _write_json(output / "evaluation_manifest.json", manifest)
    return manifest


def _load(output: Path) -> dict[str, Any]:
    manifest = json.loads((Path(output) / "evaluation_manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported evaluation schema")
    checks = (
        (sha256(PROTOCOL), manifest["protocol_sha256"], "protocol"),
        (sha256(RUNNER), manifest["runner_sha256"], "runner"),
        (code_sha256(), manifest["processing_code_sha256"], "code"),
        (sha256(SOURCE / "study_manifest.json"), SOURCE_MANIFEST_SHA256, "archived study"),
        (canonical_sha256(manifest["processing"]), manifest["processing_sha256"], "processing"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(f"{label} SHA mismatch")
    _load_study(SOURCE)
    if manifest["specs"] != _specs({"recordings": manifest["recordings"],
                                    "source_banks": manifest["source_banks"],
                                    "geometry": manifest["geometry"]}):
        raise ValueError("frozen evaluation matrix changed")
    return manifest


def _directory(output: Path, spec: dict[str, Any]) -> Path:
    return Path(output) / "runs" / f"{spec['index']:02d}_{spec['trajectory']}_d{spec['distance_m']}_r{spec['replicate']}_{spec['run_id']}"


def _scenario(manifest: dict[str, Any], spec: dict[str, Any]):
    processing = manifest["processing"]
    stations = stations_from_experiment({"processing": processing})
    info = manifest["recordings"][spec["trajectory"]]
    recording = load_gazebo_recording(ROOT / info["path"])
    if recording.csv_sha256 != info["state_sha256"]:
        raise ValueError("Gazebo state changed")
    trajectory, offset = _translated(recording, stations, info["reception_start_s"],
                                     spec["distance_m"], spec["trajectory"])
    return stations, trajectory, np.asarray(offset, float), info


def _restore_bearings(output: Path, manifest: dict[str, Any], spec: dict[str, Any], directory: Path):
    saved = directory / "bearing_records.csv.gz"
    metadata_path = directory / "bearing_manifest.json"
    if saved.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["run_id"] != spec["run_id"] or sha256(saved) != metadata["bearing_records_sha256"]:
            raise ValueError("saved bearing cache mismatch")
        return _measurement_rows(saved), metadata, False
    stations, trajectory, offset, info = _scenario(manifest, spec)
    bank = manifest["source_banks"][spec["trajectory"]]
    bank_path = SOURCE / bank["path"]
    if sha256(bank_path) != bank["file_sha256"]:
        raise ValueError("source bank SHA mismatch")
    source_signal = np.load(bank_path, allow_pickle=False)
    if hashlib.sha256(source_signal.tobytes()).hexdigest() != bank["signal_sha256"]:
        raise ValueError("source signal SHA mismatch")
    processing = manifest["processing"]
    audio, frontend = processing["audio"], processing["frontend"]
    started = time.perf_counter()
    stream = synthesize_multistation_audio(
        stations, trajectory, duration_s=float(info["duration_s"]),
        reception_start_time_s=float(info["reception_start_s"]),
        sampling_rate_hz=float(audio["sampling_rate_hz"]), sound_speed=float(audio["sound_speed_mps"]),
        signal_model=str(audio["signal_model"]), snr_db=10.0, seed=ROOT_SEED,
        chunk_size_samples=int(audio["chunk_size_samples"]), fir_length=int(audio["fir_length"]),
        geometric_attenuation=True, maximum_emitted_frequency_hz=float(audio["source_maximum_frequency_hz"]),
        external_source_signal=source_signal, external_source_start_time_s=float(bank["start_time_s"]),
        noise_model="fixed_reference_snr", reference_distance_m=REFERENCE_DISTANCE_M,
        source_seed=int(spec["source_seed"]), noise_seed=int(spec["noise_seed"]),
        reference_source_rms=1.0)
    audio_runtime = time.perf_counter() - started
    extracted, frontend_runtime = extract_audio_bearing_records(
        stream, stations, trajectory, split="robust_confirmation_evaluation",
        configuration_index=int(spec["index"]), sequence_index=int(spec["replicate"]),
        frame_length=int(frontend["frame_length"]), hop_length=int(frontend["hop_length"]),
        modeled_processing_delay_s=float(frontend["modeled_processing_delay_s"]),
        station_delivery_delay_s=frontend["station_delivery_delay_s"],
        sequence_id=spec["run_id"])
    serialized = [_serialize_bearing(row, spec["run_id"]) for row in extracted]
    _write_csv_gz(saved, serialized, BEARING_COLUMNS)
    metadata = {"schema_version": SCHEMA_VERSION, "run_id": spec["run_id"],
                "audio_restoration_count": 1, "bearing_record_count": len(serialized),
                "bearing_records_sha256": sha256(saved), "recording_state_sha256": info["state_sha256"],
                "source_signal_sha256": bank["signal_sha256"],
                "standardized_noise_sha256": _standard_noise_sha256(stream),
                "translation_enu_m": offset.tolist(), "audio_synthesis_runtime_s": audio_runtime,
                "bearing_frontend_runtime_s": frontend_runtime,
                "actual_station_snr_db": {item.station_id: item.effective_snr_db for item in stream.stations}}
    _write_json(metadata_path, metadata)
    return _measurement_rows(saved), metadata, True


def nominal_state_uncertainty(covariance: np.ndarray) -> dict[str, Any]:
    """Return the literal covariance blocks and nominal Gaussian 95% axes."""
    matrix = np.asarray(covariance, float)
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise ValueError("state covariance must be finite 6x6")
    matrix = 0.5 * (matrix + matrix.T)
    np.linalg.cholesky(matrix)
    result: dict[str, Any] = {}
    for name, block in (("position", matrix[:3, :3]), ("velocity", matrix[3:, 3:])):
        eigenvalues, eigenvectors = np.linalg.eigh(block)
        if np.any(eigenvalues <= 0):
            raise ValueError(f"{name} covariance must be positive definite")
        result[f"{name}_covariance"] = block.tolist()
        result[f"{name}_nominal_95_axes"] = (np.sqrt(CHI3 * eigenvalues)).tolist()
        result[f"{name}_nominal_95_axis_directions"] = eigenvectors.tolist()
        result[f"{name}_sigma_rss"] = float(np.sqrt(np.trace(block)))
    return result


def _json_rows(items) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        row = asdict(item)
        for key, value in row.items():
            if isinstance(value, (tuple, list, dict)):
                row[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        rows.append(row)
    return rows


def _run_tracker(stations, trajectory, measurements, method: str, variant: str,
                 history: ManoeuvreHistoryConfig, recovery: InitializationRecoveryConfig,
                 reception_start_s: float):
    estimator = CausalManoeuvreRetardedTimeEKF(
        stations, measurements, estimator_variant=method,
        history_config=history, recovery_config=recovery)
    epochs = sorted({item.available_timestamp_s for item in measurements})
    started = time.perf_counter()
    publications = [estimator.advance_to(epoch) for epoch in epochs]
    runtime = time.perf_counter() - started
    station_centroid = np.mean([item.position_world_m for item in stations], axis=0)
    tracks, updates = [], []
    last_accepted = None
    for publication in publications:
        epoch = float(publication.processing_time_s)
        for item in publication.update_diagnostics:
            residual = item.residual_tangent_rad
            innovation = item.innovation_covariance_tangent_rad2
            if residual is None and not item.residual_unavailable_reason:
                raise ValueError("missing angular residual has no reason")
            if np.isfinite(item.pre_update_nis):
                if residual is None or innovation is None:
                    raise ValueError("finite NIS lacks its actual residual and innovation covariance")
                vector = np.asarray(residual, float)
                s = np.asarray(innovation, float)
                calculated = float(vector @ np.linalg.solve(s, vector))
                if not np.isclose(calculated, item.pre_update_nis, rtol=1e-10, atol=1e-10):
                    raise ValueError("saved residual disagrees with pre-update NIS")
            if item.update_applied:
                last_accepted = float(item.processing_time_s)
            updates.append({
                "event_id": item.event_id, "processing_time_s": item.processing_time_s,
                "reception_time_s": item.reception_time_s, "emission_time_s_estimated": item.emission_time_s,
                "update_applied": item.update_applied, "failure_reason": item.failure_reason or "",
                "pre_update_nis": item.pre_update_nis,
                "residual_tangent_rad_json": json.dumps(residual) if residual is not None else "",
                "residual_tangent_norm_rad": float(np.linalg.norm(residual)) if residual is not None else "",
                "residual_unavailable_reason": item.residual_unavailable_reason or "",
                "innovation_covariance_tangent_rad2_json": json.dumps(innovation) if innovation is not None else "",
                "history_node_count": item.history_node_count,
                "history_memory_bytes": item.history_memory_bytes,
                "truth_used_by_tracker": False,
            })
        truth = np.concatenate((trajectory.q(epoch), trajectory.v(epoch)))
        valid = bool(publication.valid and publication.state is not None)
        row: dict[str, Any] = {
            "processing_time_s": epoch, "confirmed": bool(publication.confirmed), "valid": valid,
            "status": publication.status, "failure_reason": publication.failure_reason or "",
            "position_enu_m_json": "", "velocity_enu_mps_json": "",
            "position_covariance_m2_json": "", "velocity_covariance_m2ps2_json": "",
            "position_nominal_95_axes_m_json": "", "position_nominal_95_axis_directions_json": "",
            "velocity_nominal_95_axes_mps_json": "", "velocity_nominal_95_axis_directions_json": "",
            "position_sigma_rss_m": "", "velocity_sigma_rss_mps": "",
            "position_error_m": "", "velocity_error_mps": "", "radial_error_m": "", "transverse_error_m": "",
            "position_nees": "", "state_nees": "", "position_nominal_95_covered": "", "state_nominal_95_covered": "",
            "last_accepted_update_time_s": last_accepted if last_accepted is not None else "",
            "time_since_last_accepted_update_s": epoch - last_accepted if last_accepted is not None else "",
            "batch_optimization_count": publication.batch_optimization_count,
            "history_memory_bytes": publication.history_memory_bytes,
            "generation": publication.generation, "reset_count": publication.reset_count,
            "truth_used_by_tracker": False,
        }
        if valid:
            state = np.asarray(publication.state.vector, float)
            covariance = np.asarray(publication.covariance_state, float)
            nominal = nominal_state_uncertainty(covariance)
            delta = state - truth
            radial = truth[:3] - station_centroid
            radial /= np.linalg.norm(radial)
            radial_error = float(delta[:3] @ radial)
            position_nees = float(delta[:3] @ np.linalg.solve(covariance[:3, :3], delta[:3]))
            state_nees = float(delta @ np.linalg.solve(covariance, delta))
            row.update({
                "position_enu_m_json": json.dumps(state[:3].tolist()),
                "velocity_enu_mps_json": json.dumps(state[3:].tolist()),
                "position_covariance_m2_json": json.dumps(nominal["position_covariance"]),
                "velocity_covariance_m2ps2_json": json.dumps(nominal["velocity_covariance"]),
                "position_nominal_95_axes_m_json": json.dumps(nominal["position_nominal_95_axes"]),
                "position_nominal_95_axis_directions_json": json.dumps(nominal["position_nominal_95_axis_directions"]),
                "velocity_nominal_95_axes_mps_json": json.dumps(nominal["velocity_nominal_95_axes"]),
                "velocity_nominal_95_axis_directions_json": json.dumps(nominal["velocity_nominal_95_axis_directions"]),
                "position_sigma_rss_m": nominal["position_sigma_rss"],
                "velocity_sigma_rss_mps": nominal["velocity_sigma_rss"],
                "position_error_m": float(np.linalg.norm(delta[:3])),
                "velocity_error_mps": float(np.linalg.norm(delta[3:])),
                "radial_error_m": radial_error,
                "transverse_error_m": float(np.linalg.norm(delta[:3] - radial_error * radial)),
                "position_nees": position_nees, "state_nees": state_nees,
                "position_nominal_95_covered": position_nees <= CHI3,
                "state_nominal_95_covered": state_nees <= CHI6,
            })
        tracks.append(row)
    final = publications[-1]
    confirmed_rows = [row for row in tracks if row["confirmed"] and row["valid"]]
    first = confirmed_rows[0] if confirmed_rows else None
    valid_rows = [row for row in tracks if row["valid"]]
    errors = np.asarray([row["position_error_m"] for row in valid_rows], float)
    reasons = Counter(row["failure_reason"] or "unspecified" for row in updates if not row["update_applied"])
    fit_rows = _json_rows(estimator.batch_fit_diagnostics)
    denied = [row for row in fit_rows if row["reason"] == "computational_budget_exceeded_before_fit"]
    lifecycle = _json_rows(final.lifecycle_diagnostics)
    if denied:
        lifetime = [row for row in lifecycle if row["action"] == "computational_budget_exceeded"]
        if not lifetime or float(lifetime[0]["processing_time_s"]) != float(denied[0]["processing_time_s"]):
            raise ValueError("budget exhaustion fit/lifecycle time mismatch")
    uses = _json_rows(final.event_uses)
    for identity in {row["event_id"] for row in uses}:
        roles = [row["role"] for row in uses if row["event_id"] == identity]
        if "initialization" in roles and "update" in roles:
            raise ValueError("initialization event reused as EKF update")
    summary = {
        "ever_confirmed": first is not None,
        "first_confirmation_time_s_absolute": first["processing_time_s"] if first else None,
        "first_confirmation_time_s_relative": first["processing_time_s"] - reception_start_s if first else None,
        "first_confirmation_position_error_m": first["position_error_m"] if first else None,
        "first_confirmation_velocity_error_mps": first["velocity_error_mps"] if first else None,
        "first_confirmation_radial_error_m": first["radial_error_m"] if first else None,
        "first_confirmation_transverse_error_m": first["transverse_error_m"] if first else None,
        "first_confirmation_position_nees": first["position_nees"] if first else None,
        "first_confirmation_velocity_sigma_rss_mps": first["velocity_sigma_rss_mps"] if first else None,
        "final_confirmed": bool(final.confirmed), "final_status": final.status,
        "final_failure_reason": final.failure_reason or "", "publication_count": len(tracks),
        "confirmed_publication_count": sum(row["confirmed"] for row in tracks),
        "valid_publication_count": len(valid_rows), "availability_fraction": len(valid_rows) / len(tracks),
        "confirmed_availability_fraction": sum(row["confirmed"] for row in tracks) / len(tracks),
        "position_rmse_m_conditional": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "position_p95_m_conditional": float(np.percentile(errors, 95)) if len(errors) else None,
        "accepted_update_count": sum(row["update_applied"] for row in updates),
        "rejected_update_count": sum(not row["update_applied"] for row in updates),
        "update_rejection_reasons": dict(reasons), "batch_optimization_count": final.batch_optimization_count,
        "batch_optimization_budget": final.batch_optimization_budget,
        "first_budget_exhaustion_time_s": float(denied[0]["processing_time_s"]) if denied else None,
        "tracker_runtime_s": runtime,
        "maximum_history_memory_bytes": estimator.maximum_history_memory_bytes,
        "position_nees_median": float(np.median([row["position_nees"] for row in valid_rows])) if valid_rows else None,
        "position_nominal_95_coverage": float(np.mean([row["position_nominal_95_covered"] for row in valid_rows])) if valid_rows else None,
        "state_nominal_95_coverage": float(np.mean([row["state_nominal_95_covered"] for row in valid_rows])) if valid_rows else None,
    }
    diagnostics = {"batch_fits": fit_rows, "lifecycle": lifecycle,
                   "hypotheses": _json_rows(final.hypothesis_diagnostics), "event_uses": uses}
    return tracks, updates, diagnostics, summary


def run_one(output: Path, index: int) -> dict[str, Any]:
    output = Path(output)
    manifest = _load(output)
    if not 0 <= index < len(manifest["specs"]):
        raise IndexError("index outside frozen 12-stream matrix")
    spec = manifest["specs"][index]
    directory = _directory(output, spec)
    directory.mkdir(parents=True, exist_ok=True)
    experiment_path = directory / "experiment.json"
    if experiment_path.exists():
        experiment = json.loads(experiment_path.read_text())
        if experiment.get("run_id") != spec["run_id"]:
            raise ValueError("run directory identity mismatch")
        if experiment.get("status") == "complete":
            for name, digest in experiment["result_sha256"].items():
                if sha256(directory / name) != digest:
                    raise ValueError(f"result SHA mismatch: {name}")
            return {"status": "skipped_verified", "index": index, "audio_restored": False}
    _write_json(experiment_path, {"schema_version": SCHEMA_VERSION,
                                   "status": "running", "run_id": spec["run_id"], "spec": spec})
    rows, bearing_metadata, restored = _restore_bearings(output, manifest, spec, directory)
    processing = manifest["processing"]
    stations, trajectory, _, info = _scenario(manifest, spec)
    calibrations = calibration_from_experiment({"processing": processing})
    tracker = processing["tracker"]
    history = ManoeuvreHistoryConfig(
        np.asarray(tracker["qc_m2_s3"], float),
        history_step_s=float(tracker["history_step_s"]),
        history_window_s=float(tracker["history_window_s"]),
        maximum_range_m=float(tracker["maximum_range_m"]),
        maximum_transport_delay_s=float(tracker["maximum_transport_delay_s"]))
    artifacts = ["bearing_records.csv.gz", "bearing_manifest.json"]
    summaries = []
    for method in METHODS:
        measurements = _make_measurements(rows, calibrations, method, "original", int(tracker["frame_stride"]))
        for variant in VARIANTS:
            config = dict(tracker["recovery"])
            config["confirmation_station_count"] = manifest["variants"][variant]["confirmation_station_count"]
            recovery = InitializationRecoveryConfig(**config)
            tracks, updates, diagnostics, summary = _run_tracker(
                stations, trajectory, measurements, method, variant,
                history, recovery, float(info["reception_start_s"]))
            summaries.append({"run_id": spec["run_id"], "index": index,
                              "trajectory": spec["trajectory"], "distance_m": spec["distance_m"],
                              "replicate": spec["replicate"], "estimator_variant": method,
                              "confirmation_variant": variant, **summary})
            for kind, data in {"tracking": tracks, "updates": updates, **diagnostics}.items():
                name = f"{kind}_{method}_{variant}.csv.gz"
                columns = {
                    "tracking": ("processing_time_s", "confirmed", "valid", "failure_reason"),
                    "updates": ("event_id", "update_applied", "failure_reason", "pre_update_nis",
                                "residual_tangent_norm_rad", "residual_unavailable_reason"),
                    "batch_fits": ("processing_time_s", "reason"),
                    "lifecycle": ("processing_time_s", "action", "reason"),
                    "hypotheses": ("processing_time_s", "action", "reason"),
                    "event_uses": ("event_id", "role"),
                }[kind]
                _write_csv_gz(directory / name, data, columns=columns if not data else None)
                artifacts.append(name)
    summary_path = directory / "summary.json"
    _write_json(summary_path, {
        "schema_version": SCHEMA_VERSION, "run_id": spec["run_id"], "spec": spec,
        "bearing_manifest_sha256": sha256(directory / "bearing_manifest.json"),
        "audio_restoration_count": bearing_metadata["audio_restoration_count"],
        "method_variants": summaries,
    })
    artifacts.append("summary.json")
    _write_json(experiment_path, {
        "schema_version": SCHEMA_VERSION, "status": "complete", "run_id": spec["run_id"],
        "spec": spec, "result_sha256": {name: sha256(directory / name) for name in artifacts},
    })
    return {"status": "completed", "index": index, "run_id": spec["run_id"], "audio_restored": restored}


def run_all(output: Path) -> dict[str, Any]:
    manifest = _load(output)
    counts = Counter()
    for spec in manifest["specs"]:
        process = subprocess.run(
            [sys.executable, "-m", "analysis.robust_track_confirmation", "run-one",
             "--output", str(output), "--index", str(spec["index"])],
            cwd=ROOT, capture_output=True, text=True, check=True)
        result = json.loads(process.stdout.strip().splitlines()[-1])
        counts[result["status"]] += 1
        counts["audio_restored_now"] += int(result["audio_restored"])
        print(f"[{spec['index'] + 1}/12] {result['status']} {spec['run_id']}", flush=True)
    return {"completed_now": counts["completed"], "skipped_verified": counts["skipped_verified"],
            "audio_restored_now": counts["audio_restored_now"]}


def aggregate(output: Path) -> dict[str, Any]:
    output = Path(output)
    manifest = _load(output)
    rows = []
    for spec in manifest["specs"]:
        directory = _directory(output, spec)
        experiment = json.loads((directory / "experiment.json").read_text())
        if experiment.get("status") != "complete" or experiment["run_id"] != spec["run_id"]:
            raise ValueError(f"incomplete evaluation stream {spec['index']}")
        for name, digest in experiment["result_sha256"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"result SHA mismatch: {name}")
        rows.extend(json.loads((directory / "summary.json").read_text())["method_variants"])
    if len(rows) != 48:
        raise AssertionError("expected exactly 48 tracker replays")
    _write_csv(output / "run_summary.csv", rows, rows[0].keys())
    per_method = []
    for method in METHODS:
        for variant in VARIANTS:
            subset = [row for row in rows if row["estimator_variant"] == method and row["confirmation_variant"] == variant]
            confirmed = [row for row in subset if row["ever_confirmed"]]
            publications = sum(row["publication_count"] for row in subset)
            valid = sum(row["valid_publication_count"] for row in subset)
            confirmed_publications = sum(row["confirmed_publication_count"] for row in subset)
            per_method.append({
                "estimator_variant": method, "confirmation_variant": variant,
                "audio_stream_count": 12, "confirmed_stream_count": len(confirmed),
                "confirmation_fraction": len(confirmed) / 12,
                "median_first_confirmation_time_s": float(np.median([row["first_confirmation_time_s_relative"] for row in confirmed])) if confirmed else None,
                "median_first_confirmation_position_error_m": float(np.median([row["first_confirmation_position_error_m"] for row in confirmed])) if confirmed else None,
                "median_first_confirmation_velocity_error_mps": float(np.median([row["first_confirmation_velocity_error_mps"] for row in confirmed])) if confirmed else None,
                "severe_confirmation_over_50m_count_all_12": sum(row["first_confirmation_position_error_m"] is not None and row["first_confirmation_position_error_m"] > 50 for row in subset),
                "valid_publication_count": valid, "publication_count": publications,
                "availability_fraction": valid / publications,
                "confirmed_availability_fraction": confirmed_publications / publications,
                "accepted_update_count": sum(row["accepted_update_count"] for row in subset),
                "rejected_update_count": sum(row["rejected_update_count"] for row in subset),
                "batch_optimization_count_total": sum(row["batch_optimization_count"] for row in subset),
                "tracker_runtime_s_total": sum(row["tracker_runtime_s"] for row in subset),
                "maximum_history_memory_bytes": max(row["maximum_history_memory_bytes"] for row in subset),
            })
    _write_csv(output / "method_summary.csv", per_method, per_method[0].keys())
    baseline = {(row["run_id"], row["estimator_variant"]): row for row in rows if row["confirmation_variant"] == "baseline"}
    improved = {(row["run_id"], row["estimator_variant"]): row for row in rows if row["confirmation_variant"] == "three_station_confirmation"}
    pairs = []
    for key, left in baseline.items():
        right = improved[key]
        pairs.append({"run_id": key[0], "estimator_variant": key[1],
                      "baseline_confirmed": left["ever_confirmed"], "new_confirmed": right["ever_confirmed"],
                      "baseline_first_position_error_m": left["first_confirmation_position_error_m"],
                      "new_first_position_error_m": right["first_confirmation_position_error_m"],
                      "baseline_availability": left["availability_fraction"],
                      "new_availability": right["availability_fraction"],
                      "baseline_first_time_s": left["first_confirmation_time_s_relative"],
                      "new_first_time_s": right["first_confirmation_time_s_relative"]})
    _write_csv(output / "paired_summary.csv", pairs, pairs[0].keys())
    severe_left = sum(row["first_confirmation_position_error_m"] is not None and row["first_confirmation_position_error_m"] > 50 for row in baseline.values())
    severe_right = sum(row["first_confirmation_position_error_m"] is not None and row["first_confirmation_position_error_m"] > 50 for row in improved.values())
    confirmed_left = sum(row["ever_confirmed"] for row in baseline.values())
    confirmed_right = sum(row["ever_confirmed"] for row in improved.values())
    availability_left = sum(row["confirmed_publication_count"] for row in baseline.values()) / sum(row["publication_count"] for row in baseline.values())
    availability_right = sum(row["confirmed_publication_count"] for row in improved.values()) / sum(row["publication_count"] for row in improved.values())
    shared_times = [(baseline[key]["first_confirmation_time_s_relative"], improved[key]["first_confirmation_time_s_relative"])
                    for key in baseline if baseline[key]["ever_confirmed"] and improved[key]["ever_confirmed"]]
    median_time_change = float(np.median([right - left for left, right in shared_times])) if shared_times else None
    if (severe_left - severe_right >= 2 and confirmed_left - confirmed_right <= 2
            and availability_left - availability_right <= 0.10
            and median_time_change is not None and median_time_change <= 1.0):
        verdict = "improvement_confirmed_in_limited_protocol"
    elif severe_right < severe_left:
        verdict = "tradeoff_detected"
    else:
        verdict = "improvement_not_confirmed"
    result = {"schema_version": SCHEMA_VERSION, "audio_stream_count": 12,
              "tracker_replay_count": 48, "paired_method_stream_count": 24,
              "baseline_severe_confirmation_count": severe_left,
              "new_severe_confirmation_count": severe_right,
              "baseline_confirmed_count": confirmed_left, "new_confirmed_count": confirmed_right,
              "baseline_availability": availability_left, "new_availability": availability_right,
              "shared_confirmation_median_time_change_s": median_time_change,
              "verdict": verdict, "limitations": "same two motion types and source banks; 12 new noise streams only",
              "tables": {name: sha256(output / name) for name in ("run_summary.csv", "method_summary.csv", "paired_summary.csv")}}
    _write_json(output / "evaluation_summary.json", result)
    return result


def correct_archived_diagnostics(output: Path) -> dict[str, Any]:
    """Versioned derivatives only; v1 bytes and identities remain untouched."""
    from analysis.localization_error_attribution import _case_directory, _load_diagnostic

    output = Path(output)
    _load(output)
    old_output = ROOT / "results" / "localization_error_attribution"
    old_manifest = _load_diagnostic(old_output)
    old_variant_sha = sha256(old_output / "variant_summary.csv")
    old_rows = list(csv.DictReader((old_output / "variant_summary.csv").open(newline="")))
    corrected = []
    residual_rows = []
    source_hashes = {"variant_summary.csv": old_variant_sha}
    for case in old_manifest["cases"]:
        directory = _case_directory(old_output, case)
        for method in METHODS:
            for variant in ("original", "ideal_bearing", "zero_bias"):
                fit_name = f"batch_fits_{method}_{variant}.csv.gz"
                updates_name = f"updates_{method}_{variant}.csv.gz"
                source_hashes[f"{directory.relative_to(old_output).as_posix()}/{fit_name}"] = sha256(directory / fit_name)
                source_hashes[f"{directory.relative_to(old_output).as_posix()}/{updates_name}"] = sha256(directory / updates_name)
                fits = _read_csv_gz(directory / fit_name)
                denied = [row for row in fits if row["reason"] == "computational_budget_exceeded_before_fit"]
                if len(denied) > 1:
                    raise ValueError("multiple budget-exhaustion records for one method/variant")
                lifecycle = _read_csv_gz(directory / f"lifecycle_{method}_{variant}.csv.gz")
                exhausted = [row for row in lifecycle if row["action"] == "computational_budget_exceeded"]
                if bool(denied) != bool(exhausted):
                    raise ValueError("archived budget fit/lifecycle mismatch")
                if denied and float(denied[0]["processing_time_s"]) != float(exhausted[0]["processing_time_s"]):
                    raise ValueError("archived budget exhaustion times disagree")
                key = (str(case["index"]), method, variant)
                original = next(row for row in old_rows if (row["index"], row["estimator_variant"], row["diagnostic_variant"]) == key)
                revised = dict(original)
                revised["schema_version"] = 2
                revised["first_budget_exhaustion_time_s"] = denied[0]["processing_time_s"] if denied else ""
                revised["budget_time_source"] = "batch_fit_and_lifecycle_agree" if denied else "no_budget_exhaustion"
                corrected.append(revised)
                for update in _read_csv_gz(directory / updates_name):
                    residual_rows.append({
                        "schema_version": 2, "source_run_id": case["source_run_id"],
                        "estimator_variant": method, "diagnostic_variant": variant,
                        "event_id": update["event_id"], "pre_update_nis": update["pre_update_nis"],
                        "residual_tangent_norm_rad": "",
                        "residual_unavailable_reason": "not_recorded_in_v1",
                    })
    correction_dir = output / "historical_corrections_v2"
    correction_dir.mkdir(exist_ok=True)
    _write_csv(correction_dir / "variant_summary_v2.csv", corrected, corrected[0].keys())
    _write_csv(correction_dir / "residual_availability_v2.csv", residual_rows, residual_rows[0].keys())
    result = {"schema_version": 2, "source_diagnostic_manifest_sha256": sha256(old_output / "diagnostic_manifest.json"),
              "source_artifact_sha256": source_hashes, "source_variant_row_count": len(old_rows),
              "corrected_variant_row_count": len(corrected), "historical_update_row_count": len(residual_rows),
              "budget_exhaustion_row_count": sum(bool(row["first_budget_exhaustion_time_s"]) for row in corrected),
              "historical_residual_policy": "unavailable; never inferred from NIS",
              "derived_sha256": {name: sha256(correction_dir / name) for name in
                                  ("variant_summary_v2.csv", "residual_availability_v2.csv")}}
    _write_json(correction_dir / "correction_manifest.json", result)
    return {key: value for key, value in result.items() if key != "source_artifact_sha256"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "run-one", "run-all", "aggregate", "correct-archived"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "init":
        result = initialize(args.output)
        payload = {"audio_stream_limit": result["audio_stream_limit"],
                   "tracker_replay_limit": result["tracker_replay_limit"]}
    elif args.action == "run-one":
        if args.index is None:
            parser.error("run-one requires --index")
        payload = run_one(args.output, args.index)
    elif args.action == "run-all":
        payload = run_all(args.output)
    elif args.action == "aggregate":
        payload = aggregate(args.output)
    else:
        payload = correct_archived_diagnostics(args.output)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
