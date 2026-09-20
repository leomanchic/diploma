"""Portable, immutable inputs and checked outputs for an offline Gazebo run."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path, PurePath
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from model.geometry import tetrahedral_array
from model.station import StationPose
from simulation.gazebo_offline import load_gazebo_recording
from simulation.multistation_audio import multistation_audio_seeds
from validation.three_station_audio_tracking_study import (
    ESTIMATOR_VARIANTS, FRAME_LENGTH, HOP_LENGTH, TRACKER_FRAME_STRIDE,
    SIGNAL_MODEL, SOURCE_MAXIMUM_FREQUENCY_HZ, MODELED_PROCESSING_DELAY_S,
    STATION_DELIVERY_DELAY_S, QC_ALPHA_M2_S3, HISTORY_WINDOW_S,
    MAXIMUM_TRACKER_RANGE_M, MAXIMUM_TRANSPORT_DELAY_S,
    POSITION_COVERAGE_THRESHOLD, MANOEUVRE_START_S, MANOEUVRE_END_S,
)
from validation.srp_statistical import SRP_SEARCH_STEPS_DEG

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
EXPERIMENT_FILE = "experiment.json"
RESULTS_FILE = "results_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _source_name(path: PurePath, root: PurePath) -> str:
    """Use the same digest label for a source file on every host OS."""
    return path.relative_to(root).as_posix()


def code_sha256() -> str:
    """Hash runnable Python source, independent of checkout path or Git metadata."""
    digest = hashlib.sha256()
    for folder in ("model", "simulation", "estimators", "validation"):
        for path in sorted((ROOT / folder).glob("*.py")):
            digest.update(_source_name(path, ROOT).encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def _git_sha() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def stations_from_experiment(experiment: dict) -> tuple[StationPose, ...]:
    return tuple(StationPose(item["id"], item["position_m"],
                             Rotation.from_euler("xyz", item["rpy_rad"]).as_matrix(),
                             np.asarray(item["microphones_local_m"], dtype=float))
                 for item in experiment["processing"]["stations"])


def calibration_from_experiment(experiment: dict) -> dict:
    result = {}
    for item in experiment["processing"]["calibration"]["values"]:
        covariance = np.asarray(item["covariance_rad2"], dtype=float)
        bias = np.asarray(item["bias_rad"], dtype=float)
        if (covariance.shape != (2, 2) or not np.all(np.isfinite(covariance))
                or np.min(np.linalg.eigvalsh(covariance)) <= 0):
            raise ValueError("frozen calibration covariance is not positive definite")
        if bias.shape != (2,) or not np.all(np.isfinite(bias)):
            raise ValueError("frozen calibration bias must be two finite radians")
        key = item["station_id"], item["estimator_variant"]
        if key in result:
            raise ValueError("duplicate frozen calibration key")
        result[key] = SimpleNamespace(covariance_rad2=covariance,
                                      mean_residual_rad=bias)
    expected = {(station.station_id, method) for station in stations_from_experiment(experiment)
                for method in experiment["processing"]["frontend"]["methods"]}
    if set(result) != expected:
        raise ValueError("frozen calibration station/method keys do not match experiment")
    return result


def check_processing_compatibility(experiment: dict) -> None:
    frontend = experiment["processing"]["frontend"]
    if frontend["methods"] != list(ESTIMATOR_VARIANTS):
        raise ValueError("frozen estimator methods are incompatible with this code")
    if frontend["gcc"] != {
        "pair_selection": "all_6_oriented_pairs", "delay_bound": "baseline_over_c_plus_2_samples",
        "interpolation_factor": 2, "minimum_frequency_hz": 200.0,
        "maximum_frequency_hz": 10000.0, "relative_spectral_floor": 1e-8,
        "wls_sigma_tdoa_s": "one_sample",
    } or frontend["srp"] != {"pair_weighting": "equal", "search_steps_deg": list(SRP_SEARCH_STEPS_DEG)}:
        raise ValueError("frozen GCC/SRP settings are incompatible with this code")
    audio = experiment["processing"]["audio"]
    if audio["source_minimum_frequency_hz"] != 300.0 or audio["source_taper_fraction"] != 0.0:
        raise ValueError("frozen source spectrum is incompatible with this code")
    if audio["noise_model"] != "independent_station_stream_AWGN":
        raise ValueError("frozen noise model is incompatible with this code")
    source_seed, noise_seeds = multistation_audio_seeds(audio["base_seed"], len(experiment["processing"]["stations"]))
    expected_noise = dict(zip((item["id"] for item in experiment["processing"]["stations"]), noise_seeds, strict=True))
    if audio["source_seed"] != source_seed or audio["station_noise_seeds"] != expected_noise:
        raise ValueError("stored audio seeds disagree with base seed")
    tracker = experiment["processing"]["tracker"]
    if tracker["manoeuvre_start_s"] != MANOEUVRE_START_S or tracker["manoeuvre_end_s"] != MANOEUVRE_END_S:
        raise ValueError("frozen evaluation phase settings are incompatible with this code")


def _calibration_snapshot(path: Path, stations: list[dict]) -> dict:
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    values = []
    for row in rows:
        if row["split"] != "calibration" or row["evaluation_used_for_calibration"] != "False":
            raise ValueError("calibration source is not an untouched calibration split")
        values.append({
            "station_id": row["station_id"], "estimator_variant": row["estimator_variant"],
            "bias_rad": [float(row["bias_az_arc_rad"]), float(row["bias_el_arc_rad"])],
            "covariance_rad2": [[float(row["covariance_00_rad2"]), float(row["covariance_01_rad2"])],
                                [float(row["covariance_01_rad2"]), float(row["covariance_11_rad2"])]],
        })
    expected = {(item["id"], method) for item in stations for method in ESTIMATOR_VARIANTS}
    if {(item["station_id"], item["estimator_variant"]) for item in values} != expected:
        raise ValueError("calibration station/method keys do not match recording")
    for item in values:
        if np.min(np.linalg.eigvalsh(item["covariance_rad2"])) <= 0:
            raise ValueError("calibration covariance must be positive definite")
    return {"origin": {"source_name": Path(path).name, "sha256": sha256(path),
                       "split": "calibration", "evaluation_used": False},
            "values": sorted(values, key=lambda item: (item["station_id"], item["estimator_variant"]))}


def _default_processing(directory: Path, calibration_path: Path) -> dict:
    recording = load_gazebo_recording(directory)
    manifest = recording.manifest
    stations = [{**item, "microphones_local_m": tetrahedral_array().tolist()}
                for item in manifest["station_config"]]
    # Legacy audio settings are recovered from existing results, not the mutable scene JSON.
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        raise ValueError("legacy migration requires the original summary.json")
    summary = json.loads(summary_path.read_text())
    if summary["gazebo_state_sha256"] != recording.csv_sha256:
        raise ValueError("legacy summary and Gazebo recording SHA-256 disagree")
    if summary["frozen_calibration_sha256"] != sha256(calibration_path):
        raise ValueError("legacy calibration source SHA-256 disagrees with summary")
    with (directory / "bearing_results.csv").open(newline="", encoding="utf-8") as file:
        bearing = next(csv.DictReader(file))
    if float(bearing["frame_start_reception_time_s"]) != 0.5:
        raise ValueError("unexpected legacy reception start; migrate manually")
    seed = int(summary["seed"])
    source_seed, noise_seeds = multistation_audio_seeds(seed, len(stations))
    if int(bearing["source_seed"]) != source_seed:
        raise ValueError("legacy source seed disagrees with the stored base seed")
    if (summary["frame_length"], summary["hop_length"], summary["tracker_frame_stride"]) != (FRAME_LENGTH, HOP_LENGTH, TRACKER_FRAME_STRIDE):
        raise ValueError("unexpected legacy frame/stride settings; migrate manually")
    recovery = asdict(InitializationRecoveryConfig(
        confirmation_reception_span_s=0.05,
        maximum_confirmation_failures=20,
        maximum_confirmation_events=180,
        maximum_initialization_buffer_events=360,
    ))
    return {
        "stations": stations,
        "audio": {"signal_model": bearing["signal_model"], "source_minimum_frequency_hz": 300.0,
                  "source_maximum_frequency_hz": float(bearing["source_maximum_frequency_hz"]),
                  "source_taper_fraction": 0.0, "sampling_rate_hz": float(summary["audio_sampling_rate_hz"]),
                  "reception_start_s": float(bearing["frame_start_reception_time_s"]),
                  "duration_s": float(summary["audio_duration_s"]), "snr_db": float(summary["snr_db"]),
                  "base_seed": seed, "source_seed": source_seed,
                  "station_noise_seeds": dict(zip((s["id"] for s in stations), noise_seeds, strict=True)),
                  "noise_model": "independent_station_stream_AWGN", "sound_speed_mps": 343.0,
                  "fir_length": 129, "chunk_size_samples": 4096,
                  "geometric_attenuation": False},
        "frontend": {"frame_length": FRAME_LENGTH, "hop_length": HOP_LENGTH,
                     "methods": list(ESTIMATOR_VARIANTS),
                     "modeled_processing_delay_s": MODELED_PROCESSING_DELAY_S,
                     "station_delivery_delay_s": dict(STATION_DELIVERY_DELAY_S),
                     "gcc": {"pair_selection": "all_6_oriented_pairs", "delay_bound": "baseline_over_c_plus_2_samples",
                             "interpolation_factor": 2, "minimum_frequency_hz": 200.0,
                             "maximum_frequency_hz": 10000.0, "relative_spectral_floor": 1e-8,
                             "wls_sigma_tdoa_s": "one_sample"},
                     "srp": {"pair_weighting": "equal", "search_steps_deg": list(SRP_SEARCH_STEPS_DEG)}},
        "tracker": {"frame_stride": TRACKER_FRAME_STRIDE,
                    "qc_m2_s3": (np.eye(3) * QC_ALPHA_M2_S3).tolist(),
                    "history_step_s": 0.25, "history_window_s": HISTORY_WINDOW_S,
                    "maximum_range_m": MAXIMUM_TRACKER_RANGE_M,
                    "maximum_transport_delay_s": MAXIMUM_TRANSPORT_DELAY_S,
                    "recovery": recovery,
                    "position_coverage_threshold": POSITION_COVERAGE_THRESHOLD,
                    "manoeuvre_start_s": 2.5, "manoeuvre_end_s": 3.5},
        "calibration": _calibration_snapshot(calibration_path, stations),
    }


def _identity(experiment: dict) -> dict:
    recording = experiment["recording"]
    return {"schema_version": experiment["schema_version"],
            "recording": {"state_sha256": recording["state_sha256"],
                          "manifest": recording["manifest"]},
            "processing": experiment["processing"],
            "code_sha256": experiment["code"]["sha256"]}


def finalize_experiment(experiment: dict) -> dict:
    """Recompute portable identity; comparison/provenance labels are excluded."""
    result = json.loads(json.dumps(experiment))
    result["config_sha256"] = canonical_sha256(_identity(result))
    result["run_id"] = "gzrun-" + result["config_sha256"][:24]
    return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _recording_snapshot(directory: Path, recording) -> dict:
    return {"state_sha256": recording.csv_sha256,
            "manifest_sha256": sha256(directory / "manifest.json"),
            "manifest": recording.manifest}


def _new_experiment(directory: Path, processing: dict, provenance: dict,
                    comparison_group_id: str | None = None) -> dict:
    recording = load_gazebo_recording(directory)
    return finalize_experiment({
        "schema_version": SCHEMA_VERSION,
        "recording": _recording_snapshot(directory, recording),
        "processing": processing,
        "code": {"sha256": code_sha256(), "git_commit_sha_at_creation": _git_sha()},
        "provenance": provenance,
        "comparison_group_id": comparison_group_id,
    })


def initialize_experiment(directory: Path, processing_config_path: Path,
                          *, comparison_group_id: str | None = None) -> dict:
    """Freeze explicit processing inputs for a new, already recorded Gazebo scene."""
    directory, config_path = Path(directory), Path(processing_config_path)
    if (directory / EXPERIMENT_FILE).exists():
        raise FileExistsError(f"{EXPERIMENT_FILE} already exists; use a new recording directory")
    result_files = ("summary.json", "bearing_results.csv", RESULTS_FILE, "viewer.html")
    if (any((directory / name).exists() for name in result_files)
            or any(directory.glob("tracking_*.csv"))
            or any(directory.glob("updates_*.csv"))):
        raise ValueError("directory already has processing results; use `migrate` for legacy data or a new recording directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(config) != {"schema_version", "processing"} or config["schema_version"] != 1:
        raise ValueError("processing config must have schema_version=1 and a processing object")
    processing = json.loads(json.dumps(config["processing"]))
    recording = load_gazebo_recording(directory)
    stations = processing["stations"]
    station_config = [{key: item[key] for key in ("id", "position_m", "rpy_rad")}
                      for item in stations]
    if station_config != recording.manifest["station_config"]:
        raise ValueError("processing stations differ from recorded Gazebo geometry")
    # Child seeds are deterministic outputs of the explicitly supplied base seed.
    audio = processing["audio"]
    source_seed, noise_seeds = multistation_audio_seeds(int(audio["base_seed"]), len(stations))
    expected_noise = dict(zip((station["id"] for station in stations), noise_seeds, strict=True))
    if "source_seed" in audio and audio["source_seed"] != source_seed:
        raise ValueError("source_seed disagrees with base_seed")
    if "station_noise_seeds" in audio and audio["station_noise_seeds"] != expected_noise:
        raise ValueError("station_noise_seeds disagree with base_seed")
    audio["source_seed"] = source_seed
    audio["station_noise_seeds"] = expected_noise
    # Check geometry, calibration keys and algorithm compatibility before writing.
    stations_from_experiment({"processing": processing})
    candidate = {"processing": processing}
    calibration_from_experiment(candidate)
    check_processing_compatibility(candidate)
    experiment = _new_experiment(directory, processing, {
        "initialized_from_processing_config": config_path.name,
        "processing_config_sha256": sha256(config_path),
    }, comparison_group_id)
    _write_json(directory / EXPERIMENT_FILE, experiment)
    return experiment


def migrate_legacy(directory: Path, calibration_path: Path) -> dict:
    directory = Path(directory)
    if (directory / EXPERIMENT_FILE).exists():
        raise FileExistsError(f"{EXPERIMENT_FILE} already exists; migration is one-time")
    processing = _default_processing(directory, calibration_path)
    experiment = _new_experiment(directory, processing,
        {"migrated_from": "03dc7d7408bc53c783af38279b622bfba8458f6c"})
    _write_json(directory / EXPERIMENT_FILE, experiment)
    return experiment


def load_experiment(directory: Path, *, check_code: bool = False) -> dict:
    directory = Path(directory)
    path = directory / EXPERIMENT_FILE
    if not path.exists():
        raise ValueError(f"no {EXPERIMENT_FILE}; use `init --processing-config` for a new recording or `migrate` for a legacy result")
    experiment = json.loads(path.read_text(encoding="utf-8"))
    if experiment.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported experiment schema; explicit migration is required")
    expected = finalize_experiment(experiment)
    if (experiment.get("config_sha256"), experiment.get("run_id")) != (expected["config_sha256"], expected["run_id"]):
        raise ValueError("experiment config SHA-256 or run_id mismatch")
    if sha256(directory / "gazebo_state.csv") != experiment["recording"]["state_sha256"]:
        raise ValueError("Gazebo recording SHA-256 differs from frozen experiment")
    if sha256(directory / "manifest.json") != experiment["recording"]["manifest_sha256"]:
        raise ValueError("Gazebo manifest SHA-256 differs from frozen experiment")
    recording = load_gazebo_recording(directory)
    if recording.manifest != experiment["recording"]["manifest"]:
        raise ValueError("Gazebo manifest content differs from frozen experiment")
    if check_code and code_sha256() != experiment["code"]["sha256"]:
        raise ValueError("processing code SHA-256 differs from frozen experiment; use the matching code checkout or create a new experiment")
    return experiment


def refresh_code_identity(directory: Path) -> dict:
    """Explicitly supersede a verified result after processing code changes."""
    directory = Path(directory)
    experiment = load_experiment(directory)
    verify_results(directory, experiment)
    new_code_sha = code_sha256()
    if new_code_sha == experiment["code"]["sha256"]:
        raise ValueError("processing code SHA-256 is unchanged; replay the existing experiment")
    updated = json.loads(json.dumps(experiment))
    updated["provenance"].setdefault("code_refresh_history", []).append({
        "previous_run_id": experiment["run_id"],
        "previous_config_sha256": experiment["config_sha256"],
        "previous_code_sha256": experiment["code"]["sha256"],
        "previous_results_manifest_sha256": sha256(directory / RESULTS_FILE),
        "previous_git_commit_sha": _git_sha(),
    })
    updated["code"]["sha256"] = new_code_sha
    updated["code"]["git_commit_sha_at_last_refresh"] = _git_sha()
    updated = finalize_experiment(updated)
    staged = directory / (EXPERIMENT_FILE + ".pending")
    _write_json(staged, updated)
    os.replace(staged, directory / EXPERIMENT_FILE)
    # The old result files remain inspectable, but cannot pass viewer checks
    # until process publishes a complete manifest for the new identity.
    (directory / RESULTS_FILE).unlink()
    return updated


def create_experiment(source: Path, destination: Path, *, snr_db: float | None = None,
                      seed: int | None = None, calibration_path: Path | None = None,
                      comparison_group_id: str | None = None) -> dict:
    source, destination = Path(source), Path(destination)
    base = load_experiment(source, check_code=True)
    if destination.exists():
        raise FileExistsError(f"new experiment destination already exists: {destination}")
    experiment = json.loads(json.dumps(base))
    audio = experiment["processing"]["audio"]
    if snr_db is not None:
        audio["snr_db"] = float(snr_db)
    if seed is not None:
        audio["base_seed"] = int(seed)
        source_seed, noise_seeds = multistation_audio_seeds(int(seed), len(experiment["processing"]["stations"]))
        audio["source_seed"] = source_seed
        audio["station_noise_seeds"] = dict(zip((s["id"] for s in experiment["processing"]["stations"]), noise_seeds, strict=True))
    if calibration_path is not None:
        experiment["processing"]["calibration"] = _calibration_snapshot(
            calibration_path, experiment["processing"]["stations"])
    experiment["comparison_group_id"] = comparison_group_id
    experiment["provenance"] = {"derived_from_run_id": base["run_id"]}
    experiment = finalize_experiment(experiment)
    if experiment["run_id"] == base["run_id"]:
        raise ValueError("new experiment settings are unchanged; replay the original directory")
    destination.mkdir(parents=True)
    for name in ("gazebo_state.csv", "manifest.json"):
        shutil.copy2(source / name, destination / name)
    _write_json(destination / EXPERIMENT_FILE, experiment)
    return experiment


def write_results_manifest(directory: Path, experiment: dict, filenames: list[str]) -> None:
    _write_json(Path(directory) / RESULTS_FILE, {
        "schema_version": SCHEMA_VERSION, "run_id": experiment["run_id"],
        "config_sha256": experiment["config_sha256"],
        "recording_sha256": experiment["recording"]["state_sha256"],
        "files": {name: sha256(Path(directory) / name) for name in sorted(filenames)},
    })


def verify_results(directory: Path, experiment: dict | None = None) -> dict:
    directory = Path(directory)
    experiment = experiment or load_experiment(directory)
    path = directory / RESULTS_FILE
    if not path.exists():
        raise ValueError(f"no checked results_manifest.json in {directory}; run process first")
    result = json.loads(path.read_text())
    if (result.get("run_id"), result.get("config_sha256"), result.get("recording_sha256")) != (
            experiment["run_id"], experiment["config_sha256"], experiment["recording"]["state_sha256"]):
        raise ValueError("result manifest belongs to a different experiment or recording")
    expected_files = {"summary.json", "bearing_results.csv"} | {
        f"{prefix}_{method}.csv" for prefix in ("tracking", "updates")
        for method in experiment["processing"]["frontend"]["methods"]}
    if set(result.get("files", {})) != expected_files:
        raise ValueError("result file list is incomplete or mixed with another experiment")
    for name, digest in result["files"].items():
        if not (directory / name).exists() or sha256(directory / name) != digest:
            raise ValueError(f"result SHA-256 mismatch: {name}")
    summary = json.loads((directory / "summary.json").read_text())
    if (summary.get("run_id"), summary.get("config_sha256"), summary.get("gazebo_state_sha256")) != (
            experiment["run_id"], experiment["config_sha256"], experiment["recording"]["state_sha256"]):
        raise ValueError("summary belongs to a different experiment or Gazebo recording")
    for name in result["files"]:
        if not name.endswith(".csv"):
            continue
        with (directory / name).open(newline="", encoding="utf-8") as file:
            rows = csv.DictReader(file)
            if rows.fieldnames is None or "run_id" not in rows.fieldnames:
                raise ValueError(f"result has no run_id: {name}")
            for row in rows:
                if row["run_id"] != experiment["run_id"] or row.get("sequence_id", experiment["run_id"]) != experiment["run_id"]:
                    raise ValueError(f"mixed experiment IDs in {name}")
                if row.get("event_id") and not row["event_id"].startswith(experiment["run_id"] + "|"):
                    raise ValueError(f"mixed event IDs in {name}")
    return summary
