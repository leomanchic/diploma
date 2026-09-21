"""Run the frozen PX4 localization range study sequentially and resumably.

The module translates two saved Gazebo trajectories without rotating or
scaling them, synthesizes one shared source stream with either legacy
received-SNR noise or fixed acoustic background noise, and runs the existing
GCC/SRP front ends and retarded-time tracker unchanged.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray

from estimators.retarded_ekf_manoeuvre import ManoeuvreHistoryConfig
from estimators.retarded_ekf_recovery import InitializationRecoveryConfig
from simulation.gazebo_offline import GazeboRecording, load_gazebo_recording
from simulation.multistation_audio import (
    _common_source_support,
    multistation_noise_seeds,
    synthesize_multistation_audio,
)
from simulation.signals import random_bandlimited_signal
from validation.gazebo_experiment import (
    calibration_from_experiment,
    canonical_sha256,
    code_sha256,
    sha256,
    stations_from_experiment,
)
from validation.three_station_audio_tracking_study import (
    bearing_measurements_from_records,
    extract_audio_bearing_records,
    run_tracker,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "localization_range_study"
PROTOCOL = ROOT / "LOCALIZATION_RANGE_PROTOCOL.md"
REFERENCE_EXPERIMENT = ROOT / "results" / "px4_flight" / "pilot_003"
RECORDINGS = {
    "single_turn": REFERENCE_EXPERIMENT,
    "opposite_turns": ROOT / "results" / "px4_flight" / "opposite_turn_001",
}
ROOT_SEED = 20260921
DISTANCES_M = (50, 100, 200, 400, 700, 1000)
REFERENCE_SNRS_DB = (0.0, 10.0, 20.0)
REPLICATES = (0, 1, 2)
AZIMUTH_DEG = 45.0
ELEVATION_DEG = 10.0
REFERENCE_DISTANCE_M = 100.0
MAXIMUM_RANGE_M = 1300.0
HISTORY_WINDOW_S = 4.25
HISTORY_STEP_S = 0.25
MAXIMUM_TRANSPORT_DELAY_S = 0.10
PASS_P95_THRESHOLDS_M = (2.0, 5.0, 10.0)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TranslatedTrajectory:
    """A finite-support translation of an observed Gazebo trajectory."""

    base: Any
    offset_m: ArrayLike
    kind: str
    knot_times_s: NDArray[np.float64] = field(init=False)
    maximum_speed_mps: float = field(init=False)

    def __post_init__(self) -> None:
        offset = np.asarray(self.offset_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("offset_m must contain three finite ENU coordinates")
        object.__setattr__(self, "offset_m", offset.copy())
        object.__setattr__(self, "knot_times_s", self.base.knot_times_s)
        object.__setattr__(self, "maximum_speed_mps", self.base.maximum_speed_mps)

    def q(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self.base.q(time_s), dtype=float) + self.offset_m

    def v(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self.base.v(time_s), dtype=float)

    def a(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self.base.a(time_s), dtype=float)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _seed(*coordinates: int) -> int:
    return int(
        np.random.SeedSequence(list(coordinates)).generate_state(
            1, dtype=np.uint64
        )[0]
    )


def _direction() -> NDArray[np.float64]:
    azimuth = np.deg2rad(AZIMUTH_DEG)
    elevation = np.deg2rad(ELEVATION_DEG)
    return np.asarray([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])


def _processing_snapshot() -> tuple[dict, Any]:
    experiment = json.loads(
        (REFERENCE_EXPERIMENT / "experiment.json").read_text(encoding="utf-8")
    )
    processing = json.loads(json.dumps(experiment["processing"]))
    tracker = processing["tracker"]
    tracker["maximum_range_m"] = MAXIMUM_RANGE_M
    tracker["history_window_s"] = HISTORY_WINDOW_S
    tracker["history_step_s"] = HISTORY_STEP_S
    tracker["maximum_transport_delay_s"] = MAXIMUM_TRANSPORT_DELAY_S
    return processing, experiment


def _recording_inputs() -> dict[str, dict[str, Any]]:
    result = {}
    for name, path in RECORDINGS.items():
        recording = load_gazebo_recording(path)
        processing = json.loads((path / "processing_config.json").read_text())[
            "processing"
        ]
        audio = processing["audio"]
        result[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "state_sha256": recording.csv_sha256,
            "manifest_sha256": sha256(path / "manifest.json"),
            "reception_start_s": float(audio["reception_start_s"]),
            "duration_s": float(audio["duration_s"]),
            "evaluation_phases": processing["evaluation_phases"],
            "manoeuvre_start_s": float(processing["tracker"]["manoeuvre_start_s"]),
            "manoeuvre_end_s": float(processing["tracker"]["manoeuvre_end_s"]),
            "export_rate_hz": 1.0 / float(recording.manifest["export_period_s"]),
        }
    return result


def _matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for trajectory_index, trajectory_name in enumerate(RECORDINGS):
        source_seed = _seed(ROOT_SEED, 0, trajectory_index)
        noise_seed = _seed(ROOT_SEED, 1, trajectory_index, 0)
        for distance in DISTANCES_M:
            rows.append({
                "index": index, "trajectory_index": trajectory_index,
                "trajectory": trajectory_name, "experiment": "geometry_control",
                "distance_m": distance, "snr_db": 10.0, "replicate": 0,
                "source_seed": source_seed, "noise_seed": noise_seed,
                "noise_model": "received_snr", "geometric_attenuation": False,
            })
            index += 1
    for trajectory_index, trajectory_name in enumerate(RECORDINGS):
        source_seed = _seed(ROOT_SEED, 0, trajectory_index)
        for distance in DISTANCES_M:
            for snr_db in REFERENCE_SNRS_DB:
                for replicate in REPLICATES:
                    rows.append({
                        "index": index, "trajectory_index": trajectory_index,
                        "trajectory": trajectory_name,
                        "experiment": "fixed_background",
                        "distance_m": distance, "snr_db": snr_db,
                        "replicate": replicate, "source_seed": source_seed,
                        "noise_seed": _seed(
                            ROOT_SEED, 2, trajectory_index, replicate
                        ),
                        "noise_model": "fixed_reference_snr",
                        "geometric_attenuation": True,
                    })
                    index += 1
    if len(rows) != 120:
        raise AssertionError("the frozen matrix must contain exactly 120 runs")
    return rows


def _translated(
    recording: GazeboRecording, stations: tuple, reception_start_s: float,
    distance_m: float, trajectory_name: str,
) -> tuple[TranslatedTrajectory, NDArray[np.float64]]:
    centroid = np.mean([station.position_world_m for station in stations], axis=0)
    target = centroid + float(distance_m) * _direction()
    offset = target - np.asarray(recording.trajectory.q(reception_start_s))
    trajectory = TranslatedTrajectory(
        recording.trajectory, offset,
        f"transformed_px4_{trajectory_name}_d{int(distance_m)}m",
    )
    return trajectory, offset


def _geometry_summary(
    trajectory: TranslatedTrajectory, stations: tuple, start_s: float,
    duration_s: float,
) -> dict[str, Any]:
    stop_s = start_s + duration_s
    knots = trajectory.knot_times_s
    dense = np.arange(start_s, stop_s, 0.01, dtype=float)
    times = np.unique(np.concatenate((
        dense, [start_s], knots[(knots > start_s) & (knots < stop_s)], [stop_s]
    )))
    positions = trajectory.q(times)
    station_ranges = {}
    for station in stations:
        distance = np.linalg.norm(
            positions - station.position_world_m[None, :], axis=1
        )
        station_ranges[station.station_id] = {
            "minimum_m": float(np.min(distance)),
            "maximum_m": float(np.max(distance)),
        }
    minimum_height = float(np.min(positions[:, 2]))
    minimum_station_distance = min(
        value["minimum_m"] for value in station_ranges.values()
    )
    maximum_station_distance = max(
        value["maximum_m"] for value in station_ranges.values()
    )
    if minimum_height <= 0.0:
        raise ValueError("translated trajectory reaches or crosses the ground")
    if minimum_station_distance <= 1.0:
        raise ValueError("translated trajectory intersects a station")
    if maximum_station_distance > MAXIMUM_RANGE_M:
        raise ValueError("translated trajectory exceeds frozen tracker range")
    return {
        "minimum_height_m": minimum_height,
        "minimum_station_distance_m": minimum_station_distance,
        "maximum_station_distance_m": maximum_station_distance,
        "station_ranges_m": station_ranges,
    }


def _matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def initialize(output: Path = DEFAULT_OUTPUT) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("study output exists; use a new directory or resume it")
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir()
    (output / "source_banks").mkdir()
    processing, reference_experiment = _processing_snapshot()
    recordings = _recording_inputs()
    stations = stations_from_experiment(reference_experiment)
    matrix = _matrix()

    # Freeze every translated geometry and its deterministic identity first.
    geometries: dict[str, dict[str, Any]] = {}
    for spec in matrix:
        key = f"{spec['trajectory']}:{spec['distance_m']}"
        if key in geometries:
            continue
        info = recordings[spec["trajectory"]]
        recording = load_gazebo_recording(ROOT / info["path"])
        trajectory, offset = _translated(
            recording, stations, info["reception_start_s"],
            spec["distance_m"], spec["trajectory"],
        )
        geometries[key] = {
            "translation_enu_m": offset.tolist(),
            **_geometry_summary(
                trajectory, stations, info["reception_start_s"], info["duration_s"]
            ),
        }

    source_banks = {}
    audio = processing["audio"]
    sampling_rate = float(audio["sampling_rate_hz"])
    for trajectory_index, trajectory_name in enumerate(RECORDINGS):
        info = recordings[trajectory_name]
        recording = load_gazebo_recording(ROOT / info["path"])
        starts, stops = [], []
        for distance in DISTANCES_M:
            trajectory, _ = _translated(
                recording, stations, info["reception_start_s"], distance,
                trajectory_name,
            )
            reception = info["reception_start_s"] + np.arange(
                int(np.rint(info["duration_s"] * sampling_rate)), dtype=float
            ) / sampling_rate
            start, count = _common_source_support(
                reception, stations, trajectory, sampling_rate,
                float(audio["sound_speed_mps"]), int(audio["fir_length"]),
            )
            starts.append(start)
            stops.append(start + count / sampling_rate)
        bank_start = float(min(starts))
        count = int(np.ceil((max(stops) - bank_start) * sampling_rate)) + 1
        source_seed = _seed(ROOT_SEED, 0, trajectory_index)
        source = random_bandlimited_signal(
            sampling_rate, count, np.random.default_rng(source_seed),
            minimum_frequency_hz=float(audio["source_minimum_frequency_hz"]),
            maximum_frequency_hz=float(audio["source_maximum_frequency_hz"]),
            taper_fraction=float(audio["source_taper_fraction"]),
        )
        relative = Path("source_banks") / f"{trajectory_name}.npy"
        np.save(output / relative, source, allow_pickle=False)
        source_banks[trajectory_name] = {
            "path": relative.as_posix(), "start_time_s": bank_start,
            "sample_count": count, "sampling_rate_hz": sampling_rate,
            "seed": source_seed,
            "signal_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "file_sha256": sha256(output / relative),
            "rms": float(np.sqrt(np.mean(source**2))),
        }

    matrix_path = output / "run_matrix.csv"
    _matrix_csv(matrix_path, matrix)
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "protocol_sha256": sha256(PROTOCOL),
        "protocol_path": PROTOCOL.relative_to(ROOT).as_posix(),
        "code_sha256": code_sha256(),
        "git_commit_sha_at_initialization": _git_sha(),
        "root_seed": ROOT_SEED,
        "frame": "ENU x=East y=North z=Up; SI units",
        "recordings": recordings,
        "geometry": {
            "initial_azimuth_deg": AZIMUTH_DEG,
            "initial_elevation_deg": ELEVATION_DEG,
            "initial_distances_m": list(DISTANCES_M),
            "translations": geometries,
        },
        "processing": processing,
        "fixed_background": {
            "amplitude_law": "1/r at microphone emission range",
            "reference_distance_m": REFERENCE_DISTANCE_M,
            "reference_snrs_db": list(REFERENCE_SNRS_DB),
            "reference_source_rms": 1.0,
            "no_per_stream_normalization": True,
        },
        "source_banks": source_banks,
        "run_count": len(matrix),
        "run_matrix_sha256": sha256(matrix_path),
        "runs": matrix,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
        },
    }
    frozen["processing_sha256"] = canonical_sha256(frozen["processing"])
    _write_json(output / "study_manifest.json", frozen)
    return frozen


def _load_study(output: Path, *, check_code: bool = True) -> dict:
    output = Path(output)
    manifest = json.loads((output / "study_manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported range-study schema")
    if sha256(PROTOCOL) != manifest["protocol_sha256"]:
        raise ValueError("range-study protocol changed after initialization")
    if check_code and code_sha256() != manifest["code_sha256"]:
        raise ValueError("processing code changed after study initialization")
    if sha256(output / "run_matrix.csv") != manifest["run_matrix_sha256"]:
        raise ValueError("run matrix SHA-256 mismatch")
    for name, info in manifest["recordings"].items():
        path = ROOT / info["path"]
        recording = load_gazebo_recording(path)
        if recording.csv_sha256 != info["state_sha256"]:
            raise ValueError(f"Gazebo recording changed: {name}")
        if sha256(path / "manifest.json") != info["manifest_sha256"]:
            raise ValueError(f"Gazebo manifest changed: {name}")
    for name, info in manifest["source_banks"].items():
        path = output / info["path"]
        if sha256(path) != info["file_sha256"]:
            raise ValueError(f"source bank changed: {name}")
    return manifest


def _run_identity(manifest: dict, spec: dict) -> dict:
    info = manifest["recordings"][spec["trajectory"]]
    geometry = manifest["geometry"]["translations"][
        f"{spec['trajectory']}:{spec['distance_m']}"
    ]
    bank = manifest["source_banks"][spec["trajectory"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "code_sha256": manifest["code_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "processing_sha256": manifest["processing_sha256"],
        "recording_state_sha256": info["state_sha256"],
        "recording_manifest_sha256": info["manifest_sha256"],
        "source_signal_sha256": bank["signal_sha256"],
        "translation_enu_m": geometry["translation_enu_m"],
        "spec": spec,
    }


def _run_id(manifest: dict, spec: dict) -> str:
    return "range-" + canonical_sha256(_run_identity(manifest, spec))[:24]


def _run_directory(output: Path, spec: dict, run_id: str) -> Path:
    snr = f"{float(spec['snr_db']):g}".replace("-", "m")
    name = (
        f"{int(spec['index']):03d}_{spec['trajectory']}_{spec['experiment']}_"
        f"d{int(spec['distance_m'])}_snr{snr}_r{int(spec['replicate'])}_{run_id}"
    )
    return Path(output) / "runs" / name


def _phase_classifier(phases: list[dict[str, Any]]):
    def classify(time_s: float) -> str:
        for phase in phases:
            if phase["start_s"] <= time_s < phase["end_s"]:
                return phase["name"]
        return "before_report" if time_s < phases[0]["start_s"] else "after_report"
    return classify


def _standard_noise_sha256(stream) -> str:
    digest = hashlib.sha256()
    for station in stream.stations:
        digest.update(station.station_id.encode("utf-8") + b"\0")
        digest.update(str(station.noise_seed).encode("ascii") + b"\0")
        digest.update(json.dumps(station.noise.shape).encode("ascii") + b"\0")
        generator = np.random.default_rng(station.noise_seed)
        remaining = int(np.prod(station.noise.shape))
        while remaining:
            count = min(remaining, 1_000_000)
            digest.update(generator.normal(size=count).astype(np.float64).tobytes())
            remaining -= count
    return digest.hexdigest()


def _write_csv_gz(path: Path, rows: list[dict], empty_columns: tuple[str, ...] = ()) -> None:
    if not rows and not empty_columns:
        raise ValueError("empty CSV requires an explicit schema")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(
                    text, fieldnames=list(rows[0]) if rows else list(empty_columns),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
    os.replace(temporary, path)


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _method_metrics(
    method: str, bearings: list[dict], track: list[dict], updates: list[dict],
    sequence: dict, reception_start_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid_track = [row for row in track if row["valid"]]
    post_three = [
        row for row in track
        if float(row["processing_time_s"]) >= reception_start_s + 3.0
    ]
    valid_post_three = [row for row in post_three if row["valid"]]
    errors = np.asarray([row["position_error_m"] for row in valid_track], dtype=float)
    first_absolute = float(sequence["first_confirmation_time_s"])
    first_relative = (
        first_absolute - reception_start_s if np.isfinite(first_absolute) else None
    )
    p95 = float(np.percentile(errors, 95)) if errors.size else None
    availability_post_three = (
        len(valid_post_three) / len(post_three) if post_three else 0.0
    )
    failures = Counter(
        str(row["failure_reason"] or row["status"])
        for row in track if not row["valid"]
    )
    rejection_reasons = Counter(
        str(row["failure_reason"] or "unspecified")
        for row in updates if not row["update_applied"]
    )
    track_loss_count = 0
    recovery_count = 0
    has_been_valid = bool(track and track[0]["valid"])
    for previous, current in zip(track, track[1:]):
        if bool(previous["valid"]) and not bool(current["valid"]):
            track_loss_count += 1
        if has_been_valid and not bool(previous["valid"]) and bool(current["valid"]):
            recovery_count += 1
        has_been_valid = has_been_valid or bool(current["valid"])
    metrics: dict[str, Any] = {
        "estimator_variant": method,
        "bearing_frame_count": sum(
            row["estimator_variant"] == method for row in bearings
        ),
        "valid_bearing_frame_count": sum(
            row["estimator_variant"] == method and row["valid"] for row in bearings
        ),
        "publication_count": len(track),
        "valid_publication_count": len(valid_track),
        "availability_fraction": len(valid_track) / len(track) if track else 0.0,
        "availability_after_3s_fraction": availability_post_three,
        "position_rmse_m_conditional": (
            float(np.sqrt(np.mean(errors**2))) if errors.size else None
        ),
        "position_p95_m_conditional": p95,
        "first_confirmation_time_s_absolute": (
            first_absolute if np.isfinite(first_absolute) else None
        ),
        "first_confirmation_time_s_relative": first_relative,
        "accepted_update_count": int(sequence["accepted_update_count"]),
        "rejected_update_count": int(sequence["rejected_update_count"]),
        "track_loss_count": track_loss_count,
        "recovery_count": recovery_count,
        "reset_count": int(sequence["reset_count"]),
        "final_valid": bool(sequence["final_valid"]),
        "final_confirmed": bool(sequence["final_confirmed"]),
        "final_failure_reason": str(sequence["failure_reason"]),
        "failure_reasons": dict(failures),
        "update_rejection_reasons": dict(rejection_reasons),
        "tracker_runtime_s": float(sequence["tracker_runtime_s"]),
        "maximum_history_memory_bytes": int(sequence["maximum_history_memory_bytes"]),
    }
    for threshold in PASS_P95_THRESHOLDS_M:
        metrics[f"pass_p95_{threshold:g}m"] = bool(
            first_relative is not None
            and first_relative <= 3.0
            and availability_post_three >= 0.90
            and p95 is not None
            and p95 <= threshold
        )

    angular_rows = []
    for station_id in sorted({str(row["station_id"]) for row in bearings}):
        subset = [
            row for row in bearings
            if row["estimator_variant"] == method
            and row["station_id"] == station_id
        ]
        valid = np.asarray(
            [row["geodesic_error_deg"] for row in subset if row["valid"]],
            dtype=float,
        )
        angular_rows.append({
            "station_id": station_id, "estimator_variant": method,
            "frame_count": len(subset), "valid_frame_count": int(valid.size),
            "angular_rmse_deg": (
                float(np.sqrt(np.mean(valid**2))) if valid.size else None
            ),
            "angular_p95_deg": (
                float(np.percentile(valid, 95)) if valid.size else None
            ),
        })
    return metrics, angular_rows


def _compute(
    output: Path, manifest: dict, spec: dict, *, duration_override_s: float | None = None,
) -> tuple[dict, list[dict], dict[str, list[dict]], dict[str, list[dict]]]:
    run_id = _run_id(manifest, spec)
    processing = manifest["processing"]
    stations = stations_from_experiment({"processing": processing})
    calibration = calibration_from_experiment({"processing": processing})
    info = manifest["recordings"][spec["trajectory"]]
    recording = load_gazebo_recording(ROOT / info["path"])
    reception_start = float(info["reception_start_s"])
    duration = (
        float(info["duration_s"])
        if duration_override_s is None else float(duration_override_s)
    )
    trajectory, offset = _translated(
        recording, stations, reception_start, spec["distance_m"],
        spec["trajectory"],
    )
    geometry = _geometry_summary(
        trajectory, stations, reception_start, duration
    )
    bank_info = manifest["source_banks"][spec["trajectory"]]
    source = np.load(output / bank_info["path"], allow_pickle=False)
    if hashlib.sha256(source.tobytes()).hexdigest() != bank_info["signal_sha256"]:
        raise ValueError("source bank content SHA-256 mismatch")
    audio = processing["audio"]
    started = time.perf_counter()
    synthesis_started = time.perf_counter()
    stream = synthesize_multistation_audio(
        stations, trajectory, duration_s=duration,
        reception_start_time_s=reception_start,
        sampling_rate_hz=float(audio["sampling_rate_hz"]),
        sound_speed=float(audio["sound_speed_mps"]),
        signal_model=str(audio["signal_model"]), snr_db=float(spec["snr_db"]),
        seed=ROOT_SEED, chunk_size_samples=int(audio["chunk_size_samples"]),
        fir_length=int(audio["fir_length"]),
        geometric_attenuation=bool(spec["geometric_attenuation"]),
        maximum_emitted_frequency_hz=float(audio["source_maximum_frequency_hz"]),
        external_source_signal=source,
        external_source_start_time_s=float(bank_info["start_time_s"]),
        noise_model=str(spec["noise_model"]),
        reference_distance_m=REFERENCE_DISTANCE_M,
        source_seed=int(spec["source_seed"]), noise_seed=int(spec["noise_seed"]),
        reference_source_rms=1.0,
    )
    synthesis_runtime = time.perf_counter() - synthesis_started
    frontend = processing["frontend"]
    bearings, frontend_runtime = extract_audio_bearing_records(
        stream, stations, trajectory, split="evaluation",
        configuration_index=int(spec["index"]),
        sequence_index=int(spec["replicate"]),
        frame_length=int(frontend["frame_length"]),
        hop_length=int(frontend["hop_length"]),
        modeled_processing_delay_s=float(frontend["modeled_processing_delay_s"]),
        station_delivery_delay_s=frontend["station_delivery_delay_s"],
        sequence_id=run_id,
    )
    tracker = processing["tracker"]
    history = ManoeuvreHistoryConfig(
        np.asarray(tracker["qc_m2_s3"], dtype=float),
        history_step_s=float(tracker["history_step_s"]),
        history_window_s=float(tracker["history_window_s"]),
        maximum_range_m=float(tracker["maximum_range_m"]),
        maximum_transport_delay_s=float(tracker["maximum_transport_delay_s"]),
    )
    recovery = InitializationRecoveryConfig(**tracker["recovery"])
    phases = info["evaluation_phases"]
    classifier = _phase_classifier(phases)
    tracks: dict[str, list[dict]] = {}
    updates: dict[str, list[dict]] = {}
    method_metrics = {}
    angular_rows = []
    for method in frontend["methods"]:
        measurements = bearing_measurements_from_records(
            bearings, calibration, method,
            frame_stride=int(tracker["frame_stride"]),
        )
        track, sequence, method_updates = run_tracker(
            stations, trajectory, measurements, method,
            history_config=history, recovery_config=recovery,
            coverage_threshold=float(tracker["position_coverage_threshold"]),
            phase_classifier=classifier,
            manoeuvre_start_s=float(info["manoeuvre_start_s"]),
        )
        for row in track + method_updates:
            row["run_id"] = run_id
            row["sequence_id"] = run_id
        tracks[method] = track
        updates[method] = method_updates
        metrics, station_rows = _method_metrics(
            method, bearings, track, method_updates, sequence, reception_start
        )
        method_metrics[method] = metrics
        angular_rows.extend(station_rows)
    for row in bearings:
        row["run_id"] = run_id
    summary = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "sequence_id": run_id, "spec": spec,
        "recording_state_sha256": info["state_sha256"],
        "recording_manifest_sha256": info["manifest_sha256"],
        "source_signal_sha256": bank_info["signal_sha256"],
        "standardized_noise_sha256": _standard_noise_sha256(stream),
        "translation_enu_m": offset.tolist(), "geometry": geometry,
        "reception_start_s": reception_start, "duration_s": duration,
        "actual_station_snr_db": {
            station.station_id: station.effective_snr_db
            for station in stream.stations
        },
        "station_noise_rms": {
            station.station_id: station.noise_rms for station in stream.stations
        },
        "source_start_time_s": stream.source_start_time_s,
        "source_sample_count": len(stream.source_signal),
        "audio_synthesis_wall_runtime_s": synthesis_runtime,
        "bearing_frontend_wall_runtime_s": frontend_runtime,
        "total_wall_runtime_s": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "methods": method_metrics, "station_angular_metrics": angular_rows,
    }
    return summary, bearings, tracks, updates


def _artifact_hashes(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256(directory / name) for name in names}


def _verify_completed(directory: Path) -> dict:
    experiment = json.loads((directory / "experiment.json").read_text())
    if experiment.get("status") != "complete":
        raise ValueError(f"incomplete range run: {directory.name}")
    for name, expected in experiment["result_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"range run result SHA-256 mismatch: {directory.name}/{name}")
    summary = json.loads((directory / "summary.json").read_text())
    if summary["run_id"] != experiment["run_id"]:
        raise ValueError("range run summary identity mismatch")
    return summary


def run_one(output: Path, index: int) -> dict:
    output = Path(output)
    manifest = _load_study(output)
    if not 0 <= int(index) < len(manifest["runs"]):
        raise IndexError("run index lies outside the frozen matrix")
    spec = manifest["runs"][int(index)]
    run_id = _run_id(manifest, spec)
    directory = _run_directory(output, spec, run_id)
    identity = _run_identity(manifest, spec)
    experiment_path = directory / "experiment.json"
    if experiment_path.exists():
        existing = json.loads(experiment_path.read_text())
        if existing["identity"] != identity or existing["run_id"] != run_id:
            raise ValueError("existing run directory contains another experiment")
        if existing.get("status") == "complete":
            return _verify_completed(directory)
    else:
        directory.mkdir(parents=True, exist_ok=False)
    experiment = {
        "schema_version": SCHEMA_VERSION, "status": "running",
        "run_id": run_id, "sequence_id": run_id, "identity": identity,
        "configuration_frozen_before_processing": True,
    }
    _write_json(experiment_path, experiment)
    try:
        summary, bearings, tracks, updates = _compute(output, manifest, spec)
        artifacts = []
        angular_error_rows = [{
            "run_id": run_id,
            "sequence_id": row["sequence_id"],
            "station_id": row["station_id"],
            "estimator_variant": row["estimator_variant"],
            "frame_index": row["frame_index"],
            "frame_center_reception_time_s": row["frame_center_reception_time_s"],
            "true_emission_time_s_evaluator_only": row[
                "true_emission_time_s_evaluator_only"
            ],
            "valid": row["valid"],
            "invalid_reason": row["invalid_reason"],
            "boundary_hit": row["boundary_hit"],
            "geodesic_error_deg": row["geodesic_error_deg"],
            "effective_station_snr_db": row["effective_station_snr_db"],
            "truth_used_by_audio_estimator": row["truth_used_by_audio_estimator"],
        } for row in bearings]
        _write_csv_gz(directory / "angular_errors.csv.gz", angular_error_rows)
        artifacts.append("angular_errors.csv.gz")
        methods = manifest["processing"]["frontend"]["methods"]
        for method in methods:
            tracking_name = f"tracking_{method}.csv.gz"
            updates_name = f"updates_{method}.csv.gz"
            _write_csv_gz(directory / tracking_name, tracks[method])
            _write_csv_gz(
                directory / updates_name, updates[method],
                ("run_id", "sequence_id", "event_id", "station_id", "frame_index",
                 "estimator_variant", "update_applied", "failure_reason"),
            )
            artifacts.extend((tracking_name, updates_name))
        _write_json(directory / "summary.json", summary)
        artifacts.append("summary.json")
        experiment["status"] = "complete"
        experiment["result_sha256"] = _artifact_hashes(directory, artifacts)
        _write_json(experiment_path, experiment)
        return summary
    except Exception as error:
        experiment["status"] = "failed"
        experiment["error"] = f"{type(error).__name__}: {error}"
        _write_json(experiment_path, experiment)
        raise


def probe(output: Path) -> dict:
    output = Path(output)
    manifest = _load_study(output)
    spec = next(
        row for row in manifest["runs"]
        if row["trajectory"] == "single_turn"
        and row["experiment"] == "fixed_background"
        and row["distance_m"] == 100
        and row["snr_db"] == 10.0
        and row["replicate"] == 0
    )
    started = time.perf_counter()
    summary, bearings, tracks, updates = _compute(
        output, manifest, spec, duration_override_s=1.0
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_run": False,
        "purpose": "predeclared one-second technical time/memory probe",
        "spec": spec, "wall_runtime_s": time.perf_counter() - started,
        "peak_rss_bytes": _peak_rss_bytes(), "bearing_rows": len(bearings),
        "tracking_rows": {key: len(value) for key, value in tracks.items()},
        "update_rows": {key: len(value) for key, value in updates.items()},
        "component_runtime_s": {
            "audio_synthesis": summary["audio_synthesis_wall_runtime_s"],
            "bearing_frontend": summary["bearing_frontend_wall_runtime_s"],
        },
    }
    _write_json(output / "probe.json", result)
    return result


def run_all(output: Path) -> dict:
    output = Path(output)
    manifest = _load_study(output)
    completed = skipped = 0
    started = time.perf_counter()
    for spec in manifest["runs"]:
        run_id = _run_id(manifest, spec)
        directory = _run_directory(output, spec, run_id)
        if (directory / "experiment.json").exists():
            existing = json.loads((directory / "experiment.json").read_text())
            if existing.get("status") == "complete":
                _verify_completed(directory)
                skipped += 1
                continue
        subprocess.run(
            [
                sys.executable, "-m", "validation.localization_range_study",
                "run-one", "--index", str(spec["index"]),
                "--output", str(output.resolve()),
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            check=True,
        )
        completed += 1
        print(
            f"[{completed + skipped}/{len(manifest['runs'])}] completed "
            f"{directory.name}", flush=True,
        )
    return {
        "completed_now": completed, "skipped_verified": skipped,
        "total": len(manifest["runs"]),
        "wall_runtime_s": time.perf_counter() - started,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty aggregate table {path.name}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: _plain(value) for key, value in row.items()} for row in rows)


def aggregate(output: Path) -> dict:
    output = Path(output)
    manifest = _load_study(output)
    run_rows, station_rows = [], []
    summaries = []
    for spec in manifest["runs"]:
        run_id = _run_id(manifest, spec)
        directory = _run_directory(output, spec, run_id)
        summary = _verify_completed(directory)
        summaries.append(summary)
        snrs = summary["actual_station_snr_db"]
        for method, metrics in summary["methods"].items():
            run_rows.append({
                "run_id": run_id, **spec, "estimator_variant": method,
                "actual_snr_S0_db": snrs["S0"],
                "actual_snr_S1_db": snrs["S1"],
                "actual_snr_S2_db": snrs["S2"],
                "minimum_station_distance_m": summary["geometry"]["minimum_station_distance_m"],
                "maximum_station_distance_m": summary["geometry"]["maximum_station_distance_m"],
                **metrics,
                "total_wall_runtime_s": summary["total_wall_runtime_s"],
                "peak_rss_bytes": summary["peak_rss_bytes"],
            })
        for station_metric in summary["station_angular_metrics"]:
            station_rows.append({
                "run_id": run_id, **spec,
                "actual_station_snr_db": snrs[station_metric["station_id"]],
                **station_metric,
            })
    _write_csv(output / "run_summary.csv", run_rows)
    _write_csv(output / "station_summary.csv", station_rows)

    group_fields = ("experiment", "trajectory", "distance_m", "snr_db", "estimator_variant")
    grouped: dict[tuple, list[dict]] = {}
    for row in run_rows:
        key = tuple(row[name] for name in group_fields)
        grouped.setdefault(key, []).append(row)
    group_rows = []
    for key, rows in grouped.items():
        item = dict(zip(group_fields, key, strict=True))
        item["replicate_count"] = len(rows)
        for threshold in PASS_P95_THRESHOLDS_M:
            field_name = f"pass_p95_{threshold:g}m"
            success_count = sum(bool(row[field_name]) for row in rows)
            item[f"success_count_p95_{threshold:g}m"] = success_count
            item[f"group_pass_p95_{threshold:g}m"] = success_count == len(rows)
        item["availability_after_3s_min"] = min(
            row["availability_after_3s_fraction"] for row in rows
        )
        p95_values = [
            row["position_p95_m_conditional"] for row in rows
            if row["position_p95_m_conditional"] is not None
        ]
        item["position_p95_m_worst_replicate"] = max(p95_values) if p95_values else None
        group_rows.append(item)
    group_rows.sort(key=lambda row: tuple(row[name] for name in group_fields))
    _write_csv(output / "group_summary.csv", group_rows)

    boundary_rows = []
    boundary_fields = ("experiment", "trajectory", "snr_db", "estimator_variant")
    keys = sorted({tuple(row[name] for name in boundary_fields) for row in group_rows})
    for key in keys:
        values = sorted(
            [row for row in group_rows if tuple(row[name] for name in boundary_fields) == key],
            key=lambda row: row["distance_m"],
        )
        for threshold in PASS_P95_THRESHOLDS_M:
            pass_field = f"group_pass_p95_{threshold:g}m"
            passed = [row["distance_m"] for row in values if row[pass_field]]
            pattern = [bool(row[pass_field]) for row in values]
            nonmonotonic = any(
                pattern[index] and not all(pattern[:index])
                for index in range(1, len(pattern))
            )
            largest = max(passed) if passed else None
            if 1000 in passed:
                conclusion = "граница не найдена в исследованном диапазоне"
            elif largest is None:
                conclusion = "ни одна исследованная точка не прошла критерий"
            else:
                conclusion = f"наибольшая прошедшая исследованная точка: {largest:g} м"
            boundary_rows.append({
                **dict(zip(boundary_fields, key, strict=True)),
                "p95_threshold_m": threshold,
                "largest_passing_studied_distance_m": largest,
                "nonmonotonic": nonmonotonic,
                "pass_pattern_50_100_200_400_700_1000": "".join(
                    "1" if value else "0" for value in pattern
                ),
                "conclusion": conclusion,
            })
    _write_csv(output / "range_boundaries.csv", boundary_rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "audio_run_count": len(summaries),
        "method_result_count": len(run_rows),
        "station_method_result_count": len(station_rows),
        "completed_run_ids_sha256": canonical_sha256(
            sorted(summary["run_id"] for summary in summaries)
        ),
        "tables": {
            name: sha256(output / name) for name in (
                "run_summary.csv", "station_summary.csv", "group_summary.csv",
                "range_boundaries.csv",
            )
        },
        "total_measured_wall_runtime_s": float(sum(
            summary["total_wall_runtime_s"] for summary in summaries
        )),
        "maximum_peak_rss_bytes": max(
            summary["peak_rss_bytes"] or 0 for summary in summaries
        ),
    }
    _write_json(output / "study_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("init", "probe", "run-one", "run-all", "aggregate")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "run-one" and args.index is None:
        parser.error("run-one requires --index")
    if args.action != "run-one" and args.index is not None:
        parser.error("--index is allowed only with run-one")
    if args.action == "init":
        result = initialize(args.output)
    elif args.action == "probe":
        result = probe(args.output)
    elif args.action == "run-one":
        result = run_one(args.output, args.index)
    elif args.action == "run-all":
        result = run_all(args.output)
    else:
        result = aggregate(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
