"""Validate a completed PX4/Gazebo flight and freeze its acoustic input plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_PROCESSING = ROOT / "gazebo" / "processing_config.json"
STATE_COLUMNS = ("sim_time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz",
                 "vx_mps", "vy_mps", "vz_mps", "flight_phase")
PHASE_ORDER = ("preflight", "takeoff", "hover_before", "straight", "turn",
               "hover_after", "land", "landed")
REPORT_PHASES = ("hover_before", "straight", "turn", "hover_after")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _system_name() -> str:
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME", "unknown")
    except OSError:
        return platform.platform()


def phase_intervals(rows: list[dict], period_s: float) -> list[dict]:
    if not rows:
        raise ValueError("empty X500 flight recording")
    changes: list[tuple[str, float]] = []
    for row in rows:
        phase = row["flight_phase"]
        if not changes or phase != changes[-1][0]:
            changes.append((phase, float(row["sim_time_s"])))
    names = [name for name, _ in changes]
    if names != list(PHASE_ORDER) and names != list(PHASE_ORDER[1:]):
        raise ValueError(f"flight phases incomplete or out of order: {names}")
    end = float(rows[-1]["sim_time_s"]) + period_s
    return [{"name": name, "start_s": start,
             "end_s": changes[index+1][1] if index+1 < len(changes) else end}
            for index, (name, start) in enumerate(changes)]


def finalize(directory: Path, *, px4_root: Path = Path.home() / "projects/PX4-Autopilot") -> dict:
    directory = Path(directory).resolve()
    if (directory / "manifest.json").exists() or (directory / "experiment.json").exists():
        raise FileExistsError("flight already finalized; use a new recording directory")
    plan = json.loads((directory / "flight_plan.json").read_text())
    preflight = json.loads((directory / "preflight.json").read_text())
    if (_sha(directory / "flight_plan.json") != preflight["flight_plan_sha256"]
            or _sha(directory / "scene.sdf") != preflight["scene_sdf_sha256"]):
        raise ValueError("flight plan or Gazebo scene changed after creation")
    if preflight["observer_source_sha256"] != _sha(Path(__file__).with_name("observe_x500.cc")):
        raise ValueError("X500 observer source changed since scene creation")
    for name, expected in preflight["pilot_source_sha256"].items():
        if expected != _sha(Path(__file__).with_name(name)):
            raise ValueError(f"PX4 pilot source changed since scene creation: {name}")
    status = json.loads((directory / "flight_program_status.json").read_text())
    if not status["completed"]:
        raise ValueError("PX4 flight did not complete landing")
    if status.get("mavsdk_version") != "3.10.0":
        raise ValueError("flight controller did not record the pinned MAVSDK-Python version")
    state = directory / "gazebo_state.csv"
    with state.open(newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != STATE_COLUMNS:
            raise ValueError("unexpected X500 observer CSV columns")
        rows = list(reader)
    if len(rows) < 100:
        raise ValueError("X500 flight recording is too short")
    numeric = np.asarray([[float(row[column]) for column in STATE_COLUMNS[:-1]]
                          for row in rows], dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("nonfinite X500 state sample")
    times = numeric[:, 0]
    period = float(plan["gazebo"]["export_period_s"])
    gaps = np.diff(times)
    if np.any(gaps <= 0) or np.max(gaps) > 1.5*period + 1e-8:
        raise ValueError("Gazebo simulation time reset, duplicate or missing sample")
    if np.max(np.abs(np.linalg.norm(numeric[:, 4:8], axis=1)-1)) > 1e-5:
        raise ValueError("invalid X500 Gazebo quaternion")
    if np.max(np.linalg.norm(numeric[:, 8:11], axis=1)) >= 343.0:
        raise ValueError("X500 ground-truth speed is not subsonic")
    intervals = phase_intervals(rows, period)
    phases = {item["name"]: item for item in intervals}
    for name in REPORT_PHASES:
        if phases[name]["end_s"] - phases[name]["start_s"] < 1.0:
            raise ValueError(f"flight phase {name} is too short")
    for name, plan_key in (("hover_before", "hover_before_s"), ("straight", "straight_s"),
                           ("turn", "turn_s"), ("hover_after", "hover_after_s")):
        observed = phases[name]["end_s"] - phases[name]["start_s"]
        requested = float(plan["vehicle"][plan_key])
        if abs(observed-requested) > max(0.25, 3*period):
            raise ValueError(f"flight phase {name} duration disagrees with the saved program")
    if numeric[:, 3].max() < plan["vehicle"]["takeoff_reached_altitude_m"]:
        raise ValueError("recording does not contain the intended PX4 takeoff")
    if abs(numeric[-1, 3] - plan["ground_height_m"]) > 0.7:
        raise ValueError("recording does not end near the ground")
    launch_path = directory / "gazebo_launch.json"
    applied_seed = None
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        applied_seed = launch.get("seed_applied")
        if (applied_seed != plan["gazebo"]["seed"]
                or Path(launch["argv"][-1]).resolve() != (directory / "scene.sdf")):
            raise ValueError("Gazebo launch seed or scene disagrees with the saved flight plan")
    artifacts = {name: _sha(directory / name) for name in (
        "flight_plan.json", "preflight.json", "scene.sdf", "flight_commands.csv",
        "planned_route.csv", "flight_program_status.json", "autopilot_parameters.json")}
    if launch_path.exists():
        artifacts["gazebo_launch.json"] = _sha(launch_path)
    if "px4_build_patch_sha256" in preflight:
        name = "px4-v1.17.0-minimal-sitl.patch"
        artifacts[name] = _sha(directory / name)
        if artifacts[name] != preflight["px4_build_patch_sha256"]:
            raise ValueError("PX4 build patch changed after scene creation")
    px4_root = Path(px4_root)
    px4_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=px4_root, text=True).strip()
    if px4_sha != preflight["px4_git_sha"]:
        raise ValueError("PX4 checkout changed since scene creation")
    manifest = {
        "schema_version": 2,
        "kind": "px4_flight",
        "frame": "ENU x=East y=North z=Up, metres, seconds",
        "quaternion_order": "w,x,y,z",
        "body_frame": "Gazebo base_link FLU; PX4 body FRD",
        "world_ned_from_enu": [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        "body_frd_from_flu": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        "source_point": plan["vehicle"]["source_point"],
        "source_offset_body_flu_m": [0, 0, 0],
        "output_state": "post-physics Gazebo ECM world pose and world velocity of canonical X500 base_link",
        "velocity": "Gazebo WorldLinearVelocity at base_link origin, m/s in ENU",
        "recording_start_s": float(times[0]),
        "recording_end_s": float(times[-1]),
        "duration_s": float(times[-1]-times[0]),
        "export_period_s": period,
        "physics_step_s": float(plan["gazebo"]["physics_step_s"]),
        "maximum_export_gap_s": float(gaps.max()),
        "station_config": plan["stations"],
        "phase_intervals": intervals,
        "flight_program": plan["vehicle"],
        "gazebo_seed_requested": plan["gazebo"]["seed"],
        "gazebo_seed_applied": applied_seed,
        "audio_seed": plan["acoustics"]["base_seed"],
        "px4_git_sha": px4_sha,
        "px4_gz_submodule_sha": preflight["px4_gz_submodule_sha"],
        "px4_submodule_status": preflight["px4_submodule_status"],
        "px4_build_patch_sha256": preflight.get("px4_build_patch_sha256"),
        "px4_build_patch_applied": preflight.get("px4_build_patch_applied"),
        "px4_x500_model_sdf_sha256": preflight["px4_x500_model_sdf_sha256"],
        "observer_source_sha256": preflight["observer_source_sha256"],
        "pilot_source_sha256": preflight["pilot_source_sha256"],
        "acoustic_git_sha_at_scene_creation": preflight["acoustic_git_sha_at_creation"],
        "gazebo_sim_version": subprocess.check_output(["gz", "sim", "--versions"], text=True).strip(),
        "ubuntu": _system_name(),
        "mavsdk_python_version": "3.10.0",
        "controller_runtime_versions": {
            key: status[key] for key in ("python_version", "mavsdk_version",
                                        "grpcio_version", "protobuf_version")
        },
        "artifact_sha256": artifacts,
        "physical_replay_claim": "same saved recording is reproducible; a new flight is not claimed byte-identical",
    }
    if "pymavlink_version" in status:
        manifest["controller_runtime_versions"]["pymavlink_version"] = status["pymavlink_version"]
    # Prepare the explicit processing input from the flight plan, before any
    # acoustic result exists. init freezes the completed JSON in experiment.json.
    config = json.loads(BASE_PROCESSING.read_text())
    stations = {item["id"]: item for item in plan["stations"]}
    for station in config["processing"]["stations"]:
        station.update(stations[station["id"]])
    audio = config["processing"]["audio"]
    audio["base_seed"] = plan["acoustics"]["base_seed"]
    audio["snr_db"] = plan["acoustics"]["snr_db"]
    start = phases["hover_before"]["start_s"] - plan["acoustics"]["reception_lead_s"]
    end = phases["hover_after"]["end_s"] + plan["acoustics"]["reception_tail_s"]
    if start <= times[0] + 0.5 or end >= times[-1] - 0.5:
        raise ValueError("recording lacks acoustic propagation history or tail")
    audio["reception_start_s"] = start
    audio["duration_s"] = end-start
    config["processing"]["evaluation_phases"] = [phases[name] for name in REPORT_PHASES]
    tracker = config["processing"]["tracker"]
    tracker["manoeuvre_start_s"] = phases["turn"]["start_s"]
    tracker["manoeuvre_end_s"] = phases["turn"]["end_s"]
    (directory / "processing_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest["artifact_sha256"]["processing_config.json"] = _sha(directory / "processing_config.json")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--px4-root", type=Path, default=Path.home() / "projects/PX4-Autopilot")
    args = parser.parse_args()
    print(json.dumps(finalize(args.directory, px4_root=args.px4_root), indent=2))
