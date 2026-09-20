"""Generate an SDF world from the shared station/motion configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT.parent / "simulation" / "gazebo_scene.json"


def _element(parent: ET.Element, tag: str, value: object | None = None, **attributes: str) -> ET.Element:
    child = ET.SubElement(parent, tag, attributes)
    if value is not None:
        child.text = str(value)
    return child


def _box_model(world: ET.Element, name: str, position: list[float], rpy: list[float],
               size: str, color: str) -> None:
    model = _element(world, "model", name=name)
    _element(model, "static", "true")
    _element(model, "pose", " ".join(map(str, [*position, *rpy])))
    link = _element(model, "link", name="body")
    visual = _element(link, "visual", name="visible")
    geometry = _element(visual, "geometry")
    _element(_element(geometry, "box"), "size", size)
    material = _element(visual, "material")
    _element(material, "ambient", color)
    _element(material, "diffuse", color)


def create_scene(kind: str, output_directory: Path, config_path: Path = CONFIG_PATH,
                 export_period_s: float | None = None) -> Path:
    if kind not in {"constant_velocity", "smooth_turn"}:
        raise ValueError("kind must be constant_velocity or smooth_turn")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    period = float(config["export_period_s"] if export_period_s is None else export_period_s)
    step = float(config["physics_step_s"])
    if period < step or abs(period / step - round(period / step)) > 1e-9:
        raise ValueError("export period must be an integral multiple of physics step")
    output_directory = output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"Gazebo scene destination is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    sdf = ET.Element("sdf", version="1.9")
    world = _element(sdf, "world", name="gazebo_offline")
    physics = _element(world, "physics", name="fixed_step", type="ignored")
    _element(physics, "max_step_size", step)
    _element(physics, "real_time_factor", 1.0)
    for filename, name in (
        ("gz-sim-physics-system", "gz::sim::systems::Physics"),
        ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands"),
        ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster"),
    ):
        _element(world, "plugin", filename=filename, name=name)
    light = _element(world, "light", type="directional", name="sun")
    _element(light, "pose", "0 0 100 0 0 0")
    _element(light, "direction", "-0.5 0.1 -0.9")
    _box_model(world, "ground", [50, 45, -3.2], [0, 0, 0], "160 160 0.2", "0.5 0.55 0.5 1")
    for station in config["stations"]:
        _box_model(world, station["id"], station["position_m"], station["rpy_rad"],
                   "1 1 0.5", "0.1 0.25 0.9 1")
    source = config["source"]
    model = _element(world, "model", name="sound_source")
    _element(model, "static", "false")
    _element(model, "pose", " ".join(map(str, [*source["initial_position_m"], 0, 0, 0])))
    link = _element(model, "link", name="body")
    _element(link, "gravity", "false")
    inertial = _element(link, "inertial")
    _element(inertial, "mass", 1)
    inertia = _element(inertial, "inertia")
    for key, value in {"ixx": 0.4, "iyy": 0.4, "izz": 0.4,
                       "ixy": 0, "ixz": 0, "iyz": 0}.items():
        _element(inertia, key, value)
    for tag in ("visual", "collision"):
        item = _element(link, tag, name=tag)
        geom = _element(item, "geometry")
        _element(_element(geom, "sphere"), "radius", 1.0)
        if tag == "visual":
            material = _element(item, "material")
            _element(material, "ambient", "1 0.35 0.05 1")
            _element(material, "diffuse", "1 0.35 0.05 1")
    plugin = _element(model, "plugin", filename="libgazebo_offline_motion.so",
                      name="offline::MotionExport")
    settings = {
        "kind": kind,
        "initial_position_m": " ".join(map(str, source["initial_position_m"])),
        "initial_velocity_mps": " ".join(map(str, source["initial_velocity_mps"])),
        "turn_start_s": source["turn_start_s"],
        "turn_end_s": source["turn_end_s"],
        "turn_angle_rad": source["turn_angle_rad"],
        "export_period_s": period,
        "output_csv": str(output_directory / "gazebo_state.csv"),
    }
    for key, value in settings.items():
        _element(plugin, key, value)
    scene_path = output_directory / "scene.sdf"
    ET.indent(sdf)
    ET.ElementTree(sdf).write(scene_path, encoding="utf-8", xml_declaration=True)
    try:
        version = subprocess.check_output(["gz", "sim", "--versions"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        version = "unavailable"
    manifest = {
        "schema_version": 1,
        "scene_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "scene_sdf_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "exporter_source_sha256": hashlib.sha256((ROOT / "motion_export.cc").read_bytes()).hexdigest(),
        "gazebo_sim_version": version,
        "ubuntu": platform.freedesktop_os_release().get("PRETTY_NAME", "unknown"),
        "kind": kind,
        "frame": "ENU x=East y=North z=Up, metres, seconds",
        "quaternion_order": "w,x,y,z",
        "output_state": "post-physics Gazebo ECM world pose of sound_source",
        "velocity": "not exported; interpolator derivatives are inferred from observed positions",
        "physics_step_s": step,
        "export_period_s": period,
        "duration_s": config["duration_s"],
        "gazebo_seed": config["seed"],
        "audio_seed": config["seed"],
        "station_config": config["stations"],
        "source_command": source,
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return scene_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("constant_velocity", "smooth_turn"))
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--export-period-s", type=float)
    args = parser.parse_args()
    print(create_scene(args.kind, args.output_directory,
                       export_period_s=args.export_period_s))
