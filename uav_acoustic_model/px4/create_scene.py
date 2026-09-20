"""Create a fresh PX4 world with three stations and a read-only X500 observer."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = Path(__file__).with_name("flight_plan.json")
PX4_ROOT = Path.home() / "projects" / "PX4-Autopilot"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _element(parent: ET.Element, tag: str, value: object | None = None, **attributes: str) -> ET.Element:
    element = ET.SubElement(parent, tag, attributes)
    if value is not None:
        element.text = str(value)
    return element


def create_scene(destination: Path, *, plan_path: Path = DEFAULT_PLAN,
                 px4_root: Path = PX4_ROOT) -> Path:
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"new PX4 recording destination must be empty: {destination}")
    plan_path, px4_root = Path(plan_path).resolve(), Path(px4_root).resolve()
    plan = json.loads(plan_path.read_text())
    if plan["schema_version"] != 1 or plan["world_frame"] != "ENU":
        raise ValueError("unsupported PX4 flight plan")
    station_xy = []
    for station in plan["stations"]:
        x, y, z = station["position_m"]
        if z <= plan["ground_height_m"] + 0.25:
            raise ValueError("station array centroid must be above ground")
        station_xy.append((x, y))
    sx, sy, sz = plan["vehicle"]["spawn_enu_m"]
    if sz != plan["ground_height_m"] or min(
        ((sx-x)**2 + (sy-y)**2)**0.5 for x, y in station_xy
    ) < 10:
        raise ValueError("X500 needs a clear takeoff site on the ground")
    step, period = (float(plan["gazebo"][key]) for key in ("physics_step_s", "export_period_s"))
    if period < step or abs(period / step - round(period / step)) > 1e-8:
        raise ValueError("export period must be an integral multiple of physics step")
    source_world = px4_root / "Tools/simulation/gz/worlds/default.sdf"
    model_sdf = px4_root / "Tools/simulation/gz/models/x500/model.sdf"
    server_config = px4_root / "src/modules/simulation/gz_bridge/server.config"
    world = ET.parse(source_world).getroot().find("world")
    assert world is not None
    world.set("name", plan["world_name"])
    physics = world.find("physics")
    assert physics is not None
    physics.find("max_step_size").text = str(step)
    physics.find("real_time_update_rate").text = str(round(1 / step))
    # PX4's server.config has the sensor systems required by X500. A world
    # with an explicit plugin list must carry them itself.
    for item in ET.parse(server_config).getroot().findall("./plugins/plugin"):
        if item.get("name") in {"custom::OpticalFlowSystem", "custom::GstCameraSystem"}:
            continue  # X500 has no optical-flow or camera sensor.
        item.attrib.pop("entity_name", None)
        item.attrib.pop("entity_type", None)
        world.append(item)
    for station in plan["stations"]:
        name = station["id"]
        model = _element(world, "model", name=name)
        _element(model, "static", "true")
        _element(model, "pose", " ".join(map(str, [*station["position_m"], *station["rpy_rad"]])))
        link = _element(model, "link", name="array_center")
        visual = _element(link, "visual", name="station")
        geometry = _element(visual, "geometry")
        _element(_element(geometry, "box"), "size", "1 1 0.5")
        material = _element(visual, "material")
        _element(material, "ambient", "0.2 0.3 0.95 1")
        _element(material, "diffuse", "0.2 0.3 0.95 1")
    observer = _element(world, "plugin", filename="libgazebo_px4_observer.so",
                        name="acoustic::X500Observer")
    _element(observer, "model_name", plan["vehicle"]["gazebo_model_name"])
    _element(observer, "output_csv", destination / "gazebo_state.csv")
    _element(observer, "phase_file", destination / "flight_phase.txt")
    _element(observer, "export_period_s", period)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "flight_plan.json").write_bytes(plan_path.read_bytes())
    (destination / "flight_phase.txt").write_text("preflight\n")
    build_patch = Path(__file__).with_name("px4-v1.17.0-minimal-sitl.patch")
    shutil.copy2(build_patch, destination / build_patch.name)
    sdf = ET.Element("sdf", version="1.9")
    sdf.append(world)
    ET.indent(sdf)
    scene = destination / "scene.sdf"
    ET.ElementTree(sdf).write(scene, encoding="utf-8", xml_declaration=True)
    provenance = {
        "schema_version": 1,
        "flight_plan_sha256": _sha(destination / "flight_plan.json"),
        "scene_sdf_sha256": _sha(scene),
        "px4_git_sha": _git_sha(px4_root),
        "px4_gz_submodule_sha": _git_sha(px4_root / "Tools/simulation/gz"),
        "px4_submodule_status": subprocess.check_output(
            ["git", "submodule", "status", "--recursive"], cwd=px4_root, text=True).splitlines(),
        "px4_x500_model_sdf_sha256": _sha(model_sdf),
        "px4_default_world_sha256": _sha(source_world),
        "px4_server_config_sha256": _sha(server_config),
        "px4_build_patch_sha256": _sha(destination / build_patch.name),
        "px4_build_patch_applied": (subprocess.check_output(
            ["git", "diff", "--binary"], cwd=px4_root) == build_patch.read_bytes()
            if (px4_root / ".git").exists() else None),
        "observer_source_sha256": _sha(Path(__file__).with_name("observe_x500.cc")),
        "pilot_source_sha256": {
            path.name: _sha(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "acoustic_git_sha_at_creation": _git_sha(ROOT),
    }
    (destination / "preflight.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--px4-root", type=Path, default=PX4_ROOT)
    args = parser.parse_args()
    print(create_scene(args.destination, plan_path=args.plan, px4_root=args.px4_root))
