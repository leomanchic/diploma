"""Start a fresh PX4 acoustic world with a recorded Gazebo launch contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def launch(directory: Path, *, px4_root: Path, headless: bool = False) -> int:
    directory = Path(directory).resolve()
    px4_root = Path(px4_root).resolve()
    scene = directory / "scene.sdf"
    if not scene.is_file():
        raise FileNotFoundError(f"create the PX4 scene first: {scene}")
    if (directory / "gazebo_state.csv").exists() or (directory / "gazebo_launch.json").exists():
        raise FileExistsError("Gazebo flight already started here; use a new recording directory")
    plan = json.loads((directory / "flight_plan.json").read_text())
    seed = int(plan["gazebo"]["seed"])
    if seed < 0 or seed > 2**32-1:
        raise ValueError("Gazebo seed must be an unsigned 32-bit integer")
    resource_path = px4_root / "Tools/simulation/gz/models"
    observer_path = ROOT / "build/px4-observer"
    plugin_path = px4_root / "build/px4_sitl_default/src/modules/simulation/gz_plugins"
    for path in (resource_path, observer_path / "libgazebo_px4_observer.so", plugin_path):
        if not path.exists():
            raise FileNotFoundError(f"Gazebo runtime dependency missing: {path}")
    argv = ["gz", "sim", "-r", "-v", "2", "--seed", str(seed)]
    if headless:
        argv.append("-s")
    argv.append(str(scene))
    launch_environment = {
        "GZ_IP": "127.0.0.1",
        "GZ_SIM_RESOURCE_PATH": str(resource_path),
        "GZ_SIM_SYSTEM_PLUGIN_PATH": f"{observer_path}:{plugin_path}",
    }
    environment = dict(os.environ, **launch_environment)
    (directory / "gazebo_launch.json").write_text(json.dumps({
        "schema_version": 1,
        "argv": argv,
        "environment": launch_environment,
        "seed_applied": seed,
        "headless": headless,
    }, indent=2) + "\n")
    print("Starting:", " ".join(argv), flush=True)
    try:
        result = subprocess.run(argv, env=environment, check=False)
    except KeyboardInterrupt:
        print("Gazebo stopped by Ctrl+C", flush=True)
        return 130
    if result.returncode not in (0, -2, 130):
        raise RuntimeError(f"Gazebo exited with status {result.returncode}")
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_directory", type=Path)
    parser.add_argument("--px4-root", type=Path, default=Path.home() / "projects/PX4-Autopilot")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    raise SystemExit(launch(args.recording_directory, px4_root=args.px4_root,
                            headless=args.headless))
