"""Copy a finalized flight into a distinct short acoustic-memory probe."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from simulation.gazebo_offline import load_gazebo_recording


def create_probe(source: Path, destination: Path, duration_s: float = 1.0) -> Path:
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"short probe destination already exists: {destination}")
    recording = load_gazebo_recording(source)
    if recording.manifest["kind"] != "px4_flight" or duration_s <= 0:
        raise ValueError("short probe requires a finalized PX4 flight and positive duration")
    config = json.loads((source / "processing_config.json").read_text())
    audio = config["processing"]["audio"]
    if audio["reception_start_s"] + duration_s >= recording.trajectory.knot_times_s[-1] - 0.5:
        raise ValueError("short probe extends beyond recorded support")
    audio["duration_s"] = duration_s
    destination.mkdir(parents=True)
    for name in ("gazebo_state.csv", "manifest.json", *recording.manifest["artifact_sha256"]):
        shutil.copy2(source / name, destination / name)
    config_path = destination / "probe_processing_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--duration-s", type=float, default=1.0)
    args = parser.parse_args()
    print(create_probe(args.source, args.destination, args.duration_s))
