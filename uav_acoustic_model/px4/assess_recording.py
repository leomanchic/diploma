"""Numerically assess flight export cadence against Gazebo world velocity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulation.gazebo_offline import SampledTrajectory, load_gazebo_recording


def assess(directory: Path) -> dict:
    directory = Path(directory)
    recording = load_gazebo_recording(directory)
    if recording.manifest["kind"] != "px4_flight" or recording.world_velocities_mps is None:
        raise ValueError("assessment requires an observed PX4 flight with world velocity")
    times = recording.trajectory.knot_times_s
    reference = recording.world_velocities_mps
    full = recording.trajectory.v(times)
    coarse = SampledTrajectory(times[::2], recording.trajectory.knot_positions_m[::2],
                               kind="px4_flight_25hz")
    # Restrict comparison to the common support; 50 Hz samples are the only
    # independently observed velocity reference in this single flight.
    common = times <= coarse.knot_times_s[-1]
    coarse_velocity = coarse.v(times[common])
    fine_error = np.linalg.norm(full[common] - reference[common], axis=1)
    coarse_error = np.linalg.norm(coarse_velocity - reference[common], axis=1)
    acceleration = recording.trajectory.a(times)
    report = {
        "schema_version": 1,
        "recording_sha256": recording.csv_sha256,
        "export_rate_hz": 1/recording.manifest["export_period_s"],
        "maximum_gap_s": recording.maximum_time_gap_s,
        "maximum_observed_speed_mps": float(np.max(np.linalg.norm(reference, axis=1))),
        "maximum_interpolated_speed_mps": recording.trajectory.maximum_speed_mps,
        "maximum_interpolated_acceleration_mps2": float(np.max(np.linalg.norm(acceleration, axis=1))),
        "velocity_error_rms_50hz_mps": float(np.sqrt(np.mean(fine_error**2))),
        "velocity_error_p95_50hz_mps": float(np.percentile(fine_error, 95)),
        "velocity_error_rms_25hz_mps": float(np.sqrt(np.mean(coarse_error**2))),
        "comparison_sample_count": int(common.sum()),
        "cadence_assessment": "50 Hz derivative compared with observed Gazebo velocity; 25 Hz is a downsample of the same flight",
    }
    (directory / "export_assessment.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(assess(args.directory), indent=2))
