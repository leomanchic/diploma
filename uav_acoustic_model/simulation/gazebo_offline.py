"""Read observed Gazebo model poses and expose a finite-support trajectory.

The input CSV is written in Gazebo PostUpdate from the ECM world pose. It is
never reconstructed from the prescribed command trajectory. Coordinates are
ENU metres; time is simulation seconds; quaternions are (w, x, y, z).
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from model.geometry import DEFAULT_SOUND_SPEED, tetrahedral_array
from model.station import StationPose

CONFIG_PATH = Path(__file__).resolve().parent / "gazebo_scene.json"
REQUIRED_COLUMNS = ("sim_time_s", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")
PX4_COLUMNS = REQUIRED_COLUMNS + ("vx_mps", "vy_mps", "vz_mps", "flight_phase")


def shared_scene_config(path: Path = CONFIG_PATH) -> dict:
    config = json.loads(Path(path).read_text())
    if config["schema_version"] != 1 or config["frame"] != "ENU":
        raise ValueError("unsupported Gazebo scene schema/frame")
    return config


def shared_stations(path: Path = CONFIG_PATH) -> tuple[StationPose, ...]:
    """Use the same station centres/orientations in SDF and acoustic Python."""

    stations = []
    for item in shared_scene_config(path)["stations"]:
        rotation = Rotation.from_euler("xyz", item["rpy_rad"]).as_matrix()
        stations.append(StationPose(item["id"], item["position_m"], rotation,
                                    tetrahedral_array()))
    return tuple(stations)


def _speed_limit(spline: CubicSpline, times: NDArray[np.float64]) -> float:
    """Exact speed supremum of each cubic position segment, including extrema."""

    maximum = 0.0
    for index, width in enumerate(np.diff(times)):
        # scipy PPoly uses descending powers of local (t - t_i).
        coeff = spline.c[:, index, :]
        velocity_polys = [np.array([coeff[2, axis], 2*coeff[1, axis],
                                    3*coeff[0, axis]]) for axis in range(3)]
        speed2 = np.zeros(5)
        for velocity in velocity_polys:
            square = np.polynomial.polynomial.polymul(velocity, velocity)
            speed2[:len(square)] += square
        derivative = np.polynomial.polynomial.polyder(speed2)
        roots = np.polynomial.polynomial.polyroots(derivative)
        candidates = [0.0, float(width)] + [float(root.real) for root in roots
                                            if abs(root.imag) < 1e-9 and 0 < root.real < width]
        maximum = max(maximum, *(float(np.sqrt(max(0.0,
            np.polynomial.polynomial.polyval(x, speed2)))) for x in candidates))
    return maximum


@dataclass(frozen=True)
class SampledTrajectory:
    knot_times_s: ArrayLike
    knot_positions_m: ArrayLike
    kind: str = "gazebo_recorded"
    sound_speed: float = DEFAULT_SOUND_SPEED
    extrapolate: bool = field(default=False, init=False)
    _spline: CubicSpline = field(init=False, repr=False)
    maximum_speed_mps: float = field(init=False)

    def __post_init__(self) -> None:
        times = np.asarray(self.knot_times_s, dtype=float)
        positions = np.asarray(self.knot_positions_m, dtype=float)
        if times.ndim != 1 or len(times) < 4 or not np.all(np.isfinite(times)):
            raise ValueError("at least four finite Gazebo timestamps are required")
        if np.any(np.diff(times) <= 0):
            raise ValueError("duplicate timestamp or simulation time reset")
        if positions.shape != (len(times), 3) or not np.all(np.isfinite(positions)):
            raise ValueError("position samples must be finite (K,3)")
        speed = float(self.sound_speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("sound_speed must be positive")
        spline = CubicSpline(times, positions, axis=0, extrapolate=False)
        maximum = _speed_limit(spline, times)
        if maximum >= speed:
            raise ValueError("interpolated trajectory is not subsonic")
        object.__setattr__(self, "knot_times_s", times.copy())
        object.__setattr__(self, "knot_positions_m", positions.copy())
        object.__setattr__(self, "_spline", spline)
        object.__setattr__(self, "maximum_speed_mps", maximum)

    def _evaluate(self, time_s: ArrayLike, derivative: int) -> NDArray[np.float64]:
        times = np.asarray(time_s, dtype=float)
        if np.any(~np.isfinite(times)) or np.any(times < self.knot_times_s[0]) or np.any(times > self.knot_times_s[-1]):
            raise ValueError("time_s lies outside recorded Gazebo support")
        return np.asarray(self._spline(times, nu=derivative), dtype=float)

    def q(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return self._evaluate(time_s, 0)

    def v(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return self._evaluate(time_s, 1)

    def a(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return self._evaluate(time_s, 2)


@dataclass(frozen=True)
class GazeboRecording:
    trajectory: SampledTrajectory
    quaternions_wxyz: NDArray[np.float64]
    manifest: dict
    csv_sha256: str
    maximum_time_gap_s: float
    world_velocities_mps: NDArray[np.float64] | None = None
    flight_phases: tuple[str, ...] | None = None


def load_gazebo_recording(directory: Path) -> GazeboRecording:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema_version"] not in (1, 2) or not manifest["frame"].startswith("ENU"):
        raise ValueError("unsupported Gazebo recording frame/schema")
    path = directory / "gazebo_state.csv"
    raw = path.read_bytes()
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = tuple(reader.fieldnames or ())
        if columns not in (REQUIRED_COLUMNS, PX4_COLUMNS):
            raise ValueError("unexpected Gazebo state columns")
        rows = list(reader)
    values = np.asarray([[float(row[key]) for key in REQUIRED_COLUMNS] for row in rows])
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
        raise ValueError("missing/nonfinite Gazebo pose samples")
    times = values[:, 0]
    gaps = np.diff(times)
    if np.any(gaps <= 0):
        raise ValueError("duplicate timestamp or simulation time reset")
    period = float(manifest["export_period_s"])
    # First sample may be at the first physics step rather than t=0.
    if np.any(gaps > 1.5 * period + 1e-8):
        raise ValueError("missing Gazebo export sample")
    if manifest["schema_version"] == 1:
        if times[0] > period + 1e-8 or times[-1] < float(manifest["duration_s"]) - period - 1e-8:
            raise ValueError("Gazebo export does not cover declared scene duration")
    else:
        if manifest["kind"] != "px4_flight" or columns != PX4_COLUMNS:
            raise ValueError("unsupported Gazebo flight recording schema")
        if (abs(times[0] - float(manifest["recording_start_s"])) > 1e-8
                or abs(times[-1] - float(manifest["recording_end_s"])) > 1e-8):
            raise ValueError("Gazebo export does not cover declared flight interval")
        for name, expected in manifest["artifact_sha256"].items():
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"PX4 flight artifact SHA-256 mismatch: {name}")
    quaternions = values[:, 4:8]
    if np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0)) > 1e-5:
        raise ValueError("invalid Gazebo orientation quaternion")
    trajectory = SampledTrajectory(times, values[:, 1:4], kind=manifest["kind"])
    velocities = None
    phases = None
    if columns == PX4_COLUMNS:
        velocities = np.asarray([[float(row[key]) for key in PX4_COLUMNS[8:11]]
                                 for row in rows], dtype=float)
        if not np.all(np.isfinite(velocities)):
            raise ValueError("nonfinite Gazebo world velocity")
        phases = tuple(row["flight_phase"] for row in rows)
        if any(not phase or not phase.replace("_", "").isalnum() for phase in phases):
            raise ValueError("invalid Gazebo flight phase marker")
    return GazeboRecording(trajectory, quaternions, manifest,
                           hashlib.sha256(raw).hexdigest(), float(gaps.max()), velocities, phases)
