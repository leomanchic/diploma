"""Independent deterministic truth trajectories for the S7C-C benchmark.

These accelerations are prescribed functions of time.  They are not samples
from the filter's integrated-Wiener ``Qc`` model.  Each path starts with a
clean constant-velocity segment and has continuous position and velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from model.geometry import DEFAULT_SOUND_SPEED


@dataclass(frozen=True, slots=True)
class BenchmarkManoeuvreTrajectory:
    initial_position_m: ArrayLike
    initial_velocity_mps: ArrayLike
    kind: str = "constant_velocity"
    manoeuvre_start_s: float = 5.0
    manoeuvre_end_s: float = 9.0
    acceleration_mps2: ArrayLike = (0.0, 1.5, 0.35)
    turn_angle_rad: float = 0.65
    sound_speed: float = DEFAULT_SOUND_SPEED

    def __post_init__(self) -> None:
        q0 = np.asarray(self.initial_position_m, dtype=float)
        v0 = np.asarray(self.initial_velocity_mps, dtype=float)
        acceleration = np.asarray(self.acceleration_mps2, dtype=float)
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (q0, v0, acceleration)):
            raise ValueError("initial position, velocity and acceleration must be finite 3-vectors")
        if self.kind not in {"constant_velocity", "constant_acceleration_segment", "smooth_turn"}:
            raise ValueError("unsupported manoeuvre kind")
        start = float(self.manoeuvre_start_s)
        end = float(self.manoeuvre_end_s)
        speed = float(self.sound_speed)
        angle = float(self.turn_angle_rad)
        if not all(np.isfinite(value) for value in (start, end, speed, angle)) or end <= start or speed <= 0.0:
            raise ValueError("invalid manoeuvre interval or sound speed")
        maximum_speed = max(
            np.linalg.norm(v0),
            np.linalg.norm(v0 + acceleration * (end - start)),
        )
        if maximum_speed >= speed:
            raise ValueError("trajectory must remain subsonic")
        object.__setattr__(self, "initial_position_m", q0.copy())
        object.__setattr__(self, "initial_velocity_mps", v0.copy())
        object.__setattr__(self, "acceleration_mps2", acceleration.copy())
        object.__setattr__(self, "manoeuvre_start_s", start)
        object.__setattr__(self, "manoeuvre_end_s", end)
        object.__setattr__(self, "sound_speed", speed)
        object.__setattr__(self, "turn_angle_rad", angle)

    def _angle(self, t: float) -> tuple[float, float]:
        if t <= self.manoeuvre_start_s:
            return 0.0, 0.0
        duration = self.manoeuvre_end_s - self.manoeuvre_start_s
        if t >= self.manoeuvre_end_s:
            return self.turn_angle_rad, 0.0
        normalized = (t - self.manoeuvre_start_s) / duration
        angle = self.turn_angle_rad * (3 * normalized**2 - 2 * normalized**3)
        rate = self.turn_angle_rad * 6 * normalized * (1 - normalized) / duration
        return angle, rate

    def _v_scalar(self, t: float) -> NDArray[np.float64]:
        v0 = self.initial_velocity_mps
        if self.kind == "constant_velocity":
            return v0.copy()
        if self.kind == "constant_acceleration_segment":
            elapsed = np.clip(t - self.manoeuvre_start_s, 0.0, self.manoeuvre_end_s - self.manoeuvre_start_s)
            return v0 + elapsed * self.acceleration_mps2
        angle, _ = self._angle(t)
        cosine = np.cos(angle)
        sine = np.sin(angle)
        return np.asarray([
            cosine * v0[0] - sine * v0[1],
            sine * v0[0] + cosine * v0[1],
            v0[2],
        ])

    def _q_scalar(self, t: float) -> NDArray[np.float64]:
        q0 = self.initial_position_m
        v0 = self.initial_velocity_mps
        if self.kind == "constant_velocity" or t <= self.manoeuvre_start_s:
            return q0 + t * v0
        duration = self.manoeuvre_end_s - self.manoeuvre_start_s
        if self.kind == "constant_acceleration_segment":
            active = min(t - self.manoeuvre_start_s, duration)
            extra = 0.5 * active**2 * self.acceleration_mps2
            if t > self.manoeuvre_end_s:
                extra += duration * (t - self.manoeuvre_end_s) * self.acceleration_mps2
            return q0 + t * v0 + extra
        active_end = min(t, self.manoeuvre_end_s)
        initial = q0 + self.manoeuvre_start_s * v0
        integral = np.asarray([
            quad(lambda epoch: self._v_scalar(epoch)[axis], self.manoeuvre_start_s, active_end, epsabs=2e-11)[0]
            for axis in range(3)
        ])
        if t > self.manoeuvre_end_s:
            integral += (t - self.manoeuvre_end_s) * self._v_scalar(self.manoeuvre_end_s)
        return initial + integral

    def _apply(self, time_s: ArrayLike, method: str) -> NDArray[np.float64]:
        times = np.asarray(time_s, dtype=float)
        if np.any(~np.isfinite(times)):
            raise ValueError("time_s must be finite")
        scalar = times.ndim == 0
        flat = times.reshape(-1)
        result = np.vstack([getattr(self, method)(float(t)) for t in flat])
        return result[0] if scalar else result.reshape(times.shape + (3,))

    def q(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return self._apply(time_s, "_q_scalar")

    def v(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return self._apply(time_s, "_v_scalar")

    def a(self, time_s: ArrayLike) -> NDArray[np.float64]:
        times = np.asarray(time_s, dtype=float)
        if np.any(~np.isfinite(times)):
            raise ValueError("time_s must be finite")
        flat = times.reshape(-1)
        results = []
        for value in flat:
            t = float(value)
            if self.kind == "constant_acceleration_segment":
                acceleration = self.acceleration_mps2 if self.manoeuvre_start_s < t < self.manoeuvre_end_s else np.zeros(3)
            elif self.kind == "smooth_turn":
                _, rate = self._angle(t)
                velocity = self._v_scalar(t)
                acceleration = np.asarray([-rate * velocity[1], rate * velocity[0], 0.0])
            else:
                acceleration = np.zeros(3)
            results.append(acceleration)
        array = np.vstack(results)
        return array[0] if times.ndim == 0 else array.reshape(times.shape + (3,))


__all__ = ["BenchmarkManoeuvreTrajectory"]
