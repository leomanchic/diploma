"""Subsonic source trajectories in SI units.

Every trajectory exposes ``q(t)``, ``v(t)``, and ``a(t)``. Positions are in
metres, time is in seconds, velocity is in metres per second, and acceleration
is in metres per second squared. Scalar time returns shape ``(3,)``; array time
returns ``time.shape + (3,)``. Every constructor rejects speed ``>= c``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from model.geometry import DEFAULT_SOUND_SPEED


def _vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return vector.copy()


def _sound_speed(value: float) -> float:
    speed = float(value)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")
    return speed


def _subsonic(speed: ArrayLike, sound_speed: float) -> None:
    values = np.asarray(speed, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values >= sound_speed):
        raise ValueError("trajectory speed must satisfy |v| < sound_speed")


def _time_array(time_s: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    values = np.asarray(time_s, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("time_s must be finite")
    return values, values.ndim == 0


def _finish(values: NDArray[np.float64], scalar: bool) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float).reshape(3) if scalar else np.asarray(values)


@runtime_checkable
class Trajectory(Protocol):
    """Structural interface used by the moving-source solver."""

    sound_speed: float

    def q(self, time_s: ArrayLike) -> NDArray[np.float64]: ...

    def v(self, time_s: ArrayLike) -> NDArray[np.float64]: ...

    def a(self, time_s: ArrayLike) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class StationaryTrajectory:
    position_m: ArrayLike
    sound_speed: float = DEFAULT_SOUND_SPEED

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vector3(self.position_m, "position_m"))
        object.__setattr__(self, "sound_speed", _sound_speed(self.sound_speed))

    def q(self, time_s: ArrayLike) -> NDArray[np.float64]:
        times, scalar = _time_array(time_s)
        values = np.broadcast_to(self.position_m, times.shape + (3,)).copy()
        return _finish(values, scalar)

    def v(self, time_s: ArrayLike) -> NDArray[np.float64]:
        times, scalar = _time_array(time_s)
        return _finish(np.zeros(times.shape + (3,)), scalar)

    def a(self, time_s: ArrayLike) -> NDArray[np.float64]:
        return self.v(time_s)


@dataclass(frozen=True)
class ConstantVelocityTrajectory:
    position_at_reference_m: ArrayLike
    velocity_mps: ArrayLike
    reference_time_s: float = 0.0
    sound_speed: float = DEFAULT_SOUND_SPEED

    def __post_init__(self) -> None:
        position = _vector3(self.position_at_reference_m, "position_at_reference_m")
        velocity = _vector3(self.velocity_mps, "velocity_mps")
        reference = float(self.reference_time_s)
        speed = _sound_speed(self.sound_speed)
        if not np.isfinite(reference):
            raise ValueError("reference_time_s must be finite")
        _subsonic(np.linalg.norm(velocity), speed)
        object.__setattr__(self, "position_at_reference_m", position)
        object.__setattr__(self, "velocity_mps", velocity)
        object.__setattr__(self, "reference_time_s", reference)
        object.__setattr__(self, "sound_speed", speed)

    def q(self, time_s: ArrayLike) -> NDArray[np.float64]:
        times, scalar = _time_array(time_s)
        values = self.position_at_reference_m + (
            times[..., None] - self.reference_time_s
        ) * self.velocity_mps
        return _finish(values, scalar)

    def v(self, time_s: ArrayLike) -> NDArray[np.float64]:
        times, scalar = _time_array(time_s)
        values = np.broadcast_to(self.velocity_mps, times.shape + (3,)).copy()
        return _finish(values, scalar)

    def a(self, time_s: ArrayLike) -> NDArray[np.float64]:
        times, scalar = _time_array(time_s)
        return _finish(np.zeros(times.shape + (3,)), scalar)


@dataclass(frozen=True)
class CircularTrajectory:
    center_m: ArrayLike
    radius_m: float
    angular_speed_rad_s: float
    plane_normal: ArrayLike = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    reference_direction: ArrayLike | None = None
    phase_at_reference_rad: float = 0.0
    reference_time_s: float = 0.0
    sound_speed: float = DEFAULT_SOUND_SPEED
    _axis_1: NDArray[np.float64] = field(init=False, repr=False)
    _axis_2: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        center = _vector3(self.center_m, "center_m")
        normal = _vector3(self.plane_normal, "plane_normal")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm == 0.0:
            raise ValueError("plane_normal must be non-zero")
        normal /= normal_norm
        radius = float(self.radius_m)
        omega = float(self.angular_speed_rad_s)
        phase = float(self.phase_at_reference_rad)
        reference_time = float(self.reference_time_s)
        speed = _sound_speed(self.sound_speed)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_m must be finite and positive")
        if not all(np.isfinite(value) for value in (omega, phase, reference_time)):
            raise ValueError("circular trajectory parameters must be finite")
        _subsonic(abs(radius * omega), speed)
        if self.reference_direction is None:
            seed = np.array([1.0, 0.0, 0.0])
            if abs(float(seed @ normal)) > 0.9:
                seed = np.array([0.0, 1.0, 0.0])
        else:
            seed = _vector3(self.reference_direction, "reference_direction")
        axis_1 = seed - float(seed @ normal) * normal
        axis_norm = float(np.linalg.norm(axis_1))
        if axis_norm < 1e-14:
            raise ValueError("reference_direction must not be parallel to plane_normal")
        axis_1 /= axis_norm
        axis_2 = np.cross(normal, axis_1)
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "plane_normal", normal)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "angular_speed_rad_s", omega)
        object.__setattr__(self, "phase_at_reference_rad", phase)
        object.__setattr__(self, "reference_time_s", reference_time)
        object.__setattr__(self, "sound_speed", speed)
        object.__setattr__(self, "_axis_1", axis_1)
        object.__setattr__(self, "_axis_2", axis_2)

    def _phase(self, time_s: ArrayLike) -> tuple[NDArray[np.float64], bool]:
        times, scalar = _time_array(time_s)
        return (
            self.phase_at_reference_rad
            + self.angular_speed_rad_s * (times - self.reference_time_s),
            scalar,
        )

    def q(self, time_s: ArrayLike) -> NDArray[np.float64]:
        phase, scalar = self._phase(time_s)
        radial = np.cos(phase)[..., None] * self._axis_1 + np.sin(phase)[..., None] * self._axis_2
        return _finish(self.center_m + self.radius_m * radial, scalar)

    def v(self, time_s: ArrayLike) -> NDArray[np.float64]:
        phase, scalar = self._phase(time_s)
        tangent = -np.sin(phase)[..., None] * self._axis_1 + np.cos(phase)[..., None] * self._axis_2
        return _finish(self.radius_m * self.angular_speed_rad_s * tangent, scalar)

    def a(self, time_s: ArrayLike) -> NDArray[np.float64]:
        phase, scalar = self._phase(time_s)
        radial = np.cos(phase)[..., None] * self._axis_1 + np.sin(phase)[..., None] * self._axis_2
        return _finish(-self.radius_m * self.angular_speed_rad_s**2 * radial, scalar)


@dataclass(frozen=True)
class PiecewiseLinearTrajectory:
    knot_times_s: ArrayLike
    knot_positions_m: ArrayLike
    extrapolate: bool = False
    sound_speed: float = DEFAULT_SOUND_SPEED
    _segment_velocities: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        times = np.asarray(self.knot_times_s, dtype=float)
        positions = np.asarray(self.knot_positions_m, dtype=float)
        speed = _sound_speed(self.sound_speed)
        if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
            raise ValueError("knot_times_s must contain at least two finite times")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("knot_times_s must be strictly increasing")
        if positions.shape != (times.size, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("knot_positions_m must have shape (K, 3) and be finite")
        velocities = np.diff(positions, axis=0) / np.diff(times)[:, None]
        _subsonic(np.linalg.norm(velocities, axis=1), speed)
        object.__setattr__(self, "knot_times_s", times.copy())
        object.__setattr__(self, "knot_positions_m", positions.copy())
        object.__setattr__(self, "sound_speed", speed)
        object.__setattr__(self, "_segment_velocities", velocities)

    def _indices(self, times: NDArray[np.float64]) -> NDArray[np.intp]:
        if not self.extrapolate and (
            np.any(times < self.knot_times_s[0]) or np.any(times > self.knot_times_s[-1])
        ):
            raise ValueError("time_s lies outside piecewise trajectory support")
        return np.clip(
            np.searchsorted(self.knot_times_s, times, side="right") - 1,
            0,
            self.knot_times_s.size - 2,
        )

    def q(self, time_s: ArrayLike) -> NDArray[np.float64]:
        times, scalar = _time_array(time_s)
        indices = self._indices(times)
        values = self.knot_positions_m[indices] + (
            times[..., None] - self.knot_times_s[indices][..., None]
        ) * self._segment_velocities[indices]
        return _finish(values, scalar)

    def v(self, time_s: ArrayLike) -> NDArray[np.float64]:
        times, scalar = _time_array(time_s)
        values = self._segment_velocities[self._indices(times)]
        return _finish(values, scalar)

    def a(self, time_s: ArrayLike) -> NDArray[np.float64]:
        """Return zero between knots; velocity impulses at knots are omitted."""

        times, scalar = _time_array(time_s)
        self._indices(times)
        return _finish(np.zeros(times.shape + (3,)), scalar)


__all__ = [
    "CircularTrajectory",
    "ConstantVelocityTrajectory",
    "PiecewiseLinearTrajectory",
    "StationaryTrajectory",
    "Trajectory",
]
