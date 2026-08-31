"""Microphone-station poses in a right-handed ENU world frame.

World coordinates use ``x=East``, ``y=North`` and ``z=Up``.  A station
position is the world position of the microphone-array centroid, while every
stored microphone coordinate is expressed relative to that centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from model.geometry import microphone_positions


def _readonly_copy(value: ArrayLike, shape: tuple[int, ...], *, name: str) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    result.setflags(write=False)
    return result


def _validate_rotation(value: ArrayLike) -> NDArray[np.float64]:
    rotation = _readonly_copy(value, (3, 3), name="rotation_local_to_world")
    tolerance = 256.0 * np.finfo(float).eps
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=tolerance):
        raise ValueError("rotation_local_to_world must be orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("rotation_local_to_world must have determinant +1")
    return rotation


def _points(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.shape[-1:] != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with trailing shape (3,)")
    return result


@dataclass(frozen=True, slots=True)
class StationPose:
    """Immutable pose and local microphone geometry of one station.

    Lengths are metres, clock quantities are seconds, and the rotation maps
    local column vectors into the common ENU world frame.  Clock metadata is
    carried for the future asynchronous dynamic stage and is not used by the
    static triangulator.
    """

    station_id: str
    position_world_m: NDArray[np.float64]
    rotation_local_to_world: NDArray[np.float64]
    microphone_positions_local_m: NDArray[np.float64]
    clock_offset_s: float | None = None
    clock_drift_s_per_s: float | None = None

    def __post_init__(self) -> None:
        station_id = str(self.station_id)
        if not station_id:
            raise ValueError("station_id must be a non-empty string")
        position = _readonly_copy(self.position_world_m, (3,), name="position_world_m")
        rotation = _validate_rotation(self.rotation_local_to_world)
        microphones = np.array(
            microphone_positions(self.microphone_positions_local_m), dtype=float, copy=True
        )
        centroid = np.mean(microphones, axis=0)
        coordinate_scale = max(1.0, float(np.max(np.abs(microphones))))
        centroid_tolerance = 256.0 * np.finfo(float).eps * coordinate_scale
        if not np.allclose(centroid, 0.0, rtol=0.0, atol=centroid_tolerance):
            raise ValueError(
                "microphone_positions_local_m must be relative to their centroid"
            )
        microphones.setflags(write=False)
        for name, value in (
            ("clock_offset_s", self.clock_offset_s),
            ("clock_drift_s_per_s", self.clock_drift_s_per_s),
        ):
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when provided")
        object.__setattr__(self, "station_id", station_id)
        object.__setattr__(self, "position_world_m", position)
        object.__setattr__(self, "rotation_local_to_world", rotation)
        object.__setattr__(self, "microphone_positions_local_m", microphones)
        if self.clock_offset_s is not None:
            object.__setattr__(self, "clock_offset_s", float(self.clock_offset_s))
        if self.clock_drift_s_per_s is not None:
            object.__setattr__(
                self, "clock_drift_s_per_s", float(self.clock_drift_s_per_s)
            )

    @property
    def microphone_positions_world_m(self) -> NDArray[np.float64]:
        """Return world microphone coordinates, with shape ``(M, 3)``."""

        return self.local_to_world_points(self.microphone_positions_local_m)

    def local_to_world_points(self, points_local_m: ArrayLike) -> NDArray[np.float64]:
        """Transform one or many local points into ENU world coordinates."""

        points = _points(points_local_m, name="points_local_m")
        return points @ self.rotation_local_to_world.T + self.position_world_m

    def world_to_local_points(self, points_world_m: ArrayLike) -> NDArray[np.float64]:
        """Transform one or many ENU world points into station coordinates."""

        points = _points(points_world_m, name="points_world_m")
        return (points - self.position_world_m) @ self.rotation_local_to_world

    def local_to_world_direction(self, direction_local: ArrayLike) -> NDArray[np.float64]:
        """Rotate a local vector into ENU without applying translation."""

        direction = _points(direction_local, name="direction_local")
        return direction @ self.rotation_local_to_world.T

    def world_to_local_direction(self, direction_world: ArrayLike) -> NDArray[np.float64]:
        """Rotate an ENU vector into station coordinates."""

        direction = _points(direction_world, name="direction_world")
        return direction @ self.rotation_local_to_world


__all__ = ["StationPose"]
