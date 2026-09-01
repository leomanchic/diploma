"""Constant-velocity source state in a common ENU world frame.

The immutable state is referenced to ``reference_time_s`` and stores
``x=[q, v]`` in SI units.  Rebasing changes only the coordinate epoch, not
the represented physical trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _readonly_vector3(value: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ConstantVelocityState:
    """Position and velocity at one reference epoch.

    Position is in metres, velocity in metres per second and time in seconds.
    The sound-speed-dependent subsonic check is performed by the retarded-time
    measurement model, where ``c`` is known.
    """

    position_at_reference_world_m: NDArray[np.float64]
    velocity_world_mps: NDArray[np.float64]
    reference_time_s: float = 0.0

    def __post_init__(self) -> None:
        position = _readonly_vector3(
            self.position_at_reference_world_m,
            name="position_at_reference_world_m",
        )
        velocity = _readonly_vector3(
            self.velocity_world_mps,
            name="velocity_world_mps",
        )
        reference = float(self.reference_time_s)
        if not np.isfinite(reference):
            raise ValueError("reference_time_s must be finite")
        object.__setattr__(self, "position_at_reference_world_m", position)
        object.__setattr__(self, "velocity_world_mps", velocity)
        object.__setattr__(self, "reference_time_s", reference)

    @property
    def vector(self) -> NDArray[np.float64]:
        """Return a read-only ``[q, v]`` state vector."""

        result = np.concatenate(
            (self.position_at_reference_world_m, self.velocity_world_mps)
        )
        result.setflags(write=False)
        return result

    def position_at(self, time_s: float) -> NDArray[np.float64]:
        """Evaluate ``q(t)=q0+v(t-t0)`` in metres."""

        time = float(time_s)
        if not np.isfinite(time):
            raise ValueError("time_s must be finite")
        result = self.position_at_reference_world_m + self.velocity_world_mps * (
            time - self.reference_time_s
        )
        result.setflags(write=False)
        return result


def rebase_constant_velocity_state(
    state: ConstantVelocityState, new_reference_time_s: float
) -> ConstantVelocityState:
    """Represent the same trajectory at a new reference epoch."""

    if not isinstance(state, ConstantVelocityState):
        raise TypeError("state must be ConstantVelocityState")
    new_reference = float(new_reference_time_s)
    if not np.isfinite(new_reference):
        raise ValueError("new_reference_time_s must be finite")
    return ConstantVelocityState(
        state.position_at(new_reference),
        state.velocity_world_mps,
        new_reference,
    )


def constant_velocity_transition_jacobian(dt_s: float) -> NDArray[np.float64]:
    """Return the exact 6-D transition Jacobian ``[[I, dt I], [0, I]]``."""

    dt = float(dt_s)
    if not np.isfinite(dt):
        raise ValueError("dt_s must be finite")
    transition = np.eye(6, dtype=float)
    transition[:3, 3:] = dt * np.eye(3)
    return transition


__all__ = [
    "ConstantVelocityState",
    "constant_velocity_transition_jacobian",
    "rebase_constant_velocity_state",
]
