"""Analytic and numerical Jacobians for the far-field TDOA model."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import (
    DEFAULT_SOUND_SPEED,
    baselines,
    microphone_positions,
    reference_pairs,
    validate_pairs,
)
from .tdoa import far_field_tdoa


def direction_jacobian(phi: float, elevation: float) -> NDArray[np.float64]:
    """Return ``d u / d(phi, elevation)`` as a ``(3, 2)`` matrix."""

    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    cos_elevation, sin_elevation = np.cos(elevation), np.sin(elevation)
    derivative_phi = np.asarray(
        [-cos_elevation * sin_phi, cos_elevation * cos_phi, 0.0]
    )
    derivative_elevation = np.asarray(
        [-sin_elevation * cos_phi, -sin_elevation * sin_phi, cos_elevation]
    )
    return np.column_stack((derivative_phi, derivative_elevation))


def far_field_tdoa_jacobian(
    phi: float,
    elevation: float,
    positions: ArrayLike,
    pairs: Iterable[Sequence[int]] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Return the analytic ``d tau / d(phi, elevation)`` matrix."""

    coordinates = microphone_positions(positions)
    checked_pairs = (
        reference_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    speed = float(sound_speed)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("sound speed must be finite and positive")
    return baselines(coordinates, checked_pairs) @ direction_jacobian(phi, elevation) / speed


def central_difference_jacobian(
    function: Callable[[NDArray[np.float64]], ArrayLike],
    parameters: ArrayLike,
    step: float = 1e-6,
) -> NDArray[np.float64]:
    """Compute a vector-valued Jacobian with central finite differences."""

    point = np.asarray(parameters, dtype=float)
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise ValueError("parameters must be a finite one-dimensional array")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    columns = []
    for index in range(point.size):
        delta = np.zeros_like(point)
        delta[index] = step
        forward = np.asarray(function(point + delta), dtype=float)
        backward = np.asarray(function(point - delta), dtype=float)
        columns.append((forward - backward) / (2.0 * step))
    return np.column_stack(columns)


def numerical_tdoa_jacobian(
    phi: float,
    elevation: float,
    positions: ArrayLike,
    pairs: Iterable[Sequence[int]] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    step: float = 1e-6,
) -> NDArray[np.float64]:
    """Numerically differentiate the far-field TDOA model."""

    coordinates = microphone_positions(positions)
    checked_pairs = reference_pairs(coordinates.shape[0]) if pairs is None else tuple(pairs)
    return central_difference_jacobian(
        lambda angles: far_field_tdoa(
            angles[0], angles[1], coordinates, checked_pairs, sound_speed
        ),
        [phi, elevation],
        step,
    )
