"""Microphone-array geometries and direction-vector conventions.

All distances are in metres and all angles are in radians.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_SOUND_SPEED = 343.0
DEFAULT_APERTURE = 0.20

Pair = tuple[int, int]


def direction_vector(phi: ArrayLike, elevation: ArrayLike) -> NDArray[np.float64]:
    """Return the unit vector from the array centre towards the source.

    ``phi`` is azimuth and ``elevation`` is angle above the xy plane. Scalar
    inputs produce shape ``(3,)``; broadcast array inputs produce ``(..., 3)``.
    """

    phi_array, elevation_array = np.broadcast_arrays(
        np.asarray(phi, dtype=float), np.asarray(elevation, dtype=float)
    )
    cos_elevation = np.cos(elevation_array)
    return np.stack(
        (
            cos_elevation * np.cos(phi_array),
            cos_elevation * np.sin(phi_array),
            np.sin(elevation_array),
        ),
        axis=-1,
    )


def direction_angles(direction: ArrayLike) -> tuple[float, float]:
    """Convert one non-zero Cartesian direction to azimuth and elevation."""

    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3,):
        raise ValueError("direction must have shape (3,)")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("direction must be finite and non-zero")
    x, y, z = vector / norm
    phi = float(np.arctan2(y, x))
    elevation = float(np.arctan2(z, np.hypot(x, y)))
    return phi, elevation


def microphone_positions(positions: ArrayLike) -> NDArray[np.float64]:
    """Validate and return an ``(M, 3)`` microphone-coordinate array."""

    result = np.asarray(positions, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] < 2:
        raise ValueError("microphone positions must have shape (M, 3), M >= 2")
    if not np.all(np.isfinite(result)):
        raise ValueError("microphone positions must be finite")
    return result


def array_centroid(positions: ArrayLike) -> NDArray[np.float64]:
    """Return the arithmetic centroid of the microphone coordinates."""

    return np.mean(microphone_positions(positions), axis=0)


def centered_positions(positions: ArrayLike) -> NDArray[np.float64]:
    """Return microphone coordinates relative to their centroid."""

    coordinates = microphone_positions(positions)
    return coordinates - np.mean(coordinates, axis=0, keepdims=True)


def reference_pairs(microphone_count: int, reference: int = 0) -> tuple[Pair, ...]:
    """Return ``M-1`` linearly independent pairs ``(i, reference)``.

    Linear independence does not imply statistical independence: reference
    TDOAs share the error of the reference microphone under a TOA-error model.
    """

    if microphone_count < 2:
        raise ValueError("microphone_count must be at least 2")
    if not 0 <= reference < microphone_count:
        raise ValueError("reference microphone is out of range")
    return tuple((index, reference) for index in range(microphone_count) if index != reference)


def all_pairs(microphone_count: int) -> tuple[Pair, ...]:
    """Return every unordered microphone pair with ``i < j``."""

    if microphone_count < 2:
        raise ValueError("microphone_count must be at least 2")
    return tuple(
        (first, second)
        for first in range(microphone_count)
        for second in range(first + 1, microphone_count)
    )


def validate_pairs(pairs: Iterable[Sequence[int]], microphone_count: int) -> tuple[Pair, ...]:
    """Validate pair indices while preserving their orientation."""

    result: list[Pair] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each microphone pair must contain two indices")
        first, second = int(pair[0]), int(pair[1])
        if first == second:
            raise ValueError("a TDOA pair must use two different microphones")
        if not (0 <= first < microphone_count and 0 <= second < microphone_count):
            raise ValueError("microphone pair index is out of range")
        result.append((first, second))
    if not result:
        raise ValueError("at least one microphone pair is required")
    return tuple(result)


def baselines(positions: ArrayLike, pairs: Iterable[Sequence[int]]) -> NDArray[np.float64]:
    """Return rows ``r_j - r_i`` for oriented pairs ``(i, j)``."""

    coordinates = microphone_positions(positions)
    checked_pairs = validate_pairs(pairs, coordinates.shape[0])
    return np.asarray(
        [coordinates[second] - coordinates[first] for first, second in checked_pairs],
        dtype=float,
    )


def incidence_matrix(pairs: Iterable[Sequence[int]], microphone_count: int) -> NDArray[np.float64]:
    """Return B such that ``tau = B @ arrival_times`` for ``tau_ij=T_i-T_j``."""

    checked_pairs = validate_pairs(pairs, microphone_count)
    matrix = np.zeros((len(checked_pairs), microphone_count), dtype=float)
    for row, (first, second) in enumerate(checked_pairs):
        matrix[row, first] = 1.0
        matrix[row, second] = -1.0
    return matrix


def aperture(positions: ArrayLike) -> float:
    """Return the maximum inter-microphone distance."""

    coordinates = microphone_positions(positions)
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return float(np.max(np.linalg.norm(differences, axis=-1)))


def geometry_rank(positions: ArrayLike, tolerance: float | None = None) -> int:
    """Return the affine rank of the microphone coordinates."""

    centred = centered_positions(positions)
    return int(np.linalg.matrix_rank(centred, tol=tolerance))


def _centred(points: ArrayLike) -> NDArray[np.float64]:
    coordinates = np.asarray(points, dtype=float)
    return coordinates - np.mean(coordinates, axis=0, keepdims=True)


def linear_array(aperture_m: float = DEFAULT_APERTURE) -> NDArray[np.float64]:
    """Four uniformly spaced microphones on the x axis."""

    x_coordinates = np.linspace(-aperture_m / 2.0, aperture_m / 2.0, 4)
    return np.column_stack((x_coordinates, np.zeros((4, 2))))


def l_shaped_array(aperture_m: float = DEFAULT_APERTURE) -> NDArray[np.float64]:
    """Four microphones distributed along two perpendicular arms."""

    arm = aperture_m / np.sqrt(2.0)
    return _centred(
        [[0.0, 0.0, 0.0], [arm / 2.0, 0.0, 0.0], [arm, 0.0, 0.0], [0.0, arm, 0.0]]
    )


def rectangular_array(aperture_m: float = DEFAULT_APERTURE) -> NDArray[np.float64]:
    """Four corner microphones in a 3:1 rectangle with fixed diagonal."""

    height = aperture_m / np.sqrt(10.0)
    width = 3.0 * height
    return np.asarray(
        [
            [-width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, height / 2.0, 0.0],
            [-width / 2.0, height / 2.0, 0.0],
        ]
    )


def square_array(aperture_m: float = DEFAULT_APERTURE) -> NDArray[np.float64]:
    """Four corner microphones in a square with fixed diagonal."""

    side = aperture_m / np.sqrt(2.0)
    half_side = side / 2.0
    return np.asarray(
        [
            [-half_side, -half_side, 0.0],
            [half_side, -half_side, 0.0],
            [half_side, half_side, 0.0],
            [-half_side, half_side, 0.0],
        ]
    )


def tetrahedral_array(aperture_m: float = DEFAULT_APERTURE) -> NDArray[np.float64]:
    """Four microphones at the vertices of a regular tetrahedron."""

    scale = aperture_m / (2.0 * np.sqrt(2.0))
    return scale * np.asarray(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    )


def comparison_arrays(aperture_m: float = DEFAULT_APERTURE) -> dict[str, NDArray[np.float64]]:
    """Return the five requested four-microphone comparison geometries."""

    if not np.isfinite(aperture_m) or aperture_m <= 0.0:
        raise ValueError("aperture_m must be finite and positive")
    return {
        "linear": linear_array(aperture_m),
        "L-shaped": l_shaped_array(aperture_m),
        "rectangle 3:1": rectangular_array(aperture_m),
        "square": square_array(aperture_m),
        "tetrahedral": tetrahedral_array(aperture_m),
    }
