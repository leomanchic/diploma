"""Exact spherical and far-field TDOA models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import (
    DEFAULT_SOUND_SPEED,
    array_centroid,
    baselines,
    centered_positions,
    direction_vector,
    incidence_matrix,
    microphone_positions,
    reference_pairs,
    validate_pairs,
)


def _positive_standard_deviation(value: float, name: str) -> float:
    standard_deviation = float(value)
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return standard_deviation


def _sound_speed(value: float) -> float:
    speed = float(value)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("sound speed must be finite and positive")
    return speed


def travel_times(
    source_position: ArrayLike,
    positions: ArrayLike,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Return geometric propagation times from one source to all microphones."""

    source = np.asarray(source_position, dtype=float)
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise ValueError("source_position must be a finite vector with shape (3,)")
    coordinates = microphone_positions(positions)
    return np.linalg.norm(source - coordinates, axis=1) / _sound_speed(sound_speed)


def source_position_from_direction(
    phi: float,
    elevation: float,
    distance: float,
    positions: ArrayLike,
) -> NDArray[np.float64]:
    """Return ``centroid(positions) + distance * u(phi, elevation)``.

    The distance is therefore invariant to a common translation of the array.
    """

    source_distance = float(distance)
    if not np.isfinite(source_distance) or source_distance <= 0.0:
        raise ValueError("distance must be finite and positive")
    return array_centroid(positions) + source_distance * direction_vector(phi, elevation)


def directional_spherical_travel_times(
    phi: float,
    elevation: float,
    distance: float,
    positions: ArrayLike,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Return exact ``||centroid + R*u - r_m|| / c`` arrival times."""

    source = source_position_from_direction(phi, elevation, distance, positions)
    return travel_times(source, positions, sound_speed)


def plane_wave_arrival_times(
    phi: float,
    elevation: float,
    positions: ArrayLike,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    *,
    remove_common: bool = False,
) -> NDArray[np.float64]:
    """Return plane-wave relative TOAs with the established TDOA sign.

    Before optional common-time removal,
    ``T_m = -(r_m-centroid).T @ u / c``. Consequently
    ``T_i-T_j = (r_j-r_i).T @ u / c`` exactly. With
    ``remove_common=True`` the minimum time is subtracted, which changes no
    TDOA and makes every signal-synthesis delay non-negative.
    """

    times = -centered_positions(positions) @ direction_vector(phi, elevation) / _sound_speed(
        sound_speed
    )
    if remove_common:
        times = times - np.min(times)
    return times


def second_order_travel_times(
    phi: float,
    elevation: float,
    distance: float,
    positions: ArrayLike,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    r"""Return the second-order far-field expansion of spherical TOAs.

    For centroid-relative coordinates ``rho_m`` this implements

    ``(R-u.T rho_m + (||rho_m||^2-(u.T rho_m)^2)/(2R))/c``.

    The omitted distance term is ``O(D^3/R^2)`` and this function remains
    separate from the exact spherical formula.
    """

    source_distance = float(distance)
    if not np.isfinite(source_distance) or source_distance <= 0.0:
        raise ValueError("distance must be finite and positive")
    coordinates = centered_positions(positions)
    direction = direction_vector(phi, elevation)
    projections = coordinates @ direction
    transverse_squared = np.sum(coordinates**2, axis=1) - projections**2
    distances = source_distance - projections + transverse_squared / (2.0 * source_distance)
    return distances / _sound_speed(sound_speed)


def spherical_tdoa(
    source_position: ArrayLike,
    positions: ArrayLike,
    pairs: Iterable[Sequence[int]] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Return exact ``tau_ij = (||q-r_i||-||q-r_j||)/c``."""

    coordinates = microphone_positions(positions)
    checked_pairs = (
        reference_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    times = travel_times(source_position, coordinates, sound_speed)
    return np.asarray([times[first] - times[second] for first, second in checked_pairs])


def directional_spherical_tdoa(
    phi: float,
    elevation: float,
    distance: float,
    positions: ArrayLike,
    pairs: Iterable[Sequence[int]] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Return exact spherical TDOA for a centroid-relative source distance."""

    coordinates = microphone_positions(positions)
    checked_pairs = (
        reference_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    times = directional_spherical_travel_times(
        phi, elevation, distance, coordinates, sound_speed
    )
    return np.asarray([times[first] - times[second] for first, second in checked_pairs])


def second_order_tdoa(
    phi: float,
    elevation: float,
    distance: float,
    positions: ArrayLike,
    pairs: Iterable[Sequence[int]] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Return TDOA from the explicit second-order distance expansion."""

    coordinates = microphone_positions(positions)
    checked_pairs = (
        reference_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    times = second_order_travel_times(phi, elevation, distance, coordinates, sound_speed)
    return np.asarray([times[first] - times[second] for first, second in checked_pairs])


def far_field_tdoa(
    phi: float,
    elevation: float,
    positions: ArrayLike,
    pairs: Iterable[Sequence[int]] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Return plane-wave ``tau_ij = (r_j-r_i)^T u / c``."""

    coordinates = microphone_positions(positions)
    checked_pairs = (
        reference_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    return baselines(coordinates, checked_pairs) @ direction_vector(phi, elevation) / _sound_speed(
        sound_speed
    )


def tdoa_covariance_from_toa(
    toa_covariance: ArrayLike,
    pairs: Iterable[Sequence[int]],
) -> NDArray[np.float64]:
    """Propagate an arrival-time covariance using ``Sigma_tau=B Sigma_t B^T``."""

    covariance = np.asarray(toa_covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("toa_covariance must be square")
    if not np.all(np.isfinite(covariance)) or not np.allclose(
        covariance, covariance.T, rtol=1e-10, atol=1e-14
    ):
        raise ValueError("toa_covariance must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
    scale = float(np.max(np.abs(eigenvalues)))
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError("toa_covariance must be positive semidefinite")
    matrix = incidence_matrix(pairs, covariance.shape[0])
    result = matrix @ covariance @ matrix.T
    return (result + result.T) / 2.0


def independent_toa_covariance(
    microphone_count: int,
    sigma_toa: float,
) -> NDArray[np.float64]:
    """Return ``sigma_toa**2 I`` for independent microphone TOA errors.

    ``sigma_toa`` is the standard deviation of one microphone arrival-time
    error ``e_m``. It is not the standard deviation of a TDOA difference.
    """

    if microphone_count < 2:
        raise ValueError("microphone_count must be at least 2")
    standard_deviation = _positive_standard_deviation(sigma_toa, "sigma_toa")
    return np.eye(microphone_count) * standard_deviation**2


def independent_tdoa_covariance(
    pair_count: int,
    sigma_tdoa: float,
) -> NDArray[np.float64]:
    """Return ``sigma_tdoa**2 I`` for independently measured TDOA errors.

    This is an abstract measurement model applied directly to the selected
    TDOAs and is distinct from differencing independent microphone TOAs.
    """

    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    standard_deviation = _positive_standard_deviation(sigma_tdoa, "sigma_tdoa")
    return np.eye(pair_count) * standard_deviation**2


def tdoa_covariance_from_independent_toa(
    microphone_count: int,
    pairs: Iterable[Sequence[int]],
    sigma_toa: float,
) -> NDArray[np.float64]:
    """Propagate independent equal-variance TOA errors to selected TDOAs."""

    return tdoa_covariance_from_toa(
        independent_toa_covariance(microphone_count, sigma_toa), pairs
    )


def cycle_constraint_matrix(
    pairs: Iterable[Sequence[int]],
    microphone_count: int,
) -> NDArray[np.float64]:
    """Return independent triangle-cycle constraints for a complete pair set.

    For every ``1 <= j < k < M`` a row represents
    ``tau_0j + tau_jk - tau_0k = 0``. Pair ordering and orientation may be
    arbitrary. The returned rows span ``null(B.T)`` for the complete graph.
    """

    checked_pairs = validate_pairs(pairs, microphone_count)
    expected_count = microphone_count * (microphone_count - 1) // 2
    if len(checked_pairs) != expected_count:
        raise ValueError("cycle constraints require every unordered microphone pair")

    edge_lookup: dict[tuple[int, int], tuple[int, float]] = {}
    for index, (first, second) in enumerate(checked_pairs):
        edge = (min(first, second), max(first, second))
        if edge in edge_lookup:
            raise ValueError("pair set contains a duplicate undirected edge")
        orientation = 1.0 if first < second else -1.0
        edge_lookup[edge] = (index, orientation)
    if len(edge_lookup) != expected_count:
        raise ValueError("pair set is not the complete graph")

    constraints = []
    for middle in range(1, microphone_count):
        for last in range(middle + 1, microphone_count):
            row = np.zeros(len(checked_pairs), dtype=float)
            for edge, coefficient in (
                ((0, middle), 1.0),
                ((middle, last), 1.0),
                ((0, last), -1.0),
            ):
                index, orientation = edge_lookup[edge]
                row[index] = coefficient * orientation
            constraints.append(row)
    if not constraints:
        return np.zeros((0, len(checked_pairs)), dtype=float)
    return np.asarray(constraints, dtype=float)
