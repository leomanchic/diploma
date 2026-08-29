"""Quantitative far-field error analysis on configurable angular grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from model.geometry import (
    DEFAULT_SOUND_SPEED,
    all_pairs,
    centered_positions,
    direction_vector,
    microphone_positions,
    validate_pairs,
)


@dataclass(frozen=True)
class DirectionGrid:
    """Flattened inclusive azimuth/elevation grid."""

    azimuth_deg: NDArray[np.float64]
    elevation_deg: NDArray[np.float64]
    directions: NDArray[np.float64]
    azimuth_step_deg: float
    elevation_step_deg: float


@dataclass(frozen=True)
class FarFieldErrorResult:
    """Worst plane and second-order TDOA errors for one source distance."""

    distance_m: float
    max_plane_error_s: float
    max_second_order_error_s: float
    worst_plane_azimuth_deg: float
    worst_plane_elevation_deg: float
    worst_plane_pair: tuple[int, int]
    worst_second_order_azimuth_deg: float
    worst_second_order_elevation_deg: float
    worst_second_order_pair: tuple[int, int]


@dataclass(frozen=True)
class FarFieldBoundaryResult:
    """Numerically searched distance satisfying one time-error threshold."""

    distance_m: float
    target_error_s: float
    achieved_error_s: float
    worst_azimuth_deg: float
    worst_elevation_deg: float
    worst_pair: tuple[int, int]
    iterations: int


@dataclass(frozen=True)
class ContinuousFarFieldRefinement:
    """Local continuous maximum refined from several best grid candidates."""

    max_error_s: float
    worst_azimuth_deg: float
    worst_elevation_deg: float
    worst_pair: tuple[int, int]
    candidate_count: int
    successful_optimizations: int


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _inclusive_axis(stop: float, step: float) -> NDArray[np.float64]:
    checked_step = _positive_finite(step, "angular step")
    values = np.arange(0.0, stop + 0.5 * checked_step, checked_step, dtype=float)
    values = values[values <= stop + 1e-12]
    if values.size == 0 or not np.isclose(values[-1], stop, atol=1e-12):
        values = np.append(values, stop)
    return values


def _half_open_axis(stop: float, step: float) -> NDArray[np.float64]:
    checked_step = _positive_finite(step, "angular step")
    values = np.arange(0.0, stop, checked_step, dtype=float)
    return values[values < stop]


def direction_grid(
    azimuth_step_deg: float = 1.0,
    elevation_step_deg: float | None = None,
    *,
    max_elevation_deg: float = 80.0,
) -> DirectionGrid:
    """Return directions on azimuth ``[0, 360)`` and inclusive elevation."""

    azimuth_step = _positive_finite(azimuth_step_deg, "azimuth_step_deg")
    elevation_step = (
        azimuth_step
        if elevation_step_deg is None
        else _positive_finite(elevation_step_deg, "elevation_step_deg")
    )
    maximum_elevation = float(max_elevation_deg)
    if not np.isfinite(maximum_elevation) or not 0.0 <= maximum_elevation <= 90.0:
        raise ValueError("max_elevation_deg must lie in [0, 90]")
    azimuth_axis = _half_open_axis(360.0, azimuth_step)
    elevation_axis = _inclusive_axis(maximum_elevation, elevation_step)
    azimuth_mesh, elevation_mesh = np.meshgrid(
        azimuth_axis, elevation_axis, indexing="xy"
    )
    return DirectionGrid(
        azimuth_deg=azimuth_mesh.ravel(),
        elevation_deg=elevation_mesh.ravel(),
        directions=direction_vector(
            np.deg2rad(azimuth_mesh.ravel()), np.deg2rad(elevation_mesh.ravel())
        ),
        azimuth_step_deg=azimuth_step,
        elevation_step_deg=elevation_step,
    )


def _tdoa_error_surfaces(
    centred: NDArray[np.float64],
    directions: NDArray[np.float64],
    distance: float,
    pairs: tuple[tuple[int, int], ...],
    speed: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    projections = directions @ centred.T
    radii_squared = np.sum(centred**2, axis=1)[None, :]
    exact_distances = np.sqrt(distance**2 - 2.0 * distance * projections + radii_squared)
    plane_times = -projections / speed
    exact_times = exact_distances / speed
    second_order_times = (
        distance - projections + (radii_squared - projections**2) / (2.0 * distance)
    ) / speed
    first = np.asarray([pair[0] for pair in pairs], dtype=int)
    second = np.asarray([pair[1] for pair in pairs], dtype=int)
    exact_tdoa = exact_times[:, first] - exact_times[:, second]
    plane_tdoa = plane_times[:, first] - plane_times[:, second]
    second_order_tdoa = second_order_times[:, first] - second_order_times[:, second]
    return np.abs(exact_tdoa - plane_tdoa), np.abs(exact_tdoa - second_order_tdoa)


def far_field_error(
    positions: ArrayLike,
    distance_m: float,
    *,
    grid: DirectionGrid | None = None,
    azimuth_step_deg: float = 1.0,
    elevation_step_deg: float | None = None,
    max_elevation_deg: float = 80.0,
    pairs: tuple[tuple[int, int], ...] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> FarFieldErrorResult:
    """Evaluate the maximum exact-minus-asymptotic TDOA errors."""

    distance = _positive_finite(distance_m, "distance_m")
    speed = _positive_finite(sound_speed, "sound_speed")
    coordinates = microphone_positions(positions)
    centred = centered_positions(coordinates)
    checked_pairs = (
        all_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    selected_grid = (
        direction_grid(azimuth_step_deg, elevation_step_deg, max_elevation_deg=max_elevation_deg)
        if grid is None
        else grid
    )
    plane_errors, second_order_errors = _tdoa_error_surfaces(
        centred, selected_grid.directions, distance, checked_pairs, speed
    )

    plane_flat_index = int(np.argmax(plane_errors))
    second_flat_index = int(np.argmax(second_order_errors))
    plane_direction_index, plane_pair_index = np.unravel_index(
        plane_flat_index, plane_errors.shape
    )
    second_direction_index, second_pair_index = np.unravel_index(
        second_flat_index, second_order_errors.shape
    )
    return FarFieldErrorResult(
        distance_m=distance,
        max_plane_error_s=float(plane_errors[plane_direction_index, plane_pair_index]),
        max_second_order_error_s=float(
            second_order_errors[second_direction_index, second_pair_index]
        ),
        worst_plane_azimuth_deg=float(selected_grid.azimuth_deg[plane_direction_index]),
        worst_plane_elevation_deg=float(selected_grid.elevation_deg[plane_direction_index]),
        worst_plane_pair=checked_pairs[plane_pair_index],
        worst_second_order_azimuth_deg=float(
            selected_grid.azimuth_deg[second_direction_index]
        ),
        worst_second_order_elevation_deg=float(
            selected_grid.elevation_deg[second_direction_index]
        ),
        worst_second_order_pair=checked_pairs[second_pair_index],
    )


def continuous_refine_far_field_error(
    positions: ArrayLike,
    distance_m: float,
    *,
    grid: DirectionGrid,
    candidate_count: int = 12,
    neighborhood_steps: float = 2.0,
    pairs: tuple[tuple[int, int], ...] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> ContinuousFarFieldRefinement:
    """Refine the maximum near several strongest grid direction/pair points.

    Azimuth is periodic and elevation stays within the grid's configured
    interval. Each candidate optimises the squared exact-minus-plane TDOA
    error for its pair using bounded L-BFGS-B.
    """

    distance = _positive_finite(distance_m, "distance_m")
    speed = _positive_finite(sound_speed, "sound_speed")
    coordinates = microphone_positions(positions)
    centred = centered_positions(coordinates)
    checked_pairs = (
        all_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    count = int(candidate_count)
    if count < 1:
        raise ValueError("candidate_count must be positive")
    neighborhood = _positive_finite(neighborhood_steps, "neighborhood_steps")
    plane_errors, _ = _tdoa_error_surfaces(
        centred, grid.directions, distance, checked_pairs, speed
    )
    flat = plane_errors.ravel()
    count = min(count, flat.size)
    candidate_indices = np.argpartition(flat, -count)[-count:]
    candidate_indices = candidate_indices[np.argsort(flat[candidate_indices])[::-1]]
    maximum_elevation_deg = float(np.max(grid.elevation_deg))
    azimuth_radius = np.deg2rad(neighborhood * grid.azimuth_step_deg)
    elevation_radius = np.deg2rad(neighborhood * grid.elevation_step_deg)
    best_error = -np.inf
    best_azimuth = 0.0
    best_elevation = 0.0
    best_pair = checked_pairs[0]
    successful = 0

    for flat_index in candidate_indices:
        direction_index, pair_index = np.unravel_index(int(flat_index), plane_errors.shape)
        seed_azimuth = np.deg2rad(grid.azimuth_deg[direction_index])
        seed_elevation = np.deg2rad(grid.elevation_deg[direction_index])
        first, second = checked_pairs[pair_index]

        def negative_squared_error(angles: NDArray[np.float64]) -> float:
            azimuth = float(np.mod(angles[0], 2.0 * np.pi))
            elevation = float(angles[1])
            direction = direction_vector(azimuth, elevation)
            exact_distances = np.linalg.norm(
                distance * direction[None, :] - centred[[first, second]], axis=1
            )
            exact_tdoa = (exact_distances[0] - exact_distances[1]) / speed
            plane_tdoa = (
                (centred[second] - centred[first]) @ direction / speed
            )
            error_microseconds = (exact_tdoa - plane_tdoa) * 1e6
            return -float(error_microseconds**2)

        result = minimize(
            negative_squared_error,
            np.asarray([seed_azimuth, seed_elevation]),
            method="L-BFGS-B",
            bounds=(
                (seed_azimuth - azimuth_radius, seed_azimuth + azimuth_radius),
                (
                    max(0.0, seed_elevation - elevation_radius),
                    min(np.deg2rad(maximum_elevation_deg), seed_elevation + elevation_radius),
                ),
            ),
            options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 200},
        )
        successful += int(result.success)
        error = float(np.sqrt(max(0.0, -result.fun)) * 1e-6)
        if error > best_error:
            best_error = error
            best_azimuth = float(np.mod(result.x[0], 2.0 * np.pi))
            best_elevation = float(result.x[1])
            best_pair = (first, second)

    return ContinuousFarFieldRefinement(
        max_error_s=best_error,
        worst_azimuth_deg=float(np.rad2deg(best_azimuth)),
        worst_elevation_deg=float(np.rad2deg(best_elevation)),
        worst_pair=best_pair,
        candidate_count=count,
        successful_optimizations=successful,
    )


def minimum_far_field_distance(
    positions: ArrayLike,
    target_error_s: float,
    *,
    grid: DirectionGrid | None = None,
    azimuth_step_deg: float = 1.0,
    elevation_step_deg: float | None = None,
    max_elevation_deg: float = 80.0,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    lower_distance_m: float | None = None,
    initial_upper_distance_m: float = 1.0,
    relative_distance_tolerance: float = 1e-6,
    max_iterations: int = 80,
) -> FarFieldBoundaryResult:
    """Find the plane-wave boundary by bracketing and logarithmic bisection.

    The search starts outside the microphone cloud, expands the upper bound
    until the configured maximum TDOA error is acceptable, and then bisects
    geometrically. It therefore does not select a boundary only from a fixed
    list of diagnostic distances.
    """

    target = _positive_finite(target_error_s, "target_error_s")
    tolerance = _positive_finite(relative_distance_tolerance, "relative_distance_tolerance")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    centred = centered_positions(positions)
    cloud_radius = float(np.max(np.linalg.norm(centred, axis=1)))
    lower = (
        max(np.nextafter(cloud_radius, np.inf), 1e-6)
        if lower_distance_m is None
        else _positive_finite(lower_distance_m, "lower_distance_m")
    )
    if lower <= cloud_radius:
        raise ValueError("lower_distance_m must lie outside the microphone cloud")
    upper = max(_positive_finite(initial_upper_distance_m, "initial_upper_distance_m"), lower * 2.0)
    selected_grid = (
        direction_grid(azimuth_step_deg, elevation_step_deg, max_elevation_deg=max_elevation_deg)
        if grid is None
        else grid
    )

    lower_result = far_field_error(
        positions, lower, grid=selected_grid, sound_speed=sound_speed
    )
    if lower_result.max_plane_error_s <= target:
        return FarFieldBoundaryResult(
            distance_m=lower,
            target_error_s=target,
            achieved_error_s=lower_result.max_plane_error_s,
            worst_azimuth_deg=lower_result.worst_plane_azimuth_deg,
            worst_elevation_deg=lower_result.worst_plane_elevation_deg,
            worst_pair=lower_result.worst_plane_pair,
            iterations=0,
        )

    upper_result = far_field_error(
        positions, upper, grid=selected_grid, sound_speed=sound_speed
    )
    expansion_count = 0
    while upper_result.max_plane_error_s > target:
        lower = upper
        upper *= 2.0
        expansion_count += 1
        if expansion_count > max_iterations or not np.isfinite(upper):
            raise RuntimeError("failed to bracket the far-field boundary")
        upper_result = far_field_error(
            positions, upper, grid=selected_grid, sound_speed=sound_speed
        )

    iterations = expansion_count
    while upper / lower - 1.0 > tolerance:
        if iterations >= max_iterations:
            raise RuntimeError("far-field boundary search did not converge")
        middle = float(np.sqrt(lower * upper))
        middle_result = far_field_error(
            positions, middle, grid=selected_grid, sound_speed=sound_speed
        )
        if middle_result.max_plane_error_s <= target:
            upper = middle
            upper_result = middle_result
        else:
            lower = middle
        iterations += 1

    return FarFieldBoundaryResult(
        distance_m=upper,
        target_error_s=target,
        achieved_error_s=upper_result.max_plane_error_s,
        worst_azimuth_deg=upper_result.worst_plane_azimuth_deg,
        worst_elevation_deg=upper_result.worst_plane_elevation_deg,
        worst_pair=upper_result.worst_plane_pair,
        iterations=iterations,
    )


def sample_error(error_s: ArrayLike, sampling_rate_hz: float) -> NDArray[np.float64]:
    """Convert a timing error in seconds to samples."""

    return np.asarray(error_s, dtype=float) * _positive_finite(
        sampling_rate_hz, "sampling_rate_hz"
    )


def phase_error(
    error_s: ArrayLike,
    maximum_frequency_hz: float,
) -> NDArray[np.float64]:
    """Return ``2*pi*f_max*error`` in radians."""

    return 2.0 * np.pi * _positive_finite(
        maximum_frequency_hz, "maximum_frequency_hz"
    ) * np.asarray(error_s, dtype=float)
