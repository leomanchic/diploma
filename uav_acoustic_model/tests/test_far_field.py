"""Exact, asymptotic, and quantitative far-field tests."""

import numpy as np

from model.geometry import all_pairs, array_centroid, comparison_arrays
from model.tdoa import (
    directional_spherical_tdoa,
    directional_spherical_travel_times,
    far_field_tdoa,
    plane_wave_arrival_times,
    second_order_tdoa,
    second_order_travel_times,
    source_position_from_direction,
    travel_times,
)
from validation.far_field import (
    continuous_refine_far_field_error,
    direction_grid,
    far_field_error,
    minimum_far_field_distance,
    phase_error,
    sample_error,
)


def test_directional_source_and_exact_travel_times_use_array_centroid():
    positions = comparison_arrays()["tetrahedral"] + [2.0, -3.0, 1.5]
    phi, elevation, distance = 0.8, 0.35, 7.0
    source = source_position_from_direction(phi, elevation, distance, positions)
    np.testing.assert_allclose(np.linalg.norm(source - array_centroid(positions)), distance)
    np.testing.assert_allclose(
        directional_spherical_travel_times(phi, elevation, distance, positions),
        travel_times(source, positions),
    )


def test_plane_arrival_time_sign_exactly_matches_existing_tdoa_convention():
    positions = comparison_arrays()["rectangle 3:1"] + [4.0, -1.0, 8.0]
    pairs = all_pairs(len(positions))
    phi, elevation = 1.2, 0.47
    times = plane_wave_arrival_times(phi, elevation, positions)
    shifted_times = plane_wave_arrival_times(
        phi, elevation, positions, remove_common=True
    )
    from_times = np.asarray([times[i] - times[j] for i, j in pairs])
    from_shifted = np.asarray([shifted_times[i] - shifted_times[j] for i, j in pairs])
    expected = far_field_tdoa(phi, elevation, positions, pairs)
    np.testing.assert_allclose(from_times, expected, rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(from_shifted, expected, rtol=0.0, atol=1e-18)
    assert np.min(shifted_times) == 0.0


def test_second_order_travel_times_implement_the_stated_expansion():
    positions = comparison_arrays()["L-shaped"]
    phi, elevation, distance = 0.63, 0.42, 3.5
    centred = positions - np.mean(positions, axis=0)
    direction = source_position_from_direction(phi, elevation, 1.0, positions)
    direction = direction - np.mean(positions, axis=0)
    projection = centred @ direction
    expected_distance = (
        distance
        - projection
        + (np.sum(centred**2, axis=1) - projection**2) / (2.0 * distance)
    )
    np.testing.assert_allclose(
        second_order_travel_times(phi, elevation, distance, positions) * 343.0,
        expected_distance,
        rtol=2e-16,
        atol=2e-16,
    )


def test_plane_error_is_first_order_and_second_order_error_decays_faster():
    positions = comparison_arrays()["L-shaped"]
    pairs = all_pairs(len(positions))
    phi, elevation = 0.71, 0.38
    distances = np.asarray([2.0, 4.0, 8.0, 16.0])
    plane_errors = []
    second_errors = []
    plane = far_field_tdoa(phi, elevation, positions, pairs)
    for distance in distances:
        exact = directional_spherical_tdoa(
            phi, elevation, distance, positions, pairs
        )
        second = second_order_tdoa(phi, elevation, distance, positions, pairs)
        plane_errors.append(np.max(np.abs(exact - plane)))
        second_errors.append(np.max(np.abs(exact - second)))
    plane_slope = np.polyfit(np.log(distances), np.log(plane_errors), 1)[0]
    second_slope = np.polyfit(np.log(distances), np.log(second_errors), 1)[0]
    assert -1.08 < plane_slope < -0.92
    assert -2.12 < second_slope < -1.88
    assert np.all(np.asarray(second_errors) < np.asarray(plane_errors))


def test_plane_tdoa_has_same_sign_as_large_distance_spherical_limit():
    positions = comparison_arrays()["tetrahedral"]
    pairs = all_pairs(len(positions))
    plane = far_field_tdoa(0.91, 0.52, positions, pairs)
    exact = directional_spherical_tdoa(0.91, 0.52, 1e5, positions, pairs)
    nonzero = np.abs(plane) > 1e-9
    np.testing.assert_array_equal(np.sign(exact[nonzero]), np.sign(plane[nonzero]))
    np.testing.assert_allclose(exact, plane, rtol=2e-5, atol=2e-12)


def test_directional_models_are_invariant_to_common_array_translation():
    positions = comparison_arrays()["L-shaped"]
    translated = positions + [19.0, -7.0, 4.5]
    pairs = all_pairs(len(positions))
    arguments = (1.1, 0.27, 6.0)
    np.testing.assert_allclose(
        directional_spherical_tdoa(*arguments, positions, pairs),
        directional_spherical_tdoa(*arguments, translated, pairs),
        atol=2e-17,
    )
    np.testing.assert_allclose(
        second_order_tdoa(*arguments, positions, pairs),
        second_order_tdoa(*arguments, translated, pairs),
        atol=2e-17,
    )
    np.testing.assert_allclose(
        far_field_tdoa(arguments[0], arguments[1], positions, pairs),
        far_field_tdoa(arguments[0], arguments[1], translated, pairs),
        atol=2e-17,
    )


def test_grid_maximum_error_decreases_and_boundary_is_numerically_searched():
    positions = comparison_arrays()["square"]
    grid = direction_grid(azimuth_step_deg=20.0, elevation_step_deg=20.0)
    distances = np.asarray([1.0, 2.0, 4.0, 8.0])
    errors = np.asarray(
        [far_field_error(positions, distance, grid=grid).max_plane_error_s for distance in distances]
    )
    second_errors = np.asarray(
        [
            far_field_error(positions, distance, grid=grid).max_second_order_error_s
            for distance in distances
        ]
    )
    assert np.all(np.diff(errors) < 0.0)
    assert np.all(second_errors < errors)
    assert -1.08 < np.polyfit(np.log(distances), np.log(errors), 1)[0] < -0.92

    target = 0.1 / 48_000.0
    boundary = minimum_far_field_distance(
        positions,
        target,
        grid=grid,
        relative_distance_tolerance=1e-4,
    )
    assert boundary.achieved_error_s <= target
    assert (
        far_field_error(positions, boundary.distance_m * 0.999, grid=grid).max_plane_error_s
        > target
    )


def test_azimuth_grid_is_half_open_and_continuous_refinement_improves_coarse_maximum():
    coarse_grid = direction_grid(azimuth_step_deg=30.0, elevation_step_deg=20.0)
    assert np.min(coarse_grid.azimuth_deg) == 0.0
    assert np.max(coarse_grid.azimuth_deg) < 360.0
    assert not np.any(coarse_grid.azimuth_deg == 360.0)
    assert np.unique(coarse_grid.directions, axis=0).shape[0] == coarse_grid.directions.shape[0]

    positions = comparison_arrays()["square"]
    coarse = far_field_error(positions, 10.0, grid=coarse_grid)
    continuous = continuous_refine_far_field_error(
        positions, 10.0, grid=coarse_grid, candidate_count=8
    )
    fine = far_field_error(
        positions,
        10.0,
        grid=direction_grid(1.0, 1.0),
    )
    assert continuous.max_error_s >= coarse.max_plane_error_s * (1.0 - 1e-10)
    assert abs(continuous.max_error_s - fine.max_plane_error_s) / fine.max_plane_error_s < 2e-3
    assert continuous.candidate_count == 8


def test_timing_error_unit_conversions():
    error = np.asarray([1e-6, 2e-6])
    np.testing.assert_allclose(sample_error(error, 48_000.0), [0.048, 0.096])
    np.testing.assert_allclose(phase_error(error, 4_000.0), 2.0 * np.pi * 4_000.0 * error)
