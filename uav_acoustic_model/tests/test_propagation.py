"""End-to-end deterministic multi-channel propagation tests."""

import numpy as np
import pytest

from model.geometry import all_pairs, comparison_arrays
from model.tdoa import (
    directional_spherical_tdoa,
    far_field_tdoa,
    source_position_from_direction,
)
from simulation.fractional_delay import frequency_domain_delay
from simulation.propagation import simulate_propagation
from simulation.signals import deterministic_bandlimited_signal


FS = 48_000.0
PHI, ELEVATION = np.deg2rad([37.0, 24.0])


@pytest.fixture(scope="module")
def broadband_signal():
    return deterministic_bandlimited_signal(
        FS,
        0.05,
        minimum_frequency_hz=300.0,
        maximum_frequency_hz=10_000.0,
        tone_count=31,
    )


@pytest.mark.parametrize("model", ["spherical", "plane"])
def test_tdoa_metadata_matches_core_model(model, broadband_signal):
    positions = comparison_arrays()["tetrahedral"]
    pairs = all_pairs(len(positions))
    result = simulate_propagation(
        broadband_signal,
        FS,
        positions,
        phi=PHI,
        elevation=ELEVATION,
        distance_m=8.0,
        propagation_model=model,
        pairs=pairs,
        delay_method="windowed_sinc",
    )
    expected = (
        directional_spherical_tdoa(PHI, ELEVATION, 8.0, positions, pairs)
        if model == "spherical"
        else far_field_tdoa(PHI, ELEVATION, positions, pairs)
    )
    np.testing.assert_allclose(result.tdoa_seconds, expected, rtol=0.0, atol=2e-18)
    from_toa = np.asarray(
        [result.toa_seconds[i] - result.toa_seconds[j] for i, j in pairs]
    )
    np.testing.assert_allclose(result.tdoa_seconds, from_toa, rtol=0.0, atol=0.0)


def test_spherical_propagation_converges_to_plane_for_large_distance(broadband_signal):
    positions = comparison_arrays()["L-shaped"]
    plane = simulate_propagation(
        broadband_signal,
        FS,
        positions,
        phi=PHI,
        elevation=ELEVATION,
        distance_m=1000.0,
        propagation_model="plane",
    )
    spherical_errors = []
    for distance in (10.0, 100.0, 1000.0):
        spherical = simulate_propagation(
            broadband_signal,
            FS,
            positions,
            phi=PHI,
            elevation=ELEVATION,
            distance_m=distance,
            propagation_model="spherical",
        )
        spherical_errors.append(
            np.max(np.abs(spherical.tdoa_matrix_seconds - plane.tdoa_matrix_seconds))
        )
    assert spherical_errors[2] < spherical_errors[1] < spherical_errors[0]
    assert spherical_errors[0] / spherical_errors[1] > 9.0
    assert spherical_errors[1] / spherical_errors[2] > 9.0

    spherical = simulate_propagation(
        broadband_signal,
        FS,
        positions,
        phi=PHI,
        elevation=ELEVATION,
        distance_m=1000.0,
        propagation_model="spherical",
    )
    start = max(plane.valid_region[0], spherical.valid_region[0])
    stop = min(plane.valid_region[1], spherical.valid_region[1])
    assert np.max(np.abs(spherical.channels[:, start:stop] - plane.channels[:, start:stop])) < 3e-4


@pytest.mark.parametrize("model", ["spherical", "plane"])
def test_microphone_permutation_consistently_permutes_all_channel_metadata(
    model, broadband_signal
):
    positions = comparison_arrays()["square"]
    permutation = np.asarray([2, 0, 3, 1])
    baseline = simulate_propagation(
        broadband_signal,
        FS,
        positions,
        phi=PHI,
        elevation=ELEVATION,
        distance_m=12.0,
        propagation_model=model,
        delay_method="windowed_sinc",
    )
    permuted = simulate_propagation(
        broadband_signal,
        FS,
        positions[permutation],
        phi=PHI,
        elevation=ELEVATION,
        distance_m=12.0,
        propagation_model=model,
        delay_method="windowed_sinc",
    )
    np.testing.assert_allclose(permuted.channels, baseline.channels[permutation], atol=2e-13)
    np.testing.assert_allclose(permuted.toa_seconds, baseline.toa_seconds[permutation])
    np.testing.assert_allclose(
        permuted.tdoa_matrix_seconds,
        baseline.tdoa_matrix_seconds[np.ix_(permutation, permutation)],
    )


def test_common_source_and_array_translation_preserves_spherical_result(broadband_signal):
    positions = comparison_arrays()["tetrahedral"]
    source = source_position_from_direction(PHI, ELEVATION, 5.0, positions)
    translation = np.asarray([17.0, -9.0, 3.0])
    baseline = simulate_propagation(
        broadband_signal,
        FS,
        positions,
        source_position=source,
        propagation_model="spherical",
        delay_method="windowed_sinc",
    )
    translated = simulate_propagation(
        broadband_signal,
        FS,
        positions + translation,
        source_position=source + translation,
        propagation_model="spherical",
        delay_method="windowed_sinc",
    )
    np.testing.assert_allclose(translated.toa_seconds, baseline.toa_seconds, atol=1e-17)
    np.testing.assert_allclose(translated.channels, baseline.channels, atol=3e-12)


def test_plane_result_is_invariant_to_common_array_translation(broadband_signal):
    positions = comparison_arrays()["rectangle 3:1"]
    baseline = simulate_propagation(
        broadband_signal,
        FS,
        positions,
        phi=PHI,
        elevation=ELEVATION,
        distance_m=30.0,
        propagation_model="plane",
        delay_method="windowed_sinc",
    )
    translated = simulate_propagation(
        broadband_signal,
        FS,
        positions + [8.0, -4.0, 2.0],
        phi=PHI,
        elevation=ELEVATION,
        distance_m=30.0,
        propagation_model="plane",
        delay_method="windowed_sinc",
    )
    np.testing.assert_allclose(translated.tdoa_seconds, baseline.tdoa_seconds, atol=2e-18)
    np.testing.assert_allclose(translated.channels, baseline.channels, atol=3e-12)


def test_fractional_channel_delays_are_not_rounded_to_integer_samples(broadband_signal):
    positions = comparison_arrays()["square"]
    result = simulate_propagation(
        broadband_signal,
        FS,
        positions,
        phi=0.47,
        elevation=0.31,
        distance_m=20.0,
        propagation_model="plane",
        delay_method="frequency",
    )
    fractional_parts = np.abs(result.applied_delay_samples - np.rint(result.applied_delay_samples))
    channel_index = int(np.argmax(fractional_parts))
    assert fractional_parts[channel_index] > 0.1
    exact = frequency_domain_delay(
        broadband_signal,
        result.applied_delay_samples[channel_index],
        output_length=result.channels.shape[1],
    )
    rounded = frequency_domain_delay(
        broadband_signal,
        float(np.rint(result.applied_delay_samples[channel_index])),
        output_length=result.channels.shape[1],
    )
    np.testing.assert_array_equal(result.channels[channel_index], exact)
    start, stop = result.valid_region
    assert np.max(np.abs(result.channels[channel_index, start:stop] - rounded[start:stop])) > 0.05


def test_generation_is_exactly_reproducible(broadband_signal):
    arguments = dict(
        phi=PHI,
        elevation=ELEVATION,
        distance_m=7.0,
        propagation_model="spherical",
        delay_method="frequency",
    )
    positions = comparison_arrays()["tetrahedral"]
    first = simulate_propagation(broadband_signal, FS, positions, **arguments)
    second = simulate_propagation(broadband_signal, FS, positions, **arguments)
    np.testing.assert_array_equal(first.channels, second.channels)
    np.testing.assert_array_equal(first.toa_seconds, second.toa_seconds)


def test_optional_geometric_attenuation_is_exactly_one_over_distance(broadband_signal):
    positions = comparison_arrays()["tetrahedral"]
    common = dict(
        phi=PHI,
        elevation=ELEVATION,
        distance_m=3.0,
        propagation_model="spherical",
        delay_method="windowed_sinc",
    )
    unattenuated = simulate_propagation(
        broadband_signal, FS, positions, geometric_attenuation=False, **common
    )
    attenuated = simulate_propagation(
        broadband_signal, FS, positions, geometric_attenuation=True, **common
    )
    expected = 1.0 / (unattenuated.toa_seconds * 343.0)
    np.testing.assert_allclose(attenuated.amplitude_factors, expected)
    np.testing.assert_allclose(
        attenuated.channels,
        unattenuated.channels * expected[:, None],
        rtol=2e-15,
        atol=2e-15,
    )


def test_source_position_and_direction_distance_forms_are_equivalent(broadband_signal):
    positions = comparison_arrays()["L-shaped"]
    source = source_position_from_direction(PHI, ELEVATION, 11.0, positions)
    by_position = simulate_propagation(
        broadband_signal,
        FS,
        positions,
        source_position=source,
        propagation_model="spherical",
        delay_method="windowed_sinc",
    )
    by_angles = simulate_propagation(
        broadband_signal,
        FS,
        positions,
        phi=PHI,
        elevation=ELEVATION,
        distance_m=11.0,
        propagation_model="spherical",
        delay_method="windowed_sinc",
    )
    np.testing.assert_allclose(by_position.toa_seconds, by_angles.toa_seconds, atol=1e-17)
    np.testing.assert_allclose(by_position.channels, by_angles.channels, atol=3e-12)
