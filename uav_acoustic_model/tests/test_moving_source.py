"""Strict retarded-time and moving-source propagation tests."""

import numpy as np
from scipy.signal import hilbert

from model.geometry import all_pairs, comparison_arrays
from simulation.moving_source import (
    constant_velocity_emission_time,
    emission_time_residual,
    retarded_time_doppler_factor,
    simulate_moving_source,
    solve_emission_time,
)
from simulation.propagation import simulate_propagation
from simulation.signals import deterministic_bandlimited_signal
from simulation.trajectory import (
    CircularTrajectory,
    ConstantVelocityTrajectory,
    StationaryTrajectory,
)


FS = 48_000.0
C = 343.0


def test_zero_velocity_generator_matches_existing_static_generator():
    positions = comparison_arrays()["tetrahedral"]
    source_position = np.array([20.0, 5.0, 8.0])
    source = deterministic_bandlimited_signal(
        FS, 0.04, maximum_frequency_hz=8_000.0
    )
    static = simulate_propagation(
        source,
        FS,
        positions,
        source_position=source_position,
        delay_method="windowed_sinc",
        pairs=all_pairs(4),
    )
    moving_api = simulate_moving_source(
        source,
        FS,
        positions,
        StationaryTrajectory(source_position),
        pairs=all_pairs(4),
    )
    assert moving_api.channels.shape == static.channels.shape
    assert moving_api.valid_region == static.valid_region
    start, stop = static.valid_region
    np.testing.assert_allclose(
        moving_api.channels[:, start:stop], static.channels[:, start:stop], atol=3e-12
    )
    np.testing.assert_allclose(
        moving_api.tdoa_seconds,
        np.broadcast_to(static.tdoa_seconds[:, None], moving_api.tdoa_seconds.shape),
        rtol=0.0,
        atol=3e-16,
    )


def test_numeric_emission_solver_matches_independent_constant_velocity_formula():
    trajectory = ConstantVelocityTrajectory([30.0, -4.0, 9.0], [22.0, 7.0, -3.0])
    microphone = np.array([0.04, -0.07, 0.03])
    reception = np.linspace(-0.2, 0.8, 401)
    numeric = solve_emission_time(reception, microphone, trajectory)
    analytic = constant_velocity_emission_time(reception, microphone, trajectory)
    np.testing.assert_allclose(numeric, analytic, rtol=0.0, atol=1.2e-15)
    np.testing.assert_allclose(
        emission_time_residual(numeric, reception, microphone, trajectory),
        0.0,
        atol=2e-15,
    )


def test_retarded_time_is_causal_and_monotone_for_general_subsonic_motion():
    trajectory = CircularTrajectory(
        [25.0, -3.0, 7.0],
        radius_m=4.0,
        angular_speed_rad_s=7.0,
        plane_normal=[0.2, 0.4, 1.0],
    )
    reception = np.linspace(0.0, 0.5, 501)
    emission = solve_emission_time(reception, [0.0, 0.0, 0.0], trajectory)
    assert np.all(emission < reception)
    assert np.all(np.diff(emission) > 0.0)
    factor = retarded_time_doppler_factor(emission, [0, 0, 0], trajectory)
    numerical_factor = np.gradient(emission, reception)
    np.testing.assert_allclose(numerical_factor[2:-2], factor[2:-2], rtol=3e-4)


def test_measured_tone_doppler_matches_retarded_time_formula():
    sampling_rate = 16_000.0
    source_start = -0.2
    source_times = source_start + np.arange(int(1.4 * sampling_rate)) / sampling_rate
    frequency = 1_000.0
    source = np.sin(2.0 * np.pi * frequency * source_times)
    trajectory = ConstantVelocityTrajectory([50.0, 0.0, 0.0], [20.0, 0.0, 0.0])
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]])
    result = simulate_moving_source(
        source,
        sampling_rate,
        positions,
        trajectory,
        source_start_time_s=source_start,
        fir_length=65,
    )
    start, stop = result.valid_region
    start += 300
    stop -= 300
    phase = np.unwrap(np.angle(hilbert(result.channels[0, start:stop])))
    slope = np.polyfit(result.reception_times_s[start:stop], phase, 1)[0]
    measured = slope / (2.0 * np.pi)
    expected = frequency / (1.0 + 20.0 / C)
    assert abs(measured - expected) < 0.08


def test_doppler_and_time_varying_tdoa_vanish_as_speed_tends_to_zero():
    positions = comparison_arrays()["tetrahedral"]
    reception = np.linspace(0.05, 0.15, 301)
    variations = []
    factor_errors = []
    for velocity in (1.0, 0.01, 0.0001):
        trajectory = ConstantVelocityTrajectory([30.0, 8.0, 5.0], [velocity, 0, 0])
        emission = np.asarray(
            [solve_emission_time(reception, microphone, trajectory) for microphone in positions]
        )
        delays = reception[None, :] - emission
        tdoa = delays[0] - delays[1]
        variations.append(float(np.ptp(tdoa)))
        factor_errors.append(
            abs(retarded_time_doppler_factor(0.1, positions[0], trajectory) - 1.0)
        )
    assert variations[2] < variations[1] < variations[0]
    assert factor_errors[2] < factor_errors[1] < factor_errors[0]
    assert variations[2] < 1e-10
    assert factor_errors[2] < 3e-7


def test_no_circular_wrap_nan_or_interior_discontinuity():
    positions = comparison_arrays()["square"]
    impulse = np.zeros(4096)
    impulse[-20] = 1.0
    trajectory = ConstantVelocityTrajectory([15.0, 3.0, 4.0], [12.0, -1.0, 0.5])
    result = simulate_moving_source(impulse, FS, positions, trajectory)
    assert np.all(np.isfinite(result.channels))
    assert np.max(np.abs(result.channels[:, :1000])) < 1e-14
    source = deterministic_bandlimited_signal(FS, 0.08, maximum_frequency_hz=8_000.0)
    smooth = simulate_moving_source(source, FS, positions, trajectory)
    start, stop = smooth.valid_region
    assert stop > start
    derivative = np.diff(smooth.channels[:, start:stop], axis=1)
    assert np.max(np.abs(derivative)) < 1.5


def test_microphone_permutation_only_permutes_channels_and_delays():
    positions = comparison_arrays()["tetrahedral"]
    permutation = np.array([2, 0, 3, 1])
    trajectory = ConstantVelocityTrajectory([25.0, 6.0, 9.0], [-8.0, 3.0, 1.0])
    source = deterministic_bandlimited_signal(FS, 0.04, maximum_frequency_hz=8_000.0)
    original = simulate_moving_source(source, FS, positions, trajectory)
    permuted = simulate_moving_source(source, FS, positions[permutation], trajectory)
    np.testing.assert_allclose(permuted.reception_times_s, original.reception_times_s)
    np.testing.assert_allclose(permuted.channels, original.channels[permutation], atol=2e-13)
    np.testing.assert_allclose(
        permuted.propagation_delays_s, original.propagation_delays_s[permutation], atol=2e-16
    )


def test_translation_of_entire_scene_preserves_tdoa_and_channels_without_attenuation():
    positions = comparison_arrays()["tetrahedral"]
    offset = np.array([123.0, -45.0, 17.0])
    trajectory = ConstantVelocityTrajectory([25.0, 6.0, 9.0], [-8.0, 3.0, 1.0])
    translated = ConstantVelocityTrajectory(
        trajectory.position_at_reference_m + offset, trajectory.velocity_mps
    )
    source = deterministic_bandlimited_signal(FS, 0.04, maximum_frequency_hz=8_000.0)
    original = simulate_moving_source(source, FS, positions, trajectory)
    shifted = simulate_moving_source(source, FS, positions + offset, translated)
    np.testing.assert_allclose(shifted.tdoa_seconds, original.tdoa_seconds, atol=2e-16)
    np.testing.assert_allclose(shifted.channels, original.channels, atol=3e-12)


def test_frozen_delay_is_a_diagnostic_constant_tdoa_baseline():
    positions = comparison_arrays()["tetrahedral"]
    source = deterministic_bandlimited_signal(FS, 0.03, maximum_frequency_hz=8_000.0)
    trajectory = ConstantVelocityTrajectory([20.0, 4.0, 7.0], [30.0, 0.0, 0.0])
    frozen = simulate_moving_source(
        source, FS, positions, trajectory, emission_solver="frozen_delay"
    )
    np.testing.assert_allclose(
        np.ptp(frozen.tdoa_seconds, axis=1), 0.0, atol=2e-17
    )
