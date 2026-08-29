"""Continuous-stream synthesis, chunking, and overlap invariants."""

import numpy as np

from model.geometry import comparison_arrays
from simulation.continuous_stream import extract_overlapping_frames, synthesize_continuous_stream
from simulation.moving_source import simulate_moving_source
from simulation.propagation import simulate_propagation
from simulation.signals import deterministic_bandlimited_signal
from simulation.trajectory import ConstantVelocityTrajectory, StationaryTrajectory


FS = 48_000.0


def test_chunked_moving_source_matches_monolithic_to_strict_tolerances():
    positions = comparison_arrays()["tetrahedral"]
    trajectory = ConstantVelocityTrajectory([25.0, 4.0, 8.0], [17.0, -3.0, 1.0])
    source = deterministic_bandlimited_signal(FS, 0.08, maximum_frequency_hz=10_000.0)
    reception = 0.08 + np.arange(1537) / FS
    kwargs = dict(
        source_start_time_s=-0.04,
        reception_times_s=reception,
        fir_length=129,
    )
    monolithic = simulate_moving_source(source, FS, positions, trajectory, **kwargs)
    chunked = simulate_moving_source(
        source, FS, positions, trajectory, chunk_size_samples=113, **kwargs
    )
    np.testing.assert_allclose(chunked.channels, monolithic.channels, rtol=0.0, atol=3e-12)
    np.testing.assert_allclose(
        chunked.emission_times_s, monolithic.emission_times_s, rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(
        chunked.propagation_delays_s,
        monolithic.propagation_delays_s,
        rtol=0.0,
        atol=2e-15,
    )
    assert chunked.valid_region == monolithic.valid_region
    np.testing.assert_array_equal(chunked.per_channel_valid, monolithic.per_channel_valid)
    assert chunked.synthesis_mode == "chunked"
    assert monolithic.synthesis_mode == "monolithic"
    assert chunked.maximum_interpolation_working_set_elements <= 113 * 129


def test_adjacent_overlapping_frames_share_exact_stream_samples():
    channels = np.arange(4 * 4096, dtype=float).reshape(4, 4096)
    reception = 0.2 + np.arange(4096) / FS
    batch = extract_overlapping_frames(
        channels, reception, frame_length=1024, hop_length=256
    )
    assert batch.overlap_samples == 768
    assert np.shares_memory(batch.frames, channels)
    for index in range(batch.frame_count - 1):
        np.testing.assert_array_equal(
            batch.frames[index, :, 256:], batch.frames[index + 1, :, :-256]
        )


def test_chunked_zero_velocity_matches_existing_static_generator():
    positions = comparison_arrays()["tetrahedral"]
    source_position = np.array([20.0, 5.0, 8.0])
    source = deterministic_bandlimited_signal(FS, 0.04, maximum_frequency_hz=8_000.0)
    static = simulate_propagation(
        source,
        FS,
        positions,
        source_position=source_position,
        delay_method="windowed_sinc",
    )
    chunked = simulate_moving_source(
        source,
        FS,
        positions,
        StationaryTrajectory(source_position),
        chunk_size_samples=127,
    )
    assert chunked.valid_region == static.valid_region
    start, stop = static.valid_region
    np.testing.assert_allclose(
        chunked.channels[:, start:stop], static.channels[:, start:stop], rtol=0.0, atol=3e-12
    )


def test_same_seed_reproduces_complete_continuous_stream_and_noise():
    positions = comparison_arrays()["tetrahedral"]
    trajectory = ConstantVelocityTrajectory([25.0, 3.0, 9.0], [0.0, 12.0, 0.0])
    kwargs = dict(
        duration_s=0.06,
        reception_start_time_s=0.1,
        snr_db=3.0,
        seed=24680,
        chunk_size_samples=257,
    )
    first = synthesize_continuous_stream(positions, trajectory, **kwargs)
    second = synthesize_continuous_stream(positions, trajectory, **kwargs)
    np.testing.assert_array_equal(first.source_signal, second.source_signal)
    np.testing.assert_array_equal(first.noise, second.noise)
    np.testing.assert_array_equal(first.channels, second.channels)
    np.testing.assert_array_equal(
        first.propagation.emission_times_s, second.propagation.emission_times_s
    )
    assert first.source_seed == second.source_seed
    assert first.noise_seed == second.noise_seed
    assert first.noise_generated_once_for_stream is True
