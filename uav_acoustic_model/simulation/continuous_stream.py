"""Continuous multichannel synthesis and overlapping frame views.

One source waveform and one noise array are generated for the complete
reception interval.  Frames are views into the resulting channel matrix;
overlap is never resynthesized and therefore contains identical samples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from model.geometry import DEFAULT_SOUND_SPEED, microphone_positions
from simulation.fractional_delay import DEFAULT_FIR_LENGTH
from simulation.moving_source import MovingSourceResult, simulate_moving_source, solve_emission_time
from simulation.signals import deterministic_bandlimited_signal, random_bandlimited_signal
from simulation.trajectory import Trajectory


DEFAULT_STREAM_SAMPLING_RATE_HZ = 48_000.0
DEFAULT_STREAM_DURATION_S = 0.25
DEFAULT_STREAM_FRAME_LENGTH = 1024
DEFAULT_STREAM_HOP_LENGTH = 256
DEFAULT_STREAM_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class ContinuousStreamResult:
    """One continuous source, propagation result, noise, and observed channels."""

    source_signal: NDArray[np.float64]
    source_start_time_s: float
    clean_channels: NDArray[np.float64]
    noise: NDArray[np.float64]
    channels: NDArray[np.float64]
    propagation: MovingSourceResult
    nominal_snr_db: float | None
    effective_snr_db: float | None
    base_seed: int
    source_seed: int
    noise_seed: int
    dropout_intervals_samples: tuple[tuple[int, int], ...]
    noise_generated_once_for_stream: bool = True

    @property
    def reception_times_s(self) -> NDArray[np.float64]:
        return self.propagation.reception_times_s

    @property
    def valid_region(self) -> tuple[int, int]:
        return self.propagation.valid_region


@dataclass(frozen=True)
class OverlappingFrameBatch:
    """Read-only frame views and reception timestamps for one channel stream."""

    frames: NDArray[np.float64]
    start_sample_indices: NDArray[np.int64]
    start_reception_times_s: NDArray[np.float64]
    center_reception_times_s: NDArray[np.float64]
    end_reception_times_s: NDArray[np.float64]
    frame_length: int
    hop_length: int
    overlap_samples: int

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])


def reception_time_grid(
    reception_start_time_s: float,
    duration_s: float,
    sampling_rate_hz: float = DEFAULT_STREAM_SAMPLING_RATE_HZ,
) -> NDArray[np.float64]:
    """Return a uniform half-open reception grid with ``round(duration*fs)`` samples."""

    start = float(reception_start_time_s)
    duration = float(duration_s)
    sampling_rate = float(sampling_rate_hz)
    if not np.isfinite(start):
        raise ValueError("reception_start_time_s must be finite")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    sample_count = int(np.rint(duration * sampling_rate))
    if sample_count < 2:
        raise ValueError("duration_s must contain at least two samples")
    if abs(sample_count / sampling_rate - duration) > 2e-14:
        raise ValueError("duration_s must correspond to an integer number of samples")
    return start + np.arange(sample_count, dtype=float) / sampling_rate


def _stream_source_support(
    reception: NDArray[np.float64],
    positions: NDArray[np.float64],
    trajectory: Trajectory,
    sampling_rate_hz: float,
    sound_speed: float,
    fir_length: int,
) -> tuple[float, int]:
    endpoint_reception = reception[[0, -1]]
    endpoint_emission = np.asarray(
        [
            solve_emission_time(endpoint_reception, microphone, trajectory, sound_speed)
            for microphone in positions
        ]
    )
    guard = max(2 * int(fir_length), 256)
    source_start = float(np.min(endpoint_emission) - guard / sampling_rate_hz)
    source_stop_required = float(np.max(endpoint_emission) + guard / sampling_rate_hz)
    sample_count = int(np.ceil((source_stop_required - source_start) * sampling_rate_hz)) + 1
    return source_start, sample_count


def _stream_seeds(base_seed: int) -> tuple[int, int]:
    sequence = np.random.SeedSequence(int(base_seed))
    source_sequence, noise_sequence = sequence.spawn(2)
    return (
        int(source_sequence.generate_state(1)[0]),
        int(noise_sequence.generate_state(1)[0]),
    )


def _dropout_intervals(
    intervals: Sequence[Sequence[int]], sample_count: int
) -> tuple[tuple[int, int], ...]:
    checked: list[tuple[int, int]] = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("each dropout interval must contain start and stop")
        start, stop = int(interval[0]), int(interval[1])
        if not 0 <= start < stop <= sample_count:
            raise ValueError("dropout interval lies outside the reception stream")
        checked.append((start, stop))
    return tuple(checked)


def synthesize_continuous_stream(
    positions: ArrayLike,
    trajectory: Trajectory,
    *,
    duration_s: float = DEFAULT_STREAM_DURATION_S,
    reception_start_time_s: float = 0.1,
    sampling_rate_hz: float = DEFAULT_STREAM_SAMPLING_RATE_HZ,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    signal_model: str = "random_broadband",
    snr_db: float | None = 10.0,
    seed: int = 20260831,
    chunk_size_samples: int = DEFAULT_STREAM_CHUNK_SIZE,
    fir_length: int = DEFAULT_FIR_LENGTH,
    geometric_attenuation: bool = False,
    dropout_intervals_samples: Sequence[Sequence[int]] = (),
) -> ContinuousStreamResult:
    """Synthesize one continuous source/noise/channel realization.

    ``snr_db`` uses full-stream clean-channel RMS and one scalar AWGN standard
    deviation.  Noise is drawn exactly once over ``channels.shape``. Optional
    all-channel dropout intervals are an explicit data-loss diagnostic and are
    applied after noise; they are not an acoustic propagation model.
    """

    sampling_rate = float(sampling_rate_hz)
    speed = float(sound_speed)
    coordinates = microphone_positions(positions)
    reception = reception_time_grid(reception_start_time_s, duration_s, sampling_rate)
    source_start, source_count = _stream_source_support(
        reception, coordinates, trajectory, sampling_rate, speed, int(fir_length)
    )
    source_seed, noise_seed = _stream_seeds(int(seed))
    source_rng = np.random.default_rng(source_seed)
    model = str(signal_model).lower()
    if model == "random_broadband":
        source = random_bandlimited_signal(
            sampling_rate,
            source_count,
            source_rng,
            minimum_frequency_hz=300.0,
            maximum_frequency_hz=10_000.0,
            taper_fraction=0.0,
        )
    elif model == "deterministic_multisine":
        source = deterministic_bandlimited_signal(
            sampling_rate,
            source_count / sampling_rate,
            minimum_frequency_hz=300.0,
            maximum_frequency_hz=10_000.0,
            taper_fraction=0.0,
        )
    else:
        raise ValueError("signal_model must be random_broadband or deterministic_multisine")
    propagation = simulate_moving_source(
        source,
        sampling_rate,
        coordinates,
        trajectory,
        source_start_time_s=source_start,
        reception_times_s=reception,
        sound_speed=speed,
        geometric_attenuation=geometric_attenuation,
        fir_length=int(fir_length),
        chunk_size_samples=int(chunk_size_samples),
    )
    if propagation.valid_region != (0, reception.size):
        raise RuntimeError("constructed source support does not cover the full stream")
    clean = propagation.channels
    if snr_db is None:
        nominal = None
        noise = np.zeros_like(clean)
        effective = None
    else:
        nominal = float(snr_db)
        if not np.isfinite(nominal):
            raise ValueError("snr_db must be finite or None")
        clean_rms = float(np.sqrt(np.mean(clean**2)))
        noise_standard_deviation = clean_rms / 10.0 ** (nominal / 20.0)
        noise = np.random.default_rng(noise_seed).normal(
            0.0, noise_standard_deviation, size=clean.shape
        )
        realized_noise_rms = float(np.sqrt(np.mean(noise**2)))
        effective = float(20.0 * np.log10(clean_rms / realized_noise_rms))
    observed = clean + noise
    dropouts = _dropout_intervals(dropout_intervals_samples, reception.size)
    if dropouts:
        observed = observed.copy()
        for start, stop in dropouts:
            observed[:, start:stop] = 0.0
    return ContinuousStreamResult(
        source_signal=source,
        source_start_time_s=source_start,
        clean_channels=clean,
        noise=noise,
        channels=observed,
        propagation=propagation,
        nominal_snr_db=nominal,
        effective_snr_db=effective,
        base_seed=int(seed),
        source_seed=source_seed,
        noise_seed=noise_seed,
        dropout_intervals_samples=dropouts,
    )


def extract_overlapping_frames(
    channels: ArrayLike,
    reception_times_s: ArrayLike,
    *,
    frame_length: int = DEFAULT_STREAM_FRAME_LENGTH,
    hop_length: int = DEFAULT_STREAM_HOP_LENGTH,
) -> OverlappingFrameBatch:
    """Return chronological overlapping views into one continuous channel matrix."""

    matrix = np.asarray(channels, dtype=float)
    reception = np.asarray(reception_times_s, dtype=float)
    length = int(frame_length)
    hop = int(hop_length)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or not np.all(np.isfinite(matrix)):
        raise ValueError("channels must be a finite channels-by-samples matrix")
    if reception.shape != (matrix.shape[1],) or not np.all(np.isfinite(reception)):
        raise ValueError("reception_times_s must match the channel sample axis")
    if reception.size > 1 and np.any(np.diff(reception) <= 0.0):
        raise ValueError("reception timestamps must be strictly increasing")
    if length < 2 or length > matrix.shape[1]:
        raise ValueError("frame_length must be between 2 and the stream length")
    if hop < 1 or hop > length:
        raise ValueError("hop_length must satisfy 1 <= hop_length <= frame_length")
    windows = sliding_window_view(matrix, length, axis=1)[:, ::hop, :]
    frames = np.moveaxis(windows, 0, 1)
    frame_count = frames.shape[0]
    starts = np.arange(frame_count, dtype=np.int64) * hop
    ends = starts + length - 1
    start_times = reception[starts]
    end_times = reception[ends]
    center_times = 0.5 * (start_times + end_times)
    return OverlappingFrameBatch(
        frames=frames,
        start_sample_indices=starts,
        start_reception_times_s=start_times,
        center_reception_times_s=center_times,
        end_reception_times_s=end_times,
        frame_length=length,
        hop_length=hop,
        overlap_samples=length - hop,
    )


__all__ = [
    "ContinuousStreamResult",
    "DEFAULT_STREAM_CHUNK_SIZE",
    "DEFAULT_STREAM_DURATION_S",
    "DEFAULT_STREAM_FRAME_LENGTH",
    "DEFAULT_STREAM_HOP_LENGTH",
    "DEFAULT_STREAM_SAMPLING_RATE_HZ",
    "OverlappingFrameBatch",
    "extract_overlapping_frames",
    "reception_time_grid",
    "synthesize_continuous_stream",
]
