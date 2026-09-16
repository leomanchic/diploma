"""Continuous moving-source audio shared by several microphone stations.

One source waveform is propagated to every physical microphone in every
station.  Noise is generated once per complete station stream, never per
overlapping frame.  All reception timestamps use the common world clock.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from model.geometry import DEFAULT_SOUND_SPEED
from model.station import StationPose
from simulation.continuous_stream import reception_time_grid
from simulation.fractional_delay import DEFAULT_FIR_LENGTH
from simulation.moving_source import MovingSourceResult, simulate_moving_source, solve_emission_time
from simulation.signals import deterministic_bandlimited_signal, random_bandlimited_signal
from simulation.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class StationAudioStream:
    """One station's continuous clean/noise/observed channel matrices."""

    station_id: str
    clean_channels: NDArray[np.float64]
    noise: NDArray[np.float64]
    channels: NDArray[np.float64]
    propagation: MovingSourceResult
    nominal_snr_db: float | None
    effective_snr_db: float | None
    noise_seed: int


@dataclass(frozen=True, slots=True)
class MultistationAudioStream:
    """One common source waveform and synchronous station recordings."""

    source_signal: NDArray[np.float64]
    source_start_time_s: float
    reception_times_s: NDArray[np.float64]
    stations: tuple[StationAudioStream, ...]
    base_seed: int
    source_seed: int
    signal_model: str
    sampling_rate_hz: float
    maximum_emitted_frequency_hz: float
    doppler_bandlimit_checked: bool
    noise_generated_once_per_station_stream: bool = True
    frames_resynthesized_independently: bool = False


def multistation_audio_seeds(base_seed: int, station_count: int) -> tuple[int, tuple[int, ...]]:
    """Return one source seed and disjoint station-noise seeds."""

    count = int(station_count)
    if count < 1:
        raise ValueError("station_count must be positive")
    root = np.random.SeedSequence([int(base_seed), 0x53374341])
    children = root.spawn(count + 1)
    values = tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)
    if len(set(values)) != len(values):
        raise RuntimeError("source/noise seed collision")
    return values[0], values[1:]


def _common_source_support(
    reception_times_s: NDArray[np.float64],
    stations: tuple[StationPose, ...],
    trajectory: Trajectory,
    sampling_rate_hz: float,
    sound_speed: float,
    fir_length: int,
) -> tuple[float, int]:
    microphones = np.vstack([station.microphone_positions_world_m for station in stations])
    endpoint_reception = reception_times_s[[0, -1]]
    endpoint_emission = np.asarray(
        [
            solve_emission_time(endpoint_reception, microphone, trajectory, sound_speed)
            for microphone in microphones
        ]
    )
    guard_samples = max(2 * int(fir_length), 256)
    source_start = float(np.min(endpoint_emission) - guard_samples / sampling_rate_hz)
    source_stop = float(np.max(endpoint_emission) + guard_samples / sampling_rate_hz)
    sample_count = int(np.ceil((source_stop - source_start) * sampling_rate_hz)) + 1
    return source_start, sample_count


def synthesize_multistation_audio(
    stations: tuple[StationPose, ...],
    trajectory: Trajectory,
    *,
    duration_s: float,
    reception_start_time_s: float,
    sampling_rate_hz: float = 48_000.0,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    signal_model: str = "random_broadband",
    snr_db: float | None = 10.0,
    seed: int = 20260918,
    chunk_size_samples: int = 4096,
    fir_length: int = DEFAULT_FIR_LENGTH,
    geometric_attenuation: bool = False,
    maximum_emitted_frequency_hz: float = 10_000.0,
) -> MultistationAudioStream:
    """Generate one continuous source and one continuous recording per station.

    The same source samples and source-time origin feed every station.  AWGN
    uses one independent draw for each complete station channel matrix and a
    station-specific scale that realizes the requested full-stream SNR.
    """

    poses = tuple(stations)
    if not poses or len({station.station_id for station in poses}) != len(poses):
        raise ValueError("stations must be non-empty with unique station_id values")
    sampling_rate = float(sampling_rate_hz)
    speed = float(sound_speed)
    maximum_frequency = float(maximum_emitted_frequency_hz)
    if (
        not np.isfinite(maximum_frequency)
        or not 300.0 < maximum_frequency < sampling_rate / 2.0
    ):
        raise ValueError(
            "maximum_emitted_frequency_hz must satisfy 300 < f_max < Nyquist"
        )
    reception = reception_time_grid(reception_start_time_s, duration_s, sampling_rate)
    source_start, source_count = _common_source_support(
        reception, poses, trajectory, sampling_rate, speed, int(fir_length)
    )
    source_seed, noise_seeds = multistation_audio_seeds(seed, len(poses))
    source_rng = np.random.default_rng(source_seed)
    model = str(signal_model).lower()
    if model == "random_broadband":
        source = random_bandlimited_signal(
            sampling_rate,
            source_count,
            source_rng,
            minimum_frequency_hz=300.0,
            maximum_frequency_hz=maximum_frequency,
            taper_fraction=0.0,
        )
    elif model == "deterministic_multisine":
        source = deterministic_bandlimited_signal(
            sampling_rate,
            source_count / sampling_rate,
            minimum_frequency_hz=300.0,
            maximum_frequency_hz=maximum_frequency,
            taper_fraction=0.0,
            phase_offset_rad=float(source_rng.uniform(0.0, 2.0 * np.pi)),
        )
    else:
        raise ValueError("signal_model must be random_broadband or deterministic_multisine")

    station_streams: list[StationAudioStream] = []
    for station, noise_seed in zip(poses, noise_seeds, strict=True):
        propagation = simulate_moving_source(
            source,
            sampling_rate,
            station.microphone_positions_world_m,
            trajectory,
            source_start_time_s=source_start,
            reception_times_s=reception,
            sound_speed=speed,
            geometric_attenuation=geometric_attenuation,
            fir_length=int(fir_length),
            chunk_size_samples=int(chunk_size_samples),
            maximum_emitted_frequency_hz=maximum_frequency,
        )
        if propagation.valid_region != (0, reception.size):
            raise RuntimeError("common source support does not cover a station stream")
        clean = propagation.channels
        if snr_db is None:
            nominal = effective = None
            noise = np.zeros_like(clean)
        else:
            nominal = float(snr_db)
            if not np.isfinite(nominal):
                raise ValueError("snr_db must be finite or None")
            clean_rms = float(np.sqrt(np.mean(clean**2)))
            sigma = clean_rms / 10.0 ** (nominal / 20.0)
            noise = np.random.default_rng(noise_seed).normal(0.0, sigma, clean.shape)
            noise_rms = float(np.sqrt(np.mean(noise**2)))
            effective = float(20.0 * np.log10(clean_rms / noise_rms))
        station_streams.append(
            StationAudioStream(
                station.station_id,
                clean,
                noise,
                clean + noise,
                propagation,
                nominal,
                effective,
                int(noise_seed),
            )
        )
    return MultistationAudioStream(
        source,
        source_start,
        reception,
        tuple(station_streams),
        int(seed),
        int(source_seed),
        model,
        sampling_rate,
        maximum_frequency,
        True,
    )


__all__ = [
    "MultistationAudioStream",
    "StationAudioStream",
    "multistation_audio_seeds",
    "synthesize_multistation_audio",
]
