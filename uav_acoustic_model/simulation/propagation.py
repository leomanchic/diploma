"""Deterministic multi-channel plane/spherical propagation synthesis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from model.geometry import (
    DEFAULT_SOUND_SPEED,
    Pair,
    all_pairs,
    array_centroid,
    direction_angles,
    direction_vector,
    microphone_positions,
    validate_pairs,
)
from model.tdoa import (
    plane_wave_arrival_times,
    source_position_from_direction,
    travel_times,
)
from simulation.fractional_delay import (
    DEFAULT_FIR_LENGTH,
    fractional_delay_valid_region,
    frequency_domain_delay,
    windowed_sinc_delay,
)


@dataclass(frozen=True)
class PropagationResult:
    """Synchronous channels and exact model metadata.

    ``toa_seconds`` contains physical propagation times for the spherical
    model and centroid-referenced relative times for the plane model.
    ``applied_delay_seconds = toa_seconds - min(toa_seconds)`` is what is
    actually synthesized. This allowed common shift leaves every TDOA
    unchanged. ``valid_region`` is a common half-open sample interval.
    """

    channels: NDArray[np.float64]
    sampling_rate_hz: float
    propagation_model: str
    microphone_positions: NDArray[np.float64]
    source_position: NDArray[np.float64]
    direction: NDArray[np.float64]
    distance_m: float
    toa_seconds: NDArray[np.float64]
    applied_delay_seconds: NDArray[np.float64]
    applied_delay_samples: NDArray[np.float64]
    pairs: tuple[Pair, ...]
    tdoa_seconds: NDArray[np.float64]
    tdoa_matrix_seconds: NDArray[np.float64]
    amplitude_factors: NDArray[np.float64]
    valid_region: tuple[int, int]
    boundary_guard_samples: int
    delay_method: str


def _positive_finite(value: float, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return checked


def _resolve_source(
    positions: NDArray[np.float64],
    *,
    source_position: ArrayLike | None,
    phi: float | None,
    elevation: float | None,
    distance_m: float | None,
) -> tuple[NDArray[np.float64], float, float, float]:
    centroid = array_centroid(positions)
    if source_position is not None:
        if phi is not None or elevation is not None or distance_m is not None:
            raise ValueError(
                "specify source_position or (phi, elevation, distance_m), not both"
            )
        source = np.asarray(source_position, dtype=float)
        if source.shape != (3,) or not np.all(np.isfinite(source)):
            raise ValueError("source_position must be a finite vector with shape (3,)")
        displacement = source - centroid
        distance = float(np.linalg.norm(displacement))
        if distance == 0.0:
            raise ValueError("source_position must differ from the array centroid")
        resolved_phi, resolved_elevation = direction_angles(displacement)
        return source, resolved_phi, resolved_elevation, distance

    if phi is None or elevation is None or distance_m is None:
        raise ValueError(
            "provide either source_position or all of phi, elevation, distance_m"
        )
    resolved_phi = float(phi)
    resolved_elevation = float(elevation)
    if not np.isfinite(resolved_phi) or not np.isfinite(resolved_elevation):
        raise ValueError("phi and elevation must be finite")
    distance = _positive_finite(distance_m, "distance_m")
    source = source_position_from_direction(
        resolved_phi, resolved_elevation, distance, positions
    )
    return source, resolved_phi, resolved_elevation, distance


def simulate_propagation(
    signal: ArrayLike,
    sampling_rate_hz: float,
    positions: ArrayLike,
    *,
    source_position: ArrayLike | None = None,
    phi: float | None = None,
    elevation: float | None = None,
    distance_m: float | None = None,
    propagation_model: str = "spherical",
    sound_speed: float = DEFAULT_SOUND_SPEED,
    pairs: Iterable[Sequence[int]] | None = None,
    geometric_attenuation: bool = False,
    delay_method: str = "frequency",
    fir_length: int = DEFAULT_FIR_LENGTH,
    frequency_padding_samples: int | None = None,
) -> PropagationResult:
    """Generate deterministic synchronous microphone channels.

    No noise, reflection, wind, GCC-PHAT, SRP-PHAT, or UAV harmonic model is
    involved. A positive per-channel delay always means
    ``channel(t)=source(t-delay)``. Fractional delays are passed unchanged to
    the selected interpolator and are never rounded to integer samples.

    The output length is ``len(signal)+ceil(max(applied_delay_samples))``.
    Samples outside the finite source are zero. Geometric attenuation uses
    ``1/||q-r_m||`` for spherical propagation and the common ``1/R`` for a
    plane wave.
    """

    source_signal = np.asarray(signal, dtype=float)
    if source_signal.ndim != 1 or source_signal.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(source_signal)):
        raise ValueError("signal must be finite")
    sampling_rate = _positive_finite(sampling_rate_hz, "sampling_rate_hz")
    speed = _positive_finite(sound_speed, "sound_speed")
    coordinates = microphone_positions(positions)
    source, resolved_phi, resolved_elevation, resolved_distance = _resolve_source(
        coordinates,
        source_position=source_position,
        phi=phi,
        elevation=elevation,
        distance_m=distance_m,
    )
    model = str(propagation_model).lower()
    if model not in {"spherical", "plane"}:
        raise ValueError("propagation_model must be 'spherical' or 'plane'")
    if model == "spherical":
        toa = travel_times(source, coordinates, speed)
        physical_distances = toa * speed
    else:
        toa = plane_wave_arrival_times(
            resolved_phi,
            resolved_elevation,
            coordinates,
            speed,
            remove_common=False,
        )
        physical_distances = np.full(coordinates.shape[0], resolved_distance)

    checked_pairs = (
        all_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    applied_delay_seconds = toa - np.min(toa)
    applied_delay_seconds[np.abs(applied_delay_seconds) < 32.0 * np.finfo(float).eps] = 0.0
    applied_delay_samples = applied_delay_seconds * sampling_rate
    output_length = source_signal.size + int(np.ceil(np.max(applied_delay_samples)))

    method = str(delay_method).lower()
    if method not in {"frequency", "windowed_sinc"}:
        raise ValueError("delay_method must be 'frequency' or 'windowed_sinc'")
    fir_length_checked = int(fir_length)
    if fir_length_checked < 3 or fir_length_checked % 2 == 0:
        raise ValueError("fir_length must be an odd integer of at least 3")
    channels = []
    for delay in applied_delay_samples:
        if method == "frequency":
            channel = frequency_domain_delay(
                source_signal,
                float(delay),
                output_length=output_length,
                padding_samples=frequency_padding_samples,
            )
        else:
            channel = windowed_sinc_delay(
                source_signal,
                float(delay),
                output_length=output_length,
                fir_length=fir_length_checked,
            )
        channels.append(channel)
    channel_matrix = np.asarray(channels)
    if geometric_attenuation:
        amplitude_factors = 1.0 / physical_distances
        channel_matrix = channel_matrix * amplitude_factors[:, None]
    else:
        amplitude_factors = np.ones(coordinates.shape[0])

    tdoa = np.asarray([toa[first] - toa[second] for first, second in checked_pairs])
    tdoa_matrix = toa[:, None] - toa[None, :]
    boundary_guard = (fir_length_checked - 1) // 2
    valid_region = fractional_delay_valid_region(
        source_signal.size,
        applied_delay_samples,
        output_length=output_length,
        boundary_guard_samples=boundary_guard,
    )
    return PropagationResult(
        channels=channel_matrix,
        sampling_rate_hz=sampling_rate,
        propagation_model=model,
        microphone_positions=coordinates.copy(),
        source_position=source.copy(),
        direction=direction_vector(resolved_phi, resolved_elevation),
        distance_m=resolved_distance,
        toa_seconds=toa.copy(),
        applied_delay_seconds=applied_delay_seconds.copy(),
        applied_delay_samples=applied_delay_samples.copy(),
        pairs=checked_pairs,
        tdoa_seconds=tdoa,
        tdoa_matrix_seconds=tdoa_matrix,
        amplitude_factors=amplitude_factors,
        valid_region=valid_region,
        boundary_guard_samples=boundary_guard,
        delay_method=method,
    )
