r"""Retarded-time propagation from a strictly subsonic moving source.

For reception time ``t`` and microphone ``r_m``, the exact emission time is
the unique root

``t = t_e + ||q(t_e)-r_m||/c``.

The received signal is ``x_m(t)=A_m(t_e)s(t_e)``. Natural Doppler follows
from the non-uniform map ``t -> t_e``; no separate frequency shift is added.
The finite source is zero outside its declared support. A Kaiser-windowed sinc
interpolator evaluates fractional emission indices without integer rounding.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from model.geometry import (
    DEFAULT_SOUND_SPEED,
    Pair,
    all_pairs,
    array_centroid,
    microphone_positions,
    validate_pairs,
)
from simulation.fractional_delay import DEFAULT_FIR_LENGTH, DEFAULT_KAISER_BETA
from simulation.trajectory import ConstantVelocityTrajectory, Trajectory


@dataclass(frozen=True)
class MovingSourceResult:
    channels: NDArray[np.float64]
    sampling_rate_hz: float
    microphone_positions: NDArray[np.float64]
    reception_times_s: NDArray[np.float64]
    emission_times_s: NDArray[np.float64]
    propagation_delays_s: NDArray[np.float64]
    distances_m: NDArray[np.float64]
    pairs: tuple[Pair, ...]
    tdoa_seconds: NDArray[np.float64]
    amplitude_factors: NDArray[np.float64]
    valid_region: tuple[int, int]
    per_channel_valid: NDArray[np.bool_]
    source_time_support_s: tuple[float, float]
    emission_solver: str
    interpolation_method: str
    fir_length: int
    boundary_guard_samples: int
    geometric_attenuation: bool


def _positive(value: float, name: str) -> float:
    checked = float(value)
    if not np.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return checked


def _microphone(value: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError("microphone_position must be a finite vector with shape (3,)")
    return result


def _speed_check(trajectory: Trajectory, times: ArrayLike, sound_speed: float) -> None:
    velocities = np.asarray(trajectory.v(times), dtype=float)
    speeds = np.linalg.norm(velocities, axis=-1)
    if np.any(~np.isfinite(speeds)) or np.any(speeds >= sound_speed):
        raise ValueError("trajectory speed must satisfy |v| < sound_speed")


def emission_time_residual(
    emission_time_s: ArrayLike,
    reception_time_s: ArrayLike,
    microphone_position: ArrayLike,
    trajectory: Trajectory,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64]:
    """Evaluate ``t_e + ||q(t_e)-r||/c - t`` in seconds."""

    speed = _positive(sound_speed, "sound_speed")
    emission, reception = np.broadcast_arrays(
        np.asarray(emission_time_s, dtype=float),
        np.asarray(reception_time_s, dtype=float),
    )
    if np.any(~np.isfinite(emission)) or np.any(~np.isfinite(reception)):
        raise ValueError("emission and reception times must be finite")
    microphone = _microphone(microphone_position)
    distance = np.linalg.norm(trajectory.q(emission) - microphone, axis=-1)
    return emission + distance / speed - reception


def solve_emission_time(
    reception_time_s: ArrayLike,
    microphone_position: ArrayLike,
    trajectory: Trajectory,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    *,
    tolerance_s: float = 2e-13,
    maximum_iterations: int = 30,
) -> NDArray[np.float64] | float:
    """Solve retarded time with vectorized Newton and scalar Brent fallback."""

    speed = _positive(sound_speed, "sound_speed")
    tolerance = _positive(tolerance_s, "tolerance_s")
    iterations = int(maximum_iterations)
    if iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    reception = np.asarray(reception_time_s, dtype=float)
    scalar = reception.ndim == 0
    if np.any(~np.isfinite(reception)):
        raise ValueError("reception_time_s must be finite")
    microphone = _microphone(microphone_position)
    _speed_check(trajectory, reception, speed)
    position_at_reception = trajectory.q(reception)
    estimate = reception - np.linalg.norm(
        position_at_reception - microphone, axis=-1
    ) / speed
    converged = np.zeros(reception.shape, dtype=bool)
    for _ in range(iterations):
        position = trajectory.q(estimate)
        velocity = trajectory.v(estimate)
        displacement = position - microphone
        distance = np.linalg.norm(displacement, axis=-1)
        if np.any(distance <= 0.0):
            raise ValueError("source trajectory intersects a microphone")
        radial_velocity = np.sum(displacement * velocity, axis=-1) / distance
        derivative = 1.0 + radial_velocity / speed
        if np.any(derivative <= 0.0):
            raise ValueError("retarded-time map is not monotone; require |v| < c")
        residual = estimate + distance / speed - reception
        step = residual / derivative
        estimate = np.minimum(estimate - step, reception - np.finfo(float).eps)
        converged |= np.abs(step) <= tolerance
        if np.all(converged):
            break

    if not np.all(converged):
        flat_reception = reception.reshape(-1)
        flat_estimate = estimate.reshape(-1)
        flat_converged = converged.reshape(-1)
        for index in np.flatnonzero(~flat_converged):
            target = float(flat_reception[index])

            def residual_scalar(candidate: float) -> float:
                return float(
                    emission_time_residual(
                        candidate, target, microphone, trajectory, speed
                    )
                )

            upper = target
            span = max(abs(target - float(flat_estimate[index])), 1.0 / speed)
            lower = target - span
            for _ in range(80):
                if residual_scalar(lower) < 0.0:
                    break
                span *= 2.0
                lower = target - span
            else:
                raise RuntimeError("could not bracket emission time")
            flat_estimate[index] = brentq(
                residual_scalar, lower, upper, xtol=tolerance, rtol=4 * np.finfo(float).eps
            )
        estimate = flat_estimate.reshape(reception.shape)
    residual = emission_time_residual(estimate, reception, microphone, trajectory, speed)
    if np.any(np.abs(residual) > max(10.0 * tolerance, 2e-12)):
        raise RuntimeError("emission-time solver did not reach requested accuracy")
    if np.any(estimate >= reception):
        raise RuntimeError("non-causal emission time encountered")
    return float(estimate) if scalar else estimate


def constant_velocity_emission_time(
    reception_time_s: ArrayLike,
    microphone_position: ArrayLike,
    trajectory: ConstantVelocityTrajectory,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64] | float:
    """Independent closed-form retarded time for constant velocity.

    With ``p=q(t)-r`` and ``d=t-t_e>0`` the positive root is

    ``d=(-p.v + sqrt((p.v)^2 + (c^2-|v|^2)|p|^2))/(c^2-|v|^2)``.
    """

    if not isinstance(trajectory, ConstantVelocityTrajectory):
        raise TypeError("trajectory must be ConstantVelocityTrajectory")
    speed = _positive(sound_speed, "sound_speed")
    reception = np.asarray(reception_time_s, dtype=float)
    scalar = reception.ndim == 0
    if np.any(~np.isfinite(reception)):
        raise ValueError("reception_time_s must be finite")
    microphone = _microphone(microphone_position)
    velocity = trajectory.velocity_mps
    velocity_squared = float(velocity @ velocity)
    if velocity_squared >= speed**2:
        raise ValueError("trajectory speed must satisfy |v| < sound_speed")
    displacement = trajectory.q(reception) - microphone
    projection = np.sum(displacement * velocity, axis=-1)
    norm_squared = np.sum(displacement**2, axis=-1)
    denominator = speed**2 - velocity_squared
    delay = (
        -projection + np.sqrt(projection**2 + denominator * norm_squared)
    ) / denominator
    result = reception - delay
    if np.any(delay <= 0.0):
        raise ValueError("source trajectory intersects a microphone")
    return float(result) if scalar else result


def retarded_time_doppler_factor(
    emission_time_s: ArrayLike,
    microphone_position: ArrayLike,
    trajectory: Trajectory,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64] | float:
    """Return ``dt_e/dt = 1/(1+v_r/c)`` at an emission time."""

    speed = _positive(sound_speed, "sound_speed")
    emission = np.asarray(emission_time_s, dtype=float)
    scalar = emission.ndim == 0
    microphone = _microphone(microphone_position)
    displacement = trajectory.q(emission) - microphone
    distance = np.linalg.norm(displacement, axis=-1)
    if np.any(distance <= 0.0):
        raise ValueError("source trajectory intersects a microphone")
    radial_velocity = np.sum(displacement * trajectory.v(emission), axis=-1) / distance
    factor = 1.0 / (1.0 + radial_velocity / speed)
    return float(factor) if scalar else factor


def _windowed_sinc_interpolate(
    signal: NDArray[np.float64],
    coordinates: NDArray[np.float64],
    fir_length: int,
    kaiser_beta: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Evaluate a zero-extended finite sequence at fractional indices."""

    length = int(fir_length)
    if length < 3 or length % 2 == 0:
        raise ValueError("fir_length must be an odd integer of at least 3")
    beta = float(kaiser_beta)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("kaiser_beta must be finite and non-negative")
    flat = np.asarray(coordinates, dtype=float).reshape(-1)
    nearest = np.rint(flat)
    flat = np.where(np.abs(flat - nearest) < 32 * np.finfo(float).eps, nearest, flat)
    # ``ceil`` reproduces the anchor used by ``windowed_sinc_delay`` for a
    # positive fractional delay: p=n-d is interpolated around n-floor(d).
    centers = np.ceil(flat).astype(np.int64)
    fractions = flat - centers
    half = (length - 1) // 2
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    indices = centers[:, None] + offsets[None, :]
    weights = np.sinc(offsets[None, :] - fractions[:, None]) * np.kaiser(length, beta)
    weights /= np.sum(weights, axis=1, keepdims=True)
    inside = (indices >= 0) & (indices < signal.size)
    clipped = np.clip(indices, 0, signal.size - 1)
    values = np.sum(weights * signal[clipped] * inside, axis=1)
    valid = (flat >= half) & (flat <= signal.size - 1 - half)
    return values.reshape(coordinates.shape), valid.reshape(coordinates.shape)


def _common_valid_region(mask: NDArray[np.bool_]) -> tuple[int, int]:
    common = np.all(mask, axis=0)
    indices = np.flatnonzero(common)
    if indices.size == 0:
        return (0, 0)
    start, stop = int(indices[0]), int(indices[-1] + 1)
    if not np.all(common[start:stop]):
        raise RuntimeError("valid interpolation region is unexpectedly disconnected")
    return start, stop


def simulate_moving_source(
    signal: ArrayLike,
    sampling_rate_hz: float,
    positions: ArrayLike,
    trajectory: Trajectory,
    *,
    source_start_time_s: float = 0.0,
    reception_times_s: ArrayLike | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    pairs: Iterable[Sequence[int]] | None = None,
    geometric_attenuation: bool = False,
    emission_solver: str = "numeric",
    frozen_emission_time_s: float | None = None,
    fir_length: int = DEFAULT_FIR_LENGTH,
    kaiser_beta: float = DEFAULT_KAISER_BETA,
) -> MovingSourceResult:
    """Generate synchronized channels from exact or frozen retarded time.

    ``emission_solver`` is ``"numeric"``, ``"constant_velocity_analytic"``,
    or diagnostic ``"frozen_delay"``. When reception times are omitted, the
    returned grid spans the earliest arrival of the first source sample to the
    latest arrival of the last source sample. Samples outside source support
    are zero, and ``valid_region`` excludes ``(fir_length-1)/2`` source samples
    at each interpolation boundary.
    """

    source = np.asarray(signal, dtype=float)
    if source.ndim != 1 or source.size == 0 or not np.all(np.isfinite(source)):
        raise ValueError("signal must be a non-empty finite one-dimensional array")
    sampling_rate = _positive(sampling_rate_hz, "sampling_rate_hz")
    speed = _positive(sound_speed, "sound_speed")
    source_start = float(source_start_time_s)
    if not np.isfinite(source_start):
        raise ValueError("source_start_time_s must be finite")
    source_stop = source_start + (source.size - 1) / sampling_rate
    coordinates = microphone_positions(positions)
    checked_pairs = (
        all_pairs(coordinates.shape[0])
        if pairs is None
        else validate_pairs(pairs, coordinates.shape[0])
    )
    method = str(emission_solver).lower()
    if method not in {"numeric", "constant_velocity_analytic", "frozen_delay"}:
        raise ValueError(
            "emission_solver must be numeric, constant_velocity_analytic, or frozen_delay"
        )
    if method == "constant_velocity_analytic" and not isinstance(
        trajectory, ConstantVelocityTrajectory
    ):
        raise TypeError("analytic solver requires ConstantVelocityTrajectory")
    _speed_check(trajectory, [source_start, source_stop], speed)

    if frozen_emission_time_s is None:
        frozen_time = 0.5 * (source_start + source_stop)
    else:
        frozen_time = float(frozen_emission_time_s)
        if not np.isfinite(frozen_time):
            raise ValueError("frozen_emission_time_s must be finite")
    frozen_distances = np.linalg.norm(trajectory.q(frozen_time) - coordinates, axis=1)
    if np.any(frozen_distances <= 0.0):
        raise ValueError("source trajectory intersects a microphone")
    frozen_delays = frozen_distances / speed

    if reception_times_s is None:
        if method == "frozen_delay":
            first_arrivals = source_start + frozen_delays
            last_arrivals = source_stop + frozen_delays
        else:
            first_arrivals = source_start + np.linalg.norm(
                trajectory.q(source_start) - coordinates, axis=1
            ) / speed
            last_arrivals = source_stop + np.linalg.norm(
                trajectory.q(source_stop) - coordinates, axis=1
            ) / speed
        reception_start = float(np.min(first_arrivals))
        reception_stop = float(np.max(last_arrivals))
        count = int(np.ceil((reception_stop - reception_start) * sampling_rate)) + 1
        reception = reception_start + np.arange(count, dtype=float) / sampling_rate
    else:
        reception = np.asarray(reception_times_s, dtype=float)
        if reception.ndim != 1 or reception.size == 0 or not np.all(np.isfinite(reception)):
            raise ValueError("reception_times_s must be a non-empty finite vector")
        if reception.size > 1:
            np.testing.assert_allclose(
                np.diff(reception), 1.0 / sampling_rate, rtol=2e-10, atol=2e-14
            )

    emission_rows = []
    for microphone in coordinates:
        if method == "numeric":
            row = solve_emission_time(reception, microphone, trajectory, speed)
        elif method == "constant_velocity_analytic":
            row = constant_velocity_emission_time(reception, microphone, trajectory, speed)
        else:
            row = reception - frozen_distances[len(emission_rows)] / speed
        emission_rows.append(np.asarray(row, dtype=float))
    emission = np.asarray(emission_rows)
    if np.any(emission >= reception[None, :]):
        raise RuntimeError("moving-source propagation violated causality")

    source_positions = np.asarray([trajectory.q(row) for row in emission])
    displacement = source_positions - coordinates[:, None, :]
    distances = np.linalg.norm(displacement, axis=-1)
    if method == "frozen_delay":
        distances = np.broadcast_to(frozen_distances[:, None], emission.shape).copy()
    delays = reception[None, :] - emission
    source_indices = (emission - source_start) * sampling_rate
    channels, interpolation_valid = _windowed_sinc_interpolate(
        source, source_indices, int(fir_length), float(kaiser_beta)
    )
    if geometric_attenuation:
        amplitude = 1.0 / distances
        channels = channels * amplitude
    else:
        amplitude = np.ones_like(distances)
    tdoa = np.asarray(
        [delays[first] - delays[second] for first, second in checked_pairs]
    )
    if np.any(~np.isfinite(channels)) or np.any(~np.isfinite(tdoa)):
        raise RuntimeError("moving-source synthesis produced non-finite values")
    half = (int(fir_length) - 1) // 2
    return MovingSourceResult(
        channels=channels,
        sampling_rate_hz=sampling_rate,
        microphone_positions=coordinates.copy(),
        reception_times_s=reception,
        emission_times_s=emission,
        propagation_delays_s=delays,
        distances_m=distances,
        pairs=checked_pairs,
        tdoa_seconds=tdoa,
        amplitude_factors=amplitude,
        valid_region=_common_valid_region(interpolation_valid),
        per_channel_valid=interpolation_valid,
        source_time_support_s=(source_start, source_stop),
        emission_solver=method,
        interpolation_method="kaiser_windowed_sinc_time_warp",
        fir_length=int(fir_length),
        boundary_guard_samples=half,
        geometric_attenuation=bool(geometric_attenuation),
    )


def centroid_emission_time(
    reception_time_s: ArrayLike,
    positions: ArrayLike,
    trajectory: Trajectory,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> NDArray[np.float64] | float:
    """Retarded time at the array centroid, used for frame DOA truth."""

    return solve_emission_time(
        reception_time_s, array_centroid(positions), trajectory, sound_speed
    )


__all__ = [
    "MovingSourceResult",
    "centroid_emission_time",
    "constant_velocity_emission_time",
    "emission_time_residual",
    "retarded_time_doppler_factor",
    "simulate_moving_source",
    "solve_emission_time",
]
