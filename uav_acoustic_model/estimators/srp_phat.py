"""Far-field equal-pair SRP-PHAT with direct and vectorized references.

For every oriented pair ``(i, j)``, ``tau_ij = T_i - T_j`` and the PHAT
cross spectrum is ``X_i * conj(X_j) / |X_i * conj(X_j)|``.  Consequently the
steering phase is ``exp(+j 2 pi f tau_ij(u))`` with
``tau_ij(u) = (r_j-r_i)^T u / c``.  Reversing a pair conjugates its spectrum
and negates its steering delay, leaving the real SRP score unchanged.

The main estimator assigns equal weight to every requested pair.  No GCC
peak-ratio confidence is used.  Input channels may be restricted to a common
half-open ``valid_region`` before the shared GCC-PHAT spectral preparation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from estimators.gcc_phat import (
    _finite_signal,
    _positive_sampling_rate,
    _prepare_phat_spectrum,
)
from model.geometry import (
    DEFAULT_SOUND_SPEED,
    Pair,
    all_pairs,
    baselines,
    direction_angles,
    direction_vector,
    microphone_positions,
    validate_pairs,
)


@dataclass(frozen=True)
class SRPScoreGridResult:
    """SRP scores on an explicitly supplied list of unit directions."""

    directions: NDArray[np.float64]
    scores: NDArray[np.float64]
    invalid: bool
    invalid_reason: str | None
    used_spectral_energy: float
    mean_spectral_energy_fraction: float
    used_bin_count: int
    pair_count: int


@dataclass(frozen=True)
class SRPPHATResult:
    """One coarse-to-fine far-field SRP-PHAT direction estimate."""

    phi: float
    elevation: float
    direction: NDArray[np.float64]
    score: float
    boundary_hit: bool
    invalid: bool
    invalid_reason: str | None
    runtime_seconds: float
    coarse_candidate_count: int
    fine_candidate_count: int
    local_refinement_evaluations: int
    local_refinement_success: bool
    used_spectral_energy: float
    mean_spectral_energy_fraction: float
    used_bin_count: int
    pair_count: int
    pairs: tuple[Pair, ...]
    valid_region: tuple[int, int]
    search_azimuth_bounds_rad: tuple[float, float]
    search_elevation_bounds_rad: tuple[float, float]
    score_margin: float = float("nan")
    local_negative_score_hessian: NDArray[np.float64] = field(
        default_factory=lambda: np.full((2, 2), np.nan)
    )
    local_curvature_eigenvalues: NDArray[np.float64] = field(
        default_factory=lambda: np.full(2, np.nan)
    )


@dataclass(frozen=True)
class _SRPSpectrumData:
    spectra: NDArray[np.complex128]
    frequencies_hz: NDArray[np.float64]
    rfft_weights: NDArray[np.float64]
    normalizer: float
    used_spectral_energy: float
    mean_spectral_energy_fraction: float
    used_bin_count: int
    invalid_reason: str | None


def _channel_matrix(channels: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(channels, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("channels must have shape (M, N), M >= 2, N >= 2")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("channels must contain only finite values")
    return matrix


def _valid_samples(
    channels: NDArray[np.float64], valid_region: tuple[int, int] | None
) -> tuple[NDArray[np.float64], tuple[int, int]]:
    if valid_region is None:
        return channels, (0, channels.shape[1])
    if len(valid_region) != 2:
        raise ValueError("valid_region must be a (start, stop) pair")
    start, stop = int(valid_region[0]), int(valid_region[1])
    if not 0 <= start < stop <= channels.shape[1] or stop - start < 2:
        raise ValueError("valid_region must be a non-empty half-open channel interval")
    return channels[:, start:stop], (start, stop)


def _sound_speed(value: float) -> float:
    speed = float(value)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("sound_speed must be finite and positive")
    return speed


def _local_score_curvature(
    score_function,
    phi: float,
    elevation: float,
    *,
    step_rad: float = np.deg2rad(0.1),
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``-H(score)`` in local angular-arc coordinates.

    Both axes have units of radians of spherical arc. A symmetric stencil is
    required; at an elevation boundary NaNs are returned explicitly.
    """

    h = float(step_rad)
    if elevation <= h or elevation >= np.pi / 2.0 - h:
        return np.full((2, 2), np.nan), np.full(2, np.nan)
    azimuth_step = h / max(np.cos(elevation), 1e-6)

    def evaluate(azimuth_offset: float, elevation_offset: float) -> float:
        vector = direction_vector(
            (phi + azimuth_offset) % (2.0 * np.pi), elevation + elevation_offset
        )[None, :]
        return float(score_function(vector)[0])

    f00 = evaluate(0.0, 0.0)
    fpa, fma = evaluate(azimuth_step, 0.0), evaluate(-azimuth_step, 0.0)
    fpe, fme = evaluate(0.0, h), evaluate(0.0, -h)
    fpp, fpm = evaluate(azimuth_step, h), evaluate(azimuth_step, -h)
    fmp, fmm = evaluate(-azimuth_step, h), evaluate(-azimuth_step, -h)
    values = np.asarray([f00, fpa, fma, fpe, fme, fpp, fpm, fmp, fmm])
    if not np.all(np.isfinite(values)):
        return np.full((2, 2), np.nan), np.full(2, np.nan)
    cross = (fpp - fpm - fmp + fmm) / (4.0 * h**2)
    negative_hessian = -np.asarray(
        [
            [(fpa - 2.0 * f00 + fma) / h**2, cross],
            [cross, (fpe - 2.0 * f00 + fme) / h**2],
        ]
    )
    negative_hessian = 0.5 * (negative_hessian + negative_hessian.T)
    return negative_hessian, np.linalg.eigvalsh(negative_hessian)


def _unit_directions(directions: ArrayLike) -> NDArray[np.float64]:
    vectors = np.asarray(directions, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.shape[0] == 0:
        raise ValueError("directions must have shape (K, 3), K >= 1")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(vectors)) or np.any(norms <= 0.0):
        raise ValueError("directions must be finite and non-zero")
    return vectors / norms[:, None]


def _prepare_srp_spectra(
    channels: NDArray[np.float64],
    sampling_rate_hz: float,
    pairs: tuple[Pair, ...],
    *,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float | None,
    relative_spectral_floor: float,
    minimum_signal_rms: float,
    minimum_spectral_energy_fraction: float,
) -> _SRPSpectrumData:
    spectra = []
    energies = []
    fractions = []
    bins = 0
    frequencies: NDArray[np.float64] | None = None
    transform_length: int | None = None
    invalid_reason: str | None = None
    for first, second in pairs:
        spectrum = _prepare_phat_spectrum(
            _finite_signal(channels[first], f"channels[{first}]"),
            _finite_signal(channels[second], f"channels[{second}]"),
            sampling_rate_hz,
            minimum_frequency_hz=minimum_frequency_hz,
            maximum_frequency_hz=maximum_frequency_hz,
            relative_spectral_floor=relative_spectral_floor,
            minimum_signal_rms=minimum_signal_rms,
            minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
        )
        if frequencies is None:
            frequencies = spectrum.frequencies_hz
            transform_length = spectrum.transform_length
        elif spectrum.transform_length != transform_length:
            raise RuntimeError("pair spectra use inconsistent transform lengths")
        spectra.append(spectrum.phat_spectrum)
        energies.append(spectrum.used_spectral_energy)
        fractions.append(spectrum.spectral_energy_fraction)
        bins += spectrum.used_bin_count
        if spectrum.invalid_reason is not None and invalid_reason is None:
            invalid_reason = f"pair_{first}_{second}:{spectrum.invalid_reason}"
    assert frequencies is not None and transform_length is not None
    weights = np.full(frequencies.size, 2.0)
    weights[0] = 1.0
    if transform_length % 2 == 0:
        weights[-1] = 1.0
    stacked = np.asarray(spectra, dtype=complex)
    normalizer = float(np.sum(weights[None, :] * (np.abs(stacked) > 0.0)))
    if invalid_reason is None and normalizer <= 0.0:
        invalid_reason = "spectral_energy_below_threshold"
    return _SRPSpectrumData(
        spectra=stacked,
        frequencies_hz=frequencies,
        rfft_weights=weights,
        normalizer=normalizer,
        used_spectral_energy=float(np.sum(energies)),
        mean_spectral_energy_fraction=float(np.mean(fractions)),
        used_bin_count=int(bins),
        invalid_reason=invalid_reason,
    )


def _score_context(
    channels: ArrayLike,
    sampling_rate_hz: float,
    positions: ArrayLike,
    pairs: Iterable[Sequence[int]] | None,
    valid_region: tuple[int, int] | None,
    *,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float | None,
    relative_spectral_floor: float,
    minimum_signal_rms: float,
    minimum_spectral_energy_fraction: float,
) -> tuple[
    NDArray[np.float64],
    float,
    tuple[Pair, ...],
    tuple[int, int],
    _SRPSpectrumData,
]:
    matrix = _channel_matrix(channels)
    coordinates = microphone_positions(positions)
    if matrix.shape[0] != coordinates.shape[0]:
        raise ValueError("channel and microphone counts must match")
    sampling_rate = _positive_sampling_rate(sampling_rate_hz)
    samples, selected_region = _valid_samples(matrix, valid_region)
    checked_pairs = (
        all_pairs(matrix.shape[0])
        if pairs is None
        else validate_pairs(pairs, matrix.shape[0])
    )
    spectrum = _prepare_srp_spectra(
        samples,
        sampling_rate,
        checked_pairs,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        relative_spectral_floor=relative_spectral_floor,
        minimum_signal_rms=minimum_signal_rms,
        minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
    )
    return coordinates, sampling_rate, checked_pairs, selected_region, spectrum


def _invalid_grid(
    directions: NDArray[np.float64], spectrum: _SRPSpectrumData, pair_count: int
) -> SRPScoreGridResult:
    return SRPScoreGridResult(
        directions=directions.copy(),
        scores=np.full(directions.shape[0], np.nan),
        invalid=True,
        invalid_reason=spectrum.invalid_reason,
        used_spectral_energy=spectrum.used_spectral_energy,
        mean_spectral_energy_fraction=spectrum.mean_spectral_energy_fraction,
        used_bin_count=spectrum.used_bin_count,
        pair_count=pair_count,
    )


def direct_srp_phat_scores(
    channels: ArrayLike,
    sampling_rate_hz: float,
    positions: ArrayLike,
    directions: ArrayLike,
    *,
    pairs: Iterable[Sequence[int]] | None = None,
    valid_region: tuple[int, int] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    minimum_frequency_hz: float = 0.0,
    maximum_frequency_hz: float | None = None,
    relative_spectral_floor: float = 1e-10,
    minimum_signal_rms: float = 1e-10,
    minimum_spectral_energy_fraction: float = 1e-12,
) -> SRPScoreGridResult:
    r"""Evaluate the defining SRP sum with explicit direction/pair loops."""

    vectors = _unit_directions(directions)
    coordinates, sampling_rate, checked_pairs, _, spectrum = _score_context(
        channels,
        sampling_rate_hz,
        positions,
        pairs,
        valid_region,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        relative_spectral_floor=relative_spectral_floor,
        minimum_signal_rms=minimum_signal_rms,
        minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
    )
    del sampling_rate
    if spectrum.invalid_reason is not None:
        return _invalid_grid(vectors, spectrum, len(checked_pairs))
    delays = vectors @ baselines(coordinates, checked_pairs).T / _sound_speed(sound_speed)
    scores = np.zeros(vectors.shape[0])
    for direction_index in range(vectors.shape[0]):
        total = 0.0
        for pair_index in range(len(checked_pairs)):
            phase = np.exp(
                2j
                * np.pi
                * spectrum.frequencies_hz
                * delays[direction_index, pair_index]
            )
            total += float(
                np.real(
                    np.sum(
                        spectrum.rfft_weights
                        * spectrum.spectra[pair_index]
                        * phase
                    )
                )
            )
        scores[direction_index] = total / spectrum.normalizer
    return SRPScoreGridResult(
        directions=vectors,
        scores=scores,
        invalid=False,
        invalid_reason=None,
        used_spectral_energy=spectrum.used_spectral_energy,
        mean_spectral_energy_fraction=spectrum.mean_spectral_energy_fraction,
        used_bin_count=spectrum.used_bin_count,
        pair_count=len(checked_pairs),
    )


def _vectorized_scores_from_spectrum(
    spectrum: _SRPSpectrumData,
    pair_baselines: NDArray[np.float64],
    directions: NDArray[np.float64],
    sound_speed: float,
    *,
    chunk_size: int = 2048,
) -> NDArray[np.float64]:
    scores = np.zeros(directions.shape[0])
    weighted_spectra = spectrum.spectra * spectrum.rfft_weights[None, :]
    for start in range(0, directions.shape[0], chunk_size):
        stop = min(start + chunk_size, directions.shape[0])
        delays = directions[start:stop] @ pair_baselines.T / sound_speed
        block = np.zeros(stop - start)
        for pair_index in range(pair_baselines.shape[0]):
            phase = np.exp(
                2j
                * np.pi
                * delays[:, pair_index, None]
                * spectrum.frequencies_hz[None, :]
            )
            block += np.real(phase @ weighted_spectra[pair_index])
        scores[start:stop] = block / spectrum.normalizer
    return scores


def vectorized_srp_phat_scores(
    channels: ArrayLike,
    sampling_rate_hz: float,
    positions: ArrayLike,
    directions: ArrayLike,
    *,
    pairs: Iterable[Sequence[int]] | None = None,
    valid_region: tuple[int, int] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    minimum_frequency_hz: float = 0.0,
    maximum_frequency_hz: float | None = None,
    relative_spectral_floor: float = 1e-10,
    minimum_signal_rms: float = 1e-10,
    minimum_spectral_energy_fraction: float = 1e-12,
    chunk_size: int = 2048,
) -> SRPScoreGridResult:
    """Evaluate equal-pair SRP with vectorized direction/frequency blocks."""

    vectors = _unit_directions(directions)
    coordinates, _, checked_pairs, _, spectrum = _score_context(
        channels,
        sampling_rate_hz,
        positions,
        pairs,
        valid_region,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        relative_spectral_floor=relative_spectral_floor,
        minimum_signal_rms=minimum_signal_rms,
        minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
    )
    if spectrum.invalid_reason is not None:
        return _invalid_grid(vectors, spectrum, len(checked_pairs))
    scores = _vectorized_scores_from_spectrum(
        spectrum,
        baselines(coordinates, checked_pairs),
        vectors,
        _sound_speed(sound_speed),
        chunk_size=chunk_size,
    )
    return SRPScoreGridResult(
        directions=vectors,
        scores=scores,
        invalid=False,
        invalid_reason=None,
        used_spectral_energy=spectrum.used_spectral_energy,
        mean_spectral_energy_fraction=spectrum.mean_spectral_energy_fraction,
        used_bin_count=spectrum.used_bin_count,
        pair_count=len(checked_pairs),
    )


def _angle_grid(
    lower: float, upper: float, step: float, *, periodic: bool
) -> NDArray[np.float64]:
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("search bounds must be finite and increasing")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("search steps must be finite and positive")
    if periodic:
        count = max(1, int(np.ceil((upper - lower) / step)))
        return lower + np.arange(count) * (upper - lower) / count
    values = np.arange(lower, upper + step * 0.25, step)
    if values[-1] < upper - 1e-12:
        values = np.append(values, upper)
    values[-1] = min(values[-1], upper)
    return np.unique(values)


def _candidate_directions(
    azimuths: NDArray[np.float64], elevations: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    phi_grid, elevation_grid = np.meshgrid(azimuths, elevations, indexing="xy")
    return (
        phi_grid.ravel(),
        elevation_grid.ravel(),
        direction_vector(phi_grid.ravel(), elevation_grid.ravel()),
    )


def _invalid_estimate(
    reason: str | None,
    runtime: float,
    pairs: tuple[Pair, ...],
    valid_region: tuple[int, int],
    spectrum: _SRPSpectrumData,
    azimuth_bounds: tuple[float, float],
    elevation_bounds: tuple[float, float],
) -> SRPPHATResult:
    return SRPPHATResult(
        phi=float("nan"),
        elevation=float("nan"),
        direction=np.full(3, np.nan),
        score=float("nan"),
        boundary_hit=False,
        invalid=True,
        invalid_reason=reason,
        runtime_seconds=runtime,
        coarse_candidate_count=0,
        fine_candidate_count=0,
        local_refinement_evaluations=0,
        local_refinement_success=False,
        used_spectral_energy=spectrum.used_spectral_energy,
        mean_spectral_energy_fraction=spectrum.mean_spectral_energy_fraction,
        used_bin_count=spectrum.used_bin_count,
        pair_count=len(pairs),
        pairs=pairs,
        valid_region=valid_region,
        search_azimuth_bounds_rad=azimuth_bounds,
        search_elevation_bounds_rad=elevation_bounds,
    )


def srp_phat(
    channels: ArrayLike,
    sampling_rate_hz: float,
    positions: ArrayLike,
    *,
    pairs: Iterable[Sequence[int]] | None = None,
    valid_region: tuple[int, int] | None = None,
    sound_speed: float = DEFAULT_SOUND_SPEED,
    azimuth_bounds_rad: tuple[float, float] = (0.0, 2.0 * np.pi),
    elevation_bounds_rad: tuple[float, float] = (0.0, np.pi / 2.0),
    coarse_to_fine_steps_deg: Sequence[float] = (5.0, 1.0, 0.25),
    local_refinement: bool = True,
    minimum_frequency_hz: float = 0.0,
    maximum_frequency_hz: float | None = None,
    relative_spectral_floor: float = 1e-10,
    minimum_signal_rms: float = 1e-10,
    minimum_spectral_energy_fraction: float = 1e-12,
) -> SRPPHATResult:
    """Estimate one far-field direction with equal-weight all-pair SRP-PHAT."""

    started = perf_counter()
    steps = np.deg2rad(np.asarray(tuple(coarse_to_fine_steps_deg), dtype=float))
    if steps.ndim != 1 or steps.size == 0 or np.any(~np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("coarse_to_fine_steps_deg must contain positive finite values")
    if np.any(np.diff(steps) >= 0.0):
        raise ValueError("coarse_to_fine_steps_deg must be strictly decreasing")
    azimuth_bounds = (float(azimuth_bounds_rad[0]), float(azimuth_bounds_rad[1]))
    elevation_bounds = (
        float(elevation_bounds_rad[0]),
        float(elevation_bounds_rad[1]),
    )
    if not -np.pi / 2.0 <= elevation_bounds[0] < elevation_bounds[1] <= np.pi / 2.0:
        raise ValueError("elevation bounds must lie within [-pi/2, pi/2]")
    coordinates, _, checked_pairs, selected_region, spectrum = _score_context(
        channels,
        sampling_rate_hz,
        positions,
        pairs,
        valid_region,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        relative_spectral_floor=relative_spectral_floor,
        minimum_signal_rms=minimum_signal_rms,
        minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
    )
    if spectrum.invalid_reason is not None:
        return _invalid_estimate(
            spectrum.invalid_reason,
            perf_counter() - started,
            checked_pairs,
            selected_region,
            spectrum,
            azimuth_bounds,
            elevation_bounds,
        )
    pair_baselines = baselines(coordinates, checked_pairs)
    speed = _sound_speed(sound_speed)
    azimuth_span = azimuth_bounds[1] - azimuth_bounds[0]
    periodic_azimuth = np.isclose(azimuth_span, 2.0 * np.pi, rtol=0.0, atol=1e-12)
    azimuths = _angle_grid(
        *azimuth_bounds, steps[0], periodic=periodic_azimuth
    )
    elevations = _angle_grid(*elevation_bounds, steps[0], periodic=False)
    phi_candidates, elevation_candidates, directions = _candidate_directions(
        azimuths, elevations
    )
    scores = _vectorized_scores_from_spectrum(
        spectrum, pair_baselines, directions, speed
    )
    coarse_scores = scores.copy()
    best = int(np.argmax(scores))
    best_phi = float(phi_candidates[best])
    best_elevation = float(elevation_candidates[best])
    best_score = float(scores[best])
    coarse_count = int(scores.size)
    fine_count = 0

    previous_step = steps[0]
    for step in steps[1:]:
        offsets = np.arange(-previous_step, previous_step + step * 0.25, step)
        local_phi = best_phi + offsets
        if periodic_azimuth:
            local_phi = (
                (local_phi - azimuth_bounds[0]) % azimuth_span + azimuth_bounds[0]
            )
        else:
            local_phi = np.clip(local_phi, *azimuth_bounds)
        local_elevation = np.clip(best_elevation + offsets, *elevation_bounds)
        local_phi = np.unique(local_phi)
        local_elevation = np.unique(local_elevation)
        phi_candidates, elevation_candidates, directions = _candidate_directions(
            local_phi, local_elevation
        )
        scores = _vectorized_scores_from_spectrum(
            spectrum, pair_baselines, directions, speed
        )
        fine_count += int(scores.size)
        best = int(np.argmax(scores))
        best_phi = float(phi_candidates[best])
        best_elevation = float(elevation_candidates[best])
        best_score = float(scores[best])
        previous_step = step

    refinement_evaluations = 0
    refinement_success = False
    if local_refinement:
        phi_lower = best_phi - previous_step
        phi_upper = best_phi + previous_step
        if not periodic_azimuth:
            phi_lower = max(phi_lower, azimuth_bounds[0])
            phi_upper = min(phi_upper, azimuth_bounds[1])
        elevation_lower = max(best_elevation - previous_step, elevation_bounds[0])
        elevation_upper = min(best_elevation + previous_step, elevation_bounds[1])

        def objective(angles: NDArray[np.float64]) -> float:
            phi = float(angles[0])
            if periodic_azimuth:
                phi = (phi - azimuth_bounds[0]) % azimuth_span + azimuth_bounds[0]
            vector = direction_vector(phi, float(angles[1]))[None, :]
            return -float(
                _vectorized_scores_from_spectrum(
                    spectrum, pair_baselines, vector, speed
                )[0]
            )

        optimized = minimize(
            objective,
            np.asarray([best_phi, best_elevation]),
            method="L-BFGS-B",
            bounds=((phi_lower, phi_upper), (elevation_lower, elevation_upper)),
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 80},
        )
        refinement_evaluations = int(optimized.nfev)
        if np.all(np.isfinite(optimized.x)) and np.isfinite(optimized.fun):
            candidate_phi = float(optimized.x[0])
            if periodic_azimuth:
                candidate_phi = (
                    (candidate_phi - azimuth_bounds[0]) % azimuth_span
                    + azimuth_bounds[0]
                )
            candidate_elevation = float(optimized.x[1])
            candidate_score = -float(optimized.fun)
            if candidate_score >= best_score - 1e-12:
                best_phi = candidate_phi
                best_elevation = candidate_elevation
                best_score = candidate_score
            refinement_success = bool(optimized.success)

    best_direction = direction_vector(best_phi, best_elevation)
    best_phi, best_elevation = direction_angles(best_direction)
    if periodic_azimuth:
        best_phi = (best_phi - azimuth_bounds[0]) % azimuth_span + azimuth_bounds[0]
    tolerance = max(float(steps[-1]) * 0.51, 1e-10)
    boundary_hit = (
        abs(best_elevation - elevation_bounds[0]) <= tolerance
        or abs(best_elevation - elevation_bounds[1]) <= tolerance
        or (
            not periodic_azimuth
            and (
                abs(best_phi - azimuth_bounds[0]) <= tolerance
                or abs(best_phi - azimuth_bounds[1]) <= tolerance
            )
        )
    )
    second_score = (
        float(np.partition(coarse_scores, -2)[-2])
        if coarse_scores.size >= 2
        else float("nan")
    )

    def local_score(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        return _vectorized_scores_from_spectrum(spectrum, pair_baselines, vectors, speed)

    local_hessian, curvature_eigenvalues = _local_score_curvature(
        local_score, float(best_phi), float(best_elevation)
    )
    return SRPPHATResult(
        phi=float(best_phi),
        elevation=float(best_elevation),
        direction=np.asarray(best_direction, dtype=float),
        score=best_score,
        boundary_hit=bool(boundary_hit),
        invalid=False,
        invalid_reason=None,
        runtime_seconds=perf_counter() - started,
        coarse_candidate_count=coarse_count,
        fine_candidate_count=fine_count,
        local_refinement_evaluations=refinement_evaluations,
        local_refinement_success=refinement_success,
        used_spectral_energy=spectrum.used_spectral_energy,
        mean_spectral_energy_fraction=spectrum.mean_spectral_energy_fraction,
        used_bin_count=spectrum.used_bin_count,
        pair_count=len(checked_pairs),
        pairs=checked_pairs,
        valid_region=selected_region,
        search_azimuth_bounds_rad=azimuth_bounds,
        search_elevation_bounds_rad=elevation_bounds,
        score_margin=float(best_score - second_score),
        local_negative_score_hessian=local_hessian,
        local_curvature_eigenvalues=curvature_eigenvalues,
    )


__all__ = [
    "SRPPHATResult",
    "SRPScoreGridResult",
    "direct_srp_phat_scores",
    "srp_phat",
    "vectorized_srp_phat_scores",
]
