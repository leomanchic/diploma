"""Explicit normalized partial whitening; accepted baseline files stay immutable.

The FFT peak extraction below is a static copy of gcc_phat at 430cad8.
Only spectrum preparation changes; SRP uses the accepted cubic lag/search path.
No truth, source class, future data, gate or calibration enters this frontend.
"""
from __future__ import annotations
from dataclasses import replace
from collections.abc import Iterable, Sequence
from time import perf_counter
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline
from scipy.fft import next_fast_len
from model.geometry import Pair, validate_pairs, all_pairs, baselines, direction_vector
from estimators.gcc_phat import (GCCPHATResult, _finite_signal, _positive_sampling_rate,
    _prepare_phat_spectrum, _search_lags, _invalid_result,
    _direct_correlation_from_spectrum, _second_peak_ratio)
from estimators.wls_doa import estimate_doa_wls
from validation.moving_source_study import _delay_bounds, gcc_boundary_flags
from validation.srp_statistical import _srp_from_gcc_diagnostics
BETAS=(1.0, .75, .5)
GCC='all_6_equal_gcc_wls'
SRP='equal_weight_srp_phat'
METHODS=(GCC,SRP)


def checked_beta(beta):
    beta=float(beta)
    if beta not in BETAS: raise ValueError('beta must be one of 1, 0.75, 0.5')
    return beta


def additional_weights(magnitude, band, beta):
    """For beta=1, explicit ones include zero/out-of-band components."""
    beta=checked_beta(beta)
    magnitude=np.asarray(magnitude,float);band=np.asarray(band,bool)
    if magnitude.shape!=band.shape or not np.any(band):
        raise ValueError('nonempty band with matching shape required')
    if not np.all(np.isfinite(magnitude)) or np.any(magnitude<0):
        raise ValueError('finite nonnegative magnitudes required')
    peak=float(np.max(magnitude[band]))
    if peak==0: return np.ones_like(magnitude) if beta==1 else np.zeros_like(magnitude)
    if beta==1: return np.ones_like(magnitude)
    weights=np.zeros_like(magnitude)
    weights[band]=(magnitude[band]/peak)**(1-beta)
    return weights


def prepare_spectrum(first, second, sampling_rate, *, beta, **kwargs):
    beta=checked_beta(beta)
    base=_prepare_phat_spectrum(first,second,sampling_rate,**kwargs)
    # Preserve baseline invalid reasons and every zero exactly.
    if base.invalid_reason is not None: return base
    if beta==1:
        weights=np.ones_like(base.phat_spectrum.real)
    else:
        cross=np.fft.rfft(first,n=base.transform_length)*np.conj(np.fft.rfft(second,n=base.transform_length))
        high=kwargs['maximum_frequency_hz']
        if high is None: high=sampling_rate/2
        band=(base.frequencies_hz>=kwargs['minimum_frequency_hz'])&(base.frequencies_hz<=high)
        weights=additional_weights(np.abs(cross),band,beta)
    return replace(base,phat_spectrum=base.phat_spectrum*weights)


def gcc_partial(
    signal_i: ArrayLike,
    signal_j: ArrayLike,
    sampling_rate_hz: float,
    *,
    beta: float,
    maximum_delay_seconds: float,
    interpolation_factor: int = 32,
    minimum_frequency_hz: float = 0.0,
    maximum_frequency_hz: float | None = None,
    relative_spectral_floor: float = 1e-10,
    minimum_signal_rms: float = 1e-10,
    minimum_spectral_energy_fraction: float = 1e-12,
) -> GCCPHATResult:
    """Estimate ``T_i-T_j`` and return peak/confidence diagnostics.

    ``boundary_hit`` is independent from ``invalid``. A boundary peak can be
    physically valid when the configured search bound is exact; callers may
    reject it using calibration-derived thresholds.
    """

    first = _finite_signal(signal_i, "signal_i")
    second = _finite_signal(signal_j, "signal_j")
    sampling_rate = _positive_sampling_rate(sampling_rate_hz)
    maximum_delay = float(maximum_delay_seconds)
    if not np.isfinite(maximum_delay) or maximum_delay <= 0.0:
        raise ValueError("maximum_delay_seconds must be finite and positive")
    interpolation = int(interpolation_factor)
    if interpolation < 1 or interpolation != interpolation_factor:
        raise ValueError("interpolation_factor must be a positive integer")
    spectrum = prepare_spectrum(
        first,
        second,
        sampling_rate,
        beta=beta,
        minimum_frequency_hz=minimum_frequency_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        relative_spectral_floor=relative_spectral_floor,
        minimum_signal_rms=minimum_signal_rms,
        minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
    )
    maximum_shift, lags_samples = _search_lags(
        maximum_delay, sampling_rate, interpolation, spectrum.transform_length
    )
    if spectrum.invalid_reason is not None:
        return _invalid_result(
            spectrum.invalid_reason,
            lags_samples=lags_samples,
            interpolation=interpolation,
            maximum_delay=maximum_delay,
            used_spectral_energy=spectrum.used_spectral_energy,
            spectral_energy_fraction=spectrum.spectral_energy_fraction,
            used_bin_count=spectrum.used_bin_count,
        )

    oversampled_length = spectrum.transform_length * interpolation
    interpolated_spectrum = spectrum.phat_spectrum.copy()
    # The base transform's Nyquist coefficient represents a single real bin.
    # Once the inverse FFT is lengthened it becomes an ordinary positive-
    # frequency bin with an implicit conjugate, so halve it exactly once.
    if interpolation > 1 and spectrum.transform_length % 2 == 0:
        interpolated_spectrum[-1] *= 0.5
    circular_correlation = interpolation * np.fft.irfft(
        interpolated_spectrum, n=oversampled_length
    )
    negative_lags = (
        circular_correlation[-maximum_shift:]
        if maximum_shift > 0
        else np.empty(0, dtype=float)
    )
    regular_correlation = np.concatenate(
        (negative_lags, circular_correlation[: maximum_shift + 1])
    )
    regular_lags = np.arange(-maximum_shift, maximum_shift + 1, dtype=float) / interpolation
    pieces = []
    if lags_samples[0] < regular_lags[0]:
        pieces.append(
            _direct_correlation_from_spectrum(
                spectrum, np.asarray([lags_samples[0] / sampling_rate])
            )
        )
    pieces.append(regular_correlation)
    if lags_samples[-1] > regular_lags[-1]:
        pieces.append(
            _direct_correlation_from_spectrum(
                spectrum, np.asarray([lags_samples[-1] / sampling_rate])
            )
        )
    correlation = np.concatenate(pieces)
    if not np.all(np.isfinite(correlation)):
        return _invalid_result(
            "nonfinite_correlation",
            lags_samples=lags_samples,
            interpolation=interpolation,
            maximum_delay=maximum_delay,
            used_spectral_energy=spectrum.used_spectral_energy,
            spectral_energy_fraction=spectrum.spectral_energy_fraction,
            used_bin_count=spectrum.used_bin_count,
        )
    integer_peak = int(np.argmax(correlation))
    boundary_hit = integer_peak in {0, correlation.size - 1}
    fractional_peak = 0.0
    curvature = float("nan")
    uniform_neighbours = (
        not boundary_hit
        and np.isclose(
            lags_samples[integer_peak] - lags_samples[integer_peak - 1],
            lags_samples[integer_peak + 1] - lags_samples[integer_peak],
            rtol=0.0,
            atol=32 * np.finfo(float).eps,
        )
    )
    if uniform_neighbours:
        left, centre, right = correlation[integer_peak - 1 : integer_peak + 2]
        curvature = float(centre - 0.5 * (left + right))
        denominator = left - 2.0 * centre + right
        if abs(denominator) > np.finfo(float).eps * max(1.0, abs(centre)):
            fractional_peak = float(
                np.clip(0.5 * (left - right) / denominator, -0.5, 0.5)
            )
    delay_samples = lags_samples[integer_peak] + fractional_peak / interpolation
    ratio = _second_peak_ratio(correlation, integer_peak, interpolation)
    return GCCPHATResult(
        delay_seconds=float(delay_samples / sampling_rate),
        delay_samples=float(delay_samples),
        peak_value=float(correlation[integer_peak]),
        peak_to_second_peak_ratio=float(ratio),
        peak_curvature=curvature,
        boundary_hit=boundary_hit,
        invalid=False,
        invalid_reason=None,
        used_spectral_energy=spectrum.used_spectral_energy,
        spectral_energy_fraction=spectrum.spectral_energy_fraction,
        used_bin_count=spectrum.used_bin_count,
        lags_samples=lags_samples,
        correlation=correlation.copy(),
        interpolation_factor=interpolation,
        maximum_delay_seconds=maximum_delay,
    )


def estimate_tdoas_partial(
    channels: ArrayLike,
    sampling_rate_hz: float,
    pairs: Iterable[Sequence[int]],
    *,
    beta: float,
    maximum_delay_seconds: float | ArrayLike,
    interpolation_factor: int = 32,
    minimum_frequency_hz: float = 0.0,
    maximum_frequency_hz: float | None = None,
    relative_spectral_floor: float = 1e-10,
    minimum_signal_rms: float = 1e-10,
    minimum_spectral_energy_fraction: float = 1e-12,
) -> tuple[NDArray[np.float64], tuple[GCCPHATResult, ...]]:
    """Estimate all oriented pairs, preserving invalid results as ``NaN``."""

    matrix = np.asarray(channels, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("channels must have shape (channel_count, sample_count)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("channels must contain only finite values")
    checked_pairs: tuple[Pair, ...] = validate_pairs(pairs, matrix.shape[0])
    bounds = np.asarray(maximum_delay_seconds, dtype=float)
    if bounds.ndim == 0:
        bounds = np.full(len(checked_pairs), float(bounds))
    if bounds.shape != (len(checked_pairs),):
        raise ValueError("maximum_delay_seconds must be scalar or have one value per pair")
    results = tuple(
        gcc_partial(
            matrix[first],
            matrix[second],
            sampling_rate_hz,
            beta=beta,
            maximum_delay_seconds=float(bound),
            interpolation_factor=interpolation_factor,
            minimum_frequency_hz=minimum_frequency_hz,
            maximum_frequency_hz=maximum_frequency_hz,
            relative_spectral_floor=relative_spectral_floor,
            minimum_signal_rms=minimum_signal_rms,
            minimum_spectral_energy_fraction=minimum_spectral_energy_fraction,
        )
        for (first, second), bound in zip(checked_pairs, bounds, strict=True)
    )
    estimates = np.asarray([result.delay_seconds for result in results])
    return estimates, results


def estimate_frame(
    channels: NDArray[np.float64],
    positions: NDArray[np.float64],
    sampling_rate_hz: float,
    *, beta: float, methods=METHODS,
) -> dict[str, dict[str, object]]:
    pairs = all_pairs(4)
    started = perf_counter()
    observed, diagnostics = estimate_tdoas_partial(
        channels,
        sampling_rate_hz,
        pairs,
        beta=beta,
        maximum_delay_seconds=_delay_bounds(positions, pairs, sampling_rate_hz),
        interpolation_factor=2,
        minimum_frequency_hz=200.0,
        maximum_frequency_hz=10_000.0,
        relative_spectral_floor=1e-8,
    )
    gcc_runtime = perf_counter() - started
    finite = np.isfinite(observed) & ~np.asarray([item.invalid for item in diagnostics])
    method_boundary = gcc_boundary_flags(diagnostics)
    results: dict[str, dict[str, object]] = {}
    for method, indices in (
        ("reference_3_gcc_wls", np.array([0, 1, 2])),
        ("all_6_equal_gcc_wls", np.arange(6)),
    ):
        if method not in methods: continue
        started = perf_counter()
        valid = bool(np.all(finite[indices]))
        direction = np.full(3, np.nan)
        if valid:
            estimate = estimate_doa_wls(
                observed[indices],
                positions,
                tuple(pairs[index] for index in indices),
                sigma_tdoa=1.0 / sampling_rate_hz,
            )
            valid = bool(estimate.success and np.all(np.isfinite(estimate.direction)))
            if valid:
                direction = estimate.direction
        backend_runtime = perf_counter() - started
        used_diagnostics = tuple(diagnostics[index] for index in indices)
        usable_diagnostics = tuple(item for item in used_diagnostics if not item.invalid)
        peak_ratios = tuple(float(item.peak_to_second_peak_ratio) for item in used_diagnostics)
        curvatures = tuple(float(item.peak_curvature) for item in used_diagnostics)
        energies = tuple(float(item.used_spectral_energy) for item in used_diagnostics)
        finite_ratios = np.asarray(peak_ratios)[np.isfinite(peak_ratios)]
        finite_curvatures = np.asarray(curvatures)[np.isfinite(curvatures)]
        results[method] = {
            "direction": direction,
            "valid": valid,
            "boundary": method_boundary[method],
            "shared_gcc_frontend_runtime_s": gcc_runtime,
            "estimator_backend_runtime_s": backend_runtime,
            "total_runtime_s": gcc_runtime + backend_runtime,
            "gcc_frontend_pair_count": 6,
            "estimator_backend_pair_count": int(indices.size),
            "quality_score_probability_claimed": False,
            "gcc_peak_ratios_used_pairs": peak_ratios,
            "gcc_peak_curvatures_used_pairs": curvatures,
            "gcc_spectral_energies_used_pairs": energies,
            "gcc_mean_peak_ratio": float(np.mean(finite_ratios)) if finite_ratios.size else float("nan"),
            "gcc_minimum_peak_ratio": float(np.min(finite_ratios)) if finite_ratios.size else float("nan"),
            "gcc_mean_peak_curvature": float(np.mean(finite_curvatures)) if finite_curvatures.size else float("nan"),
            "gcc_total_spectral_energy": float(np.nansum(energies)),
            "gcc_boundary_count": int(sum(item.boundary_hit for item in used_diagnostics)),
            "gcc_valid_pair_count": len(usable_diagnostics),
        }
    if SRP not in methods: return results
    srp = _srp_from_gcc_diagnostics(
        diagnostics, positions, pairs, sampling_rate_hz, sample_count=channels.shape[1]
    )
    results["equal_weight_srp_phat"] = {
        "direction": srp.direction,
        "valid": not srp.invalid and np.all(np.isfinite(srp.direction)),
        "boundary": srp.boundary_hit,
        "shared_gcc_frontend_runtime_s": gcc_runtime,
        "estimator_backend_runtime_s": srp.runtime_seconds,
        "total_runtime_s": gcc_runtime + srp.runtime_seconds,
        "gcc_frontend_pair_count": 6,
        "estimator_backend_pair_count": 6,
        "quality_score_probability_claimed": False,
        "srp_peak_score": float(srp.score),
        "srp_score_margin": float(srp.score_margin),
        "srp_local_negative_score_hessian": np.asarray(
            srp.local_negative_score_hessian, dtype=float
        ),
        "srp_local_curvature_eigenvalues": np.asarray(
            srp.local_curvature_eigenvalues, dtype=float
        ),
        "srp_used_spectral_energy": float(srp.used_spectral_energy),
        "srp_mean_spectral_energy_fraction": float(srp.mean_spectral_energy_fraction),
    }
    return results


def frame_diagnostics(channels,positions,fs,beta,selected_direction):
    """Inspect actual weighted spectra/curves and the accepted SRP score model."""
    pairs=all_pairs(len(positions))
    _,diagnostics=estimate_tdoas_partial(channels,fs,pairs,beta=beta,
        maximum_delay_seconds=_delay_bounds(positions,pairs,fs),interpolation_factor=2,
        minimum_frequency_hz=200,maximum_frequency_hz=10000,relative_spectral_floor=1e-8)
    az=np.deg2rad(np.arange(0,360,5));el=np.deg2rad(np.arange(0,91,5))
    phi,elevation=np.meshgrid(az,el);grid=direction_vector(phi.ravel(),elevation.ravel())
    data={'beta':np.asarray(beta),'srp_directions':grid,'srp_selected_direction':np.asarray(selected_direction)}
    values=np.zeros(len(grid));delays=grid@baselines(positions,pairs).T/343*fs
    for k,((i,j),d) in enumerate(zip(pairs,diagnostics)):
        data[f'lags_{k}']=d.lags_samples;data[f'correlation_{k}']=d.correlation
        n=next_fast_len(channels.shape[1]*2)
        cross=np.fft.rfft(channels[i],n=n)*np.conj(np.fft.rfft(channels[j],n=n))
        freq=np.fft.rfftfreq(n,1/fs);band=(freq>=200)&(freq<=10000)
        data[f'weights_{k}']=additional_weights(np.abs(cross),band,beta)
        data[f'cross_magnitude_{k}']=np.abs(cross);data['frequencies_hz']=freq
        data[f'active_{k}']=band&(np.abs(cross)>1e-8*np.max(np.abs(cross)[band]))
        spectrum=prepare_spectrum(channels[i],channels[j],fs,beta=beta,minimum_frequency_hz=200,maximum_frequency_hz=10000,relative_spectral_floor=1e-8,minimum_signal_rms=1e-10,minimum_spectral_energy_fraction=1e-12)
        data[f'weighted_spectrum_{k}']=spectrum.phat_spectrum
        data[f'peak_delay_samples_{k}']=np.asarray(d.delay_samples)
        if d.invalid:values[:]=np.nan
        else:values+=CubicSpline(d.lags_samples,d.correlation,extrapolate=False)(delays[:,k])/len(pairs)
    data['srp_scores']=values
    return data
