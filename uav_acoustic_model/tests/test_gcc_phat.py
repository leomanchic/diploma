"""Deterministic GCC-PHAT sign, sub-sample, and generator-equivalence tests."""

import csv

import numpy as np
import pytest

from estimators.gcc_phat import estimate_tdoas_gcc_phat, gcc_phat
from model.geometry import DEFAULT_SOUND_SPEED, all_pairs, comparison_arrays
from simulation.fractional_delay import frequency_domain_delay, windowed_sinc_delay
from simulation.propagation import simulate_propagation
from simulation.signals import deterministic_bandlimited_signal
from validation.gcc_study import (
    benchmark_fractional_delay_methods,
    run_gcc_phat_validation_study,
)

FS = 48_000.0
LOW_HZ = 300.0
HIGH_HZ = 10_000.0


@pytest.fixture(scope="module")
def gcc_signal():
    return deterministic_bandlimited_signal(
        FS,
        0.05,
        minimum_frequency_hz=LOW_HZ,
        maximum_frequency_hz=HIGH_HZ,
        tone_count=31,
    )


def _estimate(first, second, maximum_samples=12.0):
    return gcc_phat(
        first,
        second,
        FS,
        maximum_delay_seconds=maximum_samples / FS,
        interpolation_factor=32,
        minimum_frequency_hz=LOW_HZ,
        maximum_frequency_hz=HIGH_HZ,
    )


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("method", [frequency_domain_delay, windowed_sinc_delay])
def test_gcc_recovers_positive_subsample_delay_with_required_sign(
    gcc_signal, fraction, method
):
    output_length = gcc_signal.size + 8
    channel_i = method(gcc_signal, 3.0 + fraction, output_length=output_length)
    channel_j = method(gcc_signal, 0.25, output_length=output_length)
    result = _estimate(channel_i, channel_j)
    expected = 2.75 + fraction
    assert result.delay_samples == pytest.approx(expected, abs=2e-4)
    assert result.delay_seconds == pytest.approx(expected / FS, abs=2e-4 / FS)
    if not np.isclose(expected, round(expected)):
        assert abs(result.delay_samples - round(result.delay_samples)) > 0.05


def test_pair_orientation_reverses_gcc_sign(gcc_signal):
    first = frequency_domain_delay(gcc_signal, 5.6, output_length=gcc_signal.size + 8)
    second = frequency_domain_delay(gcc_signal, 0.2, output_length=gcc_signal.size + 8)
    forward = _estimate(first, second)
    reverse = _estimate(second, first)
    assert forward.delay_samples == pytest.approx(5.4, abs=2e-4)
    assert reverse.delay_samples == pytest.approx(-5.4, abs=2e-4)
    assert forward.delay_samples == pytest.approx(-reverse.delay_samples, abs=1e-10)


def test_identical_channels_have_zero_gcc_delay(gcc_signal):
    result = _estimate(gcc_signal, gcc_signal)
    assert result.delay_samples == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("array_name", ["square", "tetrahedral"])
@pytest.mark.parametrize("azimuth_deg,elevation_deg", [(20.0, 10.0), (120.0, 50.0)])
def test_all_pair_gcc_matches_metadata_for_both_delay_generators(
    gcc_signal, array_name, azimuth_deg, elevation_deg
):
    positions = comparison_arrays()[array_name]
    pairs = all_pairs(len(positions))
    bounds = np.asarray(
        [
            np.linalg.norm(positions[i] - positions[j]) / DEFAULT_SOUND_SPEED + 2.0 / FS
            for i, j in pairs
        ]
    )
    estimates = {}
    for method in ("frequency", "windowed_sinc"):
        propagation = simulate_propagation(
            gcc_signal,
            FS,
            positions,
            phi=np.deg2rad(azimuth_deg),
            elevation=np.deg2rad(elevation_deg),
            distance_m=20.0,
            propagation_model="plane",
            pairs=pairs,
            delay_method=method,
        )
        estimated, _ = estimate_tdoas_gcc_phat(
            propagation.channels,
            FS,
            pairs,
            maximum_delay_seconds=bounds,
            interpolation_factor=32,
            minimum_frequency_hz=LOW_HZ,
            maximum_frequency_hz=HIGH_HZ,
        )
        estimates[method] = estimated
        assert np.max(np.abs(estimated - propagation.tdoa_seconds)) * FS < 2e-4
    assert np.max(np.abs(estimates["frequency"] - estimates["windowed_sinc"])) * FS < 2e-4


def test_channel_permutation_and_pair_orientation_are_consistent(gcc_signal):
    positions = comparison_arrays()["tetrahedral"]
    propagation = simulate_propagation(
        gcc_signal,
        FS,
        positions,
        phi=np.deg2rad(37.0),
        elevation=np.deg2rad(24.0),
        distance_m=20.0,
        propagation_model="plane",
        delay_method="windowed_sinc",
    )
    baseline = _estimate(propagation.channels[0], propagation.channels[3], 32.0)
    permutation = np.asarray([3, 1, 0, 2])
    permuted = propagation.channels[permutation]
    reversed_pair = _estimate(permuted[0], permuted[2], 32.0)
    assert reversed_pair.delay_samples == pytest.approx(-baseline.delay_samples, abs=1e-10)


def test_validation_study_confirms_generator_equivalence(tmp_path):
    path = tmp_path / "gcc.csv"
    records = run_gcc_phat_validation_study(output_csv=path, signal_duration_s=0.05)
    assert len(records) == 12
    assert max(float(row["frequency_max_abs_error_samples"]) for row in records) < 2e-4
    assert max(float(row["fir_max_abs_error_samples"]) for row in records) < 2e-4
    assert max(float(row["generator_max_difference_samples"]) for row in records) < 2e-4
    assert len(list(csv.DictReader(path.open(encoding="utf-8")))) == 12


def test_benchmark_records_both_methods_without_brittle_speed_assertion(tmp_path):
    path = tmp_path / "benchmark.csv"
    records = benchmark_fractional_delay_methods(
        durations_s=(0.01,), warmup_count=0, repeat_count=2, output_csv=path
    )
    assert {row["method"] for row in records} == {"frequency", "windowed_sinc"}
    assert all(float(row["median_time_s"]) > 0.0 for row in records)
    assert all(float(row["frequency_to_fir_speedup"]) > 0.0 for row in records)
    assert len(list(csv.DictReader(path.open(encoding="utf-8")))) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_delay_seconds": 0.0},
        {"maximum_delay_seconds": 1e-4, "interpolation_factor": 0},
        {"maximum_delay_seconds": 1e-4, "minimum_frequency_hz": 20_000.0,
         "maximum_frequency_hz": 10_000.0},
    ],
)
def test_gcc_rejects_invalid_parameters(gcc_signal, kwargs):
    with pytest.raises(ValueError):
        gcc_phat(gcc_signal, gcc_signal, FS, **kwargs)
