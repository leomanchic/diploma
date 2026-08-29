"""Validation-signal class tests for statistical GCC experiments."""

import numpy as np

from simulation.signals import harmonic_stress_signal, random_bandlimited_signal

FS = 48_000.0
N = 4096


def _out_of_band_energy_fraction(signal, low=300.0, high=10_000.0):
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / FS)
    power = np.abs(np.fft.rfft(signal)) ** 2
    outside = (frequencies < low - 200.0) | (frequencies > high + 200.0)
    return float(np.sum(power[outside]) / np.sum(power))


def test_random_broadband_is_reproducible_by_seed_but_changes_by_realization():
    first = random_bandlimited_signal(FS, N, np.random.default_rng(1234))
    repeated = random_bandlimited_signal(FS, N, np.random.default_rng(1234))
    different = random_bandlimited_signal(FS, N, np.random.default_rng(1235))
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert abs(np.sqrt(np.mean(first**2)) - 1.0) < 2e-15
    assert _out_of_band_energy_fraction(first) < 2e-3


def test_harmonic_stress_signal_has_sparse_harmonic_spectrum_and_seeded_phases():
    first = harmonic_stress_signal(FS, N, rng=np.random.default_rng(55))
    repeated = harmonic_stress_signal(FS, N, rng=np.random.default_rng(55))
    different = harmonic_stress_signal(FS, N, rng=np.random.default_rng(56))
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert abs(np.sqrt(np.mean(first**2)) - 1.0) < 2e-15
    power = np.abs(np.fft.rfft(first)) ** 2
    strongest = np.sort(power)[-96:]
    assert np.sum(strongest) / np.sum(power) > 0.97


def test_harmonic_stress_name_is_not_a_realistic_uav_claim():
    assert "not a realistic UAV model" in harmonic_stress_signal.__doc__
