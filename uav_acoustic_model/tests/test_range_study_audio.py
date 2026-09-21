"""Focused gates for the fixed-background localization-range audio model."""

import numpy as np
import pytest

from model.station import StationPose
from simulation.multistation_audio import (
    multistation_noise_seeds,
    synthesize_multistation_audio,
)
from simulation.trajectory import StationaryTrajectory


def _point_station() -> StationPose:
    # Two coincident microphones isolate the propagation law in this unit
    # test. The production study still uses the frozen tetrahedral arrays.
    return StationPose("S0", [0.0, 0.0, 0.0], np.eye(3), np.zeros((2, 3)))


def _fixed_stream(distance_m: float):
    return synthesize_multistation_audio(
        (_point_station(),),
        StationaryTrajectory([distance_m, 0.0, 0.0]),
        duration_s=0.05,
        reception_start_time_s=1.0,
        sampling_rate_hz=8_000.0,
        maximum_emitted_frequency_hz=3_000.0,
        snr_db=10.0,
        seed=5,
        source_seed=12345,
        noise_seed=67890,
        noise_model="fixed_reference_snr",
        reference_distance_m=100.0,
        reference_source_rms=1.0,
        geometric_attenuation=True,
    )


def test_fixed_background_obeys_inverse_distance_and_six_db_law() -> None:
    near = _fixed_stream(100.0)
    far = _fixed_stream(200.0)
    near_station = near.stations[0]
    far_station = far.stations[0]

    np.testing.assert_array_equal(near.source_signal, far.source_signal)
    np.testing.assert_array_equal(near_station.noise, far_station.noise)
    np.testing.assert_allclose(
        far_station.clean_channels,
        near_station.clean_channels / 2.0,
        rtol=2e-12,
        atol=2e-12,
    )
    near_power = float(np.mean(near_station.clean_channels**2))
    far_power = float(np.mean(far_station.clean_channels**2))
    assert far_power == pytest.approx(near_power / 4.0, rel=3e-12)
    assert near_station.effective_snr_db - far_station.effective_snr_db == pytest.approx(
        20.0 * np.log10(2.0), abs=2e-11
    )


def test_fixed_background_noise_power_is_distance_independent() -> None:
    near = _fixed_stream(50.0).stations[0]
    far = _fixed_stream(1000.0).stations[0]
    np.testing.assert_array_equal(near.noise, far.noise)
    assert near.noise_rms == far.noise_rms
    assert near.noise_model == far.noise_model == "fixed_reference_snr"


def test_legacy_received_snr_mode_remains_available() -> None:
    common = dict(
        stations=(_point_station(),),
        duration_s=0.05,
        reception_start_time_s=1.0,
        sampling_rate_hz=8_000.0,
        maximum_emitted_frequency_hz=3_000.0,
        snr_db=10.0,
        seed=17,
        source_seed=12345,
        noise_seed=67890,
    )
    near = synthesize_multistation_audio(
        trajectory=StationaryTrajectory([100.0, 0.0, 0.0]), **common
    ).stations[0]
    far = synthesize_multistation_audio(
        trajectory=StationaryTrajectory([1000.0, 0.0, 0.0]), **common
    ).stations[0]
    assert near.noise_model == far.noise_model == "received_snr"
    assert near.effective_snr_db == pytest.approx(far.effective_snr_db, abs=2e-12)


def test_fixed_background_rejects_disabled_attenuation_and_seeds_reproduce() -> None:
    first = multistation_noise_seeds(20260921, 3)
    second = multistation_noise_seeds(20260921, 3)
    assert first == second
    assert len(set(first)) == 3
    with pytest.raises(ValueError, match="requires geometric_attenuation"):
        synthesize_multistation_audio(
            (_point_station(),),
            StationaryTrajectory([100.0, 0.0, 0.0]),
            duration_s=0.02,
            reception_start_time_s=1.0,
            sampling_rate_hz=8_000.0,
            maximum_emitted_frequency_hz=3_000.0,
            snr_db=10.0,
            noise_model="fixed_reference_snr",
            geometric_attenuation=False,
        )
