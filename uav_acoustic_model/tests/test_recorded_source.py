"""Contracts for manifest-backed recorded-source approximations."""

import copy
from pathlib import Path

import numpy as np
import pytest

from simulation.multistation_audio import synthesize_multistation_audio
from simulation.recorded_source import (
    load_recorded_source_clip,
    load_recorded_source_manifest,
    recorded_source_split_audit,
)
from validation.three_station_audio_tracking_study import (
    pilot_stations,
    trajectory_for_audio_pilot,
)


MANIFEST = Path(__file__).resolve().parents[1] / "data" / "recorded_sources" / "manifest.json"
RECORDING_ID = "freesound-683298-sadiquecat-mavic-mini-2"


def test_recorded_source_manifest_provenance_and_single_session_scope():
    manifest = load_recorded_source_manifest(MANIFEST)
    entry = manifest["recordings"][0]
    assert entry["license"] == "CC0-1.0"
    assert entry["source_page_url"].startswith("https://freesound.org/")
    assert entry["recording_conditions"]
    assert entry["original_resolution"] == {
        "sample_rate_hz": 96000,
        "bit_depth": 24,
        "channels": 2,
        "container": "WAV",
    }
    audit = recorded_source_split_audit(manifest, RECORDING_ID)
    assert audit["intervals_disjoint"]
    assert not audit["source_data_independent_between_splits"]
    assert audit["scope"] == "single_session_integration_demonstration"


def test_recorded_source_split_is_derived_from_session_and_origin_ids():
    manifest = load_recorded_source_manifest(MANIFEST)
    audit = recorded_source_split_audit(manifest)
    assert audit["calibration_session_count"] == 2
    assert audit["evaluation_session_count"] == 2
    assert not audit["overlapping_session_ids"]
    assert not audit["overlapping_origin_asset_ids"]
    assert audit["source_data_independent_between_splits"]
    assert audit["scope"] == "held_out_independent_recording_evaluation"
    assert not audit["declared_independence_flag_used"]


def test_independence_flag_cannot_make_one_session_independent():
    manifest = load_recorded_source_manifest(MANIFEST)
    one_session = copy.deepcopy(manifest)
    entry = one_session["recordings"][0]
    entry["split_membership"] = ["calibration", "evaluation"]
    entry["independent_source_split"] = True
    one_session["recordings"] = [entry]
    audit = recorded_source_split_audit(
        one_session, minimum_sessions_per_split=1
    )
    assert audit["overlapping_session_ids"] == (entry["session_id"],)
    assert audit["overlapping_origin_asset_ids"] == (entry["origin_asset_id"],)
    assert not audit["source_data_independent_between_splits"]
    assert not audit["declared_independence_flag_used"]


@pytest.mark.parametrize(
    ("recording_id", "split"),
    [
        ("freesound-263022-alcappuccino-phantom-2", "calibration"),
        ("freesound-383904-simeonradivoev-mini-quadcopter", "evaluation"),
        ("freesound-321687-n-audioman-drone-takeoff", "evaluation"),
    ],
)
def test_all_added_recordings_have_verified_hash_and_load(recording_id, split):
    clip = load_recorded_source_clip(MANIFEST, recording_id, split)
    assert clip.recording_id == recording_id
    assert clip.origin_asset_id.startswith("freesound:")
    assert clip.samples.size == 6 * 48_000
    assert np.all(np.isfinite(clip.samples))


def test_recorded_source_loader_is_deterministic_mono_and_bandlimited():
    first = load_recorded_source_clip(MANIFEST, RECORDING_ID, "calibration")
    second = load_recorded_source_clip(MANIFEST, RECORDING_ID, "calibration")
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.samples.ndim == 1
    assert first.sampling_rate_hz == 48_000.0
    assert first.maximum_frequency_hz == 10_000.0
    assert first.samples.size == 6 * 48_000
    assert np.max(np.abs(first.samples)) == pytest.approx(0.95, rel=0.0, abs=2e-15)
    assert abs(float(np.mean(first.samples))) < 2e-16
    spectrum = np.fft.rfft(first.samples)
    frequencies = np.fft.rfftfreq(first.samples.size, 1.0 / first.sampling_rate_hz)
    assert np.max(np.abs(spectrum[frequencies > first.maximum_frequency_hz])) < 2e-10


def test_recorded_calibration_and_evaluation_intervals_differ_but_share_session():
    calibration = load_recorded_source_clip(MANIFEST, RECORDING_ID, "calibration")
    evaluation = load_recorded_source_clip(MANIFEST, RECORDING_ID, "evaluation")
    assert calibration.session_id == evaluation.session_id
    assert calibration.interval_stop_s <= evaluation.interval_start_s
    assert not calibration.source_data_independent_between_splits
    assert not np.array_equal(calibration.samples, evaluation.samples)


def test_recorded_source_reuses_continuous_propagation_reproducibly():
    clip = load_recorded_source_clip(MANIFEST, RECORDING_ID, "evaluation")
    stations = pilot_stations()
    trajectory = trajectory_for_audio_pilot("constant_velocity", 0)
    arguments = dict(
        stations=stations,
        trajectory=trajectory,
        duration_s=0.04,
        reception_start_time_s=0.5,
        sampling_rate_hz=clip.sampling_rate_hz,
        signal_model="recorded_source_approximation",
        snr_db=10.0,
        seed=8123,
        maximum_emitted_frequency_hz=clip.maximum_frequency_hz,
        external_source_signal=clip.samples,
        source_recording_id=clip.recording_id,
        source_session_id=clip.session_id,
    )
    first = synthesize_multistation_audio(**arguments)
    second = synthesize_multistation_audio(**arguments)
    assert first.source_recording_id == clip.recording_id
    assert first.source_session_id == clip.session_id
    assert first.signal_model == "recorded_source_approximation"
    np.testing.assert_array_equal(first.source_signal, second.source_signal)
    for left, right in zip(first.stations, second.stations, strict=True):
        np.testing.assert_array_equal(left.channels, right.channels)
    assert first.noise_generated_once_per_station_stream
    assert not first.frames_resynthesized_independently


def test_external_source_requires_explicit_recorded_signal_model():
    clip = load_recorded_source_clip(MANIFEST, RECORDING_ID, "evaluation")
    with pytest.raises(ValueError, match="external_source_signal requires"):
        synthesize_multistation_audio(
            pilot_stations(),
            trajectory_for_audio_pilot("constant_velocity", 0),
            duration_s=0.02,
            reception_start_time_s=0.5,
            signal_model="random_broadband",
            external_source_signal=clip.samples,
        )
