"""Manifest-backed loading of recorded source-waveform approximations.

A recording loaded here is an approximation to an emitted source waveform,
not a free-field source calibration.  It already contains the transfer
function of the original microphone, its recording environment and any
motion present during capture.  Consequently it cannot establish absolute
SPL or a detection-range claim when reused by the propagation simulator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly


@dataclass(frozen=True, slots=True)
class RecordedSourceClip:
    """One declared interval decoded, downmixed and resampled for simulation."""

    recording_id: str
    session_id: str
    split: str
    samples: NDArray[np.float64]
    sampling_rate_hz: float
    maximum_frequency_hz: float
    interval_start_s: float
    interval_stop_s: float
    source_path: Path
    sha256: str
    source_data_independent_between_splits: bool
    approximation_notice: str


def load_recorded_source_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the versioned recorded-source manifest."""

    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("recorded-source manifest schema_version must equal 1")
    recordings = payload.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        raise ValueError("recorded-source manifest must contain recordings")
    identifiers = [str(item.get("recording_id", "")) for item in recordings]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("recording_id values must be non-empty and unique")
    return payload


def _recording_entry(manifest: dict[str, Any], recording_id: str) -> dict[str, Any]:
    matches = [
        item for item in manifest["recordings"]
        if str(item["recording_id"]) == str(recording_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"recording_id {recording_id!r} is not uniquely declared")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recorded_source_split_audit(
    manifest: dict[str, Any], recording_id: str
) -> dict[str, object]:
    """Report whether calibration/evaluation come from independent sessions."""

    entry = _recording_entry(manifest, recording_id)
    intervals = entry.get("selected_intervals_s", {})
    required = {"calibration", "evaluation"}
    if set(intervals) != required:
        raise ValueError("selected_intervals_s must declare calibration and evaluation")
    calibration = tuple(float(value) for value in intervals["calibration"])
    evaluation = tuple(float(value) for value in intervals["evaluation"])
    if len(calibration) != 2 or len(evaluation) != 2:
        raise ValueError("each selected interval must be [start, stop]")
    disjoint_intervals = calibration[1] <= evaluation[0] or evaluation[1] <= calibration[0]
    independent = bool(entry.get("independent_source_split", False))
    return {
        "recording_id": str(entry["recording_id"]),
        "session_id": str(entry["session_id"]),
        "intervals_disjoint": bool(disjoint_intervals),
        "source_data_independent_between_splits": independent,
        "scope": (
            "held_out_recording_evaluation"
            if independent else "single_session_integration_demonstration"
        ),
    }


def load_recorded_source_clip(
    manifest_path: str | Path,
    recording_id: str,
    split: str,
    *,
    target_sampling_rate_hz: float = 48_000.0,
    maximum_frequency_hz: float = 10_000.0,
) -> RecordedSourceClip:
    """Decode one declared interval and return a mono floating-point waveform.

    Stereo/multichannel recordings are averaged.  The interval is resampled by
    a polyphase anti-aliasing filter, de-meaned and peak-normalized to 0.95.
    This deterministic normalization deliberately discards absolute level.
    """

    manifest_path = Path(manifest_path).resolve()
    manifest = load_recorded_source_manifest(manifest_path)
    entry = _recording_entry(manifest, recording_id)
    if split not in {"calibration", "evaluation"}:
        raise ValueError("split must be calibration or evaluation")
    interval = entry["selected_intervals_s"][split]
    start_s, stop_s = (float(interval[0]), float(interval[1]))
    if not (np.isfinite(start_s) and np.isfinite(stop_s) and 0.0 <= start_s < stop_s):
        raise ValueError("selected interval must be finite and increasing")
    source_path = (manifest_path.parent / str(entry["local_path"])).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    actual_hash = _sha256(source_path)
    expected_hash = str(entry["sha256"]).lower()
    if actual_hash != expected_hash:
        raise ValueError("recorded source SHA-256 does not match manifest")

    information = sf.info(source_path)
    if stop_s > information.duration + 0.5 / information.samplerate:
        raise ValueError("selected interval exceeds the recording duration")
    start_frame = int(round(start_s * information.samplerate))
    stop_frame = int(round(stop_s * information.samplerate))
    samples, source_rate = sf.read(
        source_path,
        start=start_frame,
        stop=stop_frame,
        dtype="float64",
        always_2d=True,
    )
    mono = np.mean(samples, axis=1)
    target_rate = float(target_sampling_rate_hz)
    if not np.isfinite(target_rate) or target_rate <= 0.0:
        raise ValueError("target_sampling_rate_hz must be positive and finite")
    if float(source_rate) != target_rate:
        ratio = Fraction(target_rate / float(source_rate)).limit_denominator(100_000)
        mono = resample_poly(mono, ratio.numerator, ratio.denominator)
    maximum_frequency = float(maximum_frequency_hz)
    if not np.isfinite(maximum_frequency) or not 0.0 < maximum_frequency < target_rate / 2.0:
        raise ValueError("maximum_frequency_hz must lie strictly below Nyquist")
    mono = np.asarray(mono - np.mean(mono), dtype=float)
    spectrum = np.fft.rfft(mono)
    frequencies = np.fft.rfftfreq(mono.size, 1.0 / target_rate)
    spectrum[frequencies > maximum_frequency] = 0.0
    mono = np.fft.irfft(spectrum, n=mono.size)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if not np.isfinite(peak) or peak <= np.finfo(float).eps:
        raise ValueError("selected recorded-source interval is silent or non-finite")
    mono *= 0.95 / peak
    audit = recorded_source_split_audit(manifest, recording_id)
    return RecordedSourceClip(
        recording_id=str(entry["recording_id"]),
        session_id=str(entry["session_id"]),
        split=split,
        samples=mono,
        sampling_rate_hz=target_rate,
        maximum_frequency_hz=maximum_frequency,
        interval_start_s=start_s,
        interval_stop_s=stop_s,
        source_path=source_path,
        sha256=actual_hash,
        source_data_independent_between_splits=bool(
            audit["source_data_independent_between_splits"]
        ),
        approximation_notice=str(manifest["approximation_notice"]),
    )


__all__ = [
    "RecordedSourceClip",
    "load_recorded_source_clip",
    "load_recorded_source_manifest",
    "recorded_source_split_audit",
]
