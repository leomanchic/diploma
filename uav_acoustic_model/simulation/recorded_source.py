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
    origin_asset_id: str
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
    if payload.get("schema_version") not in {1, 2}:
        raise ValueError("recorded-source manifest schema_version must equal 1 or 2")
    recordings = payload.get("recordings")
    if not isinstance(recordings, list) or not recordings:
        raise ValueError("recorded-source manifest must contain recordings")
    identifiers = [str(item.get("recording_id", "")) for item in recordings]
    if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("recording_id values must be non-empty and unique")
    for item in recordings:
        for field in ("session_id", "source_page_url", "license", "local_path", "sha256"):
            if not str(item.get(field, "")):
                raise ValueError(f"recording entry must declare {field}")
        if payload.get("schema_version") == 2:
            if not str(item.get("origin_asset_id", "")):
                raise ValueError("schema 2 recording entry must declare origin_asset_id")
            membership = item.get("split_membership")
            if not isinstance(membership, list) or not membership:
                raise ValueError("schema 2 recording entry must declare split_membership")
            if not set(membership) <= {"calibration", "evaluation"}:
                raise ValueError("split_membership contains an unknown split")
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
    manifest: dict[str, Any],
    recording_id: str | None = None,
    *,
    minimum_sessions_per_split: int | None = None,
) -> dict[str, object]:
    """Derive split independence from IDs and provenance, never a boolean flag.

    With ``recording_id=None`` the schema-2 ``split_membership`` declarations
    define the candidate independent split.  Passing a recording ID performs a
    backwards-compatible audit of the intervals contained in that one asset;
    this is how the historical single-session pilot remains reproducible.
    Files, fragments or transcodes that share either ``session_id`` or
    ``origin_asset_id`` count as one source session.
    """

    if recording_id is None:
        entries = list(manifest["recordings"])
        protocol = manifest.get("independent_split_protocol", {})
        default_minimum = int(protocol.get("minimum_sessions_per_split", 2))
    else:
        entries = [_recording_entry(manifest, recording_id)]
        default_minimum = 1
    minimum = default_minimum if minimum_sessions_per_split is None else int(
        minimum_sessions_per_split
    )
    if minimum < 1:
        raise ValueError("minimum_sessions_per_split must be positive")

    members: dict[str, list[dict[str, Any]]] = {
        "calibration": [],
        "evaluation": [],
    }
    intervals_disjoint = True
    for entry in entries:
        intervals = entry.get("selected_intervals_s", {})
        if recording_id is None and manifest.get("schema_version") == 2:
            membership = tuple(str(value) for value in entry.get("split_membership", ()))
        else:
            membership = tuple(split for split in members if split in intervals)
        for split in membership:
            if split not in members or split not in intervals:
                raise ValueError("split membership must have a selected interval")
            interval = tuple(float(value) for value in intervals[split])
            if len(interval) != 2 or not 0.0 <= interval[0] < interval[1]:
                raise ValueError("each selected interval must be [start, stop]")
            members[split].append(entry)
        if {"calibration", "evaluation"} <= set(membership):
            calibration = tuple(float(value) for value in intervals["calibration"])
            evaluation = tuple(float(value) for value in intervals["evaluation"])
            intervals_disjoint = intervals_disjoint and (
                calibration[1] <= evaluation[0]
                or evaluation[1] <= calibration[0]
            )

    def identifiers(split: str, field: str) -> tuple[str, ...]:
        values = []
        for entry in members[split]:
            if field == "origin_asset_id":
                value = entry.get(field, f"recording:{entry['recording_id']}")
            else:
                value = entry[field]
            values.append(str(value))
        return tuple(sorted(set(values)))

    calibration_recordings = identifiers("calibration", "recording_id")
    evaluation_recordings = identifiers("evaluation", "recording_id")
    calibration_sessions = identifiers("calibration", "session_id")
    evaluation_sessions = identifiers("evaluation", "session_id")
    calibration_origins = identifiers("calibration", "origin_asset_id")
    evaluation_origins = identifiers("evaluation", "origin_asset_id")
    overlapping_sessions = tuple(sorted(set(calibration_sessions) & set(evaluation_sessions)))
    overlapping_origins = tuple(sorted(set(calibration_origins) & set(evaluation_origins)))
    sufficient = all(
        len(values) >= minimum
        for values in (
            calibration_recordings,
            evaluation_recordings,
            calibration_sessions,
            evaluation_sessions,
            calibration_origins,
            evaluation_origins,
        )
    )
    independent = sufficient and not overlapping_sessions and not overlapping_origins
    return {
        "recording_id": None if recording_id is None else str(recording_id),
        "minimum_sessions_per_split": minimum,
        "calibration_recording_ids": calibration_recordings,
        "evaluation_recording_ids": evaluation_recordings,
        "calibration_session_ids": calibration_sessions,
        "evaluation_session_ids": evaluation_sessions,
        "calibration_origin_asset_ids": calibration_origins,
        "evaluation_origin_asset_ids": evaluation_origins,
        "overlapping_session_ids": overlapping_sessions,
        "overlapping_origin_asset_ids": overlapping_origins,
        "calibration_session_count": len(calibration_sessions),
        "evaluation_session_count": len(evaluation_sessions),
        "intervals_disjoint": bool(intervals_disjoint),
        "source_data_independent_between_splits": bool(independent),
        "declared_independence_flag_used": False,
        "scope": (
            "held_out_independent_recording_evaluation"
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
        origin_asset_id=str(
            entry.get("origin_asset_id", f"recording:{entry['recording_id']}")
        ),
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
