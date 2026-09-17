"""S8 single-session recorded-source integration demonstration.

The source recording is reused as a waveform approximation by the existing
propagation and tracking chain.  Truth is used only for simulation and offline
scoring; calibration consumes only the declared calibration interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from simulation.multistation_audio import MultistationAudioStream, synthesize_multistation_audio
from simulation.recorded_source import (
    RecordedSourceClip,
    load_recorded_source_clip,
    load_recorded_source_manifest,
    recorded_source_split_audit,
)
from validation.three_station_audio_tracking_study import (
    AudioPilotConfig,
    ESTIMATOR_VARIANTS,
    RESULTS,
    SOURCE_MAXIMUM_FREQUENCY_HZ,
    TRACKER_FRAME_STRIDE,
    _calibration_rows,
    _csv_bearing_row,
    _write_csv,
    bearing_measurements_from_records,
    calibrate_audio_bearings,
    extract_audio_bearing_records,
    pilot_stations,
    run_tracker,
    summarize_pilot,
    summarize_sequence_phases,
    trajectory_for_audio_pilot,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "recorded_sources" / "manifest.json"
RECORDING_ID = "freesound-683298-sadiquecat-mavic-mini-2"
CALIBRATION_BASE_SEED = 20260920
EVALUATION_BASE_SEED = 20260921
SMOKE_BASE_SEED = 20260922
DURATION_S = 3.0
RECEPTION_START_TIME_S = 0.5
CONFIGURATIONS = (
    AudioPilotConfig("constant_velocity", -6.0),
    AudioPilotConfig("constant_velocity", 10.0),
)
RESULT_SCOPE = "single_session_integration_demonstration"


def recorded_sequence_seed(split: str, configuration_index: int) -> int:
    base = {
        "calibration": CALIBRATION_BASE_SEED,
        "evaluation": EVALUATION_BASE_SEED,
        "smoke": SMOKE_BASE_SEED,
    }
    if split not in base:
        raise ValueError("split must be calibration, evaluation or smoke")
    return int(
        np.random.SeedSequence(
            [base[split], 0x53385253, int(configuration_index)]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def _clip_for_split(split: str) -> RecordedSourceClip:
    manifest_split = "calibration" if split == "calibration" else "evaluation"
    return load_recorded_source_clip(
        MANIFEST_PATH,
        RECORDING_ID,
        manifest_split,
        target_sampling_rate_hz=48_000.0,
        maximum_frequency_hz=SOURCE_MAXIMUM_FREQUENCY_HZ,
    )


def generate_recorded_source_sequence(
    config: AudioPilotConfig,
    configuration_index: int,
    split: str,
    *,
    duration_s: float = DURATION_S,
) -> tuple[object, object, MultistationAudioStream, list[dict[str, object]], dict[str, float]]:
    """Generate one continuous multistation sequence from a manifest interval."""

    started = time.perf_counter()
    clip = _clip_for_split(split)
    stations = pilot_stations()
    trajectory = trajectory_for_audio_pilot(config.trajectory_kind, 0)
    synthesis_started = time.perf_counter()
    stream = synthesize_multistation_audio(
        stations,
        trajectory,
        duration_s=duration_s,
        reception_start_time_s=RECEPTION_START_TIME_S,
        sampling_rate_hz=clip.sampling_rate_hz,
        signal_model="recorded_source_approximation",
        snr_db=config.snr_db,
        seed=recorded_sequence_seed(split, configuration_index),
        maximum_emitted_frequency_hz=clip.maximum_frequency_hz,
        external_source_signal=clip.samples,
        source_recording_id=clip.recording_id,
        source_session_id=clip.session_id,
    )
    synthesis_runtime = time.perf_counter() - synthesis_started
    row_split = "calibration" if split == "calibration" else "evaluation"
    rows, frontend_runtime = extract_audio_bearing_records(
        stream,
        stations,
        trajectory,
        split=row_split,
        configuration_index=configuration_index,
        sequence_index=0,
    )
    for row in rows:
        row.update(
            sequence_id=f"s8-recorded-{split}-{configuration_index}",
            source_recording_id=clip.recording_id,
            source_session_id=clip.session_id,
            source_interval_start_s=clip.interval_start_s,
            source_interval_stop_s=clip.interval_stop_s,
            source_data_independent_between_splits=(
                clip.source_data_independent_between_splits
            ),
            result_scope=RESULT_SCOPE,
            source_is_emitted_waveform_approximation=True,
            absolute_spl_or_detection_range_validated=False,
        )
    return stations, trajectory, stream, rows, {
        "audio_synthesis_wall_runtime_s": synthesis_runtime,
        "bearing_frontend_wall_runtime_s": frontend_runtime,
        "audio_pipeline_wall_runtime_s": time.perf_counter() - started,
    }


def _audit_paired_estimators(rows: list[dict[str, object]]) -> None:
    key_sets = {}
    for method in ESTIMATOR_VARIANTS:
        key_sets[method] = {
            (row["sequence_id"], row["station_id"], int(row["frame_index"]))
            for row in rows if row["estimator_variant"] == method
        }
    if key_sets[ESTIMATOR_VARIANTS[0]] != key_sets[ESTIMATOR_VARIANTS[1]]:
        raise RuntimeError("GCC and SRP did not receive identical frame keys")


def _source_calibration_rows(calibrations) -> list[dict[str, object]]:
    rows = _calibration_rows(calibrations)
    for row in rows:
        simulation_count = int(row.pop("independent_sequence_count"))
        row.update(
            simulation_sequence_count=simulation_count,
            independent_source_session_count=1,
            source_recording_id=RECORDING_ID,
            source_data_independent_between_splits=False,
            result_scope=RESULT_SCOPE,
        )
    return rows


def _comparison_rows(recorded_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline_path = RESULTS / "three_station_audio_summary.csv"
    with baseline_path.open("r", encoding="utf-8", newline="") as stream:
        broadband = list(csv.DictReader(stream))
    lookup = {
        (row["trajectory_kind"], float(row["snr_db"]), row["estimator_variant"]): row
        for row in broadband
    }
    metrics = (
        "bearing_rmse_deg_conditional",
        "bearing_p95_deg_conditional",
        "confirmed_publication_fraction",
        "final_valid_sequence_fraction",
        "position_rmse_m_conditional",
        "coverage_conditional",
    )
    result = []
    for row in recorded_summary:
        key = (str(row["trajectory_kind"]), float(row["snr_db"]), str(row["estimator_variant"]))
        baseline = lookup[key]
        comparison = {
            "trajectory_kind": key[0],
            "snr_db": key[1],
            "estimator_variant": key[2],
            "recorded_source_scope": RESULT_SCOPE,
            "recorded_source_session_count": 1,
            "broadband_independent_sequence_count": int(baseline["independent_sequence_count"]),
            "comparison_is_descriptive_only": True,
        }
        for metric in metrics:
            recorded_value = float(row[metric])
            baseline_value = float(baseline[metric])
            comparison[f"recorded_{metric}"] = recorded_value
            comparison[f"broadband_{metric}"] = baseline_value
            comparison[f"recorded_minus_broadband_{metric}"] = recorded_value - baseline_value
        result.append(comparison)
    return result


def run_recorded_source_pilot(
    *, output_directory: Path = RESULTS, progress: bool = True
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Run the frozen one-session calibration/evaluation demonstration."""

    manifest = load_recorded_source_manifest(MANIFEST_PATH)
    audit = recorded_source_split_audit(manifest, RECORDING_ID)
    if audit["source_data_independent_between_splits"]:
        raise RuntimeError("this protocol is explicitly the single-session demonstration")

    calibration_records: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for index, config in enumerate(CONFIGURATIONS):
        _, _, stream, rows, runtimes = generate_recorded_source_sequence(
            config, index, "calibration"
        )
        calibration_records.extend(rows)
        seed_rows.append({
            "split": "calibration",
            "configuration_index": index,
            "sequence_seed": stream.base_seed,
            "source_seed_unused_for_external_recording": stream.source_seed,
            "noise_seeds_json": json.dumps([item.noise_seed for item in stream.stations]),
            "source_recording_id": stream.source_recording_id,
            "source_session_id": stream.source_session_id,
            "source_data_independent_between_splits": False,
            **runtimes,
        })
        if progress:
            print(f"recorded calibration {index + 1}/{len(CONFIGURATIONS)}", flush=True)
    _audit_paired_estimators(calibration_records)
    calibrations = calibrate_audio_bearings(calibration_records, 1)

    evaluation_records: list[dict[str, object]] = []
    tracking_rows: list[dict[str, object]] = []
    update_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    for index, config in enumerate(CONFIGURATIONS):
        stations, trajectory, stream, rows, runtimes = generate_recorded_source_sequence(
            config, index, "evaluation"
        )
        evaluation_records.extend(rows)
        seed_rows.append({
            "split": "evaluation",
            "configuration_index": index,
            "sequence_seed": stream.base_seed,
            "source_seed_unused_for_external_recording": stream.source_seed,
            "noise_seeds_json": json.dumps([item.noise_seed for item in stream.stations]),
            "source_recording_id": stream.source_recording_id,
            "source_session_id": stream.source_session_id,
            "source_data_independent_between_splits": False,
            **runtimes,
        })
        for method in ESTIMATOR_VARIANTS:
            measurements = bearing_measurements_from_records(
                rows, calibrations, method, frame_stride=TRACKER_FRAME_STRIDE
            )
            track, sequence, updates = run_tracker(
                stations, trajectory, measurements, method
            )
            common = {
                "configuration_index": index,
                "sequence_index": 0,
                "sequence_seed": stream.base_seed,
                "trajectory_kind": config.trajectory_kind,
                "snr_db": config.snr_db,
                "source_recording_id": stream.source_recording_id,
                "source_session_id": stream.source_session_id,
                "source_data_independent_between_splits": False,
                "result_scope": RESULT_SCOPE,
            }
            for row in track:
                row.update(common)
            for row in updates:
                row.update(common)
            sequence.update(common, **runtimes)
            phases = summarize_sequence_phases(track, updates, sequence)
            for row in phases:
                row.update(common)
            tracking_rows.extend(track)
            update_rows.extend(updates)
            phase_rows.extend(phases)
            sequence_rows.append(sequence)
        if progress:
            print(f"recorded evaluation {index + 1}/{len(CONFIGURATIONS)}", flush=True)
    _audit_paired_estimators(evaluation_records)

    calibration_noise = {
        value for row in seed_rows if row["split"] == "calibration"
        for value in json.loads(row["noise_seeds_json"])
    }
    evaluation_noise = {
        value for row in seed_rows if row["split"] == "evaluation"
        for value in json.loads(row["noise_seeds_json"])
    }
    if not calibration_noise.isdisjoint(evaluation_noise):
        raise RuntimeError("calibration/evaluation noise seeds overlap")

    summaries = summarize_pilot(
        evaluation_records, tracking_rows, sequence_rows, CONFIGURATIONS
    )
    for row in summaries:
        row.update(
            source_recording_id=RECORDING_ID,
            independent_source_session_count=1,
            source_data_independent_between_splits=False,
            result_scope=RESULT_SCOPE,
            absolute_spl_or_detection_range_validated=False,
        )
    comparisons = _comparison_rows(summaries)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(output_directory / "recorded_source_calibration.csv", _source_calibration_rows(calibrations))
    _write_csv(output_directory / "recorded_source_bearing_results.csv", [_csv_bearing_row(row) for row in evaluation_records])
    _write_csv(output_directory / "recorded_source_tracking_results.csv", tracking_rows)
    _write_csv(output_directory / "recorded_source_update_results.csv", update_rows)
    _write_csv(output_directory / "recorded_source_phase_summary.csv", phase_rows)
    _write_csv(output_directory / "recorded_source_sequence_results.csv", sequence_rows)
    _write_csv(output_directory / "recorded_source_summary.csv", summaries)
    _write_csv(output_directory / "recorded_source_broadband_comparison.csv", comparisons)
    _write_csv(output_directory / "recorded_source_seed_provenance.csv", seed_rows)
    return summaries, comparisons


def smoke_test() -> dict[str, object]:
    config = CONFIGURATIONS[1]
    _, _, calibration_stream, calibration_rows, _ = generate_recorded_source_sequence(
        config, 1, "calibration", duration_s=0.20
    )
    _, _, evaluation_stream, evaluation_rows, _ = generate_recorded_source_sequence(
        config, 1, "evaluation", duration_s=0.20
    )
    _audit_paired_estimators(calibration_rows)
    _audit_paired_estimators(evaluation_rows)
    calibrations = calibrate_audio_bearings(calibration_rows, 1)
    return {
        "recording_id": RECORDING_ID,
        "source_session_id": calibration_stream.source_session_id,
        "same_source_session": (
            calibration_stream.source_session_id == evaluation_stream.source_session_id
        ),
        "source_data_independent_between_splits": False,
        "calibration_valid_bearings": sum(row["valid"] for row in calibration_rows),
        "evaluation_valid_bearings": sum(row["valid"] for row in evaluation_rows),
        "calibration_count": len(calibrations),
        "paired_frame_keys": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke:
        print(json.dumps(smoke_test(), indent=2, sort_keys=True))
    else:
        summary, comparison = run_recorded_source_pilot(progress=True)
        print("summary rows", len(summary), "comparison rows", len(comparison))


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIGURATIONS",
    "MANIFEST_PATH",
    "RECORDING_ID",
    "RESULT_SCOPE",
    "generate_recorded_source_sequence",
    "recorded_sequence_seed",
    "run_recorded_source_pilot",
    "smoke_test",
]
