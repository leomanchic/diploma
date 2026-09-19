"""Replay one saved 2 s S8 bearing stream to diagnose initialization cost.

This reads only historical *estimated* bearings and calibration, never audio or
truth directions.  It is a cost probe, not a second evaluation or a parameter
search.  The budget is the predeclared nonlinear batch-fit count.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from model.measurements import BearingMeasurement
from validation.independent_recordings_pilot import (
    INITIALIZATION_BATCH_OPTIMIZATION_BUDGET,
    INDEPENDENT_TRACKER_FRAME_STRIDE,
)
from validation.three_station_audio_tracking_study import (
    pilot_stations, run_tracker, trajectory_for_audio_pilot,
)


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RESULTS = ROOT / "results"


def replay_historical_cost_probe(
    *, batch_optimization_budget: int = INITIALIZATION_BATCH_OPTIMIZATION_BUDGET,
) -> dict[str, object]:
    """Replay the historically slow mini-quadcopter/-6 dB/GCC stream."""

    with (HISTORICAL_RESULTS / "independent_recordings_calibration.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        calibration = {
            (row["source_model_comparison"], row["station_id"], row["estimator_variant"]): row
            for row in csv.DictReader(source)
        }
    with (HISTORICAL_RESULTS / "independent_recordings_bearing_results.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        rows = [
            row for row in csv.DictReader(source)
            if row["paired_recording_id"] == "freesound-383904-simeonradivoev-mini-quadcopter"
            and row["source_model_comparison"] == "recorded_source_approximation"
            and float(row["snr_db"]) == -6.0
            and row["estimator_variant"] == "all_6_equal_gcc_wls"
            and int(row["frame_index"]) % INDEPENDENT_TRACKER_FRAME_STRIDE == 0
        ]
    with (HISTORICAL_RESULTS / "independent_recordings_session_results.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        historical_runtime = next(
            float(row["tracker_runtime_s"]) for row in csv.DictReader(source)
            if row["paired_recording_id"] == "freesound-383904-simeonradivoev-mini-quadcopter"
            and row["source_model_comparison"] == "recorded_source_approximation"
            and float(row["snr_db"]) == -6.0
            and row["estimator_variant"] == "all_6_equal_gcc_wls"
        )
    events = []
    for row in rows:
        fit = calibration[(row["source_model_comparison"], row["station_id"], row["estimator_variant"])]
        events.append(BearingMeasurement(
            station_id=row["station_id"],
            sequence_id=row["sequence_id"],
            frame_index=int(row["frame_index"]),
            reception_center_timestamp_s=float(row["frame_center_reception_time_s"]),
            available_timestamp_s=float(row["available_timestamp_s"]),
            direction_local=np.asarray([
                float(row[f"estimate_local_{axis}"]) for axis in range(3)
            ]),
            covariance_tangent_rad2=np.asarray([
                [float(fit["covariance_00_rad2"]), float(fit["covariance_01_rad2"])],
                [float(fit["covariance_01_rad2"]), float(fit["covariance_11_rad2"])],
            ]),
            calibration_bias_tangent_rad=np.asarray([
                float(fit["bias_az_arc_rad"]), float(fit["bias_el_arc_rad"]),
            ]),
            estimator_variant=row["estimator_variant"],
            quality_metadata=json.loads(row["quality_metadata_json"]),
        ))
    assert len(events) == 9
    _, result, _ = run_tracker(
        pilot_stations(), trajectory_for_audio_pilot("constant_velocity", 0),
        tuple(events), "all_6_equal_gcc_wls",
        maximum_batch_optimizations_per_generation=batch_optimization_budget,
    )
    return {
        "saved_event_count": len(events),
        "historical_unbounded_runtime_s": historical_runtime,
        "batch_optimization_budget": batch_optimization_budget,
        "batch_optimization_count": result["batch_optimization_count"],
        "batch_optimization_runtime_s": result["batch_optimization_runtime_s"],
        "tracker_runtime_s": result["tracker_runtime_s"],
        "failure_reason": result["failure_reason"],
        "final_confirmed": result["final_confirmed"],
    }


if __name__ == "__main__":
    print(json.dumps(replay_historical_cost_probe(), indent=2, sort_keys=True))
