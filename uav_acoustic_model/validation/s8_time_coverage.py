"""Time-weighted availability and coverage from saved S8 publications.

The estimator state at a publication is held until the next publication.
Only the span between the first and final saved publication is measured; no
claim is made before the first availability or after the last publication.
Dependent publication counts are not independent sequence counts.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from validation.independent_recordings_pilot import RESULT_PREFIX, ROOT
from validation.three_station_audio_tracking_study import _write_csv


def time_weighted_coverage_rows(publications: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in publications:
        key = (
            str(row["paired_session_id"]), str(row["source_model_comparison"]),
            str(row["snr_db"]), str(row["estimator_variant"]),
        )
        grouped[key].append(row)
    result = []
    for (session_id, source_model, snr_db, method), subset in sorted(grouped.items()):
        ordered = sorted(subset, key=lambda row: float(row["processing_time_s"]))
        times = np.asarray([float(row["processing_time_s"]) for row in ordered])
        dt = np.diff(times)
        if np.any(dt <= 0) or not np.all(np.isfinite(dt)) or not dt.size:
            raise ValueError("publication times must be strictly increasing")
        valid = np.asarray([str(row["valid"]) == "True" for row in ordered[:-1]])
        confirmed = np.asarray([str(row["confirmed"]) == "True" for row in ordered[:-1]])
        covered = np.asarray([
            str(row["valid_and_covered"]) == "True" for row in ordered[:-1]
        ])
        duration = float(np.sum(dt))
        valid_time = float(np.sum(dt[valid]))
        result.append({
            "split": "evaluation",
            "paired_session_id": session_id,
            "source_model_comparison": source_model,
            "snr_db": snr_db,
            "estimator_variant": method,
            "independent_source_session_trial": True,
            "dependent_publication_count": len(ordered),
            "time_basis": "zero_order_hold_between_consecutive_available_publications",
            "measured_span_start_s": float(times[0]),
            "measured_span_stop_s": float(times[-1]),
            "measured_span_duration_s": duration,
            "confirmed_time_fraction": float(np.sum(dt[confirmed]) / duration),
            "available_estimate_time_fraction": valid_time / duration,
            "coverage_given_available_time": (
                float(np.sum(dt[covered]) / valid_time) if valid_time else float("nan")
            ),
            "available_and_covered_time_fraction": float(np.sum(dt[covered]) / duration),
            "missing_estimate_is_zero_error": False,
        })
    return result


def write_time_coverage(directory: Path = ROOT / "results") -> list[dict[str, object]]:
    with (directory / f"{RESULT_PREFIX}tracking_results.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        publications = list(csv.DictReader(source))
    rows = time_weighted_coverage_rows(publications)
    _write_csv(directory / f"{RESULT_PREFIX}time_coverage.csv", rows)
    return rows


if __name__ == "__main__":
    print("time-coverage rows:", len(write_time_coverage()))
