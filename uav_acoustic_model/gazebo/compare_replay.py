"""Compare two frozen Gazebo replays with the committed 03dc7d7 baseline."""

from __future__ import annotations

import csv
import io
import json
import math
import subprocess
from pathlib import Path

BASE = "03dc7d7408bc53c783af38279b622bfba8458f6c"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gazebo_offline"
METHODS = ("all_6_equal_gcc_wls", "equal_weight_srp_phat")
TOLERANCE = 1e-12


def _old(kind: str, name: str) -> str:
    path = f"uav_acoustic_model/results/gazebo_offline/{kind}/{name}"
    result = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return result.stdout


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _difference(left: str, right: str) -> float:
    a, b = float(left), float(right)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b)


def _compare_rows(kind: str, name: str, columns: tuple[str, ...],
                  status_columns: tuple[str, ...], method: str | None = None) -> dict:
    before = _rows(_old(kind, name))
    after = _rows((RESULTS / kind / name).read_text())
    if method is not None:
        before = [row for row in before if row["estimator_variant"] == method]
        after = [row for row in after if row["estimator_variant"] == method]
    if len(before) != len(after):
        raise AssertionError(f"{kind}/{name}: row count changed")
    maximum = 0.0
    status_mismatches = 0
    for original, replay in zip(before, after, strict=True):
        for column in columns:
            maximum = max(maximum, _difference(original[column], replay[column]))
        status_mismatches += any(original[column] != replay[column] for column in status_columns)
    if maximum > TOLERANCE or status_mismatches:
        raise AssertionError(f"{kind}/{name}/{method}: numeric difference {maximum}, status mismatches {status_mismatches}")
    return {"rows": len(after), "maximum_numeric_difference": maximum,
            "status_mismatches": status_mismatches}


def compare() -> dict:
    report = {"baseline_commit": BASE, "tolerance_absolute": TOLERANCE, "runs": {}}
    for kind in ("constant_velocity", "smooth_turn"):
        before = json.loads(_old(kind, "summary.json"))
        after = json.loads((RESULTS / kind / "summary.json").read_text())
        run = {"run_id": after["run_id"], "recording_sha256": after["gazebo_state_sha256"], "methods": {}}
        for method in METHODS:
            bearing = _compare_rows(kind, "bearing_results.csv",
                tuple(f"estimate_{space}_{axis}" for space in ("local", "world") for axis in range(3))
                + ("geodesic_error_deg",), ("valid", "invalid_reason"), method)
            track = _compare_rows(kind, f"tracking_{method}.csv",
                tuple(f"estimate_position_{axis}_m" for axis in "xyz") + ("position_error_m",),
                ("valid", "confirmed", "status", "failure_reason"))
            updates = _compare_rows(kind, f"updates_{method}.csv",
                ("processing_time_s", "pre_update_nis"),
                ("station_id", "frame_index", "update_applied", "failure_reason"))
            previous, current = before["methods"][method], after["methods"][method]
            numeric_metrics = ("first_confirmation_time_s", "position_rmse_m_conditional",
                               "availability_fraction")
            metric_maximum = max(abs(previous[key] - current[key]) for key in numeric_metrics)
            if metric_maximum > TOLERANCE or previous["accepted_update_count"] != current["accepted_update_count"]:
                raise AssertionError(f"{kind}/{method}: aggregate metrics changed")
            old_updates = _rows(_old(kind, f"updates_{method}.csv"))
            old_rejected = sum(row["update_applied"] == "False" for row in old_updates)
            if current["rejected_update_count"] != old_rejected:
                raise AssertionError(f"{kind}/{method}: rejection count changed")
            run["methods"][method] = {
                "bearing": bearing, "tracking": track, "updates": updates,
                "first_confirmation_time_s": current["first_confirmation_time_s"],
                "accepted_updates": current["accepted_update_count"],
                "rejected_updates": current["rejected_update_count"],
                "rmse_m": current["position_rmse_m_conditional"],
                "valid_publications": current["valid_publication_count"],
                "total_publications": current["publication_count"],
                "availability_fraction": current["availability_fraction"],
                "maximum_metric_difference": metric_maximum,
            }
        report["runs"][kind] = run
    return report


if __name__ == "__main__":
    result = compare()
    path = RESULTS / "reproducibility_comparison.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(path)
