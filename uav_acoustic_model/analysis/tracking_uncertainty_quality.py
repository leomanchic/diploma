"""Read-only audit of published track uncertainty and nominal coordinate precision.

This module never synthesizes audio or invokes a tracker. All source experiments
are SHA-checked before new, versioned derivative tables are written.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2

from analysis.localization_error_attribution import _read_csv_gz, _write_csv, _write_csv_gz, _write_json
from analysis.robust_track_confirmation import (
    DEFAULT_OUTPUT as SOURCE, METHODS, ROOT, VARIANTS, _directory, _load, _scenario,
)
from validation.gazebo_experiment import sha256

SCHEMA_VERSION = 1
ANALYZER = Path(__file__).resolve()
DEFAULT_OUTPUT = ROOT / "results" / "tracking_uncertainty_quality"
TOLERANCES_M = (2.0, 5.0, 10.0)
CHI3 = float(chi2.ppf(0.95, 3))
BEFORE_FIT = "computational_budget_exceeded_before_fit"
MAX_FITS_PER_GENERATION = 128
SOURCE_MANIFEST_SHA256 = "62b3105697e6e95acc2d70b57572a10b31efd9e8646c4506edf105aed1d4f141"
SOURCE_SUMMARY_SHA256 = "b201aaccdbdc7b6c5faf1f8a5ec373dc3e8523644d6c1256db717833eda7907b"


@dataclass(frozen=True)
class PrecisionDiagnostic:
    status: str
    tolerance_m: float
    maximum_nominal_95_axis_m: float | None


def _covariance(matrix: Any, label: str) -> np.ndarray:
    p = np.asarray(matrix, dtype=float)
    if p.shape != (3, 3) or not np.all(np.isfinite(p)):
        raise ValueError(f"{label} must be finite 3x3")
    scale = max(1.0, float(np.max(np.abs(p))))
    if np.max(np.abs(p - p.T)) > 1e-11 * scale:
        raise ValueError(f"{label} must be symmetric")
    try:
        np.linalg.cholesky(p)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{label} must be positive definite") from error
    return p


def nominal_coordinate_precision(
    state_enu_m: Any | None, status: str,
    position_covariance_m2: Any | None, tolerance_m: float = 5.0,
) -> PrecisionDiagnostic:
    """Classify published nominal precision using only state/status/Pqq.

    The 95% ellipsoid's *largest semiaxis* is compared with L. This is a
    model-based precision diagnostic, never a guarantee on actual error.
    """
    limit = float(tolerance_m)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("tolerance_m must be finite and positive")
    if status != "confirmed" or state_enu_m is None:
        return PrecisionDiagnostic("unavailable", limit, None)
    q = np.asarray(state_enu_m, dtype=float)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("published position must be finite ENU 3-vector")
    p = _covariance(position_covariance_m2, "position covariance in m2")
    maximum = float(np.sqrt(CHI3 * np.linalg.eigvalsh(p)[-1]))
    label = "nominal_precision_within_target" if maximum <= limit else "nominal_precision_insufficient"
    return PrecisionDiagnostic(label, limit, maximum)


def _float(value: str) -> float | None:
    if value == "" or value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"nonfinite stored diagnostic: {value}")
    return number


def _bool(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"invalid stored boolean: {value}")
    return value == "True"


def _matrix(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=float)


def _close(actual: float, expected: float, label: str, *, atol: float = 1e-7) -> None:
    if not np.isclose(actual, expected, rtol=1e-8, atol=atol):
        raise ValueError(f"{label} differs: stored={actual}, independent={expected}")


def _verify_axes(p: np.ndarray, axes: Any, directions: Any, label: str) -> np.ndarray:
    a = np.asarray(axes, float)
    v = np.asarray(directions, float)
    if a.shape != (3,) or v.shape != (3, 3) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(v)):
        raise ValueError(f"{label} axes/directions must be finite 3 and 3x3")
    if np.any(a <= 0) or np.any(np.diff(a) < -1e-9):
        raise ValueError(f"{label} semiaxes must be positive and sorted")
    np.testing.assert_allclose(v.T @ v, np.eye(3), rtol=0, atol=1e-8,
                               err_msg=f"{label} directions not orthonormal")
    eigenvalues = np.linalg.eigvalsh(p)
    np.testing.assert_allclose(a * a / CHI3, eigenvalues, rtol=1e-8, atol=1e-9,
                               err_msg=f"{label} semiaxes disagree with P")
    np.testing.assert_allclose(p @ v, v * eigenvalues[None, :], rtol=1e-8, atol=1e-8,
                               err_msg=f"{label} directions disagree with P")
    return a


def _optional_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = np.asarray(list(values), dtype=float)
    if not len(numbers):
        return {"n": 0, "median": None, "p95": None, "mean": None}
    return {"n": len(numbers), "median": float(np.median(numbers)),
            "p95": float(np.percentile(numbers, 95)), "mean": float(np.mean(numbers))}


def summarize_pair(
    baseline: list[dict[str, Any]], improved: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare own valid epochs and their intersection, retaining zero denominators."""
    left = {float(x["processing_time_s"]): x for x in baseline}
    right = {float(x["processing_time_s"]): x for x in improved}
    if len(left) != len(baseline) or len(right) != len(improved) or set(left) != set(right):
        raise ValueError("paired publication schedules differ or have duplicate times")
    valid_left = {t for t, row in left.items() if row["valid"]}
    valid_right = {t for t, row in right.items() if row["valid"]}
    shared = valid_left & valid_right
    result: dict[str, Any] = {
        "publication_count": len(left), "baseline_valid_count": len(valid_left),
        "new_valid_count": len(valid_right), "shared_valid_count": len(shared),
        "baseline_only_count": len(valid_left - shared), "new_only_count": len(valid_right - shared),
    }
    for prefix, rows, own in (("baseline", left, valid_left), ("new", right, valid_right)):
        for basis, times in (("own", own), ("shared", shared)):
            selected = [rows[t] for t in sorted(times)]
            result[f"{prefix}_{basis}_denominator"] = len(selected)
            result[f"{prefix}_{basis}_coverage"] = (
                sum(x["position_covered"] for x in selected) / len(selected) if selected else None)
            for metric in ("position_error_m", "velocity_error_mps", "position_nees",
                           "position_maximum_axis_m", "velocity_sigma_rss_mps",
                           "time_since_last_accepted_update_s"):
                values = [x[metric] for x in selected if x[metric] is not None]
                result[f"{prefix}_{basis}_{metric}_median"] = (
                    float(np.median(values)) if values else None)
        result[f"{prefix}_accepted_updates_at_shared_epochs"] = sum(
            rows[t]["accepted_update_count"] for t in shared)
        result[f"{prefix}_rejected_updates_at_shared_epochs"] = sum(
            rows[t]["rejected_update_count"] for t in shared)
    return result


def count_batch_fits(rows: list[dict[str, str]], final_generation_count: int,
                     maximum_per_generation: int = MAX_FITS_PER_GENERATION) -> dict[str, Any]:
    """Count launched optimizations across resets, excluding before-fit denials."""
    by_generation: Counter[int] = Counter()
    rejected_before_fit = 0
    for row in rows:
        generation = int(row["generation"])
        if generation < 1:
            raise ValueError("batch-fit generation must be positive")
        if row["reason"] == BEFORE_FIT:
            rejected_before_fit += 1
            if int(row["fit_count_after_attempt"]) != by_generation[generation]:
                raise ValueError("before-fit denial changed generation fit count")
        else:
            by_generation[generation] += 1
            if int(row["fit_count_after_attempt"]) != by_generation[generation]:
                raise ValueError("generation fit count disagrees with batch journal")
            if by_generation[generation] > maximum_per_generation:
                raise ValueError("per-generation optimization budget exceeded")
    last_generation = max(by_generation, default=0)
    if rows and int(rows[-1]["fit_count_after_attempt"]) != final_generation_count:
        raise ValueError("final generation count disagrees with last batch diagnostic")
    if not rows and final_generation_count:
        raise ValueError("nonzero final count without batch diagnostics")
    return {
        "total_executed_optimizations": sum(by_generation.values()),
        "final_generation_optimizations": final_generation_count,
        "maximum_single_generation_optimizations": max(by_generation.values(), default=0),
        "attempts_rejected_before_optimization": rejected_before_fit,
        "generation_count_with_fits": len(by_generation),
        "per_generation_optimizations_json": json.dumps(dict(sorted(by_generation.items())), sort_keys=True),
        "last_fit_generation": last_generation,
    }


def _source_snapshot() -> tuple[dict[str, Any], dict[str, str]]:
    manifest = _load(SOURCE)
    if sha256(SOURCE / "evaluation_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise ValueError("published evaluation manifest SHA changed")
    if sha256(SOURCE / "evaluation_summary.json") != SOURCE_SUMMARY_SHA256:
        raise ValueError("published evaluation summary SHA changed")
    digest = {
        "evaluation_manifest.json": SOURCE_MANIFEST_SHA256,
        "evaluation_summary.json": SOURCE_SUMMARY_SHA256,
    }
    summary = json.loads((SOURCE / "evaluation_summary.json").read_text(encoding="utf-8"))
    for name, expected in summary["tables"].items():
        actual = sha256(SOURCE / name)
        if actual != expected:
            raise ValueError(f"published summary table SHA changed: {name}")
        digest[name] = expected
    for spec in manifest["specs"]:
        directory = _directory(SOURCE, spec)
        relative = directory.relative_to(SOURCE).as_posix()
        experiment_path = directory / "experiment.json"
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        if experiment["run_id"] != spec["run_id"] or experiment["status"] != "complete":
            raise ValueError(f"source experiment identity/status changed: {relative}")
        digest[f"{relative}/experiment.json"] = sha256(experiment_path)
        for name, expected in experiment["result_sha256"].items():
            actual = sha256(directory / name)
            if actual != expected:
                raise ValueError(f"source artifact SHA changed: {relative}/{name}")
            digest[f"{relative}/{name}"] = expected
    return manifest, digest


def _audit_epoch(
    raw: dict[str, str], trajectory: Any, init_by_generation: dict[int, float],
    first_valid_time: float | None, accepted_count: int, rejected_count: int,
    last_accepted_time: float | None,
) -> dict[str, Any]:
    time_s = float(raw["processing_time_s"])
    valid = _bool(raw["valid"])
    confirmed = _bool(raw["confirmed"])
    status = raw["status"]
    if valid and (not confirmed or status != "confirmed"):
        raise ValueError("valid publication without confirmed state")
    generation, reset_count = int(raw["generation"]), int(raw["reset_count"])
    if reset_count < 0 or generation < 0:
        raise ValueError("invalid generation/reset count")
    stored_last = _float(raw["last_accepted_update_time_s"])
    stored_age = _float(raw["time_since_last_accepted_update_s"])
    if stored_last is None or last_accepted_time is None:
        if stored_last is not None or last_accepted_time is not None or stored_age is not None:
            raise ValueError("last accepted update is inconsistent")
    else:
        _close(stored_last, last_accepted_time, "last accepted update time", atol=1e-9)
        if stored_age is None:
            raise ValueError("time since accepted update missing")
        _close(stored_age, time_s - last_accepted_time, "time since accepted update", atol=1e-9)
    base: dict[str, Any] = {
        "processing_time_s": time_s, "confirmed": confirmed, "valid": valid,
        "status": status, "generation": generation, "reset_count": reset_count,
        "accepted_update_count": accepted_count, "rejected_update_count": rejected_count,
        "time_since_last_accepted_update_s": stored_age,
        "position_error_m": None, "velocity_error_mps": None,
        "position_maximum_axis_m": None, "position_sigma_rss_m": None,
        "velocity_maximum_axis_mps": None, "velocity_sigma_rss_mps": None,
        "position_nees": None, "velocity_nees": None,
        "position_covered": None, "velocity_covered": None,
        "track_age_s": None,
    }
    if not valid:
        if any(raw[key] for key in ("position_enu_m_json", "velocity_enu_mps_json",
                                    "position_covariance_m2_json", "velocity_covariance_m2ps2_json")):
            raise ValueError("invalid publication contains state or covariance")
        base["phase"] = "unavailable_after_reset" if reset_count else "unavailable_before_or_between_tracks"
        for limit in TOLERANCES_M:
            base[f"precision_{int(limit)}m"] = nominal_coordinate_precision(None, status, None, limit).status
        return base
    if generation not in init_by_generation:
        raise ValueError("valid generation lacks lifecycle initialization")
    track_age = time_s - init_by_generation[generation]
    if track_age < -1e-8:
        raise ValueError("publication predates its generation initialization")
    base["track_age_s"] = max(0.0, track_age)
    base["phase"] = ("first_confirmation" if first_valid_time is None else
                     "after_reset" if reset_count else "subsequent_tracking")
    q = _matrix(raw["position_enu_m_json"])
    v = _matrix(raw["velocity_enu_mps_json"])
    if q.shape != (3,) or v.shape != (3,) or not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
        raise ValueError("published ENU state must contain finite position and velocity")
    p_q = _covariance(_matrix(raw["position_covariance_m2_json"]), "Pqq")
    p_v = _covariance(_matrix(raw["velocity_covariance_m2ps2_json"]), "Pvv")
    q_axes = _verify_axes(p_q, json.loads(raw["position_nominal_95_axes_m_json"]),
                          json.loads(raw["position_nominal_95_axis_directions_json"]), "position")
    v_axes = _verify_axes(p_v, json.loads(raw["velocity_nominal_95_axes_mps_json"]),
                          json.loads(raw["velocity_nominal_95_axis_directions_json"]), "velocity")
    truth_q, truth_v = np.asarray(trajectory.q(time_s), float), np.asarray(trajectory.v(time_s), float)
    if truth_q.shape != (3,) or truth_v.shape != (3,):
        raise ValueError("trajectory truth is not an ENU 3-vector")
    eq, ev = q - truth_q, v - truth_v
    q_error, v_error = float(np.linalg.norm(eq)), float(np.linalg.norm(ev))
    q_nees = float(eq @ np.linalg.solve(p_q, eq))
    v_nees = float(ev @ np.linalg.solve(p_v, ev))
    q_covered = q_nees <= CHI3
    _close(float(raw["position_error_m"]), q_error, "position error")
    _close(float(raw["velocity_error_mps"]), v_error, "velocity error")
    _close(float(raw["position_nees"]), q_nees, "position NEES", atol=1e-8)
    if _bool(raw["position_nominal_95_covered"]) != q_covered:
        raise ValueError("position ellipsoid coverage flag disagrees with NEES")
    _close(float(raw["position_sigma_rss_m"]), np.sqrt(np.trace(p_q)), "position sigma RSS")
    _close(float(raw["velocity_sigma_rss_mps"]), np.sqrt(np.trace(p_v)), "velocity sigma RSS")
    # The stored state NEES needs P_qv and cannot be independently reconstructed.
    base.update({
        "position_error_m": q_error, "velocity_error_mps": v_error,
        "position_maximum_axis_m": float(q_axes[-1]),
        "position_sigma_rss_m": float(np.sqrt(np.trace(p_q))),
        "velocity_maximum_axis_mps": float(v_axes[-1]),
        "velocity_sigma_rss_mps": float(np.sqrt(np.trace(p_v))),
        "position_nees": q_nees, "velocity_nees": v_nees,
        "position_covered": q_covered, "velocity_covered": v_nees <= CHI3,
    })
    for limit in TOLERANCES_M:
        diagnostic = nominal_coordinate_precision(q, status, p_q, limit)
        _close(diagnostic.maximum_nominal_95_axis_m, float(q_axes[-1]), "maximum semiaxis")
        base[f"precision_{int(limit)}m"] = diagnostic.status
    return base


def _audit_track(directory: Path, method: str, variant: str, trajectory: Any,
                 spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = f"{method}_{variant}.csv.gz"
    tracks = _read_csv_gz(directory / f"tracking_{suffix}")
    updates = _read_csv_gz(directory / f"updates_{suffix}")
    fits = _read_csv_gz(directory / f"batch_fits_{suffix}")
    lifecycle = _read_csv_gz(directory / f"lifecycle_{suffix}")
    if not tracks:
        raise ValueError("empty tracking log")
    epochs = [float(x["processing_time_s"]) for x in tracks]
    if not np.all(np.diff(epochs) > 0):
        raise ValueError("publication times not strictly increasing")
    init_by_generation = {int(row["generation"]): float(row["processing_time_s"])
                          for row in lifecycle if row["action"] in ("initialized", "reinitialized")}
    if len(init_by_generation) != sum(row["action"] in ("initialized", "reinitialized") for row in lifecycle):
        raise ValueError("duplicate generation initialization")
    update_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    finite_innovations = []
    for update in updates:
        update_map[update["processing_time_s"]].append(update)
        nis = float(update["pre_update_nis"])
        if np.isfinite(nis):
            residual = _matrix(update["residual_tangent_rad_json"])
            innovation = _matrix(update["innovation_covariance_tangent_rad2_json"])
            if residual.shape != (2,) or innovation.shape != (2, 2) or \
               not np.all(np.isfinite(residual)) or not np.all(np.isfinite(innovation)):
                raise ValueError("finite NIS lacks finite 2D angular residual/innovation S")
            np.linalg.cholesky(0.5 * (innovation + innovation.T))
            _close(nis, float(residual @ np.linalg.solve(innovation, residual)),
                   "angular innovation NIS", atol=1e-9)
            finite_innovations.append((float(np.linalg.norm(residual)), float(np.trace(innovation))))
        elif update["residual_tangent_rad_json"] == "" and not update["residual_unavailable_reason"]:
            raise ValueError("unavailable angular residual lacks explicit reason")
    if set(update_map) - {row["processing_time_s"] for row in tracks}:
        raise ValueError("update without publication at processing time")
    derived: list[dict[str, Any]] = []
    first_valid_time: float | None = None
    last_accepted_time: float | None = None
    accepted_total = rejected_total = 0
    for raw in tracks:
        at_time = update_map.get(raw["processing_time_s"], ())
        accepted = sum(_bool(u["update_applied"]) for u in at_time)
        rejected = len(at_time) - accepted
        accepted_total += accepted
        rejected_total += rejected
        if accepted:
            last_accepted_time = float(raw["processing_time_s"])
        row = _audit_epoch(raw, trajectory, init_by_generation, first_valid_time,
                           accepted, rejected, last_accepted_time)
        row.update({"run_id": spec["run_id"], "index": spec["index"],
                    "trajectory": spec["trajectory"], "distance_m": spec["distance_m"],
                    "replicate": spec["replicate"], "estimator_variant": method,
                    "confirmation_variant": variant})
        derived.append(row)
        if row["valid"] and first_valid_time is None:
            first_valid_time = row["processing_time_s"]
    cost = count_batch_fits(fits, int(tracks[-1]["batch_optimization_count"]))
    cost.update({"run_id": spec["run_id"], "index": spec["index"],
                 "trajectory": spec["trajectory"], "distance_m": spec["distance_m"],
                 "replicate": spec["replicate"], "estimator_variant": method,
                 "confirmation_variant": variant, "reset_count": int(tracks[-1]["reset_count"]),
                 "publication_count": len(tracks), "accepted_update_count": accepted_total,
                 "rejected_update_count": rejected_total,
                 "budget_per_generation": MAX_FITS_PER_GENERATION,
                 "finite_angular_innovation_nis_count": len(finite_innovations),
                 "median_angular_residual_norm_rad": _optional_stats(
                     value[0] for value in finite_innovations)["median"],
                 "median_innovation_covariance_trace_rad2": _optional_stats(
                     value[1] for value in finite_innovations)["median"]})
    return derived, cost


def _group_summary(rows: list[dict[str, Any]], *, include_unavailable: bool = True) -> dict[str, Any]:
    valid = [row for row in rows if row["valid"]]
    denominator = len(rows) if include_unavailable else len(valid)
    return {
        "publication_count": len(rows), "valid_count": len(valid),
        "availability_fraction": len(valid) / denominator if denominator else None,
        "position_coverage": (sum(row["position_covered"] for row in valid) / len(valid) if valid else None),
        "velocity_coverage": (sum(row["velocity_covered"] for row in valid) / len(valid) if valid else None),
        "position_error_median_m": _optional_stats(row["position_error_m"] for row in valid)["median"],
        "position_error_p95_m": _optional_stats(row["position_error_m"] for row in valid)["p95"],
        "velocity_error_median_mps": _optional_stats(row["velocity_error_mps"] for row in valid)["median"],
        "position_max_axis_median_m": _optional_stats(row["position_maximum_axis_m"] for row in valid)["median"],
        "velocity_max_axis_median_mps": _optional_stats(row["velocity_maximum_axis_mps"] for row in valid)["median"],
        "position_nees_median": _optional_stats(row["position_nees"] for row in valid)["median"],
        "velocity_nees_median": _optional_stats(row["velocity_nees"] for row in valid)["median"],
        "time_since_last_update_median_s": _optional_stats(
            row["time_since_last_accepted_update_s"] for row in valid
            if row["time_since_last_accepted_update_s"] is not None)["median"],
        "accepted_update_count": sum(row["accepted_update_count"] for row in rows),
        "rejected_update_count": sum(row["rejected_update_count"] for row in rows),
    }


def _quality_summary(rows: list[dict[str, Any]], tolerance_m: float) -> dict[str, Any]:
    label = f"precision_{int(tolerance_m)}m"
    statuses = Counter(row[label] for row in rows)
    if sum(statuses.values()) != len(rows):
        raise AssertionError("precision status denominator mismatch")
    sufficient = [row for row in rows if row[label] == "nominal_precision_within_target"]
    false_precision = [row for row in sufficient if row["position_error_m"] > tolerance_m]
    errors = _optional_stats(row["position_error_m"] for row in sufficient)
    return {
        "tolerance_m": tolerance_m, "all_publications": len(rows),
        "unavailable_count": statuses["unavailable"],
        "nominal_precision_insufficient_count": statuses["nominal_precision_insufficient"],
        "nominal_precision_within_target_count": len(sufficient),
        "nominal_within_fraction_all": len(sufficient) / len(rows) if rows else None,
        "conditional_error_count": errors["n"],
        "conditional_position_error_median_m": errors["median"],
        "conditional_position_error_p95_m": errors["p95"],
        "false_precision_count": len(false_precision),
        "false_precision_fraction_conditional": (
            len(false_precision) / len(sufficient) if sufficient else None),
    }


def _paired_common_phase(left: list[dict[str, Any]], right: list[dict[str, Any]],
                         phase: str) -> dict[str, Any]:
    left_by_time = {row["processing_time_s"]: row for row in left}
    right_by_time = {row["processing_time_s"]: row for row in right}
    if set(left_by_time) != set(right_by_time):
        raise ValueError("paired schedules differ")
    common = []
    for t in sorted(left_by_time):
        a, b = left_by_time[t], right_by_time[t]
        if not (a["valid"] and b["valid"]):
            continue
        if phase == "later_without_reset" and (a["reset_count"] or b["reset_count"] or
                                                  a["track_age_s"] <= 0 or b["track_age_s"] <= 0):
            continue
        if phase == "after_reset" and not (a["reset_count"] or b["reset_count"]):
            continue
        common.append((a, b))
    result: dict[str, Any] = {"phase": phase, "shared_valid_count": len(common)}
    for label, index in (("baseline", 0), ("new", 1)):
        selected = [pair[index] for pair in common]
        metrics = _group_summary(selected)
        for key, value in metrics.items():
            result[f"{label}_{key}"] = value
        result[f"{label}_track_age_median_s"] = _optional_stats(
            row["track_age_s"] for row in selected)["median"]
    return result


def _age_band(age_s: float) -> str:
    return "0_to_1s" if age_s < 1.0 else "1_to_3s" if age_s < 3.0 else "3s_plus"



def _coverage_transition_rows(
    directory: Path, method: str, baseline: list[dict[str, Any]],
    improved: list[dict[str, Any]], trajectory: Any, spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """Algebraic common-epoch comparison; never feeds truth to a tracker."""
    suffixes = (f"tracking_{method}_{variant}.csv.gz" for variant in VARIANTS)
    raw_left, raw_right = (_read_csv_gz(directory / name) for name in suffixes)
    if len(raw_left) != len(raw_right) or len(raw_left) != len(baseline):
        raise ValueError("paired tracking logs differ in length")
    result = []
    for a, b, da, db in zip(raw_left, raw_right, baseline, improved, strict=True):
        if a["processing_time_s"] != b["processing_time_s"] or \
           float(a["processing_time_s"]) != da["processing_time_s"] or \
           float(b["processing_time_s"]) != db["processing_time_s"]:
            raise ValueError("paired epoch mismatch")
        if not (da["valid"] and db["valid"]):
            continue
        t = da["processing_time_s"]
        truth = np.asarray(trajectory.q(t), float)
        e_a = _matrix(a["position_enu_m_json"]) - truth
        e_b = _matrix(b["position_enu_m_json"]) - truth
        p_a = _covariance(_matrix(a["position_covariance_m2_json"]), "baseline Pqq")
        p_b = _covariance(_matrix(b["position_covariance_m2_json"]), "new Pqq")
        fixed_error_new_p_nees = float(e_a @ np.linalg.solve(p_b, e_a))
        n_a, n_b = da["position_nees"], db["position_nees"]
        phase = ("after_reset" if da["reset_count"] or db["reset_count"] else
                 "later_without_reset" if da["track_age_s"] > 0 and db["track_age_s"] > 0 else
                 "confirmation_epoch")
        result.append({
            "run_id": spec["run_id"], "index": spec["index"],
            "trajectory": spec["trajectory"], "distance_m": spec["distance_m"],
            "replicate": spec["replicate"], "estimator_variant": method,
            "processing_time_s": t, "phase": phase,
            "baseline_covered": da["position_covered"], "new_covered": db["position_covered"],
            "baseline_position_error_m": da["position_error_m"],
            "new_position_error_m": db["position_error_m"],
            "baseline_velocity_error_mps": da["velocity_error_mps"],
            "new_velocity_error_mps": db["velocity_error_mps"],
            "baseline_max_axis_m": da["position_maximum_axis_m"],
            "new_max_axis_m": db["position_maximum_axis_m"],
            "baseline_position_nees": n_a, "new_position_nees": n_b,
            "baseline_error_under_new_covariance_nees": fixed_error_new_p_nees,
            "baseline_track_age_s": da["track_age_s"],
            "new_track_age_s": db["track_age_s"],
            "baseline_accepted_updates": da["accepted_update_count"],
            "new_accepted_updates": db["accepted_update_count"],
            "baseline_rejected_updates": da["rejected_update_count"],
            "new_rejected_updates": db["rejected_update_count"],
            "baseline_time_since_last_update_s": da["time_since_last_accepted_update_s"],
            "new_time_since_last_update_s": db["time_since_last_accepted_update_s"],
        })
    return result


def _transition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lost = [row for row in rows if row["baseline_covered"] and not row["new_covered"]]
    gained = [row for row in rows if not row["baseline_covered"] and row["new_covered"]]
    both = [row for row in rows if row["baseline_covered"] and row["new_covered"]]
    neither = [row for row in rows if not row["baseline_covered"] and not row["new_covered"]]
    return {
        "shared_valid_count": len(rows), "covered_both_count": len(both),
        "coverage_lost_count": len(lost), "coverage_gained_count": len(gained),
        "covered_neither_count": len(neither),
        "lost_with_larger_position_error_count": sum(
            row["new_position_error_m"] > row["baseline_position_error_m"] for row in lost),
        "lost_with_smaller_max_axis_count": sum(
            row["new_max_axis_m"] < row["baseline_max_axis_m"] for row in lost),
        "lost_where_new_covariance_alone_would_exclude_baseline_error_count": sum(
            row["baseline_error_under_new_covariance_nees"] > CHI3 for row in lost),
        "lost_where_changed_error_needed_to_exceed_ellipsoid_count": sum(
            row["baseline_error_under_new_covariance_nees"] <= CHI3 for row in lost),
        "lost_baseline_error_median_m": _optional_stats(
            row["baseline_position_error_m"] for row in lost)["median"],
        "lost_new_error_median_m": _optional_stats(
            row["new_position_error_m"] for row in lost)["median"],
        "lost_baseline_max_axis_median_m": _optional_stats(
            row["baseline_max_axis_m"] for row in lost)["median"],
        "lost_new_max_axis_median_m": _optional_stats(
            row["new_max_axis_m"] for row in lost)["median"],
        "lost_baseline_nees_median": _optional_stats(
            row["baseline_position_nees"] for row in lost)["median"],
        "lost_new_nees_median": _optional_stats(
            row["new_position_nees"] for row in lost)["median"],
        "lost_baseline_track_age_median_s": _optional_stats(
            row["baseline_track_age_s"] for row in lost)["median"],
        "lost_new_track_age_median_s": _optional_stats(
            row["new_track_age_s"] for row in lost)["median"],
        "lost_baseline_accepted_updates": sum(row["baseline_accepted_updates"] for row in lost),
        "lost_new_accepted_updates": sum(row["new_accepted_updates"] for row in lost),
        "lost_baseline_rejected_updates": sum(row["baseline_rejected_updates"] for row in lost),
        "lost_new_rejected_updates": sum(row["new_rejected_updates"] for row in lost),
        "lost_baseline_time_since_update_median_s": _optional_stats(
            row["baseline_time_since_last_update_s"] for row in lost
            if row["baseline_time_since_last_update_s"] is not None)["median"],
        "lost_new_time_since_update_median_s": _optional_stats(
            row["new_time_since_last_update_s"] for row in lost
            if row["new_time_since_last_update_s"] is not None)["median"],
    }


def analyze(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Audit all 48 archived tracks and publish SHA-checked derivatives once."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    manifest, source_hashes = _source_snapshot()
    epoch_rows: list[dict[str, Any]] = []
    costs: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    paired_phases: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    run_summary = {(int(row["index"]), row["estimator_variant"], row["confirmation_variant"]): row
                   for row in csv.DictReader((SOURCE / "run_summary.csv").open(newline="", encoding="utf-8"))}
    if len(run_summary) != 48:
        raise ValueError("published summary must contain 48 distinct method/variant results")
    for spec in manifest["specs"]:
        _, trajectory, _, _ = _scenario(manifest, spec)
        directory = _directory(SOURCE, spec)
        for method in METHODS:
            paired: dict[str, list[dict[str, Any]]] = {}
            for variant in VARIANTS:
                rows, cost = _audit_track(directory, method, variant, trajectory, spec)
                key = int(spec["index"]), method, variant
                published = run_summary[key]
                if cost["final_generation_optimizations"] != int(published["batch_optimization_count"]):
                    raise ValueError("published generation count changed")
                if cost["accepted_update_count"] != int(published["accepted_update_count"]) or \
                   cost["rejected_update_count"] != int(published["rejected_update_count"]):
                    raise ValueError("published update counts changed")
                if sum(row["valid"] for row in rows) != int(published["valid_publication_count"]):
                    raise ValueError("published valid count changed")
                epoch_rows.extend(rows)
                costs.append(cost)
                paired[variant] = rows
                first = next((row for row in rows if row["valid"]), None)
                first_rows.append({"run_id": spec["run_id"], "index": spec["index"],
                                   "trajectory": spec["trajectory"], "distance_m": spec["distance_m"],
                                   "replicate": spec["replicate"], "estimator_variant": method,
                                   "confirmation_variant": variant,
                                   "ever_confirmed": first is not None,
                                   "first_time_s": first["processing_time_s"] if first else None,
                                   "first_position_error_m": first["position_error_m"] if first else None,
                                   "first_velocity_error_mps": first["velocity_error_mps"] if first else None,
                                   "first_position_max_axis_m": first["position_maximum_axis_m"] if first else None,
                                   "first_velocity_max_axis_mps": first["velocity_maximum_axis_mps"] if first else None,
                                   "first_position_nees": first["position_nees"] if first else None,
                                   "first_position_covered": first["position_covered"] if first else None,
                                   "first_reset_count": first["reset_count"] if first else None})
            transition_rows.extend(_coverage_transition_rows(
                directory, method, paired["baseline"],
                paired["three_station_confirmation"], trajectory, spec))
            comparison = summarize_pair(paired["baseline"], paired["three_station_confirmation"])
            paired_rows.append({"run_id": spec["run_id"], "index": spec["index"],
                                "trajectory": spec["trajectory"], "distance_m": spec["distance_m"],
                                "replicate": spec["replicate"], "estimator_variant": method, **comparison})
            for phase in ("all_shared", "later_without_reset", "after_reset"):
                paired_phases.append({"run_id": spec["run_id"], "index": spec["index"],
                                      "estimator_variant": method,
                                      **_paired_common_phase(paired["baseline"],
                                                             paired["three_station_confirmation"], phase)})
    totals = {(method, variant): sum(row["total_executed_optimizations"] for row in costs
                                    if row["estimator_variant"] == method and row["confirmation_variant"] == variant)
              for method in METHODS for variant in VARIANTS}
    expected = {(METHODS[0], VARIANTS[0]): 633, (METHODS[0], VARIANTS[1]): 690,
                (METHODS[1], VARIANTS[0]): 379, (METHODS[1], VARIANTS[1]): 467}
    if totals != expected:
        raise ValueError(f"executed fit totals disagree with prespecified audit: {totals}")
    method_cost = []
    method_comparison = []
    phase_summary = []
    age_summary = []
    quality = []
    transition_summary = []
    transition_by_stream = []
    for spec in manifest["specs"]:
        for method in METHODS:
            selected = [row for row in transition_rows if row["run_id"] == spec["run_id"]
                        and row["estimator_variant"] == method]
            transition_by_stream.append({"run_id": spec["run_id"], "index": spec["index"],
                                         "trajectory": spec["trajectory"],
                                         "distance_m": spec["distance_m"],
                                         "replicate": spec["replicate"],
                                         "estimator_variant": method,
                                         **_transition_summary(selected)})
    for method in METHODS:
        method_transitions = [row for row in transition_rows if row["estimator_variant"] == method]
        for phase in ("all_shared", "confirmation_epoch", "later_without_reset", "after_reset"):
            selected = (method_transitions if phase == "all_shared" else
                        [row for row in method_transitions if row["phase"] == phase])
            transition_summary.append({"estimator_variant": method, "phase": phase,
                                       **_transition_summary(selected)})
        common_times = {(row["run_id"], float(row["processing_time_s"])) for row in epoch_rows
                        if row["estimator_variant"] == method and row["confirmation_variant"] == VARIANTS[0]
                        and row["valid"]}
        other_times = {(row["run_id"], float(row["processing_time_s"])) for row in epoch_rows
                       if row["estimator_variant"] == method and row["confirmation_variant"] == VARIANTS[1]
                       and row["valid"]}
        shared_times = common_times & other_times
        for variant in VARIANTS:
            subset = [row for row in epoch_rows if row["estimator_variant"] == method
                      and row["confirmation_variant"] == variant]
            cost_subset = [row for row in costs if row["estimator_variant"] == method
                           and row["confirmation_variant"] == variant]
            method_cost.append({"estimator_variant": method, "confirmation_variant": variant,
                                "stream_count": len(cost_subset),
                                "total_executed_optimizations": totals[(method, variant)],
                                "sum_final_generation_optimizations": sum(
                                    row["final_generation_optimizations"] for row in cost_subset),
                                "maximum_single_generation_optimizations": max(
                                    row["maximum_single_generation_optimizations"] for row in cost_subset),
                                "attempts_rejected_before_optimization": sum(
                                    row["attempts_rejected_before_optimization"] for row in cost_subset),
                                "streams_with_reset": sum(row["reset_count"] > 0 for row in cost_subset)})
            for basis, selected in (("own_valid", subset),
                                    ("shared_valid", [row for row in subset
                                                      if (row["run_id"], row["processing_time_s"]) in shared_times])):
                method_comparison.append({"estimator_variant": method, "confirmation_variant": variant,
                                          "basis": basis, **_group_summary(selected)})
            for phase in ("first_confirmation", "subsequent_tracking", "after_reset",
                          "unavailable_before_or_between_tracks", "unavailable_after_reset"):
                selected = [row for row in subset if row["phase"] == phase]
                phase_summary.append({"estimator_variant": method, "confirmation_variant": variant,
                                      "phase": phase, **_group_summary(selected)})
            for reset_group in ("before_reset", "after_reset"):
                for band in ("0_to_1s", "1_to_3s", "3s_plus"):
                    selected = [row for row in subset if row["valid"]
                                and ("after_reset" if row["reset_count"] else "before_reset") == reset_group
                                and _age_band(row["track_age_s"]) == band]
                    age_summary.append({"estimator_variant": method, "confirmation_variant": variant,
                                        "reset_group": reset_group, "track_age_band": band,
                                        **_group_summary(selected)})
            for distance in ("all", 200, 700, 1000):
                selected = subset if distance == "all" else [row for row in subset if row["distance_m"] == distance]
                for limit in TOLERANCES_M:
                    quality.append({"estimator_variant": method, "confirmation_variant": variant,
                                    "distance_m": distance, **_quality_summary(selected, limit)})
    if len(epoch_rows) != sum(row["publication_count"] for row in costs):
        raise AssertionError("epoch denominator mismatch")
    if len(costs) != 48 or len(paired_rows) != 24 or len(first_rows) != 48:
        raise AssertionError("expected 48 tracks and 24 paired streams")
    tables = {
        "epoch_diagnostics.csv.gz": epoch_rows,
        "cost_by_stream.csv": costs,
        "cost_by_method.csv": method_cost,
        "first_confirmation.csv": first_rows,
        "paired_stream.csv": paired_rows,
        "paired_phase.csv": paired_phases,
        "method_comparison.csv": method_comparison,
        "phase_summary.csv": phase_summary,
        "age_summary.csv": age_summary,
        "nominal_quality.csv": quality,
        "coverage_transition_epoch.csv.gz": transition_rows,
        "coverage_transition_summary.csv": transition_summary,
        "coverage_transition_by_stream.csv": transition_by_stream,
    }
    output.mkdir(parents=True, exist_ok=False)
    derived_hashes = {}
    for name, rows in tables.items():
        if not rows:
            raise AssertionError(f"empty derived table: {name}")
        path = output / name
        columns = list(rows[0])
        if name.endswith(".gz"):
            _write_csv_gz(path, rows, columns)
        else:
            _write_csv(path, rows, columns)
        derived_hashes[name] = sha256(path)
    provenance = {
        "schema_version": SCHEMA_VERSION, "analyzer_sha256": sha256(ANALYZER),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_summary_sha256": SOURCE_SUMMARY_SHA256,
        "source_files_sha256": source_hashes,
        "derived_sha256": derived_hashes,
        "units_and_frame": "ENU East-North-Up, m, s, m/s, covariance m2 and m2/s2",
        "uncertainty_contract": "posterior Pqq/Pvv; nominal chi-square(3) ellipsoid; truth only in offline audit",
        "state_nees_independent_check": "unavailable: saved Pqq/Pvv omit cross block Pqv of full 6x6 covariance",
        "audio_synthesis_count": 0, "tracker_replay_count": 0,
        "source_stream_count": 12, "source_tracker_result_count": len(costs),
        "valid_publication_count": sum(row["valid"] for row in epoch_rows),
        "all_publication_count": len(epoch_rows),
        "shared_valid_publication_pair_count": len(transition_rows),
        "per_method_executed_fit_totals": {
            f"{method}:{variant}": value for (method, variant), value in totals.items()},
    }
    _write_json(output / "analysis_manifest.json", provenance)
    return {key: value for key, value in provenance.items() if key != "source_files_sha256"}


def verify(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output)
    stored = json.loads((output / "analysis_manifest.json").read_text(encoding="utf-8"))
    if stored["schema_version"] != SCHEMA_VERSION or stored["analyzer_sha256"] != sha256(ANALYZER):
        raise ValueError("analysis schema or code SHA mismatch")
    _, source = _source_snapshot()
    if stored["source_files_sha256"] != source:
        raise ValueError("source artifact provenance mismatch")
    for name, expected in stored["derived_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"derived SHA mismatch: {name}")
    return {"status": "verified", "source_stream_count": stored["source_stream_count"],
            "source_tracker_result_count": stored["source_tracker_result_count"],
            "audio_synthesis_count": stored["audio_synthesis_count"],
            "tracker_replay_count": stored["tracker_replay_count"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.output) if args.action == "analyze" else verify(args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
