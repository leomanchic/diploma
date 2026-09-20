"""Recreate only S8 calibration bearings and audit transfer by source session.

The acoustic generator is called solely for manifest-declared calibration
sessions. Evaluation audio, tracker updates, and earlier Monte Carlo studies
are never invoked. Truth-derived tangent residuals are offline artifacts only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from validation.independent_recordings_pilot import (
    DURATION_S, SNR_LEVELS_DB, SOURCE_MODELS, generate_paired_sequences,
    recording_ids_for_split,
)
from validation.s8_calibration_transfer_analysis import ROOT, _read, _write
from validation.three_station_audio_tracking_study import calibrate_audio_bearings


RESULT_PREFIX = "s8_calibration_transfer_"
SESSION_FIELDS = (
    "source_model_comparison", "paired_session_id", "snr_db", "station_id",
    "estimator_variant",
)


def _valid_vectors(rows: list[dict[str, object]]) -> np.ndarray:
    result = np.asarray([
        [float(row["residual_rad_0"]), float(row["residual_rad_1"])]
        for row in rows if row["valid"] is True or str(row["valid"]).lower() == "true"
    ])
    if result.ndim != 2 or result.shape[1] != 2 or len(result) < 3:
        raise ValueError("at least three valid two-dimensional residuals are required")
    if not np.all(np.isfinite(result)):
        raise ValueError("calibration residuals must be finite")
    return result


def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    covariance = np.cov(values.T)
    lengths = np.linalg.norm(values, axis=1)
    return mean, covariance, lengths


def compact_calibration_records(
    rows: list[dict[str, object]], *, source_seed: int,
    noise_seeds: dict[str, int], sequence_seed: int,
) -> list[dict[str, object]]:
    """Persist offline residual evidence with explicit source-session grain."""

    result = []
    for row in rows:
        valid = bool(row["valid"])
        residual = np.asarray(row["residual_rad"], dtype=float)
        result.append({
            "split": "calibration", "source_model_comparison": row["source_model_comparison"],
            "paired_recording_id": row["paired_recording_id"],
            "paired_session_id": row["paired_session_id"],
            "paired_origin_asset_id": row["paired_origin_asset_id"],
            "sequence_id": row["sequence_id"],
            "configuration_index": int(row["configuration_index"]),
            "snr_db": float(row["snr_db"]),
            "station_id": row["station_id"],
            "estimator_variant": row["estimator_variant"],
            "frame_index": int(row["frame_index"]),
            "frame_center_reception_time_s": float(row["frame_center_reception_time_s"]),
            "valid": valid, "invalid_reason": row["invalid_reason"] or "",
            "residual_rad_0": float(residual[0]),
            "residual_rad_1": float(residual[1]),
            "geodesic_error_deg": float(row["geodesic_error_deg"]),
            "tangent_frame": "prediction", "residual_units": "rad_angular_arc",
            "truth_used_by_online_estimator": False,
            "residual_truth_is_offline_only": True,
            "dependent_frames_are_independent_trials": False,
            "sequence_seed": int(sequence_seed), "source_seed": int(source_seed),
            "noise_seed": int(noise_seeds[row["station_id"]]),
        })
    return result


def session_snr_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """One diagnostic row per original session/SNR/station/method/source."""

    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in SESSION_FIELDS)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        values = _valid_vectors(group)
        mean, covariance, lengths = _stats(values)
        central = values[lengths <= np.percentile(lengths, 95)]
        central_mean, central_covariance, _ = _stats(central)
        result.append({
            **dict(zip(SESSION_FIELDS, key)),
            "dependent_frame_count": len(group), "valid_frame_count": len(values),
            "mean_az_arc_rad": float(mean[0]), "mean_el_arc_rad": float(mean[1]),
            "bias_norm_deg": float(np.rad2deg(np.linalg.norm(mean))),
            "covariance_00_rad2": float(covariance[0, 0]),
            "covariance_01_rad2": float(covariance[0, 1]),
            "covariance_11_rad2": float(covariance[1, 1]),
            "covariance_trace_rad2": float(np.trace(covariance)),
            "median_error_deg": float(np.rad2deg(np.median(lengths))),
            "p95_error_deg": float(np.rad2deg(np.percentile(lengths, 95))),
            "p99_error_deg": float(np.rad2deg(np.percentile(lengths, 99))),
            "fraction_gt_5deg": float(np.mean(np.rad2deg(lengths) > 5)),
            "fraction_gt_10deg": float(np.mean(np.rad2deg(lengths) > 10)),
            "fraction_gt_30deg": float(np.mean(np.rad2deg(lengths) > 30)),
            "trimmed_95pct_valid_frame_count": len(central),
            "trimmed_95pct_bias_norm_deg": float(np.rad2deg(np.linalg.norm(central_mean))),
            "trimmed_95pct_mean_shift_deg": float(np.rad2deg(np.linalg.norm(mean - central_mean))),
            "trimmed_95pct_covariance_trace_rad2": float(np.trace(central_covariance)),
            "dependent_frames_are_independent_trials": False,
        })
    return result


def pooled_session_attribution(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Exact within/between sample scatter identity plus outlier diagnostics."""

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source_model_comparison"]), str(row["station_id"]),
                str(row["estimator_variant"]))].append(row)
    result = []
    for (source, station, method), group in sorted(groups.items()):
        values = _valid_vectors(group)
        pooled_mean, pooled_covariance, lengths = _stats(values)
        sessions = sorted({str(row["paired_session_id"]) for row in group})
        if len(sessions) != 2:
            raise ValueError("expected exactly two original calibration sessions")
        session_values = {
            session: _valid_vectors([
                row for row in group if row["paired_session_id"] == session
            ]) for session in sessions
        }
        between = np.zeros((2, 2))
        within = np.zeros((2, 2))
        session_within_scatter_trace = {}
        session_means = {}
        for session, subset in session_values.items():
            mean, covariance, _ = _stats(subset)
            between += len(subset) * np.outer(mean - pooled_mean, mean - pooled_mean)
            within += (len(subset) - 1) * covariance
            session_within_scatter_trace[session] = float((len(subset) - 1) * np.trace(covariance))
            session_means[session] = [float(mean[0]), float(mean[1])]
        pooled_scatter = (len(values) - 1) * pooled_covariance
        reconstruction_error = float(np.max(np.abs(pooled_scatter - between - within)))
        central = values[lengths <= np.percentile(lengths, 95)]
        central_mean, central_covariance, _ = _stats(central)
        under_30 = values[np.rad2deg(lengths) <= 30]
        under_30_mean, _, _ = _stats(under_30) if len(under_30) >= 3 else (
            np.full(2, np.nan), np.full((2, 2), np.nan), np.empty(0)
        )
        left_mean = np.mean(session_values[sessions[0]], axis=0)
        right_mean = np.mean(session_values[sessions[1]], axis=0)
        result.append({
            "source_model_comparison": source, "station_id": station,
            "estimator_variant": method,
            "calibration_session_ids_json": json.dumps(sessions),
            "session_valid_counts_json": json.dumps({
                session: len(session_values[session]) for session in sessions
            }, sort_keys=True),
            "session_mean_tangent_rad_json": json.dumps(session_means, sort_keys=True),
            "session_within_scatter_trace_fraction_json": json.dumps({
                session: contribution / float(np.trace(pooled_scatter))
                for session, contribution in session_within_scatter_trace.items()
            }, sort_keys=True),
            "pooled_valid_frame_count": len(values),
            "pooled_bias_az_arc_rad": float(pooled_mean[0]),
            "pooled_bias_el_arc_rad": float(pooled_mean[1]),
            "pooled_bias_norm_deg": float(np.rad2deg(np.linalg.norm(pooled_mean))),
            "pooled_covariance_trace_rad2": float(np.trace(pooled_covariance)),
            "between_session_scatter_trace_fraction": float(
                np.trace(between) / np.trace(pooled_scatter)
            ),
            "within_session_scatter_trace_fraction": float(
                np.trace(within) / np.trace(pooled_scatter)
            ),
            "scatter_reconstruction_max_abs_rad2": reconstruction_error,
            "session_mean_separation_deg": float(np.rad2deg(np.linalg.norm(left_mean - right_mean))),
            "leave_first_session_out_bias_shift_deg": float(
                np.rad2deg(np.linalg.norm(right_mean - pooled_mean))
            ),
            "leave_second_session_out_bias_shift_deg": float(
                np.rad2deg(np.linalg.norm(left_mean - pooled_mean))
            ),
            "top_5pct_removed_count": len(values) - len(central),
            "top_5pct_removed_fraction": (len(values) - len(central)) / len(values),
            "top_5pct_mean_shift_deg": float(np.rad2deg(np.linalg.norm(pooled_mean - central_mean))),
            "pooled_over_top_5pct_trimmed_trace_ratio": float(
                np.trace(pooled_covariance) / np.trace(central_covariance)
            ),
            "fraction_gt_30deg": float(np.mean(np.rad2deg(lengths) > 30)),
            "exclude_gt_30deg_mean_shift_deg": float(
                np.rad2deg(np.linalg.norm(pooled_mean - under_30_mean))
            ),
            "trimming_changes_operational_calibration": False,
        })
    return result


def leave_one_session_out(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Fit one original calibration session; diagnose the other by SNR."""

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source_model_comparison"]), str(row["station_id"]),
                str(row["estimator_variant"]))].append(row)
    result = []
    benchmark = float(chi2.ppf(0.95, 2))
    for (source, station, method), group in sorted(groups.items()):
        sessions = sorted({str(row["paired_session_id"]) for row in group})
        if len(sessions) != 2:
            raise ValueError("leave-one-session-out requires exactly two source sessions")
        for train_session in sessions:
            test_session = next(session for session in sessions if session != train_session)
            train = _valid_vectors([
                row for row in group if row["paired_session_id"] == train_session
            ])
            train_mean, train_covariance, _ = _stats(train)
            eigenvalues = np.linalg.eigvalsh(train_covariance)
            if np.min(eigenvalues) <= 0:
                raise ValueError("single-session calibration covariance is not PD")
            for snr in SNR_LEVELS_DB:
                test = _valid_vectors([
                    row for row in group
                    if row["paired_session_id"] == test_session
                    and float(row["snr_db"]) == snr
                ])
                test_mean, test_covariance, lengths = _stats(test)
                centered = test - train_mean
                nis = np.einsum(
                    "ni,ni->n", centered,
                    np.linalg.solve(train_covariance, centered.T).T,
                )
                result.append({
                    "source_model_comparison": source,
                    "station_id": station, "estimator_variant": method,
                    "train_session_id": train_session, "test_session_id": test_session,
                    "test_snr_db": float(snr),
                    "train_dependent_frame_count": len(train),
                    "test_dependent_frame_count": len(test),
                    "train_mean_az_arc_rad": float(train_mean[0]),
                    "train_mean_el_arc_rad": float(train_mean[1]),
                    "train_covariance_trace_rad2": float(np.trace(train_covariance)),
                    "train_covariance_min_eigenvalue_rad2": float(np.min(eigenvalues)),
                    "test_mean_az_arc_rad": float(test_mean[0]),
                    "test_mean_el_arc_rad": float(test_mean[1]),
                    "test_covariance_trace_rad2": float(np.trace(test_covariance)),
                    "test_minus_train_bias_norm_deg": float(
                        np.rad2deg(np.linalg.norm(test_mean - train_mean))
                    ),
                    "test_median_geodesic_error_deg": float(np.rad2deg(np.median(lengths))),
                    "test_p95_geodesic_error_deg": float(np.rad2deg(np.percentile(lengths, 95))),
                    "centered_nis_p50": float(np.percentile(nis, 50)),
                    "centered_nis_p95": float(np.percentile(nis, 95)),
                    "centered_nis_p99": float(np.percentile(nis, 99)),
                    "centered_nis_fraction_gt_chi2_2_p95": float(np.mean(nis > benchmark)),
                    "chi2_2_p95_gaussian_benchmark": benchmark,
                    "gaussian_distribution_claimed": False,
                    "test_session_used_to_fit_mean_or_R": False,
                    "dependent_frames_are_independent_trials": False,
                })
    return result


def _verify_pooled_fit(
    calibration_records: dict[str, list[dict[str, object]]],
    saved_rows: list[dict[str, str]],
) -> dict[str, object]:
    saved = {
        (row["source_model_comparison"], row["station_id"], row["estimator_variant"]): row
        for row in saved_rows
    }
    if len(saved) != 12:
        raise ValueError("expected 12 saved pooled calibration rows")
    max_bias_difference = 0.0
    max_covariance_difference = 0.0
    for source, records in calibration_records.items():
        fits = calibrate_audio_bearings(records, 1)
        for (station, method), fit in fits.items():
            old = saved[(source, station, method)]
            old_bias = np.asarray([
                float(old["bias_az_arc_rad"]), float(old["bias_el_arc_rad"])
            ])
            old_covariance = np.asarray([
                [float(old["covariance_00_rad2"]), float(old["covariance_01_rad2"])],
                [float(old["covariance_01_rad2"]), float(old["covariance_11_rad2"])],
            ])
            if fit.dependent_frame_count != int(old["dependent_frame_count"]):
                raise AssertionError("calibration dependent-frame count changed")
            if fit.successful_frame_count != int(old["successful_frame_count"]):
                raise AssertionError("calibration successful-frame count changed")
            max_bias_difference = max(max_bias_difference, float(np.max(np.abs(fit.mean_residual_rad-old_bias))))
            max_covariance_difference = max(max_covariance_difference, float(np.max(np.abs(fit.covariance_rad2-old_covariance))))
    if max_bias_difference > 1e-10 or max_covariance_difference > 1e-10:
        raise AssertionError("calibration-only rerun did not reproduce published pooled bias/R")
    return {
        "maximum_pooled_bias_difference_rad": max_bias_difference,
        "maximum_pooled_covariance_difference_rad2": max_covariance_difference,
        "absolute_reproducibility_tolerance": 1e-10,
    }


def run_calibration_only(
    results_dir: Path = ROOT / "results", *, progress: bool = False,
) -> dict[str, object]:
    """The only entry point that invokes acoustic synthesis in this addendum."""

    recording_ids = recording_ids_for_split("calibration")
    if len(recording_ids) != 2:
        raise ValueError("frozen protocol requires two original calibration recordings")
    saved_calibration = _read(results_dir / "s8_tracking_feasibility_calibration.csv")
    saved_seed_rows = _read(results_dir / "s8_tracking_feasibility_seed_provenance.csv")
    seed_lookup = {
        (row["paired_recording_id"], row["snr_db"], row["source_model_comparison"]): row
        for row in saved_seed_rows if row["split"] == "calibration"
    }
    compact = []
    records: dict[str, list[dict[str, object]]] = {source: [] for source in SOURCE_MODELS}
    provenance = []
    for session_index, recording_id in enumerate(recording_ids):
        for snr_index, snr_db in enumerate(SNR_LEVELS_DB):
            pair = generate_paired_sequences(
                recording_id, "calibration", session_index, snr_index,
                duration_s=DURATION_S,
            )
            for source, (_, _, stream, rows, runtimes) in pair.items():
                if source not in SOURCE_MODELS or any(row["split"] != "calibration" for row in rows):
                    raise AssertionError("calibration-only source/split contract violated")
                saved = seed_lookup[(recording_id, str(float(snr_db)), source)]
                noise_seeds = {station.station_id: int(station.noise_seed) for station in stream.stations}
                if int(saved["sequence_seed"]) != stream.base_seed:
                    raise AssertionError("calibration sequence seed differs from frozen pilot")
                if int(saved["source_seed"]) != stream.source_seed:
                    raise AssertionError("calibration source seed differs from frozen pilot")
                if tuple(json.loads(saved["noise_seeds_json"])) != tuple(noise_seeds.values()):
                    raise AssertionError("calibration noise seeds differ from frozen pilot")
                records[source].extend(rows)
                compact.extend(compact_calibration_records(
                    rows, source_seed=stream.source_seed,
                    noise_seeds=noise_seeds, sequence_seed=stream.base_seed,
                ))
                provenance.append({
                    "split": "calibration", "paired_recording_id": recording_id,
                    "paired_session_id": rows[0]["paired_session_id"],
                    "source_model_comparison": source, "snr_db": float(snr_db),
                    "sequence_seed": stream.base_seed, "source_seed": stream.source_seed,
                    "noise_seeds_json": json.dumps(noise_seeds, sort_keys=True),
                    "dependent_frame_count": len(rows),
                    "audio_synthesis_wall_runtime_s": runtimes["audio_synthesis_wall_runtime_s"],
                    "bearing_frontend_wall_runtime_s": runtimes["bearing_frontend_wall_runtime_s"],
                    "evaluation_audio_synthesized": False,
                })
            if progress:
                print(f"calibration-only session={session_index} snr={snr_db:g} complete", flush=True)
    reproducibility = _verify_pooled_fit(records, saved_calibration)
    sessions = {row["paired_session_id"] for row in compact}
    if len(sessions) != 2 or len(compact) != 20160:
        raise AssertionError("calibration-only session/frame coverage changed")
    _write(results_dir / f"{RESULT_PREFIX}calibration_residuals.csv", compact)
    _write(results_dir / f"{RESULT_PREFIX}calibration_session_snr_summary.csv", session_snr_summaries(compact))
    _write(results_dir / f"{RESULT_PREFIX}calibration_pool_attribution.csv", pooled_session_attribution(compact))
    _write(results_dir / f"{RESULT_PREFIX}calibration_leave_one_session_out.csv", leave_one_session_out(compact))
    _write(results_dir / f"{RESULT_PREFIX}calibration_seed_provenance.csv", provenance)
    audit = {
        "split": "calibration_only",
        "original_session_count": len(sessions),
        "calibration_sequence_count": len(provenance),
        "dependent_calibration_frame_count": len(compact),
        "evaluation_audio_synthesized": False,
        "tracker_invoked": False,
        **reproducibility,
    }
    (results_dir / f"{RESULT_PREFIX}calibration_recovery_audit.json").write_text(
        json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def reanalyze_saved_calibration_residuals(
    results_dir: Path = ROOT / "results",
) -> dict[str, int]:
    """Refresh only derived tables from saved residuals; no audio processing."""

    compact = _read(results_dir / f"{RESULT_PREFIX}calibration_residuals.csv")
    if not compact or any(row["split"] != "calibration" for row in compact):
        raise ValueError("saved residual source must contain calibration rows only")
    summaries = session_snr_summaries(compact)
    attribution = pooled_session_attribution(compact)
    cross = leave_one_session_out(compact)
    _write(results_dir / f"{RESULT_PREFIX}calibration_session_snr_summary.csv", summaries)
    _write(results_dir / f"{RESULT_PREFIX}calibration_pool_attribution.csv", attribution)
    _write(results_dir / f"{RESULT_PREFIX}calibration_leave_one_session_out.csv", cross)
    result = {
        "dependent_calibration_frame_count": len(compact),
        "session_snr_group_count": len(summaries),
        "pooled_attribution_group_count": len(attribution),
        "directed_leave_one_session_out_group_count": len(cross),
    }
    evaluation_path = results_dir / "s8_tracking_feasibility_bearing_results.csv"
    if evaluation_path.exists():
        evaluation = _read(evaluation_path)
        keys = [
            (row["source_model_comparison"], row["paired_session_id"],
             row["snr_db"], row["station_id"], row["estimator_variant"],
             row["frame_index"])
            for row in compact
        ]
        calibration_sessions = {row["paired_session_id"] for row in compact}
        evaluation_sessions = {row["paired_session_id"] for row in evaluation}
        calibration_recordings = {row["paired_recording_id"] for row in compact}
        evaluation_recordings = {row["paired_recording_id"] for row in evaluation}
        calibration_origins = {row["paired_origin_asset_id"] for row in compact}
        evaluation_origins = {row["paired_origin_asset_id"] for row in evaluation}
        seed_overlap = {}
        for field in ("sequence_seed", "source_seed", "noise_seed"):
            seed_overlap[field] = len(
                {int(row[field]) for row in compact}
                & {int(row[field]) for row in evaluation}
            )
        if (
            len(keys) != len(set(keys)) or calibration_sessions & evaluation_sessions
            or calibration_recordings & evaluation_recordings
            or calibration_origins & evaluation_origins
            or any(seed_overlap.values())
        ):
            raise AssertionError("calibration/evaluation provenance or frame keys overlap")
        result.update({
            "duplicate_calibration_frame_key_count": 0,
            "calibration_evaluation_session_overlap_count": 0,
            "calibration_evaluation_recording_overlap_count": 0,
            "calibration_evaluation_origin_overlap_count": 0,
            "calibration_evaluation_seed_overlap_counts": seed_overlap,
        })
        audit_path = results_dir / f"{RESULT_PREFIX}calibration_recovery_audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit.update(result)
            audit_path.write_text(
                json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--reanalyze-saved", action="store_true",
        help="recompute only derived tables from the persisted calibration residual CSV",
    )
    args = parser.parse_args()
    result = (
        reanalyze_saved_calibration_residuals(args.results_dir)
        if args.reanalyze_saved
        else run_calibration_only(args.results_dir, progress=args.progress)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
