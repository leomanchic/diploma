"""Fast algebraic checks; no recorded audio is loaded by this test module."""

from __future__ import annotations

import csv
import json

import numpy as np

from validation.s8_calibration_session_recovery import (
    leave_one_session_out, pooled_session_attribution,
    reanalyze_saved_calibration_residuals, session_snr_summaries,
)


def _synthetic_calibration_rows() -> list[dict[str, object]]:
    rng = np.random.default_rng(20260919)
    rows = []
    for session, session_bias in (("original-A", 0.0), ("original-B", 0.08)):
        for snr, snr_bias in ((-6.0, -0.01), (10.0, 0.01)):
            values = rng.normal(scale=[0.025, 0.035], size=(40, 2))
            values += [snr_bias, session_bias]
            for index, residual in enumerate(values):
                rows.append({
                    "source_model_comparison": "recorded_source_approximation",
                    "paired_session_id": session, "snr_db": snr,
                    "station_id": "S0", "estimator_variant": "all_6_equal_gcc_wls",
                    "frame_index": index, "valid": True,
                    "residual_rad_0": float(residual[0]),
                    "residual_rad_1": float(residual[1]),
                })
    return rows


def test_session_snr_grain_and_scatter_decomposition_are_exact():
    rows = _synthetic_calibration_rows()
    summaries = session_snr_summaries(rows)
    assert len(summaries) == 4
    assert all(row["dependent_frame_count"] == 40 for row in summaries)
    attribution = pooled_session_attribution(rows)
    assert len(attribution) == 1
    pooled = attribution[0]
    assert pooled["pooled_valid_frame_count"] == 160
    np.testing.assert_allclose(
        pooled["between_session_scatter_trace_fraction"]
        + pooled["within_session_scatter_trace_fraction"],
        1.0, rtol=0, atol=1e-12,
    )
    assert pooled["scatter_reconstruction_max_abs_rad2"] < 1e-12
    session_within = json.loads(pooled["session_within_scatter_trace_fraction_json"])
    np.testing.assert_allclose(
        sum(session_within.values()) + pooled["between_session_scatter_trace_fraction"],
        1.0, rtol=0, atol=1e-12,
    )
    assert pooled["session_mean_separation_deg"] > 3.0
    assert pooled["trimming_changes_operational_calibration"] is False


def test_leave_one_session_out_never_uses_test_session_to_fit_mean_or_R():
    rows = _synthetic_calibration_rows()
    before = leave_one_session_out(rows)
    assert len(before) == 4
    changed = [dict(row) for row in rows]
    for row in changed:
        if row["paired_session_id"] == "original-B":
            row["residual_rad_1"] = float(row["residual_rad_1"]) + 0.2
    after = leave_one_session_out(changed)
    for first, second in zip(before, after, strict=True):
        assert first["train_session_id"] != first["test_session_id"]
        assert first["test_session_used_to_fit_mean_or_R"] is False
        if first["train_session_id"] == "original-A":
            np.testing.assert_allclose(
                first["train_mean_el_arc_rad"], second["train_mean_el_arc_rad"],
                rtol=0, atol=0,
            )
            np.testing.assert_allclose(
                first["train_covariance_trace_rad2"],
                second["train_covariance_trace_rad2"], rtol=0, atol=0,
            )
            assert second["test_minus_train_bias_norm_deg"] > first[
                "test_minus_train_bias_norm_deg"
            ]


def test_heavy_tail_diagnostic_does_not_modify_pooled_operational_fit():
    rows = _synthetic_calibration_rows()
    rows[-1] = dict(rows[-1], residual_rad_1=1.0)
    pooled = pooled_session_attribution(rows)[0]
    assert pooled["top_5pct_removed_count"] >= 1
    assert pooled["top_5pct_mean_shift_deg"] > 0.1
    assert pooled["pooled_over_top_5pct_trimmed_trace_ratio"] > 1.0
    assert pooled["trimming_changes_operational_calibration"] is False


def test_saved_residual_reanalysis_never_calls_acoustic_generator(tmp_path, monkeypatch):
    rows = [dict(row, split="calibration") for row in _synthetic_calibration_rows()]
    path = tmp_path / "s8_calibration_transfer_calibration_residuals.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-data reanalysis must not synthesize audio")
    monkeypatch.setattr(
        "validation.s8_calibration_session_recovery.generate_paired_sequences",
        forbidden,
    )
    result = reanalyze_saved_calibration_residuals(tmp_path)
    assert result == {
        "dependent_calibration_frame_count": 160,
        "session_snr_group_count": 4,
        "pooled_attribution_group_count": 1,
        "directed_leave_one_session_out_group_count": 4,
    }
