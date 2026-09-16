"""Statistical-study contracts for S7C-D confirmation and recovery."""

import json

import numpy as np

from validation.initialization_recovery_study import (
    DEVELOPMENT_SEED,
    EVALUATION_SEED,
    SMOKE_SEED,
    VARIANTS,
    audit_recovery_results,
    generate_known_failure_journal,
    run_initialization_recovery_study,
    run_recovery_scenario,
)
from validation.retarded_ekf_robust_study import (
    DEFAULT_ROBUST_EVALUATION_SEED,
    DEFAULT_ROBUST_SMOKE_SEED,
    run_robust_scenario,
)
from validation.retarded_ekf_stress_study import (
    DEFAULT_STRESS_SEED,
    default_stress_profiles,
    generate_stress_base_block,
    generate_stress_scenario,
)


def _profile(name: str):
    return next(item for item in default_stress_profiles() if item.name == name)


def test_protocol_seeds_are_disjoint_and_variants_are_frozen():
    seeds = {
        DEFAULT_STRESS_SEED,
        DEFAULT_ROBUST_SMOKE_SEED,
        DEFAULT_ROBUST_EVALUATION_SEED,
        DEVELOPMENT_SEED,
        EVALUATION_SEED,
        SMOKE_SEED,
    }
    assert len(seeds) == 6
    assert [item.name for item in VARIANTS] == [
        "c1_baseline",
        "d2_combined_published",
        "confirmed_recovery",
    ]


def test_smoke_study_has_paired_streams_and_complete_event_partitions(tmp_path):
    rows, summaries, seeds = run_initialization_recovery_study(
        tmp_path,
        sequence_count=1,
        base_seed=SMOKE_SEED,
    )
    audit = audit_recovery_results(rows, summaries, seeds, sequence_count=1)
    assert audit == {
        "sequence_row_count": 54,
        "summary_row_count": 54,
        "seed_row_count": 2,
        "unique_mechanism_seed_count": 12,
    }
    for row in rows:
        assert not row["truth_used_by_estimator"]
        assert row["independent_unit"] == "whole_base_sequence"
        assert row["epochs_within_sequence_are_dependent"]
        assert row["delivered_event_count"] == (
            row["initialization_event_count"]
            + row["applied_update_event_count"]
            + row["rejected_event_count"]
            + row["unclassified_delivered_event_count"]
        )
    assert (tmp_path / "initialization_recovery_failure_journal.csv").exists()


def test_c1_and_published_d2_rows_reproduce_existing_variants():
    block = generate_stress_base_block(
        "informative",
        3,
        base_seed=SMOKE_SEED,
    )
    scenario = generate_stress_scenario(block, _profile("outlier_strong"))
    recovery_rows = {row["variant"]: row for row in run_recovery_scenario(scenario)}
    published_rows = {row["variant"]: row for row in run_robust_scenario(scenario)}
    mapping = {
        "c1_baseline": "c1_baseline",
        "d2_combined_published": "robust_combined",
    }
    for new_name, published_name in mapping.items():
        left = recovery_rows[new_name]
        right = published_rows[published_name]
        assert left["final_valid"] == right["final_valid"]
        np.testing.assert_allclose(
            left["position_error_m"],
            right["position_error_m"],
            rtol=0.0,
            atol=2e-12,
        )
        np.testing.assert_allclose(
            left["velocity_error_mps"],
            right["velocity_error_mps"],
            rtol=0.0,
            atol=2e-12,
        )
        assert json.loads(left["initialization_event_ids_json"]) == json.loads(
            right["initialization_event_ids_json"]
        )
        assert json.loads(left["applied_event_ids_json"]) == json.loads(
            right["applied_update_event_ids_json"]
        )


def test_known_failure_journal_captures_lock_and_recovery_without_truth_leakage():
    rows = generate_known_failure_journal()
    assert rows
    assert not any(row["truth_used_by_estimator"] for row in rows)
    finals = {
        (row["geometry"], row["variant"]): row
        for row in rows
        if row["record_kind"] == "final"
    }
    d2_informative_error = finals[("informative", "d2_combined_published")][
        "evaluator_only_position_error_m"
    ]
    d2_poor_error = finals[("poorly_conditioned", "d2_combined_published")][
        "evaluator_only_position_error_m"
    ]
    # GitHub Actions measured cross-platform differences of 1.64242124e-7 m
    # for the informative fixture and 9.322126e-7 m for the poorly conditioned
    # fixture under pinned Windows/Linux Python 3.12 jobs.  Absolute gates
    # remain below five parts per billion of their respective failure scales;
    # rtol=0 and the separate >100 m/>200 m semantic checks remain explicit.
    np.testing.assert_allclose(
        d2_informative_error,
        114.52006460072242,
        rtol=0.0,
        atol=5e-7,
    )
    np.testing.assert_allclose(
        d2_poor_error,
        221.43049150295593,
        rtol=0.0,
        atol=1e-6,
    )
    assert d2_informative_error > 100.0
    assert d2_poor_error > 200.0
    assert (
        finals[("informative", "confirmed_recovery")][
            "evaluator_only_position_error_m"
        ]
        < 1.0
    )
    assert (
        finals[("poorly_conditioned", "confirmed_recovery")][
            "evaluator_only_position_error_m"
        ]
        < 2.0
    )
    assert any(
        row["variant"] == "confirmed_recovery"
        and row["action"] == "tentative_rejected"
        for row in rows
    )
