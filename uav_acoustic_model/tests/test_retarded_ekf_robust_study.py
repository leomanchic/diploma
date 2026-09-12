"""Statistical-contract tests for the held-out S7C-D2 comparison."""

from validation.retarded_ekf_robust_study import (
    DEFAULT_ROBUST_EVALUATION_SEED,
    DEFAULT_ROBUST_SMOKE_SEED,
    MECHANISMS,
    ROBUST_VARIANTS,
    audit_robust_results,
    run_robust_scenario,
)
from validation.retarded_ekf_stress_study import (
    DEFAULT_STRESS_SEED,
    default_stress_profiles,
    generate_stress_base_block,
    generate_stress_scenario,
    stress_seed_provenance,
)


def test_ablation_matrix_and_seed_scopes_are_fixed_and_disjoint():
    assert [(item.consensus_initialization, item.nis_gate) for item in ROBUST_VARIANTS] == [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    assert len({DEFAULT_STRESS_SEED, DEFAULT_ROBUST_SMOKE_SEED, DEFAULT_ROBUST_EVALUATION_SEED}) == 3
    generated = {}
    for base_seed in (
        DEFAULT_STRESS_SEED,
        DEFAULT_ROBUST_SMOKE_SEED,
        DEFAULT_ROBUST_EVALUATION_SEED,
    ):
        values = {
            seed
            for geometry_index in range(2)
            for sequence_index in range(100)
            for seed in stress_seed_provenance(
                base_seed, geometry_index, sequence_index
            ).generated_seeds
        }
        assert len(values) == 1_200
        generated[base_seed] = values
    assert generated[DEFAULT_STRESS_SEED].isdisjoint(
        generated[DEFAULT_ROBUST_SMOKE_SEED]
    )
    assert generated[DEFAULT_STRESS_SEED].isdisjoint(
        generated[DEFAULT_ROBUST_EVALUATION_SEED]
    )
    assert generated[DEFAULT_ROBUST_SMOKE_SEED].isdisjoint(
        generated[DEFAULT_ROBUST_EVALUATION_SEED]
    )


def test_variants_receive_same_stream_and_clean_baseline_is_unchanged():
    base = generate_stress_base_block(
        "informative", 4, base_seed=DEFAULT_ROBUST_SMOKE_SEED
    )
    nominal = next(item for item in default_stress_profiles() if item.name == "nominal")
    scenario = generate_stress_scenario(base, nominal)
    rows = run_robust_scenario(scenario)
    assert len(rows) == 4
    assert {row["sequence_id"] for row in rows} == {scenario.sequence_id}
    assert {row["delivered_event_count"] for row in rows} == {len(scenario.events)}
    assert all(not row["outlier_truth_used_by_estimator"] for row in rows)
    baseline = next(row for row in rows if row["variant"] == "c1_baseline")
    assert baseline["final_valid"]
    assert baseline["robust_rejected_event_count"] == 0
    assert baseline["false_rejected_clean_event_count"] == 0
    assert baseline["remaining_unclassified_delivered_event_count"] == 0


def test_evaluator_outlier_confusion_counts_partition_delivered_events():
    base = generate_stress_base_block(
        "informative", 0, base_seed=DEFAULT_ROBUST_SMOKE_SEED
    )
    strong = next(
        item for item in default_stress_profiles() if item.name == "outlier_strong"
    )
    rows = run_robust_scenario(generate_stress_scenario(base, strong))
    combined = next(row for row in rows if row["variant"] == "robust_combined")
    assert combined["final_valid"]
    assert combined["detected_outlier_event_count"] > 0
    assert combined["missed_outlier_event_count"] == 0
    assert (
        combined["initialization_event_count"]
        + combined["applied_update_event_count"]
        + combined["robust_rejected_event_count"]
        + combined["other_rejected_event_count"]
        + combined["remaining_unclassified_delivered_event_count"]
        == combined["delivered_event_count"]
    )
    assert (
        combined["detected_outlier_event_count"]
        + combined["missed_outlier_event_count"]
        + combined["unclassified_outlier_event_count"]
        == combined["delivered_outlier_event_count"]
    )


def test_persisted_false_strings_are_not_interpreted_as_true():
    """The CSV audit must parse booleans, not apply bool() to strings."""

    sequence_rows = []
    for geometry in ("informative", "poor"):
        for profile in default_stress_profiles():
            for sequence_index in range(100):
                for variant in ROBUST_VARIANTS:
                    sequence_rows.append(
                        {
                            "geometry": geometry,
                            "profile": profile.name,
                            "sequence_index": sequence_index,
                            "variant": variant.name,
                            "delivered_event_count": 1,
                            "initialization_event_count": 1,
                            "applied_update_event_count": 0,
                            "robust_rejected_event_count": 0,
                            "other_rejected_event_count": 0,
                            "remaining_unclassified_delivered_event_count": 0,
                            "outlier_truth_used_by_estimator": "False",
                        }
                    )
    summary_rows = [object()] * 72
    seed_rows = []
    for geometry_index in range(2):
        for sequence_index in range(100):
            provenance = stress_seed_provenance(
                DEFAULT_ROBUST_EVALUATION_SEED,
                geometry_index,
                sequence_index,
            )
            row = {}
            for name, identifier, seed in zip(
                MECHANISMS,
                provenance.identifiers,
                provenance.generated_seeds,
                strict=True,
            ):
                row[f"{name}_provenance"] = identifier
                row[f"{name}_seed"] = seed
            seed_rows.append(row)
    audited = audit_robust_results(
        sequence_rows,
        summary_rows,
        seed_rows,
        sequence_count=100,
    )
    assert audited["sequence_row_count"] == 7_200
