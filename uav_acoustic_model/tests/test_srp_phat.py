"""Deterministic far-field SRP-PHAT tests."""

import numpy as np

from estimators.srp_phat import (
    direct_srp_phat_scores,
    srp_phat,
    vectorized_srp_phat_scores,
)
from model.geometry import all_pairs, comparison_arrays, direction_vector
from simulation.propagation import simulate_propagation
from simulation.signals import deterministic_bandlimited_signal
from validation.gcc_statistical import GCCStatisticalConfig
from validation.srp_statistical import (
    SRP_METHODS,
    default_srp_statistical_configurations,
    run_srp_paired_configuration,
    run_srp_statistical_validation,
)


FS = 48_000.0
BAND = dict(minimum_frequency_hz=300.0, maximum_frequency_hz=10_000.0)


def _channels(geometry="tetrahedral", azimuth_deg=40.0, elevation_deg=30.0):
    positions = comparison_arrays()[geometry]
    source = deterministic_bandlimited_signal(
        FS,
        0.06,
        minimum_frequency_hz=300.0,
        maximum_frequency_hz=10_000.0,
        tone_count=41,
    )
    propagation = simulate_propagation(
        source,
        FS,
        positions,
        phi=np.deg2rad(azimuth_deg),
        elevation=np.deg2rad(elevation_deg),
        distance_m=1000.0,
        propagation_model="plane",
        pairs=all_pairs(4),
        delay_method="frequency",
    )
    return positions, propagation


def _angular_error_deg(result, azimuth_deg, elevation_deg):
    truth = direction_vector(np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg))
    return np.rad2deg(np.arccos(np.clip(result.direction @ truth, -1.0, 1.0)))


def test_direct_and_vectorized_srp_scores_match():
    positions, propagation = _channels()
    directions = direction_vector(
        np.deg2rad([0.0, 40.0, 90.0, 220.0]),
        np.deg2rad([10.0, 30.0, 50.0, -20.0]),
    )
    direct = direct_srp_phat_scores(
        propagation.channels,
        FS,
        positions,
        directions,
        valid_region=propagation.valid_region,
        **BAND,
    )
    vectorized = vectorized_srp_phat_scores(
        propagation.channels,
        FS,
        positions,
        directions,
        valid_region=propagation.valid_region,
        **BAND,
    )
    assert not direct.invalid and not vectorized.invalid
    np.testing.assert_allclose(vectorized.scores, direct.scores, rtol=2e-13, atol=2e-13)


def test_srp_sign_and_on_grid_direction_recovery():
    positions, propagation = _channels(azimuth_deg=40.0, elevation_deg=30.0)
    result = srp_phat(
        propagation.channels,
        FS,
        positions,
        valid_region=propagation.valid_region,
        coarse_to_fine_steps_deg=(10.0,),
        local_refinement=False,
        **BAND,
    )
    assert not result.invalid
    assert _angular_error_deg(result, 40.0, 30.0) < 1e-6
    candidates = direction_vector(
        np.deg2rad([40.0, 220.0]), np.deg2rad([30.0, -30.0])
    )
    scores = direct_srp_phat_scores(
        propagation.channels,
        FS,
        positions,
        candidates,
        valid_region=propagation.valid_region,
        **BAND,
    ).scores
    assert scores[0] > scores[1]


def test_local_refinement_removes_direction_grid_quantization():
    positions, propagation = _channels(azimuth_deg=43.2, elevation_deg=27.7)
    common = dict(
        valid_region=propagation.valid_region,
        coarse_to_fine_steps_deg=(10.0, 2.0, 0.5),
        **BAND,
    )
    grid_only = srp_phat(
        propagation.channels, FS, positions, local_refinement=False, **common
    )
    refined = srp_phat(
        propagation.channels, FS, positions, local_refinement=True, **common
    )
    assert _angular_error_deg(refined, 43.2, 27.7) < 0.12
    assert _angular_error_deg(refined, 43.2, 27.7) < _angular_error_deg(
        grid_only, 43.2, 27.7
    )
    assert refined.score >= grid_only.score - 1e-12
    assert refined.local_refinement_evaluations > 0


def test_coarse_search_does_not_lock_onto_a_high_frequency_sidelobe():
    """The global grid must sample the main lobe before local refinement."""

    positions, propagation = _channels(
        "tetrahedral", azimuth_deg=20.0, elevation_deg=10.0
    )
    result = srp_phat(
        propagation.channels,
        FS,
        positions,
        valid_region=propagation.valid_region,
        coarse_to_fine_steps_deg=(5.0, 1.0, 0.25),
        **BAND,
    )
    # The finite common crop perturbs the continuous-frame PHAT maximum by
    # about 0.18 degrees; the regression target is rejection of the former
    # 27-degree sidelobe, not a zero-bias finite-window assertion.
    assert _angular_error_deg(result, 20.0, 10.0) < 0.25


def test_translation_and_microphone_permutation_invariance():
    positions, propagation = _channels(azimuth_deg=70.0, elevation_deg=20.0)
    kwargs = dict(
        valid_region=propagation.valid_region,
        coarse_to_fine_steps_deg=(15.0, 3.0, 0.75),
        **BAND,
    )
    original = srp_phat(propagation.channels, FS, positions, **kwargs)
    translated = srp_phat(
        propagation.channels,
        FS,
        positions + np.array([11.0, -4.0, 2.5]),
        **kwargs,
    )
    permutation = np.array([2, 0, 3, 1])
    permuted = srp_phat(
        propagation.channels[permutation],
        FS,
        positions[permutation],
        **kwargs,
    )
    np.testing.assert_allclose(translated.direction, original.direction, atol=2e-10)
    np.testing.assert_allclose(permuted.direction, original.direction, atol=2e-10)
    assert abs(translated.score - original.score) < 2e-13
    assert abs(permuted.score - original.score) < 2e-13


def test_pair_orientation_invariance_when_spectrum_and_delay_are_reversed():
    positions, propagation = _channels()
    pairs = all_pairs(4)
    reversed_pairs = tuple((second, first) for first, second in pairs)
    directions = direction_vector(
        np.deg2rad([10.0, 40.0, 120.0]), np.deg2rad([5.0, 30.0, -40.0])
    )
    forward = vectorized_srp_phat_scores(
        propagation.channels,
        FS,
        positions,
        directions,
        pairs=pairs,
        valid_region=propagation.valid_region,
        **BAND,
    )
    reversed_result = vectorized_srp_phat_scores(
        propagation.channels,
        FS,
        positions,
        directions,
        pairs=reversed_pairs,
        valid_region=propagation.valid_region,
        **BAND,
    )
    np.testing.assert_allclose(reversed_result.scores, forward.scores, atol=2e-13)


def test_square_mirror_ambiguity_is_preserved():
    positions, propagation = _channels("square", 40.0, 30.0)
    mirrored = direction_vector(
        np.deg2rad([40.0, 40.0]), np.deg2rad([30.0, -30.0])
    )
    scores = direct_srp_phat_scores(
        propagation.channels,
        FS,
        positions,
        mirrored,
        valid_region=propagation.valid_region,
        **BAND,
    ).scores
    np.testing.assert_allclose(scores[0], scores[1], rtol=0.0, atol=2e-13)


def test_tetrahedral_array_distinguishes_upper_and_lower_halfspaces():
    positions, upper_channels = _channels("tetrahedral", 40.0, 30.0)
    _, lower_channels = _channels("tetrahedral", 40.0, -30.0)
    candidates = direction_vector(
        np.deg2rad([40.0, 40.0]), np.deg2rad([30.0, -30.0])
    )
    upper_scores = vectorized_srp_phat_scores(
        upper_channels.channels,
        FS,
        positions,
        candidates,
        valid_region=upper_channels.valid_region,
        **BAND,
    ).scores
    lower_scores = vectorized_srp_phat_scores(
        lower_channels.channels,
        FS,
        positions,
        candidates,
        valid_region=lower_channels.valid_region,
        **BAND,
    ).scores
    assert upper_scores[0] > upper_scores[1]
    assert lower_scores[1] > lower_scores[0]
    lower_estimate = srp_phat(
        lower_channels.channels,
        FS,
        positions,
        valid_region=lower_channels.valid_region,
        elevation_bounds_rad=(-np.pi / 2.0, np.pi / 2.0),
        coarse_to_fine_steps_deg=(5.0, 1.0, 0.25),
        **BAND,
    )
    assert _angular_error_deg(lower_estimate, 40.0, -30.0) < 0.25


def test_silence_is_invalid_instead_of_returning_a_direction():
    positions = comparison_arrays()["tetrahedral"]
    result = srp_phat(np.zeros((4, 1024)), FS, positions, **BAND)
    assert result.invalid
    assert "signal_energy_below_threshold" in result.invalid_reason
    assert np.isnan(result.phi) and np.isnan(result.elevation)
    assert np.all(np.isnan(result.direction))


def test_result_is_finite_and_on_unit_sphere():
    positions, propagation = _channels(azimuth_deg=120.0, elevation_deg=50.0)
    result = srp_phat(
        propagation.channels,
        FS,
        positions,
        valid_region=propagation.valid_region,
        coarse_to_fine_steps_deg=(15.0, 3.0, 0.75),
        **BAND,
    )
    assert not result.invalid
    assert np.all(np.isfinite(result.direction))
    assert np.isfinite(result.score) and np.isfinite(result.runtime_seconds)
    assert abs(np.linalg.norm(result.direction) - 1.0) < 2e-15
    assert 0.0 <= result.phi < 2.0 * np.pi
    assert 0.0 <= result.elevation <= np.pi / 2.0


def test_paired_smoke_monte_carlo_uses_common_evaluation_set_and_separate_seeds():
    config = GCCStatisticalConfig(
        "smoke",
        "deterministic_multisine",
        "tetrahedral",
        45.0,
        30.0,
        0.0,
        1024,
    )
    doa, runtime = run_srp_paired_configuration(
        config,
        700,
        calibration_trial_count=12,
        evaluation_trial_count=20,
        exact_reference_trials=5,
    )
    assert {row["estimator_variant"] for row in doa} == set(SRP_METHODS)
    assert len({row["evaluation_seed"] for row in doa}) == 1
    assert len({row["calibration_seed"] for row in doa}) == 1
    assert doa[0]["evaluation_seed"] != doa[0]["calibration_seed"]
    assert all(row["common_random_numbers"] is True for row in doa)
    assert all(row["successful_trial_count"] + row["unsuccessful_trial_count"] == 20 for row in doa)
    assert all(row["conditional_p95_geodesic_error_deg"] <= row["conditional_p99_geodesic_error_deg"] <= row["conditional_p999_geodesic_error_deg"] for row in doa)
    srp = next(row for row in doa if row["estimator_variant"] == "equal_weight_srp_phat")
    assert srp["equal_pair_weights"] is True
    assert srp["gcc_confidence_used_by_srp"] is False
    exact = next(
        row for row in runtime if row["runtime_component"] == "srp_exact_vectorized_reference"
    )
    assert exact["runtime_sample_count"] == 5
    assert exact["exact_reference_trial_count"] == 5
    assert sum(row["exact_reference_trial_count"] for row in runtime) == 5
    assert all(row["exact_reference_trials_per_configuration"] == 5 for row in runtime)
    assert all(
        row["exact_fast_disagreement_covers_all_evaluation_trials"] is False
        for row in runtime
    )
    assert exact["exact_fast_disagreement_scope"] == "sampled_exact_reference_trials_only"
    assert all(
        row["exact_fast_disagreement_scope"] == "not_applicable"
        for row in runtime
        if row is not exact
    )
    assert exact["max_exact_fast_disagreement_deg"] < 0.05


def test_full_srp_configuration_grid_is_the_requested_cartesian_product():
    configs = default_srp_statistical_configurations()
    assert len(configs) == 2 * 3 * 3 * 11
    assert {config.geometry for config in configs} == {"square", "tetrahedral"}
    assert {config.signal_model for config in configs} == {
        "deterministic_multisine",
        "random_broadband",
        "harmonic_stress",
    }
    assert {config.snr_db for config in configs} == {
        -10.0,
        -8.0,
        -6.0,
        -4.0,
        -2.0,
        0.0,
        2.0,
        5.0,
        10.0,
        20.0,
        30.0,
    }
    assert {(config.azimuth_deg, config.elevation_deg) for config in configs} == {
        (20.0, 10.0),
        (45.0, 30.0),
        (120.0, 50.0),
    }


def test_srp_statistical_csv_schema_and_seeded_metrics_are_reproducible(tmp_path):
    config = GCCStatisticalConfig(
        "smoke",
        "deterministic_multisine",
        "square",
        20.0,
        10.0,
        5.0,
        1024,
    )
    kwargs = dict(
        configurations=(config,),
        calibration_trial_count=10,
        evaluation_trial_count=16,
        exact_reference_trials=2,
        doa_output_csv=tmp_path / "doa.csv",
        runtime_output_csv=tmp_path / "runtime.csv",
    )
    first_doa, first_runtime = run_srp_statistical_validation(**kwargs)
    second_doa, second_runtime = run_srp_statistical_validation(**kwargs)
    assert len(first_doa) == len(second_doa) == 4
    assert len(first_runtime) == len(second_runtime) == 3
    nondeterministic = {
        "mean_runtime_per_estimate_s",
        "median_runtime_per_estimate_s",
        "p95_runtime_per_estimate_s",
    }
    for first, second in zip(first_doa, second_doa, strict=True):
        assert {k: v for k, v in first.items() if k not in nondeterministic} == {
            k: v for k, v in second.items() if k not in nondeterministic
        }
    required = {
        "study_type",
        "signal_model",
        "split",
        "geometry",
        "direction",
        "SNR",
        "frame_length",
        "pair",
        "calibration_trial_count",
        "evaluation_trial_count",
        "seed",
        "estimator_variant",
        "conditional_p99_geodesic_error_deg",
        "conditional_p999_geodesic_error_deg",
        "coverage",
    }
    assert required <= set(first_doa[0])
    assert (tmp_path / "doa.csv").exists()
    assert (tmp_path / "runtime.csv").exists()
