"""Fast reproducibility checks for signal-level GCC-PHAT Monte Carlo."""

import csv

from validation.gcc_monte_carlo import run_gcc_phat_monte_carlo


def _small_study(path):
    return run_gcc_phat_monte_carlo(
        array_names=("tetrahedral",),
        directions_deg=((45.0, 30.0),),
        snr_levels_db=(0.0, 20.0),
        trial_count=24,
        signal_duration_s=0.03,
        interpolation_factor=4,
        output_csv=path,
    )


def test_gcc_monte_carlo_is_reproducible_and_writes_complete_csv(tmp_path):
    path = tmp_path / "gcc_mc.csv"
    first = _small_study(path)
    first_bytes = path.read_bytes()
    second = _small_study(path)
    assert first == second
    assert path.read_bytes() == first_bytes
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(first) == len(rows) == 2
    assert all(row["noise_model"] == "independent_channel_sample_gaussian" for row in rows)
    assert all(row["delay_method"] == "windowed_sinc" for row in rows)


def test_gcc_monte_carlo_improves_with_snr_without_brittle_random_threshold(tmp_path):
    low, high = _small_study(tmp_path / "trend.csv")
    assert high["snr_db"] > low["snr_db"]
    assert high["tdoa_rmse_us"] < low["tdoa_rmse_us"]
    assert high["geodesic_rmse_deg"] < low["geodesic_rmse_deg"]
    assert high["wls_success_fraction"] == 1.0
    assert high["geodesic_rmse_deg"] < 1.0
