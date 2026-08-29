"""Fast deterministic Monte Carlo checks for the local small-noise regime."""

import numpy as np
import pytest

from model.geometry import comparison_arrays
from validation.monte_carlo import run_monte_carlo_wls
from validation.study import write_summary_csv


DIRECTIONS_DEG = [(20.0, 10.0), (45.0, 30.0), (120.0, 50.0)]


@pytest.mark.parametrize("array_name", ["square", "tetrahedral"])
@pytest.mark.parametrize("direction_index", range(len(DIRECTIONS_DEG)))
def test_low_noise_wls_monte_carlo_approaches_crlb(array_name, direction_index):
    arrays = comparison_arrays()
    phi, elevation = np.deg2rad(DIRECTIONS_DEG[direction_index])
    array_index = 0 if array_name == "square" else 1
    result = run_monte_carlo_wls(
        arrays[array_name],
        phi,
        elevation,
        array_name=array_name,
        noise_model="independent_tdoa",
        sigma_tdoa=1e-6,
        n_trials=256,
        seed=20260827 + 10 * array_index + direction_index,
    )

    # Deliberately loose fixed-seed thresholds for a fast stochastic test.
    assert 0.80 <= result.rmse_to_crlb_ratio <= 1.20
    assert np.all(result.normalized_covariance_eigenvalues >= 0.65)
    assert np.all(result.normalized_covariance_eigenvalues <= 1.45)
    coordinate_bias_z = np.abs(result.bias_rad) / np.sqrt(np.diag(result.crlb_covariance))
    assert np.all(coordinate_bias_z < 0.30)
    assert result.optimizer_success_fraction == 1.0
    assert result.boundary_fraction == 0.0


def test_toa_monte_carlo_uses_requested_marginal_difference_std():
    positions = comparison_arrays()["tetrahedral"]
    sigma_toa = 50e-6 / np.sqrt(2.0)
    result = run_monte_carlo_wls(
        positions,
        *np.deg2rad([45.0, 30.0]),
        array_name="tetrahedral",
        noise_model="independent_toa",
        sigma_toa=sigma_toa,
        n_trials=128,
        seed=20260899,
    )
    assert result.sigma_tdoa is None
    assert result.sigma_toa == pytest.approx(sigma_toa)
    assert result.marginal_tdoa_std == pytest.approx(50e-6)
    assert np.all(np.isfinite(result.empirical_covariance))


def test_monte_carlo_is_exactly_reproducible_for_fixed_seed():
    positions = comparison_arrays()["square"]
    arguments = dict(
        positions=positions,
        phi=np.deg2rad(45.0),
        elevation=np.deg2rad(30.0),
        array_name="square",
        sigma_tdoa=5e-6,
        n_trials=64,
        seed=314159,
    )
    first = run_monte_carlo_wls(**arguments)
    second = run_monte_carlo_wls(**arguments)
    np.testing.assert_array_equal(first.bias_rad, second.bias_rad)
    np.testing.assert_array_equal(first.empirical_covariance, second.empirical_covariance)
    assert first.geodesic_rmse_rad == second.geodesic_rmse_rad


def test_identical_summary_csv_does_not_require_destructive_write_access(
    tmp_path, monkeypatch
):
    output = tmp_path / "summary.csv"
    records = [{"value": 1.25, "label": "fixed-seed"}]
    write_summary_csv(records, output)
    path_type = type(output)
    original_open = path_type.open

    def reject_writes(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == output and "w" in mode:
            raise PermissionError("simulated editor lock")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "open", reject_writes)
    assert write_summary_csv(records, output) == output
