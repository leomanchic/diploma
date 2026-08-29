"""Configuration and CSV orchestration for the detailed CRLB study."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from model.geometry import comparison_arrays
from validation.monte_carlo import MonteCarloResult, run_monte_carlo_wls

BASE_SEED = 20260827
ARRAY_NAMES = ("square", "tetrahedral")
DIRECTIONS_DEG = ((20.0, 10.0), (45.0, 30.0), (120.0, 50.0))
SIGMA_TDOA_US = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
MARGINAL_TOA_TDOA_STD_US = 50.0
SIGMA_TOA_US = MARGINAL_TOA_TDOA_STD_US / np.sqrt(2.0)
EXPECTED_CONFIGURATION_COUNT = 42

LOW_NOISE_RATIO_RANGE = (0.90, 1.10)
LOW_NOISE_NORMALIZED_COVARIANCE_RANGE = (0.80, 1.20)
LOW_NOISE_MAX_COORDINATE_BIAS_Z = 0.15


def _stream_seed(*indices: int) -> int:
    return int(np.random.SeedSequence([BASE_SEED, *indices]).generate_state(1)[0])


def _record_with_context(
    result: MonteCarloResult,
    direction_id: str,
) -> dict[str, float | int | str | bool | None]:
    record = result.to_record()
    coordinate_bias_z = np.abs(result.bias_rad) / np.sqrt(np.diag(result.crlb_covariance))
    record.update(
        {
            "direction_id": direction_id,
            "pair_scheme": "reference_0",
            "bias_z_azimuth": coordinate_bias_z[0],
            "bias_z_elevation": coordinate_bias_z[1],
        }
    )
    if result.noise_model == "independent_tdoa" and np.isclose(result.sigma_tdoa, 1e-6):
        ratio_pass = LOW_NOISE_RATIO_RANGE[0] <= result.rmse_to_crlb_ratio <= LOW_NOISE_RATIO_RANGE[1]
        covariance_pass = bool(
            np.all(
                result.normalized_covariance_eigenvalues
                >= LOW_NOISE_NORMALIZED_COVARIANCE_RANGE[0]
            )
            and np.all(
                result.normalized_covariance_eigenvalues
                <= LOW_NOISE_NORMALIZED_COVARIANCE_RANGE[1]
            )
        )
        bias_pass = bool(np.all(coordinate_bias_z <= LOW_NOISE_MAX_COORDINATE_BIAS_Z))
        record.update(
            {
                "low_noise_ratio_pass": ratio_pass,
                "low_noise_covariance_pass": covariance_pass,
                "low_noise_bias_pass": bias_pass,
                "low_noise_overall_pass": ratio_pass and covariance_pass and bias_pass,
            }
        )
    else:
        record.update(
            {
                "low_noise_ratio_pass": None,
                "low_noise_covariance_pass": None,
                "low_noise_bias_pass": None,
                "low_noise_overall_pass": None,
            }
        )
    return record


def write_summary_csv(
    records: list[dict[str, float | int | str | bool | None]],
    output_path: str | Path,
) -> Path:
    """Write aggregate results with a stable column order."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("at least one record is required")
    fieldnames = list(records[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    content = buffer.getvalue()
    # An editor or spreadsheet may keep the result file open on Windows.
    # Reproducible reruns need no write access when the exact bytes already
    # exist, so avoid opening an identical result in destructive write mode.
    if path.exists() and path.read_bytes() == content.encode("utf-8"):
        return path
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write(content)
    return path


def run_validation_study(
    *,
    n_trials: int = 2000,
    output_csv: str | Path = "results/monte_carlo_crlb_summary.csv",
) -> list[dict[str, float | int | str | bool | None]]:
    """Run all 42 requested configurations and write their aggregate CSV."""

    arrays = comparison_arrays()
    records: list[dict[str, float | int | str | bool | None]] = []
    for array_index, array_name in enumerate(ARRAY_NAMES):
        positions = arrays[array_name]
        for direction_index, direction_degrees in enumerate(DIRECTIONS_DEG):
            phi, elevation = np.deg2rad(direction_degrees)
            direction_id = f"az{direction_degrees[0]:g}_el{direction_degrees[1]:g}"
            for sigma_index, sigma_tdoa_us in enumerate(SIGMA_TDOA_US):
                result = run_monte_carlo_wls(
                    positions,
                    phi,
                    elevation,
                    array_name=array_name,
                    noise_model="independent_tdoa",
                    sigma_tdoa=sigma_tdoa_us * 1e-6,
                    n_trials=n_trials,
                    seed=_stream_seed(array_index, direction_index, 0, sigma_index),
                )
                records.append(_record_with_context(result, direction_id))

            toa_result = run_monte_carlo_wls(
                positions,
                phi,
                elevation,
                array_name=array_name,
                noise_model="independent_toa",
                sigma_toa=SIGMA_TOA_US * 1e-6,
                n_trials=n_trials,
                seed=_stream_seed(array_index, direction_index, 1, 0),
            )
            records.append(_record_with_context(toa_result, direction_id))

    if len(records) != EXPECTED_CONFIGURATION_COUNT:
        raise AssertionError(
            f"expected {EXPECTED_CONFIGURATION_COUNT} configurations, got {len(records)}"
        )
    write_summary_csv(records, output_csv)
    return records
