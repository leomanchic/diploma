import numpy as np
import pytest

from model.bearing_statistics import (
    AntipodalDirectionError,
    calibrate_bearing_covariance,
    normalized_innovation_squared,
    sphere_log_map,
    tangent_residual,
)
from model.geometry import direction_vector


def test_residual_is_zero_for_identical_direction():
    truth = direction_vector(0.7, 0.3)
    assert np.array_equal(tangent_residual(truth, truth), np.zeros(2))


def test_azimuth_wrap_359_to_zero_is_small_and_signed():
    truth = direction_vector(np.deg2rad(359.0), np.deg2rad(20.0))
    estimate = direction_vector(0.0, np.deg2rad(20.0))
    residual = tangent_residual(truth, estimate)
    assert residual[0] == pytest.approx(np.cos(np.deg2rad(20.0)) * np.deg2rad(1.0), rel=2e-5)
    # Equal-elevation endpoints do not define a constant-elevation geodesic;
    # the elevation component is therefore second order, not identically zero.
    assert abs(residual[1]) < np.deg2rad(1.0) ** 2


def test_local_residual_matches_spherical_angle_coordinates():
    phi, elevation = 1.2, 0.45
    delta_phi, delta_elevation = 2e-6, -3e-6
    residual = tangent_residual(
        direction_vector(phi, elevation),
        direction_vector(phi + delta_phi, elevation + delta_elevation),
    )
    assert residual == pytest.approx(
        [np.cos(elevation) * delta_phi, delta_elevation], abs=6e-11
    )


def test_log_map_is_invariant_to_vector_normalization():
    truth = direction_vector(0.4, 0.2)
    estimate = direction_vector(0.45, 0.23)
    assert tangent_residual(7.0 * truth, 0.03 * estimate) == pytest.approx(
        tangent_residual(truth, estimate), abs=2e-15
    )


def test_log_map_is_finite_at_small_theta():
    truth = direction_vector(0.4, 0.2)
    estimate = direction_vector(0.4 + 1e-10, 0.2 - 2e-10)
    result = sphere_log_map(truth, estimate)
    assert np.all(np.isfinite(result))
    assert np.linalg.norm(result) < 1e-8


def test_nearly_antipodal_direction_is_explicitly_rejected():
    truth = direction_vector(0.4, 0.2)
    with pytest.raises(AntipodalDirectionError, match="not unique"):
        tangent_residual(truth, -truth)


def test_calibrated_covariance_is_symmetric_psd_and_nis_uses_pseudoinverse():
    residuals = np.asarray([[0.01, -0.02], [0.02, -0.01], [-0.01, 0.01], [0.0, 0.02]])
    calibrated = calibrate_bearing_covariance(residuals)
    assert calibrated.symmetric
    assert calibrated.positive_semidefinite
    assert np.all(calibrated.eigenvalues_rad2 >= -1e-18)
    nis = normalized_innovation_squared(residuals, calibrated.covariance_rad2)
    assert nis.shape == (4,)
    assert np.all(np.isfinite(nis))
    assert np.all(nis >= 0.0)


def test_invalid_measurements_cannot_be_silently_calibrated():
    with pytest.raises(ValueError, match="finite"):
        calibrate_bearing_covariance([[0.0, 0.0], [np.nan, np.nan]])
