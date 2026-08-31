import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from estimators.bearing_triangulation import triangulate_bearings_spherical_wls
from model.geometry import tetrahedral_array
from model.measurements import BearingMeasurement
from model.station import StationPose
from visualization.multistation_scene import plot_multistation_scene


def _scene():
    positions = [np.asarray([0.0, 0.0, 0.0]), np.asarray([20.0, 0.0, 0.0]), np.asarray([5.0, 17.0, 0.0])]
    target = np.asarray([8.0, 7.0, 12.0])
    stations = []
    measurements = []
    for index, position in enumerate(positions):
        station = StationPose(f"s{index}", position, np.eye(3), tetrahedral_array())
        direction = target - position
        direction /= np.linalg.norm(direction)
        stations.append(station)
        measurements.append(
            BearingMeasurement(
                station.station_id, "v", 0, 0.0, 0.01, direction,
                np.diag(np.deg2rad([0.2, 0.3]) ** 2), np.zeros(2), "ideal"
            )
        )
    result = triangulate_bearings_spherical_wls(stations, measurements)
    return stations, measurements, target, result


def test_scene_plots_bearing_rays_position_and_covariance_only_in_validation_mode():
    stations, measurements, target, result = _scene()
    with pytest.raises(ValueError, match="validation_mode"):
        plot_multistation_scene(
            stations, measurements, result, true_position_world_m=target
        )
    figure, axis = plot_multistation_scene(
        stations,
        measurements,
        result,
        true_position_world_m=target,
        validation_mode=True,
    )
    assert figure is axis.figure
    assert len(axis.lines) >= 3
    assert "bearing triangulation" in axis.get_title()

