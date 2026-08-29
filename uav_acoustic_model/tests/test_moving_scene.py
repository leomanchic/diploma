import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.geometry import comparison_arrays, direction_vector
from visualization.moving_scene import MovingSceneFrame, plot_moving_scene_frame


def test_scene_draws_bearing_rays_without_estimated_position_markers():
    positions = comparison_arrays()["tetrahedral"]
    truth = direction_vector(np.deg2rad(45), np.deg2rad(30))
    frame = MovingSceneFrame(
        frame_index=3,
        reception_time_s=0.1,
        source_position_m=25 * truth,
        true_direction=truth,
        reference_3_direction=direction_vector(np.deg2rad(46), np.deg2rad(30)),
        all_6_direction=direction_vector(np.deg2rad(45), np.deg2rad(31)),
        srp_direction=direction_vector(np.deg2rad(44.5), np.deg2rad(30.2)),
        speed_mps=20,
        distance_m=25,
        snr_db=10,
    )
    figure = plt.figure()
    axis = figure.add_subplot(111, projection="3d")
    path = np.vstack((24 * truth, 25 * truth, 26 * truth))
    plot_moving_scene_frame(axis, positions, path, frame)
    labels = {line.get_label() for line in axis.lines}
    assert "GCC ref-3 bearing ray" in labels
    assert "GCC all-6 bearing ray" in labels
    assert "SRP bearing ray" in labels
    # Two scatter collections only: microphones and the one true source point.
    assert len(axis.collections) == 2
    assert "errors:" in axis.get_title()
    plt.close(figure)


def test_scene_frame_normalizes_directions_and_reports_finite_errors():
    frame = MovingSceneFrame(
        0,
        0.0,
        [1, 2, 3],
        [2, 0, 0],
        [1, 0.1, 0],
        [1, 0, 0.1],
        [1, -0.1, 0],
        5,
        10,
        0,
    )
    assert np.isclose(np.linalg.norm(frame.true_direction), 1.0)
    assert all(np.isfinite(value) for value in frame.angular_errors_deg().values())
