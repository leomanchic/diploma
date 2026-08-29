"""Interactive 3D view of true motion and independent bearing estimates.

No estimated 3D source point is drawn: a single array supplies bearing only.
Estimated rays use true range solely as a visualization length, explicitly
labelled as such in the legend and widget header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from model.geometry import array_centroid, microphone_positions


def _direction(value: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


@dataclass(frozen=True)
class MovingSceneFrame:
    frame_index: int
    reception_time_s: float
    source_position_m: ArrayLike
    true_direction: ArrayLike
    reference_3_direction: ArrayLike
    all_6_direction: ArrayLike
    srp_direction: ArrayLike
    speed_mps: float
    distance_m: float
    snr_db: float

    def __post_init__(self) -> None:
        position = np.asarray(self.source_position_m, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("source_position_m must be a finite vector with shape (3,)")
        object.__setattr__(self, "source_position_m", position.copy())
        for name in (
            "true_direction",
            "reference_3_direction",
            "all_6_direction",
            "srp_direction",
        ):
            object.__setattr__(self, name, _direction(getattr(self, name), name))
        for name in ("reception_time_s", "speed_mps", "distance_m", "snr_db"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if float(self.distance_m) <= 0.0 or float(self.speed_mps) < 0.0:
            raise ValueError("distance must be positive and speed non-negative")

    def angular_errors_deg(self) -> dict[str, float]:
        return {
            "GCC ref-3": float(
                np.rad2deg(
                    np.arccos(np.clip(self.true_direction @ self.reference_3_direction, -1, 1))
                )
            ),
            "GCC all-6": float(
                np.rad2deg(np.arccos(np.clip(self.true_direction @ self.all_6_direction, -1, 1)))
            ),
            "SRP": float(
                np.rad2deg(np.arccos(np.clip(self.true_direction @ self.srp_direction, -1, 1)))
            ),
        }


def _equal_axes(ax: Any, points: NDArray[np.float64]) -> None:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.55, 0.1)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_moving_scene_frame(
    ax: Any,
    positions: ArrayLike,
    trajectory_points_m: ArrayLike,
    frame: MovingSceneFrame,
    *,
    view_elevation_deg: float = 24.0,
    view_azimuth_deg: float = -55.0,
) -> Any:
    """Draw one frame with bearing rays, never estimated source points."""

    microphones = microphone_positions(positions)
    path = np.asarray(trajectory_points_m, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or not np.all(np.isfinite(path)):
        raise ValueError("trajectory_points_m must have shape (K, 3) and be finite")
    centroid = array_centroid(microphones)
    ray_length = float(frame.distance_m)
    ax.cla()
    ax.scatter(*microphones.T, color="black", marker="^", s=45, label="microphones")
    ax.plot(*path.T, color="0.55", linewidth=1.5, label="true trajectory")
    ax.scatter(*frame.source_position_m, color="tab:red", s=55, label="true source")
    rays = (
        (frame.true_direction, "true DOA", "tab:red", "-"),
        (frame.reference_3_direction, "GCC ref-3 bearing ray", "tab:blue", "--"),
        (frame.all_6_direction, "GCC all-6 bearing ray", "tab:green", "--"),
        (frame.srp_direction, "SRP bearing ray", "tab:purple", ":"),
    )
    endpoints = [centroid]
    for direction, label, color, style in rays:
        endpoint = centroid + ray_length * direction
        endpoints.append(endpoint)
        ax.plot(
            [centroid[0], endpoint[0]],
            [centroid[1], endpoint[1]],
            [centroid[2], endpoint[2]],
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=label,
        )
    errors = frame.angular_errors_deg()
    ax.set_title(
        f"frame {frame.frame_index} | v={frame.speed_mps:.1f} m/s | "
        f"R={frame.distance_m:.1f} m | SNR={frame.snr_db:.1f} dB\n"
        f"errors: ref-3 {errors['GCC ref-3']:.2f}°, "
        f"all-6 {errors['GCC all-6']:.2f}°, SRP {errors['SRP']:.2f}°"
    )
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_zlabel("z, m")
    ax.view_init(elev=float(view_elevation_deg), azim=float(view_azimuth_deg))
    _equal_axes(ax, np.vstack((microphones, path, np.asarray(endpoints))))
    ax.legend(loc="upper left", fontsize=8)
    return ax


def interactive_moving_scene(
    positions: ArrayLike,
    trajectory_points_m: ArrayLike,
    frames: list[MovingSceneFrame],
) -> Any:
    """Return Play/Pause, frame and camera controls for a notebook."""

    if not frames:
        raise ValueError("frames must not be empty")
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    play = widgets.Play(value=0, min=0, max=len(frames) - 1, step=1, interval=250)
    frame_slider = widgets.IntSlider(
        value=0, min=0, max=len(frames) - 1, step=1, description="Frame"
    )
    widgets.jslink((play, "value"), (frame_slider, "value"))
    azimuth = widgets.FloatSlider(value=-55, min=-180, max=180, step=5, description="View az")
    elevation = widgets.FloatSlider(value=24, min=-10, max=90, step=2, description="View el")
    output = widgets.Output()

    def redraw(*_: object) -> None:
        with output:
            output.clear_output(wait=True)
            figure = plt.figure(figsize=(8.5, 6.5))
            axis = figure.add_subplot(111, projection="3d")
            plot_moving_scene_frame(
                axis,
                positions,
                trajectory_points_m,
                frames[frame_slider.value],
                view_azimuth_deg=azimuth.value,
                view_elevation_deg=elevation.value,
            )
            plt.show()

    for control in (frame_slider, azimuth, elevation):
        control.observe(redraw, names="value")
    redraw()
    note = widgets.HTML(
        "<b>Bearing only.</b> Ray length uses true range for visualization only; "
        "no estimated 3D position or tracking filter is shown."
    )
    panel = widgets.VBox(
        [note, widgets.HBox([play, frame_slider]), widgets.HBox([azimuth, elevation]), output]
    )
    display(panel)
    return panel


__all__ = ["MovingSceneFrame", "interactive_moving_scene", "plot_moving_scene_frame"]
