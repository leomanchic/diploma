"""Static multi-station ENU scene visualization.

Truth is accepted only in explicit validation mode.  Bearings are rays; the
plot never invents a per-station range measurement.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from estimators.bearing_triangulation import TriangulationResult
from model.measurements import BearingMeasurement
from model.station import StationPose


CHI_SQUARE_3_P95 = 7.814727903251179


def _equal_axes(ax, points: np.ndarray) -> None:
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _covariance_ellipsoid(ax, center: np.ndarray, covariance: np.ndarray) -> None:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    if np.min(eigenvalues) < 0.0 or not np.all(np.isfinite(eigenvalues)):
        return
    radii = np.sqrt(CHI_SQUARE_3_P95 * eigenvalues)
    azimuth = np.linspace(0.0, 2.0 * np.pi, 32)
    polar = np.linspace(0.0, np.pi, 18)
    unit = np.stack(
        np.meshgrid(azimuth, polar, indexing="xy"), axis=-1
    )
    sphere = np.stack(
        (
            np.cos(unit[..., 0]) * np.sin(unit[..., 1]),
            np.sin(unit[..., 0]) * np.sin(unit[..., 1]),
            np.cos(unit[..., 1]),
        ),
        axis=-1,
    )
    transformed = sphere @ (eigenvectors @ np.diag(radii)).T + center
    ax.plot_wireframe(
        transformed[..., 0],
        transformed[..., 1],
        transformed[..., 2],
        color="tab:purple",
        alpha=0.35,
        linewidth=0.5,
        rstride=2,
        cstride=3,
    )


def plot_multistation_scene(
    stations: Sequence[StationPose],
    measurements: Sequence[BearingMeasurement],
    result: TriangulationResult,
    *,
    true_position_world_m: ArrayLike | None = None,
    validation_mode: bool = False,
    ray_length_m: float | None = None,
    ax=None,
):
    """Plot station axes, bearing rays, residuals, estimate, and covariance.

    ``true_position_world_m`` is forbidden unless ``validation_mode=True``.
    It is used only for an offline marker, never to set ray lengths or to
    recompute the estimate.
    """

    import matplotlib.pyplot as plt

    if true_position_world_m is not None and not validation_mode:
        raise ValueError("true position is allowed only in validation_mode")
    pose_by_id = {station.station_id: station for station in stations}
    if len(pose_by_id) != len(stations):
        raise ValueError("station ids must be unique")
    if ax is None:
        figure = plt.figure(figsize=(9, 7))
        ax = figure.add_subplot(111, projection="3d")
    else:
        figure = ax.figure
    station_positions = np.asarray([station.position_world_m for station in stations])
    if ray_length_m is None:
        if result.valid:
            ray_length = 1.25 * float(
                np.max(np.linalg.norm(result.position_world_m - station_positions, axis=1))
            )
        else:
            ray_length = 3.0 * max(
                float(np.max(np.linalg.norm(station_positions[:, None] - station_positions[None, :], axis=-1))),
                1.0,
            )
    else:
        ray_length = float(ray_length_m)
        if not np.isfinite(ray_length) or ray_length <= 0.0:
            raise ValueError("ray_length_m must be finite and positive")

    all_points = [station_positions]
    axis_colors = ("tab:red", "tab:green", "tab:blue")
    for station in stations:
        position = station.position_world_m
        ax.scatter(*position, color="black", marker="^", s=55)
        ax.text(*position, station.station_id)
        axis_scale = max(ray_length * 0.05, 0.5)
        for axis_index, color in enumerate(axis_colors):
            endpoint = position + axis_scale * station.rotation_local_to_world[:, axis_index]
            ax.plot(
                [position[0], endpoint[0]],
                [position[1], endpoint[1]],
                [position[2], endpoint[2]],
                color=color,
                linewidth=1.5,
            )
    for measurement in measurements:
        if not measurement.valid:
            continue
        station = pose_by_id[measurement.station_id]
        direction = station.local_to_world_direction(measurement.direction_local)
        direction /= np.linalg.norm(direction)
        endpoint = station.position_world_m + ray_length * direction
        all_points.append(np.vstack((station.position_world_m, endpoint)))
        ax.plot(
            [station.position_world_m[0], endpoint[0]],
            [station.position_world_m[1], endpoint[1]],
            [station.position_world_m[2], endpoint[2]],
            color="tab:orange",
            alpha=0.8,
            label="bearing ray" if measurement is measurements[0] else None,
        )
        if result.valid:
            signed_range = float(direction @ (result.position_world_m - station.position_world_m))
            closest = station.position_world_m + signed_range * direction
            ax.scatter(*closest, color="tab:orange", s=18)
            ax.plot(
                [closest[0], result.position_world_m[0]],
                [closest[1], result.position_world_m[1]],
                [closest[2], result.position_world_m[2]],
                color="tab:gray",
                linestyle=":",
                linewidth=1.0,
            )
    if result.valid:
        ax.scatter(
            *result.position_world_m,
            color="tab:purple",
            marker="x",
            s=90,
            linewidth=2.5,
            label="estimated 3D position",
        )
        all_points.append(result.position_world_m[None, :])
        if np.all(np.isfinite(result.covariance_position_m2)):
            _covariance_ellipsoid(
                ax, result.position_world_m, result.covariance_position_m2
            )
    if true_position_world_m is not None:
        truth = np.asarray(true_position_world_m, dtype=float)
        if truth.shape != (3,) or not np.all(np.isfinite(truth)):
            raise ValueError("true_position_world_m must be a finite three-vector")
        ax.scatter(*truth, color="tab:cyan", marker="*", s=110, label="true position (validation only)")
        all_points.append(truth[None, :])
    _equal_axes(ax, np.vstack(all_points))
    ax.set_xlabel("East x [m]")
    ax.set_ylabel("North y [m]")
    ax.set_zlabel("Up z [m]")
    ax.set_title(
        "Static multi-station bearing triangulation"
        + ("" if result.valid else f" — invalid: {result.failure_reason}")
    )
    ax.legend(loc="best")
    return figure, ax


__all__ = ["plot_multistation_scene"]
