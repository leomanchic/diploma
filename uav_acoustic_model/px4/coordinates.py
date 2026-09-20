"""Explicit world ENU/NED and body FLU/FRD coordinate transforms."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

# V_ned = WORLD_NED_FROM_ENU @ V_enu; V_frd = BODY_FRD_FROM_FLU @ V_flu.
WORLD_NED_FROM_ENU = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
BODY_FRD_FROM_FLU = np.diag([1., -1., -1.])


def enu_to_ned(vector: ArrayLike) -> NDArray[np.float64]:
    return WORLD_NED_FROM_ENU @ np.asarray(vector, dtype=float)


def ned_to_enu(vector: ArrayLike) -> NDArray[np.float64]:
    return WORLD_NED_FROM_ENU @ np.asarray(vector, dtype=float)


def flu_to_frd(vector: ArrayLike) -> NDArray[np.float64]:
    return BODY_FRD_FROM_FLU @ np.asarray(vector, dtype=float)


def gazebo_wxyz_to_px4_ned_frd_matrix(quaternion_wxyz: ArrayLike) -> NDArray[np.float64]:
    """Map Gazebo's link-FLU→world-ENU quaternion into body-FRD→world-NED."""
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or not math.isclose(
        float(q @ q), 1.0, rel_tol=0, abs_tol=1e-5
    ):
        raise ValueError("Gazebo quaternion must be unit w,x,y,z")
    w, x, y, z = q / np.linalg.norm(q)
    enu_from_flu = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return WORLD_NED_FROM_ENU @ enu_from_flu @ BODY_FRD_FROM_FLU
