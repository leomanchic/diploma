"""Weighted projection of redundant TDOAs onto the cycle-consistent subspace."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from model.geometry import Pair, incidence_matrix, validate_pairs


class DisconnectedPairGraphError(ValueError):
    """Raised when the selected microphone-pair graph is disconnected."""


@dataclass(frozen=True)
class CycleProjectionResult:
    """Weighted least-squares arrival times and consistent TDOAs."""

    arrival_times: NDArray[np.float64]
    consistent_tdoa: NDArray[np.float64]
    residual: NDArray[np.float64]
    weighted_cost: float
    cycle_residual_before: float
    cycle_residual_after: float
    incidence_rank: int
    pair_count: int
    microphone_count: int
    pairs: tuple[Pair, ...]


def _weight_matrix(weights: ArrayLike | None, pair_count: int) -> NDArray[np.float64]:
    if weights is None:
        return np.eye(pair_count)
    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim == 1:
        if matrix.shape != (pair_count,):
            raise ValueError("weight vector must have one value per pair")
        if not np.all(np.isfinite(matrix)) or np.any(matrix <= 0.0):
            raise ValueError("pair weights must be finite and positive")
        return np.diag(matrix)
    if matrix.shape != (pair_count, pair_count):
        raise ValueError("weight matrix shape must match the pair count")
    if not np.all(np.isfinite(matrix)) or not np.allclose(
        matrix, matrix.T, rtol=1e-11, atol=1e-14
    ):
        raise ValueError("weight matrix must be finite and symmetric")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-12 * scale:
        raise ValueError("weight matrix must be positive semidefinite")
    if not np.any(eigenvalues > 1e-12 * scale):
        raise ValueError("weight matrix must contain a positive-weight subspace")
    return matrix


def _square_root_weight(weight_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh(weight_matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    positive = eigenvalues > 1e-12 * scale
    return np.sqrt(eigenvalues[positive])[:, None] * eigenvectors[:, positive].T


def _cycle_coordinates(incidence: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    left_vectors, singular_values, _ = np.linalg.svd(incidence, full_matrices=True)
    tolerance = max(incidence.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    cycle_basis = left_vectors[:, rank:]
    if cycle_basis.size == 0:
        return 0.0
    return float(np.linalg.norm(cycle_basis.T @ values))


def project_tdoa_cycles(
    measured_tdoa: ArrayLike,
    pairs: Iterable[Sequence[int]],
    microphone_count: int,
    *,
    weights: ArrayLike | None = None,
) -> CycleProjectionResult:
    r"""Solve ``argmin_t (tau-Bt)^T W (tau-Bt)`` and return ``B t_hat``.

    The unidentifiable common arrival time is fixed by the zero-mean gauge.
    The pair graph must be connected even when the pair set contains no cycles.
    ``weights`` may be positive diagonal weights or a symmetric PSD matrix.
    """

    count = int(microphone_count)
    if count < 2:
        raise ValueError("microphone_count must be at least 2")
    checked_pairs = validate_pairs(pairs, count)
    observations = np.asarray(measured_tdoa, dtype=float)
    if observations.shape != (len(checked_pairs),) or not np.all(np.isfinite(observations)):
        raise ValueError("measured_tdoa must be finite and match the pair count")
    incidence = incidence_matrix(checked_pairs, count)
    incidence_rank = int(np.linalg.matrix_rank(incidence))
    if incidence_rank != count - 1:
        raise DisconnectedPairGraphError(
            f"pair graph is disconnected: incidence rank {incidence_rank}, expected {count - 1}"
        )
    weight_matrix = _weight_matrix(weights, len(checked_pairs))
    root_weight = _square_root_weight(weight_matrix)
    weighted_incidence = root_weight @ incidence
    weighted_observations = root_weight @ observations
    if int(np.linalg.matrix_rank(weighted_incidence)) != count - 1:
        raise ValueError("positive-weight subspace does not identify the connected pair graph")
    arrival_times, _, _, _ = np.linalg.lstsq(
        weighted_incidence, weighted_observations, rcond=1e-12
    )
    arrival_times = arrival_times - np.mean(arrival_times)
    consistent = incidence @ arrival_times
    residual = observations - consistent
    weighted_cost = float(residual @ weight_matrix @ residual)
    return CycleProjectionResult(
        arrival_times=np.asarray(arrival_times, dtype=float),
        consistent_tdoa=np.asarray(consistent, dtype=float),
        residual=np.asarray(residual, dtype=float),
        weighted_cost=weighted_cost,
        cycle_residual_before=_cycle_coordinates(incidence, observations),
        cycle_residual_after=_cycle_coordinates(incidence, consistent),
        incidence_rank=incidence_rank,
        pair_count=len(checked_pairs),
        microphone_count=count,
        pairs=checked_pairs,
    )


__all__ = [
    "CycleProjectionResult",
    "DisconnectedPairGraphError",
    "project_tdoa_cycles",
]
