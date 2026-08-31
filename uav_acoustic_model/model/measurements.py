"""Truth-free online measurement contracts for multi-station fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
import numpy as np
from numpy.typing import ArrayLike, NDArray


QualityValue = float | int | bool | str | None


def _array(value: ArrayLike, shape: tuple[int, ...], *, name: str) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    result.setflags(write=False)
    return result


def _validate_valid_payload(
    direction: NDArray[np.float64],
    covariance: NDArray[np.float64],
    bias: NDArray[np.float64],
) -> None:
    if not np.all(np.isfinite(direction)):
        raise ValueError("valid direction_local must be finite")
    norm = float(np.linalg.norm(direction))
    if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("valid direction_local must be a unit vector")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("valid covariance_tangent_rad2 must be finite")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-15):
        raise ValueError("covariance_tangent_rad2 must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -256.0 * np.finfo(float).eps * scale:
        raise ValueError("covariance_tangent_rad2 must be positive semidefinite")
    if not np.all(np.isfinite(bias)):
        raise ValueError("valid calibration_bias_tangent_rad must be finite")


@dataclass(frozen=True, slots=True)
class BearingMeasurement:
    """One calibrated local bearing available to an online fusion layer.

    The structure intentionally has no truth direction/position, angular
    error, future estimate, or true emission time.  Tangent quantities use
    angular-arc radians.  Bias is applied through the spherical residual
    model, never through coordinate-wise angle subtraction.
    """

    station_id: str
    sequence_id: str
    frame_index: int
    reception_center_timestamp_s: float
    available_timestamp_s: float
    direction_local: NDArray[np.float64]
    covariance_tangent_rad2: NDArray[np.float64]
    calibration_bias_tangent_rad: NDArray[np.float64]
    estimator_variant: str
    quality_metadata: Mapping[str, QualityValue] = field(default_factory=dict)
    valid: bool = True
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        station_id = str(self.station_id)
        sequence_id = str(self.sequence_id)
        estimator_variant = str(self.estimator_variant)
        if not station_id or not sequence_id or not estimator_variant:
            raise ValueError("station_id, sequence_id, and estimator_variant are required")
        frame_index = int(self.frame_index)
        if frame_index < 0 or frame_index != self.frame_index:
            raise ValueError("frame_index must be a non-negative integer")
        reception = float(self.reception_center_timestamp_s)
        available = float(self.available_timestamp_s)
        if not np.isfinite(reception) or not np.isfinite(available):
            raise ValueError("timestamps must be finite seconds")
        if available < reception:
            raise ValueError("available_timestamp_s cannot precede reception timestamp")
        direction = _array(self.direction_local, (3,), name="direction_local")
        covariance = _array(
            self.covariance_tangent_rad2, (2, 2), name="covariance_tangent_rad2"
        )
        bias = _array(
            self.calibration_bias_tangent_rad,
            (2,),
            name="calibration_bias_tangent_rad",
        )
        valid = bool(self.valid)
        invalid_reason = None if self.invalid_reason is None else str(self.invalid_reason)
        if valid:
            _validate_valid_payload(direction, covariance, bias)
            if invalid_reason:
                raise ValueError("a valid measurement cannot have invalid_reason")
        else:
            if not invalid_reason:
                raise ValueError("an invalid measurement must provide invalid_reason")
            arrays_are_finite = (
                np.all(np.isfinite(direction))
                and np.all(np.isfinite(covariance))
                and np.all(np.isfinite(bias))
            )
            arrays_are_nan = (
                np.all(np.isnan(direction))
                and np.all(np.isnan(covariance))
                and np.all(np.isnan(bias))
            )
            if arrays_are_finite:
                _validate_valid_payload(direction, covariance, bias)
            elif not arrays_are_nan:
                raise ValueError("invalid measurement payload must be usable or all-NaN")
        quality: dict[str, QualityValue] = {}
        for key, value in dict(self.quality_metadata).items():
            if not isinstance(key, str) or not key:
                raise ValueError("quality metadata keys must be non-empty strings")
            if not isinstance(value, (float, int, bool, str, type(None))):
                raise ValueError("quality metadata values must be scalar observables")
            if isinstance(value, float) and not np.isfinite(value):
                raise ValueError("finite quality metadata is required")
            quality[key] = value
        object.__setattr__(self, "station_id", station_id)
        object.__setattr__(self, "sequence_id", sequence_id)
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "reception_center_timestamp_s", reception)
        object.__setattr__(self, "available_timestamp_s", available)
        object.__setattr__(self, "direction_local", direction)
        object.__setattr__(self, "covariance_tangent_rad2", covariance)
        object.__setattr__(self, "calibration_bias_tangent_rad", bias)
        object.__setattr__(self, "estimator_variant", estimator_variant)
        object.__setattr__(self, "quality_metadata", MappingProxyType(quality))
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "invalid_reason", invalid_reason)

    @classmethod
    def invalid(
        cls,
        *,
        station_id: str,
        sequence_id: str,
        frame_index: int,
        reception_center_timestamp_s: float,
        available_timestamp_s: float,
        estimator_variant: str,
        invalid_reason: str,
        quality_metadata: Mapping[str, QualityValue] | None = None,
    ) -> "BearingMeasurement":
        """Construct an invalid record without a fictitious bearing/covariance."""

        return cls(
            station_id=station_id,
            sequence_id=sequence_id,
            frame_index=frame_index,
            reception_center_timestamp_s=reception_center_timestamp_s,
            available_timestamp_s=available_timestamp_s,
            direction_local=np.full(3, np.nan),
            covariance_tangent_rad2=np.full((2, 2), np.nan),
            calibration_bias_tangent_rad=np.full(2, np.nan),
            estimator_variant=estimator_variant,
            quality_metadata={} if quality_metadata is None else quality_metadata,
            valid=False,
            invalid_reason=invalid_reason,
        )


__all__ = ["BearingMeasurement", "QualityValue"]
