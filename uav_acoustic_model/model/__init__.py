"""Core geometry, TDOA, Jacobian, and statistical models."""

from .bearing_statistics import (
    sphere_log_map,
    tangent_residual,
    tangent_residual_jacobian_wrt_true_direction,
)
from .dynamic_state import (
    ConstantVelocityState,
    constant_velocity_transition_jacobian,
    rebase_constant_velocity_state,
)
from .geometry import DEFAULT_SOUND_SPEED, direction_vector
from .measurements import BearingMeasurement
from .retarded_bearing import (
    DynamicObservabilityResult,
    RetardedBearingPrediction,
    available_bearing_measurements,
    emission_time_jacobian_wrt_state,
    predict_retarded_bearing,
    predict_retarded_bearing_measurement,
    predicted_local_direction_jacobian,
    retarded_bearing_residual,
    retarded_bearing_residual_jacobian,
    stack_retarded_bearing_observability,
)
from .station import StationPose

__all__ = [
    "BearingMeasurement",
    "ConstantVelocityState",
    "DEFAULT_SOUND_SPEED",
    "DynamicObservabilityResult",
    "RetardedBearingPrediction",
    "StationPose",
    "available_bearing_measurements",
    "constant_velocity_transition_jacobian",
    "direction_vector",
    "emission_time_jacobian_wrt_state",
    "predict_retarded_bearing",
    "predict_retarded_bearing_measurement",
    "predicted_local_direction_jacobian",
    "rebase_constant_velocity_state",
    "retarded_bearing_residual",
    "retarded_bearing_residual_jacobian",
    "sphere_log_map",
    "tangent_residual",
    "tangent_residual_jacobian_wrt_true_direction",
    "stack_retarded_bearing_observability",
]
