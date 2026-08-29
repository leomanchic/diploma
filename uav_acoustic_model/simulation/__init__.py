"""Deterministic signal-generation and propagation utilities."""

from .fractional_delay import (
    DEFAULT_FIR_LENGTH,
    frequency_domain_delay,
    fractional_delay_valid_region,
    windowed_sinc_delay,
)
from .signals import (
    deterministic_bandlimited_signal,
    harmonic_stress_signal,
    random_bandlimited_signal,
)
from .propagation import PropagationResult, simulate_propagation
from .trajectory import (
    CircularTrajectory,
    ConstantVelocityTrajectory,
    PiecewiseLinearTrajectory,
    StationaryTrajectory,
    Trajectory,
)
from .moving_source import (
    MovingSourceResult,
    centroid_emission_time,
    constant_velocity_emission_time,
    retarded_time_doppler_factor,
    simulate_moving_source,
    solve_emission_time,
)
from .continuous_stream import (
    ContinuousStreamResult,
    OverlappingFrameBatch,
    extract_overlapping_frames,
    reception_time_grid,
    synthesize_continuous_stream,
)

__all__ = [
    "DEFAULT_FIR_LENGTH",
    "deterministic_bandlimited_signal",
    "harmonic_stress_signal",
    "PropagationResult",
    "simulate_propagation",
    "fractional_delay_valid_region",
    "frequency_domain_delay",
    "windowed_sinc_delay",
    "random_bandlimited_signal",
    "CircularTrajectory",
    "ConstantVelocityTrajectory",
    "PiecewiseLinearTrajectory",
    "StationaryTrajectory",
    "Trajectory",
    "MovingSourceResult",
    "centroid_emission_time",
    "constant_velocity_emission_time",
    "retarded_time_doppler_factor",
    "simulate_moving_source",
    "solve_emission_time",
    "ContinuousStreamResult",
    "OverlappingFrameBatch",
    "extract_overlapping_frames",
    "reception_time_grid",
    "synthesize_continuous_stream",
]
