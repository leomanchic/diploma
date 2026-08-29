"""Reproducible numerical validation helpers."""

from .far_field import (
    ContinuousFarFieldRefinement,
    FarFieldBoundaryResult,
    FarFieldErrorResult,
    continuous_refine_far_field_error,
    direction_grid,
    far_field_error,
    minimum_far_field_distance,
)
from .monte_carlo import MonteCarloResult, run_monte_carlo_wls
from .gcc_study import benchmark_fractional_delay_methods, run_gcc_phat_validation_study
from .gcc_monte_carlo import run_gcc_phat_monte_carlo
from .gcc_statistical import (
    GCCStatisticalConfig,
    run_complete_gcc_statistical_validation,
    run_gcc_statistical_configuration,
    run_gcc_statistical_validation,
)
from .srp_statistical import (
    default_srp_statistical_configurations,
    run_srp_paired_configuration,
    run_srp_statistical_validation,
)
from .propagation_study import (
    run_far_field_boundary_study,
    run_fractional_delay_accuracy_study,
)
from .study import run_validation_study
from .moving_source_study import (
    MovingStudyConfig,
    default_moving_study_configurations,
    run_moving_configuration,
    run_moving_source_study,
)
from .sequential_doa_study import (
    SequentialStudyConfig,
    default_sequential_configurations,
    run_sequential_configuration,
    run_sequential_doa_study,
)

__all__ = [
    "ContinuousFarFieldRefinement",
    "FarFieldBoundaryResult",
    "FarFieldErrorResult",
    "MonteCarloResult",
    "GCCStatisticalConfig",
    "benchmark_fractional_delay_methods",
    "continuous_refine_far_field_error",
    "direction_grid",
    "far_field_error",
    "minimum_far_field_distance",
    "run_monte_carlo_wls",
    "run_far_field_boundary_study",
    "run_fractional_delay_accuracy_study",
    "run_gcc_phat_validation_study",
    "run_gcc_phat_monte_carlo",
    "run_complete_gcc_statistical_validation",
    "run_gcc_statistical_configuration",
    "run_gcc_statistical_validation",
    "default_srp_statistical_configurations",
    "run_srp_paired_configuration",
    "run_srp_statistical_validation",
    "run_validation_study",
    "MovingStudyConfig",
    "default_moving_study_configurations",
    "run_moving_configuration",
    "run_moving_source_study",
    "SequentialStudyConfig",
    "default_sequential_configurations",
    "run_sequential_configuration",
    "run_sequential_doa_study",
]
