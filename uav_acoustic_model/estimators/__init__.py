"""Direction-of-arrival estimators."""

from .bearing_triangulation import (
    ClosestRaysResult,
    TriangulationResult,
    bearing_residual,
    bearing_residual_jacobian,
    closest_rays_triangulation,
    numerical_bearing_residual_jacobian,
    triangulate_bearings_spherical_wls,
)
from .cycle_projection import (
    CycleProjectionResult,
    DisconnectedPairGraphError,
    project_tdoa_cycles,
)
from .gcc_phat import (
    DirectGCCPHATResult,
    GCCPHATResult,
    direct_gcc_phat_correlation,
    estimate_tdoas_gcc_phat,
    gcc_phat,
)
from .srp_phat import (
    SRPPHATResult,
    SRPScoreGridResult,
    direct_srp_phat_scores,
    srp_phat,
    vectorized_srp_phat_scores,
)
from .wls_doa import (
    DOAEstimate,
    UnobservableGeometryError,
    estimate_doa_spherical_wls,
    estimate_doa_wls,
)

__all__ = [
    "DOAEstimate",
    "ClosestRaysResult",
    "CycleProjectionResult",
    "DirectGCCPHATResult",
    "GCCPHATResult",
    "SRPPHATResult",
    "SRPScoreGridResult",
    "TriangulationResult",
    "DisconnectedPairGraphError",
    "UnobservableGeometryError",
    "direct_gcc_phat_correlation",
    "direct_srp_phat_scores",
    "estimate_doa_wls",
    "estimate_doa_spherical_wls",
    "estimate_tdoas_gcc_phat",
    "bearing_residual",
    "bearing_residual_jacobian",
    "closest_rays_triangulation",
    "gcc_phat",
    "project_tdoa_cycles",
    "numerical_bearing_residual_jacobian",
    "srp_phat",
    "vectorized_srp_phat_scores",
    "triangulate_bearings_spherical_wls",
]
