"""Tests for the immutable constant-velocity state parameterization."""

import numpy as np
import pytest

from model.dynamic_state import (
    ConstantVelocityState,
    constant_velocity_transition_jacobian,
    rebase_constant_velocity_state,
)


def test_state_is_finite_read_only_and_uses_si_transition():
    state = ConstantVelocityState([1.0, 2.0, 3.0], [4.0, -2.0, 0.5], 7.0)
    np.testing.assert_allclose(state.position_at(9.0), [9.0, -2.0, 4.0])
    assert not state.position_at_reference_world_m.flags.writeable
    assert not state.velocity_world_mps.flags.writeable
    assert not state.vector.flags.writeable


@pytest.mark.parametrize(
    "position,velocity,reference",
    [([np.nan, 0, 0], [0, 0, 0], 0), ([0, 0, 0], [np.inf, 0, 0], 0), ([0, 0, 0], [0, 0, 0], np.nan)],
)
def test_state_rejects_nonfinite_values(position, velocity, reference):
    with pytest.raises(ValueError, match="finite"):
        ConstantVelocityState(position, velocity, reference)


def test_rebase_preserves_the_physical_trajectory():
    state = ConstantVelocityState([2.0, -3.0, 10.0], [7.0, 1.5, -0.2], -1.0)
    rebased = rebase_constant_velocity_state(state, 4.25)
    for time in (-3.0, 0.0, 4.25, 17.0):
        np.testing.assert_allclose(rebased.position_at(time), state.position_at(time))
    np.testing.assert_array_equal(rebased.velocity_world_mps, state.velocity_world_mps)


def test_transition_semigroup_and_inverse():
    first = constant_velocity_transition_jacobian(1.25)
    second = constant_velocity_transition_jacobian(-0.4)
    combined = constant_velocity_transition_jacobian(0.85)
    np.testing.assert_allclose(second @ first, combined, atol=2e-16)
    np.testing.assert_allclose(
        constant_velocity_transition_jacobian(-1.25) @ first,
        np.eye(6),
        atol=2e-16,
    )


def test_reference_time_invariance_matches_transition_vector():
    state = ConstantVelocityState([2.0, 3.0, 4.0], [-1.0, 2.0, 0.5], 10.0)
    dt = 3.5
    rebased = rebase_constant_velocity_state(state, state.reference_time_s + dt)
    np.testing.assert_allclose(
        rebased.vector,
        constant_velocity_transition_jacobian(dt) @ state.vector,
    )
