# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Behavioral regression tests for deformable MPM collider coupling."""

import unittest

import numpy as np
import warp as wp
import warp.sparse as wps

import newton
from newton._src.solvers.implicit_mpm.rasterized_collisions import build_rigidity_operator
from newton.solvers import SolverImplicitMPM
from newton.solvers.experimental.coupled import CouplingInterface
from newton.tests.unittest_utils import add_function_test, get_test_devices

_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
_CONTACT_POINT = np.array([0.125, 0.25, 0.0], dtype=np.float32)


def _make_deformable_collider(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    flags = int(newton.ParticleFlags.ACTIVE | newton.ParticleFlags.PROXY)
    for vertex, mass in zip(_VERTICES, (1.0, 2.0, 4.0), strict=True):
        builder.add_particle(pos=vertex, vel=(0.0, 0.0, 0.0), mass=mass, radius=0.01, flags=flags)
    builder.add_particle(pos=(2.0, 2.0, 2.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.01)
    model = builder.finalize(device=device)

    config = SolverImplicitMPM.Config()
    config.voxel_size = 0.25
    config.grid_type = "fixed"
    config.grid_padding = 2
    solver = SolverImplicitMPM(model, config)
    mesh = wp.Mesh(
        points=wp.array(_VERTICES, dtype=wp.vec3, device=device),
        indices=wp.array([0, 1, 2], dtype=int, device=device),
        velocities=wp.zeros(3, dtype=wp.vec3, device=device),
    )
    solver.setup_collider(collider_meshes=[mesh], collider_particle_ids=[[0, 1, 2]])
    return solver, mesh


def _mock_single_contact_impulse(solver, contact_impulse, device):
    """Make the solver report one collider impulse at ``_CONTACT_POINT`` after every step."""
    contact_point = wp.array([_CONTACT_POINT], dtype=wp.vec3, device=device)
    impulses = wp.array([contact_impulse], dtype=wp.vec3, device=device)
    collider_ids = wp.array([0], dtype=int, device=device)
    solver.collect_collider_impulses = lambda state: (impulses, contact_point, collider_ids)


def _harvest_proxy_particle_forces(solver, state, dt, device):
    """Harvest the proxy particle forces of the three collider vertices over a coupled step ``dt``."""
    feedback = wp.zeros(3, dtype=wp.vec3, device=device)
    solver.coupling_harvest_proxy_particle_forces(
        wp.array([0, 1, 2, -1], dtype=int, device=device),
        feedback,
        particle_qd_before=state.particle_qd,
        state=state,
        state_out=state,
        contacts=None,
        dt=dt,
    )
    return feedback.numpy()


def _assert_feedback_matches_contact(forces, expected_force):
    np.testing.assert_allclose(forces.sum(axis=0), expected_force, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        np.cross(_VERTICES, forces).sum(axis=0), np.cross(_CONTACT_POINT, expected_force), rtol=0.0, atol=1.0e-6
    )


def test_proxy_feedback_preserves_contact_force_and_moment(test, device):
    """Apply proxy feedback at its physical contact point."""
    solver, _mesh = _make_deformable_collider(device)
    contact_impulse = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    _mock_single_contact_impulse(solver, contact_impulse, device)

    state = solver.model.state()
    dt = 2.0
    solver.coupling_notify_proxy_feedback(CouplingInterface.EndpointKind.PARTICLE, True)
    solver.coupling_notify_input_state_update(state, newton.StateFlags.NONE)
    solver.step(state, state, None, None, dt)
    forces = _harvest_proxy_particle_forces(solver, state, dt, device)

    _assert_feedback_matches_contact(forces, contact_impulse / dt)


def test_proxy_feedback_sums_substep_impulses(test, device):
    """Feed back the impulses of every sub-step taken since the last coupling notification."""
    solver, _mesh = _make_deformable_collider(device)
    contact_impulse = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    _mock_single_contact_impulse(solver, contact_impulse, device)

    state = solver.model.state()
    dt = 2.0
    solver.coupling_notify_proxy_feedback(CouplingInterface.EndpointKind.PARTICLE, True)
    solver.coupling_notify_input_state_update(state, newton.StateFlags.PARTICLE_Q | newton.StateFlags.PARTICLE_QD)
    solver.step(state, state, None, None, dt / 2.0)
    # Couplers may update input forces between sub-steps; that must not drop earlier impulses.
    solver.coupling_notify_input_state_update(state, newton.StateFlags.PARTICLE_F)
    solver.step(state, state, None, None, dt / 2.0)
    forces = _harvest_proxy_particle_forces(solver, state, dt, device)

    _assert_feedback_matches_contact(forces, 2.0 * contact_impulse / dt)

    # Each harvest consumes the accumulated impulses.
    solver.step(state, state, None, None, dt)
    forces = _harvest_proxy_particle_forces(solver, state, dt, device)

    _assert_feedback_matches_contact(forces, contact_impulse / dt)

    # An iteration restart discards the impulses of the abandoned solve.
    solver.step(state, state, None, None, dt)
    solver.coupling_notify_input_state_update(state, newton.StateFlags.PARTICLE_Q, iteration_restart=True)
    solver.step(state, state, None, None, dt)
    forces = _harvest_proxy_particle_forces(solver, state, dt, device)

    _assert_feedback_matches_contact(forces, contact_impulse / dt)


def test_proxy_feedback_without_notification_uses_last_step(test, device):
    """Convert only the most recent step's impulses until the coupler declares particle proxy feedback."""
    solver, _mesh = _make_deformable_collider(device)
    contact_impulse = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    _mock_single_contact_impulse(solver, contact_impulse, device)
    kind = CouplingInterface.EndpointKind.PARTICLE

    state = solver.model.state()
    dt = 2.0
    # Input-state notifications alone (as sent by every coupler) must not start the accumulation.
    solver.coupling_notify_input_state_update(state, newton.StateFlags.PARTICLE_Q)
    solver.step(state, state, None, None, dt / 2.0)
    solver.step(state, state, None, None, dt / 2.0)
    forces = _harvest_proxy_particle_forces(solver, state, dt, device)
    _assert_feedback_matches_contact(forces, contact_impulse / dt)

    solver.coupling_notify_proxy_feedback(kind, True)
    solver.step(state, state, None, None, dt / 2.0)
    solver.step(state, state, None, None, dt / 2.0)
    forces = _harvest_proxy_particle_forces(solver, state, dt, device)
    _assert_feedback_matches_contact(forces, 2.0 * contact_impulse / dt)

    solver.coupling_notify_proxy_feedback(kind, False)
    solver.step(state, state, None, None, dt / 2.0)
    solver.step(state, state, None, None, dt / 2.0)
    forces = _harvest_proxy_particle_forces(solver, state, dt, device)
    _assert_feedback_matches_contact(forces, contact_impulse / dt)


def test_proxy_feedback_storage_follows_notification(test, device):
    """Allocate the feedback accumulators only for the endpoint kinds the coupler declared."""
    solver, _mesh = _make_deformable_collider(device)
    test.assertIsNone(solver._coupling_particle_impulse)
    test.assertIsNone(solver._coupling_body_impulse)

    solver.coupling_notify_proxy_feedback(CouplingInterface.EndpointKind.PARTICLE, True)
    test.assertIsNotNone(solver._coupling_particle_impulse)
    test.assertEqual(solver._coupling_particle_impulse.shape[0], solver.model.particle_count)
    test.assertIsNone(solver._coupling_body_impulse)

    solver.coupling_notify_proxy_feedback(CouplingInterface.EndpointKind.BODY, True)
    test.assertIsNotNone(solver._coupling_body_impulse)
    test.assertEqual(solver._coupling_body_impulse.shape[0], solver.model.body_count)


def test_collider_contact_motion_and_moment(test, device):
    """Transfer vertex motion and contact impulse at the same surface point."""
    solver, _mesh = _make_deformable_collider(device)
    contact_impulse = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    with wp.ScopedDevice(device):
        j, ijtm = build_rigidity_operator(
            cell_volume=1.0,
            node_volumes=wp.array([1.0], dtype=float, device=device),
            node_positions=wp.array([_CONTACT_POINT], dtype=wp.vec3, device=device),
            collider=solver._mpm_model.collider,
            body_q=wp.empty(0, dtype=wp.transform, device=device),
            body_mass=wp.empty(0, dtype=float, device=device),
            body_inv_inertia=wp.empty(0, dtype=wp.mat33, device=device),
            particle_mass=solver.model.particle_mass,
            collider_ids=wp.array([0], dtype=int, device=device),
        )
        vertex_velocity = wp.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3)
        contact_velocity = wp.zeros(1, dtype=wp.vec3)
        wps.bsr_mv(j, x=vertex_velocity, y=contact_velocity)

        vertex_delta_velocity = wp.zeros(3, dtype=wp.vec3)
        wps.bsr_mv(ijtm, x=wp.array([contact_impulse], dtype=wp.vec3), y=vertex_delta_velocity)

    expected_speed = 1.0 - _CONTACT_POINT[0] - _CONTACT_POINT[1]
    np.testing.assert_allclose(contact_velocity.numpy()[0], [0.0, 0.0, expected_speed], rtol=0.0, atol=1.0e-6)
    vertex_momentum = vertex_delta_velocity.numpy() * solver.model.particle_mass.numpy()[:3, None]
    np.testing.assert_allclose(vertex_momentum.sum(axis=0), contact_impulse, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        np.cross(_VERTICES, vertex_momentum).sum(axis=0),
        np.cross(_CONTACT_POINT, contact_impulse),
        rtol=0.0,
        atol=1.0e-6,
    )


class TestImplicitMPMCouplingWeights(unittest.TestCase):
    """Check physical coupling behavior for deformable collider triangles."""


devices = get_test_devices("basic")
add_function_test(
    TestImplicitMPMCouplingWeights,
    "test_proxy_feedback_preserves_contact_force_and_moment",
    test_proxy_feedback_preserves_contact_force_and_moment,
    devices=devices,
)
add_function_test(
    TestImplicitMPMCouplingWeights,
    "test_proxy_feedback_sums_substep_impulses",
    test_proxy_feedback_sums_substep_impulses,
    devices=devices,
)
add_function_test(
    TestImplicitMPMCouplingWeights,
    "test_proxy_feedback_without_notification_uses_last_step",
    test_proxy_feedback_without_notification_uses_last_step,
    devices=devices,
)
add_function_test(
    TestImplicitMPMCouplingWeights,
    "test_proxy_feedback_storage_follows_notification",
    test_proxy_feedback_storage_follows_notification,
    devices=devices,
)
add_function_test(
    TestImplicitMPMCouplingWeights,
    "test_collider_contact_motion_and_moment",
    test_collider_contact_motion_and_moment,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
