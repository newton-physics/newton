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


def test_proxy_feedback_preserves_contact_force_and_moment(test, device):
    """Apply proxy feedback at its physical contact point."""
    solver, _mesh = _make_deformable_collider(device)
    contact_impulse = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    contact_point = wp.array([_CONTACT_POINT], dtype=wp.vec3, device=device)
    impulses = wp.array([contact_impulse], dtype=wp.vec3, device=device)
    collider_ids = wp.array([0], dtype=int, device=device)
    solver.collect_collider_impulses = lambda state: (impulses, contact_point, collider_ids)

    state = solver.model.state()
    feedback = wp.zeros(3, dtype=wp.vec3, device=device)
    dt = 2.0
    solver.coupling_harvest_proxy_particle_forces(
        wp.array([0, 1, 2, -1], dtype=int, device=device),
        feedback,
        particle_qd_before=state.particle_qd,
        state=state,
        state_out=state,
        contacts=None,
        dt=dt,
    )

    forces = feedback.numpy()
    expected_force = contact_impulse / dt
    np.testing.assert_allclose(forces.sum(axis=0), expected_force, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        np.cross(_VERTICES, forces).sum(axis=0), np.cross(_CONTACT_POINT, expected_force), rtol=0.0, atol=1.0e-6
    )


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
    "test_collider_contact_motion_and_moment",
    test_collider_contact_motion_and_moment,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
