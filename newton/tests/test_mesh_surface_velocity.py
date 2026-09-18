# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh surface velocity in rigid contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _make_solver(name, model):
    """Construct a rigid solver for the conveyor regression."""
    if name == "vbd":
        return newton.solvers.SolverVBD(model, iterations=5, rigid_compliant_alm=True)
    if name == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(model)
    if name == "featherstone":
        return newton.solvers.SolverFeatherstone(model)
    return newton.solvers.SolverXPBD(model, iterations=5)


def test_mesh_surface_velocity_is_opt_in(test, device):
    """Keep the surface-velocity path disabled for ordinary meshes."""
    mesh = newton.Mesh.create_plane(1.0, 1.0, compute_inertia=False)
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=mesh)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()

    test.assertEqual(len(contacts.rigid_contact_surface_velocity), 0)


def test_mesh_surface_velocity_moves_rigid_body(test, device, solver_name):
    """Move a resting rigid body using mesh vertex surface velocity."""
    vertices = np.array(
        [
            [-2.0, -2.0, 0.0],
            [2.0, -2.0, 0.0],
            [2.0, 2.0, 0.0],
            [-2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    mesh = newton.Mesh(
        vertices,
        np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        compute_inertia=False,
        enable_surface_velocity=True,
    )
    material = newton.ModelBuilder.ShapeConfig(mu=0.8, ke=1.0e5, kd=1.0e3, kf=1.0e4)

    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=mesh, cfg=material)
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.19), wp.quat_identity()))
    builder.add_shape_box(body=body, hx=0.2, hy=0.2, hz=0.2, cfg=material)
    builder.add_articulation([builder.add_joint_free(body)])
    builder.color()

    model = builder.finalize(device=device)
    mesh.mesh.velocities.fill_(wp.vec3(1.0, 0.0, 0.0))
    solver = _make_solver(solver_name, model)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / 240.0
    for _ in range(120):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in

    position = state_in.body_q.numpy()[body, :3]
    test.assertGreater(position[0], 0.04)


class TestMeshSurfaceVelocity(unittest.TestCase):
    def test_mesh_copy_preserves_surface_velocity_opt_in(self):
        """Preserve the surface-velocity opt-in when copying a mesh."""
        mesh = newton.Mesh.create_plane(1.0, 1.0, compute_inertia=False)
        mesh.enable_surface_velocity = True

        self.assertTrue(mesh.copy().enable_surface_velocity)


devices = get_test_devices()
for test_device in devices:
    add_function_test(
        TestMeshSurfaceVelocity,
        "test_mesh_surface_velocity_is_opt_in",
        test_mesh_surface_velocity_is_opt_in,
        devices=[test_device],
    )
    for name in ("xpbd", "semi_implicit", "featherstone", "vbd"):
        if name == "vbd" and test_device.is_cpu:
            continue
        add_function_test(
            TestMeshSurfaceVelocity,
            f"test_mesh_surface_velocity_moves_rigid_body_{name}",
            test_mesh_surface_velocity_moves_rigid_body,
            devices=[test_device],
            solver_name=name,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
