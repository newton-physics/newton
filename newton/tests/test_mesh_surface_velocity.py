# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mesh surface velocity in rigid contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.collide import eval_rigid_contact_surface_velocities
from newton.solvers.experimental.coupled import SolverCoupled
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


def test_mesh_surface_velocity_is_not_recorded_on_tape(test, device):
    """Keep non-differentiable surface-velocity evaluation off Warp tapes."""
    mesh = newton.Mesh.create_plane(1.0, 1.0, compute_inertia=False)
    mesh.enable_surface_velocity = True
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=mesh)
    model = builder.finalize(device=device)

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    with wp.Tape() as tape:
        pipeline.collide(model.state(), contacts)

    recorded_kernels = [launch[0] for launch in tape.launches if not callable(launch)]
    test.assertNotIn(eval_rigid_contact_surface_velocities, recorded_kernels)


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


def test_identical_meshes_have_independent_surface_velocities(test, device):
    """Keep mutable velocity fields independent for identical mesh objects."""
    vertices = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    meshes = [
        newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=True),
        newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=True),
    ]

    builder = newton.ModelBuilder()
    mesh_shapes = []
    sphere_shapes = []
    for x, mesh in zip((-1.0, 1.0), meshes, strict=True):
        mesh_shapes.append(
            builder.add_shape_mesh(
                body=-1,
                xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                mesh=mesh,
            )
        )
        sphere_body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.09), wp.quat_identity()))
        sphere_shapes.append(builder.add_shape_sphere(body=sphere_body, radius=0.1))

    model = builder.finalize(device=device)
    prescribed_velocities = (wp.vec3(0.5, 0.0, 0.0), wp.vec3(-1.25, 0.0, 0.0))
    for mesh, velocity in zip(meshes, prescribed_velocities, strict=True):
        test.assertIsNotNone(mesh.mesh)
        mesh.mesh.velocities.fill_(velocity)

    test.assertNotEqual(meshes[0].mesh.id, meshes[1].mesh.id)

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreaterEqual(count, 2)
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    contact_velocity = contacts.rigid_contact_surface_velocity.numpy()[:count]
    expected_by_mesh_shape = {
        mesh_shapes[0]: np.array(prescribed_velocities[0]),
        mesh_shapes[1]: np.array(prescribed_velocities[1]),
    }
    observed_mesh_shapes = set()
    for i in range(count):
        for mesh_shape, sphere_shape in zip(mesh_shapes, sphere_shapes, strict=True):
            if sphere_shape not in (shape0[i], shape1[i]):
                continue
            observed_mesh_shapes.add(mesh_shape)
            sign = 1.0 if shape1[i] == mesh_shape else -1.0
            np.testing.assert_allclose(contact_velocity[i], sign * expected_by_mesh_shape[mesh_shape], atol=1.0e-6)

    test.assertEqual(observed_mesh_shapes, set(mesh_shapes))


def test_normal_surface_velocity_does_not_affect_contact(test, device, solver_name):
    """Ignore surface velocity perpendicular to the contact surface."""
    vertices = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    meshes = [
        newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=True),
        newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=True),
    ]
    material = newton.ModelBuilder.ShapeConfig(mu=0.8, ke=1.0e5, kd=1.0e3, kf=1.0e4)

    builder = newton.ModelBuilder()
    bodies = []
    for x, mesh in zip((-1.0, 1.0), meshes, strict=True):
        builder.add_shape_mesh(
            body=-1,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            mesh=mesh,
            cfg=material,
        )
        body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.19), wp.quat_identity()))
        builder.add_shape_sphere(body=body, radius=0.2, cfg=material)
        builder.add_articulation([builder.add_joint_free(body)])
        bodies.append(body)
    builder.color()

    model = builder.finalize(device=device)
    meshes[0].mesh.velocities.zero_()
    meshes[1].mesh.velocities.fill_(wp.vec3(0.0, 0.0, 2.0))
    solver = _make_solver(solver_name, model)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)

    velocity = state_out.body_qd.numpy()
    np.testing.assert_allclose(velocity[bodies[0]], velocity[bodies[1]], atol=1.0e-5)


def test_mesh_surface_velocity_moves_rigid_body_through_coupled_solver(test, device):
    """Transport a rigid body through the coupled solver contact filter."""
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
    mesh_shape = builder.add_shape_mesh(body=-1, mesh=mesh, cfg=material)
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.19), wp.quat_identity()))
    body_shape = builder.add_shape_box(body=body, hx=0.2, hy=0.2, hz=0.2, cfg=material)
    joint = builder.add_joint_free(body)
    builder.add_articulation([joint])
    builder.color()

    model = builder.finalize(device=device)
    mesh.mesh.velocities.fill_(wp.vec3(1.0, 0.0, 0.0))
    solver = SolverCoupled(
        model,
        [
            SolverCoupled.Entry(
                "rigid",
                newton.solvers.SolverSemiImplicit,
                bodies=[body],
                joints=[joint],
                shapes=[mesh_shape, body_shape],
            )
        ],
    )
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

    test.assertGreater(state_in.body_q.numpy()[body, 0], 0.04)


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
    add_function_test(
        TestMeshSurfaceVelocity,
        "test_mesh_surface_velocity_is_not_recorded_on_tape",
        test_mesh_surface_velocity_is_not_recorded_on_tape,
        devices=[test_device],
    )
    add_function_test(
        TestMeshSurfaceVelocity,
        "test_identical_meshes_have_independent_surface_velocities",
        test_identical_meshes_have_independent_surface_velocities,
        devices=[test_device],
    )
    add_function_test(
        TestMeshSurfaceVelocity,
        "test_mesh_surface_velocity_moves_rigid_body_through_coupled_solver",
        test_mesh_surface_velocity_moves_rigid_body_through_coupled_solver,
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
        add_function_test(
            TestMeshSurfaceVelocity,
            f"test_normal_surface_velocity_does_not_affect_contact_{name}",
            test_normal_surface_velocity_does_not_affect_contact,
            devices=[test_device],
            solver_name=name,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
