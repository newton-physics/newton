# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experimental BDF2 rigid time integrator of :class:`newton.solvers.SolverVBD`."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

devices = get_test_devices()
cuda_devices = [device for device in devices if device.is_cuda]

_GRAVITY = 9.81
_BOX_INERTIA = wp.mat33(np.eye(3, dtype=np.float32) / 600.0)


def _make_solver(model: newton.Model, **options) -> newton.solvers.SolverVBD:
    """Create a block-sparse compliant-ALM VBD solver with the given options."""
    settings = {"iterations": 8, "rigid_compliant_alm": True, "rigid_articulation_solve": "block_sparse_joints"}
    settings.update(options)
    return newton.solvers.SolverVBD(model, **settings)


def _add_pendulum(builder: newton.ModelBuilder, angle: float = 0.5) -> int:
    """Add a hinged pendulum (COM 0.5 m below the hinge) at ``angle`` and return its joint."""
    link = builder.add_link(
        mass=1.0, com=wp.vec3(0.0, 0.0, -0.5), inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3)
    )
    joint = builder.add_joint_revolute(
        -1, link, axis=newton.Axis.Y, parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity())
    )
    builder.add_articulation([joint])
    builder.joint_q[builder.joint_q_start[joint]] = angle
    return joint


def _pendulum_angle(device, integrator: str, dt: float, duration: float) -> float:
    """Simulate the pendulum for ``duration`` seconds and return its final angle."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_GRAVITY))
    _add_pendulum(builder)
    builder.color()
    model = builder.finalize(device=device)
    solver = _make_solver(model, rigid_time_integration=integrator, rigid_articulation_relaxation=1.0)
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    for _ in range(round(duration / dt)):
        solver.step(state_in, state_out, None, None, dt)
        state_in, state_out = state_out, state_in
    newton.eval_ik(model, state_in, state_in.joint_q, state_in.joint_qd)
    return float(state_in.joint_q.numpy()[0])


def test_bdf2_free_fall_is_exact(test, device):
    """Follow the exact parabola of a free-falling body up to the BDF1 start-up offset."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_GRAVITY))
    body = builder.add_link(mass=2.0, inertia=wp.mat33(0.02, 0.001, 0.0, 0.001, 0.03, 0.0, 0.0, 0.0, 0.04))
    builder.add_articulation([builder.add_joint_free(body)])
    builder.color()
    model = builder.finalize(device=device)

    velocity0 = np.array((0.3, -0.2, 1.0))
    gravity = np.array((0.0, 0.0, -_GRAVITY))
    dt = 2.0e-3
    errors = {}
    for integrator in ("bdf1", "bdf2"):
        solver = _make_solver(model, rigid_time_integration=integrator, rigid_articulation_relaxation=1.0)
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        state_in.body_qd.assign(np.array([[*velocity0, 0.0, 0.0, 0.0]], dtype=np.float32))
        position_errors = []
        for step in range(1, 251):
            solver.step(state_in, state_out, None, None, dt)
            state_in, state_out = state_out, state_in
            t = step * dt
            position = state_in.body_q.numpy()[body, 0:3]
            position_errors.append(np.abs(position - (velocity0 * t + 0.5 * gravity * t * t)).max())
        velocity = state_in.body_qd.numpy()[body]
        np.testing.assert_allclose(velocity[0:3], velocity0 + gravity * 250 * dt, atol=2.0e-5)
        np.testing.assert_allclose(velocity[3:6], 0.0, atol=1.0e-6)
        errors[integrator] = position_errors

    # BDF1 lags the parabola by dt g t / 2; BDF2 keeps the constant offset of its two BDF1 start-up steps.
    test.assertGreater(errors["bdf1"][-1], 4.0e-3)
    test.assertLess(errors["bdf2"][-1], 1.0e-4)
    test.assertLess(abs(errors["bdf2"][-1] - errors["bdf2"][124]), 2.0e-5)


def test_bdf2_pendulum_converges_second_order(test, device):
    """Shrink the pendulum error about 4x per halved step with BDF2 and about 2x with BDF1."""
    duration = 0.5
    for integrator, low, high in (("bdf1", 1.6, 2.5), ("bdf2", 3.2, 5.0)):
        angles = [_pendulum_angle(device, integrator, dt, duration) for dt in (4.0e-3, 2.0e-3, 1.0e-3)]
        # Successive differences estimate the error of the coarser step (Richardson).
        ratio = abs(angles[0] - angles[1]) / abs(angles[1] - angles[2])
        test.assertGreater(ratio, low, f"{integrator} angles {angles}")
        test.assertLess(ratio, high, f"{integrator} angles {angles}")


def _box_drop_heights(device, theta_policy: str) -> np.ndarray:
    """Drop an elastic box 0.1 m onto a plane with BDF2 and return its bottom height per step."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_GRAVITY))
    shape_cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e5, kd=0.0, mu=0.5, density=0.0)
    builder.add_ground_plane(cfg=shape_cfg)
    box = builder.add_link(xform=wp.transform((0.0, 0.0, 0.15), wp.quat_identity()), mass=1.0, inertia=_BOX_INERTIA)
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=shape_cfg)
    builder.add_articulation([builder.add_joint_free(box)])
    builder.color()
    model = builder.finalize(device=device)
    solver = _make_solver(model, rigid_time_integration="bdf2", rigid_bdf2_theta_policy=theta_policy)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    heights = []
    for _ in range(500):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, 2.0e-3)
        state_in, state_out = state_out, state_in
        heights.append(float(state_in.body_q.numpy()[box, 2]) - 0.05)
    return np.array(heights)


def test_bdf2_box_drop_rebound_is_bounded(test, device):
    """Settle an elastic box dropped on a plane with the event fallback to BDF1."""
    heights = _box_drop_heights(device, "event")
    first_contact = int(np.argmax(heights < 2.0e-3))
    test.assertGreater(first_contact, 0)
    test.assertLess(heights[first_contact + 5 :].max(), 0.02)
    test.assertLess(np.abs(heights[-100:]).max(), 1.0e-3)


def test_bdf2_reset_restarts_world_history(test, device):
    """Restart a reset world from a fresh BDF2 history without disturbing the other world."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_GRAVITY))
    for _ in range(2):
        world = newton.ModelBuilder(gravity=(0.0, 0.0, -_GRAVITY))
        _add_pendulum(world)
        builder.add_world(world)
    builder.color()
    model = builder.finalize(device=device)
    dt = 2.0e-3
    reset_step = 30

    def run(theta_policy: str, reset_at: int | None) -> np.ndarray:
        solver = _make_solver(
            model,
            rigid_time_integration="bdf2",
            rigid_bdf2_theta_policy=theta_policy,
            rigid_articulation_relaxation=1.0,
        )
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        start_q = state_in.body_q.numpy().copy()
        start_qd = state_in.body_qd.numpy().copy()
        poses = []
        for step in range(2 * reset_step):
            if step == reset_at:
                body_q = state_in.body_q.numpy()
                body_qd = state_in.body_qd.numpy()
                body_q[0] = start_q[0]
                body_qd[0] = start_qd[0]
                state_in.body_q.assign(body_q)
                state_in.body_qd.assign(body_qd)
                newton.eval_ik(model, state_in, state_in.joint_q, state_in.joint_qd)
                solver.reset(state_in, world_mask=wp.array([True, False, False], dtype=wp.bool, device=device), flags=0)
            solver.step(state_in, state_out, None, None, dt)
            state_in, state_out = state_out, state_in
            poses.append(state_in.body_q.numpy().copy())
        return np.array(poses)

    for theta_policy in ("event", "fresh"):
        fresh = run(theta_policy, None)
        reset = run(theta_policy, reset_step)
        np.testing.assert_array_equal(reset[:, 1], fresh[:, 1], err_msg=theta_policy)
        np.testing.assert_allclose(reset[reset_step:, 0], fresh[:reset_step, 0], atol=1.0e-6, err_msg=theta_policy)


def test_bdf2_rejects_unsupported_models(test, device):
    """Reject BDF2 with the local solve, joint mimics, particles or invalid option values."""

    def pendulum_model(add_mimic: bool = False, add_particle: bool = False) -> newton.Model:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_GRAVITY))
        if add_mimic:
            inertia = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3)
            upper = builder.add_link(mass=1.0, inertia=inertia)
            lower = builder.add_link(mass=1.0, inertia=inertia)
            leader = builder.add_joint_revolute(-1, upper, axis=newton.Axis.Y)
            follower = builder.add_joint_revolute(
                upper, lower, axis=newton.Axis.Y, parent_xform=wp.transform((0.0, 0.0, -0.5), wp.quat_identity())
            )
            builder.add_articulation([leader, follower])
            builder.set_joint_mimic(follower, leader)
        else:
            _add_pendulum(builder)
        if add_particle:
            builder.add_particle((0.0, 0.0, 2.0), (0.0, 0.0, 0.0), 1.0)
        builder.color()
        return builder.finalize(device=device)

    with test.assertRaisesRegex(ValueError, "block_sparse_joints"):
        _make_solver(pendulum_model(), rigid_time_integration="bdf2", rigid_articulation_solve="local")
    with test.assertRaisesRegex(ValueError, "mimic"):
        _make_solver(pendulum_model(add_mimic=True), rigid_time_integration="bdf2")
    with test.assertRaisesRegex(ValueError, "particles"):
        _make_solver(pendulum_model(add_particle=True), rigid_time_integration="bdf2")
    with test.assertRaisesRegex(ValueError, "rigid_time_integration"):
        _make_solver(pendulum_model(), rigid_time_integration="bdf3")
    with test.assertRaisesRegex(ValueError, "rigid_bdf2_theta_policy"):
        _make_solver(pendulum_model(), rigid_time_integration="bdf2", rigid_bdf2_theta_policy="off")

    solver = _make_solver(pendulum_model(), rigid_time_integration="bdf2")
    test.assertTrue(solver.rigid_axis_support_armature)
    test.assertTrue(solver.rigid_limit_rho_floor)
    test.assertEqual(solver.rigid_external_force_mode, "solve")


def test_bdf2_kinematic_body_reports_its_rate(test, device):
    """Report the user-imposed velocity of an animated kinematic platform during BDF2 steps."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_GRAVITY))
    platform = builder.add_body(is_kinematic=True)
    builder.add_shape_box(platform, hx=0.5, hy=0.5, hz=0.05)
    box = builder.add_link(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), mass=1.0, inertia=_BOX_INERTIA)
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    builder.add_articulation([builder.add_joint_free(box)])
    builder.color()
    model = builder.finalize(device=device)
    solver = _make_solver(model, rigid_time_integration="bdf2")
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    platform_velocity = np.array((0.2, 0.0, 0.1), dtype=np.float32)
    dt = 2.0e-3
    for step in range(60):
        body_q = state_in.body_q.numpy()
        body_qd = state_in.body_qd.numpy()
        body_q[platform, 0:3] = platform_velocity * (step + 1) * dt
        body_qd[platform, 0:3] = platform_velocity
        state_in.body_q.assign(body_q)
        state_in.body_qd.assign(body_qd)
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
        if step > 0:
            np.testing.assert_allclose(
                state_in.body_qd.numpy()[platform, 0:3], platform_velocity, atol=1.0e-5, err_msg=f"step {step}"
            )
    # The box rides on the platform.
    np.testing.assert_allclose(state_in.body_qd.numpy()[box, 0:3], platform_velocity, atol=2.0e-2)


def test_bdf2_step_captures_in_cuda_graph(test, device):
    """Replay BDF2 steps with contacts and a per-world reset from a CUDA graph exactly like eager steps."""

    def make_scene():
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_GRAVITY))
        for _ in range(2):
            world = newton.ModelBuilder(gravity=(0.0, 0.0, -_GRAVITY))
            world.add_ground_plane()
            box = world.add_link(
                xform=wp.transform((0.6, 0.0, 0.1), wp.quat_identity()), mass=1.0, inertia=_BOX_INERTIA
            )
            world.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
            world.add_articulation([world.add_joint_free(box)])
            _add_pendulum(world)
            builder.add_world(world)
        builder.color()
        model = builder.finalize(device=device)
        solver = _make_solver(model, rigid_time_integration="bdf2")
        pipeline = newton.CollisionPipeline(model)
        return model, solver, pipeline, pipeline.contacts()

    def run(use_graph: bool) -> np.ndarray:
        model, solver, pipeline, contacts = make_scene()
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        control = model.control()

        def substeps():
            # Two substeps, so every launch starts and ends in state_0.
            for state_in, state_out in ((state_0, state_1), (state_1, state_0)):
                state_in.clear_forces()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, 2.0e-3)

        graph = None
        if use_graph:
            with wp.ScopedCapture(device=device) as capture:
                substeps()
            graph = capture.graph
        world_mask = wp.array([True, False, False], dtype=wp.bool, device=device)
        for launch in range(40):
            if launch == 20:
                solver.reset(state_0, world_mask=world_mask, flags=0)
            if graph is not None:
                wp.capture_launch(graph)
            else:
                substeps()
        return np.concatenate([state_0.body_q.numpy().ravel(), state_0.body_qd.numpy().ravel()])

    eager = run(False)
    graph = run(True)
    test.assertTrue(np.isfinite(graph).all())
    np.testing.assert_allclose(graph, eager, atol=1.0e-5)


class TestSolverVBDBDF2(unittest.TestCase):
    pass


add_function_test(TestSolverVBDBDF2, "test_bdf2_free_fall_is_exact", test_bdf2_free_fall_is_exact, devices=devices)
add_function_test(
    TestSolverVBDBDF2,
    "test_bdf2_pendulum_converges_second_order",
    test_bdf2_pendulum_converges_second_order,
    devices=devices,
)
add_function_test(
    TestSolverVBDBDF2, "test_bdf2_box_drop_rebound_is_bounded", test_bdf2_box_drop_rebound_is_bounded, devices=devices
)
add_function_test(
    TestSolverVBDBDF2, "test_bdf2_reset_restarts_world_history", test_bdf2_reset_restarts_world_history, devices=devices
)
add_function_test(
    TestSolverVBDBDF2, "test_bdf2_rejects_unsupported_models", test_bdf2_rejects_unsupported_models, devices=devices
)
add_function_test(
    TestSolverVBDBDF2,
    "test_bdf2_kinematic_body_reports_its_rate",
    test_bdf2_kinematic_body_reports_its_rate,
    devices=devices,
)
add_function_test(
    TestSolverVBDBDF2,
    "test_bdf2_step_captures_in_cuda_graph",
    test_bdf2_step_captures_in_cuda_graph,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
