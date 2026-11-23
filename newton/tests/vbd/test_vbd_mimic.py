# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise mimic and limit convergence on an articulated finger."""

import unittest
from functools import partial

import numpy as np
import warp as wp

import newton
from newton._src.solvers.vbd.joint_mimic_kernels import JointMimicData, _row, accumulate_joint_mimics
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_finger(device, *, mimic=True, driven=False, d6=False):
    """Build an offset finger chain with a passive link beyond its mimic joint."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    masses = np.array([0.013, 0.039, 0.039, 0.027])
    inertias = np.array([2.0e-6, 1.3e-5, 1.2e-5, 1.2e-5])
    centers = [(0.0, -0.043, 0.054), (0.0, -0.066, 0.069), (0.0, -0.054, 0.114), (0.0, -0.032, 0.083)]
    for mass, inertia, center in zip(masses, inertias, centers, strict=True):
        builder.add_link(mass=float(mass), inertia=wp.mat33(np.eye(3) * inertia), com=wp.vec3(center))

    def hinge(parent, child, anchor, axis, **kwargs):
        frame = wp.transform(anchor, wp.quat_identity())
        if d6:
            config = newton.ModelBuilder.JointDofConfig
            return builder.add_joint_d6(
                parent,
                child,
                parent_xform=frame,
                child_xform=frame,
                angular_axes=[
                    config(axis=axis, **kwargs),
                    config.create_unlimited(newton.Axis.Y),
                    config.create_unlimited(newton.Axis.Z),
                ],
            )
        return builder.add_joint_revolute(parent, child, parent_xform=frame, child_xform=frame, axis=axis, **kwargs)

    target = np.deg2rad(47.0)
    drive = {}
    if driven:
        drive = {
            "target_q": target,
            "target_ke": 171.88734,
            "target_kd": 0.01145916,
            "limit_lower": 0.0,
            "limit_upper": target,
            "limit_ke": 1.0e4,
            "limit_kd": 10.0,
        }
    reference = hinge(-1, 0, (0.0, -0.031, 0.055), (-1.0, 0.0, 0.0), **drive)
    fixed = builder.add_joint_fixed(0, 1)
    follower = hinge(1, 2, (0.0, -0.068, 0.098), (-1.0, 0.0, 0.0))
    tip = hinge(2, 3, (0.0, -0.05, 0.105), (1.0, 0.0, 0.0))
    builder.add_articulation([reference, fixed, follower, tip])
    if mimic:
        builder.set_joint_mimic(follower, reference, coeffs=(0.0, -1.0))
    builder.color()
    return builder.finalize(device=device), masses, inertias, target


def test_vbd_mimic_articulated_chain(test, device):
    """Keep a passive articulated mimic from amplifying a small velocity perturbation."""
    model, masses, inertias, _ = _build_finger(device)
    state, other = model.state(), model.state()
    state.joint_qd[:1].fill_(0.01)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    model.body_q.assign(state.body_q)
    solver = newton.solvers.SolverVBD(model, iterations=5, rigid_joint_linear_ke=1.0e6, rigid_joint_angular_ke=1.0e6)
    for _ in range(300):
        solver.step(state, other, None, None, 1.0 / 960.0)
        state, other = other, state
        velocity = state.body_qd.numpy()
        energy = 0.5 * np.sum(masses * np.sum(velocity[:, :3] ** 2, axis=1))
        energy += 0.5 * np.sum(inertias * np.sum(velocity[:, 3:] ** 2, axis=1))
        test.assertLess(float(energy), 1.0e-5)


def test_vbd_driven_finger_limit(test, device):
    """Converge a light articulated finger's limit without delayed-force oscillation."""
    # The revolute case isolates the limit fix at the default structural
    # stiffness; the D6 case adds its fully free angular classes and the mimic
    # at the Dexmate stiffness. Previous limit conditioning violates the bounds
    # within 100 steps; the mimic case needs about 1850 steps to settle.
    for mimic, d6, steps, structural_ke in ((False, False, 1000, None), (True, True, 3000, 1.0e6)):
        with test.subTest(mimic=mimic, d6=d6):
            model, masses, inertias, target = _build_finger(device, mimic=mimic, driven=True, d6=d6)
            state, other = model.state(), model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            model.body_q.assign(state.body_q)
            solver = newton.solvers.SolverVBD(
                model, iterations=5, rigid_joint_linear_ke=structural_ke, rigid_joint_angular_ke=structural_ke
            )
            for step in range(steps):
                solver.step(state, other, None, None, 1.0 / 960.0)
                state, other = other, state
                # Sparse sampling still catches growing limit oscillations.
                if step % 10 != 9:
                    continue
                newton.eval_ik(model, state, state.joint_q, state.joint_qd)
                q = state.joint_q.numpy()
                # A finite-stiffness limit can briefly penetrate at five sweeps;
                # it must release without pumping energy into the next excursion.
                test.assertGreater(float(q[0]), -0.1)
                test.assertLess(float(q[0]), target + 0.1)
                velocity = state.body_qd.numpy()
                energy = 0.5 * np.sum(masses * np.sum(velocity[:, :3] ** 2, axis=1))
                energy += 0.5 * np.sum(inertias * np.sum(velocity[:, 3:] ** 2, axis=1))
                energy += 0.5 * 171.88734 * (float(q[0]) - target) ** 2
                test.assertLessEqual(float(energy), 0.5 * 171.88734 * target**2 * 1.02)
            test.assertAlmostEqual(float(q[0]), target, delta=0.01)
            if mimic:
                follower_start = int(model.joint_q_start.numpy()[2])
                np.testing.assert_allclose(q[:follower_start] + q[follower_start : 2 * follower_start], 0.0, atol=0.002)


@wp.kernel
def _evaluate_rows(
    data: JointMimicData,
    follower: int,
    poses: wp.array[wp.transform],
    errors: wp.array[float],
    gradients: wp.array3d[float],
):
    component = wp.tid()
    row = _row(data, follower, component, poses)
    errors[component] = row.error
    for slot in range(4):
        body = row.bodies[slot]
        if body >= 0:
            for axis in range(6):
                gradients[component, body, axis] = row.gradients[slot, axis]


def test_vbd_mimic_gradients(test, device):
    """Match mimic-coordinate derivatives, including moving axes and shared parents."""
    for joint_type, linear_count, angular_count, swap_axes in (
        (newton.JointType.PRISMATIC, 1, 0, False),
        (newton.JointType.REVOLUTE, 0, 1, False),
        (newton.JointType.D6, 2, 0, False),
        (newton.JointType.D6, 0, 1, False),
        (newton.JointType.D6, 1, 2, False),
        (newton.JointType.D6, 1, 3, False),
        (newton.JointType.D6, 1, 2, True),
        (newton.JointType.D6, 1, 3, True),
    ):
        with test.subTest(joint_type=joint_type, linear=linear_count, angular=angular_count, swap_axes=swap_axes):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            for body in range(3):
                rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, float(body + 1), 2.0)), 0.2 * (body + 1))
                builder.add_link(
                    xform=wp.transform((0.1 * body, 0.05 * body, -0.1 * body), rotation),
                    mass=1.0,
                    inertia=wp.mat33(np.eye(3)),
                    com=wp.vec3(0.03, -0.02, 0.01),
                )
            joints = [builder.add_joint_fixed(-1, 0)]
            for child in (1, 2):
                kwargs = {
                    "parent_xform": wp.transform((0.01, 0.03, -0.02), wp.quat_identity()),
                    "child_xform": wp.transform((-0.02, 0.01, 0.03), wp.quat_identity()),
                }
                if joint_type == newton.JointType.PRISMATIC:
                    joint = builder.add_joint_prismatic(0, child, axis=newton.Axis.X, **kwargs)
                elif joint_type == newton.JointType.REVOLUTE:
                    joint = builder.add_joint_revolute(0, child, axis=newton.Axis.Z, **kwargs)
                else:
                    axis = newton.ModelBuilder.JointDofConfig.create_unlimited
                    axes = (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
                    if swap_axes:
                        axes = (newton.Axis.X, newton.Axis.Z, newton.Axis.Y)
                    joint = builder.add_joint_d6(
                        0,
                        child,
                        linear_axes=[axis(a) for a in axes[:linear_count]],
                        angular_axes=[axis(a) for a in axes[:angular_count]],
                        **kwargs,
                    )
                joints.append(joint)
            builder.add_articulation(joints)
            builder.set_joint_mimic(joints[2], joints[1], coeffs=(0.12, -0.7))
            builder.color()
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverVBD(model)
            count = linear_count + angular_count
            errors = wp.zeros(count, dtype=float, device=device)
            gradients = wp.zeros((count, model.body_count, 6), dtype=float, device=device)
            poses = wp.clone(model.body_q)

            evaluate = partial(
                wp.launch,
                _evaluate_rows,
                dim=count,
                inputs=[solver._joint_mimics, joints[2], poses],
                outputs=[errors, gradients],
                device=device,
            )

            evaluate()
            analytical = gradients.numpy().copy()
            original = model.body_q.numpy().copy()
            centers = model.body_com.numpy()
            eps = 1.0e-3
            for body in range(model.body_count):
                for axis in range(6):
                    values = []
                    for sign in (-1.0, 1.0):
                        perturbed = original.copy()
                        if axis < 3:
                            perturbed[body, axis] += sign * eps
                        else:
                            rotation = wp.quat(*original[body, 3:])
                            com = wp.vec3(original[body, :3]) + wp.quat_rotate(rotation, wp.vec3(centers[body]))
                            direction = wp.vec3(0.0)
                            direction[axis - 3] = 1.0
                            rotated = wp.quat_from_axis_angle(direction, sign * eps) * rotation
                            perturbed[body, 3:] = np.array(rotated)
                            perturbed[body, :3] = np.array(com - wp.quat_rotate(rotated, wp.vec3(centers[body])))
                        poses.assign(perturbed)
                        evaluate()
                        values.append(errors.numpy().copy())
                    numeric = (values[1] - values[0]) / (2.0 * eps)
                    np.testing.assert_allclose(analytical[:, body, axis], numeric, atol=1.0e-3, rtol=2.0e-3)


def test_vbd_mimic_reset(test, device):
    """Reset mimic history only in selected worlds, including captured global resets."""

    def add_pair(builder):
        joints = []
        for _ in range(2):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joints.append(builder.add_joint_prismatic(-1, body, axis=newton.Axis.X))
        builder.add_articulation(joints)
        builder.set_joint_mimic(joints[1], joints[0])

    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    add_pair(template)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    add_pair(builder)
    builder.add_world(template)
    builder.add_world(template)
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=5)
    state, other = model.state(), model.state()
    velocities = model.joint_qd.numpy().copy()
    velocities[::2] = 0.01
    state.joint_qd.assign(velocities)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    for _ in range(3):
        solver.step(state, other, None, None, 1.0 / 960.0)
        state, other = other, state
    before = solver._joint_mimics.lambda_.numpy().copy()
    worlds = model.joint_world.numpy()
    test.assertGreater(float(np.max(np.abs(before[worlds == 0]))), 0.0)
    mask = wp.array([True, False, False], dtype=bool, device=device)
    solver.reset(state, world_mask=mask)
    after = solver._joint_mimics.lambda_.numpy()
    np.testing.assert_array_equal(after[worlds == 0], 0.0)
    np.testing.assert_array_equal(after[worlds != 0], before[worlds != 0])
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            solver.reset(state, world_mask=mask)
        solver._joint_mimics.lambda_.assign(before)
        mask.assign([False, False, True])
        wp.capture_launch(capture.graph)
        after = solver._joint_mimics.lambda_.numpy()
        np.testing.assert_array_equal(after[worlds < 0], 0.0)
        np.testing.assert_array_equal(after[worlds >= 0], before[worlds >= 0])
    solver.reset(state)
    np.testing.assert_array_equal(solver._joint_mimics.lambda_.numpy(), 0.0)


def test_vbd_mimic_property_updates(test, device):
    """Apply changed mimic coefficients and clear their history through JOINT_PROPERTIES."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    for _ in range(2):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joints.append(builder.add_joint_prismatic(-1, body, axis=newton.Axis.X))
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model, iterations=5)
    state, other = model.state(), model.state()
    state.joint_qd[:1].fill_(0.01)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver.step(state, other, None, None, 1.0 / 960.0)
    state, other = other, state
    before = solver._joint_mimics.lambda_.numpy().copy()
    test.assertGreater(float(np.max(np.abs(before))), 0.0)
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES | newton.ModelFlags.CONSTRAINT_PROPERTIES)
    np.testing.assert_array_equal(solver._joint_mimics.lambda_.numpy(), before)

    coefficients = model.joint_mimic_coeffs.numpy().copy()
    coefficients[joints[1]] = (0.001, -1.0)
    model.joint_mimic_coeffs = wp.array(coefficients, dtype=wp.vec2, device=device)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    np.testing.assert_array_equal(solver._joint_mimics.lambda_.numpy(), 0.0)
    for _ in range(96):
        solver.step(state, other, None, None, 1.0 / 960.0)
        state, other = other, state
    newton.eval_ik(model, state, state.joint_q, state.joint_qd)
    q = state.joint_q.numpy()
    test.assertAlmostEqual(float(q[0] + q[1]), 0.001, delta=1.0e-6)


def test_vbd_mimic_d6_gimbal_lock(test, device):
    """Keep D6 mimic rows finite when a coordinate sits at the Euler-angle singularity."""
    axis = newton.ModelBuilder.JointDofConfig.create_unlimited
    xyz = [newton.Axis.X, newton.Axis.Y, newton.Axis.Z]
    # An angular follower, and a linear follower of an angular reference.
    for follower_angular, dt, steps in ((True, 1.0 / 960.0, 100), (False, 1.0 / 60.0, 3)):
        with test.subTest(follower_angular=follower_angular):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            bodies = [builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
            reference = builder.add_joint_d6(-1, bodies[0], angular_axes=[axis(a) for a in xyz])
            if follower_angular:
                follower = builder.add_joint_d6(-1, bodies[1], angular_axes=[axis(a) for a in xyz])
            else:
                follower = builder.add_joint_d6(-1, bodies[1], linear_axes=[axis(a) for a in xyz])
            # Prescribe the joint that should not move: the reference, or the linear follower.
            builder.body_flags[bodies[0] if follower_angular else bodies[1]] = int(newton.BodyFlags.KINEMATIC)
            builder.add_articulation([reference, follower])
            builder.set_joint_mimic(follower, reference)
            builder.color()
            model = builder.finalize(device=device)
            q = np.array([0.1, 0.5 * np.pi, 0.2, 0.101, 0.5 * np.pi, 0.2], dtype=np.float32)
            if not follower_angular:
                q[3] = 0.1001
            model.joint_q.assign(q)
            state, other = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            model.body_q.assign(state.body_q)
            solver = newton.solvers.SolverVBD(model, iterations=5)
            for _ in range(steps):
                solver.step(state, other, None, None, dt)
                state, other = other, state
                test.assertTrue(np.all(np.isfinite(state.body_q.numpy())))
                test.assertTrue(np.all(np.isfinite(state.body_qd.numpy())))
            if follower_angular:
                body_q = state.body_q.numpy().astype(np.float64)
                # The Euler split is undefined here, so compare orientations instead.
                alignment = min(1.0, abs(float(np.dot(body_q[0, 3:], body_q[1, 3:]))))
                test.assertLess(2.0 * np.arccos(alignment), 5.0e-3)


def test_vbd_mimic_multi_turn_reference(test, device):
    """Track a reference past a full revolution with a lead-screw nut and a 1/3 gear."""
    pitch = 0.05
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    screw = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 1.0e-3))
    nut = builder.add_link(mass=0.2, inertia=wp.mat33(np.eye(3) * 1.0e-4))
    gear = builder.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 5.0e-4))
    j_screw = builder.add_joint_revolute(-1, screw, axis=newton.Axis.Z)
    j_nut = builder.add_joint_prismatic(
        -1, nut, axis=newton.Axis.Z, parent_xform=wp.transform((0.3, 0.0, 0.0), wp.quat_identity())
    )
    j_gear = builder.add_joint_revolute(
        -1, gear, axis=newton.Axis.Z, parent_xform=wp.transform((-0.3, 0.0, 0.0), wp.quat_identity())
    )
    builder.add_articulation([j_screw, j_nut, j_gear])
    builder.set_joint_mimic(j_nut, j_screw, coeffs=(0.0, pitch / (2.0 * np.pi)))
    builder.set_joint_mimic(j_gear, j_screw, coeffs=(0.0, 1.0 / 3.0))
    builder.color()
    model = builder.finalize(device=device)
    # One turn per second, with the followers moving consistently.
    model.joint_qd.assign(np.array([2.0 * np.pi, pitch, 2.0 * np.pi / 3.0], dtype=np.float32))
    state, other = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    model.body_q.assign(state.body_q)
    solver = newton.solvers.SolverVBD(model)
    for _ in range(360):
        solver.step(state, other, None, None, 1.0 / 240.0)
        state, other = other, state
    body_q = state.body_q.numpy()
    # Recover the screw's angle near its expected 1.5 turns from its orientation.
    angle = 2.0 * np.arctan2(body_q[screw, 5], body_q[screw, 6])
    angle += 2.0 * np.pi * np.round((3.0 * np.pi - angle) / (2.0 * np.pi))
    test.assertGreater(angle, 2.5 * np.pi)
    test.assertAlmostEqual(float(body_q[nut, 2]), pitch * angle / (2.0 * np.pi), delta=1.0e-3)
    gear_error = 2.0 * np.arctan2(body_q[gear, 5], body_q[gear, 6]) - angle / 3.0
    test.assertLess(abs(np.arctan2(np.sin(gear_error), np.cos(gear_error))), 0.02)


class TestSolverVBDMimic(unittest.TestCase):
    def test_mimic_kernel_module_follows_deterministic_mode(self):
        """Apply VBD's deterministic option to the separately compiled mimic kernels."""
        builder = newton.ModelBuilder()
        joints = []
        for _ in range(2):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joints.append(builder.add_joint_prismatic(-1, body, axis=newton.Axis.X))
        builder.add_articulation(joints)
        builder.set_joint_mimic(joints[1], joints[0])
        builder.color()
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverVBD(model, deterministic=wp.DeterministicMode.RUN_TO_RUN)
        module = accumulate_joint_mimics.module
        self.assertIn(module, solver._module_options)
        options = wp.get_module_options(module=module) | accumulate_joint_mimics.options
        self.assertEqual(options["deterministic"], wp.DeterministicMode.RUN_TO_RUN)


devices = get_test_devices()
add_function_test(
    TestSolverVBDMimic, "test_vbd_mimic_articulated_chain", test_vbd_mimic_articulated_chain, devices=devices
)
add_function_test(TestSolverVBDMimic, "test_vbd_driven_finger_limit", test_vbd_driven_finger_limit, devices=devices)
add_function_test(TestSolverVBDMimic, "test_vbd_mimic_gradients", test_vbd_mimic_gradients, devices=devices)
add_function_test(TestSolverVBDMimic, "test_vbd_mimic_reset", test_vbd_mimic_reset, devices=devices)
add_function_test(TestSolverVBDMimic, "test_vbd_mimic_d6_gimbal_lock", test_vbd_mimic_d6_gimbal_lock, devices=devices)
add_function_test(
    TestSolverVBDMimic, "test_vbd_mimic_multi_turn_reference", test_vbd_mimic_multi_turn_reference, devices=devices
)
add_function_test(
    TestSolverVBDMimic, "test_vbd_mimic_property_updates", test_vbd_mimic_property_updates, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
