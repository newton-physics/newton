# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.xpbd.kernels import count_joint_mimics_per_body, project_joint_mimics
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_mimic_model(device, vectorized, *, kinematic_reference=False, small_mismatch=False):
    """Build an inconsistent scalar or vectorized mimic model."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    reference_body = builder.add_link()
    follower_body = builder.add_link()
    builder.add_shape_box(reference_body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(follower_body, hx=0.1, hy=0.1, hz=0.1)
    if kinematic_reference:
        builder.body_flags[reference_body] = int(newton.BodyFlags.KINEMATIC)

    if vectorized:
        axis = newton.ModelBuilder.JointDofConfig.create_unlimited
        axes = [axis(newton.Axis.X), axis(newton.Axis.Y)]
        reference = builder.add_joint_d6(-1, reference_body, linear_axes=axes)
        follower = builder.add_joint_d6(-1, follower_body, linear_axes=axes)
        initial_q = [0.2, -0.1, -0.18, 0.23] if small_mismatch else [0.2, -0.1, 0.8, 0.5]
    else:
        reference = builder.add_joint_revolute(-1, reference_body, axis=newton.Axis.Z)
        follower = builder.add_joint_prismatic(-1, follower_body, axis=newton.Axis.X)
        initial_q = [0.35, -0.405] if small_mismatch else [0.35, 0.8]

    builder.add_articulation([reference, follower])
    offset = 0.1
    multiplier = -1.5
    builder.set_joint_mimic(follower, reference, (offset, multiplier))
    builder.color()
    model = builder.finalize(device=device)
    model.joint_q.assign(np.asarray(initial_q, dtype=np.float32))
    model.joint_qd.zero_()

    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    return model, state_in, state_out, reference, follower, offset, multiplier


def _read_joint_q(model, state, generalized):
    """Return generalized positions from either reduced or maximal solver state."""
    if generalized:
        return state.joint_q.numpy()
    joint_q = wp.empty_like(model.joint_q)
    joint_qd = wp.empty_like(model.joint_qd)
    newton.eval_ik(model, state, joint_q, joint_qd)
    return joint_q.numpy()


def _assert_mimic_relation(model, joint_q, reference, follower, offset, multiplier):
    """Assert every follower coordinate satisfies its affine mimic relationship."""
    joint_q_start = model.joint_q_start.numpy()
    reference_slice = slice(joint_q_start[reference], joint_q_start[reference + 1])
    follower_slice = slice(joint_q_start[follower], joint_q_start[follower + 1])
    np.testing.assert_allclose(
        joint_q[follower_slice],
        offset + multiplier * joint_q[reference_slice],
        atol=2.0e-3,
    )


def _test_solver_mimic(test, device, solver_name):
    """Enforce scalar and vectorized mimic relationships with one solver."""
    for vectorized in (False, True):
        with test.subTest(vectorized=vectorized):
            # VBD performs one local mimic projection per solver iteration. Keep
            # its initial pose within the projection's intended convergence basin;
            # large authored inconsistencies should be resolved with eval_mimic()
            # followed by eval_fk() before simulation.
            model, state_in, state_out, reference, follower, offset, multiplier = _build_mimic_model(
                device,
                vectorized,
                small_mismatch=solver_name == "vbd",
            )
            if solver_name == "featherstone":
                solver = newton.solvers.SolverFeatherstone(model)
                generalized = True
            elif solver_name == "semi_implicit":
                if vectorized:
                    solver = newton.solvers.SolverSemiImplicit(
                        model,
                        joint_mimic_ke=1.0e4,
                        joint_mimic_kd=1.0e2,
                    )
                else:
                    solver = newton.solvers.SolverSemiImplicit(
                        model,
                        joint_mimic_ke=1.0e2,
                        joint_mimic_kd=3.0,
                    )
                generalized = False
            elif solver_name == "vbd":
                solver = newton.solvers.SolverVBD(
                    model,
                    iterations=4,
                    rigid_compliant_alm=True,
                )
                generalized = False
            else:
                raise ValueError(f"Unknown solver {solver_name}")

            step_count = 240 if solver_name == "semi_implicit" else 1
            dt = 1.0 / 240.0 if solver_name == "semi_implicit" else 1.0 / 60.0
            for _ in range(step_count):
                solver.step(state_in, state_out, None, None, dt)
                state_in, state_out = state_out, state_in
            joint_q = _read_joint_q(model, state_in, generalized)
            _assert_mimic_relation(model, joint_q, reference, follower, offset, multiplier)
            if not generalized and not vectorized:
                reference_q = float(joint_q[model.joint_q_start.numpy()[reference]])
                test.assertNotAlmostEqual(reference_q, 0.35, places=4)


def _test_featherstone_force_transfer(test, device):
    """Transfer a follower force into the Featherstone reference coordinate."""
    model, state_in, state_out, reference, follower, _, _ = _build_mimic_model(device, False)
    control = model.control()
    follower_dof = int(model.joint_qd_start.numpy()[follower])
    reference_dof = int(model.joint_qd_start.numpy()[reference])
    control.joint_f[follower_dof : follower_dof + 1].fill_(1.0)

    solver = newton.solvers.SolverFeatherstone(model)
    solver.step(state_in, state_out, control, None, 1.0 / 60.0)

    reference_velocity = float(state_out.joint_qd.numpy()[reference_dof])
    test.assertGreater(abs(reference_velocity), 1.0e-4)


def test_featherstone_mimic(test, device):
    """Enforce scalar and vectorized mimic relationships in Featherstone."""
    _test_solver_mimic(test, device, "featherstone")
    _test_featherstone_force_transfer(test, device)


def test_semi_implicit_mimic(test, device):
    """Enforce scalar and vectorized mimic relationships in SemiImplicit."""
    _test_solver_mimic(test, device, "semi_implicit")


def test_semi_implicit_mimic_preserves_kinematic_reference(test, device):
    """Keep a prescribed kinematic reference fixed while its follower responds."""
    model, state_in, state_out, reference, follower, offset, multiplier = _build_mimic_model(
        device,
        False,
        kinematic_reference=True,
    )
    reference_body = int(model.joint_child.numpy()[reference])
    reference_pose = state_in.body_q.numpy()[reference_body].copy()
    reference_velocity = state_in.body_qd.numpy()[reference_body].copy()
    solver = newton.solvers.SolverSemiImplicit(
        model,
        joint_mimic_ke=1.0e3,
        joint_mimic_kd=1.5e2,
    )

    for _ in range(480):
        solver.step(state_in, state_out, None, None, 1.0 / 240.0)
        state_in, state_out = state_out, state_in

    np.testing.assert_allclose(state_in.body_q.numpy()[reference_body], reference_pose, atol=1.0e-7)
    np.testing.assert_allclose(state_in.body_qd.numpy()[reference_body], reference_velocity, atol=1.0e-7)
    joint_q = _read_joint_q(model, state_in, False)
    _assert_mimic_relation(model, joint_q, reference, follower, offset, multiplier)


def test_semi_implicit_mimic_validates_gains(test, device):
    """Reject negative or non-finite global mimic penalty gains."""
    model, *_ = _build_mimic_model(device, False)
    for name in ("joint_mimic_ke", "joint_mimic_kd"):
        for value in (-1.0, float("nan"), float("inf")):
            with test.subTest(name=name, value=value):
                with test.assertRaisesRegex(ValueError, f"{name} must be finite and non-negative"):
                    newton.solvers.SolverSemiImplicit(model, **{name: value})


def test_vbd_mimic(test, device):
    """Enforce scalar and vectorized mimic relationships in VBD."""
    _test_solver_mimic(test, device, "vbd")


def _test_vbd_mimic_shared_reference(test, device, *, compliant=False):
    """Keep shared-reference mimic motion bounded and transfer momentum to every follower."""
    for joint_type in (newton.JointType.PRISMATIC, newton.JointType.REVOLUTE, newton.JointType.D6):
        for follower_count in (1, 2, 3, 5):
            with test.subTest(joint_type=joint_type, followers=follower_count, compliant=compliant):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                joints = []
                for _ in range(follower_count + 1):
                    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                    if joint_type == newton.JointType.PRISMATIC:
                        joint = builder.add_joint_prismatic(-1, body, axis=newton.Axis.Z)
                    elif joint_type == newton.JointType.REVOLUTE:
                        joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z)
                    else:
                        axis = newton.ModelBuilder.JointDofConfig.create_unlimited
                        joint = builder.add_joint_d6(
                            -1, body, linear_axes=[axis(newton.Axis.X)], angular_axes=[axis(newton.Axis.Z)]
                        )
                    joints.append(joint)
                builder.add_articulation(joints)
                for follower in joints[1:]:
                    builder.set_joint_mimic(follower, joints[0])
                builder.color()
                model = builder.finalize(device=device)
                state_in, state_out = model.state(), model.state()
                # A small leader-only velocity excites the constraint from a consistent pose.
                dof_count = 2 if joint_type == newton.JointType.D6 else 1
                state_in.joint_qd[:dof_count].fill_(0.01)
                newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                model.body_q.assign(state_in.body_q)
                solver = newton.solvers.SolverVBD(model, iterations=5, rigid_compliant_alm=compliant)

                initial_energy = float(np.sum(state_in.body_qd.numpy() ** 2))
                for _ in range(32):
                    solver.step(state_in, state_out, None, None, 1.0 / 960.0)
                    state_in, state_out = state_out, state_in
                    # All bodies have unit mass and inertia, so this is twice kinetic energy.
                    energy = float(np.sum(state_in.body_qd.numpy() ** 2))
                    test.assertLessEqual(energy, initial_energy * 1.001)

                newton.eval_ik(model, state_in, state_in.joint_q, state_in.joint_qd)
                q = state_in.joint_q.numpy().reshape(-1, dof_count)
                qd = state_in.joint_qd.numpy().reshape(-1, dof_count)
                np.testing.assert_allclose(q[1:] - q[0], 0.0, atol=1.0e-6)
                np.testing.assert_allclose(qd, 0.01 / (follower_count + 1), atol=1.0e-5)

                if follower_count == 5 and joint_type == newton.JointType.PRISMATIC:
                    if device.is_cuda:
                        # Capture before modifying the model so replay must use refreshed counts.
                        with wp.ScopedCapture(device=device) as capture:
                            solver.step(state_in, state_out, None, None, 1.0 / 960.0)
                            solver.step(state_out, state_in, None, None, 1.0 / 960.0)

                    # Disabled relationships must stop diluting the remaining pair's response.
                    enabled = model.joint_enabled.numpy()
                    enabled[2:] = False
                    model.joint_enabled.assign(enabled)
                    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)

                    for remove_references in (False, True):
                        if remove_references:
                            # Re-enable the joints, then remove their mimic references instead.
                            model.joint_enabled.fill_(True)
                            solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
                            references = model.joint_mimic_joint.numpy()
                            references[2:] = -1
                            model.joint_mimic_joint.assign(references)
                            solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
                        for _ in range(3):
                            velocities = state_in.body_qd.numpy()
                            velocities[0, 2] += 0.01
                            state_in.body_qd.assign(velocities)
                            expected = (velocities[0, 2] + velocities[1, 2]) * 0.5
                            if device.is_cuda:
                                wp.capture_launch(capture.graph)
                            else:
                                solver.step(state_in, state_out, None, None, 1.0 / 960.0)
                                state_in, state_out = state_out, state_in
                            updated = state_in.body_qd.numpy()
                            np.testing.assert_allclose(updated[:2, 2], expected, atol=2.0e-5)
                            np.testing.assert_allclose(updated[2:], velocities[2:], atol=1.0e-6)


def test_vbd_mimic_shared_reference(test, device):
    """Stabilize multiple followers sharing one reference in both VBD formulations."""
    for compliant in (False, True):
        _test_vbd_mimic_shared_reference(test, device, compliant=compliant)


def test_vbd_mimic_shared_parent(test, device):
    """Reduce projection error without injecting energy when mimic pairs share a light parent."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(mass=0.01, inertia=wp.mat33(np.eye(3) * 0.01))
    root = builder.add_joint_prismatic(-1, parent, axis=newton.Axis.X)
    joints = [root]
    pairs = []
    for i in range(5):
        reference_body = builder.add_link(mass=1.0 + i, inertia=wp.mat33(np.eye(3)))
        follower_body = builder.add_link(mass=0.5 + i, inertia=wp.mat33(np.eye(3)))
        reference = builder.add_joint_prismatic(parent, reference_body, axis=newton.Axis.X)
        follower = builder.add_joint_prismatic(parent, follower_body, axis=newton.Axis.X)
        multiplier = 1.5 if i % 2 else -1.0
        builder.set_joint_mimic(follower, reference, coeffs=(0.0, multiplier))
        joints.extend((reference, follower))
        pairs.append((reference, follower, multiplier))
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    state = model.state()
    dt = 1.0 / 960.0
    qd = model.joint_qd.numpy()
    qd[root] = 0.01
    for reference, follower, _ in pairs:
        qd[reference] = -0.01
        qd[follower] = -0.01
    # Predict a small parent displacement before applying the VBD mimic projection.
    state.joint_q.assign(qd * dt)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    counts = wp.zeros(model.body_count, dtype=int, device=device)
    deltas = wp.zeros_like(state.body_qd)
    count_joint_mimics_per_body(model, counts)
    masses = model.body_mass.numpy()
    velocities = state.body_qd.numpy()[:, 0]
    energy = float(np.dot(masses, velocities**2))
    momentum = float(np.dot(masses, velocities))
    for _ in range(10):
        project_joint_mimics(
            model, state.body_q, state.body_qd, model.body_inv_mass, model.body_inv_inertia, deltas, counts, dt
        )
        velocities = state.body_qd.numpy()[:, 0]
        updated_energy = float(np.dot(masses, velocities**2))
        test.assertLessEqual(updated_energy, energy * 1.00001 + 1.0e-12)
        test.assertAlmostEqual(float(np.dot(masses, velocities)), momentum, delta=1.0e-8)
        energy = updated_energy
    q = _read_joint_q(model, state, False)
    for reference, follower, multiplier in pairs:
        initial_error = abs(float(qd[follower] - multiplier * qd[reference])) * dt
        test.assertLess(abs(float(q[follower] - multiplier * q[reference])), initial_error)


class TestSolverMimic(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestSolverMimic, "test_featherstone_mimic", test_featherstone_mimic, devices=devices)
add_function_test(TestSolverMimic, "test_semi_implicit_mimic", test_semi_implicit_mimic, devices=devices)
add_function_test(
    TestSolverMimic,
    "test_semi_implicit_mimic_preserves_kinematic_reference",
    test_semi_implicit_mimic_preserves_kinematic_reference,
    devices=devices,
)
add_function_test(
    TestSolverMimic,
    "test_semi_implicit_mimic_validates_gains",
    test_semi_implicit_mimic_validates_gains,
    devices=devices,
)
add_function_test(TestSolverMimic, "test_vbd_mimic", test_vbd_mimic, devices=devices)
add_function_test(TestSolverMimic, "test_vbd_mimic_shared_reference", test_vbd_mimic_shared_reference, devices=devices)
add_function_test(TestSolverMimic, "test_vbd_mimic_shared_parent", test_vbd_mimic_shared_parent, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
