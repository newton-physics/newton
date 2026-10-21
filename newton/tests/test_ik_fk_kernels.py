# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton import JointType
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices

# -----------------------------------------------------------------------------
# Joint types we want to hit
# -----------------------------------------------------------------------------
JOINT_KINDS: list[int] = [
    JointType.REVOLUTE,
    JointType.PRISMATIC,
    JointType.BALL,
    JointType.D6,
    JointType.FREE,
]


# -----------------------------------------------------------------------------
# Dummy (no-op) objective, gives R = 1 so IKSolver factory doesn't generate a 0xC solver
# -----------------------------------------------------------------------------


class _NoopObjective(ik.IKObjective):
    def residual_dim(self):
        return 1

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        return

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        return

    # keep analytic path trivial too
    def supports_analytic(self):
        return True

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        return


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _add_single_joint(builder: newton.ModelBuilder, jt: int) -> None:
    """
    Adds one body and connects it to the world (-1) with the joint-type `jt`.
    Makes every axis a plain JointDofConfig so the helper works for *all*
    joint kinds we care about in the parity tests.
    """
    cfg = newton.ModelBuilder.JointDofConfig  # alias
    parent_xf = wp.transform((0.1, 0.2, 0.3), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.0))
    child_xf = wp.transform((-0.05, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.5))

    # a 0.1-kg cube just so the body exists
    child = builder.add_link(
        xform=wp.transform_identity(),
        mass=0.1,
        label=f"body_{jt}",
    )
    builder.add_shape_box(
        body=child,
        xform=wp.transform_identity(),
        hx=0.05,
        hy=0.05,
        hz=0.05,
    )

    ji = builder.joint_count

    if jt == JointType.REVOLUTE:
        builder.add_joint_revolute(
            parent=-1,
            child=child,
            parent_xform=parent_xf,
            child_xform=child_xf,
            axis=[0.0, 0.0, 1.0],
        )

    elif jt == JointType.PRISMATIC:
        builder.add_joint_prismatic(
            parent=-1,
            child=child,
            parent_xform=parent_xf,
            child_xform=child_xf,
            axis=[1.0, 0.0, 0.0],
        )

    elif jt == JointType.BALL:
        builder.add_joint_ball(
            parent=-1,
            child=child,
            parent_xform=parent_xf,
            child_xform=child_xf,
        )

    elif jt == JointType.D6:
        builder.add_joint_d6(
            -1,
            child,
            linear_axes=[
                cfg(axis=newton.Axis.X),
                cfg(axis=newton.Axis.Y),
                cfg(axis=newton.Axis.Z),
            ],
            angular_axes=[
                cfg(axis=[1, 0, 0]),
                cfg(axis=[0, 1, 0]),
                cfg(axis=[0, 0, 1]),
            ],
            parent_xform=parent_xf,
            child_xform=child_xf,
        )

    elif jt == JointType.FREE:
        builder.add_joint_free(
            parent=-1,
            child=child,
            parent_xform=parent_xf,
            child_xform=child_xf,
        )

    else:
        raise ValueError(f"Unhandled joint type {jt}")

    if ji == builder.joint_count - 1:
        builder.add_articulation([ji])


def _build_model_for_joint(jt: int, device):
    builder = newton.ModelBuilder()
    _add_single_joint(builder, jt)
    model = builder.finalize(device=device, requires_grad=True)
    return model


def _randomize_joint_q(model: newton.Model, seed: int = 0) -> None:
    """In-place randomization of the model's joint coordinates.

    Keeps magnitudes modest so FK stays well-conditioned and never
    violates default limits (±π for angles, ±0.5 m for free-joint
    translation, quaternion re-normalised).
    """
    rng = np.random.default_rng(seed)

    q_np = model.joint_q.numpy()  # view to host buffer
    for j, jt in enumerate(model.joint_type.numpy()):
        q0 = model.joint_q_start.numpy()[j]  # first coord idx

        if jt == JointType.REVOLUTE:
            q_np[q0] = rng.uniform(-np.pi / 2, np.pi / 2)

        elif jt == JointType.PRISMATIC:
            q_np[q0] = rng.uniform(-0.2, 0.2)  # metres

        elif jt == JointType.BALL:
            # random small-angle quaternion
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-8
            angle = rng.uniform(-np.pi / 6, np.pi / 6)
            qw = np.cos(angle / 2.0)
            qv = axis * np.sin(angle / 2.0)
            q_np[q0 : q0 + 4] = (*qv, qw)

        elif jt == newton.JointType.D6:
            # lin X,Y,Z
            q_np[q0 + 0 : q0 + 3] = rng.uniform(-0.1, 0.1, size=3)
            # rot XYZ
            q_np[q0 + 3 : q0 + 6] = rng.uniform(-np.pi / 8, np.pi / 8, size=3)

        elif jt == newton.JointType.FREE:
            # translation
            q_np[q0 + 0 : q0 + 3] = rng.uniform(-0.3, 0.3, size=3)
            # quaternion
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-8
            angle = rng.uniform(-np.pi / 6, np.pi / 6)
            qw = np.cos(angle / 2.0)
            qv = axis * np.sin(angle / 2.0)
            q_np[q0 + 3 : q0 + 7] = (*qv, qw)

    wp.copy(model.joint_q, wp.array(q_np, dtype=wp.float32))


# -----------------------------------------------------------------------------
# Forward-kinematics two-pass vs reference
# -----------------------------------------------------------------------------


def _fk_parity_for_joint(test, device, jt):
    with wp.ScopedDevice(device):
        model = _build_model_for_joint(jt, device)
        _randomize_joint_q(model)

        # reference FK
        state_ref = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_ref)

        # two-pass FK (via IKSolver helper)
        joint_q = model.joint_q.reshape((1, model.joint_coord_count))
        ik_solver = ik.IKSolver(
            model,
            1,
            objectives=[_NoopObjective()],
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
        )
        ik_solver._fk_two_pass(model, joint_q, ik_solver.body_q, ik_solver.X_local, ik_solver.n_problems)

        assert_np_equal(state_ref.body_q.numpy(), ik_solver.body_q.numpy(), tol=1e-6)


def test_fk_two_pass_parity(test, device):
    for jt in JOINT_KINDS:
        _fk_parity_for_joint(test, device, jt)


def test_ik_loop_joint_mapping(test, device, optimizer):
    """Match tree FK and gradients without writing or optimizing loop-joint coordinates."""
    with wp.ScopedDevice(device):
        for reordered in (False, True):
            builder = newton.ModelBuilder()
            bodies = [builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(3)]
            root, left, right = (bodies[2], bodies[0], bodies[1]) if reordered else bodies
            # Cover loop joints both before and after the tree. Their child also
            # has an incoming tree joint, which must own its pose and Jacobian.
            if reordered:
                loop = builder.add_joint_ball(left, right)
            joints = [
                builder.add_joint_revolute(-1, root, axis=newton.Axis.Z),
                builder.add_joint_revolute(
                    root,
                    left,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()),
                ),
                builder.add_joint_revolute(
                    root,
                    right,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform((0.2, 0.0, 0.3), wp.quat_identity()),
                ),
            ]
            builder.add_articulation(joints)
            if not reordered:
                loop = builder.add_joint_ball(left, right)
            model = builder.finalize(device=device, requires_grad=True)
            q_start = model.joint_q_start.numpy()
            qd_start = model.joint_qd_start.numpy()
            q_np = np.tile(model.joint_q.numpy(), (2, 1))
            q_np[:, q_start[joints]] = [[0.2, -0.1, 0.3], [-0.3, 0.2, -0.2]]
            reference = []
            for row in q_np:
                state = model.state()
                newton.eval_fk(model, wp.array(row), model.joint_qd, state)
                reference.append(state.body_q.numpy())
            reference = np.stack(reference)
            targets = wp.array([[0.3, 0.1, 0.5], [0.1, -0.2, 0.6]], dtype=wp.vec3)
            # The analytic rotation Jacobian uses the small-error approximation.
            # Compare derivatives at zero orientation error, with nonzero joint q.
            rotations = wp.array(reference[:, right, 3:].copy(), dtype=wp.vec4)

            def objectives(right=right, targets=targets, rotations=rotations):
                """Create independent objective caches for each solver."""
                result = [ik.IKObjectivePosition(right, wp.vec3(0.0, 0.0, 0.2), targets)]
                # Like test_rotation_jacobian_compare, exercise rotation AD on
                # CUDA: Warp's CPU zero-angle quaternion derivative can be NaN.
                if device.is_cuda:
                    result.append(ik.IKObjectiveRotation(right, wp.quat_identity(), rotations))
                return result

            analytic = ik.IKSolver(
                model, 2, objectives(), optimizer=optimizer, jacobian_mode=ik.IKJacobianType.ANALYTIC
            )
            q_analytic = wp.array(q_np)
            # Extra rows expose the former out-of-bounds write safely, without
            # corrupting the allocator or poisoning a shared CUDA test context.
            guarded_np = np.full((2, model.joint_count + 1, 7), 42.0, dtype=np.float32)
            guarded = wp.array(guarded_np, dtype=wp.transform)
            analytic._fk_two_pass(model, q_analytic, guarded, analytic.X_local, 2)
            np.testing.assert_array_equal(guarded.numpy()[:, model.body_count :], guarded_np[:, model.body_count :])
            np.testing.assert_allclose(guarded.numpy()[:, : model.body_count], reference, atol=1.0e-6)

            analytic._impl._compute_residuals(q_analytic)
            if optimizer == ik.IKOptimizer.LM:
                derivative_analytic = analytic._impl._jacobian_at(analytic._impl._ctx_solver(q_analytic)).numpy()
            else:
                gradient = wp.zeros((2, model.joint_dof_count), dtype=float)
                analytic._impl._gradient_at(analytic._impl._ctx_solver(q_analytic), gradient)
                derivative_analytic = gradient.numpy()
            loop_dofs = slice(qd_start[loop], qd_start[loop + 1])
            np.testing.assert_array_equal(derivative_analytic[..., loop_dofs], 0.0)
            test.assertGreater(np.max(np.abs(derivative_analytic)), 1.0e-4)

            for mode in (ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED):
                other = ik.IKSolver(model, 2, objectives(), optimizer=optimizer, jacobian_mode=mode)
                q_other = wp.array(q_np, requires_grad=True)
                other._impl._compute_residuals(q_other)
                np.testing.assert_allclose(other.body_q.numpy(), reference, atol=1.0e-6)
                np.testing.assert_allclose(other.residuals.numpy(), analytic.residuals.numpy(), atol=1.0e-6)
                if optimizer == ik.IKOptimizer.LM:
                    derivative_other = other._impl._jacobian_at(other._impl._ctx_solver(q_other)).numpy()
                else:
                    gradient = wp.zeros((2, model.joint_dof_count), dtype=float)
                    other._impl._gradient_at(other._impl._ctx_solver(q_other), gradient)
                    derivative_other = gradient.numpy()
                np.testing.assert_allclose(derivative_analytic, derivative_other, atol=1.0e-5)

            initial_cost = np.sum(analytic.residuals.numpy() ** 2, axis=1)
            result = wp.empty_like(q_analytic)
            analytic.step(q_analytic, result, iterations=4)
            test.assertTrue(np.isfinite(result.numpy()).all())
            np.testing.assert_array_equal(
                result.numpy()[:, q_start[loop] : q_start[loop + 1]], q_np[:, q_start[loop] : q_start[loop + 1]]
            )
            analytic._impl._compute_residuals(result)
            test.assertTrue(np.all(np.sum(analytic.residuals.numpy() ** 2, axis=1) < initial_cost))
            if device.is_cuda:
                with wp.ScopedCapture() as capture:
                    analytic.step(q_analytic, result, iterations=4)
                wp.capture_launch(capture.graph)
                expected = result.numpy().copy()
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(result.numpy(), expected, atol=1.0e-6)


# -----------------------------------------------------------------------------
# Register tests
# -----------------------------------------------------------------------------

devices = get_test_devices()


class TestIKFKKernels(unittest.TestCase):
    pass


add_function_test(TestIKFKKernels, "test_fk_two_pass_parity", test_fk_two_pass_parity, devices)
for optimizer in (ik.IKOptimizer.LM, ik.IKOptimizer.LBFGS):
    add_function_test(
        TestIKFKKernels,
        f"test_ik_loop_joint_mapping_{optimizer.value}",
        test_ik_loop_joint_mapping,
        devices,
        optimizer=optimizer,
    )

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
