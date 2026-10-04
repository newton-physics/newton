# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mimic constraint rows in SolverFeatherPGS (matrix-free mode)."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_MIMIC
from newton.tests.unittest_utils import get_test_devices


class TestFeatherPGSMimicLayout(unittest.TestCase):
    def test_reject_quaternion_mimics_at_construction(self):
        """Reject joint-owned mimics whose position and velocity layouts differ."""
        for device in get_test_devices():
            for joint_type in (newton.JointType.BALL, newton.JointType.FREE, newton.JointType.DISTANCE):
                with self.subTest(device=device, joint_type=joint_type):
                    builder, _, follower = _build_mimic_layout(joint_type)
                    model = builder.finalize(device=device)
                    q_start = model.joint_q_start.numpy()
                    qd_start = model.joint_qd_start.numpy()
                    self.assertNotEqual(
                        q_start[follower + 1] - q_start[follower], qd_start[follower + 1] - qd_start[follower]
                    )
                    with self.assertRaisesRegex(ValueError, "joint-owned mimic.*position and velocity"):
                        newton.solvers.SolverFeatherPGS(model, pgs_mode="split")

    def test_scalar_and_d6_mimic_coordinate_maps(self):
        """Preserve componentwise mimic maps after a quaternion-layout free base."""
        for device in get_test_devices():
            for joint_type, dimensions in (
                (newton.JointType.REVOLUTE, 1),
                (newton.JointType.PRISMATIC, 1),
                (newton.JointType.D6, 6),
            ):
                with self.subTest(device=device, joint_type=joint_type):
                    builder, leader, follower = _build_mimic_layout(joint_type)
                    model = builder.finalize(device=device)
                    solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="split")
                    self.assertEqual(solver._mimic_count, dimensions)
                    for suffix, start, joint in (
                        ("q0", model.joint_q_start, follower),
                        ("q1", model.joint_q_start, leader),
                        ("dof0", model.joint_qd_start, follower),
                        ("dof1", model.joint_qd_start, leader),
                    ):
                        expected = np.arange(int(start.numpy()[joint]), int(start.numpy()[joint]) + dimensions)
                        np.testing.assert_array_equal(getattr(solver, f"_mimic_{suffix}").numpy(), expected)
                    np.testing.assert_array_equal(solver._mimic_valid_np, np.ones(dimensions))

    def test_invalid_legacy_mimic_keeps_precedence(self):
        """Preserve masking of a legacy quaternion mimic over its joint-owned entry."""
        for device in get_test_devices():
            with self.subTest(device=device):
                builder, leader, follower = _build_mimic_layout(newton.JointType.BALL)
                with self.assertWarns(DeprecationWarning):
                    builder.add_constraint_mimic(joint0=follower, joint1=leader)
                model = builder.finalize(device=device)
                with self.assertWarnsRegex(UserWarning, "references a non-1-DoF"):
                    solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="split")
                self.assertEqual(solver._mimic_count, 1)
                self.assertEqual(solver._mimic_legacy.numpy().tolist(), [0])
                self.assertEqual(solver._mimic_valid_np.tolist(), [0])


def _build_mimic_layout(joint_type):
    """Build a joint-owned mimic pair after a free base in one articulation."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [builder.add_joint_free(child=base)]
    parent = base
    for _ in range(2):
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        linear = []
        angular = []
        if joint_type == newton.JointType.PRISMATIC:
            linear = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)]
        elif joint_type == newton.JointType.REVOLUTE:
            angular = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z)]
        elif joint_type == newton.JointType.D6:
            linear = [
                newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
            ]
            angular = [
                newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
            ]
        if joint_type == newton.JointType.BALL:
            joint = builder.add_joint_ball(parent=parent, child=child)
        elif joint_type == newton.JointType.FREE:
            joint = builder.add_joint_free(parent=parent, child=child)
        elif joint_type == newton.JointType.DISTANCE:
            joint = builder.add_joint_distance(parent=parent, child=child)
        else:
            joint = builder.add_joint(joint_type, parent=parent, child=child, linear_axes=linear, angular_axes=angular)
        joints.append(joint)
        parent = child
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[2], joints[1], coeffs=(0.1, -0.5))
    return builder, joints[1], joints[2]


def _build_two_revolute_chain(coef0: float, coef1: float, legacy: bool = False):
    """Build a fixed-base chain of two revolute Z-joints with a mimic between them.

    The leader joint is position-driven; the follower joint has no drive and no spring,
    so any tracking it does comes from the mimic constraint alone.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    # add_link (not add_body): add_body eagerly wraps each body in its own
    # single-body free-joint articulation, which would split this chain.
    link_a = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_a, hx=0.1, hy=0.02, hz=0.02)
    j_leader = b.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    link_b = b.add_link(xform=wp.transform(wp.vec3(0.6, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_b, hx=0.1, hy=0.02, hz=0.02)
    j_follower = b.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    b.add_articulation([j_leader, j_follower], label="mimic_chain")
    # Drive the leader only.
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    # follower: q_follower = coef0 + coef1 * q_leader
    if legacy:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=coef0, coef1=coef1)
    else:
        b.set_joint_mimic(j_follower, j_leader, coeffs=(coef0, coef1))
    return b, j_leader, j_follower


def _run_chain(
    coef0: float, coef1: float, leader_target: float, steps: int = 600, legacy: bool = False, **solver_kwargs
):
    builder, _, _ = _build_two_revolute_chain(coef0, coef1, legacy=legacy)
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1, **solver_kwargs
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = leader_target
    control.joint_target_q.assign(targets)
    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return solver, state_0.joint_q.numpy()


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSMimic(unittest.TestCase):
    def test_identity_mimic_tracks_leader(self):
        """Verify a 1:1 mimic makes the undriven follower joint track the driven leader."""
        _, q = _run_chain(coef0=0.0, coef1=1.0, leader_target=0.5)
        self.assertAlmostEqual(q[0], 0.5, delta=0.05)
        self.assertAlmostEqual(q[1], q[0], delta=0.02)

    def test_legacy_constraint_mimic_tracks_leader(self):
        """Verify the deprecated sparse mimic constraints still couple the follower."""
        _, q = _run_chain(coef0=0.1, coef1=-0.5, leader_target=0.6, legacy=True)
        self.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)

    def test_legacy_constraint_overrides_joint_mimic(self):
        """Verify a legacy constraint on the same follower replaces its joint-owned mimic, as in SolverMuJoCo."""
        builder, j_leader, j_follower = _build_two_revolute_chain(coef0=0.0, coef1=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=0.1, coef1=-0.5)
        solver = newton.solvers.SolverFeatherPGS(builder.finalize(), pgs_mode="matrix_free")
        self.assertEqual(solver._mimic_count, 1)
        self.assertEqual(solver._mimic_legacy.numpy().tolist(), [0])

    def test_scaled_offset_mimic(self):
        """Verify q_follower converges to coef0 + coef1 * q_leader for a scaled, offset mimic."""
        _, q = _run_chain(coef0=0.1, coef1=-0.5, leader_target=0.6)
        self.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)

    def test_mimic_row_is_assembled(self):
        """Verify the solver assembles a MIMIC constraint row that carries the coupling."""
        solver, _ = _run_chain(coef0=0.0, coef1=1.0, leader_target=0.5, steps=10)
        row_types = solver.row_type.numpy()
        counts = solver.constraint_count.numpy()
        rows = [int(t) for w in range(row_types.shape[0]) for t in row_types[w, : counts[w]]]
        self.assertIn(PGS_CONSTRAINT_TYPE_MIMIC, rows)

    def test_replicated_mimics_have_articulation_local_ranges(self):
        """Verify replicated mimic rows use one compact lookup range per articulation."""
        template, _, _ = _build_two_revolute_chain(coef0=0.0, coef1=1.0)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(template, world_count=4)
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free")

        self.assertEqual(solver._mimic_art_start.numpy().tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(solver._mimic_art_list.numpy().tolist(), [0, 1, 2, 3])

    def test_disabled_mimic_is_ignored(self):
        """Verify a disabled legacy mimic constraint leaves the follower joint uncoupled."""
        builder, _, _ = _build_two_revolute_chain(0.0, 1.0, legacy=True)
        builder.constraint_mimic_enabled[0] = False
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.5
        control.joint_target_q.assign(targets)
        dt = 1.0 / 240.0
        for _ in range(600):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
        q = state_0.joint_q.numpy()
        self.assertAlmostEqual(q[0], 0.5, delta=0.05)
        # The undriven follower swings free under gravity; it must NOT sit at the leader angle.
        self.assertNotAlmostEqual(q[1], q[0], delta=0.02)

    def test_mimic_only_rows_size_propagation_dense_capacity(self):
        """Verify mimic rows alone reserve propagation dense capacity when there are no limit or drive rows."""
        mimic_count = 20
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        joints = []
        parent = -1
        for i in range(mimic_count + 1):
            link = builder.add_link(xform=wp.transform(wp.vec3(0.2 * (i + 1), 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=wp.vec3(0.0, 0.0, 1.0),
                    parent_xform=wp.transform(
                        wp.vec3(0.2, 0.0, 0.0) if parent >= 0 else wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()
                    ),
                    child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
                )
            )
            parent = link
        builder.add_articulation(joints)
        for follower in joints[1:]:
            builder.set_joint_mimic(follower, joints[0], coeffs=(0.0, 1.0))
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        scene.replicate(builder, world_count=2)  # per-world row counts exclude the global world
        solver = newton.solvers.SolverFeatherPGS(
            scene.finalize(),
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
            propagation_same_articulation_rows=True,
            dense_max_constraints=64,
            enable_joint_limits=False,
        )
        self.assertEqual(solver._mimic_count, 2 * mimic_count)
        self.assertGreaterEqual(solver._dense_internal_max_rows, mimic_count)
        self.assertGreaterEqual(solver.dense_max_constraints, mimic_count)


if __name__ == "__main__":
    unittest.main()
