# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for joint equality constraints verified with simulation steps."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton._src.solvers.mujoco.equality import _add_equality_constraint
from newton._src.solvers.mujoco.solver_mujoco import HINGE_CONNECT_AXIS_OFFSET
from newton.solvers import SolverMuJoCo


class Sim:
    """Holds the simulation objects for a single test."""

    def __init__(self, model, solver, state_in, state_out, control):
        self.model = model
        self.solver = solver
        self.state_in = state_in
        self.state_out = state_out
        self.control = control


def connect_residual(body_poses, connect_body_indices, leafbody1_anchor, leafbody2_anchor):
    """Compute the world-space residual of a CONNECT constraint.

    Transforms anchor points on leafbody1 and leafbody2 (in their respective
    body frames) to world space using the body poses, then returns the
    distance between them.

    Args:
        body_poses: Array of body transforms (from ``state.body_q.numpy()``).
        connect_body_indices: ``[leafbody1_index, leafbody2_index]`` into
            ``body_poses``.
        leafbody1_anchor: Anchor on leafbody1 in leafbody1's local frame
            (``wp.vec3``).
        leafbody2_anchor: Anchor on leafbody2 in leafbody2's local frame
            (``wp.vec3``).

    Returns:
        Euclidean distance between the two world-space anchor points.
    """
    leafbody1 = connect_body_indices[0]
    leafbody2 = connect_body_indices[1]
    bq1 = body_poses[leafbody1]
    bq2 = body_poses[leafbody2]
    T1 = wp.transform(wp.vec3(bq1[0], bq1[1], bq1[2]), wp.quat(bq1[3], bq1[4], bq1[5], bq1[6]))
    T2 = wp.transform(wp.vec3(bq2[0], bq2[1], bq2[2]), wp.quat(bq2[3], bq2[4], bq2[5], bq2[6]))
    P1 = wp.transform_get_translation(T1) + wp.quat_rotate(wp.transform_get_rotation(T1), leafbody1_anchor)
    P2 = wp.transform_get_translation(T2) + wp.quat_rotate(wp.transform_get_rotation(T2), leafbody2_anchor)
    return float(wp.length(P1 - P2))


class TestEqualityConstraintWithSimStepBase:
    def _create_solver(self, model):
        raise NotImplementedError

    def _num_worlds(self):
        raise NotImplementedError

    def _use_mujoco_cpu(self):
        raise NotImplementedError


class TestConnectConstraintWithSimStepBase(TestEqualityConstraintWithSimStepBase):
    """Test that a CONNECT equality constraint pins two bodies at a point."""

    def _build_connect_model(
        self,
        connect_body_indices: list[int],
        connect_anchor_leafbody1: list[list[float]],
        joint_types: list[str],
        joint_axes: list[int],
        joint_dof_refs: list[list[float]],
        num_worlds: int,
    ):
        """Build a 5-body articulation with a CONNECT constraint.

        Creates a fixed root (root_link), a ball-joint body (ball_link),
        an intermediate body (link0) connected by a high-armature joint,
        and two leaf bodies (leafbody1, leafbody2) connected to link0.
        A CONNECT constraint ties leafbody1 and leafbody2 at an anchor point.

        ``joint_types``, ``joint_axes``, and ``joint_dof_refs`` each have
        length 3: index 0 is for the joint from ball_link to link0,
        indices 1 and 2 are for the two leaf-body joints.
        The fixed root joint and ball joint are implicit.

        Args:
            connect_body_indices: Body indices ``[leafbody1, leafbody2]`` for the
                CONNECT constraint.
            connect_anchor_leafbody1: Anchor on leafbody1 per world as
                ``[[x, y, z], ...]`` [m].
            joint_types: Joint type per non-root joint, length 3. Each is
                ``"revolute"`` or ``"prismatic"``.
            joint_axes: Motion axis per non-root joint, length 3 (0=X, 1=Y, 2=Z).
            joint_dof_refs: Reference position per non-root joint per world,
                shape ``[num_worlds][3]`` [rad or m].
            num_worlds: Number of parallel worlds.

        Returns:
            A :class:`Sim` containing the model, solver, states, and control.
        """
        self.assertEqual(len(joint_types), 3, "joint_types must have 3 elements")
        self.assertEqual(len(joint_axes), 3, "joint_axes must have 3 elements")
        self.assertGreaterEqual(len(joint_dof_refs), num_worlds, "joint_dof_refs must have >= num_worlds rows")
        for row in joint_dof_refs:
            self.assertEqual(len(row), 3, "each joint_dof_refs row must have 3 elements")

        body_inertia = 1.0
        inertia_mat = wp.mat33(
            body_inertia,
            0.0,
            0.0,
            0.0,
            body_inertia,
            0.0,
            0.0,
            0.0,
            body_inertia,
        )

        all_worlds_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=1)

        for w in range(num_worlds):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=1)
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

            # root_link (body index 0 in Newton's list of bodies), fixed joint to world
            root_link = builder.add_link(
                mass=body_inertia,
                inertia=inertia_mat,
            )
            root_joint = builder.add_joint_fixed(parent=-1, child=root_link)

            # ball_link (body index 1 in Newton's list of bodies), ball joint from root_link
            ball_link = builder.add_link(mass=body_inertia, inertia=inertia_mat)
            ball_joint = builder.add_joint_ball(
                parent=root_link,
                child=ball_link,
                armature=1000000000000.0,
            )

            # link0 (body index 2 in Newton's list of bodies), joint0 from ball_link
            link0 = builder.add_link(mass=body_inertia, inertia=inertia_mat)
            if joint_types[0] == "prismatic":
                joint_fn = builder.add_joint_prismatic
            elif joint_types[0] == "revolute":
                joint_fn = builder.add_joint_revolute
            else:
                raise ValueError(f"Unsupported joint_type={joint_types[0]!r}")
            joint0 = joint_fn(
                parent=ball_link,
                child=link0,
                axis=joint_axes[0],
                armature=1000000000000.0,
                custom_attributes={"mujoco:dof_ref": joint_dof_refs[w][0]},
            )

            # leafbody1 (body index 3 in Newton's list of bodies), joint1,
            # leafbody2 (body index 4 in Newton's list of bodies), joint2
            connect_bodies = [None] * 2
            connect_joints = [None] * 2
            connect_joint_types = [joint_types[1], joint_types[2]]
            connect_joint_axes = [joint_axes[1], joint_axes[2]]
            connect_joint_dof_refs = [joint_dof_refs[w][1], joint_dof_refs[w][2]]
            for i in range(2):
                connect_body = builder.add_link(mass=1.0, inertia=inertia_mat, com=wp.vec3(0.0, 0.0, 0.0))

                if connect_joint_types[i] == "prismatic":
                    joint_fn = builder.add_joint_prismatic
                elif connect_joint_types[i] == "revolute":
                    joint_fn = builder.add_joint_revolute
                else:
                    raise ValueError(f"Unsupported joint_type={connect_joint_types[i]!r}")
                connect_joint = joint_fn(
                    axis=connect_joint_axes[i],
                    parent=link0,
                    child=connect_body,
                    armature=0.0,
                    custom_attributes={"mujoco:dof_ref": connect_joint_dof_refs[i]},
                )

                connect_bodies[i] = connect_body
                connect_joints[i] = connect_joint

            all_joints = [root_joint, ball_joint, joint0, connect_joints[0], connect_joints[1]]
            builder.add_articulation(joints=all_joints)

            _add_equality_constraint(
                builder,
                constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
                body1=connect_body_indices[0],
                body2=connect_body_indices[1],
                anchor=connect_anchor_leafbody1[w],
            )

            all_worlds_builder.add_world(builder)

        model = all_worlds_builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        solver = self._create_solver(model)

        return Sim(model, solver, state_in, state_out, control)

    def compute_joint_transform(self, joint_axis: int, joint_pos: float, joint_type: str) -> wp.transform:
        J = wp.transform_identity()
        if joint_type == "prismatic":
            pos = [0.0, 0.0, 0.0]
            pos[joint_axis] = joint_pos
            J = wp.transform(pos, wp.quat_identity())
        elif joint_type == "revolute":
            axes = [wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1)]
            J = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(axes[joint_axis], joint_pos))
        return J

    def compute_expected_leafbody2_anchor(self, joint_axes, ref_joint_q, joint_types, connect_anchor_leafbody1):
        """Compute the expected anchor on leafbody2 for a CONNECT constraint.

        Performs FK at the reference joint coordinates to get world poses of
        leafbody1 and leafbody2, then computes the leafbody2-local anchor
        that coincides with ``connect_anchor_leafbody1`` on leafbody1 in the
        reference configuration.  With SolverMuJoCo's offset coordinates
        (``qpos = joint_q + ref``), the reference pose is the authored pose
        at zero scalar joint coordinates, independent of ``dof_ref``.

        The model topology is: root_link (fixed) -> ball_link (ball) -> link0 (joint0) -> leafbody1 (joint1)
                                                                                      -> leafbody2 (joint2)

        Args:
            joint_axes: Motion axis per non-root joint, length 3 (0=X, 1=Y, 2=Z).
            ref_joint_q: Reference joint coordinates, length 3 [rad or m].
            joint_types: Joint type per non-root joint, length 3.
            connect_anchor_leafbody1: Anchor on leafbody1 as ``[x, y, z]``.

        Returns:
            Expected anchor on leafbody2 as ``wp.vec3``.
        """
        J0 = self.compute_joint_transform(joint_axes[0], ref_joint_q[0], joint_types[0])
        J1 = self.compute_joint_transform(joint_axes[1], ref_joint_q[1], joint_types[1])
        J2 = self.compute_joint_transform(joint_axes[2], ref_joint_q[2], joint_types[2])
        T0 = wp.transform_identity()
        T1 = wp.transform_multiply(T0, J0)
        T2 = wp.transform_multiply(T1, J1)
        T3 = wp.transform_multiply(T1, J2)
        q2 = wp.transform_get_rotation(T2)
        t2 = wp.transform_get_translation(T2)
        q3 = wp.transform_get_rotation(T3)
        t3 = wp.transform_get_translation(T3)
        q = wp.quat_inverse(q3) * q2
        t = wp.quat_rotate(wp.quat_inverse(q3), t2 - t3)
        return wp.quat_rotate(q, wp.vec3(connect_anchor_leafbody1)) + t

    def _test_connect_constraint(self):
        """Verify that the CONNECT constraint brings two separated bodies to the same point.

        Tests multiple anchor positions to exercise the constraint at different
        offsets from the body origin.
        """

        dt = 0.002
        num_steps = 250
        num_worlds = self._num_worlds()
        use_mujoco_cpu = self._use_mujoco_cpu()

        # joint0 can be prismatic or revolute but motion is always along/around Y.
        joint_0_joint_types = ["prismatic", "revolute"]
        num_joint_0_joint_types = len(joint_0_joint_types)
        joint_0_axis = 1

        # Test a range of combinations that, given the test setup,
        # should produce zero residual.
        # Don't test all combinations because that will take too long.
        connect_joint_types_and_axes = [
            ["prismatic", "prismatic", 0, 0],
            ["prismatic", "prismatic", 0, 1],
            ["prismatic", "prismatic", 0, 2],
            ["prismatic", "prismatic", 1, 1],
            ["prismatic", "prismatic", 1, 2],
            ["prismatic", "prismatic", 2, 2],
            ["prismatic", "revolute", 0, 1],
            ["prismatic", "revolute", 0, 2],
            ["prismatic", "revolute", 2, 1],
            ["revolute", "revolute", 0, 1],
            ["revolute", "revolute", 0, 2],
            ["revolute", "revolute", 1, 2],
            ["revolute", "prismatic", 0, 1],
            ["revolute", "prismatic", 0, 2],
            ["revolute", "prismatic", 2, 1],
        ]
        num_connect_joint_types_and_axes = len(connect_joint_types_and_axes)

        connect_body_indices = [3, 4]
        joint_dof_refs = [[0.75, -2.0, 4.0], [0.9, -1.7, 3.5]]
        initial_q = [[0.0, 1.0, 2.0], [0.0, 1.2, 1.7]]
        initial_qd = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        connect_anchor_leafbody1 = [[1.0, 2.0, 3.0], [1.3, 2.4, 2.6]]
        changed_connect_anchor_leafbody1 = [[-1.5, -2.5, -3.5], [-1.8, -2.2, -3.1]]
        changed_joint_dof_refs = [[0.5, -1.0, 2.0], [0.3, -0.8, 1.5]]

        # Newton scalar joint coordinates are offsets from the MuJoCo
        # reference (qpos = joint_q + ref), so the reference pose is at zero
        # scalar coordinates for any dof_ref.
        ref_pose_q = [0.0, 0.0, 0.0]

        # Ball joint identity quaternion coords (x, y, z, w)
        ball_q_identity = [0.0, 0.0, 0.0, 1.0]
        ball_qd_zero = [0.0, 0.0, 0.0]

        flat_ref_joint_q = []
        flat_initial_q = []
        flat_initial_qd = []
        flat_changed_connect_anchor_leafbody1 = []
        flat_original_dof_ref = []
        flat_changed_dof_ref = []
        num_bodies = 5
        # Ball joint adds 4 coords (quaternion) before the 3 joint coords
        ball_q_offset = 4
        for w in range(num_worlds):
            # Ball joint coords (identity quaternion for ref, identity for initial)
            for v in ball_q_identity:
                flat_ref_joint_q.append(v)
                flat_initial_q.append(v)
            # Ball joint DOFs (zero velocity)
            for v in ball_qd_zero:
                flat_initial_qd.append(v)
            for k in range(3):
                flat_ref_joint_q.append(ref_pose_q[k])
                flat_initial_q.append(initial_q[w][k])
                flat_initial_qd.append(initial_qd[w][k])
            for k in range(3):
                flat_changed_connect_anchor_leafbody1.append(changed_connect_anchor_leafbody1[w][k])
            # Ball joint has 3 DOFs, all with ref = 0
            for _ in range(3):
                flat_original_dof_ref.append(0.0)
                flat_changed_dof_ref.append(0.0)
            for k in range(3):
                flat_original_dof_ref.append(joint_dof_refs[w][k])
                flat_changed_dof_ref.append(changed_joint_dof_refs[w][k])

        for i in range(0, num_joint_0_joint_types):
            for j in range(0, num_connect_joint_types_and_axes):
                with self.subTest(joint0=joint_0_joint_types[i], joints=connect_joint_types_and_axes[j]):
                    joint_types = [
                        joint_0_joint_types[i],
                        connect_joint_types_and_axes[j][0],
                        connect_joint_types_and_axes[j][1],
                    ]
                    joint_axes = [joint_0_axis, connect_joint_types_and_axes[j][2], connect_joint_types_and_axes[j][3]]

                    sim = self._build_connect_model(
                        connect_body_indices=connect_body_indices,
                        connect_anchor_leafbody1=connect_anchor_leafbody1,
                        joint_types=joint_types,
                        joint_axes=joint_axes,
                        joint_dof_refs=joint_dof_refs,
                        num_worlds=num_worlds,
                    )

                    for w in range(num_worlds):
                        # Compute the expected anchors.
                        # leafbody1's anchor is the input connect_anchor_leafbody1.
                        # leafbody2's anchor is derived from FK at the reference pose,
                        # which is at zero scalar coordinates regardless of dof_ref.
                        expected_leafbody1_anchor = connect_anchor_leafbody1[w]
                        expected_leafbody2_anchor = self.compute_expected_leafbody2_anchor(
                            joint_axes, ref_pose_q, joint_types, connect_anchor_leafbody1[w]
                        )
                        # Check that the expected anchors match the measured anchors.
                        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                        # eq_data shape is [nworld, neq, 11]; world w, constraint 0
                        measured_leafbody1_anchor = wp.vec3(
                            measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                        )
                        measured_leafbody2_anchor = wp.vec3(
                            measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                        )
                        for k in range(3):
                            self.assertAlmostEqual(
                                float(expected_leafbody1_anchor[k]), float(measured_leafbody1_anchor[k]), places=4
                            )
                            self.assertAlmostEqual(
                                float(expected_leafbody2_anchor[k]), float(measured_leafbody2_anchor[k]), places=4
                            )
                        if use_mujoco_cpu:
                            mj_eq_data = sim.solver.mj_model.eq_data
                            for k in range(3):
                                self.assertAlmostEqual(
                                    float(expected_leafbody1_anchor[k]), float(mj_eq_data[0][k]), places=4
                                )
                                self.assertAlmostEqual(
                                    float(expected_leafbody2_anchor[k]), float(mj_eq_data[0][3 + k]), places=4
                                )

                        # Check that the reference joint positions were applied correctly.
                        # qpos0 shape is [nworld, nq]; world w
                        # First ball_q_offset entries are the ball joint quaternion, then 3 joint coords.
                        measured_dof_refs = sim.solver.mjw_model.qpos0.numpy()[w]
                        expected_dof_refs = joint_dof_refs[w]
                        for k in range(3):
                            self.assertAlmostEqual(
                                float(measured_dof_refs[ball_q_offset + k]), expected_dof_refs[k], places=4
                            )

                    ##############
                    # TEST 1
                    # Set the start state to the reference pose (zero scalar
                    # joint coordinates) to ensure that the start state
                    # satisfies the connect constraint. Nothing should move,
                    # even though the joints have nonzero dof_ref values.
                    ##############

                    sim.state_in.joint_q.assign(flat_ref_joint_q)
                    sim.state_in.joint_qd.assign(flat_initial_qd)

                    for _ in range(num_steps):
                        sim.solver.step(
                            state_in=sim.state_in,
                            state_out=sim.state_out,
                            control=sim.control,
                            dt=dt,
                            contacts=None,
                        )
                        sim.state_in, sim.state_out = sim.state_out, sim.state_in

                    # After N steps, residual should be close to 0
                    # and the joint positions should be unchanged from the
                    # start state because the start state was deliberately
                    # chosen to satisfy the connect constraint.
                    for w in range(num_worlds):
                        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                        measured_leafbody1_anchor = wp.vec3(
                            measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                        )
                        measured_leafbody2_anchor = wp.vec3(
                            measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                        )
                        measured_body_poses = sim.state_in.body_q.numpy()
                        world_body_indices = [
                            w * num_bodies + connect_body_indices[0],
                            w * num_bodies + connect_body_indices[1],
                        ]
                        residual = connect_residual(
                            measured_body_poses,
                            world_body_indices,
                            measured_leafbody1_anchor,
                            measured_leafbody2_anchor,
                        )
                        self.assertAlmostEqual(residual, 0.0, places=4)
                        if use_mujoco_cpu:
                            mj_eq_data = sim.solver.mj_model.eq_data
                            for k in range(6):
                                self.assertAlmostEqual(
                                    float(measured_eq_data[w][0][k]), float(mj_eq_data[0][k]), places=4
                                )

                        measured_joint_q = sim.state_in.joint_q.numpy()
                        nq_per_world = ball_q_offset + 3
                        for k in range(3):
                            self.assertAlmostEqual(
                                measured_joint_q[w * nq_per_world + ball_q_offset + k],
                                flat_ref_joint_q[w * nq_per_world + ball_q_offset + k],
                                places=4,
                            )

                    ##############
                    # TEST 2
                    # Set the start state to differ from the reference joint positions.
                    # The solver will now have to move the bodies to satisfy the
                    # connect constraint.
                    ##############

                    sim.state_in.joint_q.assign(flat_initial_q)
                    sim.state_in.joint_qd.assign(flat_initial_qd)

                    for _ in range(num_steps):
                        sim.solver.step(
                            state_in=sim.state_in,
                            state_out=sim.state_out,
                            control=sim.control,
                            dt=dt,
                            contacts=None,
                        )
                        sim.state_in, sim.state_out = sim.state_out, sim.state_in

                    # After N steps, the residual should be close to 0.
                    # The anchors have not changed so it is correct to continue using measured_leafbody1_anchor, measured_leafbody2_anchor
                    # as the anchors.
                    for w in range(num_worlds):
                        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                        measured_leafbody1_anchor = wp.vec3(
                            measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                        )
                        measured_leafbody2_anchor = wp.vec3(
                            measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                        )
                        measured_body_poses = sim.state_in.body_q.numpy()
                        world_body_indices = [
                            w * num_bodies + connect_body_indices[0],
                            w * num_bodies + connect_body_indices[1],
                        ]
                        residual = connect_residual(
                            measured_body_poses,
                            world_body_indices,
                            measured_leafbody1_anchor,
                            measured_leafbody2_anchor,
                        )
                        self.assertAlmostEqual(residual, 0.0, places=3)
                        if use_mujoco_cpu:
                            mj_eq_data = sim.solver.mj_model.eq_data
                            for k in range(6):
                                self.assertAlmostEqual(
                                    float(measured_eq_data[w][0][k]), float(mj_eq_data[0][k]), places=4
                                )

                    ##############
                    # TEST 3
                    # Change the anchor at runtime and verify the constraint responds
                    # to the new anchor.
                    ##############

                    sim.model.mujoco.equality_constraint_anchor.assign(
                        np.array(flat_changed_connect_anchor_leafbody1, dtype=np.float32)
                    )
                    sim.solver.notify_model_changed(ModelFlags.CONSTRAINT_PROPERTIES)

                    # Verify that mjw_model.eq_data was updated with the new anchor.
                    for w in range(num_worlds):
                        changed_expected_leafbody2_anchor = self.compute_expected_leafbody2_anchor(
                            joint_axes, ref_pose_q, joint_types, changed_connect_anchor_leafbody1[w]
                        )
                        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                        changed_measured_leafbody1_anchor = wp.vec3(
                            measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                        )
                        changed_measured_leafbody2_anchor = wp.vec3(
                            measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                        )
                        for k in range(3):
                            self.assertAlmostEqual(
                                float(changed_connect_anchor_leafbody1[w][k]),
                                float(changed_measured_leafbody1_anchor[k]),
                                places=4,
                            )
                            self.assertAlmostEqual(
                                float(changed_expected_leafbody2_anchor[k]),
                                float(changed_measured_leafbody2_anchor[k]),
                                places=4,
                            )
                        if use_mujoco_cpu:
                            mj_eq_data = sim.solver.mj_model.eq_data
                            for k in range(3):
                                self.assertAlmostEqual(
                                    float(changed_connect_anchor_leafbody1[w][k]), float(mj_eq_data[0][k]), places=4
                                )
                                self.assertAlmostEqual(
                                    float(changed_expected_leafbody2_anchor[k]), float(mj_eq_data[0][3 + k]), places=4
                                )

                    sim.state_in.joint_q.assign(flat_initial_q)
                    sim.state_in.joint_qd.assign(flat_initial_qd)

                    for _ in range(num_steps):
                        sim.solver.step(
                            state_in=sim.state_in,
                            state_out=sim.state_out,
                            control=sim.control,
                            dt=dt,
                            contacts=None,
                        )
                        sim.state_in, sim.state_out = sim.state_out, sim.state_in

                    # After N steps, the residual should be close to 0.
                    for w in range(num_worlds):
                        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                        changed_measured_leafbody1_anchor = wp.vec3(
                            measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                        )
                        changed_measured_leafbody2_anchor = wp.vec3(
                            measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                        )
                        measured_body_poses = sim.state_in.body_q.numpy()
                        world_body_indices = [
                            w * num_bodies + connect_body_indices[0],
                            w * num_bodies + connect_body_indices[1],
                        ]
                        residual = connect_residual(
                            measured_body_poses,
                            world_body_indices,
                            changed_measured_leafbody1_anchor,
                            changed_measured_leafbody2_anchor,
                        )
                        self.assertAlmostEqual(residual, 0.0, places=3)
                        if use_mujoco_cpu:
                            mj_eq_data = sim.solver.mj_model.eq_data
                            for k in range(6):
                                self.assertAlmostEqual(
                                    float(measured_eq_data[w][0][k]), float(mj_eq_data[0][k]), places=4
                                )

                    ##############
                    # TEST 4
                    # Change dof_ref at runtime via JOINT_DOF_PROPERTIES.
                    # dof_ref only relabels the joint coordinates
                    # (qpos = joint_q + ref); it does not move the reference
                    # pose, so the recomputed connect constraint anchors must
                    # be unchanged while qpos0 picks up the new values.
                    ##############

                    sim.model.mujoco.dof_ref.assign(np.array(flat_changed_dof_ref, dtype=np.float32))
                    sim.solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

                    # Verify that the anchors in mjw_model.eq_data are unchanged.
                    for w in range(num_worlds):
                        changed_ref_expected_leafbody2_anchor = self.compute_expected_leafbody2_anchor(
                            joint_axes, ref_pose_q, joint_types, changed_connect_anchor_leafbody1[w]
                        )
                        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                        changed_ref_measured_leafbody1_anchor = wp.vec3(
                            measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                        )
                        changed_ref_measured_leafbody2_anchor = wp.vec3(
                            measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                        )
                        # The 1st anchor is unaffected by the change to reference joint positions.
                        for k in range(3):
                            self.assertAlmostEqual(
                                float(changed_connect_anchor_leafbody1[w][k]),
                                float(changed_ref_measured_leafbody1_anchor[k]),
                                places=4,
                            )
                            self.assertAlmostEqual(
                                float(changed_ref_expected_leafbody2_anchor[k]),
                                float(changed_ref_measured_leafbody2_anchor[k]),
                                places=4,
                            )
                        if use_mujoco_cpu:
                            mj_eq_data = sim.solver.mj_model.eq_data
                            for k in range(3):
                                self.assertAlmostEqual(
                                    float(changed_connect_anchor_leafbody1[w][k]), float(mj_eq_data[0][k]), places=4
                                )
                                self.assertAlmostEqual(
                                    float(changed_ref_expected_leafbody2_anchor[k]),
                                    float(mj_eq_data[0][3 + k]),
                                    places=4,
                                )

                    # Also verify qpos0 was updated with the new dof_ref values.
                    for w in range(num_worlds):
                        measured_dof_refs = sim.solver.mjw_model.qpos0.numpy()[w]
                        for k in range(3):
                            self.assertAlmostEqual(
                                float(measured_dof_refs[ball_q_offset + k]),
                                changed_joint_dof_refs[w][k],
                                places=4,
                            )

                    # The reference pose is still at zero scalar coordinates,
                    # so starting there must satisfy the constraint.
                    sim.state_in.joint_q.assign(flat_ref_joint_q)
                    sim.state_in.joint_qd.assign(flat_initial_qd)

                    for _ in range(num_steps):
                        sim.solver.step(
                            state_in=sim.state_in,
                            state_out=sim.state_out,
                            control=sim.control,
                            dt=dt,
                            contacts=None,
                        )
                        sim.state_in, sim.state_out = sim.state_out, sim.state_in

                    # After N steps, the residual should be close to 0.
                    for w in range(num_worlds):
                        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                        changed_ref_measured_leafbody1_anchor = wp.vec3(
                            measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                        )
                        changed_ref_measured_leafbody2_anchor = wp.vec3(
                            measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                        )
                        measured_body_poses = sim.state_in.body_q.numpy()
                        world_body_indices = [
                            w * num_bodies + connect_body_indices[0],
                            w * num_bodies + connect_body_indices[1],
                        ]
                        residual = connect_residual(
                            measured_body_poses,
                            world_body_indices,
                            changed_ref_measured_leafbody1_anchor,
                            changed_ref_measured_leafbody2_anchor,
                        )
                        self.assertAlmostEqual(residual, 0.0, places=3)
                        if use_mujoco_cpu:
                            mj_eq_data = sim.solver.mj_model.eq_data
                            for k in range(6):
                                self.assertAlmostEqual(
                                    float(measured_eq_data[w][0][k]), float(mj_eq_data[0][k]), places=4
                                )

                    ##############
                    # TEST 5
                    # Restore the original dof_ref via JOINT_PROPERTIES alone
                    # and verify the recomputed connect constraint anchors are
                    # still the reference-pose anchors.  No simulation is run
                    # because JOINT_PROPERTIES does not sync qpos0.
                    ##############

                    sim.model.mujoco.dof_ref.assign(np.array(flat_original_dof_ref, dtype=np.float32))
                    sim.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

                    for w in range(num_worlds):
                        original_ref_expected_leafbody2_anchor = self.compute_expected_leafbody2_anchor(
                            joint_axes, ref_pose_q, joint_types, changed_connect_anchor_leafbody1[w]
                        )
                        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                        original_ref_measured_leafbody1_anchor = wp.vec3(
                            measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                        )
                        original_ref_measured_leafbody2_anchor = wp.vec3(
                            measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                        )
                        for k in range(3):
                            self.assertAlmostEqual(
                                float(changed_connect_anchor_leafbody1[w][k]),
                                float(original_ref_measured_leafbody1_anchor[k]),
                                places=4,
                            )
                            self.assertAlmostEqual(
                                float(original_ref_expected_leafbody2_anchor[k]),
                                float(original_ref_measured_leafbody2_anchor[k]),
                                places=4,
                            )
                        if use_mujoco_cpu:
                            mj_eq_data = sim.solver.mj_model.eq_data
                            for k in range(3):
                                self.assertAlmostEqual(
                                    float(changed_connect_anchor_leafbody1[w][k]), float(mj_eq_data[0][k]), places=4
                                )
                                self.assertAlmostEqual(
                                    float(original_ref_expected_leafbody2_anchor[k]),
                                    float(mj_eq_data[0][3 + k]),
                                    places=4,
                                )

    def test_connect_constraint(self):
        self._test_connect_constraint()


class TestConnectConstraintJointMuJoCoWarp(TestConnectConstraintWithSimStepBase, unittest.TestCase):
    def _num_worlds(self):
        return 2

    def _use_mujoco_cpu(self):
        return False

    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_cpu=False,
            integrator="euler",
        )


class TestConnectConstraintJointMuJoCoCPU(TestConnectConstraintWithSimStepBase, unittest.TestCase):
    def _num_worlds(self):
        return 1

    def _use_mujoco_cpu(self):
        return True

    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_cpu=True,
            separate_worlds=True,
            integrator="euler",
        )


class TestLoopJointConnectConstraintBase(TestEqualityConstraintWithSimStepBase):
    """Test the anchors of loop-joint-synthesized CONNECT constraints.

    Creates a single articulation with a revolute loop joint closing back to
    its root body. The loop joint generates 2 CONNECT constraints in MuJoCo.
    Verifies that the anchors are derived at the authored reference pose,
    stay unchanged when dof_ref changes at runtime (dof_ref only relabels
    the joint coordinates), and are recomputed when joint_X_p changes.
    """

    def _build_loop_joint_model(
        self,
        loop_joint_axis,
        joint0_axis,
        joint1_axis,
        joint0_type,
        joint1_type,
        dof_refs,
        num_worlds,
    ):
        """Build a model with a single articulation and a revolute loop joint.

        Topology per world:
            Articulation: world -> fixed -> root_body -> joint0 -> body_a -> joint1 -> body_b
            Loop joint: revolute from body_b (parent) to root_body (child), not in articulation

        Bodies per world (3 total): root_body(0), body_a(1), body_b(2)
        Joints per world (4 total): root_fixed(0), joint0(1), joint1(2), loop_joint(3)

        Args:
            loop_joint_axis: Axis for the loop revolute joint (0=X, 1=Y, 2=Z).
            joint0_axis: Axis for joint0 (0=X, 1=Y, 2=Z).
            joint1_axis: Axis for joint1 (0=X, 1=Y, 2=Z).
            joint0_type: Type of joint0 (``"revolute"`` or ``"prismatic"``).
            joint1_type: Type of joint1 (``"revolute"`` or ``"prismatic"``).
            dof_refs: Per-world DOF reference values, shape ``[num_worlds][2]``
                (one per articulation joint). The loop joint has no DOF ref.
            num_worlds: Number of worlds.

        Returns:
            A :class:`Sim` containing the model, solver, states, and control.
        """
        body_inertia = 1.0
        inertia_mat = wp.mat33(
            body_inertia,
            0.0,
            0.0,
            0.0,
            body_inertia,
            0.0,
            0.0,
            0.0,
            body_inertia,
        )

        all_worlds_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=1)

        for w in range(num_worlds):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=1)
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

            # root_body (body 0), fixed to world
            root_body = builder.add_link(mass=body_inertia, inertia=inertia_mat)
            root_joint = builder.add_joint_fixed(parent=-1, child=root_body)

            # body_a (body 1), connected to root_body via joint0
            # Use parent_xform offsets so bodies are not co-located at the origin;
            # this ensures all CONNECT constraints are active after mj_forward
            # (needed to avoid a mujoco_warp put_data reshape issue).
            joint0_xform = wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())
            body_a = builder.add_link(mass=body_inertia, inertia=inertia_mat)
            if joint0_type == "prismatic":
                joint0 = builder.add_joint_prismatic(
                    parent=root_body,
                    child=body_a,
                    axis=joint0_axis,
                    parent_xform=joint0_xform,
                    armature=1000000000000.0,
                    custom_attributes={"mujoco:dof_ref": dof_refs[w][0]},
                )
            else:
                joint0 = builder.add_joint_revolute(
                    parent=root_body,
                    child=body_a,
                    axis=joint0_axis,
                    parent_xform=joint0_xform,
                    armature=1000000000000.0,
                    custom_attributes={"mujoco:dof_ref": dof_refs[w][0]},
                )

            # body_b (body 2), connected to body_a via joint1
            joint1_xform = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
            body_b = builder.add_link(mass=body_inertia, inertia=inertia_mat)
            if joint1_type == "prismatic":
                joint1 = builder.add_joint_prismatic(
                    parent=body_a,
                    child=body_b,
                    axis=joint1_axis,
                    parent_xform=joint1_xform,
                    armature=1000000000000.0,
                    custom_attributes={"mujoco:dof_ref": dof_refs[w][1]},
                )
            else:
                joint1 = builder.add_joint_revolute(
                    parent=body_a,
                    child=body_b,
                    axis=joint1_axis,
                    parent_xform=joint1_xform,
                    armature=1000000000000.0,
                    custom_attributes={"mujoco:dof_ref": dof_refs[w][1]},
                )

            builder.add_articulation(joints=[root_joint, joint0, joint1])

            # Loop joint: revolute from body_b (parent) to root_body (child),
            # not added to articulation
            builder.add_joint_revolute(
                parent=body_b,
                child=root_body,
                axis=loop_joint_axis,
                armature=0.0,
            )

            all_worlds_builder.add_world(builder)

        model = all_worlds_builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        solver = self._create_solver(model)

        return Sim(model, solver, state_in, state_out, control)

    def _compute_loop_joint_expected_anchors(
        self,
        joint_X_p_np,
        joint_X_c_np,
        joint_axis_np,
        joint_qd_start_np,
        joint0_idx,
        joint1_idx,
        loop_joint_idx,
    ):
        """Compute expected anchor1 and anchor2 for both CONNECT constraints from a revolute loop joint.

        The anchors are derived at the reference pose.  With SolverMuJoCo's
        offset coordinates (``qpos = joint_q + ref``), the reference pose is
        the authored pose at zero scalar joint coordinates — independent of
        ``dof_ref`` — so all joint transforms are identity in the FK below.

        Args:
            joint_X_p_np: Numpy array of joint parent transforms.
            joint_X_c_np: Numpy array of joint child transforms.
            joint_axis_np: Numpy array of joint axes.
            joint_qd_start_np: Numpy array of joint qd starts.
            joint0_idx: Index of joint0 in the model arrays.
            joint1_idx: Index of joint1 in the model arrays.
            loop_joint_idx: Index of the loop joint in the model arrays.

        Returns:
            Tuple of (anchor1_a, anchor2_a, anchor1_b, anchor2_b) where
            anchor1/anchor2 are for the first and second CONNECT constraints.
        """
        # Compute world poses via FK at the reference pose.
        # Topology: root_body(identity) -> joint0(X_p0) -> body_a -> joint1(X_p1) -> body_b
        # Loop joint: body_b (parent) -> root_body (child)
        # T_child = T_parent * X_p * inv(X_c)

        # Joint 0: T_body_a = T_root * X_p0 * inv(X_c0)
        X_p0 = wp.transform(
            wp.vec3(
                float(joint_X_p_np[joint0_idx][0]),
                float(joint_X_p_np[joint0_idx][1]),
                float(joint_X_p_np[joint0_idx][2]),
            ),
            wp.quat(
                float(joint_X_p_np[joint0_idx][3]),
                float(joint_X_p_np[joint0_idx][4]),
                float(joint_X_p_np[joint0_idx][5]),
                float(joint_X_p_np[joint0_idx][6]),
            ),
        )
        X_c0 = wp.transform(
            wp.vec3(
                float(joint_X_c_np[joint0_idx][0]),
                float(joint_X_c_np[joint0_idx][1]),
                float(joint_X_c_np[joint0_idx][2]),
            ),
            wp.quat(
                float(joint_X_c_np[joint0_idx][3]),
                float(joint_X_c_np[joint0_idx][4]),
                float(joint_X_c_np[joint0_idx][5]),
                float(joint_X_c_np[joint0_idx][6]),
            ),
        )
        # Joint 1: T_body_b = T_body_a * X_p1 * inv(X_c1)
        X_p1 = wp.transform(
            wp.vec3(
                float(joint_X_p_np[joint1_idx][0]),
                float(joint_X_p_np[joint1_idx][1]),
                float(joint_X_p_np[joint1_idx][2]),
            ),
            wp.quat(
                float(joint_X_p_np[joint1_idx][3]),
                float(joint_X_p_np[joint1_idx][4]),
                float(joint_X_p_np[joint1_idx][5]),
                float(joint_X_p_np[joint1_idx][6]),
            ),
        )
        X_c1 = wp.transform(
            wp.vec3(
                float(joint_X_c_np[joint1_idx][0]),
                float(joint_X_c_np[joint1_idx][1]),
                float(joint_X_c_np[joint1_idx][2]),
            ),
            wp.quat(
                float(joint_X_c_np[joint1_idx][3]),
                float(joint_X_c_np[joint1_idx][4]),
                float(joint_X_c_np[joint1_idx][5]),
                float(joint_X_c_np[joint1_idx][6]),
            ),
        )
        T_root = wp.transform_identity()
        X_c0_inv = wp.transform_inverse(X_c0)
        X_c1_inv = wp.transform_inverse(X_c1)
        T_body_a = wp.transform_multiply(wp.transform_multiply(T_root, X_p0), X_c0_inv)
        T_body_b = wp.transform_multiply(wp.transform_multiply(T_body_a, X_p1), X_c1_inv)

        # Get the loop joint's parent transform to extract anchor
        loop_xform = joint_X_p_np[loop_joint_idx]
        parent_anchor = wp.vec3(loop_xform[0], loop_xform[1], loop_xform[2])
        parent_quat = wp.quat(loop_xform[3], loop_xform[4], loop_xform[5], loop_xform[6])

        # Hinge axis in parent body frame (body_b's frame)
        qd_start = int(joint_qd_start_np[loop_joint_idx])
        hinge_axis_local = wp.vec3(
            float(joint_axis_np[qd_start][0]),
            float(joint_axis_np[qd_start][1]),
            float(joint_axis_np[qd_start][2]),
        )
        hinge_axis = wp.quat_rotate(parent_quat, hinge_axis_local)

        # First CONNECT anchor1 = parent_anchor (in body_b frame)
        anchor1_a = parent_anchor
        # Second CONNECT anchor1 = parent_anchor + offset * hinge_axis (in body_b frame)
        d = HINGE_CONNECT_AXIS_OFFSET
        anchor1_b = parent_anchor + wp.vec3(d * hinge_axis[0], d * hinge_axis[1], d * hinge_axis[2])

        # Compute anchor2 using relative transform between parent (body_b) and child (root_body)
        # q_rel = inv(q_child) * q_parent
        # t_rel = quat_rotate(inv(q_child), pos_parent - pos_child)
        q_parent = wp.transform_get_rotation(T_body_b)
        pos_parent = wp.transform_get_translation(T_body_b)
        q_child = wp.transform_get_rotation(T_root)
        pos_child = wp.transform_get_translation(T_root)

        q_child_inv = wp.quat_inverse(q_child)
        q_rel = q_child_inv * q_parent
        t_rel = wp.quat_rotate(q_child_inv, pos_parent - pos_child)

        anchor2_a = wp.quat_rotate(q_rel, anchor1_a) + t_rel
        anchor2_b = wp.quat_rotate(q_rel, anchor1_b) + t_rel

        return anchor1_a, anchor2_a, anchor1_b, anchor2_b

    def _assert_loop_joint_eq_data(self, sim, w, anchor1_a, anchor2_a, anchor1_b, anchor2_b):
        """Assert that measured eq_data anchors match expected values for two CONNECT constraints."""
        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
        measured_a1_0 = wp.vec3(measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2])
        measured_a2_0 = wp.vec3(measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5])
        measured_a1_1 = wp.vec3(measured_eq_data[w][1][0], measured_eq_data[w][1][1], measured_eq_data[w][1][2])
        measured_a2_1 = wp.vec3(measured_eq_data[w][1][3], measured_eq_data[w][1][4], measured_eq_data[w][1][5])

        for k in range(3):
            self.assertAlmostEqual(float(anchor1_a[k]), float(measured_a1_0[k]), places=4)
            self.assertAlmostEqual(float(anchor2_a[k]), float(measured_a2_0[k]), places=4)
            self.assertAlmostEqual(float(anchor1_b[k]), float(measured_a1_1[k]), places=4)
            self.assertAlmostEqual(float(anchor2_b[k]), float(measured_a2_1[k]), places=4)

        # CPU-path: mj_model.eq_data is synced from world 0 only
        if sim.solver.use_mujoco_cpu and w == 0:
            mj_eq_data = sim.solver.mj_model.eq_data
            for k in range(3):
                self.assertAlmostEqual(float(anchor1_a[k]), float(mj_eq_data[0][k]), places=4)
                self.assertAlmostEqual(float(anchor2_a[k]), float(mj_eq_data[0][3 + k]), places=4)
                self.assertAlmostEqual(float(anchor1_b[k]), float(mj_eq_data[1][k]), places=4)
                self.assertAlmostEqual(float(anchor2_b[k]), float(mj_eq_data[1][3 + k]), places=4)

    def _test_loop_joint_connect_constraint(self):
        """Verify loop-joint CONNECT anchors at the reference pose and across runtime updates."""

        num_worlds = 2

        # Test a few joint type combinations
        joint_type_combos = [
            ["revolute", "revolute"],
            ["prismatic", "revolute"],
            ["revolute", "prismatic"],
            ["prismatic", "prismatic"],
        ]
        loop_joint_axis = 2  # Z axis for the loop revolute joint
        joint0_axis = 1  # Y axis for joint0
        joint1_axis = 0  # X axis for joint1

        dof_refs = [[0.5, -0.3], [0.7, -0.5]]
        changed_dof_refs = [[0.2, -0.8], [0.4, -0.6]]

        for combo_idx in range(len(joint_type_combos)):
            joint0_type = joint_type_combos[combo_idx][0]
            joint1_type = joint_type_combos[combo_idx][1]

            with self.subTest(joint0=joint0_type, joint1=joint1_type):
                sim = self._build_loop_joint_model(
                    loop_joint_axis=loop_joint_axis,
                    joint0_axis=joint0_axis,
                    joint1_axis=joint1_axis,
                    joint0_type=joint0_type,
                    joint1_type=joint1_type,
                    dof_refs=dof_refs,
                    num_worlds=num_worlds,
                )

                # 4 joints per world: root_fixed(0), joint0(1), joint1(2), loop_joint(3)
                joints_per_world = 4
                joint_X_p_np = sim.model.joint_X_p.numpy()
                joint_X_c_np = sim.model.joint_X_c.numpy()
                joint_axis_np = sim.model.joint_axis.numpy()
                joint_qd_start_np = sim.model.joint_qd_start.numpy()

                # There should be 2 CONNECT equality constraints from the loop joint
                # (revolute loop joint creates 2 CONNECT constraints)
                neq = sim.solver.mj_model.neq
                self.assertEqual(neq, 2, "Expected 2 CONNECT constraints from revolute loop joint")

                # Verify initial eq_data is correct.  The anchors are derived
                # at the authored reference pose and are independent of the
                # (nonzero) dof_ref values.
                for w in range(num_worlds):
                    loop_joint_idx = w * joints_per_world + 3

                    anchor1_a, anchor2_a, anchor1_b, anchor2_b = self._compute_loop_joint_expected_anchors(
                        joint_X_p_np=joint_X_p_np,
                        joint_X_c_np=joint_X_c_np,
                        joint_axis_np=joint_axis_np,
                        joint_qd_start_np=joint_qd_start_np,
                        joint0_idx=w * joints_per_world + 1,
                        joint1_idx=w * joints_per_world + 2,
                        loop_joint_idx=loop_joint_idx,
                    )

                    self._assert_loop_joint_eq_data(sim, w, anchor1_a, anchor2_a, anchor1_b, anchor2_b)

                ##############
                # TEST: Change dof_ref and verify CONNECT anchors are recomputed
                # and unchanged — dof_ref relabels the joint coordinates but does
                # not move the reference pose.
                ##############

                # Build flat dof_ref array. Per world, the DOF layout in Newton is:
                # joint0 (1 DOF) + joint1 (1 DOF) + loop_joint (1 DOF) = 3 DOFs per world.
                # The loop joint DOF exists in Newton even though it is excluded from MuJoCo's
                # joint list. dof_ref is indexed by Newton DOF, so we must include the loop
                # joint's entry (kept at 0.0).
                flat_changed_dof_ref = []
                for w in range(num_worlds):
                    flat_changed_dof_ref.append(changed_dof_refs[w][0])
                    flat_changed_dof_ref.append(changed_dof_refs[w][1])
                    flat_changed_dof_ref.append(0.0)  # loop joint DOF (unchanged)

                sim.model.mujoco.dof_ref.assign(np.array(flat_changed_dof_ref, dtype=np.float32))
                sim.solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

                # Verify eq_data still holds the reference-pose anchors
                for w in range(num_worlds):
                    loop_joint_idx = w * joints_per_world + 3

                    anchor1_a, anchor2_a, anchor1_b, anchor2_b = self._compute_loop_joint_expected_anchors(
                        joint_X_p_np=joint_X_p_np,
                        joint_X_c_np=joint_X_c_np,
                        joint_axis_np=joint_axis_np,
                        joint_qd_start_np=joint_qd_start_np,
                        joint0_idx=w * joints_per_world + 1,
                        joint1_idx=w * joints_per_world + 2,
                        loop_joint_idx=loop_joint_idx,
                    )

                    self._assert_loop_joint_eq_data(sim, w, anchor1_a, anchor2_a, anchor1_b, anchor2_b)

                ##############
                # TEST: Change joint_X_p of the loop joint and verify CONNECT anchors are recomputed
                ##############

                # Shift the loop joint's parent transform for each world
                joint_X_p_np = sim.model.joint_X_p.numpy()
                for w in range(num_worlds):
                    loop_joint_idx = w * joints_per_world + 3
                    # Apply a per-world translation offset to the loop joint
                    joint_X_p_np[loop_joint_idx][0] += 0.3 + 0.1 * w
                    joint_X_p_np[loop_joint_idx][1] += 0.2
                sim.model.joint_X_p.assign(joint_X_p_np)
                sim.solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

                # Re-read after modification
                joint_X_p_np = sim.model.joint_X_p.numpy()
                joint_axis_np = sim.model.joint_axis.numpy()

                for w in range(num_worlds):
                    loop_joint_idx = w * joints_per_world + 3

                    anchor1_a, anchor2_a, anchor1_b, anchor2_b = self._compute_loop_joint_expected_anchors(
                        joint_X_p_np=joint_X_p_np,
                        joint_X_c_np=joint_X_c_np,
                        joint_axis_np=joint_axis_np,
                        joint_qd_start_np=joint_qd_start_np,
                        joint0_idx=w * joints_per_world + 1,
                        joint1_idx=w * joints_per_world + 2,
                        loop_joint_idx=loop_joint_idx,
                    )

                    self._assert_loop_joint_eq_data(sim, w, anchor1_a, anchor2_a, anchor1_b, anchor2_b)

    def test_loop_joint_connect_constraint(self):
        self._test_loop_joint_connect_constraint()


class TestLoopJointConnectConstraintMuJoCoWarp(TestLoopJointConnectConstraintBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_cpu=False,
            separate_worlds=True,
            njmax=100,
            integrator="euler",
        )


class TestLoopJointConnectConstraintMuJoCoCPU(TestLoopJointConnectConstraintBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_cpu=True,
            separate_worlds=True,
            integrator="euler",
        )


class TestMixedWeldAndConnectLoopJointBase(TestEqualityConstraintWithSimStepBase):
    """Test that WELD (FIXED) loop joint eq_data is not corrupted by CONNECT kernel updates.

    Creates a model with both a revolute loop joint (2 CONNECT constraints) and
    a FIXED loop joint (1 WELD constraint).  Verifies that after
    ``notify_model_changed(JOINT_DOF_PROPERTIES)`` the WELD constraint's
    ``eq_data`` retains its anchor and relpose values.
    """

    def _build_mixed_weld_and_connect_model(self, num_worlds):
        """Build a model with a revolute loop joint and a FIXED loop joint.

        Topology per world:
            Articulation: world -> fixed -> root_body -> rev_joint -> body_a -> rev_joint2 -> body_b
            Revolute loop joint: body_b (parent) -> root_body (child), not in articulation
            Fixed loop joint: body_a (parent) -> root_body (child), not in articulation

        The revolute loop joint creates 2 CONNECT constraints.
        The fixed loop joint creates 1 WELD constraint.
        Total: 3 MuJoCo equality constraints per model.

        Args:
            num_worlds: Number of worlds.

        Returns:
            A :class:`Sim` containing the model, solver, states, and control.
        """
        body_inertia = 1.0
        inertia_mat = wp.mat33(
            body_inertia,
            0.0,
            0.0,
            0.0,
            body_inertia,
            0.0,
            0.0,
            0.0,
            body_inertia,
        )

        all_worlds_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=1)

        for _w in range(num_worlds):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=1)
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

            # root_body (body 0), fixed to world
            root_body = builder.add_link(mass=body_inertia, inertia=inertia_mat)
            root_joint = builder.add_joint_fixed(parent=-1, child=root_body)

            # body_a (body 1), connected to root_body via revolute joint
            joint0_xform = wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())
            body_a = builder.add_link(mass=body_inertia, inertia=inertia_mat)
            joint0 = builder.add_joint_revolute(
                parent=root_body,
                child=body_a,
                axis=1,  # Y axis
                parent_xform=joint0_xform,
                armature=1000000000000.0,
                custom_attributes={"mujoco:dof_ref": 0.5},
            )

            # body_b (body 2), connected to body_a via revolute joint
            joint1_xform = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
            body_b = builder.add_link(mass=body_inertia, inertia=inertia_mat)
            joint1 = builder.add_joint_revolute(
                parent=body_a,
                child=body_b,
                axis=0,  # X axis
                parent_xform=joint1_xform,
                armature=1000000000000.0,
                custom_attributes={"mujoco:dof_ref": -0.3},
            )

            builder.add_articulation(joints=[root_joint, joint0, joint1])

            # Revolute loop joint: body_b (parent) -> root_body (child)
            # Creates 2 CONNECT constraints
            builder.add_joint_revolute(
                parent=body_b,
                child=root_body,
                axis=2,  # Z axis
                armature=0.0,
            )

            # FIXED loop joint: body_a (parent) -> root_body (child)
            # Creates 1 WELD constraint
            builder.add_joint_fixed(
                parent=body_a,
                child=root_body,
                parent_xform=wp.transform(wp.vec3(0.0, 0.2, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.1, 0.0), wp.quat_identity()),
            )

            all_worlds_builder.add_world(builder)

        model = all_worlds_builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        solver = self._create_solver(model)

        return Sim(model, solver, state_in, state_out, control)

    def test_weld_eq_data_not_corrupted_by_connect_update(self):
        """Verify WELD eq_data matches MuJoCo ground truth and is not overwritten by CONNECT kernels.

        The CONNECT kernels launched by ``_notify_connect_constraints_changed``
        must skip WELD entries.  This test compares ``mjw_model.eq_data`` for
        the WELD constraint against the ground truth computed by MuJoCo's
        ``spec.compile()`` (stored in ``mj_model.eq_data``).  The corruption
        happens at init time (during ``notify_model_changed(ALL)``), so a
        before-vs-after comparison would not catch it.
        """
        num_worlds = 2
        sim = self._build_mixed_weld_and_connect_model(num_worlds)

        import mujoco

        # The revolute loop joint creates 2 CONNECT, the FIXED creates 1 WELD = 3 total
        neq = sim.solver.mj_model.neq
        self.assertEqual(neq, 3, "Expected 3 equality constraints (2 CONNECT + 1 WELD)")

        eq_types = sim.solver.mj_model.eq_type
        connect_type = int(mujoco.mjtEq.mjEQ_CONNECT)
        weld_type = int(mujoco.mjtEq.mjEQ_WELD)
        self.assertEqual(int(eq_types[0]), connect_type)
        self.assertEqual(int(eq_types[1]), connect_type)
        self.assertEqual(int(eq_types[2]), weld_type)

        weld_eq_idx = 2

        # Ground truth: mj_model.eq_data is set by spec.compile() and is not
        # modified by GPU kernel launches. Use it as the reference for the WELD relpose.
        expected_weld_data = np.array(sim.solver.mj_model.eq_data[weld_eq_idx], dtype=np.float32)

        # The WELD relpose translation (data[3:6]) must be non-trivial
        # because the parent/child xforms have different offsets.
        self.assertFalse(
            np.allclose(expected_weld_data[3:6], 0.0, atol=1e-10),
            f"WELD relpose translation should be non-zero, got {expected_weld_data[3:6]}",
        )

        # Check that mjw_model.eq_data matches the ground truth for all worlds
        mjw_eq_data = sim.solver.mjw_model.eq_data.numpy()
        for w in range(num_worlds):
            np.testing.assert_allclose(
                mjw_eq_data[w, weld_eq_idx, :],
                expected_weld_data,
                atol=1e-5,
                err_msg=(
                    f"World {w}: WELD eq_data in mjw_model does not match "
                    f"MuJoCo ground truth — CONNECT kernels likely overwrote it"
                ),
            )


class TestMixedWeldAndConnectMuJoCoWarp(TestMixedWeldAndConnectLoopJointBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_cpu=False,
            separate_worlds=True,
            njmax=100,
            integrator="euler",
        )


class TestMixedWeldAndConnectMuJoCoCPU(TestMixedWeldAndConnectLoopJointBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_cpu=True,
            separate_worlds=True,
            integrator="euler",
        )


class TestConnectAnchorRefPoseBase(TestEqualityConstraintWithSimStepBase):
    """Regression test: CONNECT anchors must be derived at the reference pose.

    With SolverMuJoCo's offset coordinates (``qpos = joint_q + ref``), the
    authored body poses correspond to zero scalar joint coordinates, so
    nonzero ``mujoco:dof_ref`` values must not displace the reference pose
    used to derive the CONNECT ``anchor2``.  The joint frames carry both
    translation and rotation so that ``anchor2`` differs nontrivially from
    ``anchor1`` at the reference pose.
    """

    def _joint_xforms(self):
        """Parent-frame joint transforms; the rotations make ``q_rel`` nontrivial."""
        joint0_xform = wp.transform(
            wp.vec3(1.0, 0.0, 0.5),
            wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4),
        )
        joint1_xform = wp.transform(
            wp.vec3(0.0, 0.0, 1.0),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.3),
        )
        joint2_xform = wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity())
        return joint0_xform, joint1_xform, joint2_xform

    def _build_model(self, anchor_body_b, dof_refs, num_worlds):
        """Build two branches from a fixed root, closed by a CONNECT constraint.

        Topology per world:
            root_body -> joint0 (hinge) -> body_a -> joint1 (hinge) -> body_b
            root_body -> joint2 (hinge) -> body_c
            CONNECT: body_b (``anchor_body_b``) to body_c

        All three hinges carry nonzero ``mujoco:dof_ref`` values.

        Args:
            anchor_body_b: CONNECT anchor on body_b as ``[x, y, z]`` [m].
            dof_refs: Per-world reference values, shape ``[num_worlds][3]``.
            num_worlds: Number of parallel worlds.

        Returns:
            A :class:`Sim` containing the model, solver, states, and control.
        """
        inertia_mat = wp.mat33(np.eye(3))
        joint0_xform, joint1_xform, joint2_xform = self._joint_xforms()

        all_worlds_builder = newton.ModelBuilder(gravity=0.0, up_axis=1)
        for w in range(num_worlds):
            builder = newton.ModelBuilder(gravity=0.0, up_axis=1)
            newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

            root_body = builder.add_link(mass=1.0, inertia=inertia_mat)
            root_joint = builder.add_joint_fixed(parent=-1, child=root_body)

            body_a = builder.add_link(mass=1.0, inertia=inertia_mat)
            joint0 = builder.add_joint_revolute(
                parent=root_body,
                child=body_a,
                axis=1,
                parent_xform=joint0_xform,
                custom_attributes={"mujoco:dof_ref": dof_refs[w][0]},
            )

            body_b = builder.add_link(mass=1.0, inertia=inertia_mat)
            joint1 = builder.add_joint_revolute(
                parent=body_a,
                child=body_b,
                axis=0,
                parent_xform=joint1_xform,
                custom_attributes={"mujoco:dof_ref": dof_refs[w][1]},
            )

            body_c = builder.add_link(mass=1.0, inertia=inertia_mat)
            joint2 = builder.add_joint_revolute(
                parent=root_body,
                child=body_c,
                axis=2,
                parent_xform=joint2_xform,
                custom_attributes={"mujoco:dof_ref": dof_refs[w][2]},
            )

            builder.add_articulation(joints=[root_joint, joint0, joint1, joint2])

            _add_equality_constraint(
                builder,
                constraint_type=newton.solvers.SolverMuJoCo.EqType.CONNECT,
                body1=body_b,
                body2=body_c,
                anchor=anchor_body_b,
            )

            all_worlds_builder.add_world(builder)

        model = all_worlds_builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        solver = self._create_solver(model)

        return Sim(model, solver, state_in, state_out, control)

    def _expected_anchor2(self, anchor_body_b):
        """Compute anchor2 at the reference pose (identity joint transforms)."""
        joint0_xform, joint1_xform, joint2_xform = self._joint_xforms()
        T_body_b = wp.transform_multiply(joint0_xform, joint1_xform)
        T_body_c = joint2_xform

        q_b = wp.transform_get_rotation(T_body_b)
        p_b = wp.transform_get_translation(T_body_b)
        q_c = wp.transform_get_rotation(T_body_c)
        p_c = wp.transform_get_translation(T_body_c)

        q_c_inv = wp.quat_inverse(q_c)
        q_rel = q_c_inv * q_b
        t_rel = wp.quat_rotate(q_c_inv, p_b - p_c)
        return wp.quat_rotate(q_rel, wp.vec3(anchor_body_b)) + t_rel

    def test_connect_anchors_at_reference_pose(self):
        dt = 0.002
        num_steps = 200
        num_worlds = self._num_worlds()
        use_mujoco_cpu = self._use_mujoco_cpu()

        anchor_body_b = [0.2, 0.3, 0.1]
        dof_refs = [[0.6, -0.4, 0.8], [0.9, 0.3, -0.5]]

        sim = self._build_model(anchor_body_b, dof_refs, num_worlds)
        expected_anchor2 = self._expected_anchor2(anchor_body_b)

        # The anchors are derived at the authored reference pose and must be
        # identical across worlds even though the dof_ref values differ.
        measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
        for w in range(num_worlds):
            for k in range(3):
                self.assertAlmostEqual(float(measured_eq_data[w][0][k]), anchor_body_b[k], places=4)
                self.assertAlmostEqual(float(measured_eq_data[w][0][3 + k]), float(expected_anchor2[k]), places=4)
        if use_mujoco_cpu:
            mj_eq_data = sim.solver.mj_model.eq_data
            for k in range(3):
                self.assertAlmostEqual(float(mj_eq_data[0][k]), anchor_body_b[k], places=4)
                self.assertAlmostEqual(float(mj_eq_data[0][3 + k]), float(expected_anchor2[k]), places=4)

        # Start at the reference pose (zero scalar joint coordinates). The
        # constraint is satisfied there, so nothing should move even though
        # every joint has a nonzero dof_ref.
        nq_per_world = 3
        sim.state_in.joint_q.assign([0.0] * (num_worlds * nq_per_world))
        sim.state_in.joint_qd.assign([0.0] * (num_worlds * nq_per_world))

        for _ in range(num_steps):
            sim.solver.step(
                state_in=sim.state_in,
                state_out=sim.state_out,
                control=sim.control,
                dt=dt,
                contacts=None,
            )
            sim.state_in, sim.state_out = sim.state_out, sim.state_in

        num_bodies = 4
        measured_body_poses = sim.state_in.body_q.numpy()
        measured_joint_q = sim.state_in.joint_q.numpy()
        for w in range(num_worlds):
            # body_b is body 2 and body_c is body 3 within each world
            world_body_indices = [w * num_bodies + 2, w * num_bodies + 3]
            residual = connect_residual(
                measured_body_poses,
                world_body_indices,
                wp.vec3(anchor_body_b),
                expected_anchor2,
            )
            self.assertAlmostEqual(residual, 0.0, places=4)
            for k in range(nq_per_world):
                self.assertAlmostEqual(float(measured_joint_q[w * nq_per_world + k]), 0.0, places=4)


class TestConnectAnchorRefPoseMuJoCoWarp(TestConnectAnchorRefPoseBase, unittest.TestCase):
    def _num_worlds(self):
        return 2

    def _use_mujoco_cpu(self):
        return False

    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_cpu=False,
            integrator="euler",
        )


class TestConnectAnchorRefPoseMuJoCoCPU(TestConnectAnchorRefPoseBase, unittest.TestCase):
    def _num_worlds(self):
        return 1

    def _use_mujoco_cpu(self):
        return True

    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            disable_contacts=True,
            use_mujoco_cpu=True,
            separate_worlds=True,
            integrator="euler",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
