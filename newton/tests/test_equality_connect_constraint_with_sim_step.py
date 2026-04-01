# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for joint equality constraints verified with simulation steps."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo, SolverNotifyFlags


class Sim:
    """Holds the simulation objects for a single test."""

    def __init__(self, model, solver, state_in, state_out, control):
        self.model = model
        self.solver = solver
        self.state_in = state_in
        self.state_out = state_out
        self.control = control


def connect_residual(body_poses, connect_body_indices, body2_anchor, body3_anchor):
    """Compute the world-space residual of a CONNECT constraint.

    Transforms anchor points on body2 and body3 (in their respective body
    frames) to world space using the body poses, then returns the distance
    between them.

    Args:
        body_poses: Array of body transforms (from ``state.body_q.numpy()``).
        connect_body_indices: ``[body2_index, body3_index]`` into ``body_poses``.
        body2_anchor: Anchor on body2 in body2's local frame (``wp.vec3``).
        body3_anchor: Anchor on body3 in body3's local frame (``wp.vec3``).

    Returns:
        Euclidean distance between the two world-space anchor points.
    """
    body2 = connect_body_indices[0]
    body3 = connect_body_indices[1]
    bq2 = body_poses[body2]
    bq3 = body_poses[body3]
    T2 = wp.transform(wp.vec3(bq2[0], bq2[1], bq2[2]), wp.quat(bq2[3], bq2[4], bq2[5], bq2[6]))
    T3 = wp.transform(wp.vec3(bq3[0], bq3[1], bq3[2]), wp.quat(bq3[3], bq3[4], bq3[5], bq3[6]))
    P2 = wp.transform_get_translation(T2) + wp.quat_rotate(wp.transform_get_rotation(T2), body2_anchor)
    P3 = wp.transform_get_translation(T3) + wp.quat_rotate(wp.transform_get_rotation(T3), body3_anchor)
    return float(wp.length(P2 - P3))


class TestEqualityConstraintWithSimStepBase:
    def _create_solver(self, model):
        raise NotImplementedError


class TestConnectConstraintWithSimStepBase(TestEqualityConstraintWithSimStepBase):
    """Test that a CONNECT equality constraint pins two bodies at a point."""

    class ConnectTestCase:
        """Parameters for a single CONNECT constraint test iteration."""

        def __init__(self, orig_a0, changed_a0, body2_index, initial_q, ref_q):
            self.orig_a0 = orig_a0
            self.changed_a0 = changed_a0
            self.body2_index = body2_index
            self.initial_q = initial_q
            self.ref_q = ref_q

    def _build_connect_model(
        self,
        connect_body_indices: list[int],
        connect_anchor_body2: list[float],
        joint_types: list[str],
        joint_axes: list[int],
        joint_dof_refs: list[float],
        num_worlds: int,
    ):
        """Build a 5-body articulation with a CONNECT constraint.

        Creates a fixed root (root_link), a ball-joint body (ball_link),
        an intermediate body (link0) connected by a high-armature joint,
        and two leaf bodies (body1, body2) connected to link0.
        A CONNECT constraint ties body1 and body2 at an anchor point.

        ``joint_types``, ``joint_axes``, and ``joint_dof_refs`` each have
        length 3: index 0 is for the joint from ball_link to link0,
        indices 1 and 2 are for the two leaf-body joints.
        The fixed root joint and ball joint are implicit.

        Args:
            connect_body_indices: Body indices ``[body2, body3]`` for the
                CONNECT constraint.
            connect_anchor_body2: Anchor point on body2 as ``[x, y, z]`` [m].
            joint_types: Joint type per non-root joint, length 3. Each is
                ``"revolute"`` or ``"prismatic"``.
            joint_axes: Motion axis per non-root joint, length 3 (0=X, 1=Y, 2=Z).
            joint_dof_refs: Reference joint position per non-root joint [rad or m],
                length 3.

        Returns:
            A :class:`Sim` containing the model, solver, states, and control.
        """
        self.assertEqual(len(joint_types), 3, "joint_types must have 3 elements")
        self.assertEqual(len(joint_axes), 3, "joint_axes must have 3 elements")
        self.assertEqual(len(joint_dof_refs), num_worlds, "joint_dof_refs must have num_worlds rows")
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

        all_worlds_builder = newton.ModelBuilder(gravity=0.0, up_axis=1)

        for w in range(num_worlds):
            builder = newton.ModelBuilder(gravity=0.0, up_axis=1)
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

            # body1 (body index 3 in Newton's list of bodies), joint1,
            # body2 (body index 4 in Newton's list of bodies), joint2
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
                elif connect_joint_types[i] == "ball":
                    joint_fn = builder.add_joint_ball
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

            builder.add_equality_constraint_connect(
                body1=connect_body_indices[0],
                body2=connect_body_indices[1],
                anchor=connect_anchor_body2[w],
            )

            all_worlds_builder.add_world(builder)

        model = all_worlds_builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        solver = self._create_solver(model)

        return Sim(model, solver, state_in, state_out, control)

    def computeJ(self, joint_axis: int, joint_pos: float, joint_type: str) -> wp.transform:
        J = wp.transform_identity()
        if joint_type == "prismatic":
            pos = [0.0, 0.0, 0.0]
            pos[joint_axis] = joint_pos
            J = wp.transform(pos, wp.quat_identity())
        elif joint_type == "revolute":
            axes = [wp.vec3(1, 0, 0), wp.vec3(0, 1, 0), wp.vec3(0, 0, 1)]
            J = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(axes[joint_axis], joint_pos))
        return J

    def compute_expected_body3_anchor(self, joint_axes, joint_dof_refs, joint_types, connect_anchor_body2):
        """Compute the expected anchor on body3 for a CONNECT constraint.

        Performs FK using the joint ref positions to get world poses of body2 and
        body3, then computes the body3-local anchor that coincides with
        ``connect_anchor_body2`` on body2 in the reference configuration.

        The model topology is: root_link (fixed) -> ball_link (ball) -> link0 (joint0) -> body1 (joint1)
                                                                                      -> body2 (joint2)

        Args:
            joint_axes: Motion axis per non-root joint, length 3 (0=X, 1=Y, 2=Z).
            joint_dof_refs: Reference joint positions, length 3 [rad or m].
            joint_types: Joint type per non-root joint, length 3.
            connect_anchor_body2: Anchor on body2 as ``[x, y, z]``.

        Returns:
            Expected anchor on body3 as ``wp.vec3``.
        """
        J0 = self.computeJ(joint_axes[0], joint_dof_refs[0], joint_types[0])
        J1 = self.computeJ(joint_axes[1], joint_dof_refs[1], joint_types[1])
        J2 = self.computeJ(joint_axes[2], joint_dof_refs[2], joint_types[2])
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
        return wp.quat_rotate(q, wp.vec3(connect_anchor_body2)) + t

    def _test_connect_constraint(self):
        """Verify that the CONNECT constraint brings two separated bodies to the same point.

        Tests multiple anchor positions to exercise the constraint at different
        offsets from the body origin.

        Args:
            joint_type: One of ``"revolute"`` or ``"prismatic"``.
            motion_axis: Joint motion axis (0=X, 1=Y, 2=Z).
        """

        dt = 0.01
        num_steps = 50
        num_worlds = 2

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
        connect_anchor_body2 = [[1.0, 2.0, 3.0], [1.3, 2.4, 2.6]]
        changed_connect_anchor_body2 = [[-1.5, -2.5, -3.5], [-1.8, -2.2, -3.1]]

        # Ball joint identity quaternion coords (x, y, z, w)
        ball_q_identity = [0.0, 0.0, 0.0, 1.0]
        ball_qd_zero = [0.0, 0.0, 0.0]

        flat_joint_dof_refs = []
        flat_initial_q = []
        flat_initial_qd = []
        flat_changed_connect_anchor_body2 = []
        num_bodies = 5
        # Ball joint adds 4 coords (quaternion) before the 3 joint coords
        ball_q_offset = 4
        for w in range(num_worlds):
            # Ball joint coords (identity quaternion for ref, identity for initial)
            for v in ball_q_identity:
                flat_joint_dof_refs.append(v)
                flat_initial_q.append(v)
            # Ball joint DOFs (zero velocity)
            for v in ball_qd_zero:
                flat_initial_qd.append(v)
            for k in range(3):
                flat_joint_dof_refs.append(joint_dof_refs[w][k])
                flat_initial_q.append(initial_q[w][k])
                flat_initial_qd.append(initial_qd[w][k])
            for k in range(3):
                flat_changed_connect_anchor_body2.append(changed_connect_anchor_body2[w][k])

        for i in range(0, num_joint_0_joint_types):
            for j in range(0, num_connect_joint_types_and_axes):
                joint_types = [
                    joint_0_joint_types[i],
                    connect_joint_types_and_axes[j][0],
                    connect_joint_types_and_axes[j][1],
                ]
                joint_axes = [joint_0_axis, connect_joint_types_and_axes[j][2], connect_joint_types_and_axes[j][3]]

                sim = self._build_connect_model(
                    connect_body_indices=connect_body_indices,
                    connect_anchor_body2=connect_anchor_body2,
                    joint_types=joint_types,
                    joint_axes=joint_axes,
                    joint_dof_refs=joint_dof_refs,
                    num_worlds=num_worlds,
                )

                for w in range(num_worlds):
                    # Compute the expected anchors.
                    # body2's anchor is the input connect_anchor_body2.
                    # body3's anchor is derived from FK at the reference joint positions.
                    expected_a2 = connect_anchor_body2[w]
                    expected_a3 = self.compute_expected_body3_anchor(
                        joint_axes, joint_dof_refs[w], joint_types, connect_anchor_body2[w]
                    )
                    # Check that the expected anchors match the measured anchors.
                    measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                    # eq_data shape is [nworld, neq, 11]; world w, constraint 0
                    measured_a2 = wp.vec3(
                        measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                    )
                    measured_a3 = wp.vec3(
                        measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                    )
                    for k in range(3):
                        self.assertAlmostEqual(float(expected_a2[k]), float(measured_a2[k]), places=4)
                        self.assertAlmostEqual(float(expected_a3[k]), float(measured_a3[k]), places=4)
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
                # Set the start state to the reference joint positions
                # to ensure that the start state satisfies the connect
                #  constraint. Nothing should move.
                ##############

                sim.state_in.joint_q.assign(flat_joint_dof_refs)
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

                # After N steps, tTestConnectConstraintMuJoCo.test_connect_constraint_prismatic_xe residual should be close to 0
                # and the joint positions should be unchanged from the
                # start state because the start state was deliberately
                # chosen to satisfy the connect constraint.
                for w in range(num_worlds):
                    measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                    measured_a2 = wp.vec3(
                        measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                    )
                    measured_a3 = wp.vec3(
                        measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                    )
                    measured_body_poses = sim.state_in.body_q.numpy()
                    world_body_indices = [
                        w * num_bodies + connect_body_indices[0],
                        w * num_bodies + connect_body_indices[1],
                    ]
                    residual = connect_residual(measured_body_poses, world_body_indices, measured_a2, measured_a3)
                    self.assertAlmostEqual(residual, 0.0, places=4)
                    measured_joint_q = sim.state_in.joint_q.numpy()
                    nq_per_world = ball_q_offset + 3
                    self.assertAlmostEqual(
                        measured_joint_q[w * nq_per_world + ball_q_offset + 1],
                        flat_joint_dof_refs[w * nq_per_world + ball_q_offset + 1],
                        places=4,
                    )
                    self.assertAlmostEqual(
                        measured_joint_q[w * nq_per_world + ball_q_offset + 2],
                        flat_joint_dof_refs[w * nq_per_world + ball_q_offset + 2],
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
                # The anchors have not changed so it is correct to continue using measured_a2, measured_a3
                # as the anchords.
                for w in range(num_worlds):
                    measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                    measured_a2 = wp.vec3(
                        measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                    )
                    measured_a3 = wp.vec3(
                        measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                    )
                    measured_body_poses = sim.state_in.body_q.numpy()
                    world_body_indices = [
                        w * num_bodies + connect_body_indices[0],
                        w * num_bodies + connect_body_indices[1],
                    ]
                    residual = connect_residual(measured_body_poses, world_body_indices, measured_a2, measured_a3)
                    self.assertAlmostEqual(residual, 0.0, places=3)

                ##############
                # TEST 3
                # Change the anchor at runtime and verify the constraint responds
                # to the new anchor.
                ##############

                sim.model.equality_constraint_anchor.assign(
                    np.array(flat_changed_connect_anchor_body2, dtype=np.float32)
                )
                sim.solver.notify_model_changed(SolverNotifyFlags.CONSTRAINT_PROPERTIES)

                # Verify that mjw_model.eq_data was updated with the new anchor.
                for w in range(num_worlds):
                    changed_expected_a3 = self.compute_expected_body3_anchor(
                        joint_axes, joint_dof_refs[w], joint_types, changed_connect_anchor_body2[w]
                    )
                    measured_eq_data = sim.solver.mjw_model.eq_data.numpy()
                    changed_measured_a2 = wp.vec3(
                        measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                    )
                    changed_measured_a3 = wp.vec3(
                        measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                    )
                    for k in range(3):
                        self.assertAlmostEqual(
                            float(changed_connect_anchor_body2[w][k]), float(changed_measured_a2[k]), places=4
                        )
                        self.assertAlmostEqual(float(changed_expected_a3[k]), float(changed_measured_a3[k]), places=4)

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
                    changed_measured_a2 = wp.vec3(
                        measured_eq_data[w][0][0], measured_eq_data[w][0][1], measured_eq_data[w][0][2]
                    )
                    changed_measured_a3 = wp.vec3(
                        measured_eq_data[w][0][3], measured_eq_data[w][0][4], measured_eq_data[w][0][5]
                    )
                    measured_body_poses = sim.state_in.body_q.numpy()
                    world_body_indices = [
                        w * num_bodies + connect_body_indices[0],
                        w * num_bodies + connect_body_indices[1],
                    ]
                    residual = connect_residual(
                        measured_body_poses, world_body_indices, changed_measured_a2, changed_measured_a3
                    )
                    self.assertAlmostEqual(residual, 0.0, places=3)

    def test_connect_constraint(self):
        self._test_connect_constraint()


class TestConnectConstraintJointMuJoCo(TestConnectConstraintWithSimStepBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            iterations=1,
            ls_iterations=1,
            disable_contacts=True,
            use_mujoco_cpu=False,
            integrator="euler",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
