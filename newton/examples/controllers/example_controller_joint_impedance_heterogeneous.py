# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Heterogeneous Joint Impedance
#
# Demonstrates ControllerJointImpedance on a mixed fleet:
#   Robot A — 3-DOF revolute chain (triple pendulum), left side
#   Robot B — 1-DOF revolute pendulum, right side
#
# Robot A tracks a sinusoidal target with staggered joint phases.
# Robot B holds upright at q=0.
# Both robots use model-based impedance (gravity compensation + inertia
# decoupling computed internally from their respective Newton models).
#
# The two robots have different DOF counts, exercising the heterogeneous
# gather/scatter and per-robot kernel guard paths.
#
# Command: python -m newton.examples controller_joint_impedance_heterogeneous
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
from newton import JointTargetMode
from newton.controllers import ControllerJointImpedance

# ---------------------------------------------------------------------------
# Robot geometry
# ---------------------------------------------------------------------------

LINK_LEN_A = 0.25  # length of each link in robot A [m]
LINK_LEN_B = 0.45  # length of robot B's single link [m]
DOFS_A = 3
DOFS_B = 1
MAX_DOFS = max(DOFS_A, DOFS_B)  # = 3
TOTAL_DOFS = DOFS_A + DOFS_B  # = 4

# Gains — shape (2, MAX_DOFS); robot B's columns 1 and 2 are padding (unused)
KP = np.array([[200.0, 200.0, 200.0], [200.0, 0.0, 0.0]], dtype=np.float32)
KD = np.array([[20.0, 20.0, 20.0], [20.0, 0.0, 0.0]], dtype=np.float32)

# Sinusoidal target for robot A: each joint offset by π/3
TARGET_AMP = 0.4  # [rad]
TARGET_FREQ = 0.4  # [Hz]
PHASE_A = [0.0, math.pi / 3, 2 * math.pi / 3]

# Continuous rotation for robot B
ROT_SPEED_B = 1.5  # [rad/s]


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _add_revolute_chain(builder, n_links, link_len, x_offset, label):
    """Add an n-link revolute pendulum chain to builder.

    Joints rotate about the Z axis. The first joint is anchored at
    (x_offset, 0, 0). Each link hangs downward along -Y.
    """
    joints = []
    prev_body = -1
    for i in range(n_links):
        body = builder.add_link()

        # Pivot location: world origin for first link, bottom of previous link otherwise.
        if i == 0:
            parent_xform = wp.transform(wp.vec3(x_offset, 0.0, 0.0), wp.quat_identity())
        else:
            parent_xform = wp.transform(wp.vec3(0.0, -link_len, 0.0), wp.quat_identity())

        j = builder.add_joint_revolute(
            parent=prev_body,
            child=body,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=parent_xform,
            child_xform=wp.transform_identity(),
        )
        joints.append(j)

        # Thin box as the link rod, centred at half the link length below the pivot
        builder.add_shape_box(
            body=body,
            xform=wp.transform(wp.vec3(0.0, -link_len * 0.5, 0.0), wp.quat_identity()),
            hx=0.02,
            hy=link_len * 0.5,
            hz=0.02,
        )

        prev_body = body

    builder.add_articulation(joints, label=label)
    return joints


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    @staticmethod
    def create_parser():
        return newton.examples.create_parser()

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ---- Controller model ------------------------------------------------
        # Both articulations in one builder so ControllerJointImpedance can
        # derive per-robot DOF counts and run FK/dynamics for both.
        ctrl_builder = newton.ModelBuilder()
        _add_revolute_chain(ctrl_builder, DOFS_A, LINK_LEN_A, x_offset=-0.5, label="robot_a")
        _add_revolute_chain(ctrl_builder, DOFS_B, LINK_LEN_B, x_offset=+0.5, label="robot_b")
        # ctrl_builder is passed to ControllerJointImpedance; it finalizes it internally.

        # ---- Physics scene ---------------------------------------------------
        # Identical topology to ctrl_builder, with effort-control mode.
        scene = newton.ModelBuilder()
        _add_revolute_chain(scene, DOFS_A, LINK_LEN_A, x_offset=-0.5, label="robot_a")
        _add_revolute_chain(scene, DOFS_B, LINK_LEN_B, x_offset=+0.5, label="robot_b")
        scene.add_ground_plane()

        for i in range(TOTAL_DOFS):
            scene.joint_target_ke[i] = 0.0
            scene.joint_target_kd[i] = 0.0
            scene.joint_target_mode[i] = int(JointTargetMode.EFFORT)

        self.model = scene.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        # ---- Impedance controller --------------------------------------------
        # default_dof_indices: identity — robot A occupies DOFs 0..2, robot B DOF 3.
        default_idx = wp.array(np.arange(TOTAL_DOFS, dtype=np.uint32), device=self.device)

        self.controller = ControllerJointImpedance(
            model_builder=ctrl_builder,
            default_dof_indices=default_idx,
            stiffness=wp.array(KP, dtype=wp.float32, device=self.device),
            damping=wp.array(KD, dtype=wp.float32, device=self.device),
            use_gravity_compensation=True,
            use_coriolis_compensation=False,
            use_inertia_decoupling=True,
            device=self.device,
        )

        self._input = self.controller.input()
        self._output = self.controller.output()
        # Wire torque output directly into the sim control buffer.
        self._output.joint_f = self.control.joint_f

        # Bind live sim arrays before capture so the graph records the correct
        # buffer addresses. state_0 holds the current frame result after
        # sim_substeps (even number), so these pointers remain valid each replay.
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd

        self._graph = None
        if self.controller.is_graphable() and self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._gpu_step()
            self._graph = capture.graph

        self.viewer.set_model(self.model)

    def _gpu_step(self):
        """Pure GPU work: controller compute + physics substeps. Safe to graph-capture."""
        self.controller.compute(self._input, self._output, self.sim_dt)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # Update targets on the CPU — cannot be graph-captured.
        q_des = np.zeros(TOTAL_DOFS, dtype=np.float32)
        for k, phase in enumerate(PHASE_A):
            q_des[k] = TARGET_AMP * math.sin(2 * math.pi * TARGET_FREQ * self.sim_time + phase)
        q_des[3] = ROT_SPEED_B * self.sim_time  # robot B rotates continuously
        self._input.joint_q_des.assign(q_des)

        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._gpu_step()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify that both robots are tracking their targets with finite joint positions."""
        joint_q = self.state_0.joint_q.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"

        # Robot B (DOF index 3) should be tracking its rotating target.
        expected_b = ROT_SPEED_B * self.sim_time
        np.testing.assert_allclose(
            joint_q[3],
            expected_b,
            atol=0.3,
            err_msg=f"Robot B not tracking target: q={joint_q[3]:.3f}, expected={expected_b:.3f}",
        )

        # Robot A (DOF indices 0..2) should be within the sinusoidal amplitude range.
        assert np.all(np.abs(joint_q[:3]) <= TARGET_AMP + 0.3), f"Robot A joints out of expected range: {joint_q[:3]}"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
