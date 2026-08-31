# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Operational Space Hybrid Force/Motion
#
# Demonstrates ControllerOperationalSpace on one real 7-DOF Franka Panda
# arm (redundant against the 6D task), pressing its gripper down into a
# collision plane (a tabletop) while its horizontal position is
# interactively steered. A static, uncontrolled box sits on the table too,
# to show the model can hold bodies the controller never touches.
#
# Whichever tool-local axis is closest to vertical at the home ("ready")
# pose -- the gripper's own Z axis, which points straight down there by
# construction -- is wrench-controlled with a feedforward press force; the
# other five task axes (the two in-plane linear directions and full
# orientation) are motion-controlled, tracking a desired (x, y) on the
# table while holding the tool's orientation fixed. This is a true
# zero-stiffness hybrid split: the press axis carries no position spring
# at all, only the feedforward force -- the same design Isaac Lab's own
# test_franka_hybrid_decoupled_motion (isaaclab/test/controllers/
# test_operational_space.py) uses on this exact robot. A secondary
# null-space posture task pulls the redundant DOF back toward the ready
# pose without disturbing either the force or the motion task, since the
# null-space projector guarantees zero task-space disturbance across all 6
# task dimensions, regardless of which of them are currently wrench- vs
# motion-controlled.
#
# Three sliders (x, y, press force) let you steer the commanded task
# directly; a SensorContact on the two gripper fingers reads back the
# actual contact force the table exerts on the tool, and the GUI panel
# prints commanded vs. measured/actual for all three so tracking can be
# confirmed directly, not just assumed from the control law.
#
# An earlier version of this example built its own lightweight, procedural
# 7/8-DOF arm instead of loading a real robot asset, and the same
# zero-stiffness split diverged on it: with inertia decoupling, the
# task-space effective mass (Lambda) on a redundant, very lightly built arm
# varies enormously with configuration, and a direction with no spring
# holding it near a well-conditioned pose has nothing to keep it there. A
# real Franka's masses/inertias and its well-conditioned "ready" pose don't
# have that problem.
#
# Command: python -m newton.examples controller_operational_space_hybrid_force_motion
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.solvers
import newton.utils
from newton import Contacts, JointTargetMode
from newton.controllers import ControllerOperationalSpace
from newton.sensors import SensorContact

# ---------------------------------------------------------------------------
# Robot configuration
# ---------------------------------------------------------------------------

# Franka's standard "ready" pose: well clear of joint limits and singularities,
# with the gripper pointing straight down (world -Z) -- also the null-space
# posture target throughout the run.
READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
ARM_DOFS = len(READY_POSE)  # 7; the two finger joints are left uncontrolled

# Slider ranges, centered on the ready pose's tool position.
XY_SLIDER_RANGE = 0.15  # [m], +/- around the ready-pose x/y
FORCE_SLIDER_MAX = 80.0  # [N]

# Height of the table's top surface above the ground plane [m]. Chosen so
# the full +/-XY_SLIDER_RANGE stays within the table's footprint (0.5 +/-
# 0.35 in x, 0 +/- 0.35 in y) around the ready-pose tool position.
TABLE_HEIGHT = 0.15

# Gains -- use_inertia_decoupling=True, so these are in the mass-normalized
# (acceleration) domain: [1/s^2] for stiffness, [1/s] for damping. Same
# order of magnitude as Isaac Lab's own Franka OSC tests (200-1000).
MOTION_KP = 300.0
MOTION_KD = 2.0 * MOTION_KP**0.5  # critically damped
NULL_KP = 50.0
NULL_KD = 2.0 * NULL_KP**0.5


def _closest_to_vertical_local_axis(tool_pose_world) -> int:
    """Return which tool-local linear axis (0=X, 1=Y, 2=Z) is closest to world Z.

    ``wrench_selection_axes_tool`` is tool-local, so pressing "down" means
    selecting whichever local axis the home orientation happens to align
    with world Z. For Franka's gripper at the ready pose this is local Z
    (by URDF convention the gripper points straight down there), but this
    measures it directly from the FK'd home quaternion rather than assuming
    it, so the same logic works unchanged if READY_POSE is ever changed.
    Reads the world-Z row of the rotation matrix built from the quaternion
    (the standard quaternion-to-matrix formula) and picks its
    largest-magnitude column.
    """
    qx, qy, qz, qw = tool_pose_world[3:]
    world_z_row = np.array([2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)])
    return int(np.argmax(np.abs(world_z_row)))


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

        # ---- Physics scene ---------------------------------------------------
        urdf_path = str(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf")
        builder = newton.ModelBuilder()

        arm_joints, tool_body, finger_bodies = self._add_franka(builder, urdf_path)

        builder.add_ground_plane()

        # A physical table in front of the robot -- this, not the ground
        # plane, is what the gripper presses into. Its top sits at
        # TABLE_HEIGHT; the slider range in __init__ keeps the commanded
        # (x, y) within its footprint.
        table_body = builder.add_link()
        builder.add_shape_box(table_body, hx=0.35, hy=0.35, hz=TABLE_HEIGHT / 2.0)
        table_joint = builder.add_joint_fixed(
            parent=-1,
            child=table_body,
            parent_xform=wp.transform(wp.vec3(0.5, 0.0, TABLE_HEIGHT / 2.0), wp.quat_identity()),
        )
        builder.add_articulation([table_joint], label="table")

        # A static box on the table that is part of the model but never
        # selected by the controller (no joints of its own to control) --
        # just to show the model can hold uncontrolled content alongside a
        # controlled robot. Placed off to the side, clear of the slider
        # range, so it never sits under the gripper.
        obstacle_body = builder.add_link()
        builder.add_shape_box(obstacle_body, hx=0.05, hy=0.05, hz=0.05)
        obstacle_joint = builder.add_joint_fixed(
            parent=-1,
            child=obstacle_body,
            parent_xform=wp.transform(wp.vec3(0.65, 0.28, TABLE_HEIGHT + 0.05), wp.quat_identity()),
        )
        builder.add_articulation([obstacle_joint], label="obstacle")

        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 0.0
            builder.joint_target_kd[i] = 0.0
            builder.joint_target_mode[i] = int(JointTargetMode.EFFORT)

        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Contacts stay enabled (the default) so the gripper's own collision
        # geometry actually presses against, and is resisted by, the table.
        # nconmax raised above its default: the two fingers' meshes generate
        # more simultaneous contact points against a flat plane than the
        # default budgets for.
        self.solver = newton.solvers.SolverMuJoCo(self.model, nconmax=200)

        # Force sensor: reads back the actual contact force the table
        # exerts on the gripper fingers, so the commanded press force can be
        # checked against what's really happening, not just assumed.
        self.force_sensor = SensorContact(self.model, sensing_bodies=finger_bodies)
        self.contacts = Contacts(
            self.solver.get_max_contact_count(),
            0,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        # Home tool pose (position + orientation), read off directly from FK
        # at the ready configuration -- the tool site's transform is
        # identity, so it equals fr3_hand_tcp's own world pose. The x/y
        # sliders offset from this position; the orientation and (masked)
        # z target hold it fixed throughout, and the wrench selection axis
        # below is measured from it.
        home_pose = self.state_0.body_q.numpy()[tool_body].astype(np.float32)
        self._home_pose = home_pose
        self.desired_x = float(home_pose[0])
        self.desired_y = float(home_pose[1])
        self.desired_force = 0.0

        press_axis = _closest_to_vertical_local_axis(home_pose)
        motion_selection = wp.spatial_vector(*(0.0 if axis == press_axis else 1.0 for axis in range(6)))
        wrench_selection = wp.spatial_vector(*(1.0 if axis == press_axis else 0.0 for axis in range(6)))

        # motion_stiffness/motion_damping are tool-local, like
        # motion_selection_axes_tool above (both rotated into world every
        # step from the tool's current orientation), so press_axis is the
        # right index to zero for both.
        #
        # Zeroing it is required, not just a tidy-up: with a uniform nonzero
        # gain on every axis, the dense position/velocity error term still
        # computes a large "virtual acceleration" along the press axis even
        # though it's excluded from motion_selection_axes_tool -- masking
        # only happens to the *output* of Lambda @ acceleration, and
        # Lambda's off-diagonal coupling terms leak that spurious
        # acceleration into the other axes' *inputs* before the mask is
        # ever applied. Zeroing the gain removes the spurious acceleration
        # at the source, instead of trying to mask its downstream effects.
        # Confirmed by direct comparison: with a uniform scalar gain, x
        # drifted to a wrong steady state under the press force despite x
        # being fully motion-selected; with the press axis's gain zeroed,
        # x converges correctly.
        motion_stiffness = wp.spatial_vector(*(0.0 if axis == press_axis else MOTION_KP for axis in range(6)))
        motion_damping = wp.spatial_vector(*(0.0 if axis == press_axis else MOTION_KD for axis in range(6)))

        # ---- Operational-space controller -------------------------------------
        # "tool_site" matches the one site on this robot. joints restricts
        # control to the 7 arm joints, leaving the 2 gripper finger joints
        # uncontrolled. The controller reads its FK and dynamics terms from
        # the same model the solver simulates.
        self.controller = ControllerOperationalSpace(
            self.model,
            joints=arm_joints,
            tool="tool_site",
            motion_stiffness=motion_stiffness,
            motion_damping=motion_damping,
            # Commands/gains are interpreted directly in world coordinates.
            operational_frame_pose_world=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            use_inertia_decoupling=True,
            use_gravity_compensation=True,
            use_wrench_feedforward=True,
            motion_selection_axes_tool=motion_selection,
            wrench_selection_axes_tool=wrench_selection,
            use_null_space_control=True,
            null_space_stiffness=NULL_KP,
            null_space_damping=NULL_KD,
        )

        self._input = self.controller.input()
        self._output = self.controller.output()
        # The controller's torque output is compact (one entry per controlled
        # DOF); an indexed view scatters it straight into the sim control buffer.
        self._output.joint_f = self.control.joint_f[self.controller.qd_start]

        # Bind live sim arrays before capture so the graph records the correct
        # buffer addresses. state_0 holds the current frame result after
        # sim_substeps (even number), so these pointers remain valid each replay.
        self._input.joint_q = self.state_0.joint_q
        self._input.joint_qd = self.state_0.joint_qd

        # Constant across every step: bind once, before capture. desired_twist
        # is always zero -- sliders move quasi-statically, so no feedforward
        # velocity is needed.
        self._input.desired_twist_operational.assign(np.zeros((1, 6), dtype=np.float32))
        self._input.joint_q_des_null.assign(np.array(READY_POSE, dtype=np.float32))
        self._input.joint_qd_des_null.assign(np.zeros(ARM_DOFS, dtype=np.float32))

        self._graph = None
        if self.controller.is_graphable() and self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._gpu_step()
            self._graph = capture.graph

        self.viewer.set_model(self.model)

    @staticmethod
    def _add_franka(builder, urdf_path):
        """Load one Franka at the origin, set its ready pose, and add its tool site.

        Returns:
            Tuple of (arm joint indices, fr3_hand_tcp body index, [leftfinger, rightfinger] body indices).
        """
        joint_count_before = builder.joint_count
        coord_count_before = builder.joint_coord_count
        body_count_before = builder.body_count
        builder.add_urdf(urdf_path, floating=False)

        # fr3_joint1..7 are the first 7 non-fixed joints after the (fixed,
        # 0-coordinate) base/mount joints this URDF starts with; the finger
        # joints follow. Joint indices are offset by the 2 fixed joints, but
        # coordinate indices are not, since a fixed joint contributes no
        # coordinates. Indices are relative to this call since add_urdf
        # appends them.
        arm_joints = [joint_count_before + 2 + i for i in range(ARM_DOFS)]
        arm_coords = range(coord_count_before, coord_count_before + ARM_DOFS)
        for coord, angle in zip(arm_coords, READY_POSE, strict=True):
            builder.joint_q[coord] = angle

        # Bodies 11, 12, 13 (0-based) this URDF adds are fr3_hand_tcp (the
        # fixed frame between the fingers), fr3_leftfinger, fr3_rightfinger.
        tool_body = body_count_before + 11
        finger_bodies = [body_count_before + 12, body_count_before + 13]
        builder.add_site(tool_body, label="tool_site")

        return arm_joints, tool_body, finger_bodies

    def _gpu_step(self):
        """Pure GPU work: controller step + physics substeps. Safe to graph-capture."""
        self.controller.step(inputs=self._input, outputs=self._output, dt=self.sim_dt)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        # Sliders drive the target directly -- read in gui(), applied here.
        # Cannot be graph-captured (assign() is, but the desired values
        # themselves come from Python-side UI state read after capture).
        desired_pose = self._home_pose.copy()[None, :]
        desired_pose[0, 0] = self.desired_x
        desired_pose[0, 1] = self.desired_y
        self._input.desired_tool_pose_operational.assign(desired_pose)
        self._input.desired_wrench_world.assign(
            np.array([[0.0, 0.0, -self.desired_force, 0.0, 0.0, 0.0]], dtype=np.float32)
        )

        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._gpu_step()

        # Not graph-capturable: reads self.contacts back to Python (.numpy()
        # below) to display in the GUI.
        self.force_sensor.update(self.state_0, self.contacts)

        self.sim_time += self.frame_dt

    def gui(self, ui):
        _, self.desired_x = ui.slider_float(
            "Desired x [m]", self.desired_x, self._home_pose[0] - XY_SLIDER_RANGE, self._home_pose[0] + XY_SLIDER_RANGE
        )
        _, self.desired_y = ui.slider_float(
            "Desired y [m]", self.desired_y, self._home_pose[1] - XY_SLIDER_RANGE, self._home_pose[1] + XY_SLIDER_RANGE
        )
        _, self.desired_force = ui.slider_float("Desired press force [N]", self.desired_force, 0.0, FORCE_SLIDER_MAX)

        actual_pose = self.controller._tool_pose_world.numpy()[0]
        # Force the table exerts on the fingers, summed; positive z means
        # the table is pushing back up against the commanded downward press.
        measured_force_z = float(self.force_sensor.total_force.numpy()[:, 2].sum())

        ui.text(f"actual x:   {actual_pose[0]:.3f}   (desired {self.desired_x:.3f})")
        ui.text(f"actual y:   {actual_pose[1]:.3f}   (desired {self.desired_y:.3f})")
        ui.text(f"measured press force: {measured_force_z:.1f} N   (desired {self.desired_force:.1f} N)")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the robot settled into a stable, finite configuration."""
        joint_q = self.state_0.joint_q.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"

        ready_q = np.array(READY_POSE, dtype=np.float32)
        assert np.all(np.abs(joint_q[:ARM_DOFS] - ready_q) < 1.5), (
            f"arm joints drifted far from the null-space posture target: {joint_q[:ARM_DOFS]}"
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
