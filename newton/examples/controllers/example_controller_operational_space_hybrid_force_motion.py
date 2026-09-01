# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Operational Space Hybrid Force/Motion
#
# Demonstrates ControllerOperationalSpace on one real 7-DOF Franka Panda arm
# (redundant against the 6D task), pressing its gripper into a table
# tilted 45 degrees toward it, with its position along the table steered
# interactively. The operational frame is placed on the table's top
# surface, oriented so its Z axis is normal to the (tilted) table -- so the
# same operational-frame-relative command works regardless of the tilt, and
# the axis triad drawn each frame at that frame lets you see exactly where
# the controller thinks "the table" and "into the table" are.
#
# The operational frame's local Z (into the table) is wrench-controlled
# with a feedforward press force; the other five task axes (the two
# in-plane directions along the table, and full orientation) are
# motion-controlled, tracking a desired (x, y) on the table's surface.
# A secondary null-space posture task pulls the redundant DOF back toward the ready
# pose without disturbing either the force or the motion task
#
# Three sliders (x, y, press force) let you steer the commanded task
# directly; a SensorContact on the two gripper fingers reads back the
# actual contact force the table exerts on the tool, and the GUI panel
# prints commanded vs. measured/actual for all three so tracking can be
# confirmed directly, not just assumed from the control law.
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

# Franka's standard "ready" pose
READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
ARM_DOFS = len(READY_POSE)  # 7; the two finger joints are left uncontrolled

# Slider ranges, centered on the gripper's actual starting (x, y) in the
# operational frame -- so the initial commanded position matches where the
# gripper already is, along the table's tilted surface.
XY_SLIDER_RANGE = 0.15  # [m]
FORCE_SLIDER_MAX = 80.0  # [N]

# The table: a box of half-height TABLE_HEIGHT/2 and half-footprint
# TABLE_HALF_EXTENT, centered 0.5m in front of the robot, tilted
# TABLE_TILT_ANGLE about world Y so its top surface faces up and toward the
# robot. TABLE_POSITION/TABLE_ROTATION are the box's own (center) pose; the
# operational frame is built from these but offset onto the top surface.
TABLE_HALF_EXTENT = 0.35
TABLE_HEIGHT = 0.15
TABLE_TILT_ANGLE = np.pi / 4.0
TABLE_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -TABLE_TILT_ANGLE)
TABLE_POSITION = wp.vec3(0.5, 0.0, np.sqrt(TABLE_HALF_EXTENT**2 + (TABLE_HEIGHT / 2.0) ** 2) + 0.05)

# Gains -- use_inertia_decoupling=True, so these are in the mass-normalized
# (acceleration) domain: [1/s^2] for stiffness, [1/s] for damping.
MOTION_KP = 300.0
MOTION_KD = 2.0 * MOTION_KP**0.5  # critically damped
NULL_KP = 50.0
NULL_KD = 2.0 * NULL_KP**0.5


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

        table_transform = wp.transform(TABLE_POSITION, TABLE_ROTATION)
        table_body = builder.add_link()
        builder.add_shape_box(table_body, hx=TABLE_HALF_EXTENT, hy=TABLE_HALF_EXTENT, hz=TABLE_HEIGHT / 2.0)
        table_joint = builder.add_joint_fixed(parent=-1, child=table_body, parent_xform=table_transform)
        builder.add_articulation([table_joint], label="table")

        # Cached for step()/gui()/render(): the operational frame's
        # numpy-side rotation/position/normal, so the world-frame
        # wrench/position math below doesn't need to redo transform algebra
        # in Warp every frame.
        table_rotation_np = np.array(wp.quat_to_matrix(TABLE_ROTATION), dtype=np.float32).reshape(3, 3)
        table_normal_world = table_rotation_np @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._table_rotation_np = table_rotation_np
        self._table_normal_world = table_normal_world
        self._table_position_np = np.array(TABLE_POSITION, dtype=np.float32) + table_normal_world * (TABLE_HEIGHT / 2.0)
        operational_frame_transform = wp.transform(wp.vec3(*self._table_position_np.tolist()), TABLE_ROTATION)

        # A static axis triad at the operational frame, drawn every frame in render()
        axis_length = TABLE_HALF_EXTENT + 0.1
        origin = self._table_position_np
        axis_tips = origin + axis_length * self._table_rotation_np.T
        self._operational_frame_gizmo_starts = wp.array([origin] * 3, dtype=wp.vec3, device=self.device)
        self._operational_frame_gizmo_ends = wp.array(axis_tips, dtype=wp.vec3, device=self.device)
        self._operational_frame_gizmo_colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)],
            dtype=wp.vec3,
            device=self.device,
        )

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

        # SensorContact + Contacts is Newton's contact-force readback API (see
        # example_sensor_contact.py) -- reads back the actual contact force the
        # table exerts on the gripper fingers, so the commanded press force can
        # be checked against what's really happening, not just assumed.
        self.force_sensor = SensorContact(self.model, sensing_bodies=finger_bodies)
        self.contacts = Contacts(
            self.solver.get_max_contact_count(),
            0,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        # Home tool pose, read off directly from FK at the ready
        # configuration -- the tool site's transform is identity, so it
        # equals fr3_hand_tcp's own world pose there. The desired
        # orientation, relative to the (tilted) operational frame, is
        # computed so it composes back to exactly this same world
        # orientation -- zero initial orientation error, matching the zero
        # initial position error below, rather than commanding a sudden
        # 45-degree reorientation snap at startup.
        home_pose = self.state_0.body_q.numpy()[tool_body].astype(np.float32)
        self._home_pose = home_pose
        home_orientation_world = wp.quat(*home_pose[3:7].tolist())
        desired_orientation_operational = wp.quat_inverse(TABLE_ROTATION) * home_orientation_world
        self._home_pose[3:7] = np.array(desired_orientation_operational, dtype=np.float32)
        # x/y sliders offset the target along the table's tangent plane,
        # relative to the operational frame's own origin (the table's top
        # surface). Initialized to the gripper's actual starting (x, y) in
        # that same frame -- zero initial error, like the flat-table
        # version -- rather than to the origin, which would otherwise be a
        # sudden, large initial position command. z is left at 0 since that
        # axis is wrench-, not motion-, controlled.
        home_pos_operational = self._table_rotation_np.T @ (home_pose[:3] - self._table_position_np)
        self._home_pos_operational = home_pos_operational
        self.desired_x = float(home_pos_operational[0])
        self.desired_y = float(home_pos_operational[1])
        self.desired_force = 0.0

        # The operational frame's local Z (below) is the press axis (index
        # 2); the other five task axes are motion-controlled. The linear and
        # angular selection frames (below) are both left at identity
        # relative to the operational frame, so "axis 2" here is literally
        # the table's normal -- independent of the tool's own orientation.
        motion_selection = wp.spatial_vector(1.0, 1.0, 0.0, 1.0, 1.0, 1.0)
        wrench_selection = wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

        # ---- Operational-space controller -------------------------------------
        # The controller reads its FK and dynamics terms from the same model
        # the solver simulates.
        self.controller = ControllerOperationalSpace(
            self.model,
            joints=arm_joints,
            tool="tool_site",
            motion_stiffness=MOTION_KP,
            motion_damping=MOTION_KD,
            # Commands/gains, and the linear/angular selection frames below,
            # are all interpreted relative to this frame -- the table's top
            # surface, oriented with Z normal to the (tilted) table.
            operational_frame_pose_world=operational_frame_transform,
            use_inertia_decoupling=True,
            use_gravity_compensation=True,
            use_wrench_feedforward=True,
            motion_selection_axes=motion_selection,
            wrench_selection_axes=wrench_selection,
            linear_selection_frame_operational=wp.quat(0.0, 0.0, 0.0, 1.0),
            angular_selection_frame_operational=wp.quat(0.0, 0.0, 0.0, 1.0),
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

        # Side view: robot at the origin, table centered at x=0.5 -- looking
        # along -Y at their midpoint shows the robot and the tilted table's
        # profile together, instead of the default view from behind the table.
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.25, -1.8, 0.7), pitch=0.0, yaw=90.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.25, 0.0, 0.3))

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
        # Position: (x, y) along the table's tangent plane, relative to the
        # operational frame; z left at 0 (wrench-, not motion-, controlled).
        # Orientation left at home_pose's: composed with the tilted operational
        # frame, this keeps the gripper perpendicular to the table.
        desired_pose = self._home_pose.copy()[None, :]
        desired_pose[0, 0] = self.desired_x
        desired_pose[0, 1] = self.desired_y
        desired_pose[0, 2] = 0.0
        self._input.desired_tool_pose_operational.assign(desired_pose)
        # desired_wrench_world is genuinely world-frame, so "press into the
        # table" means force along the negative table normal in world, not
        # negative world Z (only the same thing before the table was tilted).
        press_force_world = -self.desired_force * self._table_normal_world
        desired_wrench_world = np.concatenate([press_force_world, np.zeros(3, dtype=np.float32)])
        self._input.desired_wrench_world.assign(desired_wrench_world[None, :].astype(np.float32))

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
            "Desired x [m]",
            self.desired_x,
            self._home_pos_operational[0] - XY_SLIDER_RANGE,
            self._home_pos_operational[0] + XY_SLIDER_RANGE,
        )
        _, self.desired_y = ui.slider_float(
            "Desired y [m]",
            self.desired_y,
            self._home_pos_operational[1] - XY_SLIDER_RANGE,
            self._home_pos_operational[1] + XY_SLIDER_RANGE,
        )
        _, self.desired_force = ui.slider_float("Desired press force [N]", self.desired_force, 0.0, FORCE_SLIDER_MAX)

        # Actual tool position, relative to the operational frame -- same
        # frame the x/y sliders above are expressed in.
        actual_pose_world = self.controller._tool_pose_world.numpy()[0]
        actual_pos_operational = self._table_rotation_np.T @ (actual_pose_world[:3] - self._table_position_np)
        # Force the table exerts on the fingers, summed and projected onto
        # the table normal; positive means the table is pushing back against
        # the commanded press.
        total_force_world = self.force_sensor.total_force.numpy().sum(axis=0)
        measured_force_normal = float(self._table_normal_world @ total_force_world)

        ui.text(f"actual x:   {actual_pos_operational[0]:.3f}   (desired {self.desired_x:.3f})")
        ui.text(f"actual y:   {actual_pos_operational[1]:.3f}   (desired {self.desired_y:.3f})")
        ui.text(f"measured press force: {measured_force_normal:.1f} N   (desired {self.desired_force:.1f} N)")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # The operational frame itself -- a fixed RGB axis triad, not tied
        # to any body, so you can see where the controller's (x, y) target
        # and press axis actually point on the tilted table.
        self.viewer.log_lines(
            "/operational_frame",
            self._operational_frame_gizmo_starts,
            self._operational_frame_gizmo_ends,
            self._operational_frame_gizmo_colors,
        )
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
