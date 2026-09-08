# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Operational Space Hybrid Force/Motion
#
# Demonstrates ControllerOperationalSpace on two real, heterogeneous robots
# at once -- a 7-DOF Franka Panda arm (redundant against the 6D task) and a
# 6-DOF UR10 arm (not redundant) -- each independently pressing a tool into
# its own table, with its position along the table steered interactively.
# The Franka's table is tilted 45 degrees toward it; the UR10's is flat.
# One controller call handles both, each robot resolved through its own
# tool site, Jacobian, and operational frame.
#
# Each operational frame is placed on its table's top surface, oriented so
# its Z axis is normal to the table -- so the same operational-frame-relative
# command works regardless of the table's tilt, and the axis triad drawn
# each frame lets you see exactly where the controller thinks "the table"
# and "into the table" are.
#
# Each operational frame's local Z (into its table) is wrench-controlled
# with a feedforward press force plus feedback from the measured contact
# force, plus a light superimposed motion-control term damping out
# bounce; the other five task axes are purely motion-controlled, tracking
# a desired (x, y) on the table's surface.
#
# Two sets of three sliders (x, y, press force) let you steer each robot's
# commanded task directly; a SensorContact reads back the actual contact
# force each tool exerts on its table, fed into the controller as wrench
# feedback and shown in the GUI alongside the commanded force.
#
# Command: python -m newton.examples controller_operational_space_hybrid_force_motion
###########################################################################

from dataclasses import dataclass

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

# Franka's standard "ready" pose.
FRANKA_READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
FRANKA_ARM_DOFS = len(FRANKA_READY_POSE)  # 7; all controlled by the OSC controller
FRANKA_BASE_POSITION = wp.vec3(0.0, 0.0, 0.0)

# A UR10 configuration reaching forward and down, chosen (via forward
# kinematics) to put its tool roughly chest-height in front of the base,
# the same reach scale as the Franka's.
UR10_READY_POSE = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
UR10_ARM_DOFS = len(UR10_READY_POSE)  # 6; no redundant DOF, unlike the Franka
UR10_BASE_POSITION = wp.vec3(0.0, 1.8, 0.0)  # separated from the Franka along Y

# The UR10 asset is a bare arm with no gripper; a small capsule fixed to
# its wrist flange (ee_link) stands in as its pressing tool.
TOOL_CYLINDER_RADIUS = 0.02
TOOL_CYLINDER_HALF_HEIGHT = 0.05

# A small ball fixed to the Franka's fr3_hand_tcp, offset 0.04m out along
# its local +Z past the fingertip pads -- rounds off what would otherwise
# be a flat-fingered contact.
FRANKA_BALL_RADIUS = 0.02
FRANKA_BALL_OFFSET = 0.04

# Slider ranges, centered on each tool's actual starting (x, y) in its own
# operational frame -- so the initial commanded position matches where the
# tool already is, on its table's surface.
XY_SLIDER_RANGE = 0.15  # [m]
FORCE_SLIDER_MAX = 80.0  # [N]

# Each table: a box of half-height TABLE_HEIGHT/2 and half-footprint
# TABLE_HALF_EXTENT, offset TABLE_OFFSET from its robot's own base, tilted
# TABLE_TILT_ANGLE about world Y so its top surface faces up and toward the
# robot. TABLE_OFFSET/TABLE_ROTATION give the box's own (center) pose
# relative to the base; the operational frame is built from these but
# offset onto the top surface.
TABLE_HALF_EXTENT = 0.35
TABLE_HEIGHT = 0.15
TABLE_TILT_ANGLE = np.pi / 4.0
TABLE_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -TABLE_TILT_ANGLE)
TABLE_OFFSET = wp.vec3(0.5, 0.0, np.sqrt(TABLE_HALF_EXTENT**2 + (TABLE_HEIGHT / 2.0) ** 2) + 0.05)
# The UR10's table is flat (0 degrees), unlike the Franka's tilted one.
UR10_TABLE_ROTATION = wp.quat_identity()
# UR10_READY_POSE's tool tip reaches further forward than the Franka's does
# at TABLE_OFFSET's 0.5m -- push the UR10's table further out along its own
# reach direction so it still clears the table box with margin.
UR10_TABLE_OFFSET = TABLE_OFFSET + wp.vec3(0.3, 0.0, 0.0)

# Low friction between each tool and its table -- the wrench controller
# handles pressing into the surface; sliding shouldn't cost extra drag force
# on top of that.
TOOL_TABLE_FRICTION_CFG = newton.ModelBuilder.ShapeConfig(mu=0.05)

# Gains -- use_inertia_decoupling=True (the default), so these are in the
# mass-normalized (acceleration) domain: [1/s^2] for stiffness, [1/s] for
# damping. Both robots use full inertia decoupling and equal gains; neither
# needs use_partial_inertia_decoupling or softened gains to stay stable.
FRANKA_MOTION_KP = 600.0
FRANKA_MOTION_KD = 2.0 * FRANKA_MOTION_KP**0.5  # critically damped
UR10_MOTION_KP = 600.0
UR10_MOTION_KD = 2.0 * UR10_MOTION_KP**0.5  # critically damped
# A much lighter motion gain layered onto the press axis (Z) alongside the
# wrench control there, targeting the table surface (z=0 in the operational
# frame) -- its damping term pulls the tool's velocity toward zero, damping
# out the bounce a pure feedforward+feedback wrench term otherwise leaves
# unchecked, without fighting the commanded press force at steady state.
Z_MOTION_KP = 50.0
Z_MOTION_KD = 2.0 * Z_MOTION_KP**0.5  # critically damped
# Dimensionless: multiplies a wrench error (already in N/N*m) directly, not a pose error.
WRENCH_KP = 0.5


@dataclass
class _RobotRig:
    """Per-robot table geometry, gizmo, and slider/measurement state -- identical shape for each robot."""

    label: str
    operational_position_np: np.ndarray
    operational_frame_transform: wp.transform
    table_rotation_np: np.ndarray
    table_normal_world: np.ndarray
    home_pose: np.ndarray
    home_pos_operational: np.ndarray
    desired_x: float
    desired_y: float
    desired_force: float
    measured_force_normal: float
    gizmo_starts: wp.array
    gizmo_ends: wp.array
    gizmo_colors: wp.array


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
        franka_urdf_path = str(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf")
        ur10_asset_file = str(newton.utils.download_asset("universal_robots_ur10") / "usd/ur10_instanceable.usda")
        builder = newton.ModelBuilder()

        franka_joints, franka_coords, franka_tool_body, franka_tool_site_transform, franka_finger_dofs = (
            self._add_franka(builder, franka_urdf_path, FRANKA_BASE_POSITION)
        )
        ur10_joints, ur10_coords, ur10_tool_body, ur10_tool_site_transform = self._add_ur10(
            builder, ur10_asset_file, UR10_BASE_POSITION
        )
        self._franka_coords = franka_coords
        self._ur10_coords = ur10_coords

        builder.add_ground_plane()

        franka_table_body = builder.add_link()
        builder.add_shape_box(
            franka_table_body,
            hx=TABLE_HALF_EXTENT,
            hy=TABLE_HALF_EXTENT,
            hz=TABLE_HEIGHT / 2.0,
            cfg=TOOL_TABLE_FRICTION_CFG,
        )
        franka_table_joint = builder.add_joint_fixed(
            parent=-1,
            child=franka_table_body,
            parent_xform=wp.transform(FRANKA_BASE_POSITION + TABLE_OFFSET, TABLE_ROTATION),
        )
        builder.add_articulation([franka_table_joint], label="franka_table")

        ur10_table_body = builder.add_link()
        builder.add_shape_box(
            ur10_table_body,
            hx=TABLE_HALF_EXTENT,
            hy=TABLE_HALF_EXTENT,
            hz=TABLE_HEIGHT / 2.0,
            cfg=TOOL_TABLE_FRICTION_CFG,
        )
        ur10_table_joint = builder.add_joint_fixed(
            parent=-1,
            child=ur10_table_body,
            parent_xform=wp.transform(UR10_BASE_POSITION + UR10_TABLE_OFFSET, UR10_TABLE_ROTATION),
        )
        builder.add_articulation([ur10_table_joint], label="ur10_table")

        # Every arm DOF is put into EFFORT mode with zero implicit-PD gains,
        # so joint_f -- the OSC controller's torque output -- is its sole
        # driver. The Franka's finger DOFs are skipped, left at the URDF's
        # own POSITION-mode target and gains, so the solver's implicit PD
        # holds them closed instead of drifting open under a zeroed torque.
        # The tables (fixed-jointed, no DOFs) are unaffected either way.
        for i in range(builder.joint_dof_count):
            if i in franka_finger_dofs:
                continue
            builder.joint_target_ke[i] = 0.0
            builder.joint_target_kd[i] = 0.0
            builder.joint_target_mode[i] = int(JointTargetMode.EFFORT)

        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Contacts stay enabled (the default) so each tool's own collision
        # geometry actually presses against, and is resisted by, its table.
        # nconmax/njmax raised above their defaults: the Franka fingers'
        # meshes generate more simultaneous contact points against a flat
        # plane than the default budgets for, and two robots' worth of
        # contacts/constraints need more headroom than one.
        self.solver = newton.solvers.SolverMuJoCo(self.model, nconmax=400, njmax=400)

        # SensorContact + Contacts is Newton's contact-force readback API (see
        # example_sensor_contact.py) -- reads back the actual contact force
        # each tool's ball exerts on its table, fed into the controller as
        # wrench feedback and shown in the GUI alongside the commanded
        # force. One sensor covers both robots; total_force's rows are
        # ordered to match sensing_bodies below (Franka's ball, then UR10's).
        self.force_sensor = SensorContact(self.model, sensing_bodies=[franka_tool_body, ur10_tool_body])
        self.contacts = Contacts(
            self.solver.get_max_contact_count(),
            0,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        # The tool site's world pose, not the raw body's -- the two only
        # coincide when the site's own body-local transform is identity
        # (true for Franka's, not for UR10's, whose site is offset from
        # ee_link out to the pressing tool's tip).
        franka_body_pose_world = wp.transform(*self.state_0.body_q.numpy()[franka_tool_body].tolist())
        franka_home_pose_world = np.array(franka_body_pose_world * franka_tool_site_transform, dtype=np.float32)
        ur10_body_pose_world = wp.transform(*self.state_0.body_q.numpy()[ur10_tool_body].tolist())
        ur10_home_pose_world = np.array(ur10_body_pose_world * ur10_tool_site_transform, dtype=np.float32)
        self.rigs = [
            self._build_rig("Franka", np.array(FRANKA_BASE_POSITION, dtype=np.float32), franka_home_pose_world),
            self._build_rig(
                "UR10",
                np.array(UR10_BASE_POSITION, dtype=np.float32),
                ur10_home_pose_world,
                UR10_TABLE_OFFSET,
                UR10_TABLE_ROTATION,
            ),
        ]

        # The operational frame's local Z (below) is the press axis (index
        # 2); every axis is motion-controlled, and Z is additionally
        # wrench-controlled -- motion and wrench control are superimposed
        # there, rather than one replacing the other (see Z_MOTION_KP
        # above). The linear and angular selection frames (below) are both
        # left at identity relative to the operational frame, so "axis 2"
        # here is literally each table's normal -- independent of the
        # tool's own orientation. Same selection pattern for both robots,
        # since both tables use the same convention.
        motion_selection = wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        wrench_selection = wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

        # ---- Operational-space controller -------------------------------------
        # One controller call handles both robots; joints lists robot 0's
        # (Franka's) controlled joints first, then robot 1's (UR10's),
        # matching operational_frame_pose_world's per-robot ordering below.
        # The controller reads its FK and dynamics terms from the same model
        # the solver simulates.
        self.controller = ControllerOperationalSpace(
            self.model,
            joints=franka_joints + ur10_joints,
            tool_sites="tool_site",
            motion_stiffness=wp.array(
                [
                    wp.spatial_vector(
                        FRANKA_MOTION_KP,
                        FRANKA_MOTION_KP,
                        Z_MOTION_KP,
                        FRANKA_MOTION_KP,
                        FRANKA_MOTION_KP,
                        FRANKA_MOTION_KP,
                    ),
                    wp.spatial_vector(
                        UR10_MOTION_KP, UR10_MOTION_KP, Z_MOTION_KP, UR10_MOTION_KP, UR10_MOTION_KP, UR10_MOTION_KP
                    ),
                ],
                dtype=wp.spatial_vector,
                device=self.device,
            ),
            motion_damping=wp.array(
                [
                    wp.spatial_vector(
                        FRANKA_MOTION_KD,
                        FRANKA_MOTION_KD,
                        Z_MOTION_KD,
                        FRANKA_MOTION_KD,
                        FRANKA_MOTION_KD,
                        FRANKA_MOTION_KD,
                    ),
                    wp.spatial_vector(
                        UR10_MOTION_KD, UR10_MOTION_KD, Z_MOTION_KD, UR10_MOTION_KD, UR10_MOTION_KD, UR10_MOTION_KD
                    ),
                ],
                dtype=wp.spatial_vector,
                device=self.device,
            ),
            # Commands/gains, and the linear/angular selection frames below,
            # are all interpreted relative to each robot's own frame -- its
            # table's top surface, oriented with Z normal to that table.
            operational_frame_pose_world=wp.array(
                [rig.operational_frame_transform for rig in self.rigs], dtype=wp.transform, device=self.device
            ),
            use_wrench_feedforward=True,
            use_wrench_feedback=True,
            wrench_stiffness=WRENCH_KP,
            motion_selection_axes=motion_selection,
            wrench_selection_axes=wrench_selection,
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
        self._input.desired_twist_operational.assign(np.zeros((2, 6), dtype=np.float32))

        self._graph = None
        if self.controller.is_graphable() and self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._gpu_step()
            self._graph = capture.graph

        # Pulled back and to the side so both robots and tables (Franka at
        # y=0, UR10 at y=1.8) are in view together.
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(-2.1, 0.9, 3.4), pitch=-15.0, yaw=15.0)
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(0.4, 0.9, 0.4))

        self.viewer.set_model(self.model)

    def _build_rig(
        self,
        label,
        base_position_np,
        home_pose_world,
        table_offset=TABLE_OFFSET,
        table_rotation=TABLE_ROTATION,
    ):
        """Per-robot table geometry, gizmo, and home-pose/slider state -- identical setup for each robot."""
        table_position_np = base_position_np + np.array(table_offset, dtype=np.float32)
        table_rotation_np = np.array(wp.quat_to_matrix(table_rotation), dtype=np.float32).reshape(3, 3)
        table_normal_world = table_rotation_np @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        operational_position_np = table_position_np + table_normal_world * (TABLE_HEIGHT / 2.0)

        # Desired orientation, relative to the (possibly tilted) operational
        # frame, is computed so it composes back to exactly the tool's
        # actual starting world orientation -- zero initial orientation
        # error, matching the zero initial position error below, rather
        # than commanding a sudden reorientation snap at startup.
        home_pose = home_pose_world.copy()
        home_orientation_world = wp.quat(*home_pose[3:7].tolist())
        desired_orientation_operational = wp.quat_inverse(table_rotation) * home_orientation_world
        home_pose[3:7] = np.array(desired_orientation_operational, dtype=np.float32)

        # x/y sliders offset the target along the table's tangent plane,
        # relative to the operational frame's own origin. Initialized to the
        # tool's actual starting (x, y) in that same frame -- zero initial
        # error -- rather than to the origin, which would otherwise be a
        # sudden, large initial position command. z is left at 0 -- the
        # table surface -- since that axis's light motion control (see
        # Z_MOTION_KP) is there only to damp the press, not to reposition it.
        home_pos_operational = table_rotation_np.T @ (home_pose_world[:3] - operational_position_np)

        axis_length = TABLE_HALF_EXTENT + 0.1
        axis_tips = operational_position_np + axis_length * table_rotation_np.T
        gizmo_starts = wp.array([operational_position_np] * 3, dtype=wp.vec3, device=self.device)
        gizmo_ends = wp.array(axis_tips, dtype=wp.vec3, device=self.device)
        gizmo_colors = wp.array(
            [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)], dtype=wp.vec3, device=self.device
        )

        return _RobotRig(
            label=label,
            operational_position_np=operational_position_np,
            operational_frame_transform=wp.transform(wp.vec3(*operational_position_np.tolist()), table_rotation),
            table_rotation_np=table_rotation_np,
            table_normal_world=table_normal_world,
            home_pose=home_pose,
            home_pos_operational=home_pos_operational,
            desired_x=float(home_pos_operational[0]),
            desired_y=float(home_pos_operational[1]),
            desired_force=0.0,
            measured_force_normal=0.0,
            gizmo_starts=gizmo_starts,
            gizmo_ends=gizmo_ends,
            gizmo_colors=gizmo_colors,
        )

    @staticmethod
    def _add_franka(builder, urdf_path, base_position):
        """Load one Franka at base_position, set its ready pose, and add its pressing-tool ball and site.

        Returns:
            Tuple of (arm joint indices, arm coordinate indices, fr3_hand_tcp
            body index, tool site's body-local transform, finger DOF
            indices).
        """
        joint_count_before = builder.joint_count
        coord_count_before = builder.joint_coord_count
        dof_count_before = builder.joint_dof_count
        body_count_before = builder.body_count
        builder.add_urdf(urdf_path, xform=wp.transform(base_position, wp.quat_identity()), floating=False)

        # fr3_joint1..7 are the first 7 non-fixed joints after the (fixed,
        # 0-coordinate) base/mount joints this URDF starts with; the finger
        # joints follow. Joint indices are offset by the 2 fixed joints, but
        # coordinate indices are not, since a fixed joint contributes no
        # coordinates. Indices are relative to this call since add_urdf
        # appends them.
        arm_joints = [joint_count_before + 2 + i for i in range(FRANKA_ARM_DOFS)]
        arm_coords = list(range(coord_count_before, coord_count_before + FRANKA_ARM_DOFS))
        for coord, angle in zip(arm_coords, FRANKA_READY_POSE, strict=True):
            builder.joint_q[coord] = angle

        # fr3_finger_joint1/2, the two single-coordinate/single-DOF prismatic
        # joints immediately after the arm joints (and the 3 fixed joints
        # between the arm and the hand, which contribute no coordinates or
        # DOFs).
        finger_dofs = [dof_count_before + FRANKA_ARM_DOFS, dof_count_before + FRANKA_ARM_DOFS + 1]

        # Body 11 (0-based) this URDF adds is fr3_hand_tcp, the fixed frame
        # between the fingers -- give it a small ball as the pressing tool
        # (rounds off what would otherwise be a flat-fingered contact), and
        # put the tool site at the ball's center. fr3_hand_tcp's own local
        # +Z already points away from the fingers, so unlike the UR10 no
        # axis correction is needed, just an offset out past the fingertip
        # pads.
        tool_body = body_count_before + 11
        builder.add_shape_sphere(
            tool_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, FRANKA_BALL_OFFSET), wp.quat_identity()),
            radius=FRANKA_BALL_RADIUS,
            cfg=TOOL_TABLE_FRICTION_CFG,
        )
        tool_site_transform = wp.transform(wp.vec3(0.0, 0.0, FRANKA_BALL_OFFSET), wp.quat_identity())
        builder.add_site(tool_body, xform=tool_site_transform, label="tool_site")

        return arm_joints, arm_coords, tool_body, tool_site_transform, finger_dofs

    @staticmethod
    def _add_ur10(builder, asset_file, base_position):
        """Load one UR10 at base_position, set a ready pose, and add a pressing-tool site at its wrist.

        Returns:
            Tuple of (arm joint indices, arm coordinate indices, tool body
            index, tool site's body-local transform).
        """
        joint_count_before = builder.joint_count
        coord_count_before = builder.joint_coord_count
        body_count_before = builder.body_count
        builder.add_usd(
            asset_file,
            xform=wp.transform(base_position, wp.quat_identity()),
            floating=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # shoulder_pan..wrist_3 are the 6 non-fixed joints after the (fixed,
        # 0-coordinate) base mount joint this asset starts with; the fixed
        # ee_joint follows. Indices are relative to this call since add_usd
        # appends them.
        arm_joints = [joint_count_before + 1 + i for i in range(UR10_ARM_DOFS)]
        arm_coords = list(range(coord_count_before, coord_count_before + UR10_ARM_DOFS))
        for coord, angle in zip(arm_coords, UR10_READY_POSE, strict=True):
            builder.joint_q[coord] = angle

        # Body 7 (0-based) this asset adds is ee_link, the fixed wrist-flange
        # frame -- give it a small capsule (rounded ends slide across a
        # surface more easily than a flat-ended cylinder) as the pressing
        # tool, and put the tool site at its tip (the point that actually
        # presses). ee_link's own local axes aren't necessarily aligned with
        # the capsule's press direction, so the site's transform (unlike
        # Franka's identity one) is a real, non-identity offset -- callers
        # need it to resolve the site's actual world pose, not ee_link's own.
        # The wrist's actual outward direction (away from wrist_3_link, where
        # ee_link's own fixed joint offset points) is ee_link's local +X, not
        # +Z. A capsule shape extends along its own local Z by default, so
        # its xform below both rotates that Z onto ee_link's local +X (a
        # +90 degree turn about Y) and offsets it out along that same +X.
        tool_body = body_count_before + 7
        tool_direction_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2.0)
        builder.add_shape_capsule(
            tool_body,
            xform=wp.transform(wp.vec3(TOOL_CYLINDER_HALF_HEIGHT, 0.0, 0.0), tool_direction_rotation),
            radius=TOOL_CYLINDER_RADIUS,
            half_height=TOOL_CYLINDER_HALF_HEIGHT,
            cfg=TOOL_TABLE_FRICTION_CFG,
        )
        # The site sits at the capsule's rounded tip: half_height along its
        # axis, plus the radius of the rounded cap itself.
        tool_site_transform = wp.transform(
            wp.vec3(2.0 * TOOL_CYLINDER_HALF_HEIGHT + TOOL_CYLINDER_RADIUS, 0.0, 0.0), wp.quat_identity()
        )
        builder.add_site(tool_body, xform=tool_site_transform, label="tool_site")

        return arm_joints, arm_coords, tool_body, tool_site_transform

    def _gpu_step(self):
        """Pure GPU work: controller step + physics substeps. Safe to graph-capture."""
        self.controller.step(inputs=self._input, outputs=self._output, dt=self.sim_dt)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        # Force feedback needs last frame's measured contact force before
        # this frame's controller.step() runs; SensorContact.update() isn't
        # graph-capturable, so it has to happen here in Python, first.
        self.force_sensor.update(self.state_0, self.contacts)
        # One sensing body per robot (Franka's ball, then UR10's), matching
        # sensing_bodies' order above.
        per_robot_force_world = self.force_sensor.total_force.numpy()

        measured_wrench_world = np.zeros((2, 6), dtype=np.float32)
        desired_pose = np.zeros((2, 7), dtype=np.float32)
        desired_wrench_world = np.zeros((2, 6), dtype=np.float32)
        for i, rig in enumerate(self.rigs):
            rig.measured_force_normal = float(rig.table_normal_world @ per_robot_force_world[i])
            # SensorContact reports the reaction force the table exerts on
            # the tool (Newton's third law), the opposite sign of
            # desired_wrench_world (force the tool exerts on the table) --
            # negate to match.
            measured_wrench_world[i, :3] = -per_robot_force_world[i]

            # Sliders drive the target directly -- read in gui(), applied
            # here. Cannot be graph-captured (assign() is, but the desired
            # values themselves come from Python-side UI state read after
            # capture). Position: (x, y) along the table's tangent plane,
            # relative to the operational frame; z left at 0 (the table
            # surface -- its light motion control only damps the press,
            # see Z_MOTION_KP). Orientation left at home_pose's: composed
            # with the tilted operational frame, this keeps the tool
            # perpendicular to the table.
            desired_pose[i] = rig.home_pose
            desired_pose[i, 0] = rig.desired_x
            desired_pose[i, 1] = rig.desired_y
            desired_pose[i, 2] = 0.0

            # desired_wrench_world is genuinely world-frame, so "press into
            # the table" means force along the negative table normal in
            # world, not negative world Z (only the same thing before the
            # table was tilted).
            desired_wrench_world[i, :3] = -rig.desired_force * rig.table_normal_world

        self._input.measured_wrench_world.assign(measured_wrench_world)
        self._input.desired_tool_pose_operational.assign(desired_pose)
        self._input.desired_wrench_world.assign(desired_wrench_world)

        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._gpu_step()

        self.sim_time += self.frame_dt

    def gui(self, ui):
        # The controller's own resolved tool pose, as of last step()'s FK --
        # the same site pose used to compute the task-space error it acted on.
        tool_pose_world = self.controller.tool_pose_world.numpy()
        for i, rig in enumerate(self.rigs):
            _, rig.desired_x = ui.slider_float(
                f"{rig.label} desired x [m]",
                rig.desired_x,
                rig.home_pos_operational[0] - XY_SLIDER_RANGE,
                rig.home_pos_operational[0] + XY_SLIDER_RANGE,
            )
            _, rig.desired_y = ui.slider_float(
                f"{rig.label} desired y [m]",
                rig.desired_y,
                rig.home_pos_operational[1] - XY_SLIDER_RANGE,
                rig.home_pos_operational[1] + XY_SLIDER_RANGE,
            )
            _, rig.desired_force = ui.slider_float(
                f"{rig.label} desired press force [N]", rig.desired_force, 0.0, FORCE_SLIDER_MAX
            )

            # Actual tool site's world position, relative to the operational
            # frame -- the same frame the x/y sliders above are expressed in.
            actual_pos_operational = rig.table_rotation_np.T @ (tool_pose_world[i, :3] - rig.operational_position_np)

            ui.text(f"{rig.label} actual x:   {actual_pos_operational[0]:.3f}   (desired {rig.desired_x:.3f})")
            ui.text(f"{rig.label} actual y:   {actual_pos_operational[1]:.3f}   (desired {rig.desired_y:.3f})")
            ui.text(
                f"{rig.label} measured press force: {rig.measured_force_normal:.1f} N   "
                f"(desired {rig.desired_force:.1f} N)"
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Each operational frame itself -- a fixed RGB axis triad, not tied
        # to any body, so you can see where each controller's (x, y) target
        # and press axis actually point on its table.
        for i, rig in enumerate(self.rigs):
            self.viewer.log_lines(f"/operational_frame_{i}", rig.gizmo_starts, rig.gizmo_ends, rig.gizmo_colors)
        self.viewer.end_frame()

    def test_final(self):
        """Verify both robots settled into a stable, finite configuration."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"
        assert np.all(np.isfinite(joint_qd)), f"joint_qd has NaN/Inf: {joint_qd}"

        # The Franka's motion-control task holds it near its starting pose,
        # so proximity to it is a meaningful check.
        franka_q = joint_q[self._franka_coords]
        franka_ready_q = np.array(FRANKA_READY_POSE, dtype=np.float32)
        assert np.all(np.abs(franka_q - franka_ready_q) < 1.5), (
            f"Franka arm joints drifted far from its starting configuration: {franka_q}"
        )

        # The UR10 has no redundant DOF, so unlike the Franka it may settle
        # at a different, equally valid joint configuration for the same
        # task-space target -- what's checked instead is boundedness rather
        # than full convergence. Settled joint velocities are O(1e-5); 1.0
        # still leaves ample margin while catching real divergence.
        ur10_qd = joint_qd[self._ur10_coords]
        assert np.all(np.abs(ur10_qd) < 1.0), f"UR10 arm joints diverged: {ur10_qd}"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
