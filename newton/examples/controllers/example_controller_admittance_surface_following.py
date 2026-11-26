# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Controllers — Admittance Surface Following
#
# Demonstrates ControllerAdmittance on a position-controlled Franka Panda
# sliding a ball-tipped tool back and forth across a table while pressing
# on it with a set force -- without ever being told where the table is, or
# about the two speed bumps across its path. The reference pose sweeps
# along x at a fixed height well above the table; the admittance controller
# turns the measured contact force into a compliant pose that drops onto
# the table and rides up and over each bump.
#
# The controller stack is the classic one for position-controlled arms:
#
#   contact force --> ControllerAdmittance --> compliant tool pose
#                 --> ControllerDifferentialIK --> joint position targets
#                 --> the solver's joint PD
#
# The admittance output is bound directly to the IK's input array, so the
# two controllers compose with no copy in between, and the whole loop --
# contact sensing, both controllers, and physics -- runs at the physics
# rate inside one CUDA graph.
#
# The admittance gains differ per axis: x/y are a stiff spring-damper
# around the sweeping reference, while z has zero stiffness and a desired
# press force, which makes it a pure force-tracking axis -- the only steady
# state is "measured force == desired force", whatever the surface height.
# Its damping is also varied online (variable admittance): low in free
# space so the tool drops quickly, high in contact so the force loop stays
# well damped against the stiff arm and surface. A small kernel picks the
# damping from the measured force every step. Changing damping, unlike
# stiffness, never injects energy into the virtual mass-spring-damper, so
# switching it abruptly is safe.
#
# Sliders set the press force and both damping values; the GUI shows the
# measured (low-pass filtered) force. A line is drawn from the reference to the compliant
# pose: the admittance displacement.
#
# Command: python -m newton.examples controller_admittance_surface_following
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import Contacts, JointTargetMode
from newton.controllers import ControllerAdmittance, ControllerDifferentialIK
from newton.sensors import SensorContact

# ---------------------------------------------------------------------------
# Robot configuration
# ---------------------------------------------------------------------------

# Franka's standard "ready" pose.
FRANKA_READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
FRANKA_ARM_DOFS = len(FRANKA_READY_POSE)

# Joint position-control gains, making the Franka a stiff position-controlled
# robot -- the setting admittance control is designed for.
JOINT_KE = 4000.0  # [N·m/rad]
JOINT_KD = 200.0  # [N·m·s/rad]

# A small ball fixed to fr3_hand_tcp, offset out along its local +Z past
# the fingertip pads, as the pressing tool.
TOOL_BALL_RADIUS = 0.02
TOOL_BALL_OFFSET = 0.04
TOOL_SURFACE_FRICTION_CFG = newton.ModelBuilder.ShapeConfig(mu=0.1)

# ---------------------------------------------------------------------------
# Surface and reference
# ---------------------------------------------------------------------------

# A table whose top is at TABLE_TOP_HEIGHT, with two speed bumps across the
# sweep: large cylinders lying along y, sunk into the table so that only a
# low cap of each protrudes. The controller never sees any of it.
TABLE_TOP_HEIGHT = 0.30
TABLE_CENTER_X = 0.45
TABLE_HALF_EXTENTS = (0.25, 0.2, 0.15)  # a solid block down to the floor
BUMPS = ((0.39, 0.12, 0.02), (0.51, 0.08, 0.012))  # (x [m], cylinder radius [m], bump height [m])


def tool_center_height_on_surface(x):
    """Height [m] of the ball tool's center when resting on the table or a bump, at world x [m]."""
    height = TABLE_TOP_HEIGHT + TOOL_BALL_RADIUS
    for bump_x, bump_radius, bump_height in BUMPS:
        reach = bump_radius + TOOL_BALL_RADIUS
        if abs(x - bump_x) < reach:
            axis_height = TABLE_TOP_HEIGHT + bump_height - bump_radius
            height = max(height, axis_height + np.sqrt(reach**2 - (x - bump_x) ** 2))
    return height


# The reference sweeps x back and forth across the surface at a fixed
# height well above it, after easing in from the tool's starting pose.
REFERENCE_HEIGHT = 0.40
SWEEP_AMPLITUDE = 0.12
SWEEP_PERIOD = 10.0  # [s]
EASE_IN_DURATION = 1.0  # [s]

# ---------------------------------------------------------------------------
# Admittance gains (operational frame = world frame)
# ---------------------------------------------------------------------------

LATERAL_STIFFNESS = 1500.0  # [N/m]
LATERAL_MASS = 2.0  # [kg]
LATERAL_DAMPING = 2.0 * np.sqrt(LATERAL_STIFFNESS * LATERAL_MASS)  # critically damped
# Admittance control against a stiff contact needs enough virtual mass to
# low-pass the force loop: with a few kg the tool bounces on the table.
PRESS_MASS = 20.0  # [kg]
ROTATION_STIFFNESS = 50.0  # [N·m/rad]
ROTATION_MASS = 0.05  # [kg·m²]
ROTATION_DAMPING = 2.0 * np.sqrt(ROTATION_STIFFNESS * ROTATION_MASS)

DEFAULT_PRESS_FORCE = 10.0  # [N]
PRESS_FORCE_MAX = 30.0  # [N]
DEFAULT_FREE_DAMPING = 60.0  # [N·s/m]
DEFAULT_CONTACT_DAMPING = 150.0  # [N·s/m]
CONTACT_FORCE_THRESHOLD = 1.0  # [N]
FORCE_FILTER_TIME_CONSTANT = 0.02  # [s]


@wp.kernel
def _filter_contact_wrench_kernel(
    contact_force_on_tool: wp.array[wp.vec3],  # (robot_count,) environment on tool, world frame
    smoothing: float,  # first-order low-pass weight of the newest sample, in (0, 1]
    # outputs
    measured_wrench_world: wp.array[wp.spatial_vector],  # (robot_count,) tool on environment, filtered in place
):
    # ControllerAdmittance, like ControllerOperationalSpace, takes the
    # wrench the tool exerts on its environment: the sensor's reaction
    # force, negated (Newton's third law). The ball is centered on the tool
    # point and its low-friction contact forces pass close to its center,
    # so the moment about the tool point is negligible and left at zero.
    #
    # A stiff position-controlled arm on a rigid surface micro-bounces: the
    # contact force drops to zero for single steps while the tool never
    # visibly leaves the surface. Like a real force/torque signal, the
    # reading is low-pass filtered before it drives the controller.
    robot = wp.tid()
    sample = wp.spatial_vector(-contact_force_on_tool[robot], wp.vec3(0.0))
    measured_wrench_world[robot] = measured_wrench_world[robot] + smoothing * (sample - measured_wrench_world[robot])


@wp.kernel
def _schedule_press_damping_kernel(
    measured_wrench_world: wp.array[wp.spatial_vector],  # (robot_count,)
    damping_free: wp.array[wp.float32],  # (1,) press-axis damping without contact [N·s/m]
    damping_contact: wp.array[wp.float32],  # (1,) press-axis damping in contact [N·s/m]
    lateral_damping: float,
    rotation_damping: float,
    contact_force_threshold: float,
    # outputs
    virtual_damping: wp.array[wp.spatial_vector],  # (robot_count,)
):
    robot = wp.tid()
    press_damping = damping_free[0]
    if wp.length(wp.spatial_top(measured_wrench_world[robot])) > contact_force_threshold:
        press_damping = damping_contact[0]
    virtual_damping[robot] = wp.spatial_vector(
        lateral_damping, lateral_damping, press_damping, rotation_damping, rotation_damping, rotation_damping
    )


class Example:
    @staticmethod
    def create_parser():
        return newton.examples.create_parser()

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # Even, so state_0/state_1 are back in place after every frame, as
        # replaying the captured graph requires.
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # ---- Physics scene ---------------------------------------------------
        franka_urdf_path = str(newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf")
        builder = newton.ModelBuilder()
        arm_joints, tool_body, self._tool_site_transform = self._add_franka(builder, franka_urdf_path)
        self._tool_body = tool_body

        builder.add_shape_box(
            -1,
            xform=wp.transform(
                wp.vec3(TABLE_CENTER_X, 0.0, TABLE_TOP_HEIGHT - TABLE_HALF_EXTENTS[2]), wp.quat_identity()
            ),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=TOOL_SURFACE_FRICTION_CFG,
        )
        # Cylinders extend along their local z; turn it onto world y, and end
        # them just inside the table's sides (exactly flush would z-fight).
        across_sweep = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)
        for bump_x, bump_radius, bump_height in BUMPS:
            builder.add_shape_cylinder(
                -1,
                xform=wp.transform(wp.vec3(bump_x, 0.0, TABLE_TOP_HEIGHT + bump_height - bump_radius), across_sweep),
                radius=bump_radius,
                half_height=TABLE_HALF_EXTENTS[1] - 0.002,
                cfg=TOOL_SURFACE_FRICTION_CFG,
            )
        builder.add_ground_plane()

        # Arm DOFs are position-controlled by the solver's joint PD; the
        # finger DOFs keep the URDF's own targets and gains.
        for dof in range(FRANKA_ARM_DOFS):
            builder.joint_target_ke[dof] = JOINT_KE
            builder.joint_target_kd[dof] = JOINT_KD
            builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)

        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.control.joint_target_q.assign(self.model.joint_q)

        self.solver = newton.solvers.SolverMuJoCo(self.model, nconmax=200, njmax=200)

        self.force_sensor = SensorContact(self.model, sensing_bodies=[tool_body])
        self.contacts = Contacts(
            self.solver.get_max_contact_count(),
            0,
            requested_attributes=self.model.get_requested_contact_attributes(),
        )

        body_pose_world = wp.transform(*self.state_0.body_q.numpy()[tool_body].tolist())
        self._home_pose_world = np.array(body_pose_world * self._tool_site_transform, dtype=np.float32)

        # ---- Admittance controller -------------------------------------------
        # Stiffness and mass are fixed per axis; damping is live, rewritten
        # every frame by _schedule_press_damping_kernel. Zero stiffness on z
        # plus a desired wrench makes z a force-tracking axis.
        self.admittance = ControllerAdmittance(
            controlled_robot_count=1,
            virtual_stiffness=wp.spatial_vector(
                LATERAL_STIFFNESS,
                LATERAL_STIFFNESS,
                0.0,
                ROTATION_STIFFNESS,
                ROTATION_STIFFNESS,
                ROTATION_STIFFNESS,
            ),
            virtual_damping=None,
            virtual_mass=wp.spatial_vector(
                LATERAL_MASS, LATERAL_MASS, PRESS_MASS, ROTATION_MASS, ROTATION_MASS, ROTATION_MASS
            ),
            use_desired_wrench=True,
        )
        self._admittance_input = self.admittance.input()
        self._admittance_output = self.admittance.output()
        # Advance the admittance state in place.
        self._admittance_output.displacement_operational = self._admittance_input.displacement_operational
        self._admittance_output.displacement_twist_operational = self._admittance_input.displacement_twist_operational

        # ---- Differential IK controller --------------------------------------
        self.ik = ControllerDifferentialIK(
            self.model,
            joints=arm_joints,
            tool_sites="tool_site",
            # A full Gauss-Newton step toward the compliant pose each step:
            # joint_q_target = joint_q + bandwidth * dt * J⁺ · error.
            bandwidth=1.0 / self.sim_dt,
            damping=0.05,
            use_null_space_posture_control=True,
            null_space_stiffness=2.0,
            null_space_damping=0.05,
        )
        self._ik_input = self.ik.input()
        self._ik_output = self.ik.output()
        self._ik_input.q_des_null.assign(np.array(FRANKA_READY_POSE, dtype=np.float32))
        # The composition: the IK tracks the admittance's compliant pose
        # directly, reading the very array the admittance writes.
        self._ik_input.desired_tool_pose_world = self._admittance_output.compliant_tool_pose_world
        self._ik_output.joint_q_target = self.control.joint_target_q[self.ik.q_start]
        # Position mode ignores velocity targets, so the IK's own velocity
        # output goes to a scratch buffer.
        self._ik_output.joint_qd_target = wp.zeros(self.ik.total_controlled_dofs, dtype=wp.float32)

        # ---- GUI-driven parameters ---------------------------------------------
        self.press_force = DEFAULT_PRESS_FORCE
        self.free_damping = DEFAULT_FREE_DAMPING
        self.contact_damping = DEFAULT_CONTACT_DAMPING
        self._free_damping_buf = wp.array([self.free_damping], dtype=wp.float32)
        self._contact_damping_buf = wp.array([self.contact_damping], dtype=wp.float32)
        self.measured_press_force = 0.0
        # (time, measured press force, desired press force, tool x, tool z) per frame.
        self._log = []

        self._reference_line_starts = wp.zeros(1, dtype=wp.vec3)
        self._reference_line_ends = wp.zeros(1, dtype=wp.vec3)

        self._graph = None
        if self.device.is_cuda and self.admittance.is_graphable() and self.ik.is_graphable():
            with wp.ScopedCapture() as capture:
                self._gpu_step()
            self._graph = capture.graph

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.5, -1.05, 0.75), pitch=-20.0, yaw=90.0)

    @staticmethod
    def _add_franka(builder, urdf_path):
        """Load the Franka at the origin in its ready pose, and add its pressing-tool ball and site.

        Returns:
            Tuple of (arm joint indices, fr3_hand_tcp body index, tool site's
            body-local transform).
        """
        joint_count_before = builder.joint_count
        body_count_before = builder.body_count
        builder.add_urdf(urdf_path, xform=wp.transform_identity(), floating=False)

        # fr3_joint1..7 follow the URDF's two fixed base/mount joints.
        arm_joints = [joint_count_before + 2 + i for i in range(FRANKA_ARM_DOFS)]
        for coord, angle in enumerate(FRANKA_READY_POSE):
            builder.joint_q[coord] = angle

        # Body 11 this URDF adds is fr3_hand_tcp, whose local +Z points away
        # from the fingers.
        tool_body = body_count_before + 11
        tool_site_transform = wp.transform(wp.vec3(0.0, 0.0, TOOL_BALL_OFFSET), wp.quat_identity())
        builder.add_shape_sphere(
            tool_body, xform=tool_site_transform, radius=TOOL_BALL_RADIUS, cfg=TOOL_SURFACE_FRICTION_CFG
        )
        builder.add_site(tool_body, xform=tool_site_transform, label="tool_site")
        return arm_joints, tool_body, tool_site_transform

    def _gpu_step(self):
        """Sense, control, and simulate for one frame, all on the GPU. Safe to graph-capture."""
        for _ in range(self.sim_substeps):
            self.solver.update_contacts(self.contacts, self.state_0)
            self.force_sensor.update(self.state_0, self.contacts)
            wp.launch(
                _filter_contact_wrench_kernel,
                dim=1,
                inputs=[self.force_sensor.total_force, self.sim_dt / (FORCE_FILTER_TIME_CONSTANT + self.sim_dt)],
                outputs=[self._admittance_input.measured_wrench_world],
            )
            wp.launch(
                _schedule_press_damping_kernel,
                dim=1,
                inputs=[
                    self._admittance_input.measured_wrench_world,
                    self._free_damping_buf,
                    self._contact_damping_buf,
                    LATERAL_DAMPING,
                    ROTATION_DAMPING,
                    CONTACT_FORCE_THRESHOLD,
                ],
                outputs=[self._admittance_input.virtual_damping],
            )
            self.admittance.step(inputs=self._admittance_input, outputs=self._admittance_output, dt=self.sim_dt)
            # state_0 and state_1 swap every substep, so rebind the IK's
            # state ports to whichever currently holds the latest state; a
            # graph capture records each launch with the arrays bound then.
            self._ik_input.joint_q = self.state_0.joint_q
            self._ik_input.joint_qd = self.state_0.joint_qd
            self.ik.step(inputs=self._ik_input, outputs=self._ik_output, dt=self.sim_dt)

            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _reference_pose(self, t):
        """Reference tool pose [m, unitless quaternion] at time t: ease in from home, then sweep along x."""
        sweep_x = TABLE_CENTER_X - SWEEP_AMPLITUDE * np.cos(2.0 * np.pi * t / SWEEP_PERIOD)
        target = np.array([sweep_x, 0.0, REFERENCE_HEIGHT], dtype=np.float32)
        blend = 0.5 - 0.5 * np.cos(np.pi * min(t / EASE_IN_DURATION, 1.0))
        pose = self._home_pose_world.copy()
        pose[:3] = (1.0 - blend) * self._home_pose_world[:3] + blend * target
        return pose

    def step(self):
        self._admittance_input.reference_tool_pose_operational.assign(self._reference_pose(self.sim_time)[None])
        self._admittance_input.desired_wrench_world.assign(
            np.array([[0.0, 0.0, -self.press_force, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
        self._free_damping_buf.fill_(self.free_damping)
        self._contact_damping_buf.fill_(self.contact_damping)

        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._gpu_step()

        self.sim_time += self.frame_dt
        self.measured_press_force = float(-self._admittance_input.measured_wrench_world.numpy()[0, 2])
        tool_pose = wp.transform(*self.state_0.body_q.numpy()[self._tool_body].tolist()) * self._tool_site_transform
        tool_x, _, tool_z = wp.transform_get_translation(tool_pose)
        self._log.append((self.sim_time, self.measured_press_force, self.press_force, tool_x, tool_z))

    def gui(self, ui):
        _, self.press_force = ui.slider_float("press force [N]", self.press_force, 0.0, PRESS_FORCE_MAX)
        _, self.free_damping = ui.slider_float("free-space z damping [N·s/m]", self.free_damping, 10.0, 400.0)
        _, self.contact_damping = ui.slider_float("contact z damping [N·s/m]", self.contact_damping, 100.0, 2000.0)
        ui.text(f"measured press force: {self.measured_press_force:.1f} N   (desired {self.press_force:.1f} N)")

    def render(self):
        reference = self._admittance_input.reference_tool_pose_operational.numpy()[0, :3]
        compliant = self._admittance_output.compliant_tool_pose_world.numpy()[0, :3]
        self._reference_line_starts.assign(reference[None])
        self._reference_line_ends.assign(compliant[None])

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines(
            "/admittance_displacement", self._reference_line_starts, self._reference_line_ends, (1.0, 0.6, 0.0)
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify the tool held the press force while sliding over the table and both bumps."""
        joint_q = self.state_0.joint_q.numpy()
        assert np.all(np.isfinite(joint_q)), f"joint_q has NaN/Inf: {joint_q}"

        # Skip the drop onto the table; check the sweep that follows.
        log = np.array([row for row in self._log if row[0] > 3.0])
        assert log.shape[0] > 0, "the example must run for more than 3 s of simulated time to be tested"
        _, force, desired_force, tool_x, tool_z = log.T

        # Force tracking: on average at the setpoint, and in contact
        # throughout, apart from brief transients where a bump meets the
        # table and the surface slope changes abruptly.
        mean_force_error = abs(np.mean(force) - np.mean(desired_force))
        assert mean_force_error < 1.0, f"mean press force is off by {mean_force_error:.2f} N"
        assert np.mean(np.abs(force - desired_force)) < 5.0, "press force fluctuates too much"
        in_contact = np.mean(force > 1.0)
        assert in_contact > 0.95, f"the tool was in contact for only {100.0 * in_contact:.1f}% of the sweep"

        # The tool rides on the surface -- table and bumps alike -- rather
        # than hovering at the reference height or digging in.
        surface_error = tool_z - np.array([tool_center_height_on_surface(x) for x in tool_x])
        assert np.max(np.abs(surface_error)) < 0.005, f"tool left the surface by {np.max(np.abs(surface_error)):.4f} m"
        crossed = [np.any(np.abs(tool_x - bump_x) < 0.005) for bump_x, _, _ in BUMPS]
        assert all(crossed), "the sweep did not carry the tool over both bumps"


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
