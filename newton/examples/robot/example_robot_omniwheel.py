# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Four-wheel warehouse robot with passive barrel rollers."""

import math

import numpy as np
import warp as wp

import newton


@wp.kernel
def set_mecanum_targets(
    target_qd: wp.array[float],
    drive_dofs: wp.array[int],
    command: wp.vec3,
    wheel_radius: float,
    turn_radius: float,
):
    """Map a chassis velocity command to four wheel speeds."""
    wheel = wp.tid()
    linear_speed = 0.0
    if wheel == 0:
        linear_speed = command[0] - command[1] - turn_radius * command[2]
    elif wheel == 1:
        linear_speed = command[0] + command[1] + turn_radius * command[2]
    elif wheel == 2:
        linear_speed = command[0] + command[1] - turn_radius * command[2]
    else:
        linear_speed = command[0] - command[1] + turn_radius * command[2]

    # Positive rotation about the common +Y axle moves the wheel toward -X.
    target_qd[drive_dofs[wheel]] = -linear_speed / wheel_radius


@wp.kernel
def set_ballbot_torques(
    accelerometer: wp.array[wp.vec3],
    gyroscope: wp.array[wp.vec3],
    estimated_up: wp.array[wp.vec3],
    filtered_lean_target: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_qd: wp.array[float],
    joint_f: wp.array[float],
    body: int,
    ball: int,
    target_position: wp.vec3,
    command: wp.array[wp.vec3],
    drive_dofs: wp.array[int],
    drive_directions: wp.array[wp.vec3],
    gains: wp.array[float],
    dt: float,
):
    """Estimate tilt and apply torque through the three physical wheel joints."""
    up = estimated_up[0]
    angular_velocity = gyroscope[0]
    up = wp.normalize(up - dt * wp.cross(angular_velocity, up))

    acceleration = accelerometer[0]
    if wp.length(acceleration) > 1.0:
        gravity_up = wp.normalize(acceleration)
        up = wp.normalize(0.999 * up + 0.001 * gravity_up)
    estimated_up[0] = up

    body_position = wp.transform_get_translation(body_q[body])
    ball_position = wp.transform_get_translation(body_q[ball])
    relative_position = body_position - ball_position
    horizontal_offset = wp.length(wp.vec3(relative_position[0], relative_position[1], 0.0))
    if up[2] < 0.85 or relative_position[2] < 0.25 or horizontal_offset > 0.25:
        filtered_lean_target[0] = wp.vec3()
        for wheel in range(3):
            joint_f[drive_dofs[wheel]] = 0.0
        return

    rotation = wp.transform_get_rotation(body_q[body])
    body_velocity = wp.spatial_top(body_qd[body])
    position_error = body_position - target_position
    position_local = wp.quat_rotate_inv(rotation, wp.vec3(position_error[0], position_error[1], 0.0))
    velocity_local = wp.quat_rotate_inv(rotation, wp.vec3(body_velocity[0], body_velocity[1], 0.0))
    target_velocity = command[0]
    lean = wp.vec3(-up[0], -up[1], 0.0)
    lean_rate = wp.vec3(angular_velocity[1], -angular_velocity[0], 0.0)
    position_command = position_local
    if wp.length(wp.vec3(target_velocity[0], target_velocity[1], 0.0)) > 0.001:
        position_command = wp.vec3()
    velocity_error = velocity_local - wp.vec3(target_velocity[0], target_velocity[1], 0.0)
    lean_target = -gains[4] * position_command - gains[5] * velocity_error
    lean_target_length = wp.length(lean_target)
    if lean_target_length > 0.08:
        lean_target *= 0.08 / lean_target_length
    lean_target_alpha = dt / (0.03 + dt)
    lean_target = filtered_lean_target[0] + lean_target_alpha * (lean_target - filtered_lean_target[0])
    filtered_lean_target[0] = lean_target
    planar_command = gains[0] * (lean - lean_target) + gains[1] * lean_rate
    yaw_command = gains[3] * (angular_velocity[2] - target_velocity[2])

    for wheel in range(3):
        dof = drive_dofs[wheel]
        # Differential torque drives XY; the common mode drives yaw.
        torque = (
            (2.0 / 3.0) * wp.dot(planar_command, drive_directions[wheel]) + yaw_command / 3.0 - gains[2] * joint_qd[dof]
        )
        joint_f[dof] = wp.clamp(torque, -12.0, 12.0)


def add_mecanum_wheel(
    builder: newton.ModelBuilder,
    chassis: int,
    position: wp.vec3,
    handedness: float,
    label: str,
    *,
    roller_count: int = 12,
    hub_radius: float = 0.14,
    hub_half_width: float = 0.035,
    roller_radius: float = 0.026,
    roller_half_height: float = 0.06,
    roller_barrel_radius: float = 0.30,
    roller_clearance: float = 0.003,
) -> tuple[int, list[int], float]:
    """Add a driven 45-degree mecanum wheel with passive barrel rollers.

    Args:
        builder: Model builder receiving the wheel.
        chassis: Parent chassis body.
        position: Wheel center in the chassis frame [m].
        handedness: Roller tilt sign, either `-1.0` or `1.0`.
        label: Label prefix for generated bodies, joints, and shapes.
        roller_count: Number of passive rollers around the hub.
        hub_radius: Radius of the central wheel cylinder [m].
        hub_half_width: Half-width of the central wheel cylinder [m].
        roller_radius: Barrel end radius [m].
        roller_half_height: Barrel half-length [m].
        roller_barrel_radius: Radius of the barrel side-profile arc [m].
        roller_clearance: Radial clearance between hub and rollers [m].

    Returns:
        The driven hub joint, passive roller joints, and wheel outer radius [m].
    """
    if handedness not in (-1.0, 1.0):
        raise ValueError("handedness must be -1.0 or 1.0")

    roller_equatorial_radius = roller_radius + (roller_half_height * roller_half_height) / (
        roller_barrel_radius
        + math.sqrt(roller_barrel_radius * roller_barrel_radius - roller_half_height * roller_half_height)
    )
    roller_center_radius = hub_radius + roller_equatorial_radius + roller_clearance
    wheel_outer_radius = roller_center_radius + roller_equatorial_radius

    hub_cfg = newton.ModelBuilder.ShapeConfig(density=120.0, mu=1.0, gap=0.01)
    roller_cfg = newton.ModelBuilder.ShapeConfig(density=300.0, mu=2.0, gap=0.01)

    hub = builder.add_link(label=f"{label}_hub")
    hub_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * wp.pi)
    builder.add_shape_cylinder(
        hub,
        xform=wp.transform(q=hub_rotation),
        radius=hub_radius,
        half_height=hub_half_width,
        cfg=hub_cfg,
        color=(0.12, 0.14, 0.18),
        label=f"{label}_hub",
    )
    drive_joint = builder.add_joint_revolute(
        parent=chassis,
        child=hub,
        parent_xform=wp.transform(p=position, q=wp.quat_identity()),
        child_xform=wp.transform(),
        axis=newton.Axis.Y,
        target_vel=0.0,
        target_kd=40.0,
        damping=0.05,
        armature=0.02,
        effort_limit=80.0,
        actuator_mode=newton.JointTargetMode.VELOCITY,
        label=f"{label}_drive",
    )

    roller_joints = []
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    for roller_index in range(roller_count):
        angle = 2.0 * math.pi * roller_index / roller_count
        radial = wp.vec3(math.cos(angle), 0.0, math.sin(angle))
        tangent = wp.vec3(-math.sin(angle), 0.0, math.cos(angle))
        roller_axis = tangent * inv_sqrt_two + wp.vec3(0.0, handedness * inv_sqrt_two, 0.0)
        roller_rotation = newton.math.quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), roller_axis)

        roller = builder.add_link(label=f"{label}_roller_{roller_index}")
        builder.add_shape_cylinder(
            roller,
            radius=roller_radius,
            half_height=roller_half_height,
            barrel_radius=roller_barrel_radius,
            cfg=roller_cfg,
            color=(0.85, 0.38, 0.08),
            label=f"{label}_roller_{roller_index}",
        )
        roller_joints.append(
            builder.add_joint_revolute(
                parent=hub,
                child=roller,
                parent_xform=wp.transform(p=radial * roller_center_radius, q=roller_rotation),
                child_xform=wp.transform(),
                axis=newton.Axis.Z,
                damping=1.0e-4,
                armature=1.0e-5,
                label=f"{label}_roller_{roller_index}",
            )
        )

    return drive_joint, roller_joints, wheel_outer_radius


def add_omniwheel_90(
    builder: newton.ModelBuilder,
    parent: int,
    position: wp.vec3,
    orientation: wp.quat,
    label: str,
    *,
    roller_count: int = 6,
    wheel_radius: float = 0.12,
    hub_radius: float = 0.070,
    hub_half_width: float = 0.025,
    roller_radius: float = 0.010,
    roller_half_height: float = 0.045,
) -> tuple[int, list[int]]:
    """Add a driven conventional omniwheel with tangential barrel rollers.

    The barrel profile radius equals the wheel radius so the rollers collectively
    follow the wheel's circular outer profile.

    Args:
        builder: Model builder receiving the wheel.
        parent: Parent carrier body.
        position: Wheel center in the parent frame [m].
        orientation: Rotation from the wheel frame to the parent frame.
        label: Label prefix for generated bodies, joints, and shapes.
        roller_count: Number of passive rollers around the hub.
        wheel_radius: Outer wheel and roller barrel-profile radius [m].
        hub_radius: Radius of the central hub [m].
        hub_half_width: Half-width of the central hub [m].
        roller_radius: Roller end radius [m].
        roller_half_height: Roller half-length [m].

    Returns:
        The driven hub joint and passive roller joints.
    """
    if wheel_radius < roller_half_height:
        raise ValueError("wheel_radius must be at least roller_half_height")

    roller_equatorial_radius = (
        roller_radius + wheel_radius - math.sqrt(wheel_radius * wheel_radius - roller_half_height * roller_half_height)
    )
    roller_center_radius = wheel_radius - roller_equatorial_radius
    if hub_radius >= roller_center_radius - roller_equatorial_radius:
        raise ValueError("hub overlaps the omniwheel rollers")

    hub = builder.add_link(label=f"{label}_hub")
    hub_cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=1.0, gap=0.01)
    builder.add_shape_cylinder(
        hub,
        radius=hub_radius,
        half_height=hub_half_width,
        cfg=hub_cfg,
        color=(0.72, 0.74, 0.78),
        label=f"{label}_hub",
    )
    drive_joint = builder.add_joint_revolute(
        parent=parent,
        child=hub,
        parent_xform=wp.transform(p=position, q=orientation),
        child_xform=wp.transform(),
        axis=newton.Axis.Z,
        target_kd=0.0,
        damping=0.03,
        armature=0.002,
        effort_limit=12.0,
        velocity_limit=20.0,
        actuator_mode=newton.JointTargetMode.EFFORT,
        label=f"{label}_drive",
    )

    roller_cfg = newton.ModelBuilder.ShapeConfig(density=700.0, mu=2.0, gap=0.01)
    roller_joints = []
    for roller_index in range(roller_count):
        angle = 2.0 * math.pi * roller_index / roller_count
        radial = wp.vec3(math.cos(angle), math.sin(angle), 0.0)
        tangent = wp.vec3(-math.sin(angle), math.cos(angle), 0.0)
        roller_rotation = newton.math.quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), tangent)

        roller = builder.add_link(label=f"{label}_roller_{roller_index}")
        builder.add_shape_cylinder(
            roller,
            radius=roller_radius,
            half_height=roller_half_height,
            barrel_radius=wheel_radius,
            cfg=roller_cfg,
            color=(0.85, 0.38, 0.08),
            label=f"{label}_roller_{roller_index}",
        )
        roller_joints.append(
            builder.add_joint_revolute(
                parent=hub,
                child=roller,
                parent_xform=wp.transform(p=radial * roller_center_radius, q=roller_rotation),
                child_xform=wp.transform(),
                axis=newton.Axis.Z,
                damping=1.0e-4,
                armature=1.0e-5,
                label=f"{label}_roller_{roller_index}",
            )
        )

    return drive_joint, roller_joints


class Example:
    """Drive a mecanum robot and a self-balancing ballbot."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = newton.ModelBuilder()
        builder.default_joint_cfg.damping = 0.01

        chassis_half_length = 0.42
        chassis_half_width = 0.30
        chassis_half_height = 0.06
        wheel_x = 0.32
        wheel_y = 0.40
        wheel_mount_z = -0.04

        # The helper dimensions give a 0.207 m outer wheel radius.
        wheel_outer_radius = 0.207
        chassis_start_z = wheel_outer_radius - wheel_mount_z
        self.chassis_start = np.array([0.0, 0.0, chassis_start_z], dtype=np.float32)

        chassis = builder.add_link(
            xform=wp.transform(p=wp.vec3(*self.chassis_start), q=wp.quat_identity()),
            label="warehouse_robot",
        )
        chassis_cfg = newton.ModelBuilder.ShapeConfig(density=80.0, mu=0.8, gap=0.01)
        builder.add_shape_box(
            chassis,
            hx=chassis_half_length,
            hy=chassis_half_width,
            hz=chassis_half_height,
            cfg=chassis_cfg,
            color=(0.12, 0.42, 0.72),
            label="warehouse_robot",
        )

        joints = [builder.add_joint_free(chassis, label="warehouse_robot_free")]
        drive_joints = []
        wheel_specs = (
            ("front_left", wp.vec3(wheel_x, wheel_y, wheel_mount_z), -1.0),
            ("front_right", wp.vec3(wheel_x, -wheel_y, wheel_mount_z), 1.0),
            ("rear_left", wp.vec3(-wheel_x, wheel_y, wheel_mount_z), 1.0),
            ("rear_right", wp.vec3(-wheel_x, -wheel_y, wheel_mount_z), -1.0),
        )
        for label, position, handedness in wheel_specs:
            drive_joint, roller_joints, wheel_outer_radius = add_mecanum_wheel(
                builder,
                chassis,
                position,
                handedness,
                label,
            )
            drive_joints.append(drive_joint)
            joints.append(drive_joint)
            joints.extend(roller_joints)

        builder.add_articulation(joints, label="warehouse_robot")

        ballbot_x = -1.4
        ball_radius = 0.30
        ball = builder.add_link(
            xform=wp.transform(p=wp.vec3(ballbot_x, 0.0, ball_radius), q=wp.quat_identity()),
            label="ballbot_ball",
        )
        ball_shape = builder.add_shape_sphere(
            ball,
            radius=ball_radius,
            cfg=newton.ModelBuilder.ShapeConfig(density=35.0, mu=1.8, gap=0.01),
            color=(0.08, 0.09, 0.11),
            label="ballbot_ball",
        )
        builder.add_articulation([builder.add_joint_free(ball, label="ballbot_ball_free")], label="ballbot_ball")

        ballbot_body_height = 1.10
        initial_tilt_axis = wp.normalize(wp.vec3(1.0, 1.0, 0.0))
        initial_tilt = wp.quat_from_axis_angle(initial_tilt_axis, math.radians(1.0))
        ballbot_body = builder.add_link(
            xform=wp.transform(p=wp.vec3(ballbot_x, 0.0, ballbot_body_height), q=initial_tilt),
            label="ballbot_body",
        )
        ballbot_body_shape = builder.add_shape_cylinder(
            ballbot_body,
            radius=0.10,
            half_height=0.45,
            cfg=newton.ModelBuilder.ShapeConfig(density=350.0, mu=0.8, gap=0.01),
            color=(0.88, 0.30, 0.08),
            label="ballbot_body",
        )
        builder.add_shape_collision_filter_pair(ball_shape, ballbot_body_shape)
        ballbot_imu_site = builder.add_site(ballbot_body, label="ballbot_imu")

        ballbot_body_free_joint = builder.add_joint_free(ballbot_body, label="ballbot_body_free")
        ballbot_body_qd_start = builder.joint_qd_start[ballbot_body_free_joint]
        builder.joint_qd[ballbot_body_qd_start] = 0.05
        builder.joint_qd[ballbot_body_qd_start + 1] = -0.05
        ballbot_joints = [ballbot_body_free_joint]
        ballbot_drive_joints = []
        ballbot_drive_directions = []
        wheel_radius_90 = 0.12
        wheel_center_distance = ball_radius + wheel_radius_90
        wheel_radial_offset = wheel_center_distance / math.sqrt(2.0)
        wheel_height = ball_radius + wheel_radial_offset + 0.01
        for wheel_index in range(3):
            angle = 2.0 * math.pi * wheel_index / 3
            radial = wp.vec3(math.cos(angle), math.sin(angle), 0.0)
            tangent = wp.vec3(-math.sin(angle), math.cos(angle), 0.0)
            wheel_position = radial * wheel_radial_offset + wp.vec3(0.0, 0.0, wheel_height - ballbot_body_height)
            drive_axis = (-radial + wp.vec3(0.0, 0.0, 1.0)) / math.sqrt(2.0)
            wheel_orientation = newton.math.quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), drive_axis)
            drive_joint, roller_joints = add_omniwheel_90(
                builder,
                ballbot_body,
                wheel_position,
                wheel_orientation,
                f"ballbot_wheel_{wheel_index}",
                wheel_radius=wheel_radius_90,
            )
            ballbot_drive_joints.append(drive_joint)
            ballbot_drive_directions.append(tangent)
            ballbot_joints.append(drive_joint)
            ballbot_joints.extend(roller_joints)

        builder.add_articulation(ballbot_joints, label="ballbot")
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.5, gap=0.01))

        self.model = builder.finalize()
        self.chassis = chassis
        self.ballbot_ball = ball
        self.ballbot_body = ballbot_body
        self.ballbot_target_position = wp.vec3(ballbot_x, 0.0, ballbot_body_height)
        self.ballbot_command = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.wheel_outer_radius = wheel_outer_radius
        joint_qd_start = self.model.joint_qd_start.numpy()
        self.drive_dof_indices = np.array([joint_qd_start[joint] for joint in drive_joints], dtype=np.int32)
        self.drive_dofs = wp.array(self.drive_dof_indices, dtype=int, device=self.model.device)
        self.ballbot_drive_dof_indices = np.array(
            [joint_qd_start[joint] for joint in ballbot_drive_joints], dtype=np.int32
        )
        self.ballbot_drive_dofs = wp.array(self.ballbot_drive_dof_indices, dtype=int, device=self.model.device)
        self.ballbot_drive_directions = wp.array(ballbot_drive_directions, dtype=wp.vec3, device=self.model.device)
        # Tilt P/D, wheel damping, yaw-rate P, position P, and velocity P.
        self.ballbot_controller_gains = wp.array(
            [-60.0, -45.0, 0.75, -15.0, 0.10, 1.00],
            dtype=float,
            device=self.model.device,
        )
        self.ballbot_filtered_lean_target = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        self.ballbot_estimated_up = wp.array(
            [wp.quat_rotate_inv(initial_tilt, wp.vec3(0.0, 0.0, 1.0))],
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.ballbot_imu = newton.sensors.SensorIMU(self.model, [ballbot_imu_site])

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            disable_sensors=True,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=15,
            ls_iterations=50,
            njmax=2048,
            nconmax=1024,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.ballbot_imu.update(self.state_0)
        self.collision_pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=1024)
        self.contacts = self.collision_pipeline.contacts()

        self.max_forward_travel = 0.0
        self.max_lateral_travel = 0.0
        self.max_yaw = 0.0
        self.max_drive_speed = 0.0
        self.min_ballbot_up = 1.0
        self.min_ballbot_height = math.inf
        self.max_ballbot_offset = 0.0
        self.max_ballbot_drive_speed = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.4, -3.2, 2.0), pitch=-24.0, yaw=128.0)
        self.capture()

    def capture(self):
        """Capture one simulation frame."""
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        """Advance one frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                set_ballbot_torques,
                dim=1,
                inputs=[
                    self.ballbot_imu.accelerometer,
                    self.ballbot_imu.gyroscope,
                    self.ballbot_estimated_up,
                    self.ballbot_filtered_lean_target,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_0.joint_qd,
                    self.control.joint_f,
                    self.ballbot_body,
                    self.ballbot_ball,
                    self.ballbot_target_position,
                    self.ballbot_command,
                    self.ballbot_drive_dofs,
                    self.ballbot_drive_directions,
                    self.ballbot_controller_gains,
                    self.sim_dt,
                ],
            )
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.ballbot_imu.update(self.state_0)

    def step(self):
        """Set the drive command and advance one frame."""
        if self.sim_time < 1.5:
            command = wp.vec3(-1.40 * min(self.sim_time / 0.5, 1.0), 0.0, 0.0)
        elif self.sim_time < 3.0:
            command = wp.vec3(0.0, 1.20, 0.0)
        elif self.sim_time < 4.5:
            command = wp.vec3(0.0, 0.0, 3.00)
        else:
            pattern_time = self.sim_time - 4.5
            command = wp.vec3(
                math.cos(0.8 * pattern_time),
                math.sin(1.1 * pattern_time),
                2.00 * math.sin(0.6 * pattern_time),
            )

        if self.sim_time < 2.0:
            ballbot_command = wp.vec3()
        elif self.sim_time < 6.0:
            ballbot_command = wp.vec3(0.20, 0.0, 0.0)
        elif self.sim_time < 10.0:
            ballbot_command = wp.vec3(0.0, 0.20, 0.0)
        elif self.sim_time < 14.0:
            ballbot_command = wp.vec3(0.0, 0.0, 0.40)
        else:
            ballbot_command = wp.vec3()
        self.ballbot_command.fill_(ballbot_command)

        wp.launch(
            set_mecanum_targets,
            dim=4,
            inputs=[
                self.control.joint_target_qd,
                self.drive_dofs,
                command,
                self.wheel_outer_radius,
                0.72,
            ],
        )
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        """Track achieved chassis motion and wheel speeds."""
        body_q = self.state_0.body_q.numpy()[self.chassis]
        assert np.all(np.isfinite(body_q)), f"non-finite chassis pose at t={self.sim_time:.3f} s"
        self.max_forward_travel = max(self.max_forward_travel, abs(float(body_q[0] - self.chassis_start[0])))
        self.max_lateral_travel = max(self.max_lateral_travel, abs(float(body_q[1] - self.chassis_start[1])))
        qx, qy, qz, qw = body_q[3:7]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        self.max_yaw = max(self.max_yaw, abs(yaw))
        joint_qd = self.state_0.joint_qd.numpy()
        self.max_drive_speed = max(self.max_drive_speed, float(np.max(np.abs(joint_qd[self.drive_dof_indices]))))
        ballbot_q = self.state_0.body_q.numpy()[self.ballbot_body]
        ball_q = self.state_0.body_q.numpy()[self.ballbot_ball]
        ballbot_rotation = wp.quat(*ballbot_q[3:7])
        ballbot_up = wp.quat_rotate(ballbot_rotation, wp.vec3(0.0, 0.0, 1.0))
        self.min_ballbot_up = min(self.min_ballbot_up, float(ballbot_up[2]))
        self.min_ballbot_height = min(self.min_ballbot_height, float(ballbot_q[2]))
        self.max_ballbot_offset = max(
            self.max_ballbot_offset,
            float(np.linalg.norm(ballbot_q[:2] - ball_q[:2])),
        )
        self.max_ballbot_drive_speed = max(
            self.max_ballbot_drive_speed,
            float(np.max(np.abs(joint_qd[self.ballbot_drive_dof_indices]))),
        )

    def test_final(self):
        """Verify the robot completes all three commanded motions."""
        assert self.sim_time >= 4.5, f"run too short to exercise all drive phases: {self.sim_time:.2f} s"
        body_q = self.state_0.body_q.numpy()[self.chassis]
        assert np.all(np.isfinite(body_q)), "non-finite chassis pose"
        assert body_q[2] > 0.05, f"chassis fell through the ground: z={body_q[2]:.3f}"
        assert self.max_forward_travel > 0.08, (
            f"insufficient forward travel: {self.max_forward_travel:.3f} m "
            f"(lateral={self.max_lateral_travel:.3f} m, yaw={self.max_yaw:.3f} rad, "
            f"drive={self.max_drive_speed:.3f} rad/s)"
        )
        assert self.max_lateral_travel > 0.06, f"insufficient lateral travel: {self.max_lateral_travel:.3f} m"
        assert self.max_yaw > 0.10, f"insufficient rotation: {self.max_yaw:.3f} rad"
        assert self.max_drive_speed > 0.5, f"wheel drives did not engage: {self.max_drive_speed:.3f} rad/s"
        assert self.min_ballbot_up > 0.85, f"ballbot lost balance: minimum up={self.min_ballbot_up:.3f}"
        assert self.min_ballbot_height > 0.60, f"ballbot pod lost support: minimum z={self.min_ballbot_height:.3f} m"
        assert self.max_ballbot_offset < 0.20, f"ballbot lost the ball: maximum offset={self.max_ballbot_offset:.3f} m"
        assert self.max_ballbot_drive_speed > 0.1, "ballbot balance drives did not engage"

    def render(self):
        """Render the robot and contacts."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
