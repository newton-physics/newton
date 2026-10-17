# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Animate a garden of twenty-stage Stewart platform flowers using Kamino DVI.

Each plate is the moving platform of one stage and the base of the next.
Universal joints attach both ends of each hydraulic actuator. A passive axial
swivel in each cylinder prevents the universal joints from binding in torsion.
Only piston extensions are driven; all plate poses emerge from the dynamics.
The choreography blends waves, corkscrews, bows, and traveling curls over 20 s.
A lightweight 1 m Hoberman flower breathes on a pinned yoke at the tip.
Separate worlds share the choreography with small variations like flowers in wind.
Thirty flowers form a compact hexagonal patch with gently scattered base positions.
Use --flower-resolution to vary scissor cells per 36-degree ring arc; the default
of two reproduces the original twenty-cell rings procedurally.

Command: python -m newton.examples kamino_stewart_platform
"""

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.kamino._hoberman_sphere import generate_sphere
from newton.solvers import SolverKamino

# Saturated colors distinguish the interlocking flower rings.
FLOWER_COLORS = (
    (0.86, 0.06, 0.02),
    (0.92, 0.22, 0.01),
    (0.94, 0.42, 0.01),
    (0.88, 0.62, 0.01),
    (0.78, 0.72, 0.01),
    (0.46, 0.72, 0.01),
    (0.03, 0.60, 0.12),
    (0.00, 0.58, 0.38),
    (0.00, 0.62, 0.68),
    (0.00, 0.53, 0.88),
    (0.06, 0.30, 0.88),
    (0.22, 0.12, 0.78),
    (0.42, 0.10, 0.76),
    (0.58, 0.08, 0.68),
)


def _transform(values, scale=1.0):
    return wp.transform(wp.vec3(*values[:3]) * scale, wp.quat(*values[3:]))


def _garden_offsets(world_count):
    """Place flowers on a 2 m hexagonal lattice with up to 16 cm of fixed jitter."""
    radius = 0
    while 1 + 3 * radius * (radius + 1) < world_count:
        radius += 1
    cells = [(q, r) for q in range(-radius, radius + 1) for r in range(-radius, radius + 1) if abs(q + r) <= radius]
    # Fill from the center outward to avoid a rectangular boundary.
    cells.sort(key=lambda cell: (cell[0] ** 2 + cell[0] * cell[1] + cell[1] ** 2, math.atan2(cell[1], cell[0])))
    offsets = np.zeros((world_count, 3), dtype=np.float32)
    offsets[:, :2] = [(2.0 * (q + 0.5 * r), math.sqrt(3.0) * r) for q, r in cells[:world_count]]
    rng = np.random.default_rng(2026)
    angles = rng.uniform(0.0, 2.0 * math.pi, world_count)
    lengths = 0.16 * np.sqrt(rng.uniform(0.0, 1.0, world_count))
    offsets[:, 0] += lengths * np.cos(angles)
    offsets[:, 1] += lengths * np.sin(angles)
    offsets -= offsets.mean(axis=0)
    return offsets


def add_hoberman_flower(builder, plate, joints_tree, sphere):
    """Mount a scaled sphere by its outer three-hinge crossing on a driven yoke.

    A bearing supports one strut at a shared outer crossing. It turns by half
    the opening hinge angle, keeping the supported arms symmetric about the
    stem while the ring joints transmit the opening to the sphere.
    """
    scale = 1.0 / (2.0 * 2.236)
    mass_scale = 0.18 / sum(body[4] for body in sphere.bodies)
    rotation = wp.quat_between_vectors(wp.vec3(-1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, -1.0))
    pole = wp.vec3(*sphere.pole) * scale
    stem_tip = wp.vec3(0.0, 0.0, 0.055)
    placement = wp.transform_multiply(
        builder.body_q[plate], wp.transform(stem_tip - wp.quat_rotate(rotation, pole), rotation)
    )
    offset = builder.body_count
    for label, pose, com, inertia, mass in sphere.bodies:
        builder.add_link(
            xform=wp.transform_multiply(placement, _transform(pose, scale)),
            com=wp.vec3(*com) * scale,
            inertia=wp.mat33(np.asarray(inertia) * mass_scale * scale**2),
            mass=mass * mass_scale,
            lock_inertia=True,
            label=label,
        )

    opening_joint = None
    mount_child = offset + sphere.mount_body
    mount_frame = _transform(sphere.mount_frame, scale)
    closures = []
    for index, (kind, label, parent, child, xp, xc, axes, coordinates) in enumerate(sphere.joints):
        if kind == "FREE":
            joint = builder.add_joint_free(offset + child, label="flower_root")
            start = builder.joint_q_start[joint]
            builder.joint_q[start : start + 7] = list(builder.body_q[offset + child])
        else:
            driven = label == sphere.opening_label
            kwargs = {
                "joint_type": newton.JointType[kind],
                "parent": offset + parent,
                "child": offset + child,
                "parent_xform": _transform(xp, scale),
                "child_xform": _transform(xc, scale),
                "angular_axes": [
                    newton.ModelBuilder.JointDofConfig(
                        axis=wp.vec3(*axis),
                        target_ke=300.0 if driven else 0.0,
                        target_kd=3.0 if driven else 0.0,
                        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY
                        if driven
                        else newton.JointTargetMode.NONE,
                    )
                    for axis in axes
                ],
                "label": label,
            }
            if index >= sphere.tree_joint_count:
                closures.append(kwargs)
                continue
            joint = builder.add_joint(**kwargs)
            start = builder.joint_q_start[joint]
            builder.joint_q[start : start + len(coordinates)] = coordinates
            if driven:
                opening_joint = joint
        joints_tree.append(joint)

    # Both joint frames coincide at the crossing. Its pin axis is tangential,
    # so the supported strut can rotate as the sphere opens around the pole.
    child = mount_child
    child_frame = mount_frame
    parent_frame = wp.transform_multiply(
        wp.transform_inverse(builder.body_q[plate]),
        wp.transform_multiply(builder.body_q[child], child_frame),
    )
    mount = builder.add_joint_revolute(
        plate,
        child,
        parent_xform=parent_frame,
        child_xform=child_frame,
        axis=newton.Axis.X,
        target_ke=600.0,
        target_kd=6.0,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        label="flower_mount",
    )
    for kwargs in closures:
        builder.add_joint(**kwargs)

    visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
    for body, pose, extents, _color in sphere.tiles:
        color = FLOWER_COLORS[body % len(FLOWER_COLORS)]
        builder.add_shape_box(
            offset + body,
            xform=_transform(pose, scale),
            hx=extents[0] * scale,
            hy=extents[1] * scale,
            hz=extents[2] * scale,
            cfg=visual,
            color=color,
        )

    gold = (0.92, 0.65, 0.25)
    builder.add_shape_cylinder(
        plate,
        xform=wp.transform((0.0, 0.0, 0.025), wp.quat_identity()),
        radius=0.009,
        half_height=0.022,
        cfg=visual,
        color=gold,
    )
    pin_axis = wp.transform_vector(parent_frame, wp.vec3(1.0, 0.0, 0.0))
    pin_rotation = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), pin_axis)
    for sign in (-1.0, 1.0):
        bearing = stem_tip + sign * 0.012 * pin_axis
        builder.add_shape_box(
            plate,
            xform=wp.transform(bearing - wp.vec3(0.0, 0.0, 0.006), pin_rotation),
            hx=0.007,
            hy=0.013,
            hz=0.003,
            cfg=visual,
            color=(0.20, 0.25, 0.30),
        )
    builder.add_shape_cylinder(
        plate,
        xform=wp.transform(stem_tip, pin_rotation),
        radius=0.003,
        half_height=0.018,
        cfg=visual,
        color=gold,
    )
    return opening_joint, mount, offset


@wp.func
def wind_time(time: float, lag: float) -> float:
    # Fixed offsets keep the common rhythm; fading them in preserves the rest pose.
    return time + lag * (1.0 - wp.exp(-0.35 * time * time))


@wp.func
def flower_opening(time: float) -> float:
    # Start at the authored pose and ease into a slow, six-second breath.
    phase = 1.04719755 * time
    return (1.0 - wp.exp(-0.35 * time * time)) * (0.2 * (1.0 - wp.cos(phase)) + 0.9 * wp.sin(phase))


@wp.kernel
def update_flower(
    time: wp.array[float],
    dt: float,
    opening_slope: float,
    variations: wp.array[wp.vec3],
    coordinates: wp.array[int],
    dofs: wp.array[int],
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    actuator = wp.tid()
    motor = actuator % 2
    variation = variations[actuator / 2]
    ratio = 1.0
    if motor == 1:
        ratio = 0.5
    angle = variation[1] * flower_opening(wind_time(time[0], 1.5 * variation[0]))
    angle_next = variation[1] * flower_opening(wind_time(time[0] + dt, 1.5 * variation[0]))
    # Finer rings reach full compaction at smaller hinge angles. Invert their
    # aperture law to retain the original radial breathing motion safely.
    reference_slope = (2.2 - 1.35896) / (2.2 + 1.35896) / wp.tan(wp.pi / 20.0)
    if opening_slope > 1.001 * reference_slope:
        gain = wp.cos(0.5 * angle) + reference_slope * wp.sin(0.5 * angle)
        gain_next = wp.cos(0.5 * angle_next) + reference_slope * wp.sin(0.5 * angle_next)
        magnitude = wp.sqrt(1.0 + opening_slope * opening_slope)
        rest = wp.atan2(1.0, opening_slope)
        angle = 2.0 * (wp.asin(wp.clamp(gain / magnitude, -1.0, 1.0)) - rest)
        angle_next = 2.0 * (wp.asin(wp.clamp(gain_next / magnitude, -1.0, 1.0)) - rest)
    target_q[coordinates[actuator]] = ratio * angle
    target_qd[dofs[actuator]] = ratio * (angle_next - angle) / dt


@wp.func
def stage_pose(stage: int, stage_count: int, time: float) -> wp.transform:
    # Blend four phrases continuously: serpentine waves, corkscrews, a broad
    # bow, and a traveling curl. Every phrase starts at mid-stroke when t=0.
    fade = 1.0 - wp.exp(-0.35 * time * time)
    height = float(stage) / float(wp.max(stage_count - 1, 1))
    cycle = 0.31415927 * time
    weights = wp.vec4(
        wp.pow(wp.max(wp.cos(cycle), 0.0), 4.0),
        wp.pow(wp.max(wp.sin(cycle), 0.0), 4.0),
        wp.pow(wp.max(-wp.cos(cycle), 0.0), 4.0),
        wp.pow(wp.max(-wp.sin(cycle), 0.0), 4.0),
    )
    weights /= weights[0] + weights[1] + weights[2] + weights[3]
    phase = 0.85 * time - 0.35 * float(stage)
    spiral = 1.3 * time - 0.65 * float(stage)
    bow = wp.sin(wp.pi * height)
    pulse_center = 0.5 + 0.45 * wp.sin(0.65 * time)
    pulse = wp.exp(-wp.pow((height - pulse_center) / 0.18, 2.0))
    roll = wp.dot(
        weights,
        wp.vec4(
            0.12 * wp.sin(phase),
            0.11 * wp.sin(spiral),
            0.065 * bow * wp.sin(0.7 * time),
            0.16 * pulse * wp.sin(0.8 * time),
        ),
    )
    pitch = wp.dot(
        weights,
        wp.vec4(
            0.12 * wp.cos(phase * 0.83),
            0.11 * wp.cos(spiral),
            0.065 * bow * wp.cos(0.7 * time),
            0.16 * pulse * wp.cos(0.8 * time),
        ),
    )
    twist = wp.dot(
        weights,
        wp.vec4(
            0.045 * wp.sin(0.6 * time + 0.31 * float(stage)),
            0.13 * wp.sin(0.9 * time - 0.3 * float(stage)),
            0.10 * bow * wp.sin(0.8 * time),
            0.12 * pulse * wp.sin(1.5 * time),
        ),
    )
    rotation = wp.quat_rpy(
        fade * roll,
        fade * pitch,
        fade * twist,
    )
    position = wp.vec3(
        0.002 * fade * wp.sin(phase * 0.7),
        0.002 * fade * wp.cos(phase),
        0.1
        + fade
        * (
            0.007 * (1.0 - weights[2]) * wp.sin(1.1 * time - 0.65 * float(stage))
            + 0.011 * weights[2] * wp.sin(1.2 * time)
        ),
    )
    return wp.transform(position, rotation)


@wp.func
def wind_pose(stage: int, stage_count: int, time: float, variation: wp.vec3) -> wp.transform:
    common = stage_pose(stage, stage_count, wind_time(time, variation[0]))
    gain = variation[1]
    fade = 1.0 - wp.exp(-0.35 * time * time)
    flutter = 0.004 * fade * wp.sin(variation[2])
    rotation = wp.quat_slerp(wp.quat_identity(), wp.transform_get_rotation(common), gain)
    rotation *= wp.quat_rpy(
        flutter * wp.sin(1.7 * time + variation[2]),
        flutter * wp.sin(1.3 * time + variation[2]),
        0.0,
    )
    position = wp.transform_get_translation(common)
    position = wp.vec3(gain * position[0], gain * position[1], 0.1 + gain * (position[2] - 0.1))
    return wp.transform(position, rotation)


@wp.kernel
def update_actuators(
    time: wp.array[float],
    dt: float,
    stage_count: int,
    variations: wp.array[wp.vec3],
    anchors_base: wp.array[wp.vec3],
    anchors_top: wp.array[wp.vec3],
    lengths_rest: wp.array[float],
    actuator_dofs: wp.array[int],
    actuator_coordinates: wp.array[int],
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    actuator = wp.tid()
    stage = (actuator / 6) % stage_count
    variation = variations[actuator / (6 * stage_count)]
    pose = wind_pose(stage, stage_count, time[0], variation)
    pose_next = wind_pose(stage, stage_count, time[0] + dt, variation)
    length = wp.length(wp.transform_point(pose, anchors_top[actuator]) - anchors_base[actuator])
    length_next = wp.length(wp.transform_point(pose_next, anchors_top[actuator]) - anchors_base[actuator])
    dof = actuator_dofs[actuator]
    target_q[actuator_coordinates[actuator]] = length - lengths_rest[actuator]
    target_qd[dof] = (length_next - length) / dt


@wp.kernel
def advance_time(time: wp.array[float], dt: float):
    time[0] += dt


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        self.viewer = viewer
        self.stage_count = args.stage_count if args else 20
        self.world_count = getattr(args, "world_count", 30) if args else 30
        if self.stage_count < 1:
            raise ValueError("--stage-count must be positive")
        if self.world_count < 1:
            raise ValueError("--world-count must be positive")
        self.frame_dt = 1.0 / 60.0
        # Two implicit steps retain loop closure and actuator tracking while
        # avoiding repeated factorization of the flower's dense joint graph.
        self.sim_substeps = getattr(args, "substeps", 2) if args else 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.time = wp.zeros(1, dtype=float)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        # The mechanism is attached to ground. Contact between the small, closely
        # spaced fittings is excluded, as in the Kamino test-mechanism example.
        builder.default_shape_cfg.has_shape_collision = False
        builder.default_shape_cfg.density = 1600.0
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform((0.0, 0.0, 0.005), wp.quat_identity()),
            radius=0.075,
            half_height=0.005,
            color=(0.22, 0.26, 0.30),
        )

        def passive(axis):
            return newton.ModelBuilder.JointDofConfig(axis=axis, actuator_mode=newton.JointTargetMode.NONE)

        joints_tree = []
        closures = []
        self.plates = []
        anchors_base, anchors_top, lengths_rest, actuator_joints = [], [], [], []

        def add_plate(level):
            body = builder.add_link(
                xform=wp.transform((0.0, 0.0, 0.012 + 0.1 * level), wp.quat_identity()),
                label=f"plate_{level}",
            )
            color = FLOWER_COLORS[level % len(FLOWER_COLORS)]
            builder.add_shape_cylinder(body, radius=0.05, half_height=0.002, color=color)
            builder.add_shape_cylinder(body, radius=0.047, half_height=0.0024, color=(0.12, 0.16, 0.20))
            self.plates.append(body)
            return body

        base = add_plate(0)
        joints_tree.append(builder.add_joint_fixed(-1, base, parent_xform=builder.body_q[base], label="ground_anchor"))
        for stage in range(self.stage_count):
            top = add_plate(stage + 1)
            for leg in range(6):
                sector = 2.0 * math.pi * (leg // 2) / 3.0
                sign = -1.0 if leg % 2 == 0 else 1.0
                angle_base = sector + sign * math.radians(12.0)
                angle_top = sector + sign * math.radians(48.0)
                a = wp.vec3(0.0425 * math.cos(angle_base), 0.0425 * math.sin(angle_base), 0.004)
                b = wp.vec3(0.0425 * math.cos(angle_top), 0.0425 * math.sin(angle_top), -0.004)
                delta = b + wp.vec3(0.0, 0.0, 0.1) - a
                length = wp.length(delta)
                rotation = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), delta / length)
                origin_base = wp.transform_point(builder.body_q[base], a)
                origin_top = wp.transform_point(builder.body_q[top], b)
                barrel = builder.add_link(xform=wp.transform(origin_base, rotation), label=f"barrel_{stage}_{leg}")
                rod = builder.add_link(xform=wp.transform(origin_top, rotation), label=f"rod_{stage}_{leg}")
                builder.add_shape_cylinder(
                    barrel,
                    xform=wp.transform((0.0, 0.0, 0.039), wp.quat_identity()),
                    radius=0.0032,
                    half_height=0.035,
                    color=(0.16, 0.23, 0.29),
                )
                for z in (0.008, 0.070):
                    builder.add_shape_cylinder(
                        barrel,
                        xform=wp.transform((0.0, 0.0, z), wp.quat_identity()),
                        radius=0.0038,
                        half_height=0.002,
                        color=(0.9, 0.52, 0.16),
                    )
                builder.add_shape_cylinder(
                    rod,
                    xform=wp.transform((0.0, 0.0, -0.04), wp.quat_identity()),
                    radius=0.0017,
                    half_height=0.036,
                    color=(0.75, 0.82, 0.88),
                )
                # Cross pins make the two orthogonal universal-joint axes visible.
                for body in (barrel, rod):
                    for axis in (wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)):
                        builder.add_shape_cylinder(
                            body,
                            xform=wp.transform(wp.vec3(), wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis)),
                            radius=0.0018,
                            half_height=0.0042,
                            color=(0.9, 0.65, 0.24),
                        )
                lower = builder.add_joint_d6(
                    base,
                    barrel,
                    parent_xform=wp.transform(a, rotation),
                    angular_axes=[passive(newton.Axis.X), passive(newton.Axis.Y)],
                    label=f"universal_lower_{stage}_{leg}",
                )
                # A cylindrical joint combines driven sliding with passive rod spin.
                # Six U-P-U legs without that spin would overconstrain a Stewart platform.
                actuator = builder.add_joint_d6(
                    barrel,
                    rod,
                    parent_xform=wp.transform((0.0, 0.0, length), wp.quat_identity()),
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(
                            axis=newton.Axis.Z,
                            limit_lower=-0.025,
                            limit_upper=0.025,
                            target_ke=300000.0,
                            target_kd=150.0,
                            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                        )
                    ],
                    angular_axes=[passive(newton.Axis.Z)],
                    label=f"piston_{stage}_{leg}",
                )
                joints_tree.extend((lower, actuator))
                upper_args = {
                    "parent": rod,
                    "child": top,
                    "child_xform": wp.transform(b, rotation),
                    "angular_axes": [passive(newton.Axis.X), passive(newton.Axis.Y)],
                    "label": f"universal_upper_{stage}_{leg}",
                }
                if leg == 0:
                    joints_tree.append(builder.add_joint_d6(**upper_args))
                else:
                    closures.append(upper_args)
                anchors_base.append(a)
                anchors_top.append(b)
                lengths_rest.append(length)
                actuator_joints.append(actuator)
            base = top

        # The remaining upper universal joints close the six parallel leg loops.
        builder.add_articulation(joints_tree, label="stewart_tower")
        for upper_args in closures:
            builder.add_joint_d6(**upper_args)
        flower_tree = []
        self.sphere = generate_sphere(getattr(args, "flower_resolution", 2) if args else 2)
        opening, mount, flower_body_start = add_hoberman_flower(builder, top, flower_tree, self.sphere)
        builder.add_articulation(flower_tree, label="hoberman_flower")

        # Keep simulation coordinates near the origin; only the viewer spaces
        # the worlds apart. Each world gets its own constraints and controls.
        prototype = builder
        body_offsets = np.arange(self.world_count) * prototype.body_count
        joint_offsets = np.arange(self.world_count) * prototype.joint_count
        self.plates = np.asarray(self.plates)[None, :] + body_offsets[:, None]
        self.flower_body_starts = flower_body_start + body_offsets
        self.flower_joint_indices = (np.array((opening, mount))[None, :] + joint_offsets[:, None]).reshape(-1)
        actuator_joints = (np.asarray(actuator_joints)[None, :] + joint_offsets[:, None]).reshape(-1)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(prototype)
        builder.add_ground_plane(color=(0.18, 0.21, 0.25))
        self.model = builder.finalize(skip_validation_joints=True)
        q_starts = self.model.joint_q_start.numpy()
        dof_starts = self.model.joint_qd_start.numpy()
        target_starts = self.model.joint_target_q_start.numpy()
        self.flower_q_indices = q_starts[self.flower_joint_indices]
        self.flower_coords_host = target_starts[self.flower_joint_indices]
        self.flower_coordinates = wp.array(self.flower_coords_host, dtype=int)
        self.flower_dofs = wp.array(dof_starts[self.flower_joint_indices], dtype=int)
        self.anchors_base = wp.array(anchors_base * self.world_count, dtype=wp.vec3)
        self.anchors_top = wp.array(anchors_top * self.world_count, dtype=wp.vec3)
        self.lengths_rest = wp.array(lengths_rest * self.world_count, dtype=float)
        self.actuator_q_indices = q_starts[actuator_joints]
        self.actuator_coords_host = target_starts[actuator_joints]
        self.actuator_dofs = wp.array(dof_starts[actuator_joints], dtype=int)
        self.actuator_coordinates = wp.array(self.actuator_coords_host, dtype=int)
        phases = np.arange(self.world_count) * 2.39996323
        self.variations_host = np.column_stack((0.3 * np.sin(phases), 1.0 + 0.04 * np.sin(phases), phases))
        self.variations = wp.array(self.variations_host, dtype=wp.vec3)

        config = SolverKamino.Config.from_model(
            self.model, dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True
        )
        config.use_fk_solver = False
        config.use_collision_detector = False
        config.constraints.alpha = 0.1
        config.constraints.beta = 0.011
        config.constraints.gamma = 0.015
        config.dynamics.preconditioning = False
        config.dynamics.linear_solver_type = "CR"
        config.dynamics.linear_solver_kwargs = {"maxiter": 9}
        config.dvi.bilateral_solver_type = "LLTBRCM"
        config.dvi.bilateral_solver_kwargs = {"parallel_factorization": True}
        config.dvi.tolerance = 1e-5
        config.dvi.regularization = 1e-8
        config.dvi.max_alternating_iterations = 4
        config.dvi.inequality_sweeps_per_iteration = 3
        config.dvi.bilateral_solve_interval = 1
        self.solver = SolverKamino(self.model, config=config)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.viewer.set_model(self.model)
        self.world_offsets = _garden_offsets(self.world_count)
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        # Assign in place so rendering and picking retain the same offset array.
        self.viewer.world_offsets.assign(self.world_offsets)
        if hasattr(self.viewer, "set_camera"):
            height = 0.1 * self.stage_count + 1.1
            span = float(np.ptp(self.world_offsets[:, :2], axis=0).max())
            distance = height + 0.45 * span
            elevation = 0.55 * height + 0.75 * span
            pitch = -math.degrees(math.atan2(elevation - 0.5 * height, math.hypot(0.8, 1.38) * distance))
            self.viewer.set_camera(wp.vec3(0.8 * distance, -1.38 * distance, elevation), pitch=pitch, yaw=120.0)
        self.graph = None
        if self.model.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                update_actuators,
                dim=6 * self.stage_count * self.world_count,
                inputs=[
                    self.time,
                    self.sim_dt,
                    self.stage_count,
                    self.variations,
                    self.anchors_base,
                    self.anchors_top,
                    self.lengths_rest,
                    self.actuator_dofs,
                    self.actuator_coordinates,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
            )
            wp.launch(
                update_flower,
                dim=2 * self.world_count,
                inputs=[
                    self.time,
                    self.sim_dt,
                    self.sphere.opening_slope,
                    self.variations,
                    self.flower_coordinates,
                    self.flower_dofs,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, inputs=[self.time, self.sim_dt])

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Check tower tracking, flower breathing, and the mounted mechanism's loop closure."""
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "Non-finite body poses"
        assert self.model.world_count == self.world_count, "Flower worlds were not replicated"
        for plates in self.plates:
            np.testing.assert_allclose(poses[plates[0], :3], (0.0, 0.0, 0.012), atol=2e-4)
            np.testing.assert_allclose(poses[plates[0], 3:], (0.0, 0.0, 0.0, 1.0), atol=2e-4)
        parent = self.model.joint_parent.numpy()
        child = self.model.joint_child.numpy()
        xp = self.model.joint_X_p.numpy()
        xc = self.model.joint_X_c.numpy()
        for joint, label in enumerate(self.model.joint_label):
            if not (label.startswith("universal_") or label.startswith("/hoberman_sphere/") or label == "flower_mount"):
                continue
            a = wp.transform_point(wp.transform(*poses[parent[joint]]), wp.vec3(*xp[joint, :3]))
            b = wp.transform_point(wp.transform(*poses[child[joint]]), wp.vec3(*xc[joint, :3]))
            assert wp.length(a - b) < 0.001, f"Joint opened: {label}"
        joint_q = self.state_0.joint_q.numpy()
        target_q = self.control.joint_target_q.numpy()
        q = joint_q[self.actuator_q_indices]
        target = target_q[self.actuator_coords_host]
        assert np.max(np.abs(q)) < 0.026, "Piston exceeded its stroke"
        np.testing.assert_allclose(q, target, atol=0.002)
        flower_q = joint_q[self.flower_q_indices].reshape(self.world_count, 2)
        flower_target = target_q[self.flower_coords_host].reshape(self.world_count, 2)
        np.testing.assert_allclose(flower_q, flower_target, atol=0.01)
        assert np.max(np.abs(flower_q[:, 1] - 0.5 * flower_q[:, 0])) < 0.005, (
            "Flower mount lost opening synchronization"
        )
        if self.world_count > 1 and self.sim_time > 1.0:
            spread = np.max(np.ptp(q.reshape(self.world_count, -1), axis=0))
            assert 1e-5 < spread < 0.012, "Worlds lost their subtle motion variations"
        # Check each stage: the traveling curl deliberately lowers the tip, and
        # small actuator compliance accumulates along the twenty-stage tower.
        for world, plates in enumerate(self.plates):
            start = self.flower_body_starts[world]
            points = poses[start : start + len(self.sphere.bodies), :3]
            radius = np.mean(np.linalg.norm(points - points.mean(axis=0), axis=1))
            radius_rest = self.sphere.radius_rest / (2.0 * 2.236)
            assert 0.25 * radius_rest < radius < 1.8 * radius_rest, "Flower lost its spherical shape"
            opening = flower_q[world, 0]
            if abs(opening) > 0.04:
                assert (radius - radius_rest) * np.sign(opening) > 0.003, "Flower did not expand or contract"
            for stage in range(self.stage_count):
                relative = wp.transform_multiply(
                    wp.transform_inverse(wp.transform(*poses[plates[stage]])),
                    wp.transform(*poses[plates[stage + 1]]),
                )
                desired = wind_pose(stage, self.stage_count, self.sim_time, wp.vec3(*self.variations_host[world]))
                np.testing.assert_allclose(np.array(relative)[:3], np.array(desired)[:3], atol=0.003)
                rotation_error = wp.quat_inverse(wp.transform_get_rotation(desired)) * wp.transform_get_rotation(
                    relative
                )
                assert wp.length(wp.vec3(rotation_error[0], rotation_error[1], rotation_error[2])) < 0.035, (
                    f"World {world} platform {stage} did not follow the commanded tilt"
                )

    @staticmethod
    def create_parser():
        def positive_int(value: str) -> int:
            try:
                result = int(value)
            except ValueError:
                raise argparse.ArgumentTypeError("must be a positive integer") from None
            if result < 1:
                raise argparse.ArgumentTypeError("must be a positive integer")
            return result

        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=30)
        parser.add_argument(
            "--flower-resolution",
            type=positive_int,
            default=2,
            help="Scissor cells per 36-degree ring arc (2 reproduces the original sphere).",
        )
        parser.add_argument("--stage-count", type=int, default=20, help="Number of stacked Stewart platforms.")
        parser.add_argument("--substeps", type=int, choices=(2, 4, 6, 12), default=2, help="Physics steps per frame.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
