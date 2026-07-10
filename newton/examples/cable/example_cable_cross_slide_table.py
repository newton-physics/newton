# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Cross-Slide Table
#
# Demonstrates a cable-driven cross-slide table inspired by the Simscape
# Multibody cable-driven XY table example https://www.mathworks.com/help/sm/ug/cable-driven-xy-table-with-cross-base.html.
# The mechanism is laid out on the ground plane: the blue base is fixed, the green carriage moves horizontally,
# and the beige carriage moves vertically on the green carriage. The cable is
# driven only by the two blue input pulleys.
#
# The sample combines passive revolute pulleys, a closed cable loop, and two
# commanded input pulleys. The input rotations trace a rectangle with the
# beige table marker while the solver resolves cable wrapping and contact
# against the guides.
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples

TABLE_RECT_HALF_X = 0.050
TABLE_RECT_HALF_Y = 0.060
TABLE_RECT_PERIOD = 16.0
TABLE_TRACKING_MAX_ERROR_TOLERANCE = 0.005
TABLE_TRACKING_RMS_ERROR_TOLERANCE = 0.0025
CABLE_XY_ABS_BOUND = 0.30
JOINT_LIMIT_TOLERANCE = 0.003
START_RAMP_DURATION = 1.2
MOUSE_PICK_STIFFNESS = 0.01
MOUSE_PICK_DAMPING = 0.001


@wp.kernel
def drive_input_pulleys(
    sim_time: wp.array[wp.float32],
    body_indices: wp.array[wp.int32],
    body_base_xforms: wp.array[wp.transform],
    input_drive_radius: float,
    input_pulley_angles: wp.array[wp.float32],
    target_table_xy: wp.array[wp.float32],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Drive the two blue input pulleys along the rectangular table path."""
    tid = wp.tid()
    body = body_indices[tid]
    base_xform = body_base_xforms[tid]

    t = sim_time[0]

    # The two input pulley rotations are the only prescribed motion. The
    # slide and table are moved only by the cable and passive pulleys. In
    # this layout, a direct cable-drive command maps approximately to world
    # (x, y) = (command_y, -command_x), so invert that mapping first.
    ramp = wp.clamp(t / START_RAMP_DURATION, 0.0, 1.0)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    phase_time = t - wp.floor(t / TABLE_RECT_PERIOD) * TABLE_RECT_PERIOD
    side = 4.0 * phase_time / TABLE_RECT_PERIOD

    table_x = -TABLE_RECT_HALF_X
    table_y = -TABLE_RECT_HALF_Y
    if side < 1.0:
        table_x = -TABLE_RECT_HALF_X + 2.0 * TABLE_RECT_HALF_X * side
    elif side < 2.0:
        table_x = TABLE_RECT_HALF_X
        table_y = -TABLE_RECT_HALF_Y + 2.0 * TABLE_RECT_HALF_Y * (side - 1.0)
    elif side < 3.0:
        table_x = TABLE_RECT_HALF_X - 2.0 * TABLE_RECT_HALF_X * (side - 2.0)
        table_y = TABLE_RECT_HALF_Y
    else:
        table_y = TABLE_RECT_HALF_Y - 2.0 * TABLE_RECT_HALF_Y * (side - 3.0)

    target_x = ramp * table_x
    target_y = ramp * table_y
    if tid == 0:
        target_table_xy[0] = target_x
        target_table_xy[1] = target_y
    command_x = -target_y
    command_y = target_x
    q_left = (command_x + command_y) / input_drive_radius
    q_right = (command_y - command_x) / input_drive_radius

    p = wp.transform_get_translation(base_xform)
    q = wp.transform_get_rotation(base_xform)

    angle = q_left
    if tid == 1:
        angle = q_right
    input_pulley_angles[tid] = angle
    q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle), q)

    xform = wp.transform(p, q)
    body_q0[body] = xform
    body_q1[body] = xform


@wp.kernel
def advance_time(sim_time: wp.array[wp.float32], dt: float):
    """Advance a device-side time accumulator for graph-captured simulation."""
    sim_time[0] = sim_time[0] + dt


@wp.kernel
def set_body_xforms(
    body_indices: wp.array[wp.int32],
    body_xforms: wp.array[wp.transform],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    """Initialize selected body transforms in both state buffers."""
    tid = wp.tid()
    body = body_indices[tid]
    xform = body_xforms[tid]
    body_q0[body] = xform
    body_q1[body] = xform


def _symmetric_bounds(half_extent: float) -> tuple[float, float]:
    return (-half_extent, half_extent)


def _pad_bounds(bounds: tuple[float, float], padding: float) -> tuple[float, float]:
    return (bounds[0] - padding, bounds[1] + padding)


def _check_range(label: str, value: float, bounds: tuple[float, float]):
    lower, upper = bounds
    if not lower <= value <= upper:
        raise ValueError(f"{label} {value:.4f} m is outside [{lower:.4f}, {upper:.4f}] m.")


def _check_abs_bound(label: str, value: float, bound: float):
    if abs(value) > bound:
        raise ValueError(f"{label} {value:.4f} m exceeds +/-{bound:.4f} m.")


def _dim_color(color: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, c * scale)) for c in color)


def _make_body_kinematic(builder: newton.ModelBuilder, body: int):
    """Clear body mass properties so the solver treats the body as kinematic."""
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)


def add_pulley(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    radius: float,
    cable_radius: float,
    color: tuple[float, float, float],
    *,
    parent: int | None,
    sheave_mu: float,
    label: str | None = None,
) -> tuple[int, int | None]:
    """Add one XY-table pulley.

    A parent of ``None`` means the pulley is one of the two driven blue
    inputs. Pulleys with a parent are free-spinning revolute joints on that
    stage body.
    """
    body = builder.add_link(
        xform=wp.transform(center, wp.quat_identity()),
        is_kinematic=parent is None,
        label=f"{label}_body" if label else None,
    )

    joint = None
    if parent is None:
        _make_body_kinematic(builder, body)
    else:
        parent_pose = builder.body_q[parent]
        parent_position = wp.transform_get_translation(parent_pose)
        joint = builder.add_joint_revolute(
            parent=parent,
            child=body,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(center - parent_position, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            armature=1.0e-4,
            friction=1.0,
            label=f"{label}_free_axle" if label else None,
        )

    groove_half_width = 1.55 * cable_radius
    flange_half_thickness = 0.6 * cable_radius
    flange_radius = radius + 3.2 * cable_radius
    sheave_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e5, kd=0.0, mu=sheave_mu)
    flange_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e5, kd=0.0, mu=0.0)
    flange_color = _dim_color(color, 0.68)

    for suffix, z, shape_radius, half_height, cfg, shape_color in (
        ("sheave", 0.0, radius, groove_half_width, sheave_cfg, color),
        (
            "flange_neg",
            -(groove_half_width + flange_half_thickness),
            flange_radius,
            flange_half_thickness,
            flange_cfg,
            flange_color,
        ),
        (
            "flange_pos",
            groove_half_width + flange_half_thickness,
            flange_radius,
            flange_half_thickness,
            flange_cfg,
            flange_color,
        ),
    ):
        builder.add_shape_cylinder(
            body=body,
            xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()),
            radius=shape_radius,
            half_height=half_height,
            cfg=cfg,
            color=shape_color,
            label=f"{label}_{suffix}" if label else None,
        )

    marker_radius = 0.75 * cable_radius
    marker_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    builder.add_shape_sphere(
        body=body,
        xform=wp.transform(
            wp.vec3(0.78 * radius, 0.0, groove_half_width + 2.0 * flange_half_thickness + 0.35 * marker_radius),
            wp.quat_identity(),
        ),
        radius=marker_radius,
        cfg=marker_cfg,
        color=(0.96, 0.92, 0.72),
        label=f"{label}_rotation_dot" if label else None,
    )

    return body, joint


def filter_body_group_collisions(builder: newton.ModelBuilder, bodies: list[int]):
    """Disables collision pairs within a body group."""
    for i, body_a in enumerate(bodies):
        for body_b in bodies[i + 1 :]:
            for shape_a in builder.body_shapes.get(body_a, []):
                for shape_b in builder.body_shapes.get(body_b, []):
                    builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))


def append_route_point(points: list[wp.vec3], point: wp.vec3):
    """Append a point unless it duplicates the previous route point."""
    if not points or float(wp.length(point - points[-1])) > 1.0e-8:
        points.append(point)


def append_arc_xy(
    points: list[wp.vec3],
    center: wp.vec3,
    radius: float,
    start_angle: float,
    end_angle: float,
    segment_length: float,
    *,
    direction: str | None = None,
):
    """Append a polyline approximation of a circular arc in the XY plane."""
    delta = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    if direction == "cw" and delta > 0.0:
        delta -= 2.0 * math.pi
    elif direction == "ccw" and delta < 0.0:
        delta += 2.0 * math.pi

    arc_length = abs(delta) * radius
    count = max(3, int(math.ceil(arc_length / segment_length)))
    for i in range(count + 1):
        u = float(i) / float(count)
        angle = start_angle + delta * u
        point = wp.vec3(
            float(center[0]) + radius * math.cos(angle),
            float(center[1]) + radius * math.sin(angle),
            float(center[2]),
        )
        append_route_point(points, point)


def resample_equal_length_segments(route_points: list[wp.vec3], segment_length: float) -> tuple[list[wp.vec3], float]:
    """Resample a route into equal-length segments close to the requested length."""
    if len(route_points) < 2:
        raise ValueError("route_points must contain at least two points")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be positive")

    points = [route_points[0]]
    distances = [0.0]
    total_length = 0.0
    for route_point in route_points[1:]:
        length = float(wp.length(route_point - points[-1]))
        if length <= 1.0e-8:
            continue
        total_length += length
        points.append(route_point)
        distances.append(total_length)

    if total_length <= 1.0e-8:
        raise ValueError("route_points must span a non-zero length")

    segment_count = max(2, int(math.ceil(total_length / segment_length)))
    resampled_segment_length = total_length / float(segment_count)
    resampled = [points[0]]
    point_index = 1
    for segment_index in range(1, segment_count):
        target_distance = resampled_segment_length * float(segment_index)
        while point_index < len(points) - 1 and distances[point_index] < target_distance:
            point_index += 1

        previous_distance = distances[point_index - 1]
        next_distance = distances[point_index]
        u = (target_distance - previous_distance) / (next_distance - previous_distance)
        resampled.append(points[point_index - 1] * (1.0 - u) + points[point_index] * u)

    resampled.append(points[-1])
    return resampled, resampled_segment_length


def create_xy_table_cable_points(
    start: wp.vec3,
    pulley_centers: list[wp.vec3],
    pulley_radii: list[float],
    end: wp.vec3,
    segment_length: float,
    wrap_clearance: float,
) -> tuple[list[wp.vec3], float]:
    """Create the wrapped cable path with equal-length segments."""
    pulley_arcs = (
        (0.0, 0.5 * math.pi, "ccw"),
        (-0.5 * math.pi, 0.5 * math.pi, "cw"),
        (-0.5 * math.pi, 0.0, "ccw"),
        (math.pi, 0.0, "cw"),
        (math.pi, -0.5 * math.pi, "ccw"),
        (0.5 * math.pi, -0.5 * math.pi, "cw"),
        (0.5 * math.pi, math.pi, "ccw"),
    )
    if len(pulley_centers) != len(pulley_arcs) or len(pulley_radii) != len(pulley_arcs):
        raise ValueError("XY table cable route expects seven pulleys")
    if wrap_clearance < 0.0:
        raise ValueError("wrap_clearance must be non-negative")

    route_points = [start]

    # Green pulleys use their inner quadrants. Blue input pulleys and the
    # beige top pulley use the outside path.
    for center, radius, (start_angle, end_angle, direction) in zip(
        pulley_centers,
        pulley_radii,
        pulley_arcs,
        strict=True,
    ):
        append_arc_xy(
            route_points,
            center,
            radius + wrap_clearance,
            start_angle,
            end_angle,
            segment_length,
            direction=direction,
        )
    append_route_point(route_points, end)
    return resample_equal_length_segments(route_points, segment_length)


def add_visual_bar(
    builder: newton.ModelBuilder,
    *,
    body: int,
    center: wp.vec3,
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float],
    label: str,
    density: float = 1000.0,
):
    """Add a non-colliding box used to visualize a table component."""
    cfg = newton.ModelBuilder.ShapeConfig(
        density=density,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    return builder.add_shape_box(
        body=body,
        xform=wp.transform(center, wp.quat_identity()),
        hx=half_extents[0],
        hy=half_extents[1],
        hz=half_extents[2],
        cfg=cfg,
        color=color,
        label=label,
    )


class Example:
    def __init__(self, viewer, args):
        # Store viewer and configure simulation cadence.
        self.viewer = viewer

        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable and mechanism dimensions.
        cable_radius = 0.003
        self.input_pulley_radius = 0.025
        green_sheave_radius = 0.015
        beige_sheave_radius = 0.025
        input_sheave_mu = 1.0
        passive_sheave_mu = 0.35
        initial_segment_length = 0.015
        cable_wrap_clearance_scale = 1.1
        cable_wrap_clearance = cable_radius * cable_wrap_clearance_scale
        self.input_drive_radius = self.input_pulley_radius + cable_wrap_clearance

        blue = (0.12, 0.34, 0.76)
        green = (0.12, 0.58, 0.28)
        beige = (0.74, 0.63, 0.45)

        # The mechanism is flattened onto the XY plane and layered in Z only
        # enough to keep the base, slide, table, and pulleys visually distinct.
        base_z = 0.006
        self.slide_z = 0.014
        self.table_z = 0.022
        self.pulley_z = 0.046
        z_bound_pad = JOINT_LIMIT_TOLERANCE
        self.cable_z_bounds = (self.pulley_z - z_bound_pad, self.pulley_z + z_bound_pad)
        self.slide_z_bounds = (self.slide_z - z_bound_pad, self.slide_z + z_bound_pad)
        self.table_z_bounds = (self.table_z - z_bound_pad, self.table_z + z_bound_pad)

        # Build the table frame and assign stiff, frictional contact material
        # so the cable remains guided by the pulley grooves.
        builder = newton.ModelBuilder()
        builder.rigid_gap = 5.0 * cable_radius
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 1.0

        base_origin = wp.vec3(0.0, 0.0, base_z)
        slide_origin = wp.vec3(0.0, 0.0, self.slide_z)
        table_origin = wp.vec3(0.0, 0.0, self.table_z)

        # Fixed blue base.
        add_visual_bar(
            builder,
            body=-1,
            center=base_origin,
            half_extents=(0.205, 0.025, 0.006),
            color=blue,
            label="fixed_blue_base",
            density=1000.0,
        )

        # Moving table stages: the green carriage slides in X and the beige
        # carriage rides on it in Y.
        self.slide_body = builder.add_link(
            xform=wp.transform(slide_origin, wp.quat_identity()),
            label="green_x_slide",
        )
        self.table_body = builder.add_link(
            xform=wp.transform(table_origin, wp.quat_identity()),
            label="beige_y_table",
        )
        self.table_origin_xy = (float(table_origin[0]), float(table_origin[1]))
        self.table_tracking_max_error = 0.0
        self.table_tracking_error_sq_sum = 0.0
        self.table_tracking_sample_count = 0
        self.slide_x_bounds = _symmetric_bounds(TABLE_RECT_HALF_X)
        self.table_y_bounds = _symmetric_bounds(TABLE_RECT_HALF_Y)

        slide_joint = builder.add_joint_prismatic(
            parent=-1,
            child=self.slide_body,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(slide_origin, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=self.slide_x_bounds[0],
            limit_upper=self.slide_x_bounds[1],
            limit_ke=2.0e3,
            limit_kd=1.0e-4,
            friction=0.0,
            label="green_x_slide_axis",
        )
        table_joint = builder.add_joint_prismatic(
            parent=self.slide_body,
            child=self.table_body,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(table_origin - slide_origin, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=self.table_y_bounds[0],
            limit_upper=self.table_y_bounds[1],
            limit_ke=2.0e3,
            limit_kd=1.0e-4,
            friction=0.0,
            label="beige_y_table_axis",
        )
        table_articulation_joints = [slide_joint, table_joint]

        # Non-colliding stage visuals and a marker used for table tracking.
        add_visual_bar(
            builder,
            body=self.slide_body,
            center=wp.vec3(0.0, 0.0, 0.0),
            half_extents=(0.085, 0.052, 0.006),
            color=green,
            label="green_horizontal_carriage",
            density=1000.0,
        )
        add_visual_bar(
            builder,
            body=self.table_body,
            center=wp.vec3(0.0, 0.0, 0.0),
            half_extents=(0.013, 0.215, 0.006),
            color=beige,
            label="beige_vertical_carriage",
            density=1000.0,
        )
        table_marker_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        builder.add_shape_sphere(
            body=self.table_body,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.014), wp.quat_identity()),
            radius=0.008,
            cfg=table_marker_cfg,
            color=(0.5, 0.5, 0.5),
            label="beige_table_center_marker",
        )

        # Seven pulleys define the cross-base cable route. Blue pulleys are
        # kinematic inputs; green and beige pulleys spin passively on their
        # parent stage bodies.
        pulley_specs = [
            (
                "green_lower_left",
                self.slide_body,
                green,
                wp.vec3(-0.045, -0.045, self.pulley_z),
                green_sheave_radius,
                passive_sheave_mu,
            ),
            (
                "blue_input_left",
                None,
                blue,
                wp.vec3(-0.19, 0.0, self.pulley_z),
                self.input_pulley_radius,
                input_sheave_mu,
            ),
            (
                "green_upper_left",
                self.slide_body,
                green,
                wp.vec3(-0.045, 0.045, self.pulley_z),
                green_sheave_radius,
                passive_sheave_mu,
            ),
            (
                "beige_top",
                self.table_body,
                beige,
                wp.vec3(0.0, 0.19, self.pulley_z),
                beige_sheave_radius,
                passive_sheave_mu,
            ),
            (
                "green_upper_right",
                self.slide_body,
                green,
                wp.vec3(0.045, 0.045, self.pulley_z),
                green_sheave_radius,
                passive_sheave_mu,
            ),
            (
                "blue_input_right",
                None,
                blue,
                wp.vec3(0.19, 0.0, self.pulley_z),
                self.input_pulley_radius,
                input_sheave_mu,
            ),
            (
                "green_lower_right",
                self.slide_body,
                green,
                wp.vec3(0.045, -0.045, self.pulley_z),
                green_sheave_radius,
                passive_sheave_mu,
            ),
        ]

        self.pulley_bodies: list[int] = []
        driven_pulley_bodies: list[int] = []
        pulley_centers = [spec[3] for spec in pulley_specs]
        pulley_radii = [spec[4] for spec in pulley_specs]

        for i, (label, parent, color, center, sheave_radius, sheave_mu) in enumerate(pulley_specs, start=1):
            pulley_body, pulley_joint = add_pulley(
                builder,
                center,
                sheave_radius,
                cable_radius,
                color,
                parent=parent,
                sheave_mu=sheave_mu,
                label=f"xy_table_{i}_{label}",
            )
            if pulley_joint is None:
                driven_pulley_bodies.append(pulley_body)
            else:
                table_articulation_joints.append(pulley_joint)
            self.pulley_bodies.append(pulley_body)

        # The cable loop starts and ends on the bottom of the beige table.
        self.left_anchor_local = wp.vec3(-0.028, -0.21, self.pulley_z - self.table_z)
        self.right_anchor_local = wp.vec3(0.028, -0.21, self.pulley_z - self.table_z)
        left_anchor_world = table_origin + self.left_anchor_local
        right_anchor_world = table_origin + self.right_anchor_local

        anchor_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        for label, anchor in (
            ("left_bottom_cable_fix", self.left_anchor_local),
            ("right_bottom_cable_fix", self.right_anchor_local),
        ):
            builder.add_shape_sphere(
                body=self.table_body,
                xform=wp.transform(anchor, wp.quat_identity()),
                radius=0.0075,
                cfg=anchor_cfg,
                color=beige,
                label=label,
            )

        # Adjust the cable length so every segment is equal.
        cable_points, cable_segment_length = create_xy_table_cable_points(
            start=left_anchor_world,
            pulley_centers=pulley_centers,
            pulley_radii=pulley_radii,
            end=right_anchor_world,
            segment_length=initial_segment_length,
            wrap_clearance=cable_wrap_clearance,
        )
        cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
        cable_segment_count = len(cable_points) - 1
        straight_cable_points, straight_cable_quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=left_anchor_world,
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=cable_segment_count * cable_segment_length,
            num_segments=cable_segment_count,
        )

        cable_cfg = builder.default_shape_cfg.copy()
        cable_cfg.density = 200.0
        cable_cfg.gap = 2.0 * cable_radius

        self.cable_bodies, cable_joints = builder.add_rod(
            positions=straight_cable_points,
            quaternions=straight_cable_quats,
            radius=cable_radius,
            cfg=cable_cfg,
            stretch_stiffness=1.0e5,
            stretch_damping=1.0e-4,
            bend_stiffness=1.0e-2,
            bend_damping=1.0e-2,
            wrap_in_articulation=False,
            label="xy_table_cable",
            body_frame_origin="com",
        )
        initial_cable_xforms = [
            wp.transform(cable_points[i] + (cable_points[i + 1] - cable_points[i]) * 0.5, cable_quats[i])
            for i in range(len(self.cable_bodies))
        ]
        filter_body_group_collisions(builder, self.cable_bodies)

        # Ball joints close the cable loop at the table anchors.
        first_cable_body = self.cable_bodies[0]
        last_cable_body = self.cable_bodies[-1]
        first_endpoint_local = wp.vec3(0.0, 0.0, -0.5 * cable_segment_length)
        last_endpoint_local = wp.vec3(0.0, 0.0, 0.5 * cable_segment_length)
        first_cable_anchor_xform = wp.transform(first_endpoint_local, wp.quat_identity())
        last_cable_anchor_xform = wp.transform(last_endpoint_local, wp.quat_identity())
        for i, (body, xform) in enumerate(
            (
                (first_cable_body, first_cable_anchor_xform),
                (last_cable_body, last_cable_anchor_xform),
            )
        ):
            builder.add_shape_sphere(
                body=body,
                xform=xform,
                radius=1.6 * cable_radius,
                cfg=anchor_cfg,
                color=beige,
                label=f"visual_cable_end_{i}",
            )

        left_anchor_joint = builder.add_joint_ball(
            parent=self.table_body,
            child=first_cable_body,
            parent_xform=wp.transform(self.left_anchor_local, wp.quat_identity()),
            child_xform=first_cable_anchor_xform,
            armature=1.0e-5,
            friction=0.0,
            label="left_bottom_cable_fix",
        )
        builder.add_joint_ball(
            parent=self.table_body,
            child=last_cable_body,
            parent_xform=wp.transform(self.right_anchor_local, wp.quat_identity()),
            child_xform=last_cable_anchor_xform,
            armature=1.0e-5,
            friction=0.0,
            label="right_bottom_cable_fix_loop",
        )
        builder.add_articulation(
            [*table_articulation_joints, *cable_joints, left_anchor_joint],
            label="xy_table_cable_cross_slide",
        )

        kinematic_body_indices = driven_pulley_bodies
        kinematic_body_base_xforms = [builder.body_q[body] for body in kinematic_body_indices]

        builder.add_ground_plane()
        builder.color(balance_colors=False)

        # Finalize the model and use VBD with explicit broad-phase contacts.
        sim_device = wp.get_device(args.device) if args.device else None
        self.model = builder.finalize(device=sim_device)
        self.model.set_gravity((0.0, 0.0, 0.0))

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=sim_iterations,
            rigid_body_contact_buffer_size=256,
            rigid_contact_hard=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.model.contacts(collision_pipeline=pipeline)

        # Device arrays used by kernels during simulation and CUDA graph replay.
        self.kinematic_body_indices = wp.array(
            kinematic_body_indices,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.kinematic_body_base_xforms = wp.array(
            kinematic_body_base_xforms,
            dtype=wp.transform,
            device=self.model.device,
        )
        self.input_pulley_angles = wp.zeros(
            len(kinematic_body_indices),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.target_table_xy = wp.zeros(2, dtype=wp.float32, device=self.model.device)
        cable_body_indices = wp.array(
            self.cable_bodies,
            dtype=wp.int32,
            device=self.model.device,
        )
        cable_body_xforms = wp.array(
            initial_cable_xforms,
            dtype=wp.transform,
            device=self.model.device,
        )
        wp.launch(
            set_body_xforms,
            dim=cable_body_indices.shape[0],
            inputs=[
                cable_body_indices,
                cable_body_xforms,
                self.state_0.body_q,
                self.state_1.body_q,
            ],
            device=self.model.device,
        )
        self.solver.body_q_prev = wp.clone(self.state_0.body_q, device=self.solver.device)
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        # Viewer setup.
        self.viewer.set_model(self.model)
        picking = getattr(self.viewer, "picking", None)
        if picking is not None:
            pick_state = picking.pick_state.numpy()
            pick_state[0]["pick_stiffness"] = MOUSE_PICK_STIFFNESS
            pick_state[0]["pick_damping"] = MOUSE_PICK_DAMPING
            picking.pick_state.assign(pick_state)

        self.viewer.set_camera(
            pos=wp.vec3(0.0, 0.0, 0.8),
            pitch=-90.0,
            yaw=90.0,
        )

        self.capture()

    def capture(self):
        """Capture the simulation update when running on CUDA."""
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        """Advance the XY table simulation by one rendered frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                drive_input_pulleys,
                dim=self.kinematic_body_indices.shape[0],
                inputs=[
                    self.sim_time_wp,
                    self.kinematic_body_indices,
                    self.kinematic_body_base_xforms,
                    self.input_drive_radius,
                    self.input_pulley_angles,
                    self.target_table_xy,
                    self.state_0.body_q,
                    self.state_1.body_q,
                ],
                device=self.model.device,
            )

            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            wp.launch(advance_time, dim=1, inputs=[self.sim_time_wp, self.sim_dt], device=self.model.device)

    def step(self):
        """Step the simulation and update logged diagnostics."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.record_diagrams()

    def record_diagrams(self):
        """Log pulley rotations and table position for viewer diagrams."""
        input_pulley_angles = self.input_pulley_angles.numpy()
        q_left = float(input_pulley_angles[0])
        q_right = float(input_pulley_angles[1])
        body_q = self.state_0.body_q.numpy()
        target_table_xy = self.target_table_xy.numpy()
        table_pos = body_q[self.table_body, 0:3]
        table_x = float(table_pos[0]) - self.table_origin_xy[0]
        table_y = float(table_pos[1]) - self.table_origin_xy[1]
        table_xy = np.array((table_x, table_y), dtype=np.float32)
        tracking_error = float(np.linalg.norm(table_xy - target_table_xy))
        self.table_tracking_max_error = max(self.table_tracking_max_error, tracking_error)
        self.table_tracking_error_sq_sum += tracking_error * tracking_error
        self.table_tracking_sample_count += 1

        self.viewer.log_scalar("Blue left input rotation [rad]", q_left)
        self.viewer.log_scalar("Blue right input rotation [rad]", q_right)
        self.viewer.log_scalar("Beige table X position [m]", table_x)
        self.viewer.log_scalar("Beige table Y position [m]", table_y)
        self.viewer.log_scalar("Beige table tracking error [m]", tracking_error)

    def render(self):
        """Render the current simulation state and contact points."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def _check_state_bounds(self, body_q: np.ndarray):
        """Validate that the mechanism remains finite and inside its workspace."""
        if not np.all(np.isfinite(body_q)):
            raise ValueError("NaN/Inf in body transforms.")

        self._check_cable_bounds(body_q)
        self._check_table_stage_bounds(body_q)

    def _check_cable_bounds(self, body_q: np.ndarray):
        """Validate that the cable has not escaped the table workspace."""
        cable_pos = body_q[[int(body) for body in self.cable_bodies], 0:3]
        _check_range("Cable minimum Z", float(np.min(cable_pos[:, 2])), self.cable_z_bounds)
        _check_range("Cable maximum Z", float(np.max(cable_pos[:, 2])), self.cable_z_bounds)
        _check_abs_bound("Cable maximum XY displacement", float(np.max(np.abs(cable_pos[:, 0:2]))), CABLE_XY_ABS_BOUND)

    def _check_table_stage_bounds(self, body_q: np.ndarray):
        """Validate that the slide and table bodies stay near their joint limits."""
        slide_pos = body_q[self.slide_body, 0:3]
        table_pos = body_q[self.table_body, 0:3]
        slide_x_bounds = _pad_bounds(self.slide_x_bounds, JOINT_LIMIT_TOLERANCE)
        table_y_bounds = _pad_bounds(self.table_y_bounds, JOINT_LIMIT_TOLERANCE)

        _check_range("Horizontal green carriage X", float(slide_pos[0]), slide_x_bounds)
        _check_range("Vertical beige carriage Y", float(table_pos[1]), table_y_bounds)
        _check_range("Horizontal green carriage Z", float(slide_pos[2]), self.slide_z_bounds)
        _check_range("Vertical beige carriage Z", float(table_pos[2]), self.table_z_bounds)

    def test_post_step(self):
        """Catch instability as soon as a rendered frame completes."""
        if self.state_0.body_q is None:
            raise RuntimeError("Body state is not available.")

        body_q = self.state_0.body_q.numpy()
        self._check_state_bounds(body_q)

    def test_final(self):
        """Validate table drift and final mechanism bounds."""
        if self.state_0.body_q is None:
            raise RuntimeError("Body state is not available.")

        body_q = self.state_0.body_q.numpy()
        self._check_state_bounds(body_q)

        if self.table_tracking_sample_count == 0:
            raise ValueError("No table tracking samples were recorded.")

        table_tracking_rms_error = math.sqrt(self.table_tracking_error_sq_sum / self.table_tracking_sample_count)
        if self.table_tracking_max_error > TABLE_TRACKING_MAX_ERROR_TOLERANCE:
            raise ValueError(
                "XY table drifted too far from the commanded path: "
                f"max error {self.table_tracking_max_error:.4f} m exceeds "
                f"{TABLE_TRACKING_MAX_ERROR_TOLERANCE:.4f} m."
            )
        if table_tracking_rms_error > TABLE_TRACKING_RMS_ERROR_TOLERANCE:
            raise ValueError(
                "XY table tracking error stayed too high: "
                f"RMS error {table_tracking_rms_error:.4f} m exceeds "
                f"{TABLE_TRACKING_RMS_ERROR_TOLERANCE:.4f} m."
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
