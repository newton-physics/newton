# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable XY Table
#
# Demonstrates a cable-driven cross-base XY table inspired by the Simscape
# Multibody cable-driven XY table example. The mechanism is laid out on the
# ground plane: the blue base is fixed, the green carriage moves horizontally,
# and the beige carriage moves vertically on the green carriage. The cable is
# driven only by the two blue input pulleys.
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples

ATTACH_BASE = 0
ATTACH_SLIDE = 1
ATTACH_TABLE = 2

MOTOR_NONE = 0
MOTOR_INPUT_LEFT = 1
MOTOR_INPUT_RIGHT = 2

TABLE_RECT_HALF_X = 0.050
TABLE_RECT_HALF_Y = 0.060
TABLE_RECT_PERIOD = 16.0
TABLE_RECT_POINTS = (
    (-TABLE_RECT_HALF_X, -TABLE_RECT_HALF_Y),
    (TABLE_RECT_HALF_X, -TABLE_RECT_HALF_Y),
    (TABLE_RECT_HALF_X, TABLE_RECT_HALF_Y),
    (-TABLE_RECT_HALF_X, TABLE_RECT_HALF_Y),
)
TABLE_RECT_HIT_TOLERANCE = 0.025
TABLE_RECT_TEST_FRAMES = 2100
START_RAMP_DURATION = 1.2


@wp.kernel
def drive_input_pulleys(
    sim_time: wp.array[wp.float32],
    body_indices: wp.array[wp.int32],
    body_base_xforms: wp.array[wp.transform],
    body_motor: wp.array[wp.int32],
    pulley_radius: float,
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
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
    command_x = -target_y
    command_y = target_x
    q2 = (command_x + command_y) / pulley_radius
    q6 = (command_y - command_x) / pulley_radius

    p = wp.transform_get_translation(base_xform)
    q = wp.transform_get_rotation(base_xform)

    motor = body_motor[tid]
    if motor == MOTOR_INPUT_LEFT:
        q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), q2), q)
    elif motor == MOTOR_INPUT_RIGHT:
        q = wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), q6), q)

    xform = wp.transform(p, q)
    body_q0[body] = xform
    body_q1[body] = xform


@wp.kernel
def advance_time(sim_time: wp.array[wp.float32], dt: float):
    sim_time[0] = sim_time[0] + dt


@wp.kernel
def set_body_xforms(
    body_indices: wp.array[wp.int32],
    body_xforms: wp.array[wp.transform],
    body_q0: wp.array[wp.transform],
    body_q1: wp.array[wp.transform],
):
    tid = wp.tid()
    body = body_indices[tid]
    xform = body_xforms[tid]
    body_q0[body] = xform
    body_q1[body] = xform


def compute_input_pulley_rotations(t: float, pulley_radius: float) -> tuple[float, float]:
    ramp = min(1.0, max(0.0, t / START_RAMP_DURATION))
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    phase_time = t - math.floor(t / TABLE_RECT_PERIOD) * TABLE_RECT_PERIOD
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
    command_x = -target_y
    command_y = target_x
    q2 = (command_x + command_y) / pulley_radius
    q6 = (command_y - command_x) / pulley_radius
    return q2, q6


def _dim_color(color: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, c * scale)) for c in color)


def _make_body_kinematic(builder: newton.ModelBuilder, body: int):
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)


def add_guided_pulley(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    axis: wp.vec3,
    sheave_diameter: float,
    cable_radius: float,
    *,
    body: int = -1,
    color: tuple[float, float, float] = (0.42, 0.45, 0.48),
    groove_width_scale: float = 3.0,
    flange_radius_scale: float = 3.0,
    flange_thickness_scale: float = 1.0,
    ke: float = 1.0e6,
    kd: float = 1.0e-1,
    mu: float = 0.0,
    density: float = 1000.0,
    label: str | None = None,
) -> tuple[int, int, int]:
    """Adds a flanged pulley guide built from primitive collision shapes."""
    if sheave_diameter <= 0.0:
        raise ValueError("sheave_diameter must be positive")
    if cable_radius <= 0.0:
        raise ValueError("cable_radius must be positive")
    if float(wp.length(axis)) <= 1.0e-8:
        raise ValueError("axis must be non-zero")

    axis = wp.normalize(axis)
    q_axis = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis)

    sheave_radius = 0.5 * sheave_diameter
    groove_half_width = 0.5 * groove_width_scale * cable_radius
    flange_radius = sheave_radius + flange_radius_scale * cable_radius
    flange_half_thickness = 0.5 * flange_thickness_scale * cable_radius

    cfg = newton.ModelBuilder.ShapeConfig(density=density, ke=ke, kd=kd, mu=mu)
    flange_color = _dim_color(color, 0.68)

    sheave = builder.add_shape_cylinder(
        body=body,
        xform=wp.transform(center, q_axis),
        radius=sheave_radius,
        half_height=groove_half_width,
        cfg=cfg,
        color=color,
        label=f"{label}_sheave" if label else None,
    )

    flange_neg = builder.add_shape_cylinder(
        body=body,
        xform=wp.transform(center - axis * (groove_half_width + flange_half_thickness), q_axis),
        radius=flange_radius,
        half_height=flange_half_thickness,
        cfg=cfg,
        color=flange_color,
        label=f"{label}_flange_neg" if label else None,
    )

    flange_pos = builder.add_shape_cylinder(
        body=body,
        xform=wp.transform(center + axis * (groove_half_width + flange_half_thickness), q_axis),
        radius=flange_radius,
        half_height=flange_half_thickness,
        cfg=cfg,
        color=flange_color,
        label=f"{label}_flange_pos" if label else None,
    )

    return sheave, flange_neg, flange_pos


def add_pulley_rotation_dot(
    builder: newton.ModelBuilder,
    *,
    body: int,
    sheave_diameter: float,
    cable_radius: float,
    groove_width_scale: float,
    flange_thickness_scale: float,
    color: tuple[float, float, float] = (0.96, 0.92, 0.72),
    label: str | None = None,
) -> int:
    """Adds a tiny non-colliding off-center dot to show pulley rotation."""

    sheave_radius = 0.5 * sheave_diameter
    groove_half_width = 0.5 * groove_width_scale * cable_radius
    flange_half_thickness = 0.5 * flange_thickness_scale * cable_radius
    marker_radius = 0.75 * cable_radius
    marker_z = groove_half_width + 2.0 * flange_half_thickness + 0.35 * marker_radius
    marker_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
    )

    return builder.add_shape_sphere(
        body=body,
        xform=wp.transform(wp.vec3(0.78 * sheave_radius, 0.0, marker_z), wp.quat_identity()),
        radius=marker_radius,
        cfg=marker_cfg,
        color=color,
        label=f"{label}_rotation_dot" if label else None,
    )


def add_kinematic_guided_pulley(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    axis: wp.vec3,
    sheave_diameter: float,
    cable_radius: float,
    *,
    color: tuple[float, float, float] = (0.42, 0.45, 0.48),
    groove_width_scale: float = 3.0,
    flange_radius_scale: float = 3.0,
    flange_thickness_scale: float = 1.0,
    ke: float = 1.0e6,
    kd: float = 1.0e-1,
    mu: float = 1.0,
    density: float = 250.0,
    label: str | None = None,
) -> tuple[int, tuple[int, int, int, int]]:
    """Adds a kinematic flanged pulley body used as a moving cable guide."""
    if float(wp.length(axis)) <= 1.0e-8:
        raise ValueError("axis must be non-zero")

    axis = wp.normalize(axis)
    body = builder.add_link(
        xform=wp.transform(center, wp.quat_identity()),
        is_kinematic=True,
        label=f"{label}_body" if label else None,
    )

    shapes = add_guided_pulley(
        builder,
        center=wp.vec3(0.0, 0.0, 0.0),
        axis=axis,
        sheave_diameter=sheave_diameter,
        cable_radius=cable_radius,
        body=body,
        color=color,
        groove_width_scale=groove_width_scale,
        flange_radius_scale=flange_radius_scale,
        flange_thickness_scale=flange_thickness_scale,
        ke=ke,
        kd=kd,
        mu=mu,
        density=density,
        label=label,
    )
    marker = add_pulley_rotation_dot(
        builder,
        body=body,
        sheave_diameter=sheave_diameter,
        cable_radius=cable_radius,
        groove_width_scale=groove_width_scale,
        flange_thickness_scale=flange_thickness_scale,
        label=label,
    )

    _make_body_kinematic(builder, body)

    return body, (*shapes, marker)


def filter_body_group_collisions(builder: newton.ModelBuilder, bodies: list[int]):
    """Disables collision pairs within a body group."""
    for i, body_a in enumerate(bodies):
        for body_b in bodies[i + 1 :]:
            for shape_a in builder.body_shapes.get(body_a, []):
                for shape_b in builder.body_shapes.get(body_b, []):
                    builder.add_shape_collision_filter_pair(int(shape_a), int(shape_b))


def add_passive_guided_pulley(
    builder: newton.ModelBuilder,
    center: wp.vec3,
    axis: wp.vec3,
    sheave_diameter: float,
    cable_radius: float,
    *,
    parent: int,
    color: tuple[float, float, float] = (0.42, 0.45, 0.48),
    groove_width_scale: float = 3.0,
    flange_radius_scale: float = 3.0,
    flange_thickness_scale: float = 1.0,
    ke: float = 1.0e6,
    kd: float = 1.0e-1,
    mu: float = 0.35,
    density: float = 250.0,
    axle_armature: float = 1.0e-4,
    axle_friction: float = 0.0,
    label: str | None = None,
) -> tuple[int, int, tuple[int, int, int, int]]:
    """Adds a flanged pulley that spins freely on its parent body."""
    if float(wp.length(axis)) <= 1.0e-8:
        raise ValueError("axis must be non-zero")

    axis = wp.normalize(axis)
    body = builder.add_link(xform=wp.transform(center, wp.quat_identity()), label=f"{label}_body" if label else None)

    if parent == -1:
        parent_center = center
        joint_axis = axis
    else:
        parent_pose = builder.body_q[parent]
        parent_position = wp.transform_get_translation(parent_pose)
        parent_rotation = wp.transform_get_rotation(parent_pose)
        parent_center = wp.quat_rotate_inv(parent_rotation, center - parent_position)
        joint_axis = wp.quat_rotate_inv(parent_rotation, axis)

    joint = builder.add_joint_revolute(
        parent=parent,
        child=body,
        axis=joint_axis,
        parent_xform=wp.transform(parent_center, wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        armature=axle_armature,
        friction=axle_friction,
        label=f"{label}_free_axle" if label else None,
    )

    shapes = add_guided_pulley(
        builder,
        center=wp.vec3(0.0, 0.0, 0.0),
        axis=axis,
        sheave_diameter=sheave_diameter,
        cable_radius=cable_radius,
        body=body,
        color=color,
        groove_width_scale=groove_width_scale,
        flange_radius_scale=flange_radius_scale,
        flange_thickness_scale=flange_thickness_scale,
        ke=ke,
        kd=kd,
        mu=mu,
        density=density,
        label=label,
    )
    marker = add_pulley_rotation_dot(
        builder,
        body=body,
        sheave_diameter=sheave_diameter,
        cable_radius=cable_radius,
        groove_width_scale=groove_width_scale,
        flange_thickness_scale=flange_thickness_scale,
        label=label,
    )

    return body, joint, (*shapes, marker)


def append_segment(points: list[wp.vec3], end: wp.vec3, segment_length: float):
    start = points[-1]
    length = float(wp.length(end - start))
    if length <= 1.0e-8:
        return

    count = max(1, int(math.ceil(length / segment_length)))
    for i in range(1, count + 1):
        u = float(i) / float(count)
        points.append(start * (1.0 - u) + end * u)


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
        append_segment(points, point, segment_length)


def resample_equal_length_segments(points: list[wp.vec3], segment_length: float) -> tuple[list[wp.vec3], float]:
    """Resamples a polyline into globally equal-length segments."""
    if len(points) < 2:
        raise ValueError("points must contain at least two points")
    if segment_length <= 0.0:
        raise ValueError("segment_length must be positive")

    clean_points = [points[0]]
    distances = [0.0]
    total_length = 0.0
    for point in points[1:]:
        length = float(wp.length(point - clean_points[-1]))
        if length <= 1.0e-8:
            continue
        total_length += length
        clean_points.append(point)
        distances.append(total_length)

    if total_length <= 1.0e-8:
        raise ValueError("points must span a non-zero length")

    segment_count = max(2, int(math.ceil(total_length / segment_length)))
    equal_segment_length = total_length / float(segment_count)
    resampled = [clean_points[0]]

    point_index = 1
    for segment_index in range(1, segment_count):
        target_distance = equal_segment_length * float(segment_index)
        while point_index < len(clean_points) - 1 and distances[point_index] < target_distance:
            point_index += 1

        previous_distance = distances[point_index - 1]
        next_distance = distances[point_index]
        u = (target_distance - previous_distance) / (next_distance - previous_distance)
        resampled.append(clean_points[point_index - 1] * (1.0 - u) + clean_points[point_index] * u)

    resampled.append(clean_points[-1])
    return resampled, equal_segment_length


def create_xy_table_cable_points(
    start: wp.vec3,
    pulley_centers: list[wp.vec3],
    pulley_radii: list[float],
    end: wp.vec3,
    cable_radius: float,
    segment_length: float,
    wrap_clearance_scale: float = 1.1,
) -> list[wp.vec3]:
    """Creates the cable route with straight tangent spans."""
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

    points = [start]
    wrap_clearance = wrap_clearance_scale * cable_radius

    # Green pulleys use their inner quadrants. Blue input pulleys and the
    # beige top pulley use the outside path.
    for center, radius, (start_angle, end_angle, direction) in zip(
        pulley_centers,
        pulley_radii,
        pulley_arcs,
        strict=True,
    ):
        append_arc_xy(
            points,
            center,
            radius + wrap_clearance,
            start_angle,
            end_angle,
            segment_length,
            direction=direction,
        )
    append_segment(points, end, segment_length)
    return points


def add_visual_bar(
    builder: newton.ModelBuilder,
    *,
    body: int,
    center: wp.vec3,
    half_extents: tuple[float, float, float],
    color: tuple[float, float, float],
    label: str,
    density: float = 350.0,
):
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
        self.viewer = viewer

        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        cable_radius = 0.003
        self.input_pulley_radius = 0.025
        green_sheave_radius = 0.015
        beige_sheave_radius = 0.025
        segment_length = 0.015
        cable_wrap_clearance_scale = 1.0

        blue = (0.12, 0.34, 0.76)
        green = (0.12, 0.58, 0.28)
        beige = (0.74, 0.63, 0.45)

        base_z = 0.006
        slide_z = 0.014
        table_z = 0.022
        pulley_z = 0.046

        builder = newton.ModelBuilder()
        builder.rigid_gap = 5.0 * cable_radius
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.mu = 1.0

        base_origin = wp.vec3(0.0, 0.0, base_z)
        slide_origin = wp.vec3(0.0, 0.0, slide_z)
        table_origin = wp.vec3(0.0, 0.0, table_z)

        add_visual_bar(
            builder,
            body=-1,
            center=base_origin,
            half_extents=(0.205, 0.025, 0.006),
            color=blue,
            label="fixed_blue_base",
            density=0.0,
        )

        self.slide_body = builder.add_link(
            xform=wp.transform(slide_origin, wp.quat_identity()),
            label="green_x_slide",
        )
        self.table_body = builder.add_link(
            xform=wp.transform(table_origin, wp.quat_identity()),
            label="beige_y_table",
        )
        self.table_origin_xy = (float(table_origin[0]), float(table_origin[1]))
        self.table_rect_points = np.array(TABLE_RECT_POINTS, dtype=np.float32)
        self.table_rect_min_distances = np.full(len(TABLE_RECT_POINTS), np.inf, dtype=np.float32)

        slide_joint = builder.add_joint_prismatic(
            parent=-1,
            child=self.slide_body,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(slide_origin, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            limit_lower=-0.07,
            limit_upper=0.07,
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
            limit_lower=-0.08,
            limit_upper=0.08,
            limit_ke=2.0e3,
            limit_kd=1.0e-4,
            friction=0.0,
            label="beige_y_table_axis",
        )
        table_articulation_joints = [slide_joint, table_joint]

        add_visual_bar(
            builder,
            body=self.slide_body,
            center=wp.vec3(0.0, 0.0, 0.0),
            half_extents=(0.085, 0.052, 0.006),
            color=green,
            label="green_horizontal_carriage",
            density=45.0,
        )
        add_visual_bar(
            builder,
            body=self.table_body,
            center=wp.vec3(0.0, 0.0, 0.0),
            half_extents=(0.013, 0.215, 0.006),
            color=beige,
            label="beige_vertical_carriage",
            density=65.0,
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

        pulley_specs = [
            (
                "green_lower_left",
                ATTACH_SLIDE,
                MOTOR_NONE,
                green,
                wp.vec3(-0.045, -0.045, pulley_z),
                green_sheave_radius,
            ),
            (
                "blue_input_left",
                ATTACH_BASE,
                MOTOR_INPUT_LEFT,
                blue,
                wp.vec3(-0.19, 0.0, pulley_z),
                self.input_pulley_radius,
            ),
            (
                "green_upper_left",
                ATTACH_SLIDE,
                MOTOR_NONE,
                green,
                wp.vec3(-0.045, 0.045, pulley_z),
                green_sheave_radius,
            ),
            (
                "beige_top",
                ATTACH_TABLE,
                MOTOR_NONE,
                beige,
                wp.vec3(0.0, 0.19, pulley_z),
                beige_sheave_radius,
            ),
            (
                "green_upper_right",
                ATTACH_SLIDE,
                MOTOR_NONE,
                green,
                wp.vec3(0.045, 0.045, pulley_z),
                green_sheave_radius,
            ),
            (
                "blue_input_right",
                ATTACH_BASE,
                MOTOR_INPUT_RIGHT,
                blue,
                wp.vec3(0.19, 0.0, pulley_z),
                self.input_pulley_radius,
            ),
            (
                "green_lower_right",
                ATTACH_SLIDE,
                MOTOR_NONE,
                green,
                wp.vec3(0.045, -0.045, pulley_z),
                green_sheave_radius,
            ),
        ]

        self.pulley_bodies: list[int] = []
        pulley_centers = [spec[4] for spec in pulley_specs]
        pulley_radii = [spec[5] for spec in pulley_specs]

        for i, (label, attach, _motor, color, center, sheave_radius) in enumerate(pulley_specs, start=1):
            pulley_kwargs = {
                "center": center,
                "axis": wp.vec3(0.0, 0.0, 1.0),
                "sheave_diameter": 2.0 * sheave_radius,
                "cable_radius": cable_radius,
                "color": color,
                "groove_width_scale": 3.1,
                "flange_radius_scale": 3.2,
                "flange_thickness_scale": 1.2,
                "ke": 1.0e5,
                "kd": 0.0,
                "mu": 1.0,
                "density": 220.0,
                "label": f"xy_table_{i}_{label}",
            }
            if attach == ATTACH_BASE:
                pulley_body, _ = add_kinematic_guided_pulley(builder, **pulley_kwargs)
            else:
                parent = self.slide_body if attach == ATTACH_SLIDE else self.table_body
                pulley_body, pulley_joint, _ = add_passive_guided_pulley(
                    builder,
                    parent=parent,
                    axle_armature=1.0e-4,
                    axle_friction=0.0,
                    **pulley_kwargs,
                )
                table_articulation_joints.append(pulley_joint)
            self.pulley_bodies.append(pulley_body)

        self.left_anchor_local = wp.vec3(-0.028, -0.21, pulley_z - table_z)
        self.right_anchor_local = wp.vec3(0.028, -0.21, pulley_z - table_z)
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

        cable_route_points = create_xy_table_cable_points(
            start=left_anchor_world,
            pulley_centers=pulley_centers,
            pulley_radii=pulley_radii,
            end=right_anchor_world,
            cable_radius=cable_radius,
            segment_length=segment_length,
            wrap_clearance_scale=cable_wrap_clearance_scale,
        )
        cable_points, cable_segment_length = resample_equal_length_segments(cable_route_points, segment_length)
        cable_quats = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
        cable_segment_count = len(cable_points) - 1
        straight_cable_points, straight_cable_quats = newton.utils.create_straight_cable_points_and_quaternions(
            start=left_anchor_world,
            direction=wp.vec3(1.0, 0.0, 0.0),
            length=cable_segment_count * cable_segment_length,
            num_segments=cable_segment_count,
        )

        cable_cfg = builder.default_shape_cfg.copy()
        cable_cfg.density = 20.0
        cable_cfg.gap = 5.0 * cable_radius

        self.cable_bodies, cable_joints = builder.add_rod(
            positions=straight_cable_points,
            quaternions=straight_cable_quats,
            radius=cable_radius,
            cfg=cable_cfg,
            stretch_stiffness=1.0e5,
            stretch_damping=0.0,
            bend_stiffness=5.0e-5,
            bend_damping=1.0e-2,
            wrap_in_articulation=False,
            label="xy_table_cable",
        )
        initial_cable_xforms = [wp.transform(cable_points[i], cable_quats[i]) for i in range(len(self.cable_bodies))]
        filter_body_group_collisions(builder, self.cable_bodies)

        first_cable_body = self.cable_bodies[0]
        last_cable_body = self.cable_bodies[-1]
        last_segment_length = cable_segment_length
        for i, (body, xform) in enumerate(
            (
                (first_cable_body, wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())),
                (last_cable_body, wp.transform(wp.vec3(0.0, 0.0, last_segment_length), wp.quat_identity())),
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
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            armature=1.0e-5,
            friction=0.0,
            label="left_bottom_cable_fix",
        )
        builder.add_joint_ball(
            parent=self.table_body,
            child=last_cable_body,
            parent_xform=wp.transform(self.right_anchor_local, wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, last_segment_length), wp.quat_identity()),
            armature=1.0e-5,
            friction=0.0,
            label="right_bottom_cable_fix_loop",
        )
        builder.add_articulation(
            [*table_articulation_joints, *cable_joints, left_anchor_joint],
            label="xy_table_cable_cross_slide",
        )

        kinematic_body_indices = [
            body for body, spec in zip(self.pulley_bodies, pulley_specs, strict=True) if spec[2] != MOTOR_NONE
        ]
        kinematic_body_motor = [spec[2] for spec in pulley_specs if spec[2] != MOTOR_NONE]
        kinematic_body_base_xforms = [builder.body_q[body] for body in kinematic_body_indices]

        builder.add_ground_plane()
        builder.color(balance_colors=False)

        sim_device = wp.get_device(args.device) if args.device else None
        self.model = builder.finalize(device=sim_device)
        self.model.set_gravity((0.0, 0.0, 0.0))

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=sim_iterations,
            friction_epsilon=1.0e-2,
            rigid_body_contact_buffer_size=256,
            rigid_contact_hard=False,
            rigid_joint_linear_ke=1.0e5,
            rigid_joint_angular_ke=1.0e5,
            rigid_joint_linear_kd=0.0,
            rigid_joint_angular_kd=0.0,
            rigid_avbd_gamma=0.999,
        )
        self.solver.rigid_contact_stick_motion_eps = 0.0
        self.solver.rigid_contact_stick_freeze_translation_eps = 0.0
        self.solver.rigid_contact_stick_freeze_angular_eps = 0.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

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
        self.kinematic_body_motor = wp.array(
            kinematic_body_motor,
            dtype=wp.int32,
            device=self.model.device,
        )
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
        # The wrapped cable pose is the initial condition, not a one-frame
        # teleport. Keep VBD's previous-pose buffer in sync to avoid a fake
        # first-step velocity.
        self.solver.body_q_prev = wp.clone(self.state_0.body_q, device=self.solver.device)
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.0, 0.0, 0.8),
            pitch=-90.0,
            yaw=90.0,
        )

        self.capture()

    def capture(self):
        if self.solver.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                drive_input_pulleys,
                dim=self.kinematic_body_indices.shape[0],
                inputs=[
                    self.sim_time_wp,
                    self.kinematic_body_indices,
                    self.kinematic_body_base_xforms,
                    self.kinematic_body_motor,
                    self.input_pulley_radius,
                    self.state_0.body_q,
                    self.state_1.body_q,
                ],
                device=self.model.device,
            )

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            wp.launch(advance_time, dim=1, inputs=[self.sim_time_wp, self.sim_dt], device=self.model.device)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.record_diagrams()

    def record_diagrams(self):
        q2, q6 = compute_input_pulley_rotations(self.sim_time, self.input_pulley_radius)
        body_q = self.state_0.body_q.numpy()
        table_pos = body_q[self.table_body, 0:3]
        table_x = float(table_pos[0]) - self.table_origin_xy[0]
        table_y = float(table_pos[1]) - self.table_origin_xy[1]
        table_xy = np.array((table_x, table_y), dtype=np.float32)
        table_rect_distances = np.linalg.norm(self.table_rect_points - table_xy, axis=1)
        self.table_rect_min_distances = np.minimum(self.table_rect_min_distances, table_rect_distances)

        self.viewer.log_scalar("Blue pulley 2 rotation [rad]", q2)
        self.viewer.log_scalar("Blue pulley 6 rotation [rad]", q6)
        self.viewer.log_scalar("Beige table X position [m]", table_x)
        self.viewer.log_scalar("Beige table Y position [m]", table_y)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        if self.state_0.body_q is None:
            raise RuntimeError("Body state is not available.")

        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("NaN/Inf in body transforms.")

        cable_pos = body_q[[int(body) for body in self.cable_bodies], 0:3]
        if np.min(cable_pos[:, 2]) < -0.04:
            raise ValueError("Cable fell below the ground plane.")
        if np.max(cable_pos[:, 2]) > 0.12:
            raise ValueError("Cable lifted too far from the ground-plane table.")
        if np.max(np.abs(cable_pos[:, 0])) > 0.34 or np.max(np.abs(cable_pos[:, 1])) > 0.34:
            raise ValueError("Cable moved outside the expected XY table bounds.")

        slide_pos = body_q[self.slide_body, 0:3]
        table_pos = body_q[self.table_body, 0:3]
        joint_limit_tolerance = 0.005
        if not (-0.07 - joint_limit_tolerance <= slide_pos[0] <= 0.07 + joint_limit_tolerance):
            raise ValueError("Horizontal green carriage moved outside its travel range.")
        if not (-0.08 - joint_limit_tolerance <= table_pos[1] <= 0.08 + joint_limit_tolerance):
            raise ValueError("Vertical beige carriage moved outside its travel range.")
        if not (0.0 <= slide_pos[2] <= 0.04 and 0.0 <= table_pos[2] <= 0.05):
            raise ValueError("Table bodies left the ground-plane layout.")

        missed_points = np.nonzero(self.table_rect_min_distances > TABLE_RECT_HIT_TOLERANCE)[0]
        if len(missed_points) > 0:
            details = []
            for point_index in missed_points:
                point = self.table_rect_points[point_index]
                distance = self.table_rect_min_distances[point_index]
                details.append(f"({point[0]:.3f}, {point[1]:.3f}) min error {distance:.4f} m")
            raise ValueError(
                "XY table did not hit every rectangle point within "
                f"{TABLE_RECT_HIT_TOLERANCE:.3f} m: {', '.join(details)}"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=TABLE_RECT_TEST_FRAMES)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
