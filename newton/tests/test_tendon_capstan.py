# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for massless routed tendons.

These tests cover route geometry, stretch response, capstan friction, material
transfer, and dynamic roller activation.
"""

import math
import unittest
from itertools import pairwise

import numpy as np
import warp as wp

import newton
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkFlags, TendonLinkType
from newton._src.solvers.tendon_kernels import tangent_point_circle, tendon_segment_length_rate
from newton._src.solvers.xpbd.tendon_kernels import solve_tendon_slip, solve_tendon_stretch
from newton.examples.cable.example_tendon_mujoco_switch import Example as MujocoSwitchExample
from newton.tests.unittest_utils import sanitize_identifier


@wp.kernel
def _measure_tendon_segment_length_rates(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    tendon_link_body: wp.array[int],
    tendon_link_type: wp.array[int],
    tendon_link_offset: wp.array[wp.vec3],
    tendon_link_axis: wp.array[wp.vec3],
    seg_active_link_l: wp.array[int],
    seg_active_link_r: wp.array[int],
    seg_attachment_l: wp.array[wp.vec3],
    seg_attachment_r: wp.array[wp.vec3],
    length_rates: wp.array[float],
):
    seg = wp.tid()
    length_rates[seg] = tendon_segment_length_rate(
        body_q,
        body_qd,
        body_com,
        tendon_link_body,
        tendon_link_type,
        tendon_link_offset,
        tendon_link_axis,
        seg_active_link_l[seg],
        seg_active_link_r[seg],
        seg_attachment_l[seg],
        seg_attachment_r[seg],
    )


@wp.kernel
def _evaluate_tangent_point_at_circle_center(
    center: wp.vec3,
    radius: float,
    plane_normal: wp.vec3,
    tangent_point: wp.array[wp.vec3],
):
    tangent_point[0] = tangent_point_circle(center, center, radius, plane_normal, 1)


def add_test(cls, name, devices, test_fn):
    for device in devices:
        test_name = f"test_{sanitize_identifier(name)}_{sanitize_identifier(device)}"
        setattr(cls, test_name, lambda self, d=device, fn=test_fn: fn(self, d))


def _box_on_planar_joint(builder, pos, mass, half_extent, z_limits=None, vertical_only=False):
    dof = newton.ModelBuilder.JointDofConfig
    body = builder.add_link(xform=wp.transform(p=pos), mass=mass)
    builder.add_shape_box(body, hx=half_extent, hy=half_extent, hz=half_extent)
    z_dof = dof(axis=Axis.Z)
    if z_limits is not None:
        z_dof = dof(axis=Axis.Z, limit_lower=z_limits[0], limit_upper=z_limits[1])
    linear_axes = [z_dof] if vertical_only else [dof(axis=Axis.X), z_dof]
    joint = builder.add_joint_d6(
        parent=-1,
        child=body,
        linear_axes=linear_axes,
        angular_axes=[dof(axis=Axis.Y)],
        parent_xform=wp.transform(p=pos),
        child_xform=wp.transform(),
    )
    builder.add_articulation([joint])
    return body


def build_pinhole_atwood(mass_left=1.0, mass_right=3.0, mu=0.0, compliance=1.0e-5):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, -9.81))

    pin = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 3.5)),
        mass=0.0,
        is_kinematic=True,
    )
    builder.add_shape_sphere(pin, radius=0.035)

    left = _box_on_planar_joint(builder, wp.vec3(-0.45, 0.0, 2.0), mass_left, 0.06)
    right = _box_on_planar_joint(builder, wp.vec3(0.45, 0.0, 2.0), mass_right, 0.06)

    axis = (0.0, 1.0, 0.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
    )
    builder.add_tendon_link(
        body=pin,
        link_type=int(TendonLinkType.PINHOLE),
        mu=mu,
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=compliance,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
        compliance=compliance,
        damping=0.1,
        rest_length=-1.0,
    )

    builder.add_ground_plane()
    return builder.finalize(), left, right


def build_stiff_pinhole_capstan(n_pinholes=9, mu=0.1, compliance=2.0e-8):
    """Build a stiff cable around a fixed pulley using frictional pinholes.

    The 180-degree wrap has a fixed anchor on one side and a force-driven slider on the other.
    Per-segment stretch is about 1e-7 m, near the float32 resolution of the segment lengths.
    """
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
    r, drop, axis = 0.005, 0.15, (0.0, 1.0, 0.0)

    anchor = builder.add_body(xform=wp.transform(p=wp.vec3(-r, 0.0, -drop)), mass=0.0, is_kinematic=True)
    builder.add_shape_sphere(anchor, radius=r * 0.15, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    pulley = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)), mass=0.0, is_kinematic=True)
    slider_inertia = wp.mat33(1.0e-8, 0.0, 0.0, 0.0, 1.0e-8, 0.0, 0.0, 0.0, 1.0e-8)
    slider = builder.add_link(
        xform=wp.transform(p=wp.vec3(r, 0.0, -drop)),
        mass=0.01,
        inertia=slider_inertia,
        lock_inertia=True,
    )
    builder.add_shape_sphere(slider, radius=r * 0.15, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    joint = builder.add_joint_d6(
        parent=-1,
        child=slider,
        linear_axes=[newton.ModelBuilder.JointDofConfig(axis=Axis.Z)],
        angular_axes=[],
        parent_xform=wp.transform(p=wp.vec3(r, 0.0, -drop)),
        child_xform=wp.transform(),
    )
    builder.add_articulation([joint])

    builder.add_tendon()
    builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT), offset=(0.0, 0.0, 0.0), axis=axis)
    for i in range(n_pinholes):
        a = math.pi - i * math.pi / (n_pinholes - 1)  # pi..0 over the top of the pulley
        builder.add_tendon_link(
            body=pulley,
            link_type=int(TendonLinkType.PINHOLE),
            mu=mu,
            offset=(r * math.cos(a), 0.0, r * math.sin(a)),
            axis=axis,
            compliance=compliance,
            damping=0.0,
            rest_length=-1.0,
        )
    builder.add_tendon_link(
        body=slider,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=compliance,
        damping=0.0,
        rest_length=-1.0,
    )
    return builder.finalize(), slider


def build_slack_pinhole_route():
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))

    left = builder.add_body(xform=wp.transform(p=wp.vec3(-1.0, 0.0, 0.0)), mass=0.0, is_kinematic=True)
    pin = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)), mass=0.0, is_kinematic=True)
    right = builder.add_body(xform=wp.transform(p=wp.vec3(2.0, 0.0, 0.0)), mass=0.0, is_kinematic=True)
    for body in (left, pin, right):
        builder.add_shape_sphere(body, radius=0.01)

    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
    )
    builder.add_tendon_link(
        body=pin,
        link_type=int(TendonLinkType.PINHOLE),
        offset=(0.0, 0.0, 0.0),
        compliance=1.0e-6,
        rest_length=3.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        compliance=1.0e-6,
        rest_length=3.0,
    )

    return builder.finalize()


def build_frictionless_zero_span_route(compliance=1.0e-3, mu=0.0, points=None, rest_lengths=None):
    """Build a static route where one exhausted middle span separates tension plateaus."""
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))

    # The duplicate middle pinholes make segment 2 have zero geometric length.
    # A frictionless continuous cable should still equalize tension across the
    # remaining nonzero spans instead of treating the zero span as a barrier.
    if points is None:
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
        ]
    if rest_lengths is None:
        rest_lengths = [0.5, 0.5, 1.0e-6, 0.9, 0.9]

    bodies = []
    for point in points:
        body = builder.add_body(xform=wp.transform(p=wp.vec3(*point)), mass=0.0, is_kinematic=True)
        builder.add_shape_sphere(body, radius=0.01)
        bodies.append(body)

    builder.add_tendon()
    builder.add_tendon_link(
        body=bodies[0],
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )
    for body, rest in zip(bodies[1:-1], rest_lengths[:-1], strict=True):
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.PINHOLE),
            mu=mu,
            offset=(0.0, 0.0, 0.0),
            axis=(0.0, 1.0, 0.0),
            compliance=compliance,
            damping=0.0,
            rest_length=rest,
        )
    builder.add_tendon_link(
        body=bodies[-1],
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        compliance=compliance,
        damping=0.0,
        rest_length=rest_lengths[-1],
    )

    return builder.finalize(), np.asarray(rest_lengths, dtype=np.float32), compliance


def build_dynamic_pulley_atwood(
    mu=10.0,
    mass_left=1.0,
    mass_right=3.0,
    pulley_mass=5.0,
    pulley_radius=0.15,
    compliance=1.0e-5,
):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, -9.81))

    pulley_pos = wp.vec3(0.0, 0.0, 3.5)
    pulley_half_height = 0.04
    inertia_y = 0.5 * pulley_mass * pulley_radius * pulley_radius
    inertia_xz = (1.0 / 12.0) * pulley_mass * (3.0 * pulley_radius * pulley_radius + (2.0 * pulley_half_height) ** 2)
    inertia = wp.mat33(
        inertia_xz,
        0.0,
        0.0,
        0.0,
        inertia_y,
        0.0,
        0.0,
        0.0,
        inertia_xz,
    )
    pulley = builder.add_link(
        xform=wp.transform(p=pulley_pos),
        mass=pulley_mass,
        inertia=inertia,
        lock_inertia=True,
    )
    q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
    builder.add_shape_cylinder(
        pulley,
        xform=wp.transform(q=q_cyl),
        radius=pulley_radius,
        half_height=pulley_half_height,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )
    j_pulley = builder.add_joint_revolute(
        parent=-1,
        child=pulley,
        axis=Axis.Y,
        parent_xform=wp.transform(p=pulley_pos),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j_pulley])

    z_limits = (-1.25, 1.25)
    left = _box_on_planar_joint(builder, wp.vec3(-0.4, 0.0, 2.0), mass_left, 0.06, z_limits, vertical_only=True)
    right = _box_on_planar_joint(builder, wp.vec3(0.4, 0.0, 2.0), mass_right, 0.06, z_limits, vertical_only=True)

    axis = (0.0, 1.0, 0.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
    )
    builder.add_tendon_link(
        body=pulley,
        link_type=int(TendonLinkType.ROLLING),
        radius=pulley_radius,
        orientation=1,
        mu=mu,
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=compliance,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
        compliance=compliance,
        damping=0.1,
        rest_length=-1.0,
    )

    builder.add_ground_plane()
    return builder.finalize(), left, right, pulley


def build_kinematic_pulley_atwood(mu=0.0, mass_left=1.0, mass_right=3.0, pulley_radius=0.15, compliance=1.0e-5):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, -9.81))

    pulley_pos = wp.vec3(0.0, 0.0, 3.5)
    pulley = builder.add_body(
        xform=wp.transform(p=pulley_pos),
        mass=0.0,
        is_kinematic=True,
    )
    q_cyl = wp.quat(np.sin(np.pi / 4.0), 0.0, 0.0, np.cos(np.pi / 4.0))
    builder.add_shape_cylinder(
        pulley,
        xform=wp.transform(q=q_cyl),
        radius=pulley_radius,
        half_height=0.04,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
    )

    z_limits = (-1.25, 1.25)
    left = _box_on_planar_joint(builder, wp.vec3(-0.4, 0.0, 2.0), mass_left, 0.06, z_limits, vertical_only=True)
    right = _box_on_planar_joint(builder, wp.vec3(0.4, 0.0, 2.0), mass_right, 0.06, z_limits, vertical_only=True)

    axis = (0.0, 1.0, 0.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
    )
    builder.add_tendon_link(
        body=pulley,
        link_type=int(TendonLinkType.ROLLING),
        radius=pulley_radius,
        orientation=1,
        mu=mu,
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=compliance,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
        compliance=compliance,
        damping=0.1,
        rest_length=-1.0,
    )

    builder.add_ground_plane()
    return builder.finalize(), left, right, pulley


def build_kinematic_capstan_hysteresis(mu=0.2, pulley_radius=0.2, compliance=1.0e-3):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))

    # Endpoints sit just outside the pulley radius and below the pulley, giving
    # a near-pi wrap angle while still using the finite-radius rolling path.
    left = builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.21, 0.0, -2.0)),
        mass=0.0,
        is_kinematic=True,
    )
    pulley = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )
    right = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.21, 0.0, -2.0)),
        mass=0.0,
        is_kinematic=True,
    )

    for body in (left, pulley, right):
        builder.add_shape_sphere(body, radius=0.01)

    axis = (0.0, 1.0, 0.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=axis,
    )
    builder.add_tendon_link(
        body=pulley,
        link_type=int(TendonLinkType.ROLLING),
        radius=pulley_radius,
        orientation=1,
        mu=mu,
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=compliance,
        damping=0.0,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=compliance,
        damping=0.0,
        rest_length=-1.0,
    )

    return builder.finalize(), left, pulley, right


def build_pinhole_capstan_force_mode(num_pinholes=5, mu=0.2, mass=20.0, pulley_radius=0.2, compliance=1.0e-3):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, -9.81))
    endpoint_x = 0.21
    endpoint_z = -2.0

    left = builder.add_body(
        xform=wp.transform(p=wp.vec3(-endpoint_x, 0.0, endpoint_z)),
        mass=0.0,
        is_kinematic=True,
    )
    pulley = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )

    inertia = 0.4 * mass * 0.03 * 0.03
    right = builder.add_link(
        xform=wp.transform(p=wp.vec3(endpoint_x, 0.0, endpoint_z)),
        mass=mass,
        inertia=wp.mat33(
            inertia,
            0.0,
            0.0,
            0.0,
            inertia,
            0.0,
            0.0,
            0.0,
            inertia,
        ),
        lock_inertia=True,
    )
    dof = newton.ModelBuilder.JointDofConfig
    j_right = builder.add_joint_d6(
        parent=-1,
        child=right,
        linear_axes=[dof(axis=Axis.Z)],
        angular_axes=[],
        parent_xform=wp.transform(p=wp.vec3(endpoint_x, 0.0, endpoint_z)),
        child_xform=wp.transform(),
    )
    builder.add_articulation([j_right])

    seg_compliance = 2.0 * compliance / (num_pinholes + 1)
    axis = (0.0, 1.0, 0.0)
    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=axis,
    )
    for i in range(num_pinholes):
        alpha = np.pi - i * np.pi / (num_pinholes - 1)
        builder.add_tendon_link(
            body=pulley,
            link_type=int(TendonLinkType.PINHOLE),
            mu=mu,
            offset=(pulley_radius * np.cos(alpha), 0.0, pulley_radius * np.sin(alpha)),
            axis=axis,
            compliance=seg_compliance,
            damping=0.0,
            rest_length=-1.0,
        )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=axis,
        compliance=seg_compliance,
        damping=0.0,
        rest_length=-1.0,
    )

    return builder.finalize(), right, mass, endpoint_z


def build_simple_cable_gravity(mass=10.0, compliance=1.0e-3, initial_z=-0.1):
    """Build a one-segment cable from a fixed anchor to a hanging prismatic mass."""
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, -9.81))

    anchor = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )
    builder.add_shape_sphere(anchor, radius=0.01)

    inertia = 0.4 * mass * 0.02 * 0.02
    body = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, initial_z)),
        mass=mass,
        inertia=wp.mat33(
            inertia,
            0.0,
            0.0,
            0.0,
            inertia,
            0.0,
            0.0,
            0.0,
            inertia,
        ),
        lock_inertia=True,
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02, cfg=cfg)

    dof = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        parent=-1,
        child=body,
        linear_axes=[dof(axis=Axis.Z)],
        angular_axes=[],
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, initial_z)),
        child_xform=wp.transform(),
    )
    builder.add_articulation([joint])

    builder.add_tendon()
    builder.add_tendon_link(
        body=anchor,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
    )
    builder.add_tendon_link(
        body=body,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        compliance=compliance,
        damping=0.0,
        rest_length=-1.0,
    )

    return builder.finalize(), body, mass, compliance, initial_z


def build_motorized_pulley_drive(mu=0.0):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
    dof = newton.ModelBuilder.JointDofConfig

    slider = builder.add_link(xform=wp.transform(p=wp.vec3(-0.4, 0.0, 0.0)), mass=1.0)
    builder.add_shape_box(slider, hx=0.03, hy=0.03, hz=0.03)

    anchor = builder.add_link(xform=wp.transform(p=wp.vec3(0.4, 0.0, 0.0)), mass=0.0)
    builder.add_shape_sphere(anchor, radius=0.02)

    radius = 0.1
    pulley_mass = 0.1
    pulley_half_height = 0.02
    inertia_z = 0.5 * pulley_mass * radius * radius
    inertia_xy = (1.0 / 12.0) * pulley_mass * (3.0 * radius * radius + (2.0 * pulley_half_height) ** 2)
    inertia = wp.mat33(
        inertia_xy,
        0.0,
        0.0,
        0.0,
        inertia_xy,
        0.0,
        0.0,
        0.0,
        inertia_z,
    )
    pulley = builder.add_link(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)),
        mass=pulley_mass,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_cylinder(
        pulley,
        radius=radius,
        half_height=pulley_half_height,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
    )

    j_slider = builder.add_joint_d6(
        parent=-1,
        child=slider,
        linear_axes=[dof(axis=Axis.X)],
        parent_xform=wp.transform(p=wp.vec3(-0.4, 0.0, 0.0)),
        child_xform=wp.transform(),
    )
    j_anchor = builder.add_joint_fixed(
        parent=-1,
        child=anchor,
        parent_xform=wp.transform(p=wp.vec3(0.4, 0.0, 0.0)),
        child_xform=wp.transform(),
    )
    j_pulley = builder.add_joint_revolute(
        parent=-1,
        child=pulley,
        axis=Axis.Z,
        parent_xform=wp.transform(),
        child_xform=wp.transform(),
        target_ke=1000.0,
        target_kd=100.0,
        effort_limit=1000.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([j_slider])
    builder.add_articulation([j_anchor])
    builder.add_articulation([j_pulley])

    builder.add_tendon()
    for body, link_type, link_radius in [
        (slider, TendonLinkType.ATTACHMENT, 0.0),
        (pulley, TendonLinkType.ROLLING, radius),
        (anchor, TendonLinkType.ATTACHMENT, 0.0),
    ]:
        builder.add_tendon_link(
            body=body,
            link_type=int(link_type),
            radius=link_radius,
            orientation=1,
            mu=mu,
            offset=(0.0, 0.0, 0.0),
            axis=(0.0, 0.0, 1.0),
            compliance=1.0e-6,
            damping=0.01,
            rest_length=-1.0,
        )

    return builder.finalize(), slider, pulley, j_pulley


def build_kinematic_rolling_transport(mu=10.0):
    """Build a fixed anchor - rolling pulley - fixed anchor route for prescribed spin tests."""
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))

    left = builder.add_body(xform=wp.transform(p=wp.vec3(-0.4, 0.0, 0.0)), mass=0.0, is_kinematic=True)
    pulley = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)), mass=0.0, is_kinematic=True)
    right = builder.add_body(xform=wp.transform(p=wp.vec3(0.4, 0.0, 0.0)), mass=0.0, is_kinematic=True)
    for body in (left, pulley, right):
        builder.add_shape_sphere(body, radius=0.01)

    builder.add_tendon()
    builder.add_tendon_link(
        body=left,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
    )
    builder.add_tendon_link(
        body=pulley,
        link_type=int(TendonLinkType.ROLLING),
        radius=0.1,
        orientation=1,
        mu=mu,
        offset=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        compliance=1.0e-6,
        damping=0.0,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        compliance=1.0e-6,
        damping=0.0,
        rest_length=-1.0,
    )

    return builder.finalize(), pulley


def run_model(model, num_frames=80, substeps=12, fps=60, return_solver=False):
    dt = 1.0 / fps / substeps
    solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=0.8)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    if return_solver:
        return state_0, solver
    return state_0


def build_force_driven_dynamic_route(device):
    """Build an initially straight tendon with a force-driven optional rolling link."""
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
    dof = newton.ModelBuilder.JointDofConfig

    lower = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5)),
        mass=0.0,
        is_kinematic=True,
    )
    upper = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)),
        mass=0.0,
        is_kinematic=True,
    )
    initial_candidate_position = wp.vec3(0.25, 0.0, 0.0)
    candidate = builder.add_link(
        xform=wp.transform(p=initial_candidate_position),
        mass=1.0,
        inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
        lock_inertia=True,
    )
    joint = builder.add_joint_d6(
        parent=-1,
        child=candidate,
        linear_axes=[dof(axis=Axis.X)],
        parent_xform=wp.transform(p=initial_candidate_position),
        child_xform=wp.transform(),
    )
    builder.add_articulation([joint])

    builder.add_tendon()
    builder.add_tendon_link(
        body=lower,
        link_type=int(TendonLinkType.ATTACHMENT),
        axis=(0.0, 1.0, 0.0),
    )
    candidate_link = builder.add_tendon_link(
        body=candidate,
        link_type=int(TendonLinkType.ROLLING),
        radius=0.1,
        orientation=1,
        dynamic=True,
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-2,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=upper,
        link_type=int(TendonLinkType.ATTACHMENT),
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-2,
        rest_length=-1.0,
    )
    return builder.finalize(device=device), candidate, candidate_link


def build_oriented_dynamic_route(orientation, device):
    """Build a straight tendon with a kinematic rolling candidate."""
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))

    lower = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5)),
        mass=0.0,
        is_kinematic=True,
    )
    upper = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)),
        mass=0.0,
        is_kinematic=True,
    )
    candidate = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.25 * orientation, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )

    builder.add_tendon()
    builder.add_tendon_link(
        body=lower,
        link_type=int(TendonLinkType.ATTACHMENT),
        axis=(0.0, 1.0, 0.0),
    )
    candidate_link = builder.add_tendon_link(
        body=candidate,
        link_type=int(TendonLinkType.ROLLING),
        radius=0.1,
        orientation=orientation,
        dynamic=True,
        axis=(0.0, 1.0, 0.0),
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=upper,
        link_type=int(TendonLinkType.ATTACHMENT),
        axis=(0.0, 1.0, 0.0),
        rest_length=-1.0,
    )
    return builder.finalize(device=device), candidate, candidate_link


def build_explicit_inactive_dynamic_route(device, rest_lengths=(0.4, 0.6)):
    """Build an inactive dynamic roller with explicitly authored adjacent rests."""
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
    lower = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.5)),
        mass=0.0,
        is_kinematic=True,
    )
    candidate = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.25, 0.0, 0.0)),
        mass=0.0,
        is_kinematic=True,
    )
    upper = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.5)),
        mass=0.0,
        is_kinematic=True,
    )

    builder.add_tendon()
    builder.add_tendon_link(
        body=lower,
        link_type=TendonLinkType.ATTACHMENT,
        axis=(0.0, 1.0, 0.0),
    )
    candidate_link = builder.add_tendon_link(
        body=candidate,
        link_type=TendonLinkType.ROLLING,
        radius=0.1,
        orientation=1,
        dynamic=True,
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-3,
        rest_length=rest_lengths[0],
    )
    builder.add_tendon_link(
        body=upper,
        link_type=TendonLinkType.ATTACHMENT,
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-3,
        rest_length=rest_lengths[1],
    )
    return builder.finalize(device=device), candidate_link


def build_dynamic_route_neighbor_matrix_case(device, left_type, right_type, orientation):
    """Build an inactive dynamic roller between the requested link types."""
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
    base = builder.add_body(mass=0.0, is_kinematic=True)
    inactive_position = (0.05, 0.03 * orientation, 0.0)
    active_position = (0.05, 0.0025 * orientation, 0.0)
    candidate = builder.add_body(
        xform=wp.transform(p=wp.vec3(inactive_position)),
        mass=0.0,
        is_kinematic=True,
    )

    axis = (0.0, 0.0, 1.0)
    builder.add_tendon()

    def add_link(link_type, x, link_orientation):
        return builder.add_tendon_link(
            body=base,
            link_type=int(link_type),
            radius=0.005 if link_type == TendonLinkType.ROLLING else 0.0,
            orientation=link_orientation,
            mu=0.0,
            offset=(x, 0.0, 0.0),
            axis=axis,
            compliance=5.0e-6,
            damping=5.0,
            rest_length=-1.0,
        )

    if left_type != TendonLinkType.ATTACHMENT:
        add_link(TendonLinkType.ATTACHMENT, 0.0, orientation)
    add_link(left_type, 0.025, -orientation)

    candidate_link = builder.add_tendon_link(
        body=candidate,
        link_type=int(TendonLinkType.ROLLING),
        radius=0.005,
        orientation=orientation,
        mu=0.0,
        dynamic=True,
        axis=axis,
        compliance=5.0e-6,
        damping=5.0,
        rest_length=-1.0,
    )

    add_link(right_type, 0.075, orientation)
    if right_type != TendonLinkType.ATTACHMENT:
        add_link(TendonLinkType.ATTACHMENT, 0.10, orientation)

    return builder.finalize(device=device), candidate, candidate_link, inactive_position, active_position


def build_loaded_dynamic_route(dynamic: bool, device):
    """Build a force-loaded route with an active roller before the optional roller."""
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
    anchor_position = wp.vec3(-0.72, 0.0, 0.015)
    anchor = builder.add_link(
        xform=wp.transform(p=anchor_position),
        mass=0.08,
        inertia=wp.mat33(1.0e-4, 0.0, 0.0, 0.0, 1.0e-4, 0.0, 0.0, 0.0, 1.0e-4),
    )
    anchor_joint = builder.add_joint_prismatic(
        parent=-1,
        child=anchor,
        parent_xform=wp.transform(p=anchor_position),
        child_xform=wp.transform(),
        axis=Axis.X,
        target_ke=0.0,
        target_kd=4.0,
        limit_lower=-0.45,
        limit_upper=0.15,
        limit_ke=8000.0,
        limit_kd=80.0,
        actuator_mode=newton.JointTargetMode.EFFORT,
    )
    builder.add_articulation([anchor_joint])

    guide = builder.add_body(xform=wp.transform(), mass=0.0, is_kinematic=True)
    endpoint = builder.add_body(
        xform=wp.transform(
            p=wp.vec3(0.6 * math.sin(-0.27), 0.0, 0.62 + 0.6 * math.cos(-0.27)),
        ),
        mass=0.0,
        is_kinematic=True,
    )

    builder.add_tendon()
    builder.add_tendon_link(
        body=anchor,
        link_type=int(TendonLinkType.ATTACHMENT),
        axis=(0.0, 1.0, 0.0),
    )
    builder.add_tendon_link(
        body=guide,
        link_type=int(TendonLinkType.ROLLING),
        radius=0.09,
        orientation=-1,
        mu=0.2,
        offset=(0.0, 0.0, 0.05),
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-4,
        damping=0.2,
        rest_length=-1.0,
    )
    candidate = builder.add_tendon_link(
        body=guide,
        link_type=int(TendonLinkType.ROLLING),
        radius=0.085,
        orientation=-1,
        mu=0.2,
        dynamic=dynamic,
        offset=(0.0, 0.0, 0.62),
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-4,
        damping=0.2,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=endpoint,
        link_type=int(TendonLinkType.ATTACHMENT),
        axis=(0.0, 1.0, 0.0),
        compliance=1.0e-4,
        damping=0.2,
        rest_length=-1.0,
    )
    return builder.finalize(device=device), anchor, anchor_joint, candidate


def run_motorized_model(model, drive_joint, target=2.0, num_frames=70, substeps=10, fps=60):
    dt = 1.0 / fps / substeps
    solver = newton.solvers.SolverXPBD(model, iterations=12, joint_linear_relaxation=0.8)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    q_start = int(model.joint_q_start.numpy()[drive_joint])
    for _ in range(num_frames):
        control.joint_target_q[q_start : q_start + 1].fill_(target)
        for _ in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    return state_0


class TestTendonCapstan(unittest.TestCase):
    def test_rejects_link_before_tendon(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=0.0)
        with self.assertRaisesRegex(RuntimeError, "requires add_tendon"):
            builder.add_tendon_link(body=body)

    def test_rejects_invalid_link_parameters(self):
        invalid_parameters = (
            ({"orientation": 0}, "orientation"),
            ({"radius": -1.0}, "radius"),
            ({"mu": -1.0}, "friction coefficient"),
            ({"compliance": -1.0}, "compliance"),
            ({"damping": -1.0}, "damping"),
            ({"offset": (0.0, 0.0)}, "offset"),
            ({"axis": (0.0, 1.0)}, "axis"),
            ({"axis": (0.0, 0.0, 0.0)}, "axis"),
        )
        for parameters, message in invalid_parameters:
            with self.subTest(parameters=parameters):
                builder = newton.ModelBuilder()
                body = builder.add_body(mass=0.0)
                builder.add_tendon()
                with self.assertRaisesRegex(ValueError, message):
                    builder.add_tendon_link(body=body, **parameters)

    def test_normalizes_link_axis(self):
        builder = newton.ModelBuilder()
        bodies = [builder.add_body(mass=0.0) for _ in range(2)]
        builder.add_tendon()
        link = builder.add_tendon_link(body=bodies[0], axis=(0.0, 0.0, 2.0))
        builder.add_tendon_link(body=bodies[1])
        model = builder.finalize()

        np.testing.assert_array_equal(model.tendon_link_axis.numpy()[link], (0.0, 0.0, 1.0))

    def test_tangent_fallback_is_perpendicular_to_x_axis(self):
        center = wp.vec3(1.0, 2.0, 3.0)
        radius = 0.25
        tangent_point = wp.empty(1, dtype=wp.vec3)

        wp.launch(
            _evaluate_tangent_point_at_circle_center,
            dim=1,
            inputs=[center, radius, wp.vec3(1.0, 0.0, 0.0)],
            outputs=[tangent_point],
        )

        offset = tangent_point.numpy()[0] - np.asarray(center)
        self.assertAlmostEqual(float(np.linalg.norm(offset)), radius, places=6)
        self.assertAlmostEqual(float(offset[0]), 0.0, places=6)

    def test_builder_merge_schema_remains_valid(self):
        source = newton.ModelBuilder()
        source.add_body(mass=0.0)

        destination = newton.ModelBuilder()
        destination.add_builder(source)

        self.assertEqual(destination.body_count, 1)

    def test_builder_merge_rejects_tendon_data(self):
        source = newton.ModelBuilder()
        bodies = [source.add_body(mass=0.0) for _ in range(2)]
        source.add_tendon()
        source.add_tendon_link(body=bodies[0])
        source.add_tendon_link(body=bodies[1])

        destination = newton.ModelBuilder()
        with self.assertRaisesRegex(NotImplementedError, "do not yet support tendon data"):
            destination.add_builder(source)

        self.assertEqual(destination.body_count, 0)

    def test_rejects_incomplete_tendon(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=0.0)
        builder.add_tendon()
        builder.add_tendon_link(body=body)
        with self.assertRaisesRegex(ValueError, "at least two links"):
            builder.finalize()

    def test_rejects_terminal_dynamic_link(self):
        builder = newton.ModelBuilder()
        bodies = [builder.add_body(mass=0.0) for _ in range(2)]
        builder.add_tendon()
        builder.add_tendon_link(body=bodies[0])
        builder.add_tendon_link(
            body=bodies[1],
            link_type=TendonLinkType.ROLLING,
            radius=0.1,
            dynamic=True,
        )
        with self.assertRaisesRegex(ValueError, "links on both sides"):
            builder.finalize()

    def test_rejects_differentiable_tendon_model(self):
        builder = newton.ModelBuilder()
        bodies = [builder.add_body(mass=0.0) for _ in range(2)]
        builder.add_tendon()
        builder.add_tendon_link(body=bodies[0])
        builder.add_tendon_link(body=bodies[1])
        model = builder.finalize(requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "not differentiable"):
            newton.solvers.SolverXPBD(model)

    def test_rejects_dynamic_routing_on_non_rolling_link(self):
        for link_type in (TendonLinkType.ATTACHMENT, TendonLinkType.PINHOLE):
            with self.subTest(link_type=link_type):
                builder = newton.ModelBuilder()
                body = builder.add_body(mass=0.0)
                builder.add_tendon()
                with self.assertRaisesRegex(ValueError, "dynamic routing is only supported for ROLLING tendon links"):
                    builder.add_tendon_link(body=body, link_type=int(link_type), dynamic=True)

    def test_rejects_consecutive_dynamic_rolling_links(self):
        builder = newton.ModelBuilder()
        bodies = [builder.add_body(mass=0.0) for _ in range(3)]
        builder.add_tendon()
        builder.add_tendon_link(
            body=bodies[0],
            link_type=int(TendonLinkType.ATTACHMENT),
        )
        builder.add_tendon_link(
            body=bodies[1],
            link_type=int(TendonLinkType.ROLLING),
            radius=0.1,
            dynamic=True,
        )

        with self.assertRaisesRegex(ValueError, "Consecutive dynamic ROLLING tendon links"):
            builder.add_tendon_link(
                body=bodies[2],
                link_type=int(TendonLinkType.ROLLING),
                radius=0.1,
                dynamic=True,
            )


def _hinge_y_angle(body_q, body_idx):
    q = body_q[body_idx]
    return float(2.0 * np.arctan2(float(q[4]), float(q[6])))


def _dynamic_capstan_metrics(device, mu, num_frames=40):
    model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=mu, compliance=3.0e-8)
    substeps = 12
    dt = 1.0 / 60.0 / substeps
    solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=0.8)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    body_q = state_0.body_q.numpy()
    last_angle = _hinge_y_angle(body_q, pulley_idx)
    theta = 0.0

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        angle = _hinge_y_angle(body_q, pulley_idx)
        theta += float((angle - last_angle + np.pi) % (2.0 * np.pi) - np.pi)
        last_angle = angle

    left_travel = float(body_q[left_idx][2]) - 2.0
    right_travel = 2.0 - float(body_q[right_idx][2])
    cable_travel = 0.5 * (left_travel + right_travel)
    radius = float(model.tendon_link_radius.numpy()[1])
    rim_travel = theta * radius
    slip = abs(cable_travel - rim_travel)
    return body_q, left_travel, right_travel, cable_travel, theta, rim_travel, slip


def _kinematic_capstan_metrics(device, mu, num_frames=100, substeps=12, fps=60):
    # The compliant cable is a spring, so the atwood oscillates and any single-frame travel (or
    # tension ratio) is phase-dependent. The robust slip-vs-lock signal is the PEAK travel over the
    # run: a locked cable never moves much, a slipping one reaches large travel regardless of phase.
    model, _left_idx, right_idx, pulley_idx = build_kinematic_pulley_atwood(mu=mu, compliance=3.0e-7)
    solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=0.8)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    dt = 1.0 / fps / substeps
    peak_travel = 0.0
    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        peak_travel = max(peak_travel, 2.0 - float(state_0.body_q.numpy()[right_idx][2]))
    body_q = state_0.body_q.numpy()
    theta = _hinge_y_angle(body_q, pulley_idx)
    return body_q, peak_travel, theta


def _set_body_translation(state, body_idx, xyz):
    body_q = state.body_q.numpy()
    body_q[body_idx, :3] = np.asarray(xyz, dtype=np.float32)
    state.body_q.assign(body_q)


def _capstan_span_tensions(solver):
    att_l = solver.tendon_seg_attachment_l.numpy()
    att_r = solver.tendon_seg_attachment_r.numpy()
    rest = solver.tendon_seg_rest_length.numpy()
    compliance = solver.model.tendon_seg_compliance.numpy()
    lengths = np.linalg.norm(att_r - att_l, axis=1)
    tensions = np.maximum(lengths - rest, 0.0) / np.maximum(compliance, 1.0e-8)
    return tensions, att_l, att_r


def _capstan_wrap_angle(att_l, att_r):
    normal = np.array([0.0, 1.0, 0.0])
    left_radius = att_r[0]
    right_radius = att_l[1]
    return float(abs(np.atan2(np.dot(np.cross(left_radius, right_radius), normal), np.dot(left_radius, right_radius))))


def _loaded_route_arc_length(solver, model, link_idx):
    center = model.tendon_link_offset.numpy()[link_idx]
    normal = model.tendon_link_axis.numpy()[link_idx]
    radius_l = solver.tendon_seg_attachment_r.numpy()[link_idx - 1] - center
    radius_r = solver.tendon_seg_attachment_l.numpy()[link_idx] - center
    radius_l -= np.dot(radius_l, normal) * normal
    radius_r -= np.dot(radius_r, normal) * normal
    theta = abs(np.atan2(np.dot(np.cross(radius_l, radius_r), normal), np.dot(radius_l, radius_r)))
    return float(theta * model.tendon_link_radius.numpy()[link_idx])


def _loaded_route_material_length(solver, model):
    active_segments = solver.tendon_seg_active.numpy().astype(bool)
    material_length = float(np.sum(solver.tendon_seg_rest_length.numpy()[active_segments]))
    link_active = solver.tendon_link_active.numpy()
    for link_idx in (1, 2):
        if link_active[link_idx]:
            material_length += _loaded_route_arc_length(solver, model, link_idx)
    return material_length


def _route_material_length(solver, model, state):
    """Measure active free-span rest length plus rolling-link arcs."""
    active_segments = solver.tendon_seg_active.numpy().astype(bool)
    material_length = float(np.sum(solver.tendon_seg_rest_length.numpy()[active_segments]))
    body_q = state.body_q.numpy()
    link_type = model.tendon_link_type.numpy()
    link_body = model.tendon_link_body.numpy()
    link_offset = model.tendon_link_offset.numpy()
    link_axis = model.tendon_link_axis.numpy()
    link_radius = model.tendon_link_radius.numpy()
    link_active = solver.tendon_link_active.numpy()
    active_link_l = solver.tendon_seg_active_link_l.numpy()
    active_link_r = solver.tendon_seg_active_link_r.numpy()
    att_l = solver.tendon_seg_attachment_l.numpy()
    att_r = solver.tendon_seg_attachment_r.numpy()

    for link_idx in range(1, model.tendon_link_count - 1):
        if link_type[link_idx] != int(TendonLinkType.ROLLING) or not link_active[link_idx]:
            continue
        left_segments = np.flatnonzero(active_segments & (active_link_r == link_idx))
        right_segments = np.flatnonzero(active_segments & (active_link_l == link_idx))
        if len(left_segments) != 1 or len(right_segments) != 1:
            continue
        left_seg = int(left_segments[0])
        right_seg = int(right_segments[0])
        center = body_q[link_body[link_idx]][:3] + link_offset[link_idx]
        normal = link_axis[link_idx]
        radius_l = att_r[left_seg] - center
        radius_r = att_l[right_seg] - center
        radius_l -= np.dot(radius_l, normal) * normal
        radius_r -= np.dot(radius_r, normal) * normal
        theta = abs(np.atan2(np.dot(np.cross(radius_l, radius_r), normal), np.dot(radius_l, radius_r)))
        material_length += float(theta * link_radius[link_idx])

    return material_length


def _run_loaded_dynamic_route(dynamic, device):
    model, _, anchor_joint, candidate = build_loaded_dynamic_route(dynamic, device)
    solver = newton.solvers.SolverXPBD(model, iterations=12, joint_linear_relaxation=0.9)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Advance both routes once before measuring material transport under load.
    state_0.clear_forces()
    solver.step(state_0, state_1, control, None, 1.0 / 120.0)
    state_0, state_1 = state_1, state_0
    material_before_load = _loaded_route_material_length(solver, model)

    anchor_dof = int(model.joint_qd_start.numpy()[anchor_joint])
    joint_f = control.joint_f.numpy()
    joint_f[anchor_dof] = -25.0
    control.joint_f.assign(joint_f)

    for _ in range(60):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0

    rest_length = solver.tendon_seg_rest_length.numpy()
    span_length = np.linalg.norm(
        solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
    )
    tension = np.maximum(span_length - rest_length, 0.0) / model.tendon_seg_compliance.numpy()
    peak_tension = float(np.max(np.where(solver.tendon_seg_active.numpy(), tension, 0.0)))
    material_after_load = _loaded_route_material_length(solver, model)
    return model, solver, candidate, material_before_load, material_after_load, peak_tension


def test_loaded_dynamic_route_matches_fixed_route(test, device):
    """An initially active dynamic roller should match a fixed roller under load."""
    fixed_model, _, _, fixed_material_before, fixed_material_after, fixed_tension = _run_loaded_dynamic_route(
        False, device
    )
    dynamic_model, dynamic_solver, candidate, dynamic_material_before, dynamic_material_after, dynamic_tension = (
        _run_loaded_dynamic_route(True, device)
    )
    test.assertEqual(int(fixed_model.tendon_link_flags.numpy()[candidate]) & int(TendonLinkFlags.DYNAMIC), 0)
    test.assertNotEqual(
        int(dynamic_model.tendon_link_flags.numpy()[candidate]) & int(TendonLinkFlags.DYNAMIC),
        0,
    )
    test.assertTrue(dynamic_solver.tendon_link_active.numpy()[candidate])
    test.assertAlmostEqual(dynamic_material_before, fixed_material_before, delta=5.0e-5)
    test.assertAlmostEqual(
        dynamic_material_after - dynamic_material_before,
        fixed_material_after - fixed_material_before,
        delta=5.0e-5,
    )
    test.assertGreater(fixed_tension, 0.0)
    test.assertLess(
        abs(dynamic_tension - fixed_tension) / fixed_tension,
        1.0e-3,
        f"Dynamic route tension {dynamic_tension} should match fixed route tension {fixed_tension}",
    )


def _capstan_hysteresis_history(mu=0.2):
    model, _left_idx, _pulley_idx, right_idx = build_kinematic_capstan_hysteresis(mu=mu)
    solver = newton.solvers.SolverXPBD(model, iterations=16, joint_linear_relaxation=1.0)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    loading_z = np.linspace(-2.0, -2.3, 31)
    unloading_z = np.linspace(-2.3, -2.0, 31)[1:]
    history = []
    for z in np.concatenate((loading_z, unloading_z)):
        _set_body_translation(state_0, right_idx, (0.21, 0.0, z))
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0

        tensions, att_l, att_r = _capstan_span_tensions(solver)
        theta = _capstan_wrap_angle(att_l, att_r)
        history.append(
            {
                "z": float(z),
                "t_fix": float(tensions[0]),
                "t_app": float(tensions[1]),
                "theta": theta,
                "alpha": float(np.exp(mu * theta)),
            }
        )

    return history


def _run_mujoco_switch_example(num_frames=220):
    example = MujocoSwitchExample(None, None)
    for _ in range(num_frames):
        example.step()
    return example


def test_force_driven_dynamic_route_updates_inside_solver(test, device):
    """A force-driven rolling candidate should switch without host-side active-set updates."""
    with wp.ScopedDevice(device):
        model, candidate, candidate_link = build_force_driven_dynamic_route(device)
        solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=1.0)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        expected_material_length = float(solver.tendon_total_cable.numpy()[0])
        body_f = np.zeros((model.body_count, 6), dtype=np.float32)
        active_history = []
        transition_material_errors = []
        previous_active = 0
        previous_material_length = expected_material_length
        dt = 1.0 / 120.0

        def step(force_x):
            nonlocal previous_active, previous_material_length
            state_0.clear_forces()
            body_f[candidate] = (force_x, 0.0, 0.0, 0.0, 0.0, 0.0)
            state_0.body_f.assign(body_f)
            solver.step(state_0, state_1, control, contacts, dt)
            candidate_active = int(solver.tendon_link_active.numpy()[candidate_link])
            active_history.append(candidate_active)

            seg_active = solver.tendon_seg_active.numpy().astype(bool)
            material_length = float(np.sum(solver.tendon_seg_rest_length.numpy()[seg_active]))
            if candidate_active:
                candidate_center = model.tendon_link_offset.numpy()[candidate_link]
                candidate_axis = model.tendon_link_axis.numpy()[candidate_link]
                att_l_local = solver.tendon_seg_attachment_l_local.numpy()
                att_r_local = solver.tendon_seg_attachment_r_local.numpy()
                radius_l = att_r_local[0] - candidate_center
                radius_r = att_l_local[1] - candidate_center
                theta = abs(
                    np.atan2(
                        np.dot(np.cross(radius_l, radius_r), candidate_axis),
                        np.dot(radius_l, radius_r),
                    )
                )
                material_length += theta * float(model.tendon_link_radius.numpy()[candidate_link])
            if candidate_active != previous_active:
                transition_material_errors.append(abs(material_length - previous_material_length))
            previous_active = candidate_active
            previous_material_length = material_length
            return state_1, state_0

        for _ in range(60):
            state_0, state_1 = step(-2.0)

        test.assertNotEqual(active_history[-1], 0, "Solver should activate a candidate moved into the span")

        body_qd = state_0.body_qd.numpy()
        body_qd[candidate] = 0.0
        state_0.body_qd.assign(body_qd)
        for _ in range(60):
            state_0, state_1 = step(4.0)

        test.assertEqual(active_history[-1], 0, "Solver should deactivate a candidate moved clear of the span")
        test.assertEqual(len(transition_material_errors), 2, "Expected one activation and one deactivation")
        test.assertLess(
            max(transition_material_errors),
            1.0e-5,
            "Changing route representation should conserve free-span rest length plus wrapped material",
        )


def test_dynamic_route_activation_uses_accepted_pose(test, device):
    """Route activation should not use the inertial predictor pose."""
    with wp.ScopedDevice(device):
        model, candidate, candidate_link = build_force_driven_dynamic_route(device)
        solver = newton.solvers.SolverXPBD(model, iterations=1, joint_linear_relaxation=1.0)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        test.assertFalse(solver.tendon_link_active.numpy()[candidate_link])
        body_qd = state_0.body_qd.numpy()
        body_qd[candidate, 0] = -24.0
        state_0.body_qd.assign(body_qd)

        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)

        test.assertLess(abs(float(state_1.body_q.numpy()[candidate, 0])), 0.1)
        test.assertFalse(
            solver.tendon_link_active.numpy()[candidate_link],
            "The predictor crossed the cable, but activation belongs to the accepted step-start pose",
        )


def test_xpbd_tendon_diagnostics_match_final_pose(test, device):
    """Reported tendon geometry and tension should use the completed XPBD pose."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        body_l = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.5, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        body_r = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.5, 0.0, 0.0)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        compliance = 1.0e-5
        damping = 1.0
        rest_length = 0.5
        builder.add_tendon()
        builder.add_tendon_link(body=body_l, link_type=TendonLinkType.ATTACHMENT)
        builder.add_tendon_link(
            body=body_r,
            link_type=TendonLinkType.ATTACHMENT,
            compliance=compliance,
            damping=damping,
            rest_length=rest_length,
        )
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverXPBD(model, iterations=1, joint_linear_relaxation=1.0)
        state_0 = model.state()
        state_1 = model.state()

        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)

        body_q = state_1.body_q.numpy()
        direction = body_q[body_r, :3] - body_q[body_l, :3]
        final_length = float(np.linalg.norm(direction))
        direction /= final_length
        body_qd = state_1.body_qd.numpy()
        expected_damping_tension = damping * float(np.dot(direction, body_qd[body_r, :3] - body_qd[body_l, :3]))
        attachment_l = solver.tendon_seg_attachment_l.numpy()[0]
        attachment_r = solver.tendon_seg_attachment_r.numpy()[0]
        attachment_length = float(np.linalg.norm(attachment_r - attachment_l))
        expected_material_tension = max((final_length - rest_length) / compliance, 0.0)

        test.assertLess(final_length, 0.75, "The solve must move the endpoints enough to expose stale diagnostics")
        test.assertAlmostEqual(attachment_length, final_length, delta=1.0e-6)
        test.assertAlmostEqual(
            float(solver.tendon_seg_material_tension.numpy()[0]),
            expected_material_tension,
            delta=1.0e-3,
        )
        test.assertAlmostEqual(
            float(solver.tendon_seg_damping_tension.numpy()[0]),
            expected_damping_tension,
            delta=1.0e-3,
        )


def test_dynamic_route_neighbor_matrix_conserves_material(test, device):
    """Route transitions should conserve material for every supported neighbor-type pair."""
    with wp.ScopedDevice(device):
        neighbor_types = (TendonLinkType.ATTACHMENT, TendonLinkType.PINHOLE, TendonLinkType.ROLLING)
        for orientation in (-1, 1):
            for left_type in neighbor_types:
                for right_type in neighbor_types:
                    with test.subTest(orientation=orientation, left_type=left_type, right_type=right_type):
                        model, candidate, candidate_link, inactive_position, active_position = (
                            build_dynamic_route_neighbor_matrix_case(device, left_type, right_type, orientation)
                        )
                        solver = newton.solvers.SolverXPBD(model, iterations=1, joint_linear_relaxation=1.0)
                        state_0, state_1 = model.state(), model.state()

                        test.assertFalse(solver.tendon_link_active.numpy()[candidate_link])
                        expected_material_length = _route_material_length(solver, model, state_0)

                        body_q = state_0.body_q.numpy()
                        body_q[candidate, :3] = active_position
                        state_0.body_q.assign(body_q)
                        solver.step(state_0, state_1, model.control(), None, 1.0 / 4800.0)

                        test.assertTrue(solver.tendon_link_active.numpy()[candidate_link])
                        actual_material_length = _route_material_length(solver, model, state_1)
                        test.assertAlmostEqual(actual_material_length, expected_material_length, delta=1.0e-5)

                        body_q = state_1.body_q.numpy()
                        body_q[candidate, :3] = inactive_position
                        state_1.body_q.assign(body_q)
                        solver.step(state_1, state_0, model.control(), None, 1.0 / 4800.0)

                        test.assertFalse(solver.tendon_link_active.numpy()[candidate_link])
                        actual_material_length = _route_material_length(solver, model, state_0)
                        test.assertAlmostEqual(actual_material_length, expected_material_length, delta=1.0e-5)


def test_dynamic_route_initial_state_matches_geometry(test, device):
    """Dynamic links should resolve their initial state before measuring rest lengths."""
    with wp.ScopedDevice(device):
        inactive_model, _, inactive_link = build_force_driven_dynamic_route(device)
        inactive_solver = newton.solvers.SolverXPBD(inactive_model, iterations=8, joint_linear_relaxation=1.0)
        active_model, _, _, active_link = build_loaded_dynamic_route(True, device)
        active_solver = newton.solvers.SolverXPBD(active_model, iterations=8, joint_linear_relaxation=1.0)

        test.assertNotEqual(
            int(inactive_model.tendon_link_flags.numpy()[inactive_link]) & int(TendonLinkFlags.DYNAMIC),
            0,
        )
        test.assertFalse(inactive_solver.tendon_link_active.numpy()[inactive_link])
        test.assertNotEqual(
            int(active_model.tendon_link_flags.numpy()[active_link]) & int(TendonLinkFlags.DYNAMIC),
            0,
        )
        test.assertTrue(active_solver.tendon_link_active.numpy()[active_link])


def test_inactive_dynamic_route_preserves_authored_rest_length(test, device):
    """An initially inactive roller should merge both authored adjacent rests."""
    with wp.ScopedDevice(device):
        model, candidate_link = build_explicit_inactive_dynamic_route(device)
        solver = newton.solvers.SolverXPBD(model, iterations=1)

        test.assertFalse(solver.tendon_link_active.numpy()[candidate_link])
        active = solver.tendon_seg_active.numpy().astype(bool)
        test.assertAlmostEqual(float(np.sum(solver.tendon_seg_rest_length.numpy()[active])), 1.0, delta=1.0e-6)


def test_inactive_dynamic_route_preserves_mixed_rest_length(test, device):
    """Automatic material should fill the part not assigned an explicit rest length."""
    with wp.ScopedDevice(device):
        model, candidate_link = build_explicit_inactive_dynamic_route(device, rest_lengths=(-1.0, 0.6))
        solver = newton.solvers.SolverXPBD(model, iterations=1)

        test.assertFalse(solver.tendon_link_active.numpy()[candidate_link])
        active = solver.tendon_seg_active.numpy().astype(bool)
        test.assertAlmostEqual(float(np.sum(solver.tendon_seg_rest_length.numpy()[active])), 1.0, delta=1.0e-6)


def test_xpbd_reset_restores_tendon_material_state(test, device):
    """Reset should restore mutable tendon rest lengths and route history."""
    with wp.ScopedDevice(device):
        model, pulley = build_kinematic_rolling_transport(mu=10.0)
        solver = newton.solvers.SolverXPBD(model, iterations=1)
        state_0, state_1 = model.state(), model.state()
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        initial_local_l = solver.tendon_seg_attachment_l_local.numpy().copy()
        initial_local_r = solver.tendon_seg_attachment_r_local.numpy().copy()

        body_q = state_0.body_q.numpy()
        angle = 0.4
        body_q[pulley, 3:] = np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)], dtype=np.float32)
        state_0.body_q.assign(body_q)
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
        test.assertGreater(float(np.max(np.abs(solver.tendon_seg_rest_length.numpy() - initial_rest))), 1.0e-4)

        solver.reset(state_1)
        np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), initial_rest, atol=1.0e-7)
        np.testing.assert_allclose(solver.tendon_seg_attachment_l_local.numpy(), initial_local_l, atol=1.0e-7)
        np.testing.assert_allclose(solver.tendon_seg_attachment_r_local.numpy(), initial_local_r, atol=1.0e-7)

        solver.step(state_1, state_0, model.control(), None, 1.0 / 60.0)
        np.testing.assert_allclose(solver.tendon_seg_rest_length.numpy(), initial_rest, atol=1.0e-7)


def test_dynamic_route_uses_oriented_signed_distance(test, device):
    """A dynamic roller should remain active after crossing its bypass span."""
    with wp.ScopedDevice(device):
        for orientation in (-1, 1):
            model, candidate, candidate_link = build_oriented_dynamic_route(orientation, device)
            solver = newton.solvers.SolverXPBD(model, iterations=1)
            body_q = model.body_q.numpy()

            active_history = []
            for x in (
                0.25 * orientation,
                0.05 * orientation,
                -0.25 * orientation,
                0.05 * orientation,
                0.25 * orientation,
            ):
                body_q[candidate, :3] = (x, 0.0, 0.0)
                model.body_q.assign(body_q)
                solver._update_tendon_link_active(model, model.body_q)
                active_history.append(bool(solver.tendon_link_active.numpy()[candidate_link]))

            test.assertEqual(
                active_history,
                [
                    False,
                    True,
                    True,
                    True,
                    False,
                ],
                f"orientation={orientation}",
            )


def test_dynamic_route_activation_tolerance_preserves_boundary_state(test, device):
    """Dynamic routing should not chatter inside the activation tolerance."""
    with wp.ScopedDevice(device):
        radius = 0.1
        activation_tol = 1.0e-3
        activation_gap = activation_tol * radius
        for orientation in (-1, 1):
            model, candidate, candidate_link = build_oriented_dynamic_route(orientation, device)
            solver = newton.solvers.SolverXPBD(
                model,
                iterations=1,
                tendon_activation_tol=activation_tol,
            )
            body_q = model.body_q.numpy()

            transitions = (
                (radius - 0.5 * activation_gap, False),
                (radius - 1.5 * activation_gap, True),
                (radius - 0.5 * activation_gap, True),
                (radius + 0.5 * activation_gap, False),
            )
            for distance, expected_active in transitions:
                body_q[candidate, :3] = (distance * orientation, 0.0, 0.0)
                model.body_q.assign(body_q)
                solver._update_tendon_link_active(model, model.body_q)
                actual_active = bool(solver.tendon_link_active.numpy()[candidate_link])
                test.assertEqual(actual_active, expected_active)


def test_dynamic_route_requires_projection_on_bypass_span(test, device):
    """A roller beyond either neighboring site should remain inactive."""
    with wp.ScopedDevice(device):
        for orientation in (-1, 1):
            model, candidate, candidate_link = build_oriented_dynamic_route(orientation, device)
            solver = newton.solvers.SolverXPBD(model, iterations=1)
            body_q = model.body_q.numpy()
            for z in (-0.75, 0.75):
                body_q[candidate, :3] = (-0.25 * orientation, 0.0, z)
                model.body_q.assign(body_q)
                solver._update_tendon_link_active(model, model.body_q)

                test.assertFalse(
                    solver.tendon_link_active.numpy()[candidate_link],
                    f"orientation={orientation}, z={z}",
                )


def test_fixed_route_state_is_preserved(test, device):
    """A fixed link should ignore runtime writes to the solver-owned route state."""
    with wp.ScopedDevice(device):
        model, _, _, candidate_link = build_loaded_dynamic_route(True, device)
        solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=1.0)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        fixed_link = candidate_link - 1

        test.assertEqual(int(model.tendon_link_flags.numpy()[fixed_link]) & int(TendonLinkFlags.DYNAMIC), 0)
        test.assertTrue(solver.tendon_link_active.numpy()[fixed_link])
        link_active = solver.tendon_link_active.numpy()
        link_active[fixed_link] = False
        solver.tendon_link_active.assign(link_active)
        solver.step(state_0, state_1, control, None, 1.0 / 120.0)
        test.assertTrue(solver.tendon_link_active.numpy()[fixed_link])


def test_dynamic_route_cuda_graph_capture(test, device):
    """Solver-owned active-set updates should remain CUDA graph-capturable."""
    device = wp.get_device(device)
    if not wp.is_mempool_enabled(device):
        test.skipTest("CUDA graph capture requires the Warp memory pool")

    with wp.ScopedDevice(device):
        # Compile the solver kernels outside capture using a disposable solver state.
        warm_model, warm_candidate, _ = build_force_driven_dynamic_route(device)
        warm_solver = newton.solvers.SolverXPBD(warm_model, iterations=8, joint_linear_relaxation=1.0)
        warm_0, warm_1 = warm_model.state(), warm_model.state()
        newton.eval_fk(warm_model, warm_model.joint_q, warm_model.joint_qd, warm_0)
        warm_force = np.zeros((warm_model.body_count, 6), dtype=np.float32)
        warm_force[warm_candidate] = (-2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        warm_0.body_f.assign(warm_force)
        warm_solver.step(warm_0, warm_1, warm_model.control(), None, 1.0 / 120.0)

        model, candidate, candidate_link = build_force_driven_dynamic_route(device)
        solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=1.0)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
        body_f = np.zeros((model.body_count, 6), dtype=np.float32)
        body_f[candidate] = (-2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state_0.body_f.assign(body_f)
        state_1.body_f.assign(body_f)

        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_0, state_1, control, None, 1.0 / 120.0)
            solver.step(state_1, state_0, control, None, 1.0 / 120.0)

        for _ in range(30):
            wp.capture_launch(capture.graph)

        test.assertNotEqual(
            int(solver.tendon_link_active.numpy()[candidate_link]),
            0,
            "Captured solver steps should activate a force-driven routing candidate",
        )


def test_mujoco_switch_optional_middle_capstan_activates(test, device):
    """A rotating endpoint should switch one middle rolling guide in and out."""
    with wp.ScopedDevice(device):
        example = _run_mujoco_switch_example()
        example.test_final()

        active_history = np.array(example._active_history, dtype=np.int32)
        link_type = example.model.tendon_link_type.numpy()
        authored_flags = example.model.tendon_link_flags.numpy()
        test.assertEqual(link_type[example.lower_link], int(newton.TendonLinkType.ROLLING))
        test.assertEqual(link_type[example.middle_link], int(newton.TendonLinkType.ROLLING))
        test.assertEqual(int(authored_flags[example.lower_link]) & int(TendonLinkFlags.DYNAMIC), 0)
        test.assertNotEqual(int(authored_flags[example.middle_link]) & int(TendonLinkFlags.DYNAMIC), 0)
        test.assertEqual(active_history[0], 0, f"Switch route should start on lower-guide-only path: {active_history}")
        test.assertEqual(active_history[-1], 0, f"Switch route should end on lower-guide-only path: {active_history}")
        test.assertEqual(int(np.max(active_history)), 1, f"Middle capstan should activate: {active_history}")
        test.assertGreaterEqual(example._transition_count, 2)
        test.assertEqual(example._activation_mismatch_count, 0)


def test_mujoco_switch_preserves_active_route_segments(test, device):
    """Active-set routing should disable and restore the middle candidate segment."""
    with wp.ScopedDevice(device):
        example = _run_mujoco_switch_example()

        test.assertTrue(example._saw_middle_segment_disabled, "Inactive middle link should be skipped")
        test.assertTrue(example._saw_middle_segment_enabled, "Active middle link should restore its second segment")
        test.assertLess(
            example._max_inactive_middle_penetration,
            example.middle_radius * example.solver.tendon_activation_tol + 1.0e-5,
        )
        test.assertGreater(example._min_active_tangent_radius, example.middle_radius * 0.80)
        test.assertLess(example._max_active_tangent_radius, example.middle_radius * 1.20)


def test_pinhole_slip_atwood(test, device):
    """A pinhole is a frictionless slip waypoint: heavy descends, light rises."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx = build_pinhole_atwood(compliance=4.0e-7)
        state = run_model(model, num_frames=80)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite pinhole Atwood state")

        left_z = float(body_q[left_idx][2])
        right_z = float(body_q[right_idx][2])
        test.assertGreater(left_z, 2.05, f"Light side should rise through pinhole slip: z={left_z:.4f}")
        test.assertLess(right_z, 1.95, f"Heavy side should descend through pinhole slip: z={right_z:.4f}")


def _pinhole_friction_metrics(mu, num_frames=80):
    model, left_idx, right_idx = build_pinhole_atwood(mass_left=1.0, mass_right=3.0, mu=mu, compliance=4.0e-7)
    state = run_model(model, num_frames=num_frames)
    body_q = state.body_q.numpy()
    left_travel = float(body_q[left_idx][2]) - 2.0
    right_travel = 2.0 - float(body_q[right_idx][2])
    return body_q, left_travel, right_travel


def test_frictional_pinhole_mu_controls_slip_and_locking(test, device):
    """Pinhole friction should interpolate between free slip and locked cable transfer."""
    with wp.ScopedDevice(device):
        low = _pinhole_friction_metrics(mu=0.0)
        mid = _pinhole_friction_metrics(mu=0.1)
        high = _pinhole_friction_metrics(mu=10.0)

        for label, metrics in [("low", low), ("mid", mid), ("high", high)]:
            body_q, _left_travel, right_travel = metrics
            test.assertTrue(np.isfinite(body_q).all(), f"Non-finite pinhole state for {label} mu")
            test.assertGreater(right_travel, 0.03, f"Heavy side should descend for {label} mu: {right_travel:.5f}")

        _, left_low, right_low = low
        _, left_mid, right_mid = mid
        _, _left_high, right_high = high

        test.assertGreater(right_low, 0.25, f"Zero-mu pinhole should freely slip: dz={right_low:.5f}")
        test.assertGreater(left_low, 0.20, f"Zero-mu light side should rise through pinhole: dz={left_low:.5f}")
        test.assertGreater(
            left_mid, 0.08, f"Mid-mu light side should still rise through partial slip: dz={left_mid:.5f}"
        )
        test.assertGreater(
            right_low, right_mid + 0.10, f"Mid mu should slip less than zero mu: {right_low:.5f} vs {right_mid:.5f}"
        )
        test.assertGreater(
            right_mid, right_high + 0.05, f"High mu should lock more than mid mu: {right_mid:.5f} vs {right_high:.5f}"
        )
        test.assertLess(right_high, 0.10, f"High-mu pinhole should lock cable transfer: dz={right_high:.5f}")


def test_slack_pinhole_does_not_redistribute(test, device):
    """Pinholes should transfer only taut excess, not proportionally repartition slack."""
    with wp.ScopedDevice(device):
        model = build_slack_pinhole_route()
        solver = newton.solvers.SolverXPBD(model, iterations=4, joint_linear_relaxation=0.8)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()

        initial = solver.tendon_seg_rest_length.numpy().copy()
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
        final = solver.tendon_seg_rest_length.numpy()

        np.testing.assert_allclose(
            final,
            initial,
            rtol=0.0,
            atol=1.0e-6,
            err_msg=f"Slack pinhole rest lengths should not be repartitioned: {initial} -> {final}",
        )


def test_dynamic_pulley_uses_angular_jacobian(test, device):
    """High friction recovers the no-slip angular Jacobian baseline."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=10.0, compliance=3.0e-8)
        state = run_model(model, num_frames=80)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite dynamic pulley Atwood state")

        left_z = float(body_q[left_idx][2])
        right_z = float(body_q[right_idx][2])
        q = body_q[pulley_idx]
        theta = abs(2.0 * np.arctan2(float(q[4]), float(q[6])))

        test.assertGreater(left_z, 2.05, f"Light side should rise: z={left_z:.4f}")
        test.assertLess(right_z, 1.95, f"Heavy side should descend: z={right_z:.4f}")
        test.assertGreater(theta, 0.05, f"Pulley should rotate from the full angular Jacobian: theta={theta:.4f}")


def test_pulley_inertia_limit_locks_cable_travel(test, device):
    """As pulley inertia tends to infinity, no-slip cable travel tends to zero."""
    with wp.ScopedDevice(device):
        radius = 0.15
        model, _, _, pulley_idx = build_dynamic_pulley_atwood(
            mu=10.0,
            mass_left=1.0,
            mass_right=3.0,
            pulley_mass=5000.0,
            pulley_radius=radius,
            compliance=3.0e-8,
        )
        state = run_model(model, num_frames=80)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite high-inertia pulley state")

        q = body_q[pulley_idx]
        theta = abs(2.0 * np.arctan2(float(q[4]), float(q[6])))
        rim_travel = theta * radius

        test.assertLess(
            rim_travel,
            5.0e-3,
            f"High-inertia no-slip pulley should lock cable travel: R*theta={rim_travel:.6f}, theta={theta:.6f}",
        )


def test_dynamic_capstan_mu_controls_pulley_rotation(test, device):
    """Dynamic capstan: zero mu slips, mid mu grips partially, high mu approaches no-slip."""
    with wp.ScopedDevice(device):
        low = _dynamic_capstan_metrics(device, mu=0.0)
        mid = _dynamic_capstan_metrics(device, mu=0.04)
        high = _dynamic_capstan_metrics(device, mu=10.0)

        for label, metrics in [("low", low), ("mid", mid), ("high", high)]:
            body_q, left_travel, right_travel, *_ = metrics
            test.assertTrue(np.isfinite(body_q).all(), f"Non-finite dynamic capstan state for {label} mu")
            test.assertGreater(right_travel, 0.03, f"Heavy side should descend for {label} mu: {right_travel:.5f}")
            test.assertGreater(left_travel, -0.04, f"Light side should not sink for {label} mu: {left_travel:.5f}")

        _, _, _, _, theta_low, _, slip_low = low
        _, _, _, _, theta_mid, rim_mid, slip_mid = mid
        _, _, _, cable_high, theta_high, rim_high, slip_high = high

        test.assertLess(abs(theta_low), 0.035, f"Zero-mu dynamic pulley should not rotate: theta={theta_low:.5f}")
        test.assertGreater(
            theta_mid, theta_low + 0.02, f"Mid-mu pulley should rotate in cable direction: {theta_mid:.5f}"
        )
        test.assertGreater(
            theta_high, theta_mid + 0.03, f"High-mu pulley should rotate more than mid mu: {theta_high:.5f}"
        )
        test.assertLess(
            abs(cable_high - rim_high),
            0.06,
            f"High-mu dynamic capstan should approach no-slip: cable={cable_high:.5f}, rim={rim_high:.5f}",
        )
        test.assertGreater(
            slip_mid, slip_high, f"Dynamic slip should decrease from mid to high mu: {slip_mid:.5f} <= {slip_high:.5f}"
        )
        test.assertGreater(
            slip_low, slip_high, f"High friction should slip less than zero friction: {slip_low:.5f} <= {slip_high:.5f}"
        )
        test.assertGreater(rim_mid, 0.0, f"Mid-mu rim travel should be positive: {rim_mid:.5f}")


def test_kinematic_capstan_mu_controls_slip_and_locking(test, device):
    """Kinematic capstan: low friction lets the cable slip, high friction locks it.

    Measured by peak travel over the run. The compliant cable makes the atwood oscillate, so a
    single-frame travel (or tension ratio) is phase-dependent and unreliable; peak travel is a
    robust slip-vs-lock signal. The finer "more friction slips slightly less" gradient is not
    cleanly observable here (the oscillation amplitudes of the slipping cases overlap) -- the
    capstan tension cone itself is verified precisely by the quasi-static tests
    (test_finite_friction_zero_span_respects_global_capstan_cone, stiff_pinhole_capstan).
    """
    with wp.ScopedDevice(device):
        low = _kinematic_capstan_metrics(device, mu=0.0)
        mid = _kinematic_capstan_metrics(device, mu=0.08)
        high = _kinematic_capstan_metrics(device, mu=10.0)

        for label, metrics in [("low", low), ("mid", mid), ("high", high)]:
            body_q, _, theta = metrics
            test.assertTrue(np.isfinite(body_q).all(), f"Non-finite kinematic capstan state for {label} mu")
            test.assertLess(abs(theta), 1.0e-5, f"Kinematic pulley should not rotate for {label} mu: {theta:.6f}")

        peak_low, peak_mid, peak_high = low[1], mid[1], high[1]

        # mass ratio 3 exceeds the friction cone for low/mid mu (exp(mu*pi) < 3), so the cable slips;
        # for mu=10 the cone vastly exceeds it, so the cable locks.
        test.assertGreater(peak_low, 0.5, f"Zero-mu kinematic capstan should slip freely: peak={peak_low:.4f}")
        test.assertGreater(peak_mid, 0.5, f"Mid-mu cone still below mass ratio, should slip: peak={peak_mid:.4f}")
        test.assertLess(peak_high, 0.1, f"High-mu kinematic capstan should lock the cable: peak={peak_high:.4f}")


def test_kinematic_capstan_hysteresis_matches_capstan_band(test, device):
    """Loading and unloading should hit opposite sides of the capstan friction cone."""
    with wp.ScopedDevice(device):
        history = _capstan_hysteresis_history(mu=0.2)

        loading = history[:31]
        unloading = history[31:]
        peak = loading[-1]
        peak_fix = peak["t_fix"]
        peak_app = peak["t_app"]
        peak_alpha = peak["alpha"]

        test.assertGreater(peak_app, 150.0, f"Benchmark should generate a meaningful applied tension: {peak_app}")
        test.assertAlmostEqual(
            peak_app / peak_fix,
            peak_alpha,
            delta=0.01,
            msg=f"Loading branch should slip at T_app/T_fix=alpha: peak={peak}, ratio={peak_app / peak_fix}",
        )

        # All samples should remain inside the static capstan cone. Once a side
        # tries to exceed the cone, rest length transfers and the active branch
        # lands on the corresponding capstan boundary.
        for sample in history:
            t_fix = sample["t_fix"]
            t_app = sample["t_app"]
            alpha = sample["alpha"]
            if max(t_fix, t_app) < 5.0:
                continue
            test.assertLessEqual(
                t_app,
                alpha * t_fix + 1.0,
                f"Applied side escaped capstan cone: sample={sample}",
            )
            test.assertLessEqual(
                t_fix,
                alpha * t_app + 1.0,
                f"Fixed side escaped capstan cone: sample={sample}",
            )

        loading_ratios = [sample["t_app"] / sample["t_fix"] for sample in loading if sample["t_fix"] > 10.0]
        loading_alphas = [sample["alpha"] for sample in loading if sample["t_fix"] > 10.0]
        max_loading_error = max(abs(ratio - alpha) for ratio, alpha in zip(loading_ratios, loading_alphas, strict=True))
        test.assertLess(max_loading_error, 0.01, f"Loading branch did not track alpha: err={max_loading_error}")

        stick_samples = [sample for sample in unloading if sample["t_app"] > 0.8 * peak_fix]
        test.assertGreaterEqual(len(stick_samples), 5, f"Expected several unloading stick samples: {stick_samples}")
        fixed_variation = max(abs(sample["t_fix"] - peak_fix) for sample in stick_samples)
        test.assertLess(
            fixed_variation,
            1.0,
            f"Fixed-side tension should stay on the static plateau during early unloading: {stick_samples}",
        )

        reverse_threshold = peak_fix / peak_alpha
        reverse_slip = [sample for sample in unloading if 20.0 < sample["t_app"] < 0.9 * reverse_threshold]
        test.assertTrue(
            reverse_slip, f"Expected reverse-slip samples after unloading crosses the lower band: {unloading}"
        )
        reverse_errors = [abs(sample["t_fix"] / sample["t_app"] - sample["alpha"]) for sample in reverse_slip]
        test.assertLess(max(reverse_errors), 0.02, f"Unloading branch did not track alpha: err={reverse_errors}")

        target_app = 0.5 * peak_app
        loading_mid = min(loading, key=lambda sample: abs(sample["t_app"] - target_app))
        unloading_mid = min(unloading, key=lambda sample: abs(sample["t_app"] - target_app))
        test.assertLess(
            abs(loading_mid["t_app"] - unloading_mid["t_app"]),
            5.0,
            f"Could not compare similar applied tensions: loading={loading_mid}, unloading={unloading_mid}",
        )
        test.assertGreater(
            unloading_mid["t_fix"],
            loading_mid["t_fix"] + 40.0,
            f"Expected hysteresis: fixed tension should be higher on unloading at similar applied tension: "
            f"loading={loading_mid}, unloading={unloading_mid}",
        )


def test_pinhole_capstan_force_mode_uses_physical_compliance(test, device):
    """A force-driven pinhole capstan should report stretch tension in newtons."""
    with wp.ScopedDevice(device):
        num_pinholes = 5
        mu = 0.2
        model, right_idx, mass, initial_z = build_pinhole_capstan_force_mode(num_pinholes=num_pinholes, mu=mu)
        solver = newton.solvers.SolverXPBD(model, iterations=40, joint_linear_relaxation=1.0)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        dt = 1.0 / 60.0

        for _ in range(120):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite pinhole capstan force-mode state")

        drop = initial_z - float(body_q[right_idx][2])
        tensions, _att_l, _att_r = _capstan_span_tensions(solver)
        t_fix = float(tensions[0])
        t_app = float(tensions[-1])
        t_lambda = float(-solver.tendon_seg_lambda.numpy()[-1] / dt)
        applied_load = mass * 9.81
        capstan_ratio = np.exp(mu * np.pi)

        test.assertGreater(drop, 0.2, f"Force-driven cable should stretch by a physical amount: dz={drop:.6f}")
        test.assertAlmostEqual(
            t_app,
            t_lambda,
            delta=0.05 * applied_load,
            msg=f"Stretch/compliance tension should match XPBD multiplier readback: {t_app:.3f} vs {t_lambda:.3f}",
        )
        test.assertAlmostEqual(
            t_app,
            applied_load,
            delta=0.15 * applied_load,
            msg=f"Applied-side tension should balance the hanging load: T={t_app:.3f}, load={applied_load:.3f}",
        )
        test.assertLessEqual(t_app, capstan_ratio * t_fix + 5.0, f"Applied side escaped capstan cone: {t_app}, {t_fix}")
        test.assertLessEqual(t_fix, capstan_ratio * t_app + 5.0, f"Fixed side escaped capstan cone: {t_app}, {t_fix}")


def test_simple_cable_gravity_balances_mass_load(test, device):
    """A single cable segment should report the physical stretch tension under gravity."""
    with wp.ScopedDevice(device):
        model, body_idx, mass, compliance, initial_z = build_simple_cable_gravity()
        solver = newton.solvers.SolverXPBD(model, iterations=16, joint_linear_relaxation=1.0)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        dt = 1.0 / 60.0

        for _ in range(300):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite simple cable gravity state")

        tensions, _att_l, _att_r = _capstan_span_tensions(solver)
        tension = float(tensions[0])
        lambda_tension = float(-solver.tendon_seg_lambda.numpy()[0] / dt)
        load = mass * 9.81
        expected_z = initial_z - load * compliance
        z = float(body_q[body_idx][2])

        test.assertAlmostEqual(
            tension,
            load,
            delta=0.05 * load,
            msg=f"Single-segment stretch tension should balance mg: T={tension:.3f}, mg={load:.3f}",
        )
        test.assertAlmostEqual(
            lambda_tension,
            tension,
            delta=0.05 * load,
            msg=f"XPBD lambda tension should match stretch/compliance readback: {lambda_tension:.3f} vs {tension:.3f}",
        )
        test.assertAlmostEqual(
            z,
            expected_z,
            delta=5.0e-3,
            msg=f"Mass should settle near compliance extension: z={z:.6f}, expected={expected_z:.6f}",
        )


def test_simple_cable_gravity_tension_independent_of_relaxation(test, device):
    """Under-relaxed accumulated XPBD rows should converge to the same physical tension."""
    with wp.ScopedDevice(device):
        dt = 1.0 / 60.0
        results = []
        for relaxation in (1.0, 0.8, 0.5, 0.25):
            model, body_idx, mass, _compliance, _initial_z = build_simple_cable_gravity()
            solver = newton.solvers.SolverXPBD(model, iterations=24, joint_linear_relaxation=relaxation)
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()
            collision_pipeline = newton.CollisionPipeline(model)
            contacts = collision_pipeline.contacts()

            for _ in range(420):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            test.assertTrue(np.isfinite(body_q).all(), "Non-finite simple cable gravity state")

            tensions, _att_l, _att_r = _capstan_span_tensions(solver)
            tension = float(tensions[0])
            lambda_tension = float(-solver.tendon_seg_lambda.numpy()[0] / dt)
            load = mass * 9.81
            velocity_z = float(body_qd[body_idx][2])
            results.append((relaxation, tension, lambda_tension, load, velocity_z))

            test.assertAlmostEqual(
                tension,
                load,
                delta=0.04 * load,
                msg=(
                    f"Stretch tension should not be scaled by relaxation={relaxation}: T={tension:.3f}, mg={load:.3f}"
                ),
            )
            test.assertAlmostEqual(
                lambda_tension,
                tension,
                delta=0.02 * load,
                msg=(
                    f"XPBD lambda tension should match stretch readback for relaxation={relaxation}: "
                    f"lambda={lambda_tension:.3f}, stretch={tension:.3f}"
                ),
            )
            test.assertLess(
                abs(velocity_z),
                0.02,
                msg=f"Simple cable mass should be near rest before reading tension: vz={velocity_z:.6f}",
            )

        reference = results[0][1]
        for relaxation, tension, _lambda_tension, load, _velocity_z in results[1:]:
            test.assertAlmostEqual(
                tension,
                reference,
                delta=0.03 * load,
                msg=(
                    f"Relaxation should affect convergence speed, not converged tension: "
                    f"relaxation={relaxation}, T={tension:.3f}, reference={reference:.3f}"
                ),
            )


def test_frictionless_pinhole_equalizes_damped_tension(test, device):
    """A frictionless route must balance total tension, including axial damping."""
    with wp.ScopedDevice(device):
        length_rate = 1.0e-3
        compliance = 0.1
        for initial_tension in (0.0, 1.0):
            for damping in (0.0, 5.0, 100.0, 200.0):
                with test.subTest(initial_tension=initial_tension, damping=damping):
                    rest_length = 1.0 - compliance * initial_tension
                    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
                    left = builder.add_body(
                        xform=wp.transform(p=wp.vec3(-1.0, 0.0, 0.0)),
                        mass=0.0,
                        is_kinematic=True,
                    )
                    pin = builder.add_body(
                        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0)),
                        mass=0.0,
                        is_kinematic=True,
                    )
                    right = builder.add_body(
                        xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
                        mass=0.0,
                        is_kinematic=True,
                    )
                    builder.add_tendon()
                    builder.add_tendon_link(
                        body=left,
                        link_type=int(TendonLinkType.ATTACHMENT),
                    )
                    builder.add_tendon_link(
                        body=pin,
                        link_type=int(TendonLinkType.PINHOLE),
                        compliance=compliance,
                        damping=damping,
                        rest_length=rest_length,
                    )
                    builder.add_tendon_link(
                        body=right,
                        link_type=int(TendonLinkType.ATTACHMENT),
                        compliance=compliance,
                        damping=damping,
                        rest_length=rest_length,
                    )

                    model = builder.finalize()
                    solver = newton.solvers.SolverXPBD(
                        model,
                        iterations=1,
                        joint_linear_relaxation=1.0,
                        tendon_max_sweeps=32,
                        tendon_settle_tol=1.0e-3,
                    )
                    state_0, state_1 = model.state(), model.state()
                    body_qd = state_0.body_qd.numpy()
                    body_qd[left, 0] = -length_rate
                    state_0.body_qd.assign(body_qd)
                    dt = 0.01

                    solver.step(state_0, state_1, model.control(), None, dt)

                    tension = -solver.tendon_seg_lambda.numpy() / dt
                    expected_tension = initial_tension + 0.5 * damping * length_rate
                    test.assertAlmostEqual(
                        float(tension[0]),
                        float(tension[1]),
                        delta=1.0e-5,
                        msg=f"Frictionless spans should balance damped tension: {tension}",
                    )
                    test.assertAlmostEqual(float(tension[0]), expected_tension, delta=1.0e-5)
                    test.assertLess(int(solver.tendon_cone_sweep_count.numpy()[0]), 32)


def test_tendon_damping_does_not_generate_compression(test, device):
    """Axial damping may reduce cable tension to zero but cannot make it compressive."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        left = builder.add_body(
            xform=wp.transform(p=wp.vec3(-1.0, 0.0, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )
        right = builder.add_body(
            xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0)),
            mass=0.0,
            is_kinematic=True,
        )
        builder.add_tendon()
        builder.add_tendon_link(body=left, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=right,
            link_type=int(TendonLinkType.ATTACHMENT),
            compliance=0.1,
            damping=2.0,
            rest_length=1.9,
        )

        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=1,
            joint_linear_relaxation=1.0,
        )
        state_0, state_1 = model.state(), model.state()
        body_qd = state_0.body_qd.numpy()
        body_qd[left, 0] = 1.0
        state_0.body_qd.assign(body_qd)
        dt = 0.01

        solver.step(state_0, state_1, model.control(), None, dt)

        tension = -float(solver.tendon_seg_lambda.numpy()[0]) / dt
        test.assertAlmostEqual(tension, 0.0, delta=1.0e-6)


def test_tendon_slip_uses_active_route_segments(test, device):
    """The slip cone should use the current routed neighbors and their physical tension."""
    with wp.ScopedDevice(device):
        compliance_l = 1.0e-9
        compliance_r = 1.0e-7
        rest_l = 1.0 - 1.0e-6
        rest_r = 1.0 - 1.0e-4
        mu = 0.1

        body_q = wp.array([wp.transform_identity()], dtype=wp.transform)
        tendon_link_cone_seg_left = wp.array([-1, -1, 0, -1], dtype=int)
        tendon_link_cone_seg_right = wp.array([-1, -1, 2, -1], dtype=int)
        tendon_link_body = wp.array([0, 0, 0, 0], dtype=int)
        tendon_link_type = wp.array(
            [
                int(TendonLinkType.ATTACHMENT),
                int(TendonLinkType.ROLLING),
                int(TendonLinkType.ROLLING),
                int(TendonLinkType.ATTACHMENT),
            ],
            dtype=int,
        )
        tendon_link_radius = wp.array([0.0, 0.1, 1.0, 0.0], dtype=float)
        tendon_link_mu = wp.array([0.0, 0.0, mu, 0.0], dtype=float)
        tendon_link_active = wp.array([True, False, True, True], dtype=bool)
        tendon_link_offset = wp.zeros(4, dtype=wp.vec3)
        tendon_link_axis = wp.array([wp.vec3(0.0, 1.0, 0.0)] * 4, dtype=wp.vec3)
        seg_rest_length = wp.array([rest_l, 1.0, rest_r], dtype=float)
        seg_attachment_l = wp.array([wp.vec3(-1.0, 0.0, -1.0), wp.vec3(0.0), wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3)
        seg_attachment_r = wp.array([wp.vec3(-1.0, 0.0, 0.0), wp.vec3(0.0), wp.vec3(1.0, 0.0, -1.0)], dtype=wp.vec3)
        seg_compliance = wp.array([compliance_l, 1.0, compliance_r], dtype=float)
        seg_material_tension = wp.array(
            [(1.0 - rest_l) / compliance_l, 0.0, (1.0 - rest_r) / compliance_r], dtype=float
        )
        seg_damping_tension = wp.zeros(3, dtype=float)
        seg_delta_lambda = wp.array([-1.0, 0.0, 0.0], dtype=float)
        body_deltas = wp.zeros(1, dtype=wp.spatial_vector)

        wp.launch(
            solve_tendon_slip,
            dim=4,
            inputs=[
                body_q,
                tendon_link_cone_seg_left,
                tendon_link_cone_seg_right,
                tendon_link_body,
                tendon_link_type,
                tendon_link_radius,
                tendon_link_mu,
                tendon_link_active,
                tendon_link_offset,
                tendon_link_axis,
                seg_attachment_l,
                seg_attachment_r,
                seg_compliance,
                seg_material_tension,
                seg_damping_tension,
                seg_delta_lambda,
                1.0,
            ],
            outputs=[body_deltas],
        )

        rest = seg_rest_length.numpy()
        compliance = seg_compliance.numpy()
        force_l = (1.0 - float(rest[0])) / float(compliance[0])
        force_r = (1.0 - float(rest[2])) / float(compliance[2])
        cap_ratio = math.exp(mu * math.pi)
        beta = (cap_ratio - 1.0) / (cap_ratio + 1.0)
        scale = min(1.0, beta * (force_l + force_r) / max(abs(force_l - force_r), 1.0e-8))
        expected_spin_y = -scale * beta

        test.assertAlmostEqual(float(body_deltas.numpy()[0, 4]), expected_spin_y, delta=1.0e-6)


def test_same_body_tendon_segment_reports_constitutive_tension(test, device):
    """A segment internal to one rigid body should not acquire effective mass."""
    with wp.ScopedDevice(device):
        length = 1.0e-3
        compliance = 1.0e-8
        rest_length = 0.99 * length

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            lock_inertia=True,
        )
        builder.add_tendon()
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(-0.5 * length, 0.0, 0.0),
        )
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(0.5 * length, 0.0, 0.0),
            compliance=compliance,
            rest_length=rest_length,
        )

        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(model, iterations=1, joint_linear_relaxation=1.0)
        state_0, state_1 = model.state(), model.state()
        initial_pose = state_0.body_q.numpy().copy()
        dt = 1.0 / 120.0

        state_0.clear_forces()
        solver.step(state_0, state_1, model.control(), None, dt)

        span_length = float(
            np.linalg.norm(solver.tendon_seg_attachment_r.numpy()[0] - solver.tendon_seg_attachment_l.numpy()[0])
        )
        expected_tension = (span_length - rest_length) / compliance
        lambda_tension = float(-solver.tendon_seg_lambda.numpy()[0] / dt)

        test.assertAlmostEqual(lambda_tension, expected_tension, delta=0.1)
        np.testing.assert_allclose(state_1.body_q.numpy(), initial_pose, rtol=0.0, atol=1.0e-7)


def test_same_body_rolling_route_matches_length_gradient(test, device):
    """Stretch and damping rows must differentiate the complete routed length."""
    with wp.ScopedDevice(device):
        anchor_point = np.array([-0.8, -0.5], dtype=np.float64)
        roller_center_local = np.array([0.1, 0.15], dtype=np.float64)
        attachment_local = np.array([0.5, -0.3], dtype=np.float64)
        radius = 0.12
        angle = 0.23

        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor = builder.add_body(
            xform=wp.transform(p=wp.vec3(anchor_point[0], anchor_point[1], 0.0)),
            mass=0.0,
            is_kinematic=True,
        )
        body = builder.add_body(
            xform=wp.transform(q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            lock_inertia=True,
        )
        builder.add_tendon()
        builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ROLLING),
            radius=radius,
            orientation=-1,
            mu=0.0,
            offset=(*roller_center_local, 0.0),
            compliance=1.0e-3,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=body,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(*attachment_local, 0.0),
            compliance=1.0e-3,
            rest_length=-1.0,
        )

        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(model, iterations=1, joint_linear_relaxation=1.0)
        state = model.state()
        attachment_l = solver.tendon_seg_attachment_l.numpy()
        attachment_r = solver.tendon_seg_attachment_r.numpy()
        tangent_left_ref = attachment_r[0, :2].astype(np.float64)
        tangent_right_ref = attachment_l[1, :2].astype(np.float64)

        def rotate(point, q):
            c = math.cos(q)
            s = math.sin(q)
            return np.array([c * point[0] - s * point[1], s * point[0] + c * point[1]])

        def tangent_point(point, center, reference):
            offset = point - center
            distance_sq = float(np.dot(offset, offset))
            base = center + radius * radius * offset / distance_sq
            perpendicular = np.array([-offset[1], offset[0]])
            tangent_offset = radius * math.sqrt(distance_sq - radius * radius) * perpendicular / distance_sq
            candidates = (base + tangent_offset, base - tangent_offset)
            return min(candidates, key=lambda candidate: np.linalg.norm(candidate - reference))

        def routed_length(q):
            center = rotate(roller_center_local, q)
            attachment = rotate(attachment_local, q)
            tangent_left = tangent_point(anchor_point, center, tangent_left_ref)
            tangent_right = tangent_point(attachment, center, tangent_right_ref)
            radial_left = (tangent_left - center) / radius
            radial_right = (tangent_right - center) / radius
            wrap_angle = abs(
                math.atan2(
                    radial_left[0] * radial_right[1] - radial_left[1] * radial_right[0],
                    float(np.dot(radial_left, radial_right)),
                )
            )
            return (
                np.linalg.norm(tangent_left - anchor_point)
                + radius * wrap_angle
                + np.linalg.norm(attachment - tangent_right)
            )

        finite_difference_step = 1.0e-2
        expected_gradient = (
            routed_length(angle + finite_difference_step) - routed_length(angle - finite_difference_step)
        ) / (2.0 * finite_difference_step)

        span_lengths = np.linalg.norm(attachment_r - attachment_l, axis=1)
        solver.tendon_seg_rest_length.assign(span_lengths - 0.01)
        solver.tendon_seg_lambda.zero_()
        solver.tendon_seg_delta_lambda.zero_()
        body_deltas = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=model.device)
        zero_inv_mass = wp.zeros(model.body_count, dtype=float, device=model.device)
        zero_inv_inertia = wp.zeros(model.body_count, dtype=wp.mat33, device=model.device)
        dt = 1.0 / 120.0

        wp.launch(
            kernel=solve_tendon_stretch,
            dim=model.tendon_segment_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                zero_inv_mass,
                zero_inv_inertia,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_offset,
                model.tendon_link_axis,
                solver.tendon_seg_rest_length,
                solver.tendon_seg_attachment_l,
                solver.tendon_seg_attachment_r,
                solver.tendon_seg_attachment_l_local,
                solver.tendon_seg_attachment_r_local,
                solver.tendon_seg_active_compliance,
                solver.tendon_seg_active_damping,
                solver.tendon_seg_lambda,
                solver.tendon_seg_delta_lambda,
                solver.tendon_seg_active,
                solver.tendon_seg_active_link_l,
                solver.tendon_seg_active_link_r,
                1.0 / dt,
                1.0,
                dt,
            ],
            outputs=[solver.tendon_seg_material_tension, body_deltas],
            device=model.device,
        )

        delta_lambda = solver.tendon_seg_delta_lambda.numpy()
        solved_lengths = np.linalg.norm(
            solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
        )
        expected_material_tension = np.maximum(solved_lengths - solver.tendon_seg_rest_length.numpy(), 0.0)
        expected_material_tension /= solver.tendon_seg_active_compliance.numpy()
        np.testing.assert_allclose(
            solver.tendon_seg_material_tension.numpy(),
            expected_material_tension,
            rtol=1.0e-5,
            atol=1.0e-5,
        )
        test.assertAlmostEqual(float(delta_lambda[0]), float(delta_lambda[1]), delta=1.0e-6)
        stretch_gradient = float(body_deltas.numpy()[body, 5] / np.mean(delta_lambda))
        test.assertAlmostEqual(stretch_gradient, expected_gradient, delta=1.0e-4)

        body_qd = np.zeros((model.body_count, 6), dtype=np.float32)
        body_qd[body, 5] = 1.0
        state.body_qd.assign(wp.array(body_qd, dtype=wp.spatial_vector, device=device))
        length_rates = wp.zeros(model.tendon_segment_count, dtype=float, device=model.device)
        wp.launch(
            kernel=_measure_tendon_segment_length_rates,
            dim=model.tendon_segment_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.tendon_link_body,
                model.tendon_link_type,
                model.tendon_link_offset,
                model.tendon_link_axis,
                solver.tendon_seg_active_link_l,
                solver.tendon_seg_active_link_r,
                solver.tendon_seg_attachment_l,
                solver.tendon_seg_attachment_r,
            ],
            outputs=[length_rates],
            device=model.device,
        )
        test.assertAlmostEqual(float(np.sum(length_rates.numpy())), expected_gradient, delta=1.0e-4)


def test_rolling_link_body_preserves_center_motion_jacobian(test, device):
    """A rolling endpoint must retain torque caused by motion of its center."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=(0.0, 0.0, 0.0))
        anchor = builder.add_body(xform=wp.transform(p=wp.vec3(0.5, -1.5, 0.0)), mass=0.0)
        link = builder.add_link(
            mass=1.0,
            inertia=wp.mat33(
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            lock_inertia=True,
        )
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=Axis.Z,
            target_ke=0.0,
            target_kd=0.0,
        )
        builder.add_articulation([joint])

        builder.add_tendon()
        builder.add_tendon_link(body=anchor, link_type=int(TendonLinkType.ATTACHMENT))
        builder.add_tendon_link(
            body=link,
            link_type=int(TendonLinkType.ROLLING),
            radius=0.2,
            orientation=-1,
            mu=0.0,
            offset=(2.5, 2.5, 0.0),
            compliance=1.0e-3,
            rest_length=-1.0,
        )
        builder.add_tendon_link(
            body=link,
            link_type=int(TendonLinkType.ATTACHMENT),
            offset=(1.5, -2.5, 0.0),
            compliance=1.0e-3,
            rest_length=-1.0,
        )

        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=1.0)
        span_lengths = np.linalg.norm(
            solver.tendon_seg_attachment_r.numpy() - solver.tendon_seg_attachment_l.numpy(), axis=1
        )
        solver.tendon_seg_rest_length.assign(span_lengths - 0.01)

        # The total routed length increases with positive joint rotation in this
        # geometry, so tension must drive the joint in the negative direction.
        state_0, state_1 = model.state(), model.state()
        for _ in range(4):
            state_0.clear_forces()
            solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
            state_0, state_1 = state_1, state_0

        newton.eval_ik(model, state_0, model.joint_q, model.joint_qd)
        joint_q = int(model.joint_q_start.numpy()[joint])
        test.assertLess(float(model.joint_q.numpy()[joint_q]), -1.0e-3)


def test_motorized_pulley_drives_slider(test, device):
    """A rolling drive pulley must convert rotation into cable sliding."""
    with wp.ScopedDevice(device):
        model, slider_idx, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=10.0)
        state = run_motorized_model(model, drive_joint)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite motorized pulley state")

        slider_x = float(body_q[slider_idx][0])
        q = body_q[pulley_idx]
        theta = abs(2.0 * np.arctan2(float(q[5]), float(q[6])))

        test.assertGreater(theta, 0.5, f"Drive pulley should rotate under its target: theta={theta:.4f}")
        test.assertGreater(slider_x, -0.2, f"No-slip drive should pull the slider through the cable: x={slider_x:.4f}")


def test_frictionless_motorized_pulley_does_not_drive_slider(test, device):
    """With mu=0, pulley spin should not inject cable sliding through rolling transfer."""
    with wp.ScopedDevice(device):
        model, slider_idx, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=0.0)
        state = run_motorized_model(model, drive_joint)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite frictionless motorized pulley state")

        slider_x = float(body_q[slider_idx][0])
        q = body_q[pulley_idx]
        theta = abs(2.0 * np.arctan2(float(q[5]), float(q[6])))

        test.assertGreater(theta, 0.5, f"Frictionless drive pulley should still rotate: theta={theta:.4f}")
        test.assertLess(
            abs(slider_x + 0.4),
            0.02,
            f"Frictionless pulley spin should not pull cable/slider: x={slider_x:.4f}",
        )


def test_motorized_pulley_couples_without_delay(test, device):
    """A driven pulley should move the cable during the initial rotation, not later."""
    with wp.ScopedDevice(device):
        model, slider_idx, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=10.0)
        solver = newton.solvers.SolverXPBD(model, iterations=12, joint_linear_relaxation=0.8)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        q_start = int(model.joint_q_start.numpy()[drive_joint])
        initial_x = float(state_0.body_q.numpy()[slider_idx][0])
        dt = 1.0 / 60.0 / 10.0
        for _ in range(30):
            control.joint_target_q[q_start : q_start + 1].fill_(1.0)
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        slider_dx = float(body_q[slider_idx][0]) - initial_x
        theta = abs(2.0 * np.arctan2(float(body_q[pulley_idx][5]), float(body_q[pulley_idx][6])))

        test.assertGreater(theta, 0.1, f"Pulley should have started rotating: theta={theta:.4f}")
        test.assertGreater(slider_dx, 0.02, f"Pulley rotation should immediately pull cable: dx={slider_dx:.4f}")


def test_motorized_pulley_updates_rest_in_first_step(test, device):
    """Rolling surface transfer should happen in the same XPBD step as pulley rotation."""
    with wp.ScopedDevice(device):
        model, _, pulley_idx, drive_joint = build_motorized_pulley_drive(mu=10.0)
        solver = newton.solvers.SolverXPBD(model, iterations=12, joint_linear_relaxation=0.8)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        q_start = int(model.joint_q_start.numpy()[drive_joint])
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        control.joint_target_q[q_start : q_start + 1].fill_(1.0)

        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 60.0 / 10.0)

        body_q = state_1.body_q.numpy()
        theta = abs(2.0 * np.arctan2(float(body_q[pulley_idx][5]), float(body_q[pulley_idx][6])))
        rest_delta = solver.tendon_seg_rest_length.numpy() - initial_rest

        test.assertGreater(theta, 1.0e-3, f"Drive pulley should rotate in the first step: theta={theta:.6f}")
        test.assertGreater(
            float(np.max(np.abs(rest_delta))),
            1.0e-4,
            f"Pulley rotation should transfer rolling rest length in the first step: delta={rest_delta}",
        )


def _moving_rolling_route_material_error(solver_factory, mu, short_segment=None):
    model, _ = build_kinematic_rolling_transport(mu=mu)
    endpoint = int(model.tendon_link_body.numpy()[2])
    start_angle = -0.2
    body_q = model.body_q.numpy()
    body_q[endpoint, :3] = np.array([0.4 * math.cos(start_angle), 0.4 * math.sin(start_angle), 0.0], dtype=np.float32)
    model.body_q.assign(body_q)

    solver = solver_factory(model)
    if short_segment is not None:
        rest = solver.tendon_seg_rest_length.numpy()
        total_rest = float(np.sum(rest))
        rest[short_segment] = 5.0e-3
        rest[1 - short_segment] = total_rest - rest[short_segment]
        solver.tendon_seg_rest_length.assign(rest)
        solver._snapshot_tendon_step_state()

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    radius = float(model.tendon_link_radius.numpy()[1])
    normal = model.tendon_link_axis.numpy()[1]

    def material_length():
        radial_l = solver.tendon_seg_attachment_r.numpy()[0]
        radial_r = solver.tendon_seg_attachment_l.numpy()[1]
        theta = abs(
            math.atan2(
                float(np.dot(np.cross(radial_l, radial_r), normal)),
                float(np.dot(radial_l, radial_r)),
            )
        )
        return float(np.sum(solver.tendon_seg_rest_length.numpy())) + theta * radius

    expected_material = material_length()
    max_material_error = 0.0
    min_short_rest = math.inf
    for angle in np.linspace(start_angle, 0.2, 21):
        body_q = state_0.body_q.numpy()
        body_q[endpoint, :3] = np.array([0.4 * math.cos(angle), 0.4 * math.sin(angle), 0.0], dtype=np.float32)
        state_0.body_q.assign(body_q)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
        max_material_error = max(max_material_error, abs(material_length() - expected_material))
        if short_segment is not None:
            min_short_rest = min(min_short_rest, float(solver.tendon_seg_rest_length.numpy()[short_segment]))
    return max_material_error, min_short_rest


def test_moving_rolling_route_conserves_material(test, device):
    """Changing a rolling link's wrap angle should not change total cable material."""
    with wp.ScopedDevice(device):

        def solver_factory(model):
            return newton.solvers.SolverXPBD(model, iterations=4)

        for mu in (0.0, 0.2, 100.0):
            with test.subTest(mu=mu):
                error, _ = _moving_rolling_route_material_error(solver_factory, mu)
                test.assertLess(error, 1.0e-4, f"Cable material drifted by {error} m for mu={mu}")

        for short_segment in (0, 1):
            with test.subTest(short_segment=short_segment):
                error, min_rest = _moving_rolling_route_material_error(solver_factory, 100.0, short_segment)
                test.assertLess(
                    error,
                    1.0e-4,
                    f"Cable material drifted by {error} m when segment {short_segment} started near its minimum rest length",
                )
                if short_segment == 1:
                    test.assertLess(min_rest, 1.1e-6, "The material-donor span should exercise the minimum-rest clamp")


def test_kinematic_rolling_transfer_independent_of_iterations(test, device):
    """Prescribed pulley spin should transfer the same material for any XPBD iteration count."""

    def run_once(iterations):
        model, pulley_idx = build_kinematic_rolling_transport(mu=10.0)
        solver = newton.solvers.SolverXPBD(model, iterations=iterations, joint_linear_relaxation=1.0)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        angle = 0.4
        body_q = state_0.body_q.numpy()
        body_q[pulley_idx, 3:] = np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)], dtype=np.float32)
        state_0.body_q.assign(body_q)

        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
        return solver.tendon_seg_rest_length.numpy() - initial_rest

    with wp.ScopedDevice(device):
        reference = run_once(1)
        test.assertGreater(
            float(np.max(np.abs(reference))),
            1.0e-4,
            f"Prescribed rolling spin should produce nonzero material transfer: delta={reference}",
        )
        for iterations in (2, 4, 8, 16):
            rest_delta = run_once(iterations)
            np.testing.assert_allclose(
                rest_delta,
                reference,
                rtol=1.0e-6,
                atol=1.0e-6,
                err_msg=(
                    "Rolling material transport should be a time-step update, not an XPBD iteration update: "
                    f"iterations={iterations}, reference={reference}, actual={rest_delta}"
                ),
            )


def test_kinematic_rolling_transfer_independent_of_cone_sweeps(test, device):
    """Prescribed pulley spin should transfer the same material for any cone sweep count.

    The rolling transport is a one-shot kinematic move set by the pulley geometry, not a
    convergent relaxation; refining the capstan cone must not erode it. ``tendon_settle_tol=0``
    forces the full ``tendon_max_sweeps`` budget so the executed count varies.
    """

    def run_once(tendon_max_sweeps):
        model, pulley_idx = build_kinematic_rolling_transport(mu=10.0)
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=8,
            joint_linear_relaxation=1.0,
            tendon_max_sweeps=tendon_max_sweeps,
            tendon_settle_tol=0.0,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        angle = 0.4
        body_q = state_0.body_q.numpy()
        body_q[pulley_idx, 3:] = np.array([0.0, 0.0, np.sin(0.5 * angle), np.cos(0.5 * angle)], dtype=np.float32)
        state_0.body_q.assign(body_q)

        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
        rest_delta = solver.tendon_seg_rest_length.numpy() - initial_rest
        sweep_count = int(solver.tendon_cone_sweep_count.numpy()[0])
        return rest_delta, sweep_count

    with wp.ScopedDevice(device):
        reference, reference_sweeps = run_once(1)
        test.assertGreater(
            float(np.max(np.abs(reference))),
            1.0e-4,
            f"Prescribed rolling spin should produce nonzero material transfer: delta={reference}",
        )
        test.assertEqual(reference_sweeps, 1)
        for tendon_max_sweeps in (2, 4, 8, 16, 64, 256):
            rest_delta, sweep_count = run_once(tendon_max_sweeps)
            np.testing.assert_allclose(
                rest_delta,
                reference,
                rtol=1.0e-5,
                atol=1.0e-6,
                err_msg=(
                    "Rolling material transport should be invariant to the capstan sweep count: "
                    f"tendon_max_sweeps={tendon_max_sweeps}, reference={reference}, actual={rest_delta}"
                ),
            )
            test.assertEqual(sweep_count, tendon_max_sweeps)


def test_frictionless_zero_span_equalizes_global_tension(test, device):
    """A zero-rest middle span should not split a frictionless tendon into tension islands."""
    with wp.ScopedDevice(device):
        model, initial_rest, compliance = build_frictionless_zero_span_route()
        # this test checks tight equalization (rtol 1e-4), so request a tight cone tolerance
        solver = newton.solvers.SolverXPBD(model, iterations=12, joint_linear_relaxation=1.0, tendon_settle_tol=1.0e-6)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        solver.step(state_0, state_1, control, None, 1.0 / 60.0)

        att_l = solver.tendon_seg_attachment_l.numpy()
        att_r = solver.tendon_seg_attachment_r.numpy()
        rest = solver.tendon_seg_rest_length.numpy()
        lengths = np.linalg.norm(att_r - att_l, axis=1)
        tensions = np.maximum(lengths - rest, 0.0) / compliance
        nonzero_span = lengths > 1.0e-5
        expected_tension = (
            (float(np.sum(lengths[nonzero_span])) - float(np.sum(initial_rest)))
            / int(np.count_nonzero(nonzero_span))
            / compliance
        )

        np.testing.assert_allclose(
            np.sum(rest),
            np.sum(initial_rest),
            rtol=1.0e-6,
            atol=1.0e-6,
            err_msg=f"Frictionless serial traversal should preserve total free-span rest length: {rest}",
        )
        np.testing.assert_allclose(
            tensions[nonzero_span],
            expected_tension,
            rtol=1.0e-4,
            atol=1.0,
            err_msg=f"Frictionless nonzero spans should share one global tension: {tensions}, rest={rest}",
        )


def test_finite_friction_zero_span_respects_global_capstan_cone(test, device):
    """A depleted middle span should not strand finite-friction tension islands."""
    with wp.ScopedDevice(device):
        mu = 0.2
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (2.0, 0.0, 1.0),
        ]
        initial_rest = np.asarray([0.5, 1.0e-6, 0.9, 0.9], dtype=np.float32)
        model, initial_rest, compliance = build_frictionless_zero_span_route(
            mu=mu,
            points=points,
            rest_lengths=initial_rest,
        )
        solver = newton.solvers.SolverXPBD(model, iterations=12, joint_linear_relaxation=1.0)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        solver.step(state_0, state_1, control, None, 1.0 / 60.0)

        att_l = solver.tendon_seg_attachment_l.numpy()
        att_r = solver.tendon_seg_attachment_r.numpy()
        rest = solver.tendon_seg_rest_length.numpy()
        lengths = np.linalg.norm(att_r - att_l, axis=1)
        tensions = np.maximum(lengths - rest, 0.0) / compliance
        nonzero_tensions = tensions[lengths > 1.0e-5]
        capstan_ratio = np.exp(mu * np.pi)

        np.testing.assert_allclose(
            np.sum(rest),
            np.sum(initial_rest),
            rtol=1.0e-6,
            atol=1.0e-6,
            err_msg=f"Finite-friction serial traversal should preserve total free-span rest length: {rest}",
        )
        for left, right in pairwise(nonzero_tensions):
            test.assertLessEqual(
                left,
                capstan_ratio * right + 1.0,
                f"Left tension escaped finite capstan cone: tensions={tensions}, rest={rest}",
            )
            test.assertLessEqual(
                right,
                capstan_ratio * left + 1.0,
                f"Right tension escaped finite capstan cone: tensions={tensions}, rest={rest}",
            )


def test_stiff_pinhole_capstan_matches_euler_eytelwein(test, device):
    """A stiff (low-compliance) cable wrapping a pinhole pulley must still obey the capstan
    equation: under loading the anchor/slider tension ratio approaches exp(-mu*pi).

    Regression for the stiff-cable under-shoot: when the per-segment stretch d = len - rest is
    recomputed as a difference of ~1e-3 m lengths every relaxation sweep, the ~1e-9 m friction
    transfers vanish into float32 cancellation and the cone is over-applied (ratio collapses to
    ~0.5 instead of 0.73). Tracking d as precise state across the sweeps restores the bound.
    """
    with wp.ScopedDevice(device):
        mu, fmax, substeps, fps = 0.1, 10.0, 16, 12
        model, slider = build_stiff_pinhole_capstan(n_pinholes=9, mu=mu, compliance=2.0e-8)
        solver = newton.solvers.SolverXPBD(model, iterations=32, joint_linear_relaxation=1.0)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        comp = model.tendon_seg_compliance.numpy()

        frame_dt = 1.0 / fps
        sub_dt = frame_dt / substeps
        n_frames = fps  # ramp force 0 -> fmax over 1 s (loading only)
        body_f = np.zeros((model.body_count, 6), dtype=np.float32)
        ratios = []
        for frame in range(n_frames):
            f_cmd = fmax * (frame + 1) / n_frames
            for _ in range(substeps):
                state_0.clear_forces()
                body_f[slider] = (0.0, 0.0, -f_cmd, 0.0, 0.0, 0.0)  # pull slider along -Z
                state_0.body_f.assign(wp.array(body_f, dtype=wp.spatial_vector))
                solver.step(state_0, state_1, control, contacts, sub_dt)
                state_0, state_1 = state_1, state_0
            if 0.6 * fmax <= f_cmd <= 0.95 * fmax:
                att_l = solver.tendon_seg_attachment_l.numpy()
                att_r = solver.tendon_seg_attachment_r.numpy()
                rest = solver.tendon_seg_rest_length.numpy()
                tension = np.maximum(np.linalg.norm(att_r - att_l, axis=1) - rest, 0.0) / comp
                ratios.append(tension[0] / max(tension[-1], 1.0e-9))  # T_anchor / T_slider

        ratio = float(np.mean(ratios))
        target = math.exp(-mu * math.pi)  # 0.7304
        # broken kernel collapses to ~0.5 (over-friction); fixed tracks the capstan bound.
        test.assertGreater(
            ratio, 0.66, f"stiff capstan under-shoots Euler-Eytelwein: ratio={ratio:.4f}, target={target:.4f}"
        )
        test.assertLess(
            ratio, 0.81, f"stiff capstan over-shoots Euler-Eytelwein: ratio={ratio:.4f}, target={target:.4f}"
        )


def test_settle_tol_early_out_matches_full_sweeps(test, device):
    """The settle-based cone early-out must not change the converged capstan result.

    The relaxation stops once the per-sweep relative tension change drops below
    ``tendon_settle_tol``; the default (1e-3) must reach the same Euler-Eytelwein bound as
    forcing the full sweep budget (``tendon_settle_tol=0``). Regression for an early-out that
    fires before the cone has settled and leaves the cable under-relaxed.
    """
    with wp.ScopedDevice(device):
        mu, fmax, substeps, fps = 0.1, 10.0, 24, 12

        def run(settle_tol, max_sweeps):
            model, slider = build_stiff_pinhole_capstan(n_pinholes=7, mu=mu, compliance=2.0e-8)
            solver = newton.solvers.SolverXPBD(
                model,
                iterations=32,
                joint_linear_relaxation=1.0,
                tendon_max_sweeps=max_sweeps,
                tendon_settle_tol=settle_tol,
            )
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            collision_pipeline = newton.CollisionPipeline(model)
            contacts = collision_pipeline.contacts()
            comp = model.tendon_seg_compliance.numpy()
            sub_dt = (1.0 / fps) / substeps
            body_f = np.zeros((model.body_count, 6), dtype=np.float32)
            ratios = []
            for frame in range(fps):  # ramp force 0 -> fmax over 1 s (loading only)
                f_cmd = fmax * (frame + 1) / fps
                for _ in range(substeps):
                    state_0.clear_forces()
                    body_f[slider] = (0.0, 0.0, -f_cmd, 0.0, 0.0, 0.0)
                    state_0.body_f.assign(wp.array(body_f, dtype=wp.spatial_vector))
                    solver.step(state_0, state_1, control, contacts, sub_dt)
                    state_0, state_1 = state_1, state_0
                if 0.6 * fmax <= f_cmd <= 0.95 * fmax:
                    att_l = solver.tendon_seg_attachment_l.numpy()
                    att_r = solver.tendon_seg_attachment_r.numpy()
                    rest = solver.tendon_seg_rest_length.numpy()
                    tension = np.maximum(np.linalg.norm(att_r - att_l, axis=1) - rest, 0.0) / comp
                    ratios.append(tension[0] / max(tension[-1], 1.0e-9))  # T_anchor / T_slider
            return float(np.mean(ratios)), int(solver.tendon_cone_sweep_count.numpy()[0])

        target = math.exp(-mu * math.pi)  # 0.7304
        early, early_sweeps = run(1.0e-3, 256)
        full, full_sweeps = run(0.0, 64)
        test.assertAlmostEqual(
            early, full, delta=0.02, msg=f"early-out ratio {early:.4f} should match full sweeps {full:.4f}"
        )
        test.assertLess(early_sweeps, 256, "Default tolerance should stop before the sweep cap")
        test.assertEqual(full_sweeps, 64, "Zero tolerance should run the requested full sweep budget")
        test.assertGreater(early, 0.66, f"early-out under-shoots capstan bound: {early:.4f}, target={target:.4f}")
        test.assertLess(early, 0.81, f"early-out over-shoots capstan bound: {early:.4f}, target={target:.4f}")


def test_slack_tendon_settles_without_chasing_vanishing_tension(test, device):
    """An almost-slack tendon should not chase a vanishing relative tension scale."""
    with wp.ScopedDevice(device):

        def run(settle_tol, max_sweeps):
            model, initial_rest, _ = build_frictionless_zero_span_route(
                compliance=1.0e-4,
                points=[
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (2.0, 0.0, 0.0),
                    (3.0, 0.0, 0.0),
                ],
                rest_lengths=[1.001, 0.9999999, 1.001],
            )
            solver = newton.solvers.SolverXPBD(
                model,
                iterations=1,
                tendon_max_sweeps=max_sweeps,
                tendon_settle_tol=settle_tol,
            )
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
            return (
                solver.tendon_seg_rest_length.numpy(),
                int(solver.tendon_cone_sweep_count.numpy()[0]),
                initial_rest,
            )

        early_rest, early_sweeps, initial_rest = run(1.0e-3, 16)
        full_rest, full_sweeps, _ = run(0.0, 64)

        test.assertLess(early_sweeps, 16, f"Slack tendon should settle before the budget, got {early_sweeps} sweeps")
        test.assertEqual(full_sweeps, 64, "Zero tolerance should run the requested full sweep budget")
        np.testing.assert_allclose(early_rest, full_rest, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(np.sum(early_rest), np.sum(initial_rest), rtol=1.0e-6, atol=1.0e-6)


devices = ["cpu"]
if wp.is_cuda_available():
    devices.append("cuda:0")
cuda_devices = [device for device in devices if device.startswith("cuda")]

add_test(TestTendonCapstan, "pinhole_slip_atwood", devices, test_pinhole_slip_atwood)
add_test(
    TestTendonCapstan,
    "frictional_pinhole_mu_controls_slip_and_locking",
    devices,
    test_frictional_pinhole_mu_controls_slip_and_locking,
)
add_test(TestTendonCapstan, "slack_pinhole_does_not_redistribute", devices, test_slack_pinhole_does_not_redistribute)
add_test(TestTendonCapstan, "dynamic_pulley_uses_angular_jacobian", devices, test_dynamic_pulley_uses_angular_jacobian)
add_test(
    TestTendonCapstan, "pulley_inertia_limit_locks_cable_travel", devices, test_pulley_inertia_limit_locks_cable_travel
)
add_test(
    TestTendonCapstan,
    "dynamic_capstan_mu_controls_pulley_rotation",
    devices,
    test_dynamic_capstan_mu_controls_pulley_rotation,
)
add_test(
    TestTendonCapstan,
    "kinematic_capstan_mu_controls_slip_and_locking",
    devices,
    test_kinematic_capstan_mu_controls_slip_and_locking,
)
add_test(
    TestTendonCapstan,
    "kinematic_capstan_hysteresis_matches_capstan_band",
    devices,
    test_kinematic_capstan_hysteresis_matches_capstan_band,
)
add_test(
    TestTendonCapstan,
    "pinhole_capstan_force_mode_uses_physical_compliance",
    devices,
    test_pinhole_capstan_force_mode_uses_physical_compliance,
)
add_test(
    TestTendonCapstan,
    "simple_cable_gravity_balances_mass_load",
    devices,
    test_simple_cable_gravity_balances_mass_load,
)
add_test(
    TestTendonCapstan,
    "simple_cable_gravity_tension_independent_of_relaxation",
    devices,
    test_simple_cable_gravity_tension_independent_of_relaxation,
)
add_test(
    TestTendonCapstan,
    "frictionless_pinhole_equalizes_damped_tension",
    devices,
    test_frictionless_pinhole_equalizes_damped_tension,
)
add_test(
    TestTendonCapstan,
    "tendon_damping_does_not_generate_compression",
    devices,
    test_tendon_damping_does_not_generate_compression,
)
add_test(
    TestTendonCapstan,
    "same_body_tendon_segment_reports_constitutive_tension",
    devices,
    test_same_body_tendon_segment_reports_constitutive_tension,
)
add_test(
    TestTendonCapstan,
    "same_body_rolling_route_matches_length_gradient",
    devices,
    test_same_body_rolling_route_matches_length_gradient,
)
add_test(
    TestTendonCapstan,
    "rolling_link_body_preserves_center_motion_jacobian",
    devices,
    test_rolling_link_body_preserves_center_motion_jacobian,
)
add_test(
    TestTendonCapstan,
    "force_driven_dynamic_route_updates_inside_solver",
    devices,
    test_force_driven_dynamic_route_updates_inside_solver,
)
add_test(
    TestTendonCapstan,
    "dynamic_route_activation_uses_accepted_pose",
    devices,
    test_dynamic_route_activation_uses_accepted_pose,
)
add_test(
    TestTendonCapstan,
    "xpbd_tendon_diagnostics_match_final_pose",
    devices,
    test_xpbd_tendon_diagnostics_match_final_pose,
)
add_test(
    TestTendonCapstan,
    "dynamic_route_neighbor_matrix_conserves_material",
    devices,
    test_dynamic_route_neighbor_matrix_conserves_material,
)
add_test(
    TestTendonCapstan,
    "dynamic_route_initial_state_matches_geometry",
    devices,
    test_dynamic_route_initial_state_matches_geometry,
)
add_test(
    TestTendonCapstan,
    "inactive_dynamic_route_preserves_authored_rest_length",
    devices,
    test_inactive_dynamic_route_preserves_authored_rest_length,
)
add_test(
    TestTendonCapstan,
    "inactive_dynamic_route_preserves_mixed_rest_length",
    devices,
    test_inactive_dynamic_route_preserves_mixed_rest_length,
)
add_test(
    TestTendonCapstan,
    "xpbd_reset_restores_tendon_material_state",
    devices,
    test_xpbd_reset_restores_tendon_material_state,
)
add_test(
    TestTendonCapstan,
    "dynamic_route_uses_oriented_signed_distance",
    devices,
    test_dynamic_route_uses_oriented_signed_distance,
)
add_test(
    TestTendonCapstan,
    "dynamic_route_activation_tolerance_preserves_boundary_state",
    devices,
    test_dynamic_route_activation_tolerance_preserves_boundary_state,
)
add_test(
    TestTendonCapstan,
    "dynamic_route_requires_projection_on_bypass_span",
    devices,
    test_dynamic_route_requires_projection_on_bypass_span,
)
add_test(TestTendonCapstan, "fixed_route_state_is_preserved", devices, test_fixed_route_state_is_preserved)
add_test(TestTendonCapstan, "dynamic_route_cuda_graph_capture", cuda_devices, test_dynamic_route_cuda_graph_capture)
add_test(
    TestTendonCapstan,
    "loaded_dynamic_route_matches_fixed_route",
    devices,
    test_loaded_dynamic_route_matches_fixed_route,
)
add_test(
    TestTendonCapstan,
    "mujoco_switch_optional_middle_capstan_activates",
    devices,
    test_mujoco_switch_optional_middle_capstan_activates,
)
add_test(
    TestTendonCapstan,
    "mujoco_switch_preserves_active_route_segments",
    devices,
    test_mujoco_switch_preserves_active_route_segments,
)
add_test(
    TestTendonCapstan,
    "tendon_slip_uses_active_route_segments",
    devices,
    test_tendon_slip_uses_active_route_segments,
)
add_test(TestTendonCapstan, "motorized_pulley_drives_slider", devices, test_motorized_pulley_drives_slider)
add_test(
    TestTendonCapstan,
    "frictionless_motorized_pulley_does_not_drive_slider",
    devices,
    test_frictionless_motorized_pulley_does_not_drive_slider,
)
add_test(
    TestTendonCapstan, "motorized_pulley_couples_without_delay", devices, test_motorized_pulley_couples_without_delay
)
add_test(
    TestTendonCapstan,
    "motorized_pulley_updates_rest_in_first_step",
    devices,
    test_motorized_pulley_updates_rest_in_first_step,
)
add_test(
    TestTendonCapstan,
    "moving_rolling_route_conserves_material",
    devices,
    test_moving_rolling_route_conserves_material,
)
add_test(
    TestTendonCapstan,
    "kinematic_rolling_transfer_independent_of_iterations",
    devices,
    test_kinematic_rolling_transfer_independent_of_iterations,
)
add_test(
    TestTendonCapstan,
    "kinematic_rolling_transfer_independent_of_cone_sweeps",
    devices,
    test_kinematic_rolling_transfer_independent_of_cone_sweeps,
)
add_test(
    TestTendonCapstan,
    "frictionless_zero_span_equalizes_global_tension",
    devices,
    test_frictionless_zero_span_equalizes_global_tension,
)
add_test(
    TestTendonCapstan,
    "finite_friction_zero_span_respects_global_capstan_cone",
    devices,
    test_finite_friction_zero_span_respects_global_capstan_cone,
)
add_test(
    TestTendonCapstan,
    "stiff_pinhole_capstan_matches_euler_eytelwein",
    devices,
    test_stiff_pinhole_capstan_matches_euler_eytelwein,
)
add_test(
    TestTendonCapstan,
    "settle_tol_early_out_matches_full_sweeps",
    devices,
    test_settle_tol_early_out_matches_full_sweeps,
)
add_test(
    TestTendonCapstan,
    "slack_tendon_settles_without_chasing_vanishing_tension",
    devices,
    test_slack_tendon_settles_without_chasing_vanishing_tension,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
