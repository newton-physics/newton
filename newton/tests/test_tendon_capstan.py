# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Cable Joints tendon solver.

These tests cover the routed-cable baseline and the first finite-slip capstan
acceptance criteria.  Test expectation changes in this file should follow
``docs/cable_joints_slip_plan.md``.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.builder import Axis
from newton._src.sim.tendon import TendonLinkType
from newton.examples.cable.example_tendon_capstan_friction import Example as DynamicCapstanExample
from newton.tests.unittest_utils import sanitize_identifier


def add_test(cls, name, devices, test_fn):
    for device in devices:
        test_name = f"test_{sanitize_identifier(name)}_{sanitize_identifier(device)}"
        setattr(cls, test_name, lambda self, d=device, fn=test_fn: fn(self, d))


def _box_on_planar_joint(builder, pos, mass, half_extent):
    dof = newton.ModelBuilder.JointDofConfig
    body = builder.add_link(xform=wp.transform(p=pos), mass=mass)
    builder.add_shape_box(body, hx=half_extent, hy=half_extent, hz=half_extent)
    joint = builder.add_joint_d6(
        parent=-1,
        child=body,
        linear_axes=[dof(axis=Axis.X), dof(axis=Axis.Z)],
        angular_axes=[dof(axis=Axis.Y)],
        parent_xform=wp.transform(p=pos),
        child_xform=wp.transform(),
    )
    builder.add_articulation([joint])
    return body


def build_pinhole_atwood(mass_left=1.0, mass_right=3.0, mu=0.0):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

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
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )

    builder.add_ground_plane()
    return builder.finalize(), left, right


def build_slack_pinhole_route():
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=0.0)

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


def build_dynamic_pulley_atwood(mu=10.0, mass_left=1.0, mass_right=3.0, pulley_mass=5.0, pulley_radius=0.15):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

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
    pulley = builder.add_body(
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

    left = _box_on_planar_joint(builder, wp.vec3(-0.4, 0.0, 2.0), mass_left, 0.06)
    right = _box_on_planar_joint(builder, wp.vec3(0.4, 0.0, 2.0), mass_right, 0.06)

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
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )

    builder.add_ground_plane()
    return builder.finalize(), left, right, pulley


def build_kinematic_pulley_atwood(mu=0.0, mass_left=1.0, mass_right=3.0, pulley_radius=0.15):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=-9.81)

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

    left = _box_on_planar_joint(builder, wp.vec3(-0.4, 0.0, 2.0), mass_left, 0.06)
    right = _box_on_planar_joint(builder, wp.vec3(0.4, 0.0, 2.0), mass_right, 0.06)

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
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )
    builder.add_tendon_link(
        body=right,
        link_type=int(TendonLinkType.ATTACHMENT),
        offset=(0.0, 0.0, 0.06),
        axis=axis,
        compliance=1.0e-5,
        damping=0.1,
        rest_length=-1.0,
    )

    builder.add_ground_plane()
    return builder.finalize(), left, right, pulley


def build_motorized_pulley_drive(mu=0.0):
    builder = newton.ModelBuilder(up_axis=Axis.Z, gravity=0.0)
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


def run_model(model, num_frames=80, substeps=12, fps=60):
    dt = 1.0 / fps / substeps
    solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=0.8)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    return state_0


def run_motorized_model(model, drive_joint, target=2.0, num_frames=70, substeps=10, fps=60):
    dt = 1.0 / fps / substeps
    solver = newton.solvers.SolverXPBD(model, iterations=12, joint_linear_relaxation=0.8)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dof_start = int(model.joint_qd_start.numpy()[drive_joint])
    for _ in range(num_frames):
        control.joint_target_pos[dof_start : dof_start + 1].fill_(target)
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0

    return state_0


class TestTendonCapstan(unittest.TestCase):
    pass


def _hinge_y_angle(body_q, body_idx):
    q = body_q[body_idx]
    return float(2.0 * np.arctan2(float(q[4]), float(q[6])))


def _dynamic_capstan_metrics(device, mu, num_frames=40):
    model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=mu)
    substeps = 12
    dt = 1.0 / 60.0 / substeps
    solver = newton.solvers.SolverXPBD(model, iterations=8, joint_linear_relaxation=0.8)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    body_q = state_0.body_q.numpy()
    last_angle = _hinge_y_angle(body_q, pulley_idx)
    theta = 0.0

    for _ in range(num_frames):
        for _ in range(substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
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


def _dynamic_capstan_example_theta(device, num_frames=40):
    example = DynamicCapstanExample(None, None)
    for _ in range(num_frames):
        example.step()
    theta = np.array(example._pulley_rotation_history[-1], dtype=np.float64)
    return tuple(example.mus), theta


def _kinematic_capstan_metrics(device, mu, num_frames=100):
    model, left_idx, right_idx, pulley_idx = build_kinematic_pulley_atwood(mu=mu)
    state = run_model(model, num_frames=num_frames)
    body_q = state.body_q.numpy()
    left_travel = float(body_q[left_idx][2]) - 2.0
    right_travel = 2.0 - float(body_q[right_idx][2])
    theta = _hinge_y_angle(body_q, pulley_idx)
    return body_q, left_travel, right_travel, theta


def test_pinhole_slip_atwood(test, device):
    """A pinhole is a frictionless slip waypoint: heavy descends, light rises."""
    with wp.ScopedDevice(device):
        model, left_idx, right_idx = build_pinhole_atwood()
        state = run_model(model, num_frames=80)
        body_q = state.body_q.numpy()
        test.assertTrue(np.isfinite(body_q).all(), "Non-finite pinhole Atwood state")

        left_z = float(body_q[left_idx][2])
        right_z = float(body_q[right_idx][2])
        test.assertGreater(left_z, 2.05, f"Light side should rise through pinhole slip: z={left_z:.4f}")
        test.assertLess(right_z, 1.95, f"Heavy side should descend through pinhole slip: z={right_z:.4f}")


def _pinhole_friction_metrics(mu, num_frames=80):
    model, left_idx, right_idx = build_pinhole_atwood(mass_left=1.0, mass_right=3.0, mu=mu)
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
        contacts = model.contacts()

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
        model, left_idx, right_idx, pulley_idx = build_dynamic_pulley_atwood(mu=10.0)
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
            test.assertGreater(left_travel, -0.03, f"Light side should not sink for {label} mu: {left_travel:.5f}")

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
            slip_low, slip_mid, f"Dynamic slip should decrease from low to mid mu: {slip_low:.5f} <= {slip_mid:.5f}"
        )
        test.assertGreater(
            slip_mid, slip_high, f"Dynamic slip should decrease from mid to high mu: {slip_mid:.5f} <= {slip_high:.5f}"
        )
        test.assertGreater(rim_mid, 0.0, f"Mid-mu rim travel should be positive: {rim_mid:.5f}")


def test_dynamic_capstan_example_mid_mu_stays_below_high_mu(test, device):
    """The rendered finite-friction dynamic capstan case should remain visually distinct from no-slip."""
    with wp.ScopedDevice(device):
        mus, theta = _dynamic_capstan_example_theta(device)
        test.assertEqual(mus[0], 0.0, f"Dynamic capstan example should keep the zero-friction case first: {mus}")
        test.assertGreater(mus[1], 0.0, f"Dynamic capstan example mid friction should be finite: {mus}")
        test.assertGreaterEqual(mus[2], 10.0, f"Dynamic capstan example high friction should be no-slip-like: {mus}")

        theta_low, theta_mid, theta_high = theta
        test.assertLess(abs(theta_low), 0.08, f"Example zero-mu pulley should not rotate: theta={theta_low:.5f}")
        test.assertGreater(theta_mid, 0.25, f"Example mid-mu pulley should rotate in cable direction: {theta_mid:.5f}")
        test.assertLess(
            theta_mid,
            0.75 * theta_high,
            f"Example mid-mu pulley should stay visibly below high-friction/no-slip rotation: "
            f"mid={theta_mid:.5f}, high={theta_high:.5f}, mus={mus}",
        )


def test_kinematic_capstan_mu_controls_slip_and_locking(test, device):
    """Kinematic capstan: zero mu slips, mid mu slips less, high mu locks."""
    with wp.ScopedDevice(device):
        low = _kinematic_capstan_metrics(device, mu=0.0)
        mid = _kinematic_capstan_metrics(device, mu=0.08)
        high = _kinematic_capstan_metrics(device, mu=10.0)

        for label, metrics in [("low", low), ("mid", mid), ("high", high)]:
            body_q, _, _, theta = metrics
            test.assertTrue(np.isfinite(body_q).all(), f"Non-finite kinematic capstan state for {label} mu")
            test.assertLess(abs(theta), 1.0e-5, f"Kinematic pulley should not rotate for {label} mu: {theta:.6f}")

        _, left_low, right_low, _ = low
        _, _, right_mid, _ = mid
        _, _, right_high, _ = high

        test.assertGreater(right_low, 0.18, f"Zero-mu kinematic capstan should freely slip: dz={right_low:.5f}")
        test.assertGreater(left_low, 0.08, f"Zero-mu light side should rise through slip: dz={left_low:.5f}")
        test.assertGreater(
            right_low, right_mid + 0.03, f"Mid mu should slip less than zero mu: {right_low:.5f} vs {right_mid:.5f}"
        )
        test.assertGreater(
            right_mid, right_high + 0.03, f"High mu should lock more than mid mu: {right_mid:.5f} vs {right_high:.5f}"
        )
        test.assertLess(right_high, 0.06, f"High-mu kinematic capstan should lock cable motion: dz={right_high:.5f}")


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
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        dof_start = int(model.joint_qd_start.numpy()[drive_joint])
        initial_x = float(state_0.body_q.numpy()[slider_idx][0])
        dt = 1.0 / 60.0 / 10.0
        for _ in range(30):
            control.joint_target_pos[dof_start : dof_start + 1].fill_(1.0)
            state_0.clear_forces()
            model.collide(state_0, contacts)
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
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        dof_start = int(model.joint_qd_start.numpy()[drive_joint])
        initial_rest = solver.tendon_seg_rest_length.numpy().copy()
        control.joint_target_pos[dof_start : dof_start + 1].fill_(1.0)

        state_0.clear_forces()
        model.collide(state_0, contacts)
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


def test_rolling_transfer_saturates_at_zero_span(test, device):
    """Rolling transfer should clamp before a free span goes negative."""
    with wp.ScopedDevice(device):
        model, slider_idx, _, drive_joint = build_motorized_pulley_drive(mu=10.0)
        solver = newton.solvers.SolverXPBD(model, iterations=12, joint_linear_relaxation=0.8)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        dof_start = int(model.joint_qd_start.numpy()[drive_joint])
        dt = 1.0 / 60.0 / 10.0
        saturated_x = None

        for _frame in range(60):
            control.joint_target_pos[dof_start : dof_start + 1].fill_(8.0)
            for _ in range(10):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            rest = solver.tendon_seg_rest_length.numpy()
            if saturated_x is None and np.min(rest) <= 1.1e-6:
                body_q = state_0.body_q.numpy()
                saturated_x = float(body_q[slider_idx][0])

        body_q = state_0.body_q.numpy()
        final_x = float(body_q[slider_idx][0])
        rest = solver.tendon_seg_rest_length.numpy()

        test.assertIsNotNone(saturated_x, "Driven pulley should exhaust one adjacent free span")
        test.assertGreaterEqual(float(np.min(rest)), 0.99e-6, f"Rest lengths must stay non-negative: {rest}")
        test.assertLess(
            abs(final_x - saturated_x),
            1.0e-2,
            f"Slider should lock once a free span is exhausted: {final_x:.6f} vs {saturated_x:.6f}",
        )


devices = ["cpu"]
if wp.is_cuda_available():
    devices.append("cuda:0")

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
    "dynamic_capstan_example_mid_mu_stays_below_high_mu",
    devices,
    test_dynamic_capstan_example_mid_mu_stays_below_high_mu,
)
add_test(
    TestTendonCapstan,
    "kinematic_capstan_mu_controls_slip_and_locking",
    devices,
    test_kinematic_capstan_mu_controls_slip_and_locking,
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
    TestTendonCapstan, "rolling_transfer_saturates_at_zero_span", devices, test_rolling_transfer_saturates_at_zero_span
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
