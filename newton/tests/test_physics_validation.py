# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestPhysicsValidation(unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# Test 1: Free Fall
# Verify free-fall trajectory against y(t) = h0 + 0.5*g*t^2 and v(t) = g*t.
# ---------------------------------------------------------------------------
def test_free_fall(test, device, solver_fn):
    # Test parameters: gravity and initial height
    g = -10.0
    h0 = 5.0

    # Add a sphere
    builder = newton.ModelBuilder(gravity=g, up_axis=newton.Axis.Y)
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, h0, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(b, radius=0.1)
    model = builder.finalize(device=device)

    solver = solver_fn(model)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    sim_dt = 1e-3
    num_steps = 500
    check_steps = [100, 200, 300, 400, 500]
    for i in range(1, num_steps + 1):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

        if i in check_steps:
            # Checkpoint to verify correct simulation
            t = i * sim_dt
            pos = state_0.body_q.numpy()[0][:3]
            vel = state_0.body_qd.numpy()[0][:3]
            expected_pos = h0 + 0.5 * g * t * t
            expected_vel = g * t

            # Tolerance accounts for first-order integration error: ~0.5*|g|*dt*t
            integration_error = 0.5 * abs(g) * sim_dt * t
            pos_tol = max(2.0 * integration_error, 1e-3)
            vel_tol = max(abs(g) * sim_dt, 1e-3)

            test.assertAlmostEqual(
                pos[1],
                expected_pos,
                delta=pos_tol,
                msg=f"Free fall position at t={t:.3f}: got {pos[1]:.6f}, expected {expected_pos:.6f}",
            )
            test.assertAlmostEqual(
                vel[1],
                expected_vel,
                delta=vel_tol,
                msg=f"Free fall velocity at t={t:.3f}: got {vel[1]:.6f}, expected {expected_vel:.6f}",
            )

            # Horizontal components should remain zero
            test.assertAlmostEqual(pos[0], 0.0, delta=1e-4, msg=f"X drift at t={t:.3f}")
            test.assertAlmostEqual(pos[2], 0.0, delta=1e-4, msg=f"Z drift at t={t:.3f}")


# ---------------------------------------------------------------------------
# Test 2: Pendulum Period
# Verify pendulum trajectory against the analytical solution:
#             theta = theta_0*cos(2*pi*t/T)
# where T = 2*pi*sqrt(I_pivot / (m*g*d)) is the pendulum period,
# theta_0 is the initial amplitude, and t is time.
# ---------------------------------------------------------------------------
def test_pendulum_period(test, device, solver_fn, uses_generalized_coords, sim_dt=1e-3, sphere_radius=0.01):
    # Test parameters: gravity, pendulum length and initial angle
    g = -10.0
    L = 1.0
    initial_angle = 0.05  # small angle to keep analytical solution valid

    # Add a sphere
    builder = newton.ModelBuilder(gravity=g, up_axis=newton.Axis.Y)
    link = builder.add_link()
    builder.add_shape_sphere(link, radius=sphere_radius)
    j = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, L, 0.0), wp.quat_identity()),
        armature=0.0,
    )
    builder.add_articulation([j])
    model = builder.finalize(device=device)

    # Set initial angle
    q_init = model.joint_q.numpy().copy()
    q_start = model.joint_q_start.numpy()
    qi = q_start[0]
    q_init[qi] = initial_angle
    model.joint_q.assign(q_init)

    state_0 = model.state()
    state_1 = model.state()
    state_0.joint_q.assign(model.joint_q)
    state_0.joint_qd.assign(model.joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    solver = solver_fn(model)

    mass = model.body_mass.numpy()[0]
    I_cm = model.body_inertia.numpy()[0]
    I_cm_zz = I_cm[2, 2] if I_cm.ndim == 2 else I_cm[2]
    I_pivot = I_cm_zz + mass * L * L
    expected_T = 2.0 * np.pi * np.sqrt(I_pivot / (mass * abs(g) * L))

    # Simulate for ~3 full periods
    num_steps = int(3.5 * expected_T / sim_dt)

    angles = []
    for _ in range(num_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

        if uses_generalized_coords:
            angles.append(float(state_0.joint_q.numpy()[qi]))
        else:
            # Maximal-coordinate solvers don't update joint_q; recover angle from body position
            bq = state_0.body_q.numpy()[0]
            angles.append(float(np.arctan2(bq[0], -bq[1])))

    angles = np.array(angles)
    times = np.arange(1, num_steps + 1) * sim_dt

    omega = 2.0 * np.pi / expected_T
    analytical_angles = initial_angle * np.cos(omega * times)
    trajectory_error = np.mean(np.abs(angles - analytical_angles)) / abs(initial_angle)
    test.assertLess(
        trajectory_error,
        0.01,
        f"Pendulum trajectory error {trajectory_error:.4f} exceeds 1% of amplitude",
    )


# ---------------------------------------------------------------------------
# Test 3: Energy Conservation
# Verify total energy KE + PE stays constant for an undamped pendulum.
# Energy is computed as:
#        KE = 0.5 * I_pivot * theta_dot^2
#        PE = m * g * (-L * cos(theta))
#    where I_pivot = I_cm_zz + m * L^2 (parallel axis theorem).
# ---------------------------------------------------------------------------
def test_energy_conservation(test, device, solver_fn, uses_generalized_coords, sim_dt=1e-3, sphere_radius=0.01):
    # Test parameters: gravity, pendulum length and initial angle
    g = -10.0
    L = 1.0
    initial_angle = 1.0

    # Create pendulum
    builder = newton.ModelBuilder(gravity=g, up_axis=newton.Axis.Y)
    link = builder.add_link()
    builder.add_shape_sphere(link, radius=sphere_radius)
    j = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, L, 0.0), wp.quat_identity()),
        armature=0.0,
    )
    builder.add_articulation([j])
    model = builder.finalize(device=device)

    # Set initial angle
    q_init = model.joint_q.numpy().copy()
    q_start = model.joint_q_start.numpy()
    qi = q_start[0]
    q_init[qi] = initial_angle
    model.joint_q.assign(q_init)

    state_0 = model.state()
    state_1 = model.state()
    state_0.joint_q.assign(model.joint_q)
    state_0.joint_qd.assign(model.joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    solver = solver_fn(model)

    mass = float(model.body_mass.numpy()[0])
    I_body = model.body_inertia.numpy()[0]
    I_cm_zz = float(I_body[2, 2] if I_body.ndim == 2 else I_body[2])
    I_pivot = I_cm_zz + mass * L * L

    def compute_ke_pe(state):
        if uses_generalized_coords:
            theta = float(state.joint_q.numpy()[qi])
            theta_dot = float(state.joint_qd.numpy()[qi])
        else:
            bq = state.body_q.numpy()[0]
            bqd = state.body_qd.numpy()[0]
            theta = float(np.arctan2(bq[0], -bq[1]))
            theta_dot = float(bqd[5])
        ke = 0.5 * I_pivot * theta_dot**2
        pe = mass * abs(g) * (-L * np.cos(theta))
        return ke, pe

    num_steps = int(2.0 / sim_dt)

    ke0, pe0 = compute_ke_pe(state_0)
    E_initial = ke0 + pe0
    ke_values = [ke0]
    pe_values = [pe0]

    for _ in range(num_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0
        ke, pe = compute_ke_pe(state_0)
        ke_values.append(ke)
        pe_values.append(pe)

    ke_values = np.array(ke_values)
    pe_values = np.array(pe_values)
    energies = ke_values + pe_values

    # Check KE is near-zero at turning points
    min_ke = np.min(ke_values[1:])
    test.assertLess(
        min_ke / abs(E_initial),
        0.01,
        f"Min KE ({min_ke:.6e}) exceeds 1% of |E_0| ({abs(E_initial):.6e}) — "
        f"pendulum does not appear to reverse direction",
    )

    # Check total energy conservation
    max_drift = np.max(np.abs(energies - E_initial))
    rel_drift = max_drift / abs(E_initial) if abs(E_initial) > 1e-10 else max_drift
    test.assertLess(
        rel_drift,
        0.005,
        f"Energy drift {rel_drift:.6f} ({max_drift:.8e} absolute) exceeds 0.5% of initial energy {E_initial:.8f}",
    )


# ---------------------------------------------------------------------------
# Test 4: Projectile Motion
# Verify projectile trajectory against analytical parabolic equations.
# ---------------------------------------------------------------------------
def test_projectile_motion(test, device, solver_fn, uses_generalized_coords):
    # Test parameters: gravity, initial position and initial velocity
    g = -10.0
    x0, y0, z0 = 0.0, 10.0, 0.0
    vx0, vy0, vz0 = 5.0, 10.0, 0.0

    # Add a sphere
    builder = newton.ModelBuilder(gravity=g, up_axis=newton.Axis.Y)
    b = builder.add_body(xform=wp.transform(wp.vec3(x0, y0, z0), wp.quat_identity()))
    builder.add_shape_sphere(b, radius=0.1)
    model = builder.finalize(device=device)

    solver = solver_fn(model)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Set initial velocity
    velocity = np.array([vx0, vy0, vz0, 0.0, 0.0, 0.0], dtype=np.float32)
    if uses_generalized_coords:
        state_0.joint_qd.assign(velocity)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    else:
        state_0.body_qd.assign(velocity.reshape(1, 6))

    sim_dt = 1e-3
    num_steps = 100
    check_steps = [10, 20, 30, 50, 70, 100]
    for step_i in range(1, num_steps + 1):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

        if step_i in check_steps:
            # Checkpoint to verify correct simulation
            t = step_i * sim_dt
            pos = state_0.body_q.numpy()[0][:3]
            vel = state_0.body_qd.numpy()[0][:3]

            expected_pos_x = x0 + vx0 * t
            expected_pos_y = y0 + vy0 * t + 0.5 * g * t * t
            expected_pos_z = z0 + vz0 * t

            expected_vel_x = vx0
            expected_vel_y = vy0 + g * t
            expected_vel_z = vz0

            # Tolerance accounts for first-order integration error
            integration_error = 0.5 * abs(g) * sim_dt * t
            pos_tol = max(2.0 * integration_error, 1e-3)
            vel_tol = max(abs(g) * sim_dt, 1e-3)

            test.assertAlmostEqual(pos[0], expected_pos_x, delta=pos_tol, msg=f"Projectile X at t={t:.3f}")
            test.assertAlmostEqual(pos[1], expected_pos_y, delta=pos_tol, msg=f"Projectile Y at t={t:.3f}")
            test.assertAlmostEqual(pos[2], expected_pos_z, delta=pos_tol, msg=f"Projectile Z at t={t:.3f}")

            test.assertAlmostEqual(vel[0], expected_vel_x, delta=vel_tol, msg=f"Projectile vx at t={t:.3f}")
            test.assertAlmostEqual(vel[1], expected_vel_y, delta=vel_tol, msg=f"Projectile vy at t={t:.3f}")
            test.assertAlmostEqual(vel[2], expected_vel_z, delta=vel_tol, msg=f"Projectile vz at t={t:.3f}")


# ---------------------------------------------------------------------------
# Test 5: Joint Actuation Application
# Verify joint response to actuation forces for revolute and prismatic joints.
# ---------------------------------------------------------------------------
def test_joint_actuation(test, device, solver_fn):
    # Test parameters: applied force for prismatic joint and applied torque for revolute joint
    tau_rev = 5.0
    F_prismatic = 5.0

    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    # Articulation 0: revolute joint (box body)
    link_rev = builder.add_link()
    builder.add_shape_box(link_rev, hx=0.2, hy=0.2, hz=0.2)
    j_rev = builder.add_joint_revolute(
        parent=-1,
        child=link_rev,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        armature=0.0,
    )
    builder.add_articulation([j_rev])

    # Articulation 1: prismatic joint (sphere body), offset to avoid overlap
    link_prismatic = builder.add_link()
    builder.add_shape_sphere(link_prismatic, radius=0.1)
    j_prismatic = builder.add_joint_prismatic(
        parent=-1,
        child=link_prismatic,
        axis=newton.Axis.X,
        parent_xform=wp.transform(wp.vec3(0.0, 5.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        armature=0.0,
    )
    builder.add_articulation([j_prismatic])
    model = builder.finalize(device=device)

    I_body_rev = model.body_inertia.numpy()[0]
    I_cm_zz = float(I_body_rev[2, 2] if I_body_rev.ndim == 2 else I_body_rev[2])
    mass_prismatic = float(model.body_mass.numpy()[1])

    solver = solver_fn(model)
    state_0 = model.state()
    state_1 = model.state()
    state_0.joint_q.assign(model.joint_q)
    state_0.joint_qd.assign(model.joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    # Set actuation forces for both joints
    control = model.control()
    qd_start = model.joint_qd_start.numpy()
    qdi_rev = qd_start[0]
    qdi_prismatic = qd_start[1]
    joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_f[qdi_rev] = tau_rev
    joint_f[qdi_prismatic] = F_prismatic
    control.joint_f.assign(joint_f)

    sim_dt = 1e-3
    num_steps = 300
    for _ in range(num_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0

    t = num_steps * sim_dt

    # Check revolute joint: omega_z = tau * t / I_zz
    measured_omega = float(state_0.joint_qd.numpy()[qdi_rev])
    expected_omega = tau_rev * t / I_cm_zz
    tol_rev = max(tau_rev / I_cm_zz * sim_dt, 1e-3)
    test.assertAlmostEqual(
        measured_omega,
        expected_omega,
        delta=tol_rev,
        msg=f"Revolute joint velocity: got {measured_omega:.6f}, expected {expected_omega:.6f}",
    )
    test.assertGreater(measured_omega, 0.0, "Revolute joint velocity should be positive")

    # Check prismatic joint: v = F * t / m
    measured_v = float(state_0.joint_qd.numpy()[qdi_prismatic])
    expected_v = F_prismatic * t / mass_prismatic
    tol_prismatic = max(F_prismatic / mass_prismatic * sim_dt, 1e-3)
    test.assertAlmostEqual(
        measured_v,
        expected_v,
        delta=tol_prismatic,
        msg=f"Prismatic joint velocity: got {measured_v:.6f}, expected {expected_v:.6f}",
    )
    test.assertGreater(measured_v, 0.0, "Prismatic joint velocity should be positive")


# ---------------------------------------------------------------------------
# Test 6: Momentum Conservation
# Verify total linear and angular momentum is conserved for isolated free bodies.
# ---------------------------------------------------------------------------
def test_momentum_conservation(test, device, solver_fn, uses_generalized_coords):
    def compute_momenta(state):
        body_q = state.body_q.numpy()[:4]
        body_qd = state.body_qd.numpy()[:4]

        p_total = np.zeros(3)
        L_total = np.zeros(3)
        for i in range(4):
            m = float(masses[i])
            v = body_qd[i, :3]
            omega = body_qd[i, 3:6]
            r = body_q[i, :3]
            quat = body_q[i, 3:7]

            # Linear momentum
            p_total += m * v

            # Angular momentum: L = r x (m*v) + R * I_body * R^T * omega
            L_total += np.cross(r, m * v)
            R = np.array(wp.quat_to_matrix(wp.quat(*quat.tolist()))).reshape(3, 3)
            I_b = I_bodies[i]
            if I_b.ndim == 1:
                I_b = np.diag(I_b)
            I_world = R @ I_b @ R.T
            L_total += I_world @ omega

        return p_total, L_total

    # Test parameters: initial positions and velocities of 4 separated free bodies.
    positions = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 100.0, 0.0), (0.0, 0.0, 100.0)]
    velocities = [
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        (0.0, -1.0, 0.0, 0.3, 0.0, 0.0),
        (0.0, 0.0, 1.5, 0.0, -0.2, 0.0),
        (-0.5, 0.5, -0.5, 0.0, 0.0, -0.3),
    ]

    # Add 4 separated boxes
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)
    for pos in positions:
        b = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.5, hy=0.5, hz=0.5)
    model = builder.finalize(device=device)

    solver = solver_fn(model)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Set initial velocities
    qd_init = np.zeros((4, 6), dtype=np.float32)
    for i, v in enumerate(velocities):
        qd_init[i] = v
    if uses_generalized_coords:
        state_0.joint_qd.assign(qd_init.flatten())
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    else:
        state_0.body_qd.assign(qd_init)

    masses = model.body_mass.numpy()[:4]
    I_bodies = model.body_inertia.numpy()[:4]

    p0, L0 = compute_momenta(state_0)
    test.assertGreater(np.linalg.norm(p0), 0.1, "Initial linear momentum should be nonzero")
    test.assertGreater(np.linalg.norm(L0), 0.1, "Initial angular momentum should be nonzero")

    sim_dt = 1e-3
    num_steps = 1000
    for _ in range(num_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0

    p_final, L_final = compute_momenta(state_0)

    # Check momentum conservation
    p_rel = np.linalg.norm(p_final - p0) / np.linalg.norm(p0)
    L_rel = np.linalg.norm(L_final - L0) / np.linalg.norm(L0)
    test.assertLess(p_rel, 5e-4, f"Linear momentum drift: {p_rel:.6e}")
    test.assertLess(L_rel, 5e-4, f"Angular momentum drift: {L_rel:.6e}")

    # Sanity: positions should have changed
    final_pos = state_0.body_q.numpy()[:4, :3]
    initial_pos = np.array(positions)
    pos_change = np.linalg.norm(final_pos - initial_pos)
    test.assertGreater(pos_change, 0.1, "Bodies should have moved")


# ---------------------------------------------------------------------------
# Test 7: Static Friction
# Verify Coulomb static friction: no sliding before threshold, sliding above threshold.
# ---------------------------------------------------------------------------
def test_static_friction(test, device, solver_fn, uses_newton_contacts):
    # Test parameters: gravity, static friction coefficient, box size, box mass.
    g = -10.0
    mu = 0.5
    box_half_extent = 0.25
    mass = 1000.0 * (2 * box_half_extent) ** 3

    # Shape config
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = mu
    cfg.ke = 1e4
    cfg.kd = 500.0
    cfg.kf = 0.0
    cfg.gap = 0.1

    # Force below and above static friction
    F_below = 0.3 * mu * mass * abs(g)
    F_above = 2.0 * mu * mass * abs(g)

    # Two boxes on the same ground plane: body 0 gets sub-threshold force, body 1 gets above-threshold
    builder = newton.ModelBuilder(gravity=g, up_axis=newton.Axis.Y)
    builder.add_ground_plane(cfg=cfg)
    b_below = builder.add_body(xform=wp.transform(wp.vec3(0.0, box_half_extent + 0.001, 0.0), wp.quat_identity()))
    b_above = builder.add_body(xform=wp.transform(wp.vec3(0.0, box_half_extent + 0.001, 5.0), wp.quat_identity()))
    builder.add_shape_box(b_below, hx=box_half_extent, hy=box_half_extent, hz=box_half_extent, cfg=cfg)
    builder.add_shape_box(b_above, hx=box_half_extent, hy=box_half_extent, hz=box_half_extent, cfg=cfg)
    model = builder.finalize(device=device)

    solver = solver_fn(model)
    contacts = model.contacts() if uses_newton_contacts else None
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    sim_dt = 1e-3
    num_steps = 200
    wrenches = np.zeros((2, 6), dtype=np.float32)
    wrenches[0, 0] = F_below
    wrenches[1, 0] = F_above
    for i in range(num_steps):
        state_0.clear_forces()
        if i > 20:
            # Settling period in the first few steps
            state_0.body_f.assign(wrenches)
        if contacts is not None:
            model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()

    # Below threshold: box should NOT slide
    pos_below = body_q[0][:3]
    test.assertLess(abs(pos_below[0]), 0.01, f"Below threshold: box drifted X={pos_below[0]:.6f} (should be < 0.01)")
    test.assertAlmostEqual(pos_below[1], box_half_extent, delta=0.01, msg="Box should stay on ground (below)")

    # Above threshold: box SHOULD slide
    pos_above = body_q[1][:3]
    test.assertGreater(pos_above[0], 0.05, f"Above threshold: box displacement X={pos_above[0]:.6f} (should be > 0.05)")
    test.assertAlmostEqual(pos_above[1], box_half_extent, delta=0.01, msg="Box should stay on ground (above)")


# ---------------------------------------------------------------------------
# Test 8: Dynamic Friction
# Verify sliding box decelerates and stops at d_stop = v0^2 / (2*mu*g).
# ---------------------------------------------------------------------------
def test_dynamic_friction(test, device, solver_fn, uses_newton_contacts, uses_generalized_coords):
    # Test parameters: gravity, dynamic friction coefficient, initial velocity, box size
    g = -10.0
    mu = 0.4
    v0 = 2.0
    box_half_extent = 0.25

    # Analytical stopping time and distance
    t_stop = v0 / (mu * abs(g))
    d_stop_analytical = v0**2 / (2.0 * mu * abs(g))

    # Shape config
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = mu
    cfg.ke = 1e4
    cfg.kd = 500.0
    cfg.kf = 0.0
    cfg.gap = 0.1

    # A simple box on a ground plane
    builder = newton.ModelBuilder(gravity=g, up_axis=newton.Axis.Y)
    builder.add_ground_plane(cfg=cfg)
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, box_half_extent + 0.001, 0.0), wp.quat_identity()))
    builder.add_shape_box(b, hx=box_half_extent, hy=box_half_extent, hz=box_half_extent, cfg=cfg)
    model = builder.finalize(device=device)

    solver = solver_fn(model)
    contacts = model.contacts() if uses_newton_contacts else None
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    # Apply initial velocity in X
    qd = state_0.body_qd.numpy()
    qd[0, 0] = v0
    state_0.body_qd.assign(qd)

    # For generalized-coordinate solvers, sync joint_q/joint_qd from body state
    if uses_generalized_coords:
        q_ik = wp.zeros_like(model.joint_q, device=device)
        qd_ik = wp.zeros_like(model.joint_qd, device=device)
        newton.eval_ik(model, state_0, q_ik, qd_ik)
        state_0.joint_q.assign(q_ik)
        state_0.joint_qd.assign(qd_ik)

    sim_dt = 1e-3
    total_steps = int(1.5 * t_stop / sim_dt)
    for _ in range(total_steps):
        state_0.clear_forces()
        if contacts is not None:
            model.collide(state_0, contacts)
        solver.step(state_0, state_1, None, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

    # Stopping distance within 1% of analytical
    final_pos = state_0.body_q.numpy()[0][:3]
    test.assertAlmostEqual(
        final_pos[0],
        d_stop_analytical,
        delta=0.01 * d_stop_analytical,
        msg=f"Stopping distance: got {final_pos[0]:.4f}, expected {d_stop_analytical:.4f}",
    )

    # Sanity checks
    final_vel = state_0.body_qd.numpy()[0][:3]
    test.assertAlmostEqual(abs(final_vel[0]), 0.0, delta=0.01, msg="Box should be nearly stopped")
    test.assertAlmostEqual(final_pos[1], box_half_extent, delta=0.01, msg="Box should stay on ground")


# ---------------------------------------------------------------------------
# Test 9a: Restitution
# Verify bounce height h_rebound = e^2 * h_drop for different restitution coefficients.
# ---------------------------------------------------------------------------
def test_restitution(test, device, solver_fn):
    # Test parameters: gravity, initial height, sphere radius, restitution values
    g = -10.0
    h_drop = 1.0
    radius = 0.05
    restitution_values = [0.5, 0.8]

    rebound_heights = {}
    for e in restitution_values:
        # Shape config
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.mu = 0.0
        cfg.restitution = e
        cfg.ke = 1e4
        cfg.kd = 100.0
        cfg.kf = 0.0
        cfg.margin = 0.001
        cfg.gap = 0.0

        builder = newton.ModelBuilder(gravity=g, up_axis=newton.Axis.Y)
        builder.add_ground_plane(cfg=cfg)
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, radius + h_drop, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(b, radius=radius, cfg=cfg)
        model = builder.finalize(device=device)

        solver = solver_fn(model)
        contacts = model.contacts()
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        # drop time ~ sqrt(2*h/g), run 3x to capture bounce
        sim_dt = 1e-3
        total_time = 3.0 * np.sqrt(2.0 * h_drop / abs(g))
        num_steps = int(total_time / sim_dt)
        y_positions = []
        for _ in range(num_steps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, None, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
            y_positions.append(float(state_0.body_q.numpy()[0, 1]))

        y_arr = np.array(y_positions)

        # Rebound height: first local min = impact, max after that = peak
        test.assertGreater(np.min(y_arr), -0.01, f"Ground penetration detected for e={e}")
        impact_idx = None
        for i in range(1, len(y_arr) - 1):
            if y_arr[i] < y_arr[i - 1] and y_arr[i] <= y_arr[i + 1]:
                impact_idx = i
                break
        test.assertIsNotNone(impact_idx, f"No impact detected for e={e}")

        h_rebound = np.max(y_arr[impact_idx:]) - radius
        h_expected = e * e * h_drop
        rebound_heights[e] = h_rebound

        test.assertAlmostEqual(
            h_rebound,
            h_expected,
            delta=0.01 * h_expected,
            msg=f"Rebound height for e={e}: got {h_rebound:.4f}, expected {h_expected:.4f}",
        )

    # Cross-check ratio between restitution values
    if len(rebound_heights) == 2:
        ratio = rebound_heights[0.8] / max(rebound_heights[0.5], 1e-10)
        expected_ratio = (0.8**2) / (0.5**2)  # = 2.56
        test.assertAlmostEqual(
            ratio,
            expected_ratio,
            delta=0.01 * expected_ratio,
            msg=f"Rebound ratio: got {ratio:.3f}, expected {expected_ratio:.3f}",
        )


# ---------------------------------------------------------------------------
# Test 9b: Restitution (mujoco)
# Verify perfectly elastic bounce with zero damping.
# Verify perfectly inelastic bounce with high damping.
# ---------------------------------------------------------------------------
def test_restitution_mujoco(test, device, solver_fn, use_mujoco_cpu):
    # Test parameters: gravity, initial height, sphere radius, elastic and inelastic damping
    g = -10.0
    h_drop = 1.0
    radius = 0.05
    solref_elastic = [-1e4, 0.0]
    solref_inelastic = [-1e4, -1e3]

    # Shape config
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.mu = 0.0
    cfg.restitution = 0.0
    cfg.ke = 1e4
    cfg.kd = 100.0
    cfg.kf = 0.0
    cfg.margin = 0.1

    # Single model: ground + elastic sphere (body 0) + inelastic sphere (body 1)
    # Note: ground plane uses default cfg (custom cfg causes MuJoCo Warp divergence).
    # Restitution is controlled via geom_solref set directly on the solver below.
    builder = newton.ModelBuilder(gravity=g, up_axis=newton.Axis.Y)
    builder.add_ground_plane()
    b_elastic = builder.add_body(xform=wp.transform(wp.vec3(0.0, radius + h_drop, 0.0), wp.quat_identity()))
    b_inelastic = builder.add_body(xform=wp.transform(wp.vec3(2.0, radius + h_drop, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(b_elastic, radius=radius)
    builder.add_shape_sphere(b_inelastic, radius=radius)
    model = builder.finalize(device=device)

    solver = solver_fn(model)

    # geom 0: ground, geom 1: elastic sphere, geom 2: inelastic sphere.
    # Set sphere priority > ground so the sphere's solref controls each contact.
    if use_mujoco_cpu:
        solver.mj_model.geom_solref[0] = solref_elastic
        solver.mj_model.geom_solref[1] = solref_elastic
        solver.mj_model.geom_solref[2] = solref_inelastic
        solver.mj_model.geom_priority[1] = 1
        solver.mj_model.geom_priority[2] = 1
    else:
        sr = solver.mjw_model.geom_solref.numpy()
        sr[0, 0] = solref_elastic
        sr[0, 1] = solref_elastic
        sr[0, 2] = solref_inelastic
        solver.mjw_model.geom_solref.assign(sr)
        gp = solver.mjw_model.geom_priority.numpy()
        gp[1] = 1
        gp[2] = 1
        solver.mjw_model.geom_priority.assign(gp)

    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    sim_dt = 5e-4
    total_time = 3.0 * np.sqrt(2.0 * h_drop / abs(g))
    num_steps = int(total_time / sim_dt)
    y_elastic_arr = []
    y_inelastic_arr = []
    for _ in range(num_steps):
        solver.step(state_0, state_1, None, None, sim_dt)
        state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        y_elastic_arr.append(float(body_q[0, 1]))
        y_inelastic_arr.append(float(body_q[1, 1]))

    def rebound_height(y_arr):
        test.assertGreater(np.min(y_arr), -0.01, "Ground penetration detected")
        for i in range(1, len(y_arr) - 1):
            if y_arr[i] < y_arr[i - 1] and y_arr[i] <= y_arr[i + 1]:
                return np.max(y_arr[i:]) - radius
        test.fail("No impact detected")

    # Elastic case
    h_elastic = rebound_height(np.array(y_elastic_arr))
    test.assertAlmostEqual(
        h_elastic,
        h_drop,
        delta=0.01 * h_drop,
        msg=f"Elastic rebound: got {h_elastic:.4f}, expected ~{h_drop:.4f}",
    )

    # Inelastic case
    h_inelastic = rebound_height(np.array(y_inelastic_arr))
    test.assertAlmostEqual(
        h_inelastic,
        0.0,
        delta=0.01,
        msg=f"Inelastic rebound: got {h_inelastic:.4f}, expected near 0",
    )


# ---------------------------------------------------------------------------
# Test 10: Kinematic loop
# Verify four-bar linkage rocker angle against the Freudenstein equation.
# A Grashof crank-rocker linkage is driven at constant angular velocity.
# The simulated rocker angle is compared to the analytical solution from the
#  Freudenstein equation
# ---------------------------------------------------------------------------
@wp.kernel
def _velocity_pd_kernel(
    joint_qd: wp.array(dtype=wp.float32),
    joint_f: wp.array(dtype=wp.float32),
    qd_idx: int,
    f_idx: int,
    kp: float,
    target: float,
):
    omega = joint_qd[qd_idx]
    joint_f[f_idx] = kp * (target - omega)


def test_fourbar_linkage(test, device, solver_fn, use_loop_joint=False):
    def solve_fourbar(theta2, a, b, c, d):
        """Solve the Freudenstein equation for rocker angle theta4 given crank angle theta2.

        For a planar four-bar linkage with ground link d, crank a, coupler b, rocker c:
            K1*cos(theta4) - K2*cos(theta2) + K3 = cos(theta2 - theta4)
        where K1 = d/a, K2 = d/c, K3 = (a^2 - b^2 + c^2 + d^2) / (2*a*c).

        Returns (theta3, theta4) for the open configuration.
        """
        K1 = d / a
        K2 = d / c
        K3 = (a**2 - b**2 + c**2 + d**2) / (2.0 * a * c)

        # Rewrite as A*cos(theta4) + B*sin(theta4) = C
        A = K1 - np.cos(theta2)
        B = -np.sin(theta2)
        C = K2 * np.cos(theta2) - K3

        denom = np.sqrt(A**2 + B**2)
        arg = np.clip(C / denom, -1.0, 1.0)
        theta4 = np.arctan2(B, A) + np.arccos(arg)  # open configuration

        # Coupler angle from loop closure
        cx = d + c * np.cos(theta4) - a * np.cos(theta2)
        cy = c * np.sin(theta4) - a * np.sin(theta2)
        theta3 = np.arctan2(cy, cx)

        return theta3, theta4

    # Test parameters: link lengths, link thickness, angular velocity
    a_link, b_link, c_link, d_link = 0.2, 0.5, 0.4, 0.5
    link_thickness = 0.02  # half-extent for box shapes
    omega_target = 2.0 * np.pi  # 1 rev/s

    # Solve initial configuration at theta2 = 0
    theta3_0, _ = solve_fourbar(0.0, a_link, b_link, c_link, d_link)

    # Direction from coupler endpoint to rocker ground pivot at theta2=0
    rocker_dir = np.arctan2(
        -b_link * np.sin(theta3_0),
        d_link - a_link - b_link * np.cos(theta3_0),
    )
    delta_rocker = rocker_dir - theta3_0

    # Build the four-bar linkage
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 1000.0
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    crank_body = builder.add_link(xform=wp.transform_identity())
    coupler_body = builder.add_link(xform=wp.transform_identity())
    rocker_body = builder.add_link(xform=wp.transform_identity())
    builder.add_shape_box(crank_body, hx=a_link / 2.0, hy=link_thickness, hz=link_thickness, cfg=cfg)
    builder.add_shape_box(coupler_body, hx=b_link / 2.0, hy=link_thickness, hz=link_thickness, cfg=cfg)
    builder.add_shape_box(rocker_body, hx=c_link / 2.0, hy=link_thickness, hz=link_thickness, cfg=cfg)

    # Joint: world - crank
    j0 = builder.add_joint_revolute(
        parent=-1,
        child=crank_body,
        axis=(0, 0, 1),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(-a_link / 2.0, 0.0, 0.0), wp.quat_identity()),
        armature=0.0,
    )

    # Joint: crank - coupler
    j1 = builder.add_joint_revolute(
        parent=crank_body,
        child=coupler_body,
        axis=(0, 0, 1),
        parent_xform=wp.transform(
            wp.vec3(a_link / 2.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(theta3_0))
        ),
        child_xform=wp.transform(wp.vec3(-b_link / 2.0, 0.0, 0.0), wp.quat_identity()),
        armature=0.0,
    )

    # Joint: coupler - rocker
    j2 = builder.add_joint_revolute(
        parent=coupler_body,
        child=rocker_body,
        axis=(0, 0, 1),
        parent_xform=wp.transform(
            wp.vec3(b_link / 2.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(delta_rocker))
        ),
        child_xform=wp.transform(wp.vec3(-c_link / 2.0, 0.0, 0.0), wp.quat_identity()),
        armature=0.0,
    )

    builder.add_articulation([j0, j1, j2])
    # Loop closure
    if use_loop_joint:
        j_loop = builder.add_joint_revolute(
            parent=-1,
            child=rocker_body,
            axis=(0, 0, 1),
            parent_xform=wp.transform(wp.vec3(d_link, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(c_link / 2.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.joint_articulation[j_loop] = -1
    else:
        builder.add_equality_constraint_connect(body1=-1, body2=rocker_body, anchor=wp.vec3(d_link, 0.0, 0.0))
    model = builder.finalize(device=device)

    solver = solver_fn(model)
    # Stiffen equality constraint for tight loop closure
    if solver.use_mujoco_cpu:
        solver.mj_model.eq_solref[:] = [0.001, 1.0]
    else:
        sr = solver.mjw_model.eq_solref.numpy()
        sr[:] = [0.001, 1.0]
        solver.mjw_model.eq_solref.assign(sr)

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    crank_qd_start = int(model.joint_qd_start.numpy()[j0])
    crank_q_start = int(model.joint_q_start.numpy()[j0])

    kp = 100.0
    max_angle_error_deg = 0.0
    max_closure_error = 0.0
    sim_dt = 2e-4
    sim_time = 2.0
    num_steps = int(sim_time / sim_dt)
    check_interval = 10

    # Initial control (crank starts at rest, so tau = kp * omega_target)
    jf = np.zeros(model.joint_dof_count, dtype=np.float32)
    jf[crank_qd_start] = kp * omega_target
    control.joint_f.assign(jf)

    # Warmup
    solver.step(state, state, control, None, sim_dt)

    use_graph = device.is_cuda
    if use_graph:
        with wp.ScopedCapture(device) as capture:
            wp.launch(
                _velocity_pd_kernel,
                dim=1,
                inputs=[state.joint_qd, control.joint_f, crank_qd_start, crank_qd_start, kp, omega_target],
                device=device,
            )
            solver.step(state, state, control, None, sim_dt)
        graph = capture.graph

    for step_i in range(num_steps - 1):
        if use_graph:
            wp.capture_launch(graph)
        else:
            wp.launch(
                _velocity_pd_kernel,
                dim=1,
                inputs=[state.joint_qd, control.joint_f, crank_qd_start, crank_qd_start, kp, omega_target],
                device=device,
            )
            solver.step(state, state, control, None, sim_dt)

        if step_i < 20 or step_i % check_interval != 0:
            continue

        # Read crank angle
        q = state.joint_q.numpy()
        theta2 = float(q[crank_q_start])
        # Analytical rocker angle from Freudenstein
        _, theta4_analytical = solve_fourbar(theta2, a_link, b_link, c_link, d_link)

        # Simulated rocker angle from body positions
        body_q = state.body_q.numpy()
        rocker_pos = body_q[2, :3]  # rocker body (index 2)
        rx = rocker_pos[0] - d_link
        ry = rocker_pos[1]
        theta4_sim = np.arctan2(ry, rx)

        # Angle error (wrapped to [0, pi])
        angle_error = abs(theta4_sim - theta4_analytical)
        if angle_error > np.pi:
            angle_error = 2.0 * np.pi - angle_error
        max_angle_error_deg = max(max_angle_error_deg, np.degrees(angle_error))

        # Loop closure error: distance from rocker ground-end to world pivot
        rocker_quat = body_q[2, 3:7]
        rot = np.array(wp.quat_to_matrix(wp.quat(*rocker_quat.tolist()))).reshape(3, 3)
        rocker_tip = rocker_pos + rot @ np.array([c_link / 2.0, 0.0, 0.0])
        closure_err = np.linalg.norm(rocker_tip - np.array([d_link, 0.0, 0.0]))
        max_closure_error = max(max_closure_error, closure_err)

    # Read final crank angle for revolution count
    q_final = state.joint_q.numpy()
    theta2_final = float(q_final[crank_q_start])

    test.assertLess(
        max_angle_error_deg,
        0.1,
        msg=f"Max rocker angle error {max_angle_error_deg:.4f} deg exceeds 0.1 deg",
    )

    test.assertLess(
        max_closure_error,
        1e-3,
        msg=f"Max loop closure error {max_closure_error:.6f} m exceeds 1mm",
    )

    # Crank completes at least 2 full revolutions
    test.assertGreater(
        theta2_final,
        1.9 * 2.0 * np.pi,
        msg=f"Crank only reached {theta2_final:.2f} rad ({theta2_final / (2 * np.pi):.1f} rev), expected ~2",
    )


# ---------------------------------------------------------------------------
# Test 11: Revolute loop joint -- out-of-plane stability
#
# A four-bar linkage (world + crank + coupler + rocker) where the world->crank
# joint is a BALL (3 rotational DOFs), placed diagonally from the revolute loop
# closure at rocker->world.  The remaining two in-tree joints (crank->coupler,
# coupler->rocker) are revolute Z.  Gravity has a Z component that tries to
# buckle the mechanism out of plane.
#
# Why this cannot pass by accident:
#   - The ball joint at world->crank gives the crank X/Y rotation DOFs that
#     propagate through the revolute chain to the rocker.  Only the loop
#     closure's second CONNECT can lock them.
#   - With correct 2xCONNECT (revolute): the second CONNECT point along the Z
#     axis constrains the out-of-plane rotation DOFs.  The mechanism is forced
#     planar and Z displacement stays near zero.
#   - With wrong 1xCONNECT (ball behaviour): only translational DOFs are
#     constrained at the ground pivot.  The crank's out-of-plane DOFs are
#     unconstrained, so Z-gravity buckles the entire mechanism dramatically.
#   - With wrong WELD: the in-plane revolute DOF is also locked and the
#     mechanism cannot swing at all, failing the in-plane displacement check.
# ---------------------------------------------------------------------------


def test_revolute_loop_joint(test, device, solver_fn):
    # Proper four-bar linkage: 4 bodies (world + crank + coupler + rocker),
    # 4 joints (3 in-tree + 1 loop).  The world->crank joint is a BALL,
    # placed diagonally from the revolute loop closure at rocker->world.
    # This gives the crank 3 rotational DOFs (X, Y, Z) where only Z is the
    # intended four-bar motion.  The 2xCONNECT from the revolute loop must
    # constrain the out-of-plane DOFs; a single CONNECT would leave them
    # free, causing dramatic buckling under Z-gravity.
    a_link, b_link, c_link, d_link = 0.2, 0.5, 0.4, 0.5
    link_thickness = 0.02

    # Solve initial four-bar configuration at theta2 = 0 (Freudenstein equation)
    K1 = d_link / a_link
    K2 = d_link / c_link
    K3 = (a_link**2 - b_link**2 + c_link**2 + d_link**2) / (2.0 * a_link * c_link)
    A = K1 - np.cos(0.0)
    B = -np.sin(0.0)
    C = K2 * np.cos(0.0) - K3
    denom = np.sqrt(A**2 + B**2)
    theta4 = np.arctan2(B, A) + np.arccos(np.clip(C / denom, -1.0, 1.0))
    cx = d_link + c_link * np.cos(theta4) - a_link
    cy = c_link * np.sin(theta4)
    theta3_0 = np.arctan2(cy, cx)
    rocker_dir = np.arctan2(
        -b_link * np.sin(theta3_0),
        d_link - a_link - b_link * np.cos(theta3_0),
    )
    delta_rocker = rocker_dir - theta3_0

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 1000.0
    builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Y)

    crank_body = builder.add_link(xform=wp.transform_identity())
    coupler_body = builder.add_link(xform=wp.transform_identity())
    rocker_body = builder.add_link(xform=wp.transform_identity())
    builder.add_shape_box(crank_body, hx=a_link / 2, hy=link_thickness, hz=link_thickness, cfg=cfg)
    builder.add_shape_box(coupler_body, hx=b_link / 2, hy=link_thickness, hz=link_thickness, cfg=cfg)
    builder.add_shape_box(rocker_body, hx=c_link / 2, hy=link_thickness, hz=link_thickness, cfg=cfg)

    # Joint 0: world -> crank via BALL — diagonal to the loop joint.
    # The ball joint gives the crank X/Y rotation DOFs that propagate through
    # the revolute chain to the rocker.  Only the 2nd CONNECT from the loop
    # closure can constrain these; a single CONNECT leaves them free.
    j0 = builder.add_joint_ball(
        parent=-1,
        child=crank_body,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(-a_link / 2, 0, 0), wp.quat_identity()),
    )
    # Joint 1: crank -> coupler (revolute Z)
    j1 = builder.add_joint_revolute(
        parent=crank_body,
        child=coupler_body,
        axis=(0, 0, 1),
        parent_xform=wp.transform(
            wp.vec3(a_link / 2, 0, 0),
            wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(theta3_0)),
        ),
        child_xform=wp.transform(wp.vec3(-b_link / 2, 0, 0), wp.quat_identity()),
    )
    # Joint 2: coupler -> rocker (revolute Z)
    j2 = builder.add_joint_revolute(
        parent=coupler_body,
        child=rocker_body,
        axis=(0, 0, 1),
        parent_xform=wp.transform(
            wp.vec3(b_link / 2, 0, 0),
            wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(delta_rocker)),
        ),
        child_xform=wp.transform(wp.vec3(-c_link / 2, 0, 0), wp.quat_identity()),
    )
    builder.add_articulation([j0, j1, j2])

    # Loop closure: revolute Z from rocker back to world -> generates 2xCONNECT
    j_loop = builder.add_joint_revolute(
        parent=-1,
        child=rocker_body,
        axis=(0, 0, 1),
        parent_xform=wp.transform(wp.vec3(d_link, 0, 0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(c_link / 2, 0, 0), wp.quat_identity()),
    )
    builder.joint_articulation[j_loop] = -1

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    if solver.use_mujoco_cpu:
        solver.mj_model.eq_solref[:] = [0.001, 1.0]
        solver.mj_model.opt.gravity[:] = [0.0, -9.81, -5.0]
    else:
        sr = solver.mjw_model.eq_solref.numpy()
        sr[:] = [0.001, 1.0]
        solver.mjw_model.eq_solref.assign(sr)
        # Y-gravity swings the mechanism in-plane; Z-gravity tries to buckle it.
        gravity = solver.mjw_model.opt.gravity.numpy()
        gravity[0] = [0.0, -9.81, -5.0]
        solver.mjw_model.opt.gravity.assign(gravity)

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()

    # Give an initial angular velocity so the mechanism is in motion.
    # j0 is a ball joint (3 DOFs: wx, wy, wz); set wz for in-plane spin.
    qd = state.joint_qd.numpy()
    qd_start = int(model.joint_qd_start.numpy()[j0])
    qd[qd_start + 2] = 2.0 * np.pi  # wz = 1 rev/s
    state.joint_qd.assign(qd)

    sim_dt = 5e-4
    num_steps = 2000
    max_z = 0.0
    max_crank_y = 0.0
    for step_i in range(num_steps):
        solver.step(state, state, control, None, sim_dt)
        if step_i % 100 == 0:
            bq = state.body_q.numpy()
            max_z = max(max_z, abs(float(bq[rocker_body, 2])))
            max_crank_y = max(max_crank_y, abs(float(bq[crank_body, 1])))

    body_q = state.body_q.numpy()

    # With correct revolute loop (2xCONNECT), Z stays ~0 despite Z gravity,
    # because the second CONNECT constrains the ball joint's out-of-plane DOFs.
    # With only 1 CONNECT the crank's out-of-plane DOFs are free and the
    # entire mechanism buckles dramatically under Z-gravity.
    test.assertLess(
        max_z,
        0.02,
        msg=f"Revolute loop joint: max Z displacement {max_z:.4f} — "
        f"mechanism buckled out of plane, loop closure likely missing "
        f"the second CONNECT that constrains out-of-plane rotation",
    )

    # Confirm the mechanism actually swung in-plane (not trivially at rest).
    # A wrong WELD would lock the revolute DOF too, keeping everything still.
    test.assertGreater(
        max_crank_y,
        0.01,
        msg=f"Revolute loop joint: crank max Y {max_crank_y:.4f} too small — "
        f"mechanism did not swing, loop closure may be over-constraining "
        f"(WELD instead of revolute)",
    )

    # Loop closure error: rocker far end should be at (d_link, 0, 0)
    quat_r = body_q[rocker_body, 3:7]
    rot_r = np.array(wp.quat_to_matrix(wp.quat(*quat_r.tolist()))).reshape(3, 3)
    far_end = body_q[rocker_body, :3] + rot_r @ np.array([c_link / 2, 0, 0])
    closure_err = np.linalg.norm(far_end - np.array([d_link, 0, 0]))
    test.assertLess(
        closure_err,
        0.01,
        msg=f"Revolute loop closure error {closure_err:.6f} m exceeds 10mm",
    )


# ---------------------------------------------------------------------------
# Test 12: Ball loop joint -- conical pendulum orbit
#
# A pendulum body connected to the world via a free joint (in-tree, 6 DOFs)
# and a ball loop joint at the pivot.  The ball loop must constrain only the
# 3 translational DOFs, leaving all 3 rotational DOFs free.
#
# The pendulum starts tilted in X and is kicked in Z, creating nonzero
# angular momentum about Y (gravity axis).  This forces a conical/rosette
# orbit that spans both X and Z directions.
#
# Why this cannot pass by accident:
#   - With correct 1xCONNECT (ball): all 3 rotations are free, so the
#     pendulum orbits in 3D.  Both X and Z displacements are significant.
#   - With wrong 2xCONNECT (revolute around any single axis): only 1
#     rotational DOF is free.  The pendulum is confined to a plane, so
#     at least one of X or Z displacement is suppressed.
#   - With wrong WELD: all DOFs are locked, no motion at all.
#   - Without any constraint: the body flies off.  The closure-error check
#     catches that.
# ---------------------------------------------------------------------------


def test_ball_loop_joint(test, device, solver_fn):
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 1000.0
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Y)

    # A single link with COM at (0, -0.5, 0); ball pivot at origin.
    link = builder.add_link(xform=wp.transform(wp.vec3(0.0, -0.5, 0.0), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.02, hy=0.25, hz=0.02, cfg=cfg)

    # Free joint in tree gives the body all 6 DOFs.
    j_free = builder.add_joint_free(parent=-1, child=link)
    builder.add_articulation([j_free])

    # Ball loop joint at origin — must constrain only translation (3 DOFs),
    # leaving all 3 rotational DOFs free.
    j_loop = builder.add_joint_ball(
        parent=-1,
        child=link,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.25, 0.0), wp.quat_identity()),
    )
    builder.joint_articulation[j_loop] = -1

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    if solver.use_mujoco_cpu:
        solver.mj_model.eq_solref[:] = [0.001, 1.0]
    else:
        sr = solver.mjw_model.eq_solref.numpy()
        sr[:] = [0.001, 1.0]
        solver.mjw_model.eq_solref.assign(sr)

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # Tilt the pendulum in X, then give it tangential velocity in Z.
    # This creates nonzero angular momentum about the Y (gravity) axis,
    # so the pendulum orbits in a conical/rosette pattern instead of
    # swinging in a plane.  A wrong 2xCONNECT (revolute) would confine
    # the motion to a single plane, killing the orbit.
    q = state.joint_q.numpy()
    q_start = int(model.joint_q_start.numpy()[j_free])
    q[q_start + 0] = 0.2  # offset x (tilt pendulum sideways)
    q[q_start + 1] = -0.46  # offset y (adjust to keep ~0.5m length)
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    qd = state.joint_qd.numpy()
    qd_start = int(model.joint_qd_start.numpy()[j_free])
    qd[qd_start + 2] = 2.0  # vz (tangential to the tilt direction)
    state.joint_qd.assign(qd)

    control = model.control()
    sim_dt = 1e-3
    num_steps = 500

    # Track max displacement in X and Z over the second half of the trajectory.
    # Skip the first half so the initial X tilt doesn't count -- we want to
    # see that the pendulum maintains displacement in BOTH axes as it orbits.
    # A wrong 2xCONNECT (revolute) collapses X to zero almost immediately,
    # confining the motion to the YZ plane.
    max_x = 0.0
    max_z = 0.0
    half = num_steps // 2
    for step_i in range(num_steps):
        solver.step(state, state, control, None, sim_dt)
        if step_i >= half and step_i % 50 == 0:
            bq = state.body_q.numpy()
            max_x = max(max_x, abs(float(bq[link, 0])))
            max_z = max(max_z, abs(float(bq[link, 2])))

    body_q = state.body_q.numpy()
    pos = body_q[link, :3]

    # Both X and Z must have significant displacement in the second half.
    # An orbiting pendulum sweeps through both axes; a planar one stays
    # near zero in the axis perpendicular to its plane.
    test.assertGreater(
        max_x,
        0.05,
        msg=f"Ball loop joint: max X {max_x:.4f} in second half too small -- orbit confined to a plane",
    )
    test.assertGreater(
        max_z,
        0.05,
        msg=f"Ball loop joint: max Z {max_z:.4f} in second half too small -- orbit confined to a plane",
    )

    # Loop closure: the pivot end of the link should be near the origin.
    # This would fail without any constraint (body flies away).
    child_anchor_local = solver.mj_model.eq_data[0, 3:6]
    quat = body_q[link, 3:7]
    rot = np.array(wp.quat_to_matrix(wp.quat(*quat.tolist()))).reshape(3, 3)
    pivot_end = pos + rot @ child_anchor_local
    closure_err = np.linalg.norm(pivot_end)
    test.assertLess(
        closure_err,
        0.01,
        msg=f"Ball loop joint closure error {closure_err:.6f} m exceeds 10mm",
    )


# ---------------------------------------------------------------------------
# Test 13: Fixed loop joint — rigid L-shape under gravity
#
# An L-shaped structure: link_a hangs from world via revolute Z, link_b
# extends sideways from link_a's end via a free joint (in-tree, 6 DOFs).
# A fixed loop joint welds link_b to link_a's end.
#
# Why this cannot pass by accident:
#   - link_b is on a free joint in the tree, so it has all 6 DOFs.  Only
#     the fixed loop joint constrains it relative to link_a.
#   - Gravity pulls link_b's COM downward.  Since link_b extends
#     horizontally from the elbow, gravity creates a torque around the
#     connection point.
#   - With correct WELD: all 6 relative DOFs are locked.  link_b stays
#     rigidly attached to link_a.  The relative orientation is preserved and
#     the whole L-shape swings as a rigid body.
#   - With wrong CONNECT: only 3 translational DOFs are locked.  link_b's 3
#     rotational DOFs are free.  Gravity torque rotates link_b downward at
#     the elbow, changing the relative orientation significantly.
#   - With no constraint: link_b separates entirely.  The closure-error check
#     catches that.
#   - The in-plane swing check confirms the simulation actually ran (rules
#     out trivially passing because nothing moved).
# ---------------------------------------------------------------------------


def test_fixed_loop_joint(test, device, solver_fn):
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 1000.0
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Y)

    # link_a: vertical bar from (0,0,0) to (0,-0.5,0), COM at (0,-0.25,0)
    # link_b: horizontal bar extending from link_a's bottom end,
    #         COM at (0.25, -0.5, 0)
    link_a = builder.add_link(xform=wp.transform(wp.vec3(0.0, -0.25, 0.0), wp.quat_identity()))
    link_b = builder.add_link(xform=wp.transform(wp.vec3(0.25, -0.5, 0.0), wp.quat_identity()))
    builder.add_shape_box(link_a, hx=0.02, hy=0.25, hz=0.02, cfg=cfg)
    builder.add_shape_box(link_b, hx=0.25, hy=0.02, hz=0.02, cfg=cfg)

    # In-tree: world -> link_a via revolute Z
    j_rev = builder.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=(0, 0, 1),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform(wp.vec3(0.0, 0.25, 0.0), wp.quat_identity()),
    )
    # In-tree: link_b gets a free joint — all 6 DOFs unconstrained in the tree.
    # Only the fixed loop joint below should lock link_b to link_a.
    j_free = builder.add_joint_free(parent=-1, child=link_b)
    builder.add_articulation([j_rev, j_free])

    # Fixed loop joint at the elbow:
    #   parent anchor = bottom of link_a  (0, -0.25, 0) in link_a's frame
    #   child anchor  = left end of link_b (-0.25, 0, 0) in link_b's frame
    j_loop = builder.add_joint_fixed(
        parent=link_a,
        child=link_b,
        parent_xform=wp.transform(wp.vec3(0.0, -0.25, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
    )
    builder.joint_articulation[j_loop] = -1

    model = builder.finalize(device=device)
    solver = solver_fn(model)

    if solver.use_mujoco_cpu:
        solver.mj_model.eq_solref[:] = [0.001, 1.0]
    else:
        sr = solver.mjw_model.eq_solref.numpy()
        sr[:] = [0.001, 1.0]
        solver.mjw_model.eq_solref.assign(sr)

    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    # Record initial relative orientation between link_a and link_b
    body_q_init = state.body_q.numpy()
    rot_a_init = np.array(wp.quat_to_matrix(wp.quat(*body_q_init[link_a, 3:7].tolist()))).reshape(3, 3)
    rot_b_init = np.array(wp.quat_to_matrix(wp.quat(*body_q_init[link_b, 3:7].tolist()))).reshape(3, 3)
    rel_rot_init = rot_a_init.T @ rot_b_init

    control = model.control()
    sim_dt = 1e-3
    num_steps = 1000

    for _ in range(num_steps):
        solver.step(state, state, control, None, sim_dt)

    body_q = state.body_q.numpy()

    # Check that relative orientation is preserved (rotation locked by WELD).
    # With a wrong CONNECT, gravity torque on link_b's offset COM would rotate
    # it downward at the elbow, producing a large rotation difference.
    rot_a = np.array(wp.quat_to_matrix(wp.quat(*body_q[link_a, 3:7].tolist()))).reshape(3, 3)
    rot_b = np.array(wp.quat_to_matrix(wp.quat(*body_q[link_b, 3:7].tolist()))).reshape(3, 3)
    rel_rot_final = rot_a.T @ rot_b
    rot_diff = np.linalg.norm(rel_rot_final - rel_rot_init, "fro")
    test.assertLess(
        rot_diff,
        0.05,
        msg=f"Fixed loop joint: relative rotation changed by {rot_diff:.4f} — "
        f"bodies rotating freely, constraint likely missing rotational lock "
        f"(CONNECT instead of WELD)",
    )

    # Confirm the L-shape actually swung under gravity (simulation ran).
    # A wrong WELD-everywhere would still pass the rotation check, but the
    # free revolute joint j_rev should allow in-plane swinging.
    pos_a = body_q[link_a, :3]
    test.assertGreater(
        abs(float(pos_a[0])),
        0.01,
        msg="Fixed loop joint: link_a didn't swing — simulation may not have run",
    )

    # Verify the relative position is preserved (WELD locks both translation and rotation).
    # At t=0: link_b.pos - link_a.pos = (0.25, -0.25, 0).  After simulation this
    # relative position (in link_a's frame) should be maintained by the WELD.
    rel_pos_init = np.array([0.25, -0.25, 0.0])
    rel_pos_final = rot_a.T @ (body_q[link_b, :3] - body_q[link_a, :3])
    pos_err = np.linalg.norm(rel_pos_final - rel_pos_init)
    test.assertLess(
        pos_err,
        0.02,
        msg=f"Fixed loop joint: relative position drifted by {pos_err:.4f} — "
        f"WELD constraint not maintaining translational lock",
    )


# ---------------------------------------------------------------------------
# Test Registration
# ---------------------------------------------------------------------------

devices = get_test_devices()

for device in devices:
    # Free-body tests (all solvers)
    solvers = {
        "featherstone": (
            lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
            True,
        ),
        "mujoco_cpu": (
            lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True),
            True,
        ),
        "mujoco_warp": (
            lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
            True,
        ),
        "xpbd": (
            lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0),
            False,
        ),
        "semi_implicit": (
            lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
            False,
        ),
    }
    for solver_name, (solver_fn, uses_gen_coords) in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        if not device.is_cuda and solver_name in ("mujoco_warp", "xpbd"):
            continue

        add_function_test(
            TestPhysicsValidation,
            f"test_free_fall_{solver_name}",
            test_free_fall,
            devices=[device],
            solver_fn=solver_fn,
        )

        add_function_test(
            TestPhysicsValidation,
            f"test_projectile_motion_{solver_name}",
            test_projectile_motion,
            devices=[device],
            solver_fn=solver_fn,
            uses_generalized_coords=uses_gen_coords,
        )

        add_function_test(
            TestPhysicsValidation,
            f"test_momentum_conservation_{solver_name}",
            test_momentum_conservation,
            devices=[device],
            solver_fn=solver_fn,
            uses_generalized_coords=uses_gen_coords,
        )

    # Articulation tests (generalized-coord solvers only)
    articulation_solvers = {
        "featherstone": (
            lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
            True,
        ),
        "mujoco_cpu": (
            lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True),
            True,
        ),
        "mujoco_warp": (
            lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
            True,
        ),
        "semi_implicit": (
            lambda model: newton.solvers.SolverSemiImplicit(
                model, angular_damping=0.0, joint_attach_ke=1e5, joint_attach_kd=1e1
            ),
            False,
        ),
        "xpbd": (
            lambda model: newton.solvers.SolverXPBD(model, iterations=20, angular_damping=0.0),
            False,
        ),
    }
    for solver_name, (solver_fn, uses_gen_coords) in articulation_solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        if not device.is_cuda and solver_name in ("mujoco_warp", "xpbd"):
            continue

        add_function_test(
            TestPhysicsValidation,
            f"test_pendulum_period_{solver_name}",
            test_pendulum_period,
            devices=[device],
            solver_fn=solver_fn,
            uses_generalized_coords=uses_gen_coords,
            sim_dt=3e-4 if solver_name in ("xpbd", "semi_implicit") else 1e-3,
            sphere_radius=0.1 if solver_name == "semi_implicit" else 0.01,
        )

        # TODO: Check why energy conservation is not working with xpbd
        if solver_name == "xpbd":
            continue

        add_function_test(
            TestPhysicsValidation,
            f"test_energy_conservation_{solver_name}",
            test_energy_conservation,
            devices=[device],
            solver_fn=solver_fn,
            uses_generalized_coords=uses_gen_coords,
            sim_dt=3e-4 if solver_name in ("xpbd", "semi_implicit") else 1e-3,
            sphere_radius=0.1 if solver_name == "semi_implicit" else 0.01,
        )

        if solver_name == "semi_implicit":
            continue

        add_function_test(
            TestPhysicsValidation,
            f"test_joint_actuation_{solver_name}",
            test_joint_actuation,
            devices=[device],
            solver_fn=solver_fn,
        )

    # Friction tests
    solvers = {
        "xpbd": (
            lambda model: newton.solvers.SolverXPBD(model, iterations=10, angular_damping=0.0),
            True,
            False,
        ),
        "mujoco_cpu": (
            lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True),
            False,
            True,
        ),
        "mujoco_warp": (
            lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False),
            False,
            True,
        ),
    }
    for solver_name, (solver_fn, uses_newton_contacts, uses_gen_coords) in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        if not device.is_cuda and solver_name in ("mujoco_warp", "xpbd"):
            continue

        add_function_test(
            TestPhysicsValidation,
            f"test_static_friction_{solver_name}",
            test_static_friction,
            devices=[device],
            solver_fn=solver_fn,
            uses_newton_contacts=uses_newton_contacts,
        )

        add_function_test(
            TestPhysicsValidation,
            f"test_dynamic_friction_{solver_name}",
            test_dynamic_friction,
            devices=[device],
            solver_fn=solver_fn,
            uses_newton_contacts=uses_newton_contacts,
            uses_generalized_coords=uses_gen_coords,
        )

    # Restitution test
    if device.is_cuda:
        add_function_test(
            TestPhysicsValidation,
            "test_restitution_xpbd",
            test_restitution,
            devices=[device],
            solver_fn=lambda model: newton.solvers.SolverXPBD(
                model, iterations=10, angular_damping=0.0, enable_restitution=True
            ),
        )

    if not device.is_cuda:
        add_function_test(
            TestPhysicsValidation,
            "test_restitution_mujoco_cpu",
            test_restitution_mujoco,
            devices=[device],
            solver_fn=lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True),
            use_mujoco_cpu=True,
        )
    if device.is_cuda:
        add_function_test(
            TestPhysicsValidation,
            "test_restitution_mujoco_warp",
            test_restitution_mujoco,
            devices=[device],
            solver_fn=lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False),
            use_mujoco_cpu=False,
        )

    # Kinematic loop and loop joint constraint tests (MuJoCo only)
    loop_solvers = {
        "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(
            model, use_mujoco_cpu=True, iterations=100, ls_iterations=50
        ),
        "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(
            model, use_mujoco_cpu=False, iterations=100, ls_iterations=50
        ),
    }
    for solver_name, solver_fn in loop_solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        if not device.is_cuda and solver_name == "mujoco_warp":
            continue

        add_function_test(
            TestPhysicsValidation,
            f"test_fourbar_linkage_{solver_name}",
            test_fourbar_linkage,
            devices=[device],
            solver_fn=solver_fn,
        )
        add_function_test(
            TestPhysicsValidation,
            f"test_fourbar_linkage_loop_joint_{solver_name}",
            test_fourbar_linkage,
            devices=[device],
            solver_fn=solver_fn,
            use_loop_joint=True,
        )
        add_function_test(
            TestPhysicsValidation,
            f"test_revolute_loop_joint_{solver_name}",
            test_revolute_loop_joint,
            devices=[device],
            solver_fn=solver_fn,
        )
        add_function_test(
            TestPhysicsValidation,
            f"test_ball_loop_joint_{solver_name}",
            test_ball_loop_joint,
            devices=[device],
            solver_fn=solver_fn,
        )
        add_function_test(
            TestPhysicsValidation,
            f"test_fixed_loop_joint_{solver_name}",
            test_fixed_loop_joint,
            devices=[device],
            solver_fn=solver_fn,
        )

if __name__ == "__main__":
    unittest.main(verbosity=2)
