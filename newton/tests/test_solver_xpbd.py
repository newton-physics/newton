# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the XPBD solver.

Includes tests for particle-particle friction using relative velocity correctly.
"""

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_particle_particle_friction_uses_relative_velocity(test, device):
    """
    Test that particle-particle friction correctly uses relative velocity.

    This test verifies the fix for the bug where friction was computed using
    absolute velocity instead of relative velocity:
        WRONG: vt = v - n * vn        (uses absolute velocity v)
        RIGHT: vt = vrel - n * vn     (uses relative velocity vrel)

    Setup:
    - Two particles in contact (overlapping slightly)
    - Both particles moving with the same tangential velocity
    - With friction coefficient > 0

    Expected behavior:
    - Since relative tangential velocity is zero, friction should not
      affect their relative motion
    - Both particles should continue moving together at the same velocity
      (modulo normal contact forces)

    If the bug existed (using absolute velocity), the friction would
    incorrectly compute a non-zero tangential component and try to
    slow down both particles differently.
    """
    builder = newton.ModelBuilder(up_axis="Y")

    # Two particles that are slightly overlapping (in contact)
    # Positioned along X axis, both at y=0, z=0
    particle_radius = 0.5
    overlap = 0.1  # small overlap to ensure contact
    separation = 2.0 * particle_radius - overlap

    pos = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(separation, 0.0, 0.0),
    ]

    # Both particles moving with the same tangential velocity (along Z axis)
    # The contact normal will be along X axis, so Z velocity is tangential
    tangential_velocity = 10.0
    vel = [
        wp.vec3(0.0, 0.0, tangential_velocity),
        wp.vec3(0.0, 0.0, tangential_velocity),
    ]

    mass = [1.0, 1.0]
    radius = [particle_radius, particle_radius]

    builder.add_particles(pos=pos, vel=vel, mass=mass, radius=radius)

    model = builder.finalize(device=device)

    # Disable gravity so we only see friction effects
    model.set_gravity((0.0, 0.0, 0.0))

    # Set particle-particle friction coefficient (XPBD particle-particle contact uses model.particle_mu)
    model.particle_mu = 1.0  # high friction
    model.particle_cohesion = 0.0

    # Use XPBD solver which uses the solve_particle_particle_contacts kernel
    solver = newton.solvers.SolverXPBD(
        model=model,
        iterations=20,
    )

    state0 = model.state()
    state1 = model.state()
    contacts = model.contacts()

    # Apply equal and opposite forces to keep the particles in sustained contact.
    # Without this, the initial overlap may be resolved in ~1 iteration and friction becomes hard to observe,
    # making the test flaky across devices/precision.
    press_force = 50.0
    assert state0.particle_f is not None
    state0.particle_f.assign(
        wp.array(
            [
                wp.vec3(wp.float32(press_force), wp.float32(0.0), wp.float32(0.0)),
                wp.vec3(wp.float32(-press_force), wp.float32(0.0), wp.float32(0.0)),
            ],
            dtype=wp.vec3,
            device=device,
        )
    )

    dt = 1.0 / 60.0
    num_steps = 60

    # Store initial relative velocity
    initial_vel = state0.particle_qd.numpy().copy()
    initial_relative_z_vel = initial_vel[0, 2] - initial_vel[1, 2]

    # Run simulation
    for _ in range(num_steps):
        model.collide(state0, contacts)
        control = model.control()
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0

    # Get final velocities
    final_vel = state0.particle_qd.numpy()
    final_relative_z_vel = final_vel[0, 2] - final_vel[1, 2]

    # The key assertion: relative tangential velocity should remain near zero
    # since both particles started with the same tangential velocity
    test.assertAlmostEqual(
        initial_relative_z_vel,
        0.0,
        places=5,
        msg="Initial relative tangential velocity should be zero",
    )
    test.assertAlmostEqual(
        final_relative_z_vel,
        0.0,
        places=3,
        msg="Final relative tangential velocity should remain near zero "
        "(friction should not affect particles moving together)",
    )

    # Also verify both particles still have similar Z velocities
    # (they should move together, not be affected differently by friction)
    test.assertAlmostEqual(
        final_vel[0, 2],
        final_vel[1, 2],
        places=3,
        msg="Both particles should have the same tangential velocity after simulation",
    )


def test_particle_particle_friction_with_relative_motion(test, device):
    """
    Test that friction DOES affect particles with different tangential velocities.

    This is the complementary test - when particles have different tangential
    velocities, friction should work to equalize them.

    Notes on test design:
    - Particle-particle friction in XPBD is applied during constraint projection while particles are in contact.
      If particles are not kept in sustained contact, you may only get a single contact correction and the
      effect of friction can be near-zero and noisy.
    - To make this robust, we apply equal-and-opposite forces along the contact normal so the particles stay
      pressed together while sliding tangentially, and we compare against a mu=0 baseline.
    """
    # Keep this test to a single time step with guaranteed initial penetration.
    # XPBD's particle-particle friction term is limited by the *incremental* normal correction (penetration error),
    # so once the overlap is resolved to touching, friction can become effectively zero. A long multi-step
    # "relative velocity must decrease" assertion is therefore inherently flaky.

    particle_radius = 0.5
    overlap = 0.1
    separation = 2.0 * particle_radius - overlap

    dt = 1.0 / 30.0  # larger dt to make frictional slip correction clearly measurable

    def run(mu: float) -> float:
        builder = newton.ModelBuilder(up_axis="Y")

        pos = [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(separation, 0.0, 0.0),
        ]

        # Different tangential velocities along Z (tangent to the X-axis contact normal).
        vel = [
            wp.vec3(0.0, 0.0, 10.0),
            wp.vec3(0.0, 0.0, 0.0),
        ]

        mass = [1.0, 1.0]
        radius = [particle_radius, particle_radius]

        builder.add_particles(pos=pos, vel=vel, mass=mass, radius=radius)

        model = builder.finalize(device=device)
        model.set_gravity((0.0, 0.0, 0.0))
        model.particle_mu = mu
        model.particle_cohesion = 0.0

        solver = newton.solvers.SolverXPBD(model=model, iterations=30)

        state0 = model.state()
        state1 = model.state()
        contacts = model.contacts()

        # One step: measure tangential slip (relative z displacement).
        model.collide(state0, contacts)
        control = model.control()
        solver.step(state0, state1, control, contacts, dt)

        q1 = state1.particle_q.numpy()
        return float(abs(q1[0, 2] - q1[1, 2]))

    slip_no_friction = run(mu=0.0)
    slip_with_friction = run(mu=1.0)

    # With mu=0, slip should be close to v_rel * dt (~10 * dt).
    test.assertGreater(
        slip_no_friction,
        0.2,
        msg="With mu=0, relative tangential slip over one step should be significant",
    )
    test.assertLess(
        slip_with_friction,
        slip_no_friction * 0.95,
        msg="With mu>0, particle-particle friction should reduce tangential slip over one step vs mu=0 baseline",
    )


def test_particle_shape_restitution_correct_particle(test, device):
    """
    Regression test for the bug where apply_particle_shape_restitution wrote
    restitution velocity to particle_v_out[tid] (contact index) instead of
    particle_v_out[particle_index].

    Setup:
    - Particle 0 ("decoy"): high above the ground (y=10), zero velocity, no contact.
    - Particle 1 ("bouncer"): at the ground surface with downward velocity, will contact.
    - The first contact has tid=0 but contact_particle[0] = 1.
    - With the old bug, restitution dv was written to particle 0 (the decoy).
    - After fix, restitution dv is written to particle 1 (the bouncer).

    Assert: particle 1's y-velocity should be positive (bouncing up) and
    particle 0's y-velocity should remain near zero.
    """
    builder = newton.ModelBuilder(up_axis="Y")

    particle_radius = 0.1

    # Particle 0: decoy, far above the ground — should never contact
    builder.add_particle(pos=(0.0, 10.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=particle_radius)

    # Particle 1: at ground level with downward velocity — will contact
    builder.add_particle(pos=(0.0, particle_radius, 0.0), vel=(0.0, -5.0, 0.0), mass=1.0, radius=particle_radius)

    # Add a ground plane so particle 1 can bounce
    builder.add_ground_plane()

    model = builder.finalize(device=device)

    # Disable gravity so decoy particle stays at rest
    model.set_gravity((0.0, 0.0, 0.0))

    # Enable restitution
    model.soft_contact_restitution = 1.0

    solver = newton.solvers.SolverXPBD(
        model=model,
        iterations=10,
        enable_restitution=True,
    )

    state0 = model.state()
    state1 = model.state()

    dt = 1.0 / 60.0

    # Run a single step — enough for the contact + restitution pass
    contacts = model.contacts()
    model.collide(state0, contacts)
    control = model.control()
    solver.step(state0, state1, control, contacts, dt)

    vel = state1.particle_qd.numpy()

    # Particle 0 (decoy, no contact): y-velocity should be ~0
    test.assertAlmostEqual(
        float(vel[0, 1]),
        0.0,
        places=2,
        msg="Decoy particle (no contact) should have zero y-velocity; restitution was incorrectly applied to it",
    )

    # Particle 1 (bouncer): y-velocity should be positive (bouncing up)
    test.assertGreater(
        float(vel[1, 1]),
        0.0,
        msg="Bouncing particle should have positive y-velocity after restitution",
    )


def test_particle_shape_restitution_accounts_for_body_velocity(test, device):
    """
    Regression test for the bug where apply_particle_shape_restitution
    did not account for the rigid body velocity at the contact point when
    computing relative velocity for restitution (#1273).

    Setup:
    - A rigid box moving upward at 5 m/s.
    - A stationary particle sitting just above the top face of the box.
    - Restitution = 1.0, gravity disabled.

    Without the fix, the kernel computes relative velocity from the
    particle velocity alone (ignoring the approaching body), so the
    approaching normal velocity appears zero and no restitution impulse
    is applied — the particle stays nearly at rest.

    With the fix, the kernel correctly subtracts the body velocity at
    the contact point, detects the closing velocity, and applies a
    restitution impulse that launches the particle upward.
    """
    builder = newton.ModelBuilder(up_axis="Y")

    # Add a dynamic rigid box centered at origin
    body_id = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_shape_box(body=body_id, hx=1.0, hy=0.5, hz=1.0)

    # Add a stationary particle just above the box's top face (y=0.5)
    particle_radius = 0.1
    builder.add_particle(
        pos=(0.0, 0.5 + particle_radius, 0.0),
        vel=(0.0, 0.0, 0.0),
        mass=1.0,
        radius=particle_radius,
    )

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, 0.0))
    model.soft_contact_restitution = 1.0

    solver = newton.solvers.SolverXPBD(
        model=model,
        iterations=10,
        enable_restitution=True,
    )

    state0 = model.state()
    state1 = model.state()

    # Give the rigid body an upward velocity so it approaches the particle
    body_vel = np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    state0.body_qd.assign(wp.array(body_vel, dtype=wp.spatial_vector, device=device))

    dt = 1.0 / 60.0
    contacts = model.contacts()
    model.collide(state0, contacts)
    control = model.control()
    solver.step(state0, state1, control, contacts, dt)

    vel = state1.particle_qd.numpy()

    # Without the fix, the position solver alone gives the particle ~5 m/s.
    # With the fix, restitution adds another ~5 m/s on top (elastic bounce
    # against a body moving at 5 m/s), yielding ~10 m/s total.
    test.assertGreater(
        float(vel[0, 1]),
        7.0,
        msg=f"Particle should receive restitution impulse from the moving body (expected ~10 m/s, got {float(vel[0, 1]):.2f})",
    )


def test_articulation_contact_drift(test, device):
    """
    Regression test for articulated bodies drifting laterally on the ground (#2030).

    When joints are solved before contacts in the XPBD iteration loop, joint
    corrections displace bodies laterally and contact friction can't fully
    counteract the displacement. Over many steps, the residual accumulates
    into visible sliding.

    Setup:
    - Load a quadruped URDF on its side on the ground plane.
    - Let it settle for 2 seconds, then simulate for 3 more seconds.
    - Check that the root body hasn't drifted laterally.
    """
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.armature = 0.01
    builder.default_joint_cfg.target_ke = 2000.0
    builder.default_joint_cfg.target_kd = 1.0
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 1.0e2
    builder.default_shape_cfg.kf = 1.0e2
    builder.default_shape_cfg.mu = 1.0

    # Place the quadruped on its side (rotated 90 degrees around X axis)
    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.PI * 0.5)
    builder.add_urdf(
        newton.examples.get_asset("quadruped.urdf"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), rot),
        floating=True,
        enable_self_collisions=False,
        ignore_inertial_definitions=True,
    )
    armature_inertia = wp.mat33(np.eye(3, dtype=np.float32)) * 0.01
    for i in range(builder.body_count):
        builder.body_inertia[i] = builder.body_inertia[i] + armature_inertia

    builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
    builder.joint_target_pos[-12:] = builder.joint_q[-12:]
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    fps = 100
    frame_dt = 1.0 / fps
    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps

    # Let the quadruped settle after drop (2 seconds)
    for _ in range(200):
        for _ in range(sim_substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    initial_x = float(body_q[0][0])
    initial_y = float(body_q[0][1])

    # Simulate for 3 more seconds
    for _ in range(300):
        for _ in range(sim_substeps):
            state_0.clear_forces()
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    body_q = state_0.body_q.numpy()
    final_x = float(body_q[0][0])
    final_y = float(body_q[0][1])

    drift_x = abs(final_x - initial_x)
    drift_y = abs(final_y - initial_y)
    drift_xy = float(np.hypot(drift_x, drift_y))

    # The root body should not drift more than 1 cm laterally over 3 seconds
    # (Z is up, so X and Y are the lateral axes)
    # Without the fix, Y drifts ~5.9 mm/s → ~1.8 cm over 3 seconds.
    max_drift = 0.01
    test.assertLess(
        drift_xy,
        max_drift,
        msg=(
            f"Root body drifted {drift_xy:.4f} m laterally over 3 seconds "
            f"(dx={drift_x:.4f}, dy={drift_y:.4f}, max allowed: {max_drift})"
        ),
    )


def test_xpbd_contact_force_static_equilibrium(test, device):
    """Steady-state contact-force regression suite for XPBD.

    Four scenarios run together in a single model so they share one settle phase
    and one averaging window. Each scenario is placed far apart on the X axis so
    contact pairs never mix between scenarios:

    - small sphere on plane (Fz = -mg)
    - heavy sphere on plane (Fz = -mg, mass-independent)
    - box on plane (4 corner contacts; summed Fz = -mg, regression for the
      ``rigid_contact_con_weighting`` N*mg inflation bug)
    - mini pyramid (two bottom cubes + one top cube; ground reaction on each
      bottom cube = own weight + half the top cube ≈ 1.5*mg)
    """
    gravity = 9.81

    sphere_radius = 0.25
    sphere_density = 1000.0
    sphere_mass = sphere_density * (4.0 / 3.0) * np.pi * sphere_radius**3

    heavy_radius = 0.5
    heavy_density = 2000.0
    heavy_mass = heavy_density * (4.0 / 3.0) * np.pi * heavy_radius**3

    box_h = 0.5
    box_density = 1000.0
    box_mass = box_density * (2.0 * box_h) ** 3

    cube_h = 0.5
    cube_density = 1000.0
    cube_mass = cube_density * (2.0 * cube_h) ** 3
    cube_mg = cube_mass * gravity

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    ground_shape = 0

    builder.default_shape_cfg.density = sphere_density
    sphere_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, sphere_radius), wp.quat_identity()))
    builder.add_shape_sphere(body=sphere_body, radius=sphere_radius)

    builder.default_shape_cfg.density = heavy_density
    heavy_body = builder.add_body(xform=wp.transform(wp.vec3(10.0, 0.0, heavy_radius), wp.quat_identity()))
    builder.add_shape_sphere(body=heavy_body, radius=heavy_radius)

    builder.default_shape_cfg.density = box_density
    box_body = builder.add_body(xform=wp.transform(wp.vec3(20.0, 0.0, box_h), wp.quat_identity()))
    builder.add_shape_box(body=box_body, hx=box_h, hy=box_h, hz=box_h)

    builder.default_shape_cfg.density = cube_density
    pyramid_x = 30.0
    cube_left_body = builder.add_body(xform=wp.transform(wp.vec3(pyramid_x - cube_h, 0.0, cube_h), wp.quat_identity()))
    builder.add_shape_box(body=cube_left_body, hx=cube_h, hy=cube_h, hz=cube_h)
    cube_right_body = builder.add_body(xform=wp.transform(wp.vec3(pyramid_x + cube_h, 0.0, cube_h), wp.quat_identity()))
    builder.add_shape_box(body=cube_right_body, hx=cube_h, hy=cube_h, hz=cube_h)
    cube_top_body = builder.add_body(xform=wp.transform(wp.vec3(pyramid_x, 0.0, 3.0 * cube_h), wp.quat_identity()))
    builder.add_shape_box(body=cube_top_body, hx=cube_h, hy=cube_h, hz=cube_h)

    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=32, rigid_contact_con_weighting=True)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / 60.0
    num_substeps = 8
    sub_dt = dt / num_substeps
    settle_steps = 200  # max needed across scenarios (pyramid stack)
    avg_steps = 60

    for _ in range(settle_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in

    shape_body_np = model.shape_body.numpy()

    sphere_force = np.zeros(3)
    heavy_force = np.zeros(3)
    box_force = np.zeros(3)
    cube_left_fz_on_body = 0.0
    cube_right_fz_on_body = 0.0

    for _ in range(avg_steps):
        for _ in range(num_substeps):
            state_in.clear_forces()
            model.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, sub_dt)
            state_in, state_out = state_out, state_in
        solver.update_contacts(contacts, state_in)

        nc = int(contacts.rigid_contact_count.numpy()[0])
        if nc == 0:
            continue
        forces = contacts.force.numpy()[:nc, :3]
        s0 = contacts.rigid_contact_shape0.numpy()[:nc]
        s1 = contacts.rigid_contact_shape1.numpy()[:nc]

        box_step_count = 0
        for ci in range(nc):
            # ``contacts.force`` is force on body0 by body1. Sum into a "force-on-ground"
            # bucket regardless of which side ground was recorded as: flip sign when
            # ground is shape1 so the final values consistently match -mg downward.
            if s0[ci] == ground_shape:
                other_shape = s1[ci]
                f = forces[ci]
            elif s1[ci] == ground_shape:
                other_shape = s0[ci]
                f = -forces[ci]
            else:
                continue  # body-body contact (top cube against bottom cubes); not asserted
            if other_shape < 0:
                continue
            other_body = shape_body_np[other_shape]
            if other_body == sphere_body:
                sphere_force += f
            elif other_body == heavy_body:
                heavy_force += f
            elif other_body == box_body:
                box_force += f
                box_step_count += 1
            elif other_body == cube_left_body:
                cube_left_fz_on_body += -f[2]
            elif other_body == cube_right_body:
                cube_right_fz_on_body += -f[2]

        test.assertGreater(box_step_count, 1, "Box should generate multiple ground contact points")

    sphere_force /= avg_steps
    heavy_force /= avg_steps
    box_force /= avg_steps
    cube_left_fz_on_body /= avg_steps
    cube_right_fz_on_body /= avg_steps

    np.testing.assert_allclose(
        sphere_force[2],
        -sphere_mass * gravity,
        rtol=0.05,
        err_msg="Sphere on plane: vertical contact force should match -mg",
    )
    np.testing.assert_allclose(
        sphere_force[0], 0.0, atol=0.5, err_msg="Sphere on plane: horizontal X force should be ~0"
    )
    np.testing.assert_allclose(
        sphere_force[1], 0.0, atol=0.5, err_msg="Sphere on plane: horizontal Y force should be ~0"
    )

    np.testing.assert_allclose(
        heavy_force[2],
        -heavy_mass * gravity,
        rtol=0.05,
        err_msg="Heavy sphere on plane: vertical contact force should match -mg",
    )
    np.testing.assert_allclose(
        heavy_force[0], 0.0, atol=0.5, err_msg="Heavy sphere on plane: horizontal X force should be ~0"
    )
    np.testing.assert_allclose(
        heavy_force[1], 0.0, atol=0.5, err_msg="Heavy sphere on plane: horizontal Y force should be ~0"
    )

    np.testing.assert_allclose(
        box_force[2],
        -box_mass * gravity,
        rtol=0.10,
        err_msg="Box on plane: total vertical contact force over multiple contacts should match -mg, not N*mg",
    )
    np.testing.assert_allclose(box_force[0], 0.0, atol=1.0, err_msg="Box on plane: horizontal X force should be ~0")
    np.testing.assert_allclose(box_force[1], 0.0, atol=1.0, err_msg="Box on plane: horizontal Y force should be ~0")

    np.testing.assert_allclose(
        cube_left_fz_on_body,
        1.5 * cube_mg,
        rtol=0.15,
        err_msg=f"Pyramid: ground reaction on left bottom cube should be ~1.5*mg={1.5 * cube_mg:.0f}, got {cube_left_fz_on_body:.0f}",
    )
    np.testing.assert_allclose(
        cube_right_fz_on_body,
        1.5 * cube_mg,
        rtol=0.15,
        err_msg=f"Pyramid: ground reaction on right bottom cube should be ~1.5*mg={1.5 * cube_mg:.0f}, got {cube_right_fz_on_body:.0f}",
    )


def test_xpbd_contact_force_zero_when_no_contact(test, device):
    """A sphere in free-fall (no ground) should produce zero contact force."""
    radius = 0.25

    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 5.0), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=radius)
    model = builder.finalize(device=device)
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    dt = 1.0 / 60.0
    state_in.clear_forces()
    model.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, dt)
    solver.update_contacts(contacts, state_out)

    ncontacts = int(contacts.rigid_contact_count.numpy()[0])
    if ncontacts > 0:
        forces = contacts.force.numpy()[:ncontacts]
        np.testing.assert_allclose(forces, 0.0, atol=1e-6, err_msg="No contact force expected in free-fall")


def test_xpbd_contact_force_zero_when_not_touching(test, device):
    """A sphere near a ground plane with a large gap: contact pair exists but force is zero."""
    radius = 0.25
    gap = 1.0
    # Place sphere so it's within the gap (contact pair generated) but not penetrating.
    # Ground is at z=0, sphere center at z = radius + 0.5*gap (well above surface).
    z = radius + 0.5 * gap

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = gap
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=radius)
    model = builder.finalize(device=device)
    model.set_gravity(wp.vec3(0.0, 0.0, 0.0))
    model.request_contact_attributes("force")

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)

    state_in.clear_forces()
    model.collide(state_in, contacts)

    ncontacts = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(ncontacts, 0, "Gap should cause a contact pair to be generated")

    solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)
    solver.update_contacts(contacts, state_out)

    forces = contacts.force.numpy()[:ncontacts, :3]
    np.testing.assert_allclose(
        forces,
        0.0,
        atol=1e-6,
        err_msg="Contact pair within gap but not touching should report zero force",
    )


def test_xpbd_update_contacts_requires_force_attribute(test, device):
    """update_contacts should raise ValueError when contacts.force is not allocated."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=0.25)
    model = builder.finalize(device=device)

    solver = newton.solvers.SolverXPBD(model, iterations=2)
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    contacts = model.contacts()

    state_in.clear_forces()
    model.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)

    test.assertIsNone(contacts.force)
    with test.assertRaises(ValueError):
        solver.update_contacts(contacts)


devices = get_test_devices(mode="basic")


class TestSolverXPBD(unittest.TestCase):
    pass


add_function_test(
    TestSolverXPBD,
    "test_particle_particle_friction_uses_relative_velocity",
    test_particle_particle_friction_uses_relative_velocity,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_particle_particle_friction_with_relative_motion",
    test_particle_particle_friction_with_relative_motion,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_particle_shape_restitution_correct_particle",
    test_particle_shape_restitution_correct_particle,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverXPBD,
    "test_particle_shape_restitution_accounts_for_body_velocity",
    test_particle_shape_restitution_accounts_for_body_velocity,
    devices=devices,
    check_output=False,
)


add_function_test(
    TestSolverXPBD,
    "test_articulation_contact_drift",
    test_articulation_contact_drift,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_static_equilibrium",
    test_xpbd_contact_force_static_equilibrium,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_zero_when_no_contact",
    test_xpbd_contact_force_zero_when_no_contact,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_contact_force_zero_when_not_touching",
    test_xpbd_contact_force_zero_when_not_touching,
    devices=devices,
    check_output=False,
)

add_function_test(
    TestSolverXPBD,
    "test_xpbd_update_contacts_requires_force_attribute",
    test_xpbd_update_contacts_requires_force_attribute,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
