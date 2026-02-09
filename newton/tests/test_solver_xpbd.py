# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the XPBD solver.

Includes tests for particle-particle friction using relative velocity correctly.
"""

import unittest

import numpy as np
import warp as wp

import newton
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
        contacts = model.collide(state0)
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

        # One step: measure tangential slip (relative z displacement).
        contacts = model.collide(state0)
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
    contacts = model.collide(state0)
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
    contacts = model.collide(state0)
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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
