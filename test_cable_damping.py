#!/usr/bin/env python3
"""
Test script for cable damping (spring stretch and edge bend) in VBD solver.

This test verifies that damping is correctly applied to both:
1. Spring constraints (stretch damping)
2. Edge constraints (bend damping)

Run with: python test_cable_damping.py
"""

import numpy as np
import warp as wp

import newton

wp.init()


def test_spring_damping():
    """Test that spring damping reduces oscillations."""
    print("Testing spring damping (stretch)...")

    # Create a simple cable made of particles connected by springs
    builder = newton.ModelBuilder()

    # Create a chain of particles
    num_particles = 5
    particles = []
    for i in range(num_particles):
        pos = (i * 0.1, 0.0, 0.0)
        particles.append(builder.add_particle(pos, (0.0, 0.0, 0.0), mass=1.0))

    # Add springs between consecutive particles
    spring_stiffness = 1000.0
    spring_damping = 10.0  # Non-zero damping

    for i in range(num_particles - 1):
        builder.add_spring(particles[i], particles[i + 1], ke=spring_stiffness, kd=spring_damping, control=0.0)

    # VBD solver requires coloring
    builder.color()
    model = builder.finalize()
    solver = newton.solvers.SolverVBD(model)

    # Create initial state with some velocity (to test damping)
    state = model.state()
    state.particle_qd[0] = wp.vec3(1.0, 0.0, 0.0)  # Give first particle initial velocity

    state_next = model.state()
    control = model.control()
    contacts = model.collide(state)
    dt = 0.01

    # Run simulation for a few steps
    velocities = []
    for _step in range(50):
        solver.step(state, state_next, control, contacts, dt)
        state, state_next = state_next, state

        # Record velocity of first particle
        vel = state.particle_qd[0].numpy()
        velocities.append(np.linalg.norm(vel))

    # With damping, velocity should decrease over time
    initial_vel = velocities[0]
    final_vel = velocities[-1]

    print(f"  Initial velocity: {initial_vel:.6f}")
    print(f"  Final velocity: {final_vel:.6f}")
    print(f"  Velocity reduction: {(1 - final_vel / initial_vel) * 100:.2f}%")

    # Verify damping is working (velocity should decrease)
    assert final_vel < initial_vel, "Spring damping should reduce velocity"
    print("  ✓ Spring damping test passed!\n")


def test_edge_bending_damping():
    """Test that edge bending damping reduces oscillations."""
    print("Testing edge bending damping...")

    # Create a simple cloth-like structure with edges for bending
    builder = newton.ModelBuilder()

    # Create a small grid of particles
    grid_size = 3
    particles = []
    for i in range(grid_size):
        for j in range(grid_size):
            pos = (i * 0.1, j * 0.1, 0.0)
            particles.append(builder.add_particle(pos, (0.0, 0.0, 0.0), mass=1.0))

    # Add triangles (required for edges)
    def get_idx(i, j):
        return i * grid_size + j

    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Two triangles per quad
            builder.add_triangle(
                get_idx(i, j), get_idx(i + 1, j), get_idx(i, j + 1), tri_ke=1000.0, tri_kd=0.0, tri_ka=1000.0
            )
            builder.add_triangle(
                get_idx(i + 1, j), get_idx(i + 1, j + 1), get_idx(i, j + 1), tri_ke=1000.0, tri_kd=0.0, tri_ka=1000.0
            )

    # Add edges for bending with damping
    bending_stiffness = 100.0
    bending_damping = 1.0  # Non-zero damping

    # Add edges between adjacent quads
    for i in range(grid_size - 2):
        for j in range(grid_size - 1):
            # Horizontal edges
            builder.add_edge(
                get_idx(i, j),
                get_idx(i + 1, j),
                get_idx(i, j + 1),
                get_idx(i + 1, j + 1),
                rest=0.0,  # rest angle
                edge_ke=bending_stiffness,
                edge_kd=bending_damping,
            )

    # VBD solver requires coloring
    builder.color()
    model = builder.finalize()
    solver = newton.solvers.SolverVBD(model)

    # Create initial state with some angular velocity (to test bending damping)
    state = model.state()
    # Displace middle particle to create initial bending
    mid_idx = get_idx(1, 1)
    state.particle_q[mid_idx] = wp.vec3(0.1, 0.1, 0.05)  # Slight displacement

    state_next = model.state()
    control = model.control()
    contacts = model.collide(state)
    dt = 0.01

    # Run simulation
    positions = []
    for _step in range(50):
        solver.step(state, state_next, control, contacts, dt)
        state, state_next = state_next, state

        # Record position of middle particle
        pos = state.particle_q[mid_idx].numpy()
        positions.append(pos.copy())

    # With damping, oscillations should be reduced
    initial_z = positions[0][2]
    final_z = positions[-1][2]

    print(f"  Initial Z position: {initial_z:.6f}")
    print(f"  Final Z position: {final_z:.6f}")
    print(f"  Position change: {abs(final_z - initial_z):.6f}")

    # Verify damping is working (oscillations should be damped)
    # The position should stabilize rather than oscillate wildly
    z_values = [p[2] for p in positions]
    z_variance = np.var(z_values[-10:])  # Variance of last 10 steps

    print(f"  Final variance: {z_variance:.6f}")

    assert z_variance < 0.01, "Bending damping should reduce oscillations"
    print("  ✓ Edge bending damping test passed!\n")


def test_cable_with_both_damping():
    """Test a cable-like structure with both spring and edge damping."""
    print("Testing cable with both spring and edge damping...")

    builder = newton.ModelBuilder()

    # Create a chain of particles (cable)
    num_particles = 4
    particles = []
    for i in range(num_particles):
        pos = (i * 0.1, 0.0, 0.0)
        particles.append(builder.add_particle(pos, (0.0, 0.0, 0.0), mass=1.0))

    # Add springs for stretch
    spring_stiffness = 1000.0
    spring_damping = 10.0
    for i in range(num_particles - 1):
        builder.add_spring(particles[i], particles[i + 1], ke=spring_stiffness, kd=spring_damping, control=0.0)

    # Add triangles and edges for bending
    # Create triangles between consecutive triplets
    for i in range(num_particles - 2):
        # Create a triangle (degenerate, but needed for edges)
        builder.add_triangle(particles[i], particles[i + 1], particles[i + 2], tri_ke=1000.0, tri_kd=0.0, tri_ka=1000.0)

    # Add edges for bending
    bending_stiffness = 100.0
    bending_damping = 1.0
    for i in range(num_particles - 3):
        # Add edge for bending
        builder.add_edge(
            particles[i],
            particles[i + 1],
            particles[i + 2],
            particles[i + 3],
            rest=0.0,
            edge_ke=bending_stiffness,
            edge_kd=bending_damping,
        )

    # VBD solver requires coloring
    builder.color()
    model = builder.finalize()
    solver = newton.solvers.SolverVBD(model)

    # Create initial state with velocity
    state = model.state()
    state.particle_qd[0] = wp.vec3(0.5, 0.0, 0.0)

    state_next = model.state()
    control = model.control()
    contacts = model.collide(state)
    dt = 0.01

    # Run simulation
    energies = []
    for _step in range(100):
        solver.step(state, state_next, control, contacts, dt)
        state, state_next = state_next, state

        # Compute kinetic energy
        ke = 0.0
        for i in range(num_particles):
            vel = state.particle_qd[i].numpy()
            ke += 0.5 * 1.0 * np.dot(vel, vel)  # mass = 1.0
        energies.append(ke)

    initial_energy = energies[0]
    final_energy = energies[-1]

    print(f"  Initial kinetic energy: {initial_energy:.6f}")
    print(f"  Final kinetic energy: {final_energy:.6f}")
    print(f"  Energy reduction: {(1 - final_energy / initial_energy) * 100:.2f}%")

    # With damping, energy should decrease
    assert final_energy < initial_energy, "Damping should reduce energy"
    print("  ✓ Combined damping test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Cable Damping in VBD Solver")
    print("=" * 60)
    print()

    try:
        test_spring_damping()
        test_edge_bending_damping()
        test_cable_with_both_damping()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
