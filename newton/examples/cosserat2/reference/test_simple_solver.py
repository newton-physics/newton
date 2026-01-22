# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simple test for the simplified direct rod solver."""

import numpy as np

from newton.examples.cosserat2.reference.direct_solver_simple import SimpleDirectRodSolver


def test_straight_rod_no_gravity():
    """Test that a straight rod with no gravity stays straight."""
    print("Test 1: Straight rod, no gravity")
    print("-" * 40)

    n_particles = 5
    segment_length = 0.1

    # Create straight rod along Z axis
    positions = np.zeros((n_particles, 3))
    for i in range(n_particles):
        positions[i] = [0, 0, i * segment_length]

    # Identity quaternion for all edges (rod along Z = local Z)
    quaternions = np.tile([0, 0, 0, 1], (n_particles - 1, 1)).astype(np.float64)

    rest_lengths = np.full(n_particles - 1, segment_length)

    # Fix first particle and first edge
    particle_inv_mass = np.ones(n_particles)
    particle_inv_mass[0] = 0.0  # Fixed

    edge_inv_mass = np.ones(n_particles - 1)
    edge_inv_mass[0] = 0.0  # Fixed

    solver = SimpleDirectRodSolver(
        n_particles=n_particles,
        positions=positions,
        quaternions=quaternions,
        rest_lengths=rest_lengths,
        particle_inv_mass=particle_inv_mass,
        edge_inv_mass=edge_inv_mass,
        stretch_stiffness=1.0,
        bend_stiffness=1.0,
    )

    print(f"Initial positions:\n{solver.positions}")

    # Run a few steps with no gravity
    gravity = np.array([0.0, 0.0, 0.0])
    for _ in range(10):
        solver.step(1.0 / 60.0, gravity, iterations=2)

    print(f"Final positions:\n{solver.positions}")

    # Check positions didn't change much
    max_change = np.max(np.abs(solver.positions - positions))
    print(f"Max position change: {max_change:.6e}")
    print(f"PASS: {max_change < 1e-6}\n")


def test_bent_rod_relaxation():
    """Test that a bent rod with stiff constraints stays bent."""
    print("Test 2: Bent rod relaxation")
    print("-" * 40)

    n_particles = 4
    segment_length = 0.1

    # Create an L-shaped rod
    # First two particles along Z, then bend to X
    positions = np.array([
        [0, 0, 0],
        [0, 0, segment_length],
        [segment_length * 0.5, 0, segment_length + segment_length * 0.866],  # 30 degree bend
        [segment_length * 1.0, 0, segment_length + segment_length * 1.732],
    ], dtype=np.float64)

    # Compute quaternions from edge directions
    from newton.examples.cosserat2.reference.quaternion_ops import quat_normalize

    def quat_from_direction(d):
        """Compute quaternion rotating Z to direction d."""
        d = d / np.linalg.norm(d)
        z = np.array([0, 0, 1.0])
        dot = np.dot(z, d)
        if dot > 0.9999:
            return np.array([0, 0, 0, 1.0])
        elif dot < -0.9999:
            return np.array([1, 0, 0, 0.0])
        axis = np.cross(z, d)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1, 1))
        s = np.sin(angle / 2)
        c = np.cos(angle / 2)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])

    quaternions = np.zeros((n_particles - 1, 4))
    for i in range(n_particles - 1):
        edge_dir = positions[i + 1] - positions[i]
        quaternions[i] = quat_from_direction(edge_dir)

    rest_lengths = np.array([np.linalg.norm(positions[i + 1] - positions[i]) for i in range(n_particles - 1)])

    particle_inv_mass = np.ones(n_particles)
    particle_inv_mass[0] = 0.0

    edge_inv_mass = np.ones(n_particles - 1)
    edge_inv_mass[0] = 0.0

    solver = SimpleDirectRodSolver(
        n_particles=n_particles,
        positions=positions,
        quaternions=quaternions,
        rest_lengths=rest_lengths,
        particle_inv_mass=particle_inv_mass,
        edge_inv_mass=edge_inv_mass,
        stretch_stiffness=1.0,
        bend_stiffness=1.0,
    )

    initial_positions = solver.positions.copy()
    print(f"Initial positions:\n{initial_positions}")

    # Run with no gravity - should maintain shape
    gravity = np.array([0.0, 0.0, 0.0])
    for _ in range(20):
        solver.step(1.0 / 60.0, gravity, iterations=2)

    print(f"Final positions:\n{solver.positions}")

    max_change = np.max(np.abs(solver.positions - initial_positions))
    print(f"Max position change: {max_change:.6e}")
    print(f"PASS: {max_change < 0.01}\n")


def test_gravity_deformation():
    """Test rod deformation under gravity."""
    print("Test 3: Gravity deformation")
    print("-" * 40)

    n_particles = 6
    segment_length = 0.1

    # Horizontal rod along X
    positions = np.zeros((n_particles, 3))
    for i in range(n_particles):
        positions[i] = [i * segment_length, 0, 0]

    # Quaternion rotating Z to X
    # Rotation of 90 degrees around Y axis
    quaternions = np.tile([0, np.sin(np.pi / 4), 0, np.cos(np.pi / 4)], (n_particles - 1, 1)).astype(np.float64)

    rest_lengths = np.full(n_particles - 1, segment_length)

    particle_inv_mass = np.ones(n_particles)
    particle_inv_mass[0] = 0.0

    edge_inv_mass = np.ones(n_particles - 1)
    edge_inv_mass[0] = 0.0

    solver = SimpleDirectRodSolver(
        n_particles=n_particles,
        positions=positions,
        quaternions=quaternions,
        rest_lengths=rest_lengths,
        particle_inv_mass=particle_inv_mass,
        edge_inv_mass=edge_inv_mass,
        stretch_stiffness=0.9,  # Slightly compliant
        bend_stiffness=0.8,     # More compliant for bending
    )

    print(f"Initial Z positions: {solver.positions[:, 2]}")

    # Apply gravity
    gravity = np.array([0.0, 0.0, -9.81])
    for step in range(30):
        solver.step(1.0 / 60.0, gravity, iterations=3)
        if step % 10 == 9:
            print(f"Step {step + 1}, Z positions: {solver.positions[:, 2]}")

    # Tip should have dropped
    tip_drop = positions[-1, 2] - solver.positions[-1, 2]
    print(f"Tip Z drop: {tip_drop:.4f}")
    print(f"PASS: {tip_drop > 0.01}\n")


def test_segment_lengths():
    """Test that segment lengths are approximately preserved."""
    print("Test 4: Segment length preservation")
    print("-" * 40)

    n_particles = 5
    segment_length = 0.1

    positions = np.zeros((n_particles, 3))
    for i in range(n_particles):
        positions[i] = [i * segment_length, 0, 0]

    quaternions = np.tile([0, np.sin(np.pi / 4), 0, np.cos(np.pi / 4)], (n_particles - 1, 1)).astype(np.float64)
    rest_lengths = np.full(n_particles - 1, segment_length)

    particle_inv_mass = np.ones(n_particles)
    particle_inv_mass[0] = 0.0

    edge_inv_mass = np.ones(n_particles - 1)
    edge_inv_mass[0] = 0.0

    solver = SimpleDirectRodSolver(
        n_particles=n_particles,
        positions=positions,
        quaternions=quaternions,
        rest_lengths=rest_lengths,
        particle_inv_mass=particle_inv_mass,
        edge_inv_mass=edge_inv_mass,
        stretch_stiffness=1.0,
        bend_stiffness=0.5,
    )

    gravity = np.array([0.0, 0.0, -9.81])
    for _ in range(50):
        solver.step(1.0 / 60.0, gravity, iterations=3)

    # Check segment lengths
    errors = []
    for i in range(n_particles - 1):
        actual_len = np.linalg.norm(solver.positions[i + 1] - solver.positions[i])
        error = abs(actual_len - segment_length) / segment_length * 100
        errors.append(error)

    print(f"Segment length errors (%): {[f'{e:.2f}' for e in errors]}")
    max_error = max(errors)
    print(f"Max error: {max_error:.2f}%")
    print(f"PASS: {max_error < 5.0}\n")


if __name__ == "__main__":
    test_straight_rod_no_gravity()
    test_bent_rod_relaxation()
    test_gravity_deformation()
    test_segment_lengths()
