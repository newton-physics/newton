# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example: Cosserat rod pendulum using NumPy reference implementation.

This example creates a simple pendulum-like Cosserat rod fixed at one end
and simulates it falling under gravity. It demonstrates the core PBD
algorithm from the paper "Position And Orientation Based Cosserat Rods".

Run with:
    python -m newton.examples.cosserat2.reference.example_pendulum
"""

import numpy as np

from newton.examples.cosserat2.reference import (
    CosseratRodNumpy,
    SolverConfig,
    SolverCosseratNumpy,
    quat_rotate_e3,
)


def main():
    """Run the Cosserat rod pendulum simulation."""
    print("=" * 60)
    print("NumPy Reference Implementation: Cosserat Rod Pendulum")
    print("=" * 60)

    # Create a horizontal rod that will fall under gravity
    num_particles = 10
    segment_length = 0.1
    rod = CosseratRodNumpy.create_straight_rod(
        num_particles=num_particles,
        start_pos=np.array([0.0, 0.0, 1.0]),
        direction=np.array([1.0, 0.0, 0.0]),  # Horizontal rod
        segment_length=segment_length,
        particle_mass=0.1,
        edge_mass=0.1,
        fixed_particles=[0],  # Fix the first particle
    )

    print("\nRod configuration:")
    print(f"  Particles: {num_particles}")
    print(f"  Segment length: {segment_length}")
    print(f"  Total length: {(num_particles - 1) * segment_length}")
    print("  Fixed particle: 0")

    # Configure solver
    config = SolverConfig(
        dt=1.0 / 60.0,
        substeps=4,
        constraint_iterations=4,
        gravity=np.array([0.0, 0.0, -9.81]),
        stretch_stiffness=1.0,
        shear_stiffness=1.0,
        bend_stiffness=0.5,
        twist_stiffness=0.5,
        particle_damping=0.99,
        quaternion_damping=0.99,
    )

    print("\nSolver configuration:")
    print(f"  dt: {config.dt}")
    print(f"  substeps: {config.substeps}")
    print(f"  constraint_iterations: {config.constraint_iterations}")
    print(f"  gravity: {config.gravity}")
    print(f"  particle_damping: {config.particle_damping}")
    print(f"  quaternion_damping: {config.quaternion_damping}")

    # Create solver
    solver = SolverCosseratNumpy(rod, config)

    # Print initial state
    print("\n" + "-" * 60)
    print("Initial state:")
    print("-" * 60)
    _print_rod_state(rod)

    # Simulate for 2 seconds
    simulation_time = 2.0
    num_frames = int(simulation_time / config.dt)

    print(f"\nSimulating {simulation_time} seconds ({num_frames} frames)...")

    for frame in range(num_frames):
        solver.step()

        # Print progress every 0.5 seconds
        if (frame + 1) % int(0.5 / config.dt) == 0:
            time = (frame + 1) * config.dt
            tip_pos = rod.particle_positions[-1]
            print(f"  t={time:.1f}s: tip position = [{tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f}]")

    # Print final state
    print("\n" + "-" * 60)
    print("Final state:")
    print("-" * 60)
    _print_rod_state(rod)

    # Validate constraint satisfaction
    print("\n" + "-" * 60)
    print("Constraint validation:")
    print("-" * 60)
    _validate_constraints(rod)

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


def _print_rod_state(rod: CosseratRodNumpy) -> None:
    """Print the current state of the rod."""
    print("  Particle positions:")
    for i in range(min(5, rod.num_particles)):
        pos = rod.particle_positions[i]
        print(f"    [{i}]: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
    if rod.num_particles > 5:
        print("    ...")
        pos = rod.particle_positions[-1]
        print(f"    [{rod.num_particles - 1}]: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")


def _validate_constraints(rod: CosseratRodNumpy) -> None:
    """Validate constraint satisfaction."""
    # Check stretch constraints (edge lengths)
    print("  Stretch constraints (edge length errors):")
    max_stretch_error = 0.0
    for i in range(rod.num_edges):
        edge_vec = rod.particle_positions[i + 1] - rod.particle_positions[i]
        actual_length = np.linalg.norm(edge_vec)
        rest_length = rod.rest_lengths[i]
        error = abs(actual_length - rest_length)
        max_stretch_error = max(max_stretch_error, error)
    print(f"    Max stretch error: {max_stretch_error:.6f}")

    # Check shear constraints (edge alignment with d3)
    print("  Shear constraints (edge-director alignment errors):")
    max_shear_error = 0.0
    for i in range(rod.num_edges):
        edge_vec = rod.particle_positions[i + 1] - rod.particle_positions[i]
        edge_normalized = edge_vec / np.linalg.norm(edge_vec)
        d3 = quat_rotate_e3(rod.edge_quaternions[i])
        error = 1.0 - abs(np.dot(edge_normalized, d3))
        max_shear_error = max(max_shear_error, error)
    print(f"    Max shear error: {max_shear_error:.6f}")

    # Report overall status
    if max_stretch_error < 0.01 and max_shear_error < 0.01:
        print("  Status: PASS - constraints well satisfied")
    else:
        print("  Status: WARN - constraints not fully satisfied")


if __name__ == "__main__":
    main()
