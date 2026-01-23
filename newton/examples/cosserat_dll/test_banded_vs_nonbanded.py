# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Quick test to compare banded vs non-banded NumPy Cosserat rod solvers.

Runs three simulations in parallel:
- C++ DLL reference
- Non-banded NumPy (dense matrix solve)
- Banded NumPy (scipy banded solver)

Compares results to verify all three match.

Usage: python -m newton.examples.cosserat_dll.test_banded_vs_nonbanded
"""

import numpy as np

from .rod_state import create_straight_rod
from .simulation_direct import DirectCosseratRodSimulation
from .simulation_direct_numpy import DirectCosseratRodSimulationNumPy


def main():
    print("=" * 60)
    print("C++ vs Banded vs Non-banded NumPy Cosserat Rod Comparison")
    print("=" * 60)

    # Rod parameters
    n_particles = 32
    segment_length = 0.05
    dll_path = "unity_ref"

    # Create three identical rods
    state_cpp = create_straight_rod(
        n_particles=n_particles,
        start_pos=(0.0, 0.0, 1.0),
        direction=(1.0, 0.0, 0.0),
        segment_length=segment_length,
        fix_first=True,
    )
    state_nonbanded = create_straight_rod(
        n_particles=n_particles,
        start_pos=(0.0, 0.0, 1.0),
        direction=(1.0, 0.0, 0.0),
        segment_length=segment_length,
        fix_first=True,
    )
    state_banded = create_straight_rod(
        n_particles=n_particles,
        start_pos=(0.0, 0.0, 1.0),
        direction=(1.0, 0.0, 0.0),
        segment_length=segment_length,
        fix_first=True,
    )

    # Create simulations
    sim_cpp = DirectCosseratRodSimulation(state_cpp, dll_path)
    sim_nonbanded = DirectCosseratRodSimulationNumPy(state_nonbanded, dll_path)
    sim_banded = DirectCosseratRodSimulationNumPy(state_banded, dll_path)

    # Configure non-banded solver (use DLL predict to isolate solver difference)
    sim_nonbanded.use_numpy_predict_positions = False  # Use DLL for fair comparison
    sim_nonbanded.use_numpy_predict_rotations = False  # Use DLL for fair comparison
    sim_nonbanded.use_numpy_project_direct = True  # Non-banded all-in-one

    # Configure banded solver (use DLL predict to isolate solver difference)
    sim_banded.use_numpy_predict_positions = False  # Use DLL for fair comparison
    sim_banded.use_numpy_predict_rotations = False  # Use DLL for fair comparison
    sim_banded.use_numpy_prepare = True
    sim_banded.use_numpy_update = True
    sim_banded.use_numpy_jacobians = True
    sim_banded.use_numpy_assemble = True
    sim_banded.use_numpy_solve = True
    sim_banded.use_numpy_project_direct = False  # Use banded steps

    # Set matching parameters
    sim_cpp.set_gravity(0.0, 0.0, -9.81)
    sim_nonbanded.set_gravity(0.0, 0.0, -9.81)
    sim_banded.set_gravity(0.0, 0.0, -9.81)

    sim_cpp.young_modulus_mult = 1.0e6
    sim_nonbanded.young_modulus_mult = 1.0e6
    sim_banded.young_modulus_mult = 1.0e6
    sim_cpp.torsion_modulus_mult = 1.0e6
    sim_nonbanded.torsion_modulus_mult = 1.0e6
    sim_banded.torsion_modulus_mult = 1.0e6

    # Simulation parameters
    dt = 1.0 / 60.0 / 4  # 4 substeps per frame
    n_steps = 120 * 4  # 2 seconds at 60fps with 4 substeps

    max_cpp_nb_diff = 0.0
    max_cpp_b_diff = 0.0
    max_nb_b_diff = 0.0

    print(f"\nSimulating {n_steps} steps (dt={dt:.6f}s)...")
    print("-" * 60)

    for step in range(n_steps):
        # Step all three simulations
        sim_cpp.step(dt)
        sim_nonbanded.step(dt)
        sim_banded.step(dt)

        # Compare C++ vs non-banded
        cpp_nb_diff = np.abs(
            state_cpp.positions[:, :3] - state_nonbanded.positions[:, :3]
        ).max()
        max_cpp_nb_diff = max(max_cpp_nb_diff, cpp_nb_diff)

        # Compare C++ vs banded
        cpp_b_diff = np.abs(
            state_cpp.positions[:, :3] - state_banded.positions[:, :3]
        ).max()
        max_cpp_b_diff = max(max_cpp_b_diff, cpp_b_diff)

        # Compare non-banded vs banded
        nb_b_diff = np.abs(
            state_nonbanded.positions[:, :3] - state_banded.positions[:, :3]
        ).max()
        max_nb_b_diff = max(max_nb_b_diff, nb_b_diff)

        # Print progress every half second
        if step > 0 and step % (30 * 4) == 0:
            time_s = (step + 1) * dt
            tip_cpp = state_cpp.positions[-1, :3]
            tip_nb = state_nonbanded.positions[-1, :3]
            tip_b = state_banded.positions[-1, :3]
            print(
                f"t={time_s:.2f}s: cpp-nb={np.linalg.norm(tip_cpp-tip_nb)*1e6:.1f}um, "
                f"cpp-b={np.linalg.norm(tip_cpp-tip_b)*1e6:.1f}um, "
                f"nb-b={np.linalg.norm(tip_nb-tip_b)*1e6:.1f}um"
            )

    print("-" * 60)
    print("\nFinal Results (max position difference over 2 seconds):")
    print(f"  C++ vs Non-banded: {max_cpp_nb_diff * 1e6:.2f} um")
    print(f"  C++ vs Banded:     {max_cpp_b_diff * 1e6:.2f} um")
    print(f"  Non-banded vs Banded: {max_nb_b_diff * 1e6:.2f} um")

    # Final tip comparison
    tip_cpp = state_cpp.positions[-1, :3]
    tip_nb = state_nonbanded.positions[-1, :3]
    tip_b = state_banded.positions[-1, :3]
    print("\nFinal tip position differences:")
    print(f"  C++ vs Non-banded: {np.linalg.norm(tip_cpp - tip_nb) * 1e6:.2f} um")
    print(f"  C++ vs Banded:     {np.linalg.norm(tip_cpp - tip_b) * 1e6:.2f} um")
    print(f"  Non-banded vs Banded: {np.linalg.norm(tip_nb - tip_b) * 1e6:.2f} um")

    # Check for stability
    if tip_cpp[2] < -10 or tip_nb[2] < -10 or tip_b[2] < -10:
        print("\nWARNING: Simulation appears unstable (tip fell below -10m)")
        return 1

    # Check for acceptable match (100um tolerance for C++ comparison, 50um for numpy variants)
    if max_cpp_nb_diff > 0.0001:  # 100 um
        print(f"\nWARNING: Large C++ vs non-banded diff ({max_cpp_nb_diff*1e6:.1f}um)")
        return 1
    if max_cpp_b_diff > 0.0001:  # 100 um
        print(f"\nWARNING: Large C++ vs banded diff ({max_cpp_b_diff*1e6:.1f}um)")
        return 1
    if max_nb_b_diff > 0.00005:  # 50 um
        print(f"\nWARNING: Large non-banded vs banded diff ({max_nb_b_diff*1e6:.1f}um)")
        return 1

    print("\nSUCCESS: All three solvers produce matching results!")
    return 0


if __name__ == "__main__":
    exit(main())
