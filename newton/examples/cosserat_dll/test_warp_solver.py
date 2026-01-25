# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test script to verify Warp solver matches NumPy solver.

This script runs step-by-step comparisons between the NumPy and Warp
implementations of the direct Cosserat rod solver.

Usage:
    uv run python -m newton.examples.cosserat_dll.test_warp_solver
    uv run python -m newton.examples.cosserat_dll.test_warp_solver --num-particles 32 --num-steps 100
"""

import argparse

import numpy as np
import warp as wp

from .rod_state import RodState, create_straight_rod
from .simulation_direct_numpy import DirectCosseratRodSimulationNumPy
from .simulation_direct_warp import DirectCosseratRodSimulationWarp

DLL_PATH = "unity_ref"


def create_test_rod(n_particles: int = 16) -> RodState:
    """Create a test rod for verification."""
    return create_straight_rod(
        n_particles=n_particles,
        start_pos=(0.0, 0.0, 1.0),
        direction=(1.0, 0.0, 0.0),
        segment_length=0.05,
        fix_first=True,
    )


def compare_arrays(name: str, arr_np: np.ndarray, arr_wp: np.ndarray, rtol: float = 1e-4, atol: float = 1e-6) -> bool:
    """Compare numpy and warp arrays and print results."""
    if arr_np.shape != arr_wp.shape:
        print(f"  {name}: SHAPE MISMATCH - numpy {arr_np.shape} vs warp {arr_wp.shape}")
        return False

    max_abs_diff = np.max(np.abs(arr_np - arr_wp))
    max_rel_diff = np.max(np.abs(arr_np - arr_wp) / (np.abs(arr_np) + atol))

    match = np.allclose(arr_np, arr_wp, rtol=rtol, atol=atol)
    status = "PASS" if match else "FAIL"

    print(f"  {name}: {status} (max_abs={max_abs_diff:.2e}, max_rel={max_rel_diff:.2e})")

    if not match:
        # Show first few differing elements
        diff = np.abs(arr_np - arr_wp)
        flat_diff = diff.flatten()
        worst_idx = np.argmax(flat_diff)
        print(f"    Worst diff at flat index {worst_idx}")
        print(f"    numpy: {arr_np.flatten()[worst_idx]}")
        print(f"    warp:  {arr_wp.flatten()[worst_idx]}")

    return match


def test_full_step(n_particles: int = 16, verbose: bool = True) -> bool:
    """Test a full simulation step."""
    if verbose:
        print("\n=== Testing full step ===")

    state_np = create_test_rod(n_particles)
    state_wp = create_test_rod(n_particles)

    sim_np = DirectCosseratRodSimulationNumPy(state_np, dll_path=DLL_PATH)
    sim_np.use_numpy_predict_positions = True
    sim_np.use_numpy_predict_rotations = True
    sim_np.use_numpy_project_direct = True
    sim_np.use_numpy_integrate_positions = True
    sim_np.use_numpy_integrate_rotations = True

    sim_wp = DirectCosseratRodSimulationWarp(state_wp, device="cuda:0")
    sim_wp.verification_mode = True  # Enable auto-sync

    # Match parameters
    sim_np.position_damping = 0.001
    sim_np.rotation_damping = 0.001
    sim_np.gravity[:3] = [0.0, 0.0, -9.81]
    sim_np.young_modulus_mult = 1.0e6
    sim_np.torsion_modulus_mult = 1.0e6

    sim_wp.position_damping = 0.001
    sim_wp.rotation_damping = 0.001
    sim_wp.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
    sim_wp.young_modulus = 1.0e6
    sim_wp.torsion_modulus = 1.0e6

    dt = 1.0 / 60.0

    # Run one step each
    sim_np.step(dt)
    sim_wp.step(dt)

    # Compare final positions
    pos_np = state_np.positions[:, :3]
    pos_wp = state_wp.positions[:, :3]

    ori_np = state_np.orientations
    ori_wp = state_wp.orientations

    success = True
    success &= compare_arrays("positions", pos_np, pos_wp, rtol=1e-3, atol=1e-5)
    success &= compare_arrays("orientations", ori_np, ori_wp, rtol=1e-3, atol=1e-5)

    return success


def test_multi_step(n_particles: int = 16, n_steps: int = 100, verbose: bool = True) -> bool:
    """Test multiple simulation steps."""
    if verbose:
        print(f"\n=== Testing {n_steps} steps ===")

    state_np = create_test_rod(n_particles)
    state_wp = create_test_rod(n_particles)

    sim_np = DirectCosseratRodSimulationNumPy(state_np, dll_path=DLL_PATH)
    sim_np.use_numpy_predict_positions = True
    sim_np.use_numpy_predict_rotations = True
    sim_np.use_numpy_project_direct = True
    sim_np.use_numpy_integrate_positions = True
    sim_np.use_numpy_integrate_rotations = True

    sim_wp = DirectCosseratRodSimulationWarp(state_wp, device="cuda:0")
    sim_wp.verification_mode = True

    # Match parameters
    sim_np.position_damping = 0.001
    sim_np.rotation_damping = 0.001

    sim_wp.position_damping = 0.001
    sim_wp.rotation_damping = 0.001

    dt = 1.0 / 60.0

    for step in range(n_steps):
        sim_np.step(dt)
        sim_wp.step(dt)

        # Check for divergence periodically
        if step % 10 == 9 or step == n_steps - 1:
            pos_np = state_np.positions[:, :3]
            pos_wp = state_wp.positions[:, :3]

            max_diff = np.max(np.abs(pos_np - pos_wp))
            if verbose and step == n_steps - 1:
                print(f"  Step {step + 1}: max position diff = {max_diff:.2e}")

            # Check for NaN/Inf
            if not np.all(np.isfinite(pos_np)) or not np.all(np.isfinite(pos_wp)):
                print(f"  FAIL: Non-finite values detected at step {step + 1}")
                return False

    pos_np = state_np.positions[:, :3]
    pos_wp = state_wp.positions[:, :3]
    ori_np = state_np.orientations
    ori_wp = state_wp.orientations

    success = True
    # Use looser tolerance for accumulated error over many steps
    success &= compare_arrays("positions", pos_np, pos_wp, rtol=1e-2, atol=1e-4)
    success &= compare_arrays("orientations", ori_np, ori_wp, rtol=1e-2, atol=1e-4)

    return success


def test_warp_standalone(n_particles: int = 16, n_steps: int = 100, verbose: bool = True) -> bool:
    """Test Warp solver runs without crashing and produces stable results."""
    if verbose:
        print(f"\n=== Testing Warp standalone ({n_steps} steps) ===")

    state = create_test_rod(n_particles)
    sim = DirectCosseratRodSimulationWarp(state, device="cuda:0")
    sim.verification_mode = True

    dt = 1.0 / 60.0

    for step in range(n_steps):
        sim.step(dt)

        if step % 20 == 19 or step == n_steps - 1:
            pos = state.positions[:, :3]
            if not np.all(np.isfinite(pos)):
                print(f"  FAIL: Non-finite values at step {step + 1}")
                return False
            if verbose and step == n_steps - 1:
                tip_pos = pos[-1]
                print(f"  Step {step + 1}: tip position = ({tip_pos[0]:.4f}, {tip_pos[1]:.4f}, {tip_pos[2]:.4f})")

    # Check rod fell under gravity
    final_pos = state.positions[:, :3]
    initial_z = 1.0
    tip_z = final_pos[-1, 2]

    if tip_z > initial_z - 0.01:
        print(f"  FAIL: Tip didn't fall (z = {tip_z:.4f})")
        return False

    print(f"  Tip fell from z={initial_z:.2f} to z={tip_z:.4f}")
    print("  PASS: Warp solver produces stable simulation")
    return True


def run_all_tests(n_particles: int = 16, verbose: bool = True) -> bool:
    """Run all verification tests."""
    print("=" * 60)
    print("Warp Direct Cosserat Rod Solver Verification")
    print(f"Particles: {n_particles}")
    print("=" * 60)

    all_passed = True

    tests = [
        ("warp_standalone", lambda: test_warp_standalone(n_particles, 100, verbose)),
        ("full_step", lambda: test_full_step(n_particles, verbose)),
        ("multi_step", lambda: test_multi_step(n_particles, 100, verbose)),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn()
            all_passed &= passed
        except Exception as e:
            print(f"\n=== {name} ===")
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test Warp direct solver against NumPy reference")
    parser.add_argument("--test", type=str, default=None, help="Run specific test")
    parser.add_argument("--num-particles", type=int, default=16, help="Number of particles in test rod")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of steps for multi-step test")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()
    verbose = not args.quiet

    # Initialize Warp
    wp.init()

    if args.test:
        test_map = {
            "warp_standalone": lambda: test_warp_standalone(args.num_particles, args.num_steps, verbose),
            "full_step": lambda: test_full_step(args.num_particles, verbose),
            "multi_step": lambda: test_multi_step(args.num_particles, args.num_steps, verbose),
        }

        if args.test not in test_map:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {list(test_map.keys())}")
            return 1

        passed = test_map[args.test]()
        return 0 if passed else 1
    else:
        passed = run_all_tests(args.num_particles, verbose)
        return 0 if passed else 1


if __name__ == "__main__":
    exit(main())
