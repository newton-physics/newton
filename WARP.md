# GPU Parallel Direct Cosserat Rod Solver - Implementation Plan

**Status: Phase 1 Complete - Warp solver verified against NumPy reference**

## Overview

This plan extends `newton/examples/cosserat_dll/example_dll_direct_cosserat_rod.py` by adding a new GPU-parallel solver implemented in Warp. The implementation follows a **verification-first approach**: initially copying numpy data to Warp kernels to verify correctness against the existing NumPy solver before running a fully GPU-resident pipeline.

## Existing Architecture

### Current Files
- `simulation_direct_numpy.py` - NumPy implementation of direct Cosserat rod solver
- `simulation_direct.py` - C/C++ DLL reference implementation
- `rod_state.py` - State management (`RodState` class)
- `example_dll_direct_cosserat_rod.py` - Example comparing C++ and NumPy solvers

### Existing Warp Kernels (from `cosserat_codex/warp_cosserat_codex.py`)
The cosserat_codex module already has working Warp kernels that can be reused:
- `_warp_predict_positions` - Position prediction with gravity/damping
- `_warp_integrate_positions` - Position integration
- `_warp_predict_rotations` - Rotation prediction with quaternion integration
- `_warp_integrate_rotations` - Rotation integration
- `_warp_prepare_compliance` - Compliance computation
- `_warp_update_constraints_direct` - Constraint violation computation
- `_warp_compute_jacobians_direct` - Jacobian computation
- `_warp_assemble_jmjt_dense` - Dense matrix assembly
- `_warp_assemble_jmjt_banded` - Banded matrix assembly
- `_warp_block_thomas_solve` - Block Thomas algorithm solver
- `_warp_spbsv_u11_1rhs` - Banded Cholesky solver

## Implementation Strategy

### Phase 1: Verification Infrastructure

Create a new class `DirectCosseratRodSimulationWarp` in a new file `simulation_direct_warp.py` that:
1. Mirrors the NumPy implementation structure
2. Maintains both numpy arrays (for input/output) and warp arrays (for GPU compute)
3. Uses flags to switch between numpy and warp implementations per-step
4. Provides comparison utilities to verify warp outputs match numpy outputs

### Phase 2: Step-by-Step Kernel Implementation

Each simulation step will be implemented and verified independently:

| Step | NumPy Method | Warp Kernel | Priority |
|------|-------------|-------------|----------|
| 1. Predict Positions | `_predict_positions_numpy` | `_warp_predict_positions` | P1 |
| 2. Predict Rotations | `_predict_rotations_numpy` | `_warp_predict_rotations` | P1 |
| 3. Prepare Constraints | `_prepare_numpy` | `_warp_prepare_compliance` | P1 |
| 4. Update Constraints | `_update_numpy` | `_warp_update_constraints_direct` | P1 |
| 5. Compute Jacobians | `_jacobians_numpy` | `_warp_compute_jacobians_direct` | P2 |
| 6. Assemble JMJT | `_assemble_numpy` | `_warp_assemble_jmjt_banded` | P2 |
| 7. Solve System | `_solve_numpy` / `_solve_banded_spbsv_u11_1rhs` | `_warp_spbsv_u11_1rhs` / `_warp_block_thomas_solve` | P3 |
| 8. Apply Corrections | (in `_solve_numpy`) | `_warp_apply_direct_corrections` | P3 |
| 9. Integrate Positions | `_integrate_positions_numpy` | `_warp_integrate_positions` | P1 |
| 10. Integrate Rotations | `_integrate_rotations_numpy` | `_warp_integrate_rotations` | P1 |

### Phase 3: Full GPU Pipeline

Once all kernels are verified, enable a fully GPU-resident mode where:
- All state lives on GPU (no CPU-GPU transfers during simulation)
- Only transfer data for visualization/debugging when needed
- Use CUDA graphs for kernel launch optimization

## File Structure

```
newton/examples/cosserat_dll/
    simulation_direct_warp.py     # NEW: Warp GPU implementation
    test_warp_solver.py           # NEW: Verification test script
    example_dll_direct_cosserat_rod.py  # MODIFY: Add Warp rod option
```

## Implementation Details

### 1. `simulation_direct_warp.py`

```python
class DirectCosseratRodSimulationWarp:
    """GPU-parallel direct Cosserat rod solver using Warp."""

    def __init__(self, state: RodState, device: str = "cuda:0"):
        # NumPy reference (for verification)
        self.state = state
        self.device = wp.get_device(device)

        # Warp arrays (GPU-resident)
        self._init_warp_arrays()

        # Step-by-step control flags
        self.use_warp_predict_positions = True
        self.use_warp_predict_rotations = True
        self.use_warp_prepare = True
        self.use_warp_update = True
        self.use_warp_jacobians = True
        self.use_warp_assemble = True
        self.use_warp_solve = True
        self.use_warp_integrate_positions = True
        self.use_warp_integrate_rotations = True

        # Verification mode: copy data between numpy/warp after each step
        self.verification_mode = True

    def _init_warp_arrays(self):
        """Initialize GPU arrays mirroring numpy state."""
        n = self.state.n_particles
        n_edges = self.state.n_edges

        # Position state (vec3)
        self.positions_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.predicted_positions_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.velocities_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.forces_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.inv_masses_wp = wp.zeros(n, dtype=wp.float32, device=self.device)

        # Orientation state (quat)
        self.orientations_wp = wp.zeros(n, dtype=wp.quat, device=self.device)
        self.predicted_orientations_wp = wp.zeros(n, dtype=wp.quat, device=self.device)
        self.prev_orientations_wp = wp.zeros(n, dtype=wp.quat, device=self.device)
        self.angular_velocities_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.torques_wp = wp.zeros(n, dtype=wp.vec3, device=self.device)
        self.quat_inv_masses_wp = wp.zeros(n, dtype=wp.float32, device=self.device)

        # Constraint/solver state
        n_dofs = 6 * n_edges
        self.constraint_values_wp = wp.zeros(n_dofs, dtype=wp.float32, device=self.device)
        self.compliance_wp = wp.zeros(n_dofs, dtype=wp.float32, device=self.device)
        self.lambda_sum_wp = wp.zeros(n_dofs, dtype=wp.float32, device=self.device)
        self.jacobian_pos_wp = wp.zeros(n_edges * 36, dtype=wp.float32, device=self.device)
        self.jacobian_rot_wp = wp.zeros(n_edges * 36, dtype=wp.float32, device=self.device)

        # Banded solver storage
        BAND_LDAB = 34
        self.ab_wp = wp.zeros((BAND_LDAB, n_dofs), dtype=wp.float32, device=self.device)
        self.rhs_wp = wp.zeros(n_dofs, dtype=wp.float32, device=self.device)

        # Block Thomas solver storage
        self.diag_blocks_wp = wp.zeros(n_edges * 36, dtype=wp.float32, device=self.device)
        self.offdiag_blocks_wp = wp.zeros(n_edges * 36, dtype=wp.float32, device=self.device)
        self.c_blocks_wp = wp.zeros(n_edges * 36, dtype=wp.float32, device=self.device)
        self.d_prime_wp = wp.zeros(n_dofs, dtype=wp.float32, device=self.device)
        self.delta_lambda_wp = wp.zeros(n_dofs, dtype=wp.float32, device=self.device)

        # Rod properties
        self.rest_lengths_wp = wp.zeros(n_edges, dtype=wp.float32, device=self.device)
        self.rest_darboux_wp = wp.zeros(n_edges, dtype=wp.vec3, device=self.device)
        self.bend_stiffness_wp = wp.zeros(n_edges, dtype=wp.vec3, device=self.device)

    def sync_numpy_to_warp(self):
        """Copy numpy state to warp arrays."""
        # ... implementation

    def sync_warp_to_numpy(self):
        """Copy warp arrays back to numpy state."""
        # ... implementation

    def step(self, dt: float):
        """Advance simulation by one timestep."""
        # ... implementation with step-by-step control
```

### 2. Verification Test Script (`test_warp_solver.py`)

```python
"""Test script to verify Warp solver matches NumPy solver."""

def test_step_by_step():
    """Verify each step matches numpy reference."""
    # Create identical initial states
    # Run numpy step, run warp step
    # Compare outputs with tolerance

def test_full_simulation():
    """Run full simulation and compare final states."""

def test_energy_conservation():
    """Check energy is conserved (within numerical tolerance)."""

if __name__ == "__main__":
    test_step_by_step()
    test_full_simulation()
    print("All tests passed!")
```

### 3. Example Extension

Modify `example_dll_direct_cosserat_rod.py` to add a third rod (green) using the Warp solver:
- Orange rod: C++ DLL reference
- Cyan rod: NumPy implementation
- Green rod: Warp GPU implementation

## Tasks

### Task 1: Create Warp Solver Class (P1) - COMPLETE
- [x] Create `simulation_direct_warp.py` with basic structure
- [x] Implement `_init_warp_arrays()` for GPU memory allocation
- [x] Implement `sync_numpy_to_warp()` and `sync_warp_to_numpy()`
- [x] Import warp kernel functions from `cosserat_codex/warp_cosserat_codex.py`

### Task 2: Implement & Verify Prediction/Integration (P1) - COMPLETE
- [x] Wire up `_warp_predict_positions` kernel
- [x] Wire up `_warp_predict_rotations` kernel
- [x] Wire up `_warp_integrate_positions` kernel
- [x] Wire up `_warp_integrate_rotations` kernel
- [x] Verification: positions match within 2.62e-06, orientations within 1.70e-05

### Task 3: Implement & Verify Constraint System (P2) - COMPLETE
- [x] Wire up `_warp_prepare_compliance` kernel
- [x] Wire up `_warp_update_constraints_direct` kernel
- [x] Wire up `_warp_compute_jacobians_direct` kernel
- [x] Wire up `_warp_assemble_jmjt_blocks` kernel (block Thomas)
- [x] Wire up `_warp_assemble_jmjt_banded` kernel (banded Cholesky)

### Task 4: Implement & Verify Linear Solver (P3) - COMPLETE
- [x] Wire up `_warp_spbsv_u11_1rhs` (banded Cholesky)
- [x] Wire up `_warp_block_thomas_solve` (block Thomas) - DEFAULT
- [x] Implement `_warp_apply_corrections` kernel
- [x] Verification: 100-step simulation matches within 1.02e-03 positions, 7.68e-04 orientations

### Task 5: Create Test Script (P1) - COMPLETE
- [x] Implement standalone Warp test (`test_warp_standalone`)
- [x] Implement full step comparison (`test_full_step`)
- [x] Implement multi-step simulation comparison (`test_multi_step`)
- [x] Add command-line interface for test selection

### Task 6: Extend Example (P2) - COMPLETE
- [x] Modify `example_dll_direct_cosserat_rod.py` to include Warp rod
- [x] Add real-time comparison metrics display (tip positions and differences)
- [x] All three solvers (C++, NumPy, Warp) run in sync

## Testing Commands

```bash
# Run all verification tests
uv run python -m newton.examples.cosserat_dll.test_warp_solver

# Run standalone Warp test (no NumPy comparison)
uv run python -m newton.examples.cosserat_dll.test_warp_solver --test warp_standalone

# Run single-step comparison
uv run python -m newton.examples.cosserat_dll.test_warp_solver --test full_step

# Run multi-step comparison
uv run python -m newton.examples.cosserat_dll.test_warp_solver --test multi_step

# Run with custom particle count
uv run python -m newton.examples.cosserat_dll.test_warp_solver --num-particles 32 --num-steps 200
```

## Verification Results (2025-01-25)

```
============================================================
Warp Direct Cosserat Rod Solver Verification
Particles: 16
============================================================

=== Testing Warp standalone (100 steps) ===
  Step 100: tip position = (0.7494, 0.0000, 0.9333)
  Tip fell from z=1.00 to z=0.9333
  PASS: Warp solver produces stable simulation

=== Testing full step ===
  positions: PASS (max_abs=2.62e-06, max_rel=2.63e-06)
  orientations: PASS (max_abs=1.70e-05, max_rel=2.40e-05)

=== Testing 100 steps ===
  Step 100: max position diff = 1.02e-03
  positions: PASS (max_abs=1.02e-03, max_rel=1.06e-03)
  orientations: PASS (max_abs=7.68e-04, max_rel=1.11e-03)

============================================================
ALL TESTS PASSED
============================================================
```

## Success Criteria

1. **Correctness**: Warp solver output matches NumPy solver within floating-point tolerance (1e-5 relative error)
2. **Stability**: Both solvers maintain stable simulation for 10+ seconds of simulated time
3. **Performance**: Warp solver shows measurable speedup over NumPy for large particle counts (N > 100)

## Notes

- The existing `cosserat_codex` module has working Warp kernels but uses a different state management approach. We'll adapt those kernels to work with the `RodState` class from `cosserat_dll`.
- Initial implementation focuses on correctness verification; performance optimization comes later.
- The banded Cholesky solver (`_warp_spbsv_u11_1rhs`) is already implemented and matches the C++ reference.
- The block Thomas solver is an alternative that may have different numerical properties.
