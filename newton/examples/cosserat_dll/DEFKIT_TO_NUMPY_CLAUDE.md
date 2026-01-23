# DefKit to NumPy Porting Progress

This document tracks the progress of porting the Direct Position-Based Solver for Stiff Rods from C/C++ (DefKit DLL) to pure NumPy.

## Overview

The direct solver implements the algorithm from "Direct Position-Based Solver for Stiff Rods" (Deul et al.). The simulation loop consists of:

1. **predict_positions** - Semi-implicit Euler for positions
2. **predict_rotations** - Quaternion orientation prediction
3. **prepare_direct_elastic_rod_constraints** - Reset lambdas, update stiffness
4. **update_direct_constraints** - Update constraint state
5. **compute_jacobians_direct** - Compute Jacobian matrices
6. **assemble_jmjt_direct** - Assemble JMJT banded matrix
7. **solve_direct_constraints** - Solve system and apply corrections
8. **integrate_positions** - Update positions and derive velocities
9. **integrate_rotations** - Update orientations and derive angular velocities

## Porting Status

| Method | Status | Notes |
|--------|--------|-------|
| `predict_positions` | **NumPy** | Simple semi-implicit Euler |
| `predict_rotations` | DLL | Quaternion integration (placeholder) |
| `integrate_positions` | DLL | Position update and velocity derivation (placeholder) |
| `integrate_rotations` | DLL | Quaternion update and angular velocity derivation (placeholder) |
| `prepare_direct_elastic_rod_constraints` | **NumPy** | Reset lambdas, compute compliance |
| `update_direct_constraints` | **NumPy** | Compute stretch-shear and bend-twist errors |
| `compute_jacobians_direct` | **NumPy** | Compute J matrices for constraints |
| `assemble_jmjt_direct` | **NumPy** | Build banded matrix system |
| `solve_direct_constraints` | **NumPy** | Banded matrix solve (scipy) + apply corrections |

## Porting Order (Recommended)

1. **predict_positions** - Simplest, just semi-implicit Euler
2. **predict_rotations** - Quaternion math but straightforward
3. **integrate_positions** - Simple velocity derivation
4. **integrate_rotations** - Quaternion angular velocity derivation
5. **prepare_direct_elastic_rod_constraints** - State reset
6. **update_direct_constraints** - Constraint evaluation
7. **compute_jacobians_direct** - Jacobian computation
8. **assemble_jmjt_direct** - Matrix assembly
9. **solve_direct_constraints** - Banded solver (scipy.linalg.solve_banded)

## Testing

Both rods (C/C++ reference and NumPy implementation) are displayed side-by-side:
- **Orange rod (Y=0)**: C/C++ DLL reference implementation
- **Cyan rod (Y=1)**: NumPy implementation (being ported)

Both rods respond to the same UI sliders. Visual comparison shows correctness.

---

## Porting Log

### Session 1 - Initial Setup (2026-01-23)
- Created side-by-side comparison example with two rods:
  - Orange rod (Y=0): C/C++ DLL reference
  - Cyan rod (Y=1): NumPy implementation (being ported)
- Created `simulation_direct_numpy.py` with:
  - Implementation flags for each method (use_numpy_predict_positions, etc.)
  - Placeholder NumPy implementations (currently fall back to DLL)
  - `_predict_positions_numpy()` implemented as first example
- Updated `example_dll_direct_cosserat_rod.py`:
  - Two simulations running side by side
  - Both respond to same UI sliders
  - GUI shows checkboxes to toggle NumPy implementations
  - Shows tip position difference for validation
- All tests passing

### Session 1 - prepare_direct_elastic_rod_constraints (2026-01-23)
- Implemented `_prepare_numpy()` method:
  - Added internal solver state arrays:
    - `lambdas`: Lagrange multipliers (n_edges, 6)
    - `compliance`: Inverse stiffness scaled by dt² (n_edges, 6)
    - `constraint_values`: Current constraint errors (n_edges, 6)
    - `current_rest_lengths`, `current_rest_darboux`: Cached rest shape
  - Added cross-section property computation:
    - `cross_section_area = π * r²`
    - `second_moment_area = π * r⁴ / 4` (for bending)
    - `polar_moment = π * r⁴ / 2` (for torsion)
  - Computes stiffness and compliance:
    - Stretch/shear: k = E * A / L
    - Bend: k = E * I / L * stiffness_coeff
    - Twist: k = G * J / L * stiffness_coeff
    - Compliance α = 1 / (k * dt²)
- Added GUI checkbox for `prepare_constraints`

### Session 1 - update_direct_constraints (2026-01-23)
- Implemented `_update_numpy()` method:
  - Computes **Stretch-Shear constraint** (3 DOF per edge):
    - C_ss = (p1 - p0) - L * d3, projected onto local frame
    - stretch error (along d3), shear1 (along d1), shear2 (along d2)
  - Computes **Bend-Twist constraint** (3 DOF per edge):
    - C_bt = ω - ω_rest (Darboux vector difference)
    - bend1 (κ₁), bend2 (κ₂), twist (τ)
  - Darboux vector from relative quaternion: ω = 2 * im(q₀⁻¹ * q₁) / L
- Added helper functions:
  - `_quat_rotate_vector()`: Rotate vector by quaternion
  - `_quat_conjugate()`: Quaternion inverse
  - `_quat_multiply()`: Quaternion multiplication

### Session 1 - compute_jacobians_direct (2026-01-23)
- Implemented `_jacobians_numpy()` method:
  - Computes Jacobian matrices relating constraints to DOFs
  - Per edge: J_pos (6×6) and J_rot (6×6)
  - **Stretch-Shear Jacobians**:
    - ∂C_ss/∂p₀ = -R^T, ∂C_ss/∂p₁ = R^T
    - ∂C_ss/∂θ₀ = L * R^T * [d3]_× (skew-symmetric)
  - **Bend-Twist Jacobians** (simplified):
    - ∂C_bt/∂θ₀ ≈ -2/L * I, ∂C_bt/∂θ₁ ≈ 2/L * I
- Added `_skew_symmetric()` helper

### Session 1 - assemble_jmjt_direct (2026-01-23)
- Implemented `_assemble_numpy()` method:
  - Builds system matrix A = J * M⁻¹ * J^T + α
  - Uses banded storage format for scipy.linalg.solve_banded
  - Bandwidth = 6 (one constraint block)
  - Assembles JMJT = Σ(inv_m * J_p * J_p^T + inv_I * J_θ * J_θ^T)
  - Adds compliance to diagonal for regularization

### Session 1 - solve_direct_constraints (2026-01-23)
- Implemented `_solve_numpy()` method:
  - Solves banded system: A * Δλ = -C using scipy.linalg.solve_banded
  - Falls back to dense solve if scipy not available
  - Applies corrections:
    - Position: Δp = M⁻¹ * J_pos^T * Δλ
    - Orientation: Δθ = I⁻¹ * J_rot^T * Δλ
  - `_apply_quaternion_correction()`: Converts tangent vector to quaternion update

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `simulation_direct_numpy.py` | NumPy simulation class with toggleable implementations |
| `example_dll_direct_cosserat_rod.py` | Side-by-side comparison demo |
| `DEFKIT_TO_NUMPY_CLAUDE.md` | This tracking document |

---

## Next Steps

The core constraint solving pipeline is now implemented in NumPy. Remaining tasks:

1. **Test the full NumPy solver** - Enable all checkboxes and compare behavior
   - Note: The NumPy solver may not match C++ exactly due to:
     - Simplified Jacobian approximations
     - Missing off-diagonal coupling in banded matrix
     - Different numerical precision

2. **Implement `predict_rotations`** - Quaternion integration:
   ```python
   # Apply damping and torque
   ω *= (1 - damping)
   ω += dt * inv_I * τ
   # Integrate: q' = q + 0.5 * dt * [ω, 0] * q
   ```

3. **Implement `integrate_positions`** - Velocity derivation:
   ```python
   v = (p_predicted - p) / dt
   p = p_predicted
   ```

4. **Implement `integrate_rotations`** - Angular velocity derivation:
   ```python
   # ω = 2 * im(q_predicted * q^{-1}) / dt
   q_prev = q
   q = q_predicted
   ```

5. **Improve Jacobian accuracy** - Current implementation uses simplified derivatives

6. **Add off-diagonal coupling** - For more accurate banded matrix assembly
