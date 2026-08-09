# LIMX Tetrahedral ARAP Design

## Goal

Add tetrahedral As-Rigid-As-Possible (ARAP) elasticity to the existing
`SolverLIMX` projected-Newton particle solver. The first example is a
tetrahedral beam fixed at its left end and deforming under gravity.

ARAP is a four-particle static constraint, not a separate FEM solver. It reuses
the existing implicit-Euler inertia term, block-CSR assembly, block-Jacobi PCG,
and position/velocity update. Each time step performs exactly one Newton
iteration with a full step and no line search.

This milestone establishes an ARAP material implementation that can later be
shared conceptually with affine-body dynamics, without implementing affine
bodies or mixed FEM/affine-body degrees of freedom now.

## Scope

The milestone includes:

- a public `newton.solvers.ConstraintTetrahedronARAP` constraint;
- analytical tetrahedral ARAP energy derivatives;
- the full analytical `9 x 9` deformation-gradient Hessian;
- generic symmetric eigendecomposition and negative-eigenvalue clamping,
  matching the current libuipc implementation strategy;
- assembly of all sixteen ordered `3 x 3` particle-pair Hessian blocks;
- a fixed tetrahedral beam example using one Newton iteration per time step;
- focused CUDA validation of energy invariance, derivatives, PSD projection,
  assembly, and beam motion.

The milestone excludes:

- line search;
- multiple Newton iterations in the example;
- collision, contact, or self-collision;
- explicit material or velocity damping;
- inverted or degenerate tetrahedron recovery;
- affine-body dynamics and FEM/affine-body coupling;
- changes to the existing LIMX time integrator or PCG algorithm.

## Reference Behavior

The formulation follows libuipc's finite-element ARAP implementation:

- `docs/specification/constitutions/arap.md` defines
  `kappa * ||F - R||_F^2`;
- `src/backends/cuda/finite_element/constitutions/arap_function.h` defines the
  energy, gradient, and analytical deformation-gradient Hessian;
- `src/backends/cuda/finite_element/constitutions/arap_3d.cu` maps derivatives
  through `dF/dx` and calls `make_spd`;
- `src/backends/cuda/utils/make_spd.h` performs a full symmetric EVD, clamps
  negative eigenvalues to zero, and reconstructs the matrix.

The first Newton implementation deliberately retains the full `9 x 9` EVD.
An analytical twist-mode projection may replace it only after a separate
optimization change proves numerical equivalence against this baseline.

## Architecture

The new constraint lives at:

```text
newton/_src/solvers/limx/constraints/tetrahedron_arap.py
```

The same module contains private Warp functions for signed SVD, ARAP energy,
deformation-gradient derivatives, and PSD projection. The constraint calls
those functions during assembly, and focused tests call them through small
test kernels. No separate public material-energy API is introduced yet.

It follows the existing static-constraint protocol:

```python
class ConstraintTetrahedronARAP:
    def append_hessian_structure(self, builder): ...
    def bind_hessian(self, matrix): ...
    def accumulate_force(self, positions, output): ...
    def accumulate_force_and_hessian(
        self, positions, force_output, hessian_values
    ): ...
```

Construction accepts explicit tetrahedron data rather than owning a `Model`:

```python
ConstraintTetrahedronARAP(
    tetrahedron_indices,
    inverse_rest_matrices,
    stiffnesses,
    particle_count,
    device,
)
```

This is consistent with `ConstraintTriangleElastic` and avoids assigning new
semantics to `Model.tet_materials`. The example may source the arrays from
`model.tet_indices` and `model.tet_poses`, while providing its ARAP stiffness
explicitly.

The symbol is exported through the LIMX and public `newton.solvers` modules.
Examples must import it only through `newton.solvers`.

## ARAP Energy

For a tetrahedron with current vertices `x0`, `x1`, `x2`, and `x3`, define

```text
Ds = [x1 - x0, x2 - x0, x3 - x0]
F  = Ds * Dm_inverse
```

The inverse rest matrix is supplied by `Model.tet_poses`. The positive rest
volume is recovered once during construction as

```text
V0 = 1 / (6 * det(Dm_inverse)).
```

Use a signed singular-value decomposition

```text
F = U * diag(sigma) * V_transpose
R = U * V_transpose
```

with `U` and `V` proper rotations, matching libuipc's SVD convention. If Warp's
SVD produces a reflected basis, flip the last basis column and the associated
signed singular value so the factorization remains unchanged.

The element energy and deformation-gradient derivative are

```text
E      = kappa * V0 * ||F - R||_F^2
dE/dF  = 2 * kappa * V0 * (F - R).
```

The constraint adds physical force `-dE/dx` to the solver right-hand side.
`kappa` has units of pascals and must be finite and positive.

## Analytical Hessian and PSD Projection

Construct the exact unscaled `9 x 9` ARAP Hessian in column-major `vec(F)`
coordinates using libuipc's three normalized twist modes:

```text
H_F = 2 I
      - 4 / (sigma0 + sigma1) * t0 * t0_transpose
      - 4 / (sigma1 + sigma2) * t1 * t1_transpose
      - 4 / (sigma0 + sigma2) * t2 * t2_transpose.
```

The supported scene starts from positive, nondegenerate tetrahedra and must
remain positive-volume, so the singular-value sums are expected to stay away
from zero. A small denominator guard prevents non-finite output but is not an
inversion-recovery model.

Use `warp.fem.linalg.symmetric_eigenvalues_qr` on the complete symmetric
matrix. This function returns eigenvectors by row, so reconstruct with

```text
H_F_PSD = P_transpose * diag(max(eigenvalues, 0)) * P.
```

Finally multiply by `kappa * V0`. The first implementation must not replace
this operation with direct twist-mode clamping.

## Mapping to Particle Degrees of Freedom

The deformation-gradient Jacobian is constant for each tetrahedron. Define the
four material gradients from the rows of `Dm_inverse`:

```text
b1 = row0(Dm_inverse)
b2 = row1(Dm_inverse)
b3 = row2(Dm_inverse)
b0 = -(b1 + b2 + b3).
```

A vertex variation satisfies `delta_F = delta_x_i * b_i_transpose`. Therefore

```text
gradient_i = (dE/dF) * b_i
H_ij       = J_i_transpose * H_F_PSD * J_j.
```

The implementation writes four forces and all sixteen ordered Hessian blocks.
Every tetrahedron contributes those sixteen block coordinates to the existing
`BlockCsrBuilder`; duplicate coordinates across elements are accumulated by
atomic addition at runtime.

No solver changes are required. `SolverLIMX` continues to assemble

```text
M / dt^2 + H_ARAP + H_anchor
```

and solves once for the full Newton increment with its existing PCG path.

## Validation and Errors

Construction rejects:

- empty or length-mismatched arrays;
- tetrahedra without exactly four distinct in-range particle indices;
- non-finite or singular inverse rest matrices;
- non-positive recovered rest volumes;
- non-finite or non-positive stiffness values;
- runtime arrays with the wrong length, device, or Hessian storage size.

The focused numerical tests use CUDA and cover:

1. zero energy and force for rest states, rigid translations, and rigid
   rotations;
2. force balance and torque balance on a single tetrahedron;
3. energy-gradient agreement by centered finite differences;
4. the analytical unprojected `9 x 9` Hessian against a NumPy reference;
5. the projected Hessian against NumPy `eigh`, negative-eigenvalue clamping,
   and reconstruction;
6. assembled `12 x 12` particle Hessian symmetry and PSD behavior;
7. one `SolverLIMX` step with `nonlinear_iterations=1`;
8. a short fixed-beam rollout with finite state, bounded anchor error, visible
   free-end sag, and positive tetrahedron volumes.

Finite-difference Hessian checks use a deformation whose exact Hessian is
already positive semidefinite. A separate compressed deformation exercises
the PSD projection without incorrectly comparing the projected matrix to the
raw energy Hessian.

## Fixed-Beam Example

Add:

```text
newton/examples/softbody/example_softbody_limx_arap_beam.py
```

The initial scene uses:

- a `12 x 2 x 2` soft grid with `0.05 m` cells;
- dimensions `0.60 x 0.10 x 0.10 m`;
- density `1000 kg/m^3`;
- uniform ARAP stiffness `kappa = 1 MPa`;
- the leftmost particle layer anchored to its rest positions;
- quadratic anchor stiffness `1e8 N/m`;
- gravity `(0, 0, -9.81) m/s^2`;
- `dt = 0.01 s`, one physics step per rendered frame;
- `nonlinear_iterations = 1`;
- `linear_iterations = 128`;
- `velocity_damping = 1.0`;
- no collision objects or collision pipeline.

All particles retain finite positive mass. The beam does not use
`add_soft_grid(fix_left=True)`, because `SolverLIMX` intentionally rejects
zero-mass particles; the existing `ConstraintAnchor` supplies the fixed-end
condition.

The example renders the generated surface triangles, exposes the standard
Newton example parser, and implements `test_post_step()` and `test_final()`.
After visual acceptance it is registered in the example README with a
`320 x 320` screenshot.

## Future Affine Body Dynamics

An affine body uses twelve generalized coordinates: three translations and the
nine entries of a `3 x 3` affine map. Its ARAP energy uses the same
deformation-gradient energy, gradient, exact Hessian, and PSD projection. The
translation block receives no ARAP contribution; the affine `9 x 9` block does.

The current LIMX matrix stores uniform `3 x 3` particle blocks, so affine-body
dynamics is not added by disguising an affine body as four particles. A later
design will add mixed particle/affine degrees of freedom and FEM/affine-body
off-diagonal coupling to a shared global Newton system. That extension must
preserve the FEM constraint's energy convention and numerical reference tests
introduced here.

## Acceptance Criteria

The milestone is complete when:

- the public ARAP constraint builds and assembles on CUDA;
- its energy, force, exact Hessian, and PSD projection match their references;
- the example uses exactly one Newton iteration and no line search;
- the fixed beam visibly bends and oscillates under gravity without non-finite
  state or inverted tetrahedra in the tested rollout;
- focused tests and pre-commit checks pass;
- the interactive example is launched for visual review.
