# LIMX Projected-Newton Cloth Solver Design

## Goal

Replace LIMX's projective-dynamics local/global iteration with an implicit
projected-Newton solver for particle mass-spring cloth. At every nonlinear
iteration, elastic forces and Hessians are evaluated at the current iterate,
assembled into a fixed-pattern 3-by-3 block-CSR matrix, and solved with PCG.
The inertial prediction is used only by the implicit-Euler residual.

The initial milestone remains a particle-only cloth with two soft anchor
constraints and triangle-edge distance springs. Rigid bodies, bending, and
collisions are outside this change. The existing matrix-free dynamic-operator
boundary remains available for later collision work.

## Chosen Approach

Use a projected-Newton method with analytic distance-spring Hessians. Preserve
the exact Hessian in extension and at rest, and clamp only its negative
transverse eigenvalues in compression. The projected elastic Hessian is
positive semidefinite; adding the positive mass matrix makes the PCG operator
positive definite.

Two alternatives were rejected:

- Keeping the constant projective-dynamics `kI` matrix would preserve cheap
  global solves but reintroduce the slow transverse fixed-point mode that this
  change is intended to remove.
- Keeping the exact indefinite spring Hessian would require replacing PCG with
  an indefinite solver such as MINRES and would make the initial large-step
  cloth solver less robust.

## File Structure and Public API

The Newton algorithm is a solver-level implementation, parallel in role to the
existing solver implementations:

```text
newton/_src/solvers/
├── style3d/
│   └── solver_style3d.py
└── limx/
    ├── solver_newton.py
    ├── block_csr.py
    ├── operator.py
    ├── linear_solver.py
    └── constraints/
        ├── anchor.py
        └── distance.py
```

`solver_newton.py` owns time integration and nonlinear Newton iterations.
Constraint modules do not perform time integration or linear solves.
`block_csr.py` and `linear_solver.py` contain no spring-specific logic.

The public symbols remain `SolverLIMX`, `ConstraintAnchor`, and
`ConstraintDistance`. `newton._src` is internal, so replacing
`solver_limx.py` with `solver_newton.py` does not require a public deprecation.

## Implicit-Euler Newton System

For previous position `x_n`, previous velocity `v_n`, external acceleration
`a_ext`, and time step `h`, compute the inertial prediction once:

```text
y = x_n + h v_n + h^2 a_ext
```

Initialize the Newton iterate with the current position:

```text
x_0 = x_n
```

At nonlinear iteration `k`, assemble only from `x_k`:

```text
b_k = M / h^2 (y - x_k) + f_elastic(x_k) + f_dynamic(x_k)
A_k = M / h^2 + H_elastic_projected(x_k) + H_dynamic(x_k)
A_k delta_x = b_k
x_{k+1} = x_k + delta_x
```

No constraint may evaluate force or Hessian at `y`. The inertial prediction is
not an elastic linearization point. The initial version takes a full Newton
step and does not add a line search, Rayleigh damping, diagonal shift, or PD
local projection.

Each PCG solve starts from zero because the solved variable is the new Newton
increment for a changed matrix and right-hand side. `velocity_damping` remains
at its existing default of `1.0`; the update is
`v_{n+1} = (x_{n+1} - x_n) / h` when the multiplier is one.

## Distance-Spring Force and Projected Hessian

For one distance spring,

```text
d = x_j - x_i
l = ||d||
n = d / l
E = k / 2 (l - l_0)^2
```

the forces are

```text
f_i =  k (l - l_0) n
f_j = -f_i
```

and the exact relative Hessian is

```text
G = k n n^T + k (1 - l_0 / l) (I - n n^T).
```

Its radial eigenvalue is `k`. Its two transverse eigenvalues are
`k (1 - l_0 / l)`. The projected Hessian clamps only those transverse
eigenvalues:

```text
lambda_t = max(k (1 - l_0 / l), 0)
G_plus = k n n^T + lambda_t (I - n n^T)
```

The assembled two-particle blocks are

```text
[ G_plus  -G_plus ]
[ -G_plus  G_plus ]
```

This is not `kI`: at rest, `G_plus = k n n^T`, so transverse curvature is
zero. In compression, the radial eigenvalue remains `k` and the two negative
transverse eigenvalues become zero. For `l <= 1e-8`, where the original energy
is nondifferentiable, the initial implementation contributes zero force and
zero Hessian for that spring rather than inserting an unrelated `kI` block.

## Anchor Constraint

For an anchor target `t`,

```text
E_anchor = k_a / 2 ||x - t||^2
f_anchor = -k_a (x - t)
H_anchor = k_a I.
```

The anchor Hessian is already positive definite and is assembled without
projection. Anchors remain soft penalties in this milestone.

## Dynamic Block-CSR Assembly

Constraint topology defines the block-CSR sparsity pattern once during solver
construction. Matrix values do not remain fixed.

Initialization performs these operations:

1. Ask every static constraint batch to register its required block
   coordinates.
2. Finalize sorted CSR row offsets and column indices.
3. Resolve every constraint's block coordinates to CSR value indices and copy
   those indices to the simulation device.

Every Newton iteration then:

1. Zeros the block values and force buffer.
2. Adds the inertial residual.
3. Launches each constraint batch to atomically accumulate force and its four
   or one Hessian blocks from the current iterate.
4. Extracts current diagonal blocks for the block-Jacobi preconditioner.
5. Adds mass and matrix-free dynamic diagonal contributions.
6. Runs PCG and applies the resulting increment.

The matrix-free dynamic operator continues to receive the same current iterate
for force, Hessian-vector products, and diagonal accumulation. It remains an
empty implementation in the first cloth example.

## Error Handling

- Reject nonpositive time steps, nonpositive iteration counts, nonfinite
  masses, inactive particles, and rigid bodies as before.
- Reject constraints whose particle count or device differs from the model.
- Keep the zero-length spring branch finite and deterministic.
- PCG retains its existing zero-denominator guards. The projected static
  Hessian plus positive mass term is required to be SPD; no hidden diagonal
  regularization is added.

## Tests

Use `unittest` and test real Warp kernels on CPU, with targeted CUDA coverage
when CUDA is available.

1. Verify a stretched spring's four assembled blocks equal the analytic exact
   Hessian.
2. Verify a spring at rest has radial `k` and zero transverse curvature, not
   `kI`.
3. Verify a compressed spring clamps both negative transverse eigenvalues to
   zero while retaining radial `k`.
4. Verify force and Hessian accumulation remain finite for a zero-length
   spring.
5. Verify repeated assembly updates matrix values when the current positions
   rotate or change length while the CSR pattern remains unchanged.
6. Verify a one-spring large-rotation step is not suppressed by the former PD
   transverse `kI` term. This is the regression test that must fail against the
   current implementation before production code changes.
7. Verify the two-corner anchored cloth remains finite, anchors remain within
   tolerance, the center falls, and edge lengths remain bounded.
8. Run the `cloth_limx` example headlessly on CPU and CUDA. Restore a Newton-
   appropriate nonlinear/linear iteration split instead of the current
   PD-specific `64 x 10` split.

## Documentation and Compatibility

Update module and public docstrings to describe projected Newton rather than
projective dynamics. Update the existing `[Unreleased]` changelog entry instead
of adding a second entry for the same unreleased solver. Regenerate API docs if
the public documentation output changes. No new dependency is introduced.

## Completion Criteria

The change is complete when `SolverLIMX` contains no PD projection or constant
`kI` spring-Hessian path, every elastic force and Hessian is evaluated at the
current iterate, the analytic PSD-projected Hessian tests pass, the large-
rotation regression demonstrates the removed transverse locking, the focused
LIMX suite passes, and the cloth example runs on CPU and CUDA without nonfinite
state.
