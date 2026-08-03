# LIMX Dihedral Bending Design

## Goal

Add four-particle dihedral-angle bending to the LIMX cloth solver so the
two-corner hanging cloth resists mesh-scale folding while preserving the
existing anisotropic membrane energy, fixed-topology 3-by-3 block-CSR
assembly, projected-Newton solve, block-Jacobi PCG, and cross-frame warm
start.

The implementation follows the bending geometry in
`/home/limx/github/Ai-Phyiscs/src/constraint/cloth/bending_constraint.cu`.
It uses the exact elastic force and a Gauss-Newton positive-semidefinite
Hessian approximation.

## Scope

This change adds one public static constraint batch,
`newton.solvers.ConstraintDihedralBending`, and enables it in the existing
`cloth_limx` example.

The first version deliberately has:

- one scalar stiffness shared by all dihedrals;
- `k_bending = 0.01` in the example;
- no bending damping;
- no full 12-by-12 exact-Hessian eigendecomposition;
- no changes to collision, rigid bodies, PCG, time integration, or warm
  starting;
- no routine CPU validation.

CPU execution is a diagnostic fallback only after a CUDA failure or when a
kernel or numerical issue must be isolated.

## Public Constraint Interface

Add:

```python
ConstraintDihedralBending(
    dihedral_indices,
    rest_positions,
    stiffness,
    particle_count,
    device,
)
```

`dihedral_indices[e]` contains four distinct particle indices in the order

```text
[edge_v0, edge_v1, left_opposite, right_opposite].
```

The left triangle contains the directed edge `edge_v0 -> edge_v1`; the right
triangle contains the reverse directed edge. `rest_positions` contains one
finite 3D position per particle. The constructor evaluates and stores the
signed rest angle of each dihedral. `stiffness` is one finite, positive scalar
shared by the batch.

The class implements the existing static-constraint protocol:

```python
append_hessian_structure(builder)
bind_hessian(matrix)
accumulate_force(positions, output)
accumulate_force_and_hessian(positions, force_output, hessian_values)
```

It lives in
`newton/_src/solvers/limx/constraints/dihedral_bending.py` and is exported
through `newton.solvers` beside `ConstraintAnchor`, `ConstraintDistance`, and
`ConstraintTriangleElastic`.

## Topology Construction

The constraint consumes explicit four-particle dihedrals and remains
independent of the model builder. The example derives those dihedrals through
the public `newton.utils.MeshAdjacency` utility:

```python
edge_rows = newton.utils.MeshAdjacency(triangles).edge_indices
interior_edges = edge_rows[edge_rows[:, 1] >= 0]
dihedral_indices = interior_edges[:, [2, 3, 0, 1]]
```

Newton edge rows are `[o0, o1, v0, v1]`. For consistently oriented triangles,
`o0` belongs to the triangle containing directed edge `v0 -> v1`, so the
column permutation above matches the Ai-Phyiscs convention exactly. Boundary
edges are excluded because they have no second opposite vertex.

Every dihedral registers all sixteen ordered particle-pair block coordinates.
The existing `BlockCsrBuilder` deduplicates coordinates shared by multiple
constraints or neighboring hinges. Runtime kernels atomically accumulate all
four forces and all sixteen 3-by-3 Hessian blocks.

## Dihedral Geometry

For current positions `x0`, `x1`, `x2`, and `x3`, define

```text
e10 = x1 - x0
e20 = x2 - x0
e30 = x3 - x0
e_hat = e10 / ||e10||
n1 = normalize(e20 cross e10)
n2 = normalize(e10 cross e30)
```

The signed angle is

```text
theta = atan2((n1 cross n2) dot e_hat, n1 dot n2).
```

Let

```text
omega1 = (e10 dot e20) / ||e10||^2
omega2 = (e10 dot e30) / ||e10||^2
h1 = ||e20 - omega1 e10||
h2 = ||e30 - omega2 e10||.
```

The Bridson shape coefficients are

```text
t1 = (omega1 - 1, -omega1, 1, 0) / h1
t2 = (omega2 - 1, -omega2, 0, 1) / h2
```

and the four angle gradients are

```text
J_i = d theta / d x_i = t1_i n1 + t2_i n2.
```

The Warp kernels use this analytic expression directly. They do not compute a
finite-difference gradient at runtime.

## Energy and Force

The raw difference `theta - theta_rest` is not used directly because signed
angles cross the `atan2` branch cut at plus or minus pi. Use the shortest
wrapped residual

```text
delta_theta = atan2(
    sin(theta - theta_rest),
    cos(theta - theta_rest),
).
```

The per-dihedral energy is

```text
E_b = 1/2 k delta_theta^2.
```

Away from the unavoidable branch point at exactly plus or minus pi, the
physical force is

```text
f_i = -k delta_theta J_i.
```

The force is evaluated at the current Newton iterate, not at the inertial
prediction. It is accumulated into the same right-hand side as anchor and
membrane forces.

## Projected Hessian

The exact energy Hessian is

```text
H_exact = k [J^T J + delta_theta Hessian(theta)].
```

The residual-weighted second term is indefinite in general. The first version
follows Ai-Phyiscs and drops that complete term:

```text
H_ij_plus = k outer(J_i, J_j).
```

This Gauss-Newton matrix is positive semidefinite and is exact at the rest
angle, where `delta_theta` is zero. It also preserves the SPD requirement of
the existing PCG operator after the positive inertial term is added.

This approximation can remove positive curvature contained in the residual
term when a hinge is far from rest. A future higher-cost path may form the
complete local 12-by-12 Hessian and clamp only its negative eigenvalues. That
path is outside this change; the first version does not add a mode switch or
unused eigensolver scaffolding.

## Degenerate Geometry and Validation

The constructor rejects:

- an empty dihedral batch;
- a `rest_positions` count different from `particle_count`;
- rows that do not contain exactly four distinct indices;
- negative or out-of-range particle indices;
- nonfinite rest positions;
- nonfinite or nonpositive stiffness;
- rest hinges with a collapsed shared edge or either opposite vertex at zero
  height over the shared edge.

Runtime geometry can temporarily become degenerate even when the rest shape
is valid. If the shared-edge length, either face-normal length, or either
opposite height is at most `1.0e-8`, the kernel skips that dihedral's force and
Hessian contribution for the current assembly. This prevents NaNs from
poisoning the global PCG solve.

Binding and runtime-array checks mirror the existing triangle constraint:
particle counts and devices must match, `bind_hessian()` must run before
assembly, and the Hessian values buffer must have the finalized matrix's exact
block count.

## Example Integration

The `cloth_limx` example keeps:

- the 21-by-21 grid and alternating triangulation;
- two top-corner anchor constraints;
- anisotropic triangle membrane stiffness `(1.0e4, 1.0e4, 1.0e3)`;
- `dt = 0.01`;
- one Newton iteration;
- 50 PCG iterations;
- one render per physics step;
- `velocity_damping = 1.0`;
- the previous frame's PCG increment as the next frame's initial guess.

It appends one `ConstraintDihedralBending` batch after the membrane constraint,
using all interior edges, the original flat particle positions, and
`stiffness = 0.01`. It adds no bending damping or other dissipation.

## Tests and Completion Criteria

Use `unittest` and real Warp CUDA kernels. Normal validation targets `cuda:0`;
new CUDA-specific tests skip only when CUDA is unavailable. CPU is used only
after a CUDA failure for diagnosis.

The focused tests establish:

- a flat, consistently oriented two-triangle patch produces one correctly
  ordered dihedral with zero rest angle;
- force is zero at rest and remains zero after a rigid transformation of both
  rest and current geometry;
- analytic force matches the negative finite-difference gradient of the
  wrapped scalar energy at a literal deformed fixture;
- a rest angle immediately below plus pi and a current angle immediately
  above minus pi produce a small wrapped residual rather than a nearly
  two-pi residual;
- the dense local 12-by-12 matrix equals `k J^T J`, is symmetric, and has no
  eigenvalue below numerical tolerance;
- the implementation does not restore the residual-weighted exact-Hessian
  term away from rest;
- every dihedral registers sixteen ordered blocks, and reassembly changes
  values without changing CSR row offsets or column indices;
- shared global block coordinates accumulate contributions from multiple
  hinges;
- all constructor, binding, device, array-size, and runtime-degeneracy cases
  follow the behavior described above;
- `newton.solvers.ConstraintDihedralBending` resolves to the internal class;
- the example contains one dihedral-bending batch with stiffness `0.01` and
  still contains its membrane and anchor batches;
- the CUDA example completes 100 frames with finite positions and velocities,
  fixed anchors, bounded membrane deformation, and active bending resistance.

Run the focused LIMX module and the headless example on CUDA. Run the repository
formatting and lint checks on all changed files. Do not add a routine CPU
example run to the completion gate.
