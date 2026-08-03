# LIMX Projective-Dynamics Cloth Solver Design

## Goal

Build a first particle-only cloth solver under
`newton/_src/solvers/limx/`, alongside Newton's existing solvers. The solver
must keep constraint evaluation, static Hessian assembly,
linear-operator application, and PCG solution independent from one another.
The first runnable scene is a triangulated mass-spring sheet released under
gravity and held by two single-particle anchor constraints.

## Scope

The first version includes:

- Newton `Model` and `State` integration through a `SolverBase` subclass.
- Batched single-particle anchor constraints.
- Batched two-particle distance constraints on every unique triangle edge.
- A one-time assembled `3 x 3` block-CSR matrix for fixed-topology elastic
  Hessian approximations.
- A composite linear operator that combines mass, static block-CSR elasticity,
  and an optional matrix-free dynamic term.
- A standalone block-Jacobi-preconditioned PCG implementation.
- Public `SolverLIMX`, `ConstraintAnchor`, and `ConstraintDistance` exports
  through `newton.solvers`.
- A viewer example launched with `python -m newton.examples cloth_limx`.
- Unit and integration tests written with `unittest`.

The first version intentionally excludes:

- Rigid-body degrees of freedom or rigid-body constraints.
- Collision detection and collision response.
- Three-particle area constraints, anisotropic triangle elasticity, bending,
  damping constraints, tearing, and plasticity.
- Exact Dirichlet elimination for pinned particles. Anchors use a finite,
  configurable penalty stiffness and are verified by displacement tolerance.

Rigid bodies will use a separate algorithm in the future and must not leave
placeholder 6-DoF types or unused abstractions in this implementation.

## Mathematical Model

Let `x` be the current particle positions, `v` the previous velocities, `h`
the substep duration, and `M` the diagonal particle mass matrix. The inertial
target is

\[
x_{inertia} = x_n + h v_n + h^2 M^{-1} f_{external}.
\]

At every nonlinear iteration, solve for a position increment `dx`:

\[
A(x)\,dx = b(x),
\]

with

\[
A(x) = M/h^2 + P_{elastic} + H_{dynamic}(x)
\]

and

\[
b(x) = M/h^2(x_{inertia}-x)
       + f_{elastic}(x)
       + f_{dynamic}(x).
\]

`P_elastic` is a fixed symmetric-positive-semidefinite projective-dynamics
approximation assembled once from the static elastic constraints.
`H_dynamic` is a matrix-free operator intended for future collision energies;
it is empty in the first version. The positive mass term makes the free-particle
system suitable for PCG.

### Anchor constraint

For particle `i`, target `a`, and stiffness `ka`:

\[
E_a(x_i) = ka/2 \|x_i-a\|^2,
\]

\[
f_i = -ka(x_i-a), \qquad P_{ii} = ka I_3.
\]

Anchor particles remain ordinary positive-mass active particles. The example
must not set their Newton particle flags to inactive or overwrite their
positions after a solve.

### Distance constraint

For particles `i` and `j`, rest length `L`, stiffness `ks`, displacement
`d = x_j-x_i`, length `l = ||d||`, and direction `n = d/l`:

\[
E_s(x_i,x_j) = ks/2 (l-L)^2,
\]

\[
f_i = ks(l-L)n, \qquad f_j=-f_i.
\]

When `l` is below `1e-8 m`, the force contribution is zero to avoid an
undefined direction. The fixed SPD approximation is

\[
P_e = ks
\begin{bmatrix}
 I_3 & -I_3 \\
-I_3 &  I_3
\end{bmatrix}.
\]

The solver deliberately does not use the exact distance-spring Hessian because
that Hessian can become indefinite under compression and is therefore not a
safe PCG operator.

## Architecture

### Static constraint batches

Static constraints are grouped by type so each type uses one Warp kernel
launch per force evaluation rather than one launch per constraint.

`ConstraintAnchor` owns a batched set of anchors:

- particle indices;
- target positions;
- per-anchor stiffness values;
- `append_hessian(builder)` for one-time block triplet generation;
- `accumulate_force(positions, output)` for runtime force accumulation.

`ConstraintDistance` owns a batched set of springs:

- pairs of particle indices;
- rest lengths;
- per-spring stiffness values;
- `append_hessian(builder)` for one-time block triplet generation;
- `accumulate_force(positions, output)` for runtime force accumulation.

Both batches validate matching array lengths, finite positive stiffness, and
particle indices inside the model particle range before device arrays are
created.

### Static block-CSR

`BlockCsrBuilder` accepts `(row, column, wp.mat33)` triplets. It merges repeated
row-column blocks, sorts columns within each row, and produces:

- `row_offsets: wp.array[int]`, length `particle_count + 1`;
- `column_indices: wp.array[int]`, length `block_nnz`;
- `values: wp.array[wp.mat33]`, length `block_nnz`;
- `diagonal: wp.array[wp.mat33]`, length `particle_count`.

Assembly happens once on the host from host-owned constraint definitions and
is then uploaded to the selected Warp device. There is no fixed neighbor cap.
All runtime sparse matrix-vector multiplication happens on the model device.

The first constraints only emit scalar multiples of `I3`, but the stored block
type remains `wp.mat33` so later particle-only anisotropic or multi-particle
constraints do not require a new matrix format.

### Dynamic constraints

Future topology-changing collision energies implement a matrix-free boundary:

- `accumulate_force(positions, output)`;
- `hessian_multiply(positions, vector, output)`;
- `accumulate_diagonal(positions, output)`.

The first version uses an empty dynamic operator. It must implement the same
boundary as a no-op so `SolverLIMX` and `PcgSolver` contain no collision-specific
branches beyond selecting the no-op implementation.

### Composite linear operator

`CompositeLinearOperator.multiply(vector, output)` evaluates

\[
output = (M/h^2)vector + P_{elastic}vector + H_{dynamic}vector.
\]

The operator also builds the block-Jacobi diagonal

\[
D_i = (m_i/h^2)I_3 + (P_{elastic})_{ii} + (H_{dynamic})_{ii}
\]

and stores `inverse(D_i)` for active positive-mass particles. Linear-solver code
depends only on this operator contract and never imports concrete constraints.

### PCG

`PcgSolver` owns preallocated device arrays for residual, preconditioned
residual, search direction, matrix-vector product, solution, and scalar
reductions. It performs the standard PCG recurrence:

\[
r=b-Ax,\quad z=D^{-1}r,\quad
p=z+\beta p,\quad
\alpha=(r^Tz)/(p^TAp).
\]

The default runtime mode executes a fixed number of iterations and performs no
host readback, preserving CUDA Graph capture. A debug/test mode may check the
residual every configurable number of iterations; that mode is documented as
host-synchronizing and is not used by the viewer example.

Every scalar dot product uses 256-particle tiled blocks. Threads first reduce
their products cooperatively inside the block with `wp.tile_sum`; only lane zero
per block atomically adds the block sum to the global scalar. This bounds global
atomic contention to `ceil(particle_count/256)` operations while handling a
partial final block with zero-filled tile loads.

If `r^Tz` or `p^TAp` is non-finite or has absolute value below `1e-30`, the
iteration applies zero `alpha`/`beta` and terminates safely in debug mode. No
code path may write NaN to the solution.

## Time-Stepping Data Flow

`SolverLIMX` derives from Newton's `SolverBase` and implements the standard
`step(state_in, state_out, control, contacts, dt)` signature. `control` and
`contacts` are unused in the first version.

One substep performs:

1. Save the original input positions in `x_previous`.
2. Compute `x_inertia` from input position, velocity, gravity, and external
   particle force.
3. Copy input positions into an internal `x_iter` array.
4. For every nonlinear iteration:
   - initialize `rhs` with the inertia contribution;
   - launch each static constraint batch to add elastic force;
   - call the no-op dynamic constraint to add dynamic force;
   - update the composite operator and block-Jacobi inverse diagonal;
   - solve for `dx` with PCG;
   - update `x_iter += dx`.
5. Copy `x_iter` to `state_out.particle_q`.
6. Set `state_out.particle_qd = velocity_damping *
   (state_out.particle_q-x_previous)/dt`.

`velocity_damping` defaults to `1.0`, so the default update is the undamped
position finite difference. Values below one remain available as an explicit
opt-in.

`state_in.particle_q` and `state_in.particle_qd` are not modified. Solver
scratch arrays are preallocated in the constructor and reused by every step.

## File Layout

```text
newton/
├── _src/solvers/limx/
│   ├── __init__.py
│   ├── constraints/
│   │   ├── __init__.py
│   │   ├── anchor.py
│   │   └── distance.py
│   ├── block_csr.py
│   ├── operator.py
│   ├── linear_solver.py
│   └── solver_limx.py
├── examples/cloth/example_cloth_limx.py
├── tests/test_solver_limx.py
└── solvers.py
```

Implementation modules may use Newton internals because they live under
`newton._src`. The example and generated documentation must import only the
public `newton.solvers` symbols.

## Example Scene

`python -m newton.examples cloth_limx` creates a `1 m x 1 m` horizontal
triangulated grid at `z = 2 m`. Adjacent cells alternate their triangle
diagonal direction to reduce mesh-direction bias. Every unique triangle edge
becomes one distance constraint, including cell diagonals. The two corners on
one side of the sheet become anchor constraints targeting their initial
positions.

Initial defaults are:

- `20 x 20` cells;
- areal density `0.3 kg/m^2`, distributed uniformly to particles;
- distance stiffness `1e4 N/m`;
- anchor stiffness `1e7 N/m`;
- frame rate `60 Hz` with `4` substeps;
- `64` nonlinear iterations per substep;
- `10` PCG iterations per nonlinear iteration;
- velocity damping `1.0` (no damping by default).

The example has no ground plane and does not call collision generation. It
implements `test_final()` to verify finite particle state, bounded anchor drift,
visible center sag, and that the center passes the anchor line within the first
second. The crossing check distinguishes genuine pendulum-like motion from an
under-converged solve that merely creeps toward a hanging equilibrium.

The higher nonlinear budget is intentional. A distance constraint uses the
fixed projective-dynamics block `k I3`; at a spring's rest state this is much
stiffer than the exact elastic Hessian in directions transverse to the spring.
Four local-global updates therefore suppress the sheet's large rotational mode.
The `64 x 10` split was selected from trajectory measurements of the full
`21 x 21`-particle example: after one second its center has crossed the anchor
line and retains nonzero speed. Keeping ten PCG iterations per nonlinear update
limits the total linear work while matching Style3D's default inner budget.

## Testing

All tests use `unittest` and run through the repository's available Newton test
environment.

Constraint tests verify:

- anchor force is zero at its target and restoring away from the target;
- anchor Hessian is `ka I3`;
- a distance spring has zero force at rest length;
- stretched distance-spring forces are equal and opposite;
- the distance Hessian contains the four expected `+ks I3` and `-ks I3`
  blocks;
- a zero-length spring does not generate NaN.

Sparse-matrix tests verify:

- duplicate triplets accumulate into one block;
- columns are sorted per row;
- block-CSR multiplication matches a hand-computed dense result;
- matrices with empty rows are handled safely.

PCG tests verify:

- a known SPD block system converges to the expected solution;
- a nonzero initial guess produces the same solution;
- breakdown guards leave all outputs finite.
- tiled dot products remain correct across multiple blocks and a partial tail.

Solver integration tests construct a small cloth without a viewer, simulate at
least one second, and require:

- both anchor displacements remain below `1e-3 m`;
- the center particle falls at least `5e-2 m` below its initial height;
- the full example's center passes the anchor line within one second;
- every particle position and velocity remains finite;
- the maximum spring length remains below twice its rest length;
- the center particle does not undergo unrestricted ballistic free fall.

Before completion, run the focused `limx` unittest suite, the runnable example's
`test_final()`, and the repository pre-commit command required by project
instructions.

## Performance Decisions

- Constraints are batched by type.
- Static Hessian topology and values are assembled once.
- CSR has no padded per-row capacity and no fixed neighbor limit.
- Solver and PCG arrays are preallocated.
- Runtime mode has no per-iteration host synchronization.
- Dynamic topology is matrix-free rather than rebuilding the static CSR.
- The first version favors correctness and modularity over custom CUDA or
  private Warp runtime calls. Kernel timing will identify later optimization
  work.

## Acceptance Criteria

The design is complete when Newton contains a `SolverLIMX` whose public anchor
and distance constraint batches generate forces and static
Hessian blocks, whose block-CSR and composite operator feed an independent
PCG solver, and whose two-corner-anchored cloth visibly swings past its anchor
line under gravity without either anchor escaping or the sheet entering
ballistic free fall.
