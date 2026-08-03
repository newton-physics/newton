# LIMX Projective-Dynamics Cloth Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent particle-only LIMX solver whose batched anchor and distance constraints feed a static `3 x 3` block-CSR elastic operator and whose block-Jacobi PCG holds a two-corner-anchored mass-spring cloth under gravity.

**Architecture:** Static elastic constraint batches own nonlinear force evaluation and emit one-time Hessian triplets into a generic block-CSR matrix. A composite operator adds mass and an empty matrix-free dynamic-constraint term, while a standalone PCG solver only consumes matrix-vector multiplication and inverse diagonal blocks. `SolverLIMX` lives beside Newton's existing internal solvers and is exported through the public `newton.solvers` module.

**Tech Stack:** Python 3.12, NVIDIA Warp 1.13, Newton public Python API, `unittest`, repository pre-commit hooks.

## Global Constraints

- Place implementation under `newton/_src/solvers/limx/`, expose user-facing classes through `newton.solvers`, and keep examples/tests in Newton's standard directories.
- Use Warp kernels and Newton's existing model/state interface; do not add dependencies or handwritten CUDA.
- Implement particle `vec3` unknowns only; do not add rigid-body 6-DoF types or placeholders.
- Assemble fixed-topology elastic Hessians once in `wp.mat33` block-CSR.
- Keep dynamic topology behind matrix-free force, Hessian-vector-product, and diagonal accumulation methods; the first implementation is a no-op.
- Use `unittest`, never pytest.
- Follow PEP 604 unions, bracket-style Warp array annotations, Google-style docstrings, and existing SPDX conventions.
- Write every behavioral test before its production implementation and observe the expected failure.
- Preserve the user's existing uncommitted `lessons.md` changes.

---

## File Map

- `newton/_src/solvers/limx/block_csr.py`: Host block-triplet assembly, device block-CSR storage, and Warp SpMV.
- `newton/_src/solvers/limx/constraints/anchor.py`: Batched one-particle anchor validation, Hessian emission, and force kernel.
- `newton/_src/solvers/limx/constraints/distance.py`: Batched two-particle spring validation, Hessian emission, and force kernel.
- `newton/_src/solvers/limx/operator.py`: Empty dynamic operator and mass/static/dynamic composite operator.
- `newton/_src/solvers/limx/linear_solver.py`: Preallocated, capture-safe block-Jacobi PCG.
- `newton/_src/solvers/limx/solver_limx.py`: Implicit Euler nonlinear loop and Newton `SolverBase` adapter.
- `newton/examples/cloth/example_cloth_limx.py`: Alternating-diagonal grid, constraints, viewer integration, and final-state checks.
- `newton/tests/test_solver_limx.py`: Focused unit and integration tests.
- `newton/solvers.py`: Public `SolverLIMX`, `ConstraintAnchor`, and `ConstraintDistance` exports.

### Task 1: Package skeleton and block-CSR matrix

**Files:**
- Create: `newton/_src/solvers/limx/__init__.py`
- Create: `newton/_src/solvers/limx/block_csr.py`
- Create: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Produces: `BlockCsrBuilder(row_count: int)`.
- Produces: `BlockCsrBuilder.add_block(row: int, column: int, value: wp.mat33) -> None`.
- Produces: `BlockCsrBuilder.add_scaled_identity(row: int, column: int, scale: float) -> None`.
- Produces: `BlockCsrBuilder.finalize(device: wp.DeviceLike) -> BlockCsrMatrix`.
- Produces: `BlockCsrMatrix.multiply(x: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None`.
- Produces arrays `row_offsets`, `column_indices`, `values`, and `diagonal`.

- [ ] **Step 1: Create package markers and write failing assembly/SpMV tests**

Create tests that add duplicate `(row, column)` identity blocks in unsorted order, finalize on CPU, and require sorted columns and summed values. Add a second test with one empty row and require block-CSR multiplication to match:

```python
builder = BlockCsrBuilder(3)
builder.add_scaled_identity(0, 0, 2.0)
builder.add_scaled_identity(0, 1, -1.0)
builder.add_scaled_identity(1, 0, -1.0)
builder.add_scaled_identity(1, 1, 3.0)
matrix = builder.finalize("cpu")
x = wp.array([wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0), wp.vec3(7.0)], dtype=wp.vec3)
output = wp.empty_like(x)
matrix.multiply(x, output)
np.testing.assert_allclose(output.numpy(), [[-2.0, -1.0, 0.0], [11.0, 13.0, 15.0], [0.0, 0.0, 0.0]])
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx_block_csr
```

Expected: import failure for missing `BlockCsrBuilder`, not an environment error.

- [ ] **Step 3: Implement minimal host assembly and device SpMV**

Use a dictionary keyed by `(row, column)` to merge blocks and sort keys by row then column. Validate indices, dimensions, and finite values. Upload `row_offsets`, `column_indices`, `values`, and extracted diagonal blocks. Implement:

```python
@wp.kernel
def _block_csr_multiply(
    row_offsets: wp.array[int],
    column_indices: wp.array[int],
    values: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    row = wp.tid()
    value = wp.vec3(0.0)
    for block in range(row_offsets[row], row_offsets[row + 1]):
        value += values[block] * x[column_indices[block]]
    output[row] = value
```

- [ ] **Step 4: Run the Task 1 test and verify GREEN**

- [ ] **Step 5: Commit Task 1**

```bash
git add newton/_src/solvers/limx newton/tests/test_solver_limx.py
git commit -m "Add LIMX block CSR matrix"
```

### Task 2: Batched anchor and distance constraints

**Files:**
- Create: `newton/_src/solvers/limx/constraints/__init__.py`
- Create: `newton/_src/solvers/limx/constraints/anchor.py`
- Create: `newton/_src/solvers/limx/constraints/distance.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Consumes: `BlockCsrBuilder.add_scaled_identity()` from Task 1.
- Produces: `ConstraintAnchor(indices, targets, stiffnesses, particle_count, device)`.
- Produces: `ConstraintDistance(index_pairs, rest_lengths, stiffnesses, particle_count, device)`.
- Both produce `append_hessian(builder) -> None` and `accumulate_force(positions, output) -> None`.

- [ ] **Step 1: Write failing force, Hessian, and validation tests**

Require an anchor displaced from `(1, 0, 0)` to `(1.5, 0, 0)` at stiffness `10` to produce `(-5, 0, 0)`. Require a spring with endpoints `(0, 0, 0)` and `(1.5, 0, 0)`, rest length `1`, and stiffness `20` to produce equal-and-opposite forces `(10, 0, 0)` and `(-10, 0, 0)`. Verify zero force at rest, finite zero-length behavior, the anchor `+kI` block, and distance blocks `(+kI, -kI; -kI, +kI)`. Reject mismatched lengths, invalid indices, repeated endpoints, non-positive rest lengths, and non-finite/non-positive stiffness.

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx_constraints
```

- [ ] **Step 3: Implement batched validation, force kernels, and Hessian emission**

Anchor force is `-k * (x-target)`. Distance force atomically adds `k * (length-rest) * direction` to the first endpoint and its negative to the second, returning early below `1e-8 m`. Constructors retain validated host definitions for assembly and upload batched device arrays for runtime kernels.

- [ ] **Step 4: Run constraint and CSR tests and verify GREEN**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
```

- [ ] **Step 5: Commit Task 2**

```bash
git add newton/_src/solvers/limx/constraints newton/tests/test_solver_limx.py
git commit -m "Add LIMX particle constraints"
```

### Task 3: Composite operator and capture-safe PCG

**Files:**
- Create: `newton/_src/solvers/limx/operator.py`
- Create: `newton/_src/solvers/limx/linear_solver.py`
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/_src/solvers/limx/__init__.py`

**Interfaces:**
- Consumes: `BlockCsrMatrix.multiply()` and `.diagonal` from Task 1.
- Produces: `EmptyDynamicConstraintOperator` with no-op force, Hessian-vector-product, and diagonal methods.
- Produces: `CompositeLinearOperator(masses, static_matrix, dynamic_operator, device)`.
- Produces: `prepare(positions, dt)`, `multiply(vector, output)`, and `inverse_diagonal`.
- Produces: `PcgSolver(dimension, device)` and `solve(operator, rhs, solution, iterations, zero_initial_guess=True, tolerance=None, check_interval=1) -> int`.

- [ ] **Step 1: Write failing operator and PCG tests**

Construct a two-particle block system with masses `[2, 3]`, `dt=0.5`, and static blocks `[[2I, -I], [-I, 4I]]`. Verify composite multiplication includes mass blocks `[8I, 12I]`. Choose an exact two-vector solution, compute the corresponding right-hand side, and require PCG to reproduce it within `rtol=1e-5`, `atol=1e-6`. Repeat with a nonzero initial guess and verify a zero right-hand side remains finite. Add a debug-mode solve with `tolerance=1e-6` and require it to return fewer iterations than a deliberately large maximum while preserving the solution.

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx_pcg
```

- [ ] **Step 3: Implement the empty dynamic boundary and composite operator**

`prepare()` forms and inverts each Jacobi block:

```python
diag = static_diagonal[i] + wp.identity(3, float) * masses[i] / (dt * dt)
dynamic_operator.accumulate_diagonal(positions, diag_blocks)
inverse_diagonal[i] = wp.inverse(diag)
```

`multiply()` performs static CSR SpMV, adds the mass product, and then asks the dynamic operator to add its Hessian product. The empty implementation must perform no allocation or launch.

- [ ] **Step 4: Implement PCG without host reads in its fixed-iteration path**

Preallocate `r`, `z`, `p`, `Ap`, `rz`, `rz_previous`, `pAp`, and one debug residual scalar. Implement dot products using a Warp reduction kernel with atomic addition into one-element arrays and clear those arrays before reuse. Compute guarded `alpha` and `beta` inside Warp kernels; use zero whenever a denominator is non-finite or below `1e-30`. When `tolerance is None`, execute the requested fixed count without host reads and return that count. Otherwise, every `check_interval` iterations reduce `r^T r`, read the single scalar on the host, and stop once it is below `tolerance**2`. Do not import constraint modules from `pcg.py`.

- [ ] **Step 5: Run Task 3 and prior tests and verify GREEN**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
```

- [ ] **Step 6: Commit Task 3**

```bash
git add newton/_src/solvers/limx newton/tests/test_solver_limx.py
git commit -m "Add LIMX composite PCG solve"
```

### Task 4: Newton solver adapter and cloth integration test

**Files:**
- Create: `newton/_src/solvers/limx/solver_limx.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/solvers.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Consumes: constraint batches, `BlockCsrBuilder`, `CompositeLinearOperator`, and `PcgSolver`.
- Produces: `SolverLIMX(model, constraints, nonlinear_iterations=4, linear_iterations=32, velocity_damping=1.0)`.
- Produces: standard `step(state_in, state_out, control, contacts, dt) -> None`.

- [ ] **Step 1: Write a failing solver integration test**

Build a `4 x 4`-cell horizontal alternating-diagonal grid with public `ModelBuilder.add_particles()` and `add_triangles()`. Give every particle positive mass, create unique-edge distance constraints, and anchor two same-side corners. Simulate `240` substeps at `dt=1/240 s` and require:

```python
np.testing.assert_allclose(positions[anchor_indices], anchor_targets, atol=1e-3)
self.assertLess(positions[center_index, 2], initial_center_z - 5e-2)
self.assertTrue(np.isfinite(positions).all())
self.assertTrue(np.isfinite(velocities).all())
self.assertLess(max_current_edge_length, 2.0 * max_rest_length)
self.assertGreater(positions[center_index, 2], initial_center_z - 0.5 * 9.81)
```

Also snapshot `state_in` before one call and assert `step()` does not mutate it.
Require the default output velocity to equal
`(state_out.particle_q-state_in.particle_q)/dt` without a damping multiplier.

- [ ] **Step 2: Run integration test and verify RED**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx_integration
```

- [ ] **Step 3: Implement time integration and solver construction**

Validate finite positive model particle masses. Ask all static batches to append Hessian blocks and finalize one CSR matrix in the constructor. Preallocate `x_previous`, `x_inertia`, `x_iter`, `rhs`, and `dx`.

Implement Warp kernels for:

```python
x_inertia = x + dt * velocity + dt * dt * (world_gravity + external_force / mass)
rhs = mass / (dt * dt) * (x_inertia - x_iter)
x_iter += dx
velocity_out = velocity_damping * (position_out - x_previous) / dt
```

Set `velocity_damping=1.0` by default so damping is explicitly opt-in.

Use `model.particle_world` and per-world `model.gravity`. Every nonlinear iteration rebuilds the right-hand side, accumulates static and dynamic force, prepares the operator, solves from a zero increment, and updates `x_iter`. Only final arrays are written to `state_out`.

- [ ] **Step 4: Run the complete focused suite and verify GREEN**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
```

- [ ] **Step 5: Commit Task 4**

```bash
git add newton/_src/solvers newton/solvers.py newton/tests/test_solver_limx.py
git commit -m "Add LIMX particle cloth solver"
```

### Task 5: Runnable viewer example and final verification

**Files:**
- Create: `newton/examples/cloth/example_cloth_limx.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: generated API files from `docs/generate_api.py`

**Interfaces:**
- Produces command `python -m newton.examples cloth_limx`.
- Produces an `Example` class following Newton's lifecycle and implementing `test_final()`.

- [ ] **Step 1: Verify the example module is initially absent**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 60 --device cpu
```

Expected: the Newton example runner reports that `cloth_limx` is not registered.

- [ ] **Step 2: Implement the example**

Create a `20 x 20`-cell, `1 m x 1 m` alternating-diagonal horizontal grid at `z=2 m`. Add positive-mass particles and render triangles to a Newton model, distribute total mass `0.3 kg` uniformly, and create every unique triangle edge exactly once. Construct two `ConstraintAnchor` instances at stiffness `1e7 N/m`, `ConstraintDistance` springs at `1e4 N/m`, and `SolverLIMX` with four nonlinear and 32 PCG iterations. Use four substeps per 60 Hz frame, no ground, and no collision generation.

Implement fixed-iteration CUDA graph capture, state swapping, `viewer.log_state()`, and `test_final()` checks for finite state, anchor drift below `1e-3 m`, center sag above `5e-2 m`, and bounded edge stretch.

- [ ] **Step 3: Run the headless example and focused suite**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 60 --device cpu
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
```

- [ ] **Step 4: Run CUDA verification when `cuda:0` is available**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 60 --device cuda:0
```

- [ ] **Step 5: Run project lint and formatting checks**

```bash
uvx pre-commit run -a
```

If `uvx` is unavailable, run an installed repository pre-commit executable when present and explicitly report the unavailable required command.

- [ ] **Step 6: Inspect and commit the example**

```bash
git diff --check
git status --short
git add newton/examples/cloth/example_cloth_limx.py newton/solvers.py newton/_src/solvers README.md CHANGELOG.md docs/api
git commit -m "Add LIMX hanging cloth example"
```

Do not stage or revert the user's existing `lessons.md` modification.
