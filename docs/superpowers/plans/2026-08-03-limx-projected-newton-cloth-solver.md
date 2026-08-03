# LIMX Projected-Newton Cloth Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LIMX's constant-Hessian projective-dynamics loop with a projected-Newton cloth solver that assembles current-position force and analytic PSD Hessian blocks into a fixed-pattern 3-by-3 block-CSR matrix.

**Architecture:** Static constraint topology creates the CSR pattern once, while constraint kernels clear and refill block values at every Newton iteration. `solver_newton.py` assembles the implicit-Euler residual at the current iterate, `CompositeLinearOperator` adds mass and future matrix-free dynamic terms, and block-Jacobi PCG solves the SPD projected system.

**Tech Stack:** Python, Warp kernels, NumPy reference assertions, Newton `SolverBase`, `unittest`.

## Global Constraints

- Keep implementation under `newton/_src/solvers/limx/`; expose public symbols only through `newton.solvers`.
- Keep `SolverLIMX`, `ConstraintAnchor`, and `ConstraintDistance` public names unchanged.
- Evaluate force and Hessian at the current Newton iterate; use predicted positions only in the inertial residual.
- Clamp only negative transverse distance-spring Hessian eigenvalues; never substitute `kI`.
- Keep fixed-topology elasticity in 3-by-3 block-CSR and the dynamic collision boundary matrix-free.
- Keep `velocity_damping=1.0` and add no Rayleigh damping, diagonal shift, line search, bending, rigid bodies, or collision implementation.
- Use `unittest`, not pytest, and introduce no dependency.

---

### Task 1: Mutable Fixed-Pattern Block-CSR

**Files:**
- Modify: `newton/_src/solvers/limx/block_csr.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Produces: `BlockCsrBuilder.ensure_block(row: int, column: int) -> None`.
- Produces: `BlockCsrMatrix.block_index(row: int, column: int) -> int`.
- Produces: `BlockCsrMatrix.clear_values() -> None` and `BlockCsrMatrix.update_diagonal() -> None`.
- Preserves: `BlockCsrBuilder.add_block`, `add_scaled_identity`, `finalize`, and `BlockCsrMatrix.multiply`.

- [ ] **Step 1: Write failing mutable-matrix tests**

Add tests that build a zero-valued `(0, 0)` and `(0, 1)` pattern, assert literal CSR indices, overwrite `values`, call `update_diagonal`, and verify multiplication and diagonal output. The production mutation caught is a matrix that preserves initial PD values or cannot map constraint blocks after sorting.

- [ ] **Step 2: Verify the tests fail for missing APIs**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx.TestBlockCsr
```

Expected: failure because `ensure_block`, `block_index`, `clear_values`, or `update_diagonal` does not exist.

- [ ] **Step 3: Implement mutable pattern/value separation**

Retain a host coordinate-to-sorted-index map on `BlockCsrMatrix`, create device diagonal block indices during `finalize`, and add a Warp kernel that copies current diagonal blocks from `values`. `clear_values` zeros both values and cached diagonal. `ensure_block` inserts an explicit zero block without changing numerical values.

- [ ] **Step 4: Verify block-CSR tests pass**

Run the Task 1 command and require zero failures.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/limx/block_csr.py newton/tests/test_solver_limx.py
git commit -m "Make LIMX block CSR values mutable"
```

---

### Task 2: Analytic Constraint Hessian Assembly

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/anchor.py`
- Modify: `newton/_src/solvers/limx/constraints/distance.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Produces on each constraint: `append_hessian_structure(builder: BlockCsrBuilder) -> None`.
- Produces on each constraint: `bind_hessian(matrix: BlockCsrMatrix) -> None`.
- Produces on each constraint: `accumulate_force_and_hessian(positions, force_output, hessian_values) -> None`.
- The distance constraint stores four CSR block indices per spring; the anchor stores one per anchor.

- [ ] **Step 1: Replace PD Hessian tests with projected-Newton red tests**

Use a one-spring x-axis fixture and literal expected matrices:

```python
# rest: radial k, zero transverse
want_rest = np.diag([5.0, 0.0, 0.0])
# 20% compression: exact transverse is -0.25 k, projected to zero
want_compressed = np.diag([5.0, 0.0, 0.0])
# 20% extension: transverse is k / 6
want_stretched = np.diag([5.0, 5.0 / 6.0, 5.0 / 6.0])
```

Assert all four assembled blocks are `[G, -G; -G, G]`, the forces are equal and opposite, the CSR pattern is unchanged across reassembly, and zero length remains finite with zero contribution. The production mutation caught is restoring `kI`, using predicted positions, clamping the radial eigenvalue, or clamping matrix entries instead of eigenvalues.

- [ ] **Step 2: Verify rest/compression tests fail against the PD matrix**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx.TestConstraintDistance
```

Expected: rest/compression assertions fail because the current implementation assembles `5 I` once and cannot update it from positions.

- [ ] **Step 3: Implement current-position force/Hessian kernels**

For `length > 1e-8`, compute

```text
normal_outer = n n^T
tangent = I - normal_outer
lambda_t = max(k (1 - rest_length / length), 0)
G = k normal_outer + lambda_t tangent
```

Atomically add force and the four signed blocks to the bound CSR indices. For a zero-length distance, return without adding force or Hessian. Implement the anchor's exact `kI` diagonal in its combined assembly kernel.

- [ ] **Step 4: Verify constraint tests pass on CPU and available CUDA**

Run the Task 2 command and require zero failures; the test itself loops over `cpu` and `cuda:0` when CUDA is available for the analytic Hessian fixtures.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/limx/constraints newton/tests/test_solver_limx.py
git commit -m "Assemble LIMX projected spring Hessians"
```

---

### Task 3: Current-Matrix Composite Operator

**Files:**
- Modify: `newton/_src/solvers/limx/operator.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Consumes: mutable `BlockCsrMatrix.values` and refreshed `BlockCsrMatrix.diagonal`.
- Preserves: `prepare(positions, dt)`, `multiply(vector, output)`, and `inverse_diagonal` used by `PcgSolver`.

- [ ] **Step 1: Write a failing reassembly operator test**

Prepare and multiply once, mutate the static block values, refresh the diagonal, prepare again, and assert a different hand-derived matrix-vector product and inverse block diagonal. The production mutation caught is caching the initial PD diagonal or operator values.

- [ ] **Step 2: Verify the test fails against stale diagonal behavior**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx.TestCompositeLinearOperator
```

Expected: failure because current code has no supported value-refresh path.

- [ ] **Step 3: Update operator terminology and preparation order**

Describe the matrix as current fixed-topology elasticity rather than fixed Hessian. Require the caller to refresh static diagonal after assembly, then add `M / h^2` and matrix-free dynamic diagonal in `prepare`.

- [ ] **Step 4: Verify operator and PCG tests pass**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx.TestCompositeLinearOperator
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx.TestPcgSolver
```

Require zero failures.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/limx/operator.py newton/tests/test_solver_limx.py
git commit -m "Use current LIMX Hessian values"
```

---

### Task 4: Replace PD Loop with Solver-Level Projected Newton

**Files:**
- Create: `newton/_src/solvers/limx/solver_newton.py`
- Delete: `newton/_src/solvers/limx/solver_limx.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Preserves: `SolverLIMX(model, constraints, nonlinear_iterations=4, linear_iterations=32, velocity_damping=1.0, dynamic_operator=None)`.
- Produces: current-iterate assembly loop defined by the design document.

- [ ] **Step 1: Write the large-rotation regression test**

Create two particles joined by one rest-length spring, anchor the first, and give the second a transverse velocity. Run one Newton iteration with enough PCG iterations. Assert the second point follows a substantial fraction of its transverse inertia prediction. The production mutation caught is evaluating at prediction or restoring the PD `kI` transverse block, either of which suppresses the first large-rotation increment.

- [ ] **Step 2: Verify the regression fails against current PD behavior**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx.TestSolverLIMX.test_current_position_hessian_does_not_lock_transverse_prediction
```

Expected: failure because the current fixed `kI` spring Hessian strongly suppresses transverse motion.

- [ ] **Step 3: Implement `solver_newton.py`**

Build and bind the CSR pattern during construction. On every nonlinear iteration: clear values, initialize `M / h^2 (y - x_k)`, call static force/Hessian assembly at `x_k`, call dynamic force assembly at `x_k`, refresh static diagonal, prepare the operator at `x_k`, zero-start PCG for `delta_x`, and update `x_k`. Keep `x_0 = x_n` and use `y` only in the inertial residual.

- [ ] **Step 4: Redirect internal import and remove PD module**

Import `SolverLIMX` from `.solver_newton` in `limx/__init__.py`; remove `solver_limx.py`. Preserve public imports through `newton.solvers`.

- [ ] **Step 5: Verify regression and focused integration tests pass**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
```

Require zero failures, finite cloth state, bounded edges, and unchanged input state.

- [ ] **Step 6: Commit**

```bash
git add newton/_src/solvers/limx newton/tests/test_solver_limx.py
git commit -m "Replace LIMX PD with projected Newton"
```

---

### Task 5: Example and Public Documentation

**Files:**
- Modify: `newton/examples/cloth/example_cloth_limx.py`
- Modify: `CHANGELOG.md`
- Modify as generated: `docs/api/newton_solvers.rst`

**Interfaces:**
- Preserves: `python -m newton.examples cloth_limx`.
- Changes: example iteration split from PD-specific `64 x 10` to projected-Newton `4 x 32`, subject only to evidence from the existing final-state assertions.

- [ ] **Step 1: Run the current example test as a behavioral baseline**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 60 --device cpu
```

Record whether the PD-specific iteration split passes before changing it.

- [ ] **Step 2: Update example terminology and iteration split**

Describe projected Newton, use four nonlinear and 32 PCG iterations, and retain the existing finite-state, anchor, sag, swing, and edge-bound assertions.

- [ ] **Step 3: Update changelog and generated API docs**

Change the existing unreleased LIMX entry to describe current-position projected-Newton Hessian assembly. Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh docs/generate_api.py
```

- [ ] **Step 4: Verify example on CPU and CUDA**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 60 --device cpu
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 60 --device cuda:0
```

Require both commands to exit zero.

- [ ] **Step 5: Commit**

```bash
git add newton/examples/cloth/example_cloth_limx.py CHANGELOG.md docs/api/newton_solvers.rst
git commit -m "Update LIMX projected Newton example"
```

---

### Task 6: Final Verification

**Files:**
- Verify all files changed by Tasks 1-5.

**Interfaces:**
- Confirms the complete user-visible solver behavior and repository quality gates.

- [ ] **Step 1: Run focused LIMX tests**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
```

- [ ] **Step 2: Run repository lint and formatting checks**

Use the project-preferred command when available:

```bash
uvx pre-commit run -a
```

If `uvx` is unavailable, run the configured pre-commit executable from an existing compatible environment and report the exact limitation rather than silently skipping it.

- [ ] **Step 3: Inspect the final diff and status**

```bash
git diff --check HEAD~4..HEAD
git status --short
```

Confirm only the intentional uncommitted project lesson log remains and that no generated caches or runtime artifacts are staged.
