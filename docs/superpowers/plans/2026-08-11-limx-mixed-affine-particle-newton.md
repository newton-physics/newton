# LIMX Mixed Affine–Particle Newton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native 12-DOF affine-body rows beside LIMX's existing 3-DOF particle rows, solve both spaces with one heterogeneous PCG method, and validate a PSD-ARAP affine body without collision.

**Architecture:** Keep the existing particle CSR and solver unchanged. Add a specialized 12-by-12 CSR and affine operator, then combine particle and affine vectors only at the heterogeneous PCG layer. Mixed dynamic terms use a two-space matrix-free interface; no 3-by-12 CSR is introduced. Affine inertia and PSD-projected ARAP assemble directly into native 12-by-12 body blocks.

**Tech Stack:** Python 3.10+, Warp custom vector/matrix types and CUDA kernels, NumPy references, `unittest`, Newton LIMX, `uv`, and `uvx`.

## Global Constraints

- Work in the canonical `/home/limx/github/newton` checkout on `dev`; do not create a worktree.
- Preserve the existing 3-by-3 particle CSR, `PcgSolver`, and `SolverLIMX` behavior.
- Store affine rows natively as `vec12` and `mat1212`; never encode one affine body as four particle rows.
- Use one global PCG recurrence over split particle and affine arrays.
- Use a native 3-by-3 particle preconditioner and native 12-by-12 affine Cholesky preconditioner.
- Project the analytic ARAP `9 x 9` deformation Hessian to PSD before embedding it in a 12-by-12 body block.
- Do not add DCD, IPC, friction, joints, motors, CCD, or mixed collision response in this milestone.
- Add no required or optional dependency.
- Use `unittest`; every test method has a triple-double-quoted imperative docstring.
- Examples and docs import only public Newton modules, never `newton._src`.
- Use `uv` for Python commands and `uvx pre-commit` for lint and formatting.
- Preserve unrelated changes in the T-shirt test, example image, and `solver_convergence.png`.
- Use 2026 for SPDX copyright lines on new files.

## File Map

- Create `newton/_src/solvers/limx/affine_types.py`: concrete `vec12`, `mat1212`, and small tested conversion helpers.
- Create `newton/_src/solvers/limx/block_csr_12.py`: native 12-by-12 CSR matrix and builder.
- Create `newton/_src/solvers/limx/mixed_operator.py`: split particle/affine operator contract and empty mixed dynamic operator.
- Create `newton/_src/solvers/limx/mixed_linear_solver.py`: heterogeneous PCG and 12-by-12 Cholesky preconditioner.
- Create `newton/_src/solvers/limx/affine_body.py`: affine body descriptors, volume moments, state arrays, and surface reconstruction.
- Create `newton/_src/solvers/limx/constraints/affine_arap.py`: direct-affine PSD ARAP force and 12-by-12 Hessian assembly.
- Create `newton/_src/solvers/limx/solver_affine.py`: collision-free affine Newton stepping using the mixed infrastructure.
- Modify LIMX and public solver export modules for `AffineBodyModel` and `SolverLIMXAffine`.
- Create `newton/tests/test_solver_limx_affine.py`: focused CPU/CUDA tests for every new layer.
- Create `newton/examples/basic/example_basic_limx_affine_body.py`: one tetrahedral affine body in free fall with an initially perturbed affine matrix.
- Create `newton/tests/test_example_basic_limx_affine_body.py`: short CUDA rollout.
- Modify `README.md`, `CHANGELOG.md`, and generated solver API documentation when the public symbols and example are ready.

---

### Task 1: Native 12-DOF Types and CSR

**Files:**
- Create: `newton/_src/solvers/limx/affine_types.py`
- Create: `newton/_src/solvers/limx/block_csr_12.py`
- Create: `newton/tests/test_solver_limx_affine.py`

**Interfaces:**
- Produces `vec12`, `mat1212`, `BlockCsrMatrix12`, and `BlockCsrBuilder12`.
- `BlockCsrMatrix12.multiply(x: wp.array[vec12], output: wp.array[vec12]) -> None` mirrors the existing 3-by-3 API.
- `BlockCsrBuilder12` provides `add_block()`, `ensure_block()`, `ensure_stencil_blocks()`, and `finalize()`. The finalized `BlockCsrMatrix12` provides stable `block_index()` and `stencil_block_indices()` lookup; the mutable builder intentionally exposes no indices because later pattern additions can reorder them.

- [ ] **Step 1: Write failing type and CSR tests**

Add `TestAffineBlockCsr` with literal two-row block data. Assert sorted columns, duplicate accumulation, diagonal extraction, clearing, and SpMV against a hand-built NumPy 24-by-24 matrix. Assert finalized-matrix block lookup and stencil lookup, including rejection of absent coordinates. Include constructor rejection tests for zero rows, out-of-range indices, wrong shapes, and non-finite values.

```python
def test_multiplies_native_twelve_dof_blocks(self):
    """Multiply native affine blocks without particle-row expansion."""
    builder = BlockCsrBuilder12(2)
    builder.add_block(0, 0, np.eye(12, dtype=np.float32) * 2.0)
    builder.add_block(0, 1, np.eye(12, dtype=np.float32) * -1.0)
    builder.add_block(1, 0, np.eye(12, dtype=np.float32) * -1.0)
    builder.add_block(1, 1, np.eye(12, dtype=np.float32) * 3.0)
    matrix = builder.finalize("cpu")
    x = wp.array([vec12(*range(12)), vec12(*range(12, 24))], dtype=vec12, device="cpu")
    output = wp.empty_like(x)
    matrix.multiply(x, output)
    expected = np.concatenate((2.0 * np.arange(12) - np.arange(12, 24), -np.arange(12) + 3.0 * np.arange(12, 24)))
    np.testing.assert_allclose(output.numpy().reshape(-1), expected)
```

- [ ] **Step 2: Run the test and verify RED**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx_affine.TestAffineBlockCsr -v
```

Expected: import failure because the affine type and CSR modules do not exist.

- [ ] **Step 3: Implement the minimal concrete types and CSR**

Define:

```python
class vec12(wp.types.vector(length=12, dtype=wp.float32)):
    """Twelve-component affine generalized vector."""


class mat1212(wp.types.matrix(shape=(12, 12), dtype=wp.float32)):
    """Twelve-by-twelve affine generalized matrix."""
```

Implement specialized Warp kernels for CSR multiply and diagonal refresh. Keep CPU pattern construction equivalent to `BlockCsrBuilder`, but validate and store `(12, 12)` arrays with `dtype=mat1212`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command and require all tests to pass.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/limx/affine_types.py newton/_src/solvers/limx/block_csr_12.py newton/tests/test_solver_limx_affine.py
git commit -m "Add native affine block CSR"
```

---

### Task 2: Native 12-by-12 Cholesky Preconditioner

**Files:**
- Create: `newton/_src/solvers/limx/mixed_linear_solver.py`
- Modify: `newton/tests/test_solver_limx_affine.py`

**Interfaces:**
- Produces `_factor_affine_diagonal(diagonal, factors, regularization)` and `_apply_affine_preconditioner(factors, residual, output)` kernels.
- Factor storage is one lower-triangular `mat1212` per affine row.
- Factorization reports one integer regularization flag per row for diagnostics.

- [ ] **Step 1: Write failing factorization tests**

Test a literal SPD matrix `A = R.T @ R + 0.5 * I`, solve three literal residuals, and assert `A @ z == r` within `rtol=2e-4`. Add a semidefinite matrix test that requires finite output and a set regularization flag, plus a non-finite-input rejection test.

- [ ] **Step 2: Run the factorization tests and verify RED**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx_affine.TestAffinePreconditioner -v
```

Expected: missing factorization functions.

- [ ] **Step 3: Implement fixed-size Cholesky and triangular solves**

Symmetrize each block as `(A + A.T) / 2`. Attempt Cholesky with a pivot floor equal to `max(1e-8, 1e-6 * trace(A) / 12)`. If any pivot is non-finite or below the floor, restart once with that floor added to every diagonal entry and mark the row. Apply forward and backward substitution without forming an explicit inverse.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command and require all tests to pass on CPU and CUDA when available.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/limx/mixed_linear_solver.py newton/tests/test_solver_limx_affine.py
git commit -m "Add affine block preconditioner"
```

---

### Task 3: Heterogeneous Operator and PCG

**Files:**
- Create: `newton/_src/solvers/limx/mixed_operator.py`
- Modify: `newton/_src/solvers/limx/mixed_linear_solver.py`
- Modify: `newton/tests/test_solver_limx_affine.py`

**Interfaces:**
- Produces `MixedVector3x12(particle: wp.array[wp.vec3], affine: wp.array[vec12])` as a lightweight pair used at Python call boundaries.
- Produces `MixedLinearOperator` owning a particle `CompositeLinearOperator`, affine `BlockCsrMatrix12`, affine Cholesky factors, and `mixed_dynamic_operator`.
- The dynamic operator contract is `multiply(particle_input, affine_input, particle_output, affine_output)` and `accumulate_diagonal(particle_diagonal, affine_diagonal)`.
- Produces `MixedPcgSolver(particle_count, affine_count, device)` with the same fixed-iteration and optional-tolerance behavior as `PcgSolver`.

- [ ] **Step 1: Write a failing coupled-system PCG test**

Construct one particle row and one affine row. Give the particle and affine static matrices positive diagonal blocks. Supply a test-only rank-one mixed operator with literal `j_p` and `j_a`, applying `k J.T @ J` matrix-free. Solve a hand-constructed 15-by-15 dense SPD reference system and compare both solution arrays to `np.linalg.solve`.

- [ ] **Step 2: Run the mixed-PCG test and verify RED**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx_affine.TestMixedPcg -v
```

Expected: missing mixed operator and solver classes.

- [ ] **Step 3: Implement split-vector PCG**

Implement split kernels for subtraction, preconditioning, direction updates, and solution/residual updates. Dot products reduce particle and affine arrays into the same scalar accumulator. `operator.multiply()` clears both outputs, applies both static spaces, then invokes the mixed dynamic operator. The zero-count side must be legal so the same solver supports particle-only, affine-only, and mixed tests.

- [ ] **Step 4: Run mixed and legacy PCG tests**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx_affine.TestMixedPcg -v
uv run --extra dev -m newton.tests -k test_solver_limx.TestPcg
```

Expected: new tests pass and legacy particle PCG remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/limx/mixed_operator.py newton/_src/solvers/limx/mixed_linear_solver.py newton/tests/test_solver_limx_affine.py
git commit -m "Add heterogeneous LIMX PCG"
```

---

### Task 4: Affine Body Data, Mass, and Surface Mapping

**Files:**
- Create: `newton/_src/solvers/limx/affine_body.py`
- Modify: `newton/tests/test_solver_limx_affine.py`

**Interfaces:**
- Produces `AffineBodyModel(rest_vertices, tetrahedron_indices, surface_triangle_indices, density, rigidity, initial_transform, device)`.
- Exposes constant `mass_matrices: wp.array[mat1212]`, `volumes`, rest surface vertices, surface ownership, and initial `q`/`qd` arrays.
- Provides `update_surface_positions(q, output)` using `x = t + A @ x_bar`.

- [ ] **Step 1: Write failing analytic mass and mapping tests**

Use the unit tetrahedron. Check total mass `density / 6`, full 12-by-12 symmetry and positive definiteness, generalized gravity producing exactly translational gravity with zero affine acceleration, identity mapping, and a literal affine transform mapping all four vertices.

- [ ] **Step 2: Run the body-data tests and verify RED**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx_affine.TestAffineBodyModel -v
```

Expected: missing `AffineBodyModel`.

- [ ] **Step 3: Implement exact tetrahedral dyadic integration**

Accumulate `int rho`, `int rho*x`, and `int rho*x*x.T` for every rest tetrahedron, build the full generalized mass matrix for the `[t, row(A,0), row(A,1), row(A,2)]` layout, validate positive volume and SPD mass, and upload arrays. Compute generalized gravity as `solve(M, F_body)` on the host during construction.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command.

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/limx/affine_body.py newton/tests/test_solver_limx_affine.py
git commit -m "Add affine body mass data"
```

---

### Task 5: PSD-Projected Direct Affine ARAP

**Files:**
- Create: `newton/_src/solvers/limx/constraints/affine_arap.py`
- Modify: `newton/tests/test_solver_limx_affine.py`

**Interfaces:**
- Produces `ConstraintAffineARAP(rigidities, volumes, body_count, device)`.
- Provides `append_hessian_structure(builder: BlockCsrBuilder12)`, `bind_hessian(matrix)`, and `accumulate_force_and_hessian(q, force, values)`.

- [ ] **Step 1: Write failing energy, derivative, and PSD tests**

Test zero energy/force for identity and a proper non-axis-aligned rotation. Compare force against centered finite differences for a literal shear/stretch matrix. Compare the analytic 9-by-9 Hessian to finite differences away from repeated singular values, then assert the assembled projected 12-by-12 block is symmetric PSD and has zero translation rows/columns.

- [ ] **Step 2: Run the ARAP tests and verify RED**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx_affine.TestAffineArap -v
```

Expected: missing affine ARAP constraint.

- [ ] **Step 3: Implement direct-affine signed SVD ARAP**

Reuse the established proper signed-SVD and analytical twist-mode Hessian formulas from `ConstraintTetrahedronARAP`, but evaluate them directly on `A`. Apply the existing generic `9 x 9` EVD projection before embedding the block in rows and columns `3:12` of `mat1212`. Accumulate physical force `-gradient` and the PSD Hessian into one diagonal affine CSR block per body.

- [ ] **Step 4: Run focused and tetrahedral ARAP regression tests**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx_affine.TestAffineArap -v
uv run --extra dev -m newton.tests -k test_constraint_tetrahedron_arap
```

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/limx/constraints/affine_arap.py newton/tests/test_solver_limx_affine.py
git commit -m "Add affine ARAP rigidification"
```

---

### Task 6: Collision-Free Affine Newton Stepper

**Files:**
- Create: `newton/_src/solvers/limx/solver_affine.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/tests/test_solver_limx_affine.py`

**Interfaces:**
- Produces public `SolverLIMXAffine(body_model, nonlinear_iterations=4, linear_iterations=32, velocity_damping=1.0)`.
- Owns `q`, `qd`, previous generalized state, inertial target, generalized RHS, and increment.
- Provides `step(dt: float) -> None` and `update_surface_positions(output: wp.array[wp.vec3]) -> None`.

- [ ] **Step 1: Write failing integration and graph-capture tests**

Verify 100 zero-rigidity free-fall steps against `0.5*g*t^2` within first-order integration tolerance and assert affine entries remain identity. Start a second body from `diag(1.1, 0.9, 1.05)` with nonzero rigidity and assert its maximum singular-value error decreases over 100 steps. Assert finite state, positive determinant, fixed PCG iteration count, and CUDA graph capture of one complete step.

- [ ] **Step 2: Run integration tests and verify RED**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx_affine.TestSolverLIMXAffine -v
```

Expected: missing solver class.

- [ ] **Step 3: Implement implicit-Euler Newton stepping**

Assemble `M/dt^2` and `M(q_tilde-q)/dt^2`, add affine ARAP force/Hessian, refresh the native 12-by-12 diagonal, factor it, solve with `MixedPcgSolver` using a zero particle side, apply the full increment, and recover `qd = damping * (q_new-q_old)/dt`. Do not add line search or collision.

- [ ] **Step 4: Run integration and focused LIMX tests**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx_affine.TestSolverLIMXAffine -v
uv run --extra dev -m newton.tests -k test_solver_limx
```

- [ ] **Step 5: Commit**

```bash
git add newton/_src/solvers/limx/solver_affine.py newton/_src/solvers/limx/__init__.py newton/_src/solvers/__init__.py newton/tests/test_solver_limx_affine.py
git commit -m "Add affine LIMX stepper"
```

---

### Task 7: Example, Public Documentation, and Final Verification

**Files:**
- Create: `newton/examples/basic/example_basic_limx_affine_body.py`
- Create: `newton/tests/test_example_basic_limx_affine_body.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: generated public solver API documentation
- Create: `docs/images/examples/example_basic_limx_affine_body.jpg`

**Interfaces:**
- Example command: `uv run -m newton.examples basic_limx_affine_body`.
- The example reconstructs a visible tetrahedral surface from `SolverLIMXAffine.q`, writes it into a standard Newton particle state for rendering, and implements `test_post_step()` and `test_final()`.

- [ ] **Step 1: Write the failing example test**

Run 100 CUDA frames with `ViewerNull`. Assert finite reconstructed vertices, positive affine determinant, downward center-of-mass motion, and lower final singular-value error than the initial perturbation.

- [ ] **Step 2: Run the example test and verify RED**

```bash
uv run --extra dev -m newton.tests -p test_example_basic_limx_affine_body.py
```

Expected: example import failure.

- [ ] **Step 3: Implement and register the example**

Use one body, `dt=0.01`, an initial visible shear, PSD ARAP, no collision, and a camera that clearly shows free fall and shape recovery. Add the public solver symbols, regenerate API docs, add the `[Unreleased] / Added` changelog entry, register the README command, and capture the required 320-by-320 screenshot.

- [ ] **Step 4: Run focused tests and all-file pre-commit**

```bash
uv run --extra dev -m newton.tests -p test_solver_limx_affine.py
uv run --extra dev -m newton.tests -p test_example_basic_limx_affine_body.py
uvx pre-commit run -a
git diff --check
```

Expected: zero test failures, zero hook failures, and no whitespace errors.

- [ ] **Step 5: Launch visual validation**

```bash
uv run -m newton.examples basic_limx_affine_body
```

Confirm visually that the body falls while the affine distortion relaxes toward a rigid shape without NaNs or inversion.

- [ ] **Step 6: Commit the milestone**

```bash
git add newton/examples/basic/example_basic_limx_affine_body.py newton/tests/test_example_basic_limx_affine_body.py README.md CHANGELOG.md docs/api docs/images/examples/example_basic_limx_affine_body.jpg
git commit -m "Demonstrate affine LIMX dynamics"
```
