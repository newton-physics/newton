# LIMX Dihedral Bending Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a four-particle dihedral-angle bending constraint to LIMX and enable it in the CUDA hanging-cloth example.

**Architecture:** `ConstraintDihedralBending` stores fixed four-particle topology and rest angles, evaluates exact wrapped-angle forces at the current Newton iterate, and assembles sixteen Gauss-Newton PSD 3-by-3 Hessian blocks per hinge. The existing solver, block-CSR matrix, block-Jacobi PCG, dynamic operator, and cross-frame warm start remain unchanged.

**Tech Stack:** Python, Warp CUDA kernels, NumPy reference calculations, Newton `SolverBase`, `unittest`.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-03-limx-dihedral-bending-design.md` exactly.
- Use `E = 0.5 * k * wrap(theta - theta_rest)^2` and physical force `-k * delta_theta * dtheta_dx`.
- Assemble only `k * outer(dtheta_dxi, dtheta_dxj)` into the Hessian; do not add the residual-weighted second derivative.
- Use all sixteen ordered 3-by-3 blocks per four-particle dihedral in fixed block-CSR.
- Use shared `k_bending = 0.01` and no bending damping in the example.
- Preserve `dt=0.01`, one Newton iteration, 50 PCG iterations, one render per physics step, `velocity_damping=1.0`, and cross-frame PCG warm start.
- Run routine tests and example validation on `cuda:0`; use CPU only to diagnose a CUDA failure.
- Add no dependencies, collision changes, rigid-body support, line search, full 12-by-12 eigensolver, or solver mode switch.

---

### Task 1: Dihedral Force and Gauss-Newton Hessian

**Files:**
- Create: `newton/_src/solvers/limx/constraints/dihedral_bending.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Consumes: `BlockCsrBuilder`, `BlockCsrMatrix`, Warp particle-position and Hessian arrays.
- Produces: `ConstraintDihedralBending(dihedral_indices, rest_positions, stiffness, particle_count, device)` with the standard LIMX static-constraint methods.

- [ ] **Step 1: Write CUDA tests for exact force and rest behavior**

Add `TestConstraintDihedralBending` to `newton/tests/test_solver_limx.py`. Use the literal rest fixture

```python
REST = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 1.0, 0.0], [0.75, -1.0, 0.0]],
    dtype=np.float32,
)
DIHEDRAL = [(0, 1, 2, 3)]
```

Construct on `cuda:0`, assert zero force at rest, and compare a deformed
fixture's force against the negative central-difference gradient of the
independent NumPy energy

```python
delta = np.arctan2(np.sin(theta - theta_rest), np.cos(theta - theta_rest))
energy = 0.5 * stiffness * delta * delta
```

The production mutations caught are a reversed normal, wrong four-point
ordering, unwrapped angle residual, wrong force sign, and missing stiffness.

- [ ] **Step 2: Run the focused class and verify RED**

Run:

```bash
/tmp/newton-main-merge-env.oQinDr/bin/python -m unittest \
  newton.tests.test_solver_limx.TestConstraintDihedralBending
```

Expected: import failure because `constraints.dihedral_bending` does not exist.

- [ ] **Step 3: Implement rest-angle precomputation and force assembly**

Create the constraint module with a shared Warp geometry function that computes

```text
theta = atan2((n1 cross n2) dot edge_hat, n1 dot n2)
J_i = t1_i n1 + t2_i n2
```

using the exact coefficients from the approved spec. The constructor computes
rest angles on the host with the same orientation and rejects invalid rest
geometry. The force kernel wraps the residual and atomically adds

```text
force_i = -stiffness * delta_theta * J_i.
```

- [ ] **Step 4: Run the force tests and verify GREEN**

Run the Task 1 focused command and require zero failures.

- [ ] **Step 5: Write CUDA tests for the projected Hessian and CSR pattern**

For a literal deformed fixture, reconstruct the dense 12-by-12 matrix from the
sixteen blocks and compare it with the independent NumPy reference

```python
expected = stiffness * np.outer(angle_gradient, angle_gradient)
```

Assert symmetry, minimum eigenvalue at least `-1e-5`, one positive eigenvalue,
sixteen ordered block coordinates, and unchanged CSR topology after
reassembly. Also compare against a numerical exact energy Hessian away from
rest and assert they differ, catching accidental restoration of the omitted
residual term.

- [ ] **Step 6: Run the Hessian tests and verify RED**

Run the Task 1 focused command. Expected: failure because Hessian structure,
binding, or assembly is absent.

- [ ] **Step 7: Implement sixteen-block Gauss-Newton assembly**

Register every ordered pair from each four-particle row, bind a flat
`(dihedral_count, 16)` slot array, and atomically add

```text
H_ij = stiffness * outer(J_i, J_j).
```

Check the bound matrix device and exact Hessian value-buffer length before
launching the kernel.

- [ ] **Step 8: Add validation and runtime-degeneracy tests**

Use table-driven cases for empty batches, non-four rows, repeated or
out-of-range indices, wrong rest-position count, nonfinite rest positions,
collapsed rest edge, zero rest height, and nonpositive/nonfinite stiffness.
Create a valid constraint, collapse its runtime geometry, assemble force and
Hessian, and assert every output remains finite and zero for that hinge.

- [ ] **Step 9: Run Task 1 CUDA tests and commit**

Run the focused CUDA class, then commit:

```bash
git add newton/_src/solvers/limx/constraints/dihedral_bending.py \
  newton/tests/test_solver_limx.py
git commit -m "Add LIMX dihedral bending constraint"
```

---

### Task 2: Public API and CUDA Example Integration

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/examples/cloth/example_cloth_limx.py`
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `CHANGELOG.md`
- Regenerate: `docs/api/newton_solvers.rst`

**Interfaces:**
- Consumes: `ConstraintDihedralBending` from Task 1 and public `newton.utils.MeshAdjacency`.
- Produces: `newton.solvers.ConstraintDihedralBending` and a `cloth_limx` scene containing one bending batch.

- [ ] **Step 1: Write failing public-export and example tests**

Extend `test_public_exports` to assert that the public symbol is the internal
class. Change the example constraint test to construct the example inside
`wp.ScopedDevice("cuda:0")` and require exactly one anchor batch,
one triangle-membrane batch, one dihedral-bending batch, no distance springs,
and `bending.stiffness == 0.01`.

The production mutations caught are an internal-only constraint, missing
example integration, duplicate bending batches, wrong stiffness, or a return
to distance springs.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
/tmp/newton-main-merge-env.oQinDr/bin/python -m unittest \
  newton.tests.test_solver_limx.TestSolverLIMX.test_public_exports \
  newton.tests.test_solver_limx.TestSolverLIMX.test_example_uses_membrane_and_dihedral_bending
```

Expected: failure because the public export and example batch do not exist.

- [ ] **Step 3: Export the class and build interior-edge topology**

Add the symbol to the constraint, LIMX, and lazy public solver export lists.
In the example, compute

```python
edge_rows = newton.utils.MeshAdjacency(triangles).edge_indices
interior_edges = edge_rows[edge_rows[:, 1] >= 0]
dihedral_indices = interior_edges[:, [2, 3, 0, 1]]
```

and append

```python
newton.solvers.ConstraintDihedralBending(
    dihedral_indices,
    positions,
    0.01,
    particle_count,
    self.model.device,
)
```

after the membrane batch.

- [ ] **Step 4: Document the public addition**

Insert an `[Unreleased] / Added` changelog entry describing LIMX dihedral
bending with exact force and Gauss-Newton PSD block-CSR Hessian. Regenerate API
docs with

```bash
/tmp/newton-main-merge-env.oQinDr/bin/python docs/generate_api.py
```

Preserve the original SPDX year in previously existing generated files.

- [ ] **Step 5: Run focused tests and commit**

Run the Task 2 focused CUDA tests, then commit:

```bash
git add newton/_src/solvers/limx/constraints/__init__.py \
  newton/_src/solvers/limx/__init__.py \
  newton/_src/solvers/__init__.py \
  newton/examples/cloth/example_cloth_limx.py \
  newton/tests/test_solver_limx.py CHANGELOG.md docs/api/newton_solvers.rst
git commit -m "Enable bending in LIMX cloth"
```

---

### Task 3: CUDA Integration Verification

**Files:**
- Verify all files changed by Tasks 1 and 2.

**Interfaces:**
- Confirms the complete constraint, public API, block-CSR assembly, and hanging-cloth behavior.

- [ ] **Step 1: Run the bending and example tests on CUDA**

```bash
/tmp/newton-main-merge-env.oQinDr/bin/python -m unittest \
  newton.tests.test_solver_limx.TestConstraintDihedralBending \
  newton.tests.test_solver_limx.TestSolverLIMX.test_public_exports \
  newton.tests.test_solver_limx.TestSolverLIMX.test_example_uses_membrane_and_dihedral_bending \
  newton.tests.test_solver_limx.TestSolverLIMX.test_example_cuda_graph_advances_odd_substep_state
```

Do not add a routine CPU invocation. If CUDA fails, write a regression test for
the observed failure before using CPU to isolate it.

- [ ] **Step 2: Run 100 headless CUDA frames**

```bash
/tmp/newton-main-merge-env.oQinDr/bin/python -m newton.examples cloth_limx \
  --viewer null --test --num-frames 100 --device cuda:0
```

Require a zero exit code and finite final-state checks.

- [ ] **Step 3: Run formatting and lint checks**

```bash
/tmp/newton-main-merge-tools.WoYBEn/uvx pre-commit run -a
```

- [ ] **Step 4: Inspect final branch state**

Run `git diff --check`, inspect the branch log and working tree, and verify that
the user's uncommitted `lessons.md` changes and untracked
`solver_convergence.png` were neither staged nor reverted.
