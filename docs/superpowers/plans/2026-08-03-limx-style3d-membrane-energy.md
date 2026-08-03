# LIMX Style3D Membrane Energy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LIMX example's distance-spring elasticity with an energy-consistent anisotropic triangle membrane constraint using analytic projected-Newton Hessian blocks.

**Architecture:** `ConstraintTriangleElastic` owns fixed triangle rest data and assembles force plus nine 3-by-3 Hessian blocks per triangle at the current Newton iterate. It plugs into the unchanged constraint interface, block-CSR builder, composite operator, and PCG solver; the cloth example swaps only its elastic constraint batch.

**Tech Stack:** Python, Warp kernels, NumPy finite-difference references, Newton `SolverBase`, `unittest`.

## Global Constraints

- Keep implementation under `newton/_src/solvers/limx/constraints/`; expose the new class through `newton.solvers`.
- Use the energy and PSD Hessian projection in `docs/superpowers/specs/2026-08-03-limx-style3d-membrane-energy-design.md` exactly.
- Evaluate all elastic terms at the current Newton iterate and preserve static 3-by-3 block-CSR assembly.
- Keep dynamic collision terms matrix-free.
- Keep `dt=0.01`, one Newton iteration, 50 PCG iterations, cross-frame PCG warm start, and `velocity_damping=1.0` in the example.
- Add no bending, damping, rigid-body support, collision implementation, dependency, line search, or diagonal shift.
- Use `unittest`, not pytest.

---

### Task 1: Triangle Membrane Force and Hessian

**Files:**
- Create: `newton/_src/solvers/limx/constraints/triangle_elastic.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Produces: `ConstraintTriangleElastic(triangle_indices, inverse_rest_matrices, rest_areas, stiffnesses, particle_count, device)`.
- Produces: `append_hessian_structure(builder)`, `bind_hessian(matrix)`, `accumulate_force(positions, output)`, and `accumulate_force_and_hessian(positions, force_output, hessian_values)` matching existing static constraints.
- Stores: nine bound block-CSR value indices per triangle.

- [ ] **Step 1: Write force tests before production code**

Add `TestConstraintTriangleElastic` with an identity-rest triangle. Assert zero force after a rigid 3D rotation. For a literal deformed position fixture, compute the approved scalar energy in NumPy and central-difference each of nine position components; assert the Warp force equals the negative numerical gradient. The production mutations caught are normalized shear, wrong area scaling, wrong coefficient ordering, and a force sign error.

- [ ] **Step 2: Run force tests and verify RED**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx.TestConstraintTriangleElastic
```

Expected: import failure because `constraints.triangle_elastic` does not exist.

- [ ] **Step 3: Implement the minimal force path**

Create the constructor validation and a Warp force kernel using:

```text
Fu = sum_i a_i x_i
Fv = sum_i b_i x_i
c = Fu dot Fv
g_i = a_i Fv + b_i Fu
f_i = -A [ku a_i (||Fu||-1) nu + kv b_i (||Fv||-1) nv + ks c g_i].
```

Skip only a stretch-column term when its length is at most `1e-8`; keep shear finite.

- [ ] **Step 4: Run force tests and verify GREEN**

Run the Task 1 command and require the force tests to pass.

- [ ] **Step 5: Write Hessian tests and verify RED**

Bind one triangle into a `BlockCsrMatrix`, assemble at a compressed/sheared fixture, and reconstruct the dense 9-by-9 matrix from its nine blocks. Assert literal behavior:

```text
Qu+ = ku [nu nu^T + max(1 - 1/||Fu||, 0)(I - nu nu^T)]
Qv+ = kv [nv nv^T + max(1 - 1/||Fv||, 0)(I - nv nv^T)]
H_ij = A [a_i a_j Qu+ + b_i b_j Qv+ + ks g_i g_j^T].
```

Also assert symmetry, minimum eigenvalue at least `-1e-5`, nine-block topology, and unchanged CSR row/column arrays after reassembly. The production mutations caught are restoring the indefinite residual shear term, using `kI`, clamping entries instead of eigenvalues, and incomplete triangle connectivity.

Run the Task 1 command. Expected: failure because Hessian binding and assembly are not implemented.

- [ ] **Step 6: Implement projected Hessian assembly**

Register all ordered `(face[i], face[j])` coordinates, bind them to a flat `(triangle_count, 9)` integer array, and atomically add the analytic blocks above. Do not perform a numerical eigenvalue decomposition in the kernel: the stretch projection is analytic and the shear term is already `J^T J`.

- [ ] **Step 7: Verify triangle tests and commit**

Run the Task 1 command and require zero failures, then:

```bash
git add newton/_src/solvers/limx/constraints/triangle_elastic.py newton/tests/test_solver_limx.py
git commit -m "Add LIMX triangle membrane constraint"
```

---

### Task 2: Public API and Validation Coverage

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/solvers.py`
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `CHANGELOG.md`
- Modify as generated: `docs/api/newton_solvers.rst`

**Interfaces:**
- Produces: `newton.solvers.ConstraintTriangleElastic`.
- Preserves: `ConstraintAnchor`, `ConstraintDistance`, and `SolverLIMX`.

- [ ] **Step 1: Write public export and invalid-input tests**

Extend `test_public_exports` to assert identity with the internal class. Add table-driven constructor cases for repeated or out-of-range indices, mismatched lengths, nonpositive area, singular/nonfinite inverse rest matrices, negative/nonfinite stiffness, and mismatched runtime arrays. The production mutations caught are an internal-only API and accepting invalid rest topology that later produces nonfinite kernels.

- [ ] **Step 2: Run tests and verify RED**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
```

Expected: public-export failure while the focused triangle behavior from Task 1 remains green.

- [ ] **Step 3: Export the class and document the user-visible addition**

Add the class to all four `__init__`/public export lists. Insert an `[Unreleased]` `Added` entry describing anisotropic warp, weft, and shear membrane elasticity with PSD-projected block-CSR Hessians. Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh docs/generate_api.py
```

- [ ] **Step 4: Verify focused tests and commit**

Run the Task 2 test command and require zero failures, then:

```bash
git add newton/_src/solvers/limx newton/_src/solvers/__init__.py newton/solvers.py newton/tests/test_solver_limx.py CHANGELOG.md docs/api/newton_solvers.rst
git commit -m "Expose LIMX triangle elasticity"
```

---

### Task 3: Replace the Cloth Example Elasticity

**Files:**
- Modify: `newton/examples/cloth/example_cloth_limx.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Preserves: `python -m newton.examples cloth_limx`.
- Changes: the example's elastic constraint from unique triangle-edge springs to one triangle membrane batch.

- [ ] **Step 1: Write the example migration test**

Construct `ClothLimxExample` with `ViewerNull`, assert exactly one
`ConstraintTriangleElastic` and no `ConstraintDistance` in
`example.solver.constraints`, while retaining the existing `0.01` one-step
timing test. The production mutation caught is leaving mass-spring elasticity
active alongside the membrane energy.

- [ ] **Step 2: Run the migration test and verify RED**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx.TestSolverLIMX
```

Expected: failure because the example still constructs `ConstraintDistance`.

- [ ] **Step 3: Replace springs with triangle membrane rest data**

Remove unique-edge construction and pass the triangle list,
`model.tri_poses.numpy()`, `model.tri_areas.numpy()`, and per-triangle
`wp.vec3(1.0e4, 1.0e4, 1.0e3)` stiffness to
`ConstraintTriangleElastic`. Keep anchors and solver iteration parameters
unchanged. Update `test_final` to bound the three membrane invariants
`||Fu||`, `||Fv||`, and `abs(Fu dot Fv)` instead of spring edge lengths.

- [ ] **Step 4: Verify integration on CPU**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 100 --device cpu
```

Require zero failures and finite example state.

- [ ] **Step 5: Verify integration on CUDA and commit**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 100 --device cuda:0
```

Require zero exit status, then:

```bash
git add newton/examples/cloth/example_cloth_limx.py newton/tests/test_solver_limx.py
git commit -m "Use membrane energy in LIMX cloth"
```

---

### Task 4: Final Verification

**Files:**
- Verify all files changed by Tasks 1-3.

**Interfaces:**
- Confirms the complete constraint, public API, solver integration, and example behavior.

- [ ] **Step 1: Run the complete LIMX test module**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
```

- [ ] **Step 2: Run lint and format checks on changed files**

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m pre_commit run --files \
  newton/_src/solvers/limx/constraints/triangle_elastic.py \
  newton/_src/solvers/limx/constraints/__init__.py \
  newton/_src/solvers/limx/__init__.py \
  newton/_src/solvers/__init__.py \
  newton/solvers.py \
  newton/examples/cloth/example_cloth_limx.py \
  newton/tests/test_solver_limx.py \
  CHANGELOG.md \
  docs/api/newton_solvers.rst
```

- [ ] **Step 3: Inspect final status without touching user files**

```bash
git diff --check
git status --short --branch
```

Confirm `lessons.md` and `solver_convergence.png` remain unstaged and otherwise untouched.

