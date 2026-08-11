# LIMX ARAP Bunny-on-Cloth Drop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a CUDA LIMX example where a deformable ARAP bunny falls onto a four-corner-pinned cloth sheet with two-way VF/EE contact.

**Architecture:** Append bunny tetrahedra and cloth triangles to one Newton particle model. Apply material-specific static constraints to explicit global element ranges and use one combined `ConstraintSelfCollision` as the matrix-free dynamic operator.

**Tech Stack:** Python, NumPy, Warp CUDA, Newton LIMX, `unittest`, OpenGL viewer.

## Global Constraints

- Use the canonical `dev` checkout; do not create a worktree.
- Use `unittest`, not pytest.
- Run CUDA validation by default.
- Do not add dependencies, a ground plane, or a second solver.
- Preserve unrelated screenshot, table-settling, and diagnostic-image changes.

---

### Task 1: Unified bunny and cloth example

**Files:**
- Create: `newton/examples/multiphysics/example_softbody_limx_arap_bunny_cloth.py`
- Test: `newton/tests/test_example_softbody_limx_arap_bunny_cloth.py`

**Interfaces:**
- Consumes: `bunny_tet.npz`, `ConstraintTetrahedronARAP`, `ConstraintTriangleElastic`, `ConstraintDihedralBending`, `ConstraintAnchor`, and `ConstraintSelfCollision`.
- Produces: standard `Example(viewer, args=None)` with `step()`, `render()`, `test_post_step()`, `test_final()`, and `create_parser()`.

- [ ] **Step 1: Write the failing configuration test**

Create a CUDA `unittest` that imports the missing example, constructs it with `ViewerNull(num_frames=1)`, and asserts four anchors, 40-by-40 cloth cells, ARAP/cloth element ranges, one Newton iteration, 50 PCG iterations, damping `1.0`, unsigned VF, geometry radius scale `0.25`, and contact capacity `262144`.

- [ ] **Step 2: Run the test to verify RED**

```bash
uv run --extra dev -m newton.tests -p test_example_softbody_limx_arap_bunny_cloth.py
```

Expected: import failure because the example module does not exist.

- [ ] **Step 3: Implement the unified example**

Load and append the existing tet bunny, generate an offset 41-by-41 cloth grid with alternating triangles, build global constraint indices, configure the combined collision operator, and implement the standard example lifecycle. Expose particle ranges, corner indices, center index, tetrahedra, and contact buffers for validation.

- [ ] **Step 4: Run the configuration test to verify GREEN**

Run the Task 1 command. Expected: the configuration and one-step finite checks pass on CUDA.

- [ ] **Step 5: Commit the example and configuration test**

```bash
git add newton/examples/multiphysics/example_softbody_limx_arap_bunny_cloth.py newton/tests/test_example_softbody_limx_arap_bunny_cloth.py
git commit -m "Add LIMX bunny cloth drop"
```

### Task 2: Contact rollout regression

**Files:**
- Modify: `newton/examples/multiphysics/example_softbody_limx_arap_bunny_cloth.py`
- Modify: `newton/tests/test_example_softbody_limx_arap_bunny_cloth.py`

**Interfaces:**
- Consumes: Task 1 element ranges and contact buffers.
- Produces: persistent cross-component contact flag and rollout diagnostics for bunny volume, cloth sag, finite state, overflow, and scene bounds.

- [ ] **Step 1: Write the failing 300-frame rollout test**

Assert that the bunny center descends into the cloth region, the cloth center sags, all tetrahedral determinants stay positive, VF/EE/EF buffers do not overflow, and at least one VF or EE contact couples a bunny particle with cloth particles.

- [ ] **Step 2: Run the rollout test to establish RED or the first physical failure**

```bash
uv run --extra dev -m newton.tests -p test_example_softbody_limx_arap_bunny_cloth.py -k supports_bunny
```

Expected before diagnostic implementation: failure because cross-component contact and rollout metrics are unavailable.

- [ ] **Step 3: Implement diagnostics and minimally tune scene placement**

Add CUDA contact marking and per-step validation. Adjust only initial bunny height/orientation or cloth placement if the bunny misses the cloth; do not add damping, a table, extra substeps, or change the collision formulation.

- [ ] **Step 4: Run the complete example test**

Run the Task 1 command. Expected: all tests pass with zero overflow and positive bunny volumes.

- [ ] **Step 5: Commit rollout validation**

```bash
git add newton/examples/multiphysics/example_softbody_limx_arap_bunny_cloth.py newton/tests/test_example_softbody_limx_arap_bunny_cloth.py
git commit -m "Validate LIMX bunny cloth contact"
```

### Task 3: Documentation, screenshot, and interactive validation

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Create: `docs/images/examples/example_softbody_limx_arap_bunny_cloth.jpg`

**Interfaces:**
- Consumes: validated Task 2 example.
- Produces: registered command, 320-by-320 gallery image, and Unreleased changelog entry.

- [ ] **Step 1: Register the example and changelog entry**

Add `uv run -m newton.examples softbody_limx_arap_bunny_cloth` to the relevant README gallery and an Added entry describing the two-way LIMX bunny-cloth example.

- [ ] **Step 2: Run pre-commit and focused tests**

```bash
uvx pre-commit run --files CHANGELOG.md README.md newton/examples/multiphysics/example_softbody_limx_arap_bunny_cloth.py newton/tests/test_example_softbody_limx_arap_bunny_cloth.py
uv run --extra dev -m newton.tests -p test_example_softbody_limx_arap_bunny_cloth.py
```

Expected: all checks pass.

- [ ] **Step 3: Launch the interactive scene**

```bash
uv run -m newton.examples softbody_limx_arap_bunny_cloth
```

Confirm the window, GUI controls, bunny deformation, cloth sag, and absence of visible pass-through.

- [ ] **Step 4: Capture and validate the screenshot**

Write a representative 320-by-320 JPEG and verify its dimensions.

- [ ] **Step 5: Commit documentation assets**

```bash
git add CHANGELOG.md README.md docs/images/examples/example_softbody_limx_arap_bunny_cloth.jpg
git commit -m "Document LIMX bunny cloth drop"
```
