# LIMX Oriented Volume-Surface Collision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in signed VF/EE contact using outward tetrahedral boundary normals, including same-bunny self-contact, without CCD.

**Architecture:** Extend `ConstraintSelfCollision` with a backward-compatible `use_outward_normals` flag. Filter only strictly incident pairs that share primitive indices, freeze oriented signed contact data during `prepare()`, and reuse the existing balanced force and PSD Hessian assembly.

**Tech Stack:** Python 3, NumPy, Warp, Newton LIMX, `MeshAdjacency`, `unittest`.

## Global Constraints

- Keep `use_outward_normals=False` as the default cloth-compatible behavior.
- Apply signed VF/EE uniformly to same-component and cross-component pairs.
- Filter only incident VF/EE pairs that share primitive indices; retain
  nonincident pairs regardless of graph distance.
- Use the current discrete BVH band and fixed 3 mm bunny thickness.
- Do not add CCD, swept queries, EF recovery, substeps, damping, or line search.
- Assemble force, Hessian-vector products, and diagonal blocks from the same frozen outward direction.
- Use `unittest`; every new test method has an imperative triple-double-quoted docstring.

---

### Task 1: Drive oriented VF and topology filtering with failing tests

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`

**Interfaces:**
- Consumes: `ConstraintSelfCollision(model, ..., use_outward_normals=True)`.
- Produces: signed VF contact direction and depth.

- [ ] **Step 1: Add an inside-VF regression test**

Construct two disconnected triangles. Put one source vertex at signed distance
`-0.05 m` behind a target `+Z` face with thickness `0.1 m`. Assert that the
oriented contact has direction `(0, 0, 1)`, depth `0.15 m`, and a positive-Z
vertex force. The current constructor must fail on the missing keyword.

- [ ] **Step 2: Add topology exclusion and detection-band regression tests**

Construct a one-ring fixture whose candidate vertex projects inside its
neighbor within thickness. Assert that oriented mode retains the pair because
the vertex is not part of the target face. Also reject a deep point outside
the signed-distance band.

- [ ] **Step 3: Verify both tests fail for the intended reason**

Run the two named `unittest` methods with `-v`. Expect
`TypeError: unexpected keyword argument 'use_outward_normals'`.

- [ ] **Step 4: Implement the oriented VF path**

Add the optional constructor flag. Keep the shared-index incident rejection,
the outward triangle normal, and `effective_thickness - signed_distance`
inside the absolute-distance detection band. Keep the old absolute-value
branch unchanged when the flag is disabled.

- [ ] **Step 5: Run the focused VF tests and the full self-collision test class**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection -v
```

Expected: all tests pass.

### Task 2: Add oriented EE pseudo-normal response

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`

**Interfaces:**
- Consumes: oriented triangle winding and `MeshAdjacency.edge_tri_indices`.
- Produces: signed EE direction/depth stored in `_EdgeEdgeContactBuffer`.

- [ ] **Step 1: Add an already-crossed EE regression test**

Build two disconnected oriented triangle patches. Arrange their central edges
within `0.1 m` after crossing, with opposing outward pseudo-normals. Assert the
force direction on edge 0 follows `normalize(ne1 - ne0)` and the depth equals
`thickness - dot(closest0 - closest1, direction)`.

- [ ] **Step 2: Add nonincident one-ring EE retention tests**

Reuse the adjacent-opposite-edge fixture and enable outward normals. Assert
that the matching EE contact is stored because the two edges share no endpoint.

- [ ] **Step 3: Verify both tests fail against the unsigned EE implementation**

Run the two named tests. Expect a wrong direction/depth before the oriented EE
path exists.

- [ ] **Step 4: Implement incident-face pseudo-normals and signed EE depth**

Store `edge_tri_indices` on the constraint and pass it plus triangle topology
to the EE kernel. Reject only pairs sharing an endpoint, compute both edge
pseudo-normals from current incident face normals, set direction to
`normalize(ne1 - ne0)`, and set signed depth. Preserve the old closest-vector
direction, local thickness clamp, and mollifier when oriented mode is off.

- [ ] **Step 5: Run the full self-collision test class**

Use the Task 1 command and require zero failures.

### Task 3: Enable the bunny scene and verify runtime behavior

**Files:**
- Modify: `newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py`
- Modify: `newton/tests/test_example_softbody_limx_arap_bunnies_box.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `use_outward_normals=True`.
- Produces: an oriented, self-frictionless, VF/EE-only eight-bunny experiment;
  floor and wall friction remain `0.05`.

- [ ] **Step 1: Add a failing scene configuration assertion**

Assert `example.self_collision.use_outward_normals` is true. Run the focused
example unittest and expect failure before changing the scene.

- [ ] **Step 2: Enable oriented collision in the example**

Pass `use_outward_normals=True` beside `enable_edge_face=False`. Do not change
thickness, stiffness factors, time step, PCG iterations, or friction.
Use `3e5 Pa` ARAP stiffness for the bunny scene.

- [ ] **Step 3: Run focused tests and 300 frames**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_softbody_limx_arap_bunnies_box -v
uv run -m newton.examples softbody_limx_arap_bunnies_box \
  --device cuda:0 --viewer null --test --num-frames 300
```

Require finite state, positive tetrahedron determinants, no contact overflow,
and a recorded cross-bunny VF/EE contact.

- [ ] **Step 4: Update changelog and run repository checks**

Add an `[Unreleased]` entry describing opt-in oriented signed volume-surface
VF/EE. Run `uvx pre-commit run -a` and `git diff --check`.

- [ ] **Step 5: Commit and launch visual inspection**

Commit with imperative subject `Add oriented LIMX volume collision`, then run:

```bash
uv run -m newton.examples softbody_limx_arap_bunnies_box \
  --device cuda:0 --viewer gl --num-frames 3000 --render-fps 30
```
