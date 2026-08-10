# LIMX Surface-Vertex Collision Radii Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use rest-geometry-capped per-surface-vertex collision radii in the oriented tetrahedral bunny scene.

**Architecture:** Extend the existing geometry-aware radius helper to treat unreferenced particles as non-colliding interior particles with zero radius. Reuse the existing VF barycentric and EE edge interpolation, and enable the existing `geometry_radius_scale=0.25` option in the bunny example.

**Tech Stack:** Python 3, NumPy, Warp, Newton LIMX, `unittest`, CUDA.

## Global Constraints

- Keep the nominal bunny self-collision thickness at `0.003 m` as an upper bound.
- Compute radii once from rest surface geometry with scale `0.25`.
- Keep strict IPC incident filtering: VF excludes only face vertices and EE excludes only shared endpoints.
- Do not add damping, substeps, CCD, EF recovery, or dynamic radius recomputation.
- Use only surface triangle vertices for geometry validation; interior tetrahedral particles receive radius zero.

---

### Task 1: Support Interior Tetrahedral Particles

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`

**Interfaces:**
- Consumes: `_compute_geometry_aware_particle_radii(rest_positions, triangle_indices, nominal_radius, geometry_radius_scale)`.
- Produces: one radius per model particle, with zero for particles absent from `triangle_indices`.

- [ ] **Step 1: Write a failing CUDA test**

Add `test_geometry_aware_radii_ignore_unreferenced_interior_particles`. Build a
four-particle model whose only triangle is `(0, 1, 2)`, construct
`ConstraintSelfCollision(..., geometry_radius_scale=0.25)`, and assert radii
`[0.05, 0.05, 0.05, 0.0]` for nominal thickness `0.1`.

- [ ] **Step 2: Verify the test fails**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k ignore_unreferenced_interior_particles
```

Expected: fail with `geometry-aware collision requires every particle to be referenced by a triangle`.

- [ ] **Step 3: Implement surface-only radius validation**

Validate finiteness through `rest_positions[triangle_indices]`, initialize all
radii to zero, reduce minimum triangle altitude onto referenced indices, and
assign `min(nominal_radius, geometry_radius_scale * local_scale)` only where
the local scale is finite. Preserve the existing degenerate-triangle checks.

- [ ] **Step 4: Run focused geometry-radius tests**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k geometry_aware_radii
```

Expected: all selected tests pass.

### Task 2: Enable Bunny Surface Radii

**Files:**
- Modify: `newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py`
- Modify: `newton/tests/test_example_softbody_limx_arap_bunnies_box.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `ConstraintSelfCollision(..., geometry_radius_scale=0.25)`.
- Produces: a fixed-3-mm-upper-bound oriented bunny scene with geometry-aware surface radii.

- [ ] **Step 1: Write failing scene assertions**

Expect `geometry_radius_scale == 0.25`. Call `prepare(model.particle_q)` and
assert zero initial VF and EE contacts.

- [ ] **Step 2: Verify the scene test fails**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_softbody_limx_arap_bunnies_box.py -k configuration
```

Expected: fail because the example still sets `geometry_radius_scale=None`.

- [ ] **Step 3: Enable geometry-aware radii**

Pass `geometry_radius_scale=0.25` in the bunny self-collision constructor and
update the example description and changelog to call `3 mm` the nominal upper
bound.

- [ ] **Step 4: Run detection and scene regressions**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k TestConstraintSelfCollisionDetection
uv run --extra dev -m newton.tests -p test_example_softbody_limx_arap_bunnies_box.py
```

Expected: 34 detection tests and both bunny tests pass, including the 300-frame CUDA rollout.

- [ ] **Step 5: Run formatting and inspect the diff**

Run `uvx pre-commit run --files` for all modified source, test, documentation,
and changelog files, then run `git diff --check` and inspect `git diff`.
