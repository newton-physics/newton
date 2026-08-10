# LIMX Automatic Collision Thickness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate LIMX's nominal VF/EE thickness as `min(0.8 * two_ring_upper_bound, 0.005 m)` when the caller passes `thickness=None`.

**Architecture:** Add a host-side rest-geometry estimator beside the existing geometry-radius helper. Enumerate exact graph-two-ring VF/EE pairs through surface adjacency, evaluate the same interior feature regions used by the detector, and resolve automatic thickness before contact radii and BVHs are allocated.

**Tech Stack:** Python 3, NumPy, Warp, Newton LIMX, `unittest`, CUDA.

## Global Constraints

- Use `eta = 0.8` and maximum thickness `0.005 m`.
- Do not add a motion-derived lower bound.
- Do not change one-ring classification, forces, Hessians, damping, CCD, or time stepping.
- Preserve every explicit positive thickness unchanged.
- Use only NumPy, Warp, and the standard library.

---

### Task 1: Estimate the Two-Ring Rest-Geometry Bound

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`

**Interfaces:**
- Produces: `_compute_two_ring_collision_upper_bound(rest_positions, triangle_indices, edge_indices) -> float`.
- Produces: `ConstraintSelfCollision(..., thickness=None)` automatic mode.

- [ ] **Step 1: Write failing estimator and constructor tests**

Add tests with literal synthetic geometry that verify an interior two-ring VF
distance, the formula `0.8 * upper_bound`, the 5 mm cap when the bound is
larger, rejection of zero-distance automatic geometry, and preservation of an
explicit thickness.

- [ ] **Step 2: Verify the tests fail**

Run each new test with an exact `-k` substring. Expected: fail because `None`
is rejected and the estimator does not exist.

- [ ] **Step 3: Implement the host estimator**

Build vertex-to-vertex, vertex-to-triangle, and vertex-to-edge adjacency.
Enumerate candidates whose minimum vertex graph distance is exactly two.
Evaluate interior point-triangle and segment-segment distances, ignore no
valid candidate (including zero distance), and return the minimum finite
distance or infinity.

- [ ] **Step 4: Resolve automatic thickness in the constructor**

When `thickness is None`, set it to
`min(0.8 * upper_bound, 0.005)`. Use `0.005` when the bound is infinite and
raise `ValueError` when the result is not positive. Keep explicit validation
and behavior unchanged.

- [ ] **Step 5: Run focused solver tests**

Run the new automatic-thickness tests and existing topology-local geometry
radius tests. Expected: all selected CUDA tests pass.

### Task 2: Enable Automatic Bunny Thickness

**Files:**
- Modify: `newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py`
- Modify: `newton/tests/test_example_softbody_limx_arap_bunnies_box.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `ConstraintSelfCollision(..., thickness=None)`.
- Produces: an eight-bunny example whose self-collision thickness is derived from rest geometry.

- [ ] **Step 1: Write a failing scene assertion**

Assert that the bunny example enables automatic mode and resolves to the
literal expected thickness from its checked-in mesh geometry, bounded above by
`0.005 m`.

- [ ] **Step 2: Verify the scene test fails**

Run the configuration test. Expected: fail because the scene still passes
`0.003` explicitly.

- [ ] **Step 3: Enable automatic mode and document it**

Pass `thickness=None`, expose whether the constructor estimated the value, and
update the example header and `[Unreleased]` changelog entry.

- [ ] **Step 4: Verify the scene**

Run the focused configuration test and the 300-frame CUDA example test.
Expected: both pass without contact overflow or tetrahedron inversion.

- [ ] **Step 5: Format, inspect, and launch**

Run pre-commit on modified files, run `git diff --check`, inspect the complete
diff, and launch the long-running bunny visualization.
