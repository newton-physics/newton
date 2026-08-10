# LIMX Topology-Local Collision Thickness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the eight-bunny LIMX scene with uniform 3 mm nonlocal VF/EE thickness and geometry-capped topology-local VF/EE thickness.

**Architecture:** Add an opt-in topology-local-only scope to the existing geometry radii. Classify VF through surface one-ring adjacency, reuse `_is_topology_local_edge_pair` for EE, and combine local EE radii with Style3D's existing current-edge-length cap; preserve the existing global radius scope by default.

**Tech Stack:** Python 3, Warp, Newton LIMX, `unittest`, CUDA.

## Global Constraints

- Keep nominal thickness at `0.003 m`.
- Keep VF strict-incident filtering and EE shared-endpoint filtering unchanged.
- Use geometry radii only for topology-local VF/EE in the new scope.
- Use `min(thickness, geometry_thickness, (length_0 + length_1) / 4)` for topology-local EE.
- Do not add damping, substeps, CCD, EF recovery, or new dependencies.
- Preserve the existing global behavior of `geometry_radius_scale` unless the
  new topology-local-only scope is explicitly enabled.

---

### Task 1: Add a Topology-Local Geometry-Radius Scope

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: existing per-particle rest-geometry radii and surface adjacency.
- Produces: `geometry_radius_topology_local_only: bool = False` on `ConstraintSelfCollision`.

- [ ] **Step 1: Write failing CUDA VF/EE tests**

Add controlled one-ring and nonlocal VF/EE fixtures with deliberately small
particle radii. Assert local pairs use the small radii while nonlocal pairs use
the nominal thickness. Assert enabling local-only scope without
`geometry_radius_scale` raises `ValueError`.

- [ ] **Step 2: Verify the tests fail**

Run each new test by its exact `-k` substring. Expected: fail because the
constructor does not accept `geometry_radius_topology_local_only`.

- [ ] **Step 3: Implement local classification and thickness selection**

Build a CSR one-ring vertex adjacency from surface edges. Pass it and the new
scope flag to VF detection. Apply geometry interpolation only to local VF/EE
when the flag is enabled, and additionally apply the existing Style3D length
cap to local EE. Preserve global geometry-radius behavior when the flag is
disabled.

- [ ] **Step 4: Run focused collision tests**

Run the new tests plus the existing global geometry-radius, topology-local EE,
and nonlocal EE tests. Expected: all selected tests pass.

### Task 2: Select the Pair-Specific Policy in the Bunny Scene

**Files:**
- Modify: `newton/tests/test_example_softbody_limx_arap_bunnies_box.py`
- Modify: `newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py`

**Interfaces:**
- Consumes: `ConstraintSelfCollision(..., geometry_radius_scale=0.25, geometry_radius_topology_local_only=True)`.
- Produces: an eight-bunny example using uniform VF/nonlocal-EE thickness and locally capped one-ring EE thickness.

- [ ] **Step 1: Write the failing scene test**

Rename the configuration test to describe topology-local collision thickness
and assert the radius scale and local-only scope while preserving
the existing assertions for thickness, stiffness, friction, solver iterations,
and surface-only collision primitives.

- [ ] **Step 2: Verify the test fails for the intended reason**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_softbody_limx_arap_bunnies_box.py -k configuration
```

Expected: fail because the example does not enable the local-only scope.

- [ ] **Step 3: Switch only the example configuration**

Set `geometry_radius_scale=0.25`, enable the local-only scope, and update the
example header to distinguish nonlocal uniform thickness from local geometry
and Style3D edge-length caps.

- [ ] **Step 4: Verify contact-policy and scene tests**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_softbody_limx_arap_bunnies_box.py
```

Expected: all selected tests pass, including the 300-frame CUDA rollout.

- [ ] **Step 5: Format, inspect, and launch**

Run pre-commit on the modified files, run `git diff --check`, inspect the
complete diff, and launch:

```bash
uv run -m newton.examples softbody_limx_arap_bunnies_box
```
