# LIMX Style3D-Local EE Thickness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the eight-bunny LIMX scene with uniform 3 mm VF/EE thickness except for Style3D's topology-local EE length cap.

**Architecture:** Reuse `ConstraintSelfCollision`'s default uniform-thickness path and its existing `_is_topology_local_edge_pair` clamp. Disable only the bunny example's opt-in `geometry_radius_scale`, leaving the API and all other physics settings intact.

**Tech Stack:** Python 3, Warp, Newton LIMX, `unittest`, CUDA.

## Global Constraints

- Keep nominal thickness at `0.003 m`.
- Keep VF strict-incident filtering and EE shared-endpoint filtering unchanged.
- Use `min(thickness, (length_0 + length_1) / 4)` only for topology-local EE.
- Do not add damping, substeps, CCD, EF recovery, or new dependencies.
- Do not remove or change the public `geometry_radius_scale` option.

---

### Task 1: Select the Style3D Pair Policy in the Bunny Scene

**Files:**
- Modify: `newton/tests/test_example_softbody_limx_arap_bunnies_box.py`
- Modify: `newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py`

**Interfaces:**
- Consumes: `ConstraintSelfCollision(..., geometry_radius_scale=None)` and the existing topology-local EE clamp.
- Produces: an eight-bunny example using uniform VF/nonlocal-EE thickness and locally capped one-ring EE thickness.

- [ ] **Step 1: Write the failing scene test**

Rename the configuration test to describe Style3D-local EE thickness and
assert `example.self_collision.geometry_radius_scale is None` while preserving
the existing assertions for thickness, stiffness, friction, solver iterations,
and surface-only collision primitives.

- [ ] **Step 2: Verify the test fails for the intended reason**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_softbody_limx_arap_bunnies_box.py -k configuration
```

Expected: fail because the example still sets `geometry_radius_scale=0.25`.

- [ ] **Step 3: Switch only the example configuration**

Set `geometry_radius_scale=None` and update the example header to state that
ordinary VF/EE uses uniform 3 mm thickness while topology-local EE is capped
by the current edge lengths.

- [ ] **Step 4: Verify contact-policy and scene tests**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k "adjacent_opposite_edges_limit_contact_thickness or nonlocal_edge_pair_uses_rest_length_mollifier_threshold or topology_local_edge_pair_uses_half_length_penalty_and_mollifier"
uv run --extra dev -m newton.tests -p test_example_softbody_limx_arap_bunnies_box.py
```

Expected: all selected tests pass, including the 300-frame CUDA rollout.

- [ ] **Step 5: Format, inspect, and launch**

Run pre-commit on the modified files, run `git diff --check`, inspect the
complete diff, and launch:

```bash
uv run -m newton.examples softbody_limx_arap_bunnies_box
```
