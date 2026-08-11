# LIMX EE Endpoint Coverage Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether every endpoint EE contact on unsigned triangle surfaces is already covered by VF, without changing production collision behavior.

**Architecture:** Add a test-only NumPy analyzer that reads existing LIMX VF/EE buffers and mesh adjacency, classifies endpoint EE candidates, and matches them to VF contacts on incident triangles. Verify it on fixed CUDA geometries, then use a standalone diagnostic runner to aggregate bounded twist and T-shirt rollouts.

**Tech Stack:** Python, NumPy, Warp CUDA, `unittest`, existing `ConstraintSelfCollision` buffers.

## Global Constraints

- Do not modify `ConstraintSelfCollision` or any production collision response.
- Limit the coverage claim to `use_outward_normals=False` triangle surfaces.
- Do not add dependencies.
- Run only focused CUDA tests and bounded diagnostics.
- Preserve the existing dirty screenshot, settling-test, and `solver_convergence.png` files.

---

### Task 1: Endpoint EE coverage analyzer

**Files:**
- Create: `newton/tests/utils/limx_contact_coverage.py`
- Test: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Consumes: `ConstraintSelfCollision.triangle_indices`, `edge_indices`, `edge_triangle_indices`, `vertex_face_contacts`, and `edge_edge_contacts` after `prepare()`.
- Produces: `analyze_endpoint_ee_coverage(collision) -> EndpointEECoverage`, where the dataclass stores `strict_ee`, `endpoint_pe`, `endpoint_pp`, `covered`, `unmatched`, and `first_unmatched`.

- [ ] **Step 1: Write failing CUDA tests**

Add tests for endpoint-interior PE, endpoint-endpoint PP, shared-edge duplication, and a strict interior EE control. Each unsigned case must assert `unmatched == 0`; the strict control must increment only `strict_ee`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k endpoint_ee_coverage
```

Expected: import failure because `limx_contact_coverage.py` does not exist.

- [ ] **Step 3: Implement minimal buffer analysis**

Read stored contacts up to `min(count, capacity)`. Recover EE parameters from weights, classify parameters equal to `0` or `1` within `1e-6`, map each target edge to its incident triangles, and mark an endpoint EE covered when the endpoint appears as the VF query vertex against an incident triangle. For PP accept either endpoint-to-opposite-incident-triangle direction. Preserve the first unmatched IDs, parameters, and weights.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: all selected tests pass on CUDA.

- [ ] **Step 5: Commit the analyzer and tests**

```bash
git add newton/tests/utils/limx_contact_coverage.py newton/tests/test_solver_limx.py
git commit -m "Measure EE endpoint VF coverage"
```

### Task 2: Bounded scene diagnostic

**Files:**
- Create: `tools/limx_ee_endpoint_coverage.py`

**Interfaces:**
- Consumes: `analyze_endpoint_ee_coverage(collision)` from Task 1 and the public example loader.
- Produces: one textual summary per scene containing frame count, strict EE, endpoint PE, endpoint PP, covered, unmatched, overflows, and first unmatched details.

- [ ] **Step 1: Add a CLI smoke test path**

Implement `--scene {twist,three_tshirts_box}`, `--frames`, and `--device`. Reject nonpositive frame counts. The runner constructs `ViewerNull`, advances the example, calls the analyzer after each prepared step, and accumulates counts without modifying solver state.

- [ ] **Step 2: Run a one-frame smoke diagnostic**

```bash
uv run --extra dev tools/limx_ee_endpoint_coverage.py --scene twist --frames 1 --device cuda:0
```

Expected: finite integer counts, zero overflow, and a process exit code of zero.

- [ ] **Step 3: Run bounded scene experiments**

```bash
uv run --extra dev tools/limx_ee_endpoint_coverage.py --scene twist --frames 100 --device cuda:0
uv run --extra dev tools/limx_ee_endpoint_coverage.py --scene three_tshirts_box --frames 100 --device cuda:0
```

Expected hypothesis result: `unmatched=0` for both scenes. Any nonzero result is retained as the counterexample and the production EE path remains unchanged.

- [ ] **Step 4: Verify formatting and focused regression**

```bash
uvx pre-commit run --files newton/tests/utils/limx_contact_coverage.py newton/tests/test_solver_limx.py tools/limx_ee_endpoint_coverage.py
uv run --extra dev -m newton.tests -p test_solver_limx.py -k endpoint_ee_coverage
```

Expected: all checks pass.

- [ ] **Step 5: Commit the diagnostic runner**

```bash
git add tools/limx_ee_endpoint_coverage.py
git commit -m "Add EE endpoint coverage diagnostic"
```
