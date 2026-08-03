# LIMX Cloth Iteration Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two-corner LIMX cloth example visibly swing past its anchor line within one simulated second by increasing nonlinear convergence without changing the solver architecture or time integrator.

**Architecture:** Keep `SolverLIMX`, backward Euler, fixed 3×3 block-CSR constraint Hessians, and PCG unchanged. Change only the example's iteration split from 4 nonlinear × 32 linear iterations to 64 nonlinear × 10 linear iterations, and strengthen `Example.test_final()` with a trajectory-level swing assertion.

**Tech Stack:** Python 3.12, Warp 1.13, Newton `Example`/`unittest` infrastructure, CUDA Graph capture.

## Global Constraints

- Keep the simulation time step at `1/240 s` (`60 Hz` with four substeps).
- Keep `velocity_damping=1.0`; do not add explicit damping.
- Keep static elasticity in the existing 3×3 block-CSR path and dynamic constraints matrix-free.
- Do not change the public `SolverLIMX` constructor defaults in this correction.
- Use Newton's `unittest`-based runners, not pytest.
- Do not stage or modify `lessons.md` as part of the implementation commit.

---

### Task 1: Require and produce visible cloth swing

**Files:**
- Modify: `newton/examples/cloth/example_cloth_limx.py:67-132`

**Interfaces:**
- Consumes: `SolverLIMX(model, constraints, nonlinear_iterations: int, linear_iterations: int)` and `Example.test_final()` from Newton's example test runner.
- Produces: An example configured with `nonlinear_iterations=64` and `linear_iterations=10`; `test_final()` rejects a center particle that has not crossed the anchors' initial Y coordinate after 60 frames.

- [ ] **Step 1: Write the failing behavior assertion**

Store the anchors' initial Y coordinate during example construction:

```python
self.anchor_y = float(self.anchor_targets[0, 1])
```

Extend `test_final()` after the existing center-sag check:

```python
if positions[self.center_index, 1] >= self.anchor_y:
    raise AssertionError("LIMX cloth center did not swing past the anchor line")
```

- [ ] **Step 2: Run the CUDA example to verify RED**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx \
  --viewer null --test --num-frames 60 --device cuda:0
```

Expected: FAIL with `LIMX cloth center did not swing past the anchor line` because the existing `4 x 32` configuration ends near `y=-0.017`, while the anchor line is `y=-0.5`.

- [ ] **Step 3: Apply the minimal iteration-budget change**

Update only the example's solver construction:

```python
self.solver = newton.solvers.SolverLIMX(
    self.model,
    constraints,
    nonlinear_iterations=64,
    linear_iterations=10,
)
```

- [ ] **Step 4: Run the example to verify GREEN**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx \
  --viewer null --test --num-frames 60 --device cuda:0
```

Expected: PASS; the measured full-scene endpoint is approximately `y=-0.775`, past the `y=-0.5` anchor line, with finite nonzero velocity.

- [ ] **Step 5: Run focused regression tests**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_solver_limx
```

Expected: all LIMX tests pass.

- [ ] **Step 6: Verify formatting and commit**

Run:

```bash
git diff --check
```

Then commit only the example:

```bash
git add newton/examples/cloth/example_cloth_limx.py
git commit -m "Increase LIMX cloth PD iterations"
```

- [ ] **Step 7: Launch the interactive example**

Run in a persistent terminal:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --device cuda:0
```

Expected: the viewer stays open and the cloth falls, is caught by its two anchors, and visibly swings through the anchor line.
