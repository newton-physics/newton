# LIMX Cloth Twist Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and launch a CUDA LIMX cloth-twist scene that visibly exercises VF/EE self-collision and EF untangling.

**Architecture:** A procedural rectangular strip reuses LIMX membrane, dihedral bending, and `ConstraintSelfCollision`. One anchor batch owns both short boundaries; a pure NumPy target function rotates the two boundary cross-sections by equal and opposite angles before every captured CUDA solve.

**Tech Stack:** Python, NumPy, Warp CUDA graphs, Newton LIMX, `unittest`, OpenGL viewer.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-03-limx-cloth-twist-design.md`.
- Use a 32-by-12-cell procedural strip, zero gravity, and no rigid bodies or ground.
- Preserve `dt=0.01`, one Newton iteration, 50 PCG iterations, `velocity_damping=1.0`, and one render per physics step.
- Use LIMX membrane, dihedral bending, and matrix-free `ConstraintSelfCollision` with thickness `0.012 m` and stiffness `1.0e4 N/m`.
- Drive the left and right boundaries to `+theta` and `-theta` around X, ramping to two turns per side.
- Validate and run on `cuda:0`; do not run CPU simulation.
- Preserve the user's uncommitted `lessons.md` and `solver_convergence.png`.

---

### Task 1: Boundary Drive Contract

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Create: `newton/examples/cloth/example_cloth_limx_twist.py`

**Interfaces:**
- Produces: `Example._compute_anchor_targets(angle: float) -> np.ndarray`, returning targets in the same order as `anchor_indices`.

- [ ] **Step 1: Write the failing CUDA example test**

Import `Example as ClothLimxTwistExample`. Construct it with
`ViewerNull(num_frames=1)` under `wp.ScopedDevice("cuda:0")`. For
`angle = pi/2`, assert that every target keeps its rest X coordinate, left
boundary Y maps to positive Z while right boundary Y maps to negative Z,
and both rotated radial lengths equal their rest values.

- [ ] **Step 2: Run the test and verify RED**

```bash
/tmp/newton-main-merge-env.oQinDr/bin/python -m unittest \
  newton.tests.test_solver_limx.TestSolverLIMX.test_twist_example_drives_opposite_boundary_rotations
```

Expected: import failure because `example_cloth_limx_twist.py` does not exist.

- [ ] **Step 3: Implement the procedural topology and target function**

Create a 32-by-12-cell XY strip with alternating triangle diagonals. Store
the left boundary first and right boundary second. Implement

```python
rotated_y = cos(angle) * rest_y
rotated_z = sign * sin(angle) * rest_y
```

with `sign=+1` on the left and `sign=-1` on the right, preserving each
boundary's X coordinate and center height.

- [ ] **Step 4: Run the focused target test and verify GREEN**

Run the Task 1 command and require zero failures.

---

### Task 2: LIMX Solver and CUDA Graph

**Files:**
- Modify: `newton/examples/cloth/example_cloth_limx_twist.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Consumes: `Example._compute_anchor_targets()` from Task 1.
- Produces: runnable `cloth_limx_twist` example with `solver`, `anchor_constraint`, `self_collision`, `state_0`, and `state_1`.

- [ ] **Step 1: Extend the test with failing solver assertions**

Require `ConstraintSelfCollision`, `sim_dt == 0.01`,
`nonlinear_iterations == 1`, `linear_iterations == 50`, and
`velocity_damping == 1.0`. Call `step()` once and require finite positions
and velocities.

- [ ] **Step 2: Run the focused test and verify RED**

Expected: failure until the example constructs the complete LIMX solver and
CUDA graph.

- [ ] **Step 3: Implement constraints, drive, and rendering**

Build `ConstraintAnchor`, `ConstraintTriangleElastic`, and
`ConstraintDihedralBending`; construct `ConstraintSelfCollision` and pass it
as `dynamic_operator`. Before each graph launch, compute

```python
phase = min(sim_time / 4.0, 1.0)
angle = 4.0 * pi * (phase * phase * (3.0 - 2.0 * phase))
```

and assign the resulting targets to `anchor_constraint.targets`. Use standard
Newton viewer setup, camera, graph capture, state swap, render, and finite
`test_final()` checks.

- [ ] **Step 4: Run the focused CUDA test and verify GREEN**

Run the Task 1 command and require zero failures.

- [ ] **Step 5: Commit the example and tests**

```bash
git add newton/examples/cloth/example_cloth_limx_twist.py newton/tests/test_solver_limx.py
git commit -m "Add LIMX cloth twist example"
```

---

### Task 3: Smoke Test and Interactive Launch

**Files:**
- Modify: `README.md`
- Create after visual inspection: `docs/images/examples/example_cloth_limx_twist.jpg`

- [ ] **Step 1: Run the CUDA headless smoke test**

```bash
/tmp/newton-main-merge-env.oQinDr/bin/python -m newton.examples cloth_limx_twist \
  --viewer null --num-frames 10 --device cuda:0
```

Require exit code zero and finite final state.

- [ ] **Step 2: Launch the interactive CUDA viewer**

```bash
/tmp/newton-main-merge-env.oQinDr/bin/python -m newton.examples cloth_limx_twist \
  --device cuda:0
```

Confirm the process remains alive and the ImGui control panel loads without
dependency warnings before handing the window to the user.

- [ ] **Step 3: Register the example after visual approval**

Capture a 320-by-320 JPG once the cloth is visibly twisted, add the README
tile and `python -m newton.examples cloth_limx_twist` command, then run
pre-commit and commit those presentation assets separately.
