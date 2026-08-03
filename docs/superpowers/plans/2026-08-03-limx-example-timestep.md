# LIMX Example Time-Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and render the LIMX cloth example one-to-one at a 0.01 s time step using one Newton iteration and up to 50 PCG iterations per step.

**Architecture:** Keep the existing example loop and solver interfaces unchanged. Change only the example's timing and iteration configuration, then cover the values through a real `Example` instance backed by `ViewerNull` and verify the resulting dynamics on CPU and CUDA.

**Tech Stack:** Python, Warp, Newton examples, `unittest`.

## Global Constraints

- Physics time step and rendered-frame simulation interval are both exactly 0.01 s.
- Render after every physics step; do not decimate frames.
- Use exactly one Newton iteration and at most 50 PCG iterations per physics step.
- Do not change masses, spring stiffnesses, anchors, gravity, or damping behavior.
- Use `unittest`, not pytest.

---

### Task 1: Configure and verify the LIMX visualization

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/examples/cloth/example_cloth_limx.py`

**Interfaces:**
- Consumes: `newton.examples.cloth.example_cloth_limx.Example`, `newton.viewer.ViewerNull`, and the existing public solver fields `nonlinear_iterations` and `linear_iterations`.
- Produces: an example where `frame_dt == sim_dt == 0.01`, `sim_substeps == 1`, `nonlinear_iterations == 1`, and `linear_iterations == 50`.

- [ ] **Step 1: Write the failing configuration test**

Add public imports and this test to `TestSolverLIMX` in `newton/tests/test_solver_limx.py`:

```python
from newton.examples.cloth.example_cloth_limx import Example as ClothLimxExample
from newton.viewer import ViewerNull

def test_example_uses_one_to_one_001_timestep(self):
    with wp.ScopedDevice("cpu"):
        example = ClothLimxExample(ViewerNull(num_frames=1), None)

    self.assertEqual(example.fps, 100)
    self.assertAlmostEqual(example.frame_dt, 0.01)
    self.assertEqual(example.sim_substeps, 1)
    self.assertAlmostEqual(example.sim_dt, 0.01)
    self.assertEqual(example.solver.nonlinear_iterations, 1)
    self.assertEqual(example.solver.linear_iterations, 50)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_example_uses_one_to_one_001_timestep
```

Expected: FAIL because the current example uses `fps == 60`, `sim_substeps == 4`, `sim_dt == 1/240`, four Newton iterations, and 32 PCG iterations.

- [ ] **Step 3: Apply the minimal example configuration**

Change the relevant constructor assignments in `newton/examples/cloth/example_cloth_limx.py` to:

```python
self.fps = 100
self.frame_dt = 1.0 / self.fps
self.sim_substeps = 1
self.sim_dt = self.frame_dt / self.sim_substeps
```

Construct `SolverLIMX` with:

```python
nonlinear_iterations=1,
linear_iterations=50,
```

- [ ] **Step 4: Run the focused test and LIMX regression suite**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_example_uses_one_to_one_001_timestep
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k limx
```

Expected: both commands PASS.

- [ ] **Step 5: Verify one simulated second on CPU and CUDA**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 100 --device cpu
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer null --test --num-frames 100 --device cuda:0
```

Expected: both commands exit successfully and `test_final()` confirms finite state, anchored corners, sagging and swinging cloth, and bounded spring stretch.

- [ ] **Step 6: Run formatting checks and commit**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m pre_commit run --files newton/tests/test_solver_limx.py newton/examples/cloth/example_cloth_limx.py
git diff --check
```

Then commit only the test and example:

```bash
git add newton/tests/test_solver_limx.py newton/examples/cloth/example_cloth_limx.py
git commit -m "Tune LIMX Newton example timestep"
```

- [ ] **Step 7: Launch the interactive visualization**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.examples cloth_limx --viewer gl --device cuda:0
```

Keep the process running so the user can inspect the one-physics-step-per-rendered-frame motion.
