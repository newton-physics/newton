# LIMX Example Time-Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and render the LIMX cloth example one-to-one at a 0.01 s time step using one Newton iteration and up to 50 PCG iterations per step.

**Architecture:** Keep the existing example loop and solver interfaces unchanged. Change only the example's timing and iteration configuration, then exercise a real `Example` instance backed by `ViewerNull` to verify that one rendered-frame step performs exactly one 0.01 s solver call. Verify the resulting dynamics on CPU and CUDA.

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
- Produces: an example where one displayed-frame step advances simulation time by 0.01 s through exactly one 0.01 s solver call, using one Newton iteration and up to 50 PCG iterations.

- [ ] **Step 1: Write the failing configuration test**

Add public imports and this test to `TestSolverLIMX` in `newton/tests/test_solver_limx.py`:

```python
from newton.examples.cloth.example_cloth_limx import Example as ClothLimxExample
from newton.viewer import ViewerNull

def test_example_advances_one_001_second_physics_step_per_frame(self):
    with wp.ScopedDevice("cpu"):
        example = ClothLimxExample(ViewerNull(num_frames=1), None)
        solver_step = example.solver.step
        solver_time_steps = []

        def record_solver_step(state_in, state_out, control, contacts, dt):
            solver_time_steps.append(dt)
            solver_step(state_in, state_out, control, contacts, dt)

        example.solver.step = record_solver_step
        example.step()

    self.assertEqual(solver_time_steps, [0.01])
    self.assertAlmostEqual(example.sim_time, 0.01)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_example_advances_one_001_second_physics_step_per_frame
```

Expected: FAIL because the current example makes four solver calls with `dt == 1/240` and advances the displayed-frame simulation time by `1/60`.

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
/home/limx/apps/isaacsim-6.0.1/python.sh -m newton.tests -k test_example_advances_one_001_second_physics_step_per_frame
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
