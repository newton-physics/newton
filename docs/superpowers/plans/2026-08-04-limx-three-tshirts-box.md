# LIMX Three T-Shirts in an Open Box Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and launch a CUDA LIMX example that throws three mutually colliding T-shirt meshes into an open five-plane box.

**Architecture:** Concatenate three disconnected garment copies into one Newton particle model and run one global matrix-free self-collision operator over the combined topology. Compose that operator with five existing inward-facing static-plane contacts; render matching thin static boxes without adding rigid-body degrees of freedom or a new collider type.

**Tech Stack:** Python 3.12, Warp CUDA kernels and graph capture, Newton public model/solver/example APIs, NumPy, OpenUSD, `unittest`.

## Global Constraints

- Run simulation validation on `cuda:0`; do not run CPU physics.
- Keep `dt=0.01 s`, one Newton iteration, 50 PCG iterations, and velocity damping `1.0`.
- Keep membrane stiffness `(1e4, 1e4, 1e3)`, bending stiffness `1e-4`, and collision thickness `0.006 m`.
- Keep adaptive self-collision factors `(0.5, 0.1, 1.5)` and capacity `393216` per contact type.
- Keep box contact stiffness `2e4 N/m`, normal damping `0.5 N*s/m`, friction `0.4`, and friction epsilon `1e-4 m`.
- Use only public `newton` APIs in the example; do not import `newton._src`.
- Do not add PP/PE degeneration, delayed spawning, CCD, line search, or rigid bodies.
- Preserve unrelated dirty files and stage only files owned by each task.

---

### Task 1: Three-garment box example and CUDA regression

**Files:**
- Create: `newton/examples/cloth/example_cloth_limx_three_tshirts_box.py`
- Create: `newton/tests/test_example_cloth_limx_three_tshirts_box.py`

**Interfaces:**
- Consumes: `newton.examples.get_asset("unisex_shirt.usd")`, `newton.solvers.ConstraintTriangleElastic`, `ConstraintDihedralBending`, `ConstraintSelfCollision`, `ConstraintStaticPlaneContact`, `ConstraintGroupDynamic`, and `SolverLIMX`.
- Produces: `Example(viewer, args)`, attributes `garment_count`, `garment_vertex_count`, `garment_triangle_count`, `box_contacts`, `self_collision`, `box_min`, `box_max`, `box_floor`, and `box_wall_top`, plus `test_post_step()` and `test_final()`.

- [ ] **Step 1: Write the failing CUDA test**

Create a CUDA-gated `unittest.TestCase`. Import the new module inside the test and convert `ModuleNotFoundError` into an explicit assertion failure. Instantiate `Example(ViewerNull(num_frames=1), None)`, take one captured step, and assert:

```python
self.assertEqual(
    newton.examples.get_examples()["cloth_limx_three_tshirts_box"],
    "newton.examples.cloth.example_cloth_limx_three_tshirts_box",
)
self.assertEqual(example.garment_count, 3)
self.assertEqual(example.model.particle_count, 3 * example.garment_vertex_count)
self.assertEqual(example.model.tri_count, 3 * example.garment_triangle_count)
self.assertEqual(len(example.box_contacts), 5)
self.assertEqual(example.self_collision.max_contacts, 393216)
self.assertEqual(example.solver.nonlinear_iterations, 1)
self.assertEqual(example.solver.linear_iterations, 50)
self.assertEqual(example.solver.velocity_damping, 1.0)
self.assertTrue(np.isfinite(example.state_0.particle_q.numpy()).all())
self.assertTrue(np.isfinite(example.state_0.particle_qd.numpy()).all())
```

Add a second test that advances 300 frames, calls `test_post_step()` after every step, and asserts all three self-collision overflow counters remain zero.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_three_tshirts_box.py
```

Expected: FAIL with the explicit message that `newton.examples.cloth.example_cloth_limx_three_tshirts_box` is missing.

- [ ] **Step 3: Implement the combined garment model**

Load `/root/shirt`, scale vertices by `0.01`, recenter them, and compute one copy's area-weighted masses with areal density `0.3 kg/m^2`. For each of three fixed configurations, transform the local vertices and create a velocity field:

```python
velocity = linear_velocity + np.cross(
    np.broadcast_to(angular_velocity, transformed_positions.shape),
    transformed_positions - center,
)
```

Concatenate positions, velocities, masses, and offset triangle indices. Build dihedral rows separately per copy with `MeshAdjacency(copy_triangles).edge_indices`, then offset and concatenate the four-particle rows.

- [ ] **Step 4: Implement the open box and solver**

Use an interior box `x in [-0.60, 0.60]`, `y in [-0.50, 0.50]`, floor `z=0.45`, and wall top `z=1.15`. Add one floor and four wall render boxes. Create these analytic planes:

```python
((0, 0, 1), 0.45)
((1, 0, 0), -0.60)
((-1, 0, 0), -0.60)
((0, 1, 0), -0.50)
((0, -1, 0), -0.50)
```

For each plane instantiate `ConstraintStaticPlaneContact` with the global parameters. Compose `[self_collision, *box_contacts]`, construct `SolverLIMX`, capture CUDA execution, and implement the standard `step()`/`render()` methods.

`test_post_step()` must reject non-finite state, `z < box_floor - 0.04`, or a particle below the wall top that exceeds an interior wall by more than `0.04 m`. `test_final()` must call `test_post_step()` and require positive simulation time.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_three_tshirts_box.py
```

Expected: two CUDA tests PASS, with no overflow and no containment assertion.

- [ ] **Step 6: Format and commit the example slice**

Run:

```bash
uvx ruff format newton/examples/cloth/example_cloth_limx_three_tshirts_box.py newton/tests/test_example_cloth_limx_three_tshirts_box.py
uvx ruff check newton/examples/cloth/example_cloth_limx_three_tshirts_box.py newton/tests/test_example_cloth_limx_three_tshirts_box.py
```

Stage only the two task files and commit with subject `Add LIMX three-shirt box scene`.

---

### Task 2: Example documentation and visual asset

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Create: `docs/images/examples/example_cloth_limx_three_tshirts_box.jpg`

**Interfaces:**
- Consumes: launcher-discovered short name `cloth_limx_three_tshirts_box`.
- Produces: one README gallery cell, a `320 x 320` JPEG, and one Unreleased/Added changelog entry.

- [ ] **Step 1: Register documentation**

Insert the example into the cloth gallery using command:

```text
python -m newton.examples cloth_limx_three_tshirts_box
```

Add an Unreleased/Added changelog entry in imperative present tense: `Add a CUDA LIMX stress-test example that throws three mutually colliding T-shirts into an open box.`

- [ ] **Step 2: Capture the example image**

Use a headless `newton.viewer.ViewerGL`, construct the example on `cuda:0`, advance and render until the garments enter the box, then call `viewer.get_frame().numpy()`. Center-crop the RGB array to a square, resize it with Pillow's Lanczos filter, and save a quality-90 `320 x 320` JPEG at the exact path above. Verify dimensions with:

```bash
file docs/images/examples/example_cloth_limx_three_tshirts_box.jpg
```

- [ ] **Step 3: Verify docs and commit**

Run the focused CUDA test again, run `git diff --check`, stage only the README, changelog, image, and updated test, then commit with subject `Document LIMX three-shirt box scene`.

---

### Task 3: Visual handoff and focused regression

**Files:**
- Verify only; no planned source edits.

**Interfaces:**
- Consumes: `python -m newton.examples cloth_limx_three_tshirts_box`.
- Produces: a running CUDA OpenGL window for user inspection and a concise report of any known non-settling behavior.

- [ ] **Step 1: Run focused verification**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_three_tshirts_box.py
uv run --extra dev -m newton.tests -p test_solver_limx.py -k SelfCollision
uvx ruff check newton/examples/cloth/example_cloth_limx_three_tshirts_box.py newton/tests/test_example_cloth_limx_three_tshirts_box.py
git diff --check
```

Expected: the new example tests and 19 self-collision tests PASS, Ruff reports no errors, and `git diff --check` emits no output.

- [ ] **Step 2: Launch the inspection window**

Run:

```bash
uv run --extra examples -m newton.examples cloth_limx_three_tshirts_box --device cuda:0 --num-frames 1000000 --render-fps 100
```

Confirm the process remains alive, CUDA kernels load without errors, and the viewer emits no missing-GUI warning. Leave the window running for user inspection.
