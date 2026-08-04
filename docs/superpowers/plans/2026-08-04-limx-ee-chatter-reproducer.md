# LIMX EE Contact Chatter Reproducer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic CUDA example that renders a no-self-collision sleeve patch beside an identical patch exhibiting persistent VF/EE contact churn.

**Architecture:** Store the diagnosed 74-vertex patch directly in the example module, then construct two independent LIMX models and solvers from it. Advance both solvers in one CUDA graph and render their state meshes manually so the control requires no collision mask or solver API change.

**Tech Stack:** Python 3.12, NumPy, Warp CUDA and graph capture, Newton public model/solver/viewer APIs, `unittest`.

## Global Constraints

- Run all cloth simulation validation on `cuda:0`; do not run CPU physics.
- Keep `dt=0.01 s`, one Newton iteration, 50 PCG iterations, and velocity damping `1.0`.
- Keep membrane stiffness `(1e4, 1e4, 1e3)`, bending stiffness `1e-5`, self-collision thickness `0.006 m`, and adaptive factors `(0.5, 0.1, 1.5)`.
- Use a `0.003 m` particle display radius and contact capacity `4096`.
- Start both patches at zero velocity and zero gravity; fix the same 34 boundary vertices with stiffness `1e6 N/m`.
- Use only public `newton` APIs in the example; do not import `newton._src`.
- Preserve unrelated dirty files and stage only files owned by each task.
- Do not change `ConstraintSelfCollision`, `SolverLIMX`, or any public API.
- Do not add PE/PP contacts, damping, a plane, rigid bodies, CCD, line search, or extra Newton iterations.

---

### Task 1: CUDA characterization test

**Files:**
- Create: `newton/tests/test_example_cloth_limx_ee_chatter.py`
- Create later in Task 2: `newton/examples/cloth/example_cloth_limx_ee_chatter.py`

**Interfaces:**
- Consumes: launcher-discovered module `newton.examples.cloth.example_cloth_limx_ee_chatter`.
- Produces: a CUDA-gated characterization test for `Example.control_patch`, `Example.collision_patch`, and their documented counts.

- [ ] **Step 1: Write the failing discovery and configuration test**

Create a `unittest.TestCase` decorated with `@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")`. Give every test method a triple-double-quoted imperative docstring. Import the module inside the test through this helper so the missing implementation is reported intentionally:

```python
def _load_example_module(test_case: unittest.TestCase):
    module_name = "newton.examples.cloth.example_cloth_limx_ee_chatter"
    if "cloth_limx_ee_chatter" not in newton.examples.get_examples():
        test_case.fail(f"Missing example module {module_name}")
    return importlib.import_module(module_name)
```

Instantiate `Example(ViewerNull(num_frames=1), None)` on `cuda:0`, step once, call `test_post_step()` and `test_final()`, then assert:

```python
self.assertEqual(example.patch_vertex_count, 74)
self.assertEqual(example.patch_triangle_count, 112)
self.assertEqual(example.boundary_vertex_count, 34)
self.assertEqual(example.control_patch.model.particle_count, 74)
self.assertEqual(example.collision_patch.model.particle_count, 74)
self.assertIsNone(example.control_patch.self_collision)
self.assertIsNotNone(example.collision_patch.self_collision)
self.assertEqual(example.collision_patch.self_collision.max_contacts, 4096)
self.assertEqual(example.control_patch.solver.nonlinear_iterations, 1)
self.assertEqual(example.collision_patch.solver.linear_iterations, 50)
self.assertEqual(example.control_patch.solver.velocity_damping, 1.0)
```

Assert both position and velocity arrays are finite.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_ee_chatter.py
```

Expected: FAIL with `Missing example module newton.examples.cloth.example_cloth_limx_ee_chatter` because the example file does not exist.

- [ ] **Step 3: Write the failing late-time characterization test**

Add a second test that runs 1400 CUDA graph frames, samples frames 1000 through 1399, and computes per-side interior velocity RMS. Track the right patch's EE contact IDs as Python tuple sets after copying only the stored contact rows. Count births and deaths between consecutive sampled frames. Track the maximum VF, EE, and EF overflow counts.

Assert the intended characterization:

```python
self.assertLess(control_rms_mean, 1.0e-6)
self.assertGreater(collision_rms_mean, 1.0e-3)
self.assertGreater(ee_births, 100)
self.assertGreater(ee_deaths, 100)
np.testing.assert_array_equal(max_overflow, np.zeros(3, dtype=np.int32))
```

Include the measured RMS and churn counts in assertion messages.

---

### Task 2: Self-contained side-by-side example

**Files:**
- Create: `newton/examples/cloth/example_cloth_limx_ee_chatter.py`
- Test: `newton/tests/test_example_cloth_limx_ee_chatter.py`

**Interfaces:**
- Consumes: public `newton.ModelBuilder`, LIMX constraint classes, `SolverLIMX`, `newton.utils.MeshAdjacency`, and viewer `log_mesh()`.
- Produces: `Example(viewer, args)`, `control_patch`, `collision_patch`, `patch_vertex_count`, `patch_triangle_count`, `boundary_vertex_count`, `step()`, `render()`, `test_post_step()`, and `test_final()`.

- [ ] **Step 1: Add immutable patch data**

Generate the literal arrays once from the diagnosed scene on `cuda:0`. Capture
`full.model.particle_q` before stepping as the rest state, advance exactly 800
frames, and capture `full.state_0.particle_q` as the problem state. Use garment
offset `full.garment_vertex_count`, local seed indices
`np.array([6462, 7650, 7651, 7652]) - offset`, and four iterations of this
face-ring expansion:

```python
selected = set(map(int, seed_indices))
for _ring in range(4):
    selected_array = np.fromiter(selected, dtype=np.int32)
    face_mask = np.any(np.isin(local_triangles, selected_array), axis=1)
    selected.update(map(int, local_triangles[face_mask].ravel()))
```

Sort selected source vertices, keep every source triangle whose three vertices
are selected, remap those triangles to patch-local indices, and derive boundary
indices from rows where `MeshAdjacency(patch_triangles).edge_indices[:, 1] < 0`.
Print each named result with
`np.array2string(value, separator=", ", threshold=np.inf)` and paste the emitted values into module-level `_REST_POSITIONS`,
`_INITIAL_POSITIONS`, `_TRIANGLE_INDICES`, `_MASSES`, and
`_BOUNDARY_INDICES` NumPy arrays.

Assert module-local shapes `(74, 3)`, `(74, 3)`, `(112, 3)`, `(74,)`, and
`(34,)` while constructing the helper so corrupt stored data fails with a clear
`ValueError`.

- [ ] **Step 2: Implement one private patch simulation**

Define `_PatchSimulation(rest_positions, initial_positions, triangle_indices, masses, boundary_indices, translation, enable_self_collision, device)`.

Apply `translation` to rest positions, initial positions, and anchor targets. Build a 74-particle model with display radius `0.003`, zero velocity, and zero gravity. Derive interior dihedral rows from `MeshAdjacency(triangle_indices).edge_indices`. Add these static constraints:

```python
ConstraintTriangleElastic(
    triangle_indices=triangle_indices,
    inverse_rest_matrices=model.tri_poses.numpy(),
    rest_areas=model.tri_areas.numpy(),
    stiffnesses=[wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(triangle_indices),
    particle_count=len(rest_positions),
    device=model.device,
)
ConstraintDihedralBending(
    dihedral_indices=dihedral_indices,
    rest_positions=translated_rest_positions,
    stiffness=1.0e-5,
    particle_count=len(rest_positions),
    device=model.device,
)
ConstraintAnchor(
    indices=boundary_indices,
    targets=[wp.vec3(*position) for position in translated_initial_positions[boundary_indices]],
    stiffnesses=[1.0e6] * len(boundary_indices),
    particle_count=len(rest_positions),
    device=model.device,
)
```

For the reproduction only, create:

```python
ConstraintSelfCollision(
    model,
    thickness=0.006,
    stiffness=None,
    max_contacts=4096,
    stiffness_factors=(0.5, 0.1, 1.5),
)
```

Construct `SolverLIMX` with the global solver settings. Create two states and assign the translated problem snapshot to `state_0.particle_q`; explicitly zero `state_0.particle_qd`.

- [ ] **Step 3: Implement the public example and CUDA graph**

Create control and collision simulations translated by equal and opposite horizontal offsets. Store an interior-index Warp array for diagnostics. Capture one `simulate()` call that clears both force arrays, advances both solvers, and assigns both output states.

`step()` launches the graph and increments `sim_time`. `test_post_step()` rejects non-finite positions or velocities on either side. `test_final()` calls `test_post_step()` and requires positive simulation time.

- [ ] **Step 4: Render both dynamic meshes**

Upload one flattened Warp triangle-index array per patch helper. In `render()` call `viewer.log_mesh()` with each current position array and the fixed indices:

```python
self.viewer.log_mesh(
    "/control_no_collision",
    self.control_patch.state_0.particle_q,
    self.control_patch.render_indices,
    color=(0.25, 0.55, 0.95),
    backface_culling=False,
)
self.viewer.log_mesh(
    "/vf_ee_collision",
    self.collision_patch.state_0.particle_q,
    self.collision_patch.render_indices,
    color=(0.95, 0.42, 0.16),
    backface_culling=False,
)
```

Set a close default camera, use 100 FPS and one simulation step per frame, and default to 1400 frames in the module entry point.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_ee_chatter.py
```

Expected: both CUDA tests PASS; the control late RMS is below `1e-6`, the reproduction late RMS is above `1e-3`, EE births and deaths each exceed 100, and overflow remains zero.

- [ ] **Step 6: Format and inspect the implementation**

Run:

```bash
uvx ruff format newton/examples/cloth/example_cloth_limx_ee_chatter.py newton/tests/test_example_cloth_limx_ee_chatter.py
uvx ruff check newton/examples/cloth/example_cloth_limx_ee_chatter.py newton/tests/test_example_cloth_limx_ee_chatter.py
git diff --check -- newton/examples/cloth/example_cloth_limx_ee_chatter.py newton/tests/test_example_cloth_limx_ee_chatter.py
```

Stage only the example and its test, then commit with subject `Add LIMX EE chatter reproducer`.

---

### Task 3: User-facing documentation and visual asset

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Create: `docs/images/examples/example_cloth_limx_ee_chatter.jpg`

**Interfaces:**
- Consumes: launcher-discovered command `cloth_limx_ee_chatter`.
- Produces: a cloth-gallery entry, a `320 x 320` JPEG, and one Unreleased/Added changelog item.

- [ ] **Step 1: Register the example in documentation**

Add a cloth gallery cell containing:

```text
python -m newton.examples cloth_limx_ee_chatter
```

Add this entry at a non-terminal position in `CHANGELOG.md`'s Unreleased/Added category:

```text
Add a CUDA LIMX example that contrasts a settled cloth patch with persistent VF/EE contact churn.
```

- [ ] **Step 2: Capture and validate the screenshot**

Run the example with a headless OpenGL viewer on `cuda:0`, advance to a frame where the control and reproduction visibly differ, center-crop the RGB frame, resize with Pillow Lanczos, and write a quality-90 `320 x 320` JPEG to the specified path. Verify its dimensions with:

```bash
file docs/images/examples/example_cloth_limx_ee_chatter.jpg
```

- [ ] **Step 3: Verify the documented example**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_ee_chatter.py
uvx ruff check newton/examples/cloth/example_cloth_limx_ee_chatter.py newton/tests/test_example_cloth_limx_ee_chatter.py
git diff --check -- README.md CHANGELOG.md docs/images/examples/example_cloth_limx_ee_chatter.jpg
```

Stage only the README, changelog, and screenshot, then commit with subject `Document LIMX EE chatter example`.

---

### Task 4: Visual handoff

**Files:**
- Verify only; no planned source edits.

**Interfaces:**
- Consumes: `python -m newton.examples cloth_limx_ee_chatter`.
- Produces: a running CUDA OpenGL window showing the blue control patch and orange VF/EE patch.

- [ ] **Step 1: Run focused final verification**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_ee_chatter.py
uv run --extra dev -m newton.tests -p test_solver_limx.py -k SelfCollision
uvx pre-commit run -a
git diff --check
```

Expected: the characterization tests and self-collision tests pass, pre-commit reports no new failures caused by this slice, and `git diff --check` emits no output.

- [ ] **Step 2: Launch the inspection window**

Run:

```bash
uv run --extra examples -m newton.examples cloth_limx_ee_chatter --device cuda:0 --num-frames 1000000 --render-fps 100
```

Confirm the process remains alive and CUDA kernels load without errors. Leave the viewer window running for user inspection.
