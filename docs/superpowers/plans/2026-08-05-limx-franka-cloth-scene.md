# LIMX Franka Cloth Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a meter-scale example in which a Franka FR3 performs an approach, descend, close, and lift sequence over a temporarily fixed square LIMX cloth on a table.

**Architecture:** Build the robot, static table, and cloth particles in one `ModelBuilder`. Reuse Newton GPU IK to generate per-frame Franka joint targets and `SolverFeatherstone` as the kinematic integrator; keep every cloth particle inactive so this milestone validates geometry and motion without cloth-rigid coupling.

**Tech Stack:** Python, NumPy, Warp, Newton public APIs, `newton.ik`, `SolverFeatherstone`, `unittest`, ViewerGL/ViewerNull.

## Global Constraints

- Keep all physical dimensions in SI units.
- Use only public Newton modules; do not import from `newton._src`.
- Keep the 21-by-21, approximately 0.4 m square cloth flat by clearing `ParticleFlags.ACTIVE` on every cloth particle.
- Do not add cloth-table, cloth-hand, or cloth-finger contact in this milestone.
- Use `unittest`, give every test method a triple-double-quoted imperative docstring, and run commands through `uv`.
- Add no dependency; Pillow is already part of the examples extra and is used only to save the required screenshot.
- Preserve unrelated working-tree changes and stage only the files named in each task.

## File Structure

- Create `newton/examples/cloth/example_cloth_limx_franka.py`: grid construction, scene construction, Franka IK trajectory, Featherstone stepping, validation, rendering, and CLI entry point.
- Create `newton/tests/test_example_cloth_limx_franka.py`: focused geometry and CUDA rollout regression tests.
- Modify `newton/tests/test_examples.py`: register the new example in the existing cloth example smoke-test group.
- Modify `README.md`: add the new example card and command.
- Create `docs/images/examples/example_cloth_limx_franka.jpg`: 320-by-320 GL screenshot captured near the closed-gripper or lift phase.
- Modify `CHANGELOG.md`: add the user-facing example under `[Unreleased]` → `Added`.

---

### Task 1: Square Cloth Geometry

**Files:**
- Create: `newton/examples/cloth/example_cloth_limx_franka.py`
- Create: `newton/tests/test_example_cloth_limx_franka.py`

**Interfaces:**
- Produces: `_create_square_cloth_grid(grid_cells: int, width: float, center: tuple[float, float], height: float) -> tuple[np.ndarray, np.ndarray]`.
- Produces: row-major `(grid_cells + 1)^2` float32 positions and `2 * grid_cells^2` int32 triangles with alternating diagonals.

- [ ] **Step 1: Write the failing grid test**

```python
class TestClothLimxFranka(unittest.TestCase):
    def test_square_cloth_grid_has_expected_flat_topology(self):
        """Build a flat square grid with alternating triangle diagonals."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        positions, triangles = module._create_square_cloth_grid(
            grid_cells=20,
            width=0.4,
            center=(0.0, -0.5),
            height=0.205,
        )

        self.assertEqual(positions.shape, (441, 3))
        self.assertEqual(triangles.shape, (800, 3))
        self.assertEqual(positions.dtype, np.float32)
        self.assertEqual(triangles.dtype, np.int32)
        np.testing.assert_allclose(positions[:, 2], 0.205, rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(positions[:, :2].min(axis=0), (-0.2, -0.7), atol=1.0e-7)
        np.testing.assert_allclose(positions[:, :2].max(axis=0), (0.2, -0.3), atol=1.0e-7)
        np.testing.assert_array_equal(triangles[:2], ((0, 1, 22), (0, 22, 21)))
```

- [ ] **Step 2: Run the test and confirm the module is missing**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_cloth_limx_franka.TestClothLimxFranka.test_square_cloth_grid_has_expected_flat_topology
```

Expected: `ModuleNotFoundError` for `example_cloth_limx_franka`.

- [ ] **Step 3: Implement the grid helper and minimal example module header**

```python
def _create_square_cloth_grid(
    grid_cells: int,
    width: float,
    center: tuple[float, float],
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    grid_side = grid_cells + 1
    positions = np.empty((grid_side * grid_side, 3), dtype=np.float32)
    triangles: list[tuple[int, int, int]] = []
    for y in range(grid_side):
        for x in range(grid_side):
            index = y * grid_side + x
            positions[index] = (
                center[0] - 0.5 * width + width * x / grid_cells,
                center[1] - 0.5 * width + width * y / grid_cells,
                height,
            )
    for y in range(grid_cells):
        for x in range(grid_cells):
            lower_left = y * grid_side + x
            lower_right = lower_left + 1
            upper_left = lower_left + grid_side
            upper_right = upper_left + 1
            if (x + y) % 2 == 0:
                triangles.extend(((lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)))
            else:
                triangles.extend(((lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left)))
    return positions, np.asarray(triangles, dtype=np.int32)
```

- [ ] **Step 4: Run the grid test and confirm it passes**

Run the command from Step 2. Expected: one passing test.

- [ ] **Step 5: Commit the geometry slice**

```bash
git add newton/examples/cloth/example_cloth_limx_franka.py \
  newton/tests/test_example_cloth_limx_franka.py
git commit -m "Add LIMX Franka cloth geometry"
```

### Task 2: Franka Scene and Grasp Trajectory

**Files:**
- Modify: `newton/examples/cloth/example_cloth_limx_franka.py`
- Modify: `newton/tests/test_example_cloth_limx_franka.py`

**Interfaces:**
- Consumes: `_create_square_cloth_grid(...)` from Task 1.
- Produces: `Example(viewer, args=None)` with `model`, `state_0`, `cloth_rest_positions`, `cloth_particle_indices`, `table_top_z`, `grasp_position`, `lift_position`, `sequence_duration`, `step()`, `render()`, `test_post_step()`, and `test_final()`.
- Produces: `_find_label_index(labels: list[str], suffix: str) -> int`, `set_gripper_q`, and `compute_joint_qd` for IK-driven articulation motion.

- [ ] **Step 1: Write the failing scene and rollout tests**

```python
@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxFrankaCuda(unittest.TestCase):
    def test_scene_keeps_square_cloth_flat_and_inactive(self):
        """Keep every square-cloth particle fixed above the table."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=1), SimpleNamespace(graph_capture=False))
            flags = example.model.particle_flags.numpy()
            positions = example.state_0.particle_q.numpy()

        self.assertEqual(example.model.particle_count, 441)
        self.assertTrue(np.all((flags & int(newton.ParticleFlags.ACTIVE)) == 0))
        np.testing.assert_allclose(positions, example.cloth_rest_positions, rtol=0.0, atol=1.0e-7)
        self.assertGreater(float(positions[:, 2].min()), example.table_top_z)

    def test_grasp_sequence_reaches_and_lifts_from_cloth(self):
        """Drive the Franka through approach, close, and lift poses."""
        module = importlib.import_module("newton.examples.cloth.example_cloth_limx_franka")
        with wp.ScopedDevice("cuda:0"):
            frame_count = int(np.ceil(module.SEQUENCE_DURATION * module.FPS)) + 1
            example = module.Example(ViewerNull(num_frames=frame_count), SimpleNamespace(graph_capture=False))
            for _ in range(frame_count):
                example.step()
                example.test_post_step()
            example.test_final()

        self.assertLess(example.minimum_grasp_error, 0.03)
        self.assertGreater(example.maximum_tcp_height, example.grasp_position[2] + 0.10)
```

- [ ] **Step 2: Run both CUDA tests and confirm missing `Example`/constants failures**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_cloth_limx_franka.TestClothLimxFrankaCuda
```

Expected: failures because the scene class and trajectory constants are not implemented.

- [ ] **Step 3: Build the single-model table, cloth, and Franka scene**

Use these fixed scene values in the module:

```python
FPS = 60
SIM_SUBSTEPS = 10
TABLE_CENTER = (0.0, -0.5, 0.1)
TABLE_HALF_EXTENTS = (0.4, 0.4, 0.1)
TABLE_TOP_Z = 0.2
CLOTH_CENTER = (0.0, -0.5)
CLOTH_WIDTH = 0.4
CLOTH_GRID_CELLS = 20
CLOTH_HEIGHT = TABLE_TOP_Z + 0.005
CLOTH_PARTICLE_RADIUS = 0.003
FRANKA_BASE = (-0.5, -0.5, -0.1)
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)
```

In `Example.__init__`, add the fixed-base `fr3_franka_hand.urdf`, set the known
9-coordinate seed pose, add the static table box, call `_create_square_cloth_grid`,
add particles and triangles, finalize, then clear active flags:

```python
flags = self.model.particle_flags.numpy()
flags &= ~int(newton.ParticleFlags.ACTIVE)
self.model.particle_flags = wp.array(flags, dtype=self.model.particle_flags.dtype, device=self.model.device)
```

Store the NumPy rest positions in `self.cloth_rest_positions` and derive
`self.triangle_indices`, `self.inverse_rest_matrices`, and interior dihedral
indices with `newton.utils.MeshAdjacency` so the later LIMX activation does not
require rebuilding topology.

- [ ] **Step 4: Add IK keyframes and Featherstone integration**

Use one top-down TCP target at the cloth center and these phases:

```python
poses = np.asarray(
    [
        [1.5, 0.0, -0.5, 0.38, *GRIPPER_DOWN, 0.04],
        [1.2, 0.0, -0.5, CLOTH_HEIGHT, *GRIPPER_DOWN, 0.04],
        [0.8, 0.0, -0.5, CLOTH_HEIGHT, *GRIPPER_DOWN, 0.0],
        [1.5, 0.0, -0.5, 0.38, *GRIPPER_DOWN, 0.0],
        [0.8, 0.0, -0.5, 0.38, *GRIPPER_DOWN, 0.04],
    ],
    dtype=np.float32,
)
```

Locate `fr3_hand`, create position and rotation objectives with a 0.107 m TCP
offset, add `IKObjectiveJointLimit`, and solve 24 iterations per frame. Copy the
IK result to a 1-D target, compute
`joint_qd = (target_joint_q - state_0.joint_q) / frame_dt`, and call
`SolverFeatherstone.step(...)` for ten substeps while temporarily setting
`model.particle_count = 0` and gravity to zero. Restore both values after each
substep. No collision pipeline is created.

- [ ] **Step 5: Add motion metrics, validation, rendering, and CLI**

After each frame, evaluate the hand TCP from `state_0.body_q`; update:

```python
self.minimum_grasp_error = min(
    self.minimum_grasp_error,
    float(np.linalg.norm(tcp_position - np.asarray(self.grasp_position))),
)
self.maximum_tcp_height = max(self.maximum_tcp_height, float(tcp_position[2]))
```

`test_post_step()` checks finite joint/body/particle state. `test_final()` checks
the cloth stayed at `cloth_rest_positions`, `minimum_grasp_error < 0.03`, and
`maximum_tcp_height > grasp_position[2] + 0.10`. Render with `viewer.log_state`,
set a camera that shows the full table and arm, and provide a parser flag
`--no-graph-capture` matching the CUDA capture behavior.

- [ ] **Step 6: Run the focused tests**

Run:

```bash
uv run --extra dev -m unittest newton.tests.test_example_cloth_limx_franka
```

Expected: three passing tests (one CPU geometry test and two CUDA tests).

- [ ] **Step 7: Commit the runnable scene**

```bash
git add newton/examples/cloth/example_cloth_limx_franka.py \
  newton/tests/test_example_cloth_limx_franka.py
git commit -m "Add LIMX Franka grasp scene"
```

### Task 3: Example Registration, Screenshot, and Final Verification

**Files:**
- Modify: `newton/tests/test_examples.py`
- Modify: `README.md`
- Create: `docs/images/examples/example_cloth_limx_franka.jpg`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: CLI name `cloth_limx_franka` discovered from Task 2.
- Produces: README example card, generic example smoke-test registration, and 320-by-320 screenshot.

- [ ] **Step 1: Register the cloth smoke test**

Add beside `cloth.example_cloth_franka`:

```python
add_example_test(
    TestClothExamples,
    name="cloth.example_cloth_limx_franka",
    devices=cuda_test_devices,
    test_options={"num-frames": 360, "graph_capture": False},
    use_viewer=True,
)
```

- [ ] **Step 2: Run the GL scene and tune only layout values if needed**

Run:

```bash
uv run --extra examples -m newton.examples cloth_limx_franka \
  --viewer gl --num-frames 360
```

Accept when the square cloth is visibly flat on the tabletop and the Franka
approaches, descends with a top-down hand, closes around the cloth center, and
lifts without a discontinuous jump. Do not add collision to correct visual
overlap; only tune table, cloth, camera, and keyframe transforms.

- [ ] **Step 3: Capture the required 320-by-320 screenshot**

Run the scene headlessly at 320-by-320 through the close/lift transition, call
`ViewerGL.get_frame().numpy()`, and save it with Pillow as
`docs/images/examples/example_cloth_limx_franka.jpg` using JPEG quality 95:

```bash
uv run --extra examples python - <<'PY'
from types import SimpleNamespace

from PIL import Image
from newton.examples.cloth.example_cloth_limx_franka import Example
from newton.viewer import ViewerGL

viewer = ViewerGL(width=320, height=320, headless=True)
example = Example(viewer, SimpleNamespace(graph_capture=False))
for _ in range(230):
    example.step()
    example.render()
Image.fromarray(example.viewer.get_frame().numpy()).save(
    "docs/images/examples/example_cloth_limx_franka.jpg",
    quality=95,
)
viewer.close()
PY
```

Verify dimensions with:

```bash
uv run --extra examples python -c \
  'from PIL import Image; im=Image.open("docs/images/examples/example_cloth_limx_franka.jpg"); print(im.size)'
```

Expected output: `(320, 320)`.

- [ ] **Step 4: Add README and changelog entries**

Add a Cloth Examples card linking to
`newton/examples/cloth/example_cloth_limx_franka.py`, showing the new screenshot,
and displaying:

```html
<code>python -m newton.examples cloth_limx_franka</code>
```

Insert this line at a non-terminal position in `[Unreleased]` → `Added`:

```markdown
- Add a LIMX scene with a square cloth on a table and an IK-driven Franka grasp trajectory.
```

- [ ] **Step 5: Run targeted example verification**

Run:

```bash
uv run --extra dev -m unittest newton.tests.test_example_cloth_limx_franka
uv run --extra dev -m newton.tests -k example_cloth_limx_franka
uv run --extra examples -m newton.examples cloth_limx_franka \
  --viewer null --test --num-frames 360 --no-graph-capture
uvx pre-commit run --files \
  newton/examples/cloth/example_cloth_limx_franka.py \
  newton/tests/test_example_cloth_limx_franka.py \
  newton/tests/test_examples.py README.md CHANGELOG.md
```

Expected: all targeted tests and hooks pass. Do not run the full test suite for
this visual example unless a targeted failure points outside the changed files.

- [ ] **Step 6: Review and commit the presentation slice**

```bash
git diff --check
git status --short
git add CHANGELOG.md README.md \
  docs/images/examples/example_cloth_limx_franka.jpg \
  newton/tests/test_examples.py
git commit -m "Register LIMX Franka cloth example"
```

Confirm unrelated local changes remain unstaged.
