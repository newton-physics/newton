# LIMX ARAP Bunny Table Drop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained volumetric bunny example that falls 0.15 m onto a visible table and exercises LIMX tetrahedral ARAP plus static-plane penalty contact for 3 s without inversion or unbounded motion.

**Architecture:** Convert libuipc's tetrahedral `.msh` once into a dependency-free Newton `.npz` asset, then instantiate it through `ModelBuilder.add_soft_mesh()`. The existing `ConstraintTetrahedronARAP` supplies static elastic force/Hessian blocks and `ConstraintStaticPlaneContact` supplies matrix-free table force/Hessian contributions; a `body=-1` box renders the table but does not participate in the solve.

**Tech Stack:** Python 3, NumPy, Warp, Newton `TetMesh`/`ModelBuilder`, `SolverLIMX`, `unittest`, Newton example runner.

## Global Constraints

- Use the approved infinite Z-up contact plane plus a finite visualization-only box whose top is `z = 0`.
- Use contact thickness `0.003 m`, stiffness `2e4 N/m`, normal damping `0`, friction `0.05`, and friction epsilon `1e-4 m`.
- Use `dt = 0.01 s`, one step per frame, one Newton iteration, 128 PCG iterations, and `velocity_damping = 1.0`.
- Use ARAP stiffness `1e5 Pa`, density `1000 kg/m^3`, uniform scale `0.15`, and translation `(0, 0, 0.25) m`.
- Do not add line search, substeps, self-collision, material damping, contact damping, or a runtime `meshio` dependency.
- Preserve the source topology and record libuipc provenance and Apache-2.0 licensing.
- Tests use `unittest`; every test method has an imperative triple-double-quoted docstring.
- Do not add or change a public Python API.

---

### Task 1: Commit the self-contained volumetric bunny asset

**Files:**
- Create: `newton/examples/assets/bunny_tet.npz`
- Create: `newton/licenses/libuipc-bunny-tet.txt`
- Create: `newton/tests/test_example_softbody_limx_arap_bunny_table.py`

**Interfaces:**
- Consumes: libuipc `assets/sim_data/tetmesh/bunny0.msh` at revision `619a5412cf958eb59cf3c9cebc9a8e8e9625ebd3`.
- Produces: `newton.TetMesh.create_from_file(".../bunny_tet.npz")`, with 1,869 vertices, 7,356 tetrahedra, and 2,152 boundary triangles.

- [ ] **Step 1: Write the failing asset-integrity test**

Create `newton/tests/test_example_softbody_limx_arap_bunny_table.py` with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import unittest

import numpy as np

import newton


ASSET_PATH = Path(__file__).resolve().parents[1] / "examples" / "assets" / "bunny_tet.npz"


class TestBunnyTetAsset(unittest.TestCase):
    def test_preserves_source_topology_and_orientation(self):
        """Preserve the source bunny topology and positive tetrahedron orientation."""
        mesh = newton.TetMesh.create_from_file(str(ASSET_PATH))
        tetrahedra = mesh.tet_indices.reshape(-1, 4)
        edges = np.stack(
            (
                mesh.vertices[tetrahedra[:, 1]] - mesh.vertices[tetrahedra[:, 0]],
                mesh.vertices[tetrahedra[:, 2]] - mesh.vertices[tetrahedra[:, 0]],
                mesh.vertices[tetrahedra[:, 3]] - mesh.vertices[tetrahedra[:, 0]],
            ),
            axis=2,
        )

        self.assertEqual(mesh.vertex_count, 1869)
        self.assertEqual(mesh.tet_count, 7356)
        self.assertEqual(len(mesh.surface_tri_indices) // 3, 2152)
        self.assertGreater(float(np.linalg.det(edges).min()), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and confirm the asset is missing**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_softbody_limx_arap_bunny_table.TestBunnyTetAsset -v
```

Expected: FAIL with `FileNotFoundError` naming `bunny_tet.npz`.

- [ ] **Step 3: Convert only vertices and tetrahedron connectivity**

Run:

```bash
uv run --extra dev python - <<'PY'
from pathlib import Path
import newton

source_path = Path("/home/limx/github/libuipc/assets/sim_data/tetmesh/bunny0.msh")
target_path = Path("newton/examples/assets/bunny_tet.npz")
source = newton.TetMesh.create_from_file(str(source_path))
converted = newton.TetMesh(source.vertices, source.tet_indices)
converted.save(str(target_path))
PY
```

This deliberately strips meshio/Gmsh metadata while preserving the vertex array and ordered tetrahedron connectivity exactly.

- [ ] **Step 4: Add the third-party asset notice**

Create `newton/licenses/libuipc-bunny-tet.txt` with:

```text
libuipc tetrahedral bunny asset
================================

Newton file: newton/examples/assets/bunny_tet.npz
Source repository: https://github.com/spiriMirror/libuipc.git
Source path: assets/sim_data/tetmesh/bunny0.msh
Inspected revision: 619a5412cf958eb59cf3c9cebc9a8e8e9625ebd3
First asset revision: 32988ff1237d74dc0dd0eef2bec9ee6f8898c7a2
License: Apache License, Version 2.0

The Newton file preserves the source vertex positions and ordered tetrahedron
connectivity. It changes only the container from Gmsh `.msh` to NumPy `.npz`
and regenerates boundary triangles from that connectivity at load time.

The Apache License, Version 2.0 terms are provided in the repository-root
LICENSE.md file.
```

- [ ] **Step 5: Run the asset test and inspect the binary diff summary**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_softbody_limx_arap_bunny_table.TestBunnyTetAsset -v
git diff --stat
```

Expected: one test passes; the asset has 1,869 vertices, 7,356 tetrahedra, 2,152 boundary triangles, and only positive determinants.

- [ ] **Step 6: Commit the asset slice**

```bash
git add newton/examples/assets/bunny_tet.npz \
  newton/licenses/libuipc-bunny-tet.txt \
  newton/tests/test_example_softbody_limx_arap_bunny_table.py
git commit -m "Add volumetric bunny example asset"
```

### Task 2: Build the ARAP bunny table-drop example

**Files:**
- Create: `newton/examples/softbody/example_softbody_limx_arap_bunny_table.py`
- Modify: `newton/tests/test_example_softbody_limx_arap_bunny_table.py`

**Interfaces:**
- Consumes: `bunny_tet.npz`, `ConstraintTetrahedronARAP(tetrahedra, inverse_rest_matrices, stiffnesses, particle_count, device)`, and `ConstraintStaticPlaneContact(...)`.
- Produces: standard `Example(viewer, args=None)` with `step()`, `render()`, `test_post_step()`, `test_final()`, and `create_parser()`.

- [ ] **Step 1: Write a failing configuration test**

Append the imports and CUDA test below to the asset test file:

```python
import importlib
import warp as wp
from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestLimxArapBunnyTableExample(unittest.TestCase):
    def test_uses_approved_contact_and_solver_configuration(self):
        """Use the approved undamped one-step LIMX table-contact configuration."""
        module = importlib.import_module(
            "newton.examples.softbody.example_softbody_limx_arap_bunny_table"
        )
        example = module.Example(ViewerNull(num_frames=1), None)

        self.assertEqual(example.frame_dt, 0.01)
        self.assertEqual(example.model.particle_count, 1869)
        self.assertEqual(example.model.tet_count, 7356)
        self.assertEqual(example.model.body_count, 0)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 128)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertIs(example.solver.dynamic_operator, example.table_contact)
        self.assertEqual(example.table_contact.thickness, 0.003)
        self.assertEqual(example.table_contact.stiffness, 2.0e4)
        self.assertEqual(example.table_contact.normal_damping, 0.0)
        self.assertEqual(example.table_contact.friction, 0.05)
```

- [ ] **Step 2: Run the test and confirm the example module is absent**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_softbody_limx_arap_bunny_table.TestLimxArapBunnyTableExample -v
```

Expected: FAIL with `ModuleNotFoundError` for `example_softbody_limx_arap_bunny_table`.

- [ ] **Step 3: Implement the scene and its stability diagnostics**

Create `newton/examples/softbody/example_softbody_limx_arap_bunny_table.py` with these concrete elements:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    """Drop a volumetric bunny onto a table with LIMX ARAP elasticity."""

    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.frame_dt = 0.01
        self.sim_time = 0.0
        self.table_top = 0.0
        self.minimum_height = np.inf
        self.maximum_speeds: list[float] = []
        self.center_heights: list[float] = []

        mesh_path = Path(__file__).resolve().parents[1] / "assets" / "bunny_tet.npz"
        mesh = newton.TetMesh.create_from_file(str(mesh_path))
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.25),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi),
            scale=0.15,
            vel=wp.vec3(0.0),
            mesh=mesh,
            density=1000.0,
            k_mu=0.0,
            k_lambda=0.0,
            k_damp=0.0,
            add_surface_mesh_edges=False,
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.03), wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=0.03,
            color=wp.vec3(0.42, 0.46, 0.52),
        )
        self.model = builder.finalize()
        self.rest_positions = self.model.particle_q.numpy()
        self.tetrahedra = self.model.tet_indices.numpy()
        self.particle_masses = self.model.particle_mass.numpy()
        self.initial_center_height = float(
            np.average(self.rest_positions[:, 2], weights=self.particle_masses)
        )
        self.initial_minimum_height = float(self.rest_positions[:, 2].min())

        inverse_rest_matrices = self.model.tet_poses.numpy()
        self.arap_constraint = newton.solvers.ConstraintTetrahedronARAP(
            self.tetrahedra.tolist(),
            [wp.mat33(*matrix.reshape(-1)) for matrix in inverse_rest_matrices],
            [1.0e5] * self.model.tet_count,
            self.model.particle_count,
            self.model.device,
        )
        self.table_contact = newton.solvers.ConstraintStaticPlaneContact(
            normal=(0.0, 0.0, 1.0),
            offset=self.table_top,
            thickness=0.003,
            stiffness=2.0e4,
            normal_damping=0.0,
            friction=0.05,
            friction_epsilon=1.0e-4,
            particle_count=self.model.particle_count,
            device=self.model.device,
        )
        self.solver = newton.solvers.SolverLIMX(
            self.model,
            [self.arap_constraint],
            nonlinear_iterations=1,
            linear_iterations=128,
            velocity_damping=1.0,
            dynamic_operator=self.table_contact,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.62, -0.82, 0.48), -8.0, 143.0)

    def step(self):
        """Advance one undamped 0.01-second Newton step."""
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        """Render the deforming bunny and visualization table."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Keep the falling bunny finite, positive-volume, and near the scene."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise AssertionError("LIMX ARAP bunny state must remain finite")
        edges = np.stack(
            (
                positions[self.tetrahedra[:, 1]] - positions[self.tetrahedra[:, 0]],
                positions[self.tetrahedra[:, 2]] - positions[self.tetrahedra[:, 0]],
                positions[self.tetrahedra[:, 3]] - positions[self.tetrahedra[:, 0]],
            ),
            axis=2,
        )
        if float(np.linalg.det(edges).min()) <= 0.0:
            raise AssertionError("LIMX ARAP bunny tetrahedra must remain positive-volume")
        minimum_height = float(positions[:, 2].min())
        if minimum_height < -0.015:
            raise AssertionError("LIMX ARAP bunny penetrated more than 15 mm below the table")
        center_height = float(np.average(positions[:, 2], weights=self.particle_masses))
        if not -0.05 <= center_height <= 1.0:
            raise AssertionError("LIMX ARAP bunny center left the bounded scene")
        self.minimum_height = min(self.minimum_height, minimum_height)
        self.maximum_speeds.append(float(np.linalg.norm(velocities, axis=1).max()))
        self.center_heights.append(center_height)

    def test_final(self):
        """Reach the table without inversion or increasing-amplitude motion."""
        self.test_post_step()
        if self.sim_time >= 0.25:
            if self.minimum_height > 0.03:
                raise AssertionError("LIMX ARAP bunny did not reach the table")
            if self.center_heights[-1] >= self.initial_center_height - 0.05:
                raise AssertionError("LIMX ARAP bunny did not fall toward the table")
        if len(self.maximum_speeds) >= 100:
            previous_peak = max(self.maximum_speeds[-100:-50])
            late_peak = max(self.maximum_speeds[-50:])
            if late_peak > max(1.5 * previous_peak, 5.0):
                raise AssertionError("LIMX ARAP bunny motion grows after impact")

    @staticmethod
    def create_parser():
        """Create the standard Newton example parser."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
```

- [ ] **Step 4: Run the focused test and format both Python files**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_softbody_limx_arap_bunny_table -v
uvx pre-commit run ruff-format --files \
  newton/examples/softbody/example_softbody_limx_arap_bunny_table.py \
  newton/tests/test_example_softbody_limx_arap_bunny_table.py
uvx pre-commit run ruff --files \
  newton/examples/softbody/example_softbody_limx_arap_bunny_table.py \
  newton/tests/test_example_softbody_limx_arap_bunny_table.py
```

Expected: asset and configuration tests pass.

- [ ] **Step 5: Commit the example slice**

```bash
git add newton/examples/softbody/example_softbody_limx_arap_bunny_table.py \
  newton/tests/test_example_softbody_limx_arap_bunny_table.py
git commit -m "Add LIMX ARAP bunny table drop"
```

### Task 3: Cross first impact and validate the full three-second run

**Files:**
- Modify: `newton/tests/test_examples.py`
- Modify if diagnostics reveal an actual failure: `newton/examples/softbody/example_softbody_limx_arap_bunny_table.py`

**Interfaces:**
- Consumes: example runner hooks `test_post_step()` after every `--test` frame and `test_final()` after the run.
- Produces: CUDA smoke coverage that reaches first contact, plus a direct 300-frame stability result.

- [ ] **Step 1: Register a CUDA example smoke test**

Add after the existing LIMX ARAP beam registration in `newton/tests/test_examples.py`:

```python
add_example_test(
    TestSoftbodyExamples,
    name="softbody.example_softbody_limx_arap_bunny_table",
    devices=cuda_test_devices,
    test_options={"num-frames": 40},
    use_viewer=True,
)
```

Forty frames cover the approximately 0.175-second free fall and the initial contact response without turning routine CI into the three-second stability benchmark.

- [ ] **Step 2: Run the registered smoke test**

Run:

```bash
uv run --extra dev -m newton.tests \
  -k test_softbody.example_softbody_limx_arap_bunny_table
```

Expected: PASS on CUDA with finite state, positive volumes, no more than 15 mm table penetration, and the bunny having reached the table.

- [ ] **Step 3: Run the complete stability case directly**

Run:

```bash
uv run -m newton.examples softbody_limx_arap_bunny_table \
  --device cuda:0 --viewer null --test --num-frames 300
```

Expected: PASS after 3 s; no non-finite state, inversion, catastrophic penetration, escaped center of mass, or late-window speed growth.

- [ ] **Step 4: Diagnose failures without adding damping**

If either run fails, print the failing frame, minimum height, minimum determinant, center height, and previous/late speed peaks from the already-recorded arrays. Adjust only a demonstrably incorrect scene transform, diagnostic bound, contact stiffness, or linear-iteration convergence issue; retain all Global Constraints and rerun Steps 2 and 3.

- [ ] **Step 5: Commit the regression registration**

```bash
git add newton/tests/test_examples.py \
  newton/examples/softbody/example_softbody_limx_arap_bunny_table.py
git commit -m "Test LIMX ARAP bunny table impact"
```

### Task 4: Launch the scene and document the accepted example

**Files:**
- Create: `docs/images/examples/example_softbody_limx_arap_bunny_table.jpg`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: stable 300-frame example and Newton viewer.
- Produces: an interactive visual review, 320-by-320 screenshot, README command, and `[Unreleased]` changelog entry.

- [ ] **Step 1: Launch the interactive 300-frame scene**

Run:

```bash
uv run -m newton.examples softbody_limx_arap_bunny_table \
  --device cuda:0 --viewer gl --num-frames 300
```

Visually confirm that the bunny falls about 15 cm, contacts the table, stays above it, retains positive volume, and does not develop increasing-amplitude bounce or jitter.

- [ ] **Step 2: Capture and inspect the documentation image**

Capture a settled or informative impact frame as `docs/images/examples/example_softbody_limx_arap_bunny_table.jpg`, resize/crop it to exactly 320 by 320 pixels, and inspect the pixels to confirm that the bunny and table are both visible and centered.

- [ ] **Step 3: Register the example in the README Softbody table**

Add a new cell beside the beam entry using:

```html
<a href="https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_limx_arap_bunny_table.py">
  <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_softbody_limx_arap_bunny_table.jpg" alt="LIMX ARAP Bunny Table Drop">
</a>
```

and the matching command:

```html
<code>python -m newton.examples softbody_limx_arap_bunny_table</code>
```

- [ ] **Step 4: Add the user-facing changelog entry**

Insert this bullet at a non-terminal position within `[Unreleased]` → `Added`:

```markdown
- Add a LIMX tetrahedral ARAP bunny table-drop example with static-plane contact.
```

- [ ] **Step 5: Run final verification**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_softbody_limx_arap_bunny_table -v
uv run --extra dev -m newton.tests \
  -k test_softbody.example_softbody_limx_arap_bunny_table
uv run -m newton.examples softbody_limx_arap_bunny_table \
  --device cuda:0 --viewer null --test --num-frames 300
uvx pre-commit run -a
git diff --check
git status --short
```

Expected: all tests and hooks pass, the 300-frame stability run passes, and only the intended documentation changes remain before commit.

- [ ] **Step 6: Commit the documentation slice**

```bash
git add README.md CHANGELOG.md \
  docs/images/examples/example_softbody_limx_arap_bunny_table.jpg
git commit -m "Document LIMX ARAP bunny example"
```
