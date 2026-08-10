# LIMX ARAP Bunnies-in-a-Box Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run eight volumetric ARAP bunnies in one open box and prove that the existing LIMX VF/EE operator generates cross-bunny contacts with a fixed 3 mm thickness.

**Architecture:** Build eight disconnected copies of `bunny_tet.npz` in one `Model`, assemble all tetrahedra into one `ConstraintTetrahedronARAP`, and compose one shared `ConstraintSelfCollision` with five static-plane box contacts. A diagnostic Warp kernel classifies active VF/EE contact IDs by bunny index without copying the fixed-capacity contact arrays to the host.

**Tech Stack:** Python 3, NumPy, Warp, Newton `ModelBuilder`, `SolverLIMX`, `ConstraintTetrahedronARAP`, `ConstraintSelfCollision`, `ConstraintStaticPlaneContact`, `unittest`.

## Global Constraints

- Instantiate exactly eight bunnies in the approved deterministic `2 x 2 x 2` layout.
- Use scale `0.15`, density `1000 kg/m^3`, and ARAP stiffness `1e5 Pa` per bunny.
- Use fixed VF/EE thickness `0.003 m`; leave `geometry_radius_scale=None`.
- Use adaptive contact stiffness factors `(0.5, 0.3, 1.5)`, friction `0.05`, and `max_contacts=262144`.
- Set `enable_edge_face=False`; only VF and EE may detect or assemble contact.
- Use five inward box planes with `0.003 m` thickness, `2e4 N/m` stiffness, friction `0.05`, and zero normal damping.
- Derive the 8,624 collision vertices from the boundary triangle array; never
  use interior tetrahedral vertices as VF or box-plane candidates.
- Use `dt=0.01 s`, one step, one Newton iteration, 50 PCG iterations, and `velocity_damping=1.0`.
- Do not add substeps, line search, self-collision thickness adaptation, material damping, normal damping, or another collision algorithm.
- Tests use `unittest`; every test method has an imperative triple-double-quoted docstring.
- Preserve the default all-particle behavior of static-plane contact while
  adding an optional surface-particle subset.

---

### Task 1: Drive the scene contract with a failing configuration test

**Files:**
- Create: `newton/tests/test_example_softbody_limx_arap_bunnies_box.py`

**Interfaces:**
- Consumes: standard `Example(ViewerNull(...), None)` constructor.
- Produces: an executable contract for exact topology counts and collision/integration parameters.

- [ ] **Step 1: Write the missing-module test**

Create the test file with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest

import warp as wp

from newton.viewer import ViewerNull


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestLimxArapBunniesBoxExample(unittest.TestCase):
    def test_uses_fixed_thickness_multi_bunny_configuration(self):
        """Use eight bunnies with fixed 3 mm VF/EE and one undamped Newton step."""
        module = importlib.import_module(
            "newton.examples.softbody.example_softbody_limx_arap_bunnies_box"
        )
        example = module.Example(ViewerNull(num_frames=1), None)

        self.assertEqual(example.bunny_count, 8)
        self.assertEqual(example.particles_per_bunny, 1869)
        self.assertEqual(example.model.particle_count, 14952)
        self.assertEqual(example.model.tet_count, 58848)
        self.assertEqual(example.model.tri_count, 17216)
        self.assertEqual(example.model.body_count, 0)
        self.assertEqual(example.self_collision.thickness, 0.003)
        self.assertIsNone(example.self_collision.geometry_radius_scale)
        self.assertIsNone(example.self_collision.stiffness)
        self.assertEqual(example.self_collision.stiffness_factors, (0.5, 0.3, 1.5))
        self.assertEqual(example.self_collision.friction, 0.05)
        self.assertEqual(example.self_collision.max_contacts, 262144)
        self.assertEqual(len(example.box_contacts), 5)
        self.assertTrue(all(contact.thickness == 0.003 for contact in example.box_contacts))
        self.assertTrue(all(contact.normal_damping == 0.0 for contact in example.box_contacts))
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)
```

- [ ] **Step 2: Run the test and verify the intended failure**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_softbody_limx_arap_bunnies_box -v
```

Expected: FAIL with `ModuleNotFoundError` naming `example_softbody_limx_arap_bunnies_box`.

### Task 2: Implement the eight-bunny scene

**Files:**
- Create: `newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py`

**Interfaces:**
- Consumes: `bunny_tet.npz` and the existing public LIMX constraint classes.
- Produces: standard `Example` hooks plus `saw_cross_bunny_contact: wp.array[int]` for test diagnostics.

- [ ] **Step 1: Add deterministic layout and contact-classification helpers**

Define the approved eight `(center, rpy_degrees, velocity)` tuples as a module constant. Compose each orientation as:

```python
source_to_world = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
perturbation = wp.quat_rpy(*(math.radians(value) for value in rpy_degrees))
rotation = wp.normalize(perturbation * source_to_world)
```

Add this diagnostic kernel:

```python
@wp.kernel
def _mark_cross_bunny_contacts(
    contact_ids: wp.array2d[int],
    contact_count: wp.array[int],
    contact_capacity: int,
    arity: int,
    feature_split: int,
    particles_per_bunny: int,
    saw_cross_contact: wp.array[int],
):
    contact = wp.tid()
    if contact >= wp.min(contact_count[0], contact_capacity):
        return
    bunny_0 = contact_ids[contact, 0] // particles_per_bunny
    for local_index in range(feature_split, arity):
        if contact_ids[contact, local_index] // particles_per_bunny != bunny_0:
            wp.atomic_max(saw_cross_contact, 0, 1)
```

- [ ] **Step 2: Build the eight soft meshes and visible open box**

For every layout row call `builder.add_soft_mesh()` with the shared asset,
`scale=0.15`, `density=1000.0`, all native material coefficients zero, and
`add_surface_mesh_edges=False`. Add a visible floor and four wall shapes with
`body=-1`; place their inner surfaces at X `±0.36`, Y `±0.40`, and Z `0`.

After `builder.finalize()`, assert internally through normal constructor
validation that the resulting counts are 14,952 particles, 58,848 tetrahedra,
and 17,216 triangles. Store `tetrahedra`, `particle_masses`, and the eight
particle index ranges for diagnostics.

Compute the sorted unique indices referenced by `model.tri_indices`. Assert
that there are 8,624 surface vertices and use this compact set for all VF and
box-plane candidate launches. The remaining 6,328 particles are interior ARAP
degrees of freedom only.

- [ ] **Step 3: Create ARAP, VF/EE, and box constraints**

Use:

```python
self.arap_constraint = newton.solvers.ConstraintTetrahedronARAP(
    self.tetrahedra.tolist(),
    [wp.mat33(*matrix.reshape(-1)) for matrix in self.model.tet_poses.numpy()],
    [1.0e5] * self.model.tet_count,
    self.model.particle_count,
    self.model.device,
)
self.self_collision = newton.solvers.ConstraintSelfCollision(
    self.model,
    thickness=0.003,
    stiffness=None,
    max_contacts=262144,
    stiffness_factors=(0.5, 0.3, 1.5),
    geometry_radius_scale=None,
    friction=0.05,
    friction_epsilon=1.0e-2,
    enable_edge_face=False,
)
```

Create floor and wall plane equations:

```python
(
    ((0.0, 0.0, 1.0), 0.0),
    ((1.0, 0.0, 0.0), -0.36),
    ((-1.0, 0.0, 0.0), -0.36),
    ((0.0, 1.0, 0.0), -0.40),
    ((0.0, -1.0, 0.0), -0.40),
)
```

Give every plane the approved 3 mm, `2e4 N/m`, zero-damping, `0.05`
friction parameters and the shared boundary vertex indices. Compose collision
and planes through
`ConstraintGroupDynamic`, then construct `SolverLIMX` with one Newton and 50
PCG iterations.

- [ ] **Step 4: Implement step, rendering, and validation hooks**

`step()` clears forces, applies viewer forces, advances once, swaps states,
and increments time. `render()` logs the combined state.

`test_post_step()` must:

```python
positions = self.state_0.particle_q.numpy()
velocities = self.state_0.particle_qd.numpy()
if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
    raise AssertionError("LIMX ARAP bunnies state must remain finite")
edges = np.stack(
    (
        positions[self.tetrahedra[:, 1]] - positions[self.tetrahedra[:, 0]],
        positions[self.tetrahedra[:, 2]] - positions[self.tetrahedra[:, 0]],
        positions[self.tetrahedra[:, 3]] - positions[self.tetrahedra[:, 0]],
    ),
    axis=2,
)
if float(np.linalg.det(edges).min()) <= 0.0:
    raise AssertionError("LIMX ARAP bunnies tetrahedra must remain positive-volume")
below_wall_top = positions[:, 2] <= self.box_wall_top
contained = positions[below_wall_top]
if float(positions[:, 2].min()) < self.box_floor - 0.04:
    raise AssertionError("A bunny penetrated catastrophically below the box")
if len(contained) and (
    float(contained[:, 0].min()) < self.box_min[0] - 0.04
    or float(contained[:, 0].max()) > self.box_max[0] + 0.04
    or float(contained[:, 1].min()) < self.box_min[1] - 0.04
    or float(contained[:, 1].max()) > self.box_max[1] + 0.04
):
    raise AssertionError("A bunny escaped catastrophically through a box wall")
for buffer in (
    self.self_collision.vertex_face_contacts,
    self.self_collision.edge_edge_contacts,
):
    if int(buffer.overflow_count.numpy()[0]) != 0:
        raise AssertionError("LIMX ARAP bunny contact capacity overflowed")
```

Then launch `_mark_cross_bunny_contacts` over `max_contacts` for the VF buffer
with `(arity=4, feature_split=1)` and the EE buffer with
`(arity=4, feature_split=2)`. The persistent flag is never cleared.

`test_final()` calls `test_post_step()`, requires positive simulation time,
and for runs of at least 0.5 s requires `saw_cross_bunny_contact[0] == 1`.
Set the parser default to 300 frames and frame time to `0.01 s`.

- [ ] **Step 5: Run focused tests and format**

Run:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_example_softbody_limx_arap_bunnies_box -v
uvx pre-commit run --files \
  newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py \
  newton/tests/test_example_softbody_limx_arap_bunnies_box.py
```

Expected: the configuration test passes with no warnings.

- [ ] **Step 6: Commit the scene slice**

```bash
git add newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py \
  newton/tests/test_example_softbody_limx_arap_bunnies_box.py
git commit -m "Add LIMX ARAP bunnies box scene"
```

### Task 3: Exercise real cross-bunny contact

**Files:**
- Modify: `newton/tests/test_examples.py`
- Modify only if diagnostics expose an implementation defect: `newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py`

**Interfaces:**
- Consumes: example runner test hooks and persistent cross-contact flag.
- Produces: a CUDA smoke registration and a 300-frame stability/contact result.

- [ ] **Step 1: Register an 80-frame CUDA smoke run**

Add after the single-bunny entry:

```python
add_example_test(
    TestSoftbodyExamples,
    name="softbody.example_softbody_limx_arap_bunnies_box",
    devices=cuda_test_devices,
    test_options={"num-frames": 80},
    use_viewer=True,
)
```

- [ ] **Step 2: Run smoke and full stability tests**

Run:

```bash
uv run --extra dev -m newton.tests \
  -k test_softbody.example_softbody_limx_arap_bunnies_box
uv run -m newton.examples softbody_limx_arap_bunnies_box \
  --device cuda:0 --viewer null --test --num-frames 300
```

Expected: both pass, contact buffers do not overflow, all tetrahedra remain
positive-volume, the box retains particles below its open top, and at least
one cross-bunny VF/EE contact is recorded.

- [ ] **Step 3: Commit test registration**

```bash
git add newton/tests/test_examples.py
git commit -m "Test LIMX ARAP bunnies collision"
```

### Task 4: Launch the visual experiment

**Files:**
- No committed files unless the user later accepts documentation work.

**Interfaces:**
- Consumes: validated scene.
- Produces: a long-running OpenGL window for user inspection.

- [ ] **Step 1: Launch the fixed-thickness scene**

Run:

```bash
uv run -m newton.examples softbody_limx_arap_bunnies_box \
  --device cuda:0 --viewer gl --num-frames 3000
```

- [ ] **Step 2: Report the observed fixed-thickness baseline**

Report the maximum VF/EE/EF counts, any overflow, minimum tetrahedron
determinant, maximum box penetration, whether cross-bunny VF/EE occurred, and
whether visible jitter persists. Do not enable geometry-aware thickness in
this scene without a new user decision.
