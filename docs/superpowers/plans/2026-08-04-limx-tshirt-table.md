# LIMX T-Shirt Table Contact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CUDA LIMX example in which the repository's T-shirt asset is thrown onto a static table, collides with itself and the tabletop, and settles through contact-only damping and friction.

**Architecture:** Keep fixed-topology membrane and bending energies in the existing assembled 3×3 block-CSR operator. Introduce a dynamic-constraint group that composes matrix-free self-collision and a new one-particle analytic plane contact operator; cache lagged velocity once per time step and rebuild active contact force/Hessian data at each Newton linearization. The example renders a finite static box while deliberately using its infinite top plane for collision.

**Tech Stack:** Python 3.11+, Warp CUDA kernels and CUDA graphs, Newton `ModelBuilder`/`SolverLIMX`, USD mesh loading, `unittest`, NumPy.

## Global Constraints

- Run routine simulation validation on `cuda:0`; use CPU only to diagnose a CUDA failure.
- Use `uv` and Newton's `unittest` runner; do not invoke pytest.
- Do not run the complete mixed CPU/GPU `test_solver_limx.py` module; select exact CUDA tests with both `-p` and `-k`.
- Keep `velocity_damping=1.0`; all settling must come from contact-only normal damping and friction.
- Use `dt=0.01`, one Newton iteration, 50 PCG iterations, and one physics step per rendered frame.
- Keep static elastic topology in 3×3 block-CSR and dynamic contact topology matrix-free.
- Do not add rigid-body degrees of freedom, moving colliders, arbitrary mesh collision, or table-edge support.
- Examples and docs must use public Newton imports and must not import from `newton._src`.
- Add no required or optional dependencies.
- Preserve unrelated `lessons.md` and `solver_convergence.png` workspace changes.

---

### Task 1: Dynamic-constraint lifecycle and composition

**Files:**
- Create: `newton/_src/solvers/limx/constraints/dynamic_group.py`
- Create: `newton/tests/test_solver_limx_static_contact.py`
- Modify: `newton/_src/solvers/limx/operator.py`
- Modify: `newton/_src/solvers/limx/solver_newton.py`
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/tests/test_solver_limx.py`

**Interfaces:**
- Consumes: dynamic children exposing `particle_count: int`, `device: wp.context.Device`, `begin_step(positions, velocities, dt)`, `prepare(positions)`, `accumulate_force(positions, output)`, `hessian_multiply(positions, vector, output)`, and `accumulate_diagonal(positions, output)`.
- Produces: `ConstraintGroupDynamic(constraints: Sequence[Any])`; a once-per-step `begin_step(positions: wp.array[wp.vec3], velocities: wp.array[wp.vec3], dt: float) -> None` lifecycle hook on `SolverLIMX` dynamic operators.

- [ ] **Step 1: Write CUDA tests for ordered forwarding and lifecycle counts**

Add a CUDA-only test module with a small recording operator and an isolated public-export assertion:

```python
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.limx.constraints.dynamic_group import ConstraintGroupDynamic


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintGroupDynamic(unittest.TestCase):
    device = wp.get_device("cuda:0")

    def test_public_exports(self):
        self.assertIs(newton.solvers.ConstraintGroupDynamic, ConstraintGroupDynamic)

    def test_group_forwards_in_child_order(self):
        events = []

        class RecordingConstraint:
            particle_count = 2
            device = self.device

            def __init__(self, name):
                self.name = name

            def begin_step(self, positions, velocities, dt):
                events.append((self.name, "begin_step", dt))

            def prepare(self, positions):
                events.append((self.name, "prepare"))

            def accumulate_force(self, positions, output):
                events.append((self.name, "force"))

            def hessian_multiply(self, positions, vector, output):
                events.append((self.name, "hessian"))

            def accumulate_diagonal(self, positions, output):
                events.append((self.name, "diagonal"))

        group = newton.solvers.ConstraintGroupDynamic(
            [RecordingConstraint("self"), RecordingConstraint("plane")]
        )
        q = wp.zeros(2, dtype=wp.vec3, device=self.device)
        v = wp.zeros_like(q)
        blocks = wp.zeros(2, dtype=wp.mat33, device=self.device)
        group.begin_step(q, v, 0.01)
        group.prepare(q)
        group.accumulate_force(q, v)
        group.hessian_multiply(q, v, v)
        group.accumulate_diagonal(q, blocks)
        self.assertEqual(
            events,
            [
                ("self", "begin_step", 0.01),
                ("plane", "begin_step", 0.01),
                ("self", "prepare"),
                ("plane", "prepare"),
                ("self", "force"),
                ("plane", "force"),
                ("self", "hessian"),
                ("plane", "hessian"),
                ("self", "diagonal"),
                ("plane", "diagonal"),
            ],
        )
```

Extend the existing solver lifecycle test's recording dynamic operator with `begin_step`, then assert exactly one `begin_step` event occurs before the first of `nonlinear_iterations` `prepare` events. The event must contain `state_in.particle_q`, `state_in.particle_qd`, and `dt` by object identity/value.

- [ ] **Step 2: Run the focused CUDA tests and observe the missing API failures**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx_static_contact.py -k TestConstraintGroupDynamic
uv run --extra dev -m newton.tests -p test_solver_limx.py -k dynamic_contacts_prepare_once_before_each_newton_linearization
```

Expected: the new module fails to import `ConstraintGroupDynamic`/`ConstraintStaticPlaneContact`, and the lifecycle test fails because `begin_step` is never called.

- [ ] **Step 3: Implement the no-op and self-collision lifecycle hooks**

Add the following method to both `EmptyDynamicConstraintOperator` and `ConstraintSelfCollision`:

```python
def begin_step(
    self,
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    dt: float,
) -> None:
    """Cache no per-step state."""
```

The body remains empty. Do not move self-collision detection out of its existing per-Newton `prepare(positions)` call.

- [ ] **Step 4: Implement `ConstraintGroupDynamic` with strict shared-domain validation**

Create the group as a thin ordered dispatcher:

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import warp as wp


class ConstraintGroupDynamic:
    """Compose matrix-free dynamic constraints over one particle domain."""

    def __init__(self, constraints: Sequence[Any]):
        self.constraints = tuple(constraints)
        if not self.constraints:
            raise ValueError("constraints must not be empty")
        self.particle_count = self.constraints[0].particle_count
        self.device = wp.get_device(self.constraints[0].device)
        for constraint in self.constraints:
            if constraint.particle_count != self.particle_count:
                raise ValueError("Every dynamic constraint must use the same particle count")
            if wp.get_device(constraint.device) != self.device:
                raise ValueError("Every dynamic constraint must use the same device")

    def begin_step(self, positions, velocities, dt):
        for constraint in self.constraints:
            constraint.begin_step(positions, velocities, dt)

    def prepare(self, positions):
        for constraint in self.constraints:
            constraint.prepare(positions)

    def accumulate_force(self, positions, output):
        for constraint in self.constraints:
            constraint.accumulate_force(positions, output)

    def hessian_multiply(self, positions, vector, output):
        for constraint in self.constraints:
            constraint.hessian_multiply(positions, vector, output)

    def accumulate_diagonal(self, positions, output):
        for constraint in self.constraints:
            constraint.accumulate_diagonal(positions, output)
```

Use the project Warp bracket annotations on all array arguments in the actual implementation and Google-style public docstrings.

- [ ] **Step 5: Call `begin_step` once from `SolverLIMX.step()`**

Immediately after `_initialize_step` and before the Newton loop, insert:

```python
begin_dynamic_step = getattr(self.dynamic_operator, "begin_step", None)
if begin_dynamic_step is not None:
    begin_dynamic_step(state_in.particle_q, state_in.particle_qd, dt)
```

Keep `prepare(self.iterate_positions)` inside the Newton loop. This makes lagged damping/friction independent of how many nonlinear iterations are configured.

- [ ] **Step 6: Export the group through every public solver layer**

Add `ConstraintGroupDynamic` to `constraints/__init__.py`, `limx/__init__.py`, and the TYPE_CHECKING, `__all__`, and `_LAZY_IMPORTS` sections of `_src/solvers/__init__.py`.

- [ ] **Step 7: Run the focused CUDA lifecycle and group tests**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx_static_contact.py -k TestConstraintGroupDynamic
uv run --extra dev -m newton.tests -p test_solver_limx.py -k dynamic_contacts_prepare_once_before_each_newton_linearization
```

Expected: all selected tests pass on `cuda:0`, with one `begin_step` and one `prepare` per Newton iteration.

- [ ] **Step 8: Commit the independently usable composition layer**

```bash
git add newton/_src/solvers/limx/operator.py \
  newton/_src/solvers/limx/solver_newton.py \
  newton/_src/solvers/limx/constraints/self_collision.py \
  newton/_src/solvers/limx/constraints/dynamic_group.py \
  newton/_src/solvers/limx/constraints/__init__.py \
  newton/_src/solvers/limx/__init__.py \
  newton/_src/solvers/__init__.py \
  newton/tests/test_solver_limx.py \
  newton/tests/test_solver_limx_static_contact.py
git commit -m "Add LIMX dynamic constraint composition"
```

### Task 2: Matrix-free static-plane contact

**Files:**
- Create: `newton/_src/solvers/limx/constraints/static_plane_contact.py`
- Modify: `newton/tests/test_solver_limx_static_contact.py`
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `docs/api/newton_solvers.rst` (generated)

**Interfaces:**
- Consumes: the Task 1 lifecycle and standard matrix-free dynamic-operator methods.
- Produces: `ConstraintStaticPlaneContact(normal, offset, thickness, stiffness, normal_damping, friction, friction_epsilon, particle_count, device)` with one cached `wp.vec3` force and one cached `wp.mat33` PSD Hessian per particle.

- [ ] **Step 1: Write an exact force/Hessian/diagonal CUDA test**

Use three particles and the `z=0` plane:

```python
def test_force_hessian_and_diagonal(self):
    contact = newton.solvers.ConstraintStaticPlaneContact(
        normal=(0.0, 0.0, 1.0),
        offset=0.0,
        thickness=0.1,
        stiffness=10.0,
        normal_damping=2.0,
        friction=0.5,
        friction_epsilon=0.1,
        particle_count=3,
        device=self.device,
    )
    q = wp.array([(0.0, 0.0, 0.05), (0.0, 0.0, -0.02), (0.0, 0.0, 0.2)], dtype=wp.vec3, device=self.device)
    v = wp.array([(1.0, 0.0, -2.0), (0.0, 0.0, 0.0), (5.0, 0.0, -5.0)], dtype=wp.vec3, device=self.device)
    direction = wp.array([(2.0, 3.0, 4.0)] * 3, dtype=wp.vec3, device=self.device)
    force = wp.zeros_like(q)
    product = wp.zeros_like(q)
    diagonal = wp.zeros(3, dtype=wp.mat33, device=self.device)

    contact.begin_step(q, v, 0.1)
    contact.prepare(q)
    contact.accumulate_force(q, force)
    contact.hessian_multiply(q, direction, product)
    contact.accumulate_diagonal(q, diagonal)

    np.testing.assert_allclose(force.numpy(), [[-0.25, 0.0, 4.5], [0.0, 0.0, 1.2], [0.0, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(product.numpy(), [[5.0, 7.5, 120.0], [24.0, 36.0, 40.0], [0.0, 0.0, 0.0]], atol=1e-5)
    np.testing.assert_allclose(
        diagonal.numpy(),
        [np.diag([2.5, 2.5, 30.0]), np.diag([12.0, 12.0, 10.0]), np.zeros((3, 3))],
        atol=1e-5,
    )
```

The expected values follow these exact branches:

- Particle 0: `depth=.05`, normal penalty `+.5z`, damping `+4z`, `u_t=.1x`, `alpha=.5*10*.05*10=2.5`.
- Particle 1: `depth=.12`, normal penalty `+1.2z`, zero damping, smoothed static `f1=20`, `alpha=.5*10*.12*20=12`.
- Particle 2: signed distance above the shell, so force and Hessian are zero despite nonzero velocity.

- [ ] **Step 2: Write validation and lifecycle misuse tests**

Add table-driven assertions that the constructor rejects zero normals, non-finite values, `thickness <= 0`, `stiffness <= 0`, negative damping/friction, `friction_epsilon <= 0`, and `particle_count <= 0`. Add tests that `begin_step` rejects `dt <= 0`, wrong array lengths/devices, and that `prepare` before `begin_step` raises `RuntimeError`.

Also extend `test_public_exports` with:

```python
from newton._src.solvers.limx.constraints.static_plane_contact import ConstraintStaticPlaneContact

self.assertIs(newton.solvers.ConstraintStaticPlaneContact, ConstraintStaticPlaneContact)
```

- [ ] **Step 3: Run contact tests to verify the new class is missing**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx_static_contact.py -k TestConstraintStaticPlaneContact
```

Expected: FAIL because `newton.solvers.ConstraintStaticPlaneContact` is not defined.

- [ ] **Step 4: Implement contact preparation without atomics**

Create one thread per particle. Normalize `normal` on the host and store it as `wp.vec3`; allocate `forces: wp.array[wp.vec3]` and `hessians: wp.array[wp.mat33]`. `begin_step` retains the velocity array and timestep. `prepare` launches this kernel:

```python
@wp.kernel
def _prepare_static_plane_contact(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    normal: wp.vec3,
    offset: float,
    thickness: float,
    stiffness: float,
    normal_damping: float,
    friction: float,
    friction_epsilon: float,
    dt: float,
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
):
    particle = wp.tid()
    distance = wp.dot(normal, positions[particle]) - offset
    if distance >= thickness:
        forces[particle] = wp.vec3(0.0)
        hessians[particle] = wp.mat33(0.0)
        return

    depth = thickness - distance
    normal_outer = wp.outer(normal, normal)
    force = stiffness * depth * normal
    hessian = stiffness * normal_outer

    velocity = velocities[particle]
    normal_velocity = wp.dot(velocity, normal)
    if normal_velocity < 0.0 and normal_damping > 0.0:
        force -= normal_damping * normal_velocity * normal
        hessian += normal_damping / dt * normal_outer

    tangent = wp.identity(3, float) - normal_outer
    tangent_displacement = dt * (velocity - normal_velocity * normal)
    tangent_length = wp.length(tangent_displacement)
    if tangent_length > friction_epsilon:
        friction_over_length = 1.0 / tangent_length
    else:
        friction_over_length = (-tangent_length / friction_epsilon + 2.0) / friction_epsilon
    alpha = friction * stiffness * depth * friction_over_length
    force -= alpha * tangent_displacement
    hessian += alpha * tangent

    forces[particle] = force
    hessians[particle] = hessian
```

Do not derive tangential velocity from the Newton iterate; the cached previous-frame velocity is the intentional lagged quantity.

- [ ] **Step 5: Implement matrix-free accumulation kernels**

Use direct indexed writes/adds, never atomics:

```python
@wp.kernel
def _accumulate_contact_force(cached: wp.array[wp.vec3], output: wp.array[wp.vec3]):
    particle = wp.tid()
    output[particle] += cached[particle]


@wp.kernel
def _multiply_contact_hessian(
    cached: wp.array[wp.mat33],
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    particle = wp.tid()
    output[particle] += cached[particle] * vector[particle]


@wp.kernel
def _accumulate_contact_diagonal(cached: wp.array[wp.mat33], output: wp.array[wp.mat33]):
    particle = wp.tid()
    output[particle] += cached[particle]
```

Every public method validates particle count and device before launching. The Hessian is PSD because its only terms are nonnegative multiples of `nn^T` and `I-nn^T`.

- [ ] **Step 6: Complete public exports and regenerate API documentation**

Export `ConstraintStaticPlaneContact` beside `ConstraintGroupDynamic` through the three solver `__init__` layers. Run:

```bash
uv run docs/generate_api.py
```

Confirm `docs/api/newton_solvers.rst` contains both new public classes and contains no `newton._src` references.

- [ ] **Step 7: Run exact contact, composition, and export tests**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx_static_contact.py -k TestConstraintStaticPlaneContact
uv run --extra dev -m newton.tests -p test_solver_limx_static_contact.py -k TestConstraintGroupDynamic
```

Expected: all selected tests pass on `cuda:0` and the exact numeric arrays match within the stated tolerances.

- [ ] **Step 8: Commit the static contact operator**

```bash
git add newton/_src/solvers/limx/constraints/static_plane_contact.py \
  newton/_src/solvers/limx/constraints/__init__.py \
  newton/_src/solvers/limx/__init__.py \
  newton/_src/solvers/__init__.py \
  newton/tests/test_solver_limx_static_contact.py \
  docs/api/newton_solvers.rst
git commit -m "Add LIMX static plane contact"
```

### Task 3: CUDA T-shirt-on-table example and settling regression

**Files:**
- Create: `newton/examples/cloth/example_cloth_limx_tshirt_table.py`
- Create: `newton/tests/test_example_cloth_limx_tshirt_table.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `ConstraintGroupDynamic`, `ConstraintStaticPlaneContact`, `ConstraintSelfCollision`, `ConstraintTriangleElastic`, `ConstraintDihedralBending`, and `SolverLIMX` from the public `newton.solvers` module.
- Produces: the discoverable command `uv run -m newton.examples cloth_limx_tshirt_table`; `Example.step()`, `Example.render()`, `Example.test_post_step()`, and `Example.test_final()`.

- [ ] **Step 1: Write a CUDA smoke test for scene construction and one captured step**

Import the example module through `importlib`, instantiate it on `cuda:0` with a lightweight null viewer accepted by the existing example conventions, assert its fixed configuration, run one captured step, and validate finite state:

```python
@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestClothLimxTshirtTable(unittest.TestCase):
    def test_cuda_graph_step_is_finite(self):
        example = load_example(device="cuda:0")
        self.assertEqual(example.frame_dt, 0.01)
        self.assertEqual(example.sim_substeps, 1)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertEqual(example.model.particle_count, 6436)
        self.assertEqual(example.model.tri_count, 12736)
        example.step()
        self.assertTrue(np.isfinite(example.state_0.particle_q.numpy()).all())
        self.assertTrue(np.isfinite(example.state_0.particle_qd.numpy()).all())
```

The helper must use the same null-viewer or example-test loader already used by nearby Newton example tests; do not introduce a new testing framework.

- [ ] **Step 2: Run the smoke test and observe the missing example failure**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_tshirt_table.py -k cuda_graph_step_is_finite
```

Expected: FAIL because `example_cloth_limx_tshirt_table.py` does not exist.

- [ ] **Step 3: Build the garment mesh and area-weighted particle masses**

Load `unisex_shirt.usd` at prim `/root/shirt` through public `newton.examples.get_asset` and `newton.usd.get_mesh`. Apply a `0.01` coordinate scale, subtract the local bounding-box center, apply fixed mild `x`/`y` rotation matrices using NumPy, and translate the cloth center to approximately `(0, 0, 1.05) m`.

Build the triangle mesh first, then distribute areal mass from rest areas:

```python
triangles = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
triangle_vertices = rest_positions[triangles]
triangle_areas = 0.5 * np.linalg.norm(
    np.cross(triangle_vertices[:, 1] - triangle_vertices[:, 0], triangle_vertices[:, 2] - triangle_vertices[:, 0]),
    axis=1,
)
masses = np.zeros(len(rest_positions), dtype=np.float32)
for corner in range(3):
    np.add.at(masses, triangles[:, corner], 0.3 * triangle_areas / 3.0)
if not np.isfinite(masses).all() or np.any(masses <= 0.0):
    raise ValueError("T-shirt mesh must give every particle a finite positive area-weighted mass")
```

Create every cloth particle as active and unanchored. Initialize its velocity with a mild linear throw plus `cross(angular_velocity, position - center)`.

- [ ] **Step 4: Build the table, elastic constraints, and composed contacts**

Add a static rendered box of size `1.1 × 0.9 × 0.1 m` centered at `z=.60 m`; its collision plane is `normal=(0,0,1)`, `offset=.65`. Configure:

```python
membrane = newton.solvers.ConstraintTriangleElastic(
    triangle_indices=triangles,
    inverse_rest_matrices=model.tri_poses.numpy(),
    rest_areas=model.tri_areas.numpy(),
    stiffnesses=[wp.vec3(1.0e4, 1.0e4, 1.0e3)] * len(triangles),
    particle_count=model.particle_count,
    device=model.device,
)
bending = newton.solvers.ConstraintDihedralBending(
    dihedral_indices=dihedral_indices,
    rest_positions=rest_positions,
    stiffness=0.01,
    particle_count=model.particle_count,
    device=model.device,
)
self_collision = newton.solvers.ConstraintSelfCollision(
    model,
    thickness=0.006,
    stiffness=1.0e4,
    untangle_stiffness=3.0e4,
    max_contacts=131072,
)
table_contact = newton.solvers.ConstraintStaticPlaneContact(
    normal=(0.0, 0.0, 1.0),
    offset=0.65,
    thickness=0.006,
    stiffness=2.0e4,
    normal_damping=0.5,
    friction=0.4,
    friction_epsilon=1.0e-4,
    particle_count=model.particle_count,
    device=model.device,
)
dynamic = newton.solvers.ConstraintGroupDynamic([self_collision, table_contact])
solver = newton.solvers.SolverLIMX(
    model,
    constraints=[membrane, bending],
    nonlinear_iterations=1,
    linear_iterations=50,
    velocity_damping=1.0,
    dynamic_operator=dynamic,
)
```

- [ ] **Step 5: Add CUDA graph stepping, rendering, and invariant checks**

Follow `example_cloth_limx_twist.py` for state swapping, graph capture, viewer controls, and cloth rendering. Use one solver step for each `frame_dt=.01`. `test_post_step()` rejects non-finite positions/velocities and catastrophic plane penetration below `z=.62`; `test_final()` repeats those checks and verifies the scene has advanced.

- [ ] **Step 6: Run the one-step CUDA smoke test**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_tshirt_table.py -k cuda_graph_step_is_finite
```

Expected: PASS on `cuda:0`, including graph capture and one graph launch.

- [ ] **Step 7: Add an extended headless settling regression**

Advance the example long enough to include impact and a settling window. For the final 50 frames compute the mean particle speed per frame and the minimum signed gap `min(z - .65)`. Assert:

```python
self.assertGreaterEqual(minimum_gap, -0.008)
self.assertLess(float(np.mean(final_window_speeds)), 0.02)
```

The rollout must remain on `cuda:0`; reduce host transfers to the final-window diagnostics rather than copying the full state every frame.

- [ ] **Step 8: Run and tune only through the dedicated CUDA regression**

Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_tshirt_table.py -k settles_on_table
```

Expected: PASS with maximum table penetration inside the 8 mm tolerance and final-window mean speed below `0.02 m/s`. If tuning is needed, change only the example's initial pose/velocity or the approved contact damping/friction values; keep global velocity damping at exactly `1.0`.

- [ ] **Step 9: Record the user-facing feature and commit the runnable scene**

Insert an imperative `[Unreleased]` `Added` entry at a non-terminal position in that category:

```markdown
- Add a LIMX T-shirt table-contact example with self-collision, contact damping, and friction.
```

Then commit:

```bash
git add newton/examples/cloth/example_cloth_limx_tshirt_table.py \
  newton/tests/test_example_cloth_limx_tshirt_table.py \
  CHANGELOG.md
git commit -m "Add LIMX T-shirt table example"
```

### Task 4: Focused verification and interactive visual handoff

**Files:**
- Create after visual approval: `newton/examples/cloth/example_cloth_limx_tshirt_table.jpg`
- Modify after visual approval: `newton/examples/README.md`

**Interfaces:**
- Consumes: the discoverable example from Task 3.
- Produces: a live OpenGL window with the existing play/pause/reset controls, plus the repository-required 320×320 example screenshot and README registration after the motion is accepted.

- [ ] **Step 1: Run all new CUDA-only tests together**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx_static_contact.py
uv run --extra dev -m newton.tests -p test_example_cloth_limx_tshirt_table.py
uv run --extra dev -m newton.tests -p test_solver_limx.py -k dynamic_contacts_prepare_once_before_each_newton_linearization
```

Expected: every selected test passes on CUDA; no CPU LIMX kernels are compiled by these commands.

- [ ] **Step 2: Run lint and formatting verification**

Run:

```bash
uvx pre-commit run -a
```

Expected: all hooks pass. If a formatter modifies files, stage only feature files, rerun the focused CUDA tests affected by those edits, and commit the formatting with the owning task.

- [ ] **Step 3: Launch the interactive example and verify controls before handoff**

Run in a persistent terminal session:

```bash
uv run -m newton.examples cloth_limx_tshirt_table --device cuda:0
```

Inspect startup output for viewer backend warnings. Hand the window to the user only after the render loop remains alive and the GUI play/pause/reset buttons are visibly available.

- [ ] **Step 4: Capture and register the approved example**

After the user accepts the dynamics, capture a representative frame, crop/resize it to exactly 320×320 as `newton/examples/cloth/example_cloth_limx_tshirt_table.jpg`, and add a cloth-section README row using:

```markdown
| [T-shirt Table (LIMX)](cloth/example_cloth_limx_tshirt_table.py) | `python -m newton.examples cloth_limx_tshirt_table` | ![](cloth/example_cloth_limx_tshirt_table.jpg) |
```

Verify the JPEG dimensions with `file` or ImageMagick identify, then commit only those two files:

```bash
git add newton/examples/README.md newton/examples/cloth/example_cloth_limx_tshirt_table.jpg
git commit -m "Document LIMX T-shirt table example"
```

- [ ] **Step 5: Confirm the branch contains only intended feature commits**

Run:

```bash
git status --short
git log --oneline --decorate -8
```

Expected: `lessons.md` and `solver_convergence.png` remain unstaged user workspace changes; feature code, tests, docs, and screenshot are committed on `vegtsunami/limx-dihedral-bending`.
