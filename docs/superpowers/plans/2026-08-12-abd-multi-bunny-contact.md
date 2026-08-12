# ABD Multi-Bunny Contact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simulate eight frictional affine bunnies with true cross-body VF/strict-EE surface contact in one Newton/PCG solve.

**Architecture:** Repeat one tetrahedral rest mesh into a multi-body `AffineBodyModel`, reconstruct all surfaces on the GPU, and detect cross-body VF/EE contacts with refittable BVHs. Lift frozen penalty, damping, friction, and mollified-EE contributions through material-point Jacobians into native 12-by-12 affine blocks while keeping off-diagonal coupling matrix-free.

**Tech Stack:** Python, Warp GPU kernels/BVH, NumPy dense references, Newton LIMX `vec12`/`mat1212`, `unittest`, CUDA graph capture.

## Global Constraints

- Work only in the canonical checkout on `dev`; do not create a worktree.
- Preserve the user's unrelated modified image, modified T-shirt test, and `solver_convergence.png`; never stage them.
- Use penalty contact, not an IPC barrier, line search, SDF, proxy spheres, or CCD.
- Use unsigned triangle closest point for VF so VF owns PE/PP boundary regions.
- Accept EE only for strict interior-interior parameters and never generate EE endpoint PE/PP.
- Retain the IPC-style near-parallel EE mollifier for accepted EE pairs.
- Keep particle matrices in native 3-by-3 blocks and affine bodies in native 12-by-12 blocks.
- Keep complete cross-body contact coupling matrix-free; add only exact 12-by-12 body diagonals to block Jacobi.
- Use exactly one Newton iteration and 50 PCG iterations per 0.01 s example frame.
- Use `unittest`, give every test a triple-double-quoted imperative docstring, and run commands through `uv`.
- Do not add dependencies or import `newton._src` from examples or documentation.
- Add public symbols through `newton.solvers` and run `docs/generate_api.py`.
- Add the user-facing feature to a random position in `CHANGELOG.md`'s `[Unreleased]` `Added` category.
- Use imperative commit subjects around 50 characters and stage only files named by the current task.

## File Map

- Modify `newton/_src/solvers/limx/affine_body.py`: construct repeated affine instances while preserving the one-body API.
- Create `newton/_src/solvers/limx/constraints/affine_dynamic_group.py`: compose affine mixed dynamic constraints.
- Create `newton/_src/solvers/limx/constraints/affine_body_contact.py`: own affine surface topology, BVHs, contact buffers, detection, forces, HVPs, diagonals, damping, friction, and EE mollification.
- Modify `newton/_src/solvers/limx/constraints/__init__.py`, `newton/_src/solvers/limx/__init__.py`, and `newton/_src/solvers/__init__.py`: expose the new public types.
- Modify `newton/tests/test_solver_limx_affine.py`: test multi-instance mass, topology, state, reconstruction, and validation.
- Create `newton/tests/test_solver_limx_affine_body_contact.py`: test contact generation and dense affine operator references.
- Create `newton/tests/test_solver_limx_affine_dynamic_group.py`: test composition and validation.
- Create `newton/examples/basic/example_basic_limx_affine_bunnies_ground.py`: render and validate the eight-bunny pile.
- Create `newton/tests/test_example_basic_limx_affine_bunnies_ground.py`: run the focused CUDA scene regression.
- Modify `README.md`, `CHANGELOG.md`, and generated API documentation: register user-facing behavior.
- Create `docs/images/examples/example_basic_limx_affine_bunnies_ground.jpg`: add the required 320-by-320 example screenshot.

---

### Task 1: Construct Multiple Instances in `AffineBodyModel`

**Files:**
- Modify: `newton/_src/solvers/limx/affine_body.py`
- Test: `newton/tests/test_solver_limx_affine.py`

**Interfaces:**
- Consumes: existing one-body `AffineBodyModel(...)` constructor and `_initial_affine_state(transform, rest_centroid)`.
- Produces: `AffineBodyModel.from_instances(rest_vertices, tetrahedron_indices, surface_triangle_indices, density, rigidity, initial_transforms, device) -> AffineBodyModel`.
- Produces: globally indexed repeated `rest_vertices`, `tetrahedron_indices`, `surface_vertex_indices`, `rest_surface_vertices`, `surface_triangle_indices`, and per-surface-vertex `surface_ownership`.

- [ ] **Step 1: Write failing multi-instance state and topology tests**

Add tests beside `TestAffineBodyModel` using `_unit_tetrahedron()`:

```python
def test_builds_repeated_affine_instances(self):
    """Build independent affine states over one repeated rest mesh."""
    vertices, tetrahedra, surface_triangles = self._unit_tetrahedron()
    transforms = [
        wp.transform_identity(),
        wp.transform(
            wp.vec3(2.0, -1.0, 0.5),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi),
        ),
    ]
    model = AffineBodyModel.from_instances(
        vertices,
        tetrahedra,
        surface_triangles,
        density=6.0,
        rigidity=2.5,
        initial_transforms=transforms,
        device="cpu",
    )

    self.assertEqual(model.body_count, 2)
    self.assertEqual(model.surface_vertex_count, 8)
    self.assertEqual(model.surface_triangle_count, 8)
    np.testing.assert_array_equal(model.surface_ownership.numpy(), [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(model.tetrahedron_indices.numpy(), [[0, 1, 2, 3], [4, 5, 6, 7]])
    np.testing.assert_array_equal(
        model.surface_triangle_indices.numpy()[4:],
        surface_triangles + 4,
    )
    np.testing.assert_allclose(model.volumes.numpy(), [1.0 / 6.0, 1.0 / 6.0])
    np.testing.assert_allclose(model.rigidities.numpy(), [2.5, 2.5])
    np.testing.assert_allclose(model.mass_matrices.numpy()[0], model.mass_matrices.numpy()[1])

    output = wp.empty(8, dtype=wp.vec3, device="cpu")
    model.update_surface_positions(model.q, output)
    np.testing.assert_allclose(output.numpy()[:4], vertices, atol=1.0e-6)
    expected_second = np.column_stack((-vertices[:, 1], vertices[:, 0], vertices[:, 2]))
    expected_second += [2.0, -1.0, 0.5]
    np.testing.assert_allclose(output.numpy()[4:], expected_second, atol=1.0e-6)
```

Add a validation test:

```python
def test_rejects_invalid_affine_instance_transforms(self):
    """Reject empty, malformed, and non-rigid instance transforms."""
    vertices, tetrahedra, surface_triangles = self._unit_tetrahedron()
    arguments = dict(
        rest_vertices=vertices,
        tetrahedron_indices=tetrahedra,
        surface_triangle_indices=surface_triangles,
        density=1.0,
        rigidity=0.0,
        device="cpu",
    )
    with self.assertRaisesRegex(ValueError, "initial_transforms"):
        AffineBodyModel.from_instances(**arguments, initial_transforms=[])
    with self.assertRaisesRegex(ValueError, "initial_transform"):
        AffineBodyModel.from_instances(**arguments, initial_transforms=[np.zeros(6)])
    with self.assertRaisesRegex(ValueError, "unit quaternion"):
        AffineBodyModel.from_instances(
            **arguments,
            initial_transforms=[wp.transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0)],
        )
```

- [ ] **Step 2: Run the focused tests and confirm the red state**

Run:

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine"
```

Expected: both tests fail because `AffineBodyModel.from_instances` does not exist.

- [ ] **Step 3: Refactor common initialization and implement repetition**

Keep `__init__` as the one-transform entry point and add:

```python
@classmethod
def from_instances(
    cls,
    rest_vertices: Any,
    tetrahedron_indices: Any,
    surface_triangle_indices: Any,
    density: float,
    rigidity: float,
    initial_transforms: Any,
    device: Any,
) -> AffineBodyModel:
    """Create affine instances that share one tetrahedral rest mesh."""
    transforms = tuple(initial_transforms)
    if not transforms:
        raise ValueError("initial_transforms must not be empty")
    model = cls.__new__(cls)
    model._initialize(
        rest_vertices,
        tetrahedron_indices,
        surface_triangle_indices,
        density,
        rigidity,
        transforms,
        device,
    )
    return model
```

Move the constructor body into `_initialize(...)`; pass `(initial_transform,)`
from `__init__`. Integrate and center the mesh once, validate every transform
by calling `_initial_affine_state`, then form repeated arrays with explicit
offsets:

```python
body_count = len(initial_states)
vertex_count = len(centered_vertices)
surface_vertex_count_per_body = len(rest_surface_vertices)

repeated_vertices = np.tile(centered_vertices, (body_count, 1))
repeated_tetrahedra = np.concatenate(
    [tetrahedra + body * vertex_count for body in range(body_count)]
)
repeated_surface_vertices = np.tile(rest_surface_vertices, (body_count, 1))
repeated_surface_triangles = np.concatenate(
    [compact_surface_triangles + body * surface_vertex_count_per_body for body in range(body_count)]
)
repeated_surface_vertex_indices = np.concatenate(
    [surface_vertex_indices + body * vertex_count for body in range(body_count)]
)
surface_ownership = np.repeat(
    np.arange(body_count, dtype=np.int32),
    surface_vertex_count_per_body,
)
```

Repeat scalar/body arrays with `np.repeat(..., axis=0)` and initialize `q`
from the stacked states and `qd` with `body_count` rows. Update the validation
message in `update_surface_positions()` to say `vec12 states` when plural.

- [ ] **Step 4: Run the complete affine-model and affine-solver tests**

Run:

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine"
```

Expected: all affine tests pass on available CPU/CUDA devices, including the
unchanged one-body constructor tests.

- [ ] **Step 5: Commit the multi-instance model**

```bash
git add newton/_src/solvers/limx/affine_body.py newton/tests/test_solver_limx_affine.py
git commit -m "Add multi-instance affine bodies"
```

---

### Task 2: Compose Affine Dynamic Constraints

**Files:**
- Create: `newton/_src/solvers/limx/constraints/affine_dynamic_group.py`
- Create: `newton/tests/test_solver_limx_affine_dynamic_group.py`
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`

**Interfaces:**
- Consumes: the mixed dynamic operator lifecycle already called by `SolverLIMXAffine`.
- Produces: public `ConstraintGroupAffine(constraints: Sequence[Any])` with `body_count`, `device`, and ordered lifecycle forwarding.

- [ ] **Step 1: Write failing ordered-forwarding and validation tests**

Create a `unittest.TestCase` with a recording child whose methods exactly
match the affine mixed interface:

```python
def test_forwards_affine_constraint_lifecycle_in_order(self):
    """Forward every affine dynamic operation in declaration order."""
    events = []

    class RecordingConstraint:
        body_count = 2
        device = wp.get_device("cpu")

        def __init__(self, name):
            self.name = name

        def begin_step(self, q, qd, dt):
            events.append((self.name, "begin", dt))

        def prepare(self, q):
            events.append((self.name, "prepare"))

        def accumulate_force(self, q, output):
            events.append((self.name, "force"))

        def multiply(self, particle_input, affine_input, particle_output, affine_output):
            events.append((self.name, "multiply"))

        def accumulate_diagonal(self, particle_diagonal, affine_diagonal):
            events.append((self.name, "diagonal"))

    group = newton.solvers.ConstraintGroupAffine(
        [RecordingConstraint("body"), RecordingConstraint("ground")]
    )
    q = wp.zeros(2, dtype=vec12, device="cpu")
    empty = wp.empty(0, dtype=wp.vec3, device="cpu")
    group.begin_step(q, q, 0.01)
    group.prepare(q)
    group.accumulate_force(q, q)
    group.multiply(empty, q, empty, q)
    group.accumulate_diagonal(
        wp.empty(0, dtype=wp.mat33, device="cpu"),
        wp.zeros(2, dtype=mat1212, device="cpu"),
    )
    self.assertEqual(group.body_count, 2)
    self.assertEqual(
        events,
        [
            ("body", "begin", 0.01), ("ground", "begin", 0.01),
            ("body", "prepare"), ("ground", "prepare"),
            ("body", "force"), ("ground", "force"),
            ("body", "multiply"), ("ground", "multiply"),
            ("body", "diagonal"), ("ground", "diagonal"),
        ],
    )
```

Also assert that an empty list, mismatched `body_count`, and mismatched device
raise `ValueError` mentioning the rejected field.

- [ ] **Step 2: Run the group test and confirm it fails at the public symbol**

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine_dynamic_group"
```

Expected: FAIL because `newton.solvers.ConstraintGroupAffine` is absent.

- [ ] **Step 3: Implement the small forwarding class and public exports**

Mirror `ConstraintGroupDynamic`, but use the affine signatures shown in the
test. The constructor must store `tuple(constraints)`, reject an empty tuple,
normalize the first device with `wp.get_device`, and validate every child.
Each lifecycle method is an ordered loop with no allocation and no fallback
signature detection.

Add `ConstraintGroupAffine` to all three `__init__.py` import lists,
`__all__`, `TYPE_CHECKING`, and `_LAZY_IMPORTS` entries.

- [ ] **Step 4: Run the group and existing affine-contact tests**

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine"
```

Expected: both modules pass.

- [ ] **Step 5: Commit the affine constraint group**

```bash
git add newton/_src/solvers/limx/constraints/affine_dynamic_group.py \
  newton/_src/solvers/limx/constraints/__init__.py \
  newton/_src/solvers/limx/__init__.py newton/_src/solvers/__init__.py \
  newton/tests/test_solver_limx_affine_dynamic_group.py
git commit -m "Compose affine dynamic constraints"
```

---

### Task 3: Detect Cross-Body VF and Strict EE Contacts

**Files:**
- Create: `newton/_src/solvers/limx/constraints/affine_body_contact.py`
- Create: `newton/tests/test_solver_limx_affine_body_contact.py`

**Interfaces:**
- Consumes: `AffineBodyModel` concatenated rest surface, ownership, triangles, and `update_surface_positions()`.
- Produces: `ConstraintAffineBodyContact(body_model, thickness, stiffness, normal_damping, friction, friction_epsilon, max_contacts=262144)`.
- Produces: `vertex_face_contacts` and `edge_edge_contacts`, each exposing `ids`, `weights`, `directions`, `depths`, `count`, `overflow_count`, and `capacity`; EE also exposes `mollifier_thresholds` and `mollifier_active`.

- [ ] **Step 1: Write failing constructor and contact-generation tests**

Build two unit-tetrahedron instances and add helpers that copy active buffer
rows after `prepare()`. Cover these exact cases with separate tests and
imperative docstrings:

```python
from newton._src.solvers.limx.constraints.affine_body_contact import ConstraintAffineBodyContact


def _active_rows(buffer):
    count = min(int(buffer.count.numpy()[0]), buffer.capacity)
    return (
        buffer.ids.numpy()[:count],
        buffer.weights.numpy()[:count],
        buffer.depths.numpy()[:count],
    )
```

- same-body contacts are absent when the two bodies are far apart;
- translating body 1 so one vertex is 2 mm from body 0's face produces a
  cross-body VF row in at least one direction;
- positioning the closest point exactly on a target face edge and target
  vertex still produces VF with a zero or unit barycentric component;
- skew edges with closest parameters `0.25` and `0.75` produce EE;
- moving either closest parameter to exactly `0` or `1` removes that EE key;
- `max_contacts=1` increments the relevant overflow counter in a deliberately
  dense placement;
- zero/NaN parameters, negative damping/friction, empty/single-body models,
  and nonpositive capacity raise a specific `TypeError` or `ValueError`.

Use canonical keys `(vertex, sorted(face_vertices))` and
`(sorted(edge_0), sorted(edge_1))` in assertions so BVH traversal order does
not affect the test.

- [ ] **Step 2: Run the contact module and confirm the constructor is missing**

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine_body_contact"
```

Expected: FAIL importing or constructing `ConstraintAffineBodyContact`.

- [ ] **Step 3: Implement buffers, topology ownership, and BVHs**

Create focused private buffer classes. Allocate IDs as two-dimensional Warp
arrays with arity four, weights as `float`, directions as `wp.vec3`, depths as
`float`, and scalar count/overflow arrays. EE additionally allocates
`mollifier_thresholds` and `mollifier_active`.

In the constraint constructor:

```python
triangles = np.asarray(body_model.surface_triangle_indices.numpy(), dtype=np.int32)
ownership = np.asarray(body_model.surface_ownership.numpy(), dtype=np.int32)
triangle_ownership = ownership[triangles[:, 0]]
if not np.all(ownership[triangles] == triangle_ownership[:, None]):
    raise ValueError("Every surface triangle must belong to one affine body")

adjacency = MeshAdjacency(triangles)
edges = adjacency.edge_indices
edge_ownership = ownership[edges[:, 2]]
if not np.all(ownership[edges[:, 2:4]] == edge_ownership[:, None]):
    raise ValueError("Every surface edge must belong to one affine body")
```

Allocate world positions and triangle/edge bounds, launch bound-update
kernels, and construct `wp.Bvh` objects once. `prepare(q)` must reconstruct
positions, refit both BVHs, clear both buffers, and launch both detectors.

- [ ] **Step 4: Implement unsigned clamped VF and strict-interior EE detection**

VF must use `triangle_closest_point_barycentric`, reject equal body owners,
compute `separation = vertex - closest`, and accept only
`_MIN_CONTACT_DISTANCE < distance < thickness`. Store weights
`[1, -b0, -b1, -b2]`, direction `separation / distance`, and depth
`thickness - distance`.

EE must query each unordered pair once, reject equal body owners, compute
`wp.closest_point_edge_edge(..., 1.0e-5)`, and apply the exact gate:

```python
if parameter_0 <= 0.0 or parameter_0 >= 1.0:
    continue
if parameter_1 <= 0.0 or parameter_1 >= 1.0:
    continue
```

Store weights `[1-s, s, -(1-t), -t]`, normalized closest-point separation,
and `thickness - distance`. Set the mollifier threshold for every retained
cross-body EE pair to
`1.0e-3 * dot(rest_edge_0, rest_edge_0) * dot(rest_edge_1, rest_edge_1)`.
Do not add EV, VV, PE, PP, oriented projected VF, or EF kernels.

- [ ] **Step 5: Run generation tests on CPU and CUDA**

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine_body_contact"
```

Expected: constructor, validation, boundary VF, strict EE, same-body filter,
and overflow tests pass on every available device.

- [ ] **Step 6: Commit frozen affine contact detection**

```bash
git add newton/_src/solvers/limx/constraints/affine_body_contact.py \
  newton/tests/test_solver_limx_affine_body_contact.py
git commit -m "Detect affine body surface contacts"
```

---

### Task 4: Lift Ordinary Contact, Damping, and Friction

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/affine_body_contact.py`
- Modify: `newton/tests/test_solver_limx_affine_body_contact.py`

**Interfaces:**
- Consumes: frozen four-point VF/EE buffers from Task 3 and affine mixed operator calls.
- Produces: `begin_step`, `prepare`, `accumulate_force`, `multiply`, and `accumulate_diagonal` for ordinary VF and non-mollified EE.

- [ ] **Step 1: Write dense-reference tests for a single two-body contact**

For a selected frozen row, build each local point Jacobian with:

```python
def _point_jacobian(rest_position):
    x, y, z = rest_position
    return np.asarray([
        [1.0, 0.0, 0.0, x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, x, y, z, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x, y, z],
    ])
```

Accumulate `G[owner] += weight * J(rest)` and form dense references:

```python
relative_velocity = sum(G[b] @ qd[b] for b in involved_bodies)
normal_velocity = direction @ relative_velocity
tangent = np.eye(3) - np.outer(direction, direction)
tangent_displacement = dt * tangent @ relative_velocity
inverse_length = (
    1.0 / np.linalg.norm(tangent_displacement)
    if np.linalg.norm(tangent_displacement) > epsilon
    else (2.0 - np.linalg.norm(tangent_displacement) / epsilon) / epsilon
)
alpha = friction * stiffness * depth * inverse_length
world_force = stiffness * depth * direction
if normal_velocity < 0.0:
    world_force -= normal_damping * normal_velocity * direction
world_force -= alpha * tangent_displacement
world_hessian = stiffness * np.outer(direction, direction) + alpha * tangent
if normal_velocity < 0.0:
    world_hessian += normal_damping / dt * np.outer(direction, direction)
```

Assert generalized force `G[b].T @ world_force`, full 24-by-24 HVP,
and both exact diagonal blocks `G[b].T @ world_hessian @ G[b]`. Also assert
translation force sums to zero, separating velocity removes damping,
small-slip friction is finite/opposing, and the dense Hessian is symmetric
PSD within `1.0e-5`.

- [ ] **Step 2: Run the operator tests and confirm zero/missing contributions**

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine_body_contact"
```

Expected: FAIL because Task 3 only detects contacts.

- [ ] **Step 3: Implement step state and ordinary world response**

Cache `qd` and `dt` in `begin_step` with the same validation/lifecycle rules
as `ConstraintAffineStaticPlaneContact`. During force/Hessian evaluation,
compute relative world velocity from all four weighted material points.
Freeze normal damping and the regularized friction scalar at the current
contact depth and step-start velocity.

Use shared Warp helpers in the new module for affine material points and
`J.T @ world_vector`. Keep per-contact data frozen between `prepare()` and all
linear-operator calls.

- [ ] **Step 4: Implement complete matrix-free HVP and exact body diagonals**

For HVP, evaluate every local point motion `J(r_i) p_owner`, take the weighted
sum, multiply by the 3-by-3 world Hessian, and scatter
`weight_i * J(r_i).T @ product`.

For each of the two body owners, combine the local weighted Jacobians before
forming its diagonal. Do not sum four independent point diagonals because
that would omit cross-terms between face or edge endpoints owned by the same
body. A direct 12-by-12 loop over `G.T @ H @ G` is acceptable and must match
the dense test.

- [ ] **Step 5: Run the complete ordinary-contact references**

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine_body_contact"
```

Expected: ordinary VF/EE force, damping, friction, HVP, exact diagonal, PSD,
and lifecycle tests pass.

- [ ] **Step 6: Commit ordinary affine contact response**

```bash
git add newton/_src/solvers/limx/constraints/affine_body_contact.py \
  newton/tests/test_solver_limx_affine_body_contact.py
git commit -m "Lift affine body contact response"
```

---

### Task 5: Lift the EE Mollifier Exactly

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/affine_body_contact.py`
- Modify: `newton/tests/test_solver_limx_affine_body_contact.py`

**Interfaces:**
- Consumes: Task 3 EE threshold/active arrays and Task 4 affine scatter helpers.
- Produces: mollified EE force, full two-body Gauss-Newton HVP, exact 12-by-12 body diagonals, and mollifier-scaled friction load.

- [ ] **Step 1: Write a near-parallel dense/directional reference test**

Create a retained EE pair whose squared cross product lies below its stored
threshold. Assert `mollifier_active == 1`. Compute the generalized force and
24-by-24 HVP reference by applying the existing four-point world residual
Jacobian from `self_collision.py` to affine basis motions. Check every column
against `multiply()` and check each body diagonal against the corresponding
12-column slice. Require relative tolerance `3.0e-4` and absolute tolerance
`3.0e-5`, then perturb the pair out of the threshold and require
`mollifier_active == 0`.

- [ ] **Step 2: Run the mollifier reference and confirm the ordinary operator differs**

```bash
uv run --extra dev -m newton.tests -k "test_lifts_near_parallel_ee_mollifier"
```

Expected: FAIL because near-parallel EE still uses the ordinary rank-one
response or does not set `mollifier_active`.

- [ ] **Step 3: Reuse the authoritative EE residual functions**

Import the internal Warp functions
`_edge_edge_mollifier_is_active`,
`_edge_edge_mollified_residual_data`,
`_edge_edge_mollified_residual_jacobian_transpose_multiply`, and
`_edge_edge_gauss_newton_multiply` from `self_collision.py`; do not fork their
formulas. After detection, launch a preparation kernel that freezes
`mollifier_active` for every retained EE.

For force, reproduce the existing mollified four-point world gradients and
lift each through its owning body's material-point Jacobian. For HVP, first
map both bodies' generalized input to four world point motions, call
`_edge_edge_gauss_newton_multiply`, and lift all four products back.

- [ ] **Step 4: Accumulate exact mollified 12-by-12 body diagonals**

For each contact and body side, loop over the 12 generalized basis columns.
Generate the four point motions for that body's basis, call the same full
Gauss-Newton multiply, lift all outputs owned by that body, and write the
result as one block column. This explicitly retains endpoint cross-terms.
Multiply by contact stiffness and atomically add the complete block.

Scale the EE friction normal load by the existing mollifier value
`cross_squared * (2 * threshold - cross_squared) / threshold**2`, clamped to
`[0, 1]`, matching particle self-collision. Keep endpoint rejection unchanged.

- [ ] **Step 5: Run all self-collision and affine-contact tests**

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine_body_contact"
uv run --extra dev -m newton.tests -k "mollified"
uv run --extra dev -m newton.tests -k "edge_edge_detection_excludes_endpoint_features"
```

Expected: affine mollifier references and existing particle EE mollifier
regressions pass.

- [ ] **Step 6: Commit mollified affine EE**

```bash
git add newton/_src/solvers/limx/constraints/affine_body_contact.py \
  newton/tests/test_solver_limx_affine_body_contact.py
git commit -m "Lift mollified affine edge contact"
```

---

### Task 6: Export and Integrate Affine Body Contact

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/tests/test_solver_limx_affine_body_contact.py`
- Modify: `docs/api/newton_solvers.rst`

**Interfaces:**
- Consumes: completed internal `ConstraintAffineBodyContact`.
- Produces: `newton.solvers.ConstraintAffineBodyContact` and documented `AffineBodyModel.from_instances`.

- [ ] **Step 1: Add a failing public import and solver integration test**

Use only public symbols to build two bodies, create body and ground contact,
group them, and take one zero-gravity/no-contact solver step:

```python
body_contact_type = newton.solvers.ConstraintAffineBodyContact
group_type = newton.solvers.ConstraintGroupAffine
group = group_type([body_contact, ground_contact])
solver = newton.solvers.SolverLIMXAffine(
    model,
    nonlinear_iterations=1,
    linear_iterations=50,
    dynamic_operator=group,
)
solver.step(0.01)
self.assertEqual(solver.body_count, 2)
self.assertIs(solver.dynamic_operator, group)
self.assertTrue(np.isfinite(solver.q.numpy()).all())
```

- [ ] **Step 2: Run the public integration test and confirm the export is absent**

```bash
uv run --extra dev -m newton.tests -k "test_exports_and_steps_affine_body_contact"
```

Expected: FAIL resolving `newton.solvers.ConstraintAffineBodyContact`.

- [ ] **Step 3: Add lazy exports and public docstrings**

Add `ConstraintAffineBodyContact` to the constraint, LIMX, and top-level
solver import lists, `__all__`, `TYPE_CHECKING`, and `_LAZY_IMPORTS`. Ensure
public docstrings use Google-style `Args:`, bracket Warp array annotations,
SI units, and no `newton._src` references.

- [ ] **Step 4: Regenerate API documentation and run public tests**

```bash
uv run docs/generate_api.py
uv run --extra dev -m newton.tests -k "test_solver_limx_affine"
```

Expected: API generation exits zero and both public test modules pass.

- [ ] **Step 5: Commit public affine contact APIs**

Stage only the three export files, generated API files changed by the command,
and the affine body contact test, then commit:

```bash
git commit -m "Expose affine body surface contact"
```

---

### Task 7: Build and Validate the Eight-Bunny Scene

**Files:**
- Create: `newton/examples/basic/example_basic_limx_affine_bunnies_ground.py`
- Create: `newton/tests/test_example_basic_limx_affine_bunnies_ground.py`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Create: `docs/images/examples/example_basic_limx_affine_bunnies_ground.jpg`

**Interfaces:**
- Consumes: public multi-instance model, body contact, group, ground contact, and `SolverLIMXAffine`.
- Produces: `uv run -m newton.examples basic_limx_affine_bunnies_ground`.

- [ ] **Step 1: Write the failing configuration and 300-frame CUDA tests**

Create a CUDA-gated `unittest` module. The first test constructs the example
with `ViewerNull(num_frames=1)` and asserts:

```python
self.assertEqual(example.body_model.body_count, 8)
self.assertEqual(example.frame_dt, 0.01)
self.assertEqual(example.solver.nonlinear_iterations, 1)
self.assertEqual(example.solver.linear_iterations, 50)
self.assertEqual(example.body_contact.thickness, 0.003)
self.assertEqual(example.body_contact.stiffness, 2.0e4)
self.assertEqual(example.body_contact.normal_damping, 0.5)
self.assertEqual(example.body_contact.friction, 0.5)
self.assertEqual(example.ground_contact.friction, 0.5)
self.assertEqual(module.Example.create_parser().parse_args([]).num_frames, 300)
self.assertIsNotNone(example.graph)
```

The second test runs 300 steps, calls `test_post_step()` every frame and
`test_final()` once, then independently asserts positive minimum determinant,
singular-value error below `0.02`, ground height above `-0.006`, zero VF/EE
overflow, maximum contact depth below `0.012`, observed cross-body contact,
and final upper-layer support margin above `0.10` over the last 30 frames.

- [ ] **Step 2: Run the example tests and confirm the module is absent**

```bash
uv run --extra dev -m newton.tests -k "test_example_basic_limx_affine_bunnies_ground"
```

Expected: FAIL importing the example module.

- [ ] **Step 3: Implement the staggered 2-by-2-by-2 scene**

Use `bunny_tet.npz`, scale `0.15`, density `1000.0`, rigidity `1.0e8`, and
the source-to-world `+90 degree` X rotation. Define eight deterministic
center/tilt/velocity records. Keep lower centers near `z=0.25`, upper centers
near `z=0.55`, offset upper X/Y positions by at least `0.04 m` from every
lower column, and verify initial reconstructed AABBs do not overlap.

Set initial translations directly in `body_model.q[:, :3]` to the intended
centers and initial translation velocities in `body_model.qd[:, :3]` to
small opposing lateral values. Leave affine velocity rows zero.

Create body contact and the existing ground contact with the fixed parameters,
compose them with `ConstraintGroupAffine`, and create:

```python
self.solver = newton.solvers.SolverLIMXAffine(
    self.body_model,
    nonlinear_iterations=1,
    linear_iterations=50,
    velocity_damping=1.0,
    dynamic_operator=self.dynamic_constraints,
)
```

Build one render-only particle/triangle model from concatenated compact
surfaces, add a static ground box, frame the full pile, and capture
`solver.step()` plus surface reconstruction on CUDA. During `render()`, call
the public `viewer.log_mesh()` once per body's triangle range, using the
shared reconstructed position array, a body-specific index buffer, and a
distinct fixed RGB color.

- [ ] **Step 4: Implement measurable example acceptance state**

Track per frame:

- all `q`, `qd`, and surface positions finite;
- per-body `det(A)` and singular values;
- minimum surface Z;
- active VF/EE counts, overflows, and maximum stored depth;
- whether an active contact spans different owners;
- each body's center height and the final-30-frame upper/lower support margin.

Raise immediately for inversion, singular error `>=0.02`, Z below `-0.006`,
overflow, or contact depth `>=0.012`. In `test_final`, require all centers to
have fallen, cross-body contact to have occurred, and mean final support
margin `>0.10`.

- [ ] **Step 5: Tune only layout/contact capacity and rerun the CUDA rollout**

```bash
uv run --extra dev -m newton.tests -k "test_example_basic_limx_affine_bunnies_ground"
```

Expected: both configuration and 300-frame rollout tests pass. If the scene
fails, retain `1 Newton + 50 PCG`, 3 mm thickness, stiffness, rigidity, and
friction. Adjust only non-overlapping initial centers/tilts/velocities or
increase contact capacity until the specified physical test passes.

- [ ] **Step 6: Register, capture, and document the example**

Add the command and 320-by-320 image card to `README.md`. Run the example with
the repository's screenshot workflow, save the image at the exact path above,
and verify its dimensions. Insert an `Added` entry at a random position in
`CHANGELOG.md`'s `[Unreleased]` section using imperative present tense, for
example: `Add mutually colliding affine-body VF/EE contact and an eight-bunny pile example.`

- [ ] **Step 7: Run focused example and registration tests**

```bash
uv run --extra dev -m newton.tests -k "test_example_basic_limx_affine"
```

Expected: new and existing affine bunny examples pass.

- [ ] **Step 8: Commit the eight-bunny example**

```bash
git add newton/examples/basic/example_basic_limx_affine_bunnies_ground.py \
  newton/tests/test_example_basic_limx_affine_bunnies_ground.py \
  docs/images/examples/example_basic_limx_affine_bunnies_ground.jpg \
  README.md CHANGELOG.md
git commit -m "Demonstrate affine bunny pile contact"
```

---

### Task 8: Final Regression, Formatting, and Remote Checkpoint

**Files:**
- Verify all files changed in Tasks 1-7.
- Do not modify or stage unrelated user-owned files.

**Interfaces:**
- Consumes: the complete implementation and focused tests.
- Produces: a verified `dev` branch checkpoint pushed to `origin/dev`.

- [ ] **Step 1: Reconfirm the worktree boundary**

```bash
git status --short
git diff --name-only origin/dev...HEAD
```

Expected: the pre-existing user files remain unstaged; feature commits contain
only the files enumerated by this plan.

- [ ] **Step 2: Run the focused solver/contact regression set**

```bash
uv run --extra dev -m newton.tests -k "test_solver_limx_affine"
```

Expected: zero failures.

- [ ] **Step 3: Run the focused 300-frame CUDA examples**

```bash
uv run --extra dev -m newton.tests -k "test_example_basic_limx_affine"
```

Expected: zero failures, no contact overflow, and all rollout acceptance
metrics satisfied.

- [ ] **Step 4: Run pre-commit without staging unrelated changes**

First record `git status --short`. Run the required full check:

```bash
uvx pre-commit run -a
```

Then compare `git status --short` with the recorded state. If a hook reformats
an unrelated user-owned file, restore that file to its exact pre-hook bytes
from a temporary backup before proceeding. Apply and stage hook changes only
to feature-owned files, rerun `uvx pre-commit run -a`, and require every hook
to pass.

- [ ] **Step 5: Verify final diffs and push the checkpoint**

```bash
git diff --check
git status --short
git log --oneline origin/dev..HEAD
git push origin dev
```

Expected: diff check exits zero, only the known unrelated user files remain
unstaged, and `origin/dev` advances to the final feature commit.
