# LIMX Self-Collision Friction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add VBD-aligned, regularized Coulomb friction to LIMX cloth vertex-face and edge-edge self-collision and demonstrate it in the three-T-shirt box scene.

**Architecture:** `ConstraintSelfCollision` caches step-start particle positions and evaluates friction from each frozen contact's signed feature weights and normal. Shared matrix-free friction kernels handle force, Hessian-vector, and diagonal operations for fixed and adaptive normal stiffness; EE contacts additionally scale their normal load by the existing near-parallel mollifier. EF untangling remains frictionless.

**Tech Stack:** Python 3, Warp GPU kernels, NumPy, `unittest`, Newton LIMX, `uv`.

## Global Constraints

- Preserve frictionless behavior by default with `friction=0.0`.
- Define `friction_epsilon=1.0e-2` as a relative-velocity threshold [m/s], matching `SolverVBD`; use `epsilon_u = friction_epsilon * dt` in the displacement law.
- Apply friction only to VF and EE contacts. Do not add friction to EF untangling.
- Use `friction=0.4` in the three-T-shirt scene while retaining uniform 3 mm self-collision thickness and stiffness factors `(0.5, 0.3, 1.5)`.
- Treat the frozen contact normal and normal load as constants in the PSD friction Hessian, matching VBD's stability choice.
- Scale an active near-parallel EE contact's friction load by `s * (2 * threshold - s) / threshold**2`.
- Add no dependencies; use Warp, NumPy, and the standard library only.
- Use `unittest`, give every test a triple-double-quoted imperative docstring, and run commands through `uv`.
- Preserve all unrelated dirty-worktree changes. Stage only the exact hunks introduced by this plan.

## File Map

- Modify `newton/_src/solvers/limx/constraints/self_collision.py`: public parameters, step anchor state, friction math, VF/EE buffer operations, and top-level integration.
- Modify `newton/tests/test_solver_limx.py`: constructor, VF, EE, adaptive-stiffness, mollifier, and PSD regression tests.
- Modify `newton/examples/cloth/example_cloth_limx_three_tshirts_box.py`: enable cloth-cloth friction in the visual experiment.
- Modify `newton/tests/test_example_cloth_limx_three_tshirts_box.py`: assert the visual example's selected collision parameters.
- Modify `CHANGELOG.md`: document the new user-facing optional behavior under `[Unreleased] / Added`.

---

### Task 1: Public friction parameters and step anchors

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py:1711-1890`
- Test: `newton/tests/test_solver_limx.py:1306-1483`

**Interfaces:**
- Consumes: the existing `ConstraintSelfCollision` constructor and `begin_step(positions, velocities, dt)` dynamic-operator hook.
- Produces: public `friction: float`, public `friction_epsilon: float`, private `_friction_positions: wp.array[wp.vec3] | None`, private `_friction_displacement_epsilon: float`, and `_friction_state() -> tuple[wp.array[wp.vec3], float]`.

- [ ] **Step 1: Write the failing constructor and validation test**

Add this method to `TestConstraintSelfCollisionDetection`:

```python
def test_friction_parameters_validate_and_default_to_disabled(self):
    """Validate friction parameters and preserve a frictionless default."""
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    with wp.ScopedDevice("cuda:0"):
        model = self._make_model(positions, [(0, 1, 2)])
        default_collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0)
        friction_collision = ConstraintSelfCollision(
            model,
            thickness=0.1,
            stiffness=10.0,
            friction=0.4,
            friction_epsilon=1.0e-2,
        )
        invalid_cases = (
            ({"friction": -0.1}, "nonnegative"),
            ({"friction": np.inf}, "finite"),
            ({"friction": np.nan}, "finite"),
            ({"friction_epsilon": 0.0}, "positive"),
            ({"friction_epsilon": -1.0}, "positive"),
            ({"friction_epsilon": np.inf}, "finite"),
        )
        for kwargs, message in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, **kwargs)

    self.assertEqual(default_collision.friction, 0.0)
    self.assertEqual(default_collision.friction_epsilon, 1.0e-2)
    self.assertIsNone(default_collision._friction_positions)
    self.assertEqual(friction_collision.friction, 0.4)
    self.assertIsNotNone(friction_collision._friction_positions)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_friction_parameters_validate_and_default_to_disabled
```

Expected: ERROR because `ConstraintSelfCollision` does not accept `friction`.

- [ ] **Step 3: Add the parameters, validation, and state**

Append keyword-compatible parameters after `geometry_radius_scale`:

```python
friction: float = 0.0,
friction_epsilon: float = 1.0e-2,
```

Validate and store them before allocating topology data:

```python
if not np.isfinite(friction):
    raise ValueError("friction must be finite")
if friction < 0.0:
    raise ValueError("friction must be nonnegative")
if not np.isfinite(friction_epsilon):
    raise ValueError("friction_epsilon must be finite")
if friction_epsilon <= 0.0:
    raise ValueError("friction_epsilon must be positive")

self.friction = float(friction)
self.friction_epsilon = float(friction_epsilon)
self._friction_positions: wp.array[wp.vec3] | None = None
if self.friction > 0.0:
    self._friction_positions = wp.empty_like(model.particle_q)
self._friction_displacement_epsilon = 0.0
```

Update the class docstring from “Frictionless” to “Matrix-free” and document both arguments with `[m/s]` on `friction_epsilon`.

- [ ] **Step 4: Extend `begin_step()` without disturbing adaptive mode**

Use one validation path whenever adaptive stiffness or friction needs step data:

```python
if self.stiffness_factors is None and self.friction == 0.0:
    return
self._validate_positions(positions)
if velocities.device != self.device:
    raise ValueError(f"velocities must use device {self.device}")
if velocities.dtype != wp.vec3 or len(velocities) != self.particle_count:
    raise ValueError(f"velocities must contain {self.particle_count} wp.vec3 values")
if not np.isfinite(dt) or dt <= 0.0:
    raise ValueError("dt must be finite and positive")
if self.stiffness_factors is not None:
    if self._static_diagonal is None or self._masses is None:
        raise RuntimeError("bind_static_system() must be called before begin_step() in adaptive mode")
    self._inv_dt_squared = 1.0 / (dt * dt)
if self.friction > 0.0:
    if self._friction_positions is None:
        raise RuntimeError("friction anchor storage is unavailable")
    self._friction_positions.assign(positions)
    self._friction_displacement_epsilon = self.friction_epsilon * dt
```

Add the exact state accessor used by later tasks:

```python
def _friction_state(self) -> tuple[wp.array[wp.vec3], float]:
    if self._friction_positions is None or self._friction_displacement_epsilon <= 0.0:
        raise RuntimeError("begin_step() is required before friction evaluation")
    return self._friction_positions, self._friction_displacement_epsilon
```

- [ ] **Step 5: Run the focused test and verify GREEN**

Run the Step 2 command. Expected: one test passes.

- [ ] **Step 6: Commit only Task 1 hunks**

```bash
git add -p newton/_src/solvers/limx/constraints/self_collision.py newton/tests/test_solver_limx.py
git diff --cached --check
git commit -m "Add LIMX friction parameters"
```

---

### Task 2: Shared VF friction force and PSD operator

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py:610-769,1260-1486,1972-2035`
- Test: `newton/tests/test_solver_limx.py:1470-1796`

**Interfaces:**
- Consumes: `_adaptive_contact_stiffness`, frozen contact `ids/weights/directions/depths`, and Task 1's `_friction_state()`.
- Produces: `_regularized_friction_force_hessian`, fixed/adaptive friction kernels, and the exact buffer methods named in Step 5.

- [ ] **Step 1: Write a failing fixed-stiffness VF force/PSD test**

Add a test that moves only the contacting vertex tangentially from its step anchor:

```python
def test_vertex_face_friction_opposes_slip_and_adds_psd_operator(self):
    """Oppose VF slip with balanced friction and a PSD tangent operator."""
    current = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.05],
            [3.0, 3.0, 3.0],
            [4.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    anchor = current.copy()
    anchor[3, 0] -= 0.10
    displacement = current - anchor
    with wp.ScopedDevice("cuda:0"):
        model = self._make_model(current, [(0, 1, 2), (3, 4, 5)])
        collision = ConstraintSelfCollision(
            model,
            thickness=0.1,
            stiffness=1.0e3,
            friction=0.4,
            friction_epsilon=1.0e-2,
            max_contacts=32,
        )
        current_wp = wp.array(current, dtype=wp.vec3, device="cuda:0")
        anchor_wp = wp.array(anchor, dtype=wp.vec3, device="cuda:0")
        velocities = wp.zeros(model.particle_count, dtype=wp.vec3, device="cuda:0")
        collision.begin_step(anchor_wp, velocities, 0.01)
        collision.prepare(current_wp)
        force = wp.zeros(model.particle_count, dtype=wp.vec3, device="cuda:0")
        collision.accumulate_force(current_wp, force)
        product = wp.zeros_like(force)
        displacement_wp = wp.array(displacement, dtype=wp.vec3, device="cuda:0")
        collision.hessian_multiply(current_wp, displacement_wp, product)
        diagonal = wp.zeros(model.particle_count, dtype=wp.mat33, device="cuda:0")
        collision.accumulate_diagonal(current_wp, diagonal)
        force_np = force.numpy()
        product_np = product.numpy()
        diagonal_np = diagonal.numpy()

    self.assertLess(float(np.sum(force_np * displacement)), 0.0)
    np.testing.assert_allclose(force_np.sum(axis=0), 0.0, atol=1.0e-5)
    self.assertLessEqual(abs(float(force_np[3, 0])), 0.4 * 1.0e3 * 0.05 + 1.0e-4)
    self.assertGreaterEqual(float(np.sum(displacement * product_np)), -1.0e-5)
    for block in diagonal_np:
        self.assertGreaterEqual(float(np.linalg.eigvalsh(block).min()), -1.0e-4)
```

- [ ] **Step 2: Run the VF test and verify RED**

Run:

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_vertex_face_friction_opposes_slip_and_adds_psd_operator
```

Expected: FAIL because no tangential force is accumulated.

- [ ] **Step 3: Add the VBD-aligned regularized friction helper**

Place this after `_adaptive_contact_stiffness`:

```python
@wp.func
def _regularized_friction_force_hessian(
    direction: wp.vec3,
    relative_displacement: wp.vec3,
    normal_load: float,
    friction: float,
    displacement_epsilon: float,
):
    tangent_displacement = relative_displacement - direction * wp.dot(direction, relative_displacement)
    tangent_length = wp.length(tangent_displacement)
    if tangent_length <= 0.0 or normal_load <= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)
    friction_over_length = float(0.0)
    if tangent_length > displacement_epsilon:
        friction_over_length = 1.0 / tangent_length
    else:
        friction_over_length = (-tangent_length / displacement_epsilon + 2.0) / displacement_epsilon
    scale = friction * normal_load * friction_over_length
    tangent_projector = wp.identity(3, float) - wp.outer(direction, direction)
    return -scale * tangent_displacement, scale * tangent_projector
```

Also add a helper that reconstructs the feature-relative displacement from signed weights:

```python
@wp.func
def _contact_relative_displacement(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    contact: int,
    arity: int,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
):
    relative_displacement = wp.vec3(0.0)
    for local_index in range(arity):
        particle = ids[contact, local_index]
        relative_displacement += weights[contact, local_index] * (
            positions[particle] - anchor_positions[particle]
        )
    return relative_displacement
```

- [ ] **Step 4: Add fixed and adaptive matrix-free friction kernels**

Implement six kernels: force, Hessian-vector, and diagonal for fixed stiffness, plus the same three using `_adaptive_contact_stiffness`. Each kernel must:

```python
contact = wp.tid()
if contact >= wp.min(count[0], capacity):
    return
stiffness = fixed_stiffness
normal_load = stiffness * depths[contact] * load_scale
relative_displacement = _contact_relative_displacement(
    ids, weights, contact, arity, positions, anchor_positions
)
friction_force, friction_hessian = _regularized_friction_force_hessian(
    directions[contact],
    relative_displacement,
    normal_load,
    friction,
    displacement_epsilon,
)
```

The fixed force kernel distributes `weights[contact, local_index] * friction_force`. The Hessian kernel first computes
`relative_vector = sum_i(weights[i] * vector[ids[i]])`, then distributes
`weights[i] * friction_hessian * relative_vector`. The diagonal kernel adds
`weights[i] * weights[i] * friction_hessian` to each particle block.

The adaptive variants replace `fixed_stiffness` with:

```python
stiffness = _adaptive_contact_stiffness(
    ids,
    directions,
    contact,
    arity,
    feature_split,
    factor,
    static_diagonal,
    masses,
    inv_dt_squared,
)
```

For Task 2, pass `load_scale=1.0`; Task 3 supplies the EE mollifier scale.

- [ ] **Step 5: Expose the six operations through `_ContactBuffer`**

Add fixed methods with these exact signatures:

```python
def accumulate_friction_force(
    self,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
) -> None:

def friction_hessian_multiply(
    self,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    vector: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
) -> None:

def accumulate_friction_diagonal(
    self,
    stiffness: float,
    friction: float,
    displacement_epsilon: float,
    positions: wp.array[wp.vec3],
    anchor_positions: wp.array[wp.vec3],
    output: wp.array[wp.mat33],
) -> None:
```

Add adaptive counterparts named with an `_adaptive` suffix and arguments
`factor, static_diagonal, masses, inv_dt_squared` before `friction`. Validate all particle arrays with `_validate_output` and equal lengths before launch.

- [ ] **Step 6: Integrate VF friction into all three top-level operations**

Do not return early after the fixed normal path. After normal force/Hessian/diagonal accumulation, obtain:

```python
anchor_positions, displacement_epsilon = self._friction_state()
```

when `self.friction > 0.0`, then call the VF buffer's matching fixed or adaptive friction method. Leave EE for Task 3 and leave EF unchanged.

- [ ] **Step 7: Add and pass an adaptive VF regression**

Duplicate the geometric setup from Step 1 in a second docstring-bearing test, construct with
`stiffness=None, stiffness_factors=(0.5, 0.3, 1.5)`, bind a finite positive
`wp.mat33` static diagonal and `model.particle_mass`, and assert negative friction power, balanced total force, finite Hessian products, and PSD diagonal blocks.

Run both VF tests:

```bash
uv run --extra dev -m unittest \
  newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_vertex_face_friction_opposes_slip_and_adds_psd_operator \
  newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_adaptive_vertex_face_friction_remains_finite_and_psd
```

Expected: two tests pass.

- [ ] **Step 8: Commit only Task 2 hunks**

```bash
git add -p newton/_src/solvers/limx/constraints/self_collision.py newton/tests/test_solver_limx.py
git diff --cached --check
git commit -m "Add LIMX vertex-face friction"
```

---

### Task 3: EE friction with near-parallel load scaling

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py:769-1265,1487-1708,1972-2035`
- Test: `newton/tests/test_solver_limx.py:1500-1796`

**Interfaces:**
- Consumes: Task 2's friction helpers and buffer methods plus `_EdgeEdgeContactBuffer.mollifier_thresholds` and `.mollifier_active`.
- Produces: `_edge_edge_friction_load_scale` and EE calls in force, Hessian-vector, and diagonal paths for fixed and adaptive stiffness.

- [ ] **Step 1: Write a failing EE friction test**

Use the existing perpendicular-edge geometry from
`test_edge_edge_detection_uses_distinct_closest_parameters`. Translate the second edge by `+0.10 m` in X between anchor and current positions. After `begin_step()`, `prepare()`, and `accumulate_force()`:

```python
displacement = current - anchor
friction_power = float(np.sum(force_np * displacement))
first_feature_force = force_np[0] + force_np[1]
self.assertLess(friction_power, 0.0)
np.testing.assert_allclose(force_np.sum(axis=0), 0.0, atol=1.0e-5)
self.assertLessEqual(
    float(np.linalg.norm(first_feature_force[:2])),
    0.4 * 1.0e3 * 0.05 + 1.0e-4,
)
```

Also run Hessian-vector and diagonal operations and assert nonnegative quadratic form and block eigenvalues as in the VF test.

- [ ] **Step 2: Run the EE test and verify RED**

```bash
uv run --extra dev -m unittest newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_edge_edge_friction_opposes_relative_slip
```

Expected: FAIL because the top-level operator does not add EE friction.

- [ ] **Step 3: Add exact EE mollifier load scaling**

Add this Warp function near the existing mollifier functions:

```python
@wp.func
def _edge_edge_friction_load_scale(
    ids: wp.array2d[int],
    contact: int,
    thresholds: wp.array[float],
    mollifier_active: wp.array[int],
    positions: wp.array[wp.vec3],
):
    if mollifier_active[contact] == 0:
        return float(1.0)
    edge_0 = positions[ids[contact, 1]] - positions[ids[contact, 0]]
    edge_1 = positions[ids[contact, 3]] - positions[ids[contact, 2]]
    cross_product = wp.cross(edge_0, edge_1)
    cross_squared = wp.dot(cross_product, cross_product)
    threshold = thresholds[contact]
    return wp.clamp(
        cross_squared * (2.0 * threshold - cross_squared) / (threshold * threshold),
        0.0,
        1.0,
    )
```

Call this only from EE friction kernels. VF continues to use `1.0`.

- [ ] **Step 4: Connect EE friction in fixed and adaptive top-level paths**

After each VF friction call, call the corresponding EE buffer method with the
same `anchor_positions` and `displacement_epsilon`. The EE buffer launches the
same algebra as Task 2 but obtains `load_scale` from
`_edge_edge_friction_load_scale`. Add calls in `accumulate_force`,
`hessian_multiply`, and `accumulate_diagonal`; do not call the EF buffer.

- [ ] **Step 5: Add a mollified EE load-bound regression**

Reuse the geometry from
`test_nonlocal_edge_pair_uses_rest_length_mollifier_threshold`, add tangential
anchor displacement, and assert `mollifier_active == 1`. Compute

```python
cross_product = np.cross(edge_0, edge_1)
cross_squared = float(np.dot(cross_product, cross_product))
load_scale = cross_squared * (2.0 * threshold - cross_squared) / threshold**2
first_feature_force = force_np[contact_ids[0]] + force_np[contact_ids[1]]
tangent_force = first_feature_force - direction * float(np.dot(direction, first_feature_force))
limit = 0.4 * stiffness * depth * load_scale
self.assertLessEqual(float(np.linalg.norm(tangent_force)), limit + 1.0e-4)
```

Give the method the docstring `"""Scale EE friction by the active near-parallel mollifier."""`.

- [ ] **Step 6: Run all focused self-collision friction tests**

```bash
uv run --extra dev -m unittest \
  newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_friction_parameters_validate_and_default_to_disabled \
  newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_vertex_face_friction_opposes_slip_and_adds_psd_operator \
  newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_adaptive_vertex_face_friction_remains_finite_and_psd \
  newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_edge_edge_friction_opposes_relative_slip \
  newton.tests.test_solver_limx.TestConstraintSelfCollisionDetection.test_mollified_edge_edge_friction_uses_reduced_normal_load
```

Expected: five tests pass.

- [ ] **Step 7: Commit only Task 3 hunks**

```bash
git add -p newton/_src/solvers/limx/constraints/self_collision.py newton/tests/test_solver_limx.py
git diff --cached --check
git commit -m "Add LIMX edge-edge friction"
```

---

### Task 4: Enable and visualize three-T-shirt friction

**Files:**
- Modify: `newton/examples/cloth/example_cloth_limx_three_tshirts_box.py:170-176`
- Modify: `newton/tests/test_example_cloth_limx_three_tshirts_box.py:24-45`
- Modify: `CHANGELOG.md:5-50`

**Interfaces:**
- Consumes: `ConstraintSelfCollision` with `friction=0.4` and `friction_epsilon=1.0e-2` from Tasks 1-3.
- Produces: a visual three-garment experiment using VF/EE friction and an `[Unreleased]` user-facing changelog entry.

- [ ] **Step 1: Extend the focused example assertion and verify RED**

Add to `test_single_garment_configuration_builds_one_tshirt`:

```python
self.assertEqual(example.self_collision.friction, 0.4)
self.assertEqual(example.self_collision.friction_epsilon, 1.0e-2)
```

Run:

```bash
uv run --extra dev -m unittest newton.tests.test_example_cloth_limx_three_tshirts_box.TestClothLimxThreeTshirtsBox.test_single_garment_configuration_builds_one_tshirt
```

Expected: FAIL because the example still uses the frictionless default.

- [ ] **Step 2: Enable self-collision friction in the example**

Set the constructor arguments to:

```python
self.self_collision = newton.solvers.ConstraintSelfCollision(
    self.model,
    thickness=0.003,
    stiffness=None,
    max_contacts=393216,
    stiffness_factors=(0.5, 0.3, 1.5),
    friction=0.4,
    friction_epsilon=1.0e-2,
)
```

Do not change box-contact friction, garment initial conditions, or solver iteration counts.

- [ ] **Step 3: Add the changelog entry**

Insert at a randomly selected location within `[Unreleased] / Added`:

```markdown
- Add regularized Coulomb friction to LIMX vertex-face and edge-edge cloth self-collision.
```

- [ ] **Step 4: Run focused verification**

Run the five tests from Task 3 Step 6, then the example test from Task 4 Step 1. Expected: six tests pass.

Run formatting/lint checks on changed source files first:

```bash
uvx pre-commit run --files \
  newton/_src/solvers/limx/constraints/self_collision.py \
  newton/tests/test_solver_limx.py \
  newton/examples/cloth/example_cloth_limx_three_tshirts_box.py \
  newton/tests/test_example_cloth_limx_three_tshirts_box.py \
  CHANGELOG.md
```

Review any formatter edits before staging. Before the final feature commit, run the repository-required command:

```bash
uvx pre-commit run -a
```

Do not stage unrelated files changed by pre-commit.

- [ ] **Step 5: Launch the visual acceptance scene**

```bash
uv run --extra examples -m newton.examples cloth_limx_three_tshirts_box \
  --viewer gl --garment-count 3 --num-frames 2000
```

Expected: the viewer opens and runs 20 seconds. Cloth layers resist tangential sliding without persistent resting jitter, NaNs, or explosive motion.

- [ ] **Step 6: Commit example, test, changelog, and remaining implementation hunks**

```bash
git add -p \
  newton/_src/solvers/limx/constraints/self_collision.py \
  newton/tests/test_solver_limx.py \
  newton/examples/cloth/example_cloth_limx_three_tshirts_box.py \
  newton/tests/test_example_cloth_limx_three_tshirts_box.py \
  CHANGELOG.md
git diff --cached --check
git commit -m "Add friction to LIMX self-collision"
```

- [ ] **Step 7: Verify the committed scope**

```bash
git status --short
git show --stat --oneline HEAD
```

Expected: the commit contains only friction-related hunks; all pre-existing unrelated dirty files remain present and unstaged.
