# LIMX Geometry-Aware Collision Thickness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize LIMX cloth self-collision on irregular meshes by deriving fixed per-particle collision radii from rest geometry while retaining the requested scalar thickness as a nominal upper bound.

**Architecture:** Add an opt-in host-side radius estimator to `ConstraintSelfCollision`, store its result in a diagnostic Warp array, and interpolate those radii at VF and EE closest points during narrow-phase contact detection. Keep the nominal scalar for conservative BVH queries, preserve the legacy uniform path byte-for-byte in behavior when the option is omitted, and leave EF untangling unchanged.

**Tech Stack:** Python 3.12, NumPy, Warp CUDA kernels, Newton `SolverLIMX`, `unittest`, Sphinx autosummary, uv

## Global Constraints

- Treat `docs/superpowers/specs/2026-08-05-limx-geometry-aware-collision-thickness-design.md` as the source of truth.
- Append `geometry_radius_scale: float | None = None` to the public constructor; existing callers must retain their current behavior.
- Interpret `thickness` as the nominal two-surface activation distance and `0.5 * thickness` as the maximum one-sided particle radius.
- Compute radii once from rest positions; never recompute them from current deformed positions.
- Use triangle minimum altitude, the minimum over each vertex one-ring, and `radius_i = min(0.5 * thickness, geometry_radius_scale * local_scale_i)`.
- Interpolate VF face radii barycentrically and EE edge radii with the independent closest-point parameters.
- Keep the broad phase at nominal `thickness`; keep EF untangling, adaptive stiffness, contact capacities, and the IPC EE mollifier unchanged.
- Preserve the current one-ring current-edge-length clamp only in legacy uniform mode; geometry-aware mode bypasses that time-varying clamp.
- Do not add required or optional dependencies; use only NumPy, Warp, and the standard library already present.
- Use public `newton.solvers.ConstraintSelfCollision` from examples; do not import `newton._src` outside internal tests.
- Use `unittest`; every new test method needs a triple-double-quoted imperative docstring.
- Do not call `wp.synchronize()` or `wp.synchronize_device()` before `.numpy()`.
- Use `uv` for tests and examples and `uvx pre-commit run -a` for final linting.
- Preserve all pre-existing dirty changes. In particular, do not stage or revert the existing IPC mollifier edits in `self_collision.py` / `test_solver_limx.py`, the three-T-shirt edits, `lessons.md`, or `solver_convergence.png`.
- Because `self_collision.py` and `test_solver_limx.py` already contain unrelated unstaged edits, use partial staging and inspect `git diff --cached` before every commit. Stage only hunks named in this plan.

## File Map

- Modify `newton/_src/solvers/limx/constraints/self_collision.py`: validate the option, estimate fixed radii, expose `particle_radii`, and use interpolated effective thickness in VF/EE detection.
- Modify `newton/tests/test_solver_limx.py`: test radius estimation, validation, legacy compatibility, VF/EE interpolation, local EE behavior, and separating force.
- Modify `newton/examples/cloth/example_cloth_limx_ee_chatter.py`: turn the reproducer into a geometry-aware stabilization comparison without changing solver tuning.
- Modify `newton/tests/test_example_cloth_limx_ee_chatter.py`: change the long-run oracle from persistent chatter to settling and verify the chosen radius mode.
- Modify `README.md`: rename the gallery description from chatter reproduction to geometry-aware stabilization.
- Modify `CHANGELOG.md`: document the opt-in public behavior and correct the example description.
- Replace `docs/images/examples/example_cloth_limx_ee_chatter.jpg`: show the stabilized final state at 320×320.
- Regenerate `docs/api/newton_solvers.rst` through `docs/generate_api.py`; expect no new public symbol, but verify the constructor remains discoverable.

---

### Task 1: Establish the failing end-to-end settling criterion

**Files:**
- Modify: `newton/tests/test_example_cloth_limx_ee_chatter.py:70-125`

**Interfaces:**
- Consumes: existing `Example`, `collision_patch.self_collision`, three contact buffers, and the 74-vertex stored snapshot.
- Produces: `test_geometry_aware_collision_settles_without_contact_churn`, the RED acceptance test used by Task 4.

- [ ] **Step 1: Replace the late-chatter oracle with the settling oracle**

Replace `test_reproduces_late_edge_edge_contact_churn` with this complete method; do not change the example yet:

```python
    def test_geometry_aware_collision_settles_without_contact_churn(self):
        """Settle the irregular patch without persistent EE active-set churn."""
        module = _load_example_module(self)
        frame_count = 1400
        sample_start = 1000

        with wp.ScopedDevice("cuda:0"):
            example = module.Example(ViewerNull(num_frames=frame_count), None)
            control_rms_speeds = []
            collision_rms_speeds = []
            previous_ee_ids = None
            ee_births = 0
            ee_deaths = 0
            maximum_overflow = np.zeros(3, dtype=np.int32)
            for frame in range(frame_count):
                example.step()
                self_collision = example.collision_patch.self_collision
                maximum_overflow = np.maximum(
                    maximum_overflow,
                    np.asarray(
                        [
                            self_collision.vertex_face_contacts.overflow_count.numpy()[0],
                            self_collision.edge_edge_contacts.overflow_count.numpy()[0],
                            self_collision.edge_face_contacts.overflow_count.numpy()[0],
                        ],
                        dtype=np.int32,
                    ),
                )
                if frame < sample_start:
                    continue

                interior_indices = example.interior_indices
                control_velocities = example.control_patch.state_0.particle_qd.numpy()[interior_indices]
                collision_velocities = example.collision_patch.state_0.particle_qd.numpy()[interior_indices]
                control_rms_speeds.append(float(np.sqrt(np.mean(control_velocities * control_velocities))))
                collision_rms_speeds.append(float(np.sqrt(np.mean(collision_velocities * collision_velocities))))

                contacts = self_collision.edge_edge_contacts
                contact_count = min(int(contacts.count.numpy()[0]), contacts.capacity)
                current_ee_ids = {tuple(map(int, ids)) for ids in contacts.ids[:contact_count].numpy()}
                if previous_ee_ids is not None:
                    ee_births += len(current_ee_ids - previous_ee_ids)
                    ee_deaths += len(previous_ee_ids - current_ee_ids)
                previous_ee_ids = current_ee_ids

        control_rms_mean = float(np.mean(control_rms_speeds))
        collision_rms_mean = float(np.mean(collision_rms_speeds))
        total_ee_churn = ee_births + ee_deaths
        summary = (
            f"control_rms={control_rms_mean:.8f}, collision_rms={collision_rms_mean:.8f}, "
            f"EE_births={ee_births}, EE_deaths={ee_deaths}"
        )
        self.assertLess(control_rms_mean, 1.0e-6, summary)
        self.assertLess(collision_rms_mean, 1.0e-5, summary)
        self.assertLessEqual(total_ee_churn, 10, summary)
        np.testing.assert_array_equal(maximum_overflow, np.zeros(3, dtype=np.int32), err_msg=summary)
```

- [ ] **Step 2: Run the acceptance test and record the expected failure**

Run:

```bash
uv run --extra dev -m newton.tests \
  -p 'test_example_cloth_limx_ee_chatter.py' \
  -k test_geometry_aware_collision_settles_without_contact_churn \
  -j 1 -v
```

Expected: FAIL after 1400 frames because the current 6 mm uniform path has collision RMS speed above `1e-3` and hundreds of EE births/deaths. Save the reported RMS/birth/death values in the implementation notes; a skip, import error, overflow, or non-finite result is not the intended RED state.

- [ ] **Step 3: Leave the RED test unstaged until Task 4**

Run:

```bash
git diff -- newton/tests/test_example_cloth_limx_ee_chatter.py
git diff --cached -- newton/tests/test_example_cloth_limx_ee_chatter.py
```

Expected: the settling-oracle diff is visible only in the working tree. Do not commit a deliberately failing repository state.

---

### Task 2: Estimate and validate fixed per-particle radii

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py:18-25,1653-1760`
- Test: `newton/tests/test_solver_limx.py:1306-1390`

**Interfaces:**
- Consumes: rest positions shaped `[particle_count, 3]`, triangle indices shaped `[triangle_count, 3]`, nominal radius, and a positive finite scale.
- Produces: `_compute_geometry_aware_particle_radii(rest_positions: np.ndarray, triangle_indices: np.ndarray, nominal_radius: float, geometry_radius_scale: float) -> np.ndarray`, `ConstraintSelfCollision.geometry_radius_scale: float | None`, and `ConstraintSelfCollision.particle_radii: wp.array[float]`.

- [ ] **Step 1: Add failing constructor and exact-radius tests**

Add these methods to `TestConstraintSelfCollisionDetection`:

```python
    def test_geometry_radius_scale_validates_and_uniform_default_stays_available(self):
        """Validate the radius scale and expose uniform legacy radii by default."""
        positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0)
            for scale, message in ((0.0, "positive"), (-0.1, "positive"), (np.inf, "finite"), (np.nan, "finite")):
                with self.subTest(scale=scale):
                    with self.assertRaisesRegex(ValueError, message):
                        ConstraintSelfCollision(
                            model,
                            thickness=0.1,
                            stiffness=10.0,
                            geometry_radius_scale=scale,
                        )
            radii = collision.particle_radii.numpy()

        self.assertIsNone(collision.geometry_radius_scale)
        np.testing.assert_allclose(radii, np.full(3, 0.05, dtype=np.float32), rtol=0.0, atol=1.0e-7)

    def test_geometry_aware_radii_use_minimum_incident_triangle_altitude(self):
        """Cap each particle radius using its smallest incident rest altitude."""
        positions = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.25, 0.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (1, 3, 2)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.6,
                stiffness=10.0,
                geometry_radius_scale=0.5,
            )
            radii = collision.particle_radii.numpy()

        expected_small_radius = 0.25 / np.sqrt(5.0)
        expected = np.asarray([0.3, expected_small_radius, expected_small_radius, expected_small_radius])
        self.assertEqual(collision.geometry_radius_scale, 0.5)
        np.testing.assert_allclose(radii, expected, rtol=1.0e-6, atol=1.0e-7)

    def test_geometry_aware_radii_reject_invalid_rest_geometry(self):
        """Reject non-finite, degenerate, and unreferenced rest geometry."""
        valid_positions = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        with wp.ScopedDevice("cuda:0"):
            nonfinite_model = self._make_model(valid_positions, [(0, 1, 2)])
            nonfinite_positions = valid_positions.copy()
            nonfinite_positions[1, 0] = np.nan
            nonfinite_model.particle_q.assign(nonfinite_positions)
            with self.assertRaisesRegex(ValueError, "finite"):
                ConstraintSelfCollision(
                    nonfinite_model,
                    thickness=0.1,
                    stiffness=10.0,
                    geometry_radius_scale=0.25,
                )

            degenerate_model = self._make_model(valid_positions, [(0, 1, 2)])
            degenerate_model.particle_q.assign(
                np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
            )
            with self.assertRaisesRegex(ValueError, "degenerate"):
                ConstraintSelfCollision(
                    degenerate_model,
                    thickness=0.1,
                    stiffness=10.0,
                    geometry_radius_scale=0.25,
                )

            unreferenced_model = self._make_model(
                np.vstack((valid_positions, [[2.0, 2.0, 0.0]])).astype(np.float32),
                [(0, 1, 2)],
            )
            with self.assertRaisesRegex(ValueError, "referenced"):
                ConstraintSelfCollision(
                    unreferenced_model,
                    thickness=0.1,
                    stiffness=10.0,
                    geometry_radius_scale=0.25,
                )
```

- [ ] **Step 2: Run the new tests and verify the API is missing**

Run:

```bash
uv run --extra dev -m newton.tests \
  -p 'test_solver_limx.py' \
  -k geometry_aware_radii -k geometry_radius_scale \
  -j 1 -v
```

Expected: FAIL with an unexpected `geometry_radius_scale` keyword or missing `particle_radii` attribute.

- [ ] **Step 3: Implement the host-side estimator**

Add this internal helper below the constants in `self_collision.py`:

```python
def _compute_geometry_aware_particle_radii(
    rest_positions: np.ndarray,
    triangle_indices: np.ndarray,
    nominal_radius: float,
    geometry_radius_scale: float,
) -> np.ndarray:
    if not np.isfinite(rest_positions).all():
        raise ValueError("geometry-aware collision requires finite rest positions")

    triangle_positions = rest_positions[triangle_indices]
    edge_01 = triangle_positions[:, 1] - triangle_positions[:, 0]
    edge_12 = triangle_positions[:, 2] - triangle_positions[:, 1]
    edge_20 = triangle_positions[:, 0] - triangle_positions[:, 2]
    maximum_edges = np.maximum.reduce(
        (
            np.linalg.norm(edge_01, axis=1),
            np.linalg.norm(edge_12, axis=1),
            np.linalg.norm(edge_20, axis=1),
        )
    )
    twice_areas = np.linalg.norm(np.cross(edge_01, -edge_20), axis=1)
    if np.any(maximum_edges <= _MIN_GEOMETRY_NORM):
        raise ValueError("geometry-aware collision requires non-degenerate rest triangles")
    triangle_scales = twice_areas / maximum_edges
    if np.any(triangle_scales <= _MIN_GEOMETRY_NORM) or not np.isfinite(triangle_scales).all():
        raise ValueError("geometry-aware collision requires non-degenerate rest triangles")

    local_scales = np.full(len(rest_positions), np.inf, dtype=np.float64)
    for corner in range(3):
        np.minimum.at(local_scales, triangle_indices[:, corner], triangle_scales)
    if not np.isfinite(local_scales).all():
        raise ValueError("geometry-aware collision requires every particle to be referenced by a triangle")

    return np.minimum(nominal_radius, geometry_radius_scale * local_scales).astype(np.float32)
```

- [ ] **Step 4: Extend the constructor without changing the default path**

Append the parameter and document it exactly as follows:

```python
        stiffness_factors: tuple[float, float, float] | None = None,
        geometry_radius_scale: float | None = None,
```

```python
            geometry_radius_scale: Optional dimensionless rest-geometry radius
                scale. When set, each one-sided particle radius is capped by
                this value times its minimum incident triangle altitude. The
                initial recommended value is ``0.25``.
```

Immediately after validating `thickness`, validate and normalize the option:

```python
        if geometry_radius_scale is not None:
            if not np.isfinite(geometry_radius_scale):
                raise ValueError("geometry_radius_scale must be finite")
            if geometry_radius_scale <= 0.0:
                raise ValueError("geometry_radius_scale must be positive")
            geometry_radius_scale = float(geometry_radius_scale)
```

After triangle topology validation and before constructing `MeshAdjacency`, create the device array:

```python
        nominal_radius = 0.5 * self.thickness
        if geometry_radius_scale is None:
            particle_radii = np.full(model.particle_count, nominal_radius, dtype=np.float32)
        else:
            rest_positions = np.asarray(model.particle_q.numpy(), dtype=np.float64)
            particle_radii = _compute_geometry_aware_particle_radii(
                rest_positions,
                triangle_indices,
                nominal_radius,
                geometry_radius_scale,
            )
        self.geometry_radius_scale = geometry_radius_scale
        self.particle_radii = wp.array(particle_radii, dtype=wp.float32, device=self.device)
        self._use_geometry_radii = int(geometry_radius_scale is not None)
```

Change the `thickness` docstring line to `Nominal two-surface collision activation distance [m].` Do not add geometry validation to the `None` path.

Expand the class docstring so the diagnostic field is part of the public API documentation:

```python
    """Frictionless matrix-free cloth self-collision constraints.

    Attributes:
        particle_radii: One-sided collision radii [m], shape
            ``[particle_count]``.
    """

    particle_radii: wp.array[float]
```

- [ ] **Step 5: Run the focused estimator tests and legacy constructor tests**

Run:

```bash
uv run --extra dev -m newton.tests \
  -p 'test_solver_limx.py' \
  -k geometry_aware_radii -k geometry_radius_scale \
  -k untangle_stiffness -k stiffness_mode \
  -j 1 -v
```

Expected: all selected tests PASS. The non-finite and degenerate cases construct valid models first and then replace only `model.particle_q`, so failures must come from the new constructor validation.

- [ ] **Step 6: Partially stage and commit only the estimator work**

Run `git add -p` for the two overlapping files. Select only the new helper, constructor argument/validation/array, docstring, and the three new radius tests; split or edit mixed hunks so the pre-existing mollifier changes remain unstaged.

```bash
git add -p newton/_src/solvers/limx/constraints/self_collision.py newton/tests/test_solver_limx.py
git diff --cached --check
git diff --cached -- newton/_src/solvers/limx/constraints/self_collision.py newton/tests/test_solver_limx.py
git commit -m "Add geometry-aware LIMX collision radii"
```

Expected cached diff: no changes to `_EdgeEdgeContactBuffer`, mollifier residuals, mollifier Hessians, or pre-existing adaptive-stiffness tests.

---

### Task 3: Interpolate local radii in VF and EE detection

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py:150-315,1800-1860`
- Test: `newton/tests/test_solver_limx.py:1390-1545`

**Interfaces:**
- Consumes: `particle_radii: wp.array[float]`, `_use_geometry_radii: int`, VF barycentric coordinates, and EE closest parameters `(parameter_0, parameter_1)` from Task 2.
- Produces: geometry-aware VF/EE `contact_depths`; legacy mode continues to use scalar `thickness` and its topology-local EE clamp.

- [ ] **Step 1: Add failing VF and EE interpolation tests**

Add these methods to `TestConstraintSelfCollisionDetection`:

```python
    def test_geometry_aware_vertex_face_depth_interpolates_face_radii(self):
        """Compute VF depth from vertex and barycentrically interpolated face radii."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.05],
            [3.0, 3.0, 3.0],
            [4.0, 3.0, 3.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=16,
                geometry_radius_scale=0.25,
            )
            collision.particle_radii.assign([0.01, 0.02, 0.03, 0.04, 0.01, 0.01])
            collision.prepare(model.particle_q)
            ids, _, _, depths = self._stored_contacts(collision.vertex_face_contacts)

        matches = np.nonzero(np.all(ids == [3, 0, 1, 2], axis=1))[0]
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(float(depths[int(matches[0])]), 0.0075, places=6)

    def test_geometry_aware_edge_edge_depth_interpolates_both_edges(self):
        """Compute EE depth from independent closest-point radius interpolation."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.25, -0.3, 0.05],
            [0.25, 0.7, 0.05],
            [3.0, 0.7, 2.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                geometry_radius_scale=0.25,
            )
            collision.particle_radii.assign([0.02, 0.04, 0.01, 0.03, 0.05, 0.01])
            collision.prepare(model.particle_q)
            ids, weights, _, depths = self._stored_contacts(collision.edge_edge_contacts)

        matches = np.nonzero(np.all(ids == [0, 1, 3, 4], axis=1))[0]
        self.assertEqual(len(matches), 1)
        contact = int(matches[0])
        np.testing.assert_allclose(weights[contact], [0.75, 0.25, -0.7, -0.3], atol=1.0e-6)
        self.assertAlmostEqual(float(depths[contact]), 0.011, places=6)

    def test_geometry_aware_local_edge_pair_bypasses_length_clamp_and_separates(self):
        """Keep a close local EE contact active with a finite separating force."""
        positions = [
            [-0.05, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, -0.05, 0.08],
            [0.0, 0.05, 0.08],
            [1.0, 0.05, 0.08],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (2, 3, 4)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                geometry_radius_scale=0.25,
            )
            collision.particle_radii.assign([0.05] * len(positions))
            collision.prepare(model.particle_q)
            ids, _, _, depths = self._stored_contacts(collision.edge_edge_contacts)
            force = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
            collision.edge_edge_contacts.accumulate_force(10.0, model.particle_q, force)
            force_np = force.numpy()

        matches = np.nonzero(np.all(ids == [0, 1, 2, 3], axis=1))[0]
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(float(depths[int(matches[0])]), 0.02, places=6)
        self.assertTrue(np.isfinite(force_np).all())
        self.assertGreater(float(np.linalg.norm(force_np)), 0.0)
```

- [ ] **Step 2: Run the interpolation tests and verify legacy depths are still used**

Run:

```bash
uv run --extra dev -m newton.tests \
  -p 'test_solver_limx.py' \
  -k geometry_aware_vertex_face_depth \
  -k geometry_aware_edge_edge_depth \
  -k geometry_aware_local_edge_pair \
  -j 1 -v
```

Expected: VF and EE depth assertions FAIL because the kernels still use scalar `thickness`; the local pair is absent because the legacy half-length clamp rejects its 0.08 m gap.

- [ ] **Step 3: Extend the VF kernel with interpolated effective thickness**

Add these two inputs after `thickness` in `_detect_vertex_face_contacts`:

```python
    particle_radii: wp.array[float],
    use_geometry_radii: int,
```

Keep the nominal early rejection, then insert the geometry-aware narrow-phase check after the barycentric inside-triangle test:

```python
        effective_thickness = thickness
        if use_geometry_radii != 0:
            face_radius = (
                barycentric[0] * particle_radii[index_0]
                + barycentric[1] * particle_radii[index_1]
                + barycentric[2] * particle_radii[index_2]
            )
            effective_thickness = particle_radii[vertex] + face_radius
        if distance >= effective_thickness:
            continue
```

Write the depth as:

```python
        contact_depths[contact] = effective_thickness - distance
```

The earlier condition remains `distance <= _MIN_CONTACT_DISTANCE or distance >= thickness` so the nominal scalar remains a conservative candidate bound.

- [ ] **Step 4: Extend the EE kernel with two independent interpolated radii**

Add these inputs after `thickness` in `_detect_edge_edge_contacts`:

```python
    particle_radii: wp.array[float],
    use_geometry_radii: int,
```

Replace the current `limited_thickness` / topology-local block with:

```python
        limited_thickness = thickness
        if use_geometry_radii != 0:
            radius_0 = (1.0 - parameter_0) * particle_radii[index_0]
            radius_0 += parameter_0 * particle_radii[index_1]
            radius_1 = (1.0 - parameter_1) * particle_radii[index_2]
            radius_1 += parameter_1 * particle_radii[index_3]
            limited_thickness = radius_0 + radius_1
        else:
            topology_local = _is_topology_local_edge_pair(
                edge,
                other_edge,
                index_0,
                index_1,
                index_2,
                index_3,
                edge_indices,
            )
            if topology_local:
                average_length = 0.5 * (wp.length(position_1 - position_0) + wp.length(position_3 - position_2))
                limited_thickness = wp.min(limited_thickness, 0.5 * average_length)
```

Keep the existing distance test, closest-point weights, mollifier threshold, and force/Hessian code unchanged.

- [ ] **Step 5: Pass the radius inputs from `prepare()`**

Update the two launch input lists:

```python
            inputs=[
                self.triangle_bvh.id,
                self.thickness,
                self.particle_radii,
                self._use_geometry_radii,
                self.max_contacts,
                positions,
                self.particle_world,
                self.triangle_indices,
            ],
```

```python
            inputs=[
                self.edge_bvh.id,
                self.thickness,
                self.particle_radii,
                self._use_geometry_radii,
                self.max_contacts,
                positions,
                self.rest_positions,
                self.particle_world,
                self.edge_indices,
            ],
```

Do not pass radii to `_detect_edge_face_untangle_contacts`.

- [ ] **Step 6: Run geometry-aware and legacy contact tests**

Run:

```bash
uv run --extra dev -m newton.tests \
  -p 'test_solver_limx.py' \
  -k geometry_aware_vertex_face_depth \
  -k geometry_aware_edge_edge_depth \
  -k geometry_aware_local_edge_pair \
  -k vertex_face_detection_emits_signed_barycentric_contact \
  -k edge_edge_detection_uses_distinct_closest_parameters \
  -k adjacent_opposite_edges_limit_contact_thickness \
  -k edge_face_crossing_emits_five_particle_untangle_contact \
  -j 1 -v
```

Expected: all selected tests PASS. The two legacy depth tests must remain `0.05`, and the legacy adjacent-edge test must still reject the contact.

- [ ] **Step 7: Partially stage and commit only detection/interpolation work**

```bash
git add -p newton/_src/solvers/limx/constraints/self_collision.py newton/tests/test_solver_limx.py
git diff --cached --check
git diff --cached -- newton/_src/solvers/limx/constraints/self_collision.py newton/tests/test_solver_limx.py
git commit -m "Use local radii for LIMX VF and EE contact"
```

Expected cached diff: kernel signatures, narrow-phase thresholds, launch arguments, and the three new interpolation/force tests only. The EF kernel and existing mollifier equations must not appear.

---

### Task 4: Enable the mode in the diagnosed patch and turn RED green

**Files:**
- Modify: `newton/examples/cloth/example_cloth_limx_ee_chatter.py:1-3,480-505,565-585`
- Modify: `newton/tests/test_example_cloth_limx_ee_chatter.py:30-125`

**Interfaces:**
- Consumes: `ConstraintSelfCollision(..., geometry_radius_scale=0.25)` from Tasks 2-3 and the RED rollout from Task 1.
- Produces: a 6 mm nominal-thickness example whose orange collision-enabled patch settles with no contact-buffer overflow.

- [ ] **Step 1: Enable the geometry-aware radius cap without changing solver tuning**

Change the module docstring to:

```python
"""Compare a settled control with geometry-aware LIMX VF/EE self-collision."""
```

Change only the collision constructor by appending:

```python
                geometry_radius_scale=0.25,
```

Keep `thickness=0.006`, adaptive factors `(0.5, 0.1, 1.5)`, one nonlinear iteration, 50 linear iterations, `velocity_damping=1.0`, zero gravity, zero initial velocities, material coefficients, anchors, and time step unchanged.

- [ ] **Step 2: Rename the rendered diagnostic path**

Change the orange mesh path from `"/vf_ee_collision"` to:

```python
            "/geometry_aware_vf_ee_collision",
```

Keep its color and blue control path unchanged.

- [ ] **Step 3: Extend the one-step configuration test**

After the existing `max_contacts` assertion, add:

```python
        self.assertEqual(example.collision_patch.self_collision.geometry_radius_scale, 0.25)
        radii = example.collision_patch.self_collision.particle_radii.numpy()
        self.assertEqual(radii.shape, (74,))
        self.assertTrue(np.isfinite(radii).all())
        self.assertTrue(np.all(radii > 0.0))
        self.assertTrue(np.all(radii <= 0.003))
        self.assertLess(float(np.min(radii)), 0.003)
```

- [ ] **Step 4: Run the short configuration test**

```bash
uv run --extra dev -m newton.tests \
  -p 'test_example_cloth_limx_ee_chatter.py' \
  -k test_cuda_graph_step_preserves_the_two_patch_configuration \
  -j 1 -v
```

Expected: PASS with the CUDA graph still capturable and all state finite.

- [ ] **Step 5: Run the 1400-frame settling regression**

```bash
uv run --extra dev -m newton.tests \
  -p 'test_example_cloth_limx_ee_chatter.py' \
  -k test_geometry_aware_collision_settles_without_contact_churn \
  -j 1 -v
```

Expected: PASS with mean interior collision RMS speed `< 1e-5 m/s`, `EE_births + EE_deaths <= 10`, and `[VF, EE, EF]` overflow equal to `[0, 0, 0]` over all 1400 frames.

If this exact test remains red, print `particle_radii.numpy()` percentiles plus the late persistent EE IDs and their interpolated radii. Do not change damping, nonlinear iterations, linear iterations, time step, the `0.25` scale, or the test thresholds without returning to the approved design.

- [ ] **Step 6: Commit the example and now-green acceptance test**

```bash
git add newton/examples/cloth/example_cloth_limx_ee_chatter.py newton/tests/test_example_cloth_limx_ee_chatter.py
git diff --cached --check
git diff --cached -- newton/examples/cloth/example_cloth_limx_ee_chatter.py newton/tests/test_example_cloth_limx_ee_chatter.py
git commit -m "Stabilize LIMX EE chatter example"
```

Expected cached diff: the scale argument, description/path wording, radius assertions, and settling thresholds. No embedded snapshot coordinates, masses, triangles, or boundary indices change.

---

### Task 5: Update API documentation, gallery, screenshot, and changelog

**Files:**
- Modify: `README.md:392-409`
- Modify: `CHANGELOG.md:6-40`
- Replace: `docs/images/examples/example_cloth_limx_ee_chatter.jpg`
- Regenerate: `docs/api/newton_solvers.rst`

**Interfaces:**
- Consumes: final public constructor docstring and settled example from Task 4.
- Produces: user-facing discoverability for the opt-in radius mode and an accurate 320×320 gallery image.

- [ ] **Step 1: Update the gallery label**

Change only the image alt text in the existing README cell:

```html
alt="LIMX Geometry-Aware Self-Collision"
```

Keep the source link, image path, and command `python -m newton.examples cloth_limx_ee_chatter` unchanged.

- [ ] **Step 2: Update the changelog at a non-terminal position in `Added`**

Replace the stale chatter-example entry with:

```markdown
- Add a CUDA LIMX example that contrasts a settled control with geometry-aware VF/EE self-collision at a 6 mm nominal thickness.
```

Insert this separate public API entry after the existing three-T-shirt entry:

```markdown
- Add opt-in rest-geometry-aware per-particle radii to `ConstraintSelfCollision`, capped by the nominal collision thickness and interpolated for VF/EE contacts.
```

- [ ] **Step 3: Regenerate and inspect API pages**

```bash
uv run python docs/generate_api.py
git diff -- docs/api/newton_solvers.rst docs/api/_toctree.rst
rg -n "ConstraintSelfCollision" docs/api/newton_solvers.rst
```

Expected: `ConstraintSelfCollision` remains present in the public `newton.solvers` autosummary. Generated pages may have no tracked diff because the symbol already exists; do not hand-edit generated files to force a change.

- [ ] **Step 4: Capture the settled final frame at 320×320**

Use the existing RTX viewer dependency if available in the project environment; this script runs the same 1400 frames and saves the last rendered frame:

```bash
uv run --extra examples python - <<'PY'
import importlib

import warp as wp

from newton.viewer import ViewerRTX

module = importlib.import_module("newton.examples.cloth.example_cloth_limx_ee_chatter")
with wp.ScopedDevice("cuda:0"):
    viewer = ViewerRTX(width=320, height=320, headless=True, num_frames=1400, async_rendering=False)
    try:
        example = module.Example(viewer, None)
        for _ in range(1400):
            example.step()
            example.render()
        viewer.save_screenshot("docs/images/examples/example_cloth_limx_ee_chatter.jpg")
    finally:
        viewer.close()
PY
```

Verify the asset:

```bash
file docs/images/examples/example_cloth_limx_ee_chatter.jpg
```

Expected: JPEG, exactly 320×320. If `ViewerRTX` is not installed, run the visible GL example at 320×320, advance it to frame 1400, capture the framebuffer through the viewer's screenshot UI, and overwrite this exact path without installing a dependency.

- [ ] **Step 5: Commit documentation and the regenerated image**

```bash
git add README.md CHANGELOG.md docs/images/examples/example_cloth_limx_ee_chatter.jpg docs/api/newton_solvers.rst docs/api/_toctree.rst
git diff --cached --check
git diff --cached --stat
git commit -m "Document geometry-aware collision radii"
```

If either generated RST file is unchanged, omit it from `git add`; do not stage the unrelated modified three-T-shirt screenshot.

---

### Task 6: Verify compatibility, formatting, and the combined dirty-worktree state

**Files:**
- Verify all files changed in Tasks 1-5.
- Do not modify or stage unrelated dirty files.

**Interfaces:**
- Consumes: all implementation commits and the pre-existing user work still present in the shared checkout.
- Produces: evidence that geometry-aware mode fixes the diagnosed patch and that legacy LIMX self-collision remains compatible.

- [ ] **Step 1: Run the complete self-collision unit module serially**

```bash
uv run --extra dev -m newton.tests -p 'test_solver_limx.py' -j 1 -v
```

Expected: all tests PASS, including legacy one-ring clamping, adaptive stiffness, EF untangling, IPC mollifier tests already in the dirty worktree, and new geometry-aware tests.

- [ ] **Step 2: Run the complete example regression module serially**

```bash
uv run --extra dev -m newton.tests -p 'test_example_cloth_limx_ee_chatter.py' -j 1 -v
```

Expected: 4 tests PASS when CUDA is available; on CPU-only machines the two CUDA tests may skip, but this task is not complete until the CUDA rollout has passed on the configured `cuda:0` device.

- [ ] **Step 3: Run the example through its public CLI**

```bash
uv run --extra examples -m newton.examples cloth_limx_ee_chatter \
  --device cuda:0 --viewer null --num-frames 1400 --test --quiet
```

Expected: exit code 0, finite final state, and no assertion from `test_post_step()` or `test_final()`.

- [ ] **Step 4: Run API generation once more and lint the repository**

```bash
uv run python docs/generate_api.py
uvx pre-commit run -a
```

Expected: both commands exit 0. If formatting changes task-owned files, inspect and commit only those formatting hunks; never stage unrelated dirty files.

- [ ] **Step 5: Audit the final diff and staging boundaries**

```bash
git status --short
git diff --check
git diff --cached --check
git log -6 --oneline
```

Expected: task-owned changes are committed; the pre-existing edits to the three-T-shirt files, `lessons.md`, and `solver_convergence.png` remain exactly as they were. Any pre-existing mollifier hunks intentionally left unstaged also remain present and are not silently folded into geometry-aware commits.

- [ ] **Step 6: Inspect the final interactive behavior**

```bash
uv run --extra examples -m newton.examples cloth_limx_ee_chatter \
  --device cuda:0 --viewer gl --num-frames 1400
```

Expected: both patches become visually still; the orange geometry-aware patch keeps self-collision enabled and does not show persistent yellow/orange trembling. Close the viewer after confirming the late frames.

- [ ] **Step 7: Request a two-axis code review before integration**

Invoke the repository-required `code-review` skill when it is available and review both axes:

```text
Standards: public API compatibility, Warp typing, docstrings, CUDA graph capture, no new dependencies, partial-staging safety.
Spec: exact radius formula, VF/EE interpolation, legacy clamp preservation, EF unchanged, 1400-frame settling and close-contact force.
```

If that skill is unavailable in the execution environment, use `superpowers:requesting-code-review` and explicitly provide the same two axes. Address only findings supported by code or test evidence, then rerun Steps 1-5.
