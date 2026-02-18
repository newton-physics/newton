# geom_margin ↔ thickness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Map MuJoCo's `geom_margin` to Newton's `thickness` end-to-end: MJCF import, MuJoCo solver spec build, runtime updates, multi-world expansion, and contact conversion.

**Architecture:** Newton's `shape_thickness` is the first-class per-shape field equivalent to MuJoCo's `geom_margin`. Both are per-shape, both are summed for contact pairs. Currently, thickness is baked into the contact distance in the MuJoCo contact conversion kernel. This plan moves the activation-threshold semantics to `geom_margin` instead, matching MuJoCo's native behavior. The MJCF importer will parse `margin` → `thickness`, and the solver will map `thickness` → `geom_margin`.

**Tech Stack:** Python, Warp kernels, MuJoCo/MuJoCo-Warp, unittest

---

### Task 1: MJCF Import — parse `margin` to `thickness`

**Files:**
- Modify: `newton/_src/utils/import_mjcf.py` (around line 559, after solref parsing)
- Test: `newton/tests/test_import_mjcf.py`

**Step 1: Write the test**

Add `test_mjcf_geom_margin_parsing` to `test_import_mjcf.py`. Pattern: same as `test_mjcf_friction_parsing` (line 1998). Test that MJCF `margin="X"` on geoms maps to `builder.shape_thickness[i]`.

```python
def test_mjcf_geom_margin_parsing(self):
    """Test MJCF geom margin is parsed to shape thickness."""
    mjcf_content = """
    <mujoco>
        <worldbody>
            <body name="test_body">
                <geom name="geom1" type="box" size="0.1 0.1 0.1" margin="0.003"/>
                <geom name="geom2" type="sphere" size="0.1" margin="0.01"/>
                <geom name="geom3" type="capsule" size="0.1 0.2"/>
            </body>
        </worldbody>
    </mujoco>
    """
    builder = newton.ModelBuilder()
    builder.add_mjcf(mjcf_content, up_axis="Z")

    self.assertEqual(builder.shape_count, 3)
    self.assertAlmostEqual(builder.shape_thickness[0], 0.003, places=6)
    self.assertAlmostEqual(builder.shape_thickness[1], 0.01, places=6)
    # geom3 has no margin, should use ShapeConfig default (1e-5)
    self.assertAlmostEqual(builder.shape_thickness[2], 1e-5, places=8)
```

**Step 2: Run test to verify it fails**

```bash
export WARP_CACHE_ROOT=/tmp/claude/warp-cache-$$
uv run --extra dev -m newton.tests --no-cache-clear -k test_mjcf_geom_margin_parsing 2>&1 | tee /tmp/claude/test-results.txt
```

Expected: FAIL — margin attribute is not parsed.

**Step 3: Implement the parsing**

In `newton/_src/utils/import_mjcf.py`, in `parse_shapes()`, after the solref parsing block (around line 558), add:

```python
# Parse MJCF margin for collision thickness (only if explicitly specified)
if "margin" in geom_attrib:
    shape_cfg.thickness = parse_float(geom_attrib, "margin", shape_cfg.thickness)
```

This follows the same pattern as friction/solref: only override Newton default if the attribute is explicitly authored in MJCF.

**Step 4: Run test to verify it passes**

```bash
uv run --extra dev -m newton.tests --no-cache-clear -k test_mjcf_geom_margin_parsing 2>&1 | tee /tmp/claude/test-results.txt
```

Expected: PASS

**Step 5: Commit**

```bash
git add newton/_src/utils/import_mjcf.py newton/tests/test_import_mjcf.py
git commit -m "Parse MJCF geom margin attribute to shape thickness"
```

---

### Task 2: MuJoCo solver spec build — set `geom_margin` from `shape_thickness`

**Files:**
- Modify: `newton/_src/solvers/mujoco/solver_mujoco.py` (spec build ~line 2882 and ~line 3165)
- Test: `newton/tests/test_mujoco_solver.py`

**Step 1: Write the test**

Add a check to the existing `test_geom_property_conversion` test (line 1897) that verifies `geom_margin` matches `shape_thickness`. Or add a new focused test. Better to add a new focused test since thickness → margin is a new mapping:

```python
def test_geom_margin_from_thickness(self):
    """Test that Newton shape_thickness is correctly mapped to MuJoCo geom_margin."""
    # Set distinct thickness values
    shape_thickness = self.model.shape_thickness.numpy()
    for i in range(len(shape_thickness)):
        shape_thickness[i] = 0.001 * (i + 1)
    self.model.shape_thickness.assign(wp.array(shape_thickness, dtype=wp.float32, device=self.model.device))

    solver = SolverMuJoCo(self.model, iterations=1, disable_contacts=True)
    to_newton = solver.mjc_geom_to_newton_shape.numpy()
    geom_margin = solver.mjw_model.geom_margin.numpy()

    tested_count = 0
    for world_idx in range(self.model.num_worlds):
        for geom_idx in range(solver.mj_model.ngeom):
            shape_idx = to_newton[world_idx, geom_idx]
            if shape_idx < 0:
                continue
            tested_count += 1
            expected = shape_thickness[shape_idx]
            actual = geom_margin[world_idx, geom_idx]
            self.assertAlmostEqual(
                float(actual), expected, places=5,
                msg=f"geom_margin mismatch for shape {shape_idx} in world {world_idx}",
            )
    self.assertGreater(tested_count, 0)
```

**Step 2: Run test to verify it fails**

```bash
uv run --extra dev -m newton.tests --no-cache-clear -k test_geom_margin_from_thickness 2>&1 | tee /tmp/claude/test-results.txt
```

Expected: FAIL — geom_margin is MuJoCo default (0.0), not from thickness.

**Step 3: Implement**

In `solver_mujoco.py`:

A) Around line 2882, add `shape_thickness` to the numpy arrays read during spec build:
```python
shape_thickness = model.shape_thickness.numpy()
```

B) Around line 3165 (inside the geom_params block, after the gap block), add:
```python
geom_params["margin"] = float(shape_thickness[shape])
```

**Step 4: Run test to verify it passes**

```bash
uv run --extra dev -m newton.tests --no-cache-clear -k test_geom_margin_from_thickness 2>&1 | tee /tmp/claude/test-results.txt
```

Expected: PASS

**Step 5: Commit**

```bash
git add newton/_src/solvers/mujoco/solver_mujoco.py newton/tests/test_mujoco_solver.py
git commit -m "Map shape_thickness to geom_margin in MuJoCo solver spec build"
```

---

### Task 3: Runtime updates — add thickness → geom_margin to update_geom_properties_kernel

**Files:**
- Modify: `newton/_src/solvers/mujoco/kernels.py` (update_geom_properties_kernel, ~line 1500)
- Modify: `newton/_src/solvers/mujoco/solver_mujoco.py` (update_geom_properties, ~line 4405)
- Test: `newton/tests/test_mujoco_solver.py`

**Step 1: Write the test**

Follow the exact pattern of `test_geom_gap_conversion_and_update` (line 2599). Create `test_geom_margin_conversion_and_runtime_update`:

```python
def test_geom_margin_conversion_and_runtime_update(self):
    """Test shape_thickness → geom_margin conversion and runtime updates across multiple worlds."""
    num_worlds = 2
    template_builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(template_builder)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, thickness=0.005)

    body1 = template_builder.add_link(mass=0.1)
    template_builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
    joint1 = template_builder.add_joint_free(child=body1)

    body2 = template_builder.add_link(mass=0.1)
    shape_cfg2 = newton.ModelBuilder.ShapeConfig(density=1000.0, thickness=0.01)
    template_builder.add_shape_sphere(body=body2, radius=0.1, cfg=shape_cfg2)
    joint2 = template_builder.add_joint_revolute(parent=body1, child=body2, axis=(0.0, 0.0, 1.0))
    template_builder.add_articulation([joint1, joint2])

    builder = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(builder)
    builder.replicate(template_builder, num_worlds)
    model = builder.finalize()

    solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
    to_newton = solver.mjc_geom_to_newton_shape.numpy()
    num_geoms = solver.mj_model.ngeom

    # Verify initial conversion
    shape_thickness = model.shape_thickness.numpy()
    geom_margin = solver.mjw_model.geom_margin.numpy()
    tested_count = 0
    for world_idx in range(model.num_worlds):
        for geom_idx in range(num_geoms):
            shape_idx = to_newton[world_idx, geom_idx]
            if shape_idx < 0:
                continue
            tested_count += 1
            self.assertAlmostEqual(
                float(geom_margin[world_idx, geom_idx]),
                float(shape_thickness[shape_idx]),
                places=5,
                msg=f"Initial geom_margin mismatch for shape {shape_idx} in world {world_idx}",
            )
    self.assertGreater(tested_count, 0)

    # Update thickness values at runtime
    new_thickness = np.array([0.02 + i * 0.005 for i in range(model.shape_count)], dtype=np.float32)
    model.shape_thickness.assign(wp.array(new_thickness, dtype=wp.float32, device=model.device))
    solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

    # Verify runtime update
    updated_margin = solver.mjw_model.geom_margin.numpy()
    for world_idx in range(model.num_worlds):
        for geom_idx in range(num_geoms):
            shape_idx = to_newton[world_idx, geom_idx]
            if shape_idx < 0:
                continue
            self.assertAlmostEqual(
                float(updated_margin[world_idx, geom_idx]),
                float(new_thickness[shape_idx]),
                places=5,
                msg=f"Updated geom_margin mismatch for shape {shape_idx} in world {world_idx}",
            )
```

**Step 2: Run test to verify it fails**

```bash
uv run --extra dev -m newton.tests --no-cache-clear -k test_geom_margin_conversion_and_runtime_update 2>&1 | tee /tmp/claude/test-results.txt
```

Expected: FAIL — kernel doesn't update geom_margin.

**Step 3: Implement**

A) In `kernels.py`, add `shape_thickness` input and `geom_margin` output to `update_geom_properties_kernel`:

Add to inputs (after `shape_geom_gap` line 1517):
```python
shape_thickness: wp.array(dtype=float),
```

Add to outputs (after `geom_gap` line 1526):
```python
geom_margin: wp.array2d(dtype=float),
```

Add to kernel body (after the geom_gap update block, around line 1564):
```python
# update geom_margin from shape thickness
geom_margin[world, geom_idx] = shape_thickness[shape_idx]
```

B) In `solver_mujoco.py`, update `update_geom_properties()` (~line 4421):

Add `self.model.shape_thickness` to inputs list (after `shape_geom_gap`):
```python
self.model.shape_thickness,
```

Add `self.mjw_model.geom_margin` to outputs list (after `self.mjw_model.geom_gap`):
```python
self.mjw_model.geom_margin,
```

**Step 4: Run test to verify it passes**

```bash
uv run --extra dev -m newton.tests --no-cache-clear -k test_geom_margin_conversion_and_runtime_update 2>&1 | tee /tmp/claude/test-results.txt
```

Expected: PASS

**Step 5: Commit**

```bash
git add newton/_src/solvers/mujoco/kernels.py newton/_src/solvers/mujoco/solver_mujoco.py newton/tests/test_mujoco_solver.py
git commit -m "Add runtime thickness → geom_margin updates with multi-world support"
```

---

### Task 4: Contact conversion — remove thickness from dist, use geom_margin

**Files:**
- Modify: `newton/_src/solvers/mujoco/kernels.py` (convert_newton_contacts_to_mjwarp_kernel, ~line 190)
- Modify: `newton/_src/solvers/mujoco/solver_mujoco.py` (convert_contacts_to_mjwarp, ~line 2074)

**Step 1: Modify the kernel**

In `convert_newton_contacts_to_mjwarp_kernel`:

A) Remove `rigid_contact_thickness0` and `rigid_contact_thickness1` parameters (lines 209-210).

B) Replace lines 279-282:
```python
thickness = rigid_contact_thickness0[tid] + rigid_contact_thickness1[tid]

n = -rigid_contact_normal[tid]
dist = wp.dot(n, bx_b - bx_a) - thickness
```
with:
```python
n = -rigid_contact_normal[tid]
dist = wp.dot(n, bx_b - bx_a)
```

The `geom_margin` now handles the activation threshold via `includemargin = margin - gap` (already computed by `contact_params` and written by `write_contact`).

**Step 2: Update the solver launch call**

In `solver_mujoco.py` `convert_contacts_to_mjwarp()` (~line 2101), remove the two thickness inputs:
```python
contacts.rigid_contact_thickness0,
contacts.rigid_contact_thickness1,
```

**Step 3: Run existing tests to verify nothing breaks**

```bash
uv run --extra dev -m newton.tests --no-cache-clear -k test_mujoco_solver 2>&1 | tee /tmp/claude/test-results.txt
```

Expected: All existing MuJoCo solver tests pass. The dist values change but the combined effect of dist + includemargin should produce equivalent constraint activation.

**Step 4: Commit**

```bash
git add newton/_src/solvers/mujoco/kernels.py newton/_src/solvers/mujoco/solver_mujoco.py
git commit -m "Remove thickness from contact distance, use geom_margin for activation"
```

---

### Task 5: Run full test suite and fix any regressions

**Step 1: Run all MuJoCo solver tests**

```bash
uv run --extra dev -m newton.tests --no-cache-clear -k test_mujoco_solver 2>&1 | tee /tmp/claude/test-results.txt
```

**Step 2: Run MJCF import tests**

```bash
uv run --extra dev -m newton.tests --no-cache-clear -k test_import_mjcf 2>&1 | tee /tmp/claude/test-results.txt
```

**Step 3: Run pre-commit**

```bash
uvx pre-commit run -a
```

**Step 4: Fix any issues found, then final commit**
