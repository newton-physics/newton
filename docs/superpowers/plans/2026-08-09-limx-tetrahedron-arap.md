# LIMX Tetrahedral ARAP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public tetrahedral ARAP static constraint to `SolverLIMX` and demonstrate it with a fixed cantilever beam advanced by one full Newton step per time step.

**Architecture:** `ConstraintTetrahedronARAP` owns a batch of four-particle tetrahedra, evaluates the libuipc ARAP energy with a proper signed SVD, projects the complete analytical `9 x 9` deformation-gradient Hessian with generic symmetric EVD, and assembles all sixteen ordered `3 x 3` blocks into LIMX's existing block-CSR matrix. The existing inertia assembly, PCG solve, full increment, and position/velocity update remain unchanged; the example combines this new constraint with `ConstraintAnchor` and no collision or damping.

**Tech Stack:** Python 3.10+, Warp CUDA kernels, NumPy references, `unittest`, Newton's block-CSR LIMX solver, `uv`/`uvx`.

## Global Constraints

- Follow `/home/limx/github/libuipc/src/backends/cuda/finite_element/constitutions/arap_function.h` for ARAP energy, gradient, and exact Hessian.
- Follow `/home/limx/github/libuipc/src/backends/cuda/utils/make_spd.h`: decompose the full symmetric `9 x 9` Hessian, clamp negative eigenvalues to zero, and reconstruct it.
- Do not replace the generic EVD with direct analytical twist-mode eigenvalue clamping in this milestone.
- Reuse `SolverLIMX`; do not introduce another FEM solver or alter its integrator, PCG, or block size.
- Use exactly one Newton iteration and a full increment per `0.01 s` example step; do not add line search or substeps.
- Do not add collision, self-collision, material damping, velocity damping, inversion recovery, affine-body degrees of freedom, or new dependencies.
- Keep examples and documentation on the public `newton.solvers` API; only focused internal tests may import private helpers.
- Use `unittest`, and give every test method a triple-double-quoted imperative docstring.
- Use `uv` for test/example commands and `uvx pre-commit run -a` for final lint/format validation.
- Preserve the 2026 SPDX creation year on all newly created source files.

## File Map

- Create `newton/_src/solvers/limx/constraints/tetrahedron_arap.py`: Warp math helpers, force/Hessian kernels, validation, block-CSR binding, and `ConstraintTetrahedronARAP`.
- Create `newton/tests/test_constraint_tetrahedron_arap.py`: NumPy references plus constructor, energy, derivative, PSD, assembly, and rollout tests.
- Modify `newton/_src/solvers/limx/constraints/__init__.py`: export the constraint from the constraint package.
- Modify `newton/_src/solvers/limx/__init__.py`: export the constraint from LIMX.
- Modify `newton/_src/solvers/__init__.py`: add the lazy public solver export.
- Modify `docs/api/newton_solvers.rst`: regenerate the public solver autosummary after adding the symbol.
- Modify `CHANGELOG.md`: announce the public ARAP constraint and fixed-beam example under `[Unreleased] / Added`.
- Create `newton/examples/softbody/example_softbody_limx_arap_beam.py`: fixed tetrahedral cantilever example with one Newton iteration.
- Modify `newton/tests/test_examples.py`: register a short CUDA smoke run of the example.
- Create `docs/images/examples/example_softbody_limx_arap_beam.jpg`: accepted 320-by-320 example image.
- Modify `README.md`: add the example image, source link, and launch command to the Softbody table.

---

### Task 1: Constraint Data and Block Topology

**Files:**
- Create: `newton/_src/solvers/limx/constraints/tetrahedron_arap.py`
- Create: `newton/tests/test_constraint_tetrahedron_arap.py`

**Interfaces:**
- Consumes: `BlockCsrBuilder`, `BlockCsrMatrix`, tetrahedron indices shaped `[tetrahedron_count, 4]`, inverse rest matrices `Sequence[wp.mat33]`, scalar stiffnesses in pascals, particle count, and Warp device.
- Produces: `ConstraintTetrahedronARAP(tetrahedron_indices, inverse_rest_matrices, stiffnesses, particle_count, device)` with `host_tetrahedron_indices`, `host_inverse_rest_matrices`, `host_rest_volumes`, `host_stiffnesses`, runtime Warp arrays, `append_hessian_structure()`, and `bind_hessian()`.

- [ ] **Step 1: Write failing construction and topology tests**

Add a `TestConstraintTetrahedronARAPConstruction(unittest.TestCase)` class. Use the unit tetrahedron

```python
REST_POSITIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
INVERSE_REST = np.eye(3, dtype=np.float32)
```

Cover these exact behaviors:

```python
def test_recovers_rest_volume_and_creates_sixteen_blocks(self):
    """Recover positive rest volume and bind all ordered particle blocks."""
    constraint = ConstraintTetrahedronARAP(
        [(0, 1, 2, 3)], [wp.mat33(*INVERSE_REST.reshape(-1))], [7.0], 4, "cpu"
    )
    self.assertEqual(constraint.host_rest_volumes, (1.0 / 6.0,))

    builder = BlockCsrBuilder(4)
    constraint.append_hessian_structure(builder)
    matrix = builder.finalize("cpu")
    constraint.bind_hessian(matrix)

    np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 4, 8, 12, 16])
    np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 1, 2, 3] * 4)
    self.assertEqual(constraint.hessian_block_indices.shape, (1, 16))
```

Add subtests rejecting: empty or mismatched batches; a repeated tetrahedron vertex; any negative or out-of-range vertex; `particle_count <= 0`; a NaN matrix; a singular matrix; a matrix with non-positive recovered volume; and zero, negative, NaN, or infinite stiffness. Add separate tests that `append_hessian_structure()` rejects a different row count and `bind_hessian()` rejects a different row count or device.

- [ ] **Step 2: Run the construction tests and verify the import fails**

Run:

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap.TestConstraintTetrahedronARAPConstruction -v
```

Expected: fail with `ModuleNotFoundError` for `tetrahedron_arap`.

- [ ] **Step 3: Implement validated host/device data and fixed block topology**

Define the class with this exact public signature and data conversion:

```python
class ConstraintTetrahedronARAP:
    """A batch of tetrahedral as-rigid-as-possible elastic constraints."""

    def __init__(
        self,
        tetrahedron_indices: Sequence[tuple[int, int, int, int]],
        inverse_rest_matrices: Sequence[wp.mat33],
        stiffnesses: Sequence[float],
        particle_count: int,
        device: Any,
    ):
        tetrahedron_count = len(tetrahedron_indices)
        self.host_tetrahedron_indices = tuple(
            tuple(int(index) for index in tetrahedron) for tetrahedron in tetrahedron_indices
        )
        self.host_inverse_rest_matrices = tuple(
            np.asarray(matrix, dtype=np.float32).reshape(3, 3)
            for matrix in inverse_rest_matrices
        )
        self.host_rest_volumes = tuple(
            1.0 / (6.0 * float(np.linalg.det(matrix)))
            for matrix in self.host_inverse_rest_matrices
        )
        self.host_stiffnesses = tuple(float(stiffness) for stiffness in stiffnesses)
```

Validate before creating Warp arrays. Require exactly four distinct in-range indices, finite nonsingular `Dm_inverse`, finite positive recovered volume, and finite positive stiffness. Store:

```python
self.tetrahedron_indices = wp.array2d(
    self.host_tetrahedron_indices, dtype=int, device=self.device
)
self.inverse_rest_matrices = wp.array(
    np.asarray(self.host_inverse_rest_matrices), dtype=wp.mat33, device=self.device
)
self.rest_volumes = wp.array(self.host_rest_volumes, dtype=float, device=self.device)
self.stiffnesses = wp.array(self.host_stiffnesses, dtype=float, device=self.device)
self.hessian_block_indices: wp.array2d[int] | None = None
self.hessian_value_count: int | None = None
```

`append_hessian_structure()` must call `builder.ensure_block(i, j)` for all sixteen ordered local pairs. `bind_hessian()` must materialize those sixteen `matrix.block_index(i, j)` values per tetrahedron and remember `len(matrix.values)`.

- [ ] **Step 4: Re-run the construction tests**

Run the command from Step 2.

Expected: all construction/topology tests pass on CPU.

- [ ] **Step 5: Commit the independently testable topology**

```bash
git add newton/_src/solvers/limx/constraints/tetrahedron_arap.py newton/tests/test_constraint_tetrahedron_arap.py
git commit -m "Add tetrahedral ARAP constraint topology"
```

---

### Task 2: Signed SVD, Energy, and Force

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/tetrahedron_arap.py`
- Modify: `newton/tests/test_constraint_tetrahedron_arap.py`

**Interfaces:**
- Consumes: validated tetrahedron data from Task 1 and current `wp.array[wp.vec3]` positions.
- Produces: private Warp functions `_signed_svd3()`, `_arap_energy()`, `_arap_gradient()`, `_deformation_gradient()`, `_material_gradients()`; kernel `_accumulate_tetrahedron_arap_force`; and public `accumulate_force(positions, output)`.

- [ ] **Step 1: Add NumPy reference and failing CUDA force tests**

Define a proper-rotation NumPy reference:

```python
def arap_energy_reference(positions, inverse_rest, stiffness):
    ds = np.column_stack(
        (positions[1] - positions[0], positions[2] - positions[0], positions[3] - positions[0])
    )
    deformation = ds @ inverse_rest
    u, singular_values, vt = np.linalg.svd(deformation)
    if np.linalg.det(u) < 0.0:
        u[:, -1] *= -1.0
        singular_values[-1] *= -1.0
    if np.linalg.det(vt) < 0.0:
        vt[-1, :] *= -1.0
        singular_values[-1] *= -1.0
    rotation = u @ vt
    rest_volume = 1.0 / (6.0 * np.linalg.det(inverse_rest))
    return stiffness * rest_volume * np.sum((deformation - rotation) ** 2)
```

In a CUDA-only `TestConstraintTetrahedronARAPMath` class, add tests with method docstrings for:

- zero energy and force at rest, evaluated through a test-only kernel that calls `_arap_energy()`;
- zero energy and force after a rigid translation and a non-axis-aligned proper rotation;
- zero total force and zero total torque for a deformed tetrahedron;
- centered finite-difference agreement between physical force and negative energy gradient, using `epsilon=1.0e-4`, `rtol=3.0e-3`, and `atol=3.0e-3`.

Use `wp.ScopedDevice("cuda:0")`, construct the constraint on that device, and compare `.numpy()` results to the NumPy reference.

- [ ] **Step 2: Run CUDA math tests and observe the missing force method**

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap.TestConstraintTetrahedronARAPMath -v
```

Expected: fail because `accumulate_force()` and its Warp kernel are absent.

- [ ] **Step 3: Implement proper signed SVD and ARAP force**

Create local `vec9` and `mat99` Warp types. Implement `_signed_svd3(F)` with `wp.svd3(F)`. Preserve `F = U @ diag(sigma) @ transpose(V)` while enforcing proper `U` and `V`: when either determinant is negative, negate its last column and negate `sigma[2]` at the same time.

Implement these equations directly:

```python
F = wp.mat33(x1 - x0, x2 - x0, x3 - x0) @ inverse_rest
rotation = U @ wp.transpose(V)
energy = stiffness * rest_volume * wp.ddot(F - rotation, F - rotation)
gradient_f = 2.0 * stiffness * rest_volume * (F - rotation)
```

Because the `wp.mat33(vec3, vec3, vec3)` constructor layout must not be assumed, form `Ds` by assigning each column explicitly. Compute material gradients from rows of `inverse_rest`:

```python
b1 = wp.vec3(inverse_rest[0, 0], inverse_rest[0, 1], inverse_rest[0, 2])
b2 = wp.vec3(inverse_rest[1, 0], inverse_rest[1, 1], inverse_rest[1, 2])
b3 = wp.vec3(inverse_rest[2, 0], inverse_rest[2, 1], inverse_rest[2, 2])
b0 = -(b1 + b2 + b3)
```

For each local vertex, compute `gradient_x = gradient_f * b_local` and atomically subtract it from the force output. Add runtime validation requiring `wp.vec3` arrays, exact `particle_count`, and the constraint device.

- [ ] **Step 4: Re-run the CUDA force tests**

Run the command from Step 2.

Expected: all energy invariance, equilibrium, and finite-difference gradient tests pass.

- [ ] **Step 5: Commit the force implementation**

```bash
git add newton/_src/solvers/limx/constraints/tetrahedron_arap.py newton/tests/test_constraint_tetrahedron_arap.py
git commit -m "Implement tetrahedral ARAP forces"
```

---

### Task 3: Exact Hessian, Full EVD Projection, and Assembly

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/tetrahedron_arap.py`
- Modify: `newton/tests/test_constraint_tetrahedron_arap.py`

**Interfaces:**
- Consumes: signed SVD and material gradients from Task 2; `warp.fem.linalg.symmetric_eigenvalues_qr`; block indices bound in Task 1.
- Produces: `_arap_hessian_unscaled(F) -> mat99`, `_project_psd(H) -> mat99`, `_map_hessian_block(H, b_i, b_j) -> wp.mat33`, kernel `_accumulate_tetrahedron_arap_force_and_hessian`, and public `accumulate_force_and_hessian()`.

- [ ] **Step 1: Add exact NumPy Hessian references and failing CUDA tests**

Build the three normalized libuipc twist modes exactly:

```python
twists = (
    np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
)
modes = [(u @ twist @ vt / np.sqrt(2.0)).reshape(9, order="F") for twist in twists]
hessian = 2.0 * np.eye(9)
hessian -= 4.0 / (singular_values[0] + singular_values[1]) * np.outer(modes[0], modes[0])
hessian -= 4.0 / (singular_values[1] + singular_values[2]) * np.outer(modes[1], modes[1])
hessian -= 4.0 / (singular_values[0] + singular_values[2]) * np.outer(modes[2], modes[2])
```

Define test-only Warp kernels that call the private raw-Hessian and PSD helpers and write `wp.array(dtype=mat99)` outputs. Add CUDA tests with method docstrings for:

1. the unscaled analytical Hessian against the NumPy formula on a nonsymmetric positive-determinant deformation;
2. the raw analytical Hessian against a centered finite difference of `2 * (F - R)` at `F = diag(1.4, 1.3, 1.2)`, whose raw Hessian is already PSD, using `epsilon=1.0e-4`, `rtol=4.0e-3`, and `atol=4.0e-3`;
3. projected Hessian against `eigenvectors @ diag(max(eigenvalues, 0)) @ eigenvectors.T` from `np.linalg.eigh` on a compressed deformation with a negative raw eigenvalue;
4. the full assembled `12 x 12` Hessian against `J.T @ H_F_PSD @ J`;
5. assembled symmetry and minimum eigenvalue at least `-2.0e-3`;
6. all sixteen block coordinates present and values changing after reassembly without changing CSR topology;
7. runtime errors before kernel launch for unbound Hessian storage, wrong position/force size, wrong device, and Hessian value-count mismatch.

Construct the `9 x 12` reference Jacobian with column-major `vec(F)` indexing:

```python
jacobian = np.zeros((9, 12))
for local_vertex, material_gradient in enumerate((b0, b1, b2, b3)):
    for material_axis in range(3):
        for spatial_axis in range(3):
            jacobian[3 * material_axis + spatial_axis, 3 * local_vertex + spatial_axis] = (
                material_gradient[material_axis]
            )
```

- [ ] **Step 2: Run Hessian tests and observe missing helpers**

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap.TestConstraintTetrahedronARAPHessian -v
```

Expected: fail because raw Hessian, PSD projection, and force/Hessian assembly are absent.

- [ ] **Step 3: Implement the analytical `9 x 9` Hessian**

Flatten each matrix in column-major order with index `3 * column + row`. Initialize `H = 2 I`. For each libuipc twist matrix `T_k`, compute

```python
mode = U @ T_k @ wp.transpose(V) / wp.sqrt(2.0)
H -= 4.0 / guarded_sigma_sum * wp.outer(vec(mode), vec(mode))
```

Use the pairs `(0, 1)`, `(1, 2)`, and `(0, 2)`. Guard only a near-zero denominator: preserve its sign and replace magnitude below `1.0e-6` with `1.0e-6`. Do not introduce a different constitutive response for inversion.

- [ ] **Step 4: Implement strict full-matrix PSD projection**

Call:

```python
eigenvalues, eigenvectors_by_row = symmetric_eigenvalues_qr(hessian, 1.0e-6)
```

Reconstruct without assuming column eigenvectors:

```python
projected = mat99(0.0)
for mode in range(9):
    eigenvalue = wp.max(eigenvalues[mode], 0.0)
    for row in range(9):
        for column in range(9):
            projected[row, column] += (
                eigenvalue
                * eigenvectors_by_row[mode, row]
                * eigenvectors_by_row[mode, column]
            )
```

This is `P.T @ diag(clamp(lambda)) @ P`, matching Warp's row-eigenvector convention and libuipc's generic `make_spd` behavior.

- [ ] **Step 5: Map the PSD Hessian and assemble all sixteen blocks**

Scale the projected deformation Hessian by `stiffness * rest_volume`. For each ordered local pair, implement

```python
block[spatial_i, spatial_j] = sum(
    b_i[material_i]
    * hessian_f[3 * material_i + spatial_i, 3 * material_j + spatial_j]
    * b_j[material_j]
    for material_i in range(3)
    for material_j in range(3)
)
```

Atomically add every block to `hessian_values[hessian_block_indices[tet, 4 * local_i + local_j]]`. In the same kernel, add the four forces from Task 2. Validate bound state, `wp.mat33` Hessian dtype, device, and exact stored block count before launch.

- [ ] **Step 6: Run all focused constraint tests**

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap -v
```

Expected: construction, energy/force, raw-Hessian, PSD, assembly, and error-path tests all pass.

- [ ] **Step 7: Commit the Hessian implementation**

```bash
git add newton/_src/solvers/limx/constraints/tetrahedron_arap.py newton/tests/test_constraint_tetrahedron_arap.py
git commit -m "Assemble projected ARAP Hessians"
```

---

### Task 4: Solver Integration and Public API

**Files:**
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`
- Modify: `newton/tests/test_constraint_tetrahedron_arap.py`
- Modify: `docs/api/newton_solvers.rst`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `ConstraintTetrahedronARAP` from Task 3 and existing PEP 562 lazy solver exports.
- Produces: public `newton.solvers.ConstraintTetrahedronARAP` and generated API documentation entry.

- [ ] **Step 1: Add a failing public-export test**

```python
def test_public_export_resolves_constraint(self):
    """Resolve the tetrahedral ARAP constraint through the public solver API."""
    self.assertIs(newton.solvers.ConstraintTetrahedronARAP, ConstraintTetrahedronARAP)
```

- [ ] **Step 2: Run the export test and verify it fails**

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap.TestConstraintTetrahedronARAPConstruction.test_public_export_resolves_constraint -v
```

Expected: fail because the name is absent from `newton.solvers`.

- [ ] **Step 3: Add exports at all three internal boundaries**

Add `ConstraintTetrahedronARAP` to imports and `__all__` in the constraints and LIMX package initializers. Add it to the `TYPE_CHECKING` import, `__all__`, and lazy map in `newton/_src/solvers/__init__.py`:

```python
"ConstraintTetrahedronARAP": (".limx", "ConstraintTetrahedronARAP"),
```

- [ ] **Step 4: Regenerate API docs and add the changelog entry**

Run:

```bash
uv run docs/generate_api.py
```

Verify `docs/api/newton_solvers.rst` lists `ConstraintTetrahedronARAP`. Insert this line at a non-terminal position in `[Unreleased] / Added`:

```markdown
- Add a public LIMX tetrahedral ARAP constraint with exact analytical derivatives and full-matrix positive-semidefinite Hessian projection.
```

- [ ] **Step 5: Re-run export and focused tests**

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap -v
```

Expected: all focused tests pass and the public identity assertion succeeds.

- [ ] **Step 6: Commit the public API**

```bash
git add newton/_src/solvers/limx/constraints/__init__.py newton/_src/solvers/limx/__init__.py newton/_src/solvers/__init__.py newton/tests/test_constraint_tetrahedron_arap.py docs/api/newton_solvers.rst CHANGELOG.md docs/superpowers/plans/2026-08-09-limx-tetrahedron-arap.md
git commit -m "Expose LIMX tetrahedral ARAP"
```

---

### Task 5: Fixed Cantilever Beam Example and Rollout Regression

**Files:**
- Create: `newton/examples/softbody/example_softbody_limx_arap_beam.py`
- Modify: `newton/tests/test_constraint_tetrahedron_arap.py`
- Modify: `newton/tests/test_examples.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: public `ConstraintTetrahedronARAP`, `ConstraintAnchor`, `SolverLIMX`, `ModelBuilder.add_soft_grid()`, and `ViewerBase.log_state()`.
- Produces: discoverable example `softbody.example_softbody_limx_arap_beam` with `Example.step()`, `render()`, `test_post_step()`, `test_final()`, and standard parser.

- [ ] **Step 1: Add a failing one-step and short-rollout regression**

Create a test helper that builds a small `4 x 1 x 1` grid with `0.05 m` cells, active positive masses, no fixed builder particles, and CUDA device. Build anchors from particles whose initial x coordinate equals the minimum x coordinate. Use public constraints and:

```python
solver = newton.solvers.SolverLIMX(
    model,
    [anchor_constraint, arap_constraint],
    nonlinear_iterations=1,
    linear_iterations=128,
    velocity_damping=1.0,
)
```

Add two test methods:

```python
def test_single_newton_step_updates_velocity_from_position_increment(self):
    """Advance ARAP particles with exactly one Newton increment."""
    solver.step(state_in, state_out, None, None, 0.01)
    np.testing.assert_allclose(
        state_out.particle_qd.numpy(),
        (state_out.particle_q.numpy() - state_in.particle_q.numpy()) / 0.01,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    self.assertEqual(solver.nonlinear_iterations, 1)
```

```python
def test_fixed_beam_rollout_sags_without_inversion(self):
    """Keep a fixed ARAP beam finite, anchored, sagging, and positive-volume."""
    minimum_free_end_z = np.inf
    for _ in range(80):
        solver.step(state_in, state_out, None, None, 0.01)
        state_in, state_out = state_out, state_in
        positions = state_in.particle_q.numpy()
        minimum_free_end_z = min(minimum_free_end_z, np.mean(positions[free_end_indices, 2]))
        for tet in tetrahedra:
            ds = np.column_stack((positions[tet[1]] - positions[tet[0]], positions[tet[2]] - positions[tet[0]], positions[tet[3]] - positions[tet[0]]))
            self.assertGreater(np.linalg.det(ds), 0.0)
    positions = state_in.particle_q.numpy()
    np.testing.assert_allclose(positions[anchor_indices], rest_positions[anchor_indices], atol=2.0e-3)
    self.assertLess(minimum_free_end_z, np.mean(rest_positions[free_end_indices, 2]) - 2.0e-3)
    self.assertTrue(np.isfinite(positions).all())
```

Also add a separate test that imports the not-yet-created example and locks its integration controls:

```python
def test_example_uses_one_full_newton_step_without_damping(self):
    """Configure the ARAP beam with one undamped Newton step per frame."""
    module = importlib.import_module(
        "newton.examples.softbody.example_softbody_limx_arap_beam"
    )
    example = module.Example(ViewerNull(num_frames=1), None)
    self.assertEqual(example.frame_dt, 0.01)
    self.assertEqual(example.solver.nonlinear_iterations, 1)
    self.assertEqual(example.solver.linear_iterations, 128)
    self.assertEqual(example.solver.velocity_damping, 1.0)
```

- [ ] **Step 2: Run rollout tests before adding the example**

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap.TestConstraintTetrahedronARAPSolver -v
```

Expected: the numerical one-step and rollout checks pass, while the configuration test fails with `ModuleNotFoundError` for `example_softbody_limx_arap_beam`. If the numerical checks fail, correct SVD/Hessian mapping before creating the larger scene; do not weaken positive-volume, finite-state, one-Newton-step, or no-damping requirements.

- [ ] **Step 3: Implement the production fixed-beam scene**

Use this exact configuration:

```python
self.frame_dt = 0.01
builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
builder.add_soft_grid(
    pos=wp.vec3(0.0, -0.05, 0.75),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0),
    dim_x=12,
    dim_y=2,
    dim_z=2,
    cell_x=0.05,
    cell_y=0.05,
    cell_z=0.05,
    density=1000.0,
    k_mu=0.0,
    k_lambda=0.0,
    k_damp=0.0,
    fix_left=False,
)
self.model = builder.finalize()
```

Create per-tetrahedron stiffness `[1.0e6] * self.model.tet_count`; create left-layer anchors with stiffness `1.0e8`; and instantiate:

```python
self.solver = newton.solvers.SolverLIMX(
    self.model,
    [self.anchor_constraint, self.arap_constraint],
    nonlinear_iterations=1,
    linear_iterations=128,
    velocity_damping=1.0,
)
```

Use one `self.solver.step(self.state_0, self.state_1, None, None, self.frame_dt)` per `step()`, with no collision pipeline and no substep loop. Swap the two states once. Render generated surface triangles through `viewer.set_model(self.model)` and `viewer.log_state(self.state_0)`. Set a camera that frames the full `0.60 x 0.10 x 0.10 m` beam.

In `test_post_step()`, update the minimum observed free-end height and assert finite positions/velocities plus positive current tetrahedron determinants. In `test_final()`, assert anchor error below `2.0e-3 m` and observed free-end sag greater than `2.0e-3 m`.

- [ ] **Step 4: Register a focused CUDA example smoke test**

Add after the existing hanging-softbody registration:

```python
add_example_test(
    TestSoftbodyExamples,
    name="softbody.example_softbody_limx_arap_beam",
    devices=cuda_test_devices,
    test_options={"num-frames": 20},
    use_viewer=True,
)
```

Append the example to the existing ARAP changelog bullet so it reads:

```markdown
- Add a public LIMX tetrahedral ARAP constraint with exact analytical derivatives, full-matrix positive-semidefinite Hessian projection, and a fixed cantilever-beam example.
```

- [ ] **Step 5: Run the focused rollout and example tests**

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap -v
uv run -m newton.examples softbody_limx_arap_beam --device cuda:0 --viewer null --test --num-frames 20
```

Expected: the math/solver suite passes; the example completes 20 CUDA frames with finite positive-volume state and its final sag assertion passes.

- [ ] **Step 6: Commit the working example before visual polish**

```bash
git add newton/examples/softbody/example_softbody_limx_arap_beam.py newton/tests/test_constraint_tetrahedron_arap.py newton/tests/test_examples.py CHANGELOG.md
git commit -m "Add LIMX ARAP beam example"
```

---

### Task 6: Visual Review, README Registration, and Final Verification

**Files:**
- Create: `docs/images/examples/example_softbody_limx_arap_beam.jpg`
- Modify: `README.md`
- Modify: all files changed by formatter or API generation if those changes are directly caused by this feature.

**Interfaces:**
- Consumes: accepted interactive scene from Task 5.
- Produces: visual evidence, README registration, clean focused/full verification, and a reviewable branch.

- [ ] **Step 1: Launch the interactive scene for user review**

```bash
uv run -m newton.examples softbody_limx_arap_beam --device cuda:0 --num-frames 1000
```

Expected visual behavior: the left layer remains fixed, the free end bends and oscillates under gravity, no tetrahedron visibly explodes or inverts, and there is no collision/contact object. Confirm with the user before changing camera or material parameters.

- [ ] **Step 2: Capture the accepted frame as a 320-by-320 JPEG**

Use the accepted camera from the example and capture frame 80 from a square headless GL viewer:

```bash
uv run python - <<'PY'
from PIL import Image
import warp as wp
from newton.examples.softbody.example_softbody_limx_arap_beam import Example
from newton.viewer import ViewerGL

with wp.ScopedDevice("cuda:0"):
    viewer = ViewerGL(width=320, height=320, headless=True)
    try:
        example = Example(viewer, None)
        for _ in range(80):
            example.step()
            example.render()
        frame = viewer.get_frame().numpy()
        Image.fromarray(frame).convert("RGB").save(
            "docs/images/examples/example_softbody_limx_arap_beam.jpg",
            quality=95,
        )
    finally:
        viewer.close()
PY
```

Verify the saved file:

```bash
uv run python -c "from PIL import Image; p='docs/images/examples/example_softbody_limx_arap_beam.jpg'; im=Image.open(p); print(im.size, im.mode)"
```

Expected: `(320, 320)` and `RGB`.

- [ ] **Step 3: Register the example in the Softbody README table**

Fill the third Softbody column with:

```html
<a href="https://github.com/newton-physics/newton/blob/main/newton/examples/softbody/example_softbody_limx_arap_beam.py">
  <img width="320" src="https://raw.githubusercontent.com/newton-physics/newton/main/docs/images/examples/example_softbody_limx_arap_beam.jpg" alt="LIMX ARAP Beam">
</a>
```

and:

```html
<code>python -m newton.examples softbody_limx_arap_beam</code>
```

- [ ] **Step 4: Run focused tests and API drift check**

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap -v
uv run -m newton.examples softbody_limx_arap_beam --device cuda:0 --viewer null --test --num-frames 20
uv run docs/generate_api.py
git diff --check
```

Expected: all tests pass; API regeneration produces no new unstaged drift; `git diff --check` is silent.

- [ ] **Step 5: Run repository pre-commit validation**

```bash
uvx pre-commit run -a
```

Expected: every hook passes. If a hook reformats feature files, re-run Step 4 and this command until both are clean.

- [ ] **Step 6: Inspect scope and commit documentation**

```bash
git status -sb
git diff --stat HEAD
git diff --check
git add README.md docs/images/examples/example_softbody_limx_arap_beam.jpg
git commit -m "Document LIMX ARAP beam example"
```

Expected: only planned feature files are included; unrelated user changes from the primary worktree are absent.

- [ ] **Step 7: Verify the final branch head**

```bash
uv run --extra dev -m unittest newton.tests.test_constraint_tetrahedron_arap -v
uv run -m newton.examples softbody_limx_arap_beam --device cuda:0 --viewer null --test --num-frames 20
git status -sb
git log --oneline -6
```

Expected: both verification commands pass, the worktree is clean, and the feature commits follow the design commit on `vegtsunami/limx-arap-fem`.
