# ABD Bunny Ground Contact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public penalty-based affine static-plane contact operator and a CUDA-captured example in which a high-rigidity ABD bunny falls onto a frictional ground plane.

**Architecture:** Keep each body's generalized state as row-major `q = [t, A]`. Evaluate contact per compact surface vertex in world space, cache its `vec3` force and `mat33` PSD Hessian, and lift force, matrix-free products, and block-Jacobi terms with the constant material-point Jacobian `J`. Integrate the operator into `SolverLIMXAffine` through its existing mixed particle/affine operator without merging the particle `3x3` and affine `12x12` sparse systems.

**Tech Stack:** Python 3, Warp kernels, NumPy reference calculations, Newton LIMX native `vec12`/`mat1212` blocks, `unittest`, CUDA graph capture.

## Global Constraints

- Use the canonical checkout `/home/limx/github/newton` on branch `dev`; do not create a worktree or commit to `main`.
- Preserve unrelated user changes in `docs/images/examples/example_cloth_limx_three_tshirts_box.jpg`, `newton/tests/test_example_cloth_limx_tshirt_table.py`, and `solver_convergence.png`.
- Use `q = [t_x, t_y, t_z, row(A, 0), row(A, 1), row(A, 2)]`; do not replace `A+t` with four proxy control points.
- Keep particle blocks `3x3` and affine blocks `12x12`; contact contributes a complete `J.T @ H @ J` only to the affine preconditioner.
- Use penalty normal contact and the current LIMX regularized Coulomb penalty friction; do not add IPC barriers, CCD, line search, proxy particles, or cloth coupling.
- The example uses exactly one Newton iteration and 50 PCG iterations per frame.
- Add no required or optional dependency; use Warp, NumPy, and the standard library only.
- Write tests with `unittest`, give every test method a triple-double-quoted imperative docstring, and observe each new test fail for the intended missing behavior before implementation.
- Expose user-facing classes only through `newton.solvers`; examples and docs must not import `newton._src`.

---

### Task 1: Affine static-plane contact operator

**Files:**
- Create: `newton/_src/solvers/limx/constraints/affine_static_plane_contact.py`
- Create: `newton/tests/test_solver_limx_affine_contact.py`
- Modify: `newton/_src/solvers/limx/constraints/__init__.py`
- Modify: `newton/_src/solvers/limx/__init__.py`
- Modify: `newton/_src/solvers/__init__.py`

**Interfaces:**
- Consumes: `AffineBodyModel.rest_surface_vertices`, `AffineBodyModel.surface_ownership`, `AffineBodyModel.body_count`, and the mixed dynamic-operator signatures in `mixed_operator.py`.
- Produces: public `ConstraintAffineStaticPlaneContact(body_model, normal, offset, thickness, stiffness, normal_damping, friction, friction_epsilon)` with `begin_step(q, qd, dt)`, `prepare(q)`, `accumulate_force(q, output)`, `multiply(particle_input, affine_input, particle_output, affine_output)`, and `accumulate_diagonal(particle_diagonal, affine_diagonal)`.

- [ ] **Step 1: Write dense reference helpers and a failing force/HVP/diagonal test**

  Add this independent NumPy Jacobian helper and a `TestConstraintAffineStaticPlaneContact` fixture to the new test module:

  ```python
  def _point_jacobian(rest_position: np.ndarray) -> np.ndarray:
      x, y, z = rest_position
      return np.asarray(
          [
              [1.0, 0.0, 0.0, x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, x, y, z, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x, y, z],
          ],
          dtype=np.float64,
      )
  ```

  Construct the existing unit tetrahedron `AffineBodyModel`, set a literal generalized state and velocity, and call the public class through `newton.solvers`. For each surface point, independently compute `x = J @ q`, `v = J @ qd`, the plane depth, particle-contact normal/damping/friction force, and PSD `H`. Sum literal dense `J.T @ f`, `J.T @ H @ (J @ p)`, and `J.T @ H @ J` references. Assert the Warp outputs with `rtol=2e-5, atol=2e-6`.

- [ ] **Step 2: Run the focused test and verify the RED state**

  Run:

  ```bash
  uv run --extra dev -m newton.tests -p test_solver_limx_affine_contact.py -k force_hessian_product_and_exact_diagonal
  ```

  Expected: import or attribute failure because `newton.solvers.ConstraintAffineStaticPlaneContact` does not exist.

- [ ] **Step 3: Add behavior tests for inactive contact, damping, friction, and validation**

  Add separate tests that catch these mutations:

  - moving every reconstructed point above `thickness` must make force, HVP, and diagonal exactly zero;
  - changing an approaching normal velocity to separating must remove only the `normal_damping` force and `normal_damping / dt` Hessian;
  - a nonzero tangential generalized velocity must produce a lifted force with negative world-space work, while a displacement shorter than `friction_epsilon` remains finite;
  - zero/non-finite normal, offset, thickness, stiffness, damping, friction, and epsilon must raise with the parameter name;
  - `prepare()` before `begin_step()` must raise `RuntimeError`;
  - wrong `vec12` length/device/dtype and nonempty particle-side mixed vectors must raise clear errors.

- [ ] **Step 4: Implement world-space preparation and exact affine lifts**

  In `affine_static_plane_contact.py`, add Warp helpers that apply the row-major affine point Jacobian without materializing it:

  ```python
  @wp.func
  def _affine_point(state: vec12, rest: wp.vec3) -> wp.vec3:
      return wp.vec3(
          state[0] + state[3] * rest[0] + state[4] * rest[1] + state[5] * rest[2],
          state[1] + state[6] * rest[0] + state[7] * rest[1] + state[8] * rest[2],
          state[2] + state[9] * rest[0] + state[10] * rest[1] + state[11] * rest[2],
      )
  ```

  Use the same mapping for generalized velocities and directions. Implement `J.T @ world_vector` by placing each world component in its translation entry and multiplying it by the three rest coordinates in the corresponding affine row. Implement `J.T @ H @ J` with loops over 12 generalized columns and rows. Cache one `wp.vec3` force and one `wp.mat33` Hessian per compact surface vertex in `prepare()`.

  Accumulate lifted `vec12` and `mat1212` values with Warp atomics because multiple surface vertices share one affine body. Leave the particle arrays unchanged and require them to be empty. Validate array dtype, length, device, lifecycle ordering, and all constructor scalars before launching kernels.

- [ ] **Step 5: Export the public class**

  Add `ConstraintAffineStaticPlaneContact` to the imports and `__all__` lists in both LIMX initializer files and to the lazy `TYPE_CHECKING`, `__all__`, and `_LAZY_IMPORTS` entries in `newton/_src/solvers/__init__.py`.

- [ ] **Step 6: Run the complete affine contact test module**

  Run:

  ```bash
  uv run --extra dev -m newton.tests -p test_solver_limx_affine_contact.py
  ```

  Expected: every contact algebra, inactive, damping, friction, lifecycle, and validation test passes on CPU and CUDA when CUDA is available.

- [ ] **Step 7: Commit the independently tested contact operator**

  ```bash
  git add newton/_src/solvers/__init__.py newton/_src/solvers/limx/__init__.py newton/_src/solvers/limx/constraints/__init__.py newton/_src/solvers/limx/constraints/affine_static_plane_contact.py newton/tests/test_solver_limx_affine_contact.py
  git commit -m "Add affine plane contact operator"
  ```

---

### Task 2: Solver integration for optional dynamic affine contact

**Files:**
- Modify: `newton/_src/solvers/limx/solver_affine.py`
- Modify: `newton/tests/test_solver_limx_affine.py`

**Interfaces:**
- Consumes: the Task 1 lifecycle and mixed dynamic-operator methods.
- Produces: `SolverLIMXAffine(..., dynamic_operator: Any | None = None)`; `solver.dynamic_operator` is the provided operator or an `EmptyMixedDynamicOperator` by default.

- [ ] **Step 1: Write a failing real-contact solver test**

  Extend the existing unit-tetrahedron solver fixture with `initial_transform` and `dynamic_operator` arguments. Add a test that builds two identical affine models with one surface vertex inside a plane activation band and calls `model.gravity.zero_()` on both to isolate contact. Step one solver with no operator and the other with a real `ConstraintAffineStaticPlaneContact` using zero damping/friction. Assert that the default solver keeps its previous result, the contact solver gains positive normal displacement/velocity, the operator instance is retained, and the affine state stays finite.

- [ ] **Step 2: Verify the integration test fails for the missing constructor argument**

  Run:

  ```bash
  uv run --extra dev -m newton.tests -p test_solver_limx_affine.py -k dynamic_operator
  ```

  Expected: `SolverLIMXAffine.__init__()` rejects `dynamic_operator`.

- [ ] **Step 3: Add a validation and default-preservation test**

  Assert that an affine dynamic operator with the wrong `body_count` or device is rejected. Construct the solver without an operator and assert `isinstance(solver.dynamic_operator, EmptyMixedDynamicOperator)`; retain the existing free-fall and warm-start tests as behavioral regression coverage.

- [ ] **Step 4: Wire the operator into the Newton step**

  Update the solver docstring from collision-free to optional dynamic contact. At construction, choose the empty operator only when `dynamic_operator is None`, validate any supplied operator's `body_count` and `device`, store it, and pass it to `MixedLinearOperator`.

  Preserve this exact lifecycle in `step(dt)`:

  ```python
  self.dynamic_operator.begin_step(self.q, self.qd, dt)
  for nonlinear_iteration in range(self.nonlinear_iterations):
      # assemble inertia and ARAP
      self.dynamic_operator.prepare(self.q)
      self.dynamic_operator.accumulate_force(self.q, self.rhs)
      self.static_matrix.update_diagonal()
      self.operator.prepare(None, dt)
      # PCG and affine update remain unchanged
  ```

  Give `EmptyMixedDynamicOperator` no-op `begin_step`, `prepare`, and `accumulate_force` methods so the default path has one branch-free lifecycle and keeps graph capture stable.

- [ ] **Step 5: Run affine solver and mixed-operator regressions**

  Run:

  ```bash
  uv run --extra dev -m newton.tests -p test_solver_limx_affine.py
  ```

  Expected: the contact integration, default free fall, ARAP, 12x12 preconditioner, mixed PCG, warm start, and CUDA graph-capture tests pass.

- [ ] **Step 6: Commit the solver integration**

  ```bash
  git add newton/_src/solvers/limx/mixed_operator.py newton/_src/solvers/limx/solver_affine.py newton/tests/test_solver_limx_affine.py
  git commit -m "Integrate affine dynamic contact"
  ```

---

### Task 3: Frictional ABD bunny-ground example

**Files:**
- Create: `newton/examples/basic/example_basic_limx_affine_bunny_ground.py`
- Create: `newton/tests/test_example_basic_limx_affine_bunny_ground.py`

**Interfaces:**
- Consumes: public `AffineBodyModel`, `ConstraintAffineStaticPlaneContact`, and `SolverLIMXAffine`.
- Produces: discoverable command `uv run -m newton.examples basic_limx_affine_bunny_ground`; `Example.test_post_step()` and `Example.test_final()` enforce the 300-frame physical acceptance criteria.

- [ ] **Step 1: Write a failing configuration test for the example**

  Import the new module, construct it with `ViewerNull`, and assert these literal settings:

  ```python
  self.assertEqual(example.frame_dt, 0.01)
  self.assertEqual(example.solver.nonlinear_iterations, 1)
  self.assertEqual(example.solver.linear_iterations, 50)
  self.assertEqual(example.body_model.surface_vertex_count, 1078)
  self.assertAlmostEqual(example.body_mass, 2.81, delta=0.03)
  self.assertEqual(example.ground_contact.thickness, 0.003)
  self.assertEqual(example.ground_contact.stiffness, 2.0e4)
  self.assertEqual(example.ground_contact.normal_damping, 0.5)
  self.assertEqual(example.ground_contact.friction, 0.5)
  self.assertEqual(example.ground_contact.friction_epsilon, 1.0e-4)
  self.assertIs(example.solver.dynamic_operator, example.ground_contact)
  ```

- [ ] **Step 2: Run the example test and verify RED**

  Run:

  ```bash
  uv run --extra dev -m newton.tests -p test_example_basic_limx_affine_bunny_ground.py -k configuration
  ```

  Expected: module import failure because the example file does not exist.

- [ ] **Step 3: Implement the exact bunny scene and CUDA graph path**

  Load `newton/examples/assets/bunny_tet.npz` through `newton.TetMesh.create_from_file()`. Pass `0.15 * mesh.vertices`, reshaped tetrahedra, and reshaped surface triangles to `AffineBodyModel` with density `1000`, rigidity `1.0e8`, an upright x-axis quarter-turn composed with a 15-degree tilt, and translation centered at `z = 0.65`.

  Create `ConstraintAffineStaticPlaneContact` with upward normal, offset `0`, thickness `0.003`, stiffness `2.0e4`, normal damping `0.5`, friction `0.5`, and epsilon `1.0e-4`. Create `SolverLIMXAffine` with `nonlinear_iterations=1`, `linear_iterations=50`, and `velocity_damping=1.0`.

  Build a render-only Newton model from the reconstructed compact surface and its triangles, plus a static box whose top is exactly `z=0`. Allocate its state once. Define `simulate()` to call the affine solver and reconstruct the surface, capture `simulate()` once on CUDA, and replay the graph in `step()`.

- [ ] **Step 4: Add the 300-frame rollout test before tuning implementation behavior**

  Add a CUDA-only test that advances exactly 300 frames and calls `test_post_step()` every frame. Track and assert:

  - every generalized state, generalized velocity, and reconstructed surface value is finite;
  - `det(A) > 0` and `max(abs(svd(A) - 1)) < 0.02` at every frame;
  - center height falls by at least `0.20 m`;
  - at least one surface point enters the `0.003 m` activation band;
  - minimum surface height never falls below `-0.006 m`;
  - mean tangential translation speed over the final 30 frames is below `0.05 m/s`;
  - CUDA graph capture produced a non-`None` graph.

- [ ] **Step 5: Run the configuration and full rollout tests**

  Run:

  ```bash
  uv run --extra dev -m newton.tests -p test_example_basic_limx_affine_bunny_ground.py
  ```

  Expected: the scene configuration and 300-frame CUDA rollout both pass without weakening the design tolerances.

- [ ] **Step 6: Launch the interactive scene for visual inspection**

  Run:

  ```bash
  uv run --extra examples -m newton.examples basic_limx_affine_bunny_ground --device cuda:0 --num-frames 300
  ```

  Confirm that the bunny visibly falls, contacts the box top, does not invert or visibly squash, and friction prevents indefinite sliding.

- [ ] **Step 7: Commit the tested example**

  ```bash
  git add newton/examples/basic/example_basic_limx_affine_bunny_ground.py newton/tests/test_example_basic_limx_affine_bunny_ground.py
  git commit -m "Add frictional affine bunny drop"
  ```

---

### Task 4: Public API documentation, changelog, and example registration

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Modify: `docs/api/newton_solvers.rst` through `docs/generate_api.py`
- Create: `docs/images/examples/example_basic_limx_affine_bunny_ground.jpg`

**Interfaces:**
- Consumes: the Task 1 public export and Task 3 command.
- Produces: discoverable API documentation and a README example card with a 320x320 screenshot.

- [ ] **Step 1: Regenerate API pages from the public export**

  Run:

  ```bash
  uv run docs/generate_api.py
  ```

  Verify `ConstraintAffineStaticPlaneContact` appears in `docs/api/newton_solvers.rst` and no generated page imports `newton._src`.

- [ ] **Step 2: Add user-facing release notes**

  Insert this entry at a randomly selected non-terminal position in `[Unreleased] -> Added`:

  ```markdown
  - Add penalty-based static-plane contact for LIMX affine bodies and a frictional bunny-drop example.
  ```

- [ ] **Step 3: Capture and register the example image**

  Render the settled/contact phase at square resolution, save a JPEG at `docs/images/examples/example_basic_limx_affine_bunny_ground.jpg`, and verify it is 320x320. Add a README card linking to `newton/examples/basic/example_basic_limx_affine_bunny_ground.py`, showing that image at width 320, and listing:

  ```text
  uv run -m newton.examples basic_limx_affine_bunny_ground
  ```

- [ ] **Step 4: Run documentation and spelling checks**

  Run:

  ```bash
  uvx pre-commit run --files CHANGELOG.md README.md docs/api/newton_solvers.rst docs/images/examples/example_basic_limx_affine_bunny_ground.jpg
  ```

  Expected: formatting, generated API drift, image-size, and `typos` checks pass.

- [ ] **Step 5: Commit documentation and registration**

  ```bash
  git add CHANGELOG.md README.md docs/api/newton_solvers.rst docs/images/examples/example_basic_limx_affine_bunny_ground.jpg
  git commit -m "Document affine bunny contact"
  ```

---

### Task 5: Final verification and delivery

**Files:**
- Verify all files from Tasks 1-4.
- Do not stage the three unrelated user-owned paths listed in Global Constraints.

**Interfaces:**
- Consumes: all completed feature commits.
- Produces: a verified `dev` branch pushed to `origin/dev`.

- [ ] **Step 1: Run focused solver and example tests together**

  ```bash
  uv run --extra dev -m newton.tests -p test_solver_limx_affine_contact.py
  uv run --extra dev -m newton.tests -p test_solver_limx_affine.py
  uv run --extra dev -m newton.tests -p test_example_basic_limx_affine_body.py
  uv run --extra dev -m newton.tests -p test_example_basic_limx_affine_bunny_ground.py
  ```

  Expected: all selected modules pass, including the CUDA graph and 300-frame rollout.

- [ ] **Step 2: Run repository formatting and static checks**

  ```bash
  uvx pre-commit run -a
  ```

  If an unrelated user-owned file fails, report it separately and rerun pre-commit over every feature-owned file to prove this change is clean.

- [ ] **Step 3: Inspect scope and history**

  ```bash
  git diff --check origin/dev...HEAD
  git status --short
  git log --oneline origin/dev..HEAD
  ```

  Confirm no unrelated image, test edit, or `solver_convergence.png` is staged or committed.

- [ ] **Step 4: Push the verified feature commits**

  ```bash
  git push origin dev
  ```

  Report the pushed commit range, exact test commands and outcomes, remaining unrelated worktree changes, and the interactive example command.
