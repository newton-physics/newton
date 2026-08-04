# LIMX Near-Parallel EE Mollifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate persistent T-shirt jitter by smoothly suppressing ill-defined near-parallel, topology-local EE response without damping or lowering collision stiffness.

**Architecture:** Extend each matrix-free contact buffer with a frozen nonnegative response scale used identically by force, Hessian-vector, and diagonal kernels. EE detection computes a topology-aware angle mollifier while VF and EF store one; the existing adaptive stiffness, local thickness clamp, contact topology, and PSD rank-one Hessian remain otherwise unchanged.

**Tech Stack:** Python 3.12, Warp CUDA kernels and CUDA graphs, NumPy, Newton `unittest` runner.

## Global Constraints

- Run all simulation validation on `cuda:0`; use CPU only after a CUDA failure requires diagnosis.
- Use `uv` and `unittest`, never pytest.
- Do not run the complete mixed CPU/GPU `test_solver_limx.py`; select exact CUDA classes or methods with `-p` and `-k`.
- Keep `dt=0.01`, one Newton iteration, 50 PCG iterations, and previous-frame PCG warm start.
- Do not add global, material, or self-contact damping, restitution, friction, or hysteresis.
- Keep adaptive factors `(VF=0.5, EE=0.1, EF=1.5)` and table/bending parameters unchanged.
- Keep static elasticity in 3x3 block-CSR and dynamic collision matrix-free.
- Preserve the fixed-stiffness API and its base stiffness selection.
- Do not implement VE/VV classification, line search, CCD, or additional Newton iterations.
- Preserve unrelated `lessons.md` and `solver_convergence.png` workspace changes.

---

### Task 1: Freeze one response scale per matrix-free contact

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`

**Interfaces:**
- Consumes: `_ContactBuffer.ids`, `weights`, `directions`, `depths`, fixed stiffness, and adaptive directional feature stiffness.
- Produces: `_ContactBuffer.response_scales: wp.array[float]`, initialized to one, with every fixed/adaptive force, HVP, and diagonal contribution multiplied by the stored scale.

- [ ] **Step 1: Write fixed and adaptive response-scale CUDA tests**

Add `test_fixed_contact_response_scale_multiplies_force_hessian_and_diagonal`
using one four-particle contact with weights `[1,-1/3,-1/3,-1/3]`, direction
`(1,0,0)`, depth `0.2`, stiffness `8`, and response scale `0.25`. The literal
effective stiffness is `2`, so assert forces `[0.4,-0.133333,-0.133333,-0.133333]`
on X, HVP `[2,-0.666667,-0.666667,-0.666667]`, and diagonal X entries
`[2,2/9,2/9,2/9]`.

Modify `test_adaptive_contact_uses_directional_feature_stiffness` to assign
response scale `0.4`. Its existing base stiffness `3.75` becomes `1.5`; assert
force X `[0.3,-0.1,-0.1,-0.1]`, HVP X `[1.5,-0.5,-0.5,-0.5]`, and diagonal X
entries `[1.5,1.5/9,1.5/9,1.5/9]`.

- [ ] **Step 2: Run both tests and verify RED**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k contact_response_scale
uv run --extra dev -m newton.tests -p test_solver_limx.py -k adaptive_contact_uses_directional_feature_stiffness
```

Expected: FAIL because `_ContactBuffer` has no `response_scales` array and the
current kernels cannot apply it.

- [ ] **Step 3: Add the frozen response-scale buffer and kernel inputs**

Allocate the array with default-one semantics so existing manually populated
test contacts retain their old behavior:

```python
self.response_scales = wp.ones(capacity, dtype=wp.float32, device=self.device)
```

Pass `response_scales` to all six fixed/adaptive force, HVP, and diagonal
kernels. Compute each rank-one contribution with:

```python
effective_stiffness = response_scales[contact] * stiffness
```

For adaptive kernels, multiply the value returned by
`_adaptive_contact_stiffness`; do not change the adaptive feature averages or
harmonic formula.

- [ ] **Step 4: Run the complete contact-buffer CUDA class and verify GREEN**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k TestSelfCollisionContactBuffer
```

Expected: all response-scale, old fixed, and adaptive dense-reference tests
pass.

### Task 2: Compute the topology-aware near-parallel EE mollifier

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/tests/test_example_cloth_limx_tshirt_table.py`
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py`

**Interfaces:**
- Consumes: the Task 1 `response_scales` output array and existing edge adjacency columns `[opposite_0, opposite_1, endpoint_0, endpoint_1]`.
- Produces: `_edge_edge_response_scale(edge_vector_0, edge_vector_1, topology_local) -> float`, with local smoothstep scaling and non-local identity scaling, plus a final-window localized-stillness regression.

- [ ] **Step 1: Strengthen the long CUDA regression and verify RED**

Add this test-only CUDA kernel to
`newton/tests/test_example_cloth_limx_tshirt_table.py`:

```python
@wp.kernel
def _accumulate_particle_speed_squared(
    velocities: wp.array[wp.vec3],
    speed_squared_sums: wp.array[float],
):
    particle = wp.tid()
    speed_squared_sums[particle] += wp.length_sq(velocities[particle])
```

During the final 500 frames, launch it once per frame. After the rollout,
compute `particle_rms_speeds = sqrt(sums / 500)` and assert literal behavior:

```python
self.assertEqual(int(np.count_nonzero(particle_rms_speeds >= 0.02)), 0)
self.assertLess(float(particle_rms_speeds.max()), 0.02)
```

Retain the existing finite-state, global mean-speed, and table-gap assertions.
Run:

```bash
uv run --extra dev -m newton.tests -p test_example_cloth_limx_tshirt_table.py -k settles_on_table
```

Expected: FAIL with nonzero particles above the RMS threshold and a maximum
RMS speed around `0.08-0.10 m/s`.

- [ ] **Step 2: Write CUDA detection tests for local and non-local edge pairs**

Add three real detection fixtures to `TestConstraintSelfCollisionDetection`:

1. `test_parallel_topology_local_edge_edge_contact_has_zero_response` builds
   triangles `(0,1,2)` and `(2,3,4)` with edge `(0,1)` parallel to `(2,3)`,
   interior closest parameters, and distance below the local thickness. Assert
   the stored pair `[0,1,2,3]` has scale `0`.
2. `test_angled_topology_local_edge_edge_contact_has_full_response` uses the
   same topology with `sin(theta)=0.3`. Assert scale `1`.
3. `test_parallel_nonlocal_edge_edge_contact_keeps_full_response` builds
   triangles `(0,1,4)` and `(2,3,5)` with a slightly non-collinear parallel
   pair so the closest parameters remain interior. Assert scale `1`.

Extend `_stored_contacts()` to return response scales, and update existing
callers to unpack the fifth return value. Assert all stored scale arrays are
finite and within `[0,1]` in the new tests.

- [ ] **Step 3: Run the three detection tests and verify RED**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k topology_local_edge_edge_contact
uv run --extra dev -m newton.tests -p test_solver_limx.py -k parallel_nonlocal_edge_edge_contact
```

Expected: FAIL because detection does not write contact response scales.

- [ ] **Step 4: Implement the local-angle smoothstep and write every detection scale**

Extract the existing adjacency condition into one Warp function and reuse it
for both local thickness and response scaling. Define constants:

```python
_EE_PARALLEL_SINE_THRESHOLD = 0.2
_EE_PARALLEL_SINE_SQUARED_THRESHOLD = 0.04
```

For topology-local pairs compute:

```python
length_product_squared = wp.length_sq(edge_vector_0) * wp.length_sq(edge_vector_1)
if length_product_squared <= _MIN_GEOMETRY_NORM * _MIN_GEOMETRY_NORM:
    response_scale = 0.0
else:
    sine_squared = wp.length_sq(wp.cross(edge_vector_0, edge_vector_1)) / length_product_squared
    q = wp.clamp(sine_squared / _EE_PARALLEL_SINE_SQUARED_THRESHOLD, 0.0, 1.0)
    response_scale = q * q * (3.0 - 2.0 * q)
```

Return one immediately for non-local pairs. Add `contact_response_scales` to
all detection kernel outputs; VF and EF write `1`, while EE writes the computed
value. Keep zero-scale contacts stored so the response changes continuously
without a new detection threshold.

- [ ] **Step 5: Run detection, buffer, and localized-stillness CUDA regressions**

Run:

```bash
uv run --extra dev -m newton.tests -p test_solver_limx.py -k TestSelfCollisionContactBuffer
uv run --extra dev -m newton.tests -p test_solver_limx.py -k TestConstraintSelfCollisionDetection
uv run --extra dev -m newton.tests -p test_example_cloth_limx_tshirt_table.py -k settles_on_table
```

Expected: all selected tests pass, including the old local thickness clamp and
ordinary non-local EE contact behavior. The T-shirt has zero particles at or
above `0.02 m/s` final-window RMS speed, maximum RMS speed below `0.02 m/s`,
global mean below `0.02 m/s`, and minimum table gap no lower than `-0.008 m`.

### Task 3: Make localized stillness a regression requirement

**Files:**
- Modify: `newton/tests/test_example_cloth_limx_tshirt_table.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: the Task 2 mollifier and localized-stillness regression.
- Produces: changelog coverage, focused verification, and one independently reviewable commit.

- [ ] **Step 1: Record the behavior fix**

Insert this imperative entry at a non-terminal position in `[Unreleased]`'s
`Fixed` category:

```markdown
- Stabilize LIMX topology-local near-parallel EE self-collision with a smooth PSD response mollifier.
```

- [ ] **Step 2: Run formatting and focused CUDA verification**

Run:

```bash
uvx pre-commit run -a
uv run --extra dev -m newton.tests -p test_solver_limx_static_contact.py
uv run --extra dev -m newton.tests -p test_solver_limx.py -k TestSelfCollisionContactBuffer
uv run --extra dev -m newton.tests -p test_solver_limx.py -k TestConstraintSelfCollisionDetection
uv run --extra dev -m newton.tests -p test_example_cloth_limx_tshirt_table.py
```

Expected: all selected CUDA tests and pre-commit hooks pass.

- [ ] **Step 3: Commit the independently verified fix**

Stage only the production, test, and changelog files:

```bash
git add CHANGELOG.md \
  newton/_src/solvers/limx/constraints/self_collision.py \
  newton/tests/test_solver_limx.py \
  newton/tests/test_example_cloth_limx_tshirt_table.py
git commit -m "Stabilize LIMX near-parallel EE contact"
```

### Task 4: Interactive visual verification

**Files:**
- No repository files change.

**Interfaces:**
- Consumes: the committed mollified self-collision operator and unchanged T-shirt scene parameters.
- Produces: a live CUDA OpenGL viewer for user approval.

- [ ] **Step 1: Launch the exact T-shirt scene**

Run:

```bash
uv run --extra examples -m newton.examples cloth_limx_tshirt_table --device cuda:0 --num-frames 1000000 --render-fps 100
```

- [ ] **Step 2: Verify the live process and GUI controls**

Poll startup output for kernel or viewer errors and confirm the process remains
alive. Do not report the visual result as accepted; hand the live window to the
user for the final judgment.

- [ ] **Step 3: Confirm workspace scope**

Run:

```bash
git status --short
git log -3 --oneline
```

Expected: only the pre-existing `lessons.md` and `solver_convergence.png`
workspace changes remain outside the committed feature files.
