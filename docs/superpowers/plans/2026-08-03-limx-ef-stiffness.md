# LIMX Edge-Face Recovery Stiffness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Default LIMX EF untangling to three times the shared VF/EE stiffness while preserving explicit overrides and making the twist-scene ratio visible.

**Architecture:** Keep the existing `ConstraintSelfCollision` interface and its separate `untangle_stiffness` field. Change only the omitted-value derivation, then configure the twist example explicitly and lock both the library default and scene values with CUDA `unittest` coverage.

**Tech Stack:** Python 3.12, Warp CUDA kernels and arrays, NumPy, Newton's `unittest` runner.

## Global Constraints

- `stiffness=k` continues to control both VF and EE.
- An omitted `untangle_stiffness` resolves to exactly `3.0 * k`.
- An explicit positive `untangle_stiffness` remains accepted without a minimum-ratio restriction.
- Run LIMX validation on `cuda:0`; do not run routine CPU simulation.
- Add no dependency and keep the public constructor signature unchanged.
- Use `unittest`, not pytest.

---

### Task 1: Apply and verify the EF stiffness ratio

**Files:**
- Modify: `newton/tests/test_solver_limx.py`
- Modify: `newton/_src/solvers/limx/constraints/self_collision.py:549-574`
- Modify: `newton/examples/cloth/example_cloth_limx_twist.py:107-112`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `ConstraintSelfCollision(model, thickness, stiffness, untangle_stiffness=None, max_contacts=32768)`.
- Produces: `ConstraintSelfCollision.untangle_stiffness: float`, equal to `3.0 * stiffness` only when the constructor argument is omitted.

- [ ] **Step 1: Write CUDA tests for the default, override, and twist configuration**

Add these two methods to `TestConstraintSelfCollisionDetection`:

```python
def test_untangle_stiffness_defaults_to_three_times_contact_stiffness(self):
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    with wp.ScopedDevice("cuda:0"):
        model = self._make_model(positions, [(0, 1, 2)])
        collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0)

    self.assertEqual(collision.stiffness, 10.0)
    self.assertEqual(collision.untangle_stiffness, 30.0)

def test_explicit_untangle_stiffness_overrides_default_ratio(self):
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    with wp.ScopedDevice("cuda:0"):
        model = self._make_model(positions, [(0, 1, 2)])
        collision = ConstraintSelfCollision(
            model,
            thickness=0.1,
            stiffness=10.0,
            untangle_stiffness=17.0,
        )

    self.assertEqual(collision.untangle_stiffness, 17.0)
```

Extend `test_twist_example_runs_limx_self_collision_cuda_graph` with:

```python
self.assertEqual(example.self_collision.stiffness, 1.0e4)
self.assertEqual(example.self_collision.untangle_stiffness, 3.0e4)
```

- [ ] **Step 2: Run the new default-ratio test and verify RED**

Run:

```bash
uv run --extra dev -m newton.tests -k test_untangle_stiffness_defaults_to_three_times_contact_stiffness
```

Expected: FAIL because the current omitted EF stiffness is `10.0`, not `30.0`.

- [ ] **Step 3: Implement the minimal default and scene changes**

In `ConstraintSelfCollision.__init__`, derive the omitted value with:

```python
if untangle_stiffness is None:
    untangle_stiffness = 3.0 * stiffness
```

Update its argument documentation to say that EF defaults to three times `stiffness`.

In the twist example, keep `stiffness=1.0e4` and add:

```python
untangle_stiffness=3.0e4,
```

Add this entry at a non-terminal position in `CHANGELOG.md` under `[Unreleased]` → `Changed`:

```markdown
- Default LIMX edge-face untangling to three times the VF/EE contact stiffness; pass an explicit `untangle_stiffness` to retain a different recovery ratio.
```

- [ ] **Step 4: Run focused CUDA verification**

Run:

```bash
uv run --extra dev -m newton.tests -k test_untangle_stiffness
uv run --extra dev -m newton.tests -k test_twist_example_runs_limx_self_collision_cuda_graph
uv run --extra dev -m newton.tests -k test_solver_limx
```

Expected: all selected tests pass on `cuda:0`, with no non-finite twist state.

- [ ] **Step 5: Run repository checks and commit**

Run:

```bash
uvx pre-commit run -a
git diff --check
```

Stage only the four implementation files, leaving `lessons.md` and `solver_convergence.png` untouched:

```bash
git add CHANGELOG.md newton/_src/solvers/limx/constraints/self_collision.py newton/examples/cloth/example_cloth_limx_twist.py newton/tests/test_solver_limx.py
git commit -m "Strengthen LIMX EF untangling"
```
