# Heterogeneous collider feasibility proof

The implementation is feasible for **MuJoCo Warp with Newton-generated contacts**,
while keeping the body and joint layout identical across worlds. A local prototype
simulates different convex decompositions without changing the original Newton
meshes, shape counts, mass, or inertia. The installed MuJoCo and MuJoCo Warp packages
are unmodified.

This is an executable feasibility result for a bounded feature. The actual Isaac
Lab pick-and-place scene, USD import path, and training throughput have not been
validated.

## Prototype

Base: Newton `45458023`, September 23, 2026.
Branch: `feasibility/heterogeneous-collider-counts`.

```python
solver = newton.solvers.SolverMuJoCo(
    model,
    use_mujoco_contacts=False,
    allow_heterogeneous_shapes=True,
)
view = newton.selection.ArticulationView(model, "*", include_shapes=False)
```

The view retains rectangular body/joint access and masked state updates, while
omitting shape metadata and rejecting shape-frequency attribute access. Its
default behavior remains strict.

The solver exports internal geometry slots grouped by body, geometry type,
contact dimension, and priority. It allocates the maximum number needed per
group and records an explicit mapping from each world's slots to its actual
Newton shapes. Missing slots map to `-1`. Newton generates contacts against the
original geometry; those contact endpoints map to the corresponding MuJoCo
body and slot. Contact materials and other supported properties continue to
synchronize per world.

Only three production files change:

- `newton/_src/utils/selection.py`: optional body/joint-only views.
- `newton/_src/solvers/mujoco/solver_mujoco.py`: slot construction, explicit
  mappings, and validation of the experimental mode.
- `newton/_src/solvers/mujoco/kernels.py`: reject unmapped contact endpoints
  before dereferencing geom IDs.

The new flags are runtime configuration. Existing USD collider authoring is
unchanged, so this prototype needs no new USD schema.

## Executable evidence

| Experiment | Required observable result |
| --- | --- |
| Three worlds with 1, 3, and 2 convex pieces, with half-heights 0.1, 0.2, and 0.3 m | Each settles at its own height and matches a separately simulated reference at four trajectory checkpoints, on CPU and CUDA. |
| Original geometry and inertial properties | Shape count, mesh vertices, mass, and inertia remain unchanged; every original hull contributes ground contacts. |
| Masked reset | Resetting the middle world leaves the other two worlds undisturbed. |
| Shapeless body in one world | It falls through the ground while bodies in other worlds settle, proving missing slots create no phantom collider. |
| Sixteen distinct variants with 1–5 convex pieces each | All settle at their individual heights during CUDA graph replay. |
| Two stacked moving bodies with swapped 1/3 and 3/1 hull distributions | Correct shape-to-body mapping, body-to-body contacts, world isolation, and agreement with isolated reference simulations on CPU and CUDA. |
| Mixed primitive types and contact settings across two bodies | Every real shape maps to the correct world/body, contact dimension, and priority, including shapes absent from world zero. |
| Unsupported modes | Native contacts, MuJoCo C dynamics, sites, explicit contact pairs, fluid settings, and different joint parent/child layouts are rejected. |

The three-world reference comparison uses absolute tolerances of `2e-3` for
body transforms and `2e-2` for spatial velocities. Resting-height tolerance is
`0.015 m`; reset isolation is checked within `0.002 m`.

An independent stacked-body probe measured exact CPU agreement and maximum
CUDA differences of `1.62e-5` in transform components and `2.96e-4` in spatial
velocity components. These are numerical comparisons, not throughput benchmarks.
Some CUDA contact scenes emitted line-search iteration-limit warnings; the
finite-state, trajectory, and resting-height assertions still passed.

The new selection and main dynamics regressions were also run against a clean
checkout of the base revision. They fail without the new implementation.
The existing solver rejects unequal shape counts. A separate negative control
with equal total counts but swapped per-body collider counts demonstrates that
the original flat mapping sends two colliders to the wrong body; the explicit
mapping corrects both.

Independent review found and fixed a legacy single-world attachment bug in the
first prototype. A regression now checks that global-world dynamic shapes retain
their actual body attachment.

## Reproduce

The proof used Python 3.12.13, Warp 1.17.0, MuJoCo 3.12.0, MuJoCo Warp 3.12.0,
and an NVIDIA RTX PRO 3000 Blackwell laptop GPU (driver 595.91.07).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[sim]'
uv run --no-sync -m unittest \
  newton.tests.test_selection_heterogeneous \
  newton.tests.test_mujoco_heterogeneous_shapes \
  newton.tests.test_mujoco_heterogeneous_mapping \
  newton.tests.test_mujoco_heterogeneous_body_contacts -v
```

The existing selection, solver-validation, Newton-contact, friction, contact
force, reset, and planar-mesh tests are included in the broader focused
validation run: **117 tests passed in 127.497 seconds**, comprising 18 added
tests and 99 existing tests. CPU and CUDA cases are included in the new
selection, geometry-mapping, and dynamics tests.

`uvx pre-commit run -a` passed, as did a separate pre-commit run explicitly
covering all newly added files. `git diff --check` passed. The full repository
test suite was not run.

The complete focused regression command is:

```bash
uv run --no-sync -m unittest \
  newton.tests.test_selection \
  newton.tests.test_selection_heterogeneous \
  newton.tests.test_mujoco_heterogeneous_shapes \
  newton.tests.test_mujoco_heterogeneous_mapping \
  newton.tests.test_mujoco_heterogeneous_body_contacts \
  newton.tests.test_mujoco_reset \
  newton.tests.test_mujoco_solver.TestMuJoCoValidation \
  newton.tests.test_mujoco_solver.TestMuJoCoSolverNewtonContacts \
  newton.tests.test_mujoco_solver.TestFrictionPriority \
  newton.tests.test_mujoco_solver.TestImmovableContactFiltering \
  newton.tests.test_mujoco_solver.TestMuJoCoContactForce \
  newton.tests.test_solver_mujoco_planar_mesh -v
```

## PR boundary

The proposed scope is an experimental path for differing collider counts/types
and mesh geometry with identical body/joint topology and Newton collision
detection. Sites, spatial tendons, explicit MuJoCo contact pairs, and fluid
effects remain unsupported. Native MuJoCo collision detection and arbitrary
different robots per world are outside this feature.

MuJoCo's internal viewer and MJCF export show representative slot geometry,
not the distinct geometry of every Newton world. Geometry inspection should
use the original Newton model.

Slot growth depends on the maximum size of each compatible per-body group.
Production evaluation still needs the actual asset distribution, memory and
throughput measurements, and an Isaac Lab integration that selects the new
view/solver options. Robot control, grasping success, sensor/site support, and
long training runs are not established by these synthetic tests.

The evidence supports opening a narrowly scoped PR or design discussion linked
to Newton issue #3461. It does not establish that the entire broader issue is
resolved, nor does it establish a requirement to redesign MuJoCo Warp itself.
