# Heterogeneous collider feasibility proof

This branch implements **MuJoCo Warp with either native MuJoCo or Newton-generated
contacts**, while keeping the body and joint layout identical across worlds.
It simulates different convex decompositions without changing the original Newton
meshes, shape counts, mass, or inertia. The installed MuJoCo and MuJoCo Warp packages
are unmodified.

This is an executable feasibility result for a bounded feature. The actual Isaac
Lab pick-and-place scene, USD import path, and training throughput have not been
validated.

## Implementation

Base: Newton `45458023`, September 23, 2026.
Branch: `feasibility/heterogeneous-collider-counts`.

```python
solver = newton.solvers.SolverMuJoCo(
    model,
    use_mujoco_contacts=True,  # False selects Newton-generated contacts instead.
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

Native detection exports all unique mesh/scale combinations into a shared asset
bank. A geometry-only MuJoCo spec compiles their exact bounds and mesh frame
corrections once, without compiling a complete articulation for every world.
The solver installs per-world mesh IDs, sizes, bounds, and poses. Small valid
placeholder geometry reduces broadphase work for absent mesh/primitive slots;
unused plane slots remain infinite until filtered. Real geometry stays exact.

Native candidate masks represent the union of allowed pairs where possible;
an inexact mask compilation falls back to conservative masks. Deduplicated
per-world pair tables enforce the actual rules. The native contact callback
stably compacts every contact field before collision wake and constraint
processing, preserving collision overflow diagnostics. It uses preallocated
scratch, a parallel scan, and fused record-copy kernels. CUDA graph execution
skips record copying when all contacts are valid; the public Warp scan still
runs because it cannot be placed inside a CUDA conditional node.

Four production files change relative to the upstream base:

- `newton/_src/utils/selection.py`: optional body/joint-only views.
- `newton/_src/solvers/mujoco/solver_mujoco.py`: slot construction, explicit
  mappings, native asset export, and validation of the experimental mode.
- `newton/_src/solvers/mujoco/kernels.py`: reject unmapped contact endpoints
  before dereferencing geom IDs, and refresh per-world geometry poses on
  bodies welded to the world.
- `newton/_src/solvers/mujoco/heterogeneous.py`: parallel native-contact filtering.

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
| Native MuJoCo collisions | CPU/CUDA trajectories match isolated native worlds; every hull contributes correctly mapped contacts. |
| Native body-to-body collisions | Two convex bodies stack only in worlds where their shape pairs are enabled. |
| Shared meshes with different scales and transforms | Rotated, offset, nonuniformly scaled hulls match isolated native trajectories. |
| Fixed-body collider poses | Distinct collider poses on rotated, translated fixed bodies produce the correct resting heights, including after shape pose updates, on CPU and CUDA. |
| Sleeping | Resting native worlds sleep; a masked reset wakes only the selected world. |
| Contact compaction | Every contact field is preserved bitwise in stable order; overflow flags, empty buffers, and CUDA graph branch changes are covered. |
| Unsupported modes | MuJoCo C dynamics, sites, explicit contact pairs, fluid settings, and different joint parent/child layouts are rejected. Native cones/heightfields and runtime geometry/filter changes remain outside the new native mode. |

The three-world reference comparison uses absolute tolerances of `2e-3` for
body transforms and `2e-2` for spatial velocities. Resting-height tolerance is
`0.015 m`; reset isolation is checked within `0.002 m`.

An independent Newton-contact stacked-body probe measured exact CPU agreement and maximum
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

The native regression also fails against the preceding Newton-contact-only
commit `a2c1c6ba`, which rejects the native configuration. A no-op replacement
of the contact filter fails the compaction regression on CPU and CUDA. The
new CUDA graph test exposed a conditional allocation restriction during
development; moving the public scan outside the conditional fixed it.

Final review exposed a static-body pose bug: MuJoCo Warp skips geometry
kinematics for bodies welded to the world, while Newton previously refreshed
only geometry attached directly to the world body. A CPU/CUDA regression
reproduced a 0.39 m resting-height error. The fix composes each collider's local
pose with its fixed body's pose during initialization and shape updates,
without adding work to the simulation step.

## Performance evidence

Warmed CUDA graph timings use the same original geometry, 50 solver iterations,
20 line-search iterations, `nconmax=128`, and `njmax=512`. Each graph contains
eight steps. Five samples each execute 25 graph launches; the table reports
median milliseconds per simulation step across the entire batch.

| Worlds | Hull distribution | Native contacts | Newton contacts |
| ---: | --- | ---: | ---: |
| 16 | 1–5 hulls | 0.226 | 0.239 |
| 256 | 1–5 hulls | 0.267 | 0.323 |
| 1024 | 1–5 hulls | 0.328 | 0.355 |
| 256 | 1 hull in 15/16 worlds, 8 hulls otherwise | 0.287 | 0.315 |
| 1024 | 1 hull in 15/16 worlds, 8 hulls otherwise | 0.390 | 0.424 |

The homogeneous 1024-world control measured 0.325 ms both with the default native
path and with the opt-in. Identical filtering rules and exact masks omit the
callback entirely. These synthetic resting scenes establish comparable steady
state performance for the tested distributions; they are not robot training or
grasping benchmarks. Some samples varied by about 15%, so small differences are
not evidence of a general speedup.

Construction is more expensive: the warmed 1024-world native case took about
0.165 s to construct the solver versus 0.063 s for Newton contacts. Contact
filtering also allocates scratch proportional to contact capacity. Rejected
contacts still consume broadphase/narrowphase capacity before filtering.
All benchmark cases passed resting-height, finite-state, and collision-overflow
checks. Some worlds retained the existing line-search iteration-limit flag;
this is distinct from a contact-buffer overflow and is not hidden.

The reproducible workload is in
`asv/benchmarks/simulation/bench_heterogeneous_mujoco.py`; it also tracks contact
filter scratch memory. No new dependency or MuJoCo Warp patch is needed.

```bash
uv run --no-sync --with 'asv-runner<0.3.0' \
  asv/benchmarks/simulation/bench_heterogeneous_mujoco.py \
  --world-count 16 256 1024
```

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
  newton.tests.test_mujoco_heterogeneous_body_contacts \
  newton.tests.test_mujoco_heterogeneous_native \
  newton.tests.test_mujoco_heterogeneous_filter -v
```

The final combined validation run completed **178 tests in 110.305 seconds:
176 passed and two were skipped**. It covers both contact generators, selection,
solver validation, geometry properties, collision masks, sleeping, friction,
contact forces, reset, and planar meshes. CPU and CUDA cases are included in
the new selection, geometry-mapping, dynamics, and contact-filter tests.

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
  newton.tests.test_mujoco_heterogeneous_native \
  newton.tests.test_mujoco_heterogeneous_filter \
  newton.tests.test_mujoco_reset \
  newton.tests.test_mujoco_sleeping \
  newton.tests.test_mujoco_margin_zeroing \
  newton.tests.test_mujoco_solver.TestMuJoCoValidation \
  newton.tests.test_mujoco_solver.TestMuJoCoSolver \
  newton.tests.test_mujoco_solver.TestMuJoCoSolverCollisionMasks \
  newton.tests.test_mujoco_solver.TestMuJoCoSolverGeomProperties \
  newton.tests.test_mujoco_solver.TestMuJoCoSolverNewtonContacts \
  newton.tests.test_mujoco_solver.TestFrictionPriority \
  newton.tests.test_mujoco_solver.TestImmovableContactFiltering \
  newton.tests.test_mujoco_solver.TestMuJoCoContactForce \
  newton.tests.test_mujoco_solver.TestUpdateContactsPointPositions \
  newton.tests.test_mujoco_solver.TestMuJoCoConversion.test_static_worldbody_geoms_support_offset_worlds \
  newton.tests.test_mujoco_solver.TestMuJoCoArticulationConversion.test_orphan_world_fixed_body_is_exported_static \
  newton.tests.test_mujoco_solver.TestMuJoCoArticulationConversion.test_orphan_world_fixed_kinematic_body_is_exported_as_mocap \
  newton.tests.test_mujoco_solver.TestMuJoCoSolverKinematicBodyProperties.test_fixed_root_attached_to_world_uses_mocap_and_tracks_pose \
  newton.tests.test_solver_mujoco_planar_mesh -v
```

## PR boundary

The proposed scope is an experimental path for differing collider counts/types
and mesh geometry with identical body/joint topology and either contact generator.
Sites, spatial tendons, explicit MuJoCo contact pairs, and fluid effects remain
unsupported. Native contacts support non-planar convex mesh geometry and
primitives except cones and heightfields. Native geometry sizes, assets, types,
collision groups/exclusions, contact dimensions, and priorities are fixed at
construction; rebuild the solver after changing them. Shape poses and supported
materials remain updateable. Arbitrary different robots per world are outside
this feature.

Per-world fixed-root placement requires roots belonging to an articulation.
A separate CPU probe confirmed that standalone fixed roots retain the template
body pose in both the existing default solver and this experimental mode.
The static-collider correction supports distinct local geometry on those roots;
it does not add fixed-root body placement support. Use an articulated fixed root
for per-world placement and runtime root-pose updates.

MuJoCo's internal viewer and MJCF export show representative slot geometry,
not the distinct geometry of every Newton world. Geometry inspection should
use the original Newton model.

Slot growth depends on the maximum size of each compatible per-body group.
Production evaluation still needs the actual asset distribution, representative
robot memory and throughput measurements, and an Isaac Lab integration that selects the new
view/solver options. Robot control, grasping success, sensor/site support, and
long training runs are not established by these synthetic tests.

The evidence supports opening a narrowly scoped PR or design discussion linked
to Newton issue #3461. It does not establish that the entire broader issue is
resolved, nor does it establish a requirement to redesign MuJoCo Warp itself.
