# LIMX ARAP Bunny Table-Drop Design

## Goal

Add a second LIMX ARAP example in which a volumetric Stanford bunny falls
`0.15 m` onto a static table. The scene is a stability test for the existing
tetrahedral ARAP constraint and static-plane penalty contact at a `0.01 s`
time step with exactly one Newton iteration.

The example must make instability visible rather than hiding it with velocity,
material, or normal-contact damping. It reuses the solver and contact code that
already exists; it does not introduce another collision algorithm.

## Chosen Approach

Use an infinite one-sided contact plane for physics and a finite static box for
visualization.

This separates the question being tested—whether ARAP plus the existing
penalty force and Hessian remain stable—from finite-table edge and corner
contact. A multi-plane box would double-count edge contacts, while a rigid
triangle-mesh VF/EE implementation would be a separate collision-system
project.

The visible box and physical plane share the same top height, `z = 0`. The
static box is added to the particle model with body index `-1`, so it does not
create a rigid body and remains compatible with `SolverLIMX`'s particle-only
restriction.

## Volumetric Bunny Asset

Newton's existing `newton/examples/assets/bunny.usd` contains only a surface
mesh and cannot supply volumetric ARAP elements. Use libuipc's precomputed
tetrahedral bunny:

```text
/home/limx/github/libuipc/assets/sim_data/tetmesh/bunny0.msh
```

The inspected source mesh contains:

- 1,869 vertices;
- 7,356 tetrahedra;
- 2,152 boundary triangles;
- strictly positive rest determinants, with minimum determinant
  `3.721915e-06`;
- native bounds `[-0.459488, -0.666513, -0.959004]` to
  `[0.466635, 0.776868, 1.01567]`.

Convert it once to:

```text
newton/examples/assets/bunny_tet.npz
```

using `newton.TetMesh.create_from_file()` and `TetMesh.save()`. The committed
example reads only the `.npz`, so normal execution uses NumPy and does not
require `meshio` or the external libuipc checkout.

Record provenance and the Apache-2.0 source license in:

```text
newton/licenses/libuipc-bunny-tet.txt
```

The notice identifies:

- source repository `https://github.com/spiriMirror/libuipc.git`;
- inspected source revision
  `619a5412cf958eb59cf3c9cebc9a8e8e9625ebd3`;
- first asset revision
  `32988ff1237d74dc0dd0eef2bec9ee6f8898c7a2`;
- source path `assets/sim_data/tetmesh/bunny0.msh`;
- conversion as a topology-preserving `.msh` to `.npz` container change.

After conversion, validate that vertex and tetrahedron counts, connectivity,
positive orientation, and surface-triangle count match the source.

## Scene Geometry and Material

Add:

```text
newton/examples/softbody/example_softbody_limx_arap_bunny_table.py
```

Use Z-up Newton coordinates. The libuipc mesh is Y-up, so rotate it by positive
90 degrees about X, mapping native Y to world Z. Instantiate it with:

```text
uniform scale          0.15
world translation      (0, 0, 0.25) m
density                1000 kg/m^3
ARAP stiffness kappa   1e5 Pa
```

The scaled native minimum Y is approximately `-0.10 m`; the translation
therefore places the initial bottom approximately `0.15 m` above the table.
The bunny's horizontal footprint remains well inside the table.

Render a table box centered at `(0, 0, -0.03) m` with half-extents
`(0.5, 0.5, 0.03) m`, so its top is exactly `z = 0`. The table is visual only;
the plane operator owns its physical response.

`ModelBuilder.add_soft_mesh()` generates positive particle masses and the
surface triangles used by the viewer. Its Neo-Hookean material arrays are set
to zero because LIMX uses the explicit `ConstraintTetrahedronARAP` stiffness,
not `Model.tet_materials`.

## Contact and Integration

Use `ConstraintStaticPlaneContact` as `SolverLIMX.dynamic_operator` with:

```text
normal                (0, 0, 1)
offset                0 m
thickness             0.003 m
stiffness             2e4 N/m
normal_damping        0
friction              0.05
friction_epsilon      1e-4 m
```

The `2e4 N/m` stiffness matches the existing LIMX table-contact baseline. Its
force and exact PSD normal Hessian participate in the same Newton system as
inertia and ARAP. A particle at the contact thickness sees zero normal force;
penetration increases the force linearly.

Use:

```text
dt                     0.01 s
steps per frame        1
nonlinear_iterations   1
linear_iterations      128
velocity_damping       1.0
```

There is no line search, substepping, material damping, normal damping, or
self-collision. Friction remains at the previously chosen low value `0.05`.

## Runtime and Rendering

The example follows the standard `Example` interface. Each `step()`:

1. clears particle forces;
2. applies interactive viewer forces;
3. calls one `SolverLIMX.step(..., 0.01)`;
4. swaps the input and output states once;
5. records stability diagnostics.

The camera frames the complete bunny and table during the drop and after
impact. The first interactive run uses at least 300 frames (`3 s`). After
visual acceptance, add a 320-by-320 JPEG and register the example in the
README Softbody table.

## Stability Diagnostics

The scene records initial rest determinants and, after each frame, checks:

1. all positions and velocities are finite;
2. every current tetrahedron determinant remains strictly positive;
3. the lowest particle stays above `z = -0.015 m`, limiting catastrophic
   plane penetration to `15 mm`;
4. the center of mass stays within a bounded scene volume;
5. maximum particle speed remains finite and does not grow without bound
   after impact.

For a 300-frame stability run, retain the maximum speed and center-of-mass
height over the last 50 frames. The result is considered visually stable when
the bunny remains on the table without inversion or increasing-amplitude
motion. Do not add damping merely to satisfy the diagnostic; if the undamped
baseline fails, show the result and diagnose contact stiffness, time-step, or
one-Newton linearization separately.

Automated coverage stays focused:

- one asset-integrity test validates the converted topology and orientation;
- one CUDA example smoke run crosses first impact and exercises contact;
- `test_post_step()` checks finite state, orientation, and penetration;
- `test_final()` checks that the bunny fell toward and remained near the table.

The 300-frame run is also executed directly with the null viewer before the
interactive review.

## Files and Public Surface

Create or modify only:

```text
newton/examples/assets/bunny_tet.npz
newton/licenses/libuipc-bunny-tet.txt
newton/examples/softbody/example_softbody_limx_arap_bunny_table.py
newton/tests/test_example_softbody_limx_arap_bunny_table.py
newton/tests/test_examples.py
docs/images/examples/example_softbody_limx_arap_bunny_table.jpg
README.md
CHANGELOG.md
```

No new public Python symbol, dependency, solver path, or API documentation is
required.

## Acceptance Criteria

The scene is complete when:

- it loads a self-contained tetrahedral bunny without runtime `meshio` or
  libuipc-path dependency;
- it uses the approved `3 mm` contact thickness, zero normal damping, and
  `0.05` friction;
- it advances at `0.01 s` with one Newton iteration and no substeps;
- a 300-frame CUDA run remains finite, positive-volume, and within the
  penetration bound;
- the interactive example is launched for visual stability review;
- after acceptance, its screenshot and README registration are committed.
