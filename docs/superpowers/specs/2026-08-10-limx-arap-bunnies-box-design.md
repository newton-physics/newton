# LIMX ARAP Bunnies-in-a-Box Design

## Goal

Add a LIMX example that drops eight volumetric ARAP bunnies into an open box
to exercise soft-body/soft-body surface collision. The scene must demonstrate
actual cross-bunny vertex-face (VF) or edge-edge (EE) contacts while retaining
the existing `0.01 s`, one-Newton-step integration path.

## Chosen Collision Approach

Build all eight disconnected bunnies in one particle model and pass the
combined surface triangle mesh to one `ConstraintSelfCollision`. Because all
particles share one Newton world, its BVHs detect both within-bunny and
cross-bunny VF/EE contacts. The collision force, Hessian-vector product, and
diagonal then participate in the same global PCG solve as every bunny's ARAP
elasticity.

Derive collision vertices exclusively from the unique indices referenced by
the combined boundary triangle array. Interior tetrahedral vertices participate
in ARAP elasticity but never become VF candidates and are never tested against
the five box planes. EE and EF candidates already come from boundary triangle
edges and faces. Disable EF recovery for this scene: only VF and EE contacts
are detected or assembled into force and Hessian terms.

Use a fixed nominal two-surface collision thickness of `0.003 m`. Do not set
`geometry_radius_scale`; investigating geometry-aware thickness is explicitly
deferred. The fixed-thickness baseline is intentional even though the bunny
surface tessellation is nonuniform.

Rejected alternatives are:

- geometry-aware radii, because the user chose to investigate that later;
- independent per-bunny solvers, because they cannot assemble cross-object
  contact Hessians into one Newton system;
- particle-particle collision, because it does not test the requested VF/EE
  surface interaction.

## Scene Layout

Create:

```text
newton/examples/softbody/example_softbody_limx_arap_bunnies_box.py
```

Load the existing self-contained asset:

```text
newton/examples/assets/bunny_tet.npz
```

Instantiate eight copies in a `2 x 2 x 2` arrangement. Each copy uses:

```text
uniform scale          0.15
density                1000 kg/m^3
ARAP stiffness         1e5 Pa
material damping       0
```

Use the following deterministic centers, world-space rotation perturbations,
and linear velocities. Compose each perturbation after the positive 90-degree
X rotation that converts the source Y-up mesh to Newton Z-up coordinates.

| Copy | Center [m] | XYZ perturbation [deg] | Velocity [m/s] |
|---:|---|---|---|
| 0 | `(-0.11, -0.17, 0.25)` | `(0, 0, 0)` | `(0.08, 0.04, -0.10)` |
| 1 | `( 0.11, -0.17, 0.25)` | `(4, -3, 6)` | `(-0.06, 0.03, -0.12)` |
| 2 | `(-0.11,  0.17, 0.25)` | `(-5, 4, -7)` | `(0.05, -0.04, -0.08)` |
| 3 | `( 0.11,  0.17, 0.25)` | `(3, 5, 4)` | `(-0.07, -0.03, -0.11)` |
| 4 | `(-0.11, -0.17, 0.52)` | `(-4, -5, 8)` | `(0.06, 0.02, -0.20)` |
| 5 | `( 0.11, -0.17, 0.52)` | `(5, 3, -5)` | `(-0.05, 0.04, -0.18)` |
| 6 | `(-0.11,  0.17, 0.52)` | `(2, -6, 6)` | `(0.04, -0.03, -0.22)` |
| 7 | `( 0.11,  0.17, 0.52)` | `(-3, 4, -8)` | `(-0.06, -0.02, -0.19)` |

These separations leave positive gaps at frame zero while forcing the two
layers to pile up after release. Pass `add_surface_mesh_edges=False` because
`ConstraintSelfCollision` reconstructs its own complete surface edge
adjacency from the triangle array.

The combined model contains exactly:

```text
particles              14,952
tetrahedra             58,848
surface triangles      17,216
```

Construct one `ConstraintTetrahedronARAP` over the combined tetrahedron array.
The builder-generated surface triangles remain disconnected by index, which
lets contact IDs be mapped back to bunny number by integer division with
`1,869` particles per copy.

## Open Box

Use a floor and four walls with an open top. The exact interior is:

```text
X bounds               [-0.36, 0.36] m
Y bounds               [-0.40, 0.40] m
wall height            0.75 m
floor height           0 m
```

Render the floor and walls as five static `body=-1` box shapes. Use `0.05 m`
floor half-thickness and `0.025 m` wall half-thickness, placing the visible
inner faces on the exact bounds above. Physical response comes from five
inward-facing `ConstraintStaticPlaneContact` operators, not the rendered
shapes. Pass the unique boundary-triangle vertex indices to every plane so the
box does not apply direct forces or Hessian blocks to interior tetrahedral
vertices. Each plane uses:

```text
thickness              0.003 m
stiffness              2e4 N/m
normal damping         0
friction               0.05
friction epsilon       1e-4 m
```

## Soft-Body Collision

Configure the shared `ConstraintSelfCollision` with:

```text
thickness              0.003 m
geometry radius scale  None
fixed stiffness        None
stiffness factors      (0.5, 0.3, 1.5), with the EF entry unused
edge-face recovery     disabled
friction               0.05
friction epsilon       1e-2 m/s
maximum contacts       262,144 per contact type
```

The stiffness is adaptive to the assembled ARAP diagonal, particle masses,
and `dt`; only stiffness adapts. Collision thickness remains fixed at `3 mm`.
Only VF and EE are active; the EF contact buffer remains empty and contributes
no force, Hessian-vector product, or diagonal block.

Compose the self-collision operator and five box planes with
`ConstraintGroupDynamic`.

## Integration

Use:

```text
dt                     0.01 s
steps per frame        1
nonlinear iterations   1
PCG iterations         50
velocity damping       1.0
```

Do not add substeps, line search, material damping, normal damping, or
post-solve velocity damping. Use the same state swap pattern as the single
bunny example.

## Verification

The example implements `test_post_step()` and `test_final()`.

After every tested frame, verify:

1. all positions and velocities are finite;
2. all 58,848 current tetrahedron determinants remain strictly positive;
3. only the 8,624 unique boundary vertices are collision candidates;
4. particles below the open top have not escaped catastrophically through
   the floor or four walls;
5. VF and EE contact buffers report zero overflow, and EF remains disabled.

Read active VF and EE contact IDs and map their particle indices to bunny
numbers. The run succeeds only after at least one contact contains particles
from different bunnies, proving that the example tests soft-body/soft-body
collision rather than only bunny self-collision or box contact.

Automated coverage remains focused:

- one configuration test checks counts, fixed thickness, disabled geometry
  radius scaling, contact capacity, and 50 PCG iterations;
- one CUDA smoke run crosses the first pile-up;
- a direct 300-frame CUDA run checks inversion, containment, overflow, and a
  recorded cross-bunny VF/EE contact.

If the fixed `3 mm` baseline jitters or becomes unstable, preserve and show
that result. Diagnose contact counts, overflow, penetration, and speed history;
do not switch to geometry-aware thickness without a separate user decision.

## Scope

The implementation reuses the existing bunny asset and adds a surface-vertex
subset option to static-plane contact. It also fixes VF detection to derive its
candidate vertices from the triangle topology. It adds no dependency,
screenshot, or README entry until the runtime result is visually accepted.
