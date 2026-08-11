# LIMX ARAP Bunny-on-Cloth Drop Design

## Goal

Add a CUDA LIMX example in which one deformable tetrahedral ARAP bunny falls
onto a four-corner-pinned cloth sheet. The scene must visibly demonstrate
two-way deformable contact: the bunny deforms on impact while its weight bends
and stretches the cloth without a supporting table or ground plane.

## Unified Model

Build the bunny and cloth in one `ModelBuilder` so one `SolverLIMX` advances
all particles and contact impulses act on both materials in the same Newton
system.

Create the bunny first from `newton/examples/assets/bunny_tet.npz` using the
same source-to-world rotation, `0.15` scale, `1000 kg/m^3` density, and ARAP
tetrahedron setup as `softbody_limx_arap_bunny_table`. Place its center above
the cloth with zero initial velocity.

Append a horizontal `0.8 m` square cloth centered below the bunny at
`z = 0.45 m`. Use a regular `40 x 40` cell grid with alternating diagonals,
giving `41 x 41` particles. Assign a total cloth mass of `0.3 kg`. Pin exactly
the four corner particles; do not pin complete edges.

Track bunny and cloth particle, triangle, and tetrahedron ranges explicitly.
All elasticity constraints use global particle indices but only their own
element ranges.

## Elasticity

Use these static constraints:

- bunny tetrahedra: `ConstraintTetrahedronARAP`, stiffness `1.0e5`;
- cloth corners: `ConstraintAnchor`, stiffness `1.0e7 N/m`;
- cloth triangles: `ConstraintTriangleElastic`, stiffness
  `(1.0e4, 1.0e4, 1.0e3)`;
- cloth interior edges: `ConstraintDihedralBending`, stiffness `1.0e-4`.

The cloth bending value follows the visually compliant LIMX garment setting
instead of the stiffer introductory hanging-cloth value.

Use gravity `(0, 0, -9.81) m/s^2`, `dt = 0.01 s`, one Newton iteration, 50 PCG
iterations, and velocity damping `1.0`.

## Contact

Use one `ConstraintSelfCollision` over the combined triangle surface. Keep
`use_outward_normals=False` because the cloth is an open surface. The current
contact path therefore provides closest-point VF boundary coverage and strict
interior-only EE; EF recovery remains enabled.

Use automatic nominal thickness capped by the existing 5 mm limit,
`geometry_radius_scale=0.25`, and
`geometry_radius_topology_local_only=True`. Use adaptive stiffness factors
`(VF=0.5, EE=0.3, EF=1.5)`, zero friction for the first experiment, and a
contact capacity of 262,144 per type.

Do not add a table, ground plane, SDF collider, CCD, contact damping, or extra
substeps. A bunny that passes through the cloth must remain visible as a failed
contact result rather than being caught by another surface.

## Example and Rendering

Create
`newton/examples/multiphysics/example_softbody_limx_arap_bunny_cloth.py` with
the standard `Example` interface. Register the command:

```bash
uv run -m newton.examples softbody_limx_arap_bunny_cloth
```

Render the bunny and cloth from the shared model. Frame the camera so all four
anchors, the initial bunny height, and maximum cloth sag remain visible. The
default run is 300 frames.

After the simulation is validated, add a `320 x 320` JPEG to the examples
documentation and register the example in the relevant README gallery.

## Validation

Add CUDA `unittest` coverage that verifies:

- the model contains the expected bunny and cloth element ranges;
- exactly four cloth corner anchors are used;
- the solver uses one Newton iteration, 50 PCG iterations, and no damping;
- the collision operator uses unsigned VF, interior-only EE, geometry-aware
  local radii, and sufficient capacity;
- one captured step is finite;
- a 300-frame rollout has no contact overflow or non-finite state;
- every bunny tetrahedron retains positive signed volume;
- the bunny center descends and reaches the cloth region;
- the cloth center sags below its rest height;
- at least one cross bunny-cloth VF or EE contact occurs;
- neither component leaves a bounded scene volume.

The test must not require the bunny to settle within 300 frames. Visual
acceptance remains the final check for contact quality, sag, and penetration.

## Non-goals

- Rigid-body or articulated bunny dynamics.
- A second solver or proxy coupling layer.
- Ground, table, or box collision.
- Friction tuning or long-term settling.
- Changes to the public LIMX API or collision formulation.
