# LIMX Surface-Vertex Collision Radii Design

## Goal

Replace the bunny scene's uniform self-collision thickness with fixed
per-surface-vertex radii derived from rest geometry, while retaining IPC's
strict incident-pair filtering.

## Radius Construction

For every surface vertex `i`, define the one-sided radius

```text
r_i = min(nominal_thickness / 2, 0.25 * h_i)
```

where `h_i` is the minimum altitude of the rest-state surface triangles
incident to `i`. Compute these radii once from rest geometry. Interior
tetrahedral particles are not collision primitives and receive radius zero in
geometry-aware mode.

## Pair Thickness

Keep the existing contact interpolation:

- VF thickness is the candidate vertex radius plus the barycentric
  interpolation of the target face radii.
- EE thickness is the sum of the closest-point interpolations on both edges.

The nominal `3 mm` thickness remains a global upper bound, not a uniform
separation target. Radii must not be recomputed from deformed geometry because
a moving contact target can inject energy.

## Validation

- Geometry-aware construction accepts tetrahedral meshes with interior
  particles and assigns zero radius to particles absent from the surface.
- Non-finite or degenerate referenced surface triangles remain invalid.
- The bunny scene uses `geometry_radius_scale=0.25` and has no VF/EE contacts
  in its initial separated rest configuration.
- The 300-frame CUDA rollout remains finite, positive-volume, overflow-free,
  and generates cross-bunny contact after falling.

