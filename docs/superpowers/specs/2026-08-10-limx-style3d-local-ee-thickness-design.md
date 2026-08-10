# LIMX Topology-Local Collision Thickness Design

## Goal

Run the volumetric bunny collision scene with pair-specific thickness: use one
uniform collision thickness for ordinary VF/EE pairs, and apply rest-geometry
caps only to topology-local pairs. This extends Style3D's local EE treatment
to oriented VF because closed tetrahedral surfaces have signed local contacts.

## Contact Policy

- Keep the nominal collision thickness at `0.003 m`.
- VF excludes only strictly incident vertex-triangle pairs. A retained VF pair
  is topology-local when the candidate vertex shares a mesh edge with any
  target-face vertex.
- EE excludes only pairs that share an endpoint.
- For topology-local VF, use the candidate radius plus the barycentric
  interpolation of the face radii.
- For topology-local EE, use the minimum of the nominal thickness, the two
  interpolated edge radii, and Style3D's
  `(length_0 + length_1) / 4` bound.
- Every other retained VF/EE pair uses the uniform nominal thickness.

Add an opt-in `geometry_radius_topology_local_only` scope while preserving the
existing global geometry-radius behavior by default. The bunny scene enables
both `geometry_radius_scale=0.25` and the topology-local-only scope.

## Validation

- Focused CUDA tests prove that topology-local VF/EE uses geometry radii while
  nonlocal VF/EE remains active under the nominal thickness.
- Existing focused CUDA tests continue to prove that topology-local EE also
  uses the half-average-edge-length cap.
- The bunny example enables topology-local geometry radii and keeps all other
  solver, collision, friction, and time-integration settings unchanged.
- The 300-frame bunny rollout remains finite, positive-volume,
  overflow-free, and reaches cross-bunny contact.
- Launch the interactive bunny example so the resulting motion can be
  inspected visually.
