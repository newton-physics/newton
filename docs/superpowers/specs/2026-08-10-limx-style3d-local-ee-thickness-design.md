# LIMX Style3D-Local EE Thickness Design

## Goal

Run the volumetric bunny collision scene with Style3D's pair-specific
thickness policy: use one uniform collision thickness for ordinary VF/EE
pairs and cap only topology-local EE pairs by their current edge lengths.

## Contact Policy

- Keep the nominal collision thickness at `0.003 m`.
- VF excludes only strictly incident vertex-triangle pairs and otherwise uses
  the uniform nominal thickness.
- EE excludes only pairs that share an endpoint.
- For a retained topology-local EE pair, use
  `min(thickness, (length_0 + length_1) / 4)`.
- Every other retained EE pair uses the uniform nominal thickness.

The existing opt-in geometry-radius API remains available, but the bunny
scene must not enable it. This isolates the Style3D policy without changing
public API or unrelated experiments.

## Validation

- The bunny example reports `geometry_radius_scale is None` and keeps all
  other solver, collision, friction, and time-integration settings unchanged.
- Existing focused CUDA tests continue to prove that topology-local EE uses
  the half-average-edge-length cap and nonlocal EE remains active under the
  nominal thickness.
- The 300-frame bunny rollout remains finite, positive-volume,
  overflow-free, and reaches cross-bunny contact.
- Launch the interactive bunny example so the resulting motion can be
  inspected visually.
