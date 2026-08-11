# LIMX EE Endpoint Coverage Experiment Design

## Goal

Determine whether endpoint edge-edge candidates are redundant for the current
triangle-cloth collision path because unsigned vertex-face closest-point
contacts already cover the same point-edge and point-point feature regions.
The experiment must characterize coverage before changing production contact
generation.

## Scope

The claim applies only to triangle surface meshes using
`use_outward_normals=False`. In this mode, VF uses the closest point on the
closed triangle and therefore may return interior, edge, or vertex features.
The experiment does not generalize the result to oriented projected VF,
standalone segments, rods, or other codimensional geometry.

No production collision response, public API, scene parameter, or contact
filter changes in this experiment.

## Candidate Classification

Classify each detected EE candidate from its closest-point parameters `s` and
`t`:

- interior-interior: strict EE;
- endpoint-interior or interior-endpoint: PE;
- endpoint-endpoint: PP.

Use the same endpoint convention as the closest-point routine. Convert PE and
PP candidates to canonical feature keys:

- PE: `(point_vertex, canonical_edge_id)`;
- PP: `(min(vertex_0, vertex_1), max(vertex_0, vertex_1))`.

Classify each unsigned VF closest point from its barycentric weights. Convert
edge and vertex results to the same PE and PP keys. Interior VF remains PT and
is not part of exact-key matching, but it may provide a closer surface contact
than an endpoint EE.

## Coverage Rule

For every endpoint EE candidate within the active thickness, require at least
one of the following:

1. an exact canonical PE or PP key produced by VF; or
2. a VF contact involving the same endpoint vertex whose surface distance is
   no greater than the endpoint EE distance within numerical tolerance.

The second rule handles cases where the closest point on an incident triangle
lies in its interior and is closer than the edge feature selected by the EE
parent stencil.

Record unmatched candidates rather than silently discarding them. Each record
contains particle and edge IDs, closest parameters, EE distance, nearest VF
distance, topology-local status, and source scene.

## Experiments

### Synthetic sweeps

Construct CUDA tests that sweep one triangle-surface vertex past:

- the interior of a target edge;
- a target edge endpoint;
- a shared edge of two triangles;
- a boundary edge with one incident triangle.

Sample positions on both sides of each feature boundary. At every sample,
record VF count, endpoint EE count, canonical keys, total contact force, and
contact Hessian trace. Include a deliberately oriented-VF control demonstrating
that the coverage claim is not valid for that mode.

### Existing cloth scenes

Run bounded diagnostic rollouts on the LIMX twist and three-T-shirt scenes.
For every frame, count strict EE, endpoint PE, endpoint PP, exactly covered
endpoint candidates, closer-VF-covered candidates, and unmatched candidates.
Do not use aggregate count alone: retain the first unmatched candidate for
inspection.

## Success Criteria

The hypothesis is supported only if:

- every synthetic unsigned-VF endpoint EE is covered;
- `unmatched == 0` throughout both scene rollouts;
- all buffers remain finite and do not overflow;
- the diagnostic does not alter positions, velocities, forces, or solver
  iteration counts.

If these conditions pass, a later implementation may reject endpoint EE and
leave PE/PP ownership to VF. That production change requires a separate RED to
GREEN plan and must compare contact force and settling behavior before and
after the rejection.

If any endpoint EE is unmatched, preserve endpoint EE and use the recorded
counterexample to design explicit feature ownership or canonical-key
deduplication instead.

## Validation

Use CUDA-only `unittest` coverage. Verify the diagnostic classifier and key
construction on fixed geometries, then run focused scene diagnostics. Report
counts and representative failures; do not infer coverage from visual output
alone.
