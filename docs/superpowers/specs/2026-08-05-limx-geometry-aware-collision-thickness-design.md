# LIMX Geometry-Aware Cloth Collision Thickness Design

## Goal

Prevent persistent VF/EE active-set chatter caused by applying one oversized
global self-collision thickness to an irregular cloth mesh. Preserve the
existing physical-thickness input as an upper bound, while deriving fixed
one-sided collision radii from the cloth rest geometry.

The diagnosed 74-vertex sleeve patch must settle with self-collision enabled.
The change must not rely on velocity damping, extra Newton iterations, or
removing VF/EE collision support.

## Scope

This change covers proximity-based vertex-face (VF) and edge-edge (EE)
self-collision in `ConstraintSelfCollision`.

It does not change:

- edge-face (EF) intersection untangling;
- the adaptive VF/EE/EF stiffness formulas;
- the IPC near-parallel EE mollifier;
- contact capacity or broad-phase topology;
- time integration, damping, Newton iterations, or PCG iterations;
- the default behavior of existing `ConstraintSelfCollision` callers.

## Public Interface

Append one optional constructor argument:

```python
ConstraintSelfCollision(
    model,
    thickness,
    stiffness,
    untangle_stiffness=None,
    max_contacts=32768,
    stiffness_factors=None,
    geometry_radius_scale=None,
)
```

`thickness` remains the nominal two-surface activation distance [m]. Existing
callers that omit `geometry_radius_scale` retain the current uniform-thickness
behavior.

When `geometry_radius_scale` is a positive finite float, the operator treats
`0.5 * thickness` as the maximum one-sided radius and caps each particle's
radius using the rest mesh. The value is dimensionless; `0.25` is the initial
recommended value.

Expose the resulting device array as `particle_radii`, annotated
`wp.array[float]`. It contains one-sided collision radii [m], shape
`[particle_count]`, for diagnostics and tests.

Reject a non-finite or non-positive `geometry_radius_scale`. Geometry-aware
mode also rejects non-finite rest positions, degenerate rest triangles, and
particles that are not referenced by a triangle, because no local surface
scale can be estimated for them. These checks do not affect the default mode.

## Rest-Geometry Radius Estimate

Compute the estimate once in the constructor from the model's rest particle
positions and triangle topology. It remains fixed throughout the simulation.
Using rest geometry avoids feeding current deformation back into the contact
activation threshold.

For each triangle, compute its minimum altitude:

```text
twice_area       = ||(x1 - x0) × (x2 - x0)||
maximum_edge     = max(||x1-x0||, ||x2-x1||, ||x0-x2||)
triangle_scale   = twice_area / maximum_edge
```

For every particle `i`, define `local_scale_i` as the minimum
`triangle_scale` over its incident triangles. Then compute:

```text
nominal_radius = 0.5 * thickness
radius_i = min(nominal_radius, geometry_radius_scale * local_scale_i)
```

This uses the minimum altitude rather than only edge length so skinny
triangles reduce the collision envelope they can geometrically support. The
minimum over the one-ring makes the estimate local instead of allowing a
global median to hide a small problematic region.

## Contact Activation

The existing broad phase continues to expand by nominal `thickness`. Since
every local threshold is at most nominal `thickness`, this is conservative and
cannot omit a contact admitted by the narrow phase.

### Vertex-face

For a projected point with face barycentric coordinates `(b0, b1, b2)`, use:

```text
face_radius = b0*r0 + b1*r1 + b2*r2
effective_thickness = rv + face_radius
depth = effective_thickness - abs(signed_distance)
```

Retain the current inside-triangle test, direction, weights, and force/Hessian
model. Emit the contact only when the distance is positive and smaller than
`effective_thickness`.

### Edge-edge

For closest-point parameters `(s, t)`, use:

```text
radius_a = (1-s)*ra0 + s*ra1
radius_b = (1-t)*rb0 + t*rb1
effective_thickness = radius_a + radius_b
depth = effective_thickness - distance
```

Retain the current endpoint rejection, topology filtering, IPC mollifier,
direction, and weights.

In geometry-aware mode, do not apply the existing current-edge-length
one-ring clamp. The fixed local radii replace that time-varying threshold. In
default uniform mode, preserve the existing one-ring clamp exactly for
backward compatibility.

### Edge-face untangling

Keep the existing nominal-thickness EF recovery unchanged. EF handles an
already intersecting edge and face rather than proximity activation, and it
is not active in the diagnosed chatter case.

## Example Behavior

Update `cloth_limx_ee_chatter` so the orange patch enables geometry-aware
self-collision with:

```python
thickness=0.006
geometry_radius_scale=0.25
```

Keep the blue no-collision control, zero gravity, zero initial velocity,
material coefficients, anchors, time step, one Newton iteration, 50 PCG
iterations, and velocity damping `1.0` unchanged. The orange patch should now
settle instead of reproducing persistent EE churn.

The example continues to demonstrate the original nominal 6 mm request, but
the effective local envelopes are limited by the stored rest mesh. Update its
description, README text, screenshot, and changelog entry to describe the
geometry-aware stabilization rather than a permanent reproducer.

## Tests

Use `unittest` and retain CUDA gating for runtime contact tests.

Add or update tests for:

1. constructor validation of `geometry_radius_scale` and invalid rest meshes;
2. exact per-particle radii on a small nonuniform rest mesh;
3. unchanged uniform radii and legacy one-ring behavior when the new argument
   is omitted;
4. VF depth computed from the vertex radius plus barycentrically interpolated
   face radius;
5. EE depth computed from radii interpolated independently with `s` and `t`;
6. a close geometry-aware EE pair still producing a finite separating force,
   so stabilization is not implemented by disabling EE contacts;
7. the 74-vertex example preserving its topology, anchors, finite state, and
   zero contact-buffer overflow;
8. frames 1000–1399 of a 1400-frame rollout having mean interior RMS speed
   below `1e-5 m/s` and at most 10 EE births plus deaths;
9. all existing LIMX self-collision tests continuing to pass in default mode.

The existing late-churn characterization test is first changed to the new
settling expectation and must fail before the implementation is added.

## Compatibility and Failure Modes

- The new argument is appended and defaults to `None`; no existing call site
  changes behavior automatically.
- The nominal scalar thickness remains available for regular meshes and
  callers that intentionally want the legacy model.
- Geometry-aware mode may shrink radii sharply near sliver triangles. This is
  intentional and observable through `particle_radii`; invalid degenerate
  triangles are rejected rather than silently receiving zero radius.
- A conservative nominal broad phase may produce extra candidates but does
  not change contact capacity semantics or CUDA graph compatibility.
- If the diagnosed patch still chatters, do not tune damping or iteration
  counts. Record the local radii and persistent-pair thresholds, then revisit
  the radius estimator as a new hypothesis.

## Completion Criteria

The change is complete when the focused RED test turns green, the orange
geometry-aware patch is visually still, close-contact unit fixtures still
generate VF/EE forces, existing self-collision regression tests pass, API docs
are regenerated, pre-commit succeeds, and the final interactive viewer shows
both patches settled without non-finite state or contact overflow.
