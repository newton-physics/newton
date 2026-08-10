# LIMX Automatic Collision Thickness Design

## Goal

Replace the bunny scene's manually selected nominal VF/EE thickness with a
rest-geometry estimate:

```text
thickness = min(0.8 * two_ring_upper_bound, 0.005 m)
```

## Two-Ring Upper Bound

Build surface graph adjacency from triangle edges. A VF or EE pair is a
two-ring pair when the minimum graph distance between its constituent vertices
is exactly two. Evaluate the same interior feature regions used by LIMX's
discrete detector:

- VF contributes its perpendicular point-triangle distance only when the
  projected point lies inside the triangle.
- EE contributes its segment-segment distance only when both closest
  parameters lie strictly inside their edges.

The upper bound is the smallest positive rest distance across those VF and EE
pairs. If no finite two-ring pair exists, the `0.005 m` cap is the estimate. A
zero-distance two-ring configuration cannot admit a positive automatic band
and must use an explicit thickness.

## API and Compatibility

Passing `thickness=None` to `ConstraintSelfCollision` selects automatic mode.
Every explicit positive thickness keeps its existing behavior. The estimator
does not modify one-ring classification, contact forces, Hessians, stiffness,
friction, damping, CCD, or time stepping.

The eight-bunny example uses automatic mode. Its static box-plane contact
thickness remains independently configured.

## Validation

- Synthetic geometry verifies the two-ring VF/EE upper-bound calculation and
  the 5 mm cap.
- Explicit thickness remains unchanged.
- The bunny example computes its nominal self-collision thickness from rest
  geometry and remains stable for 300 CUDA frames.
