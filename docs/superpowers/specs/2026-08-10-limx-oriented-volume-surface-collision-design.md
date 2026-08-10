# LIMX Oriented Volume-Surface Collision Design

## Goal

Add an opt-in oriented mode to `ConstraintSelfCollision` for tetrahedral
boundary surfaces. The mode must use outward surface normals and signed VF/EE
separation for both same-body and cross-body contact, while preserving the
existing unsigned cloth behavior by default.

## Scope

This phase changes only discrete contact detection and the frozen contact data
consumed by the existing force and Hessian assembly. It does not add CCD,
swept bounds, line search, substeps, EF recovery, or a new solver.

The oriented mode assumes that `model.tri_indices` has consistent outward
winding. `TetMesh.compute_surface_triangles()` provides that winding for
positive-volume tetrahedra. The bunny asset was checked directly: all 2,152
boundary faces point away from their owning tetrahedron.

## Public Configuration

Extend `ConstraintSelfCollision` with:

```python
use_outward_normals: bool = False
```

`False` preserves the existing cloth-compatible unsigned VF/EE path. `True`
enables signed contact and topology filtering. The eight-bunny example opts
in explicitly.

## Topology Filtering

Signed volume contact must not treat local surface discretization as
self-contact. In the bunny rest state, all 80 VF candidates are one-ring
vertex-face pairs, and 200 of 216 EE candidates are topology-local opposite
edges.

Build a compact vertex-neighbor CSR from the surface edges. In oriented mode:

- reject a VF pair when the candidate vertex is one edge away from any of the
  face's three vertices;
- reject an EE pair when `_is_topology_local_edge_pair()` is true;
- continue rejecting contacts that share a feature endpoint.

The unsigned default retains its current topology-local EE thickness clamp
and VF behavior for compatibility.

## Signed Vertex-Face Contact

For an outward-oriented face `(x0, x1, x2)`, compute:

```text
n = normalize(cross(x1 - x0, x2 - x0))
s = dot(xv - x0, n)
depth = effective_thickness - s
```

The projected point and barycentric weights are computed as before. The
contact is active when the projection lies inside the triangle and
`s < effective_thickness`. The BVH query remains the existing discrete 3 mm
candidate band. Do not take `abs(s)` and do not flip `n` when `s < 0`.

The vertex receives force along `+n`; the face vertices receive the balanced
opposite force through the existing barycentric weights. A negative signed
distance therefore increases penetration depth and still pushes outward.

## Signed Edge-Edge Contact

Store `MeshAdjacency.edge_tri_indices` beside each edge. Recompute each
edge's outward pseudo-normal from its incident oriented faces at contact
detection time:

```text
ne = normalize(sum(unit incident face normals))
```

For edge 0 and edge 1, define the force direction on edge 0 as:

```text
n = normalize(ne1 - ne0)
s = dot(closest0 - closest1, n)
depth = effective_thickness - s
```

Reject the pair when either pseudo-normal or their difference is degenerate.
Retain the existing Euclidean-distance broad-phase/narrow-phase band and
mollifier. Do not flip `n` from the current closest-point vector; after a
crossing, `s` becomes negative while `n` continues to encode the outward
separation direction.

## Force and Hessian

The existing contact buffers already assemble balanced force, full
Hessian-vector products, and exact diagonal blocks from frozen contact
weights, directions, depths, and stiffness. Supplying the signed depth and
outward direction gives the frozen-normal PSD model:

```text
force   = k * depth * J^T
Hessian = k * J^T J
```

where `J` contains the barycentric or edge interpolation weights projected
onto `n`. Force, Hessian-vector products, and diagonal blocks therefore use
the same orientation without a separate assembly path.

## Verification

Focused CUDA tests must prove:

1. an inside VF contact keeps the target face's outward normal and uses
   `thickness - signed_distance` depth;
2. one-ring VF candidates are absent in oriented mode;
3. an already-crossed EE pair uses incident face normals rather than the
   flipped closest-point vector;
4. topology-local EE pairs are absent in oriented mode;
5. the default unsigned mode retains the existing tests;
6. the bunny scene opts in, keeps EF disabled, and remains finite,
   positive-volume, and overflow-free for 300 frames.

The visual experiment then checks whether all bunnies are pushed toward the
exterior and settle on the floor without relying on friction.
