# LIMX Cloth Self-Collision Design

## Goal

Add a first GPU cloth self-collision path to `SolverLIMX`. Fixed-topology
membrane and bending energies remain assembled in 3-by-3 block-CSR. The
time-varying vertex-face (VF), edge-edge (EE), and edge-face untangling (EF)
contacts are detected at the current Newton iterate and contribute force,
positive-semidefinite Hessian-vector products, and block-Jacobi diagonal
blocks matrix-free.

The first version is frictionless and uses discrete collision detection. EF
untangling recovers already intersecting surfaces; it is not continuous
collision detection and therefore does not guarantee that very fast motion
cannot tunnel through the cloth.

## Public Interface

Add one dynamic constraint operator:

```python
ConstraintSelfCollision(
    model,
    thickness,
    stiffness,
    untangle_stiffness=None,
    max_contacts=32768,
)
```

`model` must be a particle triangle mesh with finite triangle indices and at
least one triangle. `thickness` is the one-sided activation distance [m].
`stiffness` is the VF/EE penalty stiffness [N/m]. `untangle_stiffness`
defaults to `stiffness`. `max_contacts` is the fixed capacity of each VF, EE,
and EF contact buffer, chosen before CUDA graph capture.

The object implements the LIMX dynamic-operator protocol:

```python
prepare(positions)
accumulate_force(positions, output)
hessian_multiply(positions, vector, output)
accumulate_diagonal(positions, output)
```

It is exported as `newton.solvers.ConstraintSelfCollision` and passed to
`SolverLIMX(..., dynamic_operator=self_collision)`.

## Newton/PCG Lifetime

At the start of every nonlinear iteration, the solver calls
`dynamic_operator.prepare(iterate_positions)`. That call refits the triangle
and edge BVHs, detects contacts, and freezes their particle ids, signed
weights, directions, and penetration depths. Force, preconditioner diagonal,
and every PCG Hessian-vector product in that nonlinear iteration consume that
same snapshot. No detector is run from inside PCG.

The next Newton iteration repeats detection at its updated current position.
The inertial prediction is never used to build contacts, forces, or Hessians.

## Topology and Broad Phase

Use `newton.utils.MeshAdjacency(triangles)` on the host to derive all unique
mesh edges, including boundary edges. Each edge row is
`[opposite0, opposite1, endpoint0, endpoint1]`. Upload the full rows and the
triangle indices once.

Maintain two Warp BVHs with fixed leaf counts:

- triangle BVH for VF and EF queries;
- edge BVH for EE queries.

GPU kernels update primitive AABBs from the current iterate and call
`refit()` before narrow phase. Queries append directly into fixed-capacity
flat contact buffers with device-side atomic counters. A separate device
overflow counter records every result that did not fit, so overflow is
observable rather than silently discarded. Runtime kernels use
`min(attempted_count, max_contacts)`.

Reject topologically local pairs:

- VF when the vertex belongs to the triangle;
- EE when the two edges share an endpoint, and process only `other > self`;
- EF when either edge endpoint belongs to the triangle.

Pairs from different Newton worlds are also rejected.

## Frozen Rank-One Contact Model

For one frozen contact, let particle ids be `i_a`, signed scalar weights be
`q_a`, and the frozen unit direction be `n`. Define

```text
g(x) = n dot sum_a(q_a x_i_a)
r(x) = h - g(x)
E(x) = 1/2 k max(0, r(x))^2.
```

Detection stores only active contacts, so the frozen linearization uses the
positive detected depth `d = h - g(x_current)`. Its physical force and
Gauss-Newton Hessian are

```text
f_i_a = k d q_a n
H_ab  = k q_a q_b outer(n, n).
```

The matrix-free Hessian-vector product is evaluated without materializing a
local matrix:

```text
s = sum_b q_b (n dot v_i_b)
(Hv)_i_a += k q_a s n.
```

The block-Jacobi preconditioner receives the exact diagonal blocks of this
rank-one Hessian:

```text
H_aa = k q_a^2 outer(n, n).
```

This full matrix-free Hessian preserves translation null modes and action-
reaction symmetry. It intentionally does not copy Style3D's diagonal-only
contact Hessian.

## Vertex-Face Contacts

For vertex `v` and triangle `(a,b,c)`, project the vertex onto the triangle
plane. If the projection has barycentric coordinates `(b0,b1,b2)` inside the
triangle and signed plane distance `delta` satisfies
`epsilon < abs(delta) < thickness`, store

```text
ids = (v,a,b,c)
q   = (1,-b0,-b1,-b2)
n   = sign(delta) triangle_normal
d   = thickness - abs(delta).
```

The signed weights sum to zero. Degenerate triangles and ambiguous zero-
distance directions are skipped; an exact intersection is handled by EF
untangling.

## Edge-Edge Contacts

For nonadjacent edges `(a0,a1)` and `(b0,b1)`, compute the closest interior
parameters `s` and `t`. Endpoint contacts are left to VF. With

```text
pa = (1-s) x_a0 + s x_a1
pb = (1-t) x_b0 + t x_b1,
```

if `epsilon < ||pa-pb|| < thickness`, store

```text
ids = (a0,a1,b0,b1)
q   = (1-s,s,-(1-t),-t)
n   = normalize(pa-pb)
d   = thickness - ||pa-pb||.
```

The implementation must use `t` for the second edge. Ai-Physics currently
uses `s` for both edges in its emitted contact weights; that behavior is not
copied.

## Edge-Face Untangling

EF detects a true edge/triangle crossing: the endpoints lie strictly on
opposite sides of the face plane and the segment-plane hit has all three
barycentric coordinates at least `0.01`. For edge interpolation `u` and face
barycentrics `(b0,b1,b2)`, store five signed weights

```text
q = (1-u,u,-b0,-b1,-b2).
```

The recovery direction `G` follows Volino and Magnenat-Thalmann's
intersection-contour minimization as used by Style3D. For each triangle
adjacent to the crossing edge, form and orient the intersection direction,
apply

```text
G_part = R - 2 N_face (E dot R) / (E dot N_face),
```

sum valid contributions, and normalize. Degenerate edges, triangles,
parallel denominators, or a zero final direction are skipped.

At a detected crossing the edge and face hit points coincide, so the frozen
quadratic recovery target is `h = 2 * thickness` and detected depth is also
`2 * thickness`. EF uses the same full rank-one force, HVP, and diagonal
formulas over five particles with `untangle_stiffness`.

## Degeneracy, Capacity, and Validation

The constructor rejects nonpositive or nonfinite thickness/stiffness,
nonpositive capacity, missing or malformed triangle topology, invalid
indices, and a model/device mismatch. Runtime narrow phase skips degenerate
geometry and never emits nonfinite contact data.

Contact counters and overflow counters remain device arrays so detection and
the solve are CUDA-capture safe. Host count accessors are diagnostic only and
may synchronize; the simulation path never reads a contact count on the
host.

## Example Integration

Enable `ConstraintSelfCollision` in `cloth_limx` while preserving its current
configuration: `dt=0.01`, one Newton iteration, 50 PCG iterations, one render
per physics step, no explicit damping, anisotropic membrane energy, dihedral
bending, two anchor points, and cross-frame PCG warm start.

The example uses the particle diameter as collision thickness and a contact
stiffness comparable to the membrane stiffness. Its existing CUDA graph path
must remain operational.

## Tests and Completion Criteria

Use `unittest` and run routine validation on `cuda:0` only. Required focused
tests cover:

- correct VF ids, barycentric weights, normal, depth, and local-topology
  rejection;
- asymmetric EE closest parameters with `s != t`, correct second-edge `t`
  weights, duplicate rejection, and shared-endpoint rejection;
- EF crossing detection, five weights summing to zero, finite ICM direction,
  and exact-crossing recovery depth;
- force balance, matrix-free HVP equality to an independently assembled dense
  rank-one matrix, exact block diagonal, and nonnegative quadratic form for
  both four- and five-particle contacts;
- no-contact and degenerate inputs producing finite zero contributions;
- explicit overflow accounting with bounded writes;
- detector refresh once per Newton iteration before force and PCG;
- public export and CUDA graph-compatible example integration;
- a small colliding-cloth integration fixture that remains finite and moves
  separated surfaces apart.

Run focused CUDA tests, the headless `cloth_limx` example, generated API docs,
and formatting/lint checks. CPU simulation is reserved for debugging an
observed CUDA failure.
