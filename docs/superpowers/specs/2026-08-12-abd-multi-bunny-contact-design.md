# ABD Multi-Bunny Contact Design

## Goal

Extend the LIMX affine-body path from one bunny against a static plane to
eight mutually colliding affine bunnies falling onto the same frictional
ground. Body-body contact uses the actual reconstructed bunny surfaces and
participates in the same Newton/PCG system as affine inertia, ARAP rigidity,
and ground contact.

The example must visibly form a pile without treating the bunnies as spheres,
without solving each body independently, and without adding an IPC barrier.
Every frame retains exactly one Newton increment and 50 PCG iterations.

## Chosen Approach

Use a single multi-body `AffineBodyModel`, a GPU broad phase over all
reconstructed surfaces, and a matrix-free affine-affine contact operator.
Contacts couple the two involved 12-DOF bodies inside one linear solve. The
block-Jacobi preconditioner keeps one exact 12-by-12 diagonal block per body;
it does not materialize the off-diagonal 12-by-12 contact blocks.

Two alternatives are deliberately rejected:

- advancing eight independent affine solvers and applying impulses afterward
  would remove body-body contact from the Newton/PCG system;
- sphere, particle, or SDF proxies would be easier to collide but would not
  test the requested VF/EE response on the bunny surface.

## Multi-Body Affine Model

Add `AffineBodyModel.from_instances(...)`, a class method that accepts the
same mesh, density, rigidity, and device arguments as the constructor plus a
nonempty `initial_transforms` sequence. The existing one-body constructor,
its `initial_transform` argument, and its behavior remain unchanged.

The common mesh is integrated once. Its centered rest geometry, volume,
12-by-12 mass matrix, gravity acceleration, and rigidity are then repeated
for each instance. Runtime arrays have one entry per body:

- `q`, `qd`, `volumes`, `mass_matrices`, `rigidities`, and `gravity` contain
  `body_count` rows;
- centered surface vertices and compact surface triangles are concatenated;
- every surface vertex stores its owning body;
- triangle indices are offset into the concatenated surface array;
- volumetric vertex and tetrahedron arrays remain globally indexed and
  consistent with their repeated instances.

For body `b`, a centered material point `r` is reconstructed as

\[
x_b(r)=t_b+A_b r=J(r)q_b.
\]

The initial transforms may differ in translation and rotation, but the first
example uses the same density, rigidity, and rest mesh for all eight bodies.
Initial generalized velocities remain writable through the existing `qd`
array so the example can add small deterministic lateral perturbations
without expanding the construction API.

## Contact Topology and Broad Phase

Create `ConstraintAffineBodyContact`, a fixed-capacity GPU dynamic operator
over one multi-body affine model. It owns:

- concatenated compact surface vertices and outward-wound triangles;
- surface edges derived once from the compact triangle topology;
- body ownership for vertices, triangles, and edges;
- world-space surface positions reconstructed at each Newton iterate;
- refittable triangle and edge BVHs;
- separate fixed-capacity VF and EE contact buffers with overflow counters.

`prepare(q)` reconstructs all surface vertices, refits both BVHs, clears the
buffers, and regenerates the frozen contact set.

Every surface vertex queries the triangle BVH, which naturally evaluates VF
in both body directions. Candidates whose vertex and triangle belong to the
same affine body are rejected. The narrow phase uses unsigned triangle
closest point, including clamped edge and vertex regions. VF therefore owns
the PE and PP boundary cases.

Every surface edge queries the edge BVH. An unordered edge pair is evaluated
once, same-body pairs and shared vertices are rejected, and a contact is
accepted only when both closest parameters are strictly inside `(0, 1)`.
Endpoint EE cases do not generate PE or PP contacts; they are delegated to
the VF closest-point path. This preserves the previously validated LIMX
deduplication policy. The design does not add an independent EV or VV broad
phase and does not collapse separate VF parent stencils that happen to touch
the same mesh boundary feature.

Retained EE pairs use the existing IPC-style near-parallel mollifier, with a
threshold scaled from the two rest-edge lengths. The mollifier changes the EE
residual and its Gauss-Newton Hessian but does not reintroduce endpoint EE.

## Penalty Response and Affine Lifting

A frozen contact contains material-point IDs, scalar closest-feature weights
`w_i`, a unit direction `n`, and penetration depth

\[
\delta=h-d>0.
\]

For ordinary VF and non-mollified EE contact, define the relative world
displacement Jacobian for body `b` as

\[
G_b=\sum_{i\in b}w_iJ(r_i).
\]

Each cross-body stencil has two nonzero body Jacobians whose translational
parts sum to zero. With penalty stiffness `k`, the normal force and PSD
Gauss-Newton Hessian are

\[
f_b=G_b^T(k\delta n),
\qquad
H_{bc}=G_b^T(knn^T)G_c.
\]

The force scatter is equal and opposite in world translation. The
matrix-free multiply evaluates the complete two-body product

\[
y_b\mathrel{+}=G_b^TH_x\sum_cG_cp_c,
\]

so the off-diagonal affine-affine coupling is retained. The preconditioner
receives only the exact diagonal terms `G_b.T @ H_x @ G_b`, accumulated into
the body's native 12-by-12 block.

For an active EE mollifier, the existing four-point residual Jacobian is
evaluated on affine material-point motions. Its complete body-space
Gauss-Newton product remains matrix-free. Each exact 12-by-12 body diagonal
is obtained from the same generalized residual Jacobian, including
cross-terms between the two endpoints owned by that body.

The response is discrete and penalty-based. It does not add CCD, a line
search, an IPC barrier, deep-overlap recovery, or oriented projected VF.
Initial separation, the 3 mm activation band, and the existing 0.01 s step
are therefore part of the example's stability contract.

## Damping and Friction

`begin_step(q, qd, dt)` caches the step-start generalized velocity. For each
frozen contact, compute the relative point velocity

\[
v_{rel}=\sum_bG_b\dot q_b.
\]

Approaching normal motion receives the same lagged normal damping convention
as affine-plane contact:

\[
f_d=-c_n\min(n^Tv_{rel},0)n.
\]

With `P_t=I-nn^T`, the lagged tangential displacement is

\[
\Delta x_t=\Delta t P_tv_{rel}.
\]

Use the existing regularized Coulomb function

\[
s(r)=
\begin{cases}
1/r,&r>\epsilon,\\
(2-r/\epsilon)/\epsilon,&r\le\epsilon,
\end{cases}
\]

to form

\[
f_t=-\mu k\delta s(\|\Delta x_t\|)\Delta x_t,
\qquad
H_t=\mu k\delta s(\|\Delta x_t\|)P_t.
\]

The normal load and scalar regularization factor are frozen during one
linearization. Friction and damping are lifted through the same two-body
Jacobians, preserving equal-and-opposite translation forces and the complete
matrix-free coupling. The first example uses coefficient `0.5` for both
body-body and ground friction.

## Dynamic Constraint Composition

Add `ConstraintGroupAffine`, analogous to `ConstraintGroupDynamic`, for
operators using the mixed affine interface. It validates a common body count
and device, then forwards in deterministic order:

1. `begin_step(q, qd, dt)`;
2. `prepare(q)`;
3. `accumulate_force(q, affine_output)`;
4. `multiply(particle_input, affine_input, particle_output, affine_output)`;
5. `accumulate_diagonal(particle_diagonal, affine_diagonal)`.

The eight-bunny example groups `ConstraintAffineBodyContact` with the existing
`ConstraintAffineStaticPlaneContact`. `SolverLIMXAffine` and
`MixedPcgSolver` need no change to their solve schedule or split 3-by-3 and
12-by-12 storage.

The new model construction path, contact constraint, and affine group are
exported through `newton.solvers`; examples do not import from `newton._src`.
Public API documentation is regenerated after adding the symbols.

## Example

Add `basic_limx_affine_bunnies_ground` with these fixed first-pass settings:

- eight instances of `newton/examples/assets/bunny_tet.npz` at scale `0.15`;
- a staggered `2 x 2 x 2` layout with no initial overlap, varied heights,
  deterministic small tilts, and no vertically aligned upper columns;
- density `1000 kg/m^3` and ARAP rigidity `1.0e8 Pa` for every body;
- ground top at `z = 0` with a render box large enough for the pile;
- contact thickness `0.003 m`;
- normal stiffness `2.0e4 N/m` per retained contact;
- normal damping `0.5 N*s/m`;
- body-body and ground friction coefficient `0.5`;
- friction regularization `1.0e-4 m`;
- time step `0.01 s`;
- exactly one Newton iteration and 50 PCG iterations per frame;
- 300 default frames.

The scene reconstructs all eight surfaces into one render-only Newton model,
uses CUDA graph capture when available, assigns visually distinct colors,
and places the camera around the full drop and pile. Register the example in
`README.md` and add its 320-by-320 screenshot.

## Validation and Acceptance

Use `unittest` and write focused regression tests before implementation.

### Multi-Body Model

1. Build two instances of a small tetrahedral mesh and verify body counts,
   repeated mass/volume/rigidity data, ownership, globally offset topology,
   initial states, and independently reconstructed surfaces.
2. Verify the existing one-body constructor produces unchanged arrays and
   trajectories.
3. Reject an empty transform sequence, invalid transforms, invalid topology,
   and inconsistent devices with clear errors.

### Contact Generation

1. Verify same-body VF and EE candidates are absent.
2. Verify cross-body VF is generated in both directions where appropriate.
3. Place the closest triangle point on an edge and then a vertex and verify
   VF retains both boundary cases.
4. Verify EE accepts strict interior-interior closest points and rejects every
   case with either parameter on an endpoint.
5. Verify near-parallel retained EE activates the mollifier without producing
   PE or PP contacts.
6. Force small capacities and verify VF and EE overflows are counted rather
   than writing out of bounds.

### Force and Linear Operator

1. Compare VF and ordinary EE generalized forces, complete two-body HVPs, and
   both 12-by-12 preconditioner blocks with dense Jacobian references.
2. Compare mollified EE force, HVP, and exact body diagonals with a dense or
   directional-derivative reference.
3. Verify translation forces sum to zero, friction opposes relative tangent
   velocity, normal damping acts only while approaching, and all regularized
   results remain finite.
4. Verify the assembled two-body operator is symmetric positive semidefinite
   within numerical tolerance.
5. Verify `ConstraintGroupAffine` forwards every lifecycle method and rejects
   mismatched body counts or devices.

### Eight-Bunny CUDA Rollout

Run 300 frames with graph capture and require:

- finite generalized states, velocities, reconstructed vertices, contact
  forces, and PCG work vectors;
- `det(A) > 0` and maximum singular-value deviation below `0.02` for every
  body throughout the rollout;
- no VF or EE contact-buffer overflow;
- at least one cross-body VF or strict EE contact after release;
- all eight centers fall from their initial heights;
- ground penetration remains below `0.006 m`;
- maximum stored VF or EE penalty depth remains below `0.012 m`;
- during the final 30 frames, at least one initially upper-layer bunny keeps
  its center at least `0.10 m` above the highest initially lower-layer center,
  demonstrating sustained pile support rather than eight independent ground
  contacts.

The rollout thresholds may be tightened after measuring the implemented
scene, but they may not be weakened to hide inversion, pass-through, contact
overflow, excessive ground penetration, or loss of pile support.

## Scope Boundaries

This change does not add affine-cloth coupling, joints, motors, per-instance
meshes or material parameters, CCD, barriers, SDF collision, independent PE
or PP candidates, EE endpoint contacts, or changes to the native particle
3-by-3 block path. Those remain separate milestones.
