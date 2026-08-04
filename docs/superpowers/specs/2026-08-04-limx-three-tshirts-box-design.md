# LIMX Three T-Shirts in an Open Box Design

## Goal

Add a CUDA stress-test scene in which three copies of Newton's T-shirt mesh
are thrown into an open box, collide with one another and themselves, collide
with the floor and four walls, and remain finite. The first version is for
interactive inspection; strict settling and contact-churn targets are deferred
until the visual failure mode is known.

## Scene architecture

Build one particle model containing three disconnected copies of
`unisex_shirt.usd`. Concatenate their particle data and offset each copy's
triangle indices into the combined particle array. This gives one elastic
system and one global `ConstraintSelfCollision`, so the existing VF, EE, and EF
passes naturally cover both intra-garment and inter-garment contacts. All three
copies remain in the same simulation world.

Each garment keeps the T-shirt example's area-weighted mass, anisotropic
triangle elasticity, rest-shape dihedral bending, collision thickness, and
adaptive collision factors. Compute dihedral topology independently for each
disconnected copy before concatenating it, so no bending stencil can connect
two garments.

Use three distinct initial transforms, heights, linear velocities, and angular
velocities. Their starting surfaces must not intersect. The different heights
make the garments enter the box at different times without adding delayed
particle activation or another solver state.

## Box representation

Represent the open box physically with five existing
`ConstraintStaticPlaneContact` instances:

- floor with inward normal `+Z`;
- left wall with inward normal `+X`;
- right wall with inward normal `-X`;
- front wall with inward normal `+Y`;
- back wall with inward normal `-Y`.

Compose the five planes and the global self-collision operator with
`ConstraintGroupDynamic`. The planes define an inward-facing convex region and
remain matrix-free. No rigid body, static triangle collider, new box-contact
type, or PP/PE collision fallback is introduced.

Render the floor and walls as five thin static boxes whose inner faces match
the analytic planes. Use an interior footprint of approximately `1.2 m x 1.0
m`, a floor at `z = 0.45 m`, and walls extending to approximately `z = 1.15 m`.
The top stays open.

## Solver configuration

Keep the current large-step LIMX configuration so this scene tests the same
solver path as the single-garment example:

- time step `0.01 s`, one physics step per rendered frame;
- one Newton iteration;
- 50 PCG iterations with the existing previous-frame warm start;
- velocity damping `1.0`;
- gravity `(0, 0, -9.81) m/s^2`;
- membrane stiffness `(1e4, 1e4, 1e3)`;
- dihedral bending stiffness `1e-4`;
- self-collision thickness `0.006 m`;
- adaptive self-collision factors `(VF=0.5, EE=0.1, EF=1.5)`;
- box-contact thickness `0.006 m`, normal stiffness `2e4 N/m`, normal damping
  `0.5 N*s/m`, friction `0.4`, and friction regularization `1e-4 m`.

Raise the per-type self-collision capacity from `131072` to `393216` for the
three-garment scene. Keep overflow counters observable so later diagnosis can
distinguish a capacity failure from a solver failure.

## File and API scope

Add a new example alongside the existing T-shirt table example and register it
with the normal Newton example discovery path. Reuse public
`newton.solvers` symbols only; the example must not import `newton._src`.
Register its command in the examples README, add a `320 x 320` screenshot, and
record the new user-facing example in the root changelog's Unreleased section.

The first version does not change `SolverLIMX`, the public constraint API, the
single-garment scene, or collision coefficients. Shared T-shirt loading and
mass construction may be kept local to the new example unless implementation
shows a small existing public helper is already suitable.

## Validation

Add CUDA `unittest` coverage that verifies:

- the combined model contains exactly three disconnected garment copies;
- one captured CUDA step produces finite positions and velocities;
- the dynamic group contains one global self-collision operator and five
  static-plane contacts;
- a focused headless rollout remains finite, reports no contact-buffer
  overflow, and avoids catastrophic floor or wall escape;
- `test_final()` performs the same finite-state and catastrophic-containment
  checks for the interactive example.
- the documented example command resolves through Newton's example launcher.

Do not add a strict final-speed threshold in this first pass. After the user
inspects the window, use observed penetrations, oscillations, escapes, or
non-settling regions to define the next regression rather than tuning unseen
behavior.

## Non-goals

- Do not add rigid-body degrees of freedom or a general mesh collider.
- Do not split the garments across independent solvers.
- Do not add delayed spawning, CCD, line search, or additional Newton
  iterations in the first visual pass.
- Do not tune damping, bending, friction, or collision stiffness before the
  initial window is inspected.
- Do not add PP/PE degeneration for nearly parallel EE contacts.
