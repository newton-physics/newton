# LIMX EE Contact Chatter Reproducer Design

## Goal

Add a small CUDA example that deterministically reproduces the persistent
LIMX cloth motion caused by edge-edge contact-set churn. Present an otherwise
identical no-self-collision control patch beside the affected patch so the
collision-induced motion is visible without running the three-T-shirt box
scene.

The scene characterizes the current VF/EE implementation. It does not fix the
contact discontinuity or reintroduce PE/PP contacts.

## Reproduction asset

Use the 74-vertex, 112-triangle, four-ring sleeve patch isolated from garment
one of the three-T-shirt box scene after 800 simulation frames. Its source
region contains the local EE contacts previously observed to switch across
feature boundaries. Preserve 34 boundary vertices and 40 interior vertices.

Store the patch's rest positions, problem-state positions, triangle topology,
masses, and boundary indices as text data in the example module. The example
must not simulate the full T-shirt scene or load `unisex_shirt.usd` at startup.
This makes the initial state deterministic, keeps the example self-contained,
and avoids a hidden 800-frame warm-up.

The stored problem state is part of a diagnostic example, not a reusable
Newton API or a general cloth asset.

## Side-by-side architecture

Create two independent particle models and two independent `SolverLIMX`
instances from the same stored patch data:

- the left control uses membrane elasticity, dihedral bending, and boundary
  anchors, with no dynamic self-collision operator;
- the right reproduction uses identical static constraints plus one
  `ConstraintSelfCollision` containing only the existing VF, EE, and EF
  passes.

Both patches start with zero velocity and zero gravity. Anchor the same 34
boundary vertices to the stored problem-state positions using stiffness
`1e6 N/m`. Separate the two copies by a rigid horizontal translation that is
applied consistently to rest positions, initial positions, and anchor targets.
Because the solvers and models are independent, the control does not require a
particle collision mask and cannot collide with the reproduction patch.

Advance both solvers inside one CUDA graph. Use `dt=0.01 s`, one Newton
iteration, 50 PCG iterations, and velocity damping `1.0` for both sides.

## Material and contact configuration

Use the same coefficients as the diagnosed three-T-shirt scene:

- membrane stiffness `(1e4, 1e4, 1e3)`;
- dihedral bending stiffness `1e-5`;
- self-collision thickness `0.006 m`;
- adaptive self-collision factors `(VF=0.5, EE=0.1, EF=1.5)`;
- contact capacity `4096` per type.

Set the particle display radius to `0.003 m`, matching the Style3D contact
radius convention. The explicit self-collision activation distance remains
`0.006 m`, representing two `0.003 m` collision envelopes.

Do not add damping, gravity, a static plane, rigid bodies, PP/PE contacts, CCD,
line search, or additional Newton iterations.

## Rendering and interaction

Render the two state meshes directly through the public viewer mesh logging
interface rather than assigning either model as the viewer's sole model. Use
distinct colors: blue for the no-collision control and orange for the VF/EE
reproduction. Disable back-face culling so both sides of the cloth patch remain
visible.

Frame both patches closely in the default camera. The initial visual state must
be identical up to horizontal translation; after simulation begins, the left
patch should settle while the right patch continues to move.

No runtime toggle or reset button is required because the simultaneous control
already exposes the causal comparison.

## Example API and file scope

Add `newton/examples/cloth/example_cloth_limx_ee_chatter.py` using the normal
Newton `Example` class format. The module may define one private patch-data
container and one private patch-simulation helper, while the public example
surface remains `Example(viewer, args)` with `step()`, `render()`,
`test_post_step()`, and `test_final()`.

Expose diagnostic attributes needed by tests, including the control and
collision simulations, patch counts, and self-collision contact buffers. Do
not add or change public solver symbols.

Register `python -m newton.examples cloth_limx_ee_chatter` in the README cloth
gallery and add a `320 x 320` screenshot. Add an Unreleased changelog entry
because this is a user-facing example.

## Validation

Add CUDA `unittest` coverage that verifies:

- example discovery resolves `cloth_limx_ee_chatter`;
- each model has 74 particles and 112 triangles;
- each side uses 34 anchors, one Newton iteration, 50 PCG iterations, and
  velocity damping `1.0`;
- only the right side owns `ConstraintSelfCollision` and its capacity is
  `4096`;
- one captured step remains finite;
- neither contact buffer overflows during the focused rollout;
- after starting both copies from zero velocity and running the late-time
  window, the control interior speed converges to numerical zero while the
  collision side retains measurable nonzero RMS speed;
- the collision side continues to create and remove EE pairs during the
  late-time window.

The late-time motion and churn assertions are characterization checks for the
known defect. When the EE feature-boundary discontinuity is fixed, invert those
assertions so the same example becomes a settling regression.

## Non-goals

- Do not change `ConstraintSelfCollision` or `SolverLIMX` in this slice.
- Do not claim that `0.006 m` is physically derived from the shirt mesh.
- Do not use the synthetic falling cuff as the regression because its final
  folded state is not deterministic enough.
- Do not run CPU cloth simulation.
- Do not load or simulate the full three-garment scene in the new example.
