# LIMX Franka Cloth Scene Design

## Goal

Add a new example that places a flat square LIMX cloth on a table and drives a
fixed-base Franka FR3 through a visible grasp sequence. The first milestone
validates the scene layout, coordinate system, gripper pose, and motion before
introducing cloth-rigid contact.

## Scope

The first milestone includes:

- One fixed-base Franka FR3 with its standard two-finger hand.
- One static table in a meter-scale scene.
- One square, triangulated LIMX cloth mesh lying flat on the table.
- A keyframed IK sequence that approaches the cloth, descends, closes the
  fingers, and lifts.
- A focused example-level validity check and a visually useful default camera.

The first milestone does not include:

- Contact or force exchange between the cloth and the table, hand, or fingers.
- Cloth deformation, lifting, grasp capture, or release behavior.
- Self-collision tuning beyond constructing the cloth data needed by a later
  interaction milestone.
- Multiple environments, randomized cloth poses, or task-level control.

## Reused Patterns

The implementation will combine current Newton examples rather than introduce
a new control stack:

- Reuse the meter-scale Franka asset loading, GPU IK objectives, PD joint
  targets, and grasp keyframe interpolation from
  `multiphysics/example_franka_cable_ik_pick_place.py`.
- Reuse square triangulated particle-grid construction and LIMX material data
  from `cloth/example_cloth_limx.py`.
- Use the table and camera conventions from existing Franka manipulation
  examples.

The older `cloth/example_cloth_franka.py` is useful as behavioral reference,
but its centimeter-scale VBD setup and custom Jacobian controller will not be
copied because this example targets LIMX and should remain in SI units.

## Scene Architecture

Create `newton/examples/cloth/example_cloth_limx_franka.py` with one
`ModelBuilder` containing the robot, table, and cloth. Keeping all scene
elements in one model establishes stable body, shape, joint, and particle
indices for the later coupling work.

The table is a static box. The Franka is loaded as a non-floating articulation
and controlled by `SolverFeatherstone` through position targets produced by
Newton's IK solver. The cloth is a roughly 0.4 m square, regular 21-by-21
particle grid with alternating triangle diagonals. Its rest pose is horizontal
and offset slightly above the table top so the future contact thickness can be
introduced without starting from penetration.

All cloth particles are inactive in the first milestone. This deliberately
keeps the cloth flat without inventing temporary collision behavior or allowing
gravity to pull it through the table. The LIMX topology and rest-state arrays
are still constructed so activating the cloth later is a local change.

## Robot Motion

The gripper follows a short looping keyframe sequence:

1. Start above the selected cloth edge or corner with open fingers.
2. Descend to a grasp pose centered around the cloth surface.
3. Close the fingers while holding the grasp pose.
4. Lift vertically while keeping the fingers closed.
5. Return to the approach pose and reopen before repeating.

Position and rotation targets are interpolated over explicit durations. IK is
solved against the Franka hand TCP, and the resulting joint coordinates are
written to the articulation's PD targets. This keeps the robot dynamics path
compatible with later rigid contact while making the first milestone visually
deterministic.

## Simulation Data Flow

For each frame:

1. Evaluate the active keyframe interval from simulation time.
2. Interpolate TCP position, orientation, and finger opening.
3. Solve Franka IK and update joint position targets.
4. Advance the robot solver for the configured substeps.
5. Leave inactive cloth particles at their rest positions.
6. Render the shared model state.

No contact pipeline output is consumed by the cloth in this milestone. The
separation is intentional: any later cloth motion must come from an explicit,
reviewable rigid-cloth coupling implementation.

## Validation

The example's `test_final()` will verify only high-value invariants:

- Robot body transforms, joint state, and cloth positions remain finite.
- Cloth particles remain at their flat rest pose.
- The TCP reaches the grasp neighborhood during the sequence.
- The commanded lift pose is measurably above the grasp pose.

A focused example test will construct and run this example with the null viewer.
The primary acceptance check is a GL run showing the table, square cloth, and a
Franka that approaches, closes around the cloth, and lifts without an obviously
incorrect pose or camera angle.

## Future Interaction Milestone

The next design will reactivate the cloth and select an explicit interaction
architecture for:

- cloth-table collision and friction;
- cloth-hand and cloth-finger collision;
- bidirectional force transfer or a documented one-way kinematic proxy;
- stable pinch capture without excessive collision thickness or jitter.

Those choices are intentionally deferred so the first scene provides a clean
visual and geometric baseline for comparing coupling methods.
