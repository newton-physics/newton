# LIMX–ChysX Twist Scene Alignment Design

## Goal

Make the LIMX twist example reproduce the physical setup of the stable
`newton-ChysX` twist reference closely enough that their visible behavior can
be compared without changing LIMX collision detection or response.

## Scope

Only edit `example_cloth_limx_twist.py` and its focused example assertions.
Do not change VF, EE, EF, topology filtering, Hessian assembly, damping, or
the LIMX solver. Keep one 0.01 s physics step per rendered frame, one Newton
iteration, and 50 PCG iterations.

## Scene

Generate the reference grid procedurally so Newton does not depend on a file
from the neighboring `newton-ChysX` checkout:

- 200 columns by 100 rows (20,000 particles and 39,402 triangles);
- dimensions 1.0 m along Z by 0.5 m along Y, initially at X = 0;
- checkerboard diagonals matching `twist100.obj`;
- the 100 particles on each Z boundary are anchored;
- each boundary rotates in the XY plane at a constant 1 rad/s, in opposite
  directions;
- gravity is `(0, -9.8, 0)` and the default run is 1,000 frames.

Use area-weighted particle masses with surface density 0.1 kg/m², giving a
total cloth mass of 0.05 kg. Set membrane warp, weft, and shear stiffness to
500 N/m, bending stiffness to `5e-5`, and anchor stiffness to `1e9`.

## Collision Configuration

Let `edge_length = 0.5 / 99`. Set proximity thickness to
`1.2 * 0.2 * edge_length` (approximately 1.212 mm), VF/EE stiffness to 1,000,
and EF untangle stiffness to 2,000. Keep EF enabled. LIMX currently exposes
one thickness for proximity and untangling, so both paths use the 1.212 mm
value; adding a separate EF thickness would be an algorithm/API change and is
outside this scene-only alignment.

Allocate 131,072 contacts per feature type. This exceeds the approximately
30,000 proximity contacts observed in the ChysX reference while avoiding the
much larger reference allocation that its different buffer layout uses.

## Validation

Update the focused unittest assertions before changing the example and verify
that they fail on the old scene. Assert the aligned mesh size, dimensions,
mass, drive law, material values, collision values, timestep, and solver
iterations. Then run the focused CUDA tests and a headless 1,000-frame rollout
that reports VF, EE, EF, overflow, finite state, and maximum speed. Finally
launch the viewer for visual acceptance.
