# LIMX Cloth Twist Example Design

## Goal

Add a CUDA visualization example that makes the new LIMX cloth
self-collision path easy to inspect. A long rectangular cloth strip has both
short boundary edges anchored and driven in opposite rotations until the
strip twists into self-contact.

## Scene

Create `newton/examples/cloth/example_cloth_limx_twist.py` with a procedural
32-by-12-cell rectangular triangle grid. The rest strip lies in the XY plane,
is centered at the origin, is 1.6 m long along X and 0.6 m wide along Y, and
uses zero gravity so the observed motion is caused by twisting rather than
falling.

All particles on the left and right X boundaries belong to one
`ConstraintAnchor` batch. For a drive angle `theta`, the left boundary is
rotated by `+theta` and the right boundary by `-theta` around the global X
axis through each boundary center. The angle ramps smoothly from zero to two
full turns per side, then remains fixed. The example updates the existing
anchor target array before each captured CUDA solve; it does not rebuild the
solver or its block-CSR topology.

## Solver Configuration

Use the same LIMX path as `cloth_limx`:

- anisotropic `ConstraintTriangleElastic` membrane energy;
- `ConstraintDihedralBending` over all interior edges;
- `ConstraintSelfCollision` with GPU VF/EE detection and EF untangling;
- `dt = 0.01`;
- one Newton iteration per frame;
- 50 PCG iterations;
- `velocity_damping = 1.0`;
- one render per physics step;
- cross-frame PCG warm start.

The collision thickness is one particle diameter and collision stiffness is
`1.0e4 N/m`. Contact topology remains matrix-free. The example adds no
friction, CCD, rigid bodies, ground plane, or explicit damping.

## Runtime and Viewer

Register the example as `cloth_limx_twist` so it runs with:

```bash
python -m newton.examples cloth_limx_twist --device cuda:0
```

Use the standard Newton viewer initialization so the existing pause, single-
step, reset, and rendering controls remain available. Set a camera that shows
the full strip and its thickness during the twist.

## Tests

CUDA tests instantiate the example with `ViewerNull` and verify:

- the solver uses `ConstraintSelfCollision`;
- the timestep, Newton, PCG, and damping settings match the design;
- left and right boundary target cross-sections rotate by equal and opposite
  angles while their X coordinates remain fixed;
- one step advances the state through a CUDA graph and keeps all values
  finite.

After focused tests pass, run a short headless CUDA smoke test and then launch
the interactive CUDA viewer. CPU simulation is not part of validation.
