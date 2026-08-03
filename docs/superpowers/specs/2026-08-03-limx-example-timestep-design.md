# LIMX Example Time-Step Design

## Goal

Run the LIMX cloth visualization with a 0.01 s physics time step and render every physics step while testing the projected-Newton solver with one nonlinear iteration and 50 PCG iterations.

## Configuration

- Set the example frame rate to 100 Hz, so `frame_dt` is 0.01 s.
- Set `sim_substeps` to 1, so `sim_dt` is also 0.01 s.
- Render once after every call to the solver; do not decimate visualization frames.
- Set `nonlinear_iterations` to 1.
- Set `linear_iterations` to 50.
- Leave masses, spring stiffnesses, anchors, gravity, and damping behavior unchanged.

## Runtime Flow

Each displayed frame performs exactly one physics step:

1. Clear and apply external forces.
2. Advance the LIMX solver by 0.01 s using one Newton iteration.
3. Allow PCG up to 50 iterations for that Newton system.
4. Advance simulation time by 0.01 s.
5. Render the resulting state.

## Verification

- Add a regression assertion for the example's time-step and iteration configuration before changing the example.
- Run the focused regression test and confirm it fails with the old 1/240 s, 4-Newton, 32-PCG configuration.
- Apply the configuration change and confirm the focused test passes.
- Run the visualization example on CPU and CUDA in test mode long enough to cover one simulated second.
- Launch the interactive visualization after automated verification.
