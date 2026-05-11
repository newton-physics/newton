# Pressure Field Modulation Report

## Scope

The current implementation modifies the sampled hydroelastic pressure field in the visualization pipeline, not the core contact solver.

Base pressure is sampled from the immutable volume, then post-modulated before rendering in:

- `newton/examples/contacts/example_hydro_pressure_slice.py:87`
- `newton/examples/contacts/example_hydro_pressure_slice.py:781`

## How The Field Is Changed

1. Base pressure sampling  
   `p0(x,y,z)` is sampled from the hydroelastic volume in `sample_pressure_on_slice`.

2. Axis sine modulation (X/Y/Z)  
   Each axis applies a multiplicative factor:

   `M_a = max(0, 1 + A_a * sin(2*pi*f_a*xi_a + phi_a))`

   where `xi_a` is axis coordinate normalized by SDF center and half-extent.  
   Implemented in `apply_pressure_axis_sine_modulation`.

3. Foot-structure modulation (bone lobes + arch trough)  
   A localized field is built from Gaussian components representing:

- calcaneus
- talus
- navicular
- cuboid
- metatarsals

   minus a medial arch trough:

   `M_foot = clamp(1 + s*(0.45*B - 0.80*d*A), 0, 3)`

   where:

- `s` = structure strength
- `d` = arch depth
- `B` = weighted bone-lobe sum
- `A` = arch Gaussian

   Implemented in `apply_pressure_foot_structure_modulation`.

4. Final pressure

   `p = p0 * M_x * M_y * M_z * M_foot`

## Controls Available

- Axis sine CLI controls in parser:
  - `--pressure-x-sine-amplitude`
  - `--pressure-x-sine-cycles`
  - `--pressure-x-sine-phase`
  - `--pressure-y-sine-amplitude`
  - `--pressure-y-sine-cycles`
  - `--pressure-y-sine-phase`
  - `--pressure-z-sine-amplitude`
  - `--pressure-z-sine-cycles`
  - `--pressure-z-sine-phase`

- Foot-structure CLI controls:
  - `--pressure-foot-structure-strength`
  - `--pressure-foot-arch-depth`
  - `--pressure-foot-medial-bias`

- UI controls include:
  - `Pressure Foot Strength`
  - `Pressure Foot Arch Depth`
  - `Pressure Foot Medial Bias`

## How To Optimize To A Specific Person

1. Collect subject data  
   Gather plantar pressure insole/plate data across gait phases plus a 3D foot scan (or key dimensions).

2. Register coordinate frames  
   Align subject foot frame to model axes:

- `x`: heel -> toe
- `y`: lateral <-> medial
- `z`: plantar -> dorsal

   Normalize by SDF extents.

3. Define objective  
   Minimize mismatch between measured and predicted pressure:

   `L(theta) = w1*||P_pred - P_meas||^2 + w2*|peak_pred - peak_meas| + w3*|COP_pred - COP_meas| + lambda*||theta - theta0||^2`

   where `theta` is the modulation parameter vector.

4. Optimize in two stages

- Stage 1 (global): Bayesian optimization or CMA-ES
- Stage 2 (local): Nelder-Mead or CMA-ES refinement

5. Validate

- Hold out trials/speeds
- Compare region-wise peak pressure
- Compare center-of-pressure trajectory
- Compare pressure-time integrals
- Check stability across repeated runs

## Important Limitation

If the goal is subject-specific contact physics (not only visualization), the same modulation logic should be moved into the hydroelastic field construction used by contact resolution, not only this example's rendering path.
