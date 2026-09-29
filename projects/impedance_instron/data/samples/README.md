# Cartesian movement samples

These bundles pair prepared measured references with frozen equilibrium
trajectories and simulation traces. They are compact analysis inputs; neither
bundle contains the digital shoe's complete contact-history artifact, so a
replay requires the matching shoe bundle and runtime setup.

Both references contain time, state, velocity, hip and joint targets, native
force time and target, and preparation metadata. Both controller files contain
`duration_s` and a `[12, 4]` equilibrium coefficient array. Trace NPZ files
contain the simulated state, velocity, equilibrium, actuator loads, ground
reaction force, contact moment, and compression diagnostics. Load them with
`numpy.load(path, allow_pickle=False)`.

## Bundles

- [`fr3_1_rate_refit/`](fr3_1_rate_refit/README.md): recovered frozen FR3_1
  rate-refit controller and rollout, with its prepared reference and profile.
- [`fr3_2_peak_to_peak/`](fr3_2_peak_to_peak/README.md): FR3_2 reference
  re-windowed from source time 0.030 to 0.380 s and its full-refit controller
  and rollout. This is an exploratory fit and failed measured-fit acceptance.

The profiles preserve the mass, center-of-mass, inertia, gains, and equilibrium
bounds used for these experiments. They do not certify the inertias or shoe
contact model as physiological measurements.
