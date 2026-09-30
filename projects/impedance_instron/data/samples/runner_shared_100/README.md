# One shared runner controller

This bundle preserves the completed from-scratch identification experiment:
one shared 72-coefficient equilibrium curve, learned from 100 FR3_1/FR3_2
stances with ten held-out stances. Runtime uses initial pose and velocity;
future motion, GRF, and recorded stride duration are not controller inputs.

- `runner_controller.npz`: frozen human curve, training-only period and
  geometry, profile, simulation settings, and optimizer state.
- `report.html`: standalone fit and material experiment report.
- `fit_report.json`: training progress, native errors, timing and scoring support.
- `frozen_evaluation.json`: independent native/half-step and material experiments.
- `qualification.json`: pre-fit production parity and directional derivatives.
- `example_initial_conditions.json`: initial vectors for a deployment example.
- `sha256.json`: bundle file hashes.

The fitted training/held-out mean losses are 1.412203/1.285769. All 110 initial
conditions complete target-free full-period rollouts; all-six tracking passes
are 74/100 training and 9/10 held out. All 30 synthetic material rollouts
complete with unchanged human coefficients. These synthetic modulus variants
are sensitivities, not calibrated alternative materials.

The complete shoe artifact and raw dataset are not included. Supply a matching
shoe artifact explicitly to deploy this portable controller. Historical paths
inside metadata record experiment provenance; they are not portable data paths.

```bash
uv run -m projects.impedance_instron.cartesian.gpu.runner_rollout \
  --controller projects/impedance_instron/data/samples/runner_shared_100/runner_controller.npz \
  --initial-conditions projects/impedance_instron/data/samples/runner_shared_100/example_initial_conditions.json \
  --shoe-artifact /path/to/digital_shoe.json \
  --output outputs/impedance_instron/runner_material_new
```

See `projects/impedance_instron/RUNNER_CONTROLLER_RESULTS.md` for scope and
limitations. Experiment evidence provides validation; no unit tests were added
or run for this work.
