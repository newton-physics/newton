# Shared runner identification from scratch

## What was fitted

One shared 12-by-6 equilibrium spline (72 learned coefficients) across all 100
training stances from FR3_1/FR3_2. Initial horizontal position translates the
same shape; initial pose and velocity enter the physical state. The phase
period is a training-only median, 0.35499954 s. Human geometry is fixed to
training-only mean segment lengths; existing body properties and impedance
gains remain fixed. No per-stance controller parameters are fitted.

The previous reference-conditioned neural checkpoint is not used. The fresh
seed is one aligned mean of unfitted training-reference nominal curves. Future
motion and measured GRF supply identification targets only; runtime inference
uses initial state and velocity without future references or measured duration.

## From-scratch experiment

| Quantity | Initial | Fitted |
| --- | ---: | ---: |
| All 100 training mean loss | 18.390559 | 1.412203 |
| Ten held-out mean loss | 18.475400 | 1.285769 |
| Training native all-six pass | 0/100 | 74/100 |
| Held-out native all-six pass | 0/10 | 9/10 |

All 110 scored rollouts complete. The twenty-epoch experiment visits each
training stance twenty times and accepts 500 updates using actual physics
loss checks. Checkpoint 500 is selected using complete all-training mean loss;
held-out data never select weights, initialization, phase, geometry or settings.

Setup/initial scoring takes 33.63 s. Search including checkpoint scoring and
proposal checks takes 1443.91 s(24.07min). Total fitting takes 1491.42 s(24.86min).
The eighteen-epoch checkpoint has mean loss 1.412398, close to the final result;
the experiment is approaching a plateau, not evidence of an optimal solution.

Recorded endpoints limit available scoring observations without stretching the
shared controller clock. Native-grid scoring covers 97.26–100% of each recorded
window, mean 99.76%. Longer windows have an explicitly excluded tail. The full
stored-period continuation experiment is reported separately.

Six-channel limits are 20 mm per hip axis, 50 mrad per joint axis, and 100 N per GRF
axis. Training threshold failures by channel are 0 hip-x, 2 hip-z, 8 joint-1,
15 joint-2, 1 GRF-x,15 GRF-z; some overlap. Held-out failures are 0 hip-x,
0 hip-z, 1 joint-1, 1 joint-2, 0 GRF-x, 1 GRF-z. Per-channel distributions and every
stance's values are included in the HTML report.

These results describe a population controller, rather than reconstructing each
recorded stance exactly. They should not be compared as equivalent-quality
scores to the earlier model that consumed future reference information.

## Experimental qualification

Before fitting, four cross-trial rollouts reproduce production states and shoe
forces exactly. Target-free production rollouts also reproduce them exactly.
Three shared-coefficient finite-difference directions tangent to active spline
constraint faces pass the unchanged adjacent-epsilon 1% plus 1e-5 criterion.
Contact and constraint transitions remain nonsmooth; this is not a guarantee
of exact derivatives at every fitted state. Actual candidate rollouts decide
accepted updates.

Independent frozen native production scores agree exactly on all 110 stances.
All 110 initial conditions complete target-free full-period rollouts, including
continuation beyond shorter recordings. Removing runtime dataset/reference
paths does not change held-out states, shoe forces or generated coefficients.

All ten held-out half-timestep rollouts complete, retaining 9/10 all-six passes;
mean loss 1.285618 versus native 1.285769. Training-wide half-timestep validation
was not performed. The frozen experiment takes 169.49 s separately from fitting.
No unit tests are added or run.

## Frozen-human material sensitivity

Thirty target-free full-period rollouts use the ten held-out initial conditions
and synthetic 0.8/1.0/1.2 modulus scales. Both Hyperfoam shear-modulus terms and
their derived Pasternak coupling scale together; geometry, relaxation and
friction remain unchanged. Each rollout resets shoe history. Human coefficients
remain bitwise identical across material variants. All 30 rollouts complete.
These are synthetic parameter sensitivities, not calibrated alternative foams.

| Modulus scale | Mean peak vertical GRF (N) | Mean vertical impulse (N·s) | Mean contact (s) | Mean positive actuator work (J) |
| --- | ---: | ---: | ---: | ---: |
| 0.8 | 1334.24 | 201.00 | 0.25617 | 143.43 |
| 1.0 | 1347.46 | 204.23 | 0.25603 | 144.98 |
| 1.2 | 1360.68 | 206.94 | 0.25577 | 146.57 |

Warmed full-period forward physics takes approximately 0.46 s per material/state,
excluding engine setup, graph capture and trace reporting. No human refitting
is performed. A material exemplar also completes at half timestep for all
three variants, with peak vertical GRF differences below 0.062 N.

## Artifacts and deployment

Paths are relative to `outputs/impedance_instron/runner_shared_fit_20260929`:

- `qualification.json`: pre-fit physics and derivative evidence.
- `from_scratch_20/runner_controller.npz`: frozen shared human artifact.
- `from_scratch_20/initial_controller.npz`: unfitted common seed.
- `from_scratch_20/report.json`: progress, timings, coverage and 110 native scores.
- `from_scratch_20/report.html`: standalone experiment report.
- `from_scratch_20/sources/`: source snapshots and hashes.
- `frozen_evaluation/`: independent frozen physics and material experiments.

`runner_rollout` loads the human artifact without reading the dataset or future
targets. Supply five initial coordinates and five velocities. Optionally supply
a shoe artifact or a synthetic modulus scale. Material comparisons preserve
the human coefficients and reset shoe history for each rollout. Commands and
interface details are in `cartesian/gpu/README.md`.

`example_initial_conditions.json` contains only initial state/velocity vectors
and a source label; `standalone_example/` demonstrates the deployment CLI with
a 1.2 modulus scale, without loading future reference targets.
