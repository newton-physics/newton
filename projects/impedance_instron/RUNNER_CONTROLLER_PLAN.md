# One runner controller learned from 100 stances

## Objective

Learn one human equilibrium controller from the 100 FR3_1/FR3_2 training
stances. Given initial pose and velocity, advance that controller with the
leg/shoe physics. Compare shoe materials while holding the learned human
parameters, initial conditions, phase clock, and reset procedure fixed.

The ten held-out stances remain excluded from training and model selection.
Right-foot contacts and right-hip-height peak windows remain the dataset policy.
PPO/RNN work and unit tests remain deferred; experiments provide validation.

## Correction to the previous experiment

`conditioned_controller.Controller` uses each reference's future nominal spline,
desired GRF, and measured duration to generate its coefficients. Its weights
are shared, but its inference requires the movement being reconstructed.
Neither its 84/100 training passes nor its 9/10 evaluation passes establish an
initial-condition-only runner model. Those results remain useful evidence for
the physics derivatives, batching, and fitting machinery.

Remove these inference dependencies rather than reusing that checkpoint as a
human model. Measured motion and GRF are training targets, never runtime inputs.

## Controller contract

The inference API accepts only initial pose `q0`, initial velocity `v0`, and
the frozen human/controller artifact. The phase clock is stored in the artifact
or explicitly supplied as a requested cadence; it must not be inferred from
the end of a held-out reference. Human geometry and impedance gains are frozen
metadata. Material properties belong to the simulated shoe.

Two nested models make the architectural choice concrete:

1. **Fixed shape:** one learned 12-by-6 equilibrium spline used by all stances.
   Initial conditions change the physical response, not the learned shape.
2. **Initial-condition model:** the same shared spline plus a small, shared
   low-rank adjustment from initial pose/velocity. One set of weights generates
   the curve; there are no stance-indexed parameters or reference inputs.

The fixed shape is the baseline. Prefer the initial-condition model only if
held-out experiments show a useful improvement without requiring reference
information. Neither model is a per-stance optimization at inference time.

Hip-x and ankle-x equilibrium coordinates translate together with initial hip
position. The training data have nonzero initial hip-x coordinates, so their
shared seed must be aligned before averaging. This translation uses initial
state only and is independent of the material.

The existing impedance actuator responds to simulated state at every step.
Generating an equilibrium spline once therefore still permits causal state
feedback through impedance, while the equilibrium plan is fixed for that
rollout. Additional feedback adaptation of the plan is a separate experiment.

## Implementation stages

### 1. Explicit reference-independent artifact

Add `runner_controller.py` with both nested models, train-only statistics,
shared parameters, common phase period, spline projection, network/projection
VJP, and save/load support. Training references may initialize one aligned mean
seed; they must not be retained or queried by inference. Do not initialize from
the previously fitted per-reference controller weights.

### 2. Separate phase clock, simulation horizon, and objective

The current `Engine` and `BatchAdjoint` derive their spline basis from each
reference's duration. Decouple the controller clock from the measurement
horizon before claiming causal training or evaluation. A recorded endpoint may
define the available scoring interval; it cannot stretch the equilibrium plan.

Use a training-only shared period initially (current training median is
0.35499954 s). Define and report support explicitly when a recorded window is
shorter or longer than the generated trajectory. Do not silently clip phase,
repeat an endpoint, or normalize each held-out trace to its measured duration.
Score observed samples within the generated one-cycle support and report
coverage; later periodic continuation requires a separate boundary design.

The current fit has reference-derived hip-flight gating disabled. Preserve
that setting. Future feedback/contact gates must use simulated state/contact
or the stored phase policy, never measured future GRF events.

The engine now separates a target-free rollout API from measured-objective
scoring. `Engine.from_initial` constructs the plant without a reference or
measured objective; `rollout` advances physics and reports failure/completion.
`make_params_initial` validates only the frozen geometry and human profile.
No fake future measurements are required to compare materials.

### 3. Train one model across the population of stances

Reuse qualified full-rollout physics gradients and small trial-balanced batches.
Every batch updates the same human/controller parameters. Aggregate losses
equally over stances; retain kinematic/GRF targets as identification evidence,
without claiming that every noisy stance should be reproduced exactly.

Start with the fixed model, then compare the small initial-condition extension.
Penalize unnecessary adjustment magnitude and complexity. Preserve spline
bounds and reject incomplete simulations. Select checkpoints using training
loss only; score the held-out block after freezing the model.

Initially freeze existing impedance gains and calibrated body properties to
isolate controller learning. Jointly identifying them changes the inverse
problem and should be evaluated as its own experiment.

### 4. Experiments that establish the requested behavior

- Hold `q0`, `v0`, geometry and phase clock fixed; replacing or deleting future
  reference kinematics/GRF must not change generated coefficients or rollout.
- Full predictor-to-physics directional check for the revised controller and
  clock, retaining the existing tolerance and nonsmooth-contact limitations.
- All-100 training and ten held-out rollout distributions: mean/median/tails,
  failures, per-channel errors, contact timing and completion coverage.
- Identical initial state and shoe-history reset across material variants;
  no human reoptimization. Verify controller coefficients remain identical
  when only material changes. Report GRF, impulse, contact timing, hip/joint
  motion, actuator work and numerical failures from simulated outcomes.
- Native/half-timestep checks on the frozen held-out and material experiments.

A good training score alone does not demonstrate the reference-independent
contract. The input-removal and frozen-material experiments are required.

## Interpretation and scope

This identifies a one-cycle runner model within the observed initial-condition
range. A single right-leg peak-to-peak model does not yet establish autonomous
continuous bilateral running. Material comparisons measure the immediate
response of the same modeled runner; adaptation by a human to a new material
requires a separately stated controller model and evidence.

## Status

- Inspected current feature generator, inference, phase gate and spline clocks.
- Previous fitted model explicitly identified as reference-conditioned.
- Implemented the separate `cartesian/gpu/runner_controller.py` interface.
  Fixed mode has 72 shared parameters; the rank-four initial-condition model
  has 396. Both runtime APIs accept only initial pose and velocity, and use
  the same training-only phase duration and stored geometry. These are
  untrained prototypes, not replacement fitted human artifacts.
- CPU inference over all 110 initial states, spline-bound checks, finite VJP
  checks, and exact save/load reproduction pass. Evidence is
  `outputs/impedance_instron/runner_controller_contract_20260929/report.json`.
  This is an interface/serialization experiment, not physics qualification.
  The prior fitting experiment and its reports are preserved.
- Integrated fixed-period basis evaluation into the production engine and
  batch adjoint, retaining legacy default behavior. Fixed human lengths are
  training-only means; reference-derived contact gating is rejected in causal
  mode. Scoring support is cropped and recorded without changing phase.
- Fixed mode now projects one shared curve onto the intersection of bounds for
  the training translation range, then translates it; projection no longer
  creates a different shape for each initial state. Adam projects shared
  parameter updates and uses actual forward loss to accept/reject proposals.
- Four cross-trial qualification rollouts reproduce batch/production states
  and forces exactly. Target-free rollouts also reproduce them exactly. Three
  shared-coefficient finite-difference directions tangent to active constraint
  faces pass the unchanged criterion. Evidence:
  `outputs/impedance_instron/runner_shared_fit_20260929/qualification.json`.
- The from-scratch fixed-curve fit completes twenty passes through all 100
  training stances, 500 accepted updates, and visits every training stance
  twenty times. Training loss 18.390559 -> 1.412203; held-out loss 18.475400
  -> 1.285769. All 110 scored rollouts complete; native all-six passes are
  74/100 training and 9/10 evaluation. Search takes 1443.91 s; total fitting
  takes 1491.42 s. No fitted checkpoint initializes it. Evaluation is excluded
  from updates, controller initialization, and checkpoint selection.
- `runner_rollout` exposes initial-condition-only deployment and material
  substitution. Independent native scores agree exactly on all 110 stances.
  All 110 initial conditions complete the stored period without future targets.
  All ten half-timestep held-out rollouts complete, retaining 9/10 threshold
  passes. All 30 synthetic material rollouts complete with identical human
  coefficients across materials; warmed forward physics takes approximately
  0.46 s. The completed HTML report is
  `runner_shared_fit_20260929/from_scratch_20/report.html`.

See [RUNNER_CONTROLLER_RESULTS.md](RUNNER_CONTROLLER_RESULTS.md) for recorded
results and artifact paths.
