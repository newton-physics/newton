# Physics backpropagation experiment

## Objective

Reduce fitting time while retaining the quality of the existing 12-control,
six-channel (72-coefficient) Cartesian leg and digital-shoe fit. Qualify the
physics gradient on one stance before using it across the 100 training stances.
PPO and RNN work is deferred.

The experiment is the validation. Do not add or run unit tests at this stage.

## Fixed inputs and comparison rules

- Start with `outputs/impedance_instron/fr3_2_ankle_xy_refit_20260929/fit_run2`.
- Preserve its physics, timestep, initial state, material, friction, measured
  objective, spline bounds, failure screens, and acceptance thresholds.
- Preserve the existing 100-train/10-eval dataset: FR3_1 and FR3_2 only, right
  foot contact bracketed by preceding/following right-hip-height peaks.
- Evaluation stances cannot select updates or optimizer settings.
- Record source/input hashes, hardware, memory, compilation/capture cost,
  warm execution time, loss, channel errors, and failures.
- Serialize GPU experiments. Keep saved fits and previous experiments intact.
- Compare time to the same loss and channel quality, not iteration count alone.

## Source foundations

- `newton/examples/diffsim/example_diffsim_bear.py`: separate differentiable
  states per substep; Warp Tape; separate forward/backward CUDA graphs.
- `cartesian/gpu/mechanics.py`: existing custom adjoint of the symmetric
  mass-matrix solve.
- `cartesian/gpu/adjoint_contact.py`: existing adapter retaining continuous shoe
  memory and branch state, with shared material/contact functions.
- `cartesian/gpu/adjoint.py`: experimental coupled rollout and coefficient VJP.
- `cartesian/gpu/adjoint_audit.py`: forward parity and directional finite
  difference comparisons. Existing comments are not qualification evidence.

## Execution stages

### 1. Reconstruct the saved six-channel experiment

Load the saved input metadata and check artifact hashes. Support the actual
saved-fit format without manufacturing historical qualification manifests.
Create output reports under `outputs/impedance_instron/physics_backprop_20260929`.

Acceptance: matching duration, 12-by-6 coefficients, simulation and fitting
configuration, shoe artifact and mount. Record current source identities.

### 2. Qualify forward and backward over increasing windows

Run 32-, 128-, and 512-step coupled windows, including loaded contact. Compare
retained states, velocities, forces, controls and final shoe memory against the
production engine. Compare coefficient directional derivatives against central
finite differences at multiple epsilon values. Include contact transitions.

Acceptance: retain existing forward-parity requirements; require adjacent
epsilon agreement within the audit's existing 1% relative plus 1e-5 absolute
tolerance. Report branch changes and float32 noise explicitly. Fix demonstrated
errors without changing the physics or loosening thresholds.

Diagnostic windows hold their initial checkpoint fixed. Their gradients omit
the preceding trajectory and do not establish full-stance correctness.

### 3. Qualify the complete measured objective

Run the entire stance from its fixed initial state with the original measured
loss. Check forward parity and directional gradients again. Capture forward
and backward separately; measure repeated replay, gradient repeatability and
peak GPU memory. Preserve all shoe history needed by backward.

Acceptance: complete finite rollout, original measured score matches production,
and finite differences support the full coefficient gradient. If this fails,
save the evidence and diagnose the failing horizon/branch before optimization.

### 4. Build the gradient fitting experiment

Use normalized coefficient coordinates and a bounded, safeguarded optimizer
(initially L-BFGS with backtracking). Every proposed controller must satisfy the
same position, rate and acceleration bounds. Failed or incomplete rollouts are
rejected. Use the production engine for independent candidate/final scoring.
Do not substitute a diagnostic scalar for the measured objective.

A scalar loss VJP does not produce the residual Jacobian used by Gauss-Newton.
Keep optimizer convergence and derivative computation cost separate in reports.

Acceptance: actual measured loss decreases with bounds and failure screens
preserved; save coefficients, progress and production channel metrics.

### 5. Compare fitting quality and wall time

First compare short runs from identical saved coefficients. Then compare from
the same unfitted reference controller toward the saved fit's quality. Record
setup cost and search time separately, including rejected line searches.

Acceptance: demonstrate time to comparable loss and individual channel errors;
report honestly if gradients are correct but optimization is slower or worse.

### 6. Extend to multiple stances

After single-stance qualification, use small trial-balanced training batches.
Start with independent coefficients to separate optimizer performance from the
capacity of a shared controller. Then build a reference-conditioned coefficient
predictor and optimize it with measured physics losses. The previous single
shared residual remains a comparison, not a required controller architecture.

Use all 100 training stances and score the 10 held-out stances without adapting
to them. Bound GPU memory through small batches and, if required, checkpointing
with exact replay of shoe memory. Truncated gradients must be labeled as such.

## Status

- Plan written; source investigation complete.
- Stage 1 reconstructed the saved six-channel fit. Stale experimental interfaces
  were updated to pass the hip gate and ankle force through the current engine.
- Fixed a replay shortcut that cut the shoe-force gradient back to carrier
  kinematics. Loaded-contact 32-, 128-, and 512-step experiments now pass exact
  forward parity and directional finite differences.
- The full unfitted reference controller passes all three measured-objective
  gradient directions with the unchanged criterion. Its captured value plus
  gradient takes approximately 1.60 s; retained Warp memory is approximately
  1.17 GiB. See `physics_backprop_20260929/audit_full_unfitted.json`.
- The saved fitted controller reproduces its loss exactly, but only one of
  three full measured-objective gradient directions passes. Perturbations change
  contact branches, including at the smallest tested epsilon. This remains a
  gradient qualification limitation; its cause is not fully established.
- A separate frozen-trajectory objective experiment passes state and force
  derivatives with maximum absolute error 1.4e-13. Full terminal-state gradients
  also pass. See `physics_backprop_20260929/frozen_objective.json`.
- Plain bounded backtracking stalled near spline constraints. Dykstra projection
  of L-BFGS proposals onto the original coupled spline bounds fixed that issue;
  the production solver still checks every proposed update independently.
- The matched five-update experiments took 10.36 s with loss 3.314 for gradients,
  versus 13.50 s with loss 2.754 for the original 192-world fitter. This alone
  did not establish faster convergence to equal quality.
- The longer gradient fit starts from the exact original unfitted coefficients:
  loss 24.030 -> 0.518976 in 40 updates and 83.20 search seconds. Setup takes
  20.33 s; fitting including setup/final reporting takes 104.26 s. The original
  saved fit reached 0.520437 in 200 updates and 578.73 search seconds. This is
  approximately 6.96 times faster in search on this one reference stance.
- All six native and fine-timestep RMS thresholds pass. The existing half-step
  refinement passes: maximum differences 0.0216 mm hip, 0.141 mrad joint angle,
  and 2.266 N GRF. The frozen result is numerically accepted by those checks.
- Stages 1-5 have experiment evidence. Stage 6 now has a reusable four-world
  full-rollout adjoint, including different geometry, durations, actual
  timesteps, native sample counts, and shoe-history padding. Independent
  production state, force and loss parity passes on both trials; per-world
  gradients agree with the separate adjoints within 4.1e-7 relative error.
- A train-only PCA16/tanh32 coefficient predictor and spline-projection VJP
  pass the recorded end-to-end directional experiment. The four-world
  forward-plus-backward replay takes 2.01 seconds. Artifacts are under
  `outputs/impedance_instron/physics_backprop_100_20260929`.
- The from-scratch ten-epoch experiment completes all 100 training stances,
  visiting each ten times, with all 250 updates accepted. Training mean loss
  falls from 17.9019 to 0.968268; evaluation falls from 17.4754 to 0.993533.
  Native six-channel acceptance is 84/100 training and 9/10 evaluation.
  Search takes 942.74 s; total fitting takes 1006.93 s. Checkpoints use
  all-training loss only; evaluation never selects weights or feature statistics.
  See `physics_backprop_100_20260929/train_10_epochs/report.html`.
- Independent frozen-controller production scoring completes all 110 stances
  and agrees with batch losses within 7.99e-15. Half-timestep evaluation
  completes all ten held-out stances, retaining 9/10 all-six passes with mean
  loss 0.994006. No training-wide half-timestep claim is made. This additional
  experiment takes 89.04 s; see `physics_backprop_100_20260929/frozen_qualification.json`.
- No unit tests were added or run. Experiment reports provide validation;
  formatting, lint and whitespace checks cover the edited code.

Update this section with report paths, measured results and remaining work as
the experiment progresses.

See [PHYSICS_BACKPROP_RESULTS.md](PHYSICS_BACKPROP_RESULTS.md) for the comparison
and artifact paths. Keep PPO/RNN work deferred.

## Commands

Run from `/home/jkuzm/projects/newton`. Each command writes a new report; choose
an unused output path when rerunning. Do not run fitting after a failed audit.

```bash
uv run -m projects.impedance_instron.cartesian.gpu.adjoint_audit \
  --baseline outputs/impedance_instron/fr3_2_ankle_xy_refit_20260929/fit_run2 \
  --output outputs/impedance_instron/physics_backprop_20260929/full_measured_new.json \
  --steps 5600 --objective measured --controller-start unfitted --capture-gradient

uv run -m projects.impedance_instron.cartesian.gpu.gradient_fit \
  --baseline outputs/impedance_instron/fr3_2_ankle_xy_refit_20260929/fit_run2 \
  --qualification outputs/impedance_instron/physics_backprop_20260929/full_measured_new.json \
  --output outputs/impedance_instron/physics_backprop_20260929/gradient_fit_new \
  --iterations 250 --wall-seconds 540 --start unfitted \
  --target-loss 0.5204371516067673

uv run -m projects.impedance_instron.cartesian.gpu.gradient_qualify \
  --fit-directory outputs/impedance_instron/physics_backprop_20260929/gradient_fit_new \
  --output outputs/impedance_instron/physics_backprop_20260929/qualification_new

uv run -m projects.impedance_instron.cartesian.gpu.gradient_train \
  --dataset outputs/impedance_instron/stance_dataset_peak_hip \
  --fit-directory outputs/impedance_instron/fr3_2_ankle_xy_refit_20260929/fit_run2 \
  --qualification outputs/impedance_instron/physics_backprop_100_20260929/conditioned_physics_qualification.json \
  --output outputs/impedance_instron/physics_backprop_100_20260929/train_new \
  --epochs 10 --learning-rate 0.003

uv run -m projects.impedance_instron.cartesian.gpu.gradient_train_qualify \
  --run outputs/impedance_instron/physics_backprop_100_20260929/train_new \
  --output outputs/impedance_instron/physics_backprop_100_20260929/frozen_qualification_new.json
```
