# Physics backpropagation: single-stance experiment

Date: 2026-09-29. Hardware: NVIDIA RTX 4070, 12 GiB. Warp:
1.17.0.dev20260807. Run through `uv` from the Newton repository root.

## Result

The projected L-BFGS experiment reaches comparable measured fitting quality
with approximately **6.96 times less search time** on the original reference
stance. Its frozen winner passes the existing native and half-timestep
acceptance checks. This establishes a single-stance result, not 100-stance
performance or general gradient correctness at every contact transition.

| Measure | Saved original fitter | Physics-gradient experiment |
| --- | ---: | ---: |
| Coefficients | 72 | 72 |
| Simulation worlds | 192 candidates | 1 differentiable rollout |
| Starting loss | 24.030347 | 24.030347 |
| Final loss | 0.520437 | 0.518976 |
| Updates | 200 | 40 |
| Search time | 578.73 s | 83.20 s |
| Gradient setup | N/A | 20.33 s |
| Gradient fit including setup and final reporting | N/A | 104.26 s |

The search comparison excludes setup and final qualification for both methods.
The new half-timestep qualification is a separate experiment and is not included
in the reported 104.26 s fitting time. The original result is the saved prior
run. A new matched five-update experiment also ran during this investigation.

| Native RMS error | Original | Gradient winner |
| --- | ---: | ---: |
| Hip x [mm] | 4.383 | 3.705 |
| Hip z [mm] | 17.021 | 16.767 |
| Knee [mrad] | 13.376 | 13.552 |
| Ankle [mrad] | 12.652 | 11.899 |
| Force x [N] | 33.152 | 35.690 |
| Force z [N] | 15.178 | 20.811 |

Aggregate loss is slightly better. Individual force errors are slightly higher;
all channels remain within the original thresholds. No material, timestep,
controller bounds, failure screens or measured objective were changed.

## What changed

- Repaired experimental interfaces for six control channels, the hip gate,
  ankle Cartesian force and the current shared contact helper signature.
- Restored a differentiable carrier-kinematics path around an empty replay
  shortcut that cut the shoe-force gradient back to the leg state.
- Read measured forces through a separate differentiable buffer from the same
  shoe wrench used by integration.
- Preserved shoe memory and intermediates per timestep and captured forward
  and backward separately. One full value/gradient replay takes about 1.60 s.
- Added normalized L-BFGS with Dykstra projection onto the original linear
  spline position/rate/acceleration inequalities. Every line-search proposal
  is scored by the production solver; failed or out-of-bounds proposals cannot
  earn an accepted update.

## Experiment evidence and limits

Short contact windows pass forward and directional-gradient checks. The complete
unfitted controller passes all three directional checks at the existing 1%
relative plus 1e-5 absolute criterion, requiring adjacent epsilon agreement.

At the saved fitted controller, the complete measured-loss gradient passes only
one of three directions. Contact branches change under perturbations, including
the smallest tested epsilon. This supports a contact-sensitivity explanation
but does not rule out a remaining adjoint defect. A full terminal-state
gradient passes; the measured objective on frozen trajectories passes state and
force derivatives with maximum absolute error 1.4e-13. The limitation remains
recorded instead of loosening the tolerance.

The optimizer starts only after the unfitted-controller audit passes, and
production backtracking checks actual loss reduction on every accepted update.
Final fitting quality and timestep refinement are checked independently.

Plain L-BFGS with only backtracking barely improved the initial loss because a
coupled spline bound forced tiny steps. Projection fixed this demonstrated
optimization issue. The unsuccessful experiment is retained.

The fresh five-update comparison starts from identical coefficients:

| Method | Search time | Final loss |
| --- | ---: | ---: |
| Projected gradient fitter | 10.36 s | 3.314375 |
| Original 192-world fitter | 13.50 s | 2.753951 |

The longer run, rather than this iteration-count comparison, provides the
evidence for time to comparable fitting quality.

The half-step check integrates 11,200 steps at 3.125e-5 s and passes with maximum
differences of 0.0216 mm hip position, 0.141 mrad joint angle and 2.266 N GRF.
Native and fine-step measured RMS thresholds both pass.

No unit tests were added or run. These experiments are the validation.

## Artifacts

Paths below are relative to `outputs/impedance_instron/physics_backprop_20260929`:

- `audit_full_unfitted.json`: passing full measured-gradient experiment.
- `audit_full_measured_final.json`: retained failed fitted-point qualification.
- `audit_512_step2000.json`: contact-window gradient and timing evidence.
- `frozen_objective.json`: objective derivative isolation.
- `gradient_fit_5_unfitted/`: unsuccessful plain bounded-backtracking experiment.
- `gradient_fit_5_projected/`: projected five-update result and source snapshot.
- `comparison_5_unfitted/comparison.json`: matched original-fitter experiment.
- `gradient_fit_long/report.json`: complete progress, timing and channel scores.
- `gradient_fit_long/gradient_controller.npz`: frozen winner [12, 6].
- `gradient_fit_long/gradient_fit_source.py`: source used for the longer fit.
- `gradient_fit_long_qualification/qualification.json`: native/fine acceptance.
- `gradient_fit_long_qualification/`: native and refined trajectory archives.

## Multi-stance implementation

The reusable four-world batch preserves each stance's geometry, duration,
actual timestep, native measurement sample counts, initial state and shoe
memory. Padding after each terminal step is inactive. Separate production
rollouts reproduce its states, forces and objective exactly on the recorded
cross-trial probe. Two per-world gradients agree with individual adjoints to
4.1e-7 relative error; partial batches and A/B/A graph reuse pass.

Four-world captured forward-plus-backward replay takes 2.01 seconds. Retained
batch history uses approximately 4.89 GiB; the parity experiment including a
temporary individual adjoint uses 6.07 GiB. Full trajectories are differentiated;
there is no truncated horizon.

The shared coefficient predictor uses training-only standardization and PCA16,
a tanh32 hidden layer, and 72 normalized residual outputs. Its inputs include
nominal reference coefficients, initial state, duration, geometry, and desired
reference GRF sampled at twelve phases. Desired GRF conditions the predictor;
the shoe simulation still computes its own forces. Zero decoder initialization
reproduces the unfitted nominal controllers; no fitted labels are supplied.

Original spline position/rate/acceleration bounds are enforced with convex
projection. An active-face projection VJP connects physics gradients to the
network. The recorded full predictor-to-physics directional experiment passes
the same adjacent-epsilon 1% plus 1e-5 criterion. Contact/bound transitions
remain nonsmooth, and this does not remove the fitted-point gradient limitation
described above.

Training uses two stances from each trial per batch and visits all 100 stances
once per epoch. Adam proposals require actual forward-loss reduction; complete
all-training checkpoint scores select the winner. The ten evaluation stances
are excluded from updates, feature statistics and checkpoint selection.

Evidence is under `outputs/impedance_instron/physics_backprop_100_20260929`:

- `batch_qualification.json`: independent production parity and graph reuse.
- `controller_probe.json`: predictor and projection derivative experiment.
- `conditioned_physics_qualification.json`: end-to-end directional experiment.
- `train_10_epochs/`: from-scratch 100-training/10-evaluation run and HTML report.

## Completed 100-stance fit

| Quantity | Earlier common-residual population search | Conditioned physics-gradient model |
| --- | ---: | ---: |
| Training initial loss | 17.901881 | 17.901881 |
| Training final mean loss | 3.222979 | 0.968268 |
| Held-out initial loss | 17.475421 | 17.475421 |
| Held-out final mean loss | 1.784479 | 0.993533 |
| Training all-six pass | 28/100 | 84/100 |
| Held-out all-six pass | 3/10 | 9/10 |
| Search wall time | 1285.68 s | 942.74 s |
| Total fitting wall time | 1448.95 s | 1006.93 s |

The ten-epoch run makes 250 accepted updates and visits each training stance
exactly ten times. All 110 final native rollouts complete. The selected winner
is update 250, using complete all-training mean loss. Setup including initial
scoring takes 33.70 s; final scoring takes 30.48 s. Search includes checkpoint
scoring and proposal checks. Independent qualification is timed separately.

The four-epoch checkpoint already reaches training loss 1.583973, with 70/100
native all-six passes, after approximately five search minutes. The longer run
measures further convergence; neither experiment establishes optimal fitting
quality or a pure optimizer speedup because controller capacity also changes.

Six-channel limits are 20 mm for each hip axis, 50 mrad for each joint, and
100 N for each GRF axis. Remaining training failures by channel are
0 hip-x, 2 hip-z, 1 joint-1, 6 joint-2, 0 GRF-x, and 8 GRF-z; some overlap.
The one held-out threshold failure is GRF-z. Do not equate a low mean objective
with all-channel acceptance on every stance.

Independent frozen-controller qualification completes all 110 native production
rollouts and matches the reported batch losses within 7.99e-15. All ten held-out
rollouts complete at half timestep, with 9/10 six-channel passes and mean loss
0.994006 versus native 0.993533. Training stances have no half-timestep check.
This separate verification takes 89.04 s; its result is
`physics_backprop_100_20260929/frozen_qualification.json`. Source snapshots and
hashes accompany the trained checkpoint under `train_10_epochs/sources/`.

Reports: `physics_backprop_20260929/single_stance_report.html` and
`physics_backprop_100_20260929/train_10_epochs/report.html`. PPO/RNN work remains
deferred. No unit tests were added or run.
