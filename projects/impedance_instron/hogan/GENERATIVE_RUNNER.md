# Generative runner with variable impedance

The research target is a shared model of how the runner produces motion and
responds to contact, not a measured-trajectory tracker. The first implementation
is now in `runner.py`, `identify.py`, and `generate.py`. The old `learn.py`,
`plan.py`, `batch.py`, and `rollout.py` remain tracking baselines; their commands
and saved schedules are not silently reinterpreted as generative models.

## Current baseline (2026-10-08)

The baseline model is
[`baselines/generative_runner_f01_20261008.json`](baselines/generative_runner_f01_20261008.json).
It is one shared runner fitted to subject F01: 66 kg, treadmill belt 3.65 m/s,
wearing the Puma Fast-R Nitro Elite 3 digital shoe. The baseline learning method is:

1. **Data.** `prepare_dataset` builds the window dataset with the corrected
   subject mass and belt speed (`data/F01/FR3_1/stance_selection.json`). Each
   member records `flight_velocity_m_s` (see *Offline identification*). Only
   the ankle FK check gates; loaded clearance and COP are reported. 107 of the
   110 windows remain after dropping three where the shoe starts below ground.
2. **Initialization.** Positions come from the three-frame flight prefix. The
   hip velocity is then chosen so the model's whole-body COM moves at the flight
   velocity integrated from the preceding stride's plate force.
3. **Fit.** Levenberg–Marquardt from `Runner.seed()`, with 119 parameters, a
   forward-difference Jacobian, and batched GPU rollouts. The fitted residuals
   are the hip, angle, and GRF sample terms of `identify.score`. Peak, impulse,
   contact, and effort terms are scored but not fitted. Selection uses training
   stances only.
4. **Report.** `fit_report` shows seed-versus-fitted metrics, stick figures of
   the best and median held-out stances, the learned impedance, and LM
   convergence.

```console
uv run --no-sync -m projects.impedance_instron.hogan.identify fit --method lm --iterations 15 --chunk 16 --compression-limit 0.99 --dataset outputs/impedance_instron/generative_fit_dataset_flight_20261008 --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.65 --output outputs/impedance_instron/generative_fit_lm_flightcom_20261008
uv run --no-sync -m projects.impedance_instron.hogan.fit_report --run outputs/impedance_instron/generative_fit_lm_flightcom_20261008
```

The fit took 82 min on an RTX A4000 Laptop GPU (15 iterations, 195,608
candidate-stance rollouts). Training cost fell from 42.2 to 10.03. Free
predictions, means over stances:

| Split | Model | Loss | Hip RMSE fwd / up [mm] | Angle RMSE [deg] | Fx / Fz RMSE [N] | \|Peak Fz error\| [N] | Contact error [ms] |
|---|---|---:|---:|---:|---:|---:|---:|
| Train (98) | seed | 120.8 | 84 / 51 | 12.0 | 211 / 349 | 775 | −1 |
| Train (98) | fitted | 16.1 | 47 / 19 | 6.2 | 108 / 128 | 58 | −39 |
| Held-out (9) | seed | 125.4 | 79 / 52 | 11.8 | 214 / 359 | 866 | −2 |
| Held-out (9) | fitted | **14.8** | **42 / 18** | **6.5** | **105 / 119** | **26** | −42 |

What changed from earlier fits, and why:

- **Mass and speed.** 70 kg and 3.7 m/s were initial guesses. With 70 kg the
  measured step impulse was only 0.80 of m·g·T.
- **COM-matched flight start.** Measured force reproduces the hip path to
  about 5 mm only when the start velocity is right. The old three-frame hip
  velocity was off by about −0.36 m/s vertically, which set a hip-cost floor of
  14.3 even with perfect force. The swinging leg alone shifts the model COM
  about 0.3 m/s forward of the hip, so the estimate is matched to the model COM,
  not the hip.

On a one-stance overfit (FR3_1_train_013 with held-out FR3_1_eval_000) the
COM-matched start reached loss 8.9 / 13.1 (train / held-out), against 17.0 / 21.6
with the three-frame start. Applying the estimate to the hip instead gave
18.6 / 26.5.

Known gaps of this baseline:

- **Early toe-off.** Contact ends about 40 ms early on average. With a late
  touchdown, this explains most of the 5–10 Hz force-shape mismatch.
- **Small force wiggle.** About 20–27 N RMS at 10–20 Hz, source unidentified.
  Ringing above 20 Hz partly reflects the 20 Hz filter on the exported force.
  The optional lag-free `intrinsic_damping_nms_rad` (default 0;
  `identify fit --intrinsic-damping`) halves it but does not improve the loss.
- **Impedance decomposition not identified.** Two fits on disjoint 10-stance
  subsets agree on motion (3–4 mm, 1–2°) and torque (within 10–17%). Their K,
  D, and q<sub>eq</sub> differ 2–8× the stance-to-stance spread. Treat the
  learned impedance as one equivalent decomposition, not a measurement.

## Runtime boundary

`runner.simulate` accepts only a frozen `Runner`, a physical `Chain`, a `Shoe`,
initial `State`, known `Task`, horizon, and numerical configuration. It cannot
accept a measured trajectory, GRF target, COP target, reference clock, or
inverse-dynamics plan.

The state has six mechanical coordinates and velocities, an internal oscillator
phase, filtered simulated normal load, and three activation-filtered joint
torques. Fourteen features (schema `generative_runner_2`) drive a small shared
model:

1. Constant bias.
2. Sine and cosine of the first three phase harmonics (six features).
3. Filtered simulated vertical load in body weights.
4. Load multiplied by sine and cosine of phase, so the load response can
   differ between stance and swing.
5. Hip-over-ankle horizontal offset normalized by leg length.
6. Pelvis lean.
7. Intended minus simulated forward speed.
8. Intended speed relative to the training speed origin.

Schema-1 models (eight features) load exactly with zero weights on the added
features. A bounded
logistic map produces equilibrium angle, stiffness, and damping for hip, knee,
and ankle. The joint command is:

$$
\tau_{\mathrm{command}} = K(q,\dot q,z,u)(q_{\mathrm{eq}}(q,\dot q,z,u)-q_{\mathrm{joint}})
                         -D(q,\dot q,z,u)\dot q_{\mathrm{joint}}.
$$

Torque magnitude and slew are bounded, with a first-order response time.
An optional intrinsic damping $-c\,\dot q_{\mathrm{joint}}$ bypasses that lag;
it is zero unless set. These are declared engineering assumptions, not measured
physiological limits.
The equilibrium is generated internally; it is not sampled from each stance.
There is no desired-velocity term or measured feedforward. Moving equilibrium
and varying stiffness can supply work: this is an active model, not a claim of
passivity.

Only relative joint coordinates 3–5 receive actuation. The floating base's
horizontal force, vertical force, and absolute pelvis torque are exactly zero.
Joint reaction effects on the pelvis arise through the coupled mass matrix.

Phase advances continuously with autonomous frequency and bounded modulation
from simulated load. It never snaps to measured or simulated touchdown times;
the displayed angle wraps modulo a full turn, but sine/cosine remain continuous.
No reference-time retiming or derivative transformation is needed.

## Offline identification

`identify.py` owns measurements and computes errors **after** free prediction.
It does not call `build_plan`, perform inverse dynamics, integrate GRF into a
replacement COM trajectory, or re-solve IK to reduce prediction errors.

Target coordinates re-solve hip and knee per frame so fixed-length FK reaches
the measured hip and ankle centers, keeping the measured foot angle. The
exported knee angle alone placed the F01 ankle 26–49 mm from its measured center.
This uses positions only, never force, and is applied before initialization.

Initialization uses the first three measured position frames. Prediction
begins at the **last** of these frames, so the backward quadratic velocity
estimate does not use future frames. The hip marker is not ballistic in
flight, so when a member provides `flight_velocity_m_s` the hip velocity is
replaced. `prepare_dataset` computes that value by integrating the treadmill
plate force over the preceding full stride. The integration constant is the
hip's mean belt-frame velocity over that stride, so no data after the
prediction origin are used. The hip velocity is then set so the model COM moves
at this velocity with the prefix joint rates. Members without a preceding
stride keep the three-frame estimate. Initial oscillator phase uses a fixed
hip-angle/rate convention. Load memory and torque start at zero, and the
compatibility gate requires geometric flight at the prediction origin for the
relaxed-shoe assumption. Upstream Visual3D
processing may already be noncausal; this boundary guarantees causality relative
to the supplied prepared positions, not the original acquisition pipeline.

One parameter set is shared across training windows. It learns equilibrium,
stiffness and damping feature weights, stride frequency, and torque response
time. There are **119 active parameters at one speed**, or **129 with multiple
training speeds**. Direct task-speed-offset weights and cadence-speed slope stay
fixed when training has only one speed. State-dependent speed-error feedback
can still be fitted; its extrapolation is not certified.

The default optimizer is Levenberg–Marquardt (`--method lm`, `least_squares.py`).
The cross-entropy search (`--method cem`) remains available. It ranks failure
count before fitting error and keeps the best training candidate, including the
original seed. Evaluation targets
never choose parameters. The objective includes motion/angle error, force error,
peak vertical force, impulse, contact duration, and a small normalized torque
effort term. These are identification errors, not online tracking feedback.

Before fitting, input checks flag separated shoe bottoms under measured load,
measured-ankle versus fixed-length FK disagreement, and shoe penetration at the
prediction origin. COP outside the projected sole is reported but not gated,
because treadmill COP is unreliable at low force. Default fitting refuses
incompatible inputs. The explicit
`--allow-incompatible` option is for implementation experiments only, and the
result is always labeled `validated: false`.

## Commands

Run from the repository root. Choose a new output directory for every command.
Start with inspection of a small current-data subset:

```console
uv run --no-sync -m projects.impedance_instron.hogan.identify inspect --dataset outputs/impedance_instron/hogan_stance_dataset --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.7 --limit-per-split 1 --output outputs/impedance_instron/generative_inspect
```

On compatible data, replace `inspect` with `fit`. LM flags are `--iterations`,
`--chunk`, and `--central`; CEM (`--method cem`) takes
`--population 8 --generations 10 --seed 0`. Commands now default to `--device cuda:0`;
use `--device cpu` for the reference backend. Start small. `--limit-per-split` is
explicit subsetting, not a new random split; remove it to use all members.
No long fit on the currently incompatible dataset is recommended.

### GPU execution and limits

`gpu_runner.GpuBatch` advances candidate/trial worlds on CUDA using captured
chunks. Joint actuation, full chain dynamics, contact, phase, and sensory state
remain on device throughout each rollout. Trials group by shoe instance and
their **exact adjusted timestep**, avoiding the previous first-trial-dt error.
Equivalent but distinct shoe objects do not automatically share a group.

`gpu_objective.GpuEvaluator` keeps fitting targets separate from dynamics and
computes candidate objective reductions on CUDA. In the resident path only
16 bytes per candidate are downloaded per evaluation. A 512 MiB conservative
live-array admission budget triggers bounded trial streaming for larger sets;
streaming rebuilds groups/captures and downloads small candidate totals per
chunk. This budget excludes CUDA context, allocator reservations, and graphs.
Large single-trial allocations are rejected rather than silently overcommitted.

The expensive simulation and candidate scoring run on GPU. File preparation,
small CEM parameter sampling/ranking, detailed final report metrics, and report
serialization remain on CPU. There is no claim that every operation runs on
device. The library's `runner.simulate` and default `identify.fit` Python API
remain CPU reference paths; pass `device="cuda:0"` to the fitting API. CLI
identification, evaluation, standalone generation, and quick comparison default
to CUDA. Custom friction adapters are not supported by the GPU runner.

Both ordinary fitting and the quick comparison enforce compatibility before
creating GPU training work. The latter no longer implicitly overrides the gate;
diagnostic bypass requires `--allow-incompatible` explicitly.

Fit output includes:

- `runner.json`: frozen portable variable-impedance model.
- `summary.json`: split metrics, configuration, seed model, compatibility,
  initialization policy, and source/reference/profile/shoe fingerprints.
- `trace_NNN.npz`: prediction used for inspection, including actual K, D,
  equilibrium, torque, force, and internal state histories.
- `scenario_NNN.json`: physical model, shoe, initial state, task, horizon, and
  integration settings **without measured trajectories or forces**.

Generate independently from an exported scenario:

```console
uv run --no-sync -m projects.impedance_instron.hogan.generate --model runs/fit/runner.json --scenario runs/fit/scenario_000.json --output runs/generated
```

The generator needs only the model, scenario, and shoe artifact—not the original
measurement dataset. Scenario shoe hashes are verified. Exported artifact paths
are absolute; when moving a bundle, update the path to the copied artifact
(relative to the scenario is supported), retaining the verified hash.

For offline evaluation against a new measurement dataset, use
`identify evaluate --model runs/fit/runner.json` with the dataset and placement
arguments. This does not refit the runner. To compare a changed shoe mechanically,
keep the initial runner state and all model parameters fixed, change the scenario
shoe artifact/hash and its independently measured placement, and regenerate.

## Future speed/footwear datasets

The loader accepts the existing `peak_hip_stance_dataset_1` or the explicit
`generative_runner_dataset_1` manifest. Both contain `members`. Each member has:

| Field | Meaning |
|---|---|
| `id`, `split` | Unique identifier and `train` or `eval` |
| `reference` | Prepared measurement NPZ path |
| `profile` | Physical leg profile path |
| `shoe_artifact` | Calibrated shoe JSON path |
| `mount_m`, `pitch_rad` | Independently registered ankle/shoe placement |
| `speed_m_s` | Known task speed, not future measured mean speed |
| `height_offset_m` | Optional fixed vertical registration, default zero |
| `subject_id` | Optional subject identifier; all supplied IDs must agree |

Paths are relative to the dataset directory. Existing `shared_assets` can supply
the profile and shoe instead; declared shared hashes are verified. Missing speed
or mount requires explicit CLI arguments. Optional top-level `rest_of_body`
contains `RestOfBody` parameters; it is fixed across members and recorded.
Use one physically consistent runner profile per subject, not fitted
condition-dependent inertias. Unknown subject identity in a legacy manifest is
not independent proof that all records belong to one subject.

Hold out entire conditions/sessions when testing transfer. The loader preserves
the provided split; it does not infer that windows from the same recording are
independent. Multiple shoes currently affect the runner through simulated contact
and state, not shoe-ID-specific coefficients. This models an immediate mechanical
response. Long-term adaptation, anticipation, and training-dependent changes
need separate data and model extensions.

## Verified implementation checks

`newton.tests.test_impedance_runner` checks joint-only bounded actuation,
COM acceleration and angular momentum under internal torque, causal prefix
initialization, repeat/reset behavior, phase continuity, model serialization,
future-target independence, train-only selection, single-speed parameter
freezing, synthetic free-trajectory fitting, compatibility rejection, and CLI
generation after deleting the original measurement dataset.

A one-generation current-data smoke run used one train and one held-out stance,
with an explicit incompatibility override. Both full windows completed with
zero floating-base actuation. The seed remained the best candidate; vertical
force errors were approximately 277 N training and 310 N held-out. This verifies
execution, not fit quality, physiological identification, or improved realism.
Artifacts: `outputs/impedance_instron/generative_runner_smoke_20261007`.

## Synthetic parameter recovery

`recovery.py` asks whether the shared parameters can be identified at all,
independent of the blocked measured data. A known truth (seed plus random
offsets) generates free rollouts from real trial chains, shoes, initial states,
and clocks. Gaussian noise is added (defaults 2 mm hip, 0.01 rad angles, 20 N
GRF), and `identify.fit` starts from the seed. Measured motion and force are
never read. The report gives a search verdict (`recovered`, `not_identifiable`,
or `inconclusive_search` when the fit does not reach the truth's loss) and a
finite-difference Fisher analysis at the truth with Cramér-Rao bounds:

```console
uv run --no-sync -m projects.impedance_instron.hogan.recovery --dataset outputs/impedance_instron/hogan_stance_dataset --mount -0.03186147427106201 0 0.10943209684347802 --speed 3.7 --limit-per-split 1 --output runs/recovery
```

Random truth offsets above about 0.1 per parameter usually bottom out the shoe
(compression screen), so use `--truth-scale 0.1` with more `--truth-attempts`
for multi-trial runs.

First results (2026-10-08, same truth and noise for both 8+8 rows):

| Stances (train+eval), optimizer | Local rank at 1e-3 | CR std < 0.1 | CR std > 1 | Train loss truth / learned | Held-out loss seed / learned / truth | Impedance error seed / learned | Fit wall time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1+1, truth scale 0.3, CEM 64×40 | 31/118 | 4 | 73 | 0.95 / 64 | 654 / 1881 (failed) / 0.10 | 0.116 / 0.132 | 45 s |
| 8+8, truth scale 0.1, CEM 64×60 | 69/119 | 53 | 0 | 0.13 / 2.57 | 28.8 / 29.1 / 0.16 | 0.050 / 0.045 | 458 s |
| 8+8, truth scale 0.1, LM 15 iterations | (same) | | | 0.126 / 0.130 | 28.8 / 0.160 / 0.160 | 0.050 / **0.0097** | 255 s |

With eight training stances every parameter is locally determined to better
than the search box. CEM does not reach the truth's loss; Levenberg-Marquardt
(`least_squares.py`, default `--method lm`) reaches it by iteration 4 (about
60 s), matches the truth on held-out stances, and recovers impedance to about
1% of its range. Individual weights are only partly recovered (offset
correlation 0.65), consistent with the 50 weak directions: the *function* is
identified, not every coefficient.

The LM fitter batches the forward-difference Jacobian (one rollout per
parameter) and a five-value damping ladder as persistent GPU batches; the
119×119 damped solve runs on the host. `newton.ik.IKOptimizerLM` is not
applicable: it optimizes articulation joint coordinates against FK
objectives and tiles the full Jacobian per batch row in shared memory.
Artifacts: `outputs/impedance_instron/generative_recovery_20261008_{l1,l8,l8_lm}`.

## Remaining limitations

- Existing foot/shoe/reference incompatibility remains; see `REALISM_AUDIT.md`.
- The mechanical model still has one leg and a lumped rest-of-body. There are
  no independent trunk, contralateral-leg, or forefoot/MTP coordinates.
- Each call resets shoe history. Independent windows are supported, not
  restartable multi-step bilateral running.
- Equilibrium and impedance are effective actuation parameters; their unique
  physiological identification is not established by fitting trajectories.
- There are no explicit muscles, tendons, reflex transport delays, or metabolic
  cost. The first-order torque response is only an effective bandwidth model.
- Body parameters and material/contact laws stay fixed during identification.
- Gradient-based identification, parameter uncertainty estimates, and broad
  numerical/physical qualification remain future work.

## Contact correction status and GPU verification

See `CONTACT_INPUT_STATUS.md`: a diagnostic endpoint-frame bug is corrected in
reference preparation, but it does not move the ankle or contact columns.
Loaded clearance and COP conflicts are now reported, not gated. The gate is the
ankle FK check plus flight at the prediction origin, which 107 of 110 windows
pass. No source measurements, shoe geometry, material parameters, or fitted
offsets were altered to conceal the remaining contact conflicts.

The combined endpoint, CPU, CUDA rollout, CUDA objective, and integrated workflow
suites pass 80 tests. Two real full-window CPU/CUDA checks gave maximum state
differences below 2.1e-7 and maximum force difference about 0.0101 N. A warm
8-candidate x 2-stance GPU objective evaluation took 0.93 seconds on the RTX A4000
Laptop GPU; two serial CPU stance rollouts took 9.20 seconds. These are unequal
world counts, not a direct same-work speedup measurement. Setup/first evaluation
for the GPU objective took 1.12 seconds in that already-cached environment.

This is the generative architecture in place ahead of collection, not a claim
that the current seed or a short fit is already a realistic human runner.