# Twelve-point controller baseline

This worktree has one controller pipeline: one leg, one shoe, one stance,
with **12 cubic control points per equilibrium channel** (48 coefficients).
One shared controller uses a fixed batch of **128 CUDA worlds**.
There is no trunk, opposite leg, hip-angle motor, or added upper-body load.

Small reference and rollout bundles for the recovered FR3_1 rate refit and the
FR3_2 peak-to-peak window experiment are in [sample data](data/samples/README.md).
The FR3_2 fit is exploratory and did not meet the measured-fit tolerances.

## Import processed measurements

Raw C3D and processed Visual3D exports use separate preparation paths. To audit
the new F01 exports without guessing timing or physical metadata:

```bash
uv run --no-sync -m projects.impedance_instron visual3d inspect data/F01
```

See [Visual3D inputs](VISUAL3D_INPUTS.md) for the export scripts, manifest,
normalization command, and Cartesian preparation requirements. All three F01
trials now contain the required clocks, static measurements, joint centers, and
force channels. Subject-specific inertias remain provisional. A baseline
comparison also requires the baseline's exact shoe artifact and fixed
foot-to-shoe registration.

Subject-profile preparation now scales sagittal thigh and shank inertia by
the square of the measured-to-model segment-length ratio. Foot and toes are
combined about their shared COM, then scaled by the square of the endpoint
ratio. Population radii of gyration from de Leva (1996) are recorded as a
comparison; they do not replace subject-model segment masses or inertias.
Source inertial-frame rotations remain recorded in provenance and need to be
checked against the source model's frame convention before treating the
inertias as validated.

The GPU objective computes measured hip-velocity RMSE, maximum hip speed, peak
hip spring and damping loads, and a Coulomb-equivalent force ratio for every
candidate; the final selected candidate's diagnostics are saved in its fit
summary. These are diagnostics only and do not enter the fitted loss. The ratio is
`abs(GRF_x)/(mu*GRF_z)` where normal force exceeds 5 N; it is a proximity proxy,
not the internal saturation state of the selected viscoelastic shoe model.

## Selected baseline

`outputs/impedance_instron/baseline12_maxwell/` is the saved baseline. Its shoe
uses the friction model recorded in its manifest (`maxwell`). New Cartesian
shoe and GPU Engine instances default to `elastic_coulomb`, with per-column
stiffness `G_eq A / L`, `mu = 0.8`, and no tangential damping. `maxwell` and
`column_maxwell` remain available explicitly. Normal material/contact mechanics
are unchanged.

The gains remain those of the previous K2/D2 controller:

- Hip stiffness [8000, 12000] N/m; joint stiffness [240, 180] N m/rad.
- Hip damping [80, 80] N s/m; joint damping [12, 8] N m s/rad.
- All twelve hip-Z equilibrium coefficients are shifted by +1.5 mm to restore
  tracking margin under the saved Maxwell law. No other coefficients, gains, masses,
  initial conditions or normal parameters are changed.

The previous accepted legacy-friction baseline is archived in
[`baselines/baseline12_legacy_accepted.json`](baselines/baseline12_legacy_accepted.json).
The initial twelve-point research case remains in
[`baselines/baseline12_initial.json`](baselines/baseline12_initial.json).
[`baseline.json`](baseline.json) records the current model, numerical qualification
and bundle hashes. Numerical acceptance is not independent outsole calibration.

- Current measured RMS: hip **7.450 / 19.746 mm**, knee/ankle **0.023742 / 0.030700 rad**,
  force **91.296 / 85.947 N**; all six original limits pass.
- Half-step maximum force difference: **4.320 N**; original refinement limits pass.
- CPU/GPU parity and verified spring-contact replay pass.
- The previous legacy baseline remains archived rather than silently relabelled.

Choose `friction_model="legacy"` explicitly in `FoundationConfig`, `Shoe`, or
`Engine` to reproduce the previous contact behavior.

## Run the complete pipeline

Run from the repository root with its existing `uv` environment and CUDA.
Choose a new output directory. The local baseline bundle is required; restricted
motion and shoe data are not supplied by a source-only checkout.

```bash
uv run --no-sync -m projects.impedance_instron \
  --output outputs/impedance_instron/run12
```

This command:

1. Checks bundle hashes and replays the selected controller on CPU.
2. Runs same-step CPU/GPU, common-pose contact, and 128-world
   permutation/reset/isolation checks, including a failed candidate.
3. Fits one shared controller on GPU with the original six-channel measured loss.
4. Runs a frozen half-timestep check and writes `fit/report.html`, including
   verified spring and deformation views.

Defaults are 200 iterations, a 3,600-second soft search cap, seed 17,
plateau patience 20, and relative plateau improvement 0.0001. A plateau is not
proof of convergence. Setup, numerical validation, refinement, and reporting
are outside the search cap. `--iterations 1` is a short end-to-end smoke run.
The timestep stays 62.5 microseconds, with 31.25 microseconds for refinement.

### Fit a new controller from scratch

The default command above is a warm start. To generate new, unfitted controller
coefficients instead, use:

```bash
uv run --no-sync -m projects.impedance_instron --from-scratch   --output outputs/impedance_instron/fresh12 --iterations 200 --wall-seconds 3600
```

This mode does not use a saved controller or previous optimizer history.
It samples `q_reference + (D/K) * velocity_reference` at twelve cubic-spline
Greville abscissae, then contracts the channels toward their initial neutral
points until the original strict control-polygon bounds hold. No simulation
loss or measured GRF is used to choose the seed. This is deterministic,
measurement-based initialization, not random coefficients or prescribed motion.
The recorded data, calibrated shoe, physical model, gains, and limits stay fixed.

The prepared baseline and fit summary record the initialization formula,
contraction factors, starting coefficients, and `used_previous_controller_coefficients: false`.
Use `--from-scratch` with separate stages to require matching fresh provenance.

Stages can also run separately:

```bash
uv run --no-sync -m projects.impedance_instron --output outputs/impedance_instron/run12 --stage prepare
uv run --no-sync -m projects.impedance_instron --output outputs/impedance_instron/run12 --stage validate
uv run --no-sync -m projects.impedance_instron --output outputs/impedance_instron/run12 --stage fit
uv run --no-sync -m projects.impedance_instron --output outputs/impedance_instron/run12 --stage report
```

`--baseline DIRECTORY` selects a complete saved bundle with the same manifest
format. Existing stage outputs are not overwritten, except an explicit report
rebuild. If qualification fails, inspect the failed flags; do not enlarge the
limits or substitute evidence from another input or source version.

## Differentiable search framework

See [the reverse-mode search framework](AUTODIFF_SEARCH.md) for full-horizon
backprop, memory/checkpoint policy, exact spline constraints, and validation.
The shared mass-solve adjoint, tape-safe full-contact rollout, measured objective,
and runnable gradient audits are implemented as experimental diagnostics. Short
coupled-window checks pass, but the full-stance gradient audit remains unqualified.
There is no integrated adjoint optimizer; the current forward search stays the default.

## Search performance

Use the [complete-search profiler](cartesian/gpu/README.md#profile-complete-search)
to compare equal-work GPU searches without rebuilding an HTML report on every
repeat. It reports full iteration time and useful candidate throughput, not
kernel enqueue time. Profiling does not replace numerical qualification.

## What remains

- `pipeline.py`: the single entry point and stage order.
- `cartesian/`: reference validation, spline algebra, fixed physical model,
  measured objective, CPU reference rollout, shoe attachment, and replay.
- `cartesian/gpu/`: GPU dynamics/objective, shared resident search, numerical
  qualification, and GPU spring export.
- Newton and `projects/digital_shoe/`: shared framework and material/contact laws.

Retired bilateral, paper, two-stiffness and learned-controller rigs, old
preparation chains, serial optimizers, multi-island search, control-count
comparisons, compatibility aliases, and their tests/reports are removed.
The default pipeline starts with the frozen filtered measured reference.
Subject-specific C3D preparation is a separate, fail-closed workflow described
in the GPU README; it does not replace frozen inputs during a controller fit.

Apart from the explicitly selected 2× stiffness and damping, the inherited
foundation interface, material, friction, masses, bounds, initial physical state,
loss, and acceptance limits are unchanged. Recorded
motion after the initial state and measured GRF are targets, never applied
motion or extra forces. The reference retains its original 20 Hz filtering
metadata. This cleanup does not certify biological validity or a new shoe
interface. General Newton APIs and the separate calibration/gait projects are
not retired by this controller cleanup.

## Historical cleanup verification

The controller project shrank from **97 Python files / 43,084 source lines** to
**28 files / 7,094 lines**: a net deletion of **35,990 lines (83.5%)**.
These counts exclude tests, generated outputs, and compiler caches.
Another 48 obsolete test modules were removed. Newton public source is unchanged.

The retained pipeline passed 90 targeted tests on CPU/CUDA, all pre-commit
checks, and a fresh one-iteration end-to-end CUDA smoke run. Same-step parity,
128-world isolation/reset/permutation, frozen half-step refinement, and spring
replay passed. Spring-history errors were zero. The smoke run is verification,
not a replacement for the selected 200-iteration baseline or a convergence claim.
Its evidence is in `outputs/impedance_instron/cleanup_validation/verification.json`
and its replay is `outputs/impedance_instron/cleanup_validation/fit/report.html`.
That earlier baseline and cleanup smoke result were outside measured-fit acceptance.
They are historical evidence, not the newly selected accepted baseline above.
