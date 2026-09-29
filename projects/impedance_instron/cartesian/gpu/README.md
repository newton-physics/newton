# Twelve-point CUDA pipeline

The [project command](../../README.md) is the supported end-to-end path.
It builds new evidence and runs one shared 12-point controller on 128 fixed
worlds. Proposals, residual Jacobians, damped Gauss–Newton solves, bounds,
incumbent selection and fit history stay on CUDA during search.
The CPU submits graphs and checks the soft wall budget. Plateau stopping copies
only one status integer per completed iteration; it does not copy fit arrays.

Leg dynamics and the objective use float64. The shared shoe remains float32.
Physics, failure screens, six measured-loss channels, and acceptance limits
are unchanged. No CPU optimizer, multi-island mode, knot-insertion
continuation, or count-comparison harness is retained.

For an already prepared and qualified run, the lower-level command is:

```bash
uv run --no-sync -m projects.impedance_instron.cartesian.gpu \
  outputs/impedance_instron/run12/baseline \
  --single-validation outputs/impedance_instron/run12/single/benchmark.json \
  --batch-validation outputs/impedance_instron/run12/mixed/benchmark.json \
  --contact-rounding-evidence outputs/impedance_instron/run12/contact/report.json \
  --output outputs/impedance_instron/run12/fit
```

Use a new fit directory. All input, source and runtime identities must match.
Source changes require new qualification; old source-hash exceptions have been
removed. The specific full-trajectory ankle-moment rounding warning can still
be qualified only by the unchanged common-pose contact limits. Other failed
numerical checks block fitting. This is separate from measured-fit acceptance.

Each completed fit writes a frozen half-step check and a spring-enabled HTML
report. Passive-cap activation stays visible even when numerical checks pass.
Use a separate `WARP_CACHE_PATH` when testing while another process compiles.

## Fixed-gain screens and fresh restarts

The gain-screen command creates nine independent cases. Common stiffness and
common damping multipliers are each `0.5`, `1.0`, or `2.0`. They apply to all
four actuated channels. Shoe geometry, material laws, timesteps, measured loss,
hard bounds, failure screens, and acceptance limits remain fixed. This is not
an exhaustive search over eight independent gain values.

Multipliers are relative to the chosen bundle. The example below deliberately
uses the preserved `baseline12` bundle to repeat the original gain grid.
The default pipeline, profilers, and audits now use `baseline12_maxwell`, whose
stiffness and damping are already twice that older nominal profile. Using the
accepted bundle for a new gain screen therefore defines a different gain grid.

```bash
uv run --no-sync -m projects.impedance_instron.cartesian.gpu.gain_sweep create \
  --bundle outputs/impedance_instron/baseline12 \
  --output outputs/impedance_instron/gain_screen --subject S001 --iterations 50

uv run --no-sync -m projects.impedance_instron.cartesian.gpu.gain_sweep run \
  outputs/impedance_instron/gain_screen --case k1.00_d1.00

uv run --no-sync -m projects.impedance_instron.cartesian.gpu.gain_sweep report \
  outputs/impedance_instron/gain_screen
```

Run each case in `plan.json` serially on the GPU. Existing output directories
are never overwritten. Each case builds a new profile and bounded, unfitted
kinematic-PD controller. It then runs CPU/GPU, common-pose contact, and mixed-world
qualification before fitting. Small gains can require a different bounded
failure fixture; the builder confirms an actual failure without weakening any
screen. Each fit retains the half-timestep check and interactive replay.

`report.html`, `results.md`, `results.csv`, and `results.json` compare loss, six
RMS errors, acceptance, refinement, completed iterations, search wall time,
allocated slots, completed real candidates, padding, and actual world-steps.
Rankings are separate for each iteration budget. A numerically qualified
winner can still fail measured acceptance. Fifty iterations are a screen,
not a convergence certificate.

For a finalist, compare the same longer budget across `reference`, `attenuated`,
and `perturbed` fresh starts and search seeds `17`, `42`, and `101`. For example:

```bash
uv run --no-sync -m projects.impedance_instron.cartesian.gpu.gain_sweep run \
  outputs/impedance_instron/gain_screen --case k1.00_d1.00 \
  --mode perturbed --initial-seed 2718 --seed 42 --iterations 200
```

The perturbed-controller seed is separate from the optimizer seed. Holding
`--initial-seed` fixed lets search seeds start from the same coefficients.
Attenuated starts contract deviations toward the neutral initial anchor.
Perturbed starts add bounded smooth changes after contraction. Neither starts
from a fitted controller or saved optimizer history. The selected gain values
remain constant throughout each solve.

A different subject requires that subject's measured reference, static geometry,
inertial parameters, and shoe placement. `--subject` cannot relabel S001 inputs
as S014. Frozen shoe artifacts remain engineering assumptions unless independently
identified for the measured subject. Finite restarts and neighborhood probes
can expose sensitivity to local basins; they cannot prove a global optimum.

### Prepare another subject's measured inputs

With sealed gait assets and the frozen baseline bundle available:

```bash
uv run --no-sync --with ezc3d==1.7.2 \
  -m projects.impedance_instron.cartesian.prepare_subject \
  --subject-root /path/to/subject \
  --baseline-bundle outputs/impedance_instron/baseline12 \
  --output outputs/impedance_instron/subject_inputs
```

Preparation uses that subject's static geometry and inertial source. It uses
3D heel-cluster Kabsch transport for the anatomical ankle and foot pitch.
Dynamic toe markers do not set pitch. Native force timing and the 20 Hz
reference filter remain explicit in the output metadata.

The selector requires raw heel rigidity, finite COP, isolated single-foot
support, side-assignment evidence, and a guarded constant tied-belt interval.
Filtered rigidity cannot override a failed raw-marker gate. If no window
passes, the command stops and writes `selection_diagnostics.json`; do not
start GPU fitting from rejected or filtered-only inputs.

Successful preparation writes a separate `input_quality` receipt with raw-QC
values and artifact hashes. This is not controller acceptance: prepared inputs
retain `accepted=false`. Gain-plan creation checks that receipt and the recorded
raw rigidity, support, side, and treadmill gates. It refuses marked or
uncertified subject bundles. Existing manifest-verified frozen baselines remain
supported.

The shoe's intrinsic artifact bytes remain unchanged. Static ankle height and
ankle-to-heel geometry set the new rigid registration. The transferred heel
registration and shared shoe remain engineering assumptions, not a new
subject-specific shoe identification. The saved treadmill offset also retains
its stated synchronization limitation.

### Probe a fitted controller locally

After a gain run completes, inspect bounded coordinate and smooth perturbations:

```bash
uv run --no-sync -m projects.impedance_instron.cartesian.gpu.local_probe \
  --run-directory outputs/impedance_instron/gain_screen/runs/k1.00_d1.00_reference_s17_n50 \
  --output outputs/impedance_instron/gain_screen/local_probe
```

Use a new output directory. The command rechecks frozen inputs and numerical
qualification, then requires exact fitted-loss rescoring within `1e-12`.
It generates 192 coordinate and 128 smooth perturbations at 1% and 2% of the
profile box spans, plus the unchanged controller. Smooth directions and full
proposals are generated on GPU. Canonical bounds are screened on GPU; rejected
and padding slots do not integrate. Gains, contact laws, and the objective stay
fixed. `--seed` and `--random-direction-count` control the smooth directions.

`probes.json` records every proposal's bounds, completion, loss, and six RMS
errors. `summary.json` separates completed slots, rejected proposals, padding,
and actual integration work, including the separate baseline rescore and setup
warmup. Unique controllers are not counted. `equilibrium.npz` contains the best
completed bounded controller with its original warm-start identity.

A probe winner has not undergone its own half-timestep check or spring replay.
Do not treat it as an accepted fit. To qualify a saved probe without another
optimization:

```bash
uv run --no-sync -m projects.impedance_instron.cartesian.gpu.replay_probe \
  --probe-directory outputs/impedance_instron/gain_screen/local_probe \
  --output outputs/impedance_instron/gain_screen/qualified_probe
```

By default, this selects the lowest-loss probe among those meeting **all six
native-step RMS limits**. It can differ from the lowest-loss probe overall.
`--probe-index` selects a specific completed bounded probe instead. The command
reconstructs that controller on GPU, checks canonical bounds and exact saved-loss
agreement, and runs the unchanged half-timestep and spring-contact replay checks.
`summary.json` reports acceptance. `report.html` provides the verified replay.
Neither the optimizer's objective nor its selection policy is changed.

Nearby improvements and finite unsuccessful probes are diagnostics, not local-
or global-optimality certificates. Numerical acceptance is not physical or
physiological validation of the inherited shoe/last interface.

## Profile complete search

Measure full poll-plus-trial iterations from the saved controller:

```bash
uv run --no-sync -m projects.impedance_instron.cartesian.gpu.profile_search \
  --output outputs/impedance_instron/search_profile \
  --iterations 10 --repeats 3
```

This timing-only command checks the saved bundle's file hashes. It keeps the
128 worlds, 48 coefficients, physical timestep, objective, proposal seed, and
both search phases. It warms graph replay before timing and starts every repeat
from the same saved controller. Plateau stopping is disabled for fixed-work
comparisons. Use a fresh output directory.

`profile.json` records complete iteration wall time, completed real candidates
per second, candidate/padding counts, source and runtime identities, and losses.
`search_*.json` retain each repeat's history. Setup, compilation, and final
unloading remain separate from search timing. Completed candidate counts include
baseline reevaluations, not unique controllers. Compare equal iteration budgets
and loss histories, not GPU utilization alone.

The command deliberately produces **no numerical qualification or accepted fit**.
Historical hashes identify the input bundle; they do not certify changed source.
Use the [complete pipeline](../../README.md#run-the-complete-pipeline) for fresh
CPU/GPU qualification, refinement, and an interactive winner replay.

### Fused rollout kernels

For beds of up to 1,024 columns, the CUDA engine keeps the shoe update within
one block per independent world. Two columns per thread fit the 910-column
baseline into a 512-thread block. Each Jacobi sweep still reads the previous
sweep, including the previous driven-column values. The shared balance,
float32 FMA policy, and odd/even scratch-buffer results are unchanged.

The block also runs the shared pressure, friction, and ordered wrench reductions.
Exact zero-compression constitutive values are cached on device and refreshed
on reset or material changes. Compression diagnostics retain float64 divisions,
nonfinite checks, and integer cap counts. CPU execution and unsupported layouts
retain the shared fallback.

The float64 leg kernel advances the state and stages the next carrier. The
resident baseline loop therefore uses two kernels per timestep, plus reset,
initial staging, chunk-loop control, and objective work. The leg and shoe retain
their separate FMA policies. This scheduling follows the block-local approach
used by Newton's Kamino kernels; it does not replace either physical solver.

Bound-rejected padding slots are no longer integrated. Proposal generation,
random state, Jacobians, selection, and hard bounds are unchanged. The legacy
`physics_worlds` counter counts allocated launch slots; completed-world and
integrated-step counters report actual work. No unique-controller count is claimed.

### Measured same-results comparison

On the RTX A6000, three five-iteration repeats from the saved controller gave:

| Metric | Previous cooperative path | Fused path with padding skip and cache |
| --- | ---: | ---: |
| Median complete iteration | 2.757 s | 1.387 s |
| Completed real candidates / search second | 36.89 | 73.08 |
| Final loss after five iterations | 2.1830351003 | 2.1830351003 |
| Completed real candidates per repeat | 491 | 491 |
| Integrated world-steps per repeat | 7,372,800 | 2,828,160 |

The median time fell 49.7%. Every retained coefficient, tracking metric, and
batch loss matched exactly. Before/after runs were sequential, not interleaved.
The 789 padding slots per repeat remain allocated but do no integration work.
**Sub-second iterations were not reached.** Larger blocks, smaller blocks, and
a lower-matrix-only experiment were slower and were not retained.

One instrumented five-iteration search attributed about 98% of its wall time
to the rollout graphs. Optimizer and snapshot intervals totaled about 26 ms per
iteration, including about 22 ms for the serial damped solves. Small eager-kernel
intervals include host enqueue gaps; these are not precision microbenchmarks.

Evidence is under `outputs/impedance_instron/speed_round2/` when the local
benchmark bundle is available. The source-only checkout does not include these
generated files. Historical first-pass timings remain under `gpu_search_speed/`.

### Longer fitted run

Fresh CPU/GPU and mixed-world qualification passed for the retained code. A
200-iteration fit completed in 286.1 search seconds (1.43 s per iteration),
evaluating 25,473 real candidate slots. Loss fell from 2.18564 to 2.05824.
Both force RMS errors fell below 100 N, and the frozen half-timestep check passed.
The result is **not accepted**: vertical hip RMS error remains 20.68 mm against
the unchanged 20 mm limit. This is a longer run of the existing optimizer, not
an autodiff-fitting result or a changed acceptance criterion.

The controller, traces, qualification evidence, and verified spring replay are
under `outputs/impedance_instron/speed_round2_qualified/`. Open
`fit/report.html` to inspect the saved result.

The default contact law is area-scaled elastic Coulomb friction (`G_eq A / L`
stiffness per column, no tangential damping). Maxwell and material-derived
`column_maxwell` remain explicit options. Legacy fused friction remains an
explicit compatibility mode. The default uses the shared foundation launch
path, including retained normal-surround optimization.
