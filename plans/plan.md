# Digital Shoe Property Identification Plan

## End Goal

Use intact bench-tested shoes to identify effective shoe properties, then attach those calibrated shoe models to a digital human. Later, train one shared controller across shoes and evaluate held-out shoes by GRF, COP, and joint work.

This document scopes the immediate work to Phase 1 and Phase 2. Humanoid running and shoe-ranking remain the downstream reason for the design, not part of the first build.

## Phase 1: Effective Shoe Property ID

### Goal

Identify one shoe-level effective contact/material parameter set from intact Digital Instron tests.

### Scope

- One shoe at a time.
- Use the current rearfoot punch and fullfoot last tests.
- Serial workflow is acceptable.
- Do not use humanoid or running data.
- Do not fit separate rearfoot and fullfoot material parameters.

### Data Split

- Build cycle-window averages from raw CSVs.
- Train on cycles `90-98`.
- Validate on held-out cycles `99-100`.
- Apply the split separately for rearfoot and fullfoot.
- Fit one shared shoe parameter set across both fixtures.

### Cycle-Window Generation

The raw Instron CSV is the source of truth for train/validation splits. The existing averaged-cycle CSVs are useful diagnostics, but they are not sufficient for the official held-out gate.

For each trial:

1. Load the raw CSV using the saved frame configuration from QC.
2. Infer the cycle column from the raw CSV. Prefer `Total Cycle Count`-style columns; fall back to `Total Cycles`/`Elapsed Cycles` only if the preferred column is absent and the chosen column produces complete repeated cycles.
3. Build separate averaged traces for cycles `90-98` and `99-100`.
4. Normalize each averaged trace to a common cycle phase grid before averaging so rearfoot/fullfoot traces can be compared frame-by-frame.
5. Persist the generated traces with explicit provenance:
   - source CSV path
   - frame config path
   - cycle column
   - cycle window
   - phase grid size
   - force/displacement zero policy

The generated trace schema should match the existing averaged-cycle files closely enough that the current fitting path can consume it:

```text
phase,time_s,displacement_m,displacement_mm,force_n,position_mm_raw,force_n_raw,velocity_m_s,cycle_energy_j
```

Output naming should make the split unambiguous, for example:

```text
rearfoot_train_cycles_90_98.csv
rearfoot_validate_cycles_99_100.csv
fullfoot_train_cycles_90_98.csv
fullfoot_validate_cycles_99_100.csv
```

### Allowed Nuisance Parameters

- Force zero or preload subtraction.
- Displacement/contact zero.
- Geometry-derived alignment.
- Fixture-specific indenter/platen geometry.

### Disallowed Nuisance Parameters

- Fixture-specific stiffness.
- Fixture-specific damping.
- Fixture-specific hysteresis/state.
- Arbitrary force scaling.
- Per-trial pressure multipliers.
- Manual COP shifts.

### Model Ladder

1. Shared whole-shoe effective foundation law.
2. Add or keep stateful hysteresis terms if needed.
3. Add constrained low-dimensional spatial modifiers only if failure analysis proves a single global law cannot pass.
4. Avoid regional material maps unless rearfoot/fullfoot failures are clearly location-driven.

### Validation Gates

All gates are evaluated on validation cycles only:

- Rearfoot peak force error under `10%`.
- Rearfoot force-displacement normalized RMSE under `10%`.
- Rearfoot hysteresis loop area error under `10%`.
- Fullfoot peak force error under `10%`.
- Fullfoot force-displacement normalized RMSE under `10%`.
- Fullfoot hysteresis loop area error under `10%`.
- Same shoe parameter set across rearfoot and fullfoot.

### Deliverables

- Cycle-split generated CSVs or NPZs.
- Fit JSON with shared parameters.
- Train/validation report.
- Force-displacement plots.
- Hysteresis plots.
- Per-trial residual diagnostics.
- Explicit pass/fail summary.

### Validation Report Contract

The Phase 1 validation report must separate fitting performance from held-out performance. At minimum it should contain:

- `schema_version`
- source manifest path
- train cycle windows
- validation cycle windows
- shared fitted shoe parameters
- allowed nuisance parameters used for each fixture
- per-fixture train metrics
- per-fixture validation metrics
- pass/fail result for every validation gate
- plots and CSV paths for measured-vs-predicted traces

The top-level pass/fail value is true only when every validation metric passes on both rearfoot and fullfoot held-out cycles.

## Phase 2: Dynamic Digital Instron

### Goal

Use the Phase 1 calibrated shoe model in a real dynamics loop, not just replay/evaluate force curves.

### Scope

- One shoe at a time.
- Bench tests only.
- Implement the contact/material law in Warp.
- Couple into MuJoCo-Warp through Newton `state.body_f`.
- Replay measured indenter trajectory.
- Compare simulated force/COP/wrench to held-out bench data.

### Architecture

```text
shoe grid + calibrated params
fixture transforms + measured trajectory
        |
        v
Warp contact/material kernel
        |
        v
body wrench / diagnostics
        |
        v
Newton state.body_f
        |
        v
SolverMuJoCo / MuJoCo-Warp dynamics
```

### Core Implementation Pieces

- Warp kernel computes per-cell compression from indenter/platen geometry.
- Warp kernel applies the calibrated effective foundation law.
- Warp kernel accumulates total force, COP, and wrench.
- Apply equal/opposite wrench to shoe and fixture bodies as appropriate.
- MuJoCo-Warp advances the dynamic state.
- Diagnostics buffers record force, COP, active area, max compression, energy, and contact count.

### Validation Gates

- Same rearfoot/fullfoot validation metrics as Phase 1, each under `10%`.
- Stable dynamics under the measured trajectory.
- COP/wrench physically plausible and continuous.
- No hidden fitting to validation cycles.
- No tuning to human running.

### Dynamic Report Contract

The Phase 2 report should use the same metric formulas as Phase 1 and add dynamic diagnostics:

- solver name and settings
- time step and substep settings
- body IDs receiving foundation wrenches
- peak active cell count
- peak and mean COP location
- max force jump between adjacent frames
- max compression
- active contact area over time
- net wrench over time
- energy/work consistency summary

Phase 2 passes only if the dynamic replay meets the same held-out force/hysteresis gates as Phase 1 and has no obvious unstable wrench/COP discontinuities.

## Design Rule

Phase 1 proves the effective property law is identifiable from intact bench data. Phase 2 proves that same law survives contact dynamics. Only after both phases pass should the shoe be attached to a humanoid.

## Metric Definitions

Official validation metrics use baseline-corrected force. Raw machine force should still be plotted for diagnostics, but pass/fail is based on corrected shoe response:

```text
F_meas = measured_force_n - force_zero_n
F_sim = simulated_force_n - simulated_baseline_n
```

Use pre-contact/inactive baseline frames to estimate `force_zero_n` and `simulated_baseline_n`. If a trial has too few inactive frames, the validation report must say so and use the configured preload subtraction policy instead.

### Active Frame Mask

Use active frames for force RMSE and peak force:

```text
F_peak_meas = robust_peak(F_meas)
active = F_meas >= 0.05 * F_peak_meas
```

The `0.05` active threshold is the initial standard. Revisit only if it creates obvious touchdown/release artifacts.

### Robust Peak Force Error

Use the mean of the top five active-frame force samples as the robust peak:

```text
robust_peak(F) = mean(top_5(F[active]))
peak_force_error = abs(robust_peak(F_sim) - robust_peak(F_meas)) / robust_peak(F_meas)
```

If fewer than five active samples exist, the trace is invalid for validation rather than silently relaxing the robust-peak rule.

Pass when:

```text
peak_force_error < 0.10
```

### Force-Displacement Normalized RMSE

Compute RMSE only over active frames so baseline zeros do not dilute the score:

```text
force_rmse_relative =
    sqrt(mean_active((F_sim - F_meas)^2)) / robust_peak(F_meas)
```

Pass when:

```text
force_rmse_relative < 0.10
```

Loading and unloading RMSE should be reported as diagnostics, but they are not hard gates for Phase 1 or Phase 2.

### Hysteresis Loop Area Error

Use positive dissipated loop area, not signed loop integral. Split the trace into loading and unloading branches, then compute:

```text
loading_work = integral_loading F dD
unloading_work = -integral_unloading F dD
hysteresis_work = loading_work - unloading_work
hysteresis_error = abs(H_sim - H_meas) / abs(H_meas)
```

where `D` is displacement/compression and `H_*` is positive dissipated work.

The branch split is based on the measured displacement peak in the active region:

```text
peak_displacement_index = argmax(D_meas[active])
loading = active frames up to and including peak_displacement_index
unloading = active frames from peak_displacement_index through release
```

The same branch indices are used for simulated and measured force so the comparison is not biased by simulation timing differences.

Pass when:

```text
hysteresis_error < 0.10
```

## Implementation Sequence

1. Add cycle-window trace generation for raw Digital Instron CSVs.
2. Add metric helpers implementing the definitions in this document.
3. Update the Phase 1 fitting path to train from generated train traces and validate from generated held-out traces.
4. Write the Phase 1 validation report contract.
5. Add tests using synthetic cycle CSVs that prove cycle splitting, baseline correction, robust peak, active RMSE, and hysteresis metrics.
6. Implement the Phase 2 Warp/MuJoCo-Warp dynamic replay only after Phase 1 metrics and reporting are stable.
7. Add Phase 2 validation reporting with force, COP, wrench, active area, and energy diagnostics.
