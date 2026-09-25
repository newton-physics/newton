# Heterogeneous collider regression audit

Audit date: September 24, 2026.

Compare upstream `45458023f24629c1feffb1f336594808f537db14` with the complete
feature branch through `502ace94`, including both the Newton-contact and native
MuJoCo-contact implementations. The capture fix, test corrections, and coverage
added during this audit are described below. The audit does not assume that passing the
earlier focused feasibility tests establishes backward compatibility.

## Scope and method

Independent reviews traced the solver, selection/API, and native contact
filtering changes using the repository's Fit, Requirements, and Standards
guidelines. Existing defaults remain unchanged. Most solver additions are
behind `allow_heterogeneous_shapes=True`; the shared behavioral changes are
refreshing static collider poses and rejecting unmapped Newton contact endpoints.

The test matrix includes native and Newton-generated contacts, MuJoCo Warp
on CPU/CUDA, MuJoCo C, one and multiple worlds, reset and property updates,
fixed and spatial tendons, sites, explicit pairs, actuators, equalities,
contact force reporting, selection masks, gradients, and optional imports.

A pristine upstream tree was extracted with `git archive` and tested using
the same interpreters and dependencies as the branch. Numerical comparisons
use independent processes and verify the imported source paths. GPU timing
is performed separately from concurrent test execution.

## Results

The complete strict-warning run finished with one preexisting viewer failure
and one GPU allocation error that passed when rerun serially on both snapshots.
No introduced default-path regression was found. Full-run quantitative coverage
is unavailable because the runner discarded its coverage data on the failure
exit; supplementary incomplete runs are not counted as full-suite evidence.

| Check | Evidence |
| --- | --- |
| Full repository suite | The runner discovered 7,795 cases and reported 7,406 tests in 5,066.969 seconds: 148 skipped, one failure, and one error. Both unsuccessful cases are classified below; this is not a clean full-suite pass. |
| Default solver, CPU | All 368 recorded arrays are bitwise identical to upstream across six backend/contact/world configurations and four lifecycle phases. |
| Default solver, CUDA | All 368 recorded arrays are bitwise identical to upstream for the same comparison. |
| Contact and constraint results | Comparisons include actual contact IDs, world/shape mappings, distances, nonzero contact forces, generalized constraint forces, body/joint states, geometry, and model properties. |
| Existing features with ragged geometry | Native and Newton contacts match isolated default worlds through initialization, stepping, property updates, and reset; fixed tendons, joint/tendon/body actuators, and connect equalities are exercised on CPU and CUDA. |
| Existing static geometry | Homogeneous worlds with rotated welded bodies settle at independently computed heights, both initially and after local shape updates. |
| Selection | 42 existing/new selection tests pass on CPU. Additional probes cover partial and noncontiguous selections, per-articulation masks, empty selections, custom frequencies, and reverse-mode gradient scattering. |
| Optional imports | Import blocking and an installation without MuJoCo both preserve public selection, solver-class imports, and XPBD simulation. Five existing lazy-import tests pass. |
| Python 3.10 | 49 CPU tests and the final CUDA captured-update regression pass on CPython 3.10.20, NumPy 2.2.6, Warp 1.17.0, and MuJoCo/MuJoCo Warp 3.12.0; native geometry and contact record compaction are included. |
| Wheel packaging | The built wheel contains the new contact-filter module and passes a non-editable Python 3.10 installation smoke test with only NumPy and Warp. |
| Documentation | The corrected strict Sphinx HTML build passes with zero warnings, all 89 doctests pass, and all 16 tracked API pages regenerate unchanged. |
| CI and benchmark tooling | All 42 CI-tooling tests and 24 ASV harness tests pass; deprecation allowlists validate. |
| Feature regressions under CI policy | All 30 heterogeneous MuJoCo tests pass on CPU/CUDA with `--strict-warnings` after the final capture fix, including the four additional audit regressions (92.541 seconds). |
| Native filtering edge cases | CPU probes cover zero geometry/capacity, signed collision groups, per-world exclusions, a real collision-mask compiler fallback, inactive slots, contact disable/re-enable, and property updates. |
| Callback lifecycle | Replacing a contact buffer is rejected explicitly; supported material and inertia updates preserve buffer ownership and callback validity. |
| Captured property updates | Native heterogeneous pose/material updates now capture and replay correctly; repeated updates match eager dynamics and preserve eager rejection of unsupported geometry changes. |
| Deterministic graph replay | With sensors disabled, two CUDA `RUN_TO_RUN` reset/replay runs produce identical poses, velocities, contact IDs, and distances while actively rejecting inactive-slot contacts. No collision/constraint capacity overflow occurs. |

The default numerical probe runs six configurations: Warp native/Newton
contacts with one/two worlds, and MuJoCo C native/Newton contacts with one
world. Each records initialization, simulation, property update followed by
simulation, and reset followed by simulation. Its scene includes fixed and
spatial tendons, sites, explicit contact-pair export, three actuator target
types, connect equality, and active sphere/plane contact. This is a differential
compatibility check, not exhaustive physical validation of every feature in
every combination.

## Full-suite outcome

The four-worker run completed in 84.45 minutes. The renderer test
`test_large_quad_is_stable_under_parallel_camera_translation` failed because
shadows changed 55.81% of the frame, exceeding its 10% bound. The same assertion
and value reproduce on pristine upstream and the branch with the locked
environment and serial execution. Both runs report OpenGL/CUDA interoperability
error 999 and use the CPU-copy fallback. The renderer and its test are unchanged.

`test_reconstruct_cube_mesh` failed to allocate a 256 MiB CUDA buffer during the
parallel run. Its default voxel hash grid alone reserves approximately 2.75 GiB
across its arrays. The unchanged test passes serially on both snapshots; concurrent
GPU memory pressure is consistent with the failure. The upstream parallel
allocation failure was not independently reproduced. The two-case serial reruns
take 10.126 seconds on the branch and 10.245 seconds on upstream, each with only
the same viewer failure.

Of the 148 skips, 62 require an unconfigured USD asset folder, 42 mark behavior
not yet implemented, 14 are CPU variants of CUDA-only SDF tests, and 17 are
manual visual or trajectory debugging tests. The remainder cover unavailable
robot assets and explicitly unsupported device or solver combinations. The
runner's summary and JUnit aggregate are used for counts. Its 7,515 JUnit rows
contain 7,406 individual test results plus 109 class-level skip records; the
148 reported skips comprise those 109 class skips and 39 individual skips.

## Default-path timing

The same locked environment ran the pristine baseline and final source in
baseline/branch/branch/baseline order, after all other GPU tests exited. Each
process simulates 1,024 homogeneous worlds with three convex hulls per free
cuboid and a shared plane, using `allow_heterogeneous_shapes=False`. Native and
Newton contacts use identical options and explicit capacities. Construction,
CUDA graph capture, and 400 settling steps are excluded. Each of seven samples
times 800 steps through an eight-step graph, with device synchronization around
the sample.

| Contact generation | Baseline mean run median | Branch mean run median | Change | Baseline sample range | Branch sample range |
| --- | --- | --- | --- | --- | --- |
| Native MuJoCo | 0.285962 ms/step | 0.285638 ms/step | -0.11% | 0.284929–0.298720 ms | 0.284734–0.298398 ms |
| Newton | 0.338124 ms/step | 0.339783 ms/step | +0.49% | 0.335026–0.347134 ms | 0.334257–0.387224 ms |

Each central value is the arithmetic mean of the two per-process medians. The
small differences fall within observed sample variation and do not establish
either a regression or a speedup. This is one synthetic steady-state workload,
not a bound on initialization costs or real robot training performance.

All runs verify the imported source path, finite state, expected support height,
12,288 contacts, and no collision or constraint capacity overflow. Both snapshots
also reach the configured line-search iteration limit in some worlds
(`LS_ITERATIONS`, flag 1024); the capacity assertion explicitly excludes solver
iteration-limit flags. Raw samples and checks are retained in
`/tmp/newton-audit-bench-{base1,head1,head2,base2}.json`, with calculated comparisons
in `/tmp/newton-audit-bench-summary.json`.

## Issues found and corrected

Native heterogeneous shape-property notification originally copied immutable
geometry fields to the host even during CUDA capture. A permanent regression
failed with CUDA error 906 before the fix, while the upstream/default native
path and heterogeneous Newton contacts passed the same operation. The solver
now skips this host-only validation while capturing, preserving the existing
geometry immutability contract and eager rejection checks. The regression
compares repeated captured pose/friction updates against eager dynamics and
independent resting heights, verifies material values and contact IDs, and
checks that an eager scale change is still rejected. Documented guidance is
to validate once before capture and leave compiled geometry/filter fields
unchanged throughout capture and replay. This affects only the new native mode.

The new fixed-body regression originally emitted the solver's expected
standalone-root warning without asserting it. Ordinary `unittest` passed,
but the CI runner's `--strict-warnings` policy raised it as an error on CPU
and CUDA. The test now explicitly asserts that warning when constructing its
combined and isolated reference solvers. Production warning policy is unchanged.

The new "Different collider counts across worlds" documentation heading had
a 38-character underline for its 39-character title. The complete Sphinx HTML
build completed its notebook execution but failed the CI `-W` policy on this
single warning. The underline is corrected; strict documentation rebuilding
and doctests validate the correction.

Three additional permanent regressions cover automatic capacities with 1/3/2
hulls, construction-time and runtime contact disable/re-enable, and a large
pair graph that forces conservative native masks. The last scene starts below
ground so unused slots would create real native contacts without filtering;
the test requires exactly the 47 original shape/plane contacts with valid
world mappings and finite forces. All three pass on CPU/CUDA and fail against
the pre-feature upstream API. A stronger negative control disables only the
contact filter: the large-pair regression fails with 92 contacts instead of 47,
demonstrating that it detects the 45 phantom contacts from unused slots.

A fixed-category changelog fragment now explicitly records the static-collider
pose correction, which also applies to default solver users.

## Review disposition

**Fit: Fits** for the documented experimental feature. Selection and the MuJoCo
adapter own the affected state and geometry mappings. Grouping worlds by collider
layout into separate homogeneous solvers is an alternative, but would move
batching and scheduling into downstream applications. The bounded slot-based
implementation preserves the original geometry and existing batched state APIs.

The new flags configure runtime views and solver export; existing USD collider
authoring needs no new schema. Public defaults, lazy dependency imports,
experimental status, viewer/MJCF limitations, and packaging remain consistent.
Maintainer review is still needed for long-term ownership of MuJoCo Warp callback
compatibility and acceptable costs on real robot workloads.

## Preexisting observations

These probes reproduced the following behavior on pristine upstream as well
as the branch; they are not introduced by this feature:

- Automatic native constraint capacity can be insufficient for a five-hull
  box resting on a plane: 80 pyramidal rows exceed the default 64-row budget.
  The documented explicit `njmax` control remains necessary for such scenes.
- CPU `RUN_TO_RUN` determinism with sensors enabled can hit a Warp codegen
  limitation in MuJoCo Warp's tactile sensor kernel, which mixes atomic
  reduction types. The new contact filter passes CPU deterministic execution
  when sensors are disabled.
- Standalone fixed roots retain the template body placement across worlds.
  Per-world root placement uses articulated fixed roots, as documented in the
  feature scope. Distinct local collider poses on standalone roots are covered.

## Environment and reproduction

The comprehensive environment uses the repository's locked development and
CUDA 13 PyTorch dependencies in a separate virtual environment:

```bash
UV_PROJECT_ENVIRONMENT=/tmp/newton-regression-venv \
  uv sync --locked --extra dev --extra torch-cu13 --python 3.12
UV_PROJECT_ENVIRONMENT=/tmp/newton-regression-venv \
WARP_CACHE_ROOT=/tmp/newton-audit-warp \
COVERAGE_CORE=sysmon \
  uv run --no-sync -m newton.tests --strict-warnings --no-cache-clear \
  -j 4 --parallel-timeout 7200 \
  --junit-report-xml /tmp/newton-audit-full.xml \
  --coverage --coverage-html /tmp/newton-audit-coverage \
  --coverage-xml /tmp/newton-audit-coverage.xml
```

The first full-suite run uses the same `sysmon` line-coverage configuration as
repository CI. The canonical runner exits on test failure before combining
coverage, and its temporary-directory cleanup removed that run's coverage
parts. Its JUnit results remain available, but its requested coverage XML/HTML
were never produced and no full-run coverage percentage can be recovered.
A supplementary two-worker warm-cache repeat used the independently checked
external `/tmp/newton-preserve-coverage.py` wrapper to preserve completed coverage
parts without changing test outcomes. That optional repeat was stopped before
completion so it would not delay the completed regression audit. Its log
`/tmp/newton-audit-full-warm.log` and `/tmp/newton-audit-full-coverage-parts`
archive are retained as incomplete diagnostics only, with no full-run coverage
claim. The completed main run was not interrupted.

An earlier branch-coverage run was also stopped to reduce tracing overhead; its
incomplete log is retained as
`/tmp/newton-audit-full-branch-partial.log` and is not counted as a full pass.

Main runtime: Python 3.12.13, NumPy 2.5.0, Warp 1.17.0, MuJoCo and MuJoCo Warp
3.12.0, PyTorch 2.13.0+cu130, OpenUSD 26.8, and an NVIDIA RTX PRO 3000 Blackwell
laptop GPU with driver 595.91.07. Independent numerical A/B probes use the
existing simulation environment with NumPy 2.5.3; both revisions use the same
environment in each comparison. The full suite additionally exercises the
locked NumPy version.

Local evidence is retained under `/tmp`: `newton-audit-full.log`,
`newton-audit-full.xml`, the explicitly incomplete logs and coverage parts above,
`newton-audit-rerun-{base,head}.log`, `newton-audit-default-benchmark.py`,
`newton_audit_feature_reuse.py`,
`default_{base,head}{,_cuda}.npz`, `newton_audit_static_default.py`,
`newton-selection-probes.log`, `newton-optional-import-probe.log`,
`newton-audit-py310-tests.log`, `newton-wheel-smoke.log`,
`newton-audit-py310-capture-final.log`,
`newton-capture-notify-before.log`, `newton-capture-notify-after.log`, and
`newton_native_filter_negative_control.log`.

## Boundaries

Runtime validation covers Linux on this CPU and one GPU architecture, with
Python 3.10 and 3.12. It does not establish compatibility on every supported
OS, Python version, or GPU, multi-GPU execution, or arbitrary external caller
subclasses. The actual Isaac Lab task and long training runs remain outside
the synthetic and repository regression workloads. The feature's documented
experimental restrictions and collision-capacity costs still apply. Python
line coverage follows the repository exclusions for tests, examples, and
Warp-decorated functions; it is not device-kernel coverage or branch coverage.
