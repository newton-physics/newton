# Friction-compatible fused contact

`FoundationFused` now executes the existing `FrictionParameterAdapter` law
inside fused contact stages rather than falling back whenever an adapter is
attached. This includes the configured elastic-Coulomb default, Maxwell,
column-Maxwell, and mixed per-world methods 0, 1, 4, 5, 6, 7, 8 and 9.

The standalone and fused paths call the same per-column friction and ordered
diagnostic functions. Normal mechanics are unchanged. The normal-only stage
uses disposable bristle history; real friction history advances exactly once.
Traction capacity uses the external unilateral ground reaction, not signed
Pasternak transfer. Deflection, Maxwell force, sliding distance, release dwell,
energy and work diagnostics remain GPU-resident.

## Retained schedule

For a passive-surround bed, the previous path used ten launches per substep.
The retained path uses three:

1. Existing block-local surround sweeps: 512 threads per world.
2. Free-surface publication, pressure, normal contact and friction: 256 threads
   per world, with a barrier before neighbor compression reads.
3. Fixed-order friction diagnostics and wrench reductions: 32 threads per world.

All-driven beds omit stage 1. Legacy contact without an adapter retains its
single-block path. The single giant parameter-friction kernel was implemented
and tested, but was slower at 128 worlds on the available GPU. Smaller whole-world
blocks and forced occupancy did not consistently help. Separating the sparse
reduction tail avoids retaining a column-sized block's registers for that work.

CUDA, at most 1,024 columns, a declared ground plane, unique carriers, and
`clear_body_force=True` are required. Arbitrary adapters, including subclasses
with potentially overridden behavior, use the previous implementation.
The fusion remains forward-only, not a differentiable contact implementation.

`fused_apply=True` enables the compatible schedule automatically in existing
Cartesian engines and GPU Hogan batches. For matched comparison or a device
where it is slower, set `foundation.fused_apply=False` **before graph capture**;
this retains surround-only fusion and the same friction law. Replacing adapters,
buffers, timestep, or execution mode requires recapture. Existing settings buffers
can change coefficients or method IDs between replays without recapture.

## Reproducible measurement

Run module `projects.impedance_instron.cartesian.gpu.friction_benchmark` with
`--worlds 1 32 128 --steps 256 --repeats 9 --output <new-report.json>`.
Use `--friction-model maxwell` or `--friction-model column_maxwell` for other
configured defaults. The output parent must already exist; files are not overwritten.

The asset-free fixture has 910 columns, 400 driven and 510 passive, eight sweeps,
and a 125 microsecond timestep. It holds a loaded pose with prescribed linear
and angular velocity; **it is not a dynamically integrated running trial**.
Both graphs restore identical pristine arrays before replay. Compilation, graph
capture, resets and host readbacks are excluded from CUDA-event timing. Three
graph warmups precede alternating paired measurements. All 68 resident arrays
are compared, including the complete wrench, friction histories and diagnostics.
Reports record source hashes, launch names, compilation policy, device and runtime.

### RTX A4000 Laptop GPU, 2026-10-07

Final elastic-Coulomb run, nine paired repeats:

| Worlds | Previous median batch-step time | Fused median batch-step time | Median paired speed ratio |
| ---: | ---: | ---: | ---: |
| 1 | 99.05 us | 102.19 us | 0.973x |
| 32 | 121.63 us | 129.69 us | 0.935x |
| 128 | 372.27 us | 360.19 us | 1.022x |

All compared arrays matched exactly in this benchmark. Three-repeat 32-world
probes of Maxwell and column-Maxwell also matched exactly, with paired ratios
1.020x and 1.038x respectively. Ratios are medians of paired ratios, not ratios
of the two independent medians. The shared desktop GPU's clocks and other
activity were uncontrolled; earlier repeats varied. **No substantial or portable
speedup, and no end-to-end training speedup, is established.** The benefit
demonstrated here is friction-compatible fusion and fewer launches.

Local evidence: `outputs/friction_fused_verified_benchmark_20261007.json`,
`outputs/friction_fused_verified_maxwell_benchmark_20261007.json`, and
`outputs/friction_fused_verified_column_maxwell_benchmark_20261007.json`.

## Regression coverage

The focused 40-test run covers all laws, automatic defaults/material refresh,
exact normal fields, complete histories/diagnostics/wrenches, stick/slip reversal,
unloading/recontact, odd/even sweeps, 511/512/513 and 1,024/1,025 boundaries,
no-surround beds, CPU/generic/additive/custom fallbacks, graph reset and mutable
settings, disabled worlds, diagnostic clock, and CPU/GPU runner trace parity.
The new tests enforce the actual fused launch names and reject adapter delegation.

A broader 216-test run before the final schedule tuning reported 23 skips
(including unavailable artifact-dependent tests) and 11 failure records in
existing scope/tape tests. Baseline checks reproduced the CPU failures using
HEAD sources: Windows newline/encoding-sensitive hashes, historical normal-array
digests, and a zero-friction contact-tape stick-state mismatch. These unrelated
checks were not relaxed or rewritten. Baseline source changes still require
fresh full-trajectory qualification for a saved controller fit.