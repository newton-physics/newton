# Digital Shoe

## Start here

Digital Shoe owns the portable artifact and shared mechanics. It does not fit
bench data or implement an impedance controller. See the [three-project map](../README.md)
for package boundaries, file ownership, data paths and supported commands.

```bash
uv run --no-sync -m projects.digital_shoe --help
uv run --no-sync -m projects.digital_shoe view --mode instron --fixture rearfoot_punch --viewer gl
```

| Task | Package command |
|---|---|
| View artifact-only examples | `view` |
| Rebuild the offline report | `report` |
| Record example GIFs | `record` |
| Validate acquisition metadata | `check-acquisition` |

Append `--help` to a command for its existing options. The old `.showcase`,
`.report`, `.record_gifs` and `.acquisition` module commands remain supported.
Runtime/material/contact files and report resources retain their locations.


`projects.digital_shoe` is the path-independent runtime and presentation layer
for an intact shoe identified from mechanical test data. It does not depend on
gait, C3D, OpenSim, or a human model.

The project answers one narrow question:

> Can an effective nonlinear, rate-dependent shoe model be identified from
> controlled Instron measurements and deployed unchanged in a live simulation?

The checked data support a research proof of concept. They do not yet validate
new rates, temperatures, impacts, or shoes.

## Architecture

The portable artifact flows from identification to consumption:

```text
projects.digital_instron_v2 (data, geometry, fitting)
                    |
                    v
          digital_shoe.json
                    |
                    v
projects.digital_shoe (strict loader, runtime, report, demos)
```

The runtime artifact contains the full 910-column shoe bed, baked calibrated
midsole and shoe-last visual meshes, all six constitutive constants, neighbor
topology, coordinate semantics, held-out curves, metrics, and source hashes. It
contains no absolute file paths. The runtime never falls back to hidden
parameters when loading a named shoe.

## Full-bed Instron replay

The Virtual Instron now retains the entire exported bed for either fixture, not
only the directly pressed subset. The full-foot fixture drives 611 of the 910
columns; the rearfoot punch drives 62. The remainder relaxes through the same
neighbor and material law used during identification. Bench rendering uses the
solved compression of every column.

Both fixtures use the same mesh-derived outsole heights, rest top and shoe
orientation. Old rearfoot exports with zero-bottom column datums are rebased by
shifting the fixture anchor and free top together onto the intrinsic bed. Rest
lengths, loading patches and uniform rearfoot shortening remain unchanged. This
is not a new gap-aware contact solve against the flat punch visual.

Both fixture replays check the warmed force waveform at the force-evaluation
time, not the later display clock. The passive solve uses 32 quasi-static sweeps
per substep; a 32/64-sweep check on the retained artifact changed the force curve
by less than 0.1% of peak. This is a numerical replay check, not new experimental
validation or a material refit.

```bash
uv run --no-sync -m projects.digital_shoe.showcase --mode instron \
  --fixture fullfoot_last --viewer gl
uv run --no-sync -m projects.digital_shoe.showcase --mode instron \
  --fixture rearfoot_punch --viewer gl
```

Use `--viewer null --num-frames 180 --test` for either fixture's headless replay.

## In-depth two-term mechanics report

[Read the contact and material report](mechanics_report/REPORT.md) for the
implemented equations, identified versus assumed settings, solver behavior,
source references, and validation limits. The [rebuild instructions](mechanics_report/README.md)
produce six figures and a self-contained offline HTML report from the explicitly
pinned two-term artifact. Generated measurements, geometry figures, and reports
stay local and ignored; the tools do not change the shoe or physics sources.

## Friction-only solver mechanics and identification

[Compare tangential models](FRICTION.md) with five opt-in solver modes
(explicit/implicit legacy bristle, regularized Coulomb, and explicit/implicit
consistent deflection) while leaving normal compression, material, and contact
mechanics unchanged.

Read [FRICTION_IDENTIFICATION.md](FRICTION_IDENTIFICATION.md) and [FRICTION_ONSET.md](FRICTION_ONSET.md); see [FRICTION_CONTINUITY.md](FRICTION_CONTINUITY.md) and [FRICTION_CONTROLLER_REFIT.md](FRICTION_CONTROLLER_REFIT.md) for the raw-force mechanical comparison for the reproducible
CLI workflow, frozen-history sweeps, free-leg GPU dynamic optimization across
22,808 attempted candidates, scoring schemas, and honest qualification limits.
No calibrated parameters are promoted or installed as defaults.

```bash
# Compare tangential formulations headlessly
uv run --no-sync -m projects.digital_shoe friction --mode implicit_deflection --viewer null --num-frames 120 --test

# CLI workflow help
uv run --no-sync -m projects.digital_shoe friction-sweep --help
uv run --no-sync -m projects.digital_shoe friction-fit --help
uv run --no-sync -m projects.digital_shoe friction-check --help
```

## One shared law

This package owns the mechanics used by Digital Instron and the impedance rigs.
`material.py` defines one set of Ogden–Hill/Maxwell expressions for both vectorized
NumPy fitting and compiled Warp simulation/autodiff. `contact.py` defines the
unilateral support, symmetric neighbor coupling, passive balance and anchored
bristle law. `runtime.py` and the differentiable adapters differ in state storage,
not in their active material or friction equations.

`FoundationConfig`, the impedance leg shoe and Cartesian engine default to
`elastic_coulomb`: each column uses `G_eq A / L` stiffness with a Coulomb cap
and no tangential damping. Explicit `maxwell` and `column_maxwell` modes remain
available; see
[`FRICTION_COLUMN.md`](FRICTION_COLUMN.md) for its equations, compatibility, and
calibration limits.

`rendering.py` reconstructs shared bench and carried-shoe endpoint geometry. It
does not interpret a pressure reference or a friction anchor as a material-point
displacement. The carried passive surround follows the shoe during flight.

A bench indenter and a carried shoe require different geometric boundary inputs.
Only a carried outsole opts into `FoundationConfig.ground_height_m`; its external
ground force, friction capacity, COP and full wrench then use actual plane
contact points. Bench top anchors retain their own reference datum and report
indenter load transfer. The material and contact primitives remain shared.

See `CONSOLIDATION.md` and `projects/digital_instron_v2/README.md` for the adapter
map, compatibility notes and validation commands. Source identities used by the
impedance checkpoint include the runtime **and** shared material/contact modules.

## Identify and export

From the repository root:

```bash
uv run --extra examples -m projects.digital_instron_v2.export_digital_shoe \
  --manifest DigitalInstron/manifest_v2.json \
  --output DigitalInstron/digital_shoe_showcase
```

This command fits cycles 90–98, evaluates held-out cycles 99–100, and writes:

```text
DigitalInstron/digital_shoe_showcase/digital_shoe.json
DigitalInstron/digital_shoe_showcase/validation_report.html
```

The report is organized as methods, results, examples, and reproducibility. Its
methods section derives the column kinematics, Hyperfoam equilibrium pressure,
Maxwell recurrence, Pasternak coupling, wrench/COP assembly, and normalized fit
objective. `methods.mmd` is the Mermaid source for the workflow diagram;
`methods.svg` is its pre-rendered offline form embedded in the HTML.

The output directory is intentionally ignored. The measurements, footwear
geometry, fitted artifact, and their derivatives are not cleared for upstream
redistribution. See `ASSET_PROVENANCE.md`.

To refresh only the report layout, reuse the existing artifact and recordings:

```bash
uv run -m projects.digital_shoe.report \
  DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --output DigitalInstron/digital_shoe_showcase/validation_report.html \
  --media-dir DigitalInstron/digital_shoe_showcase
```

The report opens with its validation status and limitations. Section links lead
to methods, results, examples, and reproduction commands. Detailed equations and
source hashes are expandable; the failed gates remain visible without expanding
anything. The report is self-contained and works offline.

## Mechanical demonstrations

All three scenes consume only `digital_shoe.json`.

```bash
# Displacement-controlled validation against held-out Instron curves.
uv run --extra examples -m projects.digital_shoe.showcase \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --mode instron --viewer gl

# An 80 kg free six-DOF body-weight drop carried by the calibrated shoe last.
uv run --extra examples -m projects.digital_shoe.showcase \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --mode drop --viewer gl

# A controlled rigid rocker that moves contact and COP heel-to-toe.
uv run --extra examples -m projects.digital_shoe.showcase \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json \
  --mode rocker --viewer gl
```

Record all three audited GIF loops and rebuild the report with the GIF bytes
embedded directly in the HTML:

```bash
uv run --extra examples -m projects.digital_shoe.record_gifs \
  --artifact DigitalInstron/digital_shoe_showcase/digital_shoe.json
```

The command writes `instron.gif`, `drop.gif`, and `rocker.gif` beside the
artifact, then rewrites `validation_report.html` as a portable single file. It
records every display frame at 720 px. Instron and rocker play at 12 FPS. The
drop is sampled at 240 Hz for one second and plays at 24 FPS, making its impact
and rebound about ten times slower than real time. The report places each animation on its own
full-width row. Spring colors use each column's maximum compression within
the displayed frame on one fixed scale: blue 0 mm, cyan 6.7 mm, yellow 13.3 mm,
and red 20 mm or more. The report includes the same legend.
Add `--clear-kernel-cache` if a prior interrupted Warp compilation left a
missing `.ptx` cache entry.

Use `--viewer null --num-frames N --test` for a headless audit. Useful minimum
runs are 180 frames for Instron, 60 for drop, and 80 for rocker.

The scenes do not use proxy shoe boxes. The Virtual Instron renders the posed
shoe-last or punch above exposed springs, with two endpoint nodes per column
and no solid midsole surface. The drop is a free six-DOF 80 kg body carried by
the calibrated full-foot last; it also renders the last above exposed colored
springs. The rocker renders only its spring bed. Spring length and
contact-color changes remain visible in all three modes.

The drop demo adds 5 N·s/m per-column normal damping for impact stability. That
value is a scenario parameter and was not identified by the current
normal-compression tests. Friction is also not claimed as fitted. The 80 kg,
40 mm drop reaches about 20 mm column compression, slightly beyond the original
17.9 mm full-foot test amplitude, so it is an extrapolative mechanical showcase
rather than a validated impact prediction.

### Warp cache recovery

The OpenGL warning about falling back from MSAA is harmless. An error that says
Warp could not open a generated `.ptx` file means a kernel-cache write was
interrupted. Clear the compiled cache once and rerun:

```bash
uv run python -c "import warp as wp; wp.clear_kernel_cache()"
```

The prescribed Instron and rocker modes avoid articulation initialization, so
they do not compile the articulation module merely to set a rigid carrier pose.

## Current held-out result

| Fixture | Peak error | Active RMSE | Hysteresis error |
|---|---:|---:|---:|
| Rearfoot punch | about 12.6% | about 6.7% | about 15.5% |
| Full-foot last | about 14.7% | about 4.1% | about 17.2% |

The active-force RMSE gates pass. Peak and hysteresis gates fail. The HTML report
keeps the overall research status and every failed gate visible.

The held-out cycles are adjacent cycles from the same approximately 0.5 s runs.
This tests implementation consistency and local repeatability, not broad dynamic
generalization.

## New data

`ACQUISITION_PROTOCOL.md` defines the next multi-rate and relaxation experiment.
Use `acquisition_manifest.example.json` as a starting point and validate it with:

```bash
uv run -m projects.digital_shoe.acquisition path/to/acquisition_manifest.json
```

## Tests

The shared-law tests check source identity, float64 host precision, CPU/CUDA
values and derivatives, contact histories, and retention of the full bench bed.

```bash
uv run --no-sync -m unittest newton.tests.test_digital_shoe_material \
  newton.tests.test_digital_shoe_shared_contact newton.tests.test_digital_shoe_consumers
uv run --no-sync -m unittest newton.tests.test_digital_instron_diff
uv run --extra dev -m unittest newton.tests.test_digital_shoe
uv run --extra dev -m unittest newton.tests.test_digital_instron_core
uv run --extra dev -m unittest newton.tests.test_digital_instron_dynamics
```
