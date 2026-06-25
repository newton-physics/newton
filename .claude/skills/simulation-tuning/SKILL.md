---
name: simulation-tuning
description: Use when tuning a Newton physics scene or diagnosing instability — NaN, jitter, explosive motion, penetration, weak grasp or object slip, poor drive tracking — or when choosing solver, contact, or drive parameters for SolverMuJoCo, SolverXPBD, SolverVBD, SolverFeatherstone, or other Newton solvers.
---

# Newton Simulation Tuning

Route tuning work through the diagnostic order below, then read the matching
docs page for parameter detail. The docs are the source of truth; this skill is
triage + discipline.

## Diagnose first — what kind of problem is this?

Classify before changing parameters (most "soft contact" symptoms are not
contact problems):

- **Initialization / geometry** — initial penetration, collision-vs-visual mesh
  mismatch, wrong joint state.
- **Control** — bad controller/IK target, step changes in drive targets.
- **Model** — missing joint friction/armature/damping, missing drive import,
  bad mass/inertia.
- **Capacity** — too few contact/constraint rows.
- **Contact / solver** — tune only after the above are ruled out.

## Agent checklist

1. Identify the active solver class; read its public constructor / config object
   and the "Joint feature support" docs table for supported model attributes.
2. Verify each proposed option is supported by that solver. Do **not** copy
   option names across solvers or from external MuJoCo/Omniverse docs. Example:
   `noslip_iterations` is **not** a `SolverMuJoCo` option.
3. Record `dt`, substeps, contact-refresh cadence, solver params, contact
   materials, and drive gains before changing anything.
4. Reproduce the symptom in a minimal scene.
5. Change one category at a time: model → timestep → solver convergence →
   contacts → drives → performance.
6. Prefer physically meaningful changes before solver brute force.
7. Keep the final recommendation solver-specific.

## Where to read for depth

| Need | Read |
|------|------|
| Workflow, principles, symptom table | `docs/concepts/simulation_tuning.rst` |
| Supported knobs per solver, sanity-check math | `docs/concepts/simulation_tuning_solvers.rst` |
| MuJoCo-Warp constraint model, ke/kd↔solref/solimp, task templates | `docs/concepts/simulation_tuning_mujoco.rst` |
| Which joint features each solver supports ("Joint feature support" table) | `docs/api/newton_solvers.rst` |

## Key facts

- Newton `ke`/`kd` keep their force-space units but on the `SolverMuJoCo` path are
  converted into MuJoCo `solref` (exactly how depends on the shape's `solref_mode`);
  they are not a Young's modulus or a direct world-space penalty spring. See the
  MuJoCo-Warp page.
- Harder is not always more stable. Tune to task metrics, not one penetration
  number.
- Reduce `dt` / add substeps before raising stiffness — usually the most
  reliable stability fix.
