# Newton Coupling SDD Review

**Original review date:** 2026-03-23
**Last updated:** 2026-05-12
**Reviewer:** Gilles Daviet (with Claude)
**SDD:** [Newton Coupling SDD (WIP)](https://docs.google.com/document/d/16Bhp9Fw4lsLdMt3AhJnVR2V_iVcvBw8Xc2zmMC9L2IQ/edit?tab=t.0)

## Executive Summary

The SDD's main Python-side recommendation has been validated: a coupled solver
can preserve the standard `SolverBase.step(state_in, state_out, control,
contacts, dt)` interface while owning solver-specific model views, state
distribution, and reconciliation internally.

The implementation has converged on a small hierarchy:

- `SolverCoupled` is the common multi-solver base.
- `SolverProxyCoupled` owns lagged proxy / virtual-inertia coupling.
- `SolverAdmmCoupled` owns linearized ADMM coupling for model-derived
  attachments and internally detected contacts.

The earlier "single-pair wrapper" idea was removed. Examples now construct the
generic coupled solvers directly and are kept standalone. README gallery updates
are deferred because the API is still experimental.

USD ownership, `physicsOwner` parsing, and automatic coupled-solver
construction remain out of scope for this branch.

## Updated Codebase Findings

### Solver Architecture

All coupled solvers remain drop-in `SolverBase` implementations. The external
caller still owns one global model, one input/output state pair, one control,
and one contact object. Internally, `SolverCoupled` creates one `ModelView` and
state pair per entry, then reconciles owned quantities back to the global output
state.

### Model Views and Ownership

`ModelView` is implemented in `newton/_src/solvers/coupled/model_view.py`.
It delegates to the parent model for unchanged fields and stores view-local
overrides for:

- entity counts and enabled ranges,
- body and particle ownership,
- body and particle proxy flags,
- body mass/inertia and particle mass,
- shape flags and contact-pair filtering.

`BodyFlags.PROXY` and `ParticleFlags.PROXY` now exist. They are model-view
metadata used by coupled solvers and sub-solvers to distinguish owned dynamic
entities from finite-mass proxy entities.

### Coupling Modes

| Mode | Implementation | Current role |
|------|----------------|--------------|
| One-way / collider exchange | Normal solver construction plus ownership-specific views | Still the lowest-cost path when action-reaction feedback is not required. |
| Lagged proxy coupling | `SolverProxyCoupled` | Main path for MuJoCo/Kamino/VBD/XPBD/MPM proxy examples and the cable robot port. |
| Linearized ADMM | `SolverAdmmCoupled` | Prototype for symmetric joints, body-particle attachments, and frictional contacts without proxy bodies. |

### Contact Handling

The initial review identified the single external `contacts` argument as a risk.
The implementation handles this by letting the coupled solver own any internal
collision detection that is needed by a coupling scheme:

- proxy coupling can use the outer contacts or a proxy-local collision pipeline
  supplied by the `SolverProxyCoupled.Proxy`;
- ADMM coupling detects cross-solver rigid-rigid, rigid-particle, and
  particle-particle contacts internally;
- particle-particle ADMM detection currently uses a private hash-grid stream
  structurally similar to `newton.Contacts`, but it is not a public contact
  streaming API yet.

## Design Assessment After Implementation

### Single Model With `ModelView`

**Verdict:** Validated.

The fallback model-view approach avoided deep copies and let existing solvers
reuse normal model arrays. The important addition was explicit mutation helpers
for body inertia and particle mass, plus ordinary `notify_model_changed()` calls
when a solver must refresh private model caches after a view mutation.

### `SolverCoupled` as `SolverBase`

**Verdict:** Validated.

Keeping the normal solver interface made examples and tests look like standard
Newton simulations. The base class is useful, but the scheme-specific loop
belongs in derived classes. Moving lagged proxy logic into `SolverProxyCoupled`
made that boundary clearer and mirrors `SolverAdmmCoupled`.

### Proxy Body Integration

**Verdict:** Use hooks plus fallbacks, not solver-wide mandatory `PROXY`
semantics.

The implemented API asks solvers to declare only the hooks where the generic
fallback is insufficient:

- input-state notification,
- proxy velocity rewind,
- proxy force/impulse harvesting,
- proxy contact preparation,
- effective mass queries.

This avoids requiring every solver to understand every proxy mode. Solvers that
do need private state updates, such as VBD and MPM, opt in through
`coupling_capabilities()`.

### Force and Inertia Mutation

**Verdict:** Keep these as default public model/state operations.

Force feedback is injected by writing `state.body_f`, `state.particle_f`, or
`control.joint_f`. Virtual/proximal inertia is installed by mutating
`ModelView` mass and inertia arrays. These are not custom hooks; a solver only
needs a hook when it has private buffers or private history that must be kept in
sync with those public mutations.

### USD Representation

**Verdict:** Still deferred.

The Python API is now concrete enough to inform a USD schema, but this branch
does not implement `physicsOwner` parsing, `NewtonPhysics.SolverApi`, or
`NewtonPhysics.CouplingApi`.

## Current Risk Register

| Risk | Status |
|------|--------|
| Model view is not a full model | Mitigated by `ModelView` delegation and solver tests. |
| Solver-private state gets stale after coupler mutations | Mitigated by `NOTIFY_INPUT_STATE_UPDATE` and normal `notify_model_changed()` paths; still needs broad solver coverage as more solvers opt in. |
| Proxy momentum harvesting includes non-contact forces | Mitigated for VBD and MPM by custom harvesters; still a limitation of the generic fallback. |
| Contact indexing breaks under model views | Mitigated by using view-local collision pipelines where needed and explicit local-to-global proxy maps. |
| ADMM parameter tuning is scene-dependent | Still open. Needs better guidance for `rho`, `gamma`, Baumgarte, and effective-mass weighting. |
| Public contact streaming API is unclear | Still open. Current ADMM particle-particle stream is private. |
| USD authoring path is absent | Deferred. |

## Implementation Outcome

The original phased plan is now best interpreted as:

1. **Foundation:** Implemented. `ModelView`, proxy flags, `SolverCoupled`, hook
   dispatch, and public exports exist.
2. **Concrete coupling schemes:** Implemented as experimental prototypes.
   `SolverProxyCoupled` covers lagged proxy coupling; `SolverAdmmCoupled`
   covers model joints, body-particle attachments, and contacts.
3. **Example ports:** Partially complete. The cable robot proxy scheme and MPM
   proxy examples have framework variants. The original hand-written examples
   remain as references.
4. **USD and automatic construction:** Not started.
5. **User-facing polish:** Deferred. README gallery cards and example images are
   intentionally omitted from this PR.

## Recommended Next Work

1. Define whether the private ADMM contact stream should become a public contact
   streaming API.
2. Keep reducing custom solver requirements by improving default fallbacks, but
   preserve explicit hooks for private solver state.
3. Add tuning guidance and benchmark problems for proxy virtual inertia and ADMM
   penalty/proximal parameters.
4. Extend ADMM joint support beyond `BALL`, `FIXED`, and `REVOLUTE`.
5. Revisit USD ownership and automatic coupled-solver construction after the
   Python API stabilizes.
