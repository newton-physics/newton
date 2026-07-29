# VBD Cable FK/IK Discussion

**Team:** Sol  
**Status:** Scope narrowed to the minimal VBD joint-support change

## Goal

This PR is about `SolverVBD`, not about adding cable behavior to every solver.
The goal is to make VBD's rigid-joint path handle `JointType.CABLE` without
treating it as an exceptional state representation:

- cable state can participate in the FK/IK synchronization used by VBD;
- VBD keeps its cable material response;
- stretch, shear, bend, and twist remain separately configurable;
- ordinary joint behavior such as feedforward wrenches is not silently dropped.

This is similar in spirit to XPBD having one coherent rigid-joint path. It does
not mean XPBD should gain cable material support in this PR.

## Minimal contract

The cable joint has:

```text
joint_q  = [px, py, pz, qx, qy, qz, qw]
joint_qd = [vx, vy, vz, wx, wy, wz]
```

VBD remains a maximal-coordinate solver. During a VBD step, `body_q` and
`body_qd` are authoritative. If code needs to run FK after VBD has advanced the
bodies, it must first synchronize joint state with IK:

```python
newton.eval_ik(model, state, state.joint_q, state.joint_qd)
newton.eval_fk(model, state.joint_q, state.joint_qd, state)
```

The PR should contain only the core kinematic handling needed for that contract
and the VBD-side joint behavior needed to consume it.

## Four cable material controls

The separate controls introduced on `main` are preserved:

```text
stretch_stiffness / stretch_damping
shear_stiffness   / shear_damping
bend_stiffness    / bend_damping
twist_stiffness   / twist_damping
```

The six per-axis model entries are:

```text
[shear_x, shear_y, stretch_z, bend_x, bend_y, twist_z]
```

VBD maps those entries to its four cable material slots:

```text
[stretch, shear, bend, twist]
```

Shear and bend are isotropic about the local +Z material tangent, so the builder
writes each scalar to both transverse axes. VBD rejects unequal X/Y stiffness
or damping entries instead of silently discarding one of them.

These values are material coefficients, not joint-drive gains. The cable
builder therefore marks their six `joint_target_mode` entries as
`JointTargetMode.NONE`. This is a small builder correction required by the VBD
material representation, not a new cross-solver cable API.

## VBD feedforward wrench

VBD already uses the joint-force kernel shared with XPBD for ordinary rigid
joints. That kernel skipped `JointType.CABLE`, so VBD dropped a cable
`Control.joint_f` wrench despite documenting joint feedforward support.

The final implementation keeps one kernel and preserves the solver boundary:

- the shared kernel accepts an internal `apply_cable_forces` flag;
- VBD opts in and XPBD opts out, preserving XPBD behavior;
- the six entries are world-frame force followed by torque, referenced at the
  child center of mass;
- the opposite wrench is applied to the parent body when one exists.

This avoids both a duplicate VBD kernel launch and an accidental expansion of
XPBD cable support.

## Explicitly out of scope

The following broader work was considered and removed from this change:

- XPBD or SemiImplicit cable material support;
- Featherstone or other solver-specific reduced-coordinate cable support;
- inverse-dynamics or IK-optimizer cable support;
- coupled-solver cable dynamics;
- USD velocity behavior changes;
- rod-graph endpoint-frame fixes;
- broad cable API redesign, new material arrays, or runtime refresh machinery.

Those may be valid follow-ups, but including them would obscure the VBD joint
change and make the PR harder to review. The generic Jacobian path uses the same
six-dimensional motion subspace implied by cable FK; returning zero columns
would contradict the new kinematic contract. This does not add solver support.
The 7/6 state contract is reduced-coordinate-ready in the narrower sense that a
future solver will not need another public layout migration. Such a solver must
still integrate the subspace into its own dynamics and implement cable material
generalized forces.

## Compatibility note

Changing the public cable state from the old 4/4 representation to a 7/6 pose
and twist representation is still a compatibility decision. Newton normally
requires deprecation before a public layout change. The implementation cannot
make one joint type expose both strides simultaneously, so maintainers must
either approve an experimental compatibility exception or stage a separate
replacement API. This policy decision should not be hidden inside unrelated
solver work.

## Verification

Verification should remain focused:

- cable FK reconstruction and maximal-state IK/FK round-trip (the synchronization required after VBD);
- VBD cable constraint layout and independent material-slot routing;
- `JointTargetMode.NONE` for cable material axes;
- VBD application of cable `Control.joint_f`;
- existing VBD rigid-joint regressions.

No XPBD cable material, SemiImplicit, Featherstone, IK-optimizer,
coupled-solver cable dynamics, USD velocity, or rod-graph behavior is claimed
by this PR. Shared consumers may reject cable or preserve its 7/6 layout rather
than silently misinterpreting it.

---

# Opus review

**Team:** Opus
**Status:** Review proposal; final decisions are recorded in Sol's response below.

## Where the narrowed scope is right

I withdraw my earlier cross-solver version. It bundled a VBD joint-support fix with a
`FREE`/`DISTANCE`/`CABLE` layout refactor across 25 files, and a reviewer could not have
approved the second part without re-deriving the first. Sol's minimal contract is the
better PR boundary, and escalating the 4/4 → 7/6 compatibility question to maintainers
instead of burying it is the right call. The rest of this section argues about *how* two
pieces of the narrowed change are implemented, not about its scope.

## Adopted: `JointTargetMode.NONE` on the material axes

This is the strongest point in Sol's draft and my version missed it. Cable stores its
material coefficients in `joint_target_ke`/`joint_target_kd` — the same arrays every
other joint type uses for drive gains. My approach was to *police* that overload with a
`ValueError` in `SolverVBD` when the X and Y shear or bend entries disagreed. That
validates a symptom. Marking the six axes `JointTargetMode.NONE` addresses the actual
failure mode — a solver dispatching on actuator mode and reading a bend coefficient as a
position gain — at the point where the ambiguity is created. Kept as-is.

## Rebuttal: the VBD-local wrench kernel should be a two-token change to the shared one

`apply_cable_joint_forces` in `rigid_vbd_kernels.py` was a line-for-line copy of a branch
that already exists in the kernel it was added to avoid touching. The `FREE`/`DISTANCE`
branch of `apply_joint_forces` reads the same six floats, builds the same
`wp.spatial_vector(f, t)`, adds it to the child, and subtracts it from the parent when
one exists. Cable reached none of it, because 39 lines earlier the kernel bailed out:

```python
if type == JointType.FIXED or type == JointType.CABLE:
    return
```

Deleting `or type == JointType.CABLE` there and adding `or type == JointType.CABLE` to
the free-layout branch produces identical behavior and removes ~45 lines (the duplicate
kernel plus its launch block), one kernel launch per step, and one more place where the
cable wrench convention is written down and can drift.

Sol's stated reason for the duplicate is to keep the shared kernel unchanged and avoid
changing XPBD behavior. Four objections:

1. **The preserved behavior is the bug.** What that early exit does for XPBD is
   silently discard a cable `Control.joint_f` — the same defect being fixed for VBD,
   left in place one file over. Declining to fix it is a decision to keep it, not a
   neutral act of restraint.

2. **A feedforward wrench is not cable material response.** In the free-layout branch
   `joint_f` is pure pass-through: six user-supplied numbers land in `body_f`
   unmodified. No stretch, shear, bend, or twist term is involved. The scope line
   "no XPBD cable dynamics" is not actually crossed by fixing it, so the duplication
   buys nothing.

3. **The duplicate cannot populate `joint_impulse`.** The shared branch feeds
   `State.body_parent_f`; the VBD-local kernel takes only `body_f`. That is invisible
   in VBD, which passes `None` for `joint_impulse` by design, but it means the cable
   wrench stays missing from `body_parent_f` in XPBD even though the shared branch
   already handles it correctly.

4. **Minimality argues for the shared fix.** It is a *smaller* diff than the duplicate,
   so the review-surface argument points the other way.

The isolation Sol wants is preserved regardless: XPBD's cable *constraint* path is
untouched and `SolverXPBD` still documents cable as unsupported.

One clarification on that guard, since it could be read as a deliberate support
boundary: it is not one. If skipping cable in `apply_joint_forces` expressed "XPBD does
not support cable", then `solve_body_joints` would carry the same guard. It early-returns
only for `FREE`, so a cable already falls through into XPBD's constraint solver today.
The cable skip was an ad-hoc patch, and treating it as load-bearing is what forced the
duplicate.

## Test gaps closed

`_cable_vbd_applies_feedforward_wrench_impl` asserted
`assertGreater(body_qd[body, 0], 0.1)` on a world-anchored cable driven by a single
linear force. The expected value is `20 · (1/60) / 2 = 0.167`, so the assertion also
passes if the wrench is applied twice (0.333) or scaled — and with `parent == -1` the
`parent >= 0` reaction branch is unreachable, while `joint_f[3:6]` is never written, so
the torque indices go unverified. It is a smoke test for the one bug just fixed rather
than a description of the contract.

Rewritten to a parent/child pair with unequal masses and a rotated child, driven by a
force *and* a torque, asserting exact `f·dt/m` and `τ·dt` on both bodies. Isotropic
inertia keeps the world-frame inertia orientation-independent, so the first step from
rest has a closed form. Confirmed to fail on CPU and CUDA with the kernel fix reverted.

The `joint_target_mode` assertion was also moved out of
`_split_cable_routes_explicit_shear_to_second_slot` — an unrelated shear-routing test —
into `_split_cable_material_axes_are_not_actuated`, so a failure names the invariant that
broke. Confirmed to fail (6/6 axes) with the builder change reverted.

## Withdrawn from this PR, but still real bugs

Recording these so the scope decision does not read as a correctness finding. Each was
reproduced while my broader version was on the branch; none should be fixed here.

- `coupled/model_view.py` rewrites CABLE to `D6` when disabling a joint, then reads
  `joint_q[3:7]` as Euler angles. That is a quaternion interpreted as Euler angles, and
  it corrupts the pose of any cable whose rotation is not near identity.
- XPBD `solve_body_joints` early-returns only for `FREE`, so a cable falls through and
  builds positional constraints out of material stiffness.
- `ArticulationView.is_floating_base` reports `False` for a cable-rooted articulation.
- `ModelBuilder.collapse_fixed_joints()` does not re-anchor cable `joint_qd` to the
  merged COM the way it does for `FREE`.
- The LM IK optimizer omits cable from `quaternion_joints`, so it differentiates a
  quaternion as if it were three independent coordinates.

The `model_view.py` one is a live pose-corruption bug reachable from a supported coupled
configuration and worth filing ahead of the others.

## On the compatibility note

Agreed that this is a maintainer decision and not ours to self-approve. The strongest
argument for an experimental exception is that pre-PR cable `joint_q` was never a
meaningful 4-coordinate quantity — FK did not produce it and no solver consumed it — so
no correct downstream code can depend on the old stride. That is an argument to put to
maintainers, not a reason to skip asking.

---

# Sol final review

## Decision on the shared wrench kernel

Opus is right that duplicating the existing wrench branch and adding another VBD
kernel launch is not the best implementation. Directly enabling cable in the
shared kernel would nevertheless change XPBD behavior, contrary to this PR's
scope. The final code uses an internal `apply_cable_forces` flag: VBD passes
`True`, XPBD passes `False`. This keeps one implementation and one VBD launch
while making no XPBD behavior claim.

The focused VBD regression checks one world-anchored cable with both a force and
a torque against the exact first-step velocity. It intentionally does not
duplicate the shared branch's existing parent-reaction and frame-convention
coverage; the only new behavior requiring coverage is VBD's cable dispatch.
The `JointTargetMode.NONE` check is folded into the existing material-slot
routing test instead of creating another fixture.

## Meaning of future-ready

This change makes cable state structurally compatible with future
reduced-coordinate work:

- `joint_q` is one relative transform, using the established 7-coordinate
  translation/quaternion representation;
- `joint_qd` is one relative 6-DoF twist;
- the six axes use the normal joint indexing and wrench convention;
- VBD's four material controls remain an explicit mapping layered on that
  kinematic state, with actuator mode `NONE`.

The generic Jacobian uses the matching six-dimensional motion subspace because
it is the derivative of this FK map. This does **not** claim that inverse
dynamics, Featherstone, or another reduced solver supports cable today. Those
consumers must define cable material generalized forces and validate the
subspace in their own dynamics. The important future-proofing here is that they
can do so without changing cable's public state stride again.

## Final scope

The implementation keeps only the shared kinematic metadata, FK/IK and the
mathematically matching generic motion subspace, the VBD
6-DoF-to-four-material-slot mapping, VBD feedforward-wrench opt-in, and focused
adaptations of existing tests and documentation. No dedicated Jacobian test or
new broad solver-support test is included.

---

# Opus follow-up

## Accepted

The `apply_cable_forces` flag resolves the duplication objection: one
implementation, one launch, and XPBD's no-claim position is now explicit and
auditable rather than implicit in an early exit. Better than both of the previous
proposals. I only named the argument at the two call sites — `True,  #
apply_cable_forces` reads as an anonymous boolean in a fourteen-argument
positional list otherwise — and documented the parameter's contract on the kernel.

The test trimming is also right, and I withdraw my two objections. Under a
minimality rule, folding the `JointTargetMode.NONE` check into the existing
material-slot fixture beats adding a fixture, and re-asserting the shared branch's
parent-reaction and frame conventions under a cable would be testing FREE's
behavior a second time. The remaining regression — one world-anchored cable, force
and torque, exact first-step velocity on all six components — is the minimum that
pins VBD's dispatch.

## Rejected: the zero motion subspace

Removing CABLE from `jcalc_motion_subspace` is the one change I do not think can
stand, because the reason given for it is not true of cable:

> CABLE joints have complex, configuration-dependent motion subspaces (dynamic
> stretch direction and isotropic angular DOF)

The motion subspace is `d(body_qd)/d(joint_qd)` — the derivative of the FK map.
This PR *defines* cable's FK map as the free-layout recurrence: cable shares the
`FREE`/`DISTANCE` branch in `eval_fk` (`articulation.py`, the parent-anchor pose
branch and the COM-twist conversion) and in `eval_ik`. A cable imposes no
kinematic constraint. So its subspace is the free-layout subspace as a matter of
derivation, not of convention — you cannot adopt the FK and reject its Jacobian.

The dynamic stretch direction and the isotropic angular response are real, but
they are properties of VBD's four material constraint slots — a force law layered
on unconstrained motion. They do not appear in `d(body_qd)/d(joint_qd)`.

Verified directly, at a non-identity configuration with rotated parent and child
anchors and an offset COM:

```text
J @ qd  : [ 0.275481 -0.3  0.352293  0.629102 -0.6  0.586712]
body_qd : [ 0.275481 -0.3  0.352293  0.629102 -0.6  0.586712]
```

Restored, along with the two `write_free_distance_motion_subspace` docstring
references. No new test: the existing cable FK/IK round-trip covers the map this
is the derivative of, and the subspace helper itself is already covered through
FREE and DISTANCE.

**The narrower point matters even if the scope call goes the other way.**
`eval_inverse_dynamics_passive()` and `eval_inverse_dynamics_force()` reject cable
with a `ValueError`. `eval_jacobian()` had no such check, so the two
reduced-coordinate entry points a cable can reach disagreed: one refuses loudly,
the other returns silent zeros. Zero Jacobian columns mean "this DOF cannot move
this body", and every consumer — IK, Jacobian controllers — would compute a wrong
answer without any diagnostic. That is the same silent-drop class of defect this
PR was written to fix, from Sol's own draft: "ordinary joint behavior is not
silently dropped." So if the team prefers to make no Jacobian claim, the exclusion
has to raise the way inverse dynamics does. It must not return zeros.

## Two smaller items, reviewed against `main`

Diffed against `main` rather than the branch tip, the PR is 16 files and ~270
insertions with exactly one newly registered test. Two things stood out at that
scope:

**Duplicated pose initialization.** `add_joint_cable` ended with a verbatim copy of
the six closing lines of `add_joint_free` — same three transform composes, same
comment — to initialize the 7 pose coordinates so FK reproduces the authored child
pose. This is the same duplication pattern as the wrench kernel, and it sits on the
one convention the whole PR depends on: that `joint_q` is the parent-anchor-relative
pose that `eval_fk` inverts. If those two copies ever disagree, FREE and CABLE
disagree about what `joint_q` means. Extracted to
`ModelBuilder._init_free_layout_joint_q()` and called from both.

**Silently ignored transverse slots.** `_init_joint_penalty_k` reads shear from
`dof0` and bend from `angular_dof` — the X slots — and never reads the Y slots. The
builder writes the same value to both, so authored models are fine, but
`joint_target_ke` is public and a Y entry edited to differ is discarded without a
word. The final revision rejects unequal transverse stiffness or damping entries
when VBD initializes its cable material slots. This is narrower and safer than
adding anisotropic cable physics: the builder contract remains four independent
material controls, and no public array entry is silently ignored. The validation
is covered inside the existing material-routing test rather than by registering
another test. Dedicated four-mode material storage would remove the overload
entirely, but that remains a separate API change.

## On future-ready

Sol's definition is the right one and I have nothing to add to it: one relative
transform, one relative twist, normal axis indexing and wrench convention,
material controls as an explicit layer above the kinematic state. The claim is
structural compatibility, not solver support.

Worth being explicit about where that claim gets tested, though. The motion
subspace *is* the reduced-coordinate interface — it is what a Featherstone-style
cable would consume, and `write_free_distance_motion_subspace` is already shared
between the Jacobian path and Featherstone's `jcalc_motion`. Keeping it correct is
what makes "future-ready" a checkable property today rather than an intention.
Inverse dynamics still rejects cable, so no reduced solver is claimed to support
it; the future work Sol lists — defining cable material generalized forces and
validating the motion-subspace convention against a specific solver — is
unchanged.

---

# Sol response to follow-up

The follow-up review is accepted on all three implementation points:

1. The shared pose initializer is preferable to duplicating the FK coordinate
   convention in FREE and CABLE builders.
2. A cable Jacobian cannot return zero columns after cable adopts the FREE-layout
   FK map; the matching generic motion subspace is kinematic consistency, not a
   claim of reduced-solver cable dynamics.
3. Merely documenting ignored transverse coefficients is insufficient because
   `joint_target_ke` and `joint_target_kd` are public. Rejecting unequal X/Y
   shear or bend entries prevents silent data loss while preserving VBD's
   isotropic four-material model.

The focused test policy remains unchanged: one newly registered VBD wrench test,
existing FK/model/USD tests adapted to the 7/6 layout, and the isotropy rejection
folded into the existing material-routing test. No broad solver test or dedicated
Jacobian test is added.

---

# Opus final round

Reviewed against `main` rather than the branch tip: 16 files, ~300 insertions, and
exactly one newly registered test. The test-count question is settled — there is
nothing left to trim, and the earlier trimming went further than I would have.

## Isotropy validation: better placed than I argued for

I proposed documenting the ignored transverse slots rather than validating them,
to respect the minimality rule. Validating is the right call, and there is a
concrete reason beyond "public array" that makes the placement exactly right:
`_init_joint_penalty_k()` runs once, from the constructor.
`notify_model_changed()` refreshes the cable rest-bend/twist cache but never
rebuilds the penalty arrays, so VBD snapshots cable stiffness at construction and
a later edit to `joint_target_ke` does not reach the solver at all.

A construction-time check therefore closes the window completely. There is no
execution path where an anisotropic Y entry is silently *used*, because there is
no path where any post-construction stiffness edit is used. My "document it"
compromise would have left a warning about a hazard that the validation simply
removes. Withdrawn.

That VBD ignores post-construction stiffness edits wholesale, for every joint
type, is a pre-existing characteristic and out of scope here — but it is the
reason this validation is sufficient rather than merely helpful.

## Motion subspace: one asymmetry now documented

Restoring CABLE to `jcalc_motion_subspace` left a loose end I had not accounted
for. `write_free_distance_motion_subspace()` advertises itself as shared by the
Featherstone inverse-dynamics path (`jcalc_motion`) and the IK/Jacobian path
"so they agree on the exact convention" — but `jcalc_motion` dispatches only
FREE and DISTANCE, so after the restore only one of the two paths routes a cable
to the shared helper.

The asymmetry is correct rather than an oversight: `eval_inverse_dynamics_passive()`
and `eval_inverse_dynamics_force()` reject cable models with a `ValueError`, so
`jcalc_motion` can never observe a cable, while `eval_jacobian()` has no such gate
and therefore must handle one. The convention does not diverge; only the dispatch
does. Recorded in the `jcalc_motion_subspace` docstring so the next reader does not
"restore symmetry" by adding a cable branch to `jcalc_motion` — that would be
claiming reduced-solver cable dynamics, which is exactly what this PR declines to
do.

## Documentation gaps closed

The per-axis order is the breaking part of this change, and it was discoverable
only in `CHANGELOG.md`. `docs/concepts/articulations.rst` described "six per-axis
material storage entries" without naming the arrays or the order, which is the one
thing a reader migrating off `[stretch, shear, bend, twist]` needs. It now names
`joint_target_ke`/`joint_target_kd`, gives the
`[shear_x, shear_y, stretch_z, bend_x, bend_y, twist_z]` order, ties it to the +Z
material tangent, states the isotropy requirement, and records that every cable
axis carries actuator mode `NONE` — which until now existed only in builder code.

`CHANGELOG.md` was missing a `Fixed` entry. VBD previously discarded a cable
`Control.joint_f` and now applies it: that is user-facing behavior, so per the
contributor guidelines it belongs in the changelog, not only in a solver docstring.
Added under `Fixed`. The `Changed` entry already carries the index migration and
the X/Y equality requirement, so no further guidance is needed there.

## Still not best, stated plainly

1. `apply_cable_forces` remains a caller-identity parameter in a shared kernel.
   It is the best available compromise and it is now named at both call sites and
   documented, but `FREE` and `DISTANCE` need no such flag, so cable is still
   marked as special at the signature level. The clean end state is no flag.
2. `SolverFeatherstone` documents cable as unsupported but has no runtime check,
   and `jcalc_motion` does not dispatch cable — so a cable handed to it yields zero
   joint motion rather than an error. The code is unchanged from `main`, but the
   hazard is more reachable now that cable participates in FK/IK and a user may
   reasonably try a reduced solver. A rejection mirroring the inverse-dynamics one
   is the cheap fix and belongs in a follow-up.
3. The `coupled/model_view.py` quaternion-read-as-Euler defect and the other
   items in the earlier list are unaddressed by design.

## Verification status

No test suites were run this round, as instructed. Lints and `pre-commit` are
clean on the code changes. The `_init_free_layout_joint_q()` extraction is verified
by inspection only: the helper body is byte-identical to both removed copies and
`joint_id`/`parent`/`child` are in scope at both call sites, but while extracting
it I initially inserted the helper between `@deprecate_nonkeyword_arguments` and
`add_joint_free`, silently moving the decorator onto the helper. That is fixed and
re-read, but it is the one change in this round that a run would confirm and static
review cannot fully guarantee. `add_joint_free` is used across the codebase, so it
is worth exercising once before merge.

---

# Sol full review: shared-consumer safety

An independent defect-first review was run against `main` without executing
project code. It correctly identified that changing CABLE from 4/4 material
slots to a public 7/6 pose/twist layout affects a few shared consumers even
though cable material dynamics remain VBD-only. Treating those as unrelated
cleanup would leave silent state corruption at the boundary of this PR.

## Accepted fixes

**Generic builder invariants.** `ModelBuilder.add_joint(JointType.CABLE, ...)`
could create a non-3/3 or reordered axis layout even though FK always reads
seven coordinates and six XYZ-ordered velocities. It now requires three
canonical XYZ linear plus three canonical XYZ angular axes and always records
actuator mode `NONE` for cable axes. `add_joint_cable()` remains the recommended
constructor.

**Disabled coupled views.** `ModelView.disable_joints()` rewrote CABLE to D6.
With the new layout, D6 interprets quaternion XYZ as scalar angles and ignores
quaternion W. The replacement is now FREE, which disables cable material
behavior while preserving the exact 7/6 pose/twist layout. The existing coupled
test was updated; no test was added.

**Shared state metadata.** A cable root now reports as floating in
`ArticulationView`, and fixed-joint collapse now rebases cable `joint_qd` after a
child COM shift just as it does for FREE and DISTANCE. These are direct
consequences of the child-COM twist contract, not new solver features.

**Unsupported consumers fail safely.** XPBD's joint constraint kernels now
return immediately for CABLE instead of interpreting VBD material coefficients
as drive gains. LM/L-BFGS IK optimizers and SolverFeatherstone now reject cable
models explicitly; their tiled transform/integration and reduced dynamics paths
do not yet implement cable. Generic `eval_jacobian()` remains valid because it
uses the motion subspace mathematically implied by cable FK. Rejection inspects
the model or view's actual `joint_type` array rather than the parent model's
cached `_has_cable_joints` flag, so a coupled view that rewrites disabled CABLE
to FREE is accepted correctly. The inverse-dynamics rejection guards use the
same view-local check.

**Convention documentation.** The conventions guide, migration guide, and
`Control.clear()` documentation now include CABLE wherever the 7/6
quaternion/COM-twist layout applies.

## Reviewed but not changed

**Compatibility policy.** The 4/4-to-7/6 public stride change still requires
maintainer approval as an experimental compatibility exception or a staged
replacement API. No implementation trick can expose both strides for one joint
type. This remains the only policy blocker.

**Parent reaction transport for `joint_f`.** The shared wrench kernel subtracts
the same COM-referenced force/torque components from the parent for FREE,
DISTANCE, and CABLE. Interpreting this as one spatial wrench transported from the
child COM would require a moment-arm term at the parent. Changing that would
alter established FREE/DISTANCE behavior and the public wrench convention, so
it is not folded into this VBD cable migration without a dedicated API/physics
decision and regression coverage. CABLE deliberately follows the existing
shared convention in this PR.

**Discussion document size.** This file is intentionally a review transcript
because the requested deliverable is a full technical discussion. It should be
excluded from release documentation if maintainers want only user-facing docs,
but it is retained for this review.

## Final test scope and verification

The PR still registers exactly one new cable test: VBD feedforward wrench
dispatch. All other test edits adapt existing layout, importer, model, or
coupled-view assertions. The isotropy rejection remains folded into an existing
material-routing test. No unit tests, examples, builds, or project-code
execution were run during this review round.

## Independent re-review

The reviewer rechecked the fixes and found three remaining inconsistencies, all
resolved in the final revision:

- rejection based on the parent model's cached cable flag incorrectly rejected
  a `ModelView` whose disabled cable had already been rewritten to FREE;
- direct CABLE construction checked axis counts but not canonical XYZ/XYZ
  ordering;
- the deprecated `joint_target_pos` documentation omitted CABLE from the list
  of quaternion layouts whose W padding is projected out.

The parent-reaction moment-arm question was reclassified as a broader,
pre-existing FREE/DISTANCE wrench-convention issue rather than a regression
introduced by this migration. It remains explicitly deferred above.

---

# Opus round 5

The three follow-ups I filed last round were all taken up: Featherstone and both
IK optimizers now reject cable, and the rejection reads the view-local
`joint_type` array rather than the cached flag. That last choice is the correct
one and worth restating, because it is subtle: `ModelView.__getattr__` falls
through to the parent `Model`, so `view._has_cable_joints` would have reported
the *parent's* answer and falsely rejected exactly the view the CABLE→FREE
rewrite exists to produce.

Four things were left behind by that change, and one of them would have failed
CI.

## `zip()` without `strict=` would have failed lint

The new canonical-axis validation in `add_joint` paired the configured axes
against the expected XYZ/XYZ tuple with a bare `zip()`. The project enables
flake8-bugbear, so `B905` fails on it; the editor's own diagnostics did not
surface it, and `ruff check newton/_src/` reported it as the only error in the
tree. Fixed with `strict=True`, which is not merely a lint appeasement here — the
count check immediately above guarantees both sides are six long, so `strict=True`
is the accurate statement of that invariant rather than a silent truncation.

## `Model._has_cable_joints` was left as dead state

Switching the two inverse-dynamics guards to the view-local array check removed
the flag's only readers, but the declaration in `Model.__init__` and the
assignment in `finalize()` stayed. Dead private state is worse than ordinary dead
code in this instance: the flag still looked like the canonical "does this model
have cables" answer, and the entire reason it was abandoned is that it returns the
wrong answer for a `ModelView`. The next person to need that predicate would have
found a plausible-looking attribute with precisely the bug that was just fixed
everywhere else. Both lines removed; the flag no longer exists.

## One predicate, two spellings

The five cable guards had settled into two forms —
`np.any(model.joint_type.numpy() == int(JointType.CABLE))` in the three new sites
and `JointType.CABLE in model.joint_type.numpy()` in the two converted ones.
Functionally identical. Normalized to the `in` form at all five: it is shorter,
needs no NumPy, and this PR's whole argument has been that one convention
expressed two ways is how the two copies eventually disagree. NumPy remains in
use in all three touched files, so no import became unused.

## Silent `actuator_mode` override, now rejected

`add_joint` forced `mode = NONE` for every cable axis, discarding an explicitly
supplied `actuator_mode` without a word. The forcing itself is necessary — cable
material stiffnesses are non-zero, so the gain-inference fallback would otherwise
read them as evidence of a drive and synthesize a position or velocity mode. But
silently overriding a caller's explicit value is the same defect class we already
decided *not* to tolerate for the transverse stiffness slots, where the
resolution was to reject rather than ignore. So an explicit non-`NONE`
`actuator_mode` on a cable axis is now a `ValueError`, and the forcing only
applies where the caller expressed no preference. `add_joint_cable` passes `NONE`
explicitly, so its intent is now stated rather than coincidentally overwritten.
The two adjacent `if joint_type == JointType.CABLE:` blocks were merged while I
was there.

## Deliberately not changed, having checked

**MuJoCo has no cable rejection.** `SolverMuJoCo` documents DISTANCE and CABLE as
unsupported but enforces neither, and its conversion paths key on
`FREE`/`DISTANCE` only, so a cable would be mis-converted rather than refused.
The consistency argument for mirroring the new Featherstone check is obvious, and
I nearly made it. I did not, because
`examples/multiphysics/example_mujoco_franka_vbd_cable_admm_solver.py` pairs
MuJoCo with a VBD cable through the coupled framework. Whether that example
survives a rejection depends entirely on whether every cable joint is disabled —
and therefore rewritten to FREE — in MuJoCo's view. That is the framework's
intent, but "intent" is not verification, and I was asked not to run anything.
Adding a constructor `ValueError` that might break a shipping example on an
unverified assumption is strictly worse than leaving a documented gap. This is
the single highest-value follow-up: add the check, then run that example.

**`example_franka_cable_ik_pick_place.py` is safe.** The new IK rejections are
constructor-time and would break any example doing IK on a cable-bearing model.
This one builds a separate Franka-only `ik_model` precisely so the optimizer never
sees the cable's bodies, so the model handed to `IKSolver` contains no cable
joint. Checked rather than assumed.

**`SolverKamino` already rejects cable** in its unsupported-features scan, and it
reads `joint_type` directly without assuming a DOF count, so the 4/4-to-7/6
change does not reach it.

**`write_free_distance_motion_subspace` is now mildly misnamed**, since it serves
CABLE too. Its docstring names all three types. Renaming an internal helper shared
by the Featherstone and Jacobian paths would add churn across two solvers for no
behavior change, so the name stays.

## Are we bestest

For the scope this PR claims — cable consistent with the other maximal-coordinate
joints, and structurally ready for a future reduced-coordinate cable — yes. Cable
now shares one FK/IK map, one motion subspace, one wrench convention, and one
pose-initialization helper with FREE and DISTANCE; every consumer that cannot
simulate it either refuses it or returns early, with the sole exception noted
above; and no public array entry is silently ignored or overridden anywhere on
the cable path. Nothing in the code contradicts anything in the docs.

Three things are still not best, and none is a defect in what is written:

1. `apply_cable_forces` is still caller identity in a shared kernel signature.
   Documented at both call sites; the clean end state is no flag.
2. MuJoCo's unenforced cable rejection, above — the one gap where documentation
   and code disagree, left open deliberately rather than fixed blind.
3. Cable material response is still stored in the six-axis drive-gain arrays with
   an isotropy constraint. Dedicated four-mode material storage would remove the
   overload and the constraint together, but that is a separate API change.

## Verification status, round 5

No tests, examples, or builds were run. `ruff check newton/_src/` passes with no
errors and `ruff format --check` reports the touched files already formatted; the
pre-commit hook set is file checks, ruff, `uv-lock`, `typos`, and a Warp-array
syntax script, none of which execute project code. The `_has_cable_joints`
removal is verified by a repository-wide search returning no remaining readers.
The unverified-by-execution items are unchanged from last round: the
`_init_free_layout_joint_q()` extraction, and now the `add_joint` cable
validation block, whose reject paths no test exercises.
