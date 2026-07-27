OWNER: Team Sol

# SolverVBD previous-pose cleanup — design, rationale, and verification

Team: Sol · branch `fix/vbd-kinematic-state-handling` · comparison base
`main` @ `402d8eac`.

Status: **reviewed; final regression rerun pending.** This note records what the
cleanup removes, why the kinematic rework is the cleanest correct shape, the
coupling-only state that remains, the migration contract, and the alternatives
that were rejected.

## 1. What the cleanup is

`SolverVBD` used to maintain the rigid **previous-pose** history (`body_q_prev`)
*across* steps: the buffer persisted from one step to the next and was
selectively rebaselined on construction/reset through a per-world
`_rigid_pose_rebaseline_mask` arm-and-consume dance. That cross-step coupling
was the root of the reset machinery (arming, one-shot consumption, coupled
teleport-kick reconciliation, and the coupled-reset velocity-kick bug).

The cleanup removes the cross-step maintenance from the standalone path:

- **`body_q_prev` is now a pure per-step motion baseline.** Every `step()`
  re-establishes it from that step's input, so there is no persistent history to
  rebaseline and no reset arming needed for rigid pose. Reset becomes trivial —
  author the new pose (or `eval_fk`) and step.
- **Displacement-based friction/damping is preserved.** VBD contact friction and
  damping are driven by the `body_q_prev → body_q` displacement, not by an
  integrated velocity, so the per-step baseline must still yield the correct
  motion (see §2).
- **`body_q_step_end` was eliminated entirely.** An earlier iteration quarantined
  a "predicted then rewound" kinematic pose in a dedicated buffer; the
  backward-extrapolation design (§2) makes that buffer unnecessary.

## 2. Kinematic handling — the key idea

Kinematic bodies are prescribed by the caller, not solved. Their
motion-dependent contacts and constraints still need a `qd·dt` displacement,
but their **public pose must stay where the caller authored it** (conveyor-belt
semantics: a surface may carry velocity for friction without its geometry
translating).

The clean way to get both — verified in
`forward_step_rigid_bodies` — is to **extrapolate `body_q_prev` backward** from
the authored pose using the prescribed velocity, and leave the body at its
authored pose during the solve:

```python
if (body_flags[tid] & int(BodyFlags.KINEMATIC)) != 0:
    # Prescribed motion: hold the authored pose and place the previous pose one step
    # behind it, so contact friction and damping observe the qd*dt displacement.
    body_q_prev[tid] = _integrate_kinematic_body_pose(q_current, body_com[tid], body_qd[tid], -dt)
    body_inertia_q[tid] = q_current
    return
```

### Decision: move the baseline, not the authored pose

**Backtracking `body_q_prev` is the recommended design.** Newton's other rigid
solvers keep authored kinematic geometry at the caller's pose and consume the
prescribed velocity in their contact formulation. VBD instead derives motion
from a pose pair, so the equivalent representation is:

```text
body_q      = authored current geometry
body_q_prev = integrate(body_q, body_qd, -dt)
```

Here `body_q` is not redefined as a public end-of-step pose. It is only the
endpoint of the synthetic displacement used to recover the prescribed velocity.
The backward baseline therefore does not make VBD one step late; it expresses
velocity at the current authored geometry without retaining cross-step history.

Forward-predicting `body_q` and restoring it after the solve would produce the
same `qd·dt` displacement, but it is not equivalent to supplying velocity.
VBD reads `body_q` not only for friction and damping, but also for normal
separation, contact points, lever arms, body-particle geometry, and current
joint constraints. Forward prediction would advance all of that private
geometry by one step. It would be equivalent only if a predicted pose were
isolated to motion terms throughout the contact and joint formulations, which
would require a broader redesign and another buffer.

`body_q_prev` is also consumed by joint AVBD `C0` and cable Dahl preparation.
For consistently authored kinematic motion, the reconstructed parent baseline
matches the prior motion while the dynamic child starts from its prior solved
pose, allowing the prescribed anchor motion to enter the current solve. The
bounded exception is a pose-held, nonzero-velocity kinematic body used as a
joint parent: its synthetic baseline describes motion that the authored pose
never performs. Such surface-velocity bodies should be contact-only; jointed
kinematic bodies should update `body_q` consistently with `body_qd`.

Consequences:

- Friction/damping sees the displacement `q0 − (q0 − qd·dt) = qd·dt`. Correct.
- **Contacts resolve at the authored pose `q0`** — no one-step lead. This matches
  `main`'s historical kinematic contact placement, so the change does not shift
  moving-kinematic-contact scenes forward by a step.
- **Substeps become consistent.** The displacement scales with the `dt` of the
  call, so every substep of an entry sees `qd·dt_sub`. When a caller authored one
  pose per outer frame, `main` consumed the full-frame edit in the *first*
  substep over `dt_sub` (over-fast by the substep count), while the remaining
  substeps saw zero displacement.
- Finalization needs **no rewind and no quarantine buffer**: `update_body_velocity`
  for a kinematic body simply publishes the authored pose and the prescribed
  velocity. `state_in` and `state_out` stay consistent for fixed-binding graph
  replay.
- `_integrate_kinematic_body_pose` uses exact axis-angle rotation
  (`quat_from_axis_angle(w/|w|, |w|·dt)`), not the linearized quaternion step the
  dynamic integrator uses, so a fast-spinning kinematic body does not under-rotate
  (§5 explains why the shared integrator was not reused).

## 3. Coupling — cross-step state is intrinsic, and isolated

The coupled proxy path *does* still need cross-step state, and that is
unavoidable: a dynamic proxy is handed a new pose per outer step and its velocity
is reconstructed by differencing across coupling frames. The cleanup keeps this
state **fully isolated** from the standalone per-step scratch:

- `_coupling_body_q_accepted` — the accepted proxy pose carried across
  outer-step state distribution (read by the notify kernel's dynamic-proxy
  velocity conversion). This is the one genuinely new buffer: it takes over the
  cross-step role `body_q_prev` used to carry implicitly on `main`, where the kick
  measured against it and rewound to it.
- `_coupling_body_q_frame_start` — the outer-frame start pose, used for
  iteration-restart restore and as the harvest `body_q_prev`. This is `main`'s
  `_coupling_body_q_prev_snapshot` renamed: its old name meant "snapshot of
  `body_q_prev`", which stopped being true once that buffer became per-step. The
  notify kernel now writes it per row instead of the caller copying the whole array
  after the launch, so a pass-0 notify costs one launch and no copy where `main`
  cost a launch plus a full-array copy.

`coupling_notify_input_state_update` reworks the per-body handling:

- **Kinematic body (including a proxy):** keep the authored pose; derive the
  frame-start by integrating the prescribed velocity *backward* (`-dt`) — the
  same trick as §2, so harvest against `state_out.body_q` reproduces the solve's
  `qd·dt` displacement exactly.
- **Dynamic proxy:** continue from `_coupling_body_q_accepted`; convert the
  synchronized pose delta into a velocity kick and rewind the pose.
- **Non-proxy dynamic/static body:** frame-start is the current pose (no kick).
  `main` kicked *any* non-kinematic body whose pose moved. Gating on
  `BodyFlags.PROXY` is what makes the coupled path agree with the new standalone
  rule: a pose the caller authored is accepted, and only a pose authored by
  another solver as an *outcome* is converted to velocity so the body cannot
  teleport mid-frame.

Positive-`dt` coupling iteration restarts restore
`_coupling_body_q_accepted` from the outer-frame start and return without
relaunching the kernel: the coupler redistributes the interval's original input
state, then re-syncs proxy rows with a non-restart notify, which performs the
kick against the restored baseline. A zero-`dt` reset distribution deliberately
restores nothing: it must retain accepted dynamic-proxy poses until `reset()`
applies its world mask, otherwise unselected worlds would be rewound.

No new `CouplingInterface` lifecycle hooks are needed. `interface.py` and
`solver_coupled.py` remain unchanged from the comparison base.

## 4. Intentional decisions (not defects)

- **The `_coupling_body_q_accepted` write in `update_body_velocity` is
  unconditional on purpose.** It is fused into the existing finalization kernel,
  so captured CUDA graphs replay it without a host-gated coupling mode or an
  extra copy launch. The standalone cost is one additional transform write per
  body.
- **Three pose buffers is the minimum.** `body_q_prev` (per-step baseline) plus
  the two coupling buffers cover: per-step displacement, cross-frame dynamic-proxy
  velocity, and iteration-restart. Neither reduction works (§5).
- **`_rigid_pose_rebaseline_mask` is retained and correctly named** — it is now
  shared by the coupling notify kernel and the enabled-cable Dahl rebaseline; it
  is not cable-only, so a cable-specific rename would be wrong.
- **Coupling contact-force harvest uses `body_inv_mass_effective`** (not
  `model.body_inv_mass`). Once kinematic-ness is decoupled from mass, "does VBD
  dynamically integrate this body?" must be read from the effective array;
  `model.body_inv_mass` would misclassify a massive kinematic body as dynamic.

## 5. Alternatives considered and rejected

- **Forward-predict `body_q`, then restore the authored pose.** This gives the
  desired `qd·dt` difference, but `body_q` is VBD's current geometry and
  constraint pose, not a motion-only scratch value. The prediction would move
  contact normals, penetration, lever arms, body-particle geometry, and joint
  anchors during the solve. Isolating it to friction/damping would require
  splitting geometry from motion across the kernels and retaining an additional
  end-pose buffer. Backtracking only `body_q_prev` supplies the same prescribed
  motion without changing geometry.
- **Reuse `integrate_rigid_body()` instead of `_integrate_kinematic_body_pose()`.**
  Calling the shared integrator with zero inverse mass and inertia does reduce to
  pure twist integration (gravity is gated by `wp.nonzero(inv_mass)`, the torque
  term by `inv_inertia`), and it would remove the new `wp.func` from the diff.
  Rejected for two reasons: it advances the rotation with the linearized update
  `normalize(r + ½·dt·ω⊗r)`, which under-rotates badly once `|ω|·dt` is no longer
  small — and a kinematic body's spin rate is authored by the caller, not limited by
  a stability criterion — and the call would depend on argument values (zeros) rather
  than on a name for its meaning.
- **Keep two coupling buffers by reusing `body_q_prev` as the accepted proxy pose.**
  That is `main`'s arrangement, and it re-introduces exactly the cross-step
  ownership this change removes: `body_q_prev` would be a per-step baseline in the
  standalone path and a frame-persistent baseline in the coupled path.
- **Drop `_coupling_body_q_frame_start` and harvest against `state.body_q`.** After
  the notify kernel rewinds a dynamic proxy, `state.body_q` *is* the frame start —
  but only until the entry substeps, which integrate the proxy away from it, and
  never for a kinematic proxy (whose frame start is behind its authored pose).
  The dedicated frame-start buffer preserves that outer-frame baseline across
  entry substeps.
- **Relaunch the notify kernel on iteration restart, as `main` did.** With the
  coupler redistributing the interval's input state, the extra launch measures a
  zero delta on every row it can reach, so restoring the kick baseline is the whole
  job (§3).

## 6. Behavioral change and migration

This is not a no-op refactor — kinematic friction is re-architected. The
user-facing contract (CHANGELOG, `Changed`):

> Change experimental `SolverVBD` to rebuild rigid previous-pose baselines each
> step. Non-kinematic pose edits now reposition bodies without producing velocity
> or contact friction. For friction-driving prescribed motion, set
> `is_kinematic=True` and provide `State.body_qd`; `mass=0` alone no longer
> suffices.

One sentence per silent break — neither raises:

1. **Dynamic bodies too.** On `main` a pose edit was measured against the retained
   previous pose, so it injected a velocity *and* a friction displacement.
2. **The zero-mass idiom.** A `mass = 0` body is `DYNAMIC` unless
   `is_kinematic=True`; it still goes where its pose is authored, so the symptom is
   a conveyor that stopped gripping, not a body that stopped moving.

Deliberately *not* in the entry: the `collect_rigid_contact_forces()` contract
(`main` required a pre-`step()` snapshot of `body_q_prev`, since the buffer was
advanced afterwards, plus hand-patching rebaselined rows on a first or reset step;
both are gone). `SolverVBD` is documented as experimental — "public API and
behavior may change without prior notice" — so a niche helper's argument contract
belongs in its docstring, which now states the one-line rule. The entry is reserved
for what silently changes an existing scene's results.

All in-tree VBD examples that prescribe rigid motion now demonstrate the pattern;
a true conveyor (static geometry + surface velocity) is also expressible, which
`main` could not do. Keep that pose-held surface-velocity idiom contact-only:
when the body is a joint parent, the same synthetic displacement is interpreted
as prescribed anchor motion and will drive the child.

## 7. Implementation footprint against `main`

Three production files change, and every hunk is forced by the design:

| File | Hunks | Why the design forces it |
| --- | --- | --- |
| `rigid_vbd_kernels.py` | `_integrate_kinematic_body_pose`; `forward_step_rigid_bodies` takes `body_flags` instead of `pose_rebaseline_mask`; `update_body_velocity` takes `body_flags` and the coupling output | per-step baseline + kinematic branch |
| `vbd_coupling_kernels.py` | notify kernel rewritten around the two coupling buffers | the buffers replace `body_q_prev` |
| `solver_vbd.py` | buffer allocation, notify branches, harvest sources, launch arguments, docstrings | follows the two above |

The implementation keeps `main`'s surrounding control flow and documentation
where behavior is unchanged. New comments explain only the non-obvious
ownership rules: kinematic bodies hold their authored pose, non-proxy bodies are
accepted as-is, and dynamic proxies are rewound to VBD's accepted pose.

## 8. Verification

Compared against `main` @ `402d8eac` (branch `fix/vbd-kinematic-state-handling`,
which currently carries the work uncommitted). The Dahl update reads
`state_out.body_q` — correct because kinematic bodies stay at their authored pose
during the solve, so no separate end-of-step pose buffer is needed.

Recorded before the final repository-wide example migration (RTX PRO 5000
laptop; CPU + CUDA; not rerun during the requested static-only final review):

- `test_solver_vbd` and `test_coupled_solver`: module-scoped suites pass.
- `test_cable`: 160 tests pass.
- Focused coverage passes for zero-`dt` reset distribution, coupled frame-start
  baselines, per-step kinematic baselines, and fused rigid finalization.
- Ruff lint/format, Warp array-syntax checks, IDE diagnostics, and direct Warp
  code-generation smoke checks pass.

## 9. Verdict

The standalone path realizes the "no cross-step maintenance" goal cleanly. The
kinematic backward extrapolation is accurate, avoids a one-step geometry lead, and
makes substepped kinematic friction consistent for the first time. Coupling
cross-step state is intrinsic but isolated, reset-safe, and does not alter shared
coupling interfaces. The accepted coupling pose is fused into the existing
finalization kernel rather than maintained by a separate copy. The remaining diff
is the design's own footprint.

## 10. Independent review — Team Opus

Reviewer: Team Opus. Scope: §1–§9 re-derived against the code rather than against
this note. **Verdict: the design is correct and the rationale holds.** Two
additions follow; neither changes the implementation.

**The backward baseline is also load-bearing for joints.**
`body_q_prev` is not read only by contact friction and damping. It also feeds
`step_joint_C0_lambda`, the AVBD start-of-step residual launched immediately after
`forward_step_rigid_bodies`, and `compute_cable_dahl_parameters`. `C0` enters as
`C_stab = C - alpha * C0` with `rigid_avbd_alpha = 0.95` by default, so for a
jointed kinematic anchor the baseline choice decides whether prescribed motion
propagates at all: with consistent authored motion, backward extrapolation
reconstructs the parent's prior pose while the child begins from its prior solved
pose, so `C0` is near zero and the anchor's motion is resolved in full. Leaving
`body_q_prev` at the authored pose would let `C0` absorb that motion and
`C_stab ≈ 0.05·C` would leave the child lagging. This is what keeps
`example_cable_twist` working — its first rod body is flagged kinematic directly
through `builder.body_flags` and is the cable chain's joint parent — and it is a
second reason substepping improves on `main`.

**The conveyor idiom has one bounded hazard worth a line in §6.** A pose-held body
with nonzero `body_qd` that is *also* a joint parent yields `C0 ≈ −qd·dt` against
`C ≈ 0`, so `C_stab ≈ +0.95·qd·dt` and the solve pulls the jointed child by about
`qd·dt` every step, for motion that never happens. Contact-only conveyors are
unaffected, and kinematic bodies are root-only, so a conveyor can never be the
joint *child*; the exposure is exactly "conveyor used as a joint parent".

Independently confirmed: `main` also held kinematic bodies at the authored pose
(`rigid_vbd_kernels_base.py`), so §2's "no one-step lead" is right; §5's two
grounds for rejecting `integrate_rigid_body()` are both accurate in `solver.py`
(gravity gated by `wp.nonzero(inv_mass)`, linearized quaternion update); §6's
quoted entry matches `CHANGELOG.md`, and `is_kinematic` is the real builder
argument that sets `BodyFlags.KINEMATIC`; and §3's claim that `interface.py` and
`solver_coupled.py` remain unchanged holds against `main`. §8's recorded runs
were not re-executed during this review.

## 11. Example migration — authored velocity and out-of-band pose edits

Migrating `example_contacts_rj45_plug` produced one rule for any scene that drives a
kinematic body, and one hazard for scenes that edit dynamic poses between steps.

**Author `body_qd` as the COM twist between successive authored poses.** The baseline
is `_integrate_kinematic_body_pose(q, com, qd, -dt)`, which translates the COM by
`-v·dt` and rotates about it by `-ω·dt` with an exact axis-angle update (§2). So
authoring

```
v = (com(A_k) − com(A_{k−1})) / dt
ω = quat_velocity(rot(A_k), rot(A_{k−1}), dt)
```

inverts it exactly, and the solve's baseline *is* the previously authored pose — the
value `main` carried across the step boundary. Verified over pure translation, 0.05°
and 37° single-step rotations, and COMs offset from the body origin: worst position
error `5.96e-08` with rotation error identically zero, i.e. float32 rounding.
Transferring a driver's instantaneous velocity (`v_plug + ω × r`) is only the
local approximation and its position error scales as `O(ω²·dt²)`; the two are
indistinguishable while the driver does not rotate, which is why swapping them left
the rj45 trajectory bit-identical. `example_cable_twist` satisfies the rule for its
rotation (`angle = rate·t` about a fixed axis, so the analytic `ω` *is* the difference);
`example_cable_bundle_hysteresis` satisfies it except on the single step straddling
each triangle-wave apex; `example_cable_cross_slide_table` authors the exact derivative
of a ramped path and so deviates by `O(dt²)` through each ramp plus one `O(dt)` step
per corner. These are valid instantaneous velocities, but only differencing exactly
reconstructs the preceding authored pose; prefer it for new drivers.

For jointed kinematic bodies, perform that authoring at solver-substep cadence.
Holding one outer-frame pose while replaying a nonzero `body_qd` would synthesize the
same anchor displacement repeatedly. The migrated jointed examples instead provide a
pose and matching finite-difference velocity for every substep; pose-held nonzero
velocity remains a contact-only conveyor idiom.

One condition is easy to miss: because the baseline rotates about the **COM**, authoring
pure spin (`v = 0`, `ω ≠ 0`) is exact only when the COM lies on the rotation axis. It
does in the in-tree pure-spin drivers: cable bodies use COM-origin frames and pulleys
turn about their centers. A driver spun about an offset pivot must also author
`v = ω × (com_world − pivot)`.

**An out-of-band pose edit on a dynamic body is now silent, and rj45 depended on
it.** `_align_cable_orientations` re-aimed each rod capsule at its neighbour
*after* `step()` returned. The solver had already finalized `body_qd`, so the
kernel left every edited dynamic body with a pose and velocity describing
different motion. On `main`, `body_q_prev` still held the pre-edit pose; the next
solve therefore interpreted the correction as physical rotation and happened to
damp it. With a per-step baseline, the corrected pose becomes the next input
baseline and the stale velocity drives the body away from it again.

The correction was not small enough to ignore: over a scripted drag it averaged
0.197° per substep against 0.178° of solve rotation and cancelled 83% of that
rotation. The apparent stability on `main` therefore depended on an accidental
interaction between persistent history and an inconsistent state edit.

**Rejected: compensate with velocity or extra damping.** Folding the correction
into `body_qd` puts it into the next inertial prediction and creates a positive
feedback loop. Raising `bend_damping` suppresses the loop, but only tunes around
the same invalid post-solve edit. Neither makes the state transition coherent.

**Decision: remove the dynamic-pose correction.** The cable joints now own every
dynamic rod pose and velocity through stretch, shear, bend, and twist
constraints. `_align_cable_orientations`, its buffers, and its post-step launch
are removed, and the original bend damping is retained. If exact capsule
alignment is required later, it should be represented by an in-solver
constraint; a visual-only correction must use separate render transforms rather
than simulation `body_q`.

Removal is measured, not assumed. Published cable `|ω|` falls from 1.004 to
0.158 rad/s with no parameter change, and per-phase cable jitter now matches or beats
`main` everywhere: approach 23.3 µm against 32.6, hold 1.3 against 3.7, flick 27.2
against 29.3, second hold 1.3 against 5.7. The hold regimes — where any motion is
numerical rather than excited — are the quietest of every variant measured, against
7.3 µm for the corrector version, which trembled indefinitely because a tug-of-war
never settles. The geometry the corrector maintained degrades negligibly: over a
230-frame drag the frame-to-centerline misalignment is 0.87° mean and 3.0° max, bounded
rather than accumulating (first quarter 0.748°, last 0.893°), and the gap between
consecutive capsules is 0.096 mm mean and 0.26 mm max — 3% and 8% of the 3.25 mm cable
radius, so capsules still overlap far more than they separate. The cost is a livelier
cable while dragging (approach and flick jitter about 2.4x the corrector version, still
inside `main`'s envelope), which is the cable's own bending response no longer being
suppressed.

Generalizing: never edit a dynamic `body_q` after finalization to repair geometry.
Either prescribe a kinematic pose and matching velocity, or express the desired
dynamic motion through forces and constraints so pose and velocity are finalized
together.

## 12. Landing review — full diff

Scope: the full tracked diff against `main`. The solver and coupler state
transitions are coherent; no blocking implementation defect remains.

The review did find one migration gap: several in-tree VBD examples and one cable
test still drove zero-mass bodies by editing `body_q` alone. Those call sites now set
`BodyFlags.KINEMATIC` and author matching `body_qd`. Fixed prescribed endpoints are
also marked kinematic where the examples describe them that way, and jointed drivers
advance pose and velocity together per substep. This keeps the repository's own usage
consistent with the changelog contract instead of documenting a migration that its
examples violate.

The two dedicated tests considered earlier remain intentionally removed. Compact
standalone and coupled coverage pins the new baseline, kinematic velocity, capture
replay, reset, accepted-proxy behavior, and the distinction between a substep
`body_q_prev` and the retained outer-frame start used by harvest. A separate four-case
matrix would duplicate that state invariant.

The deadzone mirror likewise does not need a dedicated test for this change.
`body_q[tid] = pose` differs from a self-assignment only when the deadzone replaces
`pose` with `pose_prev`; remove that write together with the planned deadzone removal.
The durable kinematic path is covered end to end, including aliased input/output states.

No unit tests were rerun during this final review. Static diagnostics and
`git diff --check` were clean; §8 distinguishes the earlier recorded runs from this
static-only pass.

**Static review confirmations:**

- The `model.body_inv_mass` → `body_inv_mass_effective` swap in the two coupling
  launches is required rather than incidental: those kernels use `inv_mass > 0` as their
  "movable" test, and a body may now be `KINEMATIC` while retaining a nonzero model
  mass, which would attribute proxy forces to a body that cannot move. After the swap no
  `model.body_inv_mass` reference remains in `solver_vbd.py`, so every consumer agrees
  on one definition of movable — the same consolidation the flag change makes at the
  kernel level.
- The `iteration_restart` early return is not merely equivalent to `main`'s relaunch, it
  is *required*: `_coupling_body_q_frame_start` must stay fixed across restarts within
  one outer frame, and only the pass-0 launch establishes it. Relaunching could only
  re-derive the same value or corrupt it.
- Attribute-guard symmetry holds. `_coupling_has_rigid_avbd_state` is defined as exactly
  the allocation predicate (`not integrate_with_external_rigid_solver and body_count > 0`),
  and the one site guarding on `body_count > 0` alone sits in the `else` branch of
  `if self.integrate_with_external_rigid_solver`, so the new coupling buffers are never
  touched where they were not allocated.
- Ordering the kinematic branch *before* the reset-mask check in
  `_update_vbd_body_input_state_kernel` is what makes the coupled path agree with the
  standalone one: both now derive a kinematic baseline from `qd` even on a reset step,
  because there is no longer a first-step special case to defer to.
- The authored-velocity rewrites are analytically right where they claim to be. The
  cross-slide pulleys apply the product rule to `ramp(t)·path(t)` and the two drive
  angles differentiate consistently with their positions
  (`q_left = (target_x − target_y)/r ⇒ qd_left = (v_x − v_y)/r`); `example_cable_twist`
  advances its clock in a device kernel so a captured graph stays correct. Remaining
  prescribed examples use either analytic angular rates or finite differences between
  successive authored targets.
- `_rigid_pose_rebaseline_mask` remains an accurate name: rigid `body_q_prev` no longer
  uses it, but coupled proxy poses still use it to accept a fresh pose baseline after
  construction or reset, and cable history shares the same world selection.
- `update_body_velocity` now documents `body_q_prev` as read-only. That property lets
  post-step force collection use the same per-step baseline without another snapshot.

**Second pass — per-substep authoring.** The examples that formerly held one authored
pose across a frame's substeps while authoring a per-frame rate (`plectoneme`,
`dahl_hysteresis`, `bend_twist_analytic`) now command every substep from arrays indexed
by substep. This upgrades them from aggregate-correct to exact: the per-frame form summed
to the right displacement over a frame but attributed `1/N` of a single pose jump to each
substep, whereas the pose now genuinely advances each substep by the amount `body_qd`
claims. The frame seam is exact for the same reason — differencing substep 0 against
`command(t − sim_dt)` is differencing against the previous frame's last authored value
whenever `frame_dt = sim_substeps · sim_dt`.

Three details in that rework are subtle enough to be worth recording as verified.
`bend_twist_analytic` flattens its targets to `[substep · count + tid]` and carries
`previous_pos` as a *view* into that array; the next iteration writes a different slice,
so the differencing is genuinely against the prior substep rather than self-referential,
and the persistent `_kinematic_pos_prev_np[:] = previous_pos` is a copy. Its positions
are authored through `com_from_node` into a rod built with `body_frame_origin="com"`, so
the differenced quantity is the COM displacement the baseline expects. `dahl_hysteresis`
needed the new `if t <= 0.0: return 0.0` guard in `_drive_at_time`, because evaluating
the command at `sim_time − sim_dt` reaches negative time on the first frame and would
otherwise author a spurious opening rate.

The gripper test is the one friction-critical driver and its authored velocity is exact.
Its position reduces to `center + rot · (0, sgn·offset_mag, pull)`, so rotating
`(0, sgn·offset_speed, pull_speed)` by the same constant `rot` is the true derivative,
and both speeds are piecewise constant over piecewise-linear ramps — deviating only on
the single substep straddling each breakpoint, the same bounded case as
`example_cable_bundle_hysteresis`.

Two nits remain, neither affecting simulation. `dahl_hysteresis` now records
`twist_target` from the last substep while `force_now` is still the frame-start value, so
the two recorded series in a hysteresis plot are skewed by about one frame; reading
`_twist_angles_np[0]` restores exact pairing. And the sinusoidal cable-test drivers
author analytic derivatives rather than differences, which is the `O(dt²)` deviation this
note advises new drivers against — harmless under test tolerances, but it is the pattern
the guidance discourages.

**Final recommendation:** keep the implementation as structured. Further buffer
folding would mix per-step and coupling-frame ownership again, while restoring the
removed test matrices would add volume without a distinct invariant. The remaining
pre-landing action is verification, not redesign: rerun the affected solver, coupled,
and cable tests after the final example migrations. Give
`_split_cable_kinematic_arc_yields_uniform_curvature` particular attention — the
per-substep rework changed its *driving profile*, not just its authored velocity, since
the tip now interpolates across substeps where it was previously held, so its
curvature-uniformity and residual angular-speed thresholds are being asserted against a
different trajectory than when they were tuned.
