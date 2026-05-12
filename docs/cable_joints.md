# Cable Joints

Newton's XPBD tendon solver implements the routed-cable baseline from
Müller et al. "Cable Joints" (SCA 2018), with a finite-slip capstan extension
for rolling links.  The known-good pre-split finite capstan baseline is commit
`1a87f7a2a` (`Add finite capstan routed cable baseline`); compare against that
commit if the clean split regresses the routed-cable examples.

For the mathematical formulation behind the current split stretch/slip solver,
see [A Split Stretch/Slip Formulation for Routed Cable Joints](cable_joints_formulation.md).

## Current Scope

Implemented:

- Rolling pulley links with per-iteration tangent updates and no-slip surface
  transfer.
- Pinhole links that transfer rest length between their two adjacent spans
  subject to a local capstan ratio from their bend angle and `mu`.
- Fixed attachment links.
- Auto-computed initial rest lengths.
- Dynamic pulley bodies with revolute joints.
- Finite capstan behavior for rolling links through `mu`: zero friction slips,
  finite friction reduces slip, and high friction recovers the no-slip
  baseline in the current acceptance tests.
- Dynamic and kinematic pulleys use the same solver path; the difference comes
  from body inverse mass and inverse inertia.
- A clean XPBD split: segment stretch carries cable load, and rolling spin-axis
  torque is solved as a separate capstan friction row.

Not implemented in the current baseline:

- Explicit scalar pulley angle state.
- Dynamic rerouting merge/split or non-circular pulley profiles from the
  reference implementation.

## Link Types

| Type | Behavior |
|------|----------|
| `ROLLING` | Cable wraps around a circular body. Tangent points are recomputed from current geometry, route update applies `mu`-scaled surface-distance transfer, stretch constraints solve the free spans, and the separate rolling slip row applies capstan-limited spin-axis torque. |
| `ATTACHMENT` | Cable endpoint fixed to a body-local point. |
| `PINHOLE` | Zero-radius waypoint. Taut excess transfers between the two adjacent spans, preserving their rest-length sum while enforcing `T_tight / T_slack <= exp(mu * theta)` from the local bend angle. |

## Solver Pipeline

The XPBD tendon path currently has three kernel entry points:

1. `update_tendon_attachments`
   - Runs once per tendon per XPBD iteration.
   - Recomputes rolling tangent points.
   - Applies the paper's `surfaceDist(old, new)` rest-length transfer, scaled
     continuously by the capstan coefficient for rolling links.
   - Applies pinhole rest transfer between adjacent spans using the same local
     capstan tension-ratio bound as rolling links.
   - Applies rolling capstan rest transfer by projecting adjacent tension
     estimates onto `T_tight / T_slack <= exp(mu * theta)`.

2. `solve_tendon_stretch`
   - Runs once per segment per solver iteration.
   - Solves the unilateral stretch inequality:

     ```text
     C = |x_r - x_l| - rest <= 0
     ```

   - Uses the translational/path Jacobian at the current attachment points.
     Non-spin angular terms still keep off-axis attachment motion consistent,
     but the rolling link spin-axis component is left for the friction row:

     ```text
     Jw_l = -cross(r_l, n)
     Jw_r =  cross(r_r, n)
     ```

   - Stores the stretch increment as a lagged load estimate for friction.

3. `solve_tendon_slip`
   - Runs once per tendon per solver iteration.
   - Estimates adjacent taut tensions and wrap angle at each rolling link.
   - Applies the spin-axis pulley torque allowed by the same capstan cone.
   - `mu = 0` removes rolling transfer and pulley torque; high `mu` recovers
     the no-slip rolling limit without a separate no-slip row.

## Formulation Notes

Each segment stores a mutable free-span rest length.  Rolling links transfer
rest length with the original paper update:

```text
rest += surfaceDist(old_left,  new_left)
rest -= surfaceDist(old_right, new_right)
```

Attachment points are stored in body-local coordinates.  At the start of a
substep, the old local contacts are transformed by the current body pose, new
tangents are computed, and the signed surface distance between those two points
updates the segment rest length.  During XPBD iterations, the stretch row
transforms the same local contacts by the current body pose every iteration.
This is what makes pulley rotation and cable motion couple immediately; the
contact point is fixed on the body during the solve rather than being a stale
world-space point.  Rolling surface transfer happens before the stretch solve
inside each XPBD iteration, matching Algorithm 1's ordering from the paper.

No separate pulley-angle state is tracked.  The only rolling state is the
body-local contact point stored per segment endpoint.

Pinhole links are zero-radius slip points attached to a body-local position. A
pinhole preserves the sum of the two adjacent rest lengths and transfers only
taut excess from current span geometry. The transfer is clamped by the local
capstan ratio from the angle between incoming and outgoing cable directions:

```text
excess_left  = max(length_left  - rest_left,  0)
excess_right = max(length_right - rest_right, 0)
T_left  = excess_left  / compliance_left
T_right = excess_right / compliance_right
T_tight / T_slack <= exp(mu * theta)
```

`mu = 0` recovers a frictionless pinhole; finite `mu` reduces rest-length
transfer; high `mu` locks the cable against a kinematic pinhole except for
ordinary segment compliance.

The stretch row remains unilateral, so slack cable is allowed.  Example tests
still track geometric total cable length as straight spans plus rolling wrap
arcs; small slack/tension differences are expected, but unbounded growth or
loss is not.

Rolling capstan slip uses the same rest-length transfer mechanism, but clamps
the adjacent tension estimates to the capstan ratio set by `mu` and wrap angle.
The stretch and slip solves are separate kernels, but slip/no-slip remains a
coefficient choice instead of a solver mode.  Kinematic pulleys are ordinary
bodies with zero inverse mass and zero inverse inertia, so high-friction
locking falls out of the mass matrix.

## Current Validation

The focused regression suite is `newton.tests.test_tendon_capstan`.  Despite
the historical filename, it validates both the routed-cable baseline and the
first finite-slip capstan criteria:

- Pinhole Atwood: the heavy side descends and the light side rises through a
  pinhole.
- Frictional pinhole `mu` sweep: zero friction slips freely, mid friction
  reduces through-pinhole transfer, and high friction locks more than the mid
  case.
- Dynamic rolling pulley Atwood: the heavy side descends, the light side rises,
  and the pulley rotates from the capstan-limited rolling row.
- Dynamic capstan `mu` sweep: zero friction does not rotate the pulley, mid
  friction rotates in the cable direction, and high friction approaches
  no-slip rim/cable agreement.
- Rendered dynamic capstan regression: the authored mid-friction example
  remains visually distinct from the high-friction/no-slip case.
- Kinematic capstan `mu` sweep: zero friction slips freely, mid friction slips
  less, and high friction locks cable motion.
- Motorized rolling pulley: a driven dynamic pulley produces slider motion
  through the no-slip cable path.
- Motorized delay regression: a driven pulley must move the cable during the
  initial rotation window.

Latest run:

```bash
uv run --extra examples python -m unittest \
  newton.tests.test_tendon_capstan \
  newton.tests.test_tendon_equilibrium
```

Result: focused tendon capstan/equilibrium run passed 26 tests on CPU and CUDA.

The VBD routed-tendon path is covered by `newton.tests.test_tendon_vbd`:

- Single-span routed tendon stretch pulls a dynamic endpoint toward the
  anchored rest length.
- Pinhole, dynamic capstan, kinematic capstan, motorized rolling pulley,
  high-inertia no-slip, exhausted-span saturation, and equal-weight Atwood
  regressions mirror the XPBD physical assertions.
- Dynamic capstan VBD also tracks the pre-contact trajectory prefix so mid
  friction cannot overtake high friction, high friction cannot reverse
  direction, and the light payloads cannot crest over the pulleys.
- Compound-pulley VBD now runs the same 220-frame balanced-scene assertions as
  the render path and hard-fails while the current VBD route is visibly
  unstable.
- VBD also reuses each larger routed-tendon example's `test_post_step()` and
  `test_final()` hooks on CUDA for the pulley, pinhole, rolling pulley,
  equilibrium, cable machine, 3D routing, and capstan scenes.  The full
  600-frame XY-table VBD run is not used as a routine unit test because it
  timed out after six minutes; the CUDA prefix regression now calls the same
  example hooks.
- Gear-pulley VBD direction is currently a known expected failure: the free
  counterweight rises instead of falling.  The regression is in place so this
  cannot be presented as a VBD pass again before the solver behavior is fixed.
- VBD uses the shared routed-tendon stretch/slip projection rows inside the
  rigid iteration; rigid joints/contacts remain on the VBD path.
- Rolling-pulley VBD stiffness regression keeps the dynamic pulley's revolute
  anchor drift below 5 mm with the current routed-tendon VBD tuning.

Latest VBD run:

```bash
uv run --extra dev --extra examples python -m unittest \
  newton.tests.test_tendon_vbd.TestTendonVBD \
  -k example_assertions -q
uv run --extra dev --extra examples python -m unittest \
  newton.tests.test_tendon_vbd.TestTendonVBD \
  -k vbd_compound_pulley_stays_balanced_cuda -q
uv run --extra dev --extra examples python -m unittest \
  newton.tests.test_tendon_vbd.TestTendonVBD \
  -k vbd_xy_table_tracks_reference_prefix_cuda -q
```

Current result: VBD no longer passes the routed-tendon example assertions just
because the lower-level unit kernels pass.  The compound-pulley balanced-scene
regression is a hard failure on CPU and CUDA until that instability is fixed;
the CUDA run reports roughly one meter of drift in a balanced scene that should
stay within 4 cm.  The grouped CUDA example-hook run also hard-fails the
equilibrium example (`0.0604 m` drift against a `0.05 m` limit) and kinematic
capstan smooth-motion check (`0.101 m` and `0.071 m` steps against a `0.035 m`
limit).  The XY-table CUDA prefix now hard-fails the example `test_final()`
hook with `0.00752` RMS tracking error against the `0.006` example limit.  The
gear-pulley direction/mechanical-advantage regression remains tracked as a
CUDA expected failure.

Additional XPBD example regressions exercise the larger routed-cable scenes:

```bash
uv run --extra examples python -m unittest \
  newton.tests.test_examples.TestCableExamples \
  -k capstan -q
uv run --extra examples python -m unittest \
  newton.tests.test_examples.TestCableExamples \
  -k xy_table -k compound -k rolling_pulley -q
uv run --extra examples python -m unittest \
  newton.tests.test_examples.TestCableExamples \
  -k 3d_routing -k cable_machine -k gear -q
```

Result: all selected example groups passed on CPU and CUDA.  These tests cover
dynamic and kinematic capstan scenes, the cross-base XY table, non-coplanar 3D
routing, cable machine, balanced compound pulley, five-pulley gear route, and
the simple rolling pulley.  They check direction of motion, pulley rotation,
cable length accounting, delayed-coupling regressions, and bounded motion.

## Repair Order

The finite-slip design criteria, capstan success criteria, implementation
order, and test-update policy are recorded in
[`cable_joints_slip_plan.md`](cable_joints_slip_plan.md).  Follow that document
when changing the friction path.

1. Keep the `1a87f7a2a` pre-split finite capstan baseline available for
   comparisons.
2. Harden each finite-slip case with a small, isolated test before tuning
   larger examples.
3. Re-render and promote complex examples only after their motion has a test
   gate that catches direction, delayed coupling, and pulley sign failures.
