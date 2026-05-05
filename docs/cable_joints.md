# Cable Joints

Newton's XPBD tendon solver implements the routed-cable baseline from
Müller et al. "Cable Joints" (SCA 2018), with a first finite-slip capstan
extension for rolling links.  The known-good no-friction baseline is commit
`21ca37106` (`Document routed cable slip plan`); compare against that commit if
finite-slip work regresses the routed-cable examples.

## Current Scope

Implemented:

- Rolling pulley links with the full angular Jacobian in the stretch row.
- Pinhole links that transfer rest length between their two adjacent spans.
- Fixed attachment links.
- Auto-computed initial rest lengths.
- Dynamic pulley bodies with revolute joints.
- Finite capstan behavior for rolling links through `mu`: zero friction slips,
  finite friction reduces slip, and high friction recovers the no-slip
  baseline in the current acceptance tests.
- Dynamic and kinematic pulleys use the same solver path; the difference comes
  from body inverse mass and inverse inertia.

Not implemented in the current baseline:

- Explicit scalar pulley angle state.
- Dynamic rerouting merge/split or non-circular pulley profiles from the
  reference implementation.

## Link Types

| Type | Behavior |
|------|----------|
| `ROLLING` | Cable wraps around a circular body. Tangent points are recomputed from current geometry, stretch constraints use the full linear and angular Jacobian at those tangent points, and `mu` controls finite capstan slip. |
| `ATTACHMENT` | Cable endpoint fixed to a body-local point. |
| `PINHOLE` | Zero-radius waypoint. Taut excess transfers between the two adjacent spans, preserving their rest-length sum. |

## Solver Pipeline

The XPBD tendon path currently has two kernel entry points:

1. `update_tendon_attachments`
   - Runs once per tendon per XPBD iteration.
   - Recomputes rolling tangent points.
   - Applies the paper's `surfaceDist(old, new)` rest-length transfer.
   - Applies pinhole rest transfer between adjacent spans.
   - Applies rolling capstan rest transfer by projecting adjacent tension
     estimates onto `T_tight / T_slack <= exp(mu * theta)`.

2. `solve_tendon_segments`
   - Runs once per segment per solver iteration.
   - Solves the unilateral stretch inequality:

     ```text
     C = |x_r - x_l| - rest <= 0
     ```

   - Uses the full angular Jacobian:

     ```text
     Jw_l = -cross(r_l, n)
     Jw_r =  cross(r_r, n)
     ```

   - Projects rolling angular correction continuously with `mu`; `mu = 0`
     removes pulley torque and high `mu` recovers the full angular baseline.

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
world-space point.

No separate pulley-angle state is tracked.  The only rolling state is the
body-local contact point stored per segment endpoint.

Pinhole links are frictionless slip points.  A pinhole preserves the sum of the
two adjacent rest lengths and transfers only taut excess from current span
geometry:

```text
excess_left  = max(length_left  - rest_left,  0)
excess_right = max(length_right - rest_right, 0)
rest_left  += excess_left - excess_right
rest_right -= excess_left - excess_right
```

The stretch row remains unilateral, so slack cable is allowed.  Example tests
still track geometric total cable length as straight spans plus rolling wrap
arcs; small slack/tension differences are expected, but unbounded growth or
loss is not.

Rolling capstan slip uses the same rest-length transfer mechanism, but clamps
the adjacent tension estimates to the capstan ratio set by `mu` and wrap angle.
This keeps slip/no-slip as a coefficient choice instead of a separate solver
mode.  Kinematic pulleys are ordinary bodies with zero inverse mass and zero
inverse inertia, so high-friction locking falls out of the mass matrix.

## Current Validation

The focused regression suite is `newton.tests.test_tendon_capstan`.  Despite
the historical filename, it validates both the routed-cable baseline and the
first finite-slip capstan criteria:

- Pinhole Atwood: the heavy side descends and the light side rises through a
  pinhole.
- Dynamic rolling pulley Atwood: the heavy side descends, the light side rises,
  and the pulley rotates from the angular Jacobian.
- Dynamic capstan `mu` sweep: zero friction does not rotate the pulley, mid
  friction rotates in the cable direction, and high friction approaches
  no-slip rim/cable agreement.
- Kinematic capstan `mu` sweep: zero friction slips freely, mid friction slips
  less, and high friction locks cable motion.
- Motorized rolling pulley: a driven dynamic pulley produces slider motion
  through the no-slip cable path.
- Motorized delay regression: a driven pulley must move the cable during the
  initial rotation window.

Latest run:

```bash
uv run --extra examples python -m unittest newton.tests.test_tendon_capstan
```

Result: focused tendon capstan/equilibrium run passed 22 tests on CPU and CUDA.

Additional example regressions exercise the larger routed-cable scenes:

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

1. Keep the `21ca37106` no-friction baseline available for comparisons.
2. Harden each finite-slip case with a small, isolated test before tuning
   larger examples.
3. Re-render and promote complex examples only after their motion has a test
   gate that catches direction, delayed coupling, and pulley sign failures.
