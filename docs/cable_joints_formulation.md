<!-- SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# A Split Stretch/Slip Formulation for Routed Cable Joints

## Abstract

This note describes the reduced-order routed-cable formulation used by
Newton's tendon solver for massless cables over attachments, pinholes, and
rolling pulleys. The core idea is to separate cable behavior into two coupled
but distinct constraint families:

1. a unilateral stretch row for each free span, which carries normal cable
   tension but intentionally does not impose no-slip pulley spin, and
2. a slip/friction projection at each routing link, which conserves total cable
   rest length while enforcing the Euler-Eytelwein capstan tension ratio.

This split preserves the useful properties of Cable Joints style massless
routing while extending the model to finite capstan friction and frictional
pinholes. The resulting stateful rest-length distribution reproduces capstan
hysteresis under load/unload cycles without requiring separate static and
kinetic friction coefficients.

## 1. Motivation

Many cable-driven mechanisms need behavior that is richer than a scalar tendon
length constraint but much cheaper than a discretized rope with contact. Typical
examples include block-and-tackle systems, routed robot tendons, belts over
driven pulleys, and cables threaded through small guides. A useful reduced model
should support:

- ordered routes through rigid bodies,
- finite-radius rolling pulleys,
- zero-radius pinholes,
- dynamic and kinematic bodies,
- slack spans,
- frictional slip and no-slip limits,
- force hysteresis from static capstan friction, and
- solver integration that is compatible with XPBD and VBD-style rigid solves.

The formulation here represents a tendon as an ordered sequence of links on
rigid bodies. Adjacent links define free spans. Each free span stores its own
rest length. Slip over a pulley or through a pinhole is not represented by rope
particles sliding over a contact mesh; it is represented by transferring rest
length between neighboring spans while conserving the sum.

This is the key modeling choice: rest-length distribution is the internal state
that records which side of a guide currently owns cable material.

## 2. Tendon State

Consider a tendon with links indexed by `i = 0 ... n - 1`. Link `i` is attached
to a rigid body `b_i` and has a body-local offset `o_i`. Consecutive links
define free spans indexed by `s = 0 ... n - 2`.

Each link has one of three types:

| Type | Meaning |
| --- | --- |
| `ATTACHMENT` | Cable is fixed to a body-local point. |
| `PINHOLE` | Cable passes through a zero-radius body-local guide point. |
| `ROLLING` | Cable contacts a circular pulley with radius `R_i` and axis `a_i`. |

Each free span stores:

- `r_s`: mutable free-span rest length,
- `x_s^L`: left endpoint in world space,
- `x_s^R`: right endpoint in world space,
- `xbar_s^L`, `xbar_s^R`: the same endpoints in the corresponding body-local
  frames,
- `c_s`: compliance,
- `d_s`: damping, and
- `lambda_s`: XPBD stretch multiplier.

The current geometric span length is

```text
L_s = ||x_s^R - x_s^L||.
```

The taut extension and tension estimate used by the friction projection are

```text
e_s = max(L_s - r_s, 0)
T_s = e_s / max(c_s, epsilon).
```

Slack spans therefore carry no tension in this reduced model.

## 3. Route Update

The route update recomputes cable endpoint positions before each stretch solve.
Attachment and pinhole links are simple body-local points:

```text
x_i = X_i o_i,
```

where `X_i` is the current transform of body `b_i`.

Rolling links are finite-radius circles in a body-local cable plane. For a span
from an external point `p` to a rolling link with center `c`, radius `R`, plane
normal `a`, and orientation `sigma in {-1, +1}`, the route update computes the
tangent point on the circle from `p`. For spans between two rolling links, the
two tangent points are iterated to consistency.

The important point is that tangent points are determined by the current route
geometry: centers, radii, axes, orientations, and external points. A pure spin
of a circular pulley about its own axis does not change the geometric tangent
point. This observation is what motivates removing the spin-axis angular
component from the stretch row.

### Rolling Surface Transfer

When a rolling contact point moves on a pulley surface, the original Cable
Joints update transfers rest length using signed arc distance:

```text
Delta l = surfaceDist(x_old, x_new).
```

In the current formulation, this transfer is scaled by the local capstan
coefficient

```text
alpha = exp(mu theta)
beta = (alpha - 1) / (alpha + 1),
```

where `theta` is the wrap angle and `mu` is the contact friction coefficient.
This gives:

- `mu = 0`: `alpha = 1`, `beta = 0`, so pulley surface motion does not drag
  cable material;
- finite `mu`: partial surface transfer;
- high `mu`: `beta -> 1`, recovering the no-slip surface-transfer limit.

After this geometric transfer, the same capstan tension-ratio projection
described below is applied to prevent the adjacent span tensions from exceeding
the friction cone.

## 4. Stretch Row

For each free span, the stretch row enforces a unilateral distance constraint:

```text
C_s(q) = L_s(q) - r_s <= 0.
```

Equivalently, the row is active only when the span is taut:

```text
C_s > 0.
```

Let

```text
n_s = (x_s^R - x_s^L) / ||x_s^R - x_s^L||.
```

For a span endpoint on body `b`, with world center of mass `p_b` and endpoint
position `x`, the usual rigid-body distance Jacobian contributes

```text
J_v = +/- n_s
J_w = +/- (x - p_b) x n_s.
```

The signs are negative for the left endpoint and positive for the right
endpoint.

For `ROLLING` links, the spin-axis angular component is explicitly removed:

```text
J_w <- J_w - (J_w . a) a,
```

where `a` is the pulley axis in world space.

This is not a numerical convenience; it is part of the model. A pure spin about
the pulley axis does not change the tangent-point geometry or the free-span
length. If the stretch row kept this spin-axis term, it would impose an implicit
no-slip pulley constraint and bypass the capstan friction law. The stretch row
should carry cable tension. Pulley spin coupling belongs to the separated slip
row.

The XPBD update is the usual compliant row update. With stretch multiplier
`lambda`, compliance `c`, damping `d`, timestep `dt`, and constraint velocity
`Cdot`, the implemented increment has the form:

```text
gamma = c d
Delta lambda =
  -(C + c lambda + gamma Cdot) /
    ((dt + gamma) J M^-1 J^T + c / dt).
```

The resulting body correction is accumulated through the projected Jacobian.
The increment `Delta lambda` is also retained as a lagged load signal for the
rolling spin row.

## 5. Capstan Friction as Rest-Length Projection

At a frictional guide, let the two adjacent span tensions be `T_l` and `T_r`.
Let

```text
alpha = exp(mu theta),
```

where `theta` is the local bend or wrap angle. The capstan admissibility
condition is

```text
T_l <= alpha T_r
T_r <= alpha T_l.
```

Equivalently:

```text
1 / alpha <= T_l / T_r <= alpha.
```

When the two tensions are inside this band, the guide sticks and no rest length
is transferred. When one side exceeds the band, the model transfers rest length
between the two adjacent spans, conserving their sum.

Let

```text
e_l = max(L_l - r_l, 0)
e_r = max(L_r - r_r, 0)
T_l = e_l / c_l
T_r = e_r / c_r.
```

### Left Side Too Tight

If

```text
T_l > alpha T_r,
```

then the projection transfers rest length from the right span to the left span:

```text
r_l' = r_l + delta
r_r' = r_r - delta.
```

The new extensions are

```text
e_l' = e_l - delta
e_r' = e_r + delta.
```

The boundary condition after projection is

```text
e_l' / c_l = alpha e_r' / c_r.
```

Solving for `delta` gives

```text
delta = (c_r e_l - alpha c_l e_r) / (c_r + alpha c_l).
```

The implementation clamps `delta >= 0` and also clamps it so no free-span rest
length becomes negative.

### Right Side Too Tight

If

```text
T_r > alpha T_l,
```

then rest length transfers in the opposite direction:

```text
r_l' = r_l - delta
r_r' = r_r + delta.
```

The boundary condition is

```text
e_r' / c_r = alpha e_l' / c_l,
```

which gives

```text
delta = (c_l e_r - alpha c_r e_l) / (c_l + alpha c_r).
```

Again, `delta` is clamped to preserve non-negative rest lengths.

This projection is used for frictional pinholes and for rolling pulleys. The
same capstan inequality therefore controls both zero-radius guide slip and
finite-radius pulley slip.

## 6. Rolling Spin Row

The stretch row deliberately removes pulley spin-axis coupling. A separate
rolling slip row adds back only the spin-axis correction admitted by the
capstan cone.

For a rolling link, the solver evaluates the two adjacent span tension
estimates and computes

```text
force_sum  = T_l + T_r
force_diff = |T_l - T_r|
allowed_diff = beta force_sum
scale = min(1, allowed_diff / max(force_diff, epsilon)).
```

Using

```text
beta = (alpha - 1) / (alpha + 1),
```

is equivalent to the symmetric capstan band in difference/sum form:

```text
|T_l - T_r| <= beta (T_l + T_r).
```

The row then builds the spin-axis part of the angular correction implied by the
neighboring stretch increments, projects that correction onto the pulley axis,
and scales it by `scale beta`.

The limiting cases are useful:

- `mu = 0`: `alpha = 1`, `beta = 0`. The pulley receives no spin-axis grip
  torque from the cable.
- finite `mu`: only the capstan-admissible spin correction is applied.
- high `mu`: `alpha` is large and `beta -> 1`, recovering the no-slip rolling
  spin limit.

This row is separated from stretch so that no-slip behavior is a friction limit,
not an accidental consequence of the distance constraint.

## 7. Hysteresis from Stateful Rest-Length Distribution

The rest-length distribution is stateful. This naturally reproduces capstan
hysteresis.

Consider a fixed left endpoint, a kinematic rolling pulley, and a driven right
endpoint:

```text
fixed left endpoint -- left span -- pulley -- right span -- driven right endpoint
```

Pulling the right endpoint increases the right span's geometric length and
raises `T_r`. When

```text
T_r > alpha T_l,
```

the capstan projection transfers rest length from left to right:

```text
r_l decreases
r_r increases.
```

The left endpoint has not moved, but the left span's rest length is shorter, so
the fixed side develops tension:

```text
T_l = (L_l - r_l) / c_l.
```

At the loading boundary, the measured relation is

```text
T_r / T_l = alpha.
```

On unloading, the right endpoint moves back and `T_r` drops. The previous
rest-length distribution does not immediately reverse. As long as

```text
T_l <= alpha T_r,
```

the guide sticks, so `r_l` and `r_r` remain latched and the fixed-side tension
stays on a plateau. When unloading crosses the opposite boundary,

```text
T_l > alpha T_r,
```

rest-length transfer reverses and the unloading branch follows

```text
T_l / T_r = alpha.
```

The hysteresis loop comes from the static capstan band and the persistent
rest-length distribution. It does not require different static and kinetic
friction coefficients.

The benchmark added in `newton.tests.test_tendon_capstan` uses `mu = 0.2` and a
near-pi wrap angle. It measures:

```text
theta ~= 3.132 rad
alpha = exp(mu theta) ~= 1.871
peak T_app ~= 195.4 N
peak T_fix ~= 104.5 N
```

The loading branch tracks `T_app / T_fix = alpha`, early unloading sticks on
the fixed-side tension plateau, and reverse slip tracks
`T_fix / T_app = alpha`.

## 8. Solver Integration

The XPBD sequence is:

```text
for each solver iteration:
    update route contacts and rest-length transfer
    solve free-span stretch rows
    solve rolling spin/slip rows
```

The VBD rigid path reuses the same route update, stretch projection, and
rolling spin/slip projection rows as a tendon projection inside the rigid
iteration. This keeps the cable model conceptually identical between XPBD and
VBD: VBD regularizes the rigid solve, but it does not use a separate global
length force or a separate tendon force/Hessian formulation for routed cables.

## 9. Validation Targets

The current validation suite exercises the formulation through focused unit
tests and rendered examples:

- frictionless and frictional pinhole Atwood tests,
- pinhole slack preservation,
- dynamic and kinematic capstan `mu` sweeps,
- kinematic capstan hysteresis,
- dynamic rolling pulley angular coupling,
- high-inertia no-slip limit,
- motorized rolling pulley immediate cable drive,
- rolling transfer saturation at exhausted span length,
- compound pulley equilibrium,
- gear/block-and-tackle direction and mechanical advantage,
- cross-base XY table routing, and
- multi-guide pinhole routing on a passive cam.

The hysteresis benchmark is especially important because it checks the stateful
friction law rather than a monotonic "more friction means less slip" response.

## 10. Limitations and Future Work

This is a reduced cable model, not a resolved rope contact simulation.

Known limitations:

- The cable has no distributed mass or bending stiffness.
- Contact patches are not resolved spatially.
- Dynamic rerouting over arbitrary geometry is not yet represented as a
  geodesic/contact problem.
- Pinhole and pulley friction use local bend/wrap angle rather than a resolved
  pressure distribution.
- Displacement-driven force benchmarks require compliance somewhere: cable
  compliance, actuator compliance, or a load-cell model. With a perfectly
  inextensible cable and both endpoints prescribed, the system is
  overdetermined.
- The rolling spin row is currently an impulse-style projection derived from
  neighboring stretch increments. A future derivation could express it as an
  explicit complementarity row with a documented dual variable.

Despite these limits, the split stretch/slip formulation gives a compact,
stateful, and solver-friendly model for routed massless cables with capstan
friction. It captures the important engineering behavior of slip, stick,
no-slip limits, and hysteresis without requiring a discretized rope.
