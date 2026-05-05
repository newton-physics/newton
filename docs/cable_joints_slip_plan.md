# Routed Cable Slip Plan

This document records the target formulation and acceptance policy for adding
finite slip back to Newton's routed cable solver.  It should be treated as the
working contract for future cable/tendon friction changes.

## Known-Good Baseline

Commit `21ca37106` (`Document routed cable slip plan`) is the known-good
no-friction routed-cable baseline.  If slip work breaks existing routed-cable
behavior in a way that is not a small, justified tolerance change or a
documented physical-expectation correction, compare against this commit before
continuing.

## Design Criteria

### One routed-cable formulation

- Use the same constraints for slip and no-slip behavior.
- Do not add separate slip and no-slip solver paths.
- Do not add an explicit no-slip row.
- Do not change the physical Jacobian based on a stick/slip classification.
- The Jacobian should represent the contact and rolling coupling; friction
  projection should determine how much of that coupling is admissible.

### Friction coefficient determines the regime

- `mu = 0`: the tangential/friction impulse projects to zero, so the cable
  slips freely.
- finite `mu`: the tangential impulse is projected into the friction cone,
  giving partial slip.
- high or infinite `mu`: the projection should not clamp under ordinary loads,
  recovering the no-slip limit.
- Stick or slip is an outcome of the impulse projection, not a model branch.

### Mass properties determine dynamic vs kinematic behavior

- Dynamic pulleys have finite inverse inertia, so friction impulses can rotate
  them.
- Kinematic pulleys have zero inverse mass and zero inverse inertia, so
  friction impulses cannot move them.
- There should be no separate kinematic pulley solver path.
- High friction against a kinematic pulley should lock the cable naturally
  because the pulley cannot move to satisfy rolling.

## Success Criteria

### Dynamic capstan

- `mu = 0`: the cable slips freely and pulley rotation stays near zero.
- mid `mu`: the pulley rotates in the cable direction, but rim travel remains
  below the no-slip limit.
- high `mu`: the no-slip limit is recovered; cable displacement matches
  `R * dtheta` within tolerance.

### Kinematic capstan

- `mu = 0`: pure slip; the cable moves through the fixed pulley.
- mid `mu`: slip still occurs, but relative slip is lower than in the
  zero-friction case.
- high `mu`: the system locks; cable motion through the fixed pulley is near
  zero.

### Cross-cutting invariants

- Relative slip decreases monotonically as `mu` increases.
- Dynamic and kinematic capstan cases use the same friction solver path.
- Existing no-slip routed-cable examples remain stable.
- Pulley angular velocity signs match cable direction where rotation is
  expected.
- Total cable length remains bounded and does not accumulate systematic drift.
- The friction solve must not inject energy: pulley rim motion should not
  exceed what the cable/friction coupling can justify.

## Test Strategy

### 1. Harden the no-slip baseline first

Before changing friction behavior, preserve or expand tests for:

- weight motion direction,
- pulley spin sign,
- cable length conservation,
- immediate cable response to driven pulley motion,
- bounded pulley/body motion,
- no unexpected motion for balanced or equilibrium scenes.

### 2. Add capstan acceptance tests before implementation

Dynamic capstan tests should cover:

- zero `mu`: no pulley rotation,
- mid `mu`: small correct-direction pulley rotation,
- high `mu`: no-slip rim/cable agreement.

Kinematic capstan tests should cover:

- zero `mu`: free slip,
- mid `mu`: less slip than zero `mu`,
- high `mu`: locked motion.

Both should include monotonic slip checks over increasing `mu`.

### 3. Implement unified friction projection

The intended XPBD pipeline is:

```text
update_tendon_attachments()
solve_tendon_stretch()
solve_tendon_friction()
```

`solve_tendon_friction()` should:

- compute the physical rolling/friction row with the full angular coupling,
- use projection/clamping of the tangential impulse to enforce the friction
  limit,
- use lagged tension estimates if needed for Coulomb bounds,
- treat `mu = infinity` as a projection limit, not a separate solver branch.

### 4. Validate simple scenes before complex scenes

Implementation order:

1. Kinematic capstan.
2. Dynamic capstan.
3. Rolling pulley.
4. Compound pulley.
5. 3D routing.
6. Cable machine.
7. Cross-base XY table.
8. Gear pulley.

Only promote a complex scene after it has tests that catch direction, delayed
coupling, pulley sign, length drift, and unbounded-motion failures.

## Test Update Policy

Acceptable test updates:

- Small tolerance changes caused by the new numerical formulation.
- Removing assertions that encode implementation details rather than physical
  behavior.
- Correcting expected values when the old behavior was known to be wrong.
- Adding stronger physical invariants when any threshold is relaxed.

Not acceptable:

- Removing sign or direction tests.
- Removing monotonicity tests.
- Replacing physical assertions with only boundedness checks.
- Large tolerance increases without a documented reason.
- Example-specific hacks to preserve visuals.
- Hidden separate dynamic/kinematic or slip/no-slip paths.

Any test update made during this work should be described as one of:

- `numerical tolerance`
- `physical expectation corrected`
- `implementation-detail assertion removed`
- `new invariant added`

If a tolerance is relaxed, keep the change small by default and pair it with an
invariant that still checks the physical behavior.  A broad relaxation requires
a written reason in the commit or PR notes.
