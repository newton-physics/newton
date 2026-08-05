# LIMX Self-Collision Friction Design

## Goal

Add stable isotropic Coulomb friction to LIMX cloth vertex-face (VF) and
edge-edge (EE) self-collision. Preserve the current frictionless behavior by
default and validate the feature visually in the three-T-shirt box scene.

Edge-face (EF) untangling remains frictionless because it is an auxiliary
intersection-recovery constraint rather than a physical resting contact.

## Public API

Extend `ConstraintSelfCollision` with two optional parameters:

```python
friction: float = 0.0
friction_epsilon: float = 1.0e-2
```

`friction` is the dimensionless Coulomb coefficient. Matching `SolverVBD`,
`friction_epsilon` is the positive relative-velocity regularization threshold
[m/s]; the per-step displacement threshold is
`epsilon_u = friction_epsilon * dt`. Both must be finite; `friction` must be
nonnegative and `friction_epsilon` must be positive. The defaults retain
existing results and do not require friction work or storage at runtime.

The three-T-shirt box example will use `friction=0.4`, matching its box contact
coefficient. Its self-collision thickness remains uniformly 3 mm and its
adaptive stiffness factors remain `(0.5, 0.3, 1.5)`.

## Friction Model

At `begin_step()`, cache the particle positions at the beginning of the time
step. This follows the VBD cloth implementation, whose VF and EE friction use
the previous collision-detection positions as anchors. Contact detection at
each Newton linearization continues to freeze the contact particle IDs, signed
feature weights, normal, and penetration depth.

For one frozen VF or EE contact, compute the relative step displacement and its
tangential projection:

```text
u   = sum_i(w_i * (x_i - x_start_i))
P   = I - n n^T
u_t = P u
```

The normal load is `lambda_n = k_eff * depth` for VF and non-mollified EE
contacts. For an EE contact with the near-parallel mollifier active, multiply
the load by its scalar energy factor
`M = s * (2 * threshold - s) / threshold^2`, where `s` is the squared cross
product of the two edge vectors. Thus the Coulomb limit decreases with the
actual mollified normal response instead of applying full friction to a nearly
disabled contact.

Use the same IPC-style regularization already used by Newton's contact code:

```text
epsilon_u = friction_epsilon * dt
g(r) = 1 / r                          when r > epsilon_u
g(r) = (2 - r / epsilon_u) / epsilon_u otherwise

alpha = friction * lambda_n * g(length(u_t))
f_t   = -alpha * u_t
```

Distribute `f_t` to the contact particles with the same signed weights used by
the normal contact. This is the same anchor/weight construction used by VBD for
both VF and EE, gives equal-and-opposite feature forces, and preserves linear
momentum. Add the frozen PSD approximation `alpha * P` to the matrix-free
Hessian product and its weighted diagonal blocks. Like VBD, treat the contact
normal as constant in the friction Hessian; derivatives of the normal, normal
load, and regularization scale are lagged to favor robust Newton/CG solves.

Fixed and adaptive normal-stiffness modes use the same friction law. Adaptive
mode obtains `k_eff` from the existing feature-direction stiffness formula.

## Integration

- Allocate one step-start position array only when friction is enabled.
- Cache step-start positions in `begin_step()` for both fixed and adaptive
  stiffness modes.
- Add VF and EE friction force, Hessian-vector, and diagonal operations after
  their normal-contact operations.
- Reuse the EE mollifier state prepared for the normal operator.
- Skip all friction launches when `friction == 0.0`.
- Do not change contact detection, collision thickness, geometry radii, normal
  stiffness, EF untangling, or static-plane contact.

## Validation

Add focused `unittest` coverage that verifies:

- invalid friction parameters are rejected and defaults remain frictionless;
- VF and EE tangential forces oppose relative slip, sum to zero, remain finite,
  and respect the regularized Coulomb scale;
- the friction Hessian products and diagonal blocks are finite and PSD;
- an active EE mollifier reduces the friction load consistently.

Run only the focused LIMX self-collision tests, then launch the three-T-shirt
box scene for 20 seconds with 3 mm thickness, `friction=0.4`, and the existing
adaptive stiffness factors. The visual acceptance criterion is that fabric
layers resist sliding without restoring the previous persistent jitter.

Because this extends a public constraint's user-facing behavior, add an
`[Unreleased]` changelog entry. No API-generation step is needed because no new
public symbol is introduced.
