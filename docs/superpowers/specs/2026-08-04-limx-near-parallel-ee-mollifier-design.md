# LIMX Near-Parallel EE Mollifier Design

## Goal

Remove persistent local T-shirt jitter caused by discontinuous closest-point
weights on nearly parallel, topology-local edge-edge contacts. Preserve the
existing one-Newton, 50-PCG, `0.01 s` workflow without adding global,
material, or self-contact damping.

## Observed failure

The previous settling regression used global mean particle speed and hid a
small set of continuously oscillating particles. Over the final 500 frames of
the T-shirt scene:

- EE averaged about 227 active contacts but about 59 births and 59 deaths per
  frame;
- 90.5% of disappearing EE contacts were nearly parallel (`sin(theta) < 0.1`);
- 97.6% were topology-local pairs using the half-average-edge-length thickness
  clamp;
- 56.6% disappeared with more than `0.5 mm` penetration while their previous
  closest parameters remained at least `0.02` away from every endpoint;
- disabling self-collision removed all particles with final-window RMS speed
  above `0.02 m/s`, while increasing Newton iterations from one to two made
  the nonsmooth full-step solve unstable.

The root cause is therefore not table contact, insufficient Newton iteration,
or a global lack of damping. Nearly parallel segments have a non-unique
closest-point pair. Their parameters and endpoint force weights jump between
valid branches even when distance and geometry move only slightly.

## Scope

Change only topology-local EE collision response. A pair is topology-local
when either edge has an opposite triangle vertex that is an endpoint of the
other edge, which is the same condition already used by the local thickness
clamp.

Keep these behaviors unchanged:

- non-local EE contact, including non-local parallel edges;
- VF detection and force;
- EF untangling;
- adaptive factors `(VF=0.5, EE=0.1, EF=1.5)`;
- nominal thickness `0.006 m` and the existing local EE thickness clamp;
- table collision parameters;
- matrix-free dynamic topology and PSD rank-one contact Hessians;
- the fixed-stiffness `ConstraintSelfCollision` API and its base stiffness
  selection. The new local EE response scale applies to fixed and adaptive
  modes alike.

## Mollifier

For a topology-local EE pair with current edge vectors `e0` and `e1`, compute

```text
sin2 = |e0 x e1|^2 / (|e0|^2 |e1|^2)
q = clamp(sin2 / sin2_threshold, 0, 1)
m = q^2 (3 - 2q)
```

Use `sin_threshold = 0.2`, hence `sin2_threshold = 0.04`. This corresponds to
an angle of approximately `11.5 degrees`:

- exactly parallel topology-local edges have `m=0`;
- topology-local edges at or above the threshold have `m=1`;
- the transition has zero slope at both endpoints;
- every non-local EE pair has `m=1` regardless of angle.

The mollifier is computed and frozen during `prepare()` together with contact
ids, weights, directions, and depths. Each contact buffer stores one scalar
response scale; VF and EF store `1`, and EE stores `m`.

## Force and Hessian

Both fixed and adaptive modes use

```text
k_effective = response_scale * k_contact
f = k_effective * depth * g
H_GN = k_effective * g g^T
```

where `g_i = weight_i * direction`. Force accumulation, Hessian-vector
multiplication, and exact diagonal accumulation read the same frozen scale.
Because `response_scale >= 0` and `k_contact > 0`, the frozen Gauss-Newton
Hessian remains positive semidefinite. Derivatives of the mollifier, closest
parameters, direction, and adaptive stiffness are intentionally omitted, as
the current matrix-free contact linearization already freezes those geometric
quantities per Newton iteration.

The mollifier does not dissipate energy and does not add a velocity-dependent
force. It removes the ill-defined EE response as the local pair approaches the
parallel degeneracy, where elasticity and bending should govern the one-ring
geometry.

## Validation

Add CUDA tests that independently establish:

- fixed and adaptive contact force, HVP, and diagonal are all multiplied by
  the same stored response scale;
- a parallel topology-local EE contact is detected with scale zero;
- a topology-local EE contact at or above the threshold has scale one;
- a parallel non-local EE contact retains scale one;
- every stored scale is finite and lies in `[0, 1]`;
- the existing local thickness clamp, ordinary EE detection, VF, EF, and
  fixed-mode rank-one tests remain unchanged.

Strengthen the 3000-frame T-shirt regression to accumulate squared particle
speed over the final 500 frames and require:

```text
max_particle_rms_speed < 0.02 m/s
count(particle_rms_speed >= 0.02 m/s) == 0
```

Retain the existing global mean-speed and table-penetration checks. After the
headless regression passes, launch the CUDA viewer for visual confirmation;
the automated metric is necessary but does not replace the visual check.

## Non-goals

- Do not add collision damping, restitution, friction, or hysteresis.
- Do not lower the scene's EE stiffness factor.
- Do not implement complete VE/VV feature classification in this change.
- Do not add line search, CCD, or additional Newton iterations.
- Do not change table contact or bending parameters.
