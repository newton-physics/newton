# LIMX T-Shirt on Table Design

## Goal

Add a CUDA visualization and regression scene in which a real T-shirt mesh is
thrown onto a static table, collides with itself and the tabletop, and settles
without global velocity damping.

## Scope

The cloth is the only simulated object. The table is a rendered static box,
while collision uses an analytic one-sided plane coincident with the box's top
surface. The first version deliberately does not support falling over the table
edge, moving colliders, rigid-body degrees of freedom, or arbitrary static
meshes.

## Garment and scene

Load Newton's existing `unisex_shirt.usd` mesh at `/root/shirt`. The asset has
6,436 vertices and 12,736 triangles. Scale its centimeter coordinates by
`0.01`, recenter it horizontally, and preserve its three-dimensional rest
shape. Its scaled surface area is approximately `0.706 m^2`; an areal density
of `0.3 kg/m^2` gives a total mass of approximately `0.212 kg`. Distribute each
triangle's mass equally to its three vertices so every particle has a positive,
area-weighted mass.

Use the existing anisotropic triangle membrane constraint with equal warp and
weft stiffness because this USD mesh has no UV coordinates. Use its lower shear
stiffness, rest-shape dihedral bending, and the existing VF/EE/EF self-collision
operator. No particles are anchored.

Render a `1.1 m x 0.9 m x 0.1 m` table centered at `z = 0.60 m`, so its top is
at `z = 0.65 m`. Place the shirt center near `z = 1.05 m`, give it a mild tilt,
a downward and small horizontal linear velocity, and a particle velocity field
derived from a modest angular velocity about its center. This creates a visible
throw without relying on such a large per-frame displacement that the initial
discrete contact shell can be skipped.

## Dynamic-constraint composition

Add a public `ConstraintGroupDynamic` that owns an ordered sequence of
matrix-free dynamic constraints. It forwards step preparation, nonlinear
contact preparation, force accumulation, Hessian-vector products, and exact
diagonal accumulation to each child. The LIMX solver continues to consume one
dynamic operator, so neither the solver nor PCG needs type-specific knowledge
of self-collision or table contact.

Extend the dynamic-constraint lifecycle with:

```python
begin_step(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    dt: float,
) -> None
```

`SolverLIMX.step()` calls it once before the Newton loop. Existing dynamic
constraints implement it as a no-op unless they need lagged step data.
`prepare(positions)` remains the per-Newton-linearization contact-detection
hook.

## Static plane contact

Add a public `ConstraintStaticPlaneContact` as a one-particle, matrix-free
dynamic constraint. Its plane is represented by a normalized outward normal
`n` and offset `s`, with signed distance

```text
d(x) = dot(n, x) - s.
```

For `d < h`, cache the penetration depth `p = h - d`. The frozen-normal normal
penalty is

```text
E_n = 0.5 * k * p^2,
f_n = k * p * n,
H_n = k * outer(n, n).
```

Only active contacts receive dissipation. `begin_step()` caches the prior-frame
velocity and timestep. Approaching normal velocity receives a lagged damping
force with coefficient `c_n`. With `v_n = dot(v, n)`, use
`f_d = -c_n * min(v_n, 0) * n`; its positive-semidefinite implicit tangent adds
`(c_n / dt) * outer(n, n)` only while `v_n < 0`.

Tangential friction follows the smoothed lagged Coulomb construction used by
Ai-Physics `StaticContactSet`. Let `u_t` be the previous displacement
`dt * v` projected onto the tangent plane, `T = I - outer(n, n)`, and
`f_load = k * p`. Freeze

```text
alpha = mu * f_load * f1_SF_over_x(length(u_t), epsilon_u).
```

The force receives `-alpha * u_t`, and the positive-semidefinite contact
Hessian receives `alpha * T`. Use `epsilon_u = 1e-4 m` to regularize sticking.
The force, Hessian-vector product, and diagonal are evaluated without a sparse
matrix or contact atomics because each kernel thread owns one particle.

Validate finite positive thickness and stiffness, finite nonnegative damping
and friction, positive friction epsilon, a nonzero finite normal, matching
particle counts, and matching devices.

## Initial parameters

- Time step: `0.01 s`, one physics step per rendered frame.
- Newton iterations: `1`.
- PCG iterations: `50`, retaining the existing previous-frame warm start.
- Global velocity damping: `1.0`.
- Gravity: `(0, 0, -9.81) m/s^2`.
- Membrane stiffness: `(1e4, 1e4, 1e3)`.
- Dihedral bending stiffness: `0.01`.
- Self-collision thickness: `0.006 m`.
- VF/EE stiffness: `1e4 N/m`; EF stiffness: `3e4 N/m`.
- Table-contact thickness: `0.006 m`.
- Table normal stiffness: `2e4 N/m`.
- Table normal damping: `0.5 N*s/m`.
- Table friction coefficient: `0.4`.
- Friction regularization distance: `1e-4 m`.

These are explicit scene parameters and may be tuned after the first visual run,
without changing the constraint interfaces.

## GPU execution and tests

All simulation validation runs on `cuda:0`. Add CUDA `unittest` coverage for:

- plane contact detection above, inside, and below the activation shell;
- outward normal force, positive-semidefinite Hessian-vector action, and exact
  diagonal blocks;
- contact-only normal damping and tangential friction while preserving zero
  contributions away from contact;
- ordered composition of self-collision and static-plane contributions;
- the solver calling `begin_step()` once and `prepare()` once per Newton
  iteration;
- one CUDA-graph example step with finite positions and velocities;
- an extended headless rollout that keeps the shirt above a small table-plane
  penetration tolerance and reduces its final-window mean particle speed below
  `0.02 m/s`.

The example implements `test_final()` and uses the shared OpenGL viewer controls.
Do not run mixed CPU/GPU LIMX test modules during routine verification; select
the exact CUDA tests by file and test name.
