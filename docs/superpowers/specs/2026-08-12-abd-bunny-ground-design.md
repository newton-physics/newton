# ABD Bunny Ground Contact Design

## Goal

Add the first contact-enabled LIMX affine-body example: one high-rigidity
ARAP bunny falls onto a fixed horizontal plane, responds through penalty
normal contact, and settles with regularized Coulomb friction. The complete
contact response remains in the affine body's native 12-degree-of-freedom
space and participates in the same Newton/PCG solve as inertia and ARAP.

This example is the contact foundation for later affine-body/cloth coupling,
but particle-affine contact and cloth dynamics are outside this change.

## Affine Coordinates

Keep the existing generalized state

\[
q = [t_x,t_y,t_z,A_{00},A_{01},A_{02},A_{10},A_{11},A_{12},A_{20},A_{21},A_{22}]^T.
\]

A material point with centered rest coordinate \(\bar x\) has world position

\[
x(q)=t+A\bar x=J(\bar x)q,
\]

with the constant point Jacobian

\[
J(\bar x)=
\begin{bmatrix}
1&0&0&\bar x^T&0&0\\
0&1&0&0&\bar x^T&0\\
0&0&1&0&0&\bar x^T
\end{bmatrix}.
\]

The bunny is not represented by the world positions of four proxy vertices.
Every surface sample is reconstructed from the same \(A+t\) state. This is
consistent with libuipc's `ABDJacobi`, while retaining Newton's centered rest
coordinates and native 12-by-12 mass and preconditioner blocks.

## Penalty Contact

For a normalized plane with normal \(n\) and offset \(o\), define

\[
d(x)=n^Tx-o,\qquad \delta=[h-d(x)]_+,
\]

where \(h\) is the one-sided contact thickness. The normal penalty energy and
force are

\[
E_n=\tfrac12 k_n\delta^2,\qquad f_n=k_n\delta n.
\]

An approaching point receives the same lagged normal damping used by
`ConstraintStaticPlaneContact`:

\[
f_d=-c_n\min(n^Tv,0)n.
\]

The corresponding positive-semidefinite world-space stiffness is

\[
H_n=k_n n n^T+[n^Tv<0]\frac{c_n}{\Delta t}n n^T.
\]

Normal contact is penalty-based. This design does not add an IPC barrier,
line search, continuous collision detection, or proxy particles.

## Friction

Use the existing LIMX regularized Coulomb penalty model. At the start of the
time step, compute each surface point velocity from the affine velocity,

\[
v=J(\bar x)\dot q.
\]

With \(P_t=I-nn^T\), define the lagged tangential displacement

\[
\Delta x_t=\Delta t\,P_tv.
\]

The regularized inverse length is

\[
s(r)=
\begin{cases}
1/r, & r>\epsilon,\\
(2-r/\epsilon)/\epsilon, & r\le\epsilon.
\end{cases}
\]

For \(r=\|\Delta x_t\|\), the friction contribution is

\[
f_t=-\mu k_n\delta\,s(r)\Delta x_t,
\qquad
H_t=\mu k_n\delta\,s(r)P_t.
\]

The scalar friction coefficient is frozen while forming the PSD approximate
Hessian, matching the current particle-plane contact convention. Friction
therefore opposes the step-start tangential motion and approaches the Coulomb
limit away from the regularization zone.

## Lifting Contact to Affine Space

For every active surface vertex, form its cached world-space force \(f_x\)
and PSD stiffness \(H_x\), then lift them exactly:

\[
f_q=J^Tf_x,\qquad H_q=J^TH_xJ.
\]

The matrix-free Hessian-vector product evaluates

\[
y_q\mathrel{+}=J^TH_x(Jp_q)
\]

without storing per-contact 12-by-12 matrices. The block-Jacobi
preconditioner still receives the complete 12-by-12 sum \(J^TH_xJ\) for each
body. Surface ownership selects the destination affine body, so the kernels
remain compatible with a future multi-body model even though the current
`AffineBodyModel` contains one body.

## Components

### `ConstraintAffineStaticPlaneContact`

Add a public LIMX dynamic constraint that owns:

- the affine model's centered rest surface vertices and body ownership;
- normalized plane and penalty/friction parameters;
- per-surface-point cached world force and 3-by-3 PSD stiffness;
- step-start generalized velocity and time step.

Its lifecycle is:

1. `begin_step(q, qd, dt)` validates and caches step-start affine velocity.
2. `prepare(q)` reconstructs current world points, evaluates active penalty
   contacts, and caches world force/stiffness.
3. `accumulate_force(q, affine_output)` adds all \(J^Tf_x\) terms.
4. `multiply(...)` adds matrix-free \(J^TH_xJp_q\) terms to the affine side
   of `MixedLinearOperator`; particle arrays are empty and unchanged.
5. `accumulate_diagonal(...)` adds exact \(J^TH_xJ\) blocks to the affine
   preconditioner.

Validate finite coefficients, a nonzero plane normal, positive thickness,
stiffness, and friction regularization, nonnegative damping/friction, matching
devices and vector sizes, and the required lifecycle order.

### `SolverLIMXAffine`

Accept an optional affine dynamic operator. The default remains
`EmptyMixedDynamicOperator`, preserving the existing collision-free behavior.
With a contact operator, the solver:

1. begins the contact step after forming the inertial target;
2. prepares contact at the current affine Newton iterate;
3. adds cached contact force to the affine right-hand side;
4. lets `MixedLinearOperator` include contact HVP and exact diagonal terms;
5. applies the solved affine increment as before.

Warm start behavior remains unchanged: the first and only Newton solve uses
the previous frame's increment.

## Example

Add `basic_limx_affine_bunny_ground` with these fixed settings:

- asset: `newton/examples/assets/bunny_tet.npz`;
- geometry scale: `0.15`;
- density: `1000 kg/m^3`, giving approximately `2.81 kg` total mass;
- ARAP rigidity: `1.0e8 Pa`;
- initial orientation: the existing bunny upright rotation plus a `15 degree`
  tilt;
- initial translation: centered at `z = 0.65 m`;
- ground: visible static box whose top is `z = 0`;
- contact thickness: `0.003 m`;
- normal stiffness: `2.0e4 N/m` per surface sample;
- normal damping: `0.5 N*s/m`;
- friction coefficient: `0.5`;
- friction regularization: `1.0e-4 m`;
- time step: `0.01 s`;
- nonlinear iterations: exactly `1`;
- PCG iterations: exactly `50`;
- default frame count: `300`.

The example reconstructs the bunny surface into a render-only Newton model,
captures the simulation step into a CUDA graph when supported, and provides a
camera that shows the complete drop and resting pose.

## Tests and Acceptance

Use `unittest` and add focused tests before implementation:

1. Compare generalized force, matrix-free HVP, and the complete 12-by-12
   preconditioner block against dense `J.T @ f` and `J.T @ H @ J` references.
2. Verify inactive contacts contribute exactly zero.
3. Verify normal damping acts only while approaching the plane.
4. Verify friction opposes tangential velocity and remains finite inside its
   regularization zone.
5. Verify invalid parameters, mismatched devices/vector sizes, and calls to
   `prepare()` before `begin_step()` fail clearly.
6. Verify `SolverLIMXAffine` remains unchanged without a dynamic operator and
   includes contact contributions when one is supplied.
7. Run the bunny example for 300 frames on CUDA, including graph capture, and
   require finite generalized state, velocity, and reconstructed surface,
   `det(A) > 0`, and a maximum singular-value deviation
   `max(abs(sigma(A) - 1)) < 0.02` throughout the rollout.
8. Require the center of mass to fall by at least `0.20 m`, at least one
   surface point to enter the `0.003 m` activation band, and the minimum
   surface height to remain above `-0.006 m`.
9. Require the mean tangential translation speed over the final 30 frames to
   be below `0.05 m/s`, so the frictional scene does not slide indefinitely.

These rollout tolerances may not be weakened merely to hide visible
penetration, loss of rigidity, or continued sliding.

## References

- libuipc `docs/specification/constitutions/affine_body.md`: `A+t` state and
  material-point Jacobian.
- libuipc `src/backends/cuda/affine_body/abd_jacobi_matrix.{h,cu}`: efficient
  `J`, `J.T`, and `J.T H J` operations.
- Newton `ConstraintStaticPlaneContact`: penalty normal force, lagged normal
  damping, and regularized Coulomb friction used by this design.
