# LIMX Mixed Affine–Particle Newton Design

## Goal

Extend LIMX with native 12-DOF affine-body rows while preserving native
3-DOF particle rows. Both spaces participate in one projected-Newton and PCG
solve. The first implementation milestone validates the heterogeneous linear
algebra and a positive-semidefinite ARAP affine body without collision. DCD
mixed contact follows as a separate milestone.

## Chosen Architecture

The solver state is split by degree-of-freedom type:

- particles use `wp.array[wp.vec3]` and 3-by-3 block CSR;
- affine bodies use `wp.array[vec12]` and 12-by-12 block CSR;
- a mixed operator owns interactions that span the two spaces.

The global Newton system is logically

\[
\begin{bmatrix}
A_{pp} & A_{pa} \\
A_{ap} & A_{aa}
\end{bmatrix}
\begin{bmatrix}
\Delta x_p \\
\Delta q_a
\end{bmatrix}
=
\begin{bmatrix}
b_p \\
b_a
\end{bmatrix}.
\]

`A_pp` and `A_aa` are stored independently. Mixed blocks are not stored in a
third CSR matrix: the coupling operator applies them matrix-free and scatters
to both output arrays. PCG reductions sum contributions from both vector
spaces, so this remains one coupled solve rather than alternating particle and
affine solves.

## Affine State

Each affine body stores

\[
q=[t, a_0, a_1, a_2]^T\in\mathbb{R}^{12},
\]

where `t` is translation and the three `a_i` are the rows of the affine matrix
`A`. A rest-space point is reconstructed as

\[
x(\bar{x})=t+A\bar{x}.
\]

The body also stores generalized velocity, previous state, inertial target,
the full 12-by-12 consistent affine mass matrix, its rest volume, and ARAP
rigidity.

The initial milestone accepts explicit affine-body descriptors rather than
changing Newton's public `Model` rigid-body representation. This keeps the
new solver path isolated until its state conventions and dynamics are
validated.

## Affine Mass and Inertia

The 12-by-12 consistent mass matrix is assembled from the rest body's dyadic
moments:

\[
M=\int_\Omega \rho J(\bar{x})^T J(\bar{x})\,dV.
\]

It is constant for an affine body. The implicit-Euler inertial contribution is

\[
H_M=M/\Delta t^2,
\qquad
b_M=M(\tilde q-q)/\Delta t^2.
\]

Gravity is lifted into generalized coordinates using the same integrated
Jacobian rather than being applied only to the translation entries.

## Positive-Semidefinite ARAP Rigidification

The affine matrix is kept close to `SO(3)` with

\[
E_r(A)=\mu V\|A-R(A)\|_F^2,
\qquad R(A)=\operatorname{polar}(A)\in SO(3).
\]

The force uses the exact ARAP first Piola derivative

\[
P=2\mu(A-R).
\]

The analytic 9-by-9 derivative `dP/dA` is projected to positive semidefinite
before embedding it in the affine portion of a 12-by-12 block. Translation
rows and translation–affine columns are zero for this energy. No post-step
polar projection is applied: rigidification remains part of the objective, so
it cannot invalidate an optimized contact state.

The existing tetrahedral ARAP formulas are the numerical reference, but the
affine constraint operates directly on `A` and emits one native 12-by-12 body
block.

## Native Block CSR Storage

A dedicated affine block-CSR implementation stores:

- integer row offsets and column indices;
- contiguous `mat12` numerical blocks;
- one cached `mat12` diagonal per affine row;
- block lookup and stencil lookup compatible with the existing 3-by-3 builder.

The implementation does not generalize the existing 3-by-3 class with a
runtime block size. Warp kernels retain compile-time concrete vector and
matrix types, keeping both paths simple and specialized.

## Heterogeneous PCG

PCG owns particle and affine work vectors for residual, preconditioned
residual, direction, operator direction, and solution. Every scalar reduction
is the sum of both spaces:

\[
\langle r,z\rangle=
\sum_i r_{p,i}^Tz_{p,i}+
\sum_b r_{a,b}^Tz_{a,b}.
\]

The operator multiplication sequence is:

1. multiply the particle 3-by-3 CSR;
2. multiply the affine 12-by-12 CSR;
3. apply matrix-free particle, affine, and mixed dynamic operators;
4. accumulate all results into the two output arrays.

The block-Jacobi preconditioner uses native inverses:

- one 3-by-3 inverse for every particle row;
- one 12-by-12 Cholesky solve for every affine row.

Each affine diagonal is symmetrized before factorization. A scale-relative
diagonal regularization is added only when Cholesky detects a non-positive or
non-finite pivot. PCG must never continue with an indefinite preconditioner.

## Future DCD Coupling Interface

The mixed operator interface is included in the linear-algebra design, but DCD
generation and response are outside the first milestone. A future frozen DCD
contact computes a scalar directional residual

\[
d=n^T\left(\sum_i w_i x_i + J_aq_a\right),
\]

and applies

\[
f=k(\hat d-d)J^T,
\qquad
Hv=kJ^T(Jv).
\]

This directly supplies particle–particle, affine–affine, and mixed
particle–affine coupling without materializing 3-by-12 CSR blocks. The contact
set, normal, and closest-feature weights are frozen within one Newton
linearization and rebuilt at the next nonlinear iteration.

The existing collision policy remains authoritative: unsigned closest-point
VF owns PE/PP boundary cases, and EE is strict interior–interior. Closed affine
bodies may additionally use oriented signed VF recovery; arbitrary cloth does
not acquire an inside/outside convention.

## First Milestone Scope

The first implementation delivers:

1. Warp `vec12` and `mat12` numerical types and tested primitive operations;
2. 12-by-12 affine block CSR construction, diagonal extraction, and SpMV;
3. native 12-by-12 Cholesky preconditioning;
4. heterogeneous particle/affine PCG with an empty mixed operator;
5. affine mass, implicit inertia, gravity, and PSD-projected ARAP;
6. a one-body free-fall/rotation example and CUDA `unittest` coverage.

It deliberately excludes DCD contact, friction, joints, motors, CCD, changes
to Newton's existing rigid-body model, and public API stabilization.

## Verification

Unit tests use `unittest` and cover:

- affine CSR SpMV against a hand-computed dense result;
- 12-by-12 Cholesky application against a dense residual check;
- heterogeneous PCG against a small hand-constructed coupled SPD system;
- zero-rigidity affine free fall matching the analytic translation while `A`
  remains unchanged;
- nonzero-rigidity recovery from a perturbed affine matrix;
- positive eigenvalues of the assembled inertia-plus-projected-ARAP body
  block;
- CUDA graph capture for a complete affine step.

The milestone is accepted when all focused tests and pre-commit checks pass,
the free-fall example remains finite, and its maximum singular-value deviation
from one decreases under the ARAP objective.
