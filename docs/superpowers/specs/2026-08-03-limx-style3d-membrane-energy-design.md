# LIMX Style3D Membrane Energy Design

## Goal

Replace the distance-spring cloth elasticity in the LIMX example with an
energy-consistent anisotropic triangle membrane model based on the stretch and
shear terms used by Style3D and Baraff--Witkin cloth. Keep the existing
implicit projected-Newton integration, fixed-topology 3-by-3 block-CSR
assembly, block-Jacobi PCG, anchor constraints, and cross-frame PCG warm start.

Bending, damping, collision, rigid bodies, line search, and changes to the
time-step or iteration budget are outside this change.

## Why the Style3D Force Is Not Copied Literally

The current Style3D stretch kernel normalizes both deformation-gradient
columns before evaluating its shear force, but differentiates the shear term
as if those columns had not been normalized. A finite-difference check on a
deformed triangle gives a force Jacobian with a relative antisymmetric part of
approximately 0.158. Therefore that force is not a consistent gradient for a
Newton method and has no symmetric exact Hessian to assemble.

LIMX instead uses the energy-consistent Baraff--Witkin form below. This keeps
the same warp, weft, and shear material controls while giving Newton a defined
gradient and Hessian.

## Public Constraint Interface

Add `newton.solvers.ConstraintTriangleElastic` with constructor inputs:

```python
ConstraintTriangleElastic(
    triangle_indices,
    inverse_rest_matrices,
    rest_areas,
    stiffnesses,
    particle_count,
    device,
)
```

`stiffnesses[t] = (ku, kv, ks)` controls warp stretch, weft stretch, and
in-plane shear for triangle `t`. Components are finite and nonnegative. Rest
areas are finite and positive, inverse rest matrices are finite and
nonsingular, and the three particle indices of every triangle are distinct.

Accepting rest data explicitly keeps the constraint independent from the
model builder. Callers may pass the ordinary `Model.tri_poses` and
`Model.tri_areas`, or Style3D panel-space rest data for cloth with separate
material coordinates.

## Membrane Energy and Force

For triangle positions `x0`, `x1`, `x2` and inverse material-space rest matrix
`B = inv(Dm)`, define

```text
Fu = (x1 - x0) B00 + (x2 - x0) B10
Fv = (x1 - x0) B01 + (x2 - x0) B11
```

Equivalently, with

```text
a = (-B00 - B10, B00, B10)
b = (-B01 - B11, B01, B11),
```

the deformation columns are `Fu = sum_i a_i x_i` and
`Fv = sum_i b_i x_i`. For rest area `A`, use

```text
E = A / 2 [
    ku (||Fu|| - 1)^2
  + kv (||Fv|| - 1)^2
  + ks (Fu dot Fv)^2
].
```

Let `nu = Fu / ||Fu||`, `nv = Fv / ||Fv||`, `c = Fu dot Fv`, and
`g_i = a_i Fv + b_i Fu`. The force on vertex `i` is the negative gradient:

```text
f_i = -A [
    ku a_i (||Fu|| - 1) nu
  + kv b_i (||Fv|| - 1) nv
  + ks c g_i
].
```

If a deformation column has length at most `1e-8`, its nondifferentiable
stretch term contributes zero force and zero Hessian. The polynomial shear
term remains active and finite.

## PSD-Projected Hessian

For warp stretch, the exact material-space 3-by-3 curvature is

```text
Qu = ku [nu nu^T + (1 - 1 / ||Fu||) (I - nu nu^T)].
```

Its radial eigenvalue is `ku`; its two transverse eigenvalues are
`ku (1 - 1 / ||Fu||)`. Project only the negative transverse eigenvalues:

```text
Qu+ = ku [nu nu^T + max(1 - 1 / ||Fu||, 0) (I - nu nu^T)].
```

The weft term `Qv+` is identical with `Fv`, `nv`, and `kv`. Their triangle
Hessian blocks are

```text
H_ij_stretch = A (a_i a_j Qu+ + b_i b_j Qv+).
```

For shear, the exact block is

```text
H_ij_shear_exact = A ks [
    g_i g_j^T + c (a_i b_j + b_i a_j) I
].
```

The second term is the residual-weighted second derivative and can make the
matrix indefinite. Drop that complete term and retain the Gauss--Newton part:

```text
H_ij_shear+ = A ks g_i g_j^T.
```

The assembled block is

```text
H_ij+ = H_ij_stretch + H_ij_shear+.
```

The stretch blocks are Kronecker products of positive-semidefinite matrices,
and the shear blocks form `A ks J^T J`. Thus the local 9-by-9 membrane Hessian
is positive semidefinite. The positive mass term makes the PCG system positive
definite without an artificial diagonal shift.

## Sparse Assembly

Every triangle registers all nine particle-pair block coordinates once. The
block-CSR pattern remains fixed, while each Newton iteration clears and
reassembles numerical values from the current iterate. Shared triangle blocks
accumulate atomically. This preserves the existing separation between static
block-CSR elasticity and future matrix-free dynamic collision terms.

## Example Migration

The `cloth_limx` example keeps its 21-by-21 particle grid, alternating
triangulation, two corner anchors, `dt=0.01`, one Newton iteration, 50 PCG
iterations, and one render per physics step. Replace the unique-edge
`ConstraintDistance` batch with one `ConstraintTriangleElastic` batch using
the model's triangle rest matrices and areas. Use anisotropic stiffness in the
same 10:10:1 warp/weft/shear ratio as the Style3D defaults, scaled for the
existing scene.

The example remains membrane-only. It intentionally has no bending energy, so
folding and wrinkling are allowed while in-plane stretch and shear are
resisted.

## Tests and Completion Criteria

Use `unittest` and real Warp kernels.

- At a rotated rest pose, force is zero.
- At a deformed pose, force matches a finite-difference gradient of the stated
  scalar energy.
- In extension, the stretch Hessian matches the finite-difference force
  Jacobian with the expected sign.
- In compression, negative stretch transverse curvature is removed while the
  radial curvature remains.
- The shear blocks equal the hand-derived `A ks J^T J` matrix and the complete
  local Hessian is symmetric positive semidefinite.
- Reassembly changes numerical blocks without changing the CSR pattern.
- Invalid topology, rest data, stiffness, or device inputs are rejected.
- The public export is available from `newton.solvers`.
- The two-corner cloth example remains finite, sags, swings, keeps anchors,
  and limits in-plane deformation on CPU and CUDA.

