# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ...sim import BodyFlags, Model, State
from ..solver import integrate_rigid_body


@wp.func
def inertia_to_second_moment(inertia: wp.mat33) -> wp.mat33:
    """Convert a COM inertia tensor to its mass second-moment matrix.

    For positions r relative to the COM, the mass second moment is
    S = integral(r r^T dm). The inertia tensor satisfies I = trace(S) Id - S,
    where Id is the identity matrix. In 3D, trace(I) = 2 trace(S), so
    S = 0.5 trace(I) Id - I. Both tensors have units of kg*m^2.
    """
    return 0.5 * wp.trace(inertia) * wp.identity(3, float) - inertia


@wp.func
def affine_mass_multiply(
    mass: float,
    inertia: wp.mat33,
    v_com: wp.vec3,
    v_a0: wp.vec3,
    v_a1: wp.vec3,
    v_a2: wp.vec3,
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3]:
    """Return all four blocks of M v for v = [v_com, v_a0, v_a1, v_a2]."""
    moment = inertia_to_second_moment(inertia)

    return (
        mass * v_com,
        moment[0, 0] * v_a0 + moment[0, 1] * v_a1 + moment[0, 2] * v_a2,
        moment[1, 0] * v_a0 + moment[1, 1] * v_a1 + moment[1, 2] * v_a2,
        moment[2, 0] * v_a0 + moment[2, 1] * v_a1 + moment[2, 2] * v_a2,
    )


@wp.kernel
def eval_affine_orthogonality_kernel(
    x: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_flags: wp.array[int],
    alpha: float,
    dt: float,
    # outputs
    rhs: wp.array[wp.vec3],
    hessian_diag: wp.array[wp.mat33],
    scaled_rotation: wp.array[wp.mat33],
):
    """Accumulate the SO(3) penalty force and prepare its normal-space tangent.

    E(A) = k/2 min_{R in SO(3)} ||A - R||_F^2, k = alpha * m / dt^2.
    With R the closest proper rotation, P_R(X) = R sym(R^T X).
    The force is -k P_R(A - R), removing tangential SVD roundoff.
    Use H(X) = k P_R(X), a positive-semidefinite approximation that is
    exact at A = R and annihilates infinitesimal rotations R Omega.
    It is not the exact Hessian of the reduced energy away from SO(3).

    For columns r_i of R, H_ii = k/2 (Id + r_i r_i^T) and
    H_ij = k/2 r_j r_i^T for i != j. Cache B = sqrt(k/2) R so that
    off-diagonal products use only b_j dot(b_i, x_j).
    Overwrite constraint-only diagonal blocks and cache each nonlinear
    iteration; the caller combines these blocks with the mass diagonal.
    """
    body = wp.tid()
    start = 4 * body
    hessian_diag[start] = wp.mat33(0.0)
    if alpha <= 0.0 or body_mass[body] <= 0.0 or (body_flags[body] & (BodyFlags.DYNAMIC | BodyFlags.KINEMATIC)) == 0:
        hessian_diag[start + 1] = wp.mat33(0.0)
        hessian_diag[start + 2] = wp.mat33(0.0)
        hessian_diag[start + 3] = wp.mat33(0.0)
        scaled_rotation[body] = wp.mat33(0.0)
        return

    affine = wp.matrix_from_cols(x[start + 1], x[start + 2], x[start + 3])
    # Warp's signed SVD has det(U) = det(V) = +1, including for reflections.
    U, _, V = wp.svd3(affine)
    rotation = wp.quat_to_matrix(wp.normalize(wp.quat_from_matrix(U @ wp.transpose(V))))
    residual_local = wp.transpose(rotation) @ (affine - rotation)
    normal_residual = rotation @ (0.5 * (residual_local + wp.transpose(residual_local)))
    stiffness = alpha * body_mass[body] / (dt * dt)
    force_columns = wp.transpose(-stiffness * normal_residual)
    scaled = wp.sqrt(0.5 * stiffness) * rotation
    scaled_rotation[body] = scaled
    scaled_columns = wp.transpose(scaled)

    rhs[start + 1] += force_columns[0]
    rhs[start + 2] += force_columns[1]
    rhs[start + 3] += force_columns[2]

    hessian_diag[start + 1] = 0.5 * stiffness * wp.identity(3, float) + wp.outer(scaled_columns[0], scaled_columns[0])
    hessian_diag[start + 2] = 0.5 * stiffness * wp.identity(3, float) + wp.outer(scaled_columns[1], scaled_columns[1])
    hessian_diag[start + 3] = 0.5 * stiffness * wp.identity(3, float) + wp.outer(scaled_columns[2], scaled_columns[2])


@wp.kernel
def affine_off_diag_multiply_kernel(
    x: wp.array[wp.vec3],
    mass_off_diag: wp.array[wp.vec3],
    scaled_rotation: wp.array[wp.mat33],
    # outputs
    Ax: wp.array[wp.vec3],
):
    body = wp.tid()
    start = 4 * body
    coupling = mass_off_diag[body]

    x1 = x[start + 1]
    x2 = x[start + 2]
    x3 = x[start + 3]
    scaled_columns = wp.transpose(scaled_rotation[body])
    b1 = scaled_columns[0]
    b2 = scaled_columns[1]
    b3 = scaled_columns[2]

    # The assembled matrix already contains m / dt^2 and S_ii / dt^2.
    # Add only cross-direction mass and SO(3) terms; diagonal blocks are separate.
    Ax[start + 0] = wp.vec3(0.0)
    Ax[start + 1] = coupling[0] * x2 + coupling[1] * x3 + b2 * wp.dot(b1, x2) + b3 * wp.dot(b1, x3)
    Ax[start + 2] = coupling[0] * x1 + coupling[2] * x3 + b1 * wp.dot(b2, x1) + b3 * wp.dot(b2, x3)
    Ax[start + 3] = coupling[1] * x1 + coupling[2] * x2 + b1 * wp.dot(b3, x1) + b2 * wp.dot(b3, x2)


@wp.kernel
def body_to_affine_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_inv_inertia: wp.array[wp.mat33],
    body_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    dt: float,
    # outputs
    x: wp.array[wp.vec3],
    x_inertia: wp.array[wp.vec3],
):
    body = wp.tid()
    start = 4 * body
    pose = body_q[body]

    # Use semi-implicit rigid-body integration to define the inertial target pose.
    next_pose, _ = integrate_rigid_body(
        pose,
        body_qd[body],
        body_f[body],
        body_com[body],
        body_inertia[body],
        body_inv_mass[body],
        body_inv_inertia[body],
        gravity[body_world[body]],
        0.0,  # angular_damping
        dt,
    )

    # Anchor the affine map at the COM, accounting for the local body-frame offset.
    center = wp.transform_point(pose, body_com[body])
    next_center = wp.transform_point(next_pose, body_com[body])

    # Layout: [COM, A[:, 0], A[:, 1], A[:, 2]], all expressed in the world frame.
    # Store directions without translation to preserve precision far from the origin.
    x[start + 0] = center
    x[start + 1] = wp.transform_vector(pose, wp.vec3(1.0, 0.0, 0.0))
    x[start + 2] = wp.transform_vector(pose, wp.vec3(0.0, 1.0, 0.0))
    x[start + 3] = wp.transform_vector(pose, wp.vec3(0.0, 0.0, 1.0))

    # Keep inertial targets in the same affine coordinates as the current state.
    x_inertia[start + 0] = next_center
    x_inertia[start + 1] = wp.transform_vector(next_pose, wp.vec3(1.0, 0.0, 0.0))
    x_inertia[start + 2] = wp.transform_vector(next_pose, wp.vec3(0.0, 1.0, 0.0))
    x_inertia[start + 3] = wp.transform_vector(next_pose, wp.vec3(0.0, 0.0, 1.0))


@wp.kernel
def affine_to_body_kernel(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_inertia: wp.array[wp.mat33],
    dt: float,
    body_mass: wp.array[float],
    body_flags: wp.array[int],
    # outputs
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    body = wp.tid()

    # Bodies outside the ABD solve are owned by another solver or caller.
    if body_mass[body] <= 0.0 or (body_flags[body] & BodyFlags.DYNAMIC) == 0:
        return

    start = 4 * body
    center = x[start]
    linear = v[start]
    affine = wp.matrix_from_cols(x[start + 1], x[start + 2], x[start + 3])
    affine_velocity = wp.matrix_from_cols(v[start + 1], v[start + 2], v[start + 3])

    # The COM second moment defines the mass metric for the rigid fit.
    moment = inertia_to_second_moment(body_inertia[body])
    moment_scale = wp.trace(moment)
    if moment_scale > 0.0:
        moment = moment / moment_scale
    else:
        # Bodies without inertia still need a geometric pose fit.
        moment = wp.identity(3, float)

    # Warp's signed SVD returns proper rotations U and V, even for reflections.
    U, _, V = wp.svd3(affine @ moment)
    rotation = wp.normalize(wp.quat_from_matrix(U @ wp.transpose(V)))

    # Recover the previous configuration from the finite-difference velocity.
    previous_affine = affine - dt * affine_velocity
    U_previous, _, V_previous = wp.svd3(previous_affine @ moment)
    previous_rotation = wp.normalize(wp.quat_from_matrix(U_previous @ wp.transpose(V_previous)))

    # Invert integrate_rigid_body's normalized Euler quaternion update:
    # delta_rotation = normalize(quat(0.5 * dt * angular, 1.0)).
    delta_rotation = rotation * wp.quat_inverse(previous_rotation)
    angular = wp.vec3(delta_rotation[0], delta_rotation[1], delta_rotation[2]) * (2.0 / (dt * delta_rotation[3]))

    body_q[body] = wp.transform(center - wp.quat_rotate(rotation, body_com[body]), rotation)
    body_qd[body] = wp.spatial_vector(linear, angular)


class AffineBody:
    """Represent model bodies by a position and three basis vectors in caller-owned buffers.

    Each body occupies four consecutive vectors: its world-space center of
    mass followed by the three columns of its affine matrix. The columns
    encode directions, independent of translation.
    Inertial targets are predicted using rigid-body integration.
    Rigid poses are recovered by mass-weighted least squares, using an isotropic
    metric for bodies with zero inertia. Angular velocities invert the rigid
    integrator's normalized Euler quaternion update.
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        # Zero until orthogonality is evaluated; refreshed once per nonlinear iteration.
        self._orthogonality_scaled_rotation = wp.zeros(model.body_count, dtype=wp.mat33, device=model.device)

    def eval_orthogonality(
        self,
        x: wp.array[wp.vec3],
        rhs: wp.array[wp.vec3],
        hessian_diag: wp.array[wp.mat33],
        alpha: float,
        dt: float,
    ) -> None:
        """Accumulate SO(3) forces and overwrite the constraint-only diagonal blocks.

        Pass affine views of length 4 * body_count for x, rhs, and hessian_diag.
        Call once per nonlinear iteration before off_diag_multiply, with a
        positive dt and nonnegative alpha. The cached tangent remains fixed
        throughout the linear solve. An alpha of zero clears the constraint.
        """
        wp.launch(
            eval_affine_orthogonality_kernel,
            dim=self.model.body_count,
            inputs=[x, self.model.body_mass, self.model.body_flags, alpha, dt],
            outputs=[rhs, hessian_diag, self._orthogonality_scaled_rotation],
            device=self.model.device,
        )

    def off_diag_multiply(self, x: wp.array[wp.vec3], mass_off_diag: wp.array[wp.vec3], Ax: wp.array[wp.vec3]) -> None:
        """Write the affine off-diagonal contribution to A x.

        Combine (M - diag(M)) x / dt^2, using prepared (S_01, S_02, S_12)
        / dt^2 coefficients, with the cached SO(3) off-diagonal tangent.
        """
        wp.launch(
            affine_off_diag_multiply_kernel,
            dim=self.model.body_count,
            inputs=[x, mass_off_diag, self._orthogonality_scaled_rotation],
            outputs=[Ax],
            device=self.model.device,
        )

    def to_internal_state(self, state: State, x: wp.array[wp.vec3], x_inertia: wp.array[wp.vec3], dt: float) -> None:
        """Write current affine coordinates and inertial targets for a positive time step."""
        wp.launch(
            body_to_affine_kernel,
            dim=self.model.body_count,
            inputs=[
                state.body_q,
                state.body_qd,
                state.body_f,
                self.model.body_com,
                self.model.body_inertia,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
                self.model.body_world,
                self.model.gravity,
                dt,
            ],
            outputs=[x, x_inertia],
            device=self.model.device,
        )

    def from_internal_state(self, state_out: State, x: wp.array[wp.vec3], v: wp.array[wp.vec3], dt: float) -> None:
        """Fit the rigid states for bodies updated by the ABD solve."""
        wp.launch(
            affine_to_body_kernel,
            dim=self.model.body_count,
            inputs=[
                x,
                v,
                self.model.body_com,
                self.model.body_inertia,
                dt,
                self.model.body_mass,
                self.model.body_flags,
            ],
            outputs=[state_out.body_q, state_out.body_qd],
            device=self.model.device,
        )
