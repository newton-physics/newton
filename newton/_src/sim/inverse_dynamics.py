# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import warp as wp

from ..core.types import Devicelike
from .articulation import eval_jacobian, eval_mass_matrix

if TYPE_CHECKING:
    from .model import Model
    from .state import State


@wp.kernel
def _negate_array_kernel(arr: wp.array[wp.float32]):
    i = wp.tid()
    arr[i] = -arr[i]


class InverseDynamicsScratchBuffer:
    """Internal scratch buffers reused across calls to :func:`eval_inverse_dynamics`.

    Holds the RNEA per-body and per-DOF arrays, the mass-matrix Jacobian
    scratch, and the constant-zero inputs that the compensation passes
    feed into :func:`eval_rigid_tau` and :func:`eval_rigid_id`. All buffers
    are sized for the topology supplied at construction time; rebuilding
    the model with a different body, joint, articulation, or DOF count
    requires a new instance.
    """

    def __init__(
        self,
        body_count: int,
        articulation_count: int,
        joint_dof_count: int,
        max_dofs_per_articulation: int,
        max_joints_per_articulation: int,
        world_count: int,
        device: Devicelike | None = None,
    ):
        """Allocate scratch buffers for inverse dynamics.

        Args:
            body_count: Total number of rigid bodies across all articulations
                (matches :attr:`Model.body_count`).
            articulation_count: Number of articulations (matches
                :attr:`Model.articulation_count`).
            joint_dof_count: Total number of joint DOFs across all
                articulations (matches :attr:`Model.joint_dof_count`).
            max_dofs_per_articulation: Per-articulation DOF count (inclusive
                of floating-base root DOFs, if any). Matches
                :attr:`Model.max_dofs_per_articulation`.
            max_joints_per_articulation: Maximum number of joints in any
                articulation, used to size the per-articulation Jacobian
                scratch. Matches :attr:`Model.max_joints_per_articulation`.
            world_count: Number of simulation worlds, used to size the
                constant-zero gravity vector consumed by the Coriolis
                compensation pass. Matches :attr:`Model.world_count`.
            device: Warp device on which the buffers are allocated.
        """
        bc = body_count
        ac = articulation_count
        jdc = joint_dof_count
        max_dofs = max_dofs_per_articulation
        max_links = max_joints_per_articulation

        # RNEA scratch (rewritten by every compensation pass).
        self.body_X_com = wp.empty(bc, dtype=wp.transform, device=device)
        self.body_I_m = wp.empty(bc, dtype=wp.spatial_matrix, device=device)
        self.body_q = wp.empty(bc, dtype=wp.transform, device=device)
        self.body_q_com = wp.empty(bc, dtype=wp.transform, device=device)
        self.joint_qd_internal = wp.empty(jdc, dtype=wp.float32, device=device)
        self.joint_S_s = wp.empty(jdc, dtype=wp.spatial_vector, device=device)
        self.body_I_s = wp.empty(bc, dtype=wp.spatial_matrix, device=device)
        self.body_v_s = wp.empty(bc, dtype=wp.spatial_vector, device=device)
        self.body_f_s = wp.empty(bc, dtype=wp.spatial_vector, device=device)
        self.body_a_s = wp.empty(bc, dtype=wp.spatial_vector, device=device)
        self.body_ft_s = wp.empty(bc, dtype=wp.spatial_vector, device=device)

        # Reused as the eval_mass_matrix Jacobian scratch.
        self.J = wp.empty((ac, max_links * 6, max_dofs), dtype=wp.float32, device=device)

        # Constant-zero inputs (allocated once, never written).
        self.zeros_dof = wp.zeros(jdc, dtype=wp.float32, device=device)
        self.zeros_body = wp.zeros(bc, dtype=wp.spatial_vector, device=device)
        self.zero_gravity = wp.zeros(world_count, dtype=wp.vec3, device=device)


class InverseDynamics:
    """Inverse dynamics quantities for a batch of articulated rigid-body systems."""

    class EvalType(IntEnum):
        """Bitmask flags selecting which quantities :class:`~newton.InverseDynamics` should compute.

        Flags can be combined with bitwise-or to request multiple quantities
        simultaneously; :attr:`ALL` is the union of all individual flags.
        """

        MASS_MATRIX = 1 << 0
        """Compute the joint-space mass matrix M(q)."""

        GRAVITY_COMPENSATION_FORCE = 1 << 1
        """Compute the standard manipulator-equation gravity bias ``g(q) = ∂U/∂q``."""

        CORIOLIS_COMPENSATION_FORCE = 1 << 2
        """Compute the standard manipulator-equation Coriolis bias ``C(q, q_dot)*q_dot``."""

        COMPENSATION_FORCES = GRAVITY_COMPENSATION_FORCE | CORIOLIS_COMPENSATION_FORCE
        """Compute both manipulator-equation bias terms ``g(q)`` and ``C(q, q_dot)*q_dot``."""

        ALL = MASS_MATRIX | GRAVITY_COMPENSATION_FORCE | CORIOLIS_COMPENSATION_FORCE
        """Compute the mass matrix and both compensation forces."""

    def __init__(
        self,
        articulation_count: int,
        joint_dof_count: int,
        max_dofs_per_articulation: int,
        device: Devicelike | None = None,
    ):
        """Allocate output buffers for inverse dynamics.

        The mass matrix is stored in the shape expected by
        :func:`~newton.eval_mass_matrix`:
        ``(articulation_count, max_dofs_per_articulation, max_dofs_per_articulation)``.
        ``max_dofs_per_articulation`` already includes any 6 DOFs contributed by a
        floating-base root joint.

        The compensation-force buffers share the flat joint-space layout that
        :func:`~newton.Model.joint_qd` uses.

        Internal scratch lives on a separate
        :class:`InverseDynamicsScratchBuffer`, which is passed alongside
        :class:`InverseDynamics` into :func:`eval_inverse_dynamics`.

        Args:
            articulation_count: Number of articulations (matches
                :attr:`Model.articulation_count`).
            joint_dof_count: Total number of joint DOFs across all articulations
                (matches :attr:`Model.joint_dof_count`).
            max_dofs_per_articulation: Per-articulation DOF count (inclusive of
                floating-base root DOFs, if any). Matches
                :attr:`Model.max_dofs_per_articulation`.
            device: Warp device on which the buffers are allocated.
        """
        ac = articulation_count
        jdc = joint_dof_count
        max_dofs = max_dofs_per_articulation

        self.mass_matrix: wp.array3d[wp.float32] = wp.zeros(
            (ac, max_dofs, max_dofs), dtype=wp.float32, device=device
        )
        """Joint-space mass matrix M(q) [kg, kg·m, or kg·m^2, depending on the joint types of the row/column DOFs], shape (articulation_count, max_dofs_per_articulation, max_dofs_per_articulation), dtype float."""

        self.gravity_compensation_force: wp.array[wp.float32] = wp.zeros(jdc, dtype=wp.float32, device=device)
        """Standard manipulator-equation gravity bias ``g(q) = ∂U/∂q`` [N or N·m, depending on joint type], shape (joint_dof_count,), dtype float.
        Equivalently, the joint-space force a controller must apply to hold the articulation static under gravity (feed-forward gravity compensation)."""

        self.coriolis_compensation_force: wp.array[wp.float32] = wp.zeros(jdc, dtype=wp.float32, device=device)
        """Standard manipulator-equation Coriolis + centrifugal bias ``C(q, q_dot)*q_dot`` [N or N·m, depending on joint type], shape (joint_dof_count,), dtype float.
        Equivalently, the joint-space force a controller must apply to maintain the current ``joint_qd`` at zero acceleration in the absence of gravity (feed-forward Coriolis compensation)."""

        self.tau: wp.array[wp.float32] = wp.zeros(jdc, dtype=wp.float32, device=device)
        """Inverse-dynamics joint force ``tau = M(q)*qddot + C(q, q_dot)*q_dot + g(q)`` [N or N·m, depending on joint type], shape (joint_dof_count,), dtype float.
        Typically populated by :func:`~newton.eval_inverse_dynamics_force` from the other buffers in this container plus a user-supplied ``qddot``."""


def _rnea_compensation_pass(
    model: Model,
    state: State,
    scratch: InverseDynamicsScratchBuffer,
    joint_qd: wp.array[wp.float32],
    gravity: wp.array[wp.vec3],
    tau_out: wp.array[wp.float32],
) -> None:
    """Run one RNEA pass (forward + backward) and write the standard
    manipulator-equation bias term into ``tau_out``, reusing the buffers
    on ``scratch``.

    With ``qdd = 0`` implicit in :func:`eval_rigid_id` and the result
    sign-flipped to match the standard convention, the output is
    ``g(q) = ∂U/∂q`` when ``joint_qd`` is zero (gravity only),
    ``C(q, q_dot)*q_dot`` when ``gravity`` is zero (Coriolis only), or
    their sum when both are non-zero.
    """
    # Lazy import: featherstone/kernels.py imports from newton._src.sim, so a
    # top-level import here would create a circular import during sim package
    # initialization.
    from ..solvers.featherstone.kernels import (  # noqa: PLC0415
        compute_com_transforms,
        compute_spatial_inertia,
        convert_free_distance_joint_f_internal_to_public,
        convert_free_distance_joint_qd_public_to_internal,
        eval_rigid_fk,
        eval_rigid_id,
        eval_rigid_tau,
    )

    device = model.device
    bc = model.body_count

    # ``eval_rigid_tau`` reads body_ft_s[child] before atomic-adding into
    # body_ft_s[parent], so this buffer must start zero on every pass. The
    # other RNEA scratch arrays are fully overwritten by their producing
    # kernels (jcalc_motion, compute_link_velocity, jcalc_tau) and don't
    # need pre-zeroing.
    scratch.body_ft_s.zero_()

    # Body-local CoM transforms and spatial inertias.
    wp.launch(
        compute_com_transforms,
        dim=bc,
        inputs=[model.body_com],
        outputs=[scratch.body_X_com],
        device=device,
    )
    wp.launch(
        compute_spatial_inertia,
        dim=bc,
        inputs=[model.body_inertia, model.body_mass],
        outputs=[scratch.body_I_m],
        device=device,
    )

    # Forward kinematics into local buffers (don't mutate state.body_q).
    wp.launch(
        eval_rigid_fk,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            state.joint_q,
            model.joint_X_p,
            model.joint_X_c,
            scratch.body_X_com,
            model.joint_axis,
            model.joint_dof_dim,
        ],
        outputs=[scratch.body_q, scratch.body_q_com],
        device=device,
    )

    # Convert input joint_qd from Newton's documented free-joint convention
    # (linear = child-COM velocity in parent frame) to RNEA's internal
    # body-origin convention (linear = body-origin velocity in parent frame).
    # Non-free / non-distance joints are copied through unchanged.
    wp.launch(
        convert_free_distance_joint_qd_public_to_internal,
        dim=model.joint_count,
        inputs=[
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_qd_start,
            model.joint_X_p,
            scratch.body_q,
            model.body_com,
            joint_qd,
        ],
        outputs=[scratch.joint_qd_internal],
        device=device,
    )

    # RNEA forward pass: body bias wrenches in the spatial frame.
    wp.launch(
        eval_rigid_id,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_qd_start,
            scratch.joint_qd_internal,
            model.joint_axis,
            model.joint_dof_dim,
            scratch.body_I_m,
            scratch.body_q,
            scratch.body_q_com,
            model.joint_X_p,
            model.body_world,
            gravity,
        ],
        outputs=[
            scratch.joint_S_s,
            scratch.body_I_s,
            scratch.body_v_s,
            scratch.body_f_s,
            scratch.body_a_s,
        ],
        device=device,
    )

    # RNEA backward pass: project body wrenches to joint torques. Pure
    # compensation means zero PD gains, zero limit gains, zero applied force,
    # and zero external body force — jcalc_tau collapses to -dot(S, body_f_s).
    wp.launch(
        eval_rigid_tau,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_dof_dim,
            scratch.zeros_dof,  # joint_target_pos
            scratch.zeros_dof,  # joint_target_vel
            state.joint_q,
            scratch.joint_qd_internal,
            scratch.zeros_dof,  # joint_f
            scratch.zeros_dof,  # joint_target_ke
            scratch.zeros_dof,  # joint_target_kd
            model.joint_limit_lower,
            model.joint_limit_upper,
            scratch.zeros_dof,  # joint_limit_ke
            scratch.zeros_dof,  # joint_limit_kd
            scratch.joint_S_s,
            scratch.body_f_s,
            scratch.zeros_body,  # body_f_ext
        ],
        outputs=[scratch.body_ft_s, tau_out],
        device=device,
    )

    # Convert output tau_out from RNEA's internal body-origin convention to
    # Newton's documented free-joint joint_f convention (wrench at body CoM).
    # Subtracts the spatial-vs-classical acceleration convention bias and
    # then shifts the wrench from body origin to body CoM. Non-free /
    # non-distance joints are unaffected.
    wp.launch(
        convert_free_distance_joint_f_internal_to_public,
        dim=model.joint_count,
        inputs=[
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_qd_start,
            model.joint_X_p,
            scratch.body_q,
            model.body_com,
            model.body_mass,
            joint_qd,
        ],
        outputs=[tau_out],
        device=device,
    )

    # ``eval_rigid_tau`` produces ``tau = -dot(S, body_f_s)``, which under the
    # RNEA bias setup (``a_base = -gravity``, ``qdd = 0``) returns the negation
    # of the standard manipulator-equation bias terms. Flip the sign so the
    # buffer stores the standard `+g(q)` / `+C(q, q_dot)*q_dot` directly.
    wp.launch(
        _negate_array_kernel,
        dim=tau_out.shape[0],
        inputs=[tau_out],
        device=device,
    )


def _compute_gravity_compensation_force(
    model: Model,
    state: State,
    inverse_dynamics: InverseDynamics,
    scratch: InverseDynamicsScratchBuffer,
) -> None:
    """Compute the gravity bias ``g(q) = ∂U/∂q`` into
    ``inverse_dynamics.gravity_compensation_force``.

    Runs RNEA with joint velocities zeroed and gravity set to
    :attr:`Model.gravity`, producing the joint-space force needed to hold the
    articulation static under gravity (feed-forward gravity compensation).
    """
    _rnea_compensation_pass(
        model,
        state,
        scratch,
        scratch.zeros_dof,
        model.gravity,
        inverse_dynamics.gravity_compensation_force,
    )


def _compute_coriolis_compensation_force(
    model: Model,
    state: State,
    inverse_dynamics: InverseDynamics,
    scratch: InverseDynamicsScratchBuffer,
) -> None:
    """Compute the Coriolis bias ``C(q, q_dot)*q_dot`` into
    ``inverse_dynamics.coriolis_compensation_force``.

    Runs RNEA with the current joint velocities and gravity zeroed, producing
    the standard manipulator-equation Coriolis + centrifugal joint-space
    bias term.
    """
    _rnea_compensation_pass(
        model,
        state,
        scratch,
        state.joint_qd,
        scratch.zero_gravity,
        inverse_dynamics.coriolis_compensation_force,
    )


def eval_inverse_dynamics(
    model: Model,
    state: State,
    eval_type: InverseDynamics.EvalType,
    inverse_dynamics: InverseDynamics,
    scratch: InverseDynamicsScratchBuffer,
) -> None:
    """Compute inverse dynamics quantities for an articulation.

    Depending on the flags in ``eval_type``, populates one or more of: the
    joint-space mass matrix ``M(q)`` [kg, kg·m, or kg·m^2, depending on the
    joint types of the row/column DOFs], the gravity bias
    ``g(q) = ∂U/∂q`` [N or N·m, depending on joint type], and the Coriolis
    bias ``C(q, q_dot)*q_dot`` [N or N·m, depending on joint type] into
    ``inverse_dynamics``. All three quantities follow the standard
    manipulator-equation convention ``tau = M(q)*qddot + C(q,q_dot)*q_dot
    + g(q)``.

    Args:
        model: Model providing articulation topology and inertial parameters.
        state: State providing the current generalized coordinates and velocities.
        eval_type: Bitmask selecting which quantities to compute.
        inverse_dynamics: Output container whose buffers are written in place.
        scratch: Pre-allocated scratch buffers reused across calls.
    """
    if eval_type & InverseDynamics.EvalType.MASS_MATRIX:
        expected_shape = (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation)
        if inverse_dynamics.mass_matrix.shape != expected_shape:
            raise ValueError(
                f"inverse_dynamics.mass_matrix has shape "
                f"{inverse_dynamics.mass_matrix.shape}, expected {expected_shape}."
            )
        # eval_jacobian zeros scratch.J internally; jcalc_motion_subspace
        # fully overwrites every DOF of scratch.joint_S_s;
        # compute_body_spatial_inertia fully overwrites every body's
        # scratch.body_I_s; eval_mass_matrix zeros mass_matrix internally.
        # No pre-zeroing required here.
        eval_jacobian(model, state, J=scratch.J, joint_S_s=scratch.joint_S_s)
        eval_mass_matrix(
            model,
            state,
            H=inverse_dynamics.mass_matrix,
            J=scratch.J,
            body_I_s=scratch.body_I_s,
            joint_S_s=scratch.joint_S_s,
        )

    if eval_type & InverseDynamics.EvalType.GRAVITY_COMPENSATION_FORCE:
        _compute_gravity_compensation_force(model, state, inverse_dynamics, scratch)

    if eval_type & InverseDynamics.EvalType.CORIOLIS_COMPENSATION_FORCE:
        _compute_coriolis_compensation_force(model, state, inverse_dynamics, scratch)
