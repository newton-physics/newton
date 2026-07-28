# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..core.types import Devicelike
from .articulation import eval_jacobian, eval_mass_matrix

if TYPE_CHECKING:
    from .model import Model
    from .state import State


@wp.kernel
def _compute_body_q_com_kernel(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    # output
    body_q_com: wp.array[wp.transform],
):
    """Post-compose ``body_q`` with the local CoM offset to produce the
    body-CoM-anchored transform consumed by :func:`eval_rigid_id`."""
    i = wp.tid()
    body_q_com[i] = body_q[i] * wp.transform(body_com[i], wp.quat_identity())


class _InverseDynamicsScratchBuffer:
    """Internal scratch buffers for :func:`eval_inverse_dynamics_passive`.

    Holds the RNEA per-body and per-DOF arrays, the mass-matrix Jacobian
    scratch, and the constant-zero inputs that the compensation passes
    feed into :func:`eval_rigid_tau` and :func:`eval_rigid_id`. All buffers
    are sized for the topology supplied at construction time; rebuilding
    the model with a different body, joint, articulation, or DOF count
    requires a new instance.

    Instances are allocated for one evaluation and released when it returns.
    All attributes are internal implementation details.
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
        self.body_I_m = wp.empty(bc, dtype=wp.spatial_matrix, device=device)
        self.body_q_com = wp.empty(bc, dtype=wp.transform, device=device)
        self.joint_qd_internal = wp.empty(jdc, dtype=wp.float32, device=device)
        self.body_qd_fk = wp.empty(bc, dtype=wp.spatial_vector, device=device)
        self.body_solve_origin = wp.zeros(bc, dtype=wp.vec3, device=device)
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


def _rnea_compensation_pass(
    model: Model,
    state: State,
    scratch: _InverseDynamicsScratchBuffer,
    joint_qd: wp.array[wp.float32],
    gravity: wp.array[wp.vec3],
    tau_out: wp.array[wp.float32],
    mask: wp.array[bool] | None = None,
) -> None:
    """Run one RNEA pass (forward + backward) and write a single bias
    force (``g(q)``, ``C(q, q_dot)*q_dot``, or their sum) into
    ``tau_out``, reusing the buffers on ``scratch``.

    Requires ``state.body_q`` to be consistent with ``state.joint_q``;
    callers must invoke :func:`~newton.eval_fk` (or otherwise update
    ``state.body_q``) before calling this.

    With ``qdd = 0`` implicit in :func:`eval_rigid_id` and the result
    sign-flipped to match the standard convention, the output is
    ``g(q) = ∂U/∂q`` when ``joint_qd`` is zero (gravity only),
    ``C(q, q_dot)*q_dot`` when ``gravity`` is zero (Coriolis only), or
    their sum when both are non-zero.

    When ``mask`` is provided, only articulations whose corresponding
    entry is ``True`` contribute to ``tau_out``; entries for unselected
    DOFs are zero.
    """
    # Lazy import: featherstone/kernels.py imports from newton._src.sim, so a
    # top-level import here would create a circular import during sim package
    # initialization.
    from ..solvers.featherstone.kernels import (  # noqa: PLC0415
        compute_spatial_inertia,
        convert_free_distance_joint_f_internal_to_public,
        convert_free_distance_joint_qd_public_to_internal,
        eval_rigid_id,
        eval_rigid_tau,
    )

    device = model.device
    bc = model.body_count

    # ``eval_rigid_tau`` reads body_ft_s[child] before accumulating into
    # body_ft_s[parent], so this buffer must start zero on every pass. The
    # other RNEA scratch arrays are fully overwritten by their producing
    # kernels (jcalc_motion, compute_link_velocity, jcalc_tau) and don't
    # need pre-zeroing.
    scratch.body_ft_s.zero_()

    # Zero ``tau_out`` so unselected DOFs end up as 0 when ``mask`` is set
    # (the masked ``eval_rigid_tau`` skips those articulations entirely,
    # and the masked convert kernel does not touch their entries either).
    # When ``mask`` is None this is redundant but cheap.
    tau_out.zero_()

    # Body spatial inertias and the CoM-anchored body transforms consumed
    # by ``eval_rigid_id``. ``state.body_q`` is reused directly (already
    # populated by the caller's eval_fk), so we only need to post-compose
    # it with the local CoM offset to produce body_q_com.
    wp.launch(
        compute_spatial_inertia,
        dim=bc,
        inputs=[model.body_inertia, model.body_mass],
        outputs=[scratch.body_I_m],
        device=device,
    )
    wp.launch(
        _compute_body_q_com_kernel,
        dim=bc,
        inputs=[state.body_q, model.body_com],
        outputs=[scratch.body_q_com],
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
            state.body_q,
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
            mask,
            model.articulation_start,
            model.articulation_end,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            state.joint_q,
            scratch.joint_qd_internal,
            model.joint_axis,
            model.joint_dof_dim,
            scratch.body_I_m,
            state.body_q,
            scratch.body_q_com,
            model.joint_X_p,
            model.body_world,
            gravity,
        ],
        outputs=[
            scratch.body_qd_fk,
            scratch.joint_S_s,
            scratch.body_solve_origin,
            scratch.body_I_s,
            scratch.body_v_s,
            scratch.body_f_s,
            scratch.body_a_s,
        ],
        device=device,
    )

    # RNEA backward pass: project body wrenches to joint torques. Pure
    # compensation means zero PD gains, zero limit gains, zero applied force,
    # and zero external body force — jcalc_tau collapses to -dot(S, f_total).
    wp.launch(
        eval_rigid_tau,
        dim=model.articulation_count,
        inputs=[
            mask,
            model.articulation_start,
            model.articulation_end,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_target_q_start,
            model.joint_dof_dim,
            scratch.zeros_dof,  # joint_target_q
            scratch.zeros_dof,  # joint_target_qd
            state.joint_q,
            scratch.joint_qd_internal,
            scratch.zeros_dof,  # joint_f
            scratch.zeros_dof,  # joint_target_ke
            scratch.zeros_dof,  # joint_target_kd
            model.joint_limit_lower,
            model.joint_limit_upper,
            scratch.zeros_dof,  # joint_limit_ke
            scratch.zeros_dof,  # joint_limit_kd
            scratch.zeros_dof,  # joint_damping
            scratch.joint_S_s,
            scratch.body_q_com,
            scratch.body_solve_origin,
            scratch.body_f_s,
            scratch.zeros_body,  # body_f_ext
        ],
        outputs=[scratch.body_ft_s, tau_out],
        device=device,
    )

    # Convert output tau_out from RNEA's internal body-origin convention to
    # Newton's documented free-joint joint_f convention (wrench at body CoM)
    # and flip the RNEA sign so tau_out stores the standard ``+g(q)`` /
    # ``+C(q, q_dot)*q_dot`` directly. Subtracts the spatial-vs-classical
    # acceleration bias, shifts the wrench from body origin to body CoM,
    # then negates every per-DOF entry. Non-free / non-distance joints
    # skip the corrections (their joint_f is reference-point-invariant)
    # but still get the sign flip.
    wp.launch(
        convert_free_distance_joint_f_internal_to_public,
        dim=model.joint_count,
        inputs=[
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_qd_start,
            model.joint_articulation,
            mask,
            model.joint_X_p,
            state.body_q,
            scratch.body_q_com,
            model.body_mass,
            joint_qd,
        ],
        outputs=[tau_out],
        device=device,
    )


def _compute_gravity_force(
    model: Model,
    state: State,
    gravity_force: wp.array[wp.float32],
    scratch: _InverseDynamicsScratchBuffer,
    mask: wp.array[bool] | None = None,
) -> None:
    """Compute the gravity force ``g(q) = ∂U/∂q``.

    Runs RNEA with joint velocities zeroed and gravity set to
    :attr:`Model.gravity`, producing the joint-space force needed to hold the
    articulation static under gravity.

    When ``mask`` is provided, only the selected articulations contribute;
    DOFs belonging to unselected articulations are zero.
    """
    _rnea_compensation_pass(
        model,
        state,
        scratch,
        scratch.zeros_dof,
        model.gravity,
        gravity_force,
        mask=mask,
    )


def _compute_coriolis_force(
    model: Model,
    state: State,
    coriolis_force: wp.array[wp.float32],
    scratch: _InverseDynamicsScratchBuffer,
    mask: wp.array[bool] | None = None,
) -> None:
    """Compute the Coriolis force ``C(q, q_dot)*q_dot``.

    Runs RNEA with the current joint velocities and gravity zeroed, producing
    the Coriolis + centrifugal force in joint space.

    When ``mask`` is provided, only the selected articulations contribute;
    DOFs belonging to unselected articulations are zero.
    """
    _rnea_compensation_pass(
        model,
        state,
        scratch,
        state.joint_qd,
        scratch.zero_gravity,
        coriolis_force,
        mask=mask,
    )


def eval_inverse_dynamics_passive(
    model: Model,
    state: State,
    *,
    mass_matrix: wp.array3d[wp.float32] | None = None,
    gravity_force: wp.array[wp.float32] | None = None,
    coriolis_force: wp.array[wp.float32] | None = None,
    mask: wp.array[bool] | None = None,
) -> None:
    """Compute passive inverse-dynamics quantities for articulated systems.

    Each non-``None`` output is computed independently. Omitted outputs are
    not computed. Callers allocate the requested output arrays:

    * ``mass_matrix`` ← the joint-space mass matrix ``M(q)`` [kg, kg·m, or
      kg·m^2, depending on the joint types of the row/column DOFs];
    * ``gravity_force`` ← the gravity force ``g(q) = ∂U/∂q`` [N or N·m,
      depending on joint type], where ``U(q)`` is the system's
      gravitational potential energy ``sum_i -m_i * g . x_com_i``. This is
      the joint-space force that holds the articulation static under
      gravity;
    * ``coriolis_force`` ← the Coriolis + centrifugal force
      ``C(q, q_dot)*q_dot`` [N or N·m, depending on joint type].

    All three quantities follow the standard manipulator-equation convention
    ``tau = M(q)*joint_qdd + C(q,q_dot)*q_dot + g(q)``.

    Requires ``state.body_q`` to be consistent with ``state.joint_q``;
    callers must invoke :func:`~newton.eval_fk` (or otherwise update
    ``state.body_q``) before this function.

    Note:
        Inverse dynamics considers only the kinematic tree. As a consequence,
        loop-closure joints (``EqType.CONNECT``, ``EqType.WELD``, ``EqType.JOINT``)
        play no role in the inverse dynamics evaluation.

        :attr:`~newton.JointType.CABLE` joints are not supported because they
        do not define generalized coordinates or a motion subspace for this
        inverse-dynamics formulation.

    .. experimental::

    Args:
        model: Model providing articulation topology and inertial parameters.
        state: State providing the current generalized coordinates and velocities.
            ``state.body_q`` must already reflect ``state.joint_q``.
        mass_matrix: Optional output for the joint-space mass matrix,
            shape ``(model.articulation_count, model.max_dofs_per_articulation,
            model.max_dofs_per_articulation)``, dtype float. The padded rows
            and columns beyond each articulation's DOF count are zero.
        gravity_force: Optional output for ``g(q) = ∂U/∂q`` [N or N·m,
            depending on joint type], shape ``(model.joint_dof_count,)``,
            dtype float.
        coriolis_force: Optional output for ``C(q, q_dot)*q_dot`` [N or N·m,
            depending on joint type], shape ``(model.joint_dof_count,)``,
            dtype float.
        mask: Optional ``wp.array[bool]`` of shape
            ``(articulation_count,)`` selecting which articulations to
            compute. Entries belonging to unselected articulations are
            zero in the output buffers (mirroring
            :func:`~newton.eval_mass_matrix`'s mask convention). If
            ``None``, all articulations are computed.

    Raises:
        ValueError: If the model contains a :attr:`~newton.JointType.CABLE`
            joint, no outputs are requested, or an output or mask has an
            unexpected shape.
    """
    if model._has_cable_joints:  # pyright: ignore[reportPrivateUsage]
        raise ValueError("eval_inverse_dynamics_passive() does not support JointType.CABLE joints.")

    if mass_matrix is None and gravity_force is None and coriolis_force is None:
        raise ValueError("At least one inverse-dynamics output must be provided.")

    if mass_matrix is not None:
        expected_shape = (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation)
        if mass_matrix.shape != expected_shape:
            raise ValueError(f"mass_matrix has shape {mass_matrix.shape}, expected {expected_shape}.")

    expected_dof_shape = (model.joint_dof_count,)
    for name, array in (("gravity_force", gravity_force), ("coriolis_force", coriolis_force)):
        if array is not None and array.shape != expected_dof_shape:
            raise ValueError(f"{name} has shape {array.shape}, expected {expected_dof_shape}.")

    expected_mask_shape = (model.articulation_count,)
    if mask is not None and mask.shape != expected_mask_shape:
        raise ValueError(f"mask has shape {mask.shape}, expected {expected_mask_shape}.")

    scratch = _InverseDynamicsScratchBuffer(
        body_count=model.body_count,
        articulation_count=model.articulation_count,
        joint_dof_count=model.joint_dof_count,
        max_dofs_per_articulation=model.max_dofs_per_articulation,
        max_joints_per_articulation=model.max_joints_per_articulation,
        world_count=model.world_count,
        device=model.device,
    )

    if mass_matrix is not None:
        # eval_jacobian zeros scratch.J internally; jcalc_motion_subspace
        # fully overwrites every DOF of scratch.joint_S_s;
        # compute_body_spatial_inertia fully overwrites every body's
        # scratch.body_I_s; eval_mass_matrix zeros mass_matrix internally.
        # No pre-zeroing required here.
        eval_jacobian(model, state, J=scratch.J, joint_S_s=scratch.joint_S_s, mask=mask)
        eval_mass_matrix(
            model,
            state,
            H=mass_matrix,
            J=scratch.J,
            body_I_s=scratch.body_I_s,
            joint_S_s=scratch.joint_S_s,
            mask=mask,
        )

    if gravity_force is not None:
        _compute_gravity_force(model, state, gravity_force, scratch, mask=mask)

    if coriolis_force is not None:
        _compute_coriolis_force(model, state, coriolis_force, scratch, mask=mask)
