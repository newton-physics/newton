# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerDifferentialIKModelFree — differential-kinematics control with a
caller-supplied Jacobian and tool pose.

Every 1-D per-DOF port is compact: one entry per controlled DOF, robot 0's
DOFs first, then robot 1's. Every per-robot port (tool pose, Jacobian,
damping) has one entry per robot, since the task-space storage is always 6D
regardless of a robot's own DOF count — a robot whose ``axis_weight`` zeroes
some of the 6 canonical axes (position x/y/z, orientation x/y/z) uses only
the active ones (see ``axis_weight``). The controller owns no index tables —
a caller who needs to read from or write to a simulation-sized array binds
an indexed view instead, e.g. ``sim_array[ctrl.qd_start]`` using a paired
model-based controller's own ``q_start``/``qd_start`` properties (see
:attr:`ControllerDifferentialIK.qd_start`).

Differential-kinematics law (shown here for damped least squares; see
``DifferentialIKMethod`` for the other four solvers — pseudo-inverse, transpose,
adaptive damping, and truncated SVD):

    e = pose_error(tool_pose_world, desired_tool_pose_world)
    q̇_target = bandwidth · Jᵀ(JJᵀ + λ²I)⁻¹e
    q_target = joint_q + q̇_target · dt

where ``J`` is the tool-point Jacobian, ``λ`` the damping, and ``e`` the 6D
pose error (position, then axis-angle orientation), shown unweighted; see
``axis_weight`` for the ``diag(w)``-weighted form actually solved, including
how a zero-weighted axis is excluded from the task rather than driven to
zero.

A redundant robot (more controlled DOFs than its own task dimension needs —
6, unless ``axis_weight`` zeroes some axes) may also project a secondary
joint-space objective — joint-limit avoidance and/or a
posture target — through a damped kinematic (Moore-Penrose) null-space
projector ``N = I - Jᵀ(JJᵀ + λ_null²I)⁻¹J``, so it (approximately) never
disturbs the primary task:

    q̇_target += N @ dq_center

where ``dq_center`` is the sum of whichever secondary-objective biases are
enabled. ``λ_null`` (``null_space_damping``) is independent of the primary
task's DLS damping, and — like that damping — makes the projector
well-defined for any Jacobian, including a structurally rank-deficient one
(e.g. a redundant low-DOF arm, such as a 4R planar arm, whose task is itself
lower-dimensional than 6D). The tradeoff is the same kind the primary solve
already accepts: ``J @ N`` is no longer exactly zero, but a residual of
order ``λ_null²``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ...controller import ControllerBase
from ...utils import _bake_optional_float_array, _validate_array
from .._common import (
    _add_term_kernel,
    _block_matrix_vector_multiply_kernel,
    _null_space_projector_kernel,
    _pose_error_kernel,
    _read_port,
    _scatter_port_kernel,
)
from ._common import (
    _JACOBI_SVD_MAX_SWEEPS,
    _JACOBI_SVD_TOL,
    DifferentialIKMethod,
    _adaptive_damping_kernel,
    _gather_and_weight_jacobian_kernel,
    _gather_jacobian_by_axis_kernel,
    _gather_task_error_kernel,
    _integrate_position_kernel,
    _joint_limit_avoidance_bias_kernel,
    _pinv_singular_value_damped_kernel,
    _posture_bias_kernel,
    _qd_from_singular_basis_kernel,
    _qd_from_y_kernel,
    _qd_in_singular_basis_damped_kernel,
    _qd_in_singular_basis_truncated_kernel,
    _scatter_pinv_transpose_by_axis_kernel,
    _svd_one_sided_jacobi_kernel,
    _svd_reconstruct_scaled_kernel,
)

_CANONICAL_TASK_AXIS_COUNT = 6


def _validate_non_negative_gain(value: wp.array[wp.float32] | float | None, name: str) -> None:
    """Raise if a gain (scalar, wp.array, or already-read wp.array.numpy()) is negative anywhere.

    ``None`` (an unset live gain, or a disabled one) is not an error here --
    a missing value is a different problem, checked elsewhere. Shape/dtype
    of an array must already be validated by the caller; this only checks
    sign.
    """
    if value is None:
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
        return
    value_np = value.numpy() if isinstance(value, wp.array) else value
    if np.any(value_np < 0.0):
        raise ValueError(f"{name} must be non-negative, got {value_np.tolist()}.")


def _validate_joint_limits(
    joint_pos_lower: wp.array[wp.float32],
    joint_pos_upper: wp.array[wp.float32],
    total_controlled_dofs: int,
    device: wp.context.Device,
) -> None:
    """Raise unless ``joint_pos_lower``/``joint_pos_upper`` are finite, correctly shaped, and ``lower < upper`` everywhere.

    Shared between construction and :meth:`ControllerDifferentialIKModelFree.set_joint_limits`, since both need the
    same pair of arrays to hold the same invariants.
    """
    _validate_array(
        array=joint_pos_lower, name="joint_pos_lower", dtype=wp.float32, shape=(total_controlled_dofs,), device=device
    )
    _validate_array(
        array=joint_pos_upper, name="joint_pos_upper", dtype=wp.float32, shape=(total_controlled_dofs,), device=device
    )
    if np.any(joint_pos_lower.numpy() >= joint_pos_upper.numpy()):
        raise ValueError("joint_pos_lower must be strictly less than joint_pos_upper for every DOF.")
    if not (np.all(np.isfinite(joint_pos_lower.numpy())) and np.all(np.isfinite(joint_pos_upper.numpy()))):
        raise ValueError(
            "joint_pos_lower/joint_pos_upper must be finite for every DOF when use_joint_limit_avoidance=True; "
            "an unbounded limit produces a NaN midpoint even though the avoidance bias is zero away from a limit."
        )


class ControllerDifferentialIKModelFree(ControllerBase):
    """Differential-kinematics (Jacobian-based) controller with a caller-supplied Jacobian.

    Implements a differential-kinematics control law, selectable per instance
    via :class:`DifferentialIKMethod` (damped least squares by default). This model-free
    variant expects the tool-point Jacobian and the current tool pose to be
    computed externally — it is the caller's responsibility to provide them,
    and to keep them consistent with ``inputs.joint_q``, before every
    :meth:`step`.

    Every per-DOF port is **compact**: a 1-D array with one entry per
    controlled DOF, ordered robot 0's DOFs first, then robot 1's, matching
    ``controlled_dofs_per_robot``. Every per-robot port has one entry per
    robot, since the task-space storage is always 6D regardless of a
    robot's own :attr:`axis_weight`. A port may be
    bound either to a plain array or to an indexed view of a
    simulation-sized array, which is how a caller expresses a gather or
    scatter without the controller owning an index table — for example,
    using a paired model-based controller's own ``q_start``/``qd_start``
    properties (see :attr:`ControllerDifferentialIK.q_start`)::

        inputs.joint_q = state.joint_q[ctrl.q_start]  # gather
        outputs.joint_q_target = control.joint_target_q[ctrl.q_start]  # scatter

    Views are live and graph-capturable: bind them once, and each step (or
    graph replay) reads through to the current contents of the underlying
    array.

    Array shapes and devices are validated on each direct call to
    :meth:`step`, but not when a captured graph is replayed, since the
    checks run in Python at capture time only.

    Supports heterogeneous robot fleets — robots may have different
    controlled-DOF counts. The Jacobian is padded to ``max_controlled_dofs``;
    every other buffer is compact.

    Allocate input and output structs via :meth:`input` and :meth:`output`.
    All field names on those structs are fixed — see :class:`Inputs` and
    :class:`Outputs` for the typed schema.

    Args:
        controlled_dofs_per_robot: Controlled-DOF count for each robot. Its
            length sets :attr:`controlled_robot_count`, its sum sets
            :attr:`total_controlled_dofs` (the length of every compact
            port), and its maximum sets :attr:`max_controlled_dofs` (the
            padded width of the Jacobian). Every entry must be positive.
        axis_weight: Non-negative per-axis weight for each of the 6
            canonical task axes (position x, y, z, then orientation x, y,
            z), ``diag(w)`` applied to both the Jacobian and the pose error
            for that axis before the solve (``J_w = diag(w) @ J``,
            ``e_w = diag(w) @ e``) — a genuine soft weight for any nonzero
            value. An axis weighted exactly ``0`` is different in kind, not
            just degree: it is excluded from the solve structurally (its
            error and Jacobian rows never enter it at all), not merely
            driven toward zero by a very small weight — this also shrinks
            the task's own dimension, so a robot with fewer than 6
            controlled DOFs can still be redundant if enough axes are
            zeroed. Any combination of active axes is allowed, not just a
            leading prefix. Pass a single ``wp.spatial_vector`` to apply the
            same weights to every robot, or an array of shape
            [controlled_robot_count] to set them per robot. ``None`` (the
            default) means every axis is weighted ``1`` for every robot —
            full, equally-trusted 6D pose.
        bandwidth: Output velocity scale gain, applied per controlled DOF
            after the Jacobian solve. Must be non-negative, since a negative
            value would flip the output velocity's direction. Checked at
            construction when baked; a live value is
            the caller's own responsibility, since checking it every step
            would cost a host sync and break :meth:`is_graphable`. Pass a
            scalar to apply the same gain to every controlled DOF, an array
            of shape [total_controlled_dofs] to set them individually, or
            ``None`` to read ``inputs.bandwidth`` each step.
        damping: Damped-least-squares regularization λ, applied per robot to
            the task-space normal-equations matrix. Pass a scalar to apply
            the same damping to every robot, an array of shape
            [controlled_robot_count] to set them individually, or ``None``
            to read ``inputs.damping`` each step. Only meaningful when
            ``ik_method=DifferentialIKMethod.DAMPED_LEAST_SQUARES`` (the default); must
            be ``None`` for every other :class:`DifferentialIKMethod`, which has no λ to
            set.
        ik_method: Inverse-Jacobian solve method, a :class:`DifferentialIKMethod`.
            Defaults to ``DifferentialIKMethod.DAMPED_LEAST_SQUARES``.
        adaptive_damping_min: λ used when the smallest singular value of the
            task Jacobian is at or above ``adaptive_damping_threshold``.
            Required (and must be non-negative) when
            ``ik_method=DifferentialIKMethod.ADAPTIVE_DAMPING``; must be ``None``
            otherwise.
        adaptive_damping_max: λ used at a full singularity (smallest
            singular value zero), ramping down to ``adaptive_damping_min``
            as the smallest singular value rises to
            ``adaptive_damping_threshold``. Required (and must exceed
            ``adaptive_damping_min``) when
            ``ik_method=DifferentialIKMethod.ADAPTIVE_DAMPING``; must be ``None``
            otherwise.
        adaptive_damping_threshold: Smallest-singular-value threshold below
            which damping starts ramping from ``adaptive_damping_min``
            toward ``adaptive_damping_max``. Required (and must be
            positive) when ``ik_method=DifferentialIKMethod.ADAPTIVE_DAMPING``; must be
            ``None`` otherwise.
        truncated_svd_threshold: Per-direction singular-value threshold —
            a task-space direction with singular value above this is
            inverted exactly, one at or below it is dropped from the solve
            entirely. Required (and must be positive) when
            ``ik_method=DifferentialIKMethod.TRUNCATED_SVD``; must be ``None``
            otherwise.
        use_joint_limit_avoidance: Project a joint-limit-avoidance bias
            through the null-space projector. Requires
            ``joint_limit_avoidance_gain``, ``joint_limit_avoidance_margin``,
            ``joint_pos_lower``, and ``joint_pos_upper``.
        joint_limit_avoidance_gain: Joint-centering gain, applied once a DOF
            comes within ``joint_limit_avoidance_margin`` of either limit.
            Required (and must be positive) when
            ``use_joint_limit_avoidance=True``.
        joint_limit_avoidance_margin: Distance from either limit at which
            the avoidance bias starts ramping in [m or rad], same units as
            ``joint_pos_lower``/``joint_pos_upper``. Required (and must be
            positive) when ``use_joint_limit_avoidance=True``.
        joint_pos_lower: Lower joint position limit per controlled DOF
            [m or rad], shape [total_controlled_dofs]. Required when
            ``use_joint_limit_avoidance=True``; baked at construction, not a
            live port.
        joint_pos_upper: Upper joint position limit per controlled DOF
            [m or rad], shape [total_controlled_dofs]. Required when
            ``use_joint_limit_avoidance=True``; baked at construction, not a
            live port.
        use_null_space_posture_control: Project a proportional pull toward
            ``inputs.q_des_null`` through the null-space projector. Enables
            ``null_space_stiffness``.
        null_space_stiffness: Posture-control proportional gain, applied per
            controlled DOF. Must be non-negative; checked the same way as
            ``bandwidth``. Pass a scalar to apply the same gain to every
            controlled DOF, an array of shape [total_controlled_dofs] to set
            them individually, or ``None`` to read
            ``inputs.null_space_stiffness`` each step. Must be ``None`` when
            ``use_null_space_posture_control=False``.
        null_space_damping: Damping λ_null for the null-space projector's own
            ``(JJᵀ + λ_null²I)⁻¹``, independent of the primary task's
            ``damping``. Must be non-negative; checked the same way as
            ``bandwidth``. Only meaningful when ``use_joint_limit_avoidance``
            or ``use_null_space_posture_control`` is enabled — pass a scalar
            or an array of shape [controlled_robot_count] to bake a value,
            or leave it ``None`` to read ``inputs.null_space_damping`` each
            step (the default, and the only valid value when both are
            disabled). Unlike the primary ``damping``, ``λ_null = 0`` is
            only safe when every robot has at least as many controlled DOFs
            as its own task dimension (the number of nonzero
            ``null_space_axes`` entries) — otherwise the projector's own
            ``JJᵀ`` is rank-deficient. That stronger, per-robot requirement
            is checked at construction only when baked; a live value is the
            caller's responsibility there, since it needs the null-space
            task dimension and ``controlled_dofs_per_robot`` compared per
            robot, not just a sign check.
        null_space_axes: Which of the 6 canonical axes the null-space
            projector guarantees the secondary objective (joint-limit
            avoidance/posture control) won't disturb — zero leaves that
            axis unprotected, nonzero protects it; only the sign matters,
            unlike ``axis_weight``'s own soft magnitude. Defaults to
            ``axis_weight`` (every solved axis protected), but the two are
            independent: an axis can be softly solved for yet left
            unprotected, e.g. an under-actuated arm with too few DOFs to
            protect every solved axis and still have a usable null space
            left over. Unlike ``axis_weight``, an all-zero row is legal —
            it protects no axes, so the secondary objective is free to
            move all of them. Only meaningful when
            ``use_joint_limit_avoidance`` or
            ``use_null_space_posture_control`` is enabled.
        device: Warp device.
        requires_grad: Not supported at this time; must be ``False``.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerDifferentialIKModelFree.input`.

        Every compact 1-D field has shape [total_controlled_dofs]; every
        per-robot field has shape [controlled_robot_count]. Optional fields
        are ``None`` when the corresponding gain is baked at construction.
        """

        joint_q: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Current joint positions [m or rad], shape [total_controlled_dofs]."""
        tool_pose_world: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Current tool pose [m, unitless quaternion], world frame, shape [controlled_robot_count]."""
        desired_tool_pose_world: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Desired tool pose [m, unitless quaternion], world frame, shape [controlled_robot_count]."""
        jacobian_tool_world: wp.array3d[wp.float32] | wp.indexedarray(dtype=wp.float32, ndim=3)
        """Tool-point Jacobian, world frame, shape [controlled_robot_count, 6, max_controlled_dofs]; columns beyond a robot's own controlled-DOF count are unused. Rows 0-2 map a controlled DOF's velocity to the tool point's linear velocity [1 or m], rows 3-5 to its angular velocity [1/m or 1]."""
        bandwidth: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Output velocity scale gain, shape [total_controlled_dofs]. ``None`` when baked at construction."""
        damping: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Damped-least-squares regularization λ, shape [controlled_robot_count]. ``None`` when baked at construction."""
        q_des_null: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Null-space posture target [m or rad], shape [total_controlled_dofs]. ``None`` unless ``use_null_space_posture_control=True``."""
        null_space_stiffness: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Posture-control proportional gain, shape [total_controlled_dofs]. ``None`` when disabled, or when baked at construction."""
        null_space_damping: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Null-space projector damping λ_null, shape [controlled_robot_count]. ``None`` when both secondary objectives are disabled, or when baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerDifferentialIKModelFree.output`."""

        joint_qd_target: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Target joint velocity [m/s or rad/s], shape [total_controlled_dofs]."""
        joint_q_target: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """One-step-ahead target joint position [m or rad], shape [total_controlled_dofs] = ``joint_q + joint_qd_target * dt``."""

    def __init__(
        self,
        *,
        controlled_dofs_per_robot: wp.array[wp.int32],
        axis_weight: wp.array[wp.spatial_vector] | wp.spatial_vector | None = None,
        bandwidth: wp.array[wp.float32] | float | None,
        damping: wp.array[wp.float32] | float | None,
        ik_method: DifferentialIKMethod = DifferentialIKMethod.DAMPED_LEAST_SQUARES,
        adaptive_damping_min: float | None = None,
        adaptive_damping_max: float | None = None,
        adaptive_damping_threshold: float | None = None,
        truncated_svd_threshold: float | None = None,
        use_joint_limit_avoidance: bool = False,
        joint_limit_avoidance_gain: float = 0.0,
        joint_limit_avoidance_margin: float = 0.0,
        joint_pos_lower: wp.array[wp.float32] | None = None,
        joint_pos_upper: wp.array[wp.float32] | None = None,
        use_null_space_posture_control: bool = False,
        null_space_stiffness: wp.array[wp.float32] | float | None = None,
        null_space_damping: wp.array[wp.float32] | float | None = None,
        null_space_axes: wp.array[wp.spatial_vector] | wp.spatial_vector | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        self._device = wp.get_device(device)

        # ------------------------------------------------------------------
        # Validation: every wp.array argument is checked here, and nowhere
        # else. controlled_dofs_per_robot comes first because the shapes
        # below derive from it.
        # ------------------------------------------------------------------
        if not isinstance(controlled_dofs_per_robot, wp.array):
            raise TypeError(
                f"controlled_dofs_per_robot must be a wp.array, got {type(controlled_dofs_per_robot).__name__}."
            )
        _validate_array(
            array=controlled_dofs_per_robot,
            name="controlled_dofs_per_robot",
            dtype=wp.int32,
            shape=(controlled_dofs_per_robot.size,),
            device=self._device,
        )

        controlled_dofs_per_robot_np = controlled_dofs_per_robot.numpy()
        controlled_robot_count = int(controlled_dofs_per_robot_np.size)
        if controlled_robot_count < 1:
            raise ValueError("controlled_dofs_per_robot must not be empty.")
        if controlled_dofs_per_robot_np.min() < 1:
            raise ValueError(
                f"controlled_dofs_per_robot must be positive — a robot with no controlled DOF occupies no "
                f"slot in any buffer, so leave it out; got {controlled_dofs_per_robot_np.tolist()}."
            )

        max_controlled_dofs = int(controlled_dofs_per_robot_np.max())
        total_controlled_dofs = int(controlled_dofs_per_robot_np.sum())

        if axis_weight is None:
            axis_weight_resolved = wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        else:
            axis_weight_resolved = axis_weight
        if isinstance(axis_weight_resolved, wp.spatial_vector):
            axis_weight_np = np.tile(np.array(axis_weight_resolved, dtype=np.float32), (controlled_robot_count, 1))
        elif isinstance(axis_weight_resolved, wp.array):
            _validate_array(
                array=axis_weight_resolved,
                name="axis_weight",
                dtype=wp.spatial_vector,
                shape=(controlled_robot_count,),
                device=self._device,
            )
            axis_weight_np = axis_weight_resolved.numpy()
        else:
            raise TypeError(
                "axis_weight must be a wp.spatial_vector or a wp.array[wp.spatial_vector] of shape "
                f"(controlled_robot_count,), got {type(axis_weight_resolved).__name__}."
            )
        if np.any(axis_weight_np < 0.0):
            raise ValueError(f"axis_weight must be non-negative, got {axis_weight_np.tolist()}.")

        # Only zero vs. nonzero matters for which axes enter the solve at
        # all: an axis is either in it (its error and Jacobian rows gathered
        # and weighted, see _gather_and_weight_jacobian_kernel) or
        # structurally excluded from it. A nonzero weight's magnitude still
        # matters once its axis has been gathered -- it is a genuine soft
        # weight there.
        axis_active_np = axis_weight_np > 0.0
        task_dim_np = axis_active_np.sum(axis=1).astype(np.int32)
        if task_dim_np.min() < 1:
            bad_robots = np.flatnonzero(task_dim_np < 1)
            raise ValueError(
                f"axis_weight is all-zero for robot(s) {bad_robots.tolist()}; every robot needs at least one "
                "nonzero-weighted task axis."
            )

        # Compact-slot -> canonical-axis lookup, one row per robot: slot k of
        # the k'th active entry in axis_weight_np[robot], in canonical
        # order; entries at or beyond that robot's own task_dim are never
        # read.
        active_axis_of_slot_np = np.zeros((controlled_robot_count, _CANONICAL_TASK_AXIS_COUNT), dtype=np.int32)
        for robot in range(controlled_robot_count):
            active = np.flatnonzero(axis_active_np[robot])
            active_axis_of_slot_np[robot, : active.size] = active

        if not (isinstance(bandwidth, (int, float)) and not isinstance(bandwidth, bool)):
            _validate_array(
                array=bandwidth,
                name="bandwidth",
                dtype=wp.float32,
                shape=(total_controlled_dofs,),
                device=self._device,
                required=False,
            )
        # Read directly (not squared) by _qd_from_y_kernel, so a negative
        # baked value would flip the direction of every controlled DOF's
        # output velocity instead of merely scaling it.
        _validate_non_negative_gain(bandwidth, "bandwidth")

        if not isinstance(ik_method, DifferentialIKMethod):
            raise TypeError(f"ik_method must be a DifferentialIKMethod, got {ik_method!r}.")

        if requires_grad:
            raise ValueError("requires_grad=True is not supported at this time.")

        if ik_method == DifferentialIKMethod.DAMPED_LEAST_SQUARES:
            if not (isinstance(damping, (int, float)) and not isinstance(damping, bool)):
                _validate_array(
                    array=damping,
                    name="damping",
                    dtype=wp.float32,
                    shape=(controlled_robot_count,),
                    device=self._device,
                    required=False,
                )
        elif damping is not None:
            raise ValueError(f"damping was given but ik_method={ik_method} does not use it (pass damping=None).")

        if ik_method == DifferentialIKMethod.PSEUDO_INVERSE:
            bad_robots = np.flatnonzero(controlled_dofs_per_robot_np < task_dim_np)
            if bad_robots.size > 0:
                raise ValueError(
                    "ik_method=DifferentialIKMethod.PSEUDO_INVERSE requires every robot to have at least as many controlled "
                    f"DOFs as its own task dimension, since JJᵀ is otherwise rank-deficient at λ = 0; robot(s) "
                    f"{bad_robots.tolist()} have controlled_dofs_per_robot < their own axis_weight's active count."
                )

        if ik_method == DifferentialIKMethod.ADAPTIVE_DAMPING:
            if adaptive_damping_min is None or adaptive_damping_max is None or adaptive_damping_threshold is None:
                raise ValueError(
                    "ik_method=DifferentialIKMethod.ADAPTIVE_DAMPING requires adaptive_damping_min, adaptive_damping_max, "
                    "and adaptive_damping_threshold."
                )
            if adaptive_damping_min < 0.0:
                raise ValueError(f"adaptive_damping_min must be non-negative, got {adaptive_damping_min}.")
            if adaptive_damping_max <= adaptive_damping_min:
                raise ValueError(
                    "adaptive_damping_max must be greater than adaptive_damping_min, got "
                    f"adaptive_damping_max={adaptive_damping_max}, adaptive_damping_min={adaptive_damping_min}."
                )
            if adaptive_damping_threshold <= 0.0:
                raise ValueError(f"adaptive_damping_threshold must be positive, got {adaptive_damping_threshold}.")
        elif (
            adaptive_damping_min is not None
            or adaptive_damping_max is not None
            or adaptive_damping_threshold is not None
        ):
            raise ValueError(
                "adaptive_damping_min/adaptive_damping_max/adaptive_damping_threshold were given but "
                f"ik_method={ik_method} != DifferentialIKMethod.ADAPTIVE_DAMPING."
            )

        if ik_method == DifferentialIKMethod.TRUNCATED_SVD:
            if truncated_svd_threshold is None:
                raise ValueError("ik_method=DifferentialIKMethod.TRUNCATED_SVD requires truncated_svd_threshold.")
            if truncated_svd_threshold <= 0.0:
                raise ValueError(f"truncated_svd_threshold must be positive, got {truncated_svd_threshold}.")
        elif truncated_svd_threshold is not None:
            raise ValueError(
                f"truncated_svd_threshold was given but ik_method={ik_method} != DifferentialIKMethod.TRUNCATED_SVD."
            )

        use_null_space = bool(use_joint_limit_avoidance) or bool(use_null_space_posture_control)

        if use_joint_limit_avoidance:
            if joint_limit_avoidance_gain <= 0.0:
                raise ValueError(
                    "joint_limit_avoidance_gain must be positive when use_joint_limit_avoidance=True, got "
                    f"{joint_limit_avoidance_gain}."
                )
            if joint_limit_avoidance_margin <= 0.0:
                raise ValueError(
                    "joint_limit_avoidance_margin must be positive when use_joint_limit_avoidance=True, got "
                    f"{joint_limit_avoidance_margin}."
                )
            _validate_joint_limits(
                joint_pos_lower=joint_pos_lower,
                joint_pos_upper=joint_pos_upper,
                total_controlled_dofs=total_controlled_dofs,
                device=self._device,
            )
        elif joint_pos_lower is not None or joint_pos_upper is not None:
            raise ValueError("joint_pos_lower/joint_pos_upper were given but use_joint_limit_avoidance=False.")

        if use_null_space_posture_control:
            if not (isinstance(null_space_stiffness, (int, float)) and not isinstance(null_space_stiffness, bool)):
                _validate_array(
                    array=null_space_stiffness,
                    name="null_space_stiffness",
                    dtype=wp.float32,
                    shape=(total_controlled_dofs,),
                    device=self._device,
                    required=False,
                )
            _validate_non_negative_gain(null_space_stiffness, "null_space_stiffness")
        elif null_space_stiffness is not None:
            raise ValueError("null_space_stiffness was given but use_null_space_posture_control=False.")

        if use_null_space:
            if not (isinstance(null_space_damping, (int, float)) and not isinstance(null_space_damping, bool)):
                _validate_array(
                    array=null_space_damping,
                    name="null_space_damping",
                    dtype=wp.float32,
                    shape=(controlled_robot_count,),
                    device=self._device,
                    required=False,
                )
            _validate_non_negative_gain(null_space_damping, "null_space_damping")
        elif null_space_damping is not None:
            raise ValueError(
                "null_space_damping was given but neither use_joint_limit_avoidance nor "
                "use_null_space_posture_control is enabled."
            )

        if use_null_space:
            null_space_axes_resolved = axis_weight_resolved if null_space_axes is None else null_space_axes
            if isinstance(null_space_axes_resolved, wp.spatial_vector):
                null_space_axes_np = np.tile(
                    np.array(null_space_axes_resolved, dtype=np.float32), (controlled_robot_count, 1)
                )
            elif isinstance(null_space_axes_resolved, wp.array):
                _validate_array(
                    array=null_space_axes_resolved,
                    name="null_space_axes",
                    dtype=wp.spatial_vector,
                    shape=(controlled_robot_count,),
                    device=self._device,
                )
                null_space_axes_np = null_space_axes_resolved.numpy()
            else:
                raise TypeError(
                    "null_space_axes must be a wp.spatial_vector or a wp.array[wp.spatial_vector] of shape "
                    f"(controlled_robot_count,), got {type(null_space_axes_resolved).__name__}."
                )
            if np.any(null_space_axes_np < 0.0):
                raise ValueError(f"null_space_axes must be non-negative, got {null_space_axes_np.tolist()}.")

            # Unlike axis_weight, an all-zero row is legal here: it means
            # that robot's null-space projector is unconstrained (N = I),
            # not an error -- there's no "nothing to solve for" problem the
            # way there is for the primary task.
            null_space_axis_active_np = null_space_axes_np > 0.0
            null_space_task_dim_np = null_space_axis_active_np.sum(axis=1).astype(np.int32)
            null_space_active_axis_of_slot_np = np.zeros(
                (controlled_robot_count, _CANONICAL_TASK_AXIS_COUNT), dtype=np.int32
            )
            for robot in range(controlled_robot_count):
                active = np.flatnonzero(null_space_axis_active_np[robot])
                null_space_active_axis_of_slot_np[robot, : active.size] = active
        elif null_space_axes is not None:
            raise ValueError(
                "null_space_axes was given but neither use_joint_limit_avoidance nor "
                "use_null_space_posture_control is enabled."
            )
        # ------------------------------------------------------------------

        self._ik_method = ik_method
        self._use_joint_limit_avoidance = bool(use_joint_limit_avoidance)
        self._use_null_space_posture_control = bool(use_null_space_posture_control)
        self._use_null_space = use_null_space
        self._controlled_robot_count = controlled_robot_count
        self._max_controlled_dofs = max_controlled_dofs
        self._total_controlled_dofs = total_controlled_dofs
        self._requires_grad = requires_grad

        # Per-robot task dimension and compact-slot -> canonical-axis table,
        # derived from axis_weight above. Copied, not stored, for the same
        # reason controlled_dofs_per_robot is.
        self._task_dim = wp.array(task_dim_np, dtype=wp.int32, device=self._device)
        self._active_axis_of_slot = wp.array2d(active_axis_of_slot_np, dtype=wp.int32, device=self._device)
        self._axis_weight = wp.array(
            [wp.spatial_vector(*row) for row in axis_weight_np], dtype=wp.spatial_vector, device=self._device
        )

        # Same shape as task_dim/active_axis_of_slot above, but for the
        # null-space projector's own notion of "the task" (null_space_axes),
        # independent of the primary solve's axis_weight -- see step()'s
        # null-space section for where each is used.
        self._null_space_task_dim = (
            wp.array(null_space_task_dim_np, dtype=wp.int32, device=self._device) if use_null_space else None
        )
        self._null_space_active_axis_of_slot = (
            wp.array2d(null_space_active_axis_of_slot_np, dtype=wp.int32, device=self._device)
            if use_null_space
            else None
        )

        # Copied, not stored: the kernels use this as a loop bound while the
        # tables below are derived from the same host snapshot, so a later
        # edit to the caller's array would send a launch past the end of a
        # buffer.
        self._controlled_dofs_per_robot = wp.array(controlled_dofs_per_robot_np, dtype=wp.int32, device=self._device)

        # Flat-DOF -> (robot, slot) table, so the Jacobian-transpose finish
        # can run as a flat launch over total_controlled_dofs instead of a
        # padded 2-D one.
        self._robot_of_dof = wp.array(
            np.repeat(np.arange(controlled_robot_count, dtype=np.int32), controlled_dofs_per_robot_np),
            dtype=wp.int32,
            device=self._device,
        )
        self._slot_of_dof = wp.array(
            np.concatenate(
                [np.arange(n, dtype=np.int32) for n in controlled_dofs_per_robot_np] or [np.empty(0, np.int32)]
            ),
            dtype=wp.int32,
            device=self._device,
        )

        self._bandwidth_baked = _bake_optional_float_array(
            bandwidth, total_controlled_dofs, device=self._device, requires_grad=self._requires_grad
        )
        # PSEUDO_INVERSE/TRANSPOSE never read damping (validated above to be
        # None), so a dummy zero bake keeps them out of the live-port path
        # without a separate "does this method use damping" branch below.
        self._damping_baked = (
            _bake_optional_float_array(
                damping, controlled_robot_count, device=self._device, requires_grad=self._requires_grad
            )
            if ik_method == DifferentialIKMethod.DAMPED_LEAST_SQUARES
            else _bake_optional_float_array(
                0.0, controlled_robot_count, device=self._device, requires_grad=self._requires_grad
            )
        )
        self._use_adaptive_damping = ik_method == DifferentialIKMethod.ADAPTIVE_DAMPING
        self._adaptive_damping_min_baked = (
            _bake_optional_float_array(
                adaptive_damping_min, controlled_robot_count, device=self._device, requires_grad=self._requires_grad
            )
            if self._use_adaptive_damping
            else None
        )
        self._adaptive_damping_max_baked = (
            _bake_optional_float_array(
                adaptive_damping_max, controlled_robot_count, device=self._device, requires_grad=self._requires_grad
            )
            if self._use_adaptive_damping
            else None
        )
        self._adaptive_damping_threshold_baked = (
            _bake_optional_float_array(
                adaptive_damping_threshold,
                controlled_robot_count,
                device=self._device,
                requires_grad=self._requires_grad,
            )
            if self._use_adaptive_damping
            else None
        )
        self._use_truncated_svd = ik_method == DifferentialIKMethod.TRUNCATED_SVD
        self._truncated_svd_threshold_baked = (
            _bake_optional_float_array(
                truncated_svd_threshold,
                controlled_robot_count,
                device=self._device,
                requires_grad=self._requires_grad,
            )
            if self._use_truncated_svd
            else None
        )
        self._null_space_stiffness_baked = (
            _bake_optional_float_array(
                null_space_stiffness, total_controlled_dofs, device=self._device, requires_grad=self._requires_grad
            )
            if self._use_null_space_posture_control
            else None
        )
        self._null_space_damping_baked = (
            _bake_optional_float_array(
                null_space_damping, controlled_robot_count, device=self._device, requires_grad=self._requires_grad
            )
            if use_null_space
            else None
        )
        if self._null_space_damping_baked is not None:
            # Only checkable when baked: a live value isn't known until step(),
            # and reading it back to check would cost a host sync every step,
            # breaking is_graphable(). JJᵀ + λ_null²I is only guaranteed SPD
            # for a robot with fewer controlled DOFs than the null-space
            # projector's own task dimension (null_space_task_dim_np, not
            # the primary task_dim_np -- it's the projector's own JJᵀ being
            # inverted here) when λ_null > 0.
            null_space_damping_np = self._null_space_damping_baked.numpy()
            bad_robots = np.flatnonzero(
                (controlled_dofs_per_robot_np < null_space_task_dim_np) & (null_space_damping_np <= 0.0)
            )
            if bad_robots.size > 0:
                raise ValueError(
                    "null_space_damping must be positive for a robot with fewer controlled DOFs than its own "
                    "null-space task dimension, since the null-space projector's JJᵀ is then rank-deficient "
                    f"without damping; robot(s) {bad_robots.tolist()} have controlled_dofs_per_robot below their "
                    "own null_space_axes' active count with null_space_damping <= 0."
                )

        self._joint_limit_avoidance_gain = float(joint_limit_avoidance_gain)
        self._joint_limit_avoidance_margin = float(joint_limit_avoidance_margin)
        self._joint_pos_lower: wp.array[wp.float32] | None = (
            wp.array(joint_pos_lower.numpy(), dtype=wp.float32, device=self._device)
            if self._use_joint_limit_avoidance
            else None
        )
        self._joint_pos_upper: wp.array[wp.float32] | None = (
            wp.array(joint_pos_upper.numpy(), dtype=wp.float32, device=self._device)
            if self._use_joint_limit_avoidance
            else None
        )

        self._q_buf = wp.zeros(
            total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
        )
        self._tool_pose_buf = wp.zeros(
            controlled_robot_count, dtype=wp.transform, device=self._device, requires_grad=requires_grad
        )
        self._desired_pose_buf = wp.zeros(
            controlled_robot_count, dtype=wp.transform, device=self._device, requires_grad=requires_grad
        )
        self._jacobian_buf = wp.zeros(
            (controlled_robot_count, 6, max_controlled_dofs),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )
        self._bandwidth_buf: wp.array[wp.float32] | None = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad)
            if self._bandwidth_baked is None
            else None
        )
        self._damping_buf: wp.array[wp.float32] | None = (
            wp.zeros(controlled_robot_count, dtype=wp.float32, device=self._device, requires_grad=requires_grad)
            if self._damping_baked is None
            else None
        )

        self._pose_error_buf = wp.zeros(
            controlled_robot_count, dtype=wp.spatial_vector, device=self._device, requires_grad=requires_grad
        )
        self._pose_error_active_buf = wp.zeros(
            controlled_robot_count, dtype=wp.spatial_vector, device=self._device, requires_grad=requires_grad
        )

        # Matrix/vector types for .view()-ing the plain wp.array3d[float]/
        # array2d[float] storage below into the shapes _svd_one_sided_jacobi_kernel
        # needs -- a zero-copy reinterpret, not a separate allocation/layout
        # (see the section comment above _svd_one_sided_jacobi in _common.py).
        # Reused for both the primary task solve's SVD and (when enabled)
        # the null-space projector's own SVD, since both are the same 6 x
        # max_controlled_dofs shape.
        svd_mat_u = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
        svd_mat_j = wp.types.matrix(shape=(6, max_controlled_dofs), dtype=wp.float32)
        svd_mat_v = wp.types.matrix(shape=(max_controlled_dofs, max_controlled_dofs), dtype=wp.float32)
        svd_vec_s = wp.types.vector(length=max_controlled_dofs, dtype=wp.float32)

        # Primary task solve: J_w = diag(w) @ J is gathered+weighted into
        # compact slot order, then decomposed once per step; every
        # matrix-inverting DifferentialIKMethod (all but TRANSPOSE) shares this
        # single SVD, differing only in the per-singular-direction gain
        # applied to it below.
        self._jacobian_weighted_buf = wp.zeros(
            (controlled_robot_count, 6, max_controlled_dofs),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )
        self._jacobian_weighted_view = self._jacobian_weighted_buf.view(svd_mat_j).reshape((controlled_robot_count,))
        self._svd_u_buf = wp.zeros(
            (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
        )
        self._svd_u_view = self._svd_u_buf.view(svd_mat_u).reshape((controlled_robot_count,))
        self._svd_s_buf = wp.zeros(
            (controlled_robot_count, max_controlled_dofs),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )
        self._svd_s_view = self._svd_s_buf.view(svd_vec_s).reshape((controlled_robot_count,))
        self._svd_v_buf = wp.zeros(
            (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )
        self._svd_v_view = self._svd_v_buf.view(svd_mat_v).reshape((controlled_robot_count,))
        self._qd_in_singular_basis_buf = wp.zeros(
            (controlled_robot_count, max_controlled_dofs),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )
        if self._use_adaptive_damping:
            self._adaptive_damping_buf = wp.zeros(controlled_robot_count, dtype=wp.float32, device=self._device)
        self._qd_buf = wp.zeros(
            total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
        )
        self._q_target_buf = wp.zeros(
            total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
        )
        self._dt_buf = wp.zeros(1, dtype=wp.float32, device=self._device, requires_grad=requires_grad)

        if use_null_space:
            offsets_np = np.zeros(controlled_robot_count, dtype=np.int32)
            offsets_np[1:] = np.cumsum(controlled_dofs_per_robot_np[:-1])
            self._dof_offsets = wp.array(offsets_np, dtype=wp.int32, device=self._device)
            self._null_space_damping_buf: wp.array[wp.float32] | None = (
                wp.zeros(controlled_robot_count, dtype=wp.float32, device=self._device, requires_grad=requires_grad)
                if self._null_space_damping_baked is None
                else None
            )
            self._svd_u_null_buf = wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._svd_u_null_view = self._svd_u_null_buf.view(svd_mat_u).reshape((controlled_robot_count,))
            self._svd_s_null_buf = wp.zeros(
                (controlled_robot_count, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._svd_s_null_view = self._svd_s_null_buf.view(svd_vec_s).reshape((controlled_robot_count,))
            self._svd_v_null_buf = wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._svd_v_null_view = self._svd_v_null_buf.view(svd_mat_v).reshape((controlled_robot_count,))
            self._pinv_singular_value_null_buf = wp.zeros(
                (controlled_robot_count, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._jacobian_active_buf = wp.zeros(
                (controlled_robot_count, 6, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            # jacobian_active_buf is unweighted (every axis weight 1), so it
            # doubles as the "A" input to the null-space projector's own SVD
            # directly, with no separate weighting step (unlike the primary
            # task solve's J_w).
            self._jacobian_active_view = self._jacobian_active_buf.view(svd_mat_j).reshape((controlled_robot_count,))
            self._jacobian_pinv_transpose_slot_buf = wp.zeros(
                (controlled_robot_count, 6, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._jacobian_pinv_transpose_buf = wp.zeros(
                (controlled_robot_count, 6, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._null_space_projector_buf = wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._dq_center_buf = wp.zeros(
                total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._dq_scratch_buf = wp.zeros(
                total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._qd_null_buf = wp.zeros(
                total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
        else:
            self._dof_offsets = None
            self._null_space_damping_buf = None

        self._q_des_null_buf: wp.array[wp.float32] | None = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad)
            if self._use_null_space_posture_control
            else None
        )
        self._null_space_stiffness_buf: wp.array[wp.float32] | None = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad)
            if self._use_null_space_posture_control and self._null_space_stiffness_baked is None
            else None
        )

    @property
    def controlled_robot_count(self) -> int:
        """Number of robots, i.e. the length of ``controlled_dofs_per_robot``."""
        return self._controlled_robot_count

    @property
    def max_controlled_dofs(self) -> int:
        """Largest controlled-DOF count over the robots, the padded width of ``inputs.jacobian_tool_world``."""
        return self._max_controlled_dofs

    @property
    def total_controlled_dofs(self) -> int:
        """Total controlled-DOF count across all robots, the length of every compact port."""
        return self._total_controlled_dofs

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def is_graphable(self) -> bool:
        return True

    def input(self) -> Inputs:
        """Return a pre-allocated :class:`Inputs` with zero-initialised arrays."""
        device = self._device
        requires_grad = self._requires_grad
        total_controlled_dofs = self._total_controlled_dofs
        controlled_robot_count = self._controlled_robot_count

        inputs = ControllerDifferentialIKModelFree.Inputs()
        inputs.joint_q = wp.zeros(total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
        inputs.tool_pose_world = wp.zeros(
            controlled_robot_count, dtype=wp.transform, device=device, requires_grad=requires_grad
        )
        inputs.desired_tool_pose_world = wp.zeros(
            controlled_robot_count, dtype=wp.transform, device=device, requires_grad=requires_grad
        )
        inputs.jacobian_tool_world = wp.zeros(
            (controlled_robot_count, 6, self._max_controlled_dofs),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        inputs.bandwidth = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._bandwidth_baked is None
            else None
        )
        inputs.damping = (
            wp.zeros(controlled_robot_count, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._damping_baked is None
            else None
        )
        inputs.q_des_null = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space_posture_control
            else None
        )
        inputs.null_space_stiffness = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space_posture_control and self._null_space_stiffness_baked is None
            else None
        )
        inputs.null_space_damping = (
            wp.zeros(controlled_robot_count, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space and self._null_space_damping_baked is None
            else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with compact velocity/position arrays."""
        device = self._device
        requires_grad = self._requires_grad
        total_controlled_dofs = self._total_controlled_dofs
        outputs = ControllerDifferentialIKModelFree.Outputs()
        outputs.joint_qd_target = wp.zeros(
            total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        outputs.joint_q_target = wp.zeros(
            total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        return outputs

    def set_joint_limits(self, *, joint_pos_lower: wp.array[wp.float32], joint_pos_upper: wp.array[wp.float32]) -> None:
        """Update the joint position limits used by joint-limit avoidance, in place.

        Unlike ``bandwidth``/``damping``/etc., joint limits have no live
        ``inputs.*`` port -- they change far less often, so a setter suits
        them better than a per-step read. The caller is responsible for
        keeping ``inputs.joint_q`` consistent with the new limits.

        Args:
            joint_pos_lower: Lower joint position limit per controlled DOF
                [m or rad], shape [total_controlled_dofs].
            joint_pos_upper: Upper joint position limit per controlled DOF
                [m or rad], shape [total_controlled_dofs].
        """
        if not self._use_joint_limit_avoidance:
            raise ValueError("set_joint_limits requires use_joint_limit_avoidance=True.")
        _validate_joint_limits(
            joint_pos_lower=joint_pos_lower,
            joint_pos_upper=joint_pos_upper,
            total_controlled_dofs=self._total_controlled_dofs,
            device=self._device,
        )
        self._joint_pos_lower.assign(joint_pos_lower)
        self._joint_pos_upper.assign(joint_pos_upper)

    def step(
        self,
        *,
        inputs: Inputs,
        outputs: Outputs,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Compute one differential-kinematics step and write joint velocity/position targets.

        Args:
            inputs: Populated :class:`Inputs` struct. The Jacobian and tool
                pose fields must be filled by the caller before each call,
                consistent with ``inputs.joint_q``.
            outputs: :class:`Outputs` struct to write into.
            dt: Step duration [s], used to integrate ``joint_qd_target`` into
                ``joint_q_target``.
        """
        total_controlled_dofs = self._total_controlled_dofs
        controlled_robot_count = self._controlled_robot_count

        # (port, name, destination buffer, launch shape, dtype) for every port.
        bindings: list[tuple[Any, str, wp.array | None, tuple[int, ...], Any]] = [
            (inputs.joint_q, "inputs.joint_q", self._q_buf, (total_controlled_dofs,), wp.float32),
            (
                inputs.tool_pose_world,
                "inputs.tool_pose_world",
                self._tool_pose_buf,
                (controlled_robot_count,),
                wp.transform,
            ),
            (
                inputs.desired_tool_pose_world,
                "inputs.desired_tool_pose_world",
                self._desired_pose_buf,
                (controlled_robot_count,),
                wp.transform,
            ),
            (
                inputs.jacobian_tool_world,
                "inputs.jacobian_tool_world",
                self._jacobian_buf,
                (controlled_robot_count, 6, self._max_controlled_dofs),
                wp.float32,
            ),
        ]
        if self._bandwidth_baked is None:
            bindings.append(
                (inputs.bandwidth, "inputs.bandwidth", self._bandwidth_buf, (total_controlled_dofs,), wp.float32)
            )
        if self._damping_baked is None:
            bindings.append(
                (inputs.damping, "inputs.damping", self._damping_buf, (controlled_robot_count,), wp.float32)
            )
        if self._use_null_space_posture_control:
            bindings.append(
                (inputs.q_des_null, "inputs.q_des_null", self._q_des_null_buf, (total_controlled_dofs,), wp.float32)
            )
            if self._null_space_stiffness_baked is None:
                bindings.append(
                    (
                        inputs.null_space_stiffness,
                        "inputs.null_space_stiffness",
                        self._null_space_stiffness_buf,
                        (total_controlled_dofs,),
                        wp.float32,
                    )
                )
        if self._use_null_space and self._null_space_damping_baked is None:
            bindings.append(
                (
                    inputs.null_space_damping,
                    "inputs.null_space_damping",
                    self._null_space_damping_buf,
                    (controlled_robot_count,),
                    wp.float32,
                )
            )

        # The outputs share the ports' contract, so they are validated in the
        # same pass; a None destination marks a port as written rather than read.
        bindings.append(
            (outputs.joint_qd_target, "outputs.joint_qd_target", None, (total_controlled_dofs,), wp.float32)
        )
        bindings.append((outputs.joint_q_target, "outputs.joint_q_target", None, (total_controlled_dofs,), wp.float32))

        # A port belonging to a disabled feature or a baked gain is never
        # read, so writing one would go unnoticed. getattr because a caller
        # may leave the field unset rather than None.
        for name, live, switch in (
            ("bandwidth", self._bandwidth_baked is None, "a live bandwidth"),
            ("damping", self._damping_baked is None, "a live damping"),
            ("q_des_null", self._use_null_space_posture_control, "use_null_space_posture_control"),
            (
                "null_space_stiffness",
                self._use_null_space_posture_control and self._null_space_stiffness_baked is None,
                "a live null_space_stiffness",
            ),
            (
                "null_space_damping",
                self._use_null_space and self._null_space_damping_baked is None,
                "a live null_space_damping",
            ),
        ):
            if not live and getattr(inputs, name, None) is not None:
                raise ValueError(
                    f"inputs.{name} is set, but the controller was built without {switch}, so the value "
                    f"would be ignored."
                )

        for port, name, buf, shape, dtype in bindings:
            _validate_array(array=port, name=name, dtype=dtype, shape=shape, device=self._device, allow_indexed=True)
            if buf is not None:
                _read_port(port, buf, shape, self._device)

        bandwidth = self._bandwidth_baked if self._bandwidth_baked is not None else self._bandwidth_buf
        damping = self._damping_baked if self._damping_baked is not None else self._damping_buf

        wp.launch(
            _pose_error_kernel,
            dim=controlled_robot_count,
            inputs=[self._tool_pose_buf, self._desired_pose_buf],
            outputs=[self._pose_error_buf],
            device=self._device,
        )
        wp.launch(
            _gather_task_error_kernel,
            dim=controlled_robot_count,
            inputs=[self._pose_error_buf, self._task_dim, self._active_axis_of_slot, self._axis_weight],
            outputs=[self._pose_error_active_buf],
            device=self._device,
        )
        if self._ik_method == DifferentialIKMethod.TRANSPOSE:
            # q̇ = bandwidth · Jᵀe: no matrix to invert, so the (compact,
            # active-axes-only) pose error itself stands in for y in the
            # shared finishing kernel.
            wp.launch(
                _qd_from_y_kernel,
                dim=total_controlled_dofs,
                inputs=[
                    self._jacobian_buf,
                    self._pose_error_active_buf,
                    bandwidth,
                    self._robot_of_dof,
                    self._slot_of_dof,
                    self._task_dim,
                    self._active_axis_of_slot,
                    self._axis_weight,
                ],
                outputs=[self._qd_buf],
                device=self._device,
            )
        else:
            # Every other method shares one SVD of J_w = diag(w) @ J -- see
            # the section comment above _svd_one_sided_jacobi in _common.py
            # -- differing only in the per-singular-direction pseudo-inverse
            # singular value each applies to it below.
            wp.launch(
                _gather_and_weight_jacobian_kernel,
                dim=(controlled_robot_count, 6, self._max_controlled_dofs),
                inputs=[self._jacobian_buf, self._task_dim, self._active_axis_of_slot, self._axis_weight],
                outputs=[self._jacobian_weighted_buf],
                device=self._device,
            )
            wp.launch(
                _svd_one_sided_jacobi_kernel,
                dim=controlled_robot_count,
                inputs=[
                    self._jacobian_weighted_view,
                    self._controlled_dofs_per_robot,
                    _JACOBI_SVD_TOL,
                    _JACOBI_SVD_MAX_SWEEPS,
                ],
                outputs=[self._svd_u_view, self._svd_s_view, self._svd_v_view],
                device=self._device,
            )
            if self._use_adaptive_damping:
                wp.launch(
                    _adaptive_damping_kernel,
                    dim=controlled_robot_count,
                    inputs=[
                        self._svd_s_buf,
                        self._task_dim,
                        self._controlled_dofs_per_robot,
                        self._adaptive_damping_min_baked,
                        self._adaptive_damping_max_baked,
                        self._adaptive_damping_threshold_baked,
                    ],
                    outputs=[self._adaptive_damping_buf],
                    device=self._device,
                )
                damping = self._adaptive_damping_buf
            if self._use_truncated_svd:
                wp.launch(
                    _qd_in_singular_basis_truncated_kernel,
                    dim=(controlled_robot_count, self._max_controlled_dofs),
                    inputs=[
                        self._svd_u_buf,
                        self._svd_s_buf,
                        self._pose_error_active_buf,
                        self._truncated_svd_threshold_baked,
                        self._task_dim,
                        self._controlled_dofs_per_robot,
                    ],
                    outputs=[self._qd_in_singular_basis_buf],
                    device=self._device,
                )
            else:
                wp.launch(
                    _qd_in_singular_basis_damped_kernel,
                    dim=(controlled_robot_count, self._max_controlled_dofs),
                    inputs=[
                        self._svd_u_buf,
                        self._svd_s_buf,
                        self._pose_error_active_buf,
                        damping,
                        self._task_dim,
                        self._controlled_dofs_per_robot,
                    ],
                    outputs=[self._qd_in_singular_basis_buf],
                    device=self._device,
                )
            wp.launch(
                _qd_from_singular_basis_kernel,
                dim=total_controlled_dofs,
                inputs=[
                    self._svd_v_buf,
                    self._qd_in_singular_basis_buf,
                    bandwidth,
                    self._robot_of_dof,
                    self._slot_of_dof,
                    self._controlled_dofs_per_robot,
                ],
                outputs=[self._qd_buf],
                device=self._device,
            )

        if self._use_null_space:
            null_space_damping = (
                self._null_space_damping_baked
                if self._null_space_damping_baked is not None
                else self._null_space_damping_buf
            )
            # _null_space_projector_kernel (shared with other controller
            # families) knows nothing about axis_weight -- gather the
            # Jacobian into compact slot order before feeding it, then
            # scatter the reconstructed pinv-transpose back to canonical
            # axis order afterward, so it sees a consistent pair of inputs.
            # The null-space projector's own regularization stays
            # deliberately unweighted (every axis weight 1), so
            # jacobian_active_buf itself -- not a separately weighted
            # buffer -- is the SVD's own input. Gathered/scattered by
            # null_space_active_axis_of_slot/null_space_task_dim, not the
            # primary solve's own active_axis_of_slot/task_dim -- see
            # null_space_axes.
            wp.launch(
                _gather_jacobian_by_axis_kernel,
                dim=(controlled_robot_count, 6, self._max_controlled_dofs),
                inputs=[self._jacobian_buf, self._null_space_active_axis_of_slot, self._null_space_task_dim],
                outputs=[self._jacobian_active_buf],
                device=self._device,
            )
            wp.launch(
                _svd_one_sided_jacobi_kernel,
                dim=controlled_robot_count,
                inputs=[
                    self._jacobian_active_view,
                    self._controlled_dofs_per_robot,
                    _JACOBI_SVD_TOL,
                    _JACOBI_SVD_MAX_SWEEPS,
                ],
                outputs=[self._svd_u_null_view, self._svd_s_null_view, self._svd_v_null_view],
                device=self._device,
            )
            wp.launch(
                _pinv_singular_value_damped_kernel,
                dim=(controlled_robot_count, self._max_controlled_dofs),
                inputs=[
                    self._svd_s_null_buf,
                    null_space_damping,
                    self._null_space_task_dim,
                    self._controlled_dofs_per_robot,
                ],
                outputs=[self._pinv_singular_value_null_buf],
                device=self._device,
            )
            wp.launch(
                _svd_reconstruct_scaled_kernel,
                dim=(controlled_robot_count, 6, self._max_controlled_dofs),
                inputs=[
                    self._svd_u_null_buf,
                    self._pinv_singular_value_null_buf,
                    self._svd_v_null_buf,
                    self._controlled_dofs_per_robot,
                ],
                outputs=[self._jacobian_pinv_transpose_slot_buf],
                device=self._device,
            )
            wp.launch(
                _scatter_pinv_transpose_by_axis_kernel,
                dim=(controlled_robot_count, 6, self._max_controlled_dofs),
                inputs=[
                    self._jacobian_pinv_transpose_slot_buf,
                    self._null_space_active_axis_of_slot,
                    self._null_space_task_dim,
                    self._controlled_dofs_per_robot,
                ],
                outputs=[self._jacobian_pinv_transpose_buf],
                device=self._device,
            )
            wp.launch(
                _null_space_projector_kernel,
                dim=(controlled_robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                inputs=[self._jacobian_buf, self._jacobian_pinv_transpose_buf, self._controlled_dofs_per_robot],
                outputs=[self._null_space_projector_buf],
                device=self._device,
            )

            dq_center_written = False
            if self._use_joint_limit_avoidance:
                wp.launch(
                    _joint_limit_avoidance_bias_kernel,
                    dim=total_controlled_dofs,
                    inputs=[
                        self._q_buf,
                        self._joint_pos_lower,
                        self._joint_pos_upper,
                        self._joint_limit_avoidance_gain,
                        self._joint_limit_avoidance_margin,
                    ],
                    outputs=[self._dq_center_buf],
                    device=self._device,
                )
                dq_center_written = True

            if self._use_null_space_posture_control:
                null_space_stiffness = (
                    self._null_space_stiffness_baked
                    if self._null_space_stiffness_baked is not None
                    else self._null_space_stiffness_buf
                )
                destination = self._dq_scratch_buf if dq_center_written else self._dq_center_buf
                wp.launch(
                    _posture_bias_kernel,
                    dim=total_controlled_dofs,
                    inputs=[self._q_buf, self._q_des_null_buf, null_space_stiffness],
                    outputs=[destination],
                    device=self._device,
                )
                if dq_center_written:
                    wp.launch(
                        _add_term_kernel,
                        dim=total_controlled_dofs,
                        inputs=[self._dq_scratch_buf],
                        outputs=[self._dq_center_buf],
                        device=self._device,
                    )

            wp.launch(
                _block_matrix_vector_multiply_kernel,
                dim=total_controlled_dofs,
                inputs=[
                    self._null_space_projector_buf,
                    self._dq_center_buf,
                    self._robot_of_dof,
                    self._slot_of_dof,
                    self._dof_offsets,
                    self._controlled_dofs_per_robot,
                ],
                outputs=[self._qd_null_buf],
                device=self._device,
            )
            wp.launch(
                _add_term_kernel,
                dim=total_controlled_dofs,
                inputs=[self._qd_null_buf],
                outputs=[self._qd_buf],
                device=self._device,
            )

        if isinstance(dt, wp.array):
            _validate_array(array=dt, name="dt", dtype=wp.float32, shape=(1,), device=self._device)
            dt_buf = dt
        else:
            self._dt_buf.fill_(float(dt))
            dt_buf = self._dt_buf

        wp.launch(
            _integrate_position_kernel,
            dim=total_controlled_dofs,
            inputs=[self._q_buf, self._qd_buf, dt_buf],
            outputs=[self._q_target_buf],
            device=self._device,
        )

        for buf, port in ((self._qd_buf, outputs.joint_qd_target), (self._q_target_buf, outputs.joint_q_target)):
            if isinstance(port, wp.indexedarray):
                wp.launch(
                    _scatter_port_kernel, dim=total_controlled_dofs, inputs=[buf], outputs=[port], device=self._device
                )
            else:
                wp.copy(port, buf)
