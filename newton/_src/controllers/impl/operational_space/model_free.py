# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerOperationalSpaceModelFree — task-space (operational-space) impedance
control with caller-supplied kinematics and dynamics terms.

Every port is compact: one entry per controlled DOF, robot 0's DOFs first,
then robot 1's — for `outputs.joint_f`, the controller's only per-DOF port.
Every other port is per-robot: exactly one entry per robot, since the task
itself is always 6-dimensional regardless of how many DOFs a robot has.

This increment implements motion control (a task-space spring-damper term,
with optional inertial decoupling through the operational-space mass matrix
Lambda), contact-wrench control combined through per-axis selection
matrices, gravity compensation, and null-space posture control.

Motion law (terms enabled at construction):

    F_motion = [Lambda if use_inertia_decoupling else I] · (Kp·pose_error + Kd·twist_error)

``Kp``/``Kd`` are specified per-axis in the tool-local frame (so "stiff along
the insertion axis" stays true as the tool reorients, instead of silently
meaning whatever world axis the insertion axis started aligned with), so —
like ``motion_selection_axes_tool``/``wrench_selection_axes_tool`` below —
they are rotated into world-frame 6x6 matrices every step before being
applied to ``pose_error``/``twist_error``, which are themselves expressed in
world coordinates. A rotated tool-local diagonal is generally no longer
diagonal in world frame (only block-diagonal), which is why this is a matrix
product rather than a per-axis multiply.

When ``use_partial_inertia_decoupling=True`` (only meaningful alongside
``use_inertia_decoupling=True``), Lambda is computed as two independent 3x3
inversions — the translational block from the Jacobian's first 3 rows, the
rotational block from its last 3 — instead of one 6x6 inversion, ignoring
the linear/angular coupling. Lambda ends up block-diagonal and is used
exactly the same way afterward.

Wrench law, only when ``use_wrench_feedforward`` or ``use_wrench_feedback`` is
enabled:

    F_wrench = [desired_wrench if use_wrench_feedforward else 0]
             + [Kp·(desired_wrench - measured_wrench) if use_wrench_feedback else 0]

where ``pose_error`` and ``twist_error`` are the task-space position/orientation
and linear/angular velocity errors between the current and desired tool pose.
``desired_wrench`` is the feedforward term, commanded directly; the second
term is a feedback correction toward that same setpoint from a measured
wrench, e.g. a 6-axis force/torque sensor reading. Either may be used alone —
``use_wrench_feedback`` with ``use_wrench_feedforward=False`` regulates the
measured wrench toward the setpoint with no separate feedforward term. The
feedback ``Kp`` here is tool-local and rotated the same way the motion gains
are.

When wrench control is enabled, ``F_motion`` and ``F_wrench`` are each
masked by a world-frame 6x6 selection matrix, itself rotated every step from
a fixed tool-local selection the same way the gains are (since which axes
are free to move vs. under force control is a property of the tool's
current orientation, not a fixed world direction), mapped to joint torques
separately, and summed there:

    tau = J^T · (selection_motion_world · F_motion) + J^T · (selection_wrench_world · F_wrench)

Without wrench control, every axis is motion-controlled and no selection
matrix is applied: ``tau = J^T · F_motion``.

When ``use_gravity_compensation=True``, ``inputs.gravity_force`` (the
caller-supplied gravity generalized forces, compact over the controlled
DOFs) is added directly to the summed joint torque.

When ``use_null_space_control=True``, a secondary joint-space posture task
is pursued only in directions that leave the task-space motion undisturbed:

    tau_null = N^T · [M(q)·a_posture if use_inertia_decoupling else a_posture]
    a_posture = Kp_null·(q_des_null - q) + Kd_null·(qd_des_null - qd)

``N^T = I - J^T · jacobian_pinv_transpose`` is the null-space projector,
built from whichever pseudo-inverse variant applies: the dynamically-consistent
one (``Lambda·J·M(q)^-1``, reusing the same Lambda and mass-matrix inverse the
motion term computes) when ``use_inertia_decoupling=True`` and
``use_partial_inertia_decoupling=False``, or the kinematics-only Moore-Penrose
one (``(J·J^T)^-1·J``) otherwise — the latter needs no mass matrix, so
null-space control still works without inertial decoupling. A block-diagonal
(partially-decoupled) Lambda does not have the property the
dynamically-consistent formula needs, so partial decoupling always falls back
to Moore-Penrose here too, even though a mass matrix is available. Only a
robot with more controlled DOFs than task dimensions (6) has a nontrivial
null space to work with.
``tau_null`` is mapped through the same compact per-DOF layout as every
other term here and summed into the joint torque the same way.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ...controller import ControllerBase
from ...utils import _validate_array
from .._common import (
    _add_term_kernel,
    _block_matrix_vector_multiply_kernel,
    _pd_term_kernel,
    _read_port,
    _scatter_port_kernel,
)
from ._common import (
    _apply_mass_matrix_inv_on_right_kernel,
    _apply_spatial_matrix_kernel,
    _invert_spd_block_kernel,
    _jacobian_times_jacobian_transpose_kernel,
    _jacobian_transpose_force_kernel,
    _null_space_projector_kernel,
    _operational_space_mass_matrix_inverse_kernel,
    _pose_error_kernel,
    _rotate_selection_matrix_kernel,
    _task_matrix_times_jacobian_kernel,
    _task_space_pd_kernel,
    _wrench_feedback_only_kernel,
    _wrench_feedforward_and_feedback_kernel,
)


def _validate_selection_axes_argument(
    value: Any, name: str, controlled_robot_count: int, device: wp.DeviceLike
) -> None:
    """Validate a selection-axes constructor argument: a wp.spatial_vector, or a per-robot wp.array."""
    if isinstance(value, wp.spatial_vector):
        return
    if isinstance(value, wp.array):
        _validate_array(
            array=value,
            name=name,
            dtype=wp.spatial_vector,
            shape=(controlled_robot_count,),
            device=device,
        )
        return
    raise TypeError(
        f"{name} must be a wp.array[wp.spatial_vector] of shape (controlled_robot_count,) or a "
        f"wp.spatial_vector of per-axis weights, got {type(value).__name__}."
    )


def _validate_gain_argument(value: Any, name: str, controlled_robot_count: int, device: wp.DeviceLike) -> None:
    """Validate a baked-gain constructor argument: a float, a wp.spatial_vector, or a per-robot wp.array."""
    if value is None:
        return
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a float, wp.spatial_vector, or wp.array[wp.spatial_vector], got bool.")
    if isinstance(value, (int, float)):
        return
    if isinstance(value, wp.spatial_vector):
        return
    if isinstance(value, wp.array):
        _validate_array(
            array=value,
            name=name,
            dtype=wp.spatial_vector,
            shape=(controlled_robot_count,),
            device=device,
        )
        return
    raise TypeError(
        f"{name} must be a float, a wp.spatial_vector, or a wp.array[wp.spatial_vector] of shape "
        f"(controlled_robot_count,), got {type(value).__name__}."
    )


def _validate_wrench_construction_arguments(
    *,
    use_wrench_feedforward: bool,
    use_wrench_feedback: bool,
    motion_selection_axes_tool: wp.array[wp.spatial_vector] | wp.spatial_vector | None,
    wrench_selection_axes_tool: wp.array[wp.spatial_vector] | wp.spatial_vector | None,
    wrench_stiffness: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None,
    controlled_robot_count: int,
    device: wp.DeviceLike,
) -> wp.array[wp.spatial_vector] | wp.spatial_vector | None:
    """Validate the wrench-control constructor arguments, and resolve ``motion_selection_axes_tool``'s default.

    Returns the resolved ``motion_selection_axes_tool``, or ``None`` when
    wrench control is disabled and every wrench-only argument was correctly
    left unset.
    """
    if not (use_wrench_feedforward or use_wrench_feedback):
        for name, value in (
            ("motion_selection_axes_tool", motion_selection_axes_tool),
            ("wrench_selection_axes_tool", wrench_selection_axes_tool),
            ("wrench_stiffness", wrench_stiffness),
        ):
            if value is not None:
                raise ValueError(
                    f"{name} is set, but use_wrench_feedforward and use_wrench_feedback are both False, "
                    f"so it would be ignored."
                )
        return None

    if wrench_selection_axes_tool is None:
        raise ValueError(
            "wrench_selection_axes_tool is required when use_wrench_feedforward or use_wrench_feedback is True."
        )
    if not use_wrench_feedback and wrench_stiffness is not None:
        raise ValueError("wrench_stiffness is set, but use_wrench_feedback=False, so it would be ignored.")

    motion_selection_axes_tool_resolved = (
        motion_selection_axes_tool
        if motion_selection_axes_tool is not None
        else wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    )
    _validate_selection_axes_argument(
        motion_selection_axes_tool_resolved, "motion_selection_axes_tool", controlled_robot_count, device
    )
    _validate_selection_axes_argument(
        wrench_selection_axes_tool, "wrench_selection_axes_tool", controlled_robot_count, device
    )
    if use_wrench_feedback:
        _validate_gain_argument(wrench_stiffness, "wrench_stiffness", controlled_robot_count, device)
    return motion_selection_axes_tool_resolved


class ControllerOperationalSpaceModelFree(ControllerBase):
    """Task-space (operational-space) impedance controller with caller-supplied kinematics and dynamics.

    Implements the operational-space motion-control law. This model-free
    variant expects the tool pose, tool twist, tool-point Jacobian, and
    (when ``use_inertia_decoupling=True``) the controlled-DOF mass matrix to
    be computed externally — it is the caller's responsibility to compute
    these correctly and write them into the input struct before every
    :meth:`step`.

    Every port is **per-robot**: a 1-D array with one entry per robot,
    ordered to match ``controlled_dofs_per_robot`` — except
    ``outputs.joint_f``, which is **compact**: one entry per controlled DOF,
    robot 0's DOFs first, then robot 1's.

    Every port, of any dtype, may be bound either to a plain array or to an
    indexed view of a simulation-sized array — including the
    ``wp.transform``/``wp.spatial_vector`` ports (tool pose, tool twist,
    gains), via the ``wp.transform``/``wp.spatial_vector`` gather kernels in
    ``controllers/impl/_common.py``.

    Array shapes and devices are validated on each direct call to
    :meth:`step`, but not when a captured graph is replayed, since the
    checks run in Python at capture time only.

    Supports heterogeneous robot fleets — robots may have different
    controlled-DOF counts. Only the Jacobian and mass matrix are padded, to
    ``max_controlled_dofs``; every other buffer is compact or per-robot.

    Allocate input and output structs via :meth:`input` and :meth:`output`.
    All field names on those structs are fixed — see :class:`Inputs` and
    :class:`Outputs` for the typed schema. Fields for disabled features
    (e.g. ``mass_matrix`` when ``use_inertia_decoupling=False``) are
    allocated as ``None`` and must not be written.

    Args:
        controlled_dofs_per_robot: Controlled-DOF count for each robot. Its
            length sets :attr:`controlled_robot_count` (the length of every
            per-robot port), its sum sets :attr:`total_controlled_dofs` (the
            length of ``outputs.joint_f``), and its maximum sets
            :attr:`max_controlled_dofs` (the padded width of the Jacobian and
            mass matrix). Every entry must be positive, and — when
            ``use_inertia_decoupling=True`` — at least 6, since the
            operational-space mass matrix is only invertible for a robot
            whose Jacobian can span all 6 task dimensions.
        motion_stiffness: Task-space position/orientation-error gain Kp,
            per-axis in the tool-local frame — rotated into world every step
            before being applied, the same way ``motion_selection_axes_tool``
            below is, so e.g. "stiff along the insertion axis" stays true as
            the tool reorients. Units depend on ``use_inertia_decoupling``:
            [1/s²] when enabled, since the spring-damper term is then a
            task-space acceleration premultiplied by Lambda; otherwise [N/m]
            on the position axes and [N·m/rad] on the orientation axes. Pass
            a scalar to apply the same gain to every axis of every robot, a
            ``wp.spatial_vector`` to apply the same 6 per-axis gains to every
            robot, an array of shape [controlled_robot_count] to set them
            individually (one ``wp.spatial_vector`` of 6 gains per robot), or
            ``None`` to read ``inputs.motion_stiffness`` each step.
        motion_damping: Task-space velocity-error gain Kd, tool-local like
            ``motion_stiffness``, [1/s] when ``use_inertia_decoupling`` is
            enabled, otherwise [N·s/m] on the position axes and [N·m·s/rad]
            on the orientation axes. Same format as ``motion_stiffness``.
        use_inertia_decoupling: Premultiply the task-space spring-damper term
            by Lambda, the operational-space mass matrix.
        use_partial_inertia_decoupling: Compute Lambda as two independent 3x3
            inversions (translation, rotation) instead of one 6x6 inversion,
            ignoring the linear/angular coupling. Only meaningful when
            ``use_inertia_decoupling=True``.
        use_gravity_compensation: Add ``inputs.gravity_force`` directly to
            the summed joint torque.
        use_wrench_feedforward: Command the desired wrench directly, as a
            feedforward term in the wrench law, combined with motion control
            through per-axis selection matrices. When both this and
            ``use_wrench_feedback`` are ``False``, every axis is
            motion-controlled and ``motion_selection_axes_tool``/
            ``wrench_selection_axes_tool``/``wrench_stiffness`` must all be
            left at their defaults.
        use_wrench_feedback: Correct the wrench command by
            ``Kp · (desired - measured)`` using ``inputs.measured_wrench_world``
            each step, as a feedback term in the wrench law. May be enabled
            with or without ``use_wrench_feedforward``: without it, the
            command is the feedback correction alone, regulating the
            measured wrench toward the desired setpoint with no separate
            feedforward term.
        motion_selection_axes_tool: Diagonal selection weight per task axis
            (0/1, or any scalar weight), tool-local: (linear x, y, z, angular
            x, y, z). Rotated into a world-frame selection matrix every step
            and applied to the motion term before it is mapped to joint
            torques. Pass a ``wp.spatial_vector`` to apply the same weights
            to every robot, or an array of shape [controlled_robot_count] to
            set them individually. Only meaningful when wrench control is
            enabled; defaults to every axis motion-controlled,
            ``wp.spatial_vector(1, 1, 1, 1, 1, 1)``.
            Usually the complement of ``wrench_selection_axes_tool`` — each
            axis under motion control, not force control, and vice versa —
            but that is not enforced: nothing here requires the two to
            partition the 6 axes.
        wrench_selection_axes_tool: Diagonal selection weight per task axis,
            tool-local, same format as ``motion_selection_axes_tool``, applied
            to the wrench term. Required when wrench control is enabled.
            Usually the complement of ``motion_selection_axes_tool``, but
            that is not enforced — see its docstring above.
        wrench_stiffness: Contact-wrench proportional feedback gain Kp,
            tool-local like ``motion_stiffness`` (rotated into world every
            step before being applied), [N/m] on the force axes and [N·m/rad]
            on the moment axes. Same format as ``motion_stiffness``. Only
            meaningful when ``use_wrench_feedback=True``.
        use_null_space_control: Pursue a secondary joint-space posture task
            in the null space of the primary task, so it does not disturb
            task-space motion. Requires every robot to have more than 6
            controlled DOFs (redundant relative to the 6D task). Which
            pseudo-inverse variant builds the null-space projector follows
            ``use_inertia_decoupling``: dynamically-consistent (reusing the
            motion term's Lambda and mass-matrix inverse) when it is
            ``True``, or the kinematics-only Moore-Penrose variant (no mass
            matrix needed) when it is ``False``.
        null_space_stiffness: Joint-space posture position-error gain Kp.
            Units depend on ``use_inertia_decoupling``: [1/s²] when enabled,
            since the posture PD term is then premultiplied by the mass
            matrix; otherwise [N/m or N·m/rad]. Pass a scalar to apply the
            same gain to every controlled DOF, an array of shape
            [total_controlled_dofs] to set them individually, or ``None`` to
            read ``inputs.null_space_stiffness`` each step. Only meaningful
            when ``use_null_space_control=True``.
        null_space_damping: Joint-space posture velocity-error gain Kd,
            [1/s] when ``use_inertia_decoupling`` is enabled, otherwise
            [N·s/m or N·m·s/rad]. Same format as ``null_space_stiffness``.
        device: Warp device.
        requires_grad: Whether internal buffers need gradient support.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerOperationalSpaceModelFree.input`.

        Every field is per-robot, shape [controlled_robot_count], except
        ``jacobian_tool_world`` and ``mass_matrix``, which are padded to
        [controlled_robot_count, ..., max_controlled_dofs]. Optional fields
        are ``None`` when the corresponding feature is disabled at
        construction.
        """

        tool_pose_world: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Current world pose of the tool frame, shape [controlled_robot_count]."""
        tool_twist_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Current tool twist (linear, angular) in world coordinates [m/s, rad/s], shape [controlled_robot_count]."""
        jacobian_tool_world: wp.array3d[wp.float32] | wp.indexedarray(dtype=wp.float32, ndim=3)
        """Tool-point Jacobian in world coordinates, shape [controlled_robot_count, 6, max_controlled_dofs]."""
        mass_matrix: wp.array3d[wp.float32] | wp.indexedarray(dtype=wp.float32, ndim=3) | None
        """Joint-space mass matrix over the controlled DOFs, shape [controlled_robot_count, max_controlled_dofs, max_controlled_dofs]; a robot with fewer than ``max_controlled_dofs`` DOFs leaves the trailing rows and columns unread. Units by row/column DOF type: [kg] translational, [kg·m] mixed, [kg·m²] rotational. ``None`` unless ``use_inertia_decoupling=True``."""
        gravity_force: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Gravity generalized forces [N or N·m], compact, shape [total_controlled_dofs]. ``None`` unless ``use_gravity_compensation=True``."""
        desired_tool_pose_world: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Desired world pose of the tool frame, shape [controlled_robot_count]."""
        desired_twist_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Desired tool twist (linear, angular) in world coordinates [m/s, rad/s], shape [controlled_robot_count]."""
        motion_stiffness: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Task-space position/orientation-error gain Kp, tool-local (rotated into world every step), shape [controlled_robot_count]. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m] / [N·m/rad]. ``None`` when gains are baked at construction."""
        motion_damping: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Task-space velocity-error gain Kd, tool-local (rotated into world every step), shape [controlled_robot_count]. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m] / [N·m·s/rad]. ``None`` when gains are baked at construction."""
        desired_wrench_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Desired contact wrench (force, moment) in world coordinates [N, N·m], shape [controlled_robot_count] — the feedforward term, and/or the feedback setpoint. ``None`` unless wrench control is enabled."""
        measured_wrench_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Measured contact wrench (force, moment) in world coordinates [N, N·m], shape [controlled_robot_count], e.g. from a 6-axis force/torque sensor. ``None`` unless ``use_wrench_feedback=True``."""
        wrench_stiffness: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Contact-wrench proportional feedback gain Kp, tool-local (rotated into world every step), shape [controlled_robot_count]. [N/m] on the force axes, [N·m/rad] on the moment axes. ``None`` when gains are baked at construction, or when ``use_wrench_feedback=False``."""
        joint_q: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Current joint positions [m or rad], compact, shape [total_controlled_dofs]. ``None`` unless ``use_null_space_control=True``."""
        joint_qd: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Current joint velocities [m/s or rad/s], compact, shape [total_controlled_dofs]. ``None`` unless ``use_null_space_control=True``."""
        joint_q_des_null: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Desired joint positions for the null-space posture task [m or rad], compact, shape [total_controlled_dofs]. ``None`` unless ``use_null_space_control=True``."""
        joint_qd_des_null: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Desired joint velocities for the null-space posture task [m/s or rad/s], compact, shape [total_controlled_dofs]. ``None`` unless ``use_null_space_control=True``."""
        null_space_stiffness: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Joint-space posture position-error gain Kp, compact, shape [total_controlled_dofs]. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m or N·m/rad]. ``None`` when gains are baked at construction, or when ``use_null_space_control=False``."""
        null_space_damping: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Joint-space posture velocity-error gain Kd, compact, shape [total_controlled_dofs]. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m or N·m·s/rad]. ``None`` when gains are baked at construction, or when ``use_null_space_control=False``."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerOperationalSpaceModelFree.output`."""

        joint_f: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Joint torque command [N or N·m], shape [total_controlled_dofs]."""

    def __init__(
        self,
        *,
        controlled_dofs_per_robot: wp.array[wp.int32],
        motion_stiffness: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None,
        motion_damping: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None,
        use_inertia_decoupling: bool = True,
        use_partial_inertia_decoupling: bool = False,
        use_gravity_compensation: bool = True,
        use_wrench_feedforward: bool = False,
        use_wrench_feedback: bool = False,
        motion_selection_axes_tool: wp.array[wp.spatial_vector] | wp.spatial_vector | None = None,
        wrench_selection_axes_tool: wp.array[wp.spatial_vector] | wp.spatial_vector | None = None,
        wrench_stiffness: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None = None,
        use_null_space_control: bool = False,
        null_space_stiffness: wp.array[wp.float32] | float | None = None,
        null_space_damping: wp.array[wp.float32] | float | None = None,
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
        if use_inertia_decoupling and controlled_dofs_per_robot_np.min() < 6:
            # Lambda^-1 = J M^-1 J^T only has rank min(6, controlled_dof_count):
            # with fewer than 6 controlled DOFs it is genuinely singular, not
            # just ill-conditioned, so inertial decoupling would silently
            # produce huge, physically meaningless forces along the
            # uncontrollable task directions instead of erroring.
            raise ValueError(
                f"use_inertia_decoupling=True requires every robot to have at least 6 controlled DOFs, "
                f"since the operational-space mass matrix is only invertible when the Jacobian can span "
                f"all 6 task dimensions; got controlled_dofs_per_robot={controlled_dofs_per_robot_np.tolist()}. "
                f"Pass use_inertia_decoupling=False for an under-actuated robot."
            )
        if use_partial_inertia_decoupling and not use_inertia_decoupling:
            raise ValueError(
                "use_partial_inertia_decoupling=True requires use_inertia_decoupling=True, so it would be ignored."
            )
        if use_null_space_control and controlled_dofs_per_robot_np.min() <= 6:
            # A robot with 6 or fewer controlled DOFs has no DOF left over
            # once the 6D task is satisfied, so its null space is trivial
            # (rank 0) -- there is nothing left for a posture task to move
            # in without disturbing the primary task.
            raise ValueError(
                f"use_null_space_control=True requires every robot to have more than 6 controlled DOFs, "
                f"since a robot with 6 or fewer has no null space left over from the 6D task; got "
                f"controlled_dofs_per_robot={controlled_dofs_per_robot_np.tolist()}."
            )

        max_controlled_dofs = int(controlled_dofs_per_robot_np.max())
        total_controlled_dofs = int(controlled_dofs_per_robot_np.sum())

        for name, value in (("motion_stiffness", motion_stiffness), ("motion_damping", motion_damping)):
            _validate_gain_argument(value, name, controlled_robot_count, self._device)

        motion_selection_axes_tool_resolved = _validate_wrench_construction_arguments(
            use_wrench_feedforward=use_wrench_feedforward,
            use_wrench_feedback=use_wrench_feedback,
            motion_selection_axes_tool=motion_selection_axes_tool,
            wrench_selection_axes_tool=wrench_selection_axes_tool,
            wrench_stiffness=wrench_stiffness,
            controlled_robot_count=controlled_robot_count,
            device=self._device,
        )

        if not use_null_space_control:
            for name, value in (
                ("null_space_stiffness", null_space_stiffness),
                ("null_space_damping", null_space_damping),
            ):
                if value is not None:
                    raise ValueError(f"{name} is set, but use_null_space_control=False, so it would be ignored.")
        else:
            for name, value in (
                ("null_space_stiffness", null_space_stiffness),
                ("null_space_damping", null_space_damping),
            ):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    continue  # broadcast at bake time, not a wp.array to validate
                _validate_array(
                    array=value,
                    name=name,
                    dtype=wp.float32,
                    shape=(total_controlled_dofs,),
                    device=self._device,
                    required=False,
                )
        # ------------------------------------------------------------------

        self._controlled_robot_count = controlled_robot_count
        self._max_controlled_dofs = max_controlled_dofs
        self._total_controlled_dofs = total_controlled_dofs
        self._use_inertia = bool(use_inertia_decoupling)
        self._use_partial_inertia = bool(use_partial_inertia_decoupling)
        self._use_gravity = bool(use_gravity_compensation)
        self._use_wrench_feedforward = bool(use_wrench_feedforward)
        self._use_wrench_feedback = bool(use_wrench_feedback)
        self._use_wrench = self._use_wrench_feedforward or self._use_wrench_feedback
        self._use_null_space = bool(use_null_space_control)
        self._requires_grad = requires_grad

        # Copied, not stored: the kernels below use this as a loop bound
        # while the tables below it are derived from the same host
        # snapshot, so a later edit to the caller's array would send a
        # multiply past the end of a buffer.
        self._controlled_dofs_per_robot = wp.array(controlled_dofs_per_robot_np, dtype=wp.int32, device=self._device)

        # Flat-DOF -> (robot, slot) tables, needed so the final
        # Jacobian-transpose force mapping can write directly into the
        # compact total_controlled_dofs layout.
        self._robot_of_dof = wp.array(
            np.repeat(np.arange(controlled_robot_count, dtype=np.int32), controlled_dofs_per_robot_np),
            dtype=wp.int32,
            device=self._device,
        )
        self._slot_of_dof = wp.array(
            np.concatenate([np.arange(dof_count, dtype=np.int32) for dof_count in controlled_dofs_per_robot_np]),
            dtype=wp.int32,
            device=self._device,
        )

        # Per-robot flat starting index into the compact total_controlled_dofs
        # layout, needed only by the null-space posture term's block-matrix
        # (mass matrix, then null-space projector) multiplies.
        self._dof_offsets: wp.array[wp.int32] | None = None
        if self._use_null_space:
            dof_offsets_np = np.zeros(controlled_robot_count, dtype=np.int32)
            dof_offsets_np[1:] = np.cumsum(controlled_dofs_per_robot_np[:-1])
            self._dof_offsets = wp.array(dof_offsets_np, dtype=wp.int32, device=self._device)

        self._stiffness_baked = self._bake_gain(motion_stiffness)
        self._damping_baked = self._bake_gain(motion_damping)

        def _pose_buf():
            return wp.zeros(
                controlled_robot_count, dtype=wp.transform, device=self._device, requires_grad=requires_grad
            )

        def _twist_buf():
            return wp.zeros(
                controlled_robot_count, dtype=wp.spatial_vector, device=self._device, requires_grad=requires_grad
            )

        # Every port is copied into one of these before any kernel runs, so
        # graph replay always reads through stable buffers regardless of
        # what array object the caller binds between steps.
        self._pose_buf = _pose_buf()
        self._twist_buf = _twist_buf()
        self._desired_pose_buf = _pose_buf()
        self._desired_twist_buf = _twist_buf()
        self._jacobian_buf = wp.zeros(
            (controlled_robot_count, 6, max_controlled_dofs),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )
        self._mass_matrix_buf: wp.array3d[wp.float32] | None = (
            wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            if self._use_inertia
            else None
        )
        self._stiffness_buf: wp.array[wp.spatial_vector] | None = (
            _twist_buf() if self._stiffness_baked is None else None
        )
        self._damping_buf: wp.array[wp.spatial_vector] | None = _twist_buf() if self._damping_baked is None else None

        # Kp/Kd are specified per-axis in the tool-local frame (so e.g. "stiff
        # along the insertion axis" stays true as the tool reorients), so
        # they're rotated into these world-frame matrices every step, via
        # the same _rotate_selection_matrix_kernel used for
        # motion_selection_axes_tool/wrench_selection_axes_tool below --
        # needed unconditionally, not just when wrench control is enabled,
        # since the tool's orientation (and hence the rotation) changes
        # every step regardless.
        def _gain_matrix_buf():
            return wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )

        self._motion_stiffness_matrix_world = _gain_matrix_buf()
        self._motion_damping_matrix_world = _gain_matrix_buf()

        self._motion_selection_axes_tool: wp.array[wp.spatial_vector] | None = None
        self._wrench_selection_axes_tool: wp.array[wp.spatial_vector] | None = None
        self._motion_selection_matrix_world: wp.array3d[wp.float32] | None = None
        self._wrench_selection_matrix_world: wp.array3d[wp.float32] | None = None
        self._masked_motion_force_buf: wp.array[wp.spatial_vector] | None = None
        self._desired_wrench_buf: wp.array[wp.spatial_vector] | None = None
        self._measured_wrench_buf: wp.array[wp.spatial_vector] | None = None
        self._wrench_command_buf: wp.array[wp.spatial_vector] | None = None
        self._masked_wrench_force_buf: wp.array[wp.spatial_vector] | None = None
        self._wrench_tau_buf: wp.array[wp.float32] | None = None
        self._wrench_stiffness_baked: wp.array[wp.spatial_vector] | None = None
        self._wrench_stiffness_buf: wp.array[wp.spatial_vector] | None = None
        self._wrench_stiffness_matrix_world: wp.array3d[wp.float32] | None = None
        if self._use_wrench:
            self._motion_selection_axes_tool = self._bake_axes(motion_selection_axes_tool_resolved)
            self._wrench_selection_axes_tool = self._bake_axes(wrench_selection_axes_tool)
            self._motion_selection_matrix_world = wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._wrench_selection_matrix_world = wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._masked_motion_force_buf = _twist_buf()
            self._desired_wrench_buf = _twist_buf()
            self._masked_wrench_force_buf = _twist_buf()
            self._wrench_tau_buf = wp.zeros(
                total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            if self._use_wrench_feedback:
                self._measured_wrench_buf = _twist_buf()
                self._wrench_command_buf = _twist_buf()
                self._wrench_stiffness_baked = self._bake_gain(wrench_stiffness)
                self._wrench_stiffness_buf = _twist_buf() if self._wrench_stiffness_baked is None else None
                self._wrench_stiffness_matrix_world = _gain_matrix_buf()

        self._pose_error_buf = _twist_buf()
        self._desired_task_acceleration_buf = _twist_buf()
        self._task_space_force_buf: wp.array[wp.spatial_vector] | None = _twist_buf() if self._use_inertia else None

        # Lambda's Cholesky scratch and inverse-mass-matrix Cholesky scratch,
        # only needed when inertial decoupling is enabled.
        self._mass_matrix_cholesky: wp.array3d[wp.float32] | None = None
        self._mass_matrix_inv: wp.array3d[wp.float32] | None = None
        self._operational_space_mass_matrix_inv: wp.array3d[wp.float32] | None = None
        self._operational_space_mass_matrix_cholesky: wp.array3d[wp.float32] | None = None
        self._operational_space_mass_matrix: wp.array3d[wp.float32] | None = None
        self._task_dim: wp.array[wp.int32] | None = None
        if self._use_inertia:
            self._mass_matrix_cholesky = wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._mass_matrix_inv = wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._operational_space_mass_matrix_inv = wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._operational_space_mass_matrix_cholesky = wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._operational_space_mass_matrix = wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
        if self._use_inertia or self._use_null_space:
            # Shared by Lambda's 6x6 inverse above and, when null-space
            # control uses the kinematics-only Moore-Penrose pseudo-inverse
            # below, (J @ J^T)'s 6x6 inverse -- both are always exactly 6x6.
            self._task_dim = wp.full(controlled_robot_count, 6, dtype=wp.int32, device=self._device)
        self._partial_task_dim: wp.array[wp.int32] | None = None
        if self._use_partial_inertia:
            # block_dim for Lambda's two independent 3x3 (translation, rotation) inversions.
            self._partial_task_dim = wp.full(controlled_robot_count, 3, dtype=wp.int32, device=self._device)

        self._tau_buf = wp.zeros(
            total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
        )
        self._grav_buf: wp.array[wp.float32] | None = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad)
            if self._use_gravity
            else None
        )

        def _compact_buf():
            return wp.zeros(total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad)

        self._joint_q_buf: wp.array[wp.float32] | None = None
        self._joint_qd_buf: wp.array[wp.float32] | None = None
        self._joint_q_des_null_buf: wp.array[wp.float32] | None = None
        self._joint_qd_des_null_buf: wp.array[wp.float32] | None = None
        self._null_stiffness_baked: wp.array[wp.float32] | None = None
        self._null_stiffness_buf: wp.array[wp.float32] | None = None
        self._null_damping_baked: wp.array[wp.float32] | None = None
        self._null_damping_buf: wp.array[wp.float32] | None = None
        self._posture_acc_buf: wp.array[wp.float32] | None = None
        self._posture_force_buf: wp.array[wp.float32] | None = None
        self._null_space_jacobian_pinv_transpose: wp.array3d[wp.float32] | None = None
        self._null_space_jacobian_pinv_transpose_stage: wp.array3d[wp.float32] | None = None
        self._null_space_jjt: wp.array3d[wp.float32] | None = None
        self._null_space_jjt_cholesky: wp.array3d[wp.float32] | None = None
        self._null_space_jjt_inv: wp.array3d[wp.float32] | None = None
        self._null_space_projector: wp.array3d[wp.float32] | None = None
        self._null_space_tau_buf: wp.array[wp.float32] | None = None
        if self._use_null_space:
            self._joint_q_buf = _compact_buf()
            self._joint_qd_buf = _compact_buf()
            self._joint_q_des_null_buf = _compact_buf()
            self._joint_qd_des_null_buf = _compact_buf()
            self._null_stiffness_baked = self._bake_joint_gain(null_space_stiffness)
            self._null_stiffness_buf = _compact_buf() if self._null_stiffness_baked is None else None
            self._null_damping_baked = self._bake_joint_gain(null_space_damping)
            self._null_damping_buf = _compact_buf() if self._null_damping_baked is None else None
            self._posture_acc_buf = _compact_buf()
            self._null_space_jacobian_pinv_transpose = wp.zeros(
                (controlled_robot_count, 6, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._null_space_projector = wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._null_space_tau_buf = _compact_buf()
            if self._use_inertia:
                # The mass matrix stays fully valid even with partial inertia
                # decoupling (only Lambda becomes block-diagonal), so the
                # posture PD term is still premultiplied by it.
                self._posture_force_buf = _compact_buf()
            if self._use_inertia and not self._use_partial_inertia:
                # Dynamically-consistent pseudo-inverse transpose, Lambda @ J
                # @ M^-1: reuses the motion term's Lambda and mass-matrix
                # inverse, computed once per step regardless of null-space
                # control, so only the intermediate Lambda @ J needs its own
                # scratch buffer here.
                self._null_space_jacobian_pinv_transpose_stage = wp.zeros(
                    (controlled_robot_count, 6, max_controlled_dofs),
                    dtype=wp.float32,
                    device=self._device,
                    requires_grad=requires_grad,
                )
            else:
                # Moore-Penrose pseudo-inverse transpose, (J @ J^T)^-1 @ J:
                # kinematics-only, needs no mass matrix. Also the fallback
                # when partial inertia decoupling leaves Lambda block-diagonal,
                # since that Lambda doesn't have the property the
                # dynamically-consistent formula needs.
                self._null_space_jjt = wp.zeros(
                    (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
                )
                self._null_space_jjt_cholesky = wp.zeros(
                    (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
                )
                self._null_space_jjt_inv = wp.zeros(
                    (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
                )

    def _bake_gain(
        self, value: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None
    ) -> wp.array[wp.spatial_vector] | None:
        """Broadcast a scalar or wp.spatial_vector, or copy a gain array, into a fresh per-robot buffer.

        Returns ``None`` for live gains, which are read from the input struct
        each step instead. A wp.array is already validated by
        :func:`_validate_array`.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            v = float(value)
            return wp.full(
                self._controlled_robot_count,
                wp.spatial_vector(v, v, v, v, v, v),
                dtype=wp.spatial_vector,
                device=self._device,
                requires_grad=self._requires_grad,
            )
        if isinstance(value, wp.spatial_vector):
            return wp.full(
                self._controlled_robot_count,
                value,
                dtype=wp.spatial_vector,
                device=self._device,
                requires_grad=self._requires_grad,
            )
        baked = wp.zeros(
            self._controlled_robot_count,
            dtype=wp.spatial_vector,
            device=self._device,
            requires_grad=self._requires_grad,
        )
        wp.copy(baked, value)
        return baked

    def _bake_axes(self, value: wp.array[wp.spatial_vector] | wp.spatial_vector) -> wp.array[wp.spatial_vector]:
        """Broadcast a wp.spatial_vector, or copy a per-robot array, of tool-local selection weights into a fresh buffer.

        A wp.array is already validated by :func:`_validate_array`.
        """
        if isinstance(value, wp.array):
            baked = wp.zeros(
                self._controlled_robot_count,
                dtype=wp.spatial_vector,
                device=self._device,
                requires_grad=self._requires_grad,
            )
            wp.copy(baked, value)
            return baked
        return wp.full(
            self._controlled_robot_count,
            value,
            dtype=wp.spatial_vector,
            device=self._device,
            requires_grad=self._requires_grad,
        )

    def _bake_joint_gain(self, value: wp.array[wp.float32] | float | None) -> wp.array[wp.float32] | None:
        """Broadcast a scalar, or copy a gain array, into a fresh joint-space (compact, per-DOF) buffer.

        Returns ``None`` for live gains, which are read from the input struct
        each step instead. A wp.array is already validated by
        :func:`_validate_array`.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return wp.full(
                self._total_controlled_dofs,
                float(value),
                dtype=wp.float32,
                device=self._device,
                requires_grad=self._requires_grad,
            )
        baked = wp.zeros(
            self._total_controlled_dofs,
            dtype=wp.float32,
            device=self._device,
            requires_grad=self._requires_grad,
        )
        wp.copy(baked, value)
        return baked

    @property
    def controlled_robot_count(self) -> int:
        """Number of robots, i.e. the length of ``controlled_dofs_per_robot``."""
        return self._controlled_robot_count

    @property
    def max_controlled_dofs(self) -> int:
        """Largest controlled-DOF count over the robots, the padded width of the Jacobian and mass matrix."""
        return self._max_controlled_dofs

    @property
    def total_controlled_dofs(self) -> int:
        """Total controlled-DOF count across all robots, the length of ``outputs.joint_f``."""
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
        robot_count = self._controlled_robot_count

        inputs = ControllerOperationalSpaceModelFree.Inputs()
        inputs.tool_pose_world = wp.zeros(robot_count, dtype=wp.transform, device=device, requires_grad=requires_grad)
        inputs.tool_twist_world = wp.zeros(
            robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad
        )
        inputs.jacobian_tool_world = wp.zeros(
            (robot_count, 6, self._max_controlled_dofs), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        inputs.mass_matrix = (
            wp.zeros(
                (robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                dtype=wp.float32,
                device=device,
                requires_grad=requires_grad,
            )
            if self._use_inertia
            else None
        )
        inputs.gravity_force = (
            wp.zeros(self._total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_gravity
            else None
        )
        inputs.desired_tool_pose_world = wp.zeros(
            robot_count, dtype=wp.transform, device=device, requires_grad=requires_grad
        )
        inputs.desired_twist_world = wp.zeros(
            robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad
        )
        inputs.motion_stiffness = (
            wp.zeros(robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad)
            if self._stiffness_baked is None
            else None
        )
        inputs.motion_damping = (
            wp.zeros(robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad)
            if self._damping_baked is None
            else None
        )
        inputs.desired_wrench_world = (
            wp.zeros(robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad)
            if self._use_wrench
            else None
        )
        inputs.measured_wrench_world = (
            wp.zeros(robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad)
            if self._use_wrench_feedback
            else None
        )
        inputs.wrench_stiffness = (
            wp.zeros(robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad)
            if self._use_wrench_feedback and self._wrench_stiffness_baked is None
            else None
        )
        inputs.joint_q = (
            wp.zeros(self._total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space
            else None
        )
        inputs.joint_qd = (
            wp.zeros(self._total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space
            else None
        )
        inputs.joint_q_des_null = (
            wp.zeros(self._total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space
            else None
        )
        inputs.joint_qd_des_null = (
            wp.zeros(self._total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space
            else None
        )
        inputs.null_space_stiffness = (
            wp.zeros(self._total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space and self._null_stiffness_baked is None
            else None
        )
        inputs.null_space_damping = (
            wp.zeros(self._total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space and self._null_damping_baked is None
            else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with a compact torque array."""
        outputs = ControllerOperationalSpaceModelFree.Outputs()
        outputs.joint_f = wp.zeros(
            self._total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
        )
        return outputs

    def step(
        self,
        *,
        inputs: Inputs,
        outputs: Outputs,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Compute one operational-space motion-control step and write joint torques.

        Args:
            inputs: Populated :class:`Inputs` struct. Kinematic and dynamics
                fields must be filled by the caller before each call.
            outputs: :class:`Outputs` struct to write torques into.
            dt: Unused. Accepted for API compatibility.
        """
        robot_count = self._controlled_robot_count

        # A port belonging to a disabled feature is never read, so writing
        # one would go unnoticed. getattr because a caller may leave the
        # field unset rather than None.
        for name, enabled, switch in (
            ("mass_matrix", self._use_inertia, "use_inertia_decoupling"),
            ("gravity_force", self._use_gravity, "use_gravity_compensation"),
            ("motion_stiffness", self._stiffness_baked is None, "a live motion_stiffness"),
            ("motion_damping", self._damping_baked is None, "a live motion_damping"),
            ("desired_wrench_world", self._use_wrench, "use_wrench_feedforward or use_wrench_feedback"),
            ("measured_wrench_world", self._use_wrench_feedback, "use_wrench_feedback"),
            (
                "wrench_stiffness",
                self._use_wrench_feedback and self._wrench_stiffness_baked is None,
                "a live wrench_stiffness",
            ),
            ("joint_q", self._use_null_space, "use_null_space_control"),
            ("joint_qd", self._use_null_space, "use_null_space_control"),
            ("joint_q_des_null", self._use_null_space, "use_null_space_control"),
            ("joint_qd_des_null", self._use_null_space, "use_null_space_control"),
            (
                "null_space_stiffness",
                self._use_null_space and self._null_stiffness_baked is None,
                "a live null_space_stiffness",
            ),
            (
                "null_space_damping",
                self._use_null_space and self._null_damping_baked is None,
                "a live null_space_damping",
            ),
        ):
            if not enabled and getattr(inputs, name, None) is not None:
                raise ValueError(
                    f"inputs.{name} is set, but the controller was built without {switch}, so the value "
                    f"would be ignored."
                )

        # Per-robot (transform/spatial_vector) ports: may be bound to a plain
        # array or to an indexed view of a simulation-sized array, via the
        # same graph-capture-safe port machinery outputs.joint_f uses below.
        for port, name, dtype, buf in (
            (
                inputs.tool_pose_world,
                "inputs.tool_pose_world",
                wp.transform,
                self._pose_buf,
            ),
            (inputs.tool_twist_world, "inputs.tool_twist_world", wp.spatial_vector, self._twist_buf),
            (
                inputs.desired_tool_pose_world,
                "inputs.desired_tool_pose_world",
                wp.transform,
                self._desired_pose_buf,
            ),
            (inputs.desired_twist_world, "inputs.desired_twist_world", wp.spatial_vector, self._desired_twist_buf),
        ):
            _validate_array(
                array=port, name=name, dtype=dtype, shape=(robot_count,), device=self._device, allow_indexed=True
            )
            _read_port(port, buf, robot_count, self._device)

        if self._stiffness_baked is None:
            _validate_array(
                array=inputs.motion_stiffness,
                name="inputs.motion_stiffness",
                dtype=wp.spatial_vector,
                shape=(robot_count,),
                device=self._device,
                allow_indexed=True,
            )
            _read_port(inputs.motion_stiffness, self._stiffness_buf, robot_count, self._device)
        if self._damping_baked is None:
            _validate_array(
                array=inputs.motion_damping,
                name="inputs.motion_damping",
                dtype=wp.spatial_vector,
                shape=(robot_count,),
                device=self._device,
                allow_indexed=True,
            )
            _read_port(inputs.motion_damping, self._damping_buf, robot_count, self._device)

        # Jacobian and (optional) mass matrix: plain float32 arrays, so they
        # reuse the shared, view-aware port machinery.
        _validate_array(
            array=inputs.jacobian_tool_world,
            name="inputs.jacobian_tool_world",
            dtype=wp.float32,
            shape=(robot_count, 6, self._max_controlled_dofs),
            device=self._device,
            allow_indexed=True,
        )
        _read_port(
            inputs.jacobian_tool_world, self._jacobian_buf, (robot_count, 6, self._max_controlled_dofs), self._device
        )

        # Inertial decoupling: read the mass matrix, needed for Lambda below.
        if self._use_inertia:
            _validate_array(
                array=inputs.mass_matrix,
                name="inputs.mass_matrix",
                dtype=wp.float32,
                shape=(robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                device=self._device,
                allow_indexed=True,
            )
            _read_port(
                inputs.mass_matrix,
                self._mass_matrix_buf,
                (robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                self._device,
            )

        # Gravity compensation: read the caller-supplied compact torque term.
        if self._use_gravity:
            _validate_array(
                array=inputs.gravity_force,
                name="inputs.gravity_force",
                dtype=wp.float32,
                shape=(self._total_controlled_dofs,),
                device=self._device,
                allow_indexed=True,
            )
            _read_port(inputs.gravity_force, self._grav_buf, self._total_controlled_dofs, self._device)

        # Null-space posture control: read current/desired joint state and gains.
        if self._use_null_space:
            for port, name, buf in (
                (inputs.joint_q, "inputs.joint_q", self._joint_q_buf),
                (inputs.joint_qd, "inputs.joint_qd", self._joint_qd_buf),
                (inputs.joint_q_des_null, "inputs.joint_q_des_null", self._joint_q_des_null_buf),
                (inputs.joint_qd_des_null, "inputs.joint_qd_des_null", self._joint_qd_des_null_buf),
            ):
                _validate_array(
                    array=port,
                    name=name,
                    dtype=wp.float32,
                    shape=(self._total_controlled_dofs,),
                    device=self._device,
                    allow_indexed=True,
                )
                _read_port(port, buf, self._total_controlled_dofs, self._device)

            if self._null_stiffness_baked is None:
                _validate_array(
                    array=inputs.null_space_stiffness,
                    name="inputs.null_space_stiffness",
                    dtype=wp.float32,
                    shape=(self._total_controlled_dofs,),
                    device=self._device,
                    allow_indexed=True,
                )
                _read_port(
                    inputs.null_space_stiffness, self._null_stiffness_buf, self._total_controlled_dofs, self._device
                )
            if self._null_damping_baked is None:
                _validate_array(
                    array=inputs.null_space_damping,
                    name="inputs.null_space_damping",
                    dtype=wp.float32,
                    shape=(self._total_controlled_dofs,),
                    device=self._device,
                    allow_indexed=True,
                )
                _read_port(inputs.null_space_damping, self._null_damping_buf, self._total_controlled_dofs, self._device)

        # Wrench control: read the desired wrench, and (feedback only) the measurement and gain.
        if self._use_wrench:
            _validate_array(
                array=inputs.desired_wrench_world,
                name="inputs.desired_wrench_world",
                dtype=wp.spatial_vector,
                shape=(robot_count,),
                device=self._device,
                allow_indexed=True,
            )
            _read_port(inputs.desired_wrench_world, self._desired_wrench_buf, robot_count, self._device)

            if self._use_wrench_feedback:
                _validate_array(
                    array=inputs.measured_wrench_world,
                    name="inputs.measured_wrench_world",
                    dtype=wp.spatial_vector,
                    shape=(robot_count,),
                    device=self._device,
                    allow_indexed=True,
                )
                _read_port(inputs.measured_wrench_world, self._measured_wrench_buf, robot_count, self._device)

                if self._wrench_stiffness_baked is None:
                    _validate_array(
                        array=inputs.wrench_stiffness,
                        name="inputs.wrench_stiffness",
                        dtype=wp.spatial_vector,
                        shape=(robot_count,),
                        device=self._device,
                        allow_indexed=True,
                    )
                    _read_port(inputs.wrench_stiffness, self._wrench_stiffness_buf, robot_count, self._device)

        stiffness = self._stiffness_baked if self._stiffness_baked is not None else self._stiffness_buf
        damping = self._damping_baked if self._damping_baked is not None else self._damping_buf

        wp.launch(
            _pose_error_kernel,
            dim=robot_count,
            inputs=[self._pose_buf, self._desired_pose_buf],
            outputs=[self._pose_error_buf],
            device=self._device,
        )
        # Kp/Kd are tool-local; rotate into world every step, since the
        # tool's orientation (and hence the rotation) changes every step.
        wp.launch(
            _rotate_selection_matrix_kernel,
            dim=robot_count,
            inputs=[self._pose_buf, stiffness],
            outputs=[self._motion_stiffness_matrix_world],
            device=self._device,
        )
        wp.launch(
            _rotate_selection_matrix_kernel,
            dim=robot_count,
            inputs=[self._pose_buf, damping],
            outputs=[self._motion_damping_matrix_world],
            device=self._device,
        )
        wp.launch(
            _task_space_pd_kernel,
            dim=robot_count,
            inputs=[
                self._pose_error_buf,
                self._twist_buf,
                self._desired_twist_buf,
                self._motion_stiffness_matrix_world,
                self._motion_damping_matrix_world,
            ],
            outputs=[self._desired_task_acceleration_buf],
            device=self._device,
        )

        force_source = self._desired_task_acceleration_buf
        if self._use_inertia:
            # Lambda = (J M^-1 J^T)^-1, then premultiply the PD term by it.
            wp.launch(
                _invert_spd_block_kernel,
                dim=robot_count,
                inputs=[self._mass_matrix_buf, self._controlled_dofs_per_robot, self._mass_matrix_cholesky],
                outputs=[self._mass_matrix_inv],
                device=self._device,
            )
            if self._use_partial_inertia:
                # Lambda as two independent 3x3 inversions (translation, rotation), ignoring their coupling.
                for axis_start, axis_end in ((0, 3), (3, 6)):
                    wp.launch(
                        _operational_space_mass_matrix_inverse_kernel,
                        dim=(robot_count, 3, 3),
                        inputs=[
                            self._jacobian_buf[:, axis_start:axis_end, :],
                            self._mass_matrix_inv,
                            self._controlled_dofs_per_robot,
                        ],
                        outputs=[self._operational_space_mass_matrix_inv[:, axis_start:axis_end, axis_start:axis_end]],
                        device=self._device,
                    )
                    wp.launch(
                        _invert_spd_block_kernel,
                        dim=robot_count,
                        inputs=[
                            self._operational_space_mass_matrix_inv[:, axis_start:axis_end, axis_start:axis_end],
                            self._partial_task_dim,
                            self._operational_space_mass_matrix_cholesky[:, axis_start:axis_end, axis_start:axis_end],
                        ],
                        outputs=[self._operational_space_mass_matrix[:, axis_start:axis_end, axis_start:axis_end]],
                        device=self._device,
                    )
            else:
                wp.launch(
                    _operational_space_mass_matrix_inverse_kernel,
                    dim=(robot_count, 6, 6),
                    inputs=[self._jacobian_buf, self._mass_matrix_inv, self._controlled_dofs_per_robot],
                    outputs=[self._operational_space_mass_matrix_inv],
                    device=self._device,
                )
                wp.launch(
                    _invert_spd_block_kernel,
                    dim=robot_count,
                    inputs=[
                        self._operational_space_mass_matrix_inv,
                        self._task_dim,
                        self._operational_space_mass_matrix_cholesky,
                    ],
                    outputs=[self._operational_space_mass_matrix],
                    device=self._device,
                )
            wp.launch(
                _apply_spatial_matrix_kernel,
                dim=robot_count,
                inputs=[self._operational_space_mass_matrix, self._desired_task_acceleration_buf],
                outputs=[self._task_space_force_buf],
                device=self._device,
            )
            force_source = self._task_space_force_buf

        if self._use_wrench:
            # Mask the motion force to the motion-controlled axes before mapping to joint torques.
            wp.launch(
                _rotate_selection_matrix_kernel,
                dim=robot_count,
                inputs=[self._pose_buf, self._motion_selection_axes_tool],
                outputs=[self._motion_selection_matrix_world],
                device=self._device,
            )
            wp.launch(
                _apply_spatial_matrix_kernel,
                dim=robot_count,
                inputs=[self._motion_selection_matrix_world, force_source],
                outputs=[self._masked_motion_force_buf],
                device=self._device,
            )
            force_source = self._masked_motion_force_buf

        wp.launch(
            _jacobian_transpose_force_kernel,
            dim=self._total_controlled_dofs,
            inputs=[self._jacobian_buf, force_source, self._robot_of_dof, self._slot_of_dof],
            outputs=[self._tau_buf],
            device=self._device,
        )

        if self._use_wrench:
            # Build the wrench command: feedforward + feedback, feedback only, or feedforward only.
            wrench_command_source = self._desired_wrench_buf
            if self._use_wrench_feedback:
                wrench_stiffness = (
                    self._wrench_stiffness_baked
                    if self._wrench_stiffness_baked is not None
                    else self._wrench_stiffness_buf
                )
                # wrench_stiffness is tool-local too; rotate into world every
                # step, the same reason motion Kp/Kd are rotated above.
                wp.launch(
                    _rotate_selection_matrix_kernel,
                    dim=robot_count,
                    inputs=[self._pose_buf, wrench_stiffness],
                    outputs=[self._wrench_stiffness_matrix_world],
                    device=self._device,
                )
                wrench_command_kernel = (
                    _wrench_feedforward_and_feedback_kernel
                    if self._use_wrench_feedforward
                    else _wrench_feedback_only_kernel
                )
                wp.launch(
                    wrench_command_kernel,
                    dim=robot_count,
                    inputs=[self._desired_wrench_buf, self._measured_wrench_buf, self._wrench_stiffness_matrix_world],
                    outputs=[self._wrench_command_buf],
                    device=self._device,
                )
                wrench_command_source = self._wrench_command_buf

            wp.launch(
                _rotate_selection_matrix_kernel,
                dim=robot_count,
                inputs=[self._pose_buf, self._wrench_selection_axes_tool],
                outputs=[self._wrench_selection_matrix_world],
                device=self._device,
            )
            wp.launch(
                _apply_spatial_matrix_kernel,
                dim=robot_count,
                inputs=[self._wrench_selection_matrix_world, wrench_command_source],
                outputs=[self._masked_wrench_force_buf],
                device=self._device,
            )
            wp.launch(
                _jacobian_transpose_force_kernel,
                dim=self._total_controlled_dofs,
                inputs=[self._jacobian_buf, self._masked_wrench_force_buf, self._robot_of_dof, self._slot_of_dof],
                outputs=[self._wrench_tau_buf],
                device=self._device,
            )
            wp.launch(
                _add_term_kernel,
                dim=self._total_controlled_dofs,
                inputs=[self._wrench_tau_buf],
                outputs=[self._tau_buf],
                device=self._device,
            )

        if self._use_null_space:
            null_stiffness = (
                self._null_stiffness_baked if self._null_stiffness_baked is not None else self._null_stiffness_buf
            )
            null_damping = self._null_damping_baked if self._null_damping_baked is not None else self._null_damping_buf
            wp.launch(
                _pd_term_kernel,
                dim=self._total_controlled_dofs,
                inputs=[
                    self._joint_q_buf,
                    self._joint_qd_buf,
                    self._joint_q_des_null_buf,
                    self._joint_qd_des_null_buf,
                    null_stiffness,
                    null_damping,
                ],
                outputs=[self._posture_acc_buf],
                device=self._device,
            )

            if self._use_inertia and not self._use_partial_inertia:
                # Dynamically-consistent pinv-transpose, Lambda @ J @ M^-1 (reuses this step's Lambda/M^-1).
                wp.launch(
                    _task_matrix_times_jacobian_kernel,
                    dim=(robot_count, 6, self._max_controlled_dofs),
                    inputs=[self._operational_space_mass_matrix, self._jacobian_buf, self._controlled_dofs_per_robot],
                    outputs=[self._null_space_jacobian_pinv_transpose_stage],
                    device=self._device,
                )
                wp.launch(
                    _apply_mass_matrix_inv_on_right_kernel,
                    dim=(robot_count, 6, self._max_controlled_dofs),
                    inputs=[
                        self._null_space_jacobian_pinv_transpose_stage,
                        self._mass_matrix_inv,
                        self._controlled_dofs_per_robot,
                    ],
                    outputs=[self._null_space_jacobian_pinv_transpose],
                    device=self._device,
                )
            else:
                # Moore-Penrose pinv-transpose, (J @ J^T)^-1 @ J: kinematics-only, no mass matrix needed.
                wp.launch(
                    _jacobian_times_jacobian_transpose_kernel,
                    dim=(robot_count, 6, 6),
                    inputs=[self._jacobian_buf, self._controlled_dofs_per_robot],
                    outputs=[self._null_space_jjt],
                    device=self._device,
                )
                wp.launch(
                    _invert_spd_block_kernel,
                    dim=robot_count,
                    inputs=[self._null_space_jjt, self._task_dim, self._null_space_jjt_cholesky],
                    outputs=[self._null_space_jjt_inv],
                    device=self._device,
                )
                wp.launch(
                    _task_matrix_times_jacobian_kernel,
                    dim=(robot_count, 6, self._max_controlled_dofs),
                    inputs=[self._null_space_jjt_inv, self._jacobian_buf, self._controlled_dofs_per_robot],
                    outputs=[self._null_space_jacobian_pinv_transpose],
                    device=self._device,
                )

            wp.launch(
                _null_space_projector_kernel,
                dim=(robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                inputs=[self._jacobian_buf, self._null_space_jacobian_pinv_transpose, self._controlled_dofs_per_robot],
                outputs=[self._null_space_projector],
                device=self._device,
            )

            # Premultiply the posture PD term by M (an acceleration -> torque conversion) before projecting.
            posture_source = self._posture_acc_buf
            if self._use_inertia:
                wp.launch(
                    _block_matrix_vector_multiply_kernel,
                    dim=self._total_controlled_dofs,
                    inputs=[
                        self._mass_matrix_buf,
                        self._posture_acc_buf,
                        self._robot_of_dof,
                        self._slot_of_dof,
                        self._dof_offsets,
                        self._controlled_dofs_per_robot,
                    ],
                    outputs=[self._posture_force_buf],
                    device=self._device,
                )
                posture_source = self._posture_force_buf
            # tau_null = N @ posture_source, projected so it doesn't disturb task-space motion.
            wp.launch(
                _block_matrix_vector_multiply_kernel,
                dim=self._total_controlled_dofs,
                inputs=[
                    self._null_space_projector,
                    posture_source,
                    self._robot_of_dof,
                    self._slot_of_dof,
                    self._dof_offsets,
                    self._controlled_dofs_per_robot,
                ],
                outputs=[self._null_space_tau_buf],
                device=self._device,
            )
            wp.launch(
                _add_term_kernel,
                dim=self._total_controlled_dofs,
                inputs=[self._null_space_tau_buf],
                outputs=[self._tau_buf],
                device=self._device,
            )

        if self._use_gravity:
            wp.launch(
                _add_term_kernel,
                dim=self._total_controlled_dofs,
                inputs=[self._grav_buf],
                outputs=[self._tau_buf],
                device=self._device,
            )

        # A view needs the scatter kernel (wp.copy isn't graph-capture-safe for a non-contiguous target).
        if isinstance(outputs.joint_f, wp.indexedarray):
            wp.launch(
                _scatter_port_kernel,
                dim=self._total_controlled_dofs,
                inputs=[self._tau_buf],
                outputs=[outputs.joint_f],
                device=self._device,
            )
        else:
            wp.copy(outputs.joint_f, self._tau_buf)
