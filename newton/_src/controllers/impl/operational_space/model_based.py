# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerOperationalSpace — operational-space impedance control with
Newton model-internal kinematics and dynamics.

Calls :func:`newton.eval_fk`, :func:`newton.eval_jacobian`, and (when
inertial decoupling is enabled) :func:`newton.eval_mass_matrix` on the
supplied model each step, resolves each robot's tool-point pose, twist, and
Jacobian from a Newton *site*, then delegates the control law to an inner
:class:`ControllerOperationalSpaceModelFree` instance.
"""

from __future__ import annotations

import re

import numpy as np
import warp as wp

from newton import JointType
from newton._src.geometry.flags import ShapeFlags
from newton._src.sim.articulation import eval_fk, eval_jacobian, eval_mass_matrix
from newton._src.sim.inverse_dynamics import eval_inverse_dynamics_passive
from newton._src.sim.model import Model

from ....utils.selection import get_name_from_label, match_labels
from ...controller import ControllerBase
from ...joint_selection import select_joints
from ...utils import _validate_array
from .._common import _gather_mass_matrix_blocks_kernel, _read_port
from ._common import _shift_jacobian_to_tool_kernel, _tool_pose_and_twist_kernel
from .model_free import _IDENTITY_QUAT, _IDENTITY_TRANSFORM, ControllerOperationalSpaceModelFree


class ControllerOperationalSpace(ControllerBase):
    """Task-space (operational-space) impedance controller with internally computed dynamics.

    Implements the operational-space control law. This model-based variant
    computes the tool pose/twist, tool-point Jacobian, and (when enabled) the
    mass matrix and gravity term itself: it evaluates forward kinematics and
    the enabled dynamics terms from ``model`` on every :meth:`step`, so the
    caller supplies only joint positions and velocities plus task-space
    targets.

    ``model`` is borrowed, not owned — it is never written to, and changes to
    it are visible to the controller immediately.

    **Joint selection.** ``articulations`` and ``joints`` select which DOFs
    become the tool Jacobian's columns, following :ref:`label-matching`: each
    is a list of model indices and/or label patterns (or a single pattern),
    matched against :attr:`~newton.Model.articulation_label` and the leaf
    component of :attr:`~newton.Model.joint_label` respectively. Only joints
    spanning a single coordinate and a single DOF can be controlled.

    **Tool selection.** ``tool_sites`` selects one Newton *site* per robot
    that ends up with controlled joints — the point on the robot whose
    pose/twist is controlled. This need not be the same frame commands are
    specified in — see ``operational_frame_pose_world``. It follows the same
    ``list[index/pattern] | index | pattern`` shape as ``joints``, matched
    against the leaf component of each site's label. Every controlled robot
    must match exactly one site.

    Each articulation in ``model`` is one robot. Supports heterogeneous robot
    fleets — robots may have different controlled-DOF counts, and a robot may
    be left uncontrolled entirely by omitting it from ``articulations``.

    See also :class:`ControllerOperationalSpaceModelFree`, which takes the
    tool pose/twist, Jacobian, mass matrix, and gravity term as inputs
    instead of computing them from a :class:`~newton.Model`.

    Args:
        model: :class:`~newton.Model` whose articulations are the robots.
        articulations: Articulation indices or label patterns to control, as
            a list or as a single pattern. ``None`` selects every
            articulation in ``model``.
        joints: Model joint indices or label patterns whose DOFs become the
            tool Jacobian's columns, within the selected articulations, as a
            list or as a single pattern. ``None`` selects every joint
            spanning exactly one coordinate and one DOF in each selected
            articulation; any other joint is left uncontrolled instead of
            rejected. A joint named explicitly is not filtered this way and
            still raises ``ValueError`` if it is not 1-coordinate/1-DOF.
        tool_sites: Site indices or label patterns selecting each controlled
            robot's controlled point, as a list or as a single pattern.
            Required — there is no default tool site. Raises if a
            controlled robot matches zero or more than one site.
        motion_stiffness: Task-space position/orientation-error gain Kp,
            per-axis in the operational frame (e.g. "stiff along the
            insertion axis" stays meaningful as that frame reorients). Units
            depend on ``use_inertia_decoupling``: [1/s²] when enabled, since
            the spring-damper term is then a task-space acceleration
            premultiplied by Lambda; otherwise [N/m] on the position axes
            and [N·m/rad] on the orientation axes. Pass a scalar to apply
            the same gain to every axis of every robot, a
            ``wp.spatial_vector`` to apply the same 6 per-axis gains to
            every robot, an array of shape [controlled_robot_count] to set
            them individually (one ``wp.spatial_vector`` of 6 gains per
            robot), or ``None`` to read ``inputs.motion_stiffness`` each
            step.
        motion_damping: Task-space velocity-error gain Kd,
            operational-frame-local like ``motion_stiffness``, [1/s] when
            ``use_inertia_decoupling`` is enabled, otherwise [N·s/m] on the
            position axes and [N·m·s/rad] on the orientation axes. Same
            format as ``motion_stiffness``.
        operational_frame_pose_world: World pose of the operational frame —
            the frame ``inputs.desired_tool_pose_operational``/
            ``inputs.desired_twist_operational`` are expressed relative to,
            and that ``motion_stiffness``/``motion_damping``/
            ``wrench_stiffness`` are interpreted in. Pass a ``wp.transform`` to apply the same fixed pose
            to every robot, an array of shape [controlled_robot_count] to
            set them individually (fixed for the controller's lifetime), or
            ``None`` to read ``inputs.operational_frame_pose_world`` each
            step for a time-varying frame. Defaults to identity (coincides
            with world frame).
        use_inertia_decoupling: Premultiply the task-space spring-damper
            term by Lambda, the operational-space mass matrix, computed
            each step from ``model``'s own mass matrix (via
            :func:`newton.eval_mass_matrix`) and the resolved tool Jacobian.
            Note that if the ``joints``/``articulations`` selection omits a
            DOF that is both free and dynamically coupled to the controlled
            set, then there is not sufficient information to fully
            dynamically decouple the system. No error is raised, as
            omitting certain joints is often a useful approximation. It is
            the user's responsibility to provide the needed information.
        use_partial_inertia_decoupling: Compute Lambda ignoring the coupling
            between translational and rotational inertia. Only meaningful
            when ``use_inertia_decoupling=True``.
        use_gravity_compensation: Add the model's own gravity generalized
            forces — computed each step via
            :func:`~newton.eval_inverse_dynamics_passive`
            on ``model`` — directly to the summed joint torque.
        use_wrench_feedforward: Command the desired wrench directly, as a
            feedforward term in the wrench law, combined with motion
            control through the generalized selection matrix Omega from
            Khatib, O. (1987), "A unified approach for motion and force
            control of robot manipulators: The operational space
            formulation," IEEE Journal of Robotics and Automation, 3(1),
            43-53 (see ``linear_selection_frame_operational``/
            ``angular_selection_frame_operational`` below). When both this
            and ``use_wrench_feedback`` are ``False``, every axis is
            motion-controlled and ``motion_selection_axes``/
            ``wrench_selection_axes``/``wrench_stiffness`` must be left
            unset, and ``linear_selection_frame_operational``/
            ``angular_selection_frame_operational`` must be left at their
            identity default.
        use_wrench_feedback: Correct the wrench command by
            ``Kp · (desired - measured)`` using
            ``inputs.measured_wrench_world`` each step, as a feedback term
            in the wrench law. May be enabled with or without
            ``use_wrench_feedforward``: without it, the command is the
            feedback correction alone, regulating the measured wrench
            toward the desired setpoint with no separate feedforward term.
        motion_selection_axes: Diagonal selection weight per task axis
            (0/1, or any scalar weight): (linear x, y, z, angular x, y, z),
            the linear half interpreted in
            ``linear_selection_frame_operational`` (S_f) and the angular
            half in ``angular_selection_frame_operational`` (S_tau). Pass a
            ``wp.spatial_vector`` to apply the same weights to every robot,
            or an array of shape [controlled_robot_count] to set them
            individually. Only meaningful when wrench control is enabled;
            defaults to every axis motion-controlled,
            ``wp.spatial_vector(1, 1, 1, 1, 1, 1)``. Usually the complement
            of ``wrench_selection_axes`` — each axis under motion control,
            not force control, and vice versa — but that is not enforced:
            nothing here requires the two to partition the 6 axes.
        wrench_selection_axes: Diagonal selection weight per task axis, same
            format and selection frames as ``motion_selection_axes``,
            applied to the wrench term. Required when wrench control is
            enabled. Usually the complement of ``motion_selection_axes``,
            but that is not enforced — see its docstring above.
        wrench_stiffness: Contact-wrench proportional feedback gain Kp,
            operational-frame-local like ``motion_stiffness``, dimensionless
            (multiplies a wrench error directly, not a pose error) on both
            the force and moment axes. Same format as ``motion_stiffness``.
            Only meaningful when ``use_wrench_feedback=True``.
        linear_selection_frame_operational: Orientation of S_f, the frame
            ``motion_selection_axes``/``wrench_selection_axes``'s linear
            (force) half is interpreted in, relative to the operational
            frame — e.g. aligned to a contact surface's normal. Independent
            of ``angular_selection_frame_operational`` (S_tau); the two
            need not agree. Pass a ``wp.quat`` to apply the same fixed
            orientation to every robot, an array of shape
            [controlled_robot_count] to set them individually, or ``None``
            to read ``inputs.linear_selection_frame_operational`` each step
            for a time-varying frame. Defaults to identity.
        angular_selection_frame_operational: Orientation of S_tau, the
            frame ``motion_selection_axes``/``wrench_selection_axes``'s
            angular (moment) half is interpreted in, relative to the
            operational frame — e.g. a compliant rotation axis. Same format
            as ``linear_selection_frame_operational``.
        use_null_space_control: Pursue a secondary joint-space posture task
            in the null space of the primary task, so it does not disturb
            task-space motion. Requires every robot to have more than 6
            controlled DOFs (redundant relative to the 6D task). The
            null-space projector is dynamically consistent (accounts for
            the robot's own inertia) when ``use_inertia_decoupling=True``
            and ``use_partial_inertia_decoupling=False``, or a
            kinematics-only (Moore-Penrose) projector otherwise.
        null_space_stiffness: Joint-space posture position-error gain Kp.
            Units depend on ``use_inertia_decoupling``: [1/s²] when enabled,
            since the posture PD term is then premultiplied by the mass
            matrix; otherwise [N/m or N·m/rad]. Pass a scalar to apply the
            same gain to every controlled DOF, an array of shape
            [total_controlled_dofs] to set them individually, or ``None``
            to read ``inputs.null_space_stiffness`` each step. Only
            meaningful when ``use_null_space_control=True``.
        null_space_damping: Joint-space posture velocity-error gain Kd,
            [1/s] when ``use_inertia_decoupling`` is enabled, otherwise
            [N·s/m or N·m·s/rad]. Same format as ``null_space_stiffness``.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerOperationalSpace.input`.

        ``joint_q``/``joint_qd`` cover the whole model, since forward
        kinematics depends on uncontrolled joints too; every other field is
        per-robot, shape [controlled_robot_count]. Optional fields are
        ``None`` when the corresponding feature is disabled at construction.
        """

        joint_q: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Current joint positions [m or rad], shape [model.joint_coord_count]."""
        joint_qd: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Current joint velocities [m/s or rad/s], shape [model.joint_dof_count]."""
        operational_frame_pose_world: wp.array[wp.transform] | wp.indexedarray[wp.transform] | None
        """World pose of the operational frame, shape [controlled_robot_count]. ``None`` when fixed at construction."""
        desired_tool_pose_operational: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Desired tool pose, relative to the operational frame, shape [controlled_robot_count]."""
        desired_twist_operational: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Desired tool twist (linear, angular), components expressed in the operational frame [m/s, rad/s], shape [controlled_robot_count]."""
        motion_stiffness: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Task-space position/orientation-error gain Kp, operational-frame-local, shape [controlled_robot_count]. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m] / [N·m/rad]. ``None`` when gains are baked at construction."""
        motion_damping: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Task-space velocity-error gain Kd, operational-frame-local, shape [controlled_robot_count]. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m] / [N·m·s/rad]. ``None`` when gains are baked at construction."""
        desired_wrench_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Desired contact wrench (force, moment) in world coordinates [N, N·m], shape [controlled_robot_count]. ``None`` unless wrench control is enabled."""
        measured_wrench_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Measured contact wrench (force, moment) in world coordinates [N, N·m], shape [controlled_robot_count]. ``None`` unless ``use_wrench_feedback=True``."""
        wrench_stiffness: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Contact-wrench proportional feedback gain Kp, operational-frame-local, shape [controlled_robot_count]. Dimensionless -- multiplies a wrench error directly, not a pose error. ``None`` when gains are baked at construction, or when ``use_wrench_feedback=False``."""
        linear_selection_frame_operational: wp.array[wp.quat] | wp.indexedarray[wp.quat] | None
        """Orientation of S_f, relative to the operational frame, shape [controlled_robot_count]. ``None`` when fixed at construction, or when wrench control is disabled."""
        angular_selection_frame_operational: wp.array[wp.quat] | wp.indexedarray[wp.quat] | None
        """Orientation of S_tau, relative to the operational frame, shape [controlled_robot_count]. ``None`` when fixed at construction, or when wrench control is disabled."""
        joint_q_des_null: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Desired joint positions for the null-space posture task [m or rad], compact, shape [total_controlled_dofs]. ``None`` unless ``use_null_space_control=True``."""
        joint_qd_des_null: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Desired joint velocities for the null-space posture task [m/s or rad/s], compact, shape [total_controlled_dofs]. ``None`` unless ``use_null_space_control=True``."""
        null_space_stiffness: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Joint-space posture position-error gain Kp, compact, shape [total_controlled_dofs]. ``None`` when baked at construction, or when ``use_null_space_control=False``."""
        null_space_damping: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Joint-space posture velocity-error gain Kd, compact, shape [total_controlled_dofs]. ``None`` when baked at construction, or when ``use_null_space_control=False``."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerOperationalSpace.output`."""

        joint_f: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Joint torque command [N or N·m], compact, shape [total_controlled_dofs]."""

    def __init__(
        self,
        model: Model,
        *,
        articulations: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
        joints: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
        tool_sites: list[int | str | re.Pattern[str]] | str | re.Pattern[str],
        motion_stiffness: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None,
        motion_damping: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None,
        operational_frame_pose_world: wp.array[wp.transform] | wp.transform | None = _IDENTITY_TRANSFORM,
        use_inertia_decoupling: bool = True,
        use_partial_inertia_decoupling: bool = False,
        use_gravity_compensation: bool = True,
        use_wrench_feedforward: bool = False,
        use_wrench_feedback: bool = False,
        motion_selection_axes: wp.array[wp.spatial_vector] | wp.spatial_vector | None = None,
        wrench_selection_axes: wp.array[wp.spatial_vector] | wp.spatial_vector | None = None,
        wrench_stiffness: wp.array[wp.spatial_vector] | wp.spatial_vector | float | None = None,
        linear_selection_frame_operational: wp.array[wp.quat] | wp.quat | None = _IDENTITY_QUAT,
        angular_selection_frame_operational: wp.array[wp.quat] | wp.quat | None = _IDENTITY_QUAT,
        use_null_space_control: bool = False,
        null_space_stiffness: wp.array[wp.float32] | float | None = None,
        null_space_damping: wp.array[wp.float32] | float | None = None,
    ):
        if not isinstance(model, Model):
            raise TypeError(f"model must be a newton.Model, got {type(model).__name__}.")
        model_robot_count = model.articulation_count
        if model_robot_count < 1:
            raise ValueError("model has no articulations.")

        self._device = model.device
        self._requires_grad = model.requires_grad
        self._use_inertia = bool(use_inertia_decoupling)
        self._use_gravity = bool(use_gravity_compensation)
        self._use_wrench_feedforward = bool(use_wrench_feedforward)
        self._use_wrench_feedback = bool(use_wrench_feedback)
        self._use_wrench = self._use_wrench_feedforward or self._use_wrench_feedback
        self._use_null_space = bool(use_null_space_control)

        # Whether each gain is read live from inputs each step (None baked
        # value) or fixed at construction, exactly as the caller passed it --
        # tracked here rather than read off the inner ModelFree controller,
        # since that decision belongs to this constructor's own arguments.
        self._operational_frame_is_live = operational_frame_pose_world is None
        self._motion_stiffness_is_live = motion_stiffness is None
        self._motion_damping_is_live = motion_damping is None
        self._wrench_stiffness_is_live = self._use_wrench_feedback and wrench_stiffness is None
        self._linear_selection_frame_is_live = self._use_wrench and linear_selection_frame_operational is None
        self._angular_selection_frame_is_live = self._use_wrench and angular_selection_frame_operational is None
        self._null_stiffness_is_live = self._use_null_space and null_space_stiffness is None
        self._null_damping_is_live = self._use_null_space and null_space_damping is None

        self._model = model
        self._model_state = model.state(requires_grad=self._requires_grad)
        self._coord_count = int(model.joint_coord_count)
        self._dof_count = int(model.joint_dof_count)

        joint_selection = select_joints(model, articulations=articulations, joints=joints)
        joint_q_idx = joint_selection.q_start
        joint_qd_idx = joint_selection.qd_start

        # ------------------------------------------------------------------
        # Validate the two model-space index arrays select_joints returns:
        #   1. type/dtype/shape of q_start, then qd_start against its length
        #   2. non-empty
        #   3. both index within the model's coordinate/DOF space
        #   4. qd_start has no duplicate DOF (a duplicate coordinate in
        #      q_start alone cannot occur without one, given every joint
        #      here is 1-coordinate/1-DOF and steps 5-6 below)
        #   5. q_start[i]/qd_start[i] name the same joint for every i
        #   6. every joint spans exactly one coordinate and one DOF
        #   7. every joint belongs to a robot (articulation)
        #   8. joints are grouped by robot, ascending
        # ------------------------------------------------------------------
        if not isinstance(joint_q_idx, wp.array):
            raise TypeError(f"joint_selection.q_start must be a wp.array, got {type(joint_q_idx).__name__}.")
        _validate_array(
            array=joint_q_idx,
            name="joint_selection.q_start",
            dtype=wp.int32,
            shape=(joint_q_idx.size,),
            device=self._device,
        )
        total_controlled_dofs = int(joint_q_idx.size)
        if total_controlled_dofs < 1:
            raise ValueError("joint_selection.q_start is empty; there is nothing to control.")
        _validate_array(
            array=joint_qd_idx,
            name="joint_selection.qd_start",
            dtype=wp.int32,
            shape=(total_controlled_dofs,),
            device=self._device,
        )

        q_idx_np = joint_q_idx.numpy()
        qd_idx_np = joint_qd_idx.numpy()
        for name, idx_np, limit, space in (
            ("joint_selection.q_start", q_idx_np, self._coord_count, "coordinate"),
            ("joint_selection.qd_start", qd_idx_np, self._dof_count, "DOF"),
        ):
            if idx_np.min() < 0 or idx_np.max() >= limit:
                raise ValueError(
                    f"{name} must index the model's {space} space [0, {limit}), got "
                    f"range [{int(idx_np.min())}, {int(idx_np.max())}]."
                )

        if np.unique(qd_idx_np).size != qd_idx_np.size:
            duplicate = int(np.bincount(qd_idx_np).argmax())
            raise ValueError(
                f"joint_selection.qd_start contains DOF {duplicate} more than once; two controlled slots "
                f"cannot map to the same simulation DOF."
            )

        owning_joint = np.searchsorted(model.joint_q_start.numpy(), q_idx_np, side="right") - 1
        owning_joint_qd = np.searchsorted(model.joint_qd_start.numpy(), qd_idx_np, side="right") - 1
        if not np.array_equal(owning_joint, owning_joint_qd):
            mismatched = int(np.flatnonzero(owning_joint != owning_joint_qd)[0])
            raise ValueError(
                f"joint_selection.q_start and joint_selection.qd_start disagree at entry {mismatched}: "
                f"coordinate {int(q_idx_np[mismatched])} belongs to joint {int(owning_joint[mismatched])} "
                f"but DOF {int(qd_idx_np[mismatched])} belongs to joint {int(owning_joint_qd[mismatched])}. "
                f"Did you swap the two arrays?"
            )

        # A joint is controllable when its Jacobian column is a single scalar
        # DOF, i.e. it spans exactly one coordinate and one DOF. Derived from
        # the model's own spans.
        joint_type_np = model.joint_type.numpy()
        coord_span = np.diff(model.joint_q_start.numpy())[owning_joint]
        dof_span = np.diff(model.joint_qd_start.numpy())[owning_joint]
        unsupported = sorted(
            {
                (int(j), JointType(joint_type_np[j]).name)
                for j, coords, dofs in zip(owning_joint, coord_span, dof_span, strict=True)
                if coords != 1 or dofs != 1
            }
        )
        if unsupported:
            raise ValueError(
                f"ControllerOperationalSpace only supports controlling joints that span a single "
                f"coordinate and a single DOF; joint_selection addresses unsupported joints: {unsupported}"
            )

        owning_robot = model.joint_articulation.numpy()[owning_joint]
        loose = np.flatnonzero(owning_robot < 0)
        if loose.size:
            raise ValueError(
                f"joint_selection addresses joint {int(owning_joint[loose[0]])}, which belongs to no "
                f"robot. The controller runs forward kinematics and dynamics per robot, so such a "
                f"joint has no Jacobian, mass matrix, or gravity term."
            )
        if np.any(np.diff(owning_robot) < 0):
            raise ValueError(
                "joint_selection.q_start/qd_start must be grouped by robot (robot 0's DOFs first, "
                f"then robot 1's, ...); got robot order {owning_robot.tolist()}."
            )

        # Packed-robot bookkeeping, derived from owning_robot's distinct
        # values (model_robot_index_np) and their counts
        # (controlled_dofs_per_robot_np):
        #   - controlled_dofs_per_robot: the inner ModelFree's whole layout
        #   - controlled_robot_count: packed slot count (vs. model_robot_count,
        #     every articulation whether controlled or not)
        #   - max_controlled_dofs: sizes every padded per-robot buffer
        #   - self._model_robot_index: packed slot -> real model articulation
        #     index, needed since eval_jacobian/eval_mass_matrix output is
        #     addressed by model articulation number, not packed slot
        #   - self._controlled_robot_mask: model_robot_count-length mask so
        #     eval_fk/eval_jacobian/eval_mass_matrix skip uncontrolled robots
        model_robot_index_np, controlled_dofs_per_robot_np = np.unique(owning_robot, return_counts=True)
        model_robot_index_np = model_robot_index_np.astype(np.int32)
        controlled_dofs_per_robot_np = controlled_dofs_per_robot_np.astype(np.int32)
        controlled_robot_count = int(model_robot_index_np.size)
        controlled_dofs_per_robot = wp.array(controlled_dofs_per_robot_np, dtype=wp.int32, device=self._device)
        max_controlled_dofs = int(controlled_dofs_per_robot_np.max())
        self._model_robot_index = wp.array(model_robot_index_np, dtype=wp.int32, device=self._device)
        mask_np = np.zeros(model_robot_count, dtype=bool)
        mask_np[model_robot_index_np] = True
        self._controlled_robot_mask = wp.array(mask_np, dtype=wp.bool, device=self._device)
        # ------------------------------------------------------------------

        self._model_robot_count = model_robot_count
        self._controlled_robot_count = controlled_robot_count
        self._max_controlled_dofs = max_controlled_dofs
        self._total_controlled_dofs = total_controlled_dofs
        self._controlled_dofs_per_robot = controlled_dofs_per_robot
        self._q_idx = wp.clone(joint_q_idx)
        self._qd_idx = wp.clone(joint_qd_idx)

        # ------------------------------------------------------------------
        # Tool-site selection: one site per robot in model_robot_index_np --
        # the exact, already-ordered set joint selection resolved above, so
        # there is no second, independent articulation resolution to keep in
        # sync with the first.
        # ------------------------------------------------------------------
        joint_child_np = model.joint_child.numpy()
        joint_articulation_np = model.joint_articulation.numpy()
        body_to_articulation_np = np.full(model.body_count, -1, dtype=np.int32)
        body_to_articulation_np[joint_child_np] = joint_articulation_np

        shape_flags_np = model.shape_flags.numpy()
        shape_body_np = model.shape_body.numpy()
        shape_transform_np = model.shape_transform.numpy()
        site_indices_np = np.flatnonzero((shape_flags_np & ShapeFlags.SITE) != 0)
        if site_indices_np.size == 0:
            raise ValueError("model contains no sites; add one with ModelBuilder.add_site for the tool frame.")
        site_articulation_np = body_to_articulation_np[shape_body_np[site_indices_np]]
        site_names = [get_name_from_label(model.shape_label[s]) for s in site_indices_np]

        tool_site_entries = [tool_sites] if isinstance(tool_sites, (int, str, re.Pattern)) else tool_sites
        matched_sites: list[int] = []
        for entry in tool_site_entries:
            if isinstance(entry, int):
                if entry not in site_indices_np:
                    raise ValueError(f"tool_sites site index {entry} is not a site in the model.")
                matched_sites.append(entry)
            else:
                local_matches = match_labels(site_names, entry)
                if not local_matches:
                    raise ValueError(f"tool_sites pattern {entry!r} matches no site in the model.")
                matched_sites.extend(int(site_indices_np[m]) for m in local_matches)
        matched_sites_set = sorted(set(matched_sites))

        site_index_to_articulation = dict(zip(site_indices_np.tolist(), site_articulation_np.tolist(), strict=True))
        tool_body_np = np.zeros(controlled_robot_count, dtype=np.int32)
        tool_transform_body: list[wp.transform] = []
        for robot_slot, art in enumerate(model_robot_index_np.tolist()):
            sites_on_robot = [s for s in matched_sites_set if site_index_to_articulation[s] == art]
            if len(sites_on_robot) == 0:
                raise ValueError(f"tool_sites matches no site on articulation {art}.")
            if len(sites_on_robot) > 1:
                raise ValueError(
                    f"tool_sites matches {len(sites_on_robot)} sites on articulation {art}; exactly one "
                    f"tool site is required per robot."
                )
            site = sites_on_robot[0]
            tool_body_np[robot_slot] = shape_body_np[site]
            tool_transform_body.append(wp.transform(*shape_transform_np[site]))

        self._tool_body = wp.array(tool_body_np, dtype=wp.int32, device=self._device)
        self._tool_transform_body = wp.array(tool_transform_body, dtype=wp.transform, device=self._device)

        # robot_link_idx: the tool site's row-block index within its
        # articulation's eval_jacobian output. eval_articulation_jacobian
        # writes link i's rows at [i*6 : i*6+6], where i is the position,
        # within its articulation's own joint range, of the joint that moves
        # the tool site's body -- so this is (that joint's index) minus
        # (the articulation's first joint index).
        body_to_joint_np = np.full(model.body_count, -1, dtype=np.int32)
        body_to_joint_np[joint_child_np] = np.arange(joint_child_np.size, dtype=np.int32)
        tool_site_joint_np = body_to_joint_np[tool_body_np]
        articulation_start_np = model.articulation_start.numpy()
        robot_link_idx_np = (tool_site_joint_np - articulation_start_np[model_robot_index_np]).astype(np.int32)
        self._robot_link_idx = wp.array(robot_link_idx_np, dtype=wp.int32, device=self._device)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Dynamics buffers. Allocated up front; populated by step().
        # ------------------------------------------------------------------
        # articulation_dof_idx_of_padded_dof_idx (packed robot slot, padded
        # column -> DOF index within that robot's own articulation) is needed
        # unconditionally: the Jacobian shift below uses it too, not just the
        # mass-matrix gather.
        self._articulation_dof_idx_of_padded_dof_idx = wp.array(
            self._compute_articulation_dof_idx_of_padded_dof_idx(
                qd_idx_np=qd_idx_np,
                model_robot_index_np=model_robot_index_np,
                controlled_dofs_per_robot_np=controlled_dofs_per_robot_np,
            ),
            dtype=wp.int32,
            device=self._device,
        )

        self._model_mass_matrix: wp.array3d[wp.float32] | None = None
        self._controlled_mass_matrix: wp.array3d[wp.float32] | None = None
        if self._use_inertia:
            model_max_dofs = model.max_dofs_per_articulation
            self._model_mass_matrix = wp.zeros(
                (model_robot_count, model_max_dofs, model_max_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=self._requires_grad,
            )
            self._controlled_mass_matrix = wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=self._requires_grad,
            )

        self._gravity_flat: wp.array[wp.float32] | None = None
        if self._use_gravity:
            self._gravity_flat = wp.zeros(
                self._dof_count, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
            )

        model_max_links = model.max_joints_per_articulation
        model_max_dofs_for_jacobian = model.max_dofs_per_articulation
        self._jacobian_com_world = wp.zeros(
            (model_robot_count, model_max_links * 6, model_max_dofs_for_jacobian),
            dtype=wp.float32,
            device=self._device,
            requires_grad=self._requires_grad,
        )
        self._jacobian_tool_world = wp.zeros(
            (controlled_robot_count, 6, max_controlled_dofs),
            dtype=wp.float32,
            device=self._device,
            requires_grad=self._requires_grad,
        )
        self._tool_pose_world = wp.zeros(
            controlled_robot_count, dtype=wp.transform, device=self._device, requires_grad=self._requires_grad
        )
        self._tool_twist_world = wp.zeros(
            controlled_robot_count, dtype=wp.spatial_vector, device=self._device, requires_grad=self._requires_grad
        )
        # ------------------------------------------------------------------

        self._model_free = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=controlled_dofs_per_robot,
            motion_stiffness=motion_stiffness,
            motion_damping=motion_damping,
            operational_frame_pose_world=operational_frame_pose_world,
            use_inertia_decoupling=use_inertia_decoupling,
            use_partial_inertia_decoupling=use_partial_inertia_decoupling,
            use_gravity_compensation=use_gravity_compensation,
            use_wrench_feedforward=use_wrench_feedforward,
            use_wrench_feedback=use_wrench_feedback,
            motion_selection_axes=motion_selection_axes,
            wrench_selection_axes=wrench_selection_axes,
            wrench_stiffness=wrench_stiffness,
            linear_selection_frame_operational=linear_selection_frame_operational,
            angular_selection_frame_operational=angular_selection_frame_operational,
            use_null_space_control=use_null_space_control,
            null_space_stiffness=null_space_stiffness,
            null_space_damping=null_space_damping,
            device=self._device,
            requires_grad=self._requires_grad,
        )

        # Pre-wired fields forwarded to the inner controller each step: live
        # indexed views of the whole-model/tool buffers above, so the inner
        # controller reads current contents with no index table of its own.
        self._mf_input = ControllerOperationalSpaceModelFree.Inputs()
        self._mf_input.tool_pose_world = self._tool_pose_world
        self._mf_input.tool_twist_world = self._tool_twist_world
        self._mf_input.jacobian_tool_world = self._jacobian_tool_world
        if self._use_inertia:
            self._mf_input.mass_matrix = self._controlled_mass_matrix
        if self._use_gravity:
            self._mf_input.gravity_force = self._gravity_flat[self._qd_idx]
        if self._use_null_space:
            self._mf_input.joint_q = self._model_state.joint_q[self._q_idx]
            self._mf_input.joint_qd = self._model_state.joint_qd[self._qd_idx]

    def _compute_articulation_dof_idx_of_padded_dof_idx(
        self, *, qd_idx_np: np.ndarray, model_robot_index_np: np.ndarray, controlled_dofs_per_robot_np: np.ndarray
    ) -> np.ndarray:
        """Return, for each (controlled robot, padded slot), the DOF's index within that robot.

        ``joint_selection.qd_start`` is in the model's DOF numbering, but
        :func:`~newton.eval_mass_matrix` indexes each robot's block by
        DOF-within-that-robot, so the two differ by where the robot's DOFs
        start in the model.
        """
        robot_joint_start = self._model.articulation_start.numpy()
        robot_dof_start = self._model.joint_qd_start.numpy()[robot_joint_start[model_robot_index_np]]

        controlled_robot_count = int(model_robot_index_np.size)
        offsets = np.zeros(controlled_robot_count, dtype=np.int64)
        offsets[1:] = np.cumsum(controlled_dofs_per_robot_np[:-1])

        articulation_dof_idx_of_padded_dof_idx = np.zeros(
            (controlled_robot_count, self._max_controlled_dofs), dtype=np.int32
        )
        for robot in range(controlled_robot_count):
            dof_count = int(controlled_dofs_per_robot_np[robot])
            chunk = qd_idx_np[offsets[robot] : offsets[robot] + dof_count]
            articulation_dof_idx_of_padded_dof_idx[robot, :dof_count] = chunk - robot_dof_start[robot]
        return articulation_dof_idx_of_padded_dof_idx

    @property
    def model_robot_count(self) -> int:
        """Number of articulations in ``model``, controlled or not."""
        return self._model_robot_count

    @property
    def controlled_robot_count(self) -> int:
        """Number of robots with at least one controlled DOF."""
        return self._controlled_robot_count

    @property
    def max_controlled_dofs(self) -> int:
        """Largest controlled-DOF count over the controlled robots."""
        return self._max_controlled_dofs

    @property
    def total_controlled_dofs(self) -> int:
        """Total controlled-DOF count across all robots, the length of every compact port."""
        return self._total_controlled_dofs

    @property
    def q_start(self) -> wp.array[wp.int32]:
        """Model coordinate index of each controlled joint, shape [total_controlled_dofs]."""
        return self._q_idx

    @property
    def qd_start(self) -> wp.array[wp.int32]:
        """Model DOF index of each controlled joint, shape [total_controlled_dofs]."""
        return self._qd_idx

    @property
    def tool_body(self) -> wp.array[wp.int32]:
        """Body index of each controlled robot's tool site, shape [controlled_robot_count]."""
        return self._tool_body

    @property
    def tool_transform_body(self) -> wp.array[wp.transform]:
        """Tool site's transform relative to its body [m, unitless quaternion], shape [controlled_robot_count]."""
        return self._tool_transform_body

    @property
    def tool_pose_world(self) -> wp.array[wp.transform]:
        """World pose of each controlled robot's tool site as of the latest ``step()`` [m, unitless quaternion], shape [controlled_robot_count]."""
        return self._tool_pose_world

    @property
    def tool_twist_world(self) -> wp.array[wp.spatial_vector]:
        """World twist (linear, angular) of each controlled robot's tool site as of the latest ``step()`` [m/s, rad/s], shape [controlled_robot_count]."""
        return self._tool_twist_world

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

        inputs = ControllerOperationalSpace.Inputs()
        inputs.joint_q = wp.zeros(self._coord_count, dtype=wp.float32, device=device, requires_grad=requires_grad)
        inputs.joint_qd = wp.zeros(self._dof_count, dtype=wp.float32, device=device, requires_grad=requires_grad)
        inputs.operational_frame_pose_world = (
            wp.zeros(robot_count, dtype=wp.transform, device=device, requires_grad=requires_grad)
            if self._operational_frame_is_live
            else None
        )
        inputs.desired_tool_pose_operational = wp.zeros(
            robot_count, dtype=wp.transform, device=device, requires_grad=requires_grad
        )
        inputs.desired_twist_operational = wp.zeros(
            robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad
        )
        inputs.motion_stiffness = (
            wp.zeros(robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad)
            if self._motion_stiffness_is_live
            else None
        )
        inputs.motion_damping = (
            wp.zeros(robot_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad)
            if self._motion_damping_is_live
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
            if self._wrench_stiffness_is_live
            else None
        )
        inputs.linear_selection_frame_operational = (
            wp.zeros(robot_count, dtype=wp.quat, device=device, requires_grad=requires_grad)
            if self._linear_selection_frame_is_live
            else None
        )
        inputs.angular_selection_frame_operational = (
            wp.zeros(robot_count, dtype=wp.quat, device=device, requires_grad=requires_grad)
            if self._angular_selection_frame_is_live
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
            if self._null_stiffness_is_live
            else None
        )
        inputs.null_space_damping = (
            wp.zeros(self._total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._null_damping_is_live
            else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with a compact torque array."""
        outputs = ControllerOperationalSpace.Outputs()
        outputs.joint_f = self._model_free.output().joint_f
        return outputs

    def step(
        self,
        *,
        inputs: Inputs,
        outputs: Outputs,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Run one operational-space control step.

        Computes forward kinematics and the enabled dynamics terms from
        ``model``, resolves the tool pose/twist/Jacobian from each robot's
        tool site, then delegates the control law to the inner
        :class:`ControllerOperationalSpaceModelFree`.

        Args:
            inputs: Populated :class:`Inputs` struct.
            outputs: :class:`Outputs` struct to write torques into.
            dt: Unused. Accepted for API compatibility.
        """
        for port, name, length in (
            (inputs.joint_q, "inputs.joint_q", self._coord_count),
            (inputs.joint_qd, "inputs.joint_qd", self._dof_count),
        ):
            _validate_array(
                array=port, name=name, dtype=wp.float32, shape=(length,), device=self._device, allow_indexed=True
            )

        # A port belonging to a disabled/baked feature is never forwarded to
        # the inner controller below, so writing one would go unnoticed.
        # getattr because a caller may leave the field unset rather than None.
        for name, enabled, switch in (
            ("operational_frame_pose_world", self._operational_frame_is_live, "a live operational_frame_pose_world"),
            ("motion_stiffness", self._motion_stiffness_is_live, "a live motion_stiffness"),
            ("motion_damping", self._motion_damping_is_live, "a live motion_damping"),
            ("desired_wrench_world", self._use_wrench, "use_wrench_feedforward or use_wrench_feedback"),
            ("measured_wrench_world", self._use_wrench_feedback, "use_wrench_feedback"),
            ("wrench_stiffness", self._wrench_stiffness_is_live, "a live wrench_stiffness"),
            (
                "linear_selection_frame_operational",
                self._linear_selection_frame_is_live,
                "a live linear_selection_frame_operational",
            ),
            (
                "angular_selection_frame_operational",
                self._angular_selection_frame_is_live,
                "a live angular_selection_frame_operational",
            ),
            ("joint_q_des_null", self._use_null_space, "use_null_space_control"),
            ("joint_qd_des_null", self._use_null_space, "use_null_space_control"),
            ("null_space_stiffness", self._null_stiffness_is_live, "a live null_space_stiffness"),
            ("null_space_damping", self._null_damping_is_live, "a live null_space_damping"),
        ):
            if not enabled and getattr(inputs, name, None) is not None:
                raise ValueError(
                    f"inputs.{name} is set, but the controller was built without {switch}, so the value "
                    f"would be ignored."
                )

        # Whole-model reads: an uncontrolled joint still moves its own body,
        # and hence the tool pose/twist/Jacobian/dynamics of every controlled
        # joint downstream of it.
        _read_port(inputs.joint_q, self._model_state.joint_q, self._coord_count, self._device)
        _read_port(inputs.joint_qd, self._model_state.joint_qd, self._dof_count, self._device)

        eval_fk(
            self._model,
            self._model_state.joint_q,
            self._model_state.joint_qd,
            self._model_state,
            mask=self._controlled_robot_mask,
        )
        eval_jacobian(self._model, self._model_state, J=self._jacobian_com_world, mask=self._controlled_robot_mask)

        wp.launch(
            _tool_pose_and_twist_kernel,
            dim=self._controlled_robot_count,
            inputs=[
                self._model_state.body_q,
                self._model_state.body_qd,
                self._model.body_com,
                self._tool_body,
                self._tool_transform_body,
            ],
            outputs=[self._tool_pose_world, self._tool_twist_world],
            device=self._device,
        )
        wp.launch(
            _shift_jacobian_to_tool_kernel,
            dim=(self._controlled_robot_count, self._max_controlled_dofs),
            inputs=[
                self._jacobian_com_world,
                self._model_state.body_q,
                self._model.body_com,
                self._tool_body,
                self._tool_transform_body,
                self._model_robot_index,
                self._robot_link_idx,
                self._articulation_dof_idx_of_padded_dof_idx,
                self._controlled_dofs_per_robot,
            ],
            outputs=[self._jacobian_tool_world],
            device=self._device,
        )

        if self._use_inertia:
            eval_mass_matrix(
                self._model,
                self._model_state,
                H=self._model_mass_matrix,
                J=self._jacobian_com_world,
                mask=self._controlled_robot_mask,
            )
            wp.launch(
                _gather_mass_matrix_blocks_kernel,
                dim=(self._controlled_robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                inputs=[
                    self._model_mass_matrix,
                    self._model_robot_index,
                    self._articulation_dof_idx_of_padded_dof_idx,
                    self._controlled_dofs_per_robot,
                ],
                outputs=[self._controlled_mass_matrix],
                device=self._device,
            )

        if self._use_gravity:
            eval_inverse_dynamics_passive(
                self._model, self._model_state, gravity_force=self._gravity_flat, mask=self._controlled_robot_mask
            )

        # Forward the remaining ports onto the inner controller's pre-wired
        # input struct, then delegate the control law to it.
        self._mf_input.desired_tool_pose_operational = inputs.desired_tool_pose_operational
        self._mf_input.desired_twist_operational = inputs.desired_twist_operational
        if self._operational_frame_is_live:
            self._mf_input.operational_frame_pose_world = inputs.operational_frame_pose_world
        if self._motion_stiffness_is_live:
            self._mf_input.motion_stiffness = inputs.motion_stiffness
        if self._motion_damping_is_live:
            self._mf_input.motion_damping = inputs.motion_damping
        if self._use_wrench:
            self._mf_input.desired_wrench_world = inputs.desired_wrench_world
        if self._use_wrench_feedback:
            self._mf_input.measured_wrench_world = inputs.measured_wrench_world
        if self._wrench_stiffness_is_live:
            self._mf_input.wrench_stiffness = inputs.wrench_stiffness
        if self._linear_selection_frame_is_live:
            self._mf_input.linear_selection_frame_operational = inputs.linear_selection_frame_operational
        if self._angular_selection_frame_is_live:
            self._mf_input.angular_selection_frame_operational = inputs.angular_selection_frame_operational
        if self._use_null_space:
            self._mf_input.joint_q_des_null = inputs.joint_q_des_null
            self._mf_input.joint_qd_des_null = inputs.joint_qd_des_null
        if self._null_stiffness_is_live:
            self._mf_input.null_space_stiffness = inputs.null_space_stiffness
        if self._null_damping_is_live:
            self._mf_input.null_space_damping = inputs.null_space_damping

        self._model_free.step(inputs=self._mf_input, outputs=outputs, dt=dt)
