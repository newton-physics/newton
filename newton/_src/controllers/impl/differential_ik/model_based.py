# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerDifferentialIK — differential-kinematics control with Newton
model-internal kinematics.

Calls :func:`newton.eval_fk` and :func:`newton.eval_jacobian` on the
supplied model each step, resolves each robot's tool-point pose and Jacobian
from a Newton *site*, then delegates the control law to an inner
:class:`ControllerDifferentialIKModelFree` instance.
"""

from __future__ import annotations

import re

import numpy as np
import warp as wp

from newton._src.sim.articulation import eval_fk, eval_jacobian
from newton._src.sim.model import Model

from ...controller import ControllerBase
from ...joint_selection import resolve_joint_selection
from ...tool_selection import resolve_tool_sites
from ...utils import _validate_array
from .._common import _read_port, _shift_jacobian_to_tool_kernel
from ._common import DifferentialIKMethod, _tool_pose_kernel
from .model_free import ControllerDifferentialIKModelFree


class ControllerDifferentialIK(ControllerBase):
    """Differential-kinematics (Jacobian-based) controller with internally computed kinematics.

    Implements a differential-kinematics control law, selectable per instance
    via :class:`DifferentialIKMethod` (damped least squares by default). This model-based
    variant computes the tool pose and tool-point Jacobian itself: it
    evaluates forward kinematics and :func:`newton.eval_jacobian` from
    ``model`` on every :meth:`step`, so the caller supplies only joint
    positions/velocities plus the desired tool pose.

    ``model`` is borrowed, not owned — it is never written to, and changes to
    it are visible to the controller immediately.

    **Joint selection.** ``articulations`` and ``joints`` select which DOFs
    become the tool Jacobian's columns, following :ref:`label-matching`: each
    is a list of model indices and/or label patterns (or a single pattern),
    matched against :attr:`~newton.Model.articulation_label` and the leaf
    component of :attr:`~newton.Model.joint_label` respectively. Only joints
    spanning a single coordinate and a single DOF can be controlled.

    **Tool selection.** ``tool_sites`` selects one Newton *site* per robot
    that ends up with controlled joints — the point on the robot whose pose
    is controlled. It follows the same ``list[index/pattern] | index |
    pattern`` shape as ``joints``, matched against the leaf component of each
    site's label. Every controlled robot must match exactly one site.

    Each articulation in ``model`` is one robot. Supports heterogeneous robot
    fleets — robots may have different controlled-DOF counts, and a robot may
    be left uncontrolled entirely by omitting it from ``articulations``.

    See also :class:`ControllerDifferentialIKModelFree`, which takes the tool pose
    and Jacobian as inputs instead of computing them from a
    :class:`~newton.Model`.

    Args:
        model: :class:`~newton.Model` whose articulations are the robots.
            ``model.requires_grad=True`` is not supported at this time.
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
            value would flip the output velocity's direction. Pass a scalar
            to apply the same gain to every controlled DOF, an array of
            shape [total_controlled_dofs] to set them individually, or
            ``None`` to read ``inputs.bandwidth`` each step.
        damping: Damped-least-squares regularization λ, applied per robot to
            the task-space normal-equations matrix. Pass a scalar to apply
            the same damping to every robot, an array of shape
            [controlled_robot_count] to set them individually, or ``None``
            to read ``inputs.damping`` each step.
            Only meaningful when ``ik_method=DifferentialIKMethod.DAMPED_LEAST_SQUARES``
            (the default); must be ``None`` for every other
            :class:`DifferentialIKMethod`, which has no λ to set.
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
            controlled DOF. Must be non-negative. Pass a scalar to apply
            the same gain to every controlled DOF, an array of shape
            [total_controlled_dofs] to set them individually, or ``None``
            to read ``inputs.null_space_stiffness`` each step. Must be
            ``None`` when ``use_null_space_posture_control=False``.
        null_space_damping: Damping λ_null for the null-space projector's own
            ``(JJᵀ + λ_null²I)⁻¹``, independent of the primary task's
            ``damping``. Must be non-negative. Only meaningful when
            ``use_joint_limit_avoidance`` or ``use_null_space_posture_control``
            is enabled — pass a scalar or an array of shape
            [controlled_robot_count] to bake a value, or leave it ``None``
            to read ``inputs.null_space_damping`` each step (the default,
            and the only valid value when both are disabled). Unlike the
            primary ``damping``, ``λ_null = 0`` is only safe when every
            robot has at least as many controlled DOFs as its own task
            dimension (the number of nonzero ``null_space_axes`` entries) —
            otherwise the projector's own ``JJᵀ`` is rank-deficient. That
            stronger, per-robot requirement is checked at construction only
            when baked; a live value is the caller's responsibility there.
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
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerDifferentialIK.input`.

        ``joint_q``/``joint_qd`` cover the whole model, since forward
        kinematics depends on uncontrolled joints too; every other field is
        either per-robot or compact (one entry per controlled DOF). Optional
        fields are ``None`` when the corresponding feature is disabled at
        construction.
        """

        joint_q: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Current joint positions [m or rad], shape [model.joint_coord_count]."""
        joint_qd: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Current joint velocities [m/s or rad/s], shape [model.joint_dof_count]."""
        desired_tool_pose_world: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Desired tool pose [m, unitless quaternion], world frame, shape [controlled_robot_count]."""
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
        """Output struct returned by :meth:`~ControllerDifferentialIK.output`."""

        joint_qd_target: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Target joint velocity [m/s or rad/s], shape [total_controlled_dofs]."""
        joint_q_target: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """One-step-ahead target joint position [m or rad], shape [total_controlled_dofs] = ``joint_q + joint_qd_target * dt``."""

    def __init__(
        self,
        model: Model,
        *,
        articulations: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
        joints: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
        tool_sites: list[int | str | re.Pattern[str]] | str | re.Pattern[str],
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
    ):
        if not isinstance(model, Model):
            raise TypeError(f"model must be a newton.Model, got {type(model).__name__}.")
        model_robot_count = model.articulation_count
        if model_robot_count < 1:
            raise ValueError("model has no articulations.")

        self._device = model.device
        self._requires_grad = model.requires_grad
        self._bandwidth_is_live = bandwidth is None
        self._damping_is_live = ik_method == DifferentialIKMethod.DAMPED_LEAST_SQUARES and damping is None
        self._use_joint_limit_avoidance = bool(use_joint_limit_avoidance)
        self._use_null_space_posture_control = bool(use_null_space_posture_control)
        self._use_null_space = self._use_joint_limit_avoidance or self._use_null_space_posture_control
        self._null_stiffness_is_live = self._use_null_space_posture_control and null_space_stiffness is None
        self._null_damping_is_live = self._use_null_space and null_space_damping is None

        self._model = model
        self._model_state = model.state(requires_grad=self._requires_grad)
        self._coord_count = int(model.joint_coord_count)
        self._dof_count = int(model.joint_dof_count)

        joints_resolved = resolve_joint_selection(
            model,
            articulations=articulations,
            joints=joints,
            device=self._device,
            controller_name="ControllerDifferentialIK",
            ownerless_joint_reason="The controller runs forward kinematics per robot, so such a joint has no Jacobian.",
        )
        qd_idx_np = joints_resolved.qd_idx_np
        model_robot_index_np = joints_resolved.model_robot_index_np
        controlled_dofs_per_robot_np = joints_resolved.controlled_dofs_per_robot_np
        controlled_robot_count = joints_resolved.controlled_robot_count
        max_controlled_dofs = joints_resolved.max_controlled_dofs

        self._model_robot_index = joints_resolved.model_robot_index
        self._controlled_robot_mask = joints_resolved.controlled_robot_mask
        self._model_robot_count = model_robot_count
        self._controlled_robot_count = controlled_robot_count
        self._max_controlled_dofs = max_controlled_dofs
        self._total_controlled_dofs = joints_resolved.total_controlled_dofs
        controlled_dofs_per_robot = joints_resolved.controlled_dofs_per_robot
        self._controlled_dofs_per_robot = controlled_dofs_per_robot
        self._q_idx = joints_resolved.q_idx
        self._qd_idx = joints_resolved.qd_idx

        tool_sites_resolved = resolve_tool_sites(
            model, model_robot_index_np=model_robot_index_np, tool_sites=tool_sites, device=self._device
        )
        self._tool_body = tool_sites_resolved.tool_body
        self._tool_transform_body = tool_sites_resolved.tool_transform_body
        self._robot_link_idx = tool_sites_resolved.robot_link_idx

        self._articulation_dof_idx_of_padded_dof_idx = wp.array(
            self._compute_articulation_dof_idx_of_padded_dof_idx(
                qd_idx_np=qd_idx_np,
                model_robot_index_np=model_robot_index_np,
                controlled_dofs_per_robot_np=controlled_dofs_per_robot_np,
            ),
            dtype=wp.int32,
            device=self._device,
        )

        model_max_links = model.max_joints_per_articulation
        model_max_dofs = model.max_dofs_per_articulation
        self._jacobian_com_world = wp.zeros(
            (model_robot_count, model_max_links * 6, model_max_dofs),
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

        self._model_free = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=controlled_dofs_per_robot,
            axis_weight=axis_weight,
            bandwidth=bandwidth,
            damping=damping,
            ik_method=ik_method,
            adaptive_damping_min=adaptive_damping_min,
            adaptive_damping_max=adaptive_damping_max,
            adaptive_damping_threshold=adaptive_damping_threshold,
            truncated_svd_threshold=truncated_svd_threshold,
            use_joint_limit_avoidance=use_joint_limit_avoidance,
            joint_limit_avoidance_gain=joint_limit_avoidance_gain,
            joint_limit_avoidance_margin=joint_limit_avoidance_margin,
            joint_pos_lower=joint_pos_lower,
            joint_pos_upper=joint_pos_upper,
            use_null_space_posture_control=use_null_space_posture_control,
            null_space_stiffness=null_space_stiffness,
            null_space_damping=null_space_damping,
            null_space_axes=null_space_axes,
            device=self._device,
            requires_grad=self._requires_grad,
        )

        # Pre-wired fields forwarded to the inner controller each step: live
        # indexed views of the whole-model/tool buffers above, so the inner
        # controller reads current contents with no index table of its own.
        self._mf_input = ControllerDifferentialIKModelFree.Inputs()
        self._mf_input.tool_pose_world = self._tool_pose_world
        self._mf_input.jacobian_tool_world = self._jacobian_tool_world
        self._mf_input.joint_q = self._model_state.joint_q[self._q_idx]

    def _compute_articulation_dof_idx_of_padded_dof_idx(
        self, *, qd_idx_np: np.ndarray, model_robot_index_np: np.ndarray, controlled_dofs_per_robot_np: np.ndarray
    ) -> np.ndarray:
        """Return, for each (controlled robot, padded slot), the DOF's index within that robot.

        ``joint_selection.qd_start`` is in the model's DOF numbering, but
        :func:`~newton.eval_jacobian` indexes each robot's block by
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
        """Tool site's transform [m, unitless quaternion] relative to its body, shape [controlled_robot_count]."""
        return self._tool_transform_body

    @property
    def tool_pose_world(self) -> wp.array[wp.transform]:
        """World pose [m, unitless quaternion] of each controlled robot's tool site as of the latest ``step()``, shape [controlled_robot_count]."""
        return self._tool_pose_world

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

        inputs = ControllerDifferentialIK.Inputs()
        inputs.joint_q = wp.zeros(self._coord_count, dtype=wp.float32, device=device, requires_grad=requires_grad)
        inputs.joint_qd = wp.zeros(self._dof_count, dtype=wp.float32, device=device, requires_grad=requires_grad)
        inputs.desired_tool_pose_world = wp.zeros(
            controlled_robot_count, dtype=wp.transform, device=device, requires_grad=requires_grad
        )
        inputs.bandwidth = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._bandwidth_is_live
            else None
        )
        inputs.damping = (
            wp.zeros(controlled_robot_count, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._damping_is_live
            else None
        )
        inputs.q_des_null = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._use_null_space_posture_control
            else None
        )
        inputs.null_space_stiffness = (
            wp.zeros(total_controlled_dofs, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._null_stiffness_is_live
            else None
        )
        inputs.null_space_damping = (
            wp.zeros(controlled_robot_count, dtype=wp.float32, device=device, requires_grad=requires_grad)
            if self._null_damping_is_live
            else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with compact velocity/position arrays."""
        mf_outputs = self._model_free.output()
        outputs = ControllerDifferentialIK.Outputs()
        outputs.joint_qd_target = mf_outputs.joint_qd_target
        outputs.joint_q_target = mf_outputs.joint_q_target
        return outputs

    def set_joint_limits(self, *, joint_pos_lower: wp.array[wp.float32], joint_pos_upper: wp.array[wp.float32]) -> None:
        """Update the joint position limits used by joint-limit avoidance, in place.

        Args:
            joint_pos_lower: Lower joint position limit per controlled DOF
                [m or rad], shape [total_controlled_dofs].
            joint_pos_upper: Upper joint position limit per controlled DOF
                [m or rad], shape [total_controlled_dofs].
        """
        self._model_free.set_joint_limits(joint_pos_lower=joint_pos_lower, joint_pos_upper=joint_pos_upper)

    def step(
        self,
        *,
        inputs: Inputs,
        outputs: Outputs,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Run one differential-kinematics step.

        Computes forward kinematics and the tool-point Jacobian from
        ``model``, then delegates the control law to the inner
        :class:`ControllerDifferentialIKModelFree`.

        Args:
            inputs: Populated :class:`Inputs` struct.
            outputs: :class:`Outputs` struct to write into.
            dt: Step duration [s], used to integrate ``joint_qd_target`` into
                ``joint_q_target``.
        """
        for port, name, length in (
            (inputs.joint_q, "inputs.joint_q", self._coord_count),
            (inputs.joint_qd, "inputs.joint_qd", self._dof_count),
        ):
            _validate_array(
                array=port, name=name, dtype=wp.float32, shape=(length,), device=self._device, allow_indexed=True
            )

        # A port belonging to a disabled feature or a baked gain is never
        # forwarded to the inner controller, so writing one would go
        # unnoticed. getattr because a caller may leave the field unset
        # rather than None.
        for name, live, switch in (
            ("bandwidth", self._bandwidth_is_live, "a live bandwidth"),
            ("damping", self._damping_is_live, "a live damping"),
            ("q_des_null", self._use_null_space_posture_control, "use_null_space_posture_control"),
            ("null_space_stiffness", self._null_stiffness_is_live, "a live null_space_stiffness"),
            ("null_space_damping", self._null_damping_is_live, "a live null_space_damping"),
        ):
            if not live and getattr(inputs, name, None) is not None:
                raise ValueError(
                    f"inputs.{name} is set, but the controller was built without {switch}, so the value "
                    f"would be ignored."
                )

        # Whole-model reads: an uncontrolled joint still moves its own body,
        # and hence the tool pose/Jacobian of every controlled joint
        # downstream of it.
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
            _tool_pose_kernel,
            dim=self._controlled_robot_count,
            inputs=[self._model_state.body_q, self._tool_body, self._tool_transform_body],
            outputs=[self._tool_pose_world],
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

        # Forward the remaining ports onto the inner controller's pre-wired
        # input struct, then delegate the control law to it.
        self._mf_input.desired_tool_pose_world = inputs.desired_tool_pose_world
        if self._bandwidth_is_live:
            self._mf_input.bandwidth = inputs.bandwidth
        if self._damping_is_live:
            self._mf_input.damping = inputs.damping
        if self._use_null_space_posture_control:
            self._mf_input.q_des_null = inputs.q_des_null
        if self._null_stiffness_is_live:
            self._mf_input.null_space_stiffness = inputs.null_space_stiffness
        if self._null_damping_is_live:
            self._mf_input.null_space_damping = inputs.null_space_damping

        self._model_free.step(inputs=self._mf_input, outputs=outputs, dt=dt)
