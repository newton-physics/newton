# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerDiffIK — differential-kinematics control with Newton
model-internal kinematics.

Calls :func:`newton.eval_fk` and :func:`newton.eval_jacobian` on the
supplied model each step, resolves each robot's tool-point pose and Jacobian
from a Newton *site*, then delegates the control law to an inner
:class:`ControllerDiffIKModelFree` instance.
"""

from __future__ import annotations

import re

import numpy as np
import warp as wp

from newton import JointType
from newton._src.geometry.flags import ShapeFlags
from newton._src.sim.articulation import eval_fk, eval_jacobian
from newton._src.sim.model import Model

from ....utils.selection import get_name_from_label, match_labels
from ...controller import ControllerBase
from ...joint_selection import select_joints
from ...utils import _validate_array
from ._common import _read_port, _shift_jacobian_to_tool_kernel, _tool_pose_kernel
from .model_free import ControllerDiffIKModelFree


class ControllerDiffIK(ControllerBase):
    """Differential-kinematics (Jacobian-based) controller with internally computed kinematics.

    Implements the damped-least-squares differential-kinematics control law.
    This model-based variant computes the tool pose and tool-point Jacobian
    itself: it evaluates forward kinematics and :func:`newton.eval_jacobian`
    from ``model`` on every :meth:`step`, so the caller supplies only joint
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

    See also :class:`ControllerDiffIKModelFree`, which takes the tool pose
    and Jacobian as inputs instead of computing them from a
    :class:`~newton.Model`.

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
        tool_sites: Site index(es) or label pattern(s) selecting each
            controlled robot's controlled point, as a list or as a single
            pattern. Required — there is no default tool site. Raises if a
            controlled robot matches zero or more than one site.
        bandwidth: Output velocity scale gain, applied per controlled DOF
            after the Jacobian solve. Pass a scalar to apply the same gain
            to every controlled DOF, an array of shape
            [total_controlled_dofs] to set them individually, or ``None`` to
            read ``inputs.bandwidth`` each step.
        damping: Damped-least-squares regularization λ, applied per robot to
            the task-space normal-equations matrix. Pass a scalar to apply
            the same damping to every robot, an array of shape
            [controlled_robot_count] to set them individually, or ``None``
            to read ``inputs.damping`` each step. ``λ = 0`` reduces the
            solve to the ordinary Moore-Penrose pseudo-inverse.
        use_joint_limit_avoidance: Project a joint-limit-avoidance bias
            through the null-space projector.
        joint_limit_avoidance_gain: Joint-centering gain, applied once a DOF
            comes within ``joint_limit_avoidance_margin`` of either limit.
            Required (and must be positive) when
            ``use_joint_limit_avoidance=True``.
        joint_limit_avoidance_margin: Distance from either limit at which
            the avoidance bias starts ramping in, same units as
            ``joint_pos_lower``/``joint_pos_upper``. Required (and must be
            positive) when ``use_joint_limit_avoidance=True``.
        joint_pos_lower: Lower joint position limit per controlled DOF,
            shape [total_controlled_dofs]. Required when
            ``use_joint_limit_avoidance=True``; baked at construction, not a
            live port.
        joint_pos_upper: Upper joint position limit per controlled DOF,
            shape [total_controlled_dofs]. Required when
            ``use_joint_limit_avoidance=True``; baked at construction, not a
            live port.
        use_null_space_posture_control: Project a proportional pull toward
            ``inputs.q_des_null`` through the null-space projector.
        null_space_stiffness: Posture-control proportional gain, applied per
            controlled DOF. Pass a scalar to apply the same gain to every
            controlled DOF, an array of shape [total_controlled_dofs] to set
            them individually, or ``None`` to read
            ``inputs.null_space_stiffness`` each step. Must be ``None`` when
            ``use_null_space_posture_control=False``.
        null_space_damping: Damping λ_null for the null-space projector's own
            ``(JJᵀ + λ_null²I)⁻¹``, independent of the primary task's
            ``damping``. Only meaningful when ``use_joint_limit_avoidance``
            or ``use_null_space_posture_control`` is enabled — pass a scalar
            or an array of shape [controlled_robot_count] to bake a value,
            or leave it ``None`` to read ``inputs.null_space_damping`` each
            step (the default, and the only valid value when both are
            disabled).
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerDiffIK.input`.

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
        """Desired tool pose, world frame, shape [controlled_robot_count]."""
        bandwidth: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Output velocity scale gain, shape [total_controlled_dofs]. ``None`` when baked at construction."""
        damping: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Damped-least-squares regularization λ, shape [controlled_robot_count]. ``None`` when baked at construction."""
        q_des_null: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Null-space posture target, shape [total_controlled_dofs]. ``None`` unless ``use_null_space_posture_control=True``."""
        null_space_stiffness: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Posture-control proportional gain, shape [total_controlled_dofs]. ``None`` when disabled, or when baked at construction."""
        null_space_damping: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Null-space projector damping λ_null, shape [controlled_robot_count]. ``None`` when both secondary objectives are disabled, or when baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerDiffIK.output`."""

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
        bandwidth: wp.array[wp.float32] | float | None,
        damping: wp.array[wp.float32] | float | None,
        use_joint_limit_avoidance: bool = False,
        joint_limit_avoidance_gain: float = 0.0,
        joint_limit_avoidance_margin: float = 0.0,
        joint_pos_lower: wp.array[wp.float32] | None = None,
        joint_pos_upper: wp.array[wp.float32] | None = None,
        use_null_space_posture_control: bool = False,
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
        self._bandwidth_is_live = bandwidth is None
        self._damping_is_live = damping is None
        self._use_joint_limit_avoidance = bool(use_joint_limit_avoidance)
        self._use_null_space_posture_control = bool(use_null_space_posture_control)
        self._use_null_space = self._use_joint_limit_avoidance or self._use_null_space_posture_control
        self._null_stiffness_is_live = self._use_null_space_posture_control and null_space_stiffness is None
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
        #   4. qd_start has no duplicate DOF
        #   5. q_start[i]/qd_start[i] name the same joint for every i
        #   6. every addressed joint spans a single coordinate and single DOF
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

        # A joint is controllable when its DOF maps to exactly one Jacobian
        # column, i.e. it spans exactly one coordinate and one DOF.
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
                f"ControllerDiffIK only supports controlling joints that span a single coordinate and a "
                f"single DOF; joint_selection addresses unsupported joints: {unsupported}"
            )

        owning_robot = model.joint_articulation.numpy()[owning_joint]
        loose = np.flatnonzero(owning_robot < 0)
        if loose.size:
            raise ValueError(
                f"joint_selection addresses joint {int(owning_joint[loose[0]])}, which belongs to no "
                f"robot. The controller runs forward kinematics per robot, so such a joint has no Jacobian."
            )
        if np.any(np.diff(owning_robot) < 0):
            raise ValueError(
                "joint_selection.q_start/qd_start must be grouped by robot (robot 0's DOFs first, "
                f"then robot 1's, ...); got robot order {owning_robot.tolist()}."
            )

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

        tool_entries = [tool_sites] if isinstance(tool_sites, (int, str, re.Pattern)) else tool_sites
        matched_sites: list[int] = []
        for entry in tool_entries:
            if isinstance(entry, int):
                if entry not in site_indices_np:
                    raise ValueError(f"tool_sites index {entry} is not a site in the model.")
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
        # articulation's eval_jacobian output. eval_jacobian writes link i's
        # rows at [i*6 : i*6+6], where i is the position, within its
        # articulation's own joint range, of the joint that moves the tool
        # site's body -- so this is (that joint's index) minus (the
        # articulation's first joint index).
        body_to_joint_np = np.full(model.body_count, -1, dtype=np.int32)
        body_to_joint_np[joint_child_np] = np.arange(joint_child_np.size, dtype=np.int32)
        tool_site_joint_np = body_to_joint_np[tool_body_np]
        articulation_start_np = model.articulation_start.numpy()
        robot_link_idx_np = (tool_site_joint_np - articulation_start_np[model_robot_index_np]).astype(np.int32)
        self._robot_link_idx = wp.array(robot_link_idx_np, dtype=wp.int32, device=self._device)
        # ------------------------------------------------------------------

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

        self._model_free = ControllerDiffIKModelFree(
            controlled_dofs_per_robot=controlled_dofs_per_robot,
            bandwidth=bandwidth,
            damping=damping,
            use_joint_limit_avoidance=use_joint_limit_avoidance,
            joint_limit_avoidance_gain=joint_limit_avoidance_gain,
            joint_limit_avoidance_margin=joint_limit_avoidance_margin,
            joint_pos_lower=joint_pos_lower,
            joint_pos_upper=joint_pos_upper,
            use_null_space_posture_control=use_null_space_posture_control,
            null_space_stiffness=null_space_stiffness,
            null_space_damping=null_space_damping,
            device=self._device,
            requires_grad=self._requires_grad,
        )

        # Pre-wired fields forwarded to the inner controller each step: live
        # indexed views of the whole-model/tool buffers above, so the inner
        # controller reads current contents with no index table of its own.
        self._mf_input = ControllerDiffIKModelFree.Inputs()
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

        inputs = ControllerDiffIK.Inputs()
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
        outputs = ControllerDiffIK.Outputs()
        outputs.joint_qd_target = mf_outputs.joint_qd_target
        outputs.joint_q_target = mf_outputs.joint_q_target
        return outputs

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
        :class:`ControllerDiffIKModelFree`.

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
                raise ValueError(f"inputs.{name} is set, but the controller was built without {switch}.")

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
