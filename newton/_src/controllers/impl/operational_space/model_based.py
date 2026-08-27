# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerOperationalSpace — operational-space impedance control with
Newton model-internal kinematics and dynamics.

Calls :func:`newton.eval_fk`, :func:`newton.eval_jacobian`, and (when
inertial decoupling is enabled) :func:`newton.eval_mass_matrix` on the
supplied model each step, resolves each robot's tool-point pose, twist, and
Jacobian from a Newton *site*, then delegates the control law to an inner
:class:`ControllerOperationalSpaceModelFree` instance.

This increment implements only the constructor: robot and tool-site
selection, validation, and buffer allocation. :meth:`step` is not
implemented yet.
"""

from __future__ import annotations

import re

import numpy as np
import warp as wp

from newton._src.geometry.flags import ShapeFlags
from newton._src.sim.model import Model

from ....utils.selection import get_name_from_label, match_labels
from ...controller import ControllerBase
from ...joint_selection import select_joints
from ...utils import _validate_array
from .model_free import ControllerOperationalSpaceModelFree


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
    become the tool Jacobian's columns, following the same
    :ref:`label-matching` convention and single-coordinate/single-DOF
    restriction that :class:`~newton.controllers.ControllerJointImpedance`
    uses, resolved with the same internal joint-selection logic: each is a
    list of model indices and/or label patterns (or a single pattern),
    matched against :attr:`~newton.Model.articulation_label` and the leaf
    component of :attr:`~newton.Model.joint_label` respectively.

    **Tool selection.** ``tool`` selects one Newton *site* per robot that
    ends up with controlled joints — the task frame every task-space port is
    defined relative to. It follows the same
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
        tool: Site index(es) or label pattern(s) selecting each controlled
            robot's task frame, as a list or as a single pattern. Required —
            there is no default tool site. Raises if a controlled robot
            matches zero or more than one site.
        motion_stiffness: Forwarded to
            :class:`ControllerOperationalSpaceModelFree` unchanged; see its
            docstring for units, format, and the live-gain (``None``)
            convention.
        motion_damping: Forwarded unchanged; see
            :class:`ControllerOperationalSpaceModelFree`.
        use_inertia_decoupling: Forwarded unchanged.
        use_partial_inertia_decoupling: Forwarded unchanged.
        use_gravity_compensation: Forwarded unchanged.
        use_wrench_feedforward: Forwarded unchanged.
        use_wrench_feedback: Forwarded unchanged.
        motion_selection_axes_tool: Forwarded unchanged.
        wrench_selection_axes_tool: Forwarded unchanged.
        wrench_stiffness: Forwarded unchanged.
        use_null_space_control: Forwarded unchanged.
        null_space_stiffness: Forwarded unchanged.
        null_space_damping: Forwarded unchanged.
    """

    def __init__(
        self,
        model: Model,
        *,
        articulations: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
        joints: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
        tool: list[int | str | re.Pattern[str]] | str | re.Pattern[str],
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
        self._use_null_space = bool(use_null_space_control)

        self._model = model
        self._model_state = model.state(requires_grad=self._requires_grad)
        self._coord_count = int(model.joint_coord_count)
        self._dof_count = int(model.joint_dof_count)

        joint_selection = select_joints(model, articulations=articulations, joints=joints)
        joint_q_idx = joint_selection.q_start
        joint_qd_idx = joint_selection.qd_start

        # ------------------------------------------------------------------
        # Validation of the two model-space index arrays select_joints
        # returns. Identical to ControllerJointImpedance's block: OSC's
        # Jacobian columns are resolved by the exact same "which DOFs does
        # this robot control" problem that controller solves for its
        # joint-space PD term.
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

        tool_entries = [tool] if isinstance(tool, (int, str, re.Pattern)) else tool
        matched_sites: list[int] = []
        for entry in tool_entries:
            if isinstance(entry, int):
                if entry not in site_indices_np:
                    raise ValueError(f"tool site index {entry} is not a site in the model.")
                matched_sites.append(entry)
            else:
                local_matches = match_labels(site_names, entry)
                if not local_matches:
                    raise ValueError(f"tool pattern {entry!r} matches no site in the model.")
                matched_sites.extend(int(site_indices_np[m]) for m in local_matches)
        matched_sites_set = sorted(set(matched_sites))

        site_index_to_articulation = dict(zip(site_indices_np.tolist(), site_articulation_np.tolist(), strict=True))
        tool_body_np = np.zeros(controlled_robot_count, dtype=np.int32)
        tool_transform_body: list[wp.transform] = []
        for robot_slot, art in enumerate(model_robot_index_np.tolist()):
            sites_on_robot = [s for s in matched_sites_set if site_index_to_articulation[s] == art]
            if len(sites_on_robot) == 0:
                raise ValueError(f"tool matches no site on articulation {art}.")
            if len(sites_on_robot) > 1:
                raise ValueError(
                    f"tool matches {len(sites_on_robot)} sites on articulation {art}; exactly one "
                    f"tool site is required per robot."
                )
            site = sites_on_robot[0]
            tool_body_np[robot_slot] = shape_body_np[site]
            tool_transform_body.append(wp.transform(*shape_transform_np[site]))

        self._tool_body = wp.array(tool_body_np, dtype=wp.int32, device=self._device)
        self._tool_transform_body = wp.array(tool_transform_body, dtype=wp.transform, device=self._device)

        # robot_link_idx: the tool body's row-block index within its
        # articulation's eval_jacobian output. eval_articulation_jacobian
        # writes link i's rows at [i*6 : i*6+6], where i is the position of
        # the joint whose child is that body within the articulation's own
        # joint range -- so this is (joint index of the tool body) minus
        # (that articulation's first joint index).
        joint_of_body_np = np.full(model.body_count, -1, dtype=np.int32)
        joint_of_body_np[joint_child_np] = np.arange(joint_child_np.size, dtype=np.int32)
        articulation_start_np = model.articulation_start.numpy()
        robot_link_idx_np = (joint_of_body_np[tool_body_np] - articulation_start_np[model_robot_index_np]).astype(
            np.int32
        )
        self._robot_link_idx = wp.array(robot_link_idx_np, dtype=wp.int32, device=self._device)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Dynamics buffers. Allocated up front, mirroring
        # ControllerJointImpedance; populated by step() (not implemented
        # yet).
        # ------------------------------------------------------------------
        self._model_mass_matrix: wp.array3d[wp.float32] | None = None
        self._controlled_mass_matrix: wp.array3d[wp.float32] | None = None
        self._local_dof_idx: wp.array2d[wp.int32] | None = None
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
            self._local_dof_idx = wp.array(
                self._compute_local_dof_idx(
                    qd_idx_np=qd_idx_np,
                    model_robot_index_np=model_robot_index_np,
                    controlled_dofs_per_robot_np=controlled_dofs_per_robot_np,
                ),
                dtype=wp.int32,
                device=self._device,
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
            use_inertia_decoupling=use_inertia_decoupling,
            use_partial_inertia_decoupling=use_partial_inertia_decoupling,
            use_gravity_compensation=use_gravity_compensation,
            use_wrench_feedforward=use_wrench_feedforward,
            use_wrench_feedback=use_wrench_feedback,
            motion_selection_axes_tool=motion_selection_axes_tool,
            wrench_selection_axes_tool=wrench_selection_axes_tool,
            wrench_stiffness=wrench_stiffness,
            use_null_space_control=use_null_space_control,
            null_space_stiffness=null_space_stiffness,
            null_space_damping=null_space_damping,
            device=self._device,
            requires_grad=self._requires_grad,
        )

        # Pre-wired fields forwarded to the inner controller each step,
        # mirroring ControllerJointImpedance's self._mf_input pattern. Live
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

    def _compute_local_dof_idx(
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

        local_dof_idx = np.zeros((controlled_robot_count, self._max_controlled_dofs), dtype=np.int32)
        for robot in range(controlled_robot_count):
            dof_count = int(controlled_dofs_per_robot_np[robot])
            chunk = qd_idx_np[offsets[robot] : offsets[robot] + dof_count]
            local_dof_idx[robot, :dof_count] = chunk - robot_dof_start[robot]
        return local_dof_idx

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

    def input(self):
        """Not implemented yet -- see the module docstring."""
        raise NotImplementedError("ControllerOperationalSpace.input() is not implemented yet.")

    def output(self):
        """Not implemented yet -- see the module docstring."""
        raise NotImplementedError("ControllerOperationalSpace.output() is not implemented yet.")

    def step(self, *, inputs, outputs, dt):
        """Not implemented yet -- see the module docstring."""
        raise NotImplementedError("ControllerOperationalSpace.step() is not implemented yet.")
