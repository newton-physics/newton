# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerJointImpedance — joint-space impedance control with
Newton model-internal dynamics.

Calls :func:`newton.eval_fk` and :func:`newton.eval_mass_matrix` on the
supplied model each step, extracts the controlled block of the mass matrix,
then delegates the control law to an inner
:class:`ControllerJointImpedanceModelFree` instance.

Gravity and Coriolis compensation use :func:`newton.eval_inverse_dynamics_passive`.
"""

from __future__ import annotations

import re

import numpy as np
import warp as wp

from newton import JointType
from newton._src.sim.articulation import eval_fk, eval_mass_matrix
from newton._src.sim.inverse_dynamics import eval_inverse_dynamics_passive
from newton._src.sim.model import Model

from ...controller import ControllerBase
from ...joint_selection import select_joints
from ...utils import _validate_array
from ._common import _gather_mass_matrix_blocks_kernel, _read_port
from .model_free import ControllerJointImpedanceModelFree


class ControllerJointImpedance(ControllerBase):
    """Joint-space impedance controller with internally computed dynamics.

    Implements the joint-space impedance control law. This model-based variant
    computes the mass matrix, gravity, and Coriolis terms itself: it evaluates
    forward kinematics and the enabled dynamics terms from ``model`` on every
    :meth:`step`, so the caller supplies only joint positions and velocities.

    ``model`` is borrowed, not owned — it is never written to, and changes to
    it are visible to the controller immediately.

    **Joint selection.** ``articulations`` and ``joints`` select the
    controlled joints, following :ref:`label-matching`: each is a list of
    model indices and/or label patterns (or a single pattern), matched
    against :attr:`~newton.Model.articulation_label` and the leaf component
    of :attr:`~newton.Model.joint_label` respectively. The constructor
    resolves them to each selected joint's starting coordinate/DOF index in
    the model — one entry per joint, not per DOF — and validates the result.
    :attr:`q_start` and :attr:`qd_start` expose the resolved indices
    afterward, e.g. to gather/scatter a compact port against a
    simulation-sized array.

    **Ports.** Most arrays passed in and out are **compact**: one entry per
    controlled DOF — robot 0's DOFs first, then robot 1's — rather than one
    entry per DOF in the model. ``inputs.joint_q`` and ``inputs.joint_qd`` are
    the exception and cover the whole model, since the dynamics terms depend on
    uncontrolled joints too. A compact port may be bound to a plain array, or to
    an indexed view of a simulation-sized array::

        outputs.joint_f = control.joint_f[controller.qd_start]  # scatter to the sim

    Each articulation in ``model`` is one robot. Only joints spanning a single
    coordinate and a single DOF can be controlled, since the PD error term
    ``q_des - q`` is only a well-defined scalar subtraction for those; every
    other joint (Fixed, or any multi-DOF type) is read for FK and dynamics but
    never actuated. The default ``joints`` selection leaves such joints
    uncontrolled automatically; explicitly naming one in ``joints`` raises
    ``ValueError`` at construction instead, as does addressing a joint that
    belongs to no robot, or the same DOF twice.

    Supports heterogeneous robot fleets — robots may have different
    controlled-DOF counts, and a robot may be left uncontrolled entirely. An
    uncontrolled robot occupies no slot in any buffer and is masked out of the
    FK and dynamics evaluations, so its :attr:`~newton.State.body_q` is left
    untouched. :attr:`model_robot_count` counts every robot in the model,
    :attr:`controlled_robot_count` only those with controlled DOFs.

    See also :class:`ControllerJointImpedanceModelFree`, which takes the mass
    matrix, gravity, and Coriolis terms as inputs instead of computing them
    from a :class:`~newton.Model`.

    Impedance law (terms enabled at construction):

        τ = [M(q) if use_inertia_decoupling else I] · (q̈_des + Kp·Δq + Kd·Δq̇)
            + [C(q,q̇)·q̇ if use_coriolis_compensation else 0]
            + [g(q)      if use_gravity_compensation  else 0]

    Args:
        model: :class:`~newton.Model` whose articulations are the robots.
            Articulations may mix controlled single-DOF joints with
            uncontrolled joints of any type. The controller's device and
            ``requires_grad`` are taken from ``model``; every other array
            argument (``stiffness``, ``damping``) must match both.
        articulations: Articulation indices or label patterns to control, as a
            list or as a single pattern. ``None`` selects every articulation
            in ``model``.
        joints: Model joint indices or label patterns to control within the
            selected articulations, as a list or as a single pattern. ``None``
            selects every joint spanning exactly one coordinate and one DOF —
            the only kind this controller can actuate — in each selected
            articulation; any other joint (Fixed, or a multi-DOF type such as
            a floating base) is left uncontrolled instead of rejected. A
            joint named explicitly is not filtered this way and still raises
            ``ValueError`` if it is not 1-coordinate/1-DOF.
        stiffness: Position-error gain Kp. Units depend on
            ``use_inertia_decoupling``: [1/s²] when enabled, since the PD term
            is then an acceleration premultiplied by M(q); otherwise [N/m or
            N·m/rad]. Pass a scalar to apply the same gain to every controlled
            DOF, an array of shape [total_controlled_dofs] to set them
            individually, or ``None`` to read ``inputs.stiffness`` each step.
        damping: Velocity-error gain Kd, [1/s] when
            ``use_inertia_decoupling`` is enabled, otherwise
            [N·s/m or N·m·s/rad]. Same format as ``stiffness``.
        use_gravity_compensation: Add gravity generalized forces to τ.
        use_coriolis_compensation: Add Coriolis generalized forces to τ.
        use_inertia_decoupling: Premultiply the PD term by M(q).
        has_qdd_feedforward: Accept a desired-acceleration feedforward via
            ``inputs.joint_qdd``.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerJointImpedance.input`.

        Dynamics fields (mass matrix, gravity, Coriolis) are computed
        internally and do not appear here.
        """

        joint_q: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Current joint positions [m or rad], shape [model.joint_coord_count]."""
        joint_qd: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Current joint velocities [m/s or rad/s], shape [model.joint_dof_count]."""
        joint_q_des: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Desired joint positions [m or rad], shape [total_controlled_dofs]."""
        joint_qd_des: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Desired joint velocities [m/s or rad/s], shape [total_controlled_dofs]."""
        joint_qdd: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Desired acceleration feedforward [m/s² or rad/s²], shape [total_controlled_dofs]. ``None`` unless ``has_qdd_feedforward=True``."""
        stiffness: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Position-error gain Kp, shape [total_controlled_dofs]. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m or N·m/rad]. ``None`` when gains are baked at construction."""
        damping: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Velocity-error gain Kd, shape [total_controlled_dofs]. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m or N·m·s/rad]. ``None`` when gains are baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerJointImpedance.output`."""

        joint_f: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Joint torque command [N or N·m], shape [total_controlled_dofs]."""

    def __init__(
        self,
        model: Model,
        *,
        articulations: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
        joints: list[int | str | re.Pattern[str]] | str | re.Pattern[str] | None = None,
        stiffness: wp.array[wp.float32] | float | None,
        damping: wp.array[wp.float32] | float | None,
        use_gravity_compensation: bool = True,
        use_coriolis_compensation: bool = True,
        use_inertia_decoupling: bool = True,
        has_qdd_feedforward: bool = False,
    ):
        if not isinstance(model, Model):
            raise TypeError(f"model must be a newton.Model, got {type(model).__name__}.")
        model_robot_count = model.articulation_count
        if model_robot_count < 1:
            raise ValueError("model has no articulations.")

        self._device = model.device
        self._requires_grad = model.requires_grad
        self._use_gravity = bool(use_gravity_compensation)
        self._use_coriolis = bool(use_coriolis_compensation)
        self._use_inertia = bool(use_inertia_decoupling)
        self._has_qdd = bool(has_qdd_feedforward)
        self._needs_fk = self._use_inertia or self._use_gravity or self._use_coriolis
        self._stiffness_is_live = stiffness is None
        self._damping_is_live = damping is None

        self._model = model
        self._model_state = model.state(requires_grad=self._requires_grad)
        self._coord_count = int(model.joint_coord_count)
        self._dof_count = int(model.joint_dof_count)

        joint_selection = select_joints(model, articulations=articulations, joints=joints)
        joint_q_idx = joint_selection.q_start
        joint_qd_idx = joint_selection.qd_start

        # ------------------------------------------------------------------
        # Validation of the two model-space index arrays. Everything else the
        # controller takes is compact and validated by the inner controller.
        # ------------------------------------------------------------------
        # joint_selection.q_start defines the controlled-DOF count, so it is
        # checked against its own length; joint_selection.qd_start must then
        # match that count exactly, which is where a length mismatch between
        # the two is caught. Checked before ``.size`` is read below, since a
        # non-array would otherwise fail there with an unhelpful AttributeError.
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

        # A repeated DOF would give two controlled slots the same simulation DOF,
        # so a scatter through ``control.joint_f[selection.qd_start]`` would have
        # them overwrite each other. Checking joint_selection.qd_start alone
        # covers both arrays, since the pairing check below forces the two to
        # agree.
        if np.unique(qd_idx_np).size != qd_idx_np.size:
            duplicate = int(np.bincount(qd_idx_np).argmax())
            raise ValueError(
                f"joint_selection.qd_start contains DOF {duplicate} more than once; two controlled slots "
                f"cannot map to the same simulation DOF."
            )

        # Each pair (q_start[i], qd_start[i]) must land on the same model joint,
        # which is what catches a joint_selection whose two arrays were built
        # swapped or mismatched.
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

        # A joint is controllable when ``q_des - q`` is a scalar subtraction,
        # i.e. it spans exactly one coordinate and one DOF. Derived from the
        # model's own spans.
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
                f"ControllerJointImpedance only supports controlling joints that span a single "
                f"coordinate and a single DOF; joint_selection addresses unsupported joints: {unsupported}"
            )

        owning_robot = model.joint_articulation.numpy()[owning_joint]
        loose = np.flatnonzero(owning_robot < 0)
        if loose.size:
            raise ValueError(
                f"joint_selection addresses joint {int(owning_joint[loose[0]])}, which belongs to no "
                f"robot. The controller runs forward kinematics and dynamics per robot, so such a "
                f"joint has no mass matrix, gravity, or Coriolis term."
            )

        # Contiguous per-robot grouping is what makes every compact buffer a
        # simple concatenation of per-robot chunks; the mass-matrix block
        # extraction below slices on exactly that assumption.
        if np.any(np.diff(owning_robot) < 0):
            raise ValueError(
                "joint_selection.q_start/qd_start must be grouped by robot (robot 0's DOFs first, "
                f"then robot 1's, ...); got robot order {owning_robot.tolist()}."
            )

        # np.unique lists the controlled robots in ascending model order with
        # their DOF counts, which is exactly the packed layout: every buffer the
        # controller owns is sized to the robots it actually controls, and
        # model_robot_index translates a packed index back to the model's.
        model_robot_index_np, controlled_dofs_per_robot_np = np.unique(owning_robot, return_counts=True)
        model_robot_index_np = model_robot_index_np.astype(np.int32)
        controlled_dofs_per_robot_np = controlled_dofs_per_robot_np.astype(np.int32)
        controlled_robot_count = int(model_robot_index_np.size)
        controlled_dofs_per_robot = wp.array(controlled_dofs_per_robot_np, dtype=wp.int32, device=self._device)
        max_controlled_dofs = int(controlled_dofs_per_robot_np.max())
        self._model_robot_index = wp.array(model_robot_index_np, dtype=wp.int32, device=self._device)
        # FK and dynamics run per robot, and a robot with no controlled DOF
        # contributes nothing to the ones that have them. The mask stays
        # model-sized because that is what the eval_* entry points take.
        mask_np = np.zeros(model_robot_count, dtype=bool)
        mask_np[model_robot_index_np] = True
        self._controlled_robot_mask = wp.array(mask_np, dtype=wp.bool, device=self._device)
        # ------------------------------------------------------------------

        self._model_robot_count = model_robot_count
        self._controlled_robot_count = controlled_robot_count
        self._max_controlled_dofs = max_controlled_dofs
        self._total_controlled_dofs = total_controlled_dofs
        self._controlled_dofs_per_robot = controlled_dofs_per_robot
        # Copied rather than aliased: the robot packing, local-DOF tables, and
        # masks above are derived once from a host snapshot of these arrays, so
        # a caller mutating joint_selection.q_start/qd_start after construction must
        # not change what the live indexed views below read from.
        self._q_idx = wp.clone(joint_q_idx)
        self._qd_idx = wp.clone(joint_qd_idx)

        self._model_mass_matrix: wp.array3d[wp.float32] | None = None
        self._controlled_mass_matrix: wp.array3d[wp.float32] | None = None
        self._local_dof_idx: wp.array2d[wp.int32] | None = None
        self._gravity_flat: wp.array[wp.float32] | None = None
        self._coriolis_flat: wp.array[wp.float32] | None = None

        if self._use_inertia:
            # eval_mass_matrix writes H sized to each articulation's true DOF count
            # (which may exceed its controlled-DOF count, since uncontrolled joints
            # still occupy rows/columns), so the controlled block is extracted each
            # step into a separate (controlled_robot_count, max_controlled_dofs,
            # max_controlled_dofs) buffer.
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
        if self._use_gravity:
            self._gravity_flat = wp.zeros(
                self._dof_count, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
            )
        if self._use_coriolis:
            self._coriolis_flat = wp.zeros(
                self._dof_count, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
            )

        self._model_free = ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=controlled_dofs_per_robot,
            stiffness=stiffness,
            damping=damping,
            use_gravity_compensation=use_gravity_compensation,
            use_coriolis_compensation=use_coriolis_compensation,
            use_inertia_decoupling=use_inertia_decoupling,
            has_qdd_feedforward=has_qdd_feedforward,
            device=self._device,
            requires_grad=self._requires_grad,
        )

        # Pre-wired dynamics fields forwarded to ModelFree each step. These are
        # live indexed views of the whole-model buffers, so the inner
        # controller reads the current contents without an index table of its
        # own — including on graph replay.
        self._mf_input = ControllerJointImpedanceModelFree.Inputs()
        self._mf_input.joint_q = self._model_state.joint_q[self._q_idx]
        self._mf_input.joint_qd = self._model_state.joint_qd[self._qd_idx]
        if self._use_inertia:
            self._mf_input.mass_matrix = self._controlled_mass_matrix
        if self._use_gravity:
            self._mf_input.gravity_force = self._gravity_flat[self._qd_idx]
        if self._use_coriolis:
            self._mf_input.coriolis_force = self._coriolis_flat[self._qd_idx]

    def _compute_local_dof_idx(
        self, *, qd_idx_np: np.ndarray, model_robot_index_np: np.ndarray, controlled_dofs_per_robot_np: np.ndarray
    ) -> np.ndarray:
        """Return, for each (controlled robot, padded slot), the DOF's index within that robot.

        ``joint_selection.qd_start`` is in the model's DOF numbering, but
        :func:`~newton.eval_mass_matrix` indexes each robot's block by
        DOF-within-that-robot, so the two differ by where the robot's DOFs start
        in the model.
        """
        robot_joint_start = self._model.articulation_start.numpy()
        robot_dof_start = self._model.joint_qd_start.numpy()[robot_joint_start[model_robot_index_np]]

        controlled_robot_count = int(model_robot_index_np.size)
        offsets = np.zeros(controlled_robot_count, dtype=np.int64)
        offsets[1:] = np.cumsum(controlled_dofs_per_robot_np[:-1])

        local_dof_idx = np.zeros((controlled_robot_count, self._max_controlled_dofs), dtype=np.int32)
        for robot in range(controlled_robot_count):
            n = int(controlled_dofs_per_robot_np[robot])
            chunk = qd_idx_np[offsets[robot] : offsets[robot] + n]
            local_dof_idx[robot, :n] = chunk - robot_dof_start[robot]
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
        """Model coordinate index of each controlled joint, shape [total_controlled_dofs].

        Use to gather or scatter a compact port against a simulation-sized
        coordinate array, e.g. ``model.joint_q[controller.q_start]``.
        """
        return self._q_idx

    @property
    def qd_start(self) -> wp.array[wp.int32]:
        """Model DOF index of each controlled joint, shape [total_controlled_dofs].

        Use to scatter a compact port into a simulation-sized array, e.g.
        ``control.joint_f[controller.qd_start]``.
        """
        return self._qd_idx

    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def is_graphable(self) -> bool:
        return True

    def input(self) -> Inputs:
        """Return a pre-allocated :class:`Inputs` without dynamics fields."""
        d, rg, n = self._device, self._requires_grad, self._total_controlled_dofs

        def _compact(enabled: bool) -> wp.array[wp.float32] | None:
            return wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg) if enabled else None

        inputs = ControllerJointImpedance.Inputs()
        inputs.joint_q = wp.zeros(self._coord_count, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd = wp.zeros(self._dof_count, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_q_des = _compact(True)
        inputs.joint_qd_des = _compact(True)
        inputs.joint_qdd = _compact(self._has_qdd)
        inputs.stiffness = _compact(self._stiffness_is_live)
        inputs.damping = _compact(self._damping_is_live)
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with a compact torque array."""
        outputs = ControllerJointImpedance.Outputs()
        outputs.joint_f = self._model_free.output().joint_f
        return outputs

    def step(
        self,
        *,
        inputs: Inputs,
        outputs: Outputs,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Run one impedance-control step.

        Args:
            inputs: Populated :class:`Inputs` struct. Dynamics terms are
                computed internally from the Newton model.
            outputs: :class:`Outputs` struct to write torques into.
            dt: Unused. Accepted for API compatibility.
        """
        # Checked here because the copies below consume these two ports before
        # the inner controller (which validates the rest) ever sees them.
        for port, name, length in (
            (inputs.joint_q, "inputs.joint_q", self._coord_count),
            (inputs.joint_qd, "inputs.joint_qd", self._dof_count),
        ):
            _validate_array(
                array=port,
                name=name,
                dtype=wp.float32,
                shape=(length,),
                device=self._device,
                allow_indexed=True,
            )

        # A port belonging to a disabled feature is never forwarded to the inner
        # controller, so writing one would go unnoticed. getattr because a
        # caller may leave the field unset rather than None.
        for name, enabled, switch in (
            ("joint_qdd", self._has_qdd, "has_qdd_feedforward"),
            ("stiffness", self._stiffness_is_live, "a live stiffness"),
            ("damping", self._damping_is_live, "a live damping"),
        ):
            if not enabled and getattr(inputs, name, None) is not None:
                raise ValueError(
                    f"inputs.{name} is set, but the controller was built without {switch}, so the value "
                    f"would be ignored."
                )

        # Whole-model reads, not a gather of the controlled DOFs: an
        # uncontrolled joint still sets its own body transform, and hence the
        # gravity/Coriolis/mass-matrix terms of every joint downstream of it.
        _read_port(inputs.joint_q, self._model_state.joint_q, self._coord_count, self._device)
        _read_port(inputs.joint_qd, self._model_state.joint_qd, self._dof_count, self._device)

        if self._needs_fk:
            eval_fk(
                self._model,
                self._model_state.joint_q,
                self._model_state.joint_qd,
                self._model_state,
                mask=self._controlled_robot_mask,
            )
        if self._use_inertia:
            eval_mass_matrix(
                self._model, self._model_state, H=self._model_mass_matrix, mask=self._controlled_robot_mask
            )
            wp.launch(
                _gather_mass_matrix_blocks_kernel,
                dim=(self._controlled_robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                inputs=[
                    self._model_mass_matrix,
                    self._model_robot_index,
                    self._local_dof_idx,
                    self._controlled_dofs_per_robot,
                ],
                outputs=[self._controlled_mass_matrix],
                device=self._device,
            )
        if self._use_gravity or self._use_coriolis:
            eval_inverse_dynamics_passive(
                self._model,
                self._model_state,
                gravity_force=self._gravity_flat,
                coriolis_force=self._coriolis_flat,
                mask=self._controlled_robot_mask,
            )

        self._mf_input.joint_q_des = inputs.joint_q_des
        self._mf_input.joint_qd_des = inputs.joint_qd_des
        if self._has_qdd:
            self._mf_input.joint_qdd = inputs.joint_qdd
        if self._stiffness_is_live:
            self._mf_input.stiffness = inputs.stiffness
        if self._damping_is_live:
            self._mf_input.damping = inputs.damping

        self._model_free.step(inputs=self._mf_input, outputs=outputs, dt=dt)
