# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerJointImpedance — joint-space impedance control with
Newton model-internal dynamics.

Calls :func:`newton.eval_fk` and :func:`newton.eval_mass_matrix` on the
supplied model each step to obtain the mass matrix, then delegates all
compute work to an inner :class:`ControllerJointImpedanceModelFree` instance.

Gravity and Coriolis compensation use :func:`newton.eval_inverse_dynamics_passive`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from newton import JointType
from newton._src.sim.articulation import eval_fk, eval_mass_matrix
from newton._src.sim.inverse_dynamics import eval_inverse_dynamics_passive
from newton._src.sim.model import Model

from ...controller import ControllerBase
from ...utils import _validate_array
from ._common import _gather_mass_matrix_blocks_kernel
from .model_free import ControllerJointImpedanceModelFree

# Joints whose position error ``q_des - q`` is a well-defined scalar subtraction.
_SCALAR_JOINT_TYPES = (int(JointType.REVOLUTE), int(JointType.PRISMATIC))


class ControllerJointImpedance(ControllerBase):
    """Joint-space impedance controller with internally computed dynamics.

    Implements the joint-space impedance control law. This model-based variant
    computes the mass matrix, gravity, and Coriolis terms itself: it evaluates
    forward kinematics and the enabled dynamics terms from ``model`` on every
    :meth:`step`, so the caller supplies only joint positions and velocities.

    ``model`` is borrowed, not owned — it is never written to, and changes to
    it are visible to the controller immediately.

    **Index arrays.** ``joint_q_idx`` and ``joint_qd_idx`` are the only
    arguments that refer to the model, and they exist for one reason: the
    controller runs FK and dynamics over the whole model and has to know where
    the controlled DOFs live in it. They are not part of the control interface.
    The two spaces are distinct and must not be interchanged — ``joint_q_idx``
    indexes :attr:`newton.State.joint_q` (coordinates, via
    :attr:`~newton.Model.joint_q_start`), ``joint_qd_idx`` indexes
    :attr:`newton.State.joint_qd` (DOFs, via
    :attr:`~newton.Model.joint_qd_start`). They coincide only when the model
    contains no ball, free, or distance joint. Use
    :func:`~newton.controllers.select_joints` to build a matched pair.

    **Ports.** ``inputs.joint_q`` and ``inputs.joint_qd`` are whole-model
    arrays, since gravity, Coriolis, and mass-matrix terms depend on the state
    of the entire model, not only the controlled joints: a revolute joint
    mounted on an unactuated floating base still needs the base transform to
    compute the right gravity compensation. Every other port is **compact** —
    one entry per controlled DOF, robot 0's DOFs first, then robot 1's. A
    compact port may be bound to a plain array or to an indexed view of a
    simulation-sized array::

        outputs.joint_f = control.joint_f[selection.qd_idx]  # scatter to the sim

    One robot is derived per articulation in ``model``. Only Revolute and
    Prismatic joints can be controlled, since the PD error term ``q_des - q``
    is only a well-defined scalar subtraction for a single-coordinate joint;
    every other joint (Fixed, or any multi-DOF type) is read for FK and
    dynamics but never actuated. Addressing a non-scalar joint raises
    ``ValueError`` at construction.

    Supports heterogeneous robot fleets — robots in the batch may have
    different controlled-DOF counts, including zero for an articulation that is
    present in the model but not controlled.

    See also :class:`ControllerJointImpedanceModelFree`, which takes the mass
    matrix, gravity, and Coriolis terms as inputs instead of computing them
    from a :class:`~newton.Model`.

    Impedance law (terms enabled at construction):

        τ = [M(q) if use_inertia_decoupling else I] · (q̈_des + Kp·Δq + Kd·Δq̇)
            + [C(q,q̇)·q̇ if use_coriolis_compensation else 0]
            + [g(q)      if use_gravity_compensation  else 0]

    Args:
        model: :class:`~newton.Model` with N articulations (one per robot).
            Articulations may mix controlled 1-DOF joints with uncontrolled
            joints of any type.
        joint_q_idx: Model **coordinate** index of each controlled DOF, shape
            ``(total_dofs,)``, grouped by articulation. Typically
            ``select_joints(...).q_idx``.
        joint_qd_idx: Model **DOF** index of each controlled DOF, same length
            and ordering as ``joint_q_idx``. Typically
            ``select_joints(...).qd_idx``.
        stiffness: Position-error gain Kp, shape ``(total_dofs,)``. Units
            depend on ``use_inertia_decoupling``: [1/s²] when enabled, since
            the PD term is then an acceleration premultiplied by M(q);
            otherwise [N/m or N·m/rad]. Pass an array to copy it at
            construction, or ``None`` to read ``inputs.stiffness`` each step.
        damping: Velocity-error gain Kd, [1/s] when
            ``use_inertia_decoupling`` is enabled, otherwise
            [N·s/m or N·m·s/rad]. Same format as ``stiffness``.
        use_gravity_compensation: Add gravity generalized forces to τ.
        use_coriolis_compensation: Add Coriolis generalized forces to τ.
        use_inertia_decoupling: Premultiply the PD term by M(q).
        has_qdd_feedforward: Accept a desired-acceleration feedforward via
            ``inputs.joint_qdd``.
        device: Warp device. Must match ``model.device``.
        requires_grad: Whether internal buffers need gradient support. Must
            match ``model.requires_grad``.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerJointImpedance.input`.

        Dynamics fields (mass matrix, gravity, Coriolis) are computed
        internally and do not appear here.
        """

        joint_q: wp.array[wp.float32]
        """Current joint positions [m or rad], shape ``(model.joint_coord_count,)``."""
        joint_qd: wp.array[wp.float32]
        """Current joint velocities [m/s or rad/s], shape ``(model.joint_dof_count,)``."""
        joint_q_des: wp.array[wp.float32]
        """Desired joint positions [m or rad], shape ``(total_dofs,)``."""
        joint_qd_des: wp.array[wp.float32]
        """Desired joint velocities [m/s or rad/s], shape ``(total_dofs,)``."""
        joint_qdd: wp.array[wp.float32] | None
        """Desired acceleration feedforward [m/s² or rad/s²], shape ``(total_dofs,)``. ``None`` unless ``has_qdd_feedforward=True``."""
        stiffness: wp.array[wp.float32] | None
        """Position-error gain Kp, shape ``(total_dofs,)``. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m or N·m/rad]. ``None`` when gains are baked at construction."""
        damping: wp.array[wp.float32] | None
        """Velocity-error gain Kd, shape ``(total_dofs,)``. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m or N·m·s/rad]. ``None`` when gains are baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerJointImpedance.output`."""

        joint_f: wp.array[wp.float32]
        """Joint torque command [N or N·m], shape ``(total_dofs,)``."""

    def __init__(
        self,
        model: Model,
        *,
        joint_q_idx: wp.array[wp.int32],
        joint_qd_idx: wp.array[wp.int32],
        stiffness: wp.array[wp.float32] | None,
        damping: wp.array[wp.float32] | None,
        use_gravity_compensation: bool = True,
        use_coriolis_compensation: bool = True,
        use_inertia_decoupling: bool = True,
        has_qdd_feedforward: bool = False,
        device: Any = None,
        requires_grad: bool = False,
    ):
        if not isinstance(model, Model):
            raise TypeError(f"model must be a newton.Model, got {type(model).__name__}.")
        robot_count = model.articulation_count
        if robot_count < 1:
            raise ValueError("model has no articulations.")

        self._device = wp.get_device(device)
        if model.device != self._device:
            raise ValueError(f"model.device is {model.device}, but device resolves to {self._device}.")
        if model.requires_grad != requires_grad:
            raise ValueError(f"model.requires_grad is {model.requires_grad}, and requires_grad is {requires_grad}")

        self._requires_grad = requires_grad
        self._use_gravity = bool(use_gravity_compensation)
        self._use_coriolis = bool(use_coriolis_compensation)
        self._use_inertia = bool(use_inertia_decoupling)
        self._has_qdd = bool(has_qdd_feedforward)
        self._needs_fk = self._use_inertia or self._use_gravity or self._use_coriolis
        self._stiffness_is_live = stiffness is None
        self._damping_is_live = damping is None

        self._model = model
        self._model_state = model.state(requires_grad=requires_grad)
        self._coord_count = int(model.joint_coord_count)
        self._dof_count = int(model.joint_dof_count)

        # ------------------------------------------------------------------
        # Validation of the two model-space index arrays. Everything else the
        # controller takes is compact and validated by the inner controller.
        # ------------------------------------------------------------------
        # joint_q_idx defines the controlled-DOF count, so it is checked against
        # its own length; joint_qd_idx must then match that count exactly, which
        # is where a length mismatch between the two is caught.
        if not isinstance(joint_q_idx, wp.array):
            raise TypeError(f"joint_q_idx must be a wp.array, got {type(joint_q_idx).__name__}.")
        _validate_array(
            array=joint_q_idx, name="joint_q_idx", dtype=wp.int32, shape=(joint_q_idx.size,), device=self._device
        )
        total_dofs = int(joint_q_idx.size)
        if total_dofs < 1:
            raise ValueError("joint_q_idx is empty; there is nothing to control.")
        _validate_array(
            array=joint_qd_idx, name="joint_qd_idx", dtype=wp.int32, shape=(total_dofs,), device=self._device
        )

        q_idx_np = joint_q_idx.numpy()
        qd_idx_np = joint_qd_idx.numpy()
        for name, idx_np, limit, space in (
            ("joint_q_idx", q_idx_np, self._coord_count, "coordinate"),
            ("joint_qd_idx", qd_idx_np, self._dof_count, "DOF"),
        ):
            if idx_np.min() < 0 or idx_np.max() >= limit:
                raise ValueError(
                    f"{name} must index the model's {space} space [0, {limit}), got "
                    f"range [{int(idx_np.min())}, {int(idx_np.max())}]."
                )

        # Each pair (q_idx[i], qd_idx[i]) must land on the same model joint,
        # which is what catches a caller passing the two arrays swapped or
        # mismatched.
        owning_joint = np.searchsorted(model.joint_q_start.numpy(), q_idx_np, side="right") - 1
        owning_joint_qd = np.searchsorted(model.joint_qd_start.numpy(), qd_idx_np, side="right") - 1
        if not np.array_equal(owning_joint, owning_joint_qd):
            mismatched = int(np.flatnonzero(owning_joint != owning_joint_qd)[0])
            raise ValueError(
                f"joint_q_idx and joint_qd_idx disagree at entry {mismatched}: coordinate "
                f"{int(q_idx_np[mismatched])} belongs to joint {int(owning_joint[mismatched])} but DOF "
                f"{int(qd_idx_np[mismatched])} belongs to joint {int(owning_joint_qd[mismatched])}. "
                f"Did you swap the two arrays?"
            )

        joint_type_np = model.joint_type.numpy()
        unsupported = sorted(
            {
                (int(j), JointType(joint_type_np[j]).name)
                for j in owning_joint
                if joint_type_np[j] not in _SCALAR_JOINT_TYPES
            }
        )
        if unsupported:
            raise ValueError(
                f"ControllerJointImpedance only supports controlling 1-DOF joints "
                f"(Revolute/Prismatic); the index arrays address unsupported joints: {unsupported}"
            )

        # Contiguous per-robot grouping is what makes every compact buffer a
        # simple concatenation of per-robot chunks; the mass-matrix block
        # extraction below slices on exactly that assumption.
        owning_articulation = np.searchsorted(model.articulation_start.numpy(), owning_joint, side="right") - 1
        if np.any(np.diff(owning_articulation) < 0):
            raise ValueError(
                "joint_q_idx/joint_qd_idx must be grouped by articulation (robot 0's DOFs first, "
                f"then robot 1's, ...); got articulation order {owning_articulation.tolist()}."
            )

        dofs_per_robot_np = np.zeros(robot_count, dtype=np.int32)
        unique_arts, counts = np.unique(owning_articulation, return_counts=True)
        dofs_per_robot_np[unique_arts] = counts
        dofs_per_robot = wp.array(dofs_per_robot_np, dtype=wp.int32, device=self._device)
        max_dofs = int(dofs_per_robot_np.max())
        # ------------------------------------------------------------------

        self._robot_count = robot_count
        self._max_dofs = max_dofs
        self._total_dofs = total_dofs
        self._dofs_per_robot = dofs_per_robot
        self._q_idx = joint_q_idx
        self._qd_idx = joint_qd_idx

        self._mass_matrix_full: wp.array3d[wp.float32] | None = None
        self._mass_matrix: wp.array3d[wp.float32] | None = None
        self._local_dof_idx: wp.array2d[wp.int32] | None = None
        self._gravity_flat: wp.array[wp.float32] | None = None
        self._coriolis_flat: wp.array[wp.float32] | None = None

        if self._use_inertia:
            # eval_mass_matrix writes H sized to each articulation's true DOF count
            # (which may exceed its controlled-DOF count, since uncontrolled joints
            # still occupy rows/columns), so the controlled block is extracted each
            # step into a separate (robot_count, max_dofs, max_dofs) buffer.
            model_max_dofs = model.max_dofs_per_articulation
            self._mass_matrix_full = wp.zeros(
                (robot_count, model_max_dofs, model_max_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._mass_matrix = wp.zeros(
                (robot_count, max_dofs, max_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._local_dof_idx = wp.array(
                self._compute_local_dof_idx(qd_idx_np=qd_idx_np, dofs_per_robot_np=dofs_per_robot_np),
                dtype=wp.int32,
                device=self._device,
            )
        if self._use_gravity:
            self._gravity_flat = wp.zeros(
                self._dof_count, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
        if self._use_coriolis:
            self._coriolis_flat = wp.zeros(
                self._dof_count, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )

        self._model_free = ControllerJointImpedanceModelFree(
            dofs_per_robot=dofs_per_robot,
            stiffness=stiffness,
            damping=damping,
            use_gravity_compensation=use_gravity_compensation,
            use_coriolis_compensation=use_coriolis_compensation,
            use_inertia_decoupling=use_inertia_decoupling,
            has_qdd_feedforward=has_qdd_feedforward,
            device=self._device,
            requires_grad=requires_grad,
        )

        # Pre-wired dynamics fields forwarded to ModelFree each step. These are
        # live indexed views of the whole-model buffers, so the inner
        # controller reads the current contents without an index table of its
        # own — including on graph replay.
        self._mf_input = ControllerJointImpedanceModelFree.Inputs()
        self._mf_input.joint_q = self._model_state.joint_q[self._q_idx]
        self._mf_input.joint_qd = self._model_state.joint_qd[self._qd_idx]
        if self._use_inertia:
            self._mf_input.mass_matrix = self._mass_matrix
        if self._use_gravity:
            self._mf_input.gravity_force = self._gravity_flat[self._qd_idx]
        if self._use_coriolis:
            self._mf_input.coriolis_force = self._coriolis_flat[self._qd_idx]

    def _compute_local_dof_idx(self, *, qd_idx_np: np.ndarray, dofs_per_robot_np: np.ndarray) -> np.ndarray:
        """Return, for each (robot, padded slot), the controlled DOF's index within its articulation.

        Used to gather the controlled-DOF block out of the per-articulation mass
        matrix, which :func:`~newton.eval_mass_matrix` sizes to each
        articulation's true DOF count rather than its controlled-DOF count.
        """
        art_start = self._model.articulation_start.numpy()
        articulation_dof_start = self._model.joint_qd_start.numpy()[art_start[: self._robot_count]]

        offsets = np.zeros(self._robot_count, dtype=np.int64)
        offsets[1:] = np.cumsum(dofs_per_robot_np[:-1])

        local_dof_idx = np.zeros((self._robot_count, self._max_dofs), dtype=np.int32)
        for robot in range(self._robot_count):
            n = int(dofs_per_robot_np[robot])
            chunk = qd_idx_np[offsets[robot] : offsets[robot] + n]
            local_dof_idx[robot, :n] = chunk - articulation_dof_start[robot]
        return local_dof_idx

    @property
    def robot_count(self) -> int:
        return self._robot_count

    @property
    def max_dofs(self) -> int:
        return self._max_dofs

    @property
    def total_dofs(self) -> int:
        return self._total_dofs

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
        d, rg, n = self._device, self._requires_grad, self._total_dofs

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
        # the inner controller — which validates the rest — ever sees them.
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

        # Whole-model copies, not a gather of the controlled DOFs: an
        # uncontrolled joint still sets its own body transform, and hence the
        # gravity/Coriolis/mass-matrix terms of every joint downstream of it.
        wp.copy(self._model_state.joint_q, inputs.joint_q)
        wp.copy(self._model_state.joint_qd, inputs.joint_qd)

        if self._needs_fk:
            eval_fk(self._model, self._model_state.joint_q, self._model_state.joint_qd, self._model_state)
        if self._use_inertia:
            eval_mass_matrix(self._model, self._model_state, H=self._mass_matrix_full)
            wp.launch(
                _gather_mass_matrix_blocks_kernel,
                dim=(self._robot_count, self._max_dofs, self._max_dofs),
                inputs=[self._mass_matrix_full, self._local_dof_idx, self._dofs_per_robot],
                outputs=[self._mass_matrix],
                device=self._device,
            )
        if self._use_gravity or self._use_coriolis:
            eval_inverse_dynamics_passive(
                self._model,
                self._model_state,
                gravity_force=self._gravity_flat,
                coriolis_force=self._coriolis_flat,
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
