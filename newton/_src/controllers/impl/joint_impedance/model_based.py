# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerJointImpedance — joint-space impedance control with
Newton model-internal dynamics.

Internally calls :func:`newton.eval_fk` and :func:`newton.eval_mass_matrix`
each step to obtain the mass matrix, then delegates all gather/compute/scatter
work to an inner :class:`ControllerJointImpedanceModelFree` instance.

Gravity and Coriolis compensation use :func:`newton.eval_inverse_dynamics_passive`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from newton import JointType
from newton._src.sim.articulation import eval_fk, eval_mass_matrix
from newton._src.sim.builder import ModelBuilder
from newton._src.sim.inverse_dynamics import eval_inverse_dynamics_passive

from ...controller import ControllerBase
from ...utils import _normalize_indices
from ._common import _gather_dof_flat_kernel, _idx_max
from .model_free import ControllerJointImpedanceModelFree


class ControllerJointImpedance(ControllerBase):
    """One-step joint-space impedance controller for a batch of robots.

    Has an identical input/output interface to
    :class:`ControllerJointImpedanceModelFree` — flat 1D sim arrays in,
    flat 1D torque array out — except that the dynamics terms (mass matrix,
    gravity force, Coriolis force) are computed internally from the Newton
    model rather than supplied by the caller.

    Supports heterogeneous robot fleets — robots in the batch may have
    different DOF counts. The ``builder`` articulations define the
    per-robot topology; the controller pads internal buffers to
    ``model.max_dofs_per_articulation`` and skips padding slots in all kernels.

    Only 1-DOF joints (Revolute, Prismatic) and zero-DOF Fixed joints are
    supported. The PD error term ``q_des - q`` is only valid for scalar
    joint coordinates.

    Impedance law (terms enabled at construction):

        τ = [M(q) if use_inertia_decoupling else I] · (q̈_des + Kp·Δq + Kd·Δq̇)
            + [C(q,q̇)·q̇ if use_coriolis_compensation else 0]
            + [g(q)      if use_gravity_compensation  else 0]

    Args:
        builder: :class:`~newton.ModelBuilder` with N articulations (one
            per robot). Articulations may have different DOF counts.
        default_dof_indices: Concatenated per-robot index arrays of length
            ``sum(dofs per articulation)`` mapping controller DOF slots to
            positions in the flat simulation arrays (robot 0's indices first,
            then robot 1's, etc.).
        stiffness: Position-error gain Kp [N/m or N·m/rad], shape
            ``(N, max_dofs)``. Pass a baked array or ``None`` to read from
            ``inputs.stiffness`` each step.
        damping: Velocity-error gain Kd [N·s/m or N·m·s/rad]. Same format
            as ``stiffness``.
        use_gravity_compensation: Add gravity generalized forces to τ.
        use_coriolis_compensation: Add Coriolis generalized forces to τ.
        use_inertia_decoupling: Premultiply the PD term by M(q).
        has_qdd_feedforward: Accept a desired-acceleration feedforward via
            ``inputs.joint_qdd``.
        joint_q_idx: Optional index array (same length as
            ``default_dof_indices``) overriding it for the position read.
        joint_qd_idx: Optional index array for velocity read.
        joint_q_des_idx: Optional index array for desired position read.
        joint_qd_des_idx: Optional index array for desired velocity read.
        joint_qdd_idx: Optional index array for feedforward read.
        joint_f_idx: Optional index array overriding ``default_dof_indices``
            for the torque-output write.
        device: Warp device.
        requires_grad: Whether internal buffers need gradient support.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerJointImpedance.input`.

        Dynamics fields (mass matrix, gravity, Coriolis) are computed
        internally and do not appear here.
        """

        joint_q: wp.array[wp.float32]
        """Current joint positions [m or rad], flat sim-level array."""
        joint_qd: wp.array[wp.float32]
        """Current joint velocities [m/s or rad/s], flat sim-level array."""
        joint_q_des: wp.array[wp.float32]
        """Desired joint positions [m or rad], flat sim-level array."""
        joint_qd_des: wp.array[wp.float32]
        """Desired joint velocities [m/s or rad/s], flat sim-level array."""
        joint_qdd: wp.array[wp.float32] | None
        """Desired acceleration feedforward [m/s² or rad/s²], flat sim-level array. ``None`` unless ``has_qdd_feedforward=True``."""
        stiffness: wp.array2d[wp.float32] | None
        """Position-error gain Kp [N/m or N·m/rad], shape ``(robot_count, max_dofs)``. ``None`` when gains are baked at construction."""
        damping: wp.array2d[wp.float32] | None
        """Velocity-error gain Kd [N·s/m or N·m·s/rad], shape ``(robot_count, max_dofs)``. ``None`` when gains are baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerJointImpedance.output`."""

        joint_f: wp.array[wp.float32]
        """Joint torque command [N or N·m], flat sim-level array."""

    def __init__(
        self,
        builder: ModelBuilder,
        *,
        default_dof_indices: wp.array[wp.uint32],
        stiffness: wp.array2d[wp.float32] | None,
        damping: wp.array2d[wp.float32] | None,
        use_gravity_compensation: bool = True,
        use_coriolis_compensation: bool = True,
        use_inertia_decoupling: bool = True,
        has_qdd_feedforward: bool = False,
        joint_q_idx: wp.array[wp.uint32] | None = None,
        joint_qd_idx: wp.array[wp.uint32] | None = None,
        joint_q_des_idx: wp.array[wp.uint32] | None = None,
        joint_qd_des_idx: wp.array[wp.uint32] | None = None,
        joint_qdd_idx: wp.array[wp.uint32] | None = None,
        joint_f_idx: wp.array[wp.uint32] | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        if not isinstance(builder, ModelBuilder):
            raise TypeError(f"builder must be a newton.ModelBuilder, got {type(builder).__name__}.")
        if not isinstance(default_dof_indices, wp.array) or default_dof_indices.dtype != wp.uint32:
            raise TypeError("default_dof_indices must be wp.array[uint32].")

        robot_count = builder.articulation_count
        if robot_count < 1:
            raise ValueError("builder has no articulations.")

        # Fixed joints contribute zero DOFs and are invisible to the PD error term.
        allowed_joint_types = {int(JointType.REVOLUTE), int(JointType.PRISMATIC), int(JointType.FIXED)}
        unsupported_joints = [
            (joint_index, JointType(joint_type).name)
            for joint_index, joint_type in enumerate(builder.joint_type)
            if joint_type not in allowed_joint_types
        ]
        if unsupported_joints:
            raise ValueError(
                f"ControllerJointImpedance only supports 1-DOF joints (Revolute/Prismatic) and "
                f"zero-DOF fixed joints; found unsupported joint types: {unsupported_joints}"
            )

        self._device = device if device is not None else wp.get_device()
        self._requires_grad = requires_grad
        self._use_gravity = bool(use_gravity_compensation)
        self._use_coriolis = bool(use_coriolis_compensation)
        self._use_inertia = bool(use_inertia_decoupling)
        self._has_qdd = bool(has_qdd_feedforward)
        self._needs_fk = self._use_inertia or self._use_gravity or self._use_coriolis
        self._stiffness_is_live = stiffness is None
        self._damping_is_live = damping is None

        self._model = builder.finalize(device=self._device, requires_grad=requires_grad)
        self._model_state = self._model.state()

        max_dofs = self._model.max_dofs_per_articulation

        # Derive per-articulation DOF counts from the finalized model.
        art_start = self._model.articulation_start.numpy()
        art_end = self._model.articulation_end.numpy()
        joint_q_start = self._model.joint_q_start.numpy()
        dofs_per_robot_np = np.array(
            [joint_q_start[art_end[i]] - joint_q_start[art_start[i]] for i in range(robot_count)],
            dtype=np.int32,
        )
        dofs_per_robot = wp.array(dofs_per_robot_np, dtype=wp.int32, device=self._device)
        total_dofs = int(dofs_per_robot_np.sum())

        if int(default_dof_indices.size) != total_dofs:
            raise ValueError(
                f"default_dof_indices length {default_dof_indices.size} must equal "
                f"sum of per-robot DOF counts = {total_dofs}."
            )

        self._robot_count = robot_count
        self._max_dofs = max_dofs
        self._total_dofs = total_dofs

        self._q_idx = _normalize_indices(joint_q_idx, default_dof_indices, name="joint_q")
        self._qd_idx = _normalize_indices(joint_qd_idx, default_dof_indices, name="joint_qd")
        self._q_des_idx = _normalize_indices(joint_q_des_idx, default_dof_indices, name="joint_q_des")
        self._qd_des_idx = _normalize_indices(joint_qd_des_idx, default_dof_indices, name="joint_qd_des")
        self._qdd_idx = _normalize_indices(joint_qdd_idx, default_dof_indices, name="joint_qdd")

        self._mass_matrix: wp.array3d[wp.float32] | None = None
        self._gravity_flat: wp.array[wp.float32] | None = None
        self._coriolis_flat: wp.array[wp.float32] | None = None

        if self._use_inertia:
            self._mass_matrix = wp.zeros(
                (robot_count, max_dofs, max_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
        if self._use_gravity:
            self._gravity_flat = wp.zeros(
                total_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
        if self._use_coriolis:
            self._coriolis_flat = wp.zeros(
                total_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )

        # Newton fills dynamics in the same DOF order as model.joint_q — use identity indices.
        identity_idx = wp.array(np.arange(total_dofs, dtype=np.uint32), device=self._device)

        self._model_free = ControllerJointImpedanceModelFree(
            robot_count=robot_count,
            dofs_per_robot=dofs_per_robot,
            max_dofs=max_dofs,
            default_dof_indices=default_dof_indices,
            stiffness=stiffness,
            damping=damping,
            use_gravity_compensation=use_gravity_compensation,
            use_coriolis_compensation=use_coriolis_compensation,
            use_inertia_decoupling=use_inertia_decoupling,
            has_qdd_feedforward=has_qdd_feedforward,
            joint_q_idx=joint_q_idx,
            joint_qd_idx=joint_qd_idx,
            joint_q_des_idx=joint_q_des_idx,
            joint_qd_des_idx=joint_qd_des_idx,
            joint_qdd_idx=joint_qdd_idx,
            gravity_force_idx=identity_idx,
            coriolis_force_idx=identity_idx,
            joint_f_idx=joint_f_idx,
            device=device,
            requires_grad=requires_grad,
        )

        # Pre-wired dynamics fields forwarded to ModelFree each step.
        self._mf_input = ControllerJointImpedanceModelFree.Inputs()
        if self._use_inertia:
            self._mf_input.mass_matrix = self._mass_matrix
        if self._use_gravity:
            self._mf_input.gravity_force = self._gravity_flat
        if self._use_coriolis:
            self._mf_input.coriolis_force = self._coriolis_flat

    @property
    def robot_count(self) -> int:
        return self._robot_count

    @property
    def max_dofs(self) -> int:
        return self._max_dofs

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
        d, rg = self._device, self._requires_grad
        inputs = ControllerJointImpedance.Inputs()
        inputs.joint_q = wp.zeros(_idx_max(self._q_idx), dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd = wp.zeros(_idx_max(self._qd_idx), dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_q_des = wp.zeros(_idx_max(self._q_des_idx), dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd_des = wp.zeros(_idx_max(self._qd_des_idx), dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qdd = (
            wp.zeros(_idx_max(self._qdd_idx), dtype=wp.float32, device=d, requires_grad=rg) if self._has_qdd else None
        )
        shape_2d = (self._robot_count, self._max_dofs)
        inputs.stiffness = (
            wp.zeros(shape_2d, dtype=wp.float32, device=d, requires_grad=rg) if self._stiffness_is_live else None
        )
        inputs.damping = (
            wp.zeros(shape_2d, dtype=wp.float32, device=d, requires_grad=rg) if self._damping_is_live else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with a flat torque array."""
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
        wp.launch(
            _gather_dof_flat_kernel,
            dim=self._total_dofs,
            inputs=[inputs.joint_q, self._q_idx],
            outputs=[self._model_state.joint_q],
            device=self._device,
        )
        wp.launch(
            _gather_dof_flat_kernel,
            dim=self._total_dofs,
            inputs=[inputs.joint_qd, self._qd_idx],
            outputs=[self._model_state.joint_qd],
            device=self._device,
        )

        if self._needs_fk:
            eval_fk(self._model, self._model_state.joint_q, self._model_state.joint_qd, self._model_state)
        if self._use_inertia:
            eval_mass_matrix(self._model, self._model_state, H=self._mass_matrix)
        if self._use_gravity or self._use_coriolis:
            eval_inverse_dynamics_passive(
                self._model,
                self._model_state,
                gravity_force=self._gravity_flat,
                coriolis_force=self._coriolis_flat,
            )

        self._mf_input.joint_q = inputs.joint_q
        self._mf_input.joint_qd = inputs.joint_qd
        self._mf_input.joint_q_des = inputs.joint_q_des
        self._mf_input.joint_qd_des = inputs.joint_qd_des
        if self._has_qdd:
            self._mf_input.joint_qdd = inputs.joint_qdd
        if self._stiffness_is_live:
            self._mf_input.stiffness = inputs.stiffness
        if self._damping_is_live:
            self._mf_input.damping = inputs.damping

        self._model_free.step(inputs=self._mf_input, outputs=outputs, dt=dt)
