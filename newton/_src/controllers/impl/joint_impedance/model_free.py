# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerJointImpedanceModelFree — joint-space impedance control with
caller-supplied dynamics terms.

The controller reads flat 1D sim arrays, gathers the controlled DOFs internally,
computes a joint torque command, and scatters the result back to a flat 1D
output array.

The difference from :class:`ControllerJointImpedance` is that this controller
requires the caller to supply dynamics terms (mass matrix, gravity force, Coriolis
force) that the model-based controller computes internally.

Impedance law (terms enabled at construction):

    τ = [M(q) if use_inertia_decoupling else I] · (q̈_des + Kp·Δq + Kd·Δq̇)
        + [C(q,q̇)·q̇ if use_coriolis_compensation else 0]
        + [g(q)      if use_gravity_compensation  else 0]

where Δq = q_des - q and Δq̇ = q̇_des - q̇.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ...controller import ControllerBase
from ...utils import _normalize_indices
from ._common import (
    _add_term_kernel,
    _gather_dof_kernel,
    _idx_max,
    _mass_matrix_multiply_kernel,
    _pd_term_kernel,
    _scatter_dof_kernel,
)


class ControllerJointImpedanceModelFree(ControllerBase):
    """Joint-space impedance controller with caller-supplied dynamics.

    Supports heterogeneous robot fleets — robots in the batch may have
    different DOF counts. Internal buffers are padded to ``max_dofs``; kernels
    skip padding slots via a per-robot guard.

    Allocate input and output structs via :meth:`input` and :meth:`output`.
    All field names on those structs are fixed — see :class:`Inputs` and
    :class:`Outputs` for the typed schema. Fields for disabled features
    (e.g. ``gravity_force`` when ``use_gravity_compensation=False``) are
    allocated as ``None`` and must not be written.

    Args:
        robot_count: Number of parallel robots.
        dofs_per_robot: DOF count for each robot, length ``robot_count``.
        max_dofs: Padded buffer width — must equal
            ``int(dofs_per_robot.numpy().max())``. Passed explicitly to avoid
            a device round-trip.
        default_dof_indices: Concatenated per-robot index arrays of length
            ``sum(dofs_per_robot)`` mapping controller DOF slots to positions
            in the flat simulation arrays (robot 0's indices first, then
            robot 1's, etc.).
        stiffness: Position-error gain Kp [N/m or N·m/rad], shape
            ``(robot_count, max_dofs)``. Pass a baked array or ``None`` to
            read from ``inputs.stiffness`` each step.
        damping: Velocity-error gain Kd [N·s/m or N·m·s/rad]. Same format
            as ``stiffness``.
        use_gravity_compensation: Add gravity generalized forces to τ.
        use_coriolis_compensation: Add Coriolis generalized forces to τ.
        use_inertia_decoupling: Premultiply the PD term by M(q).
        has_qdd_feedforward: Accept a desired-acceleration feedforward via
            ``inputs.joint_qdd``.
        joint_q_idx: Optional index override for ``inputs.joint_q``; same
            length/layout as ``default_dof_indices``.
        joint_qd_idx: Optional index override for ``inputs.joint_qd``.
        joint_q_des_idx: Optional index override for ``inputs.joint_q_des``.
        joint_qd_des_idx: Optional index override for ``inputs.joint_qd_des``.
        joint_qdd_idx: Optional index override for ``inputs.joint_qdd``.
        gravity_force_idx: Optional index override for ``inputs.gravity_force``.
        coriolis_force_idx: Optional index override for ``inputs.coriolis_force``.
        joint_f_idx: Optional index override for the torque output.
        device: Warp device.
        requires_grad: Whether internal buffers need gradient support.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerJointImpedanceModelFree.input`.

        All four kinematic fields are always allocated. Optional fields are
        ``None`` when the corresponding feature is disabled at construction.
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
        gravity_force: wp.array[wp.float32] | None
        """Gravity generalized forces [N or N·m], flat sim-level array. ``None`` unless ``use_gravity_compensation=True``."""
        coriolis_force: wp.array[wp.float32] | None
        """Coriolis generalized forces [N or N·m], flat sim-level array. ``None`` unless ``use_coriolis_compensation=True``."""
        mass_matrix: wp.array3d[wp.float32] | None
        """Per-robot generalized mass matrices [kg or kg·m²], shape ``(robot_count, max_dofs, max_dofs)``. ``None`` unless ``use_inertia_decoupling=True``."""
        stiffness: wp.array2d[wp.float32] | None
        """Position-error gain Kp [N/m or N·m/rad], shape ``(robot_count, max_dofs)``. ``None`` when gains are baked at construction."""
        damping: wp.array2d[wp.float32] | None
        """Velocity-error gain Kd [N·s/m or N·m·s/rad], shape ``(robot_count, max_dofs)``. ``None`` when gains are baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerJointImpedanceModelFree.output`."""

        joint_f: wp.array[wp.float32]
        """Joint torque command [N or N·m], flat sim-level array."""

    def __init__(
        self,
        *,
        robot_count: int,
        dofs_per_robot: wp.array[wp.int32],
        max_dofs: int,
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
        gravity_force_idx: wp.array[wp.uint32] | None = None,
        coriolis_force_idx: wp.array[wp.uint32] | None = None,
        joint_f_idx: wp.array[wp.uint32] | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        if robot_count < 1:
            raise ValueError(f"robot_count must be >= 1, got {robot_count}.")
        if not isinstance(dofs_per_robot, wp.array) or dofs_per_robot.dtype != wp.int32:
            raise TypeError("dofs_per_robot must be wp.array[int32].")
        if int(dofs_per_robot.size) != robot_count:
            raise ValueError(f"dofs_per_robot length {dofs_per_robot.size} must equal robot_count={robot_count}.")
        if max_dofs < 1:
            raise ValueError(f"max_dofs must be >= 1, got {max_dofs}.")
        if not isinstance(default_dof_indices, wp.array) or default_dof_indices.dtype != wp.uint32:
            raise TypeError("default_dof_indices must be wp.array[uint32].")

        dofs_per_robot_np = dofs_per_robot.numpy()
        total_dofs = int(dofs_per_robot_np.sum())
        if int(default_dof_indices.size) != total_dofs:
            raise ValueError(
                f"default_dof_indices length {default_dof_indices.size} must equal sum(dofs_per_robot) = {total_dofs}."
            )

        self._robot_count = robot_count
        self._max_dofs = max_dofs
        self._total_dofs = total_dofs
        self._use_gravity = bool(use_gravity_compensation)
        self._use_coriolis = bool(use_coriolis_compensation)
        self._use_inertia = bool(use_inertia_decoupling)
        self._has_qdd = bool(has_qdd_feedforward)
        self._device = device if device is not None else wp.get_device()
        self._requires_grad = requires_grad

        self._dofs_per_robot = dofs_per_robot
        offsets_np = np.zeros(robot_count, dtype=np.int32)
        offsets_np[1:] = np.cumsum(dofs_per_robot_np[:-1])
        self._dof_offsets = wp.array(offsets_np, dtype=wp.int32, device=self._device)

        self._q_idx = _normalize_indices(joint_q_idx, default_dof_indices, name="joint_q")
        self._qd_idx = _normalize_indices(joint_qd_idx, default_dof_indices, name="joint_qd")
        self._q_des_idx = _normalize_indices(joint_q_des_idx, default_dof_indices, name="joint_q_des")
        self._qd_des_idx = _normalize_indices(joint_qd_des_idx, default_dof_indices, name="joint_qd_des")
        self._qdd_idx = _normalize_indices(joint_qdd_idx, default_dof_indices, name="joint_qdd")
        self._gravity_idx = _normalize_indices(gravity_force_idx, default_dof_indices, name="gravity_force")
        self._coriolis_idx = _normalize_indices(coriolis_force_idx, default_dof_indices, name="coriolis_force")
        self._f_idx = _normalize_indices(joint_f_idx, default_dof_indices, name="joint_f")

        f_idx_np = self._f_idx.numpy()
        if len(f_idx_np) != len(np.unique(f_idx_np)):
            raise ValueError(
                "joint_f output indices must be unique — two robots cannot scatter torques "
                "to the same simulation DOF slot."
            )

        self._stiffness_baked = self._normalize_gain(stiffness, "stiffness")
        self._damping_baked = self._normalize_gain(damping, "damping")

        def _buf():
            return wp.zeros((robot_count, max_dofs), dtype=wp.float32, device=self._device, requires_grad=requires_grad)

        self._q_2d = _buf()
        self._qd_2d = _buf()
        self._q_des_2d = _buf()
        self._qd_des_2d = _buf()
        self._qdd_2d: wp.array2d[wp.float32] | None = _buf() if self._has_qdd else None
        self._grav_2d: wp.array2d[wp.float32] | None = _buf() if self._use_gravity else None
        self._cor_2d: wp.array2d[wp.float32] | None = _buf() if self._use_coriolis else None

        self._tau_buf = _buf()
        self._acc_buf: wp.array2d[wp.float32] | None = _buf() if self._use_inertia else None

    def _normalize_gain(
        self,
        value: wp.array2d[wp.float32] | None,
        name: str,
    ) -> wp.array2d[wp.float32] | None:
        """Validate and copy a baked gain array, or return ``None`` for live gains."""
        if value is None:
            return None
        if isinstance(value, wp.array):
            expected = (self._robot_count, self._max_dofs)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Port '{name}': baked array shape {tuple(value.shape)} must equal "
                    f"(robot_count, max_dofs) = {expected}."
                )
            if value.dtype != wp.float32:
                raise TypeError(f"Port '{name}': baked array dtype must be wp.float32, got {value.dtype}.")
            baked = wp.zeros(expected, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad)
            wp.copy(baked, value)
            return baked
        raise TypeError(
            f"Port '{name}': must be wp.array2d[wp.float32] of shape (robot_count, max_dofs) "
            f"or None; got {type(value).__name__}."
        )

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
        """Return a pre-allocated :class:`Inputs` with zero-initialised arrays."""
        d, rg = self._device, self._requires_grad
        inputs = ControllerJointImpedanceModelFree.Inputs()
        inputs.joint_q = wp.zeros(_idx_max(self._q_idx), dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd = wp.zeros(_idx_max(self._qd_idx), dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_q_des = wp.zeros(_idx_max(self._q_des_idx), dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd_des = wp.zeros(_idx_max(self._qd_des_idx), dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qdd = (
            wp.zeros(_idx_max(self._qdd_idx), dtype=wp.float32, device=d, requires_grad=rg) if self._has_qdd else None
        )
        inputs.gravity_force = (
            wp.zeros(_idx_max(self._gravity_idx), dtype=wp.float32, device=d, requires_grad=rg)
            if self._use_gravity
            else None
        )
        inputs.coriolis_force = (
            wp.zeros(_idx_max(self._coriolis_idx), dtype=wp.float32, device=d, requires_grad=rg)
            if self._use_coriolis
            else None
        )
        shape_2d = (self._robot_count, self._max_dofs)
        inputs.mass_matrix = (
            wp.zeros((self._robot_count, self._max_dofs, self._max_dofs), dtype=wp.float32, device=d, requires_grad=rg)
            if self._use_inertia
            else None
        )
        inputs.stiffness = (
            wp.zeros(shape_2d, dtype=wp.float32, device=d, requires_grad=rg) if self._stiffness_baked is None else None
        )
        inputs.damping = (
            wp.zeros(shape_2d, dtype=wp.float32, device=d, requires_grad=rg) if self._damping_baked is None else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with a flat torque array."""
        outputs = ControllerJointImpedanceModelFree.Outputs()
        outputs.joint_f = wp.zeros(
            _idx_max(self._f_idx), dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
        )
        return outputs

    def step(
        self,
        *,
        inputs: Inputs,
        outputs: Outputs,
        dt: float | wp.array[wp.float32],
    ) -> None:
        """Compute one impedance-control step and write joint torques.

        Args:
            inputs: Populated :class:`Inputs` struct. Dynamics fields must be
                filled by the caller before each call.
            outputs: :class:`Outputs` struct to write torques into.
            dt: Unused. Accepted for API compatibility.
        """
        stiffness = self._stiffness_baked if self._stiffness_baked is not None else inputs.stiffness
        damping = self._damping_baked if self._damping_baked is not None else inputs.damping

        dim2d = (self._robot_count, self._max_dofs)

        def _gather(src, src_idx, dst_2d):
            wp.launch(
                _gather_dof_kernel,
                dim=dim2d,
                inputs=[src, src_idx, self._dof_offsets, self._dofs_per_robot],
                outputs=[dst_2d],
                device=self._device,
            )

        _gather(inputs.joint_q, self._q_idx, self._q_2d)
        _gather(inputs.joint_qd, self._qd_idx, self._qd_2d)
        _gather(inputs.joint_q_des, self._q_des_idx, self._q_des_2d)
        _gather(inputs.joint_qd_des, self._qd_des_idx, self._qd_des_2d)
        if self._has_qdd:
            _gather(inputs.joint_qdd, self._qdd_idx, self._qdd_2d)
        if self._use_gravity:
            _gather(inputs.gravity_force, self._gravity_idx, self._grav_2d)
        if self._use_coriolis:
            _gather(inputs.coriolis_force, self._coriolis_idx, self._cor_2d)

        working_buf = self._acc_buf if self._use_inertia else self._tau_buf
        wp.launch(
            _pd_term_kernel,
            dim=dim2d,
            inputs=[self._q_2d, self._qd_2d, self._q_des_2d, self._qd_des_2d, stiffness, damping, self._dofs_per_robot],
            outputs=[working_buf],
            device=self._device,
        )

        if self._has_qdd:
            wp.launch(
                _add_term_kernel,
                dim=dim2d,
                inputs=[self._qdd_2d, self._dofs_per_robot],
                outputs=[working_buf],
                device=self._device,
            )

        if self._use_inertia:
            wp.launch(
                _mass_matrix_multiply_kernel,
                dim=dim2d,
                inputs=[inputs.mass_matrix, self._acc_buf, self._dofs_per_robot],
                outputs=[self._tau_buf],
                device=self._device,
            )

        if self._use_gravity:
            wp.launch(
                _add_term_kernel,
                dim=dim2d,
                inputs=[self._grav_2d, self._dofs_per_robot],
                outputs=[self._tau_buf],
                device=self._device,
            )
        if self._use_coriolis:
            wp.launch(
                _add_term_kernel,
                dim=dim2d,
                inputs=[self._cor_2d, self._dofs_per_robot],
                outputs=[self._tau_buf],
                device=self._device,
            )

        wp.launch(
            _scatter_dof_kernel,
            dim=dim2d,
            inputs=[self._tau_buf, self._f_idx, self._dof_offsets, self._dofs_per_robot],
            outputs=[outputs.joint_f],
            device=self._device,
        )
