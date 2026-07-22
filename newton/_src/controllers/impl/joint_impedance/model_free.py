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

from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from ...controller import Controller
from ...utils import _allocate_namespace, _normalize_indices
from ._common import (
    _add_term_kernel,
    _gather_dof_kernel,
    _idx_max,
    _mass_matrix_multiply_kernel,
    _pd_term_kernel,
    _scatter_dof_kernel,
)


class ControllerJointImpedanceModelFree(Controller):
    """Joint-space impedance controller with caller-supplied dynamics.

    Supports heterogeneous robot fleets — robots in the batch may have
    different DOF counts. Internal buffers are padded to ``max_dofs``; kernels
    skip padding slots via a per-robot guard.

    **Input struct fields** (allocate via :meth:`input`):

    - ``joint_q``: flat float32 — current joint positions (sim-sized).
    - ``joint_qd``: flat float32 — current joint velocities (sim-sized).
    - ``joint_q_des``: flat float32 — desired joint positions (sim-sized).
    - ``joint_qd_des``: flat float32 — desired joint velocities (sim-sized).
    - ``joint_qdd``: flat float32 — desired acceleration feedforward
      (sim-sized). Only present when ``has_qdd_feedforward=True``.
    - ``gravity_force``: flat float32 — gravity generalized forces
      (sim-sized). Only present when ``use_gravity_compensation=True``.
    - ``coriolis_force``: flat float32 — Coriolis generalized forces
      (sim-sized). Only present when ``use_coriolis_compensation=True``.
    - ``mass_matrix``: ``(num_robots, max_dofs, max_dofs)`` float32 —
      per-robot mass matrices, padded to ``max_dofs``. Only present when
      ``use_inertia_decoupling=True``.
    - Live gain fields: present when ``stiffness`` or ``damping`` is a str.

    **Output struct fields** (allocate via :meth:`output`):

    - ``joint_f``: flat float32 — joint torque command (sim-sized).

    Args:
        num_robots: Number of parallel robots.
        dofs_per_robot: ``wp.array[int32]`` of length ``num_robots`` giving
            the DOF count for each robot. All values must be >= 1 and <=
            ``max_dofs``.
        max_dofs: Padded buffer width. Must equal
            ``int(dofs_per_robot.numpy().max())``. Exposed as a separate
            argument so callers can pass it without a device round-trip.
        default_dof_indices: ``wp.array[uint32]`` of length
            ``sum(dofs_per_robot)`` — concatenated per-robot index arrays
            mapping controller DOF slots to positions in the flat simulation
            arrays (robot 0's indices first, then robot 1's, etc.).
        stiffness: Position-error gain Kp. ``wp.array2d[float32]`` of shape
            ``(num_robots, max_dofs)`` (baked) or ``str`` (live attr on input
            struct pointing to an array of that shape).
        damping: Velocity-error gain Kd. Same format as ``stiffness``.
        use_gravity_compensation: Add gravity generalized forces to τ.
        use_coriolis_compensation: Add Coriolis generalized forces to τ.
        use_inertia_decoupling: Premultiply the PD term by M(q).
        has_qdd_feedforward: Accept a desired-acceleration feedforward
            ``joint_qdd`` in the input struct.
        joint_q_attr: Input struct attr name for current joint positions.
        joint_q_idx: Optional index override; same length/layout as
            ``default_dof_indices``.
        joint_qd_attr: Input struct attr name for current joint velocities.
        joint_qd_idx: Optional index override for current joint velocities.
        joint_q_des_attr: Input struct attr name for desired positions.
        joint_q_des_idx: Optional index override for desired positions.
        joint_qd_des_attr: Input struct attr name for desired velocities.
        joint_qd_des_idx: Optional index override for desired velocities.
        joint_qdd_attr: Input struct attr name for desired acceleration
            feedforward (only used when ``has_qdd_feedforward=True``).
        joint_qdd_idx: Optional index override for feedforward.
        gravity_force_attr: Input struct attr name for gravity forces.
        gravity_force_idx: Optional index override for gravity forces.
        coriolis_force_attr: Input struct attr name for Coriolis forces.
        coriolis_force_idx: Optional index override for Coriolis forces.
        mass_matrix_attr: Input struct attr name for the mass matrix.
        joint_f_attr: Output struct attr name for the torque command.
        joint_f_idx: Optional index override for the torque output.
        device: Warp device.
        requires_grad: Whether internal buffers need gradient support.
    """

    def __init__(
        self,
        num_robots: int,
        dofs_per_robot: wp.array[wp.int32],
        max_dofs: int,
        default_dof_indices: wp.array[wp.uint32],
        stiffness: wp.array2d[wp.float32] | str,
        damping: wp.array2d[wp.float32] | str,
        use_gravity_compensation: bool = True,
        use_coriolis_compensation: bool = True,
        use_inertia_decoupling: bool = True,
        has_qdd_feedforward: bool = False,
        joint_q_attr: str = "joint_q",
        joint_q_idx: wp.array[wp.uint32] | None = None,
        joint_qd_attr: str = "joint_qd",
        joint_qd_idx: wp.array[wp.uint32] | None = None,
        joint_q_des_attr: str = "joint_q_des",
        joint_q_des_idx: wp.array[wp.uint32] | None = None,
        joint_qd_des_attr: str = "joint_qd_des",
        joint_qd_des_idx: wp.array[wp.uint32] | None = None,
        joint_qdd_attr: str = "joint_qdd",
        joint_qdd_idx: wp.array[wp.uint32] | None = None,
        gravity_force_attr: str = "gravity_force",
        gravity_force_idx: wp.array[wp.uint32] | None = None,
        coriolis_force_attr: str = "coriolis_force",
        coriolis_force_idx: wp.array[wp.uint32] | None = None,
        mass_matrix_attr: str = "mass_matrix",
        joint_f_attr: str = "joint_f",
        joint_f_idx: wp.array[wp.uint32] | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        if num_robots < 1:
            raise ValueError(f"num_robots must be >= 1, got {num_robots}.")
        if not isinstance(dofs_per_robot, wp.array) or dofs_per_robot.dtype != wp.int32:
            raise TypeError("dofs_per_robot must be wp.array[int32].")
        if int(dofs_per_robot.size) != num_robots:
            raise ValueError(f"dofs_per_robot length {dofs_per_robot.size} must equal num_robots={num_robots}.")
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

        self._num_robots = num_robots
        self._max_dofs = max_dofs
        self._total_dofs = total_dofs
        self._use_gravity = bool(use_gravity_compensation)
        self._use_coriolis = bool(use_coriolis_compensation)
        self._use_inertia = bool(use_inertia_decoupling)
        self._has_qdd = bool(has_qdd_feedforward)
        self._device = device if device is not None else wp.get_device()
        self._requires_grad = requires_grad

        self._dofs_per_robot = dofs_per_robot
        # Cumulative offsets into the flat index / gather arrays (one per robot).
        offsets_np = np.zeros(num_robots, dtype=np.int32)
        offsets_np[1:] = np.cumsum(dofs_per_robot_np[:-1])
        self._dof_offsets = wp.array(offsets_np, dtype=wp.int32, device=self._device)

        self._q_attr = joint_q_attr
        self._qd_attr = joint_qd_attr
        self._q_des_attr = joint_q_des_attr
        self._qd_des_attr = joint_qd_des_attr
        self._qdd_attr = joint_qdd_attr
        self._gravity_attr = gravity_force_attr
        self._coriolis_attr = coriolis_force_attr
        self._mass_matrix_attr = mass_matrix_attr
        self._f_attr = joint_f_attr

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

        self._stiffness_attr, self._stiffness_baked = self._normalize_gain(stiffness, "stiffness")
        self._damping_attr, self._damping_baked = self._normalize_gain(damping, "damping")

        def _buf():
            return wp.zeros((num_robots, max_dofs), dtype=wp.float32, device=self._device, requires_grad=requires_grad)

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
        value: wp.array2d[wp.float32] | str,
        name: str,
    ) -> tuple[str | None, wp.array2d[wp.float32] | None]:
        """Validate and optionally bake a per-robot gain array."""
        if isinstance(value, str):
            return value, None
        if isinstance(value, wp.array):
            expected = (self._num_robots, self._max_dofs)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Port '{name}': baked array shape {tuple(value.shape)} must equal "
                    f"(num_robots, max_dofs) = {expected}."
                )
            if value.dtype != wp.float32:
                raise TypeError(f"Port '{name}': baked array dtype must be wp.float32, got {value.dtype}.")
            baked = wp.zeros(expected, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad)
            wp.copy(baked, value)
            return None, baked
        raise TypeError(
            f"Port '{name}': must be wp.array2d[wp.float32] of shape (num_robots, max_dofs) "
            f"or str; got {type(value).__name__}."
        )

    @property
    def num_robots(self) -> int:
        return self._num_robots

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

    def state(self) -> None:
        return None

    def input(self) -> SimpleNamespace:
        """Return a pre-allocated input struct with zero-initialised flat arrays."""
        specs = [
            (self._q_attr, wp.float32, _idx_max(self._q_idx)),
            (self._qd_attr, wp.float32, _idx_max(self._qd_idx)),
            (self._q_des_attr, wp.float32, _idx_max(self._q_des_idx)),
            (self._qd_des_attr, wp.float32, _idx_max(self._qd_des_idx)),
        ]
        if self._has_qdd:
            specs.append((self._qdd_attr, wp.float32, _idx_max(self._qdd_idx)))
        if self._use_gravity:
            specs.append((self._gravity_attr, wp.float32, _idx_max(self._gravity_idx)))
        if self._use_coriolis:
            specs.append((self._coriolis_attr, wp.float32, _idx_max(self._coriolis_idx)))
        ns = _allocate_namespace(specs, self._device, self._requires_grad)
        shape_2d = (self._num_robots, self._max_dofs)
        if self._stiffness_attr is not None:
            setattr(ns, self._stiffness_attr, wp.zeros(shape_2d, dtype=wp.float32, device=self._device))
        if self._damping_attr is not None:
            setattr(ns, self._damping_attr, wp.zeros(shape_2d, dtype=wp.float32, device=self._device))
        if self._use_inertia:
            setattr(
                ns,
                self._mass_matrix_attr,
                wp.zeros(
                    (self._num_robots, self._max_dofs, self._max_dofs),
                    dtype=wp.float32,
                    device=self._device,
                ),
            )
        return ns

    def output(self) -> SimpleNamespace:
        """Return a pre-allocated output struct with a flat torque array."""
        return _allocate_namespace(
            [(self._f_attr, wp.float32, _idx_max(self._f_idx))],
            self._device,
            self._requires_grad,
        )

    def compute(
        self,
        input_struct: Any,
        output_struct: Any,
        controller_state_now: None,
        controller_state_next: None,
        time_step: float | wp.array[wp.float32],
    ) -> None:
        """Compute one impedance-control step and write joint torques.

        Args:
            input_struct: Namespace with flat sim arrays as described in the
                class docstring. Dynamics fields must be populated by the
                caller before each call.
            output_struct: Namespace with a flat sim torque array.
            controller_state_now: Unused (stateless). Pass ``None``.
            controller_state_next: Unused. Pass ``None``.
            time_step: Unused. Accepted for API compatibility.
        """
        stiffness = (
            self._stiffness_baked if self._stiffness_baked is not None else getattr(input_struct, self._stiffness_attr)
        )
        damping = self._damping_baked if self._damping_baked is not None else getattr(input_struct, self._damping_attr)

        dim2d = (self._num_robots, self._max_dofs)

        def _gather(src_attr, src_idx, dst_2d):
            wp.launch(
                _gather_dof_kernel,
                dim=dim2d,
                inputs=[getattr(input_struct, src_attr), src_idx, self._dof_offsets, self._dofs_per_robot],
                outputs=[dst_2d],
                device=self._device,
            )

        _gather(self._q_attr, self._q_idx, self._q_2d)
        _gather(self._qd_attr, self._qd_idx, self._qd_2d)
        _gather(self._q_des_attr, self._q_des_idx, self._q_des_2d)
        _gather(self._qd_des_attr, self._qd_des_idx, self._qd_des_2d)
        if self._has_qdd:
            _gather(self._qdd_attr, self._qdd_idx, self._qdd_2d)
        if self._use_gravity:
            _gather(self._gravity_attr, self._gravity_idx, self._grav_2d)
        if self._use_coriolis:
            _gather(self._coriolis_attr, self._coriolis_idx, self._cor_2d)

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
                inputs=[getattr(input_struct, self._mass_matrix_attr), self._acc_buf, self._dofs_per_robot],
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
            outputs=[getattr(output_struct, self._f_attr)],
            device=self._device,
        )
