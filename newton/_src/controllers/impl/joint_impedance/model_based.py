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

from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from newton import JointType
from newton._src.sim.articulation import eval_fk, eval_mass_matrix
from newton._src.sim.builder import ModelBuilder
from newton._src.sim.inverse_dynamics import eval_inverse_dynamics_passive

from ...controller import Controller
from ...utils import _allocate_namespace, _normalize_indices
from ._common import _gather_dof_flat_kernel, _idx_max
from .model_free import ControllerJointImpedanceModelFree


class ControllerJointImpedance(Controller):
    """One-step joint-space impedance controller for a batch of robots.

    Has an identical input/output interface to
    :class:`ControllerJointImpedanceModelFree` — flat 1D sim arrays in,
    flat 1D torque array out — except that the dynamics terms (mass matrix,
    gravity force, Coriolis force) are computed internally from the Newton
    model rather than supplied by the caller.

    Supports heterogeneous robot fleets — robots in the batch may have
    different DOF counts. The ``model_builder`` articulations define the
    per-robot topology; the controller pads internal buffers to
    ``model.max_dofs_per_articulation`` and skips padding slots in all kernels.

    Impedance law (terms enabled at construction):

        τ = [M(q) if use_inertia_decoupling else I] · (q̈_des + Kp·Δq + Kd·Δq̇)
            + [C(q,q̇)·q̇ if use_coriolis_compensation else 0]
            + [g(q)      if use_gravity_compensation  else 0]

    Args:
        model_builder: :class:`~newton.ModelBuilder` with N articulations (one
            per robot). Articulations may have different DOF counts.
        default_dof_indices: ``wp.array[uint32]`` of length
            ``sum(dofs per articulation)`` — concatenated per-robot index
            arrays mapping controller DOF slots to positions in the flat
            simulation arrays (robot 0's indices first, then robot 1's, etc.).
        stiffness: Position-error gain Kp. ``wp.array2d[float32]`` of shape
            ``(N, max_dofs)`` (baked) or ``str`` (live attr on input struct).
        damping: Velocity-error gain Kd. Same format as ``stiffness``.
        use_gravity_compensation: Add gravity generalized forces to τ.
        use_coriolis_compensation: Add Coriolis generalized forces to τ.
        use_inertia_decoupling: Premultiply the PD term by M(q).
        has_qdd_feedforward: Accept a desired-acceleration feedforward
            ``joint_qdd`` in the input struct.
        joint_q_attr: Flat sim attr name for current joint positions.
        joint_q_idx: Optional index array (same length as
            ``default_dof_indices``) overriding it for the position read.
        joint_qd_attr: Flat sim attr name for current joint velocities.
        joint_qd_idx: Optional index array for velocity read.
        joint_q_des_attr: Flat sim attr name for desired joint positions.
        joint_q_des_idx: Optional index array for desired position read.
        joint_qd_des_attr: Flat sim attr name for desired joint velocities.
        joint_qd_des_idx: Optional index array for desired velocity read.
        joint_qdd_attr: Flat sim attr name for desired acceleration feedforward
            (only used when ``has_qdd_feedforward=True``).
        joint_qdd_idx: Optional index array for feedforward read.
        joint_f_attr: Flat sim attr name for the torque output.
        joint_f_idx: Optional index array overriding ``default_dof_indices``
            for the torque-output write.
        device: Warp device.
        requires_grad: Whether internal buffers need gradient support.
    """

    def __init__(
        self,
        model_builder: ModelBuilder,
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
        joint_f_attr: str = "joint_f",
        joint_f_idx: wp.array[wp.uint32] | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        if not isinstance(model_builder, ModelBuilder):
            raise TypeError(f"model_builder must be a newton.ModelBuilder, got {type(model_builder).__name__}.")
        if not isinstance(default_dof_indices, wp.array) or default_dof_indices.dtype != wp.uint32:
            raise TypeError("default_dof_indices must be wp.array[uint32].")

        num_robots = model_builder.articulation_count
        if num_robots < 1:
            raise ValueError("model_builder has no articulations.")

        scalar_dof_joint_types = {int(JointType.REVOLUTE), int(JointType.PRISMATIC)}
        unsupported_joints = [
            (joint_index, JointType(joint_type).name)
            for joint_index, joint_type in enumerate(model_builder.joint_type)
            if joint_type not in scalar_dof_joint_types
        ]
        if unsupported_joints:
            raise ValueError(
                f"ControllerJointImpedance requires 1-DOF joints (Revolute/Prismatic) only; "
                f"found unsupported joint types: {unsupported_joints}"
            )

        self._device = device if device is not None else wp.get_device()
        self._requires_grad = requires_grad
        self._use_gravity = bool(use_gravity_compensation)
        self._use_coriolis = bool(use_coriolis_compensation)
        self._use_inertia = bool(use_inertia_decoupling)
        self._has_qdd = bool(has_qdd_feedforward)
        self._needs_fk = self._use_inertia or self._use_gravity or self._use_coriolis

        self._model = model_builder.finalize(device=self._device, requires_grad=requires_grad)
        self._model_state = self._model.state()

        max_dofs = self._model.max_dofs_per_articulation

        # Derive per-articulation DOF counts from the finalized model.
        art_start = self._model.articulation_start.numpy()  # first joint per articulation
        art_end = self._model.articulation_end.numpy()  # one-past-last joint per articulation
        joint_q_start = self._model.joint_q_start.numpy()  # DOF start per joint (+1 sentinel)
        dofs_per_robot_np = np.array(
            [joint_q_start[art_end[i]] - joint_q_start[art_start[i]] for i in range(num_robots)],
            dtype=np.int32,
        )
        dofs_per_robot = wp.array(dofs_per_robot_np, dtype=wp.int32, device=self._device)
        total_dofs = int(dofs_per_robot_np.sum())

        if int(default_dof_indices.size) != total_dofs:
            raise ValueError(
                f"default_dof_indices length {default_dof_indices.size} must equal "
                f"sum of per-robot DOF counts = {total_dofs}."
            )

        self._num_robots = num_robots
        self._max_dofs = max_dofs
        self._total_dofs = total_dofs
        self._dofs_per_robot_np = dofs_per_robot_np

        self._q_attr = joint_q_attr
        self._q_idx = _normalize_indices(joint_q_idx, default_dof_indices, name="joint_q")
        self._qd_attr = joint_qd_attr
        self._qd_idx = _normalize_indices(joint_qd_idx, default_dof_indices, name="joint_qd")
        self._q_des_attr = joint_q_des_attr
        self._q_des_idx = _normalize_indices(joint_q_des_idx, default_dof_indices, name="joint_q_des")
        self._qd_des_attr = joint_qd_des_attr
        self._qd_des_idx = _normalize_indices(joint_qd_des_idx, default_dof_indices, name="joint_qd_des")
        self._qdd_attr = joint_qdd_attr
        self._qdd_idx = _normalize_indices(joint_qdd_idx, default_dof_indices, name="joint_qdd")
        self._f_attr = joint_f_attr
        self._f_idx = _normalize_indices(joint_f_idx, default_dof_indices, name="joint_f")

        # Validate gain port shapes only (ModelFree will store/copy them).
        self._stiffness_attr, _ = self._check_gain(stiffness, num_robots, max_dofs, "stiffness")
        self._damping_attr, _ = self._check_gain(damping, num_robots, max_dofs, "damping")

        self._mass_matrix: wp.array3d[wp.float32] | None = None
        self._gravity_flat: wp.array[wp.float32] | None = None
        self._coriolis_flat: wp.array[wp.float32] | None = None

        if self._use_inertia:
            self._mass_matrix = wp.zeros(
                (num_robots, max_dofs, max_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
        if self._use_gravity:
            self._gravity_flat = wp.zeros(total_dofs, dtype=wp.float32, device=self._device)
        if self._use_coriolis:
            self._coriolis_flat = wp.zeros(total_dofs, dtype=wp.float32, device=self._device)

        # Newton fills dynamics in the same DOF order as model.joint_q (robot-stride,
        # no sim-level remapping needed) — use identity indices.
        identity_idx = wp.array(np.arange(total_dofs, dtype=np.uint32), device=self._device)

        self._model_free = ControllerJointImpedanceModelFree(
            num_robots=num_robots,
            dofs_per_robot=dofs_per_robot,
            max_dofs=max_dofs,
            default_dof_indices=default_dof_indices,
            stiffness=stiffness,
            damping=damping,
            use_gravity_compensation=use_gravity_compensation,
            use_coriolis_compensation=use_coriolis_compensation,
            use_inertia_decoupling=use_inertia_decoupling,
            has_qdd_feedforward=has_qdd_feedforward,
            joint_q_attr=joint_q_attr,
            joint_q_idx=joint_q_idx,
            joint_qd_attr=joint_qd_attr,
            joint_qd_idx=joint_qd_idx,
            joint_q_des_attr=joint_q_des_attr,
            joint_q_des_idx=joint_q_des_idx,
            joint_qd_des_attr=joint_qd_des_attr,
            joint_qd_des_idx=joint_qd_des_idx,
            joint_qdd_attr=joint_qdd_attr,
            joint_qdd_idx=joint_qdd_idx,
            gravity_force_idx=identity_idx,
            coriolis_force_idx=identity_idx,
            joint_f_attr=joint_f_attr,
            joint_f_idx=joint_f_idx,
            device=device,
            requires_grad=requires_grad,
        )

        self._mf_input = SimpleNamespace()
        if self._use_inertia:
            self._mf_input.mass_matrix = self._mass_matrix
        if self._use_gravity:
            self._mf_input.gravity_force = self._gravity_flat
        if self._use_coriolis:
            self._mf_input.coriolis_force = self._coriolis_flat

        self._input_specs: list[tuple[str, Any, int]] = [
            (self._q_attr, wp.float32, _idx_max(self._q_idx)),
            (self._qd_attr, wp.float32, _idx_max(self._qd_idx)),
            (self._q_des_attr, wp.float32, _idx_max(self._q_des_idx)),
            (self._qd_des_attr, wp.float32, _idx_max(self._qd_des_idx)),
        ]
        if self._has_qdd:
            self._input_specs.append((self._qdd_attr, wp.float32, _idx_max(self._qdd_idx)))

    @staticmethod
    def _check_gain(
        value: wp.array2d[wp.float32] | str,
        num_robots: int,
        max_dofs: int,
        name: str,
    ) -> tuple[str | None, None]:
        """Validate gain port type/shape without copying (ModelFree does that)."""
        if isinstance(value, str):
            return value, None
        if isinstance(value, wp.array):
            expected = (num_robots, max_dofs)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Port '{name}': baked array shape {tuple(value.shape)} must equal "
                    f"(num_robots, max_dofs) = {expected}."
                )
            if value.dtype != wp.float32:
                raise TypeError(f"Port '{name}': baked array dtype must be wp.float32, got {value.dtype}.")
            return None, None
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

    def is_stateful(self) -> bool:
        return False

    def is_graphable(self) -> bool:
        return True

    def state(self) -> None:
        return None

    def input(self):
        """Return a pre-allocated input struct without dynamics fields (computed internally)."""
        ns = _allocate_namespace(self._input_specs, self._device, self._requires_grad)
        shape_2d = (self._num_robots, self._max_dofs)
        if self._stiffness_attr is not None:
            setattr(ns, self._stiffness_attr, wp.zeros(shape_2d, dtype=wp.float32, device=self._device))
        if self._damping_attr is not None:
            setattr(ns, self._damping_attr, wp.zeros(shape_2d, dtype=wp.float32, device=self._device))
        return ns

    def output(self):
        """Return a pre-allocated output struct with a flat torque array."""
        return self._model_free.output()

    def compute(
        self,
        input_struct: Any,
        output_struct: Any,
        controller_state_now: None,
        controller_state_next: None,
        time_step: float | wp.array[wp.float32],
    ) -> None:
        """Run one impedance-control step.

        Args:
            input_struct: Namespace with flat sim arrays for joint state and
                desired state. Dynamics terms are computed internally.
            output_struct: Namespace with a flat sim torque array.
            controller_state_now: Unused (stateless). Pass ``None``.
            controller_state_next: Unused. Pass ``None``.
            time_step: Unused. Accepted for API compatibility.
        """
        # Populate the Newton model state for FK/dynamics using a flat gather —
        # model_state.joint_q is a flat array of length total_dofs (no padding).
        wp.launch(
            _gather_dof_flat_kernel,
            dim=self._total_dofs,
            inputs=[getattr(input_struct, self._q_attr), self._q_idx],
            outputs=[self._model_state.joint_q],
            device=self._device,
        )
        wp.launch(
            _gather_dof_flat_kernel,
            dim=self._total_dofs,
            inputs=[getattr(input_struct, self._qd_attr), self._qd_idx],
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

        self._mf_input.joint_q = getattr(input_struct, self._q_attr)
        self._mf_input.joint_qd = getattr(input_struct, self._qd_attr)
        self._mf_input.joint_q_des = getattr(input_struct, self._q_des_attr)
        self._mf_input.joint_qd_des = getattr(input_struct, self._qd_des_attr)
        if self._has_qdd:
            self._mf_input.joint_qdd = getattr(input_struct, self._qdd_attr)
        if self._stiffness_attr is not None:
            setattr(self._mf_input, self._stiffness_attr, getattr(input_struct, self._stiffness_attr))
        if self._damping_attr is not None:
            setattr(self._mf_input, self._damping_attr, getattr(input_struct, self._damping_attr))

        self._model_free.compute(self._mf_input, output_struct, None, None, time_step)
