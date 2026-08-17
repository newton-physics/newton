# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerJointImpedance — joint-space impedance control with
Newton model-internal dynamics.

Calls :func:`newton.eval_fk` and :func:`newton.eval_mass_matrix` on the
supplied model each step to obtain the mass matrix, then delegates all
gather/compute/scatter work to an inner
:class:`ControllerJointImpedanceModelFree` instance.

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
from ...utils import _validate_array, _validate_flat_port
from ._common import _gather_mass_matrix_blocks_kernel, _idx_max
from .model_free import ControllerJointImpedanceModelFree

# Joints whose position error ``q_des - q`` is a well-defined scalar subtraction.
_SCALAR_JOINT_TYPES = {int(JointType.REVOLUTE), int(JointType.PRISMATIC)}


class ControllerJointImpedance(ControllerBase):
    """Joint-space impedance controller with internally computed dynamics.

    Implements the joint-space impedance control law. This model-based variant
    computes the mass matrix, gravity, and Coriolis terms itself: it evaluates
    forward kinematics and the enabled dynamics terms from ``model`` on every
    :meth:`step`, so the caller supplies only joint positions and velocities.

    ``model`` is borrowed, not owned — it is never written to, and changes to
    it are visible to the controller immediately.

    Every coordinate of ``model`` contributes to ``inputs.joint_q`` and
    ``inputs.joint_qd``, since gravity, Coriolis, and mass-matrix terms depend
    on the state of the whole model, not only the controlled joints: a
    revolute joint mounted on an unactuated floating base still needs the
    base transform to compute the right gravity compensation.

    One robot is derived per articulation in ``model``. Within each
    articulation, every Revolute or Prismatic joint is controlled and every
    other joint (Fixed, or any multi-DOF type) is read but not actuated,
    since the PD error term ``q_des - q`` is only a well-defined scalar
    subtraction for a single-coordinate joint.

    Supports heterogeneous robot fleets — robots in the batch may have
    different controlled-DOF counts. The controller pads internal buffers to
    the largest per-robot controlled-DOF count and skips padding slots in all
    kernels.

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
        default_dof_indices: Concatenated per-robot index arrays of length
            ``sum(controlled dofs per articulation)`` mapping controller DOF
            slots to positions in the flat simulation arrays (robot 0's
            indices first, then robot 1's, etc.).
        stiffness: Position-error gain Kp, shape ``(N, max_dofs)``. Units
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
        joint_q_idx: Optional index array (same length as
            ``default_dof_indices``) overriding it for the position read.
            Each entry must be the coordinate index of a controlled joint.
        joint_qd_idx: Optional index array for velocity read. Each entry must
            be the DOF index of a controlled joint.
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
        """Current joint positions [m or rad], shape ``(model.joint_coord_count,)``."""
        joint_qd: wp.array[wp.float32]
        """Current joint velocities [m/s or rad/s], shape ``(model.joint_dof_count,)``."""
        joint_q_des: wp.array[wp.float32]
        """Desired joint positions [m or rad], flat sim-level array."""
        joint_qd_des: wp.array[wp.float32]
        """Desired joint velocities [m/s or rad/s], flat sim-level array."""
        joint_qdd: wp.array[wp.float32] | None
        """Desired acceleration feedforward [m/s² or rad/s²], flat sim-level array. ``None`` unless ``has_qdd_feedforward=True``."""
        stiffness: wp.array2d[wp.float32] | None
        """Position-error gain Kp, shape ``(robot_count, max_dofs)``. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m or N·m/rad]. ``None`` when gains are baked at construction."""
        damping: wp.array2d[wp.float32] | None
        """Velocity-error gain Kd, shape ``(robot_count, max_dofs)``. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m or N·m·s/rad]. ``None`` when gains are baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerJointImpedance.output`."""

        joint_f: wp.array[wp.float32]
        """Joint torque command [N or N·m], flat sim-level array."""

    def __init__(
        self,
        model: Model,
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
        if not isinstance(model, Model):
            raise TypeError(f"model must be a newton.Model, got {type(model).__name__}.")
        robot_count = model.articulation_count
        if robot_count < 1:
            raise ValueError("model has no articulations.")

        if model.device != device:
            raise ValueError(f"model.device is {model.device}, and device is {device}")
        if model.requires_grad != requires_grad:
            raise ValueError(f"model.requires_grad is {model.requires_grad}, and requires_grad is {requires_grad}")

        self._device = wp.get_device(device)
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

        # Every Revolute/Prismatic joint in an articulation is controlled; every
        # other joint (Fixed, or any multi-DOF type) is read for FK/dynamics but
        # not actuated, since only a single-coordinate joint has a well-defined
        # scalar PD error.
        joint_type_np = model.joint_type.numpy()
        art_start = model.articulation_start.numpy()
        art_end = model.articulation_end.numpy()
        is_scalar_joint = (joint_type_np == JointType.REVOLUTE) | (joint_type_np == JointType.PRISMATIC)
        dofs_per_robot_np = np.array(
            [int(is_scalar_joint[art_start[i] : art_end[i]].sum()) for i in range(robot_count)],
            dtype=np.int32,
        )
        dofs_per_robot = wp.array(dofs_per_robot_np, dtype=wp.int32, device=self._device)
        total_dofs = int(dofs_per_robot_np.sum())

        max_dofs = int(dofs_per_robot_np.max()) if dofs_per_robot_np.size else 0

        # ------------------------------------------------------------------
        # Validation: every wp.array argument is checked here, and nowhere
        # else. This runs after the per-robot DOF counts above are derived,
        # since the expected shapes depend on them.
        # ------------------------------------------------------------------
        gain_shape, idx_shape = (robot_count, max_dofs), (total_dofs,)
        for name, array, expected_dtype, expected_shape, required in (
            ("default_dof_indices", default_dof_indices, wp.uint32, idx_shape, True),
            ("stiffness", stiffness, wp.float32, gain_shape, False),
            ("damping", damping, wp.float32, gain_shape, False),
            ("joint_q_idx", joint_q_idx, wp.uint32, idx_shape, False),
            ("joint_qd_idx", joint_qd_idx, wp.uint32, idx_shape, False),
            ("joint_q_des_idx", joint_q_des_idx, wp.uint32, idx_shape, False),
            ("joint_qd_des_idx", joint_qd_des_idx, wp.uint32, idx_shape, False),
            ("joint_qdd_idx", joint_qdd_idx, wp.uint32, idx_shape, False),
            ("joint_f_idx", joint_f_idx, wp.uint32, idx_shape, False),
        ):
            _validate_array(array=array, name=name, dtype=expected_dtype, shape=expected_shape, device=self._device, required=required)
        # ------------------------------------------------------------------

        self._robot_count = robot_count
        self._max_dofs = max_dofs
        self._total_dofs = total_dofs
        self._dofs_per_robot = dofs_per_robot

        self._q_idx = default_dof_indices if joint_q_idx is None else joint_q_idx
        self._qd_idx = default_dof_indices if joint_qd_idx is None else joint_qd_idx
        self._q_des_idx = default_dof_indices if joint_q_des_idx is None else joint_q_des_idx
        self._qd_des_idx = default_dof_indices if joint_qd_des_idx is None else joint_qd_des_idx
        self._qdd_idx = default_dof_indices if joint_qdd_idx is None else joint_qdd_idx

        # Only *controlled* joints are restricted to 1-DOF revolute/prismatic —
        # an uncontrolled joint elsewhere in the model may be of any type. This
        # checks the joints actually addressed by the resolved read indices,
        # which catches a caller pointing an override at the wrong coordinate.
        self._validate_controlled_joints_are_scalar(joint_q_idx=self._q_idx, joint_qd_idx=self._qd_idx)

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
                self._compute_local_dof_idx(qd_idx_np=self._qd_idx.numpy(), dofs_per_robot_np=dofs_per_robot_np),
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
            # gravity_flat/coriolis_flat are filled by eval_inverse_dynamics_passive over
            # the whole model, so they are read through the same DOF-space index as
            # inputs.joint_qd rather than through default_dof_indices.
            gravity_force_idx=self._qd_idx,
            coriolis_force_idx=self._qd_idx,
            joint_f_idx=joint_f_idx,
            device=device,
            requires_grad=requires_grad,
        )

        # Pre-wired dynamics fields forwarded to ModelFree each step.
        self._mf_input = ControllerJointImpedanceModelFree.Inputs()
        self._mf_input.joint_q = self._model_state.joint_q
        self._mf_input.joint_qd = self._model_state.joint_qd
        if self._use_inertia:
            self._mf_input.mass_matrix = self._mass_matrix
        if self._use_gravity:
            self._mf_input.gravity_force = self._gravity_flat
        if self._use_coriolis:
            self._mf_input.coriolis_force = self._coriolis_flat

    def _validate_controlled_joints_are_scalar(
        self,
        *,
        joint_q_idx: wp.array[wp.uint32],
        joint_qd_idx: wp.array[wp.uint32],
    ) -> None:
        """Raise if a read index addresses a joint that is not 1-DOF revolute/prismatic."""
        joint_type_np = self._model.joint_type.numpy()
        q_start = self._model.joint_q_start.numpy()
        qd_start = self._model.joint_qd_start.numpy()

        unsupported: list[tuple[int, str]] = []
        for idx_np, start in ((joint_q_idx.numpy(), q_start), (joint_qd_idx.numpy(), qd_start)):
            owning_joint = np.searchsorted(start, idx_np, side="right") - 1
            unsupported.extend(
                (int(j), JointType(joint_type_np[j]).name)
                for j in owning_joint
                if joint_type_np[j] not in _SCALAR_JOINT_TYPES
            )
        if unsupported:
            unique_unsupported = sorted(set(unsupported))
            raise ValueError(
                f"ControllerJointImpedance only supports controlling 1-DOF joints "
                f"(Revolute/Prismatic); the read indices address unsupported joints: "
                f"{unique_unsupported}"
            )

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
        inputs.joint_q = wp.zeros(self._coord_count, dtype=wp.float32, device=d, requires_grad=rg)
        inputs.joint_qd = wp.zeros(self._dof_count, dtype=wp.float32, device=d, requires_grad=rg)
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
        # Checked here because the copies below consume these two ports before
        # the inner controller — which validates the rest — ever sees them.
        _validate_flat_port(
            array=inputs.joint_q, name="inputs.joint_q", min_length=self._coord_count, device=self._device
        )
        _validate_flat_port(
            array=inputs.joint_qd, name="inputs.joint_qd", min_length=self._dof_count, device=self._device
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
