# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerJointImpedanceModelFree — joint-space impedance control with
caller-supplied dynamics terms.

Every port is compact: one entry per controlled DOF, robot 0's DOFs first, then
robot 1's. The controller owns no index tables — a caller who needs to read from
or write to a simulation-sized array binds an indexed view
(``sim_array[selection.qd_start]``) instead.

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
from ...utils import _validate_array
from ._common import (
    _add_term_kernel,
    _mass_matrix_multiply_kernel,
    _pd_term_kernel,
    _read_port,
    _scatter_port_kernel,
)


class ControllerJointImpedanceModelFree(ControllerBase):
    """Joint-space impedance controller with caller-supplied dynamics.

    Implements the joint-space impedance control law. This model-free variant
    expects the mass matrix, gravity, and Coriolis terms to be computed
    externally — it is the caller's responsibility to compute the enabled ones
    correctly and write them into the input struct before every :meth:`step`.

    Every port is **compact**: a 1-D array with one entry per controlled DOF,
    ordered robot 0's DOFs first, then robot 1's, matching
    ``controlled_dofs_per_robot``. A port may be bound either to a plain compact
    array or to an indexed view of a simulation-sized array, which is how a
    caller expresses a gather or scatter without the controller owning an index
    table::

        inputs.joint_q = state.joint_q[selection.q_start]  # gather
        outputs.joint_f = control.joint_f[selection.qd_start]  # scatter

    Views are live and graph-capturable: bind them once, and each step (or graph
    replay) reads through to the current contents of the underlying array.

    Array shapes and devices are validated on each direct call to :meth:`step`,
    but not when a captured graph is replayed, since the checks run in Python
    at capture time only.

    Supports heterogeneous robot fleets — robots may have different
    controlled-DOF counts. Only the mass matrix is padded, to
    ``max_controlled_dofs``; every other buffer is compact.

    Allocate input and output structs via :meth:`input` and :meth:`output`.
    All field names on those structs are fixed — see :class:`Inputs` and
    :class:`Outputs` for the typed schema. Fields for disabled features
    (e.g. ``gravity_force`` when ``use_gravity_compensation=False``) are
    allocated as ``None`` and must not be written.

    See also :class:`ControllerJointImpedance`, which computes the mass matrix,
    gravity, and Coriolis terms internally from a Newton model.

    Args:
        controlled_dofs_per_robot: Controlled-DOF count for each robot. Its
            length sets :attr:`controlled_robot_count`, its sum sets
            :attr:`total_controlled_dofs` (the length of every port), and its
            maximum sets :attr:`max_controlled_dofs` (the padded width of the
            mass matrix). Every entry must be positive.
        stiffness: Position-error gain Kp. Units depend on
            ``use_inertia_decoupling``: [1/s²] when enabled, since the PD term
            is then an acceleration premultiplied by M(q); otherwise [N/m or
            N·m/rad]. Pass a scalar to apply the same gain to every controlled
            DOF, an array of shape [total_controlled_dofs] to set them
            individually, or ``None`` to read ``inputs.stiffness`` each step.
        damping: Velocity-error gain Kd, [1/s] when ``use_inertia_decoupling``
            is enabled, otherwise [N·s/m or N·m·s/rad]. Same format as
            ``stiffness``.
        use_gravity_compensation: Add gravity generalized forces to τ.
        use_coriolis_compensation: Add Coriolis generalized forces to τ.
        use_inertia_decoupling: Premultiply the PD term by M(q).
        has_qdd_feedforward: Accept a desired-acceleration feedforward via
            ``inputs.joint_qdd``.
        device: Warp device.
        requires_grad: Whether internal buffers need gradient support.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerJointImpedanceModelFree.input`.

        Every 1-D field is compact, shape [total_controlled_dofs]. Optional
        fields are ``None`` when the corresponding feature is disabled at
        construction.
        """

        joint_q: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Current joint positions [m or rad], shape [total_controlled_dofs]."""
        joint_qd: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Current joint velocities [m/s or rad/s], shape [total_controlled_dofs]."""
        joint_q_des: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Desired joint positions [m or rad], shape [total_controlled_dofs]."""
        joint_qd_des: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Desired joint velocities [m/s or rad/s], shape [total_controlled_dofs]."""
        joint_qdd: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Desired acceleration feedforward [m/s² or rad/s²], shape [total_controlled_dofs]. ``None`` unless ``has_qdd_feedforward=True``."""
        gravity_force: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Gravity generalized forces [N or N·m], shape [total_controlled_dofs]. ``None`` unless ``use_gravity_compensation=True``."""
        coriolis_force: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Coriolis generalized forces [N or N·m], shape [total_controlled_dofs]. ``None`` unless ``use_coriolis_compensation=True``."""
        mass_matrix: wp.array3d[wp.float32] | wp.indexedarray(dtype=wp.float32, ndim=3) | None
        """Per-robot mass matrices over the controlled DOFs, shape [controlled_robot_count, max_controlled_dofs, max_controlled_dofs]; a robot with fewer than ``max_controlled_dofs`` DOFs leaves the trailing rows and columns unread. May be bound to a view selecting those robots' blocks out of a larger set. Units by row/column DOF type: [kg] translational, [kg·m] mixed, [kg·m²] rotational. ``None`` unless ``use_inertia_decoupling=True``."""
        stiffness: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Position-error gain Kp, shape [total_controlled_dofs]. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m or N·m/rad]. ``None`` when gains are baked at construction."""
        damping: wp.array[wp.float32] | wp.indexedarray[wp.float32] | None
        """Velocity-error gain Kd, shape [total_controlled_dofs]. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m or N·m·s/rad]. ``None`` when gains are baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerJointImpedanceModelFree.output`."""

        joint_f: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Joint torque command [N or N·m], shape [total_controlled_dofs]."""

    def __init__(
        self,
        *,
        controlled_dofs_per_robot: wp.array[wp.int32],
        stiffness: wp.array[wp.float32] | float | None,
        damping: wp.array[wp.float32] | float | None,
        use_gravity_compensation: bool = True,
        use_coriolis_compensation: bool = True,
        use_inertia_decoupling: bool = True,
        has_qdd_feedforward: bool = False,
        device: Any = None,
        requires_grad: bool = False,
    ):
        self._device = wp.get_device(device)

        # ------------------------------------------------------------------
        # Validation: every wp.array argument is checked here, and nowhere
        # else. controlled_dofs_per_robot comes first because the shapes below derive
        # from it.
        # ------------------------------------------------------------------
        # Checked before reading .size below, which the shape argument depends on.
        if not isinstance(controlled_dofs_per_robot, wp.array):
            raise TypeError(
                f"controlled_dofs_per_robot must be a wp.array, got {type(controlled_dofs_per_robot).__name__}."
            )
        _validate_array(
            array=controlled_dofs_per_robot,
            name="controlled_dofs_per_robot",
            dtype=wp.int32,
            shape=(controlled_dofs_per_robot.size,),
            device=self._device,
        )

        controlled_dofs_per_robot_np = controlled_dofs_per_robot.numpy()
        controlled_robot_count = int(controlled_dofs_per_robot_np.size)
        if controlled_robot_count < 1:
            raise ValueError("controlled_dofs_per_robot must not be empty.")
        if controlled_dofs_per_robot_np.min() < 1:
            raise ValueError(
                f"controlled_dofs_per_robot must be positive — a robot with no controlled DOF occupies no "
                f"slot in any buffer, so leave it out; got {controlled_dofs_per_robot_np.tolist()}."
            )

        max_controlled_dofs = int(controlled_dofs_per_robot_np.max())
        total_controlled_dofs = int(controlled_dofs_per_robot_np.sum())

        for name, array in (("stiffness", stiffness), ("damping", damping)):
            if isinstance(array, (int, float)) and not isinstance(array, bool):
                continue  # broadcast at bake time, not a wp.array to validate
            _validate_array(
                array=array,
                name=name,
                dtype=wp.float32,
                shape=(total_controlled_dofs,),
                device=self._device,
                required=False,
            )
        # ------------------------------------------------------------------

        self._controlled_robot_count = controlled_robot_count
        self._max_controlled_dofs = max_controlled_dofs
        self._total_controlled_dofs = total_controlled_dofs
        self._use_gravity = bool(use_gravity_compensation)
        self._use_coriolis = bool(use_coriolis_compensation)
        self._use_inertia = bool(use_inertia_decoupling)
        self._has_qdd = bool(has_qdd_feedforward)
        self._requires_grad = requires_grad

        # Copied, not stored: the kernels use this as a loop bound while the
        # tables below are derived from the same host snapshot, so a later edit
        # to the caller's array would send the multiply past the end of a buffer.
        self._controlled_dofs_per_robot = wp.array(controlled_dofs_per_robot_np, dtype=wp.int32, device=self._device)

        # Flat-DOF -> (robot, slot) tables, so the mass-matrix multiply can run
        # as a flat launch over total_controlled_dofs instead of a padded 2-D one.
        offsets_np = np.zeros(controlled_robot_count, dtype=np.int32)
        offsets_np[1:] = np.cumsum(controlled_dofs_per_robot_np[:-1])
        self._dof_offsets = wp.array(offsets_np, dtype=wp.int32, device=self._device)
        self._robot_of_dof = wp.array(
            np.repeat(np.arange(controlled_robot_count, dtype=np.int32), controlled_dofs_per_robot_np),
            dtype=wp.int32,
            device=self._device,
        )
        self._slot_of_dof = wp.array(
            np.concatenate(
                [np.arange(n, dtype=np.int32) for n in controlled_dofs_per_robot_np] or [np.empty(0, np.int32)]
            ),
            dtype=wp.int32,
            device=self._device,
        )

        self._stiffness_baked = self._bake_gain(stiffness)
        self._damping_baked = self._bake_gain(damping)

        def _buf():
            return wp.zeros(total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad)

        # Every port is copied into one of these before any kernel runs, so a
        # port may be bound to a plain array or to an indexed view without the
        # kernels needing to know which.
        self._q_buf = _buf()
        self._qd_buf = _buf()
        self._q_des_buf = _buf()
        self._qd_des_buf = _buf()
        self._qdd_buf: wp.array[wp.float32] | None = _buf() if self._has_qdd else None
        self._grav_buf: wp.array[wp.float32] | None = _buf() if self._use_gravity else None
        self._cor_buf: wp.array[wp.float32] | None = _buf() if self._use_coriolis else None
        self._stiffness_buf: wp.array[wp.float32] | None = _buf() if self._stiffness_baked is None else None
        self._damping_buf: wp.array[wp.float32] | None = _buf() if self._damping_baked is None else None

        self._tau_buf = _buf()
        self._acc_buf: wp.array[wp.float32] | None = _buf() if self._use_inertia else None
        # Only used when the mass matrix is bound to a view; a plain array is
        # passed to the multiply kernel as it is. Allocated up front because
        # allocation is not allowed during graph capture.
        self._mass_matrix_buf: wp.array3d[wp.float32] | None = (
            wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            if self._use_inertia
            else None
        )

    def _bake_gain(self, value: wp.array[wp.float32] | float | None) -> wp.array[wp.float32] | None:
        """Broadcast a scalar, or copy a gain array, into a fresh compact buffer.

        Returns ``None`` for live gains, which are read from the input struct
        each step instead. A wp.array is already validated by
        :func:`_validate_array`.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return wp.full(
                self._total_controlled_dofs,
                float(value),
                dtype=wp.float32,
                device=self._device,
                requires_grad=self._requires_grad,
            )
        baked = wp.zeros(
            self._total_controlled_dofs,
            dtype=wp.float32,
            device=self._device,
            requires_grad=self._requires_grad,
        )
        wp.copy(baked, value)
        return baked

    @property
    def controlled_robot_count(self) -> int:
        """Number of robots, i.e. the length of ``controlled_dofs_per_robot``."""
        return self._controlled_robot_count

    @property
    def max_controlled_dofs(self) -> int:
        """Largest controlled-DOF count over the robots, the padded width of ``inputs.mass_matrix``."""
        return self._max_controlled_dofs

    @property
    def total_controlled_dofs(self) -> int:
        """Total controlled-DOF count across all robots, the length of every compact port."""
        return self._total_controlled_dofs

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
        d, rg, n = self._device, self._requires_grad, self._total_controlled_dofs

        def _port(enabled: bool) -> wp.array[wp.float32] | None:
            return wp.zeros(n, dtype=wp.float32, device=d, requires_grad=rg) if enabled else None

        inputs = ControllerJointImpedanceModelFree.Inputs()
        inputs.joint_q = _port(True)
        inputs.joint_qd = _port(True)
        inputs.joint_q_des = _port(True)
        inputs.joint_qd_des = _port(True)
        inputs.joint_qdd = _port(self._has_qdd)
        inputs.gravity_force = _port(self._use_gravity)
        inputs.coriolis_force = _port(self._use_coriolis)
        inputs.stiffness = _port(self._stiffness_baked is None)
        inputs.damping = _port(self._damping_baked is None)
        inputs.mass_matrix = (
            wp.zeros(
                (self._controlled_robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                dtype=wp.float32,
                device=d,
                requires_grad=rg,
            )
            if self._use_inertia
            else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with a compact torque array."""
        outputs = ControllerJointImpedanceModelFree.Outputs()
        outputs.joint_f = wp.zeros(
            self._total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
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
        # (port value, name, destination buffer) for every enabled port.
        bindings: list[tuple[Any, str, wp.array[wp.float32] | None]] = [
            (inputs.joint_q, "inputs.joint_q", self._q_buf),
            (inputs.joint_qd, "inputs.joint_qd", self._qd_buf),
            (inputs.joint_q_des, "inputs.joint_q_des", self._q_des_buf),
            (inputs.joint_qd_des, "inputs.joint_qd_des", self._qd_des_buf),
        ]
        if self._has_qdd:
            bindings.append((inputs.joint_qdd, "inputs.joint_qdd", self._qdd_buf))
        if self._use_gravity:
            bindings.append((inputs.gravity_force, "inputs.gravity_force", self._grav_buf))
        if self._use_coriolis:
            bindings.append((inputs.coriolis_force, "inputs.coriolis_force", self._cor_buf))
        if self._stiffness_baked is None:
            bindings.append((inputs.stiffness, "inputs.stiffness", self._stiffness_buf))
        if self._damping_baked is None:
            bindings.append((inputs.damping, "inputs.damping", self._damping_buf))

        # The output shares the ports' contract, so it is validated in the same
        # pass; a None destination marks it as written rather than read.
        bindings.append((outputs.joint_f, "outputs.joint_f", None))

        # A port belonging to a disabled feature is never read, so writing one
        # would go unnoticed. getattr because a caller may leave the field unset
        # rather than None.
        for name, enabled, switch in (
            ("joint_qdd", self._has_qdd, "has_qdd_feedforward"),
            ("gravity_force", self._use_gravity, "use_gravity_compensation"),
            ("coriolis_force", self._use_coriolis, "use_coriolis_compensation"),
            ("mass_matrix", self._use_inertia, "use_inertia_decoupling"),
            ("stiffness", self._stiffness_baked is None, "a live stiffness"),
            ("damping", self._damping_baked is None, "a live damping"),
        ):
            if not enabled and getattr(inputs, name, None) is not None:
                raise ValueError(
                    f"inputs.{name} is set, but the controller was built without {switch}, so the value "
                    f"would be ignored."
                )

        for port, name, buf in bindings:
            _validate_array(
                array=port,
                name=name,
                dtype=wp.float32,
                shape=(self._total_controlled_dofs,),
                device=self._device,
                allow_indexed=True,
            )
            if buf is not None:
                _read_port(port, buf, self._total_controlled_dofs, self._device)

        if self._use_inertia:
            _validate_array(
                array=inputs.mass_matrix,
                name="inputs.mass_matrix",
                dtype=wp.float32,
                shape=(self._controlled_robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                device=self._device,
                allow_indexed=True,
            )

        stiffness = self._stiffness_baked if self._stiffness_baked is not None else self._stiffness_buf
        damping = self._damping_baked if self._damping_baked is not None else self._damping_buf

        dim = self._total_controlled_dofs
        working_buf = self._acc_buf if self._use_inertia else self._tau_buf
        wp.launch(
            _pd_term_kernel,
            dim=dim,
            inputs=[self._q_buf, self._qd_buf, self._q_des_buf, self._qd_des_buf, stiffness, damping],
            outputs=[working_buf],
            device=self._device,
        )

        if self._has_qdd:
            wp.launch(_add_term_kernel, dim=dim, inputs=[self._qdd_buf], outputs=[working_buf], device=self._device)

        if self._use_inertia:
            mass_matrix = inputs.mass_matrix
            if isinstance(mass_matrix, wp.indexedarray):
                _read_port(
                    mass_matrix,
                    self._mass_matrix_buf,
                    (self._controlled_robot_count, self._max_controlled_dofs, self._max_controlled_dofs),
                    self._device,
                )
                mass_matrix = self._mass_matrix_buf
            wp.launch(
                _mass_matrix_multiply_kernel,
                dim=dim,
                inputs=[
                    mass_matrix,
                    self._acc_buf,
                    self._robot_of_dof,
                    self._slot_of_dof,
                    self._dof_offsets,
                    self._controlled_dofs_per_robot,
                ],
                outputs=[self._tau_buf],
                device=self._device,
            )

        if self._use_gravity:
            wp.launch(_add_term_kernel, dim=dim, inputs=[self._grav_buf], outputs=[self._tau_buf], device=self._device)
        if self._use_coriolis:
            wp.launch(_add_term_kernel, dim=dim, inputs=[self._cor_buf], outputs=[self._tau_buf], device=self._device)

        if isinstance(outputs.joint_f, wp.indexedarray):
            wp.launch(
                _scatter_port_kernel,
                dim=self._total_controlled_dofs,
                inputs=[self._tau_buf],
                outputs=[outputs.joint_f],
                device=self._device,
            )
        else:
            wp.copy(outputs.joint_f, self._tau_buf)
