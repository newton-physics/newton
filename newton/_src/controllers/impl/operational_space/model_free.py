# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerOperationalSpaceModelFree — task-space (operational-space) impedance
control with caller-supplied kinematics and dynamics terms.

Every port is compact: one entry per controlled DOF, robot 0's DOFs first,
then robot 1's — for `outputs.joint_f`, the controller's only per-DOF port.
Every other port is per-robot: exactly one entry per robot, since the task
itself is always 6-dimensional regardless of how many DOFs a robot has.

This first increment implements motion control only: a task-space
spring-damper term, with optional inertial decoupling through the
operational-space mass matrix Lambda. Wrench (force) control and null-space
posture control are not implemented yet.

Impedance law (terms enabled at construction):

    F = [Lambda if use_inertia_decoupling else I] · (Kp·pose_error + Kd·twist_error)
    tau = J^T · F

where ``pose_error`` and ``twist_error`` are the task-space position/orientation
and linear/angular velocity errors between the current and desired tool pose.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ...controller import ControllerBase
from ...utils import _validate_array
from .._common import _read_port, _scatter_port_kernel
from ._common import (
    _apply_spatial_matrix_kernel,
    _invert_spd_block_kernel,
    _jacobian_transpose_force_kernel,
    _operational_space_mass_matrix_inverse_kernel,
    _pose_error_kernel,
    _task_space_pd_kernel,
)


class ControllerOperationalSpaceModelFree(ControllerBase):
    """Task-space (operational-space) impedance controller with caller-supplied kinematics and dynamics.

    Implements the operational-space motion-control law. This model-free
    variant expects the tool pose, tool twist, tool-point Jacobian, and
    (when ``use_inertia_decoupling=True``) the controlled-DOF mass matrix to
    be computed externally — it is the caller's responsibility to compute
    these correctly and write them into the input struct before every
    :meth:`step`.

    Every port is **per-robot**: a 1-D array with one entry per robot,
    ordered to match ``controlled_dofs_per_robot`` — except
    ``outputs.joint_f``, which is **compact**: one entry per controlled DOF,
    robot 0's DOFs first, then robot 1's, exactly like
    :class:`~newton.controllers.ControllerJointImpedanceModelFree`.

    Every port, of any dtype, may be bound either to a plain array or to an
    indexed view of a simulation-sized array, exactly like
    :class:`ControllerJointImpedanceModelFree` — including the
    ``wp.transform``/``wp.spatial_vector`` ports (tool pose, tool twist,
    gains), via the ``wp.transform``/``wp.spatial_vector`` gather kernels in
    ``controllers/impl/_common.py``.

    Array shapes and devices are validated on each direct call to
    :meth:`step`, but not when a captured graph is replayed, since the
    checks run in Python at capture time only.

    Supports heterogeneous robot fleets — robots may have different
    controlled-DOF counts. Only the Jacobian and mass matrix are padded, to
    ``max_controlled_dofs``; every other buffer is compact or per-robot.

    Allocate input and output structs via :meth:`input` and :meth:`output`.
    All field names on those structs are fixed — see :class:`Inputs` and
    :class:`Outputs` for the typed schema. Fields for disabled features
    (e.g. ``mass_matrix`` when ``use_inertia_decoupling=False``) are
    allocated as ``None`` and must not be written.

    Args:
        controlled_dofs_per_robot: Controlled-DOF count for each robot. Its
            length sets :attr:`controlled_robot_count` (the length of every
            per-robot port), its sum sets :attr:`total_controlled_dofs` (the
            length of ``outputs.joint_f``), and its maximum sets
            :attr:`max_controlled_dofs` (the padded width of the Jacobian and
            mass matrix). Every entry must be positive, and — when
            ``use_inertia_decoupling=True`` — at least 6, since the
            operational-space mass matrix is only invertible for a robot
            whose Jacobian can span all 6 task dimensions.
        motion_stiffness: Task-space position/orientation-error gain Kp.
            Units depend on ``use_inertia_decoupling``: [1/s²] when enabled,
            since the spring-damper term is then a task-space acceleration
            premultiplied by Lambda; otherwise [N/m] on the position axes and
            [N·m/rad] on the orientation axes. Pass a scalar to apply the
            same gain to every axis of every robot, an array of shape
            [controlled_robot_count] to set them individually (one
            ``wp.spatial_vector`` of 6 gains per robot), or ``None`` to read
            ``inputs.motion_stiffness`` each step.
        motion_damping: Task-space velocity-error gain Kd, [1/s] when
            ``use_inertia_decoupling`` is enabled, otherwise [N·s/m] on the
            position axes and [N·m·s/rad] on the orientation axes. Same
            format as ``motion_stiffness``.
        use_inertia_decoupling: Premultiply the task-space spring-damper term
            by Lambda, the operational-space mass matrix.
        device: Warp device.
        requires_grad: Whether internal buffers need gradient support.
    """

    class Inputs:
        """Input struct returned by :meth:`~ControllerOperationalSpaceModelFree.input`.

        Every field is per-robot, shape [controlled_robot_count], except
        ``jacobian_tool_world`` and ``mass_matrix``, which are padded to
        [controlled_robot_count, ..., max_controlled_dofs]. Optional fields
        are ``None`` when the corresponding feature is disabled at
        construction.
        """

        coordinate_change_world_from_tool: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Current world pose of the tool frame, shape [controlled_robot_count]."""
        tool_twist_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Current tool twist (linear, angular) in world coordinates [m/s, rad/s], shape [controlled_robot_count]."""
        jacobian_tool_world: wp.array3d[wp.float32] | wp.indexedarray(dtype=wp.float32, ndim=3)
        """Tool-point Jacobian in world coordinates, shape [controlled_robot_count, 6, max_controlled_dofs]."""
        mass_matrix: wp.array3d[wp.float32] | wp.indexedarray(dtype=wp.float32, ndim=3) | None
        """Joint-space mass matrix over the controlled DOFs, shape [controlled_robot_count, max_controlled_dofs, max_controlled_dofs]; a robot with fewer than ``max_controlled_dofs`` DOFs leaves the trailing rows and columns unread. Units by row/column DOF type: [kg] translational, [kg·m] mixed, [kg·m²] rotational. ``None`` unless ``use_inertia_decoupling=True``."""
        coordinate_change_world_from_desired_tool: wp.array[wp.transform] | wp.indexedarray[wp.transform]
        """Desired world pose of the tool frame, shape [controlled_robot_count]."""
        desired_twist_world: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector]
        """Desired tool twist (linear, angular) in world coordinates [m/s, rad/s], shape [controlled_robot_count]."""
        motion_stiffness: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Task-space position/orientation-error gain Kp, shape [controlled_robot_count]. [1/s²] when ``use_inertia_decoupling`` is enabled, otherwise [N/m] / [N·m/rad]. ``None`` when gains are baked at construction."""
        motion_damping: wp.array[wp.spatial_vector] | wp.indexedarray[wp.spatial_vector] | None
        """Task-space velocity-error gain Kd, shape [controlled_robot_count]. [1/s] when ``use_inertia_decoupling`` is enabled, otherwise [N·s/m] / [N·m·s/rad]. ``None`` when gains are baked at construction."""

    class Outputs:
        """Output struct returned by :meth:`~ControllerOperationalSpaceModelFree.output`."""

        joint_f: wp.array[wp.float32] | wp.indexedarray[wp.float32]
        """Joint torque command [N or N·m], shape [total_controlled_dofs]."""

    def __init__(
        self,
        *,
        controlled_dofs_per_robot: wp.array[wp.int32],
        motion_stiffness: wp.array[wp.spatial_vector] | float | None,
        motion_damping: wp.array[wp.spatial_vector] | float | None,
        use_inertia_decoupling: bool = True,
        device: Any = None,
        requires_grad: bool = False,
    ):
        self._device = wp.get_device(device)

        # ------------------------------------------------------------------
        # Validation: every wp.array argument is checked here, and nowhere
        # else. controlled_dofs_per_robot comes first because the shapes
        # below derive from it.
        # ------------------------------------------------------------------
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
        if use_inertia_decoupling and controlled_dofs_per_robot_np.min() < 6:
            # Lambda^-1 = J M^-1 J^T only has rank min(6, controlled_dof_count):
            # with fewer than 6 controlled DOFs it is genuinely singular, not
            # just ill-conditioned, so inertial decoupling would silently
            # produce huge, physically meaningless forces along the
            # uncontrollable task directions instead of erroring.
            raise ValueError(
                f"use_inertia_decoupling=True requires every robot to have at least 6 controlled DOFs, "
                f"since the operational-space mass matrix is only invertible when the Jacobian can span "
                f"all 6 task dimensions; got controlled_dofs_per_robot={controlled_dofs_per_robot_np.tolist()}. "
                f"Pass use_inertia_decoupling=False for an under-actuated robot."
            )

        max_controlled_dofs = int(controlled_dofs_per_robot_np.max())
        total_controlled_dofs = int(controlled_dofs_per_robot_np.sum())

        for name, array in (("motion_stiffness", motion_stiffness), ("motion_damping", motion_damping)):
            if isinstance(array, (int, float)) and not isinstance(array, bool):
                continue  # broadcast at bake time, not a wp.array to validate
            _validate_array(
                array=array,
                name=name,
                dtype=wp.spatial_vector,
                shape=(controlled_robot_count,),
                device=self._device,
                required=False,
            )
        # ------------------------------------------------------------------

        self._controlled_robot_count = controlled_robot_count
        self._max_controlled_dofs = max_controlled_dofs
        self._total_controlled_dofs = total_controlled_dofs
        self._use_inertia = bool(use_inertia_decoupling)
        self._requires_grad = requires_grad

        # Copied, not stored: see the identical comment in
        # ControllerJointImpedanceModelFree.__init__ for why.
        self._controlled_dofs_per_robot = wp.array(controlled_dofs_per_robot_np, dtype=wp.int32, device=self._device)

        # Flat-DOF -> (robot, slot) tables, the same compact-DOF lookup
        # ControllerJointImpedanceModelFree builds, needed here so the final
        # Jacobian-transpose force mapping can write directly into the
        # compact total_controlled_dofs layout.
        offsets_np = np.zeros(controlled_robot_count, dtype=np.int32)
        offsets_np[1:] = np.cumsum(controlled_dofs_per_robot_np[:-1])
        self._robot_of_dof = wp.array(
            np.repeat(np.arange(controlled_robot_count, dtype=np.int32), controlled_dofs_per_robot_np),
            dtype=wp.int32,
            device=self._device,
        )
        self._slot_of_dof = wp.array(
            np.concatenate([np.arange(n, dtype=np.int32) for n in controlled_dofs_per_robot_np]),
            dtype=wp.int32,
            device=self._device,
        )

        self._stiffness_baked = self._bake_gain(motion_stiffness)
        self._damping_baked = self._bake_gain(motion_damping)

        def _pose_buf():
            return wp.zeros(
                controlled_robot_count, dtype=wp.transform, device=self._device, requires_grad=requires_grad
            )

        def _twist_buf():
            return wp.zeros(
                controlled_robot_count, dtype=wp.spatial_vector, device=self._device, requires_grad=requires_grad
            )

        # Every port is copied into one of these before any kernel runs, so
        # graph replay always reads through stable buffers regardless of
        # what array object the caller binds between steps.
        self._pose_buf = _pose_buf()
        self._twist_buf = _twist_buf()
        self._desired_pose_buf = _pose_buf()
        self._desired_twist_buf = _twist_buf()
        self._jacobian_buf = wp.zeros(
            (controlled_robot_count, 6, max_controlled_dofs),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )
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
        self._stiffness_buf: wp.array[wp.spatial_vector] | None = (
            _twist_buf() if self._stiffness_baked is None else None
        )
        self._damping_buf: wp.array[wp.spatial_vector] | None = _twist_buf() if self._damping_baked is None else None

        self._pose_error_buf = _twist_buf()
        self._desired_task_acceleration_buf = _twist_buf()
        self._task_space_force_buf: wp.array[wp.spatial_vector] | None = _twist_buf() if self._use_inertia else None

        # Lambda's Cholesky scratch and inverse-mass-matrix Cholesky scratch,
        # only needed when inertial decoupling is enabled.
        self._mass_matrix_cholesky: wp.array3d[wp.float32] | None = None
        self._mass_matrix_inv: wp.array3d[wp.float32] | None = None
        self._operational_space_mass_matrix_inv: wp.array3d[wp.float32] | None = None
        self._operational_space_mass_matrix_cholesky: wp.array3d[wp.float32] | None = None
        self._operational_space_mass_matrix: wp.array3d[wp.float32] | None = None
        self._task_dim: wp.array[wp.int32] | None = None
        if self._use_inertia:
            self._mass_matrix_cholesky = wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._mass_matrix_inv = wp.zeros(
                (controlled_robot_count, max_controlled_dofs, max_controlled_dofs),
                dtype=wp.float32,
                device=self._device,
                requires_grad=requires_grad,
            )
            self._operational_space_mass_matrix_inv = wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._operational_space_mass_matrix_cholesky = wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._operational_space_mass_matrix = wp.zeros(
                (controlled_robot_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
            )
            self._task_dim = wp.full(controlled_robot_count, 6, dtype=wp.int32, device=self._device)

        self._tau_buf = wp.zeros(
            total_controlled_dofs, dtype=wp.float32, device=self._device, requires_grad=requires_grad
        )

    def _bake_gain(self, value: wp.array[wp.spatial_vector] | float | None) -> wp.array[wp.spatial_vector] | None:
        """Broadcast a scalar, or copy a gain array, into a fresh per-robot buffer.

        Returns ``None`` for live gains, which are read from the input struct
        each step instead. A wp.array is already validated by
        :func:`_validate_array`.
        """
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            v = float(value)
            return wp.full(
                self._controlled_robot_count,
                wp.spatial_vector(v, v, v, v, v, v),
                dtype=wp.spatial_vector,
                device=self._device,
                requires_grad=self._requires_grad,
            )
        baked = wp.zeros(
            self._controlled_robot_count,
            dtype=wp.spatial_vector,
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
        """Largest controlled-DOF count over the robots, the padded width of the Jacobian and mass matrix."""
        return self._max_controlled_dofs

    @property
    def total_controlled_dofs(self) -> int:
        """Total controlled-DOF count across all robots, the length of ``outputs.joint_f``."""
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
        d, rg, n = self._device, self._requires_grad, self._controlled_robot_count

        inputs = ControllerOperationalSpaceModelFree.Inputs()
        inputs.coordinate_change_world_from_tool = wp.zeros(n, dtype=wp.transform, device=d, requires_grad=rg)
        inputs.tool_twist_world = wp.zeros(n, dtype=wp.spatial_vector, device=d, requires_grad=rg)
        inputs.jacobian_tool_world = wp.zeros(
            (n, 6, self._max_controlled_dofs), dtype=wp.float32, device=d, requires_grad=rg
        )
        inputs.mass_matrix = (
            wp.zeros(
                (n, self._max_controlled_dofs, self._max_controlled_dofs), dtype=wp.float32, device=d, requires_grad=rg
            )
            if self._use_inertia
            else None
        )
        inputs.coordinate_change_world_from_desired_tool = wp.zeros(n, dtype=wp.transform, device=d, requires_grad=rg)
        inputs.desired_twist_world = wp.zeros(n, dtype=wp.spatial_vector, device=d, requires_grad=rg)
        inputs.motion_stiffness = (
            wp.zeros(n, dtype=wp.spatial_vector, device=d, requires_grad=rg) if self._stiffness_baked is None else None
        )
        inputs.motion_damping = (
            wp.zeros(n, dtype=wp.spatial_vector, device=d, requires_grad=rg) if self._damping_baked is None else None
        )
        return inputs

    def output(self) -> Outputs:
        """Return a pre-allocated :class:`Outputs` with a compact torque array."""
        outputs = ControllerOperationalSpaceModelFree.Outputs()
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
        """Compute one operational-space motion-control step and write joint torques.

        Args:
            inputs: Populated :class:`Inputs` struct. Kinematic and dynamics
                fields must be filled by the caller before each call.
            outputs: :class:`Outputs` struct to write torques into.
            dt: Unused. Accepted for API compatibility.
        """
        n = self._controlled_robot_count

        # A port belonging to a disabled feature is never read, so writing
        # one would go unnoticed. getattr because a caller may leave the
        # field unset rather than None.
        for name, enabled, switch in (
            ("mass_matrix", self._use_inertia, "use_inertia_decoupling"),
            ("motion_stiffness", self._stiffness_baked is None, "a live motion_stiffness"),
            ("motion_damping", self._damping_baked is None, "a live motion_damping"),
        ):
            if not enabled and getattr(inputs, name, None) is not None:
                raise ValueError(
                    f"inputs.{name} is set, but the controller was built without {switch}, so the value "
                    f"would be ignored."
                )

        # Per-robot (transform/spatial_vector) ports: may be bound to a plain
        # array or to an indexed view of a simulation-sized array, via the
        # same graph-capture-safe port machinery outputs.joint_f uses below.
        for port, name, dtype, buf in (
            (
                inputs.coordinate_change_world_from_tool,
                "inputs.coordinate_change_world_from_tool",
                wp.transform,
                self._pose_buf,
            ),
            (inputs.tool_twist_world, "inputs.tool_twist_world", wp.spatial_vector, self._twist_buf),
            (
                inputs.coordinate_change_world_from_desired_tool,
                "inputs.coordinate_change_world_from_desired_tool",
                wp.transform,
                self._desired_pose_buf,
            ),
            (inputs.desired_twist_world, "inputs.desired_twist_world", wp.spatial_vector, self._desired_twist_buf),
        ):
            _validate_array(array=port, name=name, dtype=dtype, shape=(n,), device=self._device, allow_indexed=True)
            _read_port(port, buf, n, self._device)

        if self._stiffness_baked is None:
            _validate_array(
                array=inputs.motion_stiffness,
                name="inputs.motion_stiffness",
                dtype=wp.spatial_vector,
                shape=(n,),
                device=self._device,
                allow_indexed=True,
            )
            _read_port(inputs.motion_stiffness, self._stiffness_buf, n, self._device)
        if self._damping_baked is None:
            _validate_array(
                array=inputs.motion_damping,
                name="inputs.motion_damping",
                dtype=wp.spatial_vector,
                shape=(n,),
                device=self._device,
                allow_indexed=True,
            )
            _read_port(inputs.motion_damping, self._damping_buf, n, self._device)

        # Jacobian and (optional) mass matrix: plain float32 arrays, so they
        # reuse the shared, view-aware port machinery.
        _validate_array(
            array=inputs.jacobian_tool_world,
            name="inputs.jacobian_tool_world",
            dtype=wp.float32,
            shape=(n, 6, self._max_controlled_dofs),
            device=self._device,
            allow_indexed=True,
        )
        _read_port(inputs.jacobian_tool_world, self._jacobian_buf, (n, 6, self._max_controlled_dofs), self._device)

        if self._use_inertia:
            _validate_array(
                array=inputs.mass_matrix,
                name="inputs.mass_matrix",
                dtype=wp.float32,
                shape=(n, self._max_controlled_dofs, self._max_controlled_dofs),
                device=self._device,
                allow_indexed=True,
            )
            _read_port(
                inputs.mass_matrix,
                self._mass_matrix_buf,
                (n, self._max_controlled_dofs, self._max_controlled_dofs),
                self._device,
            )

        stiffness = self._stiffness_baked if self._stiffness_baked is not None else self._stiffness_buf
        damping = self._damping_baked if self._damping_baked is not None else self._damping_buf

        wp.launch(
            _pose_error_kernel,
            dim=n,
            inputs=[self._pose_buf, self._desired_pose_buf],
            outputs=[self._pose_error_buf],
            device=self._device,
        )
        wp.launch(
            _task_space_pd_kernel,
            dim=n,
            inputs=[self._pose_error_buf, self._twist_buf, self._desired_twist_buf, stiffness, damping],
            outputs=[self._desired_task_acceleration_buf],
            device=self._device,
        )

        force_source = self._desired_task_acceleration_buf
        if self._use_inertia:
            wp.launch(
                _invert_spd_block_kernel,
                dim=n,
                inputs=[self._mass_matrix_buf, self._controlled_dofs_per_robot, self._mass_matrix_cholesky],
                outputs=[self._mass_matrix_inv],
                device=self._device,
            )
            wp.launch(
                _operational_space_mass_matrix_inverse_kernel,
                dim=(n, 6, 6),
                inputs=[self._jacobian_buf, self._mass_matrix_inv, self._controlled_dofs_per_robot],
                outputs=[self._operational_space_mass_matrix_inv],
                device=self._device,
            )
            wp.launch(
                _invert_spd_block_kernel,
                dim=n,
                inputs=[
                    self._operational_space_mass_matrix_inv,
                    self._task_dim,
                    self._operational_space_mass_matrix_cholesky,
                ],
                outputs=[self._operational_space_mass_matrix],
                device=self._device,
            )
            wp.launch(
                _apply_spatial_matrix_kernel,
                dim=n,
                inputs=[self._operational_space_mass_matrix, self._desired_task_acceleration_buf],
                outputs=[self._task_space_force_buf],
                device=self._device,
            )
            force_source = self._task_space_force_buf

        wp.launch(
            _jacobian_transpose_force_kernel,
            dim=self._total_controlled_dofs,
            inputs=[self._jacobian_buf, force_source, self._robot_of_dof, self._slot_of_dof],
            outputs=[self._tau_buf],
            device=self._device,
        )

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
