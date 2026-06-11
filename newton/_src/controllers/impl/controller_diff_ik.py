# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerDifferentialKinematics — one-step damped-least-squares
differential IK.

Stateless. For each robot in the batch, computes a joint-velocity command
that drives a user-defined site pose (position + orientation) toward a
target, using a damped pseudoinverse of the per-robot spatial Jacobian.

The DLS solve is split across four kernels so that the autograd-able tile
primitives (``wp.tile_cholesky``, ``wp.tile_cholesky_solve``) only see pure
tile_load -> tile_cholesky -> tile_cholesky_solve -> tile_store flows — no
element-wise mutation, which would break Warp's adjoint. All other math
(building the site-frame Jacobian, forming A = J J^T + lambda^2 I,
back-projecting q_dot from y) lives in per-element kernels that are
autograd-friendly by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ...sim.articulation import eval_fk, eval_jacobian
from ...sim.builder import ModelBuilder
from ..controller import Controller
from ..utils import (
    _allocate_namespace,
    _normalize_indices,
    _normalize_parameter_port,
)


@wp.kernel
def _gather_local_kernel(
    global_arr: wp.array[float],
    lookup_indices: wp.array[wp.uint32],
    local_arr: wp.array[float],
):
    i = wp.tid()
    local_arr[i] = global_arr[lookup_indices[i]]


@wp.kernel
def _build_site_jacobian_kernel(
    jacobian: wp.array3d[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    target_pos: wp.array[wp.vec3],
    target_pos_indices: wp.array[wp.uint32],
    target_quat: wp.array[wp.quat],
    target_quat_indices: wp.array[wp.uint32],
    # ``global_ee_body_per_robot`` indexes the flat full-model body arrays
    # (body_q / body_com), which concatenate every articulation's bodies.
    # ``local_ee_body`` is the per-articulation link index used as a row
    # offset into the per-robot Jacobian slab — the jacobian array's first
    # axis already separates robots, so this index stays local. The
    # consistency check in __init__ guarantees this value is identical
    # across all articulations, so it's passed as a scalar.
    global_ee_body_per_robot: wp.array[int],
    local_ee_body: int,
    site_xform_per_robot: wp.array[wp.transform],
    dofs_per_robot: int,
    j_site: wp.array3d[float],  # (num_robots, 6, dofs_per_robot)
    e_buffer: wp.array2d[float],  # (num_robots, 6)
):
    r = wp.tid()

    ee_body = global_ee_body_per_robot[r]
    j_row = local_ee_body * 6
    site_xform = site_xform_per_robot[r]

    t_body = body_q[ee_body]
    t_site = t_body * site_xform
    site_pos = wp.transform_get_translation(t_site)
    site_quat = wp.transform_get_rotation(t_site)
    com_world = wp.transform_point(t_body, body_com[ee_body])
    offset = site_pos - com_world

    pos_err = target_pos[target_pos_indices[r]] - site_pos
    q_err = target_quat[target_quat_indices[r]] * wp.quat_inverse(site_quat)
    s = wp.sign(q_err[3])
    rot_err = wp.vec3(2.0 * s * q_err[0], 2.0 * s * q_err[1], 2.0 * s * q_err[2])
    e_buffer[r, 0] = pos_err[0]
    e_buffer[r, 1] = pos_err[1]
    e_buffer[r, 2] = pos_err[2]
    e_buffer[r, 3] = rot_err[0]
    e_buffer[r, 4] = rot_err[1]
    e_buffer[r, 5] = rot_err[2]

    for j in range(dofs_per_robot):
        jl_x = jacobian[r, j_row + 0, j]
        jl_y = jacobian[r, j_row + 1, j]
        jl_z = jacobian[r, j_row + 2, j]
        ja_x = jacobian[r, j_row + 3, j]
        ja_y = jacobian[r, j_row + 4, j]
        ja_z = jacobian[r, j_row + 5, j]
        j_site[r, 0, j] = jl_x + ja_y * offset[2] - ja_z * offset[1]
        j_site[r, 1, j] = jl_y + ja_z * offset[0] - ja_x * offset[2]
        j_site[r, 2, j] = jl_z + ja_x * offset[1] - ja_y * offset[0]
        j_site[r, 3, j] = ja_x
        j_site[r, 4, j] = ja_y
        j_site[r, 5, j] = ja_z


@wp.kernel
def _build_dls_matrix_kernel(
    j_site: wp.array3d[float],
    solver_damping: wp.array[float],
    dofs_per_robot: int,
    A: wp.array3d[float],
):
    r, i, k = wp.tid()
    lam = solver_damping[r]
    lam_sq = lam * lam
    acc = float(0.0)
    for j in range(dofs_per_robot):
        acc += j_site[r, i, j] * j_site[r, k, j]
    if i == k:
        acc += lam_sq
    A[r, i, k] = acc


# wp.tile_cholesky*'s registered adjoints return zero gradients in Warp
# 1.14.0; marking the solve kernel enable_backward=False makes that explicit.
@wp.kernel(enable_backward=False)
def _cholesky_solve_kernel(
    A: wp.array3d[float],
    e_buffer: wp.array2d[float],
    y: wp.array2d[float],
):
    r = wp.tid()
    A_tile = wp.tile_load(A[r], shape=(6, 6))
    e_tile = wp.tile_load(e_buffer[r], shape=(6,))
    L = wp.tile_cholesky(A_tile)
    y_tile = wp.tile_cholesky_solve(L, e_tile)
    wp.tile_store(y[r], y_tile)


@wp.kernel
def _qd_from_y_kernel(
    j_site: wp.array3d[float],
    y: wp.array2d[float],
    bandwidth: wp.array[float],
    qd_target_local: wp.array2d[float],
):
    r, j = wp.tid()
    g = bandwidth[r]
    val = float(0.0)
    for i in range(6):
        val += j_site[r, i, j] * y[r, i]
    qd_target_local[r, j] = g * val


@wp.kernel
def _write_outputs_kernel(
    qd_target_local: wp.array2d[float],
    joint_q_local: wp.array[float],
    dt: float,
    dofs_per_robot: int,
    joint_target_qd_indices: wp.array[wp.uint32],
    joint_target_q_indices: wp.array[wp.uint32],
    joint_target_qd: wp.array[float],
    joint_target_q: wp.array[float],
):
    r, j = wp.tid()
    flat = r * dofs_per_robot + j
    qd = qd_target_local[r, j]
    joint_target_qd[joint_target_qd_indices[flat]] = qd
    joint_target_q[joint_target_q_indices[flat]] = joint_q_local[flat] + qd * dt


class ControllerDifferentialKinematics(Controller):
    """One-step damped-least-squares differential IK for a single
    end-effector per robot.

    Coupled per-robot: each robot's joint-velocity solution depends on its
    full configuration ``q``. Stateless. Drives the **site** pose in world
    frame toward the target. The site is identified by the ``label`` you
    gave it when calling :meth:`newton.ModelBuilder.add_site`.

    **Tape-safe, forward-only through the solve.** Every kernel in the
    chain except the DLS solve itself is autograd-able by default; the
    solve uses ``wp.tile_cholesky`` / ``wp.tile_cholesky_solve`` whose
    backward path returns zero gradients in Warp 1.14.0, so that one
    kernel is marked ``enable_backward=False``.

    Solve form (per robot, ``J_site`` is the 6xN site-frame Jacobian)::

        e        = [target_pos - site_pos ;  2 * sign(q_err.w) * q_err.xyz]
        A        = J_site J_site^T + lambda^2 * I_6                              (6x6 SPD)
        L L^T    = A                                                            (Cholesky)
        L L^T y  = e                                                            (solve)
        q_dot    = bandwidth * J_site^T y

    where ``q_err = target_quat * conj(site_quat)``. The joint_target_q
    output is written as ``q_current + q_dot * dt``.

    Construction takes a :class:`newton.ModelBuilder` containing exactly
    ``N`` topologically-identical articulations — the N robots this
    controller manages. The N articulations must share DOF count,
    link/joint count, and joint types; they may differ in physical
    parameters and per-articulation site placement. No replication: if
    the user wants R copies of a template they call
    :meth:`newton.ModelBuilder.replicate` themselves first.

    Assumptions:

    - All joints in the builder are scalar-DOF (revolute or prismatic).
    - The intended "world frame" is the builder's base frame.

    **Per-robot gain ports** (``solver_damping``, ``bandwidth``) accept
    either a :class:`wp.array` (baked; stored by copy at construction;
    length ``num_robots``, dtype ``wp.float32``) or a ``str`` (live;
    resolved from the input struct each step, read in natural per-robot
    order).

    Args:
        model_builder: Unfinalized N-articulation builder.
            ``num_robots = model_builder.articulation_count``.
        site: Label of the site to drive (added via
            :meth:`newton.ModelBuilder.add_site`). The builder must contain
            **exactly one** site with this label per articulation.
        default_dof_indices: Default indices for any live-data per-DOF
            port whose ``*_idx`` is ``None``. Length
            ``num_robots * dofs_per_robot``, laid out
            ``[r0_d0, r0_d1, ..., r1_d0, ...]``.
        solver_damping: DLS lambda per robot.
        bandwidth: Scalar multiplier on the DLS-solve output (per robot).
        target_pos_attr: Live read port — site position target in world
            frame. Dtype :class:`wp.vec3`.
        target_pos_idx: Override per-robot indices for ``target_pos_attr``,
            or ``None`` to use natural per-robot order.
        target_quat_attr: Live read port — site orientation target. Dtype
            :class:`wp.quat`.
        target_quat_idx: Override per-robot indices.
        joint_measurement_attr: Live read port — joint positions.
        joint_measurement_idx: Override per-DOF indices.
        joint_measurement_rate_attr: Live read port — joint velocities.
            Consumed by ``eval_fk`` to populate ``body_qd``; the DLS solve
            uses position-only error.
        joint_measurement_rate_idx: Override per-DOF indices.
        joint_target_q_attr: Live write port — commanded joint positions
            (``q_current + q_dot * dt``).
        joint_target_q_idx: Override per-DOF indices.
        joint_target_qd_attr: Live write port — commanded joint
            velocities.
        joint_target_qd_idx: Override per-DOF indices.
        device: Warp device for internal buffers. Defaults to
            :func:`wp.get_device`.
        requires_grad: If ``True``, internally-allocated buffers are
            created with gradient support.
    """

    def __init__(
        self,
        model_builder: ModelBuilder,
        controlled_site_label: str,
        default_dof_indices: wp.array,
        solver_damping: wp.array | str,
        bandwidth: wp.array | str,
        target_pos_attr: str = "site_target_position",
        target_pos_idx: wp.array | None = None,
        target_quat_attr: str = "site_target_quaternion",
        target_quat_idx: wp.array | None = None,
        joint_measurement_attr: str = "joint_q",
        joint_measurement_idx: wp.array | None = None,
        joint_measurement_rate_attr: str = "joint_qd",
        joint_measurement_rate_idx: wp.array | None = None,
        joint_target_q_attr: str = "joint_target_q",
        joint_target_q_idx: wp.array | None = None,
        joint_target_qd_attr: str = "joint_target_qd",
        joint_target_qd_idx: wp.array | None = None,
        device: Any = None,
        requires_grad: bool = False,
    ):
        if not isinstance(model_builder, ModelBuilder):
            raise TypeError(f"model_builder must be a newton.ModelBuilder, got {type(model_builder).__name__}.")
        if not isinstance(default_dof_indices, wp.array) or default_dof_indices.dtype != wp.uint32:
            raise TypeError("default_dof_indices must be wp.array[uint32]")
        articulation_count = model_builder.articulation_count
        if articulation_count < 1:
            raise ValueError("model_builder has no articulations.")
        if model_builder.joint_dof_count % articulation_count != 0:
            raise ValueError(
                f"model_builder.joint_dof_count={model_builder.joint_dof_count} is not divisible by "
                f"articulation_count={articulation_count}; the N articulations must share DOF count."
            )
        body_count = len(model_builder.body_q)
        if body_count % articulation_count != 0:
            raise ValueError(
                f"template body_count={body_count} is not divisible by articulation_count={articulation_count}; "
                f"the N articulations must share body count."
            )

        self._template = model_builder
        self._num_robots = articulation_count
        self._dofs_per_robot = model_builder.joint_dof_count // articulation_count
        self._bodies_per_robot = body_count // articulation_count
        self._num_outputs = self._num_robots * self._dofs_per_robot
        self._device = device if device is not None else wp.get_device()
        self._requires_grad = requires_grad

        if int(default_dof_indices.size) != self._num_outputs:
            raise ValueError(
                f"default_dof_indices length {default_dof_indices.size} must equal "
                f"num_robots * dofs_per_robot = {self._num_outputs}."
            )
        self._default_dof_indices = default_dof_indices

        # Per-articulation site lookup. Bodies are assumed grouped by
        # articulation in builder order (the natural pattern when
        # articulations are added one at a time, or via replicate()).
        all_site_idxs = [i for i, lbl in enumerate(model_builder.shape_label) if lbl == controlled_site_label]
        if len(all_site_idxs) != articulation_count:
            raise ValueError(
                f"expected exactly {articulation_count} shapes with label '{controlled_site_label}' (one per articulation), "
                f"found {len(all_site_idxs)} in model_builder; "
                f"available labels: {model_builder.shape_label}."
            )

        # Each robot can carry a different site transform (the body the site
        # rides on must be topologically identical — same per-articulation
        # link index — but its body-local pose can differ).
        # ``local_ee_body`` is that per-articulation link index, captured
        # from the first site we see and consistency-checked against the
        # rest. ``global_*`` names index into the flat full-template arrays
        # (body_q, body_com); ``local_*`` names index within a single
        # articulation.
        site_xform_per_robot_dict: dict[int, wp.transform] = {}
        local_ee_body: int | None = None
        for shape_idx in all_site_idxs:
            global_body = int(model_builder.shape_body[shape_idx])
            # Bodies are assumed grouped by articulation in builder order.
            articulation_i = global_body // self._bodies_per_robot
            local_body_i = global_body % self._bodies_per_robot

            if articulation_i in site_xform_per_robot_dict:
                raise ValueError(
                    f"multiple shapes labeled '{controlled_site_label}' resolved to articulation {articulation_i}; "
                    f"each articulation must contribute exactly one site with this label."
                )
            if local_ee_body is None:
                local_ee_body = local_body_i
            elif local_body_i != local_ee_body:
                raise ValueError(
                    f"articulation {articulation_i} has site '{controlled_site_label}' attached at a different "
                    f"per-articulation body (local index {local_body_i}) than previous articulations "
                    f"(local index {local_ee_body}); robots must be topologically identical."
                )
            site_xform_per_robot_dict[articulation_i] = model_builder.shape_transform[shape_idx]

        for r in range(articulation_count):
            if r not in site_xform_per_robot_dict:
                raise ValueError(f"articulation {r} has no shape labeled '{controlled_site_label}'.")

        global_ee_body_per_robot = [r * self._bodies_per_robot + local_ee_body for r in range(articulation_count)]
        site_xform_per_robot = [site_xform_per_robot_dict[r] for r in range(articulation_count)]

        # Resolve all of the custom/default indices:
        self._meas_attr = joint_measurement_attr
        self._meas_idx = _normalize_indices(joint_measurement_idx, default_dof_indices, name="joint_measurement")

        self._meas_rate_attr = joint_measurement_rate_attr
        self._meas_rate_idx = _normalize_indices(
            joint_measurement_rate_idx,
            default_dof_indices,
            name="joint_measurement_rate",
        )

        self._target_q_attr = joint_target_q_attr
        self._target_q_idx = _normalize_indices(joint_target_q_idx, default_dof_indices, name="joint_target_q")
        self._target_qd_attr = joint_target_qd_attr
        self._target_qd_idx = _normalize_indices(joint_target_qd_idx, default_dof_indices, name="joint_target_qd")

        # Per-robot live ports. Default natural-order = wp.arange(num_robots).
        default_robot_idx = wp.array(np.arange(articulation_count, dtype=np.uint32), device=self._device)
        self._default_robot_idx = default_robot_idx
        self._target_pos_attr = target_pos_attr
        self._target_pos_idx = _normalize_indices(target_pos_idx, default_robot_idx, name="target_pos")
        self._target_quat_attr = target_quat_attr
        self._target_quat_idx = _normalize_indices(target_quat_idx, default_robot_idx, name="target_quat")

        # Per-robot gain ports.
        self._damping_attr, self._damping_baked = _normalize_parameter_port(
            solver_damping, articulation_count, wp.float32, self._device, requires_grad, name="solver_damping"
        )
        self._bandwidth_attr, self._bandwidth_baked = _normalize_parameter_port(
            bandwidth, articulation_count, wp.float32, self._device, requires_grad, name="bandwidth"
        )

        # Allocate compute-side scratch + build the internal model.
        self._model = self._template.finalize(device=self._device, requires_grad=requires_grad)
        self._model_state = self._model.state()
        self._global_ee_body_per_robot = wp.array(global_ee_body_per_robot, dtype=int, device=self._device)
        self._local_ee_body = local_ee_body
        self._site_xform_per_robot = wp.array(site_xform_per_robot, dtype=wp.transform, device=self._device)
        self._jacobian = wp.zeros(
            (articulation_count, self._model.max_joints_per_articulation * 6, self._model.max_dofs_per_articulation),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )
        self._j_site = wp.zeros(
            (articulation_count, 6, self._dofs_per_robot),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )
        self._e_buffer = wp.zeros(
            (articulation_count, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
        )
        self._A = wp.zeros(
            (articulation_count, 6, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad
        )
        self._y = wp.zeros((articulation_count, 6), dtype=wp.float32, device=self._device, requires_grad=requires_grad)
        self._qd_target_local = wp.zeros(
            (articulation_count, self._dofs_per_robot),
            dtype=wp.float32,
            device=self._device,
            requires_grad=requires_grad,
        )

        # Input ports + any live gain ports --> fields on input_struct().
        self._input_specs: list[tuple[str, Any, int]] = [
            (self._meas_attr, wp.float32, _idx_max(self._meas_idx)),
            (self._meas_rate_attr, wp.float32, _idx_max(self._meas_rate_idx)),
            (self._target_pos_attr, wp.vec3, _idx_max(self._target_pos_idx)),
            (self._target_quat_attr, wp.quat, _idx_max(self._target_quat_idx)),
        ]
        for attr in (self._damping_attr, self._bandwidth_attr):
            if attr is not None:
                self._input_specs.append((attr, wp.float32, articulation_count))

        self._output_specs: list[tuple[str, Any, int]] = [
            (self._target_q_attr, wp.float32, _idx_max(self._target_q_idx)),
            (self._target_qd_attr, wp.float32, _idx_max(self._target_qd_idx)),
        ]

    @property
    def num_robots(self) -> int:
        return self._num_robots

    @property
    def dofs_per_robot(self) -> int:
        return self._dofs_per_robot

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

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

    def input_struct(self):
        return _allocate_namespace(self._input_specs, self._device, self._requires_grad)

    def output_struct(self):
        return _allocate_namespace(self._output_specs, self._device, self._requires_grad)

    def compute(
        self,
        input_struct: Any,
        output_struct: Any,
        controller_state_now: None,
        controller_state_next: None,
        time_step: float,
    ) -> None:
        meas = getattr(input_struct, self._meas_attr)
        meas_rate = getattr(input_struct, self._meas_rate_attr)
        target_pos = getattr(input_struct, self._target_pos_attr)
        target_quat = getattr(input_struct, self._target_quat_attr)
        damping = self._damping_baked or getattr(input_struct, self._damping_attr)
        bandwidth = self._bandwidth_baked or getattr(input_struct, self._bandwidth_attr)
        out_q = getattr(output_struct, self._target_q_attr)
        out_qd = getattr(output_struct, self._target_qd_attr)

        wp.launch(
            _gather_local_kernel,
            dim=self._num_outputs,
            inputs=[meas, self._meas_idx],
            outputs=[self._model_state.joint_q],
            device=self._device,
        )
        wp.launch(
            _gather_local_kernel,
            dim=self._num_outputs,
            inputs=[meas_rate, self._meas_rate_idx],
            outputs=[self._model_state.joint_qd],
            device=self._device,
        )

        eval_fk(self._model, self._model_state.joint_q, self._model_state.joint_qd, self._model_state)
        eval_jacobian(self._model, self._model_state, J=self._jacobian)

        wp.launch(
            _build_site_jacobian_kernel,
            dim=self._num_robots,
            inputs=[
                self._jacobian,
                self._model_state.body_q,
                self._model.body_com,
                target_pos,
                self._target_pos_idx,
                target_quat,
                self._target_quat_idx,
                self._global_ee_body_per_robot,
                self._local_ee_body,
                self._site_xform_per_robot,
                self._dofs_per_robot,
            ],
            outputs=[self._j_site, self._e_buffer],
            device=self._device,
        )
        wp.launch(
            _build_dls_matrix_kernel,
            dim=(self._num_robots, 6, 6),
            inputs=[self._j_site, damping, self._dofs_per_robot],
            outputs=[self._A],
            device=self._device,
        )
        wp.launch_tiled(
            _cholesky_solve_kernel,
            dim=[self._num_robots],
            inputs=[self._A, self._e_buffer],
            outputs=[self._y],
            block_dim=32,
            device=self._device,
        )
        wp.launch(
            _qd_from_y_kernel,
            dim=(self._num_robots, self._dofs_per_robot),
            inputs=[self._j_site, self._y, bandwidth],
            outputs=[self._qd_target_local],
            device=self._device,
        )
        wp.launch(
            _write_outputs_kernel,
            dim=(self._num_robots, self._dofs_per_robot),
            inputs=[
                self._qd_target_local,
                self._model_state.joint_q,
                time_step,
                self._dofs_per_robot,
                self._target_qd_idx,
                self._target_q_idx,
            ],
            outputs=[out_qd, out_q],
            device=self._device,
        )


def _idx_max(idx: wp.array) -> int:
    return int(np.max(idx.numpy())) + 1
