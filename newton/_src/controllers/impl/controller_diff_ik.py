# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControllerDifferentialIK — one-step damped-least-squares differential IK.

Stateless. For each robot in the batch, computes a joint-velocity command
that drives a user-defined site pose (position + orientation) toward a
target, using a damped pseudoinverse of the per-robot spatial Jacobian.
"""

from __future__ import annotations

import warp as wp

from ...sim.articulation import eval_fk, eval_jacobian
from ...sim.builder import ModelBuilder
from ..base import Controller
from ..utils import _normalize_port, _validate_per_group


@wp.kernel
def _gather_local_kernel(
    global_arr: wp.array[float],
    lookup_indices: wp.array[wp.uint32],
    local_arr: wp.array[float],
):
    i = wp.tid()
    local_arr[i] = global_arr[lookup_indices[i]]


@wp.func
def _site_j(
    jacobian: wp.array3d[float],
    r: int,
    j_row: int,
    i: int,
    j: int,
    offset: wp.vec3,
) -> float:
    # Returns the (i, j) element of the site-frame Jacobian.
    #
    # Newton's eval_jacobian gives the COM-frame body twist (v_com, omega).
    # For any other point P fixed on the body, v_P = v_com + cross(omega, P - com).
    # We use this with `offset = site_world - com_world` to convert each linear
    # row of J from "at COM" to "at site". Angular rows are unchanged.
    if i == 0:
        return jacobian[r, j_row + 0, j] + jacobian[r, j_row + 4, j] * offset[2] - jacobian[r, j_row + 5, j] * offset[1]
    if i == 1:
        return jacobian[r, j_row + 1, j] + jacobian[r, j_row + 5, j] * offset[0] - jacobian[r, j_row + 3, j] * offset[2]
    if i == 2:
        return jacobian[r, j_row + 2, j] + jacobian[r, j_row + 3, j] * offset[1] - jacobian[r, j_row + 4, j] * offset[0]
    return jacobian[r, j_row + i, j]


@wp.kernel
def _diff_ik_solve_kernel(
    # The full spatial Jacobian for the replicated model, shape
    # (num_robots, max_links * 6, max_dofs). Per-robot, row stride 6 per link,
    # with the first 3 rows being linear (COM-velocity) and the last 3
    # angular, per Newton's body-twist convention (v_com_world, omega_world).
    jacobian: wp.array3d[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    target_pos: wp.array[wp.vec3],
    target_quat: wp.array[wp.quat],
    damping: wp.array[float],
    gain: wp.array[float],
    site_xform: wp.transform,
    end_effector_link: int,
    bodies_per_robot: int,
    dofs_per_robot: int,
    # Output, length num_robots * dofs_per_robot.
    qd_target_local: wp.array[float],
):
    r = wp.tid()
    ee_body = r * bodies_per_robot + end_effector_link
    j_row = end_effector_link * 6

    # --- 1. Site pose in world frame ---
    # The site is a frame attached to the EE body at offset `site_xform`
    # from the body's reference frame. We drive this site's world-frame pose
    # toward the target, not the body's COM or origin.
    t_body = body_q[ee_body]
    t_site = t_body * site_xform
    site_pos = wp.transform_get_translation(t_site)
    site_quat = wp.transform_get_rotation(t_site)

    # Vector from COM (where Newton's Jacobian is defined) to the site point.
    # Used to convert each linear J row to site-frame velocity below.
    com_world = wp.transform_point(t_body, body_com[ee_body])
    offset = site_pos - com_world

    # --- 2. Task-space error e ∈ R^6 (3 position + 3 orientation) ---
    pos_err = target_pos[r] - site_pos

    # Orientation error via quaternion vector-part doubled.
    #
    # We form q_err = target * conj(current). q_err is the rotation that
    # takes current → target. Its scalar part is cos(theta/2) and its
    # vector part is sin(theta/2) * axis (where theta is the rotation
    # angle around `axis`).
    #
    # Doubling the vector part gives 2 * sin(theta/2) * axis, which equals
    # the rotation vector theta * axis for small theta and is a smooth,
    # singularity-free residual everywhere. This is the standard "quaternion
    # vector-part" residual used in damped-LS IK.
    #
    # Quaternions double-cover SO(3): q and -q represent the same rotation
    # but their vector parts differ in sign. Multiplying by sign(q_err.w)
    # picks the representative with positive scalar part.
    q_err = target_quat[r] * wp.quat_inverse(site_quat)
    s = wp.sign(q_err[3])
    rot_err = wp.vec3(2.0 * s * q_err[0], 2.0 * s * q_err[1], 2.0 * s * q_err[2])

    # Pack e into 6 scalars.
    e0 = pos_err[0]
    e1 = pos_err[1]
    e2 = pos_err[2]
    e3 = rot_err[0]
    e4 = rot_err[1]
    e5 = rot_err[2]

    # --- 3. Build A = J_site J_site^T + lambda^2 * I (6x6 SPD) ---
    # Each element accesses the site-corrected J via _site_j(...).
    lam_sq = damping[r] * damping[r]
    a = wp.spatial_matrix()
    for i in range(6):
        for k in range(i, 6):
            acc = float(0.0)
            for j in range(dofs_per_robot):
                acc += _site_j(jacobian, r, j_row, i, j, offset) * _site_j(jacobian, r, j_row, k, j, offset)
            if i == k:
                acc += lam_sq
            a[i, k] = acc
            a[k, i] = acc

    # --- 4. Cholesky decomposition A = L L^T (L lower triangular) ---
    ell = wp.spatial_matrix()
    for i in range(6):
        for j in range(i + 1):
            s_ij = a[i, j]
            for k in range(j):
                s_ij -= ell[i, k] * ell[j, k]
            if i == j:
                ell[i, i] = wp.sqrt(s_ij)
            else:
                ell[i, j] = s_ij / ell[j, j]

    # --- 5. Forward substitution: L z = e ---
    z0 = e0 / ell[0, 0]
    z1 = (e1 - ell[1, 0] * z0) / ell[1, 1]
    z2 = (e2 - ell[2, 0] * z0 - ell[2, 1] * z1) / ell[2, 2]
    z3 = (e3 - ell[3, 0] * z0 - ell[3, 1] * z1 - ell[3, 2] * z2) / ell[3, 3]
    z4 = (e4 - ell[4, 0] * z0 - ell[4, 1] * z1 - ell[4, 2] * z2 - ell[4, 3] * z3) / ell[4, 4]
    z5 = (e5 - ell[5, 0] * z0 - ell[5, 1] * z1 - ell[5, 2] * z2 - ell[5, 3] * z3 - ell[5, 4] * z4) / ell[5, 5]

    # --- 6. Back substitution: L^T y = z ---
    y5 = z5 / ell[5, 5]
    y4 = (z4 - ell[5, 4] * y5) / ell[4, 4]
    y3 = (z3 - ell[5, 3] * y5 - ell[4, 3] * y4) / ell[3, 3]
    y2 = (z2 - ell[5, 2] * y5 - ell[4, 2] * y4 - ell[3, 2] * y3) / ell[2, 2]
    y1 = (z1 - ell[5, 1] * y5 - ell[4, 1] * y4 - ell[3, 1] * y3 - ell[2, 1] * y2) / ell[1, 1]
    y0 = (z0 - ell[5, 0] * y5 - ell[4, 0] * y4 - ell[3, 0] * y3 - ell[2, 0] * y2 - ell[1, 0] * y1) / ell[0, 0]

    # --- 7. q_dot = J_site^T y ---
    robot_gain = gain[r]
    for j in range(dofs_per_robot):
        qd = (
            _site_j(jacobian, r, j_row, 0, j, offset) * y0
            + _site_j(jacobian, r, j_row, 1, j, offset) * y1
            + _site_j(jacobian, r, j_row, 2, j, offset) * y2
            + _site_j(jacobian, r, j_row, 3, j, offset) * y3
            + _site_j(jacobian, r, j_row, 4, j, offset) * y4
            + _site_j(jacobian, r, j_row, 5, j, offset) * y5
        ) * robot_gain
        qd_target_local[r * dofs_per_robot + j] = qd


@wp.kernel
def _accumulate_outputs_kernel(
    qd_target_local: wp.array[float],
    joint_q_local: wp.array[float],
    dt: float,
    output_qd_indices: wp.array[wp.uint32],
    output_q_indices: wp.array[wp.uint32],
    output_qd: wp.array[float],
    output_q: wp.array[float],
):
    i = wp.tid()
    qd = qd_target_local[i]
    output_qd[output_qd_indices[i]] = output_qd[output_qd_indices[i]] + qd
    output_q[output_q_indices[i]] = output_q[output_q_indices[i]] + (joint_q_local[i] + qd * dt)


class ControllerDifferentialIK(Controller):
    """One-step damped-least-squares differential IK for a single
    end-effector per robot.

    Coupled per-robot: each robot's joint-velocity solution depends on its
    full configuration ``q``. Stateless — no internal accumulators between
    steps. Drives the **site** pose in world frame toward the target. The
    site is identified by the ``label`` you gave it when calling
    :meth:`newton.ModelBuilder.add_site`; the controller looks up the EE
    link and the body-frame offset xform from the builder by that label.
    The Jacobian's COM-relative linear rows are converted to site-relative
    rows internally via the offset ``site_world - com_world`` and
    ``cross(omega, offset)``.

    Solve form (per robot, ``J_site`` is the 6xN site-frame Jacobian):

    .. code-block:: text

        e        = [target_pos - site_pos ;  2 * sign(q_err.w) * q_err.xyz]
        A        = J_site J_site^T + lambda^2 * I_6                              (6x6 SPD)
        L L^T    = A                                                            (Cholesky)
        L L^T y  = e                                                            (solve)
        q_dot    = gain * J_site^T y

    where ``q_err = target_quat * conj(site_quat)`` and ``gain`` is a
    per-robot scalar applied uniformly to every output DOF after the DLS
    solve. ``output_q`` is written as ``q_current + q_dot * dt``.

    Construction takes a :class:`newton.ModelBuilder` containing K
    topologically-identical articulations (K = ``model_builder.articulation_count``,
    K ≥ 1). All K articulations must share DOF count, link/joint count,
    and joint types; they may differ in physical parameters (mass, inertia,
    joint limits). At :meth:`finalize`, the controller replicates the
    builder ``R = len(indices) // model_builder.joint_dof_count`` times
    via :meth:`newton.ModelBuilder.replicate`, finalizes it on the chosen
    device, and allocates internal buffers.

    The controller assumes:

    - All joints in the template are **scalar-DOF** (revolute or prismatic),
      so ``joint_q.shape == joint_qd.shape == (joint_dof_count,)``.
    - The intended "world frame" is the base frame of the template model builder.

    Args:
        model_builder: Unfinalized K-articulation template.
        indices: Global DOF indices this controller writes to. Length
            ``num_robots * dofs_per_robot``;
            ``len(indices) % model_builder.joint_dof_count == 0``.
        site: Label of the site (added via :meth:`newton.ModelBuilder.add_site`)
            to drive. The controller drives the
            site's world-frame pose toward the target. Add a site at identity xform
            if you want to track an EE body's reference frame directly.
        measurement: Per-DOF port. Source of joint positions ``q``.
        measurement_rate: Per-DOF port. Source of joint velocities ``q_dot``
            (used by ``eval_fk`` to populate ``body_qd``; the solve uses
            position-only error).
        target_pos: Per-group ``wp.array[wp.vec3]`` of length ``num_robots``.
            Site position target in world (base) frame.
        target_quat: Per-group ``wp.array[wp.quat]`` of length ``num_robots``.
            Site orientation target.
        damping: Per-group ``wp.array[float]`` of length ``num_robots``. DLS
            ``lambda`` per robot. ``A = J_site J_site^T + lambda^2 * I``.
        gain: Per-group ``wp.array[float]`` of length ``num_robots``. Scalar
            multiplier applied to the DLS-solve output before writing into
            ``output_qd`` / integrating into ``output_q``. Use ``1.0`` for the
            raw DLS solution; raise or lower to tune convergence speed
            independently from the damping term.
        output_qd: Per-DOF port. Destination for ``q_dot`` (accumulated ``+=``).
        output_q: Per-DOF port. Destination for ``q_current + q_dot * dt`` (accumulated ``+=``).
    """

    def __init__(
        self,
        *,
        model_builder: ModelBuilder,
        indices: wp.array[wp.uint32],
        site: str,
        measurement,
        measurement_rate,
        target_pos,
        target_quat,
        damping,
        gain,
        output_qd,
        output_q,
    ):
        if not isinstance(model_builder, ModelBuilder):
            raise TypeError(
                f"ControllerDifferentialIK: model_builder must be a newton.ModelBuilder, "
                f"got {type(model_builder).__name__}."
            )
        K = model_builder.articulation_count
        if K < 1:
            raise ValueError("ControllerDifferentialIK: model_builder has no articulations.")
        if model_builder.joint_dof_count % K != 0:
            raise ValueError(
                f"ControllerDifferentialIK: model_builder.joint_dof_count={model_builder.joint_dof_count} "
                f"is not divisible by articulation_count={K}; the K articulations must share DOF count."
            )
        self._template = model_builder
        self._dofs_per_robot = model_builder.joint_dof_count // K
        if len(indices) % model_builder.joint_dof_count != 0:
            raise ValueError(
                f"ControllerDifferentialIK: len(indices)={len(indices)} is not a multiple of "
                f"model_builder.joint_dof_count={model_builder.joint_dof_count}."
            )
        self._replication_count = len(indices) // model_builder.joint_dof_count
        self._num_robots = K * self._replication_count
        self.indices = indices

        # Look up the site by label. Sites are stored as shapes inside the
        # builder; the label, the body it's attached to, and the body-frame
        # xform all live on parallel lists.
        try:
            site_idx = model_builder.shape_label.index(site)
        except ValueError as e:
            raise ValueError(
                f"ControllerDifferentialIK: no shape/site with label '{site}' in model_builder; "
                f"available labels: {model_builder.shape_label}."
            ) from e
        self._end_effector_link = int(model_builder.shape_body[site_idx])
        self._site_xform = model_builder.shape_transform[site_idx]

        self._target_pos = _validate_per_group(target_pos, self._num_robots, wp.vec3, name="target_pos")
        self._target_quat = _validate_per_group(target_quat, self._num_robots, wp.quat, name="target_quat")
        self._damping = _validate_per_group(damping, self._num_robots, wp.float32, name="damping")
        self._gain = _validate_per_group(gain, self._num_robots, wp.float32, name="gain")

        self._measurement = _normalize_port(measurement, indices, name="measurement")
        self._measurement_rate = _normalize_port(measurement_rate, indices, name="measurement_rate")
        self._output_qd = _normalize_port(output_qd, indices, name="output_qd")
        self._output_q = _normalize_port(output_q, indices, name="output_q")

    def finalize(self, device: wp.Device, num_outputs: int) -> None:
        # Replicate the K-articulation template R times into a fresh builder,
        # then finalize on the target device.
        builder = ModelBuilder()
        builder.replicate(self._template, world_count=self._replication_count)
        self._model = builder.finalize(device=device)
        self._model_state = self._model.state()

        if self._model.body_count % self._num_robots != 0:
            # Should not happen if the template's K articulations are homogeneous,
            # but guard against silent indexing errors.
            raise ValueError(
                f"ControllerDifferentialIK: replicated model body_count={self._model.body_count} is "
                f"not divisible by num_robots={self._num_robots}."
            )
        self._bodies_per_robot = self._model.body_count // self._num_robots

        n_total_dofs = self._num_robots * self._dofs_per_robot
        self._qd_target_local = wp.zeros(n_total_dofs, dtype=wp.float32, device=device)
        self._jacobian = wp.zeros(
            (self._num_robots, self._model.max_joints_per_articulation * 6, self._model.max_dofs_per_articulation),
            dtype=wp.float32,
            device=device,
        )

    def is_stateful(self) -> bool:
        return False

    def is_graphable(self) -> bool:
        return True

    def outputs(self) -> list[tuple[wp.array, wp.array[wp.uint32]]]:
        return [self._output_qd, self._output_q]

    def compute(
        self,
        state: Controller.State | None,
        next_state: Controller.State | None,
        dt: float,
    ) -> None:
        n = len(self.indices)

        meas, meas_idx = self._measurement
        meas_rate, mrate_idx = self._measurement_rate
        wp.launch(_gather_local_kernel, dim=n, inputs=[meas, meas_idx], outputs=[self._model_state.joint_q])
        wp.launch(_gather_local_kernel, dim=n, inputs=[meas_rate, mrate_idx], outputs=[self._model_state.joint_qd])

        eval_fk(self._model, self._model_state.joint_q, self._model_state.joint_qd, self._model_state)
        eval_jacobian(self._model, self._model_state, J=self._jacobian)

        wp.launch(
            _diff_ik_solve_kernel,
            dim=self._num_robots,
            inputs=[
                self._jacobian,
                self._model_state.body_q,
                self._model.body_com,
                self._target_pos,
                self._target_quat,
                self._damping,
                self._gain,
                self._site_xform,
                self._end_effector_link,
                self._bodies_per_robot,
                self._dofs_per_robot,
            ],
            outputs=[self._qd_target_local],
        )

        out_qd, out_qd_idx = self._output_qd
        out_q, out_q_idx = self._output_q
        wp.launch(
            _accumulate_outputs_kernel,
            dim=n,
            inputs=[
                self._qd_target_local,
                self._model_state.joint_q,
                dt,
                out_qd_idx,
                out_q_idx,
            ],
            outputs=[out_qd, out_q],
        )
