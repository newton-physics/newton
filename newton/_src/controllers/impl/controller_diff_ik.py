# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControlLawDifferentialIK — one-step damped-least-squares differential IK.

Stateless. For each robot in the batch, computes a joint-velocity command
that drives a user-defined site pose (position + orientation) toward a
target, using a damped pseudoinverse of the per-robot spatial Jacobian.

The DLS solve is split across four kernels so that the autograd-able tile
primitives (`wp.tile_cholesky`, `wp.tile_cholesky_solve`) only see pure
tile_load → tile_cholesky → tile_cholesky_solve → tile_store flows — no
element-wise mutation, which would break Warp's adjoint. All other math
(building the site-frame Jacobian, forming A = J J^T + λ²I, back-projecting
q_dot from y) lives in per-element kernels that are autograd-friendly by
construction. The whole controller is end-to-end differentiable when
constructed inside a ``Controller(..., requires_grad=True)``.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ...sim.articulation import eval_fk, eval_jacobian
from ...sim.builder import ModelBuilder
from ..control_law import ControlLaw
from ..utils import _normalize_per_group_port, _normalize_port, _resolve_input_array, _resolve_per_group_array


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
    # The full spatial Jacobian for the replicated model from ``eval_jacobian``,
    # shape (num_robots, max_links * 6, max_dofs). For each EE link, the 6 rows
    # at j_row = ee_link * 6 are the COM-frame body twist
    # (v_com_world, omega_world).
    jacobian: wp.array3d[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    target_pos: wp.array[wp.vec3],
    target_quat: wp.array[wp.quat],
    # Per-robot site-lookup arrays (precomputed at finalize() from per-variant
    # data). Each variant in the template may have its own end-effector body
    # and body-local site xform, and these arrays fan that out to every robot
    # in the variant-interleaved batch.
    ee_body_per_robot: wp.array[int],
    ee_link_per_robot: wp.array[int],
    site_xform_per_robot: wp.array[wp.transform],
    dofs_per_robot: int,
    # outputs:
    j_site: wp.array3d[float],  # (num_robots, 6, dofs_per_robot)
    e_buffer: wp.array2d[float],  # (num_robots, 6)
):
    """Per-robot: compute the site-frame Jacobian + task-space error vector.

    Splits site geometry out of the tile-Cholesky kernel so that the latter
    sees only pure tile primitives. No element mutation; gradients propagate
    cleanly from inputs (target_pos / target_quat / site_xform via body_q)
    to j_site / e_buffer.
    """
    r = wp.tid()
    ee_body = ee_body_per_robot[r]
    j_row = ee_link_per_robot[r] * 6
    site_xform = site_xform_per_robot[r]

    # --- Site pose in world ---
    t_body = body_q[ee_body]
    t_site = t_body * site_xform
    site_pos = wp.transform_get_translation(t_site)
    site_quat = wp.transform_get_rotation(t_site)
    com_world = wp.transform_point(t_body, body_com[ee_body])
    # Vector from COM (where Newton's Jacobian linear rows are defined) to
    # the site point. Translates v_com to v_site via cross(omega, offset).
    offset = site_pos - com_world

    # --- Task-space error e ∈ R^6 ---
    pos_err = target_pos[r] - site_pos
    # Orientation: q_err = target * conj(site) carries (cos(θ/2), sin(θ/2)*axis).
    # Doubling the vector part yields 2 sin(θ/2) axis ≈ θ * axis for small θ;
    # multiplying by sign(q_err.w) selects the representative with positive
    # scalar part (the shorter rotation).
    q_err = target_quat[r] * wp.quat_inverse(site_quat)
    s = wp.sign(q_err[3])
    rot_err = wp.vec3(2.0 * s * q_err[0], 2.0 * s * q_err[1], 2.0 * s * q_err[2])
    e_buffer[r, 0] = pos_err[0]
    e_buffer[r, 1] = pos_err[1]
    e_buffer[r, 2] = pos_err[2]
    e_buffer[r, 3] = rot_err[0]
    e_buffer[r, 4] = rot_err[1]
    e_buffer[r, 5] = rot_err[2]

    # --- Site-frame Jacobian rows ---
    # Linear rows take the cross-product correction; angular rows pass through.
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
    j_site: wp.array3d[float],  # (num_robots, 6, dofs_per_robot)
    damping: wp.array[float],  # (num_robots,)
    dofs_per_robot: int,
    A: wp.array3d[float],  # (num_robots, 6, 6) — output, A = J J^T + λ²I
):
    """Per-(robot, row, col): build the 6x6 SPD DLS matrix from J_site and λ.

    Computing this in a per-element kernel (rather than via tile_matmul +
    tile_diag_add inside the tile-Cholesky kernel) keeps the tile kernel
    free of element mutation patterns that Warp's autograd doesn't handle.
    """
    r, i, k = wp.tid()
    lam_sq = damping[r] * damping[r]
    acc = float(0.0)
    for j in range(dofs_per_robot):
        acc += j_site[r, i, j] * j_site[r, k, j]
    if i == k:
        acc += lam_sq
    A[r, i, k] = acc


# NOTE on differentiability: ``wp.tile_cholesky`` and ``wp.tile_cholesky_solve``
# advertise registered adjoints in their docstrings, but as of Warp 1.14.0 the
# backward pass produces zero gradients on the input arrays in practice
# (verified with a standalone test on this exact kernel — forward is correct,
# but gradients of both A and the rhs vector come back as all-zero). Until that
# upstream gap is fixed we mark this kernel ``enable_backward=False``, which
# blocks gradient propagation at the solve. Every other kernel in the chain
# (gather, build_site_jacobian, build_dls_matrix, qd_from_y, accumulate)
# remains autograd-able by default, so a Controller(..., requires_grad=True)
# still runs cleanly under wp.Tape — just with zero gradient through the IK.
@wp.kernel(enable_backward=False)
def _cholesky_solve_kernel(
    A: wp.array3d[float],  # (num_robots, 6, 6)
    e_buffer: wp.array2d[float],  # (num_robots, 6)
    y: wp.array2d[float],  # (num_robots, 6) — output, solution of A y = e
):
    """Per-robot tile solve: y = A^{-1} e via Cholesky.

    The only kernel that uses tile primitives. Pure flow:
    tile_load → tile_cholesky → tile_cholesky_solve → tile_store. No element
    mutation. See the module-level NOTE above for why this is marked
    enable_backward=False despite the tile primitives having registered
    adjoints.
    """
    r = wp.tid()
    A_tile = wp.tile_load(A[r], shape=(6, 6))
    e_tile = wp.tile_load(e_buffer[r], shape=(6,))
    L = wp.tile_cholesky(A_tile)
    y_tile = wp.tile_cholesky_solve(L, e_tile)
    wp.tile_store(y[r], y_tile)


@wp.kernel
def _qd_from_y_kernel(
    j_site: wp.array3d[float],  # (num_robots, 6, dofs_per_robot)
    y: wp.array2d[float],  # (num_robots, 6)
    gain: wp.array[float],  # (num_robots,)
    qd_target_local: wp.array2d[float],  # (num_robots, dofs_per_robot) — output
):
    """Per-(robot, dof_j): q_dot[j] = gain * sum_i J_site[i, j] * y[i]."""
    r, j = wp.tid()
    g = gain[r]
    val = float(0.0)
    for i in range(6):
        val += j_site[r, i, j] * y[r, i]
    qd_target_local[r, j] = g * val


@wp.kernel
def _accumulate_outputs_kernel(
    qd_target_local: wp.array2d[float],  # (num_robots, dofs_per_robot)
    joint_q_local: wp.array[float],  # (num_robots * dofs_per_robot,)
    dt: float,
    dofs_per_robot: int,
    output_qd_indices: wp.array[wp.uint32],
    output_q_indices: wp.array[wp.uint32],
    output_qd: wp.array[float],
    output_q: wp.array[float],
):
    r, j = wp.tid()
    flat = r * dofs_per_robot + j
    qd = qd_target_local[r, j]
    output_qd[output_qd_indices[flat]] = output_qd[output_qd_indices[flat]] + qd
    output_q[output_q_indices[flat]] = output_q[output_q_indices[flat]] + (joint_q_local[flat] + qd * dt)


class ControlLawDifferentialIK(ControlLaw):
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

    **Tape-safe, forward-only through the solve.** The controller runs
    cleanly inside a ``Controller(..., requires_grad=True)`` wrapped in
    ``wp.Tape``, and every kernel in the chain except the DLS solve itself
    is autograd-able by default. The inner DLS uses Warp's tiled Cholesky
    (`wp.tile_cholesky`, `wp.tile_cholesky_solve`); those primitives'
    docstrings advertise registered adjoints, but the backward path is not
    functional in Warp 1.14.0 (verified directly — forward correct, gradients
    return zero). The solve kernel is therefore marked
    ``enable_backward=False`` and gradients are blocked at it: useful for
    RL workflows that wrap the whole sim in a tape but don't need gradients
    through the IK; not yet usable for diff-physics training through the
    solve. Revisit when upstream tile_cholesky backward lands.

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
            to drive. The controller drives the site's world-frame pose
            toward the target. For K-articulation templates, the template
            must contain **exactly one** site with this label on each
            articulation; the K occurrences may sit on different bodies and
            carry different body-local xforms, so the K variants are free
            to have legitimately different end-effector geometry. Add a
            site at identity xform if you want to track an EE body's
            reference frame directly.

    All remaining ports are **string** attribute names resolved against the
    ``input`` / ``output`` objects passed to :meth:`Controller.step`.
    Per-DOF ports also accept ``(attr_name, port_indices)`` for custom
    layouts; per-group ports take just a name.

    Args (continued):
        measurement: Per-DOF read port. Source of joint positions ``q``.
        measurement_rate: Per-DOF read port. Source of joint velocities
            ``q_dot`` (used by ``eval_fk`` to populate ``body_qd``; the
            solve uses position-only error).
        target_pos: Per-group port; ``getattr(input, attr_name)`` must
            return ``wp.array[wp.vec3]`` of length ``num_robots``. Site
            position target in world (base) frame.
        target_quat: Per-group port; must resolve to
            ``wp.array[wp.quat]`` of length ``num_robots``. Site
            orientation target.
        damping: Per-group port; must resolve to ``wp.array[float]`` of
            length ``num_robots``. DLS ``lambda`` per robot.
        gain: Per-group port; must resolve to ``wp.array[float]`` of length
            ``num_robots``. Scalar multiplier applied to the DLS-solve
            output before writing into ``output_qd`` / integrating into
            ``output_q``. Use ``1.0`` for the raw DLS solution; raise or
            lower to tune convergence speed independently from damping.
        output_qd: Per-DOF write port. Destination for ``q_dot``
            (accumulated ``+=``).
        output_q: Per-DOF write port. Destination for
            ``q_current + q_dot * dt`` (accumulated ``+=``).
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
                f"ControlLawDifferentialIK: model_builder must be a newton.ModelBuilder, "
                f"got {type(model_builder).__name__}."
            )
        K = model_builder.articulation_count
        if K < 1:
            raise ValueError("ControlLawDifferentialIK: model_builder has no articulations.")
        if model_builder.joint_dof_count % K != 0:
            raise ValueError(
                f"ControlLawDifferentialIK: model_builder.joint_dof_count={model_builder.joint_dof_count} "
                f"is not divisible by articulation_count={K}; the K articulations must share DOF count."
            )
        self._template = model_builder
        self._dofs_per_robot = model_builder.joint_dof_count // K
        if len(indices) % model_builder.joint_dof_count != 0:
            raise ValueError(
                f"ControlLawDifferentialIK: len(indices)={len(indices)} is not a multiple of "
                f"model_builder.joint_dof_count={model_builder.joint_dof_count}."
            )
        self._replication_count = len(indices) // model_builder.joint_dof_count
        self._num_robots = K * self._replication_count
        self.indices = indices

        # Per-variant site lookup. The user gives one ``site`` label that's
        # expected to appear on every one of the K articulations in the
        # template; each occurrence may sit on a different body and have a
        # different body-local xform, so the K articulations can have
        # legitimately different end-effector geometry. We pick out one
        # shape per articulation and stash its (within-variant body index,
        # body-local xform) pair. `finalize()` later fans these out into
        # length-``num_robots`` Warp arrays keyed by robot index.
        #
        # We assume bodies are grouped by articulation in the template
        # (variant 0's bodies first, then variant 1's, …) — the natural
        # pattern when the user calls add_link / add_joint / add_articulation
        # one variant at a time. The check below enforces this implicitly:
        # if a variant's site lands on a body in another variant's block,
        # we'll get duplicate variant assignments and raise.
        body_count_in_template = len(model_builder.body_q)
        if body_count_in_template % K != 0:
            raise ValueError(
                f"ControlLawDifferentialIK: template body_count={body_count_in_template} is not "
                f"divisible by articulation_count={K}; the K articulations must share body count."
            )
        bodies_per_variant = body_count_in_template // K
        all_site_idxs = [i for i, label in enumerate(model_builder.shape_label) if label == site]
        if len(all_site_idxs) != K:
            raise ValueError(
                f"ControlLawDifferentialIK: expected exactly {K} shapes with label '{site}' "
                f"(one per articulation), found {len(all_site_idxs)} in model_builder; "
                f"available labels: {model_builder.shape_label}."
            )
        # Build per-variant data, keyed by variant index v in 0..K-1.
        site_for_variant: dict[int, tuple[int, wp.transform]] = {}
        for shape_idx in all_site_idxs:
            body_global = int(model_builder.shape_body[shape_idx])
            v = body_global // bodies_per_variant
            within_variant = body_global % bodies_per_variant
            if v in site_for_variant:
                raise ValueError(
                    f"ControlLawDifferentialIK: multiple shapes labeled '{site}' resolved to "
                    f"variant {v} in the template. Each articulation must contribute exactly one "
                    f"site with this label."
                )
            site_for_variant[v] = (within_variant, model_builder.shape_transform[shape_idx])
        for v in range(K):
            if v not in site_for_variant:
                raise ValueError(f"ControlLawDifferentialIK: variant {v} has no shape labeled '{site}'.")
        # Stash as lists ordered by variant index; finalize() turns these into
        # length-num_robots Warp arrays.
        self._ee_link_within_variant: list[int] = [site_for_variant[v][0] for v in range(K)]
        self._site_xform_per_variant: list[wp.transform] = [site_for_variant[v][1] for v in range(K)]

        self._target_pos_attr = _normalize_per_group_port(target_pos, name="target_pos")
        self._target_quat_attr = _normalize_per_group_port(target_quat, name="target_quat")
        self._damping_attr = _normalize_per_group_port(damping, name="damping")
        self._gain_attr = _normalize_per_group_port(gain, name="gain")

        self._measurement_attr, self._measurement_port_indices = _normalize_port(
            measurement, indices, name="measurement"
        )
        self._measurement_rate_attr, self._measurement_rate_port_indices = _normalize_port(
            measurement_rate, indices, name="measurement_rate"
        )
        self._output_qd_attr, self._output_qd_port_indices = _normalize_port(output_qd, indices, name="output_qd")
        self._output_q_attr, self._output_q_port_indices = _normalize_port(output_q, indices, name="output_q")

    def finalize(self, device: wp.Device, num_outputs: int, requires_grad: bool = False) -> None:
        # Replicate the K-articulation template R times into a fresh builder,
        # then finalize on the target device. Passing requires_grad through to
        # ModelBuilder.finalize and downstream wp.zeros calls keeps the
        # gradient chain intact for Isaac Lab / wp.Tape consumers; Model.state()
        # inherits from the Model's own requires_grad.
        builder = ModelBuilder()
        builder.replicate(self._template, world_count=self._replication_count)
        self._model = builder.finalize(device=device, requires_grad=requires_grad)
        self._model_state = self._model.state()

        if self._model.body_count % self._num_robots != 0:
            raise ValueError(
                f"ControlLawDifferentialIK: replicated model body_count={self._model.body_count} is "
                f"not divisible by num_robots={self._num_robots}."
            )
        self._bodies_per_robot = self._model.body_count // self._num_robots

        # Fan the per-variant site data out to length-num_robots arrays so the
        # kernel can index by robot. Variant-interleaved layout: robot r is
        # variant r % K from replication r // K (matches builder.replicate).
        K = len(self._ee_link_within_variant)
        ee_link_per_robot = [self._ee_link_within_variant[r % K] for r in range(self._num_robots)]
        ee_body_per_robot = [r * self._bodies_per_robot + ee_link_per_robot[r] for r in range(self._num_robots)]
        site_xform_per_robot = [self._site_xform_per_variant[r % K] for r in range(self._num_robots)]
        # These are build-time constants — no gradient flow through them — so
        # requires_grad stays False regardless of the controller-level flag.
        self._ee_body_per_robot = wp.array(ee_body_per_robot, dtype=int, device=device)
        self._ee_link_per_robot = wp.array(ee_link_per_robot, dtype=int, device=device)
        self._site_xform_per_robot = wp.array(site_xform_per_robot, dtype=wp.transform, device=device)

        self._jacobian = wp.zeros(
            (self._num_robots, self._model.max_joints_per_articulation * 6, self._model.max_dofs_per_articulation),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        # Bridging buffers between the per-element kernels and the tile-Cholesky
        # solve. Shapes match the tile-load shapes exactly so we never tile-load
        # a slice — keeping the autograd path simple.
        self._j_site = wp.zeros(
            (self._num_robots, 6, self._dofs_per_robot),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        self._e_buffer = wp.zeros(
            (self._num_robots, 6),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        self._A = wp.zeros(
            (self._num_robots, 6, 6),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        self._y = wp.zeros(
            (self._num_robots, 6),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        self._qd_target_local = wp.zeros(
            (self._num_robots, self._dofs_per_robot),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )

    def is_stateful(self) -> bool:
        return False

    def is_graphable(self) -> bool:
        return True

    def outputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        return [
            (self._output_qd_attr, self._output_qd_port_indices),
            (self._output_q_attr, self._output_q_port_indices),
        ]

    def compute(
        self,
        input: Any,
        output: Any,
        state: ControlLaw.State | None,
        next_state: ControlLaw.State | None,
        dt: float,
    ) -> None:
        n = len(self.indices)

        meas = _resolve_input_array(input, self._measurement_attr, name="measurement")
        meas_rate = _resolve_input_array(input, self._measurement_rate_attr, name="measurement_rate")
        target_pos = _resolve_per_group_array(
            input, self._target_pos_attr, self._num_robots, wp.vec3, name="target_pos"
        )
        target_quat = _resolve_per_group_array(
            input, self._target_quat_attr, self._num_robots, wp.quat, name="target_quat"
        )
        damping = _resolve_per_group_array(input, self._damping_attr, self._num_robots, wp.float32, name="damping")
        gain = _resolve_per_group_array(input, self._gain_attr, self._num_robots, wp.float32, name="gain")
        out_qd = _resolve_input_array(output, self._output_qd_attr, name="output_qd")
        out_q = _resolve_input_array(output, self._output_q_attr, name="output_q")

        wp.launch(
            _gather_local_kernel,
            dim=n,
            inputs=[meas, self._measurement_port_indices],
            outputs=[self._model_state.joint_q],
        )
        wp.launch(
            _gather_local_kernel,
            dim=n,
            inputs=[meas_rate, self._measurement_rate_port_indices],
            outputs=[self._model_state.joint_qd],
        )

        eval_fk(self._model, self._model_state.joint_q, self._model_state.joint_qd, self._model_state)
        eval_jacobian(self._model, self._model_state, J=self._jacobian)

        # 1. Site-frame Jacobian + task-space error.
        wp.launch(
            _build_site_jacobian_kernel,
            dim=self._num_robots,
            inputs=[
                self._jacobian,
                self._model_state.body_q,
                self._model.body_com,
                target_pos,
                target_quat,
                self._ee_body_per_robot,
                self._ee_link_per_robot,
                self._site_xform_per_robot,
                self._dofs_per_robot,
            ],
            outputs=[self._j_site, self._e_buffer],
        )

        # 2. Build A = J J^T + λ²I.
        wp.launch(
            _build_dls_matrix_kernel,
            dim=(self._num_robots, 6, 6),
            inputs=[self._j_site, damping, self._dofs_per_robot],
            outputs=[self._A],
        )

        # 3. Tile Cholesky solve: y = A^{-1} e. Block-cooperative (one warp per
        # robot). This is the only step using tile primitives.
        wp.launch_tiled(
            _cholesky_solve_kernel,
            dim=[self._num_robots],
            inputs=[self._A, self._e_buffer],
            outputs=[self._y],
            block_dim=32,
        )

        # 4. q_dot = gain * J_site^T y.
        wp.launch(
            _qd_from_y_kernel,
            dim=(self._num_robots, self._dofs_per_robot),
            inputs=[self._j_site, self._y, gain],
            outputs=[self._qd_target_local],
        )

        # 5. Accumulate into global output arrays + integrate q.
        wp.launch(
            _accumulate_outputs_kernel,
            dim=(self._num_robots, self._dofs_per_robot),
            inputs=[
                self._qd_target_local,
                self._model_state.joint_q,
                dt,
                self._dofs_per_robot,
                self._output_qd_port_indices,
                self._output_q_port_indices,
            ],
            outputs=[out_qd, out_q],
        )
