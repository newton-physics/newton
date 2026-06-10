# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ControlLawDifferentialIK — one-step damped-least-squares differential IK.

Stateless. For each robot in the batch, computes a joint-velocity command
that drives a user-defined site pose (position + orientation) toward a
target, using a damped pseudoinverse of the per-robot spatial Jacobian.

The DLS solve is split across four kernels so that the autograd-able tile
primitives (`wp.tile_cholesky`, `wp.tile_cholesky_solve`) only see pure
tile_load -> tile_cholesky -> tile_cholesky_solve -> tile_store flows — no
element-wise mutation, which would break Warp's adjoint. All other math
(building the site-frame Jacobian, forming A = J J^T + lambda^2 I,
back-projecting q_dot from y) lives in per-element kernels that are
autograd-friendly by construction.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ...sim.articulation import eval_fk, eval_jacobian
from ...sim.builder import ModelBuilder
from ..control_law import ControlLaw
from ..utils import HardwareInterface, _normalize_port, _resolve_input_array


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
    # The full spatial Jacobian for the model from ``eval_jacobian``,
    # shape (num_robots, max_links * 6, max_dofs). For each EE link, the 6
    # rows at j_row = ee_link * 6 are the COM-frame body twist
    # (v_com_world, omega_world).
    jacobian: wp.array3d[float],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    target_pos: wp.array[wp.vec3],
    target_pos_indices: wp.array[wp.uint32],
    target_quat: wp.array[wp.quat],
    target_quat_indices: wp.array[wp.uint32],
    # Per-robot site-lookup arrays (precomputed at finalize() from the
    # per-articulation site lookup). Each articulation may sit on a
    # different EE body and carry a different body-local site xform.
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

    # --- Task-space error e in R^6 ---
    pos_err = target_pos[target_pos_indices[r]] - site_pos
    # Orientation: q_err = target * conj(site) carries (cos(theta/2), sin(theta/2)*axis).
    # Doubling the vector part yields 2 sin(theta/2) axis ~ theta * axis for small theta;
    # multiplying by sign(q_err.w) selects the representative with positive
    # scalar part (the shorter rotation).
    q_err = target_quat[target_quat_indices[r]] * wp.quat_inverse(site_quat)
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
    damping: wp.array[float],
    damping_indices: wp.array[wp.uint32],
    dofs_per_robot: int,
    A: wp.array3d[float],  # (num_robots, 6, 6) — output, A = J J^T + lambda^2 I
):
    """Per-(robot, row, col): build the 6x6 SPD DLS matrix from J_site and lambda."""
    r, i, k = wp.tid()
    lam = damping[damping_indices[r]]
    lam_sq = lam * lam
    acc = float(0.0)
    for j in range(dofs_per_robot):
        acc += j_site[r, i, j] * j_site[r, k, j]
    if i == k:
        acc += lam_sq
    A[r, i, k] = acc


# NOTE on differentiability: ``wp.tile_cholesky`` and ``wp.tile_cholesky_solve``
# advertise registered adjoints, but as of Warp 1.14.0 the backward pass
# produces zero gradients on the input arrays in practice (verified with a
# standalone test on this kernel — forward correct, but gradients of both
# A and the rhs vector come back all-zero). Until that upstream gap is
# fixed we mark this kernel ``enable_backward=False``, which blocks
# gradient propagation at the solve. Every other kernel in the chain is
# autograd-able by default.
@wp.kernel(enable_backward=False)
def _cholesky_solve_kernel(
    A: wp.array3d[float],  # (num_robots, 6, 6)
    e_buffer: wp.array2d[float],  # (num_robots, 6)
    y: wp.array2d[float],  # (num_robots, 6) — output, solution of A y = e
):
    """Per-robot tile solve: y = A^{-1} e via Cholesky."""
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
    gain: wp.array[float],
    gain_indices: wp.array[wp.uint32],
    qd_target_local: wp.array2d[float],  # (num_robots, dofs_per_robot) — output
):
    """Per-(robot, dof_j): q_dot[j] = gain * sum_i J_site[i, j] * y[i]."""
    r, j = wp.tid()
    g = gain[gain_indices[r]]
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
    full configuration ``q``. Stateless. Drives the **site** pose in world
    frame toward the target. The site is identified by the ``label`` you
    gave it when calling :meth:`newton.ModelBuilder.add_site`; the
    controller looks up the EE link and the body-frame offset xform from
    the builder by that label.

    **Tape-safe, forward-only through the solve.** Every kernel in the
    chain except the DLS solve itself is autograd-able by default. The
    solve uses ``wp.tile_cholesky`` / ``wp.tile_cholesky_solve``; those
    primitives' docstrings advertise registered adjoints, but the backward
    path returns zero gradients in Warp 1.14.0. The solve kernel is
    therefore marked ``enable_backward=False`` to make the zero-gradient
    behaviour explicit. Revisit when upstream tile_cholesky backward
    lands.

    Solve form (per robot, ``J_site`` is the 6xN site-frame Jacobian)::

        e        = [target_pos - site_pos ;  2 * sign(q_err.w) * q_err.xyz]
        A        = J_site J_site^T + lambda^2 * I_6                              (6x6 SPD)
        L L^T    = A                                                            (Cholesky)
        L L^T y  = e                                                            (solve)
        q_dot    = gain * J_site^T y

    where ``q_err = target_quat * conj(site_quat)``. ``output_q`` is
    written as ``q_current + q_dot * dt``.

    Construction takes a :class:`newton.ModelBuilder` containing exactly
    ``N`` topologically-identical articulations — the N robots this
    controller manages. The N articulations must share DOF count,
    link/joint count, and joint types; they may differ in physical
    parameters (mass, inertia, joint limits) and in per-articulation site
    placement. The controller does no replication: if the user wants R
    copies of a single-robot template, they call
    :meth:`newton.ModelBuilder.replicate` themselves before passing the
    builder in.

    The controller assumes:

    - All joints in the builder are **scalar-DOF** (revolute or
      prismatic), so ``joint_q.shape == joint_qd.shape == (joint_dof_count,)``.
    - The intended "world frame" is the base frame of the builder.

    Args:
        model_builder: Unfinalized N-articulation builder. ``num_robots =
            model_builder.articulation_count``.
        site: Label of the site (added via
            :meth:`newton.ModelBuilder.add_site`) to drive. The builder
            must contain **exactly one** site with this label on each
            articulation (``N`` sites total). The N occurrences may sit on
            different bodies and carry different body-local xforms.
        measurement: Per-DOF read port ``(ControlSignal, port_indices)``.
            Source of joint positions ``q``. ``port_indices`` length
            ``num_robots * dofs_per_robot``, laid out
            ``[r0_d0, r0_d1, ..., r1_d0, ...]``.
        measurement_rate: Per-DOF read port. Source of joint velocities
            ``q_dot`` (used by ``eval_fk`` to populate ``body_qd``; the
            solve uses position-only error).
        target_pos: Per-robot read port ``(ControlSignal, robot_indices)``.
            The kernel reads ``arr[robot_indices[r]]``; the bound
            signal's ``dtype`` should be ``wp.vec3``. Site position
            target in world frame.
        target_quat: Per-robot read port. Signal dtype ``wp.quat``. Site
            orientation target.
        damping: Per-robot read port. Signal dtype ``wp.float32``. DLS
            ``lambda`` per robot.
        gain: Per-robot read port. Signal dtype ``wp.float32``. Scalar
            multiplier applied to the DLS-solve output before writing
            into ``output_qd`` / integrating into ``output_q``.
        output_qd: Per-DOF write port. Destination for ``q_dot``
            (accumulated ``+=``).
        output_q: Per-DOF write port. Destination for
            ``q_current + q_dot * dt`` (accumulated ``+=``).
    """

    INPUT_PORTS = frozenset(
        {
            "measurement",
            "measurement_rate",
            "target_pos",
            "target_quat",
            "damping",
            "gain",
        }
    )
    OUTPUT_PORTS = frozenset({"output_qd", "output_q"})

    def __init__(
        self,
        *,
        model_builder: ModelBuilder,
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
        n = model_builder.articulation_count
        if n < 1:
            raise ValueError("ControlLawDifferentialIK: model_builder has no articulations.")
        if model_builder.joint_dof_count % n != 0:
            raise ValueError(
                f"ControlLawDifferentialIK: model_builder.joint_dof_count={model_builder.joint_dof_count} "
                f"is not divisible by articulation_count={n}; the N articulations must share DOF count."
            )
        self._template = model_builder
        self._num_robots = n
        self._dofs_per_robot = model_builder.joint_dof_count // n

        # Per-articulation site lookup. The user gives one ``site`` label
        # that's expected to appear on every one of the N articulations;
        # each occurrence may sit on a different body and have a different
        # body-local xform. We pick out one shape per articulation and
        # stash its (within-articulation body index, body-local xform)
        # pair. `finalize()` later turns these into length-num_robots Warp
        # arrays keyed by robot index.
        #
        # Bodies are assumed grouped by articulation in the builder
        # (articulation 0's bodies first, then articulation 1's, …) —
        # the natural pattern when the user calls add_link / add_joint /
        # add_articulation one articulation at a time, or via
        # ModelBuilder.replicate. The check below enforces this
        # implicitly: if a site lands on a body outside its articulation's
        # block, we'll get duplicate articulation assignments and raise.
        body_count_in_template = len(model_builder.body_q)
        if body_count_in_template % n != 0:
            raise ValueError(
                f"ControlLawDifferentialIK: template body_count={body_count_in_template} is not "
                f"divisible by articulation_count={n}; the N articulations must share body count."
            )
        bodies_per_robot = body_count_in_template // n
        all_site_idxs = [i for i, lbl in enumerate(model_builder.shape_label) if lbl == site]
        if len(all_site_idxs) != n:
            raise ValueError(
                f"ControlLawDifferentialIK: expected exactly {n} shapes with label '{site}' "
                f"(one per articulation), found {len(all_site_idxs)} in model_builder; "
                f"available labels: {model_builder.shape_label}."
            )
        site_for_robot: dict[int, tuple[int, wp.transform]] = {}
        for shape_idx in all_site_idxs:
            body_global = int(model_builder.shape_body[shape_idx])
            r = body_global // bodies_per_robot
            within_robot = body_global % bodies_per_robot
            if r in site_for_robot:
                raise ValueError(
                    f"ControlLawDifferentialIK: multiple shapes labeled '{site}' resolved to "
                    f"articulation {r}. Each articulation must contribute exactly one site with this label."
                )
            site_for_robot[r] = (within_robot, model_builder.shape_transform[shape_idx])
        for r in range(n):
            if r not in site_for_robot:
                raise ValueError(f"ControlLawDifferentialIK: articulation {r} has no shape labeled '{site}'.")
        # Stash as lists ordered by articulation; finalize() turns these
        # into length-num_robots Warp arrays.
        self._ee_link_per_robot_list: list[int] = [site_for_robot[r][0] for r in range(n)]
        self._site_xform_per_robot_list: list[wp.transform] = [site_for_robot[r][1] for r in range(n)]
        self._bodies_per_robot = bodies_per_robot

        # --- Port normalization ---
        # Per-robot ports: kernel reads arr[robot_indices[r]]; port_indices
        # must be length num_robots.
        def _norm_per_robot(spec, name):
            signal, idx = _normalize_port(spec, name=name)
            if idx.shape != (self._num_robots,):
                raise ValueError(f"Port '{name}': robot_indices shape {idx.shape} must equal ({self._num_robots},).")
            return signal, idx

        self._target_pos_signal, self._target_pos_indices = _norm_per_robot(target_pos, "target_pos")
        self._target_quat_signal, self._target_quat_indices = _norm_per_robot(target_quat, "target_quat")
        self._damping_signal, self._damping_indices = _norm_per_robot(damping, "damping")
        self._gain_signal, self._gain_indices = _norm_per_robot(gain, "gain")

        # Per-DOF ports: output_qd defines num_outputs; everything else
        # must match. num_outputs == num_robots * dofs_per_robot for DiffIK.
        self._output_qd_signal, self._output_qd_port_indices = _normalize_port(output_qd, name="output_qd")
        expected_n_outputs = self._num_robots * self._dofs_per_robot
        if self._output_qd_port_indices.shape != (expected_n_outputs,):
            raise ValueError(
                f"Port 'output_qd': port_indices shape {self._output_qd_port_indices.shape} must equal "
                f"({expected_n_outputs},) (num_robots * dofs_per_robot)."
            )
        self._num_outputs = expected_n_outputs

        def _norm_per_dof(spec, name):
            signal, idx = _normalize_port(spec, name=name)
            if idx.shape != self._output_qd_port_indices.shape:
                raise ValueError(
                    f"Port '{name}': port_indices shape {idx.shape} must match "
                    f"output_qd's shape {self._output_qd_port_indices.shape}."
                )
            return signal, idx

        self._measurement_signal, self._measurement_port_indices = _norm_per_dof(measurement, "measurement")
        self._measurement_rate_signal, self._measurement_rate_port_indices = _norm_per_dof(
            measurement_rate, "measurement_rate"
        )
        self._output_q_signal, self._output_q_port_indices = _norm_per_dof(output_q, "output_q")

        self._used_inputs = frozenset(
            {
                self._measurement_signal,
                self._measurement_rate_signal,
                self._target_pos_signal,
                self._target_quat_signal,
                self._damping_signal,
                self._gain_signal,
            }
        )
        self._used_outputs = frozenset({self._output_qd_signal, self._output_q_signal})

        # Resolved attribute names — filled by _resolve().
        self._measurement_attr: str | None = None
        self._measurement_rate_attr: str | None = None
        self._target_pos_attr: str | None = None
        self._target_quat_attr: str | None = None
        self._damping_attr: str | None = None
        self._gain_attr: str | None = None
        self._output_qd_attr: str | None = None
        self._output_q_attr: str | None = None

    def _resolve(self, hw: HardwareInterface) -> None:
        self._measurement_attr = hw.inputs[self._measurement_signal]
        self._measurement_rate_attr = hw.inputs[self._measurement_rate_signal]
        self._target_pos_attr = hw.inputs[self._target_pos_signal]
        self._target_quat_attr = hw.inputs[self._target_quat_signal]
        self._damping_attr = hw.inputs[self._damping_signal]
        self._gain_attr = hw.inputs[self._gain_signal]
        self._output_qd_attr = hw.outputs[self._output_qd_signal]
        self._output_q_attr = hw.outputs[self._output_q_signal]

    def finalize(self, device: wp.Device, requires_grad: bool = False) -> None:
        # Finalize the user's N-articulation builder directly. No
        # replication: the builder is already the N-robot model the
        # controller manages.
        self._model = self._template.finalize(device=device, requires_grad=requires_grad)
        self._model_state = self._model.state()

        # Fan the per-articulation site data out to length-num_robots
        # Warp arrays keyed by robot index. ee_body is the global body
        # index (robot r's local ee_link plus the offset for r's body
        # block).
        ee_link_per_robot = self._ee_link_per_robot_list
        ee_body_per_robot = [r * self._bodies_per_robot + ee_link_per_robot[r] for r in range(self._num_robots)]
        # These are build-time constants — no gradient flow through them.
        self._ee_body_per_robot = wp.array(ee_body_per_robot, dtype=int, device=device)
        self._ee_link_per_robot = wp.array(ee_link_per_robot, dtype=int, device=device)
        self._site_xform_per_robot = wp.array(self._site_xform_per_robot_list, dtype=wp.transform, device=device)

        self._jacobian = wp.zeros(
            (self._num_robots, self._model.max_joints_per_articulation * 6, self._model.max_dofs_per_articulation),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        # Bridging buffers between the per-element kernels and the
        # tile-Cholesky solve. Shapes match the tile-load shapes exactly
        # so we never tile-load a slice — keeps the autograd path simple.
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

    def inputs(self) -> list[tuple[str, wp.array[wp.uint32]]]:
        return [
            (self._measurement_attr, self._measurement_port_indices),
            (self._measurement_rate_attr, self._measurement_rate_port_indices),
            (self._target_pos_attr, self._target_pos_indices),
            (self._target_quat_attr, self._target_quat_indices),
            (self._damping_attr, self._damping_indices),
            (self._gain_attr, self._gain_indices),
        ]

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
        meas = _resolve_input_array(input, self._measurement_attr, name="measurement")
        meas_rate = _resolve_input_array(input, self._measurement_rate_attr, name="measurement_rate")
        target_pos = _resolve_input_array(input, self._target_pos_attr, name="target_pos")
        target_quat = _resolve_input_array(input, self._target_quat_attr, name="target_quat")
        damping = _resolve_input_array(input, self._damping_attr, name="damping")
        gain = _resolve_input_array(input, self._gain_attr, name="gain")
        out_qd = _resolve_input_array(output, self._output_qd_attr, name="output_qd")
        out_q = _resolve_input_array(output, self._output_q_attr, name="output_q")

        wp.launch(
            _gather_local_kernel,
            dim=self._num_outputs,
            inputs=[meas, self._measurement_port_indices],
            outputs=[self._model_state.joint_q],
        )
        wp.launch(
            _gather_local_kernel,
            dim=self._num_outputs,
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
                self._target_pos_indices,
                target_quat,
                self._target_quat_indices,
                self._ee_body_per_robot,
                self._ee_link_per_robot,
                self._site_xform_per_robot,
                self._dofs_per_robot,
            ],
            outputs=[self._j_site, self._e_buffer],
        )

        # 2. Build A = J J^T + lambda^2 I.
        wp.launch(
            _build_dls_matrix_kernel,
            dim=(self._num_robots, 6, 6),
            inputs=[self._j_site, damping, self._damping_indices, self._dofs_per_robot],
            outputs=[self._A],
        )

        # 3. Tile Cholesky solve: y = A^{-1} e. Block-cooperative (one warp
        # per robot). This is the only step using tile primitives.
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
            inputs=[self._j_site, self._y, gain, self._gain_indices],
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
