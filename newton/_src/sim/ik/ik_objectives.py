# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Objective definitions for inverse kinematics."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..model import Model
from .ik_common import IKJacobianType


class IKObjective:
    """Base class for inverse-kinematics objectives.

    Each objective contributes one or more residual rows to the global IK
    system and can optionally provide an analytic Jacobian. Objective
    instances are shared across a batch of problems, so per-problem data such
    as targets should live in device arrays and be indexed through the
    ``problem_idx`` mapping supplied at evaluation time.

    Subclasses should override :meth:`residual_dim`,
    :meth:`compute_residuals`, and :meth:`compute_jacobian_autodiff`. They can
    additionally override :meth:`supports_analytic` and
    :meth:`compute_jacobian_analytic` when an analytic Jacobian is available.
    """

    def __init__(self) -> None:
        # Optimizers assign these before first use via set_batch_layout().
        self.total_residuals = None
        self.residual_offset = None
        self.n_batch = None

    def set_batch_layout(self, total_residuals: int, residual_offset: int, n_batch: int) -> None:
        """Register this objective's rows inside the optimizer's global system.

        Args:
            total_residuals: Total number of residual rows across all
                objectives.
            residual_offset: First row reserved for this objective inside the
                global residual vector and Jacobian.
            n_batch: Number of evaluation rows processed together. This is the
                expanded solver batch, not necessarily the number of base IK
                problems.

        .. note::
            Per-problem buffers such as targets should still be sized by the
            base problem count and accessed through the ``problem_idx`` mapping
            supplied during residual and Jacobian evaluation.
        """
        self.total_residuals = total_residuals
        self.residual_offset = residual_offset
        self.n_batch = n_batch

    def _require_batch_layout(self) -> None:
        if self.total_residuals is None or self.residual_offset is None or self.n_batch is None:
            raise RuntimeError(f"Batch layout not assigned for {type(self).__name__}; call set_batch_layout() first")

    def residual_dim(self) -> int:
        """Return the number of residual rows contributed by this objective."""
        raise NotImplementedError

    def compute_residuals(
        self,
        body_q: wp.array2d(dtype=wp.transform),
        joint_q: wp.array2d(dtype=wp.float32),
        model: Model,
        residuals: wp.array2d(dtype=wp.float32),
        start_idx: int,
        problem_idx: wp.array(dtype=wp.int32),
    ) -> None:
        """Write this objective's residual block into a global buffer.

        Args:
            body_q: Batched body transforms for the current evaluation rows,
                shape [n_batch, body_count].
            joint_q: Batched joint coordinates [m or rad] for the current
                evaluation rows, shape [n_batch, joint_coord_count].
            model: Shared articulation model.
            residuals: Global residual buffer that receives this objective's
                residual rows, shape [n_batch, total_residual_count].
            start_idx: First residual row reserved for this objective inside
                the global residual buffer.
            problem_idx: Mapping from evaluation rows to base problem
                indices, shape [n_batch]. Use this when objective data is
                stored once per original problem but the solver expands rows
                for multiple seeds or line-search candidates.
        """
        raise NotImplementedError

    def compute_jacobian_autodiff(
        self,
        tape: wp.Tape,
        model: Model,
        jacobian: wp.array3d(dtype=wp.float32),
        start_idx: int,
        dq_dof: wp.array2d(dtype=wp.float32),
    ) -> None:
        """Fill this objective's Jacobian block with autodiff gradients.

        Args:
            tape: Recorded Warp tape whose output is the global residual
                buffer.
            model: Shared articulation model.
            jacobian: Global Jacobian buffer to update,
                shape [n_batch, total_residual_count, joint_dof_count].
            start_idx: First residual row reserved for this objective.
            dq_dof: Differentiable joint update variable for the current
                batch, shape [n_batch, joint_dof_count].
        """
        raise NotImplementedError

    def supports_analytic(self) -> bool:
        """Return ``True`` when this objective implements an analytic Jacobian."""
        return False

    def bind_device(self, device: wp.DeviceLike) -> None:
        """Bind this objective to the Warp device used by the solver."""
        self.device = device

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        """Allocate any per-objective buffers needed by the chosen backend.

        Args:
            model: Shared articulation model.
            jacobian_mode: Jacobian backend that will evaluate this
                objective.
        """
        pass

    def compute_jacobian_analytic(
        self,
        body_q: wp.array2d(dtype=wp.transform),
        joint_q: wp.array2d(dtype=wp.float32),
        model: Model,
        jacobian: wp.array3d(dtype=wp.float32),
        joint_S_s: wp.array2d(dtype=wp.spatial_vector),
        start_idx: int,
    ) -> None:
        """Fill this objective's Jacobian block analytically, if supported.

        Args:
            body_q: Batched body transforms for the current evaluation rows,
                shape [n_batch, body_count].
            joint_q: Batched joint coordinates for the current evaluation rows,
                shape [n_batch, joint_coord_count].
            model: Shared articulation model.
            jacobian: Global Jacobian buffer to update,
                shape [n_batch, total_residual_count, joint_dof_count].
            joint_S_s: Batched motion-subspace columns,
                shape [n_batch, joint_dof_count].
            start_idx: First residual row reserved for this objective.
        """
        pass


@wp.kernel
def _pos_residuals(
    body_q: wp.array2d(dtype=wp.transform),  # (n_batch, n_bodies)
    target_pos: wp.array1d(dtype=wp.vec3),  # (n_problems)
    link_index: int,
    link_offset: wp.vec3,
    start_idx: int,
    weight: float,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    # outputs
    residuals: wp.array2d(dtype=wp.float32),  # (n_batch, n_residuals)
):
    row = wp.tid()
    base = problem_idx_map[row]

    body_tf = body_q[row, link_index]
    ee_pos = wp.transform_point(body_tf, link_offset)

    error = target_pos[base] - ee_pos
    residuals[row, start_idx + 0] = weight * error[0]
    residuals[row, start_idx + 1] = weight * error[1]
    residuals[row, start_idx + 2] = weight * error[2]


@wp.kernel
def _pos_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
    n_dofs: int,
    start_idx: int,
    component: int,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),  # (n_batch, n_residuals, n_dofs)
):
    problem_idx = wp.tid()
    residual_idx = start_idx + component

    for j in range(n_dofs):
        jacobian[problem_idx, residual_idx, j] = q_grad[problem_idx, j]


@wp.kernel
def _update_position_target(
    problem_idx: int,
    new_position: wp.vec3,
    # outputs
    target_array: wp.array1d(dtype=wp.vec3),  # (n_problems)
):
    target_array[problem_idx] = new_position


@wp.kernel
def _update_position_targets(
    new_positions: wp.array1d(dtype=wp.vec3),  # (n_problems)
    # outputs
    target_array: wp.array1d(dtype=wp.vec3),  # (n_problems)
):
    problem_idx = wp.tid()
    target_array[problem_idx] = new_positions[problem_idx]


@wp.kernel
def _pos_jac_analytic(
    link_index: int,
    link_offset: wp.vec3,
    affects_dof: wp.array1d(dtype=wp.uint8),  # (n_dofs)
    body_q: wp.array2d(dtype=wp.transform),  # (n_batch, n_bodies)
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),  # (n_batch, n_dofs)
    start_idx: int,
    n_dofs: int,
    weight: float,
    # output
    jacobian: wp.array3d(dtype=wp.float32),  # (n_batch, n_residuals, n_dofs)
):
    # one thread per (problem, dof)
    problem_idx, dof_idx = wp.tid()

    # skip if this DoF cannot move the EE
    if affects_dof[dof_idx] == 0:
        return

    # world-space EE position
    body_tf = body_q[problem_idx, link_index]
    rot_w = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    pos_w = wp.vec3(body_tf[0], body_tf[1], body_tf[2])
    ee_pos_world = pos_w + wp.quat_rotate(rot_w, link_offset)

    # motion sub-space column S
    S = joint_S_s[problem_idx, dof_idx]
    v_orig = wp.vec3(S[0], S[1], S[2])
    omega = wp.vec3(S[3], S[4], S[5])
    v_ee = v_orig + wp.cross(omega, ee_pos_world)

    # write three Jacobian rows (x, y, z)
    jacobian[problem_idx, start_idx + 0, dof_idx] = -weight * v_ee[0]
    jacobian[problem_idx, start_idx + 1, dof_idx] = -weight * v_ee[1]
    jacobian[problem_idx, start_idx + 2, dof_idx] = -weight * v_ee[2]


class IKObjectivePosition(IKObjective):
    """Match the world-space position of a point on a link.

    Args:
        link_index: Body index whose frame defines the constrained link.
        link_offset: Point in the link's local frame [m].
        target_positions: Target positions [m], shape [problem_count].
        weight: Scalar multiplier applied to the residual and Jacobian rows.
    """

    def __init__(
        self,
        link_index: int,
        link_offset: wp.vec3,
        target_positions: wp.array(dtype=wp.vec3),
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.link_index = link_index
        self.link_offset = link_offset
        self.target_positions = target_positions
        self.weight = weight

        self.affects_dof = None
        self.e_arrays = None

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        """Initialize caches used by analytic or autodiff Jacobian evaluation.

        Args:
            model: Shared articulation model.
            jacobian_mode: Jacobian backend selected for this objective.
        """
        self._require_batch_layout()
        if jacobian_mode == IKJacobianType.ANALYTIC:
            joint_qd_start_np = model.joint_qd_start.numpy()
            dof_to_joint_np = np.empty(joint_qd_start_np[-1], dtype=np.int32)
            for j in range(len(joint_qd_start_np) - 1):
                dof_to_joint_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]] = j

            links_per_problem = model.body_count
            joint_child_np = model.joint_child.numpy()
            body_to_joint_np = np.full(links_per_problem, -1, np.int32)
            for j in range(model.joint_count):
                child = joint_child_np[j]
                if child != -1:
                    body_to_joint_np[child] = j

            joint_q_start_np = model.joint_q_start.numpy()
            ancestors = np.zeros(len(joint_q_start_np) - 1, dtype=bool)
            joint_parent_np = model.joint_parent.numpy()
            body = self.link_index
            while body != -1:
                j = body_to_joint_np[body]
                if j != -1:
                    ancestors[j] = True
                body = joint_parent_np[j] if j != -1 else -1
            affects_dof_np = ancestors[dof_to_joint_np]
            self.affects_dof = wp.array(affects_dof_np.astype(np.uint8), device=self.device)
        elif jacobian_mode == IKJacobianType.AUTODIFF:
            self.e_arrays = []
            for component in range(3):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_batch):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self) -> bool:
        """Return ``True`` because this objective has an analytic Jacobian."""
        return True

    def set_target_position(self, problem_idx: int, new_position: wp.vec3) -> None:
        """Update the target position for a single base IK problem.

        Args:
            problem_idx: Base problem index to update.
            new_position: Replacement target position [m] in world
                coordinates.
        """
        self._require_batch_layout()
        wp.launch(
            _update_position_target,
            dim=1,
            inputs=[problem_idx, new_position],
            outputs=[self.target_positions],
            device=self.device,
        )

    def set_target_positions(self, new_positions: wp.array(dtype=wp.vec3)) -> None:
        """Replace the target positions for all base IK problems.

        Args:
            new_positions: Target positions [m], shape [problem_count].
        """
        self._require_batch_layout()
        expected = self.target_positions.shape[0]
        if new_positions.shape[0] != expected:
            raise ValueError(f"Expected {expected} target positions, got {new_positions.shape[0]}")
        wp.launch(
            _update_position_targets,
            dim=expected,
            inputs=[new_positions],
            outputs=[self.target_positions],
            device=self.device,
        )

    def residual_dim(self) -> int:
        """Return the three translational residual rows for this objective."""
        return 3

    def compute_residuals(
        self,
        body_q: wp.array2d(dtype=wp.transform),
        joint_q: wp.array2d(dtype=wp.float32),
        model: Model,
        residuals: wp.array2d(dtype=wp.float32),
        start_idx: int,
        problem_idx: wp.array(dtype=wp.int32),
    ) -> None:
        """Write weighted position errors into the global residual buffer.

        Args:
            body_q: Batched body transforms for the evaluation rows,
                shape [n_batch, body_count].
            joint_q: Batched joint coordinates, shape [n_batch, joint_coord_count].
                Present for interface compatibility and not used directly by
                this objective.
            model: Shared articulation model. Present for interface
                compatibility.
            residuals: Global residual buffer to update,
                shape [n_batch, total_residual_count].
            start_idx: First residual row reserved for this objective.
            problem_idx: Mapping from evaluation rows to base problem
                indices, shape [n_batch], used to fetch ``target_positions``.
        """
        count = body_q.shape[0]
        wp.launch(
            _pos_residuals,
            dim=count,
            inputs=[
                body_q,
                self.target_positions,
                self.link_index,
                self.link_offset,
                start_idx,
                self.weight,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(
        self,
        tape: wp.Tape,
        model: Model,
        jacobian: wp.array3d(dtype=wp.float32),
        start_idx: int,
        dq_dof: wp.array2d(dtype=wp.float32),
    ) -> None:
        """Fill the position Jacobian block using Warp autodiff.

        Args:
            tape: Recorded Warp tape whose output is the global residual
                buffer.
            model: Shared articulation model.
            jacobian: Global Jacobian buffer to update,
                shape [n_batch, total_residual_count, joint_dof_count].
            start_idx: First residual row reserved for this objective.
            dq_dof: Differentiable joint update variable for the current
                batch, shape [n_batch, joint_dof_count].
        """
        self._require_batch_layout()
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})

            q_grad = tape.gradients[dq_dof]

            n_dofs = model.joint_dof_count

            wp.launch(
                _pos_jac_fill,
                dim=self.n_batch,
                inputs=[
                    q_grad,
                    n_dofs,
                    start_idx,
                    component,
                ],
                outputs=[
                    jacobian,
                ],
                device=self.device,
            )

            tape.zero()

    def compute_jacobian_analytic(
        self,
        body_q: wp.array2d(dtype=wp.transform),
        joint_q: wp.array2d(dtype=wp.float32),
        model: Model,
        jacobian: wp.array3d(dtype=wp.float32),
        joint_S_s: wp.array2d(dtype=wp.spatial_vector),
        start_idx: int,
    ) -> None:
        """Fill the position Jacobian block from the analytic motion subspace.

        Args:
            body_q: Batched body transforms for the evaluation rows,
                shape [n_batch, body_count].
            joint_q: Batched joint coordinates, shape [n_batch, joint_coord_count].
                Present for interface compatibility and not used directly by
                this objective.
            model: Shared articulation model.
            jacobian: Global Jacobian buffer to update,
                shape [n_batch, total_residual_count, joint_dof_count].
            joint_S_s: Batched motion-subspace columns,
                shape [n_batch, joint_dof_count].
            start_idx: First residual row reserved for this objective.
        """
        n_dofs = model.joint_dof_count

        wp.launch(
            _pos_jac_analytic,
            dim=[body_q.shape[0], n_dofs],
            inputs=[
                self.link_index,
                self.link_offset,
                self.affects_dof,
                body_q,
                joint_S_s,
                start_idx,
                n_dofs,
                self.weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _limit_residuals(
    joint_q: wp.array2d(dtype=wp.float32),  # (n_batch, n_coords)
    joint_limit_lower: wp.array1d(dtype=wp.float32),  # (n_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32),  # (n_dofs)
    dof_to_coord: wp.array1d(dtype=wp.int32),  # (n_dofs)
    n_dofs: int,
    weight: float,
    start_idx: int,
    # outputs
    residuals: wp.array2d(dtype=wp.float32),  # (n_batch, n_residuals)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]

    if coord_idx < 0:
        return

    q = joint_q[problem, coord_idx]
    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]

    # treat huge ranges as no limit
    if upper - lower > 9.9e5:
        return

    viol = wp.max(0.0, q - upper) + wp.max(0.0, lower - q)
    residuals[problem, start_idx + dof_idx] = weight * viol


@wp.kernel
def _limit_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
    n_dofs: int,
    start_idx: int,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),
):
    problem_idx, dof_idx = wp.tid()

    jacobian[problem_idx, start_idx + dof_idx, dof_idx] = q_grad[problem_idx, dof_idx]


@wp.kernel
def _limit_jac_analytic(
    joint_q: wp.array2d(dtype=wp.float32),  # (n_batch, n_coords)
    joint_limit_lower: wp.array1d(dtype=wp.float32),  # (n_dofs)
    joint_limit_upper: wp.array1d(dtype=wp.float32),  # (n_dofs)
    dof_to_coord: wp.array1d(dtype=wp.int32),  # (n_dofs)
    n_dofs: int,
    start_idx: int,
    weight: float,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),  # (n_batch, n_residuals, n_dofs)
):
    problem, dof_idx = wp.tid()
    coord_idx = dof_to_coord[dof_idx]

    if coord_idx < 0:
        return

    q = joint_q[problem, coord_idx]
    lower = joint_limit_lower[dof_idx]
    upper = joint_limit_upper[dof_idx]

    if upper - lower > 9.9e5:
        return

    grad = float(0.0)
    if q >= upper:
        grad = weight
    elif q <= lower:
        grad = -weight

    jacobian[problem, start_idx + dof_idx, dof_idx] = grad


class IKObjectiveJointLimit(IKObjective):
    """Penalize violations of per-DoF joint limits.

    Each DoF contributes one residual row whose value is zero inside the valid
    range and increases linearly once the coordinate exceeds its lower or upper
    bound.

    Args:
        joint_limit_lower: Lower joint limits [m or rad], shape
            [joint_dof_count].
        joint_limit_upper: Upper joint limits [m or rad], shape
            [joint_dof_count].
        weight: Scalar multiplier applied to each limit-violation residual
            row.
    """

    def __init__(
        self,
        joint_limit_lower: wp.array(dtype=wp.float32),
        joint_limit_upper: wp.array(dtype=wp.float32),
        weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.e_array = None
        self.weight = weight

        self.n_dofs = len(joint_limit_lower)

        self.dof_to_coord = None

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        """Initialize autodiff seeds and the DoF-to-coordinate lookup table.

        Args:
            model: Shared articulation model.
            jacobian_mode: Jacobian backend selected for this objective.
        """
        self._require_batch_layout()
        if jacobian_mode == IKJacobianType.AUTODIFF:
            e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
            for prob_idx in range(self.n_batch):
                for dof_idx in range(self.n_dofs):
                    e[prob_idx, self.residual_offset + dof_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32, device=self.device)

        dof_to_coord_np = np.full(self.n_dofs, -1, dtype=np.int32)
        q_start_np = model.joint_q_start.numpy()
        qd_start_np = model.joint_qd_start.numpy()
        joint_dof_dim_np = model.joint_dof_dim.numpy()
        for j in range(model.joint_count):
            dof0 = qd_start_np[j]
            coord0 = q_start_np[j]
            lin, ang = joint_dof_dim_np[j]  # (#transl, #rot)
            for k in range(lin + ang):
                dof_to_coord_np[dof0 + k] = coord0 + k
        self.dof_to_coord = wp.array(dof_to_coord_np, dtype=wp.int32, device=self.device)

    def supports_analytic(self) -> bool:
        """Return ``True`` because this objective has an analytic Jacobian."""
        return True

    def residual_dim(self) -> int:
        """Return one residual row per joint DoF."""
        return self.n_dofs

    def compute_residuals(
        self,
        body_q: wp.array2d(dtype=wp.transform),
        joint_q: wp.array2d(dtype=wp.float32),
        model: Model,
        residuals: wp.array2d(dtype=wp.float32),
        start_idx: int,
        problem_idx: wp.array(dtype=wp.int32),
    ) -> None:
        """Write weighted joint-limit violations into the global residual buffer.

        Args:
            body_q: Batched body transforms, shape [n_batch, body_count].
                Present for interface compatibility and not used directly by
                this objective.
            joint_q: Batched joint coordinates for the evaluation rows, shape
                [n_batch, joint_coord_count].
            model: Shared articulation model. Present for interface
                compatibility.
            residuals: Global residual buffer to update,
                shape [n_batch, total_residual_count].
            start_idx: First residual row reserved for this objective.
            problem_idx: Mapping from evaluation rows to base problems, shape
                [n_batch]. Ignored because joint limits are shared across all
                problems.
        """
        count = joint_q.shape[0]
        wp.launch(
            _limit_residuals,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.dof_to_coord,
                self.n_dofs,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(
        self,
        tape: wp.Tape,
        model: Model,
        jacobian: wp.array3d(dtype=wp.float32),
        start_idx: int,
        dq_dof: wp.array2d(dtype=wp.float32),
    ) -> None:
        """Fill the limit Jacobian block using Warp autodiff.

        Args:
            tape: Recorded Warp tape whose output is the global residual
                buffer.
            model: Shared articulation model.
            jacobian: Global Jacobian buffer to update,
                shape [n_batch, total_residual_count, joint_dof_count].
            start_idx: First residual row reserved for this objective.
            dq_dof: Differentiable joint update variable for the current
                batch, shape [n_batch, joint_dof_count].
        """
        self._require_batch_layout()
        tape.backward(grads={tape.outputs[0]: self.e_array})

        q_grad = tape.gradients[dq_dof]

        wp.launch(
            _limit_jac_fill,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                q_grad,
                self.n_dofs,
                start_idx,
            ],
            outputs=[jacobian],
            device=self.device,
        )

    def compute_jacobian_analytic(
        self,
        body_q: wp.array2d(dtype=wp.transform),
        joint_q: wp.array2d(dtype=wp.float32),
        model: Model,
        jacobian: wp.array3d(dtype=wp.float32),
        joint_S_s: wp.array2d(dtype=wp.spatial_vector),
        start_idx: int,
    ) -> None:
        """Fill the limit Jacobian block with the piecewise-linear derivative.

        Args:
            body_q: Batched body transforms, shape [n_batch, body_count].
                Present for interface compatibility and not used directly by
                this objective.
            joint_q: Batched joint coordinates for the evaluation rows, shape
                [n_batch, joint_coord_count].
            model: Shared articulation model.
            jacobian: Global Jacobian buffer to update,
                shape [n_batch, total_residual_count, joint_dof_count].
            joint_S_s: Batched motion-subspace columns, shape [n_batch, joint_dof_count].
                Ignored because the joint-limit derivative is diagonal in
                joint coordinates.
            start_idx: First residual row reserved for this objective.
        """
        count = joint_q.shape[0]
        wp.launch(
            _limit_jac_analytic,
            dim=[count, self.n_dofs],
            inputs=[
                joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.dof_to_coord,
                self.n_dofs,
                start_idx,
                self.weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )


@wp.kernel
def _rot_residuals(
    body_q: wp.array2d(dtype=wp.transform),  # (n_batch, n_bodies)
    target_rot: wp.array1d(dtype=wp.vec4),  # (n_problems)
    link_index: int,
    link_offset_rotation: wp.quat,
    canonicalize_quat_err: wp.bool,
    start_idx: int,
    weight: float,
    problem_idx_map: wp.array1d(dtype=wp.int32),
    # outputs
    residuals: wp.array2d(dtype=wp.float32),  # (n_batch, n_residuals)
):
    row = wp.tid()
    base = problem_idx_map[row]

    body_tf = body_q[row, link_index]
    body_rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])

    actual_rot = body_rot * link_offset_rotation

    target_quat_vec = target_rot[base]
    target_quat = wp.quat(target_quat_vec[0], target_quat_vec[1], target_quat_vec[2], target_quat_vec[3])

    q_err = actual_rot * wp.quat_inverse(target_quat)
    if canonicalize_quat_err and wp.dot(actual_rot, target_quat) < 0.0:
        q_err = -q_err

    v_norm = wp.sqrt(q_err[0] * q_err[0] + q_err[1] * q_err[1] + q_err[2] * q_err[2])

    angle = 2.0 * wp.atan2(v_norm, q_err[3])

    eps = float(1e-8)
    axis_angle = wp.vec3(0.0, 0.0, 0.0)

    if v_norm > eps:
        axis = wp.vec3(q_err[0] / v_norm, q_err[1] / v_norm, q_err[2] / v_norm)
        axis_angle = axis * angle
    else:
        axis_angle = wp.vec3(2.0 * q_err[0], 2.0 * q_err[1], 2.0 * q_err[2])

    residuals[row, start_idx + 0] = weight * axis_angle[0]
    residuals[row, start_idx + 1] = weight * axis_angle[1]
    residuals[row, start_idx + 2] = weight * axis_angle[2]


@wp.kernel
def _rot_jac_fill(
    q_grad: wp.array2d(dtype=wp.float32),  # (n_batch, n_dofs)
    n_dofs: int,
    start_idx: int,
    component: int,
    # outputs
    jacobian: wp.array3d(dtype=wp.float32),  # (n_batch, n_residuals, n_dofs)
):
    problem_idx = wp.tid()

    residual_idx = start_idx + component

    for j in range(n_dofs):
        jacobian[problem_idx, residual_idx, j] = q_grad[problem_idx, j]


@wp.kernel
def _update_rotation_target(
    problem_idx: int,
    new_rotation: wp.vec4,
    # outputs
    target_array: wp.array1d(dtype=wp.vec4),  # (n_problems)
):
    target_array[problem_idx] = new_rotation


@wp.kernel
def _update_rotation_targets(
    new_rotation: wp.array1d(dtype=wp.vec4),  # (n_problems)
    # outputs
    target_array: wp.array1d(dtype=wp.vec4),  # (n_problems)
):
    problem_idx = wp.tid()
    target_array[problem_idx] = new_rotation[problem_idx]


@wp.kernel
def _rot_jac_analytic(
    affects_dof: wp.array1d(dtype=wp.uint8),  # (n_dofs)
    joint_S_s: wp.array2d(dtype=wp.spatial_vector),  # (n_batch, n_dofs)
    start_idx: int,  # first residual row for this objective
    n_dofs: int,  # width of the global Jacobian
    weight: float,
    # output
    jacobian: wp.array3d(dtype=wp.float32),  # (n_batch, n_residuals, n_dofs)
):
    # one thread per (problem, dof)
    problem_idx, dof_idx = wp.tid()

    # skip if this DoF cannot influence the EE rotation
    if affects_dof[dof_idx] == 0:
        return

    # ω column from motion sub-space
    S = joint_S_s[problem_idx, dof_idx]
    omega = wp.vec3(S[3], S[4], S[5])

    # write three Jacobian rows (rx, ry, rz)
    jacobian[problem_idx, start_idx + 0, dof_idx] = weight * omega[0]
    jacobian[problem_idx, start_idx + 1, dof_idx] = weight * omega[1]
    jacobian[problem_idx, start_idx + 2, dof_idx] = weight * omega[2]


class IKObjectiveRotation(IKObjective):
    """Match the world-space orientation of a link frame.

    Args:
        link_index: Body index whose frame defines the constrained link.
        link_offset_rotation: Local rotation from the body frame to the
            constrained frame, stored in ``(x, y, z, w)`` order.
        target_rotations: Target orientations, shape [problem_count]. Each
            quaternion is stored in ``(x, y, z, w)`` order.
        canonicalize_quat_err: If ``True``, flip equivalent quaternions so the
            residual follows the short rotational arc.
        weight: Scalar multiplier applied to the residual and Jacobian rows.
    """

    def __init__(
        self,
        link_index: int,
        link_offset_rotation: wp.quat,
        target_rotations: wp.array(dtype=wp.vec4),
        canonicalize_quat_err: bool = True,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.link_index = link_index
        self.link_offset_rotation = link_offset_rotation
        self.target_rotations = target_rotations
        self.weight = weight
        self.canonicalize_quat_err = canonicalize_quat_err

        self.affects_dof = None
        self.e_arrays = None

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        """Initialize caches used by analytic or autodiff Jacobian evaluation.

        Args:
            model: Shared articulation model.
            jacobian_mode: Jacobian backend selected for this objective.
        """
        self._require_batch_layout()
        if jacobian_mode == IKJacobianType.ANALYTIC:
            joint_qd_start_np = model.joint_qd_start.numpy()
            dof_to_joint_np = np.empty(joint_qd_start_np[-1], dtype=np.int32)
            for j in range(len(joint_qd_start_np) - 1):
                dof_to_joint_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]] = j

            links_per_problem = model.body_count
            joint_child_np = model.joint_child.numpy()
            body_to_joint_np = np.full(links_per_problem, -1, np.int32)
            for j in range(model.joint_count):
                child = joint_child_np[j]
                if child != -1:
                    body_to_joint_np[child] = j

            joint_q_start_np = model.joint_q_start.numpy()
            ancestors = np.zeros(len(joint_q_start_np) - 1, dtype=bool)
            joint_parent_np = model.joint_parent.numpy()
            body = self.link_index
            while body != -1:
                j = body_to_joint_np[body]
                if j != -1:
                    ancestors[j] = True
                body = joint_parent_np[j] if j != -1 else -1
            affects_dof_np = ancestors[dof_to_joint_np]
            self.affects_dof = wp.array(affects_dof_np.astype(np.uint8), device=self.device)
        elif jacobian_mode == IKJacobianType.AUTODIFF:
            self.e_arrays = []
            for component in range(3):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_batch):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self) -> bool:
        """Return ``True`` because this objective has an analytic Jacobian."""
        return True

    def set_target_rotation(self, problem_idx: int, new_rotation: wp.vec4) -> None:
        """Update the target orientation for a single base IK problem.

        Args:
            problem_idx: Base problem index to update.
            new_rotation: Replacement target quaternion in ``(x, y, z, w)``
                order.
        """
        self._require_batch_layout()
        wp.launch(
            _update_rotation_target,
            dim=1,
            inputs=[problem_idx, new_rotation],
            outputs=[self.target_rotations],
            device=self.device,
        )

    def set_target_rotations(self, new_rotations: wp.array(dtype=wp.vec4)) -> None:
        """Replace the target orientations for all base IK problems.

        Args:
            new_rotations: Target quaternions, shape [problem_count], in
                ``(x, y, z, w)`` order.
        """
        self._require_batch_layout()
        expected = self.target_rotations.shape[0]
        if new_rotations.shape[0] != expected:
            raise ValueError(f"Expected {expected} target rotations, got {new_rotations.shape[0]}")
        wp.launch(
            _update_rotation_targets,
            dim=expected,
            inputs=[new_rotations],
            outputs=[self.target_rotations],
            device=self.device,
        )

    def residual_dim(self) -> int:
        """Return the three rotational residual rows for this objective."""
        return 3

    def compute_residuals(
        self,
        body_q: wp.array2d(dtype=wp.transform),
        joint_q: wp.array2d(dtype=wp.float32),
        model: Model,
        residuals: wp.array2d(dtype=wp.float32),
        start_idx: int,
        problem_idx: wp.array(dtype=wp.int32),
    ) -> None:
        """Write weighted orientation errors into the global residual buffer.

        Args:
            body_q: Batched body transforms for the evaluation rows,
                shape [n_batch, body_count].
            joint_q: Batched joint coordinates, shape [n_batch, joint_coord_count].
                Present for interface compatibility and not used directly by
                this objective.
            model: Shared articulation model. Present for interface
                compatibility.
            residuals: Global residual buffer to update,
                shape [n_batch, total_residual_count].
            start_idx: First residual row reserved for this objective.
            problem_idx: Mapping from evaluation rows to base problem
                indices, shape [n_batch], used to fetch ``target_rotations``.
        """
        count = body_q.shape[0]
        wp.launch(
            _rot_residuals,
            dim=count,
            inputs=[
                body_q,
                self.target_rotations,
                self.link_index,
                self.link_offset_rotation,
                self.canonicalize_quat_err,
                start_idx,
                self.weight,
                problem_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_autodiff(
        self,
        tape: wp.Tape,
        model: Model,
        jacobian: wp.array3d(dtype=wp.float32),
        start_idx: int,
        dq_dof: wp.array2d(dtype=wp.float32),
    ) -> None:
        """Fill the rotation Jacobian block using Warp autodiff.

        Args:
            tape: Recorded Warp tape whose output is the global residual
                buffer.
            model: Shared articulation model.
            jacobian: Global Jacobian buffer to update,
                shape [n_batch, total_residual_count, joint_dof_count].
            start_idx: First residual row reserved for this objective.
            dq_dof: Differentiable joint update variable for the current
                batch, shape [n_batch, joint_dof_count].
        """
        self._require_batch_layout()
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})

            q_grad = tape.gradients[dq_dof]

            n_dofs = model.joint_dof_count

            wp.launch(
                _rot_jac_fill,
                dim=self.n_batch,
                inputs=[
                    q_grad,
                    n_dofs,
                    start_idx,
                    component,
                ],
                outputs=[
                    jacobian,
                ],
                device=self.device,
            )

            tape.zero()

    def compute_jacobian_analytic(
        self,
        body_q: wp.array2d(dtype=wp.transform),
        joint_q: wp.array2d(dtype=wp.float32),
        model: Model,
        jacobian: wp.array3d(dtype=wp.float32),
        joint_S_s: wp.array2d(dtype=wp.spatial_vector),
        start_idx: int,
    ) -> None:
        """Fill the rotation Jacobian block from the analytic motion subspace.

        Args:
            body_q: Batched body transforms for the evaluation rows,
                shape [n_batch, body_count].
            joint_q: Batched joint coordinates, shape [n_batch, joint_coord_count].
                Present for interface compatibility and not used directly by
                this objective.
            model: Shared articulation model.
            jacobian: Global Jacobian buffer to update,
                shape [n_batch, total_residual_count, joint_dof_count].
            joint_S_s: Batched motion-subspace columns,
                shape [n_batch, joint_dof_count].
            start_idx: First residual row reserved for this objective.
        """
        n_dofs = model.joint_dof_count

        wp.launch(
            _rot_jac_analytic,
            dim=[body_q.shape[0], n_dofs],
            inputs=[
                self.affects_dof,  # lookup mask
                joint_S_s,
                start_idx,
                n_dofs,
                self.weight,
            ],
            outputs=[jacobian],
            device=self.device,
        )
