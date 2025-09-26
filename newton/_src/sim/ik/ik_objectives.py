# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Objective definitions for inverse kinematics."""

import numpy as np
import warp as wp

from .ik_common import IKJacobianMode


class IKObjective:
    """
    Abstract base class for inverse kinematics objectives.

    Subclasses must implement the following methods:
        - residual_dim(): Returns the number of residuals (constraints) this objective contributes.
        - compute_residuals(body_q, joint_q, model, residuals, start_idx): Computes the residuals for this objective.
        - compute_jacobian_autodiff(tape, model, jacobian, start_idx, dq_dof): Computes the Jacobian using autodiff.

    Optional methods for analytic Jacobian support:
        - supports_analytic(): Returns True if analytic Jacobian is supported, otherwise False.
        - compute_jacobian_analytic(body_q, joint_q, model, jacobian, joint_S_s, start_idx): Computes the analytic Jacobian if supported.

    Device and buffer management:
        - bind_device(device): Binds the objective to a specific device.
        - init_buffers(model, jacobian_mode): Initializes any buffers required for Jacobian computation.

    Notes:
        - The interface is designed for batch processing of multiple IK problems in parallel.
        - Each subclass may store per-problem data (e.g., targets) as device arrays.
    """

    def __init__(self):
        # Optimizers assign these before first use via set_batch_layout().
        self.total_residuals = None
        self.residual_offset = None
        self.n_batch = None

    def set_batch_layout(self, total_residuals, residual_offset, n_batch):
        """
        Register the residual layout for this objective within the optimizer's
        global system.

        Parameters
        ----------
        total_residuals : int
            Total number of residual rows across all objectives (global height of J and r).
        residual_offset : int
            Starting row index (into the global residual/Jacobian) reserved for this objective.
        n_batch : int
            Number of rows that will be evaluated together by the optimizer
            (e.g., n_problems * n_seeds, or further expanded for candidates).

        Notes
        -----
        * `n_batch` is the size of the **evaluation batch**, not the number of base problems.
        * Per-problem buffers (e.g., targets) should be sized by the number of base problems
          and accessed via the `problem_idx` mapping provided at evaluation time, rather than
          assuming one target per batch row.
        * This method is called by the optimizer before any residual/Jacobian computations.
        """
        self.total_residuals = total_residuals
        self.residual_offset = residual_offset
        self.n_batch = n_batch

    def _require_batch_layout(self):
        if self.total_residuals is None or self.residual_offset is None or self.n_batch is None:
            raise RuntimeError(f"Batch layout not assigned for {type(self).__name__}; call set_batch_layout() first")

    def residual_dim(self):
        """
        Returns the number of residuals (constraints) this objective contributes.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        """
        Computes the residuals for this objective and writes them into the residuals array.

        Args:
            body_q (wp.array2d(dtype=wp.transform)): Array of body transforms for each problem.
            joint_q (wp.array2d(dtype=wp.float)): Array of joint coordinates for each problem.
            model (newton.Model): The kinematic model.
            residuals (wp.array2d(dtype=wp.float)): Output array for residuals.
            start_idx (int): Starting index in the residuals array for this objective.
            problem_idx (wp.array1d(dtype=int32) | None): Maps each batched row to
                the originating problem index when rows have been duplicated (for example,
                when evaluating multiple candidates per problem). Typical usage inside a
                kernel looks like `row_idx = wp.tid(); base = problem_idx[row_idx]; target = self.target_positions[base]`.
                Defaults to None, which indicates rows already align with the solver batch.
        """
        raise NotImplementedError

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        """
        Computes the Jacobian for this objective using automatic differentiation.

        Args:
            tape (wp.Tape): Autodiff tape.
            model (newton.Model): The kinematic model.
            jacobian (wp.array3d(dtype=wp.float)): Output array for the Jacobian.
            start_idx (int): Starting index in the Jacobian for this objective.
            dq_dof (wp.array2d(dtype=wp.float)): Derivative of q with respect to DoF.
        """
        raise NotImplementedError

    def supports_analytic(self):
        """
        Returns True if this objective supports analytic Jacobian computation.
        Subclasses should override if analytic Jacobian is available.
        """
        return False

    def bind_device(self, device):
        """
        Binds the objective to a specific device (e.g., CUDA or CPU).

        Args:
            device (wp.Device): The device to bind to.
        """
        self.device = device

    def init_buffers(self, model, jacobian_mode):
        """
        Initializes any buffers required for Jacobian computation.

        Args:
            model (newton.Model): The kinematic model.
            jacobian_mode (IKJacobianMode): The Jacobian computation mode (analytic or autodiff).
        """
        pass

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        """
        Computes the analytic Jacobian for this objective, if supported.

        Args:
            body_q (wp.array2d(dtype=wp.transform)): Array of body transforms for each problem.
            joint_q (wp.array2d(dtype=wp.float)): Array of joint coordinates for each problem.
            model (newton.Model): The kinematic model.
            jacobian (wp.array3d(dtype=wp.float)): Output array for the Jacobian.
            joint_S_s (wp.array2d(dtype=wp.spatial_vector)): Motion subspace matrices.
            start_idx (int): Starting index in the Jacobian for this objective.
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


class IKPositionObjective(IKObjective):
    """
    End-effector positional target for one link.

    Args:
        link_index (int): Body index whose frame defines the end-effector.
        link_offset (wp.vec3): Offset from the body frame (local coordinates).
        target_positions (wp.array(dtype=wp.vec3)): One target position per problem.
        weight (float, optional): Scalar weight multiplying both residual and Jacobian rows. Defaults to 1.0.
    """

    def __init__(self, link_index, link_offset, target_positions, weight=1.0):
        super().__init__()
        self.link_index = link_index
        self.link_offset = link_offset
        self.target_positions = target_positions
        self.weight = weight

        self.affects_dof = None
        self.e_arrays = None

    def init_buffers(self, model, jacobian_mode):
        """Precompute lookup tables for analytic jacobian computation."""
        self._require_batch_layout()
        if jacobian_mode == IKJacobianMode.ANALYTIC:
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
        elif jacobian_mode == IKJacobianMode.AUTODIFF:
            self.e_arrays = []
            for component in range(3):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_batch):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self):
        return True

    def set_target_position(self, problem_idx, new_position):
        self._require_batch_layout()
        wp.launch(
            _update_position_target,
            dim=1,
            inputs=[problem_idx, new_position],
            outputs=[self.target_positions],
            device=self.device,
        )

    def set_target_positions(self, new_positions):
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

    def residual_dim(self):
        return 3

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
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

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
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

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
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


class IKJointLimitObjective(IKObjective):
    """
    Joint limit constraint objective.

    Args:
        joint_limit_lower (wp.array(dtype=float)): Lower bounds for each joint DoF.
        joint_limit_upper (wp.array(dtype=float)): Upper bounds for each joint DoF.
        weight (float, optional): Scalar weight for limit violation penalty. Defaults to 0.1.
    """

    def __init__(
        self,
        joint_limit_lower,
        joint_limit_upper,
        weight=0.1,
    ):
        super().__init__()
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.e_array = None
        self.weight = weight

        self.n_dofs = len(joint_limit_lower)

        self.dof_to_coord = None

    def init_buffers(self, model, jacobian_mode):
        self._require_batch_layout()
        if jacobian_mode == IKJacobianMode.AUTODIFF:
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

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.n_dofs

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
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

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
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

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
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


class IKRotationObjective(IKObjective):
    """
    End-effector rotational target for one link.

    Args:
        link_index (int): Body index whose frame defines the end-effector.
        link_offset_rotation (wp.quat): Rotation offset from the body frame (local coordinates).
        target_rotations (wp.array(dtype=wp.vec4)): One target quaternion per problem (stored as vec4).
        canonicalize_quat_err (bool, optional): If True, the error quaternion is flipped so its scalar part is non-negative,
            yielding the short-arc residual in SO(3). When False, the quaternion is used exactly as computed, preserving any
            sign convention from the forward kinematics. Defaults to True.
        weight (float, optional): Scalar weight multiplying both residual and Jacobian rows. Defaults to 1.0.
    """

    def __init__(
        self,
        link_index,
        link_offset_rotation,
        target_rotations,
        canonicalize_quat_err=True,
        weight=1.0,
    ):
        super().__init__()
        self.link_index = link_index
        self.link_offset_rotation = link_offset_rotation
        self.target_rotations = target_rotations
        self.weight = weight
        self.canonicalize_quat_err = canonicalize_quat_err

        self.affects_dof = None
        self.e_arrays = None

    def init_buffers(self, model, jacobian_mode):
        """Precompute lookup tables for analytic jacobian computation."""
        self._require_batch_layout()
        if jacobian_mode == IKJacobianMode.ANALYTIC:
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
        elif jacobian_mode == IKJacobianMode.AUTODIFF:
            self.e_arrays = []
            for component in range(3):
                e = np.zeros((self.n_batch, self.total_residuals), dtype=np.float32)
                for prob_idx in range(self.n_batch):
                    e[prob_idx, self.residual_offset + component] = 1.0
                self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32, device=self.device))

    def supports_analytic(self):
        return True

    def set_target_rotation(self, problem_idx, new_rotation):
        self._require_batch_layout()
        wp.launch(
            _update_rotation_target,
            dim=1,
            inputs=[problem_idx, new_rotation],
            outputs=[self.target_rotations],
            device=self.device,
        )

    def set_target_rotations(self, new_rotations):
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

    def residual_dim(self):
        return 3

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
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

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
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

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
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
