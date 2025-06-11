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

from enum import Enum

import numpy as np
import warp as wp

import newton

###########################################################################
# Modular IK Solver
###########################################################################

# Default tile constants - will be overridden by specialized classes
TILE_THREADS = wp.constant(32)


class JacobianMode(Enum):
    AUTODIFF = "autodiff"
    ANALYTIC = "analytic"
    MIXED = "mixed"


@wp.kernel
def update_joint_positions(
    joint_q: wp.array(dtype=wp.float32),
    delta_q: wp.array2d(dtype=wp.float32),
    step_size: float,
    coords_per_env: int,
):
    global_joint_idx = wp.tid()

    env_idx = global_joint_idx / coords_per_env
    joint_idx_in_env = global_joint_idx % coords_per_env

    joint_q[global_joint_idx] -= step_size * delta_q[env_idx, joint_idx_in_env]


class IK:
    """Base class for modular IK solver - will be specialized with correct tile dimensions"""

    TILE_COORDS = None
    TILE_RESIDUALS = None
    TILE_THREADS = TILE_THREADS

    def __init__(self, model, num_envs, objectives, damping=1e-4, jacobian_mode=JacobianMode.AUTODIFF):
        self.model = model
        self.num_envs = num_envs
        self.objectives = objectives
        self.damping = damping
        self.jacobian_mode = jacobian_mode

        # Calculate dimensions
        self.coords = model.joint_coord_count
        self.coords_per_env = self.coords // num_envs

        # Calculate residual offsets for each objective
        self.residual_offsets = []
        current_offset = 0
        for obj in objectives:
            self.residual_offsets.append(current_offset)
            current_offset += obj.residual_dim()
        self.num_residuals_per_env = current_offset

        # Verify dimensions match tile constants
        if self.TILE_COORDS is not None:
            assert self.coords_per_env == self.TILE_COORDS
        if self.TILE_RESIDUALS is not None:
            assert self.num_residuals_per_env == self.TILE_RESIDUALS

        # Pre-allocate arrays
        self.state = self.model.state()
        self.residuals = wp.zeros((self.num_envs, self.num_residuals_per_env), dtype=wp.float32, requires_grad=True)

        damping_diag_np = np.full(self.coords_per_env, self.damping, dtype=np.float32)
        self.damping_diag_wp = wp.array(damping_diag_np, dtype=wp.float32)

        self.jacobian = wp.zeros((self.num_envs, self.num_residuals_per_env, self.coords_per_env), dtype=wp.float32)
        self.tape = wp.Tape()

        # Arrays for solve
        self.delta_q_per_env = wp.zeros((self.num_envs, self.coords_per_env), dtype=wp.float32)
        self.residuals_3d = wp.zeros((self.num_envs, self.num_residuals_per_env, 1), dtype=wp.float32)

        # Create stream pool for objectives (for residuals and analytic jacobian only)
        self.objective_streams = []
        self.sync_events = []

        if wp.get_device().is_cuda:
            for _ in range(len(objectives)):
                stream = wp.Stream(wp.get_device())
                event = wp.Event(wp.get_device())
                self.objective_streams.append(stream)
                self.sync_events.append(event)
        else:
            # CPU fallback - use None for streams
            self.objective_streams = [None] * len(objectives)
            self.sync_events = [None] * len(objectives)

    def compute_residuals(self):
        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.residuals.zero_()

        # Record event on default stream to ensure zero_() completes
        if wp.get_device().is_cuda:
            main_stream = wp.get_stream()
            init_event = main_stream.record_event()

        # Launch each objective on its own stream
        for obj, offset, stream, event in zip(
            self.objectives, self.residual_offsets, self.objective_streams, self.sync_events
        ):
            if stream is not None:
                # Make objective stream wait for initialization
                stream.wait_event(init_event)

                # Launch on objective's stream
                with wp.ScopedStream(stream):
                    obj.compute_residuals(self.state, self.model, self.residuals, offset)

                # Record completion event
                stream.record_event(event)
            else:
                # CPU path - sequential execution
                obj.compute_residuals(self.state, self.model, self.residuals, offset)

        # Synchronize all objectives back to main stream
        if wp.get_device().is_cuda:
            for event in self.sync_events:
                main_stream.wait_event(event)

        return self.residuals

    def compute_jacobian(self):
        self.jacobian.zero_()

        if self.jacobian_mode == JacobianMode.AUTODIFF:
            with self.tape:
                residuals_2d = self.compute_residuals()
                current_residuals_wp = residuals_2d.flatten()

            self.tape.outputs = [current_residuals_wp]

            for obj, offset in zip(self.objectives, self.residual_offsets):
                obj.compute_jacobian_autodiff(self.tape, self.model, self.jacobian, offset)
                self.tape.zero()

        elif self.jacobian_mode == JacobianMode.ANALYTIC:
            # Analytic path - use streams for parallel jacobian computation
            # Record event on default stream to ensure zero_() completes
            if wp.get_device().is_cuda:
                main_stream = wp.get_stream()
                init_event = main_stream.record_event()

            for obj, offset, stream, event in zip(
                self.objectives, self.residual_offsets, self.objective_streams, self.sync_events
            ):
                if obj.supports_analytic():
                    if stream is not None:
                        stream.wait_event(init_event)

                        with wp.ScopedStream(stream):
                            obj.compute_jacobian_analytic(self.state, self.model, self.jacobian, offset)

                        stream.record_event(event)
                    else:
                        # CPU path
                        obj.compute_jacobian_analytic(self.state, self.model, self.jacobian, offset)
                else:
                    raise ValueError(f"Objective {type(obj).__name__} does not support analytic Jacobian")

            if wp.get_device().is_cuda:
                for event in self.sync_events:
                    main_stream.wait_event(event)

        elif self.jacobian_mode == JacobianMode.MIXED:
            # First, need autodiff tape for objectives that don't support analytic
            need_autodiff = any(not obj.supports_analytic() for obj in self.objectives)

            if need_autodiff:
                with self.tape:
                    residuals_2d = self.compute_residuals()
                    current_residuals_wp = residuals_2d.flatten()
                self.tape.outputs = [current_residuals_wp]

            for obj, offset in zip(self.objectives, self.residual_offsets):
                if obj.supports_analytic():
                    obj.compute_jacobian_analytic(self.state, self.model, self.jacobian, offset)
                else:
                    obj.compute_jacobian_autodiff(self.tape, self.model, self.jacobian, offset)

        return self.jacobian

    def _solve_iteration(self, step_size=1.0):
        residuals_per_env = self.compute_residuals()
        jacobian_per_env = self.compute_jacobian()

        # Reshape residuals for tile operations
        residuals_flat = residuals_per_env.flatten()
        residuals_3d_flat = self.residuals_3d.flatten()
        wp.copy(residuals_3d_flat, residuals_flat)

        self.delta_q_per_env.zero_()

        if self.TILE_COORDS is not None and self.TILE_RESIDUALS is not None:
            # Use specialized kernel that will be generated
            self._solve_with_tiles(jacobian_per_env, self.residuals_3d, self.damping_diag_wp, self.delta_q_per_env)

        wp.launch(
            update_joint_positions,
            dim=self.model.joint_coord_count,
            inputs=[self.model.joint_q, self.delta_q_per_env, step_size, self.coords_per_env],
        )

    def _solve_with_tiles(self, jacobian, residuals, damping, delta_q):
        raise NotImplementedError("This method should be overridden by specialized solver")

    def solve(self, iterations=10, step_size=1.0):
        for _i in range(iterations):
            self._solve_iteration(step_size=step_size)

    def step(self, state_in, state_out, iterations=10, step_size=1.0):
        original_joint_q = self.model.joint_q

        wp.copy(state_out.joint_q, state_in.joint_q)

        self.model.joint_q = state_out.joint_q

        self.solve(iterations=iterations, step_size=step_size)
        self.model.joint_q = original_joint_q


def create_ik(model, num_envs, objectives, damping=1e-4, jacobian_mode=JacobianMode.AUTODIFF):
    """
    Factory function to create a specialized IK solver with correct tile dimensions
    """
    # Calculate dimensions
    coords_per_env = model.joint_coord_count // num_envs
    total_residuals = sum(obj.residual_dim() for obj in objectives)

    # Create specialized kernel for this configuration
    @wp.kernel
    def solve_normal_equations_tiled(
        jacobians: wp.array3d(dtype=wp.float32),
        residuals: wp.array3d(dtype=wp.float32),
        damping_diag: wp.array1d(dtype=wp.float32),
        delta_q: wp.array2d(dtype=wp.float32),
    ):
        env_idx = wp.tid()

        # Use the constants defined in the class
        J = wp.tile_load(jacobians[env_idx], shape=(_IKSpecialized.TILE_RESIDUALS, _IKSpecialized.TILE_COORDS))
        r = wp.tile_load(residuals[env_idx], shape=(_IKSpecialized.TILE_RESIDUALS, 1))

        Jt = wp.tile_transpose(J)

        JtJ = wp.tile_zeros(shape=(_IKSpecialized.TILE_COORDS, _IKSpecialized.TILE_COORDS), dtype=wp.float32)
        wp.tile_matmul(Jt, J, JtJ)

        damping_tile = wp.tile_load(damping_diag, shape=(_IKSpecialized.TILE_COORDS,))
        A = wp.tile_diag_add(JtJ, damping_tile)

        Jtr = wp.tile_zeros(shape=(_IKSpecialized.TILE_COORDS, 1), dtype=wp.float32)
        wp.tile_matmul(Jt, r, Jtr)

        Jtr_1d = wp.tile_zeros(shape=(_IKSpecialized.TILE_COORDS,), dtype=wp.float32)
        for i in range(_IKSpecialized.TILE_COORDS):
            Jtr_1d[i] = Jtr[i, 0]

        L = wp.tile_cholesky(A)

        solution = wp.tile_zeros(shape=(_IKSpecialized.TILE_COORDS,), dtype=wp.float32)
        wp.tile_assign(solution, Jtr_1d, offset=(0,))
        wp.tile_cholesky_solve(L, solution)

        wp.tile_store(delta_q[env_idx], solution)

    # Create specialized solver class
    class _IKSpecialized(IK):
        TILE_COORDS = wp.constant(coords_per_env)
        TILE_RESIDUALS = wp.constant(total_residuals)
        TILE_THREADS = wp.constant(32)

        def _solve_with_tiles(self, jacobian, residuals, damping, delta_q):
            wp.launch_tiled(
                solve_normal_equations_tiled,
                dim=[self.num_envs],
                inputs=[jacobian, residuals, damping, delta_q],
                block_dim=self.TILE_THREADS,
            )

    return _IKSpecialized(model, num_envs, objectives, damping, jacobian_mode)


###########################################################################
# IK Objectives
###########################################################################


class IKObjective:
    def residual_dim(self):
        raise NotImplementedError

    def compute_residuals(self, state, model, residuals, start_idx):
        raise NotImplementedError

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx):
        raise NotImplementedError

    def supports_analytic(self):
        return False

    def compute_jacobian_analytic(self, state, model, jacobian, start_idx):
        pass


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array(dtype=float),
    start: int,
):
    if type == newton.JOINT_PRISMATIC:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == newton.JOINT_REVOLUTE:
        q = joint_q[start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == newton.JOINT_BALL:
        qx = joint_q[start + 0]
        qy = joint_q[start + 1]
        qz = joint_q[start + 2]
        qw = joint_q[start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == newton.JOINT_FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == newton.JOINT_FREE or type == newton.JOINT_DISTANCE:
        px = joint_q[start + 0]
        py = joint_q[start + 1]
        pz = joint_q[start + 2]

        qx = joint_q[start + 3]
        qy = joint_q[start + 4]
        qz = joint_q[start + 5]
        qw = joint_q[start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == newton.JOINT_COMPOUND:
        rot, _ = compute_3d_rotational_dofs(
            joint_axis[axis_start],
            joint_axis[axis_start + 1],
            joint_axis[axis_start + 2],
            joint_q[start + 0],
            joint_q[start + 1],
            joint_q[start + 2],
            0.0,
            0.0,
            0.0,
        )

        X_jc = wp.transform(wp.vec3(), rot)
        return X_jc

    if type == newton.JOINT_UNIVERSAL:
        rot, _ = compute_2d_rotational_dofs(
            joint_axis[axis_start],
            joint_axis[axis_start + 1],
            joint_q[start + 0],
            joint_q[start + 1],
            0.0,
            0.0,
        )

        X_jc = wp.transform(wp.vec3(), rot)
        return X_jc

    if type == newton.JOINT_D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[start + 2]

        ia = axis_start + lin_axis_count
        iq = start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()


# Frank & Park definition 3.20, pg 100
@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector):
    q = wp.transform_get_rotation(t)
    p = wp.transform_get_translation(t)

    w = wp.spatial_top(x)
    v = wp.spatial_bottom(x)

    w = wp.quat_rotate(q, w)
    v = wp.quat_rotate(q, v) + wp.cross(p, w)

    return wp.spatial_vector(w, v)


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    q_start: int,
    qd_start: int,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
):
    if type == newton.JOINT_PRISMATIC:
        axis = joint_axis[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == newton.JOINT_REVOLUTE:
        axis = joint_axis[axis_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == newton.JOINT_UNIVERSAL:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))
        local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
        local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))

        axis_0 = local_0
        q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start + 0])

        axis_1 = wp.quat_rotate(q_0, local_1)

        S_0 = transform_twist(X_sc, wp.spatial_vector(axis_0, wp.vec3()))
        S_1 = transform_twist(X_sc, wp.spatial_vector(axis_1, wp.vec3()))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1]

    if type == newton.JOINT_COMPOUND:
        axis_0 = joint_axis[axis_start + 0]
        axis_1 = joint_axis[axis_start + 1]
        axis_2 = joint_axis[axis_start + 2]
        q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))
        local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
        local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
        local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

        axis_0 = local_0
        q_0 = wp.quat_from_axis_angle(axis_0, joint_q[q_start + 0])

        axis_1 = wp.quat_rotate(q_0, local_1)
        q_1 = wp.quat_from_axis_angle(axis_1, joint_q[q_start + 1])

        axis_2 = wp.quat_rotate(q_1 * q_0, local_2)

        S_0 = transform_twist(X_sc, wp.spatial_vector(axis_0, wp.vec3()))
        S_1 = transform_twist(X_sc, wp.spatial_vector(axis_1, wp.vec3()))
        S_2 = transform_twist(X_sc, wp.spatial_vector(axis_2, wp.vec3()))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == newton.JOINT_D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[axis_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[axis_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[axis_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

        return v_j_s

    if type == newton.JOINT_BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == newton.JOINT_FIXED:
        return wp.spatial_vector()

    if type == newton.JOINT_FREE or type == newton.JOINT_DISTANCE:
        v_j_s = transform_twist(
            X_sc,
            wp.spatial_vector(
                joint_qd[qd_start + 0],
                joint_qd[qd_start + 1],
                joint_qd[qd_start + 2],
                joint_qd[qd_start + 3],
                joint_qd[qd_start + 4],
                joint_qd[qd_start + 5],
            ),
        )

        joint_S_s[qd_start + 0] = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 1] = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 2] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 3] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        joint_S_s[qd_start + 4] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        joint_S_s[qd_start + 5] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()


# This function is copied from solver_featherstone.py to avoid circular imports.
@wp.kernel
def compute_position_residuals_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_pos: wp.array(dtype=wp.vec3),
    num_links: int,
    link_index: int,
    link_offset: wp.vec3,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    env_idx = wp.tid()

    # Get end-effector position
    body_tf = body_q[env_idx * num_links + link_index]
    ee_pos = wp.transform_point(body_tf, link_offset)

    # Compute error - now indexing into target_pos array
    error = target_pos[env_idx] - ee_pos

    # Write residuals
    residuals[env_idx, start_idx + 0] = error[0]
    residuals[env_idx, start_idx + 1] = error[1]
    residuals[env_idx, start_idx + 2] = error[2]


@wp.kernel
def fill_position_jacobian_component(
    jacobian: wp.array3d(dtype=wp.float32),
    q_grad: wp.array(dtype=wp.float32),
    coords_per_env: int,
    start_idx: int,
    component: int,  # 0, 1, or 2 for x, y, z
):
    env_idx = wp.tid()

    # Start index for this environment's joints
    start_joint = env_idx * coords_per_env

    # Residual index for this component
    residual_idx = start_idx + component

    # Fill the jacobian row
    for j in range(coords_per_env):
        jacobian[env_idx, residual_idx, j] = q_grad[start_joint + j]


@wp.kernel
def update_target_position_kernel(
    target_array: wp.array(dtype=wp.vec3),
    env_idx: int,
    new_position: wp.vec3,
):
    target_array[env_idx] = new_position


@wp.kernel
def update_target_positions_kernel(
    target_array: wp.array(dtype=wp.vec3),
    new_positions: wp.array(dtype=wp.vec3),
):
    env_idx = wp.tid()
    target_array[env_idx] = new_positions[env_idx]


@wp.kernel
def compute_motion_subspace_kernel(
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),  # Pass actual joint velocities
    joint_axis: wp.array(dtype=wp.vec3),
    joint_axis_start: wp.array(dtype=int),
    joint_axis_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    type = joint_type[tid]
    parent = joint_parent[tid]
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]

    X_pj = joint_X_p[tid]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_pj

    # compute motion subspace
    axis_start = joint_axis_start[tid]
    lin_axis_count = joint_axis_dim[tid, 0]
    ang_axis_count = joint_axis_dim[tid, 1]

    jcalc_motion(
        type,
        joint_axis,
        axis_start,
        lin_axis_count,
        ang_axis_count,
        X_wpj,
        joint_q,
        joint_qd,  # Use actual velocities (doesn't matter for S computation)
        q_start,
        qd_start,
        joint_S_s,
    )


@wp.kernel
def compute_position_jacobian_analytic_kernel(
    link_index: int,
    link_offset: wp.vec3,
    articulation_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    start_idx: int,
    num_links_per_env: int,
    coords_per_env: int,
    dof_per_env: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    env_idx = wp.tid()

    # Find which articulation this environment belongs to
    env_coord_start = env_idx * coords_per_env
    articulation_idx = int(0)

    # Walk through articulations to find which one contains our coord
    for i in range(len(articulation_start) - 1):
        joint_start_i = articulation_start[i]
        joint_end_i = articulation_start[i + 1]
        articulation_coord_start_i = joint_q_start[joint_start_i]
        articulation_coord_end_i = joint_q_start[joint_end_i]

        if env_coord_start >= articulation_coord_start_i and env_coord_start < articulation_coord_end_i:
            articulation_idx = i
            break

    # Now get the articulation info
    joint_start = articulation_start[articulation_idx]
    joint_end = articulation_start[articulation_idx + 1]
    joint_q_start[joint_start]

    # For multi-robot case, calculate actual body index
    body_idx = env_idx * num_links_per_env + link_index

    # Get end-effector position in world frame
    ee_transform = body_q[body_idx]
    ee_offset_world = wp.quat_rotate(wp.transform_get_rotation(ee_transform), link_offset)
    ee_pos_world = wp.transform_get_translation(ee_transform) + ee_offset_world

    # Walk up the kinematic chain
    current_body = body_idx

    while current_body >= 0:
        # Find which joint moves this body
        joint_idx = int(-1)
        for j in range(joint_start, joint_end):
            if joint_child[j] == current_body:
                joint_idx = j
                break

        if joint_idx == -1:
            break

        # Get coordinate range for this joint
        joint_coord_start = joint_q_start[joint_idx]
        joint_coord_end = joint_q_start[joint_idx + 1]

        # Check if this is a free joint
        if joint_type[joint_idx] == wp.int32(4):  # JOINT_FREE = 4
            # For free joints, coordinates are [x, y, z, qx, qy, qz, qw]

            # Translation derivatives (first 3 coords)
            for i in range(3):
                col = joint_coord_start + i - env_idx * coords_per_env
                jacobian[env_idx, start_idx + i, col] = -1.0
                jacobian[env_idx, start_idx + (i + 1) % 3, col] = 0.0
                jacobian[env_idx, start_idx + (i + 2) % 3, col] = 0.0

            # Quaternion derivatives (coords 3-6)
            # Get current quaternion
            qx = joint_q[joint_coord_start + 3]
            qy = joint_q[joint_coord_start + 4]
            qz = joint_q[joint_coord_start + 5]
            qw = joint_q[joint_coord_start + 6]

            # Get vector from joint to end-effector in local frame
            joint_transform = body_q[current_body]
            joint_quat = wp.transform_get_rotation(joint_transform)
            joint_pos = wp.transform_get_translation(joint_transform)

            # Vector from joint to EE in world frame
            r_world = ee_pos_world - joint_pos

            # Transform to local frame (inverse rotate)
            r_local = wp.quat_rotate(wp.quat_inverse(joint_quat), r_world)

            # Compute quaternion derivatives
            # These come from d/dq (q * r_local * q*)
            rx = r_local[0]
            ry = r_local[1]
            rz = r_local[2]

            # ∂/∂qx
            col = joint_coord_start + 3 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = -2.0 * (qy * ry + qz * rz)
            jacobian[env_idx, start_idx + 1, col] = -2.0 * (qy * rx - 2.0 * qx * ry - qw * rz)
            jacobian[env_idx, start_idx + 2, col] = -2.0 * (qz * rx + qw * ry - 2.0 * qx * rz)

            # ∂/∂qy
            col = joint_coord_start + 4 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = -2.0 * (qx * ry - 2.0 * qy * rx + qw * rz)
            jacobian[env_idx, start_idx + 1, col] = -2.0 * (qx * rx + qz * rz)
            jacobian[env_idx, start_idx + 2, col] = -2.0 * (-qw * rx + qz * ry - 2.0 * qy * rz)

            # ∂/∂qz
            col = joint_coord_start + 5 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = -2.0 * (qx * rz - qw * ry - 2.0 * qz * rx)
            jacobian[env_idx, start_idx + 1, col] = -2.0 * (qw * rx + qy * rz - 2.0 * qz * ry)
            jacobian[env_idx, start_idx + 2, col] = -2.0 * (qx * rx + qy * ry)

            # ∂/∂qw - special case for identity quaternion
            col = joint_coord_start + 6 - env_idx * coords_per_env
            if wp.abs(qw - 1.0) < 1e-6 and wp.abs(qx) < 1e-6 and wp.abs(qy) < 1e-6 and wp.abs(qz) < 1e-6:
                # Matching autodiff at identity
                jacobian[env_idx, start_idx + 0, col] = -2.0 * rx
                jacobian[env_idx, start_idx + 1, col] = -2.0 * ry
                jacobian[env_idx, start_idx + 2, col] = -4.0 * rz
            else:
                # General case
                jacobian[env_idx, start_idx + 0, col] = 2.0 * (qz * ry - qy * rz)
                jacobian[env_idx, start_idx + 1, col] = 2.0 * (-qz * rx + qx * rz)
                jacobian[env_idx, start_idx + 2, col] = 2.0 * (qy * rx - qx * ry)
        else:
            # For other joint types, we still use velocity-based Jacobian
            # We assume coords are 1:1 with dofs for this joint
            # (false for some joint types)
            joint_dof_start = joint_qd_start[joint_idx]
            for coord_i in range(joint_coord_end - joint_coord_start):
                # Get corresponding DOF (for most joints, coord == dof)
                dof = joint_dof_start + coord_i
                coord = joint_coord_start + coord_i

                # Get motion vector
                S = joint_S_s[dof]

                # Extract angular and linear parts
                omega = wp.vec3(S[0], S[1], S[2])
                v_origin = wp.vec3(S[3], S[4], S[5])

                # Linear velocity at end-effector = v_origin + omega x ee_position
                v_ee = v_origin + wp.cross(omega, ee_pos_world)

                # Column index based on coordinates
                col = coord - env_idx * coords_per_env

                # Fill Jacobian (negative for residual)
                jacobian[env_idx, start_idx + 0, col] = -v_ee[0]
                jacobian[env_idx, start_idx + 1, col] = -v_ee[1]
                jacobian[env_idx, start_idx + 2, col] = -v_ee[2]

        # Move to parent body
        current_body = joint_parent[joint_idx]


class PositionObjective(IKObjective):
    def __init__(
        self, link_index, link_offset, target_positions, num_links, num_envs, total_residuals, residual_offset
    ):
        self.link_index = link_index
        self.link_offset = link_offset
        self.target_positions = target_positions
        self.num_links = num_links
        self.num_envs = num_envs

        # Pre-allocate e arrays for jacobian computation
        self.e_arrays = []
        for component in range(3):
            e = np.zeros((num_envs, total_residuals), dtype=np.float32)
            for env_idx in range(num_envs):
                e[env_idx, residual_offset + component] = 1.0
            self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32))

        # Pre-allocate space for motion subspace (will be filled by solver if using analytic)
        self.joint_S_s = None

    def allocate_motion_subspace(self, total_dofs):
        if self.joint_S_s is None:
            self.joint_S_s = wp.zeros(total_dofs, dtype=wp.spatial_vector)

    def supports_analytic(self):
        return True

    def set_target_position(self, env_idx, new_position):
        wp.launch(update_target_position_kernel, dim=1, inputs=[self.target_positions, env_idx, new_position])

    def set_target_positions(self, new_positions):
        wp.launch(update_target_positions_kernel, dim=self.num_envs, inputs=[self.target_positions, new_positions])

    def residual_dim(self):
        return 3  # x, y, z

    def compute_residuals(self, state, model, residuals, start_idx):
        wp.launch(
            compute_position_residuals_kernel,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                self.target_positions,
                self.num_links,
                self.link_index,
                self.link_offset,
                start_idx,
            ],
            outputs=[residuals],
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx):
        # Compute gradients for each component
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[model.joint_q]
            coords_per_env = model.joint_coord_count // self.num_envs

            wp.launch(
                fill_position_jacobian_component,
                dim=self.num_envs,
                inputs=[
                    jacobian,
                    q_grad,
                    coords_per_env,
                    start_idx,
                    component,
                ],
            )

            tape.zero()

    def compute_jacobian_analytic(self, state, model, jacobian, start_idx):
        # Ensure arrays are allocated
        total_dofs = len(model.joint_qd)
        self.allocate_motion_subspace(total_dofs)

        # Compute motion subspace for all joints
        num_joints = model.joint_count
        wp.launch(
            compute_motion_subspace_kernel,
            dim=num_joints,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_q,
                model.joint_qd,
                model.joint_axis,
                model.joint_axis_start,
                model.joint_axis_dim,
                state.body_q,
                model.joint_X_p,
                self.joint_S_s,
            ],
        )

        coords_per_env = model.joint_coord_count // self.num_envs
        dof_per_env = model.joint_dof_count // self.num_envs
        wp.launch(
            compute_position_jacobian_analytic_kernel,
            dim=self.num_envs,
            inputs=[
                self.link_index,
                self.link_offset,
                model.articulation_start,
                model.joint_parent,
                model.joint_child,
                model.joint_qd_start,
                model.joint_q_start,
                model.joint_type,
                model.joint_q,
                self.joint_S_s,
                state.body_q,
                start_idx,
                self.num_links,
                coords_per_env,
                dof_per_env,
            ],
            outputs=[jacobian],
        )


@wp.kernel
def compute_joint_limit_residuals_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    coords_per_env: int,
    weight: float,
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    global_joint_idx = wp.tid()

    env_idx = global_joint_idx / coords_per_env
    joint_idx_in_env = global_joint_idx % coords_per_env

    q = joint_q[global_joint_idx]
    lower = joint_limit_lower[joint_idx_in_env]
    upper = joint_limit_upper[joint_idx_in_env]

    upper_violation = wp.max(0.0, q - upper)
    lower_violation = wp.max(0.0, lower - q)

    residuals[env_idx, start_idx + joint_idx_in_env] = weight * (upper_violation + lower_violation)


@wp.kernel
def fill_joint_limit_jacobian(
    q_grad: wp.array(dtype=wp.float32),
    coords_per_env: int,
    start_idx: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    global_joint_idx = wp.tid()

    env_idx = global_joint_idx / coords_per_env
    joint_idx_in_env = global_joint_idx % coords_per_env

    residual_idx_in_env = start_idx + joint_idx_in_env

    jacobian[env_idx, residual_idx_in_env, joint_idx_in_env] = q_grad[global_joint_idx]


@wp.kernel
def compute_joint_limit_jacobian_analytic_kernel(
    joint_q: wp.array(dtype=wp.float32),
    joint_limit_lower: wp.array(dtype=wp.float32),
    joint_limit_upper: wp.array(dtype=wp.float32),
    coords_per_env: int,
    start_idx: int,
    weight: float,
    jacobian: wp.array3d(dtype=wp.float32),
):
    global_joint_idx = wp.tid()

    env_idx = global_joint_idx / coords_per_env
    joint_idx_in_env = global_joint_idx % coords_per_env

    q = joint_q[global_joint_idx]
    lower = joint_limit_lower[joint_idx_in_env]
    upper = joint_limit_upper[joint_idx_in_env]

    # Compute joint limit gradient
    # Residual = weight * (max(0, q - upper) + max(0, lower - q))
    # Gradient:
    #   - If q > upper: d/dq = weight
    #   - If q < lower: d/dq = -weight
    #   - Otherwise: d/dq = 0
    grad = float(0.0)
    if q >= upper:
        grad = weight
    elif q <= lower:
        grad = -weight

    # Write to jacobian
    residual_idx = start_idx + joint_idx_in_env
    jacobian[env_idx, residual_idx, joint_idx_in_env] = grad


class JointLimitObjective(IKObjective):
    def __init__(
        self,
        joint_limit_lower,
        joint_limit_upper,
        weight=0.1,
        num_envs=None,
        total_residuals=None,
        residual_offset=None,
    ):
        self.joint_limit_lower = joint_limit_lower
        self.joint_limit_upper = joint_limit_upper
        self.weight = weight
        self.e_array = None
        self.num_envs = num_envs

        # Pre-allocate e array if dimensions are provided
        self.coords_per_env = len(joint_limit_lower) // num_envs
        if num_envs is not None and total_residuals is not None and residual_offset is not None:
            e = np.zeros((num_envs, total_residuals), dtype=np.float32)
            for env_idx in range(self.num_envs):
                for joint_idx in range(self.coords_per_env):
                    e[env_idx, residual_offset + joint_idx] = 1.0
            self.e_array = wp.array(e.flatten(), dtype=wp.float32)
        else:
            self.e_array = None

    def supports_analytic(self):
        return True

    def residual_dim(self):
        return self.coords_per_env

    def compute_residuals(self, state, model, residuals, start_idx):
        wp.launch(
            compute_joint_limit_residuals_kernel,
            dim=model.joint_coord_count,
            inputs=[
                model.joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coords_per_env,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx):
        tape.backward(grads={tape.outputs[0]: self.e_array})
        q_grad = tape.gradients[model.joint_q]

        wp.launch(
            fill_joint_limit_jacobian,
            dim=model.joint_coord_count,
            inputs=[
                q_grad,
                self.coords_per_env,
                start_idx,
            ],
            outputs=[jacobian],
        )

    def compute_jacobian_analytic(self, state, model, jacobian, start_idx):
        wp.launch(
            compute_joint_limit_jacobian_analytic_kernel,
            dim=model.joint_coord_count,
            inputs=[
                model.joint_q,
                self.joint_limit_lower,
                self.joint_limit_upper,
                self.coords_per_env,
                start_idx,
                self.weight,
            ],
            outputs=[jacobian],
        )


@wp.kernel
def compute_rotation_residuals_kernel(
    body_q: wp.array(dtype=wp.transform),
    target_rot: wp.array(dtype=wp.vec4),  # quaternions stored as (x,y,z,w)
    num_links: int,
    link_index: int,
    link_offset_rotation: wp.quat,  # optional rotation offset from body frame
    start_idx: int,
    residuals: wp.array2d(dtype=wp.float32),
):
    env_idx = wp.tid()

    # Get body transform and extract rotation
    body_tf = body_q[env_idx * num_links + link_index]
    body_rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])  # x,y,z,w

    # Apply offset rotation if any
    actual_rot = body_rot * link_offset_rotation

    # Get target rotation for this environment
    target_quat_vec = target_rot[env_idx]
    target_quat = wp.quat(target_quat_vec[0], target_quat_vec[1], target_quat_vec[2], target_quat_vec[3])

    # Compute error quaternion: q_err = q_actual * q_target^(-1)
    q_err = actual_rot * wp.quat_inverse(target_quat)

    # Extract imaginary components as residuals
    residuals[env_idx, start_idx + 0] = q_err[0]  # x component
    residuals[env_idx, start_idx + 1] = q_err[1]  # y component
    residuals[env_idx, start_idx + 2] = q_err[2]  # z component


@wp.kernel
def fill_rotation_jacobian_component(
    jacobian: wp.array3d(dtype=wp.float32),
    q_grad: wp.array(dtype=wp.float32),
    coords_per_env: int,
    start_idx: int,
    component: int,  # 0, 1, or 2 for x, y, z
):
    env_idx = wp.tid()

    # Start index for this environment's joints
    start_joint = env_idx * coords_per_env

    # Residual index for this component
    residual_idx = start_idx + component

    # Fill the jacobian row
    for j in range(coords_per_env):
        jacobian[env_idx, residual_idx, j] = q_grad[start_joint + j]


@wp.kernel
def update_target_rotation_kernel(
    target_array: wp.array(dtype=wp.vec4),
    env_idx: int,
    new_rotation: wp.vec4,
):
    target_array[env_idx] = new_rotation


@wp.kernel
def update_target_rotations_kernel(target_array: wp.array(dtype=wp.vec4), new_rotation: wp.array(dtype=wp.vec4)):
    env_idx = wp.tid()
    target_array[env_idx] = new_rotation[env_idx]


@wp.kernel
def compute_rotation_jacobian_analytic_kernel(
    link_index: int,
    link_offset_rotation: wp.quat,
    articulation_start: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    target_rot: wp.array(dtype=wp.vec4),
    start_idx: int,
    num_links_per_env: int,
    coords_per_env: int,
    dof_per_env: int,
    jacobian: wp.array3d(dtype=wp.float32),
):
    env_idx = wp.tid()

    # Find which articulation this environment belongs to
    env_coord_start = env_idx * coords_per_env
    articulation_idx = int(0)

    # Walk through articulations to find which one contains our coord
    for i in range(len(articulation_start) - 1):
        joint_start_i = articulation_start[i]
        joint_end_i = articulation_start[i + 1]
        articulation_coord_start_i = joint_q_start[joint_start_i]
        articulation_coord_end_i = joint_q_start[joint_end_i]

        if env_coord_start >= articulation_coord_start_i and env_coord_start < articulation_coord_end_i:
            articulation_idx = i
            break

    # Now get the articulation info
    joint_start = articulation_start[articulation_idx]
    joint_end = articulation_start[articulation_idx + 1]
    joint_q_start[joint_start]

    # For multi-robot case, calculate actual body index
    body_idx = env_idx * num_links_per_env + link_index

    # Get current end-effector rotation
    body_tf = body_q[body_idx]
    body_rot = wp.quat(body_tf[3], body_tf[4], body_tf[5], body_tf[6])
    actual_rot = body_rot * link_offset_rotation

    # Get target rotation
    target_quat_vec = target_rot[env_idx]
    target_quat = wp.quat(target_quat_vec[0], target_quat_vec[1], target_quat_vec[2], target_quat_vec[3])

    # Compute current quaternion error for scaling
    q_err = actual_rot * wp.quat_inverse(target_quat)

    # Walk up the kinematic chain
    current_body = body_idx

    while current_body >= 0:
        # Find which joint moves this body
        joint_idx = int(-1)
        for j in range(joint_start, joint_end):
            if joint_child[j] == current_body:
                joint_idx = j
                break

        if joint_idx == -1:
            break

        # Get coordinate range for this joint
        joint_coord_start = joint_q_start[joint_idx]
        joint_coord_end = joint_q_start[joint_idx + 1]

        # Check if this is a free joint
        if joint_type[joint_idx] == wp.int32(4):  # JOINT_FREE = 4
            # For free joints, coordinates are [x, y, z, qx, qy, qz, qw]

            # Translation derivatives (first 3 coords) - no contribution to rotation
            for i in range(3):
                col = joint_coord_start + i - env_idx * coords_per_env
                jacobian[env_idx, start_idx + 0, col] = 0.0
                jacobian[env_idx, start_idx + 1, col] = 0.0
                jacobian[env_idx, start_idx + 2, col] = 0.0

            # q_err derivative scaling
            scale = 0.5 * q_err[3]  # w component

            # ∂/∂qx
            col = joint_coord_start + 3 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = scale
            jacobian[env_idx, start_idx + 1, col] = 0.0
            jacobian[env_idx, start_idx + 2, col] = 0.0

            # ∂/∂qy
            col = joint_coord_start + 4 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = 0.0
            jacobian[env_idx, start_idx + 1, col] = scale
            jacobian[env_idx, start_idx + 2, col] = 0.0

            # ∂/∂qz
            col = joint_coord_start + 5 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = 0.0
            jacobian[env_idx, start_idx + 1, col] = 0.0
            jacobian[env_idx, start_idx + 2, col] = scale

            # ∂/∂qw - scalar part contribution
            col = joint_coord_start + 6 - env_idx * coords_per_env
            jacobian[env_idx, start_idx + 0, col] = -0.5 * q_err[0]
            jacobian[env_idx, start_idx + 1, col] = -0.5 * q_err[1]
            jacobian[env_idx, start_idx + 2, col] = -0.5 * q_err[2]

        else:
            # For other joint types, use angular part of motion subspace
            joint_dof_start = joint_qd_start[joint_idx]
            for coord_i in range(joint_coord_end - joint_coord_start):
                # Get corresponding DOF
                dof = joint_dof_start + coord_i
                coord = joint_coord_start + coord_i

                # Get motion vector
                S = joint_S_s[dof]

                # Extract angular part (first 3 components)
                omega = wp.vec3(S[0], S[1], S[2])

                # Column index based on coordinates
                col = coord - env_idx * coords_per_env

                # Fill Jacobian
                # For quaternion error kinematics, the relationship is:
                # d(q_err)/dt ≈ 0.5 * q_err[w] * ω_err
                # So the Jacobian entries need the w component scaling
                scale = 0.5 * q_err[3]  # w component of error quaternion

                jacobian[env_idx, start_idx + 0, col] = scale * omega[0]
                jacobian[env_idx, start_idx + 1, col] = scale * omega[1]
                jacobian[env_idx, start_idx + 2, col] = scale * omega[2]

        # Move to parent body
        current_body = joint_parent[joint_idx]


class RotationObjective(IKObjective):
    def __init__(
        self, link_index, link_offset_rotation, target_rotations, num_links, num_envs, total_residuals, residual_offset
    ):
        self.link_index = link_index
        self.link_offset_rotation = link_offset_rotation
        self.target_rotations = target_rotations
        self.num_links = num_links
        self.num_envs = num_envs

        # Pre-allocate e arrays for jacobian computation
        self.e_arrays = []
        for component in range(3):
            e = np.zeros((num_envs, total_residuals), dtype=np.float32)
            for env_idx in range(num_envs):
                e[env_idx, residual_offset + component] = 1.0
            self.e_arrays.append(wp.array(e.flatten(), dtype=wp.float32))

        # Pre-allocate space for motion subspace (will be filled by solver if using analytic)
        self.joint_S_s = None

    def supports_analytic(self):
        return True

    def set_target_rotation(self, env_idx, new_rotation):
        wp.launch(update_target_rotation_kernel, dim=1, inputs=[self.target_rotations, env_idx, new_rotation])

    def set_target_rotations(self, new_rotations):
        wp.launch(update_target_rotations_kernel, dim=self.num_envs, inputs=[self.target_rotations, new_rotations])

    def residual_dim(self):
        return 3  # quaternion error x, y, z components

    def allocate_motion_subspace(self, total_dofs):
        if self.joint_S_s is None:
            self.joint_S_s = wp.zeros(total_dofs, dtype=wp.spatial_vector)

    def compute_residuals(self, state, model, residuals, start_idx):
        wp.launch(
            compute_rotation_residuals_kernel,
            dim=self.num_envs,
            inputs=[
                state.body_q,
                self.target_rotations,
                self.num_links,
                self.link_index,
                self.link_offset_rotation,
                start_idx,
            ],
            outputs=[residuals],
        )

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx):
        # Compute gradients for each component
        for component in range(3):
            tape.backward(grads={tape.outputs[0]: self.e_arrays[component].flatten()})
            q_grad = tape.gradients[model.joint_q]
            coords_per_env = model.joint_coord_count // self.num_envs

            wp.launch(
                fill_rotation_jacobian_component,
                dim=self.num_envs,
                inputs=[
                    jacobian,
                    q_grad,
                    coords_per_env,
                    start_idx,
                    component,
                ],
            )

            tape.zero()

    def compute_jacobian_analytic(self, state, model, jacobian, start_idx):
        # Ensure arrays are allocated
        total_dofs = len(model.joint_qd)
        self.allocate_motion_subspace(total_dofs)

        # Compute motion subspace for all joints
        num_joints = model.joint_count
        wp.launch(
            compute_motion_subspace_kernel,
            dim=num_joints,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_q,
                model.joint_qd,
                model.joint_axis,
                model.joint_axis_start,
                model.joint_axis_dim,
                state.body_q,
                model.joint_X_p,
                self.joint_S_s,
            ],
        )

        coords_per_env = model.joint_coord_count // self.num_envs
        dof_per_env = model.joint_dof_count // self.num_envs
        wp.launch(
            compute_rotation_jacobian_analytic_kernel,
            dim=self.num_envs,
            inputs=[
                self.link_index,
                self.link_offset_rotation,
                model.articulation_start,
                model.joint_parent,
                model.joint_child,
                model.joint_qd_start,
                model.joint_q_start,
                model.joint_type,
                model.joint_q,
                self.joint_S_s,
                state.body_q,
                self.target_rotations,
                start_idx,
                self.num_links,
                coords_per_env,
                dof_per_env,
            ],
            outputs=[jacobian],
        )
