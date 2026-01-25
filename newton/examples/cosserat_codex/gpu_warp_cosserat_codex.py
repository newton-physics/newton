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

"""Direct Cosserat rod: native reference + GPU-resident Warp candidate.

This example runs two direct rods side-by-side:
- Reference rod: full native DefKitAdv.dll pipeline.
- Candidate rod: GPU-resident Warp implementation that avoids host round-trips.

Command:
    uv run python newton/examples/cosserat_codex/gpu_warp_cosserat_codex.py --dll-path "C:\\path\\to\\DefKitAdv.dll"
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.cosserat_codex import warp_cosserat_codex as base

# Warp tile configuration for direct solve
BLOCK_DIM = base.BLOCK_DIM
TILE = base.TILE

# Banded Cholesky layout (matches spbsv_u11_1rhs in C++)
BAND_KD = base.BAND_KD
BAND_LDAB = base.BAND_LDAB

DIRECT_SOLVE_WARP_BLOCK_THOMAS = base.DIRECT_SOLVE_WARP_BLOCK_THOMAS
DIRECT_SOLVE_WARP_BANDED_CHOLESKY = base.DIRECT_SOLVE_WARP_BANDED_CHOLESKY
DIRECT_SOLVE_BACKENDS = (
    DIRECT_SOLVE_WARP_BLOCK_THOMAS,
    DIRECT_SOLVE_WARP_BANDED_CHOLESKY,
)


@wp.func
def _warp_quat_correction_g(q: wp.quat, dtheta: wp.vec3) -> wp.quat:
    norm_sq = dtheta.x * dtheta.x + dtheta.y * dtheta.y + dtheta.z * dtheta.z
    if norm_sq < 1.0e-20:
        return q
    corr_x = 0.5 * (q.w * dtheta.x + q.z * dtheta.y - q.y * dtheta.z)
    corr_y = 0.5 * (-q.z * dtheta.x + q.w * dtheta.y + q.x * dtheta.z)
    corr_z = 0.5 * (q.y * dtheta.x - q.x * dtheta.y + q.w * dtheta.z)
    corr_w = 0.5 * (-q.x * dtheta.x - q.y * dtheta.y - q.z * dtheta.z)
    q_new = wp.quat(q.x + corr_x, q.y + corr_y, q.z + corr_z, q.w + corr_w)
    return base._warp_quat_normalize(q_new)


@wp.func
def _warp_jacobian_dot(
    jacobian: wp.array(dtype=wp.float32),
    edge: int,
    col: int,
    dl0: float,
    dl1: float,
    dl2: float,
    dl3: float,
    dl4: float,
    dl5: float,
) -> float:
    return (
        jacobian[base._warp_jacobian_index(edge, 0, col)] * dl0
        + jacobian[base._warp_jacobian_index(edge, 1, col)] * dl1
        + jacobian[base._warp_jacobian_index(edge, 2, col)] * dl2
        + jacobian[base._warp_jacobian_index(edge, 3, col)] * dl3
        + jacobian[base._warp_jacobian_index(edge, 4, col)] * dl4
        + jacobian[base._warp_jacobian_index(edge, 5, col)] * dl5
    )


@wp.kernel
def _warp_apply_direct_corrections(
    predicted_positions: wp.array(dtype=wp.vec3),
    predicted_orientations: wp.array(dtype=wp.quat),
    inv_masses: wp.array(dtype=wp.float32),
    quat_inv_masses: wp.array(dtype=wp.float32),
    jacobian_pos: wp.array(dtype=wp.float32),
    jacobian_rot: wp.array(dtype=wp.float32),
    delta_lambda: wp.array(dtype=wp.float32),
    lambda_sum: wp.array(dtype=wp.float32),
    n_edges: int,
    max_delta_out: wp.array(dtype=wp.float32),
    max_corr_out: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid != 0:
        return

    max_delta = float(0.0)
    max_corr = float(0.0)

    for edge in range(n_edges):
        base_idx = edge * 6
        dl0 = delta_lambda[base_idx + 0]
        dl1 = delta_lambda[base_idx + 1]
        dl2 = delta_lambda[base_idx + 2]
        dl3 = delta_lambda[base_idx + 3]
        dl4 = delta_lambda[base_idx + 4]
        dl5 = delta_lambda[base_idx + 5]

        lambda_sum[base_idx + 0] = lambda_sum[base_idx + 0] + dl0
        lambda_sum[base_idx + 1] = lambda_sum[base_idx + 1] + dl1
        lambda_sum[base_idx + 2] = lambda_sum[base_idx + 2] + dl2
        lambda_sum[base_idx + 3] = lambda_sum[base_idx + 3] + dl3
        lambda_sum[base_idx + 4] = lambda_sum[base_idx + 4] + dl4
        lambda_sum[base_idx + 5] = lambda_sum[base_idx + 5] + dl5

        abs_dl = wp.abs(dl0)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl1)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl2)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl3)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl4)
        if abs_dl > max_delta:
            max_delta = abs_dl
        abs_dl = wp.abs(dl5)
        if abs_dl > max_delta:
            max_delta = abs_dl

        inv_m0 = inv_masses[edge]
        inv_m1 = inv_masses[edge + 1]

        if inv_m0 > 0.0:
            dp0_x = _warp_jacobian_dot(jacobian_pos, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_y = _warp_jacobian_dot(jacobian_pos, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0_z = _warp_jacobian_dot(jacobian_pos, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5)
            dp0 = wp.vec3(dp0_x * inv_m0, dp0_y * inv_m0, dp0_z * inv_m0)
            predicted_positions[edge] = predicted_positions[edge] + dp0
            corr = wp.sqrt(dp0.x * dp0.x + dp0.y * dp0.y + dp0.z * dp0.z)
            if corr > max_corr:
                max_corr = corr

        if inv_m1 > 0.0:
            dp1_x = _warp_jacobian_dot(jacobian_pos, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_y = _warp_jacobian_dot(jacobian_pos, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1_z = _warp_jacobian_dot(jacobian_pos, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5)
            dp1 = wp.vec3(dp1_x * inv_m1, dp1_y * inv_m1, dp1_z * inv_m1)
            predicted_positions[edge + 1] = predicted_positions[edge + 1] + dp1
            corr = wp.sqrt(dp1.x * dp1.x + dp1.y * dp1.y + dp1.z * dp1.z)
            if corr > max_corr:
                max_corr = corr

        if quat_inv_masses[edge] > 0.0:
            dtheta0 = wp.vec3(
                _warp_jacobian_dot(jacobian_rot, edge, 0, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 1, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 2, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            corr = wp.sqrt(dtheta0.x * dtheta0.x + dtheta0.y * dtheta0.y + dtheta0.z * dtheta0.z)
            if corr > max_corr:
                max_corr = corr
            predicted_orientations[edge] = _warp_quat_correction_g(predicted_orientations[edge], dtheta0)

        if quat_inv_masses[edge + 1] > 0.0:
            dtheta1 = wp.vec3(
                _warp_jacobian_dot(jacobian_rot, edge, 3, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 4, dl0, dl1, dl2, dl3, dl4, dl5),
                _warp_jacobian_dot(jacobian_rot, edge, 5, dl0, dl1, dl2, dl3, dl4, dl5),
            )
            corr = wp.sqrt(dtheta1.x * dtheta1.x + dtheta1.y * dtheta1.y + dtheta1.z * dtheta1.z)
            if corr > max_corr:
                max_corr = corr
            predicted_orientations[edge + 1] = _warp_quat_correction_g(predicted_orientations[edge + 1], dtheta1)

    max_delta_out[0] = max_delta
    max_corr_out[0] = max_corr


@wp.kernel
def _warp_constraint_max(
    constraint_values: wp.array(dtype=wp.float32),
    n_edges: int,
    out_max: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid != 0:
        return
    max_val = float(0.0)
    for edge in range(n_edges):
        base_idx = edge * 6
        norm_sq = float(0.0)
        for j in range(6):
            val = constraint_values[base_idx + j]
            norm_sq += val * val
        norm = wp.sqrt(norm_sq)
        if norm > max_val:
            max_val = norm
    out_max[0] = max_val


@wp.kernel
def _warp_zero_float(arr: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    arr[tid] = 0.0


@wp.kernel
def _warp_zero_2d(arr: wp.array2d(dtype=wp.float32), rows: int, cols: int):
    tid = wp.tid()
    if tid < rows * cols:
        row = tid // cols
        col = tid - row * cols
        arr[row, col] = 0.0


@wp.kernel
def _warp_copy_with_offset(
    src: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    start: int,
    dst: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    dst[start + i] = src[i] + offset


@wp.kernel
def _warp_build_segment_lines(
    positions: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    starts[i] = positions[i] + offset
    ends[i] = positions[i + 1] + offset


@wp.kernel
def _warp_apply_floor_collisions(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    min_z: float,
    restitution: float,
):
    i = wp.tid()
    pos = positions[i]
    if pos.z < min_z:
        clamped = wp.vec3(pos.x, pos.y, min_z)
        positions[i] = clamped
        predicted[i] = clamped
        vel = velocities[i]
        if vel.z < 0.0:
            velocities[i] = wp.vec3(vel.x, vel.y, -restitution * vel.z)


@wp.kernel
def _warp_apply_root_translation(
    positions: wp.array(dtype=wp.vec3),
    predicted: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dx: float,
    dy: float,
    dz: float,
):
    tid = wp.tid()
    if tid != 0:
        return
    pos = positions[0]
    new_pos = wp.vec3(pos.x + dx, pos.y + dy, pos.z + dz)
    positions[0] = new_pos
    predicted[0] = new_pos
    velocities[0] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _warp_zero_root_velocities(
    velocities: wp.array(dtype=wp.vec3),
    angular_velocities: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid != 0:
        return
    velocities[0] = wp.vec3(0.0, 0.0, 0.0)
    angular_velocities[0] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _warp_set_root_orientation(
    orientations: wp.array(dtype=wp.quat),
    predicted: wp.array(dtype=wp.quat),
    prev: wp.array(dtype=wp.quat),
    q: wp.quat,
):
    tid = wp.tid()
    if tid != 0:
        return
    orientations[0] = q
    predicted[0] = q
    prev[0] = q


class WarpResidentRodState(base.DefKitDirectRodState):
    """Warp-resident direct rod state that stays on device."""

    def __init__(self, *args, device: wp.Device | None = None, **kwargs):
        self._warp_ready = False
        super().__init__(*args, **kwargs)
        self.device = device or wp.get_device()
        self.direct_solve_backend = DIRECT_SOLVE_WARP_BLOCK_THOMAS
        self.supports_non_banded = True

        self.last_constraint_max = 0.0
        self.last_delta_lambda_max = 0.0
        self.last_correction_max = 0.0

        self._init_warp_state()
        self._sync_from_host_all()
        self._warp_ready = True

    def _init_warp_state(self):
        self.positions_wp = wp.zeros(self.num_points, dtype=wp.vec3, device=self.device)
        self.predicted_positions_wp = wp.zeros(self.num_points, dtype=wp.vec3, device=self.device)
        self.velocities_wp = wp.zeros(self.num_points, dtype=wp.vec3, device=self.device)
        self.forces_wp = wp.zeros(self.num_points, dtype=wp.vec3, device=self.device)

        self.orientations_wp = wp.zeros(self.num_points, dtype=wp.quat, device=self.device)
        self.predicted_orientations_wp = wp.zeros(self.num_points, dtype=wp.quat, device=self.device)
        self.prev_orientations_wp = wp.zeros(self.num_points, dtype=wp.quat, device=self.device)
        self.angular_velocities_wp = wp.zeros(self.num_points, dtype=wp.vec3, device=self.device)
        self.torques_wp = wp.zeros(self.num_points, dtype=wp.vec3, device=self.device)

        self.inv_masses_wp = wp.zeros(self.num_points, dtype=wp.float32, device=self.device)
        self.quat_inv_masses_wp = wp.zeros(self.num_points, dtype=wp.float32, device=self.device)

        self.rest_lengths_wp = wp.zeros(self.num_edges, dtype=wp.float32, device=self.device)
        self.rest_darboux_wp = wp.zeros(self.num_edges, dtype=wp.vec3, device=self.device)
        self.bend_stiffness_wp = wp.zeros(self.num_edges, dtype=wp.vec3, device=self.device)

        self.n_dofs = self.num_edges * 6
        alloc_dofs = max(1, self.n_dofs)
        alloc_edges = max(1, self.num_edges)

        self.constraint_values_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.compliance_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.lambda_sum_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.jacobian_pos_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.jacobian_rot_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)

        self.ab_wp = wp.zeros((BAND_LDAB, alloc_dofs), dtype=wp.float32, device=self.device)
        self.A_wp = wp.zeros((TILE, TILE), dtype=wp.float32, device=self.device)
        self.rhs_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.rhs_tile_wp = wp.zeros(TILE, dtype=wp.float32, device=self.device)
        self.delta_lambda_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.delta_lambda_tile_wp = wp.zeros(TILE, dtype=wp.float32, device=self.device)

        self.diag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.offdiag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.c_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.d_prime_wp = wp.zeros(alloc_edges * 6, dtype=wp.float32, device=self.device)

        self._constraint_max_wp = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._delta_lambda_max_wp = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._correction_max_wp = wp.zeros(1, dtype=wp.float32, device=self.device)

    def _sync_from_host_all(self):
        self.positions_wp.assign(wp.array(self.positions[:, 0:3], dtype=wp.vec3, device=self.device))
        self.predicted_positions_wp.assign(
            wp.array(self.predicted_positions[:, 0:3], dtype=wp.vec3, device=self.device)
        )
        self.velocities_wp.assign(wp.array(self.velocities[:, 0:3], dtype=wp.vec3, device=self.device))
        self.forces_wp.assign(wp.array(self.forces[:, 0:3], dtype=wp.vec3, device=self.device))

        self.orientations_wp.assign(wp.array(self.orientations, dtype=wp.quat, device=self.device))
        self.predicted_orientations_wp.assign(
            wp.array(self.predicted_orientations, dtype=wp.quat, device=self.device)
        )
        self.prev_orientations_wp.assign(wp.array(self.prev_orientations, dtype=wp.quat, device=self.device))
        self.angular_velocities_wp.assign(
            wp.array(self.angular_velocities[:, 0:3], dtype=wp.vec3, device=self.device)
        )
        self.torques_wp.assign(wp.array(self.torques[:, 0:3], dtype=wp.vec3, device=self.device))

        self.inv_masses_wp.assign(wp.array(self.inv_masses, dtype=wp.float32, device=self.device))
        self.quat_inv_masses_wp.assign(wp.array(self.quat_inv_masses, dtype=wp.float32, device=self.device))

        if self.num_edges > 0:
            self.rest_lengths_wp.assign(wp.array(self.rest_lengths, dtype=wp.float32, device=self.device))
            self.rest_darboux_wp.assign(
                wp.array(self.rest_darboux[:, 0:3], dtype=wp.vec3, device=self.device)
            )
            self.bend_stiffness_wp.assign(
                wp.array(self.bend_stiffness[:, 0:3], dtype=wp.vec3, device=self.device)
            )

    def set_solver_mode(self, use_banded: bool):
        self.use_banded = use_banded
        if use_banded:
            self.direct_solve_backend = DIRECT_SOLVE_WARP_BANDED_CHOLESKY
        elif self.direct_solve_backend == DIRECT_SOLVE_WARP_BANDED_CHOLESKY:
            self.direct_solve_backend = DIRECT_SOLVE_WARP_BLOCK_THOMAS

    def set_direct_solve_backend(self, backend: str):
        if backend not in DIRECT_SOLVE_BACKENDS:
            raise ValueError(f"Unknown direct solve backend: {backend}")
        self.direct_solve_backend = backend

    def set_gravity(self, gravity: np.ndarray):
        super().set_gravity(gravity)

    def set_bend_stiffness(self, bend_stiffness: float, twist_stiffness: float):
        super().set_bend_stiffness(bend_stiffness, twist_stiffness)
        if not getattr(self, "_warp_ready", False):
            return
        if self.num_edges > 0:
            self.bend_stiffness_wp.assign(
                wp.array(self.bend_stiffness[:, 0:3], dtype=wp.vec3, device=self.device)
            )

    def set_rest_darboux(self, rest_bend_d1: float, rest_bend_d2: float, rest_twist: float):
        super().set_rest_darboux(rest_bend_d1, rest_bend_d2, rest_twist)
        if not getattr(self, "_warp_ready", False):
            return
        if self.num_edges > 0:
            self.rest_darboux_wp.assign(
                wp.array(self.rest_darboux[:, 0:3], dtype=wp.vec3, device=self.device)
            )

    def set_root_locked(self, locked: bool):
        super().set_root_locked(locked)
        self.inv_masses_wp.assign(wp.array(self.inv_masses, dtype=wp.float32, device=self.device))
        self.quat_inv_masses_wp.assign(wp.array(self.quat_inv_masses, dtype=wp.float32, device=self.device))
        if self.num_points > 0:
            wp.launch(
                _warp_zero_root_velocities,
                dim=1,
                inputs=[self.velocities_wp, self.angular_velocities_wp],
                device=self.device,
            )

    def reset(self):
        super().reset()
        self._sync_from_host_all()

    def apply_root_translation(self, dx: float, dy: float, dz: float):
        if self.num_points == 0:
            return
        wp.launch(
            _warp_apply_root_translation,
            dim=1,
            inputs=[self.positions_wp, self.predicted_positions_wp, self.velocities_wp, dx, dy, dz],
            device=self.device,
        )

    def apply_root_rotation(self, q: np.ndarray):
        if self.num_points == 0:
            return
        q_wp = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        wp.launch(
            _warp_set_root_orientation,
            dim=1,
            inputs=[self.orientations_wp, self.predicted_orientations_wp, self.prev_orientations_wp, q_wp],
            device=self.device,
        )

    def positions_numpy(self) -> np.ndarray:
        return self.positions_wp.numpy()

    def velocities_numpy(self) -> np.ndarray:
        return self.velocities_wp.numpy()

    def orientations_numpy(self) -> np.ndarray:
        return self.orientations_wp.numpy()

    def predict_positions(self, dt: float, linear_damping: float):
        if self.num_points == 0:
            return
        gravity = wp.vec3(
            float(self.gravity[0, 0]),
            float(self.gravity[0, 1]),
            float(self.gravity[0, 2]),
        )
        wp.launch(
            base._warp_predict_positions,
            dim=self.num_points,
            inputs=[
                self.positions_wp,
                self.velocities_wp,
                self.forces_wp,
                self.inv_masses_wp,
                gravity,
                float(dt),
                float(linear_damping),
                self.predicted_positions_wp,
            ],
            device=self.device,
        )

    def integrate_positions(self, dt: float):
        if self.num_points == 0:
            return
        wp.launch(
            base._warp_integrate_positions,
            dim=self.num_points,
            inputs=[
                self.positions_wp,
                self.predicted_positions_wp,
                self.velocities_wp,
                self.inv_masses_wp,
                float(dt),
            ],
            device=self.device,
        )

    def predict_rotations(self, dt: float, angular_damping: float):
        if self.num_points == 0:
            return
        wp.launch(
            base._warp_predict_rotations,
            dim=self.num_points,
            inputs=[
                self.orientations_wp,
                self.angular_velocities_wp,
                self.torques_wp,
                self.quat_inv_masses_wp,
                float(dt),
                float(angular_damping),
                self.predicted_orientations_wp,
            ],
            device=self.device,
        )

    def integrate_rotations(self, dt: float):
        if self.num_points == 0:
            return
        wp.launch(
            base._warp_integrate_rotations,
            dim=self.num_points,
            inputs=[
                self.orientations_wp,
                self.predicted_orientations_wp,
                self.prev_orientations_wp,
                self.angular_velocities_wp,
                self.quat_inv_masses_wp,
                float(dt),
            ],
            device=self.device,
        )

    def prepare_constraints(self, dt: float):
        if self.num_edges == 0:
            return
        wp.launch(
            _warp_zero_float,
            dim=self.n_dofs,
            inputs=[self.lambda_sum_wp],
            device=self.device,
        )
        wp.launch(
            base._warp_prepare_compliance,
            dim=self.num_edges,
            inputs=[
                self.rest_lengths_wp,
                self.bend_stiffness_wp,
                float(self.young_modulus),
                float(self.torsion_modulus),
                float(dt),
                self.compliance_wp,
            ],
            device=self.device,
        )

    def project_direct(self):
        if self.num_edges == 0:
            self.last_constraint_max = 0.0
            self.last_delta_lambda_max = 0.0
            self.last_correction_max = 0.0
            return

        wp.launch(
            base._warp_update_constraints_direct,
            dim=self.num_edges,
            inputs=[
                self.predicted_positions_wp,
                self.predicted_orientations_wp,
                self.rest_lengths_wp,
                self.rest_darboux_wp,
                self.constraint_values_wp,
            ],
            device=self.device,
        )
        wp.launch(
            _warp_constraint_max,
            dim=1,
            inputs=[self.constraint_values_wp, int(self.num_edges), self._constraint_max_wp],
            device=self.device,
        )

        wp.launch(
            base._warp_compute_jacobians_direct,
            dim=self.num_edges,
            inputs=[
                self.predicted_orientations_wp,
                self.rest_lengths_wp,
                self.jacobian_pos_wp,
                self.jacobian_rot_wp,
            ],
            device=self.device,
        )

        n_dofs = self.n_dofs
        if self.direct_solve_backend == DIRECT_SOLVE_WARP_BANDED_CHOLESKY:
            wp.launch(
                _warp_zero_2d,
                dim=BAND_LDAB * max(1, n_dofs),
                inputs=[self.ab_wp, int(BAND_LDAB), int(max(1, n_dofs))],
                device=self.device,
            )
            wp.launch(
                base._warp_assemble_jmjt_banded,
                dim=self.num_edges,
                inputs=[self.jacobian_pos_wp, self.jacobian_rot_wp, self.compliance_wp, int(n_dofs), self.ab_wp],
                device=self.device,
            )
            wp.launch(
                base._warp_build_rhs,
                dim=n_dofs,
                inputs=[self.constraint_values_wp, self.compliance_wp, self.lambda_sum_wp, int(n_dofs), self.rhs_wp],
                device=self.device,
            )
            wp.launch(
                base._warp_spbsv_u11_1rhs,
                dim=1,
                inputs=[int(n_dofs), self.ab_wp, self.rhs_wp],
                device=self.device,
            )
            delta_lambda = self.rhs_wp
        elif n_dofs <= TILE:
            wp.launch(
                _warp_zero_2d,
                dim=TILE * TILE,
                inputs=[self.A_wp, int(TILE), int(TILE)],
                device=self.device,
            )
            wp.launch(
                base._warp_assemble_jmjt_dense,
                dim=self.num_edges,
                inputs=[self.jacobian_pos_wp, self.jacobian_rot_wp, self.compliance_wp, int(n_dofs), self.A_wp],
                device=self.device,
            )
            wp.launch(
                base._warp_build_rhs,
                dim=TILE,
                inputs=[self.constraint_values_wp, self.compliance_wp, self.lambda_sum_wp, int(n_dofs), self.rhs_tile_wp],
                device=self.device,
            )
            if n_dofs < TILE:
                wp.launch(
                    base._warp_pad_diagonal,
                    dim=TILE,
                    inputs=[self.A_wp, int(n_dofs), int(TILE)],
                    device=self.device,
                )
            wp.launch_tiled(
                base._warp_cholesky_solve_tile,
                dim=[1, 1],
                inputs=[self.A_wp, self.rhs_tile_wp],
                outputs=[self.delta_lambda_tile_wp],
                block_dim=BLOCK_DIM,
                device=self.device,
            )
            delta_lambda = self.delta_lambda_tile_wp
        else:
            wp.launch(
                base._warp_assemble_jmjt_blocks,
                dim=self.num_edges,
                inputs=[
                    self.jacobian_pos_wp,
                    self.jacobian_rot_wp,
                    self.compliance_wp,
                    int(self.num_edges),
                    self.diag_blocks_wp,
                    self.offdiag_blocks_wp,
                ],
                device=self.device,
            )
            wp.launch(
                base._warp_build_rhs,
                dim=n_dofs,
                inputs=[self.constraint_values_wp, self.compliance_wp, self.lambda_sum_wp, int(n_dofs), self.rhs_wp],
                device=self.device,
            )
            wp.launch(
                base._warp_block_thomas_solve,
                dim=1,
                inputs=[
                    self.diag_blocks_wp,
                    self.offdiag_blocks_wp,
                    self.rhs_wp,
                    int(self.num_edges),
                    self.c_blocks_wp,
                    self.d_prime_wp,
                    self.delta_lambda_wp,
                ],
                device=self.device,
            )
            delta_lambda = self.delta_lambda_wp

        wp.launch(
            _warp_apply_direct_corrections,
            dim=1,
            inputs=[
                self.predicted_positions_wp,
                self.predicted_orientations_wp,
                self.inv_masses_wp,
                self.quat_inv_masses_wp,
                self.jacobian_pos_wp,
                self.jacobian_rot_wp,
                delta_lambda,
                self.lambda_sum_wp,
                int(self.num_edges),
                self._delta_lambda_max_wp,
                self._correction_max_wp,
            ],
            device=self.device,
        )

        self.last_constraint_max = float(self._constraint_max_wp.numpy()[0])
        self.last_delta_lambda_max = float(self._delta_lambda_max_wp.numpy()[0])
        self.last_correction_max = float(self._correction_max_wp.numpy()[0])

    def apply_floor_collisions(self, floor_z: float, restitution: float = 0.0):
        if self.num_points == 0:
            return
        min_z = float(floor_z + self.rod_radius)
        wp.launch(
            _warp_apply_floor_collisions,
            dim=self.num_points,
            inputs=[self.positions_wp, self.predicted_positions_wp, self.velocities_wp, min_z, float(restitution)],
            device=self.device,
        )

    def step(self, dt: float, linear_damping: float, angular_damping: float):
        self.predict_positions(dt, linear_damping)
        self.predict_rotations(dt, angular_damping)
        self.prepare_constraints(dt)
        self.project_direct()
        self.integrate_positions(dt)
        self.integrate_rotations(dt)


class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.substeps = args.substeps
        self.linear_damping = args.linear_damping
        self.angular_damping = args.angular_damping
        self.bend_stiffness = args.bend_stiffness
        self.twist_stiffness = args.twist_stiffness
        self.rest_bend_d1 = args.rest_bend_d1
        self.rest_bend_d2 = args.rest_bend_d2
        self.rest_twist = args.rest_twist
        self.young_modulus_scale = args.young_modulus / 1.0e6
        self.torsion_modulus_scale = args.torsion_modulus / 1.0e6
        self.use_banded = args.use_banded
        self.compare_offset = args.compare_offset
        half_offset = 0.5 * self.compare_offset
        self.ref_offset = np.array([0.0, -half_offset, 0.0], dtype=np.float32)
        self.gpu_offset = np.array([0.0, half_offset, 0.0], dtype=np.float32)

        self.base_gravity = np.array(args.gravity, dtype=np.float32)
        self.gravity_enabled = True
        self.gravity_scale = 1.0
        self.floor_collision_enabled = True
        self.floor_height = 0.0
        self.floor_restitution = 0.0

        self.show_segments = True
        self.show_directors = False
        self.director_scale = 0.1

        self.root_move_speed = 1.0
        self.root_rotate_speed = 1.0
        self.root_rotation = 0.0

        self._gravity_key_was_down = False
        self._reset_key_was_down = False
        self._banded_key_was_down = False
        self._lock_key_was_down = False

        self.lib = base.DefKitDirectLibrary(args.dll_path, args.calling_convention)
        self.supports_non_banded = self.lib.ProjectDirectElasticRodConstraints is not None
        if not self.supports_non_banded:
            self.use_banded = True

        rod_radius = args.rod_radius if args.rod_radius is not None else args.particle_radius
        self.ref_rod = base.DefKitDirectRodState(
            lib=self.lib,
            num_points=args.num_points,
            segment_length=args.segment_length,
            mass=args.particle_mass,
            particle_height=args.particle_height,
            rod_radius=rod_radius,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            rest_bend_d1=self.rest_bend_d1,
            rest_bend_d2=self.rest_bend_d2,
            rest_twist=self.rest_twist,
            young_modulus=args.young_modulus,
            torsion_modulus=args.torsion_modulus,
            gravity=self.base_gravity,
            lock_root_rotation=args.lock_root_rotation,
            use_banded=self.use_banded,
        )
        self.gpu_rod = WarpResidentRodState(
            lib=self.lib,
            num_points=args.num_points,
            segment_length=args.segment_length,
            mass=args.particle_mass,
            particle_height=args.particle_height,
            rod_radius=rod_radius,
            bend_stiffness=self.bend_stiffness,
            twist_stiffness=self.twist_stiffness,
            rest_bend_d1=self.rest_bend_d1,
            rest_bend_d2=self.rest_bend_d2,
            rest_twist=self.rest_twist,
            young_modulus=args.young_modulus,
            torsion_modulus=args.torsion_modulus,
            gravity=self.base_gravity,
            lock_root_rotation=args.lock_root_rotation,
            use_banded=self.use_banded,
        )

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        for i in range(args.num_points):
            mass = 0.0 if i == 0 else args.particle_mass
            ref_pos = tuple(self.ref_rod.positions[i, 0:3] + self.ref_offset)
            builder.add_particle(pos=ref_pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=args.particle_radius)
        for i in range(args.num_points):
            mass = 0.0 if i == 0 else args.particle_mass
            gpu_pos = tuple(self.gpu_rod.positions[i, 0:3] + self.gpu_offset)
            builder.add_particle(pos=gpu_pos, vel=(0.0, 0.0, 0.0), mass=mass, radius=args.particle_radius)

        self.model = builder.finalize()
        self.state = self.model.state()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self._ref_segment_colors = np.tile(np.array([0.2, 0.6, 1.0], dtype=np.float32), (args.num_points - 1, 1))
        self._gpu_segment_colors = np.tile(np.array([1.0, 0.6, 0.2], dtype=np.float32), (args.num_points - 1, 1))

        device = self.model.device
        self._ref_positions_wp = wp.zeros(args.num_points, dtype=wp.vec3, device=device)
        self._ref_velocities_wp = wp.zeros(args.num_points, dtype=wp.vec3, device=device)
        self._ref_segment_starts_wp = wp.zeros(args.num_points - 1, dtype=wp.vec3, device=device)
        self._ref_segment_ends_wp = wp.zeros(args.num_points - 1, dtype=wp.vec3, device=device)
        self._gpu_segment_starts_wp = wp.zeros(args.num_points - 1, dtype=wp.vec3, device=device)
        self._gpu_segment_ends_wp = wp.zeros(args.num_points - 1, dtype=wp.vec3, device=device)
        self._ref_segment_colors_wp = wp.array(self._ref_segment_colors, dtype=wp.vec3, device=device)
        self._gpu_segment_colors_wp = wp.array(self._gpu_segment_colors, dtype=wp.vec3, device=device)

        self._sync_state_from_rods()
        self._update_gravity()
        self._ref_root_base_orientation = self.ref_rod.orientations[0].copy()
        self._gpu_root_base_orientation = self.gpu_rod.orientations[0].copy()

    def __del__(self):
        if hasattr(self, "ref_rod"):
            self.ref_rod.destroy()
        if hasattr(self, "gpu_rod"):
            self.gpu_rod.destroy()

    def _update_gravity(self):
        if self.gravity_enabled:
            gravity = self.base_gravity * self.gravity_scale
        else:
            gravity = np.zeros(3, dtype=np.float32)
        self.ref_rod.set_gravity(gravity)
        self.gpu_rod.set_gravity(gravity)

    def _sync_state_from_rods(self):
        ref_positions = self.ref_rod.positions[:, 0:3].astype(np.float32)
        ref_velocities = self.ref_rod.velocities[:, 0:3].astype(np.float32)
        self._ref_positions_wp.assign(wp.array(ref_positions, dtype=wp.vec3, device=self.model.device))
        self._ref_velocities_wp.assign(wp.array(ref_velocities, dtype=wp.vec3, device=self.model.device))

        ref_offset = wp.vec3(float(self.ref_offset[0]), float(self.ref_offset[1]), float(self.ref_offset[2]))
        gpu_offset = wp.vec3(float(self.gpu_offset[0]), float(self.gpu_offset[1]), float(self.gpu_offset[2]))

        wp.launch(
            _warp_copy_with_offset,
            dim=self.ref_rod.num_points,
            inputs=[self._ref_positions_wp, ref_offset, 0, self.state.particle_q],
            device=self.model.device,
        )
        wp.launch(
            _warp_copy_with_offset,
            dim=self.gpu_rod.num_points,
            inputs=[self.gpu_rod.positions_wp, gpu_offset, self.ref_rod.num_points, self.state.particle_q],
            device=self.model.device,
        )

        zero_offset = wp.vec3(0.0, 0.0, 0.0)
        wp.launch(
            _warp_copy_with_offset,
            dim=self.ref_rod.num_points,
            inputs=[self._ref_velocities_wp, zero_offset, 0, self.state.particle_qd],
            device=self.model.device,
        )
        wp.launch(
            _warp_copy_with_offset,
            dim=self.gpu_rod.num_points,
            inputs=[self.gpu_rod.velocities_wp, zero_offset, self.ref_rod.num_points, self.state.particle_qd],
            device=self.model.device,
        )

    def _handle_keyboard_input(self):
        if not hasattr(self.viewer, "is_key_down"):
            return

        try:
            import pyglet.window.key as key
        except ImportError:
            return

        g_down = self.viewer.is_key_down(key.G)
        if g_down and not self._gravity_key_was_down:
            self.gravity_enabled = not self.gravity_enabled
            self._update_gravity()
        self._gravity_key_was_down = g_down

        b_down = self.viewer.is_key_down(key.B)
        if b_down and not self._banded_key_was_down:
            if self.supports_non_banded:
                self.use_banded = not self.use_banded
                self.ref_rod.set_solver_mode(self.use_banded)
                self.gpu_rod.set_solver_mode(self.use_banded)
                self.use_banded = self.ref_rod.use_banded
        self._banded_key_was_down = b_down

        l_down = self.viewer.is_key_down(key.L)
        if l_down and not self._lock_key_was_down:
            self.ref_rod.toggle_root_lock()
            self.gpu_rod.toggle_root_lock()
        self._lock_key_was_down = l_down

        r_down = self.viewer.is_key_down(key.R)
        if r_down and not self._reset_key_was_down:
            self.ref_rod.reset()
            self.gpu_rod.reset()
            self.root_rotation = 0.0
            self._apply_root_rotation()
            self.sim_time = 0.0
            self._sync_state_from_rods()
        self._reset_key_was_down = r_down

        dx = 0.0
        dy = 0.0
        dz = 0.0

        if self.viewer.is_key_down(key.NUM_6):
            dx += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_4):
            dx -= self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_8):
            dy += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_2):
            dy -= self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_9):
            dz += self.root_move_speed * self.frame_dt
        if self.viewer.is_key_down(key.NUM_3):
            dz -= self.root_move_speed * self.frame_dt

        rotation_changed = False
        if self.viewer.is_key_down(key.NUM_7):
            self.root_rotation += self.root_rotate_speed * self.frame_dt
            rotation_changed = True
        if self.viewer.is_key_down(key.NUM_1):
            self.root_rotation -= self.root_rotate_speed * self.frame_dt
            rotation_changed = True

        if dx != 0.0 or dy != 0.0 or dz != 0.0:
            self._apply_root_translation(dx, dy, dz)
        if rotation_changed:
            self._apply_root_rotation()

    def step(self):
        self._handle_keyboard_input()

        sub_dt = self.frame_dt / max(self.substeps, 1)
        for _ in range(self.substeps):
            self.ref_rod.step(sub_dt, self.linear_damping, self.angular_damping)
            self.gpu_rod.step(sub_dt, self.linear_damping, self.angular_damping)
            if self.floor_collision_enabled:
                self.ref_rod.apply_floor_collisions(self.floor_height, self.floor_restitution)
                self.gpu_rod.apply_floor_collisions(self.floor_height, self.floor_restitution)

        self._sync_state_from_rods()
        self.sim_time += self.frame_dt

    def _apply_root_translation(self, dx: float, dy: float, dz: float):
        delta = np.array([dx, dy, dz], dtype=np.float32)

        pos = self.ref_rod.positions[0, 0:3]
        new_pos = pos + delta
        self.ref_rod.positions[0, 0:3] = new_pos
        self.ref_rod.predicted_positions[0, 0:3] = new_pos
        self.ref_rod.velocities[0, 0:3] = 0.0

        self.gpu_rod.apply_root_translation(dx, dy, dz)

    def _apply_root_rotation(self):
        q_twist = base._quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), self.root_rotation)
        q_ref = base._quat_multiply(self._ref_root_base_orientation, q_twist)
        q_gpu = base._quat_multiply(self._gpu_root_base_orientation, q_twist)
        self.ref_rod.orientations[0] = q_ref
        self.ref_rod.predicted_orientations[0] = q_ref
        self.ref_rod.prev_orientations[0] = q_ref
        self.gpu_rod.apply_root_rotation(q_gpu)

    def _rotate_vector_by_quaternion(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        x, y, z, w = q
        vx, vy, vz = v

        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)

        return np.array(
            [
                vx + w * tx + y * tz - z * ty,
                vy + w * ty + z * tx - x * tz,
                vz + w * tz + x * ty - y * tx,
            ],
            dtype=np.float32,
        )

    def _build_director_lines(self, positions: np.ndarray, orientations: np.ndarray, offset: np.ndarray):
        num_edges = positions.shape[0] - 1
        positions = positions + offset

        starts = np.zeros((num_edges * 3, 3), dtype=np.float32)
        ends = np.zeros((num_edges * 3, 3), dtype=np.float32)
        colors = np.zeros((num_edges * 3, 3), dtype=np.float32)

        for i in range(num_edges):
            midpoint = 0.5 * (positions[i] + positions[i + 1])
            q = orientations[i]

            d1 = self._rotate_vector_by_quaternion(np.array([1.0, 0.0, 0.0], dtype=np.float32), q)
            d2 = self._rotate_vector_by_quaternion(np.array([0.0, 1.0, 0.0], dtype=np.float32), q)
            d3 = self._rotate_vector_by_quaternion(np.array([0.0, 0.0, 1.0], dtype=np.float32), q)

            base_idx = i * 3
            starts[base_idx] = midpoint
            ends[base_idx] = midpoint + d1 * self.director_scale
            colors[base_idx] = [1.0, 0.0, 0.0]

            starts[base_idx + 1] = midpoint
            ends[base_idx + 1] = midpoint + d2 * self.director_scale
            colors[base_idx + 1] = [0.0, 1.0, 0.0]

            starts[base_idx + 2] = midpoint
            ends[base_idx + 2] = midpoint + d3 * self.director_scale
            colors[base_idx + 2] = [0.0, 0.0, 1.0]

        return starts, ends, colors

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)

        if self.show_segments:
            ref_offset = wp.vec3(float(self.ref_offset[0]), float(self.ref_offset[1]), float(self.ref_offset[2]))
            gpu_offset = wp.vec3(float(self.gpu_offset[0]), float(self.gpu_offset[1]), float(self.gpu_offset[2]))
            wp.launch(
                _warp_build_segment_lines,
                dim=self.ref_rod.num_points - 1,
                inputs=[self._ref_positions_wp, ref_offset, self._ref_segment_starts_wp, self._ref_segment_ends_wp],
                device=self.model.device,
            )
            wp.launch(
                _warp_build_segment_lines,
                dim=self.gpu_rod.num_points - 1,
                inputs=[self.gpu_rod.positions_wp, gpu_offset, self._gpu_segment_starts_wp, self._gpu_segment_ends_wp],
                device=self.model.device,
            )
            self.viewer.log_lines(
                "/rod_reference",
                self._ref_segment_starts_wp,
                self._ref_segment_ends_wp,
                self._ref_segment_colors_wp,
            )
            self.viewer.log_lines(
                "/rod_gpu",
                self._gpu_segment_starts_wp,
                self._gpu_segment_ends_wp,
                self._gpu_segment_colors_wp,
            )
        else:
            self.viewer.log_lines("/rod_reference", None, None, None)
            self.viewer.log_lines("/rod_gpu", None, None, None)

        if self.show_directors:
            ref_positions = self.ref_rod.positions[:, 0:3].astype(np.float32)
            ref_orientations = self.ref_rod.orientations.astype(np.float32)
            gpu_positions = self.gpu_rod.positions_numpy().astype(np.float32)
            gpu_orientations = self.gpu_rod.orientations_numpy().astype(np.float32)

            ref_starts, ref_ends, ref_colors = self._build_director_lines(
                ref_positions, ref_orientations, self.ref_offset
            )
            gpu_starts, gpu_ends, gpu_colors = self._build_director_lines(
                gpu_positions, gpu_orientations, self.gpu_offset
            )
            self.viewer.log_lines(
                "/directors_reference",
                wp.array(ref_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(ref_colors, dtype=wp.vec3, device=self.model.device),
            )
            self.viewer.log_lines(
                "/directors_gpu",
                wp.array(gpu_starts, dtype=wp.vec3, device=self.model.device),
                wp.array(gpu_ends, dtype=wp.vec3, device=self.model.device),
                wp.array(gpu_colors, dtype=wp.vec3, device=self.model.device),
            )
        else:
            self.viewer.log_lines("/directors_reference", None, None, None)
            self.viewer.log_lines("/directors_gpu", None, None, None)

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Direct Cosserat Rod: Reference + GPU Warp")
        ui.text(f"Particles per rod: {self.ref_rod.num_points}")
        ui.text("Reference: blue, GPU: orange")
        ui.separator()

        _changed, self.substeps = ui.slider_int("Substeps", self.substeps, 1, 16)
        _changed, self.linear_damping = ui.slider_float("Linear Damping", self.linear_damping, 0.0, 0.05)
        _changed, self.angular_damping = ui.slider_float("Angular Damping", self.angular_damping, 0.0, 0.05)

        ui.separator()
        offset_changed, self.compare_offset = ui.slider_float("Compare Offset", self.compare_offset, 0.1, 5.0)
        if offset_changed:
            half_offset = 0.5 * self.compare_offset
            self.ref_offset = np.array([0.0, -half_offset, 0.0], dtype=np.float32)
            self.gpu_offset = np.array([0.0, half_offset, 0.0], dtype=np.float32)
            self._sync_state_from_rods()

        ui.separator()
        changed_bend_k, self.bend_stiffness = ui.slider_float("Bend Stiffness", self.bend_stiffness, 0.0, 1.0)
        changed_twist_k, self.twist_stiffness = ui.slider_float(
            "Twist Stiffness", self.twist_stiffness, 0.0, 1.0
        )
        if changed_bend_k or changed_twist_k:
            self.ref_rod.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)
            self.gpu_rod.set_bend_stiffness(self.bend_stiffness, self.twist_stiffness)

        ui.separator()
        changed_rest_d1, self.rest_bend_d1 = ui.slider_float(
            "Rest Bend d1", self.rest_bend_d1, -0.5, 0.5
        )
        changed_rest_d2, self.rest_bend_d2 = ui.slider_float(
            "Rest Bend d2", self.rest_bend_d2, -0.5, 0.5
        )
        changed_rest_twist, self.rest_twist = ui.slider_float("Rest Twist", self.rest_twist, -0.5, 0.5)
        if changed_rest_d1 or changed_rest_d2 or changed_rest_twist:
            self.ref_rod.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)
            self.gpu_rod.set_rest_darboux(self.rest_bend_d1, self.rest_bend_d2, self.rest_twist)

        ui.separator()
        changed_E, self.young_modulus_scale = ui.slider_float("Young Modulus (1e6)", self.young_modulus_scale, 0.1, 5.0)
        changed_G, self.torsion_modulus_scale = ui.slider_float(
            "Torsion Modulus (1e6)", self.torsion_modulus_scale, 0.1, 5.0
        )
        if changed_E or changed_G:
            self.ref_rod.young_modulus = self.young_modulus_scale * 1.0e6
            self.ref_rod.torsion_modulus = self.torsion_modulus_scale * 1.0e6
            self.gpu_rod.young_modulus = self.young_modulus_scale * 1.0e6
            self.gpu_rod.torsion_modulus = self.torsion_modulus_scale * 1.0e6

        ui.separator()
        _changed, self.gravity_scale = ui.slider_float("Gravity Scale", self.gravity_scale, 0.0, 2.0)
        _changed, self.floor_height = ui.slider_float("Floor Height", self.floor_height, -1.0, 1.0)
        _changed, self.floor_restitution = ui.slider_float(
            "Floor Restitution", self.floor_restitution, 0.0, 1.0
        )

        ui.separator()
        if self.supports_non_banded:
            changed_banded, self.use_banded = ui.checkbox("Use Banded Solver", self.use_banded)
            if changed_banded:
                self.ref_rod.set_solver_mode(self.use_banded)
                self.gpu_rod.set_solver_mode(self.use_banded)
                self.use_banded = self.ref_rod.use_banded
        else:
            ui.text("Non-banded solver not available in this DLL build.")

        ui.separator()
        ui.text("GPU Direct Stabilization")
        ui.text(f"GPU max |C|: {self.gpu_rod.last_constraint_max:.3e}")
        ui.text(f"GPU max |Δλ|: {self.gpu_rod.last_delta_lambda_max:.3e}")
        ui.text(f"GPU max correction: {self.gpu_rod.last_correction_max:.3e}")

        ui.separator()
        _changed, self.show_segments = ui.checkbox("Show Rod Segments", self.show_segments)
        _changed, self.show_directors = ui.checkbox("Show Directors", self.show_directors)
        _changed, self.director_scale = ui.slider_float("Director Scale", self.director_scale, 0.01, 0.3)

        ui.separator()
        ui.text("Root Control (Numpad, both rods)")
        _changed, self.root_move_speed = ui.slider_float("Move Speed", self.root_move_speed, 0.1, 5.0)
        _changed, self.root_rotate_speed = ui.slider_float("Rotate Speed", self.root_rotate_speed, 0.1, 3.0)
        ui.text(f"  Rotation: {self.root_rotation:.2f} rad")
        ui.text("  4/6: X-, X+  8/2: Y+, Y-  9/3: Z+, Z-")
        ui.text("  7/1: Rotate +Z/-Z")

        ui.separator()
        ui.text("Controls:")
        ui.text("  G: Toggle gravity")
        ui.text("  B: Toggle banded solver")
        ui.text("  L: Toggle root lock (position + rotation)")
        ui.text("  R: Reset")

    def test_final(self):
        ref_anchor = self.ref_rod.positions[0, 0:3]
        ref_initial = self.ref_rod._initial_positions[0, 0:3]
        ref_dist = float(np.linalg.norm(ref_anchor - ref_initial))
        assert ref_dist < 1.0e-3, f"Reference anchor moved too far: {ref_dist}"

        gpu_positions = self.gpu_rod.positions_numpy()
        gpu_anchor = gpu_positions[0]
        gpu_initial = self.gpu_rod._initial_positions[0, 0:3]
        gpu_dist = float(np.linalg.norm(gpu_anchor - gpu_initial))
        assert gpu_dist < 1.0e-3, f"GPU anchor moved too far: {gpu_dist}"

        if not np.all(np.isfinite(self.ref_rod.positions[:, 0:3])):
            raise AssertionError("Non-finite reference positions detected")
        if not np.all(np.isfinite(gpu_positions)):
            raise AssertionError("Non-finite GPU positions detected")


def create_parser():
    return base.create_parser()


if __name__ == "__main__":
    viewer, args = newton.examples.init(create_parser())

    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    example = Example(viewer, args)
    newton.examples.run(example, args)
