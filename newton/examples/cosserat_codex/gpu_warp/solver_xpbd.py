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

"""Warp-resident direct Cosserat solver aligned with XPBD interfaces."""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from newton.examples.cosserat_codex import warp_cosserat_codex as base

from . import kernels

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

        self._enable_timers = True
        self._timers_use_nvtx = False
        self._timing_accum = {
            "integration": 0.0,
            "constraints_assembly": 0.0,
            "system_solve": 0.0,
            "final_position_update": 0.0,
        }
        self._timing_count = 0
        self._timing_last_report = time.perf_counter()

        self.use_cuda_graph = False
        self.use_iterative_refinement = False
        self.iterative_refinement_iters = 2
        self._graph = None
        self._graph_params = None
        self._graph_capture_active = False

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

        # Iterative refinement workspace arrays
        self.ab_orig_wp = wp.zeros((BAND_LDAB, alloc_dofs), dtype=wp.float32, device=self.device)
        self.b_orig_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)
        self.r_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=self.device)

        self.diag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.offdiag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.c_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=self.device)
        self.d_prime_wp = wp.zeros(alloc_edges * 6, dtype=wp.float32, device=self.device)

        # Inverse inertia tensors in world frame (3x3 per particle, stored as flat array of 9 floats per particle)
        # 
        # NOTE: The original Python implementation used UNIT inertia (1.0) for JMJT assembly,
        # which provides correct stiffness behavior with the Young's modulus and torsion modulus sliders.
        # The actual C++ binary being compared against may use different values than the source code.
        # 
        # Using unit mass (1.0) and unit inertia (1.0) means:
        #   - JMJT = J * J^T + compliance (effectively)
        #   - Corrections: position uses unit mass, rotation uses unit inertia
        #   - The stiffness is controlled entirely by the compliance terms (from Young's/torsion modulus)
        #
        alloc_points = max(1, self.num_points)
        self.inv_inertia_wp = wp.zeros(alloc_points * 9, dtype=wp.float32, device=self.device)

        # Use UNIT inertia (1.0, 1.0, 1.0) to match original behavior where sliders work correctly
        self.inv_inertia_local_diag = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self._constraint_max_wp = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._delta_lambda_max_wp = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._correction_max_wp = wp.zeros(1, dtype=wp.float32, device=self.device)

    def _timer(self, name: str):
        return wp.ScopedTimer(
            f"gpu_rod::{name}",
            active=self._enable_timers,
            print=False,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        )

    def _record_timing(self, name: str, elapsed: float):
        if not self._enable_timers:
            return
        self._timing_accum[name] = self._timing_accum.get(name, 0.0) + elapsed

    def _maybe_report_timings(self):
        if not self._enable_timers:
            return
        now = time.perf_counter()
        if now - self._timing_last_report < 5.0:
            return
        if self._timing_count == 0:
            self._timing_last_report = now
            return
        scale = 1000.0 / float(self._timing_count)
        avg_integration = self._timing_accum["integration"] * scale
        avg_assembly = self._timing_accum["constraints_assembly"] * scale
        avg_solve = self._timing_accum["system_solve"] * scale
        avg_update = self._timing_accum["final_position_update"] * scale
        print(
            "GPU Warp avg timings (ms): "
            f"integration={avg_integration:.3f}, "
            f"assembly={avg_assembly:.3f}, "
            f"solve={avg_solve:.3f}, "
            f"final_update={avg_update:.3f}",
            flush=True,
        )
        for key in self._timing_accum:
            self._timing_accum[key] = 0.0
        self._timing_count = 0
        self._timing_last_report = now

    def set_use_cuda_graph(self, enabled: bool):
        if enabled and not self.device.is_cuda:
            enabled = False
        self.use_cuda_graph = enabled
        self._graph = None
        self._graph_params = None

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

    def set_iterative_refinement(self, enabled: bool, iters: int = 2):
        """Enable/disable iterative refinement for banded Cholesky solver.
        
        Args:
            enabled: Whether to use iterative refinement.
            iters: Number of refinement iterations (typically 1-3).
        """
        self.use_iterative_refinement = enabled
        self.iterative_refinement_iters = max(1, iters)
        # Invalidate CUDA graph when solver settings change
        self._graph = None
        self._graph_params = None

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
                kernels._warp_zero_root_velocities,
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
            kernels._warp_apply_root_translation,
            dim=1,
            inputs=[self.positions_wp, self.predicted_positions_wp, self.velocities_wp, dx, dy, dz],
            device=self.device,
        )

    def apply_root_rotation(self, q: np.ndarray):
        if self.num_points == 0:
            return
        q_wp = wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        wp.launch(
            kernels._warp_set_root_orientation,
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

    def _update_inv_inertia_world(self):
        """Compute world-frame inverse inertia tensors from local frame and current orientations."""
        if self.num_points == 0:
            return
        inv_inertia_local = wp.vec3(
            float(self.inv_inertia_local_diag[0]),
            float(self.inv_inertia_local_diag[1]),
            float(self.inv_inertia_local_diag[2]),
        )
        wp.launch(
            kernels._warp_compute_inv_inertia_world,
            dim=self.num_points,
            inputs=[
                self.predicted_orientations_wp,
                self.quat_inv_masses_wp,
                inv_inertia_local,
                self.inv_inertia_wp,
            ],
            device=self.device,
        )

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
        start = time.perf_counter()
        with self._timer("constraints_assembly"):
            wp.launch(
                kernels._warp_zero_float,
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
        self._record_timing("constraints_assembly", time.perf_counter() - start)

    def project_direct(self):
        if self.num_edges == 0:
            self.last_constraint_max = 0.0
            self.last_delta_lambda_max = 0.0
            self.last_correction_max = 0.0
            return

        start = time.perf_counter()
        with self._timer("constraints_assembly"):
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
                kernels._warp_constraint_max,
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

            # Compute world-frame inverse inertia tensors for mass-weighted JMJT assembly
            self._update_inv_inertia_world()
        self._record_timing("constraints_assembly", time.perf_counter() - start)

        n_dofs = self.n_dofs
        if self.direct_solve_backend == DIRECT_SOLVE_WARP_BANDED_CHOLESKY:
            start = time.perf_counter()
            with self._timer("constraints_assembly"):
                wp.launch(
                    kernels._warp_zero_2d,
                    dim=BAND_LDAB * max(1, n_dofs),
                    inputs=[self.ab_wp, int(BAND_LDAB), int(max(1, n_dofs))],
                    device=self.device,
                )
                wp.launch(
                    base._warp_assemble_jmjt_banded,
                    dim=self.num_edges,
                    inputs=[
                        self.jacobian_pos_wp,
                        self.jacobian_rot_wp,
                        self.compliance_wp,
                        self.inv_masses_wp,
                        self.inv_inertia_wp,
                        int(n_dofs),
                        self.ab_wp,
                    ],
                    device=self.device,
                )
                wp.launch(
                    base._warp_build_rhs,
                    dim=n_dofs,
                    inputs=[
                        self.constraint_values_wp,
                        self.compliance_wp,
                        self.lambda_sum_wp,
                        int(n_dofs),
                        self.rhs_wp,
                    ],
                    device=self.device,
                )
            self._record_timing("constraints_assembly", time.perf_counter() - start)
            start = time.perf_counter()
            with self._timer("system_solve"):
                if self.use_iterative_refinement:
                    # Copy matrix and RHS for iterative refinement
                    wp.copy(self.ab_orig_wp, self.ab_wp)
                    wp.copy(self.b_orig_wp, self.rhs_wp)
                    wp.launch(
                        base._warp_spbsv_u11_1rhs_iter_ref,
                        dim=1,
                        inputs=[
                            int(n_dofs),
                            self.ab_wp,
                            self.rhs_wp,
                            self.ab_orig_wp,
                            self.b_orig_wp,
                            self.r_wp,
                            int(self.iterative_refinement_iters),
                        ],
                        device=self.device,
                    )
                else:
                    wp.launch(
                        base._warp_spbsv_u11_1rhs,
                        dim=1,
                        inputs=[int(n_dofs), self.ab_wp, self.rhs_wp],
                        device=self.device,
                    )
            self._record_timing("system_solve", time.perf_counter() - start)
            delta_lambda = self.rhs_wp
        elif n_dofs <= TILE:
            start = time.perf_counter()
            with self._timer("constraints_assembly"):
                wp.launch(
                    kernels._warp_zero_2d,
                    dim=TILE * TILE,
                    inputs=[self.A_wp, int(TILE), int(TILE)],
                    device=self.device,
                )
                wp.launch(
                    base._warp_assemble_jmjt_dense,
                    dim=self.num_edges,
                    inputs=[
                        self.jacobian_pos_wp,
                        self.jacobian_rot_wp,
                        self.compliance_wp,
                        self.inv_masses_wp,
                        self.inv_inertia_wp,
                        int(n_dofs),
                        self.A_wp,
                    ],
                    device=self.device,
                )
                wp.launch(
                    base._warp_build_rhs,
                    dim=TILE,
                    inputs=[
                        self.constraint_values_wp,
                        self.compliance_wp,
                        self.lambda_sum_wp,
                        int(n_dofs),
                        self.rhs_tile_wp,
                    ],
                    device=self.device,
                )
                if n_dofs < TILE:
                    wp.launch(
                        base._warp_pad_diagonal,
                        dim=TILE,
                        inputs=[self.A_wp, int(n_dofs), int(TILE)],
                        device=self.device,
                    )
            self._record_timing("constraints_assembly", time.perf_counter() - start)
            start = time.perf_counter()
            with self._timer("system_solve"):
                wp.launch_tiled(
                    base._warp_cholesky_solve_tile,
                    dim=[1, 1],
                    inputs=[self.A_wp, self.rhs_tile_wp],
                    outputs=[self.delta_lambda_tile_wp],
                    block_dim=BLOCK_DIM,
                    device=self.device,
                )
            self._record_timing("system_solve", time.perf_counter() - start)
            delta_lambda = self.delta_lambda_tile_wp
        else:
            start = time.perf_counter()
            with self._timer("constraints_assembly"):
                wp.launch(
                    base._warp_assemble_jmjt_blocks,
                    dim=self.num_edges,
                    inputs=[
                        self.jacobian_pos_wp,
                        self.jacobian_rot_wp,
                        self.compliance_wp,
                        self.inv_masses_wp,
                        self.inv_inertia_wp,
                        int(self.num_edges),
                        self.diag_blocks_wp,
                        self.offdiag_blocks_wp,
                    ],
                    device=self.device,
                )
                wp.launch(
                    base._warp_build_rhs,
                    dim=n_dofs,
                    inputs=[
                        self.constraint_values_wp,
                        self.compliance_wp,
                        self.lambda_sum_wp,
                        int(n_dofs),
                        self.rhs_wp,
                    ],
                    device=self.device,
                )
            self._record_timing("constraints_assembly", time.perf_counter() - start)
            start = time.perf_counter()
            with self._timer("system_solve"):
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
            self._record_timing("system_solve", time.perf_counter() - start)
            delta_lambda = self.delta_lambda_wp

        start = time.perf_counter()
        with self._timer("final_position_update"):
            wp.launch(
                kernels._warp_apply_direct_corrections,
                dim=1,
                inputs=[
                    self.predicted_positions_wp,
                    self.predicted_orientations_wp,
                    self.inv_masses_wp,
                    self.quat_inv_masses_wp,
                    self.inv_inertia_wp,
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
        self._record_timing("final_position_update", time.perf_counter() - start)

        if self._graph_capture_active:
            self.last_constraint_max = 0.0
            self.last_delta_lambda_max = 0.0
            self.last_correction_max = 0.0
        else:
            self.last_constraint_max = float(self._constraint_max_wp.numpy()[0])
            self.last_delta_lambda_max = float(self._delta_lambda_max_wp.numpy()[0])
            self.last_correction_max = float(self._correction_max_wp.numpy()[0])

    def apply_floor_collisions(self, floor_z: float, restitution: float = 0.0):
        if self.num_points == 0:
            return
        min_z = float(floor_z + self.rod_radius)
        wp.launch(
            kernels._warp_apply_floor_collisions,
            dim=self.num_points,
            inputs=[self.positions_wp, self.predicted_positions_wp, self.velocities_wp, min_z, float(restitution)],
            device=self.device,
        )

    def _step_impl(self, dt: float, linear_damping: float, angular_damping: float):
        start = time.perf_counter()
        with self._timer("integration"):
            self.predict_positions(dt, linear_damping)
            self.predict_rotations(dt, angular_damping)
        self._record_timing("integration", time.perf_counter() - start)

        self.prepare_constraints(dt)
        self.project_direct()

        start = time.perf_counter()
        with self._timer("final_position_update"):
            self.integrate_positions(dt)
            self.integrate_rotations(dt)
        self._record_timing("final_position_update", time.perf_counter() - start)

        if self._enable_timers:
            self._timing_count += 1
            self._maybe_report_timings()

    def _ensure_cuda_graph(self, dt: float, linear_damping: float, angular_damping: float):
        params = (
            float(dt),
            float(linear_damping),
            float(angular_damping),
            bool(self.use_banded),
            str(self.direct_solve_backend),
        )
        if self._graph is not None and self._graph_params == params:
            return

        was_enabled = self._enable_timers
        self._enable_timers = False
        self._graph_capture_active = True
        try:
            with wp.ScopedCapture(device=self.device, force_module_load=True) as capture:
                self._step_impl(dt, linear_damping, angular_damping)
        finally:
            self._graph_capture_active = False
        self._enable_timers = was_enabled
        self._graph = capture.graph
        self._graph_params = params

    def step(self, dt: float, linear_damping: float, angular_damping: float):
        if self.use_cuda_graph and self.device.is_cuda:
            self._ensure_cuda_graph(dt, linear_damping, angular_damping)
            wp.capture_launch(self._graph)
            return

        self._step_impl(dt, linear_damping, angular_damping)


class CosseratXPBDSolver:
    """XPBD-style solver wrapper that advances batched Warp rods."""

    def __init__(
        self,
        rod_batch,
        linear_damping: float = 0.0,
        angular_damping: float = 0.0,
    ):
        self.rod_batch = rod_batch
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping

    def notify_model_changed(self, *args, **kwargs):
        return None

    def update_contacts(self, *args, **kwargs):
        return None

    def step(self, state_in, state_out, control, contacts, dt: float):
        state = state_out or state_in
        for rod in state.rods:
            rod.step(dt, self.linear_damping, self.angular_damping)


__all__ = ["CosseratXPBDSolver", "WarpResidentRodState"]
