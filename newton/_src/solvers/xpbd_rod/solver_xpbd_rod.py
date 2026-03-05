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

"""XPBD solver for Cosserat elastic rods with direct block-tridiagonal solve."""

from __future__ import annotations

from typing import override

import numpy as np
import warp as wp

from ...sim import Contacts, Control, Model, ModelBuilder, State
from ..solver import SolverBase
from .constants import (
    BAND_LDAB,
    BLOCK_DIM,
    DIRECT_SOLVE_BANDED_CHOLESKY,
    DIRECT_SOLVE_BLOCK_JACOBI,
    DIRECT_SOLVE_BLOCK_THOMAS,
    DIRECT_SOLVE_SPLIT_THOMAS,
    TILE,
)
from .kernels_assembly import (
    _warp_assemble_darboux_blocks,
    _warp_assemble_jmjt_banded,
    _warp_assemble_jmjt_blocks,
    _warp_assemble_jmjt_dense,
    _warp_assemble_stretch_blocks,
    _warp_pad_diagonal,
)
from .kernels_collision import (
    _warp_apply_accumulated_corrections,
    _warp_apply_floor_collisions,
    _warp_compute_corrections_parallel,
    _warp_compute_inv_inertia_world,
    _warp_merge_delta_lambda,
    _warp_zero_2d,
    _warp_zero_float,
    _warp_zero_vec3,
)
from .kernels_constraints import (
    _warp_build_rhs,
    _warp_build_rhs_darboux,
    _warp_build_rhs_stretch,
    _warp_compute_jacobians_direct,
    _warp_prepare_compliance,
    _warp_update_constraints_direct,
)
from .kernels_integration import (
    _warp_integrate_positions,
    _warp_integrate_rotations,
    _warp_predict_positions,
    _warp_predict_rotations,
)
from .kernels_solvers import (
    _warp_block_thomas_solve,
    _warp_block_thomas_solve_3x3,
    _warp_cholesky_solve_tile,
    _warp_solve_blocks_jacobi,
    _warp_spbsv_u11_1rhs,
)


class _RodWorkspace:
    """GPU workspace arrays for a single rod within the solver."""

    def __init__(self, num_points: int, num_edges: int, device: wp.Device):
        n_dofs = num_edges * 6
        alloc_dofs = max(1, n_dofs)
        alloc_edges = max(1, num_edges)
        alloc_points = max(1, num_points)

        self.num_points = num_points
        self.num_edges = num_edges
        self.n_dofs = n_dofs
        self.device = device

        # Per-particle state arrays
        self.positions_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)
        self.predicted_positions_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)
        self.velocities_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)
        self.forces_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)

        self.orientations_wp = wp.zeros(alloc_points, dtype=wp.quat, device=device)
        self.predicted_orientations_wp = wp.zeros(alloc_points, dtype=wp.quat, device=device)
        self.prev_orientations_wp = wp.zeros(alloc_points, dtype=wp.quat, device=device)
        self.angular_velocities_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)
        self.torques_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)

        self.inv_masses_wp = wp.zeros(alloc_points, dtype=wp.float32, device=device)
        self.quat_inv_masses_wp = wp.zeros(alloc_points, dtype=wp.float32, device=device)

        # Per-edge arrays
        self.rest_lengths_wp = wp.zeros(alloc_edges, dtype=wp.float32, device=device)
        self.rest_darboux_wp = wp.zeros(alloc_edges, dtype=wp.vec3, device=device)
        self.bend_stiffness_wp = wp.zeros(alloc_edges, dtype=wp.vec3, device=device)

        # Constraint workspace
        self.constraint_values_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=device)
        self.compliance_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=device)
        self.lambda_sum_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=device)
        self.jacobian_pos_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=device)
        self.jacobian_rot_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=device)

        # Solver workspace
        self.rhs_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=device)
        self.delta_lambda_wp = wp.zeros(alloc_dofs, dtype=wp.float32, device=device)
        self.diag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=device)
        self.offdiag_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=device)
        self.c_blocks_wp = wp.zeros(alloc_edges * 36, dtype=wp.float32, device=device)
        self.d_prime_wp = wp.zeros(alloc_edges * 6, dtype=wp.float32, device=device)

        # Dense/tiled solver workspace
        self.A_wp = wp.zeros((TILE, TILE), dtype=wp.float32, device=device)
        self.rhs_tile_wp = wp.zeros(TILE, dtype=wp.float32, device=device)
        self.delta_lambda_tile_wp = wp.zeros(TILE, dtype=wp.float32, device=device)

        # Banded solver workspace
        self.ab_wp = wp.zeros((BAND_LDAB, alloc_dofs), dtype=wp.float32, device=device)

        # Inverse inertia
        self.inv_inertia_wp = wp.zeros(alloc_points * 9, dtype=wp.float32, device=device)
        self.inv_inertia_local_diag = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Parallel correction workspace
        self.pos_corrections_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)
        self.rot_corrections_wp = wp.zeros(alloc_points, dtype=wp.vec3, device=device)

        # Diagnostics
        self._constraint_max_wp = wp.zeros(1, dtype=wp.float32, device=device)
        self._delta_lambda_max_wp = wp.zeros(1, dtype=wp.float32, device=device)
        self._correction_max_wp = wp.zeros(1, dtype=wp.float32, device=device)

        # Split Thomas solver arrays (lazily allocated)
        self._split_stretch_diag_wp = None

        # Material properties
        self.young_modulus = 1.0e6
        self.torsion_modulus = 1.0e6
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)


class SolverXPBDRod(SolverBase):
    """XPBD solver for Cosserat elastic rods with direct block-tridiagonal solve.

    This solver implements Extended Position-Based Dynamics (XPBD) for
    Cosserat elastic rods. It supports stretch and bend/twist constraints
    solved via block-tridiagonal direct solvers on GPU.

    Multiple solver backends are available:

    - ``"block_thomas"``: Block Thomas algorithm for 6x6 block-tridiagonal systems (default).
    - ``"split_thomas"``: Split into two 3x3 block-tridiagonal systems (stretch + darboux).
    - ``"block_jacobi"``: Block-diagonal Jacobi (ignores coupling between edges).
    - ``"banded_cholesky"``: Dense banded Cholesky for banded JMJT matrix.

    Args:
        model: The Newton model containing rod data.
        linear_damping: Linear velocity damping factor.
        angular_damping: Angular velocity damping factor.
        solver_backend: Solver backend to use.
        floor_z: Z coordinate of the floor plane, or ``None`` to disable.
    """

    def __init__(
        self,
        model: Model,
        linear_damping: float = 0.0,
        angular_damping: float = 0.0,
        solver_backend: str = DIRECT_SOLVE_BLOCK_THOMAS,
        floor_z: float | None = 0.0,
    ):
        super().__init__(model)
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.solver_backend = solver_backend
        self.floor_z = floor_z

        device = model.device

        # Build rod workspaces from model data stored during build
        self._rods: list[_RodWorkspace] = []

        if not hasattr(model, "xpbd_rod"):
            return

        rod_data = model.xpbd_rod
        rod_num_points = rod_data["rod_num_points"]
        rod_particle_starts = rod_data["rod_particle_start"]
        rod_young_moduli = rod_data["rod_young_modulus"]
        rod_torsion_moduli = rod_data["rod_torsion_modulus"]

        all_orientations = rod_data["orientations"]
        all_quat_inv_masses = rod_data["quat_inv_masses"]
        all_rest_lengths = rod_data["rest_lengths"]
        all_rest_darboux = rod_data["rest_darboux"]
        all_bend_stiffness = rod_data["bend_stiffness"]

        orient_cursor = 0
        edge_cursor = 0

        for rod_idx in range(len(rod_num_points)):
            np_ = rod_num_points[rod_idx]
            ne = np_ - 1
            ps = rod_particle_starts[rod_idx]

            ws = _RodWorkspace(np_, ne, device)
            ws.young_modulus = rod_young_moduli[rod_idx]
            ws.torsion_modulus = rod_torsion_moduli[rod_idx]

            # Copy particle positions/masses from model
            particle_q = model.particle_q.numpy()
            pos_slice = particle_q[ps : ps + np_]
            ws.positions_wp.assign(wp.array(pos_slice, dtype=wp.vec3, device=device))
            ws.predicted_positions_wp.assign(wp.array(pos_slice, dtype=wp.vec3, device=device))

            inv_mass_np = model.particle_inv_mass.numpy()[ps : ps + np_]
            ws.inv_masses_wp.assign(wp.array(inv_mass_np, dtype=wp.float32, device=device))

            # Copy rod-specific data
            orient_slice = np.array(all_orientations[orient_cursor : orient_cursor + np_], dtype=np.float32)
            ws.orientations_wp.assign(wp.array(orient_slice, dtype=wp.quat, device=device))
            ws.predicted_orientations_wp.assign(wp.array(orient_slice, dtype=wp.quat, device=device))
            ws.prev_orientations_wp.assign(wp.array(orient_slice, dtype=wp.quat, device=device))

            qim_slice = np.array(all_quat_inv_masses[orient_cursor : orient_cursor + np_], dtype=np.float32)
            ws.quat_inv_masses_wp.assign(wp.array(qim_slice, dtype=wp.float32, device=device))

            rl_slice = np.array(all_rest_lengths[edge_cursor : edge_cursor + ne], dtype=np.float32)
            ws.rest_lengths_wp.assign(wp.array(rl_slice, dtype=wp.float32, device=device))

            rd_slice = np.array(all_rest_darboux[edge_cursor : edge_cursor + ne], dtype=np.float32)
            ws.rest_darboux_wp.assign(wp.array(rd_slice, dtype=wp.vec3, device=device))

            bs_slice = np.array(all_bend_stiffness[edge_cursor : edge_cursor + ne], dtype=np.float32)
            ws.bend_stiffness_wp.assign(wp.array(bs_slice, dtype=wp.vec3, device=device))

            # Gravity from model
            if model.gravity is not None:
                g = model.gravity.numpy()
                ws.gravity = np.array([g[0][0], g[0][1], g[0][2]], dtype=np.float32)

            orient_cursor += np_
            edge_cursor += ne

            self._rods.append(ws)

        # Store particle start indices for syncing back
        self._rod_particle_starts = list(rod_particle_starts) if rod_num_points else []

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register rod-specific data storage on the builder.

        Must be called before adding rods and before
        :meth:`~newton.ModelBuilder.finalize`.
        """
        builder._xpbd_rod_data = {
            "rod_num_points": [],
            "rod_particle_start": [],
            "rod_young_modulus": [],
            "rod_torsion_modulus": [],
            "orientations": [],
            "quat_inv_masses": [],
            "rest_lengths": [],
            "rest_darboux": [],
            "bend_stiffness": [],
        }

        # Wrap finalize to transfer rod data to the model
        original_finalize = builder.finalize

        def _finalize_with_rod_data(*args, **kwargs):
            model = original_finalize(*args, **kwargs)
            model.xpbd_rod = builder._xpbd_rod_data
            return model

        builder.finalize = _finalize_with_rod_data

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ):
        device = self.model.device

        for rod_idx, ws in enumerate(self._rods):
            if ws.num_edges == 0:
                continue

            self._step_rod(ws, dt, device)

            # Sync positions back to state_out.particle_q
            ps = self._rod_particle_starts[rod_idx]
            # Copy rod positions into state_out particle_q at the right offset
            wp.copy(
                dest=state_out.particle_q,
                src=ws.positions_wp,
                dest_offset=ps,
                src_offset=0,
                count=ws.num_points,
            )

    def _step_rod(self, ws: _RodWorkspace, dt: float, device: wp.Device):
        """Run one XPBD step for a single rod."""
        # 1. Predict positions & rotations
        gravity = wp.vec3(float(ws.gravity[0]), float(ws.gravity[1]), float(ws.gravity[2]))
        wp.launch(
            _warp_predict_positions,
            dim=ws.num_points,
            inputs=[
                ws.positions_wp,
                ws.velocities_wp,
                ws.forces_wp,
                ws.inv_masses_wp,
                gravity,
                float(dt),
                float(self.linear_damping),
                ws.predicted_positions_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_predict_rotations,
            dim=ws.num_points,
            inputs=[
                ws.orientations_wp,
                ws.angular_velocities_wp,
                ws.torques_wp,
                ws.quat_inv_masses_wp,
                float(dt),
                float(self.angular_damping),
                ws.predicted_orientations_wp,
            ],
            device=device,
        )

        # 2. Prepare constraints
        wp.launch(_warp_zero_float, dim=ws.n_dofs, inputs=[ws.lambda_sum_wp], device=device)
        wp.launch(
            _warp_prepare_compliance,
            dim=ws.num_edges,
            inputs=[
                ws.rest_lengths_wp,
                ws.bend_stiffness_wp,
                float(ws.young_modulus),
                float(ws.torsion_modulus),
                float(dt),
                ws.compliance_wp,
            ],
            device=device,
        )

        # 3. Project constraints
        self._project_direct(ws, device)

        # 4. Floor collision (optional)
        if self.floor_z is not None:
            min_z = float(self.floor_z)
            wp.launch(
                _warp_apply_floor_collisions,
                dim=ws.num_points,
                inputs=[ws.positions_wp, ws.predicted_positions_wp, ws.velocities_wp, min_z, 0.0],
                device=device,
            )

        # 5. Integrate
        wp.launch(
            _warp_integrate_positions,
            dim=ws.num_points,
            inputs=[ws.positions_wp, ws.predicted_positions_wp, ws.velocities_wp, ws.inv_masses_wp, float(dt)],
            device=device,
        )
        wp.launch(
            _warp_integrate_rotations,
            dim=ws.num_points,
            inputs=[
                ws.orientations_wp,
                ws.predicted_orientations_wp,
                ws.prev_orientations_wp,
                ws.angular_velocities_wp,
                ws.quat_inv_masses_wp,
                float(dt),
            ],
            device=device,
        )

    def _project_direct(self, ws: _RodWorkspace, device: wp.Device):
        """Project constraints using the configured direct solver backend."""
        if ws.num_edges == 0:
            return

        # Update constraints
        wp.launch(
            _warp_update_constraints_direct,
            dim=ws.num_edges,
            inputs=[
                ws.predicted_positions_wp,
                ws.predicted_orientations_wp,
                ws.rest_lengths_wp,
                ws.rest_darboux_wp,
                ws.constraint_values_wp,
            ],
            device=device,
        )

        # Compute Jacobians
        wp.launch(
            _warp_compute_jacobians_direct,
            dim=ws.num_edges,
            inputs=[
                ws.predicted_orientations_wp,
                ws.rest_lengths_wp,
                ws.jacobian_pos_wp,
                ws.jacobian_rot_wp,
            ],
            device=device,
        )

        # Update inverse inertia
        inv_inertia_local = wp.vec3(
            float(ws.inv_inertia_local_diag[0]),
            float(ws.inv_inertia_local_diag[1]),
            float(ws.inv_inertia_local_diag[2]),
        )
        wp.launch(
            _warp_compute_inv_inertia_world,
            dim=ws.num_points,
            inputs=[
                ws.predicted_orientations_wp,
                ws.quat_inv_masses_wp,
                inv_inertia_local,
                ws.inv_inertia_wp,
            ],
            device=device,
        )

        n_dofs = ws.n_dofs
        delta_lambda = self._solve_system(ws, n_dofs, device)

        # Apply corrections (parallel two-phase)
        wp.launch(_warp_zero_vec3, dim=ws.num_points, inputs=[ws.pos_corrections_wp], device=device)
        wp.launch(_warp_zero_vec3, dim=ws.num_points, inputs=[ws.rot_corrections_wp], device=device)
        wp.launch(_warp_zero_float, dim=1, inputs=[ws._delta_lambda_max_wp], device=device)
        wp.launch(_warp_zero_float, dim=1, inputs=[ws._correction_max_wp], device=device)

        wp.launch(
            _warp_compute_corrections_parallel,
            dim=ws.num_edges,
            inputs=[
                ws.predicted_positions_wp,
                ws.inv_masses_wp,
                ws.quat_inv_masses_wp,
                ws.inv_inertia_wp,
                ws.jacobian_pos_wp,
                ws.jacobian_rot_wp,
                delta_lambda,
                ws.lambda_sum_wp,
                int(ws.num_edges),
                ws.pos_corrections_wp,
                ws.rot_corrections_wp,
                ws._delta_lambda_max_wp,
                ws._correction_max_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_apply_accumulated_corrections,
            dim=ws.num_points,
            inputs=[
                ws.predicted_positions_wp,
                ws.predicted_orientations_wp,
                ws.pos_corrections_wp,
                ws.rot_corrections_wp,
                int(ws.num_points),
            ],
            device=device,
        )

    def _solve_system(self, ws: _RodWorkspace, n_dofs: int, device: wp.Device) -> wp.array:
        """Assemble and solve the linear system based on the chosen backend."""
        if self.solver_backend == DIRECT_SOLVE_SPLIT_THOMAS:
            return self._solve_split_thomas(ws, device)

        if self.solver_backend == DIRECT_SOLVE_BLOCK_JACOBI:
            wp.launch(
                _warp_assemble_jmjt_blocks,
                dim=ws.num_edges,
                inputs=[
                    ws.jacobian_pos_wp,
                    ws.jacobian_rot_wp,
                    ws.compliance_wp,
                    ws.inv_masses_wp,
                    ws.inv_inertia_wp,
                    int(ws.num_edges),
                    ws.diag_blocks_wp,
                    ws.offdiag_blocks_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_build_rhs,
                dim=n_dofs,
                inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n_dofs), ws.rhs_wp],
                device=device,
            )
            wp.launch(
                _warp_solve_blocks_jacobi,
                dim=ws.num_edges,
                inputs=[ws.diag_blocks_wp, ws.rhs_wp, ws.delta_lambda_wp, int(ws.num_edges)],
                device=device,
            )
            return ws.delta_lambda_wp

        if self.solver_backend == DIRECT_SOLVE_BANDED_CHOLESKY:
            wp.launch(
                _warp_zero_2d,
                dim=BAND_LDAB * max(1, n_dofs),
                inputs=[ws.ab_wp, int(BAND_LDAB), int(max(1, n_dofs))],
                device=device,
            )
            wp.launch(
                _warp_assemble_jmjt_banded,
                dim=ws.num_edges,
                inputs=[
                    ws.jacobian_pos_wp,
                    ws.jacobian_rot_wp,
                    ws.compliance_wp,
                    ws.inv_masses_wp,
                    ws.inv_inertia_wp,
                    int(n_dofs),
                    ws.ab_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_build_rhs,
                dim=n_dofs,
                inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n_dofs), ws.rhs_wp],
                device=device,
            )
            wp.launch(
                _warp_spbsv_u11_1rhs,
                dim=1,
                inputs=[int(n_dofs), ws.ab_wp, ws.rhs_wp],
                device=device,
            )
            return ws.rhs_wp

        # Default: Block Thomas (or tiled Cholesky for small systems)
        if n_dofs <= TILE:
            wp.launch(
                _warp_zero_2d,
                dim=TILE * TILE,
                inputs=[ws.A_wp, int(TILE), int(TILE)],
                device=device,
            )
            wp.launch(
                _warp_assemble_jmjt_dense,
                dim=ws.num_edges,
                inputs=[
                    ws.jacobian_pos_wp,
                    ws.jacobian_rot_wp,
                    ws.compliance_wp,
                    ws.inv_masses_wp,
                    ws.inv_inertia_wp,
                    int(n_dofs),
                    ws.A_wp,
                ],
                device=device,
            )
            wp.launch(
                _warp_build_rhs,
                dim=TILE,
                inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n_dofs), ws.rhs_tile_wp],
                device=device,
            )
            if n_dofs < TILE:
                wp.launch(
                    _warp_pad_diagonal,
                    dim=TILE,
                    inputs=[ws.A_wp, int(n_dofs), int(TILE)],
                    device=device,
                )
            wp.launch_tiled(
                _warp_cholesky_solve_tile,
                dim=[1, 1],
                inputs=[ws.A_wp, ws.rhs_tile_wp],
                outputs=[ws.delta_lambda_tile_wp],
                block_dim=BLOCK_DIM,
                device=device,
            )
            return ws.delta_lambda_tile_wp

        # Block Thomas for larger systems
        wp.launch(
            _warp_assemble_jmjt_blocks,
            dim=ws.num_edges,
            inputs=[
                ws.jacobian_pos_wp,
                ws.jacobian_rot_wp,
                ws.compliance_wp,
                ws.inv_masses_wp,
                ws.inv_inertia_wp,
                int(ws.num_edges),
                ws.diag_blocks_wp,
                ws.offdiag_blocks_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_build_rhs,
            dim=n_dofs,
            inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n_dofs), ws.rhs_wp],
            device=device,
        )
        wp.launch(
            _warp_block_thomas_solve,
            dim=1,
            inputs=[
                ws.diag_blocks_wp,
                ws.offdiag_blocks_wp,
                ws.rhs_wp,
                int(ws.num_edges),
                ws.c_blocks_wp,
                ws.d_prime_wp,
                ws.delta_lambda_wp,
            ],
            device=device,
        )
        return ws.delta_lambda_wp

    def _solve_split_thomas(self, ws: _RodWorkspace, device: wp.Device) -> wp.array:
        """Solve using split 3x3 block Thomas for stretch and darboux."""
        n = ws.num_edges

        # Lazily allocate split arrays
        if ws._split_stretch_diag_wp is None:
            ws._split_stretch_diag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_stretch_offdiag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_stretch_rhs_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_stretch_c_blocks_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_stretch_d_prime_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_stretch_delta_lambda_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_darboux_diag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_darboux_offdiag_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_darboux_rhs_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_darboux_c_blocks_wp = wp.zeros(n * 9, dtype=wp.float32, device=device)
            ws._split_darboux_d_prime_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)
            ws._split_darboux_delta_lambda_wp = wp.zeros(n * 3, dtype=wp.float32, device=device)

        # Assemble
        wp.launch(
            _warp_assemble_stretch_blocks,
            dim=n,
            inputs=[
                ws.jacobian_pos_wp,
                ws.jacobian_rot_wp,
                ws.compliance_wp,
                ws.inv_masses_wp,
                ws.inv_inertia_wp,
                int(n),
                ws._split_stretch_diag_wp,
                ws._split_stretch_offdiag_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_assemble_darboux_blocks,
            dim=n,
            inputs=[
                ws.jacobian_rot_wp,
                ws.compliance_wp,
                ws.inv_inertia_wp,
                int(n),
                ws._split_darboux_diag_wp,
                ws._split_darboux_offdiag_wp,
            ],
            device=device,
        )

        # Build RHS
        wp.launch(
            _warp_build_rhs_stretch,
            dim=n,
            inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n), ws._split_stretch_rhs_wp],
            device=device,
        )
        wp.launch(
            _warp_build_rhs_darboux,
            dim=n,
            inputs=[ws.constraint_values_wp, ws.compliance_wp, ws.lambda_sum_wp, int(n), ws._split_darboux_rhs_wp],
            device=device,
        )

        # Solve
        wp.launch(
            _warp_block_thomas_solve_3x3,
            dim=1,
            inputs=[
                ws._split_stretch_diag_wp,
                ws._split_stretch_offdiag_wp,
                ws._split_stretch_rhs_wp,
                int(n),
                ws._split_stretch_c_blocks_wp,
                ws._split_stretch_d_prime_wp,
                ws._split_stretch_delta_lambda_wp,
            ],
            device=device,
        )
        wp.launch(
            _warp_block_thomas_solve_3x3,
            dim=1,
            inputs=[
                ws._split_darboux_diag_wp,
                ws._split_darboux_offdiag_wp,
                ws._split_darboux_rhs_wp,
                int(n),
                ws._split_darboux_c_blocks_wp,
                ws._split_darboux_d_prime_wp,
                ws._split_darboux_delta_lambda_wp,
            ],
            device=device,
        )

        # Merge
        wp.launch(
            _warp_merge_delta_lambda,
            dim=n,
            inputs=[ws._split_stretch_delta_lambda_wp, ws._split_darboux_delta_lambda_wp, ws.delta_lambda_wp, int(n)],
            device=device,
        )
        return ws.delta_lambda_wp

    @override
    def update_contacts(self, contacts: Contacts) -> None:
        pass
