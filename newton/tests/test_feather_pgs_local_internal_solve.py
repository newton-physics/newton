# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
    classify_local_solve_worlds,
    compact_local_pair_candidates,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_local_owned_kernel


class TestFeatherPGSLocalInternalSolve(unittest.TestCase):
    @staticmethod
    def _empty_mf_inputs(world_count: int, device: str | wp.Device) -> list:
        return [
            wp.zeros(world_count, dtype=wp.int32, device=device),
            wp.zeros((world_count, 4), dtype=wp.int32, device=device),
            wp.zeros((world_count, 1), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1, 6), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1), dtype=wp.float32, device=device),
            0,
            wp.zeros((world_count, 1), dtype=wp.float32, device=device),
            wp.zeros((world_count, 1), dtype=wp.float32, device=device),
        ]

    def test_local_owner_classifier_rejects_unsupported_worlds(self):
        device = "cpu"
        constraint_count = wp.array([5, 8, 21, 8, 8, 8, 0, 41], dtype=wp.int32, device=device)
        phase_bounds = wp.array(
            [[0, 5], [0, 2], [0, 2], [0, 2], [0, 2], [0, 8], [0, 0], [0, 2]],
            dtype=wp.int32,
            device=device,
        )
        mf_count = wp.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=wp.int32, device=device)
        mf_body_a = wp.full((8, 1), -1, dtype=wp.int32, device=device)
        mf_body_b = wp.array([[-1], [-1], [-1], [0], [1], [-1], [-1], [-1]], dtype=wp.int32, device=device)
        mf_row_type = wp.full((8, 1), PGS_CONSTRAINT_TYPE_CONTACT, dtype=wp.int32, device=device)
        body_to_articulation = wp.array([12, 99], dtype=wp.int32, device=device)
        articulation_dof_count_host = np.full(16, 20, dtype=np.int32)
        articulation_dof_count_host[5] = 7
        articulation_dof_count = wp.array(articulation_dof_count_host, device=device)
        primary = wp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=wp.int32, device=device)
        pair = wp.array([-1, 10, 11, 12, 13, -1, 14, 15], dtype=wp.int32, device=device)
        residual_pair = wp.array([-1, 10, 11, 12, 13, -1, 14, 15], dtype=wp.int32, device=device)
        owner = wp.empty(8, dtype=wp.int32, device=device)
        general_count = wp.zeros(1, dtype=wp.int32, device=device)
        general_worlds = wp.empty(8, dtype=wp.int32, device=device)

        wp.launch(
            classify_local_solve_worlds,
            dim=8,
            inputs=[
                constraint_count,
                phase_bounds,
                mf_count,
                mf_body_a,
                mf_body_b,
                mf_row_type,
                body_to_articulation,
                articulation_dof_count,
                primary,
                pair,
                residual_pair,
                20,
                40,
                12,
            ],
            outputs=[owner, general_count, general_worlds],
            device=device,
        )

        np.testing.assert_array_equal(
            owner.numpy(),
            [
                PGS_LOCAL_SOLVE_OWNER_SINGLE,
                PGS_LOCAL_SOLVE_OWNER_PAIR,
                PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
                PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
            ],
        )
        self.assertEqual(int(general_count.numpy()[0]), 3)
        np.testing.assert_array_equal(general_worlds.numpy()[:3], [4, 5, 7])

        active_count = wp.zeros(1, dtype=wp.int32, device=device)
        active_primary = wp.empty(6, dtype=wp.int32, device=device)
        active_secondary = wp.empty(6, dtype=wp.int32, device=device)
        wp.launch(
            compact_local_pair_candidates,
            dim=8,
            inputs=[
                primary,
                pair,
                wp.array(np.arange(16), dtype=wp.int32, device=device),
                owner,
                PGS_LOCAL_SOLVE_OWNER_PAIR,
            ],
            outputs=[active_count, active_primary, active_secondary],
            device=device,
        )
        self.assertEqual(int(active_count.numpy()[0]), 1)
        np.testing.assert_array_equal(active_primary.numpy()[:1], [1])
        np.testing.assert_array_equal(active_secondary.numpy()[:1], [10])

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_kernel_matches_sequential_pgs_and_respects_world_ownership(self):
        device = wp.get_cuda_device()
        dof_count = 3
        max_constraints = 4
        local_max_constraints = 2
        iterations = 4
        omega = 1.0

        lower = np.array([[2.0, 0.0, 0.0], [0.25, 1.5, 0.0], [-0.1, 0.2, 1.25]], dtype=np.float32)
        jacobian = np.zeros((2, max_constraints, dof_count), dtype=np.float32)
        jacobian[0, 0] = [1.0, 0.5, -0.25]
        jacobian[0, 1] = [-0.4, 0.3, 0.8]
        jacobian[1, 0] = [0.2, -0.1, 0.7]
        velocity = np.array([0.3, -0.5, 0.1, 9.0, 8.0, 7.0], dtype=np.float32)
        rhs = np.zeros((2, max_constraints), dtype=np.float32)
        rhs[0, :2] = [-0.2, 0.15]
        cfm = np.zeros((2, max_constraints), dtype=np.float32)
        cfm[0, :2] = [0.01, 0.02]
        row_type = np.zeros((2, max_constraints), dtype=np.int32)
        row_type[0, :2] = [PGS_CONSTRAINT_TYPE_MIMIC, PGS_CONSTRAINT_TYPE_JOINT_LIMIT]
        row_type[1, 0] = PGS_CONSTRAINT_TYPE_MIMIC

        impulses = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
        diagonal_out = wp.array(cfm, dtype=wp.float32, device=device)
        velocity_out = wp.array(velocity, dtype=wp.float32, device=device)
        candidate_articulations = wp.array([0, 1], dtype=wp.int32, device=device)
        lower_group = wp.array(np.stack([lower, np.eye(dof_count, dtype=np.float32)]), dtype=wp.float32, device=device)
        jacobian_group = wp.array(jacobian, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(max_constraints, local_max_constraints, dof_count, device.arch)
        wp.launch_tiled(
            kernel,
            dim=[2],
            inputs=[
                candidate_articulations,
                candidate_articulations,
                1,
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 3], dtype=wp.int32, device=device),
                wp.array([1, 0], dtype=wp.int32, device=device),
                wp.array([2, 1], dtype=wp.int32, device=device),
                lower_group,
                jacobian_group,
                lower_group,
                jacobian_group,
                wp.array(rhs, dtype=wp.float32, device=device),
                wp.array(row_type, dtype=wp.int32, device=device),
                wp.full((2, max_constraints), -1, dtype=wp.int32, device=device),
                wp.zeros((2, max_constraints), dtype=wp.float32, device=device),
                *self._empty_mf_inputs(2, device),
                iterations,
                omega,
                0,
                0,
            ],
            outputs=[diagonal_out, impulses, velocity_out],
            block_dim=32,
            device=device,
        )
        wp.synchronize_device(device)

        mass = lower @ lower.T
        response = np.stack([np.linalg.solve(mass, row) for row in jacobian[0, :2]])
        diagonal = np.sum(jacobian[0, :2] * response, axis=1) + cfm[0, :2]
        expected_velocity = velocity[:dof_count].copy()
        expected_impulses = np.zeros(2, dtype=np.float32)
        for _ in range(iterations):
            for row in range(2):
                delta = -(jacobian[0, row] @ expected_velocity + rhs[0, row]) / diagonal[row]
                new_impulse = expected_impulses[row] + omega * delta
                if row_type[0, row] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
                    new_impulse = max(new_impulse, 0.0)
                expected_velocity += response[row] * (new_impulse - expected_impulses[row])
                expected_impulses[row] = new_impulse

        np.testing.assert_allclose(velocity_out.numpy()[:dof_count], expected_velocity, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0, :2], expected_impulses, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(diagonal_out.numpy()[0, :2], diagonal, rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(diagonal_out.numpy()[1], cfm[1])
        np.testing.assert_array_equal(velocity_out.numpy()[dof_count:], velocity[dof_count:])
        np.testing.assert_array_equal(impulses.numpy()[1], np.zeros(max_constraints, dtype=np.float32))

        def launch_variant(
            warps_per_block: int,
            *,
            lanes_per_world: int = 32,
            contact_capable: bool = True,
            persistent_queue: bool = False,
            active_count: int = 2,
            dense_response_matrix: bool = False,
        ):
            variant_diagonal = wp.array(cfm, dtype=wp.float32, device=device)
            variant_impulses = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
            variant_velocity = wp.array(velocity, dtype=wp.float32, device=device)
            variant_kernel = _get_pgs_solve_local_owned_kernel(
                max_constraints,
                local_max_constraints,
                dof_count,
                device.arch,
                persistent_queue=persistent_queue,
                warps_per_block=warps_per_block,
                lanes_per_world=lanes_per_world,
                contact_capable=contact_capable,
                dense_response_matrix=dense_response_matrix,
            )
            inputs = [candidate_articulations, candidate_articulations]
            if persistent_queue:
                inputs.extend([wp.array([active_count], dtype=wp.int32, device=device), 2])
            inputs.extend(
                [
                    PGS_LOCAL_SOLVE_OWNER_SINGLE,
                    wp.array([0, 1], dtype=wp.int32, device=device),
                    wp.array([0, 1], dtype=wp.int32, device=device),
                    wp.array([0, 3], dtype=wp.int32, device=device),
                    wp.array([PGS_LOCAL_SOLVE_OWNER_SINGLE] * 2, dtype=wp.int32, device=device),
                    wp.array([2, 1], dtype=wp.int32, device=device),
                    lower_group,
                    jacobian_group,
                    lower_group,
                    jacobian_group,
                    wp.array(rhs, dtype=wp.float32, device=device),
                    wp.array(row_type, dtype=wp.int32, device=device),
                    wp.full((2, max_constraints), -1, dtype=wp.int32, device=device),
                    wp.zeros((2, max_constraints), dtype=wp.float32, device=device),
                    *self._empty_mf_inputs(2, device),
                    iterations,
                    omega,
                    0,
                    0,
                ]
            )
            worlds_per_block = warps_per_block * 32 // lanes_per_world
            block_count = 2 // worlds_per_block
            if persistent_queue:
                block_count = max(block_count, 1)
            wp.launch_tiled(
                variant_kernel,
                dim=[block_count],
                inputs=inputs,
                outputs=[variant_diagonal, variant_impulses, variant_velocity],
                block_dim=32 * warps_per_block,
                device=device,
            )
            wp.synchronize_device(device)
            return variant_diagonal.numpy(), variant_impulses.numpy(), variant_velocity.numpy()

        one_warp_outputs = launch_variant(1)
        exact_variants = [launch_variant(2)]
        numerical_variants = [
            launch_variant(1, contact_capable=False),
            launch_variant(2, contact_capable=False),
        ]
        matrix_variants = [
            launch_variant(1, contact_capable=False, dense_response_matrix=True),
            launch_variant(
                1,
                lanes_per_world=16,
                contact_capable=False,
                dense_response_matrix=True,
            ),
            launch_variant(
                1,
                lanes_per_world=8,
                contact_capable=False,
                persistent_queue=True,
                dense_response_matrix=True,
            ),
        ]
        for variant_outputs in exact_variants:
            for one_warp, variant in zip(one_warp_outputs, variant_outputs, strict=True):
                np.testing.assert_array_equal(variant, one_warp)
        for variant_outputs in numerical_variants + matrix_variants:
            for one_warp, variant in zip(one_warp_outputs, variant_outputs, strict=True):
                np.testing.assert_allclose(variant, one_warp, rtol=0.0, atol=2.0e-6)

        one_warp_queue = launch_variant(1, persistent_queue=True, active_count=1)
        two_warp_queue = launch_variant(2, persistent_queue=True, active_count=1)
        for one_warp, two_warp in zip(one_warp_queue, two_warp_queue, strict=True):
            np.testing.assert_array_equal(two_warp, one_warp)

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_kernel_waits_for_delayed_friction_activation(self):
        device = wp.get_cuda_device()
        dof_count = 3
        max_constraints = 3
        candidate = wp.array([0], dtype=wp.int32, device=device)
        lower = wp.array(np.eye(dof_count, dtype=np.float32)[None], dtype=wp.float32, device=device)
        jacobian = wp.array(np.eye(dof_count, dtype=np.float32)[None], dtype=wp.float32, device=device)
        impulses = wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32, device=device)
        diagonal = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        velocity = wp.array([0.0, 1.0, 0.0], dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(max_constraints, max_constraints, dof_count, device.arch)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                candidate,
                candidate,
                1,
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                wp.array([max_constraints], dtype=wp.int32, device=device),
                lower,
                jacobian,
                lower,
                jacobian,
                wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
                wp.array(
                    [[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]],
                    dtype=wp.int32,
                    device=device,
                ),
                wp.array([[-1, 0, 0]], dtype=wp.int32, device=device),
                wp.array([[0.0, 0.5, 0.5]], dtype=wp.float32, device=device),
                *self._empty_mf_inputs(1, device),
                3,
                1.0,
                2,
                0,
            ],
            outputs=[diagonal, impulses, velocity],
            block_dim=32,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(diagonal.numpy()[0], np.ones(max_constraints), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0], [1.0, -0.5, 0.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(velocity.numpy(), [0.0, 0.5, 0.0], rtol=0.0, atol=1.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_kernel_uses_patch_load_and_coupled_tangent_response(self):
        """Match the pooled patch load and two-row tangent solve used by the general owner."""
        device = wp.get_cuda_device()
        dof_count = 2
        max_constraints = 4
        candidate = wp.array([0], dtype=wp.int32, device=device)
        lower = wp.array(np.eye(dof_count, dtype=np.float32)[None], dtype=wp.float32, device=device)
        jacobian = wp.array(
            [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0]]],
            dtype=wp.float32,
            device=device,
        )
        diagonal = wp.array([[1.0, 0.0, 0.0, 1.0]], dtype=wp.float32, device=device)
        impulses = wp.array([[0.0, 0.0, 0.0, 4.0]], dtype=wp.float32, device=device)
        velocity = wp.zeros(dof_count, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(
            max_constraints,
            max_constraints,
            dof_count,
            device.arch,
            dense_response_matrix=True,
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                candidate,
                candidate,
                PGS_LOCAL_SOLVE_OWNER_SINGLE,
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([PGS_LOCAL_SOLVE_OWNER_SINGLE], dtype=wp.int32, device=device),
                wp.array([max_constraints], dtype=wp.int32, device=device),
                lower,
                jacobian,
                lower,
                jacobian,
                wp.array([[0.0, -1.0, -2.0, 0.0]], dtype=wp.float32, device=device),
                wp.array(
                    [
                        [
                            PGS_CONSTRAINT_TYPE_CONTACT,
                            PGS_CONSTRAINT_TYPE_FRICTION,
                            PGS_CONSTRAINT_TYPE_FRICTION,
                            PGS_CONSTRAINT_TYPE_CONTACT,
                        ]
                    ],
                    dtype=wp.int32,
                    device=device,
                ),
                # Normal rows 0 -> 3 -> 0 pool the load. Tangent rows point to row 0.
                wp.array([[3, 0, 0, 0]], dtype=wp.int32, device=device),
                wp.array([[0.0, 0.5, 0.5, 0.0]], dtype=wp.float32, device=device),
                *self._empty_mf_inputs(1, device),
                1,
                1.0,
                0,
                0,
            ],
            outputs=[diagonal, impulses, velocity],
            block_dim=32,
            device=device,
        )
        wp.synchronize_device(device)

        # The pooled normal impulse gives radius 2. The unconstrained coupled
        # tangent solution of [[1, .5], [.5, 1.25]] x = [1, 2] is [.25, 1.5].
        np.testing.assert_allclose(impulses.numpy()[0], [0.0, 0.25, 1.5, 4.0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(velocity.numpy(), [1.0, 1.5], rtol=0.0, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_residual_queue_matches_sequential_dense_and_matrix_free_pgs(self):
        device = wp.get_cuda_device()
        primary_dofs = 3
        secondary_dofs = 6
        max_constraints = 4
        iterations = 4

        primary_jacobian = np.zeros((2, max_constraints, primary_dofs), dtype=np.float32)
        secondary_jacobian = np.zeros((2, max_constraints, secondary_dofs), dtype=np.float32)
        primary_jacobian[0, 0] = [0.5, -0.25, 0.1]
        secondary_jacobian[1, 0, 0] = 0.4
        dense_rhs = np.zeros((1, max_constraints), dtype=np.float32)
        dense_rhs[0, 0] = 0.2
        dense_cfm = np.zeros((1, max_constraints), dtype=np.float32)
        dense_cfm[0, 0] = 0.02
        initial_velocity = np.array([0.25, -0.1, 0.05, -0.3, 0.2, -0.1, 0.0, 0.0, 0.0], dtype=np.float32)

        mf_count = 3
        mf_jacobian = np.zeros((1, max_constraints, 6), dtype=np.float32)
        mf_response = np.zeros_like(mf_jacobian)
        mf_jacobian[0, 0, 0] = 1.0
        mf_jacobian[0, 1, 1] = 1.0
        mf_jacobian[0, 2, 2] = 1.0
        mf_response[:] = mf_jacobian
        mf_rhs = np.array([-0.1, 0.0, 0.0], dtype=np.float32)
        mf_types = np.array(
            [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION],
            dtype=np.int32,
        )
        mf_mu = np.array([0.0, 0.5, 0.5], dtype=np.float32)

        def float_bits(value: float) -> np.int32:
            return np.asarray(value, dtype=np.float32).view(np.int32).item()

        packed_meta = np.zeros((1, max_constraints * 4), dtype=np.int32)
        packed_dofs = np.asarray((0xFFFF << 16) | primary_dofs, dtype=np.uint32).view(np.int32).item()
        for row in range(mf_count):
            packed_meta[0, row * 4 : row * 4 + 4] = [
                packed_dofs,
                float_bits(1.0),
                float_bits(mf_rhs[row]),
                int(mf_types[row]) | (0 << 16),
            ]

        dense_impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        mf_impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        diagonal = wp.array(dense_cfm, dtype=wp.float32, device=device)
        velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(
            max_constraints,
            2,
            primary_dofs,
            device.arch,
            paired_dof_count=secondary_dofs,
            persistent_queue=True,
            warps_per_block=2,
            mf_max_constraints=max_constraints,
            local_mf_max_constraints=max_constraints,
            dense_response_matrix=True,
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                2,
                PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 0], dtype=wp.int32, device=device),
                wp.array([0, primary_dofs], dtype=wp.int32, device=device),
                wp.array([PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                wp.array(np.stack([np.eye(primary_dofs, dtype=np.float32)] * 2), device=device),
                wp.array(primary_jacobian, device=device),
                wp.array(np.stack([np.eye(secondary_dofs, dtype=np.float32)] * 2), device=device),
                wp.array(secondary_jacobian, device=device),
                wp.array(dense_rhs, device=device),
                wp.array([[PGS_CONSTRAINT_TYPE_MIMIC, 0, 0, 0]], dtype=wp.int32, device=device),
                wp.full((1, max_constraints), -1, dtype=wp.int32, device=device),
                wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
                wp.array([mf_count], dtype=wp.int32, device=device),
                wp.array(packed_meta, device=device),
                mf_impulses,
                wp.zeros_like(wp.array(mf_jacobian, device=device)),
                wp.array(mf_jacobian, device=device),
                wp.zeros_like(wp.array(mf_response, device=device)),
                wp.array(mf_response, device=device),
                wp.array([np.pad(mf_mu, (0, 1))], device=device),
                0,
                wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
                wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
                iterations,
                1.0,
                0,
                0,
            ],
            outputs=[diagonal, dense_impulses, velocity],
            block_dim=64,
            device=device,
        )
        wp.synchronize_device(device)

        dense_jacobian = np.concatenate([primary_jacobian[0, 0], secondary_jacobian[1, 0]])
        dense_response = dense_jacobian.copy()
        dense_denominator = float(dense_jacobian @ dense_response + dense_cfm[0, 0])
        expected_velocity = initial_velocity.copy()
        expected_dense_impulse = np.float32(0.0)
        expected_mf_impulses = np.zeros(mf_count, dtype=np.float32)
        for _ in range(iterations):
            dense_delta = -(dense_jacobian @ expected_velocity + dense_rhs[0, 0]) / dense_denominator
            dense_new = expected_dense_impulse + dense_delta
            expected_velocity += dense_response * (dense_new - expected_dense_impulse)
            expected_dense_impulse = dense_new

            for row in range(mf_count):
                old_impulse = expected_mf_impulses[row]
                new_impulse = old_impulse - (expected_velocity[primary_dofs + row] + mf_rhs[row])
                if mf_types[row] == PGS_CONSTRAINT_TYPE_CONTACT:
                    new_impulse = max(new_impulse, 0.0)
                else:
                    radius = max(mf_mu[row] * expected_mf_impulses[0], 0.0)
                    sibling = 2 if row == 1 else 1
                    trial = np.array([new_impulse, expected_mf_impulses[sibling]], dtype=np.float32)
                    magnitude = float(np.linalg.norm(trial))
                    if radius <= 0.0:
                        new_impulse = 0.0
                    elif magnitude > radius:
                        trial *= radius / magnitude
                        new_impulse = trial[0]
                        sibling_delta = trial[1] - expected_mf_impulses[sibling]
                        expected_mf_impulses[sibling] = trial[1]
                        expected_velocity[primary_dofs + sibling] += sibling_delta
                expected_mf_impulses[row] = new_impulse
                expected_velocity[primary_dofs + row] += new_impulse - old_impulse

        np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=0.0, atol=3.0e-6)
        np.testing.assert_allclose(dense_impulses.numpy()[0, 0], expected_dense_impulse, rtol=0.0, atol=3.0e-6)
        np.testing.assert_allclose(mf_impulses.numpy()[0, :mf_count], expected_mf_impulses, rtol=0.0, atol=3.0e-6)


if __name__ == "__main__":
    unittest.main()
