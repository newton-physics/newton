# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.limx.affine_types import mat1212, vec12
from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.block_csr_12 import BlockCsrBuilder12
from newton._src.solvers.limx.mixed_linear_solver import (
    MixedPcgSolver,
    _apply_affine_preconditioner,
    _factor_affine_diagonal,
)
from newton._src.solvers.limx.mixed_operator import EmptyMixedDynamicOperator, MixedLinearOperator, MixedVector3x12
from newton._src.solvers.limx.operator import CompositeLinearOperator, EmptyDynamicConstraintOperator


@wp.kernel
def _apply_rank_one_mixed_operator(
    particle_input: wp.array[wp.vec3],
    affine_input: wp.array[vec12],
    particle_jacobian: wp.vec3,
    affine_jacobian: vec12,
    stiffness: float,
    particle_output: wp.array[wp.vec3],
    affine_output: wp.array[vec12],
):
    value = wp.dot(particle_jacobian, particle_input[0])
    for component in range(12):
        value += affine_jacobian[component] * affine_input[0][component]
    particle_output[0] += stiffness * value * particle_jacobian
    affine_output[0] += stiffness * value * affine_jacobian


@wp.kernel
def _accumulate_rank_one_mixed_diagonal(
    particle_jacobian: wp.vec3,
    affine_jacobian: vec12,
    stiffness: float,
    particle_diagonal: wp.array[wp.mat33],
    affine_diagonal: wp.array[mat1212],
):
    for row in range(3):
        for column in range(3):
            particle_diagonal[0][row, column] += stiffness * particle_jacobian[row] * particle_jacobian[column]
    for row in range(12):
        for column in range(12):
            affine_diagonal[0][row, column] += stiffness * affine_jacobian[row] * affine_jacobian[column]


class _RankOneMixedOperator:
    """Apply a literal one-row mixed Jacobian without storing 3-by-12 blocks."""

    def __init__(self, particle_jacobian: wp.vec3, affine_jacobian: vec12, stiffness: float):
        self.particle_jacobian = particle_jacobian
        self.affine_jacobian = affine_jacobian
        self.stiffness = stiffness

    def multiply(
        self,
        particle_input: wp.array[wp.vec3],
        affine_input: wp.array[vec12],
        particle_output: wp.array[wp.vec3],
        affine_output: wp.array[vec12],
    ) -> None:
        """Accumulate ``k J.T J`` into the split output vectors."""
        wp.launch(
            _apply_rank_one_mixed_operator,
            dim=1,
            inputs=[
                particle_input,
                affine_input,
                self.particle_jacobian,
                self.affine_jacobian,
                self.stiffness,
            ],
            outputs=[particle_output, affine_output],
            device=particle_input.device,
        )

    def accumulate_diagonal(
        self,
        particle_diagonal: wp.array[wp.mat33],
        affine_diagonal: wp.array[mat1212],
    ) -> None:
        """Accumulate the block-Jacobi diagonal of ``k J.T J``."""
        wp.launch(
            _accumulate_rank_one_mixed_diagonal,
            dim=1,
            inputs=[self.particle_jacobian, self.affine_jacobian, self.stiffness],
            outputs=[particle_diagonal, affine_diagonal],
            device=particle_diagonal.device,
        )


class TestAffineBlockCsr(unittest.TestCase):
    def test_sorts_columns_and_accumulates_duplicate_blocks(self):
        """Store one sorted block per coordinate after duplicate additions."""
        builder = BlockCsrBuilder12(2)
        builder.add_block(0, 1, np.eye(12, dtype=np.float32) * 2.0)
        builder.add_block(0, 0, np.eye(12, dtype=np.float32) * 3.0)
        builder.add_block(0, 1, np.eye(12, dtype=np.float32) * -0.5)

        matrix = builder.finalize("cpu")

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 2, 2])
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 1])
        np.testing.assert_allclose(matrix.values.numpy()[0], np.eye(12) * 3.0)
        np.testing.assert_allclose(matrix.values.numpy()[1], np.eye(12) * 1.5)

    def test_resolves_block_indices_after_finalization(self):
        """Resolve finalized coordinates and reject a coordinate absent from the pattern."""
        builder = BlockCsrBuilder12(2)
        builder.ensure_block(0, 1)
        builder.ensure_block(1, 0)

        matrix = builder.finalize("cpu")

        self.assertEqual(matrix.block_index(0, 1), 0)
        self.assertEqual(matrix.block_index(1, 0), 1)
        with self.assertRaises(ValueError):
            matrix.block_index(0, 0)

    def test_extracts_diagonal_and_clears_numerical_blocks(self):
        """Refresh cached diagonal blocks and clear all numerical values."""
        builder = BlockCsrBuilder12(2)
        builder.ensure_block(0, 0)
        builder.ensure_block(0, 1)
        matrix = builder.finalize("cpu")
        values = wp.array(
            [np.eye(12, dtype=np.float32) * 4.0, np.eye(12, dtype=np.float32) * -2.0],
            dtype=mat1212,
            device="cpu",
        )
        wp.copy(matrix.values, values)

        matrix.update_diagonal()

        np.testing.assert_allclose(matrix.diagonal.numpy()[0], np.eye(12) * 4.0)
        np.testing.assert_array_equal(matrix.diagonal.numpy()[1], np.zeros((12, 12)))

        matrix.clear_values()

        np.testing.assert_array_equal(matrix.values.numpy(), np.zeros((2, 12, 12)))
        np.testing.assert_array_equal(matrix.diagonal.numpy(), np.zeros((2, 12, 12)))

    def test_multiplies_native_twelve_dof_blocks(self):
        """Multiply native affine blocks without particle-row expansion."""
        builder = BlockCsrBuilder12(2)
        builder.add_block(0, 0, np.eye(12, dtype=np.float32) * 2.0)
        builder.add_block(0, 1, np.eye(12, dtype=np.float32) * -1.0)
        builder.add_block(1, 0, np.eye(12, dtype=np.float32) * -1.0)
        builder.add_block(1, 1, np.eye(12, dtype=np.float32) * 3.0)
        matrix = builder.finalize("cpu")
        x = wp.array([vec12(*range(12)), vec12(*range(12, 24))], dtype=vec12, device="cpu")
        output = wp.empty_like(x)

        matrix.multiply(x, output)

        dense = np.zeros((24, 24), dtype=np.float32)
        dense[:12, :12] = np.eye(12, dtype=np.float32) * 2.0
        dense[:12, 12:] = np.eye(12, dtype=np.float32) * -1.0
        dense[12:, :12] = np.eye(12, dtype=np.float32) * -1.0
        dense[12:, 12:] = np.eye(12, dtype=np.float32) * 3.0
        np.testing.assert_allclose(output.numpy().reshape(-1), dense @ np.arange(24, dtype=np.float32))

    def test_ensures_all_ordered_stencil_blocks(self):
        """Create and resolve every ordered block pair in a stencil."""
        builder = BlockCsrBuilder12(2)
        stencils = np.asarray([[1, 0]], dtype=np.int32)
        builder.ensure_stencil_blocks(stencils)

        matrix = builder.finalize("cpu")

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 2, 4])
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 1, 0, 1])
        np.testing.assert_array_equal(matrix.stencil_block_indices(stencils), [[3, 2, 1, 0]])

    def test_rejects_invalid_builder_inputs(self):
        """Reject invalid dimensions, coordinates, shapes, and coefficients."""
        with self.assertRaises(ValueError):
            BlockCsrBuilder12(0)

        builder = BlockCsrBuilder12(2)
        for row, column, value in [
            (0, 2, np.eye(12, dtype=np.float32)),
            (0, 0, np.zeros((11, 12), dtype=np.float32)),
            (0, 0, np.full((12, 12), np.nan, dtype=np.float32)),
        ]:
            with self.subTest(row=row, column=column, shape=value.shape), self.assertRaises(ValueError):
                builder.add_block(row, column, value)


class TestAffinePreconditioner(unittest.TestCase):
    def test_solves_literal_spd_block_residuals(self):
        """Solve literal residuals with one native affine SPD block."""
        r_factor = np.asarray(
            [
                [2.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 3.0, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.5, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.5, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.25, 0.0, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.75, 0.0, 0.25, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.25, 0.0, -0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.75, 0.0, 0.25],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.25, -0.75],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5],
            ],
            dtype=np.float32,
        )
        matrix = r_factor.T @ r_factor + np.eye(12, dtype=np.float32) * 0.5
        residuals = np.asarray(
            [
                np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
                np.asarray([-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
                np.asarray([3.0, -1.0, 4.0, -1.0, 5.0, -9.0, 2.0, -6.0, 5.0, -3.0, 5.0, -8.0]),
            ],
            dtype=np.float32,
        )

        for device in self._devices():
            with self.subTest(device=device):
                diagonal = wp.array([matrix, matrix, matrix], dtype=mat1212, device=device)
                factors = wp.empty_like(diagonal)
                regularization = wp.zeros(3, dtype=int, device=device)
                residual = wp.array(residuals, dtype=vec12, device=device)
                output = wp.empty_like(residual)

                wp.launch(
                    _factor_affine_diagonal,
                    dim=3,
                    inputs=[diagonal],
                    outputs=[factors, regularization],
                    device=device,
                )
                wp.launch(
                    _apply_affine_preconditioner,
                    dim=3,
                    inputs=[factors, residual],
                    outputs=[output],
                    device=device,
                )

                np.testing.assert_allclose(matrix @ output.numpy().T, residuals.T, rtol=2.0e-4, atol=2.0e-5)
                np.testing.assert_array_equal(regularization.numpy(), [0, 0, 0])

    def test_regularizes_semidefinite_block(self):
        """Regularize a semidefinite affine block before applying it."""
        matrix = np.zeros((12, 12), dtype=np.float32)
        matrix[0, 0] = 2.0
        residual_values = np.arange(1.0, 13.0, dtype=np.float32)

        for device in self._devices():
            with self.subTest(device=device):
                diagonal = wp.array([matrix], dtype=mat1212, device=device)
                factors = wp.empty_like(diagonal)
                regularization = wp.zeros(1, dtype=int, device=device)
                residual = wp.array([residual_values], dtype=vec12, device=device)
                output = wp.empty_like(residual)

                wp.launch(
                    _factor_affine_diagonal,
                    dim=1,
                    inputs=[diagonal],
                    outputs=[factors, regularization],
                    device=device,
                )
                wp.launch(
                    _apply_affine_preconditioner,
                    dim=1,
                    inputs=[factors, residual],
                    outputs=[output],
                    device=device,
                )

                self.assertEqual(regularization.numpy()[0], 1)
                self.assertTrue(np.isfinite(output.numpy()).all())

    def test_rejects_nonfinite_factorization_input(self):
        """Reject a non-finite affine block during factorization."""
        matrix = np.eye(12, dtype=np.float32)
        matrix[3, 7] = np.nan

        for device in self._devices():
            with self.subTest(device=device):
                diagonal = wp.array([matrix], dtype=mat1212, device=device)
                factors = wp.empty_like(diagonal)
                regularization = wp.zeros(1, dtype=int, device=device)

                wp.launch(
                    _factor_affine_diagonal,
                    dim=1,
                    inputs=[diagonal],
                    outputs=[factors, regularization],
                    device=device,
                )

                self.assertEqual(regularization.numpy()[0], 1)
                self.assertTrue(np.isfinite(factors.numpy()).all())

    @staticmethod
    def _devices() -> list[str]:
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        return devices


class TestMixedPcg(unittest.TestCase):
    def test_solves_rank_one_coupled_particle_affine_system_against_dense_reference(self):
        """Solve a coupled particle-affine system against a dense reference."""
        particle_static = np.diag([2.0, 3.0, 4.0]).astype(np.float32)
        affine_static = np.diag(np.linspace(2.5, 4.7, 12, dtype=np.float32))
        particle_jacobian = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
        affine_jacobian = np.asarray(
            [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 0.5, -0.25, 1.0, -0.75],
            dtype=np.float32,
        )
        stiffness = 0.75
        rhs_values = np.asarray(
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0],
            dtype=np.float32,
        )

        particle_builder = BlockCsrBuilder(1)
        particle_builder.add_block(0, 0, particle_static)
        particle_matrix = particle_builder.finalize("cpu")
        affine_builder = BlockCsrBuilder12(1)
        affine_builder.add_block(0, 0, affine_static)
        affine_matrix = affine_builder.finalize("cpu")
        particle_operator = CompositeLinearOperator(
            wp.array([1.0], dtype=float, device="cpu"),
            particle_matrix,
            EmptyDynamicConstraintOperator(),
            "cpu",
        )
        mixed_dynamic_operator = _RankOneMixedOperator(
            wp.vec3(*particle_jacobian),
            vec12(*affine_jacobian),
            stiffness,
        )
        operator = MixedLinearOperator(particle_operator, affine_matrix, mixed_dynamic_operator, "cpu")
        operator.prepare(wp.zeros(1, dtype=wp.vec3, device="cpu"), dt=1.0)
        expected_particle_diagonal = particle_static + np.eye(3, dtype=np.float32)
        expected_particle_diagonal += stiffness * np.outer(particle_jacobian, particle_jacobian)
        expected_affine_diagonal = affine_static + stiffness * np.outer(affine_jacobian, affine_jacobian)
        np.testing.assert_allclose(
            operator.particle_operator.diagonal.numpy()[0],
            expected_particle_diagonal,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            operator.affine_diagonal.numpy()[0],
            expected_affine_diagonal,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        solver = MixedPcgSolver(particle_count=1, affine_count=1, device="cpu")
        rhs = MixedVector3x12(
            wp.array([wp.vec3(*rhs_values[:3])], dtype=wp.vec3, device="cpu"),
            wp.array([vec12(*rhs_values[3:])], dtype=vec12, device="cpu"),
        )
        solution = MixedVector3x12(
            wp.empty_like(rhs.particle),
            wp.empty_like(rhs.affine),
        )

        solver.solve(operator, rhs, solution, iterations=32)

        jacobian = np.concatenate((particle_jacobian, affine_jacobian))
        dense_matrix = np.zeros((15, 15), dtype=np.float32)
        dense_matrix[:3, :3] = particle_static + np.eye(3, dtype=np.float32)
        dense_matrix[3:, 3:] = affine_static
        dense_matrix += stiffness * np.outer(jacobian, jacobian)
        expected = np.linalg.solve(dense_matrix, rhs_values)
        np.testing.assert_allclose(solution.particle.numpy().reshape(-1), expected[:3], rtol=2.0e-4, atol=2.0e-5)
        np.testing.assert_allclose(solution.affine.numpy().reshape(-1), expected[3:], rtol=2.0e-4, atol=2.0e-5)

    def test_solves_particle_only_system_with_empty_affine_vectors(self):
        """Solve a particle system while the affine side has zero rows."""
        particle_builder = BlockCsrBuilder(1)
        particle_builder.add_block(0, 0, np.eye(3, dtype=np.float32) * 2.0)
        particle_operator = CompositeLinearOperator(
            wp.array([1.0], dtype=float, device="cpu"),
            particle_builder.finalize("cpu"),
            EmptyDynamicConstraintOperator(),
            "cpu",
        )
        operator = MixedLinearOperator(particle_operator, None, EmptyMixedDynamicOperator(), "cpu")
        operator.prepare(wp.zeros(1, dtype=wp.vec3, device="cpu"), dt=1.0)
        rhs = MixedVector3x12(
            wp.array([wp.vec3(3.0, -6.0, 9.0)], dtype=wp.vec3, device="cpu"),
            wp.empty(0, dtype=vec12, device="cpu"),
        )
        solution = MixedVector3x12(wp.empty_like(rhs.particle), wp.empty_like(rhs.affine))

        MixedPcgSolver(1, 0, "cpu").solve(operator, rhs, solution, iterations=4)

        np.testing.assert_allclose(solution.particle.numpy(), [[1.0, -2.0, 3.0]], rtol=2.0e-5, atol=2.0e-6)
        self.assertEqual(len(solution.affine), 0)

    def test_solves_affine_only_system_with_empty_particle_vectors(self):
        """Solve an affine system while the particle side has zero rows."""
        affine_static = np.diag(np.linspace(2.0, 4.2, 12, dtype=np.float32))
        affine_builder = BlockCsrBuilder12(1)
        affine_builder.add_block(0, 0, affine_static)
        operator = MixedLinearOperator(None, affine_builder.finalize("cpu"), EmptyMixedDynamicOperator(), "cpu")
        operator.prepare(None, dt=1.0)
        rhs_values = np.arange(1.0, 13.0, dtype=np.float32)
        rhs = MixedVector3x12(
            wp.empty(0, dtype=wp.vec3, device="cpu"),
            wp.array([vec12(*rhs_values)], dtype=vec12, device="cpu"),
        )
        solution = MixedVector3x12(wp.empty_like(rhs.particle), wp.empty_like(rhs.affine))

        MixedPcgSolver(0, 1, "cpu").solve(operator, rhs, solution, iterations=16)

        np.testing.assert_allclose(
            solution.affine.numpy().reshape(-1),
            np.linalg.solve(affine_static, rhs_values),
            rtol=2.0e-4,
            atol=2.0e-5,
        )
        self.assertEqual(len(solution.particle), 0)
