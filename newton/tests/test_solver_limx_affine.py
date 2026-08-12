# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.limx.affine_types import mat1212, vec12
from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.block_csr_12 import BlockCsrBuilder12
from newton._src.solvers.limx.constraints.affine_arap import (
    ConstraintAffineARAP,
    _affine_arap_energy,
    _affine_arap_hessian_unscaled,
    mat99,
)
from newton._src.solvers.limx.mixed_linear_solver import (
    MixedPcgSolver,
    _apply_affine_preconditioner,
    _factor_affine_diagonal,
)
from newton._src.solvers.limx.mixed_operator import EmptyMixedDynamicOperator, MixedLinearOperator, MixedVector3x12
from newton._src.solvers.limx.operator import CompositeLinearOperator, EmptyDynamicConstraintOperator
from newton.solvers import AffineBodyModel, ConstraintAffineStaticPlaneContact, SolverLIMXAffine


@wp.kernel
def _evaluate_affine_arap_energy(
    states: wp.array[vec12],
    rigidities: wp.array[float],
    volumes: wp.array[float],
    energies: wp.array[float],
):
    body = wp.tid()
    energies[body] = _affine_arap_energy(states[body], rigidities[body], volumes[body])


@wp.kernel
def _evaluate_affine_arap_hessian(
    states: wp.array[vec12],
    hessians: wp.array[mat99],
):
    body = wp.tid()
    hessians[body] = _affine_arap_hessian_unscaled(states[body])


def _proper_rotation_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an SVD with proper left and right rotation bases."""
    left, singular_values, right_transpose = np.linalg.svd(matrix)
    if np.linalg.det(left) < 0.0:
        left[:, -1] *= -1.0
        singular_values[-1] *= -1.0
    if np.linalg.det(right_transpose) < 0.0:
        right_transpose[-1, :] *= -1.0
        singular_values[-1] *= -1.0
    return left, singular_values, right_transpose


def _affine_arap_energy_reference(matrix: np.ndarray, rigidity: float, volume: float) -> float:
    """Evaluate direct affine ARAP energy with an independent NumPy SVD."""
    left, _singular_values, right_transpose = _proper_rotation_svd(matrix)
    return float(rigidity * volume * np.sum((matrix - left @ right_transpose) ** 2))


def _affine_arap_gradient_reference(matrix: np.ndarray) -> np.ndarray:
    """Evaluate the unscaled direct affine ARAP gradient with NumPy."""
    left, _singular_values, right_transpose = _proper_rotation_svd(matrix)
    return 2.0 * (matrix - left @ right_transpose)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a proper non-axis-aligned rotation matrix."""
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _affine_state(translation: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Pack translation and row-major affine rows into one generalized state."""
    return np.concatenate((np.asarray(translation), np.asarray(matrix).reshape(-1)))


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


class TestAffineArap(unittest.TestCase):
    RIGIDITY = 3.25
    VOLUME = 0.7

    @staticmethod
    def _devices() -> list[str]:
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        return devices

    @classmethod
    def _evaluate_energy(cls, state: np.ndarray, device: str) -> float:
        energies = wp.empty(1, dtype=float, device=device)
        wp.launch(
            _evaluate_affine_arap_energy,
            dim=1,
            inputs=[
                wp.array([state], dtype=vec12, device=device),
                wp.array([cls.RIGIDITY], dtype=float, device=device),
                wp.array([cls.VOLUME], dtype=float, device=device),
            ],
            outputs=[energies],
            device=device,
        )
        return float(energies.numpy()[0])

    @classmethod
    def _assemble(
        cls,
        states: np.ndarray,
        rigidities: np.ndarray,
        volumes: np.ndarray,
        device: str,
    ) -> tuple[np.ndarray, object]:
        constraint = ConstraintAffineARAP(rigidities, volumes, len(states), device)
        builder = BlockCsrBuilder12(len(states))
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize(device)
        constraint.bind_hessian(matrix)
        forces = wp.zeros(len(states), dtype=vec12, device=device)
        constraint.accumulate_force_and_hessian(
            wp.array(states, dtype=vec12, device=device),
            forces,
            matrix.values,
        )
        return forces.numpy(), matrix

    def test_preserves_identity_and_non_axis_rotation(self):
        """Keep direct affine ARAP energy and force zero for proper rotations."""
        rotation = _rotation_matrix(np.asarray([1.0, 2.0, -0.5]), 0.73)
        states = np.asarray(
            [
                _affine_state(np.asarray([0.4, -0.7, 1.2]), np.eye(3)),
                _affine_state(np.asarray([-3.0, 0.25, 4.5]), rotation),
            ],
            dtype=np.float32,
        )

        for device in self._devices():
            with self.subTest(device=device):
                for state in states:
                    self.assertAlmostEqual(self._evaluate_energy(state, device), 0.0, delta=2.0e-6)
                forces, _matrix = self._assemble(
                    states,
                    np.full(2, self.RIGIDITY),
                    np.full(2, self.VOLUME),
                    device,
                )
                np.testing.assert_allclose(forces, 0.0, rtol=0.0, atol=5.0e-6)

    def test_force_matches_centered_energy_difference(self):
        """Match physical affine force to a centered finite-difference energy gradient."""
        matrix = np.asarray(
            [[1.18, 0.17, -0.06], [0.09, 0.83, 0.21], [-0.04, 0.12, 1.07]],
            dtype=np.float64,
        )
        state = _affine_state(np.asarray([2.5, -1.25, 0.75]), matrix)
        epsilon = 1.0e-4
        energy_gradient = np.zeros(12)
        for component in range(12):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[component] += epsilon
            state_minus[component] -= epsilon
            energy_gradient[component] = (
                _affine_arap_energy_reference(state_plus[3:].reshape(3, 3), self.RIGIDITY, self.VOLUME)
                - _affine_arap_energy_reference(state_minus[3:].reshape(3, 3), self.RIGIDITY, self.VOLUME)
            ) / (2.0 * epsilon)

        for device in self._devices():
            with self.subTest(device=device):
                forces, _matrix = self._assemble(
                    state[np.newaxis],
                    np.asarray([self.RIGIDITY]),
                    np.asarray([self.VOLUME]),
                    device,
                )
                np.testing.assert_array_equal(forces[0, :3], np.zeros(3, dtype=np.float32))
                np.testing.assert_allclose(forces[0], -energy_gradient, rtol=3.0e-3, atol=3.0e-3)

    def test_unprojected_hessian_matches_gradient_difference(self):
        """Match the analytic row-major Hessian to centered gradient differences."""
        matrix = np.asarray(
            [[1.31, 0.14, -0.07], [0.05, 0.93, 0.19], [-0.02, 0.11, 0.72]],
            dtype=np.float64,
        )
        state = _affine_state(np.zeros(3), matrix)
        epsilon = 1.0e-4
        finite_difference = np.empty((9, 9))
        for column in range(9):
            matrix_plus = matrix.copy().reshape(-1)
            matrix_minus = matrix.copy().reshape(-1)
            matrix_plus[column] += epsilon
            matrix_minus[column] -= epsilon
            gradient_plus = _affine_arap_gradient_reference(matrix_plus.reshape(3, 3)).reshape(-1)
            gradient_minus = _affine_arap_gradient_reference(matrix_minus.reshape(3, 3)).reshape(-1)
            finite_difference[:, column] = (gradient_plus - gradient_minus) / (2.0 * epsilon)

        for device in self._devices():
            with self.subTest(device=device):
                hessians = wp.empty(1, dtype=mat99, device=device)
                wp.launch(
                    _evaluate_affine_arap_hessian,
                    dim=1,
                    inputs=[wp.array([state], dtype=vec12, device=device)],
                    outputs=[hessians],
                    device=device,
                )
                np.testing.assert_allclose(hessians.numpy()[0], finite_difference, rtol=4.0e-3, atol=4.0e-3)

    def test_assembles_projected_diagonal_blocks(self):
        """Assemble one symmetric PSD affine-only diagonal block per body."""
        matrices = np.asarray(
            [
                [[0.65, 0.08, 0.0], [0.02, 0.7, -0.04], [0.0, 0.03, 0.8]],
                [[1.2, -0.1, 0.04], [0.07, 0.9, 0.16], [-0.03, 0.05, 1.05]],
            ],
            dtype=np.float64,
        )
        states = np.asarray(
            [
                _affine_state(np.asarray([1.0, 2.0, 3.0]), matrices[0]),
                _affine_state(np.asarray([-4.0, 5.0, -6.0]), matrices[1]),
            ]
        )
        rigidities = np.asarray([2.0, 4.5])
        volumes = np.asarray([0.6, 0.35])
        epsilon = 1.0e-5

        expected_blocks = []
        for matrix, rigidity, volume in zip(matrices, rigidities, volumes, strict=True):
            raw_hessian = np.empty((9, 9))
            for column in range(9):
                matrix_plus = matrix.copy().reshape(-1)
                matrix_minus = matrix.copy().reshape(-1)
                matrix_plus[column] += epsilon
                matrix_minus[column] -= epsilon
                raw_hessian[:, column] = (
                    _affine_arap_gradient_reference(matrix_plus.reshape(3, 3)).reshape(-1)
                    - _affine_arap_gradient_reference(matrix_minus.reshape(3, 3)).reshape(-1)
                ) / (2.0 * epsilon)
            eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (raw_hessian + raw_hessian.T))
            expected_blocks.append(
                rigidity * volume * (eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T)
            )

        for device in self._devices():
            with self.subTest(device=device):
                _forces, matrix = self._assemble(states, rigidities, volumes, device)
                np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 1, 2])
                np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 1])
                for body, expected in enumerate(expected_blocks):
                    block = matrix.values.numpy()[matrix.block_index(body, body)]
                    np.testing.assert_array_equal(block[:3], np.zeros((3, 12), dtype=np.float32))
                    np.testing.assert_array_equal(block[:, :3], np.zeros((12, 3), dtype=np.float32))
                    np.testing.assert_allclose(block, block.T, rtol=0.0, atol=2.0e-5)
                    np.testing.assert_allclose(block[3:, 3:], expected, rtol=4.0e-3, atol=4.0e-3)
                    self.assertGreaterEqual(float(np.linalg.eigvalsh(block)[0]), -2.0e-3)


class TestAffineBodyModel(unittest.TestCase):
    @staticmethod
    def _unit_tetrahedron() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        surface_triangles = np.asarray(
            [
                [0, 2, 1],
                [0, 1, 3],
                [0, 3, 2],
                [1, 2, 3],
            ],
            dtype=np.int32,
        )
        return vertices, tetrahedra, surface_triangles

    def test_integrates_exact_unit_tetrahedron_mass_and_gravity(self):
        """Integrate exact affine mass and lift gravity through the same Jacobian."""
        vertices, tetrahedra, surface_triangles = self._unit_tetrahedron()
        density = 6.0

        model = AffineBodyModel(
            vertices,
            tetrahedra,
            surface_triangles,
            density=density,
            rigidity=2.5,
            initial_transform=wp.transform_identity(),
            device="cpu",
        )

        expected_mass = np.zeros((12, 12), dtype=np.float64)
        spatial_blocks = ([0, 3, 4, 5], [1, 6, 7, 8], [2, 9, 10, 11])
        unit_tetrahedron_moment = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0375, -0.0125, -0.0125],
                [0.0, -0.0125, 0.0375, -0.0125],
                [0.0, -0.0125, -0.0125, 0.0375],
            ]
        )
        for indices in spatial_blocks:
            expected_mass[np.ix_(indices, indices)] = unit_tetrahedron_moment

        actual_mass = model.mass_matrices.numpy()[0]
        self.assertAlmostEqual(model.volumes.numpy()[0], 1.0 / 6.0, places=7)
        self.assertAlmostEqual(actual_mass[0, 0], density / 6.0, places=7)
        np.testing.assert_allclose(actual_mass, expected_mass, rtol=2.0e-6, atol=2.0e-7)
        np.testing.assert_allclose(actual_mass, actual_mass.T, rtol=0.0, atol=0.0)
        self.assertGreater(np.linalg.eigvalsh(actual_mass).min(), 0.0)
        np.testing.assert_allclose(
            model.gravity.numpy()[0],
            [0.0, 0.0, -9.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            rtol=2.0e-6,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(model.rigidities.numpy(), [2.5])

    def test_accepts_conditioned_thin_tetrahedron_gravity(self):
        """Accept harmless gravity-solve noise for a valid thin tetrahedron."""
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0e-4],
            ],
            dtype=np.float64,
        )
        tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        surface_triangles = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)

        model = AffineBodyModel(
            vertices,
            tetrahedra,
            surface_triangles,
            density=1.0,
            rigidity=0.0,
            initial_transform=wp.transform_identity(),
            device="cpu",
        )

        self.assertGreater(np.linalg.eigvalsh(model.mass_matrices.numpy()[0]).min(), 0.0)
        np.testing.assert_array_equal(
            model.gravity.numpy()[0],
            np.asarray(
                [0.0, 0.0, -9.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        )

    def test_maps_unit_tetrahedron_surface_with_affine_states(self):
        """Map every surface vertex with identity and a literal affine state."""
        vertices, tetrahedra, surface_triangles = self._unit_tetrahedron()
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")

        for device in devices:
            with self.subTest(device=device):
                model = AffineBodyModel(
                    vertices,
                    tetrahedra,
                    surface_triangles,
                    density=6.0,
                    rigidity=0.0,
                    initial_transform=wp.transform_identity(),
                    device=device,
                )
                output = wp.empty(4, dtype=wp.vec3, device=device)

                model.update_surface_positions(model.q, output)

                np.testing.assert_allclose(output.numpy(), vertices, rtol=0.0, atol=0.0)
                np.testing.assert_array_equal(model.surface_ownership.numpy(), [0, 0, 0, 0])
                np.testing.assert_array_equal(model.surface_triangle_indices.numpy(), surface_triangles)
                np.testing.assert_array_equal(model.qd.numpy(), np.zeros((1, 12)))

                affine_state = wp.array(
                    [vec12(2.375, -0.5, 1.1875, 2.0, 0.5, -1.0, 0.0, -1.0, 3.0, 0.25, 2.0, 0.5)],
                    dtype=vec12,
                    device=device,
                )
                model.update_surface_positions(affine_state, output)

                np.testing.assert_allclose(
                    output.numpy(),
                    [
                        [2.0, -1.0, 0.5],
                        [4.0, -1.0, 0.75],
                        [2.5, -2.0, 2.5],
                        [1.0, 2.0, 1.0],
                    ],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_orients_every_affine_surface_face_outward(self):
        """Orient every supplied tetrahedral boundary face away from its incident tetrahedron."""
        vertices, tetrahedra, outward_surface = self._unit_tetrahedron()
        mixed_winding_surface = outward_surface.copy()
        mixed_winding_surface[0, [1, 2]] = mixed_winding_surface[0, [2, 1]]

        model = AffineBodyModel(
            vertices,
            tetrahedra,
            mixed_winding_surface,
            density=1.0,
            rigidity=0.0,
            initial_transform=wp.transform_identity(),
            device="cpu",
        )

        np.testing.assert_array_equal(model.surface_triangle_indices.numpy(), outward_surface)

    def test_rejects_incomplete_affine_tetrahedral_boundary(self):
        """Reject a collision surface that omits a tetrahedral boundary face."""
        vertices, tetrahedra, surface_triangles = self._unit_tetrahedron()

        with self.assertRaisesRegex(ValueError, "complete tetrahedral boundary"):
            AffineBodyModel(
                vertices,
                tetrahedra,
                surface_triangles[:-1],
                density=1.0,
                rigidity=0.0,
                initial_transform=wp.transform_identity(),
                device="cpu",
            )

    def test_initializes_state_from_rigid_transform(self):
        """Initialize centered translation and affine rows from a rigid transform."""
        vertices, tetrahedra, surface_triangles = self._unit_tetrahedron()
        transform = wp.transform(
            wp.vec3(1.5, -2.0, 0.25),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0),
        )

        model = AffineBodyModel(
            vertices,
            tetrahedra,
            surface_triangles,
            density=1.0,
            rigidity=0.0,
            initial_transform=transform,
            device="cpu",
        )

        np.testing.assert_allclose(
            model.q.numpy()[0],
            [1.25, -1.75, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_builds_repeated_affine_instances(self):
        """Build independent affine states over one repeated rest mesh."""
        vertices, tetrahedra, surface_triangles = self._unit_tetrahedron()
        transforms = [
            wp.transform_identity(),
            wp.transform(
                wp.vec3(2.0, -1.0, 0.5),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi),
            ),
        ]

        model = AffineBodyModel.from_instances(
            vertices,
            tetrahedra,
            surface_triangles,
            density=6.0,
            rigidity=2.5,
            initial_transforms=transforms,
            device="cpu",
        )

        self.assertEqual(model.body_count, 2)
        self.assertEqual(model.surface_vertex_count, 8)
        self.assertEqual(model.surface_triangle_count, 8)
        np.testing.assert_array_equal(model.surface_ownership.numpy(), [0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(model.tetrahedron_indices.numpy(), [[0, 1, 2, 3], [4, 5, 6, 7]])
        np.testing.assert_array_equal(model.surface_triangle_indices.numpy()[4:], surface_triangles + 4)
        np.testing.assert_allclose(model.volumes.numpy(), [1.0 / 6.0, 1.0 / 6.0])
        np.testing.assert_allclose(model.rigidities.numpy(), [2.5, 2.5])
        np.testing.assert_allclose(model.mass_matrices.numpy()[0], model.mass_matrices.numpy()[1])

        output = wp.empty(8, dtype=wp.vec3, device="cpu")
        model.update_surface_positions(model.q, output)
        np.testing.assert_allclose(output.numpy()[:4], vertices, atol=1.0e-6)
        expected_second = np.column_stack((-vertices[:, 1], vertices[:, 0], vertices[:, 2]))
        expected_second += [2.0, -1.0, 0.5]
        np.testing.assert_allclose(output.numpy()[4:], expected_second, atol=1.0e-6)

    def test_rejects_invalid_affine_instance_transforms(self):
        """Reject empty, malformed, and non-rigid instance transforms."""
        vertices, tetrahedra, surface_triangles = self._unit_tetrahedron()
        arguments = {
            "rest_vertices": vertices,
            "tetrahedron_indices": tetrahedra,
            "surface_triangle_indices": surface_triangles,
            "density": 1.0,
            "rigidity": 0.0,
            "device": "cpu",
        }

        with self.assertRaisesRegex(ValueError, "initial_transforms"):
            AffineBodyModel.from_instances(**arguments, initial_transforms=[])
        with self.assertRaisesRegex(ValueError, "initial_transform"):
            AffineBodyModel.from_instances(**arguments, initial_transforms=[np.zeros(6)])
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            AffineBodyModel.from_instances(
                **arguments,
                initial_transforms=[wp.transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0)],
            )

    def test_rejects_invalid_body_data(self):
        """Reject malformed, non-finite, non-positive, and inverted body data."""
        vertices, tetrahedra, surface_triangles = self._unit_tetrahedron()
        cases = [
            {"rest_vertices": vertices[:, :2]},
            {"rest_vertices": np.where(vertices == 1.0, np.nan, vertices)},
            {"tetrahedron_indices": tetrahedra[:, :3]},
            {"tetrahedron_indices": np.asarray([[0, 1, 2, 4]], dtype=np.int32)},
            {"tetrahedron_indices": np.asarray([[0, 2, 1, 3]], dtype=np.int32)},
            {"surface_triangle_indices": surface_triangles[:, :2]},
            {"surface_triangle_indices": np.asarray([[0, 1, 4]], dtype=np.int32)},
            {"density": 0.0},
            {"rigidity": -1.0},
            {"initial_transform": wp.transform(np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)},
        ]
        defaults = {
            "rest_vertices": vertices,
            "tetrahedron_indices": tetrahedra,
            "surface_triangle_indices": surface_triangles,
            "density": 1.0,
            "rigidity": 0.0,
            "initial_transform": wp.transform_identity(),
            "device": "cpu",
        }

        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                AffineBodyModel(**(defaults | overrides))


class TestSolverLIMXAffine(unittest.TestCase):
    @staticmethod
    def _make_model(device: str, rigidity: float) -> AffineBodyModel:
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        surface_triangles = np.asarray(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.int32,
        )
        return AffineBodyModel(
            vertices,
            tetrahedra,
            surface_triangles,
            density=6.0,
            rigidity=rigidity,
            initial_transform=wp.transform_identity(),
            device=device,
        )

    @classmethod
    def _make_solver(
        cls,
        device: str,
        rigidity: float,
        matrix: np.ndarray | None = None,
        nonlinear_iterations: int = 2,
        linear_iterations: int = 8,
    ) -> SolverLIMXAffine:
        solver = SolverLIMXAffine(
            cls._make_model(device, rigidity),
            nonlinear_iterations=nonlinear_iterations,
            linear_iterations=linear_iterations,
        )
        if matrix is not None:
            state = _affine_state(np.zeros(3), matrix).astype(np.float32)
            wp.copy(solver.q, wp.array([state], dtype=vec12, device=device))
        return solver

    def test_matches_first_order_free_fall_without_affine_deformation(self):
        """Match analytic free fall to first-order accuracy and preserve identity."""
        solver = self._make_solver("cpu", rigidity=0.0, nonlinear_iterations=1, linear_iterations=4)
        initial_state = solver.q.numpy()[0].copy()
        dt = 0.01
        step_count = 100

        for _ in range(step_count):
            solver.step(dt)

        state = solver.q.numpy()[0]
        elapsed = step_count * dt
        expected_height = initial_state[2] + 0.5 * -9.81 * elapsed * elapsed
        first_order_tolerance = 0.5 * 9.81 * elapsed * dt + 5.0e-4
        self.assertAlmostEqual(float(state[2]), expected_height, delta=first_order_tolerance)
        np.testing.assert_allclose(state[:2], initial_state[:2], rtol=0.0, atol=2.0e-5)
        np.testing.assert_allclose(state[3:].reshape(3, 3), np.eye(3), rtol=0.0, atol=2.0e-5)

    def test_preserves_translated_small_body_during_free_fall(self):
        """Preserve small offset body geometry through uniform free fall."""
        local_vertices = 1.0e-3 * np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        surface_triangles = np.asarray(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
            dtype=np.int32,
        )
        dt = 0.01
        step_count = 10
        expected_displacement = np.asarray(
            [0.0, 0.0, -9.81 * dt * dt * step_count * (step_count + 1) / 2.0],
            dtype=np.float32,
        )

        for rest_offset in (0.2, 0.5):
            with self.subTest(rest_offset=rest_offset):
                vertices = local_vertices + rest_offset
                try:
                    model = AffineBodyModel(
                        vertices,
                        tetrahedra,
                        surface_triangles,
                        density=1.0,
                        rigidity=0.0,
                        initial_transform=wp.transform_identity(),
                        device="cpu",
                    )
                except ValueError as error:
                    self.fail(f"valid body construction failed at {rest_offset} m offset: {error}")
                solver = SolverLIMXAffine(
                    model,
                    nonlinear_iterations=1,
                    linear_iterations=8,
                )
                surface_positions = wp.empty(model.surface_vertex_count, dtype=wp.vec3, device="cpu")
                solver.update_surface_positions(surface_positions)
                initial_surface_positions = surface_positions.numpy().copy()

                for _ in range(step_count):
                    solver.step(dt)

                solver.update_surface_positions(surface_positions)
                final_surface_positions = surface_positions.numpy()
                displacement = final_surface_positions - initial_surface_positions
                state = solver.q.numpy()[0]

                self.assertTrue(np.isfinite(state).all())
                self.assertTrue(np.isfinite(solver.qd.numpy()).all())
                np.testing.assert_allclose(initial_surface_positions, vertices, rtol=0.0, atol=6.0e-8)
                np.testing.assert_allclose(
                    displacement,
                    np.broadcast_to(expected_displacement, displacement.shape),
                    rtol=0.0,
                    atol=2.0e-6,
                )
                np.testing.assert_allclose(state[3:].reshape(3, 3), np.eye(3), rtol=0.0, atol=2.0e-5)

    def test_reduces_affine_singular_value_error_with_rigidity(self):
        """Reduce stretch error from a perturbed positive affine matrix."""
        initial_matrix = np.diag([1.1, 0.9, 1.05]).astype(np.float32)
        solver = self._make_solver("cpu", rigidity=100.0, matrix=initial_matrix)
        initial_error = float(np.max(np.abs(np.linalg.svd(initial_matrix, compute_uv=False) - 1.0)))

        for _ in range(100):
            solver.step(0.01)

        final_matrix = solver.q.numpy()[0, 3:].reshape(3, 3)
        final_error = float(np.max(np.abs(np.linalg.svd(final_matrix, compute_uv=False) - 1.0)))
        self.assertLess(final_error, initial_error)

    def test_keeps_rigidifying_state_finite_with_positive_determinant(self):
        """Keep the recovered affine state finite and orientation preserving."""
        initial_matrix = np.diag([1.1, 0.9, 1.05]).astype(np.float32)
        solver = self._make_solver("cpu", rigidity=100.0, matrix=initial_matrix)

        for _ in range(100):
            solver.step(0.01)

        state = solver.q.numpy()[0]
        velocity = solver.qd.numpy()[0]
        self.assertTrue(np.isfinite(state).all())
        self.assertTrue(np.isfinite(velocity).all())
        self.assertGreater(float(np.linalg.det(state[3:].reshape(3, 3))), 0.0)

    def test_executes_exact_fixed_linear_iteration_count(self):
        """Execute the configured fixed PCG count without host convergence checks."""
        solver = self._make_solver(
            "cpu",
            rigidity=20.0,
            nonlinear_iterations=3,
            linear_iterations=7,
        )

        solver.step(0.01)

        self.assertEqual(solver.last_linear_iterations, 7)

    def test_warm_starts_only_first_newton_solve_of_each_frame(self):
        """Warm-start only the first Newton solve of each frame."""
        solver = self._make_solver(
            "cpu",
            rigidity=20.0,
            matrix=np.diag([1.05, 0.95, 1.02]),
            nonlinear_iterations=3,
            linear_iterations=2,
        )
        zero_initial_guess_sequence: list[bool] = []
        solve = solver.linear_solver.solve

        def record_initial_guess_policy(*args, **kwargs):
            zero_initial_guess_sequence.append(kwargs["zero_initial_guess"])
            return solve(*args, **kwargs)

        solver.linear_solver.solve = record_initial_guess_policy

        solver.step(0.01)
        solver.step(0.01)

        self.assertEqual(zero_initial_guess_sequence, [False, True, True, False, True, True])

    def test_integrates_real_affine_dynamic_contact(self):
        """Integrate a real affine plane-contact force through the Newton solve."""
        free_model = self._make_model("cpu", rigidity=0.0)
        contact_model = self._make_model("cpu", rigidity=0.0)
        free_model.gravity.zero_()
        contact_model.gravity.zero_()
        contact = ConstraintAffineStaticPlaneContact(
            contact_model,
            normal=(0.0, 0.0, 1.0),
            offset=0.0,
            thickness=0.3,
            stiffness=10.0,
            normal_damping=0.0,
            friction=0.0,
            friction_epsilon=0.1,
        )
        free_solver = SolverLIMXAffine(free_model, nonlinear_iterations=1, linear_iterations=16)
        contact_solver = SolverLIMXAffine(
            contact_model,
            nonlinear_iterations=1,
            linear_iterations=16,
            dynamic_operator=contact,
        )
        initial_state = free_solver.q.numpy().copy()

        free_solver.step(0.1)
        contact_solver.step(0.1)

        np.testing.assert_allclose(free_solver.q.numpy(), initial_state, rtol=0.0, atol=1.0e-6)
        self.assertGreater(float(contact_solver.q.numpy()[0, 2]), float(free_solver.q.numpy()[0, 2]))
        self.assertGreater(float(contact_solver.qd.numpy()[0, 2]), 0.0)
        self.assertIs(contact_solver.dynamic_operator, contact)
        self.assertTrue(np.isfinite(contact_solver.q.numpy()).all())

    def test_defaults_to_empty_affine_dynamic_operator(self):
        """Preserve collision-free behavior with the default empty operator."""
        solver = self._make_solver("cpu", rigidity=0.0, nonlinear_iterations=1)

        self.assertIsInstance(solver.dynamic_operator, EmptyMixedDynamicOperator)

    def test_rejects_mismatched_affine_dynamic_operator_domains(self):
        """Reject affine dynamic operators with mismatched bodies or devices."""
        model = self._make_model("cpu", rigidity=0.0)

        class Domain:
            def __init__(self, body_count, device):
                self.body_count = body_count
                self.device = wp.get_device(device)

        with self.assertRaisesRegex(ValueError, "body count"):
            SolverLIMXAffine(model, dynamic_operator=Domain(2, "cpu"))
        if wp.is_cuda_available():
            cuda_model = self._make_model("cuda:0", rigidity=0.0)
            cuda_contact = ConstraintAffineStaticPlaneContact(
                cuda_model,
                normal=(0.0, 0.0, 1.0),
                offset=0.0,
                thickness=0.1,
                stiffness=10.0,
                normal_damping=0.0,
                friction=0.0,
                friction_epsilon=0.1,
            )
            with self.assertRaisesRegex(ValueError, "device"):
                SolverLIMXAffine(model, dynamic_operator=cuda_contact)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_captures_and_replays_one_complete_step_on_cuda(self):
        """Capture and replay one complete affine Newton step on CUDA."""
        device = wp.get_device("cuda:0")
        if not wp.is_mempool_enabled(device):
            self.skipTest("CUDA graph capture requires the Warp memory pool")
        solver = self._make_solver("cuda:0", rigidity=25.0, nonlinear_iterations=2, linear_iterations=4)
        solver.step(0.01)

        with wp.ScopedCapture(device=device) as capture:
            solver.step(0.01)
        wp.capture_launch(capture.graph)

        self.assertIsNotNone(capture.graph)
        self.assertTrue(np.isfinite(solver.q.numpy()).all())
        self.assertTrue(np.isfinite(solver.qd.numpy()).all())
