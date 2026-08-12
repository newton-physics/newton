# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.constraints.anchor import ConstraintAnchor
from newton._src.solvers.limx.constraints.dihedral_bending import ConstraintDihedralBending
from newton._src.solvers.limx.constraints.distance import ConstraintDistance
from newton._src.solvers.limx.constraints.self_collision import (
    ConstraintSelfCollision,
    _ContactBuffer,
    _EdgeEdgeContactBuffer,
)
from newton._src.solvers.limx.constraints.triangle_elastic import ConstraintTriangleElastic
from newton._src.solvers.limx.linear_solver import PcgSolver
from newton._src.solvers.limx.operator import CompositeLinearOperator, EmptyDynamicConstraintOperator
from newton._src.solvers.limx.solver_newton import SolverLIMX
from newton.examples.cloth.example_cloth_limx import Example as ClothLimxExample
from newton.examples.cloth.example_cloth_limx_twist import Example as ClothLimxTwistExample
from newton.viewer import ViewerNull


class TestBlockCsr(unittest.TestCase):
    def test_batched_stencils_build_unique_csr_and_indices(self):
        """Build sorted unique CSR blocks and stencil mappings in one batch."""
        builder = BlockCsrBuilder(3)
        stencils = np.asarray([[2, 0], [1, 2], [2, 0]], dtype=np.int32)
        builder.ensure_stencil_blocks(stencils)

        matrix = builder.finalize("cpu")
        block_indices = matrix.stencil_block_indices(stencils)

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 2, 4, 7])
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 2, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(block_indices, [[6, 4, 1, 0], [2, 3, 5, 6], [6, 4, 1, 0]])

    def test_zero_pattern_maps_sorted_block_indices(self):
        builder = BlockCsrBuilder(2)
        builder.ensure_block(1, 0)
        builder.ensure_block(0, 1)
        builder.ensure_block(0, 0)

        matrix = builder.finalize("cpu")

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 2, 3])
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 1, 0])
        self.assertEqual(matrix.block_index(0, 0), 0)
        self.assertEqual(matrix.block_index(0, 1), 1)
        self.assertEqual(matrix.block_index(1, 0), 2)
        np.testing.assert_array_equal(matrix.values.numpy(), np.zeros((3, 3, 3)))

    def test_mutable_values_refresh_multiply_and_diagonal(self):
        builder = BlockCsrBuilder(2)
        builder.ensure_block(0, 0)
        builder.ensure_block(0, 1)
        builder.ensure_block(1, 0)
        matrix = builder.finalize("cpu")
        values = wp.array(
            [wp.mat33(np.eye(3) * 2.0), wp.mat33(np.eye(3) * -1.0), wp.mat33(np.eye(3) * -1.0)],
            dtype=wp.mat33,
            device="cpu",
        )
        wp.copy(matrix.values, values)
        matrix.update_diagonal()
        x = wp.array(
            [wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0)],
            dtype=wp.vec3,
            device="cpu",
        )
        output = wp.empty_like(x)

        matrix.multiply(x, output)

        np.testing.assert_allclose(output.numpy(), [[-2.0, -1.0, 0.0], [-1.0, -2.0, -3.0]])
        np.testing.assert_allclose(matrix.diagonal.numpy()[0], np.eye(3) * 2.0)
        np.testing.assert_array_equal(matrix.diagonal.numpy()[1], np.zeros((3, 3)))

        matrix.clear_values()
        matrix.multiply(x, output)
        np.testing.assert_array_equal(output.numpy(), np.zeros((2, 3)))
        np.testing.assert_array_equal(matrix.diagonal.numpy(), np.zeros((2, 3, 3)))

    def test_duplicate_blocks_accumulate_and_columns_sort(self):
        builder = BlockCsrBuilder(3)
        builder.add_scaled_identity(0, 2, 1.0)
        builder.add_scaled_identity(0, 1, 2.0)
        builder.add_scaled_identity(0, 2, 3.0)

        matrix = builder.finalize("cpu")

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 2, 2, 2])
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [1, 2])
        blocks = matrix.values.numpy()
        np.testing.assert_allclose(blocks[0], np.eye(3) * 2.0)
        np.testing.assert_allclose(blocks[1], np.eye(3) * 4.0)
        np.testing.assert_allclose(matrix.diagonal.numpy(), np.zeros((3, 3, 3)))

    def test_multiply_matches_dense_block_matrix_with_empty_row(self):
        builder = BlockCsrBuilder(3)
        builder.add_scaled_identity(0, 0, 2.0)
        builder.add_scaled_identity(0, 1, -1.0)
        builder.add_scaled_identity(1, 0, -1.0)
        builder.add_scaled_identity(1, 1, 3.0)
        matrix = builder.finalize("cpu")
        x = wp.array(
            [wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0), wp.vec3(7.0)],
            dtype=wp.vec3,
            device="cpu",
        )
        output = wp.empty_like(x)

        matrix.multiply(x, output)

        np.testing.assert_allclose(
            output.numpy(),
            [[-2.0, -1.0, 0.0], [11.0, 13.0, 15.0], [0.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(matrix.diagonal.numpy()[0], np.eye(3) * 2.0)
        np.testing.assert_allclose(matrix.diagonal.numpy()[1], np.eye(3) * 3.0)

    def test_rejects_out_of_range_block_index(self):
        builder = BlockCsrBuilder(2)

        with self.assertRaises(ValueError):
            builder.add_scaled_identity(0, 2, 1.0)


class TestConstraintAnchor(unittest.TestCase):
    def test_force_restores_particle_to_target(self):
        anchors = ConstraintAnchor([0], [wp.vec3(1.0, 0.0, 0.0)], [10.0], 1, "cpu")
        positions = wp.array([wp.vec3(1.5, 0.0, 0.0)], dtype=wp.vec3, device="cpu")
        forces = wp.zeros(1, dtype=wp.vec3, device="cpu")

        anchors.accumulate_force(positions, forces)

        np.testing.assert_allclose(forces.numpy(), [[-5.0, 0.0, 0.0]])

    def test_force_is_zero_at_target(self):
        anchors = ConstraintAnchor([0], [wp.vec3(1.0, 2.0, 3.0)], [10.0], 1, "cpu")
        positions = wp.array([wp.vec3(1.0, 2.0, 3.0)], dtype=wp.vec3, device="cpu")
        forces = wp.zeros(1, dtype=wp.vec3, device="cpu")

        anchors.accumulate_force(positions, forces)

        np.testing.assert_array_equal(forces.numpy(), [[0.0, 0.0, 0.0]])

    def test_hessian_adds_anchor_diagonal(self):
        anchors = ConstraintAnchor([1], [wp.vec3(0.0)], [7.0], 2, "cpu")
        builder = BlockCsrBuilder(2)
        anchors.append_hessian_structure(builder)
        matrix = builder.finalize("cpu")
        anchors.bind_hessian(matrix)
        positions = wp.zeros(2, dtype=wp.vec3, device="cpu")
        forces = wp.zeros_like(positions)

        matrix.clear_values()
        anchors.accumulate_force_and_hessian(positions, forces, matrix.values)
        matrix.update_diagonal()

        np.testing.assert_allclose(matrix.diagonal.numpy()[0], np.zeros((3, 3)))
        np.testing.assert_allclose(matrix.diagonal.numpy()[1], np.eye(3) * 7.0)

    def test_rejects_invalid_input(self):
        invalid_arguments = [
            ([0], [], [1.0], 1),
            ([-1], [wp.vec3(0.0)], [1.0], 1),
            ([1], [wp.vec3(0.0)], [1.0], 1),
            ([0], [wp.vec3(0.0)], [0.0], 1),
            ([0], [wp.vec3(0.0)], [float("nan")], 1),
        ]
        for indices, targets, stiffnesses, particle_count in invalid_arguments:
            with self.subTest(indices=indices, stiffnesses=stiffnesses), self.assertRaises(ValueError):
                ConstraintAnchor(indices, targets, stiffnesses, particle_count, "cpu")


class TestConstraintDistance(unittest.TestCase):
    @staticmethod
    def assemble_single_spring(current_position, device="cpu"):
        springs = ConstraintDistance([(0, 1)], [1.0], [5.0], 2, device)
        builder = BlockCsrBuilder(2)
        springs.append_hessian_structure(builder)
        matrix = builder.finalize(device)
        springs.bind_hessian(matrix)
        positions = wp.array([wp.vec3(0.0), wp.vec3(*current_position)], dtype=wp.vec3, device=device)
        forces = wp.zeros(2, dtype=wp.vec3, device=device)

        matrix.clear_values()
        springs.accumulate_force_and_hessian(positions, forces, matrix.values)
        matrix.update_diagonal()

        return springs, matrix, positions, forces

    def test_stretched_spring_forces_are_equal_and_opposite(self):
        springs = ConstraintDistance([(0, 1)], [1.0], [20.0], 2, "cpu")
        positions = wp.array(
            [wp.vec3(0.0), wp.vec3(1.5, 0.0, 0.0)],
            dtype=wp.vec3,
            device="cpu",
        )
        forces = wp.zeros(2, dtype=wp.vec3, device="cpu")

        springs.accumulate_force(positions, forces)

        np.testing.assert_allclose(forces.numpy(), [[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])

    def test_spring_force_is_zero_at_rest_length(self):
        springs = ConstraintDistance([(0, 1)], [1.0], [20.0], 2, "cpu")
        positions = wp.array([wp.vec3(0.0), wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3, device="cpu")
        forces = wp.zeros(2, dtype=wp.vec3, device="cpu")

        springs.accumulate_force(positions, forces)

        np.testing.assert_allclose(forces.numpy(), np.zeros((2, 3)), atol=1.0e-7)

    def test_zero_length_spring_is_finite(self):
        _, matrix, _, forces = self.assemble_single_spring((0.0, 0.0, 0.0))

        self.assertTrue(np.isfinite(forces.numpy()).all())
        np.testing.assert_array_equal(forces.numpy(), np.zeros((2, 3)))
        self.assertTrue(np.isfinite(matrix.values.numpy()).all())
        np.testing.assert_array_equal(matrix.values.numpy(), np.zeros((4, 3, 3)))

    def test_rest_hessian_has_zero_transverse_curvature(self):
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        for device in devices:
            with self.subTest(device=device):
                _, matrix, _, forces = self.assemble_single_spring((1.0, 0.0, 0.0), device)
                expected = np.diag([5.0, 0.0, 0.0])

                np.testing.assert_allclose(forces.numpy(), np.zeros((2, 3)), atol=1.0e-7)
                np.testing.assert_allclose(
                    matrix.values.numpy(),
                    [expected, -expected, -expected, expected],
                    atol=1.0e-6,
                )

    def test_stretched_hessian_preserves_positive_transverse_curvature(self):
        _, matrix, _, forces = self.assemble_single_spring((1.2, 0.0, 0.0))
        expected = np.diag([5.0, 5.0 / 6.0, 5.0 / 6.0])

        np.testing.assert_allclose(forces.numpy(), [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], atol=1.0e-6)
        np.testing.assert_allclose(
            matrix.values.numpy(),
            [expected, -expected, -expected, expected],
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_compressed_hessian_removes_negative_transverse_curvature(self):
        _, matrix, _, forces = self.assemble_single_spring((0.8, 0.0, 0.0))
        expected = np.diag([5.0, 0.0, 0.0])

        np.testing.assert_allclose(forces.numpy(), [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atol=1.0e-6)
        np.testing.assert_allclose(
            matrix.values.numpy(),
            [expected, -expected, -expected, expected],
            atol=1.0e-6,
        )

    def test_reassembly_updates_values_without_changing_pattern(self):
        springs, matrix, _, _ = self.assemble_single_spring((1.2, 0.0, 0.0))
        row_offsets = matrix.row_offsets.numpy().copy()
        column_indices = matrix.column_indices.numpy().copy()
        x_axis_values = matrix.values.numpy().copy()
        positions = wp.array([wp.vec3(0.0), wp.vec3(0.0, 1.2, 0.0)], dtype=wp.vec3, device="cpu")
        forces = wp.zeros(2, dtype=wp.vec3, device="cpu")

        matrix.clear_values()
        springs.accumulate_force_and_hessian(positions, forces, matrix.values)

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), row_offsets)
        np.testing.assert_array_equal(matrix.column_indices.numpy(), column_indices)
        self.assertFalse(np.allclose(matrix.values.numpy(), x_axis_values))
        np.testing.assert_allclose(matrix.values.numpy()[0], np.diag([5.0 / 6.0, 5.0, 5.0 / 6.0]), atol=1.0e-6)

    def test_rejects_invalid_input(self):
        invalid_arguments = [
            ([(0, 1)], [], [1.0], 2),
            ([(0, 0)], [1.0], [1.0], 1),
            ([(0, 2)], [1.0], [1.0], 2),
            ([(0, 1)], [0.0], [1.0], 2),
            ([(0, 1)], [1.0], [-1.0], 2),
            ([(0, 1)], [1.0], [float("inf")], 2),
        ]
        for index_pairs, rest_lengths, stiffnesses, particle_count in invalid_arguments:
            with self.subTest(index_pairs=index_pairs), self.assertRaises(ValueError):
                ConstraintDistance(index_pairs, rest_lengths, stiffnesses, particle_count, "cpu")


class TestConstraintTriangleElastic(unittest.TestCase):
    @staticmethod
    def membrane_energy(positions, inverse_rest_matrix, rest_area, stiffness):
        edge_01 = positions[1] - positions[0]
        edge_02 = positions[2] - positions[0]
        deformation_u = edge_01 * inverse_rest_matrix[0, 0] + edge_02 * inverse_rest_matrix[1, 0]
        deformation_v = edge_01 * inverse_rest_matrix[0, 1] + edge_02 * inverse_rest_matrix[1, 1]
        stretch_u = np.linalg.norm(deformation_u) - 1.0
        stretch_v = np.linalg.norm(deformation_v) - 1.0
        shear = float(np.dot(deformation_u, deformation_v))
        return (
            0.5
            * rest_area
            * (
                stiffness[0] * stretch_u * stretch_u
                + stiffness[1] * stretch_v * stretch_v
                + stiffness[2] * shear * shear
            )
        )

    @staticmethod
    def make_constraint(stiffness=(7.0, 11.0, 5.0), device="cpu"):
        return ConstraintTriangleElastic(
            [(0, 1, 2)],
            [wp.mat22(1.0, 0.0, 0.0, 1.0)],
            [0.5],
            [wp.vec3(*stiffness)],
            3,
            device,
        )

    @staticmethod
    def projected_hessian_reference(positions, inverse_rest_matrix, rest_area, stiffness):
        edge_01 = positions[1] - positions[0]
        edge_02 = positions[2] - positions[0]
        deformation_u = edge_01 * inverse_rest_matrix[0, 0] + edge_02 * inverse_rest_matrix[1, 0]
        deformation_v = edge_01 * inverse_rest_matrix[0, 1] + edge_02 * inverse_rest_matrix[1, 1]
        derivative_u = np.asarray(
            [
                -inverse_rest_matrix[0, 0] - inverse_rest_matrix[1, 0],
                inverse_rest_matrix[0, 0],
                inverse_rest_matrix[1, 0],
            ]
        )
        derivative_v = np.asarray(
            [
                -inverse_rest_matrix[0, 1] - inverse_rest_matrix[1, 1],
                inverse_rest_matrix[0, 1],
                inverse_rest_matrix[1, 1],
            ]
        )

        def projected_stretch_curvature(deformation, component_stiffness):
            length = np.linalg.norm(deformation)
            if length <= 1.0e-8:
                return np.zeros((3, 3))
            direction = deformation / length
            normal_outer = np.outer(direction, direction)
            return component_stiffness * (normal_outer + max(1.0 - 1.0 / length, 0.0) * (np.eye(3) - normal_outer))

        curvature_u = projected_stretch_curvature(deformation_u, stiffness[0])
        curvature_v = projected_stretch_curvature(deformation_v, stiffness[1])
        shear_gradients = [derivative_u[i] * deformation_v + derivative_v[i] * deformation_u for i in range(3)]
        hessian = np.empty((9, 9))
        for i in range(3):
            for j in range(3):
                block = rest_area * (
                    derivative_u[i] * derivative_u[j] * curvature_u
                    + derivative_v[i] * derivative_v[j] * curvature_v
                    + stiffness[2] * np.outer(shear_gradients[i], shear_gradients[j])
                )
                hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = block
        return hessian

    @staticmethod
    def dense_hessian(matrix):
        values = matrix.values.numpy()
        dense = np.empty((9, 9))
        for i in range(3):
            for j in range(3):
                dense[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = values[matrix.block_index(i, j)]
        return dense

    @classmethod
    def assemble_single_triangle(cls, positions, stiffness=(7.0, 11.0, 5.0), device="cpu"):
        constraint = cls.make_constraint(stiffness, device)
        builder = BlockCsrBuilder(3)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize(device)
        constraint.bind_hessian(matrix)
        positions_wp = wp.array(positions, dtype=wp.vec3, device=device)
        forces = wp.zeros(3, dtype=wp.vec3, device=device)

        matrix.clear_values()
        constraint.accumulate_force_and_hessian(positions_wp, forces, matrix.values)
        matrix.update_diagonal()
        return constraint, matrix, positions_wp, forces

    def test_rigidly_rotated_rest_triangle_has_zero_force(self):
        constraint = self.make_constraint()
        positions = wp.array(
            [
                wp.vec3(0.2, -0.3, 1.0),
                wp.vec3(1.2, -0.3, 1.0),
                wp.vec3(0.2, -0.3, 2.0),
            ],
            dtype=wp.vec3,
            device="cpu",
        )
        forces = wp.zeros(3, dtype=wp.vec3, device="cpu")

        constraint.accumulate_force(positions, forces)

        np.testing.assert_allclose(forces.numpy(), np.zeros((3, 3)), atol=1.0e-6)

    def test_force_matches_negative_finite_difference_energy_gradient(self):
        stiffness = np.asarray([7.0, 11.0, 5.0])
        inverse_rest_matrix = np.eye(2)
        rest_area = 0.5
        positions = np.asarray(
            [
                [0.1, -0.2, 0.3],
                [1.25, 0.1, 0.5],
                [0.25, 0.9, -0.15],
            ],
            dtype=np.float64,
        )
        constraint = self.make_constraint(stiffness)
        positions_wp = wp.array(positions, dtype=wp.vec3, device="cpu")
        forces = wp.zeros(3, dtype=wp.vec3, device="cpu")

        constraint.accumulate_force(positions_wp, forces)

        epsilon = 1.0e-5
        energy_gradient = np.empty((3, 3))
        for particle in range(3):
            for axis in range(3):
                positions_plus = positions.copy()
                positions_minus = positions.copy()
                positions_plus[particle, axis] += epsilon
                positions_minus[particle, axis] -= epsilon
                energy_gradient[particle, axis] = (
                    self.membrane_energy(positions_plus, inverse_rest_matrix, rest_area, stiffness)
                    - self.membrane_energy(positions_minus, inverse_rest_matrix, rest_area, stiffness)
                ) / (2.0 * epsilon)

        np.testing.assert_allclose(forces.numpy(), -energy_gradient, rtol=2.0e-4, atol=2.0e-4)

    def test_projected_hessian_matches_analytic_blocks_and_is_psd(self):
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.25, 1.1, 0.1],
            ]
        )
        stiffness = np.asarray([7.0, 11.0, 5.0])
        _, matrix, _, _ = self.assemble_single_triangle(positions, stiffness)

        actual = self.dense_hessian(matrix)
        expected = self.projected_hessian_reference(positions, np.eye(2), 0.5, stiffness)

        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)
        np.testing.assert_allclose(actual, actual.T, atol=1.0e-6)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(actual)[0]), -1.0e-5)

    def test_compressed_warp_hessian_removes_negative_transverse_curvature(self):
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        _, matrix, _, _ = self.assemble_single_triangle(positions, (7.0, 0.0, 0.0))

        expected_material_block = np.diag([3.5, 0.0, 0.0])
        actual = self.dense_hessian(matrix)

        np.testing.assert_allclose(actual[0:3, 0:3], expected_material_block, atol=1.0e-6)
        np.testing.assert_allclose(actual[0:3, 3:6], -expected_material_block, atol=1.0e-6)
        np.testing.assert_allclose(actual[3:6, 0:3], -expected_material_block, atol=1.0e-6)
        np.testing.assert_allclose(actual[3:6, 3:6], expected_material_block, atol=1.0e-6)
        np.testing.assert_array_equal(actual[:, 6:9], np.zeros((9, 3)))
        np.testing.assert_array_equal(actual[6:9, :], np.zeros((3, 9)))

    def test_extended_stretch_hessian_matches_negative_force_jacobian(self):
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.2, 0.1],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        constraint, matrix, _, _ = self.assemble_single_triangle(positions, (7.0, 0.0, 0.0))
        epsilon = 1.0e-4
        force_jacobian = np.empty((9, 9))
        for column in range(9):
            positions_plus = positions.copy().reshape(-1)
            positions_minus = positions.copy().reshape(-1)
            positions_plus[column] += epsilon
            positions_minus[column] -= epsilon
            force_plus = wp.zeros(3, dtype=wp.vec3, device="cpu")
            force_minus = wp.zeros(3, dtype=wp.vec3, device="cpu")
            constraint.accumulate_force(wp.array(positions_plus.reshape(3, 3), dtype=wp.vec3, device="cpu"), force_plus)
            constraint.accumulate_force(
                wp.array(positions_minus.reshape(3, 3), dtype=wp.vec3, device="cpu"), force_minus
            )
            force_jacobian[:, column] = (force_plus.numpy().reshape(-1) - force_minus.numpy().reshape(-1)) / (
                2.0 * epsilon
            )

        np.testing.assert_allclose(self.dense_hessian(matrix), -force_jacobian, rtol=2.0e-3, atol=2.0e-3)

    def test_reassembly_changes_values_without_changing_nine_block_pattern(self):
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.25, 1.1, 0.1],
            ]
        )
        constraint, matrix, _, _ = self.assemble_single_triangle(positions)
        row_offsets = matrix.row_offsets.numpy().copy()
        column_indices = matrix.column_indices.numpy().copy()
        initial_values = matrix.values.numpy().copy()
        new_positions = wp.array(
            [wp.vec3(0.0), wp.vec3(1.3, 0.2, 0.0), wp.vec3(-0.1, 0.9, 0.3)],
            dtype=wp.vec3,
            device="cpu",
        )
        forces = wp.zeros(3, dtype=wp.vec3, device="cpu")

        matrix.clear_values()
        constraint.accumulate_force_and_hessian(new_positions, forces, matrix.values)

        np.testing.assert_array_equal(row_offsets, [0, 3, 6, 9])
        np.testing.assert_array_equal(column_indices, [0, 1, 2, 0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(matrix.row_offsets.numpy(), row_offsets)
        np.testing.assert_array_equal(matrix.column_indices.numpy(), column_indices)
        self.assertFalse(np.allclose(matrix.values.numpy(), initial_values))

    def test_rejects_invalid_input(self):
        identity = wp.mat22(1.0, 0.0, 0.0, 1.0)
        valid_stiffness = wp.vec3(1.0, 1.0, 1.0)
        invalid_arguments = [
            ([], [], [], [], 3),
            ([(0, 1, 2)], [], [0.5], [valid_stiffness], 3),
            ([(0, 0, 2)], [identity], [0.5], [valid_stiffness], 3),
            ([(0, 1, 2, 2)], [identity], [0.5], [valid_stiffness], 3),
            ([(0, 1, 3)], [identity], [0.5], [valid_stiffness], 3),
            ([(0, 1, 2)], [wp.mat22(1.0, 0.0, 0.0, 0.0)], [0.5], [valid_stiffness], 3),
            ([(0, 1, 2)], [wp.mat22(float("nan"), 0.0, 0.0, 1.0)], [0.5], [valid_stiffness], 3),
            ([(0, 1, 2)], [identity], [0.0], [valid_stiffness], 3),
            ([(0, 1, 2)], [identity], [float("inf")], [valid_stiffness], 3),
            ([(0, 1, 2)], [identity], [0.5], [wp.vec3(-1.0, 1.0, 1.0)], 3),
            ([(0, 1, 2)], [identity], [0.5], [wp.vec3(1.0, float("nan"), 1.0)], 3),
        ]
        for triangle_indices, inverse_rest_matrices, rest_areas, stiffnesses, particle_count in invalid_arguments:
            with self.subTest(triangle_indices=triangle_indices, rest_areas=rest_areas), self.assertRaises(ValueError):
                ConstraintTriangleElastic(
                    triangle_indices,
                    inverse_rest_matrices,
                    rest_areas,
                    stiffnesses,
                    particle_count,
                    "cpu",
                )

    def test_rejects_runtime_array_size_mismatch(self):
        constraint = self.make_constraint()

        with self.assertRaisesRegex(ValueError, "3 particle rows"):
            constraint.accumulate_force(
                wp.zeros(2, dtype=wp.vec3, device="cpu"),
                wp.zeros(3, dtype=wp.vec3, device="cpu"),
            )

    def test_rejects_hessian_value_count_mismatch_before_kernel_launch(self):
        constraint = self.make_constraint()
        builder = BlockCsrBuilder(3)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize("cpu")
        constraint.bind_hessian(matrix)
        positions = wp.zeros(3, dtype=wp.vec3, device="cpu")
        forces = wp.zeros(3, dtype=wp.vec3, device="cpu")

        for block_count in (len(matrix.values) + 1, len(matrix.values) - 1):
            with self.subTest(block_count=block_count), self.assertRaisesRegex(ValueError, "9 Hessian blocks"):
                constraint.accumulate_force_and_hessian(
                    positions,
                    forces,
                    wp.zeros(block_count, dtype=wp.mat33, device="cpu"),
                )


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintDihedralBending(unittest.TestCase):
    REST_POSITIONS = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.25, 1.0, 0.0],
            [0.75, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    STIFFNESS = 0.01

    @staticmethod
    def signed_angle(positions):
        edge = positions[1] - positions[0]
        edge_direction = edge / np.linalg.norm(edge)
        normal_1 = np.cross(positions[2] - positions[0], edge)
        normal_2 = np.cross(edge, positions[3] - positions[0])
        normal_1 /= np.linalg.norm(normal_1)
        normal_2 /= np.linalg.norm(normal_2)
        return float(
            np.arctan2(
                np.dot(np.cross(normal_1, normal_2), edge_direction),
                np.dot(normal_1, normal_2),
            )
        )

    @classmethod
    def bending_energy(cls, positions):
        rest_angle = cls.signed_angle(cls.REST_POSITIONS.astype(np.float64))
        angle = cls.signed_angle(positions)
        residual = np.arctan2(np.sin(angle - rest_angle), np.cos(angle - rest_angle))
        return 0.5 * cls.STIFFNESS * residual * residual

    @staticmethod
    def positions_at_angle(angle):
        return np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.25, 1.0, 0.0],
                [0.75, -np.cos(angle), -np.sin(angle)],
            ],
            dtype=np.float64,
        )

    @classmethod
    def make_constraint(cls):
        return ConstraintDihedralBending(
            [(0, 1, 2, 3)],
            cls.REST_POSITIONS,
            cls.STIFFNESS,
            4,
            "cuda:0",
        )

    @classmethod
    def numerical_angle_gradient(cls, positions):
        epsilon = 1.0e-5
        gradient = np.empty((4, 3))
        for particle in range(4):
            for axis in range(3):
                positions_plus = positions.copy()
                positions_minus = positions.copy()
                positions_plus[particle, axis] += epsilon
                positions_minus[particle, axis] -= epsilon
                angle_plus = cls.signed_angle(positions_plus)
                angle_minus = cls.signed_angle(positions_minus)
                angle_difference = np.arctan2(
                    np.sin(angle_plus - angle_minus),
                    np.cos(angle_plus - angle_minus),
                )
                gradient[particle, axis] = angle_difference / (2.0 * epsilon)
        return gradient

    @classmethod
    def assemble_single_dihedral(cls, positions):
        constraint = cls.make_constraint()
        builder = BlockCsrBuilder(4)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize("cuda:0")
        constraint.bind_hessian(matrix)
        positions_wp = wp.array(positions, dtype=wp.vec3, device="cuda:0")
        forces = wp.zeros(4, dtype=wp.vec3, device="cuda:0")

        matrix.clear_values()
        constraint.accumulate_force_and_hessian(positions_wp, forces, matrix.values)
        matrix.update_diagonal()
        return constraint, matrix, positions_wp, forces

    @staticmethod
    def dense_hessian(matrix):
        values = matrix.values.numpy()
        dense = np.empty((12, 12))
        for particle_i in range(4):
            for particle_j in range(4):
                dense[3 * particle_i : 3 * particle_i + 3, 3 * particle_j : 3 * particle_j + 3] = values[
                    matrix.block_index(particle_i, particle_j)
                ]
        return dense

    def test_force_is_zero_at_rest(self):
        constraint = self.make_constraint()
        positions = wp.array(self.REST_POSITIONS, dtype=wp.vec3, device="cuda:0")
        forces = wp.zeros(4, dtype=wp.vec3, device="cuda:0")

        constraint.accumulate_force(positions, forces)

        np.testing.assert_allclose(forces.numpy(), np.zeros((4, 3)), atol=1.0e-7)

    def test_force_is_zero_after_rigid_transform_of_rest_shape(self):
        constraint = self.make_constraint()
        angle = 0.7
        rotation = np.asarray(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        positions = self.REST_POSITIONS @ rotation.T + np.asarray([0.3, -0.4, 1.2])
        forces = wp.zeros(4, dtype=wp.vec3, device="cuda:0")

        constraint.accumulate_force(wp.array(positions, dtype=wp.vec3, device="cuda:0"), forces)

        np.testing.assert_allclose(forces.numpy(), np.zeros((4, 3)), atol=1.0e-7)

    def test_force_matches_negative_wrapped_energy_gradient(self):
        positions = np.asarray(
            [
                [0.05, -0.03, 0.02],
                [1.02, 0.04, -0.01],
                [0.21, 0.95, 0.17],
                [0.78, -0.82, -0.61],
            ],
            dtype=np.float64,
        )
        constraint = self.make_constraint()
        positions_wp = wp.array(positions, dtype=wp.vec3, device="cuda:0")
        forces = wp.zeros(4, dtype=wp.vec3, device="cuda:0")

        constraint.accumulate_force(positions_wp, forces)

        epsilon = 1.0e-4
        energy_gradient = np.empty((4, 3))
        for particle in range(4):
            for axis in range(3):
                positions_plus = positions.copy()
                positions_minus = positions.copy()
                positions_plus[particle, axis] += epsilon
                positions_minus[particle, axis] -= epsilon
                energy_gradient[particle, axis] = (
                    self.bending_energy(positions_plus) - self.bending_energy(positions_minus)
                ) / (2.0 * epsilon)

        np.testing.assert_allclose(forces.numpy(), -energy_gradient, rtol=2.0e-3, atol=2.0e-5)

    def test_gauss_newton_hessian_matches_angle_gradient_outer_product(self):
        positions = np.asarray(
            [
                [0.02, -0.04, 0.01],
                [1.03, 0.02, -0.03],
                [0.23, 0.91, 0.24],
                [0.81, -0.74, -0.67],
            ],
            dtype=np.float64,
        )
        _, matrix, _, _ = self.assemble_single_dihedral(positions)
        angle_gradient = self.numerical_angle_gradient(positions).reshape(-1)
        expected = self.STIFFNESS * np.outer(angle_gradient, angle_gradient)

        actual = self.dense_hessian(matrix)

        np.testing.assert_allclose(actual, expected, rtol=3.0e-3, atol=3.0e-5)
        np.testing.assert_allclose(actual, actual.T, atol=1.0e-7)
        eigenvalues = np.linalg.eigvalsh(actual)
        self.assertGreaterEqual(float(eigenvalues[0]), -1.0e-6)
        self.assertEqual(int(np.count_nonzero(eigenvalues > 1.0e-5)), 1)

    def test_reassembly_changes_values_without_changing_sixteen_block_pattern(self):
        initial_positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.25, 0.95, 0.2],
                [0.75, -0.8, -0.6],
            ],
            dtype=np.float32,
        )
        constraint, matrix, _, _ = self.assemble_single_dihedral(initial_positions)
        row_offsets = matrix.row_offsets.numpy().copy()
        column_indices = matrix.column_indices.numpy().copy()
        initial_values = matrix.values.numpy().copy()
        new_positions = wp.array(
            [
                wp.vec3(0.0, 0.0, 0.0),
                wp.vec3(1.0, 0.1, 0.0),
                wp.vec3(0.2, 0.85, 0.4),
                wp.vec3(0.8, -0.65, -0.75),
            ],
            dtype=wp.vec3,
            device="cuda:0",
        )
        forces = wp.zeros(4, dtype=wp.vec3, device="cuda:0")

        matrix.clear_values()
        constraint.accumulate_force_and_hessian(new_positions, forces, matrix.values)

        np.testing.assert_array_equal(row_offsets, [0, 4, 8, 12, 16])
        np.testing.assert_array_equal(column_indices, [0, 1, 2, 3] * 4)
        np.testing.assert_array_equal(matrix.row_offsets.numpy(), row_offsets)
        np.testing.assert_array_equal(matrix.column_indices.numpy(), column_indices)
        self.assertFalse(np.allclose(matrix.values.numpy(), initial_values))

    def test_force_uses_short_angle_residual_across_branch_cut(self):
        rest_positions = self.positions_at_angle(np.pi - 0.05)
        positions = self.positions_at_angle(-np.pi + 0.05)
        rest_angle = self.signed_angle(rest_positions)
        constraint = ConstraintDihedralBending(
            [(0, 1, 2, 3)],
            rest_positions,
            self.STIFFNESS,
            4,
            "cuda:0",
        )
        forces = wp.zeros(4, dtype=wp.vec3, device="cuda:0")

        constraint.accumulate_force(wp.array(positions, dtype=wp.vec3, device="cuda:0"), forces)

        def energy(candidate):
            angle = self.signed_angle(candidate)
            residual = np.arctan2(np.sin(angle - rest_angle), np.cos(angle - rest_angle))
            return 0.5 * self.STIFFNESS * residual * residual

        epsilon = 1.0e-4
        expected_force = np.empty((4, 3))
        for particle in range(4):
            for axis in range(3):
                positions_plus = positions.copy()
                positions_minus = positions.copy()
                positions_plus[particle, axis] += epsilon
                positions_minus[particle, axis] -= epsilon
                expected_force[particle, axis] = -(energy(positions_plus) - energy(positions_minus)) / (2.0 * epsilon)

        np.testing.assert_allclose(forces.numpy(), expected_force, rtol=3.0e-3, atol=2.0e-5)

    def test_runtime_degenerate_hinge_contributes_finite_zeros(self):
        constraint = self.make_constraint()
        builder = BlockCsrBuilder(4)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize("cuda:0")
        constraint.bind_hessian(matrix)
        positions = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
        forces = wp.zeros(4, dtype=wp.vec3, device="cuda:0")

        matrix.clear_values()
        constraint.accumulate_force_and_hessian(positions, forces, matrix.values)

        self.assertTrue(np.isfinite(forces.numpy()).all())
        self.assertTrue(np.isfinite(matrix.values.numpy()).all())
        np.testing.assert_array_equal(forces.numpy(), np.zeros((4, 3)))
        np.testing.assert_array_equal(matrix.values.numpy(), np.zeros((16, 3, 3)))

    def test_duplicate_hinges_accumulate_force_and_hessian(self):
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.25, 0.9, 0.3],
                [0.75, -0.75, -0.66],
            ],
            dtype=np.float32,
        )
        _, single_matrix, _, single_forces = self.assemble_single_dihedral(positions)
        duplicate = ConstraintDihedralBending(
            [(0, 1, 2, 3), (0, 1, 2, 3)],
            self.REST_POSITIONS,
            self.STIFFNESS,
            4,
            "cuda:0",
        )
        builder = BlockCsrBuilder(4)
        duplicate.append_hessian_structure(builder)
        duplicate_matrix = builder.finalize("cuda:0")
        duplicate.bind_hessian(duplicate_matrix)
        duplicate_forces = wp.zeros(4, dtype=wp.vec3, device="cuda:0")

        duplicate_matrix.clear_values()
        duplicate.accumulate_force_and_hessian(
            wp.array(positions, dtype=wp.vec3, device="cuda:0"),
            duplicate_forces,
            duplicate_matrix.values,
        )

        np.testing.assert_allclose(duplicate_forces.numpy(), 2.0 * single_forces.numpy(), rtol=1.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(
            duplicate_matrix.values.numpy(),
            2.0 * single_matrix.values.numpy(),
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    def test_rejects_invalid_input(self):
        valid_rest = self.REST_POSITIONS
        collapsed_edge = valid_rest.copy()
        collapsed_edge[1] = collapsed_edge[0]
        collapsed_height = valid_rest.copy()
        collapsed_height[2] = [0.5, 0.0, 0.0]
        nonfinite_rest = valid_rest.copy()
        nonfinite_rest[2, 1] = np.nan
        invalid_arguments = [
            ([], valid_rest, self.STIFFNESS, 4),
            ([(0, 1, 2)], valid_rest, self.STIFFNESS, 4),
            ([(0, 1, 2, 2)], valid_rest, self.STIFFNESS, 4),
            ([(0, 1, 2, 4)], valid_rest, self.STIFFNESS, 4),
            ([(0, 1, 2, 3)], valid_rest[:3], self.STIFFNESS, 4),
            ([(0, 1, 2, 3)], nonfinite_rest, self.STIFFNESS, 4),
            ([(0, 1, 2, 3)], collapsed_edge, self.STIFFNESS, 4),
            ([(0, 1, 2, 3)], collapsed_height, self.STIFFNESS, 4),
            ([(0, 1, 2, 3)], valid_rest, 0.0, 4),
            ([(0, 1, 2, 3)], valid_rest, float("inf"), 4),
        ]
        for dihedrals, rest_positions, stiffness, particle_count in invalid_arguments:
            with self.subTest(dihedrals=dihedrals, stiffness=stiffness), self.assertRaises(ValueError):
                ConstraintDihedralBending(
                    dihedrals,
                    rest_positions,
                    stiffness,
                    particle_count,
                    "cuda:0",
                )

    def test_rejects_unbound_or_wrong_hessian_buffer(self):
        constraint = self.make_constraint()
        positions = wp.array(self.REST_POSITIONS, dtype=wp.vec3, device="cuda:0")
        forces = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
        with self.assertRaisesRegex(RuntimeError, "bind_hessian"):
            constraint.accumulate_force_and_hessian(
                positions,
                forces,
                wp.zeros(16, dtype=wp.mat33, device="cuda:0"),
            )

        builder = BlockCsrBuilder(4)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize("cuda:0")
        constraint.bind_hessian(matrix)
        with self.assertRaisesRegex(ValueError, "16 Hessian blocks"):
            constraint.accumulate_force_and_hessian(
                positions,
                forces,
                wp.zeros(15, dtype=wp.mat33, device="cuda:0"),
            )


class TestCompositeLinearOperator(unittest.TestCase):
    def make_operator(self):
        builder = BlockCsrBuilder(2)
        builder.add_scaled_identity(0, 0, 2.0)
        builder.add_scaled_identity(0, 1, -1.0)
        builder.add_scaled_identity(1, 0, -1.0)
        builder.add_scaled_identity(1, 1, 4.0)
        matrix = builder.finalize("cpu")
        masses = wp.array([2.0, 3.0], dtype=float, device="cpu")
        operator = CompositeLinearOperator(
            masses=masses,
            static_matrix=matrix,
            dynamic_operator=EmptyDynamicConstraintOperator(),
            device="cpu",
        )
        positions = wp.zeros(2, dtype=wp.vec3, device="cpu")
        operator.prepare(positions, 0.5)
        return operator

    def test_multiply_combines_mass_and_static_hessian(self):
        operator = self.make_operator()
        x = wp.array(
            [wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0)],
            dtype=wp.vec3,
            device="cpu",
        )
        output = wp.empty_like(x)

        operator.multiply(x, output)

        np.testing.assert_allclose(output.numpy(), [[6.0, 15.0, 24.0], [63.0, 78.0, 93.0]])
        np.testing.assert_allclose(operator.inverse_diagonal.numpy()[0], np.eye(3) * 0.1)
        np.testing.assert_allclose(operator.inverse_diagonal.numpy()[1], np.eye(3) * 0.0625)

    def test_reassembled_values_update_operator_and_preconditioner(self):
        operator = self.make_operator()
        values = wp.array(
            np.asarray([np.eye(3) * 5.0, np.eye(3) * -2.0, np.eye(3) * -2.0, np.eye(3) * 6.0]),
            dtype=wp.mat33,
            device="cpu",
        )
        wp.copy(operator.static_matrix.values, values)
        operator.static_matrix.update_diagonal()
        operator.prepare(wp.zeros(2, dtype=wp.vec3, device="cpu"), 0.5)
        x = wp.array(
            [wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0)],
            dtype=wp.vec3,
            device="cpu",
        )
        output = wp.empty_like(x)

        operator.multiply(x, output)

        np.testing.assert_allclose(output.numpy(), [[5.0, 16.0, 27.0], [70.0, 86.0, 102.0]])
        np.testing.assert_allclose(operator.inverse_diagonal.numpy()[0], np.eye(3) / 13.0)
        np.testing.assert_allclose(operator.inverse_diagonal.numpy()[1], np.eye(3) / 18.0)


class TestPcgSolver(unittest.TestCase):
    def make_operator(self):
        return TestCompositeLinearOperator().make_operator()

    def test_tiled_dot_reduction_handles_partial_final_block(self):
        dimension = 517
        lhs_values = np.ones((dimension, 3), dtype=np.float32)
        rhs_values = np.tile(np.asarray([1.0, 2.0, 3.0], dtype=np.float32), (dimension, 1))
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")

        for device in devices:
            with self.subTest(device=device):
                solver = PcgSolver(dimension, device)
                lhs = wp.array(lhs_values, dtype=wp.vec3, device=device)
                rhs = wp.array(rhs_values, dtype=wp.vec3, device=device)
                output = wp.empty(1, dtype=float, device=device)

                solver._dot(lhs, rhs, output)

                self.assertEqual(float(output.numpy()[0]), 517.0 * 6.0)

    def test_solves_known_spd_block_system(self):
        operator = self.make_operator()
        rhs = wp.array(
            [wp.vec3(9.75, -23.0, 6.0), wp.vec3(3.0, 50.0, -16.5)],
            dtype=wp.vec3,
            device="cpu",
        )
        solution = wp.zeros(2, dtype=wp.vec3, device="cpu")
        solver = PcgSolver(2, "cpu")

        executed = solver.solve(operator, rhs, solution, iterations=20)

        self.assertEqual(executed, 20)
        np.testing.assert_allclose(solution.numpy(), [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]], rtol=1e-5, atol=1e-6)

    def test_nonzero_initial_guess_converges_to_same_solution(self):
        operator = self.make_operator()
        rhs = wp.array(
            [wp.vec3(9.75, -23.0, 6.0), wp.vec3(3.0, 50.0, -16.5)],
            dtype=wp.vec3,
            device="cpu",
        )
        solution = wp.array([wp.vec3(-2.0, 1.0, 4.0), wp.vec3(3.0, -1.0, 0.5)], dtype=wp.vec3, device="cpu")
        solver = PcgSolver(2, "cpu")

        solver.solve(operator, rhs, solution, iterations=20, zero_initial_guess=False)

        np.testing.assert_allclose(solution.numpy(), [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]], rtol=1e-5, atol=1e-6)

    def test_debug_tolerance_stops_before_iteration_limit(self):
        operator = self.make_operator()
        rhs = wp.array(
            [wp.vec3(9.75, -23.0, 6.0), wp.vec3(3.0, 50.0, -16.5)],
            dtype=wp.vec3,
            device="cpu",
        )
        solution = wp.zeros(2, dtype=wp.vec3, device="cpu")
        solver = PcgSolver(2, "cpu")

        executed = solver.solve(operator, rhs, solution, iterations=100, tolerance=1.0e-6, check_interval=1)

        self.assertLess(executed, 100)
        np.testing.assert_allclose(solution.numpy(), [[1.0, -2.0, 0.5], [0.25, 3.0, -1.0]], rtol=1e-5, atol=1e-6)

    def test_zero_rhs_breakdown_guard_stays_finite(self):
        operator = self.make_operator()
        rhs = wp.zeros(2, dtype=wp.vec3, device="cpu")
        solution = wp.zeros(2, dtype=wp.vec3, device="cpu")
        solver = PcgSolver(2, "cpu")

        solver.solve(operator, rhs, solution, iterations=5)

        self.assertTrue(np.isfinite(solution.numpy()).all())
        np.testing.assert_array_equal(solution.numpy(), np.zeros((2, 3)))


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSelfCollisionContactBuffer(unittest.TestCase):
    def test_adaptive_contact_uses_directional_feature_stiffness(self):
        """Scale every adaptive contact operator by directional feature stiffness."""
        weights = np.asarray([1.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0], dtype=np.float32)
        static_diagonal = np.zeros((4, 3, 3), dtype=np.float32)
        static_diagonal[:, 0, 0] = [6.0, 16.0, 26.0, 36.0]

        with wp.ScopedDevice("cuda:0"):
            contacts = _ContactBuffer(arity=4, feature_split=1, capacity=1, device="cuda:0")
            contacts.ids.assign(np.asarray([[0, 1, 2, 3]], dtype=np.int32))
            contacts.weights.assign(weights.reshape(1, 4))
            contacts.directions.assign(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
            contacts.depths.assign(np.asarray([0.2], dtype=np.float32))
            contacts.count.assign(np.asarray([1], dtype=np.int32))

            diagonal_blocks = wp.array(static_diagonal, dtype=wp.mat33, device="cuda:0")
            masses = wp.ones(4, dtype=float, device="cuda:0")
            vector = wp.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=wp.vec3,
                device="cuda:0",
            )
            force = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
            product = wp.zeros_like(force)
            diagonal = wp.zeros(4, dtype=wp.mat33, device="cuda:0")

            contacts.accumulate_force_adaptive(0.5, diagonal_blocks, masses, 4.0, force)
            contacts.hessian_multiply_adaptive(0.5, diagonal_blocks, masses, 4.0, vector, product)
            contacts.accumulate_diagonal_adaptive(0.5, diagonal_blocks, masses, 4.0, diagonal)

            force_np = force.numpy()
            product_np = product.numpy()
            diagonal_np = diagonal.numpy()

        np.testing.assert_allclose(
            force_np,
            [[0.75, 0.0, 0.0], [-0.25, 0.0, 0.0], [-0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            product_np,
            [[3.75, 0.0, 0.0], [-1.25, 0.0, 0.0], [-1.25, 0.0, 0.0], [-1.25, 0.0, 0.0]],
            atol=1.0e-6,
        )
        expected_diagonal = np.zeros((4, 3, 3), dtype=np.float32)
        expected_diagonal[:, 0, 0] = [3.75, 3.75 / 9.0, 3.75 / 9.0, 3.75 / 9.0]
        np.testing.assert_allclose(diagonal_np, expected_diagonal, atol=1.0e-6)

    def test_four_particle_contact_matches_dense_rank_one_system(self):
        """Keep VF contacts fully coupled across their four particles."""
        self._assert_contact_matches_dense_reference(
            weights=np.asarray([1.0, -0.2, -0.3, -0.5], dtype=np.float32),
            direction=np.asarray([0.0, 0.6, 0.8], dtype=np.float32),
            depth=0.04,
            stiffness=7.0,
        )

    def test_five_particle_contact_matches_dense_rank_one_system(self):
        """Keep EF contacts fully coupled across their five particles."""
        self._assert_contact_matches_dense_reference(
            weights=np.asarray([0.35, 0.65, -0.2, -0.3, -0.5], dtype=np.float32),
            direction=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            depth=0.02,
            stiffness=11.0,
        )

    def test_adaptive_five_particle_contact_keeps_cross_particle_blocks(self):
        """Keep adaptive EF Hessians coupled across the edge and face features."""
        weights = np.asarray([0.35, 0.65, -0.2, -0.3, -0.5], dtype=np.float32)
        static_diagonal = np.zeros((5, 3, 3), dtype=np.float32)
        static_diagonal[:, 0, 0] = [6.0, 16.0, 26.0, 36.0, 46.0]
        directional_scales = static_diagonal[:, 0, 0] + 4.0
        feature_0_scale = float(np.mean(directional_scales[:2]))
        feature_1_scale = float(np.mean(directional_scales[2:]))
        stiffness = 0.5 * feature_0_scale * feature_1_scale / (feature_0_scale + feature_1_scale)

        with wp.ScopedDevice("cuda:0"):
            contacts = _ContactBuffer(arity=5, feature_split=2, capacity=1, device="cuda:0")
            contacts.ids.assign(np.arange(5, dtype=np.int32).reshape(1, 5))
            contacts.weights.assign(weights.reshape(1, 5))
            contacts.directions.assign(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
            contacts.count.assign(np.asarray([1], dtype=np.int32))
            diagonal_blocks = wp.array(static_diagonal, dtype=wp.mat33, device="cuda:0")
            masses = wp.ones(5, dtype=float, device="cuda:0")
            vector = wp.array([[1.0, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 4, dtype=wp.vec3, device="cuda:0")
            product = wp.zeros(5, dtype=wp.vec3, device="cuda:0")

            contacts.hessian_multiply_adaptive(0.5, diagonal_blocks, masses, 4.0, vector, product)
            product_np = product.numpy()

        expected = np.zeros((5, 3), dtype=np.float32)
        expected[:, 0] = stiffness * weights * weights[0]
        np.testing.assert_allclose(product_np, expected, rtol=2.0e-6, atol=1.0e-7)

    def test_ipc_mollified_edge_force_and_gauss_newton_operator_match_residual_reference(self):
        """Match the IPC-mollified EE force and Gauss-Newton operator."""
        eps_x = 1.0e-3
        sine = np.sqrt(0.5 * eps_x)
        cosine = np.sqrt(1.0 - sine * sine)
        positions = np.asarray(
            [
                [-0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [-0.5 * cosine, -0.5 * sine, 0.05],
                [0.5 * cosine, 0.5 * sine, 0.05],
            ],
            dtype=np.float64,
        )
        weights = np.asarray([0.5, 0.5, -0.5, -0.5], dtype=np.float64)
        direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        depth = 0.05
        stiffness = 7.0
        vector = np.asarray(
            [[0.2, -0.1, 0.4], [-0.3, 0.7, 0.1], [0.5, 0.2, -0.6], [-0.4, -0.2, 0.3]],
            dtype=np.float64,
        )

        def residual(flat_positions):
            current = flat_positions.reshape(4, 3)
            displacement = current - positions
            current_depth = depth - np.sum(weights[:, None] * displacement * direction)
            edge_0 = current[1] - current[0]
            edge_1 = current[3] - current[2]
            cross_product = np.cross(edge_0, edge_1)
            cross_squared = float(np.dot(cross_product, cross_product))
            beta = np.sqrt(2.0 * eps_x - cross_squared) / eps_x
            return current_depth * beta * cross_product

        flat_positions = positions.reshape(-1)
        residual_value = residual(flat_positions)
        self.assertAlmostEqual(float(np.dot(residual_value, residual_value) / (depth * depth)), 0.75, places=10)
        jacobian = np.empty((3, 12), dtype=np.float64)
        epsilon = 1.0e-6
        for column in range(12):
            offset = np.zeros(12, dtype=np.float64)
            offset[column] = epsilon
            jacobian[:, column] = (residual(flat_positions + offset) - residual(flat_positions - offset)) / (
                2.0 * epsilon
            )
        expected_force = (-stiffness * jacobian.T @ residual_value).reshape(4, 3)
        expected_hessian = stiffness * jacobian.T @ jacobian
        expected_hvp = (expected_hessian @ vector.reshape(-1)).reshape(4, 3)
        expected_diagonal = np.asarray([expected_hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] for i in range(4)])

        with wp.ScopedDevice("cuda:0"):
            contacts = _EdgeEdgeContactBuffer(capacity=1, device="cuda:0")
            contacts.ids.assign(np.asarray([[0, 1, 2, 3]], dtype=np.int32))
            contacts.weights.assign(weights.astype(np.float32).reshape(1, 4))
            contacts.directions.assign(direction.astype(np.float32).reshape(1, 3))
            contacts.depths.assign(np.asarray([depth], dtype=np.float32))
            contacts.mollifier_thresholds.assign(np.asarray([eps_x], dtype=np.float32))
            contacts.count.assign(np.asarray([1], dtype=np.int32))
            positions_wp = wp.array(positions.astype(np.float32), dtype=wp.vec3, device="cuda:0")
            vector_wp = wp.array(vector.astype(np.float32), dtype=wp.vec3, device="cuda:0")
            force = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
            hvp = wp.zeros_like(force)
            diagonal = wp.zeros(4, dtype=wp.mat33, device="cuda:0")

            contacts.prepare_hessian(positions_wp)
            contacts.accumulate_force(stiffness, positions_wp, force)
            contacts.hessian_multiply(stiffness, positions_wp, vector_wp, hvp)
            contacts.accumulate_diagonal(stiffness, positions_wp, diagonal)

            force_np = force.numpy()
            hvp_np = hvp.numpy()
            diagonal_np = diagonal.numpy()

        np.testing.assert_allclose(force_np, expected_force, rtol=2.0e-4, atol=2.0e-4)
        np.testing.assert_allclose(hvp_np, expected_hvp, rtol=3.0e-4, atol=3.0e-4)
        np.testing.assert_allclose(diagonal_np, expected_diagonal, rtol=3.0e-4, atol=3.0e-4)
        self.assertGreater(float(vector.reshape(-1) @ expected_hessian @ vector.reshape(-1)), 0.0)

    def test_mollified_edge_edge_friction_uses_reduced_normal_load(self):
        """Scale EE friction by the active near-parallel mollifier."""
        threshold = 1.0e-3
        sine = np.sqrt(0.5 * threshold)
        cosine = np.sqrt(1.0 - sine * sine)
        positions = np.asarray(
            [
                [-0.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [-0.5 * cosine, -0.5 * sine, 0.05],
                [0.5 * cosine, 0.5 * sine, 0.05],
            ],
            dtype=np.float32,
        )
        anchor_positions = positions.copy()
        anchor_positions[2:, 1] -= 0.10
        stiffness = 10.0
        depth = 0.05
        friction = 0.4

        with wp.ScopedDevice("cuda:0"):
            contacts = _EdgeEdgeContactBuffer(capacity=1, device="cuda:0")
            contacts.ids.assign(np.asarray([[0, 1, 2, 3]], dtype=np.int32))
            contacts.weights.assign(np.asarray([[0.5, 0.5, -0.5, -0.5]], dtype=np.float32))
            contacts.directions.assign(np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32))
            contacts.depths.assign(np.asarray([depth], dtype=np.float32))
            contacts.mollifier_thresholds.assign(np.asarray([threshold], dtype=np.float32))
            contacts.count.assign(np.asarray([1], dtype=np.int32))
            positions_wp = wp.array(positions, dtype=wp.vec3, device="cuda:0")
            anchor_positions_wp = wp.array(anchor_positions, dtype=wp.vec3, device="cuda:0")
            force = wp.zeros(4, dtype=wp.vec3, device="cuda:0")

            contacts.prepare_hessian(positions_wp)
            contacts.accumulate_friction_force(
                stiffness,
                friction,
                1.0e-4,
                positions_wp,
                anchor_positions_wp,
                force,
            )
            force_np = force.numpy()
            mollifier_active = int(contacts.mollifier_active.numpy()[0])

        cross_product = np.cross(positions[1] - positions[0], positions[3] - positions[2])
        cross_squared = float(np.dot(cross_product, cross_product))
        load_scale = cross_squared * (2.0 * threshold - cross_squared) / threshold**2
        first_feature_force = force_np[0] + force_np[1]
        expected_limit = friction * stiffness * depth * load_scale

        self.assertEqual(mollifier_active, 1)
        self.assertAlmostEqual(float(np.linalg.norm(first_feature_force)), expected_limit, places=5)
        np.testing.assert_allclose(force_np.sum(axis=0), np.zeros(3), atol=1.0e-6)

    def test_ipc_mollified_parallel_edge_operator_stays_finite_and_psd(self):
        """Keep the mollified EE operator finite and supporting at exact parallelism."""
        positions = np.asarray(
            [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.5, 0.0, 0.05], [0.5, 0.0, 0.05]],
            dtype=np.float32,
        )
        vector = np.asarray(
            [[0.0, 0.2, 0.0], [0.0, -0.2, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]],
            dtype=np.float32,
        )
        with wp.ScopedDevice("cuda:0"):
            contacts = _EdgeEdgeContactBuffer(capacity=1, device="cuda:0")
            contacts.ids.assign(np.asarray([[0, 1, 2, 3]], dtype=np.int32))
            contacts.weights.assign(np.asarray([[0.5, 0.5, -0.5, -0.5]], dtype=np.float32))
            contacts.directions.assign(np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32))
            contacts.depths.assign(np.asarray([0.05], dtype=np.float32))
            contacts.mollifier_thresholds.assign(np.asarray([1.0e-3], dtype=np.float32))
            contacts.count.assign(np.asarray([1], dtype=np.int32))
            positions_wp = wp.array(positions, dtype=wp.vec3, device="cuda:0")
            vector_wp = wp.array(vector, dtype=wp.vec3, device="cuda:0")
            force = wp.zeros(4, dtype=wp.vec3, device="cuda:0")
            hvp = wp.zeros_like(force)

            contacts.prepare_hessian(positions_wp)
            contacts.accumulate_force(7.0, positions_wp, force)
            contacts.hessian_multiply(7.0, positions_wp, vector_wp, hvp)

            force_np = force.numpy()
            hvp_np = hvp.numpy()

        self.assertTrue(np.isfinite(force_np).all())
        self.assertTrue(np.isfinite(hvp_np).all())
        np.testing.assert_allclose(force_np, np.zeros((4, 3)), atol=1.0e-7)
        self.assertGreater(float(np.sum(vector * hvp_np)), 0.0)

    def _assert_contact_matches_dense_reference(self, weights, direction, depth, stiffness):
        particle_count = len(weights)
        ids = np.arange(particle_count, dtype=np.int32)
        vectors = np.asarray(
            [
                [0.2, -0.1, 0.4],
                [-0.3, 0.7, 0.1],
                [0.5, 0.2, -0.6],
                [-0.4, -0.2, 0.3],
                [0.1, 0.6, -0.5],
            ][:particle_count],
            dtype=np.float32,
        )
        dense_direction = np.concatenate([weight * direction for weight in weights])
        dense_hessian = stiffness * np.outer(dense_direction, dense_direction)
        expected_force = (stiffness * depth * dense_direction).reshape(particle_count, 3)
        expected_diagonal = np.asarray(
            [stiffness * weight * weight * np.outer(direction, direction) for weight in weights],
            dtype=np.float32,
        )
        expected_hvp = (dense_hessian @ vectors.reshape(-1)).reshape(particle_count, 3)

        with wp.ScopedDevice("cuda:0"):
            contacts = _ContactBuffer(arity=particle_count, capacity=2, device="cuda:0")
            contact_ids = np.zeros((2, particle_count), dtype=np.int32)
            contact_weights = np.zeros((2, particle_count), dtype=np.float32)
            contact_ids[0] = ids
            contact_weights[0] = weights
            contacts.ids.assign(contact_ids)
            contacts.weights.assign(contact_weights)
            contacts.directions.assign(np.asarray([direction, [0.0, 0.0, 0.0]], dtype=np.float32))
            contacts.depths.assign(np.asarray([depth, 0.0], dtype=np.float32))
            contacts.count.assign(np.asarray([1], dtype=np.int32))

            vector = wp.array(vectors, dtype=wp.vec3, device="cuda:0")
            force = wp.zeros(particle_count, dtype=wp.vec3, device="cuda:0")
            hvp = wp.zeros_like(force)
            diagonal = wp.zeros(particle_count, dtype=wp.mat33, device="cuda:0")
            contacts.accumulate_force(stiffness, force)
            contacts.hessian_multiply(stiffness, vector, hvp)
            contacts.accumulate_diagonal(stiffness, diagonal)

            force_np = force.numpy()
            hvp_np = hvp.numpy()
            diagonal_np = diagonal.numpy()

        np.testing.assert_allclose(force_np, expected_force, rtol=2.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(np.sum(force_np, axis=0), np.zeros(3), atol=1.0e-7)
        np.testing.assert_allclose(hvp_np, expected_hvp, rtol=2.0e-6, atol=1.0e-7)
        np.testing.assert_allclose(diagonal_np, expected_diagonal, rtol=2.0e-6, atol=1.0e-7)
        self.assertGreaterEqual(float(vectors.reshape(-1) @ dense_hessian @ vectors.reshape(-1)), -1.0e-7)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintSelfCollisionDetection(unittest.TestCase):
    @staticmethod
    def _make_model(positions, triangles):
        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_particles(
            pos=np.asarray(positions, dtype=np.float32),
            vel=[wp.vec3(0.0)] * len(positions),
            mass=[1.0] * len(positions),
            radius=[0.01] * len(positions),
        )
        triangles = np.asarray(triangles, dtype=np.int32)
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        return builder.finalize(device="cuda:0")

    @staticmethod
    def _make_mixed_tetrahedral_cloth_model():
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.33, 0.33, 0.338],
                [3.0, 3.0, 3.0],
                [4.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_particles(
            pos=positions,
            vel=[wp.vec3(0.0)] * len(positions),
            mass=[1.0] * len(positions),
            radius=[0.01] * len(positions),
        )
        builder.add_tetrahedron(0, 1, 2, 3)
        triangles = np.asarray(
            [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [4, 5, 6]],
            dtype=np.int32,
        )
        builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
        return builder.finalize(device="cuda:0")

    @staticmethod
    def _stored_contacts(buffer):
        count = min(int(buffer.count.numpy()[0]), buffer.capacity)
        return (
            buffer.ids.numpy()[:count],
            buffer.weights.numpy()[:count],
            buffer.directions.numpy()[:count],
            buffer.depths.numpy()[:count],
        )

    @classmethod
    def _make_two_ring_thickness_model(cls, clearance: float):
        positions = [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 10.0, 10.0],
            [0.0, 0.0, clearance],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
        ]
        return cls._make_model(positions, [(0, 1, 2), (2, 3, 5), (3, 4, 6)])

    def test_automatic_thickness_uses_two_ring_vf_upper_bound(self):
        """Scale the smallest two-ring VF clearance by eta."""
        with wp.ScopedDevice("cuda:0"):
            model = self._make_two_ring_thickness_model(0.004)
            collision = ConstraintSelfCollision(model, thickness=None, stiffness=10.0)

        self.assertTrue(collision.thickness_was_estimated)
        self.assertAlmostEqual(collision.thickness, 0.0032, places=7)

    def test_automatic_thickness_caps_at_five_millimeters(self):
        """Cap an automatically estimated collision thickness at 5 mm."""
        with wp.ScopedDevice("cuda:0"):
            model = self._make_two_ring_thickness_model(0.1)
            collision = ConstraintSelfCollision(model, thickness=None, stiffness=10.0)

        self.assertTrue(collision.thickness_was_estimated)
        self.assertAlmostEqual(collision.thickness, 0.005, places=7)

    def test_automatic_thickness_rejects_zero_two_ring_clearance(self):
        """Reject automatic thickness when a valid two-ring pair has zero clearance."""
        with wp.ScopedDevice("cuda:0"):
            model = self._make_two_ring_thickness_model(0.0)
            with self.assertRaisesRegex(ValueError, "two-ring"):
                ConstraintSelfCollision(model, thickness=None, stiffness=10.0)

    def test_explicit_thickness_bypasses_geometry_estimate(self):
        """Preserve an explicitly configured collision thickness."""
        with wp.ScopedDevice("cuda:0"):
            model = self._make_two_ring_thickness_model(0.004)
            collision = ConstraintSelfCollision(model, thickness=0.0025, stiffness=10.0)

        self.assertFalse(collision.thickness_was_estimated)
        self.assertAlmostEqual(collision.thickness, 0.0025, places=7)

    def test_vertex_face_detection_uses_only_surface_vertices(self):
        """Exclude particles outside the triangle topology from VF candidates."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.05],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0)
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.vertex_face_contacts)

        self.assertEqual(collision.surface_vertex_count, 3)
        np.testing.assert_array_equal(collision.surface_vertex_indices.numpy(), [0, 1, 2])
        self.assertFalse(np.any(ids[:, 0] == 3))

    def test_vertex_face_detection_keeps_closest_edge_feature(self):
        """Keep a VF contact whose closest triangle point lies on an edge."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, -0.04, 0.02],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0)
            collision.prepare(model.particle_q)
            ids, weights, _, _ = self._stored_contacts(collision.vertex_face_contacts)

        matches = np.flatnonzero(np.all(ids == [3, 0, 1, 2], axis=1))
        self.assertEqual(len(matches), 1)
        np.testing.assert_allclose(weights[matches[0]], [1.0, -0.5, -0.5, 0.0], atol=1.0e-6)

    def test_friction_parameters_validate_and_default_to_disabled(self):
        """Validate friction parameters and preserve a frictionless default."""
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            default_collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0)
            friction_collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                friction=0.4,
                friction_epsilon=1.0e-2,
            )
            invalid_cases = (
                ({"friction": -0.1}, "nonnegative"),
                ({"friction": np.inf}, "finite"),
                ({"friction": np.nan}, "finite"),
                ({"friction_epsilon": 0.0}, "positive"),
                ({"friction_epsilon": -1.0}, "positive"),
                ({"friction_epsilon": np.inf}, "finite"),
            )
            for kwargs, message in invalid_cases:
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, message):
                        ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, **kwargs)

        self.assertEqual(default_collision.friction, 0.0)
        self.assertEqual(default_collision.friction_epsilon, 1.0e-2)
        self.assertIsNone(default_collision._friction_positions)
        self.assertEqual(friction_collision.friction, 0.4)
        self.assertIsNotNone(friction_collision._friction_positions)

    def test_geometry_radius_scale_validates_and_uniform_default_stays_available(self):
        """Validate the radius scale and expose uniform legacy radii by default."""
        positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0)
            for scale, message in ((0.0, "positive"), (-0.1, "positive"), (np.inf, "finite"), (np.nan, "finite")):
                with self.subTest(scale=scale):
                    with self.assertRaisesRegex(ValueError, message):
                        ConstraintSelfCollision(
                            model,
                            thickness=0.1,
                            stiffness=10.0,
                            geometry_radius_scale=scale,
                        )
            radii = collision.particle_radii.numpy()

        self.assertIsNone(collision.geometry_radius_scale)
        np.testing.assert_allclose(radii, np.full(3, 0.05, dtype=np.float32), rtol=0.0, atol=1.0e-7)

    def test_topology_local_geometry_radii_require_radius_scale(self):
        """Require geometry radii when selecting the topology-local scope."""
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            with self.assertRaisesRegex(ValueError, "geometry_radius_scale"):
                ConstraintSelfCollision(
                    model,
                    thickness=0.1,
                    stiffness=10.0,
                    geometry_radius_topology_local_only=True,
                )

    def test_geometry_aware_radii_use_minimum_incident_triangle_altitude(self):
        """Cap each particle radius using its smallest incident rest altitude."""
        positions = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.25, 0.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (1, 3, 2)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.6,
                stiffness=10.0,
                geometry_radius_scale=0.5,
            )
            radii = collision.particle_radii.numpy()

        expected_small_radius = 0.25 / np.sqrt(5.0)
        expected = np.asarray([0.3, expected_small_radius, expected_small_radius, expected_small_radius])
        self.assertEqual(collision.geometry_radius_scale, 0.5)
        np.testing.assert_allclose(radii, expected, rtol=1.0e-6, atol=1.0e-7)

    def test_geometry_aware_radii_ignore_unreferenced_interior_particles(self):
        """Assign zero radius to particles absent from the collision surface."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.25],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            try:
                collision = ConstraintSelfCollision(
                    model,
                    thickness=0.1,
                    stiffness=10.0,
                    geometry_radius_scale=0.25,
                )
            except ValueError as error:
                self.fail(f"Geometry-aware radii rejected an interior particle: {error}")
            radii = collision.particle_radii.numpy()

        np.testing.assert_allclose(radii, [0.05, 0.05, 0.05, 0.0], rtol=0.0, atol=1.0e-7)

    def test_geometry_aware_radii_reject_invalid_rest_geometry(self):
        """Reject non-finite and degenerate rest surface geometry."""
        valid_positions = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        with wp.ScopedDevice("cuda:0"):
            nonfinite_model = self._make_model(valid_positions, [(0, 1, 2)])
            nonfinite_positions = valid_positions.copy()
            nonfinite_positions[1, 0] = np.nan
            nonfinite_model.particle_q.assign(nonfinite_positions)
            with self.assertRaisesRegex(ValueError, "finite"):
                ConstraintSelfCollision(
                    nonfinite_model,
                    thickness=0.1,
                    stiffness=10.0,
                    geometry_radius_scale=0.25,
                )

            degenerate_model = self._make_model(valid_positions, [(0, 1, 2)])
            degenerate_model.particle_q.assign(
                np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
            )
            with self.assertRaisesRegex(ValueError, "degenerate"):
                ConstraintSelfCollision(
                    degenerate_model,
                    thickness=0.1,
                    stiffness=10.0,
                    geometry_radius_scale=0.25,
                )

    def test_untangle_stiffness_defaults_to_three_times_contact_stiffness(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0)

        self.assertEqual(collision.stiffness, 10.0)
        self.assertEqual(collision.untangle_stiffness, 30.0)

    def test_explicit_untangle_stiffness_overrides_default_ratio(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                untangle_stiffness=17.0,
            )

        self.assertEqual(collision.untangle_stiffness, 17.0)

    def test_adaptive_stiffness_mode_accepts_three_positive_feature_factors(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=None,
                stiffness_factors=(0.5, 0.1, 1.5),
            )

        self.assertIsNone(collision.stiffness)
        self.assertIsNone(collision.untangle_stiffness)
        self.assertEqual(collision.stiffness_factors, (0.5, 0.1, 1.5))

    def test_rejects_ambiguous_or_invalid_stiffness_modes(self):
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        cases = [
            ({"stiffness": None}, "stiffness_factors"),
            ({"stiffness": 10.0, "stiffness_factors": (0.5, 0.1, 1.5)}, "either"),
            (
                {"stiffness": None, "untangle_stiffness": 10.0, "stiffness_factors": (0.5, 0.1, 1.5)},
                "untangle_stiffness",
            ),
            ({"stiffness": None, "stiffness_factors": (0.5, 0.1)}, "three"),
            ({"stiffness": None, "stiffness_factors": (0.5, 0.0, 1.5)}, "positive"),
            ({"stiffness": None, "stiffness_factors": (0.5, np.nan, 1.5)}, "finite"),
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2)])
            for kwargs, message in cases:
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, message):
                        ConstraintSelfCollision(model, thickness=0.1, **kwargs)

    def test_vertex_face_detection_emits_signed_barycentric_contact(self):
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.05],
            [3.0, 3.0, 3.0],
            [4.0, 3.0, 3.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, max_contacts=16)
            collision.prepare(model.particle_q)
            ids, weights, directions, depths = self._stored_contacts(collision.vertex_face_contacts)

        matches = np.nonzero(np.all(ids == [3, 0, 1, 2], axis=1))[0]
        self.assertEqual(len(matches), 1)
        contact = int(matches[0])
        np.testing.assert_allclose(weights[contact], [1.0, -0.5, -0.25, -0.25], atol=1.0e-6)
        np.testing.assert_allclose(np.sum(weights[contact]), 0.0, atol=1.0e-7)
        np.testing.assert_allclose(directions[contact], [0.0, 0.0, 1.0], atol=1.0e-6)
        self.assertAlmostEqual(float(depths[contact]), 0.05, places=6)

    def test_oriented_vertex_face_uses_outward_normal_after_crossing(self):
        """Push an inside vertex along the target face's outward normal."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, -0.05],
            [3.0, 3.0, -0.05],
            [4.0, 3.0, -0.05],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=16,
                use_outward_normals=True,
            )
            collision.prepare(model.particle_q)
            ids, weights, directions, depths = self._stored_contacts(collision.vertex_face_contacts)
            force = wp.zeros_like(model.particle_q)
            collision.vertex_face_contacts.accumulate_force(10.0, force)
            force_np = force.numpy()

        matches = np.nonzero(np.all(ids == [3, 0, 1, 2], axis=1))[0]
        self.assertEqual(len(matches), 1)
        contact = int(matches[0])
        np.testing.assert_allclose(weights[contact], [1.0, -0.5, -0.25, -0.25], atol=1.0e-6)
        np.testing.assert_allclose(directions[contact], [0.0, 0.0, 1.0], atol=1.0e-6)
        self.assertAlmostEqual(float(depths[contact]), 0.15, places=6)
        self.assertGreater(float(force_np[3, 2]), 0.0)

    def test_tetrahedral_face_is_outward_in_mixed_unsigned_mesh(self):
        """Orient tetrahedral VF while retaining the unsigned mode for mixed cloth."""
        with wp.ScopedDevice("cuda:0"):
            model = self._make_mixed_tetrahedral_cloth_model()
            collision = ConstraintSelfCollision(
                model,
                thickness=0.01,
                stiffness=10.0,
                max_contacts=64,
                use_outward_normals=False,
            )
            collision.prepare(model.particle_q)
            ids, _weights, directions, depths = self._stored_contacts(collision.vertex_face_contacts)

        matches = np.nonzero(np.all(ids == [4, 1, 2, 3], axis=1))[0]
        self.assertEqual(len(matches), 1)
        contact = int(matches[0])
        expected_direction = np.ones(3, dtype=np.float64) / np.sqrt(3.0)
        signed_distance = (0.33 + 0.33 + 0.338 - 1.0) / np.sqrt(3.0)
        np.testing.assert_allclose(directions[contact], expected_direction, atol=5.0e-5)
        self.assertAlmostEqual(float(depths[contact]), 0.01 - signed_distance, places=6)

    def test_oriented_vertex_face_retains_nonincident_one_ring_neighbor(self):
        """Keep a nonincident one-ring vertex eligible for oriented VF contact."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.05],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (0, 1, 3)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=16,
                use_outward_normals=True,
            )
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.vertex_face_contacts)

        self.assertTrue(np.any(np.all(ids == [3, 0, 1, 2], axis=1)))

    def test_topology_local_geometry_radii_limit_only_one_ring_vertex_face(self):
        """Limit one-ring VF while retaining uniform-thickness nonlocal VF."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.05],
            [0.60, 0.20, 0.05],
            [3.0, 3.0, 2.0],
            [4.0, 3.0, 2.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (0, 1, 3), (4, 5, 6)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                geometry_radius_scale=0.25,
                geometry_radius_topology_local_only=True,
                use_outward_normals=True,
            )
            collision.particle_radii.assign([0.01] * len(positions))
            collision.prepare(model.particle_q)
            ids, _, _, depths = self._stored_contacts(collision.vertex_face_contacts)

        local_matches = np.nonzero(np.all(ids == [3, 0, 1, 2], axis=1))[0]
        nonlocal_matches = np.nonzero(np.all(ids == [4, 0, 1, 2], axis=1))[0]
        self.assertEqual(len(local_matches), 0)
        self.assertEqual(len(nonlocal_matches), 1)
        self.assertAlmostEqual(float(depths[int(nonlocal_matches[0])]), 0.05, places=6)

    def test_oriented_vertex_face_retains_two_ring_neighbor(self):
        """Keep two-ring vertices eligible for oriented VF contacts."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.25, 0.25, 0.05],
            [3.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (0, 3, 6), (3, 5, 8)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                use_outward_normals=True,
            )
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.vertex_face_contacts)

        self.assertTrue(np.any(np.all(ids == [5, 0, 1, 2], axis=1)))

    def test_oriented_vertex_face_excludes_points_beyond_detection_band(self):
        """Exclude deep inside vertices beyond the discrete VF detection band."""
        normal = np.asarray([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3.0)
        positions = [
            [1.0, -1.0, 0.0],
            [0.0, 1.0, -1.0],
            [-1.0, 0.0, 1.0],
            -0.15 * normal,
            [3.0, 3.0, 3.0],
            [4.0, 3.0, 3.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=16,
                use_outward_normals=True,
            )
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.vertex_face_contacts)

        self.assertFalse(np.any(np.all(ids == [3, 0, 1, 2], axis=1)))

    def test_vertex_face_friction_opposes_slip_and_adds_psd_operator(self):
        """Oppose VF slip with balanced friction and a PSD tangent operator."""
        current = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, 0.05],
                [3.0, 3.0, 3.0],
                [4.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        anchor = current.copy()
        anchor[3, 0] -= 0.10
        displacement = current - anchor
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(current, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=1.0e3,
                friction=0.4,
                friction_epsilon=1.0e-2,
                max_contacts=32,
            )
            current_wp = wp.array(current, dtype=wp.vec3, device="cuda:0")
            anchor_wp = wp.array(anchor, dtype=wp.vec3, device="cuda:0")
            velocities = wp.zeros(model.particle_count, dtype=wp.vec3, device="cuda:0")
            collision.begin_step(anchor_wp, velocities, 0.01)
            collision.prepare(current_wp)
            force = wp.zeros(model.particle_count, dtype=wp.vec3, device="cuda:0")
            collision.accumulate_force(current_wp, force)
            product = wp.zeros_like(force)
            displacement_wp = wp.array(displacement, dtype=wp.vec3, device="cuda:0")
            collision.hessian_multiply(current_wp, displacement_wp, product)
            diagonal = wp.zeros(model.particle_count, dtype=wp.mat33, device="cuda:0")
            collision.accumulate_diagonal(current_wp, diagonal)
            force_np = force.numpy()
            product_np = product.numpy()
            diagonal_np = diagonal.numpy()

        self.assertLess(float(np.sum(force_np * displacement)), 0.0)
        np.testing.assert_allclose(force_np.sum(axis=0), 0.0, atol=1.0e-5)
        self.assertLessEqual(abs(float(force_np[3, 0])), 0.4 * 1.0e3 * 0.05 + 1.0e-4)
        self.assertGreaterEqual(float(np.sum(displacement * product_np)), -1.0e-5)
        for block in diagonal_np:
            self.assertGreaterEqual(float(np.linalg.eigvalsh(block).min()), -1.0e-4)

    def test_adaptive_vertex_face_friction_remains_finite_and_psd(self):
        """Keep adaptive VF friction balanced, finite, and positive semidefinite."""
        current = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, 0.05],
                [3.0, 3.0, 3.0],
                [4.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        anchor = current.copy()
        anchor[3, 0] -= 0.10
        displacement = current - anchor
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(current, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=None,
                stiffness_factors=(0.5, 0.3, 1.5),
                friction=0.4,
                friction_epsilon=1.0e-2,
                max_contacts=32,
            )
            static_diagonal = wp.array(
                np.broadcast_to(100.0 * np.eye(3, dtype=np.float32), (model.particle_count, 3, 3)).copy(),
                dtype=wp.mat33,
                device="cuda:0",
            )
            current_wp = wp.array(current, dtype=wp.vec3, device="cuda:0")
            anchor_wp = wp.array(anchor, dtype=wp.vec3, device="cuda:0")
            velocities = wp.zeros(model.particle_count, dtype=wp.vec3, device="cuda:0")
            collision.bind_static_system(static_diagonal, model.particle_mass)
            collision.begin_step(anchor_wp, velocities, 0.01)
            collision.prepare(current_wp)
            force = wp.zeros(model.particle_count, dtype=wp.vec3, device="cuda:0")
            collision.accumulate_force(current_wp, force)
            product = wp.zeros_like(force)
            displacement_wp = wp.array(displacement, dtype=wp.vec3, device="cuda:0")
            collision.hessian_multiply(current_wp, displacement_wp, product)
            diagonal = wp.zeros(model.particle_count, dtype=wp.mat33, device="cuda:0")
            collision.accumulate_diagonal(current_wp, diagonal)
            force_np = force.numpy()
            product_np = product.numpy()
            diagonal_np = diagonal.numpy()

        self.assertTrue(np.isfinite(force_np).all())
        self.assertLess(float(np.sum(force_np * displacement)), 0.0)
        np.testing.assert_allclose(force_np.sum(axis=0), 0.0, atol=1.0e-4)
        self.assertTrue(np.isfinite(product_np).all())
        self.assertGreaterEqual(float(np.sum(displacement * product_np)), -1.0e-4)
        for block in diagonal_np:
            self.assertGreaterEqual(float(np.linalg.eigvalsh(block).min()), -1.0e-3)

    def test_geometry_aware_vertex_face_depth_interpolates_face_radii(self):
        """Compute VF depth from vertex and barycentrically interpolated face radii."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.05],
            [3.0, 3.0, 3.0],
            [4.0, 3.0, 3.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=16,
                geometry_radius_scale=0.25,
            )
            collision.particle_radii.assign([0.01, 0.02, 0.03, 0.04, 0.01, 0.01])
            collision.prepare(model.particle_q)
            ids, _, _, depths = self._stored_contacts(collision.vertex_face_contacts)

        matches = np.nonzero(np.all(ids == [3, 0, 1, 2], axis=1))[0]
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(float(depths[int(matches[0])]), 0.0075, places=6)

    def test_edge_edge_detection_uses_distinct_closest_parameters(self):
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.25, -0.3, 0.05],
            [0.25, 0.7, 0.05],
            [3.0, 0.7, 2.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, max_contacts=32)
            collision.prepare(model.particle_q)
            ids, weights, directions, depths = self._stored_contacts(collision.edge_edge_contacts)

        matches = np.nonzero(np.all(ids == [0, 1, 3, 4], axis=1))[0]
        self.assertEqual(len(matches), 1)
        contact = int(matches[0])
        np.testing.assert_allclose(weights[contact], [0.75, 0.25, -0.7, -0.3], atol=1.0e-6)
        np.testing.assert_allclose(np.sum(weights[contact]), 0.0, atol=1.0e-7)
        np.testing.assert_allclose(directions[contact], [0.0, 0.0, -1.0], atol=1.0e-6)
        self.assertAlmostEqual(float(depths[contact]), 0.05, places=6)

    def test_edge_edge_detection_excludes_endpoint_features(self):
        """Delegate endpoint PE and PP features to vertex-face contact."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [1.04, 0.03, 0.02],
            [1.04, 1.0, 0.02],
            [3.0, 1.0, 0.02],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0)
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.edge_edge_contacts)

        matches = [row for row in ids if {int(row[0]), int(row[1])} == {0, 1} and {int(row[2]), int(row[3])} == {3, 4}]
        self.assertEqual(len(matches), 0)

    def test_oriented_edge_edge_uses_incident_face_pseudo_normals_after_crossing(self):
        """Orient a crossed EE contact from its incident outward face normals."""
        positions = [
            [-1.0, 0.0, -0.01],
            [1.0, 0.0, -0.01],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, -0.01],
            [1.0, 0.0, 0.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(1, 0, 4), (3, 2, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                use_outward_normals=True,
            )
            collision.prepare(model.particle_q)
            ids, _, directions, depths = self._stored_contacts(collision.edge_edge_contacts)

        matches = [
            contact
            for contact, contact_ids in enumerate(ids)
            if (
                {int(contact_ids[0]), int(contact_ids[1])} == {0, 1}
                and {int(contact_ids[2]), int(contact_ids[3])} == {2, 3}
            )
            or (
                {int(contact_ids[0]), int(contact_ids[1])} == {2, 3}
                and {int(contact_ids[2]), int(contact_ids[3])} == {0, 1}
            )
        ]
        self.assertEqual(len(matches), 1)
        contact = matches[0]
        first_edge_is_lower = {int(ids[contact, 0]), int(ids[contact, 1])} == {0, 1}
        expected_direction = [0.0, 0.0, 1.0 if first_edge_is_lower else -1.0]
        np.testing.assert_allclose(directions[contact], expected_direction, atol=1.0e-6)
        self.assertAlmostEqual(float(depths[contact]), 0.11, places=6)

    def test_edge_edge_friction_opposes_relative_slip(self):
        """Oppose EE slip with balanced friction and a PSD tangent operator."""
        current = np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.1, 0.0],
                [0.0, -1.0, 0.05],
                [0.0, 1.0, 0.05],
                [0.1, 0.5, 0.05],
            ],
            dtype=np.float32,
        )
        anchor = current.copy()
        anchor[3:, 0] -= 0.10
        displacement = current - anchor
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(current, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=1.0e3,
                friction=0.4,
                friction_epsilon=1.0e-2,
                max_contacts=32,
            )
            current_wp = wp.array(current, dtype=wp.vec3, device="cuda:0")
            anchor_wp = wp.array(anchor, dtype=wp.vec3, device="cuda:0")
            velocities = wp.zeros(model.particle_count, dtype=wp.vec3, device="cuda:0")
            collision.begin_step(anchor_wp, velocities, 0.01)
            collision.prepare(current_wp)
            self.assertEqual(int(collision.vertex_face_contacts.count.numpy()[0]), 0)
            self.assertGreater(int(collision.edge_edge_contacts.count.numpy()[0]), 0)
            force = wp.zeros(model.particle_count, dtype=wp.vec3, device="cuda:0")
            collision.accumulate_force(current_wp, force)
            product = wp.zeros_like(force)
            displacement_wp = wp.array(displacement, dtype=wp.vec3, device="cuda:0")
            collision.hessian_multiply(current_wp, displacement_wp, product)
            diagonal = wp.zeros(model.particle_count, dtype=wp.mat33, device="cuda:0")
            collision.accumulate_diagonal(current_wp, diagonal)
            force_np = force.numpy()
            product_np = product.numpy()
            diagonal_np = diagonal.numpy()

            baseline = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=1.0e3,
                max_contacts=32,
            )
            baseline.prepare(current_wp)
            baseline_force = wp.zeros_like(force)
            baseline.accumulate_force(current_wp, baseline_force)
            baseline_product = wp.zeros_like(force)
            baseline.hessian_multiply(current_wp, displacement_wp, baseline_product)
            baseline_diagonal = wp.zeros_like(diagonal)
            baseline.accumulate_diagonal(current_wp, baseline_diagonal)
            friction_force_np = force_np - baseline_force.numpy()
            friction_product_np = product_np - baseline_product.numpy()
            friction_diagonal_np = diagonal_np - baseline_diagonal.numpy()

        self.assertLess(float(np.sum(friction_force_np * displacement)), 0.0)
        np.testing.assert_allclose(friction_force_np.sum(axis=0), 0.0, atol=1.0e-4)
        self.assertGreaterEqual(float(np.sum(displacement * friction_product_np)), -1.0e-4)
        for block in friction_diagonal_np:
            self.assertGreaterEqual(float(np.linalg.eigvalsh(block).min()), -1.0e-3)

    def test_geometry_aware_edge_edge_depth_interpolates_both_edges(self):
        """Compute EE depth from independent closest-point radius interpolation."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.25, -0.3, 0.05],
            [0.25, 0.7, 0.05],
            [3.0, 0.7, 2.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                geometry_radius_scale=0.25,
            )
            collision.particle_radii.assign([0.02, 0.04, 0.01, 0.03, 0.05, 0.01])
            collision.prepare(model.particle_q)
            ids, weights, _, depths = self._stored_contacts(collision.edge_edge_contacts)

        matches = np.nonzero(np.all(ids == [0, 1, 3, 4], axis=1))[0]
        self.assertEqual(len(matches), 1)
        contact = int(matches[0])
        np.testing.assert_allclose(weights[contact], [0.75, 0.25, -0.7, -0.3], atol=1.0e-6)
        self.assertAlmostEqual(float(depths[contact]), 0.011, places=6)

    def test_shared_edge_endpoints_do_not_generate_edge_edge_contacts(self):
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (1, 3, 2)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, max_contacts=32)
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.edge_edge_contacts)

        for contact_ids in ids:
            self.assertEqual(len(set(contact_ids[:2]).intersection(contact_ids[2:])), 0)

    def test_adjacent_opposite_edges_limit_contact_thickness(self):
        """Limit one-ring EE thickness using the local edge lengths."""
        positions = [
            [-0.05, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, -0.05, 0.08],
            [0.0, 0.05, 0.08],
            [1.0, 0.05, 0.08],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (2, 3, 4)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, max_contacts=32)
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.edge_edge_contacts)

        self.assertFalse(np.any(np.all(ids == [0, 1, 2, 3], axis=1)))

    def test_geometry_aware_local_edge_pair_bypasses_length_clamp_and_separates(self):
        """Keep a close local EE contact active with a finite separating force."""
        positions = [
            [-0.05, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, -0.05, 0.08],
            [0.0, 0.05, 0.08],
            [1.0, 0.05, 0.08],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (2, 3, 4)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                geometry_radius_scale=0.25,
            )
            collision.particle_radii.assign([0.05] * len(positions))
            collision.prepare(model.particle_q)
            ids, _, _, depths = self._stored_contacts(collision.edge_edge_contacts)
            force = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
            collision.edge_edge_contacts.accumulate_force(10.0, model.particle_q, force)
            force_np = force.numpy()

        matches = np.nonzero(np.all(ids == [0, 1, 2, 3], axis=1))[0]
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(float(depths[int(matches[0])]), 0.02, places=6)
        self.assertTrue(np.isfinite(force_np).all())
        self.assertGreater(float(np.linalg.norm(force_np)), 0.0)

    def test_topology_local_geometry_radii_limit_only_one_ring_edge_edge(self):
        """Limit one-ring EE while retaining uniform-thickness nonlocal EE."""
        positions = [
            [-0.05, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [0.0, -0.05, 0.02],
            [0.0, 0.05, 0.02],
            [1.0, 0.05, 0.02],
            [0.0, -1.0, 1.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            local_model = self._make_model(positions, [(0, 1, 2), (2, 3, 4)])
            local_collision = ConstraintSelfCollision(
                local_model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                geometry_radius_scale=0.25,
                geometry_radius_topology_local_only=True,
            )
            local_collision.particle_radii.assign([0.005] * len(positions))
            local_collision.prepare(local_model.particle_q)
            local_ids, _, _, _ = self._stored_contacts(local_collision.edge_edge_contacts)

            nonlocal_model = self._make_model(positions, [(0, 1, 4), (2, 3, 5)])
            nonlocal_collision = ConstraintSelfCollision(
                nonlocal_model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                geometry_radius_scale=0.25,
                geometry_radius_topology_local_only=True,
            )
            nonlocal_collision.particle_radii.assign([0.005] * len(positions))
            nonlocal_collision.prepare(nonlocal_model.particle_q)
            nonlocal_ids, _, _, nonlocal_depths = self._stored_contacts(nonlocal_collision.edge_edge_contacts)

        local_matches = np.nonzero(np.all(local_ids == [0, 1, 2, 3], axis=1))[0]
        nonlocal_matches = np.nonzero(np.all(nonlocal_ids == [0, 1, 2, 3], axis=1))[0]
        self.assertEqual(len(local_matches), 0)
        self.assertEqual(len(nonlocal_matches), 1)
        self.assertAlmostEqual(float(nonlocal_depths[int(nonlocal_matches[0])]), 0.08, places=6)

    def test_nonlocal_edge_pair_skips_rest_length_mollifier(self):
        """Keep nonlocal EE penalty active when the two edges are nearly parallel."""
        sine = 0.2
        direction = np.asarray([np.sqrt(1.0 - sine * sine), sine, 0.0], dtype=np.float32)
        center = np.asarray([0.0, 0.0, 0.02], dtype=np.float32)
        rest_positions = np.asarray(
            [
                [-0.05, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                *(center + offset * 0.04 * direction for offset in (-1.0, 1.0)),
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.5],
            ],
            dtype=np.float32,
        )
        current_positions = rest_positions.copy()
        current_positions[0] = [-0.1, 0.0, 0.0]
        current_positions[1] = [0.1, 0.0, 0.0]
        current_positions[2] = center - 0.08 * direction
        current_positions[3] = center + 0.08 * direction

        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(rest_positions, [(0, 1, 4), (2, 3, 5)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, max_contacts=32)
            current_positions_wp = wp.array(current_positions, dtype=wp.vec3, device="cuda:0")
            collision.prepare(current_positions_wp)
            ids, _, _, _ = self._stored_contacts(collision.edge_edge_contacts)
            thresholds = collision.edge_edge_contacts.mollifier_thresholds.numpy()

        matches = np.nonzero(np.all(ids == [0, 1, 2, 3], axis=1))[0]
        self.assertEqual(len(matches), 1)
        self.assertEqual(float(thresholds[int(matches[0])]), 0.0)

    def test_topology_local_edge_pair_uses_half_length_penalty_and_mollifier(self):
        """Combine local EE thickness clamping with the rest-length mollifier."""
        sine = 0.2
        direction = np.asarray([np.sqrt(1.0 - sine * sine), sine, 0.0], dtype=np.float32)
        center = np.asarray([0.0, 0.0, 0.02], dtype=np.float32)
        positions = [
            [-0.05, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            *(center + offset * 0.04 * direction for offset in (-1.0, 1.0)),
            [0.0, 1.0, 0.5],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (2, 3, 4)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, max_contacts=32)
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.edge_edge_contacts)
            thresholds = collision.edge_edge_contacts.mollifier_thresholds.numpy()

        matches = np.nonzero(np.all(ids == [0, 1, 2, 3], axis=1))[0]
        self.assertEqual(len(matches), 1)
        expected = 1.0e-3 * 0.1**2 * 0.08**2
        self.assertAlmostEqual(float(thresholds[int(matches[0])]), expected, places=12)

    def test_oriented_edge_edge_retains_nonincident_one_ring_pair(self):
        """Keep nonincident one-ring edges eligible for oriented EE contact."""
        sine = 0.2
        direction = np.asarray([np.sqrt(1.0 - sine * sine), sine, 0.0], dtype=np.float32)
        center = np.asarray([0.0, 0.0, 0.02], dtype=np.float32)
        positions = [
            [-0.05, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            *(center + offset * 0.04 * direction for offset in (-1.0, 1.0)),
            [0.0, 1.0, 0.5],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (2, 3, 4)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                use_outward_normals=True,
            )
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.edge_edge_contacts)

        self.assertTrue(np.any(np.all(ids == [0, 1, 2, 3], axis=1)))

    def test_oriented_edge_edge_retains_two_ring_pair(self):
        """Keep two-ring edge pairs eligible for oriented EE contacts."""
        positions = [
            [-1.0, 0.0, -0.01],
            [1.0, 0.0, -0.01],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, -0.01],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [4.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
        ]
        triangles = [(1, 0, 4), (3, 2, 5), (0, 6, 8), (6, 2, 10)]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, triangles)
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=64,
                use_outward_normals=True,
            )
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.edge_edge_contacts)

        central_pair_found = any(
            (
                {int(contact_ids[0]), int(contact_ids[1])} == {0, 1}
                and {int(contact_ids[2]), int(contact_ids[3])} == {2, 3}
            )
            or (
                {int(contact_ids[0]), int(contact_ids[1])} == {2, 3}
                and {int(contact_ids[2]), int(contact_ids[3])} == {0, 1}
            )
            for contact_ids in ids
        )
        self.assertTrue(central_pair_found)

    def test_edge_face_crossing_emits_five_particle_untangle_contact(self):
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, -0.5],
            [0.25, 0.25, 0.5],
            [0.75, 0.25, 0.5],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, max_contacts=32)
            collision.prepare(model.particle_q)
            ids, weights, directions, depths = self._stored_contacts(collision.edge_face_contacts)

        matches = np.nonzero(np.all(ids == [3, 4, 0, 1, 2], axis=1))[0]
        self.assertEqual(len(matches), 1)
        contact = int(matches[0])
        np.testing.assert_allclose(weights[contact], [0.5, 0.5, -0.5, -0.25, -0.25], atol=1.0e-6)
        np.testing.assert_allclose(np.sum(weights[contact]), 0.0, atol=1.0e-7)
        np.testing.assert_allclose(directions[contact], [1.0, 0.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(np.linalg.norm(directions[contact]), 1.0, atol=1.0e-6)
        self.assertAlmostEqual(float(depths[contact]), 0.2, places=6)

    def test_edge_face_detection_and_assembly_can_be_disabled(self):
        """Disable EF detection, force, Hessian product, and diagonal assembly."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, -0.5],
            [0.25, 0.25, 0.5],
            [0.75, 0.25, 0.5],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(
                model,
                thickness=0.1,
                stiffness=10.0,
                max_contacts=32,
                enable_edge_face=False,
            )
            collision.prepare(model.particle_q)
            force = wp.zeros_like(model.particle_q)
            product = wp.zeros_like(model.particle_q)
            diagonal = wp.zeros(model.particle_count, dtype=wp.mat33, device=model.device)
            direction = wp.array([wp.vec3(1.0)] * model.particle_count, dtype=wp.vec3, device=model.device)
            collision.accumulate_force(model.particle_q, force)
            collision.hessian_multiply(model.particle_q, direction, product)
            collision.accumulate_diagonal(model.particle_q, diagonal)

        self.assertEqual(int(collision.edge_face_contacts.count.numpy()[0]), 0)
        np.testing.assert_array_equal(force.numpy(), np.zeros((model.particle_count, 3)))
        np.testing.assert_array_equal(product.numpy(), np.zeros((model.particle_count, 3)))
        np.testing.assert_array_equal(diagonal.numpy(), np.zeros((model.particle_count, 3, 3)))

    def test_small_triangle_vertex_face_contact_is_not_treated_as_degenerate(self):
        positions = [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0025, 0.0025, 0.001],
            [1.0, 1.0, 1.0],
            [1.1, 1.0, 1.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            collision = ConstraintSelfCollision(model, thickness=0.005, stiffness=10.0, max_contacts=16)
            collision.prepare(model.particle_q)
            ids, _, _, _ = self._stored_contacts(collision.vertex_face_contacts)

        self.assertTrue(np.any(np.all(ids == [3, 0, 1, 2], axis=1)))

    def test_contact_overflow_is_counted_without_writing_past_capacity(self):
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.2, 0.05],
            [3.0, 3.0, 3.0],
            [4.0, 3.0, 3.0],
            [0.6, 0.2, 0.05],
            [5.0, 5.0, 4.0],
            [6.0, 5.0, 4.0],
        ]
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5), (6, 7, 8)])
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=10.0, max_contacts=1)
            collision.prepare(model.particle_q)
            attempted = int(collision.vertex_face_contacts.count.numpy()[0])
            overflow = int(collision.vertex_face_contacts.overflow_count.numpy()[0])
            stored_ids = collision.vertex_face_contacts.ids.numpy()
            stored_weights = collision.vertex_face_contacts.weights.numpy()

        self.assertGreaterEqual(attempted, 2)
        self.assertEqual(overflow, attempted - 1)
        self.assertEqual(stored_ids.shape, (1, 4))
        self.assertTrue(np.isfinite(stored_weights).all())

    def test_newton_step_increases_vertex_face_separation(self):
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.25, 0.25, 0.05],
                [3.0, 3.0, 3.0],
                [4.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        with wp.ScopedDevice("cuda:0"):
            model = self._make_model(positions, [(0, 1, 2), (3, 4, 5)])
            model.set_gravity((0.0, 0.0, 0.0))
            collision = ConstraintSelfCollision(model, thickness=0.1, stiffness=1.0e3, max_contacts=32)
            solver = SolverLIMX(
                model,
                [],
                nonlinear_iterations=1,
                linear_iterations=50,
                dynamic_operator=collision,
            )
            state_out = model.state()
            solver.step(model.state(), state_out, None, None, 0.01)
            result = state_out.particle_q.numpy()

        face_normal = np.cross(result[1] - result[0], result[2] - result[0])
        face_normal /= np.linalg.norm(face_normal)
        final_distance = abs(float(np.dot(result[3] - result[0], face_normal)))
        self.assertGreater(final_distance, 0.05)
        self.assertTrue(np.isfinite(result).all())


class TestSolverLIMX(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_twist_example_drives_opposite_boundary_rotations(self):
        """Match the ChysX mesh and rotate its end sections oppositely."""
        with wp.ScopedDevice("cuda:0"):
            example = ClothLimxTwistExample(ViewerNull(num_frames=1), None)
            targets = example._compute_anchor_targets(0.5 * np.pi)

        positions = example.model.particle_q.numpy()
        self.assertEqual(example.model.particle_count, 20_000)
        self.assertEqual(example.model.tri_count, 39_402)
        self.assertEqual(example.rot_angular_velocity, 1.0)
        self.assertEqual(example.rot_end_time, 25.0)
        example.sim_time = 2.0
        self.assertAlmostEqual(example._drive_angle(), 2.0)
        example.sim_time = 30.0
        self.assertAlmostEqual(example._drive_angle(), 25.0)
        np.testing.assert_allclose(positions.min(axis=0), [0.0, -0.25, -0.5], atol=1.0e-7)
        np.testing.assert_allclose(positions.max(axis=0), [0.0, 0.25, 0.5], atol=1.0e-7)

        boundary_count = example.boundary_particle_count
        rest_targets = example.anchor_rest_targets
        negative_z = targets[:boundary_count]
        positive_z = targets[boundary_count:]
        negative_z_rest = rest_targets[:boundary_count]
        positive_z_rest = rest_targets[boundary_count:]

        np.testing.assert_allclose(negative_z[:, 0], negative_z_rest[:, 1], atol=1.0e-6)
        np.testing.assert_allclose(positive_z[:, 0], -positive_z_rest[:, 1], atol=1.0e-6)
        np.testing.assert_allclose(targets[:, 1], np.zeros(2 * boundary_count), atol=1.0e-6)
        np.testing.assert_allclose(targets[:, 2], rest_targets[:, 2], atol=1.0e-7)
        negative_z_radius = np.linalg.norm(negative_z[:, :2], axis=1)
        positive_z_radius = np.linalg.norm(positive_z[:, :2], axis=1)
        np.testing.assert_allclose(negative_z_radius, np.abs(negative_z_rest[:, 1]), atol=1.0e-6)
        np.testing.assert_allclose(positive_z_radius, np.abs(positive_z_rest[:, 1]), atol=1.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_twist_example_runs_limx_self_collision_cuda_graph(self):
        """Keep the flat twist rest mesh contact-free with automatic thickness."""
        with wp.ScopedDevice("cuda:0"):
            example = ClothLimxTwistExample(ViewerNull(num_frames=1), None)
            example.self_collision.prepare(example.model.particle_q)
            rest_contact_counts = (
                int(example.self_collision.vertex_face_contacts.count.numpy()[0]),
                int(example.self_collision.edge_edge_contacts.count.numpy()[0]),
                int(example.self_collision.edge_face_contacts.count.numpy()[0]),
            )
            example.step()
            positions = example.state_0.particle_q.numpy()
            velocities = example.state_0.particle_qd.numpy()

        self.assertIsInstance(example.solver.dynamic_operator, ConstraintSelfCollision)
        self.assertAlmostEqual(float(np.sum(example.model.particle_mass.numpy())), 0.05, places=6)
        self.assertAlmostEqual(example.sim_dt, 0.01)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 50)
        self.assertEqual(example.solver.velocity_damping, 1.0)
        self.assertEqual(len(example.solver.constraints), 3)
        self.assertTrue(example.self_collision.thickness_was_estimated)
        self.assertTrue(example.self_collision.geometry_radius_topology_local_only)
        self.assertAlmostEqual(example.self_collision.geometry_radius_scale, 0.25)
        self.assertEqual(rest_contact_counts, (0, 0, 0))
        self.assertEqual(example.self_collision.stiffness, 1.0e3)
        self.assertEqual(example.self_collision.untangle_stiffness, 2.0e3)
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(velocities).all())

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_dynamic_contacts_prepare_once_before_each_newton_linearization(self):
        events = []
        begin_step_arguments = []
        prepare_positions = []
        bound_systems = []

        class RecordingDynamicOperator:
            def bind_static_system(self, static_diagonal, masses):
                bound_systems.append((static_diagonal, masses))

            def begin_step(self, positions, velocities, dt):
                events.append("begin_step")
                begin_step_arguments.append((positions, velocities, dt))

            def prepare(self, positions):
                events.append("prepare")
                prepare_positions.append(positions)

            def accumulate_force(self, positions, output):
                events.append("force")

            def accumulate_diagonal(self, positions, output):
                events.append("diagonal")

            def hessian_multiply(self, positions, vector, output):
                events.append("hvp")

        with wp.ScopedDevice("cuda:0"):
            builder = newton.ModelBuilder(up_axis="Z")
            builder.add_particles(pos=[wp.vec3(0.0)], vel=[wp.vec3(0.0)], mass=[1.0], radius=[0.01])
            model = builder.finalize(device="cuda:0")
            dynamic_operator = RecordingDynamicOperator()
            solver = SolverLIMX(
                model,
                [],
                nonlinear_iterations=2,
                linear_iterations=1,
                dynamic_operator=dynamic_operator,
            )
            self.assertEqual(bound_systems, [(solver.static_matrix.diagonal, model.particle_mass)])
            state_in = model.state()
            state_out = model.state()
            solver.step(state_in, state_out, None, None, 0.01)

        self.assertEqual(events.count("begin_step"), 1)
        self.assertEqual(events.count("prepare"), 2)
        self.assertEqual(events.count("force"), 2)
        self.assertEqual(events.count("diagonal"), 2)
        self.assertGreaterEqual(events.count("hvp"), 2)
        first_begin_step = events.index("begin_step")
        first_prepare = events.index("prepare")
        first_force = events.index("force")
        first_diagonal = events.index("diagonal")
        first_hvp = events.index("hvp")
        self.assertLess(first_begin_step, first_prepare)
        self.assertLess(first_prepare, first_force)
        self.assertLess(first_force, first_diagonal)
        self.assertLess(first_diagonal, first_hvp)
        self.assertIs(begin_step_arguments[0][0], state_in.particle_q)
        self.assertIs(begin_step_arguments[0][1], state_in.particle_qd)
        self.assertEqual(begin_step_arguments[0][2], 0.01)
        self.assertTrue(all(positions is solver.iterate_positions for positions in prepare_positions))

    def test_first_pcg_solve_warm_starts_from_previous_frame_increment(self):
        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_particles(pos=[wp.vec3(0.0)], vel=[wp.vec3(0.0)], mass=[1.0], radius=[0.01])
        model = builder.finalize(device="cpu")
        solver = SolverLIMX(model, [], nonlinear_iterations=1, linear_iterations=1)
        state_in = model.state()
        state_out = model.state()
        pcg = solver.linear_solver
        initial_guesses = []
        solutions = []
        zero_initial_guesses = []

        class RecordingPcgSolver:
            def solve(self, operator, rhs, solution, iterations, zero_initial_guess=True):
                initial_guesses.append(solution.numpy().copy())
                zero_initial_guesses.append(zero_initial_guess)
                executed = pcg.solve(
                    operator,
                    rhs,
                    solution,
                    iterations,
                    zero_initial_guess=zero_initial_guess,
                )
                solutions.append(solution.numpy().copy())
                return executed

        solver.linear_solver = RecordingPcgSolver()
        solver.step(state_in, state_out, None, None, 0.01)
        state_in, state_out = state_out, state_in
        solver.step(state_in, state_out, None, None, 0.01)

        self.assertEqual(zero_initial_guesses, [False, False])
        np.testing.assert_array_equal(initial_guesses[0], np.zeros((1, 3)))
        np.testing.assert_array_equal(initial_guesses[1], solutions[0])

    def test_example_advances_one_001_second_physics_step_per_frame(self):
        with wp.ScopedDevice("cpu"):
            example = ClothLimxExample(ViewerNull(num_frames=1), None)
            solver_step = example.solver.step
            solver_time_steps = []

            def record_solver_step(state_in, state_out, control, contacts, dt):
                solver_time_steps.append(dt)
                solver_step(state_in, state_out, control, contacts, dt)

            example.solver.step = record_solver_step
            example.step()

        self.assertEqual(solver_time_steps, [0.01])
        self.assertAlmostEqual(example.sim_time, 0.01)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_example_uses_membrane_and_dihedral_bending(self):
        with wp.ScopedDevice("cuda:0"):
            example = ClothLimxExample(ViewerNull(num_frames=1), None)

        anchor_constraints = [
            constraint for constraint in example.solver.constraints if isinstance(constraint, ConstraintAnchor)
        ]
        membrane_constraints = [
            constraint for constraint in example.solver.constraints if isinstance(constraint, ConstraintTriangleElastic)
        ]
        bending_constraints = [
            constraint for constraint in example.solver.constraints if isinstance(constraint, ConstraintDihedralBending)
        ]
        distance_constraints = [
            constraint for constraint in example.solver.constraints if isinstance(constraint, ConstraintDistance)
        ]
        self.assertEqual(len(anchor_constraints), 1)
        self.assertEqual(len(membrane_constraints), 1)
        self.assertEqual(len(bending_constraints), 1)
        self.assertEqual(bending_constraints[0].stiffness, 0.01)
        self.assertEqual(len(bending_constraints[0].host_dihedral_indices), 1160)
        self.assertEqual(distance_constraints, [])
        self.assertIsInstance(example.solver.dynamic_operator, ConstraintSelfCollision)
        self.assertAlmostEqual(example.solver.dynamic_operator.thickness, 0.01)
        self.assertGreater(example.solver.dynamic_operator.stiffness, 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_example_cuda_graph_advances_odd_substep_state(self):
        with wp.ScopedDevice("cuda:0"):
            example = ClothLimxExample(ViewerNull(num_frames=2), None)
            example.step()
            first_height = float(example.state_0.particle_q.numpy()[example.center_index, 2])

            example.step()
            second_height = float(example.state_0.particle_q.numpy()[example.center_index, 2])

        self.assertLess(second_height, first_height - 1.0e-4)

    @staticmethod
    def make_cloth():
        side = 5
        positions = [wp.vec3(x / (side - 1), y / (side - 1), 1.0) for y in range(side) for x in range(side)]
        triangles = []
        for y in range(side - 1):
            for x in range(side - 1):
                lower_left = y * side + x
                lower_right = lower_left + 1
                upper_left = lower_left + side
                upper_right = upper_left + 1
                if (x + y) % 2 == 0:
                    triangles.extend([(lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)])
                else:
                    triangles.extend([(lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left)])

        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_particles(
            pos=positions,
            vel=[wp.vec3(0.0)] * len(positions),
            mass=[0.3 / len(positions)] * len(positions),
            radius=[0.01] * len(positions),
        )
        triangle_array = np.asarray(triangles, dtype=np.int32)
        builder.add_triangles(triangle_array[:, 0], triangle_array[:, 1], triangle_array[:, 2])
        model = builder.finalize(device="cpu")

        edges = sorted(
            {
                tuple(sorted(edge))
                for triangle in triangles
                for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))
            }
        )
        positions_np = np.asarray(positions, dtype=np.float32)
        rest_lengths = [float(np.linalg.norm(positions_np[j] - positions_np[i])) for i, j in edges]
        anchor_indices = [0, side - 1]
        anchor_targets = [positions[index] for index in anchor_indices]
        constraints = [
            ConstraintAnchor(anchor_indices, anchor_targets, [1.0e6] * 2, len(positions), "cpu"),
            ConstraintDistance(edges, rest_lengths, [1.0e3] * len(edges), len(positions), "cpu"),
        ]
        return model, constraints, edges, np.asarray(rest_lengths), anchor_indices, side * side // 2

    def test_two_corner_anchored_cloth_sags_without_mutating_input(self):
        model, constraints, edges, rest_lengths, anchor_indices, center_index = self.make_cloth()
        solver = SolverLIMX(model, constraints, nonlinear_iterations=4, linear_iterations=32)
        state_in = model.state()
        state_out = model.state()
        initial_positions = state_in.particle_q.numpy().copy()
        input_velocities = state_in.particle_qd.numpy().copy()
        input_forces = state_in.particle_f.numpy().copy()

        dt = 1.0 / 240.0
        solver.step(state_in, state_out, None, None, dt)

        np.testing.assert_array_equal(state_in.particle_q.numpy(), initial_positions)
        np.testing.assert_array_equal(state_in.particle_qd.numpy(), input_velocities)
        np.testing.assert_array_equal(state_in.particle_f.numpy(), input_forces)
        np.testing.assert_allclose(
            state_out.particle_qd.numpy(),
            (state_out.particle_q.numpy() - initial_positions) / dt,
            rtol=1.0e-6,
            atol=1.0e-7,
        )

        state_in, state_out = state_out, state_in
        for _ in range(239):
            solver.step(state_in, state_out, None, None, dt)
            state_in, state_out = state_out, state_in

        positions = state_in.particle_q.numpy()
        velocities = state_in.particle_qd.numpy()
        current_edge_lengths = np.asarray([np.linalg.norm(positions[j] - positions[i]) for i, j in edges])

        np.testing.assert_allclose(positions[anchor_indices], initial_positions[anchor_indices], atol=1.0e-3)
        self.assertLess(positions[center_index, 2], initial_positions[center_index, 2] - 5.0e-2)
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(velocities).all())
        self.assertLess(float(np.max(current_edge_lengths)), 2.0 * float(np.max(rest_lengths)))
        self.assertGreater(positions[center_index, 2], initial_positions[center_index, 2] - 0.5 * 9.81)

    def test_current_position_hessian_does_not_lock_transverse_prediction(self):
        builder = newton.ModelBuilder(up_axis="Z")
        builder.add_particles(
            pos=[wp.vec3(0.0), wp.vec3(0.5, 0.0, 0.0)],
            vel=[wp.vec3(0.0), wp.vec3(0.0, 5.0, 0.0)],
            mass=[1.0, 1.0],
            radius=[0.01, 0.01],
        )
        model = builder.finalize(device="cpu")
        model.set_gravity((0.0, 0.0, 0.0))
        constraints = [
            ConstraintAnchor([0], [wp.vec3(0.0)], [1.0e8], 2, "cpu"),
            ConstraintDistance([(0, 1)], [0.5], [1.0e4], 2, "cpu"),
        ]
        solver = SolverLIMX(model, constraints, nonlinear_iterations=1, linear_iterations=20)
        state_in = model.state()
        state_out = model.state()

        solver.step(state_in, state_out, None, None, 0.1)

        positions = state_out.particle_q.numpy()
        self.assertGreater(positions[1, 1], 0.4)
        np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0], atol=1.0e-5)

    def test_public_exports(self):
        self.assertIs(newton.solvers.SolverLIMX, SolverLIMX)
        self.assertIs(newton.solvers.ConstraintAnchor, ConstraintAnchor)
        self.assertIs(newton.solvers.ConstraintDihedralBending, ConstraintDihedralBending)
        self.assertIs(newton.solvers.ConstraintDistance, ConstraintDistance)
        self.assertIs(newton.solvers.ConstraintSelfCollision, ConstraintSelfCollision)
        self.assertIs(newton.solvers.ConstraintTriangleElastic, ConstraintTriangleElastic)

    def test_rejects_model_with_rigid_bodies(self):
        builder = newton.ModelBuilder()
        builder.add_particles(pos=[wp.vec3(0.0)], vel=[wp.vec3(0.0)], mass=[1.0])
        builder.add_body()
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "particle-only"):
            SolverLIMX(model, [])

    def test_rejects_inactive_particles_in_favor_of_anchor_constraints(self):
        builder = newton.ModelBuilder()
        builder.add_particles(pos=[wp.vec3(0.0)], vel=[wp.vec3(0.0)], mass=[1.0], flags=[0])
        model = builder.finalize(device="cpu")

        with self.assertRaisesRegex(ValueError, "ConstraintAnchor"):
            SolverLIMX(model, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
