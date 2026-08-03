# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.constraints.anchor import ConstraintAnchor
from newton._src.solvers.limx.constraints.distance import ConstraintDistance
from newton._src.solvers.limx.constraints.triangle_elastic import ConstraintTriangleElastic
from newton._src.solvers.limx.linear_solver import PcgSolver
from newton._src.solvers.limx.operator import CompositeLinearOperator, EmptyDynamicConstraintOperator
from newton._src.solvers.limx.solver_newton import SolverLIMX
from newton.examples.cloth.example_cloth_limx import Example as ClothLimxExample
from newton.viewer import ViewerNull


class TestBlockCsr(unittest.TestCase):
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


class TestSolverLIMX(unittest.TestCase):
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

    def test_example_uses_triangle_membrane_without_distance_springs(self):
        with wp.ScopedDevice("cpu"):
            example = ClothLimxExample(ViewerNull(num_frames=1), None)

        membrane_constraints = [
            constraint for constraint in example.solver.constraints if isinstance(constraint, ConstraintTriangleElastic)
        ]
        distance_constraints = [
            constraint for constraint in example.solver.constraints if isinstance(constraint, ConstraintDistance)
        ]
        self.assertEqual(len(membrane_constraints), 1)
        self.assertEqual(distance_constraints, [])

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
        self.assertIs(newton.solvers.ConstraintDistance, ConstraintDistance)
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
