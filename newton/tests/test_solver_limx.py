# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.constraints.anchor import ConstraintAnchor
from newton._src.solvers.limx.constraints.distance import ConstraintDistance


class TestBlockCsr(unittest.TestCase):
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

        anchors.append_hessian(builder)
        matrix = builder.finalize("cpu")

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
        springs = ConstraintDistance([(0, 1)], [1.0], [20.0], 2, "cpu")
        positions = wp.zeros(2, dtype=wp.vec3, device="cpu")
        forces = wp.zeros(2, dtype=wp.vec3, device="cpu")

        springs.accumulate_force(positions, forces)

        self.assertTrue(np.isfinite(forces.numpy()).all())
        np.testing.assert_array_equal(forces.numpy(), np.zeros((2, 3)))

    def test_hessian_adds_four_projective_dynamics_blocks(self):
        springs = ConstraintDistance([(0, 1)], [1.0], [5.0], 2, "cpu")
        builder = BlockCsrBuilder(2)

        springs.append_hessian(builder)
        matrix = builder.finalize("cpu")

        np.testing.assert_allclose(matrix.diagonal.numpy()[0], np.eye(3) * 5.0)
        np.testing.assert_allclose(matrix.diagonal.numpy()[1], np.eye(3) * 5.0)
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 1, 0, 1])
        np.testing.assert_allclose(
            matrix.values.numpy(),
            [np.eye(3) * 5.0, np.eye(3) * -5.0, np.eye(3) * -5.0, np.eye(3) * 5.0],
        )

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
