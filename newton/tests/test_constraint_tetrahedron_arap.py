# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

import numpy as np
import warp as wp

from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.constraints.tetrahedron_arap import ConstraintTetrahedronARAP


def _mat33(values: np.ndarray) -> wp.mat33:
    """Convert a three-by-three NumPy matrix to a Warp matrix."""
    return wp.mat33(*np.asarray(values, dtype=np.float32).reshape(-1))


class TestConstraintTetrahedronARAPConstruction(unittest.TestCase):
    INVERSE_REST = np.eye(3, dtype=np.float32)

    @classmethod
    def make_constraint(cls, device: str = "cpu") -> ConstraintTetrahedronARAP:
        """Create one unit-rest tetrahedral ARAP constraint."""
        return ConstraintTetrahedronARAP(
            [(0, 1, 2, 3)],
            [_mat33(cls.INVERSE_REST)],
            [7.0],
            4,
            device,
        )

    def test_recovers_rest_volume_and_creates_sixteen_blocks(self):
        """Recover positive rest volume and bind all ordered particle blocks."""
        constraint = self.make_constraint()

        self.assertEqual(constraint.host_rest_volumes, (1.0 / 6.0,))

        builder = BlockCsrBuilder(4)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize("cpu")
        constraint.bind_hessian(matrix)

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), [0, 4, 8, 12, 16])
        np.testing.assert_array_equal(matrix.column_indices.numpy(), [0, 1, 2, 3] * 4)
        self.assertEqual(constraint.hessian_block_indices.shape, (1, 16))
        self.assertEqual(constraint.hessian_value_count, 16)

    def test_rejects_invalid_constructor_input(self):
        """Reject malformed topology, rest matrices, stiffnesses, and counts."""
        identity = _mat33(self.INVERSE_REST)
        nan_matrix = self.INVERSE_REST.copy()
        nan_matrix[0, 0] = np.nan
        invalid_arguments = [
            ([], [], [], 4),
            ([(0, 1, 2, 3)], [], [1.0], 4),
            ([(0, 1, 2, 3)], [identity], [], 4),
            ([(0, 1, 2)], [identity], [1.0], 4),
            ([(0, 1, 2, 2)], [identity], [1.0], 4),
            ([(-1, 1, 2, 3)], [identity], [1.0], 4),
            ([(0, 1, 2, 4)], [identity], [1.0], 4),
            ([(0, 1, 2, 3)], [identity], [1.0], 0),
            ([(0, 1, 2, 3)], [_mat33(nan_matrix)], [1.0], 4),
            ([(0, 1, 2, 3)], [_mat33(np.diag([1.0, 1.0, 0.0]))], [1.0], 4),
            ([(0, 1, 2, 3)], [_mat33(np.diag([1.0, 1.0, -1.0]))], [1.0], 4),
            ([(0, 1, 2, 3)], [identity], [0.0], 4),
            ([(0, 1, 2, 3)], [identity], [-1.0], 4),
            ([(0, 1, 2, 3)], [identity], [float("nan")], 4),
            ([(0, 1, 2, 3)], [identity], [float("inf")], 4),
        ]

        for tetrahedra, inverse_rest_matrices, stiffnesses, particle_count in invalid_arguments:
            with self.subTest(
                tetrahedra=tetrahedra,
                stiffnesses=stiffnesses,
                particle_count=particle_count,
            ):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    with self.assertRaises((TypeError, ValueError)):
                        ConstraintTetrahedronARAP(
                            tetrahedra,
                            inverse_rest_matrices,
                            stiffnesses,
                            particle_count,
                            "cpu",
                        )

    def test_rejects_mismatched_block_matrix(self):
        """Reject block structures with a different particle count."""
        constraint = self.make_constraint()

        with self.assertRaisesRegex(ValueError, "particle counts"):
            constraint.append_hessian_structure(BlockCsrBuilder(5))

        matrix = BlockCsrBuilder(5).finalize("cpu")
        with self.assertRaisesRegex(ValueError, "particle counts"):
            constraint.bind_hessian(matrix)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_rejects_mismatched_block_matrix_device(self):
        """Reject block storage allocated on a different device."""
        constraint = self.make_constraint("cpu")
        builder = BlockCsrBuilder(4)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize("cuda:0")

        with self.assertRaisesRegex(ValueError, "devices"):
            constraint.bind_hessian(matrix)


if __name__ == "__main__":
    unittest.main(verbosity=2)
