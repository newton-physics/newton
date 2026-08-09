# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

import numpy as np
import warp as wp

from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.constraints.tetrahedron_arap import ConstraintTetrahedronARAP, _arap_energy


def _mat33(values: np.ndarray) -> wp.mat33:
    """Convert a three-by-three NumPy matrix to a Warp matrix."""
    return wp.mat33(*np.asarray(values, dtype=np.float32).reshape(-1))


@wp.kernel
def _evaluate_arap_energy(
    deformation_gradients: wp.array[wp.mat33],
    stiffnesses: wp.array[float],
    rest_volumes: wp.array[float],
    energies: wp.array[float],
):
    tetrahedron = wp.tid()
    energies[tetrahedron] = _arap_energy(
        deformation_gradients[tetrahedron],
        stiffnesses[tetrahedron],
        rest_volumes[tetrahedron],
    )


def _proper_svd(deformation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute an SVD whose left and right bases are proper rotations."""
    u, singular_values, vt = np.linalg.svd(deformation)
    if np.linalg.det(u) < 0.0:
        u[:, -1] *= -1.0
        singular_values[-1] *= -1.0
    if np.linalg.det(vt) < 0.0:
        vt[-1, :] *= -1.0
        singular_values[-1] *= -1.0
    return u, singular_values, vt


def _arap_energy_reference(
    positions: np.ndarray,
    inverse_rest: np.ndarray,
    stiffness: float,
) -> float:
    """Evaluate the tetrahedral ARAP energy with a NumPy SVD."""
    deformation = _deformation_gradient_reference(positions, inverse_rest)
    u, _singular_values, vt = _proper_svd(deformation)
    rotation = u @ vt
    rest_volume = 1.0 / (6.0 * np.linalg.det(inverse_rest))
    return float(stiffness * rest_volume * np.sum((deformation - rotation) ** 2))


def _deformation_gradient_reference(positions: np.ndarray, inverse_rest: np.ndarray) -> np.ndarray:
    """Build a column-major deformation gradient from tetrahedron positions."""
    edges = np.column_stack(
        (
            positions[1] - positions[0],
            positions[2] - positions[0],
            positions[3] - positions[0],
        )
    )
    return edges @ inverse_rest


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a proper rotation from an axis and angle."""
    direction = np.asarray(axis, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x, y, z = direction
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


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


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintTetrahedronARAPMath(unittest.TestCase):
    REST_POSITIONS = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    INVERSE_REST = np.eye(3, dtype=np.float64)
    STIFFNESS = 7.0
    DEVICE = "cuda:0"

    @classmethod
    def make_constraint(cls) -> ConstraintTetrahedronARAP:
        """Create the unit tetrahedron constraint on CUDA."""
        return ConstraintTetrahedronARAP(
            [(0, 1, 2, 3)],
            [_mat33(cls.INVERSE_REST)],
            [cls.STIFFNESS],
            4,
            cls.DEVICE,
        )

    @classmethod
    def evaluate_energy(cls, positions: np.ndarray) -> float:
        """Evaluate the private Warp energy function for one tetrahedron."""
        deformation = _deformation_gradient_reference(positions, cls.INVERSE_REST)
        energies = wp.empty(1, dtype=float, device=cls.DEVICE)
        wp.launch(
            _evaluate_arap_energy,
            dim=1,
            inputs=[
                wp.array([_mat33(deformation)], dtype=wp.mat33, device=cls.DEVICE),
                wp.array([cls.STIFFNESS], dtype=float, device=cls.DEVICE),
                wp.array([1.0 / 6.0], dtype=float, device=cls.DEVICE),
            ],
            outputs=[energies],
            device=cls.DEVICE,
        )
        return float(energies.numpy()[0])

    @classmethod
    def evaluate_forces(cls, positions: np.ndarray) -> np.ndarray:
        """Evaluate physical ARAP forces for one tetrahedron."""
        constraint = cls.make_constraint()
        forces = wp.zeros(4, dtype=wp.vec3, device=cls.DEVICE)
        constraint.accumulate_force(
            wp.array(positions, dtype=wp.vec3, device=cls.DEVICE),
            forces,
        )
        return forces.numpy()

    def test_rigid_motion_has_zero_energy_and_force(self):
        """Preserve zero ARAP energy and force under proper rigid motion."""
        rotation = _rotation_matrix(np.asarray([1.0, 2.0, -1.0]), 0.73)
        translation = np.asarray([0.4, -0.7, 1.2])
        rigid_positions = self.REST_POSITIONS @ rotation.T + translation

        self.assertAlmostEqual(self.evaluate_energy(self.REST_POSITIONS), 0.0, delta=1.0e-7)
        self.assertAlmostEqual(self.evaluate_energy(rigid_positions), 0.0, delta=1.0e-7)
        np.testing.assert_allclose(self.evaluate_forces(self.REST_POSITIONS), 0.0, atol=2.0e-6)
        np.testing.assert_allclose(self.evaluate_forces(rigid_positions), 0.0, atol=2.0e-6)

    def test_internal_forces_balance_linear_and_angular_momentum(self):
        """Balance total force and torque for a deformed tetrahedron."""
        positions = np.asarray(
            [
                [0.1, -0.2, 0.3],
                [1.25, 0.05, 0.4],
                [0.15, 0.9, 0.2],
                [-0.1, 0.2, 1.1],
            ],
            dtype=np.float64,
        )

        forces = self.evaluate_forces(positions)
        torque = np.sum(np.cross(positions, forces), axis=0)

        np.testing.assert_allclose(np.sum(forces, axis=0), 0.0, atol=2.0e-6)
        np.testing.assert_allclose(torque, 0.0, atol=3.0e-6)

    def test_force_matches_negative_energy_gradient(self):
        """Match physical force to a centered finite-difference energy gradient."""
        positions = np.asarray(
            [
                [0.05, -0.1, 0.2],
                [1.2, 0.15, 0.25],
                [0.1, 1.1, -0.05],
                [-0.15, 0.1, 0.95],
            ],
            dtype=np.float64,
        )
        forces = self.evaluate_forces(positions)
        epsilon = 1.0e-4
        energy_gradient = np.empty((4, 3))

        for particle in range(4):
            for axis in range(3):
                positions_plus = positions.copy()
                positions_minus = positions.copy()
                positions_plus[particle, axis] += epsilon
                positions_minus[particle, axis] -= epsilon
                energy_gradient[particle, axis] = (
                    _arap_energy_reference(positions_plus, self.INVERSE_REST, self.STIFFNESS)
                    - _arap_energy_reference(positions_minus, self.INVERSE_REST, self.STIFFNESS)
                ) / (2.0 * epsilon)

        np.testing.assert_allclose(forces, -energy_gradient, rtol=3.0e-3, atol=3.0e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
