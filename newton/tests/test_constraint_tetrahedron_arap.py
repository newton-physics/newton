# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.constraints.tetrahedron_arap import (
    ConstraintTetrahedronARAP,
    _arap_energy,
    _arap_hessian_unscaled,
    _project_psd,
    mat99,
)
from newton.viewer import ViewerNull


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


@wp.kernel
def _evaluate_arap_hessians(
    deformation_gradients: wp.array[wp.mat33],
    raw_hessians: wp.array[mat99],
    projected_hessians: wp.array[mat99],
):
    tetrahedron = wp.tid()
    raw_hessian = _arap_hessian_unscaled(deformation_gradients[tetrahedron])
    raw_hessians[tetrahedron] = raw_hessian
    projected_hessians[tetrahedron] = _project_psd(raw_hessian)


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


def _arap_gradient_reference(deformation: np.ndarray) -> np.ndarray:
    """Evaluate the unscaled ARAP deformation-gradient derivative."""
    u, _singular_values, vt = _proper_svd(deformation)
    return 2.0 * (deformation - u @ vt)


def _arap_hessian_reference(deformation: np.ndarray) -> np.ndarray:
    """Evaluate libuipc's unscaled analytical ARAP Hessian."""
    u, singular_values, vt = _proper_svd(deformation)
    twists = (
        np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    )
    modes = [(u @ twist @ vt / np.sqrt(2.0)).reshape(9, order="F") for twist in twists]
    hessian = 2.0 * np.eye(9)
    hessian -= 4.0 / (singular_values[0] + singular_values[1]) * np.outer(modes[0], modes[0])
    hessian -= 4.0 / (singular_values[1] + singular_values[2]) * np.outer(modes[1], modes[1])
    hessian -= 4.0 / (singular_values[0] + singular_values[2]) * np.outer(modes[2], modes[2])
    return hessian


def _project_psd_reference(hessian: np.ndarray) -> np.ndarray:
    """Clamp a complete symmetric matrix through NumPy eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _deformation_jacobian(inverse_rest: np.ndarray) -> np.ndarray:
    """Build the column-major deformation-gradient Jacobian for four vertices."""
    material_gradients = (
        -np.sum(inverse_rest, axis=0),
        inverse_rest[0],
        inverse_rest[1],
        inverse_rest[2],
    )
    jacobian = np.zeros((9, 12))
    for local_vertex, material_gradient in enumerate(material_gradients):
        for material_axis in range(3):
            for spatial_axis in range(3):
                jacobian[3 * material_axis + spatial_axis, 3 * local_vertex + spatial_axis] = material_gradient[
                    material_axis
                ]
    return jacobian


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

    def test_public_export_resolves_constraint(self):
        """Resolve the tetrahedral ARAP constraint through the public solver API."""
        self.assertIs(newton.solvers.ConstraintTetrahedronARAP, ConstraintTetrahedronARAP)

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


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintTetrahedronARAPHessian(unittest.TestCase):
    DEVICE = "cuda:0"
    INVERSE_REST = np.eye(3, dtype=np.float64)
    STIFFNESS = 7.0
    REST_VOLUME = 1.0 / 6.0

    @classmethod
    def evaluate_hessians(cls, deformation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate raw and projected deformation-gradient Hessians on CUDA."""
        raw_hessians = wp.empty(1, dtype=mat99, device=cls.DEVICE)
        projected_hessians = wp.empty(1, dtype=mat99, device=cls.DEVICE)
        wp.launch(
            _evaluate_arap_hessians,
            dim=1,
            inputs=[wp.array([_mat33(deformation)], dtype=wp.mat33, device=cls.DEVICE)],
            outputs=[raw_hessians, projected_hessians],
            device=cls.DEVICE,
        )
        return raw_hessians.numpy()[0], projected_hessians.numpy()[0]

    @classmethod
    def make_constraint(cls) -> ConstraintTetrahedronARAP:
        """Create one unit-rest tetrahedron constraint on CUDA."""
        return ConstraintTetrahedronARAP(
            [(0, 1, 2, 3)],
            [_mat33(cls.INVERSE_REST)],
            [cls.STIFFNESS],
            4,
            cls.DEVICE,
        )

    @classmethod
    def assemble_single_tetrahedron(
        cls,
        positions: np.ndarray,
    ) -> tuple[ConstraintTetrahedronARAP, object, np.ndarray]:
        """Assemble force and all block-CSR Hessian values for one tetrahedron."""
        constraint = cls.make_constraint()
        builder = BlockCsrBuilder(4)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize(cls.DEVICE)
        constraint.bind_hessian(matrix)
        forces = wp.zeros(4, dtype=wp.vec3, device=cls.DEVICE)
        matrix.clear_values()
        constraint.accumulate_force_and_hessian(
            wp.array(positions, dtype=wp.vec3, device=cls.DEVICE),
            forces,
            matrix.values,
        )
        matrix.update_diagonal()
        return constraint, matrix, forces.numpy()

    @staticmethod
    def dense_hessian(matrix) -> np.ndarray:
        """Expand a four-particle block-CSR matrix into a dense matrix."""
        values = matrix.values.numpy()
        dense = np.empty((12, 12))
        for particle_i in range(4):
            for particle_j in range(4):
                dense[3 * particle_i : 3 * particle_i + 3, 3 * particle_j : 3 * particle_j + 3] = values[
                    matrix.block_index(particle_i, particle_j)
                ]
        return dense

    def test_raw_hessian_matches_libuipc_formula(self):
        """Match the complete analytical deformation-gradient Hessian."""
        deformation = np.asarray(
            [[1.2, 0.15, -0.05], [0.1, 1.1, 0.2], [0.0, -0.1, 0.95]],
            dtype=np.float64,
        )

        raw_hessian, _projected_hessian = self.evaluate_hessians(deformation)

        np.testing.assert_allclose(raw_hessian, _arap_hessian_reference(deformation), rtol=2.0e-4, atol=2.0e-4)

    def test_raw_hessian_matches_gradient_finite_difference_when_psd(self):
        """Match a PSD exact Hessian to a centered gradient finite difference."""
        deformation = np.diag([1.4, 1.3, 1.2]).astype(np.float64)
        raw_hessian, _projected_hessian = self.evaluate_hessians(deformation)
        epsilon = 1.0e-4
        finite_difference = np.empty((9, 9))

        for column in range(9):
            deformation_plus = deformation.copy().reshape(-1, order="F")
            deformation_minus = deformation.copy().reshape(-1, order="F")
            deformation_plus[column] += epsilon
            deformation_minus[column] -= epsilon
            gradient_plus = _arap_gradient_reference(deformation_plus.reshape(3, 3, order="F")).reshape(9, order="F")
            gradient_minus = _arap_gradient_reference(deformation_minus.reshape(3, 3, order="F")).reshape(9, order="F")
            finite_difference[:, column] = (gradient_plus - gradient_minus) / (2.0 * epsilon)

        self.assertGreaterEqual(float(np.linalg.eigvalsh(raw_hessian)[0]), -2.0e-4)
        np.testing.assert_allclose(raw_hessian, finite_difference, rtol=4.0e-3, atol=4.0e-3)

    def test_full_evd_projection_matches_numpy(self):
        """Clamp negative raw eigenvalues through complete symmetric EVD."""
        deformation = np.asarray(
            [[0.65, 0.08, 0.0], [0.02, 0.7, -0.04], [0.0, 0.03, 0.8]],
            dtype=np.float64,
        )
        raw_hessian, projected_hessian = self.evaluate_hessians(deformation)
        expected = _project_psd_reference(_arap_hessian_reference(deformation))

        self.assertLess(float(np.linalg.eigvalsh(raw_hessian)[0]), -1.0e-2)
        np.testing.assert_allclose(projected_hessian, expected, rtol=2.0e-3, atol=2.0e-3)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(projected_hessian)[0]), -2.0e-3)

    def test_assembled_hessian_matches_jacobian_mapping_and_is_psd(self):
        """Map the projected deformation Hessian into all particle blocks."""
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.02, 0.0],
                [0.05, 0.75, 0.03],
                [0.0, -0.02, 0.85],
            ],
            dtype=np.float64,
        )
        _constraint, matrix, _forces = self.assemble_single_tetrahedron(positions)
        deformation = _deformation_gradient_reference(positions, self.INVERSE_REST)
        hessian_f = _project_psd_reference(_arap_hessian_reference(deformation)) * self.STIFFNESS * self.REST_VOLUME
        jacobian = _deformation_jacobian(self.INVERSE_REST)
        expected = jacobian.T @ hessian_f @ jacobian
        actual = self.dense_hessian(matrix)

        np.testing.assert_allclose(actual, expected, rtol=3.0e-3, atol=3.0e-3)
        np.testing.assert_allclose(actual, actual.T, atol=2.0e-5)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(actual)[0]), -2.0e-3)

    def test_reassembly_changes_values_without_changing_block_topology(self):
        """Keep sixteen block coordinates fixed while refreshing their values."""
        initial_positions = TestConstraintTetrahedronARAPMath.REST_POSITIONS
        constraint, matrix, _forces = self.assemble_single_tetrahedron(initial_positions)
        row_offsets = matrix.row_offsets.numpy().copy()
        column_indices = matrix.column_indices.numpy().copy()
        initial_values = matrix.values.numpy().copy()
        new_positions = np.asarray(
            [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.0, 0.9, 0.15], [0.05, 0.0, 1.1]],
            dtype=np.float64,
        )
        forces = wp.zeros(4, dtype=wp.vec3, device=self.DEVICE)

        matrix.clear_values()
        constraint.accumulate_force_and_hessian(
            wp.array(new_positions, dtype=wp.vec3, device=self.DEVICE),
            forces,
            matrix.values,
        )

        np.testing.assert_array_equal(matrix.row_offsets.numpy(), row_offsets)
        np.testing.assert_array_equal(matrix.column_indices.numpy(), column_indices)
        self.assertFalse(np.allclose(matrix.values.numpy(), initial_values))

    def test_rejects_invalid_runtime_storage_before_launch(self):
        """Reject unbound, mismatched, and wrong-device runtime arrays."""
        constraint = self.make_constraint()
        positions = wp.zeros(4, dtype=wp.vec3, device=self.DEVICE)
        forces = wp.zeros(4, dtype=wp.vec3, device=self.DEVICE)

        with self.assertRaisesRegex(RuntimeError, "bind_hessian"):
            constraint.accumulate_force_and_hessian(
                positions,
                forces,
                wp.zeros(16, dtype=wp.mat33, device=self.DEVICE),
            )

        builder = BlockCsrBuilder(4)
        constraint.append_hessian_structure(builder)
        matrix = builder.finalize(self.DEVICE)
        constraint.bind_hessian(matrix)

        with self.assertRaisesRegex(ValueError, "4 particle rows"):
            constraint.accumulate_force(wp.zeros(3, dtype=wp.vec3, device=self.DEVICE), forces)
        with self.assertRaisesRegex(ValueError, "16 Hessian blocks"):
            constraint.accumulate_force_and_hessian(
                positions,
                forces,
                wp.zeros(15, dtype=wp.mat33, device=self.DEVICE),
            )
        with self.assertRaisesRegex(ValueError, "same device"):
            constraint.accumulate_force(
                wp.zeros(4, dtype=wp.vec3, device="cpu"),
                wp.zeros(4, dtype=wp.vec3, device="cpu"),
            )


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintTetrahedronARAPSolver(unittest.TestCase):
    DEVICE = "cuda:0"
    DT = 0.01

    @classmethod
    def make_beam(cls):
        """Build a small anchored ARAP beam and its LIMX solver."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_soft_grid(
            pos=wp.vec3(0.0, -0.025, 0.75),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=4,
            dim_y=1,
            dim_z=1,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1000.0,
            k_mu=0.0,
            k_lambda=0.0,
            k_damp=0.0,
            fix_left=False,
        )
        model = builder.finalize(device=cls.DEVICE)
        rest_positions = model.particle_q.numpy()
        tetrahedra = model.tet_indices.numpy()
        inverse_rest_matrices = model.tet_poses.numpy()
        minimum_x = float(np.min(rest_positions[:, 0]))
        maximum_x = float(np.max(rest_positions[:, 0]))
        anchor_indices = np.flatnonzero(np.isclose(rest_positions[:, 0], minimum_x)).tolist()
        free_end_indices = np.flatnonzero(np.isclose(rest_positions[:, 0], maximum_x)).tolist()
        constraints = [
            newton.solvers.ConstraintAnchor(
                anchor_indices,
                [wp.vec3(*position) for position in rest_positions[anchor_indices]],
                [1.0e8] * len(anchor_indices),
                model.particle_count,
                cls.DEVICE,
            ),
            newton.solvers.ConstraintTetrahedronARAP(
                tetrahedra.tolist(),
                [_mat33(matrix) for matrix in inverse_rest_matrices],
                [1.0e6] * model.tet_count,
                model.particle_count,
                cls.DEVICE,
            ),
        ]
        solver = newton.solvers.SolverLIMX(
            model,
            constraints,
            nonlinear_iterations=1,
            linear_iterations=128,
            velocity_damping=1.0,
        )
        return model, solver, rest_positions, tetrahedra, anchor_indices, free_end_indices

    def test_single_newton_step_updates_velocity_from_position_increment(self):
        """Advance ARAP particles with exactly one Newton increment."""
        model, solver, initial_positions, _tetrahedra, _anchor_indices, _free_end_indices = self.make_beam()
        state_in = model.state()
        state_out = model.state()

        solver.step(state_in, state_out, None, None, self.DT)

        np.testing.assert_allclose(
            state_out.particle_qd.numpy(),
            (state_out.particle_q.numpy() - initial_positions) / self.DT,
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        self.assertEqual(solver.nonlinear_iterations, 1)

    def test_fixed_beam_rollout_sags_without_inversion(self):
        """Keep a fixed ARAP beam finite, anchored, sagging, and positive-volume."""
        model, solver, rest_positions, tetrahedra, anchor_indices, free_end_indices = self.make_beam()
        state_in = model.state()
        state_out = model.state()
        initial_free_end_z = float(np.mean(rest_positions[free_end_indices, 2]))
        minimum_free_end_z = np.inf

        for _ in range(80):
            solver.step(state_in, state_out, None, None, self.DT)
            state_in, state_out = state_out, state_in
            positions = state_in.particle_q.numpy()
            minimum_free_end_z = min(minimum_free_end_z, float(np.mean(positions[free_end_indices, 2])))
            self.assertTrue(np.isfinite(positions).all())
            for tetrahedron in tetrahedra:
                deformation_edges = np.column_stack(
                    (
                        positions[tetrahedron[1]] - positions[tetrahedron[0]],
                        positions[tetrahedron[2]] - positions[tetrahedron[0]],
                        positions[tetrahedron[3]] - positions[tetrahedron[0]],
                    )
                )
                self.assertGreater(float(np.linalg.det(deformation_edges)), 0.0)

        positions = state_in.particle_q.numpy()
        velocities = state_in.particle_qd.numpy()
        np.testing.assert_allclose(positions[anchor_indices], rest_positions[anchor_indices], atol=2.0e-3)
        self.assertLess(minimum_free_end_z, initial_free_end_z - 2.0e-3)
        self.assertTrue(np.isfinite(velocities).all())

    def test_example_uses_one_full_newton_step_without_damping(self):
        """Configure the ARAP beam with one undamped Newton step per frame."""
        module = importlib.import_module("newton.examples.softbody.example_softbody_limx_arap_beam")
        example = module.Example(ViewerNull(num_frames=1), None)

        self.assertEqual(example.frame_dt, 0.01)
        self.assertEqual(example.solver.nonlinear_iterations, 1)
        self.assertEqual(example.solver.linear_iterations, 128)
        self.assertEqual(example.solver.velocity_damping, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
