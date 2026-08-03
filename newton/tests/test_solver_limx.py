# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton
import numpy as np
import warp as wp

from newton._src.solvers.limx.block_csr import BlockCsrBuilder
from newton._src.solvers.limx.constraints.anchor import ConstraintAnchor
from newton._src.solvers.limx.constraints.distance import ConstraintDistance
from newton._src.solvers.limx.linear_solver import PcgSolver
from newton._src.solvers.limx.operator import CompositeLinearOperator, EmptyDynamicConstraintOperator
from newton._src.solvers.limx.solver_limx import SolverLIMX


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


class TestPcgSolver(unittest.TestCase):
    def make_operator(self):
        return TestCompositeLinearOperator().make_operator()

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
                    triangles.extend(
                        [(lower_left, lower_right, upper_right), (lower_left, upper_right, upper_left)]
                    )
                else:
                    triangles.extend(
                        [(lower_left, lower_right, upper_left), (lower_right, upper_right, upper_left)]
                    )

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

        solver.step(state_in, state_out, None, None, 1.0 / 240.0)

        np.testing.assert_array_equal(state_in.particle_q.numpy(), initial_positions)
        np.testing.assert_array_equal(state_in.particle_qd.numpy(), input_velocities)
        np.testing.assert_array_equal(state_in.particle_f.numpy(), input_forces)

        state_in, state_out = state_out, state_in
        for _ in range(239):
            solver.step(state_in, state_out, None, None, 1.0 / 240.0)
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

    def test_public_exports(self):
        self.assertIs(newton.solvers.SolverLIMX, SolverLIMX)
        self.assertIs(newton.solvers.ConstraintAnchor, ConstraintAnchor)
        self.assertIs(newton.solvers.ConstraintDistance, ConstraintDistance)


if __name__ == "__main__":
    unittest.main(verbosity=2)
