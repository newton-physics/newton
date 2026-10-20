# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regressions for batching independent sparse DVI worlds."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.solvers.common import WarmStartMode
from newton._src.solvers.kamino._src.solvers.dvi import DVISolver
from newton._src.solvers.kamino._src.solvers.dvi import sparse as sparse_dvi
from newton.tests.kamino.test_kamino_solvers_dvi import _make_sparse_dual_problem
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.utils import basics


class TestKaminoDVIWorldBatches(unittest.TestCase):
    def test_batch_budget_and_eligibility(self):
        """Keep unsupported paths global and bound the inverse working set."""
        path = SimpleNamespace(
            bilateral_inverse=object(),
            data=SimpleNamespace(state=SimpleNamespace(_sparse_coupling_allocated=False)),
            device=SimpleNamespace(is_cuda=True),
            size=SimpleNamespace(
                num_worlds=8192,
                max_of_num_body_dofs=1024,
                max_of_num_bilateral_joint_cts=84,
                max_of_max_contacts=85,
            ),
        )
        for rows, expected in ((1, 2048), (84, 2048), (128, 1024), (256, 256)):
            with self.subTest(rows=rows):
                path.size.max_of_num_bilateral_joint_cts = rows
                self.assertEqual(sparse_dvi._sparse_alternating_world_batch_size(path), expected)

        path.size.max_of_num_bilateral_joint_cts = 84
        for worlds in (0, 1, 2048):
            with self.subTest(worlds=worlds):
                path.size.num_worlds = worlds
                self.assertEqual(sparse_dvi._sparse_alternating_world_batch_size(path), max(1, worlds))
        path.size.num_worlds = 8192
        for target, attribute, value in (
            (path, "bilateral_inverse", None),
            (path.data.state, "_sparse_coupling_allocated", True),
            (path.device, "is_cuda", False),
            (path.size, "max_of_num_body_dofs", 1025),
            (path.size, "max_of_max_contacts", 2048),
        ):
            with self.subTest(attribute=attribute), mock.patch.object(target, attribute, value):
                self.assertEqual(sparse_dvi._sparse_alternating_world_batch_size(path), 8192)

    def test_partial_batch_preserves_world_schedules(self):
        """Keep heterogeneous iterations and intervals intact across graph replays."""
        if not wp.get_cuda_device_count():
            self.skipTest("Requires the CUDA matrix-free inverse path")
        device = wp.get_cuda_devices()[0]
        builder = newton.ModelBuilder()
        builder.replicate(builder=basics.build_boxes_hinged(), world_count=3)
        model = ModelKamino.from_newton(builder.finalize(device=device))
        model, data, state, limits, detector, jacobians = make_containers(
            model=model, max_world_contacts=8, sparse=True
        )
        update_containers(model=model, data=data, state=state, limits=limits, detector=detector, jacobians=jacobians)
        self.assertTrue(np.all(detector.contacts.world_active_contacts.numpy() > 0))
        problem = _make_sparse_dual_problem(model, data, limits, detector.contacts, jacobians)
        configs = [
            kamino_config.DVISolverConfig(
                tolerance=0.0,
                regularization=1.0e-5,
                max_alternating_iterations=iterations,
                inequality_sweeps_per_iteration=sweeps,
                bilateral_solve_interval=interval,
            )
            for iterations, sweeps, interval in ((1, 1, 1), (4, 1, 2), (3, 2, 99))
        ]
        with mock.patch("newton._src.solvers.kamino._src.solvers.dvi.solver._MAX_CACHED_BILATERAL_COUPLING_ENTRIES", 0):
            solvers = [
                DVISolver(
                    model=model,
                    data=data,
                    limits=limits,
                    contacts=detector.contacts,
                    jacobians=jacobians,
                    problem=problem,
                    config=configs,
                    warmstart=WarmStartMode.NONE,
                )
                for _ in range(2)
            ]
            for solver in solvers:
                self.assertTrue(sparse_dvi._can_use_fused_bilateral_inverse(solver._sparse_path))

            def solve(solver, batch_size):
                with mock.patch.object(
                    sparse_dvi, "_sparse_alternating_world_batch_size", return_value=batch_size
                ) as select_batch:
                    solver.coldstart()
                    solver.solve(problem)
                    select_batch.assert_called()

            def check_results():
                for field in ("lambdas", "v_plus"):
                    expected, actual = [getattr(solver.data.solution, field).numpy() for solver in solvers]
                    self.assertTrue(np.all(np.isfinite(actual)))
                    np.testing.assert_allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)
                expected_status, actual_status = [solver.data.status.numpy() for solver in solvers]
                for field in ("iterations", "converged"):
                    np.testing.assert_array_equal(actual_status[field], expected_status[field])
                np.testing.assert_array_equal(actual_status["iterations"], [1, 4, 6])
                for field in ("r_b", "r_d", "r_c"):
                    np.testing.assert_allclose(actual_status[field], expected_status[field], atol=1.0e-6, rtol=1.0e-6)
                for solver in solvers:
                    np.testing.assert_array_equal(
                        solver.data.state.bilateral_active_dim.numpy(), problem.data.njc.numpy()
                    )

            # Compile and check eager solves before capture. The second batch
            # contains only the third world, whose bilateral interval never fires.
            for _ in range(2):
                for solver, batch_size in zip(solvers, (3, 2), strict=True):
                    solve(solver, batch_size)
                check_results()

            graphs = []
            for solver, batch_size in zip(solvers, (3, 2), strict=True):
                with wp.ScopedCapture(device) as capture:
                    solve(solver, batch_size)
                graphs.append(capture.graph)

            initial_v_f = problem.data.v_f.numpy()
            world_offsets = problem.data.vio.numpy()
            world_dims = problem.data.dim.numpy()
            for replay in range(2):
                # New inputs on replay expose stale or incorrectly offset state.
                v_f = initial_v_f.copy()
                for world, (offset, dim) in enumerate(zip(world_offsets, world_dims, strict=True)):
                    v_f[offset : offset + dim] *= 1.0 + 0.1 * (world + 1) * (replay + 1)
                problem.data.v_f.assign(v_f)
                for graph in graphs:
                    wp.capture_launch(graph)
                check_results()
