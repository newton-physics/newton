# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check compact effort elimination and preservation of rejected-world solves."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi.effort_limit_condensation import EffortLimitCondensation
from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import make_compact_schur_pgs_kernel
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIConfigStruct, DVIStatus


def _fixture(device, efforts, remaining):
    worlds, n, vector_stride, schedule_capacity = 4, 64, 135, 64
    non_effort, nc = remaining % 3, remaining // 3
    nb, nu = efforts + non_effort, efforts + remaining
    other = np.linspace(1, nb - 2, non_effort, dtype=int)
    effort_ids = sorted(set(range(nb)) - set(other))
    rng = np.random.default_rng(34791 + nu)

    def ints(values):
        return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

    def floats(values):
        return wp.array(np.asarray(values, dtype=np.float32).ravel(), dtype=wp.float32, device=device)

    def constant(value):
        return ints(np.full(worlds, value))

    vio = np.arange(worlds) * vector_stride + 3
    mio = np.arange(worlds) * 4099 + 5
    operator = np.zeros(int(mio[-1] + 4096), dtype=np.float32)
    initial = np.zeros(worlds * vector_stride + 3, dtype=np.float32)
    gradient, diagonal = initial.copy(), initial.copy()
    target = np.empty((worlds, nu))
    for world in range(worlds):
        dense = rng.normal(0.02, 0.02, (nu, nu))
        dense = dense @ dense.T + np.eye(nu)
        value = rng.uniform(-0.1, 0.1, nu)
        value[nb:] = np.tile([0.0, 0.0, 0.3], nc)
        seed = rng.uniform(-0.05, 0.05, nu)
        seed[nb:] = np.tile([0.0, 0.0, 0.1], nc)
        offset = vio[world] + n
        initial[offset : offset + nu] = seed
        gradient[offset : offset + nu] = dense @ (seed - value)
        diagonal[offset : offset + nu] = np.diag(dense)
        operator[mio[world] : mio[world] + nu * nu] = -dense.T.ravel()
        target[world] = value
    slots = nb + nc
    ids = np.zeros((worlds, schedule_capacity), dtype=np.int32)
    groups = np.zeros((worlds, schedule_capacity + 1), dtype=np.int32)
    colors = groups.copy()
    for world in range(worlds):
        ids[world, :slots] = rng.permutation(slots)
        groups[world, : slots + 1] = np.arange(slots + 1)
        colors[world, :3] = [0, slots // 2, slots]
    config = DVIConfigStruct()
    config.max_alternating_iterations = 32
    config.inequality_sweeps_per_iteration = 2
    config.regularization = 1e-6
    config.omega = 1.0
    config.tolerance = 1e-5
    status = DVIStatus()
    status.r_p = 0.125
    status.iterations = 7
    inputs = [
        ints([0]),
        ints(np.zeros(worlds * nc)),
        constant(nb),
        constant(0),
        constant(nc),
        ints(np.arange(worlds) * nb),
        constant(0),
        ints(np.arange(worlds) * nc),
        ints(np.arange(worlds) * schedule_capacity),
        constant(n),
        constant(n + nb),
        constant(n + nb),
        ints(vio),
        floats(np.full(worlds * nc, 0.5)),
        floats(np.full(worlds * nb, -10.0)),
        floats(np.full(worlds * nb, 10.0)),
        floats(np.ones_like(initial)),
        floats(np.zeros_like(initial)),
        floats(diagonal),
        floats(diagonal),
        constant(n),
        ints(mio),
        constant(64),
        floats(operator),
        floats(gradient),
        constant(2),
        ints(ids.ravel()),
        ints(colors.ravel()),
        ints(groups.ravel()),
        wp.array([config] * worlds, dtype=DVIConfigStruct, device=device),
        wp.array([status] * worlds, dtype=DVIStatus, device=device),
        floats(initial),
    ]
    helper = EffortLimitCondensation([effort_ids] * worlds, device=device, schedule_capacity=schedule_capacity)
    return helper, inputs, target, effort_ids, nu


class TestKaminoEffortLimitCondensation(unittest.TestCase):
    def setUp(self):
        if not wp.get_cuda_device_count():
            self.skipTest("Effort limit elimination uses CUDA tile kernels")
        self.device = wp.get_cuda_devices()[0]

    def launch(self, inputs):
        wp.launch(make_compact_schur_pgs_kernel(64), dim=128, inputs=inputs, block_dim=32, device=self.device)

    def test_static_fallback(self):
        """Keep small batches, short sweeps, and models without efforts on the original path."""
        for worlds, iterations, sweeps, efforts in ((4, 32, 2, 12), (4096, 16, 1, 16), (8192, 32, 2, 0)):
            with self.subTest(worlds=worlds, iterations=iterations, sweeps=sweeps, efforts=efforts):
                path = SimpleNamespace(
                    device=self.device,
                    size=SimpleNamespace(num_worlds=worlds, sum_of_num_effort_joint_cts=efforts),
                    use_schur_complement=True,
                    data=SimpleNamespace(bilateral_operator=object()),
                    max_alternating_iterations=iterations,
                    max_inequality_sweeps_per_iteration=sweeps,
                )
                self.assertIsNone(EffortLimitCondensation.create(path))

    def test_dense_stationary_solution(self):
        """Recover a known full-system solution with ragged and full effort/response tiles."""
        for efforts, remaining in ((2, 5), (1, 31), (16, 32)):
            with self.subTest(efforts=efforts, remaining=remaining):
                helper, inputs, target, _, nu = _fixture(self.device, efforts, remaining)
                initial, gradient = inputs[31].numpy(), inputs[24].numpy()
                fallback = helper.prepare_candidates(inputs)
                np.testing.assert_array_equal(inputs[31].numpy(), initial)
                np.testing.assert_array_equal(inputs[24].numpy(), gradient)
                np.testing.assert_array_equal(fallback.numpy(), np.zeros(4))
                staged_iterations = helper._overrides[30].numpy()["iterations"]
                helper.publish(inputs)
                np.testing.assert_array_equal(inputs[30].numpy()["iterations"], staged_iterations)
                np.testing.assert_array_equal(inputs[30].numpy()["r_p"], np.full(4, 0.125))
                solution, residual = inputs[31].numpy(), inputs[24].numpy()
                for world, offset in enumerate(inputs[12].numpy() + 64):
                    np.testing.assert_allclose(solution[offset : offset + nu], target[world], atol=3e-6, rtol=3e-5)
                    np.testing.assert_allclose(residual[offset : offset + nu], 0.0, atol=3e-6)

    def test_publish_staged_iterations(self):
        """Publish per-world staged sweep counts while preserving other status fields."""
        helper, inputs, _, _, _ = _fixture(self.device, 2, 5)
        helper.prepare_candidates(inputs)
        np.testing.assert_array_equal(helper.eligible.numpy(), np.ones(4))
        np.testing.assert_array_equal(helper.failure.numpy(), np.zeros(4))
        staged_status = helper._overrides[30].numpy()
        staged_status["iterations"] = [1, 3, 8, 64]
        helper._overrides[30].assign(staged_status)
        expected = inputs[30].numpy()
        expected["iterations"] = staged_status["iterations"]
        helper.publish(inputs)
        np.testing.assert_array_equal(inputs[30].numpy(), expected)

    def test_mixed_fallback_graph_replay(self):
        """Preserve original PGS on bound rejection and changing contacts, then reset acceptance."""
        helper, inputs, _, efforts, nu = _fixture(self.device, 2, 5)
        seed_x, seed_q = wp.clone(inputs[31]), wp.clone(inputs[24])
        reference = list(inputs)
        reference[31], reference[24], reference[30] = wp.clone(seed_x), wp.clone(seed_q), wp.clone(inputs[30])
        candidate_x, candidate_q = wp.clone(seed_x), wp.clone(seed_q)

        def solve():
            wp.copy(inputs[31], seed_x)
            wp.copy(inputs[24], seed_q)
            wp.copy(reference[31], seed_x)
            wp.copy(reference[24], seed_q)
            self.launch(reference)
            fallback = helper.prepare_candidates(inputs)
            wp.copy(candidate_x, inputs[31])
            wp.copy(candidate_q, inputs[24])
            masked = list(inputs)
            masked[4] = fallback
            self.launch(masked)
            helper.publish(inputs)

        solve()
        with wp.ScopedCapture(device=self.device) as capture:
            solve()
        original_lower, original_upper, original_nc = inputs[14].numpy(), inputs[15].numpy(), inputs[4].numpy()
        for rejected in (True, False, True, False):
            lower, upper, contacts = original_lower.copy(), original_upper.copy(), original_nc.copy()
            if rejected:
                lower[efforts[0]], upper[efforts[0]] = 1.0, 2.0
                contacts[1] = 0
            inputs[14].assign(lower)
            inputs[15].assign(upper)
            inputs[4].assign(contacts)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(candidate_x.numpy(), seed_x.numpy())
            np.testing.assert_array_equal(candidate_q.numpy(), seed_q.numpy())
            failure, eligible = helper.failure.numpy(), helper.eligible.numpy()
            if rejected:
                self.assertNotEqual(failure[0], 0)
                self.assertEqual(eligible[1], 0)
                np.testing.assert_array_equal(inputs[30].numpy()[:2], reference[30].numpy()[:2])
                for array, expected in ((inputs[31], reference[31]), (inputs[24], reference[24])):
                    for offset in inputs[12].numpy()[:2] + 64:
                        np.testing.assert_array_equal(
                            array.numpy()[offset : offset + nu], expected.numpy()[offset : offset + nu]
                        )
            else:
                np.testing.assert_array_equal(failure, np.zeros(4))
                np.testing.assert_array_equal(eligible, np.ones(4))
                np.testing.assert_array_equal(helper.fallback_nc.numpy(), np.zeros(4))

    def test_invalid_factor_preserves_inputs(self):
        """Reject negative and nonfinite effort pivots without publishing or retaining stale flags."""
        helper, inputs, _, efforts, nu = _fixture(self.device, 2, 5)
        matrix, initial, gradient = inputs[23].numpy(), inputs[31].numpy(), inputs[24].numpy()
        for bad in (True, False):
            changed = matrix.copy()
            if bad:
                for world, value in ((0, 100.0), (1, np.nan)):
                    changed[inputs[21].numpy()[world] + efforts[0] * (nu + 1)] = value
            inputs[23].assign(changed)
            helper.prepare_candidates(inputs)
            np.testing.assert_array_equal(inputs[31].numpy(), initial)
            np.testing.assert_array_equal(inputs[24].numpy(), gradient)
            if bad:
                self.assertTrue(np.all(helper.failure.numpy()[:2] != 0))
            else:
                np.testing.assert_array_equal(helper.failure.numpy(), np.zeros(4))


if __name__ == "__main__":
    unittest.main()
