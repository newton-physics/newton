# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for LOX colored Gauss--Seidel projection."""

import unittest
from itertools import product
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import mat36f, mat66f, vec6f
from newton._src.solvers.kamino._src.solvers.lox.colored_gauss_seidel import (
    ColoredGaussSeidelProjection,
    _ColorFamily,
)
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _empty(device, dtype):
    return wp.empty(0, dtype=dtype, device=device)


def _make_adapter(device, endpoints):
    count = len(endpoints)
    empty_int = _empty(device, wp.int32)
    empty_vec6 = _empty(device, vec6f)
    contact_jacobian = np.zeros((count, 3, 6), dtype=np.float32)
    contact_jacobian[:, 0, 1] = 1.0
    contact_jacobian[:, 1, 2] = 1.0
    contact_jacobian[:, 2, 0] = 1.0
    contact_jacobian_first = wp.array(contact_jacobian, dtype=mat36f, device=device)
    contact_jacobian_second = wp.zeros(count, dtype=mat36f, device=device)
    body_count = max((max(pair) for pair in endpoints), default=-1) + 1
    return SimpleNamespace(
        device=device,
        body_constraint_count=wp.zeros(body_count, dtype=wp.int32, device=device),
        static_body_constraint_count=wp.zeros(body_count, dtype=wp.int32, device=device),
        friction_capacity=0,
        friction_world=empty_int,
        friction_local=empty_int,
        world_friction_count=wp.zeros(1, dtype=wp.int32, device=device),
        friction_body_first=empty_int,
        friction_body_second=empty_int,
        friction_jacobian_first=empty_vec6,
        friction_jacobian_second=empty_vec6,
        friction_impulse_bound=_empty(device, wp.float32),
        friction_projection_delassus=_empty(device, wp.float32),
        friction_reaction=_empty(device, wp.float32),
        contact_capacity=count,
        contact_world=wp.zeros(count, dtype=wp.int32, device=device),
        contact_local=wp.array(np.arange(count, dtype=np.int32), dtype=wp.int32, device=device),
        world_contact_count=wp.array([count], dtype=wp.int32, device=device),
        contact_body_first=wp.array([pair[0] for pair in endpoints], dtype=wp.int32, device=device),
        contact_body_second=wp.array([pair[1] for pair in endpoints], dtype=wp.int32, device=device),
        contact_jacobian_first=contact_jacobian_first,
        contact_jacobian_second=contact_jacobian_second,
        contact_bias=wp.zeros(count, dtype=wp.vec3f, device=device),
        contact_friction=wp.zeros(count, dtype=wp.float32, device=device),
        contact_projection_delassus=wp.zeros(count, dtype=wp.mat33f, device=device),
        contact_reaction=wp.zeros(count, dtype=wp.vec3f, device=device),
        limit_capacity=0,
        limit_world=empty_int,
        limit_local=empty_int,
        world_limit_count=wp.zeros(1, dtype=wp.int32, device=device),
        limit_body_first=empty_int,
        limit_body_second=empty_int,
        limit_jacobian_first=empty_vec6,
        limit_jacobian_second=empty_vec6,
        limit_bias=_empty(device, wp.float32),
        limit_projection_delassus=_empty(device, wp.float32),
        limit_reaction=_empty(device, wp.float32),
    )


class TestLOXColoredGaussSeidel(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_colors_propagate_updates_sequentially(self):
        """Expose each completed color's velocity update to the following color."""
        adapter = _make_adapter(self.device, [(0, -1), (0, -1)])
        adapter.contact_bias.assign([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
        projection = ColoredGaussSeidelProjection(adapter, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)
        prepared_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        projection.prepare(inverse_weight, prepared_status)
        self.assertEqual(len(np.unique(projection.contact.colors.numpy())), 2)

        projected_twist = wp.zeros(1, dtype=vec6f, device=self.device)
        projection_status = wp.zeros(1, dtype=wp.int32, device=self.device)
        projection.project(
            1,
            wp.ones(1, dtype=wp.bool, device=self.device),
            wp.zeros(1, dtype=wp.int32, device=self.device),
            inverse_weight,
            projected_twist,
            wp.zeros(1, dtype=vec6f, device=self.device),
            prepared_status,
            projection_status,
        )
        np.testing.assert_allclose(projected_twist.numpy()[0, 0], 2.0, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(adapter.contact_reaction.numpy()[:, 2], [0.0, 2.0], rtol=0.0, atol=2.0e-6)

    def test_projection_clears_scratch_across_consecutive_calls(self):
        """Clear stale body deltas after valid and rejected projection calls."""
        adapter = _make_adapter(self.device, [(0, -1)])
        adapter.contact_bias.assign([[0.0, 0.0, -1.0]])
        projection = ColoredGaussSeidelProjection(adapter, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=self.device)
        prepared_status = wp.ones(1, dtype=wp.int32, device=self.device)
        projection.prepare(inverse_weight, prepared_status)
        twist_delta = wp.full(1, vec6f(3.0), dtype=vec6f, device=self.device)
        projected_twist = wp.zeros(1, dtype=vec6f, device=self.device)
        projection_status = wp.ones(1, dtype=wp.int32, device=self.device)
        arguments = (
            wp.ones(1, dtype=wp.bool, device=self.device),
            wp.zeros(1, dtype=wp.int32, device=self.device),
            inverse_weight,
            projected_twist,
            twist_delta,
            prepared_status,
            projection_status,
        )

        projection.project(1, *arguments)
        np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((1, 6), dtype=np.float32))

        twist_delta.fill_(vec6f(5.0))
        prepared_status.zero_()
        projection.project(1, *arguments)
        np.testing.assert_array_equal(projection_status.numpy(), [0])
        np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((1, 6), dtype=np.float32))

    def test_invalid_world_discards_only_its_color_delta(self):
        """Keep a malformed world's partial update out of every color barrier."""
        adapter = _make_adapter(self.device, [(0, -1), (1, -1)])
        adapter.contact_world.assign([0, 1])
        adapter.contact_local.assign([0, 0])
        adapter.world_friction_count = wp.zeros(2, dtype=wp.int32, device=self.device)
        adapter.world_contact_count = wp.ones(2, dtype=wp.int32, device=self.device)
        adapter.world_limit_count = wp.zeros(2, dtype=wp.int32, device=self.device)
        jacobians = adapter.contact_jacobian_first.numpy()
        jacobians[1] = 0.0
        adapter.contact_jacobian_first.assign(jacobians)
        adapter.contact_bias.assign([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
        projection = ColoredGaussSeidelProjection(adapter, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)] * 2, dtype=mat66f, device=self.device)
        prepared_status = wp.zeros(2, dtype=wp.int32, device=self.device)
        projection.prepare(inverse_weight, prepared_status)
        np.testing.assert_array_equal(prepared_status.numpy(), [1, 0])

        projected_twist = wp.zeros(2, dtype=vec6f, device=self.device)
        projection_status = wp.zeros(2, dtype=wp.int32, device=self.device)
        twist_delta = wp.zeros(2, dtype=vec6f, device=self.device)
        projection.project(
            1,
            wp.ones(2, dtype=wp.bool, device=self.device),
            wp.array([0, 1], dtype=wp.int32, device=self.device),
            inverse_weight,
            projected_twist,
            twist_delta,
            prepared_status,
            projection_status,
        )
        np.testing.assert_allclose(projected_twist.numpy()[:, 0], [1.0, 0.0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(projection_status.numpy(), [1, 0])
        np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((2, 6), dtype=np.float32))

    def test_projection_validates_unconstrained_body(self):
        """Accept finite twists and reject NaN/Inf even on unconstrained bodies."""
        devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))
        for device in devices:
            for fused, value in product([False, True] if device.is_cuda else [False], [2.0e38, np.inf, np.nan]):
                with self.subTest(device=device.alias, fused=fused, value=value):
                    adapter = _make_adapter(device, [(0, -1)])
                    adapter.body_constraint_count = wp.zeros(2, dtype=wp.int32, device=device)
                    adapter.static_body_constraint_count = wp.zeros(2, dtype=wp.int32, device=device)
                    adapter.world_friction_count = wp.zeros(2, dtype=wp.int32, device=device)
                    adapter.world_limit_count = wp.zeros(2, dtype=wp.int32, device=device)
                    adapter.world_contact_count = wp.array([1, 0], dtype=wp.int32, device=device)
                    adapter.contact_bias.assign([[0.0, 0.0, -1.0]])
                    if fused:
                        adapter.model = SimpleNamespace(
                            info=SimpleNamespace(
                                bodies_offset=wp.array([0, 1], dtype=wp.int32, device=device),
                                num_bodies=wp.ones(2, dtype=wp.int32, device=device),
                            )
                        )
                        adapter.world_friction_offset = wp.zeros(2, dtype=wp.int32, device=device)
                        adapter.world_limit_offset = wp.zeros(2, dtype=wp.int32, device=device)
                        adapter.world_contact_offset = wp.array([0, 1], dtype=wp.int32, device=device)
                    projection = ColoredGaussSeidelProjection(adapter, 2)
                    inverse_weight = wp.array([np.eye(6, dtype=np.float32)] * 2, dtype=mat66f, device=device)
                    prepared_status = wp.zeros(2, dtype=wp.int32, device=device)
                    projection.prepare(inverse_weight, prepared_status)
                    velocity = np.zeros((2, 6), dtype=np.float32)
                    velocity[1] = 2.0e38
                    velocity[1, 5] = value
                    projected_twist = wp.array(velocity, dtype=vec6f, device=device)
                    projection_status = wp.zeros(2, dtype=wp.int32, device=device)
                    twist_delta = wp.zeros(2, dtype=vec6f, device=device)
                    projection.project(
                        1,
                        wp.ones(2, dtype=wp.bool, device=device),
                        wp.array([0, 1], dtype=wp.int32, device=device),
                        inverse_weight,
                        projected_twist,
                        twist_delta,
                        prepared_status,
                        projection_status,
                    )
                    np.testing.assert_array_equal(projection_status.numpy(), [1, int(np.isfinite(value))])
                    np.testing.assert_array_equal(projected_twist.numpy()[1], velocity[1])
                    self.assertAlmostEqual(float(projected_twist.numpy()[0, 0]), 1.0)
                    np.testing.assert_array_equal(twist_delta.numpy(), np.zeros((2, 6), dtype=np.float32))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for block synchronization")
    def test_world_colored_matches_global_colored(self):
        """Match the single-launch world traversal to the global colored path."""
        device = wp.get_device("cuda:0")
        world_count = device.sm_count * 4
        world_ids = np.arange(world_count, dtype=np.int32)
        zeros = np.zeros(world_count, dtype=np.int32)
        ones = np.ones(world_count, dtype=np.int32)
        empty_int = _empty(device, wp.int32)
        empty_vec6 = _empty(device, vec6f)
        world_offset = wp.array(world_ids, dtype=wp.int32, device=device)
        zero_world_offset = wp.zeros(world_count, dtype=wp.int32, device=device)
        zero_world_count = wp.zeros(world_count, dtype=wp.int32, device=device)
        contact_jacobian = np.zeros((world_count, 3, 6), dtype=np.float32)
        contact_jacobian[:, 2, 0] = 1.0
        adapter = SimpleNamespace(
            device=device,
            model=SimpleNamespace(
                info=SimpleNamespace(
                    bodies_offset=world_offset,
                    num_bodies=wp.array(ones, dtype=wp.int32, device=device),
                )
            ),
            body_constraint_count=wp.ones(world_count, dtype=wp.int32, device=device),
            static_body_constraint_count=wp.zeros(world_count, dtype=wp.int32, device=device),
            friction_capacity=0,
            friction_world=empty_int,
            friction_local=empty_int,
            world_friction_offset=zero_world_offset,
            world_friction_count=zero_world_count,
            friction_body_first=empty_int,
            friction_body_second=empty_int,
            friction_jacobian_first=empty_vec6,
            friction_jacobian_second=empty_vec6,
            friction_impulse_bound=_empty(device, wp.float32),
            friction_projection_delassus=_empty(device, wp.float32),
            friction_reaction=_empty(device, wp.float32),
            contact_capacity=world_count,
            contact_world=wp.array(world_ids, dtype=wp.int32, device=device),
            contact_local=wp.array(zeros, dtype=wp.int32, device=device),
            world_contact_offset=world_offset,
            world_contact_count=wp.array(ones, dtype=wp.int32, device=device),
            contact_body_first=wp.array(world_ids, dtype=wp.int32, device=device),
            contact_body_second=wp.full(world_count, -1, dtype=wp.int32, device=device),
            contact_jacobian_first=wp.array(contact_jacobian, dtype=mat36f, device=device),
            contact_jacobian_second=wp.zeros(world_count, dtype=mat36f, device=device),
            contact_bias=wp.full(world_count, wp.vec3f(0.0, 0.0, -1.0), dtype=wp.vec3f, device=device),
            contact_friction=wp.full(world_count, 0.4, dtype=wp.float32, device=device),
            contact_projection_delassus=wp.zeros(world_count, dtype=wp.mat33f, device=device),
            contact_reaction=wp.full(
                world_count,
                wp.vec3f(0.0, 0.0, 0.2),
                dtype=wp.vec3f,
                device=device,
            ),
            limit_capacity=0,
            limit_world=empty_int,
            limit_local=empty_int,
            world_limit_offset=zero_world_offset,
            world_limit_count=zero_world_count,
            limit_body_first=empty_int,
            limit_body_second=empty_int,
            limit_jacobian_first=empty_vec6,
            limit_jacobian_second=empty_vec6,
            limit_bias=_empty(device, wp.float32),
            limit_projection_delassus=_empty(device, wp.float32),
            limit_reaction=_empty(device, wp.float32),
        )
        projection = ColoredGaussSeidelProjection(adapter, 2)
        inverse_weight = wp.array(
            np.repeat(np.eye(6, dtype=np.float32)[None, :, :], world_count, axis=0),
            dtype=mat66f,
            device=device,
        )
        prepared_status = wp.zeros(world_count, dtype=wp.int32, device=device)
        projection.prepare(inverse_weight, prepared_status)

        def run(use_world_projection):
            adapter.model.info.bodies_offset = world_offset if use_world_projection else None
            projected_twist = wp.zeros(world_count, dtype=vec6f, device=device)
            twist_delta = wp.zeros(world_count, dtype=vec6f, device=device)
            projection_status = wp.zeros(world_count, dtype=wp.int32, device=device)
            adapter.contact_reaction.fill_(wp.vec3f(0.0, 0.0, 0.2))
            projection.project(
                3,
                wp.ones(world_count, dtype=wp.bool, device=device),
                wp.array(world_ids, dtype=wp.int32, device=device),
                inverse_weight,
                projected_twist,
                twist_delta,
                prepared_status,
                projection_status,
            )
            return projected_twist.numpy(), adapter.contact_reaction.numpy(), projection_status.numpy()

        global_result = run(False)
        world_result = run(True)
        for global_value, world_value in zip(global_result[:-1], world_result[:-1], strict=True):
            np.testing.assert_allclose(world_value, global_value, rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(world_result[-1], global_result[-1])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for graph capture")
    def test_cuda_graph_capture_replays_large_color_compaction(self):
        """Capture the parallel color-prefix scan and cursor copy path."""
        device = wp.get_device("cuda:0")
        color_count = 96
        capacity = 2 * color_count + 11
        colors = np.arange(capacity, dtype=np.int32) % color_count
        colors[::17] = -1
        family = _ColorFamily(capacity, color_count, device)
        family.colors.assign(colors)
        family.compact(color_count, device)

        with wp.ScopedCapture(device=device) as capture:
            family.compact(color_count, device)
        wp.capture_launch(capture.graph)

        valid = colors >= 0
        expected_counts = np.bincount(colors[valid], minlength=color_count).astype(np.int32)
        expected_offsets = np.zeros(color_count, dtype=np.int32)
        expected_offsets[1:] = np.cumsum(expected_counts[:-1], dtype=np.int32)
        counts = family.counts.numpy()
        offsets = family.offsets.numpy()
        order = family.order.numpy()[: int(np.sum(expected_counts))]
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(offsets, expected_offsets)
        np.testing.assert_array_equal(np.sort(order), np.flatnonzero(valid))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for graph capture")
    def test_cuda_graph_capture_replays_coloring_and_projection(self):
        """Capture and replay device coloring, metric preparation, and projection."""
        device = wp.get_device("cuda:0")
        adapter = _make_adapter(device, [(0, -1), (0, -1)])
        adapter.contact_bias.assign([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
        projection = ColoredGaussSeidelProjection(adapter, 2)
        inverse_weight = wp.array([np.eye(6, dtype=np.float32)], dtype=mat66f, device=device)
        prepared_status = wp.zeros(1, dtype=wp.int32, device=device)
        projection_status = wp.zeros(1, dtype=wp.int32, device=device)
        projected_twist = wp.zeros(1, dtype=vec6f, device=device)
        twist_delta = wp.zeros(1, dtype=vec6f, device=device)
        world_active = wp.ones(1, dtype=wp.bool, device=device)
        body_world = wp.zeros(1, dtype=wp.int32, device=device)

        def run():
            projection.prepare(inverse_weight, prepared_status)
            projection.project(
                1,
                world_active,
                body_world,
                inverse_weight,
                projected_twist,
                twist_delta,
                prepared_status,
                projection_status,
            )

        run()
        projected_twist.zero_()
        adapter.contact_reaction.zero_()
        with wp.ScopedCapture(device=device) as capture:
            run()
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(projected_twist.numpy()[0, 0], 2.0, rtol=0.0, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
