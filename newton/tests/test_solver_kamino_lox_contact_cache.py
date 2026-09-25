# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check angular contact matching and cache lifecycle."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.lox.contact_cache import AngularContactCache


def _contacts(keys, positions, worlds=None, bodies=None, frames=None, *, device="cpu"):
    count = len(keys)
    return SimpleNamespace(
        model_active_contacts=wp.array([count], dtype=wp.int32, device=device),
        key=wp.array(keys, dtype=wp.uint64, device=device),
        wid=wp.array(worlds if worlds is not None else [0] * count, dtype=wp.int32, device=device),
        bid_AB=wp.array(bodies if bodies is not None else [(-1, 0)] * count, dtype=wp.vec2i, device=device),
        position_B=wp.array(positions, dtype=wp.vec3f, device=device),
        frame=wp.array(frames if frames is not None else [(0.0, 0.0, 0.0, 1.0)] * count, dtype=wp.quatf, device=device),
    )


class TestAngularContactCache(unittest.TestCase):
    def test_reordering_nearest_point_and_filtered_sources(self):
        """Match the closest point for each key across changed source and solver orders."""
        device = "cpu"
        cache = AngularContactCache(4, device)
        pose = wp.array([wp.transform_identity()], dtype=wp.transformf, device=device)
        dt = wp.array([0.1], dtype=wp.float32, device=device)
        inverse_dt = wp.array([10.0], dtype=wp.float32, device=device)
        old = _contacts([8, 8, 3, 9], [(0, 0, 0), (0.0008, 0, 0), (1, 0, 0), (2, 0, 0)])
        old_map = wp.array([2, 0, 1, -1], dtype=wp.int32, device=device)
        angular = wp.zeros(3, dtype=wp.vec3f, device=device)
        cache.import_reactions(old, old_map, pose, dt, angular)
        angular.assign(np.asarray([(2, 0, 0), (3, 0, 0), (1, 0, 0)], dtype=np.float32))
        cache.export_reactions(old, old_map, inverse_dt, angular)
        current = _contacts([3, 8, 8, 9], [(1, 0, 0), (0.0007, 0, 0), (0.0001, 0, 0), (2, 0, 0)])
        new_map = wp.array([3, 2, 1, 0], dtype=wp.int32, device=device)
        result = wp.full(4, wp.vec3f(99.0), dtype=wp.vec3f, device=device)
        cache.import_reactions(current, new_map, pose, dt, result)
        np.testing.assert_allclose(result.numpy(), [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])

    def test_frames_moving_body_and_timestep(self):
        """Preserve world torque while adapting the frame, body motion, and timestep."""
        cache = AngularContactCache(1, "cpu")
        initial_pose = wp.array([wp.transform_identity()], dtype=wp.transformf, device="cpu")
        mapping = wp.array([0], dtype=wp.int32, device="cpu")
        dt = wp.array([0.1], dtype=wp.float32, device="cpu")
        inverse_dt = wp.array([10.0], dtype=wp.float32, device="cpu")
        angular = wp.zeros(1, dtype=wp.vec3f, device="cpu")
        old = _contacts([7], [(1, 0, 0)])
        cache.import_reactions(old, mapping, initial_pose, dt, angular)
        angular.assign(np.asarray([(2, 3, 4)], dtype=np.float32))
        cache.export_reactions(old, mapping, inverse_dt, angular)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.5 * np.pi)
        moved_pose = wp.array([wp.transform(wp.vec3(5, 0, 0), rotation)], dtype=wp.transformf, device="cpu")
        current = _contacts([7], [(5, 1, 0)], frames=[rotation])
        dt.assign(np.asarray([0.2], dtype=np.float32))
        cache.import_reactions(current, mapping, moved_pose, dt, angular)
        np.testing.assert_allclose(angular.numpy(), [(4, 8, -6)], atol=2.0e-6)

    def test_world_reset_separation_and_disappearance(self):
        """Reject other worlds and separated points and remove disappeared contacts."""
        cache = AngularContactCache(2, "cpu")
        pose = wp.array([wp.transform_identity()], dtype=wp.transformf, device="cpu")
        mapping = wp.array([0, 1], dtype=wp.int32, device="cpu")
        dt = wp.array([1.0, 1.0], dtype=wp.float32, device="cpu")
        contacts = _contacts([4, 4], [(0, 0, 0), (0, 0, 0)], worlds=[0, 1])
        angular = wp.zeros(2, dtype=wp.vec3f, device="cpu")
        cache.import_reactions(contacts, mapping, pose, dt, angular)
        angular.assign(np.asarray([(1, 0, 0), (2, 0, 0)], dtype=np.float32))
        cache.export_reactions(contacts, mapping, dt, angular)
        cache.reset(wp.array([True, False], dtype=wp.bool, device="cpu"))
        cache.import_reactions(contacts, mapping, pose, dt, angular)
        np.testing.assert_array_equal(angular.numpy(), [(0, 0, 0), (2, 0, 0)])
        contacts.position_B.assign(np.asarray([(0.002, 0, 0), (0.002, 0, 0)], dtype=np.float32))
        cache.import_reactions(contacts, mapping, pose, dt, angular)
        np.testing.assert_array_equal(angular.numpy(), 0.0)
        contacts.model_active_contacts.zero_()
        cache.export_reactions(contacts, mapping, dt, angular)
        contacts.model_active_contacts.fill_(2)
        contacts.position_B.zero_()
        cache.import_reactions(contacts, mapping, pose, dt, angular)
        np.testing.assert_array_equal(angular.numpy(), 0.0)
        cache.reset()

    def test_cuda_graph_replay(self):
        """Replay cache import, export, and partial reset without host readback."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA is unavailable")
        device = wp.get_device("cuda:0")
        cache = AngularContactCache(1, device)
        contacts = _contacts([1], [(0, 0, 0)], device=device)
        pose = wp.array([wp.transform_identity()], dtype=wp.transformf, device=device)
        mapping = wp.array([0], dtype=wp.int32, device=device)
        dt = wp.array([1.0], dtype=wp.float32, device=device)
        mask = wp.array([False], dtype=wp.bool, device=device)
        angular = wp.zeros(1, dtype=wp.vec3f, device=device)
        cache.import_reactions(contacts, mapping, pose, dt, angular)
        angular.fill_(wp.vec3f(1, 2, 3))
        cache.export_reactions(contacts, mapping, dt, angular)
        cache.reset(mask)
        with wp.ScopedCapture(device=device) as capture:
            cache.import_reactions(contacts, mapping, pose, dt, angular)
            cache.export_reactions(contacts, mapping, dt, angular)
            cache.reset(mask)
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(angular.numpy(), [(1, 2, 3)])
        mask.fill_(True)
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(angular.numpy(), 0.0)

    def test_saturated_contact_count(self):
        """Clamp overflowing and negative source counts before sorting and searching."""
        cache = AngularContactCache(2, "cpu")
        contacts = _contacts([1, 2], [(0, 0, 0), (1, 0, 0)])
        mapping = wp.array([0, 1], dtype=wp.int32, device="cpu")
        pose = wp.array([wp.transform_identity()], dtype=wp.transformf, device="cpu")
        dt = wp.array([1.0], dtype=wp.float32, device="cpu")
        angular = wp.zeros(2, dtype=wp.vec3f, device="cpu")
        cache.import_reactions(contacts, mapping, pose, dt, angular)
        angular.assign(np.asarray([(1, 0, 0), (2, 0, 0)], dtype=np.float32))
        contacts.model_active_contacts.fill_(1000)
        cache.export_reactions(contacts, mapping, dt, angular)
        self.assertEqual(int(cache.data.count.numpy()[0]), 2)
        cache.import_reactions(contacts, mapping, pose, dt, angular)
        np.testing.assert_array_equal(angular.numpy(), [(1, 0, 0), (2, 0, 0)])
        contacts.model_active_contacts.fill_(-1)
        cache.export_reactions(contacts, mapping, dt, angular)
        self.assertEqual(int(cache.data.count.numpy()[0]), 0)
        contacts.model_active_contacts.fill_(2)
        cache.import_reactions(contacts, mapping, pose, dt, angular)
        np.testing.assert_array_equal(angular.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
