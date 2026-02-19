# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer.picking import Picking
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


def _make_single_sphere_model(device=None):
    """Model with one body and one sphere at origin (radius 0.5)."""
    builder = newton.ModelBuilder()
    builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        key="body",
    )
    builder.add_shape_sphere(body=0, radius=0.5)
    return builder.finalize(device=device)


def _make_model_no_shapes(device=None):
    """Model with one body and no shapes (shape_count == 0)."""
    builder = newton.ModelBuilder()
    builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        key="body",
    )
    return builder.finalize(device=device)


class TestPickingSetup(unittest.TestCase):
    """Tests for the Picking setup (construction, release, pick, update, apply_force)."""

    def test_init_state(self):
        """Picking initializes with no body picked and correct stiffness/damping."""
        model = _make_single_sphere_model(device="cpu")
        picking = Picking(model, pick_stiffness=100.0, pick_damping=10.0)

        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)
        self.assertEqual(picking.pick_stiffness, 100.0)
        self.assertEqual(picking.pick_damping, 10.0)
        self.assertIsNotNone(picking.pick_state)
        self.assertEqual(picking.pick_state.shape[0], 1)

    def test_release_clears_state(self):
        """release() clears pick_body and sets picking_active to False."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        # Ray from above origin going down hits the sphere
        ray_start = wp.vec3(0.0, 0.0, 2.0)
        ray_dir = wp.vec3(0.0, 0.0, -1.0)
        picking.pick(state, ray_start, ray_dir)
        self.assertTrue(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], 0)

        picking.release()
        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_pick_miss_remains_inactive(self):
        """pick() with a ray that misses all geometry leaves picking inactive."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        # Ray far from the sphere
        ray_start = wp.vec3(10.0, 10.0, 0.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)

        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_pick_hit_activates_picking(self):
        """pick() with a ray that hits the sphere activates picking and sets pick_body."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        # Ray from -Z toward origin hits the sphere (center at origin, radius 0.5)
        ray_start = wp.vec3(0.0, 0.0, -2.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)

        self.assertTrue(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], 0)
        self.assertGreater(picking.pick_dist, 0.0)
        self.assertLess(picking.pick_dist, 1.0e10)

    def test_pick_empty_model_no_crash(self):
        """pick() with a model that has no shapes returns without error."""
        model = _make_model_no_shapes(device="cpu")
        state = model.state()
        picking = Picking(model)

        ray_start = wp.vec3(0.0, 0.0, -2.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)

        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_update_when_not_picking_no_op(self):
        """update() when not picking does not change state."""
        model = _make_single_sphere_model(device="cpu")
        picking = Picking(model)

        self.assertFalse(picking.is_picking())
        picking.update(wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0))
        self.assertFalse(picking.is_picking())
        self.assertEqual(picking.pick_body.numpy()[0], -1)

    def test_apply_picking_force_when_not_picking(self):
        """_apply_picking_force() when not picking runs kernel without modifying body_f."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        state.body_f.zero_()
        picking._apply_picking_force(state)
        wp.synchronize()

        # No body picked -> no force applied
        forces = state.body_f.numpy()
        assert_np_equal(forces, np.zeros_like(forces), tol=1e-9)

    def test_apply_picking_force_when_picking(self):
        """_apply_picking_force() when picking runs; force is non-zero after update() moves target."""
        model = _make_single_sphere_model(device="cpu")
        state = model.state()
        picking = Picking(model)

        # Activate picking with a hit (target at hit point on sphere)
        ray_start = wp.vec3(0.0, 0.0, -2.0)
        ray_dir = wp.vec3(0.0, 0.0, 1.0)
        picking.pick(state, ray_start, ray_dir)
        self.assertTrue(picking.is_picking())

        # Move target by updating with a ray offset from center so target != attachment point
        picking.update(wp.vec3(0.5, 0.0, -2.0), wp.vec3(0.0, 0.0, 1.0))
        state.body_f.zero_()
        picking._apply_picking_force(state)
        wp.synchronize()

        forces = state.body_f.numpy()
        self.assertEqual(forces.shape[0], model.body_count)
        self.assertFalse(np.allclose(forces[0], np.zeros(6), atol=1e-9))

    def test_world_offsets_optional(self):
        """Picking can be constructed with optional world_offsets."""
        model = _make_single_sphere_model(device="cpu")
        picking = Picking(model, world_offsets=None)
        self.assertIsNone(picking.world_offsets)

        offsets = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=model.device)
        picking_with_offsets = Picking(model, world_offsets=offsets)
        self.assertIsNotNone(picking_with_offsets.world_offsets)
        self.assertEqual(picking_with_offsets.world_offsets.shape[0], 1)


def test_picking_setup_device(test: TestPickingSetup, device):
    """Picking setup works on the given device (CPU or CUDA)."""
    model = _make_single_sphere_model(device=device)
    state = model.state()
    picking = Picking(model)

    test.assertFalse(picking.is_picking())
    test.assertEqual(picking.pick_body.numpy()[0], -1)

    # Hit the sphere
    ray_start = wp.vec3(0.0, 0.0, -2.0)
    ray_dir = wp.vec3(0.0, 0.0, 1.0)
    picking.pick(state, ray_start, ray_dir)

    test.assertTrue(picking.is_picking())
    test.assertEqual(picking.pick_body.numpy()[0], 0)

    # update and apply_force should not crash
    picking.update(ray_start, ray_dir)
    picking._apply_picking_force(state)
    wp.synchronize()

    picking.release()
    test.assertFalse(picking.is_picking())
    test.assertEqual(picking.pick_body.numpy()[0], -1)


# Add device-parameterized test
add_function_test(
    TestPickingSetup,
    "test_picking_setup_device",
    test_picking_setup_device,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
