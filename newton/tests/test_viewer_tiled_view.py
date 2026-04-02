# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton.viewer import ViewerNull


class TestViewerTiledView(unittest.TestCase):
    def test_tiled_view_defaults_to_false(self):
        viewer = ViewerNull(num_frames=1)
        self.assertFalse(viewer.tiled_view)

    def test_tiled_view_toggle_with_multi_world_model(self):
        builder = newton.ModelBuilder()
        world = newton.ModelBuilder()
        world.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        world.add_shape_box(body=0, hx=0.5, hy=0.5, hz=0.5)
        builder.replicate(world, world_count=4)
        model = builder.finalize()

        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)

        viewer.tiled_view = True
        self.assertTrue(viewer.tiled_view)

        viewer.tiled_view = False
        self.assertFalse(viewer.tiled_view)

    def test_tiled_view_resets_on_clear_model(self):
        viewer = ViewerNull(num_frames=1)
        viewer.tiled_view = True
        self.assertTrue(viewer.tiled_view)

        viewer.clear_model()
        self.assertFalse(viewer.tiled_view)

    def test_last_state_stored_on_log_state(self):
        builder = newton.ModelBuilder()
        world = newton.ModelBuilder()
        world.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
        )
        world.add_shape_box(body=0, hx=0.5, hy=0.5, hz=0.5)
        builder.replicate(world, world_count=2)
        model = builder.finalize()
        state = model.state()

        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)
        self.assertIsNone(viewer._last_state)

        viewer.log_state(state)
        self.assertIs(viewer._last_state, state)


if __name__ == "__main__":
    unittest.main()
