# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices
from newton.viewer import ViewerNull


def _build_multi_world_model(world_count, device=None):
    """Create a simple multi-world model for testing."""
    builder = newton.ModelBuilder()
    world = newton.ModelBuilder()
    world.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        label="test_body",
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    world.add_shape(
        body=0,
        type=newton.GeoType.BOX,
        scale=wp.vec3(0.5, 0.5, 0.5),
        cfg=cfg,
    )
    builder.replicate(world, world_count)
    return builder.finalize(device=device)


class TestViewerVisibleWorlds(unittest.TestCase):
    def test_set_visible_worlds_filters_shapes(self):
        """Only shapes from visible worlds are populated."""
        model = _build_multi_world_model(4)
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)

        # All 4 worlds visible by default
        total_all = sum(len(b.scales) for b in viewer._shape_instances.values())
        self.assertEqual(total_all, 4)

        # Restrict to worlds 0 and 2
        viewer.set_visible_worlds([0, 2])
        total_filtered = sum(len(b.scales) for b in viewer._shape_instances.values())
        self.assertEqual(total_filtered, 2)

    def test_set_visible_worlds_none_shows_all(self):
        """Passing None restores all worlds."""
        model = _build_multi_world_model(4)
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)

        viewer.set_visible_worlds([0])
        self.assertEqual(sum(len(b.scales) for b in viewer._shape_instances.values()), 1)

        viewer.set_visible_worlds(None)
        self.assertEqual(sum(len(b.scales) for b in viewer._shape_instances.values()), 4)

    def test_runtime_switching(self):
        """Visible worlds can be changed between frames."""
        model = _build_multi_world_model(4)
        state = model.state()
        viewer = ViewerNull(num_frames=10)
        viewer.set_model(model)

        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()

        viewer.set_visible_worlds([1, 3])
        total = sum(len(b.scales) for b in viewer._shape_instances.values())
        self.assertEqual(total, 2)

        viewer.begin_frame(0.1)
        viewer.log_state(state)
        viewer.end_frame()

        # Switch again
        viewer.set_visible_worlds([0])
        total = sum(len(b.scales) for b in viewer._shape_instances.values())
        self.assertEqual(total, 1)

    def test_compact_world_offsets(self):
        """Visible worlds get compact grid positions."""
        model = _build_multi_world_model(8)
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)

        viewer.set_visible_worlds([0, 4, 7])
        viewer.set_world_offsets((10.0, 0.0, 0.0))

        offsets = viewer.world_offsets.numpy()

        # Only 3 visible worlds -> compact 1D grid with spacing 10
        visible_offsets = offsets[[0, 4, 7]]
        expected = np.array([[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        assert_np_equal(visible_offsets, expected, tol=1e-5)

        # Non-visible worlds should have zero offset
        for w in [1, 2, 3, 5, 6]:
            assert_np_equal(offsets[w], np.array([0.0, 0.0, 0.0]), tol=1e-5)

    def test_max_worlds_backward_compat(self):
        """max_worlds param in set_model still works with deprecation warning."""
        model = _build_multi_world_model(4)

        viewer = ViewerNull(num_frames=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            viewer.set_model(model, max_worlds=2)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

        total = sum(len(b.scales) for b in viewer._shape_instances.values())
        self.assertEqual(total, 2)

    def test_visible_worlds_mask(self):
        """Internal mask array is correctly built."""
        model = _build_multi_world_model(4)
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)

        viewer.set_visible_worlds([1, 3])
        mask = viewer._visible_worlds_mask.numpy()
        np.testing.assert_array_equal(mask, [0, 1, 0, 1])

        viewer.set_visible_worlds(None)
        self.assertIsNone(viewer._visible_worlds_mask)

    def test_requires_model(self):
        """set_visible_worlds raises without a model."""
        viewer = ViewerNull(num_frames=1)
        with self.assertRaises(RuntimeError):
            viewer.set_visible_worlds([0])

    def test_geometry_cache_preserved(self):
        """Geometry cache is not cleared when switching visible worlds."""
        model = _build_multi_world_model(4)
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)

        cache_before = dict(viewer._geometry_cache)
        viewer.set_visible_worlds([0, 1])
        cache_after = dict(viewer._geometry_cache)

        self.assertEqual(cache_before, cache_after)

    def test_user_spacing_preserved(self):
        """User-provided spacing is reapplied when visible worlds change."""
        model = _build_multi_world_model(4)
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)

        viewer.set_world_offsets((10.0, 0.0, 0.0))
        viewer.set_visible_worlds([0, 2])

        # Only 2 visible worlds with spacing 10 -> compact 1D offsets
        offsets = viewer.world_offsets.numpy()
        visible_offsets = offsets[[0, 2]]
        expected = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        assert_np_equal(visible_offsets, expected, tol=1e-5)


def test_visible_worlds_transforms(test: TestViewerVisibleWorlds, device):
    """Verify shape transforms are correct for visible world subset."""
    model = _build_multi_world_model(4, device=device)
    state = model.state()

    viewer = ViewerNull(num_frames=1)
    viewer.set_model(model)
    viewer.set_visible_worlds([0, 2])
    viewer.set_world_offsets((10.0, 0.0, 0.0))

    viewer.begin_frame(0.0)
    for shapes in viewer._shape_instances.values():
        shapes.update(state, world_offsets=viewer.world_offsets)

    world_xforms = []
    for shapes in viewer._shape_instances.values():
        world_xforms.append(shapes.world_xforms.numpy())
    xforms = np.concatenate(world_xforms, axis=0)

    # 2 visible shapes, body at (0,0,1), offsets at -5 and +5
    positions = xforms[:, :3]
    expected = np.array([[-5.0, 0.0, 1.0], [5.0, 0.0, 1.0]])
    assert_np_equal(positions, expected, tol=1e-4)


devices = get_test_devices()
add_function_test(
    TestViewerVisibleWorlds, "test_visible_worlds_transforms", test_visible_worlds_transforms, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
