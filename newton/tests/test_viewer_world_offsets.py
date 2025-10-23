# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices
from newton.viewer import ViewerNull

wp.config.quiet = True


class TestViewerWorldOffsets(unittest.TestCase):
    def test_compute_world_offsets_function(self):
        """Test that the shared compute_world_offsets function works correctly."""
        # Test basic functionality
        test_cases = [
            (1, (0.0, 0.0, 0.0), [[0.0, 0.0, 0.0]]),
            (1, (5.0, 5.0, 0.0), [[0.0, 0.0, 0.0]]),  # Single world always at origin
            (2, (10.0, 0.0, 0.0), [[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            (4, (5.0, 5.0, 0.0), [[-2.5, -2.5, 0.0], [-2.5, 2.5, 0.0], [2.5, -2.5, 0.0], [2.5, 2.5, 0.0]]),
        ]

        for num_worlds, spacing, expected in test_cases:
            # Test without up_axis
            offsets = newton.utils.compute_world_offsets(num_worlds, spacing)
            assert_np_equal(offsets, np.array(expected), tol=1e-5)

            # Test with up_axis
            offsets_with_up = newton.utils.compute_world_offsets(num_worlds, spacing, up_axis=newton.Axis.Z)
            assert_np_equal(offsets_with_up, np.array(expected), tol=1e-5)

    def test_auto_compute_world_offsets(self):
        """Test that viewer automatically computes world offsets when not explicitly set."""
        num_worlds = 4
        builder = newton.ModelBuilder()

        # Create a simple world with known extents
        world = newton.ModelBuilder()
        # Add a box at origin with size 2x2x2
        world.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            I_m=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            key="test_body",
        )
        world.add_shape_box(
            body=0,
            hx=1.0,
            hy=1.0,
            hz=1.0,
        )

        # Replicate without spacing
        builder.replicate(world, num_worlds)
        model = builder.finalize()

        # Create viewer and set model - should auto-compute offsets
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)

        # Check that world offsets were computed
        assert viewer.world_offsets is not None
        offsets = viewer.world_offsets.numpy()
        assert len(offsets) == num_worlds

        # Verify offsets are reasonable - worlds should be spaced apart
        # The auto-compute should create spacing based on world 0 extents
        # Box has size 2x2x2, so with 1.5x margin, spacing should be around 3.0
        for i in range(1, num_worlds):
            distance = np.linalg.norm(offsets[i] - offsets[0])
            assert distance > 2.0, f"World {i} too close to world 0: distance={distance}"

        # Verify 2D grid arrangement (all Z values should be the same)
        z_values = offsets[:, 2]
        assert np.allclose(z_values, z_values[0]), "Auto-computed offsets should use 2D grid (constant Z)"

        # Test that explicit set_world_offsets overrides auto-computed offsets
        viewer.set_world_offsets(num_worlds, spacing=(10.0, 0.0, 0.0))
        new_offsets = viewer.world_offsets.numpy()
        expected = [[-15.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [15.0, 0.0, 0.0]]
        assert_np_equal(new_offsets, np.array(expected), tol=1e-5)

        # Test with more worlds to verify 2D grid arrangement
        num_worlds_large = 16
        builder_large = newton.ModelBuilder()
        builder_large.replicate(world, num_worlds_large)
        model_large = builder_large.finalize()

        viewer_large = ViewerNull(num_frames=1)
        viewer_large.set_model(model_large)

        # Check 2D grid for 16 worlds (should be 4x4 grid in XY plane)
        offsets_large = viewer_large.world_offsets.numpy()
        z_values_large = offsets_large[:, 2]
        assert np.allclose(z_values_large, z_values_large[0]), "Large grid should also use 2D arrangement"

    def test_physics_at_origin(self):
        """Test that physics simulation runs with all worlds at origin."""
        num_worlds = 4
        builder = newton.ModelBuilder()

        # Create a simple body for each world
        world = newton.ModelBuilder()
        world.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            I_m=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            key="test_body",
        )

        # Replicate with zero spacing (new default)
        builder.replicate(world, num_worlds)
        builder.add_ground_plane()

        model = builder.finalize()
        state = model.state()

        # Verify all bodies are at the same position (no physical offset)
        body_positions = state.body_q.numpy()[:, :3]
        for i in range(1, num_worlds):
            assert_np_equal(
                body_positions[0],
                body_positions[i],
                tol=1e-6,
            )

    def test_viewer_offset_computation(self):
        """Test that viewer computes world offsets correctly."""
        test_cases = [
            (1, (0.0, 0.0, 0.0), [[0.0, 0.0, 0.0]]),
            (1, (5.0, 5.0, 0.0), [[0.0, 0.0, 0.0]]),  # Single world always at origin
            (2, (10.0, 0.0, 0.0), [[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            (4, (5.0, 5.0, 0.0), [[-2.5, -2.5, 0.0], [-2.5, 2.5, 0.0], [2.5, -2.5, 0.0], [2.5, 2.5, 0.0]]),
            # 3D grid case - 8 worlds in a 2x2x2 grid
            # Note: Z-axis correction is 0 to keep worlds above ground
            (
                8,
                (4.0, 4.0, 4.0),
                [
                    [-2.0, -2.0, 0.0],
                    [-2.0, -2.0, 4.0],
                    [-2.0, 2.0, 0.0],
                    [-2.0, 2.0, 4.0],
                    [2.0, -2.0, 0.0],
                    [2.0, -2.0, 4.0],
                    [2.0, 2.0, 0.0],
                    [2.0, 2.0, 4.0],
                ],
            ),
            # Larger 3D grid case - 27 worlds in a 3x3x3 grid
            # Note: Z-axis correction is 0 to keep worlds above ground
            (
                27,
                (2.0, 2.0, 2.0),
                [
                    [-2.0, -2.0, 0.0],
                    [-2.0, -2.0, 2.0],
                    [-2.0, -2.0, 4.0],
                    [-2.0, 0.0, 0.0],
                    [-2.0, 0.0, 2.0],
                    [-2.0, 0.0, 4.0],
                    [-2.0, 2.0, 0.0],
                    [-2.0, 2.0, 2.0],
                    [-2.0, 2.0, 4.0],
                    [0.0, -2.0, 0.0],
                    [0.0, -2.0, 2.0],
                    [0.0, -2.0, 4.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 4.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 2.0, 2.0],
                    [0.0, 2.0, 4.0],
                    [2.0, -2.0, 0.0],
                    [2.0, -2.0, 2.0],
                    [2.0, -2.0, 4.0],
                    [2.0, 0.0, 0.0],
                    [2.0, 0.0, 2.0],
                    [2.0, 0.0, 4.0],
                    [2.0, 2.0, 0.0],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 4.0],
                ],
            ),
        ]

        for num_worlds, spacing, expected in test_cases:
            viewer = ViewerNull(num_frames=1)
            # Set model is required before set_world_offsets
            builder = newton.ModelBuilder()
            model = builder.finalize()
            viewer.set_model(model)

            viewer.set_world_offsets(num_worlds, spacing)

            actual = viewer.world_offsets.numpy()
            assert_np_equal(actual, np.array(expected), tol=1e-5)

    def test_global_entities_unaffected(self):
        """Test that global entities (world -1) are not affected by world offsets."""
        num_worlds = 2
        spacing = (10.0, 0.0, 0.0)

        # Create model with both world-specific and global entities
        builder = newton.ModelBuilder()

        # Add global ground plane (world -1)
        builder.current_world = -1
        builder.add_ground_plane()

        # Add world-specific bodies
        world = newton.ModelBuilder()
        world.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            I_m=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            key="world_body",
        )
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
        world.add_shape(
            body=0,  # Attach to the first (and only) body in world
            type=newton.GeoType.SPHERE,
            scale=wp.vec3(0.5, 0.5, 0.5),
            cfg=cfg,
        )

        builder.replicate(world, num_worlds)

        model = builder.finalize()
        state = model.state()

        # Create viewer and set offsets
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)
        viewer.set_world_offsets(num_worlds, spacing)

        # Find ground plane shape instance (should be static)
        ground_instance = None
        world_instance = None
        for shapes in viewer._shape_instances.values():
            if shapes.static:
                ground_instance = shapes
            else:
                world_instance = shapes

        self.assertIsNotNone(ground_instance, "Ground plane instance not found")
        self.assertIsNotNone(world_instance, "World instance not found")

        # Update transforms
        viewer.begin_frame(0.0)
        ground_instance.update(state, world_offsets=viewer.world_offsets)
        world_instance.update(state, world_offsets=viewer.world_offsets)

        # Check ground plane is at origin (unaffected by offsets)
        ground_xform = ground_instance.world_xforms.numpy()[0]
        assert_np_equal(ground_xform[:3], np.array([0.0, 0.0, 0.0]), tol=1e-5)

        # Check world shapes are offset
        world_xforms = world_instance.world_xforms.numpy()
        expected_offsets = np.array([[-5.0, 0.0, 1.0], [5.0, 0.0, 1.0]])

        for i in range(num_worlds):
            assert_np_equal(world_xforms[i][:3], expected_offsets[i], tol=1e-5)


def test_visual_separation(test: TestViewerWorldOffsets, device):
    """Test that viewer offsets provide visual separation without affecting physics."""
    num_worlds = 4
    spacing = (5.0, 5.0, 0.0)

    # Create model
    builder = newton.ModelBuilder()
    world = newton.ModelBuilder()
    world.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        I_m=wp.mat33(np.eye(3)),
        key="test_body",
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    world.add_shape(
        body=0,  # Attach to the first (and only) body in world
        type=newton.GeoType.BOX,
        scale=wp.vec3(0.5, 0.5, 0.5),
        cfg=cfg,
    )

    builder.replicate(world, num_worlds)
    model = builder.finalize(device=device)
    state = model.state()

    # Create viewer and set offsets
    viewer = ViewerNull(num_frames=1)
    viewer.set_model(model)
    viewer.set_world_offsets(num_worlds, spacing)

    # Get shape instances from viewer
    shape_instances = next(iter(viewer._shape_instances.values()))

    # Update transforms
    viewer.begin_frame(0.0)
    shape_instances.update(state, world_offsets=viewer.world_offsets)

    # Check that world transforms have been offset
    world_xforms = shape_instances.world_xforms.numpy()

    # Expected offsets based on 2x2 grid with spacing (5, 5, 0)
    expected_offsets = np.array(
        [
            [-2.5, -2.5, 0.0],  # env 0
            [-2.5, 2.5, 0.0],  # env 1
            [2.5, -2.5, 0.0],  # env 2
            [2.5, 2.5, 0.0],  # env 3
        ]
    )

    for i in range(num_worlds):
        actual_pos = world_xforms[i][:3]
        expected_pos = expected_offsets[i] + np.array([0.0, 0.0, 1.0])  # body is at (0,0,1)
        assert_np_equal(actual_pos, expected_pos, tol=1e-4)


# Add device-specific tests
devices = get_test_devices()
for device in devices:
    add_function_test(
        TestViewerWorldOffsets,
        f"test_visual_separation_{device.alias}",
        test_visual_separation,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
