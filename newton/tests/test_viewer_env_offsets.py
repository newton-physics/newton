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


class TestViewerEnvOffsets(unittest.TestCase):
    def test_physics_at_origin(self):
        """Test that physics simulation runs with all environments at origin."""
        num_envs = 4
        builder = newton.ModelBuilder()

        # Create a simple body for each environment
        env = newton.ModelBuilder()
        env.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            I_m=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            key="test_body",
        )

        # Replicate with zero spacing (new default)
        builder.replicate(env, num_envs)
        builder.add_ground_plane()

        model = builder.finalize()
        state = model.state()

        # Verify all bodies are at the same position (no physical offset)
        body_positions = state.body_q.numpy()[:, :3]
        for i in range(1, num_envs):
            assert_np_equal(
                body_positions[0],
                body_positions[i],
                tol=1e-6,
            )

    def test_viewer_offset_computation(self):
        """Test that viewer computes environment offsets correctly."""
        test_cases = [
            (1, (0.0, 0.0, 0.0), [[0.0, 0.0, 0.0]]),
            (1, (5.0, 5.0, 0.0), [[0.0, 0.0, 0.0]]),  # Single env always at origin
            (2, (10.0, 0.0, 0.0), [[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            (4, (5.0, 5.0, 0.0), [[-2.5, -2.5, 0.0], [-2.5, 2.5, 0.0], [2.5, -2.5, 0.0], [2.5, 2.5, 0.0]]),
            # 3D grid case - 8 environments in a 2x2x2 grid
            # Note: Z-axis correction is 0 to keep environments above ground
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
            # Larger 3D grid case - 27 environments in a 3x3x3 grid
            # Note: Z-axis correction is 0 to keep environments above ground
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

        for num_envs, spacing, expected in test_cases:
            viewer = ViewerNull(num_frames=1)
            # Set model is required before set_env_offsets
            builder = newton.ModelBuilder()
            model = builder.finalize()
            viewer.set_model(model)

            viewer.set_env_offsets(num_envs, spacing)

            actual = viewer.env_offsets.numpy()
            assert_np_equal(actual, np.array(expected), tol=1e-5)

    def test_global_entities_unaffected(self):
        """Test that global entities (group -1) are not affected by environment offsets."""
        num_envs = 2
        spacing = (10.0, 0.0, 0.0)

        # Create model with both environment-specific and global entities
        builder = newton.ModelBuilder()

        # Add global ground plane (group -1)
        builder.current_env_group = -1
        builder.add_ground_plane()

        # Add environment-specific bodies
        env = newton.ModelBuilder()
        env.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            mass=1.0,
            I_m=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            key="env_body",
        )
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
        env.add_shape(
            body=0,  # Attach to the first (and only) body in env
            type=newton.GeoType.SPHERE,
            scale=wp.vec3(0.5, 0.5, 0.5),
            cfg=cfg,
        )

        builder.replicate(env, num_envs)

        model = builder.finalize()
        state = model.state()

        # Create viewer and set offsets
        viewer = ViewerNull(num_frames=1)
        viewer.set_model(model)
        viewer.set_env_offsets(num_envs, spacing)

        # Find ground plane shape instance (should be static)
        ground_instance = None
        env_instance = None
        for shapes in viewer._shape_instances.values():
            if shapes.static:
                ground_instance = shapes
            else:
                env_instance = shapes

        self.assertIsNotNone(ground_instance, "Ground plane instance not found")
        self.assertIsNotNone(env_instance, "Environment instance not found")

        # Update transforms
        viewer.begin_frame(0.0)
        ground_instance.update(state, env_offsets=viewer.env_offsets)
        env_instance.update(state, env_offsets=viewer.env_offsets)

        # Check ground plane is at origin (unaffected by offsets)
        ground_xform = ground_instance.world_xforms.numpy()[0]
        assert_np_equal(ground_xform[:3], np.array([0.0, 0.0, 0.0]), tol=1e-5)

        # Check environment shapes are offset
        env_xforms = env_instance.world_xforms.numpy()
        expected_offsets = np.array([[-5.0, 0.0, 1.0], [5.0, 0.0, 1.0]])

        for i in range(num_envs):
            assert_np_equal(env_xforms[i][:3], expected_offsets[i], tol=1e-5)


def test_visual_separation(test: TestViewerEnvOffsets, device):
    """Test that viewer offsets provide visual separation without affecting physics."""
    num_envs = 4
    spacing = (5.0, 5.0, 0.0)

    # Create model
    builder = newton.ModelBuilder()
    env = newton.ModelBuilder()
    env.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        I_m=wp.mat33(np.eye(3)),
        key="test_body",
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    env.add_shape(
        body=0,  # Attach to the first (and only) body in env
        type=newton.GeoType.BOX,
        scale=wp.vec3(0.5, 0.5, 0.5),
        cfg=cfg,
    )

    builder.replicate(env, num_envs)
    model = builder.finalize(device=device)
    state = model.state()

    # Create viewer and set offsets
    viewer = ViewerNull(num_frames=1)
    viewer.set_model(model)
    viewer.set_env_offsets(num_envs, spacing)

    # Get shape instances from viewer
    shape_instances = next(iter(viewer._shape_instances.values()))

    # Update transforms
    viewer.begin_frame(0.0)
    shape_instances.update(state, env_offsets=viewer.env_offsets)

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

    for i in range(num_envs):
        actual_pos = world_xforms[i][:3]
        expected_pos = expected_offsets[i] + np.array([0.0, 0.0, 1.0])  # body is at (0,0,1)
        assert_np_equal(actual_pos, expected_pos, tol=1e-4)


# Add device-specific tests
devices = get_test_devices()
for device in devices:
    add_function_test(
        TestViewerEnvOffsets,
        f"test_visual_separation_{device.alias}",
        test_visual_separation,
        devices=[device],
    )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
