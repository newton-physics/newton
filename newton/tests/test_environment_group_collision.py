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

import warp as wp

from newton import ModelBuilder
from newton._src.geometry.broad_phase_common import test_world_and_group_pair


class TestEnvironmentGroupCollision(unittest.TestCase):
    """Test world group collision filtering functionality."""

    def setUp(self):
        """Set up test worlds."""
        self.device = wp.get_device()

    def test_shape_collision_filtering(self):
        """Test that shapes from different worlds don't collide."""
        builder = ModelBuilder()

        # Create different bodies for each shape
        body0 = builder.add_body(xform=wp.transform_identity())
        body1 = builder.add_body(xform=wp.transform_identity())
        body2 = builder.add_body(xform=wp.transform_identity())

        # World 0: Box at origin
        builder.begin_world()
        cfg0 = ModelBuilder.ShapeConfig(collision_group=1)
        builder.add_shape_box(
            body=body0, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5, cfg=cfg0
        )
        builder.end_world()

        # World 1: Box slightly overlapping (would collide without world groups)
        builder.begin_world()
        cfg1 = ModelBuilder.ShapeConfig(collision_group=1)
        builder.add_shape_box(
            body=body1, xform=wp.transform(wp.vec3(0.8, 0, 0), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.5, cfg=cfg1
        )
        builder.end_world()

        # Global box that should collide with both
        cfg_global = ModelBuilder.ShapeConfig(collision_group=-1)  # Use -1 to collide with everything
        builder.add_shape_box(
            body=body2,
            xform=wp.transform(wp.vec3(0.4, 1, 0), wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=0.5,
            cfg=cfg_global,
        )

        model = builder.finalize(device=self.device)

        # Verify contact pairs
        # Should have 2 pairs: (global, world0) and (global, world1)
        # Should NOT have (world0, world1)
        self.assertEqual(model.shape_contact_pair_count, 2)

        # Get contact pairs as numpy array for easier checking
        contact_pairs = model.shape_contact_pairs.numpy()

        # Check that world0 (shape 0) and world1 (shape 1) are not paired
        for pair in contact_pairs:
            self.assertFalse(
                (pair[0] == 0 and pair[1] == 1) or (pair[0] == 1 and pair[1] == 0),
                f"Shapes from different worlds should not be paired: {pair}",
            )

        # Check that global shape (shape 2) is paired with both world shapes
        pairs_with_global = [(pair[0], pair[1]) for pair in contact_pairs if 2 in pair]
        self.assertEqual(len(pairs_with_global), 2, f"Global shape should have 2 pairs. Found: {pairs_with_global}")

    def test_particle_shape_collision_filtering(self):
        """Test that particles and shapes from different worlds don't collide."""
        builder = ModelBuilder()

        # World 0: Particle
        builder.begin_world()
        builder.add_particle(pos=(0, 0, 0), vel=(0, 0, 0), mass=1.0, radius=0.1)
        builder.end_world()

        # World 1: Shape that overlaps with particle
        builder.begin_world()
        builder.add_shape_sphere(body=-1, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), radius=0.2)
        builder.end_world()

        # Global shape that should collide with particle
        builder.add_shape_box(
            body=-1, xform=wp.transform(wp.vec3(0, 0.2, 0), wp.quat_identity()), hx=0.5, hy=0.1, hz=0.5
        )

        model = builder.finalize(device=self.device)
        state = model.state()

        # Run collision detection
        contacts = model.collide(state)

        # Get soft contact count
        soft_contact_count = int(contacts.soft_contact_count.numpy()[0])

        # Should only have 1 contact: particle (world0) with global box
        # Should NOT have contact between particle (world0) and sphere (world1)
        self.assertEqual(soft_contact_count, 1)

        if soft_contact_count > 0:
            # Verify the contact is with the global shape (shape index 1)
            contact_shape = int(contacts.soft_contact_shape.numpy()[0])
            self.assertEqual(contact_shape, 1, "Contact should be with global box shape")

    def test_add_world_groups(self):
        """Test that add_world correctly assigns world groups."""
        # Create a robot builder
        robot_builder = ModelBuilder()
        robot_builder.add_body(key="base")
        cfg1 = ModelBuilder.ShapeConfig(collision_group=1)
        robot_builder.add_shape_box(body=0, hx=0.5, hy=0.5, hz=0.5, cfg=cfg1)
        robot_builder.add_body(key="link1")
        cfg2 = ModelBuilder.ShapeConfig(collision_group=2)
        robot_builder.add_shape_capsule(body=1, radius=0.1, half_height=0.5, cfg=cfg2)

        # Create main builder
        main_builder = ModelBuilder()

        # Add global ground plane
        cfg_ground = ModelBuilder.ShapeConfig(collision_group=-1)  # Collides with everything
        main_builder.add_shape_box(
            body=-1, xform=wp.transform(wp.vec3(0, -1, 0), wp.quat_identity()), hx=10, hy=0.1, hz=10, cfg=cfg_ground
        )

        # Add two robot instances in different worlds
        main_builder.add_world(robot_builder)  # World 0
        main_builder.add_world(robot_builder)  # World 1

        model = main_builder.finalize(device=self.device)

        # Verify world indices
        shape_worlds = model.shape_world.numpy()
        body_groups = model.body_world.numpy()

        # Ground plane should be global
        self.assertEqual(shape_worlds[0], -1)

        # First robot shapes should be in world 0
        self.assertEqual(shape_worlds[1], 0)
        self.assertEqual(shape_worlds[2], 0)

        # Second robot shapes should be in world 1
        self.assertEqual(shape_worlds[3], 1)
        self.assertEqual(shape_worlds[4], 1)

        # Bodies should also be correctly assigned
        self.assertEqual(body_groups[0], 0)  # First robot base
        self.assertEqual(body_groups[1], 0)  # First robot link1
        self.assertEqual(body_groups[2], 1)  # Second robot base
        self.assertEqual(body_groups[3], 1)  # Second robot link1

        # Verify collision groups are preserved
        collision_groups = model.shape_collision_group.numpy()
        self.assertEqual(collision_groups[0], -1)  # Ground plane
        self.assertEqual(collision_groups[1], 1)  # First robot box
        self.assertEqual(collision_groups[2], 2)  # First robot capsule
        self.assertEqual(collision_groups[3], 1)  # Second robot box
        self.assertEqual(collision_groups[4], 2)  # Second robot capsule

    def test_mixed_collision_and_world_groups(self):
        """Test interaction between collision groups and world groups."""
        builder = ModelBuilder()

        # Create different bodies for each shape
        body_a = builder.add_body(xform=wp.transform_identity())
        body_b = builder.add_body(xform=wp.transform_identity())
        body_c = builder.add_body(xform=wp.transform_identity())
        body_d = builder.add_body(xform=wp.transform_identity())
        body_e = builder.add_body(xform=wp.transform_identity())
        body_f = builder.add_body(xform=wp.transform_identity())
        body_g = builder.add_body(xform=wp.transform_identity())

        # World 0
        builder.begin_world()
        # Shape A: collision group 1 (only collides with group 1)
        cfg_a = ModelBuilder.ShapeConfig(collision_group=1)
        builder.add_shape_sphere(
            body=body_a, xform=wp.transform(wp.vec3(-1, 0, 0), wp.quat_identity()), radius=0.5, cfg=cfg_a
        )
        # Shape B: collision group 2 (only collides with group 2)
        cfg_b = ModelBuilder.ShapeConfig(collision_group=2)
        builder.add_shape_sphere(
            body=body_b, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), radius=0.5, cfg=cfg_b
        )
        # Shape C: collision group -1 (collides with everything)
        cfg_c = ModelBuilder.ShapeConfig(collision_group=-1)
        builder.add_shape_sphere(
            body=body_c, xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()), radius=0.5, cfg=cfg_c
        )
        builder.end_world()

        # World 1
        builder.begin_world()
        # Shape D: collision group 1
        cfg_d = ModelBuilder.ShapeConfig(collision_group=1)
        builder.add_shape_sphere(
            body=body_d, xform=wp.transform(wp.vec3(-1, 2, 0), wp.quat_identity()), radius=0.5, cfg=cfg_d
        )
        # Shape E: collision group 2
        cfg_e = ModelBuilder.ShapeConfig(collision_group=2)
        builder.add_shape_sphere(
            body=body_e, xform=wp.transform(wp.vec3(0, 2, 0), wp.quat_identity()), radius=0.5, cfg=cfg_e
        )
        builder.end_world()

        # Global world
        # Shape F: collision group 2, not a colliding shape
        cfg_f = ModelBuilder.ShapeConfig(collision_group=2, has_shape_collision=False)
        builder.add_shape_sphere(
            body=body_f, xform=wp.transform(wp.vec3(0, 2, 0), wp.quat_identity()), radius=0.5, cfg=cfg_f
        )
        # Shape G: collision group 1
        cfg_g = ModelBuilder.ShapeConfig(collision_group=1)
        builder.add_shape_sphere(
            body=body_g, xform=wp.transform(wp.vec3(0, 4, 0), wp.quat_identity()), radius=0.5, cfg=cfg_g
        )

        model = builder.finalize(device=self.device)

        # Analyze contact pairs
        contact_pairs = model.shape_contact_pairs.numpy()
        contact_set = {tuple(sorted(pair)) for pair in contact_pairs}

        # Expected pairs within world 0:
        # - (0, 2): A and C (group 1 collides with group -1)
        # - (1, 2): B and C (group 2 collides with group -1)
        # NOT (0, 1): different collision groups

        # Expected pairs within world 1:
        # - None (no shapes with compatible collision groups overlap)

        # Expected cross-world pairs (only with global):
        # - (0, 6): A (world0, group1) and G (global, group1)
        # - (2, 6): C (world0, group-1) and G (global, group1)
        # - (3, 6): D (world1, group1) and G (global, group1)

        # No pairs between world0 and world1
        # F is not a colliding shape

        expected_pairs = {
            (0, 2),  # A-C in world0
            (1, 2),  # B-C in world0
            (0, 6),  # A-G (world0-global)
            (2, 6),  # C-G (world0-global)
            (3, 6),  # D-G (world1-global)
        }

        self.assertEqual(contact_set, expected_pairs, f"Contact pairs mismatch. Got: {contact_set}")

    def test_collision_filter_pair_canonicalization(self):
        """Test that collision filter pairs are properly canonicalized when merging builders."""
        # Realistic scenario: Create child body first, then parent body, then connect with joint
        # This naturally creates non-canonical filter pairs!
        builder = ModelBuilder()

        # Create child body with shapes first
        child_body = builder.add_body(xform=wp.transform_identity())
        builder.add_shape_box(body=child_body, hx=0.5, hy=0.5, hz=0.5)  # index 0
        builder.add_shape_box(body=child_body, hx=0.5, hy=0.5, hz=0.5)  # index 1

        # Create parent body with shapes after
        parent_body = builder.add_body(xform=wp.transform((2.0, 0, 0), wp.quat_identity()))
        builder.add_shape_box(body=parent_body, hx=0.5, hy=0.5, hz=0.5)  # index 2
        builder.add_shape_box(body=parent_body, hx=0.5, hy=0.5, hz=0.5)  # index 3

        # Connect with joint - this will naturally create non-canonical pairs!
        # Without canonicalization, this would add pairs like (2,0), (2,1), (3,0), (3,1)
        # where parent shapes (2,3) > child shapes (0,1)
        builder.add_joint_revolute(
            parent=parent_body,
            child=child_body,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0, 0, 1),
            collision_filter_parent=True,  # This triggers parent-child shape filtering
        )

        # Also test merging builders
        sub_builder = ModelBuilder()
        sub_body = sub_builder.add_body(xform=wp.transform_identity())
        sub_builder.add_shape_box(body=sub_body, hx=0.5, hy=0.5, hz=0.5)

        # Add more shapes to main builder to create offset
        builder.add_shape_box(body=child_body, hx=0.5, hy=0.5, hz=0.5)  # index 4

        # Merge sub_builder - its filter pairs need canonicalization after offset
        builder.add_builder(sub_builder)

        # Finalize
        model = builder.finalize(device=self.device)

        # Verify that parent-child filtering worked correctly
        contact_pairs = model.shape_contact_pairs.numpy()
        contact_set = {tuple(sorted(pair)) for pair in contact_pairs}

        # Parent shapes (2,3) should not collide with child shapes (0,1,4)
        for parent_shape in [2, 3]:
            for child_shape in [0, 1, 4]:
                self.assertNotIn(
                    (min(parent_shape, child_shape), max(parent_shape, child_shape)),
                    contact_set,
                    f"Parent shape {parent_shape} should not collide with child shape {child_shape}",
                )


class TestWorldGroupBroadphaseKernels(unittest.TestCase):
    """Test the broadphase kernels with world group filtering."""

    def test_test_world_and_group_pair(self):
        """Test the world and group pair filtering function."""
        # Test cases: (world_a, world_b, col_a, col_b, expected_result)
        test_cases = [
            # Same world, collision groups allow
            (0, 0, 1, 1, True),  # Same world, same collision group
            (1, 1, -1, 2, True),  # Same world, -1 collides with others
            (2, 2, 0, 1, False),  # Same world, but 0 doesn't collide
            # Different worlds
            (0, 1, 1, 1, False),  # Different worlds, no collision
            (2, 3, -1, -1, False),  # Different worlds, even with -1 collision groups
            # Global world (-1)
            (-1, 0, 1, 1, True),  # Global with world 0
            (1, -1, 2, 2, True),  # World 1 with global
            (-1, -1, 1, 2, False),  # Both global, different collision groups
            (-1, -1, -1, 1, True),  # Both global, -1 collision group
        ]

        # Run tests on CPU
        for world_a, world_b, col_a, col_b, expected in test_cases:

            @wp.kernel
            def test_kernel(world_a: int, world_b: int, col_a: int, col_b: int, result: wp.array(dtype=bool)):
                result[0] = test_world_and_group_pair(world_a, world_b, col_a, col_b)

            result = wp.zeros(1, dtype=bool)
            wp.launch(test_kernel, dim=1, inputs=[world_a, world_b, col_a, col_b, result])

            actual = result.numpy()[0]
            self.assertEqual(
                actual,
                expected,
                f"test_world_and_group_pair({world_a}, {world_b}, {col_a}, {col_b}) = {actual}, expected {expected}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
