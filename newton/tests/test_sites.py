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

"""Tests for site support (non-colliding reference markers)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType, ShapeFlags


class TestSiteCreation(unittest.TestCase):
    """Test site creation via ModelBuilder.add_site()."""

    def test_add_site_basic(self):
        """Test adding a site via ModelBuilder."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()))

        site = builder.add_site(
            body=body,
            xform=wp.transform(wp.vec3(0.1, 0, 0), wp.quat_identity()),
            type=GeoType.SPHERE,
            scale=(0.01, 0.01, 0.01),
            key="test_site",
        )

        model = builder.finalize()

        # Verify site properties
        shape_flags = model.shape_flags.numpy()
        shape_body = model.shape_body.numpy()
        shape_type = model.shape_type.numpy()

        self.assertTrue(shape_flags[site] & ShapeFlags.SITE)
        self.assertFalse(shape_flags[site] & ShapeFlags.COLLIDE_SHAPES)
        self.assertEqual(model.shape_key[site], "test_site")
        self.assertEqual(shape_body[site], body)
        self.assertEqual(shape_type[site], GeoType.SPHERE)

    def test_add_site_defaults(self):
        """Test site with default parameters."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        site = builder.add_site(body)

        model = builder.finalize()

        # Check defaults
        shape_flags = model.shape_flags.numpy()
        shape_type = model.shape_type.numpy()

        self.assertTrue(shape_flags[site] & ShapeFlags.SITE)
        self.assertFalse(shape_flags[site] & ShapeFlags.VISIBLE)
        self.assertEqual(shape_type[site], GeoType.SPHERE)

    def test_add_multiple_sites(self):
        """Test adding multiple sites to same body."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        site1 = builder.add_site(body, key="site_1")
        site2 = builder.add_site(body, key="site_2")
        site3 = builder.add_site(body, key="site_3")

        model = builder.finalize()

        shape_flags = model.shape_flags.numpy()

        self.assertNotEqual(site1, site2)
        self.assertNotEqual(site2, site3)
        self.assertTrue(shape_flags[site1] & ShapeFlags.SITE)
        self.assertTrue(shape_flags[site2] & ShapeFlags.SITE)
        self.assertTrue(shape_flags[site3] & ShapeFlags.SITE)

    def test_site_visibility_hidden(self):
        """Test site with visible=False (default)."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        site = builder.add_site(body, visible=False, key="hidden")

        model = builder.finalize()

        shape_flags = model.shape_flags.numpy()

        self.assertTrue(shape_flags[site] & ShapeFlags.SITE)
        self.assertFalse(shape_flags[site] & ShapeFlags.VISIBLE)

    def test_site_visibility_visible(self):
        """Test site with visible=True."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        site = builder.add_site(body, visible=True, key="visible")

        model = builder.finalize()

        shape_flags = model.shape_flags.numpy()

        self.assertTrue(shape_flags[site] & ShapeFlags.SITE)
        self.assertTrue(shape_flags[site] & ShapeFlags.VISIBLE)

    def test_site_different_types(self):
        """Test sites with different geometry types."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        site_sphere = builder.add_site(body, type=GeoType.SPHERE, key="sphere")
        site_box = builder.add_site(body, type=GeoType.BOX, key="box")
        site_capsule = builder.add_site(body, type=GeoType.CAPSULE, key="capsule")
        site_cylinder = builder.add_site(body, type=GeoType.CYLINDER, key="cylinder")

        model = builder.finalize()

        shape_type = model.shape_type.numpy()

        self.assertEqual(shape_type[site_sphere], GeoType.SPHERE)
        self.assertEqual(shape_type[site_box], GeoType.BOX)
        self.assertEqual(shape_type[site_capsule], GeoType.CAPSULE)
        self.assertEqual(shape_type[site_cylinder], GeoType.CYLINDER)

    def test_site_on_world_body(self):
        """Test site attached to world (body=-1)."""
        builder = newton.ModelBuilder()

        site = builder.add_site(-1, xform=wp.transform(wp.vec3(1, 2, 3), wp.quat_identity()), key="world_site")

        model = builder.finalize()

        shape_body = model.shape_body.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_transform = model.shape_transform.numpy()

        self.assertEqual(shape_body[site], -1)
        self.assertTrue(shape_flags[site] & ShapeFlags.SITE)
        pos = wp.transform_get_translation(wp.transform(*shape_transform[site]))
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [1, 2, 3], atol=1e-6)

    def test_site_transforms(self):
        """Test site with custom transform."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        site_xform = wp.transform(wp.vec3(0.5, 0.3, 0.1), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 1.57))
        site = builder.add_site(body, xform=site_xform, key="positioned_site")

        model = builder.finalize()

        # Check that transform was stored
        shape_transform = model.shape_transform.numpy()
        stored_xform = wp.transform(*shape_transform[site])
        pos = wp.transform_get_translation(stored_xform)
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [0.5, 0.3, 0.1], atol=1e-6)

    def test_sites_on_different_bodies(self):
        """Test adding sites to different bodies."""
        builder = newton.ModelBuilder()

        # Create three bodies at different positions
        body1 = builder.add_body(xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()))
        body2 = builder.add_body(xform=wp.transform(wp.vec3(0, 2, 0), wp.quat_identity()))
        body3 = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 3), wp.quat_identity()))

        # Add sites to each body with local offsets
        site1 = builder.add_site(body1, xform=wp.transform(wp.vec3(0.1, 0, 0), wp.quat_identity()), key="site_body1")
        site2 = builder.add_site(body2, xform=wp.transform(wp.vec3(0, 0.2, 0), wp.quat_identity()), key="site_body2")
        site3 = builder.add_site(body3, xform=wp.transform(wp.vec3(0, 0, 0.3), wp.quat_identity()), key="site_body3")

        # Add another site to body1 to test multiple sites per body
        site1_extra = builder.add_site(
            body1, xform=wp.transform(wp.vec3(-0.1, 0, 0), wp.quat_identity()), key="site_body1_extra"
        )

        model = builder.finalize()

        # Verify all sites are flagged correctly
        shape_flags = model.shape_flags.numpy()
        shape_body = model.shape_body.numpy()

        self.assertTrue(shape_flags[site1] & ShapeFlags.SITE)
        self.assertTrue(shape_flags[site2] & ShapeFlags.SITE)
        self.assertTrue(shape_flags[site3] & ShapeFlags.SITE)
        self.assertTrue(shape_flags[site1_extra] & ShapeFlags.SITE)

        # Verify correct body assignments
        self.assertEqual(shape_body[site1], body1)
        self.assertEqual(shape_body[site2], body2)
        self.assertEqual(shape_body[site3], body3)
        self.assertEqual(shape_body[site1_extra], body1)

        # Verify local transforms
        shape_transform = model.shape_transform.numpy()

        pos1 = wp.transform_get_translation(wp.transform(*shape_transform[site1]))
        np.testing.assert_allclose([pos1[0], pos1[1], pos1[2]], [0.1, 0, 0], atol=1e-6)

        pos2 = wp.transform_get_translation(wp.transform(*shape_transform[site2]))
        np.testing.assert_allclose([pos2[0], pos2[1], pos2[2]], [0, 0.2, 0], atol=1e-6)

        pos3 = wp.transform_get_translation(wp.transform(*shape_transform[site3]))
        np.testing.assert_allclose([pos3[0], pos3[1], pos3[2]], [0, 0, 0.3], atol=1e-6)

        pos1_extra = wp.transform_get_translation(wp.transform(*shape_transform[site1_extra]))
        np.testing.assert_allclose([pos1_extra[0], pos1_extra[1], pos1_extra[2]], [-0.1, 0, 0], atol=1e-6)


class TestSiteNonCollision(unittest.TestCase):
    """Test that sites don't participate in collision detection."""

    def test_site_has_no_collision_flags(self):
        """Test that sites are created without collision flags."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        site = builder.add_site(body)

        model = builder.finalize()

        shape_flags = model.shape_flags.numpy()
        flags = shape_flags[site]

        self.assertTrue(flags & ShapeFlags.SITE)
        self.assertFalse(flags & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags & ShapeFlags.COLLIDE_PARTICLES)

    def test_site_no_collision_with_shapes(self):
        """Test that sites don't collide with shapes."""
        builder = newton.ModelBuilder()

        # Body 1 with collision shape
        body1 = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()))
        builder.add_shape_sphere(body1, radius=0.5)
        builder.add_joint_free(child=body1)

        # Body 2 with site (overlapping with body1)
        body2 = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.9), wp.quat_identity()))
        builder.add_site(body2, type=GeoType.SPHERE, scale=(0.5, 0.5, 0.5))
        builder.add_joint_free(child=body2)

        model = builder.finalize()
        state = model.state()

        # Run collision detection
        contacts = model.collide(state)

        # Should have no contacts (site doesn't collide)
        count = contacts.rigid_contact_count.numpy()[0]
        self.assertEqual(count, 0, "Sites should not generate contacts")


class TestSiteInvariantEnforcement(unittest.TestCase):
    """Tests for site invariant enforcement mechanisms."""

    def test_mark_as_site_enforces_invariants(self):
        """Test that mark_as_site() method enforces all invariants."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig()

        # Initially has default values (not a site)
        self.assertFalse(cfg.is_site)
        self.assertGreater(cfg.density, 0.0)
        self.assertNotEqual(cfg.collision_group, 0)
        self.assertTrue(cfg.has_shape_collision)
        self.assertTrue(cfg.has_particle_collision)

        # Call mark_as_site() to enforce invariants
        cfg.mark_as_site()

        # Verify all invariants are enforced
        self.assertTrue(cfg.is_site)
        self.assertEqual(cfg.density, 0.0, "Site must have zero density")
        self.assertEqual(cfg.collision_group, 0, "Site must have collision_group=0")
        self.assertFalse(cfg.has_shape_collision, "Site must not have shape collision")
        self.assertFalse(cfg.has_particle_collision, "Site must not have particle collision")

    def test_flags_setter_enforces_invariants(self):
        """Test that setting flags with SITE enforces invariants."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig()

        # Set flags with SITE bit
        cfg.flags = ShapeFlags.SITE

        # Verify invariants are enforced
        self.assertTrue(cfg.is_site)
        self.assertEqual(cfg.density, 0.0)
        self.assertEqual(cfg.collision_group, 0)
        self.assertFalse(cfg.has_shape_collision)
        self.assertFalse(cfg.has_particle_collision)

    def test_flags_setter_clears_collision_when_site_set(self):
        """Test that SITE flag overrides collision flags."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig()

        # Try to set SITE along with collision flags
        cfg.flags = ShapeFlags.SITE | ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES

        # SITE should override collision flags
        self.assertTrue(cfg.is_site)
        self.assertFalse(cfg.has_shape_collision, "Collision flags should be cleared by SITE")
        self.assertFalse(cfg.has_particle_collision, "Collision flags should be cleared by SITE")

    def test_add_shape_validates_collision_flags(self):
        """Test that add_shape rejects sites with collision enabled."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        # Create config with is_site=True but collision enabled
        cfg = builder.ShapeConfig()
        cfg.is_site = True
        cfg.has_shape_collision = True  # Violate invariant

        # Should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            builder.add_shape(body=body, type=GeoType.SPHERE, cfg=cfg)

        self.assertIn("cannot have collision enabled", str(ctx.exception))

    def test_add_shape_validates_density(self):
        """Test that add_shape rejects sites with non-zero density."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        # Create config with is_site=True but non-zero density
        cfg = builder.ShapeConfig()
        cfg.is_site = True
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        cfg.density = 1.0  # Violate invariant

        # Should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            builder.add_shape(body=body, type=GeoType.SPHERE, cfg=cfg)

        self.assertIn("must have zero density", str(ctx.exception))

    def test_add_shape_validates_collision_group(self):
        """Test that add_shape rejects sites with non-zero collision group."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        # Create config with is_site=True but non-zero collision_group
        cfg = builder.ShapeConfig()
        cfg.is_site = True
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        cfg.density = 0.0
        cfg.collision_group = 5  # Violate invariant

        # Should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            builder.add_shape(body=body, type=GeoType.SPHERE, cfg=cfg)

        self.assertIn("must have collision_group=0", str(ctx.exception))

    def test_add_site_uses_mark_as_site(self):
        """Test that add_site() properly enforces invariants via mark_as_site()."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        # Create site
        site = builder.add_site(body=body, key="test_site")

        model = builder.finalize()

        # Verify site has all correct invariants
        flags = model.shape_flags.numpy()[site]
        self.assertTrue(flags & ShapeFlags.SITE)
        self.assertFalse(flags & ShapeFlags.COLLIDE_SHAPES)
        self.assertFalse(flags & ShapeFlags.COLLIDE_PARTICLES)

        # Verify collision_group is 0
        collision_group = model.shape_collision_group[site]
        self.assertEqual(collision_group, 0)

    def test_direct_is_site_assignment_no_enforcement(self):
        """Test that directly setting is_site=True does not enforce invariants (as documented)."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig()

        # Directly set is_site without using mark_as_site()
        cfg.is_site = True

        # Other properties should NOT be automatically changed
        # (This is intentional behavior per the docstring warning)
        self.assertTrue(cfg.is_site)
        # Note: density, collision_group, etc. are NOT automatically updated
        # This test documents that direct assignment doesn't enforce invariants

    def test_multiple_mark_as_site_calls_idempotent(self):
        """Test that calling mark_as_site() multiple times is safe."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig()

        # Call mark_as_site() multiple times
        cfg.mark_as_site()
        cfg.mark_as_site()
        cfg.mark_as_site()

        # Should still have correct invariants
        self.assertTrue(cfg.is_site)
        self.assertEqual(cfg.density, 0.0)
        self.assertEqual(cfg.collision_group, 0)
        self.assertFalse(cfg.has_shape_collision)
        self.assertFalse(cfg.has_particle_collision)


if __name__ == "__main__":
    unittest.main()
