# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `geometry/detector.py`"""

import logging
import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelBuilder
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry import (
    CollisionDetector,
)
from newton._src.solvers.kamino._src.geometry.detector import (
    _cap_world_contacts_at_total,
    _estimate_fallback_world_max_contacts,
    _resolve_contact_capacity,
)
from newton.tests.kamino import setup_test_output, setup_tests, test_context
from newton.tests.kamino.test_kamino_geometry_primitive import check_contacts
from newton.tests.unittest_utils import NewtonTestCase
from newton.tests.utils import basics

logger = logging.getLogger(__name__)

###
# Tests
###


class TestCollisionDetectorConfig(NewtonTestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.default_device = wp.get_device(test_context.device)

        setup_test_output(self, verbose=self.verbose)

    def test_00_make_default(self):
        """Test making default collision detector config."""
        config = CollisionDetector.Config()
        self.assertEqual(config.pipeline, "unified")
        self.assertEqual(config.broadphase, "explicit")
        self.assertEqual(config.bvtype, "aabb")

    def test_01_make_with_string_args(self):
        """Test making collision detector config with string arguments."""
        config = CollisionDetector.Config(pipeline="primitive", broadphase="explicit", bvtype="aabb")
        self.assertEqual(config.pipeline, "primitive")
        self.assertEqual(config.broadphase, "explicit")
        self.assertEqual(config.bvtype, "aabb")


class TestGeometryCollisionDetector(NewtonTestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output

        setup_test_output(self, verbose=self.verbose)

        self.build_func = basics.build_boxes_nunchaku
        self.expected_contacts = 9  # NOTE: specialized to build_boxes_nunchaku
        logger.debug("build_func: %s", self.build_func.__name__)
        logger.debug("expected_contacts: %s", self.expected_contacts)

    def test_01_primitive_pipeline(self):
        """
        Test the collision detector with the primitive pipeline
        on multiple worlds containing boxes_nunchaku model.
        """
        # Create and set up a model builder
        builder = ModelBuilder()
        builder.replicate(builder=self.build_func(), copy_count=3)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))
        data = model.data()

        # Create a collision detector with primitive pipeline
        config = CollisionDetector.Config(
            pipeline="primitive",
            broadphase="explicit",
            bvtype="aabb",
        )
        detector = CollisionDetector(model=model, config=config)
        self.assertIs(detector.device, self.default_device)

        # Run collision detection
        detector.collide(data)

        # Create a list of expected number of contacts per shape pair
        expected_world_contacts: list[int] = [self.expected_contacts] * builder.world_count
        logger.debug("expected_world_contacts:\n%s\n", expected_world_contacts)

        # Define expected contacts dictionary
        expected = {
            "model_active_contacts": sum(expected_world_contacts),
            "world_active_contacts": np.array(expected_world_contacts, dtype=np.int32),
        }

        # Check results
        check_contacts(
            detector.contacts,
            expected,
            case="boxes_nunchaku",
            header="primitive pipeline",
        )

    def test_02_unified_pipeline(self):
        """
        Test the collision detector with the unified pipeline
        on multiple worlds containing boxes_nunchaku model.
        """
        # Create and set up a model builder
        builder = ModelBuilder()
        builder.replicate(builder=self.build_func(), copy_count=3)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))
        data = model.data()

        # Create a collision detector with unified pipeline
        config = CollisionDetector.Config(
            pipeline="unified",
            broadphase="explicit",
            bvtype="aabb",
        )
        detector = CollisionDetector(model=model, config=config)
        self.assertIs(detector.device, self.default_device)

        # Run collision detection
        detector.collide(data)

        # Create a list of expected number of contacts per shape pair
        expected_world_contacts: list[int] = [self.expected_contacts] * builder.world_count
        logger.debug("expected_world_contacts:\n%s\n", expected_world_contacts)

        # Define expected contacts dictionary
        expected = {
            "model_active_contacts": sum(expected_world_contacts),
            "world_active_contacts": np.array(expected_world_contacts, dtype=np.int32),
        }

        # Check results
        check_contacts(
            detector.contacts,
            expected,
            case="boxes_nunchaku",
            header="unified pipeline",
        )

    def test_03_pair_based_world_budgets(self):
        """Verify per-world budgets follow geometry rather than an equal model split."""
        builder = ModelBuilder()
        builder.replicate(builder=self.build_func(), copy_count=3)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))

        config = CollisionDetector.Config(pipeline="primitive", broadphase="explicit")
        detector = CollisionDetector(model=model, config=config)

        self.assertEqual(detector.world_max_contacts, model.geoms.world_minimum_contacts)
        self.assertEqual(detector.model_max_contacts, sum(model.geoms.world_minimum_contacts))

    def test_04_max_contacts_caps_model_total(self):
        """Verify ``max_contacts`` caps rather than floors the geometry-based estimate."""
        builder = ModelBuilder()
        builder.replicate(builder=self.build_func(), copy_count=3)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))
        uncapped_total = sum(model.geoms.world_minimum_contacts)
        self.assertGreater(uncapped_total, 15)

        config = CollisionDetector.Config(
            pipeline="primitive",
            broadphase="explicit",
            max_contacts=15,
        )
        detector = CollisionDetector(model=model, config=config)

        self.assertEqual(detector.model_max_contacts, 15)
        self.assertEqual(sum(detector.world_max_contacts), 15)

    def test_05_heterogeneous_world_budgets(self):
        """Verify worlds with different geometry receive different contact budgets."""
        builder = ModelBuilder()
        basics.build_boxes_nunchaku(builder=builder)
        builder.begin_world(label="empty_world")
        builder.end_world()
        model = ModelKamino.from_newton(builder.finalize(self.default_device))

        config = CollisionDetector.Config(pipeline="primitive", broadphase="explicit")
        detector = CollisionDetector(model=model, config=config)

        self.assertEqual(len(detector.world_max_contacts), 2)
        self.assertGreater(detector.world_max_contacts[0], 0)
        self.assertEqual(detector.world_max_contacts[1], 0)
        self.assertEqual(detector.world_max_contacts, model.geoms.world_minimum_contacts)


class TestCollisionDetectorMaterialMixing(NewtonTestCase):
    """Check that the detector's own contacts mix materials as configured."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        setup_test_output(self, verbose=test_context.verbose)

    def _make_box_on_slab(self) -> ModelKamino:
        """A box of friction 0.1 and restitution 0.2 resting on a static slab of 1.0 and 0.6."""
        builder = ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=ModelBuilder.ShapeConfig(mu=0.1, restitution=0.2))
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()),
            hx=1.0,
            hy=1.0,
            hz=0.5,
            cfg=ModelBuilder.ShapeConfig(mu=1.0, restitution=0.6),
        )
        return ModelKamino.from_newton(builder.finalize(self.default_device))

    def test_00_mix_modes(self):
        """Both pipelines mix friction and restitution with the configured modes."""
        for pipeline in ("primitive", "unified"):
            for friction_mode, restitution_mode, expected_mu, expected_restitution in (
                (None, None, 0.55, 0.4),
                ("max", "min", 1.0, 0.2),
                ("min", "max", 0.1, 0.6),
                ("multiply", "average", 0.1, 0.4),
            ):
                with self.subTest(pipeline=pipeline, friction=friction_mode, restitution=restitution_mode):
                    model = self._make_box_on_slab()
                    data = model.data()
                    config = CollisionDetector.Config(
                        pipeline=pipeline,
                        friction_mix_mode=friction_mode,
                        restitution_mix_mode=restitution_mode,
                    )
                    detector = CollisionDetector(model=model, config=config)
                    detector.collide(data)
                    num_contacts = int(detector.contacts.model_active_contacts.numpy()[0])
                    self.assertGreater(num_contacts, 0)
                    material = detector.contacts.material.numpy()[:num_contacts]
                    np.testing.assert_allclose(material[:, 0], expected_mu, rtol=1e-6)
                    np.testing.assert_allclose(material[:, 1], expected_restitution, rtol=1e-6)

    def test_01_reject_invalid_mix_mode(self):
        """An unknown mixing mode is rejected when the config is made."""
        with self.assertRaises(ValueError):
            CollisionDetector.Config(friction_mix_mode="median")
        with self.assertRaises(ValueError):
            CollisionDetector.Config(restitution_mix_mode="median")


class TestCollisionDetectorContactCapacity(NewtonTestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

        setup_test_output(self)

    def test_00_cap_world_contacts_at_total(self):
        """Verify proportional capping preserves the configured model total."""
        capped = _cap_world_contacts_at_total([100, 50, 50], 120)
        self.assertEqual(sum(capped), 120)
        self.assertEqual(capped[0], 60)

    def test_01_fallback_explicit_per_world_pair_counts(self):
        """Verify fallback explicit broad-phase estimates accumulate pairs per world."""
        builder = ModelBuilder()
        builder.replicate(builder=basics.build_boxes_nunchaku(), copy_count=2)
        model = ModelKamino.from_newton(builder.finalize(self.default_device))
        config = CollisionDetector.Config(broadphase="explicit")

        world_max = _estimate_fallback_world_max_contacts(model, config)
        self.assertEqual(len(world_max), 2)
        self.assertGreater(world_max[0], 0)
        self.assertEqual(world_max[0], world_max[1])

    def test_02_resolve_contact_capacity_uses_pair_metadata(self):
        """Verify the default resolved budget matches pair-based metadata."""
        builder = basics.build_boxes_nunchaku()
        model = ModelKamino.from_newton(builder.finalize(self.default_device))
        config = CollisionDetector.Config()

        model_max, world_max = _resolve_contact_capacity(model, config)
        self.assertEqual(world_max, model.geoms.world_minimum_contacts)
        self.assertEqual(model_max, model.geoms.model_minimum_contacts)


###
# Test execution
###


if __name__ == "__main__":
    # Test setup
    newton.utils.setup_console_logging()
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
