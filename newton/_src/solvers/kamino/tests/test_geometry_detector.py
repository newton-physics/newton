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

"""Unit tests for `geometry/detector.py`"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.geometry import (
    CollisionDetector,
)
from newton._src.solvers.kamino._src.models.builders import basics
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.test_geometry_primitive import check_contacts

###
# Tests
###


class TestCollisionDetectorConfig(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.default_device = wp.get_device(test_context.device)

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.info("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

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


class TestGeometryCollisionDetector(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for detailed output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

        self.build_func = basics.build_boxes_nunchaku
        self.expected_contacts = 9  # NOTE: specialized to build_boxes_nunchaku
        msg.debug(f"build_func: {self.build_func.__name__}")
        msg.debug(f"expected_contacts: {self.expected_contacts}")

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_primitive_pipeline(self):
        """
        Test the collision detector with the primitive pipeline
        on multiple worlds containing boxes_nunchaku model.
        """
        # Create and set up a model builder
        builder = make_homogeneous_builder(num_worlds=3, build_fn=self.build_func)
        model = builder.finalize(self.default_device)
        data = model.data()
        state = model.state()

        # Create a collision detector with primitive pipeline
        config = CollisionDetector.Config(
            pipeline="primitive",
            broadphase="explicit",
            bvtype="aabb",
        )
        detector = CollisionDetector(model=model, config=config)
        self.assertIs(detector.device, self.default_device)

        # Run collision detection
        detector.collide(data, state)

        # Create a list of expected number of contacts per shape pair
        expected_world_contacts: list[int] = [self.expected_contacts] * builder.num_worlds
        msg.debug("expected_world_contacts:\n%s\n", expected_world_contacts)

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
        builder = make_homogeneous_builder(num_worlds=3, build_fn=self.build_func)
        model = builder.finalize(self.default_device)
        data = model.data()
        state = model.state()

        # Create a collision detector with unified pipeline
        config = CollisionDetector.Config(
            pipeline="unified",
            broadphase="explicit",
            bvtype="aabb",
        )
        detector = CollisionDetector(model=model, config=config)
        self.assertIs(detector.device, self.default_device)

        # Run collision detection
        detector.collide(data, state)

        # Create a list of expected number of contacts per shape pair
        expected_world_contacts: list[int] = [self.expected_contacts] * builder.num_worlds
        msg.debug("expected_world_contacts:\n%s\n", expected_world_contacts)

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


###
# Test execution
###


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
