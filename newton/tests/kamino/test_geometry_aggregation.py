# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `geometry/aggregation.py`"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.geometry import CollisionDetector, ContactAggregation, ContactMode
from newton._src.solvers.kamino._src.models.builders import basics
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestContactAggregation(unittest.TestCase):
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

    def test_contact_aggregation(self):
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

        # Create a contacts aggregator
        aggregator = ContactAggregation(model=model, contacts=detector.contacts)

        # Run collision detection and aggregate per-body and per-geom contacts
        detector.collide(data, state)

        # Set contact reactions to known values for testing aggregation results
        model_active_nc = int(detector.contacts.model_active_contacts.numpy()[0])
        world_body_start = model.info.bodies_offset.numpy()
        wid_np = detector.contacts.wid.numpy()
        bid_AB_np = detector.contacts.bid_AB.numpy()
        contact_reaction_np = detector.contacts.reaction.numpy().copy()
        for c in range(model_active_nc):
            cwid = int(wid_np[c])
            bid_A = max(-1, int(bid_AB_np[c, 0]) - world_body_start[cwid])
            bid_B = max(-1, int(bid_AB_np[c, 1]) - world_body_start[cwid])
            msg.info(f"Contact {c}: wid={cwid}, bid_A={bid_A}, bid_B={bid_B}")
            # First nunchaku box body
            if bid_B == 0 and bid_A == -1:
                contact_reaction_np[c, :] = np.array([1.0, 0.0, 0.0])  # Force in +x direction
            # Second nunchaku box body
            elif bid_B == 1 and bid_A == -1:
                contact_reaction_np[c, :] = np.array([0.0, 1.0, 0.0])  # Force in +y direction
            # Third nunchaku box body
            elif bid_B == 2 and bid_A == -1:
                contact_reaction_np[c, :] = np.array([0.0, 0.0, 1.0])  # Force in +z direction
        detector.contacts.reaction.assign(contact_reaction_np)
        msg.info("contacts.reaction:\n%s", detector.contacts.reaction.numpy())

        # Enable the mode of active contacts to ensure all contacts are processed by the aggregator
        contact_mode_np = detector.contacts.mode.numpy().copy()
        contact_mode_np[:model_active_nc] = ContactMode.STICKING  # Set mode to STICKING (1) for all active contacts
        detector.contacts.mode.assign(contact_mode_np)

        # Aggregate per-body and per-geom contacts
        aggregator.compute()

        # Optional debug output
        msg.info("aggregator.body_net_force:\n%s", aggregator.body_net_force)
        msg.info("aggregator.body_contact_flag:\n%s", aggregator.body_contact_flag)
        msg.info("aggregator.body_static_contact_flag:\n%s", aggregator.body_static_contact_flag)
        msg.info("aggregator.geom_net_force:\n%s", aggregator.geom_net_force)
        msg.info("aggregator.geom_contact_flag:\n%s", aggregator.geom_contact_flag)

        # Test results: check that the aggregated net forces on bodies and
        # geoms match expected values based on the contact reactions we set
        body_net_force_np = aggregator.body_net_force.numpy()
        body_contact_flag_np = aggregator.body_contact_flag.numpy()
        geom_net_force_np = aggregator.geom_net_force.numpy()
        geom_contact_flag_np = aggregator.geom_contact_flag.numpy()
        for w in range(model.info.num_worlds):
            for b in range(model.size.max_of_num_bodies):
                if b == 0:
                    expected_force = np.array([4.0, 0.0, 0.0])  # First box body should have +x force
                    expected_flag = 1
                elif b == 1:
                    expected_force = np.array([0.0, 1.0, 0.0])  # Second box body should have +y force
                    expected_flag = 1
                elif b == 2:
                    expected_force = np.array([0.0, 0.0, 4.0])  # Third box body should have +z force
                    expected_flag = 1
                np.testing.assert_allclose(
                    actual=body_net_force_np[w, b],
                    desired=expected_force,
                    err_msg=f"World {w} Body {b} net force mismatch",
                )
                self.assertEqual(
                    body_contact_flag_np[w, b], expected_flag, msg=f"World {w} Body {b} contact flag mismatch"
                )
            for g in range(model.size.max_of_num_geoms):
                if g == 0:
                    expected_force = np.array([4.0, 0.0, 0.0])  # First box body should have +x force
                    expected_flag = 1
                elif g == 1:
                    expected_force = np.array([0.0, 1.0, 0.0])  # Second box body should have +y force
                    expected_flag = 1
                elif g == 2:
                    expected_force = np.array([0.0, 0.0, 4.0])  # Third box body should have +z force
                    expected_flag = 1
                else:
                    continue  # Skip world (i.e. static) geoms
                np.testing.assert_allclose(
                    actual=geom_net_force_np[w, g],
                    desired=expected_force,
                    err_msg=f"World {w} Geom {g} net force mismatch",
                )
                self.assertEqual(
                    geom_contact_flag_np[w, g], expected_flag, msg=f"World {w} Geom {g} contact flag mismatch"
                )


###
# Test execution
###


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
