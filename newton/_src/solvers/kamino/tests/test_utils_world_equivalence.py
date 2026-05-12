# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the equivalence classes computation utility for discrete information of Kamino"""

import unittest

import warp as wp

from newton._src.solvers.kamino._src.utils.world_equivalence import DiscreteSignature, compute_equivalence_classes
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Tests
###


class TestEquivalenceClasses(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_01_discrete_signature_equivalence(self):
        # Create some discrete signatures
        with wp.ScopedDevice(self.default_device):
            sig_0 = DiscreteSignature(
                num_worlds=6,
                data=wp.array([1, 2, 1, 1, 2, 1, 0, 3, 0, 3, 1, 1, 2, 1, 0, 3, 1], dtype=wp.int32),
                world_offset=wp.array([0, 3, 6, 8, 11, 14], dtype=wp.int32),
                world_size=wp.array([3, 3, 2, 3, 3, 3], dtype=wp.int32),
            )
            # Splits per world as: 1, 2, 1 | 1, 2, 1 | 0, 3 | 0, 3, 1 | 1, 2, 1 | 0, 3, 1

            sig_1 = DiscreteSignature(
                num_worlds=6,
                data=wp.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=wp.int32),
                world_offset=wp.array([0, 2, 4, 6, 8, 10], dtype=wp.int32),
                world_size=wp.array([2, 2, 2, 2, 2, 2], dtype=wp.int32),
                world_delta=wp.array([0, 2, 4, 6, 8, 10], dtype=wp.int32),
            )
            # After subtracting delta: 1, 0 | 1, 0 | 1, 0 | 1, 0 | 1, 0 | 1, 0

            sig_2 = DiscreteSignature(
                num_worlds=6,
                data=wp.array([11, 12, 11, 11, 11, 11], dtype=wp.int32),
            )

        # Compute and check equivalence classes for various combinations of these signatures

        # Signature 0 only, expected classes (up to order): 0, 1, 4 | 2 | 3, 5
        classes_0 = compute_equivalence_classes([sig_0])
        self.assertEqual(len(classes_0), 3)
        self.assertTrue([0, 1, 4] in classes_0)
        self.assertTrue([2] in classes_0)
        self.assertTrue([3, 5] in classes_0)

        # Signature 1 only, single class expected
        classes_1 = compute_equivalence_classes([sig_1])
        self.assertEqual(classes_1, [[0, 1, 2, 3, 4, 5]])

        # Signature 2 only, expected classes (up to order): 0, 2, 3, 4, 5 | 1
        classes_2 = compute_equivalence_classes([sig_2])
        self.assertEqual(len(classes_2), 2)
        self.assertTrue([0, 2, 3, 4, 5] in classes_2)
        self.assertTrue([1] in classes_2)

        # Signatures 0 and 1, expected classes (up to order): 0, 1, 4 | 2 | 3, 5
        classes_01 = compute_equivalence_classes([sig_0, sig_1])
        self.assertEqual(len(classes_01), 3)
        self.assertTrue([0, 1, 4] in classes_01)
        self.assertTrue([2] in classes_01)
        self.assertTrue([3, 5] in classes_01)

        # Signatures 1 and 2, expected classes (up to order): 0, 2, 3, 4, 5 | 1
        classes_12 = compute_equivalence_classes([sig_1, sig_2])
        self.assertEqual(len(classes_12), 2)
        self.assertTrue([0, 2, 3, 4, 5] in classes_12)
        self.assertTrue([1] in classes_12)

        # Signatures 2 and 0, expected classes (up to order): 0, 4 | 1 | 2 | 3, 5
        classes_20 = compute_equivalence_classes([sig_2, sig_0])
        self.assertEqual(len(classes_20), 4)
        self.assertTrue([0, 4] in classes_20)
        self.assertTrue([1] in classes_20)
        self.assertTrue([2] in classes_20)
        self.assertTrue([3, 5] in classes_20)

        # All 3 signatures, expected classes (up to order): 0, 4 | 1 | 2 | 3, 5
        classes_012 = compute_equivalence_classes([sig_0, sig_1, sig_2])
        self.assertEqual(len(classes_012), 4)
        self.assertTrue([0, 4] in classes_012)
        self.assertTrue([1] in classes_012)
        self.assertTrue([2] in classes_012)
        self.assertTrue([3, 5] in classes_012)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
