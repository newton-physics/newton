# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
KAMINO: UNIT TESTS: CORE: SHAPES
"""

import unittest

from newton._src.geometry.types import GeoType
from newton._src.solvers.kamino._src.core.shapes import _max_contacts_for_shape_pair_impl
from newton.tests.kamino import setup_tests

###
# Tests
###


class TestMaxContactsForShapePair(unittest.TestCase):
    def test_catch_canonicalization_order_errors(self):
        """Verify the canonical implementation rejects every reversed shape pair.

        This catches accidentally declaring the shape pair in the wrong order.
        """
        for type_a in GeoType:
            for type_b in GeoType:
                if type_a <= type_b:
                    continue

                with self.subTest(type_a=type_a, type_b=type_b):
                    self.assertEqual(_max_contacts_for_shape_pair_impl(int(type_a), int(type_b)), (0, 0))


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
