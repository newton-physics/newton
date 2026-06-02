# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from enum import IntEnum

import newton


class TestModelStateFlags(unittest.TestCase):
    def test_public_flags_are_int_enums(self):
        self.assertTrue(issubclass(newton.ModelFlags, IntEnum))
        self.assertTrue(issubclass(newton.StateFlags, IntEnum))
        self.assertIs(newton.solvers.ModelFlags, newton.ModelFlags)
        self.assertIs(newton.solvers.StateFlags, newton.StateFlags)

    def test_flag_masks_accept_custom_int_bits(self):
        custom_model_bit = 1 << 20
        model_mask = newton.ModelFlags.MODEL_PROPERTIES | custom_model_bit

        self.assertIs(type(model_mask), int)
        self.assertTrue(model_mask & newton.ModelFlags.MODEL_PROPERTIES)
        self.assertTrue(model_mask & custom_model_bit)

        custom_state_bit = 1 << 20
        state_mask = newton.StateFlags.JOINT_Q | custom_state_bit

        self.assertIs(type(state_mask), int)
        self.assertTrue(state_mask & newton.StateFlags.JOINT_Q)
        self.assertTrue(state_mask & custom_state_bit)

    def test_solver_flag_aliases_are_removed(self):
        self.assertFalse(hasattr(newton.solvers, "SolverModelFlags"))
        self.assertFalse(hasattr(newton.solvers, "SolverStateFlags"))


if __name__ == "__main__":
    unittest.main()
