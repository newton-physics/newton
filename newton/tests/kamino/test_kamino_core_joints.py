# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `kamino.core.joints` module"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.joints import JointDoFType
from newton._src.solvers.kamino._src.utils import logger as msg
from newton.tests.kamino import setup_tests, test_context

wp.set_module_options({"enable_backward": False})

###
# Kernels
###


@wp.kernel
def _eval_dofs_axis_wp(
    dof_types: wp.array[wp.int32],
    axes: wp.array[wp.int32],
    out: wp.array[wp.int32],
):
    tid = wp.tid()
    out[tid] = JointDoFType.dofs_axis_wp(dof_types[tid], axes[tid])


###
# Tests
###


class TestCoreJoints(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_joint_dof_type_enum(self):
        doftype = JointDoFType.REVOLUTE

        # Optional verbose output
        msg.info(f"doftype: {doftype}")
        msg.info(f"doftype.value: {doftype.value}")
        msg.info(f"doftype.name: {doftype.name}")
        msg.info(f"doftype.num_cts: {doftype.num_cts}")
        msg.info(f"doftype.num_dofs: {doftype.num_dofs}")
        msg.info(f"doftype.cts_axes: {doftype.cts_axes}")
        msg.info(f"doftype.dofs_axes: {doftype.dofs_axes}")

        # Check the enum values
        self.assertEqual(doftype.value, JointDoFType.REVOLUTE)
        self.assertEqual(doftype.name, "REVOLUTE")
        self.assertEqual(doftype.num_cts, 5)
        self.assertEqual(doftype.num_dofs, 1)
        self.assertEqual(doftype.cts_axes, (0, 1, 2, 4, 5))
        self.assertEqual(doftype.dofs_axes, (3,))

    def test_dofs_axis_wp_matches_dofs_axes(self):
        """Verify dofs_axis_wp agrees with the dofs_axes property for all joint types."""
        dof_types: list[int] = []
        axes: list[int] = []
        expected: list[int] = []

        for dof_type in JointDoFType:
            dof_axes = dof_type.dofs_axes
            if dof_axes == []:
                continue
            for axis in range(len(dof_axes)):
                dof_types.append(dof_type.value)
                axes.append(axis)
                expected.append(dof_axes[axis])

        n = len(expected)
        dof_types_wp = wp.array(dof_types, dtype=wp.int32, device=self.default_device)
        axes_wp = wp.array(axes, dtype=wp.int32, device=self.default_device)
        out_wp = wp.zeros(n, dtype=wp.int32, device=self.default_device)

        wp.launch(
            _eval_dofs_axis_wp,
            dim=n,
            inputs=[dof_types_wp, axes_wp, out_wp],
            device=self.default_device,
        )

        np.testing.assert_array_equal(out_wp.numpy(), expected)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
