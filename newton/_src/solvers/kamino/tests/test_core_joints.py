# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `kamino.core.joints` module"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.joints import JointDoFType
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.solvers import SolverKamino

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

    def test_pure_three_dof_rotation_metadata(self):
        """Identify joints with exactly three rotational DoFs and no translational DoFs."""
        expected = {
            JointDoFType.SPHERICAL,
            JointDoFType.ROTATION_VECTOR,
        }
        for dof_type in JointDoFType:
            with self.subTest(dof_type=dof_type):
                self.assertEqual(dof_type.is_pure_three_dof_rotation, dof_type in expected)

    def test_validate_joint_axes(self):
        """Accept valid joint frames and reject every unsupported axis layout."""
        identity = np.eye(3, dtype=np.float32)
        valid_axes = {
            JointDoFType.FREE: np.vstack((identity, identity)),
            JointDoFType.REVOLUTE: identity[:1],
            JointDoFType.PRISMATIC: identity[:1],
            JointDoFType.CYLINDRICAL: np.vstack((identity[0], identity[0])),
            JointDoFType.UNIVERSAL: identity[:2],
            JointDoFType.SPHERICAL: identity,
            JointDoFType.CARTESIAN: identity,
            JointDoFType.FIXED: np.empty((0, 3), dtype=np.float32),
            JointDoFType.ROTATION_VECTOR: identity,
        }
        for dof_type, axes in valid_axes.items():
            with self.subTest(dof_type=dof_type, valid=True):
                self.assertIsNone(JointDoFType.validate_axes(dof_type, axes))

        invalid_axes = {
            JointDoFType.FREE: np.vstack((identity, identity[[0, 2, 1]])),
            JointDoFType.REVOLUTE: np.zeros((1, 3), dtype=np.float32),
            JointDoFType.PRISMATIC: np.zeros((1, 3), dtype=np.float32),
            JointDoFType.CYLINDRICAL: identity[:2],
            JointDoFType.UNIVERSAL: np.vstack((identity[0], identity[0])),
            JointDoFType.SPHERICAL: identity[[0, 2, 1]],
            JointDoFType.CARTESIAN: np.vstack((identity[0], identity[1], [0.1, 0.0, 1.0])),
            JointDoFType.ROTATION_VECTOR: identity[[0, 2, 1]],
        }
        for dof_type, axes in invalid_axes.items():
            with self.subTest(dof_type=dof_type, valid=False):
                self.assertIsNotNone(JointDoFType.validate_axes(dof_type, axes))

    def test_newton_conversion_rejects_noncanonical_rotation_vector_axes(self):
        """Reject noncanonical rotation-vector axes before launching frame conversion."""
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint(
            newton.JointType.D6,
            parent=-1,
            child=body,
            linear_axes=[],
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y),
            ],
            label="noncanonical_rotation",
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=self.default_device, skip_validation_joints=True)

        with self.assertRaisesRegex(ValueError, "noncanonical_rotation.*canonical"):
            ModelKamino.from_newton(model)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
