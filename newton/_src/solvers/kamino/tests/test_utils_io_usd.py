# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the USD importer utility."""

import math
import os
import unittest

import numpy as np
import warp as wp

import newton
from newton import Model, ModelBuilder
from newton._src.geometry.types import GeoType
from newton._src.solvers.kamino import SolverKamino
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.joints import JOINT_QMAX, JOINT_QMIN, JointActuationType, JointDoFType
from newton._src.solvers.kamino._src.models import get_basics_usd_assets_path, get_testing_usd_assets_path
from newton._src.solvers.kamino._src.models.builders import basics
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.io.usd import USDImporter
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.checks import assert_builders_equal
from newton.tests.unittest_utils import USD_AVAILABLE

###
# Tests
###


class TestUSDImporter(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set the paths to the assets provided by the kamino package
        self.TEST_USD_ASSETS_PATH = get_testing_usd_assets_path()
        self.BASICS_USD_ASSETS_PATH = get_basics_usd_assets_path()

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

    ###
    # Joints supported natively by USD
    ###

    def test_import_joint_revolute_passive_unary(self):
        """Test importing a passive revolute joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_revolute_passive_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [0.5 * math.pi])

    def test_import_joint_revolute_passive(self):
        """Test importing a passive revolute joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_revolute_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [0.5 * math.pi])

    def test_import_joint_revolute_actuated(self):
        """Test importing a actuated revolute joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_revolute_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0])

    def test_import_joint_prismatic_passive_unary(self):
        """Test importing a passive prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_prismatic_passive_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-1.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [1.0])

    def test_import_joint_prismatic_passive(self):
        """Test importing a passive prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_prismatic_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-1.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [1.0])

    def test_import_joint_prismatic_actuated(self):
        """Test importing a actuated prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_prismatic_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-1.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [1.0])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0])

    def test_import_joint_spherical_unary(self):
        """Test importing a passive spherical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_spherical_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [JOINT_QMIN, JOINT_QMIN, JOINT_QMIN])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [JOINT_QMAX, JOINT_QMAX, JOINT_QMAX])

    def test_import_joint_spherical(self):
        """Test importing a passive spherical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_spherical.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [JOINT_QMIN, JOINT_QMIN, JOINT_QMIN])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [JOINT_QMAX, JOINT_QMAX, JOINT_QMAX])

    ###
    # Joints based on specializations of UsdPhysicsD6Joint
    ###

    def test_import_joint_cylindrical_passive_unary(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cylindrical_passive_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-1, JOINT_QMIN])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [1, JOINT_QMAX])

    def test_import_joint_cylindrical_passive(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cylindrical_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-1, JOINT_QMIN])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [1, JOINT_QMAX])

    def test_import_joint_cylindrical_actuated(self):
        """Test importing a actuated cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cylindrical_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-1, JOINT_QMIN])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [1, JOINT_QMAX])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0, 200.0])

    def test_import_joint_universal_passive_unary(self):
        """Test importing a passive universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_universal_passive_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])

    def test_import_joint_universal_passive(self):
        """Test importing a passive universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_universal_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])

    def test_import_joint_universal_actuated(self):
        """Test importing a actuated universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_universal_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0, 200.0])

    def test_import_joint_cartesian_passive_unary(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cartesian_passive_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [10.0, 20.0, 30.0])

    def test_import_joint_cartesian_passive(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cartesian_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [10.0, 20.0, 30.0])

    def test_import_joint_cartesian_actuated(self):
        """Test importing a actuated cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cartesian_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [10.0, 20.0, 30.0])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0, 200.0, 300.0])

    ###
    # Joints based on UsdPhysicsD6Joint
    ###

    def test_import_joint_d6_revolute_passive(self):
        """Test importing a passive revolute joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_revolute_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [math.pi])

    def test_import_joint_d6_revolute_actuated(self):
        """Test importing a actuated revolute joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_revolute_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [math.pi])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0])

    def test_import_joint_d6_prismatic_passive(self):
        """Test importing a passive prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_prismatic_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-10.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [10.0])

    def test_import_joint_d6_prismatic_actuated(self):
        """Test importing a actuated prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_prismatic_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-10.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [10.0])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0])

    def test_import_joint_d6_cylindrical_passive(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_cylindrical_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-1.0, JOINT_QMIN])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [1.0, JOINT_QMAX])

    def test_import_joint_d6_cylindrical_actuated(self):
        """Test importing a actuated cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_cylindrical_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-1.0, JOINT_QMIN])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [1.0, JOINT_QMAX])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0, 200.0])

    def test_import_joint_d6_universal_passive(self):
        """Test importing a passive universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_universal_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])

    def test_import_joint_d6_universal_actuated(self):
        """Test importing a actuated universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_universal_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0, 200.0])

    def test_import_joint_d6_cartesian_passive(self):
        """Test importing a passive cartesian joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_cartesian_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [10.0, 20.0, 30.0])

    def test_importjoint__d6_cartesian_actuated(self):
        """Test importing a actuated cartesian joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_cartesian_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [10.0, 20.0, 30.0])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0, 200.0, 300.0])

    def test_import_joint_d6_spherical_passive(self):
        """Test importing a passive spherical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_spherical_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-math.pi, -math.pi, -math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [math.pi, math.pi, math.pi])

    def test_import_joint_d6_spherical_actuated(self):
        """Test importing a actuated spherical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_spherical_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0][0].wid, 0)
        self.assertEqual(builder_usd.joints[0][0].jid, 0)
        self.assertEqual(builder_usd.joints[0][0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0][0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0][0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0][0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0][0].q_j_min, [-math.pi, -math.pi, -math.pi])
        self.assertEqual(builder_usd.joints[0][0].q_j_max, [math.pi, math.pi, math.pi])
        self.assertEqual(builder_usd.joints[0][0].tau_j_max, [100.0, 200.0, 300.0])

    ###
    # Primitive geometries/shapes
    ###

    def test_import_geom_capsule(self):
        """Test importing a body with geometric primitive capsule shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_capsule.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_geoms, 2)

        # Visual geoms are loaded first
        self.assertEqual(builder_usd.geoms[0][0].wid, 0)
        self.assertEqual(builder_usd.geoms[0][0].gid, 0)
        self.assertEqual(builder_usd.geoms[0][0].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][0].uid]
        self.assertEqual(shape.type, GeoType.CAPSULE)
        self.assertAlmostEqual(shape.radius, 0.2)
        self.assertAlmostEqual(shape.half_height, 1.65)
        self.assertEqual(builder_usd.geoms[0][0].mid, -1)
        self.assertEqual(builder_usd.geoms[0][0].group, 0)
        self.assertEqual(builder_usd.geoms[0][0].collides, 0)
        self.assertEqual(builder_usd.geoms[0][0].max_contacts, 0)

        # Collidable geoms are loaded after visual geoms
        self.assertEqual(builder_usd.geoms[0][1].wid, 0)
        self.assertEqual(builder_usd.geoms[0][1].gid, 1)
        self.assertEqual(builder_usd.geoms[0][1].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][1].uid]
        self.assertEqual(shape.type, GeoType.CAPSULE)
        self.assertAlmostEqual(shape.radius, 0.1)
        self.assertAlmostEqual(shape.half_height, 1.1)
        self.assertEqual(builder_usd.geoms[0][1].mid, 0)
        self.assertEqual(builder_usd.geoms[0][1].group, 1)
        self.assertEqual(builder_usd.geoms[0][1].collides, 1)
        self.assertEqual(builder_usd.geoms[0][1].max_contacts, 10)

    def test_import_geom_cone(self):
        """Test importing a body with geometric primitive cone shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_cone.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_geoms, 2)

        # Visual geoms are loaded first
        self.assertEqual(builder_usd.geoms[0][0].wid, 0)
        self.assertEqual(builder_usd.geoms[0][0].gid, 0)
        self.assertEqual(builder_usd.geoms[0][0].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][0].uid]
        self.assertEqual(shape.type, GeoType.CONE)
        self.assertAlmostEqual(shape.radius, 0.2)
        self.assertAlmostEqual(shape.half_height, 1.65)
        self.assertEqual(builder_usd.geoms[0][0].mid, -1)
        self.assertEqual(builder_usd.geoms[0][0].group, 0)
        self.assertEqual(builder_usd.geoms[0][0].collides, 0)
        self.assertEqual(builder_usd.geoms[0][0].max_contacts, 0)

        # Collidable geoms are loaded after visual geoms
        self.assertEqual(builder_usd.geoms[0][1].wid, 0)
        self.assertEqual(builder_usd.geoms[0][1].gid, 1)
        self.assertEqual(builder_usd.geoms[0][1].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][1].uid]
        self.assertEqual(shape.type, GeoType.CONE)
        self.assertAlmostEqual(shape.radius, 0.1)
        self.assertAlmostEqual(shape.half_height, 1.1)
        self.assertEqual(builder_usd.geoms[0][1].mid, 0)
        self.assertEqual(builder_usd.geoms[0][1].group, 1)
        self.assertEqual(builder_usd.geoms[0][1].collides, 1)
        self.assertEqual(builder_usd.geoms[0][1].max_contacts, 10)

    def test_import_geom_cylinder(self):
        """Test importing a body with geometric primitive cylinder shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_cylinder.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_geoms, 2)

        # Visual geoms are loaded first
        self.assertEqual(builder_usd.geoms[0][0].wid, 0)
        self.assertEqual(builder_usd.geoms[0][0].gid, 0)
        self.assertEqual(builder_usd.geoms[0][0].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][0].uid]
        self.assertEqual(shape.type, GeoType.CYLINDER)
        self.assertAlmostEqual(shape.radius, 0.2)
        self.assertAlmostEqual(shape.half_height, 1.65)
        self.assertEqual(builder_usd.geoms[0][0].mid, -1)
        self.assertEqual(builder_usd.geoms[0][0].group, 0)
        self.assertEqual(builder_usd.geoms[0][0].collides, 0)
        self.assertEqual(builder_usd.geoms[0][0].max_contacts, 0)

        # Collidable geoms are loaded after visual geoms
        self.assertEqual(builder_usd.geoms[0][1].wid, 0)
        self.assertEqual(builder_usd.geoms[0][1].gid, 1)
        self.assertEqual(builder_usd.geoms[0][1].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][1].uid]
        self.assertEqual(shape.type, GeoType.CYLINDER)
        self.assertAlmostEqual(shape.radius, 0.1)
        self.assertAlmostEqual(shape.half_height, 1.1)
        self.assertEqual(builder_usd.geoms[0][1].mid, 0)
        self.assertEqual(builder_usd.geoms[0][1].group, 1)
        self.assertEqual(builder_usd.geoms[0][1].collides, 1)
        self.assertEqual(builder_usd.geoms[0][1].max_contacts, 10)

    def test_import_geom_sphere(self):
        """Test importing a body with geometric primitive sphere shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_sphere.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_geoms, 2)

        # Visual geoms are loaded first
        self.assertEqual(builder_usd.geoms[0][0].wid, 0)
        self.assertEqual(builder_usd.geoms[0][0].gid, 0)
        self.assertEqual(builder_usd.geoms[0][0].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][0].uid]
        self.assertEqual(shape.type, GeoType.SPHERE)
        self.assertAlmostEqual(shape.radius, 0.22)
        self.assertEqual(builder_usd.geoms[0][0].mid, -1)
        self.assertEqual(builder_usd.geoms[0][0].group, 0)
        self.assertEqual(builder_usd.geoms[0][0].collides, 0)
        self.assertEqual(builder_usd.geoms[0][0].max_contacts, 0)

        # Collidable geoms are loaded after visual geoms
        self.assertEqual(builder_usd.geoms[0][1].wid, 0)
        self.assertEqual(builder_usd.geoms[0][1].gid, 1)
        self.assertEqual(builder_usd.geoms[0][1].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][1].uid]
        self.assertEqual(shape.type, GeoType.SPHERE)
        self.assertAlmostEqual(shape.radius, 0.11)
        self.assertEqual(builder_usd.geoms[0][1].mid, 0)
        self.assertEqual(builder_usd.geoms[0][1].group, 1)
        self.assertEqual(builder_usd.geoms[0][1].collides, 1)
        self.assertEqual(builder_usd.geoms[0][1].max_contacts, 10)

    def test_import_geom_ellipsoid(self):
        """Test importing a body with geometric primitive ellipsoid shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_ellipsoid.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_geoms, 2)

        # Visual geoms are loaded first
        self.assertEqual(builder_usd.geoms[0][0].wid, 0)
        self.assertEqual(builder_usd.geoms[0][0].gid, 0)
        self.assertEqual(builder_usd.geoms[0][0].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][0].uid]
        self.assertEqual(shape.type, GeoType.ELLIPSOID)
        self.assertAlmostEqual(shape.rx, 0.22)
        self.assertAlmostEqual(shape.ry, 0.33)
        self.assertAlmostEqual(shape.rz, 0.44)
        self.assertEqual(builder_usd.geoms[0][0].mid, -1)
        self.assertEqual(builder_usd.geoms[0][0].group, 0)
        self.assertEqual(builder_usd.geoms[0][0].collides, 0)
        self.assertEqual(builder_usd.geoms[0][0].max_contacts, 0)

        # Collidable geoms are loaded after visual geoms
        self.assertEqual(builder_usd.geoms[0][1].wid, 0)
        self.assertEqual(builder_usd.geoms[0][1].gid, 1)
        self.assertEqual(builder_usd.geoms[0][1].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][1].uid]
        self.assertEqual(shape.type, GeoType.ELLIPSOID)
        self.assertAlmostEqual(shape.rx, 0.11)
        self.assertAlmostEqual(shape.ry, 0.22)
        self.assertAlmostEqual(shape.rz, 0.33)
        self.assertEqual(builder_usd.geoms[0][1].mid, 0)
        self.assertEqual(builder_usd.geoms[0][1].group, 1)
        self.assertEqual(builder_usd.geoms[0][1].collides, 1)
        self.assertEqual(builder_usd.geoms[0][1].max_contacts, 10)

    def test_import_geom_box(self):
        """Test importing a body with geometric primitive box shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_box.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_geoms, 2)

        # Visual geoms are loaded first
        self.assertEqual(builder_usd.geoms[0][0].wid, 0)
        self.assertEqual(builder_usd.geoms[0][0].gid, 0)
        self.assertEqual(builder_usd.geoms[0][0].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][0].uid]
        self.assertEqual(shape.type, GeoType.BOX)
        self.assertAlmostEqual(shape.hx, 0.111)
        self.assertAlmostEqual(shape.hy, 0.222)
        self.assertAlmostEqual(shape.hz, 0.333)
        self.assertEqual(builder_usd.geoms[0][0].mid, -1)
        self.assertEqual(builder_usd.geoms[0][0].group, 0)
        self.assertEqual(builder_usd.geoms[0][0].collides, 0)
        self.assertEqual(builder_usd.geoms[0][0].max_contacts, 0)

        # Collidable geoms are loaded after visual geoms
        self.assertEqual(builder_usd.geoms[0][1].wid, 0)
        self.assertEqual(builder_usd.geoms[0][1].gid, 1)
        self.assertEqual(builder_usd.geoms[0][1].body, 0)
        shape = builder_usd.shapes[builder_usd.geoms[0][1].uid]
        self.assertEqual(shape.type, GeoType.BOX)
        self.assertAlmostEqual(shape.hx, 0.11)
        self.assertAlmostEqual(shape.hy, 0.22)
        self.assertAlmostEqual(shape.hz, 0.33)
        self.assertEqual(builder_usd.geoms[0][1].mid, 0)
        self.assertEqual(builder_usd.geoms[0][1].group, 1)
        self.assertEqual(builder_usd.geoms[0][1].collides, 1)
        self.assertEqual(builder_usd.geoms[0][1].max_contacts, 10)

    ###
    # Basic models
    ###

    def test_import_basic_box_on_plane(self):
        """Test importing the basic box_on_plane model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "box_on_plane.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = basics.build_box_on_plane(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref, skip_materials=True)

    def test_import_basic_box_pendulum(self):
        """Test importing the basic box_pendulum model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "box_pendulum.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = basics.build_box_pendulum(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref, skip_materials=True)

    def test_import_basic_boxes_hinged(self):
        """Test importing the basic boxes_hinged model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "boxes_hinged.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = basics.build_boxes_hinged(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref, skip_colliders=True, skip_materials=True)

    def test_import_basic_boxes_nunchaku(self):
        """Test importing the basic boxes_nunchaku model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "boxes_nunchaku.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = basics.build_boxes_nunchaku(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref, skip_colliders=True, skip_materials=True)

    def test_import_basic_boxes_fourbar(self):
        """Test importing the basic boxes_fourbar model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "boxes_fourbar.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = basics.build_boxes_fourbar(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref, skip_materials=True)

    def test_import_basic_cartpole(self):
        """Test importing the basic cartpole model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "cartpole.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(
            source=usd_asset_filename, load_static_geometry=True, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = basics.build_cartpole(z_offset=0.0, ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref, skip_materials=True)

    ###
    # Reference models
    ###

    def test_import_model_dr_testmech(self):
        """Test importing the `DR Test Mechanism` example model with all joint types from a USD file"""
        print("")  # Add a newline for better readability

        # Load the DR Test Mechanism model from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        model_asset_file = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=model_asset_file)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 10)
        self.assertEqual(builder_usd.num_joints, 14)
        self.assertEqual(builder_usd.num_geoms, 10)
        self.assertEqual(builder_usd.num_materials, 1)
        self.assertEqual(builder_usd.joints[0][0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][0].dof_type, JointDoFType.FIXED)
        self.assertEqual(builder_usd.joints[0][1].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0][1].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0][2].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][2].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0][3].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][3].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0][4].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][4].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0][5].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][5].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0][6].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][6].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0][7].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][7].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0][8].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][8].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0][9].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][9].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0][10].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][10].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0][11].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][11].dof_type, JointDoFType.FIXED)
        self.assertEqual(builder_usd.joints[0][12].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][12].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0][13].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0][13].dof_type, JointDoFType.CARTESIAN)

    def test_import_model_dr_legs(self):
        """Test importing the `DR Legs` example model from a USD file"""
        print("")  # Add a newline for better readability

        # Load the default DR Legs model from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        model_asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=model_asset_file)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 31)
        self.assertEqual(builder_usd.num_joints, 36)
        self.assertEqual(builder_usd.num_geoms, 31)

    def test_import_model_dr_legs_with_boxes(self):
        """Test importing the `DR Legs` example model from a USD file"""
        print("")  # Add a newline for better readability

        # Load the primitives-only DR Legs model from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        model_asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_boxes.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=model_asset_file)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 31)
        self.assertEqual(builder_usd.num_joints, 36)
        self.assertEqual(builder_usd.num_geoms, 3)

    def test_import_model_dr_legs_with_meshes_and_boxes(self):
        """Test importing the `DR Legs` example model from a USD file"""
        print("")  # Add a newline for better readability

        # Load the primitives-plus-meshes DR Legs model from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        model_asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda")
        importer = USDImporter()
        builder_usd: ModelBuilderKamino = importer.import_from(source=model_asset_file)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 31)
        self.assertEqual(builder_usd.num_joints, 36)
        self.assertEqual(builder_usd.num_geoms, 34)


class TestUSDKaminoSceneAPIImport(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output

        # Set the paths to the assets provided by the kamino package
        self.TEST_USD_ASSETS_PATH = get_testing_usd_assets_path()
        self.BASICS_USD_ASSETS_PATH = get_basics_usd_assets_path()

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

    def generate_single_body_usd_import(self, scene: str = "") -> Model:
        from pxr import Usd

        usd_text = (
            "#usda 1.0\n\n"
            + scene
            + """def Xform "box"
{
    def Scope "RigidBodies"
    {
        def Xform "box_body" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
        )
        {
            # Body Frame
            quatf xformOp:orient = (1, 0, 0, 0)
            double3 xformOp:translate = (0.0, 0.0, 0.1)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

            # Body Velocities
            vector3f physics:linearVelocity = (0, 0, 0)
            vector3f physics:angularVelocity = (0, 0, 0)

            # Mass Properties
            float physics:mass = 1.0
            float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
            point3f physics:centerOfMass = (0, 0, 0)
            quatf physics:principalAxes = (1, 0, 0, 0)

            def Scope "Geometry"
            {
                def Cube "box_geom" (
                    prepend apiSchemas = ["PhysicsCollisionAPI"]
                )
                {
                    float3[] extent = [(-1, -1, -1), (1, 1, 1)]

                    float3 xformOp:scale = (0.1, 0.1, 0.1)
                    quatf xformOp:orient = (1, 0, 0, 0)
                    double3 xformOp:translate = (0, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
                }
            }
        }
    }
}
"""
        )

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_text)

        builder = ModelBuilder()
        SolverKamino.register_custom_attributes(builder)

        builder.add_usd(stage)

        model = builder.finalize()
        return model

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_01_kamino_scene_api_import_no_scene(self):
        """Check that custom attributes are set with the right type."""

        model = self.generate_single_body_usd_import()

        self.assertTrue(hasattr(model, "kamino"))
        kamino_attr = model.kamino

        # Check existence and type of attributes
        self.assertTrue(
            hasattr(kamino_attr, "padmm_warmstarting") and isinstance(kamino_attr.padmm_warmstarting[0], str)
        )
        self.assertTrue(
            hasattr(kamino_attr, "padmm_use_acceleration")
            and isinstance(kamino_attr.padmm_use_acceleration.numpy()[0], np.bool)
        )
        self.assertTrue(hasattr(kamino_attr, "joint_correction") and isinstance(kamino_attr.joint_correction[0], str))

        self.assertTrue(
            hasattr(kamino_attr, "constraints_use_preconditioning")
            and isinstance(kamino_attr.constraints_use_preconditioning.numpy()[0], np.bool)
        )
        self.assertTrue(
            hasattr(kamino_attr, "constraints_alpha")
            and isinstance(kamino_attr.constraints_alpha.numpy()[0], np.floating)
        )
        self.assertTrue(
            hasattr(kamino_attr, "constraints_beta")
            and isinstance(kamino_attr.constraints_beta.numpy()[0], np.floating)
        )
        self.assertTrue(
            hasattr(kamino_attr, "constraints_gamma")
            and isinstance(kamino_attr.constraints_gamma.numpy()[0], np.floating)
        )

        self.assertTrue(
            hasattr(kamino_attr, "padmm_primal_tolerance")
            and isinstance(kamino_attr.padmm_primal_tolerance.numpy()[0], np.floating)
        )
        self.assertTrue(
            hasattr(kamino_attr, "padmm_dual_tolerance")
            and isinstance(kamino_attr.padmm_dual_tolerance.numpy()[0], np.floating)
        )
        self.assertTrue(
            hasattr(kamino_attr, "padmm_complementarity_tolerance")
            and isinstance(kamino_attr.padmm_complementarity_tolerance.numpy()[0], np.floating)
        )
        self.assertTrue(
            hasattr(kamino_attr, "max_solver_iterations")
            and isinstance(kamino_attr.max_solver_iterations.numpy()[0], np.integer)
        )

        # Compare attribute values to KaminoSceneAPI defaults
        self.assertEqual(kamino_attr.padmm_warmstarting[0], "containers")
        self.assertEqual(bool(kamino_attr.padmm_use_acceleration.numpy()[0]), True)
        self.assertEqual(kamino_attr.joint_correction[0], "twopi")

        self.assertEqual(bool(kamino_attr.constraints_use_preconditioning.numpy()[0]), True)
        self.assertAlmostEqual(kamino_attr.constraints_alpha.numpy()[0], 0.01)
        self.assertAlmostEqual(kamino_attr.constraints_beta.numpy()[0], 0.01)
        self.assertAlmostEqual(kamino_attr.constraints_gamma.numpy()[0], 0.01)

        self.assertAlmostEqual(kamino_attr.padmm_primal_tolerance.numpy()[0], 1e-6)
        self.assertAlmostEqual(kamino_attr.padmm_dual_tolerance.numpy()[0], 1e-6)
        self.assertAlmostEqual(kamino_attr.padmm_complementarity_tolerance.numpy()[0], 1e-6)
        self.assertEqual(kamino_attr.max_solver_iterations.numpy()[0], -1)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_02_kamino_scene_api_import_full_scene(self):
        """Check that values defined in USD are properly imported."""

        model = self.generate_single_body_usd_import("""
def PhysicsScene "PhysicsScene" (
    prepend apiSchemas = ["NewtonKaminoSceneAPI"]
)
{
    uniform int newton:maxSolverIterations = 10
    uniform float newton:kamino:padmm:primalTolerance = 0.1
    uniform float newton:kamino:padmm:dualTolerance = 0.2
    uniform float newton:kamino:padmm:complementarityTolerance = 0.3
    uniform token newton:kamino:padmm:warmstarting = "none"
    uniform bool newton:kamino:padmm:useAcceleration = false
    uniform bool newton:kamino:constraints:usePreconditioning = false
    uniform float newton:kamino:constraints:alpha = 0.4
    uniform float newton:kamino:constraints:beta = 0.5
    uniform float newton:kamino:constraints:gamma = 0.6
    uniform token newton:kamino:jointCorrection = "continuous"
}
""")

        self.assertTrue(hasattr(model, "kamino"))
        kamino_attr = model.kamino
        self.assertEqual(kamino_attr.padmm_warmstarting[0], "none")
        self.assertEqual(bool(kamino_attr.padmm_use_acceleration.numpy()[0]), False)
        self.assertEqual(kamino_attr.joint_correction[0], "continuous")

        self.assertEqual(bool(kamino_attr.constraints_use_preconditioning.numpy()[0]), False)
        self.assertAlmostEqual(kamino_attr.constraints_alpha.numpy()[0], 0.4)
        self.assertAlmostEqual(kamino_attr.constraints_beta.numpy()[0], 0.5)
        self.assertAlmostEqual(kamino_attr.constraints_gamma.numpy()[0], 0.6)

        self.assertAlmostEqual(kamino_attr.padmm_primal_tolerance.numpy()[0], 0.1)
        self.assertAlmostEqual(kamino_attr.padmm_dual_tolerance.numpy()[0], 0.2)
        self.assertAlmostEqual(kamino_attr.padmm_complementarity_tolerance.numpy()[0], 0.3)
        self.assertEqual(kamino_attr.max_solver_iterations.numpy()[0], 10)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_03_kamino_scene_api_import_faulty_scenes(self):
        """Check that faulty string attributes raise an error."""

        with self.assertRaises(ValueError):
            self.generate_single_body_usd_import("""
def PhysicsScene "PhysicsScene" (
    prepend apiSchemas = ["NewtonKaminoSceneAPI"]
)
{
    uniform token newton:kamino:padmm:warmstarting = "non"
}
""")

        with self.assertRaises(ValueError):
            self.generate_single_body_usd_import("""
def PhysicsScene "PhysicsScene" (
    prepend apiSchemas = ["NewtonKaminoSceneAPI"]
)
{
    uniform token newton:kamino:jointCorrection = "discrete"
}
""")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_04_kamino_scene_api_import_full_scene_config(self):
        """Check that values defined in USD are properly imported."""

        model = self.generate_single_body_usd_import("""
def PhysicsScene "PhysicsScene" (
    prepend apiSchemas = ["NewtonKaminoSceneAPI"]
)
{
    uniform int newton:maxSolverIterations = 10
    uniform float newton:kamino:padmm:primalTolerance = 0.1
    uniform float newton:kamino:padmm:dualTolerance = 0.2
    uniform float newton:kamino:padmm:complementarityTolerance = 0.3
    uniform token newton:kamino:padmm:warmstarting = "none"
    uniform bool newton:kamino:padmm:useAcceleration = false
    uniform bool newton:kamino:constraints:usePreconditioning = false
    uniform float newton:kamino:constraints:alpha = 0.4
    uniform float newton:kamino:constraints:beta = 0.5
    uniform float newton:kamino:constraints:gamma = 0.6
    uniform token newton:kamino:jointCorrection = "continuous"
}
""")
        config = SolverKamino.Config.from_model(model)

        self.assertEqual(config.rotation_correction, "continuous")

        self.assertEqual(config.dynamics.preconditioning, False)

        self.assertAlmostEqual(config.constraints.alpha, 0.4)
        self.assertAlmostEqual(config.constraints.beta, 0.5)
        self.assertAlmostEqual(config.constraints.gamma, 0.6)

        self.assertEqual(config.padmm.max_iterations, 10)
        self.assertAlmostEqual(config.padmm.primal_tolerance, 0.1)
        self.assertAlmostEqual(config.padmm.dual_tolerance, 0.2)
        self.assertAlmostEqual(config.padmm.compl_tolerance, 0.3)
        self.assertEqual(config.padmm.use_acceleration, False)
        self.assertEqual(config.padmm.warmstart_mode, "none")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_05_kamino_scene_api_import_full_scene_solver(self):
        """Check that values defined in USD are properly imported."""

        model = self.generate_single_body_usd_import("""
def PhysicsScene "PhysicsScene" (
    prepend apiSchemas = ["NewtonKaminoSceneAPI"]
)
{
    uniform int newton:maxSolverIterations = 10
    uniform float newton:kamino:padmm:primalTolerance = 0.1
    uniform float newton:kamino:padmm:dualTolerance = 0.2
    uniform float newton:kamino:padmm:complementarityTolerance = 0.3
    uniform token newton:kamino:padmm:warmstarting = "none"
    uniform bool newton:kamino:padmm:useAcceleration = false
    uniform bool newton:kamino:constraints:usePreconditioning = false
    uniform float newton:kamino:constraints:alpha = 0.4
    uniform float newton:kamino:constraints:beta = 0.5
    uniform float newton:kamino:constraints:gamma = 0.6
    uniform token newton:kamino:jointCorrection = "continuous"
}
""")

        solver = SolverKamino(model)
        config = solver._solver_kamino.config

        self.assertEqual(config.rotation_correction, "continuous")

        self.assertAlmostEqual(config.constraints.alpha, 0.4)
        self.assertAlmostEqual(config.constraints.beta, 0.5)
        self.assertAlmostEqual(config.constraints.gamma, 0.6)

        self.assertEqual(config.dynamics.preconditioning, False)

        self.assertEqual(config.padmm.max_iterations, 10)
        self.assertAlmostEqual(config.padmm.primal_tolerance, 0.1)
        self.assertAlmostEqual(config.padmm.dual_tolerance, 0.2)
        self.assertAlmostEqual(config.padmm.compl_tolerance, 0.3)
        self.assertEqual(config.padmm.use_acceleration, False)
        self.assertEqual(config.padmm.warmstart_mode, "none")


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
