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

"""Unit tests for the USD importer utility."""

import math
import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.joints import JointActuationType, JointDoFType
from newton._src.solvers.kamino.core.shapes import ShapeType
from newton._src.solvers.kamino.models import (
    get_basics_usd_assets_path,
    get_examples_usd_assets_path,
    get_tests_usd_assets_path,
)
from newton._src.solvers.kamino.models.builders import (
    build_box_on_plane,
    build_box_pendulum,
    build_boxes_fourbar,
    build_boxes_hinged,
    build_boxes_nunchaku,
    build_cartpole,
)
from newton._src.solvers.kamino.tests.utils.checks import (
    assert_builders_equal,
)
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.io.usd import USDImporter

###
# Constants
###

FLOAT32_MAX = np.finfo(np.float32).max
FLOAT32_MIN = np.finfo(np.float32).min

###
# Tests
###


class TestUSDImporter(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set the paths to the assets provided by the kamino package
        self.TEST_USD_ASSETS_PATH = get_tests_usd_assets_path()
        self.BASICS_USD_ASSETS_PATH = get_basics_usd_assets_path()
        self.EXAMPLES_USD_ASSETS_PATH = get_examples_usd_assets_path()

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
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [0.5 * math.pi])

    def test_import_joint_revolute_passive(self):
        """Test importing a passive revolute joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_revolute_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [0.5 * math.pi])

    def test_import_joint_revolute_actuated(self):
        """Test importing a actuated revolute joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_revolute_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0])

    def test_import_joint_prismatic_passive_unary(self):
        """Test importing a passive prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_prismatic_passive_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-1.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [1.0])

    def test_import_joint_prismatic_passive(self):
        """Test importing a passive prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_prismatic_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-1.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [1.0])

    def test_import_joint_prismatic_actuated(self):
        """Test importing a actuated prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_prismatic_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-1.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [1.0])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0])

    def test_import_joint_spherical_unary(self):
        """Test importing a passive spherical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_spherical_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0].q_j_min, [FLOAT32_MIN, FLOAT32_MIN, FLOAT32_MIN])
        self.assertEqual(builder_usd.joints[0].q_j_max, [FLOAT32_MAX, FLOAT32_MAX, FLOAT32_MAX])

    def test_import_joint_spherical(self):
        """Test importing a passive spherical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_spherical.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0].q_j_min, [FLOAT32_MIN, FLOAT32_MIN, FLOAT32_MIN])
        self.assertEqual(builder_usd.joints[0].q_j_max, [FLOAT32_MAX, FLOAT32_MAX, FLOAT32_MAX])

    ###
    # Joints based on specializations of UsdPhysicsD6Joint
    ###

    def test_import_joint_cylindrical_passive_unary(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cylindrical_passive_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-1, FLOAT32_MIN])
        self.assertEqual(builder_usd.joints[0].q_j_max, [1, FLOAT32_MAX])

    def test_import_joint_cylindrical_passive(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cylindrical_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-1, FLOAT32_MIN])
        self.assertEqual(builder_usd.joints[0].q_j_max, [1, FLOAT32_MAX])

    def test_import_joint_cylindrical_actuated(self):
        """Test importing a actuated cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cylindrical_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-1, FLOAT32_MIN])
        self.assertEqual(builder_usd.joints[0].q_j_max, [1, FLOAT32_MAX])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0, 200.0])

    def test_import_joint_universal_passive_unary(self):
        """Test importing a passive universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_universal_passive_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])

    def test_import_joint_universal_passive(self):
        """Test importing a passive universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_universal_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])

    def test_import_joint_universal_actuated(self):
        """Test importing a actuated universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_universal_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0, 200.0])

    def test_import_joint_cartesian_passive_unary(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cartesian_passive_unary.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, -1)
        self.assertEqual(builder_usd.joints[0].bid_F, 0)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [10.0, 20.0, 30.0])

    def test_import_joint_cartesian_passive(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cartesian_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [10.0, 20.0, 30.0])

    def test_import_joint_cartesian_actuated(self):
        """Test importing a actuated cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_cartesian_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)

        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [10.0, 20.0, 30.0])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0, 200.0, 300.0])

    ###
    # Joints based on UsdPhysicsD6Joint
    ###

    def test_import_joint_d6_revolute_passive(self):
        """Test importing a passive revolute joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_revolute_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [math.pi])

    def test_import_joint_d6_revolute_actuated(self):
        """Test importing a actuated revolute joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_revolute_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [math.pi])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0])

    def test_import_joint_d6_prismatic_passive(self):
        """Test importing a passive prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_prismatic_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-10.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [10.0])

    def test_import_joint_d6_prismatic_actuated(self):
        """Test importing a actuated prismatic joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_prismatic_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 1)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 1)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-10.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [10.0])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0])

    def test_import_joint_d6_cylindrical_passive(self):
        """Test importing a passive cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_cylindrical_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-1.0, FLOAT32_MIN])
        self.assertEqual(builder_usd.joints[0].q_j_max, [1.0, FLOAT32_MAX])

    def test_import_joint_d6_cylindrical_actuated(self):
        """Test importing a actuated cylindrical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_cylindrical_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-1.0, FLOAT32_MIN])
        self.assertEqual(builder_usd.joints[0].q_j_max, [1.0, FLOAT32_MAX])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0, 200.0])

    def test_import_joint_d6_universal_passive(self):
        """Test importing a passive universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_universal_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])

    def test_import_joint_d6_universal_actuated(self):
        """Test importing a actuated universal joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_universal_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 2)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 2)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 2)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-0.5 * math.pi, -0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [0.5 * math.pi, 0.5 * math.pi])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0, 200.0])

    def test_import_joint_d6_cartesian_passive(self):
        """Test importing a passive cartesian joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_cartesian_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [10.0, 20.0, 30.0])

    def test_importjoint__d6_cartesian_actuated(self):
        """Test importing a actuated cartesian joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_cartesian_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.CARTESIAN)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-10.0, -20.0, -30.0])
        self.assertEqual(builder_usd.joints[0].q_j_max, [10.0, 20.0, 30.0])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0, 200.0, 300.0])

    def test_import_joint_d6_spherical_passive(self):
        """Test importing a passive spherical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_spherical_passive.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-math.pi, -math.pi, -math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [math.pi, math.pi, math.pi])

    def test_import_joint_d6_spherical_actuated(self):
        """Test importing a actuated spherical joint with limits from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "joints/test_joint_d6_spherical_actuated.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 2)
        self.assertEqual(builder_usd.num_joints, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[0].wid, 0)
        self.assertEqual(builder_usd.joints[0].jid, 0)
        self.assertEqual(builder_usd.joints[0].cts_offset, 0)
        self.assertEqual(builder_usd.joints[0].dofs_offset, 0)
        self.assertEqual(builder_usd.joints[0].bid_B, 0)
        self.assertEqual(builder_usd.joints[0].bid_F, 1)
        self.assertEqual(len(builder_usd.joints[0].q_j_min), 3)
        self.assertEqual(len(builder_usd.joints[0].q_j_max), 3)
        self.assertEqual(len(builder_usd.joints[0].tau_j_max), 3)
        self.assertEqual(builder_usd.joints[0].q_j_min, [-math.pi, -math.pi, -math.pi])
        self.assertEqual(builder_usd.joints[0].q_j_max, [math.pi, math.pi, math.pi])
        self.assertEqual(builder_usd.joints[0].tau_j_max, [100.0, 200.0, 300.0])

    ###
    # Primitive geometries/shapes
    ###

    def test_import_geom_capsule(self):
        """Test importing a body with geometric primitive capsule shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_capsule.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_physical_geoms, 1)
        self.assertEqual(builder_usd.num_collision_geoms, 1)
        self.assertEqual(builder_usd.collision_geoms[0].wid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].gid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].lid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].bid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].shape.type, ShapeType.CAPSULE)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.radius, 0.1)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.height, 2.2)
        self.assertEqual(builder_usd.collision_geoms[0].mid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].group, 1)
        self.assertEqual(builder_usd.collision_geoms[0].collides, 1)
        self.assertEqual(builder_usd.collision_geoms[0].max_contacts, 10)

    def test_import_geom_cone(self):
        """Test importing a body with geometric primitive cone shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_cone.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_physical_geoms, 1)
        self.assertEqual(builder_usd.num_collision_geoms, 1)
        self.assertEqual(builder_usd.collision_geoms[0].wid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].gid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].lid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].bid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].shape.type, ShapeType.CONE)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.radius, 0.1)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.height, 2.2)
        self.assertEqual(builder_usd.collision_geoms[0].mid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].group, 1)
        self.assertEqual(builder_usd.collision_geoms[0].collides, 1)
        self.assertEqual(builder_usd.collision_geoms[0].max_contacts, 10)

    def test_import_geom_cylinder(self):
        """Test importing a body with geometric primitive cylinder shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_cylinder.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_physical_geoms, 1)
        self.assertEqual(builder_usd.num_collision_geoms, 1)
        self.assertEqual(builder_usd.collision_geoms[0].wid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].gid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].lid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].bid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].shape.type, ShapeType.CYLINDER)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.radius, 0.1)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.height, 2.2)
        self.assertEqual(builder_usd.collision_geoms[0].mid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].group, 1)
        self.assertEqual(builder_usd.collision_geoms[0].collides, 1)
        self.assertEqual(builder_usd.collision_geoms[0].max_contacts, 10)

    def test_import_geom_sphere(self):
        """Test importing a body with geometric primitive sphere shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_sphere.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_physical_geoms, 1)
        self.assertEqual(builder_usd.num_collision_geoms, 1)
        self.assertEqual(builder_usd.collision_geoms[0].wid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].gid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].lid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].bid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].shape.type, ShapeType.SPHERE)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.radius, 0.11)
        self.assertEqual(builder_usd.collision_geoms[0].mid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].group, 1)
        self.assertEqual(builder_usd.collision_geoms[0].collides, 1)
        self.assertEqual(builder_usd.collision_geoms[0].max_contacts, 10)

    def test_import_geom_ellipsoid(self):
        """Test importing a body with geometric primitive ellipsoid shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_ellipsoid.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_physical_geoms, 1)
        self.assertEqual(builder_usd.num_collision_geoms, 1)
        self.assertEqual(builder_usd.collision_geoms[0].wid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].gid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].lid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].bid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].shape.type, ShapeType.ELLIPSOID)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.a, 0.11)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.b, 0.22)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.c, 0.33)
        self.assertEqual(builder_usd.collision_geoms[0].mid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].group, 1)
        self.assertEqual(builder_usd.collision_geoms[0].collides, 1)
        self.assertEqual(builder_usd.collision_geoms[0].max_contacts, 10)

    def test_import_geom_box(self):
        """Test importing a body with geometric primitive box shape from a USD file"""
        usd_asset_filename = os.path.join(self.TEST_USD_ASSETS_PATH, "geoms/test_geom_box.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 1)
        self.assertEqual(builder_usd.num_joints, 0)
        self.assertEqual(builder_usd.num_physical_geoms, 1)
        self.assertEqual(builder_usd.num_collision_geoms, 1)
        self.assertEqual(builder_usd.collision_geoms[0].wid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].gid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].lid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].bid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].shape.type, ShapeType.BOX)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.depth, 0.22)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.width, 0.44)
        self.assertAlmostEqual(builder_usd.collision_geoms[0].shape.height, 0.66)
        self.assertEqual(builder_usd.collision_geoms[0].mid, 0)
        self.assertEqual(builder_usd.collision_geoms[0].group, 1)
        self.assertEqual(builder_usd.collision_geoms[0].collides, 1)
        self.assertEqual(builder_usd.collision_geoms[0].max_contacts, 10)

    ###
    # Basic models
    ###

    def test_import_basic_box_on_plane(self):
        """Test importing the basic box_on_plane model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "box_on_plane.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = build_box_on_plane(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref)

    def test_import_basic_box_pendulum(self):
        """Test importing the basic box_pendulum model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "box_pendulum.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = build_box_pendulum(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref)

    def test_import_basic_boxes_hinged(self):
        """Test importing the basic boxes_hinged model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "boxes_hinged.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = build_boxes_hinged(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref, skip_colliders=True)

    def test_import_basic_boxes_nunchaku(self):
        """Test importing the basic boxes_nunchaku model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "boxes_nunchaku.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = build_boxes_nunchaku(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref, skip_colliders=True)

    def test_import_basic_boxes_fourbar(self):
        """Test importing the basic boxes_fourbar model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "boxes_fourbar.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(
            source=usd_asset_filename, load_static_geometry=False, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = build_boxes_fourbar(ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref)

    def test_import_basic_cartpole(self):
        """Test importing the basic cartpole model from a USD file"""

        # Construct a builder from imported USD asset
        usd_asset_filename = os.path.join(self.BASICS_USD_ASSETS_PATH, "cartpole.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(
            source=usd_asset_filename, load_static_geometry=True, load_materials=False
        )

        # Construct a reference builder using the basics generators
        builder_ref = build_cartpole(z_offset=0.0, ground=False)

        # Check the loaded contents against the reference builder
        assert_builders_equal(self, builder_usd, builder_ref)

    ###
    # Reference models
    ###

    def test_import_model_dr_testmech(self):
        """Test importing the `DR Test Mechanism` example model with all joint types from a USD file"""
        if self.EXAMPLES_USD_ASSETS_PATH is None:
            self.skipTest("EXAMPLES_USD_ASSETS_PATH is `None` - skipping `DR Test Mechanism` import test.")
        print("")  # Add a newline for better readability
        usd_asset_filename = os.path.join(self.EXAMPLES_USD_ASSETS_PATH, "dr_testmech/usd/dr_testmech.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 10)
        self.assertEqual(builder_usd.num_joints, 14)
        self.assertEqual(builder_usd.num_physical_geoms, 10)
        self.assertEqual(builder_usd.num_collision_geoms, 0)
        self.assertEqual(builder_usd.num_materials, 1)
        self.assertEqual(builder_usd.joints[0].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[0].dof_type, JointDoFType.FIXED)
        self.assertEqual(builder_usd.joints[1].act_type, JointActuationType.FORCE)
        self.assertEqual(builder_usd.joints[1].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[2].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[2].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[3].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[3].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[4].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[4].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[5].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[5].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[6].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[6].dof_type, JointDoFType.UNIVERSAL)
        self.assertEqual(builder_usd.joints[7].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[7].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[8].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[8].dof_type, JointDoFType.CYLINDRICAL)
        self.assertEqual(builder_usd.joints[9].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[9].dof_type, JointDoFType.REVOLUTE)
        self.assertEqual(builder_usd.joints[10].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[10].dof_type, JointDoFType.PRISMATIC)
        self.assertEqual(builder_usd.joints[11].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[11].dof_type, JointDoFType.FIXED)
        self.assertEqual(builder_usd.joints[12].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[12].dof_type, JointDoFType.SPHERICAL)
        self.assertEqual(builder_usd.joints[13].act_type, JointActuationType.PASSIVE)
        self.assertEqual(builder_usd.joints[13].dof_type, JointDoFType.CARTESIAN)

    def test_import_model_dr_legs(self):
        """Test importing the `DR Legs` example model from a USD file"""
        if self.EXAMPLES_USD_ASSETS_PATH is None:
            self.skipTest("EXAMPLES_USD_ASSETS_PATH is `None` - skipping `DR Legs` import test.")
        print("")  # Add a newline for better readability
        usd_asset_filename = os.path.join(self.EXAMPLES_USD_ASSETS_PATH, "dr_legs/usd/dr_legs.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 31)
        self.assertEqual(builder_usd.num_joints, 36)
        self.assertEqual(builder_usd.num_collision_geoms, 0)
        self.assertEqual(builder_usd.num_physical_geoms, 31)

    def test_import_model_dr_legs_with_boxes(self):
        """Test importing the `DR Legs` example model from a USD file"""
        if self.EXAMPLES_USD_ASSETS_PATH is None:
            self.skipTest("EXAMPLES_USD_ASSETS_PATH is `None` - skipping `DR Legs` import test.")
        print("")  # Add a newline for better readability
        usd_asset_filename = os.path.join(self.EXAMPLES_USD_ASSETS_PATH, "dr_legs/usd/dr_legs_with_boxes.usda")
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 31)
        self.assertEqual(builder_usd.num_joints, 36)
        self.assertEqual(builder_usd.num_collision_geoms, 3)
        self.assertEqual(builder_usd.num_physical_geoms, 0)

    def test_import_model_dr_legs_with_meshes_and_boxes(self):
        """Test importing the `DR Legs` example model from a USD file"""
        if self.EXAMPLES_USD_ASSETS_PATH is None:
            self.skipTest("EXAMPLES_USD_ASSETS_PATH is `None` - skipping `DR Legs` import test.")
        print("")  # Add a newline for better readability
        usd_asset_filename = os.path.join(
            self.EXAMPLES_USD_ASSETS_PATH, "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda"
        )
        importer = USDImporter()
        builder_usd: ModelBuilder = importer.import_from(source=usd_asset_filename)
        # Check the loaded contents
        self.assertEqual(builder_usd.num_bodies, 31)
        self.assertEqual(builder_usd.num_joints, 36)
        self.assertEqual(builder_usd.num_collision_geoms, 3)
        self.assertEqual(builder_usd.num_physical_geoms, 31)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Set global log-level
    # msg.set_log_level(msg.LogLevel.DEBUG)

    # Run all tests
    unittest.main(verbosity=2)
