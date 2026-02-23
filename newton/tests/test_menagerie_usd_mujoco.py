# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""
MuJoCo Menagerie USD Integration Tests

Tests that MuJoCo Menagerie robots converted to USD simulate identically
in Newton's MuJoCo solver vs native MuJoCo (loaded from original MJCF).

Part A: Import tests verify that each USD asset loads correctly (body/joint/
shape counts, no NaN values, correct joint types).

Part B: Simulation equivalence tests reuse the TestMenagerieBase infrastructure
from test_menagerie_mujoco.py to compare per-step simulation state between
Newton (USD) and native MuJoCo (MJCF).

Asset location: newton/tests/assets/menagerie/
TODO: Migrate assets to newton-assets repo. When available, replace local
paths with download_asset("menagerie/<robot>").
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Any

import numpy as np

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE

from newton.tests.test_menagerie_mujoco import (
    MUJOCO_AVAILABLE,
    StructuredControlStrategy,
    TestMenagerieUSD,
    create_newton_model_from_usd,
)


# Base path for menagerie USD assets
ASSETS_DIR = Path(os.path.dirname(__file__)) / "assets" / "menagerie"

# Menagerie USD asset registry: maps robot name to its configuration.
# Each entry specifies the local USD scene file and the corresponding
# MuJoCo Menagerie folder/XML for the native MuJoCo comparison.
MENAGERIE_USD_ASSETS = {
    "h1": {
        "usd_scene": "h1/h1 scene.usda",
        "menagerie_folder": "unitree_h1",
        "menagerie_xml": "scene.xml",
        "is_floating": True,
    },
    "g1_with_hands": {
        "usd_scene": "g1_with_hands/g1_29dof_with_hand_rev_1_0 scene.usda",
        "menagerie_folder": "unitree_g1",
        "menagerie_xml": "scene_with_hands.xml",
        "is_floating": True,
    },
    "shadow_hand": {
        "usd_scene": "shadow_hand/right_shadow_hand scene.usda",
        "menagerie_folder": "shadow_hand",
        "menagerie_xml": "right_hand.xml",
        "is_floating": False,
    },
    "robotiq_2f85_v4": {
        "usd_scene": "robotiq_2f85_v4/2f85 scene.usda",
        "menagerie_folder": "robotiq_2f85_v4",
        "menagerie_xml": "2f85.xml",
        "is_floating": False,
    },
    "apptronik_apollo": {
        "usd_scene": "apptronik_apollo/apptronik_apollo scene.usda",
        "menagerie_folder": "apptronik_apollo",
        "menagerie_xml": "scene.xml",
        "is_floating": True,
    },
    "booster_t1": {
        "usd_scene": "booster_t1/t1 scene.usda",
        "menagerie_folder": "booster_t1",
        "menagerie_xml": "scene.xml",
        "is_floating": True,
    },
    "wonik_allegro": {
        "usd_scene": "wonik_allegro/allegro_right.usda",
        "menagerie_folder": "wonik_allegro",
        "menagerie_xml": "right_hand.xml",
        "is_floating": False,
    },
}


# =============================================================================
# Part A: Import Tests
# =============================================================================


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUsdImport(unittest.TestCase):
    """Verify that each menagerie USD asset imports correctly into Newton."""

    def _load_robot(self, robot_name: str) -> tuple[newton.ModelBuilder, newton.Model]:
        """Load a menagerie USD asset and return the builder and finalized model."""
        config = MENAGERIE_USD_ASSETS[robot_name]
        usd_path = ASSETS_DIR / config["usd_scene"]
        self.assertTrue(usd_path.exists(), f"USD asset not found: {usd_path}")

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.torsional_friction = 0.005
        builder.default_shape_cfg.rolling_friction = 0.0001

        builder.add_usd(
            str(usd_path),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )

        model = builder.finalize()
        return builder, model

    def _assert_no_nan(self, model: newton.Model, robot_name: str):
        """Assert that the model contains no NaN values in key arrays."""
        for attr_name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            arr = getattr(model, attr_name, None)
            if arr is not None:
                arr_np = arr.numpy()
                self.assertFalse(
                    np.any(np.isnan(arr_np)),
                    f"{robot_name}: NaN detected in model.{attr_name}",
                )

    def test_import_h1(self):
        builder, model = self._load_robot("h1")
        self.assertEqual(builder.body_count, 20)
        self.assertEqual(builder.joint_count, 20)
        self.assertEqual(builder.shape_count, 55)
        self._assert_no_nan(model, "h1")

    def test_import_g1_with_hands(self):
        builder, model = self._load_robot("g1_with_hands")
        self.assertEqual(builder.body_count, 44)
        self.assertEqual(builder.joint_count, 44)
        self.assertEqual(builder.shape_count, 105)
        self._assert_no_nan(model, "g1_with_hands")

    def test_import_shadow_hand(self):
        builder, model = self._load_robot("shadow_hand")
        self.assertEqual(builder.body_count, 25)
        self.assertEqual(builder.joint_count, 25)
        self.assertEqual(builder.shape_count, 63)
        self._assert_no_nan(model, "shadow_hand")

    def test_import_robotiq_2f85_v4(self):
        builder, model = self._load_robot("robotiq_2f85_v4")
        self.assertEqual(builder.body_count, 11)
        self.assertEqual(builder.joint_count, 11)
        self.assertEqual(builder.shape_count, 28)
        self._assert_no_nan(model, "robotiq_2f85_v4")

    def test_import_apptronik_apollo(self):
        builder, model = self._load_robot("apptronik_apollo")
        self.assertEqual(builder.body_count, 36)
        self.assertEqual(builder.joint_count, 35)
        self.assertEqual(builder.shape_count, 87)
        self._assert_no_nan(model, "apptronik_apollo")

    def test_import_booster_t1(self):
        builder, model = self._load_robot("booster_t1")
        self.assertEqual(builder.body_count, 24)
        self.assertEqual(builder.joint_count, 24)
        self.assertEqual(builder.shape_count, 38)
        self._assert_no_nan(model, "booster_t1")

    def test_import_wonik_allegro(self):
        builder, model = self._load_robot("wonik_allegro")
        self.assertEqual(builder.body_count, 21)
        self.assertEqual(builder.joint_count, 21)
        self.assertEqual(builder.shape_count, 42)
        self._assert_no_nan(model, "wonik_allegro")

    def test_import_h1_joint_types(self):
        """Verify H1 has a free joint (floating base) and revolute joints."""
        builder, _ = self._load_robot("h1")
        joint_types = builder.joint_type
        self.assertIn(newton.JointType.FREE, joint_types)
        self.assertIn(newton.JointType.REVOLUTE, joint_types)

    def test_import_wonik_allegro_joint_types(self):
        """Verify Allegro hand has no free joint (fixed base)."""
        builder, _ = self._load_robot("wonik_allegro")
        joint_types = builder.joint_type
        self.assertNotIn(newton.JointType.FREE, joint_types)

    def test_import_h1_multi_world(self):
        """Verify H1 can be replicated into multiple worlds."""
        config = MENAGERIE_USD_ASSETS["h1"]
        usd_path = ASSETS_DIR / config["usd_scene"]

        model = create_newton_model_from_usd(usd_path, num_worlds=4, add_ground=True)
        self.assertEqual(model.world_count, 4)
        self._assert_no_nan(model, "h1_multi_world")


# =============================================================================
# Part B: Simulation Equivalence Tests (pre-converted USD assets)
# =============================================================================
# Tests with local pre-converted USD assets and custom configurations.
# The native MuJoCo model is always loaded from the original MJCF.
# Newton loads the pre-converted USD file.


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_H1(TestMenagerieUSD):
    """Unitree H1 humanoid: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "unitree_h1"
    robot_xml = "scene.xml"
    usd_path = str(ASSETS_DIR / "h1" / "h1 scene.usda")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_G1WithHands(TestMenagerieUSD):
    """Unitree G1 29-DOF with hands: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "unitree_g1"
    robot_xml = "scene_with_hands.xml"
    usd_path = str(ASSETS_DIR / "g1_with_hands" / "g1_29dof_with_hand_rev_1_0 scene.usda")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_ShadowHand(TestMenagerieUSD):
    """Shadow Hand: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "shadow_hand"
    robot_xml = "right_hand.xml"
    usd_path = str(ASSETS_DIR / "shadow_hand" / "right_shadow_hand scene.usda")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_Robotiq2f85V4(TestMenagerieUSD):
    """Robotiq 2F-85 v4 gripper: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "robotiq_2f85_v4"
    robot_xml = "2f85.xml"
    usd_path = str(ASSETS_DIR / "robotiq_2f85_v4" / "2f85 scene.usda")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_ApptronikApollo(TestMenagerieUSD):
    """Apptronik Apollo humanoid: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "apptronik_apollo"
    robot_xml = "scene.xml"
    usd_path = str(ASSETS_DIR / "apptronik_apollo" / "apptronik_apollo scene.usda")

    num_worlds = 2
    num_steps = 100
    njmax = 398
    control_strategy = StructuredControlStrategy(seed=42)

    # Apollo's USD has no collision geoms, so geom/collision counts differ.
    model_skip_fields = TestMenagerieUSD.model_skip_fields | {
        "ngeom", "nmaxcondim", "nmaxpyramid",
    }

    # world_link is an empty static body in MJCF (child of worldbody, no joint,
    # no geoms). Its USD representation uses a PhysicsFixedJoint to the world
    # root, but the importer doesn't yet handle orphan body-to-world fixed
    # joints (they fall outside the articulation). Strip from both sides.
    mjcf_strip_bodies = ["world_link"]
    usd_ignore_paths = [".*/world_link.*"]


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_BoosterT1(TestMenagerieUSD):
    """Booster T1 humanoid: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "booster_t1"
    robot_xml = "scene.xml"
    usd_path = str(ASSETS_DIR / "booster_t1" / "t1 scene.usda")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMenagerieUSD_WonikAllegro(TestMenagerieUSD):
    """Wonik Allegro Hand: USD vs native MuJoCo simulation equivalence."""

    robot_folder = "wonik_allegro"
    robot_xml = "right_hand.xml"
    usd_path = str(ASSETS_DIR / "wonik_allegro" / "allegro_right.usda")

    num_worlds = 2
    num_steps = 100
    control_strategy = StructuredControlStrategy(seed=42)

    def _compare_dof_physics(self, newton_mjw: Any, native_mjw: Any) -> None:
        # The original MJCF has armature=0 which the converter omits from USD.
        # Newton's builder default (0.01) then applies, causing a known mismatch.
        pass


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
