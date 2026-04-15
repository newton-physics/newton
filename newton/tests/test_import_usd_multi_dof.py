# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import newton
from newton.tests.unittest_utils import USD_AVAILABLE


class TestImportUsdMultiDofJoints(unittest.TestCase):
    """Tests for USD import of multi-DOF joints (multiple single-DOF joints between same bodies)."""

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_humanoid_mjc_multi_dof(self):
        """Import a MuJoCo-converted humanoid with multi-joint body pairs.

        The humanoid_mjc.usda file has 21 revolute joints across 13 bodies,
        where several body pairs share multiple joints (e.g. 3 hip joints).
        These must be merged into D6 joints without triggering cycle errors.
        """
        builder = newton.ModelBuilder()
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid_mjc.usda")

        builder.add_usd(asset_path)

        # 13 bodies (torso + 12 child bodies)
        self.assertEqual(builder.body_count, 13)
        # 13 joints: 1 free root + 7 merged D6 + 5 standalone revolute
        self.assertEqual(builder.joint_count, 13)
        # 27 DOFs: 6 free root + 21 individual DOFs from the 21 revolute joints
        self.assertEqual(builder.joint_dof_count, 27)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_humanoid_mjc_path_joint_map(self):
        """All original joint prim paths should be in the path_joint_map."""
        builder = newton.ModelBuilder()
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid_mjc.usda")

        result = builder.add_usd(asset_path)

        path_joint_map = result["path_joint_map"]
        # All 21 original revolute joint paths should be mapped
        self.assertEqual(len(path_joint_map), 21)

        # Merged joints should point to the same joint index
        # e.g. right_hip_x, right_hip_z, right_hip_y all map to the same D6
        hip_paths = [p for p in path_joint_map if "right_hip" in p]
        self.assertEqual(len(hip_paths), 3)
        hip_indices = {path_joint_map[p] for p in hip_paths}
        self.assertEqual(len(hip_indices), 1, "All right hip joints should map to the same D6 joint index")

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_humanoid_mjc_finalize(self):
        """The imported humanoid should finalize and simulate without errors."""
        builder = newton.ModelBuilder()
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid_mjc.usda")
        builder.add_usd(asset_path)

        model = builder.finalize()
        self.assertIsNotNone(model)
        self.assertEqual(model.body_count, 13)
        self.assertEqual(model.joint_count, 13)


if __name__ == "__main__":
    unittest.main()
