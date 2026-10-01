# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton._src.usd._joint_plan import _ArticulationJointPlan


class TestUsdJointPlan(unittest.TestCase):
    def test_group_resolved_body_pairs(self):
        """Group body pairs and track excluded joints separately for each plan."""
        plan = _ArticulationJointPlan()
        plan.add_joint("/root", -1, 0, excluded=False)
        plan.add_joint("/hinge", 0, 1, excluded=False)
        plan.add_joint("/slide", 0, 1, excluded=False)
        plan.add_joint("/reverse", 1, 0, excluded=False)
        plan.add_joint("/loop", 0, 1, excluded=True)

        other_plan = _ArticulationJointPlan()
        other_plan.add_joint("/other", 0, 1, excluded=False)
        self.assertEqual(other_plan.joint_names, ["/other"])
        self.assertEqual(other_plan.joint_edges, [(0, 1)])
        self.assertEqual(other_plan.merged_joint_groups, {"/other": ["/other"]})
        self.assertEqual(other_plan.joint_excluded, set())

        self.assertEqual(plan.joint_names, ["/root", "/hinge", "/reverse"])
        self.assertEqual(plan.joint_edges, [(-1, 0), (0, 1), (1, 0)])
        self.assertEqual(
            plan.merged_joint_groups,
            {"/root": ["/root"], "/hinge": ["/hinge", "/slide"], "/reverse": ["/reverse"]},
        )
        self.assertEqual(plan.joint_excluded, {"/loop"})


if __name__ == "__main__":
    unittest.main()
