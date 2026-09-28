# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from newton._src.usd._joint_plan import _ArticulationJointPlan


class TestUsdJointPlan(unittest.TestCase):
    def test_group_resolved_body_pairs(self):
        """Group ordered body IDs and keep excluded joints out of the tree."""
        plan = _ArticulationJointPlan()
        plan.add_joint("/root", -1, 0, excluded=False)
        plan.add_joint("/hinge", 0, 1, excluded=False)
        plan.add_joint("/slide", 0, 1, excluded=False)
        plan.add_joint("/reverse", 1, 0, excluded=False)
        plan.add_joint("/loop", 0, 1, excluded=True)

        self.assertEqual(plan.joint_names, ["/root", "/hinge", "/reverse"])
        self.assertEqual(plan.joint_edges, [(-1, 0), (0, 1), (1, 0)])
        self.assertEqual(
            plan.merged_joint_groups,
            {"/root": ["/root"], "/hinge": ["/hinge", "/slide"], "/reverse": ["/reverse"]},
        )
        self.assertEqual(plan.joint_excluded, {"/loop"})

    def test_joint_order(self):
        """Keep source order or traverse a branching articulation using BFS or DFS."""
        plan = _ArticulationJointPlan()
        for path, parent, child in [("/leaf", 1, 3), ("/left", 0, 1), ("/right", 0, 2), ("/root", -1, 0)]:
            plan.add_joint(path, parent, child, excluded=False)
        for ordering, expected in [(None, [0, 1, 2, 3]), ("bfs", [3, 1, 2, 0]), ("dfs", [3, 1, 0, 2])]:
            with self.subTest(ordering=ordering):
                self.assertEqual(list(plan.get_joint_order(ordering)), expected)

    def test_empty_and_single_joint_plans(self):
        """Keep empty, excluded-only and single-joint articulations independent."""
        empty = _ArticulationJointPlan()
        excluded = _ArticulationJointPlan()
        excluded.add_joint("/loop", 0, 1, excluded=True)
        single = _ArticulationJointPlan()
        single.add_joint("/root", -1, 0, excluded=False)
        for ordering in (None, "bfs", "dfs"):
            with self.subTest(ordering=ordering):
                self.assertEqual(list(empty.get_joint_order(ordering)), [])
                self.assertEqual(list(excluded.get_joint_order(ordering)), [])
                self.assertEqual(list(single.get_joint_order(ordering)), [0])
        self.assertEqual(empty.merged_joint_groups, {})
        self.assertEqual(empty.joint_excluded, set())

    def test_invalid_graphs(self):
        """Reject invalid trees only when topology ordering is requested."""
        cases = [
            ([(0, 1), (2, 3)], "Multiple roots found"),
            ([(0, 1), (2, 1)], "Reversed joints are not supported: /joint1"),
            ([(0, 1), (1, 2), (2, 0)], "cycle"),
        ]
        for edges, message in cases:
            plan = _ArticulationJointPlan()
            for index, (parent, child) in enumerate(edges):
                plan.add_joint(f"/joint{index}", parent, child, excluded=False)
            self.assertEqual(list(plan.get_joint_order(None)), list(range(len(edges))))
            for ordering in ("bfs", "dfs"):
                with self.subTest(edges=edges, ordering=ordering), self.assertRaisesRegex(ValueError, message):
                    plan.get_joint_order(ordering)


if __name__ == "__main__":
    unittest.main()
