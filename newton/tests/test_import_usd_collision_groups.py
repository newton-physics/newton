# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestImportUsdCollisionGroups(unittest.TestCase):
    @staticmethod
    def _make_stage(shape_names):
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        shapes = {}
        for name in shape_names:
            shape = UsdGeom.Cube.Define(stage, f"/{name}")
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
            shapes[name] = shape
        return stage, shapes

    @staticmethod
    def _add_group(stage, name, shapes, *, filtered=(), inverted=False, merge_group=""):
        from pxr import UsdPhysics

        group = UsdPhysics.CollisionGroup.Define(stage, f"/{name}")
        includes = group.GetCollidersCollectionAPI().CreateIncludesRel()
        for shape in shapes:
            includes.AddTarget(shape.GetPath())
        for filtered_group in filtered:
            group.CreateFilteredGroupsRel().AddTarget(filtered_group.GetPath())
        if inverted:
            group.CreateInvertFilteredGroupsAttr().Set(True)
        if merge_group:
            group.CreateMergeGroupNameAttr().Set(merge_group)
        return group

    def _assert_filtered_pairs(self, stage, shapes, expected_filtered, *, default_collision_group=1):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.collision_group = default_collision_group
        builder.add_usd(stage)

        shape_ids = {name: builder.shape_label.index(str(shape.GetPath())) for name, shape in shapes.items()}
        expected_filtered = {
            tuple(sorted((shape_ids[name_a], shape_ids[name_b]))) for name_a, name_b in expected_filtered
        }
        filtered_pairs = set(builder.shape_collision_filter_pairs)
        for name_a, shape_a in shape_ids.items():
            for name_b, shape_b in shape_ids.items():
                if shape_a >= shape_b:
                    continue
                pair = (shape_a, shape_b)
                collision_enabled = (
                    builder._test_group_pair(
                        builder.shape_collision_group[shape_a], builder.shape_collision_group[shape_b]
                    )
                    and pair not in filtered_pairs
                )
                self.assertEqual(
                    collision_enabled,
                    pair not in expected_filtered,
                    f"collision mismatch for {name_a}-{name_b}",
                )
        return builder, shape_ids

    def test_unfiltered_and_ungrouped_colliders(self):
        """Preserve collisions between unfiltered groups and ungrouped colliders."""
        stage, shapes = self._make_stage(("A", "B", "Ungrouped"))
        self._add_group(stage, "GroupA", (shapes["A"],))
        self._add_group(stage, "GroupB", (shapes["B"],))

        self._assert_filtered_pairs(stage, shapes, ())

    def test_nonpositive_builder_collision_groups(self):
        """Preserve enabled USD pairs with non-positive builder collision defaults."""
        stage, shapes = self._make_stage(("A", "B", "Ungrouped"))
        self._add_group(stage, "GroupA", (shapes["A"],))
        self._add_group(stage, "GroupB", (shapes["B"],))

        for default_collision_group in (0, -1):
            with self.subTest(default_collision_group=default_collision_group):
                self._assert_filtered_pairs(
                    stage,
                    shapes,
                    (),
                    default_collision_group=default_collision_group,
                )

    def test_normal_and_inverted_filtering(self):
        """Preserve self, cross-group, and inverted collision filtering."""
        stage, shapes = self._make_stage(("A0", "A1", "B", "C", "Ungrouped"))
        group_a = self._add_group(stage, "GroupA", (shapes["A0"], shapes["A1"]))
        group_b = self._add_group(stage, "GroupB", (shapes["B"],))
        group_c = self._add_group(stage, "GroupC", (shapes["C"],))
        group_a.CreateFilteredGroupsRel().SetTargets([group_a.GetPath(), group_b.GetPath()])
        group_c.CreateFilteredGroupsRel().AddTarget(group_b.GetPath())
        group_c.CreateInvertFilteredGroupsAttr().Set(True)

        self._assert_filtered_pairs(
            stage,
            shapes,
            (("A0", "A1"), ("A0", "B"), ("A1", "B"), ("A0", "C"), ("A1", "C"), ("C", "Ungrouped")),
        )

    def test_merged_groups_and_multiple_memberships(self):
        """Preserve merged collision groups and colliders with multiple memberships."""
        stage, shapes = self._make_stage(("MergedA", "MergedB", "Multi", "Filtered", "Other"))
        filtered_group = self._add_group(stage, "FilteredGroup", (shapes["Filtered"],))
        self._add_group(
            stage,
            "MergedGroupA",
            (shapes["MergedA"],),
            filtered=(filtered_group,),
            merge_group="shared",
        )
        self._add_group(stage, "MergedGroupB", (shapes["MergedB"],), merge_group="shared")
        self._add_group(stage, "MultiGroupA", (shapes["Multi"],), filtered=(filtered_group,))
        self._add_group(stage, "MultiGroupB", (shapes["Multi"], shapes["Other"]))

        self._assert_filtered_pairs(
            stage,
            shapes,
            (("MergedA", "Filtered"), ("MergedB", "Filtered"), ("Multi", "Filtered")),
        )

    def test_group_filters_compose_with_filtered_pairs(self):
        """Disable a pair when either group or pair filtering requests it."""
        stage, shapes = self._make_stage(("PairA", "PairB", "GroupA", "GroupB"))
        shapes["PairA"].GetPrim().CreateRelationship("physics:filteredPairs").AddTarget(shapes["PairB"].GetPath())
        group_a = self._add_group(stage, "GroupAFilter", (shapes["GroupA"],))
        group_b = self._add_group(stage, "GroupBFilter", (shapes["GroupB"],))
        group_a.CreateFilteredGroupsRel().AddTarget(group_b.GetPath())

        builder, shape_ids = self._assert_filtered_pairs(stage, shapes, (("PairA", "PairB"), ("GroupA", "GroupB")))
        filtered_pairs = set(builder.shape_collision_filter_pairs)
        self.assertIn(tuple(sorted((shape_ids["PairA"], shape_ids["PairB"]))), filtered_pairs)
        self.assertIn(tuple(sorted((shape_ids["GroupA"], shape_ids["GroupB"]))), filtered_pairs)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
