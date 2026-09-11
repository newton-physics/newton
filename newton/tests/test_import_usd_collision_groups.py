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
        """Preserve non-positive builder collision defaults on imported shapes."""
        stage, shapes = self._make_stage(("A", "B", "C", "D", "E"))

        for default_collision_group in (0, -1):
            with self.subTest(default_collision_group=default_collision_group):
                builder = newton.ModelBuilder()
                builder.default_shape_cfg.collision_group = default_collision_group
                builder.add_usd(stage)

                self.assertEqual(builder.shape_collision_group, [default_collision_group] * len(shapes))
                for shape_a in range(builder.shape_count):
                    for shape_b in range(shape_a + 1, builder.shape_count):
                        self.assertFalse(
                            builder._test_group_pair(
                                builder.shape_collision_group[shape_a], builder.shape_collision_group[shape_b]
                            )
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

    def test_collision_groups_exclude_disabled_shapes(self):
        """Keep group filters between enabled colliders only."""
        from pxr import UsdPhysics

        stage, shapes = self._make_stage(("EnabledA", "DisabledA", "EnabledB", "DisabledB"))
        for name in ("DisabledA", "DisabledB"):
            UsdPhysics.CollisionAPI(shapes[name].GetPrim()).GetCollisionEnabledAttr().Set(False)
        group_a = self._add_group(stage, "GroupA", (shapes["EnabledA"], shapes["DisabledA"]))
        group_b = self._add_group(stage, "GroupB", (shapes["EnabledB"], shapes["DisabledB"]))
        group_a.CreateFilteredGroupsRel().AddTarget(group_b.GetPath())

        for load_visual_shapes in (False, True):
            with self.subTest(load_visual_shapes=load_visual_shapes):
                builder = newton.ModelBuilder()
                builder.add_usd(stage, load_visual_shapes=load_visual_shapes)
                enabled = {builder.shape_label.index(f"/{name}") for name in ("EnabledA", "EnabledB")}
                self.assertEqual(
                    {i for i, flags in enumerate(builder.shape_flags) if flags & newton.ShapeFlags.COLLIDE_SHAPES},
                    enabled,
                )
                self.assertEqual(set(builder.shape_collision_filter_pairs), {tuple(sorted(enabled))})

    def test_filtered_pairs_exclude_disabled_shapes(self):
        """Keep authored collider and body filters between enabled shapes only."""
        from pxr import Usd, UsdGeom, UsdPhysics

        for endpoint_kind in ("collider", "body"):
            for load_visual_shapes in (False, True):
                with self.subTest(endpoint_kind=endpoint_kind, load_visual_shapes=load_visual_shapes):
                    stage = Usd.Stage.CreateInMemory()
                    for body_name in ("A", "B"):
                        body = UsdGeom.Xform.Define(stage, f"/{body_name}")
                        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                        for name, enabled in (("Enabled", True), ("Disabled", False)):
                            shape = UsdGeom.Cube.Define(stage, f"/{body_name}/{name}")
                            collision = UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                            collision.GetCollisionEnabledAttr().Set(enabled)
                    if endpoint_kind == "body":
                        stage.GetPrimAtPath("/A").CreateRelationship("physics:filteredPairs").AddTarget("/B")
                    else:
                        for name in ("Enabled", "Disabled"):
                            stage.GetPrimAtPath(f"/A/{name}").CreateRelationship("physics:filteredPairs").SetTargets(
                                ["/B/Enabled", "/B/Disabled"]
                            )

                    builder = newton.ModelBuilder()
                    builder.add_usd(stage, load_visual_shapes=load_visual_shapes)
                    enabled = {builder.shape_label.index(f"/{name}/Enabled") for name in ("A", "B")}
                    self.assertEqual(
                        {i for i, flags in enumerate(builder.shape_flags) if flags & newton.ShapeFlags.COLLIDE_SHAPES},
                        enabled,
                    )
                    self.assertEqual(set(builder.shape_collision_filter_pairs), {tuple(sorted(enabled))})

    def test_mjcf_import_after_usd(self):
        """Preserve compact collision filters for a subsequent MJCF import."""
        stage, _ = self._make_stage(("UsdShape",))
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)

        builder.add_mjcf(
            """
            <mujoco>
                <worldbody>
                    <body name="mjcf_body">
                        <geom name="mjcf_shape" type="sphere" size="0.1"/>
                    </body>
                </worldbody>
            </mujoco>
            """
        )

        self.assertEqual(builder.shape_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
