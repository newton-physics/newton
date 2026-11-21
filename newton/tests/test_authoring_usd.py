# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise scene document interchange through actual USD composition/import."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.viewer.authoring_usd import (
    NotAuthoringSceneError,
    _asset_dependencies,
    load_scene,
    prepare_usd,
    save_scene,
    template_bounds,
)
from newton.tests.unittest_utils import requires


@requires("pxr.Usd")
class TestAuthoringUsd(unittest.TestCase):
    @staticmethod
    def object(kind="box", **changes):
        values = {
            "id": "box-id",
            "name": "Box",
            "kind": SimpleNamespace(value=kind),
            "position": (1.0, 2.0, 3.0),
            "rotation": (0.0, 0.0, 0.0, 1.0),
            "size": (2.0, 2.0, 4.0),
            "color": (0.25, 0.5, 0.75),
            "mass": 3.0,
            "friction": 0.42,
            "dynamic": True,
            "source": None,
            "visible": True,
        }
        values.update(changes)
        return SimpleNamespace(**values)

    @staticmethod
    def asset(path, *, units=1.0, rigid=True, variants=False):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, units)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        cube = UsdGeom.Cube.Define(stage, "/Asset/Body")
        cube.CreateSizeAttr(2.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 0.0))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        if rigid:
            UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
            UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(2.0)
        if variants:
            UsdPhysics.MassAPI(cube.GetPrim()).GetMassAttr().Clear()
            for name in ("Newton", "MuJoCo", "PhysX"):
                variant = root.GetPrim().GetVariantSets().AddVariantSet(name)
                for selection in ("Enabled", "Disabled"):
                    variant.AddVariant(selection)
                variant.SetVariantSelection("Enabled" if name != "Newton" else "Disabled")
            variant = root.GetPrim().GetVariantSets().GetVariantSet("Newton")
            variant.SetVariantSelection("Enabled")
            with variant.GetVariantEditContext():
                UsdPhysics.MassAPI(cube.GetPrim()).GetMassAttr().Set(7.0)
            variant.SetVariantSelection("Disabled")
        stage.GetRootLayer().Save()
        return stage

    def test_primitives_roundtrip_and_native_physics(self):
        """Preserve definitions and standard geometry, mass, pose, and friction."""
        from pxr import Usd, UsdGeom

        objects = [
            self.object(position=np.array((1.0, 2.0, 3.0), dtype=np.float32), overrides=("mass", "dynamic", "mass")),
            self.object("sphere", id="sphere", size=(2.0, 2.0, 2.0), position=(4.0, 0.0, 0.0)),
            self.object("capsule", id="capsule"),
            self.object("cylinder", id="cylinder", dynamic=False, visible=False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Scene.USDA"
            save_scene(path, objects)
            definitions = load_scene(path)
            self.assertEqual([item["id"] for item in definitions], [item.id for item in objects])
            self.assertEqual(definitions[2]["kind"], "capsule")
            self.assertEqual(definitions[2]["size"], [2.0, 2.0, 4.0])
            self.assertFalse(definitions[3]["visible"])
            self.assertEqual(definitions[0]["overrides"], ["dynamic", "mass"])
            stage = Usd.Stage.Open(str(path))
            capsule = UsdGeom.Capsule(stage.GetPrimAtPath("/World/Object_2/Geometry"))
            self.assertEqual(capsule.GetHeightAttr().Get(), 2.0)
            self.assertEqual(capsule.GetRadiusAttr().Get(), 1.0)
            builder = newton.ModelBuilder()
            builder.add_usd(stage, load_visual_shapes=False)
            self.assertEqual(builder.body_count, 4)
            self.assertEqual(builder.shape_count, 5)
            np.testing.assert_allclose(builder.body_mass, [3.0] * 4)
            cylinder_body = builder.body_label.index("/World/Object_3")
            self.assertTrue(builder.body_flags[cylinder_body] & newton.BodyFlags.KINEMATIC)
            np.testing.assert_allclose(builder.body_q[0].p, (1.0, 2.0, 3.0))
            ground = builder.shape_label.index("/World/Ground")
            self.assertEqual(builder.shape_body[ground], -1)
            self.assertEqual(builder.shape_type[ground], newton.GeoType.PLANE)
            np.testing.assert_allclose(builder.shape_scale[ground], (0.0, 0.0, 0.0))
            self.assertAlmostEqual(builder.shape_material_mu[ground], 0.8, places=6)
            np.testing.assert_allclose(
                [value for index, value in enumerate(builder.shape_material_mu) if index != ground], [0.42] * 4
            )
            box_shape = builder.shape_label.index("/World/Object_0/Geometry")
            np.testing.assert_allclose(builder.shape_scale[box_shape], (1.0, 1.0, 2.0))
            np.testing.assert_allclose(builder.shape_color[box_shape], objects[0].color, atol=1.0e-6)
            self.assertIsNone(template_bounds(builder))
            builder.shape_flags[ground] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            lower, upper = template_bounds(builder)
            np.testing.assert_allclose(lower, (0, -1, -1), atol=1e-6)
            np.testing.assert_allclose(upper, (5, 3, 5), atol=1e-6)
            sphere_shape = builder.shape_label.index("/World/Object_1/Geometry")
            self.assertAlmostEqual(builder.shape_material_mu_rolling[sphere_shape], 0.005, places=6)
            self.assertEqual(stage.GetPrimAtPath("/World/Object_1/Geometry").GetAttribute("mjc:condim").Get(), 6)
            mujoco = prepare_usd(str(path), mujoco=True)
            sphere = mujoco.shape_label.index("/World/Object_1/Geometry")
            self.assertEqual(mujoco.custom_attributes["mujoco:condim"].values[sphere], 6)
            self.assertAlmostEqual(mujoco.shape_material_mu_rolling[sphere], 0.005, places=6)
            builder.shape_margin[sphere_shape] = 0.2
            lower, upper = template_bounds(builder)
            np.testing.assert_allclose(lower, (0, -1.2, -1.2), atol=1e-6)
            np.testing.assert_allclose(upper, (5.2, 3, 5), atol=1e-6)

    def test_asset_variant_units_axis_and_source_isolation(self):
        """Apply Newton variants and SI conversion without editing source layers."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Asset.usda"
            stage = self.asset(path, variants=True)
            previous = path.read_bytes()
            template = prepare_usd(str(path), scale=(2.0, 2.0, 2.0))
            self.assertEqual(template.body_count, 1)
            self.assertEqual(template.body_mass[0], 7.0)
            np.testing.assert_allclose(template.body_q[0].p, (0.0, 0.0, 4.0), atol=1.0e-6)
            lower, upper = template_bounds(template)
            np.testing.assert_allclose(lower, (-2, -2, 2), atol=1e-6)
            np.testing.assert_allclose(upper, (2, 2, 6), atol=1e-6)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(
                stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Newton").GetVariantSelection(), "Disabled"
            )
            template.body_mass[0] = 99.0
            self.assertEqual(prepare_usd(str(path)).body_mass[0], 7.0)
            visual = Path(directory) / "centimetres.usda"
            self.asset(visual, units=0.01, rigid=False)
            converted = prepare_usd(str(visual))
            np.testing.assert_allclose(converted.shape_transform[0].p, (0.0, 0.0, 0.02), atol=1.0e-6)
            np.testing.assert_allclose(converted.shape_scale[0], (0.01, 0.01, 0.01), atol=1.0e-6)
            rigid = Path(directory) / "rigid-centimetres.usda"
            self.asset(rigid, units=0.01)
            with self.assertRaisesRegex(ValueError, "metersPerUnit=1"):
                prepare_usd(str(rigid))

    def test_template_bounds_scaled_mesh_and_heightfield(self):
        """Bound host geometry with signed scale, composed frames, and terrain caps."""
        with wp.ScopedDevice("cpu"):
            mesh = newton.Mesh(
                [(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3)],
                [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
                compute_inertia=False,
            )
            builder = newton.ModelBuilder()
            body = builder.add_link(xform=wp.transform((1, 2, 3), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi / 2)))
            for shape_type in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
                builder.add_shape(
                    body=body,
                    type=shape_type,
                    src=mesh,
                    xform=wp.transform((0.25, 0, 0), wp.quat_identity()),
                    scale=(-2, 3, 0.5),
                    cfg=newton.ModelBuilder.ShapeConfig(margin=0.1),
                )
            lower, upper = template_bounds(builder)
            np.testing.assert_allclose(lower, (-5.1, 0.15, 2.9), atol=1e-6)
            np.testing.assert_allclose(upper, (1.1, 2.35, 4.6), atol=1e-6)
            field = newton.Heightfield([[0, 0], [0, 1]], nrow=2, ncol=2, hx=1, hy=2, min_z=-1, max_z=3)
            terrain = newton.ModelBuilder()
            terrain.add_shape_heightfield(
                heightfield=field,
                xform=wp.transform((4, 0, 1), wp.quat_identity()),
                scale=(2, 0.5, 3),
            )
            lower, upper = template_bounds(terrain)
            np.testing.assert_allclose(lower, (2, -1, -2.012), atol=1e-6)
            np.testing.assert_allclose(upper, (6, 1, 10), atol=1e-6)
            plane = newton.ModelBuilder()
            plane.add_shape_plane(width=2, length=4, xform=wp.transform((0, 0, 1), wp.quat_identity()))
            lower, upper = template_bounds(plane)
            np.testing.assert_allclose(lower, (-1, -2, 1), atol=1e-6)
            np.testing.assert_allclose(upper, (1, 2, 1), atol=1e-6)

    def test_reference_composition_roundtrip(self):
        """Keep referenced rigid topology, scale, and object physics in exported USD."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.usda"
            source_stage = self.asset(source)
            source_stage.CreateClassPrim("/Prototype").SetTypeName("Xform")
            prototype = UsdGeom.Cube.Define(source_stage, "/Prototype/Cube")
            prototype.GetPrim().CreateRelationship("guide:target").SetTargets(["/Prototype/Cube"])
            instance = UsdGeom.Xform.Define(source_stage, "/Asset/Geometry/Instance")
            variant = instance.GetPrim().GetVariantSets().AddVariantSet("Newton")
            variant.AddVariant("Disabled")
            variant.AddVariant("Enabled")
            variant.SetVariantSelection("Enabled")
            with variant.GetVariantEditContext():
                instance.GetPrim().GetReferences().AddInternalReference("/Prototype")
                instance.GetPrim().SetInstanceable(True)
            variant.SetVariantSelection("Disabled")
            source_stage.GetRootLayer().Save()
            lower, upper = template_bounds(prepare_usd(str(source)))
            np.testing.assert_allclose(lower, (-1, -1, 1), atol=1e-6)
            np.testing.assert_allclose(upper, (1, 1, 3), atol=1e-6)
            path = Path(directory) / "scene.usda"
            save_scene(path, [self.object("usd", source=source, size=(2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))])
            definition = load_scene(path)[0]
            self.assertEqual(definition["source"], str(source))
            stage = Usd.Stage.Open(str(path))
            asset = stage.GetPrimAtPath("/World/Object_0/Asset/Asset")
            self.assertTrue(asset.HasAuthoredReferences())
            body = stage.GetPrimAtPath("/World/Object_0/Asset/Asset/Body")
            self.assertTrue(body.HasAPI(UsdPhysics.RigidBodyAPI))
            nested = stage.GetPrimAtPath("/World/Object_0/Asset/Asset/Geometry/Instance/Cube")
            self.assertFalse(nested.IsInstanceProxy())
            self.assertEqual(nested.GetRelationship("guide:target").GetTargets(), [nested.GetPath()])
            builder = newton.ModelBuilder()
            builder.add_usd(stage, load_visual_shapes=False)
            self.assertEqual(builder.body_count, 1)
            self.assertAlmostEqual(builder.body_mass[0], 3.0)
            shape = builder.shape_label.index("/World/Object_0/Asset/Asset/Body")
            self.assertAlmostEqual(builder.shape_material_mu[shape], 0.42, places=6)
            np.testing.assert_allclose(builder.body_q[0].p, (1.0, 2.0, 7.0), atol=1.0e-6)

    def test_multiroot_joint_and_inertia_references(self):
        """Preserve cross-root joints and rescale authored inertia with total mass."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "multiroot.usda"
            stage = Usd.Stage.CreateNew(str(source))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            for name, mass in (("A", 2.0), ("B", 4.0)):
                cube = UsdGeom.Cube.Define(stage, f"/{name}")
                UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                inertia = UsdPhysics.MassAPI.Apply(cube.GetPrim())
                inertia.CreateMassAttr(mass)
                inertia.CreateDiagonalInertiaAttr(Gf.Vec3f(mass))
            joint = UsdPhysics.FixedJoint.Define(stage, "/Joint")
            joint.CreateBody0Rel().SetTargets(["/A"])
            joint.CreateBody1Rel().SetTargets(["/B"])
            stage.GetRootLayer().Save()
            template = prepare_usd(str(source))
            path = Path(directory) / "scene.usda"
            save_scene(path, [self.object("usd", source=str(source), size=(1.0, 1.0, 1.0), mass=12.0)])
            exported = Usd.Stage.Open(str(path))
            joint = UsdPhysics.FixedJoint(exported.GetPrimAtPath("/World/Object_0/Asset/Joint"))
            self.assertEqual(str(joint.GetBody0Rel().GetTargets()[0]), "/World/Object_0/Asset/A")
            self.assertEqual(str(joint.GetBody1Rel().GetTargets()[0]), "/World/Object_0/Asset/B")
            builder = newton.ModelBuilder()
            builder.add_usd(exported, collapse_fixed_joints=False, load_visual_shapes=False)
            self.assertEqual(builder.joint_count, template.joint_count)
            self.assertEqual(builder.body_count, template.body_count)
            np.testing.assert_allclose(builder.body_mass, np.asarray(template.body_mass) * 2.0)
            np.testing.assert_allclose(builder.body_inertia, np.asarray(template.body_inertia) * 2.0)

    def test_preserve_native_physics_and_promote_visual_asset(self):
        """Preserve untouched rigid physics and give bare visual props a physical root."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rigid.usda"
            self.asset(source)
            template = prepare_usd(str(source))
            path = Path(directory) / "scene.usda"
            save_scene(
                path,
                [
                    self.object(
                        "usd",
                        source=str(source),
                        size=(1.0, 1.0, 1.0),
                        color=(1.0, 1.0, 1.0),
                        mass=99.0,
                        dynamic=False,
                        overrides=(),
                    )
                ],
            )
            builder = newton.ModelBuilder()
            builder.add_usd(str(path), load_visual_shapes=False)
            self.assertAlmostEqual(builder.body_mass[0], template.body_mass[0])
            self.assertFalse(builder.body_flags[0] & newton.BodyFlags.KINEMATIC)
            shape = builder.shape_label.index("/World/Object_0/Asset/Asset/Body")
            self.assertAlmostEqual(builder.shape_material_mu[shape], template.shape_material_mu[0])
            self.assertEqual(load_scene(path)[0]["overrides"], [])
            visual = Path(directory) / "visual.usda"
            stage = self.asset(visual, rigid=False)
            stage.GetPrimAtPath("/Asset/Body").RemoveAPI(UsdPhysics.CollisionAPI)
            UsdGeom.BasisCurves.Define(stage, "/Asset/grasp_identifier_curve").CreatePurposeAttr(UsdGeom.Tokens.guide)
            stage.GetRootLayer().Save()
            lower, upper = template_bounds(prepare_usd(str(visual)))
            np.testing.assert_allclose(lower, (-1, -1, 1), atol=1e-6)
            np.testing.assert_allclose(upper, (1, 1, 3), atol=1e-6)
            UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/Asset/Body"))
            decoration = UsdGeom.Cube.Define(stage, "/Asset/Decoration")
            decoration.CreateSizeAttr(1.0)
            decoration.AddTranslateOp().Set(Gf.Vec3d(5, 0, 0))
            stage.GetRootLayer().Save()
            mixed = prepare_usd(str(visual))
            self.assertEqual(mixed.body_count, 0)
            lower, upper = template_bounds(mixed)
            np.testing.assert_allclose(lower, (-1, -1, -0.5), atol=1e-6)
            np.testing.assert_allclose(upper, (5.5, 1, 3), atol=1e-6)
            promoted = Path(directory) / "promoted.usda"
            save_scene(
                promoted,
                [self.object("usd", source=str(visual), size=(1.0, 1.0, 1.0), color=(1.0, 1.0, 1.0), overrides=())],
            )
            builder = newton.ModelBuilder()
            builder.add_usd(str(promoted), load_visual_shapes=False)
            self.assertEqual(builder.body_count, 1)
            self.assertAlmostEqual(builder.body_mass[0], 3.0)
            shape = builder.shape_label.index("/World/Object_0/Asset/Asset/Body")
            self.assertEqual(builder.shape_body[shape], 0)
            self.assertTrue(builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES)
            exported = Usd.Stage.Open(str(promoted))
            guide = exported.GetPrimAtPath("/World/Object_0/Asset/Asset/grasp_identifier_curve")
            self.assertFalse(guide.HasAPI(UsdPhysics.CollisionAPI))

    def test_jointed_asset_without_articulation_metadata(self):
        """Compile unrooted joint branches without dropping joints or changing authored roots."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "jointed.usda"
            stage = Usd.Stage.CreateNew(str(source))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            scene = UsdGeom.Xform.Define(stage, "/Scene")
            stage.SetDefaultPrim(scene.GetPrim())

            def add_body(path, position):
                body = UsdGeom.Cube.Define(stage, path)
                body.CreateSizeAttr(0.1)
                body.AddTranslateOp().Set(Gf.Vec3d(position, 0.0, 0.0))
                UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                UsdPhysics.CollisionAPI.Apply(body.GetPrim())
                UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)

            for name, count, offset in (("Prop", 6, 0.0), ("Existing", 2, 4.0)):
                group = UsdGeom.Xform.Define(stage, f"/Scene/{name}")
                if name == "Existing":
                    UsdPhysics.ArticulationRootAPI.Apply(group.GetPrim())
                for index in range(count):
                    path = f"{group.GetPath()}/Body{index}"
                    add_body(path, offset + index * 0.25)
                    if index and index != 4:
                        parent = 4 if index == 5 else 0
                        joint_type = UsdPhysics.RevoluteJoint if index in (2, 3) else UsdPhysics.FixedJoint
                        joint = joint_type.Define(stage, f"{group.GetPath()}/Joint{index}")
                        joint.CreateBody0Rel().SetTargets([f"{group.GetPath()}/Body{parent}"])
                        joint.CreateBody1Rel().SetTargets([path])
                        joint.CreateLocalPos0Attr(Gf.Vec3f((index - parent) * 0.25, 0.0, 0.0))
            # A third unowned graph crosses top-level namespaces, so no shared
            # authored prim can carry its articulation root.
            add_body("/External0", 8.0)
            add_body("/External1", 8.25)
            external = UsdPhysics.FixedJoint.Define(stage, "/ExternalJoint")
            external.CreateBody0Rel().SetTargets(["/External0"])
            external.CreateBody1Rel().SetTargets(["/External1"])
            external.CreateLocalPos0Attr(Gf.Vec3f(0.25, 0.0, 0.0))
            stage.GetRootLayer().Save()
            original = source.read_bytes()
            template = prepare_usd(str(source))
            model = template.finalize(device="cpu")
            self.assertEqual(model.body_count, 10)
            self.assertEqual(model.joint_count, 10)
            self.assertEqual(model.articulation_count, 4)
            self.assertEqual(sum(kind == newton.JointType.REVOLUTE for kind in template.joint_type), 2)
            state = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            np.testing.assert_allclose(state.body_q.numpy(), model.body_q.numpy(), atol=1.0e-6)
            output = Path(directory) / "scene.usda"
            save_scene(
                output,
                [self.object("usd", source=str(source), size=(1.0, 1.0, 1.0), color=(1.0, 1.0, 1.0), overrides=())],
            )
            exported = Usd.Stage.Open(str(output))
            for root in ("Scene/Prop/Body0", "Scene/Prop/Body4", "Scene/Existing", "External0"):
                self.assertTrue(
                    exported.GetPrimAtPath(f"/World/Object_0/Asset/{root}").HasAPI(UsdPhysics.ArticulationRootAPI)
                )
            imported = newton.ModelBuilder()
            imported.add_usd(exported, collapse_fixed_joints=False)
            rebuilt = imported.finalize(device="cpu")
            self.assertEqual((rebuilt.body_count, rebuilt.joint_count, rebuilt.articulation_count), (10, 10, 4))
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(stage.GetPrimAtPath("/Scene/Prop").HasAPI(UsdPhysics.ArticulationRootAPI))
            self.assertTrue(stage.GetPrimAtPath("/Scene/Existing").HasAPI(UsdPhysics.ArticulationRootAPI))

    def test_scalar_material_tint_and_unsupported_graph_save(self):
        """Tint scalar materials and retain the old file when a graph cannot be exported."""
        from pxr import Gf, Sdf, UsdShade

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "material.usda"
            stage = self.asset(source)
            material = UsdShade.Material.Define(stage, "/Asset/Material")
            shader = UsdShade.Shader.Define(stage, "/Asset/Material/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            color.Set(Gf.Vec3f(0.1, 0.2, 0.3))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/Asset/Body")).Bind(material)
            stage.GetRootLayer().Save()
            template = prepare_usd(str(source))
            path = Path(directory) / "scene.usda"
            tint = (0.5, 0.75, 0.9)
            obj = self.object("usd", source=str(source), size=(1.0, 1.0, 1.0), color=tint)
            save_scene(path, [obj])
            builder = newton.ModelBuilder()
            builder.add_usd(str(path), load_visual_shapes=False)
            shape = builder.shape_label.index("/World/Object_0/Asset/Asset/Body")
            np.testing.assert_allclose(
                builder.shape_color[shape], np.asarray(template.shape_color[0]) * tint, atol=1.0e-6
            )
            previous = path.read_bytes()
            reader = UsdShade.Shader.Define(stage, "/Asset/Material/Reader")
            reader.CreateIdAttr("UsdPrimvarReader_float3")
            color.ConnectToSource(reader.ConnectableAPI(), "result")
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(ValueError, "connected color"):
                save_scene(path, [obj])
            self.assertEqual(path.read_bytes(), previous)
            obj.color = (1.0, 1.0, 1.0)
            save_scene(path, [obj])

    def test_reject_deformable_content_before_partial_import(self):
        """Allow visual guides but reject explicit deformable simulation in rigid assets."""
        from pxr import Gf, Sdf, UsdGeom

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.usda"
            stage = self.asset(path)
            guide = UsdGeom.BasisCurves.Define(stage, "/Asset/grasp_identifier_curve")
            guide.CreatePurposeAttr(UsdGeom.Tokens.guide)
            guide.CreateTypeAttr(UsdGeom.Tokens.linear)
            guide.CreateCurveVertexCountsAttr([2])
            guide.CreatePointsAttr([Gf.Vec3f(0.0), Gf.Vec3f(1.0)])
            points = UsdGeom.Points.Define(stage, "/Asset/GuidePoints")
            points.CreatePurposeAttr(UsdGeom.Tokens.guide)
            points.CreatePointsAttr([Gf.Vec3f(0.0)])
            stage.GetRootLayer().Save()
            template = prepare_usd(str(path))
            self.assertEqual(template.body_count, 1)
            self.assertEqual(template.shape_count, 1)
            self.assertEqual(template.particle_count, 0)
            points.GetPrim().SetMetadata(
                "apiSchemas", Sdf.TokenListOp.Create(prependedItems=["NewtonPointsDeformableSimAPI"])
            )
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(ValueError, "rigid USD assets only"):
                prepare_usd(str(path))

    def test_failed_save_preserves_destination_and_external_detection(self):
        """Keep valid files intact and reject malformed authored metadata on load."""
        from pxr import Sdf, Usd, UsdUtils

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.usda"
            save_scene(path, [self.object()])
            previous = path.read_bytes()
            with self.assertRaises((ValueError, RuntimeError)):
                save_scene(path, [self.object("usd", source=str(Path(directory) / "missing.usda"))])
            self.assertEqual(path.read_bytes(), previous)
            for changes in (
                {"name": " "},
                {"visible": 1},
                {"position": (1e100, 0, 0)},
                {"mass": 1e100},
                {"size": (1e-100,) * 3},
                {"mass": 1e-100},
            ):
                with self.subTest(save=changes):
                    with self.assertRaises(ValueError):
                        save_scene(path, [self.object(**changes)])
                    self.assertEqual(path.read_bytes(), previous)
            stage = Usd.Stage.Open(str(path))
            prim = stage.GetPrimAtPath("/World/Object_0")
            definition = json.loads(prim.GetCustomDataByKey("newtonSceneAuthoring"))
            for changes in (
                {"name": " "},
                {"dynamic": "false"},
                {"kind": "sphere", "size": [2.0, 2.00001, 2.0]},
                {"rotation": [0.0, 0.0, 0.0, 2.0]},
            ):
                with self.subTest(load=changes):
                    prim.SetCustomDataByKey("newtonSceneAuthoring", json.dumps({**definition, **changes}))
                    stage.GetRootLayer().Save()
                    with self.assertRaises(ValueError):
                        load_scene(path)
            external = Path(directory) / "external.usda"
            source_stage = self.asset(external)
            with self.assertRaises(NotAuthoringSceneError):
                load_scene(external)
            nested = Path(directory) / "nested"
            nested.mkdir()
            dependency = nested / "dependency.usda"
            dependency_stage = Usd.Stage.CreateNew(str(dependency))
            dependency_prim = dependency_stage.DefinePrim("/Dependency", "Scope")
            dependency_stage.SetDefaultPrim(dependency_prim)
            texture = nested / "texture.png"
            texture.write_bytes(b"Dependency scanner does not decode the image")
            dependency_prim.CreateAttribute("asset:texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("texture.png"))
            dependency_prim.CreateAttribute("asset:optional", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath("missing.usda")
            )
            dependency_stage.GetRootLayer().Save()
            source_stage.DefinePrim("/Dependency", "Scope").GetReferences().AddReference(
                "nested/dependency.usda", "/Dependency"
            )
            source_stage.GetRootLayer().Save()
            self.assertEqual(
                set(_asset_dependencies(external)),
                {str(external), str(dependency), str(texture), str(nested / "missing.usda")},
            )
            # The same package-member references watch the archive, not fictitious filesystem members.
            dependency_prim.RemoveProperty("asset:optional")
            dependency_stage.GetRootLayer().Save()
            package = Path(directory) / "asset.usdz"
            self.assertTrue(UsdUtils.CreateNewUsdzPackage(str(external), str(package)))
            self.assertEqual(_asset_dependencies(package), (str(package),))
            dependency_prim.CreateAttribute("asset:optional", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath("missing.usda")
            )
            dependency_stage.GetRootLayer().Save()
            imported = self.object("usd", source=external, color=(1.0, 1.0, 1.0))
            targets = [external, dependency, nested / "missing.usda"]
            if os.name == "posix":
                for name, original in (("root-alias.usda", external), ("texture-alias.usda", texture)):
                    alias = Path(directory) / name
                    alias.hardlink_to(original)
                    targets.append(alias)
                alias = Path(directory) / "root-symlink.usda"
                alias.symlink_to(external)
                targets.append(alias)
            contents = {item: item.read_bytes() for item in (external, dependency, texture)}
            for target in targets:
                with self.subTest(destination=target.name):
                    with self.assertRaisesRegex(ValueError, "dependency"):
                        save_scene(target, [imported])
                    for item, data in contents.items():
                        self.assertEqual(item.read_bytes(), data)
            self.assertFalse((nested / "missing.usda").exists())
            # Existing authored documents remain ordinary overwrite destinations.
            save_scene(path, [self.object(name="Replacement")])
            self.assertIn(b"Replacement", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
