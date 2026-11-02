# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the USD-to-native visual mesh integration through public APIs."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.geometry
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _add_mesh_prim,
    _add_physics_attachment,
    _apply_deformable_body_api,
    _author_deformable_element_array,
    _author_pose,
    _author_unit_tet,
    _deformable_stage,
)
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUSDDeformableVisual(unittest.TestCase):
    @staticmethod
    def _bind(prim, points):
        _author_pose(prim, "bind", np.asarray(points, dtype=float).tolist())

    @classmethod
    def _object(cls, stage, path, family):
        from pxr import UsdGeom

        _apply_deformable_body_api(UsdGeom.Xform.Define(stage, path).GetPrim())
        if family == "curve":
            sim = _add_cable_curve(stage, f"{path}/Sim", [(0, 0, 1), (1, 0, 1), (2, 0, 1)])
        elif family == "surface":
            sim = _add_cloth_mesh(stage, f"{path}/Sim")
            _author_deformable_element_array(sim.GetPrim(), "thicknesses", [0.001], "constant")
        else:
            sim = _author_unit_tet(stage, f"{path}/Sim", sim_api=True)
        points = np.array([(0.1, 0.1, 1.0), (0.3, 0.1, 1.0), (0.1, 0.3, 1.0)], dtype=np.float32)
        mesh = _add_mesh_prim(stage, f"{path}/Skin", points)
        return sim, mesh, points

    def test_mixed_import_replication_and_motion(self):
        """Import all three bindings and move their skins independently in two worlds."""
        stage = _deformable_stage()
        for family in ("curve", "surface", "volume"):
            self._object(stage, f"/World/{family}", family)
        prototype = newton.ModelBuilder()
        result = prototype.add_usd(stage, return_deformable_results=True)
        visual_indices = result["path_deformable_visual_map"]
        self.assertEqual(
            visual_indices, {f"/World/{family}/Skin": i for i, family in enumerate(("curve", "surface", "volume"))}
        )
        physics = newton.ModelBuilder()
        physics_result = physics.add_usd(stage, load_visual_shapes=False, return_deformable_results=True)
        self.assertEqual(physics_result["path_deformable_visual_map"], {})
        for field in ("particle_q", "particle_mass", "body_q", "body_mass", "joint_q"):
            np.testing.assert_array_equal(getattr(prototype, field), getattr(physics, field))
        self.assertEqual(physics.finalize(device="cpu").deformable_visual_mesh_count, 0)
        builder = newton.ModelBuilder()
        builder.replicate(prototype, 2)
        # Import runs on the host; device evaluation is covered by the native mesh tests.
        model = builder.finalize(device="cpu")
        self.assertEqual(model.deformable_visual_mesh_count, 6)
        self.assertEqual([mesh.world for mesh in model.deformable_visual_meshes], [0, 0, 0, 1, 1, 1])
        for path, index in visual_indices.items():
            self.assertEqual(model.deformable_visual_meshes[index].label, path)
        self.assertEqual(
            {mesh.kind for mesh in model.deformable_visual_meshes},
            {
                newton.geometry.DeformableVisualMesh.Kind.BODY,
                newton.geometry.DeformableVisualMesh.Kind.TRIANGLE,
                newton.geometry.DeformableVisualMesh.Kind.TETRAHEDRON,
            },
        )
        state = model.state()
        output = newton.geometry.DeformableVisuals(model).update(state)
        before = [output.get_points(mesh).numpy().copy() for mesh in model.deformable_visual_meshes]
        particles = state.particle_q.numpy().copy()
        particles[model.particle_world.numpy() == 1, 2] += 0.5
        state.particle_q.assign(particles)
        bodies = state.body_q.numpy().copy()
        bodies[model.body_world.numpy() == 1, 2] += 0.5
        state.body_q.assign(bodies)
        output.update(state)
        for mesh, points in zip(model.deformable_visual_meshes, before, strict=True):
            np.testing.assert_allclose(
                output.get_points(mesh).numpy(), points + np.array([0, 0, 0.5 * mesh.world]), atol=1e-6
            )

    def test_bind_poses_uv_seams_and_transforms(self):
        """Pair visual and simulation bind poses before applying their common world transform.

        The simulation geometry starts away from its bind pose along and across the
        surface, so a skin embedded in the initial geometry instead of the bind pose
        would land at the wrong spot.
        """
        from pxr import Sdf, UsdGeom, UsdShade

        for family in ("curve", "surface", "volume"):
            with self.subTest(family=family):
                stage = _deformable_stage()
                sim, mesh, _ = self._object(stage, "/World/Object", family)
                points = np.array([(0.1, 0.1, 1), (0.3, 0.1, 1), (0.3, 0.3, 1), (0.1, 0.3, 1)])
                sim_points = np.asarray(sim.GetPointsAttr().Get())
                self._bind(sim.GetPrim(), sim_points)
                sim.GetPointsAttr().Set((sim_points + np.array([0.05, 0, 2])).tolist())
                mesh.CreatePointsAttr([(9, 9, 9)] * 4)
                corners = [0, 1, 2, 0, 2, 3]
                mesh.CreateFaceVertexCountsAttr([3, 3])
                mesh.CreateFaceVertexIndicesAttr(corners)
                self._bind(mesh.GetPrim(), points)
                uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
                uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
                # Point 0 has different UVs on its two faces.
                uv_indices = [3, 2, 1, 0, 1, 3]
                uv.SetIndices(uv_indices)
                material = UsdShade.Material.Define(stage, "/World/Material")
                shader = UsdShade.Shader.Define(stage, "/World/Material/Surface")
                shader.CreateIdAttr("UsdPreviewSurface")
                texture = UsdShade.Shader.Define(stage, "/World/Material/Texture")
                texture.CreateIdAttr("UsdUVTexture")
                texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("albedo.png"))
                texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
                reader = UsdShade.Shader.Define(stage, "/World/Material/Reader")
                reader.CreateIdAttr("UsdPrimvarReader_float2")
                reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
                reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                transform = UsdShade.Shader.Define(stage, "/World/Material/Transform")
                transform.CreateIdAttr("UsdTransform2d")
                transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    reader.ConnectableAPI(), "result"
                )
                transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set((2.0, 0.5))
                transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(90.0)
                transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set((0.25, 0.1))
                transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)
                texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                    transform.ConnectableAPI(), "result"
                )
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    texture.ConnectableAPI(), "rgb"
                )
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/Object")).AddTranslateOp().Set((3, 4, 5))
                builder = newton.ModelBuilder()
                if family == "surface":
                    subset = UsdShade.MaterialBindingAPI(mesh).CreateMaterialBindSubset("Patch", [1])
                    subset_material = UsdShade.Material.Define(stage, "/World/PatchMaterial")
                    subset_material.GetPrim().GetReferences().AddInternalReference(material.GetPath())
                    stage.GetPrimAtPath("/World/PatchMaterial/Texture").GetAttribute("inputs:file").Set(
                        Sdf.AssetPath("patch.png")
                    )
                    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(subset_material)
                    with self.assertLogs("newton", level="WARNING") as logs:
                        builder.add_usd(stage)
                    self.assertTrue(any("material subsets" in message for message in logs.output))
                else:
                    builder.add_usd(stage)
                model = builder.finalize(device="cpu")
                self.assertEqual(model.deformable_visual_mesh_count, 1)
                self.assertEqual(model.deformable_visual_meshes[0].texture, "albedo.png")
                output = newton.geometry.DeformableVisuals(model).update(model.state())
                np.testing.assert_allclose(output.points.numpy(), points[corners] + np.array([3.05, 4, 7]), atol=2e-6)
                np.testing.assert_allclose(
                    model.deformable_visual_meshes[0].uvs.numpy(),
                    np.array([[0.25, 0.1], [0.25, 2.1], [-0.25, 2.1], [-0.25, 0.1]])[uv_indices],
                    atol=1e-6,
                )

    def test_rotated_cable_bind_pose_keeps_roll(self):
        """Carry a cable skin through a bind pose that points the cable another way.

        The segment frames are transported from the initial tangent to the bind
        tangent, so the skin turns with the cable instead of rolling around it.
        """
        stage = _deformable_stage()
        sim, mesh, _ = self._object(stage, "/World/Object", "curve")
        bind = np.array(sim.GetPointsAttr().Get())
        # Short segments must use the same unit-tangent frame transport as long ones.
        bind[:, :2] *= 0.1
        self._bind(sim.GetPrim(), bind)
        # Initial geometry: the bind cable turned a quarter turn about +z.
        sim.GetPointsAttr().Set(np.column_stack([-bind[:, 1], bind[:, 0], bind[:, 2]]).tolist())
        skin = np.array([(0.5, 0.1, 1.0), (1.5, 0.1, 1.0), (1.0, 0.1, 1.2)])
        skin[:, :2] *= 0.1
        mesh.GetPointsAttr().Set(skin.tolist())
        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize(device="cpu")
        output = newton.geometry.DeformableVisuals(model).update(model.state())
        expected = np.column_stack([-skin[:, 1], skin[:, 0], skin[:, 2]])
        np.testing.assert_allclose(output.points.numpy(), expected, atol=2e-6)

    def test_graph_bind_pose_and_collapse(self):
        """Keep graph skin orientation through a translated bind pose and fixed-joint collapse."""

        stage = _deformable_stage()
        reference = [np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1]]), np.array([[1, 0, 1], [1, 1, 1], [1, 2, 1]])]
        skins = []
        for index, points in enumerate(reference):
            path = f"/World/Part{index}"
            sim, mesh, vertices = self._object(stage, path, "curve")
            sim.GetPointsAttr().Set((points + np.array([0, 0, 2])).tolist())
            self._bind(sim.GetPrim(), points)
            # One curve has authored twist; the other keeps its default frames.
            if index == 0:
                sim.CreateNormalsAttr([(0, 0, 1)] * len(points))
                sim.SetNormalsInterpolation("vertex")
            vertices += [index, index, 0]
            mesh.GetPointsAttr().Set(vertices.tolist())
            skins.append(vertices)
        _add_physics_attachment(
            stage,
            "/World/Junction",
            src0="/World/Part0/Sim",
            src1="/World/Part1/Sim",
            type0="point",
            type1="point",
            indices0=[1],
            indices1=[0],
        )
        builder = newton.ModelBuilder()
        root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_free(root), builder.add_joint_fixed(root, child)])
        builder.add_usd(stage, collapse_fixed_joints=True)
        model = builder.finalize(device="cpu")
        output = newton.geometry.DeformableVisuals(model).update(model.state())
        np.testing.assert_allclose(output.points.numpy(), np.concatenate(skins) + np.array([0, 0, 2]), atol=2e-6)

    def test_invalid_visuals_preserve_physics_and_valid_siblings(self):
        """Skip malformed graphics or simulation bind poses without losing valid siblings or physics."""
        families = ("curve", "surface", "volume")
        for invalid_sim in (False, True):
            with self.subTest(invalid_sim=invalid_sim):
                stage = _deformable_stage()
                for family in families:
                    sim, mesh, points = self._object(stage, f"/World/{family}/Bad", family)
                    self._bind(sim.GetPrim() if invalid_sim else mesh.GetPrim(), points[:1])
                    self._object(stage, f"/World/{family}/Good", family)
                builder = newton.ModelBuilder()
                with self.assertLogs("newton", level="WARNING") as logs:
                    result = builder.add_usd(stage, return_deformable_results=True)
                for family in families:
                    self.assertTrue(any(f"/World/{family}/Bad/Skin" in message for message in logs.output))
                model = builder.finalize(device="cpu")
                self.assertEqual(
                    [mesh.label for mesh in model.deformable_visual_meshes],
                    [f"/World/{family}/Good/Skin" for family in families],
                )
                self.assertEqual(
                    result["path_deformable_visual_map"],
                    {mesh.label: mesh.index for mesh in model.deformable_visual_meshes},
                )
                self.assertEqual(model.body_count, 4)  # Two two-segment cables.
                self.assertEqual(model.particle_count, 16)  # Two four-point cloths and two four-point volumes.
                self.assertEqual(model.tet_count, 2)

    def test_import_filter_and_nested_ownership(self):
        """Respect regex filters, purposes, and body ownership without loading frozen copies of skins."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        self._object(stage, "/World/Outer", "volume")
        self._object(stage, "/World/Outer/Nested", "volume")
        points = [(0.1, 0.1, 1), (0.3, 0.1, 1), (0.1, 0.3, 1)]
        _add_mesh_prim(stage, "/World/Outer/Hidden", points).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        _add_mesh_prim(stage, "/World/Outer/Guide", points).CreatePurposeAttr(UsdGeom.Tokens.guide)
        _add_mesh_prim(stage, "/World/Outer/Proxy", points).CreatePurposeAttr(UsdGeom.Tokens.proxy)
        _add_mesh_prim(stage, "/World/Outer/Skip", points)
        builder = newton.ModelBuilder()
        result = builder.add_usd(stage, ignore_paths=[r".*/Skip$"])
        self.assertNotIn("path_deformable_visual_map", result)  # The visual map is opt-in.
        model = builder.finalize(device="cpu")
        self.assertEqual(
            [mesh.label for mesh in model.deformable_visual_meshes],
            ["/World/Outer/Skin", "/World/Outer/Proxy", "/World/Outer/Nested/Skin"],
        )
        self.assertEqual(model.shape_count, 0)
        self.assertNotIn("/World/Outer/Skin", result["path_shape_map"])

        subtree = newton.ModelBuilder()
        subtree.add_usd(stage, root_path="/World/Outer/Sim")
        self.assertEqual(subtree.finalize(device="cpu").deformable_visual_mesh_count, 0)

    def test_rigid_ancestor_and_unsupported_graphics(self):
        """Keep skins out of rigid geometry and skip unsupported graphics without losing physics."""
        from pxr import UsdGeom, UsdPhysics

        stage = _deformable_stage()
        rigid = UsdGeom.Xform.Define(stage, "/World/Rigid")
        UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
        self._object(stage, "/World/Rigid/Soft", "surface")
        UsdGeom.Points.Define(stage, "/World/Rigid/Soft/Points").CreatePointsAttr([(0, 0, 1)])
        sites = [f"/World/Rigid/Soft/{api}" for api in ("NewtonSiteAPI", "MjcSiteAPI")]
        for path in sites:
            site = _add_mesh_prim(stage, path, [(0, 0, 1), (1, 0, 1), (0, 1, 1)])
            site.GetPrim().AddAppliedSchema(path.rsplit("/", 1)[1])
        for load_visuals, load_sites in ((True, True), (False, True), (True, False)):
            with self.subTest(load_visuals=load_visuals, load_sites=load_sites):
                builder = newton.ModelBuilder()
                with self.assertLogs("newton", level="WARNING") as logs:
                    result = builder.add_usd(stage, load_visual_shapes=load_visuals, load_sites=load_sites)
                self.assertTrue(any("unsupported graphics" in message for message in logs.output))
                self.assertNotIn("/World/Rigid/Soft/Skin", result["path_shape_map"])
                for path in sites:
                    self.assertEqual(path in result["path_shape_map"], load_sites)
                # The cloth skin never becomes a rigid shape; only the sites do.
                self.assertEqual(builder.shape_count, len(sites) if load_sites else 0)
                model = builder.finalize(device="cpu")
                self.assertGreater(model.particle_count, 0)
                self.assertEqual(model.deformable_visual_mesh_count, int(load_visuals))

    def test_cable_collision_visibility(self):
        """Hide cable capsules only when a visible skin was imported successfully."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        _, skin, points = self._object(stage, "/World/Cable", "curve")
        # This point binds to the second segment but is not part of any rendered face.
        skin.GetPointsAttr().Set(np.vstack([points, [1.5, 0.1, 1.0]]).tolist())
        for visible in (True, False):
            with self.subTest(visible=visible):
                skin.CreateVisibilityAttr(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)
                builder = newton.ModelBuilder()
                builder.add_usd(stage, hide_collision_shapes=True, force_show_colliders=True)
                self.assertGreater(builder.shape_count, 0)
                for body, flags in zip(builder.shape_body, builder.shape_flags, strict=True):
                    # The small skin covers only the first of the two cable segments.
                    self.assertEqual(bool(flags & newton.ShapeFlags.VISIBLE), not visible or body == 1)
                self.assertTrue(all(flags & newton.ShapeFlags.COLLIDE_SHAPES for flags in builder.shape_flags))


if __name__ == "__main__":
    unittest.main()
