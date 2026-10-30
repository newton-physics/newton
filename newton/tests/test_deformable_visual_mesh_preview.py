# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preview integration tests for imported visuals, cameras, and scene parity."""

import os
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton import usd
from newton._src.utils.import_usd_deformable_visual import _sim_bind_positions
from newton.examples.sensors.example_deformable_visual_mesh_camera import (
    Example as DeformableVisualMeshCameraExample,
)
from newton.examples.sensors.example_deformable_visual_mesh_camera import (
    _CameraRecorder as DeformableVisualMeshCameraRecorder,
)
from newton.tests._usd_deformable_test_utils import (
    _add_cable_curve,
    _add_cloth_mesh,
    _author_deformable_element_array,
    _deformable_stage,
)
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal
from newton.utils import load_texture


def _skin(model, mesh, state):
    """Skin ``mesh`` through the model's per-kind batches and return its points."""
    return newton.geometry.DeformableVisuals(model).update(state).get_points(mesh).numpy()


class TestDeformableVisualMeshNamespaces(unittest.TestCase):
    def test_replicate_validates_visual_namespace_arguments(self):
        """Require one destination namespace per world and a paired source namespace."""
        source = newton.ModelBuilder()

        with self.assertRaisesRegex(ValueError, "must be passed together"):
            newton.ModelBuilder().replicate(source, 2, source_path_prefix="/World/envs/env_0")
        with self.assertRaisesRegex(ValueError, "must be passed together"):
            newton.ModelBuilder().replicate(
                source,
                2,
                destination_path_prefixes=["/World/envs/env_0", "/World/envs/env_1"],
            )
        with self.assertRaisesRegex(ValueError, "must contain 2 entries"):
            newton.ModelBuilder().replicate(
                source,
                2,
                source_path_prefix="/World/envs/env_0",
                destination_path_prefixes=["/World/envs/env_0"],
            )


class TestDeformableVisualMeshUSDImport(unittest.TestCase):
    """AOUSD graphics-geometry discovery: hierarchy, bind poses, ownership."""

    @staticmethod
    def _stage():
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        return stage

    @staticmethod
    def _add_volume_body(stage, path):
        """A deformable body root with a unit-tet simulation mesh child."""
        from pxr import UsdGeom

        body = UsdGeom.Xform.Define(stage, path)
        body.GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
        tet = UsdGeom.TetMesh.Define(stage, f"{path}/Sim")
        tet.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        tet.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
        tet.GetPrim().AddAppliedSchema("PhysicsVolumeDeformableSimAPI")
        # Author collision like the importer fixtures: no gating warning under strict runs.
        tet.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")
        return body, tet

    @staticmethod
    def _add_graphics_mesh(stage, path, points=None):
        from pxr import UsdGeom

        gfx = UsdGeom.Mesh.Define(stage, path)
        pts = points if points is not None else [(0.2, 0.2, 0.2), (0.4, 0.2, 0.2), (0.2, 0.4, 0.2)]
        gfx.CreatePointsAttr([tuple(p) for p in pts])
        gfx.CreateFaceVertexCountsAttr([3])
        gfx.CreateFaceVertexIndicesAttr([0, 1, 2])
        return gfx

    @classmethod
    def _add_volume_visual(cls, stage, body_path):
        """Author one volume body with its default ``Skin`` graphics mesh."""
        cls._add_volume_body(stage, body_path)
        return cls._add_graphics_mesh(stage, f"{body_path}/Skin")

    def _import(self, stage, **kwargs):
        builder = newton.ModelBuilder()
        builder.add_usd(stage, **kwargs)
        return builder

    def test_untagged_mesh_under_volume_body_imports_and_skins(self):
        """An untagged graphics Mesh under a volume deformable body becomes a tet
        visual mesh with USD ownership metadata, without any custom relationship
        and without return_deformable_results."""
        stage = self._stage()
        self._add_volume_visual(stage, "/World/Tire")

        builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 1)
        rm = model.deformable_visual_meshes[0]
        self.assertEqual(rm.kind, newton.geometry.DeformableVisualMesh.Kind.TETRAHEDRON)
        self.assertEqual(rm._body_path, "/World/Tire")
        self.assertEqual(rm._sim_path, "/World/Tire/Sim")
        self.assertEqual(rm.label, "/World/Tire/Skin")

        # Rigid translation of the soft body carries the skinned vertices exactly.
        state = model.state()
        rest = _skin(model, rm, state)
        shift = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        state.particle_q = wp.array(state.particle_q.numpy() + shift, dtype=wp.vec3)
        assert_np_equal(_skin(model, rm, state), rest + shift, tol=1.0e-5)

    def test_replicate_rebases_visual_ownership_paths(self):
        """Rebase imported visual identities and drivers into each replicated namespace."""
        stage = self._stage()
        self._add_volume_visual(stage, "/World/envs/env_0/Object")
        template = self._import(stage)

        scene = newton.ModelBuilder()
        scene.replicate(
            template,
            2,
            source_path_prefix="/World/envs/env_0",
            destination_path_prefixes=["/World/envs/env_0", "/World/envs/env_1"],
        )
        model = scene.finalize()

        self.assertEqual(model.deformable_visual_mesh_count, 2)
        for world, mesh in enumerate(model.deformable_visual_meshes):
            destination = f"/World/envs/env_{world}/Object"
            with self.subTest(world=world):
                self.assertEqual(mesh.world, world)
                self.assertEqual(mesh.label, f"{destination}/Skin")
                self.assertEqual(mesh._body_path, destination)
                self.assertEqual(mesh._sim_path, f"{destination}/Sim")
                self.assertEqual(mesh._graphics_path, f"{destination}/Skin")
                self.assertEqual(mesh.parent.numpy().tolist(), [world, world, world])

        source_mesh = template.finalize().deformable_visual_meshes[0]
        self.assertEqual(source_mesh._graphics_path, "/World/envs/env_0/Object/Skin")

    def test_replicate_combines_label_and_visual_path_prefixes(self):
        """Prefix model labels while replacing deformable visual USD namespaces."""
        stage = self._stage()
        self._add_volume_visual(stage, "/World/envs/env_0/Object")
        template = self._import(stage)
        marker = template.add_body(label="marker")
        template.add_shape_box(marker, label="marker_box")

        scene = newton.ModelBuilder()
        scene.replicate(
            template,
            2,
            label_prefixes=["left", "right"],
            source_path_prefix="/World/envs/env_0",
            destination_path_prefixes=["/World/envs/env_4", "/World/envs/env_9"],
        )
        model = scene.finalize()

        self.assertEqual(scene.body_label, ["left/marker", "right/marker"])
        self.assertEqual(scene.shape_label, ["left/marker_box", "right/marker_box"])
        self.assertEqual(
            [mesh._graphics_path for mesh in model.deformable_visual_meshes],
            ["/World/envs/env_4/Object/Skin", "/World/envs/env_9/Object/Skin"],
        )

    def test_replicate_rejects_visual_paths_outside_source_namespace(self):
        """Reject source-prefix lookalikes instead of retaining stale ownership paths."""
        stage = self._stage()
        self._add_volume_visual(stage, "/World/envs/env_0/Object")
        self._add_volume_visual(stage, "/World/envs/env_01/Object")
        template = self._import(stage)

        scene = newton.ModelBuilder()
        with self.assertRaisesRegex(ValueError, "outside source_path_prefix"):
            scene.replicate(
                template,
                1,
                source_path_prefix="/World/envs/env_0",
                destination_path_prefixes=["/World/envs/env_1"],
            )
        self.assertEqual((scene.particle_count, scene.tet_count, scene.world_count), (0, 0, 0))
        self.assertEqual(scene.finalize().deformable_visual_mesh_count, 0)

    def test_multiple_graphics_meshes_and_nested_bodies(self):
        """Multiple graphics meshes under one body all import; a nested deformable
        body's graphics are never assigned to the outer body."""
        stage = self._stage()
        self._add_volume_body(stage, "/World/Outer")
        self._add_graphics_mesh(stage, "/World/Outer/SkinA")
        self._add_graphics_mesh(stage, "/World/Outer/SkinB")
        self._add_volume_body(stage, "/World/Outer/Inner")
        self._add_graphics_mesh(stage, "/World/Outer/Inner/Skin")

        builder = self._import(stage)
        model = builder.finalize()
        by_path = {rm.label: rm for rm in model.deformable_visual_meshes}
        self.assertEqual(set(by_path), {"/World/Outer/SkinA", "/World/Outer/SkinB", "/World/Outer/Inner/Skin"})
        self.assertEqual(by_path["/World/Outer/SkinA"]._body_path, "/World/Outer")
        self.assertEqual(by_path["/World/Outer/Inner/Skin"]._body_path, "/World/Outer/Inner")
        # The nested body's mesh embeds in the nested body's own tets.
        self.assertNotEqual(
            by_path["/World/Outer/SkinA"].parent.numpy()[0], by_path["/World/Outer/Inner/Skin"].parent.numpy()[0]
        )

    def test_nested_rigid_body_mesh_is_not_deformable_visual(self):
        """A mesh owned by a nested rigid body is not claimed by the outer deformable."""
        from pxr import UsdGeom, UsdPhysics

        stage = self._stage()
        self._add_volume_body(stage, "/World/Soft")
        rigid = UsdGeom.Xform.Define(stage, "/World/Soft/Rigid")
        UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
        self._add_graphics_mesh(stage, "/World/Soft/Rigid/Skin")

        builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 0)

    def test_collision_marked_geometry_is_not_visual(self):
        """A Mesh marked with a collision API under the body is not a visual mesh."""
        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        collider = self._add_graphics_mesh(stage, "/World/Tire/Collider")
        collider.GetPrim().AddAppliedSchema("PhysicsCollisionAPI")

        with self.assertWarnsRegex(UserWarning, "approximated by the simulation geometry"):
            builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 0)

    def test_bind_pose_is_honored(self):
        """PhysicsDeformablePoseAPI bindPose drives the embedding: a mesh whose
        default points are far away but whose bind pose lies inside the tet
        embeds without an outside-domain clamp warning."""
        from pxr import Sdf

        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        gfx = self._add_graphics_mesh(
            stage, "/World/Tire/Skin", points=[(9.0, 9.0, 9.0), (9.4, 9.0, 9.0), (9.0, 9.4, 9.0)]
        )
        gfx.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        gfx.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        gfx.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.2, 0.2, 0.2), (0.4, 0.2, 0.2), (0.2, 0.4, 0.2)]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder = self._import(stage)
        self.assertFalse(any("clamped" in str(w.message) for w in caught))
        model = builder.finalize()
        rm = model.deformable_visual_meshes[0]
        # The bind-pose vertices (inside the tet) are the rest vertices.
        self.assertLess(float(rm.rest_vertices.numpy().max()), 1.0)

    def test_bind_pose_survives_face_varying_uvs(self):
        """Expand authored bind points with UV corners, including coincident source points."""
        from pxr import Sdf, UsdGeom

        bind_points = np.array([[0.1, 0.1, 0.1], [0.3, 0.1, 0.1], [0.3, 0.3, 0.1], [0.1, 0.3, 0.1]])
        for indexed in (False, True):
            with self.subTest(indexed=indexed):
                stage = self._stage()
                self._add_volume_body(stage, "/World/Soft")
                mesh = UsdGeom.Mesh.Define(stage, "/World/Soft/Skin")
                mesh.CreatePointsAttr([(9.0, 9.0, 9.0)] * 4)
                mesh.CreateFaceVertexCountsAttr([4])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                mesh.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
                mesh.GetPrim().CreateAttribute(
                    "physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray
                ).Set(["bindPose"])
                mesh.GetPrim().CreateAttribute(
                    "physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray
                ).Set(bind_points.tolist())
                uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
                )
                uv.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
                if indexed:
                    uv.SetIndices([3, 2, 1, 0])

                model = self._import(stage).finalize()
                self.assertEqual(model.deformable_visual_mesh_count, 1)
                visuals = newton.geometry.DeformableVisuals(model)
                visuals.update(model.state())
                corners = [0, 1, 2, 0, 2, 3]
                assert_np_equal(visuals.points.numpy(), bind_points[corners], tol=1.0e-6)
                expected_uvs = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
                if indexed:
                    expected_uvs = expected_uvs[::-1]
                assert_np_equal(model.deformable_visual_meshes[0].uvs.numpy(), expected_uvs[corners], tol=1.0e-6)

    def test_invalid_simulation_bind_pose_skips_visual_binding(self):
        """An authored simulation bind pose with the wrong count is not replaced silently."""
        from pxr import Sdf

        stage = self._stage()
        _, sim = self._add_volume_body(stage, "/World/Tire")
        self._add_graphics_mesh(stage, "/World/Tire/Skin")
        sim.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        )

        with self.assertLogs("newton", level="WARNING") as logs:
            builder = self._import(stage)
        self.assertIn("point count must match", "\n".join(logs.output))

        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 0)

    def test_sim_bind_pose_count_must_match_imported_particle_range(self):
        """A valid USD pose cannot silently fall back when lowering realizes fewer particles."""
        from pxr import Sdf

        stage = self._stage()
        _, sim = self._add_volume_body(stage, "/World/Tire")
        sim.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )
        ctx = SimpleNamespace(
            stage=stage,
            builder=newton.ModelBuilder(),
            get_prim_world_mat=lambda *_args: wp.mat44_identity(),
            incoming_world_xform=wp.transform_identity(),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaisesRegex(ValueError, "/World/Tire/Sim: bind pose has 4 points.*3 particles"):
                _sim_bind_positions(ctx, "/World/Tire/Sim", (0, 3))

    def test_degenerate_simulation_bind_tet_skips_visual_binding(self):
        """A visual cannot bind to a tetrahedron collapsed in the simulation bind pose."""
        from pxr import Sdf

        stage = self._stage()
        _, sim = self._add_volume_body(stage, "/World/Tire")
        self._add_graphics_mesh(stage, "/World/Tire/Skin")
        sim.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
        )

        with self.assertLogs("newton", level="WARNING") as logs:
            builder = self._import(stage)
        self.assertIn("owning tet range contains no non-degenerate tetrahedron", "\n".join(logs.output))

        self.assertEqual(builder.finalize().deformable_visual_mesh_count, 0)

    def test_degenerate_simulation_bind_triangle_skips_visual_binding(self):
        """A visual cannot bind to triangles collapsed in the simulation bind pose."""
        from pxr import Sdf, UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        sim = _add_cloth_mesh(stage, "/World/Body/Sim")
        _author_deformable_element_array(sim.GetPrim(), "thicknesses", [0.001], "constant")
        self._add_graphics_mesh(stage, "/World/Body/Skin")
        sim.GetPrim().AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
            ["bindPose"]
        )
        sim.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
            [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0), (3.0, 0.0, 1.0)]
        )

        with self.assertLogs("newton", level="WARNING") as logs:
            builder = self._import(stage)
        self.assertIn("owning triangle range contains no non-degenerate triangle", "\n".join(logs.output))

        self.assertEqual(builder.finalize().deformable_visual_mesh_count, 0)

    def test_invalid_visual_bind_pose_skips_only_that_visual(self):
        """A malformed visual is diagnosed without discarding a valid sibling visual."""
        from pxr import Sdf

        for problem in ("non_finite", "missing_points", "multiple_poses", "sampled_geometry"):
            with self.subTest(problem=problem):
                stage = self._stage()
                self._add_volume_body(stage, "/World/Tire")
                bad = self._add_graphics_mesh(stage, "/World/Tire/BadSkin")
                self._add_graphics_mesh(stage, "/World/Tire/GoodSkin")
                prim = bad.GetPrim()
                prim.AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
                prim.CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
                    ["bindPose"]
                )
                points = [(0.2, 0.2, 0.2), (0.4, 0.2, 0.2), (0.2, 0.4, 0.2)]
                if problem == "non_finite":
                    points[0] = (float("nan"), 0.2, 0.2)
                if problem != "missing_points":
                    prim.CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
                        points
                    )
                if problem == "multiple_poses":
                    prim.AddAppliedSchema("PhysicsDeformablePoseAPI:other")
                    prim.CreateAttribute("physics:deformablePose:other:purposes", Sdf.ValueTypeNames.TokenArray).Set(
                        ["bindPose"]
                    )
                    prim.CreateAttribute("physics:deformablePose:other:points", Sdf.ValueTypeNames.Point3fArray).Set(
                        points
                    )
                if problem == "sampled_geometry":
                    bad.GetPointsAttr().Clear()
                    bad.GetPointsAttr().Set(points, 1.0)

                with self.assertLogs("newton", level="WARNING") as logs:
                    builder = self._import(stage)
                self.assertIn("/World/Tire/BadSkin", "\n".join(logs.output))
                model = builder.finalize()
                self.assertEqual(model.deformable_visual_mesh_count, 1)
                self.assertEqual(model.deformable_visual_meshes[0].label, "/World/Tire/GoodSkin")
                visuals = newton.geometry.DeformableVisuals(model)
                visuals.update(model.state())
                self.assertTrue(np.isfinite(visuals.points.numpy()).all())

    def test_transforms_embed_in_common_frame(self):
        """Sim and graphics prims with different local transforms meet in the
        world frame: a graphics mesh authored in a shifted local frame lands on
        the simulation geometry."""
        from pxr import Gf, UsdGeom

        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        gfx = self._add_graphics_mesh(
            stage, "/World/Tire/Skin", points=[(-0.8, 0.2, 0.2), (-0.6, 0.2, 0.2), (-0.8, 0.4, 0.2)]
        )
        UsdGeom.Xformable(gfx).AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder = self._import(stage)
        self.assertFalse(any("clamped" in str(w.message) for w in caught))
        model = builder.finalize()
        rest = model.deformable_visual_meshes[0].rest_vertices.numpy()
        assert_np_equal(rest[0], np.array([0.2, 0.2, 0.2], dtype=np.float32), tol=1.0e-6)

    def test_cloth_body_graphics_embed_in_triangles(self):
        """An untagged Mesh under a surface deformable body embeds into the
        cloth's owning triangle range."""
        stage = _deformable_stage()
        cloth = _add_cloth_mesh(stage, "/World/Body/Sim")
        _author_deformable_element_array(cloth.GetPrim(), "thicknesses", [0.001], "constant")
        from pxr import UsdGeom

        body = stage.GetPrimAtPath("/World/Body")
        if not body or not body.IsValid():
            body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        self._add_graphics_mesh(stage, "/World/Body/Skin", points=[(0.2, 0.2, 1.0), (0.6, 0.2, 1.0), (0.2, 0.6, 1.0)])
        del cloth

        builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 1)
        rm = model.deformable_visual_meshes[0]
        self.assertEqual(rm.kind, newton.geometry.DeformableVisualMesh.Kind.TRIANGLE)
        self.assertEqual(rm.weights.numpy().shape[1], 3)

    def test_cable_body_graphics_bind_to_segment_bodies(self):
        """An untagged Mesh under a cable deformable body binds to the curve's
        imported segment bodies and follows them."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        body = UsdGeom.Xform.Define(stage, "/World/Body").GetPrim()
        body.AddAppliedSchema("PhysicsDeformableBodyAPI")
        _add_cable_curve(stage, "/World/Body/Sim", [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.2, 0.0, 1.0), (0.3, 0.0, 1.0)])
        self._add_graphics_mesh(
            stage, "/World/Body/Skin", points=[(0.0, 0.02, 1.0), (0.15, 0.02, 1.0), (0.3, 0.02, 1.0)]
        )

        builder = self._import(stage)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 1)
        rm = model.deformable_visual_meshes[0]
        self.assertEqual(rm.kind, newton.geometry.DeformableVisualMesh.Kind.BODY)
        self.assertEqual(rm._body_path, "/World/Body")
        parents = rm.parent.numpy()
        self.assertTrue((parents >= 0).all() and (parents < model.body_count).all())

    def test_usd_cable_visual_survives_fixed_joint_collapse(self):
        """Keep imported cable visuals aligned when import collapses earlier bodies."""
        from pxr import UsdGeom

        stage = _deformable_stage()
        UsdGeom.Xform.Define(stage, "/World/Cable").GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
        _add_cable_curve(stage, "/World/Cable/Sim", [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)])
        vertices = np.array([[0.2, 0.1, 1.0], [0.3, 0.1, 1.0], [0.2, 0.2, 1.0]], dtype=np.float32)
        self._add_graphics_mesh(stage, "/World/Cable/Skin", points=vertices.tolist())
        builder = newton.ModelBuilder()
        root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_free(root), builder.add_joint_fixed(root, child)])
        builder.add_usd(stage, collapse_fixed_joints=True)
        model = builder.finalize()
        visuals = newton.geometry.DeformableVisuals(model)
        visuals.update(model.state())
        assert_np_equal(visuals.points.numpy(), vertices, tol=1.0e-6)

    def test_cable_visual_uses_simulation_bind_pose(self):
        """Move cable graphics from the authored bind pose to the current segment frames."""
        from pxr import Sdf, UsdGeom

        bind = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        vertices = np.array([[0.2, 0.1, 0], [0.3, 0.1, 0], [0.2, 0.2, 0]], dtype=np.float32)
        for rotate in (False, True):
            with self.subTest(rotate=rotate):
                rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) if rotate else np.eye(3)
                translation = np.array([0, 0, 10])
                stage = _deformable_stage()
                UsdGeom.Xform.Define(stage, "/World/Cable").GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
                curve = _add_cable_curve(stage, "/World/Cable/Sim", (bind @ rotation.T + translation).tolist())
                curve.CreateNormalsAttr([(0, 0, 1)] * 3)
                curve.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
                prim = curve.GetPrim()
                prim.AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
                prim.CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
                    ["bindPose"]
                )
                prim.CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
                    bind.tolist()
                )
                self._add_graphics_mesh(stage, "/World/Cable/Skin", points=vertices.tolist())
                model = self._import(stage).finalize()
                visuals = newton.geometry.DeformableVisuals(model)
                visuals.update(model.state())
                assert_np_equal(visuals.points.numpy(), vertices @ rotation.T + translation, tol=2.0e-6)

    def test_cable_bind_pose_preserves_curve_and_graph_mapping(self):
        """Use authored point ownership even when curves share a welded graph node."""
        from pxr import Sdf, UsdGeom

        from newton.tests._usd_deformable_test_utils import _add_physics_attachment  # noqa: PLC0415

        for graph in (False, True):
            with self.subTest(graph=graph):
                stage = self._stage()
                reference = [
                    np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
                    np.array([[1, 0, 0], [1, 1, 0], [1, 2, 0]], dtype=float),
                ]
                translation = np.array([0, 0, 10])
                skins = [
                    np.array([[0.2, 0.1, 0], [0.3, 0.1, 0], [0.2, 0.2, 0]]),
                    np.array([[1.1, 1.2, 0], [1.2, 1.2, 0], [1.1, 1.3, 0]]),
                ]
                UsdGeom.Xform.Define(stage, "/World/Cable").GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
                # The parent placement is independent of the bind-to-current motion.
                UsdGeom.Xformable(stage.GetPrimAtPath("/World/Cable")).AddTranslateOp().Set((3, 4, 5))
                if graph:
                    curves = [
                        _add_cable_curve(stage, f"/World/Cable/Part{i}/Sim", points + translation)
                        for i, points in enumerate(reference)
                    ]
                    for i in range(2):
                        stage.GetPrimAtPath(f"/World/Cable/Part{i}").AddAppliedSchema("PhysicsDeformableBodyAPI")
                        self._add_graphics_mesh(stage, f"/World/Cable/Part{i}/Skin", skins[i])
                    _add_physics_attachment(
                        stage,
                        "/World/Junction",
                        src0=curves[0].GetPath(),
                        src1=curves[1].GetPath(),
                        type0="point",
                        type1="point",
                        indices0=[1],
                        indices1=[0],
                    )
                    binds = reference
                else:
                    curves = [_add_cable_curve(stage, "/World/Cable/Sim", np.concatenate(reference) + translation)]
                    curves[0].CreateCurveVertexCountsAttr([3, 3])
                    self._add_graphics_mesh(stage, "/World/Cable/Skin0", skins[0])
                    self._add_graphics_mesh(stage, "/World/Cable/Skin1", skins[1])
                    binds = [np.concatenate(reference)]
                for curve, bind in zip(curves, binds, strict=True):
                    prim = curve.GetPrim()
                    prim.AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
                    prim.CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
                        ["bindPose"]
                    )
                    prim.CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
                        bind.tolist()
                    )

                builder = self._import(stage)
                physics_only = self._import(stage, load_visual_shapes=False)
                for field in ("body_q", "joint_q", "joint_target_ke", "body_mass"):
                    assert_np_equal(np.asarray(getattr(builder, field)), np.asarray(getattr(physics_only, field)))
                model = builder.finalize()
                visuals = newton.geometry.DeformableVisuals(model)
                visuals.update(model.state())
                assert_np_equal(visuals.points.numpy(), np.concatenate(skins) + translation + [3, 4, 5], tol=3.0e-6)

    def test_invalid_cable_bind_pose_skips_only_visual(self):
        from pxr import Sdf, UsdGeom

        stage = self._stage()
        UsdGeom.Xform.Define(stage, "/World/Cable").GetPrim().AddAppliedSchema("PhysicsDeformableBodyAPI")
        curve = _add_cable_curve(stage, "/World/Cable/Sim", [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        prim = curve.GetPrim()
        prim.AddAppliedSchema("PhysicsDeformablePoseAPI:bind")
        prim.CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(["bindPose"])
        prim.CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set([(0, 0, 0)])
        self._add_graphics_mesh(stage, "/World/Cable/Skin")
        with self.assertLogs("newton", level="WARNING") as logs:
            builder = self._import(stage)
        self.assertIn("point count must match", "\n".join(logs.output))
        model = builder.finalize()
        self.assertEqual(model.body_count, 2)
        self.assertEqual(model.deformable_visual_mesh_count, 0)

    def test_uvs_and_texture_colors_survive_import(self):
        """Visual import preserves UVs and the mesh reader's display-space pixels."""
        from PIL import Image
        from pxr import Sdf, UsdGeom, UsdShade

        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        gfx = self._add_graphics_mesh(stage, "/World/Tire/Skin")
        pv = UsdGeom.PrimvarsAPI(gfx.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        pv.Set([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])

        material = UsdShade.Material.Define(stage, "/World/Material")
        shader = UsdShade.Shader.Define(stage, "/World/Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        texture = UsdShade.Shader.Define(stage, "/World/Material/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(gfx.GetPrim()).Bind(material)

        with tempfile.TemporaryDirectory() as directory:
            texture_path = os.path.join(directory, "skin.png")
            Image.fromarray(np.array([[[64, 128, 255, 200]]], dtype=np.uint8)).save(texture_path)
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
            for color_space, expected in (("sRGB", [64, 128, 255, 200]), ("raw", [137, 188, 255, 200])):
                with self.subTest(color_space=color_space):
                    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
                    source_mesh = usd.get_mesh(gfx.GetPrim())
                    model = self._import(stage).finalize(device="cpu")
                    mesh = model.deformable_visual_meshes[0]
                    self.assertIsNotNone(mesh.uvs)
                    assert_np_equal(mesh.uvs.numpy(), np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32))
                    pixels = load_texture(mesh.texture)
                    np.testing.assert_array_equal(pixels, load_texture(source_mesh.texture))
                    np.testing.assert_array_equal(pixels, np.array([[expected]], dtype=np.uint8))

    def test_missing_uvs_omit_texture_not_visual_geometry(self):
        """Keep an imported visual binding when its texture has no UV coordinates."""
        from pxr import Sdf, UsdShade

        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        gfx = self._add_graphics_mesh(stage, "/World/Tire/Skin")
        material = UsdShade.Material.Define(stage, "/World/Material")
        shader = UsdShade.Shader.Define(stage, "/World/Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        texture = UsdShade.Shader.Define(stage, "/World/Material/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("skin.png"))
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(gfx.GetPrim()).Bind(material)

        with self.assertLogs("newton", level="WARNING") as logs:
            builder = self._import(stage)
        self.assertTrue(any("texture requires UVs" in entry for entry in logs.output))
        model = builder.finalize(device="cpu")
        self.assertEqual(model.deformable_visual_mesh_count, 1)
        mesh = model.deformable_visual_meshes[0]
        self.assertIsNone(mesh.texture)
        state = model.state()
        state.particle_q.assign(state.particle_q.numpy() + np.array([0, 0, 1], dtype=np.float32))
        np.testing.assert_allclose(_skin(model, mesh, state), [[0.2, 0.2, 1.2], [0.4, 0.2, 1.2], [0.2, 0.4, 1.2]])

    def test_load_visual_shapes_false_skips_visual_meshes(self):
        stage = self._stage()
        self._add_volume_body(stage, "/World/Tire")
        self._add_graphics_mesh(stage, "/World/Tire/Skin")
        builder = self._import(stage, load_visual_shapes=False)
        model = builder.finalize()
        self.assertEqual(model.deformable_visual_mesh_count, 0)


class TestDeformableVisualMeshCameraRecorder(unittest.TestCase):
    """Tests for the example's optional ffmpeg recorder."""

    def test_close_waits_for_every_recording_process(self):
        """A failed stream must not prevent the other streams from closing."""
        recorder = object.__new__(DeformableVisualMeshCameraRecorder)
        recorder._closed = False
        recorder._processes = {
            "rgb": mock.Mock(stdin=mock.Mock()),
            "depth": mock.Mock(stdin=mock.Mock()),
        }
        recorder._processes["rgb"].wait.return_value = 1
        recorder._processes["depth"].wait.return_value = 2
        processes = list(recorder._processes.values())

        with self.assertRaisesRegex(RuntimeError, "rgb.*depth"):
            recorder.close()

        for process in processes:
            process.stdin.close.assert_called_once_with()
            process.wait.assert_called_once_with()
        self.assertFalse(recorder.is_active)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestDeformableVisualMeshCameraParity(unittest.TestCase):
    """The checked-in AOUSD scene matches its procedural reference scene."""

    @staticmethod
    def _build(load_from_usd: bool):
        example = DeformableVisualMeshCameraExample.__new__(DeformableVisualMeshCameraExample)
        return example._build_model_builder(SimpleNamespace(load_from_usd=load_from_usd))

    def test_usd_asset_imports_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._build(True)

    def test_procedural_and_usd_builders_match(self):
        procedural = self._build(False)
        usd = self._build(True)

        exact_fields = (
            "body_q",
            "body_com",
            "body_inertia",
            "body_flags",
            "joint_type",
            "joint_parent",
            "joint_child",
            "joint_X_p",
            "joint_X_c",
            "joint_axis",
            "joint_q",
            "joint_qd",
            "joint_target_kd",
            "joint_target_mode",
            "tri_indices",
            "edge_indices",
            "edge_rest_angle",
            "edge_bending_properties",
            "tet_indices",
            "tet_materials",
            "shape_type",
            "shape_body",
            "shape_transform",
            "shape_scale",
            "shape_flags",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_mu",
        )
        for field in exact_fields:
            np.testing.assert_array_equal(np.asarray(getattr(procedural, field)), np.asarray(getattr(usd, field)))

        close_fields = (
            "body_mass",
            "body_inv_mass",
            "body_inv_inertia",
            "joint_target_ke",
            "particle_q",
            "particle_qd",
            "particle_mass",
            "particle_radius",
            "particle_flags",
            "tri_areas",
            "tri_materials",
            "edge_rest_length",
        )
        for field in close_fields:
            np.testing.assert_allclose(
                np.asarray(getattr(procedural, field)),
                np.asarray(getattr(usd, field)),
                rtol=1.0e-6,
                atol=1.0e-7,
            )

        self.assertEqual(len(procedural._deformable_visual_meshes), len(usd._deformable_visual_meshes))
        for procedural_visual, usd_visual in zip(
            procedural._deformable_visual_meshes, usd._deformable_visual_meshes, strict=True
        ):
            np.testing.assert_allclose(
                procedural_visual["rest_vertices"], usd_visual["rest_vertices"], rtol=1.0e-6, atol=1.0e-7
            )
            np.testing.assert_array_equal(procedural_visual["indices"], usd_visual["indices"])
            np.testing.assert_allclose(procedural_visual["uvs"], usd_visual["uvs"], rtol=1.0e-6, atol=1.0e-7)
            self.assertEqual(procedural_visual["texture"], usd_visual["texture"])

        procedural_model = procedural.finalize()
        usd_model = usd.finalize()
        final_fields = (
            "body_q",
            "body_mass",
            "body_inv_mass",
            "body_inertia",
            "body_inv_inertia",
            "joint_target_kd",
            "particle_q",
            "particle_mass",
            "particle_radius",
            "tri_areas",
            "tri_materials",
            "edge_rest_length",
            "tet_materials",
            "shape_transform",
            "shape_scale",
        )
        for field in final_fields:
            np.testing.assert_array_equal(
                getattr(procedural_model, field).numpy(),
                getattr(usd_model, field).numpy(),
            )

        np.testing.assert_allclose(
            procedural_model.joint_target_ke.numpy(),
            usd_model.joint_target_ke.numpy(),
            rtol=1.0e-6,
            atol=1.0e-7,
        )

        for procedural_visual, usd_visual in zip(
            procedural_model.deformable_visual_meshes, usd_model.deformable_visual_meshes, strict=True
        ):
            assert_np_equal(
                _skin(procedural_model, procedural_visual, procedural_model.state()),
                _skin(usd_model, usd_visual, usd_model.state()),
                tol=1.0e-6,
            )


if __name__ == "__main__":
    warnings.simplefilter("default")
    unittest.main(verbosity=2)
