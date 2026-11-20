# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Author, simulate, and reopen native deformables in the scene editor."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton import ModelBuilder
from newton.tests.unittest_utils import get_cuda_test_devices, requires
from newton.viewer import SceneEditor, ViewerNull


@requires("pxr.Usd", "newton_usd_schemas")
class TestSceneAuthoringDeformables(unittest.TestCase):
    def make_editor(self, solver, device="cpu"):
        editor = SceneEditor(ViewerNull(), solver=solver, device=device)
        self.addCleanup(editor.close)
        return editor

    def test_vbd_round_trip(self):
        from pxr import Usd, UsdGeom, UsdPhysics

        editor = self.make_editor("vbd")
        cloth = editor.add_deformable("cloth", position=(-2, 0, 1.2), resolution=2)
        editor.add_deformable("cable", position=(0, 0, 1.2), resolution=3)
        editor.add_deformable("soft", position=(2, 0, 1.2), size=(0.4, 0.4, 0.4), resolution=2)
        editor.commit()
        self.assertIsNone(editor.solver, "Browsing should defer VBD setup")
        self.assertEqual((editor.model.tri_count, editor.model.tet_count, editor.model.body_count), (56, 48, 3))
        original = editor.model.particle_q.numpy().copy()
        editor.step()
        live = editor.state_0.particle_q.numpy().copy()
        self.assertTrue(np.all(live[:, 2] < original[:, 2]))
        self.assertTrue(np.isfinite(editor.state_0.body_q.numpy()).all())

        editor.begin_edit()
        editor.update_object(cloth, density=600.0, stiffness=25_000.0)
        editor.update_object(cloth, size=(1.25, 0.8, 1.0), thickness=0.01)
        editor.end_edit()
        editor.commit()
        # The geometry change reseeds the edited cloth while its neighbors keep moving state.
        np.testing.assert_array_equal(editor.state_0.particle_q.numpy()[9:], live[9:])
        self.assertAlmostEqual(float(editor.model.particle_mass.numpy()[:9].sum()), 6.0, places=4)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.usda"
            editor.save_usd(path)
            stage = Usd.Stage.Open(str(path))
            geometry = UsdGeom.Mesh.Get(stage, "/World/Object_0/Geometry")
            points = np.asarray(geometry.GetPointsAttr().Get())
            np.testing.assert_allclose(np.ptp(points, axis=0), (1.25, 0.8, 0.0))
            self.assertAlmostEqual(geometry.GetPrim().GetAttribute("physics:thicknesses").Get()[0], 0.01)
            material = stage.GetPrimAtPath("/World/Object_0/Material")
            self.assertEqual(UsdPhysics.MaterialAPI(material).GetDensityAttr().Get(), 600.0)
            self.assertEqual(material.GetAttribute("physics:youngsModulus").Get(), 25_000.0)
            self.assertTrue(UsdGeom.BasisCurves.Get(stage, "/World/Object_1/Geometry"))
            self.assertTrue(UsdGeom.TetMesh.Get(stage, "/World/Object_2/Geometry"))

            native = ModelBuilder()
            native.add_usd(stage)
            self.assertEqual(native.tri_count, editor.model.tri_count)
            self.assertEqual(native.tet_count, editor.model.tet_count)
            np.testing.assert_allclose(native.particle_mass, editor.model.particle_mass.numpy())

            reopened = self.make_editor(SceneEditor.saved_solver(path))
            reopened.load_usd(path)
            reopened.commit()
            self.assertEqual(reopened.get_object(cloth), editor.get_object(cloth))
            np.testing.assert_array_equal(reopened.state_0.particle_q.numpy(), editor.model.particle_q.numpy())
            editor.reset_scene()
            np.testing.assert_array_equal(editor.state_0.particle_q.numpy(), reopened.state_0.particle_q.numpy())

    def test_place_referenced_cloth(self):
        from pxr import Sdf, Usd

        editor = self.make_editor("vbd")
        editor.add_primitive("box", position=(0.1, 1.1, 0.2), size=(0.5, 0.5, 0.4), dynamic=False)
        editor.commit()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "cloth.usda"
            fixture = Sdf.Layer.FindOrOpen(str(Path(__file__).parent / "assets/deformables_mixed.usda"))
            layer = Sdf.Layer.CreateNew(str(source))
            layer.TransferContent(fixture)
            stage = Usd.Stage.Open(layer)
            for prim in ("CableA", "CableB", "SoftA", "SoftB"):
                stage.RemovePrim(f"/World/{prim}")
            layer.Save()

            cloth = editor.add_usd(source, position=(0, 0, -1))
            editor.place_object(cloth)
            editor.commit()
            self.assertEqual(editor.model.particle_count, 4)
            positions = editor.state_0.particle_q.numpy().copy()
            self.assertAlmostEqual(
                float((positions[:, 2] - editor.model.particle_radius.numpy()).min()), 0.401, places=6
            )
            path = directory / "scene.usda"
            editor.save_usd(path)
            native = ModelBuilder()
            native.add_usd(str(path))
            np.testing.assert_allclose(native.particle_q, positions, atol=1.0e-6)

    def test_mpm_particle_strokes(self):
        from pxr import Usd, UsdGeom, UsdPhysics

        devices = get_cuda_test_devices()
        if not devices:
            self.skipTest("MPM requires CUDA")
        editor = self.make_editor("mpm", devices[0])
        editor.add_primitive("box", dynamic=False, position=(0, 0, 0.05), size=(0.8, 0.8, 0.1))
        editor.commit()
        editor.step()
        self.assertIsNone(editor.solver)
        points = tuple((x * 0.04, y * 0.04, z * 0.04) for x in range(3) for y in range(3) for z in range(3))
        first = editor.add_deformable("particles", position=(0, 0, 0.25), points=points, thickness=0.04)
        editor.commit()
        self.assertIsNone(editor.solver, "Browsing should defer MPM setup")
        editor.step()
        self.assertIsNotNone(editor.solver)
        live = editor.state_0.particle_q.numpy().copy()
        self.assertTrue(np.isfinite(live).all())
        self.assertLess(live[:, 2].mean(), editor.model.particle_q.numpy()[:, 2].mean())
        editor.add_deformable("particles", position=(0.25, 0, 0.25), points=points, thickness=0.04)
        editor.commit()
        np.testing.assert_array_equal(editor.state_0.particle_q.numpy()[: len(points)], live)
        editor.update_object(first, density=1300.0, stiffness=15_000.0, friction=0.4)
        editor.commit()
        np.testing.assert_allclose(editor.model.particle_mass.numpy()[: len(points)], 1300.0 * 0.04**3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strokes.usda"
            editor.save_usd(path)
            stage = Usd.Stage.Open(str(path))
            geometry = UsdGeom.Points.Get(stage, "/World/Object_1/Geometry")
            np.testing.assert_allclose(geometry.GetPointsAttr().Get(), points)
            material = stage.GetPrimAtPath("/World/Object_1/Material")
            self.assertEqual(UsdPhysics.MaterialAPI(material).GetDensityAttr().Get(), 1300.0)
            self.assertEqual(material.GetAttribute("newton:mpm:youngsModulus").Get(), 15_000.0)
            self.assertAlmostEqual(material.GetAttribute("newton:mpm:internalFriction").Get(), 0.4)
            self.assertGreater(stage.GetPrimAtPath("/World/PhysicsScene").GetAttribute("newton:mpm:voxelSize").Get(), 0)
            reopened = self.make_editor(SceneEditor.saved_solver(path), devices[0])
            reopened.load_usd(path)
            reopened.commit()
            np.testing.assert_allclose(reopened.get_object(first).points, points)
            np.testing.assert_array_equal(reopened.model.particle_q.numpy(), editor.model.particle_q.numpy())
