# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton
from newton.tests.unittest_utils import USD_AVAILABLE


def _camera_shapes(model: newton.Model) -> list[int]:
    return [i for i, source in enumerate(model.shape_source) if isinstance(source, newton.CameraSensor)]


def _make_stage():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdGeom.Xform.Define(stage, "/World")

    overview = UsdGeom.Camera.Define(stage, "/World/Overview")
    overview.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    overview.GetFocalLengthAttr().Set(35.0)
    overview.GetVerticalApertureAttr().Set(20.0)

    body = UsdGeom.Xform.Define(stage, "/World/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

    collider = UsdGeom.Cube.Define(stage, "/World/Body/Collider")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    body_camera = UsdGeom.Camera.Define(stage, "/World/Body/Camera")
    body_camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))
    body_camera.GetFocalLengthAttr().Set(50.0)
    body_camera.GetVerticalApertureAttr().Set(25.0)

    orthographic = UsdGeom.Camera.Define(stage, "/World/Ortho")
    orthographic.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)

    return stage


class TestImportUsdCameras(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_adds_camera_sensors_as_shapes(self) -> None:
        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "orthographic USD camera"):
            result = builder.add_usd(_make_stage())
        model = builder.finalize(device="cpu")

        self.assertIn("/World/Overview", result["path_camera_map"])
        self.assertIn("/World/Body/Camera", result["path_camera_map"])
        self.assertNotIn("/World/Ortho", result["path_camera_map"])

        overview = result["path_camera_map"]["/World/Overview"]
        body_camera = result["path_camera_map"]["/World/Body/Camera"]
        self.assertEqual(result["path_shape_map"]["/World/Overview"], overview)
        self.assertEqual(result["path_shape_map"]["/World/Body/Camera"], body_camera)

        self.assertEqual(int(model.shape_type.numpy()[overview]), int(newton.GeoType.CAMERA))
        self.assertIsInstance(model.shape_source[overview], newton.CameraSensor)
        self.assertEqual(model.shape_source[overview].width, 640)
        self.assertEqual(model.shape_source[overview].height, 480)
        self.assertEqual(int(model.shape_body.numpy()[overview]), -1)
        self.assertGreaterEqual(int(model.shape_body.numpy()[body_camera]), 0)

        np.testing.assert_allclose(model.shape_transform.numpy()[overview][:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(model.shape_transform.numpy()[body_camera][:3], [0.0, 0.0, 0.5])
        np.testing.assert_array_equal(model.shape_source[body_camera].shape_indices.numpy(), [body_camera])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_can_skip_camera_loading(self) -> None:
        builder = newton.ModelBuilder()
        result = builder.add_usd(_make_stage(), load_cameras=False)
        model = builder.finalize(device="cpu")

        self.assertEqual(result["path_camera_map"], {})
        self.assertEqual(_camera_shapes(model), [])


if __name__ == "__main__":
    unittest.main()
