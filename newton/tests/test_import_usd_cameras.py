# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE


def _camera_shapes(model: newton.Model) -> list[int]:
    return [i for i, source in enumerate(model.shape_source) if isinstance(source, newton.SensorCamera)]


def _make_stage():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdGeom.Xform.Define(stage, "/World")

    overview = UsdGeom.Camera.Define(stage, "/World/Overview")
    overview.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    overview.GetFocalLengthAttr().Set(35.0)
    overview.GetHorizontalApertureAttr().Set(30.0)
    overview.GetVerticalApertureAttr().Set(20.0)
    overview.GetHorizontalApertureOffsetAttr().Set(1.0)
    overview.GetVerticalApertureOffsetAttr().Set(2.0)

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


def _make_render_stage():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    UsdGeom.Xform.Define(stage, "/World")

    body = UsdGeom.Xform.Define(stage, "/World/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())

    target = UsdGeom.Cube.Define(stage, "/World/Body/Target")
    target.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -2.0))
    target.AddScaleOp().Set(Gf.Vec3f(0.4, 0.4, 0.4))
    UsdPhysics.CollisionAPI.Apply(target.GetPrim())

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.GetFocalLengthAttr().Set(35.0)
    camera.GetVerticalApertureAttr().Set(20.0)

    return stage


def _make_articulated_camera_stage():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/physicsScene")

    env = UsdGeom.Xform.Define(stage, "/World/env")
    env.AddTranslateOp().Set(Gf.Vec3d(100.0, 200.0, 0.0))

    root = UsdGeom.Xform.Define(stage, "/World/env/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    base = UsdGeom.Xform.Define(stage, "/World/env/Robot/Base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).GetMassAttr().Set(1.0)

    base_collider = UsdGeom.Cube.Define(stage, "/World/env/Robot/Base/Collider")
    base_collider.GetSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(base_collider.GetPrim())

    link = UsdGeom.Xform.Define(stage, "/World/env/Robot/Link")
    link.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).GetMassAttr().Set(0.5)

    link_collider = UsdGeom.Cube.Define(stage, "/World/env/Robot/Link/Collider")
    link_collider.GetSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(link_collider.GetPrim())

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/env/Robot/Joint")
    joint.CreateBody0Rel().SetTargets(["/World/env/Robot/Base"])
    joint.CreateBody1Rel().SetTargets(["/World/env/Robot/Link"])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 1.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateAxisAttr().Set("Z")

    body_camera = UsdGeom.Camera.Define(stage, "/World/env/Robot/Link/BodyCamera")
    body_camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.5))

    static_camera = UsdGeom.Camera.Define(stage, "/World/env/Robot/StaticCamera")
    static_camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.0))

    return stage


class TestImportUsdCameras(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_adds_camera_sensors_as_shapes(self) -> None:
        """Verify USD camera prims import as shape-backed camera sensors."""
        from pxr import UsdGeom

        stage = _make_stage()
        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "orthographic USD camera"):
            result = builder.add_usd(stage)
        model = builder.finalize(device="cpu")

        self.assertIn("/World/Overview", result["path_camera_map"])
        self.assertIn("/World/Body/Camera", result["path_camera_map"])
        self.assertNotIn("/World/Ortho", result["path_camera_map"])

        overview = result["path_camera_map"]["/World/Overview"]
        body_camera = result["path_camera_map"]["/World/Body/Camera"]
        self.assertEqual(result["path_shape_map"]["/World/Overview"], overview)
        self.assertEqual(result["path_shape_map"]["/World/Body/Camera"], body_camera)

        self.assertEqual(int(model.shape_type.numpy()[overview]), int(newton.GeoType.CAMERA))
        self.assertIsInstance(model.shape_source[overview], newton.SensorCamera)
        self.assertEqual(model.shape_source[overview].width, 640)
        self.assertEqual(model.shape_source[overview].height, 480)
        self.assertEqual(int(model.shape_body.numpy()[overview]), -1)
        self.assertGreaterEqual(int(model.shape_body.numpy()[body_camera]), 0)

        np.testing.assert_allclose(model.shape_transform.numpy()[overview][:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(model.shape_transform.numpy()[body_camera][:3], [0.0, 0.0, 0.5])
        np.testing.assert_array_equal(model.shape_source[body_camera].shape_indices.numpy(), [body_camera])

        expected_rays = newton.SensorCamera.compute_camera_rays_usd_pinhole(
            model.shape_source[overview].width,
            model.shape_source[overview].height,
            UsdGeom.Camera(stage.GetPrimAtPath("/World/Overview")),
            device="cpu",
        )
        np.testing.assert_allclose(model.shape_source[overview].rays.numpy(), expected_rays.numpy(), atol=1.0e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_rebases_cameras_with_override_root_xform(self) -> None:
        """Verify override_root_xform rebases body-attached and static cameras."""
        builder = newton.ModelBuilder()
        result = builder.add_usd(
            _make_articulated_camera_stage(),
            xform=wp.transform((5.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            override_root_xform=True,
        )
        model = builder.finalize(device="cpu")

        body_camera = result["path_camera_map"]["/World/env/Robot/Link/BodyCamera"]
        static_camera = result["path_camera_map"]["/World/env/Robot/StaticCamera"]

        self.assertGreaterEqual(int(model.shape_body.numpy()[body_camera]), 0)
        self.assertEqual(int(model.shape_body.numpy()[static_camera]), -1)
        np.testing.assert_allclose(model.shape_transform.numpy()[body_camera][:3], [0.0, 0.0, 0.5], atol=1.0e-4)
        np.testing.assert_allclose(model.shape_transform.numpy()[static_camera][:3], [5.0, 0.0, 2.0], atol=1.0e-4)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_can_skip_camera_loading(self) -> None:
        """Verify USD camera imports can be disabled."""
        builder = newton.ModelBuilder()
        result = builder.add_usd(_make_stage(), load_cameras=False)
        model = builder.finalize(device="cpu")

        self.assertEqual(result["path_camera_map"], {})
        self.assertEqual(_camera_shapes(model), [])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_import_usd_camera_sensor_renders_scene(self) -> None:
        """Verify an imported USD camera sensor renders imported geometry."""
        builder = newton.ModelBuilder()
        result = builder.add_usd(_make_render_stage())
        model = builder.finalize(device="cpu")
        state = model.state()

        target_shape = result["path_shape_map"]["/World/Body/Target"]
        camera_shape = result["path_camera_map"]["/World/Camera"]
        camera_sensor = model.shape_source[camera_shape]
        self.assertIsInstance(camera_sensor, newton.SensorCamera)

        depth = camera_sensor.create_depth_image_output()
        shape_index = camera_sensor.create_shape_index_image_output()

        model.update_render_context(state)
        camera_sensor.update(model, state, depth_image=depth, shape_index_image=shape_index)

        depth_np = depth.numpy()
        shape_index_np = shape_index.numpy()
        center = (0, camera_sensor.height // 2, camera_sensor.width // 2)
        self.assertGreater(float(depth_np[center]), 0.0)
        self.assertEqual(int(shape_index_np[center]), target_shape)


if __name__ == "__main__":
    unittest.main()
