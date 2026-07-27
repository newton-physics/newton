# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
import warnings

import numpy as np

import newton


def _camera_shapes(model: newton.Model) -> list[int]:
    return [i for i, source in enumerate(model.shape_source) if isinstance(source, newton.SensorCamera)]


class TestImportMjcfCameras(unittest.TestCase):
    def test_import_mjcf_adds_camera_sensors_as_shapes(self) -> None:
        """Verify MJCF camera elements import as shape-backed camera sensors."""
        mjcf = """
<mujoco model="camera_import">
    <default>
        <camera fovy="60" resolution="320 240"/>
        <default class="small_camera">
            <camera fovy="30" resolution="16 8"/>
        </default>
    </default>
    <worldbody>
        <camera name="overview" pos="1 2 3"/>
        <frame pos="0 0 1">
            <camera name="frame_cam" pos="0 0 0.5"/>
        </frame>
        <body name="base" pos="0 0 0.2">
            <camera name="body_cam" class="small_camera" pos="0 0 0.3"/>
        </body>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize(device="cpu")

        cameras = _camera_shapes(model)
        self.assertEqual(len(cameras), 3)
        cameras_by_label = {model.shape_label[i]: i for i in cameras}

        overview = cameras_by_label["camera_import/worldbody/overview"]
        frame_cam = cameras_by_label["camera_import/worldbody/frame_cam"]
        body_cam = cameras_by_label["camera_import/worldbody/base/body_cam"]

        self.assertEqual(int(model.shape_type.numpy()[overview]), int(newton.GeoType.CAMERA))
        self.assertEqual(int(model.shape_body.numpy()[overview]), -1)
        self.assertEqual(int(model.shape_body.numpy()[frame_cam]), -1)
        self.assertGreaterEqual(int(model.shape_body.numpy()[body_cam]), 0)

        np.testing.assert_allclose(model.shape_transform.numpy()[overview][:3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(model.shape_transform.numpy()[frame_cam][:3], [0.0, 0.0, 1.5])
        np.testing.assert_allclose(model.shape_transform.numpy()[body_cam][:3], [0.0, 0.0, 0.3])

        self.assertEqual(model.shape_source[overview].width, 320)
        self.assertEqual(model.shape_source[overview].height, 240)
        self.assertEqual(model.shape_source[body_cam].width, 16)
        self.assertEqual(model.shape_source[body_cam].height, 8)
        np.testing.assert_array_equal(model.shape_source[body_cam].shape_indices.numpy(), [body_cam])

    def test_import_mjcf_warns_for_ignored_camera_mode_and_projection(self) -> None:
        """Verify MJCF camera import warns when authored camera attributes are ignored."""
        mjcf = """
<mujoco model="camera_warnings">
    <worldbody>
        <camera name="tracking" mode="track" resolution="8 6"/>
        <camera name="ortho" orthographic="true" resolution="8 6"/>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            builder.add_mjcf(mjcf)

        messages = [str(warning.message) for warning in caught_warnings]
        self.assertTrue(
            any("tracking" in message and "authored camera mode is ignored" in message for message in messages)
        )
        self.assertTrue(
            any("ortho" in message and "authored camera projection is ignored" in message for message in messages)
        )

        model = builder.finalize(device="cpu")
        self.assertEqual(len(_camera_shapes(model)), 2)

    def test_import_mjcf_camera_uses_focalpixel_intrinsics(self) -> None:
        """Verify MJCF camera focalpixel intrinsics set the imported FOV."""
        mjcf = """
<mujoco model="camera_focalpixel">
    <worldbody>
        <camera name="cam" fovy="10" focalpixel="4 1" sensorsize="4 2" resolution="4 2"/>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize(device="cpu")

        camera_shape = model.shape_label.index("camera_focalpixel/worldbody/cam")
        camera_sensor = model.shape_source[camera_shape]
        expected_rays = newton.SensorCamera.compute_camera_rays_pinhole(4, 2, math.radians(90.0), device="cpu")

        np.testing.assert_allclose(camera_sensor.rays.numpy(), expected_rays.numpy(), rtol=1.0e-6, atol=1.0e-6)

    def test_import_mjcf_camera_rejects_unsupported_intrinsics(self) -> None:
        """Verify unsupported MJCF camera intrinsic combinations fail explicitly."""
        cases = (
            (
                '<camera name="cam" focalpixel="4 1" sensorsize="4 2"/>',
                "focalpixel requires sensorsize and resolution",
            ),
            (
                '<camera name="cam" focalpixel="4" sensorsize="4 2" resolution="4 2"/>',
                "focalpixel and sensorsize attributes must each have two values",
            ),
            (
                '<camera name="cam" fovy="60" sensorsize="4 2" resolution="4 2"/>',
                "sensorsize requires focal or focalpixel",
            ),
        )
        for camera_xml, message in cases:
            with self.subTest(camera_xml=camera_xml):
                mjcf = f"""
<mujoco model="camera_intrinsics">
    <worldbody>
        {camera_xml}
    </worldbody>
</mujoco>
"""
                builder = newton.ModelBuilder()
                with self.assertRaisesRegex(ValueError, message):
                    builder.add_mjcf(mjcf)

    def test_import_mjcf_camera_sensor_renders_scene(self) -> None:
        """Verify an imported MJCF camera sensor renders imported geometry."""
        mjcf = """
<mujoco model="render_camera">
    <worldbody>
        <geom name="target" type="sphere" size="0.2" pos="0 0 -2"/>
        <camera name="cam" pos="0 0 0" fovy="45" resolution="16 12"/>
    </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize(device="cpu")
        state = model.state()

        target_shape = model.shape_label.index("render_camera/worldbody/target")
        camera_shape = model.shape_label.index("render_camera/worldbody/cam")
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
