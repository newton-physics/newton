# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import newton


def _camera_shapes(model: newton.Model) -> list[int]:
    return [i for i, source in enumerate(model.shape_source) if isinstance(source, newton.CameraSensor)]


class TestImportMjcfCameras(unittest.TestCase):
    def test_import_mjcf_adds_camera_sensors_as_shapes(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
