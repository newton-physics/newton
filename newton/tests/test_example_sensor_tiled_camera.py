# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import types
import unittest

from newton.examples.sensors.example_sensor_tiled_camera import Example
from newton.sensors import SensorTiledCamera


class _ViewerWithMainImage:
    def __init__(self):
        self.log_image_calls = []
        self.log_main_image_calls = []

    def log_image(self, name, image):
        self.log_image_calls.append((name, image))

    def log_main_image(self, name, image):
        self.log_main_image_calls.append((name, image))


class _ViewerWithoutMainImage:
    def __init__(self):
        self.log_image_calls = []

    def log_image(self, name, image):
        self.log_image_calls.append((name, image))


class _DummyUi:
    def __init__(self, checkbox_value: bool):
        self.checkbox_value = checkbox_value
        self.checkbox_calls = []

    def checkbox(self, label, value):
        self.checkbox_calls.append((label, value))
        return True, self.checkbox_value

    def radio_button(self, _label, _active):
        return False

    def slider_float(self, _label, value, _min_value, _max_value, _format):
        return False, value

    def slider_int(self, _label, value, _min_value, _max_value, _format):
        return False, value


def _make_example(viewer, *, sensor_color_as_main_view: bool) -> Example:
    example = Example.__new__(Example)
    example.viewer = viewer
    example.sensor_color_as_main_view = sensor_color_as_main_view
    return example


class TestSensorTiledCameraMainImageToggle(unittest.TestCase):
    def test_log_color_image_uses_main_view_when_enabled(self):
        """Verify enabled main-view mode logs the color image as the frame surface."""
        viewer = _ViewerWithMainImage()
        example = _make_example(viewer, sensor_color_as_main_view=True)
        image = object()

        self.assertTrue(example._log_color_image(image))

        self.assertEqual(viewer.log_main_image_calls, [("color", image)])
        self.assertEqual(viewer.log_image_calls, [])

    def test_log_color_image_uses_overlay_when_disabled(self):
        """Verify disabled main-view mode keeps the old overlay image path."""
        viewer = _ViewerWithMainImage()
        example = _make_example(viewer, sensor_color_as_main_view=False)
        image = object()

        self.assertFalse(example._log_color_image(image))

        self.assertEqual(viewer.log_main_image_calls, [])
        self.assertEqual(viewer.log_image_calls, [("color", image)])

    def test_log_color_image_falls_back_without_main_view_support(self):
        """Verify viewers without main-image support still use overlay logging."""
        viewer = _ViewerWithoutMainImage()
        example = _make_example(viewer, sensor_color_as_main_view=True)
        image = object()

        self.assertFalse(example._log_color_image(image))

        self.assertEqual(viewer.log_image_calls, [("color", image)])

    def test_gui_checkbox_updates_main_view_mode(self):
        """Verify the side-panel checkbox controls main-image rendering."""
        viewer = _ViewerWithMainImage()
        example = _make_example(viewer, sensor_color_as_main_view=True)
        example.tiled_camera_sensor = types.SimpleNamespace(
            default_render_config=types.SimpleNamespace(
                gaussians_mode=SensorTiledCamera.GaussianRenderMode.FAST,
                gaussians_min_transmittance=0.1,
                gaussians_max_num_hits=8,
            )
        )
        ui = _DummyUi(checkbox_value=False)

        example.gui(ui)

        self.assertEqual(ui.checkbox_calls, [("Sensor Color as Main View", True)])
        self.assertFalse(example.sensor_color_as_main_view)


if __name__ == "__main__":
    unittest.main()
