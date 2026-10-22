# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip


class TestViewerGLShadows(unittest.TestCase):
    def test_small_box_shadow_reaches_sloping_receiver(self) -> None:
        """Keep a centimeter-scale shadow attached without shadowing the bare ramp."""
        viewer = _make_headless_viewer_gl_or_skip(self, width=512, height=384)
        self.addCleanup(viewer.close)
        viewer.renderer.specular_scale = 0.0
        viewer.renderer.spotlight_enabled = False
        oit_available = viewer.renderer._ensure_transparency_resources()

        for slope in (20.0, 45.0):
            with self.subTest(slope=slope):
                rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.deg2rad(slope)))

                def rotate(point: ArrayLike, rotation: wp.quat = rotation) -> wp.vec3:
                    return wp.quat_rotate(rotation, wp.vec3(point))

                builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(rotate((0.0, 0.0, -0.005)), rotation),
                    hx=50.0,
                    hy=0.6,
                    hz=0.005,
                    color=(0.2, 0.5, 0.8),
                )
                # The box's top projects 30.5 mm down-light along the ramp.
                # Its shadow must reach the receiver immediately beside the box.
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(rotate((0.0, 0.0, 0.0155)), rotation),
                    hx=0.02,
                    hy=0.02,
                    hz=0.015,
                    color=(0.9, 0.2, 0.1),
                )
                model = builder.finalize(device=viewer.device)
                viewer.set_model(model)
                viewer.renderer._sun_direction = np.array(rotate((0.0, -1.0, 1.0))) / np.sqrt(2.0)
                viewer.set_camera(pos=rotate((0.0, 0.16, 0.35)), pitch=0.0, yaw=0.0)
                viewer.camera.near = 0.005
                viewer.camera.look_at((0.0, 0.0, 0.0))

                def sample_patch(frame: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> np.ndarray:
                    points = np.array(
                        [
                            (*rotate((x, y, 0.0)), 1.0)
                            for x in np.linspace(xmin, xmax, 9)
                            for y in np.linspace(ymin, ymax, 9)
                        ]
                    )
                    view = viewer.camera.get_view_matrix().reshape(4, 4).T
                    projection = viewer.camera.get_projection_matrix().reshape(4, 4).T
                    clip = points @ (projection @ view).T
                    ndc = clip[:, :2] / clip[:, 3:4]
                    pixels = np.floor((ndc * (0.5, -0.5) + 0.5) * (512, 384)).astype(int)
                    return frame[pixels[:, 1], pixels[:, 0]]

                renderer = viewer.renderer
                renderer._xray_collision_opacity = 0.118
                renderer._xray_opacity = 0.206
                renderer._xray_proximity_range = 3.0
                renderer._xray_inspection_radius = 1.0
                renderer._xray_adaptive = True
                renderer._xray_edges = renderer._xray_collision_edges = False
                for role, fallback in (
                    (None, False),
                    ("collision", False),
                    ("visual", False),
                    ("collision", True),
                    ("visual", True),
                ):
                    with self.subTest(role=role, fallback=fallback):
                        renderer._physics_xray = role is not None
                        renderer._oit_supported = oit_available and not fallback
                        renderer._oit_fallback_warned = True
                        frames = []
                        for shadows in (False, True):
                            renderer.draw_shadows = shadows
                            viewer.begin_frame(0.0)
                            viewer.log_state(model.state())
                            for obj in viewer.objects.values():
                                obj._physics_role = role
                                obj._physics_static = True
                            viewer.end_frame()
                            frames.append(viewer.get_frame().numpy().astype(np.float32))

                        near_contact = [sample_patch(frame, -0.005, 0.005, 0.024, 0.032) for frame in frames]
                        self.assertTrue(np.all(near_contact[0][:, 2] > near_contact[0][:, 0]), "patch misses the ramp")
                        self.assertGreater(
                            float(np.mean(near_contact[0] - near_contact[1])),
                            6.0 if role else 20.0,
                            "the contact shadow is detached or lost when its receiver fades",
                        )
                        if role is not None:
                            self.assertLess(
                                float(np.mean(near_contact[0] - near_contact[1])),
                                18.0,
                                "X-ray contact shadows are too dark for the inspection view",
                            )
                            attached = [sample_patch(frame, -0.005, 0.005, 0.0208, 0.0225) for frame in frames]
                            self.assertGreater(
                                float(np.mean(attached[0] - attached[1])),
                                6.0,
                                "the close inspection view loses the shadow within 1 mm of the contact edge",
                            )
                        unoccluded = [sample_patch(frame, 0.07, 0.10, 0.04, 0.08) for frame in frames]
                        self.assertLess(
                            float(np.mean(np.abs(unoccluded[0] - unoccluded[1]))),
                            2.0,
                            "the unobstructed receiver incorrectly shadows itself",
                        )
                        if role is not None and slope == 20.0:
                            # Robot links often have both a visual skin and a
                            # collision proxy. Their shadows must not gain the
                            # extra opacity used to reveal contacts on terrain.
                            viewer.begin_frame(0.0)
                            viewer.log_state(model.state())
                            for obj in viewer.objects.values():
                                obj._physics_static = False
                            viewer.end_frame()
                            moving = sample_patch(
                                viewer.get_frame().numpy().astype(np.float32), -0.005, 0.005, 0.024, 0.032
                            )
                            self.assertLess(
                                float(np.mean(near_contact[0] - moving)),
                                0.4 * float(np.mean(near_contact[0] - near_contact[1])),
                                "shadow coverage makes moving inspection surfaces too opaque",
                            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
