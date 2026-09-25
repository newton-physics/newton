# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch

from newton.viewer import ViewerViser, ViewerViserExtensionContext


class TestViewerViserExtensionContext(unittest.TestCase):
    def test_extension_context_exposes_public_viser_handles(self):
        """Expose stable scene, GUI, and initial-camera handles without the server."""
        server = Mock()
        server.scene = Mock()
        server.gui = Mock()
        server.initial_camera = Mock()
        server.get_scene_serializer.return_value = None

        fake_viser = Mock()
        fake_viser.ViserServer.return_value = server

        with (
            patch.object(ViewerViser, "_get_viser", return_value=fake_viser),
            patch("newton._src.viewer.viewer_viser.is_jupyter_notebook", return_value=False),
        ):
            viewer = ViewerViser(verbose=False)

        self.addCleanup(viewer.close)
        context = viewer.extension_context

        self.assertIsInstance(context, ViewerViserExtensionContext)
        self.assertIs(context.scene, server.scene)
        self.assertIs(context.gui, server.gui)
        self.assertIs(context.initial_camera, server.initial_camera)
        self.assertIs(viewer.extension_context, context)
        self.assertFalse(hasattr(context, "server"))

        context.scene.add_frame("/editor/origin")
        context.gui.add_button("Apply")
        context.initial_camera.position = (3.0, 3.0, 2.0)

        server.scene.add_frame.assert_called_once_with("/editor/origin")
        server.gui.add_button.assert_called_once_with("Apply")
        self.assertEqual(server.initial_camera.position, (3.0, 3.0, 2.0))


if __name__ == "__main__":
    unittest.main()
