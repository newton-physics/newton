# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
import types
import unittest
from importlib.machinery import PathFinder
from pathlib import Path
from unittest.mock import Mock, patch

# ruff: noqa: PLC0415


class _DummyScene:
    def add_light_ambient(self, *_args, **_kwargs):
        return None

    def configure_environment_map(self, **_kwargs):
        return None


class _DummyServer:
    def __init__(self, *, port: int, label: str):
        self.port = port
        self.label = label
        self.scene = _DummyScene()

    def on_client_connect(self, _callback):
        return None

    def on_client_disconnect(self, _callback):
        return None

    def request_share_url(self) -> str:
        return "https://example.invalid"

    def get_scene_serializer(self):
        return None


def _find_spec_with_proxy(module_name: str, *args, **kwargs):
    if module_name == "jupyter_server_proxy":
        return object()
    return PathFinder.find_spec(module_name, *args, **kwargs)


class TestViewerViserNotebookUrls(unittest.TestCase):
    def _make_viewer(self, *, is_jupyter_notebook: bool = True, port: int = 9123):
        from newton._src.viewer.viewer_viser import ViewerViser

        mock_viser = types.SimpleNamespace(ViserServer=_DummyServer)

        with patch.object(ViewerViser, "_get_viser", return_value=mock_viser):
            with patch("newton._src.viewer.viewer_viser.is_jupyter_notebook", return_value=is_jupyter_notebook):
                return ViewerViser(port=port, verbose=False)

    def test_url_uses_jupyter_proxy_path_in_notebook(self):
        with patch("importlib.util.find_spec", side_effect=_find_spec_with_proxy):
            with patch.dict(os.environ, {"JUPYTERHUB_SERVICE_PREFIX": "/user/alice/"}, clear=False):
                viewer = self._make_viewer(port=9123)
                self.assertEqual(viewer.url, "/user/alice/proxy/9123/")

    def test_show_notebook_uses_proxy_url_for_live_server(self):
        mock_ipython = types.ModuleType("IPython")
        mock_ipython.__path__ = []

        mock_display_module = types.ModuleType("IPython.display")
        mock_display_module.HTML = Mock()
        mock_display_module.IFrame = Mock()
        mock_display_module.display = Mock()
        mock_ipython.display = mock_display_module

        with patch("importlib.util.find_spec", side_effect=_find_spec_with_proxy):
            with patch.dict(os.environ, {"JUPYTERHUB_SERVICE_PREFIX": "/user/alice/"}, clear=False):
                with patch.dict(
                    sys.modules,
                    {
                        "IPython": mock_ipython,
                        "IPython.display": mock_display_module,
                    },
                ):
                    viewer = self._make_viewer(port=9456)
                    viewer.show_notebook(width=640, height=480)

        mock_display_module.IFrame.assert_called_once_with(src="/user/alice/proxy/9456/", width=640, height=480)
        mock_display_module.display.assert_called_once_with(mock_display_module.IFrame.return_value)

    def test_get_viser_client_dir_prefers_installed_package_build(self):
        from newton._src.viewer.viewer_viser import ViewerViser

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            package_init = tmp_path / "viser" / "__init__.py"
            package_init.parent.mkdir(parents=True)
            package_init.write_text("# test viser module\n")

            package_build_dir = package_init.parent / "client" / "build"
            package_build_dir.mkdir(parents=True)
            (package_build_dir / "index.html").write_text("<!doctype html>\n")

            mock_viser = types.ModuleType("viser")
            mock_viser.__file__ = str(package_init)

            with patch.object(ViewerViser, "_get_viser", return_value=mock_viser):
                self.assertEqual(ViewerViser._get_viser_client_dir(), package_build_dir.resolve())

    def test_get_viser_client_dir_raises_without_installed_build(self):
        from newton._src.viewer.viewer_viser import ViewerViser

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            package_init = tmp_path / "viser" / "__init__.py"
            package_init.parent.mkdir(parents=True)
            package_init.write_text("# test viser module\n")

            mock_viser = types.ModuleType("viser")
            mock_viser.__file__ = str(package_init)

            with patch.object(ViewerViser, "_get_viser", return_value=mock_viser):
                with self.assertRaises(FileNotFoundError):
                    ViewerViser._get_viser_client_dir()

    def test_repo_does_not_ship_vendored_viser_client(self):
        static_index = Path(__file__).resolve().parents[1] / "_src" / "viewer" / "viser" / "static" / "index.html"
        self.assertFalse(static_index.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
