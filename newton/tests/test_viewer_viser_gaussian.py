# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

import newton
from newton._src.viewer.viewer_viser import ViewerViser
from newton.tests.unittest_utils import assert_np_equal

_SH_C0 = 0.28209479177387814


def _make_test_gaussian(n: int = 4, seed: int = 0) -> newton.Gaussian:
    """Build a small synthetic Gaussian asset with identity rotations."""
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(n, 3)).astype(np.float32)
    rotations = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
    scales = rng.uniform(0.1, 0.5, size=(n, 3)).astype(np.float32)
    opacities = np.full(n, 0.7, dtype=np.float32)
    rgb = np.array([0.9, 0.3, 0.2], dtype=np.float32)
    sh_coeffs = np.tile(((rgb - 0.5) / _SH_C0).astype(np.float32), (n, 1))
    return newton.Gaussian(positions, rotations, scales, opacities, sh_coeffs)


class TestViewerViserGaussian(unittest.TestCase):
    """Regression tests for ``ViewerViser.log_gaussian`` (issue #2099)."""

    def _make_viser_viewer(self):
        captured = {"add_calls": 0}

        def add_gaussian_splats(name, centers, covariances, rgbs, opacities, **kwargs):
            captured["add_calls"] += 1
            captured["last"] = {
                "name": name,
                "centers": centers,
                "covariances": covariances,
                "rgbs": rgbs,
                "opacities": opacities,
            }
            return Mock()

        scene = Mock()
        scene.add_gaussian_splats = add_gaussian_splats
        scene.add_light_ambient = Mock()
        scene.configure_environment_map = Mock()

        server = Mock()
        server.scene = scene
        server.on_client_connect = Mock()
        server.on_client_disconnect = Mock()
        server.get_scene_serializer = Mock(return_value=None)
        server.stop = Mock()

        fake_viser = Mock()
        fake_viser.ViserServer = Mock(return_value=server)

        patches = [
            patch.object(ViewerViser, "_get_viser", return_value=fake_viser),
            patch("newton._src.viewer.viewer_viser.is_jupyter_notebook", return_value=False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        viewer = ViewerViser(verbose=False)
        self.addCleanup(viewer.close)
        return viewer, captured

    def test_log_gaussian_computes_covariance_and_color(self):
        """The uploaded covariance/color/opacity data should match the Gaussian's local-space parameters."""
        viewer, captured = self._make_viser_viewer()
        gaussian = _make_test_gaussian()

        viewer.log_gaussian("/probe", gaussian, xform=wp.transformf(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))

        last = captured["last"]
        n = gaussian.count
        self.assertEqual(last["centers"].shape, (n, 3))
        self.assertEqual(last["covariances"].shape, (n, 3, 3))
        self.assertEqual(last["opacities"].shape, (n, 1))

        # Identity rotations: covariance should reduce to diag(scale^2).
        expected_cov = np.stack([np.diag(s * s) for s in gaussian.scales]).astype(np.float32)
        assert_np_equal(last["covariances"], expected_cov, tol=1e-5)

        assert_np_equal(last["opacities"], gaussian.opacities.reshape(-1, 1), tol=1e-6)

        # rgb = 0.5 + SH_C0 * sh_dc, inverted from the (0.9, 0.3, 0.2) color baked into the fixture.
        expected_rgb = np.tile(np.array([0.9, 0.3, 0.2], dtype=np.float32), (n, 1))
        assert_np_equal(last["rgbs"], expected_rgb, tol=1e-5)

    def test_log_gaussian_uploads_once_and_updates_pose_in_place(self):
        """A repeated call with the same Gaussian asset should not re-upload the point cloud."""
        viewer, captured = self._make_viser_viewer()
        gaussian = _make_test_gaussian()

        viewer.log_gaussian("/probe", gaussian, xform=wp.transformf(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity()))
        self.assertEqual(captured["add_calls"], 1)
        handle = viewer._scene_handles["/probe"]

        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 1.0)
        viewer.log_gaussian("/probe", gaussian, xform=wp.transformf(wp.vec3(4.0, 5.0, 6.0), rot))

        # Same asset (same Gaussian object) -> no second upload, same handle reused.
        self.assertEqual(captured["add_calls"], 1)
        self.assertIs(viewer._scene_handles["/probe"], handle)
        assert_np_equal(np.asarray(handle.position), np.array([4.0, 5.0, 6.0], dtype=np.float32))

    def test_log_gaussian_hidden_sets_visible_false(self):
        """Logging with hidden=True should hide the existing handle without removing it."""
        viewer, captured = self._make_viser_viewer()
        gaussian = _make_test_gaussian()

        viewer.log_gaussian("/probe", gaussian, xform=wp.transformf(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        handle = viewer._scene_handles["/probe"]

        viewer.log_gaussian("/probe", gaussian, hidden=True)

        self.assertFalse(handle.visible)
        self.assertEqual(captured["add_calls"], 1)

    def test_clear_model_drops_gaussian_cache(self):
        """clear_model() must release the Gaussian cache like it does for meshes/instances.

        Regression test: without popping ``_gaussian_splats`` alongside
        ``_scene_handles`` in ``clear_model()``, a stale cache entry could
        outlive its handle and cause a ``KeyError`` on the next
        ``log_gaussian`` call for the same name after a model reset.
        """
        viewer, _captured = self._make_viser_viewer()
        gaussian = _make_test_gaussian()

        viewer.log_gaussian("/probe", gaussian, xform=wp.transformf(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        self.assertIn("/probe", viewer._gaussian_splats)
        self.assertIn("/probe", viewer._scene_handles)

        viewer.clear_model()

        self.assertNotIn("/probe", viewer._gaussian_splats)
        self.assertNotIn("/probe", viewer._scene_handles)


if __name__ == "__main__":
    unittest.main(verbosity=2)
