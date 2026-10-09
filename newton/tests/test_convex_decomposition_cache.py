# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise persistent convex decomposition through the public builder API."""

import importlib.metadata
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import newton
from newton._src.geometry import _convex_cache
from newton.tests.unittest_utils import patch_sys_module


class TestConvexDecompositionCache(unittest.TestCase):
    def setUp(self):
        """Create an isolated cache and deterministic decomposition backends."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.cache_dir = Path(self.tmp.name) / "cache"
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        self.parts = [(self.vertices.copy(), self.faces.copy()), (self.vertices + 2, self.faces.copy())]
        self.coacd = mock.Mock(side_effect=lambda *_args, **_kwargs: self.parts)
        self.vhacd = mock.Mock(
            side_effect=lambda *_args, **_kwargs: [{"vertices": v, "faces": f} for v, f in self.parts]
        )
        self.stack.enter_context(
            patch_sys_module("coacd", SimpleNamespace(Mesh=lambda *args: args, run_coacd=self.coacd))
        )
        self.stack.enter_context(
            patch_sys_module(
                "trimesh",
                SimpleNamespace(
                    Trimesh=lambda *args: args, decomposition=SimpleNamespace(convex_decomposition=self.vhacd)
                ),
            )
        )
        self.version = self.stack.enter_context(mock.patch("importlib.metadata.version", return_value="1.0"))

    def _build(
        self,
        *,
        method="coacd",
        vertices=None,
        faces=None,
        maxhullvert=64,
        scale=(1, 1, 1),
        coacd_threshold=0.05,
        raise_on_failure=True,
        **kwargs,
    ):
        """Approximate a fresh mesh in a fresh builder."""
        mesh = newton.Mesh(
            self.vertices if vertices is None else vertices,
            self.faces if faces is None else faces,
            maxhullvert=maxhullvert,
        )
        builder = newton.ModelBuilder()
        builder.default_mesh_approximation_cfg.coacd_threshold = coacd_threshold
        shape = builder.add_shape_mesh(-1, mesh=mesh, scale=scale)
        builder.approximate_meshes(method=method, shape_indices=[shape], raise_on_failure=raise_on_failure, **kwargs)
        return builder

    def test_cache_reuses_parts_across_builders(self):
        """Reuse ordered parts without calling either backend on a cache hit."""
        for method, backend in (("coacd", self.coacd), ("vhacd", self.vhacd)):
            with self.subTest(method=method):
                first = self._build(method=method, cache_dir=self.cache_dir)
                second = self._build(method=method, cache_dir=str(self.cache_dir), scale=(2, 3, 4))
                self.assertEqual(backend.call_count, 1)
                self.assertNotIn("cache_dir", backend.call_args.kwargs)
                self.assertEqual(first.shape_count, 2)
                self.assertEqual(second.shape_type, first.shape_type)
                for a, b in zip(first.shape_source, second.shape_source, strict=True):
                    np.testing.assert_array_equal(a.vertices, b.vertices)
                    np.testing.assert_array_equal(a.indices, b.indices)
                for scale in second.shape_scale:
                    np.testing.assert_array_equal(scale, (2, 3, 4))
        self.assertEqual(len(list(self.cache_dir.glob("*.npz"))), 2)

    def test_cache_is_opt_in(self):
        """Run the backend on each call when caching is disabled."""
        self._build()
        self._build(cache_dir=None)
        self.assertEqual(self.coacd.call_count, 2)
        self.version.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_cache_invalidates_geometry_settings_and_versions(self):
        """Recompute when any geometry, effective option, or backend version changes."""
        for method, backend, option in (
            ("coacd", self.coacd, "threshold"),
            ("vhacd", self.vhacd, "resolution"),
        ):
            with self.subTest(method=method):
                self.version.return_value = "1.0"
                self._build(method=method, cache_dir=self.cache_dir)
                self._build(method=method, cache_dir=self.cache_dir, vertices=self.vertices * 2)
                self._build(method=method, cache_dir=self.cache_dir, faces=self.faces[:, ::-1])
                self._build(method=method, cache_dir=self.cache_dir, maxhullvert=32)
                self._build(method=method, cache_dir=self.cache_dir, **{option: 0.1})
                self.version.return_value = "2.0"
                self._build(method=method, cache_dir=self.cache_dir)
                self.assertEqual(backend.call_count, 6)

    def test_cache_canonicalizes_settings(self):
        """Reuse entries for reordered options and explicitly supplied Newton defaults."""
        self._build(cache_dir=self.cache_dir, seed=7, merge=False)
        self._build(cache_dir=self.cache_dir, merge=False, seed=7)
        self._build(cache_dir=self.cache_dir, seed=7)
        self._build(cache_dir=self.cache_dir, seed=7, coacd_threshold=0.5, threshold=0.05)
        self.assertEqual(self.coacd.call_count, 1)

    def test_cache_recovers_from_corrupt_files(self):
        """Recompute and replace corrupt cache entries."""
        self._build(cache_dir=self.cache_dir)
        cache_file = next(self.cache_dir.glob("*.npz"))
        cache_file.write_bytes(b"not an npz archive")
        self._build(cache_dir=self.cache_dir)
        self._build(cache_dir=self.cache_dir)
        self.assertEqual(self.coacd.call_count, 2)

    def test_cache_write_failure_keeps_decomposition(self):
        """Use successfully computed parts when the cache directory cannot be created."""
        self.cache_dir.write_text("a file blocks directory creation")
        builder = self._build(cache_dir=self.cache_dir)
        self.assertEqual(builder.shape_count, 2)
        self.assertEqual(builder.shape_type, [newton.GeoType.CONVEX_MESH] * 2)

    def test_cache_does_not_store_failures(self):
        """Leave no entry after empty or failed backend results."""
        for result in ([], RuntimeError("decomposition failed")):
            with self.subTest(result=result):
                self.coacd.side_effect = (
                    result if isinstance(result, Exception) else lambda *_a, result=result, **_kw: result
                )
                with self.assertRaises(RuntimeError):
                    self._build(cache_dir=self.cache_dir)
                self.assertEqual(list(self.cache_dir.glob("*.npz")), [])

    def test_cache_does_not_store_fallback_hulls(self):
        """Retry decomposition after an earlier call fell back to a convex hull."""
        self.coacd.side_effect = RuntimeError("decomposition failed")
        with self.assertWarnsRegex(UserWarning, "Falling back to convex_hull"):
            builder = self._build(cache_dir=self.cache_dir, raise_on_failure=False)
        self.assertEqual(builder.shape_count, 1)
        self.assertEqual(list(self.cache_dir.glob("*.npz")), [])
        self.coacd.side_effect = lambda *_a, **_kw: self.parts
        self.assertEqual(self._build(cache_dir=self.cache_dir).shape_count, 2)

    def test_cache_rejects_invalid_archives(self):
        """Recompute for stale versions, incomplete parts, invalid indices, and object arrays."""
        self._build(cache_dir=self.cache_dir)
        cache_file = next(self.cache_dir.glob("*.npz"))
        with np.load(cache_file, allow_pickle=False) as data:
            valid = dict(data)
        for changes in (
            {"version": np.asarray(-1, dtype=np.int64)},
            {"part_count": np.asarray(3, dtype=np.int64)},
            {"faces_0": self.faces + 10},
            {"vertices_0": self.vertices.astype(object)},
            {"vertices_0": self.vertices * np.nan},
        ):
            with self.subTest(changes=list(changes)):
                np.savez(cache_file, **(valid | changes))
                calls = self.coacd.call_count
                self._build(cache_dir=self.cache_dir)
                self.assertEqual(self.coacd.call_count, calls + 1)
                self._build(cache_dir=self.cache_dir)
                self.assertEqual(self.coacd.call_count, calls + 1)
        with cache_file.open("wb") as stream:
            np.save(stream, self.vertices)
        self._build(cache_dir=self.cache_dir)

    def test_cache_skips_unkeyable_settings_or_unknown_versions(self):
        """Continue decomposition when settings or backend versions cannot be keyed."""
        self._build(cache_dir=self.cache_dir, custom_option=object())
        self.version.side_effect = importlib.metadata.PackageNotFoundError("coacd")
        self._build(cache_dir=self.cache_dir)
        self.assertEqual(self.coacd.call_count, 2)
        self.assertEqual(list(self.cache_dir.glob("*.npz")), [])

    def test_cache_invalidates_builder_defaults(self):
        """Include the builder's configured CoACD threshold in the cache key."""
        self._build(cache_dir=self.cache_dir)
        self._build(cache_dir=self.cache_dir, coacd_threshold=0.5)
        self.assertEqual(self.coacd.call_count, 2)
        self.assertEqual(self.coacd.call_args.kwargs["threshold"], 0.5)

    def test_decomposition_reuse_respects_hull_vertex_limit(self):
        """Decompose identical meshes with different hull limits separately in one call."""
        builder = newton.ModelBuilder()
        for limit in (32, 64):
            mesh = newton.Mesh(self.vertices, self.faces, maxhullvert=limit)
            builder.add_shape_mesh(-1, mesh=mesh)
        builder.approximate_meshes(method="coacd", cache_dir=self.cache_dir, raise_on_failure=True)
        self.assertEqual(self.coacd.call_count, 2)
        self.assertEqual([call.kwargs["max_convex_hull"] for call in self.coacd.call_args_list], [32, 64])

    def test_cache_hit_preserves_shape_settings_and_filters(self):
        """Apply current shape settings and collision filters to every cached convex part."""
        for mu in (0.2, 0.8):
            builder = newton.ModelBuilder()
            mesh = newton.Mesh(self.vertices, self.faces, maxhullvert=64)
            shapes = []
            for i in range(2):
                body = builder.add_body()
                shape = builder.add_shape_mesh(
                    body,
                    mesh=mesh,
                    label=f"source_{i}",
                    scale=(2, 3, 4),
                    color=(mu, 0.5, 0.5),
                    cfg=newton.ModelBuilder.ShapeConfig(mu=mu, ke=123.0, collision_group=7),
                )
                shapes.append(shape)
            builder.add_shape_collision_filter_pair(*shapes)
            masses = builder.body_mass.copy()
            builder.approximate_meshes(
                method="coacd", cache_dir=self.cache_dir, keep_visual_shapes=True, raise_on_failure=True
            )
            np.testing.assert_array_equal(builder.body_mass, masses)
            pairs = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs}
            parts = []
            for shape in shapes:
                extra = builder.shape_label.index(f"source_{shape}_convex_1")
                parts.append((shape, extra))
                for part in (shape, extra):
                    self.assertEqual(builder.shape_material_mu[part], mu)
                    self.assertEqual(builder.shape_material_ke[part], 123.0)
                    self.assertEqual(builder.shape_collision_group[part], 7)
                    np.testing.assert_array_equal(builder.shape_scale[part], (2, 3, 4))
                    np.testing.assert_array_equal(builder.shape_color[part], (mu, 0.5, 0.5))
                    self.assertFalse(builder.shape_flags[part] & newton.ShapeFlags.VISIBLE)
            for a in parts[0]:
                for b in parts[1]:
                    self.assertIn(tuple(sorted((a, b))), pairs)
        self.assertEqual(self.coacd.call_count, 1)

    def test_cache_concurrent_writes_and_failed_publish(self):
        """Publish complete entries under concurrent writes and clean up after write failures."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_convex_cache.write, self.cache_dir, "shared", self.parts) for _ in range(8)]
            for future in futures:
                future.result()
        loaded = _convex_cache.try_load(self.cache_dir, "shared")
        self.assertEqual(len(loaded), 2)
        for expected, actual in zip(self.parts, loaded, strict=True):
            for a, b in zip(expected, actual, strict=True):
                np.testing.assert_array_equal(a, b)
        with mock.patch.object(_convex_cache.os, "replace", side_effect=OSError("publish failed")):
            _convex_cache.write(self.cache_dir, "failed", self.parts)
        self.assertEqual([path.name for path in self.cache_dir.iterdir()], ["shared.convex.npz"])


if __name__ == "__main__":
    unittest.main()
