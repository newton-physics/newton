# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the on-disk cooked-SDF cache.

The temp-directory convention here mirrors Warp's
``tests/cuda/test_conditional_captures.test_graph_debug_dot_print`` —
files are placed directly under ``tempfile.gettempdir()`` rather than a
``TemporaryDirectory`` context. CI environments are reliable about the
system temp dir but can be flaky about other locations, so this keeps
behaviour identical to the upstream pattern.
"""

import os
import shutil
import tempfile
import threading
import unittest
import uuid
from pathlib import Path

import numpy as np
import warp as wp

from newton import Mesh
from newton._src.geometry import _sdf_cache
from newton._src.geometry.sdf_texture import (
    QuantizationMode,
    TextureSDFData,
    texture_sample_sdf,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_cuda_available = wp.is_cuda_available()


def _make_cache_dir(tag: str) -> Path:
    """Create a fresh, uniquely-named cache directory under ``$TMPDIR``.

    Using ``tempfile.gettempdir()`` matches the Warp test convention for
    artifacts that need a writable, well-known temp location on CI.
    A short uuid suffix isolates parallel test workers.
    """

    base = Path(tempfile.gettempdir()) / f"newton_sdf_cache_test_{tag}_{uuid.uuid4().hex[:8]}"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _remove_cache_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _make_box_mesh() -> Mesh:
    hx = hy = hz = 0.5
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            0,
            2,
            1,
            0,
            3,
            2,
            4,
            5,
            6,
            4,
            6,
            7,
            0,
            1,
            5,
            0,
            5,
            4,
            2,
            3,
            7,
            2,
            7,
            6,
            0,
            4,
            7,
            0,
            7,
            3,
            1,
            2,
            6,
            1,
            6,
            5,
        ],
        dtype=np.int32,
    )
    return Mesh(vertices, indices)


def _common_hash_kwargs(vertices: np.ndarray, indices: np.ndarray) -> dict:
    return {
        "vertices": vertices,
        "indices": indices,
        "is_solid": True,
        "narrow_band_range": (-0.1, 0.1),
        "target_voxel_size": None,
        "max_resolution": 64,
        "margin": 0.05,
        "texture_format": "uint16",
        "sign_method_resolved": "parity",
        "winding_threshold": 0.5,
        "scale": None,
    }


@wp.kernel
def _sample_sdf_kernel(
    sdf: TextureSDFData,
    points: wp.array[wp.vec3],
    out: wp.array[float],
) -> None:
    tid = wp.tid()
    out[tid] = texture_sample_sdf(sdf, points[tid])


def _sample(sdf: TextureSDFData, points_np: np.ndarray, device: str) -> np.ndarray:
    pts = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.zeros(points_np.shape[0], dtype=float, device=device)
    wp.launch(_sample_sdf_kernel, dim=points_np.shape[0], inputs=[sdf, pts, out], device=device)
    return out.numpy()


# -----------------------------------------------------------------------------
# Hash + serialization tests (no GPU required)
# -----------------------------------------------------------------------------


class TestSDFDiskCachePure(unittest.TestCase):
    """Tests that exercise hashing and on-disk format only."""

    def setUp(self) -> None:
        self.mesh = _make_box_mesh()
        self.vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
        self.indices = np.asarray(self.mesh.indices, dtype=np.int32).reshape(-1)
        self.cache_dir = _make_cache_dir(self._testMethodName)

    def tearDown(self) -> None:
        _remove_cache_dir(self.cache_dir)

    def test_hash_is_stable(self) -> None:
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h1 = _sdf_cache.hash_inputs(**kwargs)
        h2 = _sdf_cache.hash_inputs(**kwargs)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 32)

    def test_hash_changes_with_params(self) -> None:
        base = _sdf_cache.hash_inputs(**_common_hash_kwargs(self.vertices, self.indices))

        sensitive = [
            ("narrow_band_range", (-0.2, 0.2)),
            ("target_voxel_size", 0.05),
            ("max_resolution", 32),
            ("margin", 0.1),
            ("texture_format", "float32"),
            ("sign_method_resolved", "winding"),
            ("winding_threshold", -0.5),
            ("scale", (2.0, 1.0, 1.0)),
            ("is_solid", False),
        ]
        for name, value in sensitive:
            kwargs = _common_hash_kwargs(self.vertices, self.indices)
            kwargs[name] = value
            h = _sdf_cache.hash_inputs(**kwargs)
            self.assertNotEqual(h, base, f"hash should differ when {name} changes")

    def test_hash_changes_with_mesh(self) -> None:
        base = _sdf_cache.hash_inputs(**_common_hash_kwargs(self.vertices, self.indices))

        moved = self.vertices.copy()
        moved[0, 0] += 0.1
        kwargs = _common_hash_kwargs(moved, self.indices)
        h = _sdf_cache.hash_inputs(**kwargs)
        self.assertNotEqual(h, base)

    def test_hash_winding_threshold_sign_only(self) -> None:
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        kwargs["winding_threshold"] = 0.5
        h_pos_a = _sdf_cache.hash_inputs(**kwargs)
        kwargs["winding_threshold"] = 0.7
        h_pos_b = _sdf_cache.hash_inputs(**kwargs)
        self.assertEqual(h_pos_a, h_pos_b, "hash must be insensitive to winding_threshold magnitude")

        kwargs["winding_threshold"] = -0.5
        h_neg = _sdf_cache.hash_inputs(**kwargs)
        self.assertNotEqual(h_pos_a, h_neg, "hash must reflect winding_threshold sign")

    def _fake_sparse_data(self) -> dict:
        """A minimal but schema-correct sparse_data dict for round-trip tests."""

        return {
            "coarse_sdf": np.zeros((4, 4, 4), dtype=np.float32),
            "subgrid_data": np.zeros((1, 1, 1), dtype=np.float32),
            "subgrid_start_slots": np.zeros((2, 2, 2), dtype=np.uint32),
            "subgrid_required": np.zeros(8, dtype=np.int32),
            "subgrid_occupied": np.zeros(8, dtype=np.int32),
            "coarse_dims": (2, 2, 2),
            "subgrid_tex_size": 1,
            "num_subgrids": 0,
            "min_extents": np.array([-0.5, -0.5, -0.5], dtype=np.float64),
            "max_extents": np.array([0.5, 0.5, 0.5], dtype=np.float64),
            "cell_size": np.array([0.25, 0.25, 0.25], dtype=np.float64),
            "subgrid_size": 8,
            "quantization_mode": int(QuantizationMode.UINT16),
            "subgrids_min_sdf_value": 0.0,
            "subgrids_sdf_value_range": 1.0,
        }

    def test_round_trip_save_and_load(self) -> None:
        sparse_data = self._fake_sparse_data()
        tmp = self.cache_dir
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h = _sdf_cache.hash_inputs(**kwargs)
        _sdf_cache.save_sparse_data(tmp, h, sparse_data, newton_version="test")
        npz_path = _sdf_cache.cache_path(tmp, h)
        self.assertTrue(npz_path.exists())

        loaded = _sdf_cache.try_load_sparse_data(tmp, h)
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded["coarse_sdf"], sparse_data["coarse_sdf"])
        np.testing.assert_array_equal(loaded["subgrid_start_slots"], sparse_data["subgrid_start_slots"])
        self.assertEqual(loaded["coarse_dims"], sparse_data["coarse_dims"])
        self.assertEqual(loaded["subgrid_size"], sparse_data["subgrid_size"])
        self.assertEqual(loaded["quantization_mode"], sparse_data["quantization_mode"])

    def test_npz_contains_provenance_and_expected_arrays(self) -> None:
        sparse_data = self._fake_sparse_data()
        tmp = self.cache_dir
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h = _sdf_cache.hash_inputs(**kwargs)
        _sdf_cache.save_sparse_data(tmp, h, sparse_data, newton_version="test")
        npz_path = _sdf_cache.cache_path(tmp, h)

        with np.load(npz_path, allow_pickle=False) as npz:
            present = set(npz.files)
            for required in (
                "__cache_format_version__",
                "__kind__",
                "__newton_version__",
                "__created_utc__",
                "coarse_sdf",
                "subgrid_data",
                "subgrid_start_slots",
                "subgrid_required",
                "subgrid_occupied",
            ):
                self.assertIn(required, present)
            self.assertEqual(int(npz["__cache_format_version__"].item()), _sdf_cache.CACHE_FORMAT_VERSION)
            self.assertEqual(str(npz["__kind__"].item()), "newton.texture_sdf")
            self.assertEqual(str(npz["__newton_version__"].item()), "test")
            created_utc = str(npz["__created_utc__"].item())
            self.assertTrue(created_utc.endswith("+00:00") or created_utc.endswith("Z"))

    def test_missing_files_is_miss(self) -> None:
        self.assertIsNone(_sdf_cache.try_load_sparse_data(self.cache_dir, "deadbeef"))

    def test_corrupt_npz_is_miss(self) -> None:
        sparse_data = self._fake_sparse_data()
        tmp = self.cache_dir
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h = _sdf_cache.hash_inputs(**kwargs)
        _sdf_cache.save_sparse_data(tmp, h, sparse_data, newton_version="test")
        npz_path = _sdf_cache.cache_path(tmp, h)
        npz_path.write_bytes(b"not an npz")
        self.assertIsNone(_sdf_cache.try_load_sparse_data(tmp, h))

    def test_newton_version_is_resolved(self) -> None:
        # Regression: the relative ``from .. import __version__`` previously
        # resolved to ``newton._src`` (no ``__version__``) and silently fell
        # back to the diagnostic string ``"unknown"``.  The diagnostic value
        # is non-load-bearing but should reflect the real package version.
        import newton  # noqa: PLC0415

        resolved = _sdf_cache._resolve_newton_version()
        self.assertEqual(resolved, newton.__version__)
        self.assertNotEqual(resolved, "unknown")

        # Round-trip: when no explicit version is passed, the resolved
        # string must end up in the on-disk metadata.
        sparse_data = self._fake_sparse_data()
        h = _sdf_cache.hash_inputs(**_common_hash_kwargs(self.vertices, self.indices))
        _sdf_cache.save_sparse_data(self.cache_dir, h, sparse_data)
        with np.load(_sdf_cache.cache_path(self.cache_dir, h), allow_pickle=False) as npz:
            self.assertEqual(str(npz["__newton_version__"].item()), newton.__version__)

    def test_concurrent_writers_do_not_collide(self) -> None:
        # Regression: a fixed ``{hash}.sdf.npz.tmp.npz`` tmp filename meant
        # two writers cooking the same hash would trample each other's
        # in-flight ``np.savez`` and ``os.replace`` could publish a
        # partially-written file.  The tmp filename now embeds the PID and
        # a random token so concurrent cookers each get their own scratch
        # file; the final published ``{hash}.sdf.npz`` is loadable.
        sparse_data = self._fake_sparse_data()
        h = _sdf_cache.hash_inputs(**_common_hash_kwargs(self.vertices, self.indices))

        num_workers = 8
        errors: list[BaseException] = []

        def _worker() -> None:
            try:
                _sdf_cache.save_sparse_data(self.cache_dir, h, sparse_data, newton_version="test")
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"concurrent writers raised: {errors}")

        loaded = _sdf_cache.try_load_sparse_data(self.cache_dir, h)
        self.assertIsNotNone(loaded, "published cache file must be loadable after concurrent writes")
        np.testing.assert_array_equal(loaded["coarse_sdf"], sparse_data["coarse_sdf"])

        # No stragglers: every writer cleans up its own tmp file via
        # ``os.replace``, so only the canonical ``.sdf.npz`` remains.
        leftover_tmp = list(self.cache_dir.glob("*.tmp.npz"))
        self.assertEqual(leftover_tmp, [], f"unexpected tmp files left behind: {leftover_tmp}")

    def test_save_uses_unique_tmp_filename(self) -> None:
        # Whitebox guard against future regressions: each ``save_sparse_data``
        # call must use a distinct tmp filename so concurrent cookers in the
        # same directory cannot collide.  We snapshot the tmp directory
        # contents from a sibling thread mid-save; with PID+token in the tmp
        # name, two saves for the same hash should produce two distinct
        # tmp files (observed across runs, not necessarily simultaneously).
        sparse_data = self._fake_sparse_data()
        h = _sdf_cache.hash_inputs(**_common_hash_kwargs(self.vertices, self.indices))

        seen_tmps: set[str] = set()
        original_replace = os.replace

        def _spy_replace(src, dst):
            seen_tmps.add(os.fspath(src))
            return original_replace(src, dst)

        try:
            os.replace = _spy_replace  # type: ignore[assignment]
            for _ in range(4):
                _sdf_cache.save_sparse_data(self.cache_dir, h, sparse_data, newton_version="test")
        finally:
            os.replace = original_replace  # type: ignore[assignment]

        self.assertEqual(len(seen_tmps), 4, f"expected 4 distinct tmp filenames, got {seen_tmps}")
        for tmp_name in seen_tmps:
            self.assertTrue(tmp_name.endswith(".tmp.npz"), tmp_name)

    def test_embedded_version_mismatch_is_miss(self) -> None:
        sparse_data = self._fake_sparse_data()
        tmp = self.cache_dir
        kwargs = _common_hash_kwargs(self.vertices, self.indices)
        h = _sdf_cache.hash_inputs(**kwargs)
        _sdf_cache.save_sparse_data(tmp, h, sparse_data, newton_version="test")
        npz_path = _sdf_cache.cache_path(tmp, h)
        with np.load(npz_path) as npz:
            contents = {k: npz[k] for k in npz.files if k != "__cache_format_version__"}
        contents["__cache_format_version__"] = np.asarray(_sdf_cache.CACHE_FORMAT_VERSION + 999, dtype=np.int32)
        np.savez(npz_path, **contents)
        self.assertIsNone(_sdf_cache.try_load_sparse_data(tmp, h))


# -----------------------------------------------------------------------------
# End-to-end hit/miss test (CUDA required)
# -----------------------------------------------------------------------------


def test_disk_cache_hit_matches_live(test, device) -> None:
    """A cache hit must produce SDF samples matching a fresh cook."""

    mesh = _make_box_mesh()
    cache_path = _make_cache_dir("hit_matches_live")
    try:
        sdf_live = mesh.build_sdf(device=device, cache_dir=cache_path)
        cache_files = list(cache_path.glob("*.sdf.npz"))
        test.assertEqual(len(cache_files), 1, f"expected exactly one cache file, found {cache_files}")
        sidecar_files = list(cache_path.glob("*.sdf.json"))
        test.assertEqual(len(sidecar_files), 0, "JSON sidecar should no longer be produced")

        rng = np.random.default_rng(seed=0)
        points = rng.uniform(-0.6, 0.6, size=(64, 3)).astype(np.float32)
        live_values = _sample(sdf_live.texture_data, points, device)

        mesh2 = _make_box_mesh()
        sdf_cached = mesh2.build_sdf(device=device, cache_dir=cache_path)
        cached_values = _sample(sdf_cached.texture_data, points, device)

        np.testing.assert_allclose(
            cached_values,
            live_values,
            rtol=1e-5,
            atol=1e-5,
            err_msg="cached SDF samples must match the freshly cooked SDF",
        )

        test.assertIsNotNone(sdf_cached.texture_block_coords)
        test.assertGreater(len(sdf_cached.texture_block_coords), 0)
    finally:
        _remove_cache_dir(cache_path)


def test_disk_cache_param_change_invalidates(test, device) -> None:
    """Different build parameters must produce different cache entries."""

    cache_path = _make_cache_dir("param_change")
    try:
        mesh = _make_box_mesh()
        mesh.build_sdf(device=device, cache_dir=cache_path, max_resolution=32)

        mesh2 = _make_box_mesh()
        mesh2.build_sdf(device=device, cache_dir=cache_path, max_resolution=64)

        files = sorted(cache_path.glob("*.sdf.npz"))
        test.assertEqual(
            len(files),
            2,
            f"expected two distinct cache entries, found {[p.name for p in files]}",
        )
    finally:
        _remove_cache_dir(cache_path)


# -----------------------------------------------------------------------------
# Test class wiring
# -----------------------------------------------------------------------------


class TestSDFDiskCacheCuda(unittest.TestCase):
    pass


_cuda_devices = get_cuda_test_devices()
add_function_test(
    TestSDFDiskCacheCuda,
    "test_disk_cache_hit_matches_live",
    test_disk_cache_hit_matches_live,
    devices=_cuda_devices,
)
add_function_test(
    TestSDFDiskCacheCuda,
    "test_disk_cache_param_change_invalidates",
    test_disk_cache_param_change_invalidates,
    devices=_cuda_devices,
)


if __name__ == "__main__":
    unittest.main()
