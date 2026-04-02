# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import errno
import hashlib
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import git
except ImportError:
    git = None

from newton._src.utils.download_assets import (
    _TEMP_DIR_RE,
    _cleanup_old_versions,
    _cleanup_stale_temp_dirs,
    _find_cached_version,
    _find_parent_cache,
    _get_latest_commit_via_git,
    _safe_rename,
    _safe_rmtree,
    _temp_cache_path,
    download_git_folder,
)


@unittest.skipIf(git is None or shutil.which("git") is None, "GitPython or git not available")
class TestDownloadAssets(unittest.TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp(prefix="nwtn_cache_")
        self.remote_dir = tempfile.mkdtemp(prefix="nwtn_remote_")
        self.work_dir = tempfile.mkdtemp(prefix="nwtn_work_")

        self.remote = git.Repo.init(self.remote_dir, bare=True)

        self.work = git.Repo.init(self.work_dir)
        with self.work.config_writer() as cw:
            cw.set_value("user", "name", "Newton CI")
            cw.set_value("user", "email", "ci@newton.dev")

        self.asset_rel = "assets/x"
        asset_path = Path(self.work_dir, self.asset_rel)
        asset_path.mkdir(parents=True, exist_ok=True)
        (asset_path / "foo.txt").write_text("v1\n", encoding="utf-8")

        self.work.index.add([str(asset_path / "foo.txt")])
        self.work.index.commit("initial")
        if "origin" not in [r.name for r in self.work.remotes]:
            self.work.create_remote("origin", self.remote_dir)
        self.work.git.branch("-M", "main")
        self.work.git.push("--set-upstream", "origin", "main")

    def tearDown(self):
        try:
            if hasattr(self, "work"):
                self.work.close()
        except Exception:
            pass
        _safe_rmtree(self.cache_dir)
        _safe_rmtree(self.work_dir)
        _safe_rmtree(self.remote_dir)

    def test_download_and_refresh(self):
        # Initial download
        p1 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
        self.assertTrue(p1.exists())
        self.assertEqual((p1 / "foo.txt").read_text(encoding="utf-8"), "v1\n")
        # Navigate up past folder_path segments to reach the SHA-named cache root
        depth = len(Path(self.asset_rel).parts)
        cache_dir_1 = p1
        for _ in range(depth):
            cache_dir_1 = cache_dir_1.parent

        # Advance remote
        (Path(self.work_dir, self.asset_rel) / "foo.txt").write_text("v2\n", encoding="utf-8")
        self.work.index.add([str(Path(self.work_dir, self.asset_rel) / "foo.txt")])
        self.work.index.commit("update")
        self.work.git.push("origin", "main")

        # Invalidate TTL so the next call checks remote
        old_mtime = time.time() - 7200
        os.utime(cache_dir_1, (old_mtime, old_mtime))

        # Refresh — should get a NEW directory (different SHA)
        p2 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
        self.assertTrue(p2.exists())
        self.assertEqual((p2 / "foo.txt").read_text(encoding="utf-8"), "v2\n")
        cache_dir_2 = p2
        for _ in range(depth):
            cache_dir_2 = cache_dir_2.parent
        self.assertNotEqual(cache_dir_1, cache_dir_2)

        # Old version should have been cleaned up (best-effort)
        self.assertFalse(cache_dir_1.exists(), "Old cache dir should be cleaned up after new download")

        # Force refresh with same SHA — should return same directory
        p3 = download_git_folder(
            self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main", force_refresh=True
        )
        cache_dir_3 = p3
        for _ in range(depth):
            cache_dir_3 = cache_dir_3.parent
        self.assertEqual(cache_dir_2, cache_dir_3)
        self.assertEqual((p3 / "foo.txt").read_text(encoding="utf-8"), "v2\n")

    def test_concurrent_download(self):
        """Multiple threads downloading the same asset do not corrupt the cache."""

        def download():
            p = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
            self.assertTrue(p.exists())
            self.assertEqual((p / "foo.txt").read_text(encoding="utf-8"), "v1\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download) for _ in range(4)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        # All threads should have resolved to the same SHA directory
        identity_hash = hashlib.md5(f"{self.remote_dir}#{self.asset_rel}#main".encode()).hexdigest()[:8]
        repo_name = Path(self.remote_dir.rstrip("/")).stem
        folder_name = self.asset_rel.replace("/", "_")
        base_prefix = f"{repo_name}_{folder_name}_{identity_hash}"
        entries = [
            e
            for e in Path(self.cache_dir).iterdir()
            if e.is_dir()
            and e.name.startswith(f"{base_prefix}_")
            and not _TEMP_DIR_RE.search(e.name[len(base_prefix) :])
        ]
        self.assertEqual(len(entries), 1, f"Expected 1 cache dir, got {len(entries)}: {entries}")

    def test_within_ttl_skips_network(self):
        """Within TTL, return cached path without calling git ls-remote."""
        p1 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
        self.assertTrue(p1.exists())

        with mock.patch("newton._src.utils.download_assets._get_latest_commit_via_git") as mock_ls:
            p2 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
            mock_ls.assert_not_called()
        self.assertEqual(p1, p2)

    def test_offline_returns_cached(self):
        """When git ls-remote fails and cache exists, return cached version."""
        p1 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
        self.assertTrue(p1.exists())
        depth = len(Path(self.asset_rel).parts)
        cache_dir_1 = p1
        for _ in range(depth):
            cache_dir_1 = cache_dir_1.parent

        # Expire TTL
        old_mtime = time.time() - 7200
        os.utime(cache_dir_1, (old_mtime, old_mtime))

        # Simulate offline
        with mock.patch("newton._src.utils.download_assets._get_latest_commit_via_git", return_value=None):
            p2 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
        self.assertEqual(p1, p2)

    def test_offline_no_cache_raises(self):
        """When git ls-remote fails and no cache exists, raise RuntimeError."""
        with mock.patch("newton._src.utils.download_assets._get_latest_commit_via_git", return_value=None):
            with self.assertRaises(RuntimeError):
                download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")

    def test_download_by_tag(self):
        """Downloading by tag name resolves correctly."""
        self.work.create_tag("v1.0", message="release v1.0")
        self.work.git.push("origin", "v1.0")

        p = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="v1.0")
        self.assertTrue(p.exists())
        self.assertEqual((p / "foo.txt").read_text(encoding="utf-8"), "v1\n")


class TestSafeRename(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="nwtn_rename_")

    def tearDown(self):
        _safe_rmtree(self.base)

    def test_rename_success(self):
        """Rename succeeds when destination does not exist."""
        src = os.path.join(self.base, "src_dir")
        dst = os.path.join(self.base, "dst_dir")
        os.makedirs(src)
        Path(src, "file.txt").write_text("hello", encoding="utf-8")

        _safe_rename(src, dst)

        self.assertTrue(os.path.isdir(dst))
        self.assertEqual(Path(dst, "file.txt").read_text(encoding="utf-8"), "hello")
        self.assertFalse(os.path.exists(src))

    def test_rename_destination_exists(self):
        """Rename is a no-op when destination already exists."""
        src = os.path.join(self.base, "src_dir")
        dst = os.path.join(self.base, "dst_dir")
        os.makedirs(src)
        os.makedirs(dst)
        Path(dst, "existing.txt").write_text("keep", encoding="utf-8")
        Path(src, "new.txt").write_text("discard", encoding="utf-8")

        _safe_rename(src, dst)

        # Destination content unchanged
        self.assertEqual(Path(dst, "existing.txt").read_text(encoding="utf-8"), "keep")
        # Source still exists (caller is responsible for cleanup)
        self.assertTrue(os.path.exists(src))

    def test_rename_retries_on_transient_error(self):
        """Transient OSError succeeds on retry."""
        src = os.path.join(self.base, "src_dir")
        dst = os.path.join(self.base, "dst_dir")
        os.makedirs(src)

        real_rename = os.rename
        call_count = 0

        def flaky_rename(s, d):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError(errno.EACCES, "transient lock")
            return real_rename(s, d)

        with mock.patch("os.rename", side_effect=flaky_rename):
            _safe_rename(src, dst, attempts=3, delay=0)

        self.assertTrue(os.path.isdir(dst))
        self.assertEqual(call_count, 2)

    def test_rename_raises_after_exhausting_retries(self):
        """Raises OSError when all retry attempts are exhausted."""
        src = os.path.join(self.base, "src_dir")
        dst = os.path.join(self.base, "dst_dir")
        os.makedirs(src)

        with mock.patch("os.rename", side_effect=OSError(errno.EACCES, "persistent lock")):
            with self.assertRaises(OSError):
                _safe_rename(src, dst, attempts=3, delay=0)

    def test_rename_enotempty_returns_silently(self):
        """ENOTEMPTY is treated the same as FileExistsError."""
        src = os.path.join(self.base, "src_dir")
        dst = os.path.join(self.base, "dst_dir")
        os.makedirs(src)
        os.makedirs(dst)

        with mock.patch("os.rename", side_effect=OSError(errno.ENOTEMPTY, "not empty")):
            _safe_rename(src, dst)

        # Both dirs still exist — caller cleans up src
        self.assertTrue(os.path.isdir(src))
        self.assertTrue(os.path.isdir(dst))


class TestTempCachePath(unittest.TestCase):
    def test_includes_pid_and_tid(self):
        """Temp path includes PID and thread ID for uniqueness."""
        base = Path("/tmp/cache_folder")
        result = _temp_cache_path(base)
        self.assertIn(f"_p{os.getpid()}", str(result))
        self.assertIn(f"_t{threading.get_ident()}", str(result))

    def test_different_threads_get_different_paths(self):
        """Different threads produce different temp paths."""
        base = Path("/tmp/cache_folder")
        results = []

        def collect():
            results.append(_temp_cache_path(base))

        t = threading.Thread(target=collect)
        t.start()
        t.join()
        results.append(_temp_cache_path(base))

        self.assertNotEqual(results[0], results[1])


class TestCleanupStaleTempDirs(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="nwtn_cleanup_")
        self.cache_path = Path(self.base)
        self.base_prefix = "repo_asset_abc12345"

    def tearDown(self):
        _safe_rmtree(self.base)

    def test_removes_old_temp_dirs(self):
        """Orphaned temp dirs older than max_age are removed."""
        old_temp = self.cache_path / f"{self.base_prefix}_deadbeef_p99999_t99999"
        old_temp.mkdir(parents=True)
        old_mtime = time.time() - 7200
        os.utime(old_temp, (old_mtime, old_mtime))

        _cleanup_stale_temp_dirs(self.cache_path, self.base_prefix, max_age=3600)

        self.assertFalse(old_temp.exists())

    def test_preserves_recent_temp_dirs(self):
        """Recent temp dirs (within max_age) are not removed."""
        recent_temp = self.cache_path / f"{self.base_prefix}_deadbeef_p99999_t99999"
        recent_temp.mkdir(parents=True)

        _cleanup_stale_temp_dirs(self.cache_path, self.base_prefix, max_age=3600)

        self.assertTrue(recent_temp.exists())

    def test_ignores_non_temp_version_dirs(self):
        """Content-hash version dirs are NOT cleaned by temp cleanup."""
        version_dir = self.cache_path / f"{self.base_prefix}_deadbeef"
        version_dir.mkdir(parents=True)
        old_mtime = time.time() - 7200
        os.utime(version_dir, (old_mtime, old_mtime))

        _cleanup_stale_temp_dirs(self.cache_path, self.base_prefix, max_age=3600)

        self.assertTrue(version_dir.exists())

    def test_ignores_unrelated_dirs(self):
        """Directories that don't match the prefix are untouched."""
        unrelated = self.cache_path / "other_asset_xyz_deadbeef_p99999_t99999"
        unrelated.mkdir(parents=True)
        old_mtime = time.time() - 7200
        os.utime(unrelated, (old_mtime, old_mtime))

        _cleanup_stale_temp_dirs(self.cache_path, self.base_prefix, max_age=3600)

        self.assertTrue(unrelated.exists())


class TestFindCachedVersion(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="nwtn_find_")
        self.cache_path = Path(self.base)
        self.base_prefix = "newton-assets_model_abc12345"

    def tearDown(self):
        _safe_rmtree(self.base)

    def test_returns_none_when_no_match(self):
        result = _find_cached_version(self.cache_path, self.base_prefix)
        self.assertIsNone(result)

    def test_finds_single_cached_dir(self):
        d = self.cache_path / f"{self.base_prefix}_deadbeef"
        d.mkdir()
        result = _find_cached_version(self.cache_path, self.base_prefix)
        self.assertEqual(result, d)

    def test_excludes_temp_dirs(self):
        temp = self.cache_path / f"{self.base_prefix}_deadbeef_p1234_t5678"
        temp.mkdir()
        result = _find_cached_version(self.cache_path, self.base_prefix)
        self.assertIsNone(result)

    def test_picks_newest_mtime_when_multiple(self):
        old = self.cache_path / f"{self.base_prefix}_aaaa1111"
        new = self.cache_path / f"{self.base_prefix}_bbbb2222"
        old.mkdir()
        new.mkdir()
        old_mtime = time.time() - 7200
        os.utime(old, (old_mtime, old_mtime))
        result = _find_cached_version(self.cache_path, self.base_prefix)
        self.assertEqual(result, new)

    def test_ignores_unrelated_dirs(self):
        unrelated = self.cache_path / "other-repo_model_abc12345_deadbeef"
        unrelated.mkdir()
        result = _find_cached_version(self.cache_path, self.base_prefix)
        self.assertIsNone(result)


class TestFindParentCache(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="nwtn_parent_")
        self.cache_path = Path(self.base)

    def tearDown(self):
        _safe_rmtree(self.base)

    def test_finds_parent_with_sha_suffix(self):
        """Parent cache with SHA suffix is found for subfolder request."""
        parent_hash = hashlib.md5(b"http://example.git#unitree_g1#main").hexdigest()[:8]
        parent_dir = self.cache_path / f"repo_unitree_g1_{parent_hash}_abcd1234"
        (parent_dir / "unitree_g1" / "usd").mkdir(parents=True)
        (parent_dir / ".git").mkdir()

        result = _find_parent_cache(self.cache_path, "repo", "unitree_g1/usd", "main", "http://example.git")
        self.assertIsNotNone(result)
        parent, target = result
        self.assertEqual(parent, parent_dir)
        self.assertTrue(target.exists())

    def test_returns_none_when_subfolder_missing(self):
        """Returns None if parent exists but subfolder does not."""
        parent_hash = hashlib.md5(b"http://example.git#unitree_g1#main").hexdigest()[:8]
        parent_dir = self.cache_path / f"repo_unitree_g1_{parent_hash}_abcd1234"
        parent_dir.mkdir(parents=True)
        (parent_dir / ".git").mkdir()

        result = _find_parent_cache(self.cache_path, "repo", "unitree_g1/usd", "main", "http://example.git")
        self.assertIsNone(result)

    def test_returns_none_for_single_path(self):
        """Single-segment paths have no parent to check."""
        result = _find_parent_cache(self.cache_path, "repo", "unitree_g1", "main", "http://example.git")
        self.assertIsNone(result)


@unittest.skipIf(git is None or shutil.which("git") is None, "GitPython or git not available")
class TestGetLatestCommitViaGit(unittest.TestCase):
    def setUp(self):
        self.remote_dir = tempfile.mkdtemp(prefix="nwtn_remote_")
        self.work_dir = tempfile.mkdtemp(prefix="nwtn_work_")

        self.remote = git.Repo.init(self.remote_dir, bare=True)
        self.work = git.Repo.init(self.work_dir)
        with self.work.config_writer() as cw:
            cw.set_value("user", "name", "Newton CI")
            cw.set_value("user", "email", "ci@newton.dev")

        (Path(self.work_dir) / "file.txt").write_text("v1\n", encoding="utf-8")
        self.work.index.add(["file.txt"])
        self.work.index.commit("initial")
        self.work.create_remote("origin", self.remote_dir)
        self.work.git.branch("-M", "main")
        self.work.git.push("--set-upstream", "origin", "main")
        self.commit_sha = self.work.head.commit.hexsha

    def tearDown(self):
        try:
            self.work.close()
        except Exception:
            pass
        _safe_rmtree(self.work_dir)
        _safe_rmtree(self.remote_dir)

    def test_resolves_branch(self):
        result = _get_latest_commit_via_git(self.remote_dir, "main")
        self.assertEqual(result, self.commit_sha)

    def test_resolves_lightweight_tag(self):
        self.work.create_tag("v1.0")
        self.work.git.push("origin", "v1.0")
        result = _get_latest_commit_via_git(self.remote_dir, "v1.0")
        self.assertEqual(result, self.commit_sha)

    def test_resolves_annotated_tag(self):
        self.work.create_tag("v2.0", message="release v2.0")
        self.work.git.push("origin", "v2.0")
        result = _get_latest_commit_via_git(self.remote_dir, "v2.0")
        # Should return the commit SHA, not the tag object SHA
        self.assertEqual(result, self.commit_sha)

    def test_full_sha_passthrough(self):
        result = _get_latest_commit_via_git(self.remote_dir, self.commit_sha)
        self.assertEqual(result, self.commit_sha)

    def test_nonexistent_ref_returns_none(self):
        result = _get_latest_commit_via_git(self.remote_dir, "no-such-branch")
        self.assertIsNone(result)


class TestCleanupOldVersions(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="nwtn_oldver_")
        self.cache_path = Path(self.base)
        self.base_prefix = "repo_asset_abc12345"

    def tearDown(self):
        _safe_rmtree(self.base)

    def test_removes_old_version_dirs(self):
        old = self.cache_path / f"{self.base_prefix}_aaaa1111"
        current = self.cache_path / f"{self.base_prefix}_bbbb2222"
        old.mkdir()
        current.mkdir()

        _cleanup_old_versions(self.cache_path, self.base_prefix, current)

        self.assertFalse(old.exists())
        self.assertTrue(current.exists())

    def test_preserves_temp_dirs(self):
        temp = self.cache_path / f"{self.base_prefix}_aaaa1111_p1234_t5678"
        current = self.cache_path / f"{self.base_prefix}_bbbb2222"
        temp.mkdir()
        current.mkdir()

        _cleanup_old_versions(self.cache_path, self.base_prefix, current)

        self.assertTrue(temp.exists())
        self.assertTrue(current.exists())

    def test_ignores_unrelated_dirs(self):
        unrelated = self.cache_path / "other_asset_xyz_aaaa1111"
        current = self.cache_path / f"{self.base_prefix}_bbbb2222"
        unrelated.mkdir()
        current.mkdir()

        _cleanup_old_versions(self.cache_path, self.base_prefix, current)

        self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
