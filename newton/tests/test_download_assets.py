# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import concurrent.futures
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

try:
    import git
except ImportError:
    git = None

from newton._src.utils.download_assets import (
    _cleanup_stale_temp_dirs,
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

    def _cache_root(self) -> Path:
        entries = list(Path(self.cache_dir).iterdir())
        self.assertTrue(entries, "cache folder should exist")
        return entries[0]

    def _stamp_file(self) -> Path:
        return self._cache_root() / ".newton_last_check"

    def _cached_sha(self) -> str:
        repo = git.Repo(self._cache_root())
        try:
            return repo.head.commit.hexsha
        finally:
            repo.close()

    def test_download_and_refresh(self):
        # Initial download
        p1 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, branch="main")
        self.assertTrue(p1.exists())
        sha1 = self._cached_sha()

        # Advance remote
        (Path(self.work_dir, self.asset_rel) / "foo.txt").write_text("v2\n", encoding="utf-8")
        self.work.index.add([str(Path(self.work_dir, self.asset_rel) / "foo.txt")])
        self.work.index.commit("update")
        self.work.git.push("origin", "main")

        # Invalidate TTL so the next call performs the remote check
        stamp = self._stamp_file()
        if stamp.exists():
            stamp.unlink()

        # Refresh
        p2 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, branch="main")
        self.assertEqual(p1, p2)
        sha2 = self._cached_sha()
        self.assertNotEqual(sha1, sha2)

        # Force refresh path
        p3 = download_git_folder(
            self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, branch="main", force_refresh=True
        )
        self.assertEqual(p2, p3)
        sha3 = self._cached_sha()
        self.assertEqual(sha2, sha3)

    def test_concurrent_download(self):
        """Multiple threads downloading the same asset do not corrupt the cache."""

        def download():
            p = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, branch="main")
            self.assertTrue(p.exists())
            self.assertEqual((p / "foo.txt").read_text(encoding="utf-8"), "v1\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download) for _ in range(4)]
            for f in concurrent.futures.as_completed(futures):
                f.result()


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
        self.cache_folder = Path(self.base) / "repo_asset_abc12345"

    def tearDown(self):
        _safe_rmtree(self.base)

    def test_removes_old_temp_dirs(self):
        """Orphaned temp dirs older than max_age are removed."""
        old_temp = Path(f"{self.cache_folder}_p99999_t99999")
        old_temp.mkdir(parents=True)
        # Backdate mtime by 2 hours
        old_mtime = time.time() - 7200
        os.utime(old_temp, (old_mtime, old_mtime))

        _cleanup_stale_temp_dirs(self.cache_folder, max_age=3600)

        self.assertFalse(old_temp.exists())

    def test_removes_old_stale_dirs(self):
        """Orphaned stale dirs older than max_age are removed."""
        old_stale = Path(f"{self.cache_folder}_stale_p99999_t99999")
        old_stale.mkdir(parents=True)
        old_mtime = time.time() - 7200
        os.utime(old_stale, (old_mtime, old_mtime))

        _cleanup_stale_temp_dirs(self.cache_folder, max_age=3600)

        self.assertFalse(old_stale.exists())

    def test_preserves_recent_temp_dirs(self):
        """Recent temp dirs (within max_age) are not removed."""
        recent_temp = Path(f"{self.cache_folder}_p99999_t99999")
        recent_temp.mkdir(parents=True)

        _cleanup_stale_temp_dirs(self.cache_folder, max_age=3600)

        self.assertTrue(recent_temp.exists())

    def test_ignores_unrelated_dirs(self):
        """Directories that don't match the temp pattern are untouched."""
        unrelated = Path(self.base) / "repo_asset_abc12345_other"
        unrelated.mkdir(parents=True)
        old_mtime = time.time() - 7200
        os.utime(unrelated, (old_mtime, old_mtime))

        _cleanup_stale_temp_dirs(self.cache_folder, max_age=3600)

        self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
