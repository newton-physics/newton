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

import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import git
except ImportError:
    git = None

from newton._src.utils.download_assets import _safe_rmtree, download_git_folder


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
