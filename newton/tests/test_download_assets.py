# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import errno
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import newton
from newton import config

try:
    import git
except ImportError:
    git = None

from newton._src.utils.download_assets import (
    _TEMP_DIR_RE,
    _cache_prefix,
    _cleanup_stale_temp_dirs,
    _find_cached_version,
    _find_parent_cache,
    _get_latest_commit_via_git,
    _read_cached_ref,
    _ref_key,
    _ref_path,
    _safe_rename,
    _safe_rmtree,
    _temp_cache_path,
    _write_cached_ref,
    clear_git_cache,
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
        """Keep a stable path until the requested folder's contents change."""
        # A trailing slash must work on a fresh cache, before parent-cache reuse is possible.
        p1 = download_git_folder(self.remote_dir, f"{self.asset_rel}/", cache_dir=self.cache_dir, ref="main")
        self.assertTrue(p1.exists())
        self.assertEqual((p1 / "foo.txt").read_text(encoding="utf-8"), "v1\n")
        for folder_path in (self.asset_rel, f"./{self.asset_rel}", self.asset_rel.replace("/", "\\") + "\\"):
            with self.subTest(folder_path=folder_path):
                cached = download_git_folder(self.remote_dir, folder_path, cache_dir=self.cache_dir, ref="main")
                self.assertEqual(cached, p1)
                self.assertEqual((cached / "foo.txt").read_text(encoding="utf-8"), "v1\n")
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
        ref_path = _ref_path(Path(self.cache_dir), _ref_key(self.remote_dir, self.asset_rel, "main"))
        os.utime(ref_path, (old_mtime, old_mtime))

        # Refresh — should get a NEW directory (different SHA)
        p2 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
        self.assertTrue(p2.exists())
        self.assertEqual((p2 / "foo.txt").read_text(encoding="utf-8"), "v2\n")
        cache_dir_2 = p2
        for _ in range(depth):
            cache_dir_2 = cache_dir_2.parent
        self.assertNotEqual(cache_dir_1, cache_dir_2)

        # Previously returned paths must remain usable when another ref shares the cache.
        self.assertEqual((p1 / "foo.txt").read_text(encoding="utf-8"), "v1\n")

        # Force refresh with same SHA — should return same directory
        p3 = download_git_folder(
            self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main", force_refresh=True
        )
        cache_dir_3 = p3
        for _ in range(depth):
            cache_dir_3 = cache_dir_3.parent
        self.assertEqual(cache_dir_2, cache_dir_3)
        self.assertEqual((p3 / "foo.txt").read_text(encoding="utf-8"), "v2\n")

        if sys.platform == "win32":
            ref_path = _ref_path(Path(self.cache_dir), _ref_key(self.remote_dir, self.asset_rel, "main"))
            temporary = _temp_cache_path(ref_path)
            # Windows readers prevent replacement or deletion until their handles close.
            for locked_path in (ref_path, temporary):
                with self.subTest(locked_path=locked_path.name):
                    temporary.touch()
                    os.utime(ref_path, (old_mtime, old_mtime))
                    expired_mtime = ref_path.stat().st_mtime
                    with locked_path.open("rb"):
                        cached = download_git_folder(self.remote_dir, self.asset_rel, self.cache_dir, ref="main")
                        self.assertEqual(cached, p3)
                        self.assertEqual((cached / "foo.txt").read_text(encoding="utf-8"), "v2\n")
                        self.assertEqual(_read_cached_ref(cache_dir_3, ref_path.name), self.work.head.commit.hexsha)
                        self.assertEqual(ref_path.stat().st_mtime, expired_mtime)
                    temporary.unlink(missing_ok=True)

    def test_downloads_literal_folder_paths(self):
        """Download the repository root and special directory names by branch and commit SHA."""
        folder_names = (".", "-models", "!models")
        for folder in folder_names:
            asset_path = Path(self.work_dir, folder, "asset.txt")
            asset_path.parent.mkdir(exist_ok=True)
            asset_path.write_text(f"{folder}\n", encoding="utf-8")
            self.work.index.add([str(asset_path)])
        commit = self.work.index.commit("Add root asset and directories with special names")
        self.work.git.push("origin", "main")

        with tempfile.TemporaryDirectory(prefix="nwtn_cache_sha_") as sha_cache_dir:
            for ref, cache_dir in (("main", self.cache_dir), (commit.hexsha, sha_cache_dir)):
                for folder in folder_names:
                    with self.subTest(ref=ref, folder=folder):
                        path = download_git_folder(self.remote_dir, folder, cache_dir, ref=ref)
                        self.assertEqual((path / "asset.txt").read_text(encoding="utf-8"), f"{folder}\n")
                        if folder == ".":
                            self.assertEqual(path.parent, Path(cache_dir))
                            self.assertTrue(path.name.endswith(f"_{commit.tree.hexsha}"))
                            self.assertEqual(download_git_folder(self.remote_dir, "./", cache_dir, ref=ref), path)
                        else:
                            self.assertFalse((path.parent / "assets").exists())

    def test_reuse_identical_content_across_refs(self):
        """Reuse unchanged assets across commits, branches, and tags."""
        first_commit = self.work.head.commit.hexsha
        first = download_git_folder(self.remote_dir, f"./{self.asset_rel}", self.cache_dir, ref=first_commit)
        first_root = first.parents[len(Path(self.asset_rel).parts) - 1]
        tree_hash = (self.work.head.commit.tree / self.asset_rel).hexsha
        self.assertTrue(first_root.name.endswith(f"_{tree_hash}"))

        # A commit outside the requested folder must not duplicate the asset.
        (Path(self.work_dir) / "README.md").write_text("unrelated change\n", encoding="utf-8")
        self.work.index.add(["README.md"])
        self.work.index.commit("Update unrelated file")
        self.work.git.push("origin", "main")
        second_commit = self.work.head.commit.hexsha
        self.work.create_tag("same-assets", message="Same asset contents")
        self.work.git.push("origin", "same-assets")

        for ref in (second_commit, "main", "same-assets"):
            with self.subTest(ref=ref):
                path = download_git_folder(self.remote_dir, self.asset_rel, self.cache_dir, ref=ref)
                self.assertEqual(path, first)
                self.assertEqual((path / "foo.txt").read_text(encoding="utf-8"), "v1\n")

        self.assertEqual(set(Path(self.cache_dir).iterdir()), {first_root, Path(self.cache_dir) / ".newton-refs"})

        # Ref resolutions stay independent even when they share a content directory.
        (Path(self.work_dir, self.asset_rel) / "foo.txt").write_text("v2\n", encoding="utf-8")
        self.work.index.add([str(Path(self.work_dir, self.asset_rel) / "foo.txt")])
        self.work.index.commit("Change asset contents")
        self.work.git.push("origin", "main")
        changed = download_git_folder(self.remote_dir, self.asset_rel, self.cache_dir, ref="main", force_refresh=True)
        self.assertNotEqual(changed, first)
        self.assertEqual((changed / "foo.txt").read_text(encoding="utf-8"), "v2\n")
        with mock.patch("newton._src.utils.download_assets._get_latest_commit_via_git", return_value=None):
            for ref in (first_commit, second_commit, "same-assets"):
                with self.subTest(offline_ref=ref):
                    old = download_git_folder(
                        self.remote_dir, self.asset_rel, self.cache_dir, ref=ref, force_refresh=True
                    )
                    self.assertEqual(old, first)
                    self.assertEqual((old / "foo.txt").read_text(encoding="utf-8"), "v1\n")

        # Ref metadata can be unavailable even when the requested content is ready.
        metadata_dir = _ref_path(Path(self.cache_dir), _ref_key(self.remote_dir, self.asset_rel, first_commit)).parent
        metadata_dir.rename(metadata_dir.with_name("newton-refs-backup"))
        metadata_dir.write_text("blocked", encoding="ascii")
        reused = download_git_folder(self.remote_dir, self.asset_rel, self.cache_dir, ref=second_commit)
        self.assertEqual(reused, first)
        self.assertEqual((reused / "foo.txt").read_text(encoding="utf-8"), "v1\n")
        self.assertEqual(metadata_dir.read_text(encoding="ascii"), "blocked")

    def test_distinct_repository_folder_identities(self):
        """Keep identical trees from distinct repository and folder pairs independently usable."""
        for folder_path in ("baz", "bar_baz"):
            target = Path(self.work_dir, folder_path)
            shutil.copytree(Path(self.work_dir, self.asset_rel), target)
            self.work.index.add([str(target / "foo.txt")])
        self.work.index.commit("Add identical assets at different paths")
        tree_hash = (self.work.head.commit.tree / "baz").hexsha
        self.assertEqual(tree_hash, (self.work.head.commit.tree / "bar_baz").hexsha)

        downloads = Path(self.cache_dir) / "downloads"
        roots = set()
        paths = []
        # The first two requests used the same readable prefix; the third shares a basename.
        for repo_path, folder_path in (("foo_bar.git", "baz"), ("foo.git", "bar_baz"), ("other/foo_bar.git", "baz")):
            with self.subTest(repo_path=repo_path, folder_path=folder_path):
                remote = Path(self.cache_dir) / "remotes" / repo_path
                with git.Repo.clone_from(self.work_dir, remote, bare=True):
                    pass
                path = download_git_folder(str(remote), folder_path, downloads, ref="main")
                self.assertEqual((path / "foo.txt").read_text(encoding="utf-8"), "v1\n")
                self.assertEqual(path.name, folder_path)
                self.assertTrue(path.parent.name.endswith(f"_{tree_hash}"))
                self.assertNotIn(path.parent, roots)
                roots.add(path.parent)
                paths.append(path)

        self.assertEqual(set(downloads.iterdir()), roots | {downloads / ".newton-refs"})
        for path in paths:
            self.assertEqual((path / "foo.txt").read_text(encoding="utf-8"), "v1\n")

    def test_global_asset_cache_directory(self):
        """Use the configured asset directory and preserve explicit overrides."""
        configured = Path(self.cache_dir) / "configured"
        with (
            mock.patch.object(config, "cache_dir_assets", configured),
            mock.patch.dict(os.environ, {"NEWTON_CACHE_PATH": self.cache_dir}),
            mock.patch("newton._src.utils.download_assets.NEWTON_ASSETS_URL", self.remote_dir),
        ):
            first = newton.utils.download_asset(self.asset_rel, ref="main")
            second = newton.utils.download_asset(self.asset_rel, cache_dir=None, ref="main")
            self.assertTrue(first.is_relative_to(configured))
            self.assertEqual(first, second)
            config.cache_dir_assets = None
            explicit = newton.utils.download_asset(self.asset_rel, cache_dir=configured, ref="main")
            self.assertEqual(explicit, first)
            config.cache_dir_assets = Path(self.cache_dir) / "relocated"
            relocated = newton.utils.download_asset(self.asset_rel, ref="main")
            self.assertTrue(relocated.is_relative_to(config.cache_dir_assets))

    def test_parent_cache_refresh(self):
        """Reuse a parent cache while resolving updates against its recorded commit."""
        parent = download_git_folder(self.remote_dir, "assets", self.cache_dir, ref="main")
        parent_root = parent.parent
        parent_ref = _ref_path(Path(self.cache_dir), _ref_key(self.remote_dir, "assets", "main"))
        old_mtime = time.time() - 7200

        for expired in (False, True):
            with self.subTest(expired=expired):
                if expired:
                    os.utime(parent_ref, (old_mtime, old_mtime))
                child = download_git_folder(self.remote_dir, self.asset_rel, self.cache_dir, ref="main")
                self.assertEqual(child, parent / "x")
                self.assertEqual((child / "foo.txt").read_text(encoding="utf-8"), "v1\n")
                self.assertEqual(set(Path(self.cache_dir).iterdir()), {parent_root, parent_ref.parent})

        (Path(self.work_dir, self.asset_rel) / "foo.txt").write_text("v2\n", encoding="utf-8")
        self.work.index.add([str(Path(self.work_dir, self.asset_rel) / "foo.txt")])
        self.work.index.commit("Update nested asset")
        self.work.git.push("origin", "main")
        os.utime(parent_ref, (old_mtime, old_mtime))
        child = download_git_folder(self.remote_dir, self.asset_rel, self.cache_dir, ref="main")
        self.assertNotEqual(child, parent / "x")
        self.assertEqual((child / "foo.txt").read_text(encoding="utf-8"), "v2\n")
        self.assertEqual((parent / "x" / "foo.txt").read_text(encoding="utf-8"), "v1\n")

    def test_missing_or_non_directory_asset(self):
        """Reject missing paths and files without retaining incomplete downloads."""
        for folder_path in ("missing", "assets/x/foo.txt"):
            with self.subTest(folder_path=folder_path):
                with self.assertRaises(RuntimeError):
                    download_git_folder(self.remote_dir, folder_path, self.cache_dir, ref="main")
                self.assertEqual(list(Path(self.cache_dir).iterdir()), [])

    def test_disabled_asset_cache_downloads_fresh(self):
        """Return independent usable downloads when persistent asset caching is disabled."""
        with (
            mock.patch.object(config, "cache_dir_assets", None),
            mock.patch.dict(os.environ, {"NEWTON_CACHE_PATH": self.cache_dir}),
            mock.patch("newton._src.utils.download_assets.NEWTON_ASSETS_URL", self.remote_dir),
        ):
            first = newton.utils.download_asset(self.asset_rel, ref="main")
            second = newton.utils.download_asset(self.asset_rel, ref="main")
            self.assertNotEqual(first, second)
            self.assertEqual((first / "foo.txt").read_text(encoding="utf-8"), "v1\n")
            self.assertEqual((second / "foo.txt").read_text(encoding="utf-8"), "v1\n")
            self.assertEqual(list(Path(self.cache_dir).iterdir()), [])

    def test_uncached_download_cleanup(self):
        """Ignore cleanup by foreign processes and remove assets when their owner exits."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent("""\
                    import sys
                    import newton
                    from newton._src.utils.download_assets import (
                        _cleanup_temporary_cache, clear_git_cache, download_git_folder,
                    )

                    newton.config.cache_dir_assets = None
                    path = download_git_folder(sys.argv[1], sys.argv[2])
                    clear_git_cache()
                    assert (path / "foo.txt").read_text() == "v1\\n"
                    # PID 0 cannot own this download. Avoid forking Warp's native threads.
                    _cleanup_temporary_cache(str(path), owner_pid=0)
                    assert (path / "foo.txt").read_text() == "v1\\n"
                    print(path)
                """),
                self.remote_dir,
                self.asset_rel,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        path = Path(result.stdout.strip().splitlines()[-1])
        self.assertFalse(path.exists())

    def test_failed_uncached_download_cleanup(self):
        """Remove temporary asset files immediately when an uncached download fails."""
        temporary = Path(self.cache_dir) / "temporary"
        temporary.mkdir()
        with (
            mock.patch.object(config, "cache_dir_assets", None),
            mock.patch("newton._src.utils.download_assets.tempfile.mkdtemp", return_value=str(temporary)),
            mock.patch("newton._src.utils.download_assets._get_latest_commit_via_git", return_value=None),
            self.assertRaises(RuntimeError),
        ):
            download_git_folder(self.remote_dir, self.asset_rel)
        self.assertFalse(temporary.exists())

    def test_clear_configured_asset_cache(self):
        """Clear the configured asset directory while respecting the disabled setting."""
        configured = Path(self.cache_dir) / "configured"
        configured.mkdir()
        with mock.patch.object(config, "cache_dir_assets", configured):
            clear_git_cache()
            self.assertFalse(configured.exists())
            configured.mkdir()
            config.cache_dir_assets = None
            clear_git_cache()
            self.assertTrue(configured.exists())

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
        base_prefix = _cache_prefix(self.remote_dir, self.asset_rel)
        entries = [
            e
            for e in Path(self.cache_dir).iterdir()
            if e.is_dir()
            and e.name.startswith(f"{base_prefix}_")
            and not _TEMP_DIR_RE.search(e.name[len(base_prefix) :])
        ]
        self.assertEqual(len(entries), 1, f"Expected 1 cache dir, got {len(entries)}: {entries}")
        ref_path = _ref_path(Path(self.cache_dir), _ref_key(self.remote_dir, self.asset_rel, "main"))
        self.assertEqual(_read_cached_ref(entries[0], ref_path.name), self.work.head.commit.hexsha)
        self.assertEqual(list(ref_path.parent.iterdir()), [ref_path])

    def test_within_ttl_skips_network(self):
        """Within TTL, return cached path without calling git ls-remote."""
        p1 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
        self.assertTrue(p1.exists())

        with mock.patch("newton._src.utils.download_assets._get_latest_commit_via_git") as mock_ls:
            p2 = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")
            mock_ls.assert_not_called()
        self.assertEqual(p1, p2)

    def test_offline_returns_cached(self):
        """Throttle offline retries for direct and parent cache hits without changing content."""
        parent = download_git_folder(self.remote_dir, "assets", cache_dir=self.cache_dir, ref="main")
        ref_path = _ref_path(Path(self.cache_dir), _ref_key(self.remote_dir, "assets", "main"))
        cached_commit = ref_path.read_text(encoding="ascii")
        old_mtime = time.time() - 7200

        # Make the local remote unavailable so the real Git lookup fails immediately.
        remote_head = Path(self.remote_dir) / "HEAD"
        offline_head = remote_head.with_name("HEAD.offline")
        remote_head.rename(offline_head)
        try:
            for folder_path in ("assets", self.asset_rel):
                with self.subTest(folder_path=folder_path):
                    os.utime(ref_path, (old_mtime, old_mtime))
                    expected = parent.parent / folder_path
                    with mock.patch(
                        "newton._src.utils.download_assets._get_latest_commit_via_git",
                        wraps=_get_latest_commit_via_git,
                    ) as resolve:
                        for _ in range(2):
                            cached = download_git_folder(self.remote_dir, folder_path, self.cache_dir, ref="main")
                            self.assertEqual(cached, expected)
                            self.assertEqual((parent / "x" / "foo.txt").read_text(encoding="utf-8"), "v1\n")
                        resolve.assert_called_once_with(self.remote_dir, "main")
                        self.assertGreater(ref_path.stat().st_mtime, old_mtime)
                        os.utime(ref_path, (old_mtime, old_mtime))
                        self.assertEqual(
                            download_git_folder(self.remote_dir, folder_path, self.cache_dir, ref="main"), expected
                        )
                        self.assertEqual(resolve.call_count, 2)
                    self.assertGreater(ref_path.stat().st_mtime, old_mtime)
                    self.assertEqual(ref_path.read_text(encoding="ascii"), cached_commit)

            # Explicit refresh must still check the unavailable remote within the TTL.
            with mock.patch(
                "newton._src.utils.download_assets._get_latest_commit_via_git", wraps=_get_latest_commit_via_git
            ) as resolve:
                refreshed = download_git_folder(
                    self.remote_dir, "assets", self.cache_dir, ref="main", force_refresh=True
                )
                self.assertEqual(refreshed, parent)
                resolve.assert_called_once_with(self.remote_dir, "main")
        finally:
            offline_head.rename(remote_head)

    def test_offline_check_does_not_reorder_versions(self):
        """Keep a successful refresh current when older direct and parent checks fail later."""
        parent = download_git_folder(self.remote_dir, "assets", self.cache_dir, ref="main")
        key = _ref_key(self.remote_dir, "assets", "main")
        old_mtime = time.time() - 7200
        os.utime(_ref_path(Path(self.cache_dir), key), (old_mtime, old_mtime))
        asset = Path(self.work_dir, self.asset_rel, "foo.txt")
        asset.write_text("v2\n", encoding="utf-8")
        self.work.index.add([str(asset)])
        self.work.index.commit("Update asset while checks are pending")
        self.work.git.push("origin", "main")

        # Both workers select the old parent before the main thread publishes the new one.
        started = threading.Barrier(3)
        resume = threading.Event()
        main_thread = threading.get_ident()

        def resolve(url, ref):
            if threading.get_ident() != main_thread:
                started.wait(timeout=30)
                self.assertTrue(resume.wait(timeout=30))
            return _get_latest_commit_via_git(url, ref)

        remote_head = Path(self.remote_dir) / "HEAD"
        offline_head = remote_head.with_name("HEAD.offline")
        with (
            mock.patch("newton._src.utils.download_assets._get_latest_commit_via_git", side_effect=resolve),
            concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor,
        ):
            futures = [
                executor.submit(download_git_folder, self.remote_dir, folder, self.cache_dir, "main")
                for folder in ("assets", self.asset_rel)
            ]
            try:
                started.wait(timeout=30)
                updated = download_git_folder(self.remote_dir, "assets", self.cache_dir, ref="main", force_refresh=True)
                self.assertEqual((updated / "x" / "foo.txt").read_text(encoding="utf-8"), "v2\n")
                # Ensure a late failed check gets a strictly later mtime even on coarse filesystems.
                published_mtime = time.time() - 10
                os.utime(_ref_path(Path(self.cache_dir), key), (published_mtime, published_mtime))
                remote_head.rename(offline_head)
            finally:
                resume.set()
            try:
                for future, expected in zip(futures, (parent, parent / "x"), strict=True):
                    self.assertEqual(future.result(timeout=30), expected)
            finally:
                offline_head.rename(remote_head)

        for folder, expected in (("assets", updated), (self.asset_rel, updated / "x")):
            with self.subTest(folder=folder):
                cached = download_git_folder(self.remote_dir, folder, self.cache_dir, ref="main")
                self.assertEqual(cached, expected)
        self.assertEqual((parent / "x" / "foo.txt").read_text(encoding="utf-8"), "v1\n")

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

    def test_transient_fetch_error_retries_from_clean_temp_dir(self):
        """Retry a transient fetch from a clean temporary directory."""
        ref = self.work.head.commit.hexsha
        real_init = git.Repo.init
        for stderr in (
            b"error: RPC failed; HTTP 503\nfatal: expected 'packfile'",
            b"fatal: unable to access 'https://github.com/example/assets.git/': Could not resolve host: github.com",
            b"fatal: unable to access 'https://github.com/example/assets.git/': Failed to connect to github.com port 443",
            b"fatal: unable to access 'https://github.com/example/assets.git/': Connection timed out",
        ):
            with self.subTest(stderr=stderr):
                clear_git_cache(self.cache_dir)
                attempts = 0
                incomplete_repo = mock.Mock()
                incomplete_repo.git.fetch.side_effect = git.exc.GitCommandError("git fetch", 128, stderr=stderr)

                def flaky_init(path, *args, incomplete_repo=incomplete_repo, **kwargs):
                    nonlocal attempts
                    attempts += 1
                    path = Path(path)
                    if attempts == 1:
                        path.mkdir(parents=True)
                        (path / "partial").write_text("incomplete", encoding="utf-8")
                        return incomplete_repo
                    self.assertFalse(path.exists())
                    return real_init(path, *args, **kwargs)

                with (
                    mock.patch("git.Repo.init", side_effect=flaky_init),
                    mock.patch("newton._src.utils.download_assets.time.sleep") as mock_sleep,
                ):
                    path = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref=ref)

                self.assertEqual((path / "foo.txt").read_text(encoding="utf-8"), "v1\n")
                self.assertEqual(attempts, 2)
                incomplete_repo.git.fetch.assert_called_once_with("origin", ref, "--depth=1", "--filter=blob:none")
                incomplete_repo.close.assert_called_once()
                mock_sleep.assert_called_once_with(1.0)

    def test_transient_clone_error_retries_from_clean_temp_dir(self):
        """Retry a transient clone from a clean temporary directory."""
        real_clone = git.Repo.clone_from
        attempts = 0

        def flaky_clone(url, path, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            path = Path(path)
            if attempts == 1:
                path.mkdir(parents=True)
                (path / "partial").write_text("incomplete", encoding="utf-8")
                raise git.exc.GitCommandError(
                    "git clone",
                    128,
                    stderr="error: RPC failed; HTTP 502\nfatal: early EOF",
                )
            self.assertFalse(path.exists())
            return real_clone(url, path, *args, **kwargs)

        with (
            mock.patch("git.Repo.clone_from", side_effect=flaky_clone),
            mock.patch("newton._src.utils.download_assets.time.sleep") as mock_sleep,
        ):
            path = download_git_folder(self.remote_dir, self.asset_rel, cache_dir=self.cache_dir, ref="main")

        self.assertEqual((path / "foo.txt").read_text(encoding="utf-8"), "v1\n")
        self.assertEqual(attempts, 2)
        mock_sleep.assert_called_once_with(1.0)

    def test_transient_fetch_error_stops_after_three_attempts(self):
        """Stop retrying a transient fetch after three attempts."""
        error = git.exc.GitCommandError(
            "git fetch",
            128,
            stderr="error: RPC failed; HTTP 503\nfatal: expected 'packfile'",
        )

        with (
            mock.patch("git.Repo.init", side_effect=error) as mock_init,
            mock.patch("newton._src.utils.download_assets.time.sleep") as mock_sleep,
            self.assertRaises(RuntimeError),
        ):
            download_git_folder(
                self.remote_dir,
                self.asset_rel,
                cache_dir=self.cache_dir,
                ref=self.work.head.commit.hexsha,
            )

        self.assertEqual(mock_init.call_count, 3)
        self.assertEqual(mock_sleep.call_args_list, [mock.call(1.0), mock.call(2.0)])

    def test_permanent_fetch_error_does_not_retry(self):
        """Do not retry a permanent fetch error."""
        error = git.exc.GitCommandError(
            "git fetch",
            128,
            stderr="error: RPC failed; HTTP 403\nfatal: Authentication failed",
        )

        with (
            mock.patch("git.Repo.init", side_effect=error) as mock_init,
            mock.patch("newton._src.utils.download_assets.time.sleep") as mock_sleep,
            self.assertRaises(RuntimeError),
        ):
            download_git_folder(
                self.remote_dir,
                self.asset_rel,
                cache_dir=self.cache_dir,
                ref=self.work.head.commit.hexsha,
            )

        mock_init.assert_called_once()
        mock_sleep.assert_not_called()


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
        self.base_prefix = "newton-assets_model"
        self.ref_key = _ref_key("http://example.git", "model", "main")

    def tearDown(self):
        _safe_rmtree(self.base)

    @unittest.skipUnless(sys.platform == "win32", "Windows file-sharing semantics required")
    def test_ref_write_retries_file_contention(self):
        """Replace ref metadata after a Windows reader releases the file."""
        directory = self.cache_path / f"{self.base_prefix}_{'a' * 40}"
        directory.mkdir()
        _write_cached_ref(directory, self.ref_key, "b" * 40)
        path = _ref_path(self.cache_path, self.ref_key)
        old_mtime = time.time() - 7200
        os.utime(path, (old_mtime, old_mtime))

        with path.open("rb") as reader:
            # Release the real OS lock on the retry without relying on thread timing.
            with mock.patch("newton._src.utils.download_assets.time.sleep", side_effect=lambda _: reader.close()):
                _write_cached_ref(directory, self.ref_key, "c" * 40)

        self.assertEqual(_read_cached_ref(directory, self.ref_key), "c" * 40)
        self.assertGreater(path.stat().st_mtime, old_mtime)
        self.assertEqual(_find_cached_version(self.cache_path, self.base_prefix, self.ref_key), directory)
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_returns_none_when_no_match(self):
        """Return no match when this request has not been cached."""
        result = _find_cached_version(self.cache_path, self.base_prefix, self.ref_key)
        self.assertIsNone(result)

    def test_finds_single_cached_dir(self):
        """Find a directory with a recorded resolution of the requested ref."""
        d = self.cache_path / f"{self.base_prefix}_{'a' * 40}"
        d.mkdir()
        _write_cached_ref(d, self.ref_key, "b" * 40)
        result = _find_cached_version(self.cache_path, self.base_prefix, self.ref_key)
        self.assertEqual(result, d)
        other_key = _ref_key("http://example.git", "model", "other-branch")
        self.assertIsNone(_find_cached_version(self.cache_path, self.base_prefix, other_key))
        for invalid in (
            b"\xffinvalid",
            b"missing directory",
            f"{d.name}\ninvalid\n".encode(),
            b"../outside\n" + b"b" * 40,
        ):
            with self.subTest(invalid=invalid):
                _ref_path(self.cache_path, self.ref_key).write_bytes(invalid)
                self.assertIsNone(_find_cached_version(self.cache_path, self.base_prefix, self.ref_key))

    def test_excludes_temp_dirs(self):
        """Exclude temporary directories even when they contain ref metadata."""
        temp = self.cache_path / f"{self.base_prefix}_{'a' * 40}_p1234_t5678"
        temp.mkdir()
        _write_cached_ref(temp, self.ref_key, "b" * 40)
        result = _find_cached_version(self.cache_path, self.base_prefix, self.ref_key)
        self.assertIsNone(result)

    def test_tracks_current_version_with_equal_mtimes(self):
        """Select the last published resolution even when filesystem timestamps tie."""
        old = self.cache_path / f"{self.base_prefix}_{'a' * 40}"
        new = self.cache_path / f"{self.base_prefix}_{'b' * 40}"
        old.mkdir()
        new.mkdir()
        old, new = list(self.cache_path.iterdir())
        tied_mtime = int(time.time())
        for directory, commit in ((old, "c" * 40), (new, "d" * 40)):
            _write_cached_ref(directory, self.ref_key, commit)
            os.utime(_ref_path(self.cache_path, self.ref_key), (tied_mtime, tied_mtime))
        other_key = _ref_key("http://example.git", "model", "other-branch")
        _write_cached_ref(old, other_key, "c" * 40)
        result = _find_cached_version(self.cache_path, self.base_prefix, self.ref_key)
        self.assertEqual(result, new)
        self.assertEqual(_find_cached_version(self.cache_path, self.base_prefix, other_key), old)
        self.assertIsNone(_read_cached_ref(old, self.ref_key))
        self.assertEqual(_read_cached_ref(new, self.ref_key), "d" * 40)

        # A later resolution can legitimately return to a previously cached version.
        _write_cached_ref(old, self.ref_key, "c" * 40)
        os.utime(_ref_path(self.cache_path, self.ref_key), (tied_mtime, tied_mtime))
        self.assertEqual(_find_cached_version(self.cache_path, self.base_prefix, self.ref_key), old)
        old.rmdir()
        self.assertIsNone(_find_cached_version(self.cache_path, self.base_prefix, self.ref_key))

    def test_ignores_unrelated_dirs(self):
        """Ignore directories belonging to a different repository prefix."""
        unrelated = self.cache_path / f"other-repo_model_{'a' * 40}"
        unrelated.mkdir()
        _write_cached_ref(unrelated, self.ref_key, "b" * 40)
        result = _find_cached_version(self.cache_path, self.base_prefix, self.ref_key)
        self.assertIsNone(result)


class TestFindParentCache(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="nwtn_parent_")
        self.cache_path = Path(self.base)

    def tearDown(self):
        _safe_rmtree(self.base)

    def test_finds_parent_with_tree_hash(self):
        """Find a parent containing the requested subfolder at the same ref."""
        parent_key = _ref_key("http://example.git", "unitree_g1", "main")
        base_prefix = _cache_prefix("http://example.git", "unitree_g1")
        parent_dir = self.cache_path / f"{base_prefix}_{'a' * 40}"
        (parent_dir / "unitree_g1" / "usd").mkdir(parents=True)
        _write_cached_ref(parent_dir, parent_key, "b" * 40)

        result = _find_parent_cache(self.cache_path, "unitree_g1/usd", "main", "http://example.git")
        self.assertIsNotNone(result)
        parent, target, key = result
        self.assertEqual(parent, parent_dir)
        self.assertEqual(key, parent_key)
        self.assertTrue(target.exists())
        self.assertIsNone(_find_parent_cache(self.cache_path, "unitree_g1/usd", "other-branch", "http://example.git"))

    def test_returns_none_when_subfolder_missing(self):
        """Returns None if parent exists but subfolder does not."""
        parent_key = _ref_key("http://example.git", "unitree_g1", "main")
        base_prefix = _cache_prefix("http://example.git", "unitree_g1")
        parent_dir = self.cache_path / f"{base_prefix}_{'a' * 40}"
        parent_dir.mkdir(parents=True)
        _write_cached_ref(parent_dir, parent_key, "b" * 40)

        result = _find_parent_cache(self.cache_path, "unitree_g1/usd", "main", "http://example.git")
        self.assertIsNone(result)

    def test_returns_none_for_single_path(self):
        """Single-segment paths have no parent to check."""
        result = _find_parent_cache(self.cache_path, "unitree_g1", "main", "http://example.git")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
