# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise the asset browser against real HTTP responses and an offline cache."""

import hashlib
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from newton._src.viewer import viewer_assets


class TestAssetLibrary(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.requests = []
        self.hold = threading.Event()
        self.release = threading.Event()
        self.catalog_failed = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                owner.requests.append(self.path)
                if self.path == "/registry":
                    if owner.catalog_failed:
                        self.send_error(403)
                        return
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write((owner.base + owner.prefix + "com.nvidia.simready.packaging.json\n").encode())
                    return
                if self.path.startswith("/api/datasets/"):
                    entries = [
                        {"type": "file", "path": "sponge/" + name, "size": len(data)}
                        for name, data in owner.files.items()
                    ]
                    if "cursor=next" not in self.path:
                        self.send_response(200)
                        self.send_header(
                            "Link",
                            f'<{owner.base}/api/datasets/test/assets/tree/{owner.revision}/sponge?cursor=next>; rel="next"',
                        )
                        entries = entries[:3]
                    else:
                        self.send_response(200)
                        entries = entries[3:]
                    self.end_headers()
                    self.wfile.write(json.dumps(entries).encode())
                    return
                name = self.path.removeprefix(owner.prefix)
                if name not in owner.files:
                    self.send_error(404)
                    return
                if name == ".metadata/LICENSE.txt" and owner.hold.is_set():
                    owner.release.wait(3)
                self.send_response(200)
                self.send_header("Content-Length", str(len(owner.files[name])))
                self.end_headers()
                self.wfile.write(owner.files[name])

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.revision = "a" * 40
        self.prefix = f"/datasets/test/assets/resolve/{self.revision}/sponge/"
        self.files = {
            "simready_usd/root.usda": b'#usda 1.0\n(defaultPrim = "Sponge")\ndef Xform "Sponge" (references = @payloads/mesh.usda@) {}\n',
            "simready_usd/payloads/mesh.usda": b'#usda 1.0\ndef Cube "Mesh" {}\n',
            "simready_usd/.thumbs/256x256/root.usda.png": b"thumbnail bytes",
            ".metadata/LICENSE.txt": b"CC-BY-4.0 attribution\n",
            ".metadata/com.nvidia.simready.root_usds.json": json.dumps(
                {"entries": ["simready_usd/root.usda"]}
            ).encode(),
        }
        items = [
            {"relative_path": name, "size": len(data), "hash": {"sha256": hashlib.sha256(data).hexdigest()}}
            for name, data in self.files.items()
            if name.startswith("simready_usd/")
        ]
        bom = json.dumps({"format_version": "1.0", "items": items}).encode()
        self.files[".metadata/com.nvidia.simready.packaging.bom.json"] = bom
        self.files["com.nvidia.simready.packaging.json"] = json.dumps(
            {
                "format_version": "1.0",
                "package_id": "simready.test.sponge.1",
                "license": "CC-BY-4.0",
                "metadata": [
                    {
                        "name": "com.nvidia.simready.packaging.bom.json",
                        "hash": {"sha256": hashlib.sha256(bom).hexdigest()},
                    }
                ],
            }
        ).encode()
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.release.set)
        # Patch endpoints only: HTTP, pagination, hashing, publication and cache code remain real.
        self.registry_patch = mock.patch.object(viewer_assets, "_REGISTRY_URL", self.base + "/registry")
        self.hub_patch = mock.patch.object(viewer_assets, "_HUB_URL", self.base)
        self.registry_patch.start()
        self.hub_patch.start()
        self.addCleanup(self.registry_patch.stop)
        self.addCleanup(self.hub_patch.stop)
        self.library = viewer_assets.AssetLibrary(self.directory.name)
        self.addCleanup(self.library.close)

    def until(self, predicate, library=None):
        library = library or self.library
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            library.poll()
            if predicate():
                return
            self.release.wait(0.01) if not self.release.is_set() else time.sleep(0.01)
        self.fail(f"Asset operation did not finish: {library.error}, {library.progress}")

    def record(self):
        self.library.search()
        self.until(lambda: not self.library.loading)
        self.assertIsNone(self.library.error)
        self.assertEqual(len(self.library.records), 1)
        return self.library.records[0]

    def test_catalog_is_lazy_searchable_and_retained_after_refresh_failure(self):
        """Keep a usable catalog when an explicit refresh fails."""
        self.assertFalse(self.requests)
        record = self.record()
        self.assertEqual(record.revision, self.revision)
        self.assertEqual(record.repo_id, "test/assets")
        self.library.search("absent")
        self.until(lambda: not self.library.loading)
        self.assertFalse(self.library.records)
        self.catalog_failed = True
        self.library.search("sponge")
        self.library.refresh()
        self.until(lambda: not self.library.loading)
        self.assertEqual(self.library.records, [record])
        self.assertIn("403", self.library.error)

    def test_package_download_preserves_dependencies_license_and_offline_readiness(self):
        """Publish complete packages and verify their files when reopening offline."""
        record = self.record()
        self.assertIsNone(self.library.thumbnail(record))
        self.until(lambda: self.library.thumbnail(record) is not None)
        self.assertEqual(self.library.thumbnail(record).read_bytes(), b"thumbnail bytes")
        self.hold.set()
        self.library.download(record)
        self.until(lambda: any(path.endswith("LICENSE.txt") for path in self.requests))
        self.assertIsNone(self.library.ready(record))
        self.release.set()
        self.until(lambda: self.library.ready(record) is not None)
        root = self.library.ready(record)
        self.assertEqual(root.read_bytes(), self.files["simready_usd/root.usda"])
        self.assertTrue((root.parent / "payloads/mesh.usda").is_file())
        self.assertEqual(
            (root.parent.parent / ".metadata/LICENSE.txt").read_bytes(), self.files[".metadata/LICENSE.txt"]
        )
        self.assertTrue(any("cursor=next" in path for path in self.requests))
        self.library.close()
        with mock.patch.dict("os.environ", {"HF_HUB_OFFLINE": "1"}):
            offline = viewer_assets.AssetLibrary(self.directory.name)
            self.addCleanup(offline.close)
            before = len(self.requests)
            offline.search()
            self.until(lambda: not offline.loading, offline)
            self.until(lambda: offline.ready(record) is not None, offline)
            self.assertEqual(offline.ready(record), root)
            self.assertEqual(len(self.requests), before)
            offline.close()
            (root.parent / "payloads/mesh.usda").unlink()
            incomplete = viewer_assets.AssetLibrary(self.directory.name)
            self.addCleanup(incomplete.close)
            incomplete.download(record)
            self.until(lambda: not incomplete.loading, incomplete)
            self.assertIsNone(incomplete.ready(record))
            self.assertIn("offline", incomplete.error.lower())

    def test_corrupt_content_never_becomes_ready(self):
        """Reject content whose bytes disagree with the pinned bill of materials."""
        record = self.record()
        self.files["simready_usd/payloads/mesh.usda"] = b"corrupted payload"
        self.library.download(record)
        self.until(lambda: not self.library.loading)
        self.assertIsNone(self.library.ready(record))
        self.assertIn("mesh.usda", self.library.error)
        self.assertIn("mesh.usda", self.library.download_error(record))
        self.assertFalse(list(Path(self.directory.name).rglob("complete.json")))
        self.files["simready_usd/payloads/mesh.usda"] = b'#usda 1.0\ndef Cube "Mesh" {}\n'
        self.library.download(record)
        self.assertIsNone(self.library.download_error(record))
        self.until(lambda: self.library.ready(record) is not None)

    def test_close_cancels_inflight_download_without_waiting(self):
        """Return from close while an HTTP response is still waiting."""
        record = self.record()
        self.hold.set()
        self.library.download(record)
        self.until(lambda: any(path.endswith("LICENSE.txt") for path in self.requests))
        started = time.monotonic()
        self.library.close()
        self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(self.release.is_set())
        self.release.set()
        self.assertFalse(self.library.loading)


if __name__ == "__main__":
    unittest.main()
