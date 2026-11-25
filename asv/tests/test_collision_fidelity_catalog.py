# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check catalog validation and semantic identities using real temporary files."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(str(Path(__file__).parents[1]))

from collision_fidelity import catalog, version


class TestCollisionFidelityCatalog(unittest.TestCase):
    """Reject ambiguous case records before a scene can be selected."""

    def test_invalid_record_counts_and_duplicates_are_rejected(self):
        """Require both the record count and the unique count to match the header."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for count, records in (
                (1, [{"case_id": "one"}, {"case_id": "one"}]),
                (2, [{"case_id": "one"}]),
                (2, [{"case_id": "one"}, {"case_id": "one"}]),
            ):
                with self.subTest(count=count, records=records):
                    (root / "cases.json").write_text(json.dumps({"case_count": count, "cases": records}))
                    with patch.object(catalog, "__file__", str(root / "catalog.py")):
                        with self.assertRaisesRegex(ValueError, "Duplicate or missing"):
                            catalog.load_cases()


class TestCollisionFidelityVersion(unittest.TestCase):
    """Keep prose out of benchmark versions while preserving executable inputs."""

    def test_asset_documentation_does_not_change_semantic_identity(self):
        """Ignore notice and README edits but detect fixture, metadata and code edits."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "collision_fidelity"
            assets = root / "assets"
            assets.mkdir(parents=True)
            wrapper = root.parent / "benchmarks/simulation/bench_collision_fidelity.py"
            wrapper.parent.mkdir(parents=True)
            wrapper.write_text("value = 1\n")
            code = root / "recipe.py"
            code.write_text("value = 1\n")
            fixture = assets / "initial.npz"
            fixture.write_bytes(b"initial fixture bytes")
            metadata = assets / "initial.json"
            metadata.write_text('{"friction": 1.0}')
            documents = [assets / "README.md", assets / "STYLE3D-LICENSE.txt"]
            for path in documents:
                path.write_text("Original text")

            def digest():
                version._source_and_assets.cache_clear()
                return version._source_and_assets()

            self.addCleanup(version._source_and_assets.cache_clear)
            with patch.object(version, "__file__", str(root / "version.py")):
                original = digest()
                for path in documents:
                    with self.subTest(document=path.name):
                        path.write_text("Corrected prose or attribution")
                        self.assertEqual(digest(), original)
                for path, changed in (
                    (fixture, b"changed fixture bytes"),
                    (metadata, b'{"friction": 0.5}'),
                    (code, b"value = 2\n"),
                    (wrapper, b"value = 2\n"),
                ):
                    with self.subTest(runtime_input=path.name):
                        previous = path.read_bytes()
                        path.write_bytes(changed)
                        self.assertNotEqual(digest(), original)
                        path.write_bytes(previous)


if __name__ == "__main__":
    unittest.main()
