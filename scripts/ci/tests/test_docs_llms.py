# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from scripts.ci.update_docs_llms import update


class TestDocsRouter(unittest.TestCase):
    def test_published_versions_and_legacy_fallback(self):
        """Route to published indexes and retain HTML for historical versions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory, entrypoint in (("stable", "index.html"), ("latest", "llms.txt"), ("1.7.0", "llms.txt")):
                (root / directory).mkdir()
                (root / directory / entrypoint).write_text("documentation", encoding="utf-8")
            (root / "switcher.json").write_text(
                json.dumps([{"version": "1.7.0"}, {"version": "1.6.0"}]), encoding="utf-8"
            )
            update(root)
            text = (root / "llms.txt").read_text(encoding="utf-8")
            self.assertIn("stable/index.html", text)
            self.assertIn("latest/llms.txt", text)
            self.assertIn("1.7.0/llms.txt", text)
            self.assertNotIn("1.6.0", text)


if __name__ == "__main__":
    unittest.main()
