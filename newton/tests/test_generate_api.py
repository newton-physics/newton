# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    from docs import generate_api
except ModuleNotFoundError as exc:
    # The ``docs`` package lives at the repository root and is not included in
    # installed/wheel builds, so these tests only run from a source checkout.
    # Re-raise anything other than a missing top-level ``docs`` package so that
    # genuine import failures (e.g. a broken ``generate_api``) are not masked.
    if exc.name != "docs":
        raise
    generate_api = None


@unittest.skipUnless(generate_api is not None, "requires the docs/ package (source checkout only)")
class TestGenerateApiCopyright(unittest.TestCase):
    def tearDown(self):
        generate_api._COPYRIGHT_LINES.clear()

    def test_copyright_line_preserves_existing_generated_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            api_page = output_dir / "newton_existing.rst"
            existing_line = ".. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers"
            api_page.write_text(
                "\n".join(
                    [
                        existing_line,
                        ".. SPDX-License-Identifier: CC-BY-4.0",
                        "",
                        "newton.existing",
                        "===============",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(generate_api, "OUTPUT_DIR", output_dir):
                generate_api._snapshot_copyright_lines()
            api_page.unlink()

            self.assertEqual(generate_api.copyright_line(api_page), existing_line)

    def test_copyright_line_uses_current_year_for_new_generated_file(self):
        class FakeDateTime:
            @classmethod
            def now(cls):
                return SimpleNamespace(year=2042)

        with tempfile.TemporaryDirectory() as tmp:
            api_page = Path(tmp) / "newton_new.rst"

            with mock.patch.object(generate_api, "datetime", FakeDateTime):
                self.assertEqual(
                    generate_api.copyright_line(api_page),
                    ".. SPDX-FileCopyrightText: Copyright (c) 2042 The Newton Developers",
                )


@unittest.skipUnless(generate_api is not None, "requires the docs/ package (source checkout only)")
class TestGenerateApiDeprecatedSymbols(unittest.TestCase):
    def test_deprecated_symbols_render_without_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with (
                mock.patch.object(generate_api, "OUTPUT_DIR", output_dir),
                mock.patch.object(generate_api, "REPO_ROOT", output_dir.parent),
            ):
                generate_api.write_module_page("newton.geometry", api_toctree_modules=set())

            page = (output_dir / "newton_geometry.rst").read_text(encoding="utf-8")
            self.assertIn("MATCH_BROKEN", page)
            self.assertIn("MATCH_NOT_FOUND", page)
            self.assertIn("Do not rely on this value", page)
            self.assertNotIn("``-1``", page)
            self.assertNotIn("``-2``", page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
