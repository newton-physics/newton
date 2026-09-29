# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
class TestGenerateApiPublicSymbols(unittest.TestCase):
    def test_public_symbols_rejects_module_without_all(self):
        """Reject a module that does not declare its public API."""
        module = ModuleType("newton.missing_all")
        module.undeclared_symbol = object()

        with self.assertRaisesRegex(ValueError, r"newton\.missing_all must define __all__"):
            generate_api.public_symbols(module)

    def test_public_symbols_rejects_non_string_entry(self):
        """Reject a public API declaration containing a non-string entry."""
        module = ModuleType("newton.invalid_all_entry")
        module.__all__ = ["declared_symbol", 1]

        with self.assertRaisesRegex(
            ValueError,
            r"newton\.invalid_all_entry\.__all__ must contain only strings; got 1",
        ):
            generate_api.public_symbols(module)


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
    @staticmethod
    def _make_module_with_deprecated_symbols(mod_name: str) -> ModuleType:
        """Build a stand-in public module using the same deprecation shim as Newton.

        The module is synthetic so this test does not depend on any real
        deprecation, which would make it fail once that deprecation is removed.
        """
        module = ModuleType(mod_name)
        module.__all__ = ["PUBLIC_VALUE"]
        module.PUBLIC_VALUE = 3

        deprecated_values = {"OLD_VALUE_A": -1, "OLD_VALUE_B": -2}
        module.__deprecated_symbols__ = dict.fromkeys(deprecated_values, "Do not rely on this value.")

        def __getattr__(name: str):
            try:
                value = deprecated_values[name]
            except KeyError:
                raise AttributeError(f"module {mod_name!r} has no attribute {name!r}") from None
            warnings.warn(f"{mod_name}.{name} is deprecated.", DeprecationWarning, stacklevel=2)
            return value

        module.__getattr__ = __getattr__
        return module

    def test_deprecated_symbols_render_without_values(self):
        mod_name = "newton_fake_deprecated_api"
        fake_module = self._make_module_with_deprecated_symbols(mod_name)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with (
                mock.patch.dict(sys.modules, {mod_name: fake_module}),
                mock.patch.object(generate_api, "OUTPUT_DIR", output_dir),
                mock.patch.object(generate_api, "REPO_ROOT", output_dir.parent),
                warnings.catch_warnings(),
            ):
                # Generating docs must not access deprecated symbols.
                warnings.simplefilter("error", DeprecationWarning)
                generate_api.write_module_page(mod_name, api_toctree_modules=set())

            page = (output_dir / f"{mod_name}.rst").read_text(encoding="utf-8")
            self.assertIn("``PUBLIC_VALUE``", page)
            self.assertIn("``3``", page)
            self.assertIn(".. rubric:: Deprecated", page)
            self.assertIn("``OLD_VALUE_A``", page)
            self.assertIn("``OLD_VALUE_B``", page)
            self.assertIn("Do not rely on this value", page)
            self.assertNotIn("``-1``", page)
            self.assertNotIn("``-2``", page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
