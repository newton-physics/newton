# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import pathlib
import unittest
import warnings

import newton
from newton.exceptions import NewtonDeprecationWarning, NewtonWarning

_NEWTON_ROOT = pathlib.Path(newton.__file__).parent


def _library_sources():
    for path in sorted(_NEWTON_ROOT.rglob("*.py")):
        relative = path.relative_to(_NEWTON_ROOT)
        if relative.parts[0] in ("tests", "examples") or "tests" in relative.parts or "examples" in relative.parts:
            continue
        yield relative, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class TestWarningCategories(unittest.TestCase):
    def test_deprecation_category_matches_builtin_filters(self):
        """Keep Newton deprecations filterable as both Newton and Python deprecation warnings."""
        self.assertTrue(issubclass(NewtonWarning, UserWarning))
        self.assertTrue(issubclass(NewtonDeprecationWarning, NewtonWarning))
        self.assertTrue(issubclass(NewtonDeprecationWarning, DeprecationWarning))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", category=NewtonWarning)
            warnings.warn("ignored", NewtonDeprecationWarning, stacklevel=1)
        self.assertEqual(caught, [])

    def test_library_warnings_use_newton_categories(self):
        """Require an explicit, non-generic category for every warning Newton emits."""
        generic = {None, "UserWarning", "DeprecationWarning", "Warning"}
        offenders = []
        for relative, tree in _library_sources():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or ast.unparse(node.func) not in ("warnings.warn", "warn"):
                    continue
                category = node.args[1] if len(node.args) > 1 else None
                category = next((k.value for k in node.keywords if k.arg == "category"), category)
                if (ast.unparse(category) if category is not None else None) in generic:
                    offenders.append(f"{relative}:{node.lineno}")
        self.assertEqual(offenders, [], "use a newton.exceptions warning category")


if __name__ == "__main__":
    unittest.main(verbosity=2)
