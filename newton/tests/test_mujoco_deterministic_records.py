# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard the MJWarp deterministic record-bound module allowlist.

``SolverMuJoCo`` assigns its model-derived ``deterministic_max_records`` bound
only to the modules named in ``_MUJOCO_WARP_DYNAMIC_RECORD_MODULES``. Warp's
code generator can prove a per-thread record count only for statically bounded
loops, so a kernel that runs an atomic inside a data-dependent loop overflows
the deterministic scatter buffer when its module is left at the default bound
of 0.

These tests re-derive the set of MJWarp modules that need the bound directly
from the installed MJWarp sources, so a new (or newly dynamic) kernel that is
not covered by the allowlist fails here instead of at simulation time.
"""

import ast
import glob
import os
import unittest

from newton._src.solvers.mujoco.solver_mujoco import _MUJOCO_WARP_DYNAMIC_RECORD_MODULES

ATOMIC_FUNCTIONS = frozenset({"atomic_add", "atomic_sub", "atomic_min", "atomic_max"})


def _has_static_trip_count(loop: ast.For | ast.While) -> bool:
    """Return True when Warp can prove the loop's trip count at build time."""
    if isinstance(loop, ast.While):
        return False
    iterator = loop.iter
    if isinstance(iterator, ast.Call) and getattr(iterator.func, "id", "") == "range":
        return all(isinstance(argument, ast.Constant) for argument in iterator.args)
    return False


def _dynamic_atomic_kernels(tree: ast.AST) -> set[str]:
    """Return functions performing an atomic inside a data-dependent loop."""
    found: set[str] = set()

    def visit(node: ast.AST, loops: list[ast.For | ast.While], function: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                visit(child, [], child.name)
                continue
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in ATOMIC_FUNCTIONS
                and any(not _has_static_trip_count(loop) for loop in loops)
            ):
                found.add(function)
            visit(child, [*loops, child] if isinstance(child, ast.For | ast.While) else loops, function)

    visit(tree, [], "<module>")
    return found


def _modules_needing_record_bound() -> dict[str, set[str]]:
    """Map each MJWarp module to its kernels with data-dependent atomic counts."""
    import mujoco_warp

    source_dir = os.path.join(os.path.dirname(mujoco_warp.__file__), "_src")
    modules: dict[str, set[str]] = {}
    for path in sorted(glob.glob(os.path.join(source_dir, "*.py"))):
        name = os.path.basename(path)
        if name.endswith("_test.py"):
            continue
        with open(path, encoding="utf-8", errors="replace") as source_file:
            try:
                tree = ast.parse(source_file.read())
            except SyntaxError:
                continue
        kernels = _dynamic_atomic_kernels(tree)
        if kernels:
            modules[f"mujoco_warp._src.{name[:-3]}"] = kernels
    return modules


class TestMuJoCoDeterministicRecordModules(unittest.TestCase):
    def setUp(self):
        try:
            import mujoco_warp  # noqa: F401
        except ImportError:
            self.skipTest("mujoco_warp is not installed")

    def test_allowlist_covers_every_dynamic_atomic_module(self):
        """Every module with a data-dependent atomic must receive the bound."""
        required = _modules_needing_record_bound()
        self.assertTrue(required, "no MJWarp modules with dynamic atomics were found; the scan is broken")

        missing = {module: sorted(kernels) for module, kernels in required.items()}
        missing = {
            module: kernels for module, kernels in missing.items() if module not in _MUJOCO_WARP_DYNAMIC_RECORD_MODULES
        }
        self.assertEqual(
            missing,
            {},
            "MJWarp modules run atomics inside data-dependent loops but are left at "
            "deterministic_max_records=0, so their scatter buffers can overflow: "
            f"{sorted(missing)}. Add them to _MUJOCO_WARP_DYNAMIC_RECORD_MODULES.",
        )

    def test_allowlist_has_no_stale_entries(self):
        """Listed modules must still exist and still need the bound."""
        required = _modules_needing_record_bound()
        stale = sorted(set(_MUJOCO_WARP_DYNAMIC_RECORD_MODULES) - set(required))
        self.assertEqual(
            stale,
            [],
            f"allowlisted MJWarp modules no longer need a dynamic record bound: {stale}. "
            "Over-allocating scatter buffers wastes memory proportional to the launch size.",
        )

    def test_solver_module_is_covered(self):
        """Regression test for the _update_gradient_JTCJ_dense overflow."""
        self.assertIn(
            "mujoco_warp._src.solver",
            _MUJOCO_WARP_DYNAMIC_RECORD_MODULES,
            "mujoco_warp._src.solver holds _update_gradient_JTCJ_dense, whose atomics run "
            "in a data-dependent loop; without the bound its scatter buffer overflows.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
