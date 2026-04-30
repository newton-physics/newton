# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import unittest


class TestSolverFlags(unittest.TestCase):
    def test_solver_implementations_use_solver_model_flags(self):
        repo_root = Path(__file__).resolve().parents[2]
        solvers_dir = repo_root / "newton" / "_src" / "solvers"
        allowed_paths = {
            solvers_dir / "__init__.py",
            solvers_dir / "flags.py",
        }

        offenders = []
        for path in sorted(solvers_dir.rglob("*.py")):
            if path in allowed_paths:
                continue
            if "SolverNotifyFlags" in path.read_text(encoding="utf-8"):
                offenders.append(path.relative_to(repo_root).as_posix())

        self.assertEqual([], offenders)
