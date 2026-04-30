# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

from newton._src.solvers.mujoco import solver_mujoco


def _mujoco_dependency_specs():
    pyproject_text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    specs = {}
    for package in ("mujoco", "mujoco-warp"):
        match = solver_mujoco.re.search(
            rf'^\s*"{solver_mujoco.re.escape(package)}(?=[<>=!~])([^";]+)',
            pyproject_text,
            solver_mujoco.re.MULTILINE,
        )
        if match:
            specs[package] = match.group(1).replace(" ", "")
    return specs


class TestMuJoCoVersionCheck(unittest.TestCase):
    def test_warns_when_installed_versions_do_not_satisfy_pyproject(self):
        specs = _mujoco_dependency_specs()
        versions = {
            package: "0.0.0"
            for package, specifier in specs.items()
            if not solver_mujoco._version_satisfies("0.0.0", specifier)
        }

        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.__getitem__):
            with self.assertWarnsRegex(
                RuntimeWarning,
                "pyproject.toml.*mujoco==0.0.0.*mujoco-warp==0.0.0",
            ):
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

    def test_warns_when_only_mujoco_warp_mismatches_pyproject(self):
        specs = _mujoco_dependency_specs()
        mujoco_warp_bad_version = "0.0.0"
        self.assertFalse(solver_mujoco._version_satisfies(mujoco_warp_bad_version, specs["mujoco-warp"]))

        versions = {"mujoco": _matching_version(specs["mujoco"]), "mujoco-warp": mujoco_warp_bad_version}
        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.__getitem__):
            with self.assertWarnsRegex(RuntimeWarning, f"mujoco-warp=={mujoco_warp_bad_version}"):
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

    def test_import_mujoco_warns_for_cached_mismatched_versions(self):
        specs = _mujoco_dependency_specs()
        versions = {
            package: "0.0.0"
            for package, specifier in specs.items()
            if not solver_mujoco._version_satisfies("0.0.0", specifier)
        }
        previous_mujoco = solver_mujoco.SolverMuJoCo._mujoco
        previous_mujoco_warp = solver_mujoco.SolverMuJoCo._mujoco_warp

        try:
            solver_mujoco.SolverMuJoCo._mujoco = types.SimpleNamespace()
            solver_mujoco.SolverMuJoCo._mujoco_warp = types.SimpleNamespace()

            with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.__getitem__):
                with self.assertWarnsRegex(RuntimeWarning, "MuJoCo dependency version mismatch"):
                    solver_mujoco.SolverMuJoCo.import_mujoco()
        finally:
            solver_mujoco.SolverMuJoCo._mujoco = previous_mujoco
            solver_mujoco.SolverMuJoCo._mujoco_warp = previous_mujoco_warp

    def test_accepts_versions_that_satisfy_pyproject(self):
        versions = {package: _matching_version(specifier) for package, specifier in _mujoco_dependency_specs().items()}

        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.__getitem__):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

        messages = [str(warning.message) for warning in caught]
        self.assertFalse(any("MuJoCo dependency version mismatch" in message for message in messages))


def _matching_version(specifier: str) -> str:
    for pattern in (r">=\s*([0-9][^,;]*)", r"~=\s*([0-9][^,;]*)"):
        match = solver_mujoco.re.search(pattern, specifier)
        if match:
            return match.group(1)
    return "0.0.0"


if __name__ == "__main__":
    unittest.main()
