# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import types
import unittest
import warnings
from unittest import mock

from newton._src.solvers.mujoco import solver_mujoco

_MOCK_REQUIREMENTS = (
    "mujoco~=3.8.0 ; extra == 'sim'",
    "mujoco-warp~=3.8.0,>=3.8.0.2 ; extra == 'sim'",
)
_MOCK_METADATA = "\n".join(f"Requires-Dist: {requirement}" for requirement in _MOCK_REQUIREMENTS)


def _mujoco_dependency_specs():
    return {
        package: solver_mujoco._required_specifier(package, _MOCK_REQUIREMENTS) for package in ("mujoco", "mujoco-warp")
    }


class TestMuJoCoVersionCheck(unittest.TestCase):
    def setUp(self):
        mock_dist = types.SimpleNamespace(read_text=lambda name: _MOCK_METADATA)
        patcher = mock.patch.object(solver_mujoco.importlib_metadata, "distribution", return_value=mock_dist)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_warns_when_installed_versions_do_not_satisfy_pyproject(self):
        specs = _mujoco_dependency_specs()
        versions = {
            package: "0.0.0"
            for package, specifier in specs.items()
            if specifier and not solver_mujoco._version_satisfies("0.0.0", specifier)
        }

        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
            with self.assertWarnsRegex(
                RuntimeWarning,
                r"MuJoCo dependency version mismatch.*mujoco==0\.0\.0.*mujoco-warp==0\.0\.0",
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
        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
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
            if specifier and not solver_mujoco._version_satisfies("0.0.0", specifier)
        }
        previous_mujoco = solver_mujoco.SolverMuJoCo._mujoco
        previous_mujoco_warp = solver_mujoco.SolverMuJoCo._mujoco_warp
        previous_versions_checked = solver_mujoco.SolverMuJoCo._versions_checked

        try:
            solver_mujoco.SolverMuJoCo._mujoco = types.SimpleNamespace()
            solver_mujoco.SolverMuJoCo._mujoco_warp = types.SimpleNamespace()
            solver_mujoco.SolverMuJoCo._versions_checked = False

            with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
                with self.assertWarnsRegex(RuntimeWarning, "MuJoCo dependency version mismatch"):
                    solver_mujoco.SolverMuJoCo.import_mujoco()
        finally:
            solver_mujoco.SolverMuJoCo._mujoco = previous_mujoco
            solver_mujoco.SolverMuJoCo._mujoco_warp = previous_mujoco_warp
            solver_mujoco.SolverMuJoCo._versions_checked = previous_versions_checked

    def test_accepts_versions_that_satisfy_pyproject(self):
        versions = {
            package: _matching_version(specifier)
            for package, specifier in _mujoco_dependency_specs().items()
            if specifier
        }

        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

        messages = [str(warning.message) for warning in caught]
        self.assertFalse(any("MuJoCo dependency version mismatch" in message for message in messages))

    def test_required_specifier_returns_none(self):
        cases = {
            "empty requirements": [],
            "package not in requirements": ["warp-lang>=1.0"],
        }
        for name, requirements in cases.items():
            with self.subTest(name):
                self.assertIsNone(solver_mujoco._required_specifier("mujoco", requirements))


def _matching_version(specifier: str) -> str:
    for pattern in (r">=\s*([0-9][^,;]*)", r"~=\s*([0-9][^,;]*)"):
        match = solver_mujoco.re.search(pattern, specifier)
        if match:
            return match.group(1)
    raise ValueError(f"_matching_version cannot derive a satisfying version from specifier {specifier!r}")


if __name__ == "__main__":
    unittest.main()
