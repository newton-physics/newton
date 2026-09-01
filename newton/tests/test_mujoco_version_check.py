# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import types
import unittest
import warnings
from unittest import mock

from newton._src.solvers.mujoco import solver_mujoco

_MOCK_REQUIREMENTS = (
    "mujoco~=3.8.0 ; extra == 'sim'",
    "mujoco-warp~=3.8.0,>=3.8.0.3 ; extra == 'sim'",
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


class TestMuJoCoDeterminismConfig(unittest.TestCase):
    def test_only_dynamic_record_modules_receive_model_bound(self):
        solver = object.__new__(solver_mujoco.SolverMuJoCo)
        solver._deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
        solver._deterministic_max_records = 17
        smooth_module = types.SimpleNamespace(__name__="mujoco_warp._src.smooth")
        forward_module = types.SimpleNamespace(__name__="mujoco_warp._src.forward")

        with (
            mock.patch.object(
                solver_mujoco,
                "_mujoco_warp_deterministic_modules",
                return_value=[smooth_module, forward_module],
            ),
            mock.patch.object(solver, "_set_module_options") as set_module_options,
        ):
            solver._set_mujoco_warp_module_options()

        dynamic_options = {
            "deterministic": solver_mujoco.wp.DeterministicMode.RUN_TO_RUN,
            "deterministic_max_records": 17,
        }
        generated_options = {
            "deterministic": solver_mujoco.wp.DeterministicMode.RUN_TO_RUN,
            "deterministic_max_records": 0,
        }
        self.assertEqual(
            set_module_options.call_args_list,
            [
                mock.call(dynamic_options, module=smooth_module),
                mock.call(generated_options, module=forward_module),
                mock.call(generated_options, module=solver_mujoco.kernels),
            ],
        )

    def test_scoped_config_uses_solver_mode_and_restores_globals(self):
        solver = object.__new__(solver_mujoco.SolverMuJoCo)
        solver._deterministic = solver_mujoco.wp.DeterministicMode.NOT_GUARANTEED
        solver._deterministic_max_records = 0

        original_mode = solver_mujoco.wp.config.deterministic
        original_max_records = solver_mujoco.wp.config.deterministic_max_records
        try:
            solver_mujoco.wp.config.deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
            solver_mujoco.wp.config.deterministic_max_records = 17
            with solver._scoped_deterministic_config():
                self.assertEqual(
                    solver_mujoco.wp.config.deterministic,
                    solver_mujoco.wp.DeterministicMode.NOT_GUARANTEED,
                )
                self.assertEqual(solver_mujoco.wp.config.deterministic_max_records, 0)

            self.assertEqual(solver_mujoco.wp.config.deterministic, solver_mujoco.wp.DeterministicMode.RUN_TO_RUN)
            self.assertEqual(solver_mujoco.wp.config.deterministic_max_records, 17)
        finally:
            solver_mujoco.wp.config.deterministic = original_mode
            solver_mujoco.wp.config.deterministic_max_records = original_max_records

    def test_notify_model_changed_uses_solver_config_for_mjwarp(self):
        solver = object.__new__(solver_mujoco.SolverMuJoCo)
        solver.use_mujoco_cpu = False
        solver._deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
        solver._deterministic_max_records = 17
        solver.has_connect_constraints = False
        solver.has_jnt_connect_constraints = False
        observed_options = []

        def observe_options():
            observed_options.append(
                (
                    solver_mujoco.wp.config.deterministic,
                    solver_mujoco.wp.config.deterministic_max_records,
                )
            )

        original_mode = solver_mujoco.wp.config.deterministic
        original_max_records = solver_mujoco.wp.config.deterministic_max_records
        try:
            solver_mujoco.wp.config.deterministic = solver_mujoco.wp.DeterministicMode.NOT_GUARANTEED
            solver_mujoco.wp.config.deterministic_max_records = 0
            with (
                mock.patch.object(solver, "_apply_module_options") as apply_module_options,
                mock.patch.object(solver, "_prepare_generated_kernels") as prepare_generated_kernels,
                mock.patch.object(solver, "_update_model_properties", side_effect=observe_options),
                mock.patch.object(solver, "_invalidate_contact_fast_path"),
            ):
                solver.notify_model_changed(solver_mujoco.ModelFlags.MODEL_PROPERTIES)

            apply_module_options.assert_called_once_with()
            prepare_generated_kernels.assert_called_once_with()
            self.assertEqual(
                observed_options,
                [(solver_mujoco.wp.DeterministicMode.RUN_TO_RUN, 17)],
            )
        finally:
            solver_mujoco.wp.config.deterministic = original_mode
            solver_mujoco.wp.config.deterministic_max_records = original_max_records

    def test_max_records_are_derived_from_model_dimensions(self):
        def make_model(nv, *, nu=0, tendon_num=(), ten_j_rownnz=(), **overrides):
            fields = {
                "nv": nv,
                "nu": nu,
                "nbody": 1,
                "body_weldid": (0,),
                "body_dofadr": (-1,),
                "body_dofnum": (0,),
                "dof_parentid": (),
                "tendon_num": tendon_num,
                "ten_J_rownnz": ten_j_rownnz,
                "flexedge_J_rownnz": (),
            }
            fields.update(overrides)
            return types.SimpleNamespace(**fields)

        independent_free_bodies = make_model(
            18,
            nbody=4,
            body_weldid=(0, 1, 2, 3),
            body_dofadr=(-1, 0, 6, 12),
            body_dofnum=(0, 6, 6, 6),
            dof_parentid=(-1, 0, 1, 2, 3, 4, -1, 6, 7, 8, 9, 10, -1, 12, 13, 14, 15, 16),
        )
        cases = {
            "constraint rows": (make_model(4), 200, 200),
            "sparse actuator Hessian": (make_model(32, nu=1), 64, 528),
            "spatial tendon": (make_model(8, tendon_num=(6,), ten_j_rownnz=(3,)), 2, 30),
            "fixed tendon armature": (make_model(1, tendon_num=(1,), ten_j_rownnz=(1,)), 0, 2),
            "independent body chains": (independent_free_bodies, 50, 78),
        }
        for name, (mj_model, njmax, expected) in cases.items():
            with self.subTest(name=name):
                mjw_data = types.SimpleNamespace(njmax=njmax)
                self.assertEqual(
                    solver_mujoco._mujoco_warp_deterministic_max_records(mj_model, mjw_data),
                    expected,
                )


def _matching_version(specifier: str) -> str:
    for pattern in (r">=\s*([0-9][^,;]*)", r"~=\s*([0-9][^,;]*)"):
        match = solver_mujoco.re.search(pattern, specifier)
        if match:
            return match.group(1)
    raise ValueError(f"_matching_version cannot derive a satisfying version from specifier {specifier!r}")


if __name__ == "__main__":
    unittest.main()
