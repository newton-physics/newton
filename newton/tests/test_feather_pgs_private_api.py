# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import unittest

import newton
import newton._src.solvers as solvers
from newton._src.solvers.feather_pgs import SolverFeatherPGS, kernels
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestFeatherPGSPrivateAPI(unittest.TestCase):
    def test_import_stays_in_private_package(self):
        self.assertNotIn("SolverFeatherPGS", solvers.__all__)
        self.assertFalse(hasattr(solvers, "SolverFeatherPGS"))

    def test_constructor_signature_is_single_path(self):
        params = inspect.signature(SolverFeatherPGS).parameters

        removed = {
            "pgs_mode",
            "pgs_kernel",
            "friction_mode",
            "effort_limit_mode",
            "dense_contact_compliance",
            "dense_max_constraints",
            "cholesky_kernel",
            "trisolve_kernel",
            "hinv_jt_kernel",
            "delassus_kernel",
            "delassus_chunk_size",
            "pgs_chunk_size",
            "small_dof_threshold",
            "pgs_debug",
            "enable_joint_velocity_limits",
            "pgs_warmstart",
            "use_parallel_streams",
            "double_buffer",
            "nvtx",
        }
        for name in removed:
            self.assertNotIn(name, params)

        self.assertIn("contact_compliance", params)
        self.assertIn("max_constraints", params)

    def test_removed_kernel_symbols_stay_deleted(self):
        removed = {
            "pgs_solve_loop",
            "pgs_solve_mf_loop",
            "delassus_par_row_col",
            "hinv_jt_par_row",
            "clamp_joint_tau",
            "friction_step_bisection",
            "friction_step_coulomb_newton",
            "solve_coulomb_row",
        }
        for name in removed:
            self.assertFalse(hasattr(kernels, name), name)


class TestFeatherPGSUnsupportedModels(unittest.TestCase):
    pass


class TestFeatherPGSValidation(unittest.TestCase):
    pass


def _build_single_revolute_model(device):
    builder = newton.ModelBuilder(gravity=0.0)
    link = builder.add_link()
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=(0.0, 0.0, 1.0),
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _build_fixed_only_articulation_model(device):
    builder = newton.ModelBuilder(gravity=0.0)
    link = builder.add_link()
    joint = builder.add_joint_fixed(parent=-1, child=link)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def run_kinematic_body_rejected(test: TestFeatherPGSUnsupportedModels, device):
    builder = newton.ModelBuilder()
    builder.add_body(is_kinematic=True, mass=1.0)
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(NotImplementedError, "kinematic bodies"):
        SolverFeatherPGS(model)


def run_zero_dof_articulation_rejected(test: TestFeatherPGSUnsupportedModels, device):
    model = _build_fixed_only_articulation_model(device)

    with test.assertRaisesRegex(NotImplementedError, "zero-DOF articulations"):
        SolverFeatherPGS(model)


def run_constructor_rejects_invalid_values(test: TestFeatherPGSValidation, device):
    model = _build_single_revolute_model(device)

    cases = [
        ({"update_mass_matrix_interval": 0}, "update_mass_matrix_interval"),
        ({"max_constraints": 0}, "max_constraints"),
        ({"max_constraints": 2}, "max_constraints"),
        ({"mf_max_constraints": 0}, "mf_max_constraints"),
    ]
    for kwargs, match in cases:
        with test.subTest(kwargs=kwargs):
            with test.assertRaisesRegex(ValueError, match):
                SolverFeatherPGS(model, **kwargs)


def run_step_rejects_invalid_dt(test: TestFeatherPGSValidation, device):
    model = _build_single_revolute_model(device)
    solver = SolverFeatherPGS(model)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for dt in (0.0, -1.0, float("nan"), float("inf")):
        with test.subTest(dt=dt):
            with test.assertRaisesRegex(ValueError, "finite dt > 0"):
                solver.step(state_0, state_1, None, None, dt)


devices = get_cuda_test_devices(mode="basic")

for device in devices:
    add_function_test(
        TestFeatherPGSUnsupportedModels,
        "test_kinematic_body_rejected",
        run_kinematic_body_rejected,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSUnsupportedModels,
        "test_zero_dof_articulation_rejected",
        run_zero_dof_articulation_rejected,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSValidation,
        "test_constructor_rejects_invalid_values",
        run_constructor_rejects_invalid_values,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSValidation,
        "test_step_rejects_invalid_dt",
        run_step_rejects_invalid_dt,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
