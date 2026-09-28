# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FPGS articulation row-scaling benchmark smoke tests.

Canonical unittest module (discovered by the ``test*.py`` unittest loader used
by ``newton.tests``): the original pytest-style module-level functions were
invisible to the harness and contributed zero tests to the reported JUnit
count. GPU-dependent cases skip cleanly when CUDA is unavailable.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"


def _load_row_scaling_module():
    script = BENCHMARKS_DIR / "fpgs_articulation_row_scaling.py"
    spec = importlib.util.spec_from_file_location("fpgs_articulation_row_scaling", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_cuda(reason: str):
    wp.init()
    if not wp.get_device("cuda:0").is_cuda:
        raise unittest.SkipTest(reason)


def _build_d6_articulated_free_contact_model(device: str):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.JointDofConfig

    link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(link, hx=0.12, hy=0.08, hz=0.06)
    joint = builder.add_joint_d6(
        parent=-1,
        child=link,
        linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y), cfg(axis=newton.Axis.Z)],
        angular_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y), cfg(axis=newton.Axis.Z)],
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform_identity(),
    )
    builder.add_articulation([joint])

    cube = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.588), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(cube, hx=0.05, hy=0.05, hz=0.04)
    return builder.finalize(device=device)


def _step_d6_contact_case(model, contacts, mode: str) -> tuple[np.ndarray, tuple[int, int, int], int | None]:
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
    state_in.clear_forces()
    state_out.clear_forces()

    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_contact_response=mode,
        pgs_iterations=2,
        pgs_velocity_iterations=0,
        enable_contact_friction=True,
        dense_max_constraints=32,
        mf_max_constraints=32,
        pgs_warmstart=False,
        mf_warmstart=False,
    )
    solver.step(state_in, state_out, control, contacts, 1.0 / 120.0)
    wp.synchronize()

    propagation_count = getattr(solver, "propagation_constraint_count", None)
    rows = (
        int(np.sum(solver.constraint_count.numpy())),
        int(np.sum(solver.mf_constraint_count.numpy())),
        int(np.sum(propagation_count.numpy())) if propagation_count is not None else 0,
    )
    return state_out.joint_qd.numpy().copy(), rows, getattr(solver, "_propagation_full_fused_size", None)


class TestFpgsArticulationRowScalingBench(unittest.TestCase):
    def test_fpgs_propagation_fused_supports_multi_dof_contact_rows(self):
        _require_cuda("CUDA is required for propagation-fused multi-DOF contact rows")

        model = _build_d6_articulated_free_contact_model("cuda:0")
        self.assertEqual(model.joint_dof_dim.numpy().tolist(), [[3, 3], [3, 3]])

        initial = model.state()
        newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(initial, contacts)
        wp.synchronize()
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

        qd_immediate, rows_immediate, fused_immediate = _step_d6_contact_case(model, contacts, "immediate")
        qd_propagation, rows_propagation, fused_propagation = _step_d6_contact_case(model, contacts, "propagation")
        qd_fused, rows_fused, fused_size = _step_d6_contact_case(model, contacts, "propagation-fused")

        self.assertGreater(rows_immediate[0], 0)
        self.assertEqual(rows_immediate[2], 0)
        self.assertIsNone(fused_immediate)

        self.assertEqual(rows_propagation[0], 0)
        self.assertGreater(rows_propagation[2], 0)
        self.assertIsNone(fused_propagation)

        self.assertEqual(rows_fused[0], 0)
        self.assertEqual(rows_fused[2], rows_propagation[2])
        self.assertEqual(fused_size, 6)

        np.testing.assert_allclose(qd_propagation, qd_immediate, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(qd_fused, qd_immediate, rtol=1.0e-5, atol=1.0e-5)

    def test_fpgs_propagation_mixed_size_free_body_response(self):
        _require_cuda("CUDA is required for propagation articulated/free contact rows")

        module = _load_row_scaling_module()

        case = module.BenchCase(
            "mixed_size_response",
            "L=4,boxes=1",
            "articulated_free",
            articulations=1,
            links=4,
            contacts_per_articulation=1,
        )
        model, initial, contacts = module._prepare_case_run(case, "cuda:0")

        state_in = model.state()
        state_out = model.state()
        module._restore_state(model, state_in, initial)
        module._restore_state(model, state_out, initial)
        state_in.clear_forces()
        state_out.clear_forces()

        solver = module.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
            pgs_iterations=2,
            pgs_velocity_iterations=0,
            enable_contact_friction=True,
            dense_max_constraints=32,
            mf_max_constraints=16,
            pgs_warmstart=False,
            mf_warmstart=False,
        )
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
        wp.synchronize()

        row_count = int(solver.propagation_constraint_count.numpy()[0])
        self.assertGreater(row_count, 0)

        body_to_art = solver.body_to_articulation.numpy()
        is_free = solver.is_free_rigid.numpy()
        body_b = solver.propagation_body_b.numpy()[0, :row_count]
        free_bodies = [int(body) for body in body_b if body >= 0 and is_free[body_to_art[int(body)]] != 0]
        self.assertTrue(free_bodies)

        response = solver.propagation_body_response.numpy()
        mijt_b = solver.propagation_MiJt_b.numpy()[0, :row_count]
        self.assertGreater(max(float(np.linalg.norm(response[body])) for body in free_bodies), 0.0)
        self.assertGreater(float(np.linalg.norm(mijt_b)), 0.0)

    def test_fpgs_articulation_row_scaling_benchmark_smoke(self):
        _require_cuda("CUDA is required for the FPGS production scaling benchmark")

        script = BENCHMARKS_DIR / "fpgs_articulation_row_scaling.py"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "fpgs_articulation_row_scaling"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--preset",
                    "smoke",
                    "--no-plots",
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
            )

            summary = (out_dir / "summary.md").read_text(encoding="utf-8")
            self.assertTrue((out_dir / "results.csv").is_file())
            results = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
        paths = {row["path"] for row in results}
        self.assertEqual(paths, {"mf_immediate", "propagation", "propagation-fused"})
        self.assertFalse(
            paths
            & {
                "new_free_fused",
                "new_articulated_fused",
                "mf_local_plus_propagation",
            }
        )
        self.assertIn("SolverFeatherPGS.step", summary)
        self.assertNotIn("private generated-kernel", summary)
        self.assertIn("## free_free_dof", summary)
        self.assertIn("## articulated_contact_rows", summary)

        free_rows = [row for row in results if row["case_kind"] == "free_free"]
        articulated_rows = [row for row in results if row["case_kind"] == "articulated_free"]
        self.assertTrue(free_rows)
        self.assertTrue(articulated_rows)

        # Contact-row placement is intrinsic to the response mode (3f282c21):
        # "propagation" routes free/free rows onto the propagation family (the
        # family allocates for all-free-body scenes), while "propagation-fused"
        # keeps free/free rows on the matrix-free family (its measured-best
        # configuration).
        free_propagation = next(row for row in free_rows if row["path"] == "propagation")
        self.assertTrue(free_propagation["propagation_active"])
        self.assertGreater(free_propagation["propagation_extra_mib"], 0.0)
        self.assertEqual(free_propagation["dense_rows_total"], 0)
        self.assertEqual(free_propagation["mf_rows_total"], 0)
        self.assertGreater(free_propagation["propagation_rows_total"], 0)

        free_fused = next(row for row in free_rows if row["path"] == "propagation-fused")
        self.assertEqual(free_fused["dense_rows_total"], 0)
        self.assertGreater(free_fused["mf_rows_total"], 0)
        self.assertEqual(free_fused["propagation_rows_total"], 0)

        articulated_baseline = next(row for row in articulated_rows if row["path"] == "mf_immediate")
        articulated_propagation = next(row for row in articulated_rows if row["path"] == "propagation")
        self.assertGreater(articulated_baseline["dense_rows_total"], 0)
        self.assertEqual(articulated_baseline["propagation_rows_total"], 0)
        self.assertTrue(articulated_propagation["propagation_active"])
        self.assertGreater(articulated_propagation["propagation_extra_mib"], 0.0)
        self.assertEqual(articulated_propagation["dense_rows_total"], 0)
        self.assertGreater(articulated_propagation["propagation_rows_total"], 0)
        if articulated_baseline["joint_dof_count"] >= 64:
            self.assertLess(articulated_propagation["row_solver_mib"], articulated_baseline["row_solver_mib"])

        for row in results:
            self.assertTrue(np.isfinite(row["ms_per_step"]))
            self.assertGreater(row["ms_per_step"], 0.0)
            self.assertTrue(np.isfinite(row["row_solver_mib"]))
            self.assertGreaterEqual(row["row_solver_mib"], 0.0)
            self.assertTrue(np.isfinite(row["propagation_extra_mib"]))
            self.assertGreaterEqual(row["propagation_extra_mib"], 0.0)
            if row["case_kind"] == "free_free":
                self.assertIsNone(row["joint_qd_rel_l2"])
                self.assertIsNone(row["state_linf"])
            else:
                self.assertTrue(np.isfinite(row["joint_qd_rel_l2"]))
                self.assertTrue(np.isfinite(row["state_linf"]))
                if row["path"] == "propagation":
                    # Propagation uses deferred articulation response, so high-contact smoke
                    # rows are not expected to be bit-close to immediate D-wide GS
                    # after only two iterations. Keep this as a gross sanity bound.
                    self.assertLess(row["joint_qd_rel_l2"], 1.0e-1)
                    self.assertLess(row["state_linf"], 1.0e-1)

    def test_fpgs_articulation_operator_diagnostic_smoke(self):
        _require_cuda("CUDA is required for the FPGS operator diagnostic")

        script = BENCHMARKS_DIR / "fpgs_articulation_operator_diagnostic.py"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "fpgs_articulation_operator_diagnostic"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--links",
                    "2",
                    "--contacts-per-articulation",
                    "1",
                    "--max-sources",
                    "3",
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
            )

            summary = (out_dir / "summary.md").read_text(encoding="utf-8")
            results = json.loads((out_dir / "operator_diagnostic.json").read_text(encoding="utf-8"))
        self.assertTrue(results)
        self.assertIn("row operator `J_world @ Y_world.T`", summary)
        self.assertIn("operator_match", summary)
        for row in results:
            self.assertGreater(row["rows"], 0)
            self.assertGreater(row["D"], 0)
            self.assertGreater(row["sampled_sources"], 0)
            self.assertEqual(row["operator_status"], "operator_match")
            self.assertTrue(np.isfinite(row["rel_fro"]))
            self.assertTrue(np.isfinite(row["abs_linf"]))
            self.assertTrue(np.isfinite(row["qd_rel_l2"]))
            self.assertTrue(np.isfinite(row["qd_abs_linf"]))
            self.assertLessEqual(row["rel_fro"], 1.0e-4)
            self.assertLessEqual(row["abs_linf"], 1.0e-4)
            self.assertTrue(row["mass_condition_est"] is None or np.isfinite(row["mass_condition_est"]))
            self.assertTrue(row["worst_columns"])

    def test_fpgs_propagation_debug_residuals_include_propagation_rows(self):
        _require_cuda("CUDA is required for propagation PGS debug diagnostics")

        module = _load_row_scaling_module()

        case = module.BenchCase(
            "debug_propagation",
            "L=4,boxes=1",
            "articulated_free",
            articulations=1,
            links=4,
            contacts_per_articulation=1,
        )
        model, initial, contacts = module._prepare_case_run(case, "cuda:0")
        solver = module.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
            pgs_iterations=2,
            enable_contact_friction=True,
            dense_max_constraints=16,
            mf_max_constraints=16,
            pgs_warmstart=False,
            mf_warmstart=False,
            pgs_debug=True,
        )

        state_in = model.state()
        state_out = model.state()
        module._restore_state(model, state_in, initial)
        module._restore_state(model, state_out, initial)
        state_in.clear_forces()
        state_out.clear_forces()
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 120.0)
        wp.synchronize()

        self.assertGreater(int(np.sum(solver.propagation_constraint_count.numpy())), 0)
        self.assertTrue(solver._pgs_convergence_log)
        self.assertTrue(solver._pgs_ncp_residual_log)
        convergence = solver._pgs_convergence_log[-1]
        ncp = solver._pgs_ncp_residual_log[-1]
        self.assertEqual(convergence.shape, (2, 4))
        self.assertEqual(ncp.shape, (2, model.world_count, 6))
        self.assertTrue(np.all(np.isfinite(convergence)))
        self.assertTrue(np.all(np.isfinite(ncp)))
        self.assertGreater(np.max(convergence[:, 0]), 0.0)

    def test_fpgs_articulation_accuracy_probe_smoke(self):
        _require_cuda("CUDA is required for the FPGS accuracy probe")

        script = BENCHMARKS_DIR / "fpgs_articulation_accuracy_probe.py"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "fpgs_articulation_accuracy_probe"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--links",
                    "4",
                    "--contacts-per-articulation",
                    "1",
                    "--iterations",
                    "1,2",
                    "--reference-iterations",
                    "4",
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
            )

            summary = (out_dir / "summary.md").read_text(encoding="utf-8")
            results = json.loads((out_dir / "accuracy_probe.json").read_text(encoding="utf-8"))
        self.assertIn("Contacts are generated once", summary)
        self.assertEqual({row["path"] for row in results}, {"mf_immediate", "propagation"})
        self.assertEqual({row["pgs_iterations"] for row in results}, {1, 2})

        propagation_rows = [row for row in results if row["path"] == "propagation"]
        self.assertTrue(propagation_rows)
        for row in results:
            self.assertTrue(np.isfinite(row["final_r_compl"]))
            self.assertTrue(np.isfinite(row["self_ref_joint_qd_abs_linf"]))
            self.assertTrue(np.isfinite(row["mf_ref_joint_qd_abs_linf"]))
            self.assertGreaterEqual(row["self_ref_joint_qd_abs_linf"], 0.0)
            self.assertGreaterEqual(row["mf_ref_joint_qd_abs_linf"], 0.0)
        for row in propagation_rows:
            self.assertEqual(row["dense_rows_total"], 0)
            self.assertGreater(row["propagation_rows_total"], 0)
            self.assertTrue(np.isfinite(row["same_iter_vs_mf_joint_qd_abs_linf"]))
            self.assertGreaterEqual(row["same_iter_vs_mf_joint_qd_abs_linf"], 0.0)

    def test_fpgs_articulation_row_scaling_stress_case_coverage(self):
        module = _load_row_scaling_module()

        cases = module._cases_for_preset("stress")

        free_free = [case for case in cases if case.sweep == "free_free_dof"]
        articulated_contacts = [case for case in cases if case.sweep == "articulated_contact_rows"]
        chain_depth = [case for case in cases if case.sweep == "chain_depth"]
        articulation_count = [case for case in cases if case.sweep == "articulation_count"]

        self.assertEqual([case.free_pairs for case in free_free], [1, 2, 4, 8, 16, 21])
        self.assertTrue(all(case.case_kind == "free_free" for case in free_free))

        self.assertEqual(
            [(case.links, case.contacts_per_articulation) for case in articulated_contacts],
            [
                (4, 4),
                (16, 8),
                (16, 16),
                (64, 8),
                (64, 32),
                (64, 64),
            ],
        )
        self.assertTrue(all(case.case_kind == "articulated_free" for case in articulated_contacts))

        self.assertEqual([case.links for case in chain_depth], [2, 16, 32, 64])
        self.assertEqual([case.contacts_per_articulation for case in chain_depth], [2, 16, 32, 64])
        self.assertEqual([case.articulations for case in articulation_count], [1, 2, 16, 32, 64, 128])


if __name__ == "__main__":
    unittest.main()
