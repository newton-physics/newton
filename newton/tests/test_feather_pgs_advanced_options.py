# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental launch, diagnostic and profiling options of SolverFeatherPGS."""

import sys
import types
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_contact_compliance import run_fixture as run_compliance_fixture
from newton.tests.test_feather_pgs_fused_crba import _trajectory, _tree
from newton.tests.test_feather_pgs_global_options import _build_contact_scene, _run, _set_box_velocities
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _assert_bitwise(test, a, b, label):
    np.testing.assert_array_equal(a[1], b[1], err_msg=f"{label}: joint_q")
    np.testing.assert_array_equal(a[2], b[2], err_msg=f"{label}: joint_qd")


_POINT = {"init": _set_box_velocities, "steps": 8, "friction_anchor_beta": 0.0, "pgs_mode": "split"}


# ---------------------------------------------------------------------------------------------------------------
# Launch options


def test_launch_options_default_to_the_automatic_selection(test, device):
    """Keep the automatic kernel selection and launch widths when the options keep their defaults."""
    solver = SolverFeatherPGS(_build_contact_scene(device), pgs_mode="split")
    test.assertEqual(solver.pgs_kernel, "auto" if device.is_cuda else "loop")
    test.assertEqual((solver._pgs_chunk_size, solver._tile_threads, solver._serial_kernel_block_dim), (1, 64, 256))
    modes = ({"pgs_mode": "split"}, {}) if device.is_cuda else ({"pgs_mode": "split"},)
    for mode in modes:
        run = {"init": _set_box_velocities, "steps": 8, **mode}
        explicit = {"pgs_kernel": "auto", "pgs_chunk_size": None, "tile_threads": 64, "serial_kernel_block_dim": 256}
        _assert_bitwise(
            test,
            _run(_build_contact_scene(device), **run),
            _run(_build_contact_scene(device), **explicit, **run),
            "defaults",
        )


def test_launch_options_match_the_kernel_override_hook(test, device):
    """Select the same kernels as the matching _kernel_overrides entries, which keep precedence."""
    cases = [
        ({"pgs_kernel": "loop"}, {"pgs_kernel": "loop"}),
        ({"pgs_kernel": "tiled_row"}, {"pgs_kernel": "tiled"}),
        ({"pgs_kernel": "tiled_contact"}, {"pgs_kernel": "tiled_contact"}),
        ({"pgs_kernel": "streaming", "pgs_chunk_size": 2}, {"pgs_kernel": "streaming", "pgs_chunk_size": 2}),
        ({"tile_threads": 128}, {"tile_threads": 128}),
        ({"serial_kernel_block_dim": 64}, {"serial_kernel_block_dim": 64}),
    ]
    for options, overrides in cases:
        with test.subTest(options=options):
            public = _run(_build_contact_scene(device), **options, **_POINT)
            hook = _run(_build_contact_scene(device), overrides=overrides, **_POINT)
            _assert_bitwise(test, public, hook, str(options))
    with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", {"tile_threads": 32, "pgs_kernel": "loop"}):
        solver = SolverFeatherPGS(
            _build_contact_scene(device), pgs_mode="split", tile_threads=128, pgs_kernel="tiled_row"
        )
    test.assertEqual((solver._tile_threads, solver.pgs_kernel), (32, "loop"))


def test_split_kernels_agree_with_the_loop_kernel(test, device):
    """Reach the loop kernel's result with the row kernels, and the block kernels' result with each other."""
    reference = _run(_build_contact_scene(device), pgs_kernel="loop", **_POINT)
    for options in ({"pgs_kernel": "tiled_row"}, {"tile_threads": 32}, {"tile_threads": 256}):
        with test.subTest(options=options):
            result = _run(_build_contact_scene(device), **options, **_POINT)
            np.testing.assert_allclose(result[1], reference[1], rtol=0.0, atol=1.0e-5)
            np.testing.assert_allclose(result[2], reference[2], rtol=0.0, atol=1.0e-3)
    block = _run(_build_contact_scene(device), pgs_kernel="tiled_contact", **_POINT)
    np.testing.assert_allclose(block[1], reference[1], rtol=0.0, atol=2.0e-2)
    for options in ({"pgs_kernel": "streaming"}, {"pgs_kernel": "streaming", "pgs_chunk_size": 4}):
        with test.subTest(options=options):
            result = _run(_build_contact_scene(device), **options, **_POINT)
            np.testing.assert_allclose(result[1], block[1], rtol=0.0, atol=1.0e-5)
            np.testing.assert_allclose(result[2], block[2], rtol=0.0, atol=1.0e-3)


def test_serial_block_dim_and_matrix_free_kernel_choice_are_bitwise(test, device):
    """Leave matrix-free results unchanged by the serial block size and the split-only kernel choice."""
    run = {"init": _set_box_velocities, "steps": 8}
    default = _run(_build_contact_scene(device), **run)
    for options in ({"serial_kernel_block_dim": 64}, {"serial_kernel_block_dim": 512}, {"pgs_kernel": "loop"}):
        with test.subTest(options=options):
            _assert_bitwise(test, default, _run(_build_contact_scene(device), **options, **run), str(options))


def test_tile_threads_keep_fused_assembly_correct_above_the_block_width(test, device):
    """Assemble articulations wider than a 32-thread block correctly through the public option."""
    model = _tree(device, 33, worlds=1)
    with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", {"sparse_mass_matrix": False}):
        reference, q_ref, qd_ref = _trajectory(model, steps=2)
        fused, q, qd = _trajectory(model, steps=2, use_parallel_streams=True, tile_threads=32)
    size = fused.size_groups[0]
    test.assertIsNotNone(fused._crba_cholesky_kernels_by_size[size])
    test.assertEqual(fused._tile_threads, 32)
    np.testing.assert_allclose(fused.L_by_size[size].numpy(), reference.L_by_size[size].numpy(), atol=2.0e-5)
    np.testing.assert_allclose(q, q_ref, rtol=1.0e-5, atol=2.0e-5)
    np.testing.assert_allclose(qd, qd_ref, rtol=1.0e-5, atol=2.0e-4)


def test_launch_options_validate_their_values(test, device):
    """Reject unknown kernels, unsupported widths and contact-only kernels on rows they cannot solve."""
    model = _build_contact_scene(device)
    for options, pattern in (
        ({"pgs_kernel": "tiled"}, "pgs_kernel"),
        ({"pgs_kernel": "fast"}, "pgs_kernel"),
        ({"pgs_chunk_size": 0}, "pgs_chunk_size"),
        ({"tile_threads": 48}, "tile_threads"),
        ({"serial_kernel_block_dim": 0}, "serial_kernel_block_dim"),
        ({"serial_kernel_block_dim": 48}, "serial_kernel_block_dim"),
    ):
        with test.subTest(options=options), test.assertRaisesRegex(ValueError, pattern):
            SolverFeatherPGS(model, pgs_mode="split", **options)
    if not device.is_cuda:
        return
    for kernel in ("tiled_contact", "streaming"):
        for options in (
            {"enable_joint_limits": True},
            {"enable_contact_friction": False},
            {"contact_friction_gap_threshold": 0.01},
            {"contact_friction_position_iterations": 2},
        ):
            with test.subTest(kernel=kernel, options=options), test.assertRaisesRegex(ValueError, "contact rows only"):
                SolverFeatherPGS(model, pgs_mode="split", pgs_kernel=kernel, **options)


# ---------------------------------------------------------------------------------------------------------------
# pgs_debug


def test_debug_logs_matrix_free_convergence_and_residuals(test, device):
    """Log one row per iteration and per-world contact residuals for every matrix-free step."""
    iterations = 6
    run = {"init": _set_box_velocities, "steps": 5, "pgs_iterations": iterations}
    for label, options in (
        ("immediate", {}),
        ("propagation", {"articulated_contact_response": "propagation"}),
        ("propagation-fused", {"articulated_contact_response": "propagation-fused"}),
    ):
        with test.subTest(solve=label):
            model = _build_contact_scene(device)
            debug = _run(model, pgs_debug=True, **options, **run)
            solver = debug[0]
            test.assertEqual(len(solver.pgs_convergence_log), 5)
            test.assertEqual(len(solver.pgs_ncp_residual_log), 5)
            test.assertIs(solver.pgs_convergence_log, solver._pgs_convergence_log)
            for convergence, residuals in zip(solver.pgs_convergence_log, solver.pgs_ncp_residual_log, strict=True):
                test.assertEqual(convergence.shape, (iterations, 4))
                test.assertEqual(residuals.shape, (iterations, model.world_count, 6))
                test.assertTrue(np.all(np.isfinite(convergence)))
                test.assertTrue(np.all(np.isfinite(residuals)))
                test.assertTrue(np.all(residuals >= 0.0))
                test.assertGreater(convergence[0, 0], 0.0)
                test.assertLessEqual(convergence[-1, 0], convergence[0, 0])
            plain = _run(_build_contact_scene(device), **options, **run)
            np.testing.assert_allclose(debug[1], plain[1], rtol=0.0, atol=1.0e-5)
            np.testing.assert_allclose(debug[2], plain[2], rtol=0.0, atol=1.0e-3)


def test_debug_iterations_reproduce_the_whole_solve(test, device):
    """Reach bitwise the same state one iteration per launch, with friction starting at the right iteration."""
    run = {"init": _set_box_velocities, "steps": 5, "pgs_iterations": 6, "contact_friction_position_iterations": 3}
    solves = [("split", {"pgs_mode": "split"})]
    if device.is_cuda:
        solves += [
            ("physx_grasp", {"pgs_schedule": "physx_grasp"}),
            ("propagation", {"articulated_contact_response": "propagation"}),
            ("propagation-colored", {"articulated_contact_response": "propagation-colored"}),
        ]
    for label, options in solves:
        with test.subTest(solve=label):
            debug = _run(_build_contact_scene(device), pgs_debug=True, **options, **run)
            _assert_bitwise(test, debug, _run(_build_contact_scene(device), **options, **run), label)
            test.assertEqual(debug[0].pgs_convergence_log[-1].shape, (6, 4))


def test_debug_logs_split_impulse_changes(test, device):
    """Log the largest impulse change of each split iteration, with zero matrix-free metrics."""
    solver = _run(_build_contact_scene(device), init=_set_box_velocities, steps=3, pgs_mode="split", pgs_debug=True)[0]
    test.assertEqual(len(solver.pgs_convergence_log), 3)
    test.assertEqual(solver.pgs_ncp_residual_log, [])
    for convergence in solver.pgs_convergence_log:
        test.assertEqual(convergence.shape, (solver.pgs_iterations, 4))
        test.assertGreater(convergence[0, 0], 0.0)
        np.testing.assert_array_equal(convergence[:, 1:], 0.0)


def test_debug_logs_split_free_body_rows(test, device):
    """Log the free-body rows of a split solve that has no dense rows, without changing the step."""
    template = newton.ModelBuilder()
    template.add_ground_plane()
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()))
    template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(template, 1)
    model = builder.finalize(device=device)
    results = []
    for debug in (False, True):
        solver = SolverFeatherPGS(model, pgs_mode="split", pgs_iterations=6, pgs_debug=debug)
        state, out = model.state(), model.state()
        state.joint_qd.assign(np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, out, model.control(), contacts, 1.0 / 240.0)
        results.append((solver, out.joint_qd.numpy()))
    solver = results[1][0]
    test.assertFalse(solver._has_mixed_contacts)
    test.assertEqual(int(solver.constraint_count.numpy()[0]), 0)
    test.assertGreater(int(solver.mf_constraint_count.numpy()[0]), 0)
    np.testing.assert_array_equal(results[0][1], results[1][1])
    log = solver.pgs_convergence_log[0]
    test.assertEqual(log.shape, (6, 4))
    # The first sweep moves every impulse from zero, so its change is the largest first-sweep impulse.
    test.assertGreater(log[0, 0], 1.0)
    test.assertLessEqual(log[-1, 0], log[0, 0])


def test_cpu_ignores_contact_only_kernels(test, device):
    """Run the scalar loop on CPU for every pgs_kernel, including rows the contact-only kernels reject."""
    model = _build_contact_scene(device)
    for kernel in ("tiled_contact", "streaming"):
        for options in ({}, {"enable_contact_friction": False}, {"enable_joint_limits": True}):
            with test.subTest(kernel=kernel, options=options):
                solver = SolverFeatherPGS(model, pgs_mode="split", pgs_kernel=kernel, **options)
                test.assertEqual(solver.pgs_kernel, "loop")


def test_debug_rejects_unsupported_combinations(test, device):
    """Reject graph capture, compliance, torsion and the contact-first schedule with pgs_debug."""
    model = _build_contact_scene(device)
    with test.assertRaisesRegex(ValueError, "contact_then_internal"):
        SolverFeatherPGS(model, pgs_debug=True, pgs_schedule="contact_then_internal")
    with test.assertRaisesRegex(ValueError, "pgs_debug"):
        SolverFeatherPGS(model, pgs_debug=True, contact_torsion_radius=0.01)
    with test.assertRaisesRegex(ValueError, "pgs_debug"):
        run_compliance_fixture(
            device=device, articulated=True, enabled=True, steps=1, solver_options={"pgs_debug": True}
        )
    with test.assertRaisesRegex(RuntimeError, "graph capture"):
        _run(model, init=_set_box_velocities, steps=2, capture=True, pgs_debug=True)


# ---------------------------------------------------------------------------------------------------------------
# nvtx


class _FakeNvtx(types.ModuleType):
    """Record NVTX ranges in place of the nvtx package."""

    def __init__(self):
        super().__init__("nvtx")
        self.events = []

    def start_range(self, message):
        self.events.append(("start", message))
        return len(self.events)

    def end_range(self, range_id):
        self.events.append(("end", range_id))


def test_nvtx_annotates_every_stage_without_changing_results(test, device):
    """Open and close one range per step stage, leaving the results bitwise unchanged."""
    fake = _FakeNvtx()
    mode = {} if device.is_cuda else {"pgs_mode": "split"}
    run = {"init": _set_box_velocities, "steps": 3, **mode}
    with mock.patch.dict(sys.modules, {"nvtx": fake}):
        annotated = _run(_build_contact_scene(device), nvtx=True, **run)
    _assert_bitwise(test, annotated, _run(_build_contact_scene(device), **run), "nvtx")
    stages = [message for kind, message in fake.events if kind == "start"]
    expected = [
        "FeatherPGS.fk_id_crba",
        "FeatherPGS.factor",
        "FeatherPGS.unconstrained_velocity",
        "FeatherPGS.rows",
        "FeatherPGS.solve",
        "FeatherPGS.integrate",
    ]
    test.assertEqual(stages, expected * 3)
    # Each range closes before the next one opens, and the last one closes at the end of the step.
    test.assertEqual([kind for kind, _ in fake.events], ["start", "end"] * len(stages))
    for index in range(0, len(fake.events), 2):
        test.assertEqual(fake.events[index + 1][1], index + 1)


def test_nvtx_closes_the_open_range_when_a_stage_raises(test, device):
    """Close the open stage range when a step raises."""
    fake = _FakeNvtx()
    mode = {} if device.is_cuda else {"pgs_mode": "split"}
    model = _build_contact_scene(device)
    with mock.patch.dict(sys.modules, {"nvtx": fake}):
        solver = SolverFeatherPGS(model, nvtx=True, **mode)
    state_in, state_out = model.state(), model.state()
    with mock.patch.object(solver, "_stage1_crba", side_effect=RuntimeError("injected")):
        with test.assertRaisesRegex(RuntimeError, "injected"):
            solver.step(state_in, state_out, model.control(), None, 1.0 / 240.0)
    test.assertEqual([kind for kind, _ in fake.events], ["start", "end"])
    test.assertIsNone(solver._nvtx_range)


def test_nvtx_requires_the_nvtx_package(test, device):
    """Raise ImportError at construction when nvtx is requested without the nvtx package."""
    with mock.patch.dict(sys.modules, {"nvtx": None}), test.assertRaises(ImportError):
        SolverFeatherPGS(_build_contact_scene(device), nvtx=True, pgs_mode="split")


class TestFeatherPGSAdvancedOptions(unittest.TestCase):
    pass


for _fn in (
    test_launch_options_default_to_the_automatic_selection,
    test_launch_options_validate_their_values,
    test_debug_iterations_reproduce_the_whole_solve,
    test_debug_logs_split_impulse_changes,
    test_debug_logs_split_free_body_rows,
    test_nvtx_annotates_every_stage_without_changing_results,
    test_nvtx_closes_the_open_range_when_a_stage_raises,
    test_nvtx_requires_the_nvtx_package,
):
    add_function_test(TestFeatherPGSAdvancedOptions, _fn.__name__, _fn, devices=get_test_devices())

for _fn in (
    test_launch_options_match_the_kernel_override_hook,
    test_split_kernels_agree_with_the_loop_kernel,
    test_serial_block_dim_and_matrix_free_kernel_choice_are_bitwise,
    test_tile_threads_keep_fused_assembly_correct_above_the_block_width,
    test_debug_logs_matrix_free_convergence_and_residuals,
    test_debug_rejects_unsupported_combinations,
):
    add_function_test(TestFeatherPGSAdvancedOptions, _fn.__name__, _fn, devices=get_cuda_test_devices())

add_function_test(
    TestFeatherPGSAdvancedOptions,
    test_cpu_ignores_contact_only_kernels.__name__,
    test_cpu_ignores_contact_only_kernels,
    devices=[device for device in get_test_devices() if device.is_cpu],
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
