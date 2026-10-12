# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental launch, diagnostic and profiling options of SolverFeatherPGS."""

import sys
import types
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import solver_feather_pgs
from newton._src.solvers.feather_pgs._testing import kernel_overrides
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_fused_crba import _trajectory, _tree
from newton.tests.test_feather_pgs_global_options import _build_contact_scene, _run, _set_box_velocities
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _assert_bitwise(test, a, b, label):
    np.testing.assert_array_equal(a[1], b[1], err_msg=f"{label}: joint_q")
    np.testing.assert_array_equal(a[2], b[2], err_msg=f"{label}: joint_qd")


_POINT = {"init": _set_box_velocities, "steps": 8, "friction_anchor_beta": 0.0, "solve_mode": "split"}


# ---------------------------------------------------------------------------------------------------------------
# Kernel and launch overrides


def test_default_launch_settings(test, device):
    """Select the kernels and launch widths automatically, matching explicitly pinned defaults bitwise."""
    solver = SolverFeatherPGS(_build_contact_scene(device), solve_mode="split")
    test.assertEqual(solver.pgs_kernel, "auto" if device.is_cuda else "loop")
    test.assertEqual((solver._pgs_chunk_size, solver._tile_threads, solver._serial_kernel_block_dim), (1, 64, 256))
    modes = ({"solve_mode": "split"}, {}) if device.is_cuda else ({"solve_mode": "split"},)
    for mode in modes:
        run = {"init": _set_box_velocities, "steps": 8, **mode}
        explicit = {"pgs_kernel": "auto", "pgs_chunk_size": 1, "tile_threads": 64, "serial_kernel_block_dim": 256}
        _assert_bitwise(
            test,
            _run(_build_contact_scene(device), **run),
            _run(_build_contact_scene(device), overrides=explicit, **run),
            "defaults",
        )


def test_kernel_overrides_are_scoped_to_construction(test, device):
    """Apply overrides to solvers built inside the block only, and keep them on that instance."""
    model = _build_contact_scene(device)
    with kernel_overrides({"tile_threads": 32, "serial_kernel_block_dim": 64}):
        pinned = SolverFeatherPGS(model, solve_mode="split")
        with kernel_overrides({"tile_threads": 128}):
            nested = SolverFeatherPGS(model, solve_mode="split")
    default = SolverFeatherPGS(model, solve_mode="split")
    test.assertEqual((pinned._tile_threads, pinned._serial_kernel_block_dim), (32, 64))
    test.assertEqual((nested._tile_threads, nested._serial_kernel_block_dim), (128, 256))
    test.assertEqual((default._tile_threads, default._serial_kernel_block_dim), (64, 256))
    test.assertEqual(pinned._kernel_overrides, {"tile_threads": 32, "serial_kernel_block_dim": 64})
    test.assertEqual(default._kernel_overrides, {})


def test_split_kernels_agree_with_the_loop_kernel(test, device):
    """Reach the loop kernel's result with the row kernels, and the block kernels' result with each other."""
    reference = _run(_build_contact_scene(device), overrides={"pgs_kernel": "loop"}, **_POINT)
    for overrides in ({"pgs_kernel": "tiled"}, {"tile_threads": 32}, {"tile_threads": 256}):
        with test.subTest(overrides=overrides):
            result = _run(_build_contact_scene(device), overrides=overrides, **_POINT)
            np.testing.assert_allclose(result[1], reference[1], rtol=0.0, atol=1.0e-5)
            np.testing.assert_allclose(result[2], reference[2], rtol=0.0, atol=1.0e-3)
    block = _run(_build_contact_scene(device), overrides={"pgs_kernel": "tiled_contact"}, **_POINT)
    np.testing.assert_allclose(block[1], reference[1], rtol=0.0, atol=2.0e-2)
    for overrides in ({"pgs_kernel": "streaming"}, {"pgs_kernel": "streaming", "pgs_chunk_size": 4}):
        with test.subTest(overrides=overrides):
            result = _run(_build_contact_scene(device), overrides=overrides, **_POINT)
            np.testing.assert_allclose(result[1], block[1], rtol=0.0, atol=1.0e-5)
            np.testing.assert_allclose(result[2], block[2], rtol=0.0, atol=1.0e-3)


def test_serial_block_dim_and_matrix_free_kernel_choice_are_bitwise(test, device):
    """Leave matrix-free results unchanged by the serial block size and the split-only kernel choice."""
    run = {"init": _set_box_velocities, "steps": 8}
    default = _run(_build_contact_scene(device), **run)
    for overrides in ({"serial_kernel_block_dim": 64}, {"serial_kernel_block_dim": 512}, {"pgs_kernel": "loop"}):
        with test.subTest(overrides=overrides):
            result = _run(_build_contact_scene(device), overrides=overrides, **run)
            _assert_bitwise(test, default, result, str(overrides))


def test_tile_threads_keep_fused_assembly_correct_above_the_block_width(test, device):
    """Assemble articulations wider than a 32-thread block correctly with 32 tile threads."""
    model = _tree(device, 33, worlds=1)
    with kernel_overrides({"sparse_mass_matrix": False}):
        reference, q_ref, qd_ref = _trajectory(model, steps=2)
    with kernel_overrides({"sparse_mass_matrix": False, "tile_threads": 32}):
        fused, q, qd = _trajectory(model, steps=2, use_parallel_streams=True)
    size = fused.size_groups[0]
    test.assertIsNotNone(fused._crba_cholesky_kernels_by_size[size])
    test.assertEqual(fused._tile_threads, 32)
    np.testing.assert_allclose(fused.L_by_size[size].numpy(), reference.L_by_size[size].numpy(), atol=2.0e-5)
    np.testing.assert_allclose(q, q_ref, rtol=1.0e-5, atol=2.0e-5)
    np.testing.assert_allclose(qd, qd_ref, rtol=1.0e-5, atol=2.0e-4)


def test_kernel_overrides_validate_their_values(test, device):
    """Reject unknown kernels, unsupported widths and contact-only kernels on rows they cannot solve."""
    model = _build_contact_scene(device)
    for overrides, pattern in (
        ({"pgs_kernel": "fast"}, "pgs_kernel"),
        ({"pgs_chunk_size": 0}, "pgs_chunk_size"),
        ({"tile_threads": 48}, "tile_threads"),
        ({"serial_kernel_block_dim": 0}, "serial_kernel_block_dim"),
        ({"serial_kernel_block_dim": 48}, "serial_kernel_block_dim"),
    ):
        with test.subTest(overrides=overrides), test.assertRaisesRegex(ValueError, pattern):
            with kernel_overrides(overrides):
                SolverFeatherPGS(model, solve_mode="split")
    if not device.is_cuda:
        return
    for kernel in ("tiled_contact", "streaming"):
        for options in (
            {"enable_joint_limits": True},
            {"contact_friction_gap_threshold": 0.01},
            {"contact_friction_position_iterations": 2},
        ):
            with test.subTest(kernel=kernel, options=options), test.assertRaisesRegex(ValueError, "contact rows only"):
                with kernel_overrides({"pgs_kernel": kernel}):
                    SolverFeatherPGS(model, solve_mode="split", **options)


def test_cpu_ignores_contact_only_kernels(test, device):
    """Run the scalar loop on CPU for every split kernel override, including rows the contact-only kernels reject."""
    model = _build_contact_scene(device)
    for kernel in ("tiled_contact", "streaming"):
        for options in ({}, {"contact_friction_gap_threshold": -float("inf")}, {"enable_joint_limits": True}):
            with test.subTest(kernel=kernel, options=options):
                with kernel_overrides({"pgs_kernel": kernel}):
                    solver = SolverFeatherPGS(model, solve_mode="split", **options)
                test.assertEqual(solver.pgs_kernel, "loop")


# ---------------------------------------------------------------------------------------------------------------
# NEWTON_PROFILE


class _FakeNvtx(types.ModuleType):
    """Record NVTX ranges in place of the nvtx package."""

    def __init__(self):
        super().__init__("nvtx")
        self.events = []

    def start_range(self, message, color=None):
        self.events.append(("start", message))
        return len(self.events)

    def end_range(self, range_id):
        self.events.append(("end", range_id))


def test_profile_annotates_every_stage_without_changing_results(test, device):
    """Open and close one NVTX range per step stage under NEWTON_PROFILE, leaving the results bitwise unchanged."""
    fake = _FakeNvtx()
    mode = {} if device.is_cuda else {"solve_mode": "split"}
    run = {"init": _set_box_velocities, "steps": 3, **mode}
    with mock.patch.dict(sys.modules, {"nvtx": fake}), mock.patch.object(solver_feather_pgs, "PROFILE_ENABLED", True):
        annotated = _run(_build_contact_scene(device), **run)
    _assert_bitwise(test, annotated, _run(_build_contact_scene(device), **run), "profile")
    stages = [message for kind, message in fake.events if kind == "start"]
    expected = [
        "FeatherPGS::fk_id_crba",
        "FeatherPGS::factor",
        "FeatherPGS::unconstrained_velocity",
        "FeatherPGS::rows",
        "FeatherPGS::solve",
        "FeatherPGS::integrate",
    ]
    test.assertEqual(stages, expected * 3)
    # Each range closes before the next one opens, and the last one closes at the end of the step.
    test.assertEqual([kind for kind, _ in fake.events], ["start", "end"] * len(stages))
    for index in range(0, len(fake.events), 2):
        test.assertEqual(fake.events[index + 1][1], index + 1)


def test_profile_closes_the_open_range_when_a_stage_raises(test, device):
    """Close the open stage range when a step raises."""
    fake = _FakeNvtx()
    mode = {} if device.is_cuda else {"solve_mode": "split"}
    model = _build_contact_scene(device)
    solver = SolverFeatherPGS(model, **mode)
    state_in, state_out = model.state(), model.state()
    with (
        mock.patch.dict(sys.modules, {"nvtx": fake}),
        mock.patch.object(solver_feather_pgs, "PROFILE_ENABLED", True),
        mock.patch.object(solver, "_stage1_crba", side_effect=RuntimeError("injected")),
        test.assertRaisesRegex(RuntimeError, "injected"),
    ):
        solver.step(state_in, state_out, model.control(), None, 1.0 / 240.0)
    test.assertEqual([kind for kind, _ in fake.events], ["start", "end"])
    test.assertIsNone(solver._profile_timer)


class TestFeatherPGSAdvancedOptions(unittest.TestCase):
    pass


for _fn in (
    test_default_launch_settings,
    test_kernel_overrides_are_scoped_to_construction,
    test_kernel_overrides_validate_their_values,
    test_profile_annotates_every_stage_without_changing_results,
    test_profile_closes_the_open_range_when_a_stage_raises,
):
    add_function_test(TestFeatherPGSAdvancedOptions, _fn.__name__, _fn, devices=get_test_devices())

for _fn in (
    test_split_kernels_agree_with_the_loop_kernel,
    test_serial_block_dim_and_matrix_free_kernel_choice_are_bitwise,
    test_tile_threads_keep_fused_assembly_correct_above_the_block_width,
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
