# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The split solve of SolverFeatherPGS: dense Delassus rows interleaved with free-body rows."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 240.0
PLATFORM_TOP = 0.5
BOX_HALF = 0.1


def _stack_on_platform(device, num_worlds=2):
    """Two free boxes stacked on a position-driven platform slider, without a ground plane.

    The lower box couples the platform's dense rows to the box-box free-body rows, so the
    stack only rests if the two row families see each other's impulses.
    """
    world = newton.ModelBuilder()
    world.default_shape_cfg.mu = 0.5
    platform = world.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, PLATFORM_TOP - 0.05), wp.quat_identity()))
    world.add_shape_box(platform, hx=0.4, hy=0.4, hz=0.05)
    slider = world.add_joint_prismatic(
        -1,
        platform,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, PLATFORM_TOP - 0.05), wp.quat_identity()),
        target_ke=5.0e4,
        target_kd=5.0e2,
    )
    world.add_articulation([slider])
    for level in range(2):
        z = PLATFORM_TOP + BOX_HALF * (2 * level + 1)
        body = world.add_body(xform=wp.transform(wp.vec3(0.02 * level, 0.0, z), wp.quat_identity()))
        world.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
    builder = newton.ModelBuilder()
    builder.replicate(world, num_worlds, spacing=(2.0, 0.0, 0.0))
    return builder.finalize(device=device)


def _run(model, solver, steps):
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return state_0


def test_mixed_world_stack_rests(test, device):
    """Rest a box stack on an articulated platform: dense and free-body rows exchange impulses every sweep."""
    model = _stack_on_platform(device)
    solver = SolverFeatherPGS(model, pgs_mode="split", pgs_iterations=16)
    test.assertTrue(solver._has_mixed_contacts)
    state = _run(model, solver, 240)
    solver.check_constraint_capacity()
    test.assertGreater(int(solver.constraint_count.numpy()[0]), 0)
    test.assertGreater(int(solver.mf_constraint_count.numpy()[0]), 0)
    # The drive spring sags under the load; the boxes rest on the platform and on each other.
    body_q = state.body_q.numpy().reshape(model.world_count, 3, 7)
    platform_top = body_q[:, 0, 2] + 0.05
    for level in range(2):
        expected = platform_top + BOX_HALF * (2 * level + 1)
        np.testing.assert_allclose(body_q[:, level + 1, 2], expected, atol=2.0e-3, err_msg=f"box {level}")
    test.assertLess(float(np.abs(state.body_qd.numpy()).max()), 2.0e-2)


def test_mixed_world_matches_matrix_free_solve(test, device):
    """Converge to the matrix-free solve's resting state on the same mixed scene."""
    final = {}
    for pgs_mode in ("split", "matrix_free"):
        model = _stack_on_platform(device)
        solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, pgs_iterations=32)
        final[pgs_mode] = _run(model, solver, 120).body_q.numpy()
    np.testing.assert_allclose(final["split"], final["matrix_free"], rtol=0.0, atol=2.0e-3)


def test_captured_mixed_step_matches_eager(test, device):
    """Replay captured collide + split step on a mixed scene like eager stepping."""
    trajectories = []
    for capture in (False, True):
        model = _stack_on_platform(device)
        solver = SolverFeatherPGS(model, pgs_mode="split")
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        def substep(
            solver=solver, pipeline=pipeline, contacts=contacts, state_0=state_0, state_1=state_1, control=control
        ):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_0, name), getattr(state_1, name))

        substep()
        if capture:
            with wp.ScopedCapture(device=device) as graph:
                substep()
            for _ in range(59):
                wp.capture_launch(graph.graph)
        else:
            for _ in range(59):
                substep()
        trajectories.append(state_0.body_q.numpy())
    # The right-hand side accumulates the contributions of several articulations with atomics,
    # so two runs agree only to accumulated round-off (about 1e-5 here).
    np.testing.assert_allclose(trajectories[1][:, :3], trajectories[0][:, :3], rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(trajectories[1][:, 3:], trajectories[0][:, 3:], rtol=0.0, atol=5.0e-4)


class TestFeatherPGSSplit(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires a CUDA device to compare against")
    def test_cpu_and_cuda_split_solves_agree(self):
        """Match the CPU split solve (scalar kernels) and the CUDA split solve (native kernels)."""
        final = {}
        for device in ("cpu", "cuda:0"):
            model = _stack_on_platform(device)
            final[device] = _run(model, SolverFeatherPGS(model, pgs_mode="split"), 120).body_q.numpy()
        np.testing.assert_allclose(final["cpu"][:, :3], final["cuda:0"][:, :3], rtol=0.0, atol=1.0e-4)
        np.testing.assert_allclose(final["cpu"][:, 3:], final["cuda:0"][:, 3:], rtol=0.0, atol=5.0e-4)


for _name, _func, _devices in (
    ("test_mixed_world_stack_rests", test_mixed_world_stack_rests, get_test_devices()),
    ("test_mixed_world_matches_matrix_free_solve", test_mixed_world_matches_matrix_free_solve, get_cuda_test_devices()),
    ("test_captured_mixed_step_matches_eager", test_captured_mixed_step_matches_eager, get_cuda_test_devices()),
):
    add_function_test(TestFeatherPGSSplit, _name, _func, devices=_devices)


if __name__ == "__main__":
    unittest.main()
