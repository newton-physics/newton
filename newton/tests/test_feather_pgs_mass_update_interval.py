# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 60.0
# Per-articulation initial pose/velocity: [base, left branch, right branch].
INITIAL_JOINT_Q = (0.7, 0.3, -0.4)
INITIAL_JOINT_QD = (0.5, -0.2, 0.3)


def _build_model(device, num_worlds=2, ground=True):
    """Two-branch pendulum: base revolute joint plus two sibling branch links.

    The sibling branches are not ancestor-related, so H has structural zeros
    between their DOFs; with a ground plane the branch tips swing into contact.
    """
    env = newton.ModelBuilder()
    env.default_shape_cfg.density = 1000.0

    base = env.add_link()
    env.add_shape_box(base, hx=0.08, hy=0.08, hz=0.08)
    left = env.add_link()
    env.add_shape_box(left, hx=0.25, hy=0.05, hz=0.05)
    right = env.add_link()
    env.add_shape_box(right, hx=0.25, hy=0.05, hz=0.05)

    j_base = env.add_joint_revolute(
        -1,
        base,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    j_left = env.add_joint_revolute(
        base,
        left,
        parent_xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    j_right = env.add_joint_revolute(
        base,
        right,
        parent_xform=wp.transform(wp.vec3(-0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    env.add_articulation([j_base, j_left, j_right])

    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _build_floating_model(device, num_worlds=2):
    """Floating two-link articulation with nonzero root and joint motion."""
    env = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    root = env.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=2.0,
        com=wp.vec3(0.07, -0.03, 0.02),
        inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.05),
    )
    child = env.add_link(
        xform=wp.transform(wp.vec3(0.3, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(0.04, 0.01, -0.02),
        inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.014),
    )
    free = env.add_joint_free(root)
    hinge = env.add_joint_revolute(
        root,
        child,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    env.add_articulation([free, hinge])
    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    return builder.finalize(device=device)


def _make_floating_state(model):
    state = model.state()
    qd = state.joint_qd.numpy()
    for articulation in range(model.articulation_count):
        start = articulation * 7
        qd[start : start + 7] = (0.4, -0.2, 0.1, 0.5, -0.7, 0.3, 0.8)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _run_floating_trajectory(model, solver, num_steps, *, invalidate_fk_id_each_step):
    state_0 = _make_floating_state(model)
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    history = {name: [] for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    for _ in range(num_steps):
        if invalidate_fk_id_each_step:
            # Reset also forces a mass refresh, changing the comparison's cadence.
            solver.notify_state_changed()
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        for name, values in history.items():
            values.append(getattr(state_0, name).numpy().copy())
    return {name: np.stack(values) for name, values in history.items()}


def _make_initial_state(model):
    state = model.state()
    num_arts = model.articulation_count
    state.joint_q.assign(np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), num_arts))
    state.joint_qd.assign(np.tile(np.asarray(INITIAL_JOINT_QD, dtype=np.float32), num_arts))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _run_trajectory(model, solver, num_steps):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    joint_q_history = []
    for _ in range(num_steps):
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        joint_q_history.append(state_0.joint_q.numpy().copy())
    return np.stack(joint_q_history)


def _run_state_trajectory(model, solver, num_steps, *, invalidate_fk_id_each_step):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    history = {name: [] for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    for _ in range(num_steps):
        if invalidate_fk_id_each_step:
            # Isolate FK/ID reuse while preserving mass factors and warm starts.
            solver.notify_state_changed()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        for name, values in history.items():
            values.append(getattr(state_0, name).numpy().copy())
    return {name: np.stack(values) for name, values in history.items()}


class TestFeatherPGSMassUpdateInterval(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_split_fk_id_cache_does_not_alias_active_step_state(self):
        """Keep split-mode cached dynamics isolated from active solve scratch."""
        model = _build_floating_model("cuda:0")
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2, pgs_mode="split")
        state_in = _make_floating_state(model)
        state_aug = solver._prepare_augmented_state(state_in, model.state(), model.control())
        cache = solver._fk_id_cache

        active_cache_pairs = (
            (state_aug.body_q_com, cache.body_q_com),
            (state_aug.joint_S_s, cache.joint_S_s),
            (state_aug.body_I_s, cache.body_I_s),
            (solver._body_inertia_terms, cache.body_inertia_terms),
            (state_aug.body_v_s, cache.body_v_s),
            (state_aug.body_f_s, cache.body_f_s),
            (state_aug.body_a_s, cache.body_a_s),
            (solver.articulation_origin, cache.articulation_origin),
        )
        for active, cached in active_cache_pairs:
            self.assertNotEqual(active.ptr, cached.ptr)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free PGS requires CUDA")
    def test_matrix_free_fk_id_cache_keeps_zero_copy_path(self):
        """Keep matrix-free dynamics publication on its validated zero-copy path."""
        model = _build_floating_model("cuda:0")
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2, pgs_mode="matrix_free")

        self.assertTrue(solver._fk_id_cache_enabled)
        self.assertFalse(solver._fk_id_cache_uses_snapshot)
        self.assertIsNone(solver._fk_id_cache)

    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID launch requires CUDA")
    def test_fk_id_launch_storage_exists_when_reuse_is_disabled(self):
        """Keep the unconditional FK/ID launch output valid without reuse."""
        model = _build_model("cuda:0", num_worlds=1, ground=False)
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", enable_joint_velocity_limits=True)
        self.assertFalse(solver._fk_id_cache_enabled)
        self.assertEqual(solver._fk_id_cache_valid.shape[0], model.articulation_count)

        state_in = _make_initial_state(model)
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, DT)
        wp.synchronize()

    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_cached_fk_id_matches_forced_recomputation(self):
        """Match forced recomputation across cached fixed-base steps."""
        trajectories = []
        for invalidate_fk_id_each_step in (False, True):
            model = _build_model("cuda:0", ground=False)
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
            trajectories.append(
                _run_state_trajectory(
                    model, solver, num_steps=20, invalidate_fk_id_each_step=invalidate_fk_id_each_step
                )
            )

        for name in trajectories[0]:
            np.testing.assert_allclose(
                trajectories[0][name], trajectories[1][name], rtol=0.0, atol=2.0e-6, err_msg=name
            )

    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_cached_fk_id_matches_forced_recomputation_for_floating_base(self):
        """Match forced recomputation across cached floating-base steps."""
        trajectories = []
        for invalidate_fk_id_each_step in (False, True):
            model = _build_floating_model("cuda:0")
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
            trajectories.append(
                _run_floating_trajectory(
                    model, solver, num_steps=20, invalidate_fk_id_each_step=invalidate_fk_id_each_step
                )
            )

        for name in trajectories[0]:
            np.testing.assert_allclose(
                trajectories[0][name], trajectories[1][name], rtol=0.0, atol=2.0e-6, err_msg=name
            )

    @unittest.skipUnless(wp.is_cuda_available(), "parallel compact inertia refresh requires CUDA")
    def test_parallel_compact_refresh_matches_serial_trajectory(self):
        trajectories = []
        for use_parallel_streams in (False, True):
            model = _build_model("cuda:0")
            solver = SolverFeatherPGS(
                model,
                update_mass_matrix_interval=1,
                double_buffer=False,
                use_parallel_streams=use_parallel_streams,
            )
            trajectories.append(_run_trajectory(model, solver, num_steps=20))

        np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=0.0, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_captured_parallel_dynamics_matches_fused_trajectory(self):
        """Match captured parallel dynamics against eager fused execution."""
        graph_model = _build_model("cuda:0", ground=False)
        eager_model = _build_model("cuda:0", ground=False)
        graph_solver = SolverFeatherPGS(
            graph_model,
            update_mass_matrix_interval=1,
            double_buffer=False,
            use_parallel_streams=True,
        )
        eager_solver = SolverFeatherPGS(
            eager_model,
            update_mass_matrix_interval=1,
            double_buffer=False,
            use_parallel_streams=False,
        )
        graph_state, graph_out = _make_initial_state(graph_model), graph_model.state()
        eager_state, eager_out = _make_initial_state(eager_model), eager_model.state()
        graph_control, eager_control = graph_model.control(), eager_model.control()
        graph_pipeline = newton.CollisionPipeline(graph_model)
        eager_pipeline = newton.CollisionPipeline(eager_model)
        graph_contacts, eager_contacts = graph_pipeline.contacts(), eager_pipeline.contacts()

        def one_step(solver, state_in, state_out, control, contacts):
            solver.step(state_in, state_out, control, contacts, DT)
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

        one_step(graph_solver, graph_state, graph_out, graph_control, graph_contacts)
        one_step(eager_solver, eager_state, eager_out, eager_control, eager_contacts)
        with wp.ScopedCapture("cuda:0") as capture:
            one_step(graph_solver, graph_state, graph_out, graph_control, graph_contacts)

        for _ in range(20):
            wp.capture_launch(capture.graph)
            one_step(eager_solver, eager_state, eager_out, eager_control, eager_contacts)

        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            np.testing.assert_allclose(
                getattr(graph_state, name).numpy(),
                getattr(eager_state, name).numpy(),
                rtol=0.0,
                atol=2.0e-6,
                err_msg=name,
            )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_captured_two_substep_fk_id_reuse_matches_eager(self):
        """Replay two captured substeps without erasing the cached body forces.

        The solver allocates its augmented dynamics buffers on the first step. When that first step runs inside a
        CUDA graph capture, a zero-initialised allocation records a memset that replays before every frame; the
        FK/ID reuse path then consumes cleared body forces while its cache still reads valid.
        """
        warm_model = _build_model("cuda:0", ground=False)
        warm_solver = SolverFeatherPGS(warm_model, update_mass_matrix_interval=2, pgs_mode="matrix_free")
        warm_solver.step(_make_initial_state(warm_model), warm_model.state(), warm_model.control(), None, DT)
        wp.synchronize_device("cuda:0")

        graph_model = _build_model("cuda:0", ground=False)
        eager_model = _build_model("cuda:0", ground=False)
        # matrix_free keeps the zero-copy FK/ID reuse path; split restores its snapshot and would mask the defect.
        graph_solver = SolverFeatherPGS(graph_model, update_mass_matrix_interval=2, pgs_mode="matrix_free")
        eager_solver = SolverFeatherPGS(eager_model, update_mass_matrix_interval=2, pgs_mode="matrix_free")
        self.assertTrue(graph_solver._fk_id_cache_enabled)
        self.assertFalse(graph_solver._fk_id_cache_uses_snapshot)
        graph_a, graph_b = _make_initial_state(graph_model), graph_model.state()
        eager_a, eager_b = _make_initial_state(eager_model), eager_model.state()
        graph_control, eager_control = graph_model.control(), eager_model.control()

        def two_substeps(solver, state_in, state_out, control):
            # Keep one input state object so the FK/ID cache stays valid across substeps and frames.
            for _ in range(2):
                solver.step(state_in, state_out, control, None, DT)
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    wp.copy(getattr(state_in, name), getattr(state_out, name))

        # The very first launches of graph_solver happen inside the capture.
        with wp.ScopedCapture("cuda:0") as capture:
            two_substeps(graph_solver, graph_a, graph_b, graph_control)
        for _ in range(64):
            wp.capture_launch(capture.graph)
            two_substeps(eager_solver, eager_a, eager_b, eager_control)
        wp.synchronize_device("cuda:0")

        for name in ("joint_q", "joint_qd"):
            captured = getattr(graph_a, name).numpy()
            self.assertTrue(np.isfinite(captured).all(), f"{name} became non-finite under graph replay")
            np.testing.assert_allclose(captured, getattr(eager_a, name).numpy(), rtol=0.0, atol=2.0e-6, err_msg=name)

    def test_interval_two_contact_trajectory_stays_close_to_reference(self):
        device = wp.get_device()
        history = {}
        for interval in (1, 2):
            model = _build_model(device)
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=interval)
            history[interval] = _run_trajectory(model, solver, num_steps=60)

        self.assertTrue(np.isfinite(history[2]).all(), "interval=2 trajectory diverged to non-finite values")
        # The trajectories are intentionally NOT bit-identical (stale H between
        # mass updates is the optimization); they must stay close on a short
        # contact-rich horizon.
        drift = np.abs(history[2] - history[1]).max()
        self.assertLess(drift, 0.05, f"interval=2 drifted {drift} rad from the interval=1 reference")
        # Sanity: the scene actually moved, so the comparison is not vacuous.
        moved = np.abs(history[1][-1] - np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), 2)).max()
        self.assertGreater(moved, 1.0e-3)

    def test_global_flag_cadence_bakes_one_zero_pattern(self):
        device = wp.get_device()
        model = _build_model(device)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
        state_0 = _make_initial_state(model)
        state_1 = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        control = model.control()

        expected_masks = ([1, 1], [0, 0], [1, 1], [0, 0])
        for step_index, expected in enumerate(expected_masks):
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            self.assertEqual(
                solver.mass_update_mask.numpy().tolist(),
                expected,
                f"unexpected mass_update_mask after step {step_index}",
            )

    def test_model_change_request_refreshes_a_reuse_step(self):
        """Refresh every articulation after notification and consume each request."""
        model = _build_model(wp.get_device(), ground=False)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
        state_0 = _make_initial_state(model)
        state_1 = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        control = model.control()

        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)

        self.assertEqual(solver.mass_update_mask.numpy().tolist(), [1, 1])
        self.assertEqual(solver._mass_update_requested.numpy().tolist(), [0] * model.articulation_count)

    def test_teardown_waits_once_for_each_owned_stream(self):
        """Synchronize current stream owners and tolerate CUDA shutdown."""
        solver = object.__new__(SolverFeatherPGS)
        local = object()
        residual = object()
        dynamics = object()
        memset = object()
        size = object()
        solver._local_internal_stream = local
        solver._local_residual_stream = residual
        solver._local_pair_stream = local
        solver._global_inertia_stream = None
        solver._articulation_dynamics_stream = dynamics
        solver._memset_stream = memset
        solver._size_streams = {3: local, 6: size}

        def synchronize(stream):
            if stream is residual:
                raise RuntimeError("CUDA is shutting down")

        with mock.patch.object(wp, "synchronize_stream", side_effect=synchronize) as synchronize_stream:
            solver.__del__()
        self.assertEqual(
            synchronize_stream.call_args_list,
            [mock.call(local), mock.call(residual), mock.call(dynamics), mock.call(memset), mock.call(size)],
        )

        # Prevent a second meaningful synchronization when Python collects the
        # deliberately uninitialized instance after this regression returns.
        solver._local_internal_stream = None
        solver._local_residual_stream = None
        solver._local_pair_stream = None
        solver._articulation_dynamics_stream = None
        solver._memset_stream = None
        solver._size_streams = {}

    def test_teardown_waits_once_for_finalized_stream_device(self):
        """Synchronize each finalized-stream device while retaining live-stream waits."""
        device = wp.get_device()
        first = SimpleNamespace(device=device)
        second = SimpleNamespace(device=device)
        live = object()
        solver = object.__new__(SolverFeatherPGS)
        solver._local_internal_stream = first
        solver._local_residual_stream = second
        solver._local_pair_stream = live
        solver._size_streams = {3: first}
        try:
            with (
                mock.patch("gc.is_finalized", side_effect=lambda stream: stream is not live),
                mock.patch.object(wp, "synchronize_device") as synchronize_device,
                mock.patch.object(wp, "synchronize_stream") as synchronize_stream,
            ):
                solver.__del__()
            synchronize_device.assert_called_once_with(device)
            synchronize_stream.assert_called_once_with(live)
        finally:
            solver._local_internal_stream = None
            solver._local_residual_stream = None
            solver._local_pair_stream = None
            solver._size_streams = {}

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA stream cleanup requires a GPU")
    def test_teardown_handles_already_finalized_stream_in_cycle(self):
        """Cyclic GC can finalize a stream before its solver without crashing cleanup."""
        # Isolate the native crash so a failure cannot abort the enclosing suite.
        code = """
import gc
import sys
import weakref
import warp as wp
from newton.solvers import SolverFeatherPGS
wp.init()
unraisable = []
sys.unraisablehook = unraisable.append
stream = wp.Stream("cuda:0")
solver = object.__new__(SolverFeatherPGS)
solver._local_internal_stream = stream
stream.solver_owner = solver
reference = weakref.ref(solver)
del stream, solver
gc.collect()
assert not unraisable, [str(error.exc_value) for error in unraisable]
assert reference() is None
print("CYCLE_CLEANUP_COMPLETE", flush=True)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CYCLE_CLEANUP_COMPLETE", result.stdout)
        self.assertNotIn("Warp CUDA error", result.stderr)

    def test_mass_refresh_has_no_obsolete_limit_count_state(self):
        solver = SolverFeatherPGS(_build_model(wp.get_device(), ground=False))
        for attribute in ("aug_limit_counts", "aug_prev_limit_counts", "limit_change_mask"):
            self.assertFalse(hasattr(solver, attribute), f"obsolete mass-refresh state {attribute!r} was restored")


if __name__ == "__main__":
    unittest.main(verbosity=2)
