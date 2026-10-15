# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mass-matrix assembly, diagonal mass, refresh cadence and model notifications of SolverSteel."""

import gc
import unittest
import weakref

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.kernels import accumulate_group_diag_worlds, compute_composite_inertia
from newton._src.solvers.steel.solver_steel import _get_composite_inertia_warp_kernel, _get_hinv_jt_kernel
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def test_warp_reduction_matches_scalar_branched_trees(test, device):
    """Reduce two branched trees, joints not ordered by body, identically to the scalar kernel."""
    articulation_start = wp.array([0, 4, 7], dtype=wp.int32, device=device)
    articulation_joint_end = wp.array([4, 7], dtype=wp.int32, device=device)
    joint_ancestor = wp.array([-1, 0, 0, 2, -1, 4, 5], dtype=wp.int32, device=device)
    joint_child = wp.array(np.random.default_rng(41).permutation(7).astype(np.int32), dtype=wp.int32, device=device)
    body_inertia = wp.array(
        np.random.default_rng(42).normal(size=(7, 6, 6)).astype(np.float32),
        dtype=wp.spatial_matrix,
        device=device,
    )

    scalar = wp.empty_like(body_inertia)
    wp.launch(
        compute_composite_inertia,
        dim=2,
        inputs=[
            articulation_start,
            articulation_joint_end,
            wp.ones(2, dtype=wp.int32, device=device),
            joint_ancestor,
            joint_child,
            body_inertia,
        ],
        outputs=[scalar],
        device=device,
    )

    parallel = wp.empty_like(body_inertia)
    warps_per_block = 4
    kernel = _get_composite_inertia_warp_kernel(device.arch, warps_per_block)
    wp.launch_tiled(
        kernel,
        dim=[1],
        inputs=[
            2,
            wp.array([0, 1], dtype=wp.int32, device=device),
            articulation_start,
            articulation_joint_end,
            joint_ancestor,
            joint_child,
            body_inertia,
            wp.ones(2, dtype=wp.int32, device=device),
        ],
        outputs=[parallel],
        block_dim=32 * warps_per_block,
        device=device,
    )
    np.testing.assert_array_equal(parallel.numpy(), scalar.numpy())

    # Independent oracle: each composite inertia is the sum over the subtree.
    ancestor = joint_ancestor.numpy()
    child = joint_child.numpy()
    inertia = body_inertia.numpy().astype(np.float64)
    for joint in range(7):
        expected = np.zeros((6, 6))
        for other in range(7):
            node = other
            while node >= 0 and node != joint:
                node = ancestor[node]
            if node == joint:
                expected += inertia[child[other]]
        np.testing.assert_allclose(parallel.numpy()[child[joint]], expected, rtol=1.0e-5, atol=1.0e-5)


class TestSteelCompositeInertia(unittest.TestCase):
    pass


add_function_test(
    TestSteelCompositeInertia,
    "test_warp_reduction_matches_scalar_branched_trees",
    test_warp_reduction_matches_scalar_branched_trees,
    devices=get_cuda_test_devices(),
)


def _build_chain_model(device, num_links=3, num_worlds=2, *, with_free_body=False):
    chain = newton.ModelBuilder()
    hx = 0.3
    joints = []
    parent = -1
    root_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45 * wp.pi)
    for _ in range(num_links):
        link = chain.add_link()
        chain.add_shape_box(link, hx=hx - 0.08, hy=0.05, hz=0.05)
        if parent == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 2.5), q=root_rot)
        else:
            parent_xform = wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            chain.add_joint_revolute(
                parent=parent,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        parent = link
    chain.add_articulation(joints)
    if with_free_body:
        body = chain.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        chain.add_articulation([chain.add_joint_free(parent=-1, child=body)])
    main = newton.ModelBuilder()
    main.replicate(chain, num_worlds, spacing=(3.0, 3.0, 0.0))
    return main.finalize(device=device)


def _build_fixed_base_star_model(
    device, num_branches=16, num_worlds=2, *, ground=False, armature=0.0, with_chain=False, ball_branch=False
):
    """Build fixed-base stars of independent prismatic branches (a structurally diagonal mass matrix).

    ``with_chain`` adds a two-link serial chain per world, so each world holds two response
    groups; ``ball_branch`` replaces the last branch by a BALL joint, whose DOFs are coupled.
    """
    star = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if ground else 0.0))
    base = star.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [
        star.add_joint_fixed(parent=-1, child=base, parent_xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
    ]
    for branch in range(num_branches):
        child = star.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        if ground:
            star.add_shape_sphere(child, radius=0.05)
        parent_xform = wp.transform(wp.vec3(0.12 * branch, 0.0, 0.0), wp.quat_identity())
        if ball_branch and branch == num_branches - 1:
            joints.append(star.add_joint_ball(parent=base, child=child, parent_xform=parent_xform))
            continue
        joints.append(
            star.add_joint_prismatic(
                parent=base,
                child=child,
                axis=newton.Axis.Z,
                parent_xform=parent_xform,
                limit_lower=-0.4,
                limit_upper=0.05,
                armature=armature,
            )
        )
    star.add_articulation(joints)
    if with_chain:
        links = [star.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
        star.add_shape_sphere(links[1], radius=0.05)
        root = star.add_joint_revolute(
            -1, links[0], axis=newton.Axis.Y, parent_xform=wp.transform((-0.5, 0.0, 0.3), wp.quat_identity())
        )
        elbow = star.add_joint_revolute(
            links[0], links[1], axis=newton.Axis.Y, parent_xform=wp.transform((0.0, 0.0, -0.15), wp.quat_identity())
        )
        star.add_articulation([root, elbow])
    main = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if ground else 0.0))
    main.replicate(star, num_worlds, spacing=(3.0, 3.0, 0.0))
    if ground:
        main.add_ground_plane()
    return main.finalize(device=device)


def _run_star(model, solver, steps, *, contacts=False):
    """Step a star model from a fixed initial state and record joint_q / joint_qd per step."""
    size = int(model.joint_dof_count // model.world_count)
    state_0, state_1 = model.state(), model.state()
    state_0.joint_q.assign(np.tile(np.linspace(-0.04, 0.04, size, dtype=np.float32), model.world_count))
    state_0.joint_qd.assign(np.tile(np.linspace(0.1, -0.1, size, dtype=np.float32), model.world_count))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    pipeline = newton.CollisionPipeline(model, deterministic=True) if contacts else None
    collision = pipeline.contacts() if contacts else None
    history = []
    for _ in range(steps):
        state_0.clear_forces()
        if contacts:
            pipeline.collide(state_0, collision)
        solver.step(state_0, state_1, control, collision, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0
        history.append((state_0.joint_q.numpy().copy(), state_0.joint_qd.numpy().copy()))
    return history


def _star_solver(model, overrides, **kwargs):
    """Construct a solver with a temporary kernel-selection override."""
    with kernel_overrides(overrides):
        return SolverSteel(model, solve_mode="matrix_free", **kwargs)


_DENSE_LOOP_KERNELS = {
    "cholesky_kernel": "loop",
    "trisolve_kernel": "loop",
    "hinv_jt_kernel": "par_row",
    "sparse_mass_matrix": False,
}


def test_fixed_base_sibling_branches_detect_diagonal_mass_topology(test, device):
    """Detect independent fixed-base branches, and keep coupled topologies off the diagonal path."""
    solver = SolverSteel(_build_fixed_base_star_model(device), solve_mode="matrix_free", articulation_constraint_max=16)
    test.assertEqual(solver._diagonal_mass_sizes, frozenset((16,)))
    test.assertTrue(solver._execution_plan.use_diagonal_mass(16))
    test.assertIsNone(solver._sparse_mass_matrix_size)
    # A serial chain couples every DOF, and pinned kernels disable the selection.
    chain = SolverSteel(
        _build_chain_model(device, num_links=3), solve_mode="matrix_free", articulation_constraint_max=16
    )
    test.assertEqual(chain._diagonal_mass_sizes, frozenset())
    pinned = _star_solver(
        _build_fixed_base_star_model(device), {"cholesky_kernel": "loop"}, articulation_constraint_max=16
    )
    test.assertEqual(pinned._diagonal_mass_sizes, frozenset())
    # The DOFs of one multi-DOF joint are coupled.
    ball = SolverSteel(_build_fixed_base_star_model(device, num_branches=4, ball_branch=True), solve_mode="matrix_free")
    test.assertEqual(ball._diagonal_mass_sizes, frozenset())


def test_fixed_base_sibling_branches_use_diagonal_mass_path(test, device):
    """Match the dense reference while selecting independent branch solves."""
    model = _build_fixed_base_star_model(device)
    optimized = SolverSteel(model, solve_mode="matrix_free", articulation_constraint_max=16)
    test.assertEqual(optimized._diagonal_mass_sizes, frozenset((16,)))
    test.assertTrue(optimized._execution_plan.use_diagonal_mass(16))
    test.assertIsNone(optimized._cholesky_kernels_by_size[16])
    test.assertIsNone(optimized._triangular_solve_kernels_by_size[16])
    test.assertIsNone(optimized._hinv_jt_kernels_by_size[16])

    reference = _star_solver(model, _DENSE_LOOP_KERNELS, articulation_constraint_max=16)
    test.assertFalse(reference._execution_plan.use_diagonal_mass(16))
    test.assertIsNone(reference._sparse_mass_matrix_size)

    for diagonal, dense in zip(_run_star(model, optimized, 5), _run_star(model, reference, 5), strict=True):
        np.testing.assert_allclose(diagonal[0], dense[0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(diagonal[1], dense[1], rtol=0.0, atol=2.0e-6)


def test_diagonal_mass_rows_match_dense_and_sparse_factors(test, device):
    """Match the dense loop kernels bitwise, and the sparse factors, with contact and limit rows."""
    # Point friction: the sparse factors compared against solve hard point-friction contacts.
    options = {
        "articulation_constraint_max": 64,
        "enable_joint_limits": True,
        "iterations": 8,
        "friction_anchor_beta": 0.0,
    }
    for with_chain in (False, True):
        with test.subTest(with_chain=with_chain):
            model = _build_fixed_base_star_model(
                device, num_branches=6, ground=True, armature=0.05, with_chain=with_chain
            )
            diagonal = SolverSteel(model, solve_mode="matrix_free", **options)
            test.assertTrue(diagonal._execution_plan.use_diagonal_mass(6))
            # With a second response group per world the responses are gathered per world.
            test.assertEqual(diagonal._hinv_jt_writes_world, with_chain)
            solvers = [diagonal, _star_solver(model, _DENSE_LOOP_KERNELS, **options)]
            if not with_chain:
                sparse = _star_solver(model, {"diagonal_mass": False}, **options)
                test.assertEqual(sparse._sparse_mass_matrix_size, 6)
                solvers.append(sparse)
            runs = [_run_star(model, solver, 60, contacts=True) for solver in solvers]
            test.assertGreater(int(diagonal._constraint_count.numpy().max()), 6, "no contact and limit rows")
            # The branches fall onto the ground; a frictionless fall would not stop them.
            test.assertLess(float(np.abs(runs[0][-1][1]).max()), 0.5)
            for step, (d, r) in enumerate(zip(runs[0], runs[1], strict=True)):
                # The diagonal kernels divide in the dense loop kernels' order.
                np.testing.assert_array_equal(d[0], r[0], err_msg=f"joint_q step {step}")
                np.testing.assert_array_equal(d[1], r[1], err_msg=f"joint_qd step {step}")
            if not with_chain:
                for step, (d, f) in enumerate(zip(runs[0], runs[2], strict=True)):
                    np.testing.assert_allclose(d[0], f[0], rtol=0.0, atol=1.0e-5, err_msg=f"sparse q step {step}")
                    np.testing.assert_allclose(d[1], f[1], rtol=0.0, atol=1.0e-4, err_msg=f"sparse qd step {step}")
            # The armature enters the factor: without it the trajectory differs.
            plain = _build_fixed_base_star_model(device, num_branches=6, ground=True, with_chain=with_chain)
            unarmed = _run_star(plain, SolverSteel(plain, solve_mode="matrix_free", **options), 60, contacts=True)
            test.assertGreater(float(np.abs(unarmed[5][1] - runs[0][5][1]).max()), 1.0e-6)


def test_sleeping_diagonal_dynamics_skip_is_exact(test, device):
    """Skip the diagonal-mass dynamics of sleeping stars without changing any published state.

    Two solvers differ only in whether sleeping islands skip their dynamics; over a sleep, a
    force wake and a resettle, every state array stays bitwise equal.
    """
    runs = []
    for skip in (False, True):
        model = _build_fixed_base_star_model(device, num_branches=3, ground=True)
        solver = SolverSteel(
            model, solve_mode="matrix_free", enable_sleeping=True, sleep_quiet_time=0.05, articulation_constraint_max=64
        )
        test.assertTrue(solver._execution_plan.use_diagonal_mass(3))
        solver._sleeping.skip_dynamics = skip
        pipeline = newton.CollisionPipeline(model)
        runs.append([solver, pipeline, pipeline.contacts(), model.state(), model.state(), model.control()])
    slept = woke = False
    for step in range(500):
        for run in runs:
            solver, pipeline, contacts, state, out, control = run
            state.clear_forces()
            if 300 <= step < 306:
                forces = state.body_f.numpy()
                forces[1, 2] = 50.0
                state.body_f.assign(forces)
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, 1.0 / 120.0)
            run[3], run[4] = out, state
        for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
            np.testing.assert_array_equal(
                getattr(runs[0][3], field).numpy(), getattr(runs[1][3], field).numpy(), err_msg=f"{step}: {field}"
            )
        awake = runs[1][0]._sleeping.art_awake.numpy()
        slept |= step < 300 and not awake.any()
        woke |= 300 <= step < 306 and bool(awake[0])
    test.assertTrue(slept, "the diagonal-mass articulations never slept")
    test.assertTrue(woke, "the external force did not wake the first articulation")
    for run in runs:
        run[0].check_constraint_capacity()


class TestSteelDiagonalMass(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


for _name, _func in (
    (
        "test_fixed_base_sibling_branches_detect_diagonal_mass_topology",
        test_fixed_base_sibling_branches_detect_diagonal_mass_topology,
    ),
    (
        "test_fixed_base_sibling_branches_use_diagonal_mass_path",
        test_fixed_base_sibling_branches_use_diagonal_mass_path,
    ),
    (
        "test_diagonal_mass_rows_match_dense_and_sparse_factors",
        test_diagonal_mass_rows_match_dense_and_sparse_factors,
    ),
    ("test_sleeping_diagonal_dynamics_skip_is_exact", test_sleeping_diagonal_dynamics_skip_is_exact),
):
    add_function_test(TestSteelDiagonalMass, _name, _func, devices=devices)


DT = 1.0 / 60.0


# Per-articulation initial pose/velocity: [base, left branch, right branch].
INITIAL_JOINT_Q = (0.7, 0.3, -0.4)


INITIAL_JOINT_QD = (0.5, -0.2, 0.3)


def _build_model(device, num_worlds=2, ground=True):
    """Two-branch pendulum: base revolute joint plus two sibling branch links.

    The sibling branches are not ancestor-related, so the mass matrix has structural zeros
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
    drive = {"axis": newton.Axis.Y, "target_ke": 15.0, "target_kd": 1.5}
    j_base = env.add_joint_revolute(
        -1, base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity()), **drive
    )
    j_left = env.add_joint_revolute(
        base,
        left,
        parent_xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        **drive,
    )
    j_right = env.add_joint_revolute(
        base,
        right,
        parent_xform=wp.transform(wp.vec3(-0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        **drive,
    )
    env.add_articulation([j_base, j_left, j_right])
    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


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
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()
    history = []
    for _ in range(num_steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        history.append(state_0.joint_q.numpy().copy())
    return np.stack(history)


def test_mass_refresh_cadence_follows_interval(test, device, solve_mode="matrix_free"):
    """Refresh every articulation's factorization on every ``interval``-th step only."""
    model = _build_model(device)
    solver = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=2)
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()
    for step_index, expected in enumerate(([1, 1], [0, 0], [1, 1], [0, 0])):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        test.assertEqual(solver._mass_update_mask.numpy().tolist(), expected, f"step {step_index}")


def test_stale_factorization_stays_close_to_interval_one(test, device, solve_mode="matrix_free"):
    """Keep an interval-2 contact trajectory close to the every-step refresh."""
    history = {}
    for interval in (1, 2):
        model = _build_model(device)
        solver = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=interval)
        history[interval] = _run_trajectory(model, solver, num_steps=60)
    test.assertTrue(np.isfinite(history[2]).all())
    # A stale factorization between refreshes is the intended trade-off, so the runs differ.
    test.assertLess(float(np.abs(history[2] - history[1]).max()), 0.05)
    moved = np.abs(history[1][-1] - np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), 2)).max()
    test.assertGreater(moved, 1.0e-3)


def test_model_change_request_refreshes_a_reuse_step(test, device, solve_mode="matrix_free"):
    """Refresh every articulation on a reuse step after a notification and consume the request."""
    model = _build_model(device, ground=False)
    solver = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=2)
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    state_0, state_1 = state_1, state_0
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    solver.step(state_0, state_1, control, None, DT)
    test.assertEqual(solver._mass_update_mask.numpy().tolist(), [1, 1])
    test.assertEqual(solver._mass_update_requested.numpy().tolist(), [0] * model.articulation_count)


def test_timestep_change_forces_a_refresh(test, device, solve_mode="matrix_free"):
    """Refresh the augmented mass matrix when the time step changes on a reuse step."""
    model = _build_model(device, ground=False)
    solver = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=4)
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    solver.step(state_1, state_0, control, None, DT)
    test.assertEqual(solver._mass_update_mask.numpy().tolist(), [0, 0])
    solver.step(state_0, state_1, control, None, 0.5 * DT)
    test.assertEqual(solver._mass_update_mask.numpy().tolist(), [1, 1])


def test_captured_two_substep_replay_matches_eager(test, device, solve_mode="matrix_free"):
    """Replay two captured substeps, including the first-step allocations, like eager stepping."""
    graph_model = _build_model(device, ground=False)
    eager_model = _build_model(device, ground=False)
    graph_solver = SolverSteel(graph_model, solve_mode=solve_mode, update_mass_matrix_interval=2)
    eager_solver = SolverSteel(eager_model, solve_mode=solve_mode, update_mass_matrix_interval=2)
    graph_a, graph_b = _make_initial_state(graph_model), graph_model.state()
    eager_a, eager_b = _make_initial_state(eager_model), eager_model.state()
    graph_control, eager_control = graph_model.control(), eager_model.control()

    def two_substeps(solver, state_in, state_out, control):
        for _ in range(2):
            solver.step(state_in, state_out, control, None, DT)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_in, name), getattr(state_out, name))

    # The very first launches of graph_solver happen inside the capture.
    with wp.ScopedCapture(device) as capture:
        two_substeps(graph_solver, graph_a, graph_b, graph_control)
    for _ in range(64):
        wp.capture_launch(capture.graph)
        two_substeps(eager_solver, eager_a, eager_b, eager_control)
    for name in ("joint_q", "joint_qd"):
        captured = getattr(graph_a, name).numpy()
        test.assertTrue(np.isfinite(captured).all(), f"{name} became non-finite under graph replay")
        np.testing.assert_allclose(captured, getattr(eager_a, name).numpy(), rtol=0.0, atol=2.0e-6, err_msg=name)


def test_mass_refresh_has_no_obsolete_limit_count_state(test, device, solve_mode="matrix_free"):
    """Keep no per-step joint-limit count state for the mass refresh."""
    solver = SolverSteel(_build_model(device, ground=False), solve_mode=solve_mode)
    for attribute in ("aug_limit_counts", "aug_prev_limit_counts", "limit_change_mask"):
        test.assertFalse(hasattr(solver, attribute), f"obsolete mass-refresh state {attribute!r} was restored")


class TestSteelMassUpdateInterval(unittest.TestCase):
    pass


for _name in (
    "test_mass_refresh_cadence_follows_interval",
    "test_stale_factorization_stays_close_to_interval_one",
    "test_model_change_request_refreshes_a_reuse_step",
    "test_timestep_change_forces_a_refresh",
    "test_captured_two_substep_replay_matches_eager",
    "test_mass_refresh_has_no_obsolete_limit_count_state",
):
    add_function_test(TestSteelMassUpdateInterval, _name, globals()[_name], devices=devices)
    split_devices = devices if _name == "test_captured_two_substep_replay_matches_eager" else get_test_devices()
    add_function_test(
        TestSteelMassUpdateInterval, f"{_name}_split", globals()[_name], devices=split_devices, solve_mode="split"
    )


INITIAL_JOINT_Q_notify_inertial = 0.3


NEW_COM = (0.15, 0.0, 0.05)


def _build_model_notify_inertial(device, com=None):
    """Single-link pendulum on a Y-axis revolute joint, box COM at the pivot.

    With the COM at the pivot gravity exerts no torque, so any swing after a COM change is
    attributable to the change alone.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 1000.0
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.25, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = INITIAL_JOINT_Q_notify_inertial
    model = builder.finalize(device=device)
    if com is not None:
        body_com = model.body_com.numpy()
        body_com[0] = com
        model.body_com.assign(body_com)
    return model


def _run_trajectory_notify_inertial(model, solver, num_steps):
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()
    history = []
    for _ in range(num_steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        history.append(state_0.joint_q.numpy().copy())
    return np.stack(history)


def test_stepped_solver_releases_resources_without_cyclic_gc(test, device, solve_mode="matrix_free"):
    """Release a stepped solver without relying on the cyclic garbage collector."""
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        model = _build_model_notify_inertial(device)
        solver = SolverSteel(model, solve_mode=solve_mode)
        reference = weakref.ref(solver)
        solver.step(model.state(), model.state(), model.control(), None, DT)
        del solver
        test.assertIsNone(reference(), "stepping must not create a solver ownership cycle")
    finally:
        if gc_enabled:
            gc.enable()


def test_step_refreshes_body_pose_after_generalized_coordinate_update(test, device, solve_mode="matrix_free"):
    """Derive body poses from joint_q in step, so a direct joint_q write needs no caller-side FK."""
    outputs = {}
    for caller_refreshes_fk in (False, True):
        model = _build_model_notify_inertial(device, com=NEW_COM)
        state_0 = model.state()
        state_1 = model.state()
        joint_q = state_0.joint_q.numpy()
        joint_q[0] = 1.1
        state_0.joint_q.assign(joint_q)
        stale_body_q = state_0.body_q.numpy().copy()
        if caller_refreshes_fk:
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        SolverSteel(model, solve_mode=solve_mode).step(state_0, state_1, model.control(), None, DT)
        outputs[caller_refreshes_fk] = {
            "joint_q": state_1.joint_q.numpy().copy(),
            "joint_qd": state_1.joint_qd.numpy().copy(),
            "body_q": state_0.body_q.numpy().copy(),
            "stale_body_q": stale_body_q,
        }
    test.assertFalse(np.allclose(outputs[False]["stale_body_q"], outputs[True]["body_q"]))
    for key in ("body_q", "joint_q", "joint_qd"):
        np.testing.assert_allclose(outputs[False][key], outputs[True][key], rtol=0.0, atol=1.0e-6)


def test_step_refreshes_reused_state_after_joint_q_write(test, device, solve_mode="matrix_free"):
    """Pick up a joint_q write into a state the solver produced on the previous step."""
    model = _build_model_notify_inertial(device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    joint_q = state_1.joint_q.numpy()
    joint_q[0] = -0.9
    state_1.joint_q.assign(joint_q)
    solver.step(state_1, state_0, control, None, DT)

    reference_model = _build_model_notify_inertial(device)
    reference_state = reference_model.state()
    reference_state.joint_q.assign(state_1.joint_q)
    reference_state.joint_qd.assign(state_1.joint_qd)
    reference_out = reference_model.state()
    SolverSteel(reference_model, solve_mode=solve_mode).step(
        reference_state, reference_out, reference_model.control(), None, DT
    )
    np.testing.assert_allclose(state_0.joint_q.numpy(), reference_out.joint_q.numpy(), rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(state_0.body_q.numpy(), reference_out.body_q.numpy(), rtol=0.0, atol=1.0e-6)


def test_notify_refreshes_baked_com_and_inertia_buffers(test, device, solve_mode="matrix_free"):
    """Re-derive the solver's COM and inertia buffers on BODY_INERTIAL_PROPERTIES."""
    model = _build_model_notify_inertial(device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    stale_X_com = solver._body_X_com.numpy().copy()
    stale_I_m = solver._body_I_m.numpy().copy()

    body_com = model.body_com.numpy()
    body_com[0] = NEW_COM
    model.body_com.assign(body_com)
    body_mass = model.body_mass.numpy()
    body_mass[0] *= 2.0
    model.body_mass.assign(body_mass)

    np.testing.assert_array_equal(solver._body_X_com.numpy(), stale_X_com)
    solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
    np.testing.assert_allclose(
        solver._body_X_com.numpy()[0][:3], np.asarray(NEW_COM, dtype=np.float32), rtol=0.0, atol=0.0
    )
    test.assertFalse(np.allclose(solver._body_I_m.numpy(), stale_I_m))
    test.assertEqual(solver._mass_update_requested.numpy()[0], 1)


def test_com_change_with_notify_matches_freshly_built_solver(test, device, solve_mode="matrix_free"):
    """Reproduce a freshly built solver's dynamics after a mid-run COM change and notification."""
    pre_steps, post_steps = 30, 60
    reference_model = _build_model_notify_inertial(device, com=NEW_COM)
    reference = _run_trajectory_notify_inertial(
        reference_model, SolverSteel(reference_model, solve_mode=solve_mode), post_steps
    )

    histories = {}
    for notify in (True, False):
        model = _build_model_notify_inertial(device)
        solver = SolverSteel(model, solve_mode=solve_mode)
        _run_trajectory_notify_inertial(model, solver, pre_steps)
        body_com = model.body_com.numpy()
        body_com[0] = NEW_COM
        model.body_com.assign(body_com)
        if notify:
            solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
        histories[notify] = _run_trajectory_notify_inertial(model, solver, post_steps)

    np.testing.assert_allclose(histories[True], reference, rtol=0.0, atol=1.0e-5)
    stale_drift = np.abs(histories[False] - reference).max()
    test.assertGreater(stale_drift, 1.0e-2, "a solver that was not notified should diverge")


def test_kinematic_flag_change_is_picked_up(test, device, solve_mode="matrix_free"):
    """Hold a body still once it is flagged kinematic and the solver is notified."""
    model = _build_model_notify_inertial(device, com=NEW_COM)
    solver = SolverSteel(model, solve_mode=solve_mode)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    flags = model.body_flags.numpy()
    flags[0] = int(newton.BodyFlags.KINEMATIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
    for _ in range(30):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    test.assertAlmostEqual(float(state_0.joint_q.numpy()[0]), INITIAL_JOINT_Q_notify_inertial, places=5)


def _shift_child_frame(model):
    joint_X_c = model.joint_X_c.numpy()
    joint_X_c[0, 0] = 0.5
    model.joint_X_c.assign(joint_X_c)


def test_joint_frame_change_with_notify_matches_freshly_built_solver(test, device, solve_mode="matrix_free"):
    """Refresh cached mass factors on JOINT_PROPERTIES, eagerly and under graph replay."""
    interval = 100
    reference_model = _build_model_notify_inertial(device)
    _shift_child_frame(reference_model)
    reference = _run_trajectory_notify_inertial(
        reference_model,
        SolverSteel(reference_model, solve_mode=solve_mode, update_mass_matrix_interval=interval),
        5,
    )

    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            model = _build_model_notify_inertial(device)
            solver = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=interval)
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()

            def substep(solver=solver, state_0=state_0, state_1=state_1, control=control):
                solver.step(state_0, state_1, control, None, DT)
                wp.copy(state_0.joint_q, state_1.joint_q)
                wp.copy(state_0.joint_qd, state_1.joint_qd)

            # Factor the original frame on the first step; later steps reuse it until notified.
            substep()
            initial_q = model.joint_q.numpy().copy()
            graph = None
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    substep()
            state_0.joint_q.assign(initial_q)
            state_0.joint_qd.zero_()
            _shift_child_frame(model)
            solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
            history = []
            for _ in range(5):
                if graph is None:
                    substep()
                else:
                    wp.capture_launch(graph.graph)
                history.append(state_0.joint_q.numpy().copy())
            np.testing.assert_allclose(np.asarray(history), reference, rtol=0.0, atol=1.0e-5)


def _check_damping_edit_matches_fresh_solver(test, device, replace: bool, solve_mode: str):
    """Edit joint damping after a step, notify, and compare the next eager or captured step to a fresh solver."""
    # Graph capture needs a CUDA device.
    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(-1, link, axis=newton.Axis.X, damping=1.0, armature=0.0)
            builder.add_articulation([joint])
            model = builder.finalize(device=device)
            solver = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.fill_(1.0)
            control = model.control()
            solver.step(state_0, state_1, control, None, 0.01)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state_0, state_1, control, None, 0.01)
            if replace:
                model.joint_damping = wp.array([3.0], dtype=wp.float32, device=device)
            else:
                model.joint_damping.assign([3.0])
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                solver.step(state_0, state_1, control, None, 0.01)

            fresh = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
            fresh_out = model.state()
            fresh.step(state_0, fresh_out, control, None, 0.01)
            np.testing.assert_allclose(fresh._joint_tau.numpy(), [-3.0], atol=1.0e-6)
            np.testing.assert_allclose(solver._joint_tau.numpy(), fresh._joint_tau.numpy(), atol=1.0e-6)
            np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-6)


def test_replaced_joint_damping_matches_fresh_solver(test, device, solve_mode="matrix_free"):
    """Read a joint damping array replaced on the model at the next JOINT_DOF_PROPERTIES notify."""
    _check_damping_edit_matches_fresh_solver(test, device, replace=True, solve_mode=solve_mode)


def test_assigned_joint_damping_matches_fresh_solver(test, device, solve_mode="matrix_free"):
    """Read joint damping modified in place at the next JOINT_DOF_PROPERTIES notify."""
    _check_damping_edit_matches_fresh_solver(test, device, replace=False, solve_mode=solve_mode)


def _check_joint_dof_edit_matches_fresh_solver(test, device, name: str, value: float, flag, solve_mode: str):
    """Assign one joint DOF property after a step, notify ``flag``, and compare the next step to a fresh solver."""
    # Graph capture needs a CUDA device.
    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(-1, link, axis=newton.Axis.X, target_kd=10.0, armature=0.0)
            builder.add_articulation([joint])
            model = builder.finalize(device=device)
            solver = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.fill_(1.0)
            control = model.control()
            solver.step(state_0, state_1, control, None, 0.01)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state_0, state_1, control, None, 0.01)
            stale_qd = state_1.joint_qd.numpy().copy()
            getattr(model, name).assign([value])
            solver.notify_model_changed(flag)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                solver.step(state_0, state_1, control, None, 0.01)

            fresh = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
            fresh_out = model.state()
            fresh.step(state_0, fresh_out, control, None, 0.01)
            test.assertFalse(np.allclose(fresh_out.joint_qd.numpy(), stale_qd, atol=1.0e-4))
            np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-6)


def test_dof_force_flag_refreshes_drive_gains(test, device, solve_mode="matrix_free"):
    """Fold a drive damping edit into the mass matrix on a JOINT_DOF_FORCE_PROPERTIES notify."""
    _check_joint_dof_edit_matches_fresh_solver(
        test, device, "joint_target_kd", 40.0, ModelFlags.JOINT_DOF_FORCE_PROPERTIES, solve_mode
    )


def test_dof_force_flag_refreshes_joint_damping(test, device, solve_mode="matrix_free"):
    """Read joint damping at the next JOINT_DOF_FORCE_PROPERTIES notify."""
    _check_joint_dof_edit_matches_fresh_solver(
        test, device, "joint_damping", 20.0, ModelFlags.JOINT_DOF_FORCE_PROPERTIES, solve_mode
    )


def test_dof_inertial_flag_refreshes_armature(test, device, solve_mode="matrix_free"):
    """Fold an armature edit into the mass matrix on a JOINT_DOF_INERTIAL_PROPERTIES notify."""
    _check_joint_dof_edit_matches_fresh_solver(
        test, device, "joint_armature", 1.0, ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES, solve_mode
    )


def _check_friction_edit_matches_fresh_solver(test, device, replace: bool, solve_mode: str):
    """Edit shape friction after a step, notify, and compare the next eager or captured step to a fresh solver."""
    # Graph capture needs a CUDA device.
    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            builder = newton.ModelBuilder()
            builder.default_shape_cfg.mu = 0.0
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize(device=device)
            solver = SolverSteel(model, solve_mode=solve_mode)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.assign([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            pipeline.collide(state_0, contacts)
            control = model.control()
            solver.step(state_0, state_1, control, contacts, DT)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state_0, state_1, control, contacts, DT)
            mu = np.full(model.shape_count, 1.0, dtype=np.float32)
            if replace:
                model.shape_material_mu = wp.array(mu, dtype=wp.float32, device=device)
            else:
                model.shape_material_mu.assign(mu)
            solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                solver.step(state_0, state_1, control, contacts, DT)

            fresh = SolverSteel(model, solve_mode=solve_mode)
            fresh_out = model.state()
            fresh.step(state_0, fresh_out, control, contacts, DT)
            test.assertLess(float(fresh_out.joint_qd.numpy()[0]), 0.95)
            np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-5)


def test_replaced_shape_friction_matches_fresh_solver(test, device, solve_mode="matrix_free"):
    """Read a friction array replaced on the model at the next SHAPE_PROPERTIES notify."""
    _check_friction_edit_matches_fresh_solver(test, device, replace=True, solve_mode=solve_mode)


def test_assigned_shape_friction_matches_fresh_solver(test, device, solve_mode="matrix_free"):
    """Read friction modified in place at the next SHAPE_PROPERTIES notify."""
    _check_friction_edit_matches_fresh_solver(test, device, replace=False, solve_mode=solve_mode)


class TestSteelNotifyInertial(unittest.TestCase):
    pass


for _name in (
    "test_stepped_solver_releases_resources_without_cyclic_gc",
    "test_step_refreshes_body_pose_after_generalized_coordinate_update",
    "test_step_refreshes_reused_state_after_joint_q_write",
    "test_notify_refreshes_baked_com_and_inertia_buffers",
    "test_com_change_with_notify_matches_freshly_built_solver",
    "test_kinematic_flag_change_is_picked_up",
    "test_joint_frame_change_with_notify_matches_freshly_built_solver",
    "test_replaced_joint_damping_matches_fresh_solver",
    "test_assigned_joint_damping_matches_fresh_solver",
    "test_dof_force_flag_refreshes_drive_gains",
    "test_dof_force_flag_refreshes_joint_damping",
    "test_dof_inertial_flag_refreshes_armature",
    "test_replaced_shape_friction_matches_fresh_solver",
    "test_assigned_shape_friction_matches_fresh_solver",
):
    add_function_test(TestSteelNotifyInertial, _name, globals()[_name], devices=devices)
    add_function_test(
        TestSteelNotifyInertial, f"{_name}_split", globals()[_name], devices=get_test_devices(), solve_mode="split"
    )


NOTIFY_WORLD_COUNT = 3


# Shape fields whose change retires friction-anchor history.
GEOMETRY_FIELDS = (
    "shape_body",
    "shape_transform",
    "shape_scale",
    "shape_type",
    "shape_source_ptr",
    "shape_margin",
    "shape_is_solid",
)


def _build_model_notify_device(device, ground=False):
    """Worlds with a free sphere, a 3-link and a 2-link revolute chain: three response sizes."""
    builder = newton.ModelBuilder()
    for world in range(NOTIFY_WORLD_COUNT):
        builder.begin_world()
        sphere = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_sphere(sphere, radius=0.1)
        for link_count, x in ((3, 0.0), (2, -1.0)):
            parent, joints = -1, []
            for link_index in range(link_count):
                link = builder.add_link()
                builder.add_shape_capsule(
                    link, radius=0.05, half_height=0.1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
                )
                joint_z = 0.5 if parent < 0 else 0.2
                joints.append(
                    builder.add_joint_revolute(
                        parent,
                        link,
                        parent_xform=wp.transform(wp.vec3(x if parent < 0 else 0.0, 0.0, joint_z), wp.quat_identity()),
                        axis=newton.Axis.Y,
                        armature=0.01 * (world + link_index + 1),
                        damping=0.1,
                    )
                )
                parent = link
            builder.add_articulation(joints)
        builder.end_world()
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _host_effective_armature(model):
    """Host reference of the effective armature: kinematic joints take a large armature."""
    armature = model.joint_armature.numpy().copy()
    kinematic = (model.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC)) != 0
    joint_qd_start = model.joint_qd_start.numpy()
    for joint, child in enumerate(model.joint_child.numpy()):
        if kinematic[child]:
            armature[joint_qd_start[joint] : joint_qd_start[joint + 1]] = 1.0e10
    return armature


def _assert_armature_matches(test, solver, fresh):
    """Compare the effective armature and its size groups with the host rebuild and a fresh solver."""
    armature = _host_effective_armature(solver.model)
    np.testing.assert_array_equal(solver._joint_armature_device.numpy(), armature)
    np.testing.assert_array_equal(solver._joint_armature_device.numpy(), fresh._joint_armature_device.numpy())
    plan = solver._model_plan
    test.assertEqual(len(solver._size_groups), 3)
    for size in solver._size_groups:
        expected = np.zeros((solver._n_arts_by_size[size], size), dtype=np.float32)
        for group, art in enumerate(np.flatnonzero(plan.response_dof_count == size)):
            start, count = plan.articulation_dof_start[art], plan.articulation_dof_count[art]
            expected[group, :count] = armature[start : start + count]
        np.testing.assert_array_equal(solver._R_by_size[size].numpy(), expected)
        np.testing.assert_array_equal(solver._R_by_size[size].numpy(), fresh._R_by_size[size].numpy())


def _randomize_history(patches, model, rng):
    """Fill the previous patch frame with valid anchors on random shapes and bodies, including -1 sentinels."""
    n = patches.previous.valid.shape[0]
    patches.previous.valid.fill_(1)
    for field, count in (("shape", model.shape_count), ("body", model.body_count)):
        for side in ("a", "b"):
            getattr(patches.previous, f"{field}_{side}").assign(rng.integers(-1, count, n).astype(np.int32))


def _host_update_geometry(model, patches, seen, valid):
    """Host reference of the friction-anchor geometry refresh: body radii and retired history."""
    geometry = {name: getattr(model, name).numpy().copy() for name in GEOMETRY_FIELDS}
    radii = np.zeros(model.body_count, dtype=np.float32)
    for body, radius, transform in zip(
        geometry["shape_body"], model.shape_collision_radius.numpy(), geometry["shape_transform"], strict=True
    ):
        if body >= 0:
            radii[body] = max(radii[body], radius + np.linalg.norm(transform[:3]))
    changed = np.zeros(model.shape_count, dtype=bool)
    for name, values in geometry.items():
        difference = values != seen[name]
        if difference.ndim > 1:
            difference = np.any(difference, axis=tuple(range(1, difference.ndim)))
        changed |= difference
    bodies = np.zeros(model.body_count, dtype=bool)
    for shape_body in (seen["shape_body"], geometry["shape_body"]):
        affected = shape_body[changed]
        bodies[affected[affected >= 0]] = True
    valid = valid.copy()
    previous = patches.previous
    for ids, mask in (
        (previous.shape_a.numpy(), changed),
        (previous.shape_b.numpy(), changed),
        (previous.body_a.numpy(), bodies),
        (previous.body_b.numpy(), bodies),
    ):
        hit = ids >= 0
        valid[hit & mask[np.maximum(ids, 0)]] = 0
    return radii, valid, geometry


def test_armature_notify_matches_host_reference_and_fresh_solver(test, device, solve_mode="matrix_free"):
    """Refresh the effective armature and its size groups like a host rebuild, from edited and replaced arrays."""
    model = _build_model_notify_device(device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    rng = np.random.default_rng(0)
    for replace, flag in ((False, ModelFlags.JOINT_DOF_PROPERTIES), (True, ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES)):
        with test.subTest(replace=replace):
            values = rng.uniform(0.0, 0.5, model.joint_dof_count).astype(np.float32)
            if replace:
                model.joint_armature = wp.array(values, dtype=wp.float32, device=device)
            else:
                model.joint_armature.assign(values)
            solver.notify_model_changed(flag)
            _assert_armature_matches(test, solver, SolverSteel(model, solve_mode=solve_mode))


def test_kinematic_notify_matches_host_reference_and_fresh_solver(test, device, solve_mode="matrix_free"):
    """Give newly kinematic joints the large armature and drop it when they become dynamic again."""
    model = _build_model_notify_device(device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    model.joint_armature.fill_(0.25)
    chain_link = int(model.joint_child.numpy()[1])
    for flag in (newton.BodyFlags.KINEMATIC, newton.BodyFlags.DYNAMIC):
        with test.subTest(flag=flag):
            body_flags = model.body_flags.numpy()
            body_flags[chain_link] = int(flag)
            model.body_flags.assign(body_flags)
            solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
            _assert_armature_matches(test, solver, SolverSteel(model, solve_mode=solve_mode))


def test_shape_notify_matches_host_reference(test, device, solve_mode="matrix_free"):
    """Refresh body radii and retire exactly the anchors whose geometry changed."""
    model = _build_model_notify_device(device)
    solver = SolverSteel(model, solve_mode=solve_mode, friction_anchor_beta=0.2)
    patches = solver._friction_patches
    rng = np.random.default_rng(1)
    seen = {name: getattr(model, name).numpy().copy() for name in GEOMETRY_FIELDS}
    for edit in ("shape_transform", "shape_scale", "shape_margin", "shape_body", "shape_collision_radius", None):
        with test.subTest(edit=edit):
            _randomize_history(patches, model, rng)
            valid_before = patches.previous.valid.numpy()
            shape = int(rng.integers(model.shape_count))
            if edit == "shape_body":
                values = model.shape_body.numpy()
                values[shape] = (values[shape] + 1) % model.body_count
                model.shape_body.assign(values)
            elif edit is not None:
                values = getattr(model, edit).numpy()
                values[shape] = values[shape] * 1.5 + 0.01
                getattr(model, edit).assign(values)
            solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
            expected_radius, expected_valid, seen = _host_update_geometry(model, patches, seen, valid_before)
            np.testing.assert_array_equal(patches.body_radius.numpy(), expected_radius)
            np.testing.assert_array_equal(patches.previous.valid.numpy(), expected_valid)
            if edit not in (None, "shape_collision_radius"):
                test.assertLess(np.count_nonzero(expected_valid), np.count_nonzero(valid_before))
    test.assertGreater(np.count_nonzero(patches.previous.valid.numpy()), 0)


def test_world_mask_limits_warmstart_invalidation(test, device, solve_mode="matrix_free"):
    """Discard warm-start impulses of the masked worlds only, keep overflow status, and validate the mask."""
    model = _build_model_notify_device(device)
    solver = SolverSteel(model, solve_mode=solve_mode, contact_warmstart=True)
    histories = (solver._ws_prev_impulses, solver._ws_prev_mf_impulses)
    test.assertEqual(histories[0].shape[0], NOTIFY_WORLD_COUNT)
    cases = (
        ([False, True, False, False], [False, True, False]),
        ([True, False, True, False], [True, False, True]),
        (None, [True, True, True]),
    )
    for flag in (ModelFlags.JOINT_DOF_PROPERTIES, ModelFlags.SHAPE_PROPERTIES):
        for mask, cleared in cases:
            with test.subTest(flag=flag, mask=mask):
                for history in histories:
                    history.fill_(7.0)
                solver.constraint_overflow.fill_(True)
                world_mask = None if mask is None else wp.array(mask, dtype=wp.bool, device=device)
                solver.notify_model_changed(flag, world_mask=world_mask)
                for history in histories:
                    expected = np.where(np.asarray(cleared)[:, None], 0.0, 7.0)
                    np.testing.assert_array_equal(history.numpy(), np.broadcast_to(expected, history.shape))
                test.assertTrue(solver.constraint_overflow.numpy().all())
    for history in histories:
        history.fill_(7.0)
    with test.assertRaisesRegex(ValueError, r"model\.world_count \+ 1 \(4\)"):
        solver.notify_model_changed(
            ModelFlags.JOINT_DOF_PROPERTIES, world_mask=wp.ones(NOTIFY_WORLD_COUNT, dtype=wp.bool, device=device)
        )
    for history in histories:
        test.assertTrue(np.all(history.numpy() == 7.0))


class TestSteelNotifyDevice(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_armature_and_shape_notify_capture_into_a_graph(self):
        """Replay a captured notify and pick up armature and geometry written in place after the capture."""
        device = wp.get_device("cuda:0")
        model = _build_model_notify_device(device)
        solver = SolverSteel(model, friction_anchor_beta=0.2)
        patches = solver._friction_patches
        flags = (
            ModelFlags.JOINT_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
        )
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(flags)
        seen = {name: getattr(model, name).numpy().copy() for name in GEOMETRY_FIELDS}
        rng = np.random.default_rng(2)
        _randomize_history(patches, model, rng)
        valid_before = patches.previous.valid.numpy()
        model.joint_armature.assign(rng.uniform(0.0, 0.5, model.joint_dof_count).astype(np.float32))
        model.shape_collision_radius.assign(model.shape_collision_radius.numpy() * 2.0)
        scale = model.shape_scale.numpy()
        scale[2] *= 1.25
        model.shape_scale.assign(scale)
        wp.capture_launch(capture.graph)
        _assert_armature_matches(self, solver, SolverSteel(model))
        expected_radius, expected_valid, _ = _host_update_geometry(model, patches, seen, valid_before)
        np.testing.assert_array_equal(patches.body_radius.numpy(), expected_radius)
        np.testing.assert_array_equal(patches.previous.valid.numpy(), expected_valid)
        self.assertLess(np.count_nonzero(expected_valid), np.count_nonzero(valid_before))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_narrow_dof_flags_and_material_copies_capture_into_a_graph(self):
        """Replay a captured narrow-flag notify and pick up armature, damping and materials edited in place."""
        device = wp.get_device("cuda:0")
        model = _build_model_notify_device(device)
        solver = SolverSteel(model)
        flags = (
            ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
            | ModelFlags.JOINT_DOF_FORCE_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
        )
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(flags)
        rng = np.random.default_rng(3)
        model.joint_armature.assign(rng.uniform(0.0, 0.5, model.joint_dof_count).astype(np.float32))
        damping = rng.uniform(0.1, 1.0, model.joint_dof_count).astype(np.float32)
        model.joint_damping.assign(damping)
        materials = {"shape_material_mu": solver._shape_mu, "shape_material_restitution": solver._shape_restitution}
        edited = {}
        for name in materials:
            edited[name] = rng.uniform(0.01, 0.9, model.shape_count).astype(np.float32)
            getattr(model, name).assign(edited[name])
        solver._mass_update_requested.zero_()
        wp.capture_launch(capture.graph)
        _assert_armature_matches(self, solver, SolverSteel(model))
        np.testing.assert_array_equal(solver._passive_joint_damping.numpy()[: model.joint_dof_count], damping)
        for name, buffer in materials.items():
            np.testing.assert_array_equal(buffer.numpy()[: model.shape_count], edited[name], err_msg=name)
        self.assertTrue(np.all(solver._mass_update_requested.numpy() == 1))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_captured_notify_reads_its_captured_arrays(self):
        """Keep reading the captured armature array after a replacement until an eager notify."""
        device = wp.get_device("cuda:0")
        model = _build_model_notify_device(device)
        solver = SolverSteel(model)
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        captured = model.joint_armature
        captured.fill_(0.125)
        model.joint_armature = wp.full(model.joint_dof_count, 0.375, dtype=wp.float32, device=device)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(solver._joint_armature_device.numpy(), np.float32(0.125))
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        _assert_armature_matches(self, solver, SolverSteel(model))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_captured_world_mask_follows_the_live_mask(self):
        """Replay a captured masked notify against the mask values written before each replay."""
        device = wp.get_device("cuda:0")
        model = _build_model_notify_device(device)
        solver = SolverSteel(model, contact_warmstart=True)
        mask = wp.zeros(NOTIFY_WORLD_COUNT + 1, dtype=wp.bool, device=device)
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES, world_mask=mask)
        for selected in range(NOTIFY_WORLD_COUNT):
            values = np.zeros(NOTIFY_WORLD_COUNT + 1, dtype=bool)
            values[selected] = True
            mask.assign(values)
            solver._ws_prev_mf_impulses.fill_(7.0)
            wp.capture_launch(capture.graph)
            history = solver._ws_prev_mf_impulses.numpy()
            self.assertTrue(np.all(history[selected] == 0.0))
            self.assertTrue(np.all(np.delete(history, selected, axis=0) == 7.0))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_captured_notify_steps_like_eager_notify(self):
        """Step bitwise like an eager notify after a captured notify of in-place model edits."""
        device = wp.get_device("cuda:0")
        trajectories = []
        for capture_notify in (False, True):
            model = _build_model_notify_device(device, ground=True)
            solver = SolverSteel(
                model, friction_anchor_beta=0.2, contact_warmstart=True, articulation_constraint_max=64
            )
            flags = ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.SHAPE_PROPERTIES
            graph = None
            if capture_notify:
                with wp.ScopedCapture(device=device) as graph:
                    solver.notify_model_changed(flags)
            pipeline = newton.CollisionPipeline(model, contact_matching="latest")
            contacts = pipeline.contacts()
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            history = []
            for step in range(40):
                if step == 20:
                    model.joint_armature.fill_(0.2)
                    model.shape_material_mu.fill_(0.2)
                    scale = model.shape_scale.numpy()
                    scale[0] *= 1.1
                    model.shape_scale.assign(scale)
                    if graph is None:
                        solver.notify_model_changed(flags)
                    else:
                        wp.capture_launch(graph.graph)
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, DT)
                state_0, state_1 = state_1, state_0
                history.append(state_0.joint_q.numpy().copy())
            trajectories.append(np.stack(history))
        np.testing.assert_array_equal(trajectories[1], trajectories[0])
        self.assertFalse(np.array_equal(trajectories[0][19], trajectories[0][39]))


for _name in (
    "test_armature_notify_matches_host_reference_and_fresh_solver",
    "test_kinematic_notify_matches_host_reference_and_fresh_solver",
    "test_shape_notify_matches_host_reference",
    "test_world_mask_limits_warmstart_invalidation",
):
    add_function_test(TestSteelNotifyDevice, _name, globals()[_name], devices=devices)
# Friction anchors and warm start require the matrix-free solve.
for _name in (
    "test_armature_notify_matches_host_reference_and_fresh_solver",
    "test_kinematic_notify_matches_host_reference_and_fresh_solver",
):
    add_function_test(
        TestSteelNotifyDevice, f"{_name}_split", globals()[_name], devices=get_test_devices(), solve_mode="split"
    )


def test_tiled_response_diagonal_matches_dense_reference(test, device):
    """Match the tiled world response and accumulated diagonal to a dense solve, with two articulations per world."""
    device = wp.get_device(device)
    rng = np.random.default_rng(41)
    num_dofs = 23
    max_constraints = 16
    num_articulations = 4
    world_constraint_count_np = np.array((13, 7), dtype=np.int32)
    articulation_world_np = np.array((0, 0, 1, 1), dtype=np.int32)
    articulation_world_dof_offset_np = np.array((0, num_dofs, 0, num_dofs), dtype=np.int32)

    factors = rng.normal(size=(num_articulations, num_dofs, num_dofs)).astype(np.float32)
    mass = factors @ np.transpose(factors, (0, 2, 1))
    mass += 2.0 * np.eye(num_dofs, dtype=np.float32)[None, :, :]
    cholesky = wp.array(np.linalg.cholesky(mass).astype(np.float32), device=device)

    jacobian_np = np.zeros((num_articulations, max_constraints, num_dofs), dtype=np.float32)
    for articulation, world in enumerate(articulation_world_np):
        count = int(world_constraint_count_np[world])
        jacobian_np[articulation, :count] = rng.normal(size=(count, num_dofs))
    jacobian = wp.array(jacobian_np, device=device)

    group_to_art = wp.array(np.arange(num_articulations, dtype=np.int32), device=device)
    art_to_world = wp.array(articulation_world_np, device=device)
    articulation_world_dof_offset = wp.array(articulation_world_dof_offset_np, device=device)
    constraint_count = wp.array(world_constraint_count_np, device=device)
    response = wp.zeros_like(jacobian)
    world_jacobian = wp.zeros((2, max_constraints, 2 * num_dofs), dtype=wp.float32, device=device)
    world_response = wp.zeros_like(world_jacobian)
    group_diag = wp.zeros((num_articulations, max_constraints), dtype=wp.float32, device=device)

    kernel = _get_hinv_jt_kernel(
        num_dofs,
        max_constraints,
        str(device.arch),
        constraint_chunk_size=8,
        write_world=True,
        write_group=False,
        compute_diag=True,
    )
    wp.launch_tiled(
        kernel,
        dim=(num_articulations, 2),
        inputs=[
            cholesky,
            jacobian,
            group_to_art,
            art_to_world,
            articulation_world_dof_offset,
            constraint_count,
        ],
        outputs=[response, world_jacobian, world_response, group_diag],
        block_dim=64,
        device=device,
    )

    world_diag = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
    wp.launch(
        accumulate_group_diag_worlds,
        dim=2 * max_constraints,
        inputs=[
            group_diag,
            wp.array((0, 2, 4), dtype=wp.int32, device=device),
            group_to_art,
            group_to_art,
            constraint_count,
            max_constraints,
        ],
        outputs=[world_diag],
        device=device,
    )

    response_ref = np.zeros_like(jacobian_np)
    group_diag_ref = np.zeros((num_articulations, max_constraints), dtype=np.float32)
    world_jacobian_ref = np.zeros((2, max_constraints, 2 * num_dofs), dtype=np.float32)
    world_response_ref = np.zeros_like(world_jacobian_ref)
    for articulation, world in enumerate(articulation_world_np):
        count = int(world_constraint_count_np[world])
        response_ref[articulation, :count] = np.linalg.solve(mass[articulation], jacobian_np[articulation, :count].T).T
        group_diag_ref[articulation, :count] = np.sum(
            jacobian_np[articulation, :count] * response_ref[articulation, :count], axis=1
        )
        dof_offset = articulation_world_dof_offset_np[articulation]
        world_jacobian_ref[world, :count, dof_offset : dof_offset + num_dofs] = jacobian_np[articulation, :count]
        world_response_ref[world, :count, dof_offset : dof_offset + num_dofs] = response_ref[articulation, :count]
    world_diag_ref = np.stack((group_diag_ref[:2].sum(axis=0), group_diag_ref[2:].sum(axis=0)))

    np.testing.assert_array_equal(response.numpy(), np.zeros_like(response_ref))
    np.testing.assert_allclose(world_jacobian.numpy(), world_jacobian_ref, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(world_response.numpy(), world_response_ref, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(group_diag.numpy(), group_diag_ref, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(world_diag.numpy(), world_diag_ref, rtol=2.0e-5, atol=2.0e-6)


class TestSteelResponseDiagonal(unittest.TestCase):
    pass


add_function_test(
    TestSteelResponseDiagonal,
    "test_tiled_response_diagonal_matches_dense_reference",
    test_tiled_response_diagonal_matches_dense_reference,
    devices=get_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
