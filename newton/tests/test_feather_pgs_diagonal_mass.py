# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagonal mass matrices of SolverFeatherPGS: detection and agreement with the factor paths."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


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
    try:
        SolverFeatherPGS._kernel_overrides = overrides
        return SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs)
    finally:
        SolverFeatherPGS._kernel_overrides = {}


_DENSE_LOOP_KERNELS = {
    "cholesky_kernel": "loop",
    "trisolve_kernel": "loop",
    "hinv_jt_kernel": "par_row",
    "sparse_mass_matrix": False,
}


def test_fixed_base_sibling_branches_detect_diagonal_mass_topology(test, device):
    """Detect independent fixed-base branches, and keep coupled topologies off the diagonal path."""
    solver = SolverFeatherPGS(_build_fixed_base_star_model(device), pgs_mode="matrix_free", dense_max_constraints=16)
    test.assertEqual(solver._diagonal_mass_sizes, frozenset((16,)))
    test.assertTrue(solver._execution_plan.use_diagonal_mass(16))
    test.assertIsNone(solver._sparse_mass_matrix_size)
    # A serial chain couples every DOF, and pinned kernels disable the selection.
    chain = SolverFeatherPGS(_build_chain_model(device, num_links=3), pgs_mode="matrix_free", dense_max_constraints=16)
    test.assertEqual(chain._diagonal_mass_sizes, frozenset())
    pinned = _star_solver(_build_fixed_base_star_model(device), {"cholesky_kernel": "loop"}, dense_max_constraints=16)
    test.assertEqual(pinned._diagonal_mass_sizes, frozenset())
    # The DOFs of one multi-DOF joint are coupled.
    ball = SolverFeatherPGS(
        _build_fixed_base_star_model(device, num_branches=4, ball_branch=True), pgs_mode="matrix_free"
    )
    test.assertEqual(ball._diagonal_mass_sizes, frozenset())


def test_fixed_base_sibling_branches_use_diagonal_mass_path(test, device):
    """Match the dense reference while selecting independent branch solves."""
    model = _build_fixed_base_star_model(device)
    optimized = SolverFeatherPGS(model, pgs_mode="matrix_free", dense_max_constraints=16)
    test.assertEqual(optimized._diagonal_mass_sizes, frozenset((16,)))
    test.assertTrue(optimized._execution_plan.use_diagonal_mass(16))
    test.assertIsNone(optimized._cholesky_kernels_by_size[16])
    test.assertIsNone(optimized._triangular_solve_kernels_by_size[16])
    test.assertIsNone(optimized._hinv_jt_kernels_by_size[16])

    reference = _star_solver(model, _DENSE_LOOP_KERNELS, dense_max_constraints=16)
    test.assertFalse(reference._execution_plan.use_diagonal_mass(16))
    test.assertIsNone(reference._sparse_mass_matrix_size)

    for diagonal, dense in zip(_run_star(model, optimized, 5), _run_star(model, reference, 5), strict=True):
        np.testing.assert_allclose(diagonal[0], dense[0], rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(diagonal[1], dense[1], rtol=0.0, atol=2.0e-6)


def test_diagonal_mass_rows_match_dense_and_sparse_factors(test, device):
    """Match the dense loop kernels bitwise, and the sparse factors, with contact and limit rows."""
    # Point friction: the sparse factors compared against solve hard point-friction contacts.
    options = {
        "dense_max_constraints": 64,
        "enable_joint_limits": True,
        "pgs_iterations": 8,
        "friction_anchor_beta": 0.0,
    }
    for with_chain in (False, True):
        with test.subTest(with_chain=with_chain):
            model = _build_fixed_base_star_model(
                device, num_branches=6, ground=True, armature=0.05, with_chain=with_chain
            )
            diagonal = SolverFeatherPGS(model, pgs_mode="matrix_free", **options)
            test.assertTrue(diagonal._execution_plan.use_diagonal_mass(6))
            # With a second response group per world the responses are gathered per world.
            test.assertEqual(diagonal._hinv_jt_writes_world, with_chain)
            solvers = [diagonal, _star_solver(model, _DENSE_LOOP_KERNELS, **options)]
            if not with_chain:
                sparse = _star_solver(model, {"diagonal_mass": False}, **options)
                test.assertEqual(sparse._sparse_mass_matrix_size, 6)
                solvers.append(sparse)
            runs = [_run_star(model, solver, 60, contacts=True) for solver in solvers]
            test.assertGreater(int(diagonal.constraint_count.numpy().max()), 6, "no contact and limit rows")
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
            unarmed = _run_star(plain, SolverFeatherPGS(plain, pgs_mode="matrix_free", **options), 60, contacts=True)
            test.assertGreater(float(np.abs(unarmed[5][1] - runs[0][5][1]).max()), 1.0e-6)


def test_sleeping_diagonal_dynamics_skip_is_exact(test, device):
    """Skip the diagonal-mass dynamics of sleeping stars without changing any published state.

    Two solvers differ only in whether sleeping islands skip their dynamics; over a sleep, a
    force wake and a resettle, every state array stays bitwise equal.
    """
    runs = []
    for skip in (False, True):
        model = _build_fixed_base_star_model(device, num_branches=3, ground=True)
        solver = SolverFeatherPGS(
            model, pgs_mode="matrix_free", enable_sleeping=True, sleep_quiet_time=0.05, dense_max_constraints=64
        )
        test.assertTrue(solver._execution_plan.use_diagonal_mass(3))
        solver.sleeping.skip_dynamics = skip
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
        awake = runs[1][0].sleeping.art_awake.numpy()
        slept |= step < 300 and not awake.any()
        woke |= 300 <= step < 306 and bool(awake[0])
    test.assertTrue(slept, "the diagonal-mass articulations never slept")
    test.assertTrue(woke, "the external force did not wake the first articulation")
    for run in runs:
        run[0].check_constraint_capacity()


class TestFeatherPGSDiagonalMass(unittest.TestCase):
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
    add_function_test(TestFeatherPGSDiagonalMass, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main()
