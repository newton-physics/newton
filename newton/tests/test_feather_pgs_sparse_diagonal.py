# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse-diagonal contact solve of SolverFeatherPGS: one diagonal-mass and one small dense articulation per world."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction import friction_pair_candidate
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _get_build_independent_sparse_contact_groups_kernel,
    _get_mark_independent_sparse_contact_candidates_kernel,
    _get_pgs_solve_sparse_diagonal_kernel,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

_PAIR_SOLVER = {
    "pgs_mode": "matrix_free",
    "dense_max_constraints": 64,
    "mf_max_constraints": 32,
    "pgs_iterations": 8,
    "enable_joint_limits": True,
}


def _build_sparse_diagonal_pair_model(
    num_branches=16, num_worlds=2, *, device=None, revolute_branches=False, with_free_body=False
):
    """Build one independent fixed-base star and one compact serial chain per world.

    ``revolute_branches`` hinges the star branches about Y instead of sliding them along Z, so the branch
    response depends on the link inertia and center of mass rather than on the mass alone.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [scene.add_joint_fixed(parent=-1, child=base)]
    add_branch_joint = scene.add_joint_revolute if revolute_branches else scene.add_joint_prismatic
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        star_joints.append(
            add_branch_joint(
                parent=base,
                child=child,
                axis=newton.Axis.Y if revolute_branches else newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.03 * branch, 0.0, 0.0), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    chain_joints = []
    parent = -1
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)
    if with_free_body:
        body = scene.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
        scene.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def _build_sparse_contact_friction_model(num_branches=16, num_worlds=2, *, device=None):
    """Build the sparse pair model with tilted sliding branches resting on a static ground plane.

    Each branch slides along ``(1, 0, 1) / sqrt(2)`` and carries a small box that touches the ground, so every
    contact normal and tangent row acts on that branch's single coordinate and friction is active.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    scene.default_shape_cfg.ke = 1.0e5
    scene.default_shape_cfg.kd = 1.0e3
    scene.default_shape_cfg.mu = 0.8
    scene.default_shape_cfg.margin = 0.0
    scene.default_shape_cfg.gap = 0.0
    scene.add_ground_plane()
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [
        scene.add_joint_fixed(
            parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
        )
    ]
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        # Four boxes keep the four-point patches within the dense row capacity of the test solvers.
        if branch % 4 == 0:
            scene.add_shape_box(child, hx=0.01, hy=0.01, hz=0.01)
        star_joints.append(
            scene.add_joint_prismatic(
                parent=base,
                child=child,
                axis=wp.normalize(wp.vec3(1.0, 0.0, 1.0)),
                parent_xform=wp.transform(wp.vec3(0.05 * branch, 0.0, -0.4905), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    chain_joints = []
    parent = -1
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 1.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def test_sparse_diagonal_gs_matches_scalar_reference(test, device):
    """Match a scalar PGS reference with coupled limits and friction."""
    device = wp.get_device(device)
    max_constraints, world_dofs, dense_dofs = 8, 4, 2
    cfm, omega, iterations = 1.0e-6, 1.0, 8
    inverse_mass = np.array((0.5, 0.25, 0.75, 1.0), dtype=np.float32)
    initial_velocity = np.array((-0.4, 0.3, -0.2, 0.1), dtype=np.float32)
    jacobian = np.array(
        ((0.6, -0.4, 0.2, 0.0), (0.1, 0.5, -0.3, 0.4), (-0.2, 0.25, 0.1, -0.35)),
        dtype=np.float32,
    )
    response = jacobian * inverse_mass
    rhs = np.zeros((1, max_constraints), dtype=np.float32)
    rhs[0, :3] = (-0.1, 0.0, 0.0)
    diag = np.zeros_like(rhs)
    diag[0, :3] = np.sum(jacobian * response, axis=1) + cfm
    row_type = np.zeros((1, max_constraints), dtype=np.int32)
    row_type[0, :3] = (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION)
    row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
    row_parent[0, 1:3] = 0
    row_mu = np.zeros((1, max_constraints), dtype=np.float32)
    row_mu[0, 1:3] = 0.7
    dense_j = np.zeros((1, max_constraints, dense_dofs), dtype=np.float32)
    dense_y = np.zeros_like(dense_j)
    dense_j[0, :3] = jacobian[:, :dense_dofs]
    dense_y[0, :3] = response[:, :dense_dofs]
    sparse_dof = np.full((1, max_constraints, 2), -1, dtype=np.int32)
    sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
    sparse_dof[0, :3] = (2, 3)
    sparse_jy[0, :3, 0] = jacobian[:, 2]
    sparse_jy[0, :3, 1] = response[:, 2]
    sparse_jy[0, :3, 2] = jacobian[:, 3]
    sparse_jy[0, :3, 3] = response[:, 3]
    limit_active = np.zeros((1, world_dofs), dtype=np.int32)
    limit_active[0, 2] = 1
    limit_lower_rhs = np.zeros((1, world_dofs), dtype=np.float32)
    limit_lower_rhs[0, 2] = -0.15

    expected_velocity = initial_velocity.copy()
    expected_impulses = np.zeros(3, dtype=np.float32)
    expected_limit_lambda = np.float32(0.0)
    for _ in range(iterations):
        old_limit = expected_limit_lambda
        residual = expected_velocity[2] + limit_lower_rhs[0, 2]
        expected_limit_lambda = np.maximum(0.0, old_limit - omega * residual / (inverse_mass[2] + cfm))
        expected_velocity[2] += inverse_mass[2] * (expected_limit_lambda - old_limit)
        old_impulse = expected_impulses[0]
        new_impulse = old_impulse - omega * (jacobian[0] @ expected_velocity + rhs[0, 0]) / diag[0, 0]
        new_impulse = max(new_impulse, 0.0)
        expected_impulses[0] = new_impulse
        expected_velocity += response[0] * (new_impulse - old_impulse)
        # The general owner solves both tangent rows together on the friction disk at the current normal
        # load (friction_pair_candidate); the second tangent visit is a no-op.
        radius = max(row_mu[0, 1] * expected_impulses[0], 0.0)
        pair = friction_pair_candidate(
            float(diag[0, 1]),
            float(jacobian[1] @ response[2]),
            float(diag[0, 2]),
            wp.vec2(
                float(jacobian[1] @ expected_velocity + rhs[0, 1]),
                float(jacobian[2] @ expected_velocity + rhs[0, 2]),
            ),
            wp.vec2(float(expected_impulses[1]), float(expected_impulses[2])),
            float(radius),
            omega,
        )
        magnitude = np.sqrt(pair[0] * pair[0] + pair[1] * pair[1])
        scale = radius / magnitude if magnitude > radius else 1.0
        for tangent, value in ((1, pair[0] * scale), (2, pair[1] * scale)):
            expected_velocity += response[tangent] * (value - expected_impulses[tangent])
            expected_impulses[tangent] = np.float32(value)

    impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    lower_lambda = wp.zeros((1, world_dofs), dtype=wp.float32, device=device)
    upper_lambda = wp.zeros_like(lower_lambda)
    velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
    kernel = _get_pgs_solve_sparse_diagonal_kernel(
        max_constraints, world_dofs, dense_dofs, str(device.arch), contact_triples=True
    )
    wp.launch_tiled(
        kernel,
        dim=[1],
        inputs=[
            wp.array((3,), dtype=wp.int32, device=device),
            wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
            wp.array(rhs, device=device),
            wp.array(diag, device=device),
            impulses,
            wp.array(row_type, device=device),
            wp.array(row_parent, device=device),
            wp.array(row_mu, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.empty((1, world_dofs), dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device),
            wp.array((0,), dtype=wp.int32, device=device),
            wp.array((0,), dtype=wp.int32, device=device),
            wp.array(dense_j, device=device),
            wp.array(dense_y, device=device),
            wp.array(sparse_dof, device=device),
            wp.array(sparse_jy, device=device),
            wp.array(limit_active, device=device),
            wp.array(limit_lower_rhs, device=device),
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            wp.array(inverse_mass, device=device),
            cfm,
            iterations,
            omega,
            0,
            0,
        ],
        outputs=[lower_lambda, upper_lambda, velocity],
        block_dim=32,
        device=device,
    )

    np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(impulses.numpy()[0, :3], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(lower_lambda.numpy()[0, 2], expected_limit_lambda, rtol=2.0e-5, atol=2.0e-6)
    test.assertEqual(float(upper_lambda.numpy()[0, 2]), 0.0)


def test_speculative_sparse_contact_batches_match_serial_reference(test, device):
    """Skip exact no-op prefixes without changing later serial contact updates."""
    device = wp.get_device(device)
    max_constraints, world_dofs, dense_dofs = 12, 8, 6
    cfm, omega, iterations = 1.0e-6, 1.0, 2
    inverse_mass = np.array((0.5, 0.25, 0.75, 1.0, 0.4, 0.6, 0.75, 0.8), dtype=np.float32)
    initial_velocity = np.array((0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.5, 0.25), dtype=np.float32)
    jacobian = np.zeros((max_constraints, world_dofs), dtype=np.float32)
    jacobian[0, 6] = 1.0
    jacobian[3, 7] = 1.0
    jacobian[6, (0, 6)] = (-0.2, -1.0)
    jacobian[9, (0, 6)] = (0.3, 1.0)
    response = jacobian * inverse_mass
    rhs = np.zeros((1, max_constraints), dtype=np.float32)
    rhs[0, 6] = -0.1
    rhs[0, 9] = -0.2
    diag = np.zeros_like(rhs)
    normal_rows = np.array((0, 3, 6, 9), dtype=np.int32)
    diag[0, normal_rows] = np.sum(jacobian[normal_rows] * response[normal_rows], axis=1) + cfm
    row_type = np.full((1, max_constraints), PGS_CONSTRAINT_TYPE_FRICTION, dtype=np.int32)
    row_type[0, normal_rows] = PGS_CONSTRAINT_TYPE_CONTACT
    row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
    for normal in normal_rows:
        row_parent[0, normal + 1 : normal + 3] = normal
    row_mu = np.zeros((1, max_constraints), dtype=np.float32)
    dense_j = jacobian[:, :dense_dofs][None, ...]
    dense_y = response[:, :dense_dofs][None, ...]
    sparse_dof = np.empty((1, max_constraints, 2), dtype=np.int32)
    sparse_dof[:, :, 0] = 6
    sparse_dof[:, :, 1] = 7
    sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
    sparse_jy[0, :, 0] = jacobian[:, 6]
    sparse_jy[0, :, 1] = response[:, 6]
    sparse_jy[0, :, 2] = jacobian[:, 7]
    sparse_jy[0, :, 3] = response[:, 7]

    expected_velocity = initial_velocity.copy()
    expected_impulses = np.zeros(max_constraints, dtype=np.float32)
    for _ in range(iterations):
        for row in normal_rows:
            old_impulse = expected_impulses[row]
            residual = jacobian[row] @ expected_velocity + rhs[0, row]
            new_impulse = max(old_impulse - omega * residual / diag[0, row], 0.0)
            expected_impulses[row] = new_impulse
            expected_velocity += response[row] * (new_impulse - old_impulse)

    impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
    kernel = _get_pgs_solve_sparse_diagonal_kernel(
        max_constraints,
        world_dofs,
        dense_dofs,
        str(device.arch),
        contact_triples=True,
        speculative_contact_batches=True,
    )
    wp.launch_tiled(
        kernel,
        dim=[1],
        inputs=[
            wp.array((max_constraints,), dtype=wp.int32, device=device),
            wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
            wp.array(rhs, device=device),
            wp.array(diag, device=device),
            impulses,
            wp.array(row_type, device=device),
            wp.array(row_parent, device=device),
            wp.array(row_mu, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
            wp.empty((1, world_dofs), dtype=wp.int32, device=device),
            wp.array((len(normal_rows),), dtype=wp.int32, device=device),
            wp.array(normal_rows[None, :], device=device),
            wp.array((0,), dtype=wp.int32, device=device),
            wp.array((0,), dtype=wp.int32, device=device),
            wp.array(dense_j, device=device),
            wp.array(dense_y, device=device),
            wp.array(sparse_dof, device=device),
            wp.array(sparse_jy, device=device),
            wp.zeros((1, world_dofs), dtype=wp.int32, device=device),
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            wp.array(inverse_mass, device=device),
            cfm,
            iterations,
            omega,
            0,
            0,
        ],
        outputs=[
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
            velocity,
        ],
        block_dim=32,
        device=device,
    )

    np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(impulses.numpy()[0], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
    np.testing.assert_array_equal(impulses.numpy()[0, :6], np.zeros(6, dtype=np.float32))


def test_independent_sparse_contact_groups_exclude_coupled_coordinates(test, device):
    """Keep scalar contacts serial when a coupled contact shares their coordinate."""
    device = wp.get_device(device)
    max_constraints, max_world_dofs, contact_count = 12, 4, 4
    sparse_response_dofs = 108
    sparse_row_dof_np = np.full((1, max_constraints, 2), -1, dtype=np.int32)
    sparse_row_dof_np[0, 0:3, 0] = 2
    sparse_row_dof_np[0, 3:6, 0] = 3
    sparse_row_dof_np[0, 6:9, 0] = 3
    sparse_row_dof_np[0, 9:12, 0] = 2
    sparse_row_dof = wp.array(sparse_row_dof_np, device=device)
    group_count = wp.zeros(1, dtype=wp.int32, device=device)
    group_heads = wp.empty((1, max_world_dofs), dtype=wp.int32, device=device)
    serial_count = wp.zeros(1, dtype=wp.int32, device=device)
    serial_normals = wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device)

    marker = _get_mark_independent_sparse_contact_candidates_kernel(sparse_response_dofs, str(device.arch))
    wp.launch(
        marker,
        dim=contact_count,
        inputs=[
            wp.array((contact_count,), dtype=wp.int32, device=device),
            contact_count,
            wp.zeros(contact_count, dtype=wp.int32, device=device),
            wp.array((0, 3, 6, 9), dtype=wp.int32, device=device),
            wp.zeros(contact_count, dtype=wp.int32, device=device),
            wp.array((-1, 1, -1, -1), dtype=wp.int32, device=device),
            wp.zeros(contact_count, dtype=wp.int32, device=device),
            wp.full(contact_count, 3, dtype=wp.int32, device=device),
            wp.array((sparse_response_dofs, 6), dtype=wp.int32, device=device),
        ],
        outputs=[sparse_row_dof],
        device=device,
    )
    schedule = _get_build_independent_sparse_contact_groups_kernel(
        max_constraints, max_world_dofs, str(device.arch), build_serial_contacts=True
    )
    wp.launch(
        schedule,
        dim=32,
        inputs=[
            wp.array((max_constraints,), dtype=wp.int32, device=device),
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
        outputs=[sparse_row_dof, group_count, group_heads, serial_count, serial_normals],
        device=device,
    )
    wp.synchronize_device(device)

    test.assertEqual(int(group_count.numpy()[0]), 1)
    test.assertEqual(int(group_heads.numpy()[0, 0]), 0)
    test.assertEqual(int(serial_count.numpy()[0]), 2)
    np.testing.assert_array_equal(serial_normals.numpy()[0, :2], np.array((3, 6), dtype=np.int32))
    normal_links = sparse_row_dof.numpy()[0, ::3, 1]
    np.testing.assert_array_equal(normal_links, np.array((-12, -1, -1, -2), dtype=np.int32))


def test_sparse_diagonal_response_matches_dense_joint_limits(test, device):
    """Match dense position-limit rows while removing the diagonal articulation's dense response storage."""
    model = _build_sparse_diagonal_pair_model(device=device)
    # Point friction keeps the contact-triple schedule under test; patch rows are not uniform triples.
    optimized = SolverFeatherPGS(model, use_parallel_streams=True, friction_anchor_beta=0.0, **_PAIR_SOLVER)
    reference = SolverFeatherPGS(model, friction_anchor_beta=0.0, **_PAIR_SOLVER)
    test.assertTrue(optimized._sparse_diagonal_contact_solve)
    test.assertTrue(optimized._sparse_diagonal_contact_triples)
    test.assertTrue(optimized._sparse_diagonal_speculative_contact_batches)
    test.assertEqual(optimized._sparse_diagonal_response_size, 16)
    test.assertEqual(optimized._sparse_diagonal_dense_size, 3)
    test.assertEqual(optimized.H_by_size[16].shape, (1, 1, 1))
    test.assertEqual(optimized.J_by_size[16].shape, (1, 1, 1))
    test.assertIsNone(optimized._pgs_solve_mf_gs_kernel)
    test.assertFalse(reference._sparse_diagonal_contact_solve)

    # Alternate branches start past the lower and the upper limit, moving further out.
    side = np.where(np.arange(16) % 2 == 0, -1.0, 1.0)
    q = np.tile(np.concatenate((0.105 * side, np.full(3, 0.205))).astype(np.float32), 2)
    qd = np.tile(np.concatenate((0.2 * side, np.full(3, 0.1))).astype(np.float32), 2)
    trajectories = []
    for solver in (optimized, reference):
        state_in, state_out = model.state(), model.state()
        state_in.joint_q.assign(q)
        state_in.joint_qd.assign(qd)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        control = model.control()
        history = []
        for _ in range(3):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, 1.0 / 120.0)
            state_in, state_out = state_out, state_in
            history.append((state_in.joint_q.numpy().copy(), state_in.joint_qd.numpy().copy()))
        trajectories.append(history)

    first_q, first_qd = trajectories[0][0]
    # Each branch recovers pgs_beta of its violation in one step, on both sides.
    np.testing.assert_allclose(first_q.reshape(2, 19)[:, :16], np.tile(0.104 * side, (2, 1)), rtol=0.0, atol=2.0e-6)
    np.testing.assert_allclose(first_qd.reshape(2, 19)[:, :16], np.tile(-0.12 * side, (2, 1)), rtol=0.0, atol=2.0e-6)
    for sparse, dense in zip(*trajectories, strict=True):
        np.testing.assert_allclose(sparse[0], dense[0], rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(sparse[1], dense[1], rtol=2.0e-5, atol=2.0e-6)


def test_sparse_diagonal_contact_friction_matches_general_owner(test, device):
    """Match the general sweep's paired friction on sparse contacts with patch and point friction."""
    model = _build_sparse_contact_friction_model(device=device)

    def make_solver(**kwargs):
        options = dict(_PAIR_SOLVER, dense_max_constraints=96)
        options.update(kwargs)
        return SolverFeatherPGS(model, **options)

    def run(solver):
        state_in, state_out = model.state(), model.state()
        joint_qd = state_in.joint_qd.numpy()
        for art in np.flatnonzero(solver._model_plan.response_dof_count == 16):
            start = int(solver._model_plan.articulation_dof_start[art])
            joint_qd[start : start + 16] = -0.3
        state_in.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
        contacts = pipeline.contacts()
        control = model.control()
        history = []
        for _ in range(4):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
            history.append((state_in.joint_q.numpy().copy(), state_in.joint_qd.numpy().copy()))
        return history

    # Normal rows only: the frictionless reference.
    slick = run(make_solver(use_parallel_streams=True, contact_friction_gap_threshold=-float("inf")))
    for label, friction_kwargs, triples in (("patch", {}, False), ("point", {"friction_anchor_beta": 0.0}, True)):
        with test.subTest(friction=label):
            optimized = make_solver(use_parallel_streams=True, **friction_kwargs)
            reference = make_solver(**friction_kwargs)
            test.assertTrue(optimized._sparse_diagonal_contact_solve)
            test.assertEqual(optimized._friction_anchors_enabled, label == "patch")
            test.assertEqual(optimized._sparse_diagonal_contact_triples, triples)
            test.assertEqual(optimized._sparse_diagonal_speculative_contact_batches, triples)
            test.assertFalse(reference._sparse_diagonal_contact_solve)
            sparse = run(optimized)
            general = run(reference)
            test.assertGreater(int(optimized.constraint_count.numpy().max()), 0, "no contact rows were generated")
            test.assertGreater(
                float(np.abs(sparse[-1][1] - slick[-1][1]).max()), 1.0e-3, "friction did not change the sparse response"
            )
            for step, ((sparse_q, sparse_qd), (general_q, general_qd)) in enumerate(zip(sparse, general, strict=True)):
                np.testing.assert_allclose(
                    sparse_q, general_q, rtol=2.0e-5, atol=2.0e-6, err_msg=f"joint_q step {step}"
                )
                np.testing.assert_allclose(
                    sparse_qd, general_qd, rtol=2.0e-5, atol=2.0e-6, err_msg=f"joint_qd step {step}"
                )


def test_sparse_diagonal_contact_compliance_matches_general_owner(test, device):
    """Solve compliant contacts on the sparse rows like the general sweep."""
    model = _build_sparse_contact_friction_model(device=device)
    results = {}
    for label, streams, compliance in (("sparse", True, True), ("general", False, True), ("rigid", False, False)):
        solver = SolverFeatherPGS(
            model,
            **dict(_PAIR_SOLVER, dense_max_constraints=96, pgs_iterations=16),
            use_parallel_streams=streams,
            friction_anchor_beta=0.0,
            contact_compliance=compliance,
        )
        state_in, state_out = model.state(), model.state()
        joint_qd = state_in.joint_qd.numpy()
        for art in np.flatnonzero(solver._model_plan.response_dof_count == 16):
            start = int(solver._model_plan.articulation_dof_start[art])
            joint_qd[start : start + 16] = -0.3
        state_in.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=256)
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(256, dtype=float, device=device))
        compliant = 0
        for _ in range(40):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            contacts.rigid_contact_stiffness.fill_(3000.0)
            contacts.rigid_contact_damping.fill_(20.0)
            contacts.rigid_contact_friction.fill_(1.0)
            solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
            compliant = max(compliant, solver.compliance_contact_count)
        results[label] = (solver, compliant, state_in.joint_qd.numpy())
    sparse, compliant, joint_qd = results["sparse"]
    test.assertTrue(sparse._sparse_diagonal_contact_solve)
    test.assertFalse(results["general"][0]._sparse_diagonal_contact_solve)
    test.assertGreater(compliant, 0)
    test.assertGreater(float(np.abs(results["rigid"][2] - results["general"][2]).max()), 1.0e-3)
    np.testing.assert_allclose(joint_qd, results["general"][2], rtol=0.0, atol=1.0e-6)


def test_sparse_diagonal_is_selected_only_where_supported(test, device):
    """Keep the general sweep without streams, on CPU, with a free body or with options it does not solve."""
    model = _build_sparse_diagonal_pair_model(device=device)
    if not wp.get_device(device).is_cuda:
        solver = SolverFeatherPGS(
            model, pgs_mode="split", enable_joint_limits=True, dense_max_constraints=64, use_parallel_streams=True
        )
        test.assertFalse(solver._sparse_diagonal_contact_solve)
        return
    options = dict(_PAIR_SOLVER, use_parallel_streams=True)
    test.assertTrue(SolverFeatherPGS(model, **options)._sparse_diagonal_contact_solve)
    for label, overrides in (
        ("no streams", {"use_parallel_streams": False}),
        ("joint limits off", {"enable_joint_limits": False}),
        ("torsion", {"contact_torsion_radius": 0.01}),
        ("regularization", {"pgs_contact_regularization": 0.02}),
        ("velocity iterations", {"pgs_velocity_iterations": 1}),
        ("warm start", {"pgs_warmstart": True}),
        ("schedule", {"pgs_schedule": "physx_grasp"}),
        ("drive rows", {"drive_mode": "physx_pgs"}),
        ("friction mode", {"friction_mode": "bisection", "friction_anchor_beta": 0.0}),
    ):
        with test.subTest(label):
            solver = SolverFeatherPGS(model, **dict(options, **overrides))
            test.assertFalse(solver._sparse_diagonal_contact_solve)
            test.assertFalse(solver._sparse_diagonal_contact_triples)
    with_free_body = _build_sparse_diagonal_pair_model(device=device, with_free_body=True)
    test.assertFalse(SolverFeatherPGS(with_free_body, **options)._sparse_diagonal_contact_solve)


class TestFeatherPGSSparseDiagonal(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
for _fn in (
    test_sparse_diagonal_gs_matches_scalar_reference,
    test_speculative_sparse_contact_batches_match_serial_reference,
    test_independent_sparse_contact_groups_exclude_coupled_coordinates,
    test_sparse_diagonal_response_matches_dense_joint_limits,
    test_sparse_diagonal_contact_friction_matches_general_owner,
    test_sparse_diagonal_contact_compliance_matches_general_owner,
):
    add_function_test(TestFeatherPGSSparseDiagonal, _fn.__name__, _fn, devices=cuda_devices)
add_function_test(
    TestFeatherPGSSparseDiagonal,
    "test_sparse_diagonal_is_selected_only_where_supported",
    test_sparse_diagonal_is_selected_only_where_supported,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
