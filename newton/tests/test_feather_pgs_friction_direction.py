# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check isotropic friction against the direction and magnitude of Coulomb sliding."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_mf_gs_kernel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _slide(device, beta, mesh, angle, articulated=False, *, locked_tangent=False):
    """Slide a symmetric, explicitly massed box under a force above the friction limit.

    Returns the final planar velocity, the push direction and the capacity status.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, gap=0.0, restitution=0.0)
    builder.add_shape_box(-1, hx=2.0, hy=2.0, hz=0.05, xform=wp.transform((0, 0, -0.05)), cfg=cfg)
    inertia = np.diag([(0.2**2 + 0.05**2) / 12] * 2 + [2 * 0.2**2 / 12]).astype(np.float32)
    add_body = builder.add_link if articulated else builder.add_body
    body = add_body(xform=wp.transform((0, 0, 0.025)), mass=1.0, inertia=wp.mat33(inertia), lock_inertia=True)
    if articulated:
        joint = builder.add_joint_d6(
            parent=-1,
            child=body,
            parent_xform=wp.transform((0, 0, 0.025)),
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(axis=axis)
                for axis in ((newton.Axis.X, newton.Axis.Z) if locked_tangent else newton.Axis)
            ],
        )
        builder.add_articulation([joint])
    if mesh:
        builder.add_shape_mesh(body, mesh=newton.Mesh.create_box(0.1, 0.1, 0.025), cfg=cfg)
    else:
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.025, cfg=cfg)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        friction_anchor_beta=beta,
        pgs_iterations=64,
        pgs_cfm=0.0 if locked_tangent else 1.0e-6,
        mf_max_constraints=128,
        dense_max_constraints=32,
    )
    solver.rigid_body_angular_damping.zero_()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, deterministic=True)
    contacts = pipeline.contacts()
    state, next_state = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    direction = np.array([np.cos(angle), np.sin(angle)])
    force = np.zeros((1, 6), dtype=np.float32)
    force[0, :2] = direction * 1.2 * 0.3 * 9.81
    hz = 240
    for step in range(hz // 5 + hz // 2):
        state.clear_forces()
        if step >= hz // 5:
            state.body_f.assign(force)
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / hz)
        state, next_state = next_state, state
    return state.body_qd.numpy()[0, :2], direction, solver.constraint_overflow.numpy()


def test_articulated_sliding_direction(test, device):
    """Preserve the sliding direction of an articulated box on the dense rows."""
    velocity, direction, overflow = _slide(device, 0.2, False, np.pi / 4, articulated=True)
    np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
    test.assertFalse(overflow.any())


def test_locked_tangent_preserves_other_friction_direction(test, device):
    """Retain friction along a slider when its first tangent has zero response."""
    for beta in (0.0, 0.2):
        with test.subTest(beta=beta):
            velocity, direction, overflow = _slide(device, beta, False, 0.0, articulated=True, locked_tangent=True)
            np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
            test.assertFalse(overflow.any())


def _solve_contact_block(device, tangent_mass, rhs, initial, capacity=32):
    """Sweep one contact's normal and two tangent rows once through the fused matrix-free kernel.

    The world has three DOFs whose rows are the unit Jacobians ``J = I`` with responses
    ``Y = M`` (``M[0, 0] = 1``, tangent block ``tangent_mass``), so the rows see the
    Delassus matrix ``M``. The starting velocity ``M @ initial`` is consistent with the
    initial impulses, as after a warm start.
    """
    matrix = np.eye(3, dtype=np.float32)
    matrix[1:3, 1:3] = tangent_mass
    J = np.zeros((1, capacity, 3), dtype=np.float32)
    Y = np.zeros((1, capacity, 3), dtype=np.float32)
    J[0, :3] = np.eye(3)
    Y[0, :3] = matrix
    row_type = np.full((1, capacity), -1, dtype=np.int32)
    row_type[0, :3] = [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]
    parents = np.full((1, capacity), -1, dtype=np.int32)
    parents[0, 1:3] = 0
    rhs_rows = np.zeros((1, capacity), dtype=np.float32)
    rhs_rows[0, :3] = rhs
    diag = np.zeros((1, capacity), dtype=np.float32)
    diag[0, :3] = np.diag(matrix)
    impulses_np = np.zeros((1, capacity), dtype=np.float32)
    impulses_np[0, :3] = initial
    impulses = wp.array(impulses_np, dtype=float, device=device)
    v_out = wp.array((matrix @ np.asarray(initial, dtype=np.float32)).astype(np.float32), dtype=float, device=device)
    kernel = _get_pgs_solve_mf_gs_kernel(
        capacity, 1, 3, wp.get_device(device).arch, has_dense_velocity_limit_rows=False, shared_metadata=True
    )
    wp.launch_tiled(
        kernel,
        dim=[1],
        inputs=[
            wp.array([3], dtype=int, device=device),
            wp.array([[0, 1, 2]], dtype=int, device=device),
            wp.array(rhs_rows, dtype=float, device=device),
            wp.array(diag, dtype=float, device=device),
            impulses,
            wp.array(J, dtype=float, device=device),
            wp.array(Y, dtype=float, device=device),
            wp.array(row_type, dtype=int, device=device),
            wp.array(parents, dtype=int, device=device),
            wp.full((1, capacity), 0.3, dtype=float, device=device),
            # Drive-row parameters: no drive rows.
            *(wp.zeros((1, 1), dtype=float, device=device) for _ in range(5)),
            wp.zeros((1,), dtype=int, device=device),
            wp.zeros((1,), dtype=int, device=device),
            wp.zeros((1, 4), dtype=int, device=device),
            wp.zeros((1, 1), dtype=float, device=device),
            wp.zeros((1, 1, 6), dtype=float, device=device),
            wp.zeros((1, 1, 6), dtype=float, device=device),
            wp.zeros((1, 1, 6), dtype=float, device=device),
            wp.zeros((1, 1, 6), dtype=float, device=device),
            wp.zeros((1, 1), dtype=float, device=device),
            wp.ones((1, 1), dtype=float, device=device),
            wp.ones((1, 1), dtype=float, device=device),
            wp.full((1, 1), -1, dtype=int, device=device),
            0.0,
            1,
            1.0,
            0,
            0,
            0,
            0,
            0,
        ],
        outputs=[v_out],
        block_dim=32,
        device=device,
    )
    return impulses.numpy()[0, :3]


def test_unequal_tangent_masses(test, device):
    """Solve sticking and sliding blocks with rotated, unequal tangent masses in one sweep."""
    for angle, small_eigenvalue in (
        (0.0, 1.0),
        (0.37, 1.0),
        (1.1, 1.0),
        (0.37, 0.0),
        (0.0, 0.0),
        (0.0, 1.0e-4),
        (0.0, 1.0e-6),
        (0.0, 1.0e-8),
    ):
        for regime in ("sticking", "sliding", "fixed", "weak_sliding", "sliding_fixed"):
            if regime == "weak_sliding" and small_eigenvalue == 0.0:
                continue
            sliding = regime in ("sliding", "weak_sliding", "sliding_fixed")
            with test.subTest(angle=angle, small_eigenvalue=small_eigenvalue, regime=regime):
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                tangent_mass = (rotation @ np.diag([small_eigenvalue, 7.0]) @ rotation.T).astype(np.float32)
                direction = np.array([0.6, 0.8])
                expected = -(0.3 if sliding else 0.1) * direction
                if regime == "fixed":
                    expected = np.array([0.125, 0.0])
                velocity = 2.0 * direction if sliding else np.zeros(2)
                if regime == "weak_sliding":
                    # A tiny positive KKT multiplier still requires a boundary impulse.
                    velocity = small_eigenvalue * direction
                rhs = np.array([-1.0, *(velocity - tangent_mass @ expected)], dtype=np.float32)
                initial = np.zeros(3, dtype=np.float32)
                if regime in ("fixed", "sliding_fixed"):
                    # The installed velocity M @ initial carries these impulses.
                    initial[:] = [1.0, *expected]
                if regime == "fixed":
                    # Power-of-two impulses make the row residual exactly zero.
                    rhs[1:3] = -(tangent_mass @ expected)
                actual = _solve_contact_block(device, tangent_mass, rhs, initial)
                if regime in ("fixed", "sliding_fixed"):
                    np.testing.assert_allclose(actual, initial, rtol=0.0, atol=1.0e-6)
                elif small_eigenvalue == 0.0 and not sliding:
                    # A singular sticking block has multiple impulse solutions.
                    np.testing.assert_allclose(tangent_mass @ actual[1:] + rhs[1:3], 0.0, atol=1.0e-6)
                    test.assertLessEqual(float(np.linalg.norm(actual[1:])), 0.300001)
                else:
                    np.testing.assert_allclose(actual, [1.0, *expected], atol=1.0e-6)


def test_sliding_direction(test, device):
    """Oppose sliding in every direction for analytic and mesh boxes, with and without patches."""
    for beta in (0.0, 0.2):
        for mesh in (False, True):
            for angle in (0.0, np.pi / 6, np.pi / 4, 2 * np.pi / 3):
                with test.subTest(beta=beta, mesh=mesh, angle=angle):
                    velocity, direction, overflow = _slide(device, beta, mesh, angle)
                    np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
                    test.assertFalse(overflow.any())


class TestFeatherPGSFrictionDirection(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (
    test_articulated_sliding_direction,
    test_locked_tangent_preserves_other_friction_direction,
    test_unequal_tangent_masses,
    test_sliding_direction,
):
    add_function_test(TestFeatherPGSFrictionDirection, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
