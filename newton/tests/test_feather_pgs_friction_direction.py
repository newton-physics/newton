# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check isotropic friction against the direction and magnitude of Coulomb sliding."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import pgs_solve_loop
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_tiled_row_kernel


def _slide(device, mode, beta, mesh, angle, response=None, *, locked_tangent=False):
    """Slide a symmetric, explicitly massed box under a force above the friction limit."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, gap=0.0, restitution=0.0)
    builder.add_shape_box(-1, hx=2.0, hy=2.0, hz=0.05, xform=wp.transform((0, 0, -0.05)), cfg=cfg)
    inertia = np.diag([(0.2**2 + 0.05**2) / 12] * 2 + [2 * 0.2**2 / 12]).astype(np.float32)
    add_body = builder.add_body if response is None else builder.add_link
    body = add_body(xform=wp.transform((0, 0, 0.025)), mass=1.0, inertia=wp.mat33(inertia), lock_inertia=True)
    if response is not None:
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
        pgs_mode=mode,
        friction_anchor_beta=beta,
        pgs_iterations=64,
        pgs_cfm=0.0 if locked_tangent else 1.0e-6,
        angular_damping=0.0,
        mf_max_constraints=128,
        dense_max_constraints=32,
        row_watermark=True,
        articulated_contact_response=response or "immediate",
    )
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
    return state.body_qd.numpy()[0, :2], direction, solver.constraint_row_watermarks()


class TestFeatherPGSFrictionDirection(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_articulated_sliding_direction(self):
        """Preserve sliding direction through dense and every articulated propagation route."""
        for response in ("immediate", "propagation", "propagation-fused", "propagation-colored"):
            with self.subTest(response=response):
                velocity, direction, _ = _slide("cuda:0", "matrix_free", 0.2, False, np.pi / 4, response)
                np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_locked_tangent_preserves_other_friction_direction(self):
        """Retain friction along a slider when its first tangent has zero response."""
        for response in ("immediate", "propagation", "propagation-fused", "propagation-colored"):
            for beta in (0.0, 0.2):
                with self.subTest(response=response, beta=beta):
                    velocity, direction, watermarks = _slide(
                        "cuda:0", "matrix_free", beta, False, 0.0, response, locked_tangent=True
                    )
                    np.testing.assert_allclose(velocity, direction * (0.2 * 0.3 * 9.81) * 0.5, atol=0.015, rtol=0.03)
                    for key, value in watermarks.items():
                        if "overflow" in key or "dropped" in key:
                            self.assertEqual(value, 0, key)

    def test_unequal_tangent_masses(self):
        """Solve sticking and sliding blocks with rotated, unequal tangent masses in one sweep."""
        devices = ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]
        capacity = 32
        for device in devices:
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
                    with self.subTest(device=device, angle=angle, small_eigenvalue=small_eigenvalue, regime=regime):
                        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                        tangent_mass = rotation @ np.diag([small_eigenvalue, 7.0]) @ rotation.T
                        direction = np.array([0.6, 0.8])
                        expected = -(0.3 if sliding else 0.1) * direction
                        if regime == "fixed":
                            expected = np.array([0.125, 0.0])
                        velocity = 2.0 * direction if sliding else np.zeros(2)
                        if regime == "weak_sliding":
                            # A tiny positive KKT multiplier still requires a boundary impulse.
                            # Absolute bisection resolution loses the weak direction here.
                            velocity = small_eigenvalue * direction
                        matrix = np.eye(capacity, dtype=np.float32)
                        matrix[1:3, 1:3] = tangent_mass
                        rhs = np.zeros((1, capacity), dtype=np.float32)
                        rhs[0, :3] = [-1.0, *(velocity - tangent_mass @ expected)]
                        if regime == "fixed":
                            # Power-of-two impulses make the stored row residual exactly zero.
                            rhs[0, 1:3] = -(matrix[1:3, 1:3] @ expected)
                        row_types = np.full((1, capacity), -1, dtype=np.int32)
                        row_types[0, :3] = [0, 2, 2]
                        parents = np.full((1, capacity), -1, dtype=np.int32)
                        parents[0, 1:3] = 0
                        initial = np.zeros((1, capacity), dtype=np.float32)
                        if regime in ("fixed", "sliding_fixed"):
                            initial[0, :3] = [1.0, *expected]
                        impulses = wp.array(initial, dtype=float, device=device)
                        args = [
                            wp.array([3], dtype=int, device=device),
                            wp.array(np.diag(matrix)[None], dtype=float, device=device),
                            wp.array(matrix[None], dtype=float, device=device),
                            wp.array(rhs, dtype=float, device=device),
                            impulses,
                            1,
                            1.0,
                            wp.array(row_types, dtype=int, device=device),
                            wp.array(parents, dtype=int, device=device),
                            wp.full((1, capacity), 0.3, dtype=float, device=device),
                            0,
                            0,
                        ]
                        if device == "cpu":
                            wp.launch(pgs_solve_loop, dim=1, inputs=[args[0], capacity, *args[1:]], device=device)
                        else:
                            kernel = _get_pgs_solve_tiled_row_kernel(capacity, str(wp.get_device(device).arch))
                            wp.launch_tiled(kernel, dim=[1], inputs=args, block_dim=32, device=device)
                        actual = impulses.numpy()[0, :3]
                        if regime in ("fixed", "sliding_fixed"):
                            np.testing.assert_array_equal(actual, initial[0, :3])
                        elif small_eigenvalue == 0.0 and not sliding:
                            # A singular sticking block has multiple impulse solutions.
                            np.testing.assert_allclose(tangent_mass @ actual[1:] + rhs[0, 1:3], 0.0, atol=1.0e-6)
                            self.assertLessEqual(float(np.linalg.norm(actual[1:])), 0.300001)
                        else:
                            np.testing.assert_allclose(actual, [1.0, *expected], atol=1.0e-6)

    def test_sliding_direction(self):
        """Oppose sliding in every direction for analytic and mesh boxes, with and without patches."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        modes = ("split", "matrix_free") if wp.is_cuda_available() else ("split",)
        for mode in modes:
            for beta in (0.0, 0.2):
                for mesh in (False, True):
                    for angle in (0.0, np.pi / 6, np.pi / 4, 2 * np.pi / 3):
                        with self.subTest(mode=mode, beta=beta, mesh=mesh, angle=angle):
                            velocity, direction, watermarks = _slide(device, mode, beta, mesh, angle)
                            expected = direction * (0.2 * 0.3 * 9.81) * 0.5
                            np.testing.assert_allclose(velocity, expected, atol=0.015, rtol=0.03)
                            for key, value in watermarks.items():
                                if "overflow" in key or "dropped" in key:
                                    self.assertEqual(value, 0, key)


if __name__ == "__main__":
    unittest.main()
