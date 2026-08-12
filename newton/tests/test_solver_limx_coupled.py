# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import AffineBodyModel, ConstraintAffineParticleContact, SolverLIMXCoupled


def _make_particle_model(device: str = "cpu"):
    positions = np.asarray(
        [[0.33, 0.33, 0.342], [3.0, 3.0, 3.0], [4.0, 3.0, 3.0]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particles(
        pos=positions,
        vel=[wp.vec3(0.0)] * len(positions),
        mass=[1.0] * len(positions),
        radius=[0.0] * len(positions),
    )
    builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
    return builder.finalize(device=device)


def _make_body_model(device: str = "cpu") -> AffineBodyModel:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    surface_triangles = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int32,
    )
    model = AffineBodyModel(
        vertices,
        tetrahedra,
        surface_triangles,
        density=6.0,
        rigidity=1.0e4,
        initial_transform=wp.transform_identity(),
        device=device,
    )
    model.gravity.zero_()
    return model


def _make_contact(particle_model, body_model) -> ConstraintAffineParticleContact:
    return ConstraintAffineParticleContact(
        particle_model,
        body_model,
        thickness=0.01,
        stiffness=1.0e3,
        normal_damping=0.0,
        friction=0.0,
        friction_epsilon=1.0e-4,
        max_contacts=64,
    )


class TestSolverLIMXCoupled(unittest.TestCase):
    def test_advances_particles_and_affine_body_in_one_solve(self):
        """Move both native domains in opposite directions through one mixed contact solve."""
        particle_model = _make_particle_model()
        body_model = _make_body_model()
        contact = _make_contact(particle_model, body_model)
        solver = SolverLIMXCoupled(
            particle_model,
            [],
            body_model,
            dynamic_operator=contact,
            nonlinear_iterations=1,
            linear_iterations=50,
        )
        state_in = particle_model.state()
        state_out = particle_model.state()
        initial_affine = solver.q.numpy().copy()

        solver.step(state_in, state_out, None, None, 0.01)

        particle_delta = state_out.particle_q.numpy()[0] - state_in.particle_q.numpy()[0]
        affine_delta = solver.q.numpy()[0, :3] - initial_affine[0, :3]
        outward = np.ones(3, dtype=np.float32) / np.sqrt(3.0)
        self.assertGreater(float(np.dot(particle_delta, outward)), 0.0)
        self.assertLess(float(np.dot(affine_delta, outward)), 0.0)
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.particle_qd.numpy()).all())
        self.assertTrue(np.isfinite(solver.q.numpy()).all())
        self.assertTrue(np.isfinite(solver.qd.numpy()).all())

    def test_rejects_mismatched_dynamic_operator_domains(self):
        """Reject mixed operators whose particle, body, or device domain differs."""
        particle_model = _make_particle_model()
        body_model = _make_body_model()

        class Domain:
            def __init__(self, particle_count: int, body_count: int, device: str):
                self.particle_count = particle_count
                self.body_count = body_count
                self.device = wp.get_device(device)

        with self.assertRaisesRegex(ValueError, "particle count"):
            SolverLIMXCoupled(
                particle_model,
                [],
                body_model,
                dynamic_operator=Domain(particle_model.particle_count + 1, body_model.body_count, "cpu"),
            )
        with self.assertRaisesRegex(ValueError, "body count"):
            SolverLIMXCoupled(
                particle_model,
                [],
                body_model,
                dynamic_operator=Domain(particle_model.particle_count, body_model.body_count + 1, "cpu"),
            )
        if wp.is_cuda_available():
            with self.assertRaisesRegex(ValueError, "device"):
                SolverLIMXCoupled(
                    particle_model,
                    [],
                    body_model,
                    dynamic_operator=Domain(particle_model.particle_count, body_model.body_count, "cuda:0"),
                )

    def test_warm_starts_only_first_newton_solve_of_each_frame(self):
        """Warm-start only the first mixed solve of each coupled frame."""
        particle_model = _make_particle_model()
        body_model = _make_body_model()
        solver = SolverLIMXCoupled(
            particle_model,
            [],
            body_model,
            dynamic_operator=_make_contact(particle_model, body_model),
            nonlinear_iterations=3,
            linear_iterations=2,
        )
        state_in = particle_model.state()
        state_out = particle_model.state()
        zero_initial_guess_sequence: list[bool] = []
        solve = solver.linear_solver.solve

        def record_initial_guess_policy(*args, **kwargs):
            zero_initial_guess_sequence.append(kwargs["zero_initial_guess"])
            return solve(*args, **kwargs)

        solver.linear_solver.solve = record_initial_guess_policy
        solver.step(state_in, state_out, None, None, 0.01)
        state_in, state_out = state_out, state_in
        solver.step(state_in, state_out, None, None, 0.01)

        self.assertEqual(zero_initial_guess_sequence, [False, True, True, False, True, True])

    def test_exports_public_coupled_types(self):
        """Expose the coupled solver and contact through the public solver module."""
        self.assertIs(newton.solvers.SolverLIMXCoupled, SolverLIMXCoupled)
        self.assertIs(newton.solvers.ConstraintAffineParticleContact, ConstraintAffineParticleContact)


if __name__ == "__main__":
    unittest.main()
