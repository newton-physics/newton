# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ADMM-coupled solvers.

These tests validate generic :class:`SolverAdmmCoupled` ADMM plumbing against a
cloth-plus-rigid-body scene.
"""

from __future__ import annotations

import math
import unittest
from typing import ClassVar

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled.admm_contact_stream import AdmmContactStream, AdmmContactType
from newton._src.solvers.coupled.admm_utils import (
    contact_lambda_update_kernel,
    contact_pp_accumulate_forces_kernel,
    contact_pp_compute_Jv_kernel,
    contact_pp_compute_u_min_kernel,
    contact_rp_accumulate_forces_kernel,
    contact_rp_compute_Jv_kernel,
    contact_rp_compute_u_min_kernel,
    contact_rr_accumulate_forces_kernel,
    contact_rr_compute_Jv_kernel,
    contact_rr_compute_u_min_kernel,
    contact_u_update_kernel,
    joint_box_friction_u_update_kernel,
    u_update_quadratic_kernel,
)
from newton._src.solvers.coupled.interface import (
    CouplingInputStateFlags,
    CouplingInterface,
)
from newton.solvers import (
    SolverBase,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverVBD,
    SolverXPBD,
)
from newton.solvers.coupled_experimental import (
    ModelView,
    SolverAdmmCoupled,
    SolverCoupled,
)


@wp.kernel(enable_backward=False)
def _set_admm_particle_force_kernel(particle_f: wp.array[wp.vec3]):
    particle_f[0] = wp.vec3(2.0, 3.0, 4.0)


@wp.kernel(enable_backward=False)
def _set_admm_particle_qd_kernel(particle_qd: wp.array[wp.vec3]):
    particle_qd[0] = wp.vec3(3.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def _set_admm_plane_angle_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], angle: float):
    body_q[0] = wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle),
    )
    body_qd[0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class _CustomAdmmParticleCopySolver(SolverBase, CouplingInterface):
    """Base test solver that copies particle state."""

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        if state_in.particle_q is not None and state_out.particle_q is not None:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)


class _KinematicAdmmPlaneSolver(_CustomAdmmParticleCopySolver):
    """Test solver that prescribes a fixed kinematic plane angle."""

    def __init__(self, model, angle):
        super().__init__(model)
        self.angle = float(angle)

    def step(self, state_in, state_out, control, contacts, dt):
        super().step(state_in, state_out, control, contacts, dt)
        wp.launch(
            _set_admm_plane_angle_kernel,
            dim=1,
            inputs=[state_out.body_q, state_out.body_qd, self.angle],
            device=self.model.device,
        )


class _AdmmParticleForceNotifySolver(_CustomAdmmParticleCopySolver):
    """Test solver that observes ADMM public particle-force input notifications."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.notified_flags = []
        self.notified_restart = []
        self.notified_particle_f = []
        self.instances.append(self)

    def coupling_notify_input_state_update(self, state, flags, *, restart=False, dt=0.0):
        del dt
        flags = CouplingInputStateFlags(flags)
        self.notified_flags.append(flags)
        self.notified_restart.append(bool(restart))
        if flags & CouplingInputStateFlags.PARTICLE_F:
            self.notified_particle_f.append(state.particle_f.numpy().copy())


class _CustomAdmmInputStateUpdateSolver(_CustomAdmmParticleCopySolver):
    """Test solver that observes ADMM input-state updates through a custom hook."""

    instances: ClassVar[list] = []

    def __init__(self, model):
        super().__init__(model)
        self.update_calls = []
        self.update_restart = []
        self.proximal_shift_calls = 0
        self.input_particle_qd = None
        self.instances.append(self)

    def coupling_notify_input_state_update(self, state, flags, *, restart=False, dt=0.0):
        del dt
        flags = CouplingInputStateFlags(flags)
        self.update_calls.append(flags)
        self.update_restart.append(bool(restart))
        if flags == CouplingInputStateFlags.PARTICLE_QD:
            self.proximal_shift_calls += 1
            self.input_particle_qd = state.particle_qd.numpy().copy()
            wp.launch(_set_admm_particle_qd_kernel, dim=1, inputs=[state.particle_qd], device=self.model.device)


class _CustomEffectiveMassSolver(_CustomAdmmParticleCopySolver):
    """Test solver that reports a fixed effective mass for ADMM endpoints."""

    instances: ClassVar[list] = []

    def __init__(self, model, effective_mass):
        super().__init__(model)
        self.effective_mass = float(effective_mass)
        self.queried_endpoints = []
        self.instances.append(self)

    def coupling_eval_effective_mass(self, endpoint_kind, endpoint_index, endpoint_local_pos, out):
        del endpoint_kind, endpoint_local_pos
        self.queried_endpoints.extend(int(i) for i in endpoint_index.numpy())
        out.fill_(self.effective_mass)


class TestAdmmParticleParticleKernels(unittest.TestCase):
    """Validate ADMM particle-particle contact sign conventions."""

    def test_contact_stream_allocates_endpoint_and_force_buffers(self):
        stream = AdmmContactStream.allocate(
            capacity=3,
            device="cpu",
            contact_type=AdmmContactType.PARTICLE_PARTICLE,
        )

        self.assertEqual(stream.capacity, 3)
        self.assertEqual(stream.contact_type, int(AdmmContactType.PARTICLE_PARTICLE))
        self.assertEqual(stream.count.shape[0], 1)
        np.testing.assert_array_equal(stream.particle_a.numpy(), [-1, -1, -1])
        np.testing.assert_allclose(stream.normal_force.numpy(), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(stream.normal_impulse.numpy(), [0.0, 0.0, 0.0])

    def test_quadratic_attachment_update_includes_damping(self):
        device = "cpu"
        kappa = wp.array([10.0], dtype=float, device=device)
        damping = wp.array([4.0], dtype=float, device=device)
        W = wp.array([3.0], dtype=float, device=device)
        lambda_ = wp.zeros(1, dtype=wp.vec3, device=device)
        Jv = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        u_target = wp.array([wp.vec3(5.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        u = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.launch(
            u_update_quadratic_kernel,
            dim=1,
            inputs=[kappa, damping, W, 2.0, lambda_, Jv, u_target],
            outputs=[u],
            device=device,
        )

        np.testing.assert_allclose(u.numpy()[0], np.array([68.0 / 32.0, 0.0, 0.0]), atol=1.0e-6)

    def test_box_friction_update_soft_thresholds_per_axis(self):
        device = "cpu"
        friction = wp.array([wp.vec3(3.0, 1.0, 0.0)], dtype=wp.vec3, device=device)
        W = wp.array([2.0], dtype=float, device=device)
        lambda_ = wp.zeros(1, dtype=wp.vec3, device=device)
        Jv = wp.array([wp.vec3(1.0, 0.04, -2.0)], dtype=wp.vec3, device=device)
        u = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.launch(
            joint_box_friction_u_update_kernel,
            dim=1,
            inputs=[friction, W, 5.0, lambda_, Jv],
            outputs=[u],
            device=device,
        )

        np.testing.assert_allclose(u.numpy()[0], np.array([0.85, 0.0, -2.0]), atol=1.0e-6)

    def test_particle_particle_jv_and_force_signs(self):
        device = "cpu"
        particle_a = wp.array([0], dtype=int, device=device)
        particle_b = wp.array([1], dtype=int, device=device)
        particle_qd_a = wp.array([wp.vec3(3.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        particle_qd_b = wp.array([wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, -2.0, 0.5)], dtype=wp.vec3, device=device)
        Jv = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.launch(
            contact_pp_compute_Jv_kernel,
            dim=1,
            inputs=[particle_a, particle_b, particle_qd_a, particle_qd_b],
            outputs=[Jv],
            device=device,
        )

        np.testing.assert_allclose(Jv.numpy()[0], np.array([2.0, 3.0, -0.5]), atol=1.0e-6)

        particle_f_a = wp.zeros(2, dtype=wp.vec3, device=device)
        particle_f_b = wp.zeros(2, dtype=wp.vec3, device=device)
        W = wp.array([1.0], dtype=float, device=device)
        lambda_ = wp.array([wp.vec3(0.0, 2.0, 0.0)], dtype=wp.vec3, device=device)
        u = wp.zeros(1, dtype=wp.vec3, device=device)
        Jv_zero = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.launch(
            contact_pp_accumulate_forces_kernel,
            dim=1,
            inputs=[particle_a, particle_b, 0.0, W, lambda_, u, Jv_zero],
            outputs=[particle_f_a, particle_f_b],
            device=device,
        )

        np.testing.assert_allclose(particle_f_a.numpy()[0], np.array([0.0, 2.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(particle_f_b.numpy()[1], np.array([0.0, -2.0, 0.0]), atol=1.0e-6)

    def test_particle_particle_contact_projection_and_force_signs(self):
        device = "cpu"
        particle_a = wp.array([0], dtype=int, device=device)
        particle_b = wp.array([1], dtype=int, device=device)
        normal = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        contact_distance = wp.array([0.0], dtype=float, device=device)
        q_a = wp.array([wp.vec3(-0.1, 0.0, 0.0)], dtype=wp.vec3, device=device)
        q_b = wp.array([wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        u_min = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            contact_pp_compute_u_min_kernel,
            dim=1,
            inputs=[particle_a, particle_b, normal, contact_distance, q_a, q_b, 1.0, 0.1],
            outputs=[u_min],
            device=device,
        )
        np.testing.assert_allclose(u_min.numpy(), [1.0], atol=1.0e-6)

        W = wp.array([1.0], dtype=float, device=device)
        friction = wp.zeros(1, dtype=float, device=device)
        lambda_ = wp.zeros(1, dtype=wp.vec3, device=device)
        Jv = wp.zeros(1, dtype=wp.vec3, device=device)
        u = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch(
            contact_u_update_kernel,
            dim=1,
            inputs=[u_min, W, 10.0, friction, normal, lambda_, Jv],
            outputs=[u],
            device=device,
        )
        np.testing.assert_allclose(u.numpy()[0], np.array([1.0, 0.0, 0.0]), atol=1.0e-6)

        particle_f_a = wp.zeros(2, dtype=wp.vec3, device=device)
        particle_f_b = wp.zeros(2, dtype=wp.vec3, device=device)
        wp.launch(
            contact_pp_accumulate_forces_kernel,
            dim=1,
            inputs=[particle_a, particle_b, 10.0, W, lambda_, u, Jv],
            outputs=[particle_f_a, particle_f_b],
            device=device,
        )
        np.testing.assert_allclose(particle_f_a.numpy()[0], np.array([10.0, 0.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(particle_f_b.numpy()[1], np.array([-10.0, 0.0, 0.0]), atol=1.0e-6)

    def test_contact_projection_satisfies_coulomb_maximum_dissipation(self):
        device = "cpu"
        u_min = wp.array([0.0], dtype=float, device=device)
        W = wp.array([1.0], dtype=float, device=device)
        friction = wp.array([0.5], dtype=float, device=device)
        normal = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        lambda_ = wp.zeros(1, dtype=wp.vec3, device=device)
        Jv = wp.array([wp.vec3(-1.0, -2.0, 0.0)], dtype=wp.vec3, device=device)
        u = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.launch(
            contact_u_update_kernel,
            dim=1,
            inputs=[u_min, W, 1.0, friction, normal, lambda_, Jv],
            outputs=[u],
            device=device,
        )
        wp.launch(
            contact_lambda_update_kernel,
            dim=1,
            inputs=[1.0, W, u, Jv],
            outputs=[lambda_],
            device=device,
        )

        u_np = u.numpy()[0]
        lambda_np = lambda_.numpy()[0]
        lambda_n = lambda_np[0]
        lambda_t = lambda_np - lambda_n * np.array([1.0, 0.0, 0.0])
        jv_t = np.array([0.0, Jv.numpy()[0, 1], Jv.numpy()[0, 2]])

        np.testing.assert_allclose(u_np, np.array([0.0, -1.5, 0.0]), atol=1.0e-6)
        self.assertGreaterEqual(lambda_n, 0.0)
        np.testing.assert_allclose(np.linalg.norm(lambda_t), 0.5 * lambda_n, atol=1.0e-6)
        self.assertLessEqual(float(np.dot(lambda_t, jv_t)), 0.0)

    def test_separated_contact_releases_warm_start(self):
        device = "cpu"
        u_min = wp.array([-1.0e8], dtype=float, device=device)
        W = wp.array([1.0], dtype=float, device=device)
        friction = wp.zeros(1, dtype=float, device=device)
        normal = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        lambda_ = wp.array([wp.vec3(5.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        Jv = wp.zeros(1, dtype=wp.vec3, device=device)
        u = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.launch(
            contact_u_update_kernel,
            dim=1,
            inputs=[u_min, W, 10.0, friction, normal, lambda_, Jv],
            outputs=[u],
            device=device,
        )
        wp.launch(
            contact_lambda_update_kernel,
            dim=1,
            inputs=[10.0, W, u, Jv],
            outputs=[lambda_],
            device=device,
        )

        np.testing.assert_allclose(u.numpy()[0], np.array([-0.5, 0.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(lambda_.numpy()[0], np.zeros(3), atol=1.0e-6)


class TestAdmmComReference(unittest.TestCase):
    """Validate ADMM rigid-particle contacts use COM-referenced twists."""

    def test_rigid_particle_contact_anchor_uses_body_com(self):
        device = "cpu"
        body_ids = wp.array([0], dtype=int, device=device)
        point_a = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        particle_ids = wp.array([0], dtype=int, device=device)
        body_sign = wp.array([1], dtype=int, device=device)
        body_q = wp.array(
            [wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())],
            dtype=wp.transform,
            device=device,
        )
        body_com = wp.array([wp.vec3(0.5, 0.0, 0.0)], dtype=wp.vec3, device=device)
        body_qd = wp.array(
            [wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 2.0)],
            dtype=wp.spatial_vector,
            device=device,
        )
        particle_qd = wp.zeros(1, dtype=wp.vec3, device=device)
        Jv = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.launch(
            contact_rp_compute_Jv_kernel,
            dim=1,
            inputs=[body_ids, point_a, particle_ids, body_sign, body_q, body_com, body_qd, particle_qd],
            outputs=[Jv],
            device=device,
        )

        np.testing.assert_allclose(Jv.numpy()[0], np.array([0.0, 1.0, 0.0]), atol=1.0e-6)

        body_f = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        particle_f = wp.zeros(1, dtype=wp.vec3, device=device)
        W = wp.array([1.0], dtype=float, device=device)
        lambda_ = wp.array([wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3, device=device)
        u = wp.zeros(1, dtype=wp.vec3, device=device)
        Jv_zero = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.launch(
            contact_rp_accumulate_forces_kernel,
            dim=1,
            inputs=[
                body_ids,
                point_a,
                particle_ids,
                body_sign,
                body_q,
                body_com,
                0.0,
                W,
                lambda_,
                u,
                Jv_zero,
            ],
            outputs=[body_f, particle_f],
            device=device,
        )

        np.testing.assert_allclose(body_f.numpy()[0], np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.5]), atol=1.0e-6)
        np.testing.assert_allclose(particle_f.numpy()[0], np.array([0.0, -1.0, 0.0]), atol=1.0e-6)

    def test_rigid_particle_contact_force_sign(self):
        device = "cpu"
        body_ids = wp.array([0], dtype=int, device=device)
        point_body = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        particle_ids = wp.array([0], dtype=int, device=device)
        normal = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        body_sign = wp.array([1], dtype=int, device=device)
        contact_distance = wp.array([0.0], dtype=float, device=device)
        body_q = wp.array(
            [wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity())],
            dtype=wp.transform,
            device=device,
        )
        body_com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        particle_q = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        u_min = wp.zeros(1, dtype=float, device=device)

        wp.launch(
            contact_rp_compute_u_min_kernel,
            dim=1,
            inputs=[
                body_ids,
                point_body,
                particle_ids,
                normal,
                body_sign,
                contact_distance,
                body_q,
                particle_q,
                1.0,
                0.1,
            ],
            outputs=[u_min],
            device=device,
        )
        np.testing.assert_allclose(u_min.numpy(), [1.0], atol=1.0e-6)

        body_f = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        particle_f = wp.zeros(1, dtype=wp.vec3, device=device)
        W = wp.array([1.0], dtype=float, device=device)
        lambda_ = wp.zeros(1, dtype=wp.vec3, device=device)
        u = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        Jv = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch(
            contact_rp_accumulate_forces_kernel,
            dim=1,
            inputs=[
                body_ids,
                point_body,
                particle_ids,
                body_sign,
                body_q,
                body_com,
                10.0,
                W,
                lambda_,
                u,
                Jv,
            ],
            outputs=[body_f, particle_f],
            device=device,
        )

        np.testing.assert_allclose(body_f.numpy()[0], np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(particle_f.numpy()[0], np.array([-10.0, 0.0, 0.0]), atol=1.0e-6)

    def test_rigid_rigid_contact_force_sign(self):
        device = "cpu"
        body_a = wp.array([0], dtype=int, device=device)
        body_b = wp.array([1], dtype=int, device=device)
        point_a = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        point_b = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        normal = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        contact_distance = wp.array([0.0], dtype=float, device=device)
        body_q = wp.array(
            [
                wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            dtype=wp.transform,
            device=device,
        )
        body_com = wp.array([wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        body_qd = wp.array(
            [
                wp.spatial_vector(1.0, 2.0, 0.0, 0.0, 0.0, 0.0),
                wp.spatial_vector(-1.0, 0.5, 0.0, 0.0, 0.0, 0.0),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        Jv = wp.zeros(1, dtype=wp.vec3, device=device)

        wp.launch(
            contact_rr_compute_Jv_kernel,
            dim=1,
            inputs=[body_a, point_a, body_b, point_b, body_q, body_com, body_qd, body_q, body_com, body_qd],
            outputs=[Jv],
            device=device,
        )
        np.testing.assert_allclose(Jv.numpy()[0], np.array([2.0, 1.5, 0.0]), atol=1.0e-6)

        u_min = wp.zeros(1, dtype=float, device=device)
        wp.launch(
            contact_rr_compute_u_min_kernel,
            dim=1,
            inputs=[body_a, point_a, body_b, point_b, normal, contact_distance, body_q, body_q, 1.0, 0.1],
            outputs=[u_min],
            device=device,
        )
        np.testing.assert_allclose(u_min.numpy(), [1.0], atol=1.0e-6)

        body_f_a = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        body_f_b = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        W = wp.array([1.0], dtype=float, device=device)
        lambda_ = wp.zeros(1, dtype=wp.vec3, device=device)
        u = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        Jv_zero = wp.zeros(1, dtype=wp.vec3, device=device)
        wp.launch(
            contact_rr_accumulate_forces_kernel,
            dim=1,
            inputs=[body_a, point_a, body_b, point_b, body_q, body_com, body_q, body_com, 10.0, W, lambda_, u, Jv_zero],
            outputs=[body_f_a, body_f_b],
            device=device,
        )

        np.testing.assert_allclose(body_f_a.numpy()[0], np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1.0e-6)
        np.testing.assert_allclose(body_f_b.numpy()[1], np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1.0e-6)


def _build_cloth_rigid_scene(
    rigid_pos: tuple[float, float, float] = (0.0, 0.0, 1.5),
    rigid_mass: float = 0.05,
    cloth_pos: tuple[float, float, float] = (-0.25, -0.25, 1.5),
    dim_xy: int = 5,
    fix_cloth_edges: bool = True,
) -> tuple[newton.Model, int, int, int]:
    """Build a pinned cloth + free rigid body scene for attachment tests."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    rigid_start = builder.body_count
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(*rigid_pos), q=wp.quat_identity()),
        mass=rigid_mass,
        inertia=wp.mat33(np.eye(3) * 0.001),
    )
    builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
    rigid_end = builder.body_count

    particle_start = builder.particle_count
    builder.add_cloth_grid(
        pos=wp.vec3(*cloth_pos),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        fix_left=fix_cloth_edges,
        fix_right=fix_cloth_edges,
        dim_x=dim_xy,
        dim_y=dim_xy,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.05,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=1e-2,
        edge_ke=0.01,
        edge_kd=1e-2,
        particle_radius=0.01,
    )
    center = dim_xy // 2
    particle_idx = particle_start + center * (dim_xy + 1) + center
    builder.color()
    model = builder.finalize()
    return model, rigid_start, rigid_end, particle_idx


def _make_solver(
    model: newton.Model,
    rigid_start: int,
    rigid_end: int,
    admm_iters: int = 5,
    rho: float = 50.0,
    gamma: float = 0.0,
    baumgarte: float = 0.1,
):
    """Standard MuJoCo/VBD ADMM configuration used across tests."""
    mjc_ids = wp.array(list(range(rigid_start, rigid_end)), dtype=int)
    vbd_ids = wp.array(
        [i for i in range(model.body_count) if i < rigid_start or i >= rigid_end],
        dtype=int,
    )
    return SolverAdmmCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="mjc",
                solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=20),
                bodies=[int(i) for i in mjc_ids.numpy()],
                joints=list(range(model.joint_count)),
            ),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=5),
                bodies=[int(i) for i in vbd_ids.numpy()],
                particles=list(range(model.particle_count)),
            ),
        ],
        coupling=SolverAdmmCoupled.Config(
            iterations=admm_iters,
            rho=rho,
            gamma=gamma,
            baumgarte=baumgarte,
        ),
    )


def _run(solver, model: newton.Model, n_steps: int = 30, dt: float = 1.0 / 60.0):
    """Run ``n_steps`` of simulation and return (body_q, particle_q)."""
    state_0 = model.state()
    state_1 = model.state()
    contacts = model.contacts()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(n_steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.particle_q.numpy().copy()


def _build_two_particle_scene() -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particle(pos=(-0.5, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.add_particle(pos=(0.5, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.color()
    return builder.finalize(device="cpu")


def _build_two_particle_contact_scene(
    gap: float = -0.1,
    vel_a: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vel_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particle(pos=(gap, 0.0, 0.0), vel=vel_a, mass=1.0, radius=0.0)
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=vel_b, mass=1.0, radius=0.0)
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _run_particles(solver, model: newton.Model, n_steps: int = 5, dt: float = 1.0 / 60.0):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.particle_q.numpy().copy()


def _make_vbd_xpbd_particle_solver(model: newton.Model):
    return SolverAdmmCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=2),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="xpbd",
                solver=lambda v: SolverXPBD(model=v, iterations=2),
                particles=[1],
            ),
        ],
        coupling=SolverAdmmCoupled.Config(
            iterations=8,
            rho=20.0,
            baumgarte=0.2,
        ),
    )


def _make_semi_particle_solver(model: newton.Model):
    return SolverAdmmCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="a",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="b",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[1],
            ),
        ],
        coupling=SolverAdmmCoupled.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _build_body_particle_contact_scene() -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.1, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_body_particle_attachment_scene(enabled: bool = True) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    particle = builder.add_particle(pos=(0.3, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    SolverAdmmCoupled.add_body_particle_attachment(
        builder,
        body,
        particle,
        stiffness=500.0,
        enabled=enabled,
    )
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_two_body_contact_scene(gap: float = -0.1) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_body(
        xform=wp.transform(p=wp.vec3(gap, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.color()
    return builder.finalize(device="cpu")


def _build_collision_contact_scene() -> tuple[newton.Model, int, int, int]:
    builder = newton.ModelBuilder(gravity=0.0)
    tray_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=0.1,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )

    tray_cfg = newton.ModelBuilder.ShapeConfig()
    tray_cfg.has_shape_collision = False
    tray_cfg.has_particle_collision = True
    tray_shape = builder.add_shape_box(
        tray_body,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.025), q=wp.quat_identity()),
        hx=0.1,
        hy=0.1,
        hz=0.025,
        cfg=tray_cfg,
    )
    particle = builder.add_particle(
        pos=(0.0, 0.0, 0.12),
        vel=(0.0, 0.0, -0.5),
        mass=0.025,
        radius=0.025,
    )
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    return model, particle, tray_body, tray_shape


def _run_body_particle(solver, model: newton.Model, n_steps: int = 4, dt: float = 1.0 / 60.0):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.particle_q.numpy().copy()


def _run_bodies(
    solver,
    model: newton.Model,
    n_steps: int = 4,
    dt: float = 1.0 / 60.0,
    body_qd: np.ndarray | None = None,
):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    if body_qd is not None:
        state_0.body_qd = wp.array(body_qd, dtype=wp.spatial_vector, device=model.device)

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.body_qd.numpy().copy()


def _make_semi_body_particle_solver(model: newton.Model):
    return SolverAdmmCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="body",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[0],
            ),
            SolverCoupled.Entry(
                name="particle",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[0],
            ),
        ],
        coupling=SolverAdmmCoupled.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _make_semi_body_body_solver(model: newton.Model):
    return SolverAdmmCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="a",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[0],
            ),
            SolverCoupled.Entry(
                name="b",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[1],
            ),
        ],
        coupling=SolverAdmmCoupled.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _build_inclined_plane_particle_box_scene(
    angle: float,
    *,
    particle_radius: float = 0.025,
    box_half_extent: float = 0.06,
    penetration: float = 0.002,
) -> tuple[newton.Model, int, int, list[int]]:
    builder = newton.ModelBuilder(gravity=-10.0)
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    plane_cfg = newton.ModelBuilder.ShapeConfig()
    plane_cfg.has_shape_collision = False
    plane_cfg.has_particle_collision = True
    plane_shape = builder.add_shape_plane(
        body=plane_body,
        xform=wp.transform_identity(),
        width=2.0,
        length=2.0,
        cfg=plane_cfg,
    )

    n = np.array([math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)
    tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)
    binormal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    center = (particle_radius - penetration) * n

    particle_ids = []
    for tangent_sign in (-1.0, 1.0):
        for binormal_sign in (-1.0, 1.0):
            pos = center + tangent_sign * box_half_extent * tangent + binormal_sign * box_half_extent * binormal
            particle_ids.append(
                builder.add_particle(
                    pos=tuple(float(x) for x in pos),
                    vel=(0.0, 0.0, 0.0),
                    mass=0.25,
                    radius=particle_radius,
                )
            )

    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    return model, plane_body, plane_shape, particle_ids


def _make_admm_inclined_plane_particle_box_solver(
    model: newton.Model,
    plane_body: int,
    particle_ids: list[int],
    angle: float,
    friction: float,
) -> SolverAdmmCoupled:
    return SolverAdmmCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="plane",
                solver=lambda v: _KinematicAdmmPlaneSolver(model=v, angle=angle),
                bodies=[plane_body],
            ),
            SolverCoupled.Entry(
                name="box",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=particle_ids,
            ),
        ],
        coupling=SolverAdmmCoupled.Config(
            iterations=18,
            rho=50.0,
            baumgarte=0.1,
            contact_pairs=[
                SolverAdmmCoupled.ContactPair(
                    source="plane",
                    destination="box",
                    detection_margin=0.04,
                )
            ],
        ),
    )


def _run_inclined_plane_particle_box(
    angle: float,
    friction: float,
    *,
    steps: int = 120,
    dt: float = 1.0 / 360.0,
) -> tuple[float, float, int]:
    model, plane_body, _, particle_ids = _build_inclined_plane_particle_box_scene(angle)
    # ADMM derives friction from material properties; set both sides so the
    # geometric-mean combine reduces to the requested coefficient.
    model.particle_mu = float(friction)
    model.shape_material_mu = wp.full(model.shape_count, float(friction), dtype=wp.float32, device=model.device)
    solver = _make_admm_inclined_plane_particle_box_solver(
        model,
        plane_body,
        particle_ids,
        angle,
        friction,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    initial_com = np.mean(state_0.particle_q.numpy()[particle_ids], axis=0)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_q = state_0.particle_q.numpy()[particle_ids]
    final_qd = state_0.particle_qd.numpy()[particle_ids]
    final_com = np.mean(final_q, axis=0)
    final_vel = np.mean(final_qd, axis=0)
    tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)
    displacement = float(np.dot(final_com - initial_com, tangent))
    velocity = float(np.dot(final_vel, tangent))
    return displacement, velocity, solver.collision_contact_count_max


def _rotate_y_np(v: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]], dtype=np.float32)


def _build_inclined_plane_rigid_box_scene(
    angle: float,
    *,
    box_half_height: float = 0.08,
    penetration: float = 0.002,
) -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=-10.0)
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    local_center = np.array([0.0, 0.0, box_half_height - penetration], dtype=np.float32)
    box_center = _rotate_y_np(local_center, angle)
    box_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(float(box_center[0]), float(box_center[1]), float(box_center[2])),
            plane_q,
        ),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    builder.color()
    return builder.finalize(device="cpu"), plane_body, box_body


def _build_collision_inclined_plane_rigid_box_scene(
    angle: float,
    *,
    box_half_height: float = 0.08,
    penetration: float = 0.004,
) -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=-10.0)
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.has_shape_collision = True
    cfg.has_particle_collision = False
    cfg.density = 0.0
    builder.add_shape_box(
        plane_body,
        xform=wp.transform(wp.vec3(1.0, 0.0, -0.025), wp.quat_identity()),
        hx=3.0,
        hy=0.4,
        hz=0.025,
        cfg=cfg,
    )

    local_center = np.array([0.0, 0.0, box_half_height - penetration], dtype=np.float32)
    box_center = _rotate_y_np(local_center, angle)
    box_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(float(box_center[0]), float(box_center[1]), float(box_center[2])),
            plane_q,
        ),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    builder.add_shape_box(
        box_body,
        hx=0.08,
        hy=0.08,
        hz=box_half_height,
        cfg=cfg,
    )
    builder.color()
    return builder.finalize(device="cpu"), plane_body, box_body


def _make_collision_admm_inclined_plane_rigid_box_solver(
    model: newton.Model,
    plane_body: int,
    box_body: int,
    angle: float,
    friction: float,
) -> SolverAdmmCoupled:
    return SolverAdmmCoupled(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="plane",
                solver=lambda v: _KinematicAdmmPlaneSolver(model=v, angle=angle),
                bodies=[plane_body],
            ),
            SolverCoupled.Entry(
                name="box",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[box_body],
            ),
        ],
        coupling=SolverAdmmCoupled.Config(
            iterations=30,
            rho=5.0,
            gamma=0.2,
            baumgarte=0.03,
            contact_pairs=[
                SolverAdmmCoupled.ContactPair(
                    source="plane",
                    destination="box",
                )
            ],
        ),
    )


def _run_collision_inclined_plane_rigid_box(
    angle: float,
    friction: float,
    *,
    steps: int = 120,
    dt: float = 1.0 / 360.0,
) -> tuple[float, float, float, int]:
    model, plane_body, box_body = _build_collision_inclined_plane_rigid_box_scene(angle)
    model.shape_material_mu = wp.full(model.shape_count, float(friction), dtype=wp.float32, device=model.device)
    solver = _make_collision_admm_inclined_plane_rigid_box_solver(model, plane_body, box_body, angle, friction)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_pos = state_0.body_q.numpy()[box_body, :3].copy()
    min_gap = math.inf
    normal = _rotate_y_np(np.array([0.0, 0.0, 1.0], dtype=np.float32), angle)
    tangent = _rotate_y_np(np.array([1.0, 0.0, 0.0], dtype=np.float32), angle)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
        center_gap = float(np.dot(normal, state_0.body_q.numpy()[box_body, :3]))
        min_gap = min(min_gap, center_gap - 0.08)

    final_pos = state_0.body_q.numpy()[box_body, :3]
    final_qd = state_0.body_qd.numpy()[box_body, :3]
    displacement = float(np.dot(final_pos - initial_pos, tangent))
    velocity = float(np.dot(final_qd, tangent))
    return displacement, velocity, min_gap, solver.collision_contact_count_max


class TestAdmmScalePartMass(unittest.TestCase):
    """ModelView.scale_particle_mass overrides particle mass/inv_mass."""

    def test_scale(self):
        builder = newton.ModelBuilder()
        for _ in range(3):
            builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=2.0)
        model = builder.finalize(device="cpu")

        view = ModelView(model, "test")
        view.scale_particle_mass(4.0)

        np.testing.assert_allclose(view.particle_mass.numpy(), [8.0, 8.0, 8.0])
        np.testing.assert_allclose(view.particle_inv_mass.numpy(), [0.125, 0.125, 0.125])
        # Parent unchanged.
        np.testing.assert_allclose(model.particle_mass.numpy(), [2.0, 2.0, 2.0])


class TestAdmmCouplingHooks(unittest.TestCase):
    """ADMM should route solver-private coupling operations through hooks."""

    def test_particle_force_input_uses_default_buffer_and_notifies(self):
        _AdmmParticleForceNotifySolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")
        solver = SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="p", solver=_AdmmParticleForceNotifySolver, particles=[0]),
            ],
            coupling=SolverAdmmCoupled.Config(iterations=1),
        )

        state_0 = model.state()
        state_1 = model.state()
        wp.launch(_set_admm_particle_force_kernel, dim=1, inputs=[state_0.particle_f], device=model.device)
        solver.step(state_0, state_1, model.control(), contacts=None, dt=1.0 / 60.0)

        custom_solver = _AdmmParticleForceNotifySolver.instances[-1]
        self.assertTrue(any(flags & CouplingInputStateFlags.PARTICLE_F for flags in custom_solver.notified_flags))
        np.testing.assert_allclose(custom_solver.notified_particle_f[-1][0], np.array([2.0, 3.0, 4.0]), atol=1.0e-6)
        np.testing.assert_allclose(
            solver._entries["p"].state_0.particle_f.numpy()[0],
            np.array([2.0, 3.0, 4.0]),
            atol=1.0e-6,
        )

    def test_velocity_proximal_shift_uses_input_state_update_hook(self):
        _CustomAdmmInputStateUpdateSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")
        solver = SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="p", solver=_CustomAdmmInputStateUpdateSolver, particles=[0]),
            ],
            coupling=SolverAdmmCoupled.Config(iterations=1, gamma=2.0),
        )

        state_0 = model.state()
        state_1 = model.state()
        solver.step(state_0, state_1, model.control(), contacts=None, dt=1.0 / 60.0)

        custom_solver = _CustomAdmmInputStateUpdateSolver.instances[-1]
        self.assertTrue(
            any(
                (flags & CouplingInputStateFlags.PARTICLE) == CouplingInputStateFlags.PARTICLE
                for flags in custom_solver.update_calls
            )
        )
        self.assertEqual(custom_solver.proximal_shift_calls, 1)
        np.testing.assert_allclose(custom_solver.input_particle_qd[0], np.zeros(3), atol=1.0e-6)
        np.testing.assert_allclose(state_1.particle_qd.numpy()[0], np.array([3.0, 0.0, 0.0]), atol=1.0e-6)

    def test_iteration_restart_notifies_input_state_update_hook(self):
        _CustomAdmmInputStateUpdateSolver.instances.clear()
        builder = newton.ModelBuilder(gravity=0.0)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
        model = builder.finalize(device="cpu")
        solver = SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="p", solver=_CustomAdmmInputStateUpdateSolver, particles=[0]),
            ],
            coupling=SolverAdmmCoupled.Config(iterations=2),
        )

        state_0 = model.state()
        state_1 = model.state()
        solver.step(state_0, state_1, model.control(), contacts=None, dt=1.0 / 60.0)

        custom_solver = _CustomAdmmInputStateUpdateSolver.instances[-1]
        self.assertTrue(
            any(
                restart and bool(flags & CouplingInputStateFlags.PARTICLE)
                for flags, restart in zip(custom_solver.update_calls, custom_solver.update_restart, strict=True)
            )
        )

    def test_collision_contact_uses_custom_effective_mass(self):
        _CustomEffectiveMassSolver.instances.clear()
        model = _build_two_particle_contact_scene(gap=-0.08)
        solver = SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=lambda v: _CustomEffectiveMassSolver(model=v, effective_mass=4.0),
                    particles=[0],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=lambda v: _CustomEffectiveMassSolver(model=v, effective_mass=9.0),
                    particles=[1],
                ),
            ],
            coupling=SolverAdmmCoupled.Config(
                contact_pairs=[
                    SolverAdmmCoupled.ContactPair(source="a", destination="b", contact_distance=0.1),
                ],
            ),
        )

        solver._refresh_collision_contact_groups(model.state())
        group = solver._admm_dynamic_pp_contact_groups[0]
        expected = (36.0 / 13.0) ** 0.5
        self.assertEqual(int(group.active_count.numpy()[0]), 1)
        np.testing.assert_allclose(group.W.numpy()[:1], [expected], atol=1.0e-6)


class TestAdmmSmoke(unittest.TestCase):
    """End-to-end: construct, run, verify state advances without NaNs."""

    def test_construct_and_step_no_attachments(self):
        model, rs, re, _ = _build_cloth_rigid_scene()
        solver = _make_solver(model, rs, re, admm_iters=1)

        body_q, particle_q = _run(solver, model, n_steps=10)
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(particle_q)))

    def test_admm_iters_idempotent_with_no_coupling(self):
        """With gamma=0 and no attachments the iteration count should not
        change the result (no coupling = idempotent outer loop)."""
        model_a, rs, re, _ = _build_cloth_rigid_scene()
        solver_a = _make_solver(model_a, rs, re, admm_iters=1, gamma=0.0)
        body_a, part_a = _run(solver_a, model_a, n_steps=5)

        model_b, rs, re, _ = _build_cloth_rigid_scene()
        solver_b = _make_solver(model_b, rs, re, admm_iters=4, gamma=0.0)
        body_b, part_b = _run(solver_b, model_b, n_steps=5)

        np.testing.assert_allclose(body_a, body_b, atol=1e-6)
        np.testing.assert_allclose(part_a, part_b, atol=1e-6)


class TestAdmmProximal(unittest.TestCase):
    """Proximal term changes the step outcome and stays stable."""

    def test_gamma_changes_state(self):
        # Place the rigid body high so it stays in free-fall across the
        # window and the proximal term's velocity damping is observable.
        model_ref, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_ref = _make_solver(model_ref, rs, re, admm_iters=3, gamma=0.0)
        body_ref, part_ref = _run(solver_ref, model_ref, n_steps=5)

        model_g, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_g = _make_solver(model_g, rs, re, admm_iters=3, gamma=5.0)
        body_g, part_g = _run(solver_g, model_g, n_steps=5)

        # Larger gamma slows the body's effective acceleration → the two
        # runs' body z should diverge during free-fall.
        dz_body = abs(body_ref[0, 2] - body_g[0, 2])
        self.assertGreater(
            dz_body,
            1e-3,
            f"gamma=5 barely affected body z: ref={body_ref[0, 2]:.5f}, g={body_g[0, 2]:.5f}",
        )
        self.assertFalse(np.allclose(part_ref, part_g, atol=1e-3))
        self.assertTrue(np.all(np.isfinite(body_g)))
        self.assertTrue(np.all(np.isfinite(part_g)))


class TestAdmmModelJointInterface(unittest.TestCase):
    """Cross-solver model joints are converted to ADMM attachments."""

    def _build_two_body_joint_scene(
        self,
        joint_type: str = "ball",
        *,
        friction: float = 0.0,
    ) -> tuple[newton.Model, int, int, int]:
        builder = newton.ModelBuilder(gravity=0.0)
        parent = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        child = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.3, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        if joint_type == "ball":
            joint = builder.add_joint_ball(
                parent=parent,
                child=child,
                friction=friction,
                collision_filter_parent=False,
            )
        elif joint_type == "fixed":
            joint = builder.add_joint_fixed(parent=parent, child=child, collision_filter_parent=False)
        elif joint_type == "revolute":
            joint = builder.add_joint_revolute(
                parent=parent, child=child, friction=friction, collision_filter_parent=False
            )
        else:
            raise ValueError(joint_type)
        builder.color()
        return builder.finalize(device="cpu"), parent, child, joint

    def _make_two_body_joint_solver(self, model: newton.Model, parent: int, child: int, **coupling_kwargs):
        return SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="parent",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[parent],
                ),
                SolverCoupled.Entry(
                    name="child",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[child],
                ),
            ],
            coupling=SolverAdmmCoupled.Config(
                iterations=12,
                rho=40.0,
                baumgarte=0.5,
                joint_stiffness=500.0,
                joint_angular_stiffness=50.0,
                **coupling_kwargs,
            ),
        )

    def test_ball_joint_attachment_closes_anchor_gap(self):
        model, parent, child, _ = self._build_two_body_joint_scene("ball")
        solver = self._make_two_body_joint_solver(model, parent, child)
        initial_gap = abs(model.state().body_q.numpy()[child, 0] - model.state().body_q.numpy()[parent, 0])

        body_q, _ = _run_bodies(solver, model, n_steps=8, dt=1.0 / 120.0)
        final_gap = abs(body_q[child, 0] - body_q[parent, 0])

        self.assertEqual(len(solver._admm_rr_groups), 1)
        self.assertEqual(len(solver._admm_rr_angular_groups), 0)
        self.assertLess(final_gap, 0.5 * initial_gap)

    def test_ball_joint_friction_builds_angular_box_friction_row(self):
        model, parent, child, _ = self._build_two_body_joint_scene("ball", friction=2.5)
        solver = self._make_two_body_joint_solver(model, parent, child)

        self.assertEqual(len(solver._admm_rr_groups), 1)
        self.assertEqual(len(solver._admm_rr_angular_groups), 0)
        self.assertEqual(len(solver._admm_rr_angular_friction_groups), 1)
        np.testing.assert_allclose(
            solver._admm_rr_angular_friction_groups[0].friction.numpy()[0],
            np.array([2.5, 2.5, 2.5]),
            atol=1.0e-6,
        )

    def test_fixed_joint_builds_linear_and_angular_rows(self):
        model, parent, child, _ = self._build_two_body_joint_scene("fixed")
        solver = self._make_two_body_joint_solver(model, parent, child)

        self.assertEqual(len(solver._admm_rr_groups), 1)
        self.assertEqual(len(solver._admm_rr_angular_groups), 1)

    def test_revolute_joint_builds_linear_and_two_axis_angular_rows(self):
        model, parent, child, _ = self._build_two_body_joint_scene("revolute")
        solver = self._make_two_body_joint_solver(model, parent, child)

        self.assertEqual(len(solver._admm_rr_groups), 1)
        self.assertEqual(len(solver._admm_rr_angular_groups), 0)
        self.assertEqual(len(solver._admm_rr_revolute_angular_groups), 1)

    def test_revolute_joint_friction_builds_hinge_friction_row(self):
        model, parent, child, _ = self._build_two_body_joint_scene("revolute", friction=2.5)
        solver = self._make_two_body_joint_solver(model, parent, child)

        self.assertEqual(len(solver._admm_rr_angular_friction_groups), 1)
        np.testing.assert_allclose(
            solver._admm_rr_angular_friction_groups[0].friction.numpy()[0],
            np.array([2.5, 0.0, 0.0]),
            atol=1.0e-6,
        )

    def test_joint_damping_is_stored_on_attachment_rows(self):
        model, parent, child, _ = self._build_two_body_joint_scene("fixed")
        solver = self._make_two_body_joint_solver(
            model,
            parent,
            child,
            joint_damping=7.0,
            joint_angular_damping=3.0,
        )

        np.testing.assert_allclose(solver._admm_rr_groups[0].damping.numpy(), [7.0])
        np.testing.assert_allclose(solver._admm_rr_angular_groups[0].damping.numpy(), [3.0])

    def test_rejects_cross_solver_joint_owned_by_subsolver(self):
        model, parent, child, joint = self._build_two_body_joint_scene("ball")
        with self.assertRaisesRegex(ValueError, "must not be owned"):
            SolverAdmmCoupled(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="parent",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[parent],
                        joints=[joint],
                    ),
                    SolverCoupled.Entry(
                        name="child",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[child],
                    ),
                ],
                coupling=SolverAdmmCoupled.Config(),
            )


class TestAdmmBodyParticleAttachment(unittest.TestCase):
    """Custom model attributes are converted to rigid-particle ADMM attachments."""

    def test_custom_attribute_attachment_closes_gap(self):
        model = _build_body_particle_attachment_scene()
        solver = _make_semi_body_particle_solver(model)
        initial_gap = np.linalg.norm(model.state().body_q.numpy()[0, :3] - model.state().particle_q.numpy()[0])

        body_q, particle_q = _run_body_particle(solver, model, n_steps=8, dt=1.0 / 120.0)
        final_gap = np.linalg.norm(body_q[0, :3] - particle_q[0])

        self.assertEqual(len(solver._admm_rp_groups), 1)
        self.assertLess(final_gap, 0.5 * initial_gap)

    def test_disabled_custom_attribute_attachment_is_ignored(self):
        model = _build_body_particle_attachment_scene(enabled=False)
        solver = _make_semi_body_particle_solver(model)

        self.assertEqual(len(solver._admm_rp_groups), 0)


class TestAdmmExternalForces(unittest.TestCase):
    """External forces set on ``state_in.body_f`` / ``particle_f`` by the
    caller (e.g. a viewer gizmo) must reach the sub-solvers."""

    def test_body_f_reaches_mujoco(self):
        """An upward ``body_f`` on the rigid sphere should slow its fall
        compared to the zero-force baseline."""
        # Baseline: no external force, body falls under gravity.
        model_a, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_a = _make_solver(model_a, rs, re, admm_iters=1)
        state_0 = model_a.state()
        state_1 = model_a.state()
        contacts = model_a.contacts()
        control = model_a.control()
        newton.eval_fk(model_a, model_a.joint_q, model_a.joint_qd, state_0)
        for _ in range(5):
            state_0.clear_forces()
            model_a.collide(state_0, contacts)
            solver_a.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        z_baseline = state_0.body_q.numpy()[0, 2]

        # With a strong upward body_f applied each step, the body should fall
        # less (or even rise).
        model_b, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_b = _make_solver(model_b, rs, re, admm_iters=1)
        state_0 = model_b.state()
        state_1 = model_b.state()
        contacts = model_b.contacts()
        control = model_b.control()
        newton.eval_fk(model_b, model_b.joint_q, model_b.joint_qd, state_0)
        body_idx = rs  # only MuJoCo body
        body_mass = float(model_b.body_mass.numpy()[body_idx])
        upward_force = 5.0 * body_mass * 9.81  # 5 g upward wrench
        for _ in range(5):
            state_0.clear_forces()
            wrench = np.zeros((model_b.body_count, 6), dtype=np.float32)
            wrench[body_idx, 2] = upward_force  # linear z
            state_0.body_f = wp.array(wrench, dtype=wp.spatial_vector, device=model_b.device)
            model_b.collide(state_0, contacts)
            solver_b.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        z_with_force = state_0.body_q.numpy()[0, 2]

        self.assertGreater(
            z_with_force,
            z_baseline + 0.02,
            f"external body_f didn't reach MuJoCo: baseline z={z_baseline:.4f}, "
            f"with 5g upward force z={z_with_force:.4f}",
        )


class TestAdmmCollisionDetection(unittest.TestCase):
    """Collision-detected ADMM contact constraints."""

    def test_collision_particle_particle_contacts_are_refreshed_in_solver(self):
        model = _build_two_particle_contact_scene(gap=-0.08)
        solver = SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[0],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[1],
                ),
            ],
            coupling=SolverAdmmCoupled.Config(
                iterations=10,
                rho=30.0,
                baumgarte=0.5,
                contact_pairs=[
                    SolverAdmmCoupled.ContactPair(source="a", destination="b", contact_distance=0.1),
                ],
            ),
        )

        q_contact = _run_particles(solver, model, n_steps=4)

        self.assertGreater(solver.collision_contact_count_max, 0)
        self.assertGreater(q_contact[1, 0] - q_contact[0, 0], 0.08 + 1.0e-3)

    def test_collision_particle_particle_contacts_are_persistent(self):
        model = _build_two_particle_contact_scene(gap=-0.08)
        solver = SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[0],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[1],
                ),
            ],
            coupling=SolverAdmmCoupled.Config(
                contact_pairs=[
                    SolverAdmmCoupled.ContactPair(source="a", destination="b", contact_distance=0.1),
                ],
            ),
        )
        state = model.state()

        solver._refresh_collision_contact_groups(state)
        group = solver._admm_dynamic_pp_contact_groups[0]
        contact_stream = group.contact_stream
        lambda_array = group.lambda_
        self.assertEqual(int(group.active_count.numpy()[0]), 1)
        self.assertEqual(int(contact_stream.count.numpy()[0]), 1)
        self.assertEqual(int(contact_stream.count_max.numpy()[0]), 1)
        np.testing.assert_array_equal(contact_stream.particle_a.numpy()[:1], [0])
        np.testing.assert_array_equal(contact_stream.particle_b.numpy()[:1], [1])

        group.u.fill_(wp.vec3(1.25, 0.0, 0.0))
        group.lambda_.fill_(wp.vec3(2.5, 0.0, 0.0))
        solver._refresh_collision_contact_groups(state)

        self.assertIs(solver._admm_dynamic_pp_contact_groups[0], group)
        self.assertIs(group.lambda_, lambda_array)
        np.testing.assert_allclose(group.u.numpy()[:1], [[1.25, 0.0, 0.0]], atol=1.0e-6)
        np.testing.assert_allclose(group.lambda_.numpy()[:1], [[2.5, 0.0, 0.0]], atol=1.0e-6)

    def test_collision_particle_particle_stream_reports_normal_force(self):
        model = _build_two_particle_contact_scene(gap=-0.08)
        solver = SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[0],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[1],
                ),
            ],
            coupling=SolverAdmmCoupled.Config(
                iterations=4,
                rho=30.0,
                baumgarte=0.5,
                contact_pairs=[
                    SolverAdmmCoupled.ContactPair(source="a", destination="b", contact_distance=0.1),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        solver.step(state_0, state_1, model.control(), contacts=None, dt=1.0 / 60.0)

        stream = solver._admm_dynamic_pp_contact_groups[0].contact_stream
        self.assertEqual(int(stream.count.numpy()[0]), 1)
        self.assertGreater(float(stream.normal_force.numpy()[0]), 0.0)
        self.assertGreater(float(stream.normal_impulse.numpy()[0]), 0.0)

    def test_collision_frictional_contact_matches_inclined_plane_box_motion(self):
        friction = 0.4
        angle = math.radians(35.0)
        dt = 1.0 / 360.0
        steps = 120
        displacement, velocity, contact_count = _run_inclined_plane_particle_box(
            angle,
            friction,
            steps=steps,
            dt=dt,
        )

        t = steps * dt
        acceleration = 10.0 * (math.sin(angle) - friction * math.cos(angle))
        expected_displacement = 0.5 * acceleration * t * t
        expected_velocity = acceleration * t

        self.assertGreater(contact_count, 0)
        self.assertGreater(acceleration, 0.0)
        self.assertAlmostEqual(displacement, expected_displacement, delta=0.45 * expected_displacement)
        self.assertAlmostEqual(velocity, expected_velocity, delta=0.45 * expected_velocity)

    def test_collision_frictional_contact_holds_subcritical_inclined_box(self):
        friction = 0.4
        angle = math.radians(15.0)
        displacement, velocity, contact_count = _run_inclined_plane_particle_box(
            angle,
            friction,
            steps=120,
            dt=1.0 / 360.0,
        )

        self.assertGreater(contact_count, 0)
        self.assertLess(math.tan(angle), friction)
        self.assertLess(abs(displacement), 0.01)
        self.assertLess(abs(velocity), 0.05)

    def test_collision_rigid_rigid_frictional_contact_matches_inclined_plane_box_motion(self):
        friction = 0.35
        angle = math.radians(24.0)
        steps = 120
        dt = 1.0 / 360.0
        displacement, velocity, min_gap, contact_count = _run_collision_inclined_plane_rigid_box(
            angle,
            friction,
            steps=steps,
            dt=dt,
        )

        t = steps * dt
        acceleration = 10.0 * (math.sin(angle) - friction * math.cos(angle))
        expected_displacement = 0.5 * acceleration * t * t
        expected_velocity = acceleration * t

        self.assertGreater(contact_count, 0)
        self.assertGreater(acceleration, 0.0)
        self.assertAlmostEqual(displacement, expected_displacement, delta=0.65 * expected_displacement)
        self.assertAlmostEqual(velocity, expected_velocity, delta=0.65 * expected_velocity)
        self.assertGreater(min_gap, -0.03)

    def test_collision_particle_shape_contacts_are_refreshed_in_solver(self):
        model, particle, tray_body, _ = _build_collision_contact_scene()
        solver = SolverAdmmCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="drop",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[particle],
                ),
                SolverCoupled.Entry(
                    name="tray",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[tray_body],
                ),
            ],
            coupling=SolverAdmmCoupled.Config(
                iterations=12,
                rho=45.0,
                gamma=0.05,
                baumgarte=0.1,
                contact_pairs=[
                    SolverAdmmCoupled.ContactPair(
                        source="drop",
                        destination="tray",
                        contact_distance=0.04,
                        detection_margin=0.08,
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        initial_tray_z = float(state_0.body_q.numpy()[tray_body, 2])
        min_gap = float(state_0.particle_q.numpy()[particle, 2] - initial_tray_z)
        for _ in range(90):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)
            state_0, state_1 = state_1, state_0
            particle_z = float(state_0.particle_q.numpy()[particle, 2])
            tray_z = float(state_0.body_q.numpy()[tray_body, 2])
            min_gap = min(min_gap, particle_z - tray_z)

        final_particle_z = float(state_0.particle_q.numpy()[particle, 2])
        final_tray_z = float(state_0.body_q.numpy()[tray_body, 2])
        final_gap = final_particle_z - final_tray_z
        self.assertGreater(solver.collision_contact_count_max, 0)
        self.assertLessEqual(min_gap, 0.08)
        self.assertGreater(min_gap, -0.02)
        self.assertGreater(final_gap, 0.02)
        self.assertLess(final_tray_z, initial_tray_z - 1.0e-3)


if __name__ == "__main__":
    unittest.main()
