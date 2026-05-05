# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for margin (rest offset) and gap (detection threshold) semantics.

Margin is a per-shape surface offset (pairwise additive) that defines the
resting separation between shapes.  Gap is an additional detection distance
on top of margin; contacts are generated when the surface distance is within
``margin + gap``, but the resting position is controlled solely by margin.
"""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointDoFType
from newton._src.solvers.kamino._src.core.math import I_3
from newton._src.solvers.kamino._src.core.shapes import BoxShape, SphereShape
from newton._src.solvers.kamino._src.core.types import mat33f, transformf, vec3f, vec6f
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.geometry.detector import CollisionDetector
from newton._src.solvers.kamino._src.geometry.primitive import CollisionPipelinePrimitive
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Helpers
###


def _make_sphere_pair_builder(
    distance: float = 0.0,
    margin_top: float = 0.0,
    margin_bottom: float = 0.0,
    gap_top: float = 0.0,
    gap_bottom: float = 0.0,
    radius: float = 0.5,
) -> ModelBuilderKamino:
    """Build a model with two spheres stacked along z, separated by *distance*."""
    builder = ModelBuilderKamino(default_world=True)
    bid0 = builder.add_rigid_body(
        name="bottom_sphere",
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(vec3f(0.0, 0.0, -radius - 0.5 * distance), wp.quat_identity()),
    )
    bid1 = builder.add_rigid_body(
        name="top_sphere",
        m_i=1.0,
        i_I_i=mat33f(np.eye(3, dtype=np.float32)),
        q_i_0=transformf(vec3f(0.0, 0.0, radius + 0.5 * distance), wp.quat_identity()),
    )
    builder.add_geometry(body=bid0, name="bottom", shape=SphereShape(radius), margin=margin_bottom, gap=gap_bottom)
    builder.add_geometry(body=bid1, name="top", shape=SphereShape(radius), margin=margin_top, gap=gap_top)
    return builder


def _run_primitive_pipeline(builder: ModelBuilderKamino, device, max_contacts_per_pair: int = 4):
    """Run broadphase + narrowphase and return the contacts container."""
    model = builder.finalize(device)
    data = model.data()
    state = model.state()
    _, world_req = builder.compute_required_contact_capacity(max_contacts_per_pair=max_contacts_per_pair)
    contacts = ContactsKamino(capacity=world_req, device=device)
    contacts.clear()
    pipeline = CollisionPipelinePrimitive(model=model, bvtype="aabb", default_gap=0.0)
    pipeline.collide(data, state, contacts)
    return contacts, model


###
# Tests — Narrowphase
###


class TestMarginGapNarrowphase(unittest.TestCase):
    """Verify that margin and gap are correctly handled in the primitive narrowphase."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        if test_context.verbose:
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        msg.reset_log_level()

    # ------------------------------------------------------------------
    # Margin tests
    # ------------------------------------------------------------------

    def test_01_margin_shifts_gapfunc_touching(self):
        """Two touching spheres with margin: gapfunc.w = 0 - margin = -margin."""
        margin = 0.05
        builder = _make_sphere_pair_builder(distance=0.0, margin_top=margin, margin_bottom=0.0)
        contacts, _model = _run_primitive_pipeline(builder, self.default_device)

        num_active = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(num_active, 1, "Expected exactly 1 contact for touching spheres with margin")

        gapfunc = contacts.gapfunc.numpy()[0]
        expected_gapfunc_w = 0.0 - margin
        np.testing.assert_allclose(
            gapfunc[3],
            expected_gapfunc_w,
            atol=1e-5,
            err_msg="gapfunc.w should be surface_distance - margin for touching spheres",
        )

    def test_02_margin_symmetric(self):
        """Margin is pairwise additive: margin_A + margin_B."""
        margin_a = 0.02
        margin_b = 0.03
        builder = _make_sphere_pair_builder(distance=0.0, margin_top=margin_a, margin_bottom=margin_b)
        contacts, _ = _run_primitive_pipeline(builder, self.default_device)

        num_active = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(num_active, 1)

        gapfunc = contacts.gapfunc.numpy()[0]
        expected_gapfunc_w = 0.0 - (margin_a + margin_b)
        np.testing.assert_allclose(
            gapfunc[3],
            expected_gapfunc_w,
            atol=1e-5,
            err_msg="gapfunc.w should be surface_distance - (margin_A + margin_B)",
        )

    def test_03_margin_penetrating(self):
        """Penetrating spheres with margin: gapfunc.w = -penetration - margin."""
        margin = 0.05
        penetration = 0.01
        builder = _make_sphere_pair_builder(distance=-penetration, margin_top=margin, margin_bottom=0.0)
        contacts, _ = _run_primitive_pipeline(builder, self.default_device)

        num_active = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(num_active, 1)

        gapfunc = contacts.gapfunc.numpy()[0]
        expected_gapfunc_w = -penetration - margin
        np.testing.assert_allclose(
            gapfunc[3],
            expected_gapfunc_w,
            atol=1e-5,
            err_msg="gapfunc.w should be -penetration - margin",
        )

    # ------------------------------------------------------------------
    # Gap tests
    # ------------------------------------------------------------------

    def test_04_gap_detects_contacts_before_touch(self):
        """Two spheres separated by a small distance: gap enables early detection."""
        separation = 0.02
        gap = 0.015
        builder = _make_sphere_pair_builder(distance=separation, gap_top=gap, gap_bottom=gap)
        contacts, _ = _run_primitive_pipeline(builder, self.default_device)

        num_active = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(
            num_active,
            1,
            f"Expected 1 contact: separation={separation} < gap_A+gap_B={2 * gap}",
        )

    def test_05_gap_no_contact_beyond_threshold(self):
        """Two spheres separated beyond the gap threshold: no contacts."""
        separation = 0.04
        gap = 0.015
        builder = _make_sphere_pair_builder(distance=separation, gap_top=gap, gap_bottom=gap)
        contacts, _ = _run_primitive_pipeline(builder, self.default_device)

        num_active = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(
            num_active,
            0,
            f"Expected 0 contacts: separation={separation} > gap_A+gap_B={2 * gap}",
        )

    def test_06_gap_does_not_shift_gapfunc(self):
        """Gap should not affect gapfunc.w — only margin shifts it."""
        gap = 0.02
        builder = _make_sphere_pair_builder(distance=0.0, gap_top=gap, gap_bottom=0.0)
        contacts, _ = _run_primitive_pipeline(builder, self.default_device)

        num_active = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(num_active, 1)

        gapfunc = contacts.gapfunc.numpy()[0]
        np.testing.assert_allclose(
            gapfunc[3],
            0.0,
            atol=1e-5,
            err_msg="gapfunc.w should be 0 for touching spheres with gap but no margin",
        )

    # ------------------------------------------------------------------
    # Combined margin + gap tests
    # ------------------------------------------------------------------

    def test_07_margin_and_gap_combined_threshold(self):
        """Detection threshold = margin + gap; contacts detected within that range."""
        margin = 0.03
        gap = 0.02
        separation = 0.04
        builder = _make_sphere_pair_builder(
            distance=separation,
            margin_top=margin,
            margin_bottom=0.0,
            gap_top=gap,
            gap_bottom=0.0,
        )
        contacts, _ = _run_primitive_pipeline(builder, self.default_device)

        num_active = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(
            num_active,
            1,
            f"Expected 1 contact: separation={separation} < margin+gap={margin + gap}",
        )

    def test_08_margin_and_gap_combined_gapfunc(self):
        """gapfunc.w = surface_distance - margin, regardless of gap."""
        margin = 0.03
        gap = 0.02
        separation = 0.01
        builder = _make_sphere_pair_builder(
            distance=separation,
            margin_top=margin,
            margin_bottom=0.0,
            gap_top=gap,
            gap_bottom=0.0,
        )
        contacts, _ = _run_primitive_pipeline(builder, self.default_device)

        num_active = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(num_active, 1)

        gapfunc = contacts.gapfunc.numpy()[0]
        expected_gapfunc_w = separation - margin
        np.testing.assert_allclose(
            gapfunc[3],
            expected_gapfunc_w,
            atol=1e-5,
            err_msg="gapfunc.w should be surface_distance - margin",
        )

    def test_09_margin_and_gap_beyond_threshold(self):
        """No contacts when separation exceeds margin + gap."""
        margin = 0.03
        gap = 0.02
        separation = 0.06
        builder = _make_sphere_pair_builder(
            distance=separation,
            margin_top=margin,
            margin_bottom=0.0,
            gap_top=gap,
            gap_bottom=0.0,
        )
        contacts, _ = _run_primitive_pipeline(builder, self.default_device)

        num_active = int(contacts.model_active_contacts.numpy()[0])
        self.assertEqual(
            num_active,
            0,
            f"Expected 0 contacts: separation={separation} > margin+gap={margin + gap}",
        )


###
# Helpers — Solver tests
###

GROUND_HALF_H = 0.5
SPHERE_R = 0.25


def _build_sphere_on_ground(
    sphere_z: float,
    margin: float = 0.0,
    gap: float = 0.0,
) -> ModelBuilderKamino:
    """Build a free sphere above a static ground box, following ``build_free_joint_test`` pattern."""
    builder = ModelBuilderKamino(default_world=True)

    bid = builder.add_rigid_body(
        name="sphere",
        m_i=1.0,
        i_I_i=I_3,
        q_i_0=transformf(vec3f(0.0, 0.0, sphere_z), wp.quat_identity()),
        u_i_0=vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    builder.add_joint(
        name="world_to_sphere",
        dof_type=JointDoFType.FREE,
        act_type=JointActuationType.FORCE,
        bid_B=-1,
        bid_F=bid,
        B_r_Bj=vec3f(0.0, 0.0, sphere_z),
        F_r_Fj=vec3f(0.0, 0.0, 0.0),
        X_j=I_3,
    )
    builder.add_geometry(
        name="sphere",
        body=bid,
        shape=SphereShape(SPHERE_R),
        margin=margin,
        gap=gap,
    )
    builder.add_geometry(
        body=-1,
        name="ground",
        shape=BoxShape(2.0, 2.0, GROUND_HALF_H),
        offset=transformf(vec3f(0.0, 0.0, 0.0), wp.quat_identity()),
        margin=margin,
        gap=gap,
    )
    return builder


def _fast_solver_config() -> SolverKaminoImpl.Config:
    """Relaxed solver config suitable for fast unit tests."""
    config = SolverKaminoImpl.Config()
    config.constraints.alpha = 0.1
    config.padmm.primal_tolerance = 1e-3
    config.padmm.dual_tolerance = 1e-3
    config.padmm.compl_tolerance = 1e-3
    config.padmm.max_iterations = 50
    config.padmm.rho_0 = 0.05
    config.padmm.use_acceleration = True
    config.padmm.warmstart_mode = "containers"
    config.padmm.use_graph_conditionals = False
    config.collect_solver_info = False
    config.compute_solution_metrics = False
    return config


def _step_solver(builder: ModelBuilderKamino, device, num_steps: int = 5, dt: float = 0.01):
    """Finalize model, create solver + detector, step, return final z-position of the sphere."""
    model = builder.finalize(device)
    state_p = model.state()
    state_n = model.state()
    control = model.control()

    detector = CollisionDetector(
        model=model,
        config=CollisionDetector.Config(pipeline="primitive", default_gap=0.0),
    )
    contacts = detector.contacts
    solver = SolverKaminoImpl(model=model, contacts=contacts, config=_fast_solver_config())

    for _ in range(num_steps):
        solver.step(
            state_in=state_p,
            state_out=state_n,
            control=control,
            contacts=contacts,
            detector=detector,
            dt=dt,
        )
        wp.synchronize()
        state_p.copy_from(state_n)

    return float(state_n.q_i.numpy()[0][2])


###
# Tests — Solver rest-distance semantics
###


class TestMarginGapSolver(unittest.TestCase):
    """Verify that the solver treats margin as the rest offset and gap as detection-only.

    Uses a single free-joint sphere above a static ground box (same pattern as
    ``build_free_joint_test``).  Only 5 steps at dt=0.005 — enough to observe
    the solver's immediate response without waiting for convergence.
    """

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        if test_context.verbose:
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        msg.reset_log_level()

    def test_10_penetrating_margin_gets_pushed_out(self):
        """Sphere placed inside the margin envelope should be pushed upward."""
        margin = 0.02
        rest_z = GROUND_HALF_H + SPHERE_R + 2.0 * margin
        start_z = rest_z - 0.01
        final_z = _step_solver(
            _build_sphere_on_ground(sphere_z=start_z, margin=margin, gap=0.01),
            self.default_device,
        )
        self.assertGreater(
            final_z,
            start_z,
            f"Sphere penetrating margin should be pushed up; z: {start_z:.5f} -> {final_z:.5f}",
        )

    def test_11_at_rest_with_margin_stays_put(self):
        """Sphere placed at the margin rest position should barely move."""
        margin = 0.02
        rest_z = GROUND_HALF_H + SPHERE_R + 2.0 * margin
        final_z = _step_solver(
            _build_sphere_on_ground(sphere_z=rest_z, margin=margin, gap=0.01),
            self.default_device,
        )
        np.testing.assert_allclose(
            final_z,
            rest_z,
            atol=2e-3,
            err_msg=f"Sphere at margin rest should stay; moved to z={final_z:.5f}",
        )

    def test_12_gap_does_not_change_rest_position(self):
        """Rest position should be the same regardless of gap value."""
        margin = 0.02
        rest_z = GROUND_HALF_H + SPHERE_R + 2.0 * margin
        for gap in (0.005, 0.05):
            final_z = _step_solver(
                _build_sphere_on_ground(sphere_z=rest_z, margin=margin, gap=gap),
                self.default_device,
            )
            np.testing.assert_allclose(
                final_z,
                rest_z,
                atol=2e-3,
                err_msg=f"gap={gap}: sphere at rest should stay at z={rest_z:.5f}, got {final_z:.5f}",
            )

    def test_13_zero_margin_pushes_out_of_surface(self):
        """With zero margin, a penetrating sphere should be pushed toward the geometric surface."""
        start_z = GROUND_HALF_H + SPHERE_R - 0.005
        final_z = _step_solver(
            _build_sphere_on_ground(sphere_z=start_z, margin=0.0, gap=0.01),
            self.default_device,
        )
        self.assertGreater(
            final_z,
            start_z,
            f"Sphere penetrating surface (margin=0) should be pushed up; z: {start_z:.5f} -> {final_z:.5f}",
        )


###
# Test execution
###

if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
