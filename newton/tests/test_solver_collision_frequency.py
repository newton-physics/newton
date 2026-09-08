# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverBase, SolverVBD
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices

Frequency = SolverBase.CollisionFrequencyType
Slot = SolverBase.CollisionSlot


class _StubSolver(SolverBase):
    """Minimal owning solver: runs the rigid pass per the resolved slot type."""

    supports_collision_pipeline = True

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.collide_calls = 0

    def step(self, state_in, state_out, control, contacts, dt):
        contacts = self._resolve_step_contacts(contacts)
        if (
            self.collision_pipeline is not None
            and self._resolved_collision_frequency_type(Slot.RIGID) != Frequency.NONE
        ):
            self._run_rigid_collision(state_in, dt)
            self.collide_calls += 1


def _build_model(device):
    builder = newton.ModelBuilder()
    # box half-extent 0.5 at z=0.45 -> penetrates the ground plane, guaranteeing contacts
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.45), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    builder.add_ground_plane()
    builder.color()
    return builder.finalize(device=device)


def test_frequency_toggle_drives_detection(test, device):
    """Verify toggling the rigid slot NONE <-> PRE_INIT controls per-step detection.

    The solver holds no cross-step counters; every-N detection is expressed by
    the caller changing the setting between steps.
    """
    model = _build_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    solver = _StubSolver(model, collision_pipeline=pipeline)
    state_a, state_b = model.state(), model.state()

    test.assertIsNotNone(solver.contacts)

    for i in range(6):
        solver.set_collision_frequency(
            collision_frequency_type={Slot.RIGID: Frequency.PRE_INIT if i % 3 == 0 else Frequency.NONE}
        )
        solver.step(state_a, state_b, None, None, 1e-3)
    test.assertEqual(solver.collide_calls, 2)

    # AUTO resolves to PRE_INIT for the rigid slot of an owning solver.
    solver.set_collision_frequency(collision_frequency_type={Slot.RIGID: Frequency.AUTO})
    solver.step(state_a, state_b, None, None, 1e-3)
    test.assertEqual(solver.collide_calls, 3)
    test.assertGreater(int(solver.contacts.rigid_contact_count.numpy()[0]), 0)


def test_owned_speculative_pipeline_receives_dt(test, device):
    """Forward the step time horizon to an owned speculative pipeline."""
    model = _build_model(device)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        speculative_config=newton.CollisionPipeline.SpeculativeContactConfig(),
    )
    solver = _StubSolver(model, collision_pipeline=pipeline)

    solver.step(model.state(), model.state(), None, None, 1e-3)

    test.assertEqual(solver.collide_calls, 1)


def test_frequency_validation_and_ownership(test, device):
    """Verify constructor/setter validation and pipeline-ownership error paths."""
    model = _build_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    test.assertIsNotNone(SolverVBD.__doc__)

    class _NonOwning(SolverBase):
        def step(self, state_in, state_out, control, contacts, dt):
            pass

    class _LegacyContactsSolver(_NonOwning):
        def __init__(self, model, contacts):
            self.contacts = contacts
            super().__init__(model)

    # Pipeline passed to a solver that does not opt in.
    with test.assertRaises(ValueError):
        _NonOwning(model, collision_pipeline=pipeline)

    other_model = _build_model(device)
    other_pipeline = newton.CollisionPipeline(other_model, broad_phase="nxn")
    with test.assertRaisesRegex(ValueError, "same model"):
        _StubSolver(model, collision_pipeline=other_pipeline)

    with test.assertRaisesRegex(ValueError, "requires contact matching"):
        SolverVBD(model, collision_pipeline=pipeline, rigid_contact_history=True, rigid_compliant_alm=False)

    # Non-owning solver: contacts property is None, external contacts pass through.
    plain = _NonOwning(model)
    test.assertIsNone(plain.contacts)
    external = pipeline.contacts()
    test.assertIs(plain._resolve_step_contacts(external), external)

    # A public SolverBase subclass may have used this attribute before the
    # ownership API introduced the property on the base class.
    legacy = _LegacyContactsSolver(model, external)
    test.assertIs(legacy.contacts, external)

    # Active rigid slot without a pipeline.
    with test.assertRaises(ValueError):
        _NonOwning(model, collision_frequency_type={Slot.RIGID: Frequency.PRE_INIT})

    # VBD supports rigid detection on both sides of solver initialization.
    SolverVBD(
        model,
        iterations=1,
        collision_pipeline=pipeline,
        rigid_contact_history=False,
        rigid_compliant_alm=False,
        collision_frequency_type={Slot.RIGID: Frequency.PRE_POST_INIT},
    )
    with test.assertRaisesRegex(ValueError, "requires contact matching"):
        _StubSolver(
            model,
            collision_pipeline=pipeline,
            collision_frequency_type={Slot.RIGID: Frequency.ITERATIONS},
        )

    # Slot and range validation.
    with test.assertRaises(ValueError):
        _StubSolver(model, collision_pipeline=pipeline, collision_frequency={Slot.RIGID: 0})
    with test.assertRaises(ValueError):
        _StubSolver(model, collision_pipeline=pipeline, collision_frequency={99: 1})
    with test.assertRaises(ValueError):
        _StubSolver(model, collision_pipeline=pipeline, collision_frequency_type={99: Frequency.PRE_INIT})

    solver = _StubSolver(
        model,
        collision_pipeline=pipeline,
        collision_frequency={Slot.SOFT_SELF_CONTACT: 2},
    )
    # Partial update keeps the other setting.
    solver.set_collision_frequency(collision_frequency_type={Slot.SOFT_SELF_CONTACT: Frequency.ITERATIONS})
    test.assertEqual(solver.collision_frequency, {Slot.RIGID: 1, Slot.SOFT_SELF_CONTACT: 2})
    test.assertEqual(
        solver.collision_frequency_type,
        {Slot.RIGID: Frequency.AUTO, Slot.SOFT_SELF_CONTACT: Frequency.ITERATIONS},
    )

    # With an owned pipeline, step() must receive contacts=None.
    with test.assertRaises(ValueError):
        solver.step(model.state(), model.state(), None, pipeline.contacts(), 1e-3)


def test_vbd_rigid_none_refreshes_external_contacts(test, device):
    """Refresh externally populated owned contacts while rigid detection is disabled."""
    model = _build_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    solver = SolverVBD(
        model,
        iterations=1,
        collision_pipeline=pipeline,
        rigid_contact_history=False,
        rigid_compliant_alm=False,
        collision_frequency_type={Slot.RIGID: Frequency.NONE, Slot.SOFT_SELF_CONTACT: Frequency.NONE},
    )
    state_a, state_b = model.state(), model.state()

    pipeline.collide(state_a, solver.contacts)
    solver.step(state_a, state_b, None, None, 1e-3)
    test.assertGreater(int(solver.body_body_contact_counts.numpy().sum()), 0)

    solver.contacts.clear()
    solver.step(state_b, state_a, None, None, 1e-3)
    test.assertEqual(int(solver.body_body_contact_counts.numpy().sum()), 0)


def test_vbd_self_contact_none_starts_empty(test, device):
    """Keep first-step self-contact inactive when its schedule is NONE."""
    model = _build_cloth_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    solver = SolverVBD(
        model,
        iterations=1,
        collision_pipeline=pipeline,
        particle_enable_self_contact=True,
        collision_frequency_type={Slot.RIGID: Frequency.NONE, Slot.SOFT_SELF_CONTACT: Frequency.NONE},
    )
    data = solver.contacts.soft_self_contact_data
    test.assertEqual(int(data.vertex_colliding_triangles_count.numpy().sum()), 0)
    test.assertEqual(int(data.edge_colliding_edges_count.numpy().sum()), 0)

    state_a, state_b = model.state(), model.state()
    solver.step(state_a, state_b, None, None, 1e-3)
    test.assertTrue(np.isfinite(state_b.particle_q.numpy()).all())


def test_vbd_rigid_iterations_mode(test, device):
    """Verify rigid ITERATIONS: k > iterations matches PRE_INIT; k = 1 re-detects mid-solve.

    With k larger than the iteration count only the pre-init baseline pass
    fires, so results must match PRE_INIT up to contact-ordering noise; with
    k = 1 the mid-solve re-detection path runs every iteration and must stay
    finite.
    """

    def run(mode, freq):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.45), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        builder.add_ground_plane()
        builder.color()
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        solver = SolverVBD(
            model,
            iterations=3,
            collision_pipeline=pipeline,
            rigid_compliant_alm=False,
            collision_frequency={Slot.RIGID: freq},
            collision_frequency_type={Slot.RIGID: mode, Slot.SOFT_SELF_CONTACT: Frequency.NONE},
        )
        s0, s1 = model.state(), model.state()
        for _ in range(3):
            solver.step(s0, s1, None, None, 1e-3)
            s0, s1 = s1, s0
        return s0.body_q.numpy()

    q_pre = run(Frequency.PRE_INIT, 1)
    q_hi = run(Frequency.ITERATIONS, 10)
    assert_np_equal(q_hi, q_pre, tol=1e-6)
    q_k1 = run(Frequency.ITERATIONS, 1)
    test.assertTrue(np.isfinite(q_k1).all())


def test_vbd_rigid_pre_post_init_mode(test, device):
    """Run rigid detection before and after solver initialization."""

    class _TrackingSolver(SolverVBD):
        def __init__(self, *args, **kwargs):
            self.collision_passes = 0
            super().__init__(*args, **kwargs)

        def _run_rigid_collision(self, state, dt=None):
            self.collision_passes += 1
            super()._run_rigid_collision(state, dt)

    model = _build_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    solver = _TrackingSolver(
        model,
        iterations=1,
        collision_pipeline=pipeline,
        rigid_contact_history=False,
        rigid_compliant_alm=False,
        collision_frequency_type={Slot.RIGID: Frequency.PRE_POST_INIT},
    )

    solver.step(model.state(), model.state(), None, None, 1e-3)

    test.assertEqual(solver.collision_passes, 2)


def test_vbd_iteration_schedules_align(test, device):
    """Run rigid and self-contact passes before the same k-th iterations."""

    class _TrackingSolver(SolverVBD):
        def __init__(self, *args, **kwargs):
            self.rigid_collision_passes = 0
            self.self_collision_passes = 0
            super().__init__(*args, **kwargs)

        def _run_rigid_collision(self, state, dt=None):
            self.rigid_collision_passes += 1
            super()._run_rigid_collision(state, dt)

        def _collision_detection_penetration_free(self, current_state):
            self.self_collision_passes += 1
            super()._collision_detection_penetration_free(current_state)

    model = _build_cloth_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
    solver = _TrackingSolver(
        model,
        iterations=4,
        collision_pipeline=pipeline,
        particle_enable_self_contact=True,
        collision_frequency={Slot.RIGID: 2, Slot.SOFT_SELF_CONTACT: 2},
        collision_frequency_type={Slot.RIGID: Frequency.ITERATIONS, Slot.SOFT_SELF_CONTACT: Frequency.ITERATIONS},
    )

    solver.step(model.state(), model.state(), None, None, 1e-3)

    # One pre-initialization baseline plus passes before iterations 2 and 4.
    test.assertEqual(solver.rigid_collision_passes, 3)
    test.assertEqual(solver.self_collision_passes, 3)


def test_vbd_rigid_iterations_preserves_contact_duals(test, device):
    """Carry in-flight rigid contact duals across scheduled re-detection."""

    class _TrackingSolver(SolverVBD):
        def __init__(self, *args, **kwargs):
            self.dual_norms = []
            super().__init__(*args, **kwargs)

        def _refresh_rigid_contact_state(self, contacts, refresh, *, restore_history=False):
            before = float(np.abs(self.body_body_contact_lambda.numpy()).max()) if restore_history else 0.0
            super()._refresh_rigid_contact_state(contacts, refresh, restore_history=restore_history)
            if restore_history:
                after = float(np.abs(self.body_body_contact_lambda.numpy()).max())
                self.dual_norms.append((before, after))

    model = _build_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
    solver = _TrackingSolver(
        model,
        iterations=3,
        collision_pipeline=pipeline,
        rigid_contact_history=False,
        rigid_compliant_alm=False,
        collision_frequency={Slot.RIGID: 1},
        collision_frequency_type={Slot.RIGID: Frequency.ITERATIONS},
    )

    solver.step(model.state(), model.state(), None, None, 1e-3)

    test.assertEqual(len(solver.dual_norms), 3)
    # The first k=1 refresh precedes the first solve, so no dual exists yet.
    test.assertEqual(solver.dual_norms[0], (0.0, 0.0))
    for before, after in solver.dual_norms[1:]:
        test.assertGreater(before, 0.0)
        test.assertGreater(after, 0.0)


def test_vbd_rigid_iterations_refreshes_contact_frame(test, device):
    """Recompute rigid contact-frame state after each mid-solve detection pass."""

    class _TrackingSolver(SolverVBD):
        def __init__(self, *args, **kwargs):
            self.contact_frame_steps = []
            super().__init__(*args, **kwargs)

        def _step_body_body_contact_frame(self, contacts, body_q, dt, lambda_retention, penalty_decay):
            self.contact_frame_steps.append((lambda_retention, penalty_decay))
            super()._step_body_body_contact_frame(contacts, body_q, dt, lambda_retention, penalty_decay)

    model = _build_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
    solver = _TrackingSolver(
        model,
        iterations=3,
        collision_pipeline=pipeline,
        rigid_compliant_alm=True,
        collision_frequency={Slot.RIGID: 1},
        collision_frequency_type={Slot.RIGID: Frequency.ITERATIONS},
    )

    solver.step(model.state(), model.state(), None, None, 1e-3)

    test.assertEqual(len(solver.contact_frame_steps), 4)
    test.assertEqual(solver.contact_frame_steps[1:], [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)])


def test_vbd_rigid_iterations_refreshes_body_particle_contacts(test, device):
    """Refresh body-particle contact state after each mid-solve collision pass."""

    class _TrackingSolver(SolverVBD):
        def __init__(self, *args, **kwargs):
            self.body_particle_refreshes = 0
            super().__init__(*args, **kwargs)

        def _refresh_body_particle_contact_state(self, contacts, refresh):
            self.body_particle_refreshes += 1
            super()._refresh_body_particle_contact_state(contacts, refresh)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(xform=wp.transform_identity())
    builder.add_shape_sphere(body, radius=0.5)
    builder.add_particle(pos=(0.55, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.1)
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=0.1, contact_matching="latest")
    solver = _TrackingSolver(
        model,
        iterations=3,
        collision_pipeline=pipeline,
        rigid_contact_history=False,
        rigid_compliant_alm=False,
        collision_frequency={Slot.RIGID: 1},
        collision_frequency_type={Slot.RIGID: Frequency.ITERATIONS, Slot.SOFT_SELF_CONTACT: Frequency.NONE},
    )

    state_a, state_b = model.state(), model.state()
    solver.step(state_a, state_b, None, None, 1e-3)

    test.assertEqual(solver.body_particle_refreshes, 4)
    test.assertGreater(int(solver.contacts.soft_contact_count.numpy()[0]), 0)
    test.assertGreater(int(solver.body_particle_contact_counts.numpy().sum()), 0)
    test.assertTrue(np.isfinite(state_b.particle_q.numpy()).all())


def test_vbd_pipeline_iterations_without_internal_bodies(test, device):
    """Run scheduled pipeline passes for particles colliding with static shapes."""

    class _TrackingSolver(SolverVBD):
        def __init__(self, *args, **kwargs):
            self.collision_passes = 0
            super().__init__(*args, **kwargs)

        def _run_rigid_collision(self, state, dt=None):
            self.collision_passes += 1
            super()._run_rigid_collision(state, dt)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle(pos=(0.0, 0.0, 0.05), vel=(0.0, 0.0, 0.0), mass=0.1, radius=0.1)
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device=device)
    test.assertEqual(model.body_count, 0)

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=0.1, contact_matching="latest")
    solver = _TrackingSolver(
        model,
        iterations=3,
        collision_pipeline=pipeline,
        collision_frequency={Slot.RIGID: 1},
        collision_frequency_type={Slot.RIGID: Frequency.ITERATIONS, Slot.SOFT_SELF_CONTACT: Frequency.NONE},
    )
    state_a, state_b = model.state(), model.state()
    solver.step(state_a, state_b, None, None, 1e-3)

    # One pre-initialization pass plus one pass before each iteration.
    test.assertEqual(solver.collision_passes, 4)
    test.assertGreater(int(solver.contacts.soft_contact_count.numpy()[0]), 0)


def test_vbd_external_rigid_iterate_view(test, device):
    """Use externally integrated body poses for mid-iteration collision detection."""
    model = _build_model(device)
    solver = SolverVBD(model, integrate_with_external_rigid_solver=True)
    state_in, state_out = model.state(), model.state()

    view = solver._rigid_iterate_view(state_in, state_out)

    test.assertIs(view.body_q, state_out.body_q)
    test.assertIs(view.body_qd, state_out.body_qd)


def test_vbd_self_contact_rebinds_owned_buffer(test, device):
    """Rebind pipeline self-contact detection to the solver-owned buffer."""
    model = _build_cloth_model(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    solver = SolverVBD(
        model,
        iterations=1,
        collision_pipeline=pipeline,
        particle_enable_self_contact=True,
        collision_frequency_type={Slot.RIGID: Frequency.NONE},
    )
    state = model.state()
    other_contacts = pipeline.contacts()

    pipeline.collide(state, other_contacts, soft_self_contact=True)
    test.assertIs(pipeline._soft_self_contact_detector.collision_info, other_contacts.soft_self_contact_data)

    solver._collision_detection_penetration_free(state)

    test.assertIs(pipeline._soft_self_contact_detector.collision_info, solver.contacts.soft_self_contact_data)


def _build_cloth_model(device):
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=8,
        dim_y=8,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.1,
        tri_ke=1e2,
        tri_ka=1e2,
        tri_kd=1e-4,
    )
    builder.color()
    return builder.finalize(device=device)


def test_vbd_pipeline_parity_and_deprecations(test, device):
    """Verify SolverVBD's pipeline path matches the legacy path and old params warn.

    Steps one cloth twice — once with the legacy externally-driven setup and
    once with a solver-owned pipeline under AUTO scheduling — and compares
    particle positions; also asserts the deprecation and conflict paths of the
    legacy self-contact parameters.
    """
    kwargs = {
        "iterations": 2,
        "particle_enable_self_contact": True,
        "particle_self_contact_margin": 0.02,
        "particle_self_contact_gap": 0.02,
    }

    model_a = _build_cloth_model(device)
    solver_a = SolverVBD(model_a, **kwargs)

    model_b = _build_cloth_model(device)
    pipeline_b = newton.CollisionPipeline(model_b, broad_phase="nxn")
    solver_b = SolverVBD(model_b, collision_pipeline=pipeline_b, **kwargs)
    test.assertIsNotNone(solver_b.contacts.soft_self_contact_data)

    def run(model, solver, contacts):
        s0, s1 = model.state(), model.state()
        for _ in range(3):
            solver.step(s0, s1, None, contacts, 1e-3)
            s0, s1 = s1, s0
        return s0.particle_q.numpy()

    q_a = run(model_a, solver_a, None)
    q_b = run(model_b, solver_b, None)
    assert_np_equal(q_b, q_a, tol=1e-6)

    # Deprecated radius selects the legacy interpretation and warns.
    with test.assertWarns(DeprecationWarning):
        SolverVBD(
            _build_cloth_model(device),
            iterations=1,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.02,
            particle_self_contact_margin=0.04,
        )
    # Margin-only calls retain their old query-radius meaning during deprecation.
    with test.assertWarns(DeprecationWarning):
        legacy_margin = SolverVBD(
            _build_cloth_model(device),
            iterations=1,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.4,
        )
    test.assertEqual(legacy_margin.particle_self_contact_margin, 0.2)
    test.assertEqual(legacy_margin.particle_self_contact_gap, 0.2)
    # Invalid ignored legacy geometry still warns but is not rejected.
    with test.assertWarns(DeprecationWarning):
        SolverVBD(_build_model(device), iterations=1, particle_self_contact_margin=0.1)
    with test.assertWarns(DeprecationWarning):
        SolverVBD(
            _build_cloth_model(device),
            iterations=1,
            particle_enable_self_contact=False,
            particle_self_contact_margin=0.1,
        )
    # The same geometry remains invalid when self-contact is active.
    with test.assertRaises(ValueError):
        SolverVBD(
            _build_cloth_model(device),
            iterations=1,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.1,
        )
    # Deprecated interval warns; combining it with an explicit slot raises.
    with test.assertWarns(DeprecationWarning):
        SolverVBD(_build_cloth_model(device), iterations=1, particle_collision_detection_interval=2)
    with test.assertRaises(ValueError):
        SolverVBD(
            _build_cloth_model(device),
            iterations=1,
            particle_collision_detection_interval=2,
            collision_frequency_type={Slot.SOFT_SELF_CONTACT: Frequency.ITERATIONS},
        )
    # gap cannot be combined with the deprecated radius.
    with test.assertRaises(ValueError):
        SolverVBD(
            _build_cloth_model(device),
            iterations=1,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.02,
            particle_self_contact_gap=0.01,
        )


devices = get_test_devices()


class TestSolverCollisionFrequency(unittest.TestCase):
    pass


add_function_test(
    TestSolverCollisionFrequency,
    "test_frequency_toggle_drives_detection",
    test_frequency_toggle_drives_detection,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_frequency_validation_and_ownership",
    test_frequency_validation_and_ownership,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_owned_speculative_pipeline_receives_dt",
    test_owned_speculative_pipeline_receives_dt,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_rigid_none_refreshes_external_contacts",
    test_vbd_rigid_none_refreshes_external_contacts,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_self_contact_none_starts_empty",
    test_vbd_self_contact_none_starts_empty,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_rigid_iterations_mode",
    test_vbd_rigid_iterations_mode,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_rigid_pre_post_init_mode",
    test_vbd_rigid_pre_post_init_mode,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_iteration_schedules_align",
    test_vbd_iteration_schedules_align,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_rigid_iterations_refreshes_body_particle_contacts",
    test_vbd_rigid_iterations_refreshes_body_particle_contacts,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_rigid_iterations_preserves_contact_duals",
    test_vbd_rigid_iterations_preserves_contact_duals,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_rigid_iterations_refreshes_contact_frame",
    test_vbd_rigid_iterations_refreshes_contact_frame,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_pipeline_iterations_without_internal_bodies",
    test_vbd_pipeline_iterations_without_internal_bodies,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_external_rigid_iterate_view",
    test_vbd_external_rigid_iterate_view,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_self_contact_rebinds_owned_buffer",
    test_vbd_self_contact_rebinds_owned_buffer,
    devices=devices,
)
add_function_test(
    TestSolverCollisionFrequency,
    "test_vbd_pipeline_parity_and_deprecations",
    test_vbd_pipeline_parity_and_deprecations,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main()
