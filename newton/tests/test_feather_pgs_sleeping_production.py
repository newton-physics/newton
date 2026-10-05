# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise sleeping with persistent friction anchors, device torsion and mimic constraints."""

import functools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_TORSION

_PROFILE = {
    "pgs_mode": "matrix_free",
    "pgs_schedule": "interleaved",
    "articulated_contact_response": "immediate",
    "pgs_iterations": 32,
    "pgs_contact_regularization": 0.01,
    "contact_shared_anchor": True,
    "contact_friction_shared_anchor": True,
    "friction_anchor_beta": 0.2,
    "contact_friction_gap_threshold": 0.001,
    "contact_torsion_radius": 0.01,
    "contact_torsion_device": True,
    "dense_max_constraints": 512,
    "mf_max_constraints": 512,
    "enable_sleeping": True,
    "sleep_quiet_time": 0.1,
}


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestSleepingProductionProfile(unittest.TestCase):
    def test_anchored_torsion_articulations_sleep_and_wake(self):
        """Drop every row of a settled scene and restore anchored torsion rows on a force wake."""
        model, pipeline, solver, states, control = _articulations(self)
        _advance(pipeline, solver, states, control, 5)
        awake_rows = int(solver.constraint_count.numpy()[0])
        awake_torsion = _torsion_rows(solver)
        self.assertGreater(awake_torsion, 0)
        _advance(pipeline, solver, states, control, 400)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
        np.testing.assert_array_equal(solver.constraint_count.numpy(), [0])
        np.testing.assert_array_equal(solver.mf_constraint_count.numpy(), [0])
        frozen = states[0].body_q.numpy().copy()
        _advance(pipeline, solver, states, control, 20)
        np.testing.assert_array_equal(states[0].body_q.numpy(), frozen)

        # Pushing the first articulation restores its rows and leaves the second asleep.
        force = np.zeros((model.body_count, 6), dtype=np.float32)
        force[0, 0] = 20.0
        for _ in range(5):
            states[0].body_f.assign(force)
            _advance(pipeline, solver, states, control, 1, clear=False)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1, 0, 0])
        self.assertEqual(int(solver.constraint_count.numpy()[0]), awake_rows // 2)
        self.assertEqual(_torsion_rows(solver), awake_torsion // 2)
        self.assertGreater(states[0].body_q.numpy()[0, 0], frozen[0, 0])
        np.testing.assert_array_equal(states[0].body_q.numpy()[2:], frozen[2:])

        _advance(pipeline, solver, states, control, 600)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
        self.assertFalse(np.any(solver.constraint_overflow.numpy()))

    def test_graph_replay_validates_torsion(self):
        """Sleep and wake inside a captured graph with mandatory torsion validation."""
        model, pipeline, solver, states, control = _articulations(self)
        contacts = pipeline.contacts()
        _advance(pipeline, solver, states, control, 2, contacts=contacts)
        solver.prepare_contact_torsion_capture(states[0], states[1])
        with wp.ScopedCapture(device=model.device) as capture:
            solver.seed_double_buffer_events()
            for _ in range(2):
                states[0].clear_forces()
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 0.005)
                states.reverse()
        for _ in range(200):
            wp.capture_launch(capture.graph)
        solver.validate_contact_torsion()
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
        solver.sleeping.wake()
        wp.capture_launch(capture.graph)
        solver.validate_contact_torsion()
        self.assertGreater(_torsion_rows(solver), 0)

    def test_masked_reset_wakes_only_selected_worlds(self):
        """Wake only the worlds a masked reset selects, including none for an empty mask."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        for mask, expected in (([False, False], [0, 0]), ([True, False], [1, 0])):
            solver.reset(states[0], world_mask=wp.array(mask, dtype=wp.bool, device="cuda:0"))
            np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), expected)
        _advance(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1, 0])

    def test_property_notification_wakes_only_changed_islands(self):
        """Wake the island whose mass changed, and nothing for a notification without a change."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        _advance(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [0, 0])
        mass = model.body_mass.numpy()
        mass[0] *= 2.0
        model.body_mass.assign(mass)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        _advance(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1, 0])

    def test_frozen_patch_carry_matches_rebuild(self):
        """Carrying frozen friction patches matches rebuilding them through sleep, wake and resettle."""
        self._check_carry(tiles=False)

    def test_frozen_patch_carry_matches_rebuild_for_flooded_pairs(self):
        """Pairs large enough for the warp flood skip it while frozen and rebuild identically on wake."""
        self._check_carry(tiles=True)

    def _check_carry(self, tiles):
        trajectories, flood_pairs = [], []
        for carry in (False, True):
            model, pipeline, solver, states, control = _articulations(self, tiles=tiles)
            solver.sleeping.carry_frozen_patches = carry
            contacts = pipeline.contacts()
            frozen = 0
            trajectory = []
            for step in range(900):
                states[0].clear_forces()
                if 400 <= step < 405:
                    force = np.zeros((model.body_count, 6), dtype=np.float32)
                    force[0, 0] = 20.0
                    states[0].body_f.assign(force)
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 0.005)
                states.reverse()
                frozen += int(solver.sleeping.frozen_bodies.numpy().sum())
                if step == 399:
                    flood_pairs.append(int(solver._friction_patches._flood_pair_count.numpy()[0]))
                trajectory.append(np.concatenate((states[0].body_q.numpy().ravel(), states[0].body_qd.numpy().ravel())))
            self.assertGreater(frozen, 0)
            np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [0, 0])
            trajectories.append(np.array(trajectory))
        if not tiles:
            np.testing.assert_array_equal(trajectories[1][:400], trajectories[0][:400])
            # Rebuilding re-derives the carried history each sleeping step, which only differs by roundoff.
            np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=0.0, atol=1.0e-6)
            return
        # Frozen pairs skip the warp flood; bodies built from many shapes are not bitwise reproducible.
        self.assertGreater(flood_pairs[0], 0)
        self.assertEqual(flood_pairs[1], 0)
        positions = [t[-1][: 7 * 4].reshape(4, 7)[:, :3] for t in trajectories]
        np.testing.assert_allclose(positions[1], positions[0], rtol=0.0, atol=2.0e-3)

    def test_successive_notifications_keep_every_wake(self):
        """A second notification before the next step keeps the first notification's wake."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        mass = model.body_mass.numpy()
        mass[0] *= 2.0
        model.body_mass.assign(mass)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        mu = model.shape_material_mu.numpy()
        mu[model.shape_body.numpy() == 1] *= 0.5
        model.shape_material_mu.assign(mu)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        _advance(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1, 1])

    def test_replaced_property_array_wakes_its_island(self):
        """A model array replaced rather than assigned in place still wakes the island it changed."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        com = model.body_com.numpy()
        com[0, 0] = 0.3
        model.body_com = wp.array(com, dtype=wp.vec3, device=model.device)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        _advance(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1, 0])

    def test_resized_property_array_is_rejected(self):
        """A model array replaced with a different entity count cannot be diffed and raises."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance(pipeline, solver, states, model.control(), 1)
        model.body_mass = wp.zeros(model.body_count + 1, dtype=float, device=model.device)
        with self.assertRaisesRegex(ValueError, "body_mass"):
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

    def test_coincident_entity_counts_keep_property_owners(self):
        """A body array is diffed only as a body property when body and joint counts coincide."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        bodies = []
        for x in (0.0, 0.5):
            body = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            bodies.append(body)
        # Reversed joint order gives body_to_articulation [1, 0] against joint_articulation [0, 1].
        for body in reversed(bodies):
            builder.add_articulation([builder.add_joint_free(child=body)])
        model = builder.finalize(device="cuda:0")
        self.assertEqual(model.body_count, model.joint_count)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
        solver = _solver(self, model)
        states = [model.state(), model.state()]
        np.testing.assert_array_equal(solver.body_to_articulation.numpy(), [1, 0])
        _advance(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        mass = model.body_mass.numpy()
        mass[0] *= 2.0
        model.body_mass.assign(mass)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        _advance(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 0])
        _advance(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        _advance(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])

    def test_solver_body_attribute_notification_wakes_its_island(self):
        """A FeatherPGS custom body attribute assigned to the model is diffed like a core body property."""
        box = newton.ModelBuilder()
        body = box.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
        box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder = newton.ModelBuilder()
        newton.solvers.SolverFeatherPGS.register_custom_attributes(builder)
        builder.add_ground_plane()
        builder.replicate(box, 2)
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
        solver = _solver(self, model)
        states = [model.state(), model.state()]
        _advance(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        depenetration = model.rigid_body_max_depenetration_velocity.numpy()
        depenetration[0] = 1.0
        model.rigid_body_max_depenetration_velocity.assign(depenetration)
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
        _advance(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1, 0])

    def test_mimic_articulation_stays_awake(self):
        """Keep an articulation with mimic rows awake while an independent box sleeps."""
        for legacy in (False, True):
            with self.subTest(legacy_constraint_mimic=legacy):
                builder = newton.ModelBuilder()
                builder.add_ground_plane()
                base = builder.add_link(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()))
                builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.05)
                joints = [builder.add_joint_free(child=base)]
                for side in (-1.0, 1.0):
                    finger = builder.add_link(xform=wp.transform((side * 0.15, 0.0, 0.15), wp.quat_identity()))
                    builder.add_shape_box(finger, hx=0.02, hy=0.02, hz=0.05)
                    joints.append(
                        builder.add_joint_revolute(
                            parent=base,
                            child=finger,
                            parent_xform=wp.transform((side * 0.15, 0.0, 0.1), wp.quat_identity()),
                        )
                    )
                builder.add_articulation(joints)
                if legacy:
                    _expect_one_warning(
                        self,
                        DeprecationWarning,
                        r"add_constraint_mimic\(\) is deprecated",
                        functools.partial(builder.add_constraint_mimic, joint0=joints[2], joint1=joints[1]),
                    )
                else:
                    builder.set_joint_mimic(joints[2], joints[1])
                box = builder.add_body(xform=wp.transform((2.0, 0.0, 0.1), wp.quat_identity()))
                builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize(device="cuda:0")
                solver = _solver(self, model, contact_torsion_radius=0.0)
                self.assertGreater(solver._mimic_count, 0)
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
                _advance(pipeline, solver, [model.state(), model.state()], model.control(), 400)
                awake = solver.sleeping.art_awake.numpy()
                self.assertEqual(int(awake[solver.body_to_articulation.numpy()[base]]), 1)
                self.assertEqual(int(awake[solver.body_to_articulation.numpy()[box]]), 0)


def _articulations(test, tiles=False):
    """Two undriven two-link articulations resting on the ground, which route contacts to dense rows.

    With ``tiles``, each base is a grid of small boxes so its ground pair exceeds the warp-flood threshold.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 1.0):
        base = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
        if tiles:
            for i in range(4):
                for j in range(3):
                    offset = wp.transform((-0.15 + 0.1 * i, -0.067 + 0.067 * j, 0.0), wp.quat_identity())
                    builder.add_shape_box(base, xform=offset, hx=0.05, hy=0.033, hz=0.1)
        else:
            builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
        tip = builder.add_link(xform=wp.transform((x + 0.3, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(tip, hx=0.1, hy=0.1, hz=0.1)
        hinge = wp.transform((0.3, 0.0, 0.0), wp.quat_identity())
        builder.add_articulation(
            [
                builder.add_joint_free(child=base),
                builder.add_joint_revolute(parent=base, child=tip, parent_xform=hinge, axis=(0.0, 1.0, 0.0)),
            ]
        )
    model = builder.finalize(device="cuda:0")
    # Deterministic contact order, as in production, so A/B trajectories compare bitwise.
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=512, deterministic=True)
    solver = _solver(test, model)
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _two_world_boxes(test):
    box = newton.ModelBuilder()
    body = box.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(box, 2)
    model = builder.finalize(device="cuda:0")
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    solver = _solver(test, model)
    return model, pipeline, solver, [model.state(), model.state()]


def _solver(test, model, **overrides):
    """Construct the profile solver, which warns that patch friction keeps shared anchors on normal rows only."""
    return _expect_one_warning(
        test,
        UserWarning,
        "contact_shared_anchor still applies to normal rows",
        lambda: newton.solvers.SolverFeatherPGS(model, **{**_PROFILE, **overrides}),
    )


def _expect_one_warning(test, category, pattern, call):
    """Return ``call()``, requiring it to emit exactly one warning, of ``category`` and matching ``pattern``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    test.assertEqual(len(caught), 1, [f"{w.category.__name__}: {w.message}" for w in caught])
    test.assertIs(caught[0].category, category)
    test.assertRegex(str(caught[0].message), pattern)
    return result


def _torsion_rows(solver):
    row_types = solver.row_type.numpy()[0, : int(solver.constraint_count.numpy()[0])]
    return int(np.count_nonzero(row_types == PGS_CONSTRAINT_TYPE_TORSION))


def _advance(pipeline, solver, states, control, steps, *, clear=True, contacts=None):
    contacts = contacts or pipeline.contacts()
    for _ in range(steps):
        if clear:
            states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 0.005)
        states.reverse()


if __name__ == "__main__":
    unittest.main()
