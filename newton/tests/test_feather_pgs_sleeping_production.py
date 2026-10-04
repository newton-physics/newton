# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sleeping of SolverFeatherPGS under a manipulation profile: friction patches, regularization, many iterations."""

import functools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

# A manipulation profile without contact torsion; the torsion variant is in test_feather_pgs_sleeping_torsion.
PROFILE = {
    "pgs_iterations": 32,
    "pgs_contact_regularization": 0.01,
    "friction_anchor_beta": 0.2,
    "contact_friction_gap_threshold": 0.001,
    "dense_max_constraints": 512,
    "mf_max_constraints": 512,
    "enable_sleeping": True,
    "sleep_quiet_time": 0.1,
}


def test_anchored_articulations_sleep_and_wake(test, device):
    """Drop every row of a settled scene and restore the anchored rows of a force-woken articulation."""
    model, pipeline, solver, states, control = _articulations(device)
    _advance(pipeline, solver, states, control, 5)
    awake_rows = int(solver.constraint_count.numpy()[0])
    test.assertGreater(awake_rows, 0)
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
    test.assertEqual(int(solver.constraint_count.numpy()[0]), awake_rows // 2)
    test.assertGreater(states[0].body_q.numpy()[0, 0], frozen[0, 0])
    np.testing.assert_array_equal(states[0].body_q.numpy()[2:], frozen[2:])

    _advance(pipeline, solver, states, control, 600)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))


def test_graph_replay_sleeps_and_wakes(test, device):
    """Sleep inside a captured graph and wake on an explicit wake before the next replay."""
    model, pipeline, solver, states, control = _articulations(device)
    contacts = pipeline.contacts()
    _advance(pipeline, solver, states, control, 2, contacts=contacts)
    with wp.ScopedCapture(device=model.device) as capture:
        for _ in range(2):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
    for _ in range(200):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(solver.constraint_count.numpy(), [0])
    solver.sleeping.wake()
    wp.capture_launch(capture.graph)
    test.assertGreater(int(solver.constraint_count.numpy()[0]), 0)


def test_masked_reset_wakes_only_selected_worlds(test, device):
    """Wake only the worlds a masked reset selects, including none for an empty mask."""
    model, pipeline, solver, states = _two_world_boxes(device)
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    # The final mask entry selects global articulations; these boxes belong to worlds 0 and 1.
    for mask, expected in (
        ([False, False, False], [0, 0]),
        ([False, False, True], [0, 0]),
        ([True, False, False], [1, 0]),
    ):
        solver.reset(states[0], world_mask=wp.array(mask, dtype=wp.bool, device=device))
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), expected)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1, 0])


def test_property_notification_wakes_only_changed_islands(test, device):
    """Wake the island whose mass changed, and nothing for a notification without a change."""
    model, pipeline, solver, states = _two_world_boxes(device)
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


def test_frozen_patch_carry_matches_rebuild(test, device):
    """Carrying frozen friction patches matches rebuilding them through sleep, wake and resettle."""
    _check_carry(test, device, tiles=False)


def test_frozen_patch_carry_matches_rebuild_for_flooded_pairs(test, device):
    """Pairs large enough for the warp flood skip it while frozen and rebuild identically on wake."""
    _check_carry(test, device, tiles=True)


def _check_carry(test, device, tiles):
    trajectories, flood_pairs, histories = [], [], []
    for carry in (False, True):
        model, pipeline, solver, states, control = _articulations(device, tiles=tiles)
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
                # The anchor history a sleeping pair hands to its next awake step.
                current = solver._friction_patches.current
                count = int(contacts.rigid_contact_count.numpy()[0])
                valid = current.valid.numpy()[:count] != 0
                histories.append(
                    (
                        valid,
                        current.displacement.numpy()[:count][valid],
                        current.tangent_impulse.numpy()[:count][valid],
                    )
                )
            trajectory.append(np.concatenate((states[0].body_q.numpy().ravel(), states[0].body_qd.numpy().ravel())))
        test.assertGreater(frozen, 0)
        np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [0, 0])
        trajectories.append(np.array(trajectory))
    if not tiles:
        np.testing.assert_array_equal(histories[1][0], histories[0][0])
        test.assertGreater(int(histories[0][0].sum()), 0)
        test.assertGreater(float(np.abs(histories[0][1]).max()), 0.0)
        for carried, rebuilt in zip(histories[1][1:], histories[0][1:], strict=True):
            np.testing.assert_allclose(carried, rebuilt, rtol=1.0e-5, atol=1.0e-9)
        np.testing.assert_array_equal(trajectories[1][:400], trajectories[0][:400])
        # Roundoff after wake can flip a near-tied patch anchor, so the resettled trajectory gets a physical bound.
        np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=0.0, atol=2.0e-3)
        return
    # Frozen pairs skip the warp flood; bodies built from many shapes are not bitwise reproducible.
    test.assertGreater(flood_pairs[0], 0)
    test.assertEqual(flood_pairs[1], 0)
    positions = [t[-1][: 7 * 4].reshape(4, 7)[:, :3] for t in trajectories]
    np.testing.assert_allclose(positions[1], positions[0], rtol=0.0, atol=2.0e-3)


def test_frozen_patch_carry_copies_the_previous_history(test, device):
    """A sleeping pair's anchor history comes from the previous step, not from the current frame's storage."""
    _model, pipeline, solver, states, control = _articulations(device)
    contacts = pipeline.contacts()
    _advance(pipeline, solver, states, control, 400, contacts=contacts)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertTrue(np.all(solver.sleeping.frozen_bodies.numpy() == 1))
    current = solver._friction_patches.current
    count = int(contacts.rigid_contact_count.numpy()[0])
    names = ("valid", "displacement", "tangent_impulse", "anchor_a", "anchor_b", "owner")
    before = {name: getattr(current, name).numpy()[:count].copy() for name in names}
    test.assertGreater(int(before["valid"].sum()), 0)
    # Overwrite the current frame; the carry must restore every field from the stored history.
    for name in names:
        getattr(current, name).fill_(7)
    _advance(pipeline, solver, states, control, 1, contacts=contacts)
    for name in names:
        np.testing.assert_array_equal(getattr(current, name).numpy()[:count], before[name], err_msg=name)


def test_successive_notifications_keep_every_wake(test, device):
    """A second notification before the next step keeps the first notification's wake."""
    model, pipeline, solver, states = _two_world_boxes(device)
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


def test_replaced_property_array_wakes_its_island(test, device):
    """A model array replaced rather than assigned in place still wakes the island it changed."""
    model, pipeline, solver, states = _two_world_boxes(device)
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    com = model.body_com.numpy()
    com[0, 0] = 0.3
    model.body_com = wp.array(com, dtype=wp.vec3, device=model.device)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1, 0])


def test_resized_property_array_is_rejected(test, device):
    """A model array replaced with a different entity count cannot be diffed and raises."""
    model, pipeline, solver, states = _two_world_boxes(device)
    _advance(pipeline, solver, states, model.control(), 1)
    model.body_mass = wp.zeros(model.body_count + 1, dtype=float, device=model.device)
    with test.assertRaisesRegex(ValueError, "body_mass"):
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)


def test_coincident_entity_counts_keep_property_owners(test, device):
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
    model = builder.finalize(device=device)
    test.assertEqual(model.body_count, model.joint_count)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    solver = newton.solvers.SolverFeatherPGS(model, **PROFILE)
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


def test_solver_body_attribute_notification_wakes_its_island(test, device):
    """A FeatherPGS custom body attribute assigned to the model is diffed like a core body property."""
    box = newton.ModelBuilder()
    body = box.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    newton.solvers.SolverFeatherPGS.register_custom_attributes(builder)
    builder.add_ground_plane()
    builder.replicate(box, 2)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    solver = newton.solvers.SolverFeatherPGS(model, **PROFILE)
    states = [model.state(), model.state()]
    _advance(pipeline, solver, states, model.control(), 400)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    depenetration = model.rigid_body_max_depenetration_velocity.numpy()
    depenetration[0] = 1.0
    model.rigid_body_max_depenetration_velocity.assign(depenetration)
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
    _advance(pipeline, solver, states, model.control(), 1)
    np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1, 0])


def test_mimic_articulation_stays_awake(test, device):
    """Keep an articulation with mimic rows awake while an independent box sleeps."""
    for legacy in (False, True):
        with test.subTest(legacy_constraint_mimic=legacy):
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
                    test,
                    DeprecationWarning,
                    r"add_constraint_mimic\(\) is deprecated",
                    functools.partial(builder.add_constraint_mimic, joint0=joints[2], joint1=joints[1]),
                )
            else:
                builder.set_joint_mimic(joints[2], joints[1])
            box = builder.add_body(xform=wp.transform((2.0, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize(device=device)
            try:
                solver = newton.solvers.SolverFeatherPGS(model, **PROFILE)
            except NotImplementedError:
                test.skipTest("this solver does not support mimic relationships")
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
            _advance(pipeline, solver, [model.state(), model.state()], model.control(), 400)
            awake = solver.sleeping.art_awake.numpy()
            test.assertEqual(int(awake[solver.body_to_articulation.numpy()[base]]), 1)
            test.assertEqual(int(awake[solver.body_to_articulation.numpy()[box]]), 0)


def _articulations(device, tiles=False, profile=None):
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
    model = builder.finalize(device=device)
    # Deterministic contact order, as in production, so A/B trajectories compare bitwise.
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=512, deterministic=True)
    solver = newton.solvers.SolverFeatherPGS(model, **(profile or PROFILE))
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _two_world_boxes(device):
    box = newton.ModelBuilder()
    body = box.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(box, 2)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    solver = newton.solvers.SolverFeatherPGS(model, **PROFILE)
    return model, pipeline, solver, [model.state(), model.state()]


def _advance(pipeline, solver, states, control, steps, *, clear=True, contacts=None):
    contacts = contacts or pipeline.contacts()
    for _ in range(steps):
        if clear:
            states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 0.005)
        states.reverse()


devices = get_cuda_test_devices()


def _expect_one_warning(test, category, pattern, call):
    """Return ``call()``, requiring it to emit exactly one warning, of ``category`` and matching ``pattern``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    test.assertEqual(len(caught), 1, [f"{w.category.__name__}: {w.message}" for w in caught])
    test.assertIs(caught[0].category, category)
    test.assertRegex(str(caught[0].message), pattern)
    return result


class TestFeatherPGSSleepingProduction(unittest.TestCase):
    pass


for _name in (
    "test_anchored_articulations_sleep_and_wake",
    "test_graph_replay_sleeps_and_wakes",
    "test_masked_reset_wakes_only_selected_worlds",
    "test_property_notification_wakes_only_changed_islands",
    "test_frozen_patch_carry_matches_rebuild",
    "test_frozen_patch_carry_matches_rebuild_for_flooded_pairs",
    "test_frozen_patch_carry_copies_the_previous_history",
    "test_successive_notifications_keep_every_wake",
    "test_replaced_property_array_wakes_its_island",
    "test_resized_property_array_is_rejected",
    "test_coincident_entity_counts_keep_property_owners",
    "test_solver_body_attribute_notification_wakes_its_island",
    "test_mimic_articulation_stays_awake",
):
    add_function_test(TestFeatherPGSSleepingProduction, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
