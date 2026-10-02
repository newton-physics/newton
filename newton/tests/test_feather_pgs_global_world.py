# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Global (world ``-1``) bodies in multi-world SolverFeatherPGS models: contacts and status."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

_DT = 0.01


def _floor_model(device, floor: str, box_x=(-2.0, 2.0), box_z=0.999, kinematic_world0=False):
    """Two worlds with one unit box each on a floor box whose top is at z = 0.5.

    ``floor`` is ``"static"`` (world geometry), ``"kinematic"`` (a global kinematic body)
    or ``"dynamic"`` (a global dynamic body). ``kinematic_world0`` makes world 0's box kinematic.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for x in box_x:
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))
        kinematic = kinematic_world0 and builder.world_count == 0
        body = world.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, box_z), wp.quat_identity()), mass=1.0, is_kinematic=kinematic
        )
        world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
        builder.add_world(world)
    if floor == "static":
        floor_body = -1
    else:
        floor_body = builder.add_body(is_kinematic=floor == "kinematic", mass=1.0)
    builder.add_shape_box(floor_body, hx=5.0, hy=5.0, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    return builder.finalize(device=device)


def _step_once(model, solver, floor_velocity=0.0, box_velocity=-1.0):
    """Step once with the boxes moving down and the floor moving up; return every body's z velocity."""
    state, output = model.state(), model.state()
    joint_qd = state.joint_qd.numpy()
    for world in range(2):
        joint_qd[6 * world + 2] = box_velocity
    if joint_qd.size > 12:
        joint_qd[14] = floor_velocity
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver.step(state, output, model.control(), contacts, _DT)
    return output.body_qd.numpy()[:, 2], contacts


def test_global_kinematic_floor_supports_every_world(test, device, pgs_mode="matrix_free"):
    """A global kinematic floor stops the boxes of every world exactly like world geometry does."""
    static_model = _floor_model(device, "static")
    static_v = _step_once(static_model, SolverFeatherPGS(static_model, pgs_mode=pgs_mode))[0][:2]
    test.assertLess(np.max(np.abs(static_v)), 0.05)

    model = _floor_model(device, "kinematic")
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, warn_constraint_overflow=False)
    v, contacts = _step_once(model, solver)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(solver.contact_path.numpy()[:count] >= 0, True)
    np.testing.assert_allclose(v[:2], static_v, atol=1.0e-5)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False, False])

    # The prescribed floor velocity enters every world's contact target.
    moving_v = _step_once(model, SolverFeatherPGS(model, pgs_mode=pgs_mode), floor_velocity=0.5)[0][:2]
    np.testing.assert_allclose(moving_v, static_v + 0.5, atol=1.0e-4)

    # The floor is elided from the response even when world 0, where it is stored, has no dynamic body.
    model = _floor_model(device, "kinematic", kinematic_world0=True)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, warn_constraint_overflow=False)
    state, output = model.state(), model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[8] = -1.0
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver.step(state, output, model.control(), contacts, _DT)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    np.testing.assert_allclose(output.body_qd.numpy()[1, 2], static_v[1], atol=1.0e-5)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False, False])


def test_global_kinematic_floor_capture_replay(test, device, pgs_mode="matrix_free"):
    """Captured collide + step matches eager stepping for both worlds on a global kinematic floor."""
    model = _floor_model(device, "kinematic")
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()

    def run(solver, steps, capture):
        state_0, state_1 = model.state(), model.state()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[[2, 8]] = -1.0
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        def substep():
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, _DT)
            wp.copy(state_0.joint_q, state_1.joint_q)
            wp.copy(state_0.joint_qd, state_1.joint_qd)
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)

        if capture:
            substep()
            with wp.ScopedCapture(device=device) as graph:
                substep()
            for _ in range(steps - 1):
                wp.capture_launch(graph.graph)
        else:
            for _ in range(steps):
                substep()
        return state_0.body_q.numpy()[:2], state_0.body_qd.numpy()[:2]

    eager_q, eager_qd = run(SolverFeatherPGS(model, pgs_mode=pgs_mode), 20, capture=False)
    captured_q, captured_qd = run(SolverFeatherPGS(model, pgs_mode=pgs_mode), 20, capture=True)
    np.testing.assert_allclose(captured_q, eager_q, atol=1.0e-6)
    np.testing.assert_allclose(captured_qd, eager_qd, atol=1.0e-5)
    # Both worlds rest on the floor instead of falling through it.
    test.assertGreater(float(np.min(eager_q[:, 2])), 0.9)
    np.testing.assert_allclose(eager_q[0, 2], eager_q[1, 2], atol=1.0e-5)


def _jointed_kinematic_floor_model(device):
    """Two worlds with one box each on a global kinematic floor driven by a prismatic joint."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for x in (-2.0, 2.0):
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = world.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.999), wp.quat_identity()), mass=1.0)
        world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        builder.add_world(world)
    floor = builder.add_link(mass=1.0, is_kinematic=True)
    builder.add_articulation([builder.add_joint_prismatic(-1, floor, axis=newton.Axis.Z)])
    builder.add_shape_box(floor, hx=5.0, hy=5.0, hz=0.5)
    return builder.finalize(device=device)


def _construct(test, model, expect_warning, **kwargs):
    """Construct the solver and check whether it warns about unsolvable global contacts."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverFeatherPGS(model, **kwargs)
    messages = [str(w.message) for w in caught if "cannot be solved" in str(w.message)]
    test.assertEqual(len(messages), 1 if expect_warning else 0, msg=messages)
    if expect_warning:
        test.assertIn("global (world -1) articulations", messages[0])
    return solver


def test_jointed_global_kinematic_flags_other_world_contacts(test, device):
    """A kinematic global articulation with joints is solved in world 0; its other-world contacts are flagged."""
    model = _jointed_kinematic_floor_model(device)
    solver = _construct(test, model, True, warn_constraint_overflow=False)
    state, output = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver.step(state, output, model.control(), contacts, _DT)

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    shape_body = model.shape_body.numpy()
    body_world = model.body_world.numpy()
    pairs = contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]
    box_world = np.maximum(body_world[shape_body[pairs[0]]], body_world[shape_body[pairs[1]]])
    path = solver.contact_path.numpy()[:count]
    test.assertTrue(np.any(box_world == 0) and np.any(box_world == 1))
    np.testing.assert_array_equal(path[box_world == 0] >= 0, True)
    np.testing.assert_array_equal(path[box_world == 1], -1)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, True, True])
    with test.assertRaisesRegex(RuntimeError, "cannot be solved per world"):
        solver.check_constraint_capacity()


def test_construction_warns_about_unsolvable_global_contacts(test, device):
    """The constructor warns once when shapes allow contacts that couple a global articulation with another world."""
    # Dynamic global bodies and jointed kinematic global articulations warn.
    _construct(test, _floor_model(device, "dynamic"), True)
    _construct(test, _jointed_kinematic_floor_model(device), True)

    # World geometry and global kinematic free bodies are solvable in every world.
    _construct(test, _floor_model(device, "static"), False)
    _construct(test, _floor_model(device, "kinematic"), False)
    _construct(test, _floor_model(device, "kinematic", kinematic_world0=True), False)

    # A dynamic global body with a single world, or whose shapes cannot collide with the
    # worlds' bodies, cannot produce such contacts.
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_body(mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    _construct(test, builder.finalize(device=device), False)
    model = _floor_model(device, "dynamic")
    flags = model.shape_flags.numpy()
    flags[model.shape_body.numpy() == 2] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    model.shape_flags.assign(flags)
    _construct(test, model, False)
    model = _floor_model(device, "dynamic")
    groups = model.shape_collision_group.numpy()
    groups[model.shape_body.numpy() == 2] = 7
    model.shape_collision_group.assign(groups)
    _construct(test, model, False)


def _lift_box(model, world):
    """Move one world's box 3 m up, out of contact."""
    joint_q = model.joint_q.numpy()
    joint_q[7 * world + 2] += 3.0
    model.joint_q.assign(joint_q)


def test_dynamic_global_body_flags_other_world_contacts(test, device, pgs_mode="matrix_free"):
    """A dynamic global body interacts with world 0; its contacts with another world are dropped and flagged."""
    # Only world 1's box touches the global dynamic floor.
    model = _floor_model(device, "dynamic")
    _lift_box(model, 0)
    solver = _construct(test, model, True, pgs_mode=pgs_mode, warn_constraint_overflow=False)
    v, contacts = _step_once(model, solver)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(solver.contact_path.numpy()[:count], -1)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, True, True])
    with test.assertRaisesRegex(RuntimeError, r"worlds \[1\].*global"):
        solver.check_constraint_capacity()

    # World 0's box against the same body is solved in world 0.
    model = _floor_model(device, "dynamic")
    _lift_box(model, 1)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, warn_constraint_overflow=False)
    v, contacts = _step_once(model, solver)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    np.testing.assert_array_equal(solver.contact_path.numpy()[:count] >= 0, True)
    np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False, False])
    # The world-0 box is stopped and pushes the (heavier) global body down.
    test.assertGreater(float(v[0]), -0.1)
    test.assertLess(float(v[2]), 0.0)


def _global_overflow_model(device):
    """World 0 and world 1 hold a free box each; a global free box rests on the ground plane.

    With ``global_on_ground`` the global box fills world 0's free-body rows; the world boxes
    float above the ground.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for x in (-3.0, 3.0):
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = world.add_body(xform=wp.transform(wp.vec3(x, 0.0, 2.0), wp.quat_identity()), mass=1.0)
        world.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_world(world)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _place(model, state, z):
    joint_q = model.joint_q.numpy().copy()
    for body, height in enumerate(z):
        joint_q[7 * body + 2] = height
    state.joint_q.assign(joint_q)
    state.joint_qd.zero_()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def test_global_overflow_has_its_own_reset_slot(test, device, pgs_mode="matrix_free"):
    """Global row loss latches the global entry; each reset-mask entry clears only its own status."""
    model = _global_overflow_model(device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, mf_max_constraints=3, warn_constraint_overflow=False)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state, output = model.state(), model.state()
    control = model.control()

    def step(z):
        _place(model, state, z)
        pipeline.collide(state, contacts)
        solver.step(state, output, control, contacts, _DT)
        return solver.constraint_overflow.numpy().tolist()

    def reset(mask):
        solver.reset(state, wp.array(mask, dtype=wp.bool, device=device))
        return solver.constraint_overflow.numpy().tolist()

    # The global box shares world 0's row storage, so its loss flags world 0 and the global entry.
    test.assertEqual(step([2.0, 2.0, 0.099]), [True, False, True])
    test.assertEqual(reset([False, False, True]), [True, False, False])
    test.assertEqual(step([2.0, 2.0, 2.0]), [True, False, False])
    test.assertEqual(reset([True, False, False]), [False, False, False])
    test.assertEqual(step([2.0, 2.0, 0.099]), [True, False, True])
    test.assertEqual(reset([True, False, False]), [False, False, True])
    test.assertEqual(reset([False, False, True]), [False, False, False])

    # A loss in world 1 alone never reaches the global entry or world 0.
    test.assertEqual(step([2.0, 0.099, 2.0]), [False, True, False])
    test.assertEqual(reset([False, False, True]), [False, True, False])
    test.assertEqual(reset([False, True, False]), [False, False, False])

    if not wp.get_device(device).is_cuda:
        return
    # Captured reset + step replays with the mask reassigned between launches.
    mask = wp.array([False, False, True], dtype=wp.bool, device=device)
    _place(model, state, [2.0, 2.0, 0.099])
    pipeline.collide(state, contacts)
    with wp.ScopedCapture(device=device) as capture:
        solver.reset(state, mask)
        solver.step(state, output, control, contacts, _DT)
    solver.reset(state)
    wp.capture_launch(capture.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), [True, False, True])
    mask.assign(np.array([False, False, True]))
    solver.constraint_overflow.assign(np.array([True, True, True]))
    _place(model, state, [2.0, 2.0, 2.0])
    pipeline.collide(state, contacts)
    wp.capture_launch(capture.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), [True, True, False])
    mask.assign(np.array([False, True, False]))
    wp.capture_launch(capture.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), [True, False, False])


def _few_articulation_models(device):
    """Models whose global status entry lies beyond their articulation count.

    Yields ``(name, model, z_contact, z_free)``: one global free box on the ground
    (one world, one articulation), and two worlds holding one articulation between them
    (world 0 has only static geometry, world 1 a box) plus a global box. ``z_contact``
    places the boxes on the ground, where their contacts overflow ``mf_max_constraints=3``;
    ``z_free`` lifts them out of contact.
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    yield "single_global_body", builder.finalize(device=device), [0.099], [2.0]

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    world.add_shape_sphere(-1, xform=wp.transform(wp.vec3(0.0, 5.0, 5.0), wp.quat_identity()), radius=0.1)
    builder.add_world(world)
    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = world.add_body(xform=wp.transform(wp.vec3(3.0, 0.0, 0.099), wp.quat_identity()), mass=1.0)
    world.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_world(world)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    yield "two_worlds_two_articulations", builder.finalize(device=device), [0.099, 0.099], [2.0, 2.0]


def _check_global_slot_reset(test, device, model, z_contact, z_free):
    entries = model.world_count + 1
    solver = SolverFeatherPGS(model, mf_max_constraints=3, warn_constraint_overflow=False)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state, output = model.state(), model.state()
    control = model.control()
    every = [True] * entries
    clear = [False] * entries

    def step(z):
        _place(model, state, z)
        pipeline.collide(state, contacts)
        solver.step(state, output, control, contacts, _DT)
        return solver.constraint_overflow.numpy().tolist()

    def reset(mask):
        solver.reset(state, None if mask is None else wp.array(mask, dtype=wp.bool, device=device))
        return solver.constraint_overflow.numpy().tolist()

    # Actual contact overflow sets every entry: the world rows, and the global entry
    # because the global box shares world 0's row storage.
    test.assertEqual(step(z_contact), every)
    with test.assertRaisesRegex(RuntimeError, "global"):
        solver.check_constraint_capacity()
    test.assertEqual(reset(None), clear)
    solver.check_constraint_capacity()

    global_only = [False] * (entries - 1) + [True]
    worlds_only = [True] * (entries - 1) + [False]
    mixed = [False] * (entries - 2) + [True, True]
    for mask in (global_only, worlds_only, mixed, every):
        test.assertEqual(step(z_contact), every)
        test.assertEqual(reset(mask), [not selected for selected in mask], msg=f"mask {mask}")

    # Stepping again recomputes the status: a lossless step keeps it clear.
    test.assertEqual(step(z_free), clear)

    # Captured resets, unmasked and with the mask reassigned between replays.
    mask = wp.array(global_only, dtype=wp.bool, device=device)
    with wp.ScopedCapture(device=device) as capture_all:
        solver.reset(state)
    with wp.ScopedCapture(device=device) as capture_masked:
        solver.reset(state, mask)
    test.assertEqual(step(z_contact), every)
    wp.capture_launch(capture_all.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), clear)
    test.assertEqual(step(z_contact), every)
    wp.capture_launch(capture_masked.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), worlds_only)
    mask.assign(np.array(worlds_only))
    wp.capture_launch(capture_masked.graph)
    test.assertEqual(solver.constraint_overflow.numpy().tolist(), clear)


def test_global_slot_reset_with_few_articulations(test, device):
    """Every reset mask clears the global entry even when articulations do not outnumber the status entries."""
    for name, model, z_contact, z_free in _few_articulation_models(device):
        with test.subTest(model=name):
            test.assertLessEqual(model.articulation_count, model.world_count)
            _check_global_slot_reset(test, device, model, z_contact, z_free)


class TestFeatherPGSGlobalWorld(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (
    ("test_global_kinematic_floor_supports_every_world", test_global_kinematic_floor_supports_every_world),
    ("test_global_kinematic_floor_capture_replay", test_global_kinematic_floor_capture_replay),
    ("test_dynamic_global_body_flags_other_world_contacts", test_dynamic_global_body_flags_other_world_contacts),
    (
        "test_jointed_global_kinematic_flags_other_world_contacts",
        test_jointed_global_kinematic_flags_other_world_contacts,
    ),
    (
        "test_construction_warns_about_unsolvable_global_contacts",
        test_construction_warns_about_unsolvable_global_contacts,
    ),
    ("test_global_overflow_has_its_own_reset_slot", test_global_overflow_has_its_own_reset_slot),
    ("test_global_slot_reset_with_few_articulations", test_global_slot_reset_with_few_articulations),
):
    add_function_test(TestFeatherPGSGlobalWorld, _name, _func, devices=devices)
    split_devices = devices if _name == "test_global_kinematic_floor_capture_replay" else get_test_devices()
    add_function_test(TestFeatherPGSGlobalWorld, f"{_name}_split", _func, devices=split_devices, pgs_mode="split")


if __name__ == "__main__":
    unittest.main()
