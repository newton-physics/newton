# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Colored propagation contact units of mixed row counts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 1.0 / 240.0
RADIUS = 0.05
ROW_CAPACITY = 6
CONTACT_COUNT = 4


def _mixed_gap_model(device):
    """Four disjoint spheres whose ground contacts take one or three rows."""
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    world = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6, margin=0.0, gap=0.015)
    for index, gap in enumerate((-0.001, 0.004, 0.007, 0.010)):
        body = world.add_link(
            xform=wp.transform(wp.vec3(0.25 * index, 0.0, RADIUS + gap), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001),
            lock_inertia=True,
        )
        world.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        world.add_articulation([world.add_joint_free(parent=-1, child=body)])
    scene.add_world(world)
    scene.add_ground_plane(cfg=cfg)
    return scene.finalize(device=device)


def _solver(model, response):
    """Six propagation rows; friction only on the penetrating contact."""
    return SolverFeatherPGS(
        model,
        articulated_contact_response=response,
        pgs_iterations=8,
        propagation_cached_response=False,
        friction_anchor_beta=0.0,
        dense_max_constraints=3,
        mf_max_constraints=3,
        contact_friction_gap_threshold=0.0,
        contact_speculative_scale=0.0,
    )


def _seed(model, state):
    """Give every sphere the same downward speed and a distinct tangential speed."""
    joint_qd = np.zeros(state.joint_qd.shape, dtype=np.float32)
    dof_starts = model.joint_qd_start.numpy()
    for index in range(CONTACT_COUNT):
        joint_qd[int(dof_starts[index])] = 0.15 + 0.03 * index
        joint_qd[int(dof_starts[index]) + 2] = -0.1
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _unit_impulses(model, solver, contacts):
    """Propagation impulses keyed by the dynamic body of each contact."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    shapes = (contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count])
    slots = solver.contact_slot.numpy()[:count]
    lengths = solver.contact_slots_needed.numpy()[:count]
    rows = solver.propagation_impulses.numpy()[0]
    result = {}
    for contact in np.flatnonzero(solver.contact_path.numpy()[:count] == 2):
        body = max(int(shape_body[shape[contact]]) if shape[contact] >= 0 else -1 for shape in shapes)
        result[body] = rows[slots[contact] : slots[contact] + lengths[contact]].copy()
    return result


def _assert_mixed_topology(test, solver, contacts):
    """Put all four disjoint units of one or three rows into the first color."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(count, CONTACT_COUNT)
    np.testing.assert_array_equal(solver.contact_path.numpy()[:count], 2)
    test.assertEqual(sorted(solver.contact_slots_needed.numpy()[:count].tolist()), [1, 1, 1, 3])
    test.assertEqual(int(solver.propagation_constraint_count.numpy()[0]), ROW_CAPACITY)
    offsets = solver.color_world_offsets.numpy().reshape(solver.world_count, -1)[0]
    counts = np.diff(offsets)
    test.assertEqual(int(offsets[-1]), CONTACT_COUNT)
    test.assertEqual(int(counts[0]), CONTACT_COUNT)
    test.assertFalse(counts[1:].any())
    test.assertEqual(sorted(solver.color_unit_sorted.numpy()[:CONTACT_COUNT].tolist()), list(range(CONTACT_COUNT)))


def test_mixed_units_match_serial_propagation(test, device):
    """Match the serial sweep's impulses, twists and trajectory with units of mixed row counts."""
    model = _mixed_gap_model(device)
    runs = {}
    for response in ("propagation-colored", "propagation"):
        solver = _solver(model, response)
        state_in, state_out = model.state(), model.state()
        _seed(model, state_in)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        runs[response] = (solver, state_in, state_out, pipeline, pipeline.contacts())
    control = model.control()
    for step in range(4):
        for response, (solver, state_in, state_out, pipeline, contacts) in runs.items():
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            runs[response] = (solver, state_out, state_in, pipeline, contacts)
        colored, colored_state, _, _, colored_contacts = runs["propagation-colored"]
        serial, serial_state, _, _, serial_contacts = runs["propagation"]
        if step == 0:
            _assert_mixed_topology(test, colored, colored_contacts)
        colored_impulses = _unit_impulses(model, colored, colored_contacts)
        serial_impulses = _unit_impulses(model, serial, serial_contacts)
        test.assertEqual(sorted(colored_impulses), sorted(serial_impulses))
        for body, expected in serial_impulses.items():
            np.testing.assert_allclose(colored_impulses[body], expected, rtol=2.0e-5, atol=2.0e-6)
        for name in ("joint_q", "joint_qd"):
            np.testing.assert_allclose(
                getattr(colored_state, name).numpy(), getattr(serial_state, name).numpy(), rtol=2.0e-5, atol=2.0e-6
            )


def test_mixed_units_graph_matches_eager(test, device):
    """Replay a captured colored step with mixed units with the eager result."""
    model = _mixed_gap_model(device)
    finals = []
    for capture in (False, True):
        solver = _solver(model, "propagation-colored")
        state, output = model.state(), model.state()
        _seed(model, state)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        contacts = pipeline.contacts()
        control = model.control()

        def substep(state=state, output=output, solver=solver, pipeline=pipeline, contacts=contacts, control=control):
            pipeline.collide(state, contacts)
            solver.step(state, output, control, contacts, DT)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state, name), getattr(output, name))

        substep()
        if capture:
            with wp.ScopedCapture(device) as scope:
                solver.seed_double_buffer_events()
                substep()
            wp.capture_launch(scope.graph)
            _assert_mixed_topology(test, solver, contacts)
        else:
            substep()
        finals.append((state.joint_qd.numpy(), solver.propagation_body_qd.numpy()))
    for got, want in zip(finals[1], finals[0], strict=True):
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1.0e-6)


class TestFeatherPGSColoredUnitCapacity(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (test_mixed_units_match_serial_propagation, test_mixed_units_graph_matches_eager):
    add_function_test(TestFeatherPGSColoredUnitCapacity, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main()
