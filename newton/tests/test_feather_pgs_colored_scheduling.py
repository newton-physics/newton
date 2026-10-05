# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact-unit coloring of ``articulated_contact_response="propagation-colored"``.

The colored sweep solves one color at a time, in parallel within a color; units past the color
cap run in an ordered serial tail. A coloring with few units per color, or a spill into the tail,
turns the sweep serial.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PROPAGATION_COLOR_TAIL
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

CONTACT_MAX = 8192


def _heap(device, nx=6, ny=6, nz=3, pitch=0.042, half=0.02, kinematic_tray=False):
    """A dense heap of touching boxes on the ground, optionally on one kinematic tray."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.rigid_gap = 0.002
    builder.add_ground_plane()
    if kinematic_tray:
        # Every box of the bottom layer rests on the tray, a hub of their contact units.
        tray = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(tray, hx=nx * pitch, hy=ny * pitch, hz=0.01)
    z0 = 0.04 if kinematic_tray else 0.021
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3((ix - nx / 2) * pitch, (iy - ny / 2) * pitch, z0 + iz * (2 * half + 0.001)),
                        wp.quat_identity(),
                    )
                )
                builder.add_shape_box(body, hx=half, hy=half, hz=half)
    model = builder.finalize(device=device)
    # Sizes the solver's contact scratch, so it is set before the solver is built.
    model.rigid_contact_max = CONTACT_MAX
    return model


def _colored_solver(model, **kwargs):
    options = {
        "articulated_contact_response": "propagation-colored",
        "pgs_iterations": 6,
        "mf_max_constraints": CONTACT_MAX,
        "dense_max_constraints": 64,
    }
    options.update(kwargs)
    return SolverFeatherPGS(model, **options)


def _settle(model, solver, steps=12):
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=CONTACT_MAX, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return state_0, int(contacts.rigid_contact_count.numpy()[0])


def _color_counts(solver):
    """Units per color of world 0, and the units of the serial tail."""
    offsets = solver.color_world_offsets.numpy()[: PROPAGATION_COLOR_TAIL + 2].astype(np.int64)
    counts = np.diff(offsets)
    return counts[:PROPAGATION_COLOR_TAIL], int(counts[PROPAGATION_COLOR_TAIL])


def _first_fit_reference(solver, world=0):
    """Color world 0's units on the host: sort by contact, first fit, stable counting sort."""
    entries = PROPAGATION_COLOR_TAIL + 2
    stride = solver.propagation_max_constraints
    n = min(int(solver.color_world_unit_cursor.numpy()[world]), stride)
    base = world * stride
    contact = solver.color_unit_contact.numpy()[base : base + n]
    body_a = solver.color_unit_body_a.numpy()[base : base + n]
    body_b = solver.color_unit_body_b.numpy()[base : base + n]
    length = solver.color_unit_len.numpy()[base : base + n]
    order = np.argsort(contact, kind="stable")
    used = {}
    color = np.zeros(n, dtype=np.int64)
    for u in order:
        c = 0
        while c < PROPAGATION_COLOR_TAIL and any(x >= 0 and c in used.get(x, ()) for x in (body_a[u], body_b[u])):
            c += 1
        color[u] = c
        if c < PROPAGATION_COLOR_TAIL:
            for x in (body_a[u], body_b[u]):
                if x >= 0:
                    used.setdefault(x, set()).add(c)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(color, minlength=entries))])[:entries]
    cursor = offsets.copy()
    position_unit = np.zeros(n, dtype=np.int64)
    for u in order:
        position_unit[cursor[color[u]]] = u
        cursor[color[u]] += 1
    row_start = np.concatenate([[0], np.cumsum(length[position_unit])])[:n]
    return n, color, offsets, row_start, contact[position_unit]


def test_dense_heap_colors_without_a_serial_tail(test, device):
    """Color a dense heap into few, full colors with nothing left for the serial tail."""
    model = _heap(device)
    solver = _colored_solver(model)
    state, contact_count = _settle(model, solver)
    test.assertTrue(np.isfinite(state.body_q.numpy()).all())
    test.assertGreater(contact_count, 400)
    counts, tail = _color_counts(solver)
    used = int(np.sum(counts > 0))
    test.assertEqual(tail, 0)
    test.assertLess(used, PROPAGATION_COLOR_TAIL // 2)
    test.assertGreater(counts[counts > 0].mean(), 8.0)


def test_kinematic_hub_does_not_serialize_its_contacts(test, device):
    """Leave a kinematic body out of the conflicts, so a tray under the heap adds few colors."""
    colors = {}
    for tray in (False, True):
        model = _heap(device, kinematic_tray=tray)
        solver = _colored_solver(model)
        state, _ = _settle(model, solver)
        test.assertTrue(np.isfinite(state.body_q.numpy()).all())
        counts, tail = _color_counts(solver)
        test.assertEqual(tail, 0)
        colors[tray] = int(np.sum(counts > 0))
    test.assertLessEqual(colors[True], 1.5 * colors[False] + 8)


def test_kinematic_free_body_is_recorded_as_static(test, device):
    """Record a kinematic free body as -1 in the units, and keep dynamic bodies."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    kinematic = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(kinematic, hx=0.1, hy=0.1, hz=0.1)
    dynamic = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.298), wp.quat_identity()))
    builder.add_shape_box(dynamic, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    model.rigid_contact_max = CONTACT_MAX
    solver = _colored_solver(model, mf_max_constraints=256)
    _settle(model, solver, steps=1)
    n = int(solver.color_world_unit_cursor.numpy()[0])
    bodies = np.concatenate([solver.color_unit_body_a.numpy()[:n], solver.color_unit_body_b.numpy()[:n]])
    test.assertGreater(n, 0)
    test.assertNotIn(kinematic, bodies)
    test.assertIn(dynamic, bodies)


def test_colored_result_matches_the_immediate_response(test, device):
    """Settle the heap like the immediate response."""
    finals = {}
    for response in ("propagation-colored", "immediate"):
        model = _heap(device, nx=4, ny=4, nz=2)
        solver = _colored_solver(model, articulated_contact_response=response, row_watermark=True)
        state, _ = _settle(model, solver, steps=24)
        test.assertEqual(solver.constraint_row_watermarks()["mf_dropped_contact_rows_high_water"], 0)
        finals[response] = state.body_q.numpy()[:, :3]
    test.assertTrue(np.isfinite(finals["immediate"]).all())
    # A different sweep order, so the heights agree only to a tolerance.
    test.assertLess(float(np.max(np.abs(finals["propagation-colored"][:, 2] - finals["immediate"][:, 2]))), 0.01)


def test_coloring_is_deterministic_for_a_fixed_contact_set(test, device):
    """Give the same partition and velocities for the same state and contacts.

    The units are gathered with an atomic cursor, so the coloring sorts them by contact first.
    """
    model = _heap(device)
    state, _ = _settle(model, _colored_solver(model))
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=CONTACT_MAX)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    control = model.control()
    results = []
    for _ in range(3):
        state_in, state_out = model.state(), model.state()
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            wp.copy(getattr(state_in, name), getattr(state, name))
        solver = _colored_solver(model)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        offsets = solver.color_world_offsets.numpy()[: PROPAGATION_COLOR_TAIL + 2].copy()
        order = solver.color_unit_sorted.numpy()[: offsets[-1]].copy()
        results.append((offsets, order, state_out.body_qd.numpy().copy()))
    offsets, order, body_qd = results[0]
    test.assertGreater(int(offsets[-1]), 400)
    for other_offsets, other_order, other_qd in results[1:]:
        np.testing.assert_array_equal(other_offsets, offsets)
        np.testing.assert_array_equal(other_order, order)
        np.testing.assert_array_equal(other_qd, body_qd)


def test_prebuild_matches_the_first_fit_reference(test, device):
    """Reproduce sort-by-contact first-fit exactly, with shared and with global used-color masks."""
    for shape in ((6, 6, 3), (7, 7, 6)):
        with test.subTest(shape=shape):
            model = _heap(device, *shape)
            # The large heap overflows the rows; the coloring covers the units that were allocated.
            solver = _colored_solver(model, warn_constraint_overflow=False)
            state, _ = _settle(model, solver, steps=4)
            test.assertTrue(np.isfinite(state.body_q.numpy()).all())
            n, color, offsets, row_start, sorted_contact = _first_fit_reference(solver)
            test.assertGreater(n, 400)
            np.testing.assert_array_equal(solver.color_unit_color.numpy()[:n], color)
            np.testing.assert_array_equal(solver.color_world_offsets.numpy()[: PROPAGATION_COLOR_TAIL + 2], offsets)
            np.testing.assert_array_equal(solver.color_world_row_order.numpy()[:n], row_start)
            np.testing.assert_array_equal(solver.color_unit_sorted.numpy()[:n], sorted_contact)


def test_colored_accepts_a_large_row_budget(test, device):
    """Keep the per-unit coloring state in global scratch, so the row budget is uncapped."""
    model = _heap(device, nx=2, ny=2, nz=1)
    model.rigid_contact_max = 65536
    solver = _colored_solver(model, mf_max_constraints=32768, dense_max_constraints=2048)
    test.assertEqual(solver.propagation_max_constraints, 32768 + 2048)


class TestFeatherPGSColoredScheduling(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (
    test_dense_heap_colors_without_a_serial_tail,
    test_kinematic_hub_does_not_serialize_its_contacts,
    test_kinematic_free_body_is_recorded_as_static,
    test_colored_result_matches_the_immediate_response,
    test_coloring_is_deterministic_for_a_fixed_contact_set,
    test_prebuild_matches_the_first_fit_reference,
    test_colored_accepts_a_large_row_budget,
):
    add_function_test(TestFeatherPGSColoredScheduling, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main()
