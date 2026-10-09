# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Report contact candidates dropped inside global contact reduction."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    export_contact_to_buffer,
)
from newton._src.geometry.contact_reduction_hydroelastic import HydroelasticContactReduction
from newton.solvers import SolverSemiImplicit
from newton.solvers.experimental.coupled import SolverCoupled
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def _fill_reducer(data: GlobalContactReducerData, allocated: wp.array[int]):
    tid = wp.tid()
    allocated[tid] = export_contact_to_buffer(0, 1, wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0), -0.01, tid, data)


def _boxes_on_mesh_model(device, height: float = 0.095):
    """Three boxes resting on a 4x4-cell triangle-mesh ground, which uses global reduction."""
    cells = 4
    xs, ys = np.meshgrid(np.linspace(-1.0, 1.0, cells + 1), np.linspace(-1.0, 1.0, cells + 1))
    vertices = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1).astype(np.float32)
    indices = []
    for i in range(cells):
        for j in range(cells):
            a = i * (cells + 1) + j
            indices += [a, a + 1, a + cells + 2, a, a + cells + 2, a + cells + 1]
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(-1, mesh=newton.Mesh(vertices, np.array(indices, dtype=np.int32)))
    for k in range(3):
        body = builder.add_body(xform=wp.transform(wp.vec3(-0.5 + 0.5 * k, 0.0, height), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize(device=device)


def test_buffer_exhaustion_is_reported_and_cleared(test, device):
    """Count failed buffer reservations even after the allocation counter rolls back."""
    reducer = GlobalContactReducer(1, device=device)
    allocated = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(_fill_reducer, dim=2, inputs=[reducer.get_data_struct(), allocated], device=device)
    test.assertEqual(np.count_nonzero(allocated.numpy() < 0), 1)
    # The rollback keeps contact_count within capacity, so the count alone cannot reveal the loss.
    test.assertEqual(int(reducer.contact_count.numpy()[0]), 1)
    test.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 1)
    reducer.clear_active()
    test.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 0)
    reducer.buffer_overflows.fill_(1)
    reducer.clear()
    test.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 0)


def test_pipeline_flags_dropped_reduction_candidates(test, device):
    """Flag a contact stream whose reducer buffer was exhausted, and clear the flag on the next pass."""
    model = _boxes_on_mesh_model(device)
    state = model.state()

    reference = newton.CollisionPipeline(model)
    reference_contacts = reference.contacts()
    reference.collide(state, reference_contacts)
    reference_reducer = reference.narrow_phase.global_contact_reducer
    candidate_count = int(reference_reducer.contact_count.numpy()[0])
    triangle_pair_count = int(reference.narrow_phase.triangle_pairs_count.numpy()[0])
    test.assertEqual(int(reference_contacts._reduction_overflow.numpy()[0]), 0)
    test.assertEqual(int(reference_reducer.buffer_overflows.numpy()[0]), 0)

    # Leave room for every triangle pair but not for every reduction candidate, so the reducer
    # buffer is the only capacity that is exceeded.
    capacity = candidate_count - 2
    test.assertGreaterEqual(capacity, triangle_pair_count)
    pipeline = newton.CollisionPipeline(model, max_triangle_pairs=capacity, verify_buffers=False)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    reducer = pipeline.narrow_phase.global_contact_reducer
    test.assertEqual(int(reducer.contact_count.numpy()[0]), capacity)
    test.assertEqual(int(reducer.ht_insert_failures.numpy()[0]), 0)
    test.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 2)
    test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 1)
    # Which candidates are dropped depends on the order of the device's atomic allocations,
    # and the reduced set may or may not change, so the count is only bounded.
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    test.assertLessEqual(count, int(reference_contacts.rigid_contact_count.numpy()[0]))

    # The flag belongs to the pass that filled the buffer: a pass without losses clears it.
    separated = model.state()
    body_q = separated.body_q.numpy()
    body_q[:, 2] += 1.0
    separated.body_q.assign(body_q)
    pipeline.collide(separated, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 0)

    # Contacts.clear() also resets the flag.
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 1)
    contacts.clear()
    test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 0)


def test_pipeline_flags_hashtable_insert_failures(test, device):
    """Flag a contact stream when the reducer reports failed hashtable inserts."""
    model = _boxes_on_mesh_model(device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, verify_buffers=False)
    contacts = pipeline.contacts()
    reducer = pipeline.narrow_phase.global_contact_reducer

    # Inject a failure count after the narrow phase clears the reducer, before it is recorded.
    original_clear_active = reducer.clear_active

    def clear_active_with_failure():
        original_clear_active()
        reducer.ht_insert_failures.fill_(1)

    reducer.clear_active = clear_active_with_failure
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 1)

    reducer.clear_active = original_clear_active
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 0)


def test_unreduced_hydroelastic_clear_resets_buffer_overflows(test, device):
    """Clear the overflow counter of an unreduced hydroelastic buffer between passes."""
    hydro = HydroelasticContactReduction(1, device=device, enable_reduction=False)
    allocated = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(_fill_reducer, dim=2, inputs=[hydro.get_data_struct(), allocated], device=device)
    test.assertEqual(int(hydro.reducer.buffer_overflows.numpy()[0]), 1)
    hydro.clear()
    test.assertEqual(int(hydro.reducer.buffer_overflows.numpy()[0]), 0)
    test.assertEqual(int(hydro.reducer.contact_count.numpy()[0]), 0)


def test_coupled_entry_contacts_keep_reduction_loss(test, device):
    """Carry reduction loss from the source contacts into a coupled entry's filtered contacts."""
    model = _boxes_on_mesh_model(device)
    state = model.state()
    reference = newton.CollisionPipeline(model)
    reference.collide(state, reference.contacts())
    capacity = int(reference.narrow_phase.global_contact_reducer.contact_count.numpy()[0]) - 2
    pipeline = newton.CollisionPipeline(model, max_triangle_pairs=capacity, verify_buffers=False)
    contacts = pipeline.contacts()
    coupled = SolverCoupled(
        model,
        entries=[SolverCoupled.Entry(name="all", solver=SolverSemiImplicit, bodies=list(range(model.body_count)))],
    )
    separated = model.state()
    body_q = separated.body_q.numpy()
    body_q[:, 2] += 1.0
    separated.body_q.assign(body_q)

    def filtered_flag():
        filtered = coupled.entry_contacts("all", contacts)
        test.assertIsNot(filtered, contacts)
        return int(filtered.rigid_contact_count.numpy()[0]), int(filtered._reduction_overflow.numpy()[0])

    # A lossy pass, a cached reuse of the same pass, a loss-free pass, then a lossy pass again.
    pipeline.collide(state, contacts)
    test.assertEqual(int(contacts._reduction_overflow.numpy()[0]), 1)
    count, flag = filtered_flag()
    test.assertEqual(count, int(contacts.rigid_contact_count.numpy()[0]))
    test.assertGreater(count, 0)
    test.assertEqual(flag, 1)
    test.assertEqual(filtered_flag()[1], 1)
    pipeline.collide(separated, contacts)
    test.assertEqual(filtered_flag(), (0, 0))
    pipeline.collide(state, contacts)
    test.assertEqual(filtered_flag()[1], 1)

    # A different source buffer refreshes the cached entry buffer and its flag.
    other = pipeline.contacts()
    pipeline.collide(separated, other)
    test.assertEqual(int(coupled.entry_contacts("all", other)._reduction_overflow.numpy()[0]), 0)


class TestContactReductionOverflow(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestContactReductionOverflow,
    "test_buffer_exhaustion_is_reported_and_cleared",
    test_buffer_exhaustion_is_reported_and_cleared,
    devices=devices,
)
add_function_test(
    TestContactReductionOverflow,
    "test_pipeline_flags_dropped_reduction_candidates",
    test_pipeline_flags_dropped_reduction_candidates,
    devices=devices,
)
add_function_test(
    TestContactReductionOverflow,
    "test_pipeline_flags_hashtable_insert_failures",
    test_pipeline_flags_hashtable_insert_failures,
    devices=devices,
)
add_function_test(
    TestContactReductionOverflow,
    "test_unreduced_hydroelastic_clear_resets_buffer_overflows",
    test_unreduced_hydroelastic_clear_resets_buffer_overflows,
    devices=devices,
)
add_function_test(
    TestContactReductionOverflow,
    "test_coupled_entry_contacts_keep_reduction_loss",
    test_coupled_entry_contacts_keep_reduction_loss,
    devices=devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
