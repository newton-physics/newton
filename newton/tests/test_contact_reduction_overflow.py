# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reject contact streams that lost candidates inside global reduction."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_global import (
    GlobalContactReducer,
    GlobalContactReducerData,
    export_contact_to_buffer,
)


@wp.kernel
def _fill_reducer(data: GlobalContactReducerData, allocated: wp.array[int]):
    tid = wp.tid()
    allocated[tid] = export_contact_to_buffer(0, 1, wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0), -0.01, tid, data)


class TestContactReductionOverflow(unittest.TestCase):
    def test_buffer_exhaustion_is_reported_and_cleared(self):
        """Count failed buffer reservations even after the allocation counter rolls back."""
        device = wp.get_device()
        reducer = GlobalContactReducer(1, device=device)
        allocated = wp.zeros(2, dtype=wp.int32, device=device)
        wp.launch(_fill_reducer, dim=2, inputs=[reducer.get_data_struct(), allocated], device=device)
        self.assertEqual(np.count_nonzero(allocated.numpy() < 0), 1)
        self.assertEqual(int(reducer.contact_count.numpy()[0]), 1)
        self.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 1)
        reducer.clear_active()
        self.assertEqual(int(reducer.buffer_overflows.numpy()[0]), 0)

    def test_reducer_failures_reach_solver_status(self):
        """Latch reduction losses without watermark mode and clear producer status on reuse."""
        device = wp.get_device()
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        pipeline = newton.CollisionPipeline(model)
        solver = newton.solvers.SolverFeatherPGS(model)
        state, output, contacts = model.state(), model.state(), pipeline.contacts()
        reducer = GlobalContactReducer(1, device=device)
        for field in ("ht_insert_failures", "buffer_overflows"):
            with self.subTest(field=field):
                reducer.clear()
                getattr(reducer, field).fill_(1)
                # Inject a producer failure to isolate the status handoff from
                # shape-pair selection; buffer exhaustion is exercised above.
                with patch.object(pipeline.narrow_phase, "global_contact_reducer", reducer):
                    pipeline.collide(state, contacts)
                solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
                with self.assertRaisesRegex(RuntimeError, "capacity"):
                    solver.check_constraint_capacity()
                solver.reset(state)
                pipeline.collide(state, contacts)
                solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
                solver.check_constraint_capacity()


if __name__ == "__main__":
    unittest.main()
