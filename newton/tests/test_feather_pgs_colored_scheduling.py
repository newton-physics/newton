# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contact-unit coloring behind ``articulated_contact_response='propagation-colored'``.

The colored sweep solves one color at a time, in parallel within a color; units that do
not fit the color budget run in an ordered serial tail. A coloring that produces few units
per color, or spills into the tail, turns the sweep serial, which is what these tests guard.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import PROPAGATION_COLOR_TAIL


def _heap(nx=6, ny=6, nz=3, pitch=0.042, half=0.02, kinematic_tray=False):
    """A dense heap of touching boxes on the ground, optionally inside a kinematic tray."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.rigid_gap = 0.002
    builder.add_ground_plane()
    tray = -1
    if kinematic_tray:
        # One kinematic body under the whole heap: every box rests on it, so it is a hub
        # touching every contact unit of the bottom layer.
        tray = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(tray, hx=nx * pitch, hy=ny * pitch, hz=0.01)
    z0 = 0.04 if kinematic_tray else 0.021
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                b = builder.add_body(
                    xform=wp.transform(
                        wp.vec3((ix - nx / 2) * pitch, (iy - ny / 2) * pitch, z0 + iz * (2 * half + 0.001)),
                        wp.quat_identity(),
                    )
                )
                builder.add_shape_box(b, hx=half, hy=half, hz=half)
    return builder.finalize(), tray


def _color_counts(solver):
    """Units per color for world 0; the last live entry is the serial overflow tail."""
    entries = PROPAGATION_COLOR_TAIL + 2
    offsets = solver.color_world_offsets.numpy()[:entries].astype(np.int64)
    counts = np.diff(offsets)
    return counts[:PROPAGATION_COLOR_TAIL], int(counts[PROPAGATION_COLOR_TAIL])


def _settle(model, steps=12, **solver_kwargs):
    model.rigid_contact_max = 8192  # sizes the solver's contact scratch; set before construction
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_contact_response="propagation-colored",
        pgs_iterations=6,
        mf_max_constraints=8192,
        dense_max_constraints=64,
        **solver_kwargs,
    )
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return solver, state_0, int(contacts.rigid_contact_count.numpy()[0])


def _first_fit_reference(solver, world=0):
    """Reference coloring of world 0's units: sort by contact, first-fit, stable counting sort."""
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


@unittest.skipUnless(wp.get_device().is_cuda, "propagation-colored requires CUDA")
class TestFeatherPGSColoredScheduling(unittest.TestCase):
    def test_dense_heap_colors_without_a_serial_tail(self):
        """A dense heap must color into few colors with nothing left for the serial tail.

        Round-based ticket bidding admitted only the units holding the maximum ticket on
        both of their bodies, which on a heap is a handful per round: it exhausted the
        color budget and spilled most units into the ordered serial tail, making the
        sweep serial. First-fit greedy edge coloring is bounded by 2*degree-1 colors.
        """
        model, _ = _heap()
        solver, state, contacts = _settle(model)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        self.assertGreater(contacts, 400, "heap did not generate a dense contact set")

        counts, tail = _color_counts(solver)
        used = int(np.sum(counts > 0))
        self.assertEqual(tail, 0, f"{tail} units spilled into the serial overflow tail")
        self.assertLess(used, PROPAGATION_COLOR_TAIL // 2, f"coloring used {used} colors")
        self.assertGreater(
            counts[counts > 0].mean(), 8.0, "colors hold too few units each; the sweep is effectively serial"
        )

    def test_kinematic_hub_does_not_serialize_its_contacts(self):
        """One kinematic body under the whole heap must not multiply the color count.

        A prescribed body takes no velocity update from a row, so rows touching it cannot
        conflict with each other. Counting it as a conflicting body gives every contact
        against the tray its own color, so the count scales with the number of boxes
        resting on it; exempting it keeps the coloring close to the tray-free heap.
        """
        model_free, _ = _heap()
        solver_free, _, _ = _settle(model_free)
        counts_free, tail_free = _color_counts(solver_free)
        colors_free = int(np.sum(counts_free > 0))

        model_hub, tray = _heap(kinematic_tray=True)
        self.assertGreaterEqual(tray, 0)
        solver_hub, state, _ = _settle(model_hub)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        counts_hub, tail_hub = _color_counts(solver_hub)
        colors_hub = int(np.sum(counts_hub > 0))

        self.assertEqual(tail_free, 0)
        self.assertEqual(tail_hub, 0)
        self.assertLessEqual(
            colors_hub,
            1.5 * colors_free + 8,
            f"a kinematic hub raised the color count from {colors_free} to {colors_hub}",
        )

    def test_colored_result_matches_the_immediate_response(self):
        """The colored schedule must settle the heap like the reference row path."""
        model, _ = _heap(nx=4, ny=4, nz=2)
        _, colored, _ = _settle(model, steps=24)
        model_ref, _ = _heap(nx=4, ny=4, nz=2)
        model_ref.rigid_contact_max = 8192
        solver = newton.solvers.SolverFeatherPGS(
            model_ref, pgs_mode="matrix_free", articulated_contact_response="immediate", pgs_iterations=6
        )
        pipeline = newton.CollisionPipeline(model_ref, rigid_contact_max=8192)
        contacts = pipeline.contacts()
        s0, s1 = model_ref.state(), model_ref.state()
        control = model_ref.control()
        for _ in range(24):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 1.0 / 240.0)
            s0, s1 = s1, s0

        a = colored.body_q.numpy()[:, :3]
        b = s0.body_q.numpy()[:, :3]
        self.assertTrue(np.isfinite(b).all())
        # Different sweep order, so not bitwise equal; the heap must still stand.
        self.assertLess(float(np.max(np.abs(a[:, 2] - b[:, 2]))), 0.01)

    def test_coloring_is_deterministic_for_a_fixed_contact_set(self):
        """Same state and contacts must give the same partition and the same velocities.

        The unit list is gathered with an atomic cursor, so its order differs between
        launches; the coloring must not inherit that order. The pre-build kernel sorts
        units by global contact index before the order-dependent greedy pass.
        """
        model, _ = _heap()
        _, state, _ = _settle(model)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192)
        contacts = pipeline.contacts()
        state.clear_forces()
        pipeline.collide(state, contacts)
        control = model.control()
        results = []
        for _ in range(3):
            state_in, state_out = model.state(), model.state()
            for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                getattr(state_in, name).assign(getattr(state, name))
            trial = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                articulated_contact_response="propagation-colored",
                pgs_iterations=6,
                pgs_warmstart=False,
                mf_max_constraints=8192,
                dense_max_constraints=64,
            )
            trial.step(state_in, state_out, control, contacts, 1.0 / 240.0)
            entries = PROPAGATION_COLOR_TAIL + 2
            offsets = trial.color_world_offsets.numpy()[:entries].copy()
            order = trial.color_unit_sorted.numpy()[: offsets[-1]].copy()
            results.append((offsets, order, state_out.body_qd.numpy().copy()))
        offsets, order, body_qd = results[0]
        self.assertGreater(int(offsets[-1]), 400)
        for other_offsets, other_order, other_qd in results[1:]:
            np.testing.assert_array_equal(other_offsets, offsets)
            np.testing.assert_array_equal(other_order, order)
            np.testing.assert_array_equal(other_qd, body_qd)

    def test_prescribed_mask_marks_only_kinematic_free_bodies(self):
        """Only a kinematic free rigid body has a zero response and may be shared by a color.

        The kinematic root of a multi-body articulation still carries the tree response of
        its joint dofs (the factorization does not read the kinematic flag), so it must
        remain a coloring conflict; the builder only allows kinematic bodies at roots.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        free_kinematic = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), is_kinematic=True
        )
        builder.add_shape_box(free_kinematic, hx=0.1, hy=0.1, hz=0.1)
        free_dynamic = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(free_dynamic, hx=0.1, hy=0.1, hz=0.1)
        # add_link, not add_body: add_body wraps each body in its own free-joint articulation.
        root = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.5), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
        j_root = builder.add_joint_revolute(parent=-1, child=root, axis=wp.vec3(0.0, 1.0, 0.0))
        link = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.8), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        j_link = builder.add_joint_revolute(
            parent=root,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()),
        )
        builder.add_articulation([j_root, j_link], label="kinematic_root_chain")
        model = builder.finalize()
        model.rigid_contact_max = 256
        solver = newton.solvers.SolverFeatherPGS(
            model, pgs_mode="matrix_free", articulated_contact_response="propagation-colored", mf_max_constraints=256
        )
        prescribed = solver._body_prescribed.numpy()
        self.assertEqual(int(prescribed[free_kinematic]), 1)
        self.assertEqual(int(prescribed[free_dynamic]), 0)
        self.assertEqual(int(prescribed[root]), 0)
        self.assertEqual(int(prescribed[link]), 0)

    def test_prebuild_matches_the_first_fit_reference(self):
        """The device coloring reproduces sort-by-contact first-fit exactly.

        The prebuild stages units in shared memory for its serial passes, sorts packed
        keys in shared memory when a world has at most 4096 units, and keeps the
        used-color masks in shared memory when the unit bodies span at most 256
        indices. The large heap spans more bodies than that and takes the global masks.
        """
        for shape in ((6, 6, 3), (7, 7, 6)):
            model, _ = _heap(*shape)
            solver, state, _ = _settle(model, steps=4)
            self.assertTrue(np.isfinite(state.body_q.numpy()).all())
            n, color, offsets, row_start, sorted_contact = _first_fit_reference(solver)
            self.assertGreater(n, 400)
            entries = PROPAGATION_COLOR_TAIL + 2
            np.testing.assert_array_equal(solver.color_unit_color.numpy()[:n], color, err_msg=f"{shape} colors")
            np.testing.assert_array_equal(solver.color_world_offsets.numpy()[:entries], offsets, err_msg=f"{shape}")
            np.testing.assert_array_equal(solver.color_world_row_order.numpy()[:n], row_start, err_msg=f"{shape}")
            np.testing.assert_array_equal(solver.color_unit_sorted.numpy()[:n], sorted_contact, err_msg=f"{shape}")

    def test_colored_accepts_a_large_row_budget(self):
        """The coloring keeps per-unit state in global scratch, so no staging cap applies."""
        model, _ = _heap(nx=2, ny=2, nz=1)
        model.rigid_contact_max = 65536
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation-colored",
            mf_max_constraints=32768,
            dense_max_constraints=2048,
        )
        self.assertEqual(int(solver.propagation_max_constraints), 32768 + 2048)


if __name__ == "__main__":
    unittest.main()
