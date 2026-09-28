# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for mixed propagation-colored contact units."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 240.0
RADIUS = 0.05
ROW_CAPACITY = 6
CONTACT_COUNT = 4
FRICTION_GAP_THRESHOLD = 0.0


def _build_mixed_gap_model(device: str) -> newton.Model:
    """Build four disjoint spheres whose ground contacts use one or three rows."""
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    world = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        mu=0.6,
        margin=0.0,
        gap=0.015,
    )
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


def _make_solver(model: newton.Model, response: str) -> SolverFeatherPGS:
    """Build a serial or colored propagation solver with six contact rows."""
    return SolverFeatherPGS(
        model,
        angular_damping=0.0,
        pgs_mode="matrix_free",
        articulated_contact_response=response,
        pgs_iterations=8,
        pgs_warmstart=False,
        mf_warmstart=False,
        propagation_cached_response=False,
        dense_max_constraints=3,
        mf_max_constraints=3,
        contact_friction_gap_threshold=FRICTION_GAP_THRESHOLD,
        contact_speculative_scale=0.0,
        use_parallel_streams=False,
    )


def _seed_state(model: newton.Model, state: newton.State) -> tuple[np.ndarray, np.ndarray]:
    """Give every sphere the same downward speed and a distinct tangent speed."""
    joint_q = state.joint_q.numpy().copy()
    joint_qd = np.zeros(state.joint_qd.shape, dtype=np.float32)
    dof_starts = model.joint_qd_start.numpy()
    for index in range(CONTACT_COUNT):
        dof = int(dof_starts[index])
        joint_qd[dof] = 0.15 + 0.03 * index
        joint_qd[dof + 2] = -0.1
    state.joint_q.assign(joint_q)
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return joint_q, joint_qd


def _unit_impulses(model: newton.Model, solver: SolverFeatherPGS, contacts: newton.Contacts):
    """Return propagation impulses keyed by the dynamic body of each contact."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    shape_body = model.shape_body.numpy()
    slots = solver.contact_slot.numpy()[:count]
    lengths = solver.contact_slots_needed.numpy()[:count]
    paths = solver.contact_path.numpy()[:count]
    rows = solver.propagation_impulses.numpy()[0]
    result = {}
    for contact in range(count):
        if int(paths[contact]) != 2:
            continue
        body_a = int(shape_body[int(shape0[contact])]) if shape0[contact] >= 0 else -1
        body_b = int(shape_body[int(shape1[contact])]) if shape1[contact] >= 0 else -1
        body = max(body_a, body_b)
        slot = int(slots[contact])
        length = int(lengths[contact])
        result[body] = rows[slot : slot + length].copy()
    return result


def _assert_mixed_topology(test: unittest.TestCase, solver: SolverFeatherPGS, contacts: newton.Contacts) -> None:
    """Keep all four mixed-length units in the first color and preserve their rows."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(count, CONTACT_COUNT)
    paths = solver.contact_path.numpy()[:count]
    np.testing.assert_array_equal(paths, np.full(CONTACT_COUNT, 2, dtype=np.int32))
    lengths = solver.contact_slots_needed.numpy()[:count]
    test.assertEqual(sorted(lengths.tolist()), [1, 1, 1, 3])
    test.assertEqual(int(solver.propagation_constraint_count.numpy()[0]), ROW_CAPACITY)
    test.assertEqual(int(solver.color_world_unit_cursor.numpy()[0]), CONTACT_COUNT)

    offsets = solver.color_world_offsets.numpy().reshape(solver.world_count, -1)[0]
    bucket_counts = np.diff(offsets)
    test.assertEqual(int(offsets[-1]), CONTACT_COUNT)
    # Coloring now stages up to the row capacity, rather than row_capacity // 3.
    # All four disjoint units fit even though three only have a normal row.
    test.assertEqual(int(offsets[-2]), CONTACT_COUNT)
    test.assertEqual(int(bucket_counts[0]), CONTACT_COUNT)
    np.testing.assert_array_equal(bucket_counts[1:-1], np.zeros_like(bucket_counts[1:-1]))
    test.assertEqual(int(bucket_counts[-1]), 0)

    ordered = solver.color_unit_sorted.numpy()[:CONTACT_COUNT]
    test.assertEqual(sorted(ordered.tolist()), list(range(CONTACT_COUNT)))


@unittest.skipUnless(wp.is_cuda_available(), "propagation-colored requires CUDA")
class TestFeatherPGSColoredUnitCapacity(unittest.TestCase):
    def test_mixed_gap_matches_serial_propagation(self):
        """Match impulses, body velocity, and a short trajectory with mixed-length units."""
        device = "cuda:0"
        model = _build_mixed_gap_model(device)
        colored = _make_solver(model, "propagation-colored")
        serial = _make_solver(model, "propagation")
        colored_in, colored_out = model.state(), model.state()
        serial_in, serial_out = model.state(), model.state()
        _seed_state(model, colored_in)
        _seed_state(model, serial_in)
        colored_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        serial_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        colored_contacts = colored_pipeline.contacts()
        serial_contacts = serial_pipeline.contacts()
        control = model.control()

        for step in range(4):
            colored_in.clear_forces()
            serial_in.clear_forces()
            colored_pipeline.collide(colored_in, colored_contacts)
            serial_pipeline.collide(serial_in, serial_contacts)
            colored.step(colored_in, colored_out, control, colored_contacts, DT)
            serial.step(serial_in, serial_out, control, serial_contacts, DT)

            if step == 0:
                _assert_mixed_topology(self, colored, colored_contacts)
            colored_impulses = _unit_impulses(model, colored, colored_contacts)
            serial_impulses = _unit_impulses(model, serial, serial_contacts)
            self.assertEqual(sorted(colored_impulses), sorted(serial_impulses))
            for body, expected in serial_impulses.items():
                np.testing.assert_allclose(colored_impulses[body], expected, rtol=2.0e-5, atol=2.0e-6)
            np.testing.assert_allclose(
                colored.propagation_body_qd.numpy(),
                serial.propagation_body_qd.numpy(),
                rtol=2.0e-5,
                atol=2.0e-6,
            )
            np.testing.assert_allclose(
                colored_out.joint_qd.numpy(), serial_out.joint_qd.numpy(), rtol=2.0e-5, atol=2.0e-6
            )
            np.testing.assert_allclose(
                colored_out.joint_q.numpy(), serial_out.joint_q.numpy(), rtol=2.0e-5, atol=2.0e-6
            )
            colored_in, colored_out = colored_out, colored_in
            serial_in, serial_out = serial_out, serial_in

    def test_mixed_gap_cuda_graph_matches_eager(self):
        """Replay a captured mixed-unit colored step with the same result as eager execution."""
        device = wp.get_device("cuda:0")
        model = _build_mixed_gap_model(str(device))
        graph_solver = _make_solver(model, "propagation-colored")
        eager_solver = _make_solver(model, "propagation-colored")
        graph_state, graph_out = model.state(), model.state()
        eager_state, eager_out = model.state(), model.state()
        initial_q, initial_qd = _seed_state(model, graph_state)
        _seed_state(model, eager_state)
        graph_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        eager_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        graph_contacts = graph_pipeline.contacts()
        eager_contacts = eager_pipeline.contacts()
        graph_control = model.control()
        eager_control = model.control()

        def one_step(solver, pipeline, contacts, state_in, state_out, control):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

        one_step(graph_solver, graph_pipeline, graph_contacts, graph_state, graph_out, graph_control)
        one_step(eager_solver, eager_pipeline, eager_contacts, eager_state, eager_out, eager_control)
        for state in (graph_state, eager_state):
            state.joint_q.assign(initial_q)
            state.joint_qd.assign(initial_qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        with wp.ScopedCapture(device) as capture:
            one_step(graph_solver, graph_pipeline, graph_contacts, graph_state, graph_out, graph_control)
        wp.capture_launch(capture.graph)
        one_step(eager_solver, eager_pipeline, eager_contacts, eager_state, eager_out, eager_control)

        _assert_mixed_topology(self, graph_solver, graph_contacts)
        np.testing.assert_allclose(graph_state.joint_q.numpy(), eager_state.joint_q.numpy(), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(graph_state.joint_qd.numpy(), eager_state.joint_qd.numpy(), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(
            graph_solver.propagation_body_qd.numpy(),
            eager_solver.propagation_body_qd.numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
