# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sleeping islands of SolverFeatherPGS skip their rows without removing collision or wake evidence."""

import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_ant_joint_limits_skip_and_restore(test, device):
    """Skip a sleeping ant's limit rows and restore them on captured force wake."""
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        str(Path(newton.__file__).parent / "examples/assets/nv_ant.xml"),
        ignore_names=["floor", "ground"],
        collapse_fixed_joints=False,
        parse_sites=False,
    )
    builder.joint_target_ke[:] = [0.0] * len(builder.joint_target_ke)
    builder.joint_target_kd[:] = [0.0] * len(builder.joint_target_kd)
    builder.joint_act[:] = [0.0] * len(builder.joint_act)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    for skip in (False, True):
        with test.subTest(skip_constraints=skip):
            solver = newton.solvers.SolverFeatherPGS(
                model,
                enable_sleeping=True,
                sleep_skip_constraints=skip,
                sleep_linear_threshold=0.05,
                sleep_angular_threshold=0.15,
                sleep_quiet_time=0.5,
                enable_joint_limits=True,
                pgs_iterations=8,
                friction_anchor_beta=0.0,
                dense_max_constraints=256,
                mf_max_constraints=256,
            )
            original_indices = solver._joint_limit_q_index.numpy().copy()
            test.assertEqual(int(np.count_nonzero(original_indices >= 0)), 8)
            test.assertTrue(solver._joint_limit_warp_kernels)
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=128)
            contacts = pipeline.contacts()
            states = [model.state(), model.state()]
            control = model.control()
            for _ in range(2):
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                states.reverse()
            with wp.ScopedCapture(device=model.device) as capture:
                for _ in range(2):
                    pipeline.collide(states[0], contacts)
                    solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                    states.reverse()
            for _ in range(1200):
                wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [0])
            if skip:
                np.testing.assert_array_equal(solver.constraint_count.numpy(), [0])
                np.testing.assert_array_equal(solver.mf_constraint_count.numpy(), [0])
                test.assertTrue(np.all(solver.sleeping.limit_q_index.numpy() < 0))
            else:
                test.assertGreaterEqual(int(solver.constraint_count.numpy().sum()), 16)
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[0, 0] = 100.0
            for state in states:
                state.body_f.assign(force)

            wp.capture_launch(capture.graph)

            np.testing.assert_array_equal(solver.sleeping.art_awake.numpy(), [1])
            test.assertGreaterEqual(int(solver.constraint_count.numpy().sum()), 16)
            np.testing.assert_array_equal(solver.sleeping.limit_q_index.numpy(), original_indices)
            test.assertTrue(np.isfinite(states[0].body_q.numpy()).all())
            test.assertTrue(np.isfinite(states[0].body_qd.numpy()).all())
            test.assertFalse(solver.constraint_overflow.numpy().any())


def test_settled_rows_drop_without_changing_contacts(test, device):
    """Remove solver rows while retaining complete collision contact inputs."""
    scene = _Scene(device)
    scene.settle()
    scene.pipeline.collide(scene.states[0], scene.contacts)
    before = _contact_inputs(scene.contacts)
    test.assertGreater(int(before["count"][0]), 0)
    pose = scene.states[0].body_q.numpy().copy()

    scene.step(collide=False)

    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    np.testing.assert_array_equal(scene.solver.sleeping.body_awake.numpy(), [0, 0])
    np.testing.assert_array_equal(scene.states[0].body_q.numpy(), pose)
    for name, values in before.items():
        np.testing.assert_array_equal(_contact_inputs(scene.contacts)[name], values, err_msg=name)


def test_explicit_full_solve_retains_rows(test, device):
    """Compare settled row counts against explicit contact-skipping disablement."""
    full, skipped = _Scene(device, skip=False), _Scene(device)
    for scene in (full, skipped):
        scene.settle()
        scene.step()
        np.testing.assert_array_equal(scene.solver.sleeping.body_awake.numpy(), [0, 0])
        test.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertGreater(int(full.row_counts().sum()), 0)
    np.testing.assert_array_equal(skipped.row_counts(), [0, 0])
    np.testing.assert_allclose(full.states[0].body_q.numpy(), skipped.states[0].body_q.numpy(), atol=1.0e-6, rtol=0)


def test_force_restores_only_independent_ground_island_rows(test, device):
    """Restore forced-body rows while another ground-supported island sleeps."""
    scene = _Scene(device)
    scene.step()
    active_rows = int(scene.row_counts().sum())
    test.assertGreater(active_rows, 0)
    scene.settle()
    test.assertNotEqual(*scene.solver.sleeping.body_island.numpy())
    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    resting_pose = scene.states[0].body_q.numpy()[1].copy()
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = 100.0
    scene.states[0].body_f.assign(force)

    scene.step(clear=False)

    rows = int(scene.row_counts().sum())
    test.assertGreater(rows, 0)
    test.assertLess(rows, active_rows)
    np.testing.assert_array_equal(scene.solver.sleeping.body_awake.numpy(), [1, 0])
    test.assertGreater(float(scene.states[0].body_qd.numpy()[0, 0]), 0.0)
    np.testing.assert_array_equal(scene.states[0].body_q.numpy()[1], resting_pose)


def test_support_loss_wakes_before_skipped_step(test, device):
    """Let a newly unsupported sleeper fall during the same small timestep."""
    scene = _Scene(device)
    scene.settle()
    scene.pipeline.collide(scene.states[0], scene.contacts)
    count = int(scene.contacts.rigid_contact_count.numpy()[0])
    shape_body = scene.model.shape_body.numpy()
    body0 = shape_body[scene.contacts.rigid_contact_shape0.numpy()[:count]]
    body1 = shape_body[scene.contacts.rigid_contact_shape1.numpy()[:count]]
    keep = np.flatnonzero(((body0 == 0) & (body1 < 0)) | ((body1 == 0) & (body0 < 0)))
    test.assertGreater(len(keep), 0)
    test.assertTrue(np.any((body0 == 1) | (body1 == 1)))
    for suffix in _CONTACT_FIELDS:
        array = getattr(scene.contacts, "rigid_contact_" + suffix)
        values = array.numpy()
        values[: len(keep)] = values[keep].copy()
        array.assign(values)
    scene.contacts.rigid_contact_count.assign(np.array([len(keep)], dtype=np.int32))
    before = float(scene.states[0].body_q.numpy()[1, 2])

    scene.step(collide=False, dt=0.001)

    np.testing.assert_array_equal(scene.solver.sleeping.body_awake.numpy(), [0, 1])
    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    test.assertLess(float(scene.states[0].body_q.numpy()[1, 2]), before)
    velocity = float(scene.states[0].body_qd.numpy()[1, 2])
    test.assertLess(velocity, 0.0)
    test.assertLess(abs(velocity), scene.solver.sleeping.linear_threshold)


def test_fixed_table_supports_sleep_and_independent_force_wake(test, device):
    """Sleep both table-supported boxes and restore only the forced box's rows."""
    scene = _Scene(device, fixed_table=True)
    scene.step()
    active_rows = int(scene.row_counts().sum())
    test.assertGreater(active_rows, 0)
    scene.settle()
    islands = scene.solver.sleeping.body_island.numpy()[scene.bodies]
    test.assertNotEqual(*islands)
    np.testing.assert_array_equal(scene.solver.sleeping.root_supported.numpy()[islands], [1, 1])
    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    resting_pose = scene.states[0].body_q.numpy()[scene.bodies[1]].copy()
    force = np.zeros((scene.model.body_count, 6), dtype=np.float32)
    force[scene.bodies[0], 0] = 100.0
    scene.states[0].body_f.assign(force)

    scene.step(clear=False)

    np.testing.assert_array_equal(scene.solver.sleeping.body_awake.numpy()[scene.bodies], [1, 0])
    rows = int(scene.row_counts().sum())
    test.assertGreater(rows, 0)
    test.assertLess(rows, active_rows)
    test.assertGreater(float(scene.states[0].body_qd.numpy()[scene.bodies[0], 0]), 0.0)
    np.testing.assert_array_equal(scene.states[0].body_q.numpy()[scene.bodies[1]], resting_pose)


def test_graph_replay_restores_rows_on_force(test, device):
    """Restore matrix-free contact rows on a captured sleeping-to-awake transition."""
    scene = _Scene(device)
    scene.step()
    test.assertGreater(int(scene.solver.mf_constraint_count.numpy().sum()), 0)
    scene.step()
    with wp.ScopedCapture(device=scene.model.device) as capture:
        for _ in range(2):
            scene.step(clear=False)
    for _ in range(100):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(scene.solver.sleeping.body_awake.numpy(), [0, 0])
    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    test.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = 100.0
    for state in scene.states:
        state.body_f.assign(force)

    wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(scene.solver.sleeping.body_awake.numpy(), [1, 0])
    test.assertGreater(int(scene.solver.mf_constraint_count.numpy().sum()), 0)
    test.assertGreater(float(scene.states[0].body_qd.numpy()[0, 0]), 0.0)
    test.assertTrue(np.isfinite(scene.states[0].body_q.numpy()).all())
    test.assertTrue(np.isfinite(scene.states[0].body_qd.numpy()).all())
    test.assertFalse(scene.solver.constraint_overflow.numpy().any())


def test_nonfinite_prescribed_velocity_wakes_before_allocation(test, device):
    """Wake supported sleepers when their fixed table has invalid velocity."""
    scene = _Scene(device, fixed_table=True)
    scene.settle()
    scene.pipeline.collide(scene.states[0], scene.contacts)
    test.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
    pose = scene.states[0].body_q.numpy().copy()
    velocity = scene.states[0].body_qd.numpy()
    velocity[0, 0] = np.nan
    scene.states[0].body_qd.assign(velocity)

    scene.solver.sleeping.begin(scene.states[0], scene.control, scene.contacts)

    np.testing.assert_array_equal(scene.solver.sleeping.art_awake.numpy()[scene.bodies], [1, 1])
    np.testing.assert_array_equal(scene.states[0].body_q.numpy(), pose)
    test.assertTrue(np.isnan(scene.states[0].body_qd.numpy()[0, 0]))


class _Scene:
    def __init__(self, device, *, skip=True, fixed_table=False):
        builder = newton.ModelBuilder()
        if fixed_table:
            table = builder.add_link()
            root = builder.add_joint_fixed(parent=-1, child=table)
            builder.add_articulation([root])
            builder.add_shape_box(table, hx=1.0, hy=0.5, hz=0.1)
        else:
            builder.add_ground_plane()
        self.bodies = []
        for x in (0.0, 0.5):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, 0.0, 0.2 if fixed_table else 0.1), wp.quat_identity())
            )
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            self.bodies.append(body)
        self.model = builder.finalize(device=device)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=64)
        self.contacts = self.pipeline.contacts()
        options = {} if skip else {"sleep_skip_constraints": False}
        self.solver = newton.solvers.SolverFeatherPGS(
            self.model,
            enable_sleeping=True,
            sleep_quiet_time=0.05,
            friction_anchor_beta=0.0,
            pgs_iterations=16,
            **options,
        )
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()

    def step(self, *, clear=True, collide=True, dt=0.005):
        if clear:
            self.states[0].clear_forces()
        if collide:
            self.pipeline.collide(self.states[0], self.contacts)
        self.solver.step(self.states[0], self.states[1], self.control, self.contacts, dt)
        self.states.reverse()

    def settle(self):
        for _ in range(120):
            self.step()
        np.testing.assert_array_equal(self.solver.sleeping.body_awake.numpy()[self.bodies], [0, 0])
        if self.solver.constraint_overflow.numpy().any():
            raise AssertionError("Contact capacity exceeded")

    def row_counts(self):
        return np.array([self.solver.constraint_count.numpy().sum(), self.solver.mf_constraint_count.numpy().sum()])


_CONTACT_FIELDS = ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1", "tids")


def _contact_inputs(contacts):
    values = {name: getattr(contacts, "rigid_contact_" + name).numpy().copy() for name in _CONTACT_FIELDS}
    values["count"] = contacts.rigid_contact_count.numpy().copy()
    return values


devices = get_cuda_test_devices()


class TestFeatherPGSSleepWorkSkipping(unittest.TestCase):
    pass


for _name in (
    "test_ant_joint_limits_skip_and_restore",
    "test_settled_rows_drop_without_changing_contacts",
    "test_explicit_full_solve_retains_rows",
    "test_force_restores_only_independent_ground_island_rows",
    "test_support_loss_wakes_before_skipped_step",
    "test_fixed_table_supports_sleep_and_independent_force_wake",
    "test_graph_replay_restores_rows_on_force",
    "test_nonfinite_prescribed_velocity_wakes_before_allocation",
):
    add_function_test(TestFeatherPGSSleepWorkSkipping, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
