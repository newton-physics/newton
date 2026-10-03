# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Integration checks for the sparse dynamics and contact path."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_propagation_free_root_warp import _build_model as _build_chain

DT = 1.0 / 400.0
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")


def _build_model(*, free_count=0, free_first=False, robot_count=1, prescribed=False, heterogeneous=False):
    """Replicate a contacting tripod, optionally surrounded by separate free bodies."""
    env = newton.ModelBuilder()
    if prescribed:
        support = env.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.02), wp.quat_identity()), is_kinematic=True)
        env.add_shape_box(support, hx=0.6, hy=0.4, hz=0.02)
        env.joint_qd[:6] = [0.2, 0.0, 0.04, 0.0, 0.0, 0.15]

    def add_objects():
        for index in range(free_count):
            body = env.add_link(
                xform=wp.transform(
                    wp.vec3(0.26 + index * 0.095, 0.0, 0.05),
                    wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.4, 0.5)), 0.6 + index * 0.2),
                ),
                mass=0.4 + index * 0.2,
                com=wp.vec3(0.01, -0.015, 0.007),
                inertia=wp.mat33(np.diag([0.003, 0.004, 0.005])),
            )
            cfg = env.default_shape_cfg.copy()
            cfg.density, cfg.mu = 0.0, 0.6
            env.add_shape_sphere(body, radius=0.05, cfg=cfg)
            env.add_articulation([env.add_joint_free(body)])

    if free_first:
        add_objects()
    for robot in range(robot_count):
        robot_q_start = len(env.joint_q)
        root = env.add_link(
            xform=wp.transform(wp.vec3(robot * 0.105, 0, 0.16), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3) * 0.03),
        )
        joints = [env.add_joint_free(root)]
        for x, y in ((0.16, 0.0), (-0.08, 0.14), (-0.08, -0.14)):
            child = env.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.002))
            cfg = env.default_shape_cfg.copy()
            cfg.density, cfg.mu = 0.0, 0.6
            env.add_shape_sphere(child, radius=0.055, cfg=cfg)
            joints.append(
                env.add_joint_revolute(
                    root,
                    child,
                    axis=newton.Axis.Y,
                    parent_xform=wp.transform(wp.vec3(x, y, -0.11), wp.quat_identity()),
                    target_pos=0.05,
                    target_ke=15.0,
                    target_kd=1.0,
                    armature=0.05,
                    limit_lower=-0.2,
                    limit_upper=0.2,
                    effort_limit=10.0,
                )
            )
        env.add_articulation(joints)
        env.joint_q[robot_q_start + 7] = 0.23  # Exercise an actual unilateral limit on the first step.
    if not free_first:
        add_objects()
    builder = newton.ModelBuilder()
    if heterogeneous:
        builder.add_world(env)
        free_world = newton.ModelBuilder()
        body = free_world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
        free_world.add_shape_sphere(body, radius=0.05)
        builder.add_world(free_world)
    else:
        builder.replicate(env, 2)
    if not prescribed:
        builder.add_ground_plane()
    return builder.finalize(device="cuda:0")


def _make_solver(model, sparse, **overrides):
    """Change only the representation selection between comparison solvers."""
    options = {
        "pgs_mode": "matrix_free",
        "articulated_contact_response": "immediate",
        "drive_mode": "augmented",
        "friction_mode": "current",
        "pgs_iterations": 32,
        "enable_joint_limits": True,
        "enable_joint_velocity_limits": False,
        "pgs_warmstart": False,
        "mf_warmstart": False,
        "dense_max_constraints": 32,
        "use_parallel_streams": True,
        "double_buffer": False,
        "update_mass_matrix_interval": 4,
    }
    options.update(overrides)
    with mock.patch.dict(SolverFeatherPGS._kernel_overrides, {"sparse_mass_matrix": sparse}):
        return SolverFeatherPGS(model, **options)


def _case(sparse, *, free_count=0, free_first=False, robot_count=1, prescribed=False, heterogeneous=False):
    """Allocate a stable input state for eager and captured cache reuse."""
    model = _build_model(
        free_count=free_count,
        free_first=free_first,
        robot_count=robot_count,
        prescribed=prescribed,
        heterogeneous=heterogeneous,
    )
    solver = _make_solver(
        model,
        sparse,
        **(
            {"dense_max_constraints": 96 if robot_count > 1 else 64, "mf_max_constraints": 64}
            if free_count or robot_count > 1
            else {}
        ),
    )
    state, out = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128 if free_count else 64)
    return model, solver, state, out, model.control(), pipeline, pipeline.contacts()


def _step(case):
    """Advance without swapping the input object used by the dynamics cache."""
    _, solver, state, out, control, pipeline, contacts = case
    state.clear_forces()
    pipeline.collide(state, contacts)
    solver.step(state, out, control, contacts, DT)
    for name in STATE_FIELDS:
        wp.copy(getattr(state, name), getattr(out, name))


@unittest.skipUnless(wp.is_cuda_available(), "Sparse integrated solve requires CUDA")
class TestFeatherPGSSparseSolver(unittest.TestCase):
    def test_heterogeneous_worlds_do_not_invent_joint_support(self):
        """Admit a robot-only world and a separate free-only world without oversized row support."""
        dense, sparse = _case(False, heterogeneous=True), _case(True, heterogeneous=True)
        self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
        np.testing.assert_array_equal(sparse[1].world_dof_count.numpy(), [9, 6])
        self.assertLessEqual(sparse[1]._sparse_row_dof.shape[2], sparse[1].max_world_dofs)
        for _ in range(4):
            for case in (dense, sparse):
                _step(case)
            self.assert_states_close(dense[2], sparse[2])
        self.assert_sparse_state(sparse)

    def test_moving_prescribed_support_and_reset(self):
        """Preserve nonzero prescribed contact targets, support velocity and masked reset."""
        dense = _case(False, free_count=1, prescribed=True)
        sparse = _case(True, free_count=1, prescribed=True)
        self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
        self.assertTrue(sparse[1]._has_prescribed_response)
        targets_seen = [False, False]
        for step in range(8):
            if step == 4:
                for model, solver, state, *_ in (dense, sparse):
                    q = state.joint_q.numpy()
                    q[2] += 0.002
                    state.joint_q.assign(q)
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
            for case in (dense, sparse):
                _step(case)
            self.assert_states_close(dense[2], sparse[2])
            for family, (counter, target) in enumerate(
                (
                    ("constraint_count", "target_velocity"),
                    ("mf_constraint_count", "mf_target_velocity"),
                )
            ):
                counts = getattr(sparse[1], counter).numpy()
                for world, count in enumerate(counts):
                    actual = getattr(sparse[1], target).numpy()[world, :count]
                    expected = getattr(dense[1], target).numpy()[world, :count]
                    np.testing.assert_allclose(actual, expected, atol=2.0e-5, rtol=2.0e-5)
                    targets_seen[family] |= bool(np.any(np.abs(actual) > 0.01))
            prescribed_bodies = np.flatnonzero(sparse[1]._body_prescribed.numpy())
            np.testing.assert_allclose(
                sparse[2].body_qd.numpy()[prescribed_bodies],
                np.tile([0.2, 0.0, 0.04, 0.0, 0.0, 0.15], (len(prescribed_bodies), 1)),
                atol=1.0e-6,
            )
        self.assertEqual(targets_seen, [True, True], "Exercise prescribed targets on both contact families")
        self.assert_sparse_state(sparse)

    def assert_states_close(self, first, second):
        """Allow float32 factor-order roundoff, not a changed physics recipe."""
        for name in STATE_FIELDS:
            # Independent elimination/reduction orders accumulate over 32 contact steps.
            tolerance = 2.0e-3 if name.endswith("qd") else 2.0e-4
            np.testing.assert_allclose(
                getattr(first, name).numpy(), getattr(second, name).numpy(), rtol=2.0e-4, atol=tolerance, err_msg=name
            )

    def assert_sparse_state(self, case):
        """Check storage elimination, row support, impulses, and residual coordinates."""
        _, solver, state, *_ = case
        size = solver._sparse_mass_matrix_size
        self.assertEqual(size, 9)
        for array in (
            solver.J_world,
            solver.Y_world,
            solver.J_by_size[size],
            solver.Y_by_size[size],
            solver.H_by_size[size],
            solver.L_by_size[size],
        ):
            self.assertEqual(array.shape, (1, 1, 1))  # Only argument stand-ins remain.
        for name in ("tau_by_size", "qdd_by_size"):
            self.assertNotIn(size, getattr(solver, name))
        for name in (
            "_cholesky_kernels_by_size",
            "_crba_cholesky_kernels_by_size",
            "_crba_cholesky_warp_kernels_by_size",
            "_triangular_solve_kernels_by_size",
        ):
            self.assertIsNone(getattr(solver, name)[size])
        np.testing.assert_array_equal(solver._sparse_mass_matrix_status.numpy(), 0)
        self.assertFalse(solver.constraint_overflow.numpy().any())
        for name in STATE_FIELDS:
            self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
        counts, impulses = solver.constraint_count.numpy(), solver.impulses.numpy()
        incidence, rhs = solver._sparse_row_incident.numpy(), solver.rhs.numpy()
        v_hat, v_out = solver.v_hat.numpy(), solver.v_out.numpy()
        for world in range(solver.world_count):
            count = int(counts[world])
            self.assertLessEqual(count, solver.dense_max_constraints)
            self.assertTrue(np.isfinite(impulses[world, :count]).all())
            kinds = solver.row_type.numpy()[world, :count]
            unilateral = (kinds == PGS_CONSTRAINT_TYPE_CONTACT) | (kinds == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
            self.assertTrue(np.all(impulses[world, :count][unilateral] >= -1.0e-7))
            z, physical_j = self._sparse_rows_physical(solver, world, count)
            velocity = solver.world_dof_indices.numpy()[world]
            np.testing.assert_allclose(physical_j @ v_hat[velocity], incidence[world, :count], atol=2.0e-5, rtol=2.0e-5)
            factor_residual = (
                z @ solver._sparse_factor_velocity_delta.numpy()[world] + incidence[world, :count] + rhs[world, :count]
            )
            np.testing.assert_allclose(
                physical_j @ v_out[velocity] + rhs[world, :count], factor_residual, atol=3.0e-5, rtol=3.0e-5
            )

    def _sparse_rows_physical(self, solver, world, count):
        """Recover physical Jacobians independently from the stored inverse factors."""
        plan = solver._sparse_mass_matrix_plan
        size = plan.dof_count
        row_dof, values = solver._sparse_row_dof.numpy(), solver._sparse_row_factor.numpy()
        z = np.zeros((count, solver.max_world_dofs))
        for row in range(count):
            support = row_dof[world, row]
            valid = support >= 0
            self.assertTrue(np.all(support[valid] < solver.max_world_dofs))
            self.assertEqual(len(set(support[valid])), int(valid.sum()))
            z[row, support[valid]] = values[world, row, valid]
        physical_j = np.zeros_like(z)
        for group, art in enumerate(solver.group_to_art[size].numpy()):
            if solver.art_to_world.numpy()[art] != world:
                continue
            inverse = np.zeros((size, size))
            inverse[plan.entry_rows, plan.columns] = solver._sparse_Linv.numpy()[group]
            offset = int(solver.articulation_world_dof_offset.numpy()[art])
            physical_j[:, offset + plan.permutation] = np.linalg.solve(inverse, z[:, offset : offset + size].T).T
        if solver._has_free_rigid_bodies:
            for art in solver.group_to_art[6].numpy():
                if solver.art_to_world.numpy()[art] == world:
                    offset = int(solver.articulation_world_dof_offset.numpy()[art])
                    physical_j[:, offset : offset + 6] = z[:, offset : offset + 6]
        return z, physical_j

    def _order_multi_articulation_rows(self, case):
        """Give this fixture identical semantic row order without changing production allocation."""
        _, solver, _, _, _, _, contacts = case
        sparse = solver._sparse_mass_matrix_size is not None
        names = [
            "rhs",
            "diag",
            "impulses",
            "row_type",
            "row_parent",
            "row_mu",
            "row_beta",
            "row_cfm",
            "phi",
            "target_velocity",
            "row_restitution",
        ]
        names += ["_sparse_row_dof", "_sparse_row_factor", "_sparse_row_incident"] if sparse else ["J_world", "Y_world"]
        arrays = {name: getattr(solver, name).numpy() for name in names}
        worlds, slots, needed = (
            getattr(solver, name).numpy() for name in ("contact_world", "contact_slot", "contact_slots_needed")
        )
        shape_a, shape_b = contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()
        identities = []
        for world, count in enumerate(solver.constraint_count.numpy()):
            jacobian = (
                self._sparse_rows_physical(solver, world, count)[1] if sparse else arrays["J_world"][world, :count]
            )
            bundles = []
            for row in np.flatnonzero(arrays["row_type"][world, :count] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT):
                dof = int(np.argmax(np.abs(jacobian[row])))
                bundles.append(((0, dof, int(np.sign(jacobian[row, dof]))), [row]))
            active = [
                c for c in range(int(contacts.rigid_contact_count.numpy()[0])) if worlds[c] == world and slots[c] >= 0
            ]
            for c in active:
                bundles.append(((1, int(shape_a[c]), int(shape_b[c])), list(range(slots[c], slots[c] + needed[c]))))
            bundles.sort(key=lambda item: item[0])
            keys = [key for key, _ in bundles]
            self.assertEqual(len(keys), len(set(keys)), "Fixture must have unique limits and shape-pair contacts")
            order = np.array([row for _, rows in bundles for row in rows], dtype=int)
            np.testing.assert_array_equal(np.sort(order), np.arange(count))
            inverse = np.argsort(order)
            for values in arrays.values():
                values[world, :count] = values[world, order]
            parent = arrays["row_parent"][world, :count]
            parent[parent >= 0] = inverse[parent[parent >= 0]]
            slots[active] = inverse[slots[active]]
            identities.append([(key, len(rows)) for key, rows in bundles])
        for name, values in arrays.items():
            getattr(solver, name).assign(values)
        solver.contact_slot.assign(slots)
        return identities

    def test_two_articulations_share_contact_solve(self):
        """Distinguish factor groups from worlds when two branched robots contact each other."""
        dense, sparse = _case(False, robot_count=2), _case(True, robot_count=2)
        self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
        self.assertEqual(sparse[1]._sparse_Linv.shape[0], 4)
        self.assertEqual(sparse[1].world_count, 2)
        contact_seen = False
        for _ in range(16):
            row_orders = []
            for case in (dense, sparse):
                solve = case[1]._launch_matrix_free_gs_solve

                def ordered_solve(*args, _case=case, _solve=solve, _orders=row_orders, **kwargs):
                    # Both production GPU paths solve the same semantic order;
                    # atomic allocation order is not a physics-equivalence invariant.
                    _orders.append(self._order_multi_articulation_rows(_case))
                    return _solve(*args, **kwargs)

                with mock.patch.object(case[1], "_launch_matrix_free_gs_solve", side_effect=ordered_solve):
                    _step(case)
            self.assertEqual(len(row_orders), 2)
            self.assertEqual(*row_orders)
            self.assert_states_close(dense[2], sparse[2])
            model, solver, _, _, _, _, contacts = sparse
            active = int(contacts.rigid_contact_count.numpy()[0])
            shape_body, body_art = model.shape_body.numpy(), solver.body_to_articulation.numpy()
            for shape_a, shape_b in zip(
                contacts.rigid_contact_shape0.numpy()[:active],
                contacts.rigid_contact_shape1.numpy()[:active],
                strict=True,
            ):
                body_a, body_b = shape_body[shape_a], shape_body[shape_b]
                contact_seen |= body_a >= 0 and body_b >= 0 and body_art[body_a] != body_art[body_b]
        self.assertTrue(contact_seen, "Fixture must couple separate responsive articulations")
        self.assert_sparse_state(sparse)

    def test_mixed_contacts_match_existing_solver(self):
        """Preserve robot/object, free/ground and free/free impulses in either articulation order."""
        for free_first in (False, True):
            with self.subTest(free_first=free_first):
                dense = _case(False, free_count=2, free_first=free_first)
                sparse = _case(True, free_count=2, free_first=free_first)
                self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
                self.assertEqual(sparse[0].articulation_count, 6)
                for case in (dense, sparse):
                    _step(case)
                self.assert_states_close(dense[2], sparse[2])
                for name in ("constraint_count", "mf_constraint_count"):
                    np.testing.assert_array_equal(getattr(dense[1], name).numpy(), getattr(sparse[1], name).numpy())
                for count_name, impulse_name in (
                    ("constraint_count", "impulses"),
                    ("mf_constraint_count", "mf_impulses"),
                ):
                    counts = getattr(sparse[1], count_name).numpy()
                    self.assertTrue(np.all(counts > 0), count_name)
                    for world, count in enumerate(counts):
                        np.testing.assert_allclose(
                            getattr(sparse[1], impulse_name).numpy()[world, :count],
                            getattr(dense[1], impulse_name).numpy()[world, :count],
                            atol=2.0e-4,
                            rtol=2.0e-3,
                        )
                contacts = sparse[-1]
                active = int(contacts.rigid_contact_count.numpy()[0])
                shape_body = sparse[0].shape_body.numpy()
                body_art = sparse[1].body_to_articulation.numpy()
                free = sparse[1].is_free_rigid.numpy()
                seen = set()
                for shape_a, shape_b in zip(
                    contacts.rigid_contact_shape0.numpy()[:active],
                    contacts.rigid_contact_shape1.numpy()[:active],
                    strict=True,
                ):
                    bodies = (shape_body[shape_a], shape_body[shape_b])
                    kinds = tuple(
                        sorted("static" if body < 0 else "free" if free[body_art[body]] else "robot" for body in bodies)
                    )
                    seen.add(kinds)
                self.assertTrue({("free", "robot"), ("free", "static"), ("free", "free")} <= seen, seen)
                for _ in range(15):
                    for case in (dense, sparse):
                        _step(case)
                    self.assert_states_close(dense[2], sparse[2])
                self.assert_sparse_state(sparse)

    def test_mixed_graph_replay_after_masked_reset(self):
        """Refresh a reset robot/free world and replay its complete shared solve graph."""
        eager = _case(True, free_count=2, free_first=True)
        graph = _case(True, free_count=2, free_first=True)
        for case in (eager, graph):
            _step(case)
            model, solver, state, *_ = case
            robot_art = int(solver.group_to_art[9].numpy()[0])
            root_joint = int(model.articulation_start.numpy()[robot_art])
            q_start = int(model.joint_q_start.numpy()[root_joint])
            q = state.joint_q.numpy()
            q[q_start + 2] += 0.01
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        held = graph[1]._sparse_Linv.numpy().copy()
        for case in (eager, graph):
            _step(case)
        np.testing.assert_array_equal(graph[1]._sparse_Linv.numpy()[1], held[1])
        with wp.ScopedCapture("cuda:0") as capture:
            for _ in range(4):
                _step(graph)
        for _ in range(4):
            wp.capture_launch(capture.graph)
            for _ in range(4):
                _step(eager)
        self.assert_states_close(eager[2], graph[2])
        self.assert_sparse_state(graph)

    def test_trajectory_masked_reset_and_inertial_notify(self):
        """Match 32 contact steps and refresh only the requested cached factors."""
        dense, sparse = _case(False), _case(True)
        self.assertIsNone(dense[1]._sparse_mass_matrix_size)
        self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
        contact_seen = limit_seen = False
        for _ in range(32):
            for case in (dense, sparse):
                _step(case)
            contact_seen |= bool(sparse[-1].rigid_contact_count.numpy()[0])
            limit_seen |= bool(np.any(sparse[1].row_type.numpy() == PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
            self.assert_states_close(dense[2], sparse[2])
        self.assertTrue(contact_seen and limit_seen, "Fixture must exercise contacts and joint limits")
        self.assert_sparse_state(sparse)
        for case in (dense, sparse):
            _step(case)  # Step32 refreshes globally; step33 is a held-mass step.
        held = sparse[1]._sparse_Linv.numpy().copy()
        for model, solver, state, *_ in (dense, sparse):
            q = state.joint_q.numpy()
            q[2] += 0.02
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1].mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(sparse[1]._sparse_Linv.numpy()[1], held[1])
        self.assert_states_close(dense[2], sparse[2])
        for model, solver, *_ in (dense, sparse):
            mass = model.body_mass.numpy()
            mass[0] *= 1.1
            model.body_mass.assign(mass)
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1].mass_update_mask.numpy(), [1, 1])
        self.assertFalse(np.array_equal(sparse[1]._sparse_Linv.numpy()[0], held[0]))
        self.assert_states_close(dense[2], sparse[2])
        self.assert_sparse_state(sparse)

    def test_graph_replay_preserves_sparse_problem(self):
        """Replay a complete four-step mass-refresh cadence without dense scratch."""
        eager, graph = _case(True), _case(True)
        for case in (eager, graph):
            _step(case)  # Finish allocation/compilation outside capture.
        with wp.ScopedCapture("cuda:0") as capture:
            for _ in range(4):
                _step(graph)
        for _ in range(8):
            wp.capture_launch(capture.graph)
            for _ in range(4):
                _step(eager)
        self.assert_states_close(eager[2], graph[2])
        self.assert_sparse_state(graph)

    def test_unsupported_configurations_keep_existing_path(self):
        """Reject full-support chains, torsion, velocity limits, and warm starts."""
        chain = _build_chain("cuda:0", base_z=1.2, contact_sphere="deep", ground=False)
        self.assertIsNone(_make_solver(chain, True)._sparse_mass_matrix_size)
        model = _build_model()
        for options in (
            {"contact_torsion_radius": 0.01},
            {"enable_joint_velocity_limits": True},
            {"pgs_warmstart": True},
            {"mf_warmstart": True},
            {"dense_max_constraints": 2048},
        ):
            with self.subTest(options=options):
                solver = _make_solver(model, True, **options)
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertGreater(solver.J_by_size[9].size, 1)


if __name__ == "__main__":
    unittest.main()
