# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental rolling and spin friction rows on SolverSteel free-body contacts (``steel:condim``)."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel.friction_patches import FrictionPatches
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_CONTACT_ROLLING,
    PGS_CONSTRAINT_TYPE_CONTACT_SPIN,
    PGS_CONSTRAINT_TYPE_FRICTION,
    build_mf_contact_rows,
)
from newton.solvers import SolverBase, SolverSteel
from newton.tests.steel._fixtures import set_torsion_radius
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 480.0
_RUN_TO_RUN = wp.DeterministicMode.RUN_TO_RUN
_RADIUS = 0.03
_MASS = 0.1
_GRAVITY = 9.81
_ROLLING = {"linear": (0.0, -0.3, 0.0), "angular": (10.0, 0.0, 0.0)}
# Principal moments 0.6, 1 and 1.4 with an x-z product: a physically valid anisotropic inertia.
_ANISOTROPIC = np.array([[1.0, 0.0, 0.4], [0.0, 1.0, 0.0], [0.4, 0.0, 1.0]])


def _rows(rolling: float = 0.0, torsional: float = 0.0) -> dict:
    """Return ``_simulate`` options that give every shape condim 6 with these coefficients [m]."""
    return {"condim": 6, "rolling_material": rolling, "torsional_material": torsional}


def _coulomb_stopping_distance(speed: float, rolling: float) -> float:
    """Return a solid sphere's rolling distance [m]: v^2 (1 + I / (m r^2)) = 2 d rolling g / r."""
    return speed**2 * 1.4 / (2.0 * rolling * _GRAVITY / _RADIUS)


def _scene(body: str, tilt_deg: float = 0.0, inertia=None, count: int = 1, anchor_deg=None):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(table, hx=2.0, hy=2.0, hz=0.05)
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    bodies = []
    massless = newton.ModelBuilder.ShapeConfig(density=0.0)
    for index in range(count):
        x = 0.2 * index
        if body == "sphere":
            moments = np.eye(3) * 0.4 * _MASS * _RADIUS**2 if inertia is None else np.asarray(inertia)
            z = _RADIUS
        else:
            moments = np.eye(3) * _MASS * (2.0 * _RADIUS) ** 2 / 6.0
            tilt = math.radians(tilt_deg)
            z = _RADIUS * (math.cos(tilt) + math.sin(tilt)) + 1.0e-4
        inertia_matrix = wp.mat33(*np.asarray(moments, dtype=np.float32).ravel())
        pose = wp.transform((x, 0.0, z), rotation)
        if anchor_deg is None:
            obj = builder.add_body(xform=pose, mass=_MASS, inertia=inertia_matrix, lock_inertia=True)
        else:
            # A FREE root whose parent anchor is rotated: joint_qd is expressed in that rotated frame.
            obj = builder.add_link(xform=pose, mass=_MASS, inertia=inertia_matrix, lock_inertia=True)
            anchor = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(anchor_deg))
            joint = builder.add_joint_free(obj, parent_xform=wp.transform((0.0, 0.0, 0.0), anchor))
            builder.add_articulation([joint])
        if body == "sphere":
            builder.add_shape_sphere(obj, radius=_RADIUS, cfg=massless)
        else:
            builder.add_shape_box(obj, hx=_RADIUS, hy=_RADIUS, hz=_RADIUS, cfg=massless)
        bodies.append(obj)
    return builder.finalize(), bodies


def _simulate(
    body,
    seconds,
    *,
    tilt_deg=0.0,
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    graph=False,
    torsion_radius=0.0,
    contact_matching="disabled",
    inertia=None,
    steps=None,
    rolling_material=None,
    torsional_material=None,
    sliding_material=None,
    condim=None,
    count=1,
    support_angular=None,
    observe=None,
    contact_f=False,
    anchor_deg=None,
    on_solver=None,
    **solver_kwargs,
):
    model, bodies = _scene(body, tilt_deg, inertia, count, anchor_deg)
    if torsion_radius:
        set_torsion_radius(model, torsion_radius)
    if rolling_material is not None:
        model.shape_material_mu_rolling.fill_(rolling_material)
    if torsional_material is not None:
        model.shape_material_mu_torsional.fill_(torsional_material)
    if sliding_material is not None:
        model.shape_material_mu.fill_(sliding_material)
    if condim is not None:
        model.steel.condim.fill_(condim)
    options = {"solve_mode": "matrix_free", "iterations": 32, "deterministic": _RUN_TO_RUN, **solver_kwargs}
    solver = SolverSteel(model, **options)
    if on_solver is not None:
        on_solver(solver)
    pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching=contact_matching)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    qd = state_0.joint_qd.numpy()
    for obj in bodies:
        joint = int(np.flatnonzero(model.joint_child.numpy() == obj)[0])
        dof = int(model.joint_qd_start.numpy()[joint])
        qd[dof : dof + 3] = linear
        qd[dof + 3 : dof + 6] = angular
    if support_angular is not None:
        joint = int(np.flatnonzero(model.joint_child.numpy() == 0)[0])
        dof = int(model.joint_qd_start.numpy()[joint])
        qd[dof + 3 : dof + 6] = support_angular
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    if anchor_deg is not None:
        # Initial velocities are world-frame twists; recover the joint coordinates of the rotated anchor.
        body_qd = state_0.body_qd.numpy()
        for obj in bodies:
            body_qd[obj] = (*linear, *angular)
        state_0.body_qd.assign(body_qd)
        newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    obj = bodies[0]
    start = state_0.body_q.numpy()[obj][:3].copy()

    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts) if contact_f else None

    def step(state_in, state_out):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT, observables=observables)

    if steps is not None:
        for _ in range(steps):
            step(state_0, state_1)
            state_0, state_1 = state_1, state_0
    elif graph:
        with wp.ScopedCapture(device=model.device) as capture:
            step(state_0, state_1)
            step(state_1, state_0)
        for _ in range(int(seconds / _DT) // 2):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(int(seconds / _DT) // 2):
            step(state_0, state_1)
            step(state_1, state_0)
    if observe is not None:
        observe(solver, contacts, observables)
    pose, twist = state_0.body_q.numpy()[obj], state_0.body_qd.numpy()[obj]
    return pose, twist, float(np.linalg.norm(pose[:2] - start[:2]))


@unittest.skipUnless(wp.is_cuda_available(), "Steel matrix-free requires CUDA")
class TestFreeRigidAngularFriction(unittest.TestCase):
    def test_rolling_sphere_stops_at_the_coulomb_distance(self):
        """Stop a rolling sphere within 5% of the analytic Coulomb distance; keep it rolling at condim 3."""
        _, twist_free, travel_free = _simulate("sphere", 0.5, rolling_material=0.004, **_ROLLING)
        self.assertGreater(np.linalg.norm(twist_free[3:6]), 9.0)
        self.assertGreater(travel_free, 0.12)
        for rolling in (0.002, 0.004):
            with self.subTest(rolling=rolling):
                _, twist, travel = _simulate("sphere", 0.8, **_rows(rolling=rolling), **_ROLLING)
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                expected = _coulomb_stopping_distance(0.3, rolling)
                self.assertAlmostEqual(travel, expected, delta=0.05 * expected)

    def test_condim_selects_the_rows(self):
        """Add a spin row at condim 4 and a rolling pair at 6, from the builder's default coefficients."""
        for condim, expected in ((3, 3), (4, 4), (6, 6)):
            seen = {}

            def observe(solver, contacts, observables, seen=seen):
                seen["slots"] = int(solver._contact_slots_needed.numpy()[0])

            with self.subTest(condim=condim):
                _simulate("sphere", 0.0, steps=1, condim=condim, observe=observe)
                self.assertEqual(seen["slots"], expected)

    def test_spin_row_stops_a_spinning_sphere_at_the_coulomb_rate(self):
        """Decelerate a sphere spinning on its contact point at mu_s m g / I, and stop it."""
        torsional = 5.0e-4
        _, free, _ = _simulate("sphere", 0.2, angular=(0.0, 0.0, 6.0))
        self.assertGreater(free[5], 5.9)
        rate = torsional * _MASS * _GRAVITY / (0.4 * _MASS * _RADIUS**2)
        for condim in (4, 6):
            with self.subTest(condim=condim):
                options = {"condim": condim, "torsional_material": torsional, "rolling_material": 0.0}
                _, twist, _ = _simulate("sphere", 0.2, angular=(0.0, 0.0, 6.0), **options)
                self.assertAlmostEqual(twist[5], 6.0 - rate * 0.2, delta=0.1)
                _, twist, _ = _simulate("sphere", 0.6, angular=(0.0, 0.0, 6.0), **options)
                self.assertLess(abs(twist[5]), 1.0e-3)

    def test_patch_spin_row_takes_the_pooled_load(self):
        """Decelerate a spinning cube at mu_s m g / I with one patch spin row, as with a spin row per contact.

        Sliding friction is off so that only the spin rows resist the spin.
        """
        torsional = 5.0e-4
        rate = torsional * _MASS * _GRAVITY / (_MASS * (2.0 * _RADIUS) ** 2 / 6.0)
        options = {"angular": (0.0, 0.0, 6.0), "condim": 4, "torsional_material": torsional, "sliding_material": 0.0}
        for beta in (0.0, 0.2):
            with self.subTest(friction_anchor_beta=beta):
                _, twist, _ = _simulate("box", 0.2, friction_anchor_beta=beta, **options)
                self.assertAlmostEqual(twist[5], 6.0 - rate * 0.2, delta=0.1 * rate * 0.2)

    def test_rows_never_add_energy(self):
        """End a contact step with no more kinetic energy than it started with, for anisotropic inertias."""
        rng = np.random.default_rng(3)
        cases = [(_ANISOTROPIC * 4.0e-5, (1.0, 0.0, -3.0), rolling) for rolling in (1.0, 1.0e-3, 1.0e-4)]
        for _ in range(6):
            axes, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            moments = np.sort(rng.uniform(0.2, 1.0, 3))
            moments[2] = min(moments[2], moments[0] + moments[1] - 1.0e-3)
            cases.append((axes @ np.diag(moments) @ axes.T * 4.0e-5, tuple(rng.normal(size=3) * 3.0), 1.0e-3))

        for index, (inertia, omega, rolling) in enumerate(cases):

            def energy(twist, inertia=inertia):
                return 0.5 * _MASS * float(twist[:3] @ twist[:3]) + 0.5 * float(twist[3:6] @ inertia @ twist[3:6])

            with self.subTest(case=index, rolling=rolling):
                kwargs = {"angular": omega, "inertia": inertia, "steps": 1}
                _, twist_free, _ = _simulate("sphere", 0.0, **kwargs)
                _, twist, _ = _simulate("sphere", 0.0, **_rows(rolling=rolling, torsional=rolling), **kwargs)
                start = 0.5 * float(np.asarray(omega) @ inertia @ np.asarray(omega))
                # Gravity does at most m g |v_z| dt of work in one step.
                self.assertLessEqual(energy(twist), start + _MASS * _GRAVITY * 1.0e-3 * _DT)
                self.assertLessEqual(energy(twist), energy(twist_free) * (1.0 + 1.0e-4))

    def test_spin_is_relative_to_a_turning_support(self):
        """Spin a resting sphere up to a kinematic turntable's rate at mu_s m g / I; leave it still at condim 3."""
        torsional = 5.0e-4
        table = (0.0, 0.0, 2.0)
        _, free, _ = _simulate("sphere", 0.5, support_angular=table, torsional_material=torsional)
        self.assertLess(abs(free[5]), 1.0e-3)
        rate = torsional * _GRAVITY / (0.4 * _RADIUS**2)
        _, twist, _ = _simulate("sphere", 0.1, support_angular=table, **_rows(torsional=torsional))
        self.assertAlmostEqual(twist[5], rate * 0.1, delta=0.05)
        _, twist, _ = _simulate("sphere", 0.5, support_angular=table, **_rows(torsional=torsional))
        self.assertAlmostEqual(twist[5], 2.0, delta=1.0e-3)

    def test_rows_act_in_world_frame_under_a_rotated_free_root_anchor(self):
        """Match the identity-anchor result with a FREE root parent anchor rotated by 45 degrees about y.

        A sphere spinning about the contact normal keeps its spin under rolling rows, rolls about the same world axis
        to the same stop, and slows its spin at the same rate under the spin row; the spin row also in a captured graph.
        """
        cases = {
            "normal spin, rolling rows": {"seconds": 0.25, "angular": (0.0, 0.0, 6.0), **_rows(rolling=0.01)},
            "rolling stop": {"seconds": 0.8, **_rows(rolling=0.004), **_ROLLING},
            "spin row": {"seconds": 0.2, "angular": (0.0, 0.0, 6.0), **_rows(torsional=5.0e-4)},
        }
        for name, kwargs in cases.items():
            _, reference, travel = _simulate("sphere", anchor_deg=0.0, **kwargs)
            if name == "normal spin, rolling rows":
                np.testing.assert_allclose(reference[3:6], (0.0, 0.0, 6.0 * (1.0 - 0.05 * _DT) ** 120), atol=1.0e-3)
            for graph in (False, True) if name == "spin row" else (False,):
                with self.subTest(case=name, graph=graph):
                    _, twist, rotated_travel = _simulate("sphere", anchor_deg=45.0, graph=graph, **kwargs)
                    np.testing.assert_allclose(twist, reference, atol=2.0e-4)
                    self.assertAlmostEqual(rotated_travel, travel, delta=1.0e-4)

    def test_patch_angular_rows_do_not_depend_on_contact_order(self):
        """Keep each material's angular rows when two shapes of one body with different condim rest side by side,
        whichever shape's contacts come first."""
        results = []
        for order in ((3, 6), (6, 3)):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            SolverSteel.register_custom_attributes(builder)
            table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
            builder.add_shape_box(table, hx=1.0, hy=1.0, hz=0.05)
            inertia = wp.mat33(*(np.eye(3) * 1.0e-4).astype(np.float32).ravel())
            body = builder.add_body(
                xform=wp.transform((0.0, 0.0, 0.0201), wp.quat_identity()),
                mass=_MASS,
                inertia=inertia,
                lock_inertia=True,
            )
            shapes = {}
            for condim in order:
                x = -0.02 if condim == 3 else 0.02
                shapes[condim] = builder.add_shape_box(
                    body,
                    xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
                    hx=0.02,
                    hy=0.02,
                    hz=0.02,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu_torsional=0.01, mu_rolling=0.002),
                    custom_attributes={"steel:condim": condim},
                )
            model = builder.finalize()
            solver = SolverSteel(model, solve_mode="matrix_free", iterations=32, friction_anchor_beta=0.2)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state_0, state_1, control = model.state(), model.state(), model.control()
            for _ in range(3):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, _DT)
                state_0, state_1 = state_1, state_0
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            slots = solver._contact_slots_needed.numpy()[:count]
            angular = {}
            for condim, shape in shapes.items():
                mask = (shape0 == shape) | (shape1 == shape)
                angular[condim] = int(np.count_nonzero(slots[mask] == 6))
            results.append(angular)
        for angular in results:
            self.assertEqual(angular[3], 0)
            self.assertEqual(angular[6], 1)

    def test_articulated_contacts_group_patches_regardless_of_condim(self):
        """Group an articulated body's patches as without angular rows: mixed condim matches the condim 3 control."""
        seen = {}
        for mixed in (False, True):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            SolverSteel.register_custom_attributes(builder)
            table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
            builder.add_shape_box(table, hx=1.0, hy=1.0, hz=0.05)
            # A free sphere elsewhere keeps the free-body route, and with it the angular rows, in use.
            sphere_inertia = wp.mat33(*(np.eye(3) * 0.4 * _MASS * _RADIUS**2).astype(np.float32).ravel())
            free = builder.add_body(
                xform=wp.transform((0.5, 0.5, _RADIUS), wp.quat_identity()),
                mass=_MASS,
                inertia=sphere_inertia,
                lock_inertia=True,
            )
            builder.add_shape_sphere(
                free,
                radius=_RADIUS,
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
                custom_attributes={"steel:condim": 6 if mixed else 3},
            )
            inertia = wp.mat33(*(np.eye(3) * 1.0e-4).astype(np.float32).ravel())
            link = builder.add_link(mass=_MASS, inertia=inertia, lock_inertia=True)
            joint = builder.add_joint_prismatic(
                -1, link, parent_xform=wp.transform((0.0, 0.0, 0.0201), wp.quat_identity()), axis=newton.Axis.X
            )
            builder.add_articulation([joint])
            for x, condim in ((-0.02, 3), (0.02, 6 if mixed else 3)):
                builder.add_shape_box(
                    link,
                    xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
                    hx=0.02,
                    hy=0.02,
                    hz=0.02,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu_torsional=0.01, mu_rolling=0.002),
                    custom_attributes={"steel:condim": condim},
                )
            model = builder.finalize()
            solver = SolverSteel(model, solve_mode="matrix_free", iterations=32, friction_anchor_beta=0.2)
            self.assertEqual(solver._free_rigid_angular_rows, mixed)
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
            state_0, state_1, control = model.state(), model.state(), model.control()
            for _ in range(3):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, _DT)
                state_0, state_1 = state_1, state_0
            count = int(contacts.rigid_contact_count.numpy()[0])
            dense = solver._contact_path.numpy()[:count] == 0
            seen[mixed] = (
                solver._contact_slots_needed.numpy()[:count][dense].tolist(),
                solver._friction_patches.view.weight.numpy()[:count][dense].tolist(),
                solver._constraint_count.numpy().tolist(),
            )
        self.assertTrue(len(seen[False][0]) > 0)
        self.assertEqual(seen[True], seen[False])

    def test_toppling_and_sliding_cube_are_unchanged(self):
        """Leave a toppling or sliding cube within 2 mm of the condim 3 pose; the spinning cube slows at most."""
        for kwargs in ({"tilt_deg": 40.0}, {"linear": (0.5, 0.0, 0.0)}, {"angular": (0.0, 0.0, 6.0)}):
            with self.subTest(**kwargs):
                pose_free, _, _ = _simulate("box", 0.5, **kwargs)
                pose, _, _ = _simulate("box", 0.5, **_rows(rolling=0.002), **kwargs)
                np.testing.assert_allclose(pose[:3], pose_free[:3], atol=2.0e-3)
                self.assertGreater(abs(float(np.dot(pose[3:7], pose_free[3:7]))), 0.999)

    def test_composes_with_sleeping_anchors_torsion_warm_start_and_velocity_iterations(self):
        """Keep the Coulomb stopping distance with each option that touches contact rows."""
        expected = _coulomb_stopping_distance(0.3, 0.004)
        for options in (
            {"enable_sleeping": True, "friction_anchor_beta": 0.2, "torsion_radius": 0.01},
            {"contact_warmstart": True, "contact_matching": "sticky"},
            {"velocity_iterations": 1},
        ):
            with self.subTest(**options):
                _, twist, travel = _simulate("sphere", 0.8, **_rows(rolling=0.004), **_ROLLING, **options)
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                self.assertAlmostEqual(travel, expected, delta=0.05 * expected)

    def test_patch_carries_one_set_of_angular_rows(self):
        """Give a friction patch's first anchor the angular rows with the whole coefficient; the other keeps sliding only."""
        seen = {}

        def observe(solver, contacts, observables):
            count = int(contacts.rigid_contact_count.numpy()[0])
            slots = solver._contact_slots_needed.numpy()[:count]
            slot = solver._contact_slot.numpy()[:count]
            types = solver._mf_row_type.numpy()[0]
            mu = solver._mf_row_mu.numpy()[0]
            anchors = [int(s) for s, n in zip(slot, slots, strict=True) if n >= 3]
            angular = [int(s) for s, n in zip(slot, slots, strict=True) if n == 6]
            seen["anchors"], seen["angular"] = len(anchors), len(angular)
            for s in angular:
                self.assertEqual(types[s], PGS_CONSTRAINT_TYPE_CONTACT)
                self.assertEqual(types[s + 1], PGS_CONSTRAINT_TYPE_FRICTION)
                self.assertEqual(types[s + 3], PGS_CONSTRAINT_TYPE_CONTACT_ROLLING)
                self.assertEqual(types[s + 4], PGS_CONSTRAINT_TYPE_CONTACT_ROLLING)
                self.assertEqual(types[s + 5], PGS_CONSTRAINT_TYPE_CONTACT_SPIN)
                self.assertAlmostEqual(mu[s + 3], 0.002, places=7)
                self.assertAlmostEqual(mu[s + 5], 0.001, places=7)
                self.assertAlmostEqual(mu[s + 1] * len(anchors), 1.0, places=6)

        _simulate(
            "box", 0.0, steps=3, friction_anchor_beta=0.2, **_rows(rolling=0.002, torsional=0.001), observe=observe
        )
        self.assertEqual((seen["anchors"], seen["angular"]), (2, 1))

    def test_repeated_and_captured_runs_are_bitwise_identical(self):
        """Match two eager runs and a CUDA-graph replay bit for bit, with several bodies."""
        kwargs = {
            "friction_anchor_beta": 0.2,
            "enable_sleeping": True,
            "count": 3,
            **_rows(rolling=0.001, torsional=0.001),
            **_ROLLING,
        }
        first, _, _ = _simulate("sphere", 0.3, **kwargs)
        second, _, _ = _simulate("sphere", 0.3, **kwargs)
        captured, _, _ = _simulate("sphere", 0.3, graph=True, **kwargs)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, captured)

    def test_rows_take_the_larger_coefficient_of_the_pair(self):
        """Average sliding friction as before while the angular rows take the larger coefficient of the two shapes."""
        mu = np.array([0.5, 0.8], dtype=np.float32)
        torsional = np.array([0.002, 0.004], dtype=np.float32)
        rolling = np.array([0.003, 0.001], dtype=np.float32)
        seen = {}

        def on_solver(solver):
            solver.model.shape_material_mu.assign(mu)
            solver.model.shape_material_mu_torsional.assign(torsional)
            solver.model.shape_material_mu_rolling.assign(rolling)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

        def observe(solver, contacts, observables):
            count = int(contacts.rigid_contact_count.numpy()[0])
            needed = solver._contact_slots_needed.numpy()[:count]
            slots = solver._contact_slot.numpy()[:count]
            row = int(slots[int(np.flatnonzero(needed == 6)[0])])
            seen["mu"] = solver._mf_row_mu.numpy()[0][[row + 1, row + 3, row + 5]]

        _simulate("sphere", 0.0, steps=1, condim=6, on_solver=on_solver, observe=observe)
        np.testing.assert_array_equal(seen["mu"], [(mu[0] + mu[1]) / np.float32(2.0), rolling.max(), torsional.max()])

    def test_shape_notify_inside_graph_capture_refreshes_coefficients(self):
        """Capture a shape-property notify that changes mu_rolling, and replay it into the rows' bound."""
        model, _ = _scene("sphere")
        model.steel.condim.fill_(6)
        model.shape_material_mu_rolling.fill_(0.001)
        solver = SolverSteel(model, solve_mode="matrix_free")
        with wp.ScopedCapture(device=model.device) as capture:
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        model.shape_material_mu_rolling.fill_(0.003)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(solver._free_rigid_mu_rolling.numpy()[: model.shape_count], 0.003)

    def test_default_condim_and_zero_coefficients_match_the_base(self):
        """Match the default bit for bit at condim 3, and add no rows for zero coefficients at condim 6."""
        default, _, _ = _simulate("sphere", 0.3, **_ROLLING)
        condim3, _, _ = _simulate("sphere", 0.3, condim=3, **_ROLLING)
        zero, _, _ = _simulate("sphere", 0.3, **_rows(), **_ROLLING)
        np.testing.assert_array_equal(default, condim3)
        np.testing.assert_array_equal(default, zero)

    def test_inactive_friction_zeroes_both_rows_of_each_pair(self):
        """Hold both rows of a sliding and a rolling pair at zero until friction activates, then solve both."""
        model, _ = _scene("sphere")
        model.steel.condim.fill_(6)
        model.shape_material_mu_rolling.fill_(0.002)
        model.shape_material_mu_torsional.fill_(0.0)
        solver = SolverSteel(
            model, solve_mode="matrix_free", iterations=32, deterministic=_RUN_TO_RUN, contact_warmstart=True
        )
        pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching="sticky")
        contacts = pipeline.contacts()
        states = [model.state(), model.state()]
        qd = states[0].joint_qd.numpy()
        qd[:3] = (0.4, 0.25, 0.0)
        states[0].joint_qd.assign(qd)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()

        def pair_impulses():
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, _DT)
            states.reverse()
            count = int(solver._mf_constraint_count.numpy()[0])
            row_type = solver._mf_row_type.numpy()[0, :count]
            impulses = solver._mf_impulses.numpy()[0, :count]
            return impulses[row_type == PGS_CONSTRAINT_TYPE_FRICTION], impulses[
                row_type == PGS_CONSTRAINT_TYPE_CONTACT_ROLLING
            ]

        for _ in range(4):
            sliding, rolling = pair_impulses()
        self.assertEqual((sliding.size, rolling.size), (2, 2))
        self.assertTrue(np.all(sliding != 0.0))
        # The constructor rejects delayed friction with warm starting; set it directly so the second tangent row
        # starts the solve with a carried impulse that only the zeroing of inactive pairs clears.
        solver._contact_friction_position_iterations = 0
        sliding, rolling = pair_impulses()
        np.testing.assert_array_equal(sliding, 0.0)
        np.testing.assert_array_equal(rolling, 0.0)
        solver._contact_friction_position_iterations = 8
        sliding, rolling = pair_impulses()
        self.assertTrue(np.all(sliding != 0.0))
        self.assertTrue(np.all(rolling != 0.0))

    def test_contact_force_keeps_the_sliding_friction_of_contacts_with_angular_rows(self):
        """Report the sliding friction of a contact whose sliding pair is followed by angular rows."""
        seen = {}

        def observe(solver, contacts, observables):
            slot = int(solver._contact_slot.numpy()[0])
            self.assertEqual(int(solver._contact_slots_needed.numpy()[0]), 6)
            row_type = solver._mf_row_type.numpy()[0]
            self.assertEqual(row_type[slot + 1], PGS_CONSTRAINT_TYPE_FRICTION)
            impulses = solver._mf_impulses.numpy()[0, slot : slot + 3]
            seen["sliding"] = float(np.linalg.norm(impulses[1:3])) / _DT
            seen["reported"] = observables.contact_f.numpy()[0, :3]

        _simulate(
            "sphere",
            0.0,
            steps=3,
            linear=(0.5, 0.0, 0.0),
            contact_f=True,
            observe=observe,
            **_rows(rolling=0.002, torsional=0.001),
        )
        self.assertGreater(seen["sliding"], 0.1)
        reported_tangent = float(np.linalg.norm(seen["reported"][:2]))
        self.assertAlmostEqual(reported_tangent, seen["sliding"], delta=1.0e-4 + 1.0e-4 * seen["sliding"])

    def test_invalid_options_raise(self):
        """Reject invalid condim values and unsupported options; a solver without the rows ignores condim edits."""
        model, _ = _scene("sphere")
        model.steel.condim.fill_(6)
        for options, option in (
            ({"solve_mode": "split"}, "solve_mode='split'"),
            # Point friction: bisection with persistent patches would warn before the rejection under test.
            ({"friction_mode": "bisection", "friction_anchor_beta": 0.0}, "friction_mode='bisection'"),
            ({"articulated_contact_response": "propagation"}, "articulated_contact_response='propagation'"),
        ):
            with self.subTest(option=option), self.assertRaisesRegex(NotImplementedError, option):
                SolverSteel(model, **{"solve_mode": "matrix_free", **options})
        for condim in (0, 1, 2, 5, 7):
            model.steel.condim.fill_(condim)
            with self.subTest(condim=condim), self.assertRaisesRegex(ValueError, "3, 4 or 6"):
                SolverSteel(model, solve_mode="matrix_free")
        model.steel.condim.fill_(6)
        solver = SolverSteel(model, solve_mode="matrix_free")
        model.steel.condim.fill_(5)
        spatial_inertia = solver._body_I_m.numpy()
        model.body_mass.assign(2.0 * model.body_mass.numpy())
        flags = newton.ModelFlags.SHAPE_PROPERTIES | newton.ModelFlags.BODY_INERTIAL_PROPERTIES
        with self.assertRaisesRegex(ValueError, "3, 4 or 6"):
            solver.notify_model_changed(flags)
        # The check raises after the notification's other updates are applied.
        self.assertFalse(np.array_equal(solver._body_I_m.numpy(), spatial_inertia))
        # A solver built without the rows ignores later condim edits.
        model.steel.condim.fill_(3)
        solver = SolverSteel(model, solve_mode="matrix_free")
        for condim in (5, 6):
            model.steel.condim.fill_(condim)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        self.assertFalse(solver._free_rigid_angular_rows)


def test_linear_row_builder_leaves_the_angular_rows(test, device):
    """Write only the normal and sliding rows of a six-row free-body contact; rows 3-5 belong to the angular builder."""
    sentinel = 123.0
    patches = FrictionPatches()
    patches.enabled = 0
    patches.weight = wp.zeros(0, dtype=float, device=device)
    patches.next_contact = wp.zeros(0, dtype=int, device=device)
    patches.point_a = wp.zeros(0, dtype=wp.vec3, device=device)
    patches.point_b = wp.zeros(0, dtype=wp.vec3, device=device)
    patches.phi = wp.array([wp.vec2(0.2, 0.3)], dtype=wp.vec2, device=device)

    def ints(value, shape=(1,)):
        return wp.full(shape, value, dtype=int, device=device)

    def floats(value, shape=(1,), dtype=float):
        return wp.full(shape, value, dtype=dtype, device=device)

    outputs = [
        ints(int(sentinel), (1, 6)),
        ints(int(sentinel), (1, 6)),
        floats(sentinel, (1, 6, 6)),
        floats(sentinel, (1, 6, 6)),
        ints(int(sentinel), (1, 6)),
        ints(int(sentinel), (1, 6)),
        floats(sentinel, (1, 6)),
        floats(sentinel, (1, 6)),
        floats(sentinel, (1, 6)),
        floats(sentinel, (1, 6)),
    ]
    wp.launch(
        build_mf_contact_rows,
        dim=1,
        inputs=[
            ints(1),
            1,
            floats(0.0, dtype=wp.vec3),
            floats(0.0, dtype=wp.vec3),
            wp.array([wp.vec3(0.0, 0.0, 1.0)], dtype=wp.vec3, device=device),
            ints(-1),
            ints(-1),
            floats(0.0),
            floats(0.0),
            ints(0),
            ints(0),
            ints(1),
            ints(-1),
            ints(-1),
            ints(0),
            floats(0.0, dtype=wp.vec3),
            ints(-1),
            floats(0.0, dtype=wp.transform),
            floats(0.0, dtype=wp.spatial_vector),
            ints(0),
            0,
            floats(0.5),
            ints(6),
            floats(0.0),
            patches,
            1.0,
            0,
            0,
        ],
        outputs=outputs,
        device=device,
    )
    row_type, row_phi = outputs[4].numpy()[0], outputs[7].numpy()[0]
    np.testing.assert_array_equal(
        row_type[:3], [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]
    )
    np.testing.assert_allclose(row_phi[1:3], (0.2, 0.3), rtol=1.0e-6)
    for output in outputs:
        np.testing.assert_array_equal(output.numpy()[0, 3:], sentinel)


class TestFreeRigidAngularRowBuilder(unittest.TestCase):
    pass


add_function_test(
    TestFreeRigidAngularRowBuilder,
    "test_linear_row_builder_leaves_the_angular_rows",
    test_linear_row_builder_leaves_the_angular_rows,
    devices=get_test_devices(),
)


class TestFreeRigidAngularFrictionImport(unittest.TestCase):
    """MJCF ``condim`` stays with MuJoCo: registering the Steel attributes never enables angular rows."""

    def test_mjcf_condim_does_not_enable_angular_rows(self):
        """Leave the registered ``steel:condim`` at 3 for an MJCF condim of 6, direct or inherited."""
        for inherited in (False, True):
            with self.subTest(inherited=inherited):
                default = '<default><geom condim="6"/></default>' if inherited else ""
                geom = "" if inherited else ' condim="6"'
                mjcf = (
                    f"<mujoco>{default}<worldbody>"
                    '<geom type="plane" size="1 1 0.1"/>'
                    f'<body pos="0 0 0.1"><freejoint/><geom type="sphere" size="0.03"{geom}/></body>'
                    "</worldbody></mujoco>"
                )
                builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                SolverSteel.register_custom_attributes(builder)
                builder.add_mjcf(mjcf)
                model = builder.finalize(device="cpu")
                np.testing.assert_array_equal(model.steel.condim.numpy(), 3)

    def test_usd_reads_the_authored_steel_condim(self):
        """Read ``newton:steel:condim`` from a USD shape and leave unauthored shapes at 3."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/physicsScene")
        for name, authored in (("ball", 6), ("plain", None)):
            xform = UsdGeom.Xform.Define(stage, f"/{name}")
            UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
            sphere = UsdGeom.Sphere.Define(stage, f"/{name}/geom")
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            if authored is not None:
                sphere.GetPrim().CreateAttribute("newton:steel:condim", Sdf.ValueTypeNames.Int).Set(authored)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverSteel.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize(device="cpu")
        labels = list(model.shape_label)
        condim = model.steel.condim.numpy()
        self.assertEqual(int(condim[next(i for i, label in enumerate(labels) if label.startswith("/ball"))]), 6)
        self.assertEqual(int(condim[next(i for i, label in enumerate(labels) if label.startswith("/plain"))]), 3)


if __name__ == "__main__":
    unittest.main()
