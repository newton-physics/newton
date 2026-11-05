# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify shared constraint recovery and implicit material compliance."""

import unittest
from itertools import product

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino.config import ConstrainedDynamicsConfig, ConstraintStabilizationConfig
from newton.solvers import SolverKamino


class TestKaminoConstraintCompliance(unittest.TestCase):
    """Compare small physical systems with analytical implicit updates."""

    @classmethod
    def setUpClass(cls):
        """Initialize the available execution device once for these fixtures."""
        cls.device = wp.get_device()

    def _sphere(
        self,
        *,
        stiffness=500.0,
        damping=20.0,
        gap=0.0,
        gravity=-10.0,
        velocity=0.0,
        restitution=0.0,
        register_attributes=False,
    ):
        """Build a unit-mass sphere and ground with controlled contact material and initial state."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
        if register_attributes:
            SolverKamino.register_custom_attributes(builder)
        shape = newton.ModelBuilder.ShapeConfig(
            density=0.0, margin=0.0, gap=0.05, mu=0.0, ke=stiffness, kd=damping, restitution=restitution
        )
        body = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(0.004, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.004),
            lock_inertia=True,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.1 + gap), wp.quat_identity()),
        )
        builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, velocity, 0.0, 0.0, 0.0)
        builder.add_shape_sphere(body, radius=0.1, cfg=shape)
        builder.add_ground_plane(cfg=shape)
        return builder.finalize(device=self.device)

    def _solver(
        self,
        model,
        constraints,
        *,
        backend="dvi",
        sparse=False,
        preconditioning=False,
        unilateral_solver="pgs",
        linear_solver=None,
        speculative=False,
        internal=False,
        collect_solver_info=False,
        compute_solution_metrics=True,
        schur=None,
        integrator="euler",
    ):
        """Construct a dual solver with tolerances suitable for analytical response checks."""
        config = SolverKamino.Config(
            dynamics_solver=backend,
            integrator=integrator,
            sparse_jacobian=sparse,
            sparse_dynamics=sparse,
            constraints=constraints,
            dynamics=ConstrainedDynamicsConfig(
                preconditioning=preconditioning,
                linear_solver_type=linear_solver or ("CR" if sparse else "LLTB"),
                cull_speculative_contacts=not speculative,
            ),
            compute_solution_metrics=compute_solution_metrics,
            collect_solver_info=collect_solver_info,
            use_collision_detector=internal,
        )
        config.dvi.unilateral_solver = unilateral_solver
        config.dvi.use_schur_complement = schur
        # Analytical Coulomb checks need converged nonlinear corrections.
        config.dvi.apgd.max_nonlinear_corrections = 20
        config.dvi.apgd.tolerance = 1.0e-6
        config.padmm.max_iterations = 300
        config.padmm.primal_tolerance = 1.0e-7
        config.padmm.dual_tolerance = 1.0e-7
        config.padmm.compl_tolerance = 1.0e-7
        return SolverKamino(model, config=config)

    def _step(self, model, solver, *, dt=0.01, state_in=None, contacts=None):
        """Advance one step and check finite velocities and the constraint residual."""
        state_in = model.state() if state_in is None else state_in
        state_out = model.state()
        pipeline = newton.CollisionPipeline(model)
        if contacts is None:
            contacts = pipeline.contacts()
            pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, dt=dt)
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())
        if solver._solver_kamino.metrics is not None:
            metrics = solver._solver_kamino.metrics.data
            self.assertLess(float(np.max(metrics.r_v_plus.numpy())), 1.0e-4)
        return state_out

    def test_contact_implicit_response(self):
        """Match the end-step spring-damper law with Euler and midpoint contact evaluation."""
        dt, stiffness, damping, gap, velocity, gravity = 0.01, 5000.0, 20.0, -0.02, -0.3, -10.0
        model = self._sphere(stiffness=stiffness, damping=damping, gap=gap, velocity=velocity, gravity=gravity)
        for backend, (integrator, internal) in product(
            ("dvi", "padmm"), (("euler", False), ("moreau", True), ("moreau", False))
        ):
            for sparse in (False, True):
                for preconditioning in (False, True) if backend == "padmm" else (False,):
                    with self.subTest(
                        backend=backend,
                        integrator=integrator,
                        internal=internal,
                        sparse=sparse,
                        preconditioning=preconditioning,
                    ):
                        constraints = ConstraintStabilizationConfig(contact_compliance=True, delta=0.0)
                        solver = self._solver(
                            model,
                            constraints,
                            backend=backend,
                            sparse=sparse,
                            preconditioning=preconditioning,
                            integrator=integrator,
                            internal=internal,
                        )
                        state = self._step(model, solver, dt=dt)
                        position_dt = dt * (0.5 if integrator == "moreau" and internal else 1.0)
                        evaluated_gap = gap + (dt - position_dt) * velocity
                        expected = (velocity + dt * gravity - dt * stiffness * evaluated_gap) / (
                            1.0 + dt * (stiffness * position_dt + damping)
                        )
                        actual = float(state.body_qd.numpy()[0, 2])
                        final_gap = float(state.body_q.numpy()[0, 2]) - 0.1
                        self.assertAlmostEqual(actual, expected, delta=2.0e-5)
                        self.assertAlmostEqual(final_gap, evaluated_gap + position_dt * actual, delta=2.0e-7)
                        self.assertAlmostEqual(
                            actual - velocity - dt * gravity,
                            -dt * (stiffness * final_gap + damping * actual),
                            delta=2.0e-5,
                        )

    def test_contact_recovery_override_and_cap(self):
        """Use the same override in compliance and bias, and cap only positional recovery."""
        stiffness, gap, velocity = 500.0, -0.02, -0.3
        model = self._sphere(stiffness=stiffness, damping=20.0, gap=gap, gravity=0.0, velocity=velocity)
        for integrator, (dt, recovery_time, speed) in product(
            ("euler", "moreau"), ((0.01, 0.1, float("inf")), (0.005, 0.0, 0.1), (0.01, 0.04, 0.0))
        ):
            with self.subTest(integrator=integrator, dt=dt, recovery_time=recovery_time, speed=speed):
                constraints = ConstraintStabilizationConfig(
                    contact_compliance=True,
                    contact_recovery_time=recovery_time,
                    contact_recovery_speed=speed,
                    delta=0.0,
                )
                solver = self._solver(
                    model, constraints, sparse=True, integrator=integrator, internal=integrator == "moreau"
                )
                state = self._step(model, solver, dt=dt)
                position_dt = dt * (0.5 if integrator == "moreau" else 1.0)
                evaluated_gap = gap + (dt - position_dt) * velocity
                compliance = 1.0 / (stiffness * dt * (position_dt + recovery_time))
                bias = max(evaluated_gap / (position_dt + recovery_time), -speed)
                impulse = -(velocity + bias) / (1.0 + compliance)
                self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), velocity + impulse, delta=2.0e-5)
                problem = solver._solver_kamino.problem_fd
                active_compliance = problem.data.compliance.numpy()[:3]
                np.testing.assert_allclose(active_compliance, [0.0, 0.0, compliance], rtol=2.0e-6)

    def test_contact_static_deflection_and_override(self):
        """Support weight and sliding friction with per-contact spring gains ahead of shape defaults."""
        model = self._sphere(stiffness=9000.0, damping=300.0, gap=-0.02)
        model.shape_material_mu.fill_(0.5)
        state_in = model.state()
        state_in.body_qd.assign([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        for name, value in (("rigid_contact_stiffness", 500.0), ("rigid_contact_damping", 20.0)):
            setattr(contacts, name, wp.full(contacts.rigid_contact_max, value, dtype=wp.float32, device=self.device))
        variants = [(backend, sparse, "pgs") for backend, sparse in product(("dvi", "padmm"), (False, True))]
        if self.device.is_cuda:
            variants.extend(("dvi", sparse, "apgd") for sparse in (False, True))
        for backend, sparse, subsolver in variants:
            with self.subTest(backend=backend, sparse=sparse, subsolver=subsolver):
                solver = self._solver(
                    model,
                    ConstraintStabilizationConfig(contact_compliance=True, delta=0.0),
                    backend=backend,
                    sparse=sparse,
                    unilateral_solver=subsolver,
                )
                state = self._step(model, solver, state_in=state_in, contacts=contacts)
                self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), 0.0, delta=2.0e-5)
                # The spring's full normal impulse supports Coulomb friction.
                self.assertAlmostEqual(float(state.body_qd.numpy()[0, 0]), 1.0 - 0.5 * 10.0 * 0.01, delta=2.0e-5)

    def test_bilateral_recovery_caps(self):
        """Apply one recovery-speed cap to both linear and angular bilateral rows."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        joint = builder.add_joint_fixed(-1, body)
        builder.add_articulation([joint])
        model = builder.finalize(device=self.device)
        for backend, sparse in (("dvi", False), ("dvi", True), ("padmm", False)):
            with self.subTest(backend=backend, sparse=sparse):
                constraints = ConstraintStabilizationConfig(
                    joint_recovery_time=0.02,
                    joint_recovery_speed=0.1,
                )
                solver = self._solver(model, constraints, backend=backend, sparse=sparse)
                state_in = model.state()
                state_in.body_q.assign(
                    [wp.transform(wp.vec3(0.03, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.03))]
                )
                state = self._step(model, solver, state_in=state_in)
                np.testing.assert_allclose(state.body_qd.numpy()[0], [-0.1, 0.0, 0.0, -0.1, 0.0, 0.0], atol=2.0e-5)
                self.assertIsNone(solver._solver_kamino.problem_fd.data.compliance)

    def test_limit_implicit_response(self):
        """Apply one recovery-speed cap to both linear and angular compliant limits."""
        for angular in (False, True):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
            add_joint = builder.add_joint_revolute if angular else builder.add_joint_prismatic
            joint = add_joint(
                -1, body, axis=newton.Axis.X, limit_lower=0.0, limit_upper=1.0, limit_ke=500.0, limit_kd=20.0
            )
            builder.add_articulation([joint])
            model = builder.finalize(device=self.device)
            model.joint_q.assign([-0.02])
            for backend, sparse, speed, integrator in product(
                ("dvi", "padmm"), (False, True), (0.0, 0.1), ("euler", "moreau")
            ):
                with self.subTest(angular=angular, backend=backend, sparse=sparse, speed=speed, integrator=integrator):
                    constraints = ConstraintStabilizationConfig(
                        limit_compliance=True,
                        limit_recovery_speed=speed,
                    )
                    solver = self._solver(
                        model,
                        constraints,
                        backend=backend,
                        sparse=sparse,
                        integrator=integrator,
                        internal=integrator == "moreau",
                    )
                    state_in = model.state()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
                    state = self._step(model, solver, state_in=state_in)
                    position_dt = 0.005 if integrator == "moreau" else 0.01
                    expected = speed / (1.0 + 1.0 / (0.01 * (500.0 * position_dt + 20.0)))
                    self.assertAlmostEqual(float(state.body_qd.numpy()[0, 3 if angular else 0]), expected, delta=2.0e-5)

    def test_configuration_validation(self):
        """Reject invalid times/caps and unsupported backend settings without changing legacy defaults."""
        for field in ("joint_recovery_time", "limit_recovery_time", "contact_recovery_time"):
            for value in (-1.0, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ConstraintStabilizationConfig(**{field: value})
        for field in (
            "joint_recovery_speed",
            "limit_recovery_speed",
            "contact_recovery_speed",
        ):
            for value in (-1.0, float("nan")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    ConstraintStabilizationConfig(**{field: value})
        with self.assertRaisesRegex(ValueError, "PADMM or DVI"):
            SolverKamino.Config(
                dynamics_solver="lox", constraints=ConstraintStabilizationConfig(joint_recovery_time=0.01)
            )
        for backend in ("dvi", "padmm"):
            config = SolverKamino.Config(dynamics_solver=backend)
            config.lox.contact_compliance = True
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "constraints.contact_compliance"):
                config.validate()

    def test_speculative_gap_and_restitution(self):
        """Keep gap closing and hard-contact restitution independent of the positional speed cap."""
        constraints = ConstraintStabilizationConfig(contact_recovery_time=0.1, contact_recovery_speed=0.0, delta=0.0)
        model = self._sphere(gap=0.005, gravity=0.0, velocity=-0.2)
        solver = self._solver(model, constraints, speculative=True)
        state = self._step(model, solver)
        # The body can close the gap without an impulse; damping or capping the
        # positive gap would stop it prematurely.
        self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), -0.2, delta=1.0e-5)
        for compliant in (False, True):
            with self.subTest(compliant=compliant):
                model = self._sphere(gravity=0.0, velocity=-1.0, restitution=0.5)
                constraints.contact_compliance = compliant
                solver = self._solver(model, constraints)
                state = self._step(model, solver)
                if compliant:
                    expected = -1.0 / (1.0 + 500.0 * 0.01**2 + 500.0 * 0.1 * 0.01)
                else:
                    expected = 0.5
                self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)

    def test_material_refresh_and_contact_removal(self):
        """Refresh material compliance and clear retired contact rows in reused solver storage."""
        model = self._sphere()
        constraints = ConstraintStabilizationConfig(contact_compliance=True, delta=0.0)
        for backend, linear_solver in (("dvi", "CR"), ("padmm", "CR"), ("padmm", "CRF")):
            if linear_solver == "CRF" and not self.device.is_cuda:
                continue
            with self.subTest(backend=backend):
                model.shape_material_ke.fill_(500.0)
                solver = self._solver(
                    model,
                    constraints,
                    backend=backend,
                    sparse=True,
                    linear_solver=linear_solver,
                    preconditioning=backend == "padmm",
                    collect_solver_info=linear_solver != "CRF",
                    compute_solution_metrics=linear_solver != "CRF",
                )
                self._step(model, solver)
                model.shape_material_ke.fill_(1000.0)
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                state = self._step(model, solver)
                expected = -0.1 / (1.0 + 1000.0 * 0.01**2 + 20.0 * 0.01)
                self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)
                state_in = model.state()
                state_in.body_q.assign([wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())])
                state = self._step(model, solver, state_in=state_in)
                self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), -0.1, delta=2.0e-5)
                np.testing.assert_array_equal(solver._solver_kamino.problem_fd.data.compliance.numpy(), 0.0)

    def test_sparse_pgs_compliance_refresh(self):
        """Refresh contact-only sparse PGS compliance eagerly and on CUDA graph replay."""
        dt, gap, velocity, gravity = 0.01, -0.02, -0.3, -10.0
        for captured in (False, True) if self.device.is_cuda else (False,):
            with self.subTest(captured=captured):
                model = self._sphere(gap=gap, velocity=velocity, gravity=gravity)
                # Diagnostics must not refresh the operator on behalf of the solve.
                solver = self._solver(
                    model,
                    ConstraintStabilizationConfig(contact_compliance=True, delta=0.0),
                    sparse=True,
                    collect_solver_info=False,
                    compute_solution_metrics=False,
                )
                state_in, state_out, control = model.state(), model.state(), model.control()
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, dt)
                graph = None
                if captured:
                    with wp.ScopedCapture(device=self.device) as capture:
                        solver.step(state_in, state_out, control, contacts, dt)
                    graph = capture.graph

                for stiffness, damping in ((500.0, 20.0), (5000.0, 5.0), (500.0, 20.0)):
                    with self.subTest(stiffness=stiffness, damping=damping):
                        model.shape_material_ke.fill_(stiffness)
                        model.shape_material_kd.fill_(damping)
                        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                        if graph is None:
                            solver.step(state_in, state_out, control, contacts, dt)
                        else:
                            wp.capture_launch(graph)
                        expected = (velocity + dt * gravity - dt * stiffness * gap) / (
                            1.0 + dt * (stiffness * dt + damping)
                        )
                        actual = float(state_out.body_qd.numpy()[0, 2])
                        final_gap = float(state_out.body_q.numpy()[0, 2]) - 0.1
                        self.assertAlmostEqual(actual, expected, delta=2.0e-5)
                        self.assertAlmostEqual(
                            actual - velocity - dt * gravity,
                            -dt * (stiffness * final_gap + damping * actual),
                            delta=2.0e-5,
                        )

    def test_apgd_and_captured_replay(self):
        """Retain compliance through APGD operator products and repeated CUDA graph replay."""
        if not self.device.is_cuda:
            self.skipTest("CUDA conditional graphs are required")
        model = self._sphere()
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                solver = self._solver(
                    model,
                    ConstraintStabilizationConfig(contact_compliance=True, delta=0.0),
                    sparse=sparse,
                    unilateral_solver="apgd",
                )
                state_in, state_out = model.state(), model.state()
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                pipeline.collide(state_in, contacts)
                control = model.control()
                solver.step(state_in, state_out, control, contacts, dt=0.01)
                with wp.ScopedCapture(device=self.device) as capture:
                    solver.step(state_in, state_out, control, contacts, dt=0.01)
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                    self.assertAlmostEqual(float(state_out.body_qd.numpy()[0, 2]), -0.08, delta=2.0e-5)

    def test_internal_collision_materials(self):
        """Resolve shape materials for contacts generated by Kamino's internal pipeline."""
        model = self._sphere()
        solver = self._solver(model, ConstraintStabilizationConfig(contact_compliance=True, delta=0.0), internal=True)
        state = self._step(model, solver)
        self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), -0.08, delta=2.0e-5)

    def test_legacy_model_configuration(self):
        """Preserve legacy model recovery fractions when Python supplies new recovery settings."""
        model = self._sphere(gap=-0.02, gravity=0.0, register_attributes=True)
        for name, value in (("alpha", 0.2), ("beta", 0.3), ("gamma", 0.25)):
            getattr(model.kamino, f"constraints_{name}").fill_(value)
        config = SolverKamino.Config.from_model(
            model,
            dynamics_solver="dvi",
            compute_solution_metrics=True,
            constraints=ConstraintStabilizationConfig(
                joint_recovery_time=0.02, limit_recovery_speed=0.1, contact_recovery_speed=1.0, delta=0.0
            ),
        )
        self.assertAlmostEqual(config.constraints.alpha, 0.2)
        self.assertAlmostEqual(config.constraints.beta, 0.3)
        self.assertAlmostEqual(config.constraints.gamma, 0.25)
        self.assertEqual(config.constraints.joint_recovery_time, 0.02)
        self.assertEqual(config.constraints.limit_recovery_speed, 0.1)
        solver = SolverKamino(model, config=config)
        state = self._step(model, solver)
        expected = 0.25 * 0.02 / 0.01
        self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), expected, delta=2.0e-5)

    def test_heterogeneous_worlds(self):
        """Keep different material laws and an empty world isolated in packed constraint storage."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
        parameters = ((1.0, 500.0, 20.0, 0.1), (1.0, 500.0, 20.0, 1.0), (2.0, 1000.0, 10.0, 0.1))
        for mass, stiffness, damping, height in parameters:
            builder.begin_world()
            shape = newton.ModelBuilder.ShapeConfig(density=0.0, margin=0.0, gap=0.05, mu=0.0, ke=stiffness, kd=damping)
            body = builder.add_body(
                mass=mass,
                inertia=wp.mat33(np.eye(3)),
                lock_inertia=True,
                xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity()),
            )
            builder.add_shape_sphere(body, radius=0.1, cfg=shape)
            builder.add_ground_plane(cfg=shape)
            builder.end_world()
        model = builder.finalize(device=self.device)
        for backend in ("dvi", "padmm"):
            with self.subTest(backend=backend):
                solver = self._solver(
                    model,
                    ConstraintStabilizationConfig(contact_compliance=True, delta=0.0),
                    backend=backend,
                    sparse=True,
                    preconditioning=backend == "padmm",
                )
                state = self._step(model, solver)
                np.testing.assert_allclose(state.body_qd.numpy()[:, 2], [-0.08, -0.1, -0.2 / 2.2], atol=2.0e-5)

    def test_bilateral_recovery_time(self):
        """Recover hard joint drift with a time scale independent of the substep size."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        builder.add_articulation([builder.add_joint_fixed(-1, body)])
        model = builder.finalize(device=self.device)
        solver = self._solver(model, ConstraintStabilizationConfig(joint_recovery_time=0.02))
        for dt in (0.01, 0.005):
            with self.subTest(dt=dt):
                state_in = model.state()
                state_in.body_q.assign([wp.transform(wp.vec3(0.003, 0.0, 0.0), wp.quat_identity())])
                state = self._step(model, solver, state_in=state_in, dt=dt)
                self.assertAlmostEqual(float(state.body_qd.numpy()[0, 0]), -0.003 / (dt + 0.02), delta=2.0e-5)

    def test_legacy_recovery_equivalence(self):
        """Match legacy fractions with recovery times for joints, limits, and contacts."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        fixed = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        builder.add_articulation([builder.add_joint_fixed(-1, fixed)])
        slider = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        builder.add_articulation(
            [builder.add_joint_prismatic(-1, slider, axis=newton.Axis.X, limit_lower=0.0, limit_upper=1.0)]
        )
        sphere = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            lock_inertia=True,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()),
        )
        shape = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, margin=0.0, gap=0.05)
        builder.add_shape_sphere(sphere, radius=0.1, cfg=shape)
        builder.add_ground_plane(cfg=shape)
        model = builder.finalize(device=self.device)
        model.joint_q.assign([-0.02])
        state_in = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        poses = state_in.body_q.numpy()
        poses[fixed, 0] = 0.003
        state_in.body_q.assign(poses)

        for backend, sparse in (("dvi", False), ("padmm", True)):
            for dt in (0.005, 0.01):
                for fractions in ((0.0, 0.1, 1.0), (0.1, 1.0, 0.0), (1.0, 0.0, 0.1)):
                    with self.subTest(backend=backend, dt=dt, fractions=fractions):
                        legacy = ConstraintStabilizationConfig(
                            alpha=fractions[0], beta=fractions[1], gamma=fractions[2], delta=0.0
                        )
                        # Zero legacy fields ensure an omitted override cannot silently pass.
                        migrated = ConstraintStabilizationConfig(alpha=0.0, beta=0.0, gamma=0.0, delta=0.0)
                        for group, fraction in zip(("joint", "limit", "contact"), fractions, strict=True):
                            time = dt * (1.0 / fraction - 1.0) if fraction else 0.0
                            setattr(migrated, f"{group}_recovery_time", time)
                            setattr(migrated, f"{group}_recovery_speed", float("inf") if fraction else 0.0)
                        results = []
                        for constraints in (legacy, migrated):
                            solver = self._solver(model, constraints, backend=backend, sparse=sparse)
                            state = self._step(model, solver, state_in=state_in, dt=dt)
                            results.append(state.body_qd.numpy())
                        expected = np.zeros((3, 6))
                        expected[fixed, 0] = -0.003 * fractions[0] / dt
                        expected[slider, 0] = 0.02 * fractions[1] / dt
                        expected[sphere, 2] = 0.02 * fractions[2] / dt
                        np.testing.assert_allclose(results[0], expected, atol=2.0e-5)
                        np.testing.assert_allclose(results[1], results[0], atol=2.0e-5)

    def test_upper_limit_override_and_schur(self):
        """Preserve compliant upper-limit response when eliminating coupled bilateral rows."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), com=wp.vec3(0.0, 0.3, 0.0), lock_inertia=True)
        builder.add_articulation(
            [
                builder.add_joint_prismatic(
                    -1,
                    body,
                    axis=newton.Axis.X,
                    limit_lower=0.0,
                    limit_upper=1.0,
                    limit_ke=500.0,
                    limit_kd=20.0,
                )
            ]
        )
        model = builder.finalize(device=self.device)
        model.joint_q.assign([1.02])
        model.joint_qd.assign([0.3])
        for sparse, subsolver, integrator in product((False, True), ("pgs", "apgd"), ("euler", "moreau")):
            if subsolver == "apgd" and not self.device.is_cuda:
                continue
            with self.subTest(sparse=sparse, subsolver=subsolver, integrator=integrator):
                solver = self._solver(
                    model,
                    ConstraintStabilizationConfig(limit_compliance=True, limit_recovery_time=0.1),
                    sparse=sparse,
                    unilateral_solver=subsolver,
                    schur=True,
                    integrator=integrator,
                    internal=integrator == "moreau",
                )
                state_in = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
                state = self._step(model, solver, state_in=state_in)
                position_dt = 0.005 if integrator == "moreau" else 0.01
                evaluated_violation = 0.02 + (0.01 - position_dt) * 0.3
                expected = (0.3 - 0.01 * 500.0 * evaluated_violation) / (
                    1.0 + 500.0 * 0.01 * position_dt + 500.0 * 0.1 * 0.01
                )
                self.assertAlmostEqual(float(state.body_qd.numpy()[0, 0]), expected, delta=2.0e-5)
                np.testing.assert_allclose(state.body_qd.numpy()[0, 1:], 0.0, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
