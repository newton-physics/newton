# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for separated normal constraints in MuJoCo Warp."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestMuJoCoSpeculativeContacts(unittest.TestCase):
    def setUp(self):
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as error:
            self.skipTest(str(error))

    def test_fast_thin_object(self):
        """Prevent tunneling through a thin support at 60 Hz."""
        builder = newton.ModelBuilder()
        cfg = builder.default_shape_cfg.copy()
        cfg.margin = 0.0
        cfg.gap = 0.0
        cfg.ke = 3600.0
        cfg.kd = 60.0
        builder.add_shape_box(-1, hx=0.2, hy=0.2, hz=0.005, cfg=cfg)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.156)), mass=0.05)
        builder.add_shape_box(body, hx=0.1, hy=0.01, hz=0.001, cfg=cfg)
        builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, -3.0, 0.0, 0.0, 0.0)
        model = builder.finalize(device="cuda:0")
        control = model.control()
        pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.1)
        contacts = pipeline.contacts()
        for enabled, integrator in ((False, "implicitfast"), (True, "implicitfast"), (True, "euler")):
            with self.subTest(enabled=enabled, integrator=integrator):
                state_in, state_out = model.state(), model.state()
                solver = SolverMuJoCo(
                    model, use_mujoco_contacts=False, use_speculative_contacts=enabled, integrator=integrator
                )
                heights = []
                for _ in range(60):
                    state_in.clear_forces()
                    pipeline.collide(state_in, contacts, dt=1.0 / 60.0)
                    solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)
                    state_in, state_out = state_out, state_in
                    heights.append(float(state_in.body_q.numpy()[body, 2]))
                self.assertTrue(np.isfinite(heights).all())
                if enabled:
                    self.assertGreater(min(heights), 0.003)
                else:
                    self.assertLess(heights[-1], -0.01)

    def test_force_limited_drive_and_graph(self):
        """Block a finite drive and restore friction during graph execution."""
        for cone in ("pyramidal", "elliptic"):
            with self.subTest(cone=cone):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                cfg = builder.default_shape_cfg.copy()
                cfg.gap, cfg.margin, cfg.ke, cfg.kd = 0.01, 0.0, 40000.0, 200.0
                body = builder.add_link(mass=1.0)
                joint = builder.add_joint_prismatic(
                    -1,
                    body,
                    axis=(1.0, 0.0, 0.0),
                    target_ke=1000.0,
                    target_kd=50.0,
                    effort_limit=50.0,
                    limit_lower=-1.0,
                    limit_upper=1.0,
                )
                builder.add_articulation([joint])
                builder.add_shape_box(body, hx=0.01, hy=0.02, hz=0.02, cfg=cfg)
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(0.08, 0.0, 0.0)),
                    hx=0.01,
                    hy=0.1,
                    hz=0.1,
                    cfg=cfg,
                )
                model = builder.finalize(device="cuda:0")
                state_in, state_out = model.state(), model.state()
                control = model.control()
                control.joint_target_q.fill_(0.15)
                pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.1)
                contacts = pipeline.contacts()
                solver = SolverMuJoCo(
                    model,
                    use_mujoco_contacts=False,
                    use_speculative_contacts=True,
                    cone=cone,
                )

                def step_pair(pipeline, state_in, state_out, solver, control, contacts):
                    pipeline.collide(state_in, contacts, dt=0.01)
                    for source, destination in ((state_in, state_out), (state_out, state_in)):
                        source.clear_forces()
                        # Exercise cached properties with unchanged generation.
                        solver.step(source, destination, control, contacts, 0.005)

                step_pair(pipeline, state_in, state_out, solver, control, contacts)
                with wp.ScopedCapture(device=model.device) as capture:
                    step_pair(pipeline, state_in, state_out, solver, control, contacts)
                for _ in range(100):
                    wp.capture_launch(capture.graph)
                position = float(state_in.body_q.numpy()[body, 0])
                self.assertGreater(position, 0.059)
                self.assertLess(position, 0.061)
                self.assertLess(position, 0.15 - 0.08)
                self.assertLessEqual(float(np.abs(solver.mjw_data.qfrc_actuator.numpy()).max()), 50.001)
                count = int(solver.mjw_data.nacon.numpy()[0])
                self.assertGreater(count, 0)
                self.assertTrue((solver.mjw_data.contact.dim.numpy()[:count] == 3).all())
                control.joint_target_q.fill_(0.0)
                for _ in range(50):
                    wp.capture_launch(capture.graph)
                self.assertLess(abs(float(state_in.body_q.numpy()[body, 0])), 0.001)

    def test_separated_contacts_have_no_friction(self):
        """Preserve free separation and tangential motion in independent worlds."""
        template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        cfg = template.default_shape_cfg.copy()
        cfg.gap = 0.02
        cfg.margin = 0.0
        cfg.mu = 1.0
        template.add_shape_box(-1, hx=0.2, hy=0.2, hz=0.005, cfg=cfg)
        body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.016)))
        template.add_shape_box(body, hx=0.01, hy=0.01, hz=0.001, cfg=cfg)
        template.body_qd[body] = wp.spatial_vector(0.1, 0.0, 0.02, 0.0, 0.0, 0.0)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_world(template)
        builder.add_world(template, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0)))
        model = builder.finalize(device="cuda:0")
        state_in, state_out = model.state(), model.state()
        state_in.joint_qd.assign([0.1, 0.0, 0.02, 0.0, 0.0, 0.0] * 2)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        control = model.control()
        pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.1)
        contacts = pipeline.contacts()
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, use_speculative_contacts=True)
        pipeline.collide(state_in, contacts, dt=0.01)
        for _ in range(3):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, contacts, 0.005)
            state_in, state_out = state_out, state_in
        np.testing.assert_allclose(state_in.body_q.numpy()[:, 0], [0.0015, 1.0015], atol=1.0e-6)
        self.assertLess(float(np.abs(solver.mjw_data.qfrc_constraint.numpy()).max()), 1.0e-5)
        count = int(solver.mjw_data.nacon.numpy()[0])
        self.assertGreater(count, 0)
        self.assertTrue((solver.mjw_data.contact.dist.numpy()[:count] > 0.0).all())

    def test_reject_unsupported_backends(self):
        """Reject unsupported contact backends and integration schemes."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
        model = builder.finalize(device="cuda:0")
        for options in ({}, {"use_mujoco_contacts": False, "use_mujoco_cpu": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SolverMuJoCo(model, use_speculative_contacts=True, **options)
        with self.assertRaisesRegex(ValueError, "Euler or implicitfast"):
            SolverMuJoCo(model, use_mujoco_contacts=False, use_speculative_contacts=True, integrator="rk4")
        solver = SolverMuJoCo(model, use_mujoco_contacts=False, use_speculative_contacts=True)
        for dt in (0.0, -0.001, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "finite positive timestep"):
                solver.step(model.state(), model.state(), model.control(), None, dt)

    def test_shared_timestep_across_worlds(self):
        """Match speculative braking with shared and per-world timesteps, including replay."""
        template = newton.ModelBuilder(gravity=wp.vec3(0.0))
        cfg = template.ShapeConfig(gap=0.0, margin=0.0)
        template.add_shape_box(-1, hx=0.2, hy=0.2, hz=0.005, cfg=cfg)
        body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05)))
        template.add_shape_box(body, hx=0.01, hy=0.01, hz=0.001, cfg=cfg)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.add_world(template)
        builder.add_world(template, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0)))
        model = builder.finalize(device="cuda:0")
        dt = 1.0 / 60.0
        for capture_enabled in (False, True):
            results = []
            for timestep_count in (1, 2):
                with self.subTest(capture=capture_enabled, timestep_count=timestep_count):
                    states = (model.state(), model.state())
                    states[0].joint_qd.assign([0.0, 0.0, -3.0, 0.0, 0.0, 0.0] * 2)
                    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
                    solver = SolverMuJoCo(model, use_mujoco_contacts=False, use_speculative_contacts=True)
                    solver.mjw_model.opt.timestep = wp.full(timestep_count, dt, dtype=float, device=model.device)
                    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.1)
                    contacts = pipeline.contacts()
                    control = model.control()

                    def step_pair(
                        *, states=states, pipeline=pipeline, solver=solver, control=control, contacts=contacts
                    ):
                        for source, destination in (states, states[::-1]):
                            source.clear_forces()
                            pipeline.collide(source, contacts, dt=dt)
                            solver.step(source, destination, control, contacts, dt)

                    # Warm up lazy collision storage and solver kernels.
                    step_pair()
                    if capture_enabled:
                        with wp.ScopedCapture(device=model.device) as capture:
                            step_pair()
                        wp.capture_launch(capture.graph)
                    else:
                        step_pair()
                    position = states[0].body_q.numpy()[:, 2]
                    velocity = states[0].body_qd.numpy()[:, 2]
                    results.append((position, velocity))
                    self.assertTrue(np.isfinite(position).all())
                    self.assertTrue(np.isfinite(velocity).all())
                    np.testing.assert_allclose(position[0], position[1], atol=1.0e-6)
                    np.testing.assert_allclose(velocity[0], velocity[1], atol=1.0e-6)
                    self.assertGreater(float(position.min()), 0.003)
            np.testing.assert_allclose(results[0], results[1], atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
