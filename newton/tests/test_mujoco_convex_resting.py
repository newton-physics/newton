# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cover resting convex cylinders with Newton contacts and MuJoCo Warp."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco.constants import SOLREF_MODE_RAW
from newton.solvers import SolverMuJoCo


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestMuJoCoConvexResting(unittest.TestCase):
    def test_convex_cylinder_settles(self):
        """Settle small convex cylinders without sustained drift or energy growth."""
        try:
            SolverMuJoCo.import_mujoco()
        except ImportError as error:
            self.skipTest(str(error))

        radius, half_height, mass = 0.00725, 0.024375, 0.05
        inertia = np.diag([mass * (3 * radius**2 + 4 * half_height**2) / 12] * 2 + [mass * radius**2 / 2]).astype(
            np.float32
        )
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        mesh = newton.Mesh.create_cylinder(radius, half_height, up_axis=newton.Axis.Z, segments=64)
        cfg = builder.ShapeConfig(density=0.0, mu=0.55, ke=40000, kd=400, gap=0.0001, margin=0.0)
        phases = np.linspace(0.0, 0.1, 20)
        for phase in phases:
            builder.begin_world()
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.585, 0.0, 0.215)),
                hx=0.3675,
                hy=0.61,
                hz=0.015,
                cfg=cfg,
            )
            rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -wp.pi / 2)
            rotation *= wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(phase))
            body = builder.add_link(
                xform=wp.transform(wp.vec3(0.585, -0.22, 0.239), rotation),
                mass=mass,
                inertia=wp.mat33(inertia),
                lock_inertia=True,
            )
            joint = builder.add_joint_free(body)
            builder.add_articulation([joint])
            builder.add_shape_convex_hull(body, mesh=mesh, cfg=cfg)
            builder.end_world()
        model = builder.finalize("cuda:0")
        solref = model.mujoco.solref.numpy()
        modes = model.mujoco.solref_mode.numpy()
        solimp = model.mujoco.geom_solimp.numpy()
        solref[1::2] = [0.003, 1.0]
        modes[1::2] = SOLREF_MODE_RAW
        solimp[1::2] = [0.99, 0.99, 0.0001, 0.5, 2.0]
        model.mujoco.solref.assign(solref)
        model.mujoco.solref_mode.assign(modes)
        model.mujoco.geom_solimp.assign(solimp)
        solver = SolverMuJoCo(
            model,
            use_mujoco_contacts=False,
            integrator="implicitfast",
            cone="elliptic",
            iterations=50,
            ls_iterations=50,
            nconmax=32,
            njmax=128,
        )
        self.assertFalse(solver.mjw_model.opt.run_collision_detection)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state, next_state = model.state(), model.state()
        control = model.control()
        dt, substeps = 1.0 / 1800, 10

        def advance():
            nonlocal state, next_state
            for _ in range(substeps):
                state.clear_forces()
                pipeline.collide(state, contacts)
                solver.step(state, next_state, control, contacts, dt)
                state, next_state = next_state, state

        advance()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        newton.eval_fk(model, model.joint_q, model.joint_qd, next_state)
        solver.reset(state)
        with wp.ScopedCapture(device="cuda:0") as capture:
            advance()
        for _ in range(360):
            wp.capture_launch(capture.graph)

        poses, velocities = state.body_q.numpy(), state.body_qd.numpy()
        self.assertTrue(np.isfinite(poses).all())
        self.assertTrue(np.isfinite(velocities).all())
        self.assertLess(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
        self.assertLess(int(solver.mjw_data.nacon.numpy()[0]), solver.get_max_contact_count())
        for phase, pose, velocity in zip(phases, poses, velocities, strict=True):
            with self.subTest(phase=phase):
                self.assertLess(np.linalg.norm(pose[:2] - [0.585, -0.22]), 0.001)
                self.assertLess(np.linalg.norm(velocity[:3]), 0.001)
                self.assertAlmostEqual(float(pose[2]), 0.23 + radius, delta=0.0001)
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
                omega = rotation.T @ velocity[3:]
                energy = 0.5 * mass * np.dot(velocity[:3], velocity[:3])
                energy += 0.5 * omega @ inertia @ omega
                energy += mass * 9.81 * (pose[2] - 0.239)
                self.assertLess(energy, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
