# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for particle proxies in the implicit MPM solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverCoupled, SolverImplicitMPM, SolverProxyCoupled, SolverXPBD


class TestImplicitMPMProxyParticles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    @staticmethod
    def _mpm_config() -> SolverImplicitMPM.Config:
        config = SolverImplicitMPM.Config()
        config.voxel_size = 0.1
        config.grid_type = "fixed"
        config.grid_padding = 1
        config.warmstart_mode = "none"
        config.transfer_scheme = "pic"
        config.max_iterations = 1
        return config

    def test_pure_proxy_particle_is_transfer_active_not_material(self):
        builder = newton.ModelBuilder(gravity=0.0)
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05)
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="xpbd", solver=SolverXPBD, particles=[0]),
                SolverCoupled.Entry(
                    name="mpm",
                    solver=SolverImplicitMPM,
                    particles=[1],
                    solver_kwargs={"config": self._mpm_config()},
                ),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[SolverProxyCoupled.Proxy(source="xpbd", destination="mpm", particles=[0])]
            ),
        )

        active = int(newton.ParticleFlags.ACTIVE)
        proxy = int(newton.ParticleFlags.PROXY)
        mpm_model = coupled.get_solver("mpm")._mpm_model

        transfer_flags = mpm_model.particle_flags.numpy()
        material_flags = mpm_model.material_particle_flags.numpy()
        material_volume = mpm_model.material_particle_volume.numpy()

        self.assertNotEqual(transfer_flags[0] & proxy, 0)
        self.assertNotEqual(transfer_flags[0] & active, 0)
        self.assertNotEqual(material_flags[0] & proxy, 0)
        self.assertEqual(material_flags[0] & active, 0)
        self.assertEqual(material_volume[0], 0.0)

        self.assertEqual(transfer_flags[1] & proxy, 0)
        self.assertNotEqual(transfer_flags[1] & active, 0)
        self.assertNotEqual(material_flags[1] & active, 0)
        self.assertGreater(material_volume[1], 0.0)

    def test_deformable_collider_proxy_particle_is_transfer_inactive(self):
        builder = newton.ModelBuilder(gravity=0.0)
        SolverImplicitMPM.register_custom_attributes(builder)
        for pos in ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (1.0, 0.0, 0.0)):
            builder.add_particle(pos=pos, vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.05)
        model = builder.finalize(device="cpu")

        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="xpbd", solver=SolverXPBD, particles=[0, 1, 2]),
                SolverCoupled.Entry(
                    name="mpm",
                    solver=SolverImplicitMPM,
                    particles=[3],
                    solver_kwargs={"config": self._mpm_config()},
                ),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[SolverProxyCoupled.Proxy(source="xpbd", destination="mpm", particles=[0, 1, 2])]
            ),
        )

        points = wp.array([(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)], dtype=wp.vec3, device=model.device)
        indices = wp.array([0, 1, 2], dtype=wp.int32, device=model.device)
        mesh = wp.Mesh(points=points, indices=indices, velocities=wp.zeros(3, dtype=wp.vec3, device=model.device))

        mpm_solver = coupled.get_solver("mpm")
        mpm_solver.setup_collider(
            collider_meshes=[mesh],
            collider_body_ids=[None],
            collider_particle_ids=[[0, 1, 2]],
            model=coupled.get_view("mpm"),
        )

        active = int(newton.ParticleFlags.ACTIVE)
        proxy = int(newton.ParticleFlags.PROXY)
        transfer_flags = mpm_solver._mpm_model.particle_flags.numpy()
        material_flags = mpm_solver._mpm_model.material_particle_flags.numpy()

        for particle_id in (0, 1, 2):
            self.assertNotEqual(transfer_flags[particle_id] & proxy, 0)
            self.assertEqual(transfer_flags[particle_id] & active, 0)
            self.assertNotEqual(material_flags[particle_id] & proxy, 0)
            self.assertEqual(material_flags[particle_id] & active, 0)

        self.assertEqual(transfer_flags[3] & proxy, 0)
        self.assertNotEqual(transfer_flags[3] & active, 0)
        self.assertNotEqual(material_flags[3] & active, 0)

    def test_transfer_proxy_rewinds_and_harvests_without_gravity_feedback(self):
        builder = newton.ModelBuilder(gravity=-9.81)
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.03)
        builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.03)
        model = builder.finalize(device="cpu")
        model.mpm.yield_pressure.fill_(1.0e5)

        config = self._mpm_config()
        config.voxel_size = 0.2
        config.grid_padding = 2
        coupled = SolverProxyCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(name="xpbd", solver=SolverXPBD, particles=[0], solver_kwargs={"iterations": 1}),
                SolverCoupled.Entry(
                    name="mpm",
                    solver=SolverImplicitMPM,
                    particles=[1],
                    solver_kwargs={"config": config},
                    in_place=True,
                ),
            ],
            coupling=SolverProxyCoupled.CouplingProxy(
                proxies=[SolverProxyCoupled.Proxy(source="xpbd", destination="mpm", particles=[0])]
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        coupled.step(state_0, state_1, control=None, contacts=None, dt=1.0 / 60.0)

        rewound_proxy_qd = coupled.get_solver("mpm")._proxy_particle_qd_before.numpy()[0]
        np.testing.assert_allclose(rewound_proxy_qd, np.zeros(3), atol=1.0e-5)

        harvested_force = coupled._proxy_particle_mappings[0].coupling_forces.numpy()[0]
        np.testing.assert_allclose(harvested_force[:2], np.zeros(2), atol=1.0e-5)
        self.assertLess(abs(float(harvested_force[2])), 1.0e-2)


if __name__ == "__main__":
    unittest.main()
