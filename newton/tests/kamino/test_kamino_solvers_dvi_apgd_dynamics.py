# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Acceptance workloads for DVI APGD: DR Legs and contact stacks."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.geometry.aggregation import ContactAggregation
from newton.solvers import SolverKamino
from newton.tests.kamino.test_kamino_solver_kamino_joint_friction import (
    _run_hold_and_breakaway_test,
    _run_spin_down_test,
)
from newton.tests.kamino.test_kamino_solvers_dvi import _build_five_box_stack
from newton.tests.kamino.test_kamino_solvers_dvi_apgd import _devices
from newton.tests.kamino.utils.solver_configs import make_dvi_dense_config, make_dvi_sparse_config
from newton.viewer import ViewerNull


class TestDVIAPGDDynamics(unittest.TestCase):
    """Exercise the actual integration, warmstart, and captured sparse solver paths."""

    def test_joint_friction_spin_down(self):
        """Reuse analytical joint-friction deceleration and stopped-pose checks for APGD."""
        for device in _devices():
            for factory in (make_dvi_dense_config, make_dvi_sparse_config):
                with self.subTest(device=device, config=factory.__name__), wp.ScopedDevice(device):
                    config = factory()
                    config.dvi.unilateral_solver = "apgd"
                    _run_spin_down_test(self, factory.__name__, config)

    def test_joint_friction_hold_and_breakaway(self):
        """Reuse static holding and saturated friction-torque checks for APGD."""
        for device in _devices():
            for factory in (make_dvi_dense_config, make_dvi_sparse_config):
                with self.subTest(device=device, config=factory.__name__), wp.ScopedDevice(device):
                    config = factory()
                    config.dvi.unilateral_solver = "apgd"
                    _run_hold_and_breakaway_test(self, factory.__name__, config)

    def test_contact_stack_rollout(self):
        """Keep a five-box stack supported through dense and sparse simulation steps."""
        for device in _devices():
            for sparse in (False, True):
                with self.subTest(device=device, sparse=sparse), wp.ScopedDevice(device):
                    model = _build_five_box_stack().finalize(device=device)
                    config = SolverKamino.Config(
                        dynamics_solver="dvi",
                        use_collision_detector=True,
                        sparse_jacobian=sparse,
                        sparse_dynamics=sparse,
                    )
                    config.dvi.unilateral_solver = "apgd"
                    config.dvi.apgd.tolerance = 1e-6
                    solver = SolverKamino(model, config=config)
                    state_0, state_1 = model.state(), model.state()
                    initial = state_0.body_q.numpy()[:, :3].copy()
                    dt = 1e-3

                    def step_pair(solver=solver, state_0=state_0, state_1=state_1, dt=dt):
                        """Advance both state buffers for a reusable capture."""
                        solver.step(state_0, state_1, control=None, contacts=None, dt=dt)
                        solver.step(state_1, state_0, control=None, contacts=None, dt=dt)

                    step_pair()
                    if wp.get_device(device).is_cuda:
                        with wp.ScopedCapture(device=device) as capture:
                            step_pair()
                        for _ in range(49):
                            wp.capture_launch(capture.graph)
                    else:
                        for _ in range(49):
                            step_pair()
                    final = state_0.body_q.numpy()[:, :3]
                    self.assertTrue(np.all(np.isfinite(final)))
                    self.assertLess(float(np.max(np.abs(final - initial))), 1e-3)
                    self.assertLess(float(np.max(np.abs(state_0.body_qd.numpy()))), 0.02)
                    info = solver._solver_kamino.solver_fd.data.status.numpy()[0]
                    self.assertEqual(int(info["apgd_line_search_failed"]), 0)
                    self.assertLess(float(info["r_d"]), 1e-4)
                    contacts = solver._contacts_kamino
                    aggregation = ContactAggregation(model=solver._model_kamino, contacts=contacts)
                    aggregation.compute()
                    force = aggregation.body_net_force.numpy()[0].sum(axis=0)
                    weight = float(model.body_mass.numpy().sum() * 9.81)
                    self.assertAlmostEqual(float(force[2] / weight), 1.0, delta=0.02)

    def test_dr_legs_support_and_reset(self):
        """Support DR Legs after impact and remain stable after a tipped-pose reset."""
        if not wp.is_cuda_available():
            self.skipTest("DR Legs acceptance exercises the CUDA graph path")
        from newton.examples.kamino.example_kamino_robot_dr_legs import Example  # noqa: PLC0415

        with wp.ScopedDevice("cuda:0"):
            args = SimpleNamespace(
                world_count=1,
                use_kamino_contacts=True,
                dynamics_solver="dvi",
                unilateral_solver="apgd",
                use_schur_complement=True,
                # Match the upstream support test; bounded rows have separate tests.
                joint_effort_limit=math.inf,
            )
            config_from_model = SolverKamino.Config.from_model

            def make_accuracy_config(*args, **kwargs):
                """Set the DR Legs residual budget before solver allocation and capture."""
                config = config_from_model(*args, **kwargs)
                config.dvi.apgd.max_nonlinear_corrections = 20
                return config

            with patch.object(SolverKamino.Config, "from_model", side_effect=make_accuracy_config):
                example = Example(ViewerNull(num_frames=1), args)
            base_z = []
            for _ in range(180):
                example.step()
                q = example.state_0.body_q.numpy()
                v = example.state_0.body_qd.numpy()
                self.assertTrue(np.all(np.isfinite(q)))
                self.assertTrue(np.all(np.isfinite(v)))
                self.assertLess(float(np.max(np.abs(v))), 100.0)
                base_z.append(float(q[0, 2]))
            contacts = example.solver._contacts_kamino
            aggregation = ContactAggregation(model=example.solver._model_kamino, contacts=contacts)
            aggregation.compute()
            force = aggregation.body_net_force.numpy()[0].sum(axis=0)
            weight = float(example.model.body_mass.numpy().sum() * 9.81)
            self.assertAlmostEqual(float(force[2] / weight), 1.0, delta=0.05)
            z = np.asarray(base_z[60:])
            t = np.arange(len(z))
            oscillation = z - np.polyval(np.polyfit(t, z, 1), t)
            self.assertLess(float(np.ptp(oscillation)), 0.001)
            info = example.solver._solver_kamino.solver_fd.data.status.numpy()[0]
            self.assertEqual(int(info["apgd_line_search_failed"]), 0)
            self.assertLess(float(info["apgd_residual"]), 1.1e-5)
            self.assertLess(float(info["r_b"]), 0.002)

            tip = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.pi * 0.5))
            example.base_q.assign([wp.transformf((0.0, 0.0, 0.25), tip)])
            reset = SolverKamino.ResetConfig(base_pose=SolverKamino.ResetConfig.FromBaseQ(example.base_q))
            example.solver.reset(state=example.state_0, config=reset)
            example.solver.reset(state=example.state_1, config=reset)
            example.capture()
            start = example.state_0.body_q.numpy()[0, :2].copy()
            penetration = []
            settled_xy = []
            for step in range(400):
                example.step()
                count = int(contacts.world_active_contacts.numpy()[0])
                if step >= 40 and count:
                    penetration.append(float(max(0.0, -np.min(contacts.gapfunc.numpy()[:count, 3]))))
                if step >= 200:
                    settled_xy.append(example.state_0.body_q.numpy()[0, :2].copy())
            self.assertTrue(np.all(np.isfinite(example.state_0.body_qd.numpy())))
            self.assertGreater(len(penetration), 0)
            self.assertLess(float(np.percentile(penetration, 95)), 0.0035)
            self.assertLess(float(np.linalg.norm(settled_xy[-1] - start)), 0.008)
            self.assertLess(float(np.linalg.norm(settled_xy[-1] - settled_xy[0])), 2e-4)


if __name__ == "__main__":
    unittest.main()
