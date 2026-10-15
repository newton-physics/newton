# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical coverage for the opt-in G1-shaped implicit kernel path."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.actuators import _g1_kernels
from newton._src.sim.articulation import eval_articulation_mass_matrix
from newton.actuators import ClampingMaxEffort, DrivePD, JointSpaceResponse


@unittest.skipUnless(wp.is_cuda_available(), "G1 tiled kernels require CUDA")
class TestG1ImplicitKernels(unittest.TestCase):
    """Exercise the specialized layout absent from the small actuator fixtures."""

    def test_mass_matrix_mask_and_armature(self):
        """Match scalar mass assembly with masks and main's armature semantics."""
        device = wp.get_cuda_device()
        rng = np.random.default_rng(17)
        starts = wp.array([0, 44], dtype=int, device=device)
        ends = wp.array([44], dtype=int, device=device)
        qd_starts = np.arange(45, dtype=np.int32)
        qd_starts[1:] += 5
        qd_starts = wp.array(qd_starts, dtype=int, device=device)
        factors = rng.normal(size=(44, 6, 6)).astype(np.float32) * 0.1
        inertia = factors @ factors.transpose(0, 2, 1) + np.eye(6, dtype=np.float32)
        inertias = wp.array(inertia, dtype=wp.spatial_matrix, device=device)
        jacobian = wp.array(rng.normal(size=(1, 264, 49)).astype(np.float32) * 0.1, device=device)
        children = wp.array(np.arange(44, dtype=np.int32), dtype=int, device=device)
        armature = wp.array(np.linspace(0.01, 0.1, 49, dtype=np.float32), device=device)
        for mask in (None, wp.array([False], dtype=bool, device=device)):
            for diagonal in (None, armature):
                values = [starts, ends, 1, mask, children, qd_starts, inertias, jacobian, diagonal]
                outputs = []
                for enabled in ("0", "1"):
                    H = wp.zeros((1, 49, 49), device=device)
                    with patch.dict(os.environ, {"NEWTON_G1_KERNELS": enabled}):
                        _g1_kernels.launch_mass(
                            eval_articulation_mass_matrix, dim=1, inputs=values, outputs=[H], device=device
                        )
                    outputs.append(H.numpy())
                np.testing.assert_allclose(outputs[1], outputs[0], atol=5e-5, rtol=2e-4)
                if mask is not None:
                    np.testing.assert_array_equal(outputs[1], 0)

    def test_response_and_clamped_pd_cuda_graph(self):
        """Match scalar response and actuator efforts through a captured graph."""
        device = wp.get_cuda_device()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        root = builder.add_link(mass=20.0, inertia=wp.mat33(np.eye(3).astype(np.float32)))
        joints = [builder.add_joint_free(child=root)]
        for index in range(43):
            child = builder.add_link(mass=0.5, inertia=wp.mat33(0.02 * np.eye(3).astype(np.float32)))
            joints.append(builder.add_joint_revolute(parent=root, child=child, axis=newton.Axis.Z, armature=0.01))
            builder.add_actuator(
                DrivePD,
                index=6 + index,
                pos_index=7 + index,
                kp=100.0,
                kd=2.0,
                clamping=[(ClampingMaxEffort, {"max_effort": 1.0})],
            )
        builder.add_articulation(joints)
        model = builder.finalize(device=device)
        state = model.state()
        state.joint_q.assign(model.joint_q)
        state.joint_qd.assign(np.linspace(-0.3, 0.3, 49, dtype=np.float32))
        control = model.control()
        control.joint_target_q.fill_(0.3)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        response = JointSpaceResponse(model)
        actuator = model.actuators[0]
        actuator.set_effort_mode_implicit(response=response)
        snapshots = []
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {"NEWTON_G1_KERNELS": enabled}):
                response.refresh(state)
                control.joint_f.zero_()
                actuator.step(state, control, dt=0.002)
                with wp.ScopedCapture(device=device) as capture:
                    response.refresh(state)
                    control.joint_f.zero_()
                    actuator.step(state, control, dt=0.002)
                wp.capture_launch(capture.graph)
                snapshots.append((response.inverse_blocks.numpy(), control.joint_f.numpy()))
        np.testing.assert_allclose(snapshots[1][0], snapshots[0][0], atol=5e-5, rtol=2e-4)
        np.testing.assert_allclose(snapshots[1][1], snapshots[0][1], atol=1e-4, rtol=5e-4)
        self.assertTrue(np.isfinite(snapshots[1][1]).all())
        self.assertLessEqual(float(np.abs(snapshots[1][1][6:]).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
