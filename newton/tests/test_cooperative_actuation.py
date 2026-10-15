# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check cooperative Newton updates with mixed actuator groups."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.actuators.effort_mode_implicit import (
    _build_coupled_cooperative_kernel,
    _build_coupled_solve_kernel,
    _compose_clamps,
)
from newton.actuators import ClampingMaxEffort, DrivePD
from newton.tests.test_actuators import _make_implicit_actuator


class TestCooperativeActuation(unittest.TestCase):
    """Cover coupling, mixed groups, inactive outputs and pivot flooring."""

    @unittest.skipUnless(wp.is_cuda_available(), "Tiled effort solve requires CUDA")
    def test_coupled_width_mixed_groups_and_pivot_floor(self):
        """Match generic callback solves with mixed group sizes and permutations."""
        if tuple(int(part) for part in wp.__version__.split(".")[:2]) < (1, 18):
            self.skipTest("Tiled kernels require Warp 1.18")
        device = wp.get_cuda_device()
        rng = np.random.default_rng(31)
        entries = ((ClampingMaxEffort.evaluate_clamp, 0),)
        chain = _compose_clamps(entries)
        key = (DrivePD.evaluate_force, entries)
        baseline_kernel = _build_coupled_solve_kernel(DrivePD.evaluate_force, chain, key)
        for width in (3, 17, 43, 64):
            dofs = width + 6
            factors = rng.normal(size=(2, dofs, dofs)).astype(np.float32) * 0.02
            response = factors @ factors.transpose(0, 2, 1) + 0.02 * np.eye(dofs, dtype=np.float32)
            slots = np.stack([rng.permutation(width), width + rng.permutation(width)]).astype(np.int32)
            local = slots % width + 6
            indices = np.concatenate([np.arange(width) + 6, np.arange(width) + dofs + 6]).astype(np.uint32)
            positions = rng.normal(size=2 * dofs).astype(np.float32) * 0.1
            velocities = rng.normal(size=2 * dofs).astype(np.float32) * 0.3
            target = positions + rng.normal(size=2 * dofs).astype(np.float32) * 0.1
            params = np.tile([100.0, 3.0, 0.2], (2 * width, 1)).astype(np.float32)
            bounds = np.linspace(1, 5, 2 * width, dtype=np.float32).reshape(-1, 1)
            base = [
                wp.array([width, max(1, width // 2)], dtype=int, device=device),
                wp.array([0, 1], dtype=int, device=device),
                wp.array(slots, dtype=int, device=device),
                wp.array(local, dtype=int, device=device),
                wp.array(response, device=device),
                *[wp.array(indices, dtype=wp.uint32, device=device) for _ in range(4)],
                wp.array(positions, device=device),
                wp.array(velocities, device=device),
                wp.array(target, device=device),
                wp.zeros(2 * dofs, device=device),
            ]
            for floor in (1e-08, 2.0):
                with self.subTest(width=width, derivative_floor=floor):
                    outputs = []
                    for tiled in (False, True):
                        scratch = [wp.zeros((2, width), dtype=wp.float64, device=device) for _ in range(3)]
                        scratch.append(wp.zeros((2, width, width), dtype=wp.float64, device=device))
                        out = [wp.full(2 * width, 3.5, device=device) for _ in range(2)]
                        inputs = [
                            *base,
                            None,
                            wp.array(params, device=device),
                            0.004,
                            4,
                            1e-05,
                            1e-05,
                            0.0001,
                            floor,
                            1,
                            wp.array(bounds, device=device),
                            *scratch,
                        ]
                        if tiled:
                            kernel = _build_coupled_cooperative_kernel(DrivePD.evaluate_force, chain, key, width)
                            wp.launch(kernel, dim=128, inputs=inputs, outputs=out, device=device, block_dim=64)
                        else:
                            wp.launch(baseline_kernel, dim=2, inputs=inputs, outputs=out, device=device)
                        outputs.append([array.numpy() for array in out])
                    for expected, actual in zip(*outputs, strict=True):
                        np.testing.assert_allclose(actual, expected, atol=0.0001, rtol=0.0005)

    def test_large_group_public_solve(self):
        """Solve a 65-joint actuator through the public scalar fallback."""
        device = wp.get_device()
        width = 65
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        joints = []
        parent = -1
        for _ in range(width):
            child = builder.add_link(mass=1.0, inertia=wp.mat33(1.0))
            joints.append(builder.add_joint_revolute(parent=parent, child=child, axis=newton.Axis.Z))
            parent = child
        builder.add_articulation(joints)
        model = builder.finalize(device=device)
        actuator, response = _make_implicit_actuator(
            model, device, kp=wp.full(width, 100.0, device=device), kd=wp.full(width, 3.0, device=device)
        )
        response._inv_block.assign(np.eye(width, dtype=np.float32)[None])
        state, control = model.state(), model.control()
        control.joint_target_q.fill_(1.0)
        actuator.step(state, control, dt=0.01)
        np.testing.assert_allclose(control.joint_f.numpy(), 100.0 / 1.04, atol=1e-4, rtol=5e-4)


if __name__ == "__main__":
    unittest.main()
