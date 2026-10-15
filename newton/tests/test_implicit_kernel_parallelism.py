# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check dimension-driven mass assembly and response factorization."""

import unittest

import numpy as np
import warp as wp

from newton._src.actuators.effort_mode_implicit import (
    _build_coupled_solve_kernel,
    _build_coupled_tiled_kernel,
    _compose_clamps,
)
from newton._src.actuators.joint_space_response import (
    _build_inverse_tiled_kernel,
    _inverse_block_from_mass_matrix_kernel,
)
from newton._src.sim.articulation import eval_articulation_mass_matrix
from newton.actuators import ClampingMaxEffort, DrivePD


@wp.kernel
def _trace_mass(H: wp.array3d[float], loss: wp.array[float]):
    dof = wp.tid()
    wp.atomic_add(loss, 0, H[0, dof, dof])


class TestImplicitKernelParallelism(unittest.TestCase):
    """Cover widths, padding and masks absent from the small robot fixtures."""

    def test_mass_matrix_gradient(self):
        """Preserve the Jacobian gradient of the mass-matrix trace."""
        device = wp.get_device()
        jacobian = np.random.default_rng(5).normal(size=(1, 12, 7)).astype(np.float32) * 0.1
        inertias = np.stack([2.0 * np.eye(6), 3.0 * np.eye(6)]).astype(np.float32)
        J = wp.array(jacobian, device=device, requires_grad=True)
        H = wp.zeros((1, 7, 7), device=device, requires_grad=True)
        loss = wp.zeros(1, device=device, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(
                eval_articulation_mass_matrix,
                dim=(1, 7, 7),
                inputs=[
                    wp.array([0, 2], dtype=int, device=device),
                    wp.array([2], dtype=int, device=device),
                    1,
                    None,
                    wp.array([0, 1], dtype=int, device=device),
                    wp.array([0, 6, 7], dtype=int, device=device),
                    wp.array(inertias, dtype=wp.spatial_matrix, device=device),
                    J,
                    None,
                ],
                outputs=[H],
                device=device,
            )
            wp.launch(_trace_mass, dim=7, inputs=[H], outputs=[loss], device=device)
        tape.backward(loss)
        expected = jacobian.copy()
        expected[:, :6] *= 4.0
        expected[:, 6:] *= 6.0
        np.testing.assert_allclose(J.grad.numpy(), expected, atol=1e-6, rtol=1e-5)

    def test_mass_matrix_width_mask_armature_and_accumulation(self):
        """Match dense contractions without overwriting masked or padded entries."""
        rng = np.random.default_rng(17)
        devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            devices.append(wp.get_cuda_device())
        for device in devices:
            for width in (1, 7, 49, 65):
                short = max(1, width - 2)
                J = rng.normal(size=(2, 18, width)).astype(np.float32) * 0.1
                factors = rng.normal(size=(4, 6, 6)).astype(np.float32) * 0.1
                inertias = factors @ factors.transpose(0, 2, 1) + np.eye(6, dtype=np.float32)
                armature = np.linspace(0.01, 0.1, width + short, dtype=np.float32)
                for mask in (None, np.array([True, False])):
                    for diagonal in (None, armature):
                        with self.subTest(
                            device=str(device), width=width, masked=mask is not None, armature=diagonal is not None
                        ):
                            expected = np.full((2, width, width), 0.2, dtype=np.float64)
                            for art, links, count in ((0, range(1), width), (1, range(1, 4), short)):
                                if mask is not None and not mask[art]:
                                    continue
                                for local, link in enumerate(links):
                                    jacobian = J[art, local * 6 : (local + 1) * 6, :count].astype(np.float64)
                                    expected[art, :count, :count] += jacobian.T @ inertias[link] @ jacobian
                                if diagonal is not None:
                                    begin = 0 if art == 0 else width
                                    expected[art, np.arange(count), np.arange(count)] += diagonal[begin : begin + count]
                            H = wp.full((2, width, width), 0.2, device=device)
                            wp.launch(
                                eval_articulation_mass_matrix,
                                dim=(2, width, width),
                                inputs=[
                                    wp.array([0, 1, 4], dtype=int, device=device),
                                    wp.array([1, 4], dtype=int, device=device),
                                    2,
                                    None if mask is None else wp.array(mask, dtype=bool, device=device),
                                    wp.array([0, 1, 2, 3], dtype=int, device=device),
                                    wp.array([0, width, width, width, width + short], dtype=int, device=device),
                                    wp.array(inertias, dtype=wp.spatial_matrix, device=device),
                                    wp.array(J, device=device),
                                    None if diagonal is None else wp.array(diagonal, device=device),
                                ],
                                outputs=[H],
                                device=device,
                            )
                            np.testing.assert_allclose(H.numpy(), expected, rtol=2e-4, atol=5e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "Tiled factorization requires CUDA")
    def test_inverse_width_padding_and_pivot_floor(self):
        """Match scalar factors and inverses across widths and singular pivots."""
        if tuple(int(part) for part in wp.__version__.split(".")[:2]) < (1, 18):
            self.skipTest("Tiled kernels require Warp 1.18")
        device = wp.get_cuda_device()
        rng = np.random.default_rng(29)
        for width in (1, 7, 49, 65, 128):
            block = max(32, 1 << (width - 1).bit_length())
            if block * (block + 1) * 4 > device.max_shared_memory_per_block:
                continue
            factors = rng.normal(size=(2, width, width)).astype(np.float32) * 0.1
            H = factors @ factors.transpose(0, 2, 1) + 0.2 * np.eye(width, dtype=np.float32)
            for singular in (False, True):
                with self.subTest(width=width, singular=singular):
                    matrix = wp.array(np.zeros_like(H) if singular else H, device=device)
                    counts = wp.array([width, max(0, width - 2)], dtype=int, device=device)
                    baseline = [wp.zeros(H.shape, device=device) for _ in range(2)]
                    actual = [wp.zeros(H.shape, device=device) for _ in range(2)]
                    wp.launch(
                        _inverse_block_from_mass_matrix_kernel,
                        dim=2,
                        inputs=[matrix, counts],
                        outputs=baseline,
                        device=device,
                    )
                    kernel, block_dim = _build_inverse_tiled_kernel(width)
                    wp.launch_tiled(
                        kernel, dim=2, inputs=[matrix, counts], outputs=actual, device=device, block_dim=block_dim
                    )
                    for scalar, tiled in zip(baseline, actual, strict=True):
                        np.testing.assert_allclose(tiled.numpy(), scalar.numpy(), rtol=2e-4, atol=5e-5)

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
            block = max(32, 1 << (width - 1).bit_length())
            if (2 * width + 7) * block * 8 > device.max_shared_memory_per_block:
                continue
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
            for floor in (1e-8, 2.0):
                with self.subTest(width=width, derivative_floor=floor):
                    outputs = []
                    for tiled in (False, True):
                        scratch = [wp.zeros((2, width), dtype=wp.float64, device=device) for _ in range(3)]
                        scratch.append(wp.zeros((2, width, width), dtype=wp.float64, device=device))
                        out = [wp.zeros(2 * width, device=device) for _ in range(2)]
                        inputs = [
                            *base,
                            None,
                            wp.array(params, device=device),
                            0.004,
                            4,
                            1e-5,
                            1e-5,
                            1e-4,
                            floor,
                            1,
                            wp.array(bounds, device=device),
                            *scratch,
                        ]
                        if tiled:
                            kernel, block = _build_coupled_tiled_kernel(DrivePD.evaluate_force, chain, key, width)
                            wp.launch_tiled(kernel, dim=2, inputs=inputs, outputs=out, device=device, block_dim=block)
                        else:
                            wp.launch(baseline_kernel, dim=2, inputs=inputs, outputs=out, device=device)
                        outputs.append([array.numpy() for array in out])
                    for expected, actual in zip(*outputs, strict=True):
                        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=5e-4)


if __name__ == "__main__":
    unittest.main()
