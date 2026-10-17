# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check additive CUDA mass assembly and the scalar gradient path."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.articulation_mass import _assemble_mass_matrix
from newton.tests.test_actuators import _build_two_link


class TestMassAssembly(unittest.TestCase):
    """Cover compact/generic assembly and public autograd dispatch."""

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA assembly requires CUDA")
    def test_compaction_accumulation_mask_and_armature(self):
        """Preserve masked padding and nonzero outputs for dense and sparse Jacobians."""
        device = wp.get_cuda_device()
        rng = np.random.default_rng(17)
        # The two batches straddle the compaction threshold; 129 DOFs exceeds its width.
        for batch, width, sparse in ((2, 7, False), (32, 7, False), (32, 7, True), (32, 129, True)):
            with self.subTest(batch=batch, width=width, sparse=sparse):
                links = 4
                jac = rng.normal(size=(batch, links * 6, width)).astype(np.float32) * 0.1
                if sparse:
                    jac[:, :, ::2] = 0.0
                factors = rng.normal(size=(batch * links, 6, 6)).astype(np.float32) * 0.1
                inertia = factors @ factors.transpose(0, 2, 1) + np.eye(6, dtype=np.float32)
                mask = np.arange(batch) % 2 == 0
                starts = np.arange(batch + 1, dtype=np.int32) * links
                qstarts = np.arange(batch * links + 1, dtype=np.int32) * width // links
                args = [
                    wp.array(starts, device=device),
                    wp.array(starts[1:], device=device),
                    batch,
                    wp.array(mask, device=device),
                    wp.array(np.arange(batch * links, dtype=np.int32), device=device),
                    wp.array(qstarts, device=device),
                    wp.array(inertia, dtype=wp.spatial_matrix, device=device),
                    wp.array(jac, device=device),
                    wp.full(batch * width, 0.2, device=device),
                ]
                expected = np.full((batch, width, width), 0.5, dtype=np.float64)
                for art in range(batch):
                    if not mask[art]:
                        continue
                    for link in range(links):
                        block = jac[art, link * 6 : (link + 1) * 6].astype(np.float64)
                        expected[art] += block.T @ inertia[art * links + link] @ block
                    expected[art, np.arange(width), np.arange(width)] += 0.2
                actual = wp.full((batch, width, width), 0.5, device=device)
                _assemble_mass_matrix(args, actual)
                np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-4, atol=5e-5)

    def test_public_mass_gradient(self):
        """Differentiate the public mass assembly with respect to its supplied Jacobian."""
        device = wp.get_device()
        model = _build_two_link(device)
        model.joint_armature = wp.array(model.joint_armature.numpy(), device=device, requires_grad=True)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        original = newton.eval_jacobian(model, state).numpy()
        J = wp.array(original, device=device, requires_grad=True)
        H = wp.zeros(
            (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation),
            device=device,
            requires_grad=True,
        )
        with wp.Tape() as tape:
            newton.eval_mass_matrix(model, state, H=H, J=J)
        tape.backward(grads={H: wp.ones_like(H)})
        np.testing.assert_allclose(model.joint_armature.grad.numpy(), 1.0, atol=1e-6)
        analytic = J.grad.numpy()
        index = np.unravel_index(np.argmax(np.abs(analytic)), analytic.shape)
        samples = []
        for delta in (-0.001, 0.001):
            perturbed = original.copy()
            perturbed[index] += delta
            samples.append(
                float(newton.eval_mass_matrix(model, state, J=wp.array(perturbed, device=device)).numpy().sum())
            )
        self.assertAlmostEqual(float(analytic[index]), (samples[1] - samples[0]) / 0.002, delta=0.005)


if __name__ == "__main__":
    unittest.main()
