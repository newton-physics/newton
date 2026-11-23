# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Finite-difference checks of the ANYmal backflip example's loss kernels.

Warp does not replay dynamic loops in the backward pass, so the example
builds its running minima and maxima from loop-free kernels (an argmin
pre-pass plus a gather, statically unrolled paw minima, chained per-element
prefix maxima). These tests check that the resulting adjoints match central
differences on small synthetic scenes.
"""

import unittest

import numpy as np
import warp as wp

from newton.examples.diffsim import example_diffsim_anymal_backflip as bf
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _knee_order_loss(device, dz):
    """Knee-ordering loss of one body whose knee points sit 5/0/5 cm below its
    origin while one paw is 5 mm above the ground: the knee-margin violation
    and the grounded mask are both active, so the gradient is nonzero."""
    pts_body = wp.array([0, 0, 0], dtype=wp.int32, device=device)
    pts_local = wp.array([[0.0, 0.0, -0.55], [0.0, 0.0, -0.5], [0.0, 0.0, -0.45]], dtype=wp.vec3, device=device)
    paw_body = wp.array([0, 0, 0, 0], dtype=wp.int32, device=device)
    paw_center = wp.array(
        [[0.0, 0.0, -0.495], [0.0, 0.0, -0.3], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=wp.vec3, device=device
    )
    paw_r = wp.zeros(4, dtype=wp.float32, device=device)
    body_q = wp.array([[0.0, 0.0, 0.5 + dz, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transform, requires_grad=True, device=device)
    argmin = wp.zeros(1, dtype=wp.int32, device=device)
    mink = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
    minp = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
    wp.launch(bf.knee_argmin_kernel, dim=1, inputs=[body_q, pts_body, pts_local, 3, argmin, 0], device=device)
    wp.launch(bf.knee_mingz_kernel, dim=1, inputs=[body_q, pts_body, pts_local, argmin, mink, 0], device=device)
    wp.launch(bf.pawraw_state_kernel, dim=1, inputs=[body_q, paw_body, paw_center, paw_r, minp, 0], device=device)
    wp.launch(
        bf.order_state_g_kernel,
        dim=1,
        inputs=[mink, minp, 0, bf.W_ORDER_A, bf.KNEE_MARGIN_G, bf.PAW_MASK_H, loss],
        device=device,
    )
    return body_q, loss


def test_knee_order_gradient(test, device):
    """The knee-ordering loss's gradient w.r.t. the body height matches central differences."""
    eps = 1.0e-4
    fd = (float(_knee_order_loss(device, eps)[1].numpy()[0]) - float(_knee_order_loss(device, -eps)[1].numpy()[0])) / (
        2.0 * eps
    )
    tape = wp.Tape()
    with tape:
        body_q, loss = _knee_order_loss(device, 0.0)
    tape.backward(loss)
    grad = float(body_q.grad.numpy()[0][2])
    test.assertGreater(abs(fd), 1e-3, "harness broken: the knee-ordering loss does not depend on the body height")
    test.assertAlmostEqual(grad, fd, delta=0.02 * abs(fd))


def _band_loss(device, z_np, requires_grad):
    """Apex-band loss over a 17-state height trajectory, with the apex built by chained cum_max launches."""
    n = z_np.size
    z = wp.array(z_np.astype(np.float32), requires_grad=requires_grad, device=device)
    cum_z = wp.zeros(n, dtype=wp.float32, requires_grad=requires_grad, device=device)
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=requires_grad, device=device)
    for t in range(n):
        wp.launch(bf.cum_max_kernel, dim=1, inputs=[z, cum_z, t], device=device)
    wp.launch(
        bf.loss_jump_band_kernel, dim=1, inputs=[cum_z, n - 1, loss, bf.C_BAND, bf.Z_GOOD, bf.Z_LOW], device=device
    )
    return z, loss


def test_apex_band_gradient(test, device):
    """The apex-band gradient lands on the apex state only and matches central differences."""
    apex = 9
    z_np = np.full(17, 1.5)
    z_np[apex] = 2.0  # above the band's ceiling, so the loss pulls the apex down
    eps = 1.0e-3
    z_hi, z_lo = z_np.copy(), z_np.copy()
    z_hi[apex] += eps
    z_lo[apex] -= eps
    fd = (
        float(_band_loss(device, z_hi, False)[1].numpy()[0]) - float(_band_loss(device, z_lo, False)[1].numpy()[0])
    ) / (2.0 * eps)
    tape = wp.Tape()
    with tape:
        z, loss = _band_loss(device, z_np, True)
    tape.backward(loss)
    grad = z.grad.numpy()
    test.assertGreater(abs(fd), 1e-3, "harness broken: the apex is inside the band")
    test.assertAlmostEqual(float(grad[apex]), fd, delta=0.02 * abs(fd))
    off_apex = np.delete(grad, apex)
    test.assertTrue(np.all(off_apex == 0.0), f"band gradient leaked off the apex state: {off_apex}")


class TestDiffsimAnymalBackflipLosses(unittest.TestCase):
    """Finite-difference checks of the backflip example's loss kernels."""


add_function_test(
    TestDiffsimAnymalBackflipLosses, "test_knee_order_gradient", test_knee_order_gradient, devices=get_test_devices()
)
add_function_test(
    TestDiffsimAnymalBackflipLosses, "test_apex_band_gradient", test_apex_band_gradient, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
