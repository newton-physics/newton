# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the experimental LOX normal contact laws against scalar solutions."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.lox.bias import (
    compute_contact_normal_regularization,
    compute_contact_penetration_bias,
)


@wp.kernel
def _evaluate_normal_law(parameters: wp.array2d[wp.float32], result: wp.array2d[wp.float32]):
    i = wp.tid()
    distance = parameters[i, 0]
    free_velocity = parameters[i, 2]
    time_step = parameters[i, 4]
    fraction = parameters[i, 5]
    compliance = parameters[i, 8]
    mechanical_delassus = parameters[i, 9]
    regularization = compute_contact_normal_regularization(compliance, time_step)
    bias = compute_contact_penetration_bias(distance, time_step, fraction)
    impulse = wp.max(-(free_velocity + bias) / (mechanical_delassus + regularization), 0.0)
    result[i, 0] = regularization
    result[i, 1] = bias
    result[i, 2] = 0.0
    result[i, 3] = impulse
    result[i, 4] = free_velocity + mechanical_delassus * impulse


class TestLOXContactLaw(unittest.TestCase):
    def _evaluate(self, rows):
        for device in wp.get_devices():
            with self.subTest(device=device.alias):
                parameters = wp.array(np.asarray(rows, dtype=np.float32), dtype=wp.float32, device=device)
                result = wp.empty((len(rows), 5), dtype=wp.float32, device=device)
                wp.launch(_evaluate_normal_law, dim=len(rows), inputs=[parameters], outputs=[result], device=device)
                yield result.numpy()

    def test_compliance_static_load_deflection(self):
        """Balance a static load at the compliant deflection for different timesteps."""
        compliance, force, inverse_mass = 0.002, 10.0, 0.5
        rows = [
            [-compliance * force, 0.0, -h * force * inverse_mass, 0.0, h, 1.0, 0.0, 0.0, compliance, inverse_mass]
            for h in (0.01, 0.02, 0.1)
        ]
        for result in self._evaluate(rows):
            np.testing.assert_allclose(result[:, 3], [0.1, 0.2, 1.0], atol=1.0e-6)
            np.testing.assert_allclose(result[:, 4], 0.0, atol=1.0e-6)

    def test_compliance_hard_limit_and_separation(self):
        """Approach the hard-contact impulse and avoid attractive separated forces."""
        rows = [[0.0, -1.0, -1.0, 0.0, 0.1, 1.0, 0.0, 0.0, chi, 0.5] for chi in (0.01, 1.0e-8, 0.0)]
        rows.append([0.01, 0.0, 1.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.01, 0.5])
        for result in self._evaluate(rows):
            np.testing.assert_allclose(result[:, 0], [1.0, 1.0e-6, 0.0, 1.0], atol=1.0e-6)
            np.testing.assert_allclose(result[:, 3], [2.0 / 3.0, 2.0, 2.0, 0.0], atol=5.0e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
