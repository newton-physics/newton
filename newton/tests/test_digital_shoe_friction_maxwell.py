# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify tangential mechanical relaxation and Coulomb return without output filtering."""

import unittest

import numpy as np
import warp as wp

from projects.digital_shoe.friction_maxwell import bristle_maxwell_step, column_maxwell_parameters


@wp.kernel
def _column_parameters(
    equilibrium: wp.array[float],
    overstress: wp.array[float],
    area: wp.array[float],
    rest: wp.array[float],
    tau: float,
    kt: wp.array[float],
    kv: wp.array[float],
):
    i = wp.tid()
    stiffness, viscosity = column_maxwell_parameters(equilibrium[i], overstress[i], area[i], rest[i], tau)
    kt[i] = stiffness
    kv[i] = viscosity


@wp.kernel
def _column_force(
    equilibrium: wp.array[float],
    overstress: wp.array[float],
    area: wp.array[float],
    rest: wp.array[float],
    forces: wp.array[wp.vec2],
):
    i = wp.tid()
    kt, kv = column_maxwell_parameters(equilibrium[i], overstress[i], area[i], rest[i], 0.005)
    force, _jac, _z, _q, _stuck, _dwell = bristle_maxwell_step(
        wp.vec2(0.01, 0.0), 0.001, 100.0, kt, kv, 0.005, 0.8, 0.0, wp.vec2(0.0), wp.vec2(0.0), 0, 0.0
    )
    forces[i] = force


@wp.kernel
def _eval(
    v: wp.array[wp.vec2],
    p: wp.array[float],
    h: float,
    n: float,
    z: wp.vec2,
    q: wp.vec2,
    outputs: wp.array[wp.vec2],
    jac: wp.array[wp.mat22],
    loss: wp.array[float],
):
    f, J, zn, qn, _s, _d = bristle_maxwell_step(v[0], h, n, p[0], p[1], p[2], p[3], 0.0, z, q, 1, 0.0)
    outputs[0] = f
    outputs[1] = zn
    outputs[2] = qn
    jac[0] = J
    loss[0] = f[0] + 0.2 * f[1]


class TestFrictionMaxwell(unittest.TestCase):
    """Test bounded stress, passive storage and force continuity."""

    def _step(self, device, v, dt=0.001, n=100.0, z=(0.0, 0.0), q=(0.0, 0.0), p=(1000.0, 10.0, 0.005, 0.8), grad=False):
        va = wp.array([v], dtype=wp.vec2, device=device, requires_grad=grad)
        pa = wp.array(p, dtype=float, device=device, requires_grad=grad)
        out = wp.zeros(3, dtype=wp.vec2, device=device, requires_grad=grad)
        jac = wp.zeros(1, dtype=wp.mat22, device=device)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=grad)
        if grad:
            tape = wp.Tape()
            with tape:
                wp.launch(_eval, dim=1, inputs=[va, pa, dt, n, wp.vec2(*z), wp.vec2(*q), out, jac, loss], device=device)
            tape.backward(loss=loss)
            return out.numpy(), jac.numpy()[0], float(loss.numpy()[0]), pa.grad.numpy(), va.grad.numpy()[0]
        wp.launch(_eval, dim=1, inputs=[va, pa, dt, n, wp.vec2(*z), wp.vec2(*q), out, jac, loss], device=device)
        return out.numpy(), jac.numpy()[0], float(loss.numpy()[0])

    def test_column_shear_parameters_scale_with_material_geometry(self):
        """Derive shear stiffness and viscosity per column and preserve patch refinement."""
        equilibrium = np.array([1200.0, 1200.0, 600.0], dtype=np.float32)
        overstress = np.array([0.5, 0.5, 2.0], dtype=np.float32)
        area = np.array([1.0e-4, 0.5e-4, 1.0e-4], dtype=np.float32)
        rest = np.array([0.02, 0.02, 0.04], dtype=np.float32)
        expected_kt = equilibrium * area / rest
        expected_kv = equilibrium * overstress * area / rest * 0.005
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            values = [wp.array(x, dtype=float, device=device) for x in (equilibrium, overstress, area, rest)]
            kt = wp.zeros(3, dtype=float, device=device)
            kv = wp.zeros(3, dtype=float, device=device)
            wp.launch(_column_parameters, dim=3, inputs=[*values, 0.005, kt, kv], device=device)
            np.testing.assert_allclose(kt.numpy(), expected_kt, rtol=2.0e-6)
            np.testing.assert_allclose(kv.numpy(), expected_kv, rtol=2.0e-6)
        # One patch and two half-area columns have identical summed stiffness.
        self.assertAlmostEqual(float(expected_kt[0]), float(expected_kt[1] + expected_kt[1]), places=5)
        self.assertAlmostEqual(float(expected_kv[0]), float(expected_kv[1] + expected_kv[1]), places=5)

    def test_column_material_changes_force_for_same_slip(self):
        """Apply a common unsaturated slip and scale force with column geometry."""
        equilibrium = np.array([1200.0, 1200.0, 600.0], dtype=np.float32)
        overstress = np.array([0.5, 0.5, 2.0], dtype=np.float32)
        area = np.array([1.0e-4, 0.5e-4, 1.0e-4], dtype=np.float32)
        rest = np.array([0.02, 0.02, 0.04], dtype=np.float32)
        expected_ratio = (equilibrium * area / rest) * (1.0 + overstress * 0.005 / (0.005 + 0.001))
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            inputs = [wp.array(x, dtype=float, device=device) for x in (equilibrium, overstress, area, rest)]
            forces = wp.zeros(3, dtype=wp.vec2, device=device)
            wp.launch(_column_force, dim=3, inputs=[*inputs, forces], device=device)
            magnitude = -forces.numpy()[:, 0]
            np.testing.assert_allclose(magnitude, 0.01 * 0.001 * expected_ratio, rtol=3e-5)
            self.assertGreater(magnitude[0], magnitude[1])
            self.assertAlmostEqual(float(magnitude[0]), float(2.0 * magnitude[1]), delta=1e-7)

    def test_velocity_step_has_no_finite_force_jump(self):
        """Make the force response to a velocity step vanish with timestep."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            magnitudes = []
            for h in [0.001, 0.0005, 0.00025, 0.000125]:
                result = self._step(device, [0.1, 0.0], dt=h)[0]
                magnitudes.append(abs(result[0, 0]))
            self.assertTrue(np.all(np.diff(magnitudes) < 0))
            self.assertLess(magnitudes[-1], 0.2 * magnitudes[0])
            # A direct 10 N s/m parallel dashpot would jump by 1 N, independent of h.
            self.assertLess(magnitudes[-1], 0.04)

    def test_relaxation_has_a_physical_time_scale(self):
        """Converge to Maxwell relaxation in seconds rather than per-step creep."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            errors = []
            for steps in [10, 20, 40]:
                z = np.array([0.001, 0.0])
                q = np.array([0.5, 0.0])
                h = 0.005 / steps
                for _ in range(steps):
                    result = self._step(device, [0.0, 0.0], dt=h, z=z, q=q)[0]
                    z = result[1]
                    q = result[2]
                errors.append(abs(q[0] - 0.5 * np.exp(-1)))
            self.assertGreater(errors[0], errors[1])
            self.assertGreater(errors[1], errors[2])
            np.testing.assert_allclose(z, [0.001, 0.0], atol=1e-8)

    def test_passivity_with_changing_normal_and_reversal(self):
        """Bound traction and dissipate both elastic and Maxwell stored energy."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            z = np.array([0.0001, -0.0002])
            q = np.array([0.3, 0.1])
            p = (1000.0, 10.0, 0.005, 0.8)
            for n, v in [
                (10.0, [0.2, 0.1]),
                (0.2, [2.0, -0.5]),
                (1.0, [-0.1, 0.3]),
                (0.0, [0.5, 0.0]),
                (10.0, [0.0, 0.0]),
            ]:
                old = 0.5 * p[0] * np.dot(z, z) + 0.5 * p[2] * np.dot(q, q) / p[1]
                result = self._step(device, v, n=n, z=z, q=q, p=p)[0]
                f, z, q = result
                energy = 0.5 * p[0] * np.dot(z, z) + 0.5 * p[2] * np.dot(q, q) / p[1]
                work = np.dot(f, v) * 0.001
                self.assertLessEqual(energy - old + work, 1e-6 * (old + energy + abs(work) + 1e-5))
                self.assertLessEqual(np.linalg.norm(f), p[3] * n + 1e-6)

    def test_tangent_and_parameter_gradients(self):
        """Match force Jacobians and parameter Tape gradients to finite differences."""
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()]:
            for n in [100.0, 0.05]:
                p = np.array([1000.0, 10.0, 0.005, 0.8])
                v = np.array([0.2, 0.1])
                result = self._step(device, v, n=n, p=p, grad=True)
                for axis in range(2):
                    vh = v.copy()
                    vl = v.copy()
                    vh[axis] += 1e-4
                    vl[axis] -= 1e-4
                    high = self._step(device, vh, n=n, p=p)
                    low = self._step(device, vl, n=n, p=p)
                    fd = (high[0][0] - low[0][0]) / 0.0002
                    np.testing.assert_allclose(result[1][:, axis], fd, rtol=0.005, atol=1e-4)
                for axis, h in enumerate([0.1, 0.001, 1e-6, 1e-4]):
                    high = p.copy()
                    low = p.copy()
                    high[axis] += h
                    low[axis] -= h
                    fd = (self._step(device, v, n=n, p=high)[2] - self._step(device, v, n=n, p=low)[2]) / (2 * h)
                    np.testing.assert_allclose(result[3][axis], fd, rtol=0.01, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
