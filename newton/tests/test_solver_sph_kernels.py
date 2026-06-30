# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton import ParticleFlags
from newton._src.solvers.sph.basic_kernels import compute_density_pressure
from newton._src.solvers.sph.kernels import (
    SPH_KERNEL_CUBIC,
    SPH_KERNEL_POLY6,
    SPH_KERNEL_SPIKY,
    SPH_KERNEL_WENDLAND,
    SPH_ROLE_FLUID,
    _kernel_gradient,
    _kernel_weight,
    _pressure_from_density,
    _viscosity_laplacian,
    sph_kernel_gradient_np,
    sph_kernel_weight_np,
    sph_pressure_from_density_np,
    sph_viscosity_laplacian_np,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def eval_kernel_weight_kernel(
    kernel_id: int,
    r: wp.array[float],
    h: float,
    weight_out: wp.array[float],
):
    i = wp.tid()
    weight_out[i] = _kernel_weight(kernel_id, r[i], h)


@wp.kernel
def eval_kernel_gradient_kernel(
    kernel_id: int,
    r_vec: wp.array[wp.vec3],
    r: wp.array[float],
    h: float,
    gradient_out: wp.array[wp.vec3],
):
    i = wp.tid()
    gradient_out[i] = _kernel_gradient(kernel_id, r_vec[i], r[i], h)


@wp.kernel
def eval_viscosity_laplacian_kernel(
    r: wp.array[float],
    h: float,
    laplacian_out: wp.array[float],
):
    i = wp.tid()
    laplacian_out[i] = _viscosity_laplacian(r[i], h)


@wp.kernel
def eval_pressure_from_density_kernel(
    rho: wp.array[float],
    rest_density: wp.array[float],
    sound_speed: wp.array[float],
    stiffness: wp.array[float],
    pressure_exponent: wp.array[float],
    pressure_min: wp.array[float],
    pressure_max: wp.array[float],
    pressure_out: wp.array[float],
):
    i = wp.tid()
    pressure_out[i] = _pressure_from_density(
        rho[i],
        rest_density[i],
        sound_speed[i],
        stiffness[i],
        pressure_exponent[i],
        pressure_min[i],
        pressure_max[i],
    )


def _kernel_ids() -> tuple[int, ...]:
    return (int(SPH_KERNEL_POLY6), int(SPH_KERNEL_CUBIC), int(SPH_KERNEL_WENDLAND), int(SPH_KERNEL_SPIKY))


def _eval_weight(kernel_id: int, r: np.ndarray, h: float, device) -> np.ndarray:
    r_wp = wp.array(r.astype(np.float32), dtype=float, device=device)
    weight_wp = wp.zeros(r.shape[0], dtype=float, device=device)
    wp.launch(eval_kernel_weight_kernel, dim=r.shape[0], inputs=[kernel_id, r_wp, h, weight_wp], device=device)
    return weight_wp.numpy()


def _eval_gradient(kernel_id: int, r_vec: np.ndarray, h: float, device) -> np.ndarray:
    r = np.linalg.norm(r_vec, axis=1).astype(np.float32)
    r_vec_wp = wp.array(r_vec.astype(np.float32), dtype=wp.vec3, device=device)
    r_wp = wp.array(r, dtype=float, device=device)
    gradient_wp = wp.zeros(r_vec.shape[0], dtype=wp.vec3, device=device)
    wp.launch(
        eval_kernel_gradient_kernel,
        dim=r_vec.shape[0],
        inputs=[kernel_id, r_vec_wp, r_wp, h, gradient_wp],
        device=device,
    )
    return gradient_wp.numpy()


def _reference_density(q: np.ndarray, mass: np.ndarray, h: float, kernel_id: int) -> np.ndarray:
    density = np.zeros(q.shape[0], dtype=np.float32)
    for i in range(q.shape[0]):
        for j in range(q.shape[0]):
            r = float(np.linalg.norm(q[i] - q[j]))
            if r < h:
                density[i] += float(mass[j]) * sph_kernel_weight_np(kernel_id, r, h)
    return density


def test_sph_kernel_weights_match_host_reference(test, device):
    h = 0.16
    h_ref = float(np.float32(h))
    radii = np.array([0.0, 0.025, 0.08, 0.14, 0.16, 0.2], dtype=np.float32)

    for kernel_id in _kernel_ids():
        result = _eval_weight(kernel_id, radii, h, device)
        expected = np.array([sph_kernel_weight_np(kernel_id, float(r), h_ref) for r in radii], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=2.0e-6, atol=1.0e-5)
        test.assertEqual(float(result[-1]), 0.0)
        test.assertEqual(float(result[-2]), 0.0)


def test_sph_kernel_gradients_match_host_reference(test, device):
    h = 0.16
    h_ref = float(np.float32(h))
    r_vec = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.025, -0.01, 0.005],
            [0.07, 0.03, -0.02],
            [0.16, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    radii = np.linalg.norm(r_vec, axis=1)

    for kernel_id in _kernel_ids():
        result = _eval_gradient(kernel_id, r_vec, h, device)
        expected = np.array(
            [sph_kernel_gradient_np(kernel_id, r_vec[i], float(radii[i]), h_ref) for i in range(r_vec.shape[0])],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result, expected, rtol=2.0e-5, atol=1.0e-4)
        np.testing.assert_array_equal(result[0], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(result[-1], np.zeros(3, dtype=np.float32))


def test_sph_viscosity_laplacian_matches_host_reference(test, device):
    h = 0.16
    h_ref = float(np.float32(h))
    radii = np.array([0.0, 0.04, 0.12, 0.16, 0.24], dtype=np.float32)
    r_wp = wp.array(radii, dtype=float, device=device)
    laplacian_wp = wp.zeros(radii.shape[0], dtype=float, device=device)

    wp.launch(eval_viscosity_laplacian_kernel, dim=radii.shape[0], inputs=[r_wp, h, laplacian_wp], device=device)

    expected = np.array([sph_viscosity_laplacian_np(float(r), h_ref) for r in radii], dtype=np.float32)
    np.testing.assert_allclose(laplacian_wp.numpy(), expected, rtol=2.0e-6, atol=1.0e-5)


def test_sph_pressure_equation_of_state_matches_host_reference(test, device):
    rho = np.array([900.0, 1000.0, 1120.0, 1250.0], dtype=np.float32)
    rest_density = np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float32)
    sound_speed = np.array([10.0, 10.0, 0.0, 20.0], dtype=np.float32)
    stiffness = np.array([0.0, 0.0, 500.0, 0.0], dtype=np.float32)
    pressure_exponent = np.array([1.0, 1.0, 2.0, 7.0], dtype=np.float32)
    pressure_min = np.array([-50.0, -50.0, -50.0, -50.0], dtype=np.float32)
    pressure_max = np.array([0.0, 0.0, 200.0, 10000.0], dtype=np.float32)
    pressure_wp = wp.zeros(rho.shape[0], dtype=float, device=device)

    wp.launch(
        eval_pressure_from_density_kernel,
        dim=rho.shape[0],
        inputs=[
            wp.array(rho, dtype=float, device=device),
            wp.array(rest_density, dtype=float, device=device),
            wp.array(sound_speed, dtype=float, device=device),
            wp.array(stiffness, dtype=float, device=device),
            wp.array(pressure_exponent, dtype=float, device=device),
            wp.array(pressure_min, dtype=float, device=device),
            wp.array(pressure_max, dtype=float, device=device),
            pressure_wp,
        ],
        device=device,
    )

    expected = sph_pressure_from_density_np(
        rho, rest_density, sound_speed, stiffness, pressure_exponent, pressure_min, pressure_max
    )
    np.testing.assert_allclose(pressure_wp.numpy(), expected, rtol=2.0e-5, atol=1.0e-4)
    test.assertGreaterEqual(float(np.min(pressure_wp.numpy())), float(np.min(pressure_min)))
    test.assertLessEqual(float(pressure_wp.numpy()[2]), float(pressure_max[2]))


def test_sph_density_summation_reference_case(test, device):
    h = 0.16
    mass = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    x = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.22, 0.0, 0.0]], dtype=np.float32)
    kernel_id = int(SPH_KERNEL_POLY6)

    density_0 = 0.0
    for j in range(x.shape[0]):
        r = float(np.linalg.norm(x[0] - x[j]))
        if r < h:
            density_0 += float(mass[j]) * sph_kernel_weight_np(kernel_id, r, h)

    expected = mass[0] * sph_kernel_weight_np(kernel_id, 0.0, h) + mass[1] * sph_kernel_weight_np(kernel_id, 0.05, h)
    test.assertAlmostEqual(density_0, expected, places=5)
    test.assertGreater(density_0, float(mass[0]) * sph_kernel_weight_np(kernel_id, 0.0, h))


def test_sph_density_pressure_kernel_matches_reference_case(test, device):
    h = 0.16
    kernel_id = int(SPH_KERNEL_POLY6)
    particle_count = 3
    q = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.22, 0.0, 0.0]], dtype=np.float32)
    mass = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    rest_density = np.array([900.0, 1000.0, 1100.0], dtype=np.float32)
    sound_speed = np.array([8.0, 10.0, 12.0], dtype=np.float32)
    stiffness = np.zeros(particle_count, dtype=np.float32)
    pressure_exponent = np.ones(particle_count, dtype=np.float32)
    pressure_min = np.array([-1.0e9, -1.0e9, -1.0e9], dtype=np.float32)
    pressure_max = np.zeros(particle_count, dtype=np.float32)

    q_wp = wp.array(q, dtype=wp.vec3, device=device)
    density_wp = wp.zeros(particle_count, dtype=float, device=device)
    pressure_wp = wp.zeros(particle_count, dtype=float, device=device)
    volume_wp = wp.zeros(particle_count, dtype=float, device=device)

    with wp.ScopedDevice(device):
        grid = wp.HashGrid(16, 16, 16)
        grid.reserve(particle_count)
        grid.build(q_wp, radius=h)

        wp.launch(
            compute_density_pressure,
            dim=particle_count,
            inputs=[
                grid.id,
                q_wp,
                wp.array(mass, dtype=float, device=device),
                wp.array(
                    np.full(particle_count, int(ParticleFlags.ACTIVE), dtype=np.int32), dtype=wp.int32, device=device
                ),
                wp.array(np.zeros(particle_count, dtype=np.int32), dtype=wp.int32, device=device),
                wp.array(np.full(particle_count, int(SPH_ROLE_FLUID), dtype=np.int32), dtype=wp.int32, device=device),
                wp.array(rest_density, dtype=float, device=device),
                wp.array(sound_speed, dtype=float, device=device),
                wp.array(stiffness, dtype=float, device=device),
                wp.array(pressure_exponent, dtype=float, device=device),
                wp.array(pressure_min, dtype=float, device=device),
                wp.array(pressure_max, dtype=float, device=device),
                wp.array(np.full(particle_count, h, dtype=np.float32), dtype=float, device=device),
                1000.0,
                10.0,
                0.0,
                1.0,
                h,
                kernel_id,
            ],
            outputs=[density_wp, pressure_wp, volume_wp],
            device=device,
        )

    expected_density = _reference_density(q, mass, h, kernel_id)
    expected_pressure = sph_pressure_from_density_np(
        expected_density, rest_density, sound_speed, stiffness, pressure_exponent, pressure_min, pressure_max
    )
    expected_volume = mass / np.maximum(expected_density, 1.0e-6)

    np.testing.assert_allclose(density_wp.numpy(), expected_density, rtol=2.0e-5, atol=1.0e-4)
    np.testing.assert_allclose(pressure_wp.numpy(), expected_pressure, rtol=2.0e-5, atol=1.0e-3)
    np.testing.assert_allclose(volume_wp.numpy(), expected_volume, rtol=2.0e-5, atol=1.0e-6)


devices = get_test_devices(mode="basic")


class TestSolverSPHKernels(unittest.TestCase):
    pass


for test_func in (
    test_sph_kernel_weights_match_host_reference,
    test_sph_kernel_gradients_match_host_reference,
    test_sph_viscosity_laplacian_matches_host_reference,
    test_sph_pressure_equation_of_state_matches_host_reference,
    test_sph_density_summation_reference_case,
    test_sph_density_pressure_kernel_matches_reference_case,
):
    add_function_test(TestSolverSPHKernels, test_func.__name__, test_func, devices=devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
