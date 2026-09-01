# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _tet_volume(q, tet):
    x0 = q[tet[0]]
    x1 = q[tet[1]]
    x2 = q[tet[2]]
    x3 = q[tet[3]]
    return np.linalg.det(np.stack((x1 - x0, x2 - x0, x3 - x0), axis=-1)) / 6.0


def _make_single_tet_model(points, device, k_mu=5.0e4, k_lambda=2.0e4):
    builder = newton.ModelBuilder()
    for p in points:
        builder.add_particle(wp.vec3(*p), wp.vec3(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.add_tetrahedron(0, 1, 2, 3, k_mu=k_mu, k_lambda=k_lambda, k_damp=0.0)
    model = builder.finalize(device=device)
    model.gravity.zero_()
    return model


def _step_compressed_tet(points, device, k_mu, k_lambda):
    model = _make_single_tet_model(points, device, k_mu, k_lambda)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    rest_q = state_0.particle_q.numpy()
    tet = model.tet_indices.numpy().reshape(-1, 4)[0]
    rest_volume = abs(_tet_volume(rest_q, tet))

    compressed_q = rest_q.copy()
    compressed_q[:, 2] *= 0.5
    compressed_volume = abs(_tet_volume(compressed_q, tet))
    state_0.particle_q = wp.array(compressed_q, dtype=wp.vec3, device=device)
    state_0.particle_qd.zero_()

    solver = newton.solvers.SolverXPBD(model, iterations=20)
    dt = 1.0 / (60.0 * 32.0)
    for _ in range(10):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    final_q = state_0.particle_q.numpy()
    final_volume = abs(_tet_volume(final_q, tet))

    return rest_volume, compressed_volume, final_volume


def _step_activated_tet(points, device, activation):
    model = _make_single_tet_model(points, device, k_mu=1.0e8, k_lambda=1.0e8)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    control.tet_activations.assign(np.array([activation], dtype=np.float32))

    rest_q = state_0.particle_q.numpy()
    tet = model.tet_indices.numpy().reshape(-1, 4)[0]
    rest_volume = abs(_tet_volume(rest_q, tet))

    solver = newton.solvers.SolverXPBD(model, iterations=20)
    dt = 1.0 / (60.0 * 32.0)
    for _ in range(5):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    final_q = state_0.particle_q.numpy()
    final_volume = abs(_tet_volume(final_q, tet))

    return rest_volume, final_volume


def test_xpbd_tetrahedra_resist_compression(test, device):
    test_cases = (
        (
            "unit",
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        ),
        (
            "skewed",
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.8, 0.1, 0.0],
                    [0.1, 0.7, 0.2],
                    [0.2, 0.1, 0.9],
                ],
                dtype=np.float32,
            ),
        ),
    )

    with wp.ScopedDevice(device):
        for name, points in test_cases:
            with test.subTest(name=name):
                rest_volume, compressed_volume, final_volume = _step_compressed_tet(
                    points, device, k_mu=5.0e4, k_lambda=2.0e4
                )
                compressed_error = rest_volume - compressed_volume
                final_error = rest_volume - final_volume

                test.assertGreater(
                    final_volume,
                    compressed_volume + 0.1 * compressed_error,
                    msg=(
                        f"XPBD tet volume did not recover for {name}: "
                        f"rest={rest_volume:.6g}, compressed={compressed_volume:.6g}, final={final_volume:.6g}"
                    ),
                )
                test.assertLess(
                    abs(final_error),
                    0.9 * abs(compressed_error),
                    msg=(
                        f"XPBD tet volume error did not improve for {name}: "
                        f"rest={rest_volume:.6g}, compressed={compressed_volume:.6g}, final={final_volume:.6g}"
                    ),
                )


def test_xpbd_tetrahedra_use_material_stiffness(test, device):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with wp.ScopedDevice(device):
        rest_volume, compressed_volume, low_volume = _step_compressed_tet(points, device, k_mu=1.0e1, k_lambda=1.0e1)
        _, _, high_volume = _step_compressed_tet(points, device, k_mu=1.0e8, k_lambda=1.0e8)

    recovery_scale = rest_volume - compressed_volume
    test.assertGreater(
        high_volume - low_volume,
        0.5 * recovery_scale,
        msg=(
            "XPBD tet material stiffness did not change compression recovery: "
            f"rest={rest_volume:.6g}, compressed={compressed_volume:.6g}, "
            f"low={low_volume:.6g}, high={high_volume:.6g}"
        ),
    )


def test_xpbd_tetrahedra_use_control_activation(test, device):
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with wp.ScopedDevice(device):
        rest_volume, inactive_volume = _step_activated_tet(points, device, activation=0.0)
        _, activated_volume = _step_activated_tet(points, device, activation=0.25)

    test.assertAlmostEqual(inactive_volume, rest_volume, delta=1.0e-5)
    test.assertLess(
        activated_volume,
        inactive_volume - 0.01,
        msg=(
            "XPBD tet control activation did not change volume: "
            f"rest={rest_volume:.6g}, inactive={inactive_volume:.6g}, activated={activated_volume:.6g}"
        ),
    )


class TestSolverXPBDTetrahedra(unittest.TestCase):
    pass


add_function_test(
    TestSolverXPBDTetrahedra,
    "test_xpbd_tetrahedra_resist_compression",
    test_xpbd_tetrahedra_resist_compression,
    devices=get_test_devices(),
)
add_function_test(
    TestSolverXPBDTetrahedra,
    "test_xpbd_tetrahedra_use_material_stiffness",
    test_xpbd_tetrahedra_use_material_stiffness,
    devices=get_test_devices(),
)
add_function_test(
    TestSolverXPBDTetrahedra,
    "test_xpbd_tetrahedra_use_control_activation",
    test_xpbd_tetrahedra_use_control_activation,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
