# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_soft_grid(device, *, fix_left=False):
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.15,
        cell_y=0.15,
        cell_z=0.15,
        density=1.0e3,
        k_mu=5.0e4,
        k_lambda=5.0e4,
        k_damp=1.0e-2,
        fix_left=fix_left,
        particle_radius=0.0,
        add_surface_mesh_edges=False,
    )
    builder.color()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    return model


def _make_solver(model, solver_name):
    if solver_name == "semi_implicit":
        return newton.solvers.SolverSemiImplicit(model)

    if solver_name == "xpbd":
        return newton.solvers.SolverXPBD(
            model,
            iterations=10,
            soft_body_relaxation=0.7,
        )

    if solver_name == "vbd":
        return newton.solvers.SolverVBD(
            model,
            iterations=10,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

    raise ValueError(f"Unsupported solver: {solver_name}")


def _tet_volumes(q, tet_indices):
    volumes = np.empty(tet_indices.shape[0], dtype=np.float32)
    for tet_id, tet in enumerate(tet_indices):
        x0 = q[tet[0]]
        x1 = q[tet[1]]
        x2 = q[tet[2]]
        x3 = q[tet[3]]
        volumes[tet_id] = np.linalg.det(np.stack((x1 - x0, x2 - x0, x3 - x0), axis=-1)) / 6.0
    return volumes


def _step(model, solver, state_0, state_1, steps=60, dt=1.0 / 300.0):
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    return state_0


def _assert_finite_state(test, q, qd):
    test.assertTrue(np.isfinite(q).all(), "soft-body positions contain non-finite values")
    test.assertTrue(np.isfinite(qd).all(), "soft-body velocities contain non-finite values")
    test.assertLess(np.linalg.norm(qd, axis=1).max(), 5.0, "soft-body rollout produced excessive velocity")


def test_soft_grid_free_fall(test, device, solver_name):
    """Check that an unconstrained tet grid falls as a coherent soft body."""
    with wp.ScopedDevice(device):
        model = _build_soft_grid(device)
        solver = _make_solver(model, solver_name)
        state_0 = model.state()
        state_1 = model.state()

        q_initial = np.array(state_0.particle_q.numpy(), copy=True)
        tet_indices = np.array(model.tet_indices.numpy(), copy=True)
        initial_volumes = _tet_volumes(q_initial, tet_indices)

        final_state = _step(model, solver, state_0, state_1)
        q_final = np.array(final_state.particle_q.numpy(), copy=True)
        qd_final = np.array(final_state.particle_qd.numpy(), copy=True)
        final_volumes = _tet_volumes(q_final, tet_indices)

    _assert_finite_state(test, q_final, qd_final)

    initial_com = q_initial.mean(axis=0)
    final_com = q_final.mean(axis=0)
    test.assertAlmostEqual(final_com[0], initial_com[0], delta=1.0e-3)
    test.assertAlmostEqual(final_com[1], initial_com[1], delta=1.0e-3)
    test.assertLess(final_com[2], initial_com[2] - 0.15)
    test.assertGreater(final_com[2], initial_com[2] - 0.25)

    volume_ratio = final_volumes / initial_volumes
    test.assertGreater(volume_ratio.min(), 0.99)
    test.assertLess(volume_ratio.max(), 1.01)


def test_soft_grid_anchored_deforms(test, device, solver_name):
    """Check that an anchored tet grid deforms while keeping fixed vertices pinned."""
    with wp.ScopedDevice(device):
        model = _build_soft_grid(device, fix_left=True)
        solver = _make_solver(model, solver_name)
        state_0 = model.state()
        state_1 = model.state()

        q_initial = np.array(state_0.particle_q.numpy(), copy=True)
        inv_mass = np.array(model.particle_inv_mass.numpy(), copy=True)
        tet_indices = np.array(model.tet_indices.numpy(), copy=True)
        initial_volumes = _tet_volumes(q_initial, tet_indices)

        final_state = _step(model, solver, state_0, state_1)
        q_final = np.array(final_state.particle_q.numpy(), copy=True)
        qd_final = np.array(final_state.particle_qd.numpy(), copy=True)
        final_volumes = _tet_volumes(q_final, tet_indices)

    _assert_finite_state(test, q_final, qd_final)

    fixed = inv_mass == 0.0
    dynamic = inv_mass > 0.0
    fixed_displacement = np.linalg.norm(q_final[fixed] - q_initial[fixed], axis=1)
    test.assertLessEqual(fixed_displacement.max(), 1.0e-6)

    initial_dynamic_com = q_initial[dynamic].mean(axis=0)
    final_dynamic_com = q_final[dynamic].mean(axis=0)
    test.assertLess(final_dynamic_com[2], initial_dynamic_com[2] - 0.04)

    volume_ratio = final_volumes / initial_volumes
    test.assertGreater(volume_ratio.min(), 0.6)
    test.assertLess(volume_ratio.max(), 1.5)
    test.assertGreater(final_volumes.min(), 0.0)


class TestSoftBodySimulation(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
for solver in ("semi_implicit", "xpbd", "vbd"):
    add_function_test(
        TestSoftBodySimulation,
        f"test_soft_grid_free_fall_{solver}",
        test_soft_grid_free_fall,
        devices=devices,
        solver_name=solver,
    )
    add_function_test(
        TestSoftBodySimulation,
        f"test_soft_grid_anchored_deforms_{solver}",
        test_soft_grid_anchored_deforms,
        devices=devices,
        solver_name=solver,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
