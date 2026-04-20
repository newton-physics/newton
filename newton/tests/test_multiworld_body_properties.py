# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test that per-world body property changes produce correct physics.

Regression test for a bug in convert_warp_coords_to_mj_kernel where
joint_type and joint_child were indexed with the per-world joint index
instead of the global index, causing incorrect CoM references for
worlds with worldid > 0.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_model_with_per_world_com(device, world_base_coms):
    """Build a multi-world model where each world has a free-floating
    body with its own base CoM.
    """
    scene = newton.ModelBuilder()

    for base_com in world_base_coms:
        template = newton.ModelBuilder()
        base = template.add_link(
            mass=5.0,
            com=base_com,
            inertia=wp.mat33(np.eye(3) * 0.1),
        )
        j_free = template.add_joint_free(parent=-1, child=base)
        template.add_articulation([j_free])

        scene.begin_world()
        scene.add_builder(template)
        scene.end_world()

    return scene.finalize(device=device)


def _run_sim(model, num_steps: int = 100, sim_dt: float = 1e-3):
    """Step simulation and return final joint_q as numpy array."""
    solver = SolverMuJoCo(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # Set initial height and angular velocity for all worlds.
    # Non-zero angular velocity is critical: the bug in
    # convert_warp_coords_to_mj_kernel affects the velocity
    # conversion v_origin = v_com - w x com. With w=0 the cross
    # product vanishes and the wrong com is harmless.
    nw = model.world_count
    q_per_world = model.joint_coord_count // nw
    qd_per_world = model.joint_dof_count // nw
    joint_q = state_0.joint_q.numpy().reshape(nw, q_per_world)
    joint_q[:, 2] = 1.0  # z = 1m for all worlds
    state_0.joint_q.assign(joint_q.flatten())
    state_1.joint_q.assign(joint_q.flatten())
    # Set initial angular velocity (world frame) on the free joint
    joint_qd = state_0.joint_qd.numpy().reshape(nw, qd_per_world)
    joint_qd[:, 3] = 1.0  # wx = 1 rad/s
    joint_qd[:, 4] = 0.5  # wy = 0.5 rad/s
    state_0.joint_qd.assign(joint_qd.flatten())
    state_1.joint_qd.assign(joint_qd.flatten())
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    newton.eval_fk(model, state_1.joint_q, state_1.joint_qd, state_1)

    for _ in range(num_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0

    return state_0.joint_q.numpy()


def test_perworld_com_produces_consistent_physics(test, device):
    """Multi-world with different per-world base CoM should match
    independent single-world runs with the same CoM."""
    com_a = wp.vec3(0.0, 0.0, 0.0)
    com_b = wp.vec3(0.05, 0.0, -0.02)

    # --- Reference: single-world runs ---
    ref_q = {}
    for com_val, label in [(com_a, "A"), (com_b, "B")]:
        model = _build_model_with_per_world_com(device, [com_val])
        ref_q[label] = _run_sim(model)

    # --- Multi-world run ---
    model = _build_model_with_per_world_com(device, [com_a, com_b])
    multi_q = _run_sim(model)

    q_per_world = model.joint_coord_count // model.world_count
    multi_q = multi_q.reshape(model.world_count, q_per_world)

    np.testing.assert_allclose(
        multi_q[0],
        ref_q["A"],
        atol=1e-4,
        err_msg="World 0 (com_A) diverges from single-world reference",
    )
    np.testing.assert_allclose(
        multi_q[1],
        ref_q["B"],
        atol=1e-4,
        err_msg="World 1 (com_B) diverges from single-world reference",
    )


class TestMultiworldBodyProperties(unittest.TestCase):
    """Verify that multi-world simulations with per-world body mass/CoM
    produce physically consistent results across all worlds."""

    pass


devices = get_test_devices()
for device in devices:
    add_function_test(
        TestMultiworldBodyProperties,
        f"test_perworld_com_produces_consistent_physics_{device}",
        test_perworld_com_produces_consistent_physics,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
