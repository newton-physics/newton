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
from newton.solvers import SolverMuJoCo, SolverNotifyFlags
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_articulated_model(device, num_worlds: int = 2):
    """Build a 2-link articulated model (free joint + revolute).

    Using multiple joints/bodies is essential because the indexing
    bug only manifests when joint_child maps to different bodies
    across the flat array -- a single-body model has identical
    indices for all worlds.
    """
    template = newton.ModelBuilder()

    # Base link (free floating)
    base = template.add_link(
        mass=5.0,
        com=wp.vec3(0.02, 0.05, -0.03),
        inertia=wp.mat33(np.eye(3) * 0.1),
    )
    template.add_shape_sphere(base, radius=0.1)
    j_free = template.add_joint_free(parent=-1, child=base)

    # Child link attached via revolute joint
    child = template.add_link(
        mass=1.0,
        com=wp.vec3(-0.05, 0.03, 0.01),
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    template.add_shape_sphere(child, radius=0.05)
    j_rev = template.add_joint_revolute(
        parent=base,
        child=child,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.3), wp.quat_identity()),
    )
    template.add_articulation([j_free, j_rev])

    if num_worlds == 1:
        model = template.finalize(device=device)
    else:
        builder = newton.ModelBuilder()
        builder.replicate(template, world_count=num_worlds)
        model = builder.finalize(device=device)
    return model


def _run_sim(model, num_steps: int = 100, sim_dt: float = 1e-3):
    """Step simulation and return final joint_q as numpy array."""
    solver = SolverMuJoCo(model)
    solver.notify_model_changed(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    # Set initial height and angular velocity for all worlds.
    # Non-zero angular velocity is critical: the bug in
    # convert_warp_coords_to_mj_kernel affects the velocity
    # conversion v_origin = v_com - w x com. With w=0 the cross
    # product vanishes and the wrong com is harmless.
    q_total = model.joint_coord_count
    qd_total = model.joint_dof_count
    nw = model.world_count
    q_per_world = q_total // nw
    qd_per_world = qd_total // nw
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
    com_a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    com_b = np.array([0.05, 0.0, -0.02], dtype=np.float32)

    # --- Reference: single-world runs ---
    ref_q = {}
    for com_val, label in [(com_a, "A"), (com_b, "B")]:
        model = _build_articulated_model(device, num_worlds=1)
        body_com = model.body_com.numpy().reshape(-1, 3)
        body_com[0] = com_val  # base body
        model.body_com.assign(body_com.flatten())
        ref_q[label] = _run_sim(model)

    # --- Multi-world run ---
    model = _build_articulated_model(device, num_worlds=2)
    bodies_per_world = model.body_count // model.world_count
    body_com = model.body_com.numpy().reshape(model.world_count, bodies_per_world, 3)
    body_com[0, 0] = com_a
    body_com[1, 0] = com_b
    model.body_com.assign(body_com.reshape(-1, 3))
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


# The buggy code path is warp-to-mj coordinate conversion, which only
# runs on CUDA (SolverMuJoCo with use_mujoco_cpu=False). Skip CPU devices
# so CPU-only CI runners don't fail.
devices = [d for d in get_test_devices() if d.is_cuda]
for device in devices:
    add_function_test(
        TestMultiworldBodyProperties,
        f"test_perworld_com_produces_consistent_physics_{device}",
        test_perworld_com_produces_consistent_physics,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
