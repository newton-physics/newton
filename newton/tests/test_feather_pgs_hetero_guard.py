# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverFeatherPGS on multi-world models whose worlds have different DOF counts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _make_chain_world(n_links: int) -> newton.ModelBuilder:
    """A fixed-base serial chain of ``n_links`` revolute links (n_links DOFs)."""
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    for _ in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.15, hy=0.03, hz=0.03)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity())
        else:
            parent_xform = wp.transform(p=wp.vec3(0.15, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    return builder


def _build_model(world_link_counts: list[int], device) -> newton.Model:
    """One chain world per entry in ``world_link_counts``, plus a ground plane."""
    scene = newton.ModelBuilder()
    for n_links in world_link_counts:
        scene.add_world(_make_chain_world(n_links))
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _final_joint_q(model, steps=60):
    solver = SolverFeatherPGS(model, dense_max_constraints=64)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0
    return state_0.joint_q.numpy()


def test_hetero_matrix_free_constructs(test, device):
    """Construct on worlds whose DOF counts differ."""
    solver = SolverFeatherPGS(_build_model([1, 3, 1, 3], device))
    np.testing.assert_array_equal(solver.world_dof_count.numpy(), [1, 3, 1, 3])


def test_homogeneous_matrix_free_constructs(test, device):
    """Construct on worlds with identical DOF counts."""
    solver = SolverFeatherPGS(_build_model([3, 3, 3, 3], device))
    np.testing.assert_array_equal(solver.world_dof_count.numpy(), [3, 3, 3, 3])


def test_hetero_worlds_match_isolated_worlds(test, device):
    """Simulate each world of a heterogeneous model exactly as it simulates alone."""
    combined = _final_joint_q(_build_model([1, 3], device))
    single = _final_joint_q(_build_model([1], device))
    triple = _final_joint_q(_build_model([3], device))
    np.testing.assert_allclose(combined, np.concatenate([single, triple]), rtol=0.0, atol=1.0e-5)


class TestFeatherPGSHeteroGuard(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSHeteroGuard, "test_hetero_matrix_free_constructs", test_hetero_matrix_free_constructs, devices=devices
)
add_function_test(
    TestFeatherPGSHeteroGuard,
    "test_homogeneous_matrix_free_constructs",
    test_homogeneous_matrix_free_constructs,
    devices=devices,
)
add_function_test(
    TestFeatherPGSHeteroGuard,
    "test_hetero_worlds_match_isolated_worlds",
    test_hetero_worlds_match_isolated_worlds,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main()
