# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverFeatherPGS on multi-world models whose worlds have different DOF counts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _make_chain_world(n_links: int, root_height: float = 1.0, free_box: bool = False) -> newton.ModelBuilder:
    """A fixed-base serial chain of ``n_links`` revolute links (n_links DOFs), optionally with a free box."""
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    for _ in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.15, hy=0.03, hz=0.03)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, root_height), q=wp.quat_identity())
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
    if free_box:
        body = builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder


def _build_model(world_link_counts: list[int], device, root_height: float = 1.0, free_box=None) -> newton.Model:
    """One chain world per entry in ``world_link_counts``, plus a ground plane."""
    scene = newton.ModelBuilder()
    for world, n_links in enumerate(world_link_counts):
        scene.add_world(_make_chain_world(n_links, root_height, free_box is not None and free_box[world]))
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _final_joint_q(model, steps=60, response="immediate", from_fk=False):
    solver = SolverFeatherPGS(model, dense_max_constraints=64, articulated_contact_response=response)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    if from_fk:
        # Start from the joint-space pose instead of the builder's body placement.
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
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


def test_homogeneous_propagation_constructs(test, device):
    """Construct both propagation responses on worlds with identical DOF counts."""
    model = _build_model([3, 3, 3, 3], device)
    for response in ("propagation", "propagation-fused"):
        with test.subTest(response=response):
            solver = SolverFeatherPGS(model, articulated_contact_response=response)
            test.assertEqual(solver.articulated_contact_response, response)
            np.testing.assert_array_equal(solver.world_dof_count.numpy(), [3, 3, 3, 3])


def test_hetero_propagation_matches_isolated_worlds(test, device):
    """Simulate each world of a heterogeneous model with a propagation response exactly as it simulates alone.

    The chains hang low enough to hit the ground, so every world solves propagation contact rows.
    """

    def run(link_counts, free_box, response):
        model = _build_model(link_counts, device, 0.35, free_box)
        return _final_joint_q(model, response=response, from_fk=True)

    combined = run([1, 3], None, "propagation")
    single = run([1], None, "propagation")
    triple = run([3], None, "propagation")
    np.testing.assert_allclose(combined, np.concatenate([single, triple]), rtol=0.0, atol=1.0e-5)

    # The fused response needs one articulation size, but a free box still makes world DOF counts differ.
    combined = run([3, 3], [False, True], "propagation-fused")
    plain = run([3], [False], "propagation-fused")
    boxed = run([3], [True], "propagation-fused")
    np.testing.assert_allclose(combined, np.concatenate([plain, boxed]), rtol=0.0, atol=1.0e-5)


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
add_function_test(
    TestFeatherPGSHeteroGuard,
    "test_homogeneous_propagation_constructs",
    test_homogeneous_propagation_constructs,
    devices=devices,
)
add_function_test(
    TestFeatherPGSHeteroGuard,
    "test_hetero_propagation_matches_isolated_worlds",
    test_hetero_propagation_matches_isolated_worlds,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main()
