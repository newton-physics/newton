# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke coverage for the pure-Warp fused FeatherPGS matrix-free backend."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _build_mixed_contact_model(device: wp.context.Device) -> newton.Model:
    """Build one articulated contact and one free-body contact in the same world."""
    builder = newton.ModelBuilder()
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=0.0, limit_kd=0.0, friction=0.0)
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    inertia = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)

    link = builder.add_link(
        xform=wp.transform(wp.vec3(-0.75, 0.0, 0.18), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        velocity_limit=0.25,
    )
    builder.add_articulation([joint])
    builder.add_shape_sphere(link, radius=0.2)

    # ``add_body`` creates a free rigid articulation during finalize. This
    # keeps the scene mixed without adding a duplicate free joint manually.
    free_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.75, 0.0, 0.18), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
    )
    builder.add_shape_sphere(free_body, radius=0.2)

    builder.add_ground_plane()
    return builder.finalize(device=device)


def _build_revolute_chain_world(dof_count: int, velocity_limit: float) -> newton.ModelBuilder:
    """Build a single-world revolute chain with a finite velocity limit."""
    builder = newton.ModelBuilder(gravity=0.0)
    inertia = wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2)
    parent = -1
    joints = []
    for _ in range(dof_count):
        link = builder.add_link(mass=1.0, inertia=inertia)
        joint = builder.add_joint_revolute(
            parent=parent,
            child=link,
            axis=wp.vec3(0.0, 0.0, 1.0),
            velocity_limit=velocity_limit,
        )
        joints.append(joint)
        parent = link
    builder.add_articulation(joints)
    return builder


def _build_heterogeneous_velocity_limit_model(device: wp.context.Device) -> newton.Model:
    """Build two worlds with different DOF counts."""
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_world(_build_revolute_chain_world(dof_count=1, velocity_limit=1.0))
    builder.add_world(_build_revolute_chain_world(dof_count=2, velocity_limit=1.0))
    return builder.finalize(device=device)


def _build_descendant_free_joint_model(device: wp.context.Device) -> newton.Model:
    """Build an articulation with a non-root free joint."""
    builder = newton.ModelBuilder(gravity=0.0)
    inertia = wp.mat33(0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.2)
    base = builder.add_link(mass=1.0, inertia=inertia)
    child = builder.add_link(mass=1.0, inertia=inertia)
    root_joint = builder.add_joint_revolute(parent=-1, child=base, axis=wp.vec3(0.0, 0.0, 1.0))
    descendant_free_joint = builder.add_joint_free(parent=base, child=child)
    builder.add_articulation([root_joint, descendant_free_joint])
    return builder.finalize(device=device)


class TestFeatherPGSFusedWarpSelector(unittest.TestCase):
    """Constructor-level coverage for the private fused-Warp API."""


def run_constructor_accepts_fused_warp(test: TestFeatherPGSFusedWarpSelector, device):
    model = _build_mixed_contact_model(device)
    solver = SolverFeatherPGS(
        model,
        pgs_iterations=1,
    )

    test.assertFalse(hasattr(solver, "pgs_mode"))
    test.assertFalse(hasattr(solver, "pgs_kernel"))
    test.assertIsNotNone(solver.mf_meta)
    test.assertIsNotNone(solver.impulses_vec3)
    test.assertEqual(solver._max_contact_triplets, solver._max_constraints_padded // 3)


def run_constructor_rejects_unsupported_combinations(test: TestFeatherPGSFusedWarpSelector, device):
    model = _build_mixed_contact_model(device)

    with test.assertRaisesRegex(ValueError, "requires enable_contact_friction=True"):
        SolverFeatherPGS(
            model,
            enable_contact_friction=False,
        )

    model = _build_descendant_free_joint_model(device)
    with test.assertRaisesRegex(NotImplementedError, "root joints attached to the world"):
        SolverFeatherPGS(model)


def run_constructor_pads_contact_triplets_internally(test: TestFeatherPGSFusedWarpSelector, device):
    model = _build_mixed_contact_model(device)
    solver = SolverFeatherPGS(model, max_constraints=32)

    test.assertEqual(solver.max_constraints, 32)
    test.assertEqual(solver._max_constraints_padded, 33)
    test.assertEqual(solver._max_contact_triplets, 11)
    test.assertEqual(solver.impulses.shape[1], 33)
    test.assertEqual(solver.diag.shape[1], 33)
    test.assertEqual(solver.rhs.shape[1], 33)


class TestFeatherPGSFusedWarpStep(unittest.TestCase):
    """Launch coverage for Zach's fused Warp tile kernel."""


def run_mixed_contact_one_step(test: TestFeatherPGSFusedWarpStep, device):
    model = _build_mixed_contact_model(device)
    solver = SolverFeatherPGS(
        model,
        pgs_iterations=1,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    qd = state_0.joint_qd.numpy()
    qd[0] = 2.0
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    contacts = model.collide(state_0)

    solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)

    test.assertTrue(np.isfinite(state_1.joint_q.numpy()).all())
    test.assertGreaterEqual(int(solver.dense_contact_row_count.numpy()[0]), 3)
    test.assertGreater(int(solver.constraint_count.numpy()[0]), int(solver.dense_contact_row_count.numpy()[0]))
    test.assertGreaterEqual(int(solver.mf_constraint_count.numpy()[0]), 3)


def run_heterogeneous_world_velocity_limit_step(test: TestFeatherPGSFusedWarpStep, device):
    model = _build_heterogeneous_velocity_limit_model(device)
    solver = SolverFeatherPGS(model, pgs_iterations=32)
    test.assertEqual(solver.max_world_dofs, 2)
    np.testing.assert_array_equal(solver.world_dof_start.numpy(), np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(solver.world_dof_count.numpy(), np.array([1, 2], dtype=np.int32))

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    qd = np.array([3.0, -4.0, 5.0], dtype=np.float32)
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver.step(state_0, state_1, control, None, 1.0 / 120.0)

    test.assertTrue(np.isfinite(state_1.joint_qd.numpy()).all())
    test.assertLess(np.max(np.abs(state_1.joint_qd.numpy())), np.max(np.abs(qd)))


def run_contact_metadata_capacity_mismatch_raises(test: TestFeatherPGSFusedWarpStep, device):
    model = _build_mixed_contact_model(device)
    model.rigid_contact_max = 1
    solver = SolverFeatherPGS(model, pgs_iterations=1)
    test.assertEqual(solver._contact_metadata_capacity, 1)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    contacts = newton.Contacts(2, 0, device=device)
    with test.assertRaisesRegex(ValueError, "rigid_contact_max=2"):
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
    with test.assertRaisesRegex(ValueError, "rigid_contact_max=2"):
        solver.update_contacts(contacts)


def run_update_contacts_rejects_different_contacts(test: TestFeatherPGSFusedWarpStep, device):
    model = _build_mixed_contact_model(device)
    solver = SolverFeatherPGS(model, pgs_iterations=1)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    contact_capacity = solver._contact_metadata_capacity
    contacts = newton.Contacts(contact_capacity, 0, device=device)
    other_contacts = newton.Contacts(contact_capacity, 0, device=device)

    solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)

    with test.assertRaisesRegex(ValueError, "same Contacts object"):
        solver.update_contacts(other_contacts)


class TestFeatherPGSBodyParentForce(unittest.TestCase):
    """Coverage for the optional ``State.body_parent_f`` output."""


def run_body_parent_force_static_pendulum(test: TestFeatherPGSBodyParentForce, device):
    builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
    builder.request_state_attributes("body_parent_f")

    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        -1,
        link,
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)

    solver = SolverFeatherPGS(model)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    solver.step(state_0, state_1, None, None, 5.0e-3)

    test.assertIsNotNone(state_1.body_parent_f)
    parent_f = state_1.body_parent_f.numpy()[0]
    weight = float(model.body_mass.numpy()[0]) * 9.81
    np.testing.assert_allclose(parent_f[:3], [0.0, 0.0, weight], rtol=1.0e-4, atol=1.0e-5)
    np.testing.assert_allclose(parent_f[3:6], [0.0, 0.0, 0.0], atol=1.0e-2)


devices = get_cuda_test_devices(mode="basic")

for device in devices:
    add_function_test(
        TestFeatherPGSFusedWarpSelector,
        "test_constructor_accepts_fused_warp",
        run_constructor_accepts_fused_warp,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFusedWarpSelector,
        "test_constructor_rejects_unsupported_combinations",
        run_constructor_rejects_unsupported_combinations,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFusedWarpSelector,
        "test_constructor_pads_contact_triplets_internally",
        run_constructor_pads_contact_triplets_internally,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFusedWarpStep,
        "test_mixed_contact_one_step",
        run_mixed_contact_one_step,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFusedWarpStep,
        "test_heterogeneous_world_velocity_limit_step",
        run_heterogeneous_world_velocity_limit_step,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFusedWarpStep,
        "test_contact_metadata_capacity_mismatch_raises",
        run_contact_metadata_capacity_mismatch_raises,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFusedWarpStep,
        "test_update_contacts_rejects_different_contacts",
        run_update_contacts_rejects_different_contacts,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSBodyParentForce,
        "test_body_parent_force_static_pendulum",
        run_body_parent_force_static_pendulum,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
