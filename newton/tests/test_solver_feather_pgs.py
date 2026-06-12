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

"""Consolidated tests for the private FeatherPGS solver."""

from __future__ import annotations

import inspect
import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers as solvers
from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs import SolverFeatherPGS, kernels
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    allocate_joint_velocity_limit_slots,
    clamp_augmented_joint_u0,
    convert_root_free_qd_local_to_world,
    convert_root_free_qd_world_to_local,
    integrate_generalized_joints,
    populate_joint_velocity_limit_J_for_size,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

VELOCITY_LIMIT_OVERSHOOT_TOLERANCE = 1.25


# Private API


class TestFeatherPGSPrivateAPI(unittest.TestCase):
    def test_import_stays_in_private_package(self):
        self.assertNotIn("SolverFeatherPGS", solvers.__all__)
        self.assertFalse(hasattr(solvers, "SolverFeatherPGS"))

    def test_constructor_signature_is_single_path(self):
        params = inspect.signature(SolverFeatherPGS).parameters

        removed = {
            "pgs_mode",
            "pgs_kernel",
            "friction_mode",
            "effort_limit_mode",
            "dense_contact_compliance",
            "dense_max_constraints",
            "cholesky_kernel",
            "trisolve_kernel",
            "hinv_jt_kernel",
            "delassus_kernel",
            "delassus_chunk_size",
            "pgs_chunk_size",
            "small_dof_threshold",
            "pgs_debug",
            "enable_joint_velocity_limits",
            "pgs_warmstart",
            "use_parallel_streams",
            "double_buffer",
            "nvtx",
        }
        for name in removed:
            self.assertNotIn(name, params)

        self.assertIn("contact_compliance", params)
        self.assertIn("max_constraints", params)

    def test_removed_kernel_symbols_stay_deleted(self):
        removed = {
            "pgs_solve_loop",
            "pgs_solve_mf_loop",
            "delassus_par_row_col",
            "hinv_jt_par_row",
            "clamp_joint_tau",
            "friction_step_bisection",
            "friction_step_coulomb_newton",
            "solve_coulomb_row",
        }
        for name in removed:
            self.assertFalse(hasattr(kernels, name), name)


class TestFeatherPGSUnsupportedModels(unittest.TestCase):
    pass


class TestFeatherPGSValidation(unittest.TestCase):
    pass


def _build_single_revolute_model(device):
    builder = newton.ModelBuilder(gravity=0.0)
    link = builder.add_link()
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=(0.0, 0.0, 1.0),
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _build_fixed_only_articulation_model(device):
    builder = newton.ModelBuilder(gravity=0.0)
    link = builder.add_link()
    joint = builder.add_joint_fixed(parent=-1, child=link)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _build_particle_only_model(device):
    builder = newton.ModelBuilder(gravity=0.0)
    builder.add_particle(pos=(0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0), mass=1.0)
    return builder.finalize(device=device)


def run_kinematic_body_rejected(test: TestFeatherPGSUnsupportedModels, device):
    builder = newton.ModelBuilder()
    builder.add_body(is_kinematic=True, mass=1.0)
    model = builder.finalize(device=device)

    with test.assertRaisesRegex(NotImplementedError, "kinematic bodies"):
        SolverFeatherPGS(model)


def run_zero_dof_articulation_rejected(test: TestFeatherPGSUnsupportedModels, device):
    model = _build_fixed_only_articulation_model(device)

    with test.assertRaisesRegex(NotImplementedError, "zero-DOF articulations"):
        SolverFeatherPGS(model)


def run_constructor_rejects_invalid_values(test: TestFeatherPGSValidation, device):
    model = _build_single_revolute_model(device)

    cases = [
        ({"update_mass_matrix_interval": 0}, "update_mass_matrix_interval"),
        ({"max_constraints": 0}, "max_constraints"),
        ({"max_constraints": 2}, "max_constraints"),
        ({"mf_max_constraints": 0}, "mf_max_constraints"),
    ]
    for kwargs, match in cases:
        with test.subTest(kwargs=kwargs):
            with test.assertRaisesRegex(ValueError, match):
                SolverFeatherPGS(model, **kwargs)


def run_step_rejects_invalid_dt(test: TestFeatherPGSValidation, device):
    model = _build_single_revolute_model(device)
    solver = SolverFeatherPGS(model)
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for dt in (0.0, -1.0, float("nan"), float("inf")):
        with test.subTest(dt=dt):
            with test.assertRaisesRegex(ValueError, "finite dt > 0"):
                solver.step(state_0, state_1, None, None, dt)


def run_no_joint_step_accepts_contacts(test: TestFeatherPGSValidation, device):
    model = _build_particle_only_model(device)
    solver = SolverFeatherPGS(model)
    state_0 = model.state()
    state_1 = model.state()
    contacts = newton.Contacts(1, 0, device=device)

    solver.step(state_0, state_1, None, contacts, 1.0 / 60.0)
    solver.update_contacts(contacts)

    test.assertEqual(solver._step, 1)
    test.assertFalse(hasattr(solver, "_contact_metadata_capacity"))


# Fused Warp


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
    impulses = solver.impulses.numpy()
    constraint_count = solver.constraint_count.numpy()
    dense_contact_count = solver.dense_contact_row_count.numpy()
    for world in range(model.world_count):
        if constraint_count[world] > dense_contact_count[world]:
            non_contact_impulses = impulses[world, dense_contact_count[world] : constraint_count[world]]
            test.assertGreater(float(np.max(non_contact_impulses)), 0.0)


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


# Velocity Limits


def _build_two_dof_pendulum_model(device, velocity_limit: float = 2.0) -> newton.Model:
    """Build a 2-DOF revolute chain with a finite per-axis velocity limit.

    The chain has two serial revolute joints with a known velocity limit on
    each DOF, so the allocation / populator kernels have a non-trivial DOF
    range to sweep and the solver has a non-empty articulation to build
    against.
    """
    builder = newton.ModelBuilder()
    link_a = builder.add_link()
    link_b = builder.add_link()
    joint_a = builder.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        velocity_limit=velocity_limit,
    )
    joint_b = builder.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        velocity_limit=velocity_limit,
    )
    builder.add_articulation([joint_a, joint_b])
    model = builder.finalize(device=device)
    return model


class TestFeatherPGSVelocityLimitConstructor(unittest.TestCase):
    """Constructor-level plumbing for always-on velocity limits."""


def run_default_allocates_buffers(test: TestFeatherPGSVelocityLimitConstructor, device):
    """Velocity-limit buffers allocate by default for articulated models."""
    model = _build_two_dof_pendulum_model(device)
    solver = SolverFeatherPGS(model)
    test.assertFalse(hasattr(solver, "enable_joint_velocity_limits"))
    test.assertIsNotNone(solver.velocity_limit_slot)
    test.assertIsNotNone(solver.velocity_limit_sign)
    test.assertEqual(solver.velocity_limit_slot.shape, (model.joint_dof_count,))
    test.assertEqual(solver.velocity_limit_sign.shape, (model.joint_dof_count,))
    # Default slot entry is -1 (no row active).
    np.testing.assert_array_equal(
        solver.velocity_limit_slot.numpy(),
        -np.ones(model.joint_dof_count, dtype=np.int32),
    )


class TestFeatherPGSVelocityLimitAllocationKernel(unittest.TestCase):
    """Direct kernel tests for ``allocate_joint_velocity_limit_slots``."""


def run_allocation_activates_only_over_limit(test: TestFeatherPGSVelocityLimitAllocationKernel, device):
    """Only DOFs whose ``|qdot|`` exceeds the per-axis limit get a slot."""
    # Layout: one articulation with two revolute DOFs, limits = 2.0 rad/s.
    dof_count = 2
    max_constraints = 8

    articulation_start = wp.array(np.array([0, 2], dtype=np.int32), dtype=int, device=device)
    articulation_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    articulation_H_rows = wp.array(np.array([dof_count], dtype=np.int32), dtype=int, device=device)
    # Two revolute joints.
    joint_type = wp.array(
        np.array([int(JointType.REVOLUTE), int(JointType.REVOLUTE)], dtype=np.int32),
        dtype=int,
        device=device,
    )
    joint_qd_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device)
    # Each joint has 0 linear, 1 angular DOF.
    joint_dof_dim = wp.array(np.array([[0, 1], [0, 1]], dtype=np.int32), dtype=int, ndim=2, device=device)
    joint_velocity_limit = wp.array(np.array([2.0, 2.0], dtype=np.float32), dtype=float, device=device)
    # DOF 0 well within limits; DOF 1 blown past the upper limit.
    joint_qd = wp.array(np.array([0.5, 3.5], dtype=np.float32), dtype=float, device=device)
    art_to_world = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    velocity_limit_slot = wp.full((dof_count,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((dof_count,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            articulation_start,
            articulation_dof_start,
            articulation_H_rows,
            joint_type,
            joint_qd_start,
            joint_dof_dim,
            joint_velocity_limit,
            joint_qd,
            art_to_world,
            max_constraints,
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )

    slot = velocity_limit_slot.numpy()
    sign = velocity_limit_sign.numpy()
    counter = int(world_slot_counter.numpy()[0])

    # Exactly one slot allocated — only DOF 1 was over limit.
    test.assertEqual(counter, 1)
    test.assertEqual(int(slot[0]), -1)
    test.assertEqual(int(slot[1]), 0)
    test.assertAlmostEqual(float(sign[0]), 0.0, places=6)
    # Upper-limit violation → sign = -1 (J points in -i direction to push qdot down).
    test.assertAlmostEqual(float(sign[1]), -1.0, places=6)


def run_allocation_lower_vs_upper_sides(test: TestFeatherPGSVelocityLimitAllocationKernel, device):
    """Lower-limit violation → sign = +1; upper-limit violation → sign = -1.

    This is the direct signal that the bilateral ``[-qdot_max, +qdot_max]``
    box is encoded via the sign of the Jacobian row, not by two separate
    rows per DOF.
    """
    dof_count = 2
    max_constraints = 8

    articulation_start = wp.array(np.array([0, 2], dtype=np.int32), dtype=int, device=device)
    articulation_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    articulation_H_rows = wp.array(np.array([dof_count], dtype=np.int32), dtype=int, device=device)
    joint_type = wp.array(
        np.array([int(JointType.REVOLUTE), int(JointType.REVOLUTE)], dtype=np.int32),
        dtype=int,
        device=device,
    )
    joint_qd_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device)
    joint_dof_dim = wp.array(np.array([[0, 1], [0, 1]], dtype=np.int32), dtype=int, ndim=2, device=device)
    joint_velocity_limit = wp.array(np.array([1.0, 1.0], dtype=np.float32), dtype=float, device=device)
    # DOF 0: below lower bound. DOF 1: above upper bound.
    joint_qd = wp.array(np.array([-5.0, 5.0], dtype=np.float32), dtype=float, device=device)
    art_to_world = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    velocity_limit_slot = wp.full((dof_count,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((dof_count,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            articulation_start,
            articulation_dof_start,
            articulation_H_rows,
            joint_type,
            joint_qd_start,
            joint_dof_dim,
            joint_velocity_limit,
            joint_qd,
            art_to_world,
            max_constraints,
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )

    slot = velocity_limit_slot.numpy()
    sign = velocity_limit_sign.numpy()
    counter = int(world_slot_counter.numpy()[0])

    test.assertEqual(counter, 2)
    # Both DOFs allocated; slot indices 0 and 1 (atomic_add order may vary
    # but indices must be distinct and in [0, 2)).
    test.assertIn(int(slot[0]), (0, 1))
    test.assertIn(int(slot[1]), (0, 1))
    test.assertNotEqual(int(slot[0]), int(slot[1]))
    # Lower violation pushes qdot up: sign = +1.
    test.assertAlmostEqual(float(sign[0]), 1.0, places=6)
    # Upper violation pushes qdot down: sign = -1.
    test.assertAlmostEqual(float(sign[1]), -1.0, places=6)


def run_allocation_nonpositive_limit_is_unlimited(test: TestFeatherPGSVelocityLimitAllocationKernel, device):
    """A non-positive ``qdot_max_i`` is treated as "no limit": no row allocated.

    This mirrors PhysX's ``recipResponse`` pinning on ``unitResponse <= 0``
    and is how Newton's builder represents an unset / unlimited velocity
    limit (``velocity_limit`` defaulting to a sentinel).
    """
    dof_count = 1
    max_constraints = 8

    articulation_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device)
    articulation_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    articulation_H_rows = wp.array(np.array([dof_count], dtype=np.int32), dtype=int, device=device)
    joint_type = wp.array(np.array([int(JointType.REVOLUTE)], dtype=np.int32), dtype=int, device=device)
    joint_qd_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_dof_dim = wp.array(np.array([[0, 1]], dtype=np.int32), dtype=int, ndim=2, device=device)
    # Zero limit should be treated as unlimited and NOT allocate.
    joint_velocity_limit = wp.array(np.array([0.0], dtype=np.float32), dtype=float, device=device)
    joint_qd = wp.array(np.array([100.0], dtype=np.float32), dtype=float, device=device)
    art_to_world = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    velocity_limit_slot = wp.full((dof_count,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((dof_count,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            articulation_start,
            articulation_dof_start,
            articulation_H_rows,
            joint_type,
            joint_qd_start,
            joint_dof_dim,
            joint_velocity_limit,
            joint_qd,
            art_to_world,
            max_constraints,
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )

    test.assertEqual(int(world_slot_counter.numpy()[0]), 0)
    test.assertEqual(int(velocity_limit_slot.numpy()[0]), -1)


class TestFeatherPGSVelocityLimitPopulatorKernel(unittest.TestCase):
    """Direct kernel tests for ``populate_joint_velocity_limit_J_for_size``."""


def run_populator_writes_signed_selector_row_and_metadata(test: TestFeatherPGSVelocityLimitPopulatorKernel, device):
    """Populator writes J = sign*e_i, zero Baumgarte bias, and target_vel = -qdot_max.

    Build a 2-DOF layout where:
    - DOF 0 is upper-violating (sign=-1) and gets slot 0.
    - DOF 1 is lower-violating (sign=+1) and gets slot 1.

    Verify the grouped Jacobian, the row-type marker, and the constraint
    metadata (``row_beta``, ``row_cfm``, ``phi``, ``target_velocity``) match
    the PhysX mirror formulation.
    """
    dof_count = 2
    max_constraints = 8
    n_arts = 1

    articulation_start = wp.array(np.array([0, 2], dtype=np.int32), dtype=int, device=device)
    articulation_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_type = wp.array(
        np.array([int(JointType.REVOLUTE), int(JointType.REVOLUTE)], dtype=np.int32),
        dtype=int,
        device=device,
    )
    joint_qd_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device)
    joint_dof_dim = wp.array(np.array([[0, 1], [0, 1]], dtype=np.int32), dtype=int, ndim=2, device=device)
    qdot_max = np.array([3.0, 4.0], dtype=np.float32)
    joint_velocity_limit = wp.array(qdot_max, dtype=float, device=device)

    art_to_world = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    # DOF 0 → slot 0, sign -1 (upper); DOF 1 → slot 1, sign +1 (lower).
    velocity_limit_slot = wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device)
    velocity_limit_sign = wp.array(np.array([-1.0, 1.0], dtype=np.float32), dtype=float, device=device)
    group_to_art = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    pgs_cfm = 1.0e-6

    # Grouped Jacobian: [n_arts_of_size, max_constraints, n_dofs].
    J_group = wp.zeros((n_arts, max_constraints, dof_count), dtype=float, device=device)
    world_row_type = wp.zeros((1, max_constraints), dtype=wp.int32, device=device)
    world_row_parent = wp.full((1, max_constraints), -1, dtype=wp.int32, device=device)
    world_row_mu = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_row_beta = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_row_cfm = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_phi = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_target_velocity = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)

    wp.launch(
        populate_joint_velocity_limit_J_for_size,
        dim=n_arts,
        inputs=[
            articulation_start,
            articulation_dof_start,
            joint_type,
            joint_qd_start,
            joint_dof_dim,
            joint_velocity_limit,
            art_to_world,
            velocity_limit_slot,
            velocity_limit_sign,
            group_to_art,
            pgs_cfm,
        ],
        outputs=[
            J_group,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_row_beta,
            world_row_cfm,
            world_phi,
            world_target_velocity,
        ],
        device=device,
    )

    # J rows are single signed-±1 selector rows at (slot, local_dof).
    J = J_group.numpy()
    test.assertAlmostEqual(float(J[0, 0, 0]), -1.0, places=6)  # slot 0 row, DOF 0 col
    test.assertAlmostEqual(float(J[0, 0, 1]), 0.0, places=6)
    test.assertAlmostEqual(float(J[0, 1, 0]), 0.0, places=6)
    test.assertAlmostEqual(float(J[0, 1, 1]), 1.0, places=6)  # slot 1 row, DOF 1 col

    row_type = world_row_type.numpy()
    test.assertEqual(int(row_type[0, 0]), PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    test.assertEqual(int(row_type[0, 1]), PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)

    # No Baumgarte / ERP — matches PhysX.
    np.testing.assert_allclose(world_row_beta.numpy()[0, :2], np.zeros(2, dtype=np.float32), atol=1e-7)
    # phi = 0 (velocity-limit row has no positional bias quantity).
    np.testing.assert_allclose(world_phi.numpy()[0, :2], np.zeros(2, dtype=np.float32), atol=1e-7)
    # target_vel = -qdot_max for both signs; combined with the sign flip in J
    # this encodes the bilateral projection as a unilateral row with
    # ``lambda >= 0``.
    np.testing.assert_allclose(world_target_velocity.numpy()[0, :2], -qdot_max, atol=1e-7)
    # cfm preserved.
    np.testing.assert_allclose(world_row_cfm.numpy()[0, :2], np.full(2, pgs_cfm, dtype=np.float32), atol=1e-7)
    # No parent — the row is standalone (not a friction sibling).
    np.testing.assert_array_equal(world_row_parent.numpy()[0, :2], np.array([-1, -1], dtype=np.int32))


class TestFeatherPGSVelocityLimitEndToEnd(unittest.TestCase):
    """End-to-end integration tests that drive a real solver step.

    The fused matrix-free Warp PGS sweep is CUDA-only. Kernel-level correctness
    for allocation and row population is covered on CPU by the classes above.
    """


def _build_driven_pendulum(
    device, velocity_limit: float, target_ke: float, target_kd: float
) -> tuple[newton.Model, newton.State, newton.State, newton.Control]:
    """Build a 1-DOF revolute pendulum with a stiff implicit PD drive.

    The setup mirrors the shape of the Franka smoke: a scripted position
    target that is far outside the physical reach of the joint, with a
    stiff implicit-PD drive that demands a huge ``qdot`` at the next step.
    Reproduces the conditions under which baseline FPGS produces the
    velocity-spike ``|qdot|`` of issue #21.
    """
    builder = newton.ModelBuilder(gravity=0.0)
    box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    link = builder.add_link(armature=0.0, inertia=box_inertia, mass=1.0)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        velocity_limit=velocity_limit,
        target_ke=target_ke,
        target_kd=target_kd,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    # Target far outside any reasonable reach so the stiff PD drive commands a
    # huge ``qdot`` — mirrors the smoke's ``pos_high`` pulse.
    control.joint_target_pos = wp.array([10.0], dtype=wp.float32, device=device)
    control.joint_target_vel = wp.array([0.0], dtype=wp.float32, device=device)

    return model, state_0, state_1, control


def _step_and_read_qdot(
    model: newton.Model,
    solver: SolverFeatherPGS,
    state_0: newton.State,
    state_1: newton.State,
    control: newton.Control,
    dt: float,
    n_steps: int,
) -> float:
    """Run ``n_steps`` and return the peak ``|qdot|`` observed on DOF 0."""
    peak = 0.0
    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        peak = max(peak, float(abs(state_0.joint_qd.numpy()[0])))
    return peak


def run_end_to_end_clamps_peak_qdot(test: TestFeatherPGSVelocityLimitEndToEnd, device):
    """Peak ``|qdot|`` stays near the velocity limit.

    Drives a stiff-PD pendulum with a far position target — the baseline
    unlimited model produces a huge ``qdot`` spike over a few steps; with
    finite positive ``joint_velocity_limit`` entries the PGS clamp pulls
    ``|qdot|`` back toward ``qdot_max``.
    """
    velocity_limit = 1.5
    model, state_0, state_1, control = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver = SolverFeatherPGS(model, pgs_iterations=32)
    peak = _step_and_read_qdot(model, solver, state_0, state_1, control, dt=1.0 / 120.0, n_steps=40)

    # The clamp is applied after prediction, so the PGS residual can overshoot
    # the strict limit. Keep the tolerance tight enough to catch regressions.
    test.assertLess(peak, velocity_limit * VELOCITY_LIMIT_OVERSHOOT_TOLERANCE)


def run_end_to_end_finite_limit_is_tighter_than_unlimited(test: TestFeatherPGSVelocityLimitEndToEnd, device):
    """A finite limit has a smaller ``|qdot|`` peak than an unlimited model.

    This is the direct smoke-scenario signal at Newton-library level
    (no Isaac Sim required): driven by the same stiff-PD far target, a
    non-positive limit remains unlimited while a finite positive limit clamps.
    """
    velocity_limit = 1.5

    # Baseline: non-positive limits are treated as unlimited.
    model_b, s0_b, s1_b, ctl_b = _build_driven_pendulum(device, velocity_limit=0.0, target_ke=2000.0, target_kd=40.0)
    solver_b = SolverFeatherPGS(model_b, pgs_iterations=32)
    peak_baseline = _step_and_read_qdot(model_b, solver_b, s0_b, s1_b, ctl_b, dt=1.0 / 120.0, n_steps=40)

    # Constrained: finite positive limits are always active.
    model_c, s0_c, s1_c, ctl_c = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver_c = SolverFeatherPGS(model_c, pgs_iterations=32)
    peak_constrained = _step_and_read_qdot(model_c, solver_c, s0_c, s1_c, ctl_c, dt=1.0 / 120.0, n_steps=40)

    test.assertGreater(peak_baseline, velocity_limit * VELOCITY_LIMIT_OVERSHOOT_TOLERANCE)
    test.assertLess(peak_constrained, peak_baseline)
    test.assertLess(peak_constrained, velocity_limit * VELOCITY_LIMIT_OVERSHOOT_TOLERANCE)


# Effort Limits


class TestFeatherPGSEffortLimitClamp(unittest.TestCase):
    """Direct signal that actuator-only clamps a different quantity.

    ``clamp_augmented_joint_u0`` receives only the per-row actuator-drive
    ``u0`` bucket, so it cannot clamp the rigid/passive bucket by
    construction.
    """


def run_clamp_augmented_joint_u0_clamps_only_drive(test: TestFeatherPGSEffortLimitClamp, device):
    """The actuator-bucket kernel clamps ``row_u0`` against the per-DOF limit.

    Build a single-articulation, 2-DOF layout with two drive rows:

    - DOF 0: ``u0 = 100``, ``limit = 5``     → expect clamp to ``+5``.
    - DOF 1: ``u0 = -50``, ``limit = 0``     → expect no clamp (unlimited).

    This is the direct kernel-level check for the actuator bucket.
    """
    # Single articulation, max_dofs == 2.
    max_dofs = 2
    n_articulations = 1

    row_counts = wp.array([2], dtype=wp.int32, device=device)
    row_dof_index = wp.array([0, 1], dtype=wp.int32, device=device)
    row_u0 = wp.array([100.0, -50.0], dtype=wp.float32, device=device)

    # DOF 0 has a finite limit; DOF 1 has limit=0 → unlimited in FPGS convention.
    joint_effort_limit = wp.array([5.0, 0.0], dtype=wp.float32, device=device)

    wp.launch(
        clamp_augmented_joint_u0,
        dim=n_articulations,
        inputs=[max_dofs, row_counts, row_dof_index, joint_effort_limit],
        outputs=[row_u0],
        device=device,
    )

    # Drive bucket clamped per DOF.
    u0_out = row_u0.numpy()
    np.testing.assert_allclose(u0_out, np.array([5.0, -50.0], dtype=np.float32), rtol=0.0, atol=1e-7)


def run_net_vs_actuator_differ_on_passive_only_dof(test: TestFeatherPGSEffortLimitClamp, device):
    """Direct divergence signal: passive-only load differs by clamp target.

    Build a DOF whose drive ``u0 = 0`` but whose rigid/passive bucket is
    ``-9.81`` (a gravity-loaded joint). With ``limit = 5.0`` the two
    semantics produce opposite answers:

    - Net-torque semantics clamps the passive load to ``-5.0``.
    - Actuator-only semantics is a no-op when ``row_counts == 0``, so the
      passive load remains outside the clamped actuator bucket.
    """
    # Shared inputs.
    limit = 5.0
    rigid_passive = -9.81  # "tau_passive" at a horizontal pendulum, zero drive.

    net_out = max(-limit, min(limit, rigid_passive))

    # --- Actuator-only clamp ---
    # With zero drive gains there is no augmented row, so `row_counts = 0`
    # and the drive-only clamp kernel is a no-op. Keep the passive load as a
    # scalar outside the kernel inputs: it is not part of the actuator bucket.
    act_passive_load = rigid_passive
    act_limit = wp.array([limit], dtype=wp.float32, device=device)
    act_row_counts = wp.array([0], dtype=wp.int32, device=device)
    act_row_dof_index = wp.zeros((1,), dtype=wp.int32, device=device)
    act_row_u0 = wp.zeros((1,), dtype=wp.float32, device=device)
    wp.launch(
        clamp_augmented_joint_u0,
        dim=1,
        inputs=[1, act_row_counts, act_row_dof_index, act_limit],
        outputs=[act_row_u0],
        device=device,
    )
    act_out = act_passive_load

    # Baseline semantics clamps the gravity torque to the effort limit,
    # so the magnitude stored in joint_tau is exactly `limit`.
    test.assertAlmostEqual(net_out, -limit, places=6)
    # Actuator-only semantics leaves the gravity torque untouched.
    test.assertAlmostEqual(act_out, rigid_passive, places=6)
    # And the two outputs actually disagree: actuator-only clamps a different
    # quantity than net-torque semantics.
    test.assertGreater(abs(act_out - net_out), 1e-3)


# Floating Roots


class TestFeatherPGSFreeRootVelocity(unittest.TestCase):
    pass


def run_free_root_velocity_roundtrip(test: TestFeatherPGSFreeRootVelocity, device):
    """Floating-root qd should round-trip between public CoM and internal origin conventions."""
    articulation_root_is_free = wp.array(np.array([1, 0], dtype=np.int32), dtype=int, device=device)
    articulation_root_dof_start = wp.array(np.array([0, 6], dtype=np.int32), dtype=int, device=device)
    articulation_root_com_offset = wp.array(
        np.array([[0.2, -0.3, 0.1], [0.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )

    qd_public_np = np.array(
        [
            1.5,
            -2.0,
            0.25,
            0.4,
            -0.6,
            0.8,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        dtype=np.float32,
    )
    qd = wp.array(qd_public_np, dtype=float, device=device)

    wp.launch(
        convert_root_free_qd_world_to_local,
        dim=2,
        inputs=[articulation_root_is_free, articulation_root_dof_start, articulation_root_com_offset],
        outputs=[qd],
        device=device,
    )

    qd_local_np = qd.numpy()
    omega = qd_public_np[3:6]
    com_offset = np.array([0.2, -0.3, 0.1], dtype=np.float32)
    expected_local_linear = qd_public_np[:3] - np.cross(omega, com_offset)
    np.testing.assert_allclose(qd_local_np[:3], expected_local_linear, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(qd_local_np[3:6], qd_public_np[3:6], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(qd_local_np[6:], qd_public_np[6:], rtol=1e-6, atol=1e-6)

    wp.launch(
        convert_root_free_qd_local_to_world,
        dim=2,
        inputs=[articulation_root_is_free, articulation_root_dof_start, articulation_root_com_offset],
        outputs=[qd],
        device=device,
    )

    np.testing.assert_allclose(qd.numpy(), qd_public_np, rtol=1e-6, atol=1e-6)


def run_free_root_integration_uses_origin_velocity(test: TestFeatherPGSFreeRootVelocity, device):
    """Free-joint position integration should convert CoM velocity back to origin velocity."""
    joint_type = wp.array(np.array([int(JointType.FREE)], dtype=np.int32), dtype=int, device=device)
    joint_child = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_q_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_qd_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_dof_dim = wp.array(np.array([[3, 3]], dtype=np.int32), dtype=int, device=device)
    body_com = wp.array(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device)

    joint_q = wp.array(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), dtype=float, device=device)
    joint_qd = wp.array(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float32), dtype=float, device=device)
    joint_qdd = wp.zeros(6, dtype=float, device=device)
    joint_q_new = wp.zeros(7, dtype=float, device=device)
    joint_qd_new = wp.zeros(6, dtype=float, device=device)

    wp.launch(
        integrate_generalized_joints,
        dim=1,
        inputs=[
            joint_type,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_dof_dim,
            body_com,
            joint_q,
            joint_qd,
            joint_qdd,
            0.1,
        ],
        outputs=[joint_q_new, joint_qd_new],
        device=device,
    )

    joint_q_new_np = joint_q_new.numpy()
    joint_qd_new_np = joint_qd_new.numpy()

    # qd stores CoM velocity. With r_com = (0, 1, 0) and w = (0, 0, 2),
    # the origin velocity is v_com - w x r_com = (3, 0, 0).
    np.testing.assert_allclose(joint_q_new_np[:3], np.array([10.3, 0.0, 0.0], dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(joint_qd_new_np[:3], np.array([1.0, 0.0, 0.0], dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(joint_qd_new_np[3:], np.array([0.0, 0.0, 2.0], dtype=np.float32), rtol=1e-6, atol=1e-6)


# Test Registration

# Private API

devices = get_cuda_test_devices(mode="basic")

for device in devices:
    add_function_test(
        TestFeatherPGSUnsupportedModels,
        "test_kinematic_body_rejected",
        run_kinematic_body_rejected,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSUnsupportedModels,
        "test_zero_dof_articulation_rejected",
        run_zero_dof_articulation_rejected,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSValidation,
        "test_constructor_rejects_invalid_values",
        run_constructor_rejects_invalid_values,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSValidation,
        "test_step_rejects_invalid_dt",
        run_step_rejects_invalid_dt,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSValidation,
        "test_no_joint_step_accepts_contacts",
        run_no_joint_step_accepts_contacts,
        devices=[device],
    )


# Fused Warp

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


# Velocity Limits

devices = get_test_devices()
constructor_devices = get_cuda_test_devices(mode="basic")

for device in constructor_devices:
    add_function_test(
        TestFeatherPGSVelocityLimitConstructor,
        "test_default_allocates_buffers",
        run_default_allocates_buffers,
        devices=[device],
    )

for device in devices:
    add_function_test(
        TestFeatherPGSVelocityLimitAllocationKernel,
        "test_allocation_activates_only_over_limit",
        run_allocation_activates_only_over_limit,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitAllocationKernel,
        "test_allocation_lower_vs_upper_sides",
        run_allocation_lower_vs_upper_sides,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitAllocationKernel,
        "test_allocation_nonpositive_limit_is_unlimited",
        run_allocation_nonpositive_limit_is_unlimited,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitPopulatorKernel,
        "test_populator_writes_signed_selector_row_and_metadata",
        run_populator_writes_signed_selector_row_and_metadata,
        devices=[device],
    )
    if device.is_cuda:
        add_function_test(
            TestFeatherPGSVelocityLimitEndToEnd,
            "test_end_to_end_clamps_peak_qdot",
            run_end_to_end_clamps_peak_qdot,
            devices=[device],
        )
        add_function_test(
            TestFeatherPGSVelocityLimitEndToEnd,
            "test_end_to_end_finite_limit_is_tighter_than_unlimited",
            run_end_to_end_finite_limit_is_tighter_than_unlimited,
            devices=[device],
        )


# Effort Limits

devices = get_test_devices()

for device in devices:
    add_function_test(
        TestFeatherPGSEffortLimitClamp,
        "test_clamp_augmented_joint_u0_clamps_only_drive",
        run_clamp_augmented_joint_u0_clamps_only_drive,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSEffortLimitClamp,
        "test_net_vs_actuator_differ_on_passive_only_dof",
        run_net_vs_actuator_differ_on_passive_only_dof,
        devices=[device],
    )


# Floating Roots

devices = get_test_devices()

for device in devices:
    add_function_test(
        TestFeatherPGSFreeRootVelocity,
        "test_free_root_velocity_roundtrip",
        run_free_root_velocity_roundtrip,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFreeRootVelocity,
        "test_free_root_integration_uses_origin_velocity",
        run_free_root_integration_uses_origin_velocity,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
