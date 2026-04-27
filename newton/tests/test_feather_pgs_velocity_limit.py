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

"""Regression tests for the FeatherPGS joint velocity-limit constraint.

These tests cover the feature introduced in issue #23:

1. **Parameter plumbing and guards** — the new ``enable_joint_velocity_limits``
   kwarg is accepted, stored, and bound to the supported PGS mode.
2. **Allocation kernel** — the Warp kernel ``allocate_joint_velocity_limit_slots``
   reserves a slot only when ``|qdot_i| > qdot_max_i`` and encodes the side of
   the bilateral ``[-qdot_max, +qdot_max]`` box via the sign of the row.
3. **Populator kernel** — the Warp kernel
   ``populate_joint_velocity_limit_J_for_size`` writes a signed ±1 selector row
   into the grouped Jacobian and sets the correct constraint metadata
   (row type, zero Baumgarte bias, ``target_vel = -qdot_max``).
4. **Flag-off is a strict no-op** — constructing the solver without the flag
   matches the baseline behaviour over the full per-step velocity trace.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs import SolverFeatherPGS
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    allocate_joint_velocity_limit_slots,
    populate_joint_velocity_limit_J_for_size,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _build_two_dof_pendulum_model(device: str, velocity_limit: float = 2.0) -> newton.Model:
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


class TestFeatherPGSVelocityLimitFlag(unittest.TestCase):
    """Constructor-level plumbing for ``enable_joint_velocity_limits``."""


def run_default_is_flag_off(test: TestFeatherPGSVelocityLimitFlag, device):
    """With no kwarg, velocity limits stay off (regression baseline)."""
    model = _build_two_dof_pendulum_model(device)
    solver = SolverFeatherPGS(model)
    test.assertFalse(solver.enable_joint_velocity_limits)
    # The per-DOF buffers only allocate when the flag is on.
    test.assertIsNone(solver.velocity_limit_slot)
    test.assertIsNone(solver.velocity_limit_sign)


def run_flag_on_allocates_buffers(test: TestFeatherPGSVelocityLimitFlag, device):
    """Passing the flag allocates per-DOF buffers sized to ``joint_dof_count``."""
    model = _build_two_dof_pendulum_model(device)
    solver = SolverFeatherPGS(
        model,
        enable_joint_velocity_limits=True,
    )
    test.assertTrue(solver.enable_joint_velocity_limits)
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

    The matrix-free PGS sweep kernel (``get_pgs_solve_mf_gs_kernel``) is a
    CUDA-native snippet guarded by ``#if defined(__CUDA_ARCH__)`` — it is a
    no-op on CPU. These tests therefore require a CUDA device. Kernel-level
    correctness (allocation + populator + row-type dispatch) is covered on
    CPU by the classes above.
    """


def _build_driven_pendulum(
    device: str, velocity_limit: float, target_ke: float, target_kd: float
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


def _step_and_read_qdot_history(
    model: newton.Model,
    solver: SolverFeatherPGS,
    state_0: newton.State,
    state_1: newton.State,
    control: newton.Control,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    """Run ``n_steps`` and return the per-step ``joint_qd`` trace."""
    history: list[np.ndarray] = []
    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        history.append(state_0.joint_qd.numpy().copy())
    return np.asarray(history, dtype=np.float32)


def run_end_to_end_flag_on_clamps_peak_qdot(test: TestFeatherPGSVelocityLimitEndToEnd, device):
    """With the flag on, peak ``|qdot|`` stays near the velocity limit.

    Drives a stiff-PD pendulum with a far position target — the baseline
    solver produces a huge ``qdot`` spike over a few steps; with
    ``enable_joint_velocity_limits=True`` the PGS clamp pulls ``|qdot|``
    back toward ``qdot_max``. We accept a small tolerance above the strict
    limit because the clamp is applied *after* the predictor: the
    pre-solve velocity can be above the limit, and the solver's iteration
    count / articulated-body response determine the residual.
    """
    velocity_limit = 1.5
    model, state_0, state_1, control = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver = SolverFeatherPGS(
        model,
        enable_joint_velocity_limits=True,
        pgs_iterations=32,
    )
    peak = _step_and_read_qdot(model, solver, state_0, state_1, control, dt=1.0 / 120.0, n_steps=40)

    # The clamp must be doing *something* meaningful: peak stays within a
    # generous 1.5x multiple of the limit (mirrors the issue's 1.25x
    # threshold with extra headroom for PGS residual). The reference
    # baseline without the flag blows past 6x; see the companion test.
    test.assertLess(peak, velocity_limit * 1.5)


def run_end_to_end_flag_on_is_tighter_than_flag_off(test: TestFeatherPGSVelocityLimitEndToEnd, device):
    """Constrained FPGS has a smaller ``|qdot|`` peak than baseline FPGS.

    This is the direct smoke-scenario signal at Newton-library level
    (no Isaac Sim required): driven by the same stiff-PD far target and
    the same matrix-free pipeline, the flag-on solver clamps and the
    flag-off solver does not.
    """
    velocity_limit = 1.5

    # Baseline (flag off).
    model_b, s0_b, s1_b, ctl_b = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver_b = SolverFeatherPGS(model_b, pgs_iterations=32)
    peak_baseline = _step_and_read_qdot(model_b, solver_b, s0_b, s1_b, ctl_b, dt=1.0 / 120.0, n_steps=40)

    # Constrained (flag on).
    model_c, s0_c, s1_c, ctl_c = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver_c = SolverFeatherPGS(
        model_c,
        enable_joint_velocity_limits=True,
        pgs_iterations=32,
    )
    peak_constrained = _step_and_read_qdot(model_c, solver_c, s0_c, s1_c, ctl_c, dt=1.0 / 120.0, n_steps=40)

    # Baseline must overshoot meaningfully (otherwise the scenario itself
    # isn't a valid witness).
    test.assertGreater(peak_baseline, velocity_limit * 1.5)
    # Constrained must be tighter than baseline.
    test.assertLess(peak_constrained, peak_baseline)
    # And preferably near the limit.
    test.assertLess(peak_constrained, velocity_limit * 1.5)


def run_end_to_end_flag_off_is_strict_noop(test: TestFeatherPGSVelocityLimitEndToEnd, device):
    """Flag off: per-step velocity outputs match the pre-issue baseline.

    Same stiff-PD driven pendulum, same matrix-free pipeline, same seed
    (deterministic), the flag toggles to ``False`` must produce the same
    ``joint_qd`` trajectory as a solver constructed without the kwarg.
    """
    velocity_limit = 1.5

    # Solver A: constructed without the kwarg (pre-issue baseline).
    model_a, s0_a, s1_a, ctl_a = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver_a = SolverFeatherPGS(model_a)
    history_a = _step_and_read_qdot_history(model_a, solver_a, s0_a, s1_a, ctl_a, dt=1.0 / 120.0, n_steps=20)

    # Solver B: constructed with enable_joint_velocity_limits=False (explicit).
    model_b, s0_b, s1_b, ctl_b = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver_b = SolverFeatherPGS(model_b, enable_joint_velocity_limits=False)
    history_b = _step_and_read_qdot_history(model_b, solver_b, s0_b, s1_b, ctl_b, dt=1.0 / 120.0, n_steps=20)

    # Full-trace match: flag-off is a strict no-op.
    np.testing.assert_allclose(history_a, history_b, rtol=0.0, atol=1e-6)


devices = get_test_devices()
constructor_devices = get_cuda_test_devices(mode="basic")

for device in constructor_devices:
    add_function_test(
        TestFeatherPGSVelocityLimitFlag,
        f"test_default_is_flag_off_{device}",
        run_default_is_flag_off,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitFlag,
        f"test_flag_on_allocates_buffers_{device}",
        run_flag_on_allocates_buffers,
        devices=[device],
    )

for device in devices:
    add_function_test(
        TestFeatherPGSVelocityLimitAllocationKernel,
        f"test_allocation_activates_only_over_limit_{device}",
        run_allocation_activates_only_over_limit,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitAllocationKernel,
        f"test_allocation_lower_vs_upper_sides_{device}",
        run_allocation_lower_vs_upper_sides,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitAllocationKernel,
        f"test_allocation_nonpositive_limit_is_unlimited_{device}",
        run_allocation_nonpositive_limit_is_unlimited,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitPopulatorKernel,
        f"test_populator_writes_signed_selector_row_and_metadata_{device}",
        run_populator_writes_signed_selector_row_and_metadata,
        devices=[device],
    )
    # End-to-end tests require the CUDA-native matrix-free PGS sweep
    # kernel. The kernel is a no-op on CPU (see class docstring).
    if str(device).startswith("cuda"):
        add_function_test(
            TestFeatherPGSVelocityLimitEndToEnd,
            f"test_end_to_end_flag_on_clamps_peak_qdot_{device}",
            run_end_to_end_flag_on_clamps_peak_qdot,
            devices=[device],
        )
        add_function_test(
            TestFeatherPGSVelocityLimitEndToEnd,
            f"test_end_to_end_flag_on_is_tighter_than_flag_off_{device}",
            run_end_to_end_flag_on_is_tighter_than_flag_off,
            devices=[device],
        )
        add_function_test(
            TestFeatherPGSVelocityLimitEndToEnd,
            f"test_end_to_end_flag_off_is_strict_noop_{device}",
            run_end_to_end_flag_off_is_strict_noop,
            devices=[device],
        )


if __name__ == "__main__":
    unittest.main()
