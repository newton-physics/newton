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


class TestFeatherPGSFusedWarpSelector(unittest.TestCase):
    """Constructor-level coverage for the private fused-Warp API."""


def run_constructor_accepts_fused_warp(test: TestFeatherPGSFusedWarpSelector, device):
    model = _build_mixed_contact_model(device)
    solver = SolverFeatherPGS(
        model,
        pgs_iterations=1,
    )

    test.assertEqual(solver.pgs_mode, "matrix_free")
    test.assertEqual(solver.pgs_kernel, "fused_warp")
    test.assertIsNotNone(solver.mf_meta)
    test.assertIsNotNone(solver.impulses_vec3)
    test.assertEqual(solver._max_contact_triplets, solver.max_constraints // 3)


def run_constructor_rejects_unsupported_combinations(test: TestFeatherPGSFusedWarpSelector, device):
    model = _build_mixed_contact_model(device)

    with test.assertRaisesRegex(ValueError, "multiple of 3"):
        SolverFeatherPGS(
            model,
            max_constraints=32,
        )

    with test.assertRaisesRegex(ValueError, "requires enable_contact_friction=True"):
        SolverFeatherPGS(
            model,
            enable_contact_friction=False,
        )


class TestFeatherPGSFusedWarpStep(unittest.TestCase):
    """Launch coverage for Zach's fused Warp tile kernel."""


def run_mixed_contact_one_step(test: TestFeatherPGSFusedWarpStep, device):
    model = _build_mixed_contact_model(device)
    solver = SolverFeatherPGS(
        model,
        enable_joint_velocity_limits=True,
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
    wp.synchronize_device(device)

    test.assertTrue(np.isfinite(state_1.joint_q.numpy()).all())
    test.assertGreaterEqual(int(solver.dense_contact_row_count.numpy()[0]), 3)
    test.assertGreater(int(solver.constraint_count.numpy()[0]), int(solver.dense_contact_row_count.numpy()[0]))
    test.assertGreaterEqual(int(solver.mf_constraint_count.numpy()[0]), 3)


devices = get_cuda_test_devices(mode="basic")

for device in devices:
    add_function_test(
        TestFeatherPGSFusedWarpSelector,
        f"test_constructor_accepts_fused_warp_{device}",
        run_constructor_accepts_fused_warp,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFusedWarpSelector,
        f"test_constructor_rejects_unsupported_combinations_{device}",
        run_constructor_rejects_unsupported_combinations,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFusedWarpStep,
        f"test_mixed_contact_one_step_{device}",
        run_mixed_contact_one_step,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
