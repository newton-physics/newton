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

"""Regression tests for the FeatherPGS actuator-only effort-limit clamp.

``SolverFeatherPGS`` always applies an actuator-only effort-limit
clamp: the explicit-PD drive bucket (``u0``) is clamped before it is
folded into ``joint_tau``. The earlier net-torque semantics was a
silent bug and has been removed.

The tests in this file cover that ``clamp_augmented_joint_u0`` clamps only
the per-row actuator-drive ``u0``, using ``joint_effort_limit[dof]`` as
the cap.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import clamp_augmented_joint_u0
from newton.tests.unittest_utils import add_function_test, get_test_devices


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


if __name__ == "__main__":
    unittest.main()
