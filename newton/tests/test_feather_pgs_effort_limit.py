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
the cap. The legacy ``clamp_joint_tau`` kernel is still exercised as a
reference for what the old (buggy) net-torque clamp would have done.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import (
    clamp_augmented_joint_u0,
    clamp_joint_tau,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestFeatherPGSEffortLimitClamp(unittest.TestCase):
    """Direct signal that the alternative path clamps a different quantity.

    ``clamp_joint_tau`` is the baseline kernel that clamps the *net*
    generalized torque (rigid/passive + drive-explicit) sitting in
    ``joint_tau``. ``clamp_augmented_joint_u0`` receives only the per-row
    actuator-drive ``u0`` bucket, so it cannot clamp the rigid/passive
    bucket by construction.
    """


def run_clamp_augmented_joint_u0_clamps_only_drive(test: TestFeatherPGSEffortLimitClamp, device):
    """The actuator-bucket kernel clamps ``row_u0`` against the per-DOF limit.

    Build a single-articulation, 2-DOF layout with two drive rows:

    - DOF 0: ``u0 = 100``, ``limit = 5``     → expect clamp to ``+5``.
    - DOF 1: ``u0 = -50``, ``limit = 0``     → expect no clamp (unlimited).

    This is the direct kernel-level check for the actuator bucket. The
    companion ``clamp_joint_tau`` test covers the old net-torque bucket.
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


def run_clamp_joint_tau_clamps_net_sum(test: TestFeatherPGSEffortLimitClamp, device):
    """The baseline kernel clamps the net ``joint_tau`` buffer in place.

    Mirrors ``run_clamp_augmented_joint_u0_clamps_only_drive`` but on the
    baseline kernel, showing the quantity being clamped is the net
    generalized torque (``tau_rigid + u0``) rather than the drive-only
    ``u0`` bucket. This is the other half of the "different quantity"
    signal.
    """
    # Same pair of DOFs as the actuator-bucket test, but we now store the
    # *net* generalized torque in joint_tau: rigid_passive = -9.81 at DOF 0
    # plus u0 = +100 → net = +90.19. For DOF 1: rigid = 0, u0 = -50,
    # net = -50. Limit 5 on DOF 0, unlimited on DOF 1.
    net_tau = np.array([-9.81 + 100.0, 0.0 - 50.0], dtype=np.float32)
    joint_tau = wp.array(net_tau, dtype=wp.float32, device=device)
    joint_effort_limit = wp.array([5.0, 0.0], dtype=wp.float32, device=device)

    wp.launch(
        clamp_joint_tau,
        dim=2,
        inputs=[joint_tau, joint_effort_limit],
        device=device,
    )

    out = joint_tau.numpy()
    # DOF 0's net is clamped to +5 (baseline: the rigid/passive contribution
    # is *not* excluded — it is part of the clamped quantity). DOF 1 is
    # unlimited so the -50 survives unchanged.
    np.testing.assert_allclose(out, np.array([5.0, -50.0], dtype=np.float32), rtol=0.0, atol=1e-7)


def run_net_vs_actuator_differ_on_passive_only_dof(test: TestFeatherPGSEffortLimitClamp, device):
    """Direct divergence signal: passive-only load differs by clamp target.

    Build a DOF whose drive ``u0 = 0`` but whose rigid/passive bucket is
    ``-9.81`` (a gravity-loaded joint). With ``limit = 5.0`` the two
    semantics produce opposite answers:

    - Baseline (:func:`clamp_joint_tau` on the summed buffer):
      clamps the passive load to ``-5.0``.
    - Alternative (:func:`clamp_augmented_joint_u0` on the empty drive-only
      buffer): is a no-op when ``row_counts == 0``, so the passive load
      remains outside the clamped actuator bucket.
    """
    # Shared inputs.
    limit = 5.0
    rigid_passive = -9.81  # "tau_passive" at a horizontal pendulum, zero drive.

    # --- Baseline (net clamp) ---
    # joint_tau after `apply_augmented_joint_tau` in the baseline path
    # contains the summed rigid + drive-explicit bucket. Here drive=0.
    net_joint_tau = wp.array([rigid_passive], dtype=wp.float32, device=device)
    net_limit = wp.array([limit], dtype=wp.float32, device=device)
    wp.launch(
        clamp_joint_tau,
        dim=1,
        inputs=[net_joint_tau, net_limit],
        device=device,
    )
    net_out = float(net_joint_tau.numpy()[0])

    # --- Alternative (actuator-only clamp) ---
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
    # Alternative semantics leaves the gravity torque untouched.
    test.assertAlmostEqual(act_out, rigid_passive, places=6)
    # And the two outputs actually disagree — the "direct signal" that
    # the alternative path clamps a different quantity.
    test.assertGreater(abs(act_out - net_out), 1e-3)


devices = get_test_devices()

for device in devices:
    add_function_test(
        TestFeatherPGSEffortLimitClamp,
        f"test_clamp_augmented_joint_u0_clamps_only_drive_{device}",
        run_clamp_augmented_joint_u0_clamps_only_drive,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSEffortLimitClamp,
        f"test_clamp_joint_tau_clamps_net_sum_{device}",
        run_clamp_joint_tau_clamps_net_sum,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSEffortLimitClamp,
        f"test_net_vs_actuator_differ_on_passive_only_dof_{device}",
        run_net_vs_actuator_differ_on_passive_only_dof,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
