# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton import JointType
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices

# -----------------------------------------------------------------------------
# Joint types we want to hit
# -----------------------------------------------------------------------------
JOINT_KINDS: list[int] = [
    JointType.REVOLUTE,
    JointType.PRISMATIC,
    JointType.BALL,
    JointType.D6,
    JointType.FREE,
]


# -----------------------------------------------------------------------------
# Dummy (no-op) objective, gives R = 1 so IKSolver factory doesn't generate a 0xC solver
# -----------------------------------------------------------------------------


class _NoopObjective(ik.IKObjective):
    def residual_dim(self):
        return 1

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        return

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof):
        return

    # keep analytic path trivial too
    def supports_analytic(self):
        return True

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        return


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _add_single_joint(builder: newton.ModelBuilder, jt: int) -> None:
    """
    Adds one body and connects it to the world (-1) with the joint-type `jt`.
    Makes every axis a plain JointDofConfig so the helper works for *all*
    joint kinds we care about in the parity tests.
    """
    cfg = newton.ModelBuilder.JointDofConfig  # alias
    parent_xf = wp.transform((0.1, 0.2, 0.3), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.0))
    child_xf = wp.transform((-0.05, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.5))

    # a 0.1-kg cube just so the body exists
    child = builder.add_link(
        xform=wp.transform_identity(),
        mass=0.1,
        label=f"body_{jt}",
    )
    builder.add_shape_box(
        body=child,
        xform=wp.transform_identity(),
        hx=0.05,
        hy=0.05,
        hz=0.05,
    )

    ji = builder.joint_count

    if jt == JointType.REVOLUTE:
        builder.add_joint_revolute(
            parent=-1,
            child=child,
            parent_xform=parent_xf,
            child_xform=child_xf,
            axis=[0.0, 0.0, 1.0],
        )

    elif jt == JointType.PRISMATIC:
        builder.add_joint_prismatic(
            parent=-1,
            child=child,
            parent_xform=parent_xf,
            child_xform=child_xf,
            axis=[1.0, 0.0, 0.0],
        )

    elif jt == JointType.BALL:
        builder.add_joint_ball(
            parent=-1,
            child=child,
            parent_xform=parent_xf,
            child_xform=child_xf,
        )

    elif jt == JointType.D6:
        builder.add_joint_d6(
            -1,
            child,
            linear_axes=[
                cfg(axis=newton.Axis.X),
                cfg(axis=newton.Axis.Y),
                cfg(axis=newton.Axis.Z),
            ],
            angular_axes=[
                cfg(axis=[1, 0, 0]),
                cfg(axis=[0, 1, 0]),
                cfg(axis=[0, 0, 1]),
            ],
            parent_xform=parent_xf,
            child_xform=child_xf,
        )

    elif jt == JointType.FREE:
        builder.add_joint_free(
            parent=-1,
            child=child,
            parent_xform=parent_xf,
            child_xform=child_xf,
        )

    else:
        raise ValueError(f"Unhandled joint type {jt}")

    if ji == builder.joint_count - 1:
        builder.add_articulation([ji])


def _build_model_for_joint(jt: int, device):
    builder = newton.ModelBuilder()
    _add_single_joint(builder, jt)
    model = builder.finalize(device=device, requires_grad=True)
    return model


def _randomize_joint_q(model: newton.Model, seed: int = 0) -> None:
    """In-place randomization of the model's joint coordinates.

    Keeps magnitudes modest so FK stays well-conditioned and never
    violates default limits (±π for angles, ±0.5 m for free-joint
    translation, quaternion re-normalised).
    """
    rng = np.random.default_rng(seed)

    q_np = model.joint_q.numpy()  # view to host buffer
    for j, jt in enumerate(model.joint_type.numpy()):
        q0 = model.joint_q_start.numpy()[j]  # first coord idx

        if jt == JointType.REVOLUTE:
            q_np[q0] = rng.uniform(-np.pi / 2, np.pi / 2)

        elif jt == JointType.PRISMATIC:
            q_np[q0] = rng.uniform(-0.2, 0.2)  # metres

        elif jt == JointType.BALL:
            # random small-angle quaternion
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-8
            angle = rng.uniform(-np.pi / 6, np.pi / 6)
            qw = np.cos(angle / 2.0)
            qv = axis * np.sin(angle / 2.0)
            q_np[q0 : q0 + 4] = (*qv, qw)

        elif jt == newton.JointType.D6:
            # lin X,Y,Z
            q_np[q0 + 0 : q0 + 3] = rng.uniform(-0.1, 0.1, size=3)
            # rot XYZ
            q_np[q0 + 3 : q0 + 6] = rng.uniform(-np.pi / 8, np.pi / 8, size=3)

        elif jt == newton.JointType.FREE:
            # translation
            q_np[q0 + 0 : q0 + 3] = rng.uniform(-0.3, 0.3, size=3)
            # quaternion
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-8
            angle = rng.uniform(-np.pi / 6, np.pi / 6)
            qw = np.cos(angle / 2.0)
            qv = axis * np.sin(angle / 2.0)
            q_np[q0 + 3 : q0 + 7] = (*qv, qw)

    wp.copy(model.joint_q, wp.array(q_np, dtype=wp.float32))


# -----------------------------------------------------------------------------
# Forward-kinematics two-pass vs reference
# -----------------------------------------------------------------------------


def _fk_parity_for_joint(test, device, jt):
    with wp.ScopedDevice(device):
        model = _build_model_for_joint(jt, device)
        _randomize_joint_q(model)

        # reference FK
        state_ref = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_ref)

        # two-pass FK (via IKSolver helper)
        joint_q = model.joint_q.reshape((1, model.joint_coord_count))
        ik_solver = ik.IKSolver(
            model,
            1,
            objectives=[_NoopObjective()],
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
        )
        ik_solver._fk_two_pass(model, joint_q, ik_solver.body_q, ik_solver.X_local, ik_solver.n_problems)

        assert_np_equal(state_ref.body_q.numpy(), ik_solver.body_q.numpy(), tol=1e-6)


def test_fk_two_pass_parity(test, device):
    for jt in JOINT_KINDS:
        _fk_parity_for_joint(test, device, jt)


# -----------------------------------------------------------------------------
# Register tests
# -----------------------------------------------------------------------------

devices = get_test_devices()


class TestIKFKKernels(unittest.TestCase):
    pass


add_function_test(TestIKFKKernels, "test_fk_two_pass_parity", test_fk_two_pass_parity, devices)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
