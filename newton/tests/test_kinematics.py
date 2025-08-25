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

import math
import unittest

import warp as wp

import newton.examples
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices


def test_fk_ik(test, device):
    builder = newton.ModelBuilder()

    num_envs = 1

    for i in range(num_envs):
        builder.add_mjcf(newton.examples.get_asset("nv_ant.xml"), up_axis="Y")

        coord_count = 15
        dof_count = 14

        coord_start = i * coord_count
        dof_start = i * dof_count

        # base
        builder.joint_q[coord_start : coord_start + 3] = [i * 2.0, 0.70, 0.0]
        builder.joint_q[coord_start + 3 : coord_start + 7] = wp.quat_from_axis_angle(
            wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5
        )

        # joints
        builder.joint_q[coord_start + 7 : coord_start + coord_count] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
        builder.joint_qd[dof_start + 6 : dof_start + dof_count] = [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]

    # finalize model
    model = builder.finalize(device=device)

    state = model.state()

    # save a copy of joint values
    q_fk = model.joint_q.numpy()
    qd_fk = model.joint_qd.numpy()

    newton.eval_fk(model, model.joint_q, model.joint_qd, state)

    q_ik = wp.zeros_like(model.joint_q, device=device)
    qd_ik = wp.zeros_like(model.joint_qd, device=device)

    newton.eval_ik(model, state, q_ik, qd_ik)

    assert_np_equal(q_fk, q_ik.numpy(), tol=1e-6)
    assert_np_equal(qd_fk, qd_ik.numpy(), tol=1e-6)


devices = get_test_devices()


class TestSimKinematics(unittest.TestCase):
    pass


add_function_test(TestSimKinematics, "test_fk_ik", test_fk_ik, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
