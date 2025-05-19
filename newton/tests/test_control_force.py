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

import unittest

import numpy as np
import warp as wp
from warp.render import OpenGLRenderer

import newton
from newton.core.inertia import (
    compute_box_inertia,
    compute_mesh_inertia,
    compute_sphere_inertia,
)
from newton.tests.unittest_utils import assert_np_equal


class TestControlForce(unittest.TestCase):
    def test_floating_body(self):
        builder = newton.ModelBuilder(gravity=0.0)

        # easy case: identity transform, zero center of mass
        b = builder.add_body()
        builder.add_shape_box(b)
        builder.add_joint_free(b)

        model = builder.finalize()

        # solver = newton.solvers.XPBDSolver(
        #     model
        # )
        # solver = newton.solvers.FeatherstoneSolver(
        #     model
        # )
        # solver = newton.solvers.MuJoCoSolver(
        #     model,
        #     solver="newton",
        #     integrator="euler",
        #     iterations=10,
        #     ls_iterations=5,
        # )
        solver = newton.solvers.SemiImplicitSolver(
            model,
        )

        state_0, state_1 = model.state(), model.state()

        control = model.control()
        control.joint_f.assign(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))

        sim_dt = 1.0 / 10.0

        for _ in range(10):
            solver.step(model, state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

        body_qd = state_0.body_qd.numpy()[0]
        self.assertGreater(body_qd[2], 0.005)
        self.assertLess(body_qd[2], 0.02)
        self.assertEqual(np.sum(body_qd[0:2]), 0.0)
        self.assertEqual(np.sum(body_qd[3:6]), 0.0)

        joint_qd = state_0.joint_qd.numpy()
        self.assertGreater(joint_qd[2], 0.005)
        self.assertLess(joint_qd[2], 0.01)
        self.assertEqual(np.sum(joint_qd[0:2]), 0.0)
        self.assertEqual(np.sum(joint_qd[3:6]), 0.0)


        # Compute hollow box inertia
        mass_0_hollow, com_0_hollow, I_0_hollow, volume_0_hollow = compute_mesh_inertia(
            density=1000,
            vertices=vertices,
            indices=indices,
            is_solid=False,
            thickness=0.1,
        )
        assert_np_equal(np.array(com_0_hollow), np.array([0.5, 0.5, 0.5]), tol=1e-6)

        # Add vertex between [0.0, 0.0, 0.0] and [1.0, 0.0, 0.0]
        vertices.append([0.5, 0.0, 0.0])
        indices[5] = [0, 8, 7]
        indices.append([8, 3, 7])
        indices[6] = [0, 1, 8]
        indices.append([8, 1, 3])

        mass_1, com_1, I_1, volume_1 = compute_mesh_inertia(
            density=1000, vertices=vertices, indices=indices, is_solid=True
        )

        # Inertia values should be the same as before
        self.assertAlmostEqual(mass_1, mass_0, delta=1e-6)
        self.assertAlmostEqual(volume_1, volume_0, delta=1e-6)
        assert_np_equal(np.array(com_1), np.array([0.5, 0.5, 0.5]), tol=1e-6)
        assert_np_equal(np.array(I_1), np.array(I_0), tol=1e-4)

        # Compute hollow box inertia
        mass_1_hollow, com_1_hollow, I_1_hollow, volume_1_hollow = compute_mesh_inertia(
            density=1000,
            vertices=vertices,
            indices=indices,
            is_solid=False,
            thickness=0.1,
        )

        # Inertia values should be the same as before
        self.assertAlmostEqual(mass_1_hollow, mass_0_hollow, delta=2e-3)
        self.assertAlmostEqual(volume_1_hollow, volume_0_hollow, delta=1e-6)
        assert_np_equal(np.array(com_1_hollow), np.array([0.5, 0.5, 0.5]), tol=1e-6)
        assert_np_equal(np.array(I_1_hollow), np.array(I_0_hollow), tol=1e-4)



if __name__ == "__main__":
    # wp.clear_kernel_cache()
    unittest.main(verbosity=2)
