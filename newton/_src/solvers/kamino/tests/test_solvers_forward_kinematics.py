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

"""
KAMINO: UNIT TESTS: SOLVERS: FORWARD KINEMATICS
"""

import hashlib
import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.core.joints import JointActuationType, JointDoFType
from newton._src.solvers.kamino.core.model import Model
from newton._src.solvers.kamino.core.types import mat33f, vec3f
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.solvers.fk import ForwardKinematicsSolver
from newton._src.solvers.kamino.tests.utils.diff_check import diff_check, run_test_single_joint_examples
from newton._src.solvers.kamino.utils.io.usd import USDImporter

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Tests
###


class JacobianCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.has_cuda = self.default_device.is_cuda

    def tearDown(self):
        self.default_device = None

    def test_Jacobian_check(self):
        # Initialize RNG
        test_name = "Forward Kinematics Jacobian check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        def test_function(model: Model):
            assert model.size.num_worlds == 1  # For simplicity we assume a single world

            # Generate (random) state
            data = model.data(device=self.default_device)
            num_bodies_q_i = data.bodies.q_i.shape[0] * 7
            random_state = rng.uniform(-1.0, 1.0, num_bodies_q_i).astype("float32")
            data.bodies.q_i.assign(random_state)

            # Generate (random) controls
            num_joints_q_j = data.joints.q_j.shape[0]
            data.joints.q_j.assign(rng.uniform(-1.0, 1.0, num_joints_q_j).astype("float32"))

            # Evaluate analytic Jacobian
            solver = ForwardKinematicsSolver(model=model)
            pos_control_transforms = solver.eval_position_control_transformations(data)
            jacobian = solver.eval_kinematic_constraints_jacobian(data, pos_control_transforms)

            # Check against finite differences Jacobian
            rb_state_init_np = data.bodies.q_i.numpy().flatten()  # Save current state of bodies

            def eval_constraints(rb_state_np):
                data.bodies.q_i.assign(rb_state_np)
                constraints = solver.eval_kinematic_constraints(data, pos_control_transforms)
                data.bodies.q_i.assign(rb_state_init_np)  # Reset state
                return constraints.numpy()[0]

            return diff_check(
                eval_constraints,
                jacobian.numpy()[0],
                rb_state_init_np,
                epsilon=1e-4,
                tolerance_abs=5e-3,
                tolerance_rel=5e-3,
            )

        success = run_test_single_joint_examples(test_function, test_name, device=self.default_device)
        self.assertTrue(success)


def simulate_random_poses(
    model: Model,
    num_poses: int,
    min_controls: np.ndarray,
    max_controls: np.ndarray,
    rng: np.random._generator.Generator,
    use_graph: bool = False,
    verbose: bool = False,
):
    num_controls = model.size.sum_of_num_actuated_joint_dofs
    assert len(min_controls) == num_controls
    assert len(max_controls) == num_controls

    # Generate (random) controls
    num_joints = model.info.num_joints.numpy()
    num_joint_dofs = model.joints.num_dofs.numpy()
    joint_act_types = model.joints.act_type.numpy()
    first_joint_dof = np.concatenate(([0], model.info.num_joint_dofs.numpy().cumsum()))
    joint_dof_offsets_loc = model.joints.dofs_offset.numpy()  # Offset within dofs of a single world
    num_gen_pos = model.size.sum_of_num_joint_dofs
    gen_pos_random = np.zeros((num_poses, num_gen_pos))
    id_control = 0
    for wd_id in range(model.size.num_worlds):
        for i in range(num_joints[wd_id]):
            if joint_act_types[i] == JointActuationType.PASSIVE:
                continue
            joint_dof_offset = first_joint_dof[wd_id] + joint_dof_offsets_loc[i]
            for j in range(num_joint_dofs[i]):
                gen_pos_random[:, joint_dof_offset + j] = rng.uniform(
                    min_controls[id_control], max_controls[id_control], num_poses
                )
                id_control += 1

    # Run forward kinematics on all random poses
    model_data = model.data(device=model.device)
    solver = ForwardKinematicsSolver(model)
    success_flags = []
    for i in range(num_poses):
        model_data.joints.q_j.assign(gen_pos_random[i, :])
        status = solver.solve_fk(model_data, reset_state=True, use_graph=use_graph, verbose=verbose, return_status=True)
        success_flags.append(status.success.min() == 1)

    success = np.sum(success_flags) == num_poses
    if not success:
        print(f"Random poses simulation failed, {np.sum(success_flags)}/{num_poses} poses successful")
    return success


class DRTestMechanismRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.has_cuda = self.default_device.is_cuda
        self.verbose = False

    def tearDown(self):
        self.default_device = None

    def test_mechanism_FK_random_poses(self):
        # Initialize RNG
        test_name = "Test mechanism FK random poses check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Load model
        examples_path = get_examples_usd_assets_path()
        if not examples_path:
            self.skipTest("Examples USD assets path not found. Skipping test.")
        model_path = os.path.join(examples_path, "dr_testmech/usd/dr_testmech.usda")
        builder = USDImporter().import_from(model_path)
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max = np.radians(180.0)
        success = simulate_random_poses(
            model, num_poses, np.array([-theta_max]), np.array([theta_max]), rng, self.has_cuda, self.verbose
        )
        self.assertTrue(success)


class DRLegsRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.has_cuda = self.default_device.is_cuda
        self.verbose = False

    def tearDown(self):
        self.default_device = None

    def test_dr_legs_FK_random_poses(self):
        # Initialize RNG
        test_name = "FK random poses check for dr_legs model"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Load model and add fixed joint on pelvis
        examples_path = get_examples_usd_assets_path()
        if not examples_path:
            self.skipTest("Examples USD assets path not found. Skipping test.")
        model_path = os.path.join(examples_path, "dr_legs/usd/dr_legs_with_boxes.usda")
        builder = USDImporter().import_from(model_path)
        builder.add_joint(
            JointActuationType.PASSIVE,
            JointDoFType.FIXED,
            -1,
            0,
            builder.bodies[0].q_i_0[:3],
            vec3f(0.0, 0.0, 0.0),
            mat33f(np.identity(3)),
            name="Fixed pelvis",
        )
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max = np.radians(10.0)  # Angles too far from the initial pose lead to singularities
        num_controls = model.info.num_actuated_joint_dofs.numpy()[0]
        success = simulate_random_poses(
            model,
            num_poses,
            np.array(num_controls * [-theta_max]),
            np.array(num_controls * [theta_max]),
            rng,
            self.has_cuda,
            self.verbose,
        )
        self.assertTrue(success)


class HeterogenousModelRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        self.default_device = wp.get_device()
        self.has_cuda = self.default_device.is_cuda
        self.verbose = False

    def tearDown(self):
        self.default_device = None

    def test_heterogenous_model_FK_random_poses(self):
        # Initialize RNG
        test_name = "Heterogenous model (test mechanism + dr_legs) FK random poses check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Load models
        examples_path = get_examples_usd_assets_path()
        if not examples_path:
            self.skipTest("Examples USD assets path not found. Skipping test.")
        model_path = os.path.join(examples_path, "dr_testmech/usd/dr_testmech.usda")
        builder = USDImporter().import_from(model_path)
        model_path1 = os.path.join(examples_path, "dr_legs/usd/dr_legs_with_boxes.usda")
        builder1 = USDImporter().import_from(model_path1)
        builder1.add_joint(
            JointActuationType.PASSIVE,
            JointDoFType.FIXED,
            -1,
            0,
            builder1.bodies[0].q_i_0[:3],
            vec3f(0.0, 0.0, 0.0),
            mat33f(np.identity(3)),
            name="Fixed pelvis",
        )
        builder.add_builder(builder1)
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max_test_mech = np.radians(180.0)
        theta_max_dr_legs = np.radians(10.0)
        max_controls = np.array([theta_max_test_mech] + builder1.num_actuated_joint_dofs * [theta_max_dr_legs])
        success = simulate_random_poses(model, num_poses, -max_controls, max_controls, rng, self.has_cuda, self.verbose)
        self.assertTrue(success)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(
        linewidth=999999, edgeitems=999999, threshold=999999, precision=10, suppress=True
    )  # Suppress scientific notation

    # Global warp configurations
    wp.config.enable_backward = False
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
