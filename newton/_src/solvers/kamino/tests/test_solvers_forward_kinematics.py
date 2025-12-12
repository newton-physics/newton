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

from newton._src.solvers.kamino.core.joints import JointActuationType, JointCorrectionMode, JointDoFType
from newton._src.solvers.kamino.core.model import Model
from newton._src.solvers.kamino.core.types import vec6f
from newton._src.solvers.kamino.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.solvers.fk import ForwardKinematicsSolver, ForwardKinematicsSolverSettings
from newton._src.solvers.kamino.tests import setup_tests, test_context
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
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
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

            # Generate (random) body poses
            bodies_q_np = rng.uniform(-1.0, 1.0, 7 * model.size.sum_of_num_bodies).astype("float32")
            bodies_q = wp.from_numpy(bodies_q_np, dtype=wp.transformf, device=model.device)

            # Generate (random) actuated coordinates
            actuators_q_np = rng.uniform(-1.0, 1.0, model.size.sum_of_num_actuated_joint_coords).astype("float32")
            actuators_q = wp.from_numpy(actuators_q_np, dtype=wp.float32, device=model.device)

            # Evaluate analytic Jacobian
            solver = ForwardKinematicsSolver(model=model)
            pos_control_transforms = solver.eval_position_control_transformations(actuators_q, None)
            jacobian = solver.eval_kinematic_constraints_jacobian(bodies_q, pos_control_transforms)

            # Check against finite differences Jacobian
            def eval_constraints(bodies_q_stepped_np):
                bodies_q.assign(bodies_q_stepped_np)
                constraints = solver.eval_kinematic_constraints(bodies_q, pos_control_transforms)
                bodies_q.assign(bodies_q_np)  # Reset state
                return constraints.numpy()[0]

            return diff_check(
                eval_constraints,
                jacobian.numpy()[0],
                bodies_q_np,
                epsilon=1e-4,
                tolerance_abs=5e-3,
                tolerance_rel=5e-3,
            )

        success = run_test_single_joint_examples(test_function, test_name, device=self.default_device)
        self.assertTrue(success)


def get_actuators_q_quaternion_first_ids(model: Model):
    """Lists the first index of every unit quaternion 4-segment in the model's actuated coordinates."""
    act_types = model.joints.act_type.numpy()
    dof_types = model.joints.dof_type.numpy()
    num_coords = model.joints.num_coords.numpy()
    coord_id = 0
    quat_ids = []
    for jt_id in range(model.size.sum_of_num_joints):
        if act_types[jt_id] == JointActuationType.PASSIVE:
            continue
        if dof_types[jt_id] == JointDoFType.SPHERICAL:
            quat_ids.append(coord_id)
        elif dof_types[jt_id] == JointDoFType.FREE:
            quat_ids.append(coord_id + 3)
        coord_id += num_coords[jt_id]
    return quat_ids


def compute_actuated_coords_and_dofs_offsets(model: Model):
    """
    Helper function computing the offsets and sizes needed to extract actuated joint coordinates
    and dofs from all joint coordinates/dofs
    Returns actuated_coords_offsets, actuated_coords_sizes, actuated_dofs_offsets, actuated_dofs_sizes
    """
    # Joints
    num_joints = model.info.num_joints.numpy()  # Num joints per world
    first_joint_id = np.concatenate(([0], num_joints.cumsum()))  # First joint id per world

    # Joint coordinates
    num_coords = model.info.num_joint_coords.numpy()  # Num coords per world
    first_coord = np.concatenate(([0], num_coords.cumsum()))  # First coord id per world
    coord_offsets = model.joints.coords_offset.numpy().copy()  # First coord id per joint within world
    for wd_id in range(model.size.num_worlds):  # Convert to first coord id per joint globally
        coord_offsets[first_joint_id[wd_id] : first_joint_id[wd_id + 1]] += first_coord[wd_id]
    joint_num_coords = model.joints.num_coords.numpy()  # Num coords per joint

    # Joint dofs
    num_dofs = model.info.num_joint_dofs.numpy()  # Num dofs per world
    first_dof = np.concatenate(([0], num_dofs.cumsum()))  # First dof id per world
    dof_offsets = model.joints.dofs_offset.numpy().copy()  # First dof id per joint within world
    for wd_id in range(model.size.num_worlds):  # Convert to first dof id per joint globally
        dof_offsets[first_joint_id[wd_id] : first_joint_id[wd_id + 1]] += first_dof[wd_id]
    joint_num_dofs = model.joints.num_dofs.numpy()  # Num dofs per joint

    # Filter for actuators only
    joint_is_actuator = model.joints.act_type.numpy() == JointActuationType.FORCE
    actuated_coord_offsets = coord_offsets[joint_is_actuator]
    actuated_coords_sizes = joint_num_coords[joint_is_actuator]
    actuated_dof_offsets = dof_offsets[joint_is_actuator]
    actuated_dofs_sizes = joint_num_dofs[joint_is_actuator]

    return actuated_coord_offsets, actuated_coords_sizes, actuated_dof_offsets, actuated_dofs_sizes


def extract_segments(array, offsets, sizes):
    """
    Helper function extracting from a flat array the segments with given offsets and sizes
    and returning their concatenation
    """
    res = []
    for i in range(len(offsets)):
        res.extend(array[offsets[i] : offsets[i] + sizes[i]])
    return np.array(res)


def compute_constraint_residual_mask(model: Model):
    """
    Computes a boolean mask for constraint residuals, True for most constraints but False
    for base joints (to filter out residuals for fixed base models if the base is reset
    to a different pose) and passive universal joints (residual implementation is currently flawed)
    """
    # Precompute constraint offsets
    num_joints = model.info.num_joints.numpy()  # Num joints per world
    first_joint_id = np.concatenate(([0], num_joints.cumsum()))  # First joint id per world
    num_cts = model.info.num_joint_cts.numpy()  # Num joint cts per world
    first_ct_id = np.concatenate(([0], num_cts.cumsum()))  # First joint ct id per world
    first_joint_ct_id = model.joints.cts_offset.numpy().copy()  # First ct id per joint within world
    for wd_id in range(model.size.num_worlds):  # Convert to first ct id per joint globally
        first_joint_ct_id[first_joint_id[wd_id] : first_joint_id[wd_id + 1]] += first_ct_id[wd_id]
    num_joint_cts = model.joints.num_cts.numpy()  # Num cts per joint

    mask = np.array(model.size.sum_of_num_joint_cts * [True])

    # Exclude base joints
    for wd_id in range(model.size.num_worlds):
        if not model.worlds[wd_id].has_base_joint:
            continue
        base_jt_id = first_joint_id[wd_id] + model.worlds[wd_id].base_joint_idx
        ct_offset = first_joint_ct_id[base_jt_id]
        mask[ct_offset : ct_offset + num_joint_cts[base_jt_id]] = False

    # Exclude passive universal joints
    act_types = model.joints.act_type.numpy()
    dof_types = model.joints.dof_type.numpy()
    for jt_id in range(model.size.sum_of_num_joints):
        if act_types[jt_id] != JointActuationType.PASSIVE or dof_types[jt_id] != JointDoFType.UNIVERSAL:
            continue
        ct_offset = first_joint_ct_id[jt_id]
        mask[ct_offset : ct_offset + num_joint_cts[jt_id]] = False

    return mask


def simulate_random_poses(
    model: Model,
    num_poses: int,
    max_base_q: np.ndarray,
    max_actuators_q: np.ndarray,
    max_base_u: np.ndarray,
    max_actuators_u: np.ndarray,
    rng: np.random._generator.Generator,
    use_graph: bool = False,
    verbose: bool = False,
):
    # Check dimensions
    base_q_size = 7 * model.size.num_worlds
    actuators_q_size = model.size.sum_of_num_actuated_joint_dofs
    base_u_size = 6 * model.size.num_worlds
    actuators_u_size = model.size.sum_of_num_actuated_joint_dofs
    assert len(max_base_q) == base_q_size
    assert len(max_actuators_q) == actuators_q_size
    assert len(max_base_u) == base_u_size
    assert len(max_actuators_u) == actuators_u_size

    # Generate (random) base_q, actuators_q
    base_q_np = np.zeros((num_poses, base_q_size))
    for i in range(base_q_size):
        base_q_np[:, i] = rng.uniform(-max_base_q[i], max_base_q[i], num_poses)
    actuators_q_np = np.zeros((num_poses, actuators_q_size))
    for i in range(actuators_q_size):
        actuators_q_np[:, i] = rng.uniform(-max_actuators_q[i], max_actuators_q[i], num_poses)

    # Normalize quaternions in base_q, actuators_q
    for i in range(model.size.num_worlds):
        base_q_np[:, 7 * i + 3 : 7 * i + 7] /= np.linalg.norm(base_q_np[:, 7 * i + 3 : 7 * i + 7], axis=1)[:, None]
    quat_ids = get_actuators_q_quaternion_first_ids(model)
    for i in quat_ids:
        actuators_q_np[:, i : i + 4] /= np.linalg.norm(actuators_q_np[:, i : i + 4], axis=1)[:, None]

    # Generate (random) base_u, actuators_u
    base_u_np = np.zeros((num_poses, base_u_size))
    for i in range(base_u_size):
        base_u_np[:, i] = rng.uniform(-max_base_u[i], max_base_u[i], num_poses)
    actuators_u_np = np.zeros((num_poses, actuators_u_size))
    for i in range(actuators_u_size):
        actuators_u_np[:, i] = rng.uniform(-max_actuators_u[i], max_actuators_u[i], num_poses)

    # Precompute offset arrays for extracting actuator coordinates/dofs
    actuated_coord_offsets, actuated_coords_sizes, actuated_dof_offsets, actuated_dofs_sizes = (
        compute_actuated_coords_and_dofs_offsets(model)
    )

    # Precompute boolean mask for extracting relevant constraint residuals
    residual_mask = compute_constraint_residual_mask(model)

    # Run forward kinematics on all random poses
    settings = ForwardKinematicsSolverSettings()
    settings.reset_state = True
    solver = ForwardKinematicsSolver(model, settings)
    success_flags = []
    with wp.ScopedDevice(model.device):
        bodies_q = wp.array(shape=(model.size.sum_of_num_bodies), dtype=wp.transformf)
        base_q = wp.array(shape=(model.size.num_worlds), dtype=wp.transformf)
        actuators_q = wp.array(shape=(actuators_q_size), dtype=wp.float32)
        bodies_u = wp.array(shape=(model.size.sum_of_num_bodies), dtype=vec6f)
        base_u = wp.array(shape=(model.size.num_worlds), dtype=vec6f)
        actuators_u = wp.array(shape=(actuators_u_size), dtype=wp.float32)
    data = model.data(device=model.device)
    epsilon = 1e-2
    for pose_id in range(num_poses):
        # Run FK solve and check convergence
        base_q.assign(base_q_np[pose_id])
        actuators_q.assign(actuators_q_np[pose_id])
        base_u.assign(base_u_np[pose_id])
        actuators_u.assign(actuators_u_np[pose_id])
        status = solver.solve_fk(
            actuators_q,
            bodies_q,
            base_q=base_q,
            base_u=base_u,
            actuators_u=actuators_u,
            bodies_u=bodies_u,
            use_graph=use_graph,
            verbose=verbose,
            return_status=True,
        )
        if status.success.min() < 1:
            success_flags.append(False)
            continue
        else:
            success_flags.append(True)

        # Update joints data from body states for validation
        wp.copy(data.bodies.q_i, bodies_q)
        wp.copy(data.bodies.u_i, bodies_u)
        compute_joints_data(model, model.joints.q_j_ref, data, JointCorrectionMode.CONTINUOUS)

        # Validate positions computation
        residual_ct_pos = np.max(np.abs(data.joints.r_j.numpy()[residual_mask]))
        if residual_ct_pos > epsilon:
            print(f"Large constraint residual ({residual_ct_pos}) for pose {pose_id}")
            success_flags[-1] = False
        actuators_q_check = extract_segments(data.joints.q_j.numpy(), actuated_coord_offsets, actuated_coords_sizes)
        residual_actuators_q = np.max(np.abs(actuators_q_check - actuators_q_np[pose_id]))
        if residual_actuators_q > epsilon:
            print(f"Large error on prescribed actuator coordinates ({residual_actuators_q}) for pose {pose_id}")
            success_flags[-1] = False

        # Validate velocities computation
        residual_ct_vel = np.max(np.abs(data.joints.dr_j.numpy()[residual_mask]))
        if residual_ct_vel > epsilon:
            print(f"Large constraint velocity residual ({residual_ct_vel}) for pose {pose_id}")
            success_flags[-1] = False
        actuators_u_check = extract_segments(data.joints.dq_j.numpy(), actuated_dof_offsets, actuated_dofs_sizes)
        residual_actuators_u = np.max(np.abs(actuators_u_check - actuators_u_np[pose_id]))
        if residual_actuators_u > epsilon:
            print(f"Large error on prescribed actuator velocities ({residual_actuators_u}) for pose {pose_id}")
            success_flags[-1] = False

    success = np.sum(success_flags) == num_poses
    if not success:
        print(f"Random poses simulation & validation failed, {np.sum(success_flags)}/{num_poses} poses successful")

    return success


class DRTestMechanismRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.has_cuda = self.default_device.is_cuda
        self.verbose = test_context.verbose

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
        builder.set_base_joint("base")
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        base_q_max = np.array(3 * [0.2] + 4 * [1.0])
        actuators_q_max = np.radians([95.0])
        base_u_max = np.array(3 * [0.1] + 3 * [0.5])
        actuators_u_max = np.array([0.5])
        success = simulate_random_poses(
            model,
            num_poses,
            base_q_max,
            actuators_q_max,
            base_u_max,
            actuators_u_max,
            rng,
            self.has_cuda,
            self.verbose,
        )
        self.assertTrue(success)


class DRLegsRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.has_cuda = self.default_device.is_cuda
        self.verbose = test_context.verbose

    def tearDown(self):
        self.default_device = None

    def test_dr_legs_FK_random_poses(self):
        # Initialize RNG
        test_name = "FK random poses check for dr_legs model"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Load model and set base body to pelvis
        examples_path = get_examples_usd_assets_path()
        if not examples_path:
            self.skipTest("Examples USD assets path not found. Skipping test.")
        model_path = os.path.join(examples_path, "dr_legs/usd/dr_legs_with_boxes.usda")
        builder = USDImporter().import_from(model_path)
        builder.set_base_body("pelvis")
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max = np.radians(10.0)  # Angles too far from the initial pose lead to singularities
        base_q_max = np.array(3 * [0.2] + 4 * [1.0])
        actuators_q_max = np.array(model.size.sum_of_num_actuated_joint_coords * [theta_max])
        base_u_max = np.array(3 * [0.5] + 3 * [0.5])
        actuators_u_max = np.array(model.size.sum_of_num_actuated_joint_dofs * [0.5])
        success = simulate_random_poses(
            model,
            num_poses,
            base_q_max,
            actuators_q_max,
            base_u_max,
            actuators_u_max,
            rng,
            self.has_cuda,
            self.verbose,
        )
        self.assertTrue(success)


class HeterogenousModelRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.has_cuda = self.default_device.is_cuda
        self.verbose = test_context.verbose

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
        builder.set_base_joint("base")
        model_path1 = os.path.join(examples_path, "dr_legs/usd/dr_legs_with_boxes.usda")
        builder1 = USDImporter().import_from(model_path1)
        builder1.set_base_body("pelvis")
        builder.add_builder(builder1)
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max_test_mech = np.radians(100.0)
        theta_max_dr_legs = np.radians(10.0)
        base_q_max = np.array(3 * [0.2] + 4 * [1.0] + 3 * [0.2] + 4 * [1.0])
        actuators_q_max = np.array([theta_max_test_mech] + builder1.num_actuated_joint_coords * [theta_max_dr_legs])
        base_u_max = np.array(3 * [0.1] + 3 * [0.5] + 3 * [0.5] + 3 * [0.5])
        actuators_u_max = np.array(model.size.sum_of_num_actuated_joint_dofs * [0.5])
        success = simulate_random_poses(
            model, num_poses, base_q_max, actuators_q_max, base_u_max, actuators_u_max, rng, self.has_cuda, self.verbose
        )
        self.assertTrue(success)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
