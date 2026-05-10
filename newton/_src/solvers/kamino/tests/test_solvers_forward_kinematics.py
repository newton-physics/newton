# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the ForwardKinematicsSolver class of Kamino, in `solvers/fk.py`.
"""

import copy
import hashlib
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointCorrectionMode, JointDoFType
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.types import vec6f
from newton._src.solvers.kamino._src.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino._src.models.builders.basics import build_boxes_fourbar
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.solvers.fk import ForwardKinematicsSolver
from newton._src.solvers.kamino._src.utils.io.usd import USDImporter
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.diff_check import diff_check, run_test_single_joint_examples

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Tests
###


def create_four_bar_tie_rod() -> ModelBuilderKamino:
    """
    Creates a four-bar linkage, but with two revolute joints replaced with
    spherical joints so as to create a tie rod (to test axis joints).
    """
    builder_revolute = build_boxes_fourbar(
        fixedbase=False,
        floatingbase=True,
        limits=False,
        ground=False,
        verbose=False,
        dynamic_joints=False,
        implicit_pd=False,
        actuator_ids=[1],
    )
    builder_spherical = ModelBuilderKamino(default_world=True)
    for body in builder_revolute.bodies[0]:
        builder_spherical.add_rigid_body_descriptor(copy.deepcopy(body))
    for joint in builder_revolute.joints[0]:
        joint_copy = copy.deepcopy(joint)
        if joint.name == "link2_to_link3" or joint.name == "link3_to_link4":
            joint_copy.dof_type = JointDoFType.SPHERICAL
        builder_spherical.add_joint_descriptor(joint_copy)
    for geom in builder_revolute.geoms[0]:
        geom_copy = copy.deepcopy(geom)
        geom_copy.shape = builder_revolute.shapes[geom.uid]
        builder_spherical.add_geometry_descriptor(geom_copy)
    return builder_spherical


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

        def test_function(model: ModelKamino):
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


def get_actuators_q_quaternion_first_ids(model: ModelKamino):
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


def compute_actuated_coords_and_dofs_data(model: ModelKamino):
    """
    Helper function computing the offsets and sizes needed to extract actuated joint coordinates
    and dofs from all joint coordinates/dofs, as well as the corresponding dof types.
    Returns actuated_coords_offsets, actuated_coords_sizes, actuated_dofs_offsets, actuated_dofs_sizes,
            actuator_dof_types
    """
    # Retrieve data for all joints (offset arrays include a trailing total)
    coord_offsets = model.joints.coords_offset.numpy()[:-1]
    joint_num_coords = model.joints.num_coords.numpy()
    dof_offsets = model.joints.dofs_offset.numpy()[:-1]
    joint_num_dofs = model.joints.num_dofs.numpy()
    joint_dof_types = model.joints.dof_type.numpy()

    # Filter for actuators only
    joint_is_actuator = model.joints.act_type.numpy() != JointActuationType.PASSIVE
    actuated_coord_offsets = coord_offsets[joint_is_actuator]
    actuated_coords_sizes = joint_num_coords[joint_is_actuator]
    actuated_dof_offsets = dof_offsets[joint_is_actuator]
    actuated_dofs_sizes = joint_num_dofs[joint_is_actuator]
    actuator_dof_types = joint_dof_types[joint_is_actuator]

    return actuated_coord_offsets, actuated_coords_sizes, actuated_dof_offsets, actuated_dofs_sizes, actuator_dof_types


def standardize_actuated_coords(
    actuators_q: np.ndarray, actuated_coords_sizes: np.ndarray, actuator_dof_types: np.ndarray
) -> np.ndarray:
    """
    Helper function converting actuator coordinates to their canonical, comparable form.
    More specifically, angles are mapped to the [0, 2 * pi) range, and unit quaternions to their
    representation with a positive real part.
    """

    def standardize_angle(angle):
        return np.mod(angle, 2.0 * np.pi)

    def standardize_quat(quat):
        return -quat if quat[3] < 0.0 else quat

    res = actuators_q.copy()
    coord_id = 0
    for i, dof_type in enumerate(actuator_dof_types):
        if dof_type == JointDoFType.CYLINDRICAL:
            res[coord_id + 1] = standardize_angle(res[coord_id + 1])
        elif dof_type == JointDoFType.FREE:
            res[coord_id + 3 : coord_id + 7] = standardize_quat(res[coord_id + 3 : coord_id + 7])
        if dof_type == JointDoFType.REVOLUTE:
            res[coord_id] = standardize_angle(res[coord_id])
        elif dof_type == JointDoFType.SPHERICAL:
            res[coord_id : coord_id + 4] = standardize_quat(res[coord_id : coord_id + 4])
        if dof_type == JointDoFType.UNIVERSAL:
            res[coord_id] = standardize_angle(res[coord_id])
            res[coord_id + 1] = standardize_angle(res[coord_id + 1])
        coord_id += actuated_coords_sizes[i]
    return res


def extract_segments(array, offsets, sizes):
    """
    Helper function extracting from a flat array the segments with given offsets and sizes
    and returning their concatenation
    """
    res = []
    for i in range(len(offsets)):
        res.extend(array[offsets[i] : offsets[i] + sizes[i]])
    return np.array(res)


def compute_constraint_residual_mask(model: ModelKamino):
    """
    Computes a boolean mask for constraint residuals, True for most constraints but False
    for base joints (to filter out residuals for fixed base models if the base is reset
    to a different pose)
    """
    mask = np.array(model.size.sum_of_num_joint_cts * [True])

    # Exclude base joints
    first_joint_ct_id = model.joints.kinematic_cts_offset.numpy().copy()  # Cts offset per joint
    num_joint_cts = model.joints.num_kinematic_cts.numpy()  # Num cts per joint
    base_joint_index = model.info.base_joint_index.numpy().tolist()
    for wd_id in range(model.size.num_worlds):
        if base_joint_index[wd_id] < 0:
            continue
        base_jt_id = base_joint_index[wd_id]
        ct_offset = first_joint_ct_id[base_jt_id]
        mask[ct_offset : ct_offset + num_joint_cts[base_jt_id]] = False

    return mask


def generate_random_inputs_q(
    model: ModelKamino,
    num_poses: int,
    max_base_q: np.ndarray,
    max_actuators_q: np.ndarray,
    rng: np.random._generator.Generator,
    unit_quaternions=True,
):
    # Check dimensions
    base_q_size = 7 * model.size.num_worlds
    actuators_q_size = model.size.sum_of_num_actuated_joint_dofs
    assert len(max_base_q) == base_q_size
    assert len(max_actuators_q) == actuators_q_size

    # Generate (random) base_q, actuators_q
    base_q_np = np.zeros((num_poses, base_q_size))
    for i in range(base_q_size):
        base_q_np[:, i] = rng.uniform(-max_base_q[i], max_base_q[i], num_poses)
    actuators_q_np = np.zeros((num_poses, actuators_q_size))
    for i in range(actuators_q_size):
        actuators_q_np[:, i] = rng.uniform(-max_actuators_q[i], max_actuators_q[i], num_poses)

    # Normalize quaternions in base_q, actuators_q
    if unit_quaternions:
        for i in range(model.size.num_worlds):
            base_q_np[:, 7 * i + 3 : 7 * i + 7] /= np.linalg.norm(base_q_np[:, 7 * i + 3 : 7 * i + 7], axis=1)[:, None]
        quat_ids = get_actuators_q_quaternion_first_ids(model)
        for i in quat_ids:
            actuators_q_np[:, i : i + 4] /= np.linalg.norm(actuators_q_np[:, i : i + 4], axis=1)[:, None]

    return base_q_np, actuators_q_np


def generate_random_inputs_u(
    model: ModelKamino,
    num_poses: int,
    max_base_u: np.ndarray,
    max_actuators_u: np.ndarray,
    rng: np.random._generator.Generator,
):
    # Check dimensions
    base_u_size = 6 * model.size.num_worlds
    actuators_u_size = model.size.sum_of_num_actuated_joint_dofs
    assert len(max_base_u) == base_u_size
    assert len(max_actuators_u) == actuators_u_size

    # Generate (random) base_u, actuators_u
    base_u_np = np.zeros((num_poses, base_u_size))
    for i in range(base_u_size):
        base_u_np[:, i] = rng.uniform(-max_base_u[i], max_base_u[i], num_poses)
    actuators_u_np = np.zeros((num_poses, actuators_u_size))
    for i in range(actuators_u_size):
        actuators_u_np[:, i] = rng.uniform(-max_actuators_u[i], max_actuators_u[i], num_poses)

    return base_u_np, actuators_u_np


def generate_random_poses(
    model: ModelKamino,
    num_poses: int,
    max_bodies_q: np.ndarray,
    rng: np.random._generator.Generator,
    unit_quaternions=True,
):
    # Check dimensions
    bodies_q_size = 7 * model.size.sum_of_num_bodies
    assert len(max_bodies_q) == bodies_q_size

    # Generate (random) bodies_q
    bodies_q_np = np.zeros((num_poses, bodies_q_size))
    for i in range(bodies_q_size):
        bodies_q_np[:, i] = rng.uniform(-max_bodies_q[i], max_bodies_q[i], num_poses)

    # Normalize quaternions in bodies_q
    if unit_quaternions:
        for i in range(model.size.num_worlds):
            bodies_q_np[:, 7 * i + 3 : 7 * i + 7] /= np.linalg.norm(bodies_q_np[:, 7 * i + 3 : 7 * i + 7], axis=1)[
                :, None
            ]

    return bodies_q_np


def simulate_random_poses(
    model: ModelKamino,
    num_poses: int,
    max_base_q: np.ndarray,
    max_actuators_q: np.ndarray,
    max_base_u: np.ndarray,
    max_actuators_u: np.ndarray,
    rng: np.random._generator.Generator,
    use_graph: bool = False,
    verbose: bool = False,
):
    # Generate random inputs
    base_q_np, actuators_q_np = generate_random_inputs_q(model, num_poses, max_base_q, max_actuators_q, rng)
    base_u_np, actuators_u_np = generate_random_inputs_u(model, num_poses, max_base_u, max_actuators_u, rng)

    # Precompute offset arrays for extracting actuator coordinates/dofs
    actuated_coord_offsets, actuated_coords_sizes, actuated_dof_offsets, actuated_dofs_sizes, actuator_dof_types = (
        compute_actuated_coords_and_dofs_data(model)
    )

    # Precompute boolean mask for extracting relevant constraint residuals
    residual_mask = compute_constraint_residual_mask(model)

    # Run forward kinematics on all random poses
    config = ForwardKinematicsSolver.Config()
    config.reset_state = True
    config.use_sparsity = False  # Change for sparse/dense solver
    config.preconditioner = "jacobi_block_diagonal"  # Change to test preconditioners
    config.add_axis_joints = True  # Set to False to check that axis joints are needed for tie rod example
    solver = ForwardKinematicsSolver(model, config)
    success_flags = []
    with wp.ScopedDevice(model.device):
        bodies_q = wp.array(shape=(model.size.sum_of_num_bodies), dtype=wp.transformf)
        base_q = wp.array(shape=(model.size.num_worlds), dtype=wp.transformf)
        actuators_q = wp.array(shape=(actuators_q_np.shape[1]), dtype=wp.float32)
        bodies_u = wp.array(shape=(model.size.sum_of_num_bodies), dtype=vec6f)
        base_u = wp.array(shape=(model.size.num_worlds), dtype=vec6f)
        actuators_u = wp.array(shape=(actuators_u_np.shape[1]), dtype=wp.float32)
    data = model.data(device=model.device)
    epsilon = 1e-4
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
        compute_joints_data(model=model, data=data, q_j_p=model.joints.q_j_0, correction=JointCorrectionMode.CONTINUOUS)

        # Validate positions computation
        residual_ct_pos = np.max(np.abs(data.joints.r_j.numpy()[residual_mask]))
        if residual_ct_pos > epsilon:
            print(f"Large constraint residual ({residual_ct_pos}) for pose {pose_id}")
            success_flags[-1] = False
        actuators_q_check = extract_segments(data.joints.q_j.numpy(), actuated_coord_offsets, actuated_coords_sizes)
        actuators_q_check = standardize_actuated_coords(actuators_q_check, actuated_coords_sizes, actuator_dof_types)
        actuators_q_ref = standardize_actuated_coords(
            actuators_q_np[pose_id], actuated_coords_sizes, actuator_dof_types
        )
        residual_actuators_q = np.max(np.abs(actuators_q_check - actuators_q_ref))
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

        # Load the DR TestMech model from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")

        # Load model
        builder = USDImporter().import_from(asset_file)
        builder.set_base_joint("base")
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        base_q_max = np.array(3 * [0.2] + 4 * [1.0])
        actuators_q_max = np.radians([360.0])
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

        # Load the DR TestMech and DR Legs models from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_boxes.usda")
        builder = USDImporter().import_from(asset_file)
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

        # Load the DR TestMech and DR Legs models from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file_0 = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")
        asset_file_1 = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_boxes.usda")
        builder = USDImporter().import_from(asset_file_0)
        builder.set_base_joint("base")
        builder1 = USDImporter().import_from(asset_file_1)
        builder1.set_base_body("pelvis")
        builder.add_builder(builder1)
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max_test_mech = np.radians(360.0)
        theta_max_dr_legs = np.radians(10.0)
        base_q_max = np.array(3 * [0.2] + 4 * [1.0] + 3 * [0.2] + 4 * [1.0])
        actuators_q_max = np.array([theta_max_test_mech] + builder1.num_actuated_joint_coords * [theta_max_dr_legs])
        base_u_max = np.array(3 * [0.1] + 3 * [0.5] + 3 * [0.5] + 3 * [0.5])
        actuators_u_max = np.array(model.size.sum_of_num_actuated_joint_dofs * [0.5])
        success = simulate_random_poses(
            model, num_poses, base_q_max, actuators_q_max, base_u_max, actuators_u_max, rng, self.has_cuda, self.verbose
        )
        self.assertTrue(success)


class FourBarTieRodRandomPosesCheckForwardKinematics(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.has_cuda = self.default_device.is_cuda
        self.verbose = test_context.verbose

    def tearDown(self):
        self.default_device = None

    def test_four_bar_tie_rod_model_FK_random_poses(self):
        # Initialize RNG
        test_name = "Four-bar with tie rod FK random poses check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Create a builder with 10 worlds, each with a four-bar with a tie rod
        builder = make_homogeneous_builder(num_worlds=10, build_fn=create_four_bar_tie_rod)
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Simulate random poses
        num_poses = 30
        theta_max = np.radians(30.0)
        base_q_max = np.array(builder.num_worlds * (3 * [0.2] + 4 * [1.0]))
        actuators_q_max = np.array(builder.num_actuated_joint_coords * [theta_max])
        base_u_max = np.array(builder.num_worlds * (3 * [0.5] + 3 * [0.5]))
        actuators_u_max = np.array(model.size.sum_of_num_actuated_joint_dofs * [0.5])
        success = simulate_random_poses(
            model, num_poses, base_q_max, actuators_q_max, base_u_max, actuators_u_max, rng, self.has_cuda, self.verbose
        )
        self.assertTrue(success)


class HeterogenousModelSparseJacobianAssemblyCheck(unittest.TestCase):
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
        test_name = "Heterogenous model (test mechanism + dr_legs) sparse Jacobian assembly check"
        seed = int(hashlib.sha256(test_name.encode("utf8")).hexdigest(), 16)
        rng = np.random.default_rng(seed)

        # Load the DR TestMech and DR Legs models from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file_0 = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")
        asset_file_1 = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_boxes.usda")
        builder = USDImporter().import_from(asset_file_0)
        builder.set_base_joint("base")
        builder1 = USDImporter().import_from(asset_file_1)
        builder1.set_base_body("pelvis")
        builder.add_builder(builder1)
        model = builder.finalize(device=self.default_device, requires_grad=False)

        # Generate random poses
        num_poses = 30
        bodies_q_max = np.array(model.size.sum_of_num_bodies * [0.2, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0])
        theta_max_test_mech = np.radians(360.0)
        theta_max_dr_legs = np.radians(10.0)
        base_q_max = np.array(3 * [0.2] + 4 * [1.0] + 3 * [0.2] + 4 * [1.0])
        actuators_q_max = np.array([theta_max_test_mech] + builder1.num_actuated_joint_coords * [theta_max_dr_legs])
        bodies_q_np = generate_random_poses(model, num_poses, bodies_q_max, rng, False)
        base_q_np, actuators_q_np = generate_random_inputs_q(model, num_poses, base_q_max, actuators_q_max, rng)

        # Assemble and compare dense and sparse Jacobian for each pose
        solver = ForwardKinematicsSolver(model, config=ForwardKinematicsSolver.Config(use_sparsity=True))
        with wp.ScopedDevice(model.device):
            bodies_q = wp.array(shape=(model.size.sum_of_num_bodies), dtype=wp.transformf)
            base_q = wp.array(shape=(model.size.num_worlds), dtype=wp.transformf)
            actuators_q = wp.array(shape=(actuators_q_np.shape[1]), dtype=wp.float32)
        dims = solver.sparse_jacobian.dims.numpy()

        for pose_id in range(num_poses):
            bodies_q.assign(bodies_q_np[pose_id])
            base_q.assign(base_q_np[pose_id])
            actuators_q.assign(actuators_q_np[pose_id])
            transforms = solver.eval_position_control_transformations(actuators_q, base_q)

            jac_dense_np = solver.eval_kinematic_constraints_jacobian(bodies_q, transforms).numpy()
            solver.assemble_sparse_jacobian(bodies_q, transforms)
            jac_sparse_np = solver.sparse_jacobian.numpy()

            for wd_id in range(model.size.num_worlds):
                rows, cols = int(dims[wd_id][0]), int(dims[wd_id][1])
                residual = jac_dense_np[wd_id, :rows, :cols] - jac_sparse_np[wd_id]
                self.assertTrue(np.max(np.abs(residual)) < 1e-10)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
