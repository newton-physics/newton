# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the ForwardKinematicsSolver class of Kamino, in `solvers/fk.py`.
"""

import hashlib
import unittest
from functools import partial

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.joints import JointActuationType, JointCorrectionMode, JointDoFType
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.kinematics.joints import compute_joints_data
from newton._src.solvers.kamino._src.solvers.fk import ForwardKinematicsSolver
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.diff_check import diff_check
from newton.tests.kamino.utils.joints import (
    run_test_single_joint_examples,
)
from newton.tests.kamino.utils.sampling import (
    sample_actuator_coords,
    sample_actuator_velocities,
    sample_base_state,
    sample_body_poses,
    sample_world_mask,
)
from newton.tests.utils.basics import build_boxes_fourbar, build_cartpole
from newton.tests.utils.testing import (
    build_all_joints_test,
    build_unary_revolute_joint_test,
    build_unary_universal_joint_test,
)

###
# Helpers
###


def rng_seed_from_string(name: str) -> int:
    """Hash a test name to a deterministic seed for ``np.random.default_rng``."""
    return int(hashlib.sha256(name.encode("utf8")).hexdigest(), 16)


def create_four_bar_tie_rod() -> newton.ModelBuilder:
    """
    Creates a four-bar linkage, but with two revolute joints replaced with
    spherical joints so as to create a tie rod (to test axis joints).
    """
    return build_boxes_fourbar(
        fixedbase=False,
        floatingbase=True,
        limits=False,
        ground=False,
        verbose=False,
        dynamic_joints=False,
        implicit_pd=False,
        actuator_ids=[1],
        spherical_joints=[2, 3],
    )


def compute_actuated_coords_and_dofs_data(model: ModelKamino):
    """
    Compute offsets/sizes needed to extract actuated joint coordinates and DoFs from all joint
    coordinates/DoFs, along with the corresponding DoF types.

    Returns ``(actuated_coords_offsets, actuated_coords_sizes, actuated_dofs_offsets,
    actuated_dofs_sizes, actuator_dof_types)``.
    """
    # Retrieve data for all joints (offset arrays include a trailing total)
    coord_offsets = model.joints.coords_offset.numpy()[:-1]
    joint_num_coords = model.joints.num_coords.numpy()
    dof_offsets = model.joints.dofs_offset.numpy()[:-1]
    joint_num_dofs = model.joints.num_dofs.numpy()
    joint_dof_types = model.joints.dof_type.numpy()

    # Filter for actuators only
    joint_is_actuator = model.joints.act_type.numpy() != JointActuationType.PASSIVE
    if model.joints.fk_act_flag is not None:
        fk_act_flag_np = model.joints.fk_act_flag.numpy()
        joint_is_actuator_fk = fk_act_flag_np == 1
        overwrite_mask = fk_act_flag_np != -1
        joint_is_actuator[overwrite_mask] = joint_is_actuator_fk[overwrite_mask]
    actuated_coord_offsets = coord_offsets[joint_is_actuator]
    actuated_coords_sizes = joint_num_coords[joint_is_actuator]
    actuated_dof_offsets = dof_offsets[joint_is_actuator]
    actuated_dofs_sizes = joint_num_dofs[joint_is_actuator]
    actuator_dof_types = joint_dof_types[joint_is_actuator]

    return actuated_coord_offsets, actuated_coords_sizes, actuated_dof_offsets, actuated_dofs_sizes, actuator_dof_types


def standardize_actuated_coords(
    actuator_q: np.ndarray, actuated_coords_sizes: np.ndarray, actuator_dof_types: np.ndarray
) -> np.ndarray:
    """
    Convert actuator coordinates to their canonical, comparable form: angles are mapped to the
    ``[0, 2 * pi)`` range and unit quaternions to their representation with a positive real part.
    """

    def standardize_angle(angle):
        return np.mod(angle, 2.0 * np.pi)

    def standardize_quat(quat):
        return -quat if quat[3] < 0.0 else quat

    res = actuator_q.copy()
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
    """Extract from a flat array the segments with given offsets and sizes, and concatenate them."""
    res = []
    for i in range(len(offsets)):
        res.extend(array[offsets[i] : offsets[i] + sizes[i]])
    return np.array(res)


def solve_and_check_fk_random_poses(
    model: ModelKamino,
    num_poses: int,
    rng: np.random.Generator,
    max_pos: float = 0.1,
    max_angle: float = np.radians(20.0),
    max_lin_vel: float = 0.5,
    max_ang_vel: float = np.radians(90.0),
    randomize_base: bool = True,
    use_graph: bool = False,
    verbose: bool = False,
    epsilon: float | None = None,
    **config_kwargs,
):
    """Sample random actuator/base coords and twists for a given model, run position- and
    velocity-level FK on these inputs, and validate the result
    """

    num_worlds = model.size.num_worlds

    # Sample per-pose inputs and a non-trivial world mask
    base_q_np, base_u_np = sample_base_state(num_worlds, rng, num_poses)
    actuator_q_np = sample_actuator_coords(
        model, rng, num_poses, max_pos=max_pos, max_angle=max_angle, use_fk_actuators=True
    )
    actuator_u_np = sample_actuator_velocities(
        model, rng, num_poses, max_lin_vel=max_lin_vel, max_ang_vel=max_ang_vel, use_fk_actuators=True
    )
    world_masks_np = sample_world_mask(num_worlds, rng, num_samples=num_poses, target_inactive_rate=0.2)

    # Precompute offset arrays for extracting actuator coordinates/dofs
    (
        actuated_coord_offsets,
        actuated_coords_sizes,
        actuated_dof_offsets,
        actuated_dofs_sizes,
        actuator_dof_types,
    ) = compute_actuated_coords_and_dofs_data(model)

    # Precompute cumulative offsets into the flat reference actuator vectors
    ref_coord_offsets = np.concatenate(([0], np.cumsum(actuated_coords_sizes)[:-1])).astype(np.int32)
    ref_dof_offsets = np.concatenate(([0], np.cumsum(actuated_dofs_sizes)[:-1])).astype(np.int32)

    # Per-world slicing metadata used to validate active worlds and check inactive worlds are untouched
    bodies_offset_np = model.info.bodies_offset.numpy()  # shape (num_worlds + 1,)
    kin_cts_offset_np = model.info.joint_kinematic_cts_offset.numpy()  # shape (num_worlds,)
    kin_cts_size_np = model.info.num_joint_kinematic_cts.numpy()
    joints_wid_np = model.joints.wid.numpy()
    is_actuator_np = model.joints.act_type.numpy() != JointActuationType.PASSIVE
    if model.joints.fk_act_flag is not None:
        fk_act_flag_np = model.joints.fk_act_flag.numpy()
        is_actuator_np = is_actuator_np.copy()
        is_actuator_np[fk_act_flag_np != -1] = fk_act_flag_np[fk_act_flag_np != -1] == 1
    actuator_joint_wid_np = joints_wid_np[is_actuator_np]

    # Allocate solver I/O arrays and run FK on each pose
    config = ForwardKinematicsSolver.Config(**config_kwargs)
    solver = ForwardKinematicsSolver(model, config)
    success_flags = []
    with wp.ScopedDevice(model.device):
        body_q = wp.zeros(shape=(model.size.sum_of_num_bodies), dtype=wp.transformf)
        base_q = wp.zeros(shape=(num_worlds), dtype=wp.transformf)
        actuator_q = wp.zeros(shape=(actuator_q_np.shape[1]), dtype=wp.float32)
        body_u = wp.zeros(shape=(model.size.sum_of_num_bodies), dtype=wp.spatial_vectorf)
        base_u = wp.zeros(shape=(num_worlds), dtype=wp.spatial_vectorf)
        actuator_u = wp.zeros(shape=(actuator_u_np.shape[1]), dtype=wp.float32)
        world_mask = wp.zeros(shape=(num_worlds,), dtype=wp.bool)
    data = model.data(device=model.device)
    if epsilon is None:
        epsilon = 1e-3 if config.use_regularization else 1e-4
    for pose_id in range(num_poses):
        # Assign per-pose inputs and world mask
        base_q.assign(base_q_np[pose_id])
        actuator_q.assign(actuator_q_np[pose_id])
        base_u.assign(base_u_np[pose_id])
        actuator_u.assign(actuator_u_np[pose_id])
        active_mask_np = world_masks_np[pose_id]
        world_mask.assign(active_mask_np)
        actuator_joint_is_active = active_mask_np[actuator_joint_wid_np]
        active_world_ids = np.where(active_mask_np)[0]
        inactive_world_ids = np.where(~active_mask_np)[0]

        # Snapshot body arrays so we can verify inactive worlds are not modified by the solve
        body_q_before_np = body_q.numpy().copy()
        body_u_before_np = body_u.numpy().copy()

        # Run FK solve
        status = solver.solve_fk(
            actuator_q,
            body_q,
            base_q=base_q if randomize_base else None,
            base_u=base_u if randomize_base else None,
            actuator_u=actuator_u,
            body_u=body_u,
            world_mask=world_mask,
            use_graph=use_graph,
            verbose=verbose,
            return_status=True,
        )

        # Verify body poses/velocities of inactive worlds were left untouched by the solver
        body_q_after_np = body_q.numpy()
        body_u_after_np = body_u.numpy()
        inactive_untouched = True
        for wid in inactive_world_ids:
            b0, b1 = int(bodies_offset_np[wid]), int(bodies_offset_np[wid + 1])
            if not np.array_equal(body_q_before_np[b0:b1], body_q_after_np[b0:b1]) or not np.array_equal(
                body_u_before_np[b0:b1], body_u_after_np[b0:b1]
            ):
                inactive_untouched = False
                print(f"Inactive world {wid} was modified by the solver for pose {pose_id}")
                break
        if not inactive_untouched:
            success_flags.append(False)
            continue

        # Check convergence on active worlds
        if status.success[active_mask_np].min() < 1:
            success_flags.append(False)
            continue
        success_flags.append(True)

        # Update joints data from body states for validation
        wp.copy(data.bodies.q_i, body_q)
        wp.copy(data.bodies.u_i, body_u)
        compute_joints_data(model=model, data=data, q_j_p=model.joints.q_j_0, correction=JointCorrectionMode.CONTINUOUS)

        # Filter joint- and actuator-level slicing to active worlds
        active_coord_offsets = actuated_coord_offsets[actuator_joint_is_active]
        active_coord_sizes = actuated_coords_sizes[actuator_joint_is_active]
        active_dof_offsets = actuated_dof_offsets[actuator_joint_is_active]
        active_dof_sizes = actuated_dofs_sizes[actuator_joint_is_active]
        active_dof_types = actuator_dof_types[actuator_joint_is_active]
        active_ref_coord_offsets = ref_coord_offsets[actuator_joint_is_active]
        active_ref_dof_offsets = ref_dof_offsets[actuator_joint_is_active]

        # Validate positions computation on active worlds
        r_j_np = data.joints.r_j.numpy()
        residual_ct_pos = 0.0
        for wid in active_world_ids:
            r0 = int(kin_cts_offset_np[wid])
            r1 = r0 + int(kin_cts_size_np[wid])
            if r1 > r0:
                residual_ct_pos = max(residual_ct_pos, float(np.max(np.abs(r_j_np[r0:r1]))))
        if residual_ct_pos > epsilon:
            print(f"Large constraint residual ({residual_ct_pos}) for pose {pose_id}")
            success_flags[-1] = False
        actuator_q_check = extract_segments(data.joints.q_j.numpy(), active_coord_offsets, active_coord_sizes)
        actuator_q_check = standardize_actuated_coords(actuator_q_check, active_coord_sizes, active_dof_types)
        actuator_q_ref = extract_segments(actuator_q_np[pose_id], active_ref_coord_offsets, active_coord_sizes)
        actuator_q_ref = standardize_actuated_coords(actuator_q_ref, active_coord_sizes, active_dof_types)
        residual_actuator_q = float(np.max(np.abs(actuator_q_check - actuator_q_ref))) if actuator_q_check.size else 0.0
        if residual_actuator_q > epsilon:
            print(f"Large error on prescribed actuator coordinates ({residual_actuator_q}) for pose {pose_id}")
            success_flags[-1] = False

        # Validate velocities computation on active worlds
        dr_j_np = data.joints.dr_j.numpy()
        residual_ct_vel = 0.0
        for wid in active_world_ids:
            r0 = int(kin_cts_offset_np[wid])
            r1 = r0 + int(kin_cts_size_np[wid])
            if r1 > r0:
                residual_ct_vel = max(residual_ct_vel, float(np.max(np.abs(dr_j_np[r0:r1]))))
        if residual_ct_vel > epsilon:
            print(f"Large constraint velocity residual ({residual_ct_vel}) for pose {pose_id}")
            success_flags[-1] = False
        actuator_u_check = extract_segments(data.joints.dq_j.numpy(), active_dof_offsets, active_dof_sizes)
        actuator_u_ref = extract_segments(actuator_u_np[pose_id], active_ref_dof_offsets, active_dof_sizes)
        residual_actuator_u = float(np.max(np.abs(actuator_u_check - actuator_u_ref))) if actuator_u_check.size else 0.0
        if residual_actuator_u > epsilon:
            print(f"Large error on prescribed actuator velocities ({residual_actuator_u}) for pose {pose_id}")
            success_flags[-1] = False

    success = np.sum(success_flags) == num_poses
    if not success:
        print(f"Random poses simulation & validation failed, {np.sum(success_flags)}/{num_poses} poses successful")

    return success


###
# Tests
###


class TestJacobianAssembly(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.has_cuda = self.default_device.is_cuda
        self.verbose = test_context.verbose

    def tearDown(self):
        self.default_device = None

    def test_finite_differences_single_joint(self):
        """Match analytic and finite-difference Jacobians on every single-joint fixture."""
        test_name = "Forward Kinematics Jacobian check"
        rng = np.random.default_rng(rng_seed_from_string(test_name))

        def test_function(model: ModelKamino):
            assert model.size.num_worlds == 1  # For simplicity we assume a single world

            # Generate (random) body poses
            body_q_np = rng.uniform(-1.0, 1.0, 7 * model.size.sum_of_num_bodies).astype("float32")
            body_q = wp.from_numpy(body_q_np, dtype=wp.transformf, device=model.device)

            # Generate (random) actuated coordinates
            actuator_q_np = rng.uniform(-1.0, 1.0, model.size.sum_of_num_actuated_joint_coords).astype("float32")
            actuator_q = wp.from_numpy(actuator_q_np, dtype=wp.float32, device=model.device)

            # Evaluate analytic Jacobian
            solver = ForwardKinematicsSolver(model=model)
            target_rel_transforms = solver.eval_target_relative_transforms(actuator_q, None)
            jacobian = solver.eval_kinematic_constraints_jacobian(body_q, target_rel_transforms)

            # Check against finite differences Jacobian
            def eval_constraints(body_q_stepped_np):
                body_q.assign(body_q_stepped_np)
                constraints = solver.eval_kinematic_constraints(body_q, target_rel_transforms)
                body_q.assign(body_q_np)  # Reset state
                return constraints.numpy()[0]

            return diff_check(
                eval_constraints,
                jacobian.numpy()[0],
                body_q_np,
                epsilon=1e-4,
                tolerance_abs=5e-3,
                tolerance_rel=5e-3,
            )

        success = run_test_single_joint_examples(test_function, test_name, device=self.default_device)
        self.assertTrue(success)

    def test_sparse_matches_dense_single_joint(self):
        """Match dense and sparse Jacobians for every single-joint fixture."""
        test_name = "Single-joint sparse Jacobian assembly check"
        rng = np.random.default_rng(42)

        def test_function(model: ModelKamino):
            """Compare the dense and sparse Jacobians for a random body state."""
            body_q_np = rng.uniform(-1.0, 1.0, 7 * model.size.sum_of_num_bodies).astype("float32")
            body_q = wp.from_numpy(body_q_np, dtype=wp.transformf, device=model.device)
            actuator_q = wp.zeros(
                shape=model.size.sum_of_num_actuated_joint_coords, dtype=wp.float32, device=model.device
            )
            solver = ForwardKinematicsSolver(model, config=ForwardKinematicsSolver.Config(use_sparsity=True))
            transforms = solver.eval_target_relative_transforms(actuator_q, None)

            jac_dense_np = solver.eval_kinematic_constraints_jacobian(body_q, transforms).numpy()
            solver.assemble_sparse_jacobian(body_q, transforms)
            jac_sparse_np = solver.data.problem.sparse_jacobian.numpy()
            rows, cols = solver.data.problem.sparse_jacobian.dims.numpy()[0]
            return np.allclose(jac_dense_np[0, :rows, :cols], jac_sparse_np[0], atol=1e-6, rtol=0.0)

        success = run_test_single_joint_examples(test_function, test_name, device=self.default_device)
        self.assertTrue(success)

    def test_sparse_matches_dense_heterogenous(self):
        """Match dense and sparse Jacobians on a heterogenous (TestMech + DR Legs) model."""
        test_name = "Heterogenous model (test mechanism + dr_legs) sparse Jacobian assembly check"
        rng = np.random.default_rng(rng_seed_from_string(test_name))

        # Load the DR TestMech and DR Legs models from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file_0 = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")
        asset_file_1 = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_boxes.usda")
        builder = newton.ModelBuilder()
        builder.begin_world()
        builder.add_usd(source=asset_file_0)
        builder.end_world()
        builder.begin_world()
        builder.add_usd(source=asset_file_1)
        builder.end_world()
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # Generate random poses
        num_poses = 30
        body_q_np = sample_body_poses(
            model.size.sum_of_num_bodies,
            rng,
            num_poses,
            max_pos=0.05,
            max_angle=np.radians(20.0),
            unit_quaternions=False,
        )
        base_q_np, _ = sample_base_state(
            model.size.num_worlds,
            rng,
            num_poses,
        )
        actuator_q_np = sample_actuator_coords(model, rng, num_poses)

        # Assemble and compare dense and sparse Jacobian for each pose
        solver = ForwardKinematicsSolver(model, config=ForwardKinematicsSolver.Config(use_sparsity=True))
        with wp.ScopedDevice(model.device):
            body_q = wp.zeros(shape=(model.size.sum_of_num_bodies), dtype=wp.transformf)
            base_q = wp.zeros(shape=(model.size.num_worlds), dtype=wp.transformf)
            actuator_q = wp.zeros(shape=(actuator_q_np.shape[1]), dtype=wp.float32)
        dims = solver.data.problem.sparse_jacobian.dims.numpy()

        for pose_id in range(num_poses):
            body_q.assign(body_q_np[pose_id])
            base_q.assign(base_q_np[pose_id])
            actuator_q.assign(actuator_q_np[pose_id])
            transforms = solver.eval_target_relative_transforms(actuator_q, base_q)

            jac_dense_np = solver.eval_kinematic_constraints_jacobian(body_q, transforms).numpy()
            solver.assemble_sparse_jacobian(body_q, transforms)
            jac_sparse_np = solver.data.problem.sparse_jacobian.numpy()

            for wd_id in range(model.size.num_worlds):
                rows, cols = int(dims[wd_id][0]), int(dims[wd_id][1])
                residual = jac_dense_np[wd_id, :rows, :cols] - jac_sparse_np[wd_id]
                self.assertLess(np.max(np.abs(residual)), 3e-6)


class TestPerDofActuation(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_reject_mixed_passive_and_actuated_dofs(self):
        """Reject a joint with a mixed passive and actuated DoF partition."""
        builder = build_unary_universal_joint_test(limits=True, ground=False)
        builder.joint_target_mode[0] = newton.JointTargetMode.NONE
        builder.joint_target_mode[1] = newton.JointTargetMode.POSITION
        builder.joint_target_ke[1] = 1.0
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        with self.assertRaisesRegex(ValueError, "all DoFs must be passive or all must be actuated"):
            ForwardKinematicsSolver(model)

    def test_accept_differing_actuated_modes(self):
        """Accept differing non-passive modes because FK uses one joint partition."""
        builder = build_unary_universal_joint_test(limits=True, ground=False)
        builder.joint_target_mode[0] = newton.JointTargetMode.POSITION
        builder.joint_target_mode[1] = newton.JointTargetMode.VELOCITY
        builder.joint_target_ke[0] = 1.0
        builder.joint_target_kd[1] = 1.0
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        ForwardKinematicsSolver(model)


class TestPassiveUniversalJointFrame(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_follower_joint_frame(self):
        """Test kinematic constraints and jacobian for a passive universal joint.

        The follower joint frame is rotated relative to the base joint frame,
        while the follower body is counter-rotated so both joint frames coincide
        in world coordinates. The base-frame X axis and follower-frame Y axis
        are therefore orthogonal, as required by a universal joint.
        """
        # Build a single body attached to the world and make both rotational
        # degrees of freedom passive.
        builder = build_unary_universal_joint_test(limits=True, ground=False)
        builder.joint_target_mode[0] = newton.JointTargetMode.NONE
        builder.joint_target_mode[1] = newton.JointTargetMode.NONE
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # Rotate the follower-local joint frame by 90 degrees about Z without
        # changing the base-local joint frame.
        q_X_F = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), 0.5 * wp.pi)
        X_Fj = model.joints.X_Fj.numpy()
        X_Fj[0] = np.asarray(wp.quat_to_matrix(q_X_F), dtype=np.float32).reshape(3, 3)
        model.joints.X_Fj.assign(X_Fj)

        # Counter-rotate and translate the follower body so its joint frame has
        # the same world-space pose as the base joint frame.
        q_F = wp.quat_inverse(q_X_F)
        B_r_Bj = wp.vec3f(model.joints.B_r_Bj.numpy()[0])
        F_r_Fj = wp.vec3f(model.joints.F_r_Fj.numpy()[0])
        r_F = B_r_Bj - wp.quat_rotate(q_F, F_r_Fj)
        body_q = wp.array([wp.transformf(r_F, q_F)], dtype=wp.transformf, device=self.default_device)

        # Coincident joint frames satisfy all three anchor constraints and the
        # universal joint's rotational orthogonality constraint.
        solver = ForwardKinematicsSolver(model)
        actuator_q = wp.empty(0, dtype=wp.float32, device=self.default_device)
        target_transforms = solver.eval_target_relative_transforms(actuator_q, None)
        constraints = solver.eval_kinematic_constraints(body_q, target_transforms).numpy()[0]
        np.testing.assert_allclose(constraints, 0.0, atol=1.0e-6)

        # Validate jacobian with finite differences
        body_q_np = body_q.numpy().reshape(-1)
        jacobian = solver.eval_kinematic_constraints_jacobian(body_q, target_transforms).numpy()[0]

        def eval_constraints(body_q_stepped_np):
            body_q.assign(body_q_stepped_np)
            stepped_constraints = solver.eval_kinematic_constraints(body_q, target_transforms).numpy()[0]
            body_q.assign(body_q_np)
            return stepped_constraints

        self.assertTrue(
            diff_check(
                eval_constraints,
                jacobian,
                body_q_np,
                epsilon=1.0e-4,
                tolerance_abs=5.0e-3,
                tolerance_rel=5.0e-3,
            )
        )


class TestRandomPoses(unittest.TestCase):
    """Validate FK solves on random poses for multiple models."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.has_cuda = self.default_device.is_cuda
        self.verbose = test_context.verbose

    def tearDown(self):
        self.default_device = None

    def _make_solve_and_check(self, model, num_poses, rng, **overrides):
        """Return a ``solve_and_check_fk_random_poses`` partial with the common defaults for random-pose tests."""
        kwargs = {
            "randomize_base": False,
            "use_graph": self.has_cuda and not wp.config.verify_cuda,
            "verbose": self.verbose,
            "reset_state": True,
            "use_incremental_solve": True,
            "tolerance": 1e-6,
            "preconditioner": "jacobi_block_diagonal",
        }
        kwargs.update(overrides)
        return partial(solve_and_check_fk_random_poses, model, num_poses, rng, **kwargs)

    def test_dr_testmech_random_poses(self):
        """Validate FK on random poses for the DR Test mechanism"""
        rng = np.random.default_rng(rng_seed_from_string("Test mechanism FK random poses check"))

        # Import the DR TestMech asset once and replicate it into 10 identical worlds
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")
        builder_single = newton.ModelBuilder()
        builder_single.begin_world()
        builder_single.add_usd(source=asset_file)
        builder_single.end_world()
        builder = newton.ModelBuilder()
        builder.replicate(builder=builder_single, world_count=10)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # Solve and check FK on random poses, on both dense and sparse paths
        solve_and_check_fn = self._make_solve_and_check(model, num_poses=3, rng=rng)
        self.assertTrue(solve_and_check_fn(use_sparsity=False))
        self.assertTrue(solve_and_check_fn(use_sparsity=True))

    def test_dr_legs_random_poses(self):
        """Validate FK on random poses for DR Legs"""
        rng = np.random.default_rng(rng_seed_from_string("FK random poses check for dr_legs model"))

        # Import the DR Legs asset once (with the pelvis base body set) and replicate it into 10 worlds
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_boxes.usda")
        builder_single = newton.ModelBuilder()
        builder_single.begin_world()
        builder_single.add_usd(source=asset_file)
        builder_single.end_world()
        builder = newton.ModelBuilder()
        builder.replicate(builder=builder_single, world_count=10)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # Solve and check FK on random poses, on both dense and sparse paths
        # Note: Angles too far from the initial pose lead to singularities
        solve_and_check_fn = self._make_solve_and_check(
            model,
            num_poses=3,
            rng=rng,
            max_angle=np.radians(5.0),
            max_ang_vel=np.radians(20.0),
        )
        self.assertTrue(solve_and_check_fn(use_sparsity=False))
        self.assertTrue(solve_and_check_fn(use_sparsity=True))

    def test_heterogenous_dr_testmech_and_legs_random_poses(self):
        """Validate FK on random poses for a heterogenous model (DR TestMechanism + DR Legs)"""
        rng = np.random.default_rng(
            rng_seed_from_string("Heterogenous model (test mechanism + dr_legs) FK random poses check")
        )

        # Combine 5 DR TestMech + 5 DR Legs worlds from the `newton-assets` repository
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file_0 = str(asset_path / "dr_testmech" / "usd" / "dr_testmech.usda")
        asset_file_1 = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_boxes.usda")
        builder_0 = newton.ModelBuilder()
        builder_0.begin_world()
        builder_0.add_usd(source=asset_file_0)
        builder_0.end_world()
        builder_1 = newton.ModelBuilder()
        builder_1.begin_world()
        builder_1.add_usd(source=asset_file_1)
        builder_1.end_world()
        builder = newton.ModelBuilder()
        for _ in range(5):
            builder.begin_world()
            builder.add_builder(builder_0)
            builder.end_world()
        for _ in range(5):
            builder.begin_world()
            builder.add_builder(builder_1)
            builder.end_world()
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # Solve and check FK on random poses, on both dense and sparse paths
        # Note 1: Angles too far from the initial pose lead to singularities.
        # Note 2: Expect a warning due to a specified base for DR TestMech worlds (fixed-base)
        solve_and_check_fn = self._make_solve_and_check(
            model,
            num_poses=3,
            rng=rng,
            max_angle=np.radians(5.0),
            max_ang_vel=np.radians(20.0),
            randomize_base=True,
        )
        with self.assertLogs(level="WARNING"):
            self.assertTrue(solve_and_check_fn(use_sparsity=False))
        with self.assertLogs(level="WARNING"):
            self.assertTrue(solve_and_check_fn(use_sparsity=True))

    def test_four_bar_tie_rod_axis_joints_random_poses(self):
        """Validate FK on random poses for a four-bar with a tie rod, using axis joints"""
        rng = np.random.default_rng(rng_seed_from_string("Four-bar with tie rod FK random poses check (axis joints)"))
        builder = newton.ModelBuilder()
        builder.replicate(builder=create_four_bar_tie_rod(), world_count=10)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device, requires_grad=False))

        # Solve and check FK on random poses, on both dense and sparse paths
        solve_and_check_fn = self._make_solve_and_check(
            model,
            num_poses=3,
            rng=rng,
            add_axis_joints=True,
        )
        self.assertTrue(solve_and_check_fn(use_sparsity=False))
        self.assertTrue(solve_and_check_fn(use_sparsity=True))

    def test_four_bar_tie_rod_regularized_random_poses(self):
        """Validate FK on random poses for a four-bar with a tie rod, using regularization"""
        rng = np.random.default_rng(
            rng_seed_from_string("Four-bar with tie rod FK random poses check (regularization)")
        )
        builder = newton.ModelBuilder()
        builder.replicate(builder=create_four_bar_tie_rod(), world_count=10)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device, requires_grad=False))

        # Solve and check FK on random poses, on both dense and sparse paths
        solve_and_check_fn = self._make_solve_and_check(
            model,
            num_poses=3,
            rng=rng,
            add_axis_joints=False,
            use_regularization=True,
            tolerance=1e-5,
        )
        self.assertTrue(solve_and_check_fn(use_sparsity=False))
        self.assertTrue(solve_and_check_fn(use_sparsity=True))

    def test_all_joints_random_poses(self):
        """Validate FK on random poses for a model with all joint types"""
        rng = np.random.default_rng(rng_seed_from_string("All-joints example FK random poses check"))
        builder = build_all_joints_test(unary_joints=True, binary_joints=True, actuated=True, floating_base=False)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # Solve and check FK on random poses, on both dense and sparse paths
        solve_and_check_fn = self._make_solve_and_check(model, num_poses=3, rng=rng)
        self.assertTrue(solve_and_check_fn(use_sparsity=False))
        self.assertTrue(solve_and_check_fn(use_sparsity=True))

    def test_all_joints_asymmetric_frames_random_poses(self):
        """Validate FK on random poses for a model with all joint types and asymmetric joint frames"""
        rng = np.random.default_rng(
            rng_seed_from_string("All-joints example FK random poses check with asymmetric frames")
        )
        builder = build_all_joints_test(unary_joints=True, binary_joints=True, actuated=True, floating_base=False)

        # Set asymmetric joint frames (X_B != X_F) into joints while preserving the initial pose
        num_joints = builder.joint_count
        random_quats = np.resize(rng.uniform(-1.0, 1.0, 4 * num_joints), (num_joints, 4))
        random_quats /= np.linalg.norm(random_quats, axis=1)[:, None]
        for jid in range(num_joints):
            parent = builder.joint_parent[jid]
            child = builder.joint_child[jid]
            q_B = (
                wp.quat_identity(dtype=wp.float32) if parent < 0 else wp.transform_get_rotation(builder.body_q[parent])
            )
            q_F = wp.transform_get_rotation(builder.body_q[child])
            r_Bj = wp.transform_get_translation(builder.joint_X_p[jid])
            r_Fj = wp.transform_get_translation(builder.joint_X_c[jid])
            q_Fj = wp.quatf(random_quats[jid])
            q_Bj = wp.quat_inverse(q_B) * q_F * q_Fj  # Compute X_B given X_F to preserve a valid pose
            builder.joint_X_c[jid] = wp.transform(r_Fj, q_Fj)
            builder.joint_X_p[jid] = wp.transform(r_Bj, q_Bj)
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))

        # Solve and check FK on random poses, on both dense and sparse paths
        solve_and_check_fn = self._make_solve_and_check(model, num_poses=3, rng=rng)
        self.assertTrue(solve_and_check_fn(use_sparsity=False))
        self.assertTrue(solve_and_check_fn(use_sparsity=True))

    def test_cartpole_random_poses(self):
        """Validate FK on random poses for the Cartpole model"""
        rng = np.random.default_rng(rng_seed_from_string("Cartpole FK random poses check"))

        # Build a 10-world cartpole model with the revolute joint marked as FK-actuated
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        fk_actuation_flags = {1: 1}
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder, fk_actuation_flags=fk_actuation_flags)
        build_cartpole(builder=robot_builder, ground=False)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(10):
            builder.add_world(robot_builder)
        model_newton = builder.finalize(skip_validation_joints=True)
        model = ModelKamino.from_newton(model_newton)

        # Solve and check FK on random poses, on both dense and sparse paths
        solve_and_check_fn = self._make_solve_and_check(model, num_poses=3, rng=rng)
        self.assertTrue(solve_and_check_fn(use_sparsity=False))
        self.assertTrue(solve_and_check_fn(use_sparsity=True))


class TestFourBarTieRodAxisFrames(unittest.TestCase):
    """Non-random single-purpose check for axis-joint frames after model changes."""

    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_axis_joint_frames_update_after_notify(self):
        """Synthetic axis frames match a fresh solver after model changes."""
        model = ModelKamino.from_newton(
            create_four_bar_tie_rod().finalize(device=self.default_device, requires_grad=False)
        )
        config = ForwardKinematicsSolver.Config(add_axis_joints=True)
        solver = ForwardKinematicsSolver(model, config)
        axis_body_id = int(solver.data.joints.axis_body_id.numpy()[0])
        source_joint = int(solver.data.joints.axis_source_joint_0.numpy()[0])

        body_q = model.bodies.q_i_0.numpy()
        body_q[axis_body_id] = np.array(
            wp.transformf(
                wp.vec3f(*body_q[axis_body_id, :3]),
                wp.quat_from_axis_angle(wp.vec3f(0.0, 1.0, 0.0), 0.3),
            )
        )
        model.bodies.q_i_0.assign(body_q)
        if model.joints.bid_B.numpy()[source_joint] == axis_body_id:
            joint_anchor = model.joints.B_r_Bj.numpy()
            joint_anchor[source_joint] += np.array([0.05, -0.02, 0.01], dtype=np.float32)
            model.joints.B_r_Bj.assign(joint_anchor)
        else:
            joint_anchor = model.joints.F_r_Fj.numpy()
            joint_anchor[source_joint] += np.array([0.05, -0.02, 0.01], dtype=np.float32)
            model.joints.F_r_Fj.assign(joint_anchor)

        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES | newton.ModelFlags.BODY_PROPERTIES)
        reference = ForwardKinematicsSolver(model, ForwardKinematicsSolver.Config(add_axis_joints=True))
        axis_joints = solver.data.joints.axis_joint_id.numpy()

        np.testing.assert_allclose(
            solver.data.joints.X_Bj.numpy()[axis_joints],
            reference.data.joints.X_Bj.numpy()[axis_joints],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            solver.data.joints.X_Fj.numpy()[axis_joints],
            reference.data.joints.X_Fj.numpy()[axis_joints],
            atol=1e-6,
        )


class TestWarnings(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def test_solve_fk_warns_without_base_body_when_base_provided(self):
        """
        Validate that solve_fk() warns about worlds without a base body, only when a base is provided.
        """
        builder = build_unary_revolute_joint_test(ground=False)
        model_newton = builder.finalize(device=self.default_device)
        model = ModelKamino.from_newton(model_newton)
        self.assertTrue(model.info.has_world_without_base_body)

        solver = ForwardKinematicsSolver(model=model)
        identity = wp.transformf(wp.vec3f(0.0, 0.0, 0.0), wp.quat_identity(dtype=wp.float32))
        actuator_q = wp.empty(
            model.size.sum_of_num_fk_actuated_joint_coords, dtype=wp.float32, device=self.default_device
        )
        body_q = wp.array([identity] * model.size.sum_of_num_bodies, dtype=wp.transformf, device=self.default_device)

        # Without a base pose, the solve stays silent.
        with self.assertNoLogs(level="WARNING"):
            solver.solve_fk(actuator_q, body_q, use_graph=False)

        # Providing a base pose triggers the deferred warning.
        base_q = wp.array([identity], dtype=wp.transformf, device=self.default_device)
        with self.assertLogs(level="WARNING") as logs:
            solver.solve_fk(actuator_q, body_q, base_q=base_q, use_graph=False)
        self.assertTrue(any("no free-floating base body" in message for message in logs.output))


class TestMultiRhsVelocity(unittest.TestCase):
    """Verify shared-factorization velocity FK."""

    def setUp(self):
        """Initialize the shared Kamino test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        """Release the test device reference."""
        self.default_device = None

    def test_multi_rhs_matches_repeated_velocity_solves(self):
        """Match every multi-RHS body twist to an independent velocity solve."""
        builder = build_boxes_fourbar(
            fixedbase=False,
            floatingbase=True,
            limits=False,
            ground=False,
            verbose=False,
            dynamic_joints=False,
            implicit_pd=False,
            actuator_ids=[1],
        )
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        solver = ForwardKinematicsSolver(model=model)
        body_q = wp.clone(model.bodies.q_i_0)

        rhs_size = 4
        actuator_count = model.size.sum_of_num_fk_actuated_joint_dofs
        actuator_u_np = np.linspace(-0.7, 0.8, rhs_size * actuator_count, dtype=np.float32).reshape(
            rhs_size, actuator_count
        )
        base_u_np = np.array(
            [
                [0.1, -0.2, 0.3, 0.0, 0.1, -0.1],
                [0.0, 0.0, 0.0, 0.2, -0.1, 0.3],
                [-0.3, 0.1, 0.0, -0.2, 0.0, 0.1],
                [0.2, 0.2, -0.1, 0.0, -0.3, 0.0],
            ],
            dtype=np.float32,
        )

        expected = []
        for rhs_index in range(rhs_size):
            actuator_u = wp.array(actuator_u_np[rhs_index], dtype=wp.float32, device=self.default_device)
            base_u = wp.array(
                base_u_np[rhs_index : rhs_index + 1], dtype=wp.spatial_vectorf, device=self.default_device
            )
            body_u = wp.zeros(model.size.sum_of_num_bodies, dtype=wp.spatial_vectorf, device=self.default_device)
            solver.solve_for_body_velocities(actuator_u, body_q, body_u, base_u=base_u)
            expected.append(body_u.numpy())

        actuator_u = wp.array(actuator_u_np, dtype=wp.float32, device=self.default_device)
        base_u = wp.array(base_u_np[:, None, :], dtype=wp.spatial_vectorf, device=self.default_device)
        body_u = wp.zeros(
            (rhs_size, model.size.sum_of_num_bodies), dtype=wp.spatial_vectorf, device=self.default_device
        )
        with self.assertRaisesRegex(ValueError, "request_velocity_solve_batch_size"):
            solver.solve_for_body_velocities(actuator_u, body_q, body_u, base_u=base_u)

        solver.request_velocity_solve_batch_size(rhs_size)
        solver.solve_for_body_velocities(actuator_u, body_q, body_u, base_u=base_u)

        np.testing.assert_allclose(body_u.numpy(), np.asarray(expected), rtol=2.0e-4, atol=2.0e-4)

    def test_multi_rhs_preserves_linearity(self):
        """Map summed velocity inputs to the sum of their body-twist responses."""
        builder = build_boxes_fourbar(
            fixedbase=False,
            floatingbase=True,
            limits=False,
            ground=False,
            verbose=False,
            dynamic_joints=False,
            implicit_pd=False,
            actuator_ids=[1],
        )
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        solver = ForwardKinematicsSolver(model=model)
        body_q = wp.clone(model.bodies.q_i_0)

        actuator_u = wp.array([[0.0], [0.7], [0.7]], dtype=wp.float32, device=self.default_device)
        base_u = wp.array(
            [
                [[0.2, 0.0, -0.1, 0.0, 0.3, 0.0]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.2, 0.0, -0.1, 0.0, 0.3, 0.0]],
            ],
            dtype=wp.spatial_vectorf,
            device=self.default_device,
        )
        body_u = wp.zeros((3, model.size.sum_of_num_bodies), dtype=wp.spatial_vectorf, device=self.default_device)

        solver.request_velocity_solve_batch_size(3)
        solver.solve_for_body_velocities(actuator_u, body_q, body_u, base_u=base_u)

        result = body_u.numpy()
        np.testing.assert_allclose(result[2], result[0] + result[1], rtol=2.0e-4, atol=2.0e-4)
        self.assertGreater(float(np.max(np.abs(result))), 1.0e-3)

    def test_multi_rhs_refreshes_gimbal_coords_with_explicit_transforms(self):
        """Evaluate gimbal velocity axes from the current body pose."""
        builder = newton.ModelBuilder()
        body_id = builder.add_link(
            label="gimbal_body",
            mass=1.0,
            inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
        d6 = builder.add_joint_d6(
            parent=-1,
            child=body_id,
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(
                    axis=axis, actuator_mode=newton.JointTargetMode.POSITION, target_ke=1.0
                )
                for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
            ],
        )
        builder.add_articulation([d6])
        model = ModelKamino.from_newton(builder.finalize(device=self.default_device))
        solver = ForwardKinematicsSolver(model=model)
        actuator_q = wp.array([0.4, -0.3, 0.2], dtype=wp.float32, device=self.default_device)
        body_q = wp.clone(model.bodies.q_i_0)
        solver.solve_fk(actuator_q, body_q, use_graph=False)
        target_transforms = solver.eval_target_relative_transforms(actuator_q)

        actuator_u_np = np.array([[0.3, -0.4, 0.5], [-0.2, 0.1, 0.35]], dtype=np.float32)
        actuator_u = wp.array(actuator_u_np, dtype=wp.float32, device=self.default_device)
        base_u = wp.zeros(1, dtype=wp.spatial_vectorf, device=self.default_device)

        # Use independent single-RHS solves as the reference for both velocity
        # vectors at the same converged gimbal pose.
        expected = []
        for rhs_index in range(actuator_u_np.shape[0]):
            actuator_u_single = wp.array(actuator_u_np[rhs_index], dtype=wp.float32, device=self.default_device)
            body_u = wp.zeros(model.size.sum_of_num_bodies, dtype=wp.spatial_vectorf, device=self.default_device)
            solver.solve_for_body_velocities(
                actuator_u_single,
                body_q,
                body_u,
                base_u=base_u,
                target_rel_transforms=target_transforms,
            )
            expected.append(body_u.numpy())

        # Poison the coordinate scratch buffer with a different gimbal pose. The
        # batched solve must refresh it from body_q so its velocity axes do not
        # depend on state left behind by an earlier operation.
        solver.data.problem.actuator_q_next.assign([1.1, 0.7, -0.8])
        rhs_size = actuator_u_np.shape[0]
        solver.request_velocity_solve_batch_size(rhs_size)
        actual = wp.zeros(
            (rhs_size, model.size.sum_of_num_bodies), dtype=wp.spatial_vectorf, device=self.default_device
        )
        solver.solve_for_body_velocities(
            actuator_u,
            body_q,
            actual,
            base_u=wp.zeros((1, 1), dtype=wp.spatial_vectorf, device=self.default_device),
            target_rel_transforms=target_transforms,
        )

        np.testing.assert_allclose(actual.numpy(), np.asarray(expected), rtol=2.0e-4, atol=2.0e-4)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
