# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the `kamino.kinematics.resets` module"""

from __future__ import annotations

import functools
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.model import DataKamino, ModelKamino
from newton._src.solvers.kamino._src.kinematics.joints import JointCorrectionMode, compute_joints_data
from newton._src.solvers.kamino._src.kinematics.resets import reset_joints_state_from_bodies_state, set_floating_base
from newton._src.solvers.kamino._src.models.builders.basics import build_boxes_fourbar
from newton._src.solvers.kamino._src.models.builders.testing import build_all_joints_test_model
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.solvers import ForwardKinematicsSolver
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton._src.solvers.kamino.tests.utils.sampling import (
    sample_actuator_coords,
    sample_actuator_velocities,
    sample_base_state,
    sample_world_mask,
)

###
# Utils
###

rtol = 1e-6
atol = 1e-5


def assert_binary_joint_states_equal(
    model: ModelKamino,
    joint_q: wp.array[wp.float32],
    joint_q_ref: wp.array[wp.float32],
    joint_u: wp.array[wp.float32],
    joint_u_ref: wp.array[wp.float32],
):
    """Check that joint coords/velocities match provided references, except possibly for unary joints."""
    # Build boolean mask for joint coords/dofs, excluding unary joints
    coords_mask = np.array(model.size.sum_of_num_joint_coords * [True])
    dofs_mask = np.array(model.size.sum_of_num_joint_dofs * [True])
    bid_B_np = model.joints.bid_B.numpy()
    coords_offset_np = model.joints.coords_offset.numpy()
    dofs_offset_np = model.joints.dofs_offset.numpy()
    for jid in range(model.size.sum_of_num_joints):
        if bid_B_np[jid] < 0:
            coords_mask[coords_offset_np[jid] : coords_offset_np[jid + 1]] = False
            dofs_mask[dofs_offset_np[jid] : dofs_offset_np[jid + 1]] = False

    # Run masked comparison
    np.testing.assert_allclose(joint_q.numpy()[coords_mask], joint_q_ref.numpy()[coords_mask], rtol=rtol, atol=atol)
    np.testing.assert_allclose(joint_u.numpy()[dofs_mask], joint_u_ref.numpy()[dofs_mask], rtol=rtol, atol=atol)


def assert_body_states_equal_masked(
    model: ModelKamino,
    body_q: wp.array[wp.transformf],
    body_q_ref: wp.array[wp.transformf],
    body_u: wp.array[wp.spatial_vectorf],
    body_u_ref: wp.array[wp.spatial_vectorf],
    world_mask: wp.array[wp.bool],
):
    """Check that body poses/velocities match provided references for worlds where the mask is False."""
    bodies_offset = model.info.bodies_offset.numpy()
    world_mask_np = world_mask.numpy()
    body_q_np = body_q.numpy()
    body_q_ref_np = body_q_ref.numpy()
    body_u_np = body_u.numpy()
    body_u_ref_np = body_u_ref.numpy()

    for wid in range(model.size.num_worlds):
        if world_mask_np[wid]:
            continue
        np.testing.assert_allclose(
            body_q_np[bodies_offset[wid] : bodies_offset[wid + 1]],
            body_q_ref_np[bodies_offset[wid] : bodies_offset[wid + 1]],
            rtol=rtol,
            atol=atol,
            err_msg=f"\nWorld wid={wid}: `body_q` mismatch:\n",
        )
        np.testing.assert_allclose(
            body_u_np[bodies_offset[wid] : bodies_offset[wid + 1]],
            body_u_ref_np[bodies_offset[wid] : bodies_offset[wid + 1]],
            rtol=rtol,
            atol=atol,
            err_msg=f"\nWorld wid={wid}: `body_u` mismatch:\n",
        )


def assert_rigid_poses_close(
    pose: np.ndarray,
    pose_ref: np.ndarray,
):
    """Check that two rigid poses (position + quaternion) match up to unit quaternion sign."""
    quat = pose[3:]
    quat_ref = pose_ref[3:]
    if np.linalg.norm(quat - quat_ref) > np.linalg.norm(quat + quat_ref):
        pose = np.concatenate((pose[:3], -quat))
    np.testing.assert_allclose(pose, pose_ref, rtol=rtol, atol=atol)


def validate_base_pose_reset(
    model: ModelKamino,
    base_q: wp.array[wp.transformf] | None,
    data_prev: DataKamino,
    data_new: DataKamino,
    world_mask: wp.array[wp.bool],
):
    """
    Check that the result of set_floating_base() has the expected base pose
    Args:
        model: Kamino model
        base_q: Base pose that was passed to set_floating_base().
        data_prev: Model state before set_floating_base(); only body states are needed.
        data_new: Model state after set_floating_base(); we assume joint states be up-to-date with bodies.
        world_mask: Per-world mask that was passed to set_floating_base().
    """
    # Move useful to data to numpy
    base_body_id_np = model.info.base_body_index.numpy()
    base_joint_id_np = model.info.base_joint_index.numpy()
    base_q_np = base_q.numpy() if base_q is not None else None
    body_q_prev_np = data_prev.bodies.q_i.numpy()
    body_q_new_np = data_new.bodies.q_i.numpy()
    joint_q_new_np = data_new.joints.q_j.numpy()
    coords_offset_np = model.joints.coords_offset.numpy()
    world_mask_np = world_mask.numpy()

    # Validate base pose in each world
    for wid in range(model.size.num_worlds):
        if not world_mask_np[wid]:
            continue
        if base_q is None:  # Check that base body pose is preserved if base_q is not provided
            bid = base_body_id_np[wid]
            assert_rigid_poses_close(body_q_new_np[bid], body_q_prev_np[bid])
        else:
            jid = base_joint_id_np[wid]
            if jid >= 0:  # If a base joint was set, check that base_q matches joint_q for that joint
                assert coords_offset_np[jid + 1] - coords_offset_np[jid] == 7
                # The test example has a free base joint, so we can read the transformation that was applied
                # in joint frame directly in joint_q. This makes testing easier but remains general since
                # set_floating_base() ignores the base joint type.
                assert_rigid_poses_close(
                    joint_q_new_np[coords_offset_np[jid] : coords_offset_np[jid + 1]], base_q_np[wid]
                )
            else:  # If no base joint was set, check that base_q matches body_q for the base body
                bid = base_body_id_np[wid]
                assert_rigid_poses_close(body_q_new_np[bid], base_q_np[wid])


def validate_base_velocity_reset(
    model: ModelKamino,
    base_u: wp.array[wp.spatial_vectorf] | None,
    data_prev: DataKamino,
    data_new: DataKamino,
    world_mask: wp.array[wp.bool],
):
    """
    Check that the result of set_floating_base() has the expected base velocity (relative_base_u = False case)
    Args:
        model: Kamino model
        base_u: Base velocity that was passed to set_floating_base().
        data_prev: Model state before set_floating_base(); only body states are needed.
        data_new: Model state after set_floating_base(); we assume joint states be up-to-date with bodies.
        world_mask: Per-world mask that was passed to set_floating_base().
    """
    # Move useful to data to numpy
    base_body_id_np = model.info.base_body_index.numpy()
    base_joint_id_np = model.info.base_joint_index.numpy()
    base_u_np = base_u.numpy() if base_u is not None else None
    body_q_prev_np = data_prev.bodies.q_i.numpy()
    body_u_prev_np = data_prev.bodies.u_i.numpy()
    body_q_new_np = data_new.bodies.q_i.numpy()
    body_u_new_np = data_new.bodies.u_i.numpy()
    joint_u_new_np = data_new.joints.dq_j.numpy()
    dofs_offset_np = model.joints.dofs_offset.numpy()
    world_mask_np = world_mask.numpy()

    # Helper to apply a rotation to both the linear and angular part of a twist
    def rotate_twist(R, u):
        return (R @ u.reshape(2, 3).T).T.ravel()

    # Validate base velocity in each world
    for wid in range(model.size.num_worlds):
        if not world_mask_np[wid]:
            continue
        if base_u is None:  # Check that base body velocity is preserved up to base rotation
            bid = base_body_id_np[wid]
            base_body_q_prev = wp.quatf(body_q_prev_np[bid, 3:])
            base_body_q_new = wp.quatf(body_q_new_np[bid, 3:])
            R_rel_wp = wp.quat_to_matrix(base_body_q_new * wp.quat_inverse(base_body_q_prev))
            R_rel = np.array([*R_rel_wp])
            body_u_prev_rotated = rotate_twist(R_rel, body_u_prev_np[bid])
            np.testing.assert_allclose(body_u_new_np[bid], body_u_prev_rotated, rtol=rtol, atol=atol)
            continue
        else:
            jid = base_joint_id_np[wid]
            if jid >= 0:  # If a base joint was set, check that target base_u matches joint_u for that joint
                assert dofs_offset_np[jid + 1] - dofs_offset_np[jid] == 6
                # The test example has a free base joint, so we can read the velocity that was applied
                # in joint frame directly in joint_u. This makes testing easier but remains general since
                # set_floating_base() ignores the base joint type.
                np.testing.assert_allclose(
                    joint_u_new_np[dofs_offset_np[jid] : dofs_offset_np[jid + 1]],
                    base_u_np[wid],
                    rtol=rtol,
                    atol=atol,
                )
            else:  # If no base joint was set, check that target base_u matches body_u for the base body
                bid = base_body_id_np[wid]
                np.testing.assert_allclose(body_u_new_np[bid], base_u_np[wid], rtol=rtol, atol=atol)


def run_set_floating_base_check(
    model: ModelKamino,
    base_q: wp.array[wp.transformf] | None,
    base_u: wp.array[wp.spatial_vectorf] | None,
    world_mask: wp.array[wp.bool],
    data_prev: DataKamino,
):
    """
    Call set_floating_base() with provided arguments, and validate that it behaves as expected.
    Args:
        model: Kamino model
        base_q: new base_q to set with set_floating_base()
        base_u: new base_u to set with set_floating_base()
        world_mask: mask for set_floating_base()
        data_prev: Kamino data, representing the state of the model before the floating base reset
    """
    try:
        # Create a new model data, with body states set as per previous data
        data = model.data(unilateral_cts=False, joint_wrenches=False, device=model.device)
        wp.copy(data.bodies.q_i, data_prev.bodies.q_i)
        wp.copy(data.bodies.u_i, data_prev.bodies.u_i)

        # Call set_floating_base()
        set_floating_base(
            model=model,
            base_q=base_q,
            base_u=base_u,
            body_q=data.bodies.q_i,
            body_u=data.bodies.u_i,
            world_mask=world_mask,
        )

        # Check that new state is consistent and preserves relative poses and velocities
        # Equivalently, check for constraint satisfaction and binary joint coords/velocities preservation
        compute_joints_data(
            model=model, data=data, q_j_p=data_prev.joints.q_j, correction=JointCorrectionMode.CONTINUOUS
        )
        np.testing.assert_allclose(data.joints.r_j.numpy(), 0.0, rtol=0, atol=atol)
        np.testing.assert_allclose(data.joints.dr_j.numpy(), 0.0, rtol=0, atol=atol)
        assert_binary_joint_states_equal(
            model, data.joints.q_j, data_prev.joints.q_j, data.joints.dq_j, data_prev.joints.dq_j
        )

        # Check that worlds with the mask set to False are not modified
        assert_body_states_equal_masked(
            model, data.bodies.q_i, data_prev.bodies.q_i, data.bodies.u_i, data_prev.bodies.u_i, world_mask
        )

        # Check that the base pose and velocity were correctly set or preserved
        validate_base_pose_reset(model, base_q, data_prev, data, world_mask)
        validate_base_velocity_reset(model, base_u, data_prev, data, world_mask)

    except AssertionError as e:
        base_q_str = "base_q provided" if base_q is not None else "base_q not provided"
        base_u_str = "base_u provided" if base_u is not None else "base_u not provided"
        raise AssertionError(f"set_floating_base() check failed for {base_q_str}, {base_u_str}\n{e}") from e


def setup_test_fourbar_model(
    base_joint: bool, num_worlds: int, rng: np.random.Generator, device: wp.DeviceLike
) -> ModelKamino:
    """Helper setting up a floating-base actuated four-bar model, with a base joint or a base body."""
    build_fn = functools.partial(
        build_boxes_fourbar, actuator_ids=[1], floatingbase=base_joint, fixedbase=False, ground=False
    )
    builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_fn, limits=False)
    if base_joint:
        # Set non-trivial r_B, X_B, r_F, X_F into base joint for testing (while preserving initial pose)
        random_frames = np.resize(rng.uniform(-1.0, 1.0, 7 * num_worlds), (num_worlds, 7))
        random_frames[:, 3:] /= np.linalg.norm(random_frames[:, 3:], axis=1)[:, None]  # Normalize quaternions
        for wid in range(num_worlds):
            joint = builder.joints[wid][0]
            assert joint.bid_B == -1
            body_F = builder.bodies[wid][joint.bid_F]
            c_F = wp.transform_get_translation(body_F.q_i_0)
            R_F = wp.quat_to_matrix(wp.transform_get_rotation(body_F.q_i_0))
            joint.F_r_Fj = wp.vec3f(random_frames[wid, :3])
            joint.X_Fj = wp.quat_to_matrix(wp.quatf(random_frames[wid, 3:]))
            joint.B_r_Bj = c_F + R_F * joint.F_r_Fj  # Compute r_B, X_B given r_F, X_F to preserve a valid pose
            joint.X_Bj = R_F * joint.X_Fj
    model = builder.finalize(device=device)
    return model


def sample_base_state_wp(model: ModelKamino, rng: np.random.Generator):
    base_q_np, base_u_np = sample_base_state(model.size.num_worlds, rng)
    base_q = wp.from_numpy(base_q_np[0], dtype=wp.transformf, device=model.device)
    base_u = wp.from_numpy(base_u_np[0], dtype=wp.spatial_vectorf, device=model.device)
    return base_q, base_u


def sample_actuator_state_wp(model: ModelKamino, rng: np.random.Generator):
    actuator_q_np = sample_actuator_coords(model, rng)[0]
    actuator_u_np = sample_actuator_velocities(model, rng)[0]
    actuator_q = wp.from_numpy(actuator_q_np, dtype=wp.float32, device=model.device)
    actuator_u = wp.from_numpy(actuator_u_np, dtype=wp.float32, device=model.device)
    return actuator_q, actuator_u


def set_model_to_random_pose(
    test_case: unittest.TestCase,
    model: ModelKamino,
    rng: np.random.Generator,
):
    """
    Helper sampling a random valid pose & velocity for a model, setting the model
    into this pose with FK, and computing joint data as a post-processing.
    """
    # Sample random pose
    base_q, base_u = sample_base_state_wp(model, rng)
    actuator_q, actuator_u = sample_actuator_state_wp(model, rng)

    # Set the model into generated non-trivial pose using FK
    fk_solver = ForwardKinematicsSolver(model=model)
    data = model.data(unilateral_cts=False, joint_wrenches=False, device=model.device)
    fk_solver.run_fk_solve(
        actuators_q=actuator_q,
        actuators_u=actuator_u,
        base_q=base_q,
        base_u=base_u,
        bodies_q=data.bodies.q_i,
        bodies_u=data.bodies.u_i,
    )
    test_case.assertTrue(fk_solver.newton_success.numpy().sum() == model.size.num_worlds)

    # Evaluate joint state and check constraint residuals
    compute_joints_data(model=model, data=data, q_j_p=model.joints.q_j_0, correction=JointCorrectionMode.CONTINUOUS)
    np.testing.assert_allclose(data.joints.r_j.numpy(), 0.0, rtol=0, atol=atol)
    np.testing.assert_allclose(data.joints.dr_j.numpy(), 0.0, rtol=0, atol=atol)

    return data


###
# Tests
###


class TestSetFloatingBase(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output
        self.progress = test_context.verbose  # Set to True to show progress bars during long tests
        self.seed = 42

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_set_floating_base_with_base_joint(self):
        """
        Validate that set_floating_base() sets the base pose/velocity as expected,
        while preserving relative poses and velocities, for a model with a base joint.
        """
        # Initialize rng
        rng = np.random.default_rng(self.seed)

        # Set up an actuated four-bar model with a floating base, using a base joint
        num_worlds = 3
        model = setup_test_fourbar_model(base_joint=True, num_worlds=num_worlds, rng=rng, device=self.default_device)

        # Set model into non-trivial pose
        data = set_model_to_random_pose(self, model, rng)

        # Sample non-trivial world mask and base state
        world_mask = wp.array(sample_world_mask(num_worlds, rng)[0], dtype=wp.bool, device=self.default_device)
        base_q, base_u = sample_base_state_wp(model, rng)

        # Check validity of set_floating_base for all options combinations
        run_set_floating_base_check(model, base_q, base_u, world_mask, data)
        run_set_floating_base_check(model, base_q, None, world_mask, data)
        run_set_floating_base_check(model, None, base_u, world_mask, data)
        run_set_floating_base_check(model, None, None, world_mask, data)

    def test_02_set_floating_base_with_base_body(self):
        """
        Validate that set_floating_base() sets the base pose/velocity as expected,
        while preserving relative poses and velocities, for a model with only a base body.
        """
        # Initialize rng
        rng = np.random.default_rng(self.seed)

        # Set up an actuated four-bar model with a floating base, using a base body
        num_worlds = 3
        model = setup_test_fourbar_model(base_joint=False, num_worlds=num_worlds, rng=rng, device=self.default_device)

        # Set model into non-trivial pose
        data = set_model_to_random_pose(self, model, rng)

        # Sample non-trivial world mask and base state
        world_mask = wp.array(sample_world_mask(num_worlds, rng)[0], dtype=wp.bool, device=self.default_device)
        base_q, base_u = sample_base_state_wp(model, rng)

        # Check validity of set_floating_base for all options combinations
        run_set_floating_base_check(model, base_q, base_u, world_mask, data)
        run_set_floating_base_check(model, base_q, None, world_mask, data)
        run_set_floating_base_check(model, None, base_u, world_mask, data)
        run_set_floating_base_check(model, None, None, world_mask, data)

    def test_03_relative_base_u_with_base_joint(self):
        """
        Validate the relative_base_u flag of set_floating_base(), for a model with a base joint.
        """
        # Initialize rng
        rng = np.random.default_rng(self.seed)

        # Set up an actuated four-bar model with a floating base, using a base joint
        num_worlds = 3
        model = setup_test_fourbar_model(base_joint=True, num_worlds=num_worlds, rng=rng, device=self.default_device)

        # Set model into non-trivial pose
        data = set_model_to_random_pose(self, model, rng)

        # Sample non-trivial world mask and base state
        world_mask = wp.array(sample_world_mask(num_worlds, rng)[0], dtype=wp.bool, device=self.default_device)
        base_q, base_u = sample_base_state_wp(model, rng)

        # Check that a call to set_floating_base() with relative_base_u enabled is equivalent
        # to a first call changing only base_u, followed by a second call changing only base_q
        body_q = wp.clone(data.bodies.q_i, device=self.default_device)
        body_u = wp.clone(data.bodies.u_i, device=self.default_device)
        set_floating_base(
            model=model,
            base_q=base_q,
            base_u=base_u,
            body_q=body_q,
            body_u=body_u,
            world_mask=world_mask,
            relative_base_u=True,
        )
        body_q_check = wp.clone(data.bodies.q_i, device=self.default_device)
        body_u_check = wp.clone(data.bodies.u_i, device=self.default_device)
        set_floating_base(
            model=model,
            base_q=None,
            base_u=base_u,
            body_q=body_q_check,
            body_u=body_u_check,
            world_mask=world_mask,
            relative_base_u=False,
        )
        set_floating_base(
            model=model,
            base_q=base_q,
            base_u=None,
            body_q=body_q_check,
            body_u=body_u_check,
            world_mask=world_mask,
            relative_base_u=False,
        )
        np.testing.assert_allclose(body_q.numpy(), body_q_check.numpy(), rtol=rtol, atol=atol)
        np.testing.assert_allclose(body_u.numpy(), body_u_check.numpy(), rtol=rtol, atol=atol)

    def test_04_relative_base_u_with_base_body(self):
        """
        Validate the relative_base_u flag of set_floating_base(), for a model with a base body.
        """
        # Initialize rng
        rng = np.random.default_rng(self.seed)

        # Set up an actuated four-bar model with a floating base, using a base body
        num_worlds = 3
        model = setup_test_fourbar_model(base_joint=False, num_worlds=num_worlds, rng=rng, device=self.default_device)

        # Set model into non-trivial pose
        data = set_model_to_random_pose(self, model, rng)

        # Sample non-trivial world mask and base state
        world_mask = wp.array(sample_world_mask(num_worlds, rng)[0], dtype=wp.bool, device=self.default_device)
        base_q, base_u = sample_base_state_wp(model, rng)

        # Check that a call to set_floating_base() with relative_base_u enabled is equivalent
        # to a first call changing only base_u, followed by a second call changing only base_q
        body_q = wp.clone(data.bodies.q_i, device=self.default_device)
        body_u = wp.clone(data.bodies.u_i, device=self.default_device)
        set_floating_base(
            model=model,
            base_q=base_q,
            base_u=base_u,
            body_q=body_q,
            body_u=body_u,
            world_mask=world_mask,
            relative_base_u=True,
        )
        body_q_check = wp.clone(data.bodies.q_i, device=self.default_device)
        body_u_check = wp.clone(data.bodies.u_i, device=self.default_device)
        set_floating_base(
            model=model,
            base_q=None,
            base_u=base_u,
            body_q=body_q_check,
            body_u=body_u_check,
            world_mask=world_mask,
            relative_base_u=False,
        )
        set_floating_base(
            model=model,
            base_q=base_q,
            base_u=None,
            body_q=body_q_check,
            body_u=body_u_check,
            world_mask=world_mask,
            relative_base_u=False,
        )
        np.testing.assert_allclose(body_q.numpy(), body_q_check.numpy(), rtol=rtol, atol=atol)
        np.testing.assert_allclose(body_u.numpy(), body_u_check.numpy(), rtol=rtol, atol=atol)


class TestJointBodyStateConversions(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output
        self.progress = test_context.verbose  # Set to True to show progress bars during long tests
        self.seed = 42

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_reset_joint_states_from_body_state(self):
        """
        Validate that reset_joints_state_from_bodies_state() against compute_joints_data()
        on a model with all joint types.
        """
        # Initialize rng
        rng = np.random.default_rng(self.seed)

        # Setup a model with all joint types
        builder = build_all_joints_test_model(binary_joints=True, unary_joints=False, actuated=True, floating_base=True)
        model = builder.finalize(device=self.default_device)

        # Set the model into a non-trivial pose
        data = set_model_to_random_pose(self, model, rng)

        # Compute joint states from bodies state
        state = model.state()
        wp.copy(state.q_i, data.bodies.q_i)
        wp.copy(state.u_i, data.bodies.u_i)
        all_worlds_mask = wp.ones(shape=model.size.num_worlds, dtype=wp.bool, device=model.device)
        reset_joints_state_from_bodies_state(model, state, world_mask=all_worlds_mask)

        # Compare against joint state in joint data
        # Note: both functions are correcting coords w.r.t. initial coords, so values are directly comparable
        np.testing.assert_allclose(state.q_j.numpy(), data.joints.q_j.numpy(), rtol=rtol, atol=atol)
        np.testing.assert_allclose(state.dq_j.numpy(), data.joints.dq_j.numpy(), rtol=rtol, atol=atol)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
