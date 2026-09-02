# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the :class:`SolverKaminoImpl` class"""

import time
import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.control import ControlKamino
from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.joints import JointActuationType
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.shapes import GeoType
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.dynamics import DualProblem
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.geometry.detector import CollisionDetector
from newton._src.solvers.kamino._src.integrators import IntegratorEuler, IntegratorMoreauJean
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from newton._src.solvers.kamino._src.kinematics.joints import JointCorrectionMode, compute_joints_data
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.models.builders import testing
from newton._src.solvers.kamino._src.models.builders.basics import build_boxes_fourbar
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.solvers import PADMMSolver
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.examples import print_progress_bar
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.sampling import sample_world_mask
from newton.tests.utils import basics
from newton.tests.utils.testing import build_binary_revolute_joint_test

###
# Module configs
###

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

BASE_HEIGHT = 0.5
SIM_DT = 1.0 / 60.0
UNIT_INERTIA = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)

###
# Kernels
###


@wp.kernel
def _test_control_callback(
    model_dt: wp.array[wp.float32],
    data_time: wp.array[wp.float32],
    control_tau_j: wp.array[wp.float32],
):
    """
    An example control callback kernel.
    """
    # Retrieve the world index from the thread ID
    wid = wp.tid()

    # Get the fixed time-step and current time
    dt = model_dt[wid]
    t = data_time[wid]

    # Define the time window for the active external force profile
    t_start = wp.float32(0.0)
    t_end = 10.0 * dt

    # Compute the first actuated joint index for the current world
    aid = wid * 2 + 0

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        control_tau_j[aid] = 0.1
    else:
        control_tau_j[aid] = 0.0


###
# Launchers
###


def test_prestep_callback(
    solver: SolverKaminoImpl,
    state_in: StateKamino,
    state_out: StateKamino,
    control: ControlKamino,
    contacts: ContactsKamino,
):
    """
    A control callback function
    """
    wp.launch(
        _test_control_callback,
        dim=solver._model.size.num_worlds,
        inputs=[
            solver._model.time.dt,
            solver._data.time.time,
            control.tau_j,
        ],
        device=solver.device,
    )


###
# Utils
###

rtol = 1e-7
atol = 1e-6


def assert_solver_config(testcase: unittest.TestCase, config: SolverKaminoImpl.Config):
    testcase.assertIsInstance(config, SolverKaminoImpl.Config)
    testcase.assertIsInstance(config.constraints, kamino_config.ConstraintStabilizationConfig)
    testcase.assertIsInstance(config.dynamics, kamino_config.ConstrainedDynamicsConfig)
    testcase.assertIsInstance(config.padmm, kamino_config.PADMMSolverConfig)
    testcase.assertIsInstance(config.rotation_correction, str)


def assert_solver_components(testcase: unittest.TestCase, solver: SolverKaminoImpl):
    testcase.assertIsInstance(solver, SolverKaminoImpl)
    testcase.assertIsInstance(solver.config, SolverKaminoImpl.Config)
    testcase.assertIsInstance(solver._model, ModelKamino)
    testcase.assertIsInstance(solver._data, DataKamino)
    testcase.assertIsInstance(solver._limits, LimitsKamino)
    if solver._problem_fd.sparse:
        testcase.assertIsInstance(solver._jacobians, SparseSystemJacobians)
    else:
        testcase.assertIsInstance(solver._jacobians, DenseSystemJacobians)
    testcase.assertIsInstance(solver._problem_fd, DualProblem)
    testcase.assertIsInstance(solver._solver_fd, PADMMSolver)


def assert_states_equal(testcase: unittest.TestCase, state_0: StateKamino, state_1: StateKamino):
    testcase.assertIsInstance(state_0, StateKamino)
    testcase.assertIsInstance(state_1, StateKamino)
    np.testing.assert_array_equal(state_0.q_i.numpy(), state_1.q_i.numpy())
    np.testing.assert_array_equal(state_0.u_i.numpy(), state_1.u_i.numpy())
    np.testing.assert_array_equal(state_0.w_i.numpy(), state_1.w_i.numpy())
    np.testing.assert_array_equal(state_0.q_j.numpy(), state_1.q_j.numpy())
    np.testing.assert_array_equal(state_0.q_j_p.numpy(), state_1.q_j_p.numpy())
    np.testing.assert_array_equal(state_0.dq_j.numpy(), state_1.dq_j.numpy())
    np.testing.assert_array_equal(state_0.lambda_kin_j.numpy(), state_1.lambda_kin_j.numpy())
    np.testing.assert_array_equal(state_0.lambda_dyn_j.numpy(), state_1.lambda_dyn_j.numpy())
    np.testing.assert_array_equal(state_0.lambda_f_j.numpy(), state_1.lambda_f_j.numpy())
    np.testing.assert_array_equal(state_0.lambda_tau_j.numpy(), state_1.lambda_tau_j.numpy())


def assert_states_close(testcase: unittest.TestCase, state_0: StateKamino, state_1: StateKamino):
    testcase.assertIsInstance(state_0, StateKamino)
    testcase.assertIsInstance(state_1, StateKamino)
    np.testing.assert_allclose(state_0.q_i.numpy(), state_1.q_i.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.u_i.numpy(), state_1.u_i.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.w_i.numpy(), state_1.w_i.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.q_j.numpy(), state_1.q_j.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.q_j_p.numpy(), state_1.q_j_p.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.dq_j.numpy(), state_1.dq_j.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.lambda_kin_j.numpy(), state_1.lambda_kin_j.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.lambda_dyn_j.numpy(), state_1.lambda_dyn_j.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.lambda_f_j.numpy(), state_1.lambda_f_j.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.lambda_tau_j.numpy(), state_1.lambda_tau_j.numpy(), rtol=rtol, atol=atol)


def assert_states_close_masked(
    model: ModelKamino,
    state: StateKamino,
    state_true: StateKamino | None,
    state_false: StateKamino | None,
    world_mask: wp.array[wp.bool],
    positions: bool = True,
    velocities: bool = True,
    forces: bool = True,
    match_q_j_p_with_q_j: bool = False,
):
    """
    Check that state attributes match one of two reference states, based on the world mask.

    Args:
        model: Kamino model.
        state: Kamino state to compare to reference states.
        state_true: Kamino reference state for worlds in which the mask is True.
        state_false: Kamino reference state for worlds in which the mask is False.
        world_mask: Per-world boolean mask.
        positions: Whether to compare position attributes, i.e. q_i, q_j, q_j_p (skipped if False).
        velocities: Whether to compare velocity attributes, i.e. u_i, dq_j (skipped if False).
        forces: Whether to compare force attributes, i.e. w_i, w_i_e, lambda_kin_j, lambda_dyn_j, lambda_f_j, lambda_tau_j (skipped if False).
        match_q_j_p_with_q_j: Whether to compare q_j_p against q_j in the reference (instead of q_j_p).
    """
    bodies_offset = model.info.bodies_offset.numpy()
    coords_offset = np.array([*model.info.joint_coords_offset.numpy(), model.size.sum_of_num_joint_coords])
    dofs_offset = np.array([*model.info.joint_dofs_offset.numpy(), model.size.sum_of_num_joint_dofs])
    kin_cts_offset = np.array(
        [*model.info.joint_kinematic_cts_offset.numpy(), model.size.sum_of_num_kinematic_joint_cts]
    )
    dyn_cts_offset = np.array([*model.info.joint_dynamic_cts_offset.numpy(), model.size.sum_of_num_dynamic_joint_cts])
    friction_cts_offset = np.array(
        [*model.info.joint_friction_cts_offset.numpy(), model.size.sum_of_num_friction_joint_cts]
    )
    effort_cts_offset = np.array([*model.info.joint_effort_cts_offset.numpy(), model.size.sum_of_num_effort_joint_cts])
    world_mask_np = world_mask.numpy()

    # List state attributes to compare
    body_attributes = []
    coord_attributes = []
    dof_attributes = []
    cts_attributes = []
    if positions:
        body_attributes.append("q_i")
        coord_attributes.append("q_j")
        if not match_q_j_p_with_q_j:
            coord_attributes.append("q_j_p")
    if velocities:
        body_attributes.append("u_i")
        dof_attributes.append("dq_j")
    if forces:
        body_attributes.extend(["w_i", "w_i_e"])
        cts_attributes.extend(
            [
                ("lambda_kin_j", kin_cts_offset),
                ("lambda_dyn_j", dyn_cts_offset),
                ("lambda_f_j", friction_cts_offset),
                ("lambda_tau_j", effort_cts_offset),
            ]
        )

    for wid in range(model.size.num_worlds):
        # Select reference state based on world mask
        state_ref = state_true if world_mask_np[wid] else state_false
        if state_ref is None:
            continue

        # Check state attributes for the current world
        for attr in body_attributes:
            np.testing.assert_allclose(
                getattr(state, attr).numpy()[bodies_offset[wid] : bodies_offset[wid + 1]],
                getattr(state_ref, attr).numpy()[bodies_offset[wid] : bodies_offset[wid + 1]],
                rtol=rtol,
                atol=atol,
                err_msg=f"\nWorld wid={wid}: attribute `{attr}` mismatch:\n",
            )
        for attr in coord_attributes:
            np.testing.assert_allclose(
                getattr(state, attr).numpy()[coords_offset[wid] : coords_offset[wid + 1]],
                getattr(state_ref, attr).numpy()[coords_offset[wid] : coords_offset[wid + 1]],
                rtol=rtol,
                atol=atol,
                err_msg=f"\nWorld wid={wid}: attribute `{attr}` mismatch:\n",
            )
        for attr in dof_attributes:
            np.testing.assert_allclose(
                getattr(state, attr).numpy()[dofs_offset[wid] : dofs_offset[wid + 1]],
                getattr(state_ref, attr).numpy()[dofs_offset[wid] : dofs_offset[wid + 1]],
                rtol=rtol,
                atol=atol,
                err_msg=f"\nWorld wid={wid}: attribute `{attr}` mismatch:\n",
            )
        for attr, offset in cts_attributes:
            np.testing.assert_allclose(
                getattr(state, attr).numpy()[offset[wid] : offset[wid + 1]],
                getattr(state_ref, attr).numpy()[offset[wid] : offset[wid + 1]],
                rtol=rtol,
                atol=atol,
                err_msg=f"\nWorld wid={wid}: attribute `{attr}` mismatch:\n",
            )
        if positions and match_q_j_p_with_q_j:
            np.testing.assert_allclose(
                state.q_j_p.numpy()[coords_offset[wid] : coords_offset[wid + 1]],
                state_ref.q_j.numpy()[coords_offset[wid] : coords_offset[wid + 1]],
                rtol=rtol,
                atol=atol,
                err_msg=f"\nWorld wid={wid}: attribute `q_j_p` mismatch:\n",
            )


def check_body_and_joint_state_consistency(
    model: ModelKamino,
    body_q: wp.array[wp.transformf],
    body_u: wp.array[wp.spatial_vectorf],
    joint_q: wp.array[wp.float32],
    joint_u: wp.array[wp.float32],
):
    """Check that provided joint/body positions and velocities are consistent for a given model."""
    # Check dimensions
    np.testing.assert_equal(body_q.shape[0], model.size.sum_of_num_bodies)
    np.testing.assert_equal(body_u.shape[0], model.size.sum_of_num_bodies)
    np.testing.assert_equal(joint_q.shape[0], model.size.sum_of_num_joint_coords)
    np.testing.assert_equal(joint_u.shape[0], model.size.sum_of_num_joint_dofs)

    # Create a model data, and evaluate joint data given provided body states
    data = model.data(device=model.device)
    wp.copy(data.bodies.q_i, body_q)
    wp.copy(data.bodies.u_i, body_u)
    compute_joints_data(model=model, data=data, q_j_p=joint_q, correction=JointCorrectionMode.CONTINUOUS)

    # Check that recovered joint coordinates/velocities match provided ones
    joint_q_np = joint_q.numpy()
    joint_u_np = joint_u.numpy()
    joint_q_check_np = data.joints.q_j.numpy()
    joint_u_check_np = data.joints.dq_j.numpy()
    coords_offset = np.array([*model.info.joint_coords_offset.numpy(), model.size.sum_of_num_joint_coords])
    dofs_offset = np.array([*model.info.joint_dofs_offset.numpy(), model.size.sum_of_num_joint_dofs])
    for wid in range(model.size.num_worlds):
        np.testing.assert_allclose(
            joint_q_np[coords_offset[wid] : coords_offset[wid + 1]],
            joint_q_check_np[coords_offset[wid] : coords_offset[wid + 1]],
            rtol=rtol,
            atol=atol,
            err_msg=f"\nWorld wid={wid}: joint_q mismatch:\n",
        )
        np.testing.assert_allclose(
            joint_u_np[dofs_offset[wid] : dofs_offset[wid + 1]],
            joint_u_check_np[dofs_offset[wid] : dofs_offset[wid + 1]],
            rtol=rtol,
            atol=atol,
            err_msg=f"\nWorld wid={wid}: joint u_mismatch:\n",
        )

    # Check that position- and velocity-level joint constraint residuals are below tolerance
    np.testing.assert_allclose(data.joints.r_j.numpy(), 0.0, rtol=0, atol=atol)
    np.testing.assert_allclose(data.joints.dr_j.numpy(), 0.0, rtol=0, atol=atol)


def check_post_reset_state_consistency(model: ModelKamino, state: StateKamino):
    """
    Check that provided state is a valid post-reset state (consistent states for bodies
    and joints, current and previous joint states matching)
    """
    check_body_and_joint_state_consistency(
        model=model,
        body_q=state.q_i,
        body_u=state.u_i,
        joint_q=state.q_j,
        joint_u=state.dq_j,
    )
    np.testing.assert_allclose(state.q_j.numpy(), state.q_j_p.numpy(), rtol=rtol, atol=atol)


def step_solver(
    num_steps: int,
    solver: SolverKaminoImpl,
    state_p: StateKamino,
    state_n: StateKamino,
    control: ControlKamino,
    contacts: ContactsKamino | None = None,
    detector: CollisionDetector | None = None,
    dt: float = 0.001,
    show_progress: bool = False,
):
    start_time = time.time()
    for step in range(num_steps):
        solver.step(state_in=state_p, state_out=state_n, control=control, contacts=contacts, detector=detector, dt=dt)
        state_p.copy_from(state_n)
        if show_progress:
            wp.synchronize()
            print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")


###
# Tests
###


class TestSolverKaminoConfig(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True to enable verbose output

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

    def test_00_make_default(self):
        config = SolverKaminoImpl.Config()
        assert_solver_config(self, config)
        self.assertEqual(config.rotation_correction, "twopi")
        self.assertEqual(config.dynamics.linear_solver_type, "LLTB")
        self.assertEqual(config.padmm.warmstart_mode, "containers")
        self.assertEqual(config.padmm.warmstart_scale, 0.9)

    def test_01_make_explicit(self):
        """Construct an explicit solver configuration with requested overrides."""
        config = SolverKaminoImpl.Config(
            dynamics=kamino_config.ConstrainedDynamicsConfig(linear_solver_type="CR"),
            padmm=kamino_config.PADMMSolverConfig(warmstart_mode="internal", warmstart_scale=0.5),
            rotation_correction="continuous",
        )
        assert_solver_config(self, config)
        self.assertEqual(config.rotation_correction, "continuous")
        self.assertEqual(config.dynamics.linear_solver_type, "CR")
        self.assertEqual(config.padmm.warmstart_mode, "internal")
        self.assertEqual(config.padmm.warmstart_scale, 0.5)

    def test_02_validate_warmstart_scale(self):
        """Reject PADMM warmstart scales outside the supported range."""
        for scale in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                kamino_config.PADMMSolverConfig(warmstart_scale=scale)

    def test_02_reject_dense_adaptive_padmm_penalty(self):
        """Reject adaptive PADMM penalties with dense dynamics."""
        with self.assertRaisesRegex(ValueError, "Adaptive PADMM penalty updates require `sparse_dynamics=True`"):
            SolverKamino.Config(
                sparse_dynamics=False,
                padmm=kamino_config.PADMMSolverConfig(penalty_update_method="balanced"),
            )


class TestCollisionCapacityInitialization(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _make_three_world_model(self) -> newton.Model:
        source_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(source_builder)
        basics.build_sphere_on_plane(builder=source_builder, z_offset=0.5)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        builder.replicate(source_builder, world_count=3)
        return builder.finalize(device=self.default_device, skip_validation_joints=True)

    def test_capacity_allocation_for_pipeline_allocated_after_kamino(self):
        """Verify capacity allocation for a pipeline allocated after Kamino."""
        model = self._make_three_world_model()

        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=False))

        pipeline = newton.CollisionPipeline(model)

        # Kamino's capacity should be equal to the pipeline's capacity.
        self.assertEqual(solver._contacts_kamino.model_max_contacts_host, pipeline.rigid_contact_max)

    def test_capacity_allocation_for_pipeline_allocated_before_kamino(self):
        """Verify capacity allocation for a pipeline allocated before Kamino."""
        model = self._make_three_world_model()

        pipeline = newton.CollisionPipeline(model)
        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=False))

        # Kamino's capacity should be at least as large as the pipeline's capacity. Due to rounding up to the nearest multiple of the world count.
        self.assertGreaterEqual(solver._contacts_kamino.model_max_contacts_host, pipeline.rigid_contact_max)

    def test_internal_collision_capacity_updates_model(self):
        """Verify internal collision capacity updates the Newton model."""
        model = self._make_three_world_model()

        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=True))
        pipeline = newton.CollisionPipeline(model)

        self.assertEqual(model.rigid_contact_max, solver._contacts_kamino.model_max_contacts_host)
        self.assertEqual(pipeline.rigid_contact_max, solver._contacts_kamino.model_max_contacts_host)

    def test_moreau_uses_internal_detection(self):
        """Verify Moreau-Jean is selected with internal collision detection."""
        model = self._make_three_world_model()
        solver = SolverKamino(
            model,
            config=SolverKamino.Config(integrator="moreau", use_collision_detector=True),
        )

        self.assertIsInstance(solver._solver_kamino._integrator, IntegratorMoreauJean)

    def test_moreau_detects_midpoint_contact(self):
        """Verify Moreau-Jean detects contacts created at the midpoint."""
        # Start the sphere surface 0.3 m above the zero-gap ground plane.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        basics.build_sphere_on_plane(builder=builder, z_offset=0.3)
        model = builder.finalize(device=self.default_device, skip_validation_joints=True)

        config = SolverKamino.Config(integrator="moreau", use_collision_detector=True)
        solver = SolverKamino(model, config=config)

        detector = solver._collision_detector_kamino
        contacts = solver._contacts_kamino
        assert detector is not None
        assert contacts is not None
        assert contacts.model_active_contacts is not None

        # Establish that the step-start configuration has no contacts.
        detector.collide(data=solver._solver_kamino._data, contacts=contacts)
        self.assertEqual(int(contacts.model_active_contacts.numpy()[0]), 0)

        state_in = model.state()
        state_out = model.state()
        assert state_in.body_qd is not None
        # Over the first half-step, the sphere moves 0.35 m down and penetrates the plane.
        state_in.body_qd.assign(np.array([[0.0, 0.0, -7.0, 0.0, 0.0, 0.0]], dtype=np.float32))

        solver.step(state_in, state_out, control=None, contacts=None, dt=0.1)

        # This contact exists only if detection uses the midpoint rather than state_in.
        self.assertGreater(int(contacts.model_active_contacts.numpy()[0]), 0)

    def test_external_contacts_use_euler(self):
        """Verify external-contact configurations warn and pick Euler."""
        model = self._make_three_world_model()
        with self.assertLogs(level="WARNING") as logs:
            solver = SolverKamino(
                model,
                config=SolverKamino.Config(integrator="moreau", use_collision_detector=False),
            )

        self.assertIsInstance(solver._solver_kamino._integrator, IntegratorEuler)
        self.assertTrue(any("Falling back to the 'euler' integrator" in message for message in logs.output))

    def test_external_collisions_preserve_explicit_rigid_contact_max(self):
        """Verify external collisions preserve an explicit contact capacity."""
        model = self._make_three_world_model()
        model.rigid_contact_max = 1000

        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=False))

        self.assertEqual(model.rigid_contact_max, 1000)
        self.assertEqual(solver._contacts_kamino.world_max_contacts_host, [334, 334, 334])
        self.assertEqual(solver._contacts_kamino.model_max_contacts_host, 1002)

        contacts = newton.CollisionPipeline(model).contacts()
        with self.assertNoLogs(level="WARNING"):
            solver.update_contacts(contacts, model.state())

    def test_step_with_zero_max_contacts(self):
        """Verify SolverKamino.step() succeeds when the model admits no possible contacts."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(builder)
        build_binary_revolute_joint_test(builder=builder, ground=False)
        model = builder.finalize(device=self.default_device)

        solver = SolverKamino(model, config=SolverKamino.Config(use_collision_detector=True))

        # Check that collision detector was initialized, but no collision pipeline or contacts were
        # assigned.
        self.assertIsNotNone(solver._collision_detector_kamino)
        self.assertEqual(solver._collision_detector_kamino._model_max_contacts, 0)
        self.assertIsNone(solver._collision_detector_kamino._unified_pipeline)
        self.assertIsNone(solver._collision_detector_kamino._primitive_pipeline)
        self.assertIsNone(solver._contacts_kamino)

        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, None, None, dt=1.0 / 60.0)


class TestSolverKaminoStatus(unittest.TestCase):
    def setUp(self):
        """Configure the default Warp device for status tests."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def _make_three_world_model(self) -> newton.Model:
        """Build a three-world model for status contract checks."""
        source_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(source_builder)
        basics.build_sphere_on_plane(builder=source_builder, z_offset=0.5)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.replicate(source_builder, world_count=3)
        return builder.finalize(device=self.default_device, skip_validation_joints=True)

    def _step_and_assert_status_contract(
        self,
        solver: newton.solvers.SolverKamino,
        model: newton.Model,
    ) -> None:
        """Advance one step and verify the public terminal-status contract."""
        solver.step(model.state(), model.state(), control=None, contacts=None, dt=SIM_DT)
        status = solver.status
        common_fields = ("converged", "iterations", "r_p", "r_d", "r_c")

        self.assertIsInstance(status, wp.array)
        self.assertIs(status, solver._solver_kamino.solver_fd.data.status)
        self.assertEqual(status.shape, (3,))
        self.assertEqual(status.device, self.default_device)
        self.assertTrue(set(common_fields).issubset(status.dtype.vars))

        status_host = status.numpy()
        for field in common_fields:
            self.assertEqual(status_host[field].shape, (3,))
        self.assertTrue(np.all((status_host["converged"] == 0) | (status_host["converged"] == 1)))
        self.assertTrue(np.all(status_host["iterations"] > 0))
        for field in ("r_p", "r_d", "r_c"):
            self.assertTrue(np.all(np.isfinite(status_host[field])))
            self.assertTrue(np.all(status_host[field] >= 0.0))

    def test_padmm_status_is_always_available(self):
        """Verify PADMM terminal status is available regardless of detailed collection."""
        model = self._make_three_world_model()

        for collect_solver_info in (False, True):
            with self.subTest(collect_solver_info=collect_solver_info):
                solver = newton.solvers.SolverKamino(
                    model,
                    config=newton.solvers.SolverKamino.Config(
                        dynamics_solver="padmm",
                        collect_solver_info=collect_solver_info,
                    ),
                )
                self._step_and_assert_status_contract(solver, model)

    def test_dvi_status_is_always_available(self):
        """Verify DVI terminal status is available regardless of detailed collection."""
        model = self._make_three_world_model()

        for collect_solver_info in (False, True):
            with self.subTest(collect_solver_info=collect_solver_info):
                solver = newton.solvers.SolverKamino(
                    model,
                    config=newton.solvers.SolverKamino.Config(
                        dynamics_solver="dvi",
                        collect_solver_info=collect_solver_info,
                    ),
                )
                self._step_and_assert_status_contract(solver, model)


class TestSolverKaminoImpl(unittest.TestCase):
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

    ###
    # Test Solver Construction
    ###

    def test_00_make_default_invalid(self):
        """
        Test that creating a default Kamino solver without a model raises an error.
        """
        self.assertRaises(TypeError, SolverKaminoImpl)

    def test_01_make_default_valid_with_limits_and_without_contacts(self):
        """
        Test creating a default Kamino solver without support for contacts.
        """
        builder = make_homogeneous_builder(num_worlds=1, build_fn=build_boxes_fourbar)
        model = builder.finalize(device=self.default_device)
        solver = SolverKaminoImpl(model=model)
        self.assertIsInstance(solver, SolverKaminoImpl)
        assert_solver_components(self, solver)

    def test_02_make_default_valid_with_limits_and_with_contacts(self):
        """
        Test creating a default Kamino solver with support for contacts.
        """
        builder = make_homogeneous_builder(num_worlds=1, build_fn=build_boxes_fourbar)
        model = builder.finalize(device=self.default_device)
        _, world_max_contacts = builder.compute_required_contact_capacity(max_contacts_per_pair=16)
        contacts = ContactsKamino(capacity=world_max_contacts, device=model.device)
        solver = SolverKaminoImpl(model=model, contacts=contacts)
        self.assertIsInstance(solver, SolverKaminoImpl)
        assert_solver_components(self, solver)

    def test_03_make_default_valid_without_limits_and_without_contacts(self):
        """
        Test creating a default Kamino solver without support for contacts.
        """
        builder = make_homogeneous_builder(num_worlds=1, build_fn=build_boxes_fourbar, limits=False)
        model = builder.finalize(device=self.default_device)
        solver = SolverKaminoImpl(model=model)
        self.assertIsInstance(solver, SolverKaminoImpl)
        assert_solver_components(self, solver)
        self.assertIsNone(solver._limits.data.wid)

    def test_04_make_default_valid_without_limits_and_with_contacts(self):
        """
        Test creating a default Kamino solver with support for contacts.
        """
        builder = make_homogeneous_builder(num_worlds=1, build_fn=build_boxes_fourbar, limits=False)
        model = builder.finalize(device=self.default_device)
        _, world_max_contacts = builder.compute_required_contact_capacity(max_contacts_per_pair=16)
        contacts = ContactsKamino(capacity=world_max_contacts, device=model.device)
        solver = SolverKaminoImpl(model=model, contacts=contacts)
        self.assertIsInstance(solver, SolverKaminoImpl)
        assert_solver_components(self, solver)
        self.assertIsNone(solver._limits.data.wid)

    ###
    # Test Reset Operations
    ###

    def test_05_reset_to_default_state(self):
        """
        Test resetting multiple world solvers to default state defined in the model.
        """
        builder = make_homogeneous_builder(num_worlds=3, build_fn=build_boxes_fourbar, limits=False)
        model = builder.finalize(device=self.default_device)
        solver = SolverKaminoImpl(model=model)

        # Set a pre-step control callback to apply external forces
        # that will sufficiently perturb the system state
        solver.set_pre_step_callback(test_prestep_callback)

        # Create a state container to hold the output of the reset
        # and a world_mask array to specify which worlds to reset
        state_0 = model.state()
        state_p = model.state()
        state_n = model.state()
        control = model.control()
        world_mask = wp.array([False, True, False], dtype=wp.bool, device=self.default_device)

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the initial state
        solver.reset(state=state_n)

        # Check that all worlds were reset
        assert_states_equal(self, state_n, state_0)

        # Step the solver a few times to change the state
        solver._reset()
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Create a copy of the current state before reset
        state_n_ref = model.state()
        state_n_ref.copy_from(state_n)

        # Reset only the specified worlds to the initial state
        solver.reset(state=state_n, world_mask=world_mask)

        # Check that only the specified worlds were reset
        assert_states_close_masked(model, state_n, state_0, state_n_ref, world_mask)

    def test_06_reset_but_preserve_state(self):
        """
        Test resetting multiple world solvers while preserving the state.
        """
        builder = make_homogeneous_builder(num_worlds=3, build_fn=build_boxes_fourbar, limits=False)
        model = builder.finalize(device=self.default_device)
        solver = SolverKaminoImpl(model=model)

        # Set a pre-step control callback to apply external forces
        # that will sufficiently perturb the system state
        solver.set_pre_step_callback(test_prestep_callback)

        # Create a state container to hold the output of the reset
        # and a world_mask array to specify which worlds to reset
        state_0 = model.state()
        state_p = model.state()
        state_n = model.state()
        control = model.control()
        world_mask = wp.array([False, True, False], dtype=wp.bool, device=self.default_device)

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Create a copy of the current state before reset
        state_n_ref = model.state()
        state_n_ref.copy_from(state_n)

        # Reset select worlds, while preserving the state
        solver.reset(state=state_n, world_mask=world_mask, config=SolverKamino.ResetConfig.preserve())

        # Check that masked out worlds are fully preserved
        assert_states_close_masked(model, state_n, None, state_n_ref, world_mask=world_mask)

        # Check that positions/velocities are preserved in reset worlds
        # (except q_j_p, reset to match q_j)
        assert_states_close_masked(
            model,
            state_n,
            state_n_ref,
            None,
            world_mask=world_mask,
            positions=True,
            velocities=True,
            forces=False,
            match_q_j_p_with_q_j=True,
        )

        # Check that wrenches and multipliers are reset in reset worlds
        assert_states_close_masked(
            model,
            state_n,
            state_0,
            None,
            world_mask=world_mask,
            positions=False,
            velocities=False,
            forces=True,
        )

    def test_07_reset_to_base_state(self):
        """
        Test resetting multiple world solvers to specified floating base states.
        """
        builder = make_homogeneous_builder(num_worlds=3, build_fn=build_boxes_fourbar, limits=False)
        model = builder.finalize(device=self.default_device)
        solver = SolverKaminoImpl(model=model)

        # Set a pre-step control callback to apply external forces
        # that will sufficiently perturb the system state
        solver.set_pre_step_callback(test_prestep_callback)

        # Create a state container to hold the output of the reset
        # and a world_mask array to specify which worlds to reset
        state_p = model.state()
        state_n = model.state()
        control = model.control()
        world_mask = wp.array([True, True, False], dtype=wp.bool, device=self.default_device)

        # Define the reset base pose
        base_q_0_np = [0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        base_q_0_np = np.tile(base_q_0_np, reps=model.size.num_worlds).astype(np.float32)
        base_q_0_np = base_q_0_np.reshape(model.size.num_worlds, 7)
        base_q_0: wp.array[wp.transformf] = wp.array(base_q_0_np, dtype=wp.transformf, device=self.default_device)

        # Define the reset base twist
        base_u_0_np = [0.0, 1.5, 0.0, 0.0, 0.0, 0.0]
        base_u_0_np = np.tile(base_u_0_np, reps=model.size.num_worlds).astype(np.float32)
        base_u_0_np = base_u_0_np.reshape(model.size.num_worlds, 6)
        base_u_0: wp.array[wp.spatial_vectorf] = wp.array(
            base_u_0_np, dtype=wp.spatial_vectorf, device=self.default_device
        )

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the specified base pose
        reset_config = SolverKamino.ResetConfig(base_pose=SolverKamino.ResetConfig.FromBaseQ(base_q_0))
        solver.reset(state=state_n, config=reset_config)

        # Check consistency of state after reset
        check_post_reset_state_consistency(model=model, state=state_n)

        # Check if the assigned base body was correctly reset
        base_body_idx = model.info.base_body_index.numpy().copy()
        for wid in range(model.size.num_worlds):
            base_idx = base_body_idx[wid]
            if base_idx >= 0:
                np.testing.assert_allclose(
                    state_n.q_i.numpy()[base_idx],
                    base_q_0_np[wid],
                    rtol=rtol,
                    atol=atol,
                )

        # Step the solver a few times to change the state
        solver._reset()
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the specified base pose + velocity
        reset_config = SolverKamino.ResetConfig(
            base_pose=SolverKamino.ResetConfig.FromBaseQ(base_q_0),
            base_velocity=SolverKamino.ResetConfig.FromBaseU(base_u_0),
        )
        solver.reset(state=state_n, config=reset_config)

        # Check consistency of state after reset
        check_post_reset_state_consistency(model=model, state=state_n)

        # Check if the assigned base body was correctly reset
        for wid in range(model.size.num_worlds):
            base_idx = base_body_idx[wid]
            np.testing.assert_allclose(
                state_n.q_i.numpy()[base_idx],
                base_q_0_np[wid],
                rtol=rtol,
                atol=atol,
            )
            np.testing.assert_allclose(
                state_n.u_i.numpy()[base_idx],
                base_u_0_np[wid],
                rtol=rtol,
                atol=atol,
            )

        # Create a copy of the state after reset in all worlds
        state_n_reset_ref = model.state()
        state_n_reset_ref.copy_from(state_n)

        # Step the solver a few times to change the state
        solver._reset()
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Create a copy of the current state after stepping the solver
        state_n_stepped_ref = model.state()
        state_n_stepped_ref.copy_from(state_n)

        # Reset selected worlds to the specified base pose + velocity
        reset_config = SolverKamino.ResetConfig(
            base_pose=SolverKamino.ResetConfig.FromBaseQ(base_q_0),
            base_velocity=SolverKamino.ResetConfig.FromBaseU(base_u_0),
        )
        solver.reset(state=state_n, world_mask=world_mask, config=reset_config)

        # Check that state was correctly preserved or reset based on mask
        assert_states_close_masked(model, state_n, state_n_reset_ref, state_n_stepped_ref, world_mask)

    def test_08_reset_to_joint_state(self):
        """
        Test resetting multiple world solvers to specified joint states.
        """
        builder = make_homogeneous_builder(num_worlds=3, build_fn=build_boxes_fourbar, limits=False)
        model = builder.finalize(device=self.default_device)
        config = SolverKaminoImpl.Config(use_fk_solver=True)
        solver = SolverKaminoImpl(model=model, config=config)

        # Set a pre-step control callback to apply external forces
        # that will sufficiently perturb the system state
        solver.set_pre_step_callback(test_prestep_callback)

        # Create a state container to hold the output of the reset
        # and a world_mask array to specify which worlds to reset
        state_p = model.state()
        state_n = model.state()
        control = model.control()
        world_mask = wp.array([True, False, True], dtype=wp.bool, device=self.default_device)

        # Set default reset joint coordinates
        joint_q_0_np = [0.1, 0.1, 0.1, 0.1]
        joint_q_0_np = np.tile(joint_q_0_np, reps=model.size.num_worlds).astype(np.float32)
        joint_q_0: wp.array[wp.float32] = wp.array(joint_q_0_np, dtype=wp.float32, device=self.default_device)

        # Set default reset joint velocities
        joint_u_0_np = [0.1, 0.1, 0.1, 0.1]
        joint_u_0_np = np.tile(joint_u_0_np, reps=model.size.num_worlds).astype(np.float32)
        joint_u_0: wp.array[wp.float32] = wp.array(joint_u_0_np, dtype=wp.float32, device=self.default_device)

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the specified joint coords
        reset_config = SolverKamino.ResetConfig(
            body_poses=SolverKamino.ResetConfig.FromJointQ(joint_q_0),
        )
        solver.reset(state=state_n, config=reset_config)

        # Check consistency of state after reset
        check_post_reset_state_consistency(model=model, state=state_n)

        # Check that joint_q matches prescribed values for actuators
        joint_q_np = state_n.q_j.numpy()
        coords_offset = model.joints.coords_offset.numpy()
        is_actuator = model.joints.act_type.numpy() != JointActuationType.PASSIVE
        for jid in range(model.size.sum_of_num_joints):
            if not is_actuator[jid]:
                continue
            np.testing.assert_allclose(
                joint_q_np[coords_offset[jid] : coords_offset[jid + 1]],
                joint_q_0_np[coords_offset[jid] : coords_offset[jid + 1]],
                rtol=rtol,
                atol=atol,
            )

        # Step the solver a few times to change the state
        solver._reset()
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the specified joint coords + velocities
        reset_config = SolverKamino.ResetConfig(
            body_poses=SolverKamino.ResetConfig.FromJointQ(joint_q_0),
            body_velocities=SolverKamino.ResetConfig.FromJointU(joint_u_0),
        )
        solver.reset(state=state_n, config=reset_config)

        # Check consistency of state after reset
        check_post_reset_state_consistency(model=model, state=state_n)

        # Check that joint_q, joint_u matches prescribed values for actuators
        joint_q_np = state_n.q_j.numpy()
        joint_u_np = state_n.dq_j.numpy()
        coords_offset = model.joints.coords_offset.numpy()
        dofs_offset = model.joints.dofs_offset.numpy()
        is_actuator = model.joints.act_type.numpy() != JointActuationType.PASSIVE
        for jid in range(model.size.sum_of_num_joints):
            if not is_actuator[jid]:
                continue
            np.testing.assert_allclose(
                joint_q_np[coords_offset[jid] : coords_offset[jid + 1]],
                joint_q_0_np[coords_offset[jid] : coords_offset[jid + 1]],
                rtol=rtol,
                atol=atol,
            )
            np.testing.assert_allclose(
                joint_u_np[dofs_offset[jid] : dofs_offset[jid + 1]],
                joint_u_0_np[dofs_offset[jid] : dofs_offset[jid + 1]],
                rtol=rtol,
                atol=atol,
            )

        # Create a copy of the state after reset in all worlds
        state_n_reset_ref = model.state()
        state_n_reset_ref.copy_from(state_n)

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Create a copy of the current state after stepping the solver
        state_n_stepped_ref = model.state()
        state_n_stepped_ref.copy_from(state_n)

        # Reset selected worlds to the specified joint coords + velocities
        reset_config = SolverKamino.ResetConfig(
            body_poses=SolverKamino.ResetConfig.FromJointQ(joint_q_0),
            body_velocities=SolverKamino.ResetConfig.FromJointU(joint_u_0),
        )
        solver.reset(state=state_n, world_mask=world_mask, config=reset_config)

        # Check that state was correctly preserved or reset based on mask
        assert_states_close_masked(model, state_n, state_n_reset_ref, state_n_stepped_ref, world_mask)

    def test_09_reset_to_actuator_state(self):
        """
        Test resetting multiple world solvers to specified actuator states.
        """
        builder = make_homogeneous_builder(num_worlds=3, build_fn=build_boxes_fourbar, limits=False)
        model = builder.finalize(device=self.default_device)
        config = SolverKaminoImpl.Config(use_fk_solver=True)
        solver = SolverKaminoImpl(model=model, config=config)

        # Set a pre-step control callback to apply external forces
        # that will sufficiently perturb the system state
        solver.set_pre_step_callback(test_prestep_callback)

        # Create a state container to hold the output of the reset
        # and a world_mask array to specify which worlds to reset
        state_p = model.state()
        state_n = model.state()
        control = model.control()
        world_mask = wp.array([True, False, True], dtype=wp.bool, device=self.default_device)

        # Set default reset joint coordinates
        actuator_q_0_np = [0.25, 0.25]
        actuator_q_0_np = np.tile(actuator_q_0_np, reps=model.size.num_worlds)
        actuator_q_0: wp.array[wp.float32] = wp.array(actuator_q_0_np, dtype=wp.float32, device=self.default_device)

        # Set default reset joint velocities
        actuator_u_0_np = [-1.0, -1.0]
        actuator_u_0_np = np.tile(actuator_u_0_np, reps=model.size.num_worlds)
        actuator_u_0: wp.array[wp.float32] = wp.array(actuator_u_0_np, dtype=wp.float32, device=self.default_device)

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the specified actuator coords
        reset_config = SolverKamino.ResetConfig(
            body_poses=SolverKamino.ResetConfig.FromActuatorQ(actuator_q_0),
        )
        solver.reset(state=state_n, config=reset_config)

        # Check consistency of state after reset
        check_post_reset_state_consistency(model=model, state=state_n)

        # Check that joint_q matches prescribed values for actuators
        joint_q_np = state_n.q_j.numpy()
        coords_offset = model.joints.coords_offset.numpy()
        act_coords_offset = model.joints.actuated_coords_offset.numpy()
        is_actuator = model.joints.act_type.numpy() != JointActuationType.PASSIVE
        for jid in range(model.size.sum_of_num_joints):
            if not is_actuator[jid]:
                continue
            np.testing.assert_allclose(
                joint_q_np[coords_offset[jid] : coords_offset[jid + 1]],
                actuator_q_0_np[act_coords_offset[jid] : act_coords_offset[jid + 1]],
                rtol=rtol,
                atol=atol,
            )

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the specified actuator coords and velocities
        reset_config = SolverKamino.ResetConfig(
            body_poses=SolverKamino.ResetConfig.FromActuatorQ(actuator_q_0),
            body_velocities=SolverKamino.ResetConfig.FromActuatorU(actuator_u_0),
        )
        solver.reset(state=state_n, config=reset_config)

        # Check consistency of state after reset
        check_post_reset_state_consistency(model=model, state=state_n)

        # Check that joint_q, joint_u matches prescribed values for actuators
        joint_q_np = state_n.q_j.numpy()
        joint_u_np = state_n.dq_j.numpy()
        coords_offset = model.joints.coords_offset.numpy()
        act_coords_offset = model.joints.actuated_coords_offset.numpy()
        dofs_offset = model.joints.dofs_offset.numpy()
        act_dofs_offset = model.joints.actuated_dofs_offset.numpy()
        is_actuator = model.joints.act_type.numpy() != JointActuationType.PASSIVE
        for jid in range(model.size.sum_of_num_joints):
            if not is_actuator[jid]:
                continue
            np.testing.assert_allclose(
                joint_q_np[coords_offset[jid] : coords_offset[jid + 1]],
                actuator_q_0_np[act_coords_offset[jid] : act_coords_offset[jid + 1]],
                rtol=rtol,
                atol=atol,
            )
            np.testing.assert_allclose(
                joint_u_np[dofs_offset[jid] : dofs_offset[jid + 1]],
                actuator_u_0_np[act_dofs_offset[jid] : act_dofs_offset[jid + 1]],
                rtol=rtol,
                atol=atol,
            )

        # Create a copy of the state after reset in all worlds
        state_n_reset_ref = model.state()
        state_n_reset_ref.copy_from(state_n)

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Create a copy of the current state after stepping the solver
        state_n_stepped_ref = model.state()
        state_n_stepped_ref.copy_from(state_n)

        # Reset selected worlds to the specified actuator coords + velocities
        reset_config = SolverKamino.ResetConfig(
            body_poses=SolverKamino.ResetConfig.FromActuatorQ(actuator_q_0),
            body_velocities=SolverKamino.ResetConfig.FromActuatorU(actuator_u_0),
        )
        solver.reset(state=state_n, world_mask=world_mask, config=reset_config)

        # Check that state was correctly preserved or reset based on mask
        assert_states_close_masked(model, state_n, state_n_reset_ref, state_n_stepped_ref, world_mask)

    def test_10_reset_warmstarting_for_select_worlds(self):
        """Test resetting the contacts warmstarting cache for select worlds."""
        # Create a heterogenous, contact-rich model
        shape_types = [GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.BOX, GeoType.CONE]
        shape_types = [type_.name.lower() for type_ in shape_types]
        shape_pairs = [(type1, type2) for type1 in shape_types for type2 in shape_types]
        builder = testing.make_shape_pairs_builder(
            shape_pairs=shape_pairs,
            distance=0.0,
            ground_box=True,
            ground_z=-1.0,  # Ensures non-zero contact reactions after just a few steps
        )
        model = builder.finalize(device=self.default_device)

        # Initialize the solver and step it a few times
        contacts = ContactsKamino(model=model)
        config = SolverKamino.Config()
        config.padmm.warmstart_mode = "containers"
        solver = SolverKaminoImpl(model=model, contacts=contacts, config=config)
        state_p = model.state()
        state_n = model.state()
        control = model.control()
        detector = CollisionDetector(model)
        step_solver(
            num_steps=10,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            contacts=contacts,
            detector=detector,
        )

        # Check that all worlds have contacts, and that most reactions / velocities are non-trivial
        self.assertTrue(np.all(contacts.world_active_contacts.numpy() > 0))
        num_active = contacts.model_active_contacts.numpy()[0]
        reaction_norms = np.linalg.norm(contacts.reaction.numpy()[:num_active], axis=1)
        velocity_norms = np.linalg.norm(contacts.velocity.numpy()[:num_active], axis=1)
        self.assertTrue(np.count_nonzero(reaction_norms + velocity_norms) > 0.8 * num_active)

        # Reset the solver internals (including warmstarting) with a non-trivial mask
        rng = np.random.default_rng(self.seed)
        world_mask_np = sample_world_mask(model.size.num_worlds, rng, target_inactive_rate=0.5)[0]
        world_mask = wp.array(world_mask_np, dtype=wp.bool, device=self.default_device)
        solver.reset(state=state_n, world_mask=world_mask, config=SolverKamino.ResetConfig.preserve())

        # Run contact warmstarting, and check that contacts in reset worlds get cold-started
        solver._ws_contacts.warmstart(model=model, data=solver.data, contacts=contacts)
        reaction_norms = np.linalg.norm(contacts.reaction.numpy()[:num_active], axis=1)
        velocity_norms = np.linalg.norm(contacts.velocity.numpy()[:num_active], axis=1)
        contact_wid = contacts.wid.numpy()[:num_active]
        contact_reset_mask = world_mask_np[contact_wid]
        np.testing.assert_equal(reaction_norms[contact_reset_mask], 0.0)
        np.testing.assert_equal(velocity_norms[contact_reset_mask], 0.0)

        # Sanity check that not everything got cold-started
        self.assertTrue(max(reaction_norms) + max(velocity_norms) > 0.0)

    ###
    # Test Step Operations
    ###

    def test_11_step_multiple_worlds_from_initial_state_without_contacts(self):
        """
        Test stepping multiple worlds solvers initialized
        uniformly from the default initial state multiple times.
        """
        # Create a single-instance system
        single_builder = build_boxes_fourbar(ground=False)
        for i, body in enumerate(single_builder.all_bodies):
            msg.info(f"[single]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[single]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create a model and states from the builder
        single_model = single_builder.finalize(device=self.default_device)
        single_state_p = single_model.state()
        single_state_n = single_model.state()
        single_control = single_model.control()
        self.assertEqual(single_model.size.sum_of_num_bodies, 4)
        self.assertEqual(single_model.size.sum_of_num_joints, 4)
        for i, body in enumerate(single_builder.all_bodies):
            np.testing.assert_allclose(single_model.bodies.q_i_0.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_model.bodies.u_i_0.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_state_p.q_i.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_state_p.u_i.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_state_n.q_i.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_state_n.u_i.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[single]: [init]: model.size:\n{single_model.size}\n\n")
        msg.info(f"[single]: [init]: single_state_p.q_i:\n{single_state_p.q_i}\n\n")
        msg.info(f"[single]: [init]: single_state_p.u_i:\n{single_state_p.u_i}\n\n")
        msg.info(f"[single]: [init]: single_state_p.w_i:\n{single_state_p.w_i}\n\n")
        msg.info(f"[single]: [init]: single_state_p.q_j:\n{single_state_p.q_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.dq_j:\n{single_state_p.dq_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.lambda_kin_j:\n{single_state_p.lambda_kin_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.lambda_dyn_j:\n{single_state_p.lambda_dyn_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.lambda_f_j:\n{single_state_p.lambda_f_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.lambda_tau_j:\n{single_state_p.lambda_tau_j}\n\n")
        # Create simulator and check if the initial state is consistent with the contents of the builder
        single_solver = SolverKaminoImpl(model=single_model)
        self.assertIsInstance(single_solver, SolverKaminoImpl)
        assert_solver_components(self, single_solver)
        self.assertIs(single_solver._model, single_model)

        # Define the total number of sample steps to collect, and the
        # total number of execution steps from which to collect them
        num_worlds = 42
        num_steps = 1000

        # Collect the initial states
        initial_q_i = single_state_p.q_i.numpy().copy()
        initial_u_i = single_state_p.u_i.numpy().copy()
        initial_q_j = single_state_p.q_j.numpy().copy()
        initial_dq_j = single_state_p.dq_j.numpy().copy()
        initial_lambda_kin_j = single_state_p.lambda_kin_j.numpy().copy()
        initial_lambda_dyn_j = single_state_p.lambda_dyn_j.numpy().copy()
        msg.info(f"[samples]: [single]: [init]: q_i (shape={initial_q_i.shape}):\n{initial_q_i}\n")
        msg.info(f"[samples]: [single]: [init]: u_i (shape={initial_u_i.shape}):\n{initial_u_i}\n")
        msg.info(f"[samples]: [single]: [init]: w_i (shape={initial_u_i.shape}):\n{initial_u_i}\n")
        msg.info(f"[samples]: [single]: [init]: q_j (shape={initial_q_j.shape}):\n{initial_q_j}\n")
        msg.info(f"[samples]: [single]: [init]: dq_j (shape={initial_dq_j.shape}):\n{initial_dq_j}\n")
        msg.info(
            f"[samples]: [single]: [init]: lambda_kin_j (shape={initial_lambda_kin_j.shape}):\n{initial_lambda_kin_j}\n"
        )
        msg.info(
            f"[samples]: [single]: [init]: lambda_dyn_j (shape={initial_lambda_dyn_j.shape}):\n{initial_lambda_dyn_j}\n"
        )

        # Set a simple control callback that applies control inputs
        # NOTE: We use this to disturb the system from its initial state
        single_solver.set_pre_step_callback(test_prestep_callback)

        # Run the simulation for the specified number of steps
        msg.info(f"[single]: Executing {num_steps} single-world steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            single_solver.step(state_in=single_state_p, state_out=single_state_n, control=single_control, dt=0.001)
            if self.verbose or self.progress:
                wp.synchronize()
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Collect the initial and final states
        final_q_i = single_state_n.q_i.numpy().copy()
        final_u_i = single_state_n.u_i.numpy().copy()
        final_w_i = single_state_n.w_i.numpy().copy()
        final_q_j = single_state_n.q_j.numpy().copy()
        final_dq_j = single_state_n.dq_j.numpy().copy()
        final_lambda_kin_j = single_state_n.lambda_kin_j.numpy().copy()
        final_lambda_dyn_j = single_state_n.lambda_dyn_j.numpy().copy()
        final_lambda_f_j = single_state_n.lambda_f_j.numpy().copy()
        final_lambda_tau_j = single_state_n.lambda_tau_j.numpy().copy()
        msg.info(f"[samples]: [single]: [final]: q_i (shape={final_q_i.shape}):\n{final_q_i}\n")
        msg.info(f"[samples]: [single]: [final]: u_i (shape={final_u_i.shape}):\n{final_u_i}\n")
        msg.info(f"[samples]: [single]: [final]: w_i (shape={final_w_i.shape}):\n{final_w_i}\n")
        msg.info(f"[samples]: [single]: [final]: q_j (shape={final_q_j.shape}):\n{final_q_j}\n")
        msg.info(f"[samples]: [single]: [final]: dq_j (shape={final_dq_j.shape}):\n{final_dq_j}\n")
        msg.info(
            f"[samples]: [single]: [final]: lambda_kin_j (shape={final_lambda_kin_j.shape}):\n{final_lambda_kin_j}\n"
        )
        msg.info(
            f"[samples]: [single]: [final]: lambda_dyn_j (shape={final_lambda_dyn_j.shape}):\n{final_lambda_dyn_j}\n"
        )
        msg.info(f"[samples]: [single]: [final]: lambda_f_j (shape={final_lambda_f_j.shape}):\n{final_lambda_f_j}\n")
        msg.info(
            f"[samples]: [single]: [final]: lambda_tau_j (shape={final_lambda_tau_j.shape}):\n{final_lambda_tau_j}\n"
        )

        # Tile the collected states for comparison against the multi-instance simulator
        multi_init_q_i = np.tile(initial_q_i, (num_worlds, 1))
        multi_init_u_i = np.tile(initial_u_i, (num_worlds, 1))
        multi_init_q_j = np.tile(initial_q_j, (num_worlds, 1)).reshape(-1)
        multi_init_dq_j = np.tile(initial_dq_j, (num_worlds, 1)).reshape(-1)
        multi_final_q_i = np.tile(final_q_i, (num_worlds, 1))
        multi_final_u_i = np.tile(final_u_i, (num_worlds, 1))
        multi_final_q_j = np.tile(final_q_j, (num_worlds, 1)).reshape(-1)
        multi_final_dq_j = np.tile(final_dq_j, (num_worlds, 1)).reshape(-1)
        msg.info(f"[samples]: [multi] [init]: q_i (shape={multi_init_q_i.shape}):\n{multi_init_q_i}\n")
        msg.info(f"[samples]: [multi] [init]: u_i (shape={multi_init_u_i.shape}):\n{multi_init_u_i}\n")
        msg.info(f"[samples]: [multi] [init]: q_j (shape={multi_init_q_j.shape}):\n{multi_init_q_j}\n")
        msg.info(f"[samples]: [multi] [init]: dq_j (shape={multi_init_dq_j.shape}):\n{multi_init_dq_j}\n")
        msg.info(f"[samples]: [multi] [final]: q_i (shape={multi_final_q_i.shape}):\n{multi_final_q_i}\n")
        msg.info(f"[samples]: [multi] [final]: u_i (shape={multi_final_u_i.shape}):\n{multi_final_u_i}\n")
        msg.info(f"[samples]: [multi] [final]: q_j (shape={multi_final_q_j.shape}):\n{multi_final_q_j}\n")
        msg.info(f"[samples]: [multi] [final]: dq_j (shape={multi_final_dq_j.shape}):\n{multi_final_dq_j}\n")

        # Create a multi-instance system by replicating the single-instance builder
        multi_builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_fourbar, ground=False)
        for i, body in enumerate(multi_builder.all_bodies):
            msg.info(f"[multi]: [builder]: body {i}: bid: {body.bid}")
            msg.info(f"[multi]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[multi]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create a model and states from the builder
        multi_model = multi_builder.finalize(device=self.default_device)
        multi_state_p = multi_model.state()
        multi_state_n = multi_model.state()
        multi_control = multi_model.control()

        # Create simulator and check if the initial state is consistent with the contents of the builder
        multi_solver = SolverKaminoImpl(model=multi_model)
        self.assertEqual(multi_model.size.sum_of_num_bodies, single_model.size.sum_of_num_bodies * num_worlds)
        self.assertEqual(multi_model.size.sum_of_num_joints, single_model.size.sum_of_num_joints * num_worlds)
        for i, body in enumerate(multi_builder.all_bodies):
            np.testing.assert_allclose(multi_model.bodies.q_i_0.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_model.bodies.u_i_0.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_state_p.q_i.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_state_p.u_i.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_state_n.q_i.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_state_n.u_i.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [init]: sim.model.size:\n{multi_model.size}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.q_i:\n{multi_state_p.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.u_i:\n{multi_state_p.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.q_j:\n{multi_state_p.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.dq_j:\n{multi_state_p.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_i:\n{multi_state_n.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.u_i:\n{multi_state_n.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_j:\n{multi_state_n.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.dq_j:\n{multi_state_n.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.control.tau_j:\n{multi_control.tau_j}\n\n")

        # Check if the multi-instance simulator has initial states matching the tiled samples
        np.testing.assert_allclose(multi_state_p.q_i.numpy(), multi_init_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_p.u_i.numpy(), multi_init_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.q_i.numpy(), multi_init_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.u_i.numpy(), multi_init_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_p.q_j.numpy(), multi_init_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_p.dq_j.numpy(), multi_init_dq_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.q_j.numpy(), multi_init_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.dq_j.numpy(), multi_init_dq_j, rtol=rtol, atol=atol)

        # Set a simple control callback that applies control inputs
        # NOTE: We use this to disturb the system from its initial state
        multi_solver.set_pre_step_callback(test_prestep_callback)

        # Step the multi-instance simulator for the same number of steps
        msg.info(f"[multi]: Executing {num_steps} multi-world steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            multi_solver.step(state_in=multi_state_p, state_out=multi_state_n, control=multi_control, dt=0.001)
            if self.verbose or self.progress:
                wp.synchronize()
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [final]: multi_state_n.q_i:\n{multi_state_n.q_i}\n\n")
        msg.info(f"[multi]: [final]: multi_state_n.u_i:\n{multi_state_n.u_i}\n\n")
        msg.info(f"[multi]: [final]: multi_state_n.q_j:\n{multi_state_n.q_j}\n\n")
        msg.info(f"[multi]: [final]: multi_state_n.dq_j:\n{multi_state_n.dq_j}\n\n")

        # Check that the next states match the collected samples
        np.testing.assert_allclose(multi_state_n.q_i.numpy(), multi_final_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.u_i.numpy(), multi_final_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.q_j.numpy(), multi_final_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.dq_j.numpy(), multi_final_dq_j, rtol=rtol, atol=atol)

    def test_12_step_multiple_worlds_from_initial_state_with_contacts(self):
        """
        Test stepping multiple world solvers initialized
        uniformly from the default initial state multiple times.
        """
        # Create a single-instance system
        single_builder = build_boxes_fourbar(ground=True)
        for i, body in enumerate(single_builder.all_bodies):
            msg.info(f"[single]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[single]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create a model and states from the builder
        single_model = single_builder.finalize(device=self.default_device)
        single_state_p = single_model.state()
        single_state_n = single_model.state()
        single_control = single_model.control()
        self.assertEqual(single_model.size.sum_of_num_bodies, 4)
        self.assertEqual(single_model.size.sum_of_num_joints, 4)
        for i, body in enumerate(single_builder.all_bodies):
            np.testing.assert_allclose(single_model.bodies.q_i_0.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_model.bodies.u_i_0.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_state_p.q_i.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_state_p.u_i.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_state_n.q_i.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(single_state_n.u_i.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[single]: [init]: model.size:\n{single_model.size}\n\n")
        msg.info(f"[single]: [init]: single_state_p.q_i:\n{single_state_p.q_i}\n\n")
        msg.info(f"[single]: [init]: single_state_p.u_i:\n{single_state_p.u_i}\n\n")
        msg.info(f"[single]: [init]: single_state_p.w_i:\n{single_state_p.w_i}\n\n")
        msg.info(f"[single]: [init]: single_state_p.q_j:\n{single_state_p.q_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.dq_j:\n{single_state_p.dq_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.lambda_kin_j:\n{single_state_p.lambda_kin_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.lambda_dyn_j:\n{single_state_p.lambda_dyn_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.lambda_f_j:\n{single_state_p.lambda_f_j}\n\n")
        msg.info(f"[single]: [init]: single_state_p.lambda_tau_j:\n{single_state_p.lambda_tau_j}\n\n")
        # Create a contacts container for the single-instance system
        _, single_world_max_contacts = single_builder.compute_required_contact_capacity(max_contacts_per_pair=16)
        single_contacts = ContactsKamino(capacity=single_world_max_contacts, device=single_model.device)

        # Create simulator and check if the initial state is consistent with the contents of the builder
        single_solver = SolverKaminoImpl(model=single_model, contacts=single_contacts)
        self.assertIsInstance(single_solver, SolverKaminoImpl)
        assert_solver_components(self, single_solver)
        self.assertIs(single_solver._model, single_model)

        # Define the total number of sample steps to collect, and the
        # total number of execution steps from which to collect them
        num_worlds = 42
        num_steps = 1000

        # Collect the initial states
        initial_q_i = single_state_p.q_i.numpy().copy()
        initial_u_i = single_state_p.u_i.numpy().copy()
        initial_q_j = single_state_p.q_j.numpy().copy()
        initial_dq_j = single_state_p.dq_j.numpy().copy()
        msg.info(f"[samples]: [single]: [init]: q_i (shape={initial_q_i.shape}):\n{initial_q_i}\n")
        msg.info(f"[samples]: [single]: [init]: u_i (shape={initial_u_i.shape}):\n{initial_u_i}\n")
        msg.info(f"[samples]: [single]: [init]: w_i (shape={initial_u_i.shape}):\n{initial_u_i}\n")
        msg.info(f"[samples]: [single]: [init]: q_j (shape={initial_q_j.shape}):\n{initial_q_j}\n")
        msg.info(f"[samples]: [single]: [init]: dq_j (shape={initial_dq_j.shape}):\n{initial_dq_j}\n")
        msg.info(f"[samples]: [single]: [init]: lambda_kin_j (shape={initial_dq_j.shape}):\n{initial_dq_j}\n")

        # Set a simple control callback that applies control inputs
        # NOTE: We use this to disturb the system from its initial state
        single_solver.set_pre_step_callback(test_prestep_callback)

        # Run the simulation for the specified number of steps
        msg.info(f"[single]: Executing {num_steps} single-world steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            single_solver.step(single_state_p, single_state_n, single_control, contacts=single_contacts, dt=0.001)
            if self.verbose or self.progress:
                wp.synchronize()
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Collect the initial and final states
        final_q_i = single_state_n.q_i.numpy().copy()
        final_u_i = single_state_n.u_i.numpy().copy()
        final_w_i = single_state_n.w_i.numpy().copy()
        final_q_j = single_state_n.q_j.numpy().copy()
        final_dq_j = single_state_n.dq_j.numpy().copy()
        final_lambda_kin_j = single_state_n.lambda_kin_j.numpy().copy()
        final_lambda_dyn_j = single_state_n.lambda_dyn_j.numpy().copy()
        final_lambda_f_j = single_state_n.lambda_f_j.numpy().copy()
        final_lambda_tau_j = single_state_n.lambda_tau_j.numpy().copy()
        msg.info(f"[samples]: [single]: [final]: q_i (shape={final_q_i.shape}):\n{final_q_i}\n")
        msg.info(f"[samples]: [single]: [final]: u_i (shape={final_u_i.shape}):\n{final_u_i}\n")
        msg.info(f"[samples]: [single]: [final]: w_i (shape={final_w_i.shape}):\n{final_w_i}\n")
        msg.info(f"[samples]: [single]: [final]: q_j (shape={final_q_j.shape}):\n{final_q_j}\n")
        msg.info(f"[samples]: [single]: [final]: dq_j (shape={final_dq_j.shape}):\n{final_dq_j}\n")
        msg.info(
            f"[samples]: [single]: [final]: lambda_kin_j (shape={final_lambda_kin_j.shape}):\n{final_lambda_kin_j}\n"
        )
        msg.info(
            f"[samples]: [single]: [final]: lambda_dyn_j (shape={final_lambda_dyn_j.shape}):\n{final_lambda_dyn_j}\n"
        )
        msg.info(f"[samples]: [single]: [final]: lambda_f_j (shape={final_lambda_f_j.shape}):\n{final_lambda_f_j}\n")
        msg.info(
            f"[samples]: [single]: [final]: lambda_tau_j (shape={final_lambda_tau_j.shape}):\n{final_lambda_tau_j}\n"
        )

        # Tile the collected states for comparison against the multi-instance simulator
        multi_init_q_i = np.tile(initial_q_i, (num_worlds, 1))
        multi_init_u_i = np.tile(initial_u_i, (num_worlds, 1))
        multi_init_q_j = np.tile(initial_q_j, (num_worlds, 1)).reshape(-1)
        multi_init_dq_j = np.tile(initial_dq_j, (num_worlds, 1)).reshape(-1)
        multi_final_q_i = np.tile(final_q_i, (num_worlds, 1))
        multi_final_u_i = np.tile(final_u_i, (num_worlds, 1))
        multi_final_q_j = np.tile(final_q_j, (num_worlds, 1)).reshape(-1)
        multi_final_dq_j = np.tile(final_dq_j, (num_worlds, 1)).reshape(-1)
        msg.info(f"[samples]: [multi] [init]: q_i (shape={multi_init_q_i.shape}):\n{multi_init_q_i}\n")
        msg.info(f"[samples]: [multi] [init]: u_i (shape={multi_init_u_i.shape}):\n{multi_init_u_i}\n")
        msg.info(f"[samples]: [multi] [init]: q_j (shape={multi_init_q_j.shape}):\n{multi_init_q_j}\n")
        msg.info(f"[samples]: [multi] [init]: dq_j (shape={multi_init_dq_j.shape}):\n{multi_init_dq_j}\n")
        msg.info(f"[samples]: [multi] [final]: q_i (shape={multi_final_q_i.shape}):\n{multi_final_q_i}\n")
        msg.info(f"[samples]: [multi] [final]: u_i (shape={multi_final_u_i.shape}):\n{multi_final_u_i}\n")
        msg.info(f"[samples]: [multi] [final]: q_j (shape={multi_final_q_j.shape}):\n{multi_final_q_j}\n")
        msg.info(f"[samples]: [multi] [final]: dq_j (shape={multi_final_dq_j.shape}):\n{multi_final_dq_j}\n")

        # Create a multi-instance system by replicating the single-instance builder
        multi_builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_boxes_fourbar, ground=True)
        for i, body in enumerate(multi_builder.all_bodies):
            msg.info(f"[multi]: [builder]: body {i}: bid: {body.bid}")
            msg.info(f"[multi]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[multi]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create a model and states from the builder
        multi_model = multi_builder.finalize(device=self.default_device)
        multi_state_p = multi_model.state()
        multi_state_n = multi_model.state()
        multi_control = multi_model.control()

        # Create a contacts container for the multi-instance system
        _, multi_world_max_contacts = multi_builder.compute_required_contact_capacity(max_contacts_per_pair=16)
        multi_contacts = ContactsKamino(capacity=multi_world_max_contacts, device=multi_model.device)

        # Create simulator and check if the initial state is consistent with the contents of the builder
        multi_solver = SolverKaminoImpl(model=multi_model, contacts=multi_contacts)
        self.assertEqual(multi_model.size.sum_of_num_bodies, single_model.size.sum_of_num_bodies * num_worlds)
        self.assertEqual(multi_model.size.sum_of_num_joints, single_model.size.sum_of_num_joints * num_worlds)
        for i, body in enumerate(multi_builder.all_bodies):
            np.testing.assert_allclose(multi_model.bodies.q_i_0.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_model.bodies.u_i_0.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_state_p.q_i.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_state_p.u_i.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_state_n.q_i.numpy()[i], body.q_i_0, rtol=rtol, atol=atol)
            np.testing.assert_allclose(multi_state_n.u_i.numpy()[i], body.u_i_0, rtol=rtol, atol=atol)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [init]: sim.model.size:\n{multi_model.size}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.q_i:\n{multi_state_p.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.u_i:\n{multi_state_p.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.q_j:\n{multi_state_p.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.dq_j:\n{multi_state_p.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_i:\n{multi_state_n.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.u_i:\n{multi_state_n.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_j:\n{multi_state_n.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.dq_j:\n{multi_state_n.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.control.tau_j:\n{multi_control.tau_j}\n\n")

        # Check if the multi-instance simulator has initial states matching the tiled samples
        np.testing.assert_allclose(multi_state_p.q_i.numpy(), multi_init_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_p.u_i.numpy(), multi_init_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.q_i.numpy(), multi_init_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.u_i.numpy(), multi_init_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_p.q_j.numpy(), multi_init_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_p.dq_j.numpy(), multi_init_dq_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.q_j.numpy(), multi_init_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.dq_j.numpy(), multi_init_dq_j, rtol=rtol, atol=atol)

        # Set a simple control callback that applies control inputs
        # NOTE: We use this to disturb the system from its initial state
        multi_solver.set_pre_step_callback(test_prestep_callback)

        # Step the multi-instance simulator for the same number of steps
        msg.info(f"[multi]: Executing {num_steps} multi-world steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            multi_solver.step(multi_state_p, multi_state_n, multi_control, contacts=multi_contacts, dt=0.001)
            if self.verbose or self.progress:
                wp.synchronize()
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [final]: multi_state_n.q_i:\n{multi_state_n.q_i}\n\n")
        msg.info(f"[multi]: [final]: multi_state_n.u_i:\n{multi_state_n.u_i}\n\n")
        msg.info(f"[multi]: [final]: multi_state_n.q_j:\n{multi_state_n.q_j}\n\n")
        msg.info(f"[multi]: [final]: multi_state_n.dq_j:\n{multi_state_n.dq_j}\n\n")

        # Check that the next states match the collected samples
        np.testing.assert_allclose(multi_state_n.q_i.numpy(), multi_final_q_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.u_i.numpy(), multi_final_u_i, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.q_j.numpy(), multi_final_q_j, rtol=rtol, atol=atol)
        np.testing.assert_allclose(multi_state_n.dq_j.numpy(), multi_final_dq_j, rtol=rtol, atol=atol)


class TestSolverKaminoMasslessBodies(unittest.TestCase):
    """Massless bodies are supported when welded to the world and rejected otherwise."""

    def test_massless_body_welded_to_world_stays_at_rest(self):
        """Verify a massless body welded to the world stays at rest, along with its welded child.

        This is the ``world -> fixed -> base -> fixed -> link0`` topology produced by importing a
        fixed-base URDF whose root link carries no inertial properties.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        base_xform = wp.transform(wp.vec3(0.0, 0.0, BASE_HEIGHT), wp.quat_identity())
        base = builder.add_link(xform=base_xform, mass=0.0, label="base")
        link0 = builder.add_link(xform=base_xform, mass=2.0, inertia=UNIT_INERTIA, label="link0")
        joint_base = builder.add_joint_fixed(parent=-1, child=base, parent_xform=base_xform)
        joint_link0 = builder.add_joint_fixed(parent=base, child=link0)
        builder.add_articulation([joint_base, joint_link0])
        model = builder.finalize()

        inv_mass = model.body_inv_mass.numpy()
        self.assertEqual(inv_mass[base], 0.0)
        self.assertGreater(inv_mass[link0], 0.0)

        solver = SolverKamino(model)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        # A spurious velocity on the massless base is unrecoverable, so it must be caught per step
        # rather than only at the end, where it would have grown far beyond the tolerance.
        for step in range(30):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts=None, dt=SIM_DT)
            state_0, state_1 = state_1, state_0
            np.testing.assert_allclose(
                state_0.body_qd.numpy()[[base, link0]],
                0.0,
                atol=1.0e-5,
                err_msg=f"welded chain gained velocity at step {step}",
            )

        np.testing.assert_allclose(state_0.body_q.numpy()[[base, link0]][:, 2], BASE_HEIGHT, atol=1.0e-5)

    def test_massless_tip_behind_revolute_joint_is_rejected(self):
        """Verify a massless tip that is not welded to the world is rejected at construction.

        Kamino cannot simulate this model.

        The tip reaches the world only through a revolute joint, so it is free to move yet cannot be
        accelerated by every constraint reaction. Its missing response would propagate through the
        weld and prevent physically meaningful motion of the massive link carrying it, so the model
        must be rejected rather than simulated.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        link_xform = wp.transform(wp.vec3(0.0, 0.0, BASE_HEIGHT), wp.quat_identity())
        link0 = builder.add_link(xform=link_xform, mass=2.0, inertia=UNIT_INERTIA, label="link0")
        tip = builder.add_link(xform=link_xform, mass=0.0, label="tip")
        joint_link0 = builder.add_joint_revolute(parent=-1, child=link0, axis=newton.Axis.Y)
        joint_tip = builder.add_joint_fixed(parent=link0, child=tip)
        builder.add_articulation([joint_link0, joint_tip])
        model = builder.finalize()

        with self.assertRaises(ValueError) as ctx:
            SolverKamino(model)

        message = str(ctx.exception)
        self.assertIn("singular inertial properties", message)
        self.assertIn("'tip'", message)
        self.assertNotIn("'link0'", message)

    def test_partially_singular_tip_behind_revolute_joint_is_rejected(self):
        """Verify a partially singular tip is rejected at construction."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        link_xform = wp.transform(wp.vec3(0.0, 0.0, BASE_HEIGHT), wp.quat_identity())
        link0 = builder.add_link(xform=link_xform, mass=2.0, inertia=UNIT_INERTIA, label="link0")
        tip = builder.add_link(xform=link_xform, mass=2.0, inertia=UNIT_INERTIA, label="tip")
        joint_link0 = builder.add_joint_revolute(parent=-1, child=link0, axis=newton.Axis.Y)
        joint_tip = builder.add_joint_fixed(parent=link0, child=tip)
        builder.add_articulation([joint_link0, joint_tip])
        model = builder.finalize()

        inv_inertia = model.body_inv_inertia.numpy()
        inv_inertia[tip] = np.diag(np.array([0.0, 1.0, 1.0], dtype=np.float32))
        model.body_inv_inertia.assign(inv_inertia)

        with self.assertRaises(ValueError) as ctx:
            SolverKamino(model)

        message = str(ctx.exception)
        self.assertIn("singular inverse inertia", message)
        self.assertIn("'tip'", message)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
