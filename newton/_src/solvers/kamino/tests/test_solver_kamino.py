# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the :class:`SolverKaminoImpl` class"""

import time
import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.control import ControlKamino
from newton._src.solvers.kamino._src.core.data import DataKamino
from newton._src.solvers.kamino._src.core.joints import JointActuationType
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.core.state import StateKamino
from newton._src.solvers.kamino._src.core.types import float32, int32, transformf, vec6f
from newton._src.solvers.kamino._src.dynamics import DualProblem
from newton._src.solvers.kamino._src.geometry.contacts import ContactsKamino
from newton._src.solvers.kamino._src.kinematics.jacobians import DenseSystemJacobians, SparseSystemJacobians
from newton._src.solvers.kamino._src.kinematics.limits import LimitsKamino
from newton._src.solvers.kamino._src.models.builders.basics import build_boxes_fourbar
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino._src.solvers import PADMMSolver
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino.examples import print_progress_bar
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Kernels
###


@wp.kernel
def _test_control_callback(
    model_dt: wp.array[float32],
    data_time: wp.array[float32],
    control_tau_j: wp.array[float32],
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
    t_start = float32(0.0)
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
    np.testing.assert_array_equal(state_0.lambda_j.numpy(), state_1.lambda_j.numpy())


def assert_states_close(testcase: unittest.TestCase, state_0: StateKamino, state_1: StateKamino):
    testcase.assertIsInstance(state_0, StateKamino)
    testcase.assertIsInstance(state_1, StateKamino)
    np.testing.assert_allclose(state_0.q_i.numpy(), state_1.q_i.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.u_i.numpy(), state_1.u_i.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.w_i.numpy(), state_1.w_i.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.q_j.numpy(), state_1.q_j.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.q_j_p.numpy(), state_1.q_j_p.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.dq_j.numpy(), state_1.dq_j.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_allclose(state_0.lambda_j.numpy(), state_1.lambda_j.numpy(), rtol=rtol, atol=atol)


def assert_states_close_masked(
    testcase: unittest.TestCase,
    model: ModelKamino,
    state_0: StateKamino,
    state_n: StateKamino,
    state_n_ref: StateKamino,
    world_mask: wp.array,
):
    testcase.assertIsInstance(model, ModelKamino)
    testcase.assertIsInstance(state_0, StateKamino)
    testcase.assertIsInstance(state_n, StateKamino)

    num_bodies_per_world = model.size.max_of_num_bodies
    num_joint_dofs_per_world = model.size.max_of_num_joint_dofs
    num_joint_cts_per_world = model.size.max_of_num_joint_cts

    bodies_start = 0
    joint_dofs_start = 0
    joint_cts_start = 0
    world_mask_np = world_mask.numpy().copy()
    for wid in range(model.size.num_worlds):
        # Select reference state based on world mask
        if world_mask_np[wid]:
            state_ref = state_0
        else:
            state_ref = state_n_ref
        # Check state attributes for the current world
        for attr in ["q_i", "u_i", "w_i"]:
            np.testing.assert_allclose(
                getattr(state_n, attr).numpy()[bodies_start : bodies_start + num_bodies_per_world],
                getattr(state_ref, attr).numpy()[bodies_start : bodies_start + num_bodies_per_world],
                rtol=rtol,
                atol=atol,
                err_msg=f"\nWorld wid={wid}: attribute `{attr}` mismatch:\n",
            )
        for attr in ["q_j", "q_j_p", "dq_j"]:
            np.testing.assert_allclose(
                getattr(state_n, attr).numpy()[joint_dofs_start : joint_dofs_start + num_joint_dofs_per_world],
                getattr(state_ref, attr).numpy()[joint_dofs_start : joint_dofs_start + num_joint_dofs_per_world],
                rtol=rtol,
                atol=atol,
                err_msg=f"\nWorld wid={wid}: attribute `{attr}` mismatch:\n",
            )
        for attr in ["lambda_j"]:
            np.testing.assert_allclose(
                getattr(state_n, attr).numpy()[joint_cts_start : joint_cts_start + num_joint_cts_per_world],
                getattr(state_ref, attr).numpy()[joint_cts_start : joint_cts_start + num_joint_cts_per_world],
                rtol=rtol,
                atol=atol,
                err_msg=f"\nWorld wid={wid}: attribute `{attr}` mismatch:\n",
            )

        bodies_start += num_bodies_per_world
        joint_dofs_start += num_joint_dofs_per_world
        joint_cts_start += num_joint_cts_per_world


def step_solver(
    num_steps: int,
    solver: SolverKaminoImpl,
    state_p: StateKamino,
    state_n: StateKamino,
    control: ControlKamino,
    contacts: ContactsKamino | None = None,
    dt: float = 0.001,
    show_progress: bool = False,
):
    start_time = time.time()
    for step in range(num_steps):
        solver.step(state_in=state_p, state_out=state_n, control=control, contacts=contacts, dt=dt)
        wp.synchronize()
        state_p.copy_from(state_n)
        if show_progress:
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

    def test_01_make_explicit(self):
        config = SolverKaminoImpl.Config(
            dynamics=kamino_config.ConstrainedDynamicsConfig(linear_solver_type="CR"),
            padmm=kamino_config.PADMMSolverConfig(warmstart_mode="internal"),
            rotation_correction="continuous",
        )
        assert_solver_config(self, config)
        self.assertEqual(config.rotation_correction, "continuous")
        self.assertEqual(config.dynamics.linear_solver_type, "CR")
        self.assertEqual(config.padmm.warmstart_mode, "internal")


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

    def test_05_reset_with_invalid_args(self):
        """
        Test resetting multiple world solvers to default state defined in the model.
        """
        builder = make_homogeneous_builder(num_worlds=3, build_fn=build_boxes_fourbar, limits=False)
        model = builder.finalize(device=self.default_device)
        solver = SolverKaminoImpl(model=model)

        # Create reset argument arrays
        state_0 = model.state()
        with wp.ScopedDevice(self.default_device):
            base_q_0 = wp.zeros(shape=(model.size.num_worlds,), dtype=transformf)
            base_u_0 = wp.zeros(shape=(model.size.num_worlds,), dtype=vec6f)
            joint_q_0 = wp.zeros(shape=(model.size.sum_of_num_joint_coords,), dtype=float32)
            joint_u_0 = wp.zeros(shape=(model.size.sum_of_num_joint_dofs,), dtype=float32)
            actuator_q_0 = wp.zeros(shape=(model.size.sum_of_num_actuated_joint_coords,), dtype=float32)
            actuator_u_0 = wp.zeros(shape=(model.size.sum_of_num_actuated_joint_dofs,), dtype=float32)

        ###
        # Test invalid argument combinations to ensure errors are correctly raised
        ###

        # Setting only joint or actuator velocities is not supported
        self.assertRaises(ValueError, lambda: solver.reset(state_out=state_0, joint_u=joint_u_0))
        self.assertRaises(ValueError, lambda: solver.reset(state_out=state_0, actuator_u=actuator_u_0))

        # Setting both joint and actuator coordinates or velocities is not supported
        self.assertRaises(
            ValueError, lambda: solver.reset(state_out=state_0, joint_q=joint_q_0, actuator_q=actuator_q_0)
        )
        self.assertRaises(
            ValueError, lambda: solver.reset(state_out=state_0, joint_q=joint_q_0, actuator_u=actuator_u_0)
        )
        self.assertRaises(
            ValueError, lambda: solver.reset(state_out=state_0, joint_u=joint_u_0, actuator_q=actuator_q_0)
        )
        self.assertRaises(
            ValueError, lambda: solver.reset(state_out=state_0, joint_u=joint_u_0, actuator_u=actuator_u_0)
        )
        self.assertRaises(
            ValueError,
            lambda: solver.reset(state_out=state_0, base_q=base_q_0, joint_u=joint_u_0, actuator_u=actuator_u_0),
        )
        self.assertRaises(
            ValueError,
            lambda: solver.reset(state_out=state_0, base_u=base_u_0, joint_u=joint_u_0, actuator_u=actuator_u_0),
        )
        self.assertRaises(
            ValueError,
            lambda: solver.reset(
                state_out=state_0, base_q=base_q_0, base_u=base_u_0, joint_u=joint_u_0, actuator_u=actuator_u_0
            ),
        )

    def test_06_reset_to_default_state(self):
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
        world_mask = wp.array([0, 1, 0], dtype=int32, device=self.default_device)

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
        solver.reset(state_out=state_n)

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
        solver.reset(state_out=state_n, world_mask=world_mask)

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

        # Check that only the specified worlds were reset
        assert_states_close_masked(self, model, state_0, state_n, state_n_ref, world_mask)

    def test_07_reset_to_base_state(self):
        """
        Test resetting multiple world solvers to specified base states.
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
        world_mask = wp.array([1, 1, 0], dtype=int32, device=self.default_device)

        # Define the reset base pose
        base_q_0_np = [0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        base_q_0_np = np.tile(base_q_0_np, reps=model.size.num_worlds).astype(np.float32)
        base_q_0_np = base_q_0_np.reshape(model.size.num_worlds, 7)
        base_q_0: wp.array = wp.array(base_q_0_np, dtype=transformf, device=self.default_device)

        # Define the reset base twist
        base_u_0_np = [0.0, 1.5, 0.0, 0.0, 0.0, 0.0]
        base_u_0_np = np.tile(base_u_0_np, reps=model.size.num_worlds).astype(np.float32)
        base_u_0_np = base_u_0_np.reshape(model.size.num_worlds, 6)
        base_u_0: wp.array = wp.array(base_u_0_np, dtype=vec6f, device=self.default_device)

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the specified base poses
        solver.reset(
            state_out=state_n,
            base_q=base_q_0,
        )

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

        # Check if the assigned base body was correctly reset
        base_body_idx = model.info.base_body_index.numpy().copy()
        for wid in range(model.size.num_worlds):
            base_idx = base_body_idx[wid]
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

        # Reset all worlds to the specified base states
        solver.reset(
            state_out=state_n,
            base_q=base_q_0,
            base_u=base_u_0,
        )

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

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

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

        # Reset selected worlds to the specified base states
        solver.reset(
            state_out=state_n,
            world_mask=world_mask,
            base_q=base_q_0,
            base_u=base_u_0,
        )

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

        # Check if the assigned base body was correctly reset
        world_mask_np = world_mask.numpy().copy()
        for wid in range(model.size.num_worlds):
            if world_mask_np[wid]:
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
        world_mask = wp.array([1, 0, 1], dtype=int32, device=self.default_device)

        # Set default default reset joint coordinates
        joint_q_0_np = [0.1, 0.1, 0.1, 0.1]
        joint_q_0_np = np.tile(joint_q_0_np, reps=model.size.num_worlds).astype(np.float32)
        joint_q_0: wp.array = wp.array(joint_q_0_np, dtype=float32, device=self.default_device)

        # Set default default reset joint velocities
        joint_u_0_np = [0.1, 0.1, 0.1, 0.1]
        joint_u_0_np = np.tile(joint_u_0_np, reps=model.size.num_worlds).astype(np.float32)
        joint_u_0: wp.array = wp.array(joint_u_0_np, dtype=float32, device=self.default_device)

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the specified joint states
        solver.reset(
            state_out=state_n,
            joint_q=joint_q_0,
        )

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

        # Check if the assigned joint states were correctly reset
        np.testing.assert_allclose(
            state_n.q_j.numpy(),
            joint_q_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.q_j` does not match joint_q target\n",
        )
        np.testing.assert_allclose(
            state_n.q_j_p.numpy(),
            joint_q_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.q_j_p` does not match joint_q target\n",
        )
        self.assertTrue(
            np.isfinite(state_n.q_i.numpy()).all(),
            msg="\n`state_out.q_i` contains non-finite values (Inf/NaN)\n",
        )
        self.assertTrue(
            np.isfinite(state_n.u_i.numpy()).all(),
            msg="\n`state_out.u_i` contains non-finite values (Inf/NaN)\n",
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

        # Reset all worlds to the specified joint states
        solver.reset(
            state_out=state_n,
            joint_q=joint_q_0,
            joint_u=joint_u_0,
        )

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

        # Check if the assigned joint states were correctly reset
        np.testing.assert_allclose(
            state_n.q_j.numpy(),
            joint_q_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.q_j` does not match joint_q target\n",
        )
        np.testing.assert_allclose(
            state_n.q_j_p.numpy(),
            joint_q_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.q_j_p` does not match joint_q target\n",
        )
        np.testing.assert_allclose(
            state_n.dq_j.numpy(),
            joint_u_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.dq_j` does not match joint_u target\n",
        )
        self.assertTrue(
            np.isfinite(state_n.q_i.numpy()).all(),
            msg="\n`state_out.q_i` contains non-finite values (Inf/NaN)\n",
        )
        self.assertTrue(
            np.isfinite(state_n.u_i.numpy()).all(),
            msg="\n`state_out.u_i` contains non-finite values (Inf/NaN)\n",
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

        # Reset selected worlds to the specified joint states
        solver.reset(
            state_out=state_n,
            world_mask=world_mask,
            joint_q=joint_q_0,
            joint_u=joint_u_0,
        )

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

        # Check if the assigned joint states were correctly reset
        num_world_coords = model.size.max_of_num_joint_coords
        num_world_dofs = model.size.max_of_num_joint_dofs
        coords_start = 0
        dofs_start = 0
        world_mask_np = world_mask.numpy().copy()
        for wid in range(model.size.num_worlds):
            if world_mask_np[wid]:
                np.testing.assert_allclose(
                    state_n.q_j.numpy()[coords_start : coords_start + num_world_coords],
                    joint_q_0_np[coords_start : coords_start + num_world_coords],
                    rtol=rtol,
                    atol=atol,
                    err_msg="\n`state_out.q_j` does not match joint_q target\n",
                )
                np.testing.assert_allclose(
                    state_n.q_j_p.numpy()[coords_start : coords_start + num_world_coords],
                    joint_q_0_np[coords_start : coords_start + num_world_coords],
                    rtol=rtol,
                    atol=atol,
                    err_msg="\n`state_out.q_j_p` does not match joint_q target\n",
                )
                np.testing.assert_allclose(
                    state_n.dq_j.numpy()[dofs_start : dofs_start + num_world_dofs],
                    joint_u_0_np[dofs_start : dofs_start + num_world_dofs],
                    rtol=rtol,
                    atol=atol,
                    err_msg="\n`state_out.dq_j` does not match joint_u target\n",
                )
            coords_start += num_world_coords
            dofs_start += num_world_dofs
        self.assertTrue(
            np.isfinite(state_n.q_i.numpy()).all(),
            msg="\n`state_out.q_i` contains non-finite values (Inf/NaN)\n",
        )
        self.assertTrue(
            np.isfinite(state_n.u_i.numpy()).all(),
            msg="\n`state_out.u_i` contains non-finite values (Inf/NaN)\n",
        )

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
        world_mask = wp.array([1, 0, 1], dtype=int32, device=self.default_device)

        # Set default default reset joint coordinates
        actuator_q_0_np = [0.25, 0.25]
        actuator_q_0_np = np.tile(actuator_q_0_np, reps=model.size.num_worlds)
        actuator_q_0: wp.array = wp.array(actuator_q_0_np, dtype=float32, device=self.default_device)

        # Set default default reset joint velocities
        actuator_u_0_np = [-1.0, -1.0]
        actuator_u_0_np = np.tile(actuator_u_0_np, reps=model.size.num_worlds)
        actuator_u_0: wp.array = wp.array(actuator_u_0_np, dtype=float32, device=self.default_device)

        # Step the solver a few times to change the state
        step_solver(
            num_steps=11,
            solver=solver,
            state_p=state_p,
            state_n=state_n,
            control=control,
            show_progress=self.progress or self.verbose,
        )

        # Reset all worlds to the specified joint states
        solver.reset(
            state_out=state_n,
            actuator_q=actuator_q_0,
        )

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

        # Create expanded actuator state arrays matching the full joint state size
        joint_q_0_np = state_n.q_j.numpy().copy()
        joint_act_type = model.joints.act_type.numpy().copy()
        joint_num_coords = model.joints.num_coords.numpy().copy()
        jnt_coords_start = 0
        act_coords_start = 0
        for j in range(model.size.max_of_num_joints):
            nq = joint_num_coords[j]
            act_type = joint_act_type[j]
            if act_type > JointActuationType.PASSIVE:
                joint_q_0_np[jnt_coords_start : jnt_coords_start + nq] = actuator_q_0_np[
                    act_coords_start : act_coords_start + nq
                ]
                act_coords_start += nq
            jnt_coords_start += nq
        msg.info("state_n.q_q_j:\n%s\n", state_n.q_j)
        msg.info("joint_q_0_np:\n%s\n", joint_q_0_np)

        # Check if the assigned joint states were correctly reset
        np.testing.assert_allclose(
            state_n.q_j.numpy(),
            joint_q_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.q_j` does not match joint_q target\n",
        )
        np.testing.assert_allclose(
            state_n.q_j_p.numpy(),
            joint_q_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.q_j_p` does not match joint_q target\n",
        )
        self.assertTrue(
            np.isfinite(state_n.q_i.numpy()).all(),
            msg="\n`state_out.q_i` contains non-finite values (Inf/NaN)\n",
        )
        self.assertTrue(
            np.isfinite(state_n.u_i.numpy()).all(),
            msg="\n`state_out.u_i` contains non-finite values (Inf/NaN)\n",
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

        # Reset all worlds to the specified joint states
        solver.reset(
            state_out=state_n,
            actuator_q=actuator_q_0,
            actuator_u=actuator_u_0,
        )

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

        # Create expanded actuator state arrays matching the full joint state size
        joint_q_0_np = state_n.q_j.numpy().copy()
        joint_u_0_np = state_n.dq_j.numpy().copy()
        joint_num_dofs = model.joints.num_dofs.numpy().copy()
        jnt_coords_start = 0
        jnt_dofs_start = 0
        act_coord_start = 0
        act_dof_start = 0
        for j in range(model.size.max_of_num_joints):
            nq = joint_num_coords[j]
            nu = joint_num_dofs[j]
            act_type = joint_act_type[j]
            if act_type > JointActuationType.PASSIVE:
                joint_q_0_np[jnt_coords_start : jnt_coords_start + nq] = actuator_q_0_np[
                    act_coord_start : act_coord_start + nq
                ]
                joint_u_0_np[jnt_dofs_start : jnt_dofs_start + nu] = actuator_u_0_np[act_dof_start : act_dof_start + nu]
                act_coord_start += nq
                act_dof_start += nu
            jnt_coords_start += nq
            jnt_dofs_start += nu

        # Check if the assigned joint states were correctly reset
        np.testing.assert_allclose(
            state_n.q_j.numpy(),
            joint_q_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.q_j` does not match joint_q target\n",
        )
        np.testing.assert_allclose(
            state_n.q_j_p.numpy(),
            joint_q_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.q_j_p` does not match joint_q target\n",
        )
        np.testing.assert_allclose(
            state_n.dq_j.numpy(),
            joint_u_0_np,
            rtol=rtol,
            atol=atol,
            err_msg="\n`state_out.dq_j` does not match joint_u target\n",
        )
        self.assertTrue(
            np.isfinite(state_n.q_i.numpy()).all(),
            msg="\n`state_out.q_i` contains non-finite values (Inf/NaN)\n",
        )
        self.assertTrue(
            np.isfinite(state_n.u_i.numpy()).all(),
            msg="\n`state_out.u_i` contains non-finite values (Inf/NaN)\n",
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

        # Reset all worlds to the specified joint states
        solver.reset(
            state_out=state_n,
            world_mask=world_mask,
            actuator_q=actuator_q_0,
            actuator_u=actuator_u_0,
        )

        # Optionally print the reset state for debugging
        msg.info("state_n.q_i:\n%s\n", state_n.q_i)
        msg.info("state_n.u_i:\n%s\n", state_n.u_i)
        msg.info("state_n.w_i:\n%s\n", state_n.w_i)
        msg.info("state_n.q_j:\n%s\n", state_n.q_j)
        msg.info("state_n.q_j_p:\n%s\n", state_n.q_j_p)
        msg.info("state_n.dq_j:\n%s\n", state_n.dq_j)
        msg.info("state_n.lambda_j:\n%s\n", state_n.lambda_j)

        # Create expanded actuator state arrays matching the full joint state size
        joint_q_0_np = state_n.q_j.numpy().copy()
        joint_u_0_np = state_n.dq_j.numpy().copy()
        jnt_coords_start = 0
        jnt_dofs_start = 0
        act_coord_start = 0
        act_dof_start = 0
        for j in range(model.size.max_of_num_joints):
            nq = joint_num_coords[j]
            nu = joint_num_dofs[j]
            act_type = joint_act_type[j]
            if act_type > JointActuationType.PASSIVE:
                joint_q_0_np[jnt_coords_start : jnt_coords_start + nq] = actuator_q_0_np[
                    act_coord_start : act_coord_start + nq
                ]
                joint_u_0_np[jnt_dofs_start : jnt_dofs_start + nu] = actuator_u_0_np[act_dof_start : act_dof_start + nu]
                act_coord_start += nq
                act_dof_start += nu
            jnt_coords_start += nq
            jnt_dofs_start += nu

        # Check if the assigned joint states were correctly reset
        num_world_coords = model.size.max_of_num_joint_coords
        num_world_dofs = model.size.max_of_num_joint_dofs
        coords_start = 0
        dofs_start = 0
        world_mask_np = world_mask.numpy().copy()
        for wid in range(model.size.num_worlds):
            if world_mask_np[wid]:
                np.testing.assert_allclose(
                    state_n.q_j.numpy()[coords_start : coords_start + num_world_coords],
                    joint_q_0_np[coords_start : coords_start + num_world_coords],
                    rtol=rtol,
                    atol=atol,
                    err_msg="\n`state_out.q_j` does not match joint_q target\n",
                )
                np.testing.assert_allclose(
                    state_n.q_j_p.numpy()[coords_start : coords_start + num_world_coords],
                    joint_q_0_np[coords_start : coords_start + num_world_coords],
                    rtol=rtol,
                    atol=atol,
                    err_msg="\n`state_out.q_j_p` does not match joint_q target\n",
                )
                np.testing.assert_allclose(
                    state_n.dq_j.numpy()[dofs_start : dofs_start + num_world_dofs],
                    joint_u_0_np[dofs_start : dofs_start + num_world_dofs],
                    rtol=rtol,
                    atol=atol,
                    err_msg="\n`state_out.dq_j` does not match joint_u target\n",
                )
            coords_start += num_world_coords
            dofs_start += num_world_dofs
        self.assertTrue(
            np.isfinite(state_n.q_i.numpy()).all(),
            msg="\n`state_out.q_i` contains non-finite values (Inf/NaN)\n",
        )
        self.assertTrue(
            np.isfinite(state_n.u_i.numpy()).all(),
            msg="\n`state_out.u_i` contains non-finite values (Inf/NaN)\n",
        )

    ###
    # Test Step Operations
    ###

    def test_10_step_multiple_worlds_from_initial_state_without_contacts(self):
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
        msg.info(f"[single]: [init]: single_state_p.lambda_j:\n{single_state_p.lambda_j}\n\n")

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
        msg.info(f"[samples]: [single]: [init]: q_i (shape={initial_q_i.shape}):\n{initial_q_i}\n")
        msg.info(f"[samples]: [single]: [init]: u_i (shape={initial_u_i.shape}):\n{initial_u_i}\n")
        msg.info(f"[samples]: [single]: [init]: w_i (shape={initial_u_i.shape}):\n{initial_u_i}\n")
        msg.info(f"[samples]: [single]: [init]: q_j (shape={initial_q_j.shape}):\n{initial_q_j}\n")
        msg.info(f"[samples]: [single]: [init]: dq_j (shape={initial_dq_j.shape}):\n{initial_dq_j}\n")
        msg.info(f"[samples]: [single]: [init]: lambda_j (shape={initial_dq_j.shape}):\n{initial_dq_j}\n")

        # Set a simple control callback that applies control inputs
        # NOTE: We use this to disturb the system from its initial state
        single_solver.set_pre_step_callback(test_prestep_callback)

        # Run the simulation for the specified number of steps
        msg.info(f"[single]: Executing {num_steps} single-world steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            single_solver.step(state_in=single_state_p, state_out=single_state_n, control=single_control, dt=0.001)
            wp.synchronize()
            if self.verbose or self.progress:
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Collect the initial and final states
        final_q_i = single_state_n.q_i.numpy().copy()
        final_u_i = single_state_n.u_i.numpy().copy()
        final_w_i = single_state_n.w_i.numpy().copy()
        final_q_j = single_state_n.q_j.numpy().copy()
        final_dq_j = single_state_n.dq_j.numpy().copy()
        final_lambda_j = single_state_n.lambda_j.numpy().copy()
        msg.info(f"[samples]: [single]: [final]: q_i (shape={final_q_i.shape}):\n{final_q_i}\n")
        msg.info(f"[samples]: [single]: [final]: u_i (shape={final_u_i.shape}):\n{final_u_i}\n")
        msg.info(f"[samples]: [single]: [final]: w_i (shape={final_w_i.shape}):\n{final_w_i}\n")
        msg.info(f"[samples]: [single]: [final]: q_j (shape={final_q_j.shape}):\n{final_q_j}\n")
        msg.info(f"[samples]: [single]: [final]: dq_j (shape={final_dq_j.shape}):\n{final_dq_j}\n")
        msg.info(f"[samples]: [single]: [final]: lambda_j (shape={final_lambda_j.shape}):\n{final_lambda_j}\n")

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
            wp.synchronize()
            if self.verbose or self.progress:
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

    def test_11_step_multiple_worlds_from_initial_state_with_contacts(self):
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
        msg.info(f"[single]: [init]: single_state_p.lambda_j:\n{single_state_p.lambda_j}\n\n")

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
        msg.info(f"[samples]: [single]: [init]: lambda_j (shape={initial_dq_j.shape}):\n{initial_dq_j}\n")

        # Set a simple control callback that applies control inputs
        # NOTE: We use this to disturb the system from its initial state
        single_solver.set_pre_step_callback(test_prestep_callback)

        # Run the simulation for the specified number of steps
        msg.info(f"[single]: Executing {num_steps} single-world steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            single_solver.step(single_state_p, single_state_n, single_control, contacts=single_contacts, dt=0.001)
            wp.synchronize()
            if self.verbose or self.progress:
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Collect the initial and final states
        final_q_i = single_state_n.q_i.numpy().copy()
        final_u_i = single_state_n.u_i.numpy().copy()
        final_w_i = single_state_n.w_i.numpy().copy()
        final_q_j = single_state_n.q_j.numpy().copy()
        final_dq_j = single_state_n.dq_j.numpy().copy()
        final_lambda_j = single_state_n.lambda_j.numpy().copy()
        msg.info(f"[samples]: [single]: [final]: q_i (shape={final_q_i.shape}):\n{final_q_i}\n")
        msg.info(f"[samples]: [single]: [final]: u_i (shape={final_u_i.shape}):\n{final_u_i}\n")
        msg.info(f"[samples]: [single]: [final]: w_i (shape={final_w_i.shape}):\n{final_w_i}\n")
        msg.info(f"[samples]: [single]: [final]: q_j (shape={final_q_j.shape}):\n{final_q_j}\n")
        msg.info(f"[samples]: [single]: [final]: dq_j (shape={final_dq_j.shape}):\n{final_dq_j}\n")
        msg.info(f"[samples]: [single]: [final]: lambda_j (shape={final_lambda_j.shape}):\n{final_lambda_j}\n")

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
            wp.synchronize()
            if self.verbose or self.progress:
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


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
