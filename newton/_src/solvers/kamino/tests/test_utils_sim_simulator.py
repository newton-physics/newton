# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the high-level Simulator class utility of Kamino"""

import time
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.core.types import float32
from newton._src.solvers.kamino._src.models.builders.basics import build_cartpole
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.sim.simulator import Simulator
from newton._src.solvers.kamino.examples import print_progress_bar
from newton._src.solvers.kamino.tests import setup_tests, test_context

###
# Kernels
###


@wp.kernel
def _test_control_callback(
    model_dt: wp.array(dtype=float32),
    data_time: wp.array(dtype=float32),
    control_tau_j: wp.array(dtype=float32),
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


def test_control_callback(sim: Simulator):
    """
    A control callback function
    """
    wp.launch(
        _test_control_callback,
        dim=sim.model.size.num_worlds,
        inputs=[
            sim.model.time.dt,
            sim.solver.data.time.time,
            sim.control.tau_j,
        ],
    )


###
# Tests
###


class TestCartpoleSimulator(unittest.TestCase):
    def setUp(self):
        # Configs
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.seed = 42
        self.default_device = wp.get_device(test_context.device)
        self.verbose = test_context.verbose  # Set to True for verbose output
        self.progress = test_context.verbose  # Set to True for progress output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_01_step_multiple_cartpoles_all_from_initial_state(self):
        """
        Test stepping multiple cartpole simulators initialized
        uniformly from the default initial state multiple times.
        """

        # Create a single-instance system
        single_builder = build_cartpole(ground=False)
        for i, body in enumerate(single_builder.bodies):
            msg.info(f"[single]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[single]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create simulator and check if the initial state is consistent with the contents of the builder
        single_sim = Simulator(builder=single_builder, device=self.default_device)
        single_sim.set_control_callback(test_control_callback)
        self.assertEqual(single_sim.model.size.sum_of_num_bodies, 2)
        self.assertEqual(single_sim.model.size.sum_of_num_joints, 2)
        for i, body in enumerate(single_builder.bodies):
            np.testing.assert_allclose(single_sim.model.bodies.q_i_0.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(single_sim.model.bodies.u_i_0.numpy()[i], body.u_i_0)
            np.testing.assert_allclose(single_sim.state.q_i.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(single_sim.state.u_i.numpy()[i], body.u_i_0)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[single]: [init]: sim.model.size:\n{single_sim.model.size}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.q_i:\n{single_sim.state.q_i}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.u_i:\n{single_sim.state.u_i}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.q_j:\n{single_sim.state.q_j}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.dq_j:\n{single_sim.state.dq_j}\n\n")

        # Define the total number of sample steps to collect, and the
        # total number of execution steps from which to collect them
        num_worlds = 42
        num_steps = 1000

        # Collect the initial states
        initial_q_i = single_sim.state.q_i.numpy().copy()
        initial_u_i = single_sim.state.u_i.numpy().copy()
        initial_q_j = single_sim.state.q_j.numpy().copy()
        initial_dq_j = single_sim.state.dq_j.numpy().copy()
        msg.info(f"[samples]: [single]: [init]: q_i (shape={initial_q_i.shape}):\n{initial_q_i}\n")
        msg.info(f"[samples]: [single]: [init]: u_i (shape={initial_u_i.shape}):\n{initial_u_i}\n")
        msg.info(f"[samples]: [single]: [init]: q_j (shape={initial_q_j.shape}):\n{initial_q_j}\n")
        msg.info(f"[samples]: [single]: [init]: dq_j (shape={initial_dq_j.shape}):\n{initial_dq_j}\n")

        # Run the simulation for the specified number of steps
        msg.info(f"[single]: Executing {num_steps} simulator steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            single_sim.step()
            wp.synchronize()
            if self.verbose or self.progress:
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Collect the initial and final states
        final_q_i = single_sim.state.q_i.numpy().copy()
        final_u_i = single_sim.state.u_i.numpy().copy()
        final_q_j = single_sim.state.q_j.numpy().copy()
        final_dq_j = single_sim.state.dq_j.numpy().copy()
        msg.info(f"[samples]: [single]: [final]: q_i (shape={final_q_i.shape}):\n{final_q_i}\n")
        msg.info(f"[samples]: [single]: [final]: u_i (shape={final_u_i.shape}):\n{final_u_i}\n")
        msg.info(f"[samples]: [single]: [final]: q_j (shape={final_q_j.shape}):\n{final_q_j}\n")
        msg.info(f"[samples]: [single]: [final]: dq_j (shape={final_dq_j.shape}):\n{final_dq_j}\n")

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
        multi_builder = make_homogeneous_builder(num_worlds=num_worlds, build_fn=build_cartpole, ground=False)
        for i, body in enumerate(multi_builder.bodies):
            msg.info(f"[multi]: [builder]: body {i}: bid: {body.bid}")
            msg.info(f"[multi]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[multi]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create simulator and check if the initial state is consistent with the contents of the builder
        multi_sim = Simulator(builder=multi_builder, device=self.default_device)
        multi_sim.set_control_callback(test_control_callback)
        self.assertEqual(multi_sim.model.size.sum_of_num_bodies, single_sim.model.size.sum_of_num_bodies * num_worlds)
        self.assertEqual(multi_sim.model.size.sum_of_num_joints, single_sim.model.size.sum_of_num_joints * num_worlds)
        for i, body in enumerate(multi_builder.bodies):
            np.testing.assert_allclose(multi_sim.model.bodies.q_i_0.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(multi_sim.model.bodies.u_i_0.numpy()[i], body.u_i_0)
            np.testing.assert_allclose(multi_sim.state_previous.q_i.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(multi_sim.state_previous.u_i.numpy()[i], body.u_i_0)
            np.testing.assert_allclose(multi_sim.state.q_i.numpy()[i], body.q_i_0)
            np.testing.assert_allclose(multi_sim.state.u_i.numpy()[i], body.u_i_0)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [init]: sim.model.size:\n{multi_sim.model.size}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.q_i:\n{multi_sim.state_previous.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.u_i:\n{multi_sim.state_previous.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.q_j:\n{multi_sim.state_previous.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state_previous.dq_j:\n{multi_sim.state_previous.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.control.tau_j:\n{multi_sim.control.tau_j}\n\n")

        # Check if the multi-instance simulator has initial states matching the tiled samples
        np.testing.assert_allclose(multi_sim.state_previous.q_i.numpy(), multi_init_q_i)
        np.testing.assert_allclose(multi_sim.state_previous.u_i.numpy(), multi_init_u_i)
        np.testing.assert_allclose(multi_sim.state.q_i.numpy(), multi_init_q_i)
        np.testing.assert_allclose(multi_sim.state.u_i.numpy(), multi_init_u_i)
        np.testing.assert_allclose(multi_sim.state_previous.q_j.numpy(), multi_init_q_j)
        np.testing.assert_allclose(multi_sim.state_previous.dq_j.numpy(), multi_init_dq_j)
        np.testing.assert_allclose(multi_sim.state.q_j.numpy(), multi_init_q_j)
        np.testing.assert_allclose(multi_sim.state.dq_j.numpy(), multi_init_dq_j)

        # Step the multi-instance simulator for the same number of steps
        msg.info(f"[multi]: Executing {num_steps} simulator steps")
        start_time = time.time()
        for step in range(num_steps):
            # Execute a single simulation step
            multi_sim.step()
            wp.synchronize()
            if self.verbose or self.progress:
                print_progress_bar(step + 1, num_steps, start_time, prefix="Progress", suffix="")

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [final]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [final]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [final]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [final]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")

        # Check that the next states match the collected samples
        np.testing.assert_allclose(multi_sim.state.q_i.numpy(), multi_final_q_i)
        np.testing.assert_allclose(multi_sim.state.u_i.numpy(), multi_final_u_i)
        np.testing.assert_allclose(multi_sim.state.q_j.numpy(), multi_final_q_j)
        np.testing.assert_allclose(multi_sim.state.dq_j.numpy(), multi_final_dq_j)

    def test_step_02_multiple_cartpoles_reset_all_from_sampled_states(self):
        """
        Test stepping multiple cartpole simulators once but initialized from
        states collected from a single-instance simulator over multiple steps.
        """

        # Create a single-instance system
        single_builder = build_cartpole(ground=False)
        for i, body in enumerate(single_builder.bodies):
            msg.info(f"[single]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[single]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create simulator and check if the initial state is consistent with the contents of the builder
        single_sim = Simulator(builder=single_builder, device=self.default_device)
        single_sim.set_control_callback(test_control_callback)
        self.assertEqual(single_sim.model.size.sum_of_num_bodies, 2)
        self.assertEqual(single_sim.model.size.sum_of_num_joints, 2)
        for i, b in enumerate(single_builder.bodies):
            np.testing.assert_allclose(single_sim.model.bodies.q_i_0.numpy()[i], b.q_i_0)
            np.testing.assert_allclose(single_sim.model.bodies.u_i_0.numpy()[i], b.u_i_0)
            np.testing.assert_allclose(single_sim.state.q_i.numpy()[i], b.q_i_0)
            np.testing.assert_allclose(single_sim.state.u_i.numpy()[i], b.u_i_0)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[single]: [init]: sim.model.size:\n{single_sim.model.size}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.q_i:\n{single_sim.state.q_i}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.u_i:\n{single_sim.state.u_i}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.q_j:\n{single_sim.state.q_j}\n\n")
        msg.info(f"[single]: [init]: sim.model.state.dq_j:\n{single_sim.state.dq_j}\n\n")

        # Define the total number of sample steps to collect, and the
        # total number of execution steps from which to collect them
        num_sample_steps = 37
        num_skip_steps = 0
        num_exec_steps = 1000

        # Allocate arrays to hold the collected samples
        num_bodies = single_sim.model.size.sum_of_num_bodies
        num_joint_dofs = single_sim.model.size.sum_of_num_joint_dofs
        num_joint_cts = single_sim.model.size.sum_of_num_joint_cts
        sample_init_q_i = np.zeros((num_sample_steps, num_bodies, 7), dtype=np.float32)
        sample_init_u_i = np.zeros((num_sample_steps, num_bodies, 6), dtype=np.float32)
        sample_next_q_i = np.zeros((num_sample_steps, num_bodies, 7), dtype=np.float32)
        sample_next_u_i = np.zeros((num_sample_steps, num_bodies, 6), dtype=np.float32)
        sample_init_q_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)
        sample_init_dq_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)
        sample_init_lambda_j = np.zeros((num_sample_steps, num_joint_cts), dtype=np.float32)
        sample_next_q_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)
        sample_next_dq_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)
        sample_ctrl_tau_j = np.zeros((num_sample_steps, num_joint_dofs), dtype=np.float32)

        # Run the simulation for the specified number of steps
        sample_freq = max(1, num_exec_steps // num_sample_steps)
        sample = 0
        msg.info(f"[sample]: sampling {num_sample_steps} transitions over {num_exec_steps} simulator steps")
        total_steps = num_skip_steps + num_exec_steps
        start_time = time.time()
        for step in range(total_steps):
            # Execute a single simulation step
            single_sim.step()
            wp.synchronize()
            if self.verbose or self.progress:
                print_progress_bar(step + 1, total_steps, start_time, prefix="Progress", suffix="")
            # Collect the initial and next state samples at the specified frequency
            if step >= num_skip_steps and step % sample_freq == 0 and sample < num_sample_steps:
                sample_init_q_i[sample, :, :] = single_sim.state_previous.q_i.numpy().copy()
                sample_init_u_i[sample, :, :] = single_sim.state_previous.u_i.numpy().copy()
                sample_next_q_i[sample, :, :] = single_sim.state.q_i.numpy().copy()
                sample_next_u_i[sample, :, :] = single_sim.state.u_i.numpy().copy()
                sample_init_q_j[sample, :] = single_sim.state_previous.q_j.numpy().copy()
                sample_init_dq_j[sample, :] = single_sim.state_previous.dq_j.numpy().copy()
                sample_init_lambda_j[sample, :] = single_sim.state_previous.lambda_j.numpy().copy()
                sample_next_q_j[sample, :] = single_sim.state.q_j.numpy().copy()
                sample_next_dq_j[sample, :] = single_sim.state.dq_j.numpy().copy()
                sample_ctrl_tau_j[sample, :] = single_sim.control.tau_j.numpy().copy()
                sample += 1

        # Reshape samples for easier comparison later
        sample_init_q_i = sample_init_q_i.reshape(-1, 7)
        sample_init_u_i = sample_init_u_i.reshape(-1, 6)
        sample_next_q_i = sample_next_q_i.reshape(-1, 7)
        sample_next_u_i = sample_next_u_i.reshape(-1, 6)
        sample_init_q_j = sample_init_q_j.reshape(-1)
        sample_init_dq_j = sample_init_dq_j.reshape(-1)
        sample_init_lambda_j = sample_init_lambda_j.reshape(-1)
        sample_next_q_j = sample_next_q_j.reshape(-1)
        sample_next_dq_j = sample_next_dq_j.reshape(-1)
        sample_ctrl_tau_j = sample_ctrl_tau_j.reshape(-1)

        # Optional verbose output
        msg.info(f"[samples]: init q_i (shape={sample_init_q_i.shape}):\n{sample_init_q_i}\n")
        msg.info(f"[samples]: init u_i (shape={sample_init_u_i.shape}):\n{sample_init_u_i}\n")
        msg.info(f"[samples]: init q_j (shape={sample_init_q_j.shape}):\n{sample_init_q_j}\n")
        msg.info(f"[samples]: init dq_j (shape={sample_init_dq_j.shape}):\n{sample_init_dq_j}\n")
        msg.info(f"[samples]: init lambda_j (shape={sample_init_lambda_j.shape}):\n{sample_init_lambda_j}\n")
        msg.info(f"[samples]: next q_i (shape={sample_next_q_i.shape}):\n{sample_next_q_i}\n")
        msg.info(f"[samples]: next u_i (shape={sample_next_u_i.shape}):\n{sample_next_u_i}\n")
        msg.info(f"[samples]: next q_j (shape={sample_next_q_j.shape}):\n{sample_next_q_j}\n")
        msg.info(f"[samples]: next dq_j (shape={sample_next_dq_j.shape}):\n{sample_next_dq_j}\n")
        msg.info(f"[samples]: control tau_j (shape={sample_ctrl_tau_j.shape}):\n{sample_ctrl_tau_j}\n")

        # Create a multi-instance system by replicating the single-instance builder
        multi_builder = make_homogeneous_builder(num_worlds=num_sample_steps, build_fn=build_cartpole, ground=False)
        for i, body in enumerate(multi_builder.bodies):
            msg.info(f"[multi]: [builder]: body {i}: bid: {body.bid}")
            msg.info(f"[multi]: [builder]: body {i}: q_i: {body.q_i_0}")
            msg.info(f"[multi]: [builder]: body {i}: u_i: {body.u_i_0}")

        # Create simulator and check if the initial state is consistent with the contents of the builder
        multi_sim = Simulator(builder=multi_builder, device=self.default_device)
        self.assertEqual(multi_sim.model.size.sum_of_num_bodies, 2 * num_sample_steps)
        self.assertEqual(multi_sim.model.size.sum_of_num_joints, 2 * num_sample_steps)
        for i, b in enumerate(multi_builder.bodies):
            np.testing.assert_allclose(multi_sim.model.bodies.q_i_0.numpy()[i], b.q_i_0)
            np.testing.assert_allclose(multi_sim.model.bodies.u_i_0.numpy()[i], b.u_i_0)
            np.testing.assert_allclose(multi_sim.state_previous.q_i.numpy()[i], b.q_i_0)
            np.testing.assert_allclose(multi_sim.state_previous.u_i.numpy()[i], b.u_i_0)
            np.testing.assert_allclose(multi_sim.state.q_i.numpy()[i], b.q_i_0)
            np.testing.assert_allclose(multi_sim.state.u_i.numpy()[i], b.u_i_0)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [start]: sim.model.size:\n{multi_sim.model.size}\n\n")
        msg.info(f"[multi]: [start]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [start]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [start]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [start]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")
        msg.info(f"[multi]: [start]: sim.model.control.tau_j:\n{multi_sim.control.tau_j}\n\n")

        # Create a state & control containers to hold the sampled initial states
        state_0 = multi_sim.model.state()
        state_0.q_i.assign(sample_init_q_i)
        state_0.u_i.assign(sample_init_u_i)
        state_0.q_j.assign(sample_init_q_j)
        state_0.q_j_p.assign(sample_init_q_j)
        state_0.dq_j.assign(sample_init_dq_j)
        state_0.lambda_j.assign(sample_init_lambda_j)
        control_0 = multi_sim.model.control()
        control_0.tau_j.assign(sample_ctrl_tau_j)

        # Reset the multi-instance simulator to load the new initial states
        multi_sim.data.state_n.copy_from(state_0)
        multi_sim.data.state_p.copy_from(state_0)
        multi_sim.data.control.copy_from(control_0)
        msg.info(f"[multi]: [reset]: sim.model.state_previous.q_i:\n{multi_sim.state_previous.q_i}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state_previous.u_i:\n{multi_sim.state_previous.u_i}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state_previous.q_j:\n{multi_sim.state_previous.q_j}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state_previous.dq_j:\n{multi_sim.state_previous.dq_j}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [reset]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")
        np.testing.assert_allclose(multi_sim.state_previous.q_i.numpy(), sample_init_q_i)
        np.testing.assert_allclose(multi_sim.state_previous.u_i.numpy(), sample_init_u_i)
        np.testing.assert_allclose(multi_sim.state.q_i.numpy(), sample_init_q_i)
        np.testing.assert_allclose(multi_sim.state.u_i.numpy(), sample_init_u_i)
        np.testing.assert_allclose(multi_sim.state_previous.q_j.numpy(), sample_init_q_j)
        np.testing.assert_allclose(multi_sim.state_previous.dq_j.numpy(), sample_init_dq_j)
        np.testing.assert_allclose(multi_sim.state.q_j.numpy(), sample_init_q_j)
        np.testing.assert_allclose(multi_sim.state.dq_j.numpy(), sample_init_dq_j)

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [init]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")
        msg.info(f"[multi]: [init]: sim.model.control.tau_j:\n{multi_sim.control.tau_j}\n\n")

        # Step the multi-instance simulator once
        multi_sim.step()
        wp.synchronize()

        # Optional verbose output - enabled globally via self.verbose
        msg.info(f"[multi]: [next]: sim.model.state.q_i:\n{multi_sim.state.q_i}\n\n")
        msg.info(f"[multi]: [next]: sim.model.state.u_i:\n{multi_sim.state.u_i}\n\n")
        msg.info(f"[multi]: [next]: sim.model.state.q_j:\n{multi_sim.state.q_j}\n\n")
        msg.info(f"[multi]: [next]: sim.model.state.dq_j:\n{multi_sim.state.dq_j}\n\n")

        # Check that the next states match the collected samples
        np.testing.assert_allclose(multi_sim.solver.data.joints.tau_j.numpy(), sample_ctrl_tau_j)
        np.testing.assert_allclose(multi_sim.state_previous.q_i.numpy(), sample_init_q_i)
        np.testing.assert_allclose(multi_sim.state_previous.u_i.numpy(), sample_init_u_i)
        np.testing.assert_allclose(multi_sim.state.q_i.numpy(), sample_next_q_i)
        np.testing.assert_allclose(multi_sim.state.u_i.numpy(), sample_next_u_i)
        np.testing.assert_allclose(multi_sim.state_previous.q_j.numpy(), sample_init_q_j)
        np.testing.assert_allclose(multi_sim.state_previous.dq_j.numpy(), sample_init_dq_j)
        np.testing.assert_allclose(multi_sim.state.q_j.numpy(), sample_next_q_j)
        np.testing.assert_allclose(multi_sim.state.dq_j.numpy(), sample_next_dq_j)
        np.testing.assert_allclose(multi_sim.control.tau_j.numpy(), sample_ctrl_tau_j)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
