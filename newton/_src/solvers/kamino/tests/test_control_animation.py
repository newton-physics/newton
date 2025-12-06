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

"""Unit tests for the AnimationJointReference class."""

import os
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino.control.animation import AnimationJointReference
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.io.usd import USDImporter

###
# Tests
###


class TestAnimationJointReference(unittest.TestCase):
    def setUp(self):
        # Configs
        self.seed = 42
        self.default_device = wp.get_device()
        self.verbose = False  # Set to True for verbose output

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.DEBUG)
        else:
            msg.set_log_level(msg.LogLevel.WARNING)

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_default(self):
        animation = AnimationJointReference()
        self.assertIsNotNone(animation)
        self.assertEqual(animation.device, None)
        self.assertEqual(animation._data, None)

    def test_01_make_with_numpy_data(self):
        EXAMPLES_USD_ASSETS_PATH = get_examples_usd_assets_path()
        if EXAMPLES_USD_ASSETS_PATH is None:
            self.skipTest("EXAMPLES_USD_ASSETS_PATH is `None` - skipping `DR Legs` import test.")

        # Set paths to model and animation data
        USD_MODEL_PATH = os.path.join(EXAMPLES_USD_ASSETS_PATH, "dr_legs/usd/dr_legs_with_boxes.usda")
        NUMPY_ANIMATION_PATH = os.path.join(EXAMPLES_USD_ASSETS_PATH, "dr_legs/animation/dr_legs_animation_100fps.npy")

        # Import USD model of DR Legs
        importer = USDImporter()
        builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH)
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)

        # Retrieve the number of actuated coordinates and DoFs
        njaq = model.size.sum_of_num_actuated_joint_coords
        njad = model.size.sum_of_num_actuated_joint_dofs
        msg.info(f"number of actuated joint coordinates: {njaq}")
        msg.info(f"number of actuated joint DoFs: {njad}")
        self.assertEqual(njaq, njad)  # Ensure only 1-DoF joints

        # Load numpy animation data
        animation_np = np.load(NUMPY_ANIMATION_PATH, allow_pickle=True)
        msg.info(f"animation_np (shape={animation_np.shape}):\n{animation_np}\n")
        self.assertEqual(animation_np.shape[1], njaq)  # Ensure data matches number of joints

        # Set animation parameters
        animation_dt = 0.01  # 100 Hz/FPS
        sim_dt = animation_dt  # 100 Hz/FPS
        decimation: int = 1  # No decimation, step through every frame
        rate: int = 1  # No rate (i.e. skip), step through every frame
        loop: bool = True
        use_fd: bool = False

        # Create a joint-space animation reference generator
        animation = AnimationJointReference(
            model=model,
            data=animation_np,
            data_dt=animation_dt,
            target_dt=sim_dt,
            decimation=decimation,
            rate=rate,
            loop=loop,
            use_fd=use_fd,
            device=self.default_device,
        )
        self.assertIsNotNone(animation)
        self.assertIsNotNone(animation.data)
        self.assertEqual(animation.device, self.default_device)
        self.assertEqual(animation.sequence_length, animation_np.shape[0])
        self.assertEqual(animation.data.q_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.dq_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.loop.shape, (1,))
        self.assertEqual(animation.data.loop.numpy()[0], 1 if loop else 0)
        self.assertEqual(animation.data.rate.shape, (1,))
        self.assertEqual(animation.data.rate.numpy()[0], rate)
        self.assertEqual(animation.data.frame.shape, (1,))
        self.assertEqual(animation.data.frame.numpy()[0], 0)

        # Check that the internal numpy arrays match the input data
        np.testing.assert_array_almost_equal(animation.data.q_j_ref.numpy(), animation_np, decimal=6)
        np.testing.assert_array_almost_equal(animation.data.dq_j_ref.numpy(), np.zeros_like(animation_np), decimal=6)

        # Allocate output arrays for joint references
        q_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)
        dq_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)

        # Retrieve the reference at the initial step (0)
        animation.reset(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
        np.testing.assert_array_equal(animation.data.frame.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[0, :], decimal=6)
        np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Initialize simulation time steps
        data.time.steps.fill_(0)

        # Step through the animation and verify outputs
        num_steps = 10
        for step in range(1, num_steps + 1):
            data.time.steps.fill_(step)
            animation.step(time=data.time, q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
            expected_step = (rate * step) % animation.sequence_length  # Loop around if exceeding number of frames
            np.testing.assert_array_equal(animation.data.frame.numpy(), np.array([expected_step], dtype=np.int32))
            np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[expected_step, :], decimal=6)
            np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Reset the reference at the initial step (0)
        animation.reset(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
        np.testing.assert_array_equal(animation.data.frame.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[0, :], decimal=6)
        np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Initialize simulation time steps
        data.time.steps.fill_(0)

        # Step through again but exceeding the number of frames to test looping
        num_steps = animation.sequence_length + 5
        for step in range(1, num_steps + 1):
            data.time.steps.fill_(step)
            animation.step(time=data.time, q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
            expected_step = (rate * step) % animation.sequence_length  # Loop around if exceeding number of frames
            np.testing.assert_array_equal(animation.data.frame.numpy(), np.array([expected_step], dtype=np.int32))
            np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[expected_step, :], decimal=6)
            np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

    def test_02_make_with_numpy_data_and_decimation(self):
        EXAMPLES_USD_ASSETS_PATH = get_examples_usd_assets_path()
        if EXAMPLES_USD_ASSETS_PATH is None:
            self.skipTest("EXAMPLES_USD_ASSETS_PATH is `None` - skipping `DR Legs` import test.")

        # Set paths to model and animation data
        USD_MODEL_PATH = os.path.join(EXAMPLES_USD_ASSETS_PATH, "dr_legs/usd/dr_legs_with_boxes.usda")
        NUMPY_ANIMATION_PATH = os.path.join(EXAMPLES_USD_ASSETS_PATH, "dr_legs/animation/dr_legs_animation_100fps.npy")

        # Import USD model of DR Legs
        importer = USDImporter()
        builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH)
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)

        # Retrieve the number of actuated coordinates and DoFs
        njaq = model.size.sum_of_num_actuated_joint_coords
        njad = model.size.sum_of_num_actuated_joint_dofs
        msg.info(f"number of actuated joint coordinates: {njaq}")
        msg.info(f"number of actuated joint DoFs: {njad}")
        self.assertEqual(njaq, njad)  # Ensure only 1-DoF joints

        # Load numpy animation data
        animation_np = np.load(NUMPY_ANIMATION_PATH, allow_pickle=True)
        msg.info(f"animation_np (shape={animation_np.shape}):\n{animation_np}\n")
        self.assertEqual(animation_np.shape[1], njaq)  # Ensure data matches number of joints

        # Set animation parameters
        animation_dt = 0.01  # 100 Hz/FPS
        sim_dt = animation_dt  # 100 Hz/FPS
        decimation: int = 15  # Advance frame index every 15th step
        rate: int = 1  # No rate (i.e. skip), step through every frame
        loop: bool = True
        use_fd: bool = False

        # Create a joint-space animation reference generator
        animation = AnimationJointReference(
            model=model,
            data=animation_np,
            data_dt=animation_dt,
            target_dt=sim_dt,
            decimation=decimation,
            rate=rate,
            loop=loop,
            use_fd=use_fd,
            device=self.default_device,
        )
        self.assertIsNotNone(animation)
        self.assertIsNotNone(animation.data)
        self.assertEqual(animation.device, self.default_device)
        self.assertEqual(animation.sequence_length, animation_np.shape[0])
        self.assertEqual(animation.data.q_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.dq_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.length.shape, (1,))
        self.assertEqual(animation.data.length.numpy()[0], animation_np.shape[0])
        self.assertEqual(animation.data.decimation.shape, (1,))
        self.assertEqual(animation.data.decimation.numpy()[0], decimation)
        self.assertEqual(animation.data.rate.shape, (1,))
        self.assertEqual(animation.data.rate.numpy()[0], rate)
        self.assertEqual(animation.data.loop.shape, (1,))
        self.assertEqual(animation.data.loop.numpy()[0], 1 if loop else 0)
        self.assertEqual(animation.data.frame.shape, (1,))
        self.assertEqual(animation.data.frame.numpy()[0], 0)

        # Check that the internal numpy arrays match the input data
        np.testing.assert_array_almost_equal(animation.data.q_j_ref.numpy(), animation_np, decimal=6)
        np.testing.assert_array_almost_equal(animation.data.dq_j_ref.numpy(), np.zeros_like(animation_np), decimal=6)

        # Allocate output arrays for joint references
        q_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)
        dq_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)

        # Retrieve the reference at the initial step (0)
        animation.reset(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
        np.testing.assert_array_equal(animation.data.frame.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[0, :], decimal=6)
        np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Initialize simulation time steps
        data.time.steps.fill_(0)

        # Step through the animation and verify outputs
        num_steps = 3 * decimation + 2  # Step through multiple decimation cycles
        for s in range(1, num_steps + 1):
            # Increment the global simulation step array
            data.time.steps.fill_(s)
            step = data.time.steps.numpy()[0]
            msg.info(f"[s={s}]: step index: {step}")
            self.assertEqual(step, s)

            # Step the animation
            # NOTE: In actual uses-cases this will be called on every sim step
            # and we only want to update the frame index every `decimation` steps
            animation.step(time=data.time, q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)

            # Retrieve the actual frame index
            frame = animation.data.frame.numpy()[0]
            msg.info(f"[s={s}]: frame index: {frame}")
            # Compute the expected frame index based on decimation
            expected = (step // decimation) % animation.sequence_length
            msg.info(f"[s={s}]: expected index: {expected}")
            # Check expected vs actual frame index and outputs
            self.assertEqual(frame, expected)

            # Check output references match expected frame
            np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[expected, :], decimal=6)
            np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Reset the reference at the initial step (0)
        animation.reset(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
        np.testing.assert_array_equal(animation.data.frame.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[0, :], decimal=6)
        np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Initialize simulation time steps
        data.time.steps.fill_(0)

        # Step through again but exceeding the number of frames to test looping
        num_steps = animation.sequence_length + 5
        for s in range(1, num_steps + 1):
            # Increment the global simulation step array
            data.time.steps.fill_(s)
            step = data.time.steps.numpy()[0]
            msg.info(f"[s={s}]: step index: {step}")
            self.assertEqual(step, s)

            # Step the animation
            # NOTE: In actual uses-cases this will be called on every sim step
            # and we only want to update the frame index every `decimation` steps
            animation.step(time=data.time, q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)

            # Retrieve the actual frame index
            frame = animation.data.frame.numpy()[0]
            msg.info(f"[s={s}]: frame index: {frame}")
            # Compute the expected frame index based on decimation
            expected = (step // decimation) % animation.sequence_length
            msg.info(f"[s={s}]: expected index: {expected}")
            # Check expected vs actual frame index and outputs
            self.assertEqual(frame, expected)

            # Check output references match expected frame
            np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[expected, :], decimal=6)
            np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

    def test_03_make_with_numpy_data_and_decimation_plus_rate(self):
        EXAMPLES_USD_ASSETS_PATH = get_examples_usd_assets_path()
        if EXAMPLES_USD_ASSETS_PATH is None:
            self.skipTest("EXAMPLES_USD_ASSETS_PATH is `None` - skipping `DR Legs` import test.")
        # Set paths to DR Legs model and animation data
        USD_MODEL_PATH = os.path.join(EXAMPLES_USD_ASSETS_PATH, "dr_legs/usd/dr_legs_with_boxes.usda")
        NUMPY_ANIMATION_PATH = os.path.join(EXAMPLES_USD_ASSETS_PATH, "dr_legs/animation/dr_legs_animation_100fps.npy")

        # Import USD model of DR Legs
        importer = USDImporter()
        builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH)
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)

        # Retrieve the number of actuated coordinates and DoFs
        njaq = model.size.sum_of_num_actuated_joint_coords
        njad = model.size.sum_of_num_actuated_joint_dofs
        msg.info(f"number of actuated joint coordinates: {njaq}")
        msg.info(f"number of actuated joint DoFs: {njad}")
        self.assertEqual(njaq, njad)  # Ensure only 1-DoF joints

        # Load numpy animation data
        animation_np = np.load(NUMPY_ANIMATION_PATH, allow_pickle=True)
        msg.info(f"animation_np (shape={animation_np.shape}):\n{animation_np}\n")
        self.assertEqual(animation_np.shape[1], njaq)  # Ensure data matches number of joints

        # Set animation parameters
        animation_dt = 0.01  # 100 Hz/FPS
        sim_dt = animation_dt  # 100 Hz/FPS
        decimation: int = 15  # Advance frame index every 15th step
        rate: int = 10  # No rate (i.e. skip), step through every frame
        loop: bool = True
        use_fd: bool = False

        # Create a joint-space animation reference generator
        animation = AnimationJointReference(
            model=model,
            data=animation_np,
            data_dt=animation_dt,
            target_dt=sim_dt,
            decimation=decimation,
            rate=rate,
            loop=loop,
            use_fd=use_fd,
            device=self.default_device,
        )
        self.assertIsNotNone(animation)
        self.assertIsNotNone(animation.data)
        self.assertEqual(animation.device, self.default_device)
        self.assertEqual(animation.sequence_length, animation_np.shape[0])
        self.assertEqual(animation.data.q_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.dq_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.length.shape, (1,))
        self.assertEqual(animation.data.length.numpy()[0], animation_np.shape[0])
        self.assertEqual(animation.data.decimation.shape, (1,))
        self.assertEqual(animation.data.decimation.numpy()[0], decimation)
        self.assertEqual(animation.data.rate.shape, (1,))
        self.assertEqual(animation.data.rate.numpy()[0], rate)
        self.assertEqual(animation.data.loop.shape, (1,))
        self.assertEqual(animation.data.loop.numpy()[0], 1 if loop else 0)
        self.assertEqual(animation.data.frame.shape, (1,))
        self.assertEqual(animation.data.frame.numpy()[0], 0)

        # Check that the internal numpy arrays match the input data
        np.testing.assert_array_almost_equal(animation.data.q_j_ref.numpy(), animation_np, decimal=6)
        np.testing.assert_array_almost_equal(animation.data.dq_j_ref.numpy(), np.zeros_like(animation_np), decimal=6)

        # Allocate output arrays for joint references
        q_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)
        dq_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)

        # Retrieve the reference at the initial step (0)
        animation.reset(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
        np.testing.assert_array_equal(animation.data.frame.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[0, :], decimal=6)
        np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Initialize simulation time steps
        data.time.steps.fill_(0)

        # Step through the animation and verify outputs
        num_steps = 3 * decimation + 2  # Step through multiple decimation cycles
        for s in range(1, num_steps + 1):
            # Increment the global simulation step array
            data.time.steps.fill_(s)
            step = data.time.steps.numpy()[0]
            msg.info(f"[s={s}]: step index: {step}")
            self.assertEqual(step, s)

            # Step the animation
            # NOTE: In actual uses-cases this will be called on every sim step
            # and we only want to update the frame index every `decimation` steps
            animation.step(time=data.time, q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)

            # Retrieve the actual frame index
            frame = animation.data.frame.numpy()[0]
            msg.info(f"[s={s}]: frame index: {frame}")
            # Compute the expected frame index based on decimation
            expected = ((step // decimation) * rate) % animation.sequence_length
            msg.info(f"[s={s}]: expected index: {expected}")
            # Check expected vs actual frame index and outputs
            self.assertEqual(frame, expected)

            # Check output references match expected frame
            np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[expected, :], decimal=6)
            np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Reset the reference at the initial step (0)
        animation.reset(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
        np.testing.assert_array_equal(animation.data.frame.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[0, :], decimal=6)
        np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Initialize simulation time steps
        data.time.steps.fill_(0)

        # Step through again but exceeding the number of frames to test looping
        num_steps = animation.sequence_length + 5
        for s in range(1, num_steps + 1):
            # Increment the global simulation step array
            data.time.steps.fill_(s)
            step = data.time.steps.numpy()[0]
            msg.info(f"[s={s}]: step index: {step}")
            self.assertEqual(step, s)

            # Step the animation
            # NOTE: In actual uses-cases this will be called on every sim step
            # and we only want to update the frame index every `decimation` steps
            animation.step(time=data.time, q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)

            # Retrieve the actual frame index
            frame = animation.data.frame.numpy()[0]
            msg.info(f"[s={s}]: frame index: {frame}")
            # Compute the expected frame index based on decimation
            expected = ((step // decimation) * rate) % animation.sequence_length
            msg.info(f"[s={s}]: expected index: {expected}")
            # Check expected vs actual frame index and outputs
            self.assertEqual(frame, expected)

            # Check output references match expected frame
            np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[expected, :], decimal=6)
            np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

    def test_04_make_with_numpy_data_and_decimation_plus_rate_no_looping(self):
        EXAMPLES_USD_ASSETS_PATH = get_examples_usd_assets_path()
        if EXAMPLES_USD_ASSETS_PATH is None:
            self.skipTest("EXAMPLES_USD_ASSETS_PATH is `None` - skipping `DR Legs` import test.")
        # Set paths to DR Legs model and animation data
        USD_MODEL_PATH = os.path.join(EXAMPLES_USD_ASSETS_PATH, "dr_legs/usd/dr_legs_with_boxes.usda")
        NUMPY_ANIMATION_PATH = os.path.join(EXAMPLES_USD_ASSETS_PATH, "dr_legs/animation/dr_legs_animation_100fps.npy")

        # Import USD model of DR Legs
        importer = USDImporter()
        builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH)
        model = builder.finalize(device=self.default_device)
        data = model.data(device=self.default_device)

        # Retrieve the number of actuated coordinates and DoFs
        njaq = model.size.sum_of_num_actuated_joint_coords
        njad = model.size.sum_of_num_actuated_joint_dofs
        msg.info(f"number of actuated joint coordinates: {njaq}")
        msg.info(f"number of actuated joint DoFs: {njad}")
        self.assertEqual(njaq, njad)  # Ensure only 1-DoF joints

        # Load numpy animation data
        animation_np = np.load(NUMPY_ANIMATION_PATH, allow_pickle=True)
        msg.info(f"animation_np (shape={animation_np.shape}):\n{animation_np}\n")
        self.assertEqual(animation_np.shape[1], njaq)  # Ensure data matches number of joints

        # Set animation parameters
        animation_dt = 0.01  # 100 Hz/FPS
        sim_dt = animation_dt  # 100 Hz/FPS
        decimation: int = 15  # Advance frame index every 15th step
        rate: int = 10  # No rate (i.e. skip), step through every frame
        loop: bool = False
        use_fd: bool = False

        # Create a joint-space animation reference generator
        animation = AnimationJointReference(
            model=model,
            data=animation_np,
            data_dt=animation_dt,
            target_dt=sim_dt,
            decimation=decimation,
            rate=rate,
            loop=loop,
            use_fd=use_fd,
            device=self.default_device,
        )
        self.assertIsNotNone(animation)
        self.assertIsNotNone(animation.data)
        self.assertEqual(animation.device, self.default_device)
        self.assertEqual(animation.sequence_length, animation_np.shape[0])
        self.assertEqual(animation.data.q_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.dq_j_ref.shape, animation_np.shape)
        self.assertEqual(animation.data.length.shape, (1,))
        self.assertEqual(animation.data.length.numpy()[0], animation_np.shape[0])
        self.assertEqual(animation.data.decimation.shape, (1,))
        self.assertEqual(animation.data.decimation.numpy()[0], decimation)
        self.assertEqual(animation.data.rate.shape, (1,))
        self.assertEqual(animation.data.rate.numpy()[0], rate)
        self.assertEqual(animation.data.loop.shape, (1,))
        self.assertEqual(animation.data.loop.numpy()[0], 1 if loop else 0)
        self.assertEqual(animation.data.frame.shape, (1,))
        self.assertEqual(animation.data.frame.numpy()[0], 0)

        # Check that the internal numpy arrays match the input data
        np.testing.assert_array_almost_equal(animation.data.q_j_ref.numpy(), animation_np, decimal=6)
        np.testing.assert_array_almost_equal(animation.data.dq_j_ref.numpy(), np.zeros_like(animation_np), decimal=6)

        # Allocate output arrays for joint references
        q_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)
        dq_j_ref_out = wp.zeros(njad, dtype=float32, device=self.default_device)

        # Reset the reference at the initial step (0)
        animation.reset(q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)
        np.testing.assert_array_equal(animation.data.frame.numpy(), np.array([0], dtype=np.int32))
        np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[0, :], decimal=6)
        np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)

        # Initialize simulation time steps
        data.time.steps.fill_(0)

        # Step through again but exceeding the number of frames to test looping
        num_steps = animation.sequence_length + 5
        for s in range(1, num_steps + 1):
            # Increment the global simulation step array
            data.time.steps.fill_(s)
            step = data.time.steps.numpy()[0]
            msg.info(f"[s={s}]: step index: {step}")
            self.assertEqual(step, s)

            # Step the animation
            # NOTE: In actual uses-cases this will be called on every sim step
            # and we only want to update the frame index every `decimation` steps
            animation.step(time=data.time, q_j_ref_out=q_j_ref_out, dq_j_ref_out=dq_j_ref_out)

            # Retrieve the actual frame index
            frame = animation.data.frame.numpy()[0]
            msg.info(f"[s={s}]: frame index: {frame}")
            # Compute the expected frame index based on decimation
            expected = (step // decimation) * rate
            msg.info(f"[s={s}]: expected index: {expected}")
            # Check expected vs actual frame index and outputs
            if expected >= animation.sequence_length:
                expected = animation.sequence_length - 1  # Clamp to last frame if exceeding length
            self.assertEqual(frame, expected)

            # Check output references match expected frame
            np.testing.assert_array_almost_equal(q_j_ref_out.numpy(), animation_np[expected, :], decimal=6)
            np.testing.assert_array_almost_equal(dq_j_ref_out.numpy(), np.zeros(njad, dtype=np.float32), decimal=6)


###
# Test execution
###

if __name__ == "__main__":
    # Global numpy configurations
    np.set_printoptions(linewidth=500, precision=10, suppress=True)  # Suppress scientific notation

    # Global warp configurations
    wp.config.verbose = False
    wp.clear_kernel_cache()
    wp.clear_lto_cache()

    # Run all tests
    unittest.main(verbosity=2)
