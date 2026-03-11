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

import time

import warp as wp

from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.control import JointSpacePIDController
from newton._src.solvers.kamino._src.utils.sim import SimulationLogger, Simulator, ViewerKamino
from newton._src.solvers.kamino.examples import print_progress_bar

###
# Interfaces
###


class SimulationRunner:
    def __init__(
        self,
        builder: ModelBuilderKamino,
        simulator: Simulator,
        controller: JointSpacePIDController | None = None,
        device: wp.DeviceLike = None,
        max_steps: int = 0,
        max_time: float = 0.0,
        viewer_fps: float = 60.0,
        sim_dt: float = 0.001,
        gravity: bool = True,
        ground: bool = True,
        logging: bool = False,
        headless: bool = False,
        record_video: bool = False,
        async_save: bool = False,
    ):
        """TODO"""

        # Initialize target frames per second and corresponding time-steps
        self.fps = viewer_fps
        self.sim_dt = sim_dt
        self.frame_dt = 1.0 / self.fps
        self.substeps = max(1, round(self.frame_dt / self.sim_dt))

        # Cache the device and other internal flags
        self.device = device
        self.use_cuda_graph: bool = False
        self.logging: bool = logging

        # TODO
        self.builder = builder
        self.sim = simulator
        self.ctrl = controller

        # Initialize the data logger
        self.logger: SimulationLogger | None = None
        if self.logging:
            msg.notif("Creating the sim data logger...")
            self.logger = SimulationLogger(max_steps, self.builder, self.sim, self.ctrl)

        # Initialize the 3D viewer
        self.viewer: ViewerKamino | None = None
        if not headless:
            self.viewer = ViewerKamino(
                builder=self.builder,
                simulator=self.sim,
                record_video=record_video,
                async_save=async_save,
            )

        # Declare and initialize the optional computation graphs
        # NOTE: These are used for most efficient GPU runtime
        self.reset_graph = None
        self.step_graph = None

        # Capture CUDA graph if requested and available
        self._capture()

        # Warm-start the simulator before rendering
        # NOTE: This compiles and loads the warp kernels prior to execution
        msg.notif("Warming up simulation...")
        self.step()
        self.reset()

    ###
    # Simulation API
    ###

    def reset(self):
        """TODO"""
        if self.reset_graph:
            wp.capture_launch(self.reset_graph)
        else:
            self._run_reset()
        if self.logging:
            self.logger.reset()
            self.logger.log()

    def step(self):
        """TODO"""
        if self.step_graph:
            wp.capture_launch(self.step_graph)
        else:
            self._run_step()
        if self.logging:
            self.logger.log()

    def render(self):
        """TODO"""
        if self.viewer:
            self.viewer.render_frame()

    def test(self):
        """TODO"""
        pass

    def export(self, path: str | None = None, show: bool = False, keep_frames: bool = False):
        """TODO"""
        pass

    def run(self, num_frames: int = 0, progress: bool = True):
        """TODO"""
        msg.notif(f"Running for {num_frames} frames...")
        start_time = time.time()
        for i in range(num_frames):
            self.step()
            wp.synchronize()
            if progress:
                print_progress_bar(i + 1, num_frames, start_time, prefix="Progress", suffix="")

    ###
    # Internals
    ###

    def _capture(self):
        """Capture CUDA graph if requested and available."""
        if self.use_cuda_graph:
            msg.info("Running with CUDA graphs...")
            with wp.ScopedCapture(self.device) as reset_capture:
                self._run_reset()
            self.reset_graph = reset_capture.graph
            with wp.ScopedCapture(self.device) as step_capture:
                self._run_step()
            self.step_graph = step_capture.graph
        else:
            msg.info("Running with kernels...")

    def _run_reset(self):
        """TODO"""
        self.sim.reset()

    def _run_step(self):
        """TODO"""
        for _i in range(self.substeps):
            self.sim.step()
