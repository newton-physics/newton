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

import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp
from warp.context import Devicelike

import newton
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.simulation.simulator import Simulator, SimulatorSettings
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.viewer import ViewerKamino

###
# Example class
###


class Example:
    def __init__(
        self,
        usd_model_path: Path,
        device: Devicelike = None,
        use_cuda_graph: bool = False,
        max_steps: int = 1000,
        headless: bool = False,
    ):
        # Initialize target frames per second and corresponding time-steps
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = 0.001
        self.sim_substeps = int(self.frame_dt / self.sim_dt)
        self.max_steps = max_steps

        # Initialize internal time-keeping
        self.sim_time = 0.0
        self.sim_steps = 0

        # Cache the device and other internal flags
        self.device = device
        self.use_cuda_graph: bool = use_cuda_graph

        # Create a model builder from the imported USD
        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        self.builder: ModelBuilder = importer.import_from(source=usd_model_path)
        msg.info("total mass: %f", self.builder.worlds[0].mass_total)
        msg.info("total diag inertia: %f", self.builder.worlds[0].inertia_total)

        # Set gravity
        self.builder.gravity[0].enabled = True

        # Set solver settings
        settings = SimulatorSettings()
        settings.dt = self.sim_dt
        settings.problem.alpha = 0.1
        settings.solver.primal_tolerance = 1e-6
        settings.solver.dual_tolerance = 1e-6
        settings.solver.compl_tolerance = 1e-6
        settings.solver.max_iterations = 200
        settings.solver.rho_0 = 0.05

        # Create a simulator
        msg.notif("Building the simulator...")
        self.sim = Simulator(builder=self.builder, settings=settings, device=device)

        # Initialize the viewer
        if not headless:
            self.viewer = ViewerKamino(
                builder=self.builder,
                simulator=self.sim,
            )
        else:
            self.viewer = None

        # Declare and initialize the optional computation graphs
        # NOTE: These are used for most efficient GPU runtime
        self.reset_graph = None
        self.step_graph = None
        self.simulate_graph = None

        # Capture CUDA graph if requested and available
        self.capture()

        # Warm-start the simulator before rendering
        # NOTE: This compiles and loads the warp kernels prior to execution
        msg.notif("Warming up simulator...")
        self.step_once()
        self.reset()

    def capture(self):
        """Capture CUDA graph if requested and available."""
        if self.use_cuda_graph:
            msg.info("Running with CUDA graphs...")
            with wp.ScopedCapture(self.device) as reset_capture:
                self.sim.reset()
            self.reset_graph = reset_capture.graph
            with wp.ScopedCapture(self.device) as step_capture:
                self.sim.step()
            self.step_graph = step_capture.graph
            with wp.ScopedCapture(self.device) as sim_capture:
                self.simulate()
            self.simulate_graph = sim_capture.graph
        else:
            msg.info("Running with kernels...")

    def simulate(self):
        """Run simulation substeps."""
        for _i in range(self.sim_substeps):
            self.sim.step()
            self.sim_steps += 1

    def reset(self):
        """Reset the simulation."""
        if self.reset_graph:
            wp.capture_launch(self.reset_graph)
        else:
            self.sim.reset()
        self.sim_steps = 0
        self.sim_time = 0.0

    def step_once(self):
        """Run the simulation for a single time-step."""
        if self.step_graph:
            wp.capture_launch(self.step_graph)
        else:
            self.sim.step()
        self.sim_steps += 1
        self.sim_time += self.sim_dt

    def step(self):
        """Step the simulation."""
        if self.simulate_graph:
            wp.capture_launch(self.simulate_graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current frame."""
        if self.viewer:
            self.viewer.render_frame()

    def test(self):
        """Test function for compatibility."""
        pass


###
# Main function
###


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Legs simulation example")
    parser.add_argument("--usd-model-path", type=str, required=True, help="Path to the USD model to simulate")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps for headless mode")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument("--cuda-graph", action="store_true", default=True, help="Use CUDA graphs")
    parser.add_argument("--clear-cache", action="store_true", default=False, help="Clear warp cache")
    parser.add_argument("--test", action="store_true", default=False, help="Run tests")
    args = parser.parse_args()

    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)  # Suppress scientific notation

    # Clear warp cache if requested
    if args.clear_cache:
        wp.clear_kernel_cache()
        wp.clear_lto_cache()

    # TODO: Make optional
    # Set the verbosity of the global message logger
    msg.set_log_level(msg.LogLevel.INFO)

    # Set device if specified, otherwise use Warp's default
    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()

    # Ensure USD model path exists
    usd_model_path = Path(args.usd_model_path)
    if not usd_model_path.exists():
        msg.critical(f"USD model path does not exist: {usd_model_path}")
        sys.exit(1)

    # Determine if CUDA graphs should be used for execution
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    use_cuda_graph = can_use_cuda_graph & args.cuda_graph
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")
    msg.info(f"use_cuda_graph: {use_cuda_graph}")
    msg.info(f"device: {device}")

    # Create example instance
    example = Example(
        device=device,
        usd_model_path=usd_model_path,
        use_cuda_graph=use_cuda_graph,
        max_steps=args.num_steps,
        headless=args.headless,
    )

    # Run a brute-force similation loop if headless
    if args.headless:
        msg.notif("Running in headless mode...")
        run_headless(example, progress=True)

    # Otherwise launch using a debug viewer
    else:
        msg.notif("Running in Viewer mode...")
        # Set initial camera position for better view of the system
        if hasattr(example.viewer, "set_camera"):
            camera_pos = wp.vec3(0.6, 0.6, 0.3)
            pitch = -10.0
            yaw = 225.0
            example.viewer.set_camera(camera_pos, pitch, yaw)

        # Launch the example using Newton's built-in runtime
        newton.examples.run(example, args)
