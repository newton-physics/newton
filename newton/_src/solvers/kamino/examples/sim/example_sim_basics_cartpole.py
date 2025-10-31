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
import os
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from warp.context import Devicelike

import newton
import newton._src.solvers.kamino.utils.logger as msg
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.math import TWO_PI
from newton._src.solvers.kamino.core.types import float32, uint32
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.models import get_basics_usd_assets_path
from newton._src.solvers.kamino.models.builders import build_cartpole
from newton._src.solvers.kamino.models.utils import make_homogeneous_builder
from newton._src.solvers.kamino.simulation.simulator import Simulator, SimulatorSettings
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.viewer import ViewerKamino

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# RL Interfaces
###


@dataclass
class CartpoleStates:
    q_j: torch.Tensor | None = None
    dq_j: torch.Tensor | None = None


@dataclass
class CartpoleActions:
    tau_j: torch.Tensor | None = None


###
# Kernels
###


@wp.kernel
def _test_control_callback(
    state_t: wp.array(dtype=float32),
    control_tau_j: wp.array(dtype=float32),
):
    """
    An example control callback kernel.
    """
    # Retrieve the world index from the thread ID
    wid = wp.tid()

    # Define the time window for the active external force profile
    t_start = float32(1.0)
    t_end = float32(1.1)

    # Get the current time
    t = state_t[wid]

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        control_tau_j[wid * 2 + 0] = 0.1 * wp.sin(1.0 * TWO_PI * (t - t_start)) * wp.randf(uint32(wid), -1.0, 1.0)
        control_tau_j[wid * 2 + 1] = 0.1 * wp.sin(1.0 * TWO_PI * (t - t_start)) * wp.randf(uint32(wid), -1.0, 1.0)
    else:
        control_tau_j[wid * 2 + 0] = 0.0
        control_tau_j[wid * 2 + 1] = 0.0


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
            sim.data.solver.time.time,
            sim.data.control_n.tau_j,
        ],
    )


###
# Example class
###


class Example:
    def __init__(
        self,
        device: Devicelike,
        num_worlds: int,
        max_steps: int = 1000,
        use_cuda_graph: bool = False,
        load_from_usd: bool = False,
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

        # Construct model builder
        if load_from_usd:
            msg.info("Constructing builder from imported USD ...")
            USD_MODEL_PATH = os.path.join(get_basics_usd_assets_path(), "cartpole.usda")
            importer = USDImporter()
            self.builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH, load_static_geometry=True)
        else:
            msg.info("Constructing builder using model generator ...")
            self.builder: ModelBuilder = make_homogeneous_builder(num_worlds=num_worlds, build_func=build_cartpole)
        msg.warning(f"self.builder.bodies: {self.builder.bodies}")
        msg.warning(f"self.builder.joints: {self.builder.joints}")

        # Set gravity
        self.builder.gravity.enabled = True

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
        msg.info("Building the simulator...")
        self.sim = Simulator(builder=self.builder, settings=settings, device=device)
        self.sim.set_control_callback(test_control_callback)

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
        msg.info("Warming up simulator...")
        self.step_once()
        self.reset()

        # Declare a PyTorch data interface for the current state and controls data
        self.states: CartpoleStates | None = None
        self.actions: CartpoleActions | None = None

        # Initialize RL interfaces
        self.make_rl_interface()

    def make_rl_interface(self):
        """
        Constructs data interfaces for batched MDP states and actions.

        Notes:
        - Each torch.Tensor wraps the underlying kamino simulator data arrays without copying.
        """
        # Retrieve the batched system dimensions
        num_worlds = self.sim.model.size.num_worlds
        num_joint_dofs = self.sim.model.size.max_of_num_joint_dofs

        # Construct state and action tensors wrapping the underlying simulator data
        self.states = CartpoleStates(
            q_j=wp.to_torch(self.sim.data.state_n.q_j).reshape(num_worlds, num_joint_dofs),
            dq_j=wp.to_torch(self.sim.data.state_n.dq_j).reshape(num_worlds, num_joint_dofs),
        )
        self.actions = CartpoleActions(
            tau_j=wp.to_torch(self.sim.data.control_n.tau_j).reshape(num_worlds, num_joint_dofs),
        )

    def capture(self):
        """Capture CUDA graph if requested and available."""
        if self.use_cuda_graph:
            msg.info("Running with CUDA graphs...")
            with wp.ScopedCapture(device=self.device) as reset_capture:
                self.sim.reset()
            self.reset_graph = reset_capture.graph
            with wp.ScopedCapture(device=self.device) as step_capture:
                self.sim.step()
            self.step_graph = step_capture.graph
            with wp.ScopedCapture(device=self.device) as sim_capture:
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
    parser = argparse.ArgumentParser(description="Cartpole simulation example")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
    parser.add_argument("--num-worlds", type=int, default=3, help="Number of worlds to simulate in parallel")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps for headless mode")
    parser.add_argument("--load-from-usd", action="store_true", default=True, help="Load model from USD file")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument("--cuda-graph", action="store_true", default=True, help="Use CUDA graphs")
    parser.add_argument("--clear-cache", action="store_true", default=False, help="Clear warp cache")
    parser.add_argument("--test", action="store_true", default=False, help="Run tests")
    args = parser.parse_args()

    # Set global numpy configurations
    np.set_printoptions(linewidth=20000, precision=10, threshold=10000, suppress=True)

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

    # Determine if CUDA graphs should be used for execution
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    use_cuda_graph = can_use_cuda_graph & args.cuda_graph
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")
    msg.info(f"use_cuda_graph: {use_cuda_graph}")
    msg.info(f"device: {device}")

    # Create example instance
    example = Example(
        device=device,
        use_cuda_graph=use_cuda_graph,
        load_from_usd=args.load_from_usd,
        num_worlds=args.num_worlds,
        max_steps=args.num_steps,
        headless=args.headless,
    )

    # Run a brute-force similation loop if headless
    if args.headless:
        msg.info("Running in headless mode...")
        run_headless(example, progress=True)

    # Otherwise launch using a debug viewer
    else:
        msg.info("Running in Viewer mode...")
        # Set initial camera position for better view of the system
        if hasattr(example.viewer, "set_camera"):
            camera_pos = wp.vec3(2.0, 2.0, 0.3)
            pitch = -10.0
            yaw = 205.0
            example.viewer.set_camera(camera_pos, pitch, yaw)

        # Launch the example using Newton's built-in runtime
        newton.examples.run(example, args)
