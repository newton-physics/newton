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

import numpy as np
import warp as wp
from warp.context import Devicelike

import newton
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.shapes import ShapeType
from newton._src.solvers.kamino.core.types import float32
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.geometry import CollisionPipelineType
from newton._src.solvers.kamino.geometry.primitive.broadphase import PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES
from newton._src.solvers.kamino.geometry.primitive.narrowphase import PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS
from newton._src.solvers.kamino.models import builders
from newton._src.solvers.kamino.simulation.simulator import Simulator, SimulatorSettings
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.viewer import ViewerKamino

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


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
    # Set world index
    wid = int(0)

    # Define the time window for the active external force profile
    t_start = float32(1.0)
    t_end = float32(3.0)

    # Get the current time
    t = state_t[wid]

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        control_tau_j[0] = 1.0
    else:
        control_tau_j[0] = 0.0


###
# Launchers
###


def test_control_callback(sim: Simulator):
    """
    A control callback function
    """
    wp.launch(
        _test_control_callback,
        dim=1,
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
        max_steps: int = 1000,
        use_cuda_graph: bool = False,
        pipeline_name: str = "primitive",
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

        # Set collision detection pipeline type based on user input
        if pipeline_name.lower() == "unified":
            cd_pipeline = CollisionPipelineType.UNIFIED
        elif pipeline_name.lower() == "primitive":
            cd_pipeline = CollisionPipelineType.PRIMITIVE
        else:
            raise ValueError(f"Unsupported collision pipeline name: {pipeline_name}")

        # Define excluded shape types for broadphase / narrowphase (temporary)
        excluded_types = [
            ShapeType.EMPTY,  # NOTE: Need to skip empty shapes
            ShapeType.PLANE,  # NOTE: Currently not supported well by the viewer
            ShapeType.ELLIPSOID,  # NOTE: Currently not supported well by the viewer
            ShapeType.MESH,  # NOTE: Currently not supported any pipeline
            ShapeType.CONVEX,  # NOTE: Currently not supported any pipeline
            ShapeType.HFIELD,  # NOTE: Currently not supported any pipeline
            ShapeType.SDF,  # NOTE: Currently not supported any pipeline
        ]

        # Generate a list of all supported shape-pair combinations for the configured pipeline
        supported_shape_pairs: list[tuple[str, str]] = []
        if cd_pipeline == CollisionPipelineType.UNIFIED:
            supported_shape_types = [st.value for st in ShapeType]
            for shape_bottom in supported_shape_types:
                shape_bottom_name = ShapeType(shape_bottom).name.lower()
                for shape_top in supported_shape_types:
                    shape_top_name = ShapeType(shape_top).name.lower()
                    if shape_top in excluded_types or shape_bottom in excluded_types:
                        continue
                    supported_shape_pairs.append((shape_top_name, shape_bottom_name))
        elif cd_pipeline == CollisionPipelineType.PRIMITIVE:
            excluded_types.extend([ShapeType.CYLINDER])
            supported_shape_types = PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES
            supported_type_pairs = PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS
            supported_type_pairs_reversed = [(b, a) for (a, b) in supported_type_pairs]
            supported_type_pairs.extend(supported_type_pairs_reversed)
            for shape_bottom in supported_shape_types:
                shape_bottom_name = shape_bottom.name.lower()
                for shape_top in supported_shape_types:
                    shape_top_name = shape_top.name.lower()
                    if shape_top in excluded_types or shape_bottom in excluded_types:
                        continue
                    if (shape_top, shape_bottom) in supported_type_pairs:
                        supported_shape_pairs.append((shape_top_name, shape_bottom_name))
        else:
            raise ValueError(f"Unsupported collision pipeline type: {cd_pipeline}")
        msg.notif(f"Supported shape pairs for pipeline '{cd_pipeline.name}': {supported_shape_pairs}")

        # Construct model builder containing all shape-pair combinations supported by the configured pipeline
        msg.info("Constructing builder using model generator ...")
        self.builder: ModelBuilder = builders.make_shape_pairs_builder(
            shape_pairs=supported_shape_pairs,
            distance=0.0,
            ground_box=True,
            ground_z=-2.0,
        )

        # Set solver settings
        settings = SimulatorSettings()
        settings.dt = 0.001
        settings.solver.primal_tolerance = 1e-6
        settings.solver.dual_tolerance = 1e-6
        settings.solver.compl_tolerance = 1e-6
        settings.solver.rho_0 = 0.1
        settings.collision_detector.pipeline = cd_pipeline

        # Create a simulator
        msg.info("Building the simulator...")
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
    parser = argparse.ArgumentParser(description="A demo of all supported joint types.")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps for headless mode")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument("--cuda-graph", action="store_true", default=True, help="Use CUDA graphs")
    parser.add_argument("--clear-cache", action="store_true", default=False, help="Clear warp cache")
    parser.add_argument("--test", action="store_true", default=False, help="Run tests")
    parser.add_argument(
        "--pipeline-name",
        type=str,
        choices=["primitive", "unified"],
        default="primitive",
        help="Collision detection pipeline name ('primitive' or 'unified')",
    )
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
        max_steps=args.num_steps,
        headless=args.headless,
        pipeline_name=args.pipeline_name,
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
            camera_pos = wp.vec3(8.7, -26.0, 1.0)
            pitch = 2.0
            yaw = 140.0
            example.viewer.set_camera(camera_pos, pitch, yaw)

        # Launch the example using Newton's built-in runtime
        newton.examples.run(example, args)
