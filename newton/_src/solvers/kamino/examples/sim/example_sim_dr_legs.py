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

import numpy as np
import warp as wp
from warp.context import Devicelike

import newton
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.examples import get_examples_output_path, run_headless
from newton._src.solvers.kamino.linalg.linear import SolverShorthand as LinearSolverShorthand
from newton._src.solvers.kamino.models import get_examples_usd_assets_path
from newton._src.solvers.kamino.models.builders.utils import (
    add_ground_box,
    make_homogeneous_builder,
    set_uniform_body_pose_offset,
)
from newton._src.solvers.kamino.simulation.simulator import Simulator, SimulatorSettings
from newton._src.solvers.kamino.solvers.padmm import PADMMWarmStartMode
from newton._src.solvers.kamino.solvers.warmstart import WarmstarterContacts
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.control import AnimationJointReference, JointSpacePIDController
from newton._src.solvers.kamino.utils.datalog import SimulationLogger
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.viewer import ViewerKamino

###
# Example class
###


class Example:
    def __init__(
        self,
        device: Devicelike = None,
        num_worlds: int = 1,
        max_steps: int = 1000,
        use_cuda_graph: bool = False,
        gravity: bool = True,
        ground: bool = True,
        logging: bool = False,
        linear_solver: str = "LLTB",
        linear_solver_maxiter: int = 0,
        headless: bool = False,
        record_video: bool = False,
        async_save: bool = False,
    ):
        # Initialize target frames per second and corresponding time-steps
        self.fps = 60
        self.sim_dt = 0.001
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.max_steps = max_steps

        # Cache the device and other internal flags
        self.device = device
        self.use_cuda_graph: bool = use_cuda_graph
        self.logging: bool = logging

        # Set the path to the external USD assets
        EXAMPLE_ASSETS_PATH = get_examples_usd_assets_path()
        if EXAMPLE_ASSETS_PATH is None:
            raise FileNotFoundError("Failed to find USD assets path for examples: ensure `newton-assets` is installed.")
        USD_MODEL_PATH = os.path.join(EXAMPLE_ASSETS_PATH, "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda")

        # Create a model builder from the imported USD
        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        self.builder: ModelBuilder = make_homogeneous_builder(
            num_worlds=num_worlds, build_fn=importer.import_from, load_static_geometry=True, source=USD_MODEL_PATH
        )
        msg.info("total mass: %f", self.builder.worlds[0].mass_total)
        msg.info("total diag inertia: %f", self.builder.worlds[0].inertia_total)

        # Offset the model to place it above the ground
        # NOTE: The USD model is centered at the origin
        offset = wp.transformf(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0)
        set_uniform_body_pose_offset(builder=self.builder, offset=offset)

        # Add a static collision layer and geometry for the plane
        if ground:
            for w in range(num_worlds):
                add_ground_box(self.builder, world_index=w, layer="world")

        # Set gravity
        for w in range(self.builder.num_worlds):
            self.builder.gravity[w].enabled = gravity

        # Set solver settings
        settings = SimulatorSettings()
        settings.dt = self.sim_dt
        settings.problem.alpha = 0.1
        settings.solver.primal_tolerance = 1e-4
        settings.solver.dual_tolerance = 1e-4
        settings.solver.compl_tolerance = 1e-4
        settings.solver.max_iterations = 200
        settings.solver.eta = 1e-5
        settings.solver.rho_0 = 0.05
        settings.use_solver_acceleration = True
        settings.warmstart = PADMMWarmStartMode.CONTAINERS
        settings.contact_warmstart_method = WarmstarterContacts.Method.GEOM_PAIR_NET_FORCE
        settings.collect_solver_info = False
        settings.compute_metrics = logging and not use_cuda_graph
        linear_solver_cls = {v: k for k, v in LinearSolverShorthand.items()}[linear_solver.upper()]
        settings.linear_solver_type = linear_solver_cls
        settings.linear_solver_maxiter = linear_solver_maxiter

        # Create a simulator
        msg.notif("Building the simulator...")
        self.sim = Simulator(builder=self.builder, settings=settings, device=device)

        # Load animation data for dr_legs
        NUMPY_ANIMATION_PATH = os.path.join(EXAMPLE_ASSETS_PATH, "dr_legs/animation/dr_legs_animation_100fps.npy")
        animation_np = np.load(NUMPY_ANIMATION_PATH, allow_pickle=True)
        msg.debug("animation_np (shape={%s}):\n{%s}\n", animation_np.shape, animation_np)

        # Compute animation time step and rate
        animation_dt = 0.01  # 100 fps
        animation_rate = round(animation_dt / settings.dt)
        msg.info(f"animation_dt: {animation_dt}")
        msg.info(f"animation_rate: {animation_rate}")

        # Create a joint-space animation reference generator
        self.animation = AnimationJointReference(
            model=self.sim.model,
            data=animation_np,
            data_dt=animation_dt,
            target_dt=settings.dt,
            decimation=1,
            rate=1,
            loop=False,
            use_fd=True,
            device=device,
        )

        # Create a joint-space PID controller
        njaq = self.sim.model.size.sum_of_num_actuated_joint_dofs
        K_p = 80.0 * np.ones(njaq, dtype=np.float32)
        K_d = 0.1 * np.ones(njaq, dtype=np.float32)
        K_i = 0.01 * np.ones(njaq, dtype=np.float32)
        decimation = 1 * np.ones(self.sim.model.size.num_worlds, dtype=np.int32)
        self.controller = JointSpacePIDController(
            model=self.sim.model, K_p=K_p, K_i=K_i, K_d=K_d, decimation=decimation, device=device
        )

        # Define a callback function to reset the controller
        def reset_jointspace_pid_control_callback(simulator: Simulator):
            self.controller.reset(model=simulator.model, state=simulator.data.state_n)
            self.animation.reset(q_j_ref_out=self.controller.data.q_j_ref, dq_j_ref_out=self.controller.data.dq_j_ref)

        # Define a callback function to wrap the execution of the controller
        def compute_jointspace_pid_control_callback(simulator: Simulator):
            self.animation.step(
                time=simulator.data.solver.time,
                q_j_ref_out=self.controller.data.q_j_ref,
                dq_j_ref_out=self.controller.data.dq_j_ref,
            )
            self.controller.compute(
                model=simulator.model,
                state=simulator.data.state_n,
                time=simulator.data.solver.time,
                control=simulator.data.control_n,
            )

        # Set the reference tracking generation & control callbacks into the simulator
        self.sim.set_post_reset_callback(reset_jointspace_pid_control_callback)
        self.sim.set_control_callback(compute_jointspace_pid_control_callback)

        # Initialize the data logger
        self.logger: SimulationLogger | None = None
        if self.logging:
            msg.notif("Creating the sim data logger...")
            self.logger = SimulationLogger(self.max_steps, self.sim, self.builder, self.controller)

        # Initialize the 3D viewer
        self.viewer: ViewerKamino | None = None
        if not headless:
            msg.notif("Creating the 3D viewer...")
            # Set up video recording folder
            video_folder = None
            if record_video:
                video_folder = os.path.join(get_examples_output_path(), "dr_legs/frames")
                os.makedirs(video_folder, exist_ok=True)
                msg.info(f"Frame recording enabled ({'async' if async_save else 'sync'} mode)")
                msg.info(f"Frames will be saved to: {video_folder}")

            self.viewer = ViewerKamino(
                builder=self.builder,
                simulator=self.sim,
                record_video=record_video,
                video_folder=video_folder,
                async_save=async_save,
            )

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
            if not self.use_cuda_graph and self.logging:
                self.logger.log()

    def reset(self):
        """Reset the simulation."""
        if self.reset_graph:
            wp.capture_launch(self.reset_graph)
        else:
            self.sim.reset()
        if not self.use_cuda_graph and self.logging:
            self.logger.reset()
            self.logger.log()

    def step_once(self):
        """Run the simulation for a single time-step."""
        if self.step_graph:
            wp.capture_launch(self.step_graph)
        else:
            self.sim.step()
        if not self.use_cuda_graph and self.logging:
            self.logger.log()

    def step(self):
        """Step the simulation."""
        if self.simulate_graph:
            wp.capture_launch(self.simulate_graph)
        else:
            self.simulate()

    def render(self):
        """Render the current frame."""
        if self.viewer:
            self.viewer.render_frame()

    def test(self):
        """Test function for compatibility."""
        pass

    def plot(self, path: str | None = None, show: bool = False, keep_frames: bool = False):
        """
        Plot logged data and generate video from recorded frames.

        Args:
            path: Output directory path (uses video_folder if None)
            show: If True, display plots after saving
            keep_frames: If True, keep PNG frames after video creation
        """
        # Plot the animation sequence references
        animation_path = os.path.join(path, "animation_references.png") if path is not None else None
        self.animation.plot(path=animation_path, show=show)

        # Optionally plot the logged simulation data
        if self.logging:
            self.logger.plot_solver_info(path=path, show=show)
            self.logger.plot_joint_tracking(path=path, show=show)
            self.logger.plot_solution_metrics(path=path, show=show)

        # Optionally generate video from recorded frames
        if self.viewer is not None and self.viewer._record_video:
            output_dir = path if path is not None else self.viewer._video_folder
            output_path = os.path.join(output_dir, "recording.mp4")
            self.viewer.generate_video(output_filename=output_path, fps=self.fps, keep_frames=keep_frames)


###
# Main function
###


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Legs simulation example")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False, help="Run in headless mode")
    parser.add_argument("--num-worlds", type=int, default=1, help="Number of worlds to simulate in parallel")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps for headless mode")
    parser.add_argument(
        "--gravity", action=argparse.BooleanOptionalAction, default=True, help="Enables gravity in the simulation"
    )
    parser.add_argument(
        "--ground", action=argparse.BooleanOptionalAction, default=True, help="Adds a ground plane to the simulation"
    )
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA graphs")
    parser.add_argument("--clear-cache", action=argparse.BooleanOptionalAction, default=False, help="Clear warp cache")
    parser.add_argument(
        "--logging", action=argparse.BooleanOptionalAction, default=True, help="Enable logging of simulation data"
    )
    parser.add_argument(
        "--show-plots", action=argparse.BooleanOptionalAction, default=False, help="Show plots of logging data"
    )
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=False, help="Run tests")
    parser.add_argument(
        "--record",
        type=str,
        choices=["sync", "async"],
        default=None,
        help="Enable frame recording: 'sync' for synchronous, 'async' for asynchronous (non-blocking)",
    )
    parser.add_argument(
        "--linear-solver",
        default="LLTB",
        choices=LinearSolverShorthand.values(),
        type=str.upper,
        help="Linear solver to use",
    )
    parser.add_argument(
        "--linear-solver-maxiter", default=0, type=int, help="Max number of iterations for iterative linear solvers"
    )
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

    # Determine if CUDA graphs should be used for execution
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    use_cuda_graph = can_use_cuda_graph and args.cuda_graph
    msg.info(f"can_use_cuda_graph: {can_use_cuda_graph}")
    msg.info(f"use_cuda_graph: {use_cuda_graph}")
    msg.info(f"device: {device}")

    # Create example instance
    example = Example(
        device=device,
        use_cuda_graph=use_cuda_graph,
        num_worlds=args.num_worlds,
        linear_solver=args.linear_solver,
        linear_solver_maxiter=args.linear_solver_maxiter,
        max_steps=args.num_steps,
        gravity=args.gravity,
        ground=args.ground,
        headless=args.headless,
        logging=args.logging,
        record_video=args.record is not None and not args.headless,
        async_save=args.record == "async",
    )

    # Run a brute-force simulation loop if headless
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

    # Plot logged data after the viewer is closed
    if args.logging or args.record:
        OUTPUT_PLOT_PATH = os.path.join(get_examples_output_path(), "dr_legs")
        os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
        example.plot(path=OUTPUT_PLOT_PATH, show=args.show_plots)
