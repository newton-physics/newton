# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.types import float32, int32, vec6f
from newton._src.solvers.kamino._src.models.builders.basics import build_box_on_plane
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.io.usd import USDImporter
from newton._src.solvers.kamino._src.utils.sim import SimulationLogger, Simulator, ViewerKamino
from newton._src.solvers.kamino.examples import get_examples_output_path, run_headless
from newton.tests import get_kamino_basics_asset

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _control_callback(
    model_body_wid: wp.array[int32],
    contact_world_num_active: wp.array[int32],
    data_t: wp.array[float32],
    state_w_i_e: wp.array[vec6f],
):
    """
    An example control callback kernel.
    """
    # Retrieve the body index from the thread index
    bid = wp.tid()

    # Retrieve world index of the body
    wid = model_body_wid[bid]

    # Retrieve number of active contacts in the world
    wnc = contact_world_num_active[wid]

    # Define the time window for the active external force profile
    t_start = float32(0.0)
    t_end = float32(6.0)

    # Get the current time
    t = data_t[wid]

    # Apply a time-dependent external force
    if t > t_start and t < t_end and wnc > 0:
        m = float32(1.0)  # Mass of the box
        g = float32(9.8067)  # Gravitational acceleration
        mu = float32(0.9)  # Friction coefficient
        f_ext = 1.1 * m * g * mu  # Magnitude of the external force
        state_w_i_e[bid] = vec6f(f_ext, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        state_w_i_e[bid] = vec6f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


###
# Launchers
###


def control_callback(sim: Simulator):
    """
    A control callback function
    """
    wp.launch(
        _control_callback,
        dim=sim.model.size.sum_of_num_bodies,
        inputs=[
            sim.model.bodies.bid,
            sim.contacts.data.world_active_contacts,
            sim.solver.data.time.time,
            sim.state.w_i_e,
        ],
        device=sim._device,
    )


###
# Example class
###


class Example:
    def __init__(
        self,
        device: wp.DeviceLike = None,
        num_worlds: int = 1,
        max_steps: int = 1000,
        use_cuda_graph: bool = False,
        load_from_usd: bool = False,
        gravity: bool = True,
        ground: bool = True,
        logging: bool = False,
        headless: bool = False,
        record_video: bool = False,
        async_save: bool = False,
    ):
        # Initialize target frames per second and corresponding time-steps
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / 0.001))
        self.sim_dt = self.frame_dt / self.sim_substeps
        msg.info(f"Using sim_dt = {self.sim_dt} ({self.sim_substeps} substeps per frame)")
        self.max_steps = max_steps

        # Cache the device and other internal flags
        self.device = device
        self.use_cuda_graph: bool = use_cuda_graph
        self.logging: bool = logging

        # Construct model builder
        if load_from_usd:
            msg.notif("Constructing builder from imported USD ...")
            USD_MODEL_PATH = get_kamino_basics_asset("box_on_plane.usda")
            importer = USDImporter()
            self.builder: ModelBuilderKamino = make_homogeneous_builder(
                num_worlds=num_worlds,
                build_fn=importer.import_from,
                source=USD_MODEL_PATH,
                load_static_geometry=ground,
            )
        else:
            msg.notif("Constructing builder using model generator ...")
            self.builder: ModelBuilderKamino = make_homogeneous_builder(
                num_worlds=num_worlds, build_fn=build_box_on_plane, ground=ground
            )

        # Set gravity
        for w in range(self.builder.num_worlds):
            self.builder.gravity[w].enabled = gravity

        # Set solver config
        config = Simulator.Config()
        config.dt = self.sim_dt
        config.solver.dynamics.preconditioning = True
        config.solver.padmm.primal_tolerance = 1e-6
        config.solver.padmm.dual_tolerance = 1e-6
        config.solver.padmm.compl_tolerance = 1e-6
        config.solver.padmm.max_iterations = 200
        config.solver.padmm.rho_0 = 0.1
        config.solver.padmm.use_acceleration = True
        config.solver.padmm.warmstart_mode = "containers"
        config.solver.padmm.contact_warmstart_method = "key_and_position_with_net_force_backup"
        config.solver.collect_solver_info = True
        config.solver.compute_solution_metrics = logging and not use_cuda_graph

        # Create a simulator
        msg.notif("Building the simulator...")
        self.sim = Simulator(builder=self.builder, config=config, device=device)
        self.sim.set_control_callback(control_callback)

        # Initialize the data logger
        self.logger: SimulationLogger | None = None
        if self.logging:
            msg.notif("Creating the sim data logger...")
            self.logger = SimulationLogger(self.max_steps, self.sim, self.builder)

        # Initialize the 3D viewer
        self.viewer: ViewerKamino | None = None
        if not headless:
            msg.notif("Creating the 3D viewer...")
            # Set up video recording folder
            video_folder = None
            if record_video:
                video_folder = os.path.join(get_examples_output_path(), "box_on_plane/frames")
                os.makedirs(video_folder, exist_ok=True)
                msg.info(f"Frame recording enabled ({'async' if async_save else 'sync'} mode)")
                msg.info(f"Frames will be saved to: {video_folder}")

            self.viewer = ViewerKamino(
                builder=self.builder,
                simulator=self.sim,
                show_contacts=True,
                record_video=record_video,
                video_folder=video_folder,
                async_save=async_save,
            )

        # Declare and initialize the optional computation graphs
        # NOTE: These are used for most efficient GPU runtime
        self.reset_graph = None
        self.step_graph = None
        self.simulate_graph = None

        # Warm-start the simulator before rendering
        # NOTE: This compiles and loads the warp kernels prior to execution
        msg.notif("Warming up simulator...")
        self.step_once()
        self.reset()

        # Capture CUDA graph if requested and available
        self.capture()

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
        # Optionally plot the logged simulation data
        if self.logging:
            self.logger.plot_solver_info(path=path, show=show)
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
    parser = argparse.ArgumentParser(description="Box-on-Plane simulation example")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False, help="Run in headless mode")
    parser.add_argument("--num-worlds", type=int, default=1, help="Number of worlds to simulate in parallel")
    parser.add_argument("--num-steps", type=int, default=2000, help="Number of steps for headless mode")
    parser.add_argument(
        "--load-from-usd", action=argparse.BooleanOptionalAction, default=True, help="Load model from USD file"
    )
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
    use_cuda_graph = can_use_cuda_graph and args.cuda_graph
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
            camera_pos = wp.vec3(2.0, 2.0, 0.5)
            pitch = -5.0
            yaw = 180.0 + 45.0
            example.viewer.set_camera(camera_pos, pitch, yaw)

        # Launch the example using Newton's built-in runtime
        newton.examples.run(example, args)

    # Plot logged data after the viewer is closed
    if args.logging or args.record:
        OUTPUT_PLOT_PATH = os.path.join(get_examples_output_path(), "box_on_plane")
        os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
        example.plot(path=OUTPUT_PLOT_PATH, show=args.show_plots)
