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

import newton
import newton.examples
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.types import float32
from newton._src.solvers.kamino._src.models import get_basics_usd_assets_path
from newton._src.solvers.kamino._src.models.builders.basics import build_boxes_fourbar
from newton._src.solvers.kamino._src.models.builders.utils import (
    make_homogeneous_builder,
    set_uniform_body_pose_offset,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.io.usd import USDImporter
from newton._src.solvers.kamino._src.utils.sim import SimulationLogger, Simulator, ViewerKamino
from newton._src.solvers.kamino.examples import get_examples_output_path, run_headless

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _pd_control_callback(
    state_t: wp.array(dtype=float32),
    control_q_j_ref: wp.array(dtype=float32),
    control_dq_j_ref: wp.array(dtype=float32),
    control_tau_j_ref: wp.array(dtype=float32),
):
    """
    An example control callback kernel.
    """
    # Set world index
    wid = int(0)
    jid = int(0)

    # Define the time window for the active external force profile
    t_start = float32(3.0)
    t_window = float32(3.0)
    t_0 = t_start + t_window
    t_1 = t_0 + t_window
    t_2 = t_1 + t_window
    t_3 = t_2 + t_window
    t_4 = t_3 + t_window
    t_5 = t_4 + t_window

    # Get the current time
    t = state_t[wid]

    # Apply a time-dependent joint references
    if t > t_start and t < t_0:
        control_q_j_ref[jid] = 0.1
        control_dq_j_ref[jid] = 0.0
        control_tau_j_ref[jid] = 0.0
    elif t > t_0 and t < t_1:
        control_q_j_ref[jid] = -0.1
        control_dq_j_ref[jid] = 0.0
        control_tau_j_ref[jid] = 0.0
    elif t > t_1 and t < t_2:
        control_q_j_ref[jid] = 0.2
        control_dq_j_ref[jid] = 0.0
        control_tau_j_ref[jid] = 0.0
    elif t > t_2 and t < t_3:
        control_q_j_ref[jid] = -0.2
        control_dq_j_ref[jid] = 0.0
        control_tau_j_ref[jid] = 0.0
    elif t > t_3 and t < t_4:
        control_q_j_ref[jid] = 0.3
        control_dq_j_ref[jid] = 0.0
        control_tau_j_ref[jid] = 0.0
    elif t > t_4 and t < t_5:
        control_q_j_ref[jid] = -0.3
        control_dq_j_ref[jid] = 0.0
        control_tau_j_ref[jid] = 0.0
    else:
        control_q_j_ref[jid] = 0.0
        control_dq_j_ref[jid] = 0.0
        control_tau_j_ref[jid] = 0.0


@wp.kernel
def _torque_control_callback(
    state_t: wp.array(dtype=float32),
    control_tau_j: wp.array(dtype=float32),
):
    """
    An example control callback kernel.
    """
    # Set world index
    wid = int(0)
    jid = int(0)

    # Define the time window for the active external force profile
    t_start = float32(2.0)
    t_end = float32(2.5)

    # Get the current time
    t = state_t[wid]

    # Apply a time-dependent external force
    if t > t_start and t < t_end:
        control_tau_j[jid] = 0.1
    else:
        control_tau_j[jid] = 0.0


###
# Launchers
###


def pd_control_callback(sim: Simulator):
    """
    A control callback function
    """
    wp.launch(
        _pd_control_callback,
        dim=1,
        inputs=[
            sim.solver.data.time.time,
            sim.control.q_j_ref,
            sim.control.dq_j_ref,
            sim.control.tau_j_ref,
        ],
    )


def torque_control_callback(sim: Simulator):
    """
    A control callback function
    """
    wp.launch(
        _torque_control_callback,
        dim=1,
        inputs=[
            sim.solver.data.time.time,
            sim.control.tau_j,
        ],
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
        implicit_pd: bool = True,
        gravity: bool = True,
        ground: bool = True,
        logging: bool = False,
        headless: bool = False,
        record_video: bool = False,
        async_save: bool = False,
    ):
        # Initialize target frames per second and corresponding time-steps
        self.fps = 60
        self.sim_dt = 0.01 if implicit_pd else 0.001
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.max_steps = max_steps

        # Cache the device and other internal flags
        self.device = device
        self.use_cuda_graph: bool = use_cuda_graph
        self.logging: bool = logging

        # Construct model builder
        if load_from_usd:
            msg.notif("Constructing builder from imported USD ...")
            USD_MODEL_PATH = os.path.join(get_basics_usd_assets_path(), "boxes_fourbar.usda")
            importer = USDImporter()
            self.builder: ModelBuilderKamino = make_homogeneous_builder(
                num_worlds=num_worlds,
                build_fn=importer.import_from,
                source=USD_MODEL_PATH,
                load_drive_dynamics=implicit_pd,
                load_static_geometry=ground,
                use_angular_drive_scaling=True,
            )
            # Set joint armature and damping because the purely
            # UsdPhysics schema does not support these properties yet
            if implicit_pd:
                for joint in self.builder.joints:
                    if joint.is_dynamic or joint.is_implicit_pd:
                        joint.a_j = [0.1]
                        joint.b_j = [0.001]
        else:
            msg.notif("Constructing builder using model generator ...")
            self.builder: ModelBuilderKamino = make_homogeneous_builder(
                num_worlds=num_worlds,
                build_fn=build_boxes_fourbar,
                ground=ground,
                limits=True,
                dynamic_joints=implicit_pd,
                implicit_pd=implicit_pd,
            )

        # Offset the model to place it above the ground
        # NOTE: The USD model is centered at the origin
        offset = wp.transformf(0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0)
        set_uniform_body_pose_offset(builder=self.builder, offset=offset)

        # Set gravity
        for w in range(self.builder.num_worlds):
            self.builder.gravity[w].enabled = gravity

        # Set solver config
        config = Simulator.Config()
        config.dt = self.sim_dt
        config.solver.constraints.gamma = 0.1
        config.solver.sparse_jacobian = False
        config.solver.sparse_dynamics = False
        config.solver.integrator = "euler"  # Select from {"euler", "moreau"}
        config.solver.dynamics.preconditioning = True
        config.solver.padmm.primal_tolerance = 1e-4
        config.solver.padmm.dual_tolerance = 1e-4
        config.solver.padmm.compl_tolerance = 1e-4
        config.solver.padmm.max_iterations = 200
        config.solver.padmm.rho_0 = 0.1
        config.solver.padmm.use_acceleration = True
        config.solver.padmm.warmstart_mode = "containers"
        config.solver.padmm.contact_warmstart_method = "geom_pair_net_force"
        config.solver.collect_solver_info = False
        config.solver.compute_solution_metrics = logging and not use_cuda_graph

        # Create a simulator
        msg.notif("Building the simulator...")
        self.sim = Simulator(builder=self.builder, config=config, device=device)

        # Set the control callback based on whether implicit PD control is enabled
        if implicit_pd:
            self.sim.set_control_callback(pd_control_callback)
        else:
            self.sim.set_control_callback(torque_control_callback)

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
                video_folder = os.path.join(get_examples_output_path(), "boxes_fourbar/frames")
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
    parser = argparse.ArgumentParser(description="Boxes-Fourbar simulation example")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False, help="Run in headless mode")
    parser.add_argument("--num-worlds", type=int, default=1, help="Number of worlds to simulate in parallel")
    parser.add_argument("--num-steps", type=int, default=3000, help="Number of steps for headless mode")
    parser.add_argument(
        "--load-from-usd", action=argparse.BooleanOptionalAction, default=True, help="Load model from USD file"
    )
    parser.add_argument(
        "--implicit-pd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enables implicit PD control of joints",
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
        implicit_pd=args.implicit_pd,
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
            camera_pos = wp.vec3(-0.2, -0.5, 0.1)
            pitch = -5.0
            yaw = 70.0
            example.viewer.set_camera(camera_pos, pitch, yaw)

        # Launch the example using Newton's built-in runtime
        newton.examples.run(example, args)

    # Plot logged data after the viewer is closed
    if args.logging or args.record:
        OUTPUT_PLOT_PATH = os.path.join(get_examples_output_path(), "boxes_fourbar")
        os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
        example.plot(path=OUTPUT_PLOT_PATH, show=args.show_plots)
