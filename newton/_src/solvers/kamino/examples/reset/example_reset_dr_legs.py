# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation  # noqa: TID253

import newton
import newton.examples
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.types import float32, int32, transformf, vec6f
from newton._src.solvers.kamino._src.models.builders.utils import (
    make_homogeneous_builder,
    set_uniform_body_pose_offset,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.io.usd import USDImporter
from newton._src.solvers.kamino._src.utils.sim import ViewerKamino
from newton._src.solvers.kamino._src.utils.sim.simulator import Simulator
from newton._src.solvers.kamino.examples import get_examples_output_path, run_headless

###
# Kernels
###


@wp.kernel
def _test_control_callback(
    sim_has_started_resets: wp.array(dtype=wp.bool),
    sim_reset_index: wp.array(dtype=wp.int32),
    actuated_joint_idx: wp.array(dtype=int32),
    state_t: wp.array(dtype=float32),
    control_tau_j: wp.array(dtype=float32),
):
    """
    An example control callback kernel.
    """
    # Skip if no joint is selected for actuation
    if not sim_has_started_resets[0]:
        return

    # Hack to handle negative reset index
    joint_reset_index = sim_reset_index[0]
    if joint_reset_index < 0:
        joint_reset_index = actuated_joint_idx.size - 1

    # Define the time window for the active external force profile
    t_start = float32(0.0)
    t_end = float32(0.5)

    # Get the current time
    t = state_t[0]

    # Ad-hoc torque magnitude based on the selected joint
    # because we want higher actuation for the two hip joints
    if joint_reset_index == 0 or joint_reset_index == 6:
        torque = 0.01
    else:
        torque = 0.001

    # Reverse torque direction for the first leg
    if joint_reset_index < 6:
        torque = -torque

    # Apply a time-dependent external force
    if t >= t_start and t < t_end:
        control_tau_j[actuated_joint_idx[joint_reset_index]] = torque
    else:
        control_tau_j[actuated_joint_idx[joint_reset_index]] = 0.0


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
        logging: bool = False,
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

        # Define internal counters
        self.sim_steps = 0
        self.sim_reset_mode = 0

        # Cache the device and other internal flags
        self.device = device
        self.use_cuda_graph: bool = use_cuda_graph
        self.logging: bool = logging

        # Load the DR Legs USD and add it to the builder
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_meshes_and_boxes.usda")

        # Create a model builder from the imported USD
        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        self.builder: ModelBuilderKamino = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=importer.import_from,
            load_drive_dynamics=False,
            load_static_geometry=True,
            source=asset_file,
        )
        msg.info("total mass: %f", self.builder.worlds[0].mass_total)
        msg.info("total diag inertia: %f", self.builder.worlds[0].inertia_total)

        # Offset the model to place it above the ground
        # NOTE: The USD model is centered at the origin
        q_base = wp.transformf((0.0, 0.0, 0.265), wp.quat_identity(dtype=float32))
        set_uniform_body_pose_offset(builder=self.builder, offset=q_base)

        # Set gravity
        for w in range(self.builder.num_worlds):
            self.builder.gravity[w].enabled = False

        # Set solver config
        config = Simulator.Config()
        config.dt = self.sim_dt
        config.solver.constraints.alpha = 0.1
        config.solver.padmm.primal_tolerance = 1e-4
        config.solver.padmm.dual_tolerance = 1e-4
        config.solver.padmm.compl_tolerance = 1e-4
        config.solver.padmm.max_iterations = 100
        config.solver.padmm.eta = 1e-5
        config.solver.padmm.rho_0 = 0.02
        config.solver.padmm.rho_min = 0.01
        config.solver.padmm.use_acceleration = True
        config.solver.padmm.warmstart_mode = "containers"
        config.solver.padmm.contact_warmstart_method = "geom_pair_net_force"
        config.solver.collect_solver_info = False
        config.solver.compute_solution_metrics = False
        config.solver.use_fk_solver = True

        # Create a simulator
        msg.notif("Building the simulator...")
        self.sim = Simulator(builder=self.builder, config=config, device=device)

        # Create a list of actuated joint indices from the model and builder
        self.actuated_joint_idx_np = np.zeros(shape=(self.sim.model.size.sum_of_num_actuated_joints,), dtype=np.int32)
        jidx = 0
        for j, joint in enumerate(self.builder.joints):
            if joint.is_actuated:
                self.actuated_joint_idx_np[jidx] = j
                jidx += 1
        msg.warning("actuated_joint_idx_np: %s", self.actuated_joint_idx_np)
        msg.warning("actuated_joint_names:\n%s", self.builder.worlds[0].actuated_joint_names)

        # Allocate utility arrays for resetting
        with wp.ScopedDevice(self.device):
            self.base_q = wp.zeros(shape=(self.sim.model.size.num_worlds,), dtype=transformf)
            self.base_u = wp.zeros(shape=(self.sim.model.size.num_worlds,), dtype=vec6f)
            self.joint_q = wp.zeros(shape=(self.sim.model.size.sum_of_num_joint_coords,), dtype=float32)
            self.joint_u = wp.zeros(shape=(self.sim.model.size.sum_of_num_joint_dofs,), dtype=float32)
            self.actuator_q = wp.zeros(shape=(self.sim.model.size.sum_of_num_actuated_joint_coords,), dtype=float32)
            self.actuator_u = wp.zeros(shape=(self.sim.model.size.sum_of_num_actuated_joint_dofs,), dtype=float32)
            self.actuated_joint_idx = wp.array(self.actuated_joint_idx_np, dtype=int32)
            self.sim_has_started_resets = wp.full(shape=(1,), dtype=wp.bool, value=False)
            self.sim_reset_index = wp.full(shape=(1,), dtype=wp.int32, value=-1)

        # Define the control callback function that will actuate a single joint
        def test_control_callback(sim: Simulator):
            wp.launch(
                _test_control_callback,
                dim=1,
                inputs=[
                    self.sim_has_started_resets,
                    self.sim_reset_index,
                    self.actuated_joint_idx,
                    sim.solver.data.time.time,
                    sim.data.control.tau_j,
                ],
            )

        # Set the test control callback into the simulator
        self.sim.set_control_callback(test_control_callback)

        # Initialize the 3D viewer
        self.viewer: ViewerKamino | None = None
        if not headless:
            msg.notif("Creating the 3D viewer...")
            # Set up video recording folder
            video_folder = None
            if record_video:
                video_folder = os.path.join(get_examples_output_path(), "reset_dr_legs/frames")
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
            self.sim_steps += 1

    def reset(self):
        """Reset the simulation."""
        if self.reset_graph:
            wp.capture_launch(self.reset_graph)
        else:
            self.sim.reset()

    def step_once(self):
        """Run the simulation for a single time-step."""
        if self.step_graph:
            wp.capture_launch(self.step_graph)
        else:
            self.sim.step()

    def update_reset_config(self):
        """Update the reset configuration based on the current reset index and mode."""
        self.sim_steps = 0
        self.sim.data.control.tau_j.zero_()
        self.sim_has_started_resets.fill_(True)
        joint_reset_index = self.sim_reset_index.numpy()[0]
        num_actuated_joints = len(self.actuated_joint_idx_np)
        joint_reset_index = (joint_reset_index + 1) % num_actuated_joints
        # If all joints have been cycled through, proceed to the next reset mode
        if joint_reset_index == num_actuated_joints - 1:
            self.sim_reset_mode = (self.sim_reset_mode + 1) % 6
            joint_reset_index = -1
        self.sim_reset_index.fill_(joint_reset_index)
        msg.warning(f"Next joint_reset_index: {joint_reset_index}")
        msg.warning(f"Next sim_reset_mode: {self.sim_reset_mode}")

    def step(self):
        """Step the simulation."""
        if self.simulate_graph:
            wp.capture_launch(self.simulate_graph)
            self.sim_steps += self.sim_substeps
        else:
            self.simulate()

        # Demo of resetting to the default state defined in the model
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 0:
            msg.notif("Resetting to default model state...")
            self.update_reset_config()
            self.sim.reset()

        # Demo of resetting only the base pose
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 1:
            msg.notif("Resetting with base pose...")
            self.update_reset_config()
            R_b = Rotation.from_rotvec(np.pi / 4 * np.array([0, 0, 1]))
            q_b = R_b.as_quat()  # x, y, z, w
            q_base = wp.transformf((0.1, 0.1, 0.3), q_b)
            self.base_q.assign([q_base] * self.sim.model.size.num_worlds)
            self.sim.reset(base_q=self.base_q)

        # Demo of resetting the base pose and twist
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 2:
            msg.notif("Resetting with base pose and twist...")
            self.update_reset_config()
            R_b = Rotation.from_rotvec(np.pi / 4 * np.array([0, 0, 1]))
            q_b = R_b.as_quat()  # x, y, z, w
            q_base = wp.transformf((0.1, 0.1, 0.3), q_b)
            u_base = vec6f(0.0, 0.0, 0.05, 0.0, 0.0, 0.3)
            self.base_q.assign([q_base] * self.sim.model.size.num_worlds)
            self.base_u.assign([u_base] * self.sim.model.size.num_worlds)
            self.sim.reset(base_q=self.base_q, base_u=self.base_u)

        # Demo of resetting the base state and joint configurations
        # NOTE: This will invoke the FK solver to update body poses
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 3:
            msg.notif("Resetting with base pose and joint configurations...")
            self.update_reset_config()
            R_b = Rotation.from_rotvec(np.pi / 4 * np.array([0, 0, 1]))
            q_b = R_b.as_quat()  # x, y, z, w
            q_base = wp.transformf((0.1, 0.1, 0.3), q_b)
            u_base = vec6f(0.0, 0.0, 0.05, 0.0, 0.0, -0.3)
            self.base_q.assign([q_base] * self.sim.model.size.num_worlds)
            self.base_u.assign([u_base] * self.sim.model.size.num_worlds)
            joint_q_np = np.zeros(self.sim.model.size.sum_of_num_joint_coords, dtype=np.float32)
            self.joint_q.assign(joint_q_np)
            self.sim.reset(base_q=self.base_q, base_u=self.base_u, joint_q=self.joint_q, joint_u=self.joint_u)

        # Demo of resetting the base state and joint configurations to specific poses
        # NOTE: This will invoke the FK solver to update body poses
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 4:
            msg.notif("Resetting with base pose and specific joint configurations...")
            self.update_reset_config()
            joint_reset_index = self.sim_reset_index.numpy()[0]
            msg.warning(f"Resetting joint {self.actuated_joint_idx_np[joint_reset_index]}...")
            R_b = Rotation.from_rotvec(np.pi / 4 * np.array([0, 0, 1]))
            q_b = R_b.as_quat()  # x, y, z, w
            q_base = wp.transformf((0.1, 0.1, 0.3), q_b)
            u_base = vec6f(0.0, 0.0, -0.05, 0.0, 0.0, 0.3)
            self.base_q.assign([q_base] * self.sim.model.size.num_worlds)
            self.base_u.assign([u_base] * self.sim.model.size.num_worlds)
            actuated_joint_config = np.array(
                [
                    np.pi / 12,
                    np.pi / 12,
                    np.pi / 12,
                    np.pi / 12,
                    np.pi / 12,
                    np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                ],
                dtype=np.float32,
            )
            joint_q_np = np.zeros(self.sim.model.size.sum_of_num_joint_coords, dtype=np.float32)
            joint_q_np[self.actuated_joint_idx_np[joint_reset_index]] = actuated_joint_config[joint_reset_index]
            self.joint_q.assign(joint_q_np)
            self.sim.reset(base_q=self.base_q, base_u=self.base_u, joint_q=self.joint_q, joint_u=self.joint_u)

        # Demo of resetting the base state and joint configurations to specific poses
        # NOTE: This will invoke the FK solver to update body poses
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 5:
            msg.notif("Resetting with base pose and specific actuator configurations...")
            self.update_reset_config()
            joint_reset_index = self.sim_reset_index.numpy()[0]
            msg.warning(f"Resetting joint {self.actuated_joint_idx_np[joint_reset_index]}...")
            R_b = Rotation.from_rotvec(np.pi / 4 * np.array([0, 0, 1]))
            q_b = R_b.as_quat()  # x, y, z, w
            q_base = wp.transformf((0.1, 0.1, 0.3), q_b)
            u_base = vec6f(0.0, 0.0, -0.05, 0.0, 0.0, -0.3)
            self.base_q.assign([q_base] * self.sim.model.size.num_worlds)
            self.base_u.assign([u_base] * self.sim.model.size.num_worlds)
            actuated_joint_config = np.array(
                [
                    np.pi / 12,
                    np.pi / 12,
                    np.pi / 12,
                    np.pi / 12,
                    np.pi / 12,
                    np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                    -np.pi / 12,
                ],
                dtype=np.float32,
            )
            actuator_q_np = np.zeros(self.sim.model.size.sum_of_num_actuated_joint_coords, dtype=np.float32)
            actuator_q_np[joint_reset_index] = actuated_joint_config[joint_reset_index]
            self.actuator_q.assign(actuator_q_np)
            self.sim.reset(
                base_q=self.base_q, base_u=self.base_u, actuator_q=self.actuator_q, actuator_u=self.actuator_u
            )

    def render(self):
        """Render the current frame."""
        if self.viewer:
            self.viewer.render_frame()

    def test(self):
        """Test function for compatibility."""
        pass

    def plot(self, path: str | None = None, keep_frames: bool = False):
        """
        Plot logged data and generate video from recorded frames.

        Args:
            path: Output directory path (uses video_folder if None)
            keep_frames: If True, keep PNG frames after video creation
        """
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
    parser.add_argument("--num-steps", type=int, default=400, help="Number of steps for headless mode")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA graphs")
    parser.add_argument("--clear-cache", action=argparse.BooleanOptionalAction, default=False, help="Clear warp cache")
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
        max_steps=args.num_steps,
        headless=args.headless,
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
    if args.record:
        OUTPUT_PLOT_PATH = os.path.join(get_examples_output_path(), "reset_dr_legs")
        os.makedirs(OUTPUT_PLOT_PATH, exist_ok=True)
        example.plot(path=OUTPUT_PLOT_PATH)
