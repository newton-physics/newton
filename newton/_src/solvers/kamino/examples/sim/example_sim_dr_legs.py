# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.core.builder import ModelBuilderKamino
from newton._src.solvers.kamino._src.core.joints import JointActuationType
from newton._src.solvers.kamino._src.core.types import float32, int32
from newton._src.solvers.kamino._src.linalg.linear import LinearSolverTypeToName as LinearSolverShorthand
from newton._src.solvers.kamino._src.models.builders.utils import (
    add_ground_box,
    make_homogeneous_builder,
    set_uniform_body_pose_offset,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.control import AnimationJointReference, JointSpacePIDController
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
    # Inputs:
    decimation: int32,
    model_info_joint_actuated_coords_offset: wp.array[int32],
    model_info_joint_actuated_dofs_offset: wp.array[int32],
    model_joints_wid: wp.array[int32],
    model_joints_act_type: wp.array[int32],
    model_joint_coords_offset: wp.array[int32],
    model_joint_dofs_offset: wp.array[int32],
    model_joint_actuated_coords_offset: wp.array[int32],
    model_joint_actuated_dofs_offset: wp.array[int32],
    data_time_steps: wp.array[int32],
    animation_frame: wp.array[int32],
    animation_q_j_ref: wp.array2d[float32],
    animation_dq_j_ref: wp.array2d[float32],
    # Outputs:
    control_q_j_ref: wp.array[float32],
    control_dq_j_ref: wp.array[float32],
    control_tau_j_ref: wp.array[float32],
):
    """
    A kernel to compute joint-space PID control outputs for force-actuated joints.
    """
    # Retrieve the the joint index from the thread indices
    jid = wp.tid()

    # Retrieve the joint actuation type
    act_type = model_joints_act_type[jid]

    # Retrieve the world index from the thread indices
    wid = model_joints_wid[jid]

    # Retrieve the current simulation step
    step = data_time_steps[wid]

    # Only proceed for force actuated joints and at
    # simulation steps matching the control decimation
    if act_type != JointActuationType.POSITION_VELOCITY or step % decimation != 0:
        return

    # Retrieve joint-specific mode info
    coords_offset_j = model_joint_coords_offset[jid]
    num_coords_j = model_joint_coords_offset[jid + 1] - coords_offset_j
    dofs_offset_j = model_joint_dofs_offset[jid]
    num_dofs_j = model_joint_dofs_offset[jid + 1] - dofs_offset_j
    actuated_coords_offset_j = model_joint_actuated_coords_offset[jid] - model_info_joint_actuated_coords_offset[wid]
    actuated_dofs_offset_j = model_joint_actuated_dofs_offset[jid] - model_info_joint_actuated_dofs_offset[wid]

    # Retrieve the current frame of the animation reference for the world
    frame = animation_frame[wid]

    # Copy the joint reference coordinates and velocities
    # from the animation data to the controller data
    for coord in range(num_coords_j):
        joint_coord_index = coords_offset_j + coord
        actuator_coord_index = actuated_coords_offset_j + coord
        control_q_j_ref[joint_coord_index] = animation_q_j_ref[frame, actuator_coord_index]
    for dof in range(num_dofs_j):
        joint_dof_index = dofs_offset_j + dof
        actuator_dof_index = actuated_dofs_offset_j + dof
        control_dq_j_ref[joint_dof_index] = animation_dq_j_ref[frame, actuator_dof_index]
        control_tau_j_ref[joint_dof_index] = 0.0  # No feed-forward term in this example


###
# Launchers
###


def pd_control_callback(sim: Simulator, animation: AnimationJointReference, decimation: int = 1):
    wp.launch(
        _pd_control_callback,
        dim=sim.model.size.sum_of_num_joints,
        inputs=[
            # Inputs
            int32(decimation),
            sim.model.info.joint_actuated_coords_offset,
            sim.model.info.joint_actuated_dofs_offset,
            sim.model.joints.wid,
            sim.model.joints.act_type,
            sim.model.joints.coords_offset,
            sim.model.joints.dofs_offset,
            sim.model.joints.actuated_coords_offset,
            sim.model.joints.actuated_dofs_offset,
            sim.solver.data.time.steps,
            animation.data.frame,
            animation.data.q_j_ref,
            animation.data.dq_j_ref,
            # Outputs:
            sim.control.q_j_ref,
            sim.control.dq_j_ref,
            sim.control.tau_j_ref,
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
        implicit_pd: bool = False,
        gravity: bool = True,
        ground: bool = True,
        logging: bool = False,
        linear_solver: str = "LLTB",
        linear_solver_maxiter: int = 0,
        use_graph_conditionals: bool = False,
        headless: bool = False,
        record_video: bool = False,
        async_save: bool = False,
    ):
        # Initialize target frames per second and corresponding time-steps
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        target_sim_dt = 0.01 if implicit_pd else 0.001
        self.sim_substeps = max(1, round(self.frame_dt / target_sim_dt))
        self.sim_dt = self.frame_dt / self.sim_substeps
        msg.info(f"Using sim_dt = {self.sim_dt} ({self.sim_substeps} substeps per frame)")
        self.max_steps = max_steps

        # Cache the device and other internal flags
        self.device = device
        self.use_cuda_graph: bool = use_cuda_graph
        self.logging: bool = logging
        self.implicit_pd: bool = implicit_pd

        # Load the DR Legs USD and add it to the builder
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_meshes_and_boxes.usda")

        # Create a model builder from the imported USD
        msg.notif("Constructing builder from imported USD ...")
        importer = USDImporter()
        self.builder: ModelBuilderKamino = make_homogeneous_builder(
            num_worlds=num_worlds,
            build_fn=importer.import_from,
            load_drive_dynamics=implicit_pd,
            load_static_geometry=True,
            source=asset_file,
            use_angular_drive_scaling=True,
        )
        msg.info("total mass: %f", self.builder.worlds[0].mass_total)
        msg.info("total diag inertia: %f", self.builder.worlds[0].inertia_total)

        # Offset the model to place it above the ground
        # NOTE: The USD model is centered at the origin
        offset = wp.transformf(0.0, 0.0, 0.265, 0.0, 0.0, 0.0, 1.0)
        set_uniform_body_pose_offset(builder=self.builder, offset=offset)

        # Add a static collision geometry for the plane
        if ground:
            for w in range(num_worlds):
                add_ground_box(self.builder, world_index=w)

        # Set gravity
        for w in range(self.builder.num_worlds):
            self.builder.gravity[w].enabled = gravity

        # Set joint armatures, and verify that correct gains were loaded from the USD file
        for joint in self.builder.all_joints:
            if joint.is_dynamic or joint.is_implicit_pd:
                joint.a_j = [0.011]  # Set joint armature according to Dynamixel XH540-V150 specs
                joint.b_j = [0.044]  # Set joint damping according to Dynamixel XH540-V150 specs
                assert abs(joint.k_p_j[0] - 50.0) < 1e-4
                assert abs(joint.k_d_j[0] - 1.0) < 1e-4

        # Parse the linear solver max iterations for iterative solvers from the command-line arguments
        linear_solver_kwargs = {"maxiter": linear_solver_maxiter} if linear_solver_maxiter > 0 else {}

        # Set solver config
        config = Simulator.Config()
        config.dt = self.sim_dt
        config.collision_detector.pipeline = "unified"  # Select from {"primitive", "unified"}
        config.solver.sparse_jacobian = False
        config.solver.sparse_dynamics = False
        config.solver.integrator = "moreau"  # Select from {"euler", "moreau"}
        config.solver.constraints.alpha = 0.1
        config.solver.constraints.beta = 0.011
        config.solver.constraints.gamma = 0.05
        config.solver.padmm.primal_tolerance = 1e-4
        config.solver.padmm.dual_tolerance = 1e-4
        config.solver.padmm.compl_tolerance = 1e-4
        config.solver.padmm.max_iterations = 200
        config.solver.padmm.eta = 1e-5
        config.solver.padmm.rho_0 = 0.02  # try 0.02 for Balanced update
        config.solver.padmm.rho_min = 0.05
        config.solver.padmm.penalty_update_method = "fixed"  # try "balanced"
        config.solver.padmm.use_acceleration = True
        config.solver.padmm.warmstart_mode = "containers"
        config.solver.padmm.contact_warmstart_method = "geom_pair_net_force"
        config.solver.collect_solver_info = False
        config.solver.compute_solution_metrics = logging and not use_cuda_graph
        config.solver.dynamics.linear_solver_type = linear_solver
        config.solver.dynamics.linear_solver_kwargs = linear_solver_kwargs
        config.solver.padmm.use_graph_conditionals = use_graph_conditionals
        config.solver.angular_velocity_damping = 0.0

        # Create a simulator
        msg.notif("Building the simulator...")
        self.sim = Simulator(builder=self.builder, config=config, device=device)

        # Load animation data for dr_legs
        animation_asset = str(asset_path / "dr_legs/animation" / "dr_legs_animation_100fps.npy")
        animation_np = np.load(animation_asset, allow_pickle=True)
        msg.debug("animation_np (shape={%s}):\n{%s}\n", animation_np.shape, animation_np)

        # Compute animation time step and rate
        animation_dt = 0.01  # 100 fps
        animation_rate = round(animation_dt / config.dt)
        msg.info(f"animation_dt: {animation_dt}")
        msg.info(f"animation_rate: {animation_rate}")

        # Create a joint-space animation reference generator
        self.animation = AnimationJointReference(
            model=self.sim.model,
            data=animation_np,
            data_dt=animation_dt,
            target_dt=config.dt,
            decimation=1,
            rate=1,
            loop=False,
            use_fd=True,
        )

        # Create a joint-space PID controller
        njaq = self.sim.model.size.sum_of_num_actuated_joint_dofs
        K_p = 80.0 * np.ones(njaq, dtype=np.float32)
        K_d = 0.1 * np.ones(njaq, dtype=np.float32)
        K_i = 0.01 * np.ones(njaq, dtype=np.float32)
        decimation = 1 * np.ones(self.sim.model.size.num_worlds, dtype=np.int32)
        self.controller = JointSpacePIDController(
            model=self.sim.model, K_p=K_p, K_i=K_i, K_d=K_d, decimation=decimation
        )

        # Define a callback function to reset the controller
        def reset_jointspace_pid_control_callback(simulator: Simulator):
            self.animation.reset(q_j_ref_out=self.controller.data.q_j_ref, dq_j_ref_out=self.controller.data.dq_j_ref)
            self.controller.reset(model=simulator.model, state=simulator.state)

        # Define a callback function to wrap the execution of the controller
        def compute_jointspace_pid_control_callback(simulator: Simulator):
            if self.implicit_pd:
                self.animation.advance(time=simulator.solver.data.time)
                pd_control_callback(sim=simulator, animation=self.animation, decimation=decimation[0])
            else:
                self.animation.step(
                    time=simulator.solver.data.time,
                    q_j_ref_out=self.controller.data.q_j_ref,
                    dq_j_ref_out=self.controller.data.dq_j_ref,
                )
                self.controller.compute(
                    model=simulator.model,
                    state=simulator.state,
                    time=simulator.solver.data.time,
                    control=simulator.control,
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
    parser.add_argument("--device", type=str, default=None, help="The compute device to use")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False, help="Run in headless mode")
    parser.add_argument("--num-worlds", type=int, default=1, help="Number of worlds to simulate in parallel")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps for headless mode")
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
    parser.add_argument(
        "--use-graph-conditionals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA graph conditional nodes in iterative solvers",
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
        use_graph_conditionals=args.use_graph_conditionals,
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
