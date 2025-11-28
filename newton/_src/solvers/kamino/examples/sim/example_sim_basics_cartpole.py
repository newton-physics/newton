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
import newton.examples
from newton._src.solvers.kamino.core.builder import ModelBuilder
from newton._src.solvers.kamino.core.math import I_3, R_x, screw
from newton._src.solvers.kamino.core.model import Model, ModelData
from newton._src.solvers.kamino.core.types import float32, int32, mat33f, transformf, uint32, vec3f, vec6f
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.models import get_basics_usd_assets_path
from newton._src.solvers.kamino.models.builders import add_ground_geom, build_cartpole
from newton._src.solvers.kamino.models.utils import make_homogeneous_builder
from newton._src.solvers.kamino.simulation.simulator import Simulator, SimulatorSettings
from newton._src.solvers.kamino.utils import logger as msg
from newton._src.solvers.kamino.utils.io.usd import USDImporter
from newton._src.solvers.kamino.viewer import ViewerKamino

###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Kernels
###


@wp.kernel
def _reset_select_worlds_to_dof_state(
    # Inputs:
    mask: wp.array(dtype=int32),
    # Inputs:
    model_info_body_offset: wp.array(dtype=int32),
    model_info_joint_offset: wp.array(dtype=int32),
    model_info_joint_coords_offset: wp.array(dtype=int32),
    model_info_joint_dofs_offset: wp.array(dtype=int32),
    model_info_joint_cts_offset: wp.array(dtype=int32),
    model_body_i_I_i: wp.array(dtype=mat33f),
    model_body_inv_i_I_i: wp.array(dtype=mat33f),
    model_joint_B_r_Bj: wp.array(dtype=vec3f),
    model_joint_F_r_Fj: wp.array(dtype=vec3f),
    model_joint_X_j: wp.array(dtype=mat33f),
    state_q_j: wp.array(dtype=float32),
    state_dq_j: wp.array(dtype=float32),
    # Outputs:
    data_time: wp.array(dtype=float32),
    data_steps: wp.array(dtype=int32),
    data_body_q_i: wp.array(dtype=transformf),
    data_body_u_i: wp.array(dtype=vec6f),
    data_body_I_i: wp.array(dtype=mat33f),
    data_body_inv_I_i: wp.array(dtype=mat33f),
    data_body_w_i: wp.array(dtype=vec6f),
    data_body_w_a_i: wp.array(dtype=vec6f),
    data_body_w_j_i: wp.array(dtype=vec6f),
    data_body_w_l_i: wp.array(dtype=vec6f),
    data_body_w_c_i: wp.array(dtype=vec6f),
    data_body_w_e_i: wp.array(dtype=vec6f),
    data_joint_p_j: wp.array(dtype=transformf),
    data_joint_r_j: wp.array(dtype=float32),
    data_joint_dr_j: wp.array(dtype=float32),
    data_joint_q_j: wp.array(dtype=float32),
    data_joint_dq_j: wp.array(dtype=float32),
    data_joint_lambda_j: wp.array(dtype=float32),
    data_joint_j_w_j: wp.array(dtype=vec6f),
    data_joint_j_w_a_j: wp.array(dtype=vec6f),
    data_joint_j_w_c_j: wp.array(dtype=vec6f),
    data_joint_j_w_l_j: wp.array(dtype=vec6f),
):
    # Retrieve the body index from the 1D thread index
    wid = wp.tid()

    # Retrieve the reset flag for the corresponding world
    world_has_reset = mask[wid]

    # Skip resetting this body if the world has not been marked for reset
    if not world_has_reset:
        return

    # Reset both the physical time and step count to zero
    data_time[wid] = 0.0
    data_steps[wid] = 0

    # Retrieve model info index ranges for this world
    bodies_start = model_info_body_offset[wid]
    joints_start = model_info_joint_offset[wid]
    joints_coords_start = model_info_joint_coords_offset[wid]
    joints_coords_end = joints_coords_start + 2
    joints_dofs_start = model_info_joint_dofs_offset[wid]
    joints_dofs_end = joints_dofs_start + 2
    joints_cts_start = model_info_joint_cts_offset[wid]

    # Retrieve the joint states for this world
    q_j = state_q_j[joints_coords_start:joints_coords_end]
    dq_j = state_dq_j[joints_dofs_start:joints_dofs_end]
    r_j = wp.zeros(shape=(5,), dtype=float32)

    # Retrieve joint model parameters
    B_r_Bj_cart_to_pole = model_joint_B_r_Bj[joints_start + 1]
    F_r_Fj_cart_to_pole = model_joint_F_r_Fj[joints_start + 1]
    X_j_cart_to_pole = model_joint_X_j[joints_start + 1]

    # Initialize state-dependent relative body quantities
    R_cart = I_3
    q_cart = wp.quat_identity()
    omega_cart = vec3f(0.0, 0.0, 0.0)
    r_cart = vec3f(0.0, q_j[0], 0.0)
    v_cart = vec3f(0.0, dq_j[0], 0.0)
    R_cart_to_pole = R_x(q_j[1])
    omega_cart_to_pole = vec3f(dq_j[1], 0.0, 0.0)

    # Compute body poses and twists in world coordinates
    r_pole = r_cart + R_cart * (
        B_r_Bj_cart_to_pole - X_j_cart_to_pole @ R_cart_to_pole @ wp.transpose(X_j_cart_to_pole) @ F_r_Fj_cart_to_pole
    )
    R_pole = R_cart @ X_j_cart_to_pole @ R_cart_to_pole @ wp.transpose(X_j_cart_to_pole)
    q_pole = wp.quat_from_matrix(R_pole)
    omega_pole = omega_cart + R_cart @ X_j_cart_to_pole @ omega_cart_to_pole
    v_pole = v_cart + wp.cross(omega_cart, -R_pole @ F_r_Fj_cart_to_pole)

    # Compute world-frame inertia tensors of each body
    I_cart = R_cart @ model_body_i_I_i[bodies_start + 0] @ wp.transpose(R_cart)
    I_pole = R_pole @ model_body_i_I_i[bodies_start + 1] @ wp.transpose(R_pole)
    inv_I_cart = R_cart @ model_body_inv_i_I_i[bodies_start + 0] @ wp.transpose(R_cart)
    inv_I_pole = R_pole @ model_body_inv_i_I_i[bodies_start + 1] @ wp.transpose(R_pole)

    # Compute joint-frame poses in world coordinates
    q_j_world_to_cart = q_cart
    r_j_world_to_cart = r_cart
    R_j_cart_to_pole = R_cart @ X_j_cart_to_pole
    q_j_cart_to_pole = wp.quat_from_matrix(R_j_cart_to_pole)
    r_j_cart_to_pole = r_cart + R_cart @ B_r_Bj_cart_to_pole

    # Define a zero screw vector constant
    ZEROS6 = vec6f(0.0)

    # Set per-body data
    data_body_q_i[bodies_start + 0] = wp.transformf(r_cart, q_cart)
    data_body_u_i[bodies_start + 0] = screw(v_cart, omega_cart)
    data_body_I_i[bodies_start + 0] = I_cart
    data_body_w_i[bodies_start + 0] = ZEROS6
    data_body_w_a_i[bodies_start + 0] = ZEROS6
    data_body_w_j_i[bodies_start + 0] = ZEROS6
    data_body_w_l_i[bodies_start + 0] = ZEROS6
    data_body_w_c_i[bodies_start + 0] = ZEROS6
    data_body_w_e_i[bodies_start + 0] = ZEROS6
    data_body_inv_I_i[bodies_start + 0] = inv_I_cart
    data_body_q_i[bodies_start + 1] = wp.transformf(r_pole, q_pole)
    data_body_u_i[bodies_start + 1] = screw(v_pole, omega_pole)
    data_body_I_i[bodies_start + 1] = I_pole
    data_body_inv_I_i[bodies_start + 1] = inv_I_pole
    data_body_w_i[bodies_start + 1] = ZEROS6
    data_body_w_a_i[bodies_start + 1] = ZEROS6
    data_body_w_j_i[bodies_start + 1] = ZEROS6
    data_body_w_l_i[bodies_start + 1] = ZEROS6
    data_body_w_c_i[bodies_start + 1] = ZEROS6
    data_body_w_e_i[bodies_start + 1] = ZEROS6

    # Set per-joint data
    for j in range(2):
        data_joint_q_j[joints_coords_start + j] = q_j[j]
        data_joint_dq_j[joints_dofs_start + j] = dq_j[j]
    for j in range(5):
        data_joint_r_j[joints_cts_start + j] = r_j[j]
        data_joint_dr_j[joints_cts_start + j] = r_j[j]
        data_joint_lambda_j[joints_cts_start + j] = r_j[j]
    data_joint_p_j[joints_start + 0] = wp.transformf(r_j_world_to_cart, q_j_world_to_cart)
    data_joint_j_w_j[joints_start + 0] = ZEROS6
    data_joint_j_w_a_j[joints_start + 0] = ZEROS6
    data_joint_j_w_c_j[joints_start + 0] = ZEROS6
    data_joint_j_w_l_j[joints_start + 0] = ZEROS6
    data_joint_p_j[joints_start + 1] = wp.transformf(r_j_cart_to_pole, q_j_cart_to_pole)
    data_joint_j_w_j[joints_start + 1] = ZEROS6
    data_joint_j_w_a_j[joints_start + 1] = ZEROS6
    data_joint_j_w_c_j[joints_start + 1] = ZEROS6
    data_joint_j_w_l_j[joints_start + 1] = ZEROS6


###
# Launchers
###


def reset_select_worlds_to_dof_state(
    model: Model,
    q_j: wp.array,
    dq_j: wp.array,
    mask: wp.array,
    data: ModelData,
):
    """
    Reset the state of the selected worlds given an array of per-world flags.

    Args:
        model: Input model container holding the time-invariant data of the system.
        state: Input state container specifying the target state to be reset to.
        mask: Array of per-world flags indicating which worlds should be reset.
        data: Output solver data to be configured for the target state.
    """
    wp.launch(
        _reset_select_worlds_to_dof_state,
        dim=model.size.num_worlds,
        inputs=[
            # Inputs:
            mask,
            model.info.bodies_offset,
            model.info.joints_offset,
            model.info.joint_coords_offset,
            model.info.joint_dofs_offset,
            model.info.joint_cts_offset,
            model.bodies.i_I_i,
            model.bodies.inv_i_I_i,
            model.joints.B_r_Bj,
            model.joints.F_r_Fj,
            model.joints.X_j,
            q_j,
            dq_j,
            # Outputs:
            data.time.time,
            data.time.steps,
            data.bodies.q_i,
            data.bodies.u_i,
            data.bodies.I_i,
            data.bodies.inv_I_i,
            data.bodies.w_i,
            data.bodies.w_a_i,
            data.bodies.w_j_i,
            data.bodies.w_l_i,
            data.bodies.w_c_i,
            data.bodies.w_e_i,
            data.joints.p_j,
            data.joints.r_j,
            data.joints.dr_j,
            data.joints.q_j,
            data.joints.dq_j,
            data.joints.lambda_j,
            data.joints.j_w_j,
            data.joints.j_w_a_j,
            data.joints.j_w_c_j,
            data.joints.j_w_l_j,
        ],
    )


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
    t_end = float32(3.1)

    # Get the current time
    t = state_t[wid]

    # Apply a time-dependent external force
    if t >= 0.0 and t < t_start:
        control_tau_j[wid * 2 + 0] = 1.0 * wp.randf(uint32(wid) + uint32(t), -1.0, 1.0)
        control_tau_j[wid * 2 + 1] = 0.0
    elif t > t_start and t < t_end:
        control_tau_j[wid * 2 + 0] = 10.0
        control_tau_j[wid * 2 + 1] = 0.0
    else:
        control_tau_j[wid * 2 + 0] = -10.0
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
        ground: bool = True,
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
            msg.notif("Constructing builder from imported USD ...")
            USD_MODEL_PATH = os.path.join(get_basics_usd_assets_path(), "cartpole.usda")
            importer = USDImporter()
            # self.builder: ModelBuilder = importer.import_from(source=USD_MODEL_PATH, load_static_geometry=ground)
            self.builder: ModelBuilder = make_homogeneous_builder(
                num_worlds=num_worlds, build_fn=importer.import_from, load_static_geometry=True, source=USD_MODEL_PATH
            )
            if ground:
                for w in range(num_worlds):
                    add_ground_geom(self.builder, world_index=w)
        else:
            msg.notif("Constructing builder using model generator ...")
            self.builder: ModelBuilder = make_homogeneous_builder(
                num_worlds=num_worlds, build_fn=build_cartpole, ground=False
            )

        # Demo of printing builder contents in debug logging mode
        msg.info("self.builder.gravity:\n{%s}", self.builder.gravity)
        msg.info("self.builder.bodies:\n{%s}", self.builder.bodies)
        msg.info("self.builder.joints:\n{%s}", self.builder.joints)
        msg.info("self.builder.collision_geoms:\n{%s}", self.builder.collision_geoms)
        msg.info("self.builder.physical_geoms:\n{%s}", self.builder.physical_geoms)

        # Set solver settings
        settings = SimulatorSettings()
        settings.dt = self.sim_dt
        settings.problem.alpha = 0.1
        settings.problem.beta = 0.1
        settings.solver.primal_tolerance = 1e-6
        settings.solver.dual_tolerance = 1e-6
        settings.solver.compl_tolerance = 1e-6
        settings.solver.max_iterations = 200
        settings.solver.rho_0 = 0.05

        # Create a simulator
        msg.notif("Building the simulator...")
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

        # Declare a PyTorch data interface for the current state and controls data
        self.states: CartpoleStates | None = None
        self.actions: CartpoleActions | None = None
        self.world_mask_wp: wp.array | None = None
        self.world_mask_pt: torch.Tensor | None = None

        # Set default default reset joint coordinates
        _q_j_ref = [0.0, 0.0]
        q_j_ref = np.tile(_q_j_ref, reps=self.sim.model.size.num_worlds)
        self.q_j_ref: wp.array = wp.array(q_j_ref, dtype=float32, device=self.device)

        # Set default default reset joint velocities
        _dq_j_ref = [0.0, 0.0]
        dq_j_ref = np.tile(_dq_j_ref, reps=self.sim.model.size.num_worlds)
        self.dq_j_ref: wp.array = wp.array(dq_j_ref, dtype=float32, device=self.device)

        # Initialize RL interfaces
        self.make_rl_interface()

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
        # Create a world mask array+tensor for per-world selective resets
        self.world_mask_wp = wp.ones((num_worlds,), dtype=wp.int32, device=self.device)
        self.world_mask_pt = wp.to_torch(self.world_mask_wp)

    def _reset_worlds(self):
        """TODO"""
        self.sim.reset_custom(
            reset_fn=reset_select_worlds_to_dof_state,
            model=self.sim.model,
            q_j=self.q_j_ref,
            dq_j=self.dq_j_ref,
            mask=self.world_mask_wp,
            data=self.sim.data.solver,
        )

    def capture(self):
        """Capture CUDA graph if requested and available."""
        if self.use_cuda_graph:
            msg.notif("Running with CUDA graphs...")
            with wp.ScopedCapture(device=self.device) as reset_capture:
                self._reset_worlds()
            self.reset_graph = reset_capture.graph
            with wp.ScopedCapture(device=self.device) as step_capture:
                self.sim.step()
            self.step_graph = step_capture.graph
            with wp.ScopedCapture(device=self.device) as sim_capture:
                self.simulate()
            self.simulate_graph = sim_capture.graph
        else:
            msg.notif("Running with kernels...")

    def simulate(self):
        """Run simulation substeps."""
        for _i in range(self.sim_substeps):
            self.sim.step()

    def reset(self):
        """Reset the simulation."""
        if self.reset_graph:
            wp.capture_launch(self.reset_graph)
        else:
            self._reset_worlds()
        self.sim_steps = 0
        self.sim_time = 0.0

    def step_once(self):
        """Run the simulation for a single time-step."""
        if self.step_graph:
            wp.capture_launch(self.step_graph)
        else:
            self.sim.step()
        self.sim_time += self.sim_dt
        self.sim_steps += 1

    def step(self):
        """Step the simulation."""
        if self.simulate_graph:
            wp.capture_launch(self.simulate_graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.sim_steps += self.sim_substeps

        # TODO
        if self.sim_steps > 2000:
            msg.warning("Resetting simulation after %d steps", self.sim_steps)
            self.reset()

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
    parser.add_argument("--num-worlds", type=int, default=4, help="Number of worlds to simulate in parallel")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps for headless mode")
    parser.add_argument("--load-from-usd", action="store_true", default=False, help="Load model from USD file")
    parser.add_argument("--ground", action="store_true", default=True, help="Adds a ground plane to the simulation")
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
    msg.set_log_level(msg.LogLevel.NOTIF)

    # Set device if specified, otherwise use Warp's default
    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()

    # Determine if CUDA graphs should be used for execution
    can_use_cuda_graph = device.is_cuda and wp.is_mempool_enabled(device)
    use_cuda_graph = can_use_cuda_graph & args.cuda_graph
    msg.notif(f"can_use_cuda_graph: {can_use_cuda_graph}")
    msg.notif(f"use_cuda_graph: {use_cuda_graph}")
    msg.notif(f"device: {device}")

    # Create example instance
    example = Example(
        device=device,
        use_cuda_graph=use_cuda_graph,
        load_from_usd=args.load_from_usd,
        num_worlds=args.num_worlds,
        max_steps=args.num_steps,
        ground=args.ground,
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
            camera_pos = wp.vec3(5.0, 5.0, 1.5)
            pitch = -10.0
            yaw = 218.0
            example.viewer.set_camera(camera_pos, pitch, yaw)

        # Launch the example using Newton's built-in runtime
        newton.examples.run(example, args)
