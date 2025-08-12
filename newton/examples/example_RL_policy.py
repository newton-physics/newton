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

###########################################################################
# Example Robot control via keyboard
#
# Shows how to control robot pretrained in IL via mjwarp.
#
###########################################################################

import time
from typing import Any

import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples.robot_configs import G1_23DOF, G1_29DOF, Anymal, Go2
from newton.sim import State

wp.config.enable_backward = False


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]  # w component is at index 3 for XYZW format
    q_vec = q[..., :3]  # xyz components are at indices 0, 1, 2
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(
    actions: torch.Tensor,
    state: State,
    joint_pos_initial: torch.Tensor,
    device: str,
    indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Compute observation for robot policy.

    Args:
        actions: Previous actions tensor
        state: Current simulation state
        joint_pos_initial: Initial joint positions
        device: PyTorch device string
        indices: Index mapping for joint reordering
        gravity_vec: Gravity vector in world frame
        command: Command vector

    Returns:
        Observation tensor for policy input
    """
    # Extract state information with proper handling
    joint_q = state.joint_q if state.joint_q is not None else []
    joint_qd = state.joint_qd if state.joint_qd is not None else []

    root_quat_w = torch.tensor(joint_q[3:7], device=device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel_w = torch.tensor(joint_qd[3:6], device=device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(joint_qd[:3], device=device, dtype=torch.float32).unsqueeze(0)
    joint_pos_current = torch.tensor(joint_q[7:], device=device, dtype=torch.float32).unsqueeze(0)
    joint_vel_current = torch.tensor(joint_qd[6:], device=device, dtype=torch.float32).unsqueeze(0)

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)

    return obs


def load_policy_and_setup_tensors(example: Any, policy_path: str, num_dofs: int, joint_pos_slice: slice):
    """Load policy and setup initial tensors for robot control.

    Args:
        example: Robot example instance
        policy_path: Path to the policy file
        num_dofs: Number of degrees of freedom
        joint_pos_slice: Slice for extracting joint positions from state
    """
    device = example.torch_device
    print("[INFO] Loading policy from:", policy_path)
    example.policy = torch.jit.load(policy_path, map_location=device)

    # Handle potential None state
    joint_q = example.state_0.joint_q if example.state_0.joint_q is not None else []
    example.joint_pos_initial = torch.tensor(joint_q[joint_pos_slice], device=device, dtype=torch.float32).unsqueeze(0)
    example.act = torch.zeros(1, num_dofs, device=device, dtype=torch.float32)
    example.rearranged_act = torch.zeros(1, num_dofs, device=device, dtype=torch.float32)


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    """
    Finds the mapping between PhysX and MJWarp joint names.
    Returns a tuple of two lists: (mjc_to_physx, physx_to_mjc).
    """
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc


"""
Robot Keyboard Controller

A simple keyboard control interface for robot command input.
"""
try:
    import pygame  # type: ignore

    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None  # type: ignore
    PYGAME_AVAILABLE = False


class RobotKeyboardController:
    """
    A simple keyboard controller for robot movement commands.
    """

    def __init__(
        self,
        command_size: int = 3,
        command_limits: tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Initialize the keyboard controller.

        Args:
            command_size: Size of command tensor (default 3 for [forward, lateral, rotation])
            command_limits: Min and max values for commands
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for RobotKeyboardController")

        self.command_size = command_size
        self.min_val, self.max_val = command_limits

        # Initialize command tensor
        self.command = torch.zeros((1, command_size), dtype=torch.float32)

        # Simple key mappings
        self.key_mappings = {
            pygame.K_w: (0, 1.0),  # forward
            pygame.K_s: (0, -1.0),  # backward
            pygame.K_a: (1, 0.5),  # left (reduced speed)
            pygame.K_d: (1, -0.5),  # right (reduced speed)
            pygame.K_q: (2, 1.0),  # rotate left
            pygame.K_e: (2, -1.0),  # rotate right
        }

        self._running = True
        self._reset_requested = False
        pygame.init()
        pygame.font.init()

        # Create window for input and display
        self._screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Robot Control")

        # Initialize fonts
        self._font = pygame.font.Font(None, 28)
        self._small_font = pygame.font.Font(None, 24)

    def update(self, verbose: bool = False) -> bool:
        """
        Update the controller state based on keyboard input.

        Args:
            verbose: If True, print command changes to console

        Returns:
            False if user wants to quit, True otherwise
        """
        # Process events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self._reset_requested = True

        # Reset commands
        self.command.fill_(0.0)

        # Check pressed keys
        keys = pygame.key.get_pressed()
        command_changed = False

        for key, (index, value) in self.key_mappings.items():
            if keys[key] and index < self.command_size:
                clamped_value = max(self.min_val, min(self.max_val, value))
                self.command[0, index] = clamped_value
                command_changed = True

        # Update display
        self._update_display()

        # Print feedback if requested
        if verbose and command_changed:
            cmd_str = ", ".join([f"{self.command[0, i].item():.3f}" for i in range(self.command_size)])
            print(f"Command: [{cmd_str}]")

        return self._running

    def _update_display(self):
        """Update the pygame window with current command values and instructions."""
        # Clear screen with dark background
        self._screen.fill((20, 30, 50))

        # Display current command values
        y_pos = 70

        instructions = [
            "Controls:",
            "W/S: Forward/Backward",
            "A/D: Left/Right",
            "Q/E: Rotate Left/Right",
            "R: Reset",
            "Close window to exit",
        ]

        for instruction in instructions:
            color = (255, 255, 255) if instruction.endswith(":") else (200, 200, 200)
            inst_surface = self._small_font.render(instruction, True, color)
            self._screen.blit(inst_surface, (20, y_pos))
            y_pos += 25

        # Update display
        pygame.display.flip()

    def get_command(self) -> torch.Tensor:
        """Get the current command tensor."""
        return self.command.clone()

    def reset_commands(self):
        """Reset all commands to zero."""
        self.command.fill_(0.0)
        self._update_display()

    def cleanup(self):
        """Clean up pygame resources."""
        if PYGAME_AVAILABLE and pygame is not None and pygame.get_init():
            pygame.quit()

    def consume_reset_request(self) -> bool:
        """Return whether a reset was requested and clear the request flag."""
        requested = self._reset_requested
        self._reset_requested = False
        return requested

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class Example:
    def __init__(self, config):
        self.device = wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"
        self.use_mujoco = False
        self.config = config
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        newton.utils.parse_usd(
            newton.examples.get_asset(config.asset_path),
            builder,
            joint_drive_gains_scaling=1.0,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
        )
        builder.approximate_meshes("convex_hull")

        builder.add_ground_plane()
        builder.gravity = wp.vec3(0.0, 0.0, -9.81)
        self.sim_time = 0.0
        self.sim_step = 0
        fps = 200
        self.decimation = 4
        self.frame_dt = 1.0e0 / fps
        self.cycle_time = 1 / fps * self.decimation

        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = config.mjw_joint_pos

        for i in range(len(builder.joint_dof_mode)):
            builder.joint_dof_mode[i] = newton.JOINT_MODE_TARGET_POSITION

        for i in range(len(config.mjw_joint_stiffness)):
            builder.joint_target_ke[i + 6] = config.mjw_joint_stiffness[i]
            builder.joint_target_kd[i + 6] = config.mjw_joint_damping[i]
            builder.joint_armature[i + 6] = config.mjw_joint_armature[i]

        self.model = builder.finalize()
        self.solver = newton.solvers.MuJoCoSolver(
            self.model,
            use_mujoco=self.use_mujoco,
            solver="newton",
            ncon_per_env=30,
            contact_stiffness_time_const=0.01,
        )

        self.renderer = newton.utils.SimRendererOpenGL(self.model, "RL Policy Example")
        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        # Store initial joint state for fast reset.
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)
        # Pre-compute tensors that don't change during simulation
        self.physx_to_mjc_indices = torch.tensor(
            [physx_to_mjc[i] for i in range(len(physx_to_mjc))], device=self.torch_device
        )
        self.mjc_to_physx_indices = torch.tensor(
            [mjc_to_physx[i] for i in range(len(mjc_to_physx))], device=self.torch_device
        )
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device()) and not self.use_mujoco
        if self.use_cuda_graph:
            torch_tensor = torch.zeros(config.num_dofs + 6, device=self.torch_device, dtype=torch.float32)
            self.control.joint_target = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        state_0_dict = self.state_0.__dict__
        state_1_dict = self.state_1.__dict__
        state_temp_dict = self.state_temp.__dict__
        self.contacts = self.model.collide(self.state_0)
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if i < self.sim_substeps - 1 or not self.use_cuda_graph:
                # we can just swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0
            elif self.use_cuda_graph:
                # swap states by copying the state arrays for graph capture
                for key, value in state_0_dict.items():
                    if isinstance(value, wp.array):
                        if key not in state_temp_dict:
                            state_temp_dict[key] = wp.empty_like(value)
                        state_temp_dict[key].assign(value)
                        state_0_dict[key].assign(state_1_dict[key])
                        state_1_dict[key].assign(state_temp_dict[key])

    def reset(self):
        print("[INFO] Resetting example")
        # Restore initial joint positions and velocities in-place.
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.sim.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def step(self):
        with wp.ScopedTimer("step"):
            obs = compute_obs(
                self.act,
                self.state_0,
                self.joint_pos_initial,
                self.torch_device,
                self.physx_to_mjc_indices,
                self.gravity_vec,
                self.command,
            )
            with torch.no_grad():
                self.act = self.policy(obs)
                self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices)
                a = self.joint_pos_initial + self.config.action_scale * self.rearranged_act
                a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
                a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
                wp.copy(self.control.joint_target, a_wp)

            for _ in range(example.decimation):
                if self.use_cuda_graph:
                    wp.capture_launch(self.graph)
                else:
                    self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=100000, help="Total number of frames.")
    parser.add_argument(
        "--robot", type=str, default="g1_29dof", help="Robot to use. Choose between g1_29dof, g1_23dof, go2, anymal"
    )
    parser.add_argument("--physx", action=argparse.BooleanOptionalAction, help="Run physX policy instead of MJWarp.")

    args = parser.parse_known_args()[0]
    robots = {"g1_29dof": G1_29DOF, "g1_23dof": G1_23DOF, "go2": Go2, "anymal": Anymal}

    config = robots[args.robot]()
    print("[INFO] Selected robot:", args.robot)
    mjc_to_physx = list(range(config.num_dofs))
    physx_to_mjc = list(range(config.num_dofs))

    with wp.ScopedDevice(args.device):
        if args.physx:
            policy_path = config.policy_path["physx"]
            mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(config.mjw_joint_names, config.physx_joint_names)
            if config.policy_path.get("physx") is None:
                raise ValueError("PhysX policy path not found in robot configuration.")
        else:
            policy_path = config.policy_path["mjw"]

        example = Example(config)

        # Use utility function to load policy and setup tensors
        load_policy_and_setup_tensors(example, policy_path, config.num_dofs, slice(7, None))

        # Initialize keyboard controller
        keyboard_controller = RobotKeyboardController(
            command_size=3,
            command_limits=(-1.0, 1.0),
        )

        show_mujoco_viewer = False
        if show_mujoco_viewer:
            import mujoco
            import mujoco.viewer
            import mujoco_warp

            mjm, mjd = example.solver.mj_model, example.solver.mj_data
            m, d = example.solver.mjw_model, example.solver.mjw_data
            viewer = mujoco.viewer.launch_passive(mjm, mjd)

        running = True
        frame_count = 0
        for _ in range(args.num_frames):
            start_time = time.monotonic()
            if not running:
                break

            # Handle keyboard input and check if we should continue
            running = keyboard_controller.update(verbose=False)

            # Update the robot's command from the keyboard controller
            cpu_command = keyboard_controller.get_command()
            example.command = cpu_command.to(example.torch_device)

            # Print current command values for debugging
            if frame_count % 180 == 0:  # Every second at 100 FPS
                cmd = example.command[0]
                kb_cmd = keyboard_controller.get_command()[0]
                print(
                    f"Frame {frame_count}: Robot cmd="
                    f"[{cmd[0].item():.3f}, {cmd[1].item():.3f}], "
                    f"KB cmd=[{kb_cmd[0].item():.3f}, "
                    f"{kb_cmd[1].item():.3f}]"
                )

            example.step()
            example.render()
            if show_mujoco_viewer:
                if not example.solver.use_mujoco:
                    mujoco_warp.get_data_into(mjd, mjm, d)
                viewer.sync()
            elapsed_time = time.monotonic() - start_time
            sleep_time = example.cycle_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

            frame_count += 1

            # Reset only if the user requested a reset via keyboard (R key).
            if keyboard_controller.consume_reset_request():
                example.reset()

        if example.renderer:
            example.renderer.save()

        keyboard_controller.cleanup()
