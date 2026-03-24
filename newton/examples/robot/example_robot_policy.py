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
# Shows how to control robot pretrained in IsaacLab with RL.
# The policy is loaded from a file and the robot is controlled via keyboard.
#
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
# Run this example with:
# python -m newton.examples robot_policy --robot g1_29dof
# python -m newton.examples robot_policy --robot g1_23dof
# python -m newton.examples robot_policy --robot go2
# python -m newton.examples robot_policy --robot anymal
# python -m newton.examples robot_policy --robot anymal --physx
# to run the example with a PhysX-trained policy run with --physx
###########################################################################

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp
import yaml

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import JointTargetMode, State

# Use new asset for G1, but copy model attributes from the original asset.
MODEL_HACK = True


@dataclass
class RobotConfig:
    """Configuration for a robot including asset paths and policy paths."""

    asset_dir: str
    policy_path: dict[str, str]
    asset_path: str
    yaml_path: str  # Path within the asset directory to the configuration YAML


# Robot configurations pointing to newton-assets repository
ROBOT_CONFIGS = {
    "anymal": RobotConfig(
        asset_dir="anybotics_anymal_c",
        policy_path={"mjw": "rl_policies/mjw_anymal.pt", "physx": "rl_policies/physx_anymal.pt"},
        asset_path="usd/anymal_c.usda",
        yaml_path="rl_policies/anymal.yaml",
    ),
    "go2": RobotConfig(
        asset_dir="unitree_go2",
        policy_path={"mjw": "rl_policies/mjw_go2.pt", "physx": "rl_policies/physx_go2.pt"},
        asset_path="usd/go2.usda",
        yaml_path="rl_policies/go2.yaml",
    ),
    "g1_29dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_29DOF.pt"},
        asset_path="usd/g1_isaac.usd",
        yaml_path="rl_policies/g1_29dof.yaml",
    ),
    "g1_23dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_23DOF.pt", "physx": "rl_policies/physx_g1_23DOF.pt"},
        asset_path="usd/g1_minimal.usd",
        yaml_path="rl_policies/g1_23dof.yaml",
    ),
}

if MODEL_HACK:
    ROBOT_CONFIGS["g1_29dof"].asset_path = "usd_structured/g1_29dof_with_hand_rev_1_0.usda"


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


def add_terrain(builder,
                file_path: str | None = None,
                root_prim: str | None = None,
                usd_collide: bool = False,
                add_ground_plane: bool = False
):
    """Add collision geometry from a USD file to a builder.

    Args:
        file_path: Path to USD file. If None, only create ground plane.
        root_prim: The root USD prim path for the collision geometry. Shapes will be loaded
            recursively from this path. If None, load shapes from the entire file.
        add_ground_plane: Add a ground plane collider for a flat walking area.
    """
    if file_path is not None:
        # check if we should load collisions
        if usd_collide:
            if root_prim is None:
                root_prim = "/"

            shape_start = builder.shape_count
            builder.add_usd(file_path, root_path=root_prim, hide_collision_shapes=True, load_visual_shapes=False)

            # clear ShapeFlags.VISIBLE to hide the collision shape
            for idx in range(shape_start, builder.shape_count):
                flags = builder.shape_flags[idx]
                builder.shape_flags[idx] = flags & ~newton.ShapeFlags.VISIBLE

        # add ground plane, but make it invisible
        if add_ground_plane:
            cfg = newton.ModelBuilder.ShapeConfig(is_visible=False)
            builder.add_ground_plane(cfg=cfg)
    else:
        # add visible ground plane when no USD background is specified
        builder.add_ground_plane()


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
    root_lin_vel_w = torch.tensor(joint_qd[:3], device=device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(joint_qd[3:6], device=device, dtype=torch.float32).unsqueeze(0)
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


def get_camera_from_usd(usd_path: str, up_axis: int = 2) -> tuple[wp.vec3, float, float] | None:
    """Read camera position and orientation from USD file at /World/Camera.

    Args:
        usd_path: Path to the USD file.
        up_axis: Up axis index (0=X, 1=Y, 2=Z). Default is 2 (Z-up).

    Returns:
        Tuple of (position, pitch, yaw) if camera exists, None otherwise.
    """
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(usd_path)
        if not stage:
            return None

        camera_prim = stage.GetPrimAtPath("/World/Camera")
        if not camera_prim or not camera_prim.IsValid():
            return None

        # Get the transform from the camera prim
        xform = newton.usd.get_transform(camera_prim, local=False)
        # Decompose transform to get position and rotation
        mat = wp.transform_to_matrix(xform)
        pos, rot, _ = wp.transform_decompose(mat)

        # Convert quaternion to forward direction vector
        # Camera forward is typically -Z in camera space
        # We need to rotate (0, 0, -1) by the quaternion to get world-space forward
        forward_camera_space = wp.vec3f(0.0, 0.0, -1.0)
        forward_world = wp.quat_rotate(rot, forward_camera_space)

        # Convert forward vector to pitch/yaw based on up_axis
        # For Z-up (up_axis=2):
        #   front_x = cos(yaw) * cos(pitch)
        #   front_y = sin(yaw) * cos(pitch)
        #   front_z = sin(pitch)
        forward_array = np.array([float(forward_world[0]), float(forward_world[1]), float(forward_world[2])])
        if up_axis == 2:  # Z up
            pitch_rad = np.arcsin(np.clip(forward_array[2], -1.0, 1.0))
            yaw_rad = np.arctan2(forward_array[1], forward_array[0])
        elif up_axis == 0:  # X up
            pitch_rad = np.arcsin(np.clip(forward_array[0], -1.0, 1.0))
            yaw_rad = np.arctan2(forward_array[2], forward_array[1])
        else:  # Y up (default)
            pitch_rad = np.arcsin(np.clip(forward_array[1], -1.0, 1.0))
            yaw_rad = np.arctan2(forward_array[2], forward_array[0])

        pitch = float(np.rad2deg(pitch_rad))
        yaw = float(np.rad2deg(yaw_rad))

        return (pos, pitch, yaw)
    except Exception:
        # If anything fails, return None to fall back to defaults
        return None


class Example:
    def __init__(
        self,
        viewer,
        robot_config: RobotConfig,
        config,
        asset_directory: str,
        mjc_to_physx: list[int],
        physx_to_mjc: list[int],
        background_usd_path: str | None = None,
        background_usd_root: str | None = None,
        background_usd_collide: bool = False,
    ):
        # Setup simulation parameters first
        fps = 200
        self.frame_dt = 1.0e0 / fps
        self.decimation = 4
        self.cycle_time = 1 / fps * self.decimation

        # Group related attributes by prefix
        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Save a reference to the viewer
        self.viewer = viewer

        # Store configuration
        self.use_mujoco = False
        self.config = config
        self.robot_config = robot_config

        # Device setup
        self.device = wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        # Build the model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        builder.add_usd(
            newton.examples.get_asset(asset_directory + "/" + robot_config.asset_path),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        # builder.approximate_meshes("convex_hull")

        add_terrain(
            builder,
            file_path=background_usd_path,
            root_prim=background_usd_root,
            usd_collide=background_usd_collide,
            add_ground_plane=True,
        )

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = config["mjw_joint_pos"]

        for i in range(len(config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
            builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        if MODEL_HACK:
            # construct a shadow model using the old G1 asset
            shadow_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            newton.solvers.SolverMuJoCo.register_custom_attributes(shadow_builder)
            shadow_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
                armature=0.1,
                limit_ke=1.0e2,
                limit_kd=1.0e0,
            )
            shadow_builder.default_shape_cfg.ke = 5.0e4
            shadow_builder.default_shape_cfg.kd = 5.0e2
            shadow_builder.default_shape_cfg.kf = 1.0e3
            shadow_builder.default_shape_cfg.mu = 0.75

            shadow_builder.add_usd(
                newton.examples.get_asset(asset_directory + "/usd/g1_isaac.usd"),
                xform=wp.transform(wp.vec3(0, 0, 0.8)),
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                joint_ordering="dfs",
                hide_collision_shapes=True,
            )
            # shadow_builder.approximate_meshes("convex_hull")

            add_terrain(
                builder,
                file_path=background_usd_path,
                root_prim=background_usd_root,
                usd_collide=background_usd_collide,
                add_ground_plane=True,
            )

            shadow_builder.joint_q[:3] = [0.0, 0.0, 0.76]
            shadow_builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
            shadow_builder.joint_q[7:] = config["mjw_joint_pos"]

            for i in range(len(config["mjw_joint_stiffness"])):
                shadow_builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
                shadow_builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
                shadow_builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
                shadow_builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

            shadow_model = shadow_builder.finalize()

            print(f"particle_count            {self.model.particle_count} {shadow_model.particle_count}")
            print(f"body_count                {self.model.body_count} {shadow_model.body_count}")
            print(f"shape_count               {self.model.shape_count} {shadow_model.shape_count}")
            print(f"joint_count               {self.model.joint_count} {shadow_model.joint_count}")
            print(f"tri_count                 {self.model.tri_count} {shadow_model.tri_count}")
            print(f"tet_count                 {self.model.tet_count} {shadow_model.tet_count}")
            print(f"edge_count                {self.model.edge_count} {shadow_model.edge_count}")
            print(f"spring_count              {self.model.spring_count} {shadow_model.spring_count}")
            print(f"muscle_count              {self.model.muscle_count} {shadow_model.muscle_count}")
            print(f"articulation_count        {self.model.articulation_count} {shadow_model.articulation_count}")
            print(f"joint_dof_count           {self.model.joint_dof_count} {shadow_model.joint_dof_count}")
            print(f"joint_coord_count         {self.model.joint_coord_count} {shadow_model.joint_coord_count}")
            print(f"joint_constraint_count    {self.model.joint_constraint_count} {shadow_model.joint_constraint_count}")
            print(f"equality_constraint_count {self.model.equality_constraint_count} {shadow_model.equality_constraint_count}")
            print(f"constraint_mimic_count    {self.model.constraint_mimic_count} {shadow_model.constraint_mimic_count}")

            assert self.model.joint_count == shadow_model.joint_count
            assert self.model.joint_dof_count == shadow_model.joint_dof_count
            assert self.model.body_count == shadow_model.body_count

            def get_name(label):
                return label.split("/")[-1]

            print("Joints:")
            for i in range(builder.joint_count):
                name = get_name(builder.joint_label[i])
                shadow_name = get_name(shadow_builder.joint_label[i])
                print(f"  {i}: {name} <- {shadow_name}")
                # assert name == shadow_name
                assert builder.joint_type[i] == shadow_builder.joint_type[i]

            # attributes to copy from shadow model
            attrib_names = [
                "joint_q",
                "joint_qd",
                "joint_f",
                "joint_target_pos",
                "joint_target_vel",
                "joint_act",
                "joint_type",
                "joint_articulation",
                "joint_parent",
                "joint_child",
                "joint_ancestor",
                "joint_X_p",
                "joint_X_c",
                "joint_axis",
                "joint_armature",
                "joint_target_mode",
                "joint_target_ke",
                "joint_target_kd",
                "joint_effort_limit",
                "joint_velocity_limit",
                "joint_friction",
                "joint_dof_dim",
                "joint_enabled",
                "joint_limit_lower",
                "joint_limit_upper",
                "joint_limit_ke",
                "joint_limit_kd",
                "joint_twist_lower",
                "joint_twist_upper",
                "joint_label",

                "body_q",
                "body_qd",
                "body_com",
                "body_inertia",
                "body_inv_inertia",
                "body_mass",
                "body_inv_mass",
                "body_flags",
                "body_label",
            ]

            for attrib in attrib_names:
                setattr(self.model, attrib, getattr(shadow_model, attrib))

            print("Copying custom attributes...")
            for name, ca in shadow_builder.custom_attributes.items():
                # print(f"  {name}, {ca.name}, {ca.assignment}, {ca.namespace}")
                if ca.assignment == newton.Model.AttributeAssignment.MODEL:
                    if ca.name == "condim" or ca.name.startswith("geom_"):
                        continue
                    if ca.namespace:
                        source = getattr(shadow_model, ca.namespace)
                        if not hasattr(self.model, ca.namespace):
                            setattr(self, ca.namespace, newton.Model.AttributeNamespace(ca.namespace))
                        target = getattr(self.model, ca.namespace)
                    else:
                        source = shadow_model
                        target = self.model
                    try:
                        if hasattr(source, ca.name):
                            print(f"  {name}, {ca.name}, {ca.namespace}, {ca.assignment}")
                            setattr(target, ca.name, getattr(source, ca.name))
                    except Exception as e:
                        print(f"*** Error copying custom attribute: {e}")

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco,
            solver="newton",
            nconmax=200,
            njmax=1000,
        )

        # Initialize state objects
        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        # Set model in viewer
        self.viewer.set_model(self.model)
        self.viewer.vsync = True

        # Try to get camera from USD file if provided, fall back to defaults
        if background_usd_path:
            camera_data = get_camera_from_usd(background_usd_path, up_axis=2)  # Z-up
            if camera_data is not None:
                self.cam_pos, self.cam_pitch, self.cam_yaw = camera_data
                print(f"[INFO] Loaded camera from USD: pos={self.cam_pos}, pitch={self.cam_pitch:.1f}, yaw={self.cam_yaw:.1f}")
            else:
                # Default camera values
                self.cam_pos = wp.vec3(-4.33, -2.07, 1.27)
                self.cam_pitch = -8.5
                self.cam_yaw = -335.0
                print(f"[INFO] Using default camera: pos={self.cam_pos}, pitch={self.cam_pitch:.1f}, yaw={self.cam_yaw:.1f}")
        else:
            # Default camera values when no USD file provided
            self.cam_pos = wp.vec3(-4.33, -2.07, 1.27)
            self.cam_pitch = -8.5
            self.cam_yaw = -335.0
            print(f"[INFO] Using default camera: pos={self.cam_pos}, pitch={self.cam_pitch:.1f}, yaw={self.cam_yaw:.1f}")

        # Set initial camera position (must be after set_model which resets camera)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                pos=self.cam_pos,
                pitch=self.cam_pitch,
                yaw=self.cam_yaw,
            )
        # For RTX viewer, also set camera directly to ensure it sticks
        if hasattr(self.viewer, "camera"):
            from pyglet.math import Vec3 as PyVec3
            self.viewer.camera.pos = PyVec3(float(self.cam_pos[0]), float(self.cam_pos[1]), float(self.cam_pos[2]))
            self.viewer.camera.pitch = self.cam_pitch
            self.viewer.camera.yaw = self.cam_yaw
            if hasattr(self.viewer, "_camera_dirty"):
                self.viewer._camera_dirty = True

        # Load background USD file if provided
        if background_usd_path:
            try:
                self.viewer.add_background_usd(background_usd_path, background_usd_root)
            except FileNotFoundError:
                print("File not found:", background_usd_path)
                # Silently skip if file doesn't exist
                pass
            except AttributeError:
                # Viewer doesn't support add_background_usd (e.g., not RTX viewer)
                pass

            # Re-set camera after background USD (in case it was reset)
            if hasattr(self.viewer, "set_camera"):
                self.viewer.set_camera(
                    pos=self.cam_pos,
                    pitch=self.cam_pitch,
                    yaw=self.cam_yaw,
                )
            if hasattr(self.viewer, "camera"):
                from pyglet.math import Vec3 as PyVec3
                self.viewer.camera.pos = PyVec3(float(self.cam_pos[0]), float(self.cam_pos[1]), float(self.cam_pos[2]))
                self.viewer.camera.pitch = self.cam_pitch
                self.viewer.camera.yaw = self.cam_yaw
                if hasattr(self.viewer, "_camera_dirty"):
                    self.viewer._camera_dirty = True

        # Ensure FK evaluation (for non-MuJoCo solvers)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Store initial joint state for fast reset
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        # Pre-compute tensors that don't change during simulation
        self.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=self.torch_device, dtype=torch.long)
        self.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=self.torch_device, dtype=torch.long)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self._reset_key_prev = False

        # Initialize policy-related attributes
        # (will be set by load_policy_and_setup_tensors)
        self.policy = None
        self.joint_pos_initial = None
        self.act = None
        self.rearranged_act = None

        # Track if camera has been set in first render
        self._camera_set_in_render = False

        # Call capture at the end
        self.capture()

    def capture(self):
        """Put graph capture into it's own method."""
        self.graph = None
        self.use_cuda_graph = False
        if wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
            print("[INFO] Using CUDA graph")
            self.use_cuda_graph = True
            torch_tensor = torch.zeros(self.config["num_dofs"] + 6, device=self.torch_device, dtype=torch.float32)
            self.control.joint_target_pos = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Simulate performs one frame's worth of updates."""
        need_state_copy = self.use_cuda_graph and self.sim_substeps % 2 == 1

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            # Swap states - handle CUDA graph case specially
            if need_state_copy and i == self.sim_substeps - 1:
                # Swap states by copying the state arrays for graph capture
                self.state_0.assign(self.state_1)
            else:
                # We can just swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0

        self.solver.update_contacts(self.contacts, self.state_0)

    def reset(self):
        print("[INFO] Resetting example")
        # Restore initial joint positions and velocities in-place.
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def step(self):
        # Build command from viewer keyboard
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)
            # Reset when 'P' is pressed (edge-triggered)
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

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
            a = self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act
            a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, a_wp)

        for _ in range(self.decimation):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        # Set camera on first render to ensure it's applied
        if not self._camera_set_in_render:
            if hasattr(self.viewer, "set_camera"):
                self.viewer.set_camera(
                    pos=self.cam_pos,
                    pitch=self.cam_pitch,
                    yaw=self.cam_yaw,
                )
            if hasattr(self.viewer, "camera"):
                from pyglet.math import Vec3 as PyVec3
                self.viewer.camera.pos = PyVec3(float(self.cam_pos[0]), float(self.cam_pos[1]), float(self.cam_pos[2]))
                self.viewer.camera.pitch = self.cam_pitch
                self.viewer.camera.yaw = self.cam_yaw
                if hasattr(self.viewer, "_camera_dirty"):
                    self.viewer._camera_dirty = True
            self._camera_set_in_render = True

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds
    # example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--robot", type=str, default="g1_29dof", choices=list(ROBOT_CONFIGS.keys()), help="Robot name to load"
    )
    parser.add_argument("--physx", action="store_true", help="Run physX policy instead of MJWarp.")
    parser.add_argument(
        "--usd-background", type=str, default=None, help="Path to background USD file to load"
    )
    parser.add_argument(
        "--usd-background-root",
        help="Root path of the background geometry in the USD file",
    )
    parser.add_argument(
        "--usd-background-collide",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable collisions with USD background",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Get robot configuration
    if args.robot not in ROBOT_CONFIGS:
        print(f"[ERROR] Unknown robot: {args.robot}")
        print(f"[INFO] Available robots: {list(ROBOT_CONFIGS.keys())}")
        exit(1)

    robot_config = ROBOT_CONFIGS[args.robot]
    print(f"[INFO] Selected robot: {args.robot}")

    # Download assets from newton-assets repository
    asset_directory = str(newton.utils.download_asset(robot_config.asset_dir))
    print(f"[INFO] Asset directory: {asset_directory}")

    # Load robot configuration from YAML file in the downloaded assets
    yaml_file_path = f"{asset_directory}/{robot_config.yaml_path}"
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Robot config file not found: {yaml_file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        exit(1)

    print(f"[INFO] Loaded config with {config['num_dofs']} DOFs")

    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    if args.physx:
        if "physx" not in robot_config.policy_path or "physx_joint_names" not in config:
            physx_robots = [name for name, cfg in ROBOT_CONFIGS.items() if "physx" in cfg.policy_path]
            print(f"[ERROR] PhysX policy not available for robot '{args.robot}'.")
            print(f"[INFO] Robots with PhysX support: {physx_robots}")
            exit(1)
        policy_path = f"{asset_directory}/{robot_config.policy_path['physx']}"
        mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(config["mjw_joint_names"], config["physx_joint_names"])
    else:
        policy_path = f"{asset_directory}/{robot_config.policy_path['mjw']}"

    example = Example(
        viewer,
        robot_config,
        config,
        asset_directory,
        mjc_to_physx,
        physx_to_mjc,
        background_usd_path=args.usd_background,
        background_usd_root=args.usd_background_root,
        background_usd_collide=args.usd_background_collide,
    )

    # Use utility function to load policy and setup tensors
    load_policy_and_setup_tensors(example, policy_path, config["num_dofs"], slice(7, None))

    # Run using standard example loop
    newton.examples.run(example, args)
