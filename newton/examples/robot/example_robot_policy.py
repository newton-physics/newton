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
# Example Robot control via keyboard or Xbox gamepad
#
# Shows how to control robot pretrained in IsaacLab with RL.
# When an Xbox 360/One controller is detected it is used automatically;
# otherwise falls back to keyboard input via the 3-D viewer.
#
# Xbox gamepad mapping:
#   Left stick Y           — forward / backward
#   Left stick X           — turn left / right (yaw)
#   Triggers (L minus R)   — strafe left / right
#   Right stick X/Y        — orbit / rotate camera (both follow and free modes)
#   D-pad up/down          — dolly camera forward / back (free mode)
#   D-pad left/right       — pan camera left / right (free mode)
#   Y button               — toggle follow cam
#   B button               — snap camera behind robot
#   A button               — snap camera to front of robot
#   Select / Back          — reset
#
# Keyboard mapping (viewer window must have focus):
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
# Press "x" to toggle third-person camera follow mode.
# In follow mode, left-drag to orbit; press SHIFT to snap behind, CTRL to snap to front.
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

SPAWN_POS = wp.vec3(0.0, 0.0, 0.8)
SPAWN_ROT = wp.quat(0.0, 0.0, 0.7071, 0.7071)

BALL_RADIUS = 0.3
BALL_Z = 1.5
BALL_POSITIONS = [
    wp.vec3(0.5, 0.0, BALL_Z),
    wp.vec3(-0.6, 0.4, BALL_Z),
    wp.vec3(0.3, -0.5, BALL_Z),
    wp.vec3(-0.4, -0.3, BALL_Z),
    wp.vec3(0.7, 0.6, BALL_Z),
]


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

            cfg = newton.ModelBuilder.ShapeConfig()
            cfg.margin = 0.01
            cfg.gap = 0.02
            saved_cfg = builder.default_shape_cfg
            builder.default_shape_cfg = cfg

            shape_start = builder.shape_count
            builder.add_usd(file_path, root_path=root_prim, hide_collision_shapes=True, load_visual_shapes=False)

            builder.default_shape_cfg = saved_cfg

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


def add_balls(builder, num_balls: int = 1):
    """Add a bunch of balls to the builder."""
    positions = BALL_POSITIONS[:num_balls]

    ball_cfg = newton.ModelBuilder.ShapeConfig()
    ball_cfg.density = 50.0
    ball_cfg.mu = 0.5
    for i, pos in enumerate(positions):
        ball_body = builder.add_body(
            xform=wp.transform(p=pos, q=wp.quat_identity()),
            label=f"ball_{i}",
        )
        builder.add_shape_sphere(ball_body, radius=BALL_RADIUS, cfg=ball_cfg)


def create_model(
        asset_path: str,
        background_usd_path: str | None = None,
        background_usd_root: str | None = None,
        background_usd_collide: bool = False,
        num_balls: int = 1,
):
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
        newton.examples.get_asset(asset_directory + "/" + asset_path),
        xform=wp.transform(SPAWN_POS, SPAWN_ROT),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        joint_ordering="dfs",
        hide_collision_shapes=True,
    )
    builder.approximate_meshes("convex_hull")

    builder.joint_q[7:builder.joint_coord_count] = config["mjw_joint_pos"]

    for i in range(len(config["mjw_joint_stiffness"])):
        builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
        builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
        builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
        builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

    add_terrain(
        builder,
        file_path=background_usd_path,
        root_prim=background_usd_root,
        usd_collide=background_usd_collide,
        add_ground_plane=True,
    )

    add_balls(builder, num_balls=num_balls)

    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))

    return model, builder


def clone_model_parameters(src_model, src_builder, dst_model, dst_builder):
    """Clone model parameters to make RL policy work."""

    print(f"particle_count            {dst_model.particle_count} {src_model.particle_count}")
    print(f"body_count                {dst_model.body_count} {src_model.body_count}")
    print(f"shape_count               {dst_model.shape_count} {src_model.shape_count}")
    print(f"joint_count               {dst_model.joint_count} {src_model.joint_count}")
    print(f"tri_count                 {dst_model.tri_count} {src_model.tri_count}")
    print(f"tet_count                 {dst_model.tet_count} {src_model.tet_count}")
    print(f"edge_count                {dst_model.edge_count} {src_model.edge_count}")
    print(f"spring_count              {dst_model.spring_count} {src_model.spring_count}")
    print(f"muscle_count              {dst_model.muscle_count} {src_model.muscle_count}")
    print(f"articulation_count        {dst_model.articulation_count} {src_model.articulation_count}")
    print(f"joint_dof_count           {dst_model.joint_dof_count} {src_model.joint_dof_count}")
    print(f"joint_coord_count         {dst_model.joint_coord_count} {src_model.joint_coord_count}")
    print(f"joint_constraint_count    {dst_model.joint_constraint_count} {src_model.joint_constraint_count}")
    print(f"equality_constraint_count {dst_model.equality_constraint_count} {src_model.equality_constraint_count}")
    print(f"constraint_mimic_count    {dst_model.constraint_mimic_count} {src_model.constraint_mimic_count}")

    assert dst_model.joint_count == src_model.joint_count
    assert dst_model.joint_dof_count == src_model.joint_dof_count
    assert dst_model.body_count == src_model.body_count

    def get_name(label):
        return label.split("/")[-1]

    print("Joints:")
    for i in range(dst_builder.joint_count):
        name = get_name(dst_builder.joint_label[i])
        shadow_name = get_name(src_builder.joint_label[i])
        print(f"  {i}: {name} <- {shadow_name}")
        # assert name == shadow_name
        assert dst_builder.joint_type[i] == src_builder.joint_type[i]

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
        setattr(dst_model, attrib, getattr(src_model, attrib))

    print("Copying custom attributes...")
    for name, ca in src_builder.custom_attributes.items():
        # print(f"  {name}, {ca.name}, {ca.assignment}, {ca.namespace}")
        if ca.assignment == newton.Model.AttributeAssignment.MODEL:
            if ca.name == "condim" or ca.name.startswith("geom_"):
                continue
            if ca.namespace:
                source = getattr(src_model, ca.namespace)
                if not hasattr(dst_model, ca.namespace):
                    setattr(dst_model, ca.namespace, newton.Model.AttributeNamespace(ca.namespace))
                target = getattr(dst_model, ca.namespace)
            else:
                source = src_model
                target = dst_model
            try:
                if hasattr(source, ca.name):
                    print(f"  {name}, {ca.name}, {ca.namespace}, {ca.assignment}")
                    setattr(target, ca.name, getattr(source, ca.name))
            except Exception as e:
                print(f"*** Error copying custom attribute: {e}")


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
    num_dofs = joint_pos_initial.shape[1]
    joint_pos_current = torch.tensor(joint_q[7:7 + num_dofs], device=device, dtype=torch.float32).unsqueeze(0)
    joint_vel_current = torch.tensor(joint_qd[6:6 + num_dofs], device=device, dtype=torch.float32).unsqueeze(0)

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


class _GamepadInput:
    """Optional Xbox gamepad input with keyboard fallback.

    Tries to connect an Xbox 360/One controller on initialisation.  When
    none is found, falls back to viewer keyboard (same keys as before).

    Gamepad camera mapping::

        Right stick X/Y   orbit / rotate camera (both follow and free modes)
        D-pad up/down     dolly forward / back   (free mode only)
        D-pad left/right  pan left / right       (free mode only)
        Y button          toggle follow cam      (keyboard: x)
        B button          snap camera behind     (keyboard: shift)
        A button          snap camera to front   (keyboard: ctrl)
    """

    _DEADBAND         = 0.2
    _ORBIT_YAW_RATE   = 120.0  # deg/s — right stick X
    _ORBIT_PITCH_RATE =  80.0  # deg/s — right stick Y
    _DOLLY_SPEED      =   3.0  # m/s   — D-pad up/down
    _PAN_SPEED        =   3.0  # m/s   — D-pad left/right

    def __init__(self, viewer=None) -> None:
        self._viewer = viewer
        self._controller = None
        self._mode: str | None = None  # "joystick" | "keyboard" | None
        self._reset_prev = False
        # Camera button edge-detection state
        self._follow_toggle_prev = False
        self._snap_behind_prev = False
        self._snap_front_prev = False

        try:
            from xbox360controller import Xbox360Controller  # noqa: PLC0415

            self._controller = Xbox360Controller(0, axis_threshold=0.015)
            self._mode = "joystick"
            print("[INFO] Xbox controller connected.")
        except Exception:
            if viewer is not None and hasattr(viewer, "is_key_down"):
                self._mode = "keyboard"
                print(
                    "[INFO] No gamepad found.  Using keyboard:\n"
                    "  I/K — forward/backward    J/L — strafe left/right\n"
                    "  U/O — turn left/right     P   — reset\n"
                    "  X   — toggle follow cam   Shift — snap behind   Ctrl — snap front"
                )

    @staticmethod
    def _db(v: float, threshold: float) -> float:
        return v if abs(v) > threshold else 0.0

    def read(self) -> tuple[float, float, float]:
        """Return ``(forward, lateral, angular)`` commands in ``[-1, 1]``."""
        db = self._DEADBAND
        if self._mode == "joystick":
            c = self._controller
            return (
                self._db(-c.axis_l.y, db),                          # forward
                self._db(c.trigger_l.value - c.trigger_r.value, db),  # lateral
                self._db(-c.axis_l.x, db),                          # angular
            )
        if self._mode == "keyboard":
            v = self._viewer
            fwd = 1.0 if v.is_key_down("i") else (-1.0 if v.is_key_down("k") else 0.0)
            lat = 0.5 if v.is_key_down("j") else (-0.5 if v.is_key_down("l") else 0.0)
            ang = 1.0 if v.is_key_down("u") else (-1.0 if v.is_key_down("o") else 0.0)
            return fwd, lat, ang
        return 0.0, 0.0, 0.0

    def check_reset(self) -> bool:
        """Return ``True`` on the rising edge of the reset input.

        Gamepad: Select / Back button.  Keyboard: ``p`` key.
        """
        pressed = False
        if self._mode == "joystick":
            pressed = bool(self._controller.button_select.is_pressed)
            # Also allow keyboard 'p' when a gamepad is connected
            if not pressed and self._viewer is not None and hasattr(self._viewer, "is_key_down"):
                pressed = bool(self._viewer.is_key_down("p"))
        elif self._mode == "keyboard" and self._viewer is not None:
            pressed = bool(self._viewer.is_key_down("p"))
        triggered = pressed and not self._reset_prev
        self._reset_prev = pressed
        return triggered

    def read_camera_orbit(self, dt: float) -> tuple[float, float]:
        """Return ``(dyaw_deg, dpitch_deg)`` from right stick scaled by *dt*.

        Positive dyaw  → orbit / rotate camera right.
        Positive dpitch → tilt camera up.
        Returns ``(0, 0)`` in keyboard mode (mouse drag handles orbit there).
        """
        if self._mode != "joystick":
            return 0.0, 0.0
        c = self._controller
        dyaw   = self._db( c.axis_r.x, self._DEADBAND) * self._ORBIT_YAW_RATE   * dt
        dpitch = self._db(-c.axis_r.y, self._DEADBAND) * self._ORBIT_PITCH_RATE * dt
        return dyaw, dpitch

    def read_camera_move(self, dt: float) -> tuple[float, float]:
        """Return ``(dolly, pan)`` distances [m] from D-pad scaled by *dt*.

        Positive dolly → move forward.  Positive pan → strafe right.
        Returns ``(0, 0)`` in keyboard mode.
        """
        if self._mode != "joystick":
            return 0.0, 0.0
        c = self._controller
        dolly = float( c.hat.y) * self._DOLLY_SPEED * dt
        pan   = float( c.hat.x) * self._PAN_SPEED   * dt
        return dolly, pan

    def check_follow_toggle(self) -> bool:
        """Rising edge of follow-cam toggle: gamepad Y or keyboard ``x``."""
        pressed = False
        if self._mode == "joystick":
            pressed = bool(self._controller.button_y.is_pressed)
        if not pressed and self._viewer is not None and hasattr(self._viewer, "is_key_down"):
            pressed = bool(self._viewer.is_key_down("x"))
        triggered = pressed and not self._follow_toggle_prev
        self._follow_toggle_prev = pressed
        return triggered

    def check_snap_behind(self) -> bool:
        """Rising edge of snap-behind: gamepad B or keyboard ``shift``."""
        pressed = False
        if self._mode == "joystick":
            pressed = bool(self._controller.button_b.is_pressed)
        if not pressed and self._viewer is not None and hasattr(self._viewer, "is_key_down"):
            pressed = bool(self._viewer.is_key_down("shift"))
        triggered = pressed and not self._snap_behind_prev
        self._snap_behind_prev = pressed
        return triggered

    def check_snap_front(self) -> bool:
        """Rising edge of snap-to-front: gamepad A or keyboard ``ctrl``."""
        pressed = False
        if self._mode == "joystick":
            pressed = bool(self._controller.button_a.is_pressed)
        if not pressed and self._viewer is not None and hasattr(self._viewer, "is_key_down"):
            pressed = bool(self._viewer.is_key_down("ctrl"))
        triggered = pressed and not self._snap_front_prev
        self._snap_front_prev = pressed
        return triggered

    def close(self) -> None:
        """Release gamepad resources so the process can exit cleanly."""
        if self._controller is not None:
            try:
                self._controller.close()
            except Exception:
                pass
            self._controller = None


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
        num_balls: int = 1,
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
        self.model, builder = create_model(
            robot_config.asset_path,
            background_usd_path=background_usd_path,
            background_usd_root=background_usd_root,
            background_usd_collide=background_usd_collide,
            num_balls=num_balls,
        )

        if MODEL_HACK:
            # construct a shadow model using the old G1 asset
            shadow_model, shadow_builder = create_model(
                "usd/g1_isaac.usd",
                background_usd_path=background_usd_path,
                background_usd_root=background_usd_root,
                background_usd_collide=background_usd_collide,
                num_balls=num_balls,
            )

            # clone the parameters to make the policy work
            clone_model_parameters(shadow_model, shadow_builder, self.model, builder)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco,
            solver="newton",
            nconmax=200,
            njmax=1000,
            use_mujoco_contacts=False,
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

        # Gamepad / keyboard input (auto-detects Xbox controller, falls back to keyboard)
        self._gamepad = _GamepadInput(viewer=viewer)

        # Initialize policy-related attributes
        # (will be set by load_policy_and_setup_tensors)
        self.policy = None
        self.joint_pos_initial = None
        self.act = None
        self.rearranged_act = None

        # Track if camera has been set in first render
        self._camera_set_in_render = False

        # Third-person follow camera state
        self._follow_cam_active = False
        self._follow_cam_pos: np.ndarray | None = None  # smoothed robot position (orbit centre)
        self._follow_cam_yaw: float = 0.0               # smoothed robot yaw in radians
        self._orbit_yaw_offset: float = 0.0             # accumulated mouse-drag orbit offset (degrees)
        self._orbit_pitch_offset: float = 0.0           # accumulated mouse-drag pitch offset (degrees)
        self._orbit_last_yaw: float = 0.0               # camera.yaw stored after last set_camera call (mouse mode)
        self._orbit_last_pitch: float = 0.0             # camera.pitch stored after last set_camera call (mouse mode)

        # Force-initialize Newton's collision pipeline now, before CUDA graph capture.
        self.model.collide(self.state_0, self.contacts)

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

            self.model.collide(self.state_0, self.contacts)

            self.solver.step(self.state_0, self.state_1, self.control,
                             self.contacts, self.sim_dt)

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
        # Clear follow-camera smoothing so it snaps to the reset position.
        self._follow_cam_pos = None
        self._follow_cam_yaw = 0.0
        self._orbit_yaw_offset = 0.0
        self._orbit_pitch_offset = 0.0

    def step(self):
        # Read velocity commands from gamepad or keyboard
        fwd, lat, rot = self._gamepad.read()
        self.command[0, 0] = float(fwd)
        self.command[0, 1] = float(lat)
        self.command[0, 2] = float(rot)
        if self._gamepad.check_reset():
            self.reset()

        # Toggle follow cam: gamepad Y or keyboard 'x' (edge-triggered)
        if self._gamepad.check_follow_toggle():
            self._follow_cam_active = not self._follow_cam_active
            if not self._follow_cam_active:
                self._follow_cam_pos = None
                self._follow_cam_yaw = 0.0
                self._orbit_yaw_offset = 0.0
                self._orbit_pitch_offset = 0.0

        # Snap camera: gamepad B/A or keyboard shift/ctrl (follow mode only, edge-triggered)
        if self._follow_cam_active:
            if self._gamepad.check_snap_behind():
                self._orbit_yaw_offset = 0.0
                self._orbit_pitch_offset = 0.0
            if self._gamepad.check_snap_front():
                self._orbit_yaw_offset = 180.0
                self._orbit_pitch_offset = 0.0

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

    # --- third-person camera constants ---
    _FOLLOW_DIST      = 3.0   # metres from robot
    _FOLLOW_HEIGHT    = 1.5   # metres above robot root
    _FOLLOW_PITCH     = -15.0 # default pitch in degrees
    _FOLLOW_POS_ALPHA = 0.05  # position smoothing (smaller = more stable)
    _FOLLOW_YAW_ALPHA = 0.02  # yaw smoothing (slow to suppress body wobble)

    def _update_follow_camera(self):
        """Orbit camera: follows the robot, mouse left-drag to orbit 360°."""
        joint_q = self.state_0.joint_q.numpy()
        rx, ry, rz = joint_q[0], joint_q[1], joint_q[2]
        qx, qy, qz, qw = joint_q[3], joint_q[4], joint_q[5], joint_q[6]

        robot_yaw_rad = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        if self._follow_cam_pos is None:
            # First frame: snap to robot position and zero orbit offset
            self._follow_cam_pos = np.array([rx, ry, rz], dtype=np.float32)
            self._follow_cam_yaw = robot_yaw_rad
            self._orbit_yaw_offset = 0.0
            self._orbit_pitch_offset = 0.0
            if hasattr(self.viewer, "camera"):
                self._orbit_last_yaw = self.viewer.camera.yaw
                self._orbit_last_pitch = self.viewer.camera.pitch
        else:
            # Accumulate orbit offsets: gamepad right stick OR mouse drag
            if self._gamepad._mode == "joystick":
                dyaw, dpitch = self._gamepad.read_camera_orbit(self.frame_dt)
                self._orbit_yaw_offset += dyaw
                self._orbit_pitch_offset = max(-70.0, min(70.0, self._orbit_pitch_offset + dpitch))
            elif hasattr(self.viewer, "camera"):
                dyaw = (self.viewer.camera.yaw - self._orbit_last_yaw + 180.0) % 360.0 - 180.0
                dpitch = self.viewer.camera.pitch - self._orbit_last_pitch
                self._orbit_yaw_offset += dyaw
                self._orbit_pitch_offset = max(-70.0, min(70.0, self._orbit_pitch_offset + dpitch))

            # Smooth-track robot position and heading
            dyaw = (robot_yaw_rad - self._follow_cam_yaw + np.pi) % (2 * np.pi) - np.pi
            self._follow_cam_yaw += self._FOLLOW_YAW_ALPHA * dyaw
            target = np.array([rx, ry, rz], dtype=np.float32)
            self._follow_cam_pos += self._FOLLOW_POS_ALPHA * (target - self._follow_cam_pos)

        # Camera sits at orbit angle around the smoothed robot position
        cam_orbit_rad = self._follow_cam_yaw + np.radians(self._orbit_yaw_offset)
        cam_pos = wp.vec3(
            float(self._follow_cam_pos[0] - self._FOLLOW_DIST * np.cos(cam_orbit_rad)),
            float(self._follow_cam_pos[1] - self._FOLLOW_DIST * np.sin(cam_orbit_rad)),
            float(self._follow_cam_pos[2] + self._FOLLOW_HEIGHT),
        )
        cam_pitch = self._FOLLOW_PITCH + self._orbit_pitch_offset
        cam_yaw_deg = float(np.degrees(cam_orbit_rad))

        self.viewer.set_camera(cam_pos, cam_pitch, cam_yaw_deg)

        # Store normalised camera angles for next-frame mouse-delta computation
        if hasattr(self.viewer, "camera") and self._gamepad._mode != "joystick":
            self._orbit_last_yaw = self.viewer.camera.yaw
            self._orbit_last_pitch = self.viewer.camera.pitch

    def _update_free_camera_gamepad(self):
        """Drive free camera with gamepad right stick (rotate) and D-pad (dolly/pan)."""
        if self._gamepad._mode != "joystick" or not hasattr(self.viewer, "camera"):
            return

        dt = self.frame_dt
        dyaw, dpitch = self._gamepad.read_camera_orbit(dt)
        dolly, pan   = self._gamepad.read_camera_move(dt)

        if dyaw == 0.0 and dpitch == 0.0 and dolly == 0.0 and pan == 0.0:
            return

        # Read current camera state
        pos      = self.viewer.camera.pos
        cx, cy, cz = float(pos.x), float(pos.y), float(pos.z)
        yaw   = float(self.viewer.camera.yaw)
        pitch = float(self.viewer.camera.pitch)

        # Apply rotation
        new_yaw   = yaw + dyaw
        new_pitch = max(-89.0, min(89.0, pitch + dpitch))

        # Dolly/pan along the camera's horizontal look direction
        yaw_rad = np.radians(new_yaw)
        fwd_x, fwd_y =  np.cos(yaw_rad),  np.sin(yaw_rad)   # forward in XY
        rgt_x, rgt_y =  np.sin(yaw_rad), -np.cos(yaw_rad)   # right in XY
        cx += dolly * fwd_x + pan * rgt_x
        cy += dolly * fwd_y + pan * rgt_y

        self.viewer.set_camera(wp.vec3(cx, cy, cz), new_pitch, new_yaw)
        from pyglet.math import Vec3 as PyVec3
        self.viewer.camera.pos   = PyVec3(cx, cy, cz)
        self.viewer.camera.pitch = new_pitch
        self.viewer.camera.yaw   = new_yaw
        if hasattr(self.viewer, "_camera_dirty"):
            self.viewer._camera_dirty = True

    def render(self):
        # Set camera on first render to ensure it's applied (free-cam mode only)
        if not self._camera_set_in_render and not self._follow_cam_active:
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
        if not self._camera_set_in_render:
            self._camera_set_in_render = True

        if self._follow_cam_active:
            self._update_follow_camera()
        else:
            self._update_free_camera_gamepad()

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
    parser.add_argument("--num-balls", type=int, default=1, help="Number of pushable balls to spawn")

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
        num_balls=args.num_balls,
    )

    # Use utility function to load policy and setup tensors
    load_policy_and_setup_tensors(example, policy_path, config["num_dofs"], slice(7, 7 + config["num_dofs"]))

    # Run using standard example loop
    try:
        newton.examples.run(example, args)
    finally:
        example._gamepad.close()
