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
# Example: Bipedal RL policy play-back
#
# Runs a trained RL walking policy on the robot using the
# Kamino solver with implicit PD joint control.  Velocity commands come
# from an Xbox gamepad or, when no gamepad is connected, from keyboard
# input via the 3-D viewer.
#
# Usage:
#   python example_rl_bipedal.py
#
# Press "x" to toggle third-person camera follow mode.
# In follow mode, left-drag to orbit; press SHIFT to snap behind, CTRL to snap to front.
###########################################################################

# Python
import argparse
import os
from collections.abc import Callable

# Thirdparty
import numpy as np
import torch  # noqa: TID253
import warp as wp

# Newton
import newton

# Kamino
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.viewer import ViewerConfig
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.examples.rl.joystick import JoystickController
from newton._src.solvers.kamino.examples.rl.observations import BipedalObservation
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim
from newton._src.solvers.kamino.examples.rl.simulation_runner import SimulationRunner
from newton._src.solvers.kamino.examples.rl.utils import _load_policy_checkpoint, quat_to_projected_yaw

# Asset directory next to this file
_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "bipedal")

# ---------------------------------------------------------------------------
# Bipedal joint normalization
# ---------------------------------------------------------------------------
# Each entry maps joint name -> (position_offset, position_scale) used to
# normalise joint positions in the observation vector.

_BIPEDAL_JOINT_NORMALIZATION = {
    "NECK_FORWARD": (1.23, 0.19),
    "NECK_PITCH": (-1.09, 0.44),
    "NECK_YAW": (0.0, 0.35),
    "NECK_ROLL": (0.0, 0.11),
    "RIGHT_HIP_YAW": (0.0, 0.26),
    "RIGHT_HIP_ROLL": (0.06, 0.32),
    "RIGHT_HIP_PITCH": (0.49, 0.75),
    "RIGHT_KNEE_PITCH": (-0.91, 0.61),
    "RIGHT_ANKLE_PITCH": (0.22, 0.66),
    "LEFT_HIP_YAW": (0.0, 0.26),
    "LEFT_HIP_ROLL": (-0.06, 0.32),
    "LEFT_HIP_PITCH": (0.49, 0.75),
    "LEFT_KNEE_PITCH": (-0.91, 0.61),
    "LEFT_ANKLE_PITCH": (0.22, 0.66),
}

_BIPEDAL_JOINT_VELOCITY_SCALE = 5.0
_BIPEDAL_PATH_DEVIATION_SCALE = 0.1
_BIPEDAL_PHASE_EMBEDDING_DIM = 4


def _build_normalization(joint_names: list[str]):
    """Build ordered (offset, scale) lists from simulator joint names."""
    offsets: list[float] = []
    scales: list[float] = []
    for name in joint_names:
        if name in _BIPEDAL_JOINT_NORMALIZATION:
            o, s = _BIPEDAL_JOINT_NORMALIZATION[name]
        else:
            msg.warning(f"Joint '{name}' not in BIPEDAL normalization dict -- using identity.")
            o, s = 0.0, 1.0
        offsets.append(o)
        scales.append(s)
    return offsets, scales


def _make_heightfield_terrain_fn(
    nrow: int = 40,
    ncol: int = 40,
    hx: float = 10.0,
    hy: float = 10.0,
    amplitude: float = 0.35,
    seed: int = 42,
):
    """Return a callback that adds a smooth heightfield terrain to a builder.

    The elevation is a sum of low-frequency sine waves — gentle enough for
    a bipedal robot to walk on yet clearly non-flat.

    Args:
        nrow: Grid rows.
        ncol: Grid columns.
        hx: Half-extent in X [m].
        hy: Half-extent in Y [m].
        amplitude: Peak-to-peak height variation [m].
        seed: RNG seed for random phase offsets.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-hx, hx, ncol)
    y = np.linspace(-hy, hy, nrow)
    xx, yy = np.meshgrid(x, y)

    elevation = np.zeros_like(xx)
    for freq in (0.4, 0.7, 1.1):
        px, py = rng.uniform(0, 2 * np.pi, size=2)
        elevation += np.sin(freq * xx + px) * np.cos(freq * yy + py)
    elevation *= amplitude / np.ptp(elevation)
    center_r, center_c = nrow // 2, ncol // 2
    elevation -= elevation[center_r, center_c]  # surface at origin == z=0 (robot feet)

    hfield = newton.Heightfield(
        data=elevation.astype(np.float32),
        nrow=nrow,
        ncol=ncol,
        hx=hx,
        hy=hy,
    )

    def _add_terrain(builder):
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.margin = 0.01
        cfg.gap = 0.02
        builder.add_shape_heightfield(heightfield=hfield, cfg=cfg)

    return _add_terrain


def _make_usd_terrain_fn(file_path: str, root_prim: str | None = None, add_ground_plane: bool = False):
    """Return a callback that adds collision geometry from a USD file to a builder.

    Args:
        file_path: The path of the USD file.
        root_prim: The root USD prim path for the collision geometry. Shapes will be loaded
            recursively from this path. If None, load shapes from the entire file.
        add_ground_plane: Add a ground plane collider for a flat walking area.
    """

    if root_prim is None:
        root_prim = "/"

    def _add_terrain(builder):
        shape_start = builder.shape_count
        builder.add_usd(file_path, root_path=root_prim, hide_collision_shapes=True, load_visual_shapes=False)
        # clear ShapeFlags.VISIBLE to hide the collision shape
        for idx in range(shape_start, builder.shape_count):
            flags = builder.shape_flags[idx]
            builder.shape_flags[idx] = flags & ~newton.ShapeFlags.VISIBLE

        if add_ground_plane:
            cfg = newton.ModelBuilder.ShapeConfig(is_visible=False)
            builder.add_ground_plane(cfg=cfg)

    return _add_terrain


def _make_usd_background_fn(file_path: str, root_prim: str | None = None):
    def _add_background(viewer):
        viewer.add_background_usd(file_path, root_prim)

    return _add_background


###########################################################################
# Scene callback - adds pushable balls
###########################################################################

BALL_RADIUS = 0.2
BALL_POSITIONS = [
    wp.vec3(0.5, 0.0, BALL_RADIUS + 0.01),
    wp.vec3(-0.6, 0.4, BALL_RADIUS + 0.01),
    wp.vec3(0.3, -0.5, BALL_RADIUS + 0.01),
    wp.vec3(-0.4, -0.3, BALL_RADIUS + 0.01),
    wp.vec3(0.7, 0.6, BALL_RADIUS + 0.01),
]


def _make_balls_fn(num_balls: int = 1):
    """Return a callback that adds *num_balls* pushable spheres."""
    positions = BALL_POSITIONS[:num_balls]

    def _add_balls(robot_builder):
        ball_cfg = newton.ModelBuilder.ShapeConfig()
        ball_cfg.density = 50.0
        ball_cfg.mu = 0.5
        for i, pos in enumerate(positions):
            ball_body = robot_builder.add_body(
                xform=wp.transform(p=pos, q=wp.quat_identity()),
                label=f"ball_{i}",
            )
            robot_builder.add_shape_sphere(ball_body, radius=BALL_RADIUS, cfg=ball_cfg)

    return _add_balls


###########################################################################
# Example class
###########################################################################


class Example:
    def __init__(
        self,
        device: wp.DeviceLike = None,
        policy=None,
        headless: bool = False,
        viewer_type: str = "gl",
        num_balls: int = 1,
        background_fn: Callable | None = None,
        terrain_fn: Callable | None = None,
    ):
        # Timing
        self.sim_dt = 0.02
        self.control_decimation = 1
        num_worlds = 1
        self.env_dt = self.sim_dt * self.control_decimation

        # USD model path
        USD_MODEL_PATH = os.path.join(_ASSETS_DIR, "bdx_merged.usda")

        # Create generic articulated body simulator with rolling terrain
        self.sim_wrapper = RigidBodySim(
            usd_model_path=USD_MODEL_PATH,
            num_worlds=1,
            sim_dt=self.sim_dt,
            device=device,
            headless=headless,
            body_pose_offset=(0.0, 0.0, 0.33, 0.0, 0.0, 0.0, 1.0),
            use_cuda_graph=True,
            render_config=ViewerConfig(
                diffuse_scale=1.0,
                specular_scale=0.3,
                shadow_radius=10.0,
            ),
            viewer_type=viewer_type,
            background_fn=background_fn,
            terrain_fn=terrain_fn,
            scene_callback=_make_balls_fn(num_balls),
        )

        # Override PD gains
        self.sim_wrapper.sim.model.joints.k_p_j.fill_(15.0)
        self.sim_wrapper.sim.model.joints.k_d_j.fill_(0.6)
        self.sim_wrapper.sim.model.joints.a_j.fill_(0.004)
        self.sim_wrapper.sim.model.joints.b_j.fill_(0.0)

        # Build normalization from actuated joints only (excludes passive free joints
        # such as the ball, which should not feed into the RL policy).
        joint_pos_offset, joint_pos_scale = _build_normalization(self.sim_wrapper.actuated_joint_names)
        self.joint_pos_offset = torch.tensor(joint_pos_offset, device=self.torch_device)
        self.joint_pos_scale = torch.tensor(joint_pos_scale, device=self.torch_device)
        self._act_idx = self.sim_wrapper.actuated_dof_indices_tensor

        # Observation builder
        self.obs = BipedalObservation(
            body_sim=self.sim_wrapper,
            joint_position_default=joint_pos_offset,
            joint_position_range=joint_pos_scale,
            joint_velocity_scale=_BIPEDAL_JOINT_VELOCITY_SCALE,
            path_deviation_scale=_BIPEDAL_PATH_DEVIATION_SCALE,
            phase_embedding_dim=_BIPEDAL_PHASE_EMBEDDING_DIM,
            phase_rate_policy_path=PHASE_RATE_POLICY_PATH,
            dt=self.env_dt,
            num_joints=len(self.joint_pos_offset),
        )
        msg.info(f"Observation dim: {self.obs.num_observations}")

        # Joystick / keyboard command controller
        self.joystick = JoystickController(
            dt=self.env_dt,
            viewer=self.sim_wrapper.viewer,
            num_worlds=num_worlds,
            device=self.torch_device,
        )
        # Initialize path to current robot pose
        root_pos_2d = self.sim_wrapper.q_i[:, 0, :2]
        root_yaw = quat_to_projected_yaw(self.sim_wrapper.q_i[:, 0, 3:])
        self.joystick.reset(root_pos_2d=root_pos_2d, root_yaw=root_yaw)

        # Action buffer (actuated joints only)
        self.actions = self.sim_wrapper.q_j[:, self._act_idx].clone()

        # Pre-allocated command buffers (eliminates per-step torch.tensor())
        self._cmd_vel_buf = torch.zeros(1, 2, device=self.torch_device)
        self._neck_cmd_buf = torch.zeros(4, device=self.torch_device)

        # Policy (None = zero actions)
        self.policy = policy

        # Third-person follow camera state
        self._follow_cam_active = False
        self._follow_cam_pos: np.ndarray | None = None  # smoothed robot position (orbit centre)
        self._follow_cam_yaw: float = 0.0               # smoothed robot yaw in radians
        self._orbit_yaw_offset: float = 0.0
        self._orbit_pitch_offset: float = 0.0
        self._orbit_last_yaw: float = 0.0
        self._orbit_last_pitch: float = 0.0

    # Convenience accessors for the main block
    @property
    def torch_device(self) -> str:
        return self.sim_wrapper.torch_device

    @property
    def viewer(self):
        return self.sim_wrapper.viewer

    def reset(self):
        """Reset the simulation and internal state."""
        self.sim_wrapper.reset()
        self.obs.reset()
        root_pos_2d = self.sim_wrapper.q_i[:, 0, :2]
        root_yaw = quat_to_projected_yaw(self.sim_wrapper.q_i[:, 0, 3:])
        self.joystick.reset(root_pos_2d=root_pos_2d, root_yaw=root_yaw)
        self.actions[:] = self.sim_wrapper.q_j[:, self._act_idx]
        # Reset camera to initial state.
        self._follow_cam_active = False
        self._follow_cam_pos = None
        self._follow_cam_yaw = 0.0
        self._orbit_yaw_offset = 0.0
        self._orbit_pitch_offset = 0.0
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(self._INIT_CAM_POS, self._INIT_CAM_PITCH, self._INIT_CAM_YAW)

    def step_once(self):
        """Single physics step (used by run_headless warm-up)."""
        self.sim_wrapper.step()

    def update_input(self):
        """Transfer joystick commands to the observation command tensor."""
        cmd = self.obs.command
        cmd[:, BipedalObservation.CMD_PATH_HEADING] = self.joystick.path_heading[:, 0]
        cmd[:, BipedalObservation.CMD_PATH_POSITION] = self.joystick.path_position
        self._cmd_vel_buf[0, 0] = self.joystick.forward_velocity
        self._cmd_vel_buf[0, 1] = self.joystick.lateral_velocity
        cmd[:, BipedalObservation.CMD_VEL] = self._cmd_vel_buf
        cmd[:, BipedalObservation.CMD_YAW_RATE] = self.joystick.angular_velocity

        # Head command: head_forward is an up-bias coupled to head pitch
        # (looking up also raises the head). head_pitch = forward + pitch.
        js = self.joystick
        head_forward = max(js.head_pitch, 0.0) * 0.4
        head_z_des = max(-1.0, min(head_forward, 0.3))
        head_roll_des = 0.0
        head_pitch_des = max(-0.6, min(head_forward + js.head_pitch, 1.0))
        head_yaw_des = max(-1.0, min(js.head_yaw, 1.0))
        self._neck_cmd_buf[0] = head_z_des
        self._neck_cmd_buf[1] = head_roll_des
        self._neck_cmd_buf[2] = head_pitch_des
        self._neck_cmd_buf[3] = head_yaw_des
        cmd[:, BipedalObservation.CMD_HEAD] = self._neck_cmd_buf

    def sim_step(self):
        """Observations -> policy inference -> actions -> physics step."""
        # Compute observation from current state (with previous setpoints)
        obs = self.obs.compute(setpoints=self.actions)

        # Policy inference (in-place: no clone, no intermediates)
        with torch.inference_mode():
            raw = self.policy(obs)
            torch.mul(raw, self.joint_pos_scale, out=self.actions)
            self.actions.add_(self.joint_pos_offset)

        # Write action targets to actuated joints only
        self.sim_wrapper.q_j_ref[:, self._act_idx] = self.actions

        # Step physics
        for _ in range(self.control_decimation):
            self.sim_wrapper.step()

    def step(self):
        """One RL step: commands -> observe -> infer -> apply -> simulate."""
        if self.joystick.check_reset():
            self.reset()
        self.joystick.update(root_pos_2d=self.sim_wrapper.q_i[:, 0, :2])
        self.update_input()
        self.sim_step()

        # Toggle follow cam: gamepad Y or keyboard 'x' (edge-triggered)
        if self.joystick.check_follow_toggle():
            self._follow_cam_active = not self._follow_cam_active
            if not self._follow_cam_active:
                self._follow_cam_pos = None
                self._follow_cam_yaw = 0.0
                self._orbit_yaw_offset = 0.0
                self._orbit_pitch_offset = 0.0

        # Snap camera: gamepad A/B or keyboard shift/ctrl (follow mode only, edge-triggered)
        if self._follow_cam_active:
            if self.joystick.check_snap_behind():
                self._orbit_yaw_offset = 0.0
                self._orbit_pitch_offset = 0.0
            if self.joystick.check_snap_front():
                self._orbit_yaw_offset = 180.0
                self._orbit_pitch_offset = 0.0

    # --- third-person camera constants ---
    _FOLLOW_DIST      = 3.0
    _FOLLOW_HEIGHT    = 1.5
    _FOLLOW_PITCH     = -15.0
    _FOLLOW_POS_ALPHA = 0.05
    _FOLLOW_YAW_ALPHA = 0.02

    # --- initial free-camera defaults (must match the set_camera call in __main__) ---
    _INIT_CAM_POS   = wp.vec3(2.5, 1.5, 1.0)
    _INIT_CAM_PITCH = -10.0
    _INIT_CAM_YAW   = 225.0

    def _update_follow_camera(self):
        """Orbit camera: follows the robot, mouse left-drag to orbit 360°."""
        q = self.sim_wrapper.q_i[0, 0]   # (7,) tensor: xyz + xyzw quat
        rx, ry, rz = q[0].item(), q[1].item(), q[2].item()
        robot_yaw_rad = quat_to_projected_yaw(q[3:].unsqueeze(0))[0, 0].item()

        if self._follow_cam_pos is None:
            self._follow_cam_pos = np.array([rx, ry, rz], dtype=np.float32)
            self._follow_cam_yaw = robot_yaw_rad
            self._orbit_yaw_offset = 0.0
            self._orbit_pitch_offset = 0.0
            if hasattr(self.viewer, "camera"):
                self._orbit_last_yaw = self.viewer.camera.yaw
                self._orbit_last_pitch = self.viewer.camera.pitch
        else:
            # Accumulate mouse-drag deltas as orbit offsets
            if hasattr(self.viewer, "camera"):
                dyaw = (self.viewer.camera.yaw - self._orbit_last_yaw + 180.0) % 360.0 - 180.0
                dpitch = self.viewer.camera.pitch - self._orbit_last_pitch
                self._orbit_yaw_offset += dyaw
                self._orbit_pitch_offset = max(-70.0, min(70.0, self._orbit_pitch_offset + dpitch))

            # Smooth-track robot position and heading
            dyaw = (robot_yaw_rad - self._follow_cam_yaw + np.pi) % (2 * np.pi) - np.pi
            self._follow_cam_yaw += self._FOLLOW_YAW_ALPHA * dyaw
            target = np.array([rx, ry, rz], dtype=np.float32)
            self._follow_cam_pos += self._FOLLOW_POS_ALPHA * (target - self._follow_cam_pos)

        cam_orbit_rad = self._follow_cam_yaw + np.radians(self._orbit_yaw_offset)
        cam_pos = wp.vec3(
            float(self._follow_cam_pos[0] - self._FOLLOW_DIST * np.cos(cam_orbit_rad)),
            float(self._follow_cam_pos[1] - self._FOLLOW_DIST * np.sin(cam_orbit_rad)),
            float(self._follow_cam_pos[2] + self._FOLLOW_HEIGHT),
        )
        cam_pitch = self._FOLLOW_PITCH + self._orbit_pitch_offset
        cam_yaw_deg = float(np.degrees(cam_orbit_rad))

        self.viewer.set_camera(cam_pos, cam_pitch, cam_yaw_deg)

        if hasattr(self.viewer, "camera"):
            self._orbit_last_yaw = self.viewer.camera.yaw
            self._orbit_last_pitch = self.viewer.camera.pitch

    def render(self):
        """Render the current frame."""
        if self._follow_cam_active:
            self._update_follow_camera()
        self.sim_wrapper.render()


###########################################################################
# Main
###########################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bipedal RL play example")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run in headless mode",
    )
    parser.add_argument(
        "--viewer",
        type=str,
        default="gl",
        choices=["gl", "rtx"],
        help="Viewer type: gl (OpenGL) or rtx (ray-traced)",
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "async"],
        default="sync",
        help="Sim loop mode: sync (default) or async",
    )
    parser.add_argument(
        "--num-balls",
        type=int,
        default=1,
        help="Number of balls to add to the scene (max 5, default: 1)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Use a flat ground plane instead of the heightfield terrain",
    )
    parser.add_argument(
        "--usd-background",
        help="USD file to load as background",
    )
    parser.add_argument(
        "--usd-background-root",
        help="Root path of the background geometry in the USD background file",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=30.0,
        help="Target render FPS for async mode (default: 30)",
    )
    args = parser.parse_args()

    np.set_printoptions(linewidth=20000, precision=6, threshold=10000, suppress=True)
    msg.set_log_level(msg.LogLevel.INFO)

    if args.device:
        device = wp.get_device(args.device)
        wp.set_device(device)
    else:
        device = wp.get_preferred_device()

    msg.info(f"device: {device}")

    # Convert warp device to torch device string for checkpoint loading
    torch_device = "cuda" if device.is_cuda else "cpu"

    # Load trained policy
    POLICY_PATH = os.path.join(_ASSETS_DIR, "model.pt")
    PHASE_RATE_POLICY_PATH = os.path.join(_ASSETS_DIR, "phase_rate.pt")
    policy = _load_policy_checkpoint(POLICY_PATH, device=torch_device)
    msg.info(f"Loaded policy from: {POLICY_PATH}")

    terrain_fn = _make_heightfield_terrain_fn()
    background_fn = None
    if args.flat:
        terrain_fn = None
    elif args.usd_background:
        terrain_fn = _make_usd_terrain_fn(args.usd_background, args.usd_background_root, add_ground_plane=True)
        background_fn = _make_usd_background_fn(args.usd_background, args.usd_background_root)

    example = Example(
        device=device,
        policy=policy,
        headless=args.headless,
        viewer_type=args.viewer,
        num_balls=args.num_balls,
        background_fn=background_fn,
        terrain_fn=terrain_fn,
    )

    try:
        if args.headless:
            msg.notif("Running in headless mode...")
            run_headless(example, progress=True)
        else:
            msg.notif(f"Running in Viewer mode ({args.mode})...")
            if hasattr(example.viewer, "set_camera"):
                example.viewer.set_camera(Example._INIT_CAM_POS, Example._INIT_CAM_PITCH, Example._INIT_CAM_YAW)
            SimulationRunner(example, mode=args.mode, render_fps=args.render_fps).run()
    except KeyboardInterrupt:
        pass
    finally:
        example.joystick.close()
        if example.viewer is not None:
            example.viewer.close()
