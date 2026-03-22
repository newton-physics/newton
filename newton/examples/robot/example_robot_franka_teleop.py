# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Franka Teleop
#
# Interactive teleoperation of a Franka Emika Panda arm with a gripper.
# Use Ctrl+QWEASD for keyboard jog, and the popup UI for preset
# grasp actions.
#
# Command: python -m newton.examples robot_franka_teleop
#
###########################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyglet
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.solvers
import newton.utils
import newton.viewer
from newton import JointTargetMode

# =========================================================================
# Config
# =========================================================================


@dataclass(frozen=True)
class Config:
    fps: int = 60
    sim_substeps: int = 4
    ui_translate_step: float = 0.01
    ui_rotate_step_deg: float = 5.0
    min_target_height: float = 0.10
    gripper_min: float = 0.0
    gripper_max: float = 0.04
    grasp_close_q: float = 0.02
    arm_coord_count: int = 7
    finger_coord_indices: tuple[int, int] = (7, 8)
    table_top_center: tuple[float, float, float] = (0.62, 0.0, 0.34)
    table_top_half_extents: tuple[float, float, float] = (0.28, 0.34, 0.04)
    cube_half_extent: float = 0.03
    cube_spawn_center: tuple[float, float, float] = (0.62, 0.0, 0.46)
    initial_target_euler_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)
    grasp_hover_clearance: float = 0.12
    grasp_target_z_offset: float = 0.06
    grasp_lift_height: float = 0.10
    gripper_speed: float = 0.08
    target_linear_speed: float = 0.35
    target_angular_speed_deg: float = 180.0
    mujoco_njmax: int = 256
    mujoco_nconmax: int = 128
    robot_contact_mu: float = 1.0
    robot_contact_mu_torsional: float = 0.2
    robot_contact_mu_rolling: float = 0.02
    table_contact_mu: float = 2.4
    table_contact_kd: float = 180.0
    table_contact_ke: float = 4.0e3
    table_contact_mu_torsional: float = 0.2
    table_contact_mu_rolling: float = 0.1
    cube_contact_mu: float = 1.0
    cube_contact_kd: float = 120.0
    cube_contact_ke: float = 3.0e3
    cube_contact_density: float = 250.0
    cube_contact_mu_torsional: float = 0.5
    cube_contact_mu_rolling: float = 0.5


# =========================================================================
# Scene builder
# =========================================================================


def build_scene(builder: newton.ModelBuilder, cfg: Config) -> int:
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = cfg.robot_contact_mu
    builder.default_shape_cfg.mu_torsional = cfg.robot_contact_mu_torsional
    builder.default_shape_cfg.mu_rolling = cfg.robot_contact_mu_rolling
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
    )
    robot_joint_count = len(builder.joint_target_mode)
    builder.add_ground_plane()

    table_cfg = newton.ModelBuilder.ShapeConfig(
        mu=cfg.table_contact_mu,
        kd=cfg.table_contact_kd,
        ke=cfg.table_contact_ke,
        density=0.0,
        mu_torsional=cfg.table_contact_mu_torsional,
        mu_rolling=cfg.table_contact_mu_rolling,
    )
    cube_cfg = newton.ModelBuilder.ShapeConfig(
        mu=cfg.cube_contact_mu,
        kd=cfg.cube_contact_kd,
        ke=cfg.cube_contact_ke,
        density=cfg.cube_contact_density,
        mu_torsional=cfg.cube_contact_mu_torsional,
        mu_rolling=cfg.cube_contact_mu_rolling,
    )

    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(*cfg.table_top_center), wp.quat_identity()),
        hx=cfg.table_top_half_extents[0],
        hy=cfg.table_top_half_extents[1],
        hz=cfg.table_top_half_extents[2],
        cfg=table_cfg,
        label="table_top",
    )
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(0.62, 0.0, 0.12), wp.quat_identity()),
        hx=0.09,
        hy=0.12,
        hz=0.18,
        cfg=table_cfg,
        label="table_pedestal",
    )
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(0.62, 0.0, -0.06), wp.quat_identity()),
        hx=0.18,
        hy=0.22,
        hz=0.02,
        cfg=table_cfg,
        label="table_base",
    )

    cube_body = builder.add_link(
        xform=wp.transform(wp.vec3(*cfg.cube_spawn_center), wp.quat_identity()), mass=0.0, label="table_cube"
    )
    cube_joint = builder.add_joint_free(cube_body, label="table_cube_joint")
    builder.add_articulation([cube_joint], label="table_cube_articulation")
    builder.add_shape_box(
        cube_body,
        hx=cfg.cube_half_extent,
        hy=cfg.cube_half_extent,
        hz=cfg.cube_half_extent,
        cfg=cube_cfg,
        label="table_cube_shape",
    )
    return robot_joint_count


# =========================================================================
# ViewerLW — overrides keyboard so Ctrl+keys control the robot
# =========================================================================


class ViewerLW(newton.viewer.ViewerGL):
    def is_ctrl_key(self, key: str | int) -> bool:
        if isinstance(key, int):
            return key in (pyglet.window.key.LCTRL, pyglet.window.key.RCTRL)
        return key.lower() in {"ctrl", "lctrl", "rctrl", "left_ctrl", "right_ctrl"}

    def is_ctrl_active(self, modifiers: int | None = None) -> bool:
        if modifiers is not None and modifiers & pyglet.window.key.MOD_CTRL:
            return True
        try:
            return self.renderer.is_key_down(pyglet.window.key.LCTRL) or self.renderer.is_key_down(
                pyglet.window.key.RCTRL
            )
        except Exception:
            return False

    def is_key_down_raw(self, key: str | int) -> bool:
        return super().is_key_down(key)

    def on_key_press(self, symbol: int, modifiers: int):
        if not self.is_ctrl_active(modifiers):
            super().on_key_press(symbol, modifiers)

    def on_key_release(self, symbol: int, modifiers: int):
        if not self.is_ctrl_active(modifiers):
            super().on_key_release(symbol, modifiers)

    def is_key_down(self, key: str | int) -> bool:
        if self.is_ctrl_active() and not self.is_ctrl_key(key):
            return False
        return super().is_key_down(key)

    def _update_camera(self, dt: float):
        if self.is_ctrl_active():
            self._cam_vel[:] = 0.0
            return
        super()._update_camera(dt)


def init_viewer(parser=None):
    """Initialize viewer, using ViewerLW for the GL backend."""
    if parser is None:
        parser = newton.examples.create_parser()
        args = parser.parse_known_args()[0]
    else:
        args = parser.parse_args()
    if args.quiet:
        wp.config.quiet = True
    if args.device:
        wp.set_device(args.device)

    if args.viewer == "gl":
        viewer = ViewerLW(headless=args.headless)
    elif args.viewer == "usd":
        if args.output_path is None:
            raise ValueError("--output-path is required when using usd viewer")
        viewer = newton.viewer.ViewerUSD(output_path=args.output_path, num_frames=args.num_frames)
    elif args.viewer == "rerun":
        viewer = newton.viewer.ViewerRerun(address=args.rerun_address)
    elif args.viewer == "null":
        viewer = newton.viewer.ViewerNull(num_frames=args.num_frames)
    elif args.viewer == "viser":
        viewer = newton.viewer.ViewerViser()
    else:
        raise ValueError(f"Invalid viewer: {args.viewer}")
    return viewer, args


# =========================================================================
# Example
# =========================================================================


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.config = cfg = Config()

        self.frame_dt = 1.0 / cfg.fps
        self.sim_dt = self.frame_dt / cfg.sim_substeps
        self.sim_time = 0.0
        self.ui_translate_step = cfg.ui_translate_step
        self.ui_rotate_step_deg = cfg.ui_rotate_step_deg
        self.min_target_height = cfg.min_target_height
        self.gripper_min = cfg.gripper_min
        self.gripper_max = cfg.gripper_max
        self.gripper_q = cfg.gripper_max
        self.active_gripper_q = cfg.gripper_max
        self.grasp_close_q = cfg.grasp_close_q
        self.gripper_speed = cfg.gripper_speed
        self.cube_half_extent = cfg.cube_half_extent
        self.grasp_hover_clearance = cfg.grasp_hover_clearance
        self.grasp_target_z_offset = cfg.grasp_target_z_offset
        self.grasp_lift_height = cfg.grasp_lift_height
        self.target_linear_speed = cfg.target_linear_speed
        self.target_angular_speed_deg = cfg.target_angular_speed_deg
        self.arm_coord_count = cfg.arm_coord_count
        self.finger_coord_indices = list(cfg.finger_coord_indices)

        # Build scene
        builder = newton.ModelBuilder()
        robot_joint_count = build_scene(builder, cfg)
        for j in range(robot_joint_count):
            builder.joint_target_mode[j] = int(JointTargetMode.POSITION)
            builder.joint_target_ke[j] = 600.0
            builder.joint_target_kd[j] = 80.0

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=cfg.mujoco_njmax, nconmax=cfg.mujoco_nconmax)
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.0, 0, 0.7), pitch=0.0, yaw=-180.0)

        for s in (self.state_0, self.state_1):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, s)
        self.setup_ik()
        self.run_ik()
        for s in (self.state_0, self.state_1):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, s)

        self.home_joint_q = wp.clone(self.model.joint_q)
        self.home_joint_qd = wp.clone(self.state_0.joint_qd)

        # Initial control targets
        wp.copy(self.control.joint_target_pos[: self.arm_coord_count], self.model.joint_q[: self.arm_coord_count])
        for idx in self.finger_coord_indices:
            if idx < self.control.joint_target_pos.shape[0]:
                self.control.joint_target_pos[idx : idx + 1].fill_(self.gripper_q)

        self.cube_body_index = self.find_body("table_cube")

        # Register popup UI
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_popup_ui, "free")

        print("\nCtrl+QWEASD: translate, Ctrl+UIOJKL: rotate, Ctrl+ZX: gripper")

    # ----- Body helpers --------------------------------------------------

    def find_body(self, suffix: str) -> int:
        for i, label in enumerate(self.model.body_label):
            if label.endswith(f"/{suffix}") or label == suffix:
                return i
        raise ValueError(f"Body not found: {suffix}")

    def get_cube_center(self) -> np.ndarray:
        return self.state_0.body_q.numpy()[self.cube_body_index, :3].astype(np.float32)

    # ----- Target pose management ----------------------------------------

    @staticmethod
    def to_pose(pos, quat: wp.quat, min_z):
        pos = np.array(pos, dtype=np.float32, copy=True)
        pos[2] = max(min_z, float(pos[2]))
        q = wp.normalize(quat)
        return pos, q, wp.vec3(*pos.tolist())

    def set_target_pose(self, pos: np.ndarray, quat: wp.quat) -> None:
        self._target_pos_np, self.target_quat, self.target_pos = self.to_pose(pos, quat, self.min_target_height)
        q = self.target_quat
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        self.target_euler_deg = np.rad2deg(
            np.array(
                [
                    np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
                    float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))),
                    np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
                ],
                dtype=np.float64,
            )
        ).astype(np.float32)

    def set_active_target(self, pos: np.ndarray, quat: wp.quat) -> None:
        self.active_target_pos_np, self.active_target_quat, self.active_target_pos = self.to_pose(
            pos, quat, self.min_target_height
        )

    def snap_active_to_command(self) -> None:
        self.set_active_target(self._target_pos_np, self.target_quat)

    def advance_active_target(self) -> None:
        delta = self._target_pos_np - self.active_target_pos_np
        dist = float(np.linalg.norm(delta))
        max_step = self.target_linear_speed * self.frame_dt
        next_pos = (
            self._target_pos_np
            if dist <= max(max_step, 1e-6)
            else (self.active_target_pos_np + delta * (max_step / dist))
        )
        q0 = wp.normalize(self.active_target_quat)
        q1 = wp.normalize(self.target_quat)
        dot = abs(float(q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]))
        angle = 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))
        max_ang = np.deg2rad(self.target_angular_speed_deg) * self.frame_dt
        next_quat = q1 if angle <= max(max_ang, 1e-6) else wp.quat_slerp(q0, q1, max_ang / angle)
        self.set_active_target(next_pos, next_quat)

    def reset_target_pose(self) -> None:
        self.set_target_pose(self.home_target_pos_np, self.home_target_quat)
        self.snap_active_to_command()

    def apply_translate(self, dx=0.0, dy=0.0, dz=0.0) -> None:
        pos = self._target_pos_np.copy()
        pos += np.array([dx, dy, dz], dtype=np.float32)
        self.set_target_pose(pos, self.target_quat)

    def apply_rotate(self, droll=0.0, dpitch=0.0, dyaw=0.0) -> None:
        euler = self.target_euler_deg.copy()
        euler += np.array([droll, dpitch, dyaw], dtype=np.float32)
        r, p, y = np.deg2rad(euler.astype(np.float64))
        self.set_target_pose(self._target_pos_np, wp.quat_rpy(float(r), float(p), float(y)))

    # ----- IK setup & step -----------------------------------------------

    def setup_ik(self) -> None:
        self.hand_index = self.find_body("fr3_hand")
        tcp_index = self.find_body("fr3_hand_tcp")

        tmp = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, tmp)
        bq = tmp.body_q.numpy()

        hand_pos = bq[self.hand_index, :3].astype(np.float32)
        hand_quat = bq[self.hand_index, 3:7].astype(np.float32)
        tcp_pos = bq[tcp_index, :3].astype(np.float32)
        tcp_quat = bq[tcp_index, 3:7].astype(np.float32)

        hand_wq = wp.quat(float(hand_quat[0]), float(hand_quat[1]), float(hand_quat[2]), float(hand_quat[3]))
        hand_inv = wp.quat_inverse(hand_wq)
        offset_vec = wp.vec3(*(tcp_pos - hand_pos).tolist())
        self.tool_offset_pos = wp.quat_rotate(hand_inv, offset_vec)
        tcp_wq = wp.quat(float(tcp_quat[0]), float(tcp_quat[1]), float(tcp_quat[2]), float(tcp_quat[3]))
        self.tool_offset_quat = hand_inv * tcp_wq

        r, p, y = np.deg2rad(np.array(self.config.initial_target_euler_deg, dtype=np.float64))
        self.set_target_pose(tcp_pos, wp.quat_rpy(float(r), float(p), float(y)))
        self.snap_active_to_command()
        self.home_target_pos_np = self._target_pos_np.copy()
        self.home_target_quat = self.target_quat
        self.grasp_target_quat = self.home_target_quat

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.hand_index,
            link_offset=self.tool_offset_pos,
            target_positions=wp.array([self.active_target_pos], dtype=wp.vec3),
        )
        tq = self.target_quat
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.hand_index,
            link_offset_rotation=self.tool_offset_quat,
            target_rotations=wp.array([wp.vec4(tq[0], tq[1], tq[2], tq[3])], dtype=wp.vec4),
        )
        self.jlimit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        self.joint_q_ik = self.model.joint_q.reshape((1, self.model.joint_coord_count))
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.jlimit_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def run_ik(self) -> None:
        self.pos_obj.set_target_position(0, self.active_target_pos)
        aq = self.active_target_quat
        self.rot_obj.set_target_rotation(0, wp.vec4(aq[0], aq[1], aq[2], aq[3]))
        self.ik_solver.step(self.joint_q_ik, self.joint_q_ik, iterations=28)

    # ----- Grasp actions -------------------------------------------------

    def move_gripper_above_cube(self) -> None:
        self.gripper_q = self.gripper_max
        hover = self.get_cube_center().copy()
        hover[2] += self.grasp_target_z_offset + self.cube_half_extent + self.grasp_hover_clearance
        self.set_target_pose(hover, self.grasp_target_quat)

    def move_gripper_to_cube(self) -> None:
        grasp = self.get_cube_center().copy()
        grasp[2] += self.grasp_target_z_offset
        self.set_target_pose(grasp, self.grasp_target_quat)

    def execute_grasp(self) -> None:
        self.gripper_q = self.grasp_close_q

    def lift_grasped_object(self) -> None:
        lift = self._target_pos_np.copy()
        lift[2] += self.grasp_lift_height
        self.set_target_pose(lift, self.grasp_target_quat)

    def reset_scene(self) -> None:
        self.gripper_q = self.gripper_max
        self.reset_target_pose()
        self.model.joint_q.assign(self.home_joint_q)
        for s in (self.state_0, self.state_1):
            s.joint_q.assign(self.home_joint_q)
            s.joint_qd.assign(self.home_joint_qd)
            newton.eval_fk(self.model, s.joint_q, s.joint_qd, s)
        wp.copy(self.control.joint_target_pos[: self.arm_coord_count], self.home_joint_q[: self.arm_coord_count])
        self.active_gripper_q = self.gripper_q
        for idx in self.finger_coord_indices:
            if idx < self.control.joint_target_pos.shape[0]:
                self.control.joint_target_pos[idx : idx + 1].fill_(self.active_gripper_q)

    # ----- Keyboard teleop (Ctrl+key) ------------------------------------

    def handle_keyboard(self) -> None:
        if not isinstance(self.viewer, ViewerLW):
            return
        if hasattr(self.viewer, "ui") and self.viewer.ui and self.viewer.ui.is_capturing():
            return
        if not self.viewer.is_ctrl_active():
            return

        s = self.ui_translate_step
        r = self.ui_rotate_step_deg

        for key, fn, kw in [
            ("w", self.apply_translate, {"dx": s}),
            ("s", self.apply_translate, {"dx": -s}),
            ("a", self.apply_translate, {"dy": s}),
            ("d", self.apply_translate, {"dy": -s}),
            ("q", self.apply_translate, {"dz": s}),
            ("e", self.apply_translate, {"dz": -s}),
            ("u", self.apply_rotate, {"droll": r}),
            ("o", self.apply_rotate, {"droll": -r}),
            ("i", self.apply_rotate, {"dpitch": r}),
            ("k", self.apply_rotate, {"dpitch": -r}),
            ("j", self.apply_rotate, {"dyaw": r}),
            ("l", self.apply_rotate, {"dyaw": -r}),
        ]:
            if self.viewer.is_key_down_raw(key):
                fn(**kw)

        if self.viewer.is_key_down_raw("z"):
            self.gripper_q = min(self.gripper_max, self.gripper_q + s)
        if self.viewer.is_key_down_raw("x"):
            self.gripper_q = max(self.gripper_min, self.gripper_q - s)

    # ----- Gripper advance -----------------------------------------------

    def advance_gripper(self) -> None:
        delta = float(self.gripper_q - self.active_gripper_q)
        max_step = self.gripper_speed * self.frame_dt
        if abs(delta) <= max_step or abs(delta) <= 1e-6:
            self.active_gripper_q = self.gripper_q
        else:
            self.active_gripper_q += np.sign(delta) * max_step

    # ----- Simulation step -----------------------------------------------

    def step(self) -> None:
        self.handle_keyboard()
        self.advance_active_target()
        self.advance_gripper()
        self.run_ik()

        wp.copy(self.control.joint_target_pos[: self.arm_coord_count], self.model.joint_q[: self.arm_coord_count])
        for idx in self.finger_coord_indices:
            if idx < self.model.joint_coord_count:
                self.control.joint_target_pos[idx : idx + 1].fill_(self.active_gripper_q)

        for _ in range(self.config.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    # ----- Render --------------------------------------------------------

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    # ----- Popup UI (free-floating window) --------------------------------

    def render_popup_ui(self, imgui) -> None:
        if not hasattr(self.viewer, "ui") or not self.viewer.ui.is_available:
            return

        io = self.viewer.ui.io
        win_size = (420, 560)
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - win_size[0] - 24, 24), imgui.Cond_.first_use_ever.value
        )
        imgui.set_next_window_size(imgui.ImVec2(*win_size), imgui.Cond_.first_use_ever.value)

        flags = imgui.WindowFlags_.no_collapse.value
        if not imgui.begin("Franka Teleop", flags=flags):
            imgui.end()
            return

        imgui.text("TCP Target Control")
        imgui.separator()

        cube_pos = self.get_cube_center()

        # ---- Status & parameters ----
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if imgui.collapsing_header("Status & Parameters"):
            p = self._target_pos_np
            r = self.target_euler_deg
            imgui.text(f"Position [m]: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
            imgui.text(f"Rotation XYZ [deg]: ({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f})")
            imgui.text(f"Cube center [m]: ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f})")
            imgui.separator()

            changed, position = imgui.drag_float3(
                "Target position", self._target_pos_np.tolist(), self.ui_translate_step, -1.5, 1.5, "%.3f"
            )
            if changed:
                self.set_target_pose(np.array(position, dtype=np.float32), self.target_quat)

            changed, rotation = imgui.slider_float3(
                "Target rotation XYZ", self.target_euler_deg.tolist(), -180.0, 180.0, "%.1f deg"
            )
            if changed:
                r, p, y = np.deg2rad(np.array(rotation, dtype=np.float64))
                self.set_target_pose(self._target_pos_np, wp.quat_rpy(float(r), float(p), float(y)))

            _, self.ui_translate_step = imgui.slider_float(
                "Translate step [m]", self.ui_translate_step, 0.001, 0.05, "%.3f"
            )
            _, self.ui_rotate_step_deg = imgui.slider_float(
                "Rotate step [deg]", self.ui_rotate_step_deg, 1.0, 30.0, "%.1f"
            )
            _, self.gripper_q = imgui.slider_float(
                "Gripper [m]", self.gripper_q, self.gripper_min, self.gripper_max, "%.3f"
            )

        # ---- Preset actions ----
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if imgui.collapsing_header("Preset Actions"):
            for label, action in [
                ("Move above cube (open)", self.move_gripper_above_cube),
                ("Descend to grasp height", self.move_gripper_to_cube),
                ("Grasp", self.execute_grasp),
                ("Lift grasped object", self.lift_grasped_object),
                ("Reset scene", self.reset_scene),
            ]:
                if imgui.button(label):
                    action()

        # ---- Hotkeys ----
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if imgui.collapsing_header("Hotkeys"):
            imgui.text("Hold Ctrl for Franka keyboard jog")
            for hint in (
                "Ctrl+W / S: +X / -X",
                "Ctrl+A / D: +Y / -Y",
                "Ctrl+Q / E: +Z / -Z",
                "Ctrl+U / O: +Roll X / -Roll X",
                "Ctrl+I / K: +Pitch Y / -Pitch Y",
                "Ctrl+J / L: +Yaw Z / -Yaw Z",
                "Ctrl+Z / X: Open / Close gripper",
            ):
                imgui.bullet_text(hint)

        imgui.end()

    # ----- Test ----------------------------------------------------------

    def test_final(self) -> None:
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        assert np.all(np.isfinite(joint_q)), "joint_q contains non-finite values"
        assert np.all(np.isfinite(joint_qd)), "joint_qd contains non-finite values"
        cube_center = self.get_cube_center()
        assert np.all(np.isfinite(cube_center)), "cube pose contains non-finite values"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=600)
        return parser


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = init_viewer(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
