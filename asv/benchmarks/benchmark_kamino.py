# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import time
import warnings

import numpy as np
import warp as wp

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton

if __package__:
    from .benchmark_metrics import validate_simulation_state
else:
    from benchmark_metrics import validate_simulation_state

_NUM_ACTIONS = 12
_OBS_DIM = 94
_MIN_STANDING_HEIGHT = 0.20
_MAX_STANDING_HEIGHT = 0.35
_MIN_HEALTHY_WORLD_FRACTION = 0.90
_MIN_ACTION_NORM = 1.0e-4
_MAX_BODY_LINEAR_SPEED = 10.0
_MAX_BODY_ANGULAR_SPEED = 50.0

# Joint order the policy was trained with: USD declaration order. The Kamino
# RL stack imports with joint_ordering=None to preserve it (Newton's default
# "dfs" tree ordering would differ); joints are looked up by name, so this
# list tolerates either ordering.
_POLICY_JOINT_NAMES = []
for leg in ("l", "r"):
    for linkage in ("i", "o"):
        _POLICY_JOINT_NAMES.extend(f"j{joint}_{leg}_{linkage}" for joint in range(1, 10))
_DRIVEN_JOINT_NAMES = [
    "j1_l_i",
    "j2_l_i",
    "j6_l_i",
    "j7_l_i",
    "j2_l_o",
    "j7_l_o",
    "j1_r_i",
    "j2_r_i",
    "j6_r_i",
    "j7_r_i",
    "j2_r_o",
    "j7_r_o",
]

# Walk-policy constants inlined from dr_legs/rl_policies/drlegs_walk.yaml at
# the pinned asset ref, so an asset-side config change cannot silently alter
# the benchmarked workload.
_DRLEGS_WALK_CONFIG = {
    # Policy
    "action_scale": 0.4,  # joint position scale [rad]
    "policy_file": "drlegs_walk.pt",
    # Gait timing
    "contact_duration": 0.3,  # seconds per foot contact phase
    "phase_embedding_k": 2,  # periodic encoding order -> 2k-D embedding
    # Velocity commands
    "vel_cmd_max": 0.3,  # max linear velocity command [m/s]
    "yaw_cmd_max": 0.8,  # max yaw rate command [rad/s]
    # Implicit PD gains (actuated joints only)
    "pd_kp": 15.0,  # proportional gain [N·m/rad]
    "pd_kd": 0.6,  # derivative gain [N·m·s/rad]
    "pd_armature": 0.01,  # rotor inertia [kg·m²]
    # Observation scaling
    "path_deviation_scale": 0.1,
    "linear_path_error_limit": 0.1,  # max path-to-root deviation before clipping [m]
    "height_error_scale": 0.05,
    # Height command range
    "standing_height": 0.265,  # default pelvis Z [m]
    "height_cmd_min": 0.16,  # minimum pelvis height command [m]
    "height_cmd_max": 0.27,  # maximum pelvis height command [m]
    # Simulation
    "sim_dt": 0.004,  # physics substep duration [s]
    "control_decimation": 5,  # physics substeps per RL step
    "body_pose_offset_z": 0.265,  # initial pelvis Z offset [m]
    "usd_model": "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda",
}
_DRLEGS_WALK_CONFIG["phase_rate"] = 1.0 / (2.0 * _DRLEGS_WALK_CONFIG["contact_duration"])

ROBOT_CONFIGS = {
    "dr_legs": {
        "asset_name": "disneyresearch",
        # TODO: Update the asset ref and the inlined config above when a policy
        # trained for the current DR Legs asset is published.
        "asset_ref": "d69b2e04bc1fc246c415f0549e5f02d8aae1ef31",
        "config": _DRLEGS_WALK_CONFIG,
    },
}


def _load_robot_config(robot):
    asset_cfg = ROBOT_CONFIGS[robot]
    asset_path = newton.utils.download_asset(asset_cfg["asset_name"], ref=asset_cfg["asset_ref"])
    return asset_path, asset_cfg["config"]


class PolicyController:
    """Runs the deployed 94D DR Legs policy with articulated name mapping."""

    def __init__(
        self,
        policy_pt,
        model,
        world_count,
        config,
        command=(0.25, 0.0, 0.0),
    ):
        import torch  # noqa: PLC0415

        # Reuse the observation helpers from the Kamino RL example to mirror the deployed workload.
        from newton._src.solvers.kamino.examples.rl.utils import (  # noqa: PLC0415
            _load_policy_checkpoint,
            periodic_encoding,
            quat_inv_mul,
            quat_rotate_inv,
            quat_to_projected_yaw,
            quat_to_rotation9d,
            yaw_apply_2d,
            yaw_to_quat,
        )

        self._torch = torch
        self._quat_inv_mul = quat_inv_mul
        self._quat_rotate_inv = quat_rotate_inv
        self._quat_to_projected_yaw = quat_to_projected_yaw
        self._quat_to_rotation9d = quat_to_rotation9d
        self._yaw_apply_2d = yaw_apply_2d
        self._yaw_to_quat = yaw_to_quat
        self._device = str(model.device)
        self._model = _load_policy_checkpoint(policy_pt, device=self._device)
        with torch.no_grad():
            policy_output = self._model(torch.zeros((1, _OBS_DIM), dtype=torch.float32, device=self._device))
        if policy_output.shape != (1, _NUM_ACTIONS):
            raise ValueError(
                f"Expected policy shape (1, {_NUM_ACTIONS}) for a (1, {_OBS_DIM}) input, "
                f"got {tuple(policy_output.shape)}"
            )

        self._wc = world_count
        self._jcc = model.joint_coord_count // world_count
        self._body_count = model.body_count // world_count
        self._action_scale = config["action_scale"]
        self._control_dt = config["sim_dt"] * config["control_decimation"]
        self._phase_rate = config["phase_rate"]
        self._path_deviation_scale = config["path_deviation_scale"]
        self._linear_path_error_limit = config["linear_path_error_limit"]
        self._standing_height = config["standing_height"]
        self._height_error_scale = config["height_error_scale"]

        self._obs = torch.zeros((world_count, _OBS_DIM), dtype=torch.float32, device=self._device)
        command = torch.as_tensor(command, dtype=torch.float32, device=self._device)
        if command.shape != (3,):
            raise ValueError(f"Expected a 3D velocity command, got shape {command.shape}")
        self._commands = command.expand(world_count, 3).clone()

        joint_count = model.joint_count // world_count
        labels = [label.rsplit("/", 1)[-1] for label in model.joint_label[:joint_count]]
        indices = {label: i for i, label in enumerate(labels)}
        missing = set(_POLICY_JOINT_NAMES) - indices.keys()
        if missing:
            raise ValueError(f"DR Legs model is missing policy joints: {sorted(missing)}")
        q_start = model.joint_q_start.numpy()
        target_q_start = model.joint_target_q_start.numpy()
        self._policy_coord_offsets = torch.tensor(
            [q_start[indices[name]] for name in _POLICY_JOINT_NAMES], dtype=torch.long, device=self._device
        )
        self._driven_target_offsets = torch.tensor(
            [target_q_start[indices[name]] for name in _DRIVEN_JOINT_NAMES], dtype=torch.long, device=self._device
        )

        body_labels = [label.rsplit("/", 1)[-1] for label in model.body_label[: self._body_count]]
        try:
            self._root_body = body_labels.index("pelvis")
        except ValueError as e:
            raise ValueError("DR Legs model has no pelvis root body") from e
        body_com = wp.to_torch(model.body_com).reshape(world_count, self._body_count, 3)
        self._root_com = body_com[:, self._root_body].clone()

        self._actions = torch.zeros((world_count, _NUM_ACTIONS), dtype=torch.float32, device=self._device)
        self._setpoint_current = torch.zeros_like(self._actions)
        self._setpoint_previous = torch.zeros_like(self._actions)
        self._path_position = torch.zeros((world_count, 2), dtype=torch.float32, device=self._device)
        self._path_heading = torch.zeros(world_count, dtype=torch.float32, device=self._device)
        self._phase = torch.zeros(world_count, dtype=torch.float32, device=self._device)
        freq_2pi, offset = periodic_encoding(k=config["phase_embedding_k"])
        self._freq_2pi = torch.from_numpy(freq_2pi).float().to(self._device)
        self._phase_offset = torch.from_numpy(offset).float().to(self._device)
        self._phase_encoding = torch.zeros((world_count, 4), dtype=torch.float32, device=self._device)
        self._zeros = torch.zeros((world_count, 1), dtype=torch.float32, device=self._device)
        self._path_initialized = False

    def _quat_rotate(self, q, v):
        q_vec = q[..., :3]
        t = 2.0 * self._torch.linalg.cross(q_vec, v, dim=-1)
        return v + q[..., 3:4] * t + self._torch.linalg.cross(q_vec, t, dim=-1)

    def step(self, state, control):
        torch = self._torch
        wc = self._wc
        joint_q = wp.to_torch(state.joint_q).reshape(wc, self._jcc)
        body_q = wp.to_torch(state.body_q).reshape(wc, self._body_count, 7)
        body_qd = wp.to_torch(state.body_qd).reshape(wc, self._body_count, 6)
        root_body_q = body_q[:, self._root_body]
        root_quat = root_body_q[:, 3:7]
        root_pos = root_body_q[:, :3] + self._quat_rotate(root_quat, self._root_com)
        root_lin_vel = body_qd[:, self._root_body, :3]
        root_ang_vel = body_qd[:, self._root_body, 3:6]

        if not self._path_initialized:
            self._path_position[:] = root_pos[:, :2]
            self._path_initialized = True

        cmd_vel = self._commands[:, :2]
        cmd_yaw = self._commands[:, 2]
        mid_heading = self._path_heading + 0.5 * self._control_dt * cmd_yaw
        self._path_position += self._yaw_apply_2d(mid_heading, cmd_vel) * self._control_dt
        self._path_heading += self._control_dt * cmd_yaw

        path_error = self._path_position - root_pos[:, :2]
        path_error = path_error.renorm(p=2, dim=0, maxnorm=self._linear_path_error_limit)
        self._path_position[:] = root_pos[:, :2] + path_error

        path_quat = self._yaw_to_quat(self._path_heading)
        root_in_path = self._quat_inv_mul(path_quat, root_quat)

        diff_xy = root_pos[:, :2] - self._path_position
        diff_3d = torch.cat((diff_xy, self._zeros), dim=-1)
        dev_in_path = self._quat_rotate_inv(path_quat, diff_3d)[:, :2]
        path_dev = dev_in_path / self._path_deviation_scale

        root_heading = self._quat_to_projected_yaw(root_in_path)
        heading_quat = self._yaw_to_quat(root_heading)
        neg_dev = torch.cat((-dev_in_path, self._zeros), dim=-1)
        path_dev_heading = self._quat_rotate_inv(heading_quat, neg_dev)[:, :2] / self._path_deviation_scale

        cmd_linvel = torch.cat((cmd_vel, self._zeros), dim=-1)
        cmd_angvel = torch.cat((self._zeros, self._zeros, self._commands[:, 2:3]), dim=-1)
        cmd_linvel_root = self._quat_rotate_inv(root_in_path, cmd_linvel)
        cmd_angvel_root = self._quat_rotate_inv(root_in_path, cmd_angvel)

        self._phase.add_(self._control_dt * self._phase_rate).remainder_(1.0)
        torch.sin(torch.outer(self._phase, self._freq_2pi).add_(self._phase_offset), out=self._phase_encoding)
        root_linvel_local = self._quat_rotate_inv(root_quat, root_lin_vel)
        root_angvel_local = self._quat_rotate_inv(root_quat, root_ang_vel)

        self._setpoint_previous[:] = self._setpoint_current
        self._setpoint_current[:] = self._action_scale * self._actions

        obs = self._obs
        obs[:, 0:9] = self._quat_to_rotation9d(root_in_path)
        obs[:, 9:11] = path_dev
        obs[:, 11:13] = path_dev_heading
        obs[:, 13:16] = self._commands
        obs[:, 16:19] = cmd_linvel_root
        obs[:, 19:22] = cmd_angvel_root
        obs[:, 22:26] = self._phase_encoding
        obs[:, 26:29] = root_linvel_local
        obs[:, 29:32] = root_angvel_local
        obs[:, 32] = self._standing_height
        obs[:, 33] = (root_pos[:, 2] - self._standing_height) / self._height_error_scale
        obs[:, 34:70] = joint_q[:, self._policy_coord_offsets]
        obs[:, 70:82] = self._setpoint_current
        obs[:, 82:94] = self._setpoint_previous

        with torch.no_grad():
            self._actions[:] = self._model(obs)

        target_count = len(control.joint_target_q) // wc
        target_q = wp.to_torch(control.joint_target_q).reshape(wc, target_count)
        target_q.zero_()
        target_q[:, self._driven_target_offsets] = self._action_scale * self._actions
        wp.to_torch(control.joint_target_qd).zero_()

    def test_final(self):
        if not self._torch.isfinite(self._actions).all().item():
            raise RuntimeError("Policy produced non-finite actions")
        action_norms = self._torch.linalg.vector_norm(self._actions, dim=1)
        nontrivial_count = self._torch.count_nonzero(action_norms >= _MIN_ACTION_NORM).item()
        nontrivial_fraction = nontrivial_count / self._wc
        if nontrivial_fraction < _MIN_HEALTHY_WORLD_FRACTION:
            raise RuntimeError(
                f"Only {nontrivial_count}/{self._wc} robots have policy action norm >= "
                f"{_MIN_ACTION_NORM:.1e} ({nontrivial_fraction:.1%})"
            )


class DRLegsBenchmarkWorkload:
    """Public SolverKamino mirror of the deployed DR Legs RL workload.

    Contact aggregation is omitted because the walk policy does not consume it.
    Policy work runs on the simulation device but outside the physics timer.
    The default command is a deterministic 0.25 m/s forward joystick input.
    """

    def __init__(
        self,
        robot="dr_legs",
        world_count=1,
        use_cuda_graph=True,
        use_policy=True,
        builder=None,
        viewer=None,
        command=(0.25, 0.0, 0.0),
    ):
        asset_path, cfg = _load_robot_config(robot)

        self.sim_time = 0.0
        self.benchmark_time = 0.0
        self.sim_dt = cfg["sim_dt"]
        self.decimation = cfg["control_decimation"]
        self.sim_substeps = self.decimation
        self.frame_dt = self.sim_dt * self.sim_substeps
        self.world_count = world_count
        self.viewer = viewer

        if builder is None:
            builder = DRLegsBenchmarkWorkload.create_model_builder(robot, world_count)
        self.model = builder.finalize(skip_validation_joints=True)

        # Match RigidBodySim.body_pose_offset: translate initial body poses only,
        # after USD import has established the local articulation frames.
        body_q = self.model.body_q.numpy()
        body_q[:, 2] += cfg["body_pose_offset_z"]
        self.model.body_q.assign(body_q)

        DRLegsBenchmarkWorkload._set_newton_joint_params(self.model, cfg, world_count)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.solver = DRLegsBenchmarkWorkload.create_solver(self.model, self.sim_dt)
        self.solver.reset(state=self.state_0)

        self._world_reset_mask = wp.zeros(world_count, dtype=wp.bool, device=self.model.device)
        self._reset_config = newton.solvers.SolverKamino.ResetConfig.to_default()

        if self.viewer is not None:
            self.viewer.set_model(self.model)

        self.policy_controller = None
        if use_policy:
            self.policy_controller = PolicyController(
                str(asset_path / "dr_legs" / "rl_policies" / cfg["policy_file"]),
                self.model,
                world_count,
                cfg,
                command=command,
            )

        self.graph = None
        self.reset_graph = None
        self._capturing_graph = False
        if use_cuda_graph:
            device = self.model.device
            if device.is_cuda and wp.is_mempool_enabled(device):
                with wp.ScopedCapture() as reset_capture:
                    self._reset_tick()
                self.reset_graph = reset_capture.graph
                self._capturing_graph = True
                with wp.ScopedCapture() as capture:
                    self.simulate_tick()
                self._capturing_graph = False
                self.graph = capture.graph
                self.solver.reset(state=self.state_0)
            else:
                warnings.warn(
                    f"use_cuda_graph=True but CUDA graph capture is unavailable on device '{device}' "
                    "(requires a CUDA device with the mempool allocator enabled); "
                    "falling back to eager kernel launches.",
                    stacklevel=2,
                )

        wp.synchronize_device()

    @staticmethod
    def _set_newton_joint_params(model, cfg, world_count):
        dofs_per_world = model.joint_dof_count // world_count
        joints_per_world = model.joint_count // world_count
        labels = [label.rsplit("/", 1)[-1] for label in model.joint_label[:joints_per_world]]
        indices = {label: i for i, label in enumerate(labels)}
        missing = set(_DRIVEN_JOINT_NAMES) - indices.keys()
        if missing:
            raise ValueError(f"DR Legs model is missing driven joints: {sorted(missing)}")
        qd_start = model.joint_qd_start.numpy()
        driven_offsets = np.asarray([qd_start[indices[name]] for name in _DRIVEN_JOINT_NAMES])
        world_offsets = np.arange(world_count)[:, None] * dofs_per_world
        driven_dofs = (world_offsets + driven_offsets).ravel()

        ke = np.zeros(model.joint_dof_count, dtype=np.float32)
        kd = np.zeros(model.joint_dof_count, dtype=np.float32)
        armature = model.joint_armature.numpy()
        ke[driven_dofs] = cfg["pd_kp"]
        kd[driven_dofs] = cfg["pd_kd"]
        armature[driven_dofs] = cfg["pd_armature"]

        model.joint_target_ke.assign(ke)
        model.joint_target_kd.assign(kd)
        model.joint_armature.assign(armature)
        model.joint_friction.zero_()

    def _reset_tick(self):
        self.solver.reset(
            state=self.state_0,
            world_mask=self._world_reset_mask,
            config=self._reset_config,
        )

    def simulate_tick(self):
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        if self._capturing_graph:
            self.state_0.assign(self.state_1)
        else:
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self):
        for _ in range(self.decimation):
            self._reset_tick()
            self.simulate_tick()

    def step(self):
        if self.policy_controller is not None:
            self.policy_controller.step(self.state_0, self.control)

        wp.synchronize_device()
        start_time = time.perf_counter()
        if self.graph is not None:
            for _ in range(self.decimation):
                wp.capture_launch(self.reset_graph)
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        wp.synchronize_device()
        self.benchmark_time += time.perf_counter() - start_time
        self.sim_time += self.frame_dt

    def test_final(self):
        validate_simulation_state(
            self.state_0,
            max_linear_speed=_MAX_BODY_LINEAR_SPEED,
            max_angular_speed=_MAX_BODY_ANGULAR_SPEED,
        )

        if self.policy_controller is None:
            return

        body_count = self.model.body_count // self.world_count
        body_labels = [label.rsplit("/", 1)[-1] for label in self.model.body_label[:body_count]]
        try:
            pelvis_index = body_labels.index("pelvis")
        except ValueError as e:
            raise RuntimeError("DR Legs model has no pelvis root body") from e

        body_q = self.state_0.body_q.numpy().reshape(self.world_count, body_count, 7)[:, pelvis_index]
        body_com = self.model.body_com.numpy().reshape(self.world_count, body_count, 3)[:, pelvis_index]
        quat_vector = body_q[:, 3:6]
        twice_cross = 2.0 * np.cross(quat_vector, body_com)
        rotated_com = body_com + body_q[:, 6:7] * twice_cross + np.cross(quat_vector, twice_cross)
        pelvis_height = body_q[:, 2] + rotated_com[:, 2]
        standing_count = np.count_nonzero(
            (pelvis_height >= _MIN_STANDING_HEIGHT) & (pelvis_height <= _MAX_STANDING_HEIGHT)
        )
        standing_fraction = standing_count / self.world_count
        if standing_fraction < _MIN_HEALTHY_WORLD_FRACTION:
            raise RuntimeError(
                f"Only {standing_count}/{self.world_count} robots have pelvis height within "
                f"[{_MIN_STANDING_HEIGHT:.2f}, {_MAX_STANDING_HEIGHT:.2f}] m ({standing_fraction:.1%})"
            )

        self.policy_controller.test_final()

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_model_builder(robot, world_count):
        asset_path, cfg = _load_robot_config(robot)
        usda = str(asset_path / cfg["usd_model"])

        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 0.0
        robot_builder.default_shape_cfg.gap = 0.0
        robot_builder.add_usd(
            usda,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(world_count):
            builder.add_world(robot_builder)
        builder.add_ground_plane()

        return builder

    @staticmethod
    def create_solver(model, sim_dt):
        # Reuse the Kamino RL example's solver settings to mirror the deployed workload.
        from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim  # noqa: PLC0415

        settings = RigidBodySim.default_settings(sim_dt)
        settings.solver.collision_detector = settings.collision_detector
        # Pin the linear solver so a change to default_settings cannot
        # silently switch what this benchmark measures.
        settings.solver.dynamics.linear_solver_type = "LLTBRCM"
        return newton.solvers.SolverKamino(model, config=settings.solver)


if __name__ == "__main__":
    import newton.examples

    parser = newton.examples.create_parser()
    newton.examples.add_world_count_arg(parser)
    parser.add_argument("--no-policy", action="store_true", help="Run without RL policy")
    parser.set_defaults(world_count=1)
    viewer, args = newton.examples.init(parser)

    workload = DRLegsBenchmarkWorkload(
        world_count=args.world_count,
        use_cuda_graph=True,
        use_policy=not args.no_policy,
        viewer=viewer,
    )
    newton.examples.run(workload, args)
