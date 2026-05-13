# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot control via keyboard
#
# Shows how to control robots pretrained in IsaacLab with RL.  Policies are
# loaded from ONNX files and run via Newton's Warp-backed
# :class:`~newton.utils.OnnxRuntime` (no PyTorch dependency).
#
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
# Run this example with:
# python -m newton.examples robot_policy --robot g1_29dof
# python -m newton.examples robot_policy --robot g1_23dof
# python -m newton.examples robot_policy --robot go2
# python -m newton.examples robot_policy --robot anymal
# python -m newton.examples robot_policy --robot anymal --physx
###########################################################################

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
import yaml

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode


@dataclass
class RobotConfig:
    """Configuration for a robot including asset paths and policy paths."""

    asset_dir: str
    policy_path: dict[str, str]
    asset_path: str
    yaml_path: str


# Policy paths are now ONNX files.  These names mirror the original ``.pt``
# layout under ``rl_policies/`` but with the ``.onnx`` extension.
ROBOT_CONFIGS = {
    "anymal": RobotConfig(
        asset_dir="anybotics_anymal_c",
        policy_path={"mjw": "rl_policies/mjw_anymal.onnx", "physx": "rl_policies/physx_anymal.onnx"},
        asset_path="usd/anymal_c.usda",
        yaml_path="rl_policies/anymal.yaml",
    ),
    "go2": RobotConfig(
        asset_dir="unitree_go2",
        policy_path={"mjw": "rl_policies/mjw_go2.onnx", "physx": "rl_policies/physx_go2.onnx"},
        asset_path="usd/go2.usda",
        yaml_path="rl_policies/go2.yaml",
    ),
    "g1_29dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_29DOF.onnx"},
        asset_path="usd/g1_isaac.usd",
        yaml_path="rl_policies/g1_29dof.yaml",
    ),
    "g1_23dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_23DOF.onnx", "physx": "rl_policies/physx_g1_23DOF.onnx"},
        asset_path="usd/g1_minimal.usd",
        yaml_path="rl_policies/g1_23dof.yaml",
    ),
}


@wp.kernel
def _compute_obs_kernel(
    joint_q: wp.array[float],  # full joint_q, [px,py,pz, qx,qy,qz,qw, q0..]
    joint_qd: wp.array[float],  # full joint_qd, [vx,vy,vz, wx,wy,wz, qd0..]
    joint_pos_initial: wp.array[float],  # (num_dofs,) reference posture
    physx_to_mjc_idx: wp.array[int],  # (num_dofs,) reordering
    gravity_w: wp.vec3,
    command: wp.vec3,
    prev_act: wp.array2d[float],  # (1, num_dofs)
    num_dofs: int,
    obs: wp.array2d[float],  # (1, 12 + 3*num_dofs)
):
    """Build the IsaacLab-style observation entirely on-device.

    Layout (constant 12-float prefix + three num_dofs blocks):
      [0:3]                 body-frame linear velocity
      [3:6]                 body-frame angular velocity
      [6:9]                 body-frame projected gravity
      [9:12]                command (fwd, lat, yaw)
      [12 : 12+N]           joint position error, reordered
      [12+N : 12+2N]        joint velocity, reordered
      [12+2N : 12+3N]       previous action
    """
    q = wp.quat(joint_q[3], joint_q[4], joint_q[5], joint_q[6])

    lin_w = wp.vec3(joint_qd[0], joint_qd[1], joint_qd[2])
    ang_w = wp.vec3(joint_qd[3], joint_qd[4], joint_qd[5])

    vel_b = wp.quat_rotate_inv(q, lin_w)
    avel_b = wp.quat_rotate_inv(q, ang_w)
    grav_b = wp.quat_rotate_inv(q, gravity_w)

    obs[0, 0] = vel_b[0]
    obs[0, 1] = vel_b[1]
    obs[0, 2] = vel_b[2]
    obs[0, 3] = avel_b[0]
    obs[0, 4] = avel_b[1]
    obs[0, 5] = avel_b[2]
    obs[0, 6] = grav_b[0]
    obs[0, 7] = grav_b[1]
    obs[0, 8] = grav_b[2]
    obs[0, 9] = command[0]
    obs[0, 10] = command[1]
    obs[0, 11] = command[2]

    for k in range(num_dofs):
        idx = physx_to_mjc_idx[k]
        obs[0, 12 + k] = joint_q[7 + idx] - joint_pos_initial[idx]
        obs[0, 12 + num_dofs + k] = joint_qd[6 + idx]
        obs[0, 12 + 2 * num_dofs + k] = prev_act[0, k]


@wp.kernel
def _build_joint_target_pos_kernel(
    act: wp.array2d[float],  # (1, num_dofs) on device
    joint_pos_initial: wp.array[float],  # (num_dofs,) on device
    reorder: wp.array[int],  # (num_dofs,) mjc_to_physx mapping
    action_scale: float,
    num_prefix_zeros: int,
    out: wp.array[float],  # (num_prefix_zeros + num_dofs,)
):
    """Reorder, scale, and prepend ``num_prefix_zeros`` zeros into ``out``.

    Single kernel that replaces the NumPy reorder + concatenate + per-step
    ``wp.array(..., device=device)`` upload, keeping the control target on
    device for the rest of the simulation step.
    """
    i = wp.tid()
    if i < num_prefix_zeros:
        out[i] = 0.0
    else:
        j = i - num_prefix_zeros
        idx = reorder[j]
        out[i] = joint_pos_initial[j] + action_scale * act[0, idx]


def load_policy_and_setup_arrays(example: Any, policy_path: str, num_dofs: int, joint_pos_slice: slice):
    """Load ONNX policy and setup device buffers for the on-device policy step."""
    print("[INFO] Loading policy from:", policy_path)
    example.policy = newton.utils.OnnxRuntime(policy_path, device=str(example.device))
    example.policy_input_name = example.policy.input_names[0]
    example.policy_output_name = example.policy.output_names[0]

    # All policy inputs and intermediate buffers live on device.  Reference
    # posture is sliced on-device with wp.clone to avoid a startup host sync.
    if example.state_0.joint_q is not None:
        example._joint_pos_initial_wp = wp.clone(example.state_0.joint_q[joint_pos_slice])
    else:
        example._joint_pos_initial_wp = wp.zeros(num_dofs, dtype=wp.float32, device=example.device)

    # Cross-check three things that all have to agree: the YAML's declared
    # obs size (when present), the ONNX model's input shape, and the
    # 12 + 3*num_dofs layout that ``_compute_obs_kernel`` actually writes.
    # If they drift, fail loudly here instead of producing wrong inferences
    # at the first OnnxRuntime call.
    model_obs_dim = int(example.policy._shapes[example.policy_input_name][1])
    expected = 12 + 3 * num_dofs
    config_obs_dim = int(example.config["num_observations"]) if "num_observations" in example.config else None
    if config_obs_dim is not None and config_obs_dim != model_obs_dim:
        raise ValueError(
            f"load_policy_and_setup_arrays: config num_observations={config_obs_dim} does not match "
            f"the ONNX model's declared input shape ({model_obs_dim}); update the YAML or re-export "
            f"the policy."
        )
    if model_obs_dim != expected:
        raise ValueError(
            f"load_policy_and_setup_arrays: policy obs_dim={model_obs_dim} does not match the expected "
            f"layout (12 + 3*num_dofs = {expected}); the on-device _compute_obs_kernel only "
            f"supports the IsaacLab-style 12 + 3N observation."
        )
    example._obs_wp = wp.zeros((1, model_obs_dim), dtype=wp.float32, device=example.device)
    example._prev_act_wp = wp.zeros((1, num_dofs), dtype=wp.float32, device=example.device)

    # The two reorder permutations are pre-uploaded as int32 device arrays
    # (the kernel takes them as wp.array[int]).
    example._physx_to_mjc_wp = wp.array(
        np.asarray(example.physx_to_mjc_indices, dtype=np.int32), dtype=wp.int32, device=example.device
    )
    example._mjc_to_physx_wp = wp.array(
        np.asarray(example.mjc_to_physx_indices, dtype=np.int32), dtype=wp.int32, device=example.device
    )
    example._num_dofs = num_dofs


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc


class Example:
    def __init__(self, viewer, args):
        # Resolve robot configuration, asset paths, and joint mapping from args.
        # Done in __init__ (rather than at module level) so the example can be
        # rebuilt by _ExampleBrowser.reset() with the same (viewer, args) call
        # convention as every other example.
        if args.robot not in ROBOT_CONFIGS:
            raise ValueError(f"Unknown robot: {args.robot}. Available: {list(ROBOT_CONFIGS.keys())}")
        robot_config = ROBOT_CONFIGS[args.robot]
        print(f"[INFO] Selected robot: {args.robot}")

        asset_directory = str(newton.utils.download_asset(robot_config.asset_dir))
        print(f"[INFO] Asset directory: {asset_directory}")

        yaml_file_path = f"{asset_directory}/{robot_config.yaml_path}"
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"[INFO] Loaded config with {config['num_dofs']} DOFs")

        mjc_to_physx = list(range(config["num_dofs"]))
        physx_to_mjc = list(range(config["num_dofs"]))

        if args.physx:
            if "physx" not in robot_config.policy_path or "physx_joint_names" not in config:
                physx_robots = [name for name, cfg in ROBOT_CONFIGS.items() if "physx" in cfg.policy_path]
                raise ValueError(
                    f"PhysX policy not available for robot '{args.robot}'. Robots with PhysX support: {physx_robots}"
                )
            policy_path = f"{asset_directory}/{robot_config.policy_path['physx']}"
            mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(
                config["mjw_joint_names"], config["physx_joint_names"]
            )
        else:
            policy_path = f"{asset_directory}/{robot_config.policy_path['mjw']}"

        fps = 200
        self.frame_dt = 1.0e0 / fps
        self.decimation = 4
        self.cycle_time = 1 / fps * self.decimation

        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        self.use_mujoco = False
        self.config = config
        self.robot_config = robot_config

        self.device = wp.get_device()

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
        builder.approximate_meshes("convex_hull")

        builder.add_ground_plane()

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

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco,
            solver="newton",
            nconmax=30,
            njmax=100,
        )

        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        self.viewer.set_model(self.model)
        self.viewer.vsync = True

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.physx_to_mjc_indices = np.asarray(physx_to_mjc, dtype=np.int64)
        self.mjc_to_physx_indices = np.asarray(mjc_to_physx, dtype=np.int64)
        # Kernel reads these as wp.vec3 values passed by argument; the
        # command is mutated each frame from keyboard input.
        self._gravity_w = wp.vec3(0.0, 0.0, -1.0)
        self._command = wp.vec3(0.0, 0.0, 0.0)
        self._reset_key_prev = False

        self.policy = None
        self.policy_input_name = None
        self.policy_output_name = None
        # Device-resident buffers; populated by load_policy_and_setup_arrays.
        self._joint_pos_initial_wp = None
        self._obs_wp = None
        self._prev_act_wp = None
        self._physx_to_mjc_wp = None
        self._mjc_to_physx_wp = None
        self._num_dofs = None

        # Load policy weights and prepare device buffers used by step().
        load_policy_and_setup_arrays(self, policy_path, config["num_dofs"], slice(7, None))

        self.capture()

    def capture(self):
        self.graph = None
        self.use_cuda_graph = False
        if wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
            print("[INFO] Using CUDA graph")
            self.use_cuda_graph = True
            self.control.joint_target_pos = wp.zeros(self.config["num_dofs"] + 6, dtype=wp.float32, device=self.device)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        need_state_copy = self.use_cuda_graph and self.sim_substeps % 2 == 1

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            if need_state_copy and i == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

        self.solver.update_contacts(self.contacts, self.state_0)

    def reset(self):
        print("[INFO] Resetting example")
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        # Clear stale policy history so the first observation after reset is
        # built purely from the restored kinematic state.
        if self._prev_act_wp is not None:
            self._prev_act_wp.zero_()

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            # ``wp.vec3`` is immutable; rebuild it on key change so the next
            # _compute_obs_kernel launch sees the latest command.
            self._command = wp.vec3(float(fwd), float(lat), float(rot))
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        # Build obs on device.  No host syncs in the policy step pipeline.
        wp.launch(
            _compute_obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._joint_pos_initial_wp,
                self._physx_to_mjc_wp,
                self._gravity_w,
                self._command,
                self._prev_act_wp,
                self._num_dofs,
                self._obs_wp,
            ],
            device=self.device,
        )
        out = self.policy({self.policy_input_name: self._obs_wp})
        act_wp = out[self.policy_output_name]

        # Build joint_target_pos on device: reorder, scale, prepend zeros.
        wp.launch(
            _build_joint_target_pos_kernel,
            dim=6 + self._num_dofs,
            inputs=[
                act_wp,
                self._joint_pos_initial_wp,
                self._mjc_to_physx_wp,
                float(self.config["action_scale"]),
                6,
                self.control.joint_target_pos,
            ],
            device=self.device,
        )

        # Stash the action for the *next* obs build, on device (no sync).
        wp.copy(self._prev_act_wp, act_wp)

        for _ in range(self.decimation):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
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

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--robot",
            type=str,
            default="g1_29dof",
            choices=list(ROBOT_CONFIGS.keys()),
            help="Robot name to load",
        )
        parser.add_argument(
            "--physx",
            action="store_true",
            help="Run physX policy instead of MJWarp.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
