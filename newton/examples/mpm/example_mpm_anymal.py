# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM ANYmal
#
# Shows ANYmal C with a pretrained policy coupled with implicit MPM sand.
#
# Example usage (via unified runner):
#   python -m newton.examples mpm_anymal --viewer gl
###########################################################################

import sys

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples.robot.example_robot_anymal_c_walk import (
    _build_joint_target_pos_kernel,
    _compute_obs_kernel,
    lab_to_mujoco,
    mujoco_to_lab,
)
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, args):
        voxel_size = args.voxel_size
        particles_per_cell = args.particles_per_cell
        tolerance = args.tolerance
        grid_type = args.grid_type

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        self.device = wp.get_device()

        # import the robot model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        builder.add_urdf(
            stage_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        # Disable collisions with bodies other than shanks
        for body in range(builder.body_count):
            if "SHANK" not in builder.body_label[body]:
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        # set initial joint positions
        initial_q = {
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
        }
        # Set initial joint positions (skip first 7 position coordinates which are the free joint), e.g. for "LF_HAA" value will be written at index 1+6 = 7.
        for name, value in initial_q.items():
            idx = next(i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}"))
            builder.joint_q[idx + 6] = value

        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        # Register MPM custom attributes before adding particles
        SolverImplicitMPM.register_custom_attributes(builder)

        # add sand particles
        density = 2500.0
        particle_lo = np.array([-0.5, -0.5, 0.0])  # emission lower bound
        particle_hi = np.array([0.5, 2.5, 0.15])  # emission upper bound
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )
        _spawn_particles(builder, particle_res, particle_lo, particle_hi, density)

        # finalize model
        self.model = builder.finalize()

        # setup mpm solver
        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = tolerance
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = grid_type

        mpm_options.grid_padding = 50 if grid_type == "fixed" else 0
        mpm_options.max_active_cell_count = 1 << 15 if grid_type == "fixed" else -1

        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0
        mpm_options.collider_velocity_mode = "backward"

        # setup solvers
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            ls_iterations=50,
            njmax=50,  # ls_iterations=50 for determinism
        )
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        # simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Configure collider: treat robot bodies as kinematic and update initial state
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        # Setup control policy
        self.control = self.model.control()

        # Load the ONNX policy bundled with Newton (Warp-backed runtime, no torch dependency).
        policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.onnx")
        self.policy = newton.utils.OnnxRuntime(policy_path, device=str(self.device))
        self._policy_input_name = self.policy.input_names[0]
        self._policy_output_name = self.policy.output_names[0]

        # Everything the policy step touches is on-device.  The host never
        # syncs joint state, builds the obs vector, or copies the action back.
        # Reference posture is sliced on-device to avoid a startup host sync.
        self._joint_pos_initial_wp = wp.clone(self.state_0.joint_q[7:])
        self._lab_to_mujoco_wp = wp.array(np.asarray(lab_to_mujoco, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._mujoco_to_lab_wp = wp.array(np.asarray(mujoco_to_lab, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._gravity_w = wp.vec3(0.0, 0.0, -1.0)
        self._command = wp.vec3(0.0, 0.0, 0.0)

        self._obs_wp = wp.zeros((1, 48), dtype=wp.float32, device=self.device)
        self._prev_act_wp = wp.zeros((1, 12), dtype=wp.float32, device=self.device)

        # joint_target_pos is allocated by self.model.control() above; ensure
        # it lives on device so the kernel can write to it directly.
        if self.control.joint_target_pos is None or self.control.joint_target_pos.device != self.device:
            self.control.joint_target_pos = wp.zeros(18, dtype=wp.float32, device=self.device)

        self._auto_forward = True

        # set model on viewer and setup capture
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.graph = capture.graph

        self.sand_graph = None
        if wp.get_device().is_cuda and self.mpm_solver.grid_type == "fixed":
            with wp.ScopedCapture() as capture:
                self.simulate_sand()
            self.sand_graph = capture.graph

    def apply_control(self):
        # Build obs -> run policy -> write joint_target_pos, all on device.
        wp.launch(
            _compute_obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._joint_pos_initial_wp,
                self._lab_to_mujoco_wp,
                self._gravity_w,
                self._command,
                self._prev_act_wp,
                self._obs_wp,
            ],
            device=self.device,
        )
        out = self.policy({self._policy_input_name: self._obs_wp})
        act_wp = out[self._policy_output_name]

        wp.launch(
            _build_joint_target_pos_kernel,
            dim=6 + 12,
            inputs=[
                act_wp,
                self._joint_pos_initial_wp,
                self._mujoco_to_lab_wp,
                0.5,
                6,
                self.control.joint_target_pos,
            ],
            device=self.device,
        )

        # Stash the action for the *next* obs build, on device (no sync).
        wp.copy(self._prev_act_wp, act_wp)

    def simulate_robot(self):
        # robot substeps
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        # sand step (in-place on frame dt)
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def step(self):
        # Build command from viewer keyboard
        fwd = 0.0
        lat = 0.0
        rot = 0.0
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)

            if fwd or lat or rot:
                # disable forward motion
                self._auto_forward = False

        if self._auto_forward:
            fwd = 1.0
        # ``wp.vec3`` is immutable; rebuild it so the next _compute_obs_kernel
        # launch sees the latest command without touching device memory.
        self._command = wp.vec3(float(fwd), float(lat), float(rot))

        # compute control before graph/step
        self.apply_control()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate_robot()

        if self.sand_graph:
            wp.capture_launch(self.sand_graph)
        else:
            self.simulate_sand()

        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.1,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot went in the right direction",
            lambda q, qd: q[1] > 0.9,  # This threshold assumes 100 frames
        )

        forward_vel_min = wp.spatial_vector(-0.2, 0.9, -0.2, -0.8, -1.5, -0.5)
        forward_vel_max = wp.spatial_vector(0.2, 1.1, 0.2, 0.8, 1.5, 0.5)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot is moving forward and not falling",
            lambda q, qd: newton.math.vec_inside_limits(qd, forward_vel_min, forward_vel_max),
            indices=[0],
        )
        voxel_size = self.mpm_solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -1.1 * voxel_size,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)
        parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
        parser.add_argument("--grid-type", "-gt", choices=["sparse", "dense", "fixed"], default="sparse")
        parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-6)
        return parser


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, density):
    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)
    radius = np.max(cell_size) * 0.5
    mass = np.prod(cell_volume) * density

    builder.add_particle_grid(
        pos=wp.vec3(bounds_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=res[0] + 1,
        dim_y=res[1] + 1,
        dim_z=res[2] + 1,
        cell_x=cell_size[0],
        cell_y=cell_size[1],
        cell_z=cell_size[2],
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
    )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    if wp.get_device().is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    newton.examples.run(Example(viewer, args), args)
