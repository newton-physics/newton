# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Anymal D
#
# Shows how to simulate Anymal D with multiple worlds using MuJoCo or Kamino.
#
# Command: python -m newton.examples robot_anymal_d --world-count 16
#
###########################################################################

import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode, JointType


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count
        self.solver_type = args.solver

        self.viewer = viewer

        self.device = wp.get_device()

        articulation_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        if self.solver_type == "kamino":
            newton.solvers.SolverKamino.register_custom_attributes(articulation_builder)
        else:
            newton.solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
        )
        articulation_builder.default_shape_cfg.ke = 2.0e3
        articulation_builder.default_shape_cfg.kd = 1.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_d")
        asset_file = str(asset_path / "usd" / "anymal_d.usda")
        articulation_builder.add_usd(
            asset_file,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        articulation_builder.joint_q[:3] = [0.0, 0.0, 0.68]
        if len(articulation_builder.joint_q) > 6:
            articulation_builder.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]

        for joint_id, joint_type in enumerate(articulation_builder.joint_type):
            if joint_type != JointType.REVOLUTE:
                continue
            dof_start = articulation_builder.joint_qd_start[joint_id]
            dof_end = (
                articulation_builder.joint_qd_start[joint_id + 1]
                if joint_id + 1 < articulation_builder.joint_count
                else articulation_builder.joint_dof_count
            )
            for dof_id in range(dof_start, dof_end):
                articulation_builder.joint_target_ke[dof_id] = 150
                articulation_builder.joint_target_kd[dof_id] = 5
                articulation_builder.joint_target_mode[dof_id] = int(JointTargetMode.POSITION)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(articulation_builder)

        self.root_dof_indices = []
        for root_joint_id in builder.articulation_start:
            root_dof_start = builder.joint_qd_start[root_joint_id]
            root_dof_end = builder.joint_qd_start[root_joint_id + 1]
            self.root_dof_indices.extend(range(root_dof_start, root_dof_end))

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()
        use_mujoco_contacts = self.solver_type == "mujoco" and args.use_mujoco_contacts
        if self.solver_type == "kamino":
            solver_config = newton.solvers.SolverKamino.Config.from_model(
                self.model, dynamics_solver="dvi", sparse_dynamics=True, sparse_jacobian=True
            )
            solver_config.dvi.max_alternating_iterations = 8
            solver_config.dvi.bilateral_solve_interval = 8
            self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                cone="elliptic",
                impratio=100,
                iterations=100,
                ls_iterations=50,
                nconmax=45,
                njmax=100,
                use_mujoco_contacts=use_mujoco_contacts,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics for collision detection
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.use_mujoco_contacts = use_mujoco_contacts
        if use_mujoco_contacts:
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
        else:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.collision_pipeline.contacts()

        # ensure this is called at the end of the Example constructor
        self.viewer.set_model(self.model)

        # put graph capture into it's own function
        self.capture()

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        if not self.use_mujoco_contacts:
            self.collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.use_mujoco_contacts:
            self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
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
        """Verify the robot settles without actuating its floating base."""
        target_mode = self.model.joint_target_mode.numpy()
        target_ke = self.model.joint_target_ke.numpy()
        target_kd = self.model.joint_target_kd.numpy()
        if any(
            target_mode[dof_id] != int(JointTargetMode.NONE) or target_ke[dof_id] != 0.0 or target_kd[dof_id] != 0.0
            for dof_id in self.root_dof_indices
        ):
            raise AssertionError("floating-base DOFs must remain unactuated")

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.006,
        )
        # Only check velocities on CUDA where we run 500 frames (enough time to settle)
        # On CPU we only run 10 frames and the robot is still falling (~0.65 m/s)
        if self.device.is_cuda:
            # fmt: off
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "body velocities are small",
                lambda q, qd: max(abs(qd))
                < 0.25,  # Relaxed from 0.1 - collision pipeline has residual velocities up to ~0.2
            )
            # fmt: on

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_mujoco_contacts_arg(parser)
        parser.add_argument("--solver", choices=["mujoco", "kamino"], default="mujoco")
        parser.set_defaults(world_count=8)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
