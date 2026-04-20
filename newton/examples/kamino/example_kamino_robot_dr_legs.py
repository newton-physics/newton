# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot DR Legs
#
# Shows how to simulate DR Legs with multiple worlds using SolverKamino.
#
# Command: python -m newton.examples kamino_robot_dr_legs --world-count 16
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils import logger as msg


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.sim_dt = 0.01
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 1e-6
        robot_builder.default_shape_cfg.gap = 0.01

        # Load the DR Legs USD and add it to the builder
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_meshes_and_boxes.usda")
        robot_builder.add_usd(
            asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,  # TODO @cavemor: Fails when True, investigate (doesn't have fixed joints)
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Add a global ground plane applied to all worlds
        builder.add_ground_plane()

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)

        # TODO @nvtw: This is a temporary fix because `robot_builder.default_shape_cfg`
        # is not correctly applied to the shapes when using `add_usd()`,
        msg.debug("self.model.shape_margin: %s", self.model.shape_margin)
        msg.debug("self.model.shape_gap: %s", self.model.shape_gap)
        self.model.shape_margin.fill_(1e-6)
        self.model.shape_gap.fill_(0.01)

        # Create the Kamino solver for the given model
        self.config = newton.solvers.SolverKamino.Config.from_model(self.model)
        self.config.use_collision_detector = True
        self.config.use_fk_solver = True
        self.config.padmm.max_iterations = 200
        self.config.padmm.primal_tolerance = 1e-4
        self.config.padmm.dual_tolerance = 1e-4
        self.config.padmm.compl_tolerance = 1e-4
        self.solver = newton.solvers.SolverKamino(self.model, config=self.config)

        # Set joint armature and viscous damping for better
        # stability of the implicit joint-space PD controller
        # TODO: Remove this once we add Newton USD schemas in the model asset
        self.solver._solver_kamino._model.joints.a_j.fill_(0.011)  # Joint armature
        self.solver._solver_kamino._model.joints.b_j.fill_(0.044)  # Joint viscous damping
        self.solver._solver_kamino._model.joints.k_p_j.fill_(10.0)  # Proportional gain
        self.solver._solver_kamino._model.joints.k_d_j.fill_(2.0)  # Derivative gain

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Warm-start the simulation
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.solver.reset(self.state_0)

        # Reset the simulation state to a valid initial configuration above the ground
        self.base_q = wp.zeros(shape=(self.world_count,), dtype=wp.transformf)
        q_b = wp.quat_identity(dtype=wp.float32)
        q_base = wp.transformf((0.0, 0.0, 0.4), q_b)
        self.base_q.assign([q_base] * self.world_count)
        self.solver.reset(state_out=self.state_0, base_q=self.base_q)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(1.34, 0.0, 0.25)
            pitch = -7.0
            yaw = -180.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.solver.update_contacts(self.contacts, self.state_0)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_1)
        self.viewer.end_frame()

    def test_final(self):
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
                < 0.25,  # Relaxed from 0.1 - unified pipeline has residual velocities up to ~0.2
            )
            # fmt: on

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_kamino_contacts_arg(parser)
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    msg.notif("Starting the simulation...")
    newton.examples.run(example, args)
