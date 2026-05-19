# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example of a sphere on a plane using SolverKamino.
#
# Used for testing the SolverKamino contact filtering and constraint stabilization.
#
# Command: python -m newton.examples kamino_basic_sphere --world-count 16
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils import logger as msg
from newton.tests.utils import basics


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1  # max(1, round(self.frame_dt / 0.01))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        self.use_kamino_contacts = args.use_kamino_contacts if args else False
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        scene_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(scene_builder)
        scene_builder.default_shape_cfg.margin = 1e-3
        scene_builder.default_shape_cfg.gap = 0.1

        # Load the DR Legs USD and add it to the builder
        basics.build_sphere_on_plane(
            builder=scene_builder,
            z_offset=0.5,
            ground=False,
        )

        # Create the multi-world model by duplicating the prototype
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.request_contact_attributes("force")
        builder.default_shape_cfg.margin = 2e-3
        builder.default_shape_cfg.gap = 0.15
        for _ in range(self.world_count):
            builder.add_world(scene_builder)

        # Add a global ground plane applied to all worlds
        builder.add_ground_plane(
            label="ground",
            height=0.0,
        )

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)
        self.model.rigid_contact_max = 4

        # Create the Kamino solver for the given model
        self.config = newton.solvers.SolverKamino.Config.from_model(self.model)
        self.config.use_fk_solver = True
        self.config.use_collision_detector = self.use_kamino_contacts
        self.config.constraints.gamma = 0.001
        self.config.constraints.delta = 1e-4
        self.config.padmm.max_iterations = 200
        self.config.padmm.primal_tolerance = 1e-6
        self.config.padmm.dual_tolerance = 1e-6
        self.config.padmm.compl_tolerance = 1e-6
        self.solver = newton.solvers.SolverKamino(self.model, config=self.config)

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Configure CD components based on whether we want to use Kamino's
        # internal contact solver or Newton's collision pipeline
        if not self.use_kamino_contacts:
            self.collision_pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.model.contacts(collision_pipeline=self.collision_pipeline)
        else:
            self.collision_pipeline = None
            self.contacts = self.model.contacts()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.graph = None
        # self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        self.viewer._paused = True
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
            print("\n\n--------------------------------------------------------------------")
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if not self.use_kamino_contacts:
                self.collision_pipeline.collide(self.state_0, self.contacts)
                nc = self.contacts.rigid_contact_count.numpy()[0]
                msg.warning("contacts.rigid_contact_count: %s", nc)
                msg.warning("contacts.rigid_contact_margin0: %s", self.contacts.rigid_contact_margin0.numpy()[:nc])
                msg.warning("contacts.rigid_contact_margin1: %s", self.contacts.rigid_contact_margin1.numpy()[:nc])
                msg.warning("contacts.rigid_contact_offset0:\n%s", self.contacts.rigid_contact_offset0.numpy()[:nc])
                msg.warning("contacts.rigid_contact_offset1:\n%s", self.contacts.rigid_contact_offset1.numpy()[:nc])
                msg.warning("contacts.rigid_contact_point0:\n%s", self.contacts.rigid_contact_point0.numpy()[:nc])
                msg.warning("contacts.rigid_contact_point1:\n%s", self.contacts.rigid_contact_point1.numpy()[:nc])
                msg.warning("contacts.rigid_contact_normal:\n%s\n", self.contacts.rigid_contact_normal.numpy()[:nc])
                msg.notif("state_0.body_q:\n%s", self.state_0.body_q.numpy())
                msg.notif("state_0.body_qd:\n%s\n", self.state_0.body_qd.numpy())
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            else:
                msg.notif("state_0.body_q:\n%s", self.state_0.body_q.numpy())
                msg.notif("state_0.body_qd:\n%s\n", self.state_0.body_qd.numpy())
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
        self.viewer.log_state(self.state_1)
        self.viewer.log_contacts(self.contacts, self.state_1)
        self.viewer.end_frame()

    def test_final(self):
        pass  # TODO: Add some assertions here once we have a more meaningful test scenario

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_kamino_contacts_arg(parser)
        parser.set_defaults(world_count=1)
        parser.set_defaults(use_kamino_contacts=True)
        return parser


if __name__ == "__main__":
    np.set_printoptions(precision=10, linewidth=20000, threshold=10000, suppress=True)
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
