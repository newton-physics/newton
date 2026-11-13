# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example for basic DR TestMech system

Shows how to simulate DR TestMech with multiple worlds using SolverKamino.

Success criteria:
- A mechanism with multiple linked pieces moves back and forth.
- Motion propagates through its joints and links without disconnection.

Runs:
- `python -m newton.examples kamino_basic_dr_testmech`
- `python -m newton.examples kamino_basic_dr_testmech --world-count 16`
"""

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.sim_dt = 0.001
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
        # Load the DR TestMech USD and add it to the builder
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_testmech/usd" / "dr_testmech.usda")
        robot_builder.add_usd(
            asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)

        # The mechanism is simulated without collisions, so reserve only a
        # single contact slot per world.
        self.model.rigid_contact_max = self.world_count

        # Create the Kamino solver for the given model
        self.config = newton.solvers.SolverKamino.Config.from_model(self.model)
        self.config.use_collision_detector = False
        self.config.use_fk_solver = False
        self.config.padmm.max_iterations = 200
        self.config.padmm.primal_tolerance = 1e-6
        self.config.padmm.dual_tolerance = 1e-6
        self.config.padmm.compl_tolerance = 1e-6
        self.config.padmm.rho_0 = 0.01
        self.solver = newton.solvers.SolverKamino(self.model, config=self.config)

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(0.2, 0.2, 0.15)
            pitch = -20.0
            yaw = 215.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def capture(self):
        self.graph = None
        if self.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
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
        self.viewer.end_frame()

    def test_post_step(self):
        # Track each body's position range to verify the mechanism keeps swinging.
        positions = self.state_0.body_q.numpy()[:, :3]
        if not hasattr(self, "_position_min"):
            self._position_min = positions.copy()
            self._position_max = positions.copy()
        else:
            np.minimum(self._position_min, positions, out=self._position_min)
            np.maximum(self._position_max, positions, out=self._position_max)

    def test_final(self):
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "Non-finite body poses"

        # Mechanism must move back and forth instead of staying frozen.
        travel = np.linalg.norm(self._position_max - self._position_min, axis=1)
        world_travel = np.zeros(self.world_count)
        np.maximum.at(world_travel, self.model.body_world.numpy(), travel)
        assert np.all(world_travel > 0.01), f"Mechanism did not move: max body travel per world {world_travel}"

        # Revolute, ball, and fixed joints keep their anchors coincident, so the links stay connected.
        parent = self.model.joint_parent.numpy()
        child = self.model.joint_child.numpy()
        joint_type = self.model.joint_type.numpy()
        xp = self.model.joint_X_p.numpy()
        xc = self.model.joint_X_c.numpy()
        anchored = (int(newton.JointType.REVOLUTE), int(newton.JointType.BALL), int(newton.JointType.FIXED))
        for joint, label in enumerate(self.model.joint_label):
            if joint_type[joint] not in anchored:
                continue
            parent_pose = wp.transform(*poses[parent[joint]]) if parent[joint] >= 0 else wp.transform_identity()
            a = wp.transform_point(parent_pose, wp.vec3(*xp[joint, :3]))
            b = wp.transform_point(wp.transform(*poses[child[joint]]), wp.vec3(*xc[joint, :3]))
            gap = wp.length(a - b)
            assert gap < 1e-3, f"Joint opened by {gap:.2e} m: {label}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
