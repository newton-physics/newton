# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Box Pyramid
#
# Builds pyramids of box-shaped cubes with a wrecking ball on a ramp
# to stress-test narrow-phase contact generation.
#
# Command: python -m newton.examples pyramid
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_NUM_PYRAMIDS = 20
DEFAULT_PYRAMID_SIZE = 20
CUBE_HALF = 0.4
CUBE_SPACING = 2.1 * CUBE_HALF
PYRAMID_SPACING = 2.0 * CUBE_SPACING
Y_STACK = 15.0

WRECKING_BALL_RADIUS = 2.0
WRECKING_BALL_DENSITY_MULT = 100.0
RAMP_LENGTH = 20.0
RAMP_WIDTH = 5.0
RAMP_THICKNESS = 0.5

XPBD_ITERATIONS = 2
XPBD_CONTACT_RELAXATION = 0.8


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.test_mode = args.test
        self.world_count = args.world_count

        num_pyramids = args.num_pyramids
        pyramid_size = args.pyramid_size

        builder = newton.ModelBuilder()
        builder.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        box_count = 0
        top_body_indices = []
        pyramid_height = pyramid_size * CUBE_SPACING

        for pyramid in range(num_pyramids):
            y_offset = pyramid * PYRAMID_SPACING
            for level in range(pyramid_size):
                num_cubes_in_row = pyramid_size - level
                row_width = (num_cubes_in_row - 1) * CUBE_SPACING
                for i in range(num_cubes_in_row):
                    x_pos = -row_width / 2 + i * CUBE_SPACING
                    z_pos = level * CUBE_SPACING + CUBE_HALF
                    y_pos = Y_STACK - y_offset
                    body = builder.add_body(
                        xform=wp.transform(p=wp.vec3(x_pos, y_pos, z_pos), q=wp.quat_identity()),
                    )
                    builder.add_shape_box(body, hx=CUBE_HALF, hy=CUBE_HALF, hz=CUBE_HALF)
                    if level == pyramid_size - 1:
                        top_body_indices.append(body)
                    box_count += 1

        self.box_count = box_count
        self.top_body_indices = top_body_indices
        print(f"Built {num_pyramids} pyramids x {pyramid_size} rows = {box_count} boxes")

        if not self.test_mode:
            # Wrecking ball
            ramp_height = 8.4
            ramp_angle = float(np.arctan2(ramp_height, RAMP_LENGTH))
            ball_x = 0.0
            ball_y = Y_STACK + RAMP_LENGTH * 0.9
            ball_z = ramp_height + WRECKING_BALL_RADIUS + 0.1

            body_ball = builder.add_body(
                xform=wp.transform(p=wp.vec3(ball_x, ball_y, ball_z), q=wp.quat_identity()),
            )
            ball_cfg = newton.ModelBuilder.ShapeConfig()
            ball_cfg.density = builder.default_shape_cfg.density * WRECKING_BALL_DENSITY_MULT
            builder.add_shape_sphere(body_ball, radius=WRECKING_BALL_RADIUS, cfg=ball_cfg)

            # Ramp (static)
            ramp_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(ramp_angle))
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(
                    p=wp.vec3(ball_x, Y_STACK + RAMP_LENGTH / 2, ramp_height / 2),
                    q=ramp_quat,
                ),
                hx=RAMP_WIDTH / 2,
                hy=RAMP_LENGTH / 2,
                hz=RAMP_THICKNESS / 2,
            )

        if self.world_count > 1:
            main_builder = newton.ModelBuilder()
            main_builder.replicate(builder, world_count=self.world_count)
            self.model = main_builder.finalize()
        else:
            self.model = builder.finalize()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=args.broad_phase,
        )

        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=XPBD_ITERATIONS,
            rigid_contact_relaxation=XPBD_CONTACT_RELAXATION,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.top_initial_positions = self.state_0.body_q.numpy()[:, :3].copy()

        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)

        cam_dist = max(pyramid_height, num_pyramids * PYRAMID_SPACING * 0.3)
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_dist * 0.4),
            pitch=-15.0,
            yaw=135.0,
        )

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
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
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify pyramid top cubes remain near their initial positions.

        In test mode the wrecking ball is omitted so the pyramids should
        settle under gravity without toppling.  Each top cube must stay
        within ``max_displacement`` of its initial position.
        """
        body_q = self.state_0.body_q.numpy()
        max_displacement = 0.5  # [m]
        for idx in self.top_body_indices:
            current_pos = body_q[idx, :3]
            initial_pos = self.top_initial_positions[idx]
            displacement = np.linalg.norm(current_pos - initial_pos)
            assert displacement < max_displacement, (
                f"Top cube body {idx}: displaced {displacement:.4f} m (max allowed {max_displacement:.4f} m)"
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=1)
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(broad_phase="sap")
        parser.add_argument(
            "--num-pyramids",
            type=int,
            default=DEFAULT_NUM_PYRAMIDS,
            help="Number of pyramids to build.",
        )
        parser.add_argument(
            "--pyramid-size",
            type=int,
            default=DEFAULT_PYRAMID_SIZE,
            help="Number of rows in each pyramid base.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
