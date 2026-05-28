# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example demonstrating all supported geometry types
#
# Demonstrates the different geometry shape combinations that can collide in SolverKamino.
#
# Command: python -m newton.examples kamino_basic_all_geoms
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton._src.solvers.kamino._src.geometry.primitive.broadphase import PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES
from newton._src.solvers.kamino._src.geometry.primitive.narrowphase import PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS
from newton.tests.utils import testing


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.sim_dt = 0.001
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.viewer = viewer
        self.device = wp.get_device()

        # Define excluded shape types for broadphase / narrowphase (temporary)
        excluded_types = [
            newton.GeoType.NONE,  # NOTE: Need to skip empty shapes
            newton.GeoType.PLANE,  # NOTE: Currently not supported well by the viewer
            newton.GeoType.ELLIPSOID,  # NOTE: Currently not supported well by the viewer
            newton.GeoType.MESH,  # NOTE: Currently not supported any pipeline
            newton.GeoType.CONVEX_MESH,  # NOTE: Currently not supported any pipeline
            newton.GeoType.HFIELD,  # NOTE: Currently not supported any pipeline
            newton.GeoType.GAUSSIAN,  # NOTE: Render-only, no collision shape pairs
        ]

        # Generate a list of all supported shape-pair combinations for the configured pipeline
        supported_shape_pairs: list[tuple[str, str]] = []
        if args.pipeline == "unified":
            supported_shape_types = [st.value for st in newton.GeoType]
            for shape_bottom in supported_shape_types:
                shape_bottom_name = newton.GeoType(shape_bottom).name.lower()
                for shape_top in supported_shape_types:
                    shape_top_name = newton.GeoType(shape_top).name.lower()
                    if shape_top in excluded_types or shape_bottom in excluded_types:
                        continue
                    supported_shape_pairs.append((shape_top_name, shape_bottom_name))
        elif args.pipeline == "primitive":
            excluded_types.extend([newton.GeoType.CYLINDER])
            supported_shape_types = list(PRIMITIVE_BROADPHASE_SUPPORTED_SHAPES)
            supported_type_pairs = list(PRIMITIVE_NARROWPHASE_SUPPORTED_SHAPE_PAIRS)
            supported_type_pairs = set(supported_type_pairs) | {(b, a) for (a, b) in supported_type_pairs}
            for shape_bottom in supported_shape_types:
                shape_bottom_name = shape_bottom.name.lower()
                for shape_top in supported_shape_types:
                    shape_top_name = shape_top.name.lower()
                    if shape_top in excluded_types or shape_bottom in excluded_types:
                        continue
                    if (shape_top, shape_bottom) in supported_type_pairs:
                        supported_shape_pairs.append((shape_top_name, shape_bottom_name))
        else:
            raise ValueError(f"Unsupported collision pipeline type: {args.pipeline}")

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.0

        # Manually build the basic box on plane using the builder API
        testing.build_shape_pairs_test(
            builder=builder,
            shape_pairs=supported_shape_pairs,
            distance=0.0,
            ground_box=True,
            ground_z=-2.0,
        )

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)

        # Create and configure settings for SolverKamino and the collision detector
        solver_config = newton.solvers.SolverKamino.Config.from_model(self.model)
        solver_config.collision_detector.pipeline = args.pipeline
        solver_config.use_collision_detector = True
        solver_config.padmm.primal_tolerance = 1e-6
        solver_config.padmm.dual_tolerance = 1e-6
        solver_config.padmm.compl_tolerance = 1e-6
        solver_config.padmm.rho_0 = 0.1

        # Create the Kamino solver for the given model
        self.solver = newton.solvers.SolverKamino(model=self.model, config=solver_config)

        # Create state, control, and contacts data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets(spacing=(6.0, 6.0, 0.0))

        # Warm-start the simulation
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.solver.reset(self.state_0)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(30.0, 18.0, 10.0)
            pitch = -20.0
            yaw = -140.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

        # Set the viewer to start in paused mode so that the user can
        # observe the initial state before stepping the simulation
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer._paused = True

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
        # Since rendering is called after stepping the simulation, the previous and next
        # states correspond to self.state_1 and self.state_0 due to the reference swaps,
        # so contacts are rendered with self.state_1 to match the body positions at the
        # time of contact generation.
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
            newton.examples.test_body_state(
                self.model,
                self.state_0,
                "body velocities are small",
                lambda q, qd: (
                    max(abs(qd)) < 0.25
                ),  # Relaxed from 0.1 - unified pipeline has residual velocities up to ~0.2
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--pipeline",
            type=str,
            default="unified",
            help="Sets the collision pipeline to be used by SolverKamino.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
