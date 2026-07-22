# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverImplicitMPM


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps

        # group related attributes by prefix
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # save a reference to the viewer
        self.viewer = viewer
        builder = newton.ModelBuilder()

        # Particle samples, point-subset materials, and solver configuration are authored in USD.
        # This fixture derives particle masses from bound physics:density values and authored widths.
        import_result = builder.add_usd(
            newton.examples.get_asset("mpm_sand.usda"),
            load_visual_shapes=False,
        )
        mpm_options = import_result["mpm_config"]
        if mpm_options is None:
            raise ValueError("mpm_sand.usda must author NewtonMPMSceneAPI")

        # Setup collision geometry
        self.collider = args.collider
        if self.collider == "concave":
            extents = (1.0, 2.0, 0.25)
            left_xform = wp.transform(
                wp.vec3(-0.7, 0.0, 0.8), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0)
            )
            right_xform = wp.transform(
                wp.vec3(0.7, 0.0, 0.8), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -np.pi / 4.0)
            )

            builder.add_shape_box(
                body=-1,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.1, density=0.0),
                xform=left_xform,
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
            )
            builder.add_shape_box(
                body=-1,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.1, density=0.0),
                xform=right_xform,
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
            )
        elif self.collider != "none":
            if self.collider == "cube":
                extents = (0.5, 2.0, 0.8)
                xform = wp.transform(wp.vec3(0.75, 0.0, 0.8), wp.quat_identity())
            elif self.collider == "wedge":
                extents = (0.5, 2.0, 0.5)
                xform = wp.transform(
                    wp.vec3(0.1, 0.0, 0.5), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 4.0)
                )

            builder.add_shape_box(
                body=-1,
                cfg=newton.ModelBuilder.ShapeConfig(mu=0.1),
                xform=xform,
                hx=extents[0],
                hy=extents[1],
                hz=extents[2],
            )

        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))

        self.model = builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Initialize MPM solver
        self.solver = SolverImplicitMPM(self.model, config=mpm_options)

        self.viewer.set_model(self.model)

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

        self.viewer.show_particles = True
        self.show_normals = False
        self.show_stress = False

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.sim_substeps % 2 != 0:
                warnings.warn("Sim substeps must be even for graph capture of MPM step", stacklevel=2)
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        voxel_size = self.solver.voxel_size
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -voxel_size,
        )

        if self.collider == "cube":
            cube_extents = wp.vec3(0.5, 2.0, 0.8) - wp.vec3(voxel_size)
            cube_center = wp.vec3(0.75, 0, 0.8)
            cube_lower = cube_center - cube_extents
            cube_upper = cube_center + cube_extents
            newton.examples.test_particle_state(
                self.state_0,
                "all particles are outside the cube",
                lambda q, qd: not newton.math.vec_inside_limits(q, cube_lower, cube_upper),
            )

        # Test that some particles are still high-enough
        if self.collider in ("concave", "cube"):
            max_z = np.max(self.state_0.particle_q.numpy()[:, 2])
            assert max_z > 0.8, "All particles have collapsed"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        if self.show_normals:
            # for debugging purposes, we can visualize the collider normals
            _impulses, pos, _cid = self.solver.collect_collider_impulses(self.state_0)
            normals = self.state_0.collider_normal_field.dof_values

            normal_vecs = 0.25 * self.solver.voxel_size * normals
            root = pos
            mid = pos + normal_vecs
            tip = mid + normal_vecs

            # draw two segments per normal so we can visualize direction (red roots, orange tips)
            self.viewer.log_lines(
                "/normal_roots",
                starts=root,
                ends=mid,
                colors=wp.full(pos.shape[0], value=wp.vec3(0.8, 0.0, 0.0), dtype=wp.vec3),
            )
            self.viewer.log_lines(
                "/normal_tips",
                starts=mid,
                ends=tip,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.5, 0.3), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/normal_roots", None, None, None)
            self.viewer.log_lines("/normal_tips", None, None, None)

        self.viewer.end_frame()

    def render_ui(self, imgui):
        _changed, self.show_normals = imgui.checkbox("Show Normals", self.show_normals)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()

        # Scene configuration
        parser.add_argument("--collider", default="cube", choices=["cube", "wedge", "concave", "none"], type=str)
        parser.add_argument("--fps", type=float, default=60.0)
        parser.add_argument("--substeps", type=int, default=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example and run
    newton.examples.run(Example(viewer, args), args)
