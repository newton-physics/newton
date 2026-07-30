# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Visual Mesh
#
# A simulated cloth grid drives a textured visual mesh that shares its
# topology. The visual mesh is bound per-particle to the cloth and skinned
# from the simulation state each frame, so the diagnostic texture and UVs
# follow the cloth as it folds and swings. Demonstrates the shared-topology
# (cloth) path of the deformable visual-mesh workflow
# (https://github.com/newton-physics/newton/issues/3223).
#
# Command: uv run -m newton.examples cloth_visual_mesh
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.iterations = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()

        # Static sphere for the cloth to drape over.
        sphere_radius = 0.35
        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = 0.0
        sphere_cfg.ke = 1.0e5
        sphere_cfg.kd = 1.0e1
        sphere_cfg.mu = 0.5
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            radius=sphere_radius,
            cfg=sphere_cfg,
            label="cloth_obstacle",
        )

        # Coarse simulation cloth, released flat above the sphere so it drapes.
        dim = 40
        cell = 0.035
        span = dim * cell
        particle_radius = 0.01
        builder.add_ground_plane()

        particle_start = builder.particle_count
        tri_start = len(builder.tri_indices)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5 * span, -0.5 * span, 1.05),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            cell_x=cell,
            cell_y=cell,
            mass=0.1,
            particle_radius=particle_radius,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e1,
        )
        particle_count = builder.particle_count - particle_start

        # Visual mesh shares the cloth topology: bind visual vertex i to cloth
        # particle (particle_start + i). UVs come from the rest-pose extent so
        # the diagnostic texture maps cleanly across the sheet.
        rest = np.asarray(builder.particle_q[particle_start:], dtype=np.float32)
        indices = (np.asarray(builder.tri_indices[tri_start:], dtype=np.int32) - particle_start).reshape(-1)
        spans = rest.max(axis=0) - rest.min(axis=0)
        u_axis, v_axis = np.argsort(spans)[-2:]
        uvs = (rest[:, [u_axis, v_axis]] - rest[:, [u_axis, v_axis]].min(0)) / spans[[u_axis, v_axis]]

        builder.add_deformable_visual_mesh(
            rest,
            indices,
            kind="particle",
            particles=np.arange(particle_start, particle_start + particle_count, dtype=np.int32),
            uvs=uvs.astype(np.float32),
            texture=self._diagnostic_texture(),
            label="cloth_skin",
        )

        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 0.5

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        # The visual mesh shares the cloth topology; hide the raw simulation surface
        # so the textured skin does not z-fight with it.
        self.viewer.show_triangles = False
        self.viewer.set_camera(pos=wp.vec3(1.6, -1.8, 1.2), pitch=-18.0, yaw=135.0)
        self.capture()

    @staticmethod
    def _diagnostic_texture() -> str:
        """Return the shared asymmetric UV diagnostic texture."""
        return newton.examples.get_asset("deformable_visual_uv_grid.png")

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
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        p_lower = wp.vec3(-2.0, -2.0, 0.0)
        p_upper = wp.vec3(2.0, 2.0, 2.5)
        newton.examples.test_particle_state(
            self.state_0,
            "cloth particles are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
