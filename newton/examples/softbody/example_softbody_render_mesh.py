# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody Render Mesh
#
# A coarse tetrahedral soft body (a low-resolution cube) drives a
# high-resolution render mesh embedded inside it. The render mesh is a
# detailed UV sphere with texture coordinates; it is bound to the cube's
# tetrahedra at build time and skinned from the simulation state each frame
# for visualization only, demonstrating the deformable render-mesh workflow
# from https://github.com/newton-physics/newton/issues/3223.
#
# Command: uv run -m newton.examples softbody.example_softbody_render_mesh
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
        builder.add_ground_plane()

        # Coarse simulation cube spanning [0, L]^3, fixed on the left so it sags.
        dim = 3
        cell = 0.2
        length = dim * cell
        origin = wp.vec3(0.0, 0.0, 1.0)
        builder.add_soft_grid(
            pos=origin,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            dim_z=dim,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            density=1.0e3,
            k_mu=2.0e4,
            k_lambda=2.0e4,
            k_damp=1.0e1,
            fix_left=True,
        )

        # High-resolution render mesh: a UV sphere that fits inside the cube,
        # so every render vertex embeds in a tetrahedron. A checkerboard texture
        # makes the deformation of the embedded surface easy to see.
        center = np.array([origin[0], origin[1], origin[2]], dtype=np.float32) + 0.5 * length
        sphere = newton.Mesh.create_sphere(radius=0.45 * length, num_latitudes=48, num_longitudes=48)
        render_verts = np.asarray(sphere.vertices, dtype=np.float32) + center
        render_indices = np.asarray(sphere.indices, dtype=np.int32)
        render_uvs = sphere._uvs

        builder.add_deformable_render_mesh(
            render_verts,
            render_indices,
            kind="tet",
            uvs=render_uvs,
            texture=self._checker_texture(),
            label="sphere_skin",
        )

        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_mu = 1.0

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            particle_enable_self_contact=False,
            particle_enable_tile_solve=False,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.4, -1.3, 1.45), pitch=-8.0, yaw=90.0)
        self.capture()

    @staticmethod
    def _checker_texture(tiles: int = 8, size: int = 512) -> np.ndarray:
        """Build an RGB checkerboard texture (H, W, 3) uint8."""
        image = np.zeros((size, size, 3), dtype=np.uint8)
        step = size // tiles
        for i in range(tiles):
            for j in range(tiles):
                color = (235, 90, 40) if (i + j) % 2 else (40, 120, 255)
                image[i * step : (i + 1) * step, j * step : (j + 1) * step] = color
        return image

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
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        # The coarse cube should deform but stay within a reasonable volume.
        p_lower = wp.vec3(-1.0, -1.0, 0.0)
        p_upper = wp.vec3(2.0, 2.0, 2.5)
        newton.examples.test_particle_state(
            self.state_0,
            "soft body particles are within a reasonable volume",
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
