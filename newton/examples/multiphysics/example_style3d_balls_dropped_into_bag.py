# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop five rigid balls into a plastic bag suspended by its two handles.

The open bag and its handles form one cloth mesh. SolverStyle3D advances the
cloth and affine bodies together, including two-way particle-body contact,
at 60 frames per second with two substeps per frame.

Success criteria:
- Five balls fall through the upward-facing opening and remain in the bag.
- The bag deforms under their weight without touching the ground.
- The upper ends of both handles remain fixed; the remaining cloth can move.
- Positions and velocities remain finite.

Runs:
- `python -m newton.examples style3d_balls_dropped_into_bag`
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverStyle3D, style3d


def _make_bag():
    """Join five rectangular panels and cut handle holes in the front and back."""
    vertices, indices, panel_vertices, panel_indices = [], [], [], []
    shared = {}

    def panel(origin, u, v, nu, nv, handles=False):
        origin, u, v = np.asarray(origin), np.asarray(u), np.asarray(v)
        points = {}
        uv = {}
        for j in range(nv):
            for i in range(nu):
                # Above the mouth, retain two straps and their connecting bar.
                if handles and j >= 22 and not (6 <= i < 8 or 16 <= i < 18 or (j >= 26 and 6 <= i < 18)):
                    continue
                corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
                for a, b in corners:
                    if (a, b) not in points:
                        position = origin + (a / nu) * u + (b / nv) * v
                        key = tuple(np.round(position, 8))
                        if key not in shared:
                            shared[key] = len(vertices)
                            vertices.append(position.tolist())
                        points[a, b] = shared[key]
                        uv[a, b] = len(panel_vertices)
                        panel_vertices.append((a * np.linalg.norm(u) / nu, b * np.linalg.norm(v) / nv))
                for triangle in ((0, 1, 2), (0, 2, 3)):
                    indices.extend(points[corners[k]] for k in triangle)
                    panel_indices.extend(uv[corners[k]] for k in triangle)

    # Shared seam vertices keep the bottom and sides closed, with no stitch constraints.
    panel((-0.3, -0.175, 0.45), (0.6, 0.0, 0.0), (0.0, 0.0, 0.7), 24, 28, handles=True)
    panel((0.3, 0.175, 0.45), (-0.6, 0.0, 0.0), (0.0, 0.0, 0.7), 24, 28, handles=True)
    panel((-0.3, 0.175, 0.45), (0.0, -0.35, 0.0), (0.0, 0.0, 0.55), 14, 22)
    panel((0.3, -0.175, 0.45), (0.0, 0.35, 0.0), (0.0, 0.0, 0.55), 14, 22)
    panel((-0.3, 0.175, 0.45), (0.6, 0.0, 0.0), (0.0, -0.35, 0.0), 24, 14)
    return np.asarray(vertices, dtype=np.float32), indices, panel_vertices, panel_indices


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        SolverStyle3D.register_custom_attributes(builder)
        vertices, indices, panel_vertices, panel_indices = _make_bag()
        style3d.add_cloth_mesh(
            builder,
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            panel_verts=panel_vertices,
            panel_indices=panel_indices,
            density=0.2,
            particle_radius=0.004,
            tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e1),
            edge_aniso_ke=wp.vec3(2.0e-7),
            tri_kd=0.0,
            label="plastic_bag",
        )
        self.fixed_indices = np.flatnonzero(np.isclose(vertices[:, 2], vertices[:, 2].max()))
        self.fixed_positions = vertices[self.fixed_indices].copy()
        for index in self.fixed_indices:
            builder.particle_flags[index] &= ~newton.ParticleFlags.ACTIVE

        self.ball_radius = 0.07
        ball_cfg = newton.ModelBuilder.ShapeConfig(density=25.0, ke=2.0e4, kd=2.0, kf=10.0, mu=0.3)
        positions = (
            (-0.13, -0.04, 1.15),
            (0.13, 0.04, 1.28),
            (0.0, 0.0, 1.48),
            (-0.10, 0.03, 1.68),
            (0.10, -0.03, 1.88),
        )
        colors = ((0.95, 0.3, 0.1), (0.15, 0.5, 0.95), (0.95, 0.7, 0.1), (0.55, 0.25, 0.85), (0.15, 0.75, 0.4))
        for i, (position, color) in enumerate(zip(positions, colors, strict=True)):
            body = builder.add_body(xform=wp.transform(position, wp.quat_identity()), label=f"ball_{i}")
            builder.add_shape_sphere(body, radius=self.ball_radius, cfg=ball_cfg, color=color)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.model.tri_color = wp.full(self.model.tri_count, wp.vec3(0.65, 0.85, 0.95), device=self.model.device)
        self.model.tri_opacity = wp.full(self.model.tri_count, 0.4, device=self.model.device)
        self.model.soft_contact_ke = 2.0e3
        self.model.soft_contact_kd = 0.2
        self.model.soft_contact_kf = 1.0
        self.model.soft_contact_mu = 0.3
        self.solver = SolverStyle3D(
            self.model,
            config=SolverStyle3D.Config(iterations=15),
            dynamics=SolverStyle3D.Dynamics.ALL,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model, soft_contact_gap=0.04)
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.7, -2.4, 2.1), -18.0, 125.0)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def test_post_step(self):
        """Keep both handle tops fixed and the coupled simulation finite."""
        particles = self.state_0.particle_q.numpy()
        np.testing.assert_array_equal(particles[self.fixed_indices], self.fixed_positions)
        for values in (
            particles,
            self.state_0.particle_qd.numpy(),
            self.state_0.body_q.numpy(),
            self.state_0.body_qd.numpy(),
        ):
            assert np.isfinite(values).all(), "The bag or balls became non-finite"

    def test_final(self):
        """Keep all five balls suspended inside the deformed bag."""
        particles = self.state_0.particle_q.numpy()
        centers = self.state_0.body_q.numpy()[:, :3]
        assert particles[:, 2].min() > 0.1, "The bag reached the floor"
        assert np.all(centers[:, 2] > particles[:, 2].min() + 0.04), "A ball fell through the bottom"
        assert np.all(centers[:, 2] < 1.0), "A ball did not enter the bag"
        assert np.all(np.abs(centers[:, 0]) < 0.35), "A ball escaped sideways"
        assert np.all(np.abs(centers[:, 1]) < 0.25), "A ball escaped sideways"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
