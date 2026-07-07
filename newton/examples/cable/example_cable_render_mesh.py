# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cable Render Mesh
#
# A cable simulated as a chain of capsule rigid bodies (add_rod) drives a
# smooth, textured render tube. Each render vertex is rigidly bound to its
# nearest cable segment and follows that body's pose, so the checkerboard
# tube bends and swings with the cable. Demonstrates the rigid-body
# (cable/rod) path of the deformable render-mesh workflow
# (https://github.com/newton-physics/newton/issues/3223).
#
# Command: uv run -m newton.examples cable_render_mesh
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

        # Cable centerline: a straight horizontal rod, pinned at the left end so
        # it swings down and bends under gravity.
        num_elements = 40
        length = 2.0
        radius = 0.04
        z0 = 1.6
        nodes = np.stack(
            [np.linspace(0.0, length, num_elements + 1), np.zeros(num_elements + 1), np.full(num_elements + 1, z0)],
            axis=1,
        ).astype(np.float32)

        rod_bodies, _ = builder.add_rod(
            positions=[wp.vec3(*p) for p in nodes],
            radius=radius,
            stretch_stiffness=1.0e5,
            bend_stiffness=5.0e1,
            bend_damping=1.0e0,
            label="cable",
            body_frame_origin="com",
        )

        # Pin the first segment.
        first = rod_bodies[0]
        builder.body_mass[first] = 0.0
        builder.body_inv_mass[first] = 0.0
        builder.body_inertia[first] = wp.mat33(0.0)
        builder.body_inv_inertia[first] = wp.mat33(0.0)

        # High-resolution textured tube around the centerline, bound rigidly to
        # the cable segments. A larger radius hides the underlying capsules.
        verts, indices, uvs = self._tube_mesh(nodes, radius=radius * 1.3, segments=20)
        builder.add_deformable_render_mesh(
            verts,
            indices,
            kind="body",
            bodies=rod_bodies,
            uvs=uvs,
            texture=self._checker_texture(),
            label="cable_skin",
        )

        builder.color()

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverVBD(self.model, iterations=self.iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.0, -2.6, 1.2), pitch=-12.0, yaw=100.0)
        self.capture()

    @staticmethod
    def _tube_mesh(centerline: np.ndarray, radius: float, segments: int):
        """Build a tube (vertices, flat triangle indices, UVs) around a polyline."""
        n = len(centerline)
        tangents = np.gradient(centerline, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12

        verts, uvs = [], []
        for i, (p, t) in enumerate(zip(centerline, tangents, strict=True)):
            up = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            nrm = np.cross(t, up)
            nrm /= np.linalg.norm(nrm) + 1e-12
            binrm = np.cross(t, nrm)
            for j in range(segments):
                a = 2.0 * np.pi * j / segments
                offset = np.cos(a) * nrm + np.sin(a) * binrm
                verts.append(p + radius * offset)
                uvs.append([4.0 * i / (n - 1), j / segments])

        faces = []
        for i in range(n - 1):
            for j in range(segments):
                a = i * segments + j
                b = i * segments + (j + 1) % segments
                c = (i + 1) * segments + j
                d = (i + 1) * segments + (j + 1) % segments
                faces += [a, c, b, b, c, d]

        return (
            np.asarray(verts, dtype=np.float32),
            np.asarray(faces, dtype=np.int32),
            np.asarray(uvs, dtype=np.float32),
        )

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
        p_lower = wp.vec3(-3.0, -3.0, -0.5)
        p_upper = wp.vec3(3.0, 3.0, 3.0)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cable bodies are within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(wp.transform_get_translation(q), p_lower, p_upper),
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
