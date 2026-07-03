# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

import newton
import newton.examples


@wp.kernel
def compute_ray_lines(
    site_idx: int,
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    ray_directions: wp.array[wp.vec3],
    distances: wp.array2d[float],
    # output
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
    colors: wp.array[wp.vec3],
):
    """Kernel mapping lidar returns to world-space debug lines (misses are collapsed)."""
    tid = wp.tid()

    X_ws = shape_transform[site_idx]
    body_idx = shape_body[site_idx]
    if body_idx >= 0:
        X_ws = wp.transform_multiply(body_q[body_idx], X_ws)

    origin = X_ws.p
    direction = wp.quat_rotate(X_ws.q, ray_directions[tid])

    dist = distances[0, tid]
    length = wp.max(dist, 0.0)

    starts[tid] = origin
    ends[tid] = origin + direction * length
    colors[tid] = wp.where(dist >= 0.0, wp.vec3(0.9, 0.3, 0.1), wp.vec3(0.3, 0.3, 0.3))


class Example:
    def __init__(self, viewer, args):
        # setup simulation parameters first
        self.fps = 200
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # static pillars surrounding the sensor
        self.pillar_positions = [(3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0)]
        for x, y in self.pillar_positions:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(x, y, 1.0), wp.quat_identity()),
                hx=0.25,
                hy=0.25,
                hz=1.0,
            )

        # falling cube carrying the lidar site on its top face
        scale = 0.2
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=scale, hy=scale, hz=scale, cfg=newton.ModelBuilder.ShapeConfig(density=200))
        self.lidar_site = builder.add_site(
            body, xform=wp.transform(wp.vec3(0.0, 0.0, scale), wp.quat_identity()), label="lidar"
        )

        # finalize model
        self.model = builder.finalize()

        # min_range skips the cube the sensor is mounted on
        self.lidar = newton.sensors.SensorLidar(
            self.model,
            sites="lidar",
            azimuth_count=72,
            elevation_count=4,
            elevation_min=-np.pi / 12.0,
            elevation_max=0.0,
            min_range=0.5,
            max_range=20.0,
        )

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=50)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        self.ray_starts = wp.zeros(self.lidar.n_rays, dtype=wp.vec3)
        self.ray_ends = wp.zeros(self.lidar.n_rays, dtype=wp.vec3)
        self.ray_colors = wp.zeros(self.lidar.n_rays, dtype=wp.vec3)

        self.viewer.set_model(self.model)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(wp.vec3(5.0, 0.0, 3.0), -25.0, self.viewer.camera.yaw)

        # Warm up: run one simulate() step before graph capture to ensure the collision
        # pipeline (and any D2H copies it needs) is initialized outside of capture.
        self.simulate()
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

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

        # the cube carrying the sensor moves, so refit the shape BVH before scanning
        self.model.bvh_refit_shapes(self.state_0)
        self.lidar.update(self.state_0)

        # convert returns to debug lines
        wp.launch(
            compute_ray_lines,
            dim=self.lidar.n_rays,
            inputs=[
                self.lidar_site,
                self.model.shape_body,
                self.model.shape_transform,
                self.state_0.body_q,
                self.lidar.ray_directions,
                self.lidar.distances,
            ],
            outputs=[self.ray_starts, self.ray_ends, self.ray_colors],
        )

        self.solver.update_contacts(self.contacts, self.state_0)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def test_final(self):
        distances = self.lidar.distances.numpy()
        assert distances.shape == (1, self.lidar.n_rays)

        hits = distances[distances >= 0.0]
        assert len(hits) > 0

        # all returns respect the configured range limits
        assert np.all(hits >= self.lidar.min_range)
        assert np.all(hits <= self.lidar.max_range + 1e-4)

        # the downward ring sees the ground well before max_range once the cube has settled
        assert hits.min() < 3.0

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.log_lines("/lidar_rays", self.ray_starts, self.ray_ends, self.ray_colors)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    newton.examples.run(Example(viewer, args), args)
