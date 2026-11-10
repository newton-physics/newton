# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ray casting benchmarks for :func:`newton.intersect_ray` with shape filters.

The scene replicates a robot body of 24 visual sphere meshes and 24 invisible
collision capsules over a global terrain mesh, the setup of an Isaac Lab
height scanner. Each world casts an 11 x 17 grid of rays straight down onto
its robot and the terrain. The ``shape_filter`` parameter selects the shapes
the rays can hit:

- ``default``: visible and colliding shapes, the shape BVH of a scene that
  shares one BVH between cameras and ray casters.
- ``terrain``: the terrain shape group only.
- ``colliders``: the colliding shapes.

Run directly to benchmark::

    uv run asv/benchmarks/simulation/bench_raycast.py
"""

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import numpy as np

import newton

SCAN_SHAPE = (11, 17)
"""Rays per world of the height scan."""


def _robot() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0.0, 0.0, 0.6), wp.quat_identity()), label="robot")
    visual = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
    collider = newton.ModelBuilder.ShapeConfig(is_visible=False)
    mesh = newton.Mesh.create_sphere(0.08, num_latitudes=24, num_longitudes=24)
    for i in range(24):
        xform = wp.transform((0.15 * (i % 6 - 2.5), 0.15 * (i // 6 - 1.5), 0.0), wp.quat_identity())
        builder.add_shape_mesh(body, xform=xform, mesh=mesh, cfg=visual)
        builder.add_shape_capsule(body, xform=xform, radius=0.07, half_height=0.02, cfg=collider)
    return builder


def _terrain(size: float = 200.0, resolution: int = 200) -> newton.Mesh:
    x = np.linspace(-size, size, resolution)
    xx, yy = np.meshgrid(x, x)
    vertices = np.stack([xx.ravel(), yy.ravel(), 0.05 * (np.sin(xx) * np.cos(yy)).ravel()], axis=1)
    index = np.arange(resolution * resolution).reshape(resolution, resolution)
    lower = np.stack([index[:-1, :-1], index[1:, :-1], index[:-1, 1:]], axis=-1)
    upper = np.stack([index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)
    return newton.Mesh(vertices.astype(np.float32), np.concatenate([lower, upper]).ravel())


class IntersectRayHeightScan:
    """Height scan ray casts against a replicated scene, with and without shape filters."""

    params = (["default", "terrain", "colliders"], [4096])
    param_names = ["shape_filter", "world_count"]
    repeat = 8
    number = 1

    def setup(self, shape_filter: str, world_count: int):
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(-1, mesh=_terrain(), cfg=newton.ModelBuilder.ShapeConfig(groups=("terrain",)))
        builder.default_bvh_cfg.shape_flags = newton.ShapeFlags.VISIBLE | newton.ShapeFlags.COLLIDE_SHAPES
        builder.replicate(_robot(), world_count, spacing=(1.0, 1.0, 0.0))
        filters = {
            "default": None,
            "terrain": builder.add_shape_filter(newton.ShapeFilter(groups=["terrain"])),
            "colliders": builder.add_shape_filter(newton.ShapeFilter(flags=newton.ShapeFlags.COLLIDE_SHAPES)),
        }
        self.model = builder.finalize()
        state = self.model.state()

        x, y = np.meshgrid(np.linspace(-0.5, 0.5, SCAN_SHAPE[0]), np.linspace(-0.8, 0.8, SCAN_SHAPE[1]))
        offsets = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
        origins = state.body_q.numpy()[:, None, :3] * np.array([1.0, 1.0, 0.0]) + offsets[None] + (0.0, 0.0, 2.0)
        ray_count = origins.shape[0] * origins.shape[1]
        self.ray_origins = wp.array(origins.reshape(-1, 3), dtype=wp.vec3)
        self.ray_directions = wp.full(ray_count, wp.vec3(0.0, 0.0, -1.0), dtype=wp.vec3)
        self.ray_worlds = wp.array(np.repeat(np.arange(world_count), offsets.shape[0]), dtype=wp.int32)
        self.dist = wp.empty(ray_count, dtype=float)
        self.shape_filter = filters[shape_filter]

        self._cast()
        with wp.ScopedCapture() as capture:
            self._cast()
        self.graph = capture.graph
        wp.synchronize_device()

    def _cast(self):
        newton.intersect_ray(
            self.model,
            ray_origins=self.ray_origins,
            ray_directions=self.ray_directions,
            ray_worlds=self.ray_worlds,
            shape_filter=self.shape_filter,
            out_dist=self.dist,
        )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_intersect_ray(self, shape_filter: str, world_count: int):
        for _ in range(50):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(IntersectRayHeightScan)
