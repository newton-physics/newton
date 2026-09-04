# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.log_level = wp.LOG_WARNING

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_config import pr_gate_repeat

import newton.examples
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionDetector
from newton.examples.cloth.example_cloth_franka import Example as ExampleClothManipulation
from newton.examples.cloth.example_cloth_twist import Example as ExampleClothTwist
from newton.viewer import ViewerNull

DEFORMABLE_COLLISION_CASES = ((256, 1), (16, 1024))


def _make_collision_grid(resolution, height):
    x, y = np.meshgrid(np.arange(resolution) * 0.01, np.arange(resolution) * 0.01)
    vertices = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height))).astype(np.float32)
    triangles = []
    for row in range(resolution - 1):
        for column in range(resolution - 1):
            lower = row * resolution + column
            triangles.extend(
                ((lower, lower + 1, lower + resolution), (lower + 1, lower + resolution + 1, lower + resolution))
            )
    return vertices, np.asarray(triangles, dtype=np.int32)


def _make_collision_world(resolution):
    vertices_a, triangles_a = _make_collision_grid(resolution, 0.0)
    vertices_b, triangles_b = _make_collision_grid(resolution, 0.006)
    triangles_b += len(vertices_a)
    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    world.add_cloth_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=np.concatenate((vertices_a, vertices_b)),
        indices=np.concatenate((triangles_a, triangles_b)).reshape(-1),
        density=1.0,
        tri_ke=1.0,
        tri_ka=1.0,
        tri_kd=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    return world


class DeformableSelfCollision:
    """Benchmark dense self-collision in one large and many RL-style worlds."""

    params = (DEFORMABLE_COLLISION_CASES,)
    param_names = ["case"]
    repeat = 3
    number = 1
    warmup_count = 3
    launch_count = 10

    def setup(self, case):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented

        resolution, world_count = case
        builder = newton.ModelBuilder()
        builder.replicate(_make_collision_world(resolution), world_count)
        self.model = builder.finalize(device=device)
        self.detector = TriMeshCollisionDetector(
            self.model,
            init_collision_info=True,
            topological_contact_filter_threshold=0,
            vertex_collision_buffer_pre_alloc=32,
            edge_collision_buffer_pre_alloc=64,
        )
        self.radius = 0.012

        for _ in range(self.warmup_count):
            self._detect()
        with wp.ScopedCapture(device=device) as capture:
            self._detect()
        self.graph = capture.graph

    def _detect(self):
        self.detector.refit(self.model.particle_q)
        self.detector.vertex_triangle_collision_detection(self.radius)
        self.detector.edge_edge_collision_detection(self.radius)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_detect(self, case):
        for _ in range(self.launch_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


class FastExampleClothManipulation:
    timeout = 300
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = 30
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        self.example = ExampleClothManipulation(ViewerNull(num_frames=self.num_frames), args)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, args=None)

        wp.synchronize_device()


class FastExampleClothTwist:
    repeat = pr_gate_repeat(5)
    number = 1

    def setup(self):
        self.num_frames = 100
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        self.example = ExampleClothTwist(ViewerNull(num_frames=self.num_frames), args)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, None)

        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "DeformableSelfCollision": DeformableSelfCollision,
        "FastExampleClothManipulation": FastExampleClothManipulation,
        "FastExampleClothTwist": FastExampleClothTwist,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
