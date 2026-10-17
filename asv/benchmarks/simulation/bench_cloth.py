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
from newton._src.geometry.tri_mesh_collision import TriMeshCollisionDetector, TriMeshCollisionInfo
from newton._src.solvers.vbd.particle_vbd_kernels import (
    NUM_THREADS_PER_COLLISION_PRIMITIVE,
    accumulate_self_contact_force_and_hessian,
)
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


class FastDeformableSelfCollision:
    """Benchmark self-collision detection in one large cloth and many replicated worlds."""

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
            vertex_collision_buffer_pre_alloc=64,
            edge_collision_buffer_pre_alloc=64,
        )
        self.radius = 0.012

        for _ in range(self.warmup_count):
            self._detect()
        if self.detector.resize_flags.numpy().any():
            raise RuntimeError("collision buffers overflowed; increase the pre-allocated sizes")
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


class FastVBDSelfContactForces:
    """Accumulate fixed self-contact candidates over a complete particle-color sweep."""

    params = (["empty", "sparse", "dense"], [3, 8])
    param_names = ["candidate_density", "color_count"]
    repeat = pr_gate_repeat(5)
    number = 1
    launch_count = 10

    def setup(self, candidate_density, color_count):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented

        builder = _make_collision_world(32)
        builder.color()
        if color_count == 8:
            # Refine a valid three-color partition without changing the mesh or candidates.
            groups = []
            for group, split_count in zip(builder.particle_color_groups, (3, 3, 2), strict=True):
                groups.extend(np.array_split(group, split_count))
            builder.set_coloring(groups)
        self.model = builder.finalize(device=device)
        self.detector = TriMeshCollisionDetector(
            self.model,
            init_collision_info=True,
            topological_contact_filter_threshold=2,
            vertex_collision_buffer_pre_alloc=1024,
            edge_collision_buffer_pre_alloc=1024,
        )
        query_radius = {"empty": 0.001, "sparse": 0.012, "dense": 0.032}[candidate_density]
        self.detector.refit(self.model.particle_q)
        self.detector.vertex_triangle_collision_detection(query_radius)
        self.detector.edge_edge_collision_detection(query_radius)
        if self.detector.resize_flags.numpy().any():
            raise RuntimeError("collision buffers overflowed; increase the pre-allocated sizes")

        self.forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=device)
        self.hessians = wp.zeros(self.model.particle_count, dtype=wp.mat33, device=device)
        self.collision_info = wp.array([self.detector.collision_info], dtype=TriMeshCollisionInfo, device=device)
        self._accumulate()
        force = self.forces.numpy()
        if (
            not np.isfinite(force).all()
            or not np.isfinite(self.hessians.numpy()).all()
            or (candidate_density != "empty" and not np.any(force))
        ):
            raise RuntimeError("self-contact benchmark must produce finite, nonzero forces")
        with wp.ScopedCapture(device=device) as capture:
            self._accumulate()
        self.graph = capture.graph

    def _accumulate(self):
        self.forces.zero_()
        self.hessians.zero_()
        for color in range(len(self.model.particle_color_groups)):
            wp.launch(
                accumulate_self_contact_force_and_hessian,
                dim=max(self.model.particle_count, self.model.edge_count) * NUM_THREADS_PER_COLLISION_PRIMITIVE,
                inputs=[
                    1.0 / 120.0,
                    color,
                    self.model.particle_q,
                    self.model.particle_q,
                    self.model.particle_colors,
                    self.model.tri_indices,
                    self.model.edge_indices,
                    self.collision_info,
                    0.01,
                    100.0,
                    0.0,
                    0.0,
                    0.01,
                    1.0e-5,
                ],
                outputs=[self.forces, self.hessians],
                device=self.model.device,
                max_blocks=self.model.device.sm_count,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_accumulate(self, candidate_density, color_count):
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
        "FastDeformableSelfCollision": FastDeformableSelfCollision,
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
