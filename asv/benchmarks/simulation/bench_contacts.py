# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import importlib
from typing import ClassVar

import numpy as np

import newton.examples
from newton.viewer import ViewerNull

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
IRREGULAR_ROCK_VERTEX_COUNTS = (10, 14, 18, 26)
CONVEX_COLLISION_CASES = (("hulls", 56), ("hulls_duplicate", 192), ("mixed", 191))
MIXED_CONVEX_PAIR_TYPES = (
    ("sphere", "sphere"),
    ("capsule", "capsule"),
    ("sphere", "capsule"),
    ("box", "hull"),
    ("ellipsoid", "box"),
    ("cylinder", "cylinder"),
    ("cone", "cylinder"),
    ("capsule", "hull"),
    ("sphere", "cone"),
    ("ellipsoid", "hull"),
    ("box", "box"),
    ("hull", "hull"),
    ("cone", "hull"),
    ("capsule", "ellipsoid"),
    ("sphere", "hull"),
    ("cylinder", "box"),
)

try:
    from newton.examples import download_external_git_folder as _download_external_git_folder
except ImportError:
    from newton._src.utils.download_assets import download_git_folder as _download_external_git_folder


def _import_example_class(module_names: list[str]):
    """Import and return the ``Example`` class from candidate modules.

    Args:
        module_names: Ordered module names to try importing.

    Returns:
        The first successfully imported module's ``Example`` class.

    Raises:
        SkipNotImplemented: If none of the module names can be imported.
    """
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        return module.Example

    raise SkipNotImplemented


def _make_irregular_rock(vertex_count: int, seed: int, triangle_local_vertices: bool = False) -> newton.Mesh:
    """Create a closed irregular convex bipyramid for collision benchmarks."""
    ring_count = vertex_count - 2
    rng = np.random.default_rng(seed)
    vertices = []
    for index in range(ring_count):
        angle = 2.0 * np.pi * index / ring_count
        radius = 0.42 * rng.uniform(0.82, 1.18)
        vertices.append([radius * np.cos(angle), radius * np.sin(angle), rng.uniform(-0.09, 0.09)])

    vertices.extend(
        [
            [0.04, -0.03, rng.uniform(0.48, 0.60)],
            [-0.03, 0.04, -rng.uniform(0.48, 0.60)],
        ]
    )
    top = ring_count
    bottom = ring_count + 1
    indices = []
    for index in range(ring_count):
        next_index = (index + 1) % ring_count
        indices.extend([top, index, next_index])
        indices.extend([bottom, next_index, index])

    vertices = np.asarray(vertices, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int32)
    if triangle_local_vertices:
        vertices = vertices[indices]
        indices = np.arange(len(indices), dtype=np.int32)
    return newton.Mesh(vertices, indices)


def _add_mixed_convex_shape(
    builder: newton.ModelBuilder,
    body: int,
    shape_kind: str,
    rocks: list[newton.Mesh],
    rock_index: int,
    cfg: newton.ModelBuilder.ShapeConfig,
) -> None:
    """Add one convex shape to the mixed collision workload."""
    if shape_kind == "sphere":
        builder.add_shape_sphere(body, radius=0.5, cfg=cfg)
    elif shape_kind == "box":
        builder.add_shape_box(body, hx=0.48, hy=0.42, hz=0.45, cfg=cfg)
    elif shape_kind == "capsule":
        builder.add_shape_capsule(body, radius=0.32, half_height=0.25, cfg=cfg)
    elif shape_kind == "ellipsoid":
        builder.add_shape_ellipsoid(body, rx=0.52, ry=0.43, rz=0.38, cfg=cfg)
    elif shape_kind == "cylinder":
        builder.add_shape_cylinder(body, radius=0.48, half_height=0.45, cfg=cfg)
    elif shape_kind == "cone":
        builder.add_shape_cone(body, radius=0.5, half_height=0.48, cfg=cfg)
    elif shape_kind == "hull":
        builder.add_shape_convex_hull(body, mesh=rocks[rock_index % len(rocks)], cfg=cfg)
    else:
        raise ValueError(f"Unsupported convex shape kind: {shape_kind}")


def _build_convex_scene(
    world_count: int,
    pair_types: tuple[tuple[str, str], ...],
    *,
    triangle_local_vertices: bool = False,
) -> newton.Model:
    """Build replicated isolated convex pairs."""
    newton.use_coord_layout_targets = True
    rocks = [
        _make_irregular_rock(count, 100 + index, triangle_local_vertices)
        for index, count in enumerate(IRREGULAR_ROCK_VERTEX_COUNTS)
    ]

    world_builder = newton.ModelBuilder()
    shape_cfg = newton.ModelBuilder.ShapeConfig(gap=0.01, margin=0.0)
    axis = wp.normalize(wp.vec3(0.3, 0.2, 1.0))
    for pair_index, (shape_a, shape_b) in enumerate(pair_types):
        x = 3.0 * (pair_index % 4)
        y = 3.0 * (pair_index // 4)
        angle = 0.11 * pair_index
        body_a = world_builder.add_body(xform=wp.transform(wp.vec3(x, y, 1.0), wp.quat_from_axis_angle(axis, angle)))
        body_b = world_builder.add_body(
            xform=wp.transform(
                wp.vec3(x + 0.84, y + 0.03 * ((pair_index % 3) - 1), 1.0),
                wp.quat_from_axis_angle(axis, -0.7 * angle),
            )
        )
        _add_mixed_convex_shape(world_builder, body_a, shape_a, rocks, 2 * pair_index, shape_cfg)
        _add_mixed_convex_shape(world_builder, body_b, shape_b, rocks, 2 * pair_index + 1, shape_cfg)

    builder = newton.ModelBuilder()
    builder.replicate(world_builder, world_count=world_count)
    return builder.finalize()


class FastExampleContactSdfDefaults:
    """Benchmark the SDF nut-bolt example default configuration."""

    repeat = 2
    number = 1

    def setup_cache(self):
        _download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_nut_bolt_sdf",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                world_count=100,
                num_per_world=1,
                scene="nut_bolt",
                solver="mujoco",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactHydroWorkingDefaults:
    """Benchmark the hydroelastic nut-bolt example default configuration."""

    repeat = 2
    number = 1

    def setup_cache(self):
        _download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_nut_bolt_hydro",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                world_count=20,
                num_per_world=1,
                scene="nut_bolt",
                solver="mujoco",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastExampleContactPyramidDefaults:
    """Benchmark the box pyramid example with default configuration."""

    repeat = 2
    number = 1

    def setup(self):
        example_cls = _import_example_class(
            [
                "newton.examples.contacts.example_pyramid",
            ]
        )
        self.num_frames = 20
        if hasattr(newton.examples, "default_args") and hasattr(example_cls, "create_parser"):
            args = newton.examples.default_args(example_cls.create_parser())
            self.example = example_cls(ViewerNull(num_frames=self.num_frames), args)
        else:
            self.example = example_cls(
                viewer=ViewerNull(num_frames=self.num_frames),
                solver="xpbd",
                test_mode=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.step()
        wp.synchronize_device()


class FastConvexCollision:
    """Benchmark lean-hull and mixed-type convex collision workloads."""

    params = (CONVEX_COLLISION_CASES,)
    param_names: ClassVar[list[str]] = ["case"]
    repeat = 5
    number = 1

    def setup(self, case):
        device = wp.get_device()
        if not device.is_cuda or not wp.is_mempool_enabled(device):
            raise SkipNotImplemented

        self.launch_count = 100
        scene, world_count = case
        if scene in ("hulls", "hulls_duplicate"):
            pair_types = (("hull", "hull"),) * len(MIXED_CONVEX_PAIR_TYPES)
        elif scene == "mixed":
            pair_types = MIXED_CONVEX_PAIR_TYPES
        else:
            raise ValueError(f"Unsupported convex benchmark scene: {scene}")
        self.model = _build_convex_scene(
            world_count,
            pair_types,
            triangle_local_vertices=scene == "hulls_duplicate",
        )
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=self.model.shape_count * 8,
            verify_buffers=False,
        )
        self.contacts = self.collision_pipeline.contacts()

        for _ in range(5):
            self.collision_pipeline.collide(self.state, self.contacts)
        if int(self.collision_pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]) == 0:
            raise RuntimeError("convex benchmark scene produced no GJK candidate pairs")

        with wp.ScopedCapture(device=device) as capture:
            self.collision_pipeline.collide(self.state, self.contacts)
        self.graph = capture.graph

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide(self, case):
        for _ in range(self.launch_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleContactSdfDefaults": FastExampleContactSdfDefaults,
        "FastExampleContactHydroWorkingDefaults": FastExampleContactHydroWorkingDefaults,
        "FastExampleContactPyramidDefaults": FastExampleContactPyramidDefaults,
        "FastConvexCollision": FastConvexCollision,
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
