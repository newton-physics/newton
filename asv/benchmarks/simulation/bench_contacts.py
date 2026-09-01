# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import importlib
import os
import sys
from typing import ClassVar

import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_config import pr_gate_repeat

import newton.examples
from newton.viewer import ViewerNull

ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"
IRREGULAR_ROCK_VERTEX_COUNTS = (10, 14, 18, 26)
CONVEX_COLLISION_CASES = (("hulls", 56), ("hulls_duplicate", 192), ("mixed", 191))
BROAD_PHASE_COLLISION_CASES = (("sap", 10_000), ("nxn", 1_000), ("explicit", 10_000))
COMPLEX_CONTACT_CASES = ("mesh_convex", "mesh_sdf")
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


def _build_single_world_scene(pair_count: int) -> newton.Model:
    """Build sparse mixed contacts in one large global SAP segment."""
    newton.use_coord_layout_targets = True
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    shape_cfg = newton.ModelBuilder.ShapeConfig(gap=0.01, margin=0.0)
    rock = _make_irregular_rock(26, seed=701)
    pair_types = (("sphere", "sphere"), ("box", "box"), ("box", "hull"), ("hull", "hull"))

    for pair_index in range(pair_count):
        x = 2.5 * pair_index
        body_a = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0)))
        body_b = builder.add_body(xform=wp.transform(wp.vec3(x + 0.72, 0.02, 0.01)))
        shape_a, shape_b = pair_types[pair_index % len(pair_types)]
        _add_mixed_convex_shape(builder, body_a, shape_a, [rock], 0, shape_cfg)
        _add_mixed_convex_shape(builder, body_b, shape_b, [rock], 0, shape_cfg)

    return builder.finalize()


def _make_two_sided_grid(resolution: int, half_extent: float) -> newton.Mesh:
    """Create a flat two-sided triangle grid for dense mesh contacts."""
    axis = np.linspace(-half_extent, half_extent, resolution + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    vertices = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size, dtype=np.float32)))
    triangles = []
    row = resolution + 1
    for y in range(resolution):
        for x in range(resolution):
            i0 = y * row + x
            i1 = i0 + 1
            i2 = i0 + row
            i3 = i2 + 1
            triangles.extend(((i0, i1, i3), (i0, i3, i2), (i3, i1, i0), (i2, i3, i0)))
    return newton.Mesh(vertices, np.asarray(triangles, dtype=np.int32), compute_inertia=False)


def _build_mesh_convex_scene(world_count: int, resolution: int) -> tuple[newton.Model, int]:
    """Build replicated dense mesh-convex contacts without external assets."""
    grid = _make_two_sided_grid(resolution, half_extent=1.0)
    hull = newton.Mesh.create_sphere(
        0.55,
        num_latitudes=24,
        num_longitudes=48,
        compute_normals=False,
        compute_uvs=False,
    )
    shape_cfg = newton.ModelBuilder.ShapeConfig(gap=0.002, margin=0.0)
    world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    world.add_shape_mesh(body=-1, mesh=grid, cfg=shape_cfg)
    body = world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.20)))
    world.add_shape_convex_hull(body=body, mesh=hull, cfg=shape_cfg)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(world, world_count=world_count)
    triangles_per_world = 4 * resolution * resolution
    return builder.finalize(), triangles_per_world * world_count


def _build_mesh_sdf_scene(world_count: int, device) -> newton.Model:
    """Build replicated regular mesh-SDF contacts without external assets."""
    mesh_a = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False)
    mesh_b = newton.Mesh.create_box(0.5, 0.5, 0.5, duplicate_vertices=False)
    mesh_a.build_sdf(max_resolution=32, device=device)
    mesh_b.build_sdf(max_resolution=32, device=device)

    world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body_a = world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0)))
    body_b = world.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.9)))
    world.add_shape_mesh(body=body_a, mesh=mesh_a)
    world.add_shape_mesh(body=body_b, mesh=mesh_b)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(world, world_count=world_count)
    return builder.finalize(device=device)


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
    repeat = pr_gate_repeat(5)
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


class BroadPhaseCollision:
    """Benchmark sparse contacts through every rigid broad phase."""

    params = (BROAD_PHASE_COLLISION_CASES,)
    param_names: ClassVar[list[str]] = ["case"]
    repeat = 5
    number = 1

    def setup(self, case):
        device = wp.get_device()
        if not device.is_cuda or not wp.is_mempool_enabled(device):
            raise SkipNotImplemented

        broad_phase, pair_count = case
        self.launch_count = 20
        self.model = _build_single_world_scene(pair_count)
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        pipeline_kwargs = {
            "broad_phase": broad_phase,
            "rigid_contact_max": 8 * pair_count,
            "contact_matching": "latest",
            "verify_buffers": False,
        }
        if broad_phase == "explicit":
            shape_indices = np.arange(2 * pair_count, dtype=np.int32).reshape(-1, 2)
            pipeline_kwargs["shape_pairs_filtered"] = wp.array(shape_indices, dtype=wp.vec2i, device=device)
        else:
            pipeline_kwargs["shape_pairs_max"] = 32_768
        self.collision_pipeline = newton.CollisionPipeline(self.model, **pipeline_kwargs)
        self.contacts = self.collision_pipeline.contacts()
        if broad_phase == "sap" and not self.collision_pipeline.broad_phase._single_segment_identity_map:
            raise RuntimeError("SAP benchmark did not select the single-segment identity-map path")

        for _ in range(3):
            self.collision_pipeline.collide(self.state, self.contacts)
        wp.synchronize_device()
        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        candidate_count = int(self.collision_pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0])
        match_indices = self.contacts.rigid_contact_match_index.numpy()[:contact_count]
        if contact_count < pair_count or candidate_count == 0 or not np.any(match_indices >= 0):
            raise RuntimeError("broad-phase benchmark did not produce the intended persistent mixed contacts")
        if broad_phase == "sap" and not self.collision_pipeline.narrow_phase.split_gjk_mpr:
            raise RuntimeError("SAP benchmark did not select split GJK/MPR")

        with wp.ScopedCapture(device=device) as capture:
            self.collision_pipeline.collide(self.state, self.contacts)
        self.graph = capture.graph

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide(self, case):
        for _ in range(self.launch_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


class ComplexContactCollision:
    """Benchmark dense mesh-convex manifolds and regular mesh-SDF contacts."""

    params = (COMPLEX_CONTACT_CASES,)
    param_names: ClassVar[list[str]] = ["case"]
    repeat = 5
    number = 1

    def setup(self, case):
        device = wp.get_device()
        if not device.is_cuda or not wp.is_mempool_enabled(device):
            raise SkipNotImplemented

        self.launch_count = 20
        world_count = 64
        if case == "mesh_convex":
            self.model, max_triangle_pairs = _build_mesh_convex_scene(world_count, resolution=24)
            pipeline_kwargs = {
                "broad_phase": "explicit",
                "max_triangle_pairs": max_triangle_pairs,
                "rigid_contact_max": 1024 * world_count,
            }
        elif case == "mesh_sdf":
            self.model = _build_mesh_sdf_scene(world_count, device)
            pipeline_kwargs = {
                "broad_phase": "explicit",
                "rigid_contact_max": 256 * world_count,
            }
        else:
            raise ValueError(f"Unsupported complex-contact benchmark case: {case}")

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            deterministic=True,
            verify_buffers=False,
            **pipeline_kwargs,
        )
        self.contacts = self.collision_pipeline.contacts()
        if case == "mesh_convex" and not self.collision_pipeline.narrow_phase.convex_support_acceleration:
            raise RuntimeError("mesh-convex benchmark did not enable convex support acceleration")

        for _ in range(3):
            self.collision_pipeline.collide(self.state, self.contacts)
        wp.synchronize_device()

        if case == "mesh_convex":
            pair_count = int(self.collision_pipeline.narrow_phase.shape_pairs_mesh_count.numpy()[0])
        else:
            pair_count = int(self.collision_pipeline.narrow_phase.shape_pairs_mesh_mesh_count.numpy()[0])
        if pair_count == 0 or int(self.contacts.rigid_contact_count.numpy()[0]) == 0:
            raise RuntimeError(f"complex-contact benchmark did not produce {case} contacts")

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
        "BroadPhaseCollision": BroadPhaseCollision,
        "ComplexContactCollision": ComplexContactCollision,
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
