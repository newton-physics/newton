# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import inspect

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

COARSE_CLOTH_RIGID_CASES = (
    ("box_coarse_cloth_16", "box", 4, 16, 1, False),
    ("box_coarse_cloth_64", "box", 4, 64, 1, False),
    ("capsule_coarse_cloth_16", "capsule", 4, 16, 1, False),
    ("mesh_coarse_cloth_16", "mesh", 4, 16, 1, False),
)

COARSE_CLOTH_VBD_CASES = (
    ("box_coarse_cloth_16", "box", 8, 16, 1, False),
    ("box_coarse_cloth_64", "box", 8, 64, 1, False),
    ("box_coarse_cloth_256", "box", 8, 256, 1, False),
    ("capsule_coarse_cloth_16", "capsule", 8, 16, 1, False),
    ("mesh_coarse_cloth_16", "mesh", 8, 16, 1, False),
)

DEFORMABLE_RIGID_CASES = (
    ("sphere_dense", "sphere", 64, 128, 1, False),
    ("sphere_sparse", "sphere", 64, 128, 1, True),
    ("box_single_world", "box", 724, 1, 1, False),
    ("box_multi_shape", "box", 32, 128, 16, False),
    ("capsule", "capsule", 32, 64, 16, False),
    ("cylinder", "cylinder", 32, 64, 16, False),
    ("cone", "cone", 32, 64, 16, False),
    ("ellipsoid", "ellipsoid", 32, 64, 16, False),
    ("mesh_sdf", "mesh", 32, 64, 16, False),
    ("mesh_high_poly_fallback", "mesh_high_poly", 64, 32, 1, False),
    ("infinite_plane", "plane", 64, 128, 1, False),
    ("finite_plane", "finite_plane", 64, 128, 1, False),
    ("heightfield", "heightfield", 64, 128, 1, False),
    ("mixed", "mixed", 32, 64, 20, False),
    ("mixed_sparse", "mixed", 32, 64, 20, True),
    ("mixed_gpu", "mixed", 128, 64, 10, False),
)


# Cases with rigid meshes, repeated with the "sdf" mesh back-end.
COARSE_CLOTH_RIGID_SDF_MESH_CASES = tuple(case for case in COARSE_CLOTH_RIGID_CASES if case[1] == "mesh")
COARSE_CLOTH_VBD_SDF_MESH_CASES = tuple(case for case in COARSE_CLOTH_VBD_CASES if case[1] == "mesh")


WRAPPED_CLOTH_PARTICLES = (81, 1089, 16641)
WRAPPED_CLOTH_RIGID_TRIANGLES = (64, 16384)
# Offset [m] of the cloth from the mesh surface; the contact gap is 0.05 m.
WRAPPED_CLOTH_OFFSETS = {"none": 1.0, "all": 0.02}
WRAPPED_CLOTH_MESH_BACKENDS = ("bvh", "sdf")
# Soft-contact capacity of the one scene whose rows exceed the default capacity.
WRAPPED_CLOTH_SOFT_CONTACT_MAX = {(16641, 16384, "all", "bvh"): 1250000}


DEFORMABLE_RIGID_SCALE_CASES = (
    ("sphere_10m_rl", "sphere", 64, 1250, 1, False),
    ("mesh_10m_rl", "mesh", 64, 1250, 1, False),
)


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


def _make_deformable_rigid_world(kind, resolution, shape_count, sparse):
    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    extent = 2.0
    cloth_height = 4.0 if sparse else (0.02 if kind in ("plane", "finite_plane", "heightfield") else 0.45)
    world.add_cloth_grid(
        pos=wp.vec3(-0.5 * extent, -0.5 * extent, cloth_height),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=resolution,
        dim_y=resolution,
        cell_x=extent / resolution,
        cell_y=extent / resolution,
        mass=0.1,
        particle_radius=0.0,
    )

    side = int(np.ceil(np.sqrt(shape_count)))
    if kind == "mesh_high_poly":
        mesh = newton.Mesh.create_sphere(0.18, num_latitudes=64, num_longitudes=128)
    elif kind in ("mesh", "mixed"):
        mesh = newton.Mesh.create_box(0.18, 0.18, 0.18)
    else:
        mesh = None
    heightfield = (
        newton.Heightfield(data=np.zeros((17, 17), dtype=np.float32), nrow=17, ncol=17, hx=0.5, hy=0.5)
        if kind in ("heightfield", "mixed")
        else None
    )
    shape_kinds = (
        "sphere",
        "box",
        "capsule",
        "cylinder",
        "cone",
        "ellipsoid",
        "mesh",
        "plane",
        "finite_plane",
        "heightfield",
    )
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    for shape_index in range(shape_count):
        row, column = divmod(shape_index, side)
        x = (column + 0.5) * extent / side - 0.5 * extent
        y = (row + 0.5) * extent / side - 0.5 * extent
        shape_kind = shape_kinds[shape_index % len(shape_kinds)] if kind == "mixed" else kind
        if shape_kind == "mesh_high_poly":
            shape_kind = "mesh"
        z = 0.42 if shape_kind in ("plane", "finite_plane", "heightfield") else 0.4
        xform = wp.transform(wp.vec3(x, y, z), wp.quat_identity())
        if shape_kind == "sphere":
            world.add_shape_sphere(body=-1, xform=xform, radius=0.5 / side, cfg=cfg)
        elif shape_kind == "box":
            world.add_shape_box(body=-1, xform=xform, hx=0.5 / side, hy=0.5 / side, hz=0.5, cfg=cfg)
        elif shape_kind == "capsule":
            world.add_shape_capsule(body=-1, xform=xform, radius=0.3 / side, half_height=0.3, cfg=cfg)
        elif shape_kind == "cylinder":
            world.add_shape_cylinder(body=-1, xform=xform, radius=0.4 / side, half_height=0.5, cfg=cfg)
        elif shape_kind == "cone":
            world.add_shape_cone(body=-1, xform=xform, radius=0.4 / side, half_height=0.5, cfg=cfg)
        elif shape_kind == "ellipsoid":
            world.add_shape_ellipsoid(body=-1, xform=xform, rx=0.5 / side, ry=0.35 / side, rz=0.5, cfg=cfg)
        elif shape_kind == "mesh":
            shape = world.add_shape_mesh(body=-1, xform=xform, mesh=mesh, scale=(1.0 / side,) * 3, cfg=cfg)
            world.shape_force_sdf[shape] = True
        elif shape_kind == "plane":
            world.add_shape_plane(body=-1, xform=xform, width=0.0, length=0.0, cfg=cfg)
        elif shape_kind == "finite_plane":
            world.add_shape_plane(body=-1, xform=xform, width=0.8 / side, length=0.8 / side, cfg=cfg)
        elif shape_kind == "heightfield":
            world.add_shape_heightfield(xform=xform, heightfield=heightfield, cfg=cfg)

    return world


@wp.kernel
def _mesh_distance_along_rays(mesh: wp.uint64, directions: wp.array[wp.vec3], distances: wp.array[float]):
    """Distance [m] from the origin to the mesh surface along each unit direction, or -1 on a miss."""
    tid = wp.tid()
    direction = directions[tid]
    query = wp.mesh_query_ray(mesh, 4.0 * direction, -direction, 8.0)
    distances[tid] = 4.0 - query.t if query.result else -1.0


def _make_wrapped_cloth_world(cloth_particles, rigid_triangles, offset, mesh_sdf, device):
    """Wrap a square cloth over a cap of a unit sphere mesh.

    Every particle lies on a ray from the sphere centre, at *offset* [m] from
    the point where the ray meets the faceted surface, so the cloth keeps the
    same distance to the mesh at every mesh resolution.
    """
    resolution = round(np.sqrt(cloth_particles)) - 1
    latitudes = round(np.sqrt(rigid_triangles / 4))
    mesh = newton.Mesh.create_sphere(1.0, num_latitudes=latitudes, num_longitudes=2 * latitudes)
    indices = np.asarray(mesh.indices).reshape(-1)
    cap = 0.8  # tangent of the cap half-angle along each cloth axis

    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    world.add_cloth_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=resolution,
        dim_y=resolution,
        cell_x=2.0 * cap / resolution,
        cell_y=2.0 * cap / resolution,
        mass=0.1,
        particle_radius=0.0,
    )
    if world.particle_count != cloth_particles or len(indices) != 3 * rigid_triangles:
        raise ValueError(f"no wrapped-cloth scene has {cloth_particles} particles and {rigid_triangles} triangles")

    # Ray directions in the particle order of the grid, whose first axis varies fastest.
    grid = np.linspace(-cap, cap, resolution + 1)
    v, u = np.meshgrid(grid, grid, indexing="ij")
    directions = np.stack((np.ones_like(u), u, v), axis=-1).reshape(-1, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    ray_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=wp.int32, device=device),
    )
    distances = wp.zeros(len(directions), dtype=float, device=device)
    wp.launch(
        _mesh_distance_along_rays,
        dim=len(directions),
        inputs=[ray_mesh.id, wp.array(directions, dtype=wp.vec3, device=device)],
        outputs=[distances],
        device=device,
    )
    distances = distances.numpy()
    if np.any(distances < 0.0):
        raise RuntimeError("a cloth ray missed the sphere mesh")
    for index, position in enumerate(directions * (distances + offset)[:, None]):
        world.particle_q[index] = wp.vec3(*position)

    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    if mesh_sdf:
        cfg.configure_sdf(force_sdf=True)
    world.add_shape_mesh(body=-1, mesh=mesh, cfg=cfg)
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
        # Keep one-time device work out of the samples.
        wp.capture_launch(self.graph)
        wp.synchronize_device()

    def _detect(self):
        self.detector.refit(self.model.particle_q)
        self.detector.vertex_triangle_collision_detection(self.radius)
        self.detector.edge_edge_collision_detection(self.radius)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_detect(self, case):
        for _ in range(self.launch_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


class DeformableRigidCollision:
    """Benchmark full-surface deformable contact against individual and mixed rigid geometry."""

    params = (DEFORMABLE_RIGID_CASES,)
    param_names = ["case"]
    rounds = 1
    repeat = 3
    number = 1
    warmup_count = 2
    launch_count = 10
    timeout = 600
    mesh_backend = "bvh"

    def _mesh_backend_args(self):
        """Return the pipeline argument selecting :attr:`mesh_backend`.

        The default back-end passes nothing, so those benchmarks also run on
        revisions that predate the argument.
        """
        if self.mesh_backend == "bvh":
            return {}
        if "rigid_soft_full_surface_mesh_backend" not in inspect.signature(newton.CollisionPipeline).parameters:
            raise SkipNotImplemented
        return {"rigid_soft_full_surface_mesh_backend": self.mesh_backend}

    def setup(self, case):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented
        mesh_backend_args = self._mesh_backend_args()

        # ASV repeats setup per sample; collision-only graph replays preserve state.
        if getattr(self, "_case", None) == case:
            return
        self._case = None

        _name, kind, resolution, world_count, shape_count, sparse = case
        builder = newton.ModelBuilder()
        builder.replicate(
            _make_deformable_rigid_world(kind, resolution, shape_count, sparse),
            world_count,
        )
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_gap=0.05,
            enable_rigid_soft_full_surface_contact=True,
            verify_buffers=False,
            **mesh_backend_args,
        )
        self.contacts = self.pipeline.contacts()

        for _ in range(self.warmup_count):
            self.pipeline.collide(self.state, self.contacts)
        self._verify_contact_capacity()
        if kind == "mixed" and not sparse:
            contact_count = int(self.contacts.soft_contact_count.numpy()[0])
            contacted_shapes = np.unique(self.contacts.soft_contact_shape.numpy()[:contact_count])
            if len(contacted_shapes) != self.model.shape_count:
                raise RuntimeError(
                    f"mixed benchmark contacts {len(contacted_shapes)} of {self.model.shape_count} rigid shapes"
                )
        with wp.ScopedCapture(device=device) as capture:
            self.pipeline.collide(self.state, self.contacts)
        self.graph = capture.graph
        self._case = case

    def _verify_contact_capacity(self):
        """Reject truncated warmup results before capturing the timed collision graph."""
        for name, counter, capacity in (
            ("candidate pairs", self.pipeline.broad_phase_pair_count, self.pipeline.shape_pairs_max),
            ("rigid contacts", self.contacts.rigid_contact_count, self.contacts.rigid_contact_max),
            ("soft contacts", self.contacts.soft_contact_count, self.contacts.soft_contact_max),
        ):
            count = int(counter.numpy()[0])
            if count > capacity:
                raise RuntimeError(f"deformable-rigid benchmark overflows {name}: {count} > {capacity}")

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_collide(self, case):
        for _ in range(self.launch_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


class DeformableRigidCollisionScale(DeformableRigidCollision):
    """Benchmark GPU-saturating deformable-rigid scales."""

    params = (DEFORMABLE_RIGID_SCALE_CASES,)
    warmup_count = 1
    launch_count = 3


class FastDeformableRigidCoarseCloth(DeformableRigidCollision):
    """Guard collision latency for small replicated deformable scenes."""

    params = (COARSE_CLOTH_RIGID_CASES,)
    rounds = 2
    repeat = pr_gate_repeat(5)


class FastDeformableRigidCoarseClothSdfMesh(FastDeformableRigidCoarseCloth):
    """Guard collision latency of the ``"sdf"`` mesh back-end for small replicated scenes."""

    params = (COARSE_CLOTH_RIGID_SDF_MESH_CASES,)
    mesh_backend = "sdf"


class FastDeformableRigidCoarseClothVBD(DeformableRigidCollision):
    """Measure small-scene full-surface contact through the VBD solve."""

    params = (COARSE_CLOTH_VBD_CASES,)
    rounds = 2
    repeat = pr_gate_repeat(5)
    launch_count = 5
    # This graph includes the solve; expose it only through time_simulate.
    time_collide = None

    def setup(self, case):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented
        mesh_backend_args = self._mesh_backend_args()

        _name, kind, resolution, world_count, shape_count, sparse = case
        builder = newton.ModelBuilder()
        builder.replicate(_make_deformable_rigid_world(kind, resolution, shape_count, sparse), world_count)
        builder.color()
        self._capture_simulate(builder.finalize(device=device), mesh_backend_args)

    def _capture_simulate(self, model, pipeline_args, restore_state=False):
        """Capture one collision pass and one VBD step of *model* as the timed graph.

        The VBD step advances the input state in place. With *restore_state*,
        the graph first restores the initial particle state, so every launch
        repeats the same work.
        """
        device = model.device
        self.model = model
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_gap=0.05,
            enable_rigid_soft_full_surface_contact=True,
            verify_buffers=False,
            **pipeline_args,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=10)
        self.control = self.model.control()

        def simulate():
            if restore_state:
                wp.copy(self.state_in.particle_q, self.model.particle_q)
                wp.copy(self.state_in.particle_qd, self.model.particle_qd)
            self.pipeline.collide(self.state_in, self.contacts)
            self.solver.step(self.state_in, self.state_out, self.control, self.contacts, 1.0 / 120.0)

        for _ in range(self.warmup_count):
            simulate()
        self._verify_contact_capacity()
        with wp.ScopedCapture(device=device) as capture:
            simulate()
        self.graph = capture.graph
        # Keep one-time device work out of the samples.
        wp.capture_launch(self.graph)
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self, case):
        for _ in range(self.launch_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


class FastDeformableRigidCoarseClothVBDSdfMesh(FastDeformableRigidCoarseClothVBD):
    """Measure small-scene contact of the ``"sdf"`` mesh back-end through the VBD solve."""

    params = (COARSE_CLOTH_VBD_SDF_MESH_CASES,)
    mesh_backend = "sdf"


class DeformableRigidMeshBackendScaling(FastDeformableRigidCoarseClothVBD):
    """Compare the mesh back-ends as the cloth, the rigid mesh, and the contact grow.

    One cloth is wrapped over one static sphere mesh, either out of contact or
    with all of it inside the contact gap. The ``"sdf"`` back-end runs on a
    mesh with a volume SDF, the ``"bvh"`` back-end on the same mesh without one.
    """

    params = (
        WRAPPED_CLOTH_PARTICLES,
        WRAPPED_CLOTH_RIGID_TRIANGLES,
        tuple(WRAPPED_CLOTH_OFFSETS),
        WRAPPED_CLOTH_MESH_BACKENDS,
    )
    param_names = ["cloth_particles", "rigid_triangles", "contact", "mesh_backend"]
    rounds = 1
    repeat = 3
    # Report the time of one collision pass and one VBD step, averaged over five.
    number = 5
    launch_count = 1

    def setup(self, cloth_particles, rigid_triangles, contact, mesh_backend):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented
        self.mesh_backend = mesh_backend
        pipeline_args = self._mesh_backend_args()
        pipeline_args["soft_contact_max"] = WRAPPED_CLOTH_SOFT_CONTACT_MAX.get(
            (cloth_particles, rigid_triangles, contact, mesh_backend)
        )

        builder = _make_wrapped_cloth_world(
            cloth_particles, rigid_triangles, WRAPPED_CLOTH_OFFSETS[contact], mesh_backend == "sdf", device
        )
        builder.color()
        self._capture_simulate(builder.finalize(device=device), pipeline_args, restore_state=True)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self, cloth_particles, rigid_triangles, contact, mesh_backend):
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
        self.example = ExampleClothManipulation(ViewerNull(num_frames=self.num_frames + 1), args)
        # Keep one-time device work out of the samples.
        self.example.step()
        self.example.render()
        wp.synchronize_device()

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
        self.example = ExampleClothTwist(ViewerNull(num_frames=self.num_frames + 1), args)
        # Keep one-time device work out of the samples.
        self.example.step()
        self.example.render()
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, None)

        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastDeformableSelfCollision": FastDeformableSelfCollision,
        "DeformableRigidCollision": DeformableRigidCollision,
        "DeformableRigidCollisionScale": DeformableRigidCollisionScale,
        "FastDeformableRigidCoarseCloth": FastDeformableRigidCoarseCloth,
        "FastDeformableRigidCoarseClothSdfMesh": FastDeformableRigidCoarseClothSdfMesh,
        "FastDeformableRigidCoarseClothVBD": FastDeformableRigidCoarseClothVBD,
        "FastDeformableRigidCoarseClothVBDSdfMesh": FastDeformableRigidCoarseClothVBDSdfMesh,
        "DeformableRigidMeshBackendScaling": DeformableRigidMeshBackendScaling,
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
