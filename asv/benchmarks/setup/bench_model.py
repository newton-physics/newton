# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import inspect
import os
import sys
import time
from functools import partial
from statistics import fmean

# Force headless mode for CI environments before any pyglet imports
os.environ["PYGLET_HEADLESS"] = "1"

import warp as wp

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "simulation"))

from bench_sensor_tiled_camera import SCENES as TILED_CAMERA_SCENES
from benchmark_metrics import collect_startup_metrics
from benchmark_mujoco import Example, _startup_phase

import newton
from newton.sensors import SensorTiledCamera
from newton.solvers import SolverXPBD
from newton.viewer import ViewerGL


class _KpiInitialize:
    """Cache warm startup times of MuJoCo KPI workloads through their first completed simulation frame.

    Besides the model, replication, finalize, and solver phases, the startup time
    covers state setup, graph capture, and the first completed frame, which instantiates the graph.
    """

    params = (["humanoid", "g1", "cartpole", "ant"], [8192])
    param_names = ["robot", "world_count"]
    samples = 3
    timeout = 3600

    @staticmethod
    def _create_workload(robot, world_count, startup_phase_times):
        workload = Example(
            robot=robot,
            world_count=world_count,
            randomize=False,
            headless=True,
            actuation="random",
            startup_phase_times=startup_phase_times,
        )
        if workload.graph is None:
            raise RuntimeError("KPI benchmark requires CUDA graph capture (is the CUDA mempool allocator enabled?)")
        return workload

    def setup_cache(self):
        if wp.get_cuda_device_count() == 0:
            return None

        # ASV runs an inherited setup_cache once and shares its result, so collect
        # the base parameters rather than those of the class that triggers it.
        robots, world_counts = _KpiInitialize.params
        metrics = {}
        for robot in robots:
            for world_count in world_counts:
                # Warm the measured configuration: multi-world workloads can use different kernels.
                collect_startup_metrics(partial(self._create_workload, robot, world_count), samples=1)
                metrics[robot, world_count] = collect_startup_metrics(
                    partial(self._create_workload, robot, world_count), self.samples
                )
        return metrics

    setup_cache.timeout = 3600


class KpiInitializeModel(_KpiInitialize):
    params = (["humanoid", "g1", "cartpole"], [8192])

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_initialize_model(self, metrics, robot, world_count):
        return metrics[robot, world_count].model_time

    track_initialize_model.unit = "s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_mean_replication_time(self, metrics, robot, world_count):
        return metrics[robot, world_count].replication_time

    track_mean_replication_time.unit = "s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_mean_finalize_time(self, metrics, robot, world_count):
        return metrics[robot, world_count].finalize_time

    track_mean_finalize_time.unit = "s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_mean_startup_time(self, metrics, robot, world_count):
        return metrics[robot, world_count].total_time

    track_mean_startup_time.unit = "s"


class KpiInitializeSolver(_KpiInitialize):
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_initialize_solver(self, metrics, robot, world_count):
        return metrics[robot, world_count].solver_time

    track_initialize_solver.unit = "s"


class KpiInitializeModelLargeWorld:
    """Measure warm initialization of a large single-world model, SAP pipeline, and XPBD solver.

    Model time includes finalization, and solver time includes collision pipeline
    construction. Total startup stops after initialization, without a simulation
    frame. This workload has no replication phase.
    """

    samples = 3
    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1
    timeout = 3600

    def setup_cache(self):
        # Warm CPU kernels for the separate peak-memory measurement.
        with wp.ScopedDevice("cpu"):
            self._initialize(4)
        if wp.get_cuda_device_count() == 0:
            return None

        # Warm the measured configuration before collecting phase averages.
        workload = self._initialize(10000)
        del workload
        gc.collect()
        wp.synchronize_device()

        sample_times = []
        for _ in range(self.samples):
            phase_times = {}
            start_time = time.perf_counter()
            workload = self._initialize(10000, phase_times)
            phase_times["total"] = time.perf_counter() - start_time
            del workload
            gc.collect()
            wp.synchronize_device()
            sample_times.append(phase_times)

        return {name: fmean(times[name] for times in sample_times) for name in ("model", "finalize", "solver", "total")}

    setup_cache.timeout = 3600

    @staticmethod
    def _initialize(shape_count, startup_phase_times=None):
        with _startup_phase(startup_phase_times, "model"):
            builder = newton.ModelBuilder()
            mesh = newton.Mesh.create_box(0.1)
            for i in range(shape_count):
                body = builder.add_body(xform=wp.transform(wp.vec3(0.5 * (i % 200), 0.5 * (i // 200), 1.0)))
                if i % 4 == 0:
                    builder.add_shape_mesh(body, mesh=mesh)
                elif i % 4 == 1:
                    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                elif i % 4 == 2:
                    builder.add_shape_sphere(body, radius=0.1)
                else:
                    builder.add_shape_capsule(body, radius=0.1, half_height=0.1)
            with _startup_phase(startup_phase_times, "finalize"):
                model = builder.finalize()
        with _startup_phase(startup_phase_times, "solver"):
            # Bound runtime collision buffers independently of the model's pair count.
            pipeline = newton.CollisionPipeline(
                model, broad_phase="sap", shape_pairs_max=256, rigid_contact_max=128, max_triangle_pairs=128
            )
            solver = SolverXPBD(model)
        return model, pipeline, solver

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_initialize_model(self, metrics):
        return metrics["model"]

    track_initialize_model.unit = "s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_mean_finalize_time(self, metrics):
        return metrics["finalize"]

    track_mean_finalize_time.unit = "s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_initialize_solver(self, metrics):
        return metrics["solver"]

    track_initialize_solver.unit = "s"

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def track_mean_startup_time(self, metrics):
        return metrics["total"]

    track_mean_startup_time.unit = "s"

    def peakmem_initialize_model_cpu(self, metrics=None):
        gc.collect()

        with wp.ScopedDevice("cpu"):
            model, pipeline, solver = self._initialize(10000)

        del model, pipeline, solver


class KpiInitializeViewerGL:
    params = (["g1"], [8192])
    param_names = ["robot", "world_count"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup(self, robot, world_count):
        wp.init()
        builder = Example.create_model_builder(robot, world_count, randomize=True, seed=123)

        # finalize model
        self._model = builder.finalize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_renderer(self, robot, world_count):
        # Setting up the renderer
        self.renderer = ViewerGL(headless=True)
        self.renderer.set_model(self._model)

        wp.synchronize_device()
        self.renderer.close()

    def teardown(self, robot, world_count):
        del self._model


class _InitializeModelTiledCamera:
    """Build models and tiled cameras from scene presets with collision handling enabled."""

    param_names = ["scene", "world_count"]
    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup(self, scene, world_count):
        self.world = TILED_CAMERA_SCENES[scene].build()
        warmup = self._replicate(1)
        _model, _sensor = self._initialize_model_and_sensor(warmup)
        self.replicated_builder = self._replicate(world_count)
        wp.synchronize_device()

    def _replicate(self, world_count):
        builder = newton.ModelBuilder()
        builder.replicate(self.world, world_count)
        builder.add_ground_plane()
        return builder

    def _initialize_model_and_sensor(self, builder):
        model = builder.finalize()
        sensor = SensorTiledCamera(model)
        return model, sensor

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_model(self, scene, world_count):
        _model, _sensor = self._initialize_model_and_sensor(self._replicate(world_count))
        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_finalize_model(self, scene, world_count):
        _model, _sensor = self._initialize_model_and_sensor(self.replicated_builder)
        wp.synchronize_device()

    def teardown(self, scene, world_count):
        del self.replicated_builder
        del self.world


class KpiInitializeModelTiledCamera(_InitializeModelTiledCamera):
    params = (["franka_cabinet", "quadruped"], [4096])
    timeout = 3600


class FastInitializeModel:
    params = (["humanoid", "g1", "cartpole"], [256])
    param_names = ["robot", "world_count"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup_cache(self):
        # Finalize small models first so the asset download and one-time kernel
        # compilation stay out of the timed builds. Warp compiles per device
        # target, so warm both the default device (for time_initialize_model)
        # and CPU (for peakmem_initialize_model_cpu). Fresh builder per
        # finalize: finalize() mutates builder state in place.
        for robot in self.params[0]:
            for device in (None, "cpu"):
                builder = Example.create_model_builder(robot, 1, randomize=False, seed=123)
                model = builder.finalize(device=device)
                del model
                del builder

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_model(self, robot, world_count):
        builder = Example.create_model_builder(robot, world_count, randomize=True, seed=123)

        # finalize model
        _model = builder.finalize()
        wp.synchronize_device()

    def peakmem_initialize_model_cpu(self, robot, world_count):
        gc.collect()

        with wp.ScopedDevice("cpu"):
            builder = Example.create_model_builder(robot, world_count, randomize=True, seed=123)

            # finalize model
            model = builder.finalize()

        del model


class FastInitializeSolver:
    params = (["humanoid", "g1", "cartpole"], [256])
    param_names = ["robot", "world_count"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup(self, robot, world_count):
        wp.init()
        builder = Example.create_model_builder(robot, world_count, randomize=True, seed=123)

        # finalize model
        self._model = builder.finalize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_solver(self, robot, world_count):
        self._solver = Example.create_solver(self._model, robot, use_mujoco_cpu=False)
        wp.synchronize_device()

    def teardown(self, robot, world_count):
        del self._solver
        del self._model


class FastInitializeViewerGL:
    params = (["g1"], [256])
    param_names = ["robot", "world_count"]

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup(self, robot, world_count):
        wp.init()
        builder = Example.create_model_builder(robot, world_count, randomize=True, seed=123)

        # finalize model
        self._model = builder.finalize()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_initialize_renderer(self, robot, world_count):
        # Setting up the renderer
        self.renderer = ViewerGL(headless=True)
        self.renderer.set_model(self._model)

        wp.synchronize_device()
        self.renderer.close()

    def teardown(self, robot, world_count):
        del self._model


class FastInitializeModelTiledCamera(_InitializeModelTiledCamera):
    params = (["franka_cabinet", "quadruped"], [256])


class FastInitializeModelStackedBoxes:
    """Stamp columns of stacked boxes into one shared world and finalize the model."""

    params = (["add_builder", "replicate"],)
    param_names = ["method"]
    grid = (16, 8, 16)

    rounds = 1
    repeat = 3
    number = 1
    min_run_count = 1

    def setup_cache(self):
        # Compile the CPU finalize kernels once, outside the timed builds.
        self._build("add_builder", 2, 2, 2).finalize(device="cpu")

    def setup(self, method):
        if (
            method == "replicate"
            and "separate_worlds" not in inspect.signature(newton.ModelBuilder.replicate).parameters
        ):
            # Skips commits that predate replicate(separate_worlds=...).
            raise SkipNotImplemented

    @staticmethod
    def _build(method, nx, ny, nz):
        column = newton.ModelBuilder()
        for j in range(ny):
            offset = float((j + 1) & 1)
            body = column.add_body(xform=wp.transform((offset, offset, 1.0 + 2.0 * j), wp.quat_identity()))
            column.add_shape_box(body, hx=1.0, hy=1.0, hz=1.0)

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        xforms = [wp.transform((2.2 * i, 2.2 * k, 0.0), wp.quat_identity()) for i in range(nx) for k in range(nz)]
        if method == "replicate":
            builder.replicate(column, len(xforms), separate_worlds=False, xforms=xforms)
        else:
            for xform in xforms:
                builder.add_builder(column, xform=xform)
        return builder

    def time_initialize_model(self, method):
        self._build(method, *self.grid).finalize(device="cpu")


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "KpiInitializeModel": KpiInitializeModel,
        "FastInitializeModel": FastInitializeModel,
        "KpiInitializeSolver": KpiInitializeSolver,
        "KpiInitializeModelLargeWorld": KpiInitializeModelLargeWorld,
        "FastInitializeSolver": FastInitializeSolver,
        "KpiInitializeViewerGL": KpiInitializeViewerGL,
        "FastInitializeViewerGL": FastInitializeViewerGL,
        "KpiInitializeModelTiledCamera": KpiInitializeModelTiledCamera,
        "FastInitializeModelTiledCamera": FastInitializeModelTiledCamera,
        "FastInitializeModelStackedBoxes": FastInitializeModelStackedBoxes,
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
