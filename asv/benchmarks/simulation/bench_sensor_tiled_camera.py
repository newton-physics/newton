# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.enable_backward = False
wp.config.quiet = True

import math

import newton
from newton.sensors import SensorTiledCamera

NICE_NAMES = {}


def nice_name(value):
    def decorator(func):
        func._nice_name = value
        return func

    return decorator


def nice_name_collector():
    def decorator(instance):
        for name, attr in instance.__dict__.items():
            if nice_name := getattr(attr, "_nice_name", None):
                NICE_NAMES[name] = nice_name
        return instance

    return decorator


@nice_name_collector()
class SensorTiledCameraBenchmark:
    param_names = ["resolution", "world_count", "iterations"]
    params = ([64], [4096], [50])

    def setup(self, resolution: int, world_count: int, iterations: int):
        self.device = wp.get_preferred_device()

        franka = newton.ModelBuilder()
        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )

        scene = newton.ModelBuilder()
        scene.replicate(franka, world_count)
        scene.add_ground_plane()

        self.model = scene.finalize()
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.camera_transforms = wp.array(
            [
                [
                    wp.transformf(
                        wp.vec3f(2.4, 0.0, 0.8),
                        wp.quatf(0.4187639653682709, 0.4224344491958618, 0.5708873867988586, 0.5659270882606506),
                    )
                ]
                * world_count
            ],
            dtype=wp.transformf,
        )

        self.tiled_camera_sensor = SensorTiledCamera(model=self.model)
        self.tiled_camera_sensor.utils.create_default_light(enable_shadows=False)
        self.tiled_camera_sensor.utils.assign_checkerboard_material_to_all_shapes()

        self.camera_rays = self.tiled_camera_sensor.utils.compute_pinhole_camera_rays(
            resolution, resolution, math.radians(45.0)
        )
        self.color_image = self.tiled_camera_sensor.utils.create_color_image_output(resolution, resolution)
        self.depth_image = self.tiled_camera_sensor.utils.create_depth_image_output(resolution, resolution)

        newton.geometry.build_bvh_shape(self.model, self.state)
        newton.geometry.build_bvh_particle(self.model, self.state)
        self.tiled_camera_sensor.sync_transforms(self.state)

        # Warmup Kernels
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.TILED
        self.tiled_camera_sensor.render_config.tile_width = 8
        self.tiled_camera_sensor.render_config.tile_height = 8
        for out_color, out_depth in [(True, True), (True, False), (False, True)]:
            for _ in range(iterations):
                self.tiled_camera_sensor.update(
                    self.state,
                    self.camera_transforms,
                    self.camera_rays,
                    color_image=self.color_image if out_color else None,
                    depth_image=self.depth_image if out_depth else None,
                )

        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
        for out_color, out_depth in [(True, True), (True, False), (False, True)]:
            for _ in range(iterations):
                self.tiled_camera_sensor.update(
                    self.state,
                    self.camera_transforms,
                    self.camera_rays,
                    color_image=self.color_image if out_color else None,
                    depth_image=self.depth_image if out_depth else None,
                )

    @nice_name("Rendering (Pixel)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_pixel_priority_color_depth(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                color_image=self.color_image,
                depth_image=self.depth_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Pixel) (Color Only)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_pixel_priority_color_only(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                color_image=self.color_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Pixel) (Depth Only)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_pixel_priority_depth_only(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.PIXEL_PRIORITY
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                depth_image=self.depth_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Tiled)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_tiled_color_depth(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.TILED
        self.tiled_camera_sensor.render_config.tile_width = 8
        self.tiled_camera_sensor.render_config.tile_height = 8
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                color_image=self.color_image,
                depth_image=self.depth_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Tiled) (Color Only)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_tiled_color_only(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.TILED
        self.tiled_camera_sensor.render_config.tile_width = 8
        self.tiled_camera_sensor.render_config.tile_height = 8
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                color_image=self.color_image,
            )
        wp.synchronize()

    @nice_name("Rendering (Tiled) (Depth Only)")
    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_tiled_depth_only(self, resolution: int, world_count: int, iterations: int):
        self.tiled_camera_sensor.render_config.render_order = SensorTiledCamera.RenderOrder.TILED
        self.tiled_camera_sensor.render_config.tile_width = 8
        self.tiled_camera_sensor.render_config.tile_height = 8
        for _ in range(iterations):
            self.tiled_camera_sensor.update(
                self.state,
                self.camera_transforms,
                self.camera_rays,
                depth_image=self.depth_image,
            )
        wp.synchronize()


def print_fps(name: str, duration: float, resolution: int, world_count: int, iterations: int):
    camera_count = 1

    title = f"{name}"
    if iterations > 1:
        title += " average"

    average = f"{duration * 1000.0 / iterations:.2f} ms"

    fps = f"({(1.0 / (duration / iterations) * (world_count * camera_count)):,.2f} fps)"
    print(f"{title} {'.' * (50 - len(title) - len(average))} {average} {fps if iterations > 1 else ''}")


def print_fps_results(results: dict[tuple[str, tuple[int, int, int]], float]):
    print("")
    print("=== Benchmark Results (FPS) ===")
    for (method_name, params), avg in results.items():
        print_fps(NICE_NAMES.get(method_name, method_name), avg, *params)
    print("")


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "SensorTiledCameraBenchmark": SensorTiledCameraBenchmark,
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
        result = run_benchmark(benchmark)
        print_fps_results(result)
