# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warp as wp
from asv_runner.benchmarks.mark import skip_benchmark_if

wp.config.quiet = True

import math
import os

import newton
from newton.sensors import SensorTiledCamera


class SensorTiledCameraBenchmark:
    param_names = ["resolution", "num_worlds", "iterations"]
    params = ([64], [4096], [50])

    def setup(self, resolution: int, num_worlds: int, iterations: int):
        self.timings = {}

        franka = newton.ModelBuilder()
        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )

        scene = newton.ModelBuilder()
        scene.replicate(franka, num_worlds)
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
                * num_worlds
            ],
            dtype=wp.transformf,
        )

        self.tiled_camera_sensor = SensorTiledCamera(
            model=self.model,
            options=SensorTiledCamera.Options(default_light=True, colors_per_shape=True, checkerboard_texture=True),
        )
        self.camera_rays = self.tiled_camera_sensor.compute_pinhole_camera_rays(
            resolution, resolution, math.radians(45.0)
        )
        self.color_image = self.tiled_camera_sensor.create_color_image_output(resolution, resolution)
        self.depth_image = self.tiled_camera_sensor.create_depth_image_output(resolution, resolution)

        self.tiled_camera_sensor.update_from_state(self.state)

        with wp.ScopedTimer("Refit BVH", synchronize=True, print=False) as timer:
            self.tiled_camera_sensor.render_context.refit_bvh()
        self.timings["refit"] = timer.elapsed

        for _ in range(iterations):
            self.tiled_camera_sensor.render(
                None,
                self.camera_transforms,
                self.camera_rays,
                self.color_image,
                self.depth_image,
                refit_bvh=False,
            )

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_pixel(self, resolution: int, num_worlds: int, iterations: int):
        self.tiled_camera_sensor.render_context.options.tile_rendering = False
        with wp.ScopedTimer("Rendering", synchronize=True, print=True) as timer:
            for _ in range(iterations):
                self.tiled_camera_sensor.render(
                    None,
                    self.camera_transforms,
                    self.camera_rays,
                    self.color_image,
                    self.depth_image,
                    refit_bvh=False,
                    clear_data=None,
                )
        self.timings["render_pixel"] = timer.elapsed

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_tiled(self, resolution: int, num_worlds: int, iterations: int):
        self.tiled_camera_sensor.render_context.options.tile_rendering = True
        self.tiled_camera_sensor.render_context.options.tile_size = 8
        with wp.ScopedTimer("Tiled Rendering", synchronize=True, print=False) as timer:
            for _ in range(iterations):
                self.tiled_camera_sensor.render(
                    None,
                    self.camera_transforms,
                    self.camera_rays,
                    self.color_image,
                    self.depth_image,
                    refit_bvh=False,
                    clear_data=None,
                )
        self.timings["render_tiled"] = timer.elapsed

    def teardown(self, resolution: int, num_worlds: int, iterations: int):
        print("")
        print("=== Benchmark Results (FPS) ===")
        if "refit" in self.timings:
            self.__print_timer("Refit BVH", self.timings["refit"], 1, self.tiled_camera_sensor)
        if "render_pixel" in self.timings:
            self.__print_timer("Rendering (Pixel)", self.timings["render_pixel"], iterations, self.tiled_camera_sensor)
        if "render_tiled" in self.timings:
            self.__print_timer("Rendering (Tiled)", self.timings["render_tiled"], iterations, self.tiled_camera_sensor)

        if os.environ.get("SAVE_IMAGES", "0") != "0":
            from PIL import Image  # noqa: PLC0415

            color_image = self.tiled_camera_sensor.flatten_color_image_to_rgba(resolution, resolution, self.color_image)
            depth_image = self.tiled_camera_sensor.flatten_depth_image_to_rgba(resolution, resolution, self.depth_image)
            Image.fromarray(color_image.numpy()).save("benchmark_color.png")
            Image.fromarray(depth_image.numpy()).save("benchmark_depth.png")

    def __print_timer(self, name: str, elapsed: float, iterations: int, sensor: SensorTiledCamera):
        num_cameras = 1
        title = f"{name}"
        if iterations > 1:
            title += " average"
        average = f"{elapsed / iterations:.2f} ms"
        fps = f"({(1000.0 / (elapsed / iterations) * (sensor.render_context.num_worlds * num_cameras)):,.2f} fps)"

        print(f"{title} {'.' * (40 - len(title) - len(average))} {average} {fps if iterations > 1 else ''}")


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "SensorTiledCamera": SensorTiledCameraBenchmark,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b", "--bench", default=None, action="append", choices=benchmark_list.keys(), help="Run a single benchmark."
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
