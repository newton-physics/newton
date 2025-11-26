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
import newton
from newton.sensors import TiledCameraSensor


class TiledCameraSensorBenchmark:
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

        # camera_position = wp.vec3f(2.0, 0.0, 0.6)
        # camera_orientation = wp.mat33f(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        camera_position = wp.vec3f(2.4, 0.0, 0.8)
        camera_orientation = wp.mat33f(
            -0.008726535,
            -0.29236057,
            0.95626837,
            0.9999619,
            -0.002551392,
            0.008345228,
            1.3010426e-18,
            0.9563047,
            0.2923717,
        )

        transform = wp.transformf(camera_position, wp.quat_from_matrix(camera_orientation))
        camera_transforms = wp.array([transform], dtype=wp.transformf)

        self.tiled_camera_sensor = TiledCameraSensor(model=self.model, num_cameras=1, width=resolution, height=resolution)
        self.tiled_camera_sensor.assign_debug_colors_per_shape()
        self.tiled_camera_sensor.assign_default_checkerboard_material()
        self.tiled_camera_sensor.create_default_light(False)
        self.tiled_camera_sensor.compute_camera_rays(wp.array([math.radians(45.0)], dtype=wp.float32))
        self.color_image = self.tiled_camera_sensor.create_color_image_output()
        self.depth_image = self.tiled_camera_sensor.create_depth_image_output()

        self.tiled_camera_sensor.update_cameras(camera_transforms)
        self.tiled_camera_sensor.update_from_state(self.state)

        with wp.ScopedTimer("Refit BVH", synchronize=True, print=False) as timer:
            self.tiled_camera_sensor.render_context.refit_bvh()
        self.timings["refit"] = timer.elapsed

        for _ in range(iterations):
            self.tiled_camera_sensor.render(self.color_image, self.depth_image, refit_bvh=False, clear_images=False)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_pixel(self, resolution: int, num_worlds: int, iterations: int):
        self.tiled_camera_sensor.render_context.tile_rendering = False
        with wp.ScopedTimer("Rendering", synchronize=True, print=True) as timer:
            for _ in range(iterations):
                self.tiled_camera_sensor.render(self.color_image, self.depth_image, refit_bvh=False, clear_images=False)
        self.timings["render_pixel"] = timer.elapsed

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_rendering_tiled(self, resolution: int, num_worlds: int, iterations: int):
        self.tiled_camera_sensor.render_context.tile_rendering = True
        self.tiled_camera_sensor.render_context.tile_size = 8
        with wp.ScopedTimer("Tiled Rendering", synchronize=True, print=False) as timer:
            for _ in range(iterations):
                self.tiled_camera_sensor.render(self.color_image, self.depth_image, refit_bvh=False, clear_images=False)
        self.timings["render_tiledl"] = timer.elapsed
    
    def teardown(self, resolution: int, num_worlds: int, iterations: int):
        print("")
        print("=== Benchmark Results (FPS) ===")
        self.__print_timer("Refit BVH", self.timings["refit"], 1, self.tiled_camera_sensor)
        self.__print_timer("Rendering (Pixel)", self.timings["render_pixel"], iterations, self.tiled_camera_sensor)
        self.__print_timer("Rendering (Tiled)", self.timings["render_tiledl"], iterations, self.tiled_camera_sensor)
        self.tiled_camera_sensor.save_color_image(self.color_image, "benchmark_color.png")
        self.tiled_camera_sensor.save_depth_image(self.depth_image, "benchmark_depth.png")

    def __print_timer(self, name: str, elapsed: float, iterations: int, sensor: TiledCameraSensor):
        title = f"{name}"
        if iterations > 1:
            title += " average"
        average = f"{elapsed / iterations:.2f} ms"
        fps = f"({(1000.0 / (elapsed / iterations) * (sensor.render_context.num_worlds * sensor.render_context.num_cameras)):,.2f} fps)"

        print(f"{title} {'.' * (40 - len(title) - len(average))} {average} {fps if iterations > 1 else ''}")


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "TiledCameraSensor": TiledCameraSensorBenchmark,
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
