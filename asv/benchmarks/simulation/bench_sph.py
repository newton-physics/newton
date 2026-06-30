# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton
from newton.solvers import SolverWCSPH, sph


class WCSPHDamBreak:
    """WCSPH stepping throughput for approximately 10k and 100k particles."""

    params = ([22, 46],)
    param_names = ["resolution"]
    repeat = 5
    number = 1
    step_count = 10

    def setup(self, resolution):
        if wp.get_cuda_device_count() == 0:
            raise SkipNotImplemented

        with wp.ScopedDevice("cuda:0"):
            spacing = 0.02
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=-9.81)
            sph.add_sph_particle_grid(
                builder,
                pos=wp.vec3(-0.5 * resolution * spacing, spacing, -0.5 * resolution * spacing),
                dim_x=resolution,
                dim_y=resolution,
                dim_z=resolution,
                cell_x=spacing,
                cell_y=spacing,
                cell_z=spacing,
                material=sph.SPHMaterial(
                    rest_density=1000.0,
                    sound_speed=20.0,
                    viscosity=0.001,
                    smoothing_length=2.0 * spacing,
                ),
                radius_mean=0.5 * spacing,
            )
            builder.add_ground_plane()
            self.model = builder.finalize(device="cuda:0")
            self.solver = SolverWCSPH(self.model, SolverWCSPH.Config(xsph=0.03))
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.dt = 2.0e-4

            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self, resolution):
        del resolution
        with wp.ScopedDevice("cuda:0"):
            for _ in range(self.step_count):
                self.solver.step(self.state_0, self.state_1, None, None, self.dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            wp.synchronize_device()


class WCSPHColliderBroadphase:
    """WCSPH collision throughput as the selected shape count grows."""

    params = ([1, 32, 128],)
    param_names = ["shape_count"]
    repeat = 5
    number = 1
    step_count = 10

    def setup(self, shape_count):
        if wp.get_cuda_device_count() == 0:
            raise SkipNotImplemented

        with wp.ScopedDevice("cuda:0"):
            spacing = 0.02
            resolution = 22
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
            sph.add_sph_particle_grid(
                builder,
                pos=wp.vec3(-0.5 * resolution * spacing, 0.2, -0.5 * resolution * spacing),
                dim_x=resolution,
                dim_y=resolution,
                dim_z=resolution,
                cell_x=spacing,
                cell_y=spacing,
                cell_z=spacing,
                material=sph.SPHMaterial(
                    sound_speed=20.0,
                    viscosity=0.001,
                    smoothing_length=2.0 * spacing,
                ),
                radius_mean=0.5 * spacing,
            )
            shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, is_visible=False)
            for shape in range(shape_count):
                x = 1.0 + 0.08 * float(shape % 16)
                y = 0.05 + 0.08 * float((shape // 16) % 8)
                z = 0.08 * float(shape // 128)
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()),
                    hx=0.025,
                    hy=0.025,
                    hz=0.025,
                    cfg=shape_cfg,
                )

            self.model = builder.finalize(device="cuda:0")
            self.solver = SolverWCSPH(self.model, SolverWCSPH.Config(xsph=0.03))
            self.state_0 = self.model.state()
            self.state_1 = self.model.state()
            self.dt = 2.0e-4
            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self, shape_count):
        del shape_count
        with wp.ScopedDevice("cuda:0"):
            for _ in range(self.step_count):
                self.solver.step(self.state_0, self.state_1, None, None, self.dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
            wp.synchronize_device()


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(WCSPHDamBreak)
