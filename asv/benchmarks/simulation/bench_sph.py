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


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(WCSPHDamBreak)
