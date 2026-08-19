# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton
from newton.solvers import SolverImplicitMPM


class ImplicitMPMSingleWorld:
    """Track the fixed-grid single-world fast path independently of batching."""

    number = 1
    repeat = 5
    rounds = 2

    def setup(self):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented

        builder = newton.ModelBuilder(gravity=0.0)
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_particle_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=14,
            dim_y=14,
            dim_z=14,
            cell_x=0.025,
            cell_y=0.025,
            cell_z=0.025,
            mass=0.01,
            jitter=0.0,
            radius_mean=0.0125,
            custom_attributes={"mpm:young_modulus": 1.0e4, "mpm:poisson_ratio": 0.2},
        )
        self.model = builder.finalize(device=device)

        config = SolverImplicitMPM.Config()
        config.grid_type = "fixed"
        config.grid_padding = 3
        config.voxel_size = 0.05
        config.transfer_scheme = "pic"
        config.integration_scheme = "pic"
        config.solver = "jacobi"
        config.max_iterations = 10
        config.tolerance = 0.0
        config.warmstart_mode = "none"

        self.solver = SolverImplicitMPM(self.model, config=config, enable_timers=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.dt = 0.001

        # Compile and populate persistent fixed-grid data before timing. Two
        # steps preserve the state-buffer orientation used by each ASV repeat.
        self.solver.step(self.state_0, self.state_1, None, None, self.dt)
        self.solver.step(self.state_1, self.state_0, None, None, self.dt)
        wp.synchronize_device(device)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_step(self):
        for _ in range(10):
            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize_device()


if __name__ == "__main__":
    from newton.utils import run_benchmark

    run_benchmark(ImplicitMPMSingleWorld)
