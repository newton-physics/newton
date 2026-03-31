# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.quiet = True

import numpy as np

import newton


def _build_heightfield_scene(num_bodies=200, nrow=100, ncol=100):
    """Build a scene with many spheres dropped onto a large heightfield."""
    builder = newton.ModelBuilder()

    hx, hy = 20.0, 20.0
    x = np.linspace(-hx, hx, ncol)
    y = np.linspace(-hy, hy, nrow)
    xx, yy = np.meshgrid(x, y)
    elevation = np.sin(xx * 0.5) * np.cos(yy * 0.5) * 1.0

    hfield = newton.Heightfield(data=elevation, nrow=nrow, ncol=ncol, hx=hx, hy=hy)
    builder.add_shape_heightfield(heightfield=hfield)

    # Grid of spheres above the terrain
    grid_size = int(np.ceil(np.sqrt(num_bodies)))
    spacing = 2.0 * hx / (grid_size + 1)
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= num_bodies:
                break
            x_pos = -hx + spacing * (i + 1)
            y_pos = -hy + spacing * (j + 1)
            body = builder.add_body(
                xform=wp.transform(p=wp.vec3(x_pos, y_pos, 3.0), q=wp.quat_identity()),
            )
            builder.add_shape_sphere(body=body, radius=0.3)
            count += 1

    model = builder.finalize()
    model.rigid_contact_max = num_bodies * 20
    return model


class HeightfieldCollision:
    """Benchmark heightfield collision with many spheres on a 100x100 grid."""

    repeat = 8
    number = 1

    def setup(self):
        cuda_graph_comp = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if not cuda_graph_comp:
            raise SkipNotImplemented

        self.num_frames = 50
        self.model = _build_heightfield_scene(num_bodies=200, nrow=100, ncol=100)
        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)
        self.contacts = self.model.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.sim_substeps = 10
        self.sim_dt = (1.0 / 100.0) / self.sim_substeps

        wp.synchronize_device()

        with wp.ScopedCapture() as capture:
            for _sub in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.model.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0
        self.graph = capture.graph

        wp.synchronize_device()

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _frame in range(self.num_frames):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "HeightfieldCollision": HeightfieldCollision,
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
