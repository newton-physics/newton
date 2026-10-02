# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-soft contact substep cost of ``SolverVBD`` across world counts.

Each world holds a soft tetrahedral slab resting on a static box with full-surface (particle, edge
and face) soft contacts. The sweep covers the latency-bound small-batch regime and the
throughput-bound large-batch regime of the body-particle contact accumulation
(newton-physics/newton#4404).
"""

import os
import sys

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from benchmark_config import pr_gate_repeat

import newton

WORLD_COUNTS = (1, 16, 128, 1024)


def _make_contact_world(cells: int = 8) -> newton.ModelBuilder:
    """One soft slab of ``cells`` x ``cells`` x 2 cells (1 cm each) touching a static box."""
    world = newton.ModelBuilder(gravity=wp.vec3(0.0))
    world.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, -0.003),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=cells,
        dim_y=cells,
        dim_z=2,
        cell_x=0.01,
        cell_y=0.01,
        cell_z=0.01,
        density=1000.0,
        k_mu=1.0e4,
        k_lambda=1.0e4,
        k_damp=0.1,
        tri_ke=0.0,
        tri_ka=0.0,
        particle_radius=0.001,
    )
    world.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.04, 0.04, -0.05), wp.quat_identity()),
        hx=0.2,
        hy=0.2,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=0.0, mu=0.0),
    )
    world.color()
    return world


class FastVBDRigidSoftContact:
    """Time a captured rigid-soft contact substep of ``SolverVBD`` for several world counts."""

    params = (WORLD_COUNTS,)
    param_names = ["world_count"]
    repeat = pr_gate_repeat(5)
    number = 1
    timeout = 900
    warmup_count = 20
    launch_count = 20

    def setup(self, world_count):
        device = wp.get_device()
        if not device.is_cuda:
            raise SkipNotImplemented

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.replicate(_make_contact_world(), world_count)
        self.model = builder.finalize(device=device)
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_gap=0.005,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(self.model, iterations=10)

        # Compile and size every buffer before capture.
        self._substep()
        active = int(self.contacts.soft_contact_count.numpy()[0])
        if not 0 < active <= self.contacts.soft_contact_max:
            raise RuntimeError(f"unexpected soft contact count {active} (capacity {self.contacts.soft_contact_max})")

        with wp.ScopedCapture(device=device) as capture:
            self._substep()
        self.graph = capture.graph
        for _ in range(self.warmup_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()

    def _substep(self):
        # Reset the particle state so every replay solves the same contact configuration.
        wp.copy(self.state_in.particle_q, self.model.particle_q)
        wp.copy(self.state_in.particle_qd, self.model.particle_qd)
        self.pipeline.collide(self.state_in, self.contacts)
        self.solver.step(self.state_in, self.state_out, self.control, self.contacts, 1.0 / 600.0)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_substep(self, world_count):
        for _ in range(self.launch_count):
            wp.capture_launch(self.graph)
        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastVBDRigidSoftContact": FastVBDRigidSoftContact,
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
