# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverVBD rod joints: graph-colouring iterations vs the direct (serial, per-rod) solve.

Both layouts hold the same number of rod bodies: a few long rods (deep serial walk, little parallelism across rods)
and many short rods (shallow walk, wide parallelism). Each rod hangs from a kinematic first segment under gravity.
"""

import inspect

import warp as wp

wp.config.enable_backward = False
wp.config.log_level = wp.LOG_WARNING

import newton

LAYOUTS = {"few_long": (4, 128), "many_short": (128, 4)}  # (rods, dynamic segments per rod)
SEGMENT = 0.01  # m
DEVICE = None  # Warp's default device; set to "cpu" to time the CPU path


def _has_direct_solve() -> bool:
    return "rigid_rod_direct_solve" in inspect.signature(newton.solvers.SolverVBD).parameters


def build_rods(rods: int, segments: int, device=None) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for r in range(rods):
        points = [wp.vec3(i * SEGMENT, 0.05 * r, 1.0) for i in range(segments + 2)]
        bodies, _ = builder.add_rod(
            rod=newton.Rod(points, radius=0.002),
            stretch_stiffness=2.0e5,
            bend_stiffness=0.08,
            bend_damping=1.0e-3,
            label=f"rod_{r}",
        )
        anchor = int(bodies[0])
        builder.body_mass[anchor] = 0.0
        builder.body_inv_mass[anchor] = 0.0
        builder.body_inertia[anchor] = wp.mat33(0.0)
        builder.body_inv_inertia[anchor] = wp.mat33(0.0)
        builder.body_flags[anchor] = int(newton.BodyFlags.KINEMATIC)
    builder.color()
    return builder.finalize(device=device)


class RodDirectSolve:
    params = (list(LAYOUTS), ["colouring", "direct"])
    param_names = ["layout", "rod_solve"]
    number = 1
    rounds = 2
    repeat = 2
    steps = 60

    def setup(self, layout, rod_solve):
        if rod_solve == "direct" and not _has_direct_solve():
            raise NotImplementedError("SolverVBD has no rigid_rod_direct_solve option")
        self.model = build_rods(*LAYOUTS[layout], device=DEVICE)
        kwargs = {"rigid_rod_direct_solve": True} if rod_solve == "direct" else {}
        self.solver = newton.solvers.SolverVBD(self.model, iterations=10, **kwargs)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.dt = 1.0 / 600.0
        self.time_step(layout, rod_solve)  # compile kernels outside the timed region
        wp.synchronize_device(self.model.device)

    def time_step(self, layout, rod_solve):
        for _ in range(self.steps):
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        wp.synchronize_device(self.model.device)


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default=None, help="Warp device, e.g. cpu or cuda:0")
    args = parser.parse_known_args()[0]
    DEVICE = args.device
    run_benchmark(RodDirectSolve)
