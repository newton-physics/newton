# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Coupled Concurrent Entries
#
# Two independent XPBD particle pads fall onto the ground plane. The pads
# are two entries of a SolverCoupled, and the example overrides the
# _step_coupled template method to step each entry on its own CUDA stream,
# so the two sub-solvers execute concurrently on the device. The whole frame
# (including the multi-stream fork/join) is captured into a single CUDA
# graph and replayed.
#
# This relies on entry model views owning PRIVATE particle hash grids:
# the grid is per-step solver scratch, and entries stepping concurrently
# would race on a shared one.
#
# Pass ``--stepping serial`` to run the standard sequential entry loop for
# comparison; both modes produce the same trajectories.
#
# Command: python -m newton.examples coupled_concurrent_entries
#          python -m newton.examples coupled_concurrent_entries --stepping serial
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled

import newton
import newton.examples
from newton.solvers import SolverXPBD


class ConcurrentEntrySolverCoupled(SolverCoupled):
    """SolverCoupled that steps every entry concurrently on a private stream.

    ``_step_coupled`` is the documented template method for coupling
    algorithms; the default implementation steps entries sequentially. The
    entries of this example are fully independent (no coupling terms), so
    each one can run on its own stream, forked from and joined back into the
    device's current stream. The fork/join events are legal inside CUDA
    graph capture, so the concurrent frame can be captured and replayed.
    """

    def _step_coupled(self, state_in, state_out, control, contacts, dt):
        del state_out
        device = self.model.device
        streams = getattr(self, "_entry_streams", None)
        if streams is None:
            streams = {name: wp.Stream(device) for name in self._entries}
            self._entry_streams = streams
        main = device.stream
        for name, entry in self._entries.items():
            stream = streams[name]
            stream.wait_stream(main)
            with wp.ScopedStream(stream, sync_enter=False):
                self._step_entry(entry, control, contacts, dt)
        for stream in streams.values():
            main.wait_stream(stream)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(gravity=-9.81)
        rng = np.random.default_rng(args.seed)
        self.pads = []
        n = int(args.particles_per_pad)
        for x0 in (-0.5, 0.5):
            ids = [
                builder.add_particle(
                    pos=wp.vec3(*(rng.uniform(-0.12, 0.12, 3) + np.array([x0, 0.0, args.drop_height]))),
                    vel=wp.vec3(0.0),
                    mass=0.01,
                    radius=0.025,
                )
                for _ in range(n)
            ]
            self.pads.append(ids)
        builder.add_ground_plane()
        self.model = builder.finalize()

        solver_cls = ConcurrentEntrySolverCoupled if args.stepping == "concurrent" else SolverCoupled
        self.solver = solver_cls(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="pad_a",
                    solver=lambda v: SolverXPBD(model=v, iterations=args.xpbd_iterations),
                    particles=self.pads[0],
                ),
                SolverCoupled.Entry(
                    name="pad_b",
                    solver=lambda v: SolverXPBD(model=v, iterations=args.xpbd_iterations),
                    particles=self.pads[1],
                ),
            ],
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.particle_radii = wp.full(2 * n, 0.025, dtype=wp.float32, device=self.model.device)
        self.particle_colors = wp.array(
            [wp.vec3(0.12, 0.38, 0.92)] * n + [wp.vec3(0.92, 0.42, 0.12)] * n,
            dtype=wp.vec3,
            device=self.model.device,
        )

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        if not self.model.device.is_cuda:
            self.graph = None
            return
        with wp.ScopedDevice(self.model.device):
            with wp.ScopedCapture() as capture:
                self.simulate()
        self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            with wp.ScopedDevice(self.model.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_points(
            "/coupled_concurrent/particles",
            self.state_0.particle_q,
            radii=self.particle_radii,
            colors=self.particle_colors,
        )
        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        assert np.isfinite(particle_q).all(), "Particle positions contain NaN or inf values"
        # Both pads must have fallen and come to rest on (not through) the ground.
        assert particle_q[:, 2].min() > -0.05, f"particles fell through the ground: min_z={particle_q[:, 2].min():.3f}"
        assert particle_q[:, 2].max() < 0.4, f"particles did not settle: max_z={particle_q[:, 2].max():.3f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--stepping",
            help="'concurrent' steps each entry on its own CUDA stream, 'serial' uses the default entry loop",
            type=str,
            choices=["concurrent", "serial"],
            default="concurrent",
        )
        parser.add_argument("--particles-per-pad", help="Particles per pad", type=int, default=27)
        parser.add_argument("--xpbd-iterations", help="XPBD iterations per entry solve", type=int, default=6)
        parser.add_argument("--drop-height", help="Initial pad height above the ground", type=float, default=0.5)
        parser.add_argument("--seed", help="Random seed for pad particle placement", type=int, default=0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
