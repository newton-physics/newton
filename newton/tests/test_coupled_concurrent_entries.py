# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-entry particle hash-grid ownership in SolverCoupled.

The particle hash grid is per-step solver scratch: each particle solver
rebuilds it over its own particles before querying it. Entry model views must
therefore own a private grid — aliasing the parent model's grid makes its
contents valid only for whichever entry stepped last, and is a data race when
entries step concurrently (multi-stream or CUDA-graph task parallelism).

These tests assert the ownership invariant and validate that stepping the
entries of a SolverCoupled concurrently on private streams — eagerly and
under CUDA graph capture — reproduces serial stepping.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverXPBD
from newton.solvers.experimental.coupled import SolverCoupled


def _build_two_pad_coupled(seed: int = 0, solver_cls=None):
    """Two disjoint XPBD particle pads above a ground plane."""
    solver_cls = solver_cls or SolverCoupled
    builder = newton.ModelBuilder(gravity=-9.81)
    rng = np.random.default_rng(seed)
    pads = []
    for x0 in (-0.5, 0.5):
        ids = [
            builder.add_particle(
                pos=wp.vec3(*(rng.uniform(-0.1, 0.1, 3) + np.array([x0, 0.0, 0.5]))),
                vel=wp.vec3(0.0),
                mass=0.01,
                radius=0.025,
            )
            for _ in range(27)
        ]
        pads.append(ids)
    builder.add_ground_plane()
    model = builder.finalize()
    solver = solver_cls(
        model=model,
        entries=[
            SolverCoupled.Entry(name="pad_a", solver=lambda v: SolverXPBD(model=v), particles=pads[0]),
            SolverCoupled.Entry(name="pad_b", solver=lambda v: SolverXPBD(model=v), particles=pads[1]),
        ],
    )
    return model, solver


class _ConcurrentEntrySolverCoupled(SolverCoupled):
    """SolverCoupled stepping every entry concurrently on a private stream."""

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


def _build_two_pad_concurrent(seed: int = 0):
    return _build_two_pad_coupled(seed, solver_cls=_ConcurrentEntrySolverCoupled)


def _run_frames(model, solver, frames: int, use_graph: bool, graph_warmup: int = 30) -> np.ndarray:
    """Step ``frames`` frames and return the final particle positions.

    States ping-pong between two buffers so each step consumes the previous
    step's output. With ``use_graph``, an even pair of steps is captured (so
    every replay starts and ends on the same physical buffer) after
    ``graph_warmup`` eager frames: the capture must happen once the contact
    path is warm — capturing before the first contact bakes the lazily
    initialized contact filtering state and deterministically changes the
    trajectory (verified: warm capture is bit-exact vs eager stepping).
    """
    device = model.device
    state_0, state_1 = model.state(), model.state()
    contacts = model.contacts()
    dt = 1.0 / 60.0
    solver.prepare_contacts(contacts)

    def step(state_in, state_out):
        state_in.clear_forces()
        model.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)

    states = [state_0, state_1]
    n = 0
    if use_graph:
        if (frames - graph_warmup) <= 0 or (frames - graph_warmup) % 2 != 0:
            raise ValueError("use_graph requires frames > graph_warmup with an even remainder")
        for _ in range(graph_warmup):
            step(states[n % 2], states[(n + 1) % 2])
            n += 1
        with wp.ScopedDevice(device):
            with wp.ScopedCapture() as capture:
                step(states[n % 2], states[(n + 1) % 2])
                step(states[(n + 1) % 2], states[n % 2])
        for _ in range((frames - graph_warmup) // 2):
            wp.capture_launch(capture.graph)
        final = states[n % 2]
    else:
        for _ in range(frames):
            step(states[n % 2], states[(n + 1) % 2])
            n += 1
        final = states[n % 2]
    return final.particle_q.numpy().copy()


class TestCoupledEntryParticleGridOwnership(unittest.TestCase):
    def test_entry_views_own_private_particle_grids(self):
        model, solver = _build_two_pad_coupled()
        grids = {name: entry.solver.model.particle_grid for name, entry in solver._entries.items()}
        self.assertIsNotNone(model.particle_grid)
        for name, grid in grids.items():
            self.assertIsNotNone(grid, f"entry {name!r} should own a particle grid")
            self.assertIsNot(grid, model.particle_grid, f"entry {name!r} aliases the parent grid")
        self.assertIsNot(grids["pad_a"], grids["pad_b"], "entries alias one particle grid")


@unittest.skipUnless(wp.get_device().is_cuda, "requires a CUDA device")
class TestCoupledConcurrentEntryStepping(unittest.TestCase):
    FRAMES = 60
    TOLERANCE = 1.0e-5

    def test_concurrent_stepping_matches_serial(self):
        model_ref, solver_ref = _build_two_pad_coupled()
        ref = _run_frames(model_ref, solver_ref, self.FRAMES, use_graph=False)

        model_c, solver_c = _build_two_pad_concurrent()
        con = _run_frames(model_c, solver_c, self.FRAMES, use_graph=False)

        self.assertTrue(np.isfinite(con).all())
        self.assertLess(float(np.abs(con - ref).max()), self.TOLERANCE)

    def test_concurrent_stepping_under_graph_capture_matches_serial(self):
        model_ref, solver_ref = _build_two_pad_coupled()
        ref = _run_frames(model_ref, solver_ref, self.FRAMES, use_graph=True)

        model_c, solver_c = _build_two_pad_concurrent()
        con = _run_frames(model_c, solver_c, self.FRAMES, use_graph=True)

        self.assertTrue(np.isfinite(con).all())
        self.assertLess(float(np.abs(con - ref).max()), self.TOLERANCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
