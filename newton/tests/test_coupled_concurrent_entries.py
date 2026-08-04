# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-entry particle hash-grid ownership in SolverCoupled.

The particle hash grid is per-step solver scratch: each particle solver
rebuilds it over its own particles before querying it. Solvers therefore own
a private grid, created at construction when the model-level grid signals
that particle-particle contacts are enabled — sharing one grid across the
entries of a SolverCoupled makes its contents valid only for whichever entry
stepped last, and is a data race when entries step concurrently (multi-stream
or CUDA-graph task parallelism).

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
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM


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
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    dt = 1.0 / 60.0
    solver.prepare_contacts(contacts)

    def step(state_in, state_out):
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)

    src, dst = state_0, state_1
    if use_graph:
        if (frames - graph_warmup) <= 0 or (frames - graph_warmup) % 2 != 0:
            raise ValueError("use_graph requires frames > graph_warmup with an even remainder")
        for _ in range(graph_warmup):
            step(src, dst)
            src, dst = dst, src
        with wp.ScopedDevice(device):
            with wp.ScopedCapture() as capture:
                step(src, dst)
                step(dst, src)
        # The captured pair starts and ends on `src`.
        for _ in range((frames - graph_warmup) // 2):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(frames):
            step(src, dst)
            src, dst = dst, src
    # In both paths the most recently written state is `src`.
    return src.particle_q.numpy().copy()


class TestCoupledEntryParticleGridOwnership(unittest.TestCase):
    def test_entry_solvers_own_private_particle_grids(self):
        """Each particle solver owns its grid scratch instead of sharing the model grid."""
        model, solver = _build_two_pad_coupled()
        grids = {name: entry.solver.particle_grid for name, entry in solver._entries.items()}
        self.assertIsNotNone(model.particle_grid)
        for name, grid in grids.items():
            self.assertIsNotNone(grid, f"entry {name!r} solver should own a particle grid")
            self.assertIsNot(grid, model.particle_grid, f"entry {name!r} solver uses the model grid")
        self.assertIsNot(grids["pad_a"], grids["pad_b"], "entry solvers share one particle grid")


def _build_stacked_pads_admm():
    """Two overlapping XPBD particle pads coupled through an ADMM contact pair.

    The pads share the same footprint at different heights so the upper pad
    lands on the lower one: the pad-to-pad response goes through the ADMM
    contact coupling, exercising the coupling machinery (contact groups,
    per-entry filtering, dual updates) that the plain two-pad scene cannot.
    """
    builder = newton.ModelBuilder(gravity=-9.81)
    rng = np.random.default_rng(0)
    pads = []
    for z0 in (0.15, 0.45):
        ids = [
            builder.add_particle(
                pos=wp.vec3(*(rng.uniform(-0.1, 0.1, 3) * np.array([1.0, 1.0, 0.5]) + np.array([0.0, 0.0, z0]))),
                vel=wp.vec3(0.0),
                mass=0.01,
                radius=0.025,
            )
            for _ in range(27)
        ]
        pads.append(ids)
    builder.add_ground_plane()
    model = builder.finalize()
    solver = SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(name="pad_lo", solver=lambda v: SolverXPBD(model=v), particles=pads[0]),
            SolverCoupled.Entry(name="pad_hi", solver=lambda v: SolverXPBD(model=v), particles=pads[1]),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=6,
            contact_pairs=[SolverCoupledADMM.ContactPair(source="pad_hi", destination="pad_lo")],
        ),
    )
    return model, solver


class _AdmmPhaseParallelStepper:
    """Replica of SolverCoupledADMM._step_coupled with per-entry phases fanned
    out on private streams.

    The prepare and solve phases are per-entry independent (contact filtering
    included); the accumulate and dual phases are coupling reductions and stay
    serial. ``test_admm_replica_matches_upstream`` guards against this replica
    drifting from the upstream loop.
    """

    def __init__(self, solver, parallel: bool):
        self.s = solver
        self.parallel = parallel
        self.device = solver.model.device
        self.streams = {name: wp.Stream(self.device) for name in solver._entries}

    def _fanout(self, body):
        main = self.device.stream
        if not self.parallel:
            for name, entry in self.s._entries.items():
                body(name, entry)
            return
        for name, entry in self.s._entries.items():
            stream = self.streams[name]
            stream.wait_stream(main)
            with wp.ScopedStream(stream, sync_enter=False):
                body(name, entry)
        for stream in self.streams.values():
            main.wait_stream(stream)

    def step_coupled(self, state_in, state_out, control, contacts, dt):
        del state_out
        s = self.s
        coupling = s._coupling
        s._refresh_collision_contact_groups(state_in)
        if float(coupling.gamma) > 0.0:
            s._refresh_admm_proximal_masks()
            s._refresh_admm_proximal_view_overrides(refresh_supported_solvers=True)

        for name, entry in s._entries.items():
            buf = s._admm_buffers[name]
            if buf.body_q_n is not None:
                wp.copy(buf.body_q_n, entry.state_0.body_q)
                wp.copy(buf.body_qd_n, entry.state_0.body_qd)
                wp.copy(buf.body_qd_k, entry.state_0.body_qd)
            if buf.particle_q_n is not None:
                wp.copy(buf.particle_q_n, entry.state_0.particle_q)
                wp.copy(buf.particle_qd_n, entry.state_0.particle_qd)
                wp.copy(buf.particle_qd_k, entry.state_0.particle_qd)
            if buf.joint_q_n is not None:
                wp.copy(buf.joint_q_n, entry.state_0.joint_q)
                wp.copy(buf.joint_qd_n, entry.state_0.joint_qd)
                wp.copy(buf.joint_qd_k, entry.state_0.joint_qd)

        s._admm_begin_step(dt)

        for k in range(int(coupling.iterations)):
            self._fanout(
                lambda name, entry, restart=k > 0: s._prepare_admm_iteration_state(
                    entry, s._admm_buffers[name], state_in, dt, iteration_restart=restart
                )
            )

            s._accumulate_admm_forces(k, dt, refresh_jv=k == 0, initialize_contact_u=k == 0)

            def solve(name, entry):
                buf = s._admm_buffers[name]
                s._apply_admm_force_inputs(entry, buf, dt)
                s._step_entry(entry, control, contacts, dt)
                if buf.body_qd_k is not None:
                    wp.copy(buf.body_qd_k, entry.state_1.body_qd)
                if buf.particle_qd_k is not None:
                    wp.copy(buf.particle_qd_k, entry.state_1.particle_qd)
                if buf.joint_qd_k is not None:
                    wp.copy(buf.joint_qd_k, entry.state_1.joint_qd)

            self._fanout(solve)

            s._update_admm_dual(k, dt)


def _run_admm(parallel: bool | None, frames: int) -> tuple[np.ndarray, int]:
    """Run the stacked-pads ADMM scene; parallel=None uses the upstream loop."""
    model, solver = _build_stacked_pads_admm()
    if parallel is not None:
        stepper = _AdmmPhaseParallelStepper(solver, parallel)
        solver._step_coupled = stepper.step_coupled
    positions = _run_frames(model, solver, frames, use_graph=False)
    return positions, int(solver.collision_contact_count_max)


@unittest.skipUnless(wp.get_device().is_cuda, "requires a CUDA device")
class TestAdmmConcurrentEntryStepping(unittest.TestCase):
    FRAMES = 30
    TOLERANCE = 1.0e-5

    def test_admm_replica_matches_upstream(self):
        """The serial phase replica must track the upstream ADMM loop."""
        ref, ref_contacts = _run_admm(None, self.FRAMES)
        rep, _ = _run_admm(False, self.FRAMES)
        self.assertGreater(ref_contacts, 0, "scene produced no ADMM coupling contacts")
        self.assertLess(float(np.abs(rep - ref).max()), self.TOLERANCE)

    def test_admm_concurrent_entry_phases_match_serial(self):
        """Per-entry prepare/solve phases on private streams reproduce serial ADMM."""
        ref, ref_contacts = _run_admm(None, self.FRAMES)
        con, _ = _run_admm(True, self.FRAMES)
        self.assertGreater(ref_contacts, 0, "scene produced no ADMM coupling contacts")
        self.assertTrue(np.isfinite(con).all())
        self.assertLess(float(np.abs(con - ref).max()), self.TOLERANCE)


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
