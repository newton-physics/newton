# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Standalone viewer entry point for loading and simulating physics assets."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import warp as wp

import newton

SOLVER_MAP = {
    "mujoco": "SolverMuJoCo",
    "xpbd": "SolverXPBD",
    "featherstone": "SolverFeatherstone",
    "semi-implicit": "SolverSemiImplicit",
}


@dataclass
class SimState:
    """Holds all simulation state for the standalone viewer."""

    model: newton.Model
    solver: object
    control: newton.Control
    state_0: newton.State
    state_1: newton.State
    initial_state: newton.State
    contacts: newton.Contacts
    graph: object | None
    path: str
    dt: float
    sim_substeps: int
    sim_dt: float
    sim_time: float = 0.0


def _create_solver(solver_name: str, model: newton.Model):
    """Create a solver instance from a string name."""
    cls_name = SOLVER_MAP[solver_name]
    cls = getattr(newton.solvers, cls_name)
    return cls(model)


def load_file(
    path: str,
    solver_name: str = "mujoco",
    device: str | None = None,
    ground: bool = True,
) -> SimState:
    """Load a USD/URDF/MJCF file and build a ready-to-simulate SimState.

    Args:
        path: Path to asset file.
        solver_name: Key into SOLVER_MAP.
        device: Warp device string, or None for default.
        ground: Whether to add a ground plane.

    Returns:
        A fully initialized SimState.
    """
    builder = newton.ModelBuilder()
    ext = os.path.splitext(path)[1].lower()

    if ext in (".usd", ".usda", ".usdc"):
        builder.add_usd(path)
    elif ext == ".urdf":
        builder.add_urdf(path)
    elif ext in (".xml", ".mjcf"):
        builder.add_mjcf(path)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Expected .usd, .usda, .usdc, .urdf, .xml, or .mjcf")

    if ground:
        builder.add_shape_plane()

    model = builder.finalize(device=device)
    solver = _create_solver(solver_name, model)
    control = model.control()
    state_0 = model.state()
    state_1 = model.state()
    initial_state = model.state()
    contacts = model.contacts()

    # Determine timestep and substeps
    dt = getattr(model, "dt", 1.0 / 60.0) or 1.0 / 60.0
    sim_substeps = max(1, round(dt * 120.0))
    sim_dt = dt / sim_substeps

    sim = SimState(
        model=model,
        solver=solver,
        control=control,
        state_0=state_0,
        state_1=state_1,
        initial_state=initial_state,
        contacts=contacts,
        graph=None,
        path=path,
        dt=dt,
        sim_substeps=sim_substeps,
        sim_dt=sim_dt,
    )

    # Capture CUDA graph
    if model.device.is_cuda:
        sim.graph = _capture_graph(sim)

    return sim


def _simulate(sim: SimState):
    """Run one frame of simulation (sim_substeps steps).

    Used both for direct execution (CPU) and inside wp.ScopedCapture()
    for CUDA graph recording.  During graph capture, Python executes
    normally (including attribute swaps on sim) while CUDA records kernel
    launches — so the swapped attribute references correctly resolve to
    alternating GPU buffers in the captured graph.

    This is the same pattern used by all existing Newton examples.
    """
    for _ in range(sim.sim_substeps):
        sim.state_0.clear_forces()
        sim.model.collide(sim.state_0, sim.contacts)
        sim.solver.step(sim.state_0, sim.state_1, sim.control, sim.contacts, sim.sim_dt)
        sim.state_0, sim.state_1 = sim.state_1, sim.state_0


def _simulate_with_forces(sim: SimState, viewer):
    """CPU-only simulation path that includes viewer interactive forces."""
    for _ in range(sim.sim_substeps):
        sim.state_0.clear_forces()
        viewer.apply_forces(sim.state_0)
        sim.model.collide(sim.state_0, sim.contacts)
        sim.solver.step(sim.state_0, sim.state_1, sim.control, sim.contacts, sim.sim_dt)
        sim.state_0, sim.state_1 = sim.state_1, sim.state_0


def _capture_graph(sim: SimState):
    """Capture a CUDA graph for the simulation step."""
    with wp.ScopedCapture() as capture:
        _simulate(sim)
    return capture.graph


def _create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the standalone viewer."""
    parser = argparse.ArgumentParser(
        prog="newton-viewer",
        description="Newton standalone viewer — load and simulate physics assets.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to a USD, URDF, or MJCF file to load.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="mujoco",
        choices=list(SOLVER_MAP.keys()),
        help="Solver backend (default: mujoco).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Warp device (default: best available).",
    )
    parser.add_argument(
        "--no-ground",
        action="store_true",
        default=False,
        help="Do not add a ground plane.",
    )
    return parser


def main():
    """Main entry point for the standalone viewer."""
    from newton.viewer import ViewerGL  # noqa: PLC0415

    args = _create_parser().parse_args()

    if args.device:
        wp.set_device(args.device)

    viewer = ViewerGL()
    sim = None
    solver_name = args.solver
    ground = not args.no_ground

    def load_and_setup(path: str):
        nonlocal sim
        new_sim = load_file(path, solver_name=solver_name, device=args.device, ground=ground)
        viewer.set_model(new_sim.model)
        sim = new_sim

    def on_drop(path: str):
        nonlocal sim
        try:
            load_and_setup(path)
        except Exception as e:
            viewer.show_error(str(e))

    def on_reset():
        if sim is not None:
            sim.state_0.assign(sim.initial_state)
            sim.sim_time = 0.0
            # Re-capture graph after state reset
            if sim.model.device.is_cuda:
                sim.graph = _capture_graph(sim)

    def on_solver_change(new_solver: str):
        nonlocal solver_name
        solver_name = new_solver
        if sim is not None:
            try:
                load_and_setup(sim.path)
            except Exception as e:
                viewer.show_error(str(e))

    viewer.on_file_drop = on_drop
    viewer.on_reset = on_reset
    viewer.on_solver_change = on_solver_change

    # Load initial file if provided
    if args.file:
        try:
            load_and_setup(args.file)
        except Exception as e:
            viewer.show_error(str(e))

    # Main loop
    while viewer.is_running():
        should_step = sim is not None and (not viewer.is_paused() or viewer.consume_step_request())

        if should_step:
            if sim.graph:
                wp.capture_launch(sim.graph)
            else:
                _simulate_with_forces(sim, viewer)
            sim.sim_time += sim.dt

        viewer.begin_frame(sim.sim_time if sim else 0.0)
        if sim:
            viewer.log_state(sim.state_0)
        viewer.end_frame()
