# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared builders for the SolverVBD ``rigid_rod_direct_solve`` tests (not a test module)."""

from __future__ import annotations

import warp as wp

import newton

SEG = 0.05  # m
EI = 0.01  # N m^2
EA = 4.0 * EI / SEG**2  # stretch/bend stiffness ratio 1
RADIUS = 0.01  # m
DENSITY = 1000.0  # kg/m^3
GRAVITY = 9.81


def add_rod(builder: newton.ModelBuilder, rod: newton.Rod, wrap: bool = True) -> tuple[list[int], list[int]]:
    """Add a rod with explicit per-joint stiffness (a branched graph has no automatic section rigidity)."""
    k_stretch, k_bend = EA / SEG, EI / SEG
    bodies, joints = builder.add_rod(
        rod=rod,
        cfg=newton.ModelBuilder.ShapeConfig(density=DENSITY),
        stretch_stiffness=k_stretch,
        stretch_damping=0.05 * k_stretch,
        shear_stiffness=k_stretch,
        shear_damping=0.05 * k_stretch,
        bend_stiffness=k_bend,
        bend_damping=0.05 * k_bend,
        twist_stiffness=k_bend,
        twist_damping=0.05 * k_bend,
        wrap_in_articulation=wrap,
    )
    return [int(b) for b in bodies], [int(j) for j in joints]


def make_rod(points, **kwargs) -> newton.Rod:
    return newton.Rod([wp.vec3(*p) for p in points], radius=RADIUS, **kwargs)


def clamp(builder: newton.ModelBuilder, body: int) -> None:
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)
    builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)


def clamped_chain(segments: int, gravity: float = GRAVITY):
    """A straight rod along +X whose first segment is a kinematic clamp."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
    bodies, joints = add_rod(builder, make_rod([((i - 1) * SEG, 0.0, 1.0) for i in range(segments + 1)]))
    clamp(builder, bodies[0])
    return builder, bodies, joints


def clamped_tree(device) -> tuple[newton.Model, list[int]]:
    """A horizontal trunk clamped at its first edge, with a branch leaving its middle node sideways."""
    nodes = [(i * SEG, 0.0, 1.0) for i in range(6)] + [(3 * SEG, SEG, 1.0), (3 * SEG, 2 * SEG, 1.0)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 6), (6, 7)]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    bodies, _ = add_rod(builder, make_rod(nodes, edges=edges))
    clamp(builder, bodies[0])
    builder.color()
    return builder.finalize(device=device), bodies


def run_solver(model: newton.Model, direct: bool, steps: int, dt: float, iterations: int = 10, prepare=None):
    """Step from the model's initial state; ``prepare(solver)`` runs before the first step."""
    solver = newton.solvers.SolverVBD(model, iterations=iterations, rigid_rod_direct_solve=direct)
    if prepare is not None:
        prepare(solver)
    s0, s1, control = model.state(), model.state(), model.control()
    for _ in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
    return solver, s0
