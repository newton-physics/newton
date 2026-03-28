# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact objects scene: 9 spheres + 9 tilted boxes per world.

Shared scene definition used by demos and benchmarks. No main(), no CLI,
no viewer logic.
"""

import math

import warp as wp

import newton
import newton.solvers

DT_OUTER = 0.01  # 100 Hz control / render cadence [s]
TOL = 1e-3
DT_INNER_MIN = 1e-6
LOG_EVERY = 250

SPHERE_RADIUS = 0.050
BOX_HALF = 0.050
GRID_STEP = 0.200
GRID_OFFSETS = [-GRID_STEP, 0.0, GRID_STEP]
Z_SPHERES = 1.00
Z_BOXES = 1.25


def build_template() -> newton.ModelBuilder:
    """Single-world template: 9 spheres + 9 tilted boxes."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(template)

    cfg_obj = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.3, margin=0.005)

    for ox in GRID_OFFSETS:
        for oy in GRID_OFFSETS:
            b = template.add_body(
                xform=wp.transform(p=wp.vec3(ox, oy, Z_SPHERES), q=wp.quat_identity()),
            )
            template.add_shape_sphere(b, radius=SPHERE_RADIUS, cfg=cfg_obj)

    _box_angles = [
        (15, 0, 0),
        (-20, 10, 0),
        (35, 0, 15),
        (0, 25, -10),
        (49, 0, 0),
        (-30, 20, 5),
        (10, -35, 0),
        (0, 15, 40),
        (-15, 0, -25),
    ]
    for (ox, oy), (ax, ay, az) in zip(
        [(ox, oy) for ox in GRID_OFFSETS for oy in GRID_OFFSETS],
        _box_angles,
    ):
        rx, ry, rz = math.radians(ax), math.radians(ay), math.radians(az)
        cx, sx = math.cos(rx / 2), math.sin(rx / 2)
        cy, sy = math.cos(ry / 2), math.sin(ry / 2)
        cz, sz = math.cos(rz / 2), math.sin(rz / 2)
        q = wp.quat(
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        )
        b = template.add_body(xform=wp.transform(p=wp.vec3(ox, oy, Z_BOXES), q=q))
        template.add_shape_box(b, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg_obj)

    return template


def build_model(n_worlds: int) -> newton.Model:
    """N replicated worlds + ground plane + invisible walls."""
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()

    cfg_wall = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.3, margin=0.005, is_visible=False)
    half_inner = 0.350
    wt = 0.025
    wh = 0.750
    for px, py, hx, hy in [
        (-(half_inner + wt), 0.0, wt, half_inner + wt),
        (half_inner + wt, 0.0, wt, half_inner + wt),
        (0.0, -(half_inner + wt), half_inner + wt, wt),
        (0.0, half_inner + wt, half_inner + wt, wt),
    ]:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(px, py, wh), q=wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=wh,
            cfg=cfg_wall,
        )
    return builder.finalize()


def make_solver(model: newton.Model, tol: float = TOL) -> newton.solvers.SolverMuJoCoCENIC:
    """CENIC solver with canonical contact-demo parameters."""
    return newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=tol,
        dt_inner_init=DT_OUTER,
        dt_inner_min=DT_INNER_MIN,
        dt_inner_max=DT_OUTER,
        nconmax=128,
        njmax=640,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    """Fixed-step SolverMuJoCo with matching contact parameters."""
    return newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=128, njmax=640,
    )
