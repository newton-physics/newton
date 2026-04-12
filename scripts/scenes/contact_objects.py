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


def build_model_perturbed(n_worlds: int, epsilon: float = 1e-4) -> newton.Model:
    """N replicated worlds with deterministic per-world z-perturbation.

    Each world's bodies get z-offset = world_index * epsilon [m].
    World 0 is unperturbed (identical to build_model output).
    """
    model = build_model(n_worlds)

    # Perturb joint_q on CPU, then write back.
    joint_q_np = model.joint_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds
    bodies_per_world = model.body_count // n_worlds

    for w in range(n_worlds):
        offset = w * epsilon
        for b in range(bodies_per_world):
            z_idx = w * coords_per_world + b * 7 + 2  # z-component of position
            joint_q_np[z_idx] += offset

    model.joint_q.assign(joint_q_np)

    # Also perturb body_q (used by renderer / solver sync).
    body_q_np = model.body_q.numpy()
    for w in range(n_worlds):
        offset = w * epsilon
        for b in range(bodies_per_world):
            body_idx = w * bodies_per_world + b
            body_q_np[body_idx][2] += offset  # z component of position in transform

    model.body_q.assign(body_q_np)

    return model


def _random_unit_quaternion(rng) -> tuple[float, float, float, float]:
    """Sample a uniform random rotation quaternion (Shoemake's method)."""
    import numpy as np

    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    a1 = 2.0 * math.pi * u2
    a2 = 2.0 * math.pi * u3
    return (s1 * math.sin(a1), s1 * math.cos(a1), s2 * math.sin(a2), s2 * math.cos(a2))


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """N replicated worlds with fully randomized per-world object positions.

    Each world gets deterministically randomized xyz positions and orientations
    for all 18 bodies (9 spheres + 9 boxes). Same object count and shapes,
    completely different spatial arrangement. Seeded per-world for reproducibility.

    Args:
        n_worlds: Number of parallel worlds.
        seed: Base RNG seed. World w uses seed + w.
    """
    import numpy as np

    model = build_model(n_worlds)

    joint_q_np = model.joint_q.numpy()
    body_q_np = model.body_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds
    bodies_per_world = model.body_count // n_worlds

    # Bounds: stay inside the walled enclosure with margin.
    xy_lo, xy_hi = -0.25, 0.25
    z_lo, z_hi = 0.15, 1.50

    for w in range(n_worlds):
        rng = np.random.default_rng(seed + w)

        for b in range(bodies_per_world):
            x = rng.uniform(xy_lo, xy_hi)
            y = rng.uniform(xy_lo, xy_hi)
            z = rng.uniform(z_lo, z_hi)
            qx, qy, qz, qw = _random_unit_quaternion(rng)

            # joint_q: 7 floats per body [px, py, pz, qx, qy, qz, qw]
            base = w * coords_per_world + b * 7
            joint_q_np[base + 0] = x
            joint_q_np[base + 1] = y
            joint_q_np[base + 2] = z
            joint_q_np[base + 3] = qx
            joint_q_np[base + 4] = qy
            joint_q_np[base + 5] = qz
            joint_q_np[base + 6] = qw

            # body_q: transform [px, py, pz, qx, qy, qz, qw]
            body_idx = w * bodies_per_world + b
            body_q_np[body_idx] = (x, y, z, qx, qy, qz, qw)

    model.joint_q.assign(joint_q_np)
    model.body_q.assign(body_q_np)

    return model


def make_solver(
    model: newton.Model,
    tol: float = TOL,
    dt_mode: str = "per_world",
) -> newton.solvers.SolverMuJoCoCENIC:
    """CENIC solver with canonical contact-demo parameters.

    Args:
        model: The model to simulate.
        tol: Inf-norm error tolerance on joint_q per world.
        dt_mode: ``"per_world"`` (default) or ``"global"``.  ``"global"`` forces
            every world to share a single dt driven by the worst-case error,
            used as a baseline for measuring the value of per-world adaptivity.
    """
    return newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=tol,
        dt_inner_init=DT_OUTER,
        dt_inner_min=DT_INNER_MIN,
        dt_inner_max=DT_OUTER,
        dt_mode=dt_mode,
        nconmax=128,
        njmax=640,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    """Fixed-step SolverMuJoCo with matching contact parameters."""
    return newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=128, njmax=640,
    )
