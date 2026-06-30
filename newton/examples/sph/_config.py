# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverWCSPH, sph

SPH_KERNEL_NAMES = ("poly6", "cubic", "wendland", "spiky")
SPH_TANK_WALL_ORDER = ("floor", "left", "right", "back", "front")
_SPH_TANK_WALL_COLORS = {
    "floor": (0.55, 0.55, 0.58),
    "left": (0.50, 0.52, 0.56),
    "right": (0.50, 0.52, 0.56),
    "back": (0.50, 0.52, 0.56),
    "front": (0.50, 0.52, 0.56),
}

_NON_POSITIVE_AUTOMATIC_OPTIONS = {
    "smoothing_length",
}


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def assert_sph_state_finite(state: newton.State, *sph_fields: str) -> None:
    """Assert that core particle state and selected SPH fields are finite."""

    assert np.isfinite(state.particle_q.numpy()).all()
    assert np.isfinite(state.particle_qd.numpy()).all()
    for field_name in sph_fields:
        assert np.isfinite(getattr(state.sph, field_name).numpy()).all()


def assert_sph_role_count(model: newton.Model, role: sph.SPHRole, indices: np.ndarray) -> None:
    """Assert that the model contains exactly ``indices`` particles with an SPH role."""

    assert np.count_nonzero(model.sph.role.numpy() == int(role)) == len(indices)


def sph_particle_spacing_from_args(args: object) -> tuple[float, float, float]:
    spacing = args.spacing
    radius = args.radius if args.radius > 0.0 else 0.5 * spacing
    smoothing_length = args.smoothing_length if args.smoothing_length > 0.0 else 2.0 * spacing
    return spacing, radius, smoothing_length


def add_sph_particle_grid_excluding_spheres(
    builder: newton.ModelBuilder,
    *,
    pos: Any,
    vel: Any,
    dim_x: int,
    dim_y: int,
    dim_z: int,
    cell_x: float,
    cell_y: float,
    cell_z: float,
    material: sph.SPHMaterial,
    exclude_spheres: tuple[tuple[tuple[float, float, float], float], ...],
    jitter: float,
    radius_mean: float,
    seed: int = 17,
) -> np.ndarray:
    """Add a fluid particle grid while skipping particles inside spherical exclusions."""

    SolverWCSPH.register_custom_attributes(builder)
    attrs = {
        "sph:role": int(sph.SPHRole.FLUID),
        **material.custom_attributes(),
    }
    origin = np.asarray(pos, dtype=np.float64)
    velocity = wp.vec3(*np.asarray(vel, dtype=np.float64))
    exclusions = tuple((np.asarray(center, dtype=np.float64), float(radius)) for center, radius in exclude_spheres)
    mass = material.rest_density * cell_x * cell_y * cell_z
    rng = np.random.default_rng(seed)
    indices = []

    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                point = origin + np.array((i * cell_x, j * cell_y, k * cell_z), dtype=np.float64)
                if jitter > 0.0:
                    point += rng.uniform(-jitter, jitter, size=3)
                if any(np.linalg.norm(point - center) < radius for center, radius in exclusions):
                    continue
                indices.append(
                    builder.add_particle(
                        pos=wp.vec3(float(point[0]), float(point[1]), float(point[2])),
                        vel=velocity,
                        mass=mass,
                        radius=radius_mean,
                        custom_attributes=attrs,
                    )
                )

    return np.asarray(indices, dtype=np.int64)


def add_sph_particle_arguments(
    parser,
    *,
    spacing: float,
    jitter: float | None = 0.001,
    spacing_help: str = "Particle lattice spacing [m].",
) -> None:
    """Add common SPH particle-size CLI arguments to an example parser."""

    parser.add_argument("--spacing", type=_positive_float, default=spacing, help=spacing_help)
    parser.add_argument(
        "--radius",
        type=_non_negative_float,
        default=0.0,
        help="Particle render/collision radius [m]; 0 derives from spacing.",
    )
    parser.add_argument(
        "--smoothing-length",
        type=_non_negative_float,
        default=0.0,
        help="SPH support radius [m]; 0 derives from spacing.",
    )
    if jitter is not None:
        parser.add_argument(
            "--jitter", type=_non_negative_float, default=jitter, help="Initial random particle jitter [m]."
        )


def add_sph_block_dimension_arguments(
    parser,
    *,
    dim_x: int,
    dim_z: int,
    dim_y: int | None = None,
    label: str = "Particle",
) -> None:
    """Add common lattice block dimension arguments to an SPH example parser."""

    parser.add_argument("--dim-x", type=_positive_int, default=dim_x, help=f"{label} count along X.")
    if dim_y is not None:
        parser.add_argument("--dim-y", type=_positive_int, default=dim_y, help=f"{label} count along Y.")
    parser.add_argument("--dim-z", type=_positive_int, default=dim_z, help=f"{label} count along Z.")


def add_sph_tank_arguments(
    parser,
    *,
    tank_width: float,
    wall_height: float,
    fluid_offset_y: float,
    tank_length: float = 0.8,
    wall_thickness: float = 0.035,
) -> None:
    """Add common open-top tank geometry arguments to an SPH example parser."""

    parser.add_argument("--gravity", type=_finite_float, default=-9.81, help="Vertical gravity acceleration [m/s^2].")
    parser.add_argument("--tank-length", type=_positive_float, default=tank_length, help="Tank length along X [m].")
    parser.add_argument("--tank-width", type=_positive_float, default=tank_width, help="Tank width along Z [m].")
    parser.add_argument("--wall-height", type=_positive_float, default=wall_height, help="Tank wall height [m].")
    parser.add_argument(
        "--wall-thickness", type=_positive_float, default=wall_thickness, help="Tank wall/floor thickness [m]."
    )
    parser.add_argument(
        "--fluid-offset-y",
        type=_non_negative_float,
        default=fluid_offset_y,
        help="Initial water offset above the floor [m].",
    )


def add_sph_timestep_arguments(parser, *, fps: float = 60.0, substeps: int = 8) -> None:
    """Add common SPH example timestep CLI arguments."""

    parser.add_argument("--fps", type=_positive_float, default=fps, help="Frames per second.")
    parser.add_argument("--substeps", type=_positive_int, default=substeps, help="SPH substeps per rendered frame.")


def validate_sph_example_timestep_args(args: object) -> None:
    """Validate shared SPH example timestep fields."""

    if args.fps <= 0.0:
        raise ValueError("fps must be positive")
    if args.substeps <= 0:
        raise ValueError("substeps must be positive")


def add_sph_solver_config_arguments(
    parser,
    *,
    kernel: str = "wendland",
    rest_density: float | None = 1000.0,
    sound_speed: float | None = 12.0,
    viscosity: float | None = 0.001,
    xsph: float | None = 0.04,
    boundary_friction: float | None = None,
) -> None:
    """Add common SPH solver CLI arguments to an example parser."""

    parser.add_argument("--kernel", choices=SPH_KERNEL_NAMES, default=kernel, help="SPH smoothing kernel family.")
    if rest_density is not None:
        parser.add_argument(
            "--rest-density", type=_positive_float, default=rest_density, help="Fluid rest density [kg/m^3]."
        )
    if sound_speed is not None:
        parser.add_argument(
            "--sound-speed", type=_positive_float, default=sound_speed, help="WCSPH artificial sound speed [m/s]."
        )
    if viscosity is not None:
        parser.add_argument(
            "--viscosity", type=_non_negative_float, default=viscosity, help="SPH viscosity coefficient."
        )
    if xsph is not None:
        parser.add_argument(
            "--xsph", type=_non_negative_float, default=xsph, help="XSPH velocity smoothing coefficient."
        )
    if boundary_friction is not None:
        parser.add_argument(
            "--boundary-friction", type=_non_negative_float, default=boundary_friction, help="Boundary friction."
        )


def _apply_base_options(options: SolverWCSPH.Config, base_options: SolverWCSPH.Config | Mapping[str, Any]) -> None:
    if isinstance(base_options, SolverWCSPH.Config):
        for field in fields(SolverWCSPH.Config):
            setattr(options, field.name, getattr(base_options, field.name))
        return

    config = base_options.get("config")
    if isinstance(config, SolverWCSPH.Config):
        _apply_base_options(options, config)

    for key, value in base_options.items():
        if key == "config":
            continue
        if hasattr(options, key):
            setattr(options, key, value)


def sph_options_from_args(
    args: object,
    *,
    base_options: SolverWCSPH.Config | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> SolverWCSPH.Config:
    """Build ``SolverWCSPH.Config`` from matching example CLI arguments."""

    options = SolverWCSPH.Config()
    if base_options is not None:
        _apply_base_options(options, base_options)
    for key, value in vars(args).items():
        if key in _NON_POSITIVE_AUTOMATIC_OPTIONS and (value is None or value <= 0.0):
            continue
        if hasattr(options, key):
            setattr(options, key, value)
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


def sph_velocity_colors(velocities: np.ndarray, speed_scale: float | None = None) -> np.ndarray:
    """Return fluid point colors from velocity magnitude."""

    speeds = np.linalg.norm(velocities, axis=1)
    if speed_scale is None:
        speed_scale = float(np.percentile(speeds, 95)) if speeds.size else 0.0
    if speed_scale <= 1.0e-8:
        speed_scale = 1.0

    t = np.clip(speeds / speed_scale, 0.0, 1.0).astype(np.float32)
    slow = np.array((0.10, 0.32, 0.95), dtype=np.float32)
    mid = np.array((0.05, 0.75, 1.00), dtype=np.float32)
    fast = np.array((1.00, 0.78, 0.18), dtype=np.float32)

    colors = np.empty((velocities.shape[0], 3), dtype=np.float32)
    lower = t < 0.55
    upper = ~lower
    lower_t = np.clip(t[lower] / 0.55, 0.0, 1.0)[:, None]
    upper_t = np.clip((t[upper] - 0.55) / 0.45, 0.0, 1.0)[:, None]
    colors[lower] = (1.0 - lower_t) * slow + lower_t * mid
    colors[upper] = (1.0 - upper_t) * mid + upper_t * fast
    return colors


def log_sph_fluid_points(
    viewer: object,
    state: newton.State,
    model: newton.Model,
    fluid_indices: np.ndarray,
    *,
    name: str = "/sph_fluid",
    radius_scale: float = 0.55,
    hidden: bool = False,
) -> None:
    """Log SPH fluid particles as a clean Newton point cloud."""

    indices = np.asarray(fluid_indices, dtype=np.int64)
    if indices.size == 0:
        return

    q = state.particle_q.numpy()[indices].astype(np.float32)
    qd = state.particle_qd.numpy()[indices].astype(np.float32)
    radii = (float(radius_scale) * model.particle_radius.numpy()[indices]).astype(np.float32)
    colors = sph_velocity_colors(qd)

    viewer.log_points(
        name,
        points=wp.array(q, dtype=wp.vec3, device=model.device),
        radii=wp.array(radii, dtype=wp.float32, device=model.device),
        colors=wp.array(colors, dtype=wp.vec3, device=model.device),
        hidden=hidden,
    )


def create_sph_visual_box_mesh(half_extents: tuple[float, float, float]) -> newton.Mesh:
    """Create an opaque box visual with consistent lighting from either winding."""

    mesh = newton.Mesh.create_box(*half_extents, compute_inertia=False)
    triangles = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
    # Preserve the outward normals so either front-face convention shades the box consistently.
    indices = np.concatenate((triangles, triangles[:, [0, 2, 1]]), axis=0).reshape(-1)
    return newton.Mesh(
        vertices=mesh.vertices,
        indices=indices,
        normals=mesh.normals,
        uvs=mesh.uvs,
        compute_inertia=False,
    )


def add_sph_analytic_tank_shapes(
    builder: newton.ModelBuilder,
    *,
    tank_length: float,
    tank_width: float,
    wall_height: float,
    wall_thickness: float,
    boundary_friction: float,
    wall_order: tuple[str, ...] = SPH_TANK_WALL_ORDER,
    wall_colors: Mapping[str, tuple[float, float, float]] | None = None,
) -> dict[str, int]:
    """Add invisible Newton collider shapes for a simple open-top rectangular tank."""

    half_length = 0.5 * tank_length
    half_width = 0.5 * tank_width
    half_height = 0.5 * wall_height
    half_wall = 0.5 * wall_thickness
    colors = dict(_SPH_TANK_WALL_COLORS)
    if wall_colors is not None:
        colors.update(wall_colors)

    wall_specs = {
        "floor": (wp.vec3(0.0, 0.0, 0.0), half_length, half_wall, half_width),
        "left": (wp.vec3(-half_length, half_height, 0.0), half_wall, half_height, half_width),
        "right": (wp.vec3(half_length, half_height, 0.0), half_wall, half_height, half_width),
        "back": (wp.vec3(0.0, half_height, -half_width), half_length, half_height, half_wall),
        "front": (wp.vec3(0.0, half_height, half_width), half_length, half_height, half_wall),
    }
    invalid_walls = tuple(wall for wall in wall_order if wall not in wall_specs)
    if invalid_walls:
        raise ValueError(f"Unknown SPH tank wall name(s): {', '.join(invalid_walls)}")

    result: dict[str, int] = {}
    for wall_name in wall_order:
        center, hx, hy, hz = wall_specs[wall_name]
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=boundary_friction)
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = True
        shape_cfg.is_visible = False
        result[wall_name] = builder.add_shape_box(
            body=-1,
            xform=wp.transform(center, wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=shape_cfg,
            color=colors[wall_name],
        )
    return result


class SPHExampleBase:
    """Shared fixed-substep loop for simple SPH examples."""

    advance_time_per_substep = False
    fluid_render_radius_scale = 0.55

    def __init__(self, viewer, args):
        validate_sph_example_timestep_args(args)
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

    def before_substep(self):
        return None

    def after_substep(self):
        return None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.before_substep()
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, control=None, contacts=None, dt=self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.after_substep()
            if self.advance_time_per_substep:
                self.sim_time += self.sim_dt

    def step(self):
        self.simulate()
        if not self.advance_time_per_substep:
            self.sim_time += self.frame_dt

    def render(self):
        show_particles = self.viewer.show_particles
        self.viewer.begin_frame(self.sim_time)
        self.viewer.show_particles = False
        self.viewer.log_state(self.state_0)
        self.viewer.show_particles = show_particles
        if hasattr(self, "fluid_indices"):
            log_sph_fluid_points(
                self.viewer,
                self.state_0,
                self.model,
                self.fluid_indices,
                radius_scale=self.fluid_render_radius_scale,
                hidden=not show_particles,
            )
        self.viewer.end_frame()
