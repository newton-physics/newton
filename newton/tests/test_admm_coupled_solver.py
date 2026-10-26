# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ADMM-coupled solvers.

These tests validate generic :class:`SolverCoupledADMM` ADMM plumbing against a
cloth-plus-rigid-body scene.
"""

from __future__ import annotations

import inspect
import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.sim.collide import _TRIANGLE_PAIRS_MIN_CAPACITY
from newton._src.solvers.coupled.interface import CouplingInterface
from newton.solvers import (
    SolverBase,
    SolverFeatherstone,
    SolverKamino,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverVBD,
    SolverXPBD,
)
from newton.solvers.experimental.coupled import (
    ModelView,
    SolverCoupled,
    SolverCoupledADMM,
)


@wp.kernel(enable_backward=False)
def _set_admm_plane_angle_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], angle: float):
    body_q[0] = wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle),
    )
    body_qd[0] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class _CustomAdmmParticleCopySolver(SolverBase, CouplingInterface):
    """Base test solver that copies particle state."""

    def step(self, state_in, state_out, control, contacts, dt):
        del control, contacts, dt
        if state_in.particle_q is not None and state_out.particle_q is not None:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
        if state_in.body_q is not None and state_out.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)


class _KinematicAdmmPlaneSolver(_CustomAdmmParticleCopySolver):
    """Test solver that prescribes a fixed kinematic plane angle."""

    def __init__(self, model, angle):
        super().__init__(model)
        self.angle = float(angle)

    def step(self, state_in, state_out, control, contacts, dt):
        super().step(state_in, state_out, control, contacts, dt)
        wp.launch(
            _set_admm_plane_angle_kernel,
            dim=1,
            inputs=[state_out.body_q, state_out.body_qd, self.angle],
            device=self.model.device,
        )


def _build_cloth_rigid_scene(
    rigid_pos: tuple[float, float, float] = (0.0, 0.0, 1.5),
    rigid_mass: float = 0.05,
    cloth_pos: tuple[float, float, float] = (-0.25, -0.25, 1.5),
    dim_xy: int = 5,
    fix_cloth_edges: bool = True,
) -> tuple[newton.Model, int, int, int]:
    """Build a pinned cloth + free rigid body scene for attachment tests."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()

    rigid_start = builder.body_count
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(*rigid_pos), q=wp.quat_identity()),
        mass=rigid_mass,
        inertia=wp.mat33(np.eye(3) * 0.001),
    )
    builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03)
    rigid_end = builder.body_count

    particle_start = builder.particle_count
    builder.add_cloth_grid(
        pos=wp.vec3(*cloth_pos),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        fix_left=fix_cloth_edges,
        fix_right=fix_cloth_edges,
        dim_x=dim_xy,
        dim_y=dim_xy,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.05,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=1e-2,
        edge_ke=0.01,
        edge_kd=1e-2,
        particle_radius=0.01,
    )
    center = dim_xy // 2
    particle_idx = particle_start + center * (dim_xy + 1) + center
    builder.color()
    model = builder.finalize()
    return model, rigid_start, rigid_end, particle_idx


def _make_solver(
    model: newton.Model,
    rigid_start: int,
    rigid_end: int,
    admm_iters: int = 5,
    rho: float = 50.0,
    gamma: float = 0.0,
    baumgarte: float = 0.1,
):
    """Standard MuJoCo/VBD ADMM configuration used across tests."""
    mjc_ids = wp.array(list(range(rigid_start, rigid_end)), dtype=int)
    vbd_ids = wp.array(
        [i for i in range(model.body_count) if i < rigid_start or i >= rigid_end],
        dtype=int,
    )
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="mjc",
                solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=20),
                bodies=[int(i) for i in mjc_ids.numpy()],
                joints=list(range(model.joint_count)),
            ),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=5, rigid_compliant_alm=True),
                bodies=[int(i) for i in vbd_ids.numpy()],
                particles=list(range(model.particle_count)),
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=admm_iters,
            rho=rho,
            gamma=gamma,
            baumgarte=baumgarte,
        ),
    )


def _run(solver, model: newton.Model, n_steps: int = 30, dt: float = 1.0 / 60.0):
    """Run ``n_steps`` of simulation and return (body_q, particle_q)."""
    state_0 = model.state()
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(n_steps):
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.particle_q.numpy().copy()


def _build_two_particle_scene() -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle(pos=(-0.5, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.add_particle(pos=(0.5, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.color()
    return builder.finalize(device="cpu")


def _build_two_particle_contact_scene(
    gap: float = -0.1,
    vel_a: tuple[float, float, float] = (0.0, 0.0, 0.0),
    vel_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.05,
) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle(pos=(gap, 0.0, 0.0), vel=vel_a, mass=1.0, radius=radius)
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=vel_b, mass=1.0, radius=radius)
    builder.color()
    return builder.finalize(device="cpu")


def _run_particles(solver, model: newton.Model, n_steps: int = 5, dt: float = 1.0 / 60.0):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.particle_q.numpy().copy()


def _make_vbd_xpbd_particle_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda v: SolverVBD(model=v, iterations=2),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="xpbd",
                solver=lambda v: SolverXPBD(model=v, iterations=2),
                particles=[1],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=8,
            rho=20.0,
            baumgarte=0.2,
        ),
    )


def _make_semi_particle_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="a",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[0],
            ),
            SolverCoupled.Entry(
                name="b",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[1],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _build_body_particle_contact_scene() -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(
        xform=wp.transform(p=wp.vec3(-0.1, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_body_particle_attachment_scene(enabled: bool = True) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    particle = builder.add_particle(pos=(0.3, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.0)
    builder.add_attachment_particle_body(
        particle,
        body,
        stiffness=500.0,
        enabled=enabled,
    )
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_two_world_body_particle_attachment_scene() -> newton.Model:
    """Build two worlds with one rigid-particle attachment each."""
    world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = world.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    particle = world.add_particle(pos=(0.3, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.0)
    world.add_attachment_particle_body(particle, body, stiffness=500.0)
    world.color()

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_world(world)
    builder.add_world(world)
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    return model


def _build_two_body_contact_scene(gap: float = -0.1) -> newton.Model:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(
        xform=wp.transform(p=wp.vec3(gap, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
    )
    builder.color()
    return builder.finalize(device="cpu")


def _build_collision_contact_scene() -> tuple[newton.Model, int, int, int]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    tray_body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        mass=0.1,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )

    tray_cfg = newton.ModelBuilder.ShapeConfig()
    tray_cfg.has_shape_collision = False
    tray_cfg.has_particle_collision = True
    tray_shape = builder.add_shape_box(
        tray_body,
        xform=wp.transform(p=wp.vec3(0.0, 0.0, -0.025), q=wp.quat_identity()),
        hx=0.1,
        hy=0.1,
        hz=0.025,
        cfg=tray_cfg,
    )
    particle = builder.add_particle(
        pos=(0.0, 0.0, 0.12),
        vel=(0.0, 0.0, -0.5),
        mass=0.025,
        radius=0.025,
    )
    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    return model, particle, tray_body, tray_shape


def _run_body_particle(solver, model: newton.Model, n_steps: int = 4, dt: float = 1.0 / 60.0):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.particle_q.numpy().copy()


def _run_bodies(
    solver,
    model: newton.Model,
    n_steps: int = 4,
    dt: float = 1.0 / 60.0,
    body_qd: np.ndarray | None = None,
):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    if body_qd is not None:
        state_0.body_qd = wp.array(body_qd, dtype=wp.spatial_vector, device=model.device)

    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    return state_0.body_q.numpy().copy(), state_0.body_qd.numpy().copy()


def _make_semi_body_particle_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="body",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[0],
            ),
            SolverCoupled.Entry(
                name="particle",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=[0],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _make_semi_body_body_solver(model: newton.Model):
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="a",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[0],
            ),
            SolverCoupled.Entry(
                name="b",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[1],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=10,
            rho=30.0,
            baumgarte=0.5,
        ),
    )


def _build_inclined_plane_particle_box_scene(
    angle: float,
    *,
    particle_radius: float = 0.025,
    box_half_extent: float = 0.06,
    penetration: float = 0.002,
) -> tuple[newton.Model, int, int, list[int]]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    plane_cfg = newton.ModelBuilder.ShapeConfig()
    plane_cfg.has_shape_collision = False
    plane_cfg.has_particle_collision = True
    plane_shape = builder.add_shape_plane(
        body=plane_body,
        xform=wp.transform_identity(),
        width=2.0,
        length=2.0,
        cfg=plane_cfg,
    )

    n = np.array([math.sin(angle), 0.0, math.cos(angle)], dtype=np.float32)
    tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)
    binormal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    center = (particle_radius - penetration) * n

    particle_ids = []
    for tangent_sign in (-1.0, 1.0):
        for binormal_sign in (-1.0, 1.0):
            pos = center + tangent_sign * box_half_extent * tangent + binormal_sign * box_half_extent * binormal
            particle_ids.append(
                builder.add_particle(
                    pos=tuple(float(x) for x in pos),
                    vel=(0.0, 0.0, 0.0),
                    mass=0.25,
                    radius=particle_radius,
                )
            )

    builder.color()
    model = builder.finalize(device="cpu")
    model.particle_grid = None
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_kf = 0.0
    model.soft_contact_mu = 0.0
    return model, plane_body, plane_shape, particle_ids


def _make_admm_inclined_plane_particle_box_solver(
    model: newton.Model,
    plane_body: int,
    particle_ids: list[int],
    angle: float,
    friction: float,
) -> SolverCoupledADMM:
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="plane",
                solver=lambda v: _KinematicAdmmPlaneSolver(model=v, angle=angle),
                bodies=[plane_body],
            ),
            SolverCoupled.Entry(
                name="box",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                particles=particle_ids,
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=18,
            # Migrate the legacy tuning at the reference substep of 1/360 s.
            rho=50.0 / 360.0,
            gamma=0.0,
            baumgarte=0.1,
            contact_pairs=[
                SolverCoupledADMM.ContactPair(
                    source="plane",
                    destination="box",
                )
            ],
        ),
    )


def _run_inclined_plane_particle_box(
    angle: float,
    friction: float,
    *,
    steps: int = 120,
    dt: float = 1.0 / 360.0,
) -> tuple[float, float, int]:
    model, plane_body, _, particle_ids = _build_inclined_plane_particle_box_scene(angle)
    # ADMM derives friction from material properties; set both sides so the
    # geometric-mean combine reduces to the requested coefficient.
    model.particle_mu = float(friction)
    model.shape_material_mu = wp.full(model.shape_count, float(friction), dtype=wp.float32, device=model.device)
    solver = _make_admm_inclined_plane_particle_box_solver(
        model,
        plane_body,
        particle_ids,
        angle,
        friction,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    initial_com = np.mean(state_0.particle_q.numpy()[particle_ids], axis=0)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0

    final_q = state_0.particle_q.numpy()[particle_ids]
    final_qd = state_0.particle_qd.numpy()[particle_ids]
    final_com = np.mean(final_q, axis=0)
    final_vel = np.mean(final_qd, axis=0)
    tangent = np.array([math.cos(angle), 0.0, -math.sin(angle)], dtype=np.float32)
    displacement = float(np.dot(final_com - initial_com, tangent))
    velocity = float(np.dot(final_vel, tangent))
    return displacement, velocity, solver.collision_contact_count_max


def _rotate_y_np(v: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]], dtype=np.float32)


def _build_inclined_plane_rigid_box_scene(
    angle: float,
    *,
    box_half_height: float = 0.08,
    penetration: float = 0.002,
) -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    local_center = np.array([0.0, 0.0, box_half_height - penetration], dtype=np.float32)
    box_center = _rotate_y_np(local_center, angle)
    box_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(float(box_center[0]), float(box_center[1]), float(box_center[2])),
            plane_q,
        ),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    builder.color()
    return builder.finalize(device="cpu"), plane_body, box_body


def _build_collision_inclined_plane_rigid_box_scene(
    angle: float,
    *,
    box_half_height: float = 0.08,
    penetration: float = 0.004,
) -> tuple[newton.Model, int, int]:
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    plane_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    plane_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), plane_q),
        mass=0.0,
        inertia=wp.mat33(),
        is_kinematic=True,
    )

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.has_shape_collision = True
    cfg.has_particle_collision = False
    cfg.density = 0.0
    builder.add_shape_box(
        plane_body,
        xform=wp.transform(wp.vec3(1.0, 0.0, -0.025), wp.quat_identity()),
        hx=3.0,
        hy=0.4,
        hz=0.025,
        cfg=cfg,
    )

    local_center = np.array([0.0, 0.0, box_half_height - penetration], dtype=np.float32)
    box_center = _rotate_y_np(local_center, angle)
    box_body = builder.add_body(
        xform=wp.transform(
            wp.vec3(float(box_center[0]), float(box_center[1]), float(box_center[2])),
            plane_q,
        ),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    builder.add_shape_box(
        box_body,
        hx=0.08,
        hy=0.08,
        hz=box_half_height,
        cfg=cfg,
    )
    builder.color()
    return builder.finalize(device="cpu"), plane_body, box_body


def _make_collision_admm_inclined_plane_rigid_box_solver(
    model: newton.Model,
    plane_body: int,
    box_body: int,
    angle: float,
    friction: float,
    *,
    rigid_contact_matching: str = "disabled",
    contact_matching_pos_threshold: float | None = None,
    contact_matching_normal_dot_threshold: float | None = None,
    contact_matching_force_scale: float = 1.0,
) -> SolverCoupledADMM:
    del friction
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="plane",
                solver=lambda v: _KinematicAdmmPlaneSolver(model=v, angle=angle),
                bodies=[plane_body],
            ),
            SolverCoupled.Entry(
                name="box",
                solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                bodies=[box_body],
            ),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=30,
            # Migrate the legacy tuning at the reference substep of 1/360 s.
            rho=5.0 / 360.0,
            gamma=72.0,
            baumgarte=0.03,
            rigid_contact_matching=rigid_contact_matching,
            contact_matching_pos_threshold=contact_matching_pos_threshold,
            contact_matching_normal_dot_threshold=contact_matching_normal_dot_threshold,
            contact_matching_force_scale=contact_matching_force_scale,
            contact_pairs=[
                SolverCoupledADMM.ContactPair(
                    source="plane",
                    destination="box",
                )
            ],
        ),
    )


def _run_collision_inclined_plane_rigid_box(
    angle: float,
    friction: float,
    *,
    steps: int = 120,
    dt: float = 1.0 / 360.0,
) -> tuple[float, float, float, int]:
    model, plane_body, box_body = _build_collision_inclined_plane_rigid_box_scene(angle)
    model.shape_material_mu = wp.full(model.shape_count, float(friction), dtype=wp.float32, device=model.device)
    solver = _make_collision_admm_inclined_plane_rigid_box_solver(model, plane_body, box_body, angle, friction)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    initial_pos = state_0.body_q.numpy()[box_body, :3].copy()
    min_gap = math.inf
    normal = _rotate_y_np(np.array([0.0, 0.0, 1.0], dtype=np.float32), angle)
    tangent = _rotate_y_np(np.array([1.0, 0.0, 0.0], dtype=np.float32), angle)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None, dt=dt)
        state_0, state_1 = state_1, state_0
        center_gap = float(np.dot(normal, state_0.body_q.numpy()[box_body, :3]))
        min_gap = min(min_gap, center_gap - 0.08)

    final_pos = state_0.body_q.numpy()[box_body, :3]
    final_qd = state_0.body_qd.numpy()[box_body, :3]
    displacement = float(np.dot(final_pos - initial_pos, tangent))
    velocity = float(np.dot(final_qd, tangent))
    return displacement, velocity, min_gap, solver.collision_contact_count_max


class TestAdmmSmoke(unittest.TestCase):
    """End-to-end: construct, run, verify state advances without NaNs."""

    def test_rejects_invalid_timesteps(self):
        """Reject nonpositive and non-finite timesteps before advancing state."""
        model = _build_two_particle_scene()
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(name="a", solver=SolverSemiImplicit, particles=[0]),
                SolverCoupled.Entry(name="b", solver=SolverSemiImplicit, particles=[1]),
            ],
            SolverCoupledADMM.Config(),
        )
        for dt in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "dt"):
                solver.step(model.state(), model.state(), model.control(), contacts=None, dt=dt)

    def test_rejects_invalid_numerical_config(self):
        """Reject invalid ADMM coefficients and collision capacities before allocation."""
        model = _build_two_particle_scene()
        entries = [
            SolverCoupled.Entry(name="a", solver=SolverSemiImplicit, particles=[0]),
            SolverCoupled.Entry(name="b", solver=SolverSemiImplicit, particles=[1]),
        ]
        invalid_configs = (
            ({"iterations": 0}, "iterations"),
            ({"iterations": 1.5}, "iterations"),
            ({"rho": 0.0}, "rho"),
            ({"rho": float("nan")}, "rho"),
            ({"gamma": -1.0}, "gamma"),
            ({"baumgarte": float("inf")}, "baumgarte"),
            ({"joint_stiffness": -1.0}, "joint_stiffness"),
            ({"joint_proximal_mass_scale": 0.0}, "joint_proximal_mass_scale"),
            ({"contact_matching_normal_dot_threshold": 1.1}, "normal_dot_threshold"),
            ({"contact_max_triangle_pairs": 0}, "contact_max_triangle_pairs"),
            ({"contact_max_triangle_pairs": 1.5}, "contact_max_triangle_pairs"),
            ({"contact_max_triangle_pairs": True}, "contact_max_triangle_pairs"),
            ({"contact_reduction_hashtable_size_factor": 0.0}, "contact_reduction_hashtable_size_factor"),
            ({"contact_reduction_hashtable_size_factor": float("nan")}, "contact_reduction_hashtable_size_factor"),
            ({"contact_reduction_hashtable_size_factor": float("inf")}, "contact_reduction_hashtable_size_factor"),
        )
        for kwargs, message in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                SolverCoupledADMM(model=model, entries=entries, coupling=SolverCoupledADMM.Config(**kwargs))

    def test_contact_capacity_matching_limit(self):
        """Allow larger triangle-pair capacities only when contact matching is disabled."""
        model = _build_two_particle_scene()
        entries = [
            SolverCoupled.Entry(name="a", solver=SolverSemiImplicit, particles=[0]),
            SolverCoupled.Entry(name="b", solver=SolverSemiImplicit, particles=[1]),
        ]
        for matching in ("disabled", "latest", "sticky"):
            for capacity in (None, 2**20 - 1, 2**20, 2**20 + 1):
                with self.subTest(matching=matching, capacity=capacity):
                    config = SolverCoupledADMM.Config(
                        rigid_contact_matching=matching, contact_max_triangle_pairs=capacity
                    )
                    if matching != "disabled" and capacity is not None and capacity >= 2**20:
                        with self.assertRaisesRegex(ValueError, r"contact_max_triangle_pairs.*2\*\*20"):
                            SolverCoupledADMM(model, entries, config)
                    else:
                        SolverCoupledADMM(model, entries, config)

    def test_construct_and_step_no_attachments(self):
        model, rs, re, _ = _build_cloth_rigid_scene()
        solver = _make_solver(model, rs, re, admm_iters=1)

        body_q, particle_q = _run(solver, model, n_steps=10)
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertTrue(np.all(np.isfinite(particle_q)))

    def test_admm_iters_idempotent_with_no_coupling(self):
        """With gamma=0 and no attachments the iteration count should not
        change the result (no coupling = idempotent outer loop)."""
        model_a, rs, re, _ = _build_cloth_rigid_scene()
        solver_a = _make_solver(model_a, rs, re, admm_iters=1, gamma=0.0)
        body_a, part_a = _run(solver_a, model_a, n_steps=5)

        model_b, rs, re, _ = _build_cloth_rigid_scene()
        solver_b = _make_solver(model_b, rs, re, admm_iters=4, gamma=0.0)
        body_b, part_b = _run(solver_b, model_b, n_steps=5)

        np.testing.assert_allclose(body_a, body_b, atol=1e-6)
        np.testing.assert_allclose(part_a, part_b, atol=1e-6)


class TestAdmmProximal(unittest.TestCase):
    """Proximal terms affect constrained DOFs only."""

    def test_default_proximal_stabilizes_stiff_attachment(self):
        """Converge to a shared velocity for a nearly rigid attachment."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        particle = builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), mass=1.0, radius=0.0)
        builder.add_attachment_particle_body(particle, body, stiffness=1.0e12)
        builder.color()
        model = builder.finalize(device="cpu")
        model.particle_grid = None
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    "body",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    bodies=[body],
                ),
                SolverCoupled.Entry(
                    "particle",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    particles=[particle],
                ),
            ],
            SolverCoupledADMM.Config(iterations=40),
        )
        state_in, state_out = model.state(), model.state()
        solver.step(state_in, state_out, model.control(), contacts=None, dt=1.0 / 120.0)

        # Equal masses share the initial momentum in the hard-constraint limit.
        np.testing.assert_allclose(state_out.body_qd.numpy()[body, :3], (0.5, 0.0, 0.0), atol=1.0e-4)
        np.testing.assert_allclose(state_out.particle_qd.numpy()[particle], (0.5, 0.0, 0.0), atol=1.0e-4)

    def test_gamma_does_not_change_unconstrained_freefall(self):
        # Place the rigid body high so it stays in free-fall across the
        # window; with no ADMM constraints, gamma should not alter the result.
        model_ref, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_ref = _make_solver(model_ref, rs, re, admm_iters=3, gamma=0.0)
        body_ref, part_ref = _run(solver_ref, model_ref, n_steps=5)

        model_g, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_g = _make_solver(model_g, rs, re, admm_iters=3, gamma=5.0)
        body_g, part_g = _run(solver_g, model_g, n_steps=5)

        np.testing.assert_allclose(body_ref, body_g, atol=1.0e-6)
        np.testing.assert_allclose(part_ref, part_g, atol=1.0e-6)
        self.assertTrue(np.all(np.isfinite(body_g)))
        self.assertTrue(np.all(np.isfinite(part_g)))


class TestAdmmGraphCapture(unittest.TestCase):
    """CUDA graph-capture smoke tests for dynamic proximal refresh."""

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_xpbd_vbd_contact_proximal_refresh_is_graph_capturable(self):
        device = "cuda:0"
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.default_shape_cfg.density = 1000.0
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=body_a, hx=0.05, hy=0.05, hz=0.05)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=body_b, hx=0.05, hy=0.05, hz=0.05)
        builder.color()
        model = builder.finalize(device=device)
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="xpbd",
                    solver=lambda v: SolverXPBD(model=v, iterations=1),
                    bodies=[body_a],
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda v: SolverVBD(model=v, iterations=1, rigid_compliant_alm=True),
                    bodies=[body_b],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=1,
                gamma=1.0,
                contact_pairs=[SolverCoupledADMM.ContactPair(source="xpbd", destination="vbd")],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)
        state_0, state_1 = state_1, state_0
        wp.synchronize_device(device)

        with wp.ScopedCapture(device=device) as capture:
            solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)

        self.assertIsNotNone(capture.graph)
        wp.capture_launch(capture.graph)
        q = state_1.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(q)))


def _build_split_ball_pair(tree: bool) -> newton.Model:
    """Build two 1 kg bodies joined by a satisfied ball joint, 0.3 m apart along x, without gravity.

    With ``tree`` the ball is the child's articulation joint below the parent's FREE joint; otherwise
    each body has its own FREE joint and the ball closes a loop.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(np.eye(3) * 0.01)
    parent = builder.add_link(mass=1.0, inertia=inertia)
    child = builder.add_link(xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
    root = builder.add_joint_free(parent)
    if tree:
        ball = builder.add_joint_ball(
            parent, child, child_xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity())
        )
        builder.add_articulation([root, ball])
    else:
        builder.add_articulation([root])
        builder.add_articulation([builder.add_joint_free(child)])
        builder.add_joint_ball(parent, child, child_xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()))
    builder.color()
    return builder.finalize(device="cpu")


def _make_split_ball_solver(model: newton.Model, make_solver, tree: bool, **coupling_kwargs) -> SolverCoupledADMM:
    """Split the ball pair across two entries; a tree-modeled child owns no joint and gets a joint proxy."""
    return SolverCoupledADMM(
        model=model,
        entries=[
            SolverCoupled.Entry(name="parent", solver=make_solver, bodies=[0], joints=[0]),
            SolverCoupled.Entry(name="child", solver=make_solver, bodies=[1], joints=[] if tree else [1]),
        ],
        coupling=SolverCoupledADMM.Config(
            iterations=50,
            rho=1.0 / 3.0,
            gamma=0.0,
            baumgarte=0.5,
            joint_proximal_bodies=tree,
            **coupling_kwargs,
        ),
    )


def _push_split_ball(model: newton.Model, solver, body_f: tuple[float, ...], n_steps: int = 20) -> tuple:
    """Apply a constant wrench to one body (index 0 of ``body_f``) and return the final body state."""
    state_0, state_1 = model.state(), model.state()
    wrench = np.zeros((model.body_count, 6), dtype=np.float32)
    wrench[int(body_f[0])] = body_f[1:]
    for _ in range(n_steps):
        state_0.body_f.assign(wrench)
        solver.step(state_0, state_1, None, None, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0
    return state_0.body_q.numpy(), state_0.body_qd.numpy()


class TestAdmmModelJointInterface(unittest.TestCase):
    """Cross-solver model joints are converted to ADMM attachments."""

    def test_revolute_axis_frames_handle_antiparallel_x_axis(self):
        identity_row = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        axis = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        frame_child, frame_parent = SolverCoupledADMM._revolute_axis_frames_from_rows(
            identity_row,
            identity_row,
            axis,
        )

        for frame in (frame_child, frame_parent):
            rotation = wp.transform_get_rotation(frame)
            rotated_axis = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
            np.testing.assert_allclose(np.asarray(rotated_axis), axis, atol=1.0e-6)

    def _build_two_body_joint_scene(
        self,
        joint_type: str = "ball",
        *,
        friction: float = 0.0,
    ) -> tuple[newton.Model, int, int, int]:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        child = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.3, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 0.01),
        )
        if joint_type == "ball":
            joint = builder.add_joint_ball(
                parent=parent,
                child=child,
                friction=friction,
                collision_filter_parent=False,
            )
        elif joint_type == "fixed":
            joint = builder.add_joint_fixed(parent=parent, child=child, collision_filter_parent=False)
        elif joint_type == "revolute":
            joint = builder.add_joint_revolute(
                parent=parent, child=child, friction=friction, collision_filter_parent=False
            )
        else:
            raise ValueError(joint_type)
        builder.color()
        return builder.finalize(device="cpu"), parent, child, joint

    def _make_two_body_joint_solver(self, model: newton.Model, parent: int, child: int, **coupling_kwargs):
        return SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="parent",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[parent],
                ),
                SolverCoupled.Entry(
                    name="child",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[child],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=12,
                rho=40.0,
                baumgarte=0.5,
                joint_stiffness=500.0,
                joint_angular_stiffness=50.0,
                **coupling_kwargs,
            ),
        )

    def test_ball_joint_attachment_closes_anchor_gap(self):
        model, parent, child, _ = self._build_two_body_joint_scene("ball")
        solver = self._make_two_body_joint_solver(model, parent, child)
        initial_gap = abs(model.state().body_q.numpy()[child, 0] - model.state().body_q.numpy()[parent, 0])

        body_q, _ = _run_bodies(solver, model, n_steps=8, dt=1.0 / 120.0)
        final_gap = abs(body_q[child, 0] - body_q[parent, 0])

        self.assertLess(final_gap, 0.5 * initial_gap)

    def test_split_reduced_articulation_child_gets_builder_free_root_joint(self):
        """Move a split articulation's child through a builder-equivalent FREE root joint.

        The child entry owns no joint, so its compact view synthesizes the
        FREE joint ``ModelBuilder.add_body()`` would create, through which
        the reduced-coordinate child entry moves the body.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="dof_gain",
                dtype=wp.float32,
                frequency=newton.Model.AttributeFrequency.JOINT_DOF,
                namespace="test",
                default=2.5,
            )
        )
        inertia = wp.mat33(np.eye(3) * 0.01)
        parent = builder.add_link(mass=1.0, inertia=inertia)
        child = builder.add_link(
            xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia, label="child"
        )
        root = builder.add_joint_free(parent)
        ball = builder.add_joint_ball(
            parent, child, collision_filter_parent=False, custom_attributes={"test:dof_gain": 7.0}
        )
        builder.add_articulation([root, ball])
        model = builder.finalize(device="cpu")

        reference_builder = newton.ModelBuilder()
        reference_builder.add_body()
        reference = reference_builder.finalize(device="cpu")

        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(name="parent", solver=SolverFeatherstone, bodies=[parent], joints=[root]),
                SolverCoupled.Entry(name="child", solver=SolverFeatherstone, bodies=[child]),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=12,
                rho=1.0 / 3.0,
                gamma=0.0,
                baumgarte=0.5,
                joint_stiffness=6.0e4,
                joint_angular_stiffness=6.0e3,
                joint_proximal_bodies=False,
            ),
        )

        view = solver.view("child")
        self.assertEqual(view.joint_count, 1)
        np.testing.assert_array_equal(view.joint_parent.numpy(), [-1])
        np.testing.assert_array_equal(view.joint_child.numpy(), [0])
        np.testing.assert_array_equal(view.joint_ancestor.numpy(), [-1])
        np.testing.assert_array_equal(view.articulation_start.numpy(), [0, 1])
        np.testing.assert_array_equal(view.articulation_end.numpy(), [1])
        np.testing.assert_allclose(view.joint_q.numpy(), model.body_q.numpy()[child], atol=1.0e-7)
        self.assertEqual(view.joint_label, ["child_free_joint"])
        for name in (
            "joint_type",
            "joint_X_p",
            "joint_X_c",
            "joint_dof_dim",
            "joint_enabled",
            "joint_target_q",
            "joint_axis",
            "joint_target_mode",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "joint_armature",
            "joint_effort_limit",
            "joint_velocity_limit",
            "joint_friction",
        ):
            np.testing.assert_array_equal(getattr(view, name).numpy(), getattr(reference, name).numpy(), err_msg=name)
        # Custom attributes take their registered default, like a builder-created FREE joint.
        np.testing.assert_array_equal(view.test.dof_gain.numpy(), np.full(6, 2.5, dtype=np.float32))

        # Step from the authored body poses; forward kinematics of the
        # identity-frame ball joint would place both bodies at the origin.
        state_0, state_1 = model.state(), model.state()
        for _ in range(8):
            solver.step(state_0, state_1, None, None, 1.0 / 120.0)
            state_0, state_1 = state_1, state_0
        body_q = state_0.body_q.numpy()
        initial_gap = abs(model.body_q.numpy()[child, 0] - model.body_q.numpy()[parent, 0])

        # Equal masses meet near the midpoint; a welded child would drag the parent onto x = 0.3.
        self.assertLess(abs(body_q[child, 0] - body_q[parent, 0]), 0.05 * initial_gap)
        self.assertLess(body_q[child, 0], 0.25)
        self.assertGreater(body_q[parent, 0], 0.05)

    def test_split_articulation_proximal_targets_follow_view_articulations(self):
        """Evaluate proximal targets through each synthesized FREE root articulation.

        The chain A-B-C-D alternates entries, so the ``outer`` view holds A's
        FREE root in one articulation and C's synthesized FREE root with the
        C-D revolute in another. D moves with C's FREE DOFs, not A's.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        inertia = wp.mat33(np.eye(3) * 0.01)
        a, b, c, d = (
            builder.add_link(
                xform=wp.transform(wp.vec3(0.3 * i, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia
            )
            for i in range(4)
        )
        half = wp.transform(wp.vec3(0.15, 0.0, 0.0), wp.quat_identity())
        half_back = wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity())
        root = builder.add_joint_free(a)
        ab = builder.add_joint_ball(a, b, parent_xform=half, child_xform=half_back)
        bc = builder.add_joint_ball(b, c, parent_xform=half, child_xform=half_back)
        cd = builder.add_joint_revolute(c, d, axis=(0.0, 1.0, 0.0), parent_xform=half, child_xform=half_back)
        builder.add_articulation([root, ab, bc, cd])
        model = builder.finalize(device="cpu")

        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(name="outer", solver=SolverFeatherstone, bodies=[a, c, d], joints=[root, cd]),
                SolverCoupled.Entry(name="middle", solver=SolverFeatherstone, bodies=[b]),
            ],
            coupling=SolverCoupledADMM.Config(iterations=4, rho=40.0, gamma=0.5, joint_proximal_bodies=False),
        )

        entry = solver._entries["outer"]
        buf = solver._admm_buffers["outer"]
        # View DOFs: A's FREE root 0-5, C's synthesized FREE root 6-11, C-D revolute 12.
        # A unit velocity on a root DOF moves exactly the bodies of that articulation.
        for dof, moving_bodies in ((0, {a}), (6, {c, d})):
            qd = wp.zeros(entry.view.joint_dof_count, dtype=float, device=model.device)
            values = qd.numpy()
            values[dof] = 1.0
            qd.assign(values)
            velocity = buf.reduced_gamma.target_velocity(entry.state_0.joint_q, qd).numpy()
            for local, body in enumerate(entry.body_local_to_global.numpy()):
                expected = np.zeros(6)
                expected[0] = float(int(body) in moving_bodies)
                np.testing.assert_allclose(velocity[local], expected, atol=1.0e-7)

        state_0, state_1 = model.state(), model.state()
        for _ in range(4):
            solver.step(state_0, state_1, None, None, 1.0 / 120.0)
            state_0, state_1 = state_1, state_0
        self.assertTrue(np.all(np.isfinite(state_0.body_q.numpy())))

    def test_joint_proxy_shape_collisions_disabled_after_shape_compaction(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        unrelated = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=unrelated, radius=0.05)
        parent = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=parent, radius=0.05)
        child = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_sphere(body=child, radius=0.05)
        builder.add_joint_ball(parent=parent, child=child, collision_filter_parent=False)
        builder.color()
        model = builder.finalize(device="cpu")

        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="parent",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[parent],
                ),
                SolverCoupled.Entry(
                    name="child",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[child],
                ),
            ],
            coupling=SolverCoupledADMM.Config(),
        )

        collision_mask = int(
            newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES | newton.ShapeFlags.HYDROELASTIC
        )
        proxy_flag = int(newton.BodyFlags.PROXY)
        for name, owned in (("parent", parent), ("child", child)):
            with self.subTest(name=name):
                view = solver.view(name)
                self.assertEqual(view.shape_count, model.shape_count)
                shape_body = view.shape_body.numpy()
                body_global = solver._entries[name].body_local_to_global.numpy()
                shape_flags = view.shape_flags.numpy()
                proxy_shapes = [
                    shape_id
                    for shape_id, body_id in enumerate(shape_body)
                    if int(body_id) >= 0 and int(body_global[int(body_id)]) != owned
                ]
                body_flags = view.body_flags.numpy()
                for body_id, global_id in enumerate(body_global):
                    self.assertEqual(bool(body_flags[body_id] & proxy_flag), int(global_id) != owned)
                owned_shapes = [
                    shape_id
                    for shape_id, body_id in enumerate(shape_body)
                    if int(body_id) >= 0 and shape_id not in proxy_shapes
                ]
                hidden_shapes = [shape_id for shape_id, body_id in enumerate(shape_body) if int(body_id) < 0]

                self.assertEqual(len(proxy_shapes), 1)
                self.assertEqual(len(owned_shapes), 1)
                self.assertEqual(len(hidden_shapes), 1)
                self.assertEqual(int(shape_flags[proxy_shapes[0]]), 0)
                self.assertNotEqual(int(shape_flags[owned_shapes[0]]) & collision_mask, 0)
                self.assertEqual(int(shape_flags[hidden_shapes[0]]) & collision_mask, 0)

    def test_rejects_cross_solver_joint_owned_by_subsolver(self):
        model, parent, child, joint = self._build_two_body_joint_scene("ball")
        with self.assertRaisesRegex(ValueError, "must not be owned"):
            SolverCoupledADMM(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="parent",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[parent],
                        joints=[joint],
                    ),
                    SolverCoupled.Entry(
                        name="child",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[child],
                    ),
                ],
                coupling=SolverCoupledADMM.Config(),
            )


def _semi_implicit(view):
    return SolverSemiImplicit(model=view, enable_tri_contact=False)


def _build_off_center_ball_pair(
    *, kamino: bool = False, joint_rho: float | None = None, parallel: bool = False
) -> newton.Model:
    """Build two free bodies joined by a ball joint anchored off the first body's COM."""
    builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
    if kamino:
        SolverKamino.register_custom_attributes(builder)
    joint_attributes = {}
    if joint_rho is not None:
        SolverCoupledADMM.register_custom_attributes(builder)
        joint_attributes = {"custom_attributes": {"coupling:joint_rho": joint_rho}}
    for mass in (2.0, 5.0):
        builder.add_body(mass=mass, inertia=wp.mat33(np.diag([1.0, 2.0, 3.0])))
    builder.add_joint_ball(0, 1, parent_xform=wp.transform((0.3, -0.4, 0.2), wp.quat_identity()), **joint_attributes)
    if parallel:
        anchor = wp.transform((-0.5, 0.1, 0.0), wp.quat_identity())
        builder.add_joint_ball(0, 1, parent_xform=anchor, child_xform=anchor)
    builder.color()
    return builder.finalize(device="cpu")


def _step_off_center_ball_pair(
    model: newton.Model, factories, *, in_place: bool = False, substeps: int = 1, **config
) -> tuple[SolverCoupledADMM, np.ndarray]:
    """Split the pair across two entries and take one coupled step from spinning initial velocities."""

    def configure(view: ModelView) -> None:
        # Solvers such as Kamino rewrite body and joint frames in place; keep those writes local.
        for name in ("body_com", "body_inertia", "joint_X_p", "joint_X_c"):
            setattr(view, name, wp.clone(getattr(view, name)))

    solver = SolverCoupledADMM(
        model,
        [
            SolverCoupled.Entry(
                name=str(body),
                bodies=[body],
                joints=[body],
                solver=factories[body],
                in_place=in_place,
                substeps=substeps,
                configure_view=configure,
            )
            for body in range(2)
        ],
        SolverCoupledADMM.Config(**{"gamma": 0.0, "joint_stiffness": 5000.0, **config}),
    )
    state, out = model.state(), model.state()
    initial = np.array([[1.3, 0.2, -0.3, 0.7, -0.4, 0.3], [-0.7, -0.1, 0.2, -0.2, 0.1, 0.3]], dtype=np.float32)
    state.body_qd.assign(initial)
    joint_qd = state.joint_qd.numpy()
    joint_qd[:12] = initial.ravel()
    state.joint_qd.assign(joint_qd)
    solver.step(state, out, None, None, 0.02)
    return solver, out.body_qd.numpy()


def _build_split_four_bar(world_count: int) -> tuple[newton.Model, list[tuple[str, list[int], list[int]]]]:
    """Build a closed four-bar of free links on a fixed base, with alternate links in two entries."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))

    def bar(start: tuple[float, ...], end: tuple[float, ...], mass: float) -> tuple[int, wp.transform, tuple, tuple]:
        start, end = np.asarray(start), np.asarray(end)
        length = float(np.linalg.norm(end - start))
        direction = (end - start) / length
        quat = np.array([*np.cross([1.0, 0.0, 0.0], direction), 1.0 + direction[0]])
        pose = wp.transform(wp.vec3(*(0.5 * (start + end))), wp.quat(*(quat / np.linalg.norm(quat))))
        body = template.add_link(xform=pose, mass=mass, inertia=wp.mat33(np.eye(3) * mass * length * length / 12.0))
        return body, pose, (-0.5 * length, 0.0, 0.0), (0.5 * length, 0.0, 0.0)

    left_base, right_base, left_top, right_top = (
        (-0.48, 0.0, 1.05),
        (0.48, 0.0, 1.05),
        (-0.30, 0.0, 1.55),
        (0.34, 0.0, 1.46),
    )
    ground, ground_pose, ground_left, ground_right = bar(left_base, right_base, 1.0)
    base = template.add_joint_fixed(-1, ground, parent_xform=ground_pose)
    template.add_articulation([base])
    crank, _, crank_base, crank_top = bar(left_base, left_top, 0.35)
    coupler, _, coupler_left, coupler_right = bar(left_top, right_top, 0.42)
    rocker, _, rocker_base, rocker_top = bar(right_base, right_top, 0.36)
    roots = []
    for link in (crank, coupler, rocker):
        roots.append(template.add_joint_free(link))
        template.add_articulation([roots[-1]])
    for parent, parent_point, child, child_point in (
        (ground, ground_left, crank, crank_base),
        (crank, crank_top, coupler, coupler_left),
        (coupler, coupler_right, rocker, rocker_top),
        (rocker, rocker_base, ground, ground_right),
    ):
        template.add_joint_revolute(
            parent,
            child,
            parent_xform=wp.transform(parent_point, wp.quat_identity()),
            child_xform=wp.transform(child_point, wp.quat_identity()),
            axis=wp.vec3(0.0, 1.0, 0.0),
        )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.replicate(template, world_count=world_count)
    builder.color()
    model = builder.finalize(device="cpu")

    def per_world(ids: list[int], stride: int) -> list[int]:
        return [world * stride + index for world in range(world_count) for index in ids]

    bodies, joints = template.body_count, template.joint_count
    return model, [
        ("a", per_world([crank, rocker], bodies), per_world([roots[0], roots[2]], joints)),
        ("b", per_world([ground, coupler], bodies), per_world([base, roots[1]], joints)),
    ]


def _build_loop_before_tree_joint() -> tuple[newton.Model, list[SolverCoupled.Entry]]:
    """Build a body proxied by a loop joint whose index precedes its tree joint into the same entry."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    inertia = wp.mat33(np.eye(3) * 0.01)
    a0, b0, a1 = (
        builder.add_link(xform=wp.transform((x, 0.0, 1.0), wp.quat_identity()), mass=1.0, inertia=inertia)
        for x in (0.0, 0.3, 0.6)
    )
    hinge = {
        "parent_xform": wp.transform((0.15, 0.0, 0.0), wp.quat_identity()),
        "child_xform": wp.transform((-0.15, 0.0, 0.0), wp.quat_identity()),
        "axis": wp.vec3(0.0, 1.0, 0.0),
    }
    builder.add_joint_revolute(b0, a1, **hinge)
    root = builder.add_joint_free(a0)
    builder.add_articulation([root, builder.add_joint_revolute(a0, b0, **hinge)])
    root1 = builder.add_joint_free(a1)
    builder.add_articulation([root1])
    model = builder.finalize(device="cpu")
    return model, [
        SolverCoupled.Entry(name="a", solver=SolverFeatherstone, bodies=[a0, a1], joints=[root, root1]),
        SolverCoupled.Entry(name="b", solver=SolverFeatherstone, bodies=[b0], joints=[]),
    ]


class TestAdmmJointProxies(unittest.TestCase):
    """Joint proxies only change ADMM convergence, never the converged solution."""

    def test_tree_joint_proxy_ignores_joint_order(self):
        """Give a tree joint and a loop joint into one entry each a proxy row, whichever comes first."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, entries = _build_loop_before_tree_joint()
            solver = SolverCoupledADMM(model, entries, SolverCoupledADMM.Config())
        for group in (*solver._admm_rr_groups, *solver._admm_rr_revolute_angular_groups):
            self.assertTrue(np.all(group.proxy_slots_a.numpy() >= 0))
            self.assertTrue(np.all(group.proxy_slots_b.numpy() >= 0))
        self.assertEqual(sum("_copy_" in label for label in solver.view("a").body_label), 1)

    def test_proxy_mass_keeps_solution(self):
        """Match the proxy-free solution for any proxy mass, attachment stiffness, and coordinate type."""
        push = (0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        solvers = {
            "featherstone": SolverFeatherstone,
            "semi_implicit": lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
        }
        for name, make_solver in solvers.items():
            for stiffness in (6.0e4, 6.0e8):
                stiffness_kwargs = {"joint_stiffness": stiffness, "joint_angular_stiffness": 0.1 * stiffness}
                model = _build_split_ball_pair(tree=False)
                reference, _ = _push_split_ball(
                    model, _make_split_ball_solver(model, make_solver, tree=False, **stiffness_kwargs), push
                )
                for mass_scale in (0.5, 2.0):
                    with self.subTest(solver=name, stiffness=stiffness, mass_scale=mass_scale):
                        model = _build_split_ball_pair(tree=True)
                        solver = _make_split_ball_solver(
                            model, make_solver, tree=True, joint_proximal_mass_scale=mass_scale, **stiffness_kwargs
                        )
                        body_q, body_qd = _push_split_ball(model, solver, push)
                        np.testing.assert_allclose(body_q[:, :3], reference[:, :3], atol=2.0e-4)
                        # A unit force over 20 steps gives the pair 1/6 N s of momentum.
                        self.assertAlmostEqual(float(body_qd[0, 0] + body_qd[1, 0]), 20.0 / 120.0, delta=1.0e-3)

    def test_proxy_joints_exert_no_drive_or_control(self):
        """Keep the drive, armature, and joint control of proxied joints out of the coupled solution."""
        joint_f = np.zeros(7, dtype=np.float32)
        joint_f[0] = 1.0  # the parent's FREE joint, a proxy path in the child's view
        wrench = np.zeros((2, 6), dtype=np.float32)
        wrench[1, 5] = 0.01  # turns the hinge, which a leftover drive would resist

        def run(model: newton.Model, solver):
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            control.joint_f.assign(joint_f)
            for _ in range(20):
                state_0.body_f.assign(wrench)
                solver.step(state_0, state_1, control, None, 1.0 / 120.0)
                state_0, state_1 = state_1, state_0
            return state_0.body_q.numpy(), state_0.body_qd.numpy()

        def coupled(make_solver, tree: bool, stiffness: float):
            model = _build_split_hinge_pair(tree)
            return run(
                model,
                _make_split_ball_solver(
                    model, make_solver, tree=tree, joint_stiffness=stiffness, joint_angular_stiffness=0.1 * stiffness
                ),
            )

        # Nothing simulates a cross-solver joint's drive or armature, so the reference hinge has none.
        reference = _build_split_hinge_pair(True, driven=False)
        cases = (
            ("featherstone", run(reference, SolverFeatherstone(reference)), coupled(SolverFeatherstone, True, 6.0e8)),
            ("semi_implicit", coupled(_semi_implicit, False, 6.0e6), coupled(_semi_implicit, True, 6.0e6)),
        )
        for name, (expected_q, expected_qd), (body_q, body_qd) in cases:
            with self.subTest(solver=name):
                np.testing.assert_allclose(body_q, expected_q, atol=2.0e-4)
                np.testing.assert_allclose(body_qd, expected_qd, atol=2.0e-3)

    def test_proxy_is_a_single_free_rooted_link(self):
        """Keep one proxy body per cross-solver joint; a reduced view roots it with a FREE joint."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = -1
        joints = []
        for _ in range(3):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), com=wp.vec3(0.0, 0.0, 0.15))
            offset = 0.0 if parent < 0 else 0.3
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    body,
                    parent_xform=wp.transform(
                        wp.vec3(0.0, 0.0, offset), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)
                    ),
                    axis=newton.Axis.X,
                )
            )
            parent = body
        builder.add_articulation(joints)
        model = builder.finalize(device="cpu")

        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(name="base", solver=SolverFeatherstone, bodies=[0, 1], joints=[0, 1]),
                SolverCoupled.Entry(name="tip", solver=SolverFeatherstone, bodies=[2], joints=[]),
            ],
            coupling=SolverCoupledADMM.Config(),
        )
        # The tip sees only a copy of link 1, rooted by a FREE joint, above its own link through joint 2.
        tip = solver.view("tip")
        np.testing.assert_array_equal(solver._entries["tip"].body_local_to_global.numpy(), [1, 2])
        np.testing.assert_array_equal(
            tip.joint_type.numpy(), [int(newton.JointType.FREE), int(newton.JointType.REVOLUTE)]
        )
        np.testing.assert_array_equal(solver._entries["base"].body_local_to_global.numpy(), [0, 1, 2])

    def test_proxied_child_keeps_rotation(self):
        """Accumulate a reduced-coordinate child's rotation through its cross-solver joint across steps."""
        torque = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01)
        model = _build_split_ball_pair(tree=True)
        expected_q, expected_qd = _push_split_ball(model, SolverFeatherstone(model), torque)
        model = _build_split_ball_pair(tree=True)
        solver = _make_split_ball_solver(
            model,
            SolverFeatherstone,
            tree=True,
            joint_stiffness=6.0e8,
            joint_angular_stiffness=6.0e7,
        )
        body_q, body_qd = _push_split_ball(model, solver, torque)
        self.assertGreater(abs(float(expected_q[1, 5])), 1.0e-3)
        np.testing.assert_allclose(body_q[1, 3:], expected_q[1, 3:], atol=1.0e-4)
        np.testing.assert_allclose(body_qd[1, 3:], expected_qd[1, 3:], atol=1.0e-3)

    def test_closed_loop_proxies_cover_every_joint(self):
        """Give every joint row of a split four-bar a proxy through body copies, in every world."""
        model, entries = _build_split_four_bar(world_count=2)
        config = {
            "iterations": 60,
            "rho": 0.5,
            "gamma": 1.0,
            "joint_stiffness": 3.6e6,
            "joint_angular_stiffness": 3.6e6,
        }
        body_q = {}
        # The linearized update diverges on this stiff loop, so the converged answer is checked
        # for independence from the proxy mass.
        for mass_scale in (1.0, 2.0):
            solver = SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry(name=name, solver=SolverFeatherstone, bodies=bodies, joints=joints)
                    for name, bodies, joints in entries
                ],
                SolverCoupledADMM.Config(joint_proximal_mass_scale=mass_scale, **config),
            )
            for group in (*solver._admm_rr_groups, *solver._admm_rr_revolute_angular_groups):
                self.assertTrue(np.all(group.proxy_slots_a.numpy() >= 0))
                self.assertTrue(np.all(group.proxy_slots_b.numpy() >= 0))
            # Body copies keep their labels and PROXY flags through a body-property notification.
            for notify in (False, True):
                if notify:
                    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
                for name, _, _ in entries:
                    # Each view holds two linkage bodies joined to it twice, so two copies per world.
                    view = solver.view(name)
                    self.assertEqual(sum("_copy_" in label for label in view.body_label), 4)
                    body_flags = view.body_flags.numpy()
                    for label, flags in zip(view.body_label, body_flags, strict=True):
                        if "_copy_" in label:
                            self.assertTrue(flags & int(newton.BodyFlags.PROXY))
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            for _ in range(10):
                solver.step(state_0, state_1, None, None, 1.0 / 180.0)
                state_0, state_1 = state_1, state_0
            body_q[mass_scale] = state_0.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(body_q[1.0])))
        np.testing.assert_allclose(body_q[2.0], body_q[1.0], atol=1.0e-5)
        np.testing.assert_allclose(body_q[1.0][:4], body_q[1.0][4:], atol=1.0e-6)

    def test_kamino_lox_moves_joint_proxies(self):
        """Move PROXY-flagged joint proxies as dynamic bodies in a Kamino LOX view."""
        model = _build_off_center_ball_pair(kamino=True)

        def kamino(view: ModelView) -> SolverKamino:
            config = SolverKamino.Config(dynamics_solver="lox")
            config.use_collision_detector = False
            config.use_fk_solver = False
            return SolverKamino(model=view, config=config)

        factories = (kamino, SolverFeatherstone)
        solver, expected = _step_off_center_ball_pair(
            model, factories, iterations=400, rho=0.4, joint_proximal_bodies=False
        )
        solver, actual = _step_off_center_ball_pair(model, factories, iterations=64, rho=0.4)
        self.assertTrue(np.any(solver.view("0").body_flags.numpy() & int(newton.BodyFlags.PROXY)))
        # A frozen proxy errs by O(1).
        np.testing.assert_allclose(actual, expected, atol=1.0e-3)

    def test_proxy_inertia_uses_joint_rho(self):
        """Size each proxy from its joint's rho override."""
        model = _build_off_center_ball_pair(joint_rho=2.0)
        solver, _ = _step_off_center_ball_pair(model, (SolverFeatherstone, SolverFeatherstone), iterations=4, rho=0.5)
        group = solver._admm_rr_groups[0]
        self.assertEqual(group.parameters.rho, 2.0)
        w = float(group.W.numpy()[0])
        body_mass, body_inertia = model.body_mass.numpy(), model.body_inertia.numpy()
        for name, proxies in solver._admm_joint_proxies.items():
            np.testing.assert_allclose(proxies.mass.numpy(), [2.0 * w * w], rtol=1.0e-6)
            # Ball-joint proxies keep the proxied body's radii of gyration.
            proxied = solver._entries[name].body_local_to_global.numpy()[proxies.proxies.numpy()[0]]
            reference = 2.0 * w * w / body_mass[proxied] * body_inertia[proxied]
            np.testing.assert_allclose(proxies.inertia.numpy()[0], reference, rtol=1.0e-5, atol=1.0e-9)

    def test_in_place_entries_keep_proxy_fixed_point(self):
        """Target position-based proxies at the previous entry step's poses when entries step in place."""
        model = _build_off_center_ball_pair()
        factories = (lambda view: SolverXPBD(view, iterations=10, angular_damping=0.0),) * 2
        for rho in (0.4, 2.0):
            with self.subTest(rho=rho):
                _, expected = _step_off_center_ball_pair(model, factories, iterations=64, rho=rho)
                _, actual = _step_off_center_ball_pair(model, factories, iterations=64, rho=rho, in_place=True)
                np.testing.assert_allclose(actual, expected, atol=1.0e-6)

    def test_substepped_entries_keep_proxy_fixed_point(self):
        """Keep the converged answer close to the linearized one when entries take several substeps."""
        model = _build_off_center_ball_pair()
        for label, factory in (
            ("featherstone", SolverFeatherstone),
            ("xpbd", lambda view: SolverXPBD(view, iterations=10, angular_damping=0.0)),
        ):
            with self.subTest(solver=label):
                factories = (factory, factory)
                _, expected = _step_off_center_ball_pair(
                    model, factories, substeps=3, iterations=600, rho=0.05, joint_proximal_bodies=False
                )
                _, actual = _step_off_center_ball_pair(model, factories, substeps=3, iterations=200, rho=0.4)
                # Proxies follow the iterate over the whole step, not within it; that leaves about 2e-4 here.
                np.testing.assert_allclose(actual, expected, atol=1.0e-3)

    def test_inertia_notification_keeps_proxy_state(self):
        """Leave a reduced-coordinate proxied step unchanged by an inertia notification without a change."""
        velocities = []
        for notify in (False, True):
            builder = newton.ModelBuilder(gravity=(0.0, -9.81, 0.0))
            inertia = wp.mat33(np.eye(3) * 0.01)
            parent = builder.add_link(mass=1.0, inertia=inertia, com=wp.vec3(0.05, 0.02, 0.0))
            child = builder.add_link(
                xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=inertia,
                com=wp.vec3(0.1, 0.0, 0.03),
            )
            root = builder.add_joint_free(parent)
            ball = builder.add_joint_ball(
                parent, child, child_xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity())
            )
            builder.add_articulation([root, ball])
            model = builder.finalize(device="cpu")
            solver = SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry(name="parent", solver=SolverFeatherstone, bodies=[parent], joints=[root]),
                    SolverCoupled.Entry(name="child", solver=SolverFeatherstone, bodies=[child]),
                ],
                SolverCoupledADMM.Config(iterations=20, rho=0.5, gamma=0.5),
            )
            state_0, state_1 = model.state(), model.state()
            joint_qd = state_0.joint_qd.numpy()
            joint_qd[:6] = [0.3, 0.1, 0.0, 2.0, -1.0, 3.0]
            state_0.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            for step in range(6):
                if notify and step == 3:
                    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                solver.step(state_0, state_1, None, None, 1.0 / 120.0)
                state_0, state_1 = state_1, state_0
            velocities.append(state_0.body_qd.numpy())
        np.testing.assert_allclose(velocities[1], velocities[0], atol=1.0e-6)

    def test_parallel_joints_each_get_a_proxy(self):
        """Give each of two joints between the same bodies its own proxy."""
        with warnings.catch_warnings():
            # The builder warns that parallel joints have undefined semantics.
            warnings.simplefilter("ignore")
            model = _build_off_center_ball_pair(parallel=True)
        factories = (SolverFeatherstone, SolverFeatherstone)
        # The linearized update needs a small rho on parallel joints.
        _, expected = _step_off_center_ball_pair(model, factories, iterations=800, rho=0.1, joint_proximal_bodies=False)
        solver, actual = _step_off_center_ball_pair(model, factories, iterations=96, rho=2.0)
        self.assertEqual(sorted(solver._admm_rr_groups[0].proxy_slots_a.numpy().tolist()), [0, 1])
        np.testing.assert_allclose(actual, expected, atol=1.0e-5)


def _build_split_hinge_pair(tree: bool, driven: bool = True) -> newton.Model:
    """Build the split pair of :func:`_build_split_ball_pair` joined by a revolute hinge about z.

    With ``driven`` the hinge has a position drive and armature, which no coupled entry simulates
    for a cross-solver joint.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(np.eye(3) * 0.01)
    parent = builder.add_link(mass=1.0, inertia=inertia)
    child = builder.add_link(xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
    root = builder.add_joint_free(parent)
    hinge = {"child_xform": wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()), "axis": newton.Axis.Z}
    if driven:
        hinge.update(target_ke=50.0, target_kd=5.0, armature=0.1)
    if tree:
        builder.add_articulation([root, builder.add_joint_revolute(parent, child, **hinge)])
    else:
        builder.add_articulation([root])
        builder.add_articulation([builder.add_joint_free(child)])
        builder.add_joint_revolute(parent, child, **hinge)
    builder.color()
    return builder.finalize(device="cpu")


class TestAdmmBodyParticleAttachment(unittest.TestCase):
    """Model attachment rows are converted to rigid-particle ADMM attachments."""

    def test_model_attachment_closes_gap(self):
        """Couple a cross-entry model attachment and pull the endpoints together."""
        model = _build_body_particle_attachment_scene()
        solver = _make_semi_body_particle_solver(model)
        initial_gap = np.linalg.norm(model.state().body_q.numpy()[0, :3] - model.state().particle_q.numpy()[0])

        body_q, particle_q = _run_body_particle(solver, model, n_steps=8, dt=1.0 / 120.0)
        final_gap = np.linalg.norm(body_q[0, :3] - particle_q[0])

        self.assertLess(final_gap, 0.5 * initial_gap)

    def test_deprecated_helper_preserves_legacy_attachment_row(self):
        """Keep the deprecated ADMM helper compatible with its custom-row layout."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        particle = builder.add_particle(pos=(0.3, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.0)

        with self.assertWarnsRegex(DeprecationWarning, "add_attachment_particle_body"):
            attachment = SolverCoupledADMM.add_body_particle_attachment(builder, body, particle, stiffness=500.0)

        self.assertEqual(attachment, 0)
        self.assertEqual(builder.attachment_particle_body_count, 0)

        builder.color()
        model = builder.finalize(device="cpu")
        self.assertEqual(model.attachment_particle_body_count, 0)
        self.assertEqual(
            model.custom_frequency_counts[SolverCoupledADMM.BODY_PARTICLE_ATTACHMENT_FREQUENCY],
            1,
        )
        self.assertIn(
            SolverCoupledADMM.BODY_PARTICLE_ATTACHMENT_FREQUENCY,
            model.custom_frequency_counts,
        )

    def test_single_unsupported_entry_attachment_is_reported(self):
        """Warn when a same-entry solver cannot apply a model attachment."""
        model = _build_body_particle_attachment_scene()
        with self.assertLogs("newton._src.solvers.coupled.solver_coupled_admm", level="WARNING") as logs:
            solver = SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry(
                        "both",
                        lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                        bodies=range(model.body_count),
                        particles=range(model.particle_count),
                    ),
                ],
                SolverCoupledADMM.Config(iterations=1),
            )

        self.assertEqual(solver._admm_rp_groups, [])
        self.assertTrue(any("only SolverVBD" in message for message in logs.output))

    def test_single_vbd_entry_attachment_is_left_to_vbd(self):
        """Leave a same-entry model attachment to SolverVBD without an attachment warning."""
        model = _build_body_particle_attachment_scene()
        with (
            self.assertNoLogs("newton._src.solvers.coupled.solver_coupled_admm", level="WARNING"),
            self.assertWarns(DeprecationWarning),
        ):
            solver = SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry(
                        "both",
                        lambda view: SolverVBD(view, iterations=0, rigid_compliant_alm=False),
                        bodies=range(model.body_count),
                        particles=range(model.particle_count),
                    ),
                ],
                SolverCoupledADMM.Config(iterations=1),
            )

        self.assertEqual(solver._admm_rp_groups, [])
        self.assertEqual(solver.view("both").attachment_particle_body_count, 1)

    def test_unowned_attachment_endpoint_is_reported(self):
        """Warn about attachment rows that no entry can apply."""
        model = _build_body_particle_attachment_scene()
        with self.assertLogs("newton._src.solvers.coupled.solver_coupled_admm", level="WARNING") as logs:
            solver = SolverCoupledADMM(
                model,
                [
                    SolverCoupled.Entry(
                        "body",
                        lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                        bodies=range(model.body_count),
                    ),
                ],
                SolverCoupledADMM.Config(iterations=1),
            )

        self.assertEqual(solver._admm_rp_groups, [])
        self.assertTrue(any("not owned by any entry" in message for message in logs.output))

    def test_masked_reset_preserves_unselected_attachment_history(self):
        """Clear selected attachment duals and preserve every other row."""
        model = _build_two_world_body_particle_attachment_scene()
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    "body",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    bodies=range(model.body_count),
                ),
                SolverCoupled.Entry(
                    "particle",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    particles=range(model.particle_count),
                ),
            ],
            SolverCoupledADMM.Config(iterations=1),
        )
        group = solver._admm_rp_groups[0]
        u_before = np.arange(1, group.u.numpy().size + 1, dtype=np.float32).reshape(group.u.numpy().shape)
        lambda_before = u_before + 10.0
        group.u.assign(u_before)
        group.lambda_.assign(lambda_before)

        solver.reset(
            model.state(),
            world_mask=wp.array((False, False, False), dtype=wp.bool, device=model.device),
            flags=0,
        )
        np.testing.assert_array_equal(group.u.numpy(), u_before)
        np.testing.assert_array_equal(group.lambda_.numpy(), lambda_before)

        solver.reset(
            model.state(),
            world_mask=wp.array((True, False, False), dtype=wp.bool, device=model.device),
            flags=0,
        )
        np.testing.assert_array_equal(group.u.numpy()[0], 0.0)
        np.testing.assert_array_equal(group.lambda_.numpy()[0], 0.0)
        np.testing.assert_array_equal(group.u.numpy()[1], u_before[1])
        np.testing.assert_array_equal(group.lambda_.numpy()[1], lambda_before[1])


class TestAdmmExternalForces(unittest.TestCase):
    """External forces set on ``state_in.body_f`` / ``particle_f`` by the
    caller (e.g. a viewer gizmo) must reach the sub-solvers."""

    def test_body_f_reaches_mujoco(self):
        """An upward ``body_f`` on the rigid sphere should slow its fall
        compared to the zero-force baseline."""
        # Baseline: no external force, body falls under gravity.
        model_a, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_a = _make_solver(model_a, rs, re, admm_iters=1)
        state_0 = model_a.state()
        state_1 = model_a.state()
        collision_pipeline = newton.CollisionPipeline(model_a)
        contacts = collision_pipeline.contacts()
        control = model_a.control()
        newton.eval_fk(model_a, model_a.joint_q, model_a.joint_qd, state_0)
        for _ in range(5):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver_a.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        z_baseline = state_0.body_q.numpy()[0, 2]

        # With a strong upward body_f applied each step, the body should fall
        # less (or even rise).
        model_b, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_b = _make_solver(model_b, rs, re, admm_iters=1)
        state_0 = model_b.state()
        state_1 = model_b.state()
        collision_pipeline = newton.CollisionPipeline(model_b)
        contacts = collision_pipeline.contacts()
        control = model_b.control()
        newton.eval_fk(model_b, model_b.joint_q, model_b.joint_qd, state_0)
        body_idx = rs  # only MuJoCo body
        body_mass = float(model_b.body_mass.numpy()[body_idx])
        upward_force = 5.0 * body_mass * 9.81  # 5 g upward wrench
        for _ in range(5):
            state_0.clear_forces()
            wrench = np.zeros((model_b.body_count, 6), dtype=np.float32)
            wrench[body_idx, 2] = upward_force  # linear z
            state_0.body_f = wp.array(wrench, dtype=wp.spatial_vector, device=model_b.device)
            collision_pipeline.collide(state_0, contacts)
            solver_b.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        z_with_force = state_0.body_q.numpy()[0, 2]

        self.assertGreater(
            z_with_force,
            z_baseline + 0.02,
            f"external body_f didn't reach MuJoCo: baseline z={z_baseline:.4f}, "
            f"with 5g upward force z={z_with_force:.4f}",
        )


class TestAdmmCollisionDetection(unittest.TestCase):
    """Collision-detected ADMM contact constraints."""

    def test_rigid_contact_collision_capacity(self):
        """Apply independent capacity overrides while retaining mesh contacts and matching."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        mesh_body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
        mesh = newton.Mesh(
            vertices=np.array([[0.0, -0.1, -0.1], [0.0, 0.1, -0.1], [0.0, 0.0, 0.1]], dtype=np.float32),
            indices=np.array([0, 1, 2], dtype=np.int32),
            compute_inertia=False,
        )
        builder.add_shape_mesh(body=mesh_body, mesh=mesh)
        sphere_body = builder.add_body(xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_sphere(body=sphere_body, radius=0.1)
        model = builder.finalize(device="cpu")
        entries = [
            SolverCoupled.Entry(name="mesh", solver=SolverXPBD, bodies=[mesh_body]),
            SolverCoupled.Entry(name="sphere", solver=SolverXPBD, bodies=[sphere_body]),
        ]
        defaults = inspect.signature(newton.CollisionPipeline).parameters
        for capacity, factor, matching in (
            (None, None, "disabled"),
            (8192, None, "latest"),
            (None, 0.5, "sticky"),
            (8192, 256.0, "latest"),
        ):
            with self.subTest(capacity=capacity, factor=factor, matching=matching):
                solver = SolverCoupledADMM(
                    model,
                    entries,
                    SolverCoupledADMM.Config(
                        iterations=1,
                        contact_pairs=[SolverCoupledADMM.ContactPair(source="mesh", destination="sphere")],
                        rigid_contact_matching=matching,
                        contact_max_triangle_pairs=capacity,
                        contact_reduction_hashtable_size_factor=factor,
                    ),
                )
                pipeline = solver._admm_collision_pipeline
                expected_capacity = capacity if capacity is not None else _TRIANGLE_PAIRS_MIN_CAPACITY
                expected_factor = (
                    factor if factor is not None else defaults["contact_reduction_hashtable_size_factor"].default
                )
                self.assertEqual(pipeline.narrow_phase.max_triangle_pairs, expected_capacity)
                requested_slots = max(1024, int(expected_capacity * expected_factor))
                self.assertEqual(
                    pipeline.narrow_phase.global_contact_reducer.hashtable.capacity,
                    1 << (requested_slots - 1).bit_length(),
                )
                self.assertEqual(pipeline.contact_matching, matching)
                state_in, state_out = model.state(), model.state()
                solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
                self.assertGreater(solver.collision_contact_count_max, 0)
                self.assertTrue(np.all(np.isfinite(state_out.body_q.numpy())))

    def test_masked_reset_preserves_unselected_contact_history(self):
        """Clear only dynamic contact duals touching the reset world."""
        world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        world.add_particle(pos=(-0.04, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.05)
        world.add_particle(pos=(0.04, 0.0, 0.0), vel=wp.vec3(), mass=1.0, radius=0.05)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_world(world)
        builder.add_world(world, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))
        model = builder.finalize(device="cpu")
        solver = SolverCoupledADMM(
            model,
            [
                SolverCoupled.Entry(
                    "a",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    particles=(0, 2),
                ),
                SolverCoupled.Entry(
                    "b",
                    lambda view: SolverSemiImplicit(view, enable_tri_contact=False),
                    particles=(1, 3),
                ),
            ],
            SolverCoupledADMM.Config(
                iterations=1,
                contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")],
            ),
        )
        solver._refresh_collision_contact_groups(model.state())
        group = solver._admm_dynamic_pp_contact_groups[0]
        active = np.flatnonzero(group.active.numpy())
        self.assertEqual(active.size, 2)
        ids_a = group.particle_ids_a.numpy()[active]
        ids_b = group.particle_ids_b.numpy()[active]
        worlds_a = solver._entries[group.particle_entry_name_a].view.particle_world.numpy()[ids_a]
        worlds_b = solver._entries[group.particle_entry_name_b].view.particle_world.numpy()[ids_b]
        selected = (worlds_a == 0) | (worlds_b == 0)

        u_before = np.arange(1, group.u.numpy().size + 1, dtype=np.float32).reshape(group.u.numpy().shape)
        lambda_before = u_before + 20.0
        group.u.assign(u_before)
        group.lambda_.assign(lambda_before)
        solver.reset(
            model.state(),
            world_mask=wp.array((True, False, False), dtype=wp.bool, device=model.device),
            flags=0,
        )

        after_u = group.u.numpy()[active]
        after_lambda = group.lambda_.numpy()[active]
        np.testing.assert_array_equal(after_u[selected], 0.0)
        np.testing.assert_array_equal(after_lambda[selected], 0.0)
        np.testing.assert_array_equal(after_u[~selected], u_before[active][~selected])
        np.testing.assert_array_equal(after_lambda[~selected], lambda_before[active][~selected])

    def test_rigid_contact_detection_rejects_cross_world_pairs(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = 1000.0

        builder.begin_world()
        body_a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        shape_a = builder.add_shape_box(body=body_a, hx=0.05, hy=0.05, hz=0.05)
        builder.end_world()

        builder.begin_world()
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        shape_b = builder.add_shape_box(body=body_b, hx=0.05, hy=0.05, hz=0.05)
        builder.end_world()

        model = builder.finalize(device="cpu")
        model.shape_contact_pairs = wp.array(
            np.asarray([(shape_a, shape_b)], dtype=np.int32), dtype=wp.vec2i, device=model.device
        )

        with self.assertRaisesRegex(ValueError, "same world"):
            SolverCoupledADMM(
                model=model,
                entries=[
                    SolverCoupled.Entry(
                        name="a",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[body_a],
                    ),
                    SolverCoupled.Entry(
                        name="b",
                        solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                        bodies=[body_b],
                    ),
                ],
                coupling=SolverCoupledADMM.Config(
                    contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")],
                ),
            )

    def test_collision_particle_particle_contacts_are_refreshed_in_solver(self):
        model = _build_two_particle_contact_scene(gap=-0.08)
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[0],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[1],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=10,
                rho=30.0,
                baumgarte=0.5,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(source="a", destination="b"),
                ],
            ),
        )

        q_contact = _run_particles(solver, model, n_steps=4)

        self.assertGreater(solver.collision_contact_count_max, 0)
        self.assertGreater(q_contact[1, 0] - q_contact[0, 0], 0.08 + 1.0e-3)

    def test_collision_particle_particle_contacts_respect_disabled_model_particle_grid(self):
        model = _build_two_particle_contact_scene(gap=-0.08)
        model.particle_grid = None
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="a",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[0],
                ),
                SolverCoupled.Entry(
                    name="b",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[1],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                contact_pairs=[SolverCoupledADMM.ContactPair(source="a", destination="b")],
            ),
        )

        _run_particles(solver, model, n_steps=1)

        self.assertEqual(solver.collision_contact_count_max, 0)

    def test_collision_frictional_contact_matches_inclined_plane_box_motion(self):
        friction = 0.4
        angle = math.radians(35.0)
        dt = 1.0 / 360.0
        steps = 120
        displacement, velocity, contact_count = _run_inclined_plane_particle_box(
            angle,
            friction,
            steps=steps,
            dt=dt,
        )

        t = steps * dt
        acceleration = 10.0 * (math.sin(angle) - friction * math.cos(angle))
        expected_displacement = 0.5 * acceleration * t * t
        expected_velocity = acceleration * t

        self.assertGreater(contact_count, 0)
        self.assertGreater(acceleration, 0.0)
        self.assertAlmostEqual(displacement, expected_displacement, delta=0.45 * expected_displacement)
        self.assertAlmostEqual(velocity, expected_velocity, delta=0.45 * expected_velocity)

    def test_collision_frictional_contact_holds_subcritical_inclined_box(self):
        friction = 0.4
        angle = math.radians(15.0)
        displacement, velocity, contact_count = _run_inclined_plane_particle_box(
            angle,
            friction,
            steps=120,
            dt=1.0 / 360.0,
        )

        self.assertGreater(contact_count, 0)
        self.assertLess(math.tan(angle), friction)
        self.assertLess(abs(displacement), 0.01)
        self.assertLess(abs(velocity), 0.05)

    def test_collision_rigid_rigid_frictional_contact_matches_inclined_plane_box_motion(self):
        friction = 0.35
        angle = math.radians(24.0)
        steps = 120
        dt = 1.0 / 360.0
        displacement, velocity, min_gap, contact_count = _run_collision_inclined_plane_rigid_box(
            angle,
            friction,
            steps=steps,
            dt=dt,
        )

        t = steps * dt
        acceleration = 10.0 * (math.sin(angle) - friction * math.cos(angle))
        expected_displacement = 0.5 * acceleration * t * t
        expected_velocity = acceleration * t

        self.assertGreater(contact_count, 0)
        self.assertGreater(acceleration, 0.0)
        self.assertAlmostEqual(displacement, expected_displacement, delta=0.65 * expected_displacement)
        self.assertAlmostEqual(velocity, expected_velocity, delta=0.65 * expected_velocity)
        self.assertGreater(min_gap, -0.03)

    def test_collision_particle_shape_contacts_are_refreshed_in_solver(self):
        model, particle, tray_body, _ = _build_collision_contact_scene()
        solver = SolverCoupledADMM(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="drop",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    particles=[particle],
                ),
                SolverCoupled.Entry(
                    name="tray",
                    solver=lambda v: SolverSemiImplicit(model=v, enable_tri_contact=False),
                    bodies=[tray_body],
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=12,
                # Migrate the legacy tuning at the reference substep of 1/120 s.
                rho=0.375,
                gamma=6.0,
                baumgarte=0.1,
                contact_pairs=[
                    SolverCoupledADMM.ContactPair(
                        source="drop",
                        destination="tray",
                    ),
                ],
            ),
        )

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        initial_tray_z = float(state_0.body_q.numpy()[tray_body, 2])
        min_gap = float(state_0.particle_q.numpy()[particle, 2] - initial_tray_z)
        for _ in range(90):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts=None, dt=1.0 / 120.0)
            state_0, state_1 = state_1, state_0
            particle_z = float(state_0.particle_q.numpy()[particle, 2])
            tray_z = float(state_0.body_q.numpy()[tray_body, 2])
            min_gap = min(min_gap, particle_z - tray_z)

        final_particle_z = float(state_0.particle_q.numpy()[particle, 2])
        final_tray_z = float(state_0.body_q.numpy()[tray_body, 2])
        final_gap = final_particle_z - final_tray_z
        self.assertGreater(solver.collision_contact_count_max, 0)
        self.assertLessEqual(min_gap, 0.08)
        self.assertGreater(min_gap, -0.02)
        self.assertGreater(final_gap, 0.02)
        self.assertLess(final_tray_z, initial_tray_z - 1.0e-3)


if __name__ == "__main__":
    unittest.main()
