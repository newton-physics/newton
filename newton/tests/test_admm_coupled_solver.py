# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ADMM-coupled solvers.

These tests validate generic :class:`SolverCoupledADMM` ADMM plumbing against a
cloth-plus-rigid-body scene.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled.interface import CouplingInterface
from newton.solvers import (
    SolverBase,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverVBD,
    SolverXPBD,
)
from newton.solvers.experimental.coupled import (
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
                solver=lambda v: SolverVBD(model=v, iterations=5),
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
    contacts = model.contacts()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(n_steps):
        state_0.clear_forces()
        model.collide(state_0, contacts)
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
    SolverCoupledADMM.add_body_particle_attachment(
        builder,
        body,
        particle,
        stiffness=500.0,
        enabled=enabled,
    )
    builder.color()
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
            rho=50.0,
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
            rho=5.0,
            gamma=0.2,
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

    def test_rejects_invalid_numerical_config(self):
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
        )
        for kwargs, message in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                SolverCoupledADMM(model=model, entries=entries, coupling=SolverCoupledADMM.Config(**kwargs))

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
                    solver=lambda v: SolverVBD(model=v, iterations=1),
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
        for name in ("parent", "child"):
            with self.subTest(name=name):
                view = solver.view(name)
                self.assertEqual(view.shape_count, model.shape_count)
                shape_body = view.shape_body.numpy()
                body_flags = view.body_flags.numpy()
                shape_flags = view.shape_flags.numpy()
                proxy_shapes = [
                    shape_id
                    for shape_id, body_id in enumerate(shape_body)
                    if int(body_id) >= 0 and int(body_flags[int(body_id)]) & proxy_flag
                ]
                owned_shapes = [
                    shape_id
                    for shape_id, body_id in enumerate(shape_body)
                    if int(body_id) >= 0 and shape_id not in proxy_shapes
                ]
                hidden_shapes = [shape_id for shape_id, body_id in enumerate(shape_body) if int(body_id) < 0]

                self.assertEqual(len(proxy_shapes), 1)
                self.assertEqual(len(owned_shapes), 1)
                self.assertEqual(len(hidden_shapes), 1)
                self.assertEqual(int(shape_flags[proxy_shapes[0]]) & collision_mask, 0)
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


class TestAdmmBodyParticleAttachment(unittest.TestCase):
    """Custom model attributes are converted to rigid-particle ADMM attachments."""

    def test_custom_attribute_attachment_closes_gap(self):
        model = _build_body_particle_attachment_scene()
        solver = _make_semi_body_particle_solver(model)
        initial_gap = np.linalg.norm(model.state().body_q.numpy()[0, :3] - model.state().particle_q.numpy()[0])

        body_q, particle_q = _run_body_particle(solver, model, n_steps=8, dt=1.0 / 120.0)
        final_gap = np.linalg.norm(body_q[0, :3] - particle_q[0])

        self.assertLess(final_gap, 0.5 * initial_gap)


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
        contacts = model_a.contacts()
        control = model_a.control()
        newton.eval_fk(model_a, model_a.joint_q, model_a.joint_qd, state_0)
        for _ in range(5):
            state_0.clear_forces()
            model_a.collide(state_0, contacts)
            solver_a.step(state_0, state_1, control, contacts, 1.0 / 60.0)
            state_0, state_1 = state_1, state_0
        z_baseline = state_0.body_q.numpy()[0, 2]

        # With a strong upward body_f applied each step, the body should fall
        # less (or even rise).
        model_b, rs, re, _ = _build_cloth_rigid_scene(rigid_pos=(0.0, 0.0, 5.0))
        solver_b = _make_solver(model_b, rs, re, admm_iters=1)
        state_0 = model_b.state()
        state_1 = model_b.state()
        contacts = model_b.contacts()
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
            model_b.collide(state_0, contacts)
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
        model.shape_contact_pair_count = 1

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
                rho=45.0,
                gamma=0.05,
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
