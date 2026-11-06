# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Map Newton particles and their springs, triangles, and tetrahedra to MuJoCo flexes."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import warp as wp

from ...exceptions import NewtonWarning
from ...geometry import ParticleFlags
from ...sim import Model
from .collision_masks import mujoco_mask_to_signed


def add_particle_flexes(
    mujoco: Any, spec: Any, model: Model, particles: np.ndarray, particle_geoms: set[str]
) -> list[Any]:
    """Add a body per particle and one flex per element dimension to ``spec``.

    Each simulated particle becomes a body at its initial position with three
    slide joints, so its MuJoCo coordinates are the displacement from that
    position. Massless or inactive particles get no joints and stay pinned.

    Tetrahedra, triangles, and springs become 3D, 2D, and 1D flexes. An element
    whose particles all belong to a higher-dimensional flex is skipped, so the
    surface triangles of a tetrahedral mesh are not duplicated. Flexes collide
    with the geoms named in ``particle_geoms`` but not with themselves.

    Returns:
        The MuJoCo bodies created for ``particles``, in the same order.
    """
    if len(particles) == 0:
        return []

    # Reserve a mask bit that no geom uses as contype. Flexes collide only with geoms
    # that carry it in their conaffinity, which leaves rigid filtering unchanged.
    used = 0
    for geom in spec.geoms:
        used |= geom.contype & 0xFFFFFFFF
    free = ~used & 0xFFFFFFFF
    if not free:
        warnings.warn("No free MuJoCo collision bit is left; flexes will not collide.", NewtonWarning, stacklevel=3)
    bit = mujoco_mask_to_signed(free & -free)
    for geom in spec.geoms:
        mask = geom.conaffinity | bit if geom.name in particle_geoms else geom.conaffinity & ~bit
        geom.conaffinity = mujoco_mask_to_signed(mask)

    q = model.particle_q.numpy()[particles]
    mass = model.particle_mass.numpy()[particles]
    radius = model.particle_radius.numpy()[particles]
    simulated = (mass > 0.0) & (model.particle_flags.numpy()[particles] & int(ParticleFlags.ACTIVE) != 0)

    bodies = []
    for i, p in enumerate(particles):
        body = spec.worldbody.add_body(name=f"particle_{p}", pos=q[i])
        if simulated[i]:
            body.mass = mass[i]
            body.ipos = [0.0, 0.0, 0.0]
            body.inertia = [0.4 * mass[i] * radius[i] ** 2] * 3
            body.explicitinertial = True
            for axis in np.eye(3):
                body.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
        bodies.append(body)

    local = np.full(model.particle_count, -1, dtype=np.int64)
    local[particles] = np.arange(len(particles))
    claimed = np.zeros(model.particle_count, dtype=bool)
    for dim, indices, params in (
        (3, model.tet_indices, model.tet_materials),
        (2, model.tri_indices, model.tri_materials),
        (1, model.spring_indices, model.spring_stiffness),
    ):
        if indices is None or indices.size == 0:
            continue
        elems = indices.numpy().reshape(-1, dim + 1)
        keep = np.all(local[elems] >= 0, axis=1) & ~np.all(claimed[elems], axis=1)
        if not np.any(keep):
            continue
        elems = elems[keep]
        claimed[elems] = True
        verts = np.unique(elems)

        flex = spec.add_flex()
        flex.name = f"flex_{dim}d"
        flex.dim = dim
        flex.vertbody = [bodies[i].name for i in local[verts]]
        flex.elem = np.searchsorted(verts, elems).ravel().tolist()
        flex.radius = float(radius[local[verts]].max())
        flex.friction[0] = model.particle_mu
        flex.contype = bit
        flex.conaffinity = 0
        flex.selfcollide = mujoco.mjtFlexSelf.mjFLEXSELF_NONE

        if dim == 1:
            values = np.stack([params.numpy(), model.spring_damping.numpy()], axis=1)[keep]
            flex.edgestiffness, flex.edgedamping = _uniform(values, "spring stiffness and damping")
        else:
            # Newton materials store Lame parameters (mu, lambda) in their first two columns.
            mu, lam = _uniform(params.numpy()[keep, :2], f"{dim}D element material")
            if mu + lam > 0.0:
                flex.young = mu * (3.0 * lam + 2.0 * mu) / (mu + lam)
                flex.poisson = lam / (2.0 * (mu + lam))
            if dim == 2:
                # Newton triangle stiffness is per unit area, and MuJoCo membrane
                # stiffness is young * thickness, so a unit thickness passes it through.
                flex.thickness = 1.0
                flex.elastic2d = 2  # stretch only
    return bodies


def _uniform(values: np.ndarray, what: str) -> np.ndarray:
    """Return the mean row of ``values``, warning if the rows differ, since MuJoCo stores one value per flex."""
    if np.ptp(values, axis=0).any():
        warnings.warn(
            f"MuJoCo flexes use a single {what}; averaging non-uniform Newton values.", NewtonWarning, stacklevel=4
        )
    return values.mean(axis=0)


def particle_coord_adr(mj_model: Any, bodies: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the qpos and qvel start of each particle body's slide joints, or -1 for pinned particles."""
    jnt = mj_model.body_jntadr[[body.id for body in bodies]]
    return (
        np.where(jnt >= 0, mj_model.jnt_qposadr[jnt], -1).astype(np.int32),
        np.where(jnt >= 0, mj_model.jnt_dofadr[jnt], -1).astype(np.int32),
    )


@wp.kernel
def particles_to_mj_kernel(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    world_mask: wp.array[wp.bool],
    particle_q0: wp.array[wp.vec3],
    qpos_adr: wp.array[wp.int32],
    qvel_adr: wp.array[wp.int32],
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
):
    world, p = wp.tid()
    if world_mask and not world_mask[world]:
        return
    qa = qpos_adr[p]
    if qa < 0:
        return
    va = qvel_adr[p]
    i = world * qpos_adr.shape[0] + p
    x = particle_q[i] - particle_q0[p]
    v = particle_qd[i]
    for k in range(3):
        qpos[world, qa + k] = x[k]
        qvel[world, va + k] = v[k]


@wp.kernel
def mj_to_particles_kernel(
    qpos: wp.array2d[wp.float32],
    qvel: wp.array2d[wp.float32],
    particle_q0: wp.array[wp.vec3],
    qpos_adr: wp.array[wp.int32],
    qvel_adr: wp.array[wp.int32],
    particle_q_prev: wp.array[wp.vec3],
    particle_qd_prev: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    world, p = wp.tid()
    i = world * qpos_adr.shape[0] + p
    qa = qpos_adr[p]
    if qa < 0:
        particle_q[i] = particle_q_prev[i]
        particle_qd[i] = particle_qd_prev[i]
        return
    va = qvel_adr[p]
    particle_q[i] = particle_q0[p] + wp.vec3(qpos[world, qa], qpos[world, qa + 1], qpos[world, qa + 2])
    particle_qd[i] = wp.vec3(qvel[world, va], qvel[world, va + 1], qvel[world, va + 2])
