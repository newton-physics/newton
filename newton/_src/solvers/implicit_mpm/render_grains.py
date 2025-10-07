# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warp as wp
import warp.fem as fem

import newton

__all__ = ["sample_render_grains", "update_render_grains"]


@wp.kernel
def sample_grains(
    particles: wp.array(dtype=wp.vec3),
    radius: wp.array(dtype=float),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    rng = wp.rand_init(pid * positions.shape[1] + k)

    pos_loc = 2.0 * wp.vec3(wp.randf(rng) - 0.5, wp.randf(rng) - 0.5, wp.randf(rng) - 0.5) * radius[pid]
    positions[pid, k] = particles[pid] + pos_loc


@wp.kernel
def transform_grains(
    particle_pos_prev: wp.array(dtype=wp.vec3),
    particle_transform_prev: wp.array(dtype=wp.mat33),
    particle_pos: wp.array(dtype=wp.vec3),
    particle_transform: wp.array(dtype=wp.mat33),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    pos_adv = positions[pid, k]

    p_pos = particle_pos[pid]
    p_frame = particle_transform[pid]
    p_pos_prev = particle_pos_prev[pid]
    p_frame_prev = particle_transform_prev[pid]

    pos_loc = wp.inverse(p_frame_prev) @ (pos_adv - p_pos_prev)

    p_pos_adv = p_frame @ pos_loc + p_pos
    positions[pid, k] = p_pos_adv


@fem.integrand
def advect_grains(
    s: fem.Sample,
    domain: fem.Domain,
    grid_vel: fem.Field,
    dt: float,
    positions: wp.array(dtype=wp.vec3),
):
    x = domain(s)
    vel = grid_vel(s)
    pos_adv = x + dt * vel
    positions[s.qp_index] = pos_adv


@wp.kernel
def advect_grains_from_particles(
    dt: float,
    particle_pos_prev: wp.array(dtype=wp.vec3),
    particle_pos: wp.array(dtype=wp.vec3),
    particle_vel_grad: wp.array(dtype=wp.mat33),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    p_pos = particle_pos[pid]
    p_pos_prev = particle_pos_prev[pid]

    pos_loc = positions[pid, k] - p_pos_prev

    p_vel_grad = particle_vel_grad[pid]

    displ = dt * p_vel_grad * pos_loc + (p_pos - p_pos_prev)
    positions[pid, k] += displ


@wp.kernel
def project_grains(
    radius: wp.array(dtype=float),
    particle_pos: wp.array(dtype=wp.vec3),
    particle_frames: wp.array(dtype=wp.mat33),
    positions: wp.array2d(dtype=wp.vec3),
):
    pid, k = wp.tid()

    pos_adv = positions[pid, k]

    p_pos = particle_pos[pid]
    p_frame = particle_frames[pid]

    # keep within source particle
    # pos_loc = wp.inverse(p_frame) @ (pos_adv - p_pos)
    # dist = wp.max(wp.abs(pos_loc))
    # if dist > radius:
    #     pos_loc = pos_loc / dist * radius
    # p_pos_adv = p_frame @ pos_loc + p_pos

    p_frame = (radius[pid] * radius[pid]) * p_frame * wp.transpose(p_frame)
    pos_loc = pos_adv - p_pos
    vn = wp.max(1.0, wp.dot(pos_loc, wp.inverse(p_frame) * pos_loc))
    p_pos_adv = pos_loc / wp.sqrt(vn) + p_pos

    positions[pid, k] = p_pos_adv


def sample_render_grains(state: newton.State, particle_radius: wp.array, grains_per_particle: int):
    """Generate per-particle point samples used for high-resolution rendering.

    For each simulation particle, this creates ``grains_per_particle`` random
    points uniformly within a cube of size ``2 * particle_radius`` centered at
    the particle position. The resulting 2D array can be updated each time step
    using ``update_render_grains`` to passively advect and project grains within
    the affinely deformed particle shape.

    Args:
        state: Current Newton state providing particle positions.
        particle_radius: Rendering grain sampling radius per particle.
        grains_per_particle: Number of grains to sample per particle.

    Returns:
        A ``wp.array`` with shape ``(num_particles, grains_per_particle)`` of
        type ``wp.vec3`` containing grain positions.
    """

    grains = wp.empty((state.particle_count, grains_per_particle), dtype=wp.vec3, device=state.particle_q.device)

    wp.launch(
        sample_grains,
        dim=grains.shape,
        inputs=[
            state.particle_q,
            particle_radius,
            grains,
        ],
        device=state.particle_q.device,
    )

    return grains


def update_render_grains(
    state_prev: newton.State,
    state: newton.State,
    grains: wp.array,
    particle_radius: wp.array,
    dt: float,
):
    """Advect grain samples with the grid velocity and keep them inside the deformed particle.

     The grains are advanced by composing two motions within the time step:
     1) a particle-local affine update using the particle velocity gradient and
        particle positions (APIC-like), and 2) a grid-based PIC advection using
        the current velocity field. After advection, positions are projected
        back using an ellipsoidal approximation of the particle defined by its
        deformation frame and ``particle_radius``.

    If no velocity field is available in the ``state`` the function
    returns without modification.

     Args:
         state_prev: Previous state (t_n) with particle positions and frames.
         state: Current state (t_{n+1}) providing velocity field and particles.
         grains: 2D array of grain positions per particle to be updated in place.
         particle_radius: Per-particle radius used for projection.
         dt: Time step duration.
    """

    if state.velocity_field is None:
        return
    grain_pos = grains.flatten()
    domain = fem.Cells(state.velocity_field.space.geometry)
    grain_pic = fem.PicQuadrature(domain, positions=grain_pos)

    wp.launch(
        advect_grains_from_particles,
        dim=grains.shape,
        inputs=[
            dt,
            state_prev.particle_q,
            state.particle_q,
            state.particle_qd_grad,
            grains,
        ],
        device=grains.device,
    )

    fem.interpolate(
        advect_grains,
        quadrature=grain_pic,
        values={
            "dt": dt,
            "positions": grain_pos,
        },
        fields={
            "grid_vel": state.velocity_field,
        },
        device=grains.device,
    )

    wp.launch(
        project_grains,
        dim=grains.shape,
        inputs=[
            particle_radius,
            state.particle_q,
            state.particle_transform,
            grains,
        ],
        device=grains.device,
    )
