# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed pool of implicit-MPM particles that follows an anchor body in each world."""

import math
from collections.abc import Sequence

import numpy as np
import warp as wp

import newton


@wp.func
def _surface_height(mesh: wp.uint64, x: float, y: float, ray_z: float, ray_length: float):
    """Cast downward at world XY [m] and return whether the mesh was hit and its height [m]."""
    query = wp.mesh_query_ray(mesh, wp.vec3(x, y, ray_z), wp.vec3(0.0, 0.0, -1.0), ray_length)
    height = ray_z - query.t
    return query.result, height


@wp.kernel(enable_backward=False)
def _mark_resets(selected: wp.array[wp.bool], pending: wp.array[int]):
    """Mark selected worlds for particle reinitialization."""
    i = wp.tid()
    if selected[i]:
        pending[i] = 1


@wp.kernel(enable_backward=False)
def _update_window_center(
    body_q: wp.array[wp.transform],
    body_indices: wp.array[int],
    world_origins: wp.array[wp.vec3],
    surface_center: wp.vec2,
    surface_size: wp.vec2,
    pool_size: wp.vec2,
    tile_size: float,
    window_centers: wp.array[wp.vec2],
):
    """Snap each anchor's XY position [m] to the tile grid and keep the whole pool on the surface."""
    i = wp.tid()
    p = wp.transform_get_translation(body_q[body_indices[i]])
    o = world_origins[i]
    allowed_half = (surface_size - pool_size) * 0.5
    x = wp.floor((p[0] - o[0]) / tile_size + 0.5) * tile_size
    y = wp.floor((p[1] - o[1]) / tile_size + 0.5) * tile_size
    window_centers[i] = wp.vec2(
        wp.clamp(o[0] + x, surface_center[0] - allowed_half[0], surface_center[0] + allowed_half[0]),
        wp.clamp(o[1] + y, surface_center[1] - allowed_half[1], surface_center[1] + allowed_half[1]),
    )


@wp.kernel(enable_backward=False)
def _recycle_particles(
    particle_world: wp.array[int],
    window_centers: wp.array[wp.vec2],
    template_positions: wp.array[wp.vec3],
    reset_requested: wp.array[int],
    dynamic_mass: wp.array[float],
    initial_plastic_volume_ratio: wp.array[float],
    window_size: wp.vec2,
    pool_size: wp.vec2,
    surface_mesh: wp.uint64,
    ray_z: float,
    ray_length: float,
    reset_positions: wp.array[wp.vec3],
    is_dynamic: wp.array[int],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    particle_inv_mass: wp.array[float],
    particle_qd_grad: wp.array[wp.mat33],
    particle_elastic_strain: wp.array[wp.mat33],
    particle_plastic_volume_ratio: wp.array[float],
    particle_stress: wp.array[wp.mat33],
    particle_transform: wp.array[wp.mat33],
    recycled: wp.array[int],
    mass_change_count: wp.array[int],
    surface_miss_count: wp.array[int],
    recycle_count: wp.array[int],
):
    """Wrap particles into the pool, restore the default material state, and zero the mass of the ring.

    ``template_positions`` holds fixed world XY coordinates [m] and particle-layer Z offsets [m].
    ``reset_positions`` holds surface-adjusted world XYZ positions [m]. ``reset_requested`` flags
    request reinitialization of a world. Each counter is a one-element array shared by the launch.
    """
    i = wp.tid()
    world = particle_world[i]
    center = window_centers[world]
    template = template_positions[i]
    # Toroidal wrapping retains overlap and restores fresh material at the entering edge.
    x = template[0] + wp.floor((center[0] + 0.5 * pool_size[0] - template[0]) / pool_size[0]) * pool_size[0]
    y = template[1] + wp.floor((center[1] + 0.5 * pool_size[1] - template[1]) / pool_size[1]) * pool_size[1]
    # Float32 wrapping can round past the pool edge and miss the surface at its boundary.
    x = wp.clamp(x, center[0] - 0.5 * pool_size[0], center[0] + 0.5 * pool_size[0])
    y = wp.clamp(y, center[1] - 0.5 * pool_size[1], center[1] + 0.5 * pool_size[1])
    reset_position = reset_positions[i]
    reset_site = wp.abs(reset_position[0] - x) > 1.0e-5 or wp.abs(reset_position[1] - y) > 1.0e-5
    if reset_site or reset_requested[world] != 0:
        hit, height = _surface_height(surface_mesh, x, y, ray_z, ray_length)
        if not hit:
            wp.atomic_add(surface_miss_count, 0, 1)
            recycled[i] = 0
            return
        reset_position = wp.vec3(x, y, height + template[2])
    dynamic = int(wp.abs(x - center[0]) < 0.5 * window_size[0] and wp.abs(y - center[1]) < 0.5 * window_size[1])
    reset_position_changed = wp.length(reset_position - reset_positions[i]) > 1.0e-5
    escaped = (
        wp.abs(particle_q[i][0] - center[0]) >= 0.5 * pool_size[0]
        or wp.abs(particle_q[i][1] - center[1]) >= 0.5 * pool_size[1]
    )
    # Ring particles are rewritten every step; only a new site or state counts as recycled.
    moved = reset_position_changed or escaped or dynamic != is_dynamic[i] or reset_requested[world] != 0
    recycled[i] = wp.where(moved, 1, 0)
    if moved:
        wp.atomic_add(recycle_count, 0, 1)
    if moved or dynamic == 0:
        particle_q[i] = reset_position
        particle_qd[i] = wp.vec3(0.0)
        particle_qd_grad[i] = wp.mat33(0.0)
        particle_elastic_strain[i] = wp.identity(n=3, dtype=float)
        particle_plastic_volume_ratio[i] = initial_plastic_volume_ratio[i]
        particle_stress[i] = wp.mat33(0.0)
        particle_transform[i] = wp.identity(n=3, dtype=float)
    target_mass = float(dynamic) * dynamic_mass[i]
    if particle_mass[i] != target_mass:
        wp.atomic_add(mass_change_count, 0, 1)
        particle_mass[i] = target_mass
        particle_inv_mass[i] = wp.where(dynamic != 0, 1.0 / dynamic_mass[i], 0.0)
    if moved:
        reset_positions[i] = reset_position
        is_dynamic[i] = dynamic


@wp.kernel(enable_backward=False)
def _pin_ring_particles(
    is_dynamic: wp.array[int],
    reset_positions: wp.array[wp.vec3],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    """Return ring particles to their reset positions [m] with zero velocity [m/s]."""
    i = wp.tid()
    if is_dynamic[i] == 0:
        particle_q[i] = reset_positions[i]
        particle_qd[i] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def _refresh_density(mass: wp.array[float], volume: wp.array[float], density: wp.array[float]):
    """Update particle densities [kg/m^3] from masses [kg] and fixed reference volumes [m^3]."""
    i = wp.tid()
    density[i] = mass[i] / volume[i]


class ParticleStreaming:
    """Fixed pool of particles that follows one anchor body per world.

    The pool is a rectangle of the dynamic window plus a ring of width ``margin`` on each side.
    Every particle keeps a lattice site that wraps around the pool, so when the window center moves
    by one ``tile_size`` step, the strip left behind reappears at the leading edge in the default
    material state, placed on ``surface_mesh``. Ring particles are kinematic: they carry zero mass and
    are returned to their sites by :meth:`post_step`. No particle is created or destroyed, and no
    history is kept for sand that leaves the pool.

    Create with :meth:`SolverImplicitMPM.set_particle_streaming`. Call :meth:`update` before each step
    with the state holding the anchor poses and particle kinematics, and :meth:`post_step` after it.
    Both can be recorded in a CUDA graph on a device supporting conditional nodes.

    With particle sleeping enabled, ring particles stay kinematic and recycled particles are woken
    with a fresh quiet timer.

    .. experimental::

        Runtime-only pool management. The interface may change.
    """

    def __init__(
        self,
        solver,
        state,
        *,
        body_indices: Sequence[int],
        size: Sequence[float],
        margin: float,
        tile_size: float,
        surface_mesh: wp.Mesh,
        world_origins,
        spawn_heights,
    ):
        self.solver = solver
        model = self.model = solver.model
        device = model.device
        world_count = model.world_count
        self.window_size = tuple(float(v) for v in size)
        self.margin = float(margin)
        self.tile_size = float(tile_size)
        if len(self.window_size) != 2 or not all(
            math.isfinite(v) and v > 0.0 for v in (*self.window_size, self.margin, self.tile_size)
        ):
            raise ValueError("Streaming window size, margin and tile size must be positive and finite.")
        if self.tile_size > self.margin:
            raise ValueError("Streaming tile size must not exceed the margin.")
        self.pool_size = tuple(v + 2.0 * self.margin for v in self.window_size)

        indices = np.asarray(body_indices)
        if indices.shape != (world_count,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Particle streaming needs exactly one anchor body index per world.")
        if np.any(indices < 0) or np.any(indices >= model.body_count):
            raise ValueError("Streaming anchor body indices must be valid body rows.")
        anchor_worlds = model.body_world.numpy()[indices]
        if np.any((anchor_worlds >= 0) & (anchor_worlds != np.arange(world_count))):
            raise ValueError("Each streaming anchor must belong to its pool's world.")
        solver._validate_reset_array(
            state.body_q, name="body_q", shape=(model.body_count,), dtype=wp.transform, device=device
        )
        self.body_indices = wp.array(indices, dtype=int, device=device)

        particle_world = model.particle_world.numpy()
        if (
            len(particle_world) == 0
            or (world_count > 1 and np.any(particle_world < 0))
            or np.any(particle_world >= world_count)
        ):
            raise ValueError("Streamed particles must belong to valid worlds.")
        # A single standalone world can use the builder's unscoped world ID.
        self.particle_world = (
            model.particle_world
            if np.all(particle_world >= 0)
            else wp.zeros(len(particle_world), dtype=int, device=device)
        )
        particle_world = np.maximum(particle_world, 0)
        flags = model.particle_flags.numpy()
        if (
            np.any((flags & int(newton.ParticleFlags.ACTIVE)) == 0)
            or np.any(flags & int(newton.ParticleFlags.PROXY))
            or solver._mpm_model.deformable_collider_vertex_ranges
        ):
            raise ValueError("Particle streaming requires ordinary active material particles.")
        origins = np.asarray(world_origins, dtype=np.float32)
        heights = np.asarray(spawn_heights, dtype=np.float32)
        if (
            origins.shape != (world_count, 3)
            or heights.shape != (world_count,)
            or not np.isfinite(origins).all()
            or not np.isfinite(heights).all()
        ):
            raise ValueError("World origins and spawn heights must provide one finite entry per world.")
        self.world_origins = wp.array(origins, dtype=wp.vec3, device=device)
        self.window_centers = wp.array(origins[:, :2], dtype=wp.vec2, device=device)

        if surface_mesh.points.device != device or surface_mesh.points.shape[0] == 0:
            raise ValueError("The streaming surface mesh must be nonempty and on the particle device.")
        self.surface_mesh = surface_mesh
        points = surface_mesh.points.numpy()
        if not np.isfinite(points).all():
            raise ValueError("Streaming surface mesh vertices must be finite.")
        self.surface_bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        self._ray_z = float(self.surface_bounds[1, 2] + 1.0)
        self._ray_length = float(self.surface_bounds[1, 2] - self.surface_bounds[0, 2] + 2.0)
        if np.any(origins[:, :2] - 0.5 * np.asarray(self.pool_size) < self.surface_bounds[0, :2]) or np.any(
            origins[:, :2] + 0.5 * np.asarray(self.pool_size) > self.surface_bounds[1, :2]
        ):
            raise ValueError("The surface mesh must contain every initial particle pool.")

        template = model.particle_q.numpy().copy()
        template[:, 2] -= heights[particle_world]
        self.template_positions = wp.array(template, dtype=wp.vec3, device=device)
        self.reset_positions = wp.clone(model.particle_q)
        self.particle_is_dynamic = wp.ones(model.particle_count, dtype=int, device=device)
        self.dynamic_mass = wp.clone(model.particle_mass)
        mass = self.dynamic_mass.numpy()
        if np.any(mass <= 0.0) or not np.isfinite(mass).all():
            raise ValueError("Initial particle masses must be positive and finite.")
        self._dynamic_inv_mass = wp.clone(model.particle_inv_mass)
        self.initial_plastic_volume_ratio = wp.clone(state.mpm.particle_Jp)
        self.reset_requested = wp.ones(world_count, dtype=int, device=device)
        self.recycled = wp.zeros(model.particle_count, dtype=int, device=device)
        self._counts = wp.zeros(3, dtype=int, device=device)
        # Mass changes and recycles control GPU branches; surface misses accumulate until check_status().
        self.mass_change_count = self._counts[:1]
        self.surface_miss_count = self._counts[1:2]
        self.recycle_count = self._counts[2:]

    def update(self, state: newton.State) -> None:
        """Move the pools to their anchors and recycle particles before one solver step.

        Args:
            state: State holding the anchor ``body_q`` and the particle kinematics to update.
        """
        model = self.model
        bounds = self.surface_bounds
        wp.launch(
            _update_window_center,
            dim=model.world_count,
            inputs=[
                state.body_q,
                self.body_indices,
                self.world_origins,
                wp.vec2(*((bounds[0, :2] + bounds[1, :2]) / 2)),
                wp.vec2(*(bounds[1, :2] - bounds[0, :2])),
                wp.vec2(*self.pool_size),
                self.tile_size,
            ],
            outputs=[self.window_centers],
            device=model.device,
        )
        self.mass_change_count.zero_()
        self.recycle_count.zero_()
        material = state.mpm
        wp.launch(
            _recycle_particles,
            dim=model.particle_count,
            inputs=[
                self.particle_world,
                self.window_centers,
                self.template_positions,
                self.reset_requested,
                self.dynamic_mass,
                self.initial_plastic_volume_ratio,
                wp.vec2(*self.window_size),
                wp.vec2(*self.pool_size),
                self.surface_mesh.id,
                self._ray_z,
                self._ray_length,
            ],
            outputs=[
                self.reset_positions,
                self.particle_is_dynamic,
                state.particle_q,
                state.particle_qd,
                model.particle_mass,
                model.particle_inv_mass,
                material.particle_qd_grad,
                material.particle_elastic_strain,
                material.particle_Jp,
                material.particle_stress,
                material.particle_transform,
                self.recycled,
                self.mass_change_count,
                self.surface_miss_count,
                self.recycle_count,
            ],
            device=model.device,
        )
        self.reset_requested.zero_()
        wp.capture_if(self.mass_change_count, self._refresh_particle_mass)
        if self.solver._particle_activity is not None:
            wp.capture_if(
                self.recycle_count, lambda: self.solver._particle_activity.reset_particles(state, self.recycled)
            )

    def post_step(self, state: newton.State) -> None:
        """Remove numerical drift of the kinematic ring from the step's output state."""
        wp.launch(
            _pin_ring_particles,
            dim=self.model.particle_count,
            inputs=[self.particle_is_dynamic, self.reset_positions, state.particle_q, state.particle_qd],
            device=self.model.device,
        )

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Request reinitialization of the selected worlds' pools at the next :meth:`update`.

        Args:
            world_mask: Boolean mask with at least ``world_count`` entries; None selects every world.
        """
        if world_mask is None:
            self.reset_requested.fill_(1)
        else:
            self.solver._validate_reset_array(
                world_mask,
                name="world_mask",
                shape=(self.model.world_count,),
                dtype=wp.bool,
                device=self.model.device,
            )
            wp.launch(
                _mark_resets,
                dim=self.model.world_count,
                inputs=[world_mask, self.reset_requested],
                device=self.model.device,
            )

    def check_status(self) -> None:
        """Raise if a recycled particle missed the surface mesh since the last check. Reads from the device."""
        if self.surface_miss_count.numpy()[0]:
            self.surface_miss_count.zero_()
            raise ValueError("Recycled particles left the streaming surface mesh.")

    def _restore_mass(self) -> None:
        """Restore the model's masses when streaming is disabled."""
        self.model.particle_mass.assign(self.dynamic_mass)
        self.model.particle_inv_mass.assign(self._dynamic_inv_mass)
        self._refresh_particle_mass()

    def _refresh_particle_mass(self) -> None:
        """Synchronize changed masses and invalidate warm starts inside the GPU branch."""
        # Radius, volume, collider membership and constitutive parameters are unchanged.
        derived = self.solver._mpm_model
        wp.launch(
            _refresh_density,
            dim=self.model.particle_count,
            inputs=[self.model.particle_mass, derived.particle_volume],
            outputs=[derived.particle_density],
            device=self.model.device,
        )
        activity = self.solver._particle_activity
        if activity is not None:
            wp.launch(
                _refresh_density,
                dim=self.model.particle_count,
                inputs=[self.model.particle_mass, derived.particle_volume],
                outputs=[activity.base_density],
                device=self.model.device,
            )
        # A full solver reset also clears sparse-grid error status and is forbidden during capture.
        # Recycling only invalidates warm starts and collider history.
        self.solver._clear_reset_warmstarts(None, (None, None))
