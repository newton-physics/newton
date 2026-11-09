# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-resident sleeping patches for implicit MPM."""

import warp as wp

import newton

from .rasterized_collisions import Collider

_MAX_SWEEP_CELLS = wp.constant(4096)


@wp.struct
class _PatchTable:
    positions: wp.array[wp.vec3]
    worlds: wp.array[int]
    slots: wp.array[int]
    particle_slot: wp.array[int]
    requested: wp.array[int]
    urgent: wp.array[int]
    boundary: wp.array[int]
    motion_counts: wp.array[int]
    motion_particles: wp.array[int]
    sweep_fallback: wp.array[int]
    size: float


@wp.func
def _coordinates(position: wp.vec3, size: float):
    return wp.vec3i(
        int(wp.floor(position[0] / size)), int(wp.floor(position[1] / size)), int(wp.floor(position[2] / size))
    )


@wp.func
def _patch_coordinates(table: _PatchTable, particle: int):
    return _coordinates(table.positions[particle], table.size)


@wp.func
def _patch_world(table: _PatchTable, particle: int):
    world = int(0)
    if table.worlds:
        world = table.worlds[particle]
    return world


@wp.func
def _patch_hash(coordinates: wp.vec3i, world: int, mask: int):
    value = (
        wp.uint32(coordinates[0]) * wp.uint32(73856093)
        ^ wp.uint32(coordinates[1]) * wp.uint32(19349663)
        ^ wp.uint32(coordinates[2]) * wp.uint32(83492791)
        ^ wp.uint32(world) * wp.uint32(-1640531535)
    )
    return int(value & wp.uint32(mask))


@wp.func
def _find_patch(table: _PatchTable, coordinates: wp.vec3i, world: int):
    mask = table.slots.shape[0] - 1
    slot = _patch_hash(coordinates, world, mask)
    for _ in range(table.slots.shape[0]):
        representative = table.slots[slot]
        if representative < 0:
            return -1
        if _patch_coordinates(table, representative) == coordinates and _patch_world(table, representative) == world:
            return slot
        slot = (slot + 1) & mask
    return -1


@wp.kernel(enable_backward=False)
def _insert_particles(table: _PatchTable, base_flags: wp.array[int]):
    i = wp.tid()
    table.particle_slot[i] = -1
    position = table.positions[i]
    finite = wp.isfinite(position[0]) and wp.isfinite(position[1]) and wp.isfinite(position[2])
    if (base_flags[i] & newton.ParticleFlags.ACTIVE) == 0 or not finite:
        return
    coordinates = _patch_coordinates(table, i)
    world = _patch_world(table, i)
    mask = table.slots.shape[0] - 1
    slot = _patch_hash(coordinates, world, mask)
    for _ in range(table.slots.shape[0]):
        representative = wp.atomic_cas(table.slots, slot, -1, i)
        if representative < 0:
            table.particle_slot[i] = slot
            return
        if _patch_coordinates(table, representative) == coordinates and _patch_world(table, representative) == world:
            table.particle_slot[i] = slot
            return
        slot = (slot + 1) & mask


@wp.kernel(enable_backward=False)
def _check_patch_keys(
    table: _PatchTable,
    base_flags: wp.array[int],
    cells: wp.array[wp.vec3i],
    worlds: wp.array[int],
    valid: wp.array[int],
    changed: wp.array[int],
):
    """Invalidate only for exact cell, world, or insertion-eligibility changes."""
    i = wp.tid()
    position = table.positions[i]
    finite = wp.isfinite(position[0]) and wp.isfinite(position[1]) and wp.isfinite(position[2])
    active = int((base_flags[i] & newton.ParticleFlags.ACTIVE) != 0 and finite)
    cell = wp.vec3i(0)
    if active != 0:
        cell = _patch_coordinates(table, i)
    world = _patch_world(table, i)
    if valid[0] == 0 or int(table.particle_slot[i] >= 0) != active or not (cells[i] == cell) or worlds[i] != world:
        wp.atomic_max(changed, 0, 1)
    cells[i] = cell
    worlds[i] = world


@wp.func
def _request_swept_patches(table: _PatchTable, particle: int, displacement: wp.vec3):
    start = table.positions[particle] / table.size
    travel = displacement / table.size
    end = (table.positions[particle] + displacement) / table.size
    world = _patch_world(table, particle)
    fallback_world = int(0)
    if table.worlds:
        fallback_world = world + 1
    # Match collider sweeps' bounded empty-space traversal. Large or invalid
    # sweeps conservatively wake occupied patches in the same world instead.
    bounded = wp.isfinite(wp.length_sq(start) + wp.length_sq(end) + wp.length_sq(travel))
    bounded = bounded and wp.max(wp.max(wp.abs(start[0]), wp.abs(start[1])), wp.abs(start[2])) < 1.0e9
    bounded = bounded and wp.max(wp.max(wp.abs(end[0]), wp.abs(end[1])), wp.abs(end[2])) < 1.0e9
    if not bounded:
        wp.atomic_max(table.sweep_fallback, fallback_world, 1)
        return
    coordinates = _patch_coordinates(table, particle)
    target_coordinates = _coordinates(table.positions[particle] + displacement, table.size)
    distance = (
        float(wp.abs(target_coordinates[0] - coordinates[0]))
        + float(wp.abs(target_coordinates[1] - coordinates[1]))
        + float(wp.abs(target_coordinates[2] - coordinates[2]))
    )
    if distance > float(_MAX_SWEEP_CELLS):
        wp.atomic_max(table.sweep_fallback, fallback_world, 1)
        return
    step = wp.vec3i(0)
    crossing = wp.vec3(1.0e30)
    interval = wp.vec3(1.0e30)
    for axis in range(3):
        if travel[axis] != 0.0:
            step[axis] = wp.where(travel[axis] > 0.0, 1, -1)
            interval[axis] = wp.abs(1.0 / travel[axis])
            crossing[axis] = (float(coordinates[axis] + wp.where(step[axis] > 0, 1, 0)) - start[axis]) / travel[axis]
    crossings = int(0)
    while not (coordinates == target_coordinates) and crossings < _MAX_SWEEP_CELLS:
        # Near-tied crossings at the endpoint must not advance a completed axis.
        for axis in range(3):
            if coordinates[axis] == target_coordinates[axis]:
                crossing[axis] = 1.0e30
        axis = int(0)
        if crossing[1] < crossing[axis]:
            axis = 1
        if crossing[2] < crossing[axis]:
            axis = 2
        coordinates[axis] += step[axis]
        crossing[axis] += interval[axis]
        crossings += 1
        target = _find_patch(table, coordinates, world)
        if target >= 0:
            wp.atomic_max(table.urgent, target, 1)
            wp.atomic_max(table.requested, target, 1)
    if not (coordinates == target_coordinates):
        wp.atomic_max(table.sweep_fallback, fallback_world, 1)


@wp.kernel(enable_backward=False)
def _advance_planning(
    dt: float,
    interval: int,
    countdown: wp.array[int],
    elapsed: wp.array[float],
    planning_dt: wp.array[float],
    due: wp.array[int],
):
    elapsed[0] += dt
    due[0] = 0
    if countdown[0] == 0:
        due[0] = 1
        planning_dt[0] = elapsed[0]
        elapsed[0] = 0.0
        countdown[0] = interval - 1
    else:
        countdown[0] -= 1


@wp.kernel(enable_backward=False)
def _request_particles(
    table: _PatchTable,
    planning_due: wp.array[int],
    planning_dt: wp.array[float],
    activity: wp.array[int],
    velocities: wp.array[wp.vec3],
    gradients: wp.array[wp.mat33],
    stresses: wp.array[wp.mat33],
    forces: wp.array[wp.vec3],
    base_material_flags: wp.array[int],
    base_density: wp.array[float],
    previous_stress: wp.array[wp.mat33],
    stress_filter_valid: wp.array[int],
    quiet_time: wp.array[float],
    stress_elapsed: wp.array[float],
    dt: float,
    sleep_time: float,
    velocity_threshold: float,
    strain_threshold: float,
    stress_threshold: float,
    stress_filter_time: float,
    stress_min_strain_rate: float,
    lookahead: float,
    travel_padding: float,
    min_moving_particles: int,
    spill_velocity_threshold: float,
):
    """Advance quiet timers and combine urgent and scheduled motion requests."""
    i = wp.tid()
    planning = planning_due[0] != 0
    if planning and min_moving_particles > 1:
        table.motion_particles[i] = 0
    slot = table.particle_slot[i]
    if slot < 0:
        return
    quiet_time[i] = wp.min(quiet_time[i] + dt, sleep_time)
    stress_elapsed[i] += dt
    velocity = velocities[i]
    speed = wp.length(velocity)
    forced = bool(False)
    if forces:
        force_squared = wp.length_sq(forces[i])
        forced = force_squared > 0.0 or not wp.isfinite(force_squared)
    if not planning:
        # Retain awake material between planning passes, while urgent forces
        # and fast particles can wake and sweep into new patches every tick.
        if activity[i] == 2 and base_density[i] > 0.0 and (base_material_flags[i] & newton.ParticleFlags.ACTIVE) != 0:
            wp.atomic_max(table.requested, slot, 1)
        if forced or not wp.isfinite(speed) or speed > spill_velocity_threshold:
            wp.atomic_max(table.requested, slot, 1)
            wp.atomic_max(table.urgent, slot, 1)
            quiet_time[i] = 0.0
            if wp.isfinite(speed) and speed > 1.0e-12:
                displacement = velocity * (wp.max(dt, lookahead) + travel_padding / speed)
                if wp.isfinite(wp.length_sq(displacement)):
                    _request_swept_patches(table, i, displacement)
        return
    raw_stress = stresses[i]
    stress = raw_stress
    old_stress = previous_stress[i]
    finite_stress = wp.isfinite(wp.ddot(raw_stress, raw_stress))
    if stress_filter_time > 0.0:
        if stress_filter_valid[i] == 0 or not wp.isfinite(wp.ddot(old_stress, old_stress)):
            old_stress = raw_stress
        weight = 1.0 - wp.exp(-stress_elapsed[i] / stress_filter_time)
        stress = old_stress + weight * (raw_stress - old_stress)
        stress_filter_valid[i] = wp.where(finite_stress, 1, 0)
    pressure = wp.trace(stress) / 3.0
    old_pressure = wp.trace(old_stress) / 3.0
    identity = wp.identity(n=3, dtype=float)
    shear = stress - pressure * identity
    old_shear = old_stress - old_pressure * identity
    shear_delta = shear - old_shear
    gradient_norm = wp.sqrt(wp.ddot(gradients[i], gradients[i]))
    invalid = not wp.isfinite(speed) or not finite_stress or not wp.isfinite(gradient_norm)
    moving = speed > velocity_threshold or gradient_norm > strain_threshold
    prescribed = base_density[i] <= 0.0 or (base_material_flags[i] & newton.ParticleFlags.ACTIVE) == 0
    bypass = (
        invalid
        or forced
        or gradient_norm > strain_threshold
        or speed > spill_velocity_threshold
        or (moving and prescribed)
    )
    # Stress redistributes when a neighboring patch freezes. Retain the local
    # patch without treating that response as material moving into its neighbors.
    pressure_scale = wp.max(100.0, wp.abs(old_pressure))
    shear_scale = wp.max(100.0, wp.sqrt(wp.ddot(old_shear, old_shear)))
    if stress_filter_time > 0.0:
        pressure_scale = wp.max(pressure_scale, wp.abs(pressure))
        shear_scale = wp.max(shear_scale, wp.sqrt(wp.ddot(shear, shear)))
    changing = (
        wp.abs(pressure - old_pressure) > stress_threshold * pressure_scale
        or wp.sqrt(wp.ddot(shear_delta, shear_delta)) > stress_threshold * shear_scale
    )
    if stress_filter_time > 0.0:
        strain_rate = 0.5 * (gradients[i] + wp.transpose(gradients[i]))
        changing = changing and wp.sqrt(wp.ddot(strain_rate, strain_rate)) > stress_min_strain_rate
    previous_stress[i] = stress
    stress_elapsed[i] = 0.0
    if changing or bypass:
        quiet_time[i] = 0.0
    direct = bypass or (moving and min_moving_particles == 1)
    if direct:
        wp.atomic_max(table.urgent, slot, 1)
        wp.atomic_max(table.requested, slot, 1)
        if speed > 1.0e-12 and wp.isfinite(speed):
            # Anticipate transport into sleeping material without waking an
            # isotropic ring around stationary affine motion or applied forces.
            displacement = velocity * (wp.max(planning_dt[0], lookahead) + travel_padding / speed)
            if not wp.isfinite(wp.length_sq(displacement)) and (forced or speed > spill_velocity_threshold):
                # Retain urgent per-tick transport if the longer planning sweep overflows.
                displacement = velocity * (wp.max(dt, lookahead) + travel_padding / speed)
            if wp.isfinite(wp.length_sq(displacement)):
                _request_swept_patches(table, i, displacement)
    elif moving:
        count = wp.atomic_add(table.motion_counts, slot, 1)
        table.motion_particles[i] = 1
        if count == min_moving_particles - 1:
            wp.atomic_max(table.requested, slot, 1)
            wp.atomic_max(table.urgent, slot, 1)
    if (
        direct
        or changing
        or (
            (base_material_flags[i] & newton.ParticleFlags.ACTIVE) != 0
            and base_density[i] > 0.0
            and quiet_time[i] < sleep_time
        )
    ):
        wp.atomic_max(table.requested, slot, 1)


@wp.kernel(enable_backward=False)
def _request_motion_transport(
    table: _PatchTable,
    planning_due: wp.array[int],
    planning_dt: wp.array[float],
    velocities: wp.array[wp.vec3],
    min_moving_particles: int,
    lookahead: float,
    travel_padding: float,
):
    if planning_due[0] == 0:
        return
    i = wp.tid()
    slot = table.particle_slot[i]
    if slot < 0 or table.motion_particles[i] == 0 or table.motion_counts[slot] < min_moving_particles:
        return
    speed = wp.length(velocities[i])
    if speed > 1.0e-12 and wp.isfinite(speed):
        displacement = velocities[i] * (wp.max(planning_dt[0], lookahead) + travel_padding / speed)
        if wp.isfinite(wp.length_sq(displacement)):
            _request_swept_patches(table, i, displacement)


@wp.kernel(enable_backward=False)
def _mesh_bounds(
    points: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    previous_points: wp.array[wp.vec3],
    dt: float,
    collider_index: int,
    collider: Collider,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    speed: wp.array[float],
    radius: wp.array[float],
):
    i = wp.tid()
    point = points[i]
    previous_point = previous_points[i]
    wp.atomic_min(lower, collider_index, point)
    wp.atomic_max(upper, collider_index, point)
    if velocities:
        wp.atomic_max(speed, collider_index, wp.length(velocities[i]))
    if dt > 0.0:
        wp.atomic_max(speed, collider_index, wp.length(point - previous_point) / dt)
    body = collider.collider_body_index[collider_index]
    if body >= 0:
        com = collider.body_com[body]
        wp.atomic_max(radius, collider_index, wp.max(wp.length(point - com), wp.length(previous_point - com)))
    previous_points[i] = point


@wp.kernel(enable_backward=False)
def _reset_mesh_bounds(
    mutable_meshes: wp.array[int],
    initialize: bool,
    collider: Collider,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    speed: wp.array[float],
    radius: wp.array[float],
    radius_com: wp.array[wp.vec3],
):
    c = wp.tid()
    speed[c] = 0.0
    if initialize or mutable_meshes[c] != 0:
        lower[c] = wp.vec3(wp.inf)
        upper[c] = wp.vec3(-wp.inf)
        radius[c] = 0.0
        radius_com[c] = wp.vec3(0.0)
        body = collider.collider_body_index[c]
        if body >= 0:
            radius_com[c] = collider.body_com[body]


@wp.kernel(enable_backward=False)
def _collider_wake_bounds(
    collider: Collider,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_q_prev: wp.array[wp.transform],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    previous_lower: wp.array[wp.vec3],
    previous_upper: wp.array[wp.vec3],
    mesh_speed: wp.array[float],
    mesh_radius: wp.array[float],
    mesh_radius_com: wp.array[wp.vec3],
    margin: float,
    lookahead: float,
    dt: float,
    patch_size: float,
    world_count: int,
    separate_worlds: bool,
    wake_lower: wp.array[wp.vec3],
    wake_upper: wp.array[wp.vec3],
    wake_mode: wp.array[int],
    fallback_count: wp.array[int],
):
    c = wp.tid()
    wake_mode[c] = 0
    if not wp.isfinite(wp.length_sq(upper[c] - lower[c])):
        return
    swept_lower = wp.min(lower[c], previous_lower[c])
    swept_upper = wp.max(upper[c], previous_upper[c])
    moving = mesh_speed[c] > 1.0e-7 or wp.length(lower[c] - previous_lower[c]) > 1.0e-7
    moving = moving or wp.length(upper[c] - previous_upper[c]) > 1.0e-7
    velocity = wp.vec3(0.0)
    body = collider.collider_body_index[c]
    if body >= 0:
        # Cached local radii enclose arbitrary rigid rotations about the COM.
        com = collider.body_com[body]
        center = wp.transform_point(body_q[body], com)
        previous_center = center
        if body_q_prev:
            previous_center = wp.transform_point(body_q_prev[body], com)
            rotation = wp.transform_get_rotation(body_q[body])
            old_rotation = wp.transform_get_rotation(body_q_prev[body])
            rotation_delta = wp.min(wp.length(rotation - old_rotation), wp.length(rotation + old_rotation))
            moving = moving or rotation_delta > 1.0e-7
            moving = moving or wp.length(center - previous_center) > 1.0e-7
            if dt > 0.0:
                velocity = (center - previous_center) / dt
        if body_qd:
            prescribed_velocity = wp.spatial_top(body_qd[body])
            if wp.length(prescribed_velocity) > 1.0e-7:
                velocity = prescribed_velocity
                moving = True
            moving = moving or wp.length(wp.spatial_bottom(body_qd[body])) > 1.0e-7
        predicted_center = center + wp.max(dt, lookahead) * velocity
        if not wp.isfinite(wp.length_sq(center) + wp.length_sq(previous_center) + wp.length_sq(predicted_center)):
            # Nonfinite transforms conservatively wake every occupied patch.
            wake_mode[c] = 3
            wp.atomic_add(fallback_count, 0, 1)
            return
        # Changing mass properties may move the COM without rebuilding the
        # rigid mesh. Keep the cached enclosing radius valid about that COM.
        radius = mesh_radius[c] + wp.length(com - mesh_radius_com[c])
        if not wp.isfinite(radius):
            wake_mode[c] = 3
            wp.atomic_add(fallback_count, 0, 1)
            return
        swept_lower = wp.min(wp.min(center, previous_center), predicted_center) - wp.vec3(radius)
        swept_upper = wp.max(wp.max(center, previous_center), predicted_center) + wp.vec3(radius)
    if not moving:
        return
    padding = margin + mesh_speed[c] * wp.max(dt, lookahead) + wp.max(0.0, collider.collider_max_thickness[c])
    swept_lower -= wp.vec3(padding)
    swept_upper += wp.vec3(padding)
    wake_lower[c] = swept_lower
    wake_upper[c] = swept_upper
    # Bound the total 3D hash probes, including shared colliders in many worlds.
    # Large sweeps use the occupied-patch fallback instead of traversing empty space.
    lo = swept_lower / patch_size
    hi = swept_upper / patch_size
    span = wp.vec3(
        wp.floor(hi[0]) - wp.floor(lo[0]) + 2.0,
        wp.floor(hi[1]) - wp.floor(lo[1]) + 2.0,
        wp.floor(hi[2]) - wp.floor(lo[2]) + 2.0,
    )
    worlds = int(1)
    if separate_worlds and collider.collider_world[c] < 0:
        worlds = world_count + 1
    bounded = wp.isfinite(wp.length_sq(lo) + wp.length_sq(hi))
    bounded = bounded and wp.max(wp.max(wp.abs(lo[0]), wp.abs(lo[1])), wp.abs(lo[2])) < 1.0e9
    bounded = bounded and wp.max(wp.max(wp.abs(hi[0]), wp.abs(hi[1])), wp.abs(hi[2])) < 1.0e9
    bounded = bounded and span[0] * span[1] * span[2] * float(worlds) <= 4096.0
    wake_mode[c] = wp.where(bounded, 1, 2)
    if not bounded:
        wp.atomic_add(fallback_count, 0, 1)


@wp.func
def _overlaps_patch(patch_lower: wp.vec3, patch_size: float, lower: wp.vec3, upper: wp.vec3):
    patch_upper = patch_lower + wp.vec3(patch_size)
    return (
        patch_upper[0] >= lower[0]
        and patch_lower[0] <= upper[0]
        and patch_upper[1] >= lower[1]
        and patch_lower[1] <= upper[1]
        and patch_upper[2] >= lower[2]
        and patch_lower[2] <= upper[2]
    )


@wp.kernel(enable_backward=False)
def _request_collider_regions(
    table: _PatchTable,
    collider: Collider,
    wake_lower: wp.array[wp.vec3],
    wake_upper: wp.array[wp.vec3],
    wake_mode: wp.array[int],
):
    c, world_index = wp.tid()
    if wake_mode[c] != 1:
        return
    world = int(0)
    if table.worlds:
        world = world_index - 1
        if collider.collider_world[c] >= 0 and collider.collider_world[c] != world:
            return
    lower = _coordinates(wake_lower[c], table.size) - wp.vec3i(1)
    upper = _coordinates(wake_upper[c], table.size)
    for x in range(lower[0], upper[0] + 1):
        for y in range(lower[1], upper[1] + 1):
            for z in range(lower[2], upper[2] + 1):
                coordinates = wp.vec3i(x, y, z)
                if _overlaps_patch(wp.vec3(coordinates) * table.size, table.size, wake_lower[c], wake_upper[c]):
                    slot = _find_patch(table, coordinates, world)
                    if slot >= 0:
                        wp.atomic_max(table.requested, slot, 1)
                        wp.atomic_max(table.urgent, slot, 1)


@wp.kernel(enable_backward=False)
def _request_colliders_fallback(
    table: _PatchTable,
    collider: Collider,
    wake_lower: wp.array[wp.vec3],
    wake_upper: wp.array[wp.vec3],
    wake_mode: wp.array[int],
    fallback_count: wp.array[int],
):
    if fallback_count[0] == 0:
        return
    i = wp.tid()
    slot = table.particle_slot[i]
    if slot < 0 or table.slots[slot] != i:
        return
    patch_lower = wp.vec3(_patch_coordinates(table, i)) * table.size
    for c in range(collider.collider_mesh.shape[0]):
        if wake_mode[c] < 2:
            continue
        if table.worlds and collider.collider_world[c] >= 0 and collider.collider_world[c] != table.worlds[i]:
            continue
        if wake_mode[c] == 3 or _overlaps_patch(patch_lower, table.size, wake_lower[c], wake_upper[c]):
            wp.atomic_max(table.requested, slot, 1)
            wp.atomic_max(table.urgent, slot, 1)
            return


@wp.kernel(enable_backward=False)
def _dilate_patches(table: _PatchTable, source: wp.array[int], destination: wp.array[int]):
    i = wp.tid()
    slot = table.particle_slot[i]
    if slot < 0 or table.slots[slot] != i:
        return
    coordinates = _patch_coordinates(table, i)
    world = _patch_world(table, i)
    fallback_world = int(0)
    if table.worlds:
        fallback_world = world + 1
    if table.sweep_fallback[fallback_world] != 0:
        source[slot] = 1
        table.urgent[slot] = 1
        destination[slot] = 1
        return
    requested = int(0)
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                neighbor = _find_patch(table, coordinates + wp.vec3i(x, y, z), world)
                if neighbor >= 0 and source[neighbor] != 0:
                    requested = 1
    destination[slot] = requested


@wp.kernel(enable_backward=False)
def _classify_particles(
    table: _PatchTable,
    base_flags: wp.array[int],
    base_material_flags: wp.array[int],
    base_density: wp.array[float],
    velocities: wp.array[wp.vec3],
    gradients: wp.array[wp.mat33],
    stresses: wp.array[wp.mat33],
    previous_stress: wp.array[wp.mat33],
    stress_filter_valid: wp.array[int],
    stress_filter_time: float,
    quiet_time: wp.array[float],
    stress_elapsed: wp.array[float],
    flags: wp.array[int],
    material_flags: wp.array[int],
    density: wp.array[float],
    advect_flags: wp.array[int],
    transfer_velocity: wp.array[wp.vec3],
    transfer_gradient: wp.array[wp.mat33],
    activity: wp.array[int],
):
    i = wp.tid()
    active = int(newton.ParticleFlags.ACTIVE)
    previous_mode = activity[i]
    mode = int(0)
    if (base_flags[i] & active) != 0:
        mode = 2
        slot = table.particle_slot[i]
        if slot >= 0 and base_density[i] > 0.0 and (base_material_flags[i] & active) != 0:
            mode = wp.where(table.requested[slot] != 0, 2, wp.where(table.boundary[slot] != 0, 1, 0))
            if table.urgent[slot] != 0:
                quiet_time[i] = 0.0
    activity[i] = mode
    if stress_filter_time > 0.0 and mode != previous_mode:
        previous_stress[i] = stresses[i]
        stress_filter_valid[i] = wp.where(wp.isfinite(wp.ddot(stresses[i], stresses[i])), 1, 0)
        stress_elapsed[i] = 0.0
    flags[i] = wp.where(mode > 0, base_flags[i], base_flags[i] & ~active)
    # Frozen support keeps its prestress and stiffness in the material assembly.
    # Zero density and the advection mask preserve its constitutive history.
    material_flags[i] = wp.where(mode > 0, base_material_flags[i], base_material_flags[i] & ~active)
    advect_flags[i] = wp.where(mode == 2, base_flags[i], base_flags[i] & ~active)
    density[i] = wp.where(mode == 1, 0.0, base_density[i])
    transfer_velocity[i] = wp.where(mode == 1, wp.vec3(0.0), velocities[i])
    transfer_gradient[i] = wp.where(mode == 1, wp.mat33(0.0), gradients[i])


@wp.kernel(enable_backward=False)
def _initialize_activity(flags: wp.array[int], activity: wp.array[int]):
    i = wp.tid()
    activity[i] = wp.where((flags[i] & newton.ParticleFlags.ACTIVE) != 0, 2, 0)


@wp.kernel(enable_backward=False)
def _reset_activity(
    worlds: wp.array[int],
    world_mask: wp.array[wp.bool],
    world_count: int,
    base_flags: wp.array[int],
    base_material_flags: wp.array[int],
    base_density: wp.array[float],
    flags: wp.array[int],
    material_flags: wp.array[int],
    density: wp.array[float],
    advect_flags: wp.array[int],
    quiet_time: wp.array[float],
    stress: wp.array[wp.mat33],
    previous_stress: wp.array[wp.mat33],
    stress_filter_valid: wp.array[int],
    stress_elapsed: wp.array[float],
    activity: wp.array[int],
):
    i = wp.tid()
    world = worlds[i]
    if world_mask and not world_mask[wp.where(world >= 0, world, world_count)]:
        return
    flags[i] = base_flags[i]
    material_flags[i] = base_material_flags[i]
    density[i] = base_density[i]
    advect_flags[i] = base_flags[i]
    quiet_time[i] = 0.0
    previous_stress[i] = stress[i]
    stress_filter_valid[i] = wp.where(wp.isfinite(wp.ddot(stress[i], stress[i])), 1, 0)
    stress_elapsed[i] = 0.0
    activity[i] = wp.where((base_flags[i] & newton.ParticleFlags.ACTIVE) != 0, 2, 0)


class ParticleActivity:
    """Own solver-local sleep masks, sparse patch requests, and quiet timers."""

    def __init__(
        self,
        model,
        mpm_model,
        *,
        voxel_size,
        patch_size,
        velocity_threshold,
        strain_threshold,
        stress_threshold,
        sleep_time,
        collider_margin,
        collider_lookahead,
        stress_filter_time=0.0,
        stress_min_strain_rate=0.02,
        planning_interval=1,
        min_moving_particles=1,
        spill_velocity_threshold=0.3,
    ):
        self.model = model
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.velocity_threshold = velocity_threshold
        self.strain_threshold = strain_threshold
        self.stress_threshold = stress_threshold
        self.stress_filter_time = stress_filter_time
        self.stress_min_strain_rate = stress_min_strain_rate
        self.planning_interval = planning_interval
        self.min_moving_particles = min_moving_particles
        self.spill_velocity_threshold = spill_velocity_threshold
        self.sleep_time = sleep_time
        self.collider_margin = collider_margin
        self.collider_lookahead = collider_lookahead
        count = model.particle_count
        device = model.device
        capacity = 1 << max(1, (2 * count - 1).bit_length())
        self.table = _PatchTable()
        self.table.size = patch_size
        self.table.slots = wp.full(capacity, -1, dtype=int, device=device)
        self.table.particle_slot = wp.full(count, -1, dtype=int, device=device)
        for name in ("requested", "urgent", "boundary"):
            setattr(self.table, name, wp.zeros(capacity, dtype=int, device=device))
        self.table.motion_counts = wp.zeros(capacity if min_moving_particles > 1 else 0, dtype=int, device=device)
        self.table.motion_particles = wp.zeros(count if min_moving_particles > 1 else 0, dtype=int, device=device)
        self.table.sweep_fallback = wp.zeros(model.world_count + 1, dtype=int, device=device)
        self._patch_cells = wp.zeros(count, dtype=wp.vec3i, device=device)
        self._patch_worlds = wp.zeros(count, dtype=int, device=device)
        self._patch_hash_valid = wp.zeros(1, dtype=int, device=device)
        self._patch_hash_changed = wp.zeros(1, dtype=int, device=device)
        self.quiet_time = wp.zeros(count, dtype=float, device=device)
        self.stress_elapsed = wp.zeros(count, dtype=float, device=device)
        self.planning_countdown = wp.zeros(1, dtype=int, device=device)
        self.planning_elapsed = wp.zeros(1, dtype=float, device=device)
        self.planning_dt = wp.zeros(1, dtype=float, device=device)
        self.planning_due = wp.zeros(1, dtype=int, device=device)
        self.previous_stress = wp.zeros(count, dtype=wp.mat33, device=device)
        self.stress_filter_valid = wp.zeros(count, dtype=int, device=device)
        self.transfer_velocity = wp.empty(count, dtype=wp.vec3, device=device)
        self.transfer_gradient = wp.empty(count, dtype=wp.mat33, device=device)
        self.advect_flags = wp.empty(count, dtype=int, device=device)
        self.activity = wp.zeros(count, dtype=int, device=device)
        self.refresh(mpm_model)

    def refresh(self, mpm_model):
        """Refresh eligibility after particle properties or colliders change."""
        self.mpm_model = mpm_model
        self.base_flags = wp.clone(mpm_model.particle_flags)
        self.base_material_flags = wp.clone(mpm_model.material_particle_flags)
        self.base_density = wp.clone(mpm_model.particle_density)
        self.advect_flags.assign(self.base_flags)
        wp.launch(
            _initialize_activity,
            dim=self.model.particle_count,
            inputs=[self.base_flags, self.activity],
            device=self.model.device,
        )
        self.quiet_time.zero_()
        self.stress_elapsed.zero_()
        self.planning_countdown.zero_()
        self.planning_elapsed.zero_()
        self.previous_stress.zero_()
        self.stress_filter_valid.zero_()
        self._patch_hash_valid.zero_()
        count = mpm_model.collider.collider_mesh.shape[0]
        device = self.model.device
        self.lower = wp.empty(count, dtype=wp.vec3, device=device)
        self.upper = wp.empty(count, dtype=wp.vec3, device=device)
        self.previous_lower = wp.empty_like(self.lower)
        self.previous_upper = wp.empty_like(self.upper)
        self.mesh_speed = wp.zeros(count, dtype=float, device=device)
        self.mesh_radius = wp.zeros(count, dtype=float, device=device)
        self.mesh_radius_com = wp.zeros(count, dtype=wp.vec3, device=device)
        # Only model-generated meshes are immutable rigid geometry. In
        # particular, a custom mesh with zero vertex velocities may still edit
        # its points between substeps and must retain the deformation scan.
        self._generated_meshes = getattr(mpm_model, "_collider_generated", (False,) * count)
        self._mutable_meshes = wp.array(
            [not generated for generated in self._generated_meshes], dtype=int, device=device
        )
        self.wake_lower = wp.empty_like(self.lower)
        self.wake_upper = wp.empty_like(self.upper)
        self.wake_mode = wp.zeros(count, dtype=int, device=device)
        self.collider_fallback_count = wp.zeros(1, dtype=int, device=device)
        self._previous_mesh_points = [wp.clone(mesh.points) for mesh in mpm_model._collider_meshes]
        self._update_mesh_bounds(0.0, initialize=True)
        self.previous_lower.assign(self.lower)
        self.previous_upper.assign(self.upper)

    def restore(self):
        """Restore the caller's eligible transfer and material properties."""
        self.mpm_model.particle_flags.assign(self.base_flags)
        self.mpm_model.material_particle_flags.assign(self.base_material_flags)
        self.mpm_model.particle_density.assign(self.base_density)

    def reset(self, state, world_mask=None):
        """Restore selected worlds and restart their quiet grace periods."""
        wp.launch(
            _reset_activity,
            dim=self.model.particle_count,
            inputs=[
                self.model.particle_world,
                world_mask,
                self.model.world_count,
                self.base_flags,
                self.base_material_flags,
                self.base_density,
                self.mpm_model.particle_flags,
                self.mpm_model.material_particle_flags,
                self.mpm_model.particle_density,
                self.advect_flags,
                self.quiet_time,
                state.mpm.particle_stress,
                self.previous_stress,
                self.stress_filter_valid,
                self.stress_elapsed,
                self.activity,
            ],
            device=self.model.device,
        )
        if world_mask is None:
            self.planning_countdown.zero_()
            self.planning_elapsed.zero_()
        self._patch_hash_valid.zero_()

    def _rebuild_patch_table(self):
        self.table.slots.fill_(-1)
        wp.launch(
            _insert_particles,
            dim=self.model.particle_count,
            inputs=[self.table, self.base_flags],
            device=self.model.device,
        )
        self._patch_hash_valid.fill_(1)

    def _update_mesh_bounds(self, dt, initialize=False):
        wp.launch(
            _reset_mesh_bounds,
            dim=self.lower.shape[0],
            inputs=[
                self._mutable_meshes,
                initialize,
                self.mpm_model.collider,
                self.lower,
                self.upper,
                self.mesh_speed,
                self.mesh_radius,
                self.mesh_radius_com,
            ],
            device=self.model.device,
        )
        for index, mesh in enumerate(self.mpm_model._collider_meshes):
            if not initialize and self._generated_meshes[index]:
                continue
            wp.launch(
                _mesh_bounds,
                dim=mesh.points.shape[0],
                inputs=[
                    mesh.points,
                    mesh.velocities,
                    self._previous_mesh_points[index],
                    dt,
                    index,
                    self.mpm_model.collider,
                    self.lower,
                    self.upper,
                    self.mesh_speed,
                    self.mesh_radius,
                ],
                device=self.model.device,
            )

    def update(self, state, dt, collider, body_q_prev, separate_worlds, *, collider_velocity_mode="forward"):
        """Classify current patches without changing caller-owned model arrays."""
        table = self.table
        table.positions = state.particle_q
        table.worlds = self.model.particle_world if separate_worlds else None
        self._patch_hash_changed.zero_()
        wp.launch(
            _check_patch_keys,
            dim=self.model.particle_count,
            inputs=[
                table,
                self.base_flags,
                self._patch_cells,
                self._patch_worlds,
                self._patch_hash_valid,
                self._patch_hash_changed,
            ],
            device=self.model.device,
        )
        if self.model.device.is_capturing and wp.is_conditional_graph_supported():
            wp.capture_if(self._patch_hash_changed, self._rebuild_patch_table)
        else:
            # capture_if reads its device condition back in eager execution.
            self._rebuild_patch_table()
        table.requested.zero_()
        table.urgent.zero_()
        table.sweep_fallback.zero_()
        if self.min_moving_particles > 1:
            table.motion_counts.zero_()
        self._update_mesh_bounds(dt)
        count = self.model.particle_count
        device = self.model.device
        wp.launch(
            _advance_planning,
            dim=1,
            inputs=[
                dt,
                self.planning_interval,
                self.planning_countdown,
                self.planning_elapsed,
                self.planning_dt,
                self.planning_due,
            ],
            device=device,
        )
        wp.launch(
            _request_particles,
            dim=count,
            inputs=[
                table,
                self.planning_due,
                self.planning_dt,
                self.activity,
                state.particle_qd,
                state.mpm.particle_qd_grad,
                state.mpm.particle_stress,
                state.particle_f,
                self.base_material_flags,
                self.base_density,
                self.previous_stress,
                self.stress_filter_valid,
                self.quiet_time,
                self.stress_elapsed,
                dt,
                self.sleep_time,
                self.velocity_threshold,
                self.strain_threshold,
                self.stress_threshold,
                self.stress_filter_time,
                self.stress_min_strain_rate,
                self.collider_lookahead,
                2.0 * self.voxel_size,
                self.min_moving_particles,
                self.spill_velocity_threshold,
            ],
            device=device,
        )
        if self.min_moving_particles > 1:
            wp.launch(
                _request_motion_transport,
                dim=count,
                inputs=[
                    table,
                    self.planning_due,
                    self.planning_dt,
                    state.particle_qd,
                    self.min_moving_particles,
                    self.collider_lookahead,
                    2.0 * self.voxel_size,
                ],
                device=device,
            )
        self.collider_fallback_count.zero_()
        collider_count = collider.collider_mesh.shape[0]
        wp.launch(
            _collider_wake_bounds,
            dim=collider_count,
            inputs=[
                collider,
                state.body_q,
                state.body_qd if collider_velocity_mode == "forward" else None,
                body_q_prev,
                self.lower,
                self.upper,
                self.previous_lower,
                self.previous_upper,
                self.mesh_speed,
                self.mesh_radius,
                self.mesh_radius_com,
                self.collider_margin,
                self.collider_lookahead,
                dt,
                self.patch_size,
                self.model.world_count,
                separate_worlds,
                self.wake_lower,
                self.wake_upper,
                self.wake_mode,
                self.collider_fallback_count,
            ],
            device=device,
        )
        wp.launch(
            _request_collider_regions,
            dim=(collider_count, self.model.world_count + 1 if separate_worlds else 1),
            inputs=[table, collider, self.wake_lower, self.wake_upper, self.wake_mode],
            device=device,
        )
        wp.launch(
            _request_colliders_fallback,
            dim=count,
            inputs=[table, collider, self.wake_lower, self.wake_upper, self.wake_mode, self.collider_fallback_count],
            device=device,
        )
        wp.launch(
            _dilate_patches,
            dim=count,
            inputs=[table, table.requested, table.boundary],
            device=device,
        )
        wp.launch(
            _classify_particles,
            dim=count,
            inputs=[
                table,
                self.base_flags,
                self.base_material_flags,
                self.base_density,
                state.particle_qd,
                state.mpm.particle_qd_grad,
                state.mpm.particle_stress,
                self.previous_stress,
                self.stress_filter_valid,
                self.stress_filter_time,
                self.quiet_time,
                self.stress_elapsed,
                self.mpm_model.particle_flags,
                self.mpm_model.material_particle_flags,
                self.mpm_model.particle_density,
                self.advect_flags,
                self.transfer_velocity,
                self.transfer_gradient,
                self.activity,
            ],
            device=device,
        )
        self.previous_lower.assign(self.lower)
        self.previous_upper.assign(self.upper)
