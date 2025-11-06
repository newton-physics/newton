import warp as wp

from . import ray_cast


@wp.func
def compute_lighting(
    use_shadows: bool,
    bvh_id: wp.uint64,
    group_roots: wp.array(dtype=wp.int32),
    num_geom_in_bvh: int,
    geom_enabled: wp.array(dtype=int),
    world_id: int,
    light_active: bool,
    light_type: int,
    light_cast_shadow: bool,
    light_position: wp.vec3,
    light_orientation: wp.vec3,
    normal: wp.vec3,
    geom_types: wp.array(dtype=int),
    geom_mesh_indices: wp.array(dtype=int),
    geom_sizes: wp.array(dtype=wp.vec3),
    mesh_ids: wp.array(dtype=wp.uint64),
    geom_positions: wp.array(dtype=wp.vec3),
    geom_orientations: wp.array(dtype=wp.mat33),
    hit_point: wp.vec3,
) -> wp.float32:
    light_contribution = wp.float32(0.0)

    if not light_active:
        return light_contribution

    L = wp.vec3(0.0, 0.0, 0.0)
    dist_to_light = wp.float32(wp.inf)
    attenuation = wp.float32(1.0)

    if light_type == 1:  # directional light
        L = wp.normalize(-light_orientation)
    else:
        to_light = light_position - hit_point
        dist_to_light = wp.length(to_light)
        L = wp.normalize(to_light)
        attenuation = 1.0 / (1.0 + 0.02 * dist_to_light * dist_to_light)
        if light_type == 0:  # spot light
            spot_dir = wp.normalize(light_orientation)
            cos_theta = wp.dot(-L, spot_dir)
            inner = 0.95
            outer = 0.85
            spot_factor = wp.min(1.0, wp.max(0.0, (cos_theta - outer) / (inner - outer)))
            attenuation = attenuation * spot_factor

    ndotl = wp.max(0.0, wp.dot(normal, L))

    if ndotl == 0.0:
        return light_contribution

    visible = wp.float32(1.0)
    shadow_min_visibility = wp.float32(0.3)  # reduce shadow darkness (0: full black, 1: no shadow)

    if use_shadows and light_cast_shadow:
        # Nudge the origin slightly along the surface normal to avoid
        # self-intersection when casting shadow rays
        eps = 1.0e-4
        shadow_origin = hit_point + normal * eps
        # Distance-limited shadows: cap by dist_to_light (for non-directional)
        max_t = wp.float32(dist_to_light - 1.0e-3)
        if light_type == 1:  # directional light
            max_t = wp.float32(1.0e8)

        shadow_hit = ray_cast.first_hit(
            bvh_id,
            group_roots,
            world_id,
            num_geom_in_bvh,
            geom_enabled,
            geom_types,
            geom_mesh_indices,
            geom_sizes,
            mesh_ids,
            geom_positions,
            geom_orientations,
            shadow_origin,
            L,
            max_t,
        )

        if shadow_hit:
            visible = shadow_min_visibility

    return ndotl * attenuation * visible
