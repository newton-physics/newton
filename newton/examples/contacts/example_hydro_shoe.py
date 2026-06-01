# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Foot-Shoe Hydroelastic Contact Simulation
#
# Simulates a foot mesh interacting with a shoe midsole mesh
# using a custom differentiable Ogden hydroelastic contact kernel and MuJoCo.
#
# Command: python -m newton.examples hydro_shoe
#
###########################################################################

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.geometry import HydroelasticSDF
from projects.digital_instron_v2.foundation import FoundationMaterial
from projects.digital_instron_v2.geometry import _load_obj_mesh, compute_grid_neighbors, condition_midsole_mesh
from projects.digital_instron_v2.manifest import load_manifest
from projects.digital_instron_v2.workflow import _load_spring_grid


@dataclass(frozen=True)
class KimHyperfoamMaterial:
    """First-order hyperfoam constants reported by Kim et al. (2026)."""

    name: str
    density_kg_m3: float
    mu1_pa: float
    alpha1: float
    poisson_ratio: float

    @property
    def young_modulus_pa(self) -> float:
        return 2.0 * self.mu1_pa * (1.0 + self.poisson_ratio)

    @property
    def bulk_modulus_pa(self) -> float:
        return self.young_modulus_pa / (3.0 * (1.0 - 2.0 * self.poisson_ratio))

    @property
    def d1_inv_pa(self) -> float:
        return 0.5 * self.bulk_modulus_pa


KIM_HYPERFOAM_MATERIALS = {
    "eva": KimHyperfoamMaterial("EVA", 170.0, 0.148e6, 3.595, 0.20),
    "tpu": KimHyperfoamMaterial("TPU", 240.0, 0.0878e6, 2.110, 0.25),
    "peba": KimHyperfoamMaterial("PEBA", 90.0, 0.085e6, 5.050, 0.30),
}

CONTACT_LAW_CALIBRATED_OGDEN = 0
CONTACT_LAW_KIM_HYPERFOAM = 1
CONTACT_LAW_KIM_LAYERED = 2
PARAM_STIFFNESS_OR_BULK = 0
PARAM_ALPHA = 1
PARAM_LOCK_STRAIN = 2
PARAM_DAMPING = 3
PARAM_DAMPING_POWER = 4
PARAM_CONTACT_LAW = 5
PARAM_LOWER_BULK = 6
PARAM_LOWER_ALPHA = 7
PARAM_UPPER_FRACTION = 8
PARAM_HYDRO_FORCE_SCALE = 9
STATS_LAST_FORCE_N = 0
STATS_PEAK_FORCE_N = 1
STATS_LAST_ELASTIC_ENERGY_J = 2
STATS_PEAK_ELASTIC_ENERGY_J = 3
STATS_LAST_DISSIPATED_ENERGY_J = 4
SURFACE_TOP_AREA = 0
SURFACE_TOP_DISP_AREA = 1
SURFACE_TOP_VEL_AREA = 2
SURFACE_GROUND_AREA = 3
SURFACE_GROUND_DISP_AREA = 4
SURFACE_GROUND_VEL_AREA = 5


def _kim_material_to_foundation(material: KimHyperfoamMaterial, damping_pa_s: float) -> FoundationMaterial:
    return FoundationMaterial(
        stiffness_pa=material.mu1_pa,
        ogden_alpha=material.alpha1,
        lock_strain=0.99,
        damping_pa_s=damping_pa_s,
        damping_power=1.0,
    )


def _kim_pressure_pa(strain: np.ndarray, material: KimHyperfoamMaterial) -> np.ndarray:
    strain = np.clip(strain, 0.0, 0.99)
    power = max(material.alpha1 - 1.0, 0.0)
    return material.bulk_modulus_pa * np.power(strain, power)


def _kim_layered_pressure_pa(
    strain: np.ndarray,
    upper_material: KimHyperfoamMaterial,
    lower_material: KimHyperfoamMaterial,
    upper_fraction: float,
) -> np.ndarray:
    strain = np.clip(strain, 0.0, 0.99)
    upper_fraction = float(np.clip(upper_fraction, 0.0, 1.0))
    lower_fraction = 1.0 - upper_fraction
    if upper_fraction <= 1.0e-6:
        return _kim_pressure_pa(strain, lower_material)
    if lower_fraction <= 1.0e-6:
        return _kim_pressure_pa(strain, upper_material)

    upper_max = _kim_pressure_pa(np.asarray([0.99]), upper_material)[0]
    lower_max = _kim_pressure_pa(np.asarray([0.99]), lower_material)[0]
    lo = np.zeros_like(strain, dtype=np.float64)
    hi = np.full_like(strain, max(float(upper_max), float(lower_max), 1.0), dtype=np.float64)
    upper_power = 1.0 / max(upper_material.alpha1 - 1.0, 1.0e-4)
    lower_power = 1.0 / max(lower_material.alpha1 - 1.0, 1.0e-4)

    for _ in range(32):
        mid = 0.5 * (lo + hi)
        upper_strain = np.power(mid / max(upper_material.bulk_modulus_pa, 1.0e-6), upper_power)
        lower_strain = np.power(mid / max(lower_material.bulk_modulus_pa, 1.0e-6), lower_power)
        total = upper_fraction * upper_strain + lower_fraction * lower_strain
        lo = np.where(total < strain, mid, lo)
        hi = np.where(total >= strain, mid, hi)

    pressure = 0.5 * (lo + hi)
    return np.where(strain <= 1.0e-8, 0.0, pressure)


def _rotate_vec_by_quat(v: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    qv = q_xyzw[:3]
    qw = q_xyzw[3]
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + qw * v)


def _rotation_matrix_xyz_deg(rotation_deg: np.ndarray | list[float]) -> np.ndarray:
    angles = np.asarray(rotation_deg, dtype=np.float64)
    rx, ry, rz = np.deg2rad(angles)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _quat_from_euler_xyz_deg(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([roll, pitch, yaw])
    cx, sx = np.cos(rx / 2.0), np.sin(rx / 2.0)
    cy, sy = np.cos(ry / 2.0), np.sin(ry / 2.0)
    cz, sz = np.cos(rz / 2.0), np.sin(rz / 2.0)

    qx = np.array([sx, 0.0, 0.0, cx])
    qy = np.array([0.0, sy, 0.0, cy])
    qz = np.array([0.0, 0.0, sz, cz])

    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )

    return quat_mul(qz, quat_mul(qy, qx))


@wp.func
def find_nearest_spring(pos: wp.vec3, spring_xy: wp.array[wp.vec2], num_springs: int) -> int:
    """Find the index of the nearest spring in the 2D grid plane."""
    min_dist = float(1e10)
    nearest = int(-1)
    for k in range(num_springs):
        dx = pos[0] - spring_xy[k][0]
        dy = pos[1] - spring_xy[k][1]
        dist = dx * dx + dy * dy
        if dist < min_dist:
            min_dist = dist
            nearest = k
    return nearest


@wp.func
def find_nearest_spring_grid(
    pos: wp.vec3,
    grid_to_spring: wp.array2d[wp.int32],
    min_u: float,
    min_v: float,
    spacing: float,
    num_u: int,
    num_v: int,
    spring_xy: wp.array[wp.vec2],
) -> int:
    u = pos[0]
    v = pos[1]

    # Calculate grid indices
    center_iu = int(wp.round((u - min_u) / spacing))
    center_iv = int(wp.round((v - min_v) / spacing))

    # 1. Check the cell itself
    if center_iu >= 0 and center_iu < num_u and center_iv >= 0 and center_iv < num_v:
        k = grid_to_spring[center_iu, center_iv]
        if k != -1:
            return k

    # 2. Check 3x3 neighborhood (radius 1)
    nearest = int(-1)
    min_dist = float(1e10)
    for du in range(-1, 2):
        for dv in range(-1, 2):
            iu = center_iu + du
            iv = center_iv + dv
            if iu >= 0 and iu < num_u and iv >= 0 and iv < num_v:
                k = grid_to_spring[iu, iv]
                if k != -1:
                    dx = u - spring_xy[k][0]
                    dy = v - spring_xy[k][1]
                    dist = dx * dx + dy * dy
                    if dist < min_dist:
                        min_dist = dist
                        nearest = k
    if nearest != -1:
        return nearest

    # 3. Check 5x5 neighborhood (radius 2)
    for du in range(-2, 3):
        for dv in range(-2, 3):
            # Skip the 3x3 area we already checked
            if wp.abs(du) <= 1 and wp.abs(dv) <= 1:
                continue
            iu = center_iu + du
            iv = center_iv + dv
            if iu >= 0 and iu < num_u and iv >= 0 and iv < num_v:
                k = grid_to_spring[iu, iv]
                if k != -1:
                    dx = u - spring_xy[k][0]
                    dy = v - spring_xy[k][1]
                    dist = dx * dx + dy * dy
                    if dist < min_dist:
                        min_dist = dist
                        nearest = k

    return nearest


@wp.func
def evaluate_elastic_contact_stress(strain: float, params: wp.array[float]) -> float:
    """Evaluate the elastic part of the selected pressure law."""
    alpha = wp.max(params[PARAM_ALPHA], 1.0e-4)
    elastic_stress = float(0.0)
    if int(params[PARAM_CONTACT_LAW]) == CONTACT_LAW_KIM_HYPERFOAM:
        elastic_stress = params[PARAM_STIFFNESS_OR_BULK] * wp.pow(wp.max(strain, 1.0e-8), wp.max(alpha - 1.0, 0.0))
    elif int(params[PARAM_CONTACT_LAW]) == CONTACT_LAW_KIM_LAYERED:
        if strain <= 1.0e-8:
            elastic_stress = 0.0
        else:
            lower_bulk = wp.max(params[PARAM_LOWER_BULK], 1.0e-6)
            lower_alpha = wp.max(params[PARAM_LOWER_ALPHA], 1.0e-4)
            upper_fraction = wp.clamp(params[PARAM_UPPER_FRACTION], 0.0, 1.0)
            lower_fraction = 1.0 - upper_fraction
            if upper_fraction <= 1.0e-6:
                elastic_stress = lower_bulk * wp.pow(wp.max(strain, 1.0e-8), wp.max(lower_alpha - 1.0, 0.0))
            elif lower_fraction <= 1.0e-6:
                elastic_stress = params[PARAM_STIFFNESS_OR_BULK] * wp.pow(
                    wp.max(strain, 1.0e-8), wp.max(alpha - 1.0, 0.0)
                )
            else:
                upper_bulk = wp.max(params[PARAM_STIFFNESS_OR_BULK], 1.0e-6)
                upper_max = upper_bulk * wp.pow(0.99, wp.max(alpha - 1.0, 0.0))
                lower_max = lower_bulk * wp.pow(0.99, wp.max(lower_alpha - 1.0, 0.0))
                lo = float(0.0)
                hi = wp.max(wp.max(upper_max, lower_max), 1.0)
                upper_power = 1.0 / wp.max(alpha - 1.0, 1.0e-4)
                lower_power = 1.0 / wp.max(lower_alpha - 1.0, 1.0e-4)
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    upper_strain = wp.pow(mid / upper_bulk, upper_power)
                    lower_strain = wp.pow(mid / lower_bulk, lower_power)
                    total_strain = upper_fraction * upper_strain + lower_fraction * lower_strain
                    if total_strain < strain:
                        lo = mid
                    else:
                        hi = mid
            elastic_stress = 0.5 * (lo + hi)
    else:
        lock = wp.max(params[PARAM_LOCK_STRAIN], 1.0e-4)
        normalized_strain = wp.min(strain / lock, 0.999)
        elastic_stress = params[PARAM_STIFFNESS_OR_BULK] * (wp.pow(1.0 - normalized_strain, -alpha) - 1.0) / alpha

    return wp.max(elastic_stress, 0.0)


@wp.func
def evaluate_viscous_contact_stress(strain: float, comp_vel: float, params: wp.array[float]) -> float:
    """Evaluate the signed viscous pressure contribution."""
    damping_weight = wp.pow(wp.max(strain, 1.0e-8), wp.max(params[PARAM_DAMPING_POWER], 0.0))
    return params[PARAM_DAMPING] * damping_weight * comp_vel


@wp.func
def evaluate_contact_stress(strain: float, comp_vel: float, params: wp.array[float]) -> float:
    """Evaluate compressive stress from elastic storage plus limited viscous loss."""
    elastic_stress = evaluate_elastic_contact_stress(strain, params)
    viscous_stress = evaluate_viscous_contact_stress(strain, comp_vel, params)
    return wp.max(elastic_stress + viscous_stress, 0.0)


@wp.func
def evaluate_dissipated_energy_rate(strain: float, comp_vel: float, area: float, params: wp.array[float]) -> float:
    """Evaluate non-negative dashpot dissipation rate for one spring cell."""
    if comp_vel == 0.0:
        return 0.0

    elastic_stress = evaluate_elastic_contact_stress(strain, params)
    viscous_stress = evaluate_viscous_contact_stress(strain, comp_vel, params)
    effective_viscous_stress = viscous_stress
    if viscous_stress < -elastic_stress:
        effective_viscous_stress = -elastic_stress

    return wp.max(effective_viscous_stress * comp_vel * area, 0.0)


@wp.func
def top_displacement_for_spring(
    spring: int,
    foot_z: float,
    midsole_z: float,
    top_m: wp.array[float],
    foot_sole_z_m: wp.array[float],
    foot_contact_valid: wp.array[wp.int32],
    slack_length_m: wp.array[float],
) -> float:
    if foot_contact_valid[spring] == 0:
        return 0.0

    slack = wp.max(slack_length_m[spring], 1.0e-6)
    top_world_z = midsole_z + top_m[spring]
    foot_sole_world_z = foot_z + foot_sole_z_m[spring]
    return wp.clamp(top_world_z - foot_sole_world_z, 0.0, slack)


@wp.func
def ground_displacement_for_spring(
    spring: int,
    midsole_z: float,
    bottom_m: wp.array[float],
    slack_length_m: wp.array[float],
) -> float:
    slack = wp.max(slack_length_m[spring], 1.0e-6)
    bottom_world_z = midsole_z + bottom_m[spring]
    return wp.clamp(-bottom_world_z, 0.0, slack)


@wp.func
def displacement_laplacian(
    spring: int,
    center_displacement: float,
    neighbor_displacement_0: float,
    neighbor_displacement_1: float,
    neighbor_displacement_2: float,
    neighbor_displacement_3: float,
    spacing_m: float,
) -> float:
    h2 = wp.max(spacing_m * spacing_m, 1.0e-12)
    return (
        neighbor_displacement_0
        + neighbor_displacement_1
        + neighbor_displacement_2
        + neighbor_displacement_3
        - 4.0 * center_displacement
    ) / h2


@wp.func
def shear_edge_energy(
    center_displacement: float,
    neighbor_displacement: float,
    slack: float,
    spacing_m: float,
    shear_modulus_pa: float,
) -> float:
    shear_layer_stiffness = wp.max(shear_modulus_pa, 0.0) * slack
    gradient = (neighbor_displacement - center_displacement) / wp.max(spacing_m, 1.0e-6)
    area = spacing_m * spacing_m
    return 0.5 * shear_layer_stiffness * gradient * gradient * area


@wp.func
def evaluate_elastic_energy(displacement: float, slack: float, area: float, params: wp.array[float]) -> float:
    """Integrate elastic pressure over compression depth for one spring cell."""
    if displacement <= 0.0:
        return 0.0

    energy = float(0.0)
    steps = int(16)
    for i in range(steps):
        sample = (float(i) + 0.5) / float(steps)
        sample_displacement = displacement * sample
        strain = wp.clamp(sample_displacement / slack, 0.0, 0.99)
        elastic_stress = evaluate_elastic_contact_stress(strain, params)
        energy += elastic_stress * area

    return energy * displacement / float(steps)


@wp.kernel
def apply_non_contact_forces_kernel(
    body_f: wp.array[wp.spatial_vector],
    foot_body_id: int,
    midsole_body_id: int,
    foot_mass: float,
    midsole_mass: float,
    gravity: float,
    ext_force: float,
    kinematic: int,
):
    # Foot body: apply gravity and the vertical test load in dynamic mode.
    if kinematic == 0:
        foot_grav = foot_mass * gravity - ext_force
        body_f[foot_body_id] = wp.spatial_vector(wp.vec3(0.0, 0.0, foot_grav), wp.vec3(0.0))
    else:
        body_f[foot_body_id] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))

    # Midsole body always experiences gravity (since it's dynamic)
    midsole_grav = midsole_mass * gravity
    body_f[midsole_body_id] = wp.spatial_vector(wp.vec3(0.0, 0.0, midsole_grav), wp.vec3(0.0))


@wp.kernel
def accumulate_bottom_hydro_state_kernel(
    points: wp.array[wp.vec3f],
    depths: wp.array[wp.float32],
    shape_pairs: wp.array[wp.vec2i],
    face_count_ptr: wp.array[wp.int32],
    midsole_shape_idx: int,
    ground_shape_idx: int,
    midsole_body_idx: int,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    spring_xy: wp.array[wp.vec2],
    num_springs: int,
    surface_state: wp.array[float],
    # Grid lookup parameters
    grid_to_spring: wp.array2d[wp.int32],
    grid_min_u: float,
    grid_min_v: float,
    grid_spacing: float,
    grid_num_u: int,
    grid_num_v: int,
):
    tid = wp.tid()
    face_count = face_count_ptr[0]
    if tid >= face_count:
        return

    pair = shape_pairs[tid]
    is_midsole_ground = (pair[0] == midsole_shape_idx and pair[1] == ground_shape_idx) or (
        pair[1] == midsole_shape_idx and pair[0] == ground_shape_idx
    )

    if not is_midsole_ground:
        return

    v0 = wp.vec3(points[3 * tid])
    v1 = wp.vec3(points[3 * tid + 1])
    v2 = wp.vec3(points[3 * tid + 2])
    centroid = (v0 + v1 + v2) / 3.0

    e1 = v1 - v0
    e2 = v2 - v0
    n = wp.cross(e1, e2)
    n_sq = wp.dot(n, n)
    if n_sq < 1e-12:
        return

    normal = n / wp.sqrt(n_sq)
    area = wp.sqrt(n_sq) / 2.0
    if area <= 0.0:
        return
    if normal[2] < 0.0:
        normal = -normal

    displacement = -2.0 * depths[tid]
    if displacement <= 0.0:
        return

    nearest = find_nearest_spring_grid(
        centroid,
        grid_to_spring,
        grid_min_u,
        grid_min_v,
        grid_spacing,
        grid_num_u,
        grid_num_v,
        spring_xy,
    )
    if nearest == -1:
        return

    midsole_com = wp.transform_point(body_q[midsole_body_idx], body_com[midsole_body_idx])
    qd_midsole = body_qd[midsole_body_idx]
    v_midsole_c = wp.spatial_top(qd_midsole) + wp.cross(wp.spatial_bottom(qd_midsole), centroid - midsole_com)
    comp_vel = -wp.dot(v_midsole_c, normal)

    wp.atomic_add(surface_state, nearest * 6 + SURFACE_GROUND_AREA, area)
    wp.atomic_add(surface_state, nearest * 6 + SURFACE_GROUND_DISP_AREA, displacement * area)
    wp.atomic_add(surface_state, nearest * 6 + SURFACE_GROUND_VEL_AREA, comp_vel * area)


@wp.kernel
def accumulate_bonded_top_state_kernel(
    top_m: wp.array[float],
    foot_sole_z_m: wp.array[float],
    foot_contact_valid: wp.array[wp.int32],
    slack_length_m: wp.array[float],
    spring_count: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    foot_body_idx: int,
    midsole_body_idx: int,
    top_reference_z: float,
    spacing_m: float,
    surface_state: wp.array[float],
    foot_displacement_out: wp.array[float],
):
    spring = wp.tid()
    if spring >= spring_count:
        return
    if foot_contact_valid[spring] == 0:
        return

    foot_z = body_q[foot_body_idx].p[2]
    foot_vz = wp.spatial_top(body_qd[foot_body_idx])[2]
    midsole_vz = wp.spatial_top(body_qd[midsole_body_idx])[2]

    slack = wp.max(slack_length_m[spring], 1.0e-6)
    top_world_z = top_reference_z + top_m[spring]
    foot_sole_world_z = foot_z + foot_sole_z_m[spring]
    displacement = wp.clamp(top_world_z - foot_sole_world_z, 0.0, slack)
    comp_vel = midsole_vz - foot_vz
    area = spacing_m * spacing_m

    # The top interface is a bonded spring-grid constraint, not an SDF contact surface.
    wp.atomic_add(surface_state, spring * 6 + SURFACE_TOP_AREA, area)
    wp.atomic_add(surface_state, spring * 6 + SURFACE_TOP_DISP_AREA, displacement * area)
    wp.atomic_add(surface_state, spring * 6 + SURFACE_TOP_VEL_AREA, comp_vel * area)
    foot_displacement_out[spring] = displacement


@wp.kernel
def evaluate_bottom_hydroelastic_ogden_kernel(
    points: wp.array[wp.vec3f],  # World-space positions of contact surface triangle vertices (3 per face)
    depths: wp.array[wp.float32],  # Penetration depth at each face centroid
    shape_pairs: wp.array[wp.vec2i],  # Shape pair indices (shape_a, shape_b) for each face
    face_count_ptr: wp.array[wp.int32],  # Active face count
    midsole_shape_idx: int,
    ground_shape_idx: int,
    midsole_body_idx: int,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    spring_xy: wp.array[wp.vec2],
    spring_slack: wp.array[float],
    num_springs: int,
    coupled_surface_state: wp.array[float],
    params: wp.array[float],
    body_f: wp.array[wp.spatial_vector],  # Output accumulated body forces
    wrench_out: wp.array[float],  # Output accumulated wrench (midsole-ground)
    energy_out: wp.array[float],
    dissipated_energy_total_out: wp.array[float],
    sim_dt: float,
    ground_displacement_out: wp.array[float],
    ground_pressure_kpa_out: wp.array[float],
    stack_displacement_out: wp.array[float],
    stack_pressure_kpa_out: wp.array[float],
    # Grid lookup parameters
    grid_to_spring: wp.array2d[wp.int32],
    grid_min_u: float,
    grid_min_v: float,
    grid_spacing: float,
    grid_num_u: int,
    grid_num_v: int,
):
    tid = wp.tid()
    face_count = face_count_ptr[0]
    if tid >= face_count:
        return

    # Filter for bottom hydroelastic contact only. The top interface is applied by
    # apply_bonded_top_forces_kernel() from the spring-grid state.
    pair = shape_pairs[tid]
    is_midsole_ground = (pair[0] == midsole_shape_idx and pair[1] == ground_shape_idx) or (
        pair[1] == midsole_shape_idx and pair[0] == ground_shape_idx
    )

    if not is_midsole_ground:
        return

    # 1. Retrieve triangle vertices and calculate centroid, normal, and area
    v0 = wp.vec3(points[3 * tid])
    v1 = wp.vec3(points[3 * tid + 1])
    v2 = wp.vec3(points[3 * tid + 2])

    centroid = (v0 + v1 + v2) / 3.0

    e1 = v1 - v0
    e2 = v2 - v0
    n = wp.cross(e1, e2)
    n_sq = wp.dot(n, n)
    if n_sq < 1e-12:
        return

    normal = n / wp.sqrt(n_sq)
    area = wp.sqrt(n_sq) / 2.0

    # Ensure normal points upward (pushing up on the foot/midsole)
    if normal[2] < 0.0:
        normal = -normal

    # 2. Query nearest midsole thickness (L_0) in local spring grid plane
    nearest = find_nearest_spring_grid(
        centroid,
        grid_to_spring,
        grid_min_u,
        grid_min_v,
        grid_spacing,
        grid_num_u,
        grid_num_v,
        spring_xy,
    )
    local_thick = float(0.01)  # Default 10mm fallback
    if nearest != -1:
        local_thick = spring_slack[nearest]

    # Calculate vertical displacement relative to depths (negative in Warp)
    displacement = -2.0 * depths[tid]
    if displacement <= 0.0:
        return

    slack = wp.max(local_thick, 1.0e-6)
    stack_displacement = displacement
    stack_comp_vel = float(0.0)
    coupled_energy_scale = float(1.0)
    if nearest != -1:
        top_area = coupled_surface_state[nearest * 6 + SURFACE_TOP_AREA]
        ground_area = coupled_surface_state[nearest * 6 + SURFACE_GROUND_AREA]
        if top_area > 0.0 and ground_area > 0.0:
            top_displacement = coupled_surface_state[nearest * 6 + SURFACE_TOP_DISP_AREA] / top_area
            ground_displacement = coupled_surface_state[nearest * 6 + SURFACE_GROUND_DISP_AREA] / ground_area
            top_comp_vel = coupled_surface_state[nearest * 6 + SURFACE_TOP_VEL_AREA] / top_area
            ground_comp_vel = coupled_surface_state[nearest * 6 + SURFACE_GROUND_VEL_AREA] / ground_area
            stack_displacement = wp.clamp(top_displacement + ground_displacement, 0.0, slack)
            stack_comp_vel = top_comp_vel + ground_comp_vel
            coupled_energy_scale = 0.5

    strain = wp.clamp(stack_displacement / slack, 0.0, 0.99)

    # 3. Calculate compressive velocity at contact point
    midsole_com = wp.transform_point(body_q[midsole_body_idx], body_com[midsole_body_idx])
    qd_midsole = body_qd[midsole_body_idx]
    w_midsole = wp.spatial_bottom(qd_midsole)
    v_midsole = wp.spatial_top(qd_midsole)
    v_midsole_c = v_midsole + wp.cross(w_midsole, centroid - midsole_com)
    comp_vel = -wp.dot(v_midsole_c, normal)

    # 4. Evaluate contact stress and force
    if stack_comp_vel != 0.0:
        comp_vel = stack_comp_vel

    stress = evaluate_contact_stress(strain, comp_vel, params)
    force_magnitude = stress * area * wp.max(params[PARAM_HYDRO_FORCE_SCALE], 0.0)
    force_vec = normal * force_magnitude

    # 5. Apply bottom contact force: midsole-ground contact pushes the midsole up.
    r_midsole = centroid - midsole_com
    wp.atomic_add(body_f, midsole_body_idx, wp.spatial_vector(force_vec, wp.cross(r_midsole, force_vec)))

    # Accumulate to wrench_out for stats (midsole-ground contact)
    wp.atomic_add(wrench_out, 0, force_vec[0])
    wp.atomic_add(wrench_out, 1, force_vec[1])
    wp.atomic_add(wrench_out, 2, force_vec[2])

    torque = wp.cross(r_midsole, force_vec)
    wp.atomic_add(wrench_out, 3, torque[0])
    wp.atomic_add(wrench_out, 4, torque[1])
    wp.atomic_add(wrench_out, 5, torque[2])

    dissipated_energy = evaluate_dissipated_energy_rate(strain, comp_vel, area, params) * sim_dt * coupled_energy_scale
    wp.atomic_add(
        energy_out, 1, evaluate_elastic_energy(stack_displacement, slack, area, params) * coupled_energy_scale
    )
    wp.atomic_add(energy_out, 3, dissipated_energy)
    wp.atomic_add(dissipated_energy_total_out, 0, dissipated_energy)

    if nearest != -1:
        wp.atomic_max(ground_displacement_out, nearest, displacement)
        wp.atomic_max(ground_pressure_kpa_out, nearest, stress * 0.001)
        wp.atomic_max(stack_displacement_out, nearest, stack_displacement)
        wp.atomic_max(stack_pressure_kpa_out, nearest, stress * 0.001)


@wp.kernel
def apply_bonded_top_forces_kernel(
    grid_xy: wp.array[wp.vec2],
    top_m: wp.array[float],
    slack_length_m: wp.array[float],
    foot_contact_valid: wp.array[wp.int32],
    spring_count: int,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    foot_body_idx: int,
    midsole_body_idx: int,
    coupled_surface_state: wp.array[float],
    params: wp.array[float],
    body_f: wp.array[wp.spatial_vector],
    energy_out: wp.array[float],
    dissipated_energy_total_out: wp.array[float],
    sim_dt: float,
    foot_pressure_kpa_out: wp.array[float],
    stack_displacement_out: wp.array[float],
    stack_pressure_kpa_out: wp.array[float],
):
    spring = wp.tid()
    if spring >= spring_count:
        return
    if foot_contact_valid[spring] == 0:
        return

    top_area = coupled_surface_state[spring * 6 + SURFACE_TOP_AREA]
    if top_area <= 0.0:
        return

    top_displacement = coupled_surface_state[spring * 6 + SURFACE_TOP_DISP_AREA] / top_area
    top_comp_vel = coupled_surface_state[spring * 6 + SURFACE_TOP_VEL_AREA] / top_area
    stack_displacement = top_displacement
    stack_comp_vel = top_comp_vel
    coupled_energy_scale = float(1.0)

    ground_area = coupled_surface_state[spring * 6 + SURFACE_GROUND_AREA]
    if ground_area > 0.0:
        ground_displacement = coupled_surface_state[spring * 6 + SURFACE_GROUND_DISP_AREA] / ground_area
        ground_comp_vel = coupled_surface_state[spring * 6 + SURFACE_GROUND_VEL_AREA] / ground_area
        stack_displacement = top_displacement + ground_displacement
        stack_comp_vel = top_comp_vel + ground_comp_vel
        coupled_energy_scale = 0.5

    slack = wp.max(slack_length_m[spring], 1.0e-6)
    stack_displacement = wp.clamp(stack_displacement, 0.0, slack)
    if stack_displacement <= 0.0:
        return

    midsole_z = body_q[midsole_body_idx].p[2]

    strain = wp.clamp(stack_displacement / slack, 0.0, 0.99)
    stress = evaluate_contact_stress(strain, stack_comp_vel, params)
    force_magnitude = stress * top_area * wp.max(params[PARAM_HYDRO_FORCE_SCALE], 0.0)
    force_vec = wp.vec3(0.0, 0.0, force_magnitude)

    xy = grid_xy[spring]
    point = wp.vec3(xy[0], xy[1], midsole_z + top_m[spring])
    foot_com = wp.transform_point(body_q[foot_body_idx], body_com[foot_body_idx])
    midsole_com = wp.transform_point(body_q[midsole_body_idx], body_com[midsole_body_idx])
    r_foot = point - foot_com
    r_midsole = point - midsole_com

    wp.atomic_add(body_f, foot_body_idx, wp.spatial_vector(force_vec, wp.cross(r_foot, force_vec)))
    wp.atomic_sub(body_f, midsole_body_idx, wp.spatial_vector(force_vec, wp.cross(r_midsole, force_vec)))

    dissipated_energy = evaluate_dissipated_energy_rate(strain, stack_comp_vel, top_area, params) * sim_dt
    dissipated_energy *= coupled_energy_scale
    wp.atomic_add(
        energy_out, 0, evaluate_elastic_energy(stack_displacement, slack, top_area, params) * coupled_energy_scale
    )
    wp.atomic_add(energy_out, 2, dissipated_energy)
    wp.atomic_add(dissipated_energy_total_out, 0, dissipated_energy)
    foot_pressure_kpa_out[spring] = stress * 0.001
    stack_displacement_out[spring] = stack_displacement
    stack_pressure_kpa_out[spring] = stress * 0.001


@wp.kernel
def update_peak_maps_kernel(
    displacement: wp.array[float],
    pressure: wp.array[float],
    peak_displacement: wp.array[float],
    peak_pressure: wp.array[float],
    foot_displacement: wp.array[float],
    foot_pressure: wp.array[float],
    peak_foot_displacement: wp.array[float],
    peak_foot_pressure: wp.array[float],
    stack_displacement: wp.array[float],
    stack_pressure: wp.array[float],
    peak_stack_displacement: wp.array[float],
    peak_stack_pressure: wp.array[float],
    size: int,
):
    tid = wp.tid()
    if tid >= size:
        return
    peak_displacement[tid] = wp.max(peak_displacement[tid], displacement[tid])
    peak_pressure[tid] = wp.max(peak_pressure[tid], pressure[tid])
    peak_foot_displacement[tid] = wp.max(peak_foot_displacement[tid], foot_displacement[tid])
    peak_foot_pressure[tid] = wp.max(peak_foot_pressure[tid], foot_pressure[tid])
    peak_stack_displacement[tid] = wp.max(peak_stack_displacement[tid], stack_displacement[tid])
    peak_stack_pressure[tid] = wp.max(peak_stack_pressure[tid], stack_pressure[tid])


@wp.kernel
def set_kinematic_foot_state_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    body_index: int,
    joint_q_start: int,
    joint_qd_start: int,
    z_m: float,
    vertical_velocity_mps: float,
):
    body_q[body_index] = wp.transform(wp.vec3(0.0, 0.0, z_m), wp.quat_identity())
    body_qd[body_index] = wp.spatial_vector(
        wp.vec3(0.0, 0.0, vertical_velocity_mps),
        wp.vec3(0.0, 0.0, 0.0),
    )
    joint_q[joint_q_start + 0] = 0.0
    joint_q[joint_q_start + 1] = 0.0
    joint_q[joint_q_start + 2] = z_m
    joint_q[joint_q_start + 3] = 0.0
    joint_q[joint_q_start + 4] = 0.0
    joint_q[joint_q_start + 5] = 0.0
    joint_q[joint_q_start + 6] = 1.0
    joint_qd[joint_qd_start + 0] = 0.0
    joint_qd[joint_qd_start + 1] = 0.0
    joint_qd[joint_qd_start + 2] = vertical_velocity_mps
    joint_qd[joint_qd_start + 3] = 0.0
    joint_qd[joint_qd_start + 4] = 0.0
    joint_qd[joint_qd_start + 5] = 0.0


@wp.kernel
def set_prismatic_joint_state_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_q_start: int,
    joint_qd_start: int,
    q_m: float,
    qd_mps: float,
):
    joint_q[joint_q_start] = q_m
    joint_qd[joint_qd_start] = qd_mps


@wp.kernel
def update_shoe_stats_kernel(
    wrench: wp.array[float],
    elastic_energy: wp.array[float],
    dissipated_energy_total: wp.array[float],
    stats: wp.array[float],
):
    last_force = wrench[2]
    last_elastic_energy = elastic_energy[0] + elastic_energy[1]
    stats[STATS_LAST_FORCE_N] = last_force
    stats[STATS_PEAK_FORCE_N] = wp.max(stats[STATS_PEAK_FORCE_N], last_force)
    stats[STATS_LAST_ELASTIC_ENERGY_J] = last_elastic_energy
    stats[STATS_PEAK_ELASTIC_ENERGY_J] = wp.max(stats[STATS_PEAK_ELASTIC_ENERGY_J], last_elastic_energy)
    stats[STATS_LAST_DISSIPATED_ENERGY_J] = dissipated_energy_total[0]


def _colors_from_compression(compression: np.ndarray, max_compression: float) -> np.ndarray:
    normalized = np.clip(compression / max(max_compression, 1.0e-9), 0.0, 1.0)
    colors = np.empty((len(compression), 3), dtype=np.float32)
    colors[:, 0] = 0.12 + 0.88 * normalized
    colors[:, 1] = 0.55 * (1.0 - normalized) + 0.12
    colors[:, 2] = 0.95 * (1.0 - normalized) + 0.10
    return colors


def _colors_from_pressure(pressure_kpa: np.ndarray, max_pressure_kpa: float) -> np.ndarray:
    normalized = np.clip(pressure_kpa / max(max_pressure_kpa, 1.0e-9), 0.0, 1.0)
    colors = np.empty((len(pressure_kpa), 3), dtype=np.float32)

    # Classic Jet colormap approximation:
    r = np.clip(np.minimum(4.0 * normalized - 1.5, -4.0 * normalized + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(4.0 * normalized - 0.5, -4.0 * normalized + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(4.0 * normalized + 0.5, -4.0 * normalized + 2.5), 0.0, 1.0)

    colors[:, 0] = r
    colors[:, 1] = g
    colors[:, 2] = b
    return colors


def _compute_pressures(
    current_lengths: np.ndarray,
    slack_lengths: np.ndarray,
    velocities: np.ndarray,
    material: FoundationMaterial,
    spacing_m: float,
    neighbors: np.ndarray | None = None,
) -> np.ndarray:
    slack = np.maximum(slack_lengths, 1.0e-6)
    comp = np.maximum(slack - current_lengths, 0.0)
    strain = comp / slack

    lock = max(material.lock_strain, 1.0e-4)
    normalized = np.minimum(strain / lock, 0.999)
    alpha = max(material.ogden_alpha, 1.0e-4)
    ogden_stress = material.stiffness_pa * (np.power(1.0 - normalized, -alpha) - 1.0) / alpha

    h2 = spacing_m * spacing_m
    laplacian = np.zeros_like(comp)
    if h2 > 1.0e-12 and neighbors is not None:
        n_indices = neighbors.copy()
        self_indices = np.arange(len(comp))
        for col in range(4):
            mask = n_indices[:, col] == -1
            n_indices[mask, col] = self_indices[mask]

        val_left = comp[n_indices[:, 0]]
        val_right = comp[n_indices[:, 1]]
        val_bottom = comp[n_indices[:, 2]]
        val_top = comp[n_indices[:, 3]]

        laplacian = (val_left + val_right + val_bottom + val_top - 4.0 * comp) / h2

    elastic_stress = ogden_stress - material.pasternak_stiffness_n_per_m * laplacian

    damping_strain = np.maximum(strain, 1.0e-8)
    damping_weight = np.power(damping_strain, max(material.damping_power, 0.0))
    compression_velocity = -velocities
    viscous_stress = material.damping_pa_s * damping_weight * compression_velocity

    pressures_pa = np.maximum(elastic_stress + viscous_stress, 0.0)
    return pressures_pa


def _compute_kim_pressures(
    displacement_m: np.ndarray,
    slack_lengths: np.ndarray,
    velocities: np.ndarray,
    material: KimHyperfoamMaterial,
    damping_pa_s: float,
    lower_material: KimHyperfoamMaterial | None = None,
    upper_fraction: float = 1.0,
) -> np.ndarray:
    slack = np.maximum(slack_lengths, 1.0e-6)
    strain = np.maximum(displacement_m, 0.0) / slack
    if lower_material is None:
        elastic_stress = _kim_pressure_pa(strain, material)
    else:
        elastic_stress = _kim_layered_pressure_pa(strain, material, lower_material, upper_fraction)
    damping_weight = np.power(np.maximum(strain, 1.0e-8), 1.0)
    viscous_stress = damping_pa_s * damping_weight * -velocities
    return np.maximum(elastic_stress + viscous_stress, 0.0)


class Example:
    """Foot-Shoe interactive ground contact simulation using a custom 3D Hydroelastic contact model."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.test_mode = args.test
        self.kinematic = args.kinematic
        self.debug = bool(args.debug)
        self.contact_law = args.contact_law
        self.kim_material = KIM_HYPERFOAM_MATERIALS[args.foam_material]
        self.kim_upper_material = KIM_HYPERFOAM_MATERIALS[args.upper_foam_material]
        self.kim_lower_material = KIM_HYPERFOAM_MATERIALS[args.lower_foam_material]
        self.upper_layer_fraction = float(np.clip(args.upper_layer_fraction, 0.0, 1.0))
        self.kim_damping_pa_s = args.kim_damping_pa_s
        if args.min_peak_force_n is None:
            self.min_peak_force_n = 0.5 if self.contact_law in ("kim-hyperfoam", "kim-layered") else 500.0
        else:
            self.min_peak_force_n = args.min_peak_force_n
        self.hydro_force_scale = float(args.hydro_force_scale)
        self.shoe_attach_mode = args.shoe_attach_mode
        self.shoe_compression_limit_m = args.shoe_compression_limit_mm * 0.001
        self.shoe_lift_limit_m = args.shoe_lift_limit_mm * 0.001
        self.shoe_joint_limit_ke = args.shoe_joint_limit_ke
        self.shoe_joint_limit_kd = args.shoe_joint_limit_kd
        self.shoe_joint_friction = args.shoe_joint_friction

        # 1. Load resources from manifest and calibrated v2 cache
        self.manifest = load_manifest(args.manifest)

        # Load default configuration for midsole if it exists
        self.config_path = self.manifest.midsole_mesh.parent / f"{self.manifest.midsole_mesh.stem}_foot_config.json"
        loaded_config = self.load_foot_config()

        # Override default args if not explicitly provided
        for field in [
            "foot_roll_deg",
            "foot_pitch_deg",
            "foot_yaw_deg",
            "foot_offset_x_mm",
            "foot_offset_y_mm",
            "foot_offset_z_mm",
        ]:
            if getattr(args, field, 0.0) == 0.0 and field in loaded_config:
                setattr(args, field, loaded_config[field])

        self.args = args
        self.init_roll = args.foot_roll_deg
        self.init_pitch = args.foot_pitch_deg
        self.init_yaw = args.foot_yaw_deg
        self.init_ox = args.foot_offset_x_mm
        self.init_oy = args.foot_offset_y_mm
        self.init_oz = args.foot_offset_z_mm

        # GUI state variables
        self.gui_simulate_enabled = True
        self.foot_roll_deg = args.foot_roll_deg
        self.foot_pitch_deg = args.foot_pitch_deg
        self.foot_yaw_deg = args.foot_yaw_deg
        self.foot_offset_x_mm = args.foot_offset_x_mm
        self.foot_offset_y_mm = args.foot_offset_y_mm
        self.foot_offset_z_mm = args.foot_offset_z_mm

        self.output_dir = Path(args.output_dir) if args.output_dir else self.manifest.cache_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.spring_grid, self.midsole_vertices = _load_spring_grid(self.manifest, self.output_dir)

        with open(args.material) as f:
            mat_data = json.load(f)["material"]
        self.calibrated_material = FoundationMaterial(**mat_data)
        if self.contact_law == "kim-hyperfoam":
            self.material = _kim_material_to_foundation(self.kim_material, self.kim_damping_pa_s)
        elif self.contact_law == "kim-layered":
            self.material = _kim_material_to_foundation(self.kim_upper_material, self.kim_damping_pa_s)
        else:
            self.material = self.calibrated_material

        self.neighbors = compute_grid_neighbors(self.spring_grid.grid_uv_m, self.spring_grid.spacing_m)
        self.max_display_pressure_kpa = 800.0
        self.min_bottom_m = np.min(self.spring_grid.bottom_m)
        self.start_z = -self.min_bottom_m + 0.005
        self.midsole_ground_stop_z = -self.min_bottom_m
        if self.contact_law == "kim-hyperfoam":
            print(
                "[material] contact_law="
                f"{self.contact_law} foam={self.kim_material.name} "
                f"mu1={self.kim_material.mu1_pa:.3g}Pa alpha1={self.kim_material.alpha1:.3g} "
                f"nu={self.kim_material.poisson_ratio:.2f} bulk={self.kim_material.bulk_modulus_pa:.3g}Pa"
            )
        elif self.contact_law == "kim-layered":
            print(
                "[material] contact_law=kim-layered "
                f"upper={self.kim_upper_material.name} lower={self.kim_lower_material.name} "
                f"upper_fraction={self.upper_layer_fraction:.3g}"
            )
        else:
            print(
                "[material] contact_law=calibrated-ogden "
                f"stiffness={self.material.stiffness_pa:.3g}Pa alpha={self.material.ogden_alpha:.3g} "
                f"lock_strain={self.material.lock_strain:.3g}"
            )

        # 2. Load and mirror/align the foot model
        foot_v, foot_f = _load_obj_mesh(Path(args.foot_mesh))

        sign = 1.0 if args.mirror_foot else -1.0

        foot_v_transformed = np.zeros_like(foot_v)
        foot_v_transformed[:, 0] = sign * foot_v[:, 2]  # X_midsole = width (from Z_obj)
        foot_v_transformed[:, 1] = foot_v[:, 0]  # Y_midsole = length (from X_obj)
        foot_v_transformed[:, 2] = foot_v[:, 1]  # Z_midsole = height (from Y_obj)

        foot_f_transformed = foot_f.copy()
        if args.mirror_foot:
            foot_f_transformed[:, [1, 2]] = foot_f_transformed[:, [2, 1]]

        # Center horizontally on the spring grid
        foot_center = 0.5 * (np.min(foot_v_transformed, axis=0) + np.max(foot_v_transformed, axis=0))
        midsole_center = 0.5 * (np.min(self.spring_grid.grid_uv_m, axis=0) + np.max(self.spring_grid.grid_uv_m, axis=0))
        foot_v_transformed[:, 0] += midsole_center[0] - foot_center[0]
        foot_v_transformed[:, 1] += midsole_center[1] - foot_center[1]

        # Align foot yaw, pitch, roll
        foot_center = 0.5 * (np.min(foot_v_transformed, axis=0) + np.max(foot_v_transformed, axis=0))
        foot_v_rel = foot_v_transformed - foot_center

        R = _rotation_matrix_xyz_deg([args.foot_roll_deg, args.foot_pitch_deg, args.foot_yaw_deg])
        foot_v_transformed = foot_v_rel @ R.T + foot_center

        # Apply translation offsets in X, Y
        foot_v_transformed[:, 0] += args.foot_offset_x_mm * 0.001
        foot_v_transformed[:, 1] += args.foot_offset_y_mm * 0.001

        # Align vertically (Z-axis) so foot sole just touches the top surface of midsole
        spacing = self.spring_grid.spacing_m
        z_foot_sole = np.full(len(self.spring_grid.grid_uv_m), np.nan)
        for i, (x_g, y_g) in enumerate(self.spring_grid.grid_uv_m):
            in_cell = (np.abs(foot_v_transformed[:, 0] - x_g) <= spacing * 0.5) & (
                np.abs(foot_v_transformed[:, 1] - y_g) <= spacing * 0.5
            )
            if np.any(in_cell):
                z_foot_sole[i] = np.min(foot_v_transformed[in_cell, 2])

        valid = np.isfinite(z_foot_sole)
        if np.any(valid):
            z_offsets = self.spring_grid.top_m[valid] - z_foot_sole[valid]
            Z_offset = np.max(z_offsets)
            # Apply Z alignment and manual Z offset
            foot_v_transformed[:, 2] += Z_offset + args.foot_offset_z_mm * 0.001
            z_foot_sole[valid] += Z_offset + args.foot_offset_z_mm * 0.001

            missing = ~valid
            if np.any(missing):
                grid_xy = self.spring_grid.grid_uv_m
                valid_xy = grid_xy[valid]
                missing_xy = grid_xy[missing]
                nearest = np.argmin(np.sum((missing_xy[:, None, :] - valid_xy[None, :, :]) ** 2, axis=2), axis=1)
                nearest_dist = np.linalg.norm(missing_xy - valid_xy[nearest], axis=1)
                fill_mask = nearest_dist <= max(2.5 * spacing, 0.012)
                missing_indices = np.flatnonzero(missing)
                z_foot_sole[missing_indices[fill_mask]] = z_foot_sole[np.flatnonzero(valid)[nearest[fill_mask]]]

        self.foot_sole_z_m = z_foot_sole
        self.foot_contact_valid = np.isfinite(self.foot_sole_z_m)
        if self.debug:
            print(
                f"[DEBUG] Foot mesh bounds: min={np.min(foot_v_transformed, axis=0)}, "
                f"max={np.max(foot_v_transformed, axis=0)}"
            )
            print(f"[DEBUG] start_z={self.start_z}")

        # Physical parameters
        self.mass = 80.0  # kg
        self.gravity = -9.81  # m/s^2

        # 3. Create Model with Hydroelastic Mesh collision
        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)

        narrow_band = 0.080
        contact_gap = 0.005

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            gap=contact_gap,
            kh=1.0e7,
            sdf_max_resolution=128,
        )
        midsole_center = 0.5 * (np.min(self.spring_grid.grid_uv_m, axis=0) + np.max(self.spring_grid.grid_uv_m, axis=0))
        self.ground_shape_id = builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(midsole_center[0], midsole_center[1], -0.05), q=wp.quat_identity()),
            hx=0.25,
            hy=0.25,
            hz=0.05,
            cfg=ground_cfg,
            color=wp.vec3(0.125, 0.125, 0.15),
            label="ground_box",
        )

        # Add foot body as link
        self.foot_body_id = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.start_z), q=wp.quat_identity()),
            mass=self.mass,
            inertia=wp.mat33(10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0),
            lock_inertia=True,
            is_kinematic=self.kinematic,
            label="foot_body",
        )
        self.foot_joint_id = builder.add_joint_free(
            child=self.foot_body_id,
            label="foot_free_joint",
        )

        foot_mesh = newton.Mesh(foot_v_transformed, foot_f_transformed.reshape(-1))

        # Build SDF for foot mesh
        foot_mesh.build_sdf(
            max_resolution=128,
            narrow_band_range=(-narrow_band, narrow_band),
            margin=contact_gap,
            device=viewer.device,
        )

        foot_cfg = newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            gap=contact_gap,
            kh=1.0e14,
        )
        self.foot_shape_id = builder.add_shape_mesh(
            self.foot_body_id,
            mesh=foot_mesh,
            cfg=foot_cfg,
            color=wp.vec3(0.65, 0.72, 0.88),
            label="foot_mesh",
        )

        # Load midsole mesh directly from the watertight repaired mesh
        report = condition_midsole_mesh(
            self.manifest.midsole_mesh,
            self.output_dir,
            source_units=str(self.manifest.qc.get("mesh_source_units", "mm")),
            min_thickness_m=float(self.manifest.qc.get("min_midsole_thickness_m", 0.005)),
            max_thickness_m=float(self.manifest.qc.get("max_midsole_thickness_m", 0.08)),
        )
        midsole_v, midsole_f = _load_obj_mesh(Path(str(report["repaired_mesh"])))

        # Transform midsole mesh to Newton coordinates (X=width, Y=length, Z=height)
        # In the source midsole mesh: X=width, Y=height, Z=length
        midsole_v_transformed = np.zeros_like(midsole_v)
        midsole_v_transformed[:, 0] = midsole_v[:, 0]  # X_newton = X_midsole (width)
        midsole_v_transformed[:, 1] = midsole_v[:, 2]  # Y_newton = Z_midsole (length)
        midsole_v_transformed[:, 2] = midsole_v[:, 1]  # Z_newton = Y_midsole (height)

        midsole_f_transformed = midsole_f.copy()
        # Since we swapped Y and Z (odd permutation), swap face indices 1 and 2 to preserve winding
        midsole_f_transformed[:, [1, 2]] = midsole_f_transformed[:, [2, 1]]
        midsole_tri = midsole_v_transformed[midsole_f_transformed]
        midsole_face_centroids = np.mean(midsole_tri, axis=1)
        midsole_face_normals = np.cross(
            midsole_tri[:, 1] - midsole_tri[:, 0],
            midsole_tri[:, 2] - midsole_tri[:, 0],
        )
        midsole_face_normal_norm = np.linalg.norm(midsole_face_normals, axis=1)
        midsole_face_nz = np.divide(
            midsole_face_normals[:, 2],
            midsole_face_normal_norm,
            out=np.zeros_like(midsole_face_normal_norm),
            where=midsole_face_normal_norm > 0.0,
        )
        self.bonded_top_surface_xy = midsole_face_centroids[
            (midsole_face_nz > 0.35)
            & (midsole_face_centroids[:, 2] > np.percentile(midsole_face_centroids[:, 2], 45.0)),
            :2,
        ].astype(np.float32)
        if len(self.bonded_top_surface_xy) > 0:
            self.bonded_top_surface_nearest = np.argmin(
                np.sum(
                    (self.bonded_top_surface_xy[:, None, :] - self.spring_grid.grid_uv_m[None, :, :]) ** 2,
                    axis=2,
                ),
                axis=1,
            )
        else:
            self.bonded_top_surface_nearest = np.zeros(0, dtype=np.int64)

        midsole_mesh = newton.Mesh(midsole_v_transformed, midsole_f_transformed.reshape(-1))
        if self.debug:
            print(
                f"[DEBUG] Midsole mesh bounds: min={np.min(midsole_v_transformed, axis=0)}, "
                f"max={np.max(midsole_v_transformed, axis=0)}"
            )

        # Build SDF for midsole mesh
        midsole_mesh.build_sdf(
            max_resolution=128,
            narrow_band_range=(-narrow_band, narrow_band),
            margin=contact_gap,
            device=viewer.device,
        )

        midsole_cfg = newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            gap=contact_gap,
            kh=1.0e7,
        )

        # Add midsole body as link
        self.midsole_body_id = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, self.start_z), q=wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            lock_inertia=True,
            is_kinematic=False,
            label="midsole_body",
        )
        if self.shoe_attach_mode == "foot-prismatic":
            self.midsole_joint_id = builder.add_joint_prismatic(
                parent=self.foot_body_id,
                child=self.midsole_body_id,
                parent_xform=wp.transform_identity(),
                child_xform=wp.transform_identity(),
                axis=newton.Axis.Z,
                limit_lower=-self.shoe_lift_limit_m,
                limit_upper=self.shoe_compression_limit_m,
                limit_ke=self.shoe_joint_limit_ke,
                limit_kd=self.shoe_joint_limit_kd,
                friction=self.shoe_joint_friction,
                label="foot_to_midsole_vertical_slide",
                collision_filter_parent=False,
            )

            builder.add_articulation(
                [self.foot_joint_id, self.midsole_joint_id],
                label="foot_shoe_articulation",
            )

        else:
            self.midsole_joint_id = builder.add_joint_prismatic(
                parent=-1,
                child=self.midsole_body_id,
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, self.start_z), q=wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=newton.Axis.Z,
                limit_lower=-0.05,
                limit_upper=0.02,
                limit_ke=self.shoe_joint_limit_ke,
                limit_kd=self.shoe_joint_limit_kd,
                friction=self.shoe_joint_friction,
                label="midsole_vertical_slide",
            )

            builder.add_articulation([self.foot_joint_id], label="foot_articulation")
            builder.add_articulation([self.midsole_joint_id], label="midsole_articulation")

        # Add midsole as a shape on the midsole body
        self.midsole_shape_id = builder.add_shape_mesh(
            body=self.midsole_body_id,
            mesh=midsole_mesh,
            cfg=midsole_cfg,
            color=wp.vec3(0.4, 0.4, 0.4),
            label="midsole_mesh",
        )

        # Finalize model with Hydroelastic config
        self.sdf_config = HydroelasticSDF.Config(
            reduce_contacts=False,
            pre_prune_contacts=False,
            output_contact_surface=True,
            anchor_contact=True,
            buffer_fraction=1.0,
        )
        self.model = builder.finalize(device=viewer.device)
        if hasattr(self.viewer, "set_model"):
            self.viewer.set_model(self.model)
        if self.debug:
            print(
                f"[DEBUG] foot shape_id={self.foot_shape_id} flag={self.model.shape_flags.numpy()[self.foot_shape_id]}"
            )
            print(
                f"[DEBUG] midsole shape_id={self.midsole_shape_id} flag={self.model.shape_flags.numpy()[self.midsole_shape_id]}"
            )
            print(f"[DEBUG] ShapeFlags.HYDROELASTIC value={int(newton.ShapeFlags.HYDROELASTIC)}")
            print(
                f"[DEBUG] shape_world: {self.model.shape_world.numpy() if hasattr(self.model, 'shape_world') else None}"
            )
            print(
                "[DEBUG] shape_collision_group: "
                f"{self.model.shape_collision_group.numpy() if hasattr(self.model, 'shape_collision_group') else None}"
            )
            print(
                "[DEBUG] shape_contact_pairs: "
                f"{self.model.shape_contact_pairs.numpy() if hasattr(self.model, 'shape_contact_pairs') else None}"
            )

        self.state_0 = self.model.state()
        if self.debug:
            print(f"[DEBUG] foot_mesh.sdf is None? {foot_mesh.sdf is None}")
            if foot_mesh.sdf is not None:
                print(
                    "[DEBUG] foot_mesh.sdf bounds: "
                    f"{foot_mesh.sdf.texture_data.sdf_box_lower} to {foot_mesh.sdf.texture_data.sdf_box_upper}"
                )
            print(f"[DEBUG] midsole_mesh.sdf is None? {midsole_mesh.sdf is None}")
            if midsole_mesh.sdf is not None:
                print(
                    "[DEBUG] midsole_mesh.sdf bounds: "
                    f"{midsole_mesh.sdf.texture_data.sdf_box_lower} to {midsole_mesh.sdf.texture_data.sdf_box_upper}"
                )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        joint_child = self.model.joint_child.numpy()
        foot_joint_ids = np.flatnonzero(joint_child == self.foot_body_id)
        if len(foot_joint_ids) != 1:
            raise RuntimeError(f"Expected one free joint for foot body, found {len(foot_joint_ids)}")
        self.foot_joint_id = int(foot_joint_ids[0])
        self.foot_joint_q_start = int(self.model.joint_q_start.numpy()[self.foot_joint_id])
        self.foot_joint_qd_start = int(self.model.joint_qd_start.numpy()[self.foot_joint_id])
        self.midsole_joint_q_start = int(self.model.joint_q_start.numpy()[self.midsole_joint_id])
        self.midsole_joint_qd_start = int(self.model.joint_qd_start.numpy()[self.midsole_joint_id])

        # Instantiate the collision pipeline
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=20000,
            sdf_hydroelastic_config=self.sdf_config,
        )
        self.contacts = self.collision_pipeline.contacts()

        # Use SolverMuJoCo as requested
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=False,
            disable_contacts=True,  # Disable MuJoCo's built-in contacts
        )

        self.device = self.model.device

        # Warp arrays for search lookups inside the Ogden kernel
        self.wp_spring_xy = wp.array(self.spring_grid.grid_uv_m, dtype=wp.vec2, device=self.device)
        self.wp_spring_slack = wp.array(self.spring_grid.slack_length_m, dtype=float, device=self.device)
        self.wp_spring_top = wp.array(self.spring_grid.top_m, dtype=float, device=self.device)
        self.wp_spring_bottom = wp.array(self.spring_grid.bottom_m, dtype=float, device=self.device)
        self.wp_foot_sole_z = wp.array(np.nan_to_num(self.foot_sole_z_m, nan=0.0), dtype=float, device=self.device)
        self.wp_foot_contact_valid = wp.array(
            self.foot_contact_valid.astype(np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        self.num_springs = len(self.spring_grid.slack_length_m)

        # Determine regular grid bounds from spring grid
        grid_xy = self.spring_grid.grid_uv_m
        min_u = float(np.min(grid_xy[:, 0]))
        max_u = float(np.max(grid_xy[:, 0]))
        min_v = float(np.min(grid_xy[:, 1]))
        max_v = float(np.max(grid_xy[:, 1]))
        spacing = float(self.spring_grid.spacing_m)

        # Grid dimensions (number of cells along u and v)
        num_u = int(np.round((max_u - min_u) / spacing)) + 1
        num_v = int(np.round((max_v - min_v) / spacing)) + 1

        # Create 2D lookup table: grid index (i_u, i_v) -> spring index
        grid_to_spring = np.full((num_u, num_v), -1, dtype=np.int32)
        for k, (u, v) in enumerate(grid_xy):
            i_u = int(np.round((u - min_u) / spacing))
            i_v = int(np.round((v - min_v) / spacing))
            if 0 <= i_u < num_u and 0 <= i_v < num_v:
                grid_to_spring[i_u, i_v] = k

        # Upload to Warp
        self.wp_grid_to_spring = wp.array2d(grid_to_spring, dtype=wp.int32, device=self.device)
        self.grid_min_u = min_u
        self.grid_min_v = min_v
        self.grid_spacing = spacing
        self.grid_num_u = num_u
        self.grid_num_v = num_v

        law_mode = CONTACT_LAW_CALIBRATED_OGDEN
        upper_bulk_or_stiffness = self.material.stiffness_pa
        upper_alpha = self.material.ogden_alpha
        lower_bulk = self.material.stiffness_pa
        lower_alpha = self.material.ogden_alpha
        upper_fraction = 1.0
        if self.contact_law == "kim-hyperfoam":
            law_mode = CONTACT_LAW_KIM_HYPERFOAM
            upper_bulk_or_stiffness = self.kim_material.bulk_modulus_pa
            upper_alpha = self.kim_material.alpha1
        elif self.contact_law == "kim-layered":
            law_mode = CONTACT_LAW_KIM_LAYERED
            upper_bulk_or_stiffness = self.kim_upper_material.bulk_modulus_pa
            upper_alpha = self.kim_upper_material.alpha1
            lower_bulk = self.kim_lower_material.bulk_modulus_pa
            lower_alpha = self.kim_lower_material.alpha1
            upper_fraction = self.upper_layer_fraction

        self.wp_params = wp.array(
            [
                upper_bulk_or_stiffness,
                upper_alpha,
                self.material.lock_strain,
                self.material.damping_pa_s,
                self.material.damping_power,
                float(law_mode),
                lower_bulk,
                lower_alpha,
                upper_fraction,
                self.hydro_force_scale,
            ],
            dtype=float,
            device=self.device,
        )
        self.current_z = self.start_z
        self.current_vz = 0.0
        self.current_midsole_z = self.start_z
        self.current_midsole_vz = 0.0
        self.previous_midsole_z = self.start_z
        self.save_plots = bool(args.save_plots)

        # Initialize body and free-joint positions.
        wp.launch(
            set_kinematic_foot_state_kernel,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.foot_body_id,
                self.foot_joint_q_start,
                self.foot_joint_qd_start,
                float(self.start_z),
                0.0,
            ],
            device=self.device,
        )

        wp.launch(
            set_prismatic_joint_state_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.midsole_joint_q_start,
                self.midsole_joint_qd_start,
                0.0,
                0.0,
            ],
            device=self.device,
        )

        wp.launch(
            set_prismatic_joint_state_kernel,
            dim=1,
            inputs=[
                self.state_1.joint_q,
                self.state_1.joint_qd,
                self.midsole_joint_q_start,
                self.midsole_joint_qd_start,
                0.0,
                0.0,
            ],
            device=self.device,
        )

        # Logging / stats
        self.peak_force_n = 0.0
        self.history_z = []
        self.history_force = []
        self.last_force_n = 0.0
        self.last_elastic_energy_j = 0.0
        self.peak_elastic_energy_j = 0.0
        self.dissipated_energy_j = 0.0
        self.peak_compression_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_foot_top_displacement_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_foot_top_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_ground_bottom_displacement_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_ground_bottom_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_stack_displacement_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_stack_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)
        self.wp_foot_top_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_foot_top_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_ground_bottom_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_ground_bottom_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_stack_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_stack_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_coupled_surface_state = wp.zeros(self.num_springs * 6, dtype=float, device=self.device)
        self.wp_peak_foot_top_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_peak_foot_top_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_peak_ground_bottom_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_peak_ground_bottom_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_peak_stack_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_peak_stack_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_contact_wrench = wp.zeros(6, dtype=float, device=self.device)
        self.wp_contact_energy = wp.zeros(4, dtype=float, device=self.device)
        self.wp_dissipated_energy_total = wp.zeros(1, dtype=float, device=self.device)
        self.wp_step_stats = wp.zeros(5, dtype=float, device=self.device)
        self.foot_body_com_local = (
            self.model.body_com.numpy()[self.foot_body_id].copy()
            if hasattr(self.model, "body_com")
            else np.zeros(3, dtype=float)
        )
        self.midsole_body_com_local = (
            self.model.body_com.numpy()[self.midsole_body_id].copy()
            if hasattr(self.model, "body_com")
            else np.zeros(3, dtype=float)
        )
        self.history_foot_z = []
        self.history_midsole_z = []
        self.history_foot_disp = []
        self.history_foot_pres = []
        self.history_ground_disp = []
        self.history_ground_pres = []
        self.history_stack_disp = []
        self.history_stack_pres = []
        self.history_top_surface_samples = []
        self.history_ground_surface_samples = []
        self.history_elastic_energy = []
        self.history_dissipated_energy = []

        self.point_radius = max(float(self.spring_grid.spacing_m) * 0.22, 0.0008)
        self.contact_color = wp.vec3(0.05, 0.95, 0.52)
        self.max_display_compression = 0.02
        self._contact_mesh_buckets: dict[str, tuple[str, int, tuple[float, float, float]]] = {}

        self.setup_viser_gui()

    def _pressure_maps_from_warp(
        self,
        foot_z_m: float,
        midsole_z_m: float,
        update_peaks: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if update_peaks:
            self._update_peak_maps()
        return (
            self.wp_foot_top_displacement.numpy(),
            self.wp_foot_top_pressure_kpa.numpy(),
            self.wp_ground_bottom_displacement.numpy(),
            self.wp_ground_bottom_pressure_kpa.numpy(),
        )

    def _update_peak_maps(self):
        wp.launch(
            update_peak_maps_kernel,
            dim=self.num_springs,
            inputs=[
                self.wp_ground_bottom_displacement,
                self.wp_ground_bottom_pressure_kpa,
                self.wp_peak_ground_bottom_displacement,
                self.wp_peak_ground_bottom_pressure_kpa,
                self.wp_foot_top_displacement,
                self.wp_foot_top_pressure_kpa,
                self.wp_peak_foot_top_displacement,
                self.wp_peak_foot_top_pressure_kpa,
                self.wp_stack_displacement,
                self.wp_stack_pressure_kpa,
                self.wp_peak_stack_displacement,
                self.wp_peak_stack_pressure_kpa,
                self.num_springs,
            ],
            device=self.device,
        )

    def _sync_peak_maps_from_warp(self):
        self.peak_foot_top_displacement_m = self.wp_peak_foot_top_displacement.numpy()
        self.peak_foot_top_pressure_kpa = self.wp_peak_foot_top_pressure_kpa.numpy()
        self.peak_ground_bottom_displacement_m = self.wp_peak_ground_bottom_displacement.numpy()
        self.peak_ground_bottom_pressure_kpa = self.wp_peak_ground_bottom_pressure_kpa.numpy()
        self.peak_stack_displacement_m = self.wp_peak_stack_displacement.numpy()
        self.peak_stack_pressure_kpa = self.wp_peak_stack_pressure_kpa.numpy()
        self.peak_compression_m = self.peak_stack_displacement_m
        self.peak_pressure_kpa = self.peak_stack_pressure_kpa

    def _sync_step_stats_from_warp(self):
        stats = self.wp_step_stats.numpy()
        self.last_force_n = float(stats[STATS_LAST_FORCE_N])
        self.peak_force_n = float(stats[STATS_PEAK_FORCE_N])
        self.last_elastic_energy_j = float(stats[STATS_LAST_ELASTIC_ENERGY_J])
        self.peak_elastic_energy_j = float(stats[STATS_PEAK_ELASTIC_ENERGY_J])
        self.dissipated_energy_j = float(stats[STATS_LAST_DISSIPATED_ENERGY_J])

    def _surface_samples_from_hydro(self, contact_surface, face_count: int) -> tuple[np.ndarray, np.ndarray]:
        if contact_surface is None or face_count <= 0:
            return self._bonded_top_surface_samples(), np.empty((0, 5), dtype=np.float32)

        points = contact_surface.contact_surface_point.numpy()[: 3 * face_count].reshape(face_count, 3, 3)
        pairs = contact_surface.contact_surface_shape_pair.numpy()[:face_count]
        depths = contact_surface.contact_surface_depth.numpy()[:face_count]
        centroids = np.mean(points, axis=1)
        face_displacement_m = np.maximum(-2.0 * depths, 0.0)

        is_ground = ((pairs[:, 0] == self.midsole_shape_id) & (pairs[:, 1] == self.ground_shape_id)) | (
            (pairs[:, 1] == self.midsole_shape_id) & (pairs[:, 0] == self.ground_shape_id)
        )

        stack_displacement_m = self.wp_stack_displacement.numpy()
        stack_pressure_kpa = self.wp_stack_pressure_kpa.numpy()
        grid_xy = self.spring_grid.grid_uv_m

        def build_samples(mask: np.ndarray) -> np.ndarray:
            if not np.any(mask):
                return np.empty((0, 5), dtype=np.float32)

            xy = centroids[mask, :2]
            nearest = np.argmin(np.sum((xy[:, None, :] - grid_xy[None, :, :]) ** 2, axis=2), axis=1)
            return np.column_stack(
                (
                    xy[:, 0],
                    xy[:, 1],
                    face_displacement_m[mask],
                    stack_displacement_m[nearest],
                    stack_pressure_kpa[nearest],
                )
            ).astype(np.float32)

        return self._bonded_top_surface_samples(), build_samples(is_ground)

    def _bonded_top_surface_samples(self) -> np.ndarray:
        if len(self.bonded_top_surface_xy) == 0:
            return np.empty((0, 5), dtype=np.float32)

        nearest = self.bonded_top_surface_nearest
        foot_displacement_m = self.wp_foot_top_displacement.numpy()
        stack_displacement_m = self.wp_stack_displacement.numpy()
        stack_pressure_kpa = self.wp_stack_pressure_kpa.numpy()
        return np.column_stack(
            (
                self.bonded_top_surface_xy[:, 0],
                self.bonded_top_surface_xy[:, 1],
                foot_displacement_m[nearest],
                stack_displacement_m[nearest],
                stack_pressure_kpa[nearest],
            )
        ).astype(np.float32)

    def _bonded_top_surface_peak_samples(self) -> np.ndarray:
        if len(self.bonded_top_surface_xy) == 0:
            return np.empty((0, 5), dtype=np.float32)

        nearest = self.bonded_top_surface_nearest
        return np.column_stack(
            (
                self.bonded_top_surface_xy[:, 0],
                self.bonded_top_surface_xy[:, 1],
                self.peak_foot_top_displacement_m[nearest],
                self.peak_stack_displacement_m[nearest],
                self.peak_stack_pressure_kpa[nearest],
            )
        ).astype(np.float32)

    def _body_com_world(self, body_q: np.ndarray) -> np.ndarray:
        origin = body_q[self.foot_body_id, :3]
        if not hasattr(self.model, "body_com"):
            return origin

        body_com = self.model.body_com.numpy()[self.foot_body_id]
        if np.max(np.abs(body_com)) <= 1.0e-12:
            return origin
        return origin + _rotate_vec_by_quat(body_com, body_q[self.foot_body_id, 3:7])

    def simulate(self):
        if not self.gui_simulate_enabled:
            self.update_foot_visual_pose()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            return

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Update kinematic trajectory of the foot or apply gravity in dynamic mode
            if self.kinematic:
                omega = 2.0 * np.pi * 1.0
                disp = 0.0125 * (1.0 - np.cos(omega * self.sim_time))
                self.current_z = self.start_z - disp
                self.current_vz = -0.0125 * omega * np.sin(omega * self.sim_time)

                wp.launch(
                    set_kinematic_foot_state_kernel,
                    dim=1,
                    inputs=[
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        self.state_0.joint_q,
                        self.state_0.joint_qd,
                        self.foot_body_id,
                        self.foot_joint_q_start,
                        self.foot_joint_qd_start,
                        float(self.current_z),
                        float(self.current_vz),
                    ],
                    device=self.device,
                )

            # Evaluate FK so shape world transforms are updated in state_0
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

            # Apply non-contact forces for the one-dimensional vertical test.
            ext_force = 0.0
            if not self.kinematic:
                omega = 2.0 * np.pi * 1.0
                ext_force = 1000.0 * max(np.sin(omega * self.sim_time * 0.5), 0.0)

            wp.launch(
                apply_non_contact_forces_kernel,
                dim=1,
                inputs=[
                    self.state_0.body_f,
                    self.foot_body_id,
                    self.midsole_body_id,
                    self.mass,
                    2.0,  # Midsole mass
                    self.gravity,
                    ext_force,
                    int(self.kinematic),
                ],
                device=self.device,
            )

            # Run GPU broadphase/narrowphase to extract the contact surface
            self.collision_pipeline.collide(self.state_0, self.contacts)
            hydro = self.collision_pipeline.narrow_phase.hydroelastic_sdf
            contact_surface = hydro.get_contact_surface()

            self.wp_contact_wrench.zero_()
            self.wp_contact_energy.zero_()
            self.wp_coupled_surface_state.zero_()
            self.wp_foot_top_displacement.zero_()
            self.wp_foot_top_pressure_kpa.zero_()
            self.wp_ground_bottom_displacement.zero_()
            self.wp_ground_bottom_pressure_kpa.zero_()
            self.wp_stack_displacement.zero_()
            self.wp_stack_pressure_kpa.zero_()

            wp.launch(
                accumulate_bonded_top_state_kernel,
                dim=self.num_springs,
                inputs=[
                    self.wp_spring_top,
                    self.wp_foot_sole_z,
                    self.wp_foot_contact_valid,
                    self.wp_spring_slack,
                    self.num_springs,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.foot_body_id,
                    self.midsole_body_id,
                    float(self.start_z),
                    float(self.spring_grid.spacing_m),
                    self.wp_coupled_surface_state,
                    self.wp_foot_top_displacement,
                ],
                device=self.device,
            )

            if contact_surface is not None:
                wp.launch(
                    accumulate_bottom_hydro_state_kernel,
                    dim=self.collision_pipeline.rigid_contact_max,
                    inputs=[
                        contact_surface.contact_surface_point,
                        contact_surface.contact_surface_depth,
                        contact_surface.contact_surface_shape_pair,
                        contact_surface.face_contact_count,
                        self.midsole_shape_id,
                        self.ground_shape_id,
                        self.midsole_body_id,
                        self.model.body_com,
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        self.wp_spring_xy,
                        self.num_springs,
                        self.wp_coupled_surface_state,
                        # Grid lookup parameters
                        self.wp_grid_to_spring,
                        self.grid_min_u,
                        self.grid_min_v,
                        self.grid_spacing,
                        self.grid_num_u,
                        self.grid_num_v,
                    ],
                    device=self.device,
                )
                wp.launch(
                    evaluate_bottom_hydroelastic_ogden_kernel,
                    dim=self.collision_pipeline.rigid_contact_max,
                    inputs=[
                        contact_surface.contact_surface_point,
                        contact_surface.contact_surface_depth,
                        contact_surface.contact_surface_shape_pair,
                        contact_surface.face_contact_count,
                        self.midsole_shape_id,
                        self.ground_shape_id,
                        self.midsole_body_id,
                        self.model.body_com,
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        self.wp_spring_xy,
                        self.wp_spring_slack,
                        self.num_springs,
                        self.wp_coupled_surface_state,
                        self.wp_params,
                        self.state_0.body_f,
                        self.wp_contact_wrench,
                        self.wp_contact_energy,
                        self.wp_dissipated_energy_total,
                        float(self.sim_dt),
                        self.wp_ground_bottom_displacement,
                        self.wp_ground_bottom_pressure_kpa,
                        self.wp_stack_displacement,
                        self.wp_stack_pressure_kpa,
                        # Grid lookup parameters
                        self.wp_grid_to_spring,
                        self.grid_min_u,
                        self.grid_min_v,
                        self.grid_spacing,
                        self.grid_num_u,
                        self.grid_num_v,
                    ],
                    device=self.device,
                )

            wp.launch(
                apply_bonded_top_forces_kernel,
                dim=self.num_springs,
                inputs=[
                    self.wp_spring_xy,
                    self.wp_spring_top,
                    self.wp_spring_slack,
                    self.wp_foot_contact_valid,
                    self.num_springs,
                    self.model.body_com,
                    self.state_0.body_q,
                    self.foot_body_id,
                    self.midsole_body_id,
                    self.wp_coupled_surface_state,
                    self.wp_params,
                    self.state_0.body_f,
                    self.wp_contact_energy,
                    self.wp_dissipated_energy_total,
                    float(self.sim_dt),
                    self.wp_foot_top_pressure_kpa,
                    self.wp_stack_displacement,
                    self.wp_stack_pressure_kpa,
                ],
                device=self.device,
            )

            wp.launch(
                update_shoe_stats_kernel,
                dim=1,
                inputs=[
                    self.wp_contact_wrench,
                    self.wp_contact_energy,
                    self.wp_dissipated_energy_total,
                    self.wp_step_stats,
                ],
                device=self.device,
            )

            # Step solver
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            self._update_peak_maps()

            if self.debug:
                body_q_np = self.state_0.body_q.numpy()
                foot_z_debug = float(body_q_np[self.foot_body_id, 2])
                midsole_z_debug = float(body_q_np[self.midsole_body_id, 2])
                joint_q_np = self.state_0.joint_q.numpy()
                shoe_rel_q_debug = float(joint_q_np[self.midsole_joint_q_start])
                face_count_debug = 0
                if contact_surface is not None:
                    face_count_debug = int(contact_surface.face_contact_count.numpy()[0])
                print(
                    f"[DEBUG_STEP] t={self.sim_time:.4f} "
                    f"foot_z={foot_z_debug:.6f} midsole_z={midsole_z_debug:.6f} "
                    f"shoe_rel_q={shoe_rel_q_debug:.6f} "
                    f"contacts={self.contacts.rigid_contact_count.numpy()[0]} "
                    f"face_count={face_count_debug}"
                )

            if self.save_plots:
                body_q_np = self.state_0.body_q.numpy()
                foot_z_plot = float(body_q_np[self.foot_body_id, 2])
                midsole_z_plot = float(body_q_np[self.midsole_body_id, 2])
                self._sync_step_stats_from_warp()
                face_count_plot = 0
                if contact_surface is not None:
                    face_count_plot = int(contact_surface.face_contact_count.numpy()[0])
                top_surface_samples, ground_surface_samples = self._surface_samples_from_hydro(
                    contact_surface, face_count_plot
                )
                self.history_z.append(foot_z_plot)
                self.history_foot_z.append(foot_z_plot)
                self.history_midsole_z.append(midsole_z_plot)
                self.history_force.append(self.last_force_n)
                self.history_elastic_energy.append(self.last_elastic_energy_j)
                self.history_dissipated_energy.append(self.dissipated_energy_j)
                self.history_foot_disp.append(self.wp_foot_top_displacement.numpy().copy())
                self.history_foot_pres.append(self.wp_foot_top_pressure_kpa.numpy().copy())
                self.history_ground_disp.append(self.wp_ground_bottom_displacement.numpy().copy())
                self.history_ground_pres.append(self.wp_ground_bottom_pressure_kpa.numpy().copy())
                self.history_stack_disp.append(self.wp_stack_displacement.numpy().copy())
                self.history_stack_pres.append(self.wp_stack_pressure_kpa.numpy().copy())
                self.history_top_surface_samples.append(top_surface_samples)
                self.history_ground_surface_samples.append(ground_surface_samples)

            # Advance simulation time
            self.sim_time += self.sim_dt

        if self.debug or self.save_plots:
            self._sync_step_stats_from_warp()
            self._sync_peak_maps_from_warp()

    def step(self):
        self.simulate()

    def _log_contact_patch_mesh(
        self,
        name: str,
        active_points: np.ndarray,
        color: tuple[float, float, float],
    ):
        active_count = int(len(active_points))
        if active_count < 3:
            previous = self._contact_mesh_buckets.get(name)
            if previous is None:
                return
            previous_name, previous_capacity, previous_color = previous
            points_wp = wp.zeros(previous_capacity, dtype=wp.vec3, device=self.device)
            indices_wp = wp.zeros(0, dtype=wp.int32, device=self.device)
            self.viewer.log_mesh(
                previous_name,
                points=points_wp,
                indices=indices_wp,
                color=previous_color,
                backface_culling=False,
                hidden=True,
            )
            return

        active_faces = max(1, active_count // 3)
        capacity_faces = 1 << (active_faces - 1).bit_length()
        capacity = 3 * capacity_faces
        bucket_name = f"{name}_{capacity}"

        previous = self._contact_mesh_buckets.get(name)
        if previous is not None and previous[0] != bucket_name:
            previous_name, previous_capacity, previous_color = previous
            self.viewer.log_mesh(
                previous_name,
                points=wp.zeros(previous_capacity, dtype=wp.vec3, device=self.device),
                indices=wp.zeros(0, dtype=wp.int32, device=self.device),
                color=previous_color,
                backface_culling=False,
                hidden=True,
            )
        self._contact_mesh_buckets[name] = (bucket_name, capacity, color)

        points = np.zeros((capacity, 3), dtype=np.float32)
        points[:active_count] = np.asarray(active_points, dtype=np.float32)

        points_wp = wp.array(points, dtype=wp.vec3, device=self.device)
        indices_wp = wp.array(np.arange(active_count), dtype=wp.int32, device=self.device)
        self.viewer.log_mesh(
            bucket_name,
            points=points_wp,
            indices=indices_wp,
            color=color,
            backface_culling=False,
            hidden=False,
        )

    def _log_compression_point_clouds(self, midsole_z: float):
        grid_xy = self.spring_grid.grid_uv_m.astype(np.float32)
        top_disp = self.wp_foot_top_displacement.numpy()
        top_pressure = self.wp_foot_top_pressure_kpa.numpy()
        bottom_disp = self.wp_ground_bottom_displacement.numpy()
        bottom_pressure = self.wp_ground_bottom_pressure_kpa.numpy()

        top_active = top_disp > 1.0e-6
        if np.any(top_active):
            top_z = self.start_z + self.spring_grid.top_m - top_disp
            top_points = np.column_stack((grid_xy[top_active, 0], grid_xy[top_active, 1], top_z[top_active])).astype(
                np.float32
            )
            top_colors = _colors_from_pressure(top_pressure[top_active], self.max_display_pressure_kpa)
            self.viewer.log_points(
                "/foot_shoe/top_compression_points",
                wp.array(top_points, dtype=wp.vec3, device=self.device),
                self.point_radius * 2.0,
                wp.array(top_colors, dtype=wp.vec3, device=self.device),
            )
        else:
            self.viewer.log_points("/foot_shoe/top_compression_points", None)

        bottom_active = bottom_disp > 1.0e-6
        if np.any(bottom_active):
            bottom_z = midsole_z + self.spring_grid.bottom_m + bottom_disp
            bottom_points = np.column_stack(
                (grid_xy[bottom_active, 0], grid_xy[bottom_active, 1], bottom_z[bottom_active])
            ).astype(np.float32)
            bottom_colors = _colors_from_pressure(bottom_pressure[bottom_active], self.max_display_pressure_kpa)
            self.viewer.log_points(
                "/foot_shoe/bottom_compression_points",
                wp.array(bottom_points, dtype=wp.vec3, device=self.device),
                self.point_radius * 2.0,
                wp.array(bottom_colors, dtype=wp.vec3, device=self.device),
            )
        else:
            self.viewer.log_points("/foot_shoe/bottom_compression_points", None)

    def render(self):
        body_q = self.state_0.body_q.numpy()
        foot_pos = body_q[self.foot_body_id, :3]
        midsole_pos = body_q[self.midsole_body_id, :3]

        # Retrieve the contact surface
        hydro = self.collision_pipeline.narrow_phase.hydroelastic_sdf
        contact_surface = hydro.get_contact_surface()

        face_count = 0
        if contact_surface is not None:
            face_count = int(contact_surface.face_contact_count.numpy()[0])

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self._log_compression_point_clouds(float(midsole_pos[2]))

        # Log contact patch if active
        if face_count > 0:
            points_np = contact_surface.contact_surface_point.numpy()[: 3 * face_count]
            pairs_np = contact_surface.contact_surface_shape_pair.numpy()[:face_count]

            # Mask for foot-midsole contact faces:
            is_foot_midsole = ((pairs_np[:, 0] == self.foot_shape_id) & (pairs_np[:, 1] == self.midsole_shape_id)) | (
                (pairs_np[:, 1] == self.foot_shape_id) & (pairs_np[:, 0] == self.midsole_shape_id)
            )
            # Mask for midsole-ground contact faces:
            is_midsole_ground = (
                (pairs_np[:, 0] == self.midsole_shape_id) & (pairs_np[:, 1] == self.ground_shape_id)
            ) | ((pairs_np[:, 1] == self.midsole_shape_id) & (pairs_np[:, 0] == self.ground_shape_id))

            # Extract points for foot-midsole:
            foot_midsole_points_mask = np.repeat(is_foot_midsole, 3)
            foot_midsole_points = points_np[foot_midsole_points_mask]

            if len(foot_midsole_points) > 0:
                self._log_contact_patch_mesh(
                    "/foot_shoe/contact_foot_midsole",
                    foot_midsole_points,
                    (0.1, 0.8, 0.4),
                )
            else:
                self._log_contact_patch_mesh(
                    "/foot_shoe/contact_foot_midsole",
                    np.empty((0, 3), dtype=np.float32),
                    (0.1, 0.8, 0.4),
                )

            # Extract points for midsole-ground:
            midsole_ground_points_mask = np.repeat(is_midsole_ground, 3)
            midsole_ground_points = points_np[midsole_ground_points_mask]

            if len(midsole_ground_points) > 0:
                self._log_contact_patch_mesh(
                    "/foot_shoe/contact_midsole_ground",
                    midsole_ground_points,
                    (0.9, 0.4, 0.1),
                )
            else:
                self._log_contact_patch_mesh(
                    "/foot_shoe/contact_midsole_ground",
                    np.empty((0, 3), dtype=np.float32),
                    (0.9, 0.4, 0.1),
                )
        else:
            self._log_contact_patch_mesh(
                "/foot_shoe/contact_foot_midsole",
                np.empty((0, 3), dtype=np.float32),
                (0.1, 0.8, 0.4),
            )
            self._log_contact_patch_mesh(
                "/foot_shoe/contact_midsole_ground",
                np.empty((0, 3), dtype=np.float32),
                (0.9, 0.4, 0.1),
            )

        self.viewer.log_array(
            "/foot_shoe/stats",
            np.asarray(
                [
                    self.sim_time,
                    foot_pos[2],
                    float(self.last_force_n),
                    float(self.peak_force_n),
                    float(self.last_elastic_energy_j),
                    float(self.peak_elastic_energy_j),
                    float(self.dissipated_energy_j),
                    float(
                        np.max(self.wp_ground_bottom_displacement.numpy())
                        if len(self.wp_ground_bottom_displacement.numpy()) > 0
                        else 0.0
                    ),
                    float(
                        np.max(self.wp_foot_top_displacement.numpy())
                        if len(self.wp_foot_top_displacement.numpy()) > 0
                        else 0.0
                    ),
                    float(
                        np.max(self.wp_stack_displacement.numpy())
                        if len(self.wp_stack_displacement.numpy()) > 0
                        else 0.0
                    ),
                    float(face_count),
                    float(self.state_0.joint_q.numpy()[self.midsole_joint_q_start]),
                ],
                dtype=np.float32,
            ),
        )
        self.viewer.end_frame()

    def test_final(self):
        self._sync_step_stats_from_warp()
        self._sync_peak_maps_from_warp()
        print(
            "Simulation completed. "
            f"Peak Vertical Force reached: {self.peak_force_n:.2f} N, "
            f"Peak Elastic Energy stored: {self.peak_elastic_energy_j:.4f} J, "
            f"Dissipated Energy: {self.dissipated_energy_j:.4f} J"
        )
        assert self.peak_force_n >= self.min_peak_force_n, (
            f"Expected peak vertical force to reach {self.min_peak_force_n:.2f} N, "
            f"but only reached {self.peak_force_n:.2f} N"
        )
        if self.save_plots:
            self._plot()

    def _plot(self):
        try:
            import matplotlib.pyplot as plt  # noqa: PLC0415
        except ImportError:
            print("matplotlib is not installed. Skipping plot generation.")
            return

        if len(self.history_z) == 0:
            print("No history recorded. Skipping plot generation.")
            return

        n = len(self.history_z)
        time = np.arange(n, dtype=np.float32) * self.sim_dt
        pos_foot_z = np.array(self.history_foot_z)
        pos_midsole_z = np.array(self.history_midsole_z)
        force_n = np.array(self.history_force)

        disp_foot_mm = (self.start_z - pos_foot_z) * 1000.0
        disp_midsole_mm = (self.start_z - pos_midsole_z) * 1000.0
        stack_disp_mm = np.asarray([np.max(frame) * 1000.0 for frame in self.history_stack_disp], dtype=np.float32)

        _fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        color = "tab:red"
        axs[0].set_xlabel("Time [s]")
        axs[0].set_ylabel("Force [N]", color=color)
        axs[0].plot(time, force_n, color=color, linewidth=2, label="Force")
        axs[0].tick_params(axis="y", labelcolor=color)
        axs[0].grid(True)

        axs0_twin = axs[0].twinx()
        axs0_twin.set_ylabel("Displacement [mm]")
        axs0_twin.plot(time, disp_foot_mm, color="tab:blue", linewidth=2, label="Foot")
        axs0_twin.plot(time, disp_midsole_mm, color="tab:cyan", linewidth=2, linestyle="--", label="Midsole")
        axs0_twin.plot(time, stack_disp_mm, color="tab:green", linewidth=2, linestyle=":", label="Stack")
        axs0_twin.tick_params(axis="y")
        axs0_twin.legend(loc="upper right")
        axs0_twin.set_title("Force & Displacement vs Time")

        axs[1].plot(disp_foot_mm, force_n, color="purple", linewidth=2.5, label="Foot")
        axs[1].plot(disp_midsole_mm, force_n, color="green", linewidth=2.0, linestyle="--", label="Midsole")
        axs[1].plot(stack_disp_mm, force_n, color="black", linewidth=2.0, linestyle=":", label="Coupled Stack")
        axs[1].set_xlabel("Displacement [mm]")
        axs[1].set_ylabel("Force [N]")
        axs[1].set_title("Force-Displacement Hysteresis Loops")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plot_path = "foot_shoe_hysteresis.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Hysteresis plot saved to {plot_path}")
        plt.close()

        top_surface = self._bonded_top_surface_peak_samples()
        ground_surface = (
            np.concatenate([s for s in self.history_ground_surface_samples if len(s) > 0], axis=0)
            if any(len(s) > 0 for s in self.history_ground_surface_samples)
            else np.empty((0, 5), dtype=np.float32)
        )
        stack_surface = (
            np.concatenate([top_surface, ground_surface], axis=0)
            if len(top_surface) > 0 or len(ground_surface) > 0
            else np.empty((0, 5), dtype=np.float32)
        )

        fig2, axs2 = plt.subplots(3, 2, figsize=(12, 14))
        pressure_vmax = (
            float(np.percentile(stack_surface[:, 4], 99.0))
            if len(stack_surface) > 0 and np.max(stack_surface[:, 4]) > 0.0
            else 1.0
        )

        def plot_surface(ax, samples, value_col, title, cmap, label, vmax=None):
            if len(samples) == 0:
                ax.set_title(f"{title} (no hydro faces)")
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("Width [mm]")
                ax.set_ylabel("Length [mm]")
                return None

            samples = samples[np.argsort(samples[:, value_col])]
            scatter = ax.scatter(
                samples[:, 0] * 1000.0,
                samples[:, 1] * 1000.0,
                c=samples[:, value_col],
                s=8,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                linewidths=0.0,
                alpha=0.72,
            )
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Width [mm]")
            ax.set_ylabel("Length [mm]")
            fig2.colorbar(scatter, ax=ax, label=label)
            return scatter

        plot_surface(
            axs2[0, 0],
            top_surface * np.array([1.0, 1.0, 1000.0, 1.0, 1.0], dtype=np.float32),
            2,
            "Bonded Top Interface Displacement Samples",
            "inferno",
            "Face displacement [mm]",
        )
        plot_surface(
            axs2[0, 1],
            top_surface,
            4,
            "Bonded Top Interface Coupled Pressure",
            "jet",
            "Pressure [kPa]",
            vmax=pressure_vmax,
        )
        plot_surface(
            axs2[1, 0],
            ground_surface * np.array([1.0, 1.0, 1000.0, 1.0, 1.0], dtype=np.float32),
            2,
            "Hydro Bottom Face Displacement Samples",
            "inferno",
            "Face displacement [mm]",
        )
        plot_surface(
            axs2[1, 1],
            ground_surface,
            4,
            "Hydro Bottom Face Coupled Pressure",
            "jet",
            "Pressure [kPa]",
            vmax=pressure_vmax,
        )
        plot_surface(
            axs2[2, 0],
            stack_surface * np.array([1.0, 1.0, 1.0, 1000.0, 1.0], dtype=np.float32),
            3,
            "Hydro Coupled Stack Displacement Samples",
            "inferno",
            "Stack displacement [mm]",
        )
        plot_surface(
            axs2[2, 1],
            stack_surface,
            4,
            "Hydro Coupled Stack Pressure Samples",
            "jet",
            "Pressure [kPa]",
            vmax=pressure_vmax,
        )

        plt.tight_layout()
        heatmap_path = "foot_shoe_peak_heatmap.png"
        fig2.savefig(heatmap_path, dpi=150)
        print(f"Peak heatmaps saved to {heatmap_path}")
        plt.close(fig2)

        try:
            from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: PLC0415

            max_anim_frames = 60
            if n > max_anim_frames:
                indices = np.linspace(0, n - 1, max_anim_frames, dtype=np.int64)
            else:
                indices = np.arange(n, dtype=np.int64)

            fig_anim, axs_anim = plt.subplots(2, 2, figsize=(12, 10))

            foot_disp_vmax = (
                np.max(top_surface[:, 2] * 1000.0) if len(top_surface) > 0 and np.max(top_surface[:, 2]) > 0.0 else 1.0
            )
            ground_disp_vmax = (
                np.max(ground_surface[:, 2] * 1000.0)
                if len(ground_surface) > 0 and np.max(ground_surface[:, 2]) > 0.0
                else 1.0
            )
            all_anim_samples = (
                np.concatenate([top_surface, ground_surface], axis=0)
                if len(top_surface) > 0 or len(ground_surface) > 0
                else np.column_stack(
                    (
                        self.spring_grid.grid_uv_m[:, 0],
                        self.spring_grid.grid_uv_m[:, 1],
                        np.zeros(self.num_springs),
                        np.zeros(self.num_springs),
                        np.zeros(self.num_springs),
                    )
                ).astype(np.float32)
            )
            xlim = (float(np.min(all_anim_samples[:, 0]) * 1000.0), float(np.max(all_anim_samples[:, 0]) * 1000.0))
            ylim = (float(np.min(all_anim_samples[:, 1]) * 1000.0), float(np.max(all_anim_samples[:, 1]) * 1000.0))

            def init_surface_scatter(ax, title_text, cmap, vmax, label):
                scatter = ax.scatter(
                    [],
                    [],
                    c=[],
                    s=10,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=vmax,
                    linewidths=0.0,
                    alpha=0.78,
                )
                ax.set_title(title_text)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel("Width [mm]")
                ax.set_ylabel("Length [mm]")
                fig_anim.colorbar(scatter, ax=ax, label=label)
                return scatter

            sc_foot_disp = init_surface_scatter(
                axs_anim[0, 0],
                "Bonded Top Interface Displacement [mm]",
                "inferno",
                foot_disp_vmax,
                "Displacement [mm]",
            )
            sc_foot_pres = init_surface_scatter(
                axs_anim[0, 1],
                "Bonded Top Interface Coupled Pressure [kPa]",
                "jet",
                pressure_vmax,
                "Pressure [kPa]",
            )
            sc_ground_disp = init_surface_scatter(
                axs_anim[1, 0],
                "Hydro Bottom Face Displacement [mm]",
                "inferno",
                ground_disp_vmax,
                "Displacement [mm]",
            )
            sc_ground_pres = init_surface_scatter(
                axs_anim[1, 1],
                "Hydro Bottom Face Coupled Pressure [kPa]",
                "jet",
                pressure_vmax,
                "Pressure [kPa]",
            )

            title = fig_anim.suptitle("")

            def update_anim(frame_idx):
                idx = indices[frame_idx]
                t_val = time[idx]
                f_val = force_n[idx]

                top_samples = self.history_top_surface_samples[idx]
                ground_samples = self.history_ground_surface_samples[idx]
                sc_foot_disp.set_offsets(top_samples[:, :2] * 1000.0)
                sc_foot_disp.set_array(top_samples[:, 2] * 1000.0)
                sc_foot_pres.set_offsets(top_samples[:, :2] * 1000.0)
                sc_foot_pres.set_array(top_samples[:, 4])
                sc_ground_disp.set_offsets(ground_samples[:, :2] * 1000.0)
                sc_ground_disp.set_array(ground_samples[:, 2] * 1000.0)
                sc_ground_pres.set_offsets(ground_samples[:, :2] * 1000.0)
                sc_ground_pres.set_array(ground_samples[:, 4])
                title.set_text(f"Foot-Shoe Impact | t={t_val:.3f} s | Force={f_val:.1f} N")
                return sc_foot_disp, sc_foot_pres, sc_ground_disp, sc_ground_pres, title

            anim_path = "foot_shoe_contact_heatmap.gif"
            anim = FuncAnimation(fig_anim, update_anim, frames=len(indices), interval=100, blit=False)
            anim.save(anim_path, writer=PillowWriter(fps=10), dpi=100)
            print(f"Heatmap video saved to {anim_path}")
            plt.close(fig_anim)
        except Exception as e:
            print(f"Failed to generate animation: {e}")

    def load_foot_config(self) -> dict[str, float]:
        config = {
            "foot_roll_deg": 0.0,
            "foot_pitch_deg": 0.0,
            "foot_yaw_deg": 0.0,
            "foot_offset_x_mm": 0.0,
            "foot_offset_y_mm": 0.0,
            "foot_offset_z_mm": 0.0,
        }
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                for key in config:
                    if key in data:
                        config[key] = float(data[key])
                print(f"[config] Loaded default foot configuration from {self.config_path}")
            except Exception as e:
                print(f"[config] Failed to load config from {self.config_path}: {e}")
        return config

    def save_foot_config(self):
        config = {
            "foot_roll_deg": self.foot_roll_deg,
            "foot_pitch_deg": self.foot_pitch_deg,
            "foot_yaw_deg": self.foot_yaw_deg,
            "foot_offset_x_mm": self.foot_offset_x_mm,
            "foot_offset_y_mm": self.foot_offset_y_mm,
            "foot_offset_z_mm": self.foot_offset_z_mm,
        }
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"[config] Saved foot configuration to {self.config_path}")
        except Exception as e:
            print(f"[config] Failed to save config to {self.config_path}: {e}")

    def setup_viser_gui(self):
        if not hasattr(self.viewer, "_server"):
            return

        server = self.viewer._server
        folder = server.gui.add_folder("Foot Alignment Setup")
        with folder:
            self.gui_simulate = server.gui.add_checkbox("Simulate", initial_value=True)
            self.gui_roll = server.gui.add_slider(
                "Roll (deg)", min=-45.0, max=45.0, step=0.5, initial_value=self.foot_roll_deg
            )
            self.gui_pitch = server.gui.add_slider(
                "Pitch (deg)", min=-45.0, max=45.0, step=0.5, initial_value=self.foot_pitch_deg
            )
            self.gui_yaw = server.gui.add_slider(
                "Yaw (deg)", min=-180.0, max=180.0, step=0.5, initial_value=self.foot_yaw_deg
            )
            self.gui_offset_x = server.gui.add_slider(
                "Offset X (mm)", min=-100.0, max=100.0, step=0.5, initial_value=self.foot_offset_x_mm
            )
            self.gui_offset_y = server.gui.add_slider(
                "Offset Y (mm)", min=-100.0, max=100.0, step=0.5, initial_value=self.foot_offset_y_mm
            )
            self.gui_offset_z = server.gui.add_slider(
                "Offset Z (mm)", min=-50.0, max=50.0, step=0.5, initial_value=self.foot_offset_z_mm
            )
            self.gui_apply = server.gui.add_button("Save & Rebuild SDF")

        @self.gui_roll.on_update
        def _(event):
            self.foot_roll_deg = self.gui_roll.value
            if not self.gui_simulate_enabled:
                self.update_foot_visual_pose()

        @self.gui_pitch.on_update
        def _(event):
            self.foot_pitch_deg = self.gui_pitch.value
            if not self.gui_simulate_enabled:
                self.update_foot_visual_pose()

        @self.gui_yaw.on_update
        def _(event):
            self.foot_yaw_deg = self.gui_yaw.value
            if not self.gui_simulate_enabled:
                self.update_foot_visual_pose()

        @self.gui_offset_x.on_update
        def _(event):
            self.foot_offset_x_mm = self.gui_offset_x.value
            if not self.gui_simulate_enabled:
                self.update_foot_visual_pose()

        @self.gui_offset_y.on_update
        def _(event):
            self.foot_offset_y_mm = self.gui_offset_y.value
            if not self.gui_simulate_enabled:
                self.update_foot_visual_pose()

        @self.gui_offset_z.on_update
        def _(event):
            self.foot_offset_z_mm = self.gui_offset_z.value
            if not self.gui_simulate_enabled:
                self.update_foot_visual_pose()

        @self.gui_simulate.on_update
        def _(event):
            self.gui_simulate_enabled = self.gui_simulate.value

        @self.gui_apply.on_click
        def _(event):
            self.save_foot_config()
            if hasattr(self.viewer, "_reset_callback") and self.viewer._reset_callback is not None:
                self.viewer._reset_callback()

    def gui(self, ui):
        ui.text("Foot Alignment Setup")

        _, self.gui_simulate_enabled = ui.checkbox("Simulate", self.gui_simulate_enabled)

        changed_roll, self.foot_roll_deg = ui.slider_float("Roll (deg)", self.foot_roll_deg, -45.0, 45.0)
        changed_pitch, self.foot_pitch_deg = ui.slider_float("Pitch (deg)", self.foot_pitch_deg, -45.0, 45.0)
        changed_yaw, self.foot_yaw_deg = ui.slider_float("Yaw (deg)", self.foot_yaw_deg, -180.0, 180.0)
        changed_ox, self.foot_offset_x_mm = ui.slider_float("Offset X (mm)", self.foot_offset_x_mm, -100.0, 100.0)
        changed_oy, self.foot_offset_y_mm = ui.slider_float("Offset Y (mm)", self.foot_offset_y_mm, -100.0, 100.0)
        changed_oz, self.foot_offset_z_mm = ui.slider_float("Offset Z (mm)", self.foot_offset_z_mm, -50.0, 50.0)

        if changed_roll or changed_pitch or changed_yaw or changed_ox or changed_oy or changed_oz:
            if not self.gui_simulate_enabled:
                self.update_foot_visual_pose()

        if ui.button("Save & Rebuild SDF"):
            self.save_foot_config()
            if hasattr(self.viewer, "_reset_callback") and self.viewer._reset_callback is not None:
                self.viewer._reset_callback()

    def update_foot_visual_pose(self):
        roll = self.foot_roll_deg
        pitch = self.foot_pitch_deg
        yaw = self.foot_yaw_deg
        ox = self.foot_offset_x_mm
        oy = self.foot_offset_y_mm
        oz = self.foot_offset_z_mm

        q_init = _quat_from_euler_xyz_deg(self.init_roll, self.init_pitch, self.init_yaw)
        q_slider = _quat_from_euler_xyz_deg(roll, pitch, yaw)

        q_init_conj = np.array([-q_init[0], -q_init[1], -q_init[2], q_init[3]])

        def quat_mul(q1, q2):
            x1, y1, z1, w1 = q1
            x2, y2, z2, w2 = q2
            return np.array(
                [
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ]
            )

        q_delta = quat_mul(q_slider, q_init_conj)

        dx = (ox - self.init_ox) * 0.001
        dy = (oy - self.init_oy) * 0.001
        dz = (oz - self.init_oz) * 0.001

        body_q_np = self.state_0.body_q.numpy()
        body_q_np[self.foot_body_id, :3] = np.array([dx, dy, self.start_z + dz])
        body_q_np[self.foot_body_id, 3:7] = q_delta
        self.state_0.body_q.assign(body_q_np)

        joint_q_np = self.state_0.joint_q.numpy()
        joint_q_np[self.foot_joint_q_start : self.foot_joint_q_start + 3] = np.array([dx, dy, self.start_z + dz])
        joint_q_np[self.foot_joint_q_start + 3 : self.foot_joint_q_start + 7] = q_delta
        self.state_0.joint_q.assign(joint_q_np)

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Freeze midsole world pose at default starting transform
        body_q_np = self.state_0.body_q.numpy()
        body_q_np[self.midsole_body_id, :3] = np.array([0.0, 0.0, self.start_z])
        body_q_np[self.midsole_body_id, 3:7] = np.array([0.0, 0.0, 0.0, 1.0])
        self.state_0.body_q.assign(body_q_np)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--manifest", default="DigitalInstron/manifest_v2.json", help="Path to manifest file")
        parser.add_argument("--output-dir", default=None, help="Directory for cache")
        parser.add_argument(
            "--material",
            default="DigitalInstron/processed/v2_cache/digital_instron_v2_foundation_material.json",
            help="Calibrated Digital Instron material JSON path used by --contact-law calibrated-ogden",
        )
        parser.add_argument(
            "--contact-law",
            choices=("calibrated-ogden", "kim-hyperfoam", "kim-layered"),
            default="kim-hyperfoam",
            help="Hydroelastic pressure law: calibrated Ogden, single Kim hyperfoam, or layered Kim hyperfoam.",
        )
        parser.add_argument(
            "--foam-material",
            choices=tuple(KIM_HYPERFOAM_MATERIALS),
            default="peba",
            help="Kim et al. midsole foam preset used by --contact-law kim-hyperfoam.",
        )
        parser.add_argument(
            "--upper-foam-material",
            choices=tuple(KIM_HYPERFOAM_MATERIALS),
            default="peba",
            help="Upper midsole foam preset used by --contact-law kim-layered.",
        )
        parser.add_argument(
            "--lower-foam-material",
            choices=tuple(KIM_HYPERFOAM_MATERIALS),
            default="peba",
            help="Lower midsole foam preset used by --contact-law kim-layered.",
        )
        parser.add_argument(
            "--upper-layer-fraction",
            type=float,
            default=0.5,
            help="Fraction of stack height assigned to the upper foam layer in --contact-law kim-layered.",
        )
        parser.add_argument(
            "--kim-damping-pa-s",
            type=float,
            default=2.5e4,
            help="Viscous pressure damping [Pa*s] added to the Kim et al. hyperfoam pressure field.",
        )
        parser.add_argument(
            "--hydro-force-scale",
            type=float,
            default=1.0,
            help=("Debug multiplier applied to custom hydroelastic pressure integration after stress * contact area."),
        )
        parser.add_argument(
            "--min-peak-force-n",
            type=float,
            default=None,
            help="Override the example test_final peak-force gate [N].",
        )
        parser.add_argument(
            "--save-plots",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Write hysteresis and heatmap plot artifacts during test_final.",
        )
        parser.add_argument("--foot-mesh", default="FeetFinder/0002-B.obj", help="Foot OBJ mesh path")
        parser.add_argument(
            "--mirror-foot",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Mirror the right-foot mesh laterally to match the left shoe bed",
        )
        parser.add_argument(
            "--kinematic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run kinematic trajectory or dynamic simulation",
        )
        parser.add_argument(
            "--foot-yaw-deg",
            type=float,
            default=0.0,
            help="Rotation angle of foot about Z-axis in degrees",
        )
        parser.add_argument(
            "--foot-pitch-deg",
            type=float,
            default=0.0,
            help="Rotation angle of foot about Y-axis (pitch) in degrees",
        )
        parser.add_argument(
            "--foot-roll-deg",
            type=float,
            default=0.0,
            help="Rotation angle of foot about X-axis (roll) in degrees",
        )
        parser.add_argument(
            "--foot-offset-x-mm",
            type=float,
            default=0.0,
            help="Translation offset of foot along X-axis in mm",
        )
        parser.add_argument(
            "--foot-offset-y-mm",
            type=float,
            default=0.0,
            help="Translation offset of foot along Y-axis in mm",
        )
        parser.add_argument(
            "--foot-offset-z-mm",
            type=float,
            default=0.0,
            help="Translation offset of foot along Z-axis in mm",
        )
        parser.add_argument(
            "--shoe-attach-mode",
            choices=("world-slide", "foot-prismatic"),
            default="foot-prismatic",
            help=(
                "How the midsole/shoe body is attached. "
                "'world-slide' preserves the old world prismatic joint. "
                "'foot-prismatic' attaches the shoe to the foot with a local vertical slide."
            ),
        )
        parser.add_argument(
            "--shoe-compression-limit-mm",
            type=float,
            default=35.0,
            help="Maximum allowed shoe-to-foot compression travel along the prismatic axis [mm].",
        )
        parser.add_argument(
            "--shoe-lift-limit-mm",
            type=float,
            default=3.0,
            help="Small allowed lift/clearance travel in the opposite direction [mm].",
        )
        parser.add_argument(
            "--shoe-joint-limit-ke",
            type=float,
            default=1.0e5,
            help="Prismatic joint limit stiffness for foot-shoe attachment.",
        )
        parser.add_argument(
            "--shoe-joint-limit-kd",
            type=float,
            default=1.0e3,
            help="Prismatic joint limit damping for foot-shoe attachment.",
        )
        parser.add_argument(
            "--shoe-joint-friction",
            type=float,
            default=2.0,
            help="Friction/damping-like resistance along the foot-shoe prismatic joint.",
        )
        parser.add_argument("--debug", action="store_true", help="Print collision and SDF diagnostics every substep.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    if not args.test:
        example.test_final()
