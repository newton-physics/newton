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
    "peba": KimHyperfoamMaterial("PEBA", 90.0, 0.112e6, 5.050, 0.30),
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
STATS_LAST_FORCE_N = 0
STATS_PEAK_FORCE_N = 1
STATS_LAST_PLATE_TORQUE_NM = 2
PLATE_ACCUM_REAR_DISP = 0
PLATE_ACCUM_REAR_COUNT = 1
PLATE_ACCUM_REAR_Y = 2
PLATE_ACCUM_FORE_DISP = 3
PLATE_ACCUM_FORE_COUNT = 4
PLATE_ACCUM_FORE_Y = 5


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
def evaluate_contact_stress(strain: float, comp_vel: float, params: wp.array[float]) -> float:
    """Evaluate the selected pressure law for a local compressive strain."""
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

    damping_weight = wp.pow(wp.max(strain, 1.0e-8), wp.max(params[PARAM_DAMPING_POWER], 0.0))
    viscous_stress = params[PARAM_DAMPING] * damping_weight * comp_vel
    return wp.max(elastic_stress + viscous_stress, 0.0)


@wp.kernel
def apply_non_contact_forces_kernel(
    body_f: wp.array[wp.spatial_vector],
    foot_body_id: int,
    midsole_body_id: int,
    foot_mass: float,
    midsole_mass: float,
    gravity: float,
    ext_force: float,
    plate_torque: wp.array[float],
    kinematic: int,
):
    # Foot body: apply gravity, external force, and plate torque (if not kinematic)
    if kinematic == 0:
        foot_grav = foot_mass * gravity - ext_force
        torque = wp.vec3(plate_torque[0], plate_torque[1], plate_torque[2])
        body_f[foot_body_id] = wp.spatial_vector(wp.vec3(0.0, 0.0, foot_grav), torque)
    else:
        body_f[foot_body_id] = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
        
    # Midsole body always experiences gravity (since it's dynamic)
    midsole_grav = midsole_mass * gravity
    body_f[midsole_body_id] = wp.spatial_vector(wp.vec3(0.0, 0.0, midsole_grav), wp.vec3(0.0))


@wp.kernel
def evaluate_hydroelastic_ogden_kernel(
    points: wp.array[wp.vec3f],  # World-space positions of contact surface triangle vertices (3 per face)
    depths: wp.array[wp.float32],  # Penetration depth at each face centroid
    shape_pairs: wp.array[wp.vec2i],  # Shape pair indices (shape_a, shape_b) for each face
    face_count_ptr: wp.array[wp.int32],  # Active face count
    foot_shape_idx: int,
    midsole_shape_idx: int,
    ground_shape_idx: int,
    foot_body_idx: int,
    midsole_body_idx: int,
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    spring_xy: wp.array[wp.vec2],
    spring_slack: wp.array[float],
    num_springs: int,
    params: wp.array[float],
    body_f: wp.array[wp.spatial_vector],  # Output accumulated body forces
    wrench_out: wp.array[float],  # Output accumulated wrench (midsole-ground)
    foot_displacement_out: wp.array[float],
    foot_pressure_kpa_out: wp.array[float],
    ground_displacement_out: wp.array[float],
    ground_pressure_kpa_out: wp.array[float],
):
    tid = wp.tid()
    face_count = face_count_ptr[0]
    if tid >= face_count:
        return

    # Filter for contact pair
    pair = shape_pairs[tid]
    is_foot_midsole = False
    is_midsole_ground = False

    if (pair[0] == foot_shape_idx and pair[1] == midsole_shape_idx) or (pair[1] == foot_shape_idx and pair[0] == midsole_shape_idx):
        is_foot_midsole = True
    elif (pair[0] == midsole_shape_idx and pair[1] == ground_shape_idx) or (pair[1] == midsole_shape_idx and pair[0] == ground_shape_idx):
        is_midsole_ground = True

    if not is_foot_midsole and not is_midsole_ground:
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
    nearest = find_nearest_spring(centroid, spring_xy, num_springs)
    local_thick = float(0.01)  # Default 10mm fallback
    if nearest != -1:
        local_thick = spring_slack[nearest]

    # Calculate vertical displacement relative to depths (negative in Warp)
    displacement = -depths[tid]
    if displacement <= 0.0:
        return

    strain = wp.clamp(displacement / local_thick, 0.0, 0.99)

    # 3. Calculate compressive velocity at contact point
    comp_vel = float(0.0)
    if is_foot_midsole:
        foot_com = wp.transform_point(body_q[foot_body_idx], body_com[foot_body_idx])
        midsole_com = wp.transform_point(body_q[midsole_body_idx], body_com[midsole_body_idx])
        
        qd_foot = body_qd[foot_body_idx]
        w_foot = wp.spatial_bottom(qd_foot)
        v_foot = wp.spatial_top(qd_foot)
        
        qd_midsole = body_qd[midsole_body_idx]
        w_midsole = wp.spatial_bottom(qd_midsole)
        v_midsole = wp.spatial_top(qd_midsole)
        
        v_foot_c = v_foot + wp.cross(w_foot, centroid - foot_com)
        v_midsole_c = v_midsole + wp.cross(w_midsole, centroid - midsole_com)
        
        v_rel = v_foot_c - v_midsole_c
        comp_vel = -wp.dot(v_rel, normal)
    else:
        # Midsole-ground contact (ground is static)
        midsole_com = wp.transform_point(body_q[midsole_body_idx], body_com[midsole_body_idx])
        
        qd_midsole = body_qd[midsole_body_idx]
        w_midsole = wp.spatial_bottom(qd_midsole)
        v_midsole = wp.spatial_top(qd_midsole)
        
        v_midsole_c = v_midsole + wp.cross(w_midsole, centroid - midsole_com)
        comp_vel = -wp.dot(v_midsole_c, normal)

    # 4. Evaluate contact stress and force
    stress = evaluate_contact_stress(strain, comp_vel, params)
    force_magnitude = stress * area * 20.0
    force_vec = normal * force_magnitude

    # 5. Apply forces and torques to the bodies
    if is_foot_midsole:
        foot_com = wp.transform_point(body_q[foot_body_idx], body_com[foot_body_idx])
        midsole_com = wp.transform_point(body_q[midsole_body_idx], body_com[midsole_body_idx])
        
        r_foot = centroid - foot_com
        r_midsole = centroid - midsole_com
        
        # Push foot UP (+force_vec), push midsole DOWN (-force_vec)
        wp.atomic_add(body_f, foot_body_idx, wp.spatial_vector(force_vec, wp.cross(r_foot, force_vec)))
        wp.atomic_sub(body_f, midsole_body_idx, wp.spatial_vector(force_vec, wp.cross(r_midsole, force_vec)))
        
        if nearest != -1:
            wp.atomic_max(foot_displacement_out, nearest, displacement)
            wp.atomic_max(foot_pressure_kpa_out, nearest, stress * 0.001)
    else:
        # Midsole-ground contact: push midsole UP
        midsole_com = wp.transform_point(body_q[midsole_body_idx], body_com[midsole_body_idx])
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
        
        if nearest != -1:
            wp.atomic_max(ground_displacement_out, nearest, displacement)
            wp.atomic_max(ground_pressure_kpa_out, nearest, stress * 0.001)


@wp.kernel
def integrate_shoe_foundation_kernel(
    grid_xy: wp.array[wp.vec2],
    top_m: wp.array[float],
    foot_sole_z_m: wp.array[float],
    foot_contact_valid: wp.array[wp.int32],
    slack_length_m: wp.array[float],
    spring_count: int,
    body_z: float,
    start_z: float,
    spacing_m: float,
    params: wp.array[float],
    foot_vel: wp.vec3,
    foot_omega: wp.vec3,
    foot_com: wp.vec3,
    wrench_out: wp.array[float],
):
    spring = wp.tid()
    if spring >= spring_count:
        return
    if foot_contact_valid[spring] == 0:
        return

    slack = wp.max(slack_length_m[spring], 1.0e-6)
    top_rest_world = start_z + top_m[spring]
    foot_sole_world = body_z + foot_sole_z_m[spring]
    displacement = wp.clamp(top_rest_world - foot_sole_world, 0.0, slack)
    if displacement <= 0.0:
        return

    strain = wp.clamp(displacement / slack, 0.0, 0.99)

    xy = grid_xy[spring]
    point = wp.vec3(xy[0], xy[1], top_rest_world)
    normal = wp.vec3(0.0, 0.0, 1.0)
    r = point - foot_com
    v_contact = foot_vel + wp.cross(foot_omega, r)
    comp_vel = -wp.dot(v_contact, normal)

    stress = evaluate_contact_stress(strain, comp_vel, params)
    area = spacing_m * spacing_m
    force_vec = normal * (stress * area)
    torque = wp.cross(r, force_vec)

    wp.atomic_add(wrench_out, 0, force_vec[0])
    wp.atomic_add(wrench_out, 1, force_vec[1])
    wp.atomic_add(wrench_out, 2, force_vec[2])
    wp.atomic_add(wrench_out, 3, torque[0])
    wp.atomic_add(wrench_out, 4, torque[1])
    wp.atomic_add(wrench_out, 5, torque[2])


@wp.kernel
def accumulate_plate_bending_kernel(
    grid_xy: wp.array[wp.vec2],
    top_m: wp.array[float],
    foot_sole_z_m: wp.array[float],
    foot_contact_valid: wp.array[wp.int32],
    slack_length_m: wp.array[float],
    spring_count: int,
    body_z: float,
    start_z: float,
    rear_cut_y: float,
    fore_cut_y: float,
    accum: wp.array[float],
):
    spring = wp.tid()
    if spring >= spring_count:
        return
    if foot_contact_valid[spring] == 0:
        return

    xy = grid_xy[spring]
    y = xy[1]
    if y >= rear_cut_y and y <= fore_cut_y:
        return

    top_rest_world = start_z + top_m[spring]
    foot_sole_world = body_z + foot_sole_z_m[spring]
    displacement = wp.clamp(top_rest_world - foot_sole_world, 0.0, wp.max(slack_length_m[spring], 1.0e-6))

    if y < rear_cut_y:
        wp.atomic_add(accum, PLATE_ACCUM_REAR_DISP, displacement)
        wp.atomic_add(accum, PLATE_ACCUM_REAR_COUNT, 1.0)
        wp.atomic_add(accum, PLATE_ACCUM_REAR_Y, y)
    else:
        wp.atomic_add(accum, PLATE_ACCUM_FORE_DISP, displacement)
        wp.atomic_add(accum, PLATE_ACCUM_FORE_COUNT, 1.0)
        wp.atomic_add(accum, PLATE_ACCUM_FORE_Y, y)


@wp.kernel
def finalize_plate_bending_kernel(
    accum: wp.array[float],
    pitch_rate: float,
    plate_params: wp.array[float],
    torque_out: wp.array[float],
):
    rear_count = accum[PLATE_ACCUM_REAR_COUNT]
    fore_count = accum[PLATE_ACCUM_FORE_COUNT]
    if rear_count <= 0.0 or fore_count <= 0.0:
        torque_out[0] = 0.0
        torque_out[1] = 0.0
        torque_out[2] = 0.0
        return

    rear_disp = accum[PLATE_ACCUM_REAR_DISP] / rear_count
    fore_disp = accum[PLATE_ACCUM_FORE_DISP] / fore_count
    rear_y = accum[PLATE_ACCUM_REAR_Y] / rear_count
    fore_y = accum[PLATE_ACCUM_FORE_Y] / fore_count
    lever = wp.max(fore_y - rear_y, 1.0e-4)
    theta = wp.atan2(fore_disp - rear_disp, lever)

    young_pa = plate_params[0]
    thickness_m = plate_params[1]
    poisson = plate_params[2]
    width_m = plate_params[3]
    length_m = plate_params[4]
    damping_ratio = plate_params[5]
    plate_d = young_pa * thickness_m * thickness_m * thickness_m / (12.0 * (1.0 - poisson * poisson))
    pitch_stiffness = plate_d * width_m / wp.max(length_m, 1.0e-4)
    damping = damping_ratio * pitch_stiffness

    torque_out[0] = 0.0
    torque_out[1] = -pitch_stiffness * theta - damping * pitch_rate
    torque_out[2] = 0.0


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
    size: int,
):
    tid = wp.tid()
    if tid >= size:
        return
    peak_displacement[tid] = wp.max(peak_displacement[tid], displacement[tid])
    peak_pressure[tid] = wp.max(peak_pressure[tid], pressure[tid])
    peak_foot_displacement[tid] = wp.max(peak_foot_displacement[tid], foot_displacement[tid])
    peak_foot_pressure[tid] = wp.max(peak_foot_pressure[tid], foot_pressure[tid])


@wp.kernel
def evaluate_pressure_maps_kernel(
    top_m: wp.array[float],
    bottom_m: wp.array[float],
    foot_sole_z_m: wp.array[float],
    foot_contact_valid: wp.array[wp.int32],
    slack_length_m: wp.array[float],
    spring_count: int,
    body_z: float,
    start_z: float,
    vertical_velocity: float,
    params: wp.array[float],
    foot_displacement_out: wp.array[float],
    foot_pressure_kpa_out: wp.array[float],
    ground_displacement_out: wp.array[float],
    ground_pressure_kpa_out: wp.array[float],
    peak_foot_displacement_out: wp.array[float],
    peak_foot_pressure_kpa_out: wp.array[float],
    peak_ground_displacement_out: wp.array[float],
    peak_ground_pressure_kpa_out: wp.array[float],
    update_peak: int,
):
    spring = wp.tid()
    if spring >= spring_count:
        return

    slack = wp.max(slack_length_m[spring], 1.0e-6)

    ground_displacement = wp.clamp(-(body_z + bottom_m[spring]), 0.0, slack)
    ground_comp_vel = float(0.0)
    if ground_displacement > 1.0e-6:
        ground_comp_vel = -vertical_velocity
    ground_strain = wp.clamp(ground_displacement / slack, 0.0, 0.99)
    ground_stress = evaluate_contact_stress(ground_strain, ground_comp_vel, params)

    foot_displacement = float(0.0)
    foot_stress = float(0.0)
    if foot_contact_valid[spring] != 0:
        top_rest_world = start_z + top_m[spring]
        foot_sole_world = body_z + foot_sole_z_m[spring]
        foot_displacement = wp.clamp(top_rest_world - foot_sole_world, 0.0, slack)
        foot_comp_vel = float(0.0)
        if foot_displacement > 1.0e-6:
            foot_comp_vel = -vertical_velocity
        foot_strain = wp.clamp(foot_displacement / slack, 0.0, 0.99)
        foot_stress = evaluate_contact_stress(foot_strain, foot_comp_vel, params)

    foot_displacement_out[spring] = foot_displacement
    foot_pressure_kpa_out[spring] = foot_stress * 0.001
    ground_displacement_out[spring] = ground_displacement
    ground_pressure_kpa_out[spring] = ground_stress * 0.001

    if update_peak != 0:
        peak_foot_displacement_out[spring] = wp.max(peak_foot_displacement_out[spring], foot_displacement)
        peak_foot_pressure_kpa_out[spring] = wp.max(peak_foot_pressure_kpa_out[spring], foot_stress * 0.001)
        peak_ground_displacement_out[spring] = wp.max(peak_ground_displacement_out[spring], ground_displacement)
        peak_ground_pressure_kpa_out[spring] = wp.max(peak_ground_pressure_kpa_out[spring], ground_stress * 0.001)


@wp.kernel
def apply_shoe_body_force_kernel(
    body_f: wp.array[wp.spatial_vector],
    body_index: int,
    wrench: wp.array[float],
    plate_torque: wp.array[float],
    gravity_force_z: float,
):
    force = wp.vec3(wrench[0], wrench[1], wrench[2] + gravity_force_z)
    torque = wp.vec3(wrench[3] + plate_torque[0], wrench[4] + plate_torque[1], wrench[5] + plate_torque[2])
    body_f[body_index] = wp.spatial_vector(force, torque)


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
def update_shoe_stats_kernel(
    wrench: wp.array[float],
    plate_torque: wp.array[float],
    stats: wp.array[float],
):
    last_force = wrench[2]
    stats[STATS_LAST_FORCE_N] = last_force
    stats[STATS_PEAK_FORCE_N] = wp.max(stats[STATS_PEAK_FORCE_N], last_force)
    stats[STATS_LAST_PLATE_TORQUE_NM] = plate_torque[1]


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
            self.min_peak_force_n = 50.0 if self.contact_law in ("kim-hyperfoam", "kim-layered") else 500.0
        else:
            self.min_peak_force_n = args.min_peak_force_n
        self.enable_plate = bool(args.enable_plate)
        self.plate_thickness_m = args.plate_thickness_mm * 0.001
        self.plate_width_m = args.plate_width_mm * 0.001
        self.plate_length_m = args.plate_length_mm * 0.001
        self.plate_young_pa = args.plate_young_gpa * 1.0e9
        self.plate_poisson = args.plate_poisson

        # 1. Load resources from manifest and calibrated v2 cache
        self.manifest = load_manifest(args.manifest)
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
        foot_v_transformed[:, 0] = sign * foot_v[:, 1]  # X_midsole = width
        foot_v_transformed[:, 1] = foot_v[:, 0]  # Y_midsole = length
        foot_v_transformed[:, 2] = foot_v[:, 2]  # Z_midsole = height

        foot_f_transformed = foot_f.copy()
        if args.mirror_foot:
            foot_f_transformed[:, [1, 2]] = foot_f_transformed[:, [2, 1]]

        # Center horizontally on the spring grid
        foot_center = 0.5 * (np.min(foot_v_transformed, axis=0) + np.max(foot_v_transformed, axis=0))
        midsole_center = 0.5 * (np.min(self.spring_grid.grid_uv_m, axis=0) + np.max(self.spring_grid.grid_uv_m, axis=0))
        foot_v_transformed[:, 0] += midsole_center[0] - foot_center[0]
        foot_v_transformed[:, 1] += midsole_center[1] - foot_center[1]

        # Align foot yaw
        foot_yaw = np.radians(args.foot_yaw_deg)
        cos_yaw = np.cos(foot_yaw)
        sin_yaw = np.sin(foot_yaw)

        foot_center = 0.5 * (np.min(foot_v_transformed, axis=0) + np.max(foot_v_transformed, axis=0))
        foot_v_rel = foot_v_transformed - foot_center

        x_rot = foot_v_rel[:, 0] * cos_yaw - foot_v_rel[:, 1] * sin_yaw
        y_rot = foot_v_rel[:, 0] * sin_yaw + foot_v_rel[:, 1] * cos_yaw

        foot_v_transformed[:, 0] = x_rot + foot_center[0]
        foot_v_transformed[:, 1] = y_rot + foot_center[1]

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
            foot_v_transformed[:, 2] += Z_offset
            z_foot_sole[valid] += Z_offset

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
            label="foot_body",
        )
        self.foot_joint_id = builder.add_joint_free(
            child=self.foot_body_id,
            label="foot_free_joint",
        )

        foot_mesh = newton.Mesh(foot_v_transformed, foot_f_transformed.reshape(-1))

        # Build SDF for foot mesh
        foot_mesh.build_sdf(
            max_resolution=64,
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

        midsole_mesh = newton.Mesh(midsole_v_transformed, midsole_f_transformed.reshape(-1))
        if self.debug:
            print(
                f"[DEBUG] Midsole mesh bounds: min={np.min(midsole_v_transformed, axis=0)}, "
                f"max={np.max(midsole_v_transformed, axis=0)}"
            )

        # Build SDF for midsole mesh
        midsole_mesh.build_sdf(
            max_resolution=64,
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
            label="midsole_body",
        )
        # Add prismatic joint connecting foot and midsole
        self.midsole_joint_id = builder.add_joint_prismatic(
            parent=self.foot_body_id,
            child=self.midsole_body_id,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=newton.Axis.Z,
            target_pos=0.0,
            target_ke=1e5,
            target_kd=1e3,
            limit_lower=-0.05,
            limit_upper=0.05,
            collision_filter_parent=False,  # Allow them to collide!
            label="midsole_prismatic_joint",
        )
        # Add articulation
        builder.add_articulation([self.foot_joint_id, self.midsole_joint_id], label="foot_shoe_articulation")

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
            reduce_contacts=True,
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

        # Instantiate the collision pipeline
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=2000,
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
        valid_y = self.spring_grid.grid_uv_m[self.foot_contact_valid, 1]
        if len(valid_y) > 0:
            self.plate_rear_cut_y = float(np.percentile(valid_y, 35.0))
            self.plate_fore_cut_y = float(np.percentile(valid_y, 65.0))
        else:
            self.plate_rear_cut_y = 0.0
            self.plate_fore_cut_y = 0.0
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
            ],
            dtype=float,
            device=self.device,
        )
        self.wp_plate_params = wp.array(
            [
                self.plate_young_pa,
                self.plate_thickness_m,
                self.plate_poisson,
                self.plate_width_m,
                self.plate_length_m,
                0.05,
            ],
            dtype=float,
            device=self.device,
        )

        self.current_z = self.start_z
        self.current_vz = 0.0
        self.save_plots = bool(args.save_plots)
        self.use_hydro_surface_wrench = bool(args.use_hydro_surface_wrench)

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

        # Logging / stats
        self.peak_force_n = 0.0
        self.history_z = []
        self.history_force = []
        self.last_force_n = 0.0
        self.last_plate_torque_nm = 0.0
        self.peak_compression_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_foot_top_displacement_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_foot_top_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_ground_bottom_displacement_m = np.zeros_like(self.spring_grid.slack_length_m)
        self.peak_ground_bottom_pressure_kpa = np.zeros_like(self.spring_grid.slack_length_m)
        self.wp_foot_top_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_foot_top_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_ground_bottom_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_ground_bottom_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_peak_foot_top_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_peak_foot_top_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_peak_ground_bottom_displacement = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_peak_ground_bottom_pressure_kpa = wp.zeros(self.num_springs, dtype=float, device=self.device)
        self.wp_contact_wrench = wp.zeros(6, dtype=float, device=self.device)
        self.wp_plate_accum = wp.zeros(6, dtype=float, device=self.device)
        self.wp_plate_torque = wp.zeros(3, dtype=float, device=self.device)
        self.wp_step_stats = wp.zeros(3, dtype=float, device=self.device)
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

        self.point_radius = max(float(self.spring_grid.spacing_m) * 0.22, 0.0008)
        self.contact_color = wp.vec3(0.05, 0.95, 0.52)
        self.max_display_compression = 0.02

    def _ground_bottom_displacement_m(self, body_z_m: float) -> np.ndarray:
        compression = np.maximum(-(body_z_m + self.spring_grid.bottom_m), 0.0)
        return np.minimum(compression, self.spring_grid.slack_length_m)

    def _foot_top_displacement_m(self, body_z_m: float) -> np.ndarray:
        displacement = np.zeros_like(self.spring_grid.slack_length_m)
        if np.any(self.foot_contact_valid):
            top_rest_world = self.start_z + self.spring_grid.top_m[self.foot_contact_valid]
            foot_sole_world = body_z_m + self.foot_sole_z_m[self.foot_contact_valid]
            displacement[self.foot_contact_valid] = np.maximum(top_rest_world - foot_sole_world, 0.0)
        return np.minimum(displacement, self.spring_grid.slack_length_m)

    def _lengths_and_velocities_from_displacement(
        self,
        displacement_m: np.ndarray,
        vertical_velocity_mps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        displacement = np.minimum(np.maximum(displacement_m, 0.0), self.spring_grid.slack_length_m)
        current_lengths = np.maximum(self.spring_grid.slack_length_m - displacement, 0.0)
        velocities = np.zeros_like(self.spring_grid.slack_length_m)
        velocities[displacement > 1.0e-6] = vertical_velocity_mps
        return current_lengths, velocities

    def _pressure_kpa_from_displacement(
        self,
        displacement_m: np.ndarray,
        vertical_velocity_mps: float,
    ) -> np.ndarray:
        current_lengths, velocities = self._lengths_and_velocities_from_displacement(
            displacement_m,
            vertical_velocity_mps,
        )
        if self.contact_law in ("kim-hyperfoam", "kim-layered"):
            pressures_pa = _compute_kim_pressures(
                displacement_m,
                self.spring_grid.slack_length_m,
                velocities,
                self.kim_material if self.contact_law == "kim-hyperfoam" else self.kim_upper_material,
                self.kim_damping_pa_s,
                None if self.contact_law == "kim-hyperfoam" else self.kim_lower_material,
                self.upper_layer_fraction,
            )
        else:
            pressures_pa = _compute_pressures(
                current_lengths,
                self.spring_grid.slack_length_m,
                velocities,
                self.material,
                self.spring_grid.spacing_m,
                self.neighbors,
            )
        return pressures_pa / 1000.0

    def _pressure_maps_from_warp(
        self,
        foot_z_m: float,
        midsole_z_m: float,
        update_peaks: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if update_peaks:
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
                    self.num_springs,
                ],
                device=self.device,
            )
        return (
            self.wp_foot_top_displacement.numpy(),
            self.wp_foot_top_pressure_kpa.numpy(),
            self.wp_ground_bottom_displacement.numpy(),
            self.wp_ground_bottom_pressure_kpa.numpy(),
        )

    def _sync_peak_maps_from_warp(self):
        self.peak_foot_top_displacement_m = self.wp_peak_foot_top_displacement.numpy()
        self.peak_foot_top_pressure_kpa = self.wp_peak_foot_top_pressure_kpa.numpy()
        self.peak_ground_bottom_displacement_m = self.wp_peak_ground_bottom_displacement.numpy()
        self.peak_ground_bottom_pressure_kpa = self.wp_peak_ground_bottom_pressure_kpa.numpy()
        self.peak_compression_m = self.peak_ground_bottom_displacement_m
        self.peak_pressure_kpa = self.peak_ground_bottom_pressure_kpa

    def _sync_step_stats_from_warp(self):
        stats = self.wp_step_stats.numpy()
        self.last_force_n = float(stats[STATS_LAST_FORCE_N])
        self.peak_force_n = float(stats[STATS_PEAK_FORCE_N])
        self.last_plate_torque_nm = float(stats[STATS_LAST_PLATE_TORQUE_NM])

    def _body_com_world(self, body_q: np.ndarray) -> np.ndarray:
        origin = body_q[self.foot_body_id, :3]
        if not hasattr(self.model, "body_com"):
            return origin

        body_com = self.model.body_com.numpy()[self.foot_body_id]
        if np.max(np.abs(body_com)) <= 1.0e-12:
            return origin
        return origin + _rotate_vec_by_quat(body_com, body_q[self.foot_body_id, 3:7])

    def _update_plate_bending_torque(self, body_z_m: float, pitch_rate: float) -> wp.array:
        self.wp_plate_torque.zero_()
        if not self.enable_plate:
            return self.wp_plate_torque

        if not np.any(self.foot_contact_valid):
            return self.wp_plate_torque

        self.wp_plate_accum.zero_()
        wp.launch(
            accumulate_plate_bending_kernel,
            dim=self.num_springs,
            inputs=[
                self.wp_spring_xy,
                self.wp_spring_top,
                self.wp_foot_sole_z,
                self.wp_foot_contact_valid,
                self.wp_spring_slack,
                self.num_springs,
                float(body_z_m),
                float(self.start_z),
                self.plate_rear_cut_y,
                self.plate_fore_cut_y,
                self.wp_plate_accum,
            ],
            device=self.device,
        )
        wp.launch(
            finalize_plate_bending_kernel,
            dim=1,
            inputs=[self.wp_plate_accum, float(pitch_rate), self.wp_plate_params, self.wp_plate_torque],
            device=self.device,
        )
        return self.wp_plate_torque

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Update kinematic trajectory or apply gravity in dynamic mode
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

            # Get latest positions and velocities from state_0
            body_q = self.state_0.body_q.numpy()
            body_qd = self.state_0.body_qd.numpy()
            foot_z = float(body_q[self.foot_body_id, 2])
            midsole_z = float(body_q[self.midsole_body_id, 2])
            foot_vz = float(body_qd[self.foot_body_id, 2])
            midsole_vz = float(body_qd[self.midsole_body_id, 2])
            pitch_rate = float(body_qd[self.foot_body_id, 4])

            # Apply non-contact forces (gravity, external forces, plate bending torque)
            ext_force = 0.0
            if not self.kinematic:
                omega = 2.0 * np.pi * 1.0
                ext_force = 1000.0 * max(np.sin(omega * self.sim_time * 0.5), 0.0)

            plate_torque = self._update_plate_bending_torque(foot_z, pitch_rate)
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
                    plate_torque,
                    int(self.kinematic),
                ],
                device=self.device,
            )

            # Run GPU broadphase and narrowphase collision checks
            self.collision_pipeline.collide(self.state_0, self.contacts)

            # Get the extracted marching cubes contact surface
            hydro = self.collision_pipeline.narrow_phase.hydroelastic_sdf
            contact_surface = hydro.get_contact_surface()

            face_count = 0
            if contact_surface is not None:
                face_count = int(contact_surface.face_contact_count.numpy()[0])

            if self.debug:
                print(
                    f"[DEBUG_STEP] t={self.sim_time:.4f} foot_z={foot_z:.6f} midsole_z={midsole_z:.6f} "
                    f"contacts={self.contacts.rigid_contact_count.numpy()[0]} face_count={face_count}"
                )

            # Evaluate custom Ogden kernel on the 3D contact patch
            self.wp_contact_wrench.zero_()
            self.wp_foot_top_displacement.zero_()
            self.wp_foot_top_pressure_kpa.zero_()
            self.wp_ground_bottom_displacement.zero_()
            self.wp_ground_bottom_pressure_kpa.zero_()
            if face_count > 0:
                wp.launch(
                    evaluate_hydroelastic_ogden_kernel,
                    dim=face_count,
                    inputs=[
                        contact_surface.contact_surface_point,
                        contact_surface.contact_surface_depth,
                        contact_surface.contact_surface_shape_pair,
                        contact_surface.face_contact_count,
                        self.foot_shape_id,
                        self.midsole_shape_id,
                        self.ground_shape_id,
                        self.foot_body_id,
                        self.midsole_body_id,
                        self.model.body_com,
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        self.wp_spring_xy,
                        self.wp_spring_slack,
                        self.num_springs,
                        self.wp_params,
                        self.state_0.body_f,
                        self.wp_contact_wrench,
                        self.wp_foot_top_displacement,
                        self.wp_foot_top_pressure_kpa,
                        self.wp_ground_bottom_displacement,
                        self.wp_ground_bottom_pressure_kpa,
                    ],
                    device=self.device,
                )

            wp.launch(
                update_shoe_stats_kernel,
                dim=1,
                inputs=[self.wp_contact_wrench, self.wp_plate_torque, self.wp_step_stats],
                device=self.device,
            )

            # Step solver
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

            # Get latest positions and velocities from state_0 after stepping
            body_q = self.state_0.body_q.numpy()
            body_qd = self.state_0.body_qd.numpy()
            foot_z = float(body_q[self.foot_body_id, 2])
            midsole_z = float(body_q[self.midsole_body_id, 2])

            self._pressure_maps_from_warp(foot_z, midsole_z, update_peaks=True)

            if self.save_plots:
                self._sync_step_stats_from_warp()
                self.history_z.append(foot_z)
                self.history_foot_z.append(foot_z)
                self.history_midsole_z.append(midsole_z)
                self.history_force.append(self.last_force_n)
                self.history_foot_disp.append(self.wp_foot_top_displacement.numpy().copy())
                self.history_foot_pres.append(self.wp_foot_top_pressure_kpa.numpy().copy())
                self.history_ground_disp.append(self.wp_ground_bottom_displacement.numpy().copy())
                self.history_ground_pres.append(self.wp_ground_bottom_pressure_kpa.numpy().copy())

            # Advance simulation time
            self.sim_time += self.sim_dt

        self._sync_step_stats_from_warp()
        self._sync_peak_maps_from_warp()

    def step(self):
        self.simulate()

    def render(self):
        body_q = self.state_0.body_q.numpy()
        foot_pos = body_q[self.foot_body_id, :3]

        # Retrieve the contact surface
        hydro = self.collision_pipeline.narrow_phase.hydroelastic_sdf
        contact_surface = hydro.get_contact_surface()

        face_count = 0
        if contact_surface is not None:
            face_count = int(contact_surface.face_contact_count.numpy()[0])

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        # Log contact patch if active
        if face_count > 0:
            points_np = contact_surface.contact_surface_point.numpy()[: 3 * face_count]
            pairs_np = contact_surface.contact_surface_shape_pair.numpy()[:face_count]
            
            # Mask for foot-midsole contact faces:
            is_foot_midsole = (
                ((pairs_np[:, 0] == self.foot_shape_id) & (pairs_np[:, 1] == self.midsole_shape_id)) |
                ((pairs_np[:, 1] == self.foot_shape_id) & (pairs_np[:, 0] == self.midsole_shape_id))
            )
            # Mask for midsole-ground contact faces:
            is_midsole_ground = (
                ((pairs_np[:, 0] == self.midsole_shape_id) & (pairs_np[:, 1] == self.ground_shape_id)) |
                ((pairs_np[:, 1] == self.midsole_shape_id) & (pairs_np[:, 0] == self.ground_shape_id))
            )
            
            # Extract points for foot-midsole:
            foot_midsole_points_mask = np.repeat(is_foot_midsole, 3)
            foot_midsole_points = points_np[foot_midsole_points_mask]
            
            if len(foot_midsole_points) > 0:
                fm_face_count = len(foot_midsole_points) // 3
                fm_points_wp = wp.array(foot_midsole_points, dtype=wp.vec3, device=self.device)
                fm_indices_wp = wp.array(np.arange(3 * fm_face_count, dtype=np.int32), dtype=np.int32, device=self.device)
                
                self.viewer.log_mesh(
                    "/foot_shoe/contact_foot_midsole",
                    points=fm_points_wp,
                    indices=fm_indices_wp,
                    color=(0.1, 0.8, 0.4), # Green
                    backface_culling=False,
                )
            else:
                self.viewer.log_mesh(
                    "/foot_shoe/contact_foot_midsole",
                    points=wp.zeros(0, dtype=wp.vec3, device=self.device),
                    indices=wp.zeros(0, dtype=wp.int32, device=self.device),
                    hidden=True,
                )
                
            # Extract points for midsole-ground:
            midsole_ground_points_mask = np.repeat(is_midsole_ground, 3)
            midsole_ground_points = points_np[midsole_ground_points_mask]
            
            if len(midsole_ground_points) > 0:
                mg_face_count = len(midsole_ground_points) // 3
                mg_points_wp = wp.array(midsole_ground_points, dtype=wp.vec3, device=self.device)
                mg_indices_wp = wp.array(np.arange(3 * mg_face_count, dtype=np.int32), dtype=np.int32, device=self.device)
                
                self.viewer.log_mesh(
                    "/foot_shoe/contact_midsole_ground",
                    points=mg_points_wp,
                    indices=mg_indices_wp,
                    color=(0.9, 0.4, 0.1), # Orange/Red
                    backface_culling=False,
                )
            else:
                self.viewer.log_mesh(
                    "/foot_shoe/contact_midsole_ground",
                    points=wp.zeros(0, dtype=wp.vec3, device=self.device),
                    indices=wp.zeros(0, dtype=wp.int32, device=self.device),
                    hidden=True,
                )
        else:
            self.viewer.log_mesh(
                "/foot_shoe/contact_foot_midsole",
                points=wp.zeros(0, dtype=wp.vec3, device=self.device),
                indices=wp.zeros(0, dtype=wp.int32, device=self.device),
                hidden=True,
            )
            self.viewer.log_mesh(
                "/foot_shoe/contact_midsole_ground",
                points=wp.zeros(0, dtype=wp.vec3, device=self.device),
                indices=wp.zeros(0, dtype=wp.int32, device=self.device),
                hidden=True,
            )

        self.viewer.log_array(
            "/foot_shoe/stats",
            np.asarray(
                [
                    self.sim_time,
                    foot_pos[2],
                    float(self.last_force_n),
                    float(self.peak_force_n),
                    float(np.max(self.wp_ground_bottom_displacement.numpy()) if len(self.wp_ground_bottom_displacement.numpy()) > 0 else 0.0),
                    float(face_count),
                    float(self.last_plate_torque_nm),
                ],
                dtype=np.float32,
            ),
        )
        self.viewer.end_frame()

    def test_final(self):
        print(f"Simulation completed. Peak Vertical Force reached: {self.peak_force_n:.2f} N")
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
        axs0_twin.tick_params(axis="y")
        axs0_twin.legend(loc="upper right")
        axs0_twin.set_title("Force & Displacement vs Time")

        axs[1].plot(disp_foot_mm, force_n, color="purple", linewidth=2.5, label="Foot")
        axs[1].plot(disp_midsole_mm, force_n, color="green", linewidth=2.0, linestyle="--", label="Midsole")
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

        fig2, axs2 = plt.subplots(2, 2, figsize=(12, 10))

        sc1 = axs2[0, 0].scatter(
            self.spring_grid.grid_uv_m[:, 0] * 1000.0,
            self.spring_grid.grid_uv_m[:, 1] * 1000.0,
            c=self.peak_foot_top_displacement_m * 1000.0,
            s=18,
            cmap="inferno",
        )
        axs2[0, 0].set_title("Peak Top Displacement from Foot Mesh")
        axs2[0, 0].set_aspect("equal", adjustable="box")
        axs2[0, 0].set_xlabel("Width [mm]")
        axs2[0, 0].set_ylabel("Length [mm]")
        fig2.colorbar(sc1, ax=axs2[0, 0], label="Displacement [mm]")

        sc2 = axs2[0, 1].scatter(
            self.spring_grid.grid_uv_m[:, 0] * 1000.0,
            self.spring_grid.grid_uv_m[:, 1] * 1000.0,
            c=self.peak_foot_top_pressure_kpa,
            s=18,
            cmap="jet",
        )
        axs2[0, 1].set_title("Peak Top Pressure from Foot Mesh")
        axs2[0, 1].set_aspect("equal", adjustable="box")
        axs2[0, 1].set_xlabel("Width [mm]")
        axs2[0, 1].set_ylabel("Length [mm]")
        fig2.colorbar(sc2, ax=axs2[0, 1], label="Pressure [kPa]")

        sc3 = axs2[1, 0].scatter(
            self.spring_grid.grid_uv_m[:, 0] * 1000.0,
            self.spring_grid.grid_uv_m[:, 1] * 1000.0,
            c=self.peak_ground_bottom_displacement_m * 1000.0,
            s=18,
            cmap="inferno",
        )
        axs2[1, 0].set_title("Peak Bottom Displacement from Ground")
        axs2[1, 0].set_aspect("equal", adjustable="box")
        axs2[1, 0].set_xlabel("Width [mm]")
        axs2[1, 0].set_ylabel("Length [mm]")
        fig2.colorbar(sc3, ax=axs2[1, 0], label="Displacement [mm]")

        sc4 = axs2[1, 1].scatter(
            self.spring_grid.grid_uv_m[:, 0] * 1000.0,
            self.spring_grid.grid_uv_m[:, 1] * 1000.0,
            c=self.peak_ground_bottom_pressure_kpa,
            s=18,
            cmap="jet",
        )
        axs2[1, 1].set_title("Peak Bottom Pressure from Ground")
        axs2[1, 1].set_aspect("equal", adjustable="box")
        axs2[1, 1].set_xlabel("Width [mm]")
        axs2[1, 1].set_ylabel("Length [mm]")
        fig2.colorbar(sc4, ax=axs2[1, 1], label="Pressure [kPa]")

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

            foot_disp_init = self.history_foot_disp[indices[0]]
            foot_pressure_init = self.history_foot_pres[indices[0]]
            ground_disp_init = self.history_ground_disp[indices[0]]
            ground_pressure_init = self.history_ground_pres[indices[0]]

            foot_disp_vmax = (
                np.max(self.peak_foot_top_displacement_m * 1000.0)
                if np.max(self.peak_foot_top_displacement_m) > 0.0
                else 1.0
            )
            ground_disp_vmax = (
                np.max(self.peak_ground_bottom_displacement_m * 1000.0)
                if np.max(self.peak_ground_bottom_displacement_m) > 0.0
                else 1.0
            )

            sc_foot_disp = axs_anim[0, 0].scatter(
                self.spring_grid.grid_uv_m[:, 0] * 1000.0,
                self.spring_grid.grid_uv_m[:, 1] * 1000.0,
                c=foot_disp_init * 1000.0,
                s=18,
                cmap="inferno",
                vmin=0.0,
                vmax=foot_disp_vmax,
            )
            axs_anim[0, 0].set_title("Top Displacement from Foot [mm]")
            axs_anim[0, 0].set_aspect("equal", adjustable="box")
            axs_anim[0, 0].set_xlabel("Width [mm]")
            axs_anim[0, 0].set_ylabel("Length [mm]")
            fig_anim.colorbar(sc_foot_disp, ax=axs_anim[0, 0], label="Displacement [mm]")

            sc_foot_pres = axs_anim[0, 1].scatter(
                self.spring_grid.grid_uv_m[:, 0] * 1000.0,
                self.spring_grid.grid_uv_m[:, 1] * 1000.0,
                c=foot_pressure_init,
                s=18,
                cmap="jet",
                vmin=0.0,
                vmax=self.max_display_pressure_kpa,
            )
            axs_anim[0, 1].set_title("Top Pressure from Foot [kPa]")
            axs_anim[0, 1].set_aspect("equal", adjustable="box")
            axs_anim[0, 1].set_xlabel("Width [mm]")
            axs_anim[0, 1].set_ylabel("Length [mm]")
            fig_anim.colorbar(sc_foot_pres, ax=axs_anim[0, 1], label="Pressure [kPa]")

            sc_ground_disp = axs_anim[1, 0].scatter(
                self.spring_grid.grid_uv_m[:, 0] * 1000.0,
                self.spring_grid.grid_uv_m[:, 1] * 1000.0,
                c=ground_disp_init * 1000.0,
                s=18,
                cmap="inferno",
                vmin=0.0,
                vmax=ground_disp_vmax,
            )
            axs_anim[1, 0].set_title("Bottom Displacement from Ground [mm]")
            axs_anim[1, 0].set_aspect("equal", adjustable="box")
            axs_anim[1, 0].set_xlabel("Width [mm]")
            axs_anim[1, 0].set_ylabel("Length [mm]")
            fig_anim.colorbar(sc_ground_disp, ax=axs_anim[1, 0], label="Displacement [mm]")

            sc_ground_pres = axs_anim[1, 1].scatter(
                self.spring_grid.grid_uv_m[:, 0] * 1000.0,
                self.spring_grid.grid_uv_m[:, 1] * 1000.0,
                c=ground_pressure_init,
                s=18,
                cmap="jet",
                vmin=0.0,
                vmax=self.max_display_pressure_kpa,
            )
            axs_anim[1, 1].set_title("Bottom Pressure from Ground [kPa]")
            axs_anim[1, 1].set_aspect("equal", adjustable="box")
            axs_anim[1, 1].set_xlabel("Width [mm]")
            axs_anim[1, 1].set_ylabel("Length [mm]")
            fig_anim.colorbar(sc_ground_pres, ax=axs_anim[1, 1], label="Pressure [kPa]")

            title = fig_anim.suptitle("")

            def update_anim(frame_idx):
                idx = indices[frame_idx]
                t_val = time[idx]
                f_val = force_n[idx]

                foot_disp = self.history_foot_disp[idx]
                foot_pressure = self.history_foot_pres[idx]
                ground_disp = self.history_ground_disp[idx]
                ground_pressure = self.history_ground_pres[idx]

                sc_foot_disp.set_array(foot_disp * 1000.0)
                sc_foot_pres.set_array(foot_pressure)
                sc_ground_disp.set_array(ground_disp * 1000.0)
                sc_ground_pres.set_array(ground_pressure)
                title.set_text(f"Foot-Shoe Impact | t={t_val:.3f} s | Force={f_val:.1f} N")
                return sc_foot_disp, sc_foot_pres, sc_ground_disp, sc_ground_pres, title

            anim_path = "foot_shoe_contact_heatmap.gif"
            anim = FuncAnimation(fig_anim, update_anim, frames=len(indices), interval=100, blit=False)
            anim.save(anim_path, writer=PillowWriter(fps=10), dpi=100)
            print(f"Heatmap video saved to {anim_path}")
            plt.close(fig_anim)
        except Exception as e:
            print(f"Failed to generate animation: {e}")

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
        parser.add_argument(
            "--use-hydro-surface-wrench",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Use Newton's extracted hydroelastic contact surface for the custom material wrench. "
                "The default Warp foundation grid is better conditioned for this shoe stack-height model."
            ),
        )
        parser.add_argument(
            "--enable-plate",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Add a lightweight carbon-plate longitudinal bending torque approximation.",
        )
        parser.add_argument(
            "--plate-thickness-mm",
            type=float,
            choices=(0.75, 1.0, 1.5, 2.25, 3.0),
            default=1.5,
            help="Carbon plate thickness level [mm] from the Kim et al. design sweep.",
        )
        parser.add_argument("--plate-width-mm", type=float, default=70.0, help="Effective carbon plate width [mm].")
        parser.add_argument("--plate-length-mm", type=float, default=150.0, help="Effective bending span [mm].")
        parser.add_argument("--plate-young-gpa", type=float, default=33.0, help="Carbon plate Young's modulus [GPa].")
        parser.add_argument("--plate-poisson", type=float, default=0.4, help="Carbon plate Poisson's ratio.")
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
        parser.add_argument("--debug", action="store_true", help="Print collision and SDF diagnostics every substep.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    if not args.test:
        example.test_final()
