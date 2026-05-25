# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import warp as wp
import warp.fem as fem
import warp.sparse as wps
from warp.types import type_size

import newton

from .implicit_mpm_model import MaterialParameters
from .rheology_solver_kernels import YieldParamVec, project_stress

wp.set_module_options({"enable_backward": False})

USE_HENCKY_STRAIN_MEASURE = wp.constant(True)
"""Use Hencky instead of co-rotated elastic model (replaces (S - I) with log S in Hooke's law)"""

MIN_PRINCIPAL_STRAIN = wp.constant(1.0e-6 if USE_HENCKY_STRAIN_MEASURE else 1.0e-2)
"""Minimum elastic strain for the elastic model (singular value of the elastic deformation gradient)"""

MAX_PRINCIPAL_STRAIN = wp.constant(1.0e6 if USE_HENCKY_STRAIN_MEASURE else 1.0e2)
"""Maximum elastic strain for the elastic model (singular value of the elastic deformation gradient)"""

MIN_HARDENING_JP = wp.constant(0.1)
"""Minimum plastic compression ratio for the hardening law (determinant of the plastic deformation gradient)"""

MIN_JP_DELTA = wp.constant(0.01)
"""Minimum delta for the plastic deformation gradient"""

MAX_JP_DELTA = wp.constant(10.0)
"""Maximum delta for the plastic deformation gradient"""

INFINITY = wp.constant(1.0e12)
"""Value above which quantities are considered infinite"""

EPSILON = wp.constant(1.0 / INFINITY)
"""Value below which quantities are considered zero"""

_QR_TOLERANCE = wp.constant(1.0e-12)
"""Convergence tolerance for the QR eigenvalue decomposition"""

_NAN_THRESHOLD = wp.constant(1.0e16)
"""Threshold above which eigenvalue results are considered NaN/divergent"""

_EIGENVALUE_FLOOR = wp.constant(1.0e-6)
"""Eigenvalues at or below this value are treated as zero (mode is dropped)"""

vec6 = wp.types.vector(length=6, dtype=wp.float32)
mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat63 = wp.types.matrix(shape=(6, 3), dtype=wp.float32)
mat36 = wp.types.matrix(shape=(3, 6), dtype=wp.float32)
mat13 = wp.types.matrix(shape=(1, 3), dtype=wp.float32)
mat31 = wp.types.matrix(shape=(3, 1), dtype=wp.float32)
mat11 = wp.types.matrix(shape=(1, 1), dtype=wp.float32)

YIELD_PARAM_LENGTH = type_size(YieldParamVec)


@fem.integrand
def integrate_fraction(s: fem.Sample, phi: fem.Field, domain: fem.Domain, inv_cell_volume: float):
    return phi(s) * inv_cell_volume


@fem.integrand
def integrate_collider_fraction(
    s: fem.Sample,
    domain: fem.Domain,
    phi: fem.Field,
    sdf: fem.Field,
    inv_cell_volume: float,
):
    return phi(s) * wp.where(sdf(s) <= 0.0, inv_cell_volume, 0.0)


@fem.integrand
def integrate_collider_fraction_apic(
    s: fem.Sample,
    domain: fem.Domain,
    phi: fem.Field,
    sdf: fem.Field,
    sdf_gradient: fem.Field,
    inv_cell_volume: float,
):
    # APIC collider fraction prediction
    node_count = fem.node_count(sdf, s)
    pos = domain(s)
    min_sdf = float(INFINITY)
    for k in range(node_count):
        s_node = fem.at_node(sdf, s, k)
        sdf_value = sdf(s_node, k)
        sdf_gradient_value = sdf_gradient(s_node, k)

        node_offset = pos - domain(s_node)
        min_sdf = wp.min(min_sdf, sdf_value + wp.dot(sdf_gradient_value, node_offset))

    return phi(s) * wp.where(min_sdf <= 0.0, inv_cell_volume, 0.0)


@fem.integrand
def integrate_mass(
    s: fem.Sample,
    phi: fem.Field,
    domain: fem.Domain,
    inv_cell_volume: float,
    particle_density: wp.array[float],
):
    # Particles with density == 0 are kinematic boundary conditions: they contribute
    # infinite mass so the grid velocity at their location is prescribed.
    # This is distinct from ~ACTIVE particles (checked in advect/strain updates),
    # which are completely ignored during transfers.
    density = wp.where(particle_density[s.qp_index] > 0.0, particle_density[s.qp_index], INFINITY)
    return phi(s) * density * inv_cell_volume


@fem.integrand
def integrate_velocity(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    velocities: wp.array[wp.vec3],
    dt: float,
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    inv_cell_volume: float,
    particle_density: wp.array[float],
):
    vel_adv = velocities[s.qp_index]
    world_idx = particle_world[s.qp_index]
    world_g = gravity[wp.max(world_idx, 0)]

    rho = particle_density[s.qp_index]
    vel_adv = wp.where(
        rho > 0.0,
        rho * (vel_adv + dt * world_g),
        INFINITY * vel_adv,
    )
    return wp.dot(u(s), vel_adv) * inv_cell_volume


@fem.integrand
def integrate_velocity_apic(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    velocity_gradients: wp.array[wp.mat33],
    inv_cell_volume: float,
    particle_density: wp.array[float],
):
    # APIC velocity prediction
    node_offset = domain(fem.at_node(u, s)) - domain(s)
    vel_apic = velocity_gradients[s.qp_index] * node_offset

    rho = particle_density[s.qp_index]
    vel_adv = wp.where(rho > 0.0, rho, INFINITY) * vel_apic
    return wp.dot(u(s), vel_adv) * inv_cell_volume


@wp.kernel
def free_velocity(
    velocity_int: wp.array[wp.vec3],
    node_particle_mass: wp.array[float],
    drag: float,
    inv_mass_matrix: wp.array[float],
    velocity_avg: wp.array[wp.vec3],
):
    i = wp.tid()

    pmass = node_particle_mass[i]
    inv_particle_mass = 1.0 / (pmass + drag)

    vel = velocity_int[i] * inv_particle_mass
    inv_mass_matrix[i] = inv_particle_mass

    velocity_avg[i] = vel


@wp.func
def hardening_law(Jp: float, hardening: float):
    if hardening == 0.0:
        return 1.0

    eps = wp.log(wp.clamp(Jp, MIN_HARDENING_JP, 1.0))
    h = wp.sinh(-hardening * eps)

    return h


@wp.func
def get_elastic_parameters(
    i: int,
    material_parameters: MaterialParameters,
):
    # Hardening only affects yield parameters, not elastic stiffness.
    # This separates the elastic response from the plastic history.
    E = material_parameters.young_modulus[i]
    nu = material_parameters.poisson_ratio[i]
    d = material_parameters.damping[i]

    return wp.vec3(E, nu, d)


@wp.func
def extract_elastic_parameters(
    params_vec: wp.vec3,
):
    compliance = 1.0 / params_vec[0]
    poisson = params_vec[1]
    damping = params_vec[2]
    return compliance, poisson, damping


@wp.func
def get_yield_parameters(i: int, material_parameters: MaterialParameters, particle_Jp: float, dt: float):
    h = hardening_law(particle_Jp, material_parameters.hardening[i])

    mu = material_parameters.friction[i]

    return YieldParamVec.from_values(
        mu,
        material_parameters.yield_pressure[i] * h,
        material_parameters.tensile_yield_ratio[i],
        material_parameters.yield_stress[i] * h,
        material_parameters.dilatancy[i],
        material_parameters.viscosity[i] / dt,
    )


@fem.integrand
def integrate_elastic_parameters(
    s: fem.Sample,
    u: fem.Field,
    inv_cell_volume: float,
    material_parameters: MaterialParameters,
):
    i = s.qp_index
    params_vec = get_elastic_parameters(i, material_parameters)
    return wp.dot(u(s), params_vec) * inv_cell_volume


@fem.integrand
def integrate_yield_parameters(
    s: fem.Sample,
    u: fem.Field,
    inv_cell_volume: float,
    material_parameters: MaterialParameters,
    particle_Jp: wp.array[float],
    dt: float,
):
    i = s.qp_index
    params_vec = get_yield_parameters(i, material_parameters, particle_Jp[i], dt)
    return wp.dot(u(s), params_vec) * inv_cell_volume


@fem.integrand
def integrate_particle_stress(
    s: fem.Sample,
    tau: fem.Field,
    inv_cell_volume: float,
    particle_stress: wp.array[wp.mat33],
):
    i = s.qp_index

    return wp.ddot(tau(s), particle_stress[i]) * inv_cell_volume


@wp.kernel
def average_yield_parameters(
    yield_parameters_int: wp.array[YieldParamVec],
    particle_volume: wp.array[float],
    yield_parameters_avg: wp.array[YieldParamVec],
):
    i = wp.tid()
    pvol = particle_volume[i]
    yield_parameters_avg[i] = wp.max(YieldParamVec(0.0), yield_parameters_int[i] / wp.max(pvol, EPSILON))


@wp.kernel
def average_elastic_parameters(
    elastic_parameters_int: wp.array[wp.vec3],
    particle_volume: wp.array[float],
    elastic_parameters_avg: wp.array[wp.vec3],
):
    i = wp.tid()
    pvol = particle_volume[i]
    elastic_parameters_avg[i] = elastic_parameters_int[i] / wp.max(pvol, EPSILON)


@fem.integrand
def advect_particles(
    s: fem.Sample,
    domain: fem.Domain,
    grid_vel: fem.Field,
    dt: float,
    max_vel: float,
    particle_flags: wp.array[wp.int32],
    particle_volume: wp.array[float],
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    vel_grad: wp.array[wp.mat33],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return

    p_vel = grid_vel(s)
    vel_n_sq = wp.length_sq(p_vel)

    p_vel_cfl = wp.where(vel_n_sq > max_vel * max_vel, p_vel * max_vel / wp.sqrt(vel_n_sq), p_vel)

    p_vel_grad = fem.grad(grid_vel, s)

    delta_pos = dt * p_vel_cfl

    gimp_weight = s.qp_weight * fem.measure(domain, s) / particle_volume[s.qp_index]
    wp.atomic_add(pos, s.qp_index, gimp_weight * delta_pos)
    wp.atomic_add(vel, s.qp_index, gimp_weight * p_vel_cfl)
    wp.atomic_add(vel_grad, s.qp_index, gimp_weight * p_vel_grad)


@fem.integrand
def update_particle_strains(
    s: fem.Sample,
    domain: fem.Domain,
    grid_vel: fem.Field,
    plastic_strain_delta: fem.Field,
    elastic_strain_delta: fem.Field,
    stress: fem.Field,
    dt: float,
    particle_flags: wp.array[wp.int32],
    particle_density: wp.array[float],
    particle_volume: wp.array[float],
    material_parameters: MaterialParameters,
    elastic_strain_prev: wp.array[wp.mat33],
    particle_Jp_prev: wp.array[float],
    elastic_strain: wp.array[wp.mat33],
    particle_Jp: wp.array[float],
    particle_stress: wp.array[wp.mat33],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        elastic_strain[s.qp_index] = elastic_strain_prev[s.qp_index]
        particle_Jp[s.qp_index] = particle_Jp_prev[s.qp_index]
        return
    if particle_density[s.qp_index] == 0.0:
        elastic_strain[s.qp_index] = elastic_strain_prev[s.qp_index]
        particle_Jp[s.qp_index] = particle_Jp_prev[s.qp_index]
        return

    # plastic strain
    p_strain_delta = plastic_strain_delta(s)
    p_rate = wp.trace(p_strain_delta)

    delta_Jp = wp.exp(
        p_rate
        * wp.where(
            p_rate < 0.0, material_parameters.hardening_rate[s.qp_index], material_parameters.softening_rate[s.qp_index]
        )
    )
    particle_Jp_new = particle_Jp_prev[s.qp_index] * wp.clamp(delta_Jp, MIN_JP_DELTA, MAX_JP_DELTA)

    elastic_parameters_vec = get_elastic_parameters(s.qp_index, material_parameters)
    compliance, _poisson, _damping = extract_elastic_parameters(elastic_parameters_vec)

    yield_parameters_vec = get_yield_parameters(s.qp_index, material_parameters, particle_Jp_new, dt)
    stress_0 = fem.SymmetricTensorMapper.value_to_dof_3d(stress(s))
    particle_stress_new = fem.SymmetricTensorMapper.dof_to_value_3d(project_stress(stress_0, yield_parameters_vec))

    # elastic strain
    prev_strain = elastic_strain_prev[s.qp_index]
    vel_grad = fem.grad(grid_vel, s)
    skew = 0.5 * dt * (vel_grad - wp.transpose(vel_grad))
    strain_delta = elastic_strain_delta(s) + skew

    # The skew-symmetric part of the velocity gradient is used as a linearized
    # approximation of the finite rotation increment (matches standard deformation gradient update).
    elastic_strain_new = prev_strain + strain_delta @ prev_strain
    elastic_strain_new = project_particle_strain(elastic_strain_new, prev_strain, compliance)

    gimp_weight = s.qp_weight * fem.measure(domain, s) / particle_volume[s.qp_index]
    wp.atomic_add(particle_Jp, s.qp_index, gimp_weight * particle_Jp_new)
    wp.atomic_add(particle_stress, s.qp_index, gimp_weight * particle_stress_new)
    wp.atomic_add(elastic_strain, s.qp_index, gimp_weight * elastic_strain_new)


@wp.func
def project_particle_strain(
    F: wp.mat33,
    F_prev: wp.mat33,
    compliance: float,
):
    if compliance <= EPSILON:
        return wp.identity(n=3, dtype=float)

    _U, xi, _V = wp.svd3(F)

    if wp.min(xi) < MIN_PRINCIPAL_STRAIN or wp.max(xi) > MAX_PRINCIPAL_STRAIN:
        return F_prev  # non-recoverable, discard update

    return F


@wp.kernel
def update_particle_frames(
    dt: float,
    min_stretch: float,
    max_stretch: float,
    vel_grad: wp.array[wp.mat33],
    transform_prev: wp.array[wp.mat33],
    transform: wp.array[wp.mat33],
):
    i = wp.tid()

    p_vel_grad = vel_grad[i]

    # transform, for grain-level rendering
    F_prev = transform_prev[i]
    # dX1/dx = dX1/dX0 dX0/dx
    F = F_prev + dt * p_vel_grad @ F_prev

    # clamp eigenvalues of F
    if min_stretch >= 0.0 and max_stretch >= 0.0:
        U = wp.mat33()
        S = wp.vec3()
        V = wp.mat33()
        wp.svd3(F, U, S, V)
        S = wp.max(wp.min(S, wp.vec3(max_stretch)), wp.vec3(min_stretch))
        F = U @ wp.diag(S) @ wp.transpose(V)

    transform[i] = F


@fem.integrand
def strain_delta_form(
    s: fem.Sample,
    u: fem.Field,
    tau: fem.Field,
    dt: float,
    domain: fem.Domain,
    inv_cell_volume: float,
):
    # The full strain matrix can be recovered from this divergence
    # see _symmetric_part_op in rheology_solver_kernels.py
    return fem.div(u, s) * tau(s) * (dt * inv_cell_volume)


@wp.kernel
def compute_unilateral_strain_offset(
    max_fraction: float,
    particle_volume: wp.array[float],
    collider_volume: wp.array[float],
    node_volume: wp.array[float],
    unilateral_strain_offset: wp.array[float],
):
    i = wp.tid()

    spherical_part = max_fraction * (node_volume[i] - collider_volume[i]) - particle_volume[i]
    spherical_part = wp.max(spherical_part, 0.0)
    unilateral_strain_offset[i] = spherical_part


@wp.func
def stress_strain_relationship(sig: wp.mat33, compliance: float, poisson: float):
    return (sig * (1.0 + poisson) - poisson * (wp.trace(sig) * wp.identity(n=3, dtype=float))) * compliance


@fem.integrand
def strain_rhs(
    s: fem.Sample,
    tau: fem.Field,
    elastic_parameters: fem.Field,
    elastic_strains: wp.array[wp.mat33],
    inv_cell_volume: float,
    dt: float,
):
    _compliance, _poisson, damping = extract_elastic_parameters(elastic_parameters(s))
    alpha = 1.0 / (1.0 + damping / dt)

    F_prev = elastic_strains[s.qp_index]
    U_prev, xi_prev, _V_prev = wp.svd3(F_prev)

    if wp.static(USE_HENCKY_STRAIN_MEASURE):
        RlogSRt_prev = (
            U_prev @ wp.diag(wp.vec3(wp.log(xi_prev[0]), wp.log(xi_prev[1]), wp.log(xi_prev[2]))) @ wp.transpose(U_prev)
        )
        strain = alpha * wp.ddot(tau(s), RlogSRt_prev)
    else:
        RSinvRt_prev = U_prev @ wp.diag(1.0 / xi_prev) @ wp.transpose(U_prev)
        Id = wp.identity(n=3, dtype=float)
        strain = -alpha * wp.ddot(tau(s), RSinvRt_prev - Id)

    return strain * inv_cell_volume


@fem.integrand
def compliance_form(
    s: fem.Sample,
    domain: fem.Domain,
    tau: fem.Field,
    sig: fem.Field,
    elastic_parameters: fem.Field,
    elastic_strains: wp.array[wp.mat33],
    inv_cell_volume: float,
    dt: float,
):
    F = elastic_strains[s.qp_index]

    compliance, poisson, damping = extract_elastic_parameters(elastic_parameters(s))
    gamma = compliance / (1.0 + damping / dt)

    U, xi, V = wp.svd3(F)
    Rt = V @ wp.transpose(U)

    if wp.static(USE_HENCKY_STRAIN_MEASURE):
        R = wp.transpose(Rt)
        return wp.ddot(Rt @ tau(s) @ R, stress_strain_relationship(Rt @ sig(s) @ R, gamma, poisson)) * inv_cell_volume
    else:
        FinvT = U @ wp.diag(1.0 / xi) @ wp.transpose(V)
        return (
            wp.ddot(Rt @ tau(s) @ FinvT, stress_strain_relationship(Rt @ sig(s) @ FinvT, gamma, poisson))
            * inv_cell_volume
        )


@fem.integrand
def collision_weight_field(
    s: fem.Sample,
    normal: fem.Field,
    trial: fem.Field,
):
    n = normal(s)
    if wp.length_sq(n) == 0.0:
        # invalid normal, contact is disabled
        return 0.0

    return trial(s)


@fem.integrand
def mass_form(
    s: fem.Sample,
    p: fem.Field,
    q: fem.Field,
    inv_cell_volume: float,
):
    return p(s) * q(s) * inv_cell_volume


@wp.kernel(module="unique")
def compute_eigenvalues(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    ones: wp.array2d[float],
    yield_parameters: wp.array[YieldParamVec],
    eigenvalues: wp.array2d[float],
    eigenvectors: wp.array3d[float],
    rotated_volume: wp.array2d[float],
):
    row = wp.tid()

    diag_index = wps.bsr_block_index(row, row, offsets, columns)

    if diag_index == -1:
        ev = values.dtype(0.0)
        scales = type(ev[0])(0.0)
        rv = type(ev[0])(0.0)

    else:
        diag_block = values[diag_index]
        scales, ev = fem.linalg.symmetric_eigenvalues_qr(diag_block, _QR_TOLERANCE)

        # symmetric_eigenvalues_qr may return nans for small coefficients
        if not (wp.ddot(ev, ev) < _NAN_THRESHOLD and wp.length_sq(scales) < _NAN_THRESHOLD):
            scales = wp.get_diag(diag_block)
            ev = wp.identity(n=scales.length, dtype=float)

        rv = type(scales)(0.0)
        nodes_per_elt = eigenvectors.shape[1]
        for k in range(scales.length):
            s = float(0.0)
            if scales[k] <= _EIGENVALUE_FLOOR:
                scales[k] = 1.0
                rv[k] = 1.0
                ev_s = 0.0
            else:
                ys = float(0.0)

                for j in range(scales.length):
                    node_index = row * nodes_per_elt + j
                    s += ev[k, j] * ones[row, j]
                    ys += ev[k, j] * yield_parameters[node_index][0]

                ev_s = wp.sign(s)
                rv[k] = ev_s * s

                if ys * ev_s < 0.0:
                    ev_s = 0.0

            for j in range(scales.length):
                ev[k, j] *= ev_s

    size = int(scales.length)
    for k in range(size):
        eigenvalues[row, k] = scales[k]
        rotated_volume[row, k] = rv[k]
        for j in range(scales.length):
            eigenvectors[row, k, j] = ev[k, j]


@wp.kernel(module="unique")
def rotate_matrix_rows(
    eigenvectors: wp.array3d[float],
    mat_offsets: wp.array[int],
    mat_columns: wp.array[int],
    mat_values: wp.array[Any],
    mat_values_out: wp.array[Any],
):
    block = wp.tid()

    nodes_per_elt = eigenvectors.shape[1]
    node_count = eigenvectors.shape[0] * nodes_per_elt

    row = wps.bsr_row_index(mat_offsets, node_count, block)
    if row == -1:
        return

    col = mat_columns[block]

    element = row // nodes_per_elt
    elt_node = row - element * nodes_per_elt

    ev = eigenvectors[element]
    val = mat_values.dtype(0.0)
    for k in range(nodes_per_elt):
        row_k = element * nodes_per_elt + k
        block_k = wps.bsr_block_index(row_k, col, mat_offsets, mat_columns)
        if block_k != -1:
            val += ev[elt_node, k] * mat_values[block_k]

    mat_values_out[block] = val


def make_rotate_vectors(nodes_per_element: int):
    @fem.cache.dynamic_kernel(suffix=nodes_per_element, kernel_options={"enable_mathdx_gemm": False})
    def rotate_vectors(
        eigenvectors: wp.array3d[float],
        strain_rhs: wp.array2d[float],
        stress: wp.array2d[float],
        yield_parameters: wp.array2d[float],
        unilateral_strain_offset: wp.array2d[float],
    ):
        elem = wp.tid()
        ev = wp.tile_load(eigenvectors[elem], shape=(nodes_per_element, nodes_per_element))

        strain_rhs_tile = wp.tile_load(strain_rhs, shape=(nodes_per_element, 6), offset=(elem * nodes_per_element, 0))
        rotated_strain_rhs = wp.tile_matmul(ev, strain_rhs_tile)
        wp.tile_store(strain_rhs, rotated_strain_rhs, offset=(elem * nodes_per_element, 0))

        stress_tile = wp.tile_load(stress, shape=(nodes_per_element, 6), offset=(elem * nodes_per_element, 0))
        rotated_stress = wp.tile_matmul(ev, stress_tile)
        wp.tile_store(stress, rotated_stress, offset=(elem * nodes_per_element, 0))

        yield_tile = wp.tile_load(
            yield_parameters, shape=(nodes_per_element, YIELD_PARAM_LENGTH), offset=(elem * nodes_per_element, 0)
        )
        rotated_yield = wp.tile_matmul(ev, yield_tile)
        wp.tile_store(yield_parameters, rotated_yield, offset=(elem * nodes_per_element, 0))

        unilateral_strain_offset_tile = wp.tile_load(
            unilateral_strain_offset, shape=(nodes_per_element, 1), offset=(elem * nodes_per_element, 0)
        )
        rotated_unilateral_strain_offset = wp.tile_matmul(ev, unilateral_strain_offset_tile)
        wp.tile_store(unilateral_strain_offset, rotated_unilateral_strain_offset, offset=(elem * nodes_per_element, 0))

    return rotate_vectors


def make_inverse_rotate_vectors(nodes_per_element: int):
    @fem.cache.dynamic_kernel(suffix=nodes_per_element)
    def inverse_rotate_vectors(
        eigenvectors: wp.array3d[float],
        plastic_strain: wp.array2d[float],
        elastic_strain: wp.array2d[float],
        stress: wp.array2d[float],
    ):
        elem = wp.tid()

        ev_t = wp.tile_transpose(wp.tile_load(eigenvectors[elem], shape=(nodes_per_element, nodes_per_element)))

        stress_tile = wp.tile_load(stress, shape=(nodes_per_element, 6), offset=(elem * nodes_per_element, 0))
        rotated_stress = wp.tile_matmul(ev_t, stress_tile)
        wp.tile_store(stress, rotated_stress, offset=(elem * nodes_per_element, 0))

        plastic_strain_tile = wp.tile_load(
            plastic_strain, shape=(nodes_per_element, 6), offset=(elem * nodes_per_element, 0)
        )
        rotated_plastic_strain = wp.tile_matmul(ev_t, plastic_strain_tile)
        wp.tile_store(plastic_strain, rotated_plastic_strain, offset=(elem * nodes_per_element, 0))

        elastic_strain_tile = wp.tile_load(
            elastic_strain, shape=(nodes_per_element, 6), offset=(elem * nodes_per_element, 0)
        )
        rotated_elastic_strain = wp.tile_matmul(ev_t, elastic_strain_tile)
        wp.tile_store(elastic_strain, rotated_elastic_strain, offset=(elem * nodes_per_element, 0))

    return inverse_rotate_vectors


@wp.kernel(module="unique")
def inverse_scale_vector(
    eigenvalues: wp.array[float],
    vector: wp.array[Any],
):
    node = wp.tid()
    scale = eigenvalues[node]

    zero = vector.dtype(0.0)
    vector[node] = wp.where(scale == 0.0, zero, vector[node] / scale)


wp.overload(inverse_scale_vector, {"vector": wp.array[YieldParamVec]})


@wp.kernel
def inverse_scale_sym_tensor(
    eigenvalues: wp.array[float],
    vector: wp.array[vec6],
):
    node = wp.tid()

    # Symmetric tensor norm is orthonormal to sig:tau/2
    scale = eigenvalues[node] * 2.0

    vector[node] = wp.where(scale == 0.0, vec6(0.0), vector[node] / scale)


@wp.kernel(module="unique")
def rotate_matrix_columns(
    eigenvectors: wp.array3d[float],
    mat_offsets: wp.array[int],
    mat_columns: wp.array[int],
    mat_values: wp.array[Any],
    mat_values_out: wp.array[Any],
):
    block = wp.tid()

    nodes_per_elt = eigenvectors.shape[1]
    node_count = eigenvectors.shape[0] * nodes_per_elt

    row = wps.bsr_row_index(mat_offsets, node_count, block)
    if row == -1:
        return

    col = mat_columns[block]

    nodes_per_elt = eigenvectors.shape[1]
    element = col // nodes_per_elt
    elt_node = col - element * nodes_per_elt

    ev = eigenvectors[element]
    val = mat_values.dtype(0.0)
    for k in range(nodes_per_elt):
        col_k = element * nodes_per_elt + k
        block_k = wps.bsr_block_index(row, col_k, mat_offsets, mat_columns)
        if block_k != -1:
            val += ev[elt_node, k] * mat_values[block_k]

    mat_values_out[block] = val


@wp.kernel
def compute_bounds(
    pos: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    lower_bounds: wp.array[wp.vec3],
    upper_bounds: wp.array[wp.vec3],
):
    block_id, lane = wp.tid()
    i = block_id * wp.block_dim() + lane

    # pad with +- inf for min/max
    # tile_min scalar only, so separate components
    # no tile_atomic_min yet, extract first and use lane 0

    if i >= pos.shape[0]:
        valid = False
    elif ~particle_flags[i] & newton.ParticleFlags.ACTIVE:
        valid = False
    else:
        valid = True

    if valid:
        p = pos[i]
        min_x = p[0]
        min_y = p[1]
        min_z = p[2]
        max_x = p[0]
        max_y = p[1]
        max_z = p[2]
    else:
        min_x = INFINITY
        min_y = INFINITY
        min_z = INFINITY
        max_x = -INFINITY
        max_y = -INFINITY
        max_z = -INFINITY

    tile_min_x = wp.tile_min(wp.tile(min_x))[0]
    tile_max_x = wp.tile_max(wp.tile(max_x))[0]
    tile_min_y = wp.tile_min(wp.tile(min_y))[0]
    tile_max_y = wp.tile_max(wp.tile(max_y))[0]
    tile_min_z = wp.tile_min(wp.tile(min_z))[0]
    tile_max_z = wp.tile_max(wp.tile(max_z))[0]
    tile_min = wp.vec3(tile_min_x, tile_min_y, tile_min_z)
    tile_max = wp.vec3(tile_max_x, tile_max_y, tile_max_z)
    if lane == 0:
        wp.atomic_min(lower_bounds, 0, tile_min)
        wp.atomic_max(upper_bounds, 0, tile_max)


@wp.kernel
def clamp_coordinates(
    coords: wp.array[wp.vec3],
):
    i = wp.tid()
    coords[i] = wp.min(wp.max(coords[i], wp.vec3(0.0)), wp.vec3(1.0))


@wp.kernel
def pad_voxels(particle_q: wp.array[wp.vec3i], padded_q: wp.array4d[wp.vec3i]):
    pid = wp.tid()

    for i in range(3):
        for j in range(3):
            for k in range(3):
                padded_q[pid, i, j, k] = particle_q[pid] + wp.vec3i(i - 1, j - 1, k - 1)


@wp.func
def positive_modn(x: int, n: int):
    return (x % n + n) % n


def allocate_by_voxels(particle_q, voxel_size, padding_voxels: int = 0):
    volume = wp.Volume.allocate_by_voxels(
        voxel_points=particle_q.flatten(),
        voxel_size=voxel_size,
    )

    for _pad_i in range(padding_voxels):
        voxels = wp.empty((volume.get_voxel_count(),), dtype=wp.vec3i)
        volume.get_voxels(voxels)

        padded_voxels = wp.zeros((voxels.shape[0], 3, 3, 3), dtype=wp.vec3i)
        wp.launch(pad_voxels, voxels.shape[0], (voxels, padded_voxels))

        volume = wp.Volume.allocate_by_voxels(
            voxel_points=padded_voxels.flatten(),
            voxel_size=voxel_size,
        )

    return volume


@wp.kernel
def node_color(
    space_node_indices: wp.array[int],
    stencil_size: int,
    voxels: wp.array[wp.vec3i],
    res: wp.vec3i,
    colors: wp.array[int],
    color_indices: wp.array[int],
):
    nid = wp.tid()
    vid = space_node_indices[nid]

    if vid == fem.NULL_NODE_INDEX:
        colors[nid] = _NULL_COLOR
        color_indices[nid] = nid
        return

    if voxels:
        c = voxels[vid]
    else:
        c = fem.Grid3D.get_cell(res, vid)

    colors[nid] = (
        positive_modn(c[0], stencil_size) * stencil_size * stencil_size
        + positive_modn(c[1], stencil_size) * stencil_size
        + positive_modn(c[2], stencil_size)
    )
    color_indices[nid] = nid


def make_cell_color_kernel(geo_partition: fem.GeometryPartition):
    @fem.cache.dynamic_kernel(geo_partition.name)
    def cell_color(
        partition_arg: geo_partition.CellArg,
        stencil_size: int,
        voxels: wp.array[wp.vec3i],
        res: wp.vec3i,
        colors: wp.array[int],
        color_indices: wp.array[int],
    ):
        pid = wp.tid()

        cell = geo_partition.cell_index(partition_arg, pid)
        if cell == -1:
            colors[pid] = _NULL_COLOR
            color_indices[pid] = pid
            return

        if voxels:
            c = voxels[cell]
        else:
            c = fem.Grid3D.get_cell(res, cell)

        colors[pid] = (
            positive_modn(c[0], stencil_size) * stencil_size * stencil_size
            + positive_modn(c[1], stencil_size) * stencil_size
            + positive_modn(c[2], stencil_size)
        )
        color_indices[pid] = pid

    return cell_color


@wp.kernel
def fill_uniform_color_block_indices(
    nodes_per_element: int,
    color_indices: wp.array2d[int],
):
    i = wp.tid()
    elem_idx = color_indices[0, i]
    color_indices[0, i] = elem_idx * nodes_per_element
    color_indices[1, i] = (elem_idx + 1) * nodes_per_element


def make_dynamic_color_block_indices_kernel(geo_partition: fem.GeometryPartition):
    @fem.cache.dynamic_kernel(geo_partition.name)
    def fill_dynamic_color_block_indices(
        partition_arg: geo_partition.CellArg,
        cell_node_offsets: wp.array[int],
        color_indices: wp.array2d[int],
    ):
        i = wp.tid()
        elem_idx = color_indices[0, i]
        cell = geo_partition.cell_index(partition_arg, elem_idx)
        if cell == -1:
            color_indices[0, i] = 0
            color_indices[1, i] = 0
            return
        color_indices[0, i] = cell_node_offsets[cell]
        color_indices[1, i] = cell_node_offsets[cell + 1]

    return fill_dynamic_color_block_indices


_NULL_COLOR = (1 << 31) - 1  # color for null nodes. make sure it is sorted last


@wp.kernel
def compute_color_offsets(
    max_color_count: int,
    unique_count: wp.array[int],
    unique_colors: wp.array[int],
    color_counts: wp.array[int],
    color_offsets: wp.array[int],
):
    current_sum = int(0)
    count = unique_count[0]

    for k in range(count):
        color_offsets[k] = current_sum
        color = unique_colors[k]
        local_count = wp.where(color == _NULL_COLOR, 0, color_counts[k])
        current_sum += local_count

    for k in range(count, max_color_count + 1):
        color_offsets[k] = current_sum


@fem.integrand
def mark_active_cells(
    s: fem.Sample,
    domain: fem.Domain,
    positions: wp.array[wp.vec3],
    particle_flags: wp.array[int],
    active_cells: wp.array[int],
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return

    x = positions[s.qp_index]
    s_grid = fem.lookup(domain, x)

    if s_grid.element_index != fem.NULL_ELEMENT_INDEX:
        active_cells[s_grid.element_index] = 1


@wp.kernel(module="unique")
def scatter_field_dof_values(
    space_node_indices: wp.array[int],
    src: wp.array[Any],
    dest: wp.array[Any],
):
    nid = wp.tid()

    sid = space_node_indices[nid]
    if sid != fem.NULL_NODE_INDEX:
        dest[sid] = src[nid]


wp.overload(scatter_field_dof_values, {"src": wp.array[wp.vec3], "dest": wp.array[wp.vec3]})
wp.overload(scatter_field_dof_values, {"src": wp.array[vec6], "dest": wp.array[vec6]})
