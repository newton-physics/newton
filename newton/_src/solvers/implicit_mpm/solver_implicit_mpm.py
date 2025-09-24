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

"""Implicit MPM solver."""

import math
from dataclasses import dataclass

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.context import assert_conditional_graph_support

import newton

from ...core.types import override
from ..solver import SolverBase
from .rasterized_collisions import (
    Collider,
    allot_collider_mass,
    build_rigidity_matrix,
    project_outside_collider,
    rasterize_collider,
)
from .render_grains import sample_render_grains, update_render_grains
from .solve_rheology import YieldParamVec, solve_rheology

__all__ = ["SolverImplicitMPM"]


MIN_PRINCIPAL_STRAIN = wp.constant(0.01)
"""Minimum elastic strain for the elastic model (singular value of the elastic deformation gradient)"""

MAX_PRINCIPAL_STRAIN = wp.constant(4.0)
"""Maximum elastic strain for the elastic model (singular value of the elastic deformation gradient)"""

MIN_HARDENING_JP = wp.constant(0.01)
"""Minimum hardening for the elastic model (determinant of the plastic deformation gradient)"""

MAX_HARDENING_JP = wp.constant(4.0)
"""Maximum hardening for the elastic model (determinant of the plastic deformation gradient)"""

MIN_JP_DELTA = wp.constant(0.1)
"""Minimum delta for the plastic deformation gradient"""

MAX_JP_DELTA = wp.constant(10.0)
"""Maximum delta for the plastic deformation gradient"""

_INFINITY = wp.constant(1.0e12)
"""Value above which quantities are considered infinite"""

_EPSILON = wp.constant(1.0 / _INFINITY)
"""Value below which quantities are considered zero"""


_DEFAULT_PROJECTION_THRESHOLD = wp.constant(0.5)
"""Default threshold for projection outside of collider, as a fraction of the voxel size"""

_DEFAULT_THICKNESS = 0.5
"""Default thickness for colliders"""
_DEFAULT_FRICTION = 0.0
"""Default friction coefficient for colliders"""
_DEFAULT_ADHESION = 0.0
"""Default adhesion coefficient for colliders (Pa)"""


vec6 = wp.types.vector(length=6, dtype=wp.float32)
mat66 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
mat63 = wp.types.matrix(shape=(6, 3), dtype=wp.float32)
mat36 = wp.types.matrix(shape=(3, 6), dtype=wp.float32)


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
def integrate_mass(
    s: fem.Sample,
    phi: fem.Field,
    domain: fem.Domain,
    inv_cell_volume: float,
    particle_density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
):
    density = wp.where(
        particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE, particle_density[s.qp_index], _INFINITY
    )
    return phi(s) * density * inv_cell_volume


@fem.integrand
def integrate_velocity(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
    gravity: wp.array(dtype=wp.vec3),
    inv_cell_volume: float,
    particle_density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
):
    vel_adv = velocities[s.qp_index]

    vel_adv = wp.where(
        particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE,
        particle_density[s.qp_index] * (vel_adv + dt * gravity[0]),
        _INFINITY * vel_adv,
    )
    return wp.dot(u(s), vel_adv) * inv_cell_volume


@fem.integrand
def integrate_velocity_apic(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    velocity_gradients: wp.array(dtype=wp.mat33),
    inv_cell_volume: float,
    particle_density: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
):
    # APIC velocity prediction
    node_offset = domain(fem.at_node(u, s)) - domain(s)
    vel_apic = velocity_gradients[s.qp_index] * node_offset

    vel_adv = (
        wp.where(particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE, particle_density[s.qp_index], _INFINITY)
        * vel_apic
    )
    return wp.dot(u(s), vel_adv) * inv_cell_volume


@wp.kernel
def free_velocity(
    velocity_int: wp.array(dtype=wp.vec3),
    node_particle_mass: wp.array(dtype=float),
    drag: float,
    inv_mass_matrix: wp.array(dtype=float),
    velocity_avg: wp.array(dtype=wp.vec3),
):
    i = wp.tid()

    pmass = node_particle_mass[i]
    inv_particle_mass = 1.0 / (pmass + drag)

    vel = velocity_int[i] * inv_particle_mass
    inv_mass_matrix[i] = inv_particle_mass

    velocity_avg[i] = vel


@wp.struct
class MaterialParameters:
    young_modulus: wp.array(dtype=float)
    poisson_ratio: wp.array(dtype=float)
    damping: wp.array(dtype=float)
    hardening: wp.array(dtype=float)

    friction: wp.array(dtype=float)
    yield_pressure: wp.array(dtype=float)
    tensile_yield_ratio: wp.array(dtype=float)
    yield_stress: wp.array(dtype=float)


@wp.func
def hardening_law(Jp: float, hardening: float):
    return wp.exp(hardening * (1.0 - wp.clamp(Jp, MIN_HARDENING_JP, MAX_HARDENING_JP)))


@wp.func
def get_elastic_parameters(
    i: int,
    material_parameters: MaterialParameters,
    particle_Jp: wp.array(dtype=float),
):
    E = material_parameters.young_modulus[i] * hardening_law(particle_Jp[i], material_parameters.hardening[i])
    nu = material_parameters.poisson_ratio[i]
    damping = material_parameters.damping[i]

    return wp.vec3(E, nu, damping)


@wp.func
def extract_elastic_parameters(
    params_vec: wp.vec3,
):
    compliance = 1.0 / params_vec[0]
    poisson = params_vec[1]
    damping = params_vec[2]
    return compliance, poisson, damping


@wp.func
def get_yield_parameters(
    i: int,
    material_parameters: MaterialParameters,
    particle_Jp: wp.array(dtype=float),
):
    h = hardening_law(particle_Jp[i], material_parameters.hardening[i])
    mu = material_parameters.friction[i]

    return YieldParamVec.from_values(
        mu,
        material_parameters.yield_pressure[i] * h,
        material_parameters.tensile_yield_ratio[i] / h,  # keep tensile yield stress constant
        material_parameters.yield_stress[i] * h,
    )


@fem.integrand
def integrate_elastic_parameters(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    inv_cell_volume: float,
    material_parameters: MaterialParameters,
    particle_Jp: wp.array(dtype=float),
):
    i = s.qp_index
    params_vec = get_elastic_parameters(i, material_parameters, particle_Jp)
    return wp.dot(u(s), params_vec) * inv_cell_volume


@fem.integrand
def integrate_yield_parameters(
    s: fem.Sample,
    u: fem.Field,
    inv_cell_volume: float,
    material_parameters: MaterialParameters,
    particle_Jp: wp.array(dtype=float),
):
    i = s.qp_index
    params_vec = get_yield_parameters(i, material_parameters, particle_Jp)
    return wp.dot(u(s), params_vec) * inv_cell_volume


@wp.kernel
def average_yield_parameters(
    yield_parameters_int: wp.array(dtype=YieldParamVec),
    particle_volume: wp.array(dtype=float),
    yield_parameters_avg: wp.array(dtype=YieldParamVec),
):
    i = wp.tid()
    pvol = particle_volume[i]
    yield_parameters_avg[i] = wp.max(YieldParamVec(0.0), yield_parameters_int[i] / wp.max(pvol, _EPSILON))


@fem.integrand
def averaged_elastic_parameters(
    s: fem.Sample,
    elastic_parameters_int: wp.array(dtype=wp.vec3),
    particle_volume: wp.array(dtype=float),
):
    pvol = particle_volume[s.qp_index]
    return elastic_parameters_int[s.qp_index] / wp.max(pvol, _EPSILON)


@wp.kernel
def average_elastic_strain_delta(
    elastic_strain_delta_int: wp.array(dtype=vec6),
    particle_volume: wp.array(dtype=float),
    elastic_strain_delta_avg: wp.array(dtype=vec6),
):
    i = wp.tid()
    pvol = particle_volume[i]

    # The 2 factor is due to the SymTensorMapping being othonormal with (tau:sig)/2
    elastic_strain_delta_avg[i] = elastic_strain_delta_int[i] / wp.max(2.0 * pvol, _EPSILON)


@fem.integrand
def advect_particles(
    s: fem.Sample,
    grid_vel: fem.Field,
    dt: float,
    max_vel: float,
    particle_flags: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    vel_grad: wp.array(dtype=wp.mat33),
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        pos[s.qp_index] = pos_prev[s.qp_index]
        return

    p_vel = grid_vel(s)
    vel_n_sq = wp.length_sq(p_vel)

    p_vel_cfl = wp.where(vel_n_sq > max_vel * max_vel, p_vel * max_vel / wp.sqrt(vel_n_sq), p_vel)

    p_vel_grad = fem.grad(grid_vel, s)

    pos_adv = pos_prev[s.qp_index] + dt * p_vel_cfl

    pos[s.qp_index] = pos_adv
    vel[s.qp_index] = p_vel_cfl
    vel_grad[s.qp_index] = p_vel_grad


@fem.integrand
def update_particle_strains(
    s: fem.Sample,
    grid_vel: fem.Field,
    plastic_strain_delta: fem.Field,
    elastic_strain_delta: fem.Field,
    dt: float,
    particle_flags: wp.array(dtype=wp.int32),
    material_parameters: MaterialParameters,
    elastic_strain_prev: wp.array(dtype=wp.mat33),
    particle_Jp_prev: wp.array(dtype=float),
    elastic_strain: wp.array(dtype=wp.mat33),
    particle_Jp: wp.array(dtype=float),
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return

    # plastic strain
    p_strain_delta = plastic_strain_delta(s)
    delta_Jp = wp.determinant(p_strain_delta + wp.identity(n=3, dtype=float))
    particle_Jp[s.qp_index] = particle_Jp_prev[s.qp_index] * wp.clamp(delta_Jp, MIN_JP_DELTA, MAX_JP_DELTA)

    # elastic strain
    prev_strain = elastic_strain_prev[s.qp_index]
    strain_delta = elastic_strain_delta(s)  # + skew * dt
    strain_new = prev_strain + strain_delta @ prev_strain

    elastic_parameters_vec = get_elastic_parameters(s.qp_index, material_parameters, particle_Jp)
    compliance, poisson, _damping = extract_elastic_parameters(elastic_parameters_vec)

    yield_parameters_vec = get_yield_parameters(s.qp_index, material_parameters, particle_Jp)

    strain_proj = project_particle_strain(
        s.qp_index, strain_new, prev_strain, compliance, poisson, yield_parameters_vec
    )

    # rotation
    rot = fem.curl(grid_vel, s) * dt
    q = wp.quat_from_axis_angle(wp.normalize(rot), wp.length(rot))
    R = wp.quat_to_matrix(q)

    elastic_strain[s.qp_index] = R @ strain_proj


@wp.func
def project_particle_strain(
    i: int,
    F: wp.mat33,
    F_prev: wp.mat33,
    compliance: float,
    poisson: float,
    yield_parameters_vec: YieldParamVec,
):
    if compliance <= _EPSILON:
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
    vel_grad: wp.array(dtype=wp.mat33),
    transform_prev: wp.array(dtype=wp.mat33),
    transform: wp.array(dtype=wp.mat33),
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
    return wp.ddot(fem.grad(u, s), tau(s)) * (dt * inv_cell_volume)


@wp.kernel
def compute_unilateral_strain_offset(
    max_fraction: float,
    particle_volume: wp.array(dtype=float),
    collider_volume: wp.array(dtype=float),
    node_volume: wp.array(dtype=float),
    unilateral_strain_offset: wp.array(dtype=float),
):
    i = wp.tid()

    spherical_part = max_fraction * (node_volume[i] - collider_volume[i]) - particle_volume[i]
    spherical_part = wp.max(spherical_part, 0.0)

    strain_offset = spherical_part / 3.0 * wp.identity(n=3, dtype=float)

    offset_vec = fem.SymmetricTensorMapper.value_to_dof_3d(strain_offset)
    unilateral_strain_offset[i] = offset_vec[0]


@wp.func
def stress_strain_relationship(sig: wp.mat33, compliance: float, poisson: float):
    return (sig * (1.0 + poisson) - poisson * (wp.trace(sig) * wp.identity(n=3, dtype=float))) * compliance


@fem.integrand
def strain_rhs(
    s: fem.Sample,
    tau: fem.Field,
    elastic_parameters: fem.Field,
    elastic_strains: wp.array(dtype=wp.mat33),
    inv_cell_volume: float,
    dt: float,
):
    F_prev = elastic_strains[s.qp_index]

    U_prev, xi_prev, _V_prev = wp.svd3(F_prev)

    _compliance, _poisson, damping = extract_elastic_parameters(elastic_parameters(s))

    alpha = 1.0 / (1.0 + damping * dt)

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
    elastic_strains: wp.array(dtype=wp.mat33),
    inv_cell_volume: float,
    dt: float,
):
    F = elastic_strains[s.qp_index]

    compliance, poisson, damping = extract_elastic_parameters(elastic_parameters(s))

    U, xi, V = wp.svd3(F)

    Rt = V @ wp.transpose(U)
    FinvT = U @ wp.diag(1.0 / xi) @ wp.transpose(V)
    return (
        wp.ddot(
            Rt @ tau(s) @ FinvT,
            stress_strain_relationship(Rt @ sig(s) @ FinvT, compliance / (1.0 + damping * dt), poisson),
        )
        * inv_cell_volume
    )


@dataclass
class ImplicitMPMOptions:
    """Implicit MPM solver options."""

    # numerics
    max_iterations: int = 250
    """Maximum number of iterations for the rheology solver."""
    tolerance: float = 1.0e-5
    """Tolerance for the rheology solver."""
    voxel_size: float = 0.1
    """Size of the grid voxels."""
    grid_padding: int = 0
    """Number of empty cells to add around particles."""
    strain_basis: str = "P0"
    """Strain basis functions. May be one of P0, Q1"""
    solver: str = "gauss-seidel"
    """Solver to use for the rheology solver. May be one of gauss-seidel, jacobi."""

    # grid
    grid_type: str = "sparse"
    """Type of grid to use. May be one of sparse, dense, fixed."""
    transfer_scheme: str = "apic"
    """Transfer scheme to use for particle-grid transfers. May be one of apic, pic."""

    # plasticity
    yield_pressure: float = 1.0e12
    """Yield pressure for the plasticity model. (Pa)"""
    tensile_yield_ratio: float = 0.0
    """Tensile yield ratio for the plasticity model."""
    yield_stress: float = 0.0
    """Yield stress for the plasticity model. (Pa)"""
    hardening: float = 0.0
    """Hardening factor for the plasticity model (Multiplier for det(Fp))."""
    critical_fraction: float = 0.0
    """Fraction for particles under which the yield surface collapses."""

    # elasticity (experimental)
    young_modulus: float = _INFINITY
    """Young's modulus for the elasticity model. (Pa)"""
    poisson_ratio: float = 0.3
    """Poisson's ratio for the elasticity model."""
    damping: float = 0.0
    """Damping for the elasticity model."""

    # background
    air_drag: float = 1.0
    """Numerical drag for the background air."""


class _ImplicitMPMScratchpad:
    """Per-step spaces, fields, and temporaries for the implicit MPM solver."""

    def __init__(self):
        self.velocity_test = None
        self.velocity_trial = None
        self.fraction_test = None

        self.sym_strain_test = None
        self.sym_strain_trial = None
        self.divergence_test = None
        self.fraction_field = None
        self.elastic_parameters_field = None

        self.collider_velocity_field = None
        self.plastic_strain_delta_field = None
        self.elastic_strain_delta_field = None
        self.strain_yield_parameters_field = None
        self.strain_yield_parameters_test = None

        self.strain_matrix = sp.bsr_zeros(0, 0, mat63)
        self.transposed_strain_matrix = sp.bsr_zeros(0, 0, mat36)

        self.color_offsets = None
        self.color_indices = None
        self.color_nodes_per_element = 1

        self.collider_quadrature = None

        self.inv_mass_matrix = None
        self.collider_normal = None
        self.collider_friction = None
        self.collider_adhesion = None
        self.collider_inv_mass_matrix = None

        self.strain_node_particle_volume = None
        self.strain_node_volume = None
        self.strain_node_collider_volume = None

        self.int_symmetric_strain = None

        self.collider_total_volumes = None
        self.vel_node_volume = None

    def create_function_spaces(
        self,
        geo_partition: fem.GeometryPartition,
        strain_basis_str: str,
    ):
        """Create velocity and strain function spaces over the given geometry."""
        self.domain = fem.Cells(geo_partition)

        self._create_velocity_function_space()
        self._create_strain_function_space(strain_basis_str)

    def _create_velocity_function_space(self):
        """Create velocity and fraction spaces and their partition/restriction."""
        domain = self.domain
        grid = domain.geometry

        # Define function spaces: linear (Q1) for velocity and volume fraction,
        # zero or first order for pressure
        velocity_basis = fem.make_polynomial_basis_space(grid, degree=1)

        velocity_space = fem.make_collocated_function_space(velocity_basis, dtype=wp.vec3)

        vel_space_partition = fem.make_space_partition(
            space_topology=velocity_space.topology, geometry_partition=domain.geometry_partition, with_halo=False
        )

        vel_space_restriction = fem.make_space_restriction(space_partition=vel_space_partition, domain=domain)

        self._velocity_basis = velocity_basis
        self._velocity_space = velocity_space
        self._vel_space_restriction = vel_space_restriction

    def _create_strain_function_space(self, strain_basis_str: str):
        """Create symmetric strain space (P0 or Q1) and its partition/restriction."""
        domain = self.domain
        grid = domain.geometry

        if strain_basis_str not in ("P0", "Q1"):
            raise ValueError(f"Unsupported strain basis: {strain_basis_str}")

        strain_degree = 0 if strain_basis_str == "P0" else 1
        discontinuous = strain_basis_str != "Q1"

        strain_basis = fem.make_polynomial_basis_space(
            grid,
            degree=strain_degree,
            discontinuous=discontinuous,
        )

        sym_strain_space = fem.make_collocated_function_space(
            strain_basis,
            dof_mapper=fem.SymmetricTensorMapper(dtype=wp.mat33, mapping=fem.SymmetricTensorMapper.Mapping.DB16),
        )

        strain_space_partition = fem.make_space_partition(
            space_topology=sym_strain_space.topology, geometry_partition=domain.geometry_partition, with_halo=False
        )

        strain_space_restriction = fem.make_space_restriction(space_partition=strain_space_partition, domain=domain)

        self._strain_basis = strain_basis
        self._sym_strain_space = sym_strain_space
        self._strain_space_restriction = strain_space_restriction

    def require_velocity_space_fields(self, has_compliant_particles: bool):
        """Ensure velocity-space fields exist and match current spaces."""
        velocity_basis = self._velocity_basis
        velocity_space = self._velocity_space
        vel_space_restriction = self._vel_space_restriction
        domain = vel_space_restriction.domain
        vel_space_partition = vel_space_restriction.space_partition

        fraction_space = fem.make_collocated_function_space(velocity_basis, dtype=float)

        # test, trial and discrete fields
        self.velocity_test = fem.make_test(velocity_space, domain=domain, space_restriction=vel_space_restriction)
        self.fraction_test = fem.make_test(fraction_space, space_restriction=vel_space_restriction)

        self.velocity_trial = fem.make_trial(velocity_space, domain=domain, space_partition=vel_space_partition)

        self.fraction_field = fem.make_discrete_field(fraction_space, space_partition=vel_space_partition)
        self.collider_velocity_field = velocity_space.make_field(space_partition=vel_space_partition)
        self.collider_distance_field = fraction_space.make_field(space_partition=vel_space_partition)

        if has_compliant_particles:
            elastic_parameters_space = fem.make_collocated_function_space(velocity_basis, dtype=wp.vec3)
            self.elastic_parameters_field = elastic_parameters_space.make_field(space_partition=vel_space_partition)

        collider_quadrature_order = self._sym_strain_space.degree + 1
        self.collider_quadrature = fem.RegularQuadrature(
            domain=domain,
            order=collider_quadrature_order,
            family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE,
        )
        self.background_impulse_field = fem.UniformField(domain, wp.vec3(0.0))

        self.impulse_field = velocity_space.make_field(space_partition=vel_space_partition)
        self.velocity_field = velocity_space.make_field(space_partition=vel_space_partition)
        self.collider_ids = wp.empty(velocity_space.node_count(), dtype=int)

    def require_strain_space_fields(self):
        """Ensure strain-space fields exist and match current spaces."""
        strain_basis = self._strain_basis
        sym_strain_space = self._sym_strain_space
        strain_space_restriction = self._strain_space_restriction
        domain = strain_space_restriction.domain
        strain_space_partition = strain_space_restriction.space_partition

        divergence_space = fem.make_collocated_function_space(strain_basis, dtype=float)
        strain_yield_parameters_space = fem.make_collocated_function_space(strain_basis, dtype=YieldParamVec)

        self.sym_strain_test = fem.make_test(sym_strain_space, space_restriction=strain_space_restriction)
        self.divergence_test = fem.make_test(divergence_space, space_restriction=strain_space_restriction)
        self.strain_yield_parameters_test = fem.make_test(
            strain_yield_parameters_space, space_restriction=strain_space_restriction
        )

        self.sym_strain_trial = fem.make_trial(sym_strain_space, domain=domain, space_partition=strain_space_partition)

        self.elastic_strain_delta_field = sym_strain_space.make_field(space_partition=strain_space_partition)
        self.plastic_strain_delta_field = sym_strain_space.make_field(space_partition=strain_space_partition)
        self.strain_yield_parameters_field = strain_yield_parameters_space.make_field(
            space_partition=strain_space_partition
        )

        self.background_stress_field = fem.UniformField(domain, wp.mat33(0.0))

        self.stress_field = sym_strain_space.make_field(space_partition=strain_space_partition)

    def allocate_temporaries(
        self,
        collider_count: int,
        has_compliant_bodies: bool,
        has_critical_fraction: bool,
        coloring: bool,
        temporary_store: fem.TemporaryStore,
    ):
        """Allocate transient arrays sized to current grid and options."""
        vel_node_count = self._vel_space_restriction.space_partition.node_count()
        strain_node_count = self._strain_space_restriction.space_partition.node_count()

        self.inv_mass_matrix = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)
        self.node_positions = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=wp.vec3)

        self.collider_normal = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=wp.vec3)
        self.collider_friction = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)
        self.collider_adhesion = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)
        self.collider_inv_mass_matrix = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)

        self.strain_node_particle_volume = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)
        self.int_symmetric_strain = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=vec6)

        sp.bsr_set_zero(self.strain_matrix, rows_of_blocks=strain_node_count, cols_of_blocks=vel_node_count)

        if has_critical_fraction:
            self.strain_node_volume = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)
            self.strain_node_collider_volume = fem.borrow_temporary(
                temporary_store, shape=strain_node_count, dtype=float
            )

        if has_compliant_bodies:
            self.collider_total_volumes = fem.borrow_temporary(temporary_store, shape=collider_count, dtype=float)
            self.vel_node_volume = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)

        if coloring:
            self.color_indices = fem.borrow_temporary(temporary_store, shape=strain_node_count * 2 + 1, dtype=int)

    def release_temporaries(self):
        """Release previously allocated temporaries to the store."""
        self.inv_mass_matrix.release()
        self.node_positions.release()
        self.collider_normal.release()
        self.collider_friction.release()
        self.collider_adhesion.release()
        self.collider_inv_mass_matrix.release()
        self.int_symmetric_strain.release()
        self.strain_node_particle_volume.release()

        if self.strain_node_volume is not None:
            self.strain_node_volume.release()
            self.strain_node_collider_volume.release()

        if self.vel_node_volume is not None:
            self.collider_total_volumes.release()
            self.vel_node_volume.release()

        if self.color_indices is not None:
            self.color_indices.release()


def _particle_parameter(
    num_particles, model_value: float | wp.array | None = None, default_value=None, model_scale: wp.array | None = None
):
    """Helper function to create a particle-wise parameter array, taking defaults either from the model
    or the global options."""

    if model_value is None:
        return wp.full(num_particles, default_value, dtype=float)
    elif isinstance(model_value, wp.array):
        if model_value.shape[0] != num_particles:
            raise ValueError(f"Model value array must have {num_particles} elements")

        return model_value if model_scale is None else model_value * model_scale
    else:
        return wp.full(num_particles, model_value, dtype=float) if model_scale is None else model_value * model_scale


class ImplicitMPMModel:
    """Wrapper augmenting a ``newton.Model`` with implicit MPM data and setup.

    Holds particle material parameters, collider parameters, and convenience
    arrays derived from the wrapped ``model`` and ``ImplicitMPMOptions``. The
    instance is consumed by ``SolverImplicitMPM`` during time stepping.

    Args:
        model: The base Newton model to augment.
        options: Options controlling particle and collider defaults.
    """

    def __init__(self, model: newton.Model, options: ImplicitMPMOptions):
        self.model = model

        self.critical_fraction = float(options.critical_fraction)
        """Maximum fraction of the grid volume that can be occupied by particles"""

        self.voxel_size = float(options.voxel_size)
        """Size of the grid voxels"""

        self.air_drag = float(options.air_drag)
        """Drag for the background air"""

        self.material_parameters = MaterialParameters()
        """Material parameters struct"""

        self.collider = Collider()
        """Collider struct"""

        self.collider_coms = None
        self.collider_inv_inertia = None

        self.setup_particle_material(options)
        self.setup_collider()

    def notify_particle_material_changed(self):
        """Refresh cached extrema for material parameters.

        Tracks the minimum Young's modulus and maximum hardening across
        particles to quickly toggle code paths (e.g., compliant particles or
        hardening enabled) without recomputing per step.
        """
        self.min_young_modulus = np.min(self.material_parameters.young_modulus.numpy())
        self.max_hardening = np.max(self.material_parameters.hardening.numpy())

    def notify_collider_changed(self):
        """Refresh cached extrema for collider parameters.

        Tracks the minimum collider mass to determine whether compliant
        colliders are present and to enable/disable related computations.
        """
        self.min_collider_mass = np.min(self.collider.masses.numpy()) if len(self.collider.masses) > 0 else _INFINITY
        self.collider.query_max_dist = self.voxel_size * math.sqrt(3.0)

    def setup_particle_material(self, options: ImplicitMPMOptions):
        """Initialize per-particle material and derived fields from the model.

        Computes particle volumes and densities from the model's particle mass
        and radius, sets up elastic and plasticity parameters with optional
        hardening, and stores them into ``self.material_parameters``. Also
        caches extrema used by the solver for fast feature toggles.

        Args:
            options: Solver options used to fill defaults for missing model
                parameters (e.g., global Young's modulus, damping, Poisson).
        """
        model = self.model

        num_particles = model.particle_q.shape[0]

        with wp.ScopedDevice(model.device):
            # Assume that particles represent a cuboid volume of space
            # (they are typically laid out on a grid)
            self.particle_radius = _particle_parameter(num_particles, model.particle_radius)
            self.particle_volume = wp.array(8.0 * self.particle_radius.numpy() ** 3)
            self.particle_density = model.particle_mass / self.particle_volume

            # Map newton.Model parameters to MPM material parameters
            # young_modulus = particle_ke / particle_volume
            self.material_parameters.young_modulus = _particle_parameter(
                num_particles, model.particle_ke, options.young_modulus, model_scale=1.0 / self.particle_volume
            )
            # damping = particle_kd / particle_ke = particle_kd / (young modulus * particle_volume)
            self.material_parameters.damping = _particle_parameter(
                num_particles,
                model.particle_kd,
                options.damping,
                model_scale=1.0 / (self.particle_volume * self.material_parameters.young_modulus),
            )
            self.material_parameters.poisson_ratio = wp.full(num_particles, options.poisson_ratio, dtype=float)

            self.material_parameters.hardening = wp.full(num_particles, options.hardening, dtype=float)
            self.material_parameters.friction = _particle_parameter(num_particles, model.particle_mu)
            self.material_parameters.yield_pressure = wp.full(num_particles, options.yield_pressure, dtype=float)

            # tensile yield ratio = adhesion * young modulus / yield pressure
            self.material_parameters.tensile_yield_ratio = _particle_parameter(
                num_particles,
                model.particle_adhesion,
                options.tensile_yield_ratio,
                model_scale=self.material_parameters.young_modulus / self.material_parameters.yield_pressure,
            )
            # deviatoric yield stress = cohesion * young modulus
            self.material_parameters.yield_stress = _particle_parameter(
                num_particles,
                model.particle_cohesion,
                options.yield_stress,
                model_scale=self.material_parameters.young_modulus,
            )

        self.notify_particle_material_changed()

    def setup_collider(
        self,
        # TODO: read colliders from model
        colliders: list[wp.Mesh] | None = None,
        collider_thicknesses: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        collider_masses: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_adhesion: list[float] | None = None,
        ground_height: float = 0.0,
        ground_normal: wp.vec3 | None = None,
    ):
        """Initialize collider parameters and defaults from inputs.

        Populates the ``Collider`` struct with mesh handles and per-mesh
        properties (thickness, friction, adhesion, mass, projection threshold),
        using solver defaults when lists are not provided. Also sets the ground
        plane parameters from the wrapped model's up axis.

        Args:
            colliders: Warp triangular meshes used as colliders.
            collider_thicknesses: Per-mesh signed distance offsets (m).
            collider_projection_threshold: Per-mesh projection thresholds as a
                fraction of the voxel size.
            collider_masses: Per-mesh masses; very large values mark kinematic colliders.
            collider_friction: Per-mesh Coulomb friction coefficients.
            collider_adhesion: Per-mesh adhesion (Pa).
            ground_height: Height of the ground plane.
            ground_normal: Normal of the ground plane (default to model.up_axis).
        """

        self._collider_meshes = colliders  # Keep a ref so that meshes are not garbage collected

        if colliders is None:
            colliders = []

        collider = self.collider

        with wp.ScopedDevice(self.model.device):
            collider.meshes = wp.array([collider.id for collider in colliders], dtype=wp.uint64)
            collider.thicknesses = (
                wp.full(len(collider.meshes), _DEFAULT_THICKNESS * self.voxel_size, dtype=float)
                if collider_thicknesses is None
                else wp.array(collider_thicknesses, dtype=float)
            )
            collider.friction = (
                wp.full(len(collider.meshes), _DEFAULT_FRICTION, dtype=float)
                if collider_friction is None
                else wp.array(collider_friction, dtype=float)
            )
            collider.adhesion = (
                wp.full(len(collider.meshes), _DEFAULT_ADHESION, dtype=float)
                if collider_adhesion is None
                else wp.array(collider_adhesion, dtype=float)
            )
            collider.masses = (
                wp.full(len(collider.meshes), _INFINITY * 2.0, dtype=float)
                if collider_masses is None
                else wp.array(collider_masses, dtype=float)
            )
            collider.projection_threshold = (
                wp.full(len(collider.meshes), _DEFAULT_PROJECTION_THRESHOLD, dtype=float)
                if collider_projection_threshold is None
                else wp.array(collider_projection_threshold, dtype=float)
            )

            self.collider_coms = wp.zeros(len(collider.meshes), dtype=wp.vec3)
            self.collider_inv_inertia = wp.zeros(len(collider.meshes), dtype=wp.mat33)

        collider.ground_height = ground_height
        if ground_normal is None:
            collider.ground_normal = wp.vec3(0.0)
            collider.ground_normal[self.model.up_axis] = 1.0
        else:
            collider.ground_normal = wp.vec3(ground_normal)

        self.notify_collider_changed()

    def state(self):
        """
        Create and return a new :class:`State` object for this model.

        The returned state is initialized with the initial configuration from the model description.

        Returns:
            State: The state object, enriched with MPM-specific attributes
        """
        state = self.model.state()
        SolverImplicitMPM.enrich_state(state)
        return state


@wp.kernel
def compute_bounds(
    pos: wp.array(dtype=wp.vec3),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    block_id, lane = wp.tid()
    i = block_id * wp.block_dim() + lane

    # pad with +- inf for min/max
    # tile_min scalar only, so separate components
    # no tile_atomic_min yet, extract first and use lane 0

    if i >= pos.shape[0]:
        min_x = _INFINITY
        min_y = _INFINITY
        min_z = _INFINITY
        max_x = -_INFINITY
        max_y = -_INFINITY
        max_z = -_INFINITY
    else:
        p = pos[i]
        min_x = p[0]
        min_y = p[1]
        min_z = p[2]
        max_x = p[0]
        max_y = p[1]
        max_z = p[2]

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
    coords: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    coords[i] = wp.min(wp.max(coords[i], wp.vec3(0.0)), wp.vec3(1.0))


@wp.kernel
def pad_voxels(particle_q: wp.array(dtype=wp.vec3i), padded_q: wp.array4d(dtype=wp.vec3i)):
    pid = wp.tid()

    for i in range(3):
        for j in range(3):
            for k in range(3):
                padded_q[pid, i, j, k] = particle_q[pid] + wp.vec3i(i - 1, j - 1, k - 1)


@wp.func
def positive_mod3(x: int):
    return (x % 3 + 3) % 3


def _allocate_by_voxels(particle_q, voxel_size, padding_voxels: int = 0):
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


class SolverImplicitMPM(SolverBase):
    """Implicit MPM solver.

    This solver implements an implicit MPM algorithm for granular materials,
    roughly following [1] but with a GPU-friendly rheology solver.

    This variant of MPM is mostly interesting for very stiff materials, especially
    in the fully inelastic limit, but is not as versatile as more traditional explicit approaches.

    [1] https://doi.org/10.1145/2897824.2925877

    Args:
        model: The model to solve.
        options: The solver options.

    Returns:
        The solver.
    """

    Options = ImplicitMPMOptions
    Model = ImplicitMPMModel

    def __init__(
        self,
        model: newton.Model | ImplicitMPMModel,
        options: ImplicitMPMOptions,
    ):
        if isinstance(model, ImplicitMPMModel):
            self.mpm_model = model
            model = self.mpm_model.model
        else:
            self.mpm_model = ImplicitMPMModel(model, options)

        super().__init__(model)

        self.max_iterations = options.max_iterations
        self.tolerance = float(options.tolerance)

        self.strain_basis = options.strain_basis

        self.grid_padding = options.grid_padding
        self.grid_type = options.grid_type
        self.coloring = options.solver == "gauss-seidel"
        self.apic = options.transfer_scheme == "apic"

        self.temporary_store = fem.TemporaryStore()
        self._scratchpad = None

        self._use_cuda_graph = False
        if self.model.device.is_cuda:
            try:
                assert_conditional_graph_support()
                self._use_cuda_graph = True
            except Exception:
                pass

        self._enable_timers = False
        self._timers_use_nvtx = False

        if self.grid_type == "fixed":
            self._rebuild_scratchpad(model.particle_q)

    @staticmethod
    def enrich_state(state: newton.State):
        """Allocate additional per-particle and per-step fields used by the solver.

        Adds velocity gradient, elastic and plastic deformation storage, and
        per-particle rendering transforms. Initializes grid-attached fields to
        None so they can be created on demand during stepping.
        """

        device = state.particle_qd.device

        state.particle_qd_grad = wp.zeros(state.particle_qd.shape[0], dtype=wp.mat33, device=device)
        """Velocity gradient, for APIC particle-grid transfer"""

        identity = wp.mat33(np.eye(3))
        state.particle_elastic_strain = wp.full(
            state.particle_qd.shape[0], value=identity, dtype=wp.mat33, device=device
        )
        """Elastic deformation gradient"""

        state.particle_Jp = wp.ones(state.particle_qd.shape[0], dtype=float, device=device)
        """Determinant of plastic deformation gradient, for hardening"""

        state.particle_transform = wp.full(state.particle_qd.shape[0], value=identity, dtype=wp.mat33, device=device)
        """Overall deformation gradient, for grain-based rendering"""

        state.velocity_field = None
        state.impulse_field = None
        state.stress_field = None
        state.collider_ids = None

    @override
    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control,
        contacts: newton.Contacts,
        dt: float,
    ):
        model = self.model

        with wp.ScopedDevice(model.device):
            if self.grid_type != "fixed":
                self._rebuild_scratchpad(state_in.particle_q)

            self._step_impl(state_in, state_out, dt, self._scratchpad)

            if self.grid_type != "fixed":
                self._scratchpad.release_temporaries()

    @override
    def notify_model_changed(self, flags: int):
        if flags & newton.SolverNotifyFlags.PARTICLE_PROPERTIES:
            self.mpm_model.notify_particle_material_changed()

    def project_outside(
        self, state_in: newton.State, state_out: newton.State, dt: float, max_dist: float | None = None
    ):
        """Project particles outside of colliders, and adjust their velocity and velocity gradients

        Args:
            state_in: The input state.
            state_out: The output state. Only particle_q, particle_qd, and particle_qd_grad are written.
            dt: The time step, for extrapolating the collider end-of-step positions from its current position and velocity.
            max_dist: Maximum distance for closest-point queries. If None, the default is the voxel size times sqrt(3).
        """

        if max_dist is not None:
            # Update max query dist if provided
            prev_max_dist, self.mpm_model.collider.query_max_dist = self.mpm_model.collider.query_max_dist, max_dist

        wp.launch(
            project_outside_collider,
            dim=state_in.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_qd_grad,
                self.model.particle_flags,
                self.mpm_model.collider,
                self.mpm_model.voxel_size,
                dt,
            ],
            outputs=[
                state_out.particle_q,
                state_out.particle_qd,
                state_out.particle_qd_grad,
            ],
            device=state_in.particle_q.device,
        )

        if max_dist is not None:
            # Restore previous max query dist
            self.mpm_model.collider.query_max_dist = prev_max_dist

    def collect_collider_impulses(self, state: newton.State):
        """Collect current collider impulses and their application positions.

        The impulses are returned in world units, integrating the internal
        nodal impulse density over the voxel volume. Collider ids can be
        retrieved from ``state.collider_ids`` if needed.
        """
        x = state.velocity_field.space.node_positions()

        collider_impulse = wp.zeros_like(state.impulse_field.dof_values)
        cell_volume = self.mpm_model.voxel_size**3
        fem.utils.array_axpy(
            y=collider_impulse,
            x=state.impulse_field.dof_values,
            alpha=-cell_volume,
            beta=0.0,
        )

        return collider_impulse, x

    def update_particle_frames(
        self,
        state_prev: newton.State,
        state: newton.State,
        dt: float,
        min_stretch: float = 0.25,
        max_stretch: float = 2.0,
    ):
        """Update per-particle deformation frames for rendering and projection.

        Integrates the particle deformation gradient using the velocity gradient
        and clamps its principal stretches to the provided bounds for
        robustness.
        """

        wp.launch(
            update_particle_frames,
            dim=state.particle_count,
            inputs=[
                dt,
                min_stretch,
                max_stretch,
                state.particle_qd_grad,
                state_prev.particle_transform,
                state.particle_transform,
            ],
            device=state.particle_qd_grad.device,
        )

    def sample_render_grains(self, state: newton.State, grains_per_particle: int):
        """Generate per-particle point samples used for high-resolution rendering.

        Args:
            state: Current Newton state providing particle positions.
            grains_per_particle: Number of grains to sample per particle.

        Returns:
            A ``wp.array`` with shape ``(num_particles, grains_per_particle)`` of
            type ``wp.vec3`` containing grain positions.
        """

        return sample_render_grains(state, self.mpm_model.particle_radius, grains_per_particle)

    def update_render_grains(
        self,
        state_prev: newton.State,
        state: newton.State,
        grains: wp.array,
        dt: float,
    ):
        """Advect grain samples with the grid velocity and keep them inside the deformed particle.

        Args:
            state_prev: Previous state (t_n).
            state: Current state (t_{n+1}).
            grains: 2D array of grain positions per particle to be updated in place. See ``sample_render_grains``.
            dt: Time step duration.
        """

        return update_render_grains(state_prev, state, grains, self.mpm_model.particle_radius, dt)

    def _allocate_grid(
        self, positions: wp.array, voxel_size, temporary_store: fem.TemporaryStore, padding_voxels: int = 0
    ):
        """Create a grid (sparse or dense) covering all particle positions.

        Uses a sparse ``Nanogrid`` when requested; otherwise computes an axis
        aligned bounding box and instantiates a dense ``Grid3D`` with optional
        padding in voxel units.

        Args:
            positions: Particle positions to bound.
            voxel_size: Grid voxel edge length.
            temporary_store: Temporary storage for intermediate buffers.
            padding_voxels: Additional empty voxels to add around the bounds.

        Returns:
            A geometry partition suitable for FEM field assembly.
        """
        with wp.ScopedTimer(
            "Allocate grid",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            if self.grid_type == "dense":
                # Compute bounds and transfer to host
                device = positions.device
                if device.is_cuda:
                    min_dev = fem.borrow_temporary(temporary_store, shape=1, dtype=wp.vec3, device=device)
                    max_dev = fem.borrow_temporary(temporary_store, shape=1, dtype=wp.vec3, device=device)

                    min_dev.array.fill_(wp.vec3(_INFINITY))
                    max_dev.array.fill_(wp.vec3(-_INFINITY))

                    tile_size = 256
                    wp.launch(
                        compute_bounds,
                        dim=((positions.shape[0] + tile_size - 1) // tile_size, tile_size),
                        block_dim=tile_size,
                        inputs=[positions, min_dev.array, max_dev.array],
                        device=device,
                    )

                    min_host = fem.borrow_temporary(
                        temporary_store, shape=1, dtype=wp.vec3, device="cpu", pinned=device.is_cuda
                    )
                    max_host = fem.borrow_temporary(
                        temporary_store, shape=1, dtype=wp.vec3, device="cpu", pinned=device.is_cuda
                    )
                    wp.copy(src=min_dev.array, dest=min_host.array)
                    wp.copy(src=max_dev.array, dest=max_host.array)
                    wp.synchronize_stream()
                    bbox_min, bbox_max = min_host.array.numpy(), max_host.array.numpy()
                else:
                    bbox_min, bbox_max = np.min(positions.numpy(), axis=0), np.max(positions.numpy(), axis=0)

                # Round to nearest voxel
                grid_min = np.floor(bbox_min / voxel_size) - padding_voxels
                grid_max = np.ceil(bbox_max / voxel_size) + padding_voxels

                grid = fem.Grid3D(
                    bounds_lo=wp.vec3(grid_min * voxel_size),
                    bounds_hi=wp.vec3(grid_max * voxel_size),
                    res=wp.vec3i((grid_max - grid_min).astype(int)),
                )
            else:
                volume = _allocate_by_voxels(positions, voxel_size, padding_voxels=padding_voxels)
                grid = fem.Nanogrid(volume)

        return grid

    def _rebuild_scratchpad(self, positions: wp.array):
        """(Re)create function spaces and allocate per-step temporaries.

        Allocates the grid based on current particle positions, rebuilds
        velocity and strain spaces as needed, configures collision data, and
        optionally computes a Gauss-Seidel coloring for the strain nodes.
        """
        geo_partition = self._allocate_grid(
            positions,
            voxel_size=self.mpm_model.voxel_size,
            temporary_store=self.temporary_store,
            padding_voxels=self.grid_padding,
        )

        with wp.ScopedTimer(
            "Scratchpad",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            if self._scratchpad is None:
                self._scratchpad = _ImplicitMPMScratchpad()
                self._scratchpad.create_function_spaces(geo_partition, strain_basis_str=self.strain_basis)
            elif self.grid_type != "fixed":
                self._scratchpad.create_function_spaces(geo_partition, strain_basis_str=self.strain_basis)

            has_compliant_colliders = self.mpm_model.min_collider_mass < _INFINITY

            self._scratchpad.allocate_temporaries(
                collider_count=self.mpm_model.collider.meshes.shape[0],
                has_compliant_bodies=has_compliant_colliders,
                has_critical_fraction=self.mpm_model.critical_fraction > 0.0,
                coloring=self.coloring,
                temporary_store=self.temporary_store,
            )

            if self.coloring:
                self._compute_coloring(scratch=self._scratchpad)

    def _step_impl(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
        scratch: _ImplicitMPMScratchpad,
    ):
        """Single implicit MPM step: bin, rasterize, assemble, solve, advect.

        Executes the full pipeline for one time step, including particle
        binning, collider rasterization, RHS assembly, strain/compliance matrix
        computation, warm-starting, coupled rheology/contact solve, strain
        updates, and particle advection.
        """
        domain = scratch.domain
        inv_cell_volume = 1.0 / self.mpm_model.voxel_size**3

        model = self.model
        mpm_model = self.mpm_model
        has_compliant_particles = mpm_model.min_young_modulus < _INFINITY
        has_compliant_colliders = mpm_model.min_collider_mass < _INFINITY
        has_hardening = mpm_model.max_hardening > 0.0

        # Bin particles to grid cells
        with wp.ScopedTimer(
            "Bin particles",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            pic = fem.PicQuadrature(
                domain=domain,
                positions=state_in.particle_q,
                measures=mpm_model.particle_volume,
            )

            if self.grid_type == "fixed":
                wp.launch(
                    clamp_coordinates,
                    dim=pic.particle_coords.shape,
                    inputs=[pic.particle_coords],
                )

        scratch.require_velocity_space_fields(has_compliant_particles=has_compliant_particles)
        vel_node_count = scratch.velocity_field.space.node_count()

        with wp.ScopedTimer(
            "Rasterize collider",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            scratch.collider_velocity_field.space.node_positions(out=scratch.node_positions.array)
            wp.launch(
                rasterize_collider,
                dim=vel_node_count,
                inputs=[
                    self.mpm_model.collider,
                    self.mpm_model.voxel_size,
                    dt,
                    scratch.node_positions.array,
                    scratch.collider_distance_field.dof_values,
                    scratch.collider_velocity_field.dof_values,
                    scratch.collider_normal.array,
                    scratch.collider_friction.array,
                    scratch.collider_adhesion.array,
                    scratch.collider_ids,
                ],
            )
        # Velocity right-hand side and inverse mass matrix
        with wp.ScopedTimer(
            "Unconstrained velocity",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            velocity_int = fem.integrate(
                integrate_velocity,
                quadrature=pic,
                fields={"u": scratch.velocity_test},
                values={
                    "velocities": state_in.particle_qd,
                    "dt": dt,
                    "gravity": model.gravity,
                    "particle_density": mpm_model.particle_density,
                    "particle_flags": model.particle_flags,
                    "inv_cell_volume": inv_cell_volume,
                },
                output_dtype=wp.vec3,
            )

            if self.apic:
                fem.integrate(
                    integrate_velocity_apic,
                    quadrature=pic,
                    fields={"u": scratch.velocity_test},
                    values={
                        "velocity_gradients": state_in.particle_qd_grad,
                        "particle_density": mpm_model.particle_density,
                        "particle_flags": model.particle_flags,
                        "inv_cell_volume": inv_cell_volume,
                    },
                    output=velocity_int,
                    add=True,
                )

            node_particle_mass = fem.integrate(
                integrate_mass,
                quadrature=pic,
                fields={"phi": scratch.fraction_test},
                values={
                    "inv_cell_volume": inv_cell_volume,
                    "particle_density": mpm_model.particle_density,
                    "particle_flags": model.particle_flags,
                },
                output_dtype=float,
            )

            drag = mpm_model.air_drag * dt
            wp.launch(
                free_velocity,
                dim=vel_node_count,
                inputs=[
                    velocity_int,
                    node_particle_mass,
                    drag,
                ],
                outputs=[
                    scratch.inv_mass_matrix.array,
                    scratch.velocity_field.dof_values,
                ],
            )

        if has_compliant_colliders:
            with wp.ScopedTimer(
                "Collider compliance",
                active=self._enable_timers,
                use_nvtx=self._timers_use_nvtx,
                synchronize=not self._timers_use_nvtx,
            ):
                fem.integrate(
                    integrate_fraction,
                    fields={"phi": scratch.fraction_test},
                    values={"inv_cell_volume": inv_cell_volume},
                    output=scratch.vel_node_volume.array,
                )

                allot_collider_mass(
                    voxel_size=self.mpm_model.voxel_size,
                    node_volumes=scratch.vel_node_volume.array,
                    collider=self.mpm_model.collider,
                    collider_quadrature=scratch.collider_quadrature,
                    collider_ids=scratch.collider_ids,
                    collider_total_volumes=scratch.collider_total_volumes.array,
                    collider_inv_mass_matrix=scratch.collider_inv_mass_matrix.array,
                )

                rigidity_matrix = build_rigidity_matrix(
                    voxel_size=self.mpm_model.voxel_size,
                    node_volumes=scratch.vel_node_volume.array,
                    node_positions=scratch.node_positions.array,
                    collider=self.mpm_model.collider,
                    collider_ids=scratch.collider_ids,
                    collider_coms=self.mpm_model.collider_coms,
                    collider_inv_inertia=self.mpm_model.collider_inv_inertia,
                    collider_total_volumes=scratch.collider_total_volumes.array,
                )
        else:
            rigidity_matrix = None
            scratch.collider_inv_mass_matrix.array.zero_()

        scratch.require_strain_space_fields()
        strain_node_count = scratch.stress_field.space.node_count()

        if has_compliant_particles:
            with wp.ScopedTimer(
                "Elasticity",
                active=self._enable_timers,
                use_nvtx=self._timers_use_nvtx,
                synchronize=not self._timers_use_nvtx,
            ):
                node_particle_volume = fem.integrate(
                    integrate_fraction,
                    quadrature=pic,
                    fields={"phi": scratch.fraction_test},
                    values={"inv_cell_volume": inv_cell_volume},
                    output_dtype=float,
                )

                elastic_parameters_int = fem.integrate(
                    integrate_elastic_parameters,
                    quadrature=pic,
                    fields={"u": scratch.velocity_test},
                    values={
                        "particle_Jp": state_in.particle_Jp,
                        "material_parameters": mpm_model.material_parameters,
                        "inv_cell_volume": inv_cell_volume,
                    },
                    output_dtype=wp.vec3,
                )

                fem.interpolate(
                    averaged_elastic_parameters,
                    dest=fem.make_restriction(
                        scratch.elastic_parameters_field, space_restriction=scratch.velocity_test.space_restriction
                    ),
                    values={"elastic_parameters_int": elastic_parameters_int, "particle_volume": node_particle_volume},
                )

                fem.integrate(
                    strain_rhs,
                    quadrature=pic,
                    fields={
                        "tau": scratch.sym_strain_test,
                        "elastic_parameters": scratch.elastic_parameters_field,
                    },
                    values={
                        "elastic_strains": state_in.particle_elastic_strain,
                        "inv_cell_volume": inv_cell_volume,
                        "dt": dt,
                    },
                    output=scratch.int_symmetric_strain.array,
                )

                C = fem.integrate(
                    compliance_form,
                    quadrature=pic,
                    fields={
                        "tau": scratch.sym_strain_test,
                        "sig": scratch.sym_strain_trial,
                        "elastic_parameters": scratch.elastic_parameters_field,
                    },
                    values={
                        "elastic_strains": state_in.particle_elastic_strain,
                        "inv_cell_volume": inv_cell_volume,
                        "dt": dt,
                    },
                    output_dtype=float,
                )
        else:
            scratch.int_symmetric_strain.array.zero_()

        with wp.ScopedTimer(
            "Compute strain-node volumes",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": scratch.divergence_test},
                values={"inv_cell_volume": inv_cell_volume},
                output=scratch.strain_node_particle_volume.array,
            )

        with wp.ScopedTimer(
            "Interpolated yield parameters",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            yield_parameters_int = fem.integrate(
                integrate_yield_parameters,
                quadrature=pic,
                fields={
                    "u": scratch.strain_yield_parameters_test,
                },
                values={
                    "particle_Jp": state_in.particle_Jp,
                    "material_parameters": mpm_model.material_parameters,
                    "inv_cell_volume": inv_cell_volume,
                },
                output_dtype=YieldParamVec,
            )

            wp.launch(
                average_yield_parameters,
                dim=scratch.strain_node_particle_volume.array.shape[0],
                inputs=[
                    yield_parameters_int,
                    scratch.strain_node_particle_volume.array,
                    scratch.strain_yield_parameters_field.dof_values,
                ],
            )

        # Void fraction (unilateral incompressibility offset)
        unilateral_strain_offset = wp.zeros_like(scratch.strain_node_particle_volume.array)
        if mpm_model.critical_fraction > 0.0:
            with wp.ScopedTimer(
                "Unilateral offset",
                active=self._enable_timers,
                use_nvtx=self._timers_use_nvtx,
                synchronize=not self._timers_use_nvtx,
            ):
                fem.integrate(
                    integrate_fraction,
                    fields={"phi": scratch.divergence_test},
                    values={"inv_cell_volume": inv_cell_volume},
                    output=scratch.strain_node_volume.array,
                )
                fem.integrate(
                    integrate_collider_fraction,
                    quadrature=scratch.collider_quadrature,
                    fields={
                        "phi": scratch.divergence_test,
                        "sdf": scratch.collider_distance_field,
                    },
                    values={
                        "inv_cell_volume": inv_cell_volume,
                    },
                    output=scratch.strain_node_collider_volume.array,
                )

                wp.launch(
                    compute_unilateral_strain_offset,
                    dim=strain_node_count,
                    inputs=[
                        mpm_model.critical_fraction,
                        scratch.strain_node_particle_volume.array,
                        scratch.strain_node_collider_volume.array,
                        scratch.strain_node_volume.array,
                        unilateral_strain_offset,
                    ],
                )

        # Strain jacobian
        with wp.ScopedTimer(
            "Strain matrix",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            fem.integrate(
                strain_delta_form,
                quadrature=pic,
                fields={
                    "u": scratch.velocity_trial,
                    "tau": scratch.sym_strain_test,
                },
                values={
                    "dt": dt,
                    "inv_cell_volume": inv_cell_volume,
                },
                output_dtype=float,
                output=scratch.strain_matrix,
            )

        with wp.ScopedTimer(
            "Warmstart fields",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            self._warmstart_fields(scratch, state_in, state_out)

        with wp.ScopedTimer(
            "Strain solve",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            # Retain graph to avoid immediate CPU synch
            _solve_graph = solve_rheology(
                self.max_iterations,
                self.tolerance,
                scratch.strain_matrix,
                scratch.transposed_strain_matrix,
                C if has_compliant_particles else None,
                scratch.inv_mass_matrix.array,
                scratch.strain_node_particle_volume.array,
                scratch.strain_yield_parameters_field.dof_values,
                unilateral_strain_offset,
                scratch.int_symmetric_strain.array,
                scratch.plastic_strain_delta_field.dof_values,
                state_out.stress_field.dof_values,
                state_out.velocity_field.dof_values,
                scratch.collider_friction.array,
                scratch.collider_adhesion.array,
                scratch.collider_normal.array,
                scratch.collider_velocity_field.dof_values,
                scratch.collider_inv_mass_matrix.array,
                state_out.impulse_field.dof_values,
                color_offsets=scratch.color_offsets,
                color_indices=None if scratch.color_indices is None else scratch.color_indices.array,
                color_nodes_per_element=scratch.color_nodes_per_element,
                rigidity_mat=rigidity_matrix,
                temporary_store=self.temporary_store,
                use_graph=self._use_cuda_graph,
            )

        if has_compliant_particles:
            delta_strain = scratch.int_symmetric_strain.array
            wp.launch(
                average_elastic_strain_delta,
                dim=strain_node_count,
                inputs=[
                    delta_strain,
                    scratch.strain_node_particle_volume.array,
                    scratch.elastic_strain_delta_field.dof_values,
                ],
            )

        if has_compliant_particles or has_hardening:
            with wp.ScopedTimer(
                "Particle strain update",
                active=self._enable_timers,
                use_nvtx=self._timers_use_nvtx,
                synchronize=not self._timers_use_nvtx,
            ):
                # Update particle elastic strain from grid strain delta
                fem.interpolate(
                    update_particle_strains,
                    quadrature=pic,
                    values={
                        "dt": dt,
                        "particle_flags": model.particle_flags,
                        "elastic_strain_prev": state_in.particle_elastic_strain,
                        "elastic_strain": state_out.particle_elastic_strain,
                        "particle_Jp_prev": state_in.particle_Jp,
                        "particle_Jp": state_out.particle_Jp,
                        "material_parameters": mpm_model.material_parameters,
                    },
                    fields={
                        "grid_vel": state_out.velocity_field,
                        "plastic_strain_delta": scratch.plastic_strain_delta_field,
                        "elastic_strain_delta": scratch.elastic_strain_delta_field,
                    },
                )

        # (A)PIC advection
        with wp.ScopedTimer(
            "Advection",
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        ):
            fem.interpolate(
                advect_particles,
                quadrature=pic,
                values={
                    "particle_flags": model.particle_flags,
                    "pos": state_out.particle_q,
                    "pos_prev": state_in.particle_q,
                    "vel": state_out.particle_qd,
                    "vel_grad": state_out.particle_qd_grad,
                    "dt": dt,
                    "max_vel": model.particle_max_velocity,
                },
                fields={
                    "grid_vel": state_out.velocity_field,
                },
            )

    def _warmstart_fields(self, scratch: _ImplicitMPMScratchpad, state_in: newton.State, state_out: newton.State):
        """Interpolate previous grid fields into the current grid layout.

        Transfers impulse and stress fields from the previous grid to the new
        grid (handling nonconforming cases), and initializes the output state's
        grid fields to the current scratchpad fields.
        """
        domain = scratch.velocity_test.domain

        if state_in.impulse_field is None:
            state_in.impulse_field = scratch.impulse_field
            state_in.stress_field = scratch.stress_field
            state_in.velocity_field = scratch.velocity_field
            state_in.collider_ids = scratch.collider_ids

        if (
            state_out.velocity_field is None
            or state_out.velocity_field.space_partition != scratch.velocity_field.space_partition
        ):
            state_out.impulse_field = scratch.impulse_field
            state_out.stress_field = scratch.stress_field
            state_out.velocity_field = scratch.velocity_field
            state_out.collider_ids = scratch.collider_ids

        # Interpolate previous impulse
        if state_in.impulse_field is not state_out.impulse_field:
            prev_impulse_field = (
                state_in.impulse_field
                if self.grid_type == "fixed"
                else fem.NonconformingField(domain, state_in.impulse_field, background=scratch.background_impulse_field)
            )
            fem.interpolate(
                prev_impulse_field,
                dest=fem.make_restriction(
                    scratch.impulse_field, space_restriction=scratch.velocity_test.space_restriction
                ),
            )

        # Interpolate previous stress
        if state_in.stress_field is not state_out.stress_field:
            prev_stress_field = (
                state_in.stress_field
                if self.grid_type == "fixed"
                else fem.NonconformingField(domain, state_in.stress_field, background=scratch.background_stress_field)
            )
            fem.interpolate(
                prev_stress_field,
                dest=fem.make_restriction(
                    scratch.stress_field, space_restriction=scratch.sym_strain_test.space_restriction
                ),
            )

    def _compute_coloring(
        self,
        scratch: _ImplicitMPMScratchpad,
    ):
        """Compute Gauss-Seidel coloring of strain nodes to avoid write conflicts.

        Writes scratch.color_offsets, scratch.color_indices and scratch.color_nodes_per_element.
        """

        space_partition = scratch._strain_space_restriction.space_partition
        grid = space_partition.geo_partition.geometry

        nodes_per_element = space_partition.space_topology.MAX_NODES_PER_ELEMENT
        is_dg = space_partition.geo_partition.cell_count() * nodes_per_element == space_partition.node_count()
        strain_node_count = space_partition.node_count()

        if is_dg:
            # nodes in each element solved sequentially
            colored_element_count = strain_node_count // nodes_per_element
        else:
            # color each node individually
            colored_element_count = strain_node_count
            nodes_per_element = 1

        colors = fem.borrow_temporary(self.temporary_store, shape=colored_element_count * 2, dtype=int)
        color_indices = scratch.color_indices

        partition_arg = space_partition.partition_arg_value(colors.array.device)
        if is_dg:

            @fem.cache.dynamic_kernel(suffix=space_partition.name)
            def node_color_8_stencil(
                voxels: wp.array(dtype=wp.vec3i),
                colors: wp.array(dtype=int),
                color_indices: wp.array(dtype=int),
                nodes_per_element: int,
                partition_arg: space_partition.PartitionArg,
            ):
                pid = wp.tid()
                vid = space_partition.partition_node_index(partition_arg, pid * nodes_per_element) // nodes_per_element

                c = voxels[vid]
                colors[pid] = ((c[0] & 1) << 2) + ((c[1] & 1) << 1) + (c[2] & 1)
                color_indices[pid] = vid

            if isinstance(grid, fem.Nanogrid):
                voxels = grid._cell_ijk
            else:
                voxels = wp.array(
                    np.stack(
                        np.meshgrid(
                            np.arange(grid.res[0]),
                            np.arange(grid.res[1]),
                            np.arange(grid.res[2]),
                            indexing="ij",
                        ),
                        axis=-1,
                    ).reshape(-1, 3),
                    dtype=wp.vec3i,
                )
            wp.launch(
                node_color_8_stencil,
                dim=colored_element_count,
                inputs=[voxels, colors.array, color_indices.array, nodes_per_element, partition_arg],
            )
            max_color_count = 8

        elif self.strain_basis == "Q1":

            @fem.cache.dynamic_kernel(suffix=space_partition.name)
            def node_color_27_stencil(
                voxels: wp.array(dtype=wp.vec3i),
                colors: wp.array(dtype=int),
                color_indices: wp.array(dtype=int),
                partition_arg: space_partition.PartitionArg,
            ):
                pid = wp.tid()
                vid = space_partition.partition_node_index(partition_arg, pid)

                c = voxels[vid]
                colors[pid] = positive_mod3(c[0]) * 9 + positive_mod3(c[1]) * 3 + positive_mod3(c[2])
                color_indices[pid] = pid

            if isinstance(grid, fem.Nanogrid):
                voxels = grid._node_ijk
            else:
                voxels = wp.array(
                    np.stack(
                        np.meshgrid(
                            np.arange(grid.res[0] + 1),
                            np.arange(grid.res[1] + 1),
                            np.arange(grid.res[2] + 1),
                            indexing="ij",
                        ),
                        axis=-1,
                    ).reshape(-1, 3),
                    dtype=wp.vec3i,
                )
            wp.launch(
                node_color_27_stencil,
                dim=colored_element_count,
                inputs=[voxels, colors.array, color_indices.array, partition_arg],
            )
            max_color_count = 27
        else:
            raise RuntimeError("Unsupported strain basis for coloring")

        max_color_count = min(max_color_count, colored_element_count)

        color_counts_host = fem.borrow_temporary(
            self.temporary_store, shape=max_color_count + 1, dtype=int, device="cpu", pinned=colors.array.device.is_cuda
        )

        wp.utils.radix_sort_pairs(
            keys=colors.array,
            values=color_indices.array,
            count=colored_element_count,
        )

        unique_colors = colors.array[colored_element_count:]
        color_node_counts = color_indices.array[colored_element_count:]

        wp.utils.runlength_encode(
            colors.array,
            run_values=unique_colors,
            run_lengths=color_node_counts,
            value_count=colored_element_count,
            run_count=color_node_counts[max_color_count : max_color_count + 1],
        )

        wp.copy(src=color_node_counts[: max_color_count + 1], dest=color_counts_host.array)

        color_counts_np = color_counts_host.array.numpy()

        color_count = color_counts_np[-1]

        color_offsets = np.concatenate([[0], np.cumsum(color_counts_np[:color_count])])

        colors.release()
        color_counts_host.release()

        scratch.color_offsets = color_offsets
        scratch.color_nodes_per_element = nodes_per_element
