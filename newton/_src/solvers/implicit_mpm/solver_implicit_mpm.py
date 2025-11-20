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
from typing import Any

import numpy as np
import warp as wp
import warp.fem as fem
import warp.sparse as sp
from warp.context import assert_conditional_graph_support

import newton
import newton.utils

from ...core.types import override
from ..solver import SolverBase
from .rasterized_collisions import (
    Collider,
    allot_collider_mass,
    build_rigidity_matrix,
    interpolate_collider_normals,
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

_DEFAULT_PROJECTION_THRESHOLD = 0.01
"""Default threshold for projection outside of collider, as a fraction of the voxel size"""

_DEFAULT_THICKNESS = 0.01
"""Default thickness for colliders, as a fraction of the voxel size"""
_DEFAULT_FRICTION = 0.5
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
    min_sdf = float(_INFINITY)
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


@wp.kernel
def average_elastic_parameters(
    elastic_parameters_int: wp.array(dtype=wp.vec3),
    particle_volume: wp.array(dtype=float),
    elastic_parameters_avg: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    pvol = particle_volume[i]
    elastic_parameters_avg[i] = elastic_parameters_int[i] / wp.max(pvol, _EPSILON)


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


def _make_grid_basis_space(grid: fem.Geometry, basis_str: str, family: fem.Polynomial | None = None):
    assert len(basis_str) >= 2

    degree = int(basis_str[1])

    if basis_str[0] == "Q":
        element_basis = fem.ElementBasis.LAGRANGE
    elif basis_str[0] == "S":
        element_basis = fem.ElementBasis.SERENDIPITY
    elif basis_str[0] == "P" and (degree == 0 or basis_str[-1] == "d"):
        element_basis = fem.ElementBasis.NONCONFORMING_POLYNOMIAL
    else:
        raise ValueError(f"Unsupported basis: {basis_str}")

    return fem.make_polynomial_basis_space(grid, degree=degree, element_basis=element_basis, family=family)


def _make_pic_basis_space(pic: fem.PicQuadrature, basis_str: str):
    try:
        max_points_per_cell = int(basis_str[3:])
    except ValueError:
        max_points_per_cell = -1

    return fem.PointBasisSpace(pic, max_nodes_per_element=max_points_per_cell)


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
    strain_basis: str = "P0"
    """Strain basis functions. May be one of P0, Q1"""
    solver: str = "gauss-seidel"
    """Solver to use for the rheology solver. May be one of gauss-seidel, jacobi."""

    # grid
    grid_type: str = "sparse"
    """Type of grid to use. May be one of sparse, dense, fixed."""
    grid_padding: int = 0
    """Number of empty cells to add around particles when allocating the grid."""
    max_active_cell_count: int = -1
    """Maximum number of active cells to use for active subsets of dense grids. -1 means unlimited."""
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

    # experimental
    collider_normal_from_sdf_gradient: bool = False
    """Compute collider normals from sdf gradient rather than closest point"""
    collider_basis: str = "Q1"
    """Collider basis function string. Examples: P0 (piecewise constant), Q1 (trilinear), S2 (quadratic serendipity), pic8 (particle-based with max 8 points per cell)"""


class _ImplicitMPMScratchpad:
    """Per-step spaces, fields, and temporaries for the implicit MPM solver."""

    def __init__(self):
        self.grid = None

        self.velocity_test = None
        self.velocity_trial = None
        self.fraction_test = None

        self.sym_strain_test = None
        self.sym_strain_trial = None
        self.divergence_test = None
        self.fraction_field = None
        self.elastic_parameters_field = None

        self.plastic_strain_delta_field = None
        self.elastic_strain_delta_field = None
        self.strain_yield_parameters_field = None
        self.strain_yield_parameters_test = None

        self.strain_matrix = sp.bsr_zeros(0, 0, mat63)
        self.transposed_strain_matrix = sp.bsr_zeros(0, 0, mat36)

        self.compliance_matrix = sp.bsr_zeros(0, 0, mat66)

        self.color_offsets = None
        self.color_indices = None
        self.color_nodes_per_element = 1

        self.inv_mass_matrix = None

        self.collider_fraction_test = None

        self.collider_normal_field = None
        self.collider_distance_field = None

        self.collider_velocity = None
        self.collider_friction = None
        self.collider_adhesion = None
        self.collider_inv_mass_matrix = None

        self.collider_matrix = sp.bsr_zeros(0, 0, block_type=float)
        self.transposed_collider_matrix = sp.bsr_zeros(0, 0, block_type=float)

        self.strain_node_particle_volume = None
        self.strain_node_volume = None
        self.strain_node_collider_volume = None

        self.int_symmetric_strain = None

        self.collider_total_volumes = None
        self.collider_node_volume = None

    def rebuild_function_spaces(
        self,
        pic: fem.PicQuadrature,
        strain_basis_str: str,
        collider_basis_str: str,
        max_cell_count: int,
        temporary_store: fem.TemporaryStore,
    ):
        """Define velocity and strain function spaces over the given geometry."""

        self.domain = pic.domain

        use_pic_collider_basis = collider_basis_str[:3] == "pic"

        if self.domain.geometry is not self.grid:
            self.grid = self.domain.geometry

            # Define function spaces: linear (Q1) for velocity and volume fraction,
            # zero or first order for pressure
            self._velocity_basis = fem.make_polynomial_basis_space(self.grid, degree=1)

            self._strain_basis = _make_grid_basis_space(self.grid, strain_basis_str)

            if not use_pic_collider_basis:
                self._collision_basis = _make_grid_basis_space(
                    self.grid, collider_basis_str, family=fem.Polynomial.EQUISPACED_CLOSED
                )

        # Point-based basis space need to be rebuilt evemn when the geo does not change
        if use_pic_collider_basis:
            self._collision_basis = _make_pic_basis_space(pic, collider_basis_str)

        self._create_velocity_function_space(temporary_store, max_cell_count)
        self._create_collider_function_space(temporary_store, max_cell_count)
        self._create_strain_function_space(temporary_store, max_cell_count)

    def _create_velocity_function_space(self, temporary_store: fem.TemporaryStore, max_cell_count: int = -1):
        """Create velocity and fraction spaces and their partition/restriction."""
        domain = self.domain

        velocity_space = fem.make_collocated_function_space(self._velocity_basis, dtype=wp.vec3)

        # overly conservative
        max_vel_node_count = (
            velocity_space.topology.MAX_NODES_PER_ELEMENT * max_cell_count if max_cell_count >= 0 else -1
        )

        vel_space_partition = fem.make_space_partition(
            space_topology=velocity_space.topology,
            geometry_partition=domain.geometry_partition,
            with_halo=False,
            max_node_count=max_vel_node_count,
            temporary_store=temporary_store,
        )
        vel_space_restriction = fem.make_space_restriction(
            space_partition=vel_space_partition, domain=domain, temporary_store=temporary_store
        )

        self._velocity_space = velocity_space
        self._vel_space_restriction = vel_space_restriction

    def _create_collider_function_space(self, temporary_store: fem.TemporaryStore, max_cell_count: int = -1):
        """Create velocity and fraction spaces and their partition/restriction."""

        if self._velocity_basis == self._collision_basis:
            self._collision_space = self._velocity_space
            self._collision_space_restriction = self._vel_space_restriction
            return

        domain = self.domain

        collision_space = fem.make_collocated_function_space(self._collision_basis, dtype=wp.vec3)

        if isinstance(collision_space.basis, fem.PointBasisSpace):
            max_collision_node_count = collision_space.node_count()
        else:
            # overly conservative
            max_collision_node_count = (
                collision_space.topology.MAX_NODES_PER_ELEMENT * domain.element_count() if max_cell_count >= 0 else -1
            )

        collision_space_partition = fem.make_space_partition(
            space_topology=collision_space.topology,
            geometry_partition=domain.geometry_partition,
            with_halo=False,
            max_node_count=max_collision_node_count,
            temporary_store=temporary_store,
        )
        collision_space_restriction = fem.make_space_restriction(
            space_partition=collision_space_partition, domain=domain, temporary_store=temporary_store
        )

        self._collision_space = collision_space
        self._collision_space_restriction = collision_space_restriction

    def _create_strain_function_space(self, temporary_store: fem.TemporaryStore, max_cell_count: int = -1):
        """Create symmetric strain space (P0 or Q1) and its partition/restriction."""
        domain = self.domain

        sym_strain_space = fem.make_collocated_function_space(
            self._strain_basis,
            dof_mapper=fem.SymmetricTensorMapper(dtype=wp.mat33, mapping=fem.SymmetricTensorMapper.Mapping.DB16),
        )

        max_strain_node_count = (
            sym_strain_space.topology.MAX_NODES_PER_ELEMENT * max_cell_count if max_cell_count >= 0 else -1
        )

        strain_space_partition = fem.make_space_partition(
            space_topology=sym_strain_space.topology,
            geometry_partition=domain.geometry_partition,
            with_halo=False,
            max_node_count=max_strain_node_count,
            temporary_store=temporary_store,
        )

        strain_space_restriction = fem.make_space_restriction(
            space_partition=strain_space_partition, domain=domain, temporary_store=temporary_store
        )

        self._sym_strain_space = sym_strain_space
        self._strain_space_restriction = strain_space_restriction

    def require_velocity_space_fields(self, has_compliant_particles: bool):
        velocity_basis = self._velocity_basis
        velocity_space = self._velocity_space
        vel_space_restriction = self._vel_space_restriction
        domain = vel_space_restriction.domain
        vel_space_partition = vel_space_restriction.space_partition

        if (
            self.velocity_test is not None
            and self.velocity_test.space_restriction.space_partition == vel_space_partition
        ):
            return

        fraction_space = fem.make_collocated_function_space(velocity_basis, dtype=float)

        # test, trial and discrete fields
        if self.velocity_test is None:
            self.velocity_test = fem.make_test(velocity_space, domain=domain, space_restriction=vel_space_restriction)
            self.fraction_test = fem.make_test(fraction_space, space_restriction=vel_space_restriction)

            self.velocity_trial = fem.make_trial(velocity_space, domain=domain, space_partition=vel_space_partition)
            self.fraction_trial = fem.make_trial(fraction_space, domain=domain, space_partition=vel_space_partition)

            self.fraction_field = fem.make_discrete_field(fraction_space, space_partition=vel_space_partition)

            if has_compliant_particles:
                elastic_parameters_space = fem.make_collocated_function_space(velocity_basis, dtype=wp.vec3)
                self.elastic_parameters_field = elastic_parameters_space.make_field(space_partition=vel_space_partition)

        else:
            self.velocity_test.rebind(velocity_space, vel_space_restriction)
            self.fraction_test.rebind(fraction_space, vel_space_restriction)

            self.velocity_trial.rebind(velocity_space, vel_space_partition, domain)
            self.fraction_trial.rebind(fraction_space, vel_space_partition, domain)
            self.fraction_field.rebind(fraction_space, vel_space_partition)

            if has_compliant_particles:
                elastic_parameters_space = fem.make_collocated_function_space(velocity_basis, dtype=wp.vec3)
                self.elastic_parameters_field.rebind(elastic_parameters_space, vel_space_partition)

        self.velocity_field = velocity_space.make_field(space_partition=vel_space_partition)

    def require_collision_space_fields(self):
        collision_basis = self._collision_basis
        collision_space = self._collision_space
        collision_space_restriction = self._collision_space_restriction
        domain = collision_space_restriction.domain
        collision_space_partition = collision_space_restriction.space_partition

        if (
            self.collider_fraction_test is not None
            and self.collider_fraction_test.space_restriction.space_partition == collision_space_partition
        ):
            return
        collider_fraction_space = fem.make_collocated_function_space(collision_basis, dtype=float)

        # test, trial and discrete fields
        if self.collider_fraction_test is None:
            self.collider_fraction_test = fem.make_test(
                collider_fraction_space, space_restriction=collision_space_restriction
            )
            self.collider_distance_field = collider_fraction_space.make_field(space_partition=collision_space_partition)

            self.collider_velocity_field = collision_space.make_field(space_partition=collision_space_partition)
            self.collider_normal_field = collision_space.make_field(space_partition=collision_space_partition)

            self.background_impulse_field = fem.UniformField(domain, wp.vec3(0.0))
        else:
            self.collider_fraction_test.rebind(collider_fraction_space, collision_space_restriction)
            self.collider_distance_field.rebind(collider_fraction_space, collision_space_partition)

            self.collider_velocity_field.rebind(collision_space, collision_space_partition)
            self.collider_normal_field.rebind(collision_space, collision_space_partition)

            self.background_impulse_field.domain = domain

        self.impulse_field = collision_space.make_field(space_partition=collision_space_partition)
        self.collider_position_field = collision_space.make_field(space_partition=collision_space_partition)
        self.collider_ids = wp.empty(collision_space_partition.node_count(), dtype=int)

    def require_strain_space_fields(self):
        """Ensure strain-space fields exist and match current spaces."""
        strain_basis = self._strain_basis
        sym_strain_space = self._sym_strain_space
        strain_space_restriction = self._strain_space_restriction
        domain = strain_space_restriction.domain
        strain_space_partition = strain_space_restriction.space_partition

        if (
            self.sym_strain_test is not None
            and self.sym_strain_test.space_restriction.space_partition == strain_space_partition
        ):
            return

        divergence_space = fem.make_collocated_function_space(strain_basis, dtype=float)
        strain_yield_parameters_space = fem.make_collocated_function_space(strain_basis, dtype=YieldParamVec)

        if self.sym_strain_test is None:
            self.sym_strain_test = fem.make_test(sym_strain_space, space_restriction=strain_space_restriction)
            self.divergence_test = fem.make_test(divergence_space, space_restriction=strain_space_restriction)
            self.strain_yield_parameters_test = fem.make_test(
                strain_yield_parameters_space, space_restriction=strain_space_restriction
            )
            self.sym_strain_trial = fem.make_trial(
                sym_strain_space, domain=domain, space_partition=strain_space_partition
            )

            self.elastic_strain_delta_field = sym_strain_space.make_field(space_partition=strain_space_partition)
            self.plastic_strain_delta_field = sym_strain_space.make_field(space_partition=strain_space_partition)
            self.strain_yield_parameters_field = strain_yield_parameters_space.make_field(
                space_partition=strain_space_partition
            )

            self.background_stress_field = fem.UniformField(domain, wp.mat33(0.0))
        else:
            self.sym_strain_test.rebind(sym_strain_space, strain_space_restriction)
            self.divergence_test.rebind(divergence_space, strain_space_restriction)
            self.strain_yield_parameters_test.rebind(strain_yield_parameters_space, strain_space_restriction)

            self.sym_strain_trial.rebind(sym_strain_space, strain_space_partition, domain)

            self.elastic_strain_delta_field.rebind(sym_strain_space, strain_space_partition)
            self.plastic_strain_delta_field.rebind(sym_strain_space, strain_space_partition)
            self.strain_yield_parameters_field.rebind(strain_yield_parameters_space, strain_space_partition)

            self.background_stress_field.domain = domain

        self.stress_field = sym_strain_space.make_field(space_partition=strain_space_partition)

    @property
    def collider_node_count(self) -> int:
        return self._collision_space_restriction.space_partition.node_count()

    @property
    def velocity_node_count(self) -> int:
        return self._vel_space_restriction.space_partition.node_count()

    @property
    def strain_node_count(self) -> int:
        return self._strain_space_restriction.space_partition.node_count()

    def allocate_temporaries(
        self,
        collider_count: int,
        has_compliant_bodies: bool,
        has_critical_fraction: bool,
        max_colors: int,
        temporary_store: fem.TemporaryStore,
    ):
        """Allocate transient arrays sized to current grid and options."""
        vel_node_count = self.velocity_node_count
        collider_node_count = self.collider_node_count
        strain_node_count = self.strain_node_count

        self.inv_mass_matrix = fem.borrow_temporary(temporary_store, shape=(vel_node_count,), dtype=float)

        self.collider_velocity = fem.borrow_temporary(temporary_store, shape=(collider_node_count,), dtype=wp.vec3)
        self.collider_friction = fem.borrow_temporary(temporary_store, shape=(collider_node_count,), dtype=float)
        self.collider_adhesion = fem.borrow_temporary(temporary_store, shape=(collider_node_count,), dtype=float)
        self.collider_inv_mass_matrix = fem.borrow_temporary(temporary_store, shape=(collider_node_count,), dtype=float)
        self.collider_node_volume = fem.borrow_temporary(temporary_store, shape=collider_node_count, dtype=float)

        self.strain_node_particle_volume = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)
        self.int_symmetric_strain = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=vec6)
        self.unilateral_strain_offset = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)

        sp.bsr_set_zero(self.strain_matrix, rows_of_blocks=strain_node_count, cols_of_blocks=vel_node_count)
        sp.bsr_set_zero(self.compliance_matrix, rows_of_blocks=strain_node_count, cols_of_blocks=strain_node_count)

        if has_critical_fraction:
            self.strain_node_volume = fem.borrow_temporary(temporary_store, shape=strain_node_count, dtype=float)
            self.strain_node_collider_volume = fem.borrow_temporary(
                temporary_store, shape=strain_node_count, dtype=float
            )

        if has_compliant_bodies:
            self.collider_total_volumes = fem.borrow_temporary(temporary_store, shape=collider_count, dtype=float)

        if max_colors > 0:
            self.color_indices = fem.borrow_temporary(temporary_store, shape=strain_node_count * 2, dtype=int)
            self.color_offsets = fem.borrow_temporary(temporary_store, shape=max_colors + 1, dtype=int)

    def release_temporaries(self):
        """Release previously allocated temporaries to the store."""
        self.inv_mass_matrix.release()
        self.collider_velocity.release()
        self.collider_friction.release()
        self.collider_adhesion.release()
        self.collider_inv_mass_matrix.release()
        self.collider_node_volume.release()
        self.int_symmetric_strain.release()
        self.strain_node_particle_volume.release()
        self.unilateral_strain_offset.release()

        if self.strain_node_volume is not None:
            self.strain_node_volume.release()
            self.strain_node_collider_volume.release()

        if self.collider_total_volumes is not None:
            self.collider_total_volumes.release()

        if self.color_indices is not None:
            self.color_indices.release()
            self.color_offsets.release()


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


def _merge_meshes(
    points: list[np.array] = (),
    indices: list[np.array] = (),
    shape_ids: np.array = (),
    material_ids: np.array = (),
) -> tuple[wp.array, wp.array, wp.array, np.array]:
    """Merges the points and indices of several meshes into a single one"""

    pt_count = np.array([len(pts) for pts in points])
    face_count = np.array([len(idx) // 3 for idx in indices])
    offsets = np.cumsum(pt_count) - pt_count

    merged_points = np.vstack([pts[:, :3] for pts in points])
    merged_indices = np.concatenate([idx + offsets[k] for k, idx in enumerate(indices)])
    vertex_shape_ids = np.repeat(np.arange(len(points), dtype=int), repeats=pt_count)
    face_shape_ids = np.repeat(np.arange(len(points), dtype=int), repeats=face_count)

    return (
        wp.array(merged_points, dtype=wp.vec3),
        wp.array(merged_indices, dtype=int),
        wp.array(shape_ids[vertex_shape_ids], dtype=int),
        np.array(material_ids, dtype=int)[face_shape_ids],
    )


def _get_shape_mesh(model: newton.Model, shape_id: int, geo_type: newton.GeoType, geo_scale: wp.vec3):
    """Get a shape mesh from a model."""

    if geo_type == newton.GeoType.MESH:
        src_mesh = model.shape_source[shape_id]
        vertices = src_mesh.vertices * np.array(geo_scale)
        indices = src_mesh.indices
        return vertices, indices
    if geo_type == newton.GeoType.PLANE:
        # Handle "infinite" planes encoded with non-positive scales
        width = geo_scale[0] if len(geo_scale) > 0 and geo_scale[0] > 0.0 else 1000.0
        length = geo_scale[1] if len(geo_scale) > 1 and geo_scale[1] > 0.0 else 1000.0
        return newton.utils.create_plane_mesh(width, length)
    elif geo_type == newton.GeoType.SPHERE:
        radius = geo_scale[0]
        return newton.utils.create_sphere_mesh(radius)

    elif geo_type == newton.GeoType.CAPSULE:
        radius, half_height = geo_scale[:2]
        return newton.utils.create_capsule_mesh(radius, half_height, up_axis=2)

    elif geo_type == newton.GeoType.CYLINDER:
        radius, half_height = geo_scale[:2]
        return newton.utils.create_cylinder_mesh(radius, half_height, up_axis=2)

    elif geo_type == newton.GeoType.CONE:
        radius, half_height = geo_scale[:2]
        return newton.utils.create_cone_mesh(radius, half_height, up_axis=2)

    elif geo_type == newton.GeoType.BOX:
        if len(geo_scale) == 1:
            ext = (geo_scale[0],) * 3
        else:
            ext = tuple(geo_scale[:3])
        return newton.utils.create_box_mesh(ext)

    raise NotImplementedError(f"Shape type {geo_type} not supported")


@wp.kernel
def _apply_shape_transforms(
    points: wp.array(dtype=wp.vec3), shape_ids: wp.array(dtype=int), shape_transforms: wp.array(dtype=wp.transform)
):
    v = wp.tid()
    p = points[v]
    shape_id = shape_ids[v]
    shape_transform = shape_transforms[shape_id]
    p = wp.transform_point(shape_transform, p)
    points[v] = p


def _get_body_collision_shapes(model: newton.Model, body_index: int):
    """Returns the ids of the shapes of a body with active collision flags."""

    shape_flags = model.shape_flags.numpy()
    body_shape_ids = np.array(model.body_shapes[body_index], dtype=int)

    return body_shape_ids[(shape_flags[body_shape_ids] & newton.ShapeFlags.COLLIDE_PARTICLES) > 0]


def _get_shape_collision_materials(model: newton.Model, shape_ids: list[int]):
    """Returns the collision materials from the model for a list of shapes"""
    thicknesses = model.shape_thickness.numpy()[shape_ids]
    friction = model.shape_material_mu.numpy()[shape_ids]

    return thicknesses, friction


def _create_body_collider_mesh(
    model: newton.Model,
    shape_ids: list[int],
    material_ids: list[int],
):
    """Create a collider mesh from a body."""

    shape_scale = model.shape_scale.numpy()
    shape_type = model.shape_type.numpy()

    shape_meshes = [_get_shape_mesh(model, sid, newton.GeoType(shape_type[sid]), shape_scale[sid]) for sid in shape_ids]

    collider_points, collider_indices, vertex_shape_ids, face_material_ids = _merge_meshes(
        *zip(*shape_meshes, strict=True),
        shape_ids=shape_ids,
        material_ids=material_ids,
    )

    wp.launch(
        _apply_shape_transforms,
        dim=collider_points.shape[0],
        inputs=[
            collider_points,
            vertex_shape_ids,
            model.shape_transform,
        ],
    )

    return wp.Mesh(collider_points, collider_indices, wp.zeros_like(collider_points)), face_material_ids


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

        self.collider_body_mass = None
        self.collider_body_inv_inertia = None

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
        body_ids = self.collider.collider_body_index.numpy()
        body_mass = self.collider_body_mass.numpy()
        dynamic_body_ids = body_ids[body_ids >= 0]
        dynamic_body_ids = dynamic_body_ids[body_mass[dynamic_body_ids] > 0.0]
        dynamic_body_masses = body_mass[dynamic_body_ids]

        self.min_collider_mass = np.min(dynamic_body_masses, initial=np.inf)
        self.collider.query_max_dist = self.voxel_size * math.sqrt(3.0)
        self.collider_body_count = int(np.max(body_ids + 1, initial=0))

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
        collider_meshes: list[wp.Mesh] | None = None,
        collider_body_ids: list[int] | None = None,
        collider_thicknesses: list[float] | None = None,
        collider_friction: list[float] | None = None,
        collider_adhesion: list[float] | None = None,
        collider_projection_threshold: list[float] | None = None,
        ground_height: float = -_INFINITY,
        ground_normal: wp.vec3 | None = None,
        model: newton.Model | None = None,
        body_com: wp.array | None = None,
        body_mass: wp.array | None = None,
        body_inv_inertia: wp.array | None = None,
    ):
        """Initialize collider parameters and defaults from inputs.

        Populates the ``Collider`` struct with meshes, body mapping, and per-material
        properties (thickness, friction, adhesion, projection threshold).

        By default, this will setup collisions against all collision shapes in the model with flag `newton.ShapeFlag.COLLIDE_PARTICLES`.
        Rigid body colliders will be treated as kinematic if their mass is zero; for all model bodies to be treated as kinematic,
        pass ``body_mass=wp.zeros_like(model.body_mass)``.

        For any collider index `i`, only one of ``collider_meshes[i]`` and ``collider_body_ids`` may not be `None`.
        If material properties are not provided for a collider, but a body index is provided,
        the material will be read from the body shape material attributes on the model.

        Args:
            collider_meshes: Warp triangular meshes used as colliders.
            collider_body_ids: For dynamic colliders, per-mesh body ids.
            collider_thicknesses: Per-mesh signed distance offsets (m).
            collider_friction: Per-mesh Coulomb friction coefficients.
            collider_adhesion: Per-mesh adhesion (Pa).
            collider_projection_threshold: Per-mesh projection threshold, i.e. how far below the surface the
              particle may be before it is projected out. (m)
            ground_height: Height of the ground plane.
            ground_normal: Normal of the ground plane (default to model.up_axis).
            model: The model to read collider properties from. Default to self.model.
            body_com: For dynamic colliders, per-mesh body center of mass. Default to model.body_com.
            body_mass: For dynamic colliders, per-mesh body mass. Default to model.body_mass.
            body_inv_inertia: For dynamic colliders, per-mesh body inverse inertia. Default to model.body_inv_inertia.
        """

        if model is None:
            model = self.model

        if collider_body_ids is None:
            if collider_meshes is None:
                collider_body_ids = [
                    body_id
                    for body_id in range(-1, model.body_count)
                    if len(_get_body_collision_shapes(model, body_id)) > 0
                ]
            else:
                collider_body_ids = [None] * len(collider_meshes)
        if collider_meshes is None:
            collider_meshes = [None] * len(collider_body_ids)

        for collider_id, (mesh, body_id) in enumerate(zip(collider_meshes, collider_body_ids, strict=True)):
            if mesh is None:
                if body_id is None:
                    raise ValueError(
                        f"Either a mesh or a body_id must be provided for each collider; collider {collider_id} is missing both"
                    )
            elif body_id is not None:
                raise ValueError(
                    f"Either a mesh or a body_id must be provided for each collider; collider {collider_id} provides both"
                )

        collider_count = len(collider_body_ids)

        if collider_thicknesses is None:
            collider_thicknesses = [None] * collider_count
        if collider_projection_threshold is None:
            collider_projection_threshold = [None] * collider_count
        if collider_friction is None:
            collider_friction = [None] * collider_count
        if collider_adhesion is None:
            collider_adhesion = [None] * collider_count

        assert len(collider_body_ids) == len(collider_thicknesses)
        assert len(collider_body_ids) == len(collider_projection_threshold)
        assert len(collider_body_ids) == len(collider_friction)
        assert len(collider_body_ids) == len(collider_adhesion)

        if body_com is None:
            body_com = model.body_com
        if body_mass is None:
            body_mass = model.body_mass
        if body_inv_inertia is None:
            body_inv_inertia = model.body_inv_inertia

        # count materials and shapes
        material_count = 1  # ground material
        body_shapes = {}
        collider_material_ids = []
        for body_id in collider_body_ids:
            if body_id is not None:
                shapes = _get_body_collision_shapes(model, body_id)
                if len(shapes) == 0:
                    raise ValueError(f"Body {body_id} has no collision shapes")

                body_shapes[body_id] = shapes
                collider_material_ids.append(list(range(material_count, material_count + len(shapes))))
                material_count += len(shapes)
            else:
                collider_material_ids.append([material_count])
                material_count += 1

        # assign material values
        material_thickness = [_DEFAULT_THICKNESS * self.voxel_size] * material_count
        material_friction = [_DEFAULT_FRICTION] * material_count
        material_adhesion = [_DEFAULT_ADHESION] * material_count
        material_projection_threshold = [_DEFAULT_PROJECTION_THRESHOLD * self.voxel_size] * material_count

        def assign_material(
            material_id: int,
            thickness: float | None = None,
            friction: float | None = None,
            adhesion: float | None = None,
            projection_threshold: float | None = None,
        ):
            if thickness is not None:
                material_thickness[material_id] = thickness
            if friction is not None:
                material_friction[material_id] = friction
            if adhesion is not None:
                material_adhesion[material_id] = adhesion
            if projection_threshold is not None:
                material_projection_threshold[material_id] = projection_threshold

        def assign_collider_material(material_id: int, collider_id: int):
            assign_material(
                material_id,
                collider_thicknesses[collider_id],
                collider_friction[collider_id],
                collider_adhesion[collider_id],
                collider_projection_threshold[collider_id],
            )

        for collider_id, body_id in enumerate(collider_body_ids):
            if body_id is not None:
                for material_id, shape_thickness, shape_friction in zip(
                    collider_material_ids[collider_id],
                    *_get_shape_collision_materials(model, body_shapes[body_id]),
                    strict=True,
                ):
                    # use material from shapes as default
                    assign_material(material_id, thickness=shape_thickness, friction=shape_friction)
                    # override with user-provided material
                    assign_collider_material(material_id, collider_id)
            else:
                # user-provided collider, single material
                assign_collider_material(collider_material_ids[collider_id][0], collider_id)

        collider_max_thickness = [
            max((material_thickness[material_id] for material_id in collider_material_ids[collider_id]), default=0.0)
            for collider_id in range(collider_count)
        ]

        # Create device arrays
        with wp.ScopedDevice(self.model.device):
            # Create collider meshes from bodies if necessary
            face_material_ids = [[]]
            for collider_id in range(collider_count):
                body_index = collider_body_ids[collider_id]

                if body_index is None:
                    # Set body index to -1 to indicate a static collider
                    # This may not correspond to the model's body -1, but as far as the collision kernels
                    # are concerned, it does not matter.

                    collider_body_ids[collider_id] = -1
                    material_id = collider_material_ids[collider_id][0]
                    face_count = collider_meshes[collider_id].indices.shape[0] // 3
                    mesh_face_material_ids = np.full(face_count, material_id, dtype=int)
                else:
                    collider_meshes[collider_id], mesh_face_material_ids = _create_body_collider_mesh(
                        model, body_shapes[body_index], collider_material_ids[collider_id]
                    )

                face_material_ids.append(mesh_face_material_ids)

            self.collider.collider_body_index = wp.array(collider_body_ids, dtype=int)
            self.collider.collider_mesh = wp.array([collider.id for collider in collider_meshes], dtype=wp.uint64)
            self.collider.collider_max_thickness = wp.array(collider_max_thickness, dtype=float)

            self.collider.face_material_index = wp.array(np.concatenate(face_material_ids), dtype=int)

            self.collider.material_thickness = wp.array(material_thickness, dtype=float)
            self.collider.material_friction = wp.array(material_friction, dtype=float)
            self.collider.material_adhesion = wp.array(material_adhesion, dtype=float)
            self.collider.material_projection_threshold = wp.array(material_projection_threshold, dtype=float)

        self.collider.body_com = body_com
        self.collider_body_mass = body_mass
        self.collider_body_inv_inertia = body_inv_inertia
        self._collider_meshes = collider_meshes  # Keep a ref so that meshes are not garbage collected

        self.collider.ground_height = ground_height
        if ground_normal is None:
            self.collider.ground_normal = wp.vec3(0.0)
            self.collider.ground_normal[model.up_axis] = 1.0
        else:
            self.collider.ground_normal = wp.vec3(ground_normal)

        self.notify_collider_changed()

    @property
    def has_compliant_particles(self):
        return self.min_young_modulus < _INFINITY

    @property
    def has_hardening(self):
        return self.max_hardening > 0.0

    @property
    def has_compliant_colliders(self):
        return self.min_collider_mass < _INFINITY


@wp.kernel
def compute_bounds(
    pos: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
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
        min_x = _INFINITY
        min_y = _INFINITY
        min_z = _INFINITY
        max_x = -_INFINITY
        max_y = -_INFINITY
        max_z = -_INFINITY

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
def _positive_modn(x: int, n: int):
    return (x % n + n) % n


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


@wp.kernel
def node_color(
    space_node_indices: wp.array(dtype=int),
    nodes_per_element: int,
    stencil_size: int,
    voxels: wp.array(dtype=wp.vec3i),
    res: wp.vec3i,
    colors: wp.array(dtype=int),
    color_indices: wp.array(dtype=int),
):
    nid = wp.tid()
    vid = space_node_indices[nid * nodes_per_element] // nodes_per_element

    if voxels:
        c = voxels[vid]
    else:
        c = fem.Grid3D.get_cell(res, vid)

    colors[nid] = (
        _positive_modn(c[0], stencil_size) * stencil_size * stencil_size
        + _positive_modn(c[1], stencil_size) * stencil_size
        + _positive_modn(c[2], stencil_size)
    )
    color_indices[nid] = nid * nodes_per_element


_NULL_COLOR = 1 << 31 - 1  # color for null nodes. make sure it is sorted last


@wp.kernel
def compute_color_offsets(
    max_color_count: int,
    unique_count: wp.array(dtype=int),
    unique_colors: wp.array(dtype=int),
    color_counts: wp.array(dtype=int),
    color_offsets: wp.array(dtype=int),
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
    positions: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=int),
    active_cells: wp.array(dtype=int),
):
    if ~particle_flags[s.qp_index] & newton.ParticleFlags.ACTIVE:
        return

    x = positions[s.qp_index]
    s_grid = fem.lookup(domain, x)

    if s_grid.element_index != fem.NULL_ELEMENT_INDEX:
        active_cells[s_grid.element_index] = 1


@wp.kernel
def scatter_field_dof_values(
    space_node_indices: wp.array(dtype=int),
    src: wp.array(dtype=Any),
    dest: wp.array(dtype=Any),
):
    nid = wp.tid()

    if nid != fem.NULL_NODE_INDEX:
        dest[space_node_indices[nid]] = src[nid]


wp.overload(scatter_field_dof_values, {"src": wp.array(dtype=wp.vec3), "dest": wp.array(dtype=wp.vec3)})
wp.overload(scatter_field_dof_values, {"src": wp.array(dtype=vec6), "dest": wp.array(dtype=vec6)})


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
        self.max_active_cell_count = options.max_active_cell_count

        self.collider_normal_from_sdf_gradient = options.collider_normal_from_sdf_gradient
        self.collider_basis = options.collider_basis

        self.temporary_store = fem.TemporaryStore()

        self._use_cuda_graph = False
        if self.model.device.is_cuda:
            try:
                assert_conditional_graph_support()
                self._use_cuda_graph = True
            except Exception:
                pass

        self._enable_timers = False
        self._timers_use_nvtx = False

        self._scratchpad = None

        with wp.ScopedDevice(model.device):
            pic = self._particles_to_cells(model.particle_q)
            self._rebuild_scratchpad(pic)

    def enrich_state(self, state: newton.State):
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

        # Store a few additional fields that are necessary for warmstarting, two-way coupling (collect_collider_impulses)
        # or grain-based rendering

        state.ws_impulse_field = None
        state.ws_stress_field = None
        with wp.ScopedDevice(self.model.device):
            scratch = self._scratchpad
            self._require_velocity_space_fields(state, scratch, self.mpm_model.has_compliant_particles)
            self._require_collision_space_fields(state, scratch)
            self._require_strain_space_fields(state, scratch)

            if state.body_q is None and self.mpm_model.collider_body_count > 0:
                state.body_q = wp.zeros(self.mpm_model.collider_body_count, dtype=wp.transform, device=device)
            if state.body_qd is None and self.mpm_model.collider_body_count > 0:
                state.body_qd = wp.zeros(self.mpm_model.collider_body_count, dtype=wp.spatial_vector, device=device)

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
            pic = self._particles_to_cells(state_in.particle_q)
            scratch = self._rebuild_scratchpad(pic)
            self._step_impl(state_in, state_out, dt, pic, scratch)
            scratch.release_temporaries()

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
                state_in.body_q,
                state_in.body_qd,
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

    def collect_collider_impulses(self, state: newton.State) -> tuple[wp.array, wp.array, wp.array]:
        """Collect current collider impulses and their application positions.

        Returns a tuple of 3 arrays:
            - Impulse values in world units.
            - Collider positions in world units.
            - Collider ids.
        """

        cell_volume = self.mpm_model.voxel_size**3
        return (
            -cell_volume * state.impulse_field.dof_values,
            state.collider_position_field.dof_values,
            state.collider_ids,
        )

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
        self,
        positions: wp.array,
        particle_flags: wp.array,
        voxel_size: float,
        temporary_store: fem.TemporaryStore,
        padding_voxels: int = 0,
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
        with self._timer("Allocate grid"):
            if self.grid_type == "sparse":
                volume = _allocate_by_voxels(positions, voxel_size, padding_voxels=padding_voxels)
                grid = fem.Nanogrid(volume, temporary_store=temporary_store)
            else:
                # Compute bounds and transfer to host
                device = positions.device
                if device.is_cuda:
                    min_dev = fem.borrow_temporary(temporary_store, shape=1, dtype=wp.vec3, device=device)
                    max_dev = fem.borrow_temporary(temporary_store, shape=1, dtype=wp.vec3, device=device)

                    min_dev.fill_(wp.vec3(_INFINITY))
                    max_dev.fill_(wp.vec3(-_INFINITY))

                    tile_size = 256
                    wp.launch(
                        compute_bounds,
                        dim=((positions.shape[0] + tile_size - 1) // tile_size, tile_size),
                        block_dim=tile_size,
                        inputs=[positions, particle_flags, min_dev, max_dev],
                        device=device,
                    )

                    min_host = fem.borrow_temporary(
                        temporary_store, shape=1, dtype=wp.vec3, device="cpu", pinned=device.is_cuda
                    )
                    max_host = fem.borrow_temporary(
                        temporary_store, shape=1, dtype=wp.vec3, device="cpu", pinned=device.is_cuda
                    )
                    wp.copy(src=min_dev, dest=min_host)
                    wp.copy(src=max_dev, dest=max_host)
                    wp.synchronize_stream()
                    bbox_min, bbox_max = min_host.numpy(), max_host.numpy()
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

        return grid

    def _create_geometry_partition(
        self, grid: fem.Geometry, positions: wp.array, particle_flags: wp.array, max_cell_count: int
    ):
        """Create a geometry partition for the given positions."""

        active_cells = fem.borrow_temporary(self.temporary_store, shape=grid.cell_count(), dtype=int)
        active_cells.zero_()
        fem.interpolate(
            mark_active_cells,
            dim=positions.shape[0],
            at=fem.Cells(grid),
            values={
                "positions": positions,
                "particle_flags": particle_flags,
                "active_cells": active_cells,
            },
        )

        partition = fem.ExplicitGeometryPartition(
            grid,
            cell_mask=active_cells,
            max_cell_count=max_cell_count,
            max_side_count=0,
            temporary_store=self.temporary_store,
        )
        active_cells.release()

        return partition

    def _rebuild_scratchpad(self, pic: fem.PicQuadrature):
        """(Re)create function spaces and allocate per-step temporaries.

        Allocates the grid based on current particle positions, rebuilds
        velocity and strain spaces as needed, configures collision data, and
        optionally computes a Gauss-Seidel coloring for the strain nodes.
        """

        if self._scratchpad is None:
            self._scratchpad = _ImplicitMPMScratchpad()

        scratch = self._scratchpad

        with self._timer("Scratchpad"):
            scratch.rebuild_function_spaces(
                pic,
                strain_basis_str=self.strain_basis,
                collider_basis_str=self.collider_basis,
                max_cell_count=self.max_active_cell_count,
                temporary_store=self.temporary_store,
            )

            scratch.allocate_temporaries(
                collider_count=self.mpm_model.collider.collider_mesh.shape[0],
                has_compliant_bodies=self.mpm_model.has_compliant_colliders,
                has_critical_fraction=self.mpm_model.critical_fraction > 0.0,
                max_colors=self._max_colors(),
                temporary_store=self.temporary_store,
            )

            if self.coloring:
                self._compute_coloring(scratch=scratch)

        return scratch

    def _particles_to_cells(self, positions: wp.array) -> fem.PicQuadrature:
        """
        Rebuild the grid and grid partition around particles if required,
        then assign particles to grid cells.
        """

        # Rebuild grid

        if self._scratchpad is not None and self.grid_type == "fixed":
            grid = self._scratchpad.grid
        else:
            grid = self._allocate_grid(
                positions,
                self.model.particle_flags,
                voxel_size=self.mpm_model.voxel_size,
                temporary_store=self.temporary_store,
                padding_voxels=self.grid_padding,
            )

        # Build active partition
        with self._timer("Build active partition"):
            if self.grid_type == "sparse":
                max_cell_count = -1
                geo_partition = grid
            else:
                max_cell_count = self.max_active_cell_count
                geo_partition = self._create_geometry_partition(
                    grid, positions, self.model.particle_flags, max_cell_count
                )

        # Bin particles to grid cells
        with self._timer("Bin particles"):
            domain = fem.Cells(geo_partition)
            pic = fem.PicQuadrature(
                domain=domain,
                positions=positions,
                measures=self.mpm_model.particle_volume,
            )

            if self.grid_type == "fixed":
                wp.launch(
                    clamp_coordinates,
                    dim=pic.particle_coords.shape,
                    inputs=[pic.particle_coords],
                )

        return pic

    def _step_impl(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: _ImplicitMPMScratchpad,
    ):
        """Single implicit MPM step: bin, rasterize, assemble, solve, advect.

        Executes the full pipeline for one time step, including particle
        binning, collider rasterization, RHS assembly, strain/compliance matrix
        computation, warm-starting, coupled rheology/contact solve, strain
        updates, and particle advection.
        """

        cell_volume = self.mpm_model.voxel_size**3
        inv_cell_volume = 1.0 / cell_volume

        mpm_model = self.mpm_model

        self._require_collision_space_fields(state_out, scratch)
        self._require_velocity_space_fields(state_out, scratch, mpm_model.has_compliant_particles)

        # Rasterize colliders to discrete space
        self._rasterize_colliders(state_in, state_out, dt, scratch, inv_cell_volume)

        # Velocity right-hand side and inverse mass matrix
        self._compute_unconstrained_velocity(state_in, state_out, dt, pic, scratch, inv_cell_volume)

        # Build collider rigidity matrix
        rigidity_matrix = self._build_collider_rigidity_matrix(state_in, scratch, cell_volume)

        self._require_strain_space_fields(state_out, scratch)

        # Build elasticity compliance matrix and right-hand-side
        self._build_elasticity_system(state_in, dt, pic, scratch, inv_cell_volume)

        # Build strain matrix and offset, setup yield surface parameters
        self._build_plasticity_system(state_in, dt, pic, scratch, inv_cell_volume)

        # Solve implicit system
        self._solve_rheology(state_in, state_out, scratch, rigidity_matrix)

        # Update and advect particles
        self._update_particles(state_in, state_out, dt, pic, scratch)

    def _compute_unconstrained_velocity(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: _ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        """Compute the unconstrained (ballistic) velocity at grid nodes, as well as inverse mass matrix."""

        model = self.model
        mpm_model = self.mpm_model

        with self._timer("Unconstrained velocity"):
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
                dim=scratch.velocity_node_count,
                inputs=[
                    velocity_int,
                    node_particle_mass,
                    drag,
                ],
                outputs=[
                    scratch.inv_mass_matrix,
                    state_out.velocity_field.dof_values,
                ],
            )

    def _rasterize_colliders(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
        scratch: _ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        # Rasterize collider to grid
        collider_node_count = scratch.collider_node_count
        vel_node_count = scratch.velocity_node_count

        with self._timer("Rasterize collider"):
            # volume associated to each collider node
            fem.integrate(
                integrate_fraction,
                fields={"phi": scratch.collider_fraction_test},
                values={"inv_cell_volume": inv_cell_volume},
                assembly="nodal",
                output=scratch.collider_node_volume,
            )

            # rasterize sdf and properties to grid
            rasterize_collider(
                self.mpm_model.collider,
                state_in.body_q,
                state_in.body_qd,
                self.mpm_model.voxel_size,
                dt,
                scratch.collider_fraction_test.space_restriction,
                scratch.collider_node_volume,
                state_out.collider_position_field,
                scratch.collider_distance_field,
                scratch.collider_normal_field,
                scratch.collider_velocity,
                scratch.collider_friction,
                scratch.collider_adhesion,
                state_out.collider_ids,
            )

            # normal interpolation
            if self.collider_normal_from_sdf_gradient:
                interpolate_collider_normals(
                    self.mpm_model.voxel_size,
                    scratch.collider_fraction_test.space_restriction,
                    scratch.collider_distance_field,
                    scratch.collider_normal_field,
                )

            # Subgrid collisions
            if scratch.fraction_trial.space.name != scratch.collider_fraction_test.space.name:
                #  Map from collider nodes to velocity nodes
                sp.bsr_set_zero(
                    scratch.collider_matrix, rows_of_blocks=collider_node_count, cols_of_blocks=vel_node_count
                )
                fem.interpolate(
                    collision_weight_field,
                    dest=scratch.collider_matrix,
                    dest_space=scratch.collider_fraction_test.space,
                    at=scratch.collider_fraction_test.space_restriction,
                    reduction="first",
                    fields={"trial": scratch.fraction_trial, "normal": scratch.collider_normal_field},
                )

    def _build_collider_rigidity_matrix(
        self,
        state_in: newton.State,
        scratch: _ImplicitMPMScratchpad,
        cell_volume: float,
    ):
        has_compliant_colliders = self.mpm_model.min_collider_mass < _INFINITY

        if not has_compliant_colliders:
            scratch.collider_inv_mass_matrix.zero_()
            return None

        with self._timer("Collider compliance"):
            allot_collider_mass(
                cell_volume=cell_volume,
                node_volumes=scratch.collider_node_volume,
                collider=self.mpm_model.collider,
                body_mass=self.mpm_model.collider_body_mass,
                collider_ids=scratch.collider_ids,
                collider_total_volumes=scratch.collider_total_volumes,
                collider_inv_mass_matrix=scratch.collider_inv_mass_matrix,
            )

            rigidity_matrix = build_rigidity_matrix(
                cell_volume=cell_volume,
                node_volumes=scratch.collider_node_volume,
                node_positions=scratch.collider_position_field.dof_values,
                collider=self.mpm_model.collider,
                body_q=state_in.body_q,
                body_mass=self.mpm_model.collider_body_mass,
                body_inv_inertia=self.mpm_model.collider_body_inv_inertia,
                collider_ids=scratch.collider_ids,
                collider_total_volumes=scratch.collider_total_volumes,
            )

        return rigidity_matrix

    def _build_elasticity_system(
        self,
        state_in: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: _ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        """Build the elasticity and compliance system."""

        mpm_model = self.mpm_model

        has_compliant_particles = mpm_model.min_young_modulus < _INFINITY

        if not has_compliant_particles:
            scratch.int_symmetric_strain.zero_()
            return

        with self._timer("Elasticity"):
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

            wp.launch(
                average_elastic_parameters,
                dim=scratch.elastic_parameters_field.space_partition.node_count(),
                inputs=[
                    elastic_parameters_int,
                    node_particle_volume,
                    scratch.elastic_parameters_field.dof_values,
                ],
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
                output=scratch.int_symmetric_strain,
            )

            fem.integrate(
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
                output=scratch.compliance_matrix,
            )

    def _build_plasticity_system(
        self,
        state_in: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: _ImplicitMPMScratchpad,
        inv_cell_volume: float,
    ):
        mpm_model = self.mpm_model

        with self._timer("Compute strain-node volumes"):
            fem.integrate(
                integrate_fraction,
                quadrature=pic,
                fields={"phi": scratch.divergence_test},
                values={"inv_cell_volume": inv_cell_volume},
                output=scratch.strain_node_particle_volume,
            )

        with self._timer("Interpolated yield parameters"):
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
                dim=scratch.strain_node_particle_volume.shape[0],
                inputs=[
                    yield_parameters_int,
                    scratch.strain_node_particle_volume,
                    scratch.strain_yield_parameters_field.dof_values,
                ],
            )

        # Void fraction (unilateral incompressibility offset)
        if mpm_model.critical_fraction > 0.0:
            with self._timer("Unilateral offset"):
                fem.integrate(
                    integrate_fraction,
                    fields={"phi": scratch.divergence_test},
                    values={"inv_cell_volume": inv_cell_volume},
                    output=scratch.strain_node_volume,
                )

                if isinstance(scratch.collider_distance_field.space.basis, fem.PointBasisSpace):
                    fem.integrate(
                        integrate_collider_fraction_apic,
                        fields={
                            "phi": scratch.divergence_test,
                            "sdf": scratch.collider_distance_field,
                            "sdf_gradient": scratch.collider_normal_field,
                        },
                        values={
                            "inv_cell_volume": inv_cell_volume,
                        },
                        output=scratch.strain_node_collider_volume,
                    )
                else:
                    fem.integrate(
                        integrate_collider_fraction,
                        fields={
                            "phi": scratch.divergence_test,
                            "sdf": scratch.collider_distance_field,
                        },
                        values={
                            "inv_cell_volume": inv_cell_volume,
                        },
                        output=scratch.strain_node_collider_volume,
                    )

                wp.launch(
                    compute_unilateral_strain_offset,
                    dim=scratch.strain_node_count,
                    inputs=[
                        mpm_model.critical_fraction,
                        scratch.strain_node_particle_volume,
                        scratch.strain_node_collider_volume,
                        scratch.strain_node_volume,
                        scratch.unilateral_strain_offset,
                    ],
                )
        else:
            scratch.unilateral_strain_offset.zero_()

        # Strain jacobian
        with self._timer("Strain matrix"):
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

    def _solve_rheology(
        self,
        state_in: newton.State,
        state_out: newton.State,
        scratch: _ImplicitMPMScratchpad,
        rigidity_matrix: sp.BsrMatrix | None,
    ):
        strain_node_count = scratch.strain_node_count
        has_compliant_particles = self.mpm_model.has_compliant_particles

        with self._timer("Warmstart fields"):
            self._warmstart_fields(state_in.ws_impulse_field, state_in.ws_stress_field, scratch)

        with self._timer("Strain solve"):
            # Retain graph to avoid immediate CPU synch
            solve_graph = solve_rheology(
                self.max_iterations,
                self.tolerance,
                scratch.strain_matrix,
                scratch.transposed_strain_matrix,
                scratch.compliance_matrix,
                scratch.inv_mass_matrix,
                scratch.strain_node_particle_volume,
                scratch.strain_yield_parameters_field.dof_values,
                scratch.unilateral_strain_offset,
                scratch.int_symmetric_strain,
                scratch.plastic_strain_delta_field.dof_values,
                state_out.stress_field.dof_values,
                state_out.velocity_field.dof_values,
                scratch.collider_matrix,
                scratch.transposed_collider_matrix,
                scratch.collider_friction,
                scratch.collider_adhesion,
                scratch.collider_normal_field.dof_values,
                scratch.collider_velocity,
                scratch.collider_inv_mass_matrix,
                state_out.impulse_field.dof_values,
                color_offsets=scratch.color_offsets,
                color_indices=scratch.color_indices,
                color_nodes_per_element=scratch.color_nodes_per_element,
                rigidity_mat=rigidity_matrix,
                temporary_store=self.temporary_store,
                use_graph=self._use_cuda_graph,
            )

        if has_compliant_particles:
            wp.launch(
                average_elastic_strain_delta,
                dim=strain_node_count,
                inputs=[
                    scratch.int_symmetric_strain,
                    scratch.strain_node_particle_volume,
                    scratch.elastic_strain_delta_field.dof_values,
                ],
            )

        with self._timer("Save warmstart"):
            self._save_for_next_warmstart(state_out)

        return solve_graph

    def _update_particles(
        self,
        state_in: newton.State,
        state_out: newton.State,
        dt: float,
        pic: fem.PicQuadrature,
        scratch: _ImplicitMPMScratchpad,
    ):
        """Update particle quantities (strains, velocities, ...) from grid fields an advect them."""

        model = self.model
        mpm_model = self.mpm_model

        has_compliant_particles = mpm_model.min_young_modulus < _INFINITY
        has_hardening = mpm_model.max_hardening > 0.0

        if has_compliant_particles or has_hardening:
            with self._timer("Particle strain update"):
                # Update particle elastic strain from grid strain delta
                fem.interpolate(
                    update_particle_strains,
                    at=pic,
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
        with self._timer("Advection"):
            fem.interpolate(
                advect_particles,
                at=pic,
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

    @staticmethod
    def _require_velocity_space_fields(
        state_out: newton.State, scratch: _ImplicitMPMScratchpad, has_compliant_particles: bool
    ):
        """Ensure velocity-space fields exist and match current spaces."""

        scratch.require_velocity_space_fields(has_compliant_particles)
        scratch.require_collision_space_fields()

        # Necessary fields for grains rendering
        # Re-generated at each step, defined on space partition
        state_out.velocity_field = scratch.velocity_field

    @staticmethod
    def _require_collision_space_fields(state_out: newton.State, scratch: _ImplicitMPMScratchpad):
        """Ensure collision-space fields exist and match current spaces."""
        scratch.require_collision_space_fields()

        # Necessary fields for two-way coupling
        state_out.impulse_field = scratch.impulse_field
        state_out.collider_ids = scratch.collider_ids
        state_out.collider_position_field = scratch.collider_position_field

        # Impulse warmstarting, defined at space level
        collision_space = scratch.impulse_field.space
        if state_out.ws_impulse_field is None or state_out.ws_impulse_field.geometry != collision_space.geometry:
            state_out.ws_impulse_field = collision_space.make_field()

    @staticmethod
    def _require_strain_space_fields(state_out: newton.State, scratch: _ImplicitMPMScratchpad):
        """Ensure strain-space fields exist and match current spaces."""
        scratch.require_strain_space_fields()

        # Re-generated at each step, defined on space partition
        state_out.stress_field = scratch.stress_field

        # Stress warmstarting, define at space level
        sym_strain_space = scratch.sym_strain_test.space
        if state_out.ws_stress_field is None or state_out.ws_stress_field.geometry != sym_strain_space.geometry:
            state_out.ws_stress_field = sym_strain_space.make_field()

    @staticmethod
    def _warmstart_fields(prev_impulse_field: fem.Field, prev_stress_field: fem.Field, scratch: _ImplicitMPMScratchpad):
        """Interpolate previous grid fields into the current grid layout.

        Transfers impulse and stress fields from the previous grid to the new
        grid (handling nonconforming cases), and initializes the output state's
        grid fields to the current scratchpad fields.
        """
        domain = scratch.velocity_test.domain

        if isinstance(prev_impulse_field.space.basis, fem.PointBasisSpace):
            # point-based collisions, simply copy the previous impulses
            scratch.impulse_field.dof_values.assign(prev_impulse_field.dof_values)
        else:
            # Interpolate previous impulse
            prev_impulse_field = fem.NonconformingField(
                domain, prev_impulse_field, background=scratch.background_impulse_field
            )
            fem.interpolate(
                prev_impulse_field,
                dest=scratch.impulse_field,
                at=scratch.collider_fraction_test.space_restriction,
                reduction="first",
            )

        # Interpolate previous stress
        prev_stress_field = fem.NonconformingField(
            domain, prev_stress_field, background=scratch.background_stress_field
        )
        fem.interpolate(
            prev_stress_field,
            dest=scratch.stress_field,
            at=scratch.sym_strain_test.space_restriction,
            reduction="first",
        )

    @staticmethod
    def _save_for_next_warmstart(state_out: newton.State):
        if isinstance(state_out.ws_impulse_field.space.basis, fem.PointBasisSpace):
            # point-based collisions, simply copy the previous impulses
            state_out.ws_impulse_field.dof_values.assign(state_out.impulse_field.dof_values)
        else:
            state_out.ws_impulse_field.dof_values.zero_()
            wp.launch(
                scatter_field_dof_values,
                dim=state_out.impulse_field.space_partition.node_count(),
                inputs=[
                    state_out.impulse_field.space_partition.space_node_indices(),
                    state_out.impulse_field.dof_values,
                    state_out.ws_impulse_field.dof_values,
                ],
            )

        state_out.ws_stress_field.dof_values.zero_()
        wp.launch(
            scatter_field_dof_values,
            dim=state_out.stress_field.space_partition.node_count(),
            inputs=[
                state_out.stress_field.space_partition.space_node_indices(),
                state_out.stress_field.dof_values,
                state_out.ws_stress_field.dof_values,
            ],
        )

    def _max_colors(self):
        if not self.coloring:
            return 0
        return 27 if self.strain_basis == "Q1" else 8

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
        is_dg = space_partition.space_topology.node_count() == nodes_per_element * grid.cell_count()

        if is_dg:
            # nodes in each element solved sequentially
            stencil_size = 2
            if isinstance(grid, fem.Nanogrid):
                voxels = grid._cell_ijk
                res = wp.vec3i(0)
            else:
                voxels = None
                res = grid.res
        elif self.strain_basis == "Q1":
            nodes_per_element = 1
            stencil_size = 3
            if isinstance(grid, fem.Nanogrid):
                voxels = grid._node_ijk
                res = wp.vec3i(0)
            else:
                voxels = None
                res = grid.res + wp.vec3i(1)
        else:
            raise RuntimeError("Unsupported strain basis for coloring")

        strain_node_count = space_partition.node_count()
        colored_element_count = strain_node_count // nodes_per_element
        colors = fem.borrow_temporary(self.temporary_store, shape=colored_element_count * 2 + 1, dtype=int)
        color_indices = scratch.color_indices
        space_node_indices = space_partition.space_node_indices()

        wp.launch(
            node_color,
            dim=colored_element_count,
            inputs=[
                space_node_indices,
                nodes_per_element,
                stencil_size,
                voxels,
                res,
                colors,
                color_indices,
            ],
        )

        wp.utils.radix_sort_pairs(
            keys=colors,
            values=color_indices,
            count=colored_element_count,
        )

        unique_colors = colors[colored_element_count:]
        color_count = unique_colors[colored_element_count:]
        color_node_counts = color_indices[colored_element_count:]

        wp.utils.runlength_encode(
            colors,
            value_count=colored_element_count,
            run_values=unique_colors,
            run_lengths=color_node_counts,
            run_count=color_count,
        )
        wp.launch(
            compute_color_offsets,
            dim=1,
            inputs=[self._max_colors(), color_count, unique_colors, color_node_counts, scratch.color_offsets],
        )

        scratch.color_nodes_per_element = nodes_per_element

        colors.release()

    def _timer(self, name: str):
        return wp.ScopedTimer(
            name,
            active=self._enable_timers,
            use_nvtx=self._timers_use_nvtx,
            synchronize=not self._timers_use_nvtx,
        )
